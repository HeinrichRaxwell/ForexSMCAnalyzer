import math
import os
import sys
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.live_risk_governor import evaluate_daily_risk, get_mt5_daily_pip_summary

load_dotenv()


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value


def _live_spread_price(tick) -> float:
    """Return live ask-bid spread as a price value, or zero if unavailable."""
    try:
        ask = float(getattr(tick, "ask", 0.0))
        bid = float(getattr(tick, "bid", 0.0))
    except (TypeError, ValueError):
        return 0.0
    spread = ask - bid
    return float(spread) if np.isfinite(spread) and spread > 0 else 0.0


def _spread_adjusted_pending_entry(raw_entry: float, direction: int, tick) -> tuple[float, float]:
    """Adjust pending entry for the broker-side spread requested by the strategy owner."""
    spread = _live_spread_price(tick)
    if int(direction) == 1:
        return raw_entry + spread, spread
    return raw_entry - spread, -spread


def _is_stronger_stop_loss(direction: int, candidate_sl: float, current_sl: float) -> bool:
    if int(direction) == 1:
        return current_sl == 0 or candidate_sl > current_sl
    return current_sl == 0 or candidate_sl < current_sl


def _is_valid_stop_loss_for_market(direction: int, candidate_sl: float, tick) -> bool:
    if int(direction) == 1:
        return candidate_sl < float(tick.bid)
    return candidate_sl > float(tick.ask)


def _select_best_stop_loss(selected_sl, candidate_sl, current_sl: float, direction: int, tick):
    if candidate_sl is None:
        return selected_sl
    try:
        candidate_value = float(candidate_sl)
    except (TypeError, ValueError):
        return selected_sl
    if not np.isfinite(candidate_value):
        return selected_sl
    if not _is_stronger_stop_loss(direction, candidate_value, current_sl):
        return selected_sl
    if not _is_valid_stop_loss_for_market(direction, candidate_value, tick):
        return selected_sl
    if selected_sl is None or _is_stronger_stop_loss(direction, candidate_value, selected_sl):
        return candidate_value
    return selected_sl


def _atr_pips_from_df(df_tf, pip_multiplier: float, period: int = 14):
    """
    Average True Range over closed candles, expressed in PIPS.

    Used to size profit-lock distances by live volatility so that a normal
    New York pullback (large ATR) does not look like a reversal. Returns None
    when there is not enough closed-candle data to trust the estimate.
    """
    if df_tf is None or float(pip_multiplier) <= 0:
        return None
    closed = _closed_candle_frame(df_tf)
    if closed is None or len(closed) < 2:
        return None
    if not {"High", "Low", "Close"}.issubset(closed.columns):
        return None

    high = pd.to_numeric(closed["High"], errors="coerce")
    low = pd.to_numeric(closed["Low"], errors="coerce")
    prev_close = pd.to_numeric(closed["Close"], errors="coerce").shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1).dropna()
    if true_range.empty:
        return None

    atr_price = float(true_range.tail(int(period)).mean())
    if not np.isfinite(atr_price) or atr_price <= 0:
        return None
    return atr_price / float(pip_multiplier)


def compute_atr_lock_policy(
    atr_pips: float,
    *,
    base_step_pips: float,
    base_gap_pips: float,
    atr_arm_mult: float,
    atr_gap_mult: float,
) -> tuple[float, float]:
    """
    Turn the current ATR (in pips) into the two distances that drive the
    profit-lock ladder, so BEP/Protected-Profit doesn't trigger prematurely
    when volatility is high (e.g. the New York session).

    Returns ``(step_pips, gap_pips)`` which feed the existing ladder formula
    ``floor(profit / step) * step - gap``:

      * ``step_pips`` — spacing between rungs AND the first arm distance.
        Bigger step  => BEP arms later and rungs are further apart.
      * ``gap_pips``  — how far *behind* the reached rung the SL is parked
        (breathing room). Bigger gap => looser, less likely to be wicked out.

    Contract: never return less than the fixed baseline, so a quiet session
    falls straight back to your 50/50 ladder (50->BEP, 100->50, ...).

    ★ THIS IS YOUR POLICY — tune the two lines below to Gold/NY behaviour.
      The default scales linearly with ATR. You might instead cap it, use a
      percentile band, or only widen above a volatility threshold. You know
      how Gold breathes in NY better than the formula does.
    """
    atr = float(atr_pips) if atr_pips and np.isfinite(atr_pips) and atr_pips > 0 else 0.0
    step_pips = max(float(base_step_pips), float(atr_arm_mult) * atr)   # ★ tune
    gap_pips = max(float(base_gap_pips), float(atr_gap_mult) * atr)     # ★ tune
    return step_pips, gap_pips


def _profit_lock_stop_loss(
    entry_price: float,
    direction: int,
    tick,
    pip_multiplier: float,
    spread_buffer: float,
    symbol: str = "",
    position_profit: float = 0.0,
    atr_pips: float | None = None,
):
    if not _read_bool_env("MT5_PROFIT_LOCK_ENABLED", True):
        return None

    exit_price = float(tick.bid) if int(direction) == 1 else float(tick.ask)
    profit_pips = ((exit_price - float(entry_price)) * int(direction)) / float(pip_multiplier) if pip_multiplier > 0 else 0.0
    is_gold = isinstance(symbol, str) and ("XAU" in symbol.upper() or "GOLD" in symbol.upper())
    custom_candidate = None  # Initialize to avoid NameError if no branch assigns it
    if is_gold:
        # XAUUSD custom trailing stop ladder (pip = $0.10 for Gold):
        # Phase 1 (< 80 pips / < $8):    No SL adjustment, let trade breathe
        # Phase 2 (80–149 pips / $8–$14.9): SL to BEP + spread buffer (risk-free)
        # Phase 3 (>= 150 pips / >= $15):  Trail at floor(profit/50)*50 - 100 pips
        #   e.g.  150 pips profit → lock 50 pips ($5)
        #         200 pips profit → lock 100 pips ($10)
        #         250 pips profit → lock 150 pips ($15)
        # SL is only ever moved forward by caller via _select_best_stop_loss
        if profit_pips >= 150.0:
            reached_steps = math.floor(profit_pips / 50.0)
            locked_pips = reached_steps * 50.0 - 100.0
            custom_candidate = float(entry_price) + (locked_pips * float(pip_multiplier) * int(direction))
        elif profit_pips >= 80.0:
            custom_candidate = float(entry_price) + (spread_buffer * int(direction))
        return custom_candidate  # None means no change (<80 pips); handled by _select_best_stop_loss

    if _read_bool_env("MT5_CUSTOM_STAGED_PROFIT_LOCK", False):
        try:
            profit_val = float(position_profit) if position_profit is not None else 0.0
        except (TypeError, ValueError):
            profit_val = 0.0

        should_lock_50 = profit_pips >= 100.0

        if should_lock_50:
            custom_candidate = float(entry_price) + (50.0 * float(pip_multiplier) * int(direction))
        elif profit_pips >= 50.0:
            custom_candidate = float(entry_price) + (spread_buffer * int(direction))

    # Standard trailing logic fallback
    step_pips = _read_float_env("MT5_PROFIT_LOCK_STEP_PIPS", 100.0)
    gap_pips = _read_float_env("MT5_PROFIT_LOCK_GAP_PIPS", 100.0)

    # ATR-scaled widening: keep the same ladder shape but stretch the rungs and
    # the trailing gap by live volatility so NY noise doesn't arm BEP early.
    if _read_bool_env("MT5_ATR_SCALED_LOCK", False) and atr_pips and atr_pips > 0:
        step_pips, gap_pips = compute_atr_lock_policy(
            atr_pips,
            base_step_pips=step_pips,
            base_gap_pips=gap_pips,
            atr_arm_mult=_read_float_env("MT5_ATR_LOCK_ARM_MULT", 1.5),
            atr_gap_mult=_read_float_env("MT5_ATR_LOCK_GAP_MULT", 1.0),
        )

    if step_pips <= 0 or gap_pips < 0 or pip_multiplier <= 0:
        return custom_candidate

    if profit_pips < step_pips:
        return custom_candidate

    reached_steps = math.floor(profit_pips / step_pips)
    locked_pips = reached_steps * step_pips - gap_pips
    if locked_pips < 0:
        std_candidate = None
    elif locked_pips == 0:
        std_candidate = float(entry_price) + (spread_buffer * int(direction))
    else:
        std_candidate = float(entry_price) + (locked_pips * float(pip_multiplier) * int(direction))

    if custom_candidate is None:
        return std_candidate
    if std_candidate is None:
        return custom_candidate

    if int(direction) == 1:
        return max(custom_candidate, std_candidate)
    else:
        return min(custom_candidate, std_candidate)


def _configured_option_lot(opt_name: str) -> float:
    opt_name = str(opt_name or "")
    default_lot = 0.01
    lot_env = "MT5_LOT_SIZE_OPTION_A"
    if "GoldenPocket" in opt_name or "0.618" in opt_name or "Option B" in opt_name:
        default_lot = 0.02
        lot_env = "MT5_LOT_SIZE_OPTION_B"
    return _read_float_env(lot_env, default_lot)


def _normalize_broker_volume(volume: float, symbol: str = None) -> float:
    try:
        value = float(volume)
    except (TypeError, ValueError):
        value = 0.01

    info = mt5.symbol_info(symbol) if symbol else None
    min_volume = float(getattr(info, "volume_min", 0.01) or 0.01)
    step = float(getattr(info, "volume_step", 0.01) or 0.01)
    max_volume = float(getattr(info, "volume_max", 100.0) or 100.0)

    env_max = _read_float_env("MT5_DYNAMIC_LOT_MAX", max_volume)
    if env_max > 0:
        max_volume = min(max_volume, env_max)

    value = min(max(value, min_volume), max_volume)
    steps = math.floor((value + 1e-12) / step)
    normalized = max(min_volume, steps * step)
    decimals = max(2, int(abs(math.floor(math.log10(step)))) if step < 1 else 0)
    return round(normalized, decimals)


def resolve_lot_size(opt_name: str = "", symbol: str = None) -> float:
    """Resolve order lot size, optionally scaling by account balance ladder."""
    configured_lot = _configured_option_lot(opt_name)
    if not _read_bool_env("MT5_DYNAMIC_LOT_ENABLED", False):
        return _normalize_broker_volume(configured_lot, symbol)

    account = mt5.account_info()
    balance = float(getattr(account, "balance", 0.0) or 0.0) if account is not None else 0.0
    if balance <= 0:
        return _normalize_broker_volume(configured_lot, symbol)

    currency = getattr(account, "currency", None)
    if currency is None:
        if balance > 100000.0:
            currency = "IDR"
        else:
            currency = "USD"

    if str(currency).upper() == "IDR":
        base_balance = _read_float_env("MT5_DYNAMIC_LOT_BASE_BALANCE_IDR", 2_000_000.0)
        balance_step = _read_float_env("MT5_DYNAMIC_LOT_BALANCE_STEP_IDR", 1_000_000.0)
    else:
        base_balance = _read_float_env("MT5_DYNAMIC_LOT_BASE_BALANCE_USD", 100.0)
        balance_step = _read_float_env("MT5_DYNAMIC_LOT_BALANCE_STEP_USD", 50.0)

    base_lot = _read_float_env("MT5_DYNAMIC_LOT_BASE_LOT", 0.01)
    step_lot = _read_float_env("MT5_DYNAMIC_LOT_STEP_LOT", 0.01)
    if balance_step <= 0:
        return _normalize_broker_volume(base_lot, symbol)

    base_bucket = math.floor(base_balance / balance_step)
    current_bucket = math.floor(balance / balance_step)
    increments = max(0, current_bucket - base_bucket)
    return _normalize_broker_volume(base_lot + increments * step_lot, symbol)


def _cap_tp_distance(tp_price: float, entry_price: float, direction: int, symbol: str) -> float:
    """Clamp the broker-side TP so it never sits further than MT5_TP_MAX_PIPS from entry."""
    max_pips = _read_float_env("MT5_TP_MAX_PIPS", 150.0)
    if max_pips <= 0:
        return tp_price

    from src.smc_detector import get_pip_multiplier
    pip_multiplier = get_pip_multiplier(symbol)
    if pip_multiplier <= 0:
        return tp_price

    max_distance = max_pips * pip_multiplier
    if int(direction) == 1:
        return min(float(tp_price), float(entry_price) + max_distance)
    return max(float(tp_price), float(entry_price) - max_distance)


def _server_take_profit_price(setup: dict, entry_price: float, direction: int, symbol: str):
    mode = os.getenv("MT5_SERVER_TP_MODE", "fibo0").strip().lower()
    if mode in {"fibo0", "tp1", "primary"}:
        tp_price = setup["tp_price"]
    elif mode in {"furthest", "tp3"}:
        tp_price = setup.get("tp3_price", setup.get("tp2_price", setup["tp_price"]))
    else:
        tp_price = setup["tp_price"]
    return _cap_tp_distance(tp_price, entry_price, direction, symbol)


def _position_exit_price(direction: int, tick) -> float:
    return float(tick.bid) if int(direction) == 1 else float(tick.ask)


def _runner_near_primary_tp(direction: int, exit_price: float, tp_price: float, pip_multiplier: float) -> bool:
    arm_distance = _read_float_env("MT5_RUNNER_ARM_DISTANCE_PIPS", 10.0) * float(pip_multiplier)
    if arm_distance <= 0:
        return False
    if int(direction) == 1:
        return float(tp_price) - arm_distance <= float(exit_price) < float(tp_price)
    return float(tp_price) < float(exit_price) <= float(tp_price) + arm_distance


def _runner_continuation_supported(feat: dict, direction: int, tick, pip_multiplier: float, h1_trend=None, h4_trend=None) -> bool:
    if not _read_bool_env("MT5_RUNNER_ENABLED", True):
        return False
    max_spread_pips = _read_float_env("MT5_RUNNER_MAX_SPREAD_PIPS", 5.0)
    if pip_multiplier > 0 and (_live_spread_price(tick) / float(pip_multiplier)) > max_spread_pips:
        return False

    aligned_value = (feat or {}).get("floop_trend_aligned")
    floop_aligned = str(aligned_value).strip().lower() in {"1", "true", "yes"}
    htf_aligned = h1_trend == int(direction) or h4_trend == int(direction)
    if _read_bool_env("MT5_RUNNER_REQUIRE_CONFLUENCE", True):
        return floop_aligned or htf_aligned
    return True


def _runner_protected_stop(entry_price: float, current_sl: float, direction: int, tick, pip_multiplier: float):
    protect_pips = _read_float_env("MT5_RUNNER_PROTECT_PIPS", 2.0)
    candidate = float(entry_price) + protect_pips * float(pip_multiplier) * int(direction)
    selected = _select_best_stop_loss(None, candidate, current_sl, direction, tick)
    return selected if selected is not None else current_sl


def get_active_broker_symbol(symbol: str) -> str:
    """
    Find the matching symbol name supported by the active broker terminal.
    """
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    for sym in symbols_to_try:
        info = mt5.symbol_info(sym)
        if info is not None:
            return sym
    return symbol # Fallback


def _matches_magic(record, magic: int) -> bool:
    try:
        return int(getattr(record, "magic", None)) == int(magic)
    except (TypeError, ValueError):
        return False


def _filter_by_magic(records, magic: int) -> list:
    return [record for record in records or [] if _matches_magic(record, magic)]


def _positions_for_symbol_magic(symbol: str, magic: int) -> list:
    positions = mt5.positions_get(symbol=symbol)
    return _filter_by_magic(positions, magic)


def _orders_for_symbol_magic(symbol: str, magic: int) -> list:
    orders = mt5.orders_get(symbol=symbol)
    return _filter_by_magic(orders, magic)


def get_active_trade_count(symbol: str, magic: int) -> int:
    """
    Get the total count of active positions and pending orders for this magic number.
    """
    positions = _positions_for_symbol_magic(symbol, magic)
    pos_count = len(positions)
    
    orders = _orders_for_symbol_magic(symbol, magic)
    ord_count = len(orders)
    
    return pos_count + ord_count


def _max_concurrent_trade_message(symbol: str, magic: int) -> str | None:
    max_trades = _read_int_env("MT5_MAX_CONCURRENT_TRADES", 0)
    if max_trades <= 0:
        return None
    active_count = get_active_trade_count(symbol, magic)
    if active_count >= max_trades:
        return f"max concurrent trades reached ({active_count}/{max_trades}) for {symbol}"
    return None


def _daily_risk_governor_message(symbol: str, magic: int) -> str | None:
    if not _read_bool_env("MT5_DAILY_GOVERNOR_ENABLED", False):
        return None

    try:
        from src.smc_detector import get_pip_multiplier

        pip_multiplier = get_pip_multiplier(symbol)
        summary = get_mt5_daily_pip_summary(mt5, symbol, magic, pip_multiplier)
    except Exception as exc:
        return f"daily governor unavailable: {exc}"

    decision = evaluate_daily_risk(
        realized_pips=summary.realized_pips,
        consecutive_losses=summary.consecutive_losses,
        min_target_pips=_read_float_env("MT5_DAILY_MIN_TARGET_PIPS", 100.0),
        runner_target_pips=_read_float_env("MT5_DAILY_RUNNER_TARGET_PIPS", 300.0),
        max_loss_pips=_read_float_env("MT5_DAILY_MAX_LOSS_PIPS", 200.0),
        max_consecutive_losses=_read_int_env("MT5_DAILY_MAX_CONSECUTIVE_LOSSES", 3),
    )
    if decision.allowed:
        return None
    return (
        f"daily risk governor blocked new order: {decision.reason} "
        f"(realized={decision.realized_pips:.1f} pips, losses={decision.consecutive_losses})"
    )


def _parse_order_comment(comment: str) -> tuple[str | None, str | None]:
    if not comment or not comment.startswith("SMC "):
        return None, None
    parts = comment.split(" ")
    if len(parts) < 3:
        return None, None
    timeframe = parts[1]
    strategy_part = parts[2]
    for strat in ["FVG", "OB", "IC", "Swapzone", "BPR", "Breaker", "SND", "Pivot"]:
        if strat in strategy_part:
            return timeframe, strat
    return timeframe, None


def _max_pending_orders_message(symbol: str, magic: int, setup: dict, broker_entry_price: float) -> str | None:
    # 1. Pending orders limit per direction (max pending buy = max_pending, max pending sell = max_pending)
    max_pending = _read_int_env("MT5_MAX_PENDING_ORDERS", 0)
    orders = _orders_for_symbol_magic(symbol, magic)
    if max_pending > 0:
        direction = setup.get("direction", 1)
        if direction == 1:
            buy_orders = [
                o for o in orders 
                if getattr(o, "type", None) is None 
                or getattr(o, "type", None) in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_STOP_LIMIT)
            ]
            if len(buy_orders) >= max_pending:
                return f"max pending orders reached for buy direction ({len(buy_orders)}/{max_pending})"
        else:
            sell_orders = [
                o for o in orders 
                if getattr(o, "type", None) is None 
                or getattr(o, "type", None) in (mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP, mt5.ORDER_TYPE_SELL_STOP_LIMIT)
            ]
            if len(sell_orders) >= max_pending:
                return f"max pending orders reached for sell direction ({len(sell_orders)}/{max_pending})"

    # 2. Proximity check (prevent too many pending orders placed close to each other from different setups)
    from src.smc_detector import get_pip_multiplier
    pip_multiplier = get_pip_multiplier(symbol)
    proximity_pips = _read_float_env("MT5_PENDING_PROXIMITY_PIPS", 15.0)
    proximity_limit = proximity_pips * pip_multiplier if pip_multiplier > 0 else proximity_pips

    tf = setup.get("timeframe", "M15")
    opt_name = setup.get("option_name", "")
    
    candidate_strategy = "FVG"
    for strat in ["OB", "BPR", "IC", "Swap", "Breaker", "SND", "Pivot"]:
        if strat in opt_name:
            candidate_strategy = "Swapzone" if strat == "Swap" else strat
            break

    # Same TF + same strategy setups from consecutive scan cycles can still stack up.
    # Apply a second, configurable proximity limit specifically for those pairs.
    # Exception: Option A and Option B of the SAME dual-fib structure are intentional
    # counterparts and must never block each other (they are placed as a pair).
    same_tf_proximity_pips = _read_float_env("MT5_SAME_TF_PROXIMITY_PIPS", 30.0)
    same_tf_limit = same_tf_proximity_pips * pip_multiplier if pip_multiplier > 0 else same_tf_proximity_pips

    def _is_option_b(text: str) -> bool:
        return any(k in str(text) for k in ("Option B", "0.618", "GoldenPocket"))

    new_is_b = _is_option_b(opt_name)

    for o in orders:
        o_tf, o_strat = _parse_order_comment(getattr(o, "comment", ""))
        is_same_setup = (o_tf == tf and o_strat == candidate_strategy)
        if is_same_setup and (_is_option_b(getattr(o, "comment", "")) != new_is_b):
            continue  # A+B counterparts of the same dual-fib structure — never block each other
        effective_limit = same_tf_limit if is_same_setup else proximity_limit
        effective_pips = same_tf_proximity_pips if is_same_setup else proximity_pips
        dist = abs(o.price_open - broker_entry_price)
        if dist < effective_limit:
            return (
                f"proximity block: entry price {broker_entry_price:.3f} is too close to existing "
                f"pending order #{o.ticket} ({o.comment}) at {o.price_open:.3f} "
                f"(dist: {dist/pip_multiplier if pip_multiplier > 0 else dist:.1f} pips < {effective_pips} pips limit)"
            )

    positions = _positions_for_symbol_magic(symbol, magic)
    for p in positions:
        p_tf, p_strat = _parse_order_comment(getattr(p, "comment", ""))
        is_same_setup = (p_tf == tf and p_strat == candidate_strategy)
        if is_same_setup and (_is_option_b(getattr(p, "comment", "")) != new_is_b):
            continue  # A+B counterparts of the same dual-fib structure — never block each other
        effective_limit = same_tf_limit if is_same_setup else proximity_limit
        effective_pips = same_tf_proximity_pips if is_same_setup else proximity_pips
        dist = abs(p.price_open - broker_entry_price)
        if dist < effective_limit:
            return (
                f"proximity block: entry price {broker_entry_price:.3f} is too close to existing "
                f"position #{p.ticket} ({p.comment}) at {p.price_open:.3f} "
                f"(dist: {dist/pip_multiplier if pip_multiplier > 0 else dist:.1f} pips < {effective_pips} pips limit)"
            )

    # 3. Per timeframe + direction + strategy restriction (optional, defaults to True)
    allow_mixed = _read_bool_env("MT5_ALLOW_MIXED_STRATEGIES_PER_TF", True)
    if not allow_mixed:
        direction = setup.get("direction", 1)
        for o in orders:
            o_type = getattr(o, "type", None)
            if o_type == mt5.ORDER_TYPE_BUY_LIMIT:
                o_direction = 1
            elif o_type == mt5.ORDER_TYPE_SELL_LIMIT:
                o_direction = -1
            else:
                continue

            if o_direction != direction:
                continue

            o_tf, o_strat = _parse_order_comment(getattr(o, "comment", ""))
            if o_tf == tf and o_strat is not None:
                if o_strat != candidate_strategy:
                    return f"blocked mixed strategy on {tf}: already have pending {o_strat} limit order on {tf}"

    return None


def validate_market_indicators(symbol: str, tf_str: str, direction: int) -> tuple:
    """
    Validate current market indicators (RSI, Stoch RSI, EMA, Volume) before placing a trade.
    Returns:
        (bool, str): (is_valid, reason)
    """
    from src.data_loader import fetch_historical_data
    
    # 1. Map timeframe string to MT5 timeframe
    mapping = {
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    tf = mapping.get(tf_str, mt5.TIMEFRAME_M15)
    
    # 2. Fetch history (recent 100 candles)
    df = fetch_historical_data(symbol, tf, 100)
    if df is None or df.empty or len(df) < 30:
        return True, "Insufficient historical data to validate indicators"
        
    close = df['Close']
    
    # 3. Calculate RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 0.00001)
    rsi = 100 - (100 / (1 + rs))
    curr_rsi = rsi.iloc[-1]
    
    # 4. Calculate Stochastic RSI (14)
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, 0.00001)
    curr_stoch_rsi = stoch_rsi.iloc[-1]
    
    # 5. Volume check
    curr_vol = df['Volume'].iloc[-1]
    avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
    
    # Validation checks
    # A. Volume spike check (market reaction too violent)
    if curr_vol > 3.5 * avg_vol:
        return False, f"Abnormal volume spike detected ({curr_vol:.0f} > 3.5x avg {avg_vol:.0f})"
        
    # B. RSI & Stoch RSI Extreme checks (Disabled for pending limit orders)
    # if direction == 1:  # Buy
    #     if curr_rsi < 20.0:
    #         return False, f"RSI is extremely oversold ({curr_rsi:.1f} < 20), price dumping too fast"
    #     if curr_rsi > 70.0:
    #         return False, f"RSI is overbought ({curr_rsi:.1f} > 70)"
    #     if curr_stoch_rsi > 0.85:
    #         return False, f"Stoch RSI is overbought ({curr_stoch_rsi:.2f} > 0.85)"
    # else:  # Sell
    #     if curr_rsi > 80.0:
    #         return False, f"RSI is extremely overbought ({curr_rsi:.1f} > 80), price pumping too fast"
    #     if curr_rsi < 30.0:
    #         return False, f"RSI is oversold ({curr_rsi:.1f} < 30)"
    #     if curr_stoch_rsi < 0.15:
    #         return False, f"Stoch RSI is oversold ({curr_stoch_rsi:.2f} < 0.15)"
    pass
            
    return True, f"Indicators valid: RSI={curr_rsi:.1f}, StochRSI={curr_stoch_rsi:.2f}, Volume={curr_vol:.0f}"

def try_evict_lowest_confidence_pending_order(symbol: str, magic: int, new_setup: dict, limit_type: str, broker_entry_price: float = None) -> bool:
    """
    Finds the lowest confidence pending order for the given symbol/magic.
    If the new setup has higher confidence, cancels the pending order and registers it as shadow/paper.
    Returns True if an order was successfully evicted, False otherwise.
    """
    import MetaTrader5 as mt5
    from datetime import datetime
    
    # 1. Get new setup's probability/confidence
    new_prob = new_setup.get("probability")
    if new_prob is None:
        new_prob = new_setup.get("max_prob")
    if new_prob is None:
        new_prob = 0.0
        
    # 2. Get active pending orders in MT5
    orders = _orders_for_symbol_magic(symbol, magic)
    if not orders:
        return False
        
    # Filter by direction if checking direction limit
    direction = new_setup.get("direction", 1)
    if limit_type == "direction":
        if direction == 1:
            orders = [
                o for o in orders 
                if getattr(o, "type", None) is None 
                or getattr(o, "type", None) in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_STOP_LIMIT)
            ]
        else:
            orders = [
                o for o in orders 
                if getattr(o, "type", None) is None 
                or getattr(o, "type", None) in (mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP, mt5.ORDER_TYPE_SELL_STOP_LIMIT)
            ]
            
    if not orders:
        return False
        
    # 3. Load sent signals registry to find probabilities of existing orders
    sent_signals = load_sent_signals()
    
    candidates = []
    for o in orders:
        sig_data, suffix, feature_key, features = _find_ticket_signal(sent_signals, o.ticket)
        prob = 0.5  # fallback
        sig_key = None
        if sig_data:
            # Find the key in sent_signals
            for k, v in sent_signals.items():
                if v == sig_data:
                    sig_key = k
                    break
            # Extract probability
            if suffix == "_a":
                prob = sig_data.get("probability_a", 0.5)
            elif suffix == "_b":
                prob = sig_data.get("probability_b", 0.5)
            else:
                prob = sig_data.get("probability", 0.5)
        candidates.append((o.ticket, prob, sig_key, sig_data, suffix, o))
        
    if not candidates:
        return False
        
    # Sort candidates by probability ascending
    candidates.sort(key=lambda x: x[1])
    lowest_ticket, lowest_prob, lowest_key, lowest_data, lowest_suffix, lowest_order = candidates[0]
    
    # Only evict if new setup has strictly higher confidence
    if new_prob > lowest_prob:
        # Cancel the lowest confidence order in MT5
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": lowest_ticket
        }
        res = mt5.order_send(request)
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[Execution Engine] Evicting pending order #{lowest_ticket} (prob: {lowest_prob:.1%}) to make room for higher-confidence setup (prob: {new_prob:.1%})")
            
            # Update sent_signals to mark it as evicted
            if lowest_data:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                lowest_data[f"manager_exit_trigger{lowest_suffix}"] = "evicted_by_higher_confidence"
                lowest_data[f"manager_exit_detail{lowest_suffix}"] = f"Evicted by higher confidence setup: {new_prob:.1%} > {lowest_prob:.1%}"
                lowest_data[f"evicted_at{lowest_suffix}"] = now_str
                save_sent_signals(sent_signals)
                
            # Register in shadow signals JSON so it is tracked as paper trade
            try:
                from src.shadow_tracker import build_shadow_signal_records, upsert_shadow_signals
                tf = "M30"
                strategy = "FVG"
                if lowest_data:
                    tf = lowest_data.get("timeframe", tf)
                    strategy = lowest_data.get("strategy", strategy)
                else:
                    comment = getattr(lowest_order, "comment", "")
                    if "M15" in comment: tf = "M15"
                    elif "H1" in comment: tf = "H1"
                    elif "H4" in comment: tf = "H4"
                    
                shadow_key = lowest_key or f"EVICTED_{lowest_ticket}"
                
                opt = {
                    "entry_price": lowest_order.price_open,
                    "sl_price": lowest_order.sl,
                    "tp_price": lowest_order.tp,
                    "timeframe": tf,
                    "direction": 1 if lowest_order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP) else -1,
                    "option_name": getattr(lowest_order, "comment", f"Option {lowest_suffix}")
                }
                
                records = build_shadow_signal_records(
                    signal_id=shadow_key,
                    symbol=symbol,
                    timeframe=tf,
                    strategy=strategy,
                    direction_name="BULL" if opt["direction"] == 1 else "BEAR",
                    accept_threshold=lowest_prob,
                    opt=opt,
                    probability=lowest_prob,
                    filtered_reason=f"evicted_for_higher_confidence_setup_({new_prob:.1%}_vs_{lowest_prob:.1%})",
                )
                upsert_shadow_signals(records)
                print(f"[Execution Engine] Evicted order #{lowest_ticket} registered in shadow tracker for monitoring.")
            except Exception as e:
                print(f"[Execution Engine] Failed to register evicted order in shadow tracker: {e}")
                
            # Send Telegram Alert
            try:
                from src.telegram_bot import send_telegram_alert
                msg = (
                    f"⚠️ <b>[SMC Eviction Engine] Order Evicted!</b> ⚠️\n\n"
                    f"Pending order #{lowest_ticket} ({lowest_order.comment or 'SMC order'}) telah dibatalkan untuk mengosongkan slot karena limit penuh.\n"
                    f"• <b>Evicted Setup Confidence:</b> <code>{lowest_prob:.1%}</code>\n"
                    f"• <b>New Setup Confidence:</b> <code>{new_prob:.1%}</code>\n"
                    f"• <b>Tindakan:</b> Order #{lowest_ticket} dipindahkan ke Paper/Shadow Tracker untuk dipelajari hasilnya secara offline."
                )
                send_telegram_alert(msg)
            except Exception as e:
                print(f"[Execution Engine] Telegram notification failed: {e}")
                
            return True
        else:
            err_msg = res.comment if res is not None else "None result"
            print(f"[Execution Engine] Failed to cancel evicted pending order #{lowest_ticket}: {err_msg}")
            
    return False


def execute_trade_for_setup(setup: dict, base_symbol: str = "XAUUSD") -> tuple:
    """
    Sends a pending limit order to MT5 for the given setup.
    
    Args:
        setup (dict): The setup dictionary with keys: timeframe, direction, entry_price, sl_price, tp_price, option_name.
        base_symbol (str): The default symbol (e.g., 'XAUUSD').
        
    Returns:
        (int, str): (ticket_id, success_message) or (None, error_message)
    """
    # Check if execution is enabled in .env
    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").strip().lower() == "true"
    if not execute_enabled:
        return None, "Auto-execution disabled (MT5_EXECUTE_TRADES=False in .env)"
        
    # Check allowed timeframes
    allowed_tfs_str = os.getenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    allowed_tfs = [tf.strip() for tf in allowed_tfs_str.split(",")]
    tf = setup.get("timeframe", "M15")
    if tf not in allowed_tfs:
        return None, f"Timeframe {tf} disabled in .env"
        
    # Get active broker symbol
    symbol = get_active_broker_symbol(base_symbol)
    magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
    
    concurrent_message = _max_concurrent_trade_message(symbol, magic)
    if concurrent_message:
        # Try to evict a lower-confidence pending order
        if try_evict_lowest_confidence_pending_order(symbol, magic, setup, "concurrent"):
            concurrent_message = _max_concurrent_trade_message(symbol, magic)
            
        if concurrent_message:
            return None, concurrent_message
            
    governor_message = _daily_risk_governor_message(symbol, magic)
    if governor_message:
        return None, governor_message
    
    opt_name = setup.get("option_name", "")
    lot = resolve_lot_size(opt_name, symbol)
            
    # 2. Determine order type
    # 1 for Bullish (Buy Limit), -1 for Bearish (Sell Limit)
    direction = setup.get("direction", 1)
    if direction == 1:
        order_type = mt5.ORDER_TYPE_BUY_LIMIT
    else:
        order_type = mt5.ORDER_TYPE_SELL_LIMIT
        
    # Ensure symbol is selected in Market Watch
    mt5.symbol_select(symbol, True)

    raw_entry_price = float(setup["entry_price"])
    tick = mt5.symbol_info_tick(symbol)
    broker_entry_price, spread_adjustment = _spread_adjusted_pending_entry(raw_entry_price, direction, tick)
    setup["raw_entry_price"] = raw_entry_price
    setup["broker_entry_price"] = broker_entry_price
    setup["entry_spread_adjustment"] = spread_adjustment

    pending_message = _max_pending_orders_message(symbol, magic, setup, broker_entry_price)
    if pending_message:
        # Try to evict a lower-confidence pending order in the same direction
        if try_evict_lowest_confidence_pending_order(symbol, magic, setup, "direction", broker_entry_price):
            pending_message = _max_pending_orders_message(symbol, magic, setup, broker_entry_price)
            
        if pending_message:
            return None, pending_message

    # Check duplicate broker entry price (within 0.15 USD/points for Gold)
    orders = _orders_for_symbol_magic(symbol, magic)
    for o in orders:
        if abs(o.price_open - broker_entry_price) < 0.15:
            return None, f"Duplicate pending order already exists at price {o.price_open:.3f}"

    positions = _positions_for_symbol_magic(symbol, magic)
    for p in positions:
        if abs(p.price_open - broker_entry_price) < 0.15:
            return None, f"Duplicate position already exists at price {p.price_open:.3f}"
    
    # Check entry price distance to market price (prevent spamming far orders)
    if tick is not None:
        current_price = tick.ask if direction == 1 else tick.bid
        price_diff = abs(broker_entry_price - current_price)
        from src.smc_detector import get_pip_multiplier
        pip_multiplier = get_pip_multiplier(symbol)
        price_diff_pips = price_diff / pip_multiplier if pip_multiplier > 0 else price_diff
        
        max_dist_pips = float(os.getenv("MT5_MAX_PENDING_DISTANCE_PIPS", "200.0"))
        if tf == 'H4':
            max_dist_pips = max(max_dist_pips, 1000.0)  # Allow up to 1000 pips for H4 setups
        elif tf == 'H1':
            max_dist_pips = max(max_dist_pips, 600.0)   # Allow up to 600 pips for H1 setups
        elif tf == 'M30':
            max_dist_pips = max(max_dist_pips, 300.0)   # Allow up to 300 pips for M30 setups
        elif tf == 'D1':
            max_dist_pips = max(max_dist_pips, 2000.0)  # Allow up to 2000 pips for D1 setups
            
        if price_diff_pips > max_dist_pips:
            return None, f"price is too far from market ({price_diff_pips:.1f} pips > {max_dist_pips} pips limit)"
            
        # Price is close! Now check current market indicators before placing order
        is_valid_mkt, mkt_reason = validate_market_indicators(symbol, tf, direction)
        if not is_valid_mkt:
            return None, f"Market indicators check failed: {mkt_reason}"
            
    # Get symbol info for digit formatting
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None, f"Failed to get symbol info for {symbol}"
        
    digits = symbol_info.digits
    
    entry_price = round(float(broker_entry_price), digits)
    sl_price = round(float(setup["sl_price"]), digits)
    
    # Keep the broker-side hard target at the primary Fibo-0/TP1 level.
    # The runner manager can remove this TP before it is hit when momentum supports continuation.
    safety_tp_price = _server_take_profit_price(setup, entry_price, direction, symbol)
    tp_price = round(float(safety_tp_price), digits)

    comment = f"SMC {setup.get('timeframe', 'M15')} {opt_name[:15]}"
    
    # Try different filling types (RETURN, IOC, FOK)
    filling_types = [
        mt5.ORDER_FILLING_RETURN,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK
    ]
    
    last_error = ""
    for fill in filling_types:
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill,
        }
        
        # Send order
        res = mt5.order_send(request)
        if res is None:
            last_error = "mt5.order_send returned None (check MT5 connection)"
            continue
            
        if res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
            print(f"[Execution Engine] Order successfully placed: Ticket {res.order} using filling {fill}")
            return res.order, f"PENDING ORDER PLACED: Ticket #{res.order} ({lot} lot @ {entry_price:.3f})"
        else:
            last_error = f"retcode={res.retcode}, comment='{res.comment}'"
            print(f"[Execution Engine] Try filling={fill} failed: {last_error}")
            
    return None, f"Failed to place order: {last_error}"


def _check_market_proximity_message(symbol: str, magic: int, setup: dict, current_price: float) -> str | None:
    from src.smc_detector import get_pip_multiplier
    pip_multiplier = get_pip_multiplier(symbol)
    proximity_pips = _read_float_env("MT5_PENDING_PROXIMITY_PIPS", 15.0)
    proximity_limit = proximity_pips * pip_multiplier if pip_multiplier > 0 else proximity_pips

    tf = setup.get("timeframe", "M15")
    opt_name = setup.get("option_name", "")
    direction = setup.get("direction", 1)
    
    candidate_strategy = "FVG"
    for strat in ["OB", "BPR", "IC", "Swap", "Breaker", "SND", "Pivot"]:
        if strat in opt_name:
            candidate_strategy = "Swapzone" if strat == "Swap" else strat
            break

    same_tf_proximity_pips = _read_float_env("MT5_SAME_TF_PROXIMITY_PIPS", 30.0)
    same_tf_limit = same_tf_proximity_pips * pip_multiplier if pip_multiplier > 0 else same_tf_proximity_pips

    def _is_option_b(text: str) -> bool:
        return any(k in str(text) for k in ("Option B", "0.618", "GoldenPocket"))

    new_is_b = _is_option_b(opt_name)

    positions = _positions_for_symbol_magic(symbol, magic)
    for p in positions:
        p_type = getattr(p, "type", None)
        p_direction = 1 if p_type == mt5.ORDER_TYPE_BUY else (-1 if p_type == mt5.ORDER_TYPE_SELL else 0)
        if p_direction != direction:
            continue
            
        p_tf, p_strat = _parse_order_comment(getattr(p, "comment", ""))
        is_same_setup = (p_tf == tf and p_strat == candidate_strategy)
        if is_same_setup and (_is_option_b(getattr(p, "comment", "")) != new_is_b):
            continue  # A+B counterparts of the same dual-fib structure — never block each other
            
        effective_limit = same_tf_limit if is_same_setup else proximity_limit
        effective_pips = same_tf_proximity_pips if is_same_setup else proximity_pips
        dist = abs(p.price_open - current_price)
        if dist < effective_limit:
            return (
                f"market proximity block: current price {current_price:.3f} is too close to existing "
                f"position #{p.ticket} ({p.comment}) at {p.price_open:.3f} "
                f"(dist: {dist/pip_multiplier if pip_multiplier > 0 else dist:.1f} pips < {effective_pips} pips limit)"
            )
    return None


def execute_market_order_for_setup(setup: dict, base_symbol: str = "XAUUSD") -> tuple:
    """
    Sends a market buy/sell order (Instant Execution) to MT5 for the given setup.
    
    Args:
        setup (dict): The setup dictionary with keys: timeframe, direction, entry_price, sl_price, tp_price, option_name.
        base_symbol (str): The default symbol (e.g., 'XAUUSD').
        
    Returns:
        (int, str): (ticket_id, success_message) or (None, error_message)
    """
    # Check if execution is enabled in .env
    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").strip().lower() == "true"
    if not execute_enabled:
        return None, "Auto-execution disabled (MT5_EXECUTE_TRADES=False in .env)"
        
    # Check allowed timeframes
    allowed_tfs_str = os.getenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    allowed_tfs = [tf.strip() for tf in allowed_tfs_str.split(",")]
    tf = setup.get("timeframe", "M15")
    if tf not in allowed_tfs:
        return None, f"Timeframe {tf} disabled in .env"
        
    # Get active broker symbol
    symbol = get_active_broker_symbol(base_symbol)
    magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
    concurrent_message = _max_concurrent_trade_message(symbol, magic)
    if concurrent_message:
        # Try to evict a lower-confidence pending order
        if try_evict_lowest_confidence_pending_order(symbol, magic, setup, "concurrent"):
            concurrent_message = _max_concurrent_trade_message(symbol, magic)
            
        if concurrent_message:
            return None, concurrent_message
    governor_message = _daily_risk_governor_message(symbol, magic)
    if governor_message:
        return None, governor_message
    
    opt_name = setup.get("option_name", "")
    lot = resolve_lot_size(opt_name, symbol)
            
    # 2. Determine order type
    # 1 for Bullish (Buy), -1 for Bearish (Sell)
    direction = setup.get("direction", 1)
    
    # Ensure symbol is selected in Market Watch
    mt5.symbol_select(symbol, True)
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, "Failed to get current price tick from MT5"
        
    if direction == 1:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        
    # Get symbol info for digit formatting
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None, f"Failed to get symbol info for {symbol}"
        
    digits = symbol_info.digits
    
    price_formatted = round(float(price), digits)
    
    # Check market proximity to prevent over-exposure / clustering
    proximity_msg = _check_market_proximity_message(symbol, magic, setup, price_formatted)
    if proximity_msg:
        return None, proximity_msg
        
    sl_price = round(float(setup["sl_price"]), digits)
    safety_tp_price = _server_take_profit_price(setup, price_formatted, direction, symbol)
    tp_price = round(float(safety_tp_price), digits)
    
    comment = f"SMC Mkt {setup.get('timeframe', 'M15')} {opt_name[:12]}"
    
    # Try different filling types (RETURN, IOC, FOK)
    filling_types = [
        mt5.ORDER_FILLING_RETURN,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK
    ]
    
    last_error = ""
    for fill in filling_types:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price_formatted,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_filling": fill,
        }
        
        # Send order
        res = mt5.order_send(request)
        if res is None:
            last_error = "mt5.order_send returned None"
            continue
            
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[Execution Engine] Market Order successfully placed: Position Ticket {res.deal} (Order Ticket {res.order}) using filling {fill}")
            ticket = res.order if res.order else res.deal
            return ticket, f"MARKET ORDER PLACED: Ticket #{ticket} ({lot} lot @ {price_formatted:.3f})"
        else:
            last_error = f"retcode={res.retcode}, comment='{res.comment}'"
            print(f"[Execution Engine] Try market filling={fill} failed: {last_error}")
            
    return None, f"Failed to place market order: {last_error}"


def modify_position_sltp(ticket: int, symbol: str, sl: float, tp: float) -> bool:
    """Modify the Stop Loss and Take Profit of an active position."""
    import MetaTrader5 as mt5
    # Get digits for formatting
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[Execution Engine] Failed to get symbol info for {symbol} to modify SL/TP")
        return False
    digits = symbol_info.digits
    sl = round(sl, digits)
    tp = round(tp, digits)
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": float(sl),
        "tp": float(tp)
    }
    
    res = mt5.order_send(request)
    if res is None:
        print(f"[Execution Engine] Modify SL/TP for position #{ticket} returned None")
        return False
        
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[Execution Engine] Modify SL/TP for position #{ticket} successful (SL: {sl:.3f}, TP: {tp:.3f})")
        return True
    else:
        print(f"[Execution Engine] Modify SL/TP for position #{ticket} failed: retcode={res.retcode}, comment='{res.comment}'")
        return False


def close_position(ticket: int, symbol: str, volume: float = None) -> bool:
    """Close an active position or partial close if volume is specified."""
    import MetaTrader5 as mt5
    positions = mt5.positions_get(ticket=ticket)
    if positions is None or len(positions) == 0:
        print(f"[Execution Engine] Position #{ticket} not found to close")
        return False
        
    pos = positions[0]
    broker_symbol = pos.symbol
    
    # Determine close volume
    close_vol = float(volume) if volume is not None else float(pos.volume)
    # Ensure close volume does not exceed position volume
    close_vol = min(close_vol, float(pos.volume))
    
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    
    # Get current price
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is None:
        print(f"[Execution Engine] Failed to get price tick to close position #{ticket}")
        return False
    price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
    
    # Try different filling types
    filling_types = [
        mt5.ORDER_FILLING_RETURN,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK
    ]
    
    for fill in filling_types:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "volume": close_vol,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": f"SMC Bot Soft TP ({close_vol:.2f} lot)",
            "type_filling": fill,
        }
        
        res = mt5.order_send(request)
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[Execution Engine] Position #{ticket} successfully closed/partially closed: {close_vol:.2f} lot using filling {fill}")
            return True
            
    print(f"[Execution Engine] Failed to close position #{ticket} using all filling types")
    return False


def load_sent_signals() -> dict:
    """Load the registry of already alerted signals (internal helper to avoid circular imports)."""
    import json
    import os
    path = _sent_signals_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _sent_signals_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sent_signals.json")


def save_sent_signals(sent_signals: dict):
    """Persist the live signal registry after trade-manager state changes."""
    import json
    path = _sent_signals_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(sent_signals, f, indent=4)


def _ticket_registry_suffix(sig_data: dict, ticket: int):
    if sig_data.get("ticket_a") == ticket:
        return "_a", "features_0.5"
    if sig_data.get("ticket_b") == ticket:
        return "_b", "features_0.618"
    if sig_data.get("ticket_id") == ticket:
        return "", "features"
    return None, None


def record_manager_exit_trigger(
    sent_signals: dict,
    ticket: int,
    *,
    trigger: str,
    timeframe: str,
    detail: str,
) -> bool:
    """Record the exact trade-manager rule that closed a ticket."""
    from datetime import datetime

    changed = False
    recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for sig_data in sent_signals.values():
        suffix, feature_key = _ticket_registry_suffix(sig_data, ticket)
        if suffix is None:
            continue

        updates = {
            f"manager_exit_trigger{suffix}": trigger,
            f"manager_exit_timeframe{suffix}": timeframe,
            f"manager_exit_detail{suffix}": detail,
            f"manager_exit_recorded_at{suffix}": recorded_at,
        }
        for key, value in updates.items():
            if sig_data.get(key) != value:
                sig_data[key] = value
                changed = True

        features = sig_data.get(feature_key)
        if isinstance(features, dict):
            feature_updates = {
                "manager_exit_trigger": trigger,
                "manager_exit_timeframe": timeframe,
                "manager_exit_detail": detail,
                "manager_exit_recorded_at": recorded_at,
            }
            for key, value in feature_updates.items():
                if features.get(key) != value:
                    features[key] = value
                    changed = True

    if changed:
        save_sent_signals(sent_signals)
    return changed


def _find_ticket_signal(sent_signals: dict, ticket: int):
    for sig_data in sent_signals.values():
        suffix, feature_key = _ticket_registry_suffix(sig_data, ticket)
        if suffix is not None:
            features = sig_data.get(feature_key)
            return sig_data, suffix, feature_key, features if isinstance(features, dict) else {}
    return None, None, None, {}


def record_runner_state(sent_signals: dict, ticket: int, updates: dict) -> bool:
    """Persist per-ticket runner state in both registry and feature payload."""
    sig_data, suffix, feature_key, features = _find_ticket_signal(sent_signals, ticket)
    if sig_data is None:
        return False

    changed = False
    for key, value in updates.items():
        registry_key = f"{key}{suffix}"
        if sig_data.get(registry_key) != value:
            sig_data[registry_key] = value
            changed = True
        if isinstance(features, dict) and features.get(key) != value:
            features[key] = value
            changed = True

    if changed:
        save_sent_signals(sent_signals)
    return changed


def _last_closed_trend(df: pd.DataFrame):
    """Return the latest closed-candle trend, ignoring the active candle when present."""
    if df is None or df.empty or 'Trend' not in df.columns:
        return None

    trends = pd.to_numeric(df['Trend'], errors='coerce').dropna()
    if trends.empty:
        return None

    closed_trends = trends if bool(getattr(df, "attrs", {}).get("closed_only", False)) else (trends.iloc[:-1] if len(trends) >= 2 else trends)
    if closed_trends.empty:
        return None
    return int(closed_trends.iloc[-1])


def _consecutive_closed_trend_count(df: pd.DataFrame, trend_value: int) -> int:
    """Count how many latest closed candles share the requested trend value."""
    if df is None or df.empty or 'Trend' not in df.columns:
        return 0

    trends = pd.to_numeric(df['Trend'], errors='coerce').dropna()
    if trends.empty:
        return 0

    closed_trends = trends if bool(getattr(df, "attrs", {}).get("closed_only", False)) else (trends.iloc[:-1] if len(trends) >= 2 else trends)
    count = 0
    for value in reversed(closed_trends.tolist()):
        if int(value) != int(trend_value):
            break
        count += 1
    return count


def _closed_candle_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the closed-candle slice used by live trade management."""
    if df is None or df.empty:
        return pd.DataFrame()
    if bool(getattr(df, "attrs", {}).get("closed_only", False)):
        return df.copy()
    return df.iloc[:-1].copy() if len(df) >= 2 else df.copy()


def _latest_closed_candle(df: pd.DataFrame):
    closed_df = _closed_candle_frame(df)
    if closed_df.empty:
        return None
    return closed_df.iloc[-1]


def should_emergency_exit_on_reversal(
    df_tf: pd.DataFrame,
    setup_timeframe: str,
    direction: int,
    h1_trend=None,
    h4_trend=None,
) -> bool:
    """
    Confirm opposite CHoCH before closing a live trade.
    Mitigation is strict to the setup timeframe: M15/M30 need two closed
    opposite candles, while H1/H4/D1 can exit on one closed opposite candle.
    """
    opposite_direction = -int(direction)
    latest_closed_trend = _last_closed_trend(df_tf)
    if latest_closed_trend != opposite_direction:
        return False

    timeframe = str(setup_timeframe).upper()
    if timeframe in {"H1", "H4", "D1"}:
        return True

    consecutive_opposite = _consecutive_closed_trend_count(df_tf, opposite_direction)
    if consecutive_opposite >= 2:
        return True

    return False


def should_early_mitigate_on_market_deterioration(
    df_tf: pd.DataFrame,
    setup_timeframe: str,
    direction: int,
    tick,
    entry_price: float,
    original_sl: float,
    pip_multiplier: float,
) -> tuple[bool, str]:
    """
    Allow an early same-timeframe cut only when structure, volume, and bid/ask
    price action all deteriorate together on closed-candle evidence.
    """
    if not _read_bool_env("MT5_EARLY_MITIGATION_ENABLED", True):
        return False, "early mitigation disabled"

    opposite_direction = -int(direction)
    if _last_closed_trend(df_tf) != opposite_direction:
        return False, "same-TF closed trend has not flipped"

    latest = _latest_closed_candle(df_tf)
    if latest is None:
        return False, "no closed candle available"

    try:
        open_price = float(latest["Open"])
        close_price = float(latest["Close"])
    except (KeyError, TypeError, ValueError):
        return False, "closed candle body unavailable"

    adverse_body = (close_price - open_price) * int(direction) < 0
    if not adverse_body:
        return False, "latest closed candle body is not adverse"

    closed_df = _closed_candle_frame(df_tf)
    volume_ok = True
    volume_detail = "volume unavailable"
    if "Volume" in closed_df.columns and len(closed_df) >= 2:
        volumes = pd.to_numeric(closed_df["Volume"], errors="coerce").dropna()
        if len(volumes) >= 2:
            latest_volume = float(volumes.iloc[-1])
            baseline = float(volumes.iloc[:-1].tail(20).mean())
            multiplier = _read_float_env("MT5_EARLY_MITIGATION_VOLUME_MULTIPLIER", 1.8)
            if baseline > 0 and np.isfinite(baseline):
                volume_ok = latest_volume >= baseline * multiplier
                volume_detail = f"volume {latest_volume:.0f} vs baseline {baseline:.0f} x {multiplier:.2f}"
            else:
                volume_detail = f"volume {latest_volume:.0f}; baseline unavailable"
    if not volume_ok:
        return False, volume_detail

    try:
        exit_price = _position_exit_price(direction, tick)
        entry_value = float(entry_price)
    except (TypeError, ValueError):
        return False, "entry or tick price unavailable"

    adverse_price_move = (entry_value - float(exit_price)) * int(direction)
    min_adverse_price = _read_float_env("MT5_EARLY_MITIGATION_MIN_ADVERSE_PIPS", 35.0) * float(pip_multiplier)
    try:
        risk_price = abs(entry_value - float(original_sl))
    except (TypeError, ValueError):
        risk_price = 0.0
    risk_fraction = _read_float_env("MT5_EARLY_MITIGATION_RISK_FRACTION", 0.40)
    required_adverse = max(min_adverse_price, risk_price * risk_fraction)

    if adverse_price_move < required_adverse:
        return (
            False,
            f"adverse bid/ask move {adverse_price_move:.3f} < required {required_adverse:.3f}",
        )

    spread_pips = (_live_spread_price(tick) / float(pip_multiplier)) if pip_multiplier > 0 else 0.0
    max_spread_pips = _read_float_env("MT5_EARLY_MITIGATION_MAX_SPREAD_PIPS", 12.0)
    if spread_pips > max_spread_pips:
        return False, f"spread too wide for reliable early mitigation ({spread_pips:.1f} pips)"

    return (
        True,
        (
            f"same-TF {str(setup_timeframe).upper()} closed trend flipped; "
            f"adverse closed body; {volume_detail}; "
            f"bid/ask adverse move {adverse_price_move:.3f} >= {required_adverse:.3f}; "
            f"spread {spread_pips:.1f} pips"
        ),
    )


_PROCESS_EXITED_TICKETS = set()


def manage_active_trades(symbol: str, magic: int, timeframes_data: dict):
    """
    Scans all active positions for the magic number and applies:
    1. Break Even Point (BEP) when price reaches 1:1 Risk-to-Reward.
    2. Structural Trailing Stop (SMC strategy 2) based on Swing Lows/Highs.
    3. Soft TP execution at TP 1 (Fibo TP) with Partial Close (Option B) if HTF trend is confluent, 
       or full close if HTF trend is not aligned.
    4. Emergency trend reversal exit (opposite CHoCH/Trend reversal) on the setup timeframe.
    """
    import MetaTrader5 as mt5
    from src.smc_detector import get_pip_multiplier
    from src.telegram_bot import send_telegram_alert
    
    broker_symbol = get_active_broker_symbol(symbol)
    positions = _positions_for_symbol_magic(broker_symbol, magic)
    if len(positions) == 0:
        return
        
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is None:
        return
        
    pip_multiplier = get_pip_multiplier(symbol)
    spread_buffer = 2.0 * pip_multiplier # 2 pips buffer (0.20 USD for Gold)
    
    # Check Higher Timeframe Trends for confluence
    h1_trend = _last_closed_trend(timeframes_data['H1']) if 'H1' in timeframes_data else None
    h4_trend = _last_closed_trend(timeframes_data['H4']) if 'H4' in timeframes_data else None
    
    for p in positions:
        ticket = p.ticket
        if ticket in _PROCESS_EXITED_TICKETS:
            continue
        entry_price = p.price_open
        current_sl = p.sl
        current_tp = p.tp
        current_vol = p.volume
        direction = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
        
        # We parse the timeframe and option type from the position comment (if set by bot)
        comment = p.comment
        tf = "M30" # default fallback
        if "H4" in comment:
            tf = "H4"
        elif "H1" in comment:
            tf = "H1"
        elif "D1" in comment:
            tf = "D1"
        elif "M15" in comment:
            tf = "M15"
            
        is_option_b = "Option B" in comment or "0.618" in comment or "GoldenPocket" in comment
        
        # 1. Look up original Stop Loss, TP1, and TP2 from sent signals registry
        original_sl = current_sl
        tp1 = 0.0
        tp2 = 0.0
        sent_signals = load_sent_signals()
        matched_sig_data = None
        matched_suffix = ""
        feat = {}
        for sig_key, sig_data in sent_signals.items():
            if sig_data.get('ticket_a') == ticket:
                matched_sig_data = sig_data
                matched_suffix = "_a"
                feat = sig_data.get('features_0.5', {}) or {}
                original_sl = feat.get('sl_price', current_sl)
                tp1 = feat.get('tp_price', 0.0)
                tp2 = feat.get('tp2_price', 0.0)
                break
            if sig_data.get('ticket_b') == ticket:
                matched_sig_data = sig_data
                matched_suffix = "_b"
                feat = sig_data.get('features_0.618', {}) or {}
                original_sl = feat.get('sl_price', current_sl)
                tp1 = feat.get('tp_price', 0.0)
                tp2 = feat.get('tp2_price', 0.0)
                break
            if sig_data.get('ticket_id') == ticket:
                matched_sig_data = sig_data
                matched_suffix = ""
                feat = sig_data.get('features', {}) or {}
                original_sl = feat.get('sl_price', current_sl)
                tp1 = feat.get('tp_price', 0.0)
                tp2 = feat.get('tp2_price', 0.0)
                break
                
        # Check if already fully exited by manager to prevent duplicate close spamming
        if matched_sig_data is not None:
            exit_trigger = matched_sig_data.get(f"manager_exit_trigger{matched_suffix}")
            if exit_trigger in {"emergency_reversal", "early_market_deterioration", "tp1_full_close", "runner_exhaustion_cut"}:
                print(f"[Execution Engine] Position #{ticket} already fully exited via '{exit_trigger}' according to registry. Skipping management.")
                continue

        # Fallback values if registry lookup fails
        if original_sl == 0:
            original_sl = current_sl

        exit_price = _position_exit_price(direction, tick)
        if matched_sig_data is not None and _read_bool_env("MT5_RUNNER_ENABLED", True):
            runner_active = bool(matched_sig_data.get(f"runner_active{matched_suffix}", False))
            if runner_active:
                previous_best = matched_sig_data.get(f"runner_best_price{matched_suffix}", exit_price)
                try:
                    previous_best = float(previous_best)
                except (TypeError, ValueError):
                    previous_best = exit_price
                best_price = max(previous_best, exit_price) if direction == 1 else min(previous_best, exit_price)
                if best_price != previous_best:
                    record_runner_state(sent_signals, ticket, {"runner_best_price": best_price})

                retrace_pips = ((best_price - exit_price) * direction) / pip_multiplier if pip_multiplier > 0 else 0.0
                cut_pips = _read_float_env("MT5_RUNNER_EXHAUSTION_RETRACE_PIPS", 25.0)
                if retrace_pips >= cut_pips:
                    print(
                        f"[Execution Engine] Runner exhaustion detected for #{ticket}: "
                        f"retraced {retrace_pips:.1f} pips from best price. Closing profit."
                    )
                    success = close_position(ticket, broker_symbol)
                    if success:
                        _PROCESS_EXITED_TICKETS.add(ticket)
                        record_runner_state(sent_signals, ticket, {"runner_active": False})
                        record_manager_exit_trigger(
                            sent_signals,
                            ticket,
                            trigger="runner_exhaustion_cut",
                            timeframe=tf,
                            detail=f"runner retraced {retrace_pips:.1f} pips from best price {best_price:.3f}",
                        )
                        continue

            elif tp1 > 0.0 and current_tp > 0.0 and _runner_near_primary_tp(direction, exit_price, tp1, pip_multiplier):
                if _runner_continuation_supported(feat, direction, tick, pip_multiplier, h1_trend, h4_trend):
                    protected_sl = _runner_protected_stop(entry_price, current_sl, direction, tick, pip_multiplier)
                    print(
                        f"[Execution Engine] Runner armed for #{ticket}: near TP1 {tp1:.3f}; "
                        f"removing server TP and protecting SL at {protected_sl:.3f}."
                    )
                    success = modify_position_sltp(ticket, broker_symbol, protected_sl, 0.0)
                    if success:
                        record_runner_state(
                            sent_signals,
                            ticket,
                            {
                                "runner_active": True,
                                "runner_best_price": exit_price,
                                "runner_primary_tp": float(tp1),
                            },
                        )
                        record_manager_exit_trigger(
                            sent_signals,
                            ticket,
                            trigger="runner_tp_removed",
                            timeframe=tf,
                            detail=f"near TP1 {tp1:.3f}; continuation confluence supported runner mode",
                        )
                        continue
            
        df_tf = timeframes_data.get(tf)
        
        # --- A. Emergency Reversal Exit (opposite CHoCH/Trend reversal) ---
        if df_tf is not None and not df_tf.empty and 'Trend' in df_tf.columns:
            if should_emergency_exit_on_reversal(df_tf, tf, direction, h1_trend, h4_trend):
                # Reversal detected on setup timeframe! Close the trade immediately.
                print(f"[Execution Engine] Confirmed trend reversal (opposite CHoCH) detected on {tf} for position #{ticket}. Closing deal.")
                success = close_position(ticket, broker_symbol)
                if success:
                    _PROCESS_EXITED_TICKETS.add(ticket)
                    record_manager_exit_trigger(
                        sent_signals,
                        ticket,
                        trigger="emergency_reversal",
                        timeframe=tf,
                        detail=f"opposite CHoCH confirmed on closed {tf} candles",
                    )
                    try:
                        send_telegram_alert(
                            f"🚨 <b>[SMC Trade Manager] Emergency Exit!</b>\n\n"
                            f"Timeframe: {tf}\n"
                            f"Posisi #{ticket} ({'BUY' if direction == 1 else 'SELL'}) ditutup lebih cepat di harga pasar "
                            f"karena terdeteksi pembalikan trend (Opposite CHoCH terjadi pada candle terakhir)."
                        )
                    except Exception:
                        pass
                    continue

            should_early_exit, early_detail = should_early_mitigate_on_market_deterioration(
                df_tf,
                tf,
                direction,
                tick,
                entry_price,
                original_sl,
                pip_multiplier,
            )
            if should_early_exit:
                print(
                    f"[Execution Engine] Early market deterioration detected on {tf} for "
                    f"position #{ticket}: {early_detail}. Closing deal."
                )
                success = close_position(ticket, broker_symbol)
                if success:
                    _PROCESS_EXITED_TICKETS.add(ticket)
                    record_manager_exit_trigger(
                        sent_signals,
                        ticket,
                        trigger="early_market_deterioration",
                        timeframe=tf,
                        detail=early_detail,
                    )
                    try:
                        send_telegram_alert(
                            f"🚨 <b>[SMC Trade Manager] Early Mitigation Exit</b>\n\n"
                            f"Timeframe: {tf}\n"
                            f"Posisi #{ticket} ({'BUY' if direction == 1 else 'SELL'}) ditutup lebih cepat karena "
                            f"same-timeframe closed candle, volume, dan bid/ask menunjukkan deteriorasi: <i>{early_detail}</i>."
                        )
                    except Exception:
                        pass
                    continue
                    
        # --- B. Soft TP Logic at TP 1 (Fibo TP) ---
        if tp1 > 0.0:
            is_tp1_reached = False
            if direction == 1: # Buy
                if tick.bid >= tp1:
                    is_tp1_reached = True
            else: # Sell
                if tick.ask <= tp1:
                    is_tp1_reached = True
                    
            if is_tp1_reached:
                # Check structure quality (H1 or H4 trend confluence)
                is_structure_good = (h1_trend == direction) or (h4_trend == direction)
                
                # If Option A or structure is not confluent (bad), close 100%
                if not is_option_b or not is_structure_good:
                    print(f"[Execution Engine] TP 1 reached for #{ticket}. Closing 100% (Confluence={is_structure_good}, OptB={is_option_b}).")
                    success = close_position(ticket, broker_symbol)
                    if success:
                        _PROCESS_EXITED_TICKETS.add(ticket)
                        reason = "Struktur HTF kurang mendukung untuk hold sisa" if not is_structure_good else "Option A selalu ditutup penuh di TP 1"
                        record_manager_exit_trigger(
                            sent_signals,
                            ticket,
                            trigger="tp1_full_close",
                            timeframe=tf,
                            detail=f"TP1 reached at {tp1:.3f}; {reason}",
                        )
                        try:
                            send_telegram_alert(
                                f"🎯 <b>[SMC Trade Manager] TP 1 Hit! (Sapu Bersih)</b>\n\n"
                                f"Timeframe: {tf}\n"
                                f"Posisi #{ticket} ({'BUY' if direction == 1 else 'SELL'}) ditutup penuh di level Fibo TP 1: <code>{tp1:.3f}</code>.\n"
                                f"• Alasan: <i>{reason}</i>."
                            )
                        except Exception:
                            pass
                        continue
                else:
                    # Option B and structure is GOOD! Do partial close if not already done
                    # (we check if current volume is still original 0.02 or larger)
                    has_partial = matched_sig_data.get(f"manager_exit_trigger{matched_suffix}") == "tp1_partial_close" if matched_sig_data is not None else False
                    if not has_partial and current_vol >= 0.02:
                        print(f"[Execution Engine] TP 1 reached for Option B #{ticket}. Performing 50% partial close.")
                        success_partial = close_position(ticket, broker_symbol, volume=0.01)
                        if success_partial:
                            record_manager_exit_trigger(
                                sent_signals,
                                ticket,
                                trigger="tp1_partial_close",
                                timeframe=tf,
                                detail=f"TP1 reached at {tp1:.3f}; partial close and move remainder toward TP2",
                            )
                            # Move SL of remaining portion to BEP and TP to TP2 (Dynamic TP)
                            new_sl = entry_price + spread_buffer * direction
                            new_tp = tp2 if tp2 > 0 else (entry_price + (entry_price - original_sl) * 3 if direction == 1 else entry_price - (original_sl - entry_price) * 3)
                            success_modify = modify_position_sltp(ticket, broker_symbol, new_sl, new_tp)
                            
                            try:
                                send_telegram_alert(
                                    f"🎯 <b>[SMC Trade Manager] TP 1 Hit! (Partial Close)</b>\n\n"
                                    f"Timeframe: {tf}\n"
                                    f"Posisi Option B #{ticket} ({'BUY' if direction == 1 else 'SELL'}) mengenai Fibo TP 1: <code>{tp1:.3f}</code>.\n"
                                    f"• <b>Tindakan:</b> Ditutup 50% (0.01 lot) untuk mengunci profit.\n"
                                    f"• <b>Sisa 0.01 Lot:</b> Dibiarkan jalan ke TP 2 (Struktur): <code>{new_tp:.3f}</code> dengan SL digeser ke BEP: <code>{new_sl:.3f}</code>."
                                )
                            except Exception:
                                pass
                            continue
                            
        # --- C. Profit protection: BEP, pip ladder, and structural trailing ---
        # Volatility-aware floor: in high-ATR sessions (e.g. New York) a normal
        # pullback is large, so we delay the 1R->BEP arm until profit clears the
        # ATR noise band. Falls back to the old behaviour when scaling is off.
        atr_pips = _atr_pips_from_df(df_tf, pip_multiplier)
        bep_arm_floor_price = 0.0
        if _read_bool_env("MT5_ATR_SCALED_LOCK", False) and atr_pips and atr_pips > 0:
            arm_floor_pips = _read_float_env("MT5_ATR_LOCK_ARM_MULT", 1.5) * atr_pips
            bep_arm_floor_price = arm_floor_pips * pip_multiplier

        selected_sl = None
        if original_sl != 0:
            initial_risk = abs(entry_price - original_sl)
            # Require the move to clear BOTH 1R and the ATR noise floor before BEP.
            required_advance = max(initial_risk, bep_arm_floor_price)
            if initial_risk > 0:
                if direction == 1 and tick.bid >= entry_price + required_advance:
                    selected_sl = _select_best_stop_loss(
                        selected_sl,
                        entry_price + spread_buffer,
                        current_sl,
                        direction,
                        tick,
                    )
                elif direction == -1 and tick.ask <= entry_price - required_advance:
                    selected_sl = _select_best_stop_loss(
                        selected_sl,
                        entry_price - spread_buffer,
                        current_sl,
                        direction,
                        tick,
                    )

        profit_lock_sl = _profit_lock_stop_loss(
            entry_price,
            direction,
            tick,
            pip_multiplier,
            spread_buffer,
            symbol=p.symbol,
            position_profit=p.profit,
            atr_pips=atr_pips,
        )
        selected_sl = _select_best_stop_loss(selected_sl, profit_lock_sl, current_sl, direction, tick)

        # --- D. SMC-based Structural Trailing (Strategy 2) ---
        if df_tf is not None and not df_tf.empty:
            buffer = 2.0 * pip_multiplier
            if direction == 1: # Buy
                swing_lows = df_tf['Swing_Low'].dropna()
                if not swing_lows.empty:
                    recent_swing_low = swing_lows.iloc[-1]
                    new_sl = recent_swing_low - buffer
                    selected_sl = _select_best_stop_loss(selected_sl, new_sl, current_sl, direction, tick)
            else: # Sell
                swing_highs = df_tf['Swing_High'].dropna()
                if not swing_highs.empty:
                    recent_swing_high = swing_highs.iloc[-1]
                    new_sl = recent_swing_high + buffer
                    selected_sl = _select_best_stop_loss(selected_sl, new_sl, current_sl, direction, tick)

        if selected_sl is not None:
            success = modify_position_sltp(ticket, broker_symbol, selected_sl, current_tp)
            if success:
                current_sl = selected_sl
