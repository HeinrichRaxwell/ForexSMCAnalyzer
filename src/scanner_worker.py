import os
import sys
import time
import json
import argparse
from datetime import datetime
from html import escape
import pandas as pd
import numpy as np

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import (
    detect_swing_points,
    detect_structures,
    detect_fvg_and_ob,
    detect_snr_and_swapzones,
    detect_bpr,
    get_pip_multiplier,
    detect_indecision_candles,
    detect_supply_demand_zones,
)
from src.labeler import get_killzone
from src.inference import predict_setup_probability, process_mt5_history_feedback
from src.rejection_detector import detect_rejection_at_level
from src.main import find_dynamic_tp, extract_active_htf_fvgs, get_active_setups, plot_smc_chart
from src.telegram_bot import send_telegram_alert
from src.indicators.knn_classifier import run_knn_classifier, calculate_knn_probability_at_bar
from src.indicators.volume_clusters import calculate_volume_clusters
from src.execution import execute_trade_for_setup, execute_market_order_for_setup, manage_active_trades
from src.shadow_tracker import (
    build_shadow_signal_records,
    process_shadow_signal_outcomes,
    should_shadow_signal,
    upsert_shadow_signals,
)

# Storage for sent signal signatures
SENT_SIGNALS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sent_signals.json")
SHADOW_SIGNALS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shadow_signals.json")
DEFAULT_ACCEPT_THRESHOLD = 0.50


def get_accept_threshold(cli_threshold=None, default: float = DEFAULT_ACCEPT_THRESHOLD) -> float:
    """Resolve live execution confidence threshold from CLI, env, then default."""
    if cli_threshold is not None:
        return float(cli_threshold)

    raw_value = os.getenv("ML_ACCEPT_THRESHOLD")
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        print(f"[Scanner Config] Invalid ML_ACCEPT_THRESHOLD={raw_value!r}; using {default}.")
        return default


STRATEGY_DISPLAY_NAMES = {
    'FVG': "Fair Value Gap",
    'OB': "Order Block",
    'BPR': "Balanced Price Range",
    'IC': "Indecision Candle",
    'SND': "Supply/Demand Zone",
    'Pivot': "Pivot Rejection",
    'Swapzone': "Swapzone (SBR/RBS)",
    'Breaker': "Breaker Block",
}


def _html_text(value) -> str:
    return escape(str(value), quote=False)


def _format_price(value) -> str:
    try:
        return f"<code>{float(value):.3f}</code>"
    except (TypeError, ValueError):
        return "<code>-</code>"


def _format_percent(value) -> str:
    try:
        return f"<code>{float(value):.2%}</code>"
    except (TypeError, ValueError):
        return "<code>-</code>"


def get_strategy_display_name(strategy: str) -> str:
    return STRATEGY_DISPLAY_NAMES.get(strategy, strategy or "SMC Setup")


def format_direction_label(direction: int) -> str:
    return "BUY / Long" if int(direction) == 1 else "SELL / Short"


def format_rejection_status(rejection_sources) -> str:
    clean_sources = sorted({
        str(src)
        for src in rejection_sources
        if src not in (None, "", "None")
    })
    if clean_sources:
        return f"Confirmed ({'/'.join(clean_sources)})"
    return "Not confirmed on current LTF touch"


def format_htf_priority_status(is_prioritized: bool) -> str:
    return "Confirmed" if is_prioritized else "Not active"


def format_execution_status(ticket_id, execution_message: str, skipped_peer: str = None, monitoring_only: bool = False) -> str:
    if monitoring_only:
        return "Monitoring only (MT5_EXECUTE_TRADES disabled)"
    if ticket_id and "MARKET" in str(execution_message).upper():
        return f"Market order active (ticket #{ticket_id})"
    if ticket_id:
        return f"Pending order placed (ticket #{ticket_id})"
    if skipped_peer:
        return f"Skipped ({skipped_peer} market order active)"
    return f"Failed ({_html_text(execution_message)})"


def _format_confluence_lines(confluences) -> str:
    if not confluences:
        return "No extra confluence detail recorded."
    return "\n".join(f"{idx}. {_html_text(reason)}" for idx, reason in enumerate(confluences, start=1))


def _format_htf_match_lines(htf_matches) -> str:
    if not htf_matches:
        return ""
    lines = ["", "<b>HTF Match</b>"]
    for match in htf_matches:
        timeframe = _html_text(match.get('timeframe', '-'))
        lines.append(f"- <code>{timeframe}</code> FVG {_format_price(match.get('bottom'))}-{_format_price(match.get('top'))}")
    return "\n".join(lines)


def format_dual_signal_message(
    *,
    symbol: str,
    timeframe: str,
    direction: int,
    setup_desc: str,
    probability_a: float,
    probability_b: float,
    confidence_threshold: float,
    opt_a: dict,
    opt_b: dict,
    execution_status_a: str,
    execution_status_b: str,
    htf_priority_status: str,
    rejection_status: str,
    confluences,
    htf_matches,
) -> str:
    """Build the Telegram body for a dual-fib trade signal."""
    return (
        f"<b>SMC Trade Signal - {_html_text(symbol)}</b>\n\n"
        f"<b>Signal</b>\n"
        f"Symbol: <code>{_html_text(symbol)}</code>\n"
        f"Timeframe: <code>{_html_text(timeframe)}</code>\n"
        f"Direction: <b>{format_direction_label(direction)}</b>\n"
        f"Setup: {_html_text(setup_desc)}\n\n"
        f"<b>Model Confidence</b>\n"
        f"0.500 entry (0.01 lot): {_format_percent(probability_a)}\n"
        f"0.618 entry (0.02 lot): {_format_percent(probability_b)}\n"
        f"Accept threshold: {_format_percent(confidence_threshold)}\n\n"
        f"<b>Execution</b>\n"
        f"HTF priority: {_html_text(htf_priority_status)}\n"
        f"LTF rejection: {_html_text(rejection_status)}\n"
        f"Order 0.500: {execution_status_a}\n"
        f"Order 0.618: {execution_status_b}\n\n"
        f"<b>Levels</b>\n"
        f"Entry 0.500 (0.01 lot): {_format_price(opt_a.get('entry_price'))}\n"
        f"Entry 0.618 (0.02 lot): {_format_price(opt_b.get('entry_price'))}\n"
        f"Stop Loss: {_format_price(opt_a.get('sl_price'))}\n"
        f"TP1: {_format_price(opt_a.get('tp_price'))}\n"
        f"TP2 dynamic: {_format_price(opt_a.get('tp2_price'))}\n"
        f"TP3 extension: {_format_price(opt_a.get('tp3_price'))}\n\n"
        f"<b>Confluence</b>\n"
        f"{_format_confluence_lines(confluences)}"
        f"{_format_htf_match_lines(htf_matches)}\n\n"
        f"<i>Automated alert from Forex SMC AI Analyzer.</i>"
    )


def format_single_signal_message(
    *,
    symbol: str,
    timeframe: str,
    direction: int,
    setup_desc: str,
    probability: float,
    confidence_threshold: float,
    setup: dict,
    execution_status: str,
    htf_priority_status: str,
    rejection_status: str,
    confluences,
    htf_matches,
) -> str:
    """Build the Telegram body for a single-entry trade signal."""
    return (
        f"<b>SMC Trade Signal - {_html_text(symbol)}</b>\n\n"
        f"<b>Signal</b>\n"
        f"Symbol: <code>{_html_text(symbol)}</code>\n"
        f"Timeframe: <code>{_html_text(timeframe)}</code>\n"
        f"Direction: <b>{format_direction_label(direction)}</b>\n"
        f"Setup: {_html_text(setup_desc)}\n\n"
        f"<b>Model Confidence</b>\n"
        f"Entry confidence: {_format_percent(probability)}\n"
        f"Accept threshold: {_format_percent(confidence_threshold)}\n\n"
        f"<b>Execution</b>\n"
        f"HTF priority: {_html_text(htf_priority_status)}\n"
        f"LTF rejection: {_html_text(rejection_status)}\n"
        f"Order status: {execution_status}\n\n"
        f"<b>Levels</b>\n"
        f"Entry: {_format_price(setup.get('entry_price'))}\n"
        f"Stop Loss: {_format_price(setup.get('sl_price'))}\n"
        f"TP1: {_format_price(setup.get('tp_price'))}\n"
        f"TP2 dynamic: {_format_price(setup.get('tp2_price'))}\n"
        f"TP3 extension: {_format_price(setup.get('tp3_price'))}\n\n"
        f"<b>Confluence</b>\n"
        f"{_format_confluence_lines(confluences)}"
        f"{_format_htf_match_lines(htf_matches)}\n\n"
        f"<i>Automated alert from Forex SMC AI Analyzer.</i>"
    )


def send_recovery_alert_with_chart(
    message: str,
    *,
    timeframes_data: dict,
    timeframe: str,
    symbol: str,
    direction_name: str,
    strategy: str,
    setups: list,
    image_suffix: str,
) -> bool:
    """Send a recovery alert with the same chart snapshot flow as new entries."""
    image_filename = None
    try:
        tf_df = timeframes_data.get(timeframe)
        if tf_df is None:
            raise KeyError(f"timeframe data not found: {timeframe}")

        setup_index = "unknown"
        if setups:
            setup_index = setups[0].get("index", "unknown")
        image_filename = f"temp_alert_{timeframe}_{image_suffix}_{setup_index}.png"
        title = f"{symbol} {timeframe} - {direction_name} Recovery {get_strategy_display_name(strategy)} Confluence"
        plot_smc_chart(tf_df, title=title, active_setups=setups, output_filename=image_filename)
    except Exception as e:
        print(f"Failed to generate recovery chart image: {e}")
        image_filename = None

    try:
        return send_telegram_alert(message, image_filename)
    finally:
        if image_filename and os.path.exists(image_filename):
            try:
                os.remove(image_filename)
            except Exception:
                pass


def should_market_enter_setup(setup: dict, current_price: float, entry_buffer: float = 0.5) -> bool:
    """Return True when price is inside the instant-entry zone after confirmed rejection."""
    if current_price is None or not setup.get("rejection_confirmed", False):
        return False

    direction = int(setup.get("direction", 1))
    entry_price = float(setup["entry_price"])
    sl_price = float(setup["sl_price"])
    current_price = float(current_price)

    if direction == 1:
        return sl_price + entry_buffer <= current_price <= entry_price + entry_buffer
    return entry_price - entry_buffer <= current_price <= sl_price - entry_buffer


def choose_recovery_execution_mode(setup: dict, current_price: float) -> str:
    """Choose market recovery when price has returned to a confirmed rejection zone."""
    return "market" if should_market_enter_setup(setup, current_price) else "pending"


def should_promote_low_confidence_record(sig_data: dict, ticket_fields) -> bool:
    """Return True when a shadow-tracked setup can fall through to live execution."""
    if not sig_data or not sig_data.get("is_low_confidence", False):
        return False

    if any(sig_data.get(field) is not None for field in ticket_fields):
        return False

    outcome_fields = ("outcome_recorded", "outcome_a_recorded", "outcome_b_recorded")
    return not any(sig_data.get(field, False) for field in outcome_fields)


def choose_dual_market_entry_option(opt_a: dict, opt_b: dict, current_price: float, entry_buffer: float = 0.5):
    """Choose which dual fib layer should become a market order, preserving 0.5 priority near entry."""
    if current_price is None:
        return None

    direction = int(opt_a.get("direction", 1))
    current_price = float(current_price)

    if direction == 1:
        if (
            opt_a.get("rejection_confirmed", False)
            and float(opt_b["entry_price"]) <= current_price <= float(opt_a["entry_price"]) + entry_buffer
        ):
            return "a"
        if (
            opt_b.get("rejection_confirmed", False)
            and float(opt_a["sl_price"]) + entry_buffer <= current_price < float(opt_b["entry_price"])
        ):
            return "b"
    else:
        if (
            opt_a.get("rejection_confirmed", False)
            and float(opt_a["entry_price"]) - entry_buffer <= current_price <= float(opt_b["entry_price"])
        ):
            return "a"
        if (
            opt_b.get("rejection_confirmed", False)
            and float(opt_b["entry_price"]) < current_price <= float(opt_a["sl_price"]) - entry_buffer
        ):
            return "b"

    return None


def choose_dual_recovery_execution_mode(opt_a: dict, opt_b: dict, current_price: float, option: str) -> str:
    """Use the same dual-fib market priority during recovery as during first execution."""
    market_option = choose_dual_market_entry_option(opt_a, opt_b, current_price)
    if market_option == option:
        return "market"
    if market_option in {"a", "b"}:
        return "skip"
    return "pending"


def drop_latest_forming_candle(df_tf: pd.DataFrame) -> pd.DataFrame:
    """Return only fully closed candles from an MT5 OHLC frame."""
    if df_tf is None or df_tf.empty:
        return df_tf
    return df_tf.iloc[:-1].copy()


def apply_smc_detectors(df_tf: pd.DataFrame, symbol: str, closed_only: bool = False) -> pd.DataFrame:
    """Run the full live SMC detector pipeline for one timeframe."""
    if closed_only:
        df_tf = drop_latest_forming_candle(df_tf)

    df_tf = detect_swing_points(df_tf)
    df_tf = detect_structures(df_tf)
    df_tf = detect_fvg_and_ob(df_tf, symbol=symbol)
    df_tf = detect_snr_and_swapzones(df_tf)
    df_tf = detect_bpr(df_tf, symbol=symbol)
    df_tf = detect_indecision_candles(df_tf, symbol=symbol)
    df_tf = detect_supply_demand_zones(df_tf, symbol=symbol)
    return df_tf

def load_sent_signals() -> dict:
    """Load the registry of already alerted signals."""
    if os.path.exists(SENT_SIGNALS_FILE):
        try:
            with open(SENT_SIGNALS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_sent_signals(sent_dict: dict):
    """Save the registry of alerted signals to disk."""
    os.makedirs(os.path.dirname(SENT_SIGNALS_FILE), exist_ok=True)
    with open(SENT_SIGNALS_FILE, "w") as f:
        json.dump(sent_dict, f, indent=4)


def register_shadow_candidate(
    sig_key: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction_name: str,
    accept_threshold: float,
    opt: dict = None,
    probability: float = None,
    opt_a: dict = None,
    probability_a: float = None,
    opt_b: dict = None,
    probability_b: float = None,
    shadow_signals_file: str = SHADOW_SIGNALS_FILE,
    now: str = None,
) -> bool:
    """Store below-threshold candidates for virtual outcome tracking without executing them."""
    if opt is not None:
        if not should_shadow_signal(probability, accept_threshold):
            return False
    else:
        probs = [p for p in (probability_a, probability_b) if p is not None]
        if not probs or not any(should_shadow_signal(p, accept_threshold) for p in probs):
            return False

    records = build_shadow_signal_records(
        signal_id=sig_key,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction_name=direction_name,
        accept_threshold=accept_threshold,
        opt=opt,
        probability=probability,
        opt_a=opt_a,
        probability_a=probability_a,
        opt_b=opt_b,
        probability_b=probability_b,
        now=now,
    )
    return upsert_shadow_signals(records, shadow_signals_file=shadow_signals_file)


def register_low_confidence_lead(
    lead: dict,
    sent_signals: dict,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction_name: str,
    accept_threshold: float,
    shadow_signals_file: str = SHADOW_SIGNALS_FILE,
    now: str = None,
) -> bool:
    """Register a below-threshold lead in the silent registry and shadow tracker."""
    if lead['is_dual']:
        opt_a = lead['opt_a']
        opt_b = lead['opt_b']
        prob_a = lead['prob_a']
        prob_b = lead['prob_b']

        setup_time_str = str(opt_a['time'])
        sig_key = f"{timeframe}_{strategy}_DUAL_{direction_name}_{opt_a['entry_price']:.3f}_{opt_b['entry_price']:.3f}_{setup_time_str.replace(' ', '_')}"

        opt_a['status'] = "FILTERED (Low Confidence)"
        opt_b['status'] = "FILTERED (Low Confidence)"

        changed = False
        if sig_key not in sent_signals:
            print(f"[Scanner Registry] Registering low confidence dual {timeframe} {strategy} at {opt_a['time']} for manual tracking.")
            sent_signals[sig_key] = {
                'time_sent': now or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'timeframe': timeframe,
                'direction': direction_name,
                'type': strategy,
                'price_0.5': opt_a['entry_price'],
                'price_0.618': opt_b['entry_price'],
                'probability_0.5': prob_a,
                'probability_0.618': prob_b,
                'ticket_a': None,
                'ticket_b': None,
                'reentries_count': 0,
                'features_0.5': opt_a['features'],
                'features_0.618': opt_b['features'],
                'is_low_confidence': True,
            }
            changed = True

        shadow_changed = register_shadow_candidate(
            sig_key=sig_key,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            direction_name=direction_name,
            accept_threshold=accept_threshold,
            opt_a=opt_a,
            probability_a=prob_a,
            opt_b=opt_b,
            probability_b=prob_b,
            shadow_signals_file=shadow_signals_file,
            now=now,
        )
        return changed or shadow_changed

    opt = lead['opt']
    prob = lead['max_prob']

    setup_time_str = str(opt['time'])
    sig_key = f"{timeframe}_{strategy}_SINGLE_{direction_name}_{opt['entry_price']:.3f}_{setup_time_str.replace(' ', '_')}"

    opt['status'] = "FILTERED (Low Confidence)"

    changed = False
    if sig_key not in sent_signals:
        print(f"[Scanner Registry] Registering low confidence single {timeframe} {strategy} at {opt['time']} for manual tracking.")
        sent_signals[sig_key] = {
            'time_sent': now or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframe': timeframe,
            'direction': direction_name,
            'type': strategy,
            'price': opt['entry_price'],
            'probability': prob,
            'reentries_count': 0,
            'features': opt['features'],
            'is_low_confidence': True,
        }
        changed = True

    shadow_changed = register_shadow_candidate(
        sig_key=sig_key,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction_name=direction_name,
        accept_threshold=accept_threshold,
        opt=opt,
        probability=prob,
        shadow_signals_file=shadow_signals_file,
        now=now,
    )
    return changed or shadow_changed


def process_existing_shadow_outcomes(
    timeframes_data: dict,
    shadow_signals_file: str = SHADOW_SIGNALS_FILE,
    shadow_labeled_data_path: str = None,
    now: str = None,
    trigger_retrain: bool = True,
) -> dict:
    """Resolve previously tracked shadow signals using the latest fetched candles."""
    kwargs = {
        "shadow_signals_file": shadow_signals_file,
        "now": now,
    }
    if shadow_labeled_data_path is not None:
        kwargs["shadow_labeled_data_path"] = shadow_labeled_data_path
    result = process_shadow_signal_outcomes(timeframes_data, **kwargs)
    result["retrain_result"] = None

    labeled_rows = int(result.get("labeled_rows_appended", 0) or 0)
    if trigger_retrain and labeled_rows > 0:
        from src.inference import check_and_trigger_retraining

        result["retrain_result"] = check_and_trigger_retraining(labeled_rows)

    return result


def prune_invalid_pending_orders(symbol: str, magic: int, active_high_confidence_setups: list):
    """
    Cancel pending orders on the MT5 account that are no longer active, 
    mitigated, or too far away from the current price.
    """
    import MetaTrader5 as mt5
    from src.execution import get_active_broker_symbol
    broker_symbol = get_active_broker_symbol(symbol)
    orders = mt5.orders_get(symbol=broker_symbol, magic=magic)
    if orders is None or len(orders) == 0:
        return
        
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is None:
        return
    current_price = tick.ask if len(orders) > 0 else tick.bid
    
    # We check if there is an active high-confidence setup with a matching direction and close entry price
    cancelled_tickets = []
    for o in orders:
        o_price = o.price_open
        o_type = o.type
        
        is_still_valid = False
        tf = None
        for s in active_high_confidence_setups:
            s_o_type = 2 if s['direction'] == 1 else 3
            if s_o_type == o_type:
                # Proximity check: 0.15 USD tolerance for Gold
                if abs(s['entry_price'] - o_price) < 0.15:
                    is_still_valid = True
                    tf = s['timeframe']
                    break
                    
        # Define maximum allowed distance from market price based on timeframe
        max_dist = 30.0  # Default fallback
        if tf == 'D1':
            max_dist = 200.0
        elif tf == 'H4':
            max_dist = 100.0
        elif tf == 'H1':
            max_dist = 60.0
        elif tf == 'M30':
            max_dist = 30.0
        elif tf == 'M15':
            max_dist = 20.0
            
        price_diff = abs(o.price_open - tick.last) if tick.last > 0 else abs(o.price_open - current_price)
        is_too_far = price_diff > max_dist
        
        if not is_still_valid or is_too_far:
            reason = "structure mitigated/invalid" if not is_still_valid else f"too far from market (>{max_dist} USD)"
            print(f"[Risk Management] Cancelling zombie/invalid pending order #{o.ticket} ({reason}).")
            
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": o.ticket,
            }
            res = mt5.order_send(request)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[Risk Management] Order #{o.ticket} successfully cancelled.")
                cancelled_tickets.append((o.ticket, reason))
                
    if cancelled_tickets:
        try:
            lines = [f"🧹 <b>[Risk Management] Cleaned up {len(cancelled_tickets)} zombie pending orders:</b>"]
            for ticket, reason in cancelled_tickets:
                lines.append(f"• Order #{ticket} ({reason})")
            send_telegram_alert("\n".join(lines))
        except Exception:
            pass

def is_good_fvg(df: pd.DataFrame, idx: int, setup: dict, symbol: str, timeframes_data: dict) -> tuple:
    """
    Applies high-quality SMC FVG filters to ensure only high-probability setups are taken.
    Returns: (is_good, reason)
    """
    # 1. HTF Trend Confluence (with D1 Trend) - Disabled per user request (Option 2 - Aggressive)
    # d1_df = timeframes_data.get('D1')
    # if d1_df is not None and not d1_df.empty:
    #     d1_trend = d1_df['Trend'].iloc[-1]
    #     if setup['direction'] != d1_trend:
    #         return False, f"Trend conflict: Setup is {'BULL' if setup['direction'] == 1 else 'BEAR'} but D1 Trend is {'BULL' if d1_trend == 1 else 'BEAR'}"
            
    # 2. Displacement Candle Volume (Buyer/Seller Pressure)
    # The candle at idx-1 is the middle candle that created the gap.
    if idx >= 1:
        vol = df['Volume'].iloc[idx-1]
        high = df['High'].iloc[idx-1]
        low = df['Low'].iloc[idx-1]
        close = df['Close'].iloc[idx-1]
        
        candle_range = high - low
        if candle_range > 0:
            if setup['direction'] == 1:  # Buy (Bullish) - We want Buyer Volume
                buyer_vol = vol * (close - low) / candle_range
                buyer_ratio = buyer_vol / vol
                if buyer_ratio < 0.5:
                    return False, f"Weak buyer volume pressure ({buyer_ratio:.2f} < 0.50)"
            else:  # Sell (Bearish) - We want Seller Volume
                seller_vol = vol * (high - close) / candle_range
                seller_ratio = seller_vol / vol
                if seller_ratio < 0.5:
                    return False, f"Weak seller volume pressure ({seller_ratio:.2f} < 0.50)"
                    
        # Calculate 20-period average volume
        # avg_vol = df['Volume'].rolling(window=20).mean().iloc[idx-1]
        # if pd.notna(avg_vol) and avg_vol > 0:
        #     if vol < 1.1 * avg_vol:
        #         return False, f"Low displacement volume ({vol:.0f} < 1.1x avg {avg_vol:.0f})"
        pass
                
    # 3. Displacement Candle Body Size
    if idx >= 1:
        high = df['High'].iloc[idx-1]
        low = df['Low'].iloc[idx-1]
        close = df['Close'].iloc[idx-1]
        open_val = df['Open'].iloc[idx-1]
        
        candle_range = high - low
        body_size = abs(close - open_val)
        if candle_range > 0:
            body_ratio = body_size / candle_range
            if body_ratio < 0.5:
                return False, f"Weak displacement candle (body/range {body_ratio:.2f} < 0.5)"
                
    # 4. FVG Width Constraints
    width = setup['fvg_width']
    atr = setup['atr_14']
    if atr > 0:
        rel_width = width / atr
        if rel_width < 0.25:
            return False, f"FVG width too narrow ({rel_width:.2f} < 0.25 ATR)"
        # Max width constraint removed per user request
        pass
            
    # Absolute width check for Gold (XAUUSD)
    symbol_upper = symbol.upper()
    if "XAUUSD" in symbol_upper or "GOLD" in symbol_upper:
        tf = setup.get('timeframe', 'H4')
        min_width = 1.0 if tf == 'M30' else 1.5
        if width < min_width:
            return False, f"FVG width too narrow ({width:.2f} USD < {min_width} USD)"
        # Max width constraint removed per user request
        pass
            
    return True, "Valid High-Quality FVG"

def run_scan(symbol: str, confidence_threshold: float):
    """Run a single scan cycle across all timeframes and send new signals to Telegram."""
    print(f"\n--- Starting Scan Cycle for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    registry_changed = False
    
    # 1. Try to connect to MT5 Exness terminal
    import MetaTrader5 as mt5
    if not connect_mt5():
        print("[Scanner Error] Failed to connect to MetaTrader 5 terminal. Skipping cycle.")
        return
        
    # 1.5. Run feedback loop to process MT5 history outcomes and retrain model
    try:
        feedback_result = process_mt5_history_feedback(return_details=True)
        new_feedback_count = feedback_result.get('feedback_count', 0)
        retrain_result = feedback_result.get('retrain_result') or {}
        if new_feedback_count > 0:
            if retrain_result.get('retrained'):
                print(f"[Feedback Loop] Learned {new_feedback_count} new outcomes and retrained model.")
                send_telegram_alert(
                    f"🔄 <b>Bot AI mempelajari {new_feedback_count} hasil trade baru dan melakukan retraining otomatis.</b>"
                )
            elif retrain_result.get('status') == 'ERROR':
                print(f"[Feedback Loop] Learned {new_feedback_count} new outcomes, but retraining failed: {retrain_result.get('error')}")
                send_telegram_alert(
                    f"⚠️ <b>Bot AI mempelajari {new_feedback_count} hasil trade baru, tetapi retraining gagal:</b> "
                    f"<code>{retrain_result.get('error')}</code>"
                )
            else:
                threshold = retrain_result.get('threshold', '?')
                accumulated = retrain_result.get('new_trades_since_last_train', new_feedback_count)
                print(f"[Feedback Loop] Learned {new_feedback_count} new outcomes. Retraining deferred ({accumulated}/{threshold}).")
                send_telegram_alert(
                    f"🧠 <b>Bot AI mempelajari {new_feedback_count} hasil trade baru.</b>\n"
                    f"Retraining ditunda sampai akumulasi mencapai <b>{threshold}</b> trade."
                )
    except Exception as e:
        print(f"[Feedback Loop Error] {e}")
        
    timeframes_data = {}
    mt5_active = True
    
    try:
        # Fetch multi-timeframe data with expanded lookback bars
        print("Fetching multi-timeframe data from MT5...")
        timeframes_data['D1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 100)
        timeframes_data['H4'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
        timeframes_data['H1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
        timeframes_data['M30'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 400)
        timeframes_data['M15'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 500)
        timeframes_data['M5'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M5, 500)
        timeframes_data['M1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M1, 500)
    except Exception as e:
        print(f"[Scanner Error] Error loading data from MT5: {e}")
        import MetaTrader5 as mt5
        mt5.shutdown()
        return

    d1_pivot_source = timeframes_data.get('D1')

    try:
        shadow_result = process_existing_shadow_outcomes(timeframes_data)
        if shadow_result.get("resolved_count", 0) or shadow_result.get("expired_count", 0):
            print(
                "[Shadow Tracker] Resolved "
                f"{shadow_result.get('resolved_count', 0)} shadow signals, "
                f"expired {shadow_result.get('expired_count', 0)}, "
                f"appended {shadow_result.get('labeled_rows_appended', 0)} labeled rows."
            )
    except Exception as e:
        print(f"[Shadow Tracker Error] {e}")

    # Draw daily pivots on MT5 charts
    from src.indicators.pivots import draw_pivots_on_mt5
    try:
        df_d1 = d1_pivot_source
        if df_d1 is not None and not df_d1.empty:
            draw_pivots_on_mt5(symbol, df_d1)
            print("[Pivots Visualizer] Drew/updated Daily Pivots on MT5 charts.")
    except Exception as e:
        print(f"[Pivots Visualizer Error] {e}")
    
    # 2. Run SMC detection algorithms on all timeframes
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        df_tf = apply_smc_detectors(df_tf, symbol=symbol, closed_only=True)
         # Calculate ATR_14
        close_prev = df_tf['Close'].shift(1).fillna(df_tf['Open'])
        tr = np.maximum(
            df_tf['High'] - df_tf['Low'],
            np.maximum(
                np.abs(df_tf['High'] - close_prev),
                np.abs(df_tf['Low'] - close_prev)
            )
        )
        df_tf['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
        timeframes_data[tf_name] = df_tf
        
    # Pre-calculate trends for FLOOP Pro MTF/HTF
    from src.indicators.floop import calculate_atr, calculate_range_filter
    tf_trends = {}
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        try:
            df_tf_copy = df_tf.copy()
            df_tf_copy['time'] = pd.to_datetime(df_tf_copy['time'])
            df_tf_copy.set_index('time', inplace=True)
            
            atr_floop = calculate_atr(df_tf_copy, 14)
            _, trend_floop, _ = calculate_range_filter(df_tf_copy['Close'], atr_floop, sensitivity=6, atr_multiplier=0.8)
            tf_trends[tf_name] = pd.Series(trend_floop, index=df_tf_copy.index)
        except Exception as e:
            print(f"Error calculating RF trend for TF {tf_name}: {e}")
            tf_trends[tf_name] = None
            
    # Pre-calculate KNN and Volume Profile data for each timeframe to use for setup features
    print("Pre-calculating KNN and Volume Profile features for live scanner...")
    tf_knn_data = {}
    tf_vp_data = {}
    for tf_name, df_tf in timeframes_data.items():
        # KNN
        try:
            pc1, pc2, pc3, pc4, target_clean = run_knn_classifier(
                df_tf,
                atr_period=10, factor=2.0,
                k_neighbors=10, sampling_window_size=1000, momentum_window=10,
                normalizing_window_size=1000,
                lazy=True
            )
            t_last = len(df_tf) - 1
            knn_up, knn_down = calculate_knn_probability_at_bar(
                t_last, pc1.values, pc2.values, pc3.values, pc4.values, target_clean.values,
                k=10, sampling_window=1000, stride=10
            )
            tf_knn_data[tf_name] = (knn_up, knn_down)
        except Exception as e:
            print(f"Error computing live KNN for TF {tf_name}: {e}")
            tf_knn_data[tf_name] = (0.0, 0.0)
            
        # Volume profile
        try:
            clusters_data = calculate_volume_clusters(
                df_tf, lookback=200, k=5, iterations=20, rows=20
            )
            tf_vp_data[tf_name] = clusters_data
        except Exception as e:
            print(f"Error computing live Volume Clusters for {tf_name}: {e}")
            tf_vp_data[tf_name] = {}
            
    # 3. Extract active HTF FVGs for hierarchy prioritization
    active_fvgs_by_tf = {}
    for tf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
        active_fvgs_by_tf[tf_name] = extract_active_htf_fvgs(timeframes_data[tf_name])
        
    def get_strategy_name(option_name: str) -> str:
        if "OB" in option_name:
            return "OB"
        elif "BPR" in option_name:
            return "BPR"
        elif "IC" in option_name:
            return "IC"
        elif "Swap" in option_name:
            return "Swapzone"
        elif "Breaker" in option_name:
            return "Breaker"
        elif "SND" in option_name:
            return "SND"
        elif "Pivot" in option_name:
            return "Pivot"
        else:
            return "FVG"
 
    # Extract setups (FVG, OB, BPR, IC, SND, Pivot)
    all_setups = []
    for tf_name in ['D1', 'H4', 'H1', 'M30', 'M15']:
        tf_setups = get_active_setups(timeframes_data[tf_name], symbol=symbol, tf_trends=tf_trends, df_d1=d1_pivot_source)
        for s in tf_setups:
            s['timeframe'] = tf_name
            s['strategy'] = get_strategy_name(s['option_name'])
            if s['strategy'] in ['FVG', 'OB', 'BPR', 'IC', 'SND', 'Pivot', 'Swapzone', 'Breaker']:
                all_setups.append(s)
            
    # 4. Multi-Timeframe Alignment, Suppression, and Rejection Checks
    tf_weights = {'M15': 1, 'M30': 2, 'H1': 3, 'H4': 4, 'D1': 5}
    tf_minutes_map = {'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    
    for setup in all_setups:
        setup['htf_prioritized'] = False
        setup['matching_htf_fvgs'] = []
        setup['suppressed'] = False
        setup['htf_conflict_reason'] = ""
        setup_tf = setup['timeframe']
        
        for htf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
            if tf_weights[htf_name] > tf_weights[setup_tf]:
                # HTF Prioritization (same direction)
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_same = (setup['direction'] == 1 and htf_fvg['type'] == 'BULLISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BEARISH')
                    if is_same:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['htf_prioritized'] = True
                            fvg_info = htf_fvg.copy()
                            fvg_info['timeframe'] = htf_name
                            setup['matching_htf_fvgs'].append(fvg_info)
                            
                # Conflict Suppression (opposite direction) - only if entry is inside the opposite HTF FVG
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_opp = (setup['direction'] == 1 and htf_fvg['type'] == 'BEARISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BULLISH')
                    if is_opp:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['suppressed'] = True
                            setup['htf_conflict_reason'] = f"Entry inside opposite active {htf_name} FVG"
                            break
                        
        # Check Rejection on lower timeframes (M15, M5, M1) for the setup
        rej_confirmed = False
        rej_tf_source = "None"
        
        # 1. Check on M5 (lookback 30 candles)
        m5_df = timeframes_data.get('M5')
        if m5_df is not None and not m5_df.empty:
            if detect_rejection_at_level(m5_df, setup['entry_price'], setup['direction'], lookback=30):
                rej_confirmed = True
                rej_tf_source = "M5"
                
        # 2. Check on M1 (lookback 90 candles) if not already confirmed on M5
        if not rej_confirmed:
            m1_df = timeframes_data.get('M1')
            if m1_df is not None and not m1_df.empty:
                if detect_rejection_at_level(m1_df, setup['entry_price'], setup['direction'], lookback=90):
                    rej_confirmed = True
                    rej_tf_source = "M1"
                    
        # 3. Fallback to M15 (lookback 15 candles) if not confirmed on M5/M1 and setup is on timeframe higher than M15
        if not rej_confirmed and setup_tf != 'M15':
            m15_df = timeframes_data.get('M15')
            if m15_df is not None and not m15_df.empty:
                if detect_rejection_at_level(m15_df, setup['entry_price'], setup['direction'], lookback=15):
                    rej_confirmed = True
                    rej_tf_source = "M15"
                    
        setup['rejection_confirmed'] = rej_confirmed
        setup['rejection_source'] = rej_tf_source

    # 4.3. Calculate ML Probability for all setups
    tf_minutes_map = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    for setup in all_setups:
        if setup.get('suppressed', False):
            setup['probability'] = 0.0
            continue
            
        tf = setup['timeframe']
        knn_up_tf, knn_down_tf = tf_knn_data.get(tf, (0.0, 0.0))
        knn_prob_sig = knn_up_tf if setup['direction'] == 1 else knn_down_tf
        knn_prob_opp = knn_down_tf if setup['direction'] == 1 else knn_up_tf
        
        clusters_data_tf = tf_vp_data.get(tf, {})
        dist_entry_to_poc = 0.0
        dist_entry_to_nearest_poc = 0.0
        if clusters_data_tf and 'current_poc' in clusters_data_tf:
            curr_poc = clusters_data_tf['current_poc']
            entry = setup['entry_price']
            dist_entry_to_poc = (entry - curr_poc) / curr_poc if curr_poc > 0 else 0.0
            
            pocs = clusters_data_tf.get('pocs', [])
            if pocs:
                dist_entry_to_nearest_poc = min(abs(entry - poc) for poc in pocs) / entry
                
        features = {
            'timeframe': tf_minutes_map[setup['timeframe']],
            'hour': setup['hour'],
            'day_of_week': setup['day_of_week'],
            'setup_type': setup['setup_type'],
            'direction': setup['direction'],
            'entry_price': setup['entry_price'],
            'sl_price': setup['sl_price'],
            'tp_price': setup['tp_price'],
            'risk_pips': setup['risk_pips'],
            'atr_14': setup['atr_14'],
            'trend': setup['trend'],
            'relative_risk': setup['relative_risk'],
            'killzone': setup['killzone'],
            'fvg_width': setup['fvg_width'],
            'relative_fvg_width': setup['relative_fvg_width'],
            'near_psychological_level': setup['near_psychological_level'],
            'knn_prob_sig': knn_prob_sig,
            'knn_prob_opp': knn_prob_opp,
            'dist_entry_to_poc': dist_entry_to_poc,
            'dist_entry_to_nearest_poc': dist_entry_to_nearest_poc,
            'dist_entry_to_pp': setup.get('dist_entry_to_pp', 0.0),
            'dist_entry_to_nearest_pivot': setup.get('dist_entry_to_nearest_pivot', 0.0),
            'floop_signal': setup['floop_signal'],
            'floop_strength': setup['floop_strength'],
            'floop_trend': setup.get('floop_trend', 0),
            'floop_trend_aligned': 1 if setup.get('floop_trend', 0) == setup['direction'] else 0
        }
        
        try:
            prob = predict_setup_probability(features)
        except Exception as e:
            print(f"Error predicting probability for {tf} {setup['strategy']}: {e}")
            prob = 0.5
            
        setup['probability'] = prob
        setup['features'] = features

    # 4.5. SMC Setup Confluence Clustering & Deduplication
    allowed_tfs = ['M15', 'M30', 'H1', 'H4']
    execution_groups = {}  # key: (timeframe, index, strategy), value: list of setups
    other_setups = []
    
    for setup in all_setups:
        if setup.get('suppressed', False):
            continue
            
        tf = setup['timeframe']
        strat = setup['strategy']
        if tf in allowed_tfs:
            key = (tf, setup['index'], strat)
            if key not in execution_groups:
                execution_groups[key] = []
            execution_groups[key].append(setup)
        else:
            other_setups.append(setup)
            
    # Build candidates (combining dual options)
    candidates = []
    for key, setups_list in execution_groups.items():
        tf, idx, strat = key
        
        # Check if it's a dual strategy with both options present
        opt_a = None
        opt_b = None
        for s in setups_list:
            if "Option A" in s['option_name'] or "Midpoint" in s['option_name'] or "0.5" in s['option_name']:
                opt_a = s
            elif "Option B" in s['option_name'] or "Golden Pocket" in s['option_name'] or "0.618" in s['option_name']:
                opt_b = s
                
        if opt_a is not None and opt_b is not None:
            max_prob = max(opt_a['probability'], opt_b['probability'])
            candidates.append({
                'id': f"DUAL_{tf}_{strat}_{idx}",
                'timeframe': tf,
                'strategy': strat,
                'direction': opt_a['direction'],
                'opt_a': opt_a,
                'opt_b': opt_b,
                'prob_a': opt_a['probability'],
                'prob_b': opt_b['probability'],
                'max_prob': max_prob,
                'entry_price': opt_a['entry_price'],  # Anchoring on Midpoint entry
                'is_dual': True
            })
        else:
            for s in setups_list:
                candidates.append({
                    'id': f"SINGLE_{tf}_{strat}_{idx}_{s['entry_price']:.3f}",
                    'timeframe': tf,
                    'strategy': strat,
                    'direction': s['direction'],
                    'opt': s,
                    'max_prob': s['probability'],
                    'entry_price': s['entry_price'],
                    'is_dual': False
                })
                
    # Sort ALL candidates by max_prob descending
    candidates.sort(key=lambda x: -x['max_prob'])
    
    clusters = []
    processed_ids = set()
    
    for c in candidates:
        if c['id'] in processed_ids:
            continue
            
        cluster = {
            'lead': c,
            'members': [c]
        }
        processed_ids.add(c['id'])
        
        # Look for other candidates in the same price zone
        for other_c in candidates:
            if other_c['id'] in processed_ids:
                continue
                
            if other_c['direction'] == c['direction']:
                # Proximity tolerance: 1.5 USD for Gold (15 pips)
                if abs(other_c['entry_price'] - c['entry_price']) <= 1.5:
                    cluster['members'].append(other_c)
                    processed_ids.add(other_c['id'])
                    
        clusters.append(cluster)
        
    # Mark non-lead members as suppressed
    for cluster in clusters:
        lead = cluster['lead']
        for member in cluster['members']:
            if member['id'] == lead['id']:
                continue
            # Mark its internal setups as suppressed
            if member['is_dual']:
                member['opt_a']['suppressed'] = True
                member['opt_a']['htf_conflict_reason'] = f"Clustered into Lead Setup {lead['id']}"
                member['opt_b']['suppressed'] = True
                member['opt_b']['htf_conflict_reason'] = f"Clustered into Lead Setup {lead['id']}"
            else:
                member['opt']['suppressed'] = True
                member['opt']['htf_conflict_reason'] = f"Clustered into Lead Setup {lead['id']}"
                
    # 5. Model Inference, Clustering Notification & Dispatch
    sent_signals = load_sent_signals()
    signals_sent_this_cycle = 0
    active_high_confidence = []
    registry_changed = False
    
    for cluster in clusters:
        lead = cluster['lead']
        tf = lead['timeframe']
        strat = lead['strategy']
        dir_name = "BULL" if lead['direction'] == 1 else "BEAR"
        
        # Build confluences/reasons list for this cluster
        reasons = []
        
        # 1. Structure & ML Probs
        for member in cluster['members']:
            m_tf = member['timeframe']
            m_strat = member['strategy']
            m_prob = member['max_prob']
            
            strat_desc = get_strategy_display_name(m_strat)
            reasons.append(f"{strat_desc} {m_tf} (Model confidence: {m_prob:.1%})")
            
            # Check rejection source of the member
            if member['is_dual']:
                rej_src_a = member['opt_a'].get('rejection_source', 'None')
                rej_src_b = member['opt_b'].get('rejection_source', 'None')
                if rej_src_a != 'None':
                    reasons.append(f"Wick Rejection touch confirmed on {rej_src_a}")
                if rej_src_b != 'None':
                    reasons.append(f"Wick Rejection touch confirmed on {rej_src_b}")
            else:
                rej_src = member['opt'].get('rejection_source', 'None')
                if rej_src != 'None':
                    reasons.append(f"Wick Rejection touch confirmed on {rej_src}")
                    
        # 2. HTF Priority, Psych levels, FLOOP, and Volume POC (from Lead candidate)
        lead_opt = lead['opt_a'] if lead['is_dual'] else lead['opt']
        
        if lead_opt.get('htf_prioritized', False):
            reasons.append("Aligned inside active Higher Timeframe (HTF) structure")
            
        if lead_opt.get('near_psychological_level', 0) == 1:
            reasons.append("Entry zone near Psychological Round Level (ends in 0 or 5)")
            
        if lead_opt.get('floop_trend', 0) == lead_opt['direction']:
            reasons.append("Supported by FLOOP Pro Trend Filter")
            
        # Volume profile check
        dist_poc = lead_opt.get('dist_entry_to_poc', 0.0)
        if abs(dist_poc) <= 0.005 and dist_poc != 0.0:
            reasons.append("Zone overlaps with high-volume POC cluster")
            
        # Filter duplicates
        unique_reasons = []
        seen_reasons = set()
        for r in reasons:
            if r not in seen_reasons:
                seen_reasons.add(r)
                unique_reasons.append(r)
                
        is_high_conf = lead['max_prob'] >= confidence_threshold
        
        if is_high_conf:
            if lead['is_dual']:
                opt_a = lead['opt_a']
                opt_b = lead['opt_b']
                prob_a = lead['prob_a']
                prob_b = lead['prob_b']
                
                # 1. Apply FVG Quality Filters (only for FVG strategy)
                if strat == 'FVG':
                    is_good, quality_reason = is_good_fvg(timeframes_data[tf], opt_a['index'], opt_a, symbol, timeframes_data)
                    if not is_good:
                        print(f"[Quality Filter] {tf} FVG at index {opt_a['index']} rejected: {quality_reason}")
                        continue
                        
                setup_time_str = str(opt_a['time'])
                sig_key = f"{tf}_{strat}_DUAL_{dir_name}_{opt_a['entry_price']:.3f}_{opt_b['entry_price']:.3f}_{setup_time_str.replace(' ', '_')}"
                
                opt_a['status'] = "HIGH CONFIDENCE SIGNAL"
                opt_b['status'] = "HIGH CONFIDENCE SIGNAL"
                
                allow_entry = True
                reentries_count = 0

                if sig_key in sent_signals and should_promote_low_confidence_record(
                    sent_signals[sig_key],
                    ("ticket_a", "ticket_b"),
                ):
                    promoted_record = sent_signals.pop(sig_key)
                    reentries_count = promoted_record.get('reentries_count', 0)
                    print(f"[Scanner Registry] Promoting low confidence dual {tf} {strat} at {opt_a['time']} to live execution.")
                
                if sig_key in sent_signals:
                    sig_data = sent_signals[sig_key]
                    reentries_count = sig_data.get('reentries_count', 0)
                    
                    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").lower() == "true"
                    if execute_enabled:
                        magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
                        from src.execution import get_active_broker_symbol
                        broker_symbol = get_active_broker_symbol(symbol)
                        tick = mt5.symbol_info_tick(broker_symbol)
                        current_price = None
                        if tick is not None:
                            current_price = tick.ask if opt_a['direction'] == 1 else tick.bid
                        
                        def is_ticket_active(t):
                            if t is None:
                                return False
                            orders_act = mt5.orders_get(ticket=t)
                            if orders_act and len(orders_act) > 0:
                                return True
                            positions_act = mt5.positions_get(ticket=t)
                            if positions_act and len(positions_act) > 0:
                                return True
                            return False
                            
                        def has_history_deals(t):
                            if t is None:
                                return False
                            deals = mt5.history_deals_get(position=t)
                            return deals is not None and len(deals) > 0
                            
                        ticket_a = sig_data.get('ticket_a')
                        ticket_b = sig_data.get('ticket_b')
                        
                        if ticket_a is not None and not is_ticket_active(ticket_a) and not has_history_deals(ticket_a) and not sig_data.get('outcome_a_recorded', False) and not sig_data.get('outcome_recorded', False):
                            print(f"[Recovery Engine] Option A (0.5) Ticket #{ticket_a} is inactive. Re-placing...")
                            recovery_mode_a = choose_dual_recovery_execution_mode(opt_a, opt_b, current_price, option="a")
                            if recovery_mode_a == "market":
                                new_ticket_a, exec_msg_a = execute_market_order_for_setup(opt_a, symbol)
                            elif recovery_mode_a == "skip":
                                new_ticket_a, exec_msg_a = None, "Skipped (Option B market recovery active)"
                            else:
                                new_ticket_a, exec_msg_a = execute_trade_for_setup(opt_a, symbol)
                            if new_ticket_a is not None:
                                sig_data['ticket_a'] = new_ticket_a
                                sig_data['outcome_a_recorded'] = False
                                registry_changed = True
                                try:
                                    recovery_title = "Market Order Recovery Executed" if recovery_mode_a == "market" else "Pending Order Re-placed"
                                    recovery_msg_a = (
                                        f"🔄 <b>[Order Recovery] {recovery_title}</b> 🔄\n\n"
                                        f"Setup 0.5 yang sebelumnya inactive telah dieksekusi ulang.\n"
                                        f"• <b>Price:</b> {opt_a['entry_price']:.3f}\n"
                                        f"• <b>Execution:</b> {exec_msg_a}\n"
                                        f"• <b>New Ticket:</b> #{new_ticket_a}"
                                    )
                                    send_recovery_alert_with_chart(
                                        recovery_msg_a,
                                        timeframes_data=timeframes_data,
                                        timeframe=tf,
                                        symbol=symbol,
                                        direction_name=dir_name,
                                        strategy=strat,
                                        setups=[opt_a, opt_b],
                                        image_suffix="recovery_dual_a",
                                    )
                                except Exception:
                                    pass
                                    
                        if ticket_b is not None and not is_ticket_active(ticket_b) and not has_history_deals(ticket_b) and not sig_data.get('outcome_b_recorded', False) and not sig_data.get('outcome_recorded', False):
                            print(f"[Recovery Engine] Option B (0.618) Ticket #{ticket_b} is inactive. Re-placing...")
                            recovery_mode_b = choose_dual_recovery_execution_mode(opt_a, opt_b, current_price, option="b")
                            if recovery_mode_b == "market":
                                new_ticket_b, exec_msg_b = execute_market_order_for_setup(opt_b, symbol)
                            elif recovery_mode_b == "skip":
                                new_ticket_b, exec_msg_b = None, "Skipped (Option A market recovery active)"
                            else:
                                new_ticket_b, exec_msg_b = execute_trade_for_setup(opt_b, symbol)
                            if new_ticket_b is not None:
                                sig_data['ticket_b'] = new_ticket_b
                                sig_data['outcome_b_recorded'] = False
                                registry_changed = True
                                try:
                                    recovery_title = "Market Order Recovery Executed" if recovery_mode_b == "market" else "Pending Order Re-placed"
                                    recovery_msg_b = (
                                        f"🔄 <b>[Order Recovery] {recovery_title}</b> 🔄\n\n"
                                        f"Setup 0.618 yang sebelumnya inactive telah dieksekusi ulang.\n"
                                        f"• <b>Price:</b> {opt_b['entry_price']:.3f}\n"
                                        f"• <b>Execution:</b> {exec_msg_b}\n"
                                        f"• <b>New Ticket:</b> #{new_ticket_b}"
                                    )
                                    send_recovery_alert_with_chart(
                                        recovery_msg_b,
                                        timeframes_data=timeframes_data,
                                        timeframe=tf,
                                        symbol=symbol,
                                        direction_name=dir_name,
                                        strategy=strat,
                                        setups=[opt_a, opt_b],
                                        image_suffix="recovery_dual_b",
                                    )
                                except Exception:
                                    pass
                                    
                    # Mark as active to protect from pruning, then continue
                    active_high_confidence.append(opt_a)
                    active_high_confidence.append(opt_b)
                    continue
                    
                # Auto-execute trades on MT5 (market execution if price inside setup entry zone and rejection confirmed, else limit orders)
                ticket_a, ticket_b = None, None
                exec_msg_a, exec_msg_b = "", ""
                
                from src.execution import get_active_broker_symbol
                broker_symbol = get_active_broker_symbol(symbol)
                tick = mt5.symbol_info_tick(broker_symbol)
                current_price = None
                if tick is not None:
                    current_price = tick.ask if opt_a['direction'] == 1 else tick.bid
                    
                is_market_entry = False
                market_option = choose_dual_market_entry_option(opt_a, opt_b, current_price)
                if market_option == "a":
                    ticket_a, exec_msg_a = execute_market_order_for_setup(opt_a, symbol)
                    ticket_b, exec_msg_b = None, "Skipped (Option A Market Order placed)"
                    is_market_entry = True
                elif market_option == "b":
                    ticket_b, exec_msg_b = execute_market_order_for_setup(opt_b, symbol)
                    ticket_a, exec_msg_a = None, "Skipped (Option B Market Order placed)"
                    is_market_entry = True
                            
                if not is_market_entry:
                    ticket_a, exec_msg_a = execute_trade_for_setup(opt_a, symbol)
                    ticket_b, exec_msg_b = execute_trade_for_setup(opt_b, symbol)
                
                is_placed = (ticket_a is not None) or (ticket_b is not None)
                is_monitoring_only = "disabled" in exec_msg_a.lower() and "disabled" in exec_msg_b.lower()
                should_alert = is_placed or is_monitoring_only
                
                if not should_alert:
                    print(f"[Execution Engine] {tf} {strat} at index {opt_a['index']} skipped: {exec_msg_a} / {exec_msg_b}")
                    continue
                    
                active_high_confidence.append(opt_a)
                active_high_confidence.append(opt_b)
                
                print(f"[New Cluster Signal Triggered] {tf} Dual {strat} at {opt_a['time']} | Direction: {dir_name} | Win Probs: 0.5={prob_a:.2%}, 0.618={prob_b:.2%}")
                
                # Generate chart
                tf_df = timeframes_data[tf]
                tf_setups = [opt_a, opt_b]
                title = f"{symbol} {tf} - {dir_name} Dual {strat} Confluence"
                image_filename = f"temp_alert_{tf}_dual_{opt_a['index']}.png"
                
                try:
                    plot_smc_chart(tf_df, title=title, active_setups=tf_setups, output_filename=image_filename)
                except Exception as e:
                    print(f"Failed to generate chart image: {e}")
                    image_filename = None
                    
                # Format Telegram message using HTML
                rej_src_a = opt_a.get('rejection_source', 'None')
                rej_src_b = opt_b.get('rejection_source', 'None')
                rej_status = format_rejection_status([rej_src_a, rej_src_b])
                htf_prior_status = format_htf_priority_status(opt_a['htf_prioritized'] or opt_b['htf_prioritized'])
                exec_status_a = format_execution_status(
                    ticket_a,
                    exec_msg_a,
                    skipped_peer="0.618" if "Skipped" in exec_msg_a else None,
                    monitoring_only=is_monitoring_only,
                )
                exec_status_b = format_execution_status(
                    ticket_b,
                    exec_msg_b,
                    skipped_peer="0.500" if "Skipped" in exec_msg_b else None,
                    monitoring_only=is_monitoring_only,
                )

                # De-duplicate matching HTF structures for the Telegram formatter.
                matching_fvgs = opt_a['matching_htf_fvgs'] + opt_b['matching_htf_fvgs']
                seen_fvgs = set()
                unique_matching = []
                for f in matching_fvgs:
                    f_key = (f['timeframe'], f['bottom'], f['top'])
                    if f_key not in seen_fvgs:
                        seen_fvgs.add(f_key)
                        unique_matching.append(f)
                
                setup_desc = f"{get_strategy_display_name(strat)} (Dual Fibonacci Entry)"
                msg = format_dual_signal_message(
                    symbol=symbol,
                    timeframe=tf,
                    direction=opt_a['direction'],
                    setup_desc=setup_desc,
                    probability_a=prob_a,
                    probability_b=prob_b,
                    confidence_threshold=confidence_threshold,
                    opt_a=opt_a,
                    opt_b=opt_b,
                    execution_status_a=exec_status_a,
                    execution_status_b=exec_status_b,
                    htf_priority_status=htf_prior_status,
                    rejection_status=rej_status,
                    confluences=unique_reasons,
                    htf_matches=unique_matching,
                )

                success = send_telegram_alert(msg, image_filename)
                
                if image_filename and os.path.exists(image_filename):
                    try:
                        os.remove(image_filename)
                    except Exception:
                        pass
                        
                if success or is_placed:
                    sent_signals[sig_key] = {
                        'time_sent': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timeframe': tf,
                        'direction': dir_name,
                        'type': strat,
                        'price_0.5': opt_a['entry_price'],
                        'price_0.618': opt_b['entry_price'],
                        'probability_0.5': prob_a,
                        'probability_0.618': prob_b,
                        'ticket_a': ticket_a,
                        'ticket_b': ticket_b,
                        'reentries_count': reentries_count + 1,
                        'features_0.5': opt_a['features'],
                        'features_0.618': opt_b['features']
                    }
                    signals_sent_this_cycle += 1
                    registry_changed = True
                    
            else:
                opt = lead['opt']
                prob = lead['max_prob']
                
                setup_time_str = str(opt['time'])
                sig_key = f"{tf}_{strat}_SINGLE_{dir_name}_{opt['entry_price']:.3f}_{setup_time_str.replace(' ', '_')}"
                
                opt['status'] = "HIGH CONFIDENCE SIGNAL"
                
                allow_entry = True
                reentries_count = 0

                if sig_key in sent_signals and should_promote_low_confidence_record(
                    sent_signals[sig_key],
                    ("ticket_id",),
                ):
                    promoted_record = sent_signals.pop(sig_key)
                    reentries_count = promoted_record.get('reentries_count', 0)
                    print(f"[Scanner Registry] Promoting low confidence single {tf} {strat} at {opt['time']} to live execution.")
                
                if sig_key in sent_signals:
                    sig_data = sent_signals[sig_key]
                    reentries_count = sig_data.get('reentries_count', 0)
                    
                    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").lower() == "true"
                    if execute_enabled:
                        from src.execution import get_active_broker_symbol
                        broker_symbol = get_active_broker_symbol(symbol)
                        tick = mt5.symbol_info_tick(broker_symbol)
                        current_price = None
                        if tick is not None:
                            current_price = tick.ask if opt['direction'] == 1 else tick.bid

                        def is_ticket_active(t):
                            if t is None:
                                return False
                            orders_act = mt5.orders_get(ticket=t)
                            if orders_act and len(orders_act) > 0:
                                return True
                            positions_act = mt5.positions_get(ticket=t)
                            if positions_act and len(positions_act) > 0:
                                return True
                            return False
                            
                        def has_history_deals(t):
                            if t is None:
                                return False
                            deals = mt5.history_deals_get(position=t)
                            return deals is not None and len(deals) > 0
                            
                        ticket_id = sig_data.get('ticket_id')
                        if ticket_id is not None and not is_ticket_active(ticket_id) and not has_history_deals(ticket_id) and not sig_data.get('outcome_recorded', False):
                            print(f"[Recovery Engine] Single Ticket #{ticket_id} is inactive. Re-placing...")
                            recovery_mode = choose_recovery_execution_mode(opt, current_price)
                            if recovery_mode == "market":
                                new_ticket, exec_msg = execute_market_order_for_setup(opt, symbol)
                            else:
                                new_ticket, exec_msg = execute_trade_for_setup(opt, symbol)
                            if new_ticket is not None:
                                sig_data['ticket_id'] = new_ticket
                                sig_data['outcome_recorded'] = False
                                registry_changed = True
                                try:
                                    recovery_title = "Market Order Recovery Executed" if recovery_mode == "market" else "Pending Order Re-placed"
                                    recovery_msg = (
                                        f"🔄 <b>[Order Recovery] {recovery_title}</b> 🔄\n\n"
                                        f"Setup yang sebelumnya inactive telah dieksekusi ulang.\n"
                                        f"• <b>Price:</b> {opt['entry_price']:.3f}\n"
                                        f"• <b>Execution:</b> {exec_msg}\n"
                                        f"• <b>New Ticket:</b> #{new_ticket}"
                                    )
                                    send_recovery_alert_with_chart(
                                        recovery_msg,
                                        timeframes_data=timeframes_data,
                                        timeframe=tf,
                                        symbol=symbol,
                                        direction_name=dir_name,
                                        strategy=strat,
                                        setups=[opt],
                                        image_suffix="recovery_single",
                                    )
                                except Exception:
                                    pass
                                    
                    # Mark as active to protect from pruning, then continue
                    active_high_confidence.append(opt)
                    continue
                    
                # Auto-execute single trade on MT5 (market execution if price inside setup entry zone and rejection confirmed, else limit orders)
                ticket_id = None
                exec_msg = ""
                is_market_entry = False
                
                from src.execution import get_active_broker_symbol
                broker_symbol = get_active_broker_symbol(symbol)
                tick = mt5.symbol_info_tick(broker_symbol)
                current_price = None
                if tick is not None:
                    current_price = tick.ask if opt['direction'] == 1 else tick.bid
                    
                if should_market_enter_setup(opt, current_price):
                    ticket_id, exec_msg = execute_market_order_for_setup(opt, symbol)
                    is_market_entry = True
                            
                if not is_market_entry:
                    ticket_id, exec_msg = execute_trade_for_setup(opt, symbol)
                    
                is_placed = ticket_id is not None
                is_monitoring_only = "disabled" in exec_msg.lower()
                should_alert = is_placed or is_monitoring_only
                
                if not should_alert:
                    print(f"[Execution Engine] Single {tf} {strat} at index {opt['index']} skipped: {exec_msg}")
                    continue
                    
                active_high_confidence.append(opt)
                print(f"[New Cluster Signal Triggered] Single {tf} {strat} at {opt['time']} | Direction: {dir_name} | Win Prob: {prob:.2%}")
                
                # Generate chart
                tf_df = timeframes_data[tf]
                tf_setups = [opt]
                title = f"{symbol} {tf} - {dir_name} Single {strat} Confluence"
                image_filename = f"temp_alert_{tf}_single_{opt['index']}.png"
                
                try:
                    plot_smc_chart(tf_df, title=title, active_setups=tf_setups, output_filename=image_filename)
                except Exception as e:
                    print(f"Failed to generate chart image: {e}")
                    image_filename = None
                    
                # Format Telegram message using HTML
                rej_src = opt.get('rejection_source', 'None')
                rej_status = format_rejection_status([rej_src])
                htf_prior_status = format_htf_priority_status(opt['htf_prioritized'])
                exec_status = format_execution_status(
                    ticket_id,
                    exec_msg,
                    monitoring_only=is_monitoring_only,
                )

                matching_fvgs = opt['matching_htf_fvgs']
                setup_desc = opt.get('option_name') or get_strategy_display_name(strat)
                msg = format_single_signal_message(
                    symbol=symbol,
                    timeframe=tf,
                    direction=opt['direction'],
                    setup_desc=setup_desc,
                    probability=prob,
                    confidence_threshold=confidence_threshold,
                    setup=opt,
                    execution_status=exec_status,
                    htf_priority_status=htf_prior_status,
                    rejection_status=rej_status,
                    confluences=unique_reasons,
                    htf_matches=matching_fvgs,
                )
                
                success = send_telegram_alert(msg, image_filename)
                
                if image_filename and os.path.exists(image_filename):
                    try:
                        os.remove(image_filename)
                    except Exception:
                        pass
                        
                if success or is_placed:
                    sent_signals[sig_key] = {
                        'time_sent': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timeframe': tf,
                        'direction': dir_name,
                        'type': strat,
                        'price': opt['entry_price'],
                        'probability': prob,
                        'ticket_id': ticket_id,
                        'reentries_count': reentries_count + 1,
                        'features': opt['features']
                    }
                    signals_sent_this_cycle += 1
                    registry_changed = True
        else:
            # Low confidence cluster - Silent Registration for Lead Setup
            if register_low_confidence_lead(
                lead=lead,
                sent_signals=sent_signals,
                symbol=symbol,
                timeframe=tf,
                strategy=strat,
                direction_name=dir_name,
                accept_threshold=confidence_threshold,
            ):
                registry_changed = True
    for setup in other_setups:
        tf = setup['timeframe']
        knn_up_tf, knn_down_tf = tf_knn_data.get(tf, (0.0, 0.0))
        knn_prob_sig = knn_up_tf if setup['direction'] == 1 else knn_down_tf
        knn_prob_opp = knn_down_tf if setup['direction'] == 1 else knn_up_tf
        
        clusters_data_tf = tf_vp_data.get(tf, {})
        dist_entry_to_poc = 0.0
        dist_entry_to_nearest_poc = 0.0
        if clusters_data_tf and 'current_poc' in clusters_data_tf:
            curr_poc = clusters_data_tf['current_poc']
            entry = setup['entry_price']
            dist_entry_to_poc = (entry - curr_poc) / curr_poc if curr_poc > 0 else 0.0
            
            pocs = clusters_data_tf.get('pocs', [])
            if pocs:
                dist_entry_to_nearest_poc = min(abs(entry - poc) for poc in pocs) / entry

        features = {
            'timeframe': tf_minutes_map[setup['timeframe']],
            'hour': setup['hour'],
            'day_of_week': setup['day_of_week'],
            'setup_type': setup['setup_type'],
            'direction': setup['direction'],
            'entry_price': setup['entry_price'],
            'sl_price': setup['sl_price'],
            'tp_price': setup['tp_price'],
            'risk_pips': setup['risk_pips'],
            'atr_14': setup['atr_14'],
            'trend': setup['trend'],
            'relative_risk': setup['relative_risk'],
            'killzone': setup['killzone'],
            'fvg_width': setup['fvg_width'],
            'relative_fvg_width': setup['relative_fvg_width'],
            'near_psychological_level': setup['near_psychological_level'],
            'knn_prob_sig': knn_prob_sig,
            'knn_prob_opp': knn_prob_opp,
            'dist_entry_to_poc': dist_entry_to_poc,
            'dist_entry_to_nearest_poc': dist_entry_to_nearest_poc,
            'dist_entry_to_pp': setup.get('dist_entry_to_pp', 0.0),
            'dist_entry_to_nearest_pivot': setup.get('dist_entry_to_nearest_pivot', 0.0),
            'floop_signal': setup['floop_signal'],
            'floop_strength': setup['floop_strength'],
            'floop_trend': setup.get('floop_trend', 0),
            'floop_trend_aligned': 1 if setup.get('floop_trend', 0) == setup['direction'] else 0
        }
        try:
            prob = predict_setup_probability(features)
        except Exception:
            prob = 0.5
        setup['probability'] = prob
        setup['status'] = "HIGH CONFIDENCE SIGNAL" if prob >= confidence_threshold else "FILTERED (LTF/Other)"
                
    if registry_changed:
        save_sent_signals(sent_signals)
        
    if signals_sent_this_cycle > 0:
        print(f"Sent {signals_sent_this_cycle} new alerts this cycle.")
    else:
        print("No new high confidence trade signals triggered this cycle.")
        
    # 8. Clean up invalid/old pending orders from MT5 and manage active positions
    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").lower() == "true"
    if execute_enabled:
        try:
            magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
            prune_invalid_pending_orders(symbol, magic, active_high_confidence)
            manage_active_trades(symbol, magic, timeframes_data)
        except Exception as e:
            print(f"[Scanner Error] Error during pending orders pruning / active trade management: {e}")
            
    # Free MT5 connection at the very end of the cycle
    import MetaTrader5 as mt5
    mt5.shutdown()
    print("--- Scan Cycle Finished ---")

def main():
    parser = argparse.ArgumentParser(description="Forex SMC Scanner background worker with Telegram Alerts.")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol (default: XAUUSD)")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold to alert/order (default: ML_ACCEPT_THRESHOLD or 0.50)")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=5, help="Scan interval in minutes (default: 5)")
    
    args = parser.parse_args()
    confidence_threshold = get_accept_threshold(args.threshold)
    
    # Verify environment file variables exist
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id or token.startswith("YOUR_") or chat_id.startswith("YOUR_"):
        print("\n[WARNING] Telegram credentials are not configured in your .env file.")
        print("Alerts will print in the console but will NOT be sent to Telegram.")
        print("Please configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable alerts.\n")
        
    if args.loop:
        print(f"Starting background worker loop. Scanning every {args.interval} minutes...")
        try:
            while True:
                run_scan(args.symbol, confidence_threshold)
                print(f"Sleeping for {args.interval} minutes...")
                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\nScanner stopped by user.")
    else:
        # Run once
        run_scan(args.symbol, confidence_threshold)

if __name__ == "__main__":
    main()
