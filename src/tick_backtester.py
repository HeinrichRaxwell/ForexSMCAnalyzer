import pandas as pd
import numpy as np


def _setup_time(setup: dict):
    return pd.Timestamp(setup.get("time"))


def _trade_profit(trade: dict, exit_price: float, contract_size: float) -> float:
    entry = float(trade["entry"])
    direction = int(trade["direction"])
    lot_size = float(trade.get("lot_size", 0.01))
    return (float(exit_price) - entry) * direction * lot_size * contract_size


def _resolve_trade_on_tick(trade: dict, tick_time, bid: float, ask: float):
    direction = int(trade["direction"])
    entry = float(trade["entry"])
    sl = float(trade["sl"])
    tp = float(trade["tp"])

    if not trade["triggered"]:
        if direction == 1:
            if ask <= entry:
                trade["triggered"] = True
                trade["entry_time"] = tick_time
            elif bid >= tp or bid <= sl:
                return "MISSED", None
        else:
            if bid >= entry:
                trade["triggered"] = True
                trade["entry_time"] = tick_time
            elif ask <= tp or ask >= sl:
                return "MISSED", None

    if trade["triggered"]:
        if direction == 1:
            if bid <= sl:
                return "LOSS", sl
            if bid >= tp:
                return "WIN", tp
        else:
            if ask >= sl:
                return "LOSS", sl
            if ask <= tp:
                return "WIN", tp

    return None, None


def _process_tick_arrays(
    tick_times,
    bid_values,
    ask_values,
    balance,
    peak_balance,
    max_drawdown_usd,
    max_drawdown_pct,
    wins,
    losses,
    missed,
    trade_history,
    active_trades,
    pending_setups,
    setup_times,
    next_setup_idx,
    max_concurrent,
    contract_size,
):
    """Inner loop: process numpy arrays of ticks and mutate simulation state.

    Returns an updated tuple of all mutable simulation fields.
    Designed to be called repeatedly (e.g., once per day when streaming).
    """
    tick_idx = 0
    n_ticks = len(tick_times)

    # Pre-convert setup times to numpy datetime64[ns] for fast zero-allocation comparisons
    setup_times_np = [pd.Timestamp(t).to_datetime64() for t in setup_times]

    while tick_idx < n_ticks:
        if balance <= 0:
            balance = 0.0
            break

        if not active_trades:
            if next_setup_idx >= len(pending_setups):
                break
            next_setup_time = setup_times_np[next_setup_idx]
            if tick_times[tick_idx] < next_setup_time:
                tick_idx = int(np.searchsorted(tick_times, next_setup_time, side="left"))
                if tick_idx >= n_ticks:
                    break

        tick_time = tick_times[tick_idx]
        bid = float(bid_values[tick_idx])
        ask = float(ask_values[tick_idx])

        while next_setup_idx < len(pending_setups) and setup_times_np[next_setup_idx] <= tick_time:
            setup = pending_setups[next_setup_idx]
            next_setup_idx += 1
            active_structure_count = len({trade["setup_idx"] for trade in active_trades})
            if active_structure_count >= max_concurrent:
                missed += 1
                continue
            active_trades.append({
                "setup_time": _setup_time(setup),
                "setup_idx": setup.get("index", next_setup_idx),
                "strategy": setup.get("strategy", ""),
                "option": setup.get("option_name", ""),
                "direction": int(setup["direction"]),
                "entry": float(setup["entry_price"]),
                "sl": float(setup["sl_price"]),
                "tp": float(setup["tp_price"]),
                "lot_size": float(setup.get("lot_size", 0.01)),
                "triggered": False,
                "entry_time": None,
            })

        resolved_trades = []
        for trade in active_trades:
            outcome, exit_price = _resolve_trade_on_tick(trade, tick_time, bid, ask)
            if outcome is None:
                continue
            resolved_trades.append((trade, outcome, exit_price))

        for trade, outcome, exit_price in resolved_trades:
            if trade in active_trades:
                active_trades.remove(trade)
            if outcome == "MISSED":
                missed += 1
                continue

            profit_usd = _trade_profit(trade, exit_price, contract_size)
            balance += profit_usd
            peak_balance = max(peak_balance, balance)
            drawdown_usd = peak_balance - balance
            drawdown_pct = (drawdown_usd / peak_balance) * 100 if peak_balance > 0 else 0.0
            max_drawdown_usd = max(max_drawdown_usd, drawdown_usd)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
            if outcome == "WIN":
                wins += 1
            else:
                losses += 1
            trade_history.append({
                "setup_time": trade["setup_time"],
                "entry_time": pd.Timestamp(trade["entry_time"]),
                "exit_time": pd.Timestamp(tick_time),
                "strategy": trade["strategy"],
                "option": trade["option"],
                "direction": "BUY" if trade["direction"] == 1 else "SELL",
                "entry": trade["entry"],
                "sl": trade["sl"],
                "tp": trade["tp"],
                "outcome": outcome,
                "profit_usd": profit_usd,
                "balance_after": balance,
            })
        tick_idx += 1

    return (
        balance, peak_balance, max_drawdown_usd, max_drawdown_pct,
        wins, losses, missed, trade_history, active_trades, next_setup_idx,
    )


def run_tick_simulation(
    ticks: pd.DataFrame,
    setups: list,
    starting_capital: float,
    contract_size: float = 100.0,
    max_concurrent: int = 1,
) -> dict:
    """Original in-memory simulation. Keeps all ticks in RAM. Use for unit tests
    or small date windows only. For long date ranges use run_tick_simulation_streaming."""
    balance = float(starting_capital)
    peak_balance = float(starting_capital)
    max_drawdown_usd = 0.0
    max_drawdown_pct = 0.0
    wins = 0
    losses = 0
    missed = 0
    trade_history = []
    active_trades = []

    if ticks is None or ticks.empty:
        return {
            "initial_balance": float(starting_capital),
            "final_balance": balance,
            "wins": 0,
            "losses": 0,
            "missed": len(setups),
            "total_resolved": 0,
            "winrate": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "blown": balance <= 0,
            "trade_history": [],
        }

    # Optimise: avoid redundant copy/sort when data is already clean and sorted.
    time_col = ticks["time"]
    if not pd.api.types.is_datetime64_any_dtype(time_col):
        time_col = pd.to_datetime(time_col, errors="coerce")

    is_sorted = time_col.is_monotonic_increasing
    has_nans = time_col.isna().any()

    if has_nans or not is_sorted:
        tick_frame = ticks.copy()
        tick_frame["time"] = time_col
        if has_nans:
            tick_frame = tick_frame.dropna(subset=["time"])
        if not is_sorted:
            tick_frame = tick_frame.sort_values("time", kind="mergesort")
        tick_times = tick_frame["time"].to_numpy(dtype="datetime64[ns]")
        bid_values = tick_frame["bid"].astype(float).to_numpy()
        ask_values = tick_frame["ask"].astype(float).to_numpy()
    else:
        tick_times = time_col.to_numpy(dtype="datetime64[ns]")
        bid_values = ticks["bid"].to_numpy(dtype=float)
        ask_values = ticks["ask"].to_numpy(dtype=float)

    pending_setups = sorted(setups, key=_setup_time)
    setup_times = [_setup_time(setup) for setup in pending_setups]
    next_setup_idx = 0

    (
        balance, peak_balance, max_drawdown_usd, max_drawdown_pct,
        wins, losses, missed, trade_history, active_trades, next_setup_idx,
    ) = _process_tick_arrays(
        tick_times, bid_values, ask_values,
        balance, peak_balance, max_drawdown_usd, max_drawdown_pct,
        wins, losses, missed, trade_history, active_trades,
        pending_setups, setup_times, next_setup_idx, max_concurrent, contract_size,
    )

    missed += len(active_trades)
    missed += len(pending_setups) - next_setup_idx
    winrate = (wins / (wins + losses)) * 100 if wins + losses else 0.0

    return {
        "initial_balance": float(starting_capital),
        "final_balance": max(0.0, balance),
        "wins": wins,
        "losses": losses,
        "missed": missed,
        "total_resolved": wins + losses,
        "winrate": winrate,
        "max_drawdown_usd": max_drawdown_usd,
        "max_drawdown_pct": max_drawdown_pct,
        "blown": balance <= 0,
        "trade_history": trade_history,
    }


def run_tick_simulation_streaming(
    setups: list,
    starting_capital: float,
    cache_dir,
    symbol: str,
    start,
    end,
    contract_size: float = 100.0,
    max_concurrent: int = 1,
    verbose: bool = False,
) -> dict:
    """Memory-efficient streaming simulation.

    Loads one day of ticks at a time from on-disk cache files instead of
    concatenating all ticks into one giant DataFrame.  For a 5-month window of
    XAUUSD data (~46 M ticks) this reduces peak RAM from >1 GB to <100 MB.

    Active trades carry over across day boundaries so multi-day H4/D1 setups
    resolve correctly.  Missing or empty cache days (86-byte stubs written when
    MT5 returned no data) are silently skipped.

    Returns the same dict structure as run_tick_simulation plus:
      _days_with_ticks: int  – calendar days that had tick data
      _days_missing:    int  – calendar days where no tick file existed / was empty
    """
    from src.tick_data import (
        CACHE_READ_ERRORS,
        iter_days,
        read_tick_cache,
        tick_cache_path,
    )

    balance = float(starting_capital)
    peak_balance = float(starting_capital)
    max_drawdown_usd = 0.0
    max_drawdown_pct = 0.0
    wins = 0
    losses = 0
    missed = 0
    trade_history = []
    active_trades = []

    pending_setups = sorted(setups, key=_setup_time)
    setup_times = [_setup_time(s) for s in pending_setups]
    next_setup_idx = 0

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    days_with_ticks = 0
    days_missing = 0

    for day in iter_days(start_ts, end_ts):
        # Fast exit when all setups are consumed and no trades are open.
        if next_setup_idx >= len(pending_setups) and not active_trades:
            break
        if balance <= 0:
            break

        path = tick_cache_path(cache_dir, symbol, day)
        try:
            day_ticks = read_tick_cache(path)
        except CACHE_READ_ERRORS:
            days_missing += 1
            continue

        if day_ticks.empty:
            days_missing += 1
            continue

        days_with_ticks += 1

        # Narrow to the exact requested time window.
        day_ticks = day_ticks[(day_ticks["time"] >= start_ts) & (day_ticks["time"] < end_ts)]
        if day_ticks.empty:
            continue

        time_col = day_ticks["time"]
        if not pd.api.types.is_datetime64_any_dtype(time_col):
            time_col = pd.to_datetime(time_col, errors="coerce")

        tick_times = time_col.to_numpy(dtype="datetime64[ns]")
        bid_values = day_ticks["bid"].to_numpy(dtype=float)
        ask_values = day_ticks["ask"].to_numpy(dtype=float)

        if verbose:
            day_str = day.date().isoformat() if hasattr(day, "date") else str(day)[:10]
            print(
                f"  Streaming {day_str}: {len(tick_times):,} ticks | "
                f"open={len(active_trades)} pending={len(pending_setups) - next_setup_idx}",
                flush=True,
            )

        (
            balance, peak_balance, max_drawdown_usd, max_drawdown_pct,
            wins, losses, missed, trade_history, active_trades, next_setup_idx,
        ) = _process_tick_arrays(
            tick_times, bid_values, ask_values,
            balance, peak_balance, max_drawdown_usd, max_drawdown_pct,
            wins, losses, missed, trade_history, active_trades,
            pending_setups, setup_times, next_setup_idx, max_concurrent, contract_size,
        )

        # Release day memory before loading the next day.
        del day_ticks, tick_times, bid_values, ask_values

    missed += len(active_trades)
    missed += len(pending_setups) - next_setup_idx
    winrate = (wins / (wins + losses)) * 100 if wins + losses else 0.0

    return {
        "initial_balance": float(starting_capital),
        "final_balance": max(0.0, balance),
        "wins": wins,
        "losses": losses,
        "missed": missed,
        "total_resolved": wins + losses,
        "winrate": winrate,
        "max_drawdown_usd": max_drawdown_usd,
        "max_drawdown_pct": max_drawdown_pct,
        "blown": balance <= 0,
        "trade_history": trade_history,
        "_days_with_ticks": days_with_ticks,
        "_days_missing": days_missing,
    }
