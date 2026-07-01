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


def run_tick_simulation(
    ticks: pd.DataFrame,
    setups: list,
    starting_capital: float,
    contract_size: float = 100.0,
    max_concurrent: int = 1,
) -> dict:
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

    tick_frame = ticks.copy()
    tick_frame["time"] = pd.to_datetime(tick_frame["time"], errors="coerce")
    tick_frame = tick_frame.dropna(subset=["time"]).sort_values("time", kind="mergesort")
    tick_times = tick_frame["time"].to_numpy(dtype="datetime64[ns]")
    bid_values = tick_frame["bid"].astype(float).to_numpy()
    ask_values = tick_frame["ask"].astype(float).to_numpy()
    pending_setups = sorted(setups, key=_setup_time)
    setup_times = [_setup_time(setup) for setup in pending_setups]
    next_setup_idx = 0

    tick_idx = 0
    while tick_idx < len(tick_times):
        if balance <= 0:
            balance = 0.0
            break

        if not active_trades:
            if next_setup_idx >= len(pending_setups):
                break
            next_setup_time = setup_times[next_setup_idx].to_datetime64()
            if tick_times[tick_idx] < next_setup_time:
                tick_idx = int(np.searchsorted(tick_times, next_setup_time, side="left"))
                if tick_idx >= len(tick_times):
                    break

        tick_time = pd.Timestamp(tick_times[tick_idx])
        bid = float(bid_values[tick_idx])
        ask = float(ask_values[tick_idx])

        while next_setup_idx < len(pending_setups) and setup_times[next_setup_idx] <= tick_time:
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
                "entry_time": trade["entry_time"],
                "exit_time": tick_time,
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
