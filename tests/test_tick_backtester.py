import pandas as pd

from src.tick_backtester import run_tick_simulation


def _ticks(rows):
    return pd.DataFrame(rows)


def _setup(**overrides):
    data = {
        "time": pd.Timestamp("2026-01-01 00:00:00"),
        "index": 1,
        "direction": 1,
        "entry_price": 100.0,
        "sl_price": 95.0,
        "tp_price": 110.0,
        "option_name": "FVG 0.5",
        "strategy": "FVG",
        "lot_size": 0.01,
        "features": {"killzone": 1, "trend": 1},
    }
    data.update(overrides)
    return data


def test_tick_simulation_buy_uses_ask_for_entry_and_bid_for_tp():
    result = run_tick_simulation(
        _ticks([
            {"time": "2026-01-01 00:00:01", "bid": 100.2, "ask": 100.4},
            {"time": "2026-01-01 00:00:02", "bid": 99.8, "ask": 100.0},
            {"time": "2026-01-01 00:00:03", "bid": 110.1, "ask": 110.3},
        ]),
        [_setup(direction=1, entry_price=100.0, sl_price=95.0, tp_price=110.0)],
        starting_capital=50.0,
    )

    assert result["wins"] == 1
    assert result["losses"] == 0
    assert result["final_balance"] == 60.0
    assert result["trade_history"][0]["entry_time"] == pd.Timestamp("2026-01-01 00:00:02")


def test_tick_simulation_sell_uses_bid_for_entry_and_ask_for_sl():
    result = run_tick_simulation(
        _ticks([
            {"time": "2026-01-01 00:00:01", "bid": 99.8, "ask": 100.0},
            {"time": "2026-01-01 00:00:02", "bid": 100.1, "ask": 100.3},
            {"time": "2026-01-01 00:00:03", "bid": 104.8, "ask": 105.1},
        ]),
        [_setup(direction=-1, entry_price=100.0, sl_price=105.0, tp_price=90.0)],
        starting_capital=50.0,
    )

    assert result["wins"] == 0
    assert result["losses"] == 1
    assert result["final_balance"] == 45.0
    assert result["trade_history"][0]["outcome"] == "LOSS"


def test_tick_simulation_pending_trade_missed_when_tp_runs_before_entry():
    result = run_tick_simulation(
        _ticks([
            {"time": "2026-01-01 00:00:01", "bid": 111.0, "ask": 111.2},
            {"time": "2026-01-01 00:00:02", "bid": 112.0, "ask": 112.2},
        ]),
        [_setup(direction=1, entry_price=100.0, sl_price=95.0, tp_price=110.0)],
        starting_capital=50.0,
    )

    assert result["wins"] == 0
    assert result["losses"] == 0
    assert result["missed"] == 1
    assert result["final_balance"] == 50.0


def test_tick_simulation_respects_max_concurrent_setups():
    result = run_tick_simulation(
        _ticks([
            {"time": "2026-01-01 00:00:01", "bid": 99.8, "ask": 100.0},
            {"time": "2026-01-01 00:00:02", "bid": 110.0, "ask": 110.2},
        ]),
        [
            _setup(index=1, entry_price=100.0, tp_price=110.0),
            _setup(index=2, entry_price=100.0, tp_price=110.0),
        ],
        starting_capital=50.0,
        max_concurrent=1,
    )

    assert result["wins"] == 1
    assert result["missed"] == 1
