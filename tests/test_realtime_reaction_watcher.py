from types import SimpleNamespace

import pytest

from src.realtime_reaction_watcher import (
    iter_realtime_watch_candidates,
    run_realtime_reaction_pass,
    should_enter_on_realtime_reaction,
)


def _tick(bid, ask):
    return SimpleNamespace(bid=bid, ask=ask)


def _setup(**overrides):
    setup = {
        "direction": 1,
        "entry_price": 4000.0,
        "sl_price": 3990.0,
        "tp_price": 4020.0,
        "rejection_confirmed": True,
    }
    setup.update(overrides)
    return setup


def test_buy_realtime_reaction_enters_when_tick_turns_up_inside_zone():
    decision = should_enter_on_realtime_reaction(
        _setup(direction=1, entry_price=4000.0, sl_price=3990.0),
        previous_tick=_tick(bid=3994.50, ask=3994.76),
        current_tick=_tick(bid=3994.72, ask=3994.98),
        min_reaction_move=0.10,
    )

    assert decision.should_enter is True
    assert decision.reason == "realtime_reaction_confirmed"
    assert decision.current_price == pytest.approx(3994.98)
    assert decision.reaction_move == pytest.approx(0.22)


def test_sell_realtime_reaction_enters_when_tick_turns_down_inside_zone():
    decision = should_enter_on_realtime_reaction(
        _setup(direction=-1, entry_price=4000.0, sl_price=4010.0),
        previous_tick=_tick(bid=4005.20, ask=4005.46),
        current_tick=_tick(bid=4004.98, ask=4005.18),
        min_reaction_move=0.10,
    )

    assert decision.should_enter is True
    assert decision.reason == "realtime_reaction_confirmed"
    assert decision.current_price == pytest.approx(4004.98)
    assert decision.reaction_move == pytest.approx(0.28)


def test_buy_realtime_reaction_waits_when_tick_is_still_falling_inside_zone():
    decision = should_enter_on_realtime_reaction(
        _setup(direction=1, entry_price=4000.0, sl_price=3990.0),
        previous_tick=_tick(bid=3994.90, ask=3995.16),
        current_tick=_tick(bid=3994.70, ask=3994.96),
        min_reaction_move=0.10,
    )

    assert decision.should_enter is False
    assert decision.reason == "realtime_reaction_not_confirmed"


def test_realtime_reaction_waits_outside_entry_zone():
    decision = should_enter_on_realtime_reaction(
        _setup(direction=1, entry_price=4000.0, sl_price=3990.0),
        previous_tick=_tick(bid=4001.00, ask=4001.26),
        current_tick=_tick(bid=4001.30, ask=4001.56),
        min_reaction_move=0.10,
    )

    assert decision.should_enter is False
    assert decision.reason == "price_outside_entry_zone"


def test_registry_single_candidate_executes_market_order_on_reaction_tick():
    sent_signals = {
        "M30_FVG_SINGLE_BULL_4000_2026-06-10": {
            "timeframe": "M30",
            "direction": "BULL",
            "type": "FVG",
            "price": 4000.0,
            "probability": 0.74,
            "ticket_id": None,
            "features": {
                "direction": 1,
                "entry_price": 4000.0,
                "sl_price": 3990.0,
                "tp_price": 4020.0,
            },
        }
    }
    calls = []

    def execute_market(setup, symbol):
        calls.append((setup, symbol))
        return 123456, "MARKET ORDER PLACED"

    result = run_realtime_reaction_pass(
        sent_signals,
        symbol="XAUUSD",
        previous_tick=_tick(bid=3994.50, ask=3994.76),
        current_tick=_tick(bid=3994.72, ask=3994.98),
        execute_market_order=execute_market,
        min_reaction_move=0.10,
        now="2026-06-10 12:00:00",
    )

    assert result.changed is True
    assert result.executed_count == 1
    assert sent_signals["M30_FVG_SINGLE_BULL_4000_2026-06-10"]["ticket_id"] == 123456
    assert sent_signals["M30_FVG_SINGLE_BULL_4000_2026-06-10"]["realtime_reaction_entry_at"] == "2026-06-10 12:00:00"
    assert calls[0][0]["entry_price"] == 4000.0
    assert calls[0][1] == "XAUUSD"


def test_registry_m15_candidate_is_monitoring_only_for_realtime_reaction():
    sent_signals = {
        "M15_FVG_SINGLE_BULL_4000_2026-06-10": {
            "timeframe": "M15",
            "direction": "BULL",
            "type": "FVG",
            "price": 4000.0,
            "probability": 0.74,
            "ticket_id": None,
            "features": {
                "timeframe": 15,
                "direction": 1,
                "entry_price": 4000.0,
                "sl_price": 3990.0,
                "tp_price": 4020.0,
            },
        }
    }
    calls = []

    def execute_market(setup, symbol):
        calls.append((setup, symbol))
        return 123456, "MARKET ORDER PLACED"

    result = run_realtime_reaction_pass(
        sent_signals,
        symbol="XAUUSD",
        previous_tick=_tick(bid=3994.50, ask=3994.76),
        current_tick=_tick(bid=3994.72, ask=3994.98),
        execute_market_order=execute_market,
        min_reaction_move=0.10,
        now="2026-06-10 12:00:00",
    )

    assert result.changed is False
    assert result.executed_count == 0
    assert sent_signals["M15_FVG_SINGLE_BULL_4000_2026-06-10"]["ticket_id"] is None
    assert calls == []


def test_registry_candidate_with_existing_ticket_is_not_duplicated():
    sent_signals = {
        "M15_FVG_SINGLE_BULL_4000_2026-06-10": {
            "timeframe": "M15",
            "direction": "BULL",
            "type": "FVG",
            "price": 4000.0,
            "probability": 0.74,
            "ticket_id": 111111,
            "features": {
                "direction": 1,
                "entry_price": 4000.0,
                "sl_price": 3990.0,
                "tp_price": 4020.0,
            },
        }
    }

    candidates = list(iter_realtime_watch_candidates(sent_signals))

    assert candidates == []


def test_dual_registry_realtime_pass_executes_only_one_leg_per_tick():
    sent_signals = {
        "M30_FVG_DUAL_BULL_4000_3998_2026-06-10": {
            "timeframe": "M30",
            "direction": "BULL",
            "type": "FVG",
            "price_0.5": 4000.0,
            "price_0.618": 3998.0,
            "probability_0.5": 0.74,
            "probability_0.618": 0.73,
            "ticket_a": None,
            "ticket_b": None,
            "features_0.5": {
                "direction": 1,
                "entry_price": 4000.0,
                "sl_price": 3990.0,
                "tp_price": 4020.0,
            },
            "features_0.618": {
                "direction": 1,
                "entry_price": 3998.0,
                "sl_price": 3990.0,
                "tp_price": 4020.0,
            },
        }
    }
    calls = []

    def execute_market(setup, symbol):
        calls.append((setup["entry_price"], symbol))
        return 123456 + len(calls), "MARKET ORDER PLACED"

    result = run_realtime_reaction_pass(
        sent_signals,
        symbol="XAUUSD",
        previous_tick=_tick(bid=3994.50, ask=3994.76),
        current_tick=_tick(bid=3994.72, ask=3994.98),
        execute_market_order=execute_market,
        min_reaction_move=0.10,
        now="2026-06-10 12:00:00",
    )

    record = sent_signals["M30_FVG_DUAL_BULL_4000_3998_2026-06-10"]
    assert result.executed_count == 1
    assert calls == [(4000.0, "XAUUSD")]
    assert record["ticket_a"] == 123457
    assert record["ticket_b"] is None


def test_scanner_realtime_reaction_cycle_saves_changed_registry(monkeypatch):
    import src.scanner_worker as scanner_worker

    saved = {}
    sent_signals = {
        "M30_FVG_SINGLE_BULL_4000_2026-06-10": {
            "timeframe": "M30",
            "direction": "BULL",
            "type": "FVG",
            "price": 4000.0,
            "probability": 0.74,
            "ticket_id": None,
            "features": {
                "direction": 1,
                "entry_price": 4000.0,
                "sl_price": 3990.0,
                "tp_price": 4020.0,
            },
        }
    }

    monkeypatch.setattr(scanner_worker, "load_sent_signals", lambda: sent_signals)
    monkeypatch.setattr(scanner_worker, "save_sent_signals", lambda payload: saved.update(payload))
    monkeypatch.setattr(scanner_worker, "execute_market_order_for_setup", lambda setup, symbol: (777, "MARKET ORDER PLACED"))
    monkeypatch.setattr(scanner_worker, "manage_active_trades", lambda symbol, magic, timeframes_data: None)
    monkeypatch.setattr(scanner_worker, "get_active_broker_symbol", lambda symbol: "XAUUSDm")
    monkeypatch.setattr(scanner_worker, "connect_mt5", lambda: True)

    fake_mt5 = SimpleNamespace(symbol_info_tick=lambda symbol: _tick(bid=3994.72, ask=3994.98))
    monkeypatch.setattr(scanner_worker, "mt5", fake_mt5, raising=False)

    result = scanner_worker.run_realtime_reaction_cycle(
        "XAUUSD",
        previous_tick=_tick(bid=3994.50, ask=3994.76),
        min_reaction_move=0.10,
    )

    assert result.executed_count == 1
    assert saved["M30_FVG_SINGLE_BULL_4000_2026-06-10"]["ticket_id"] == 777
