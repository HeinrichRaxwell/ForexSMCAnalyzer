from types import SimpleNamespace

import pytest

from src.live_risk_governor import (
    evaluate_daily_risk,
    summarize_daily_pips_from_deals,
)


def _deal(**overrides):
    data = {
        "position_id": 1,
        "entry": 0,
        "type": 0,
        "price": 4000.0,
        "magic": 202606,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_daily_risk_allows_trading_before_target_or_loss_limits():
    decision = evaluate_daily_risk(realized_pips=42.0, consecutive_losses=1)

    assert decision.allowed is True
    assert decision.mode == "normal"


def test_daily_risk_switches_to_profit_protection_after_min_target():
    decision = evaluate_daily_risk(realized_pips=125.0, consecutive_losses=0)

    assert decision.allowed is True
    assert decision.mode == "protect_profit"
    assert decision.reason == "daily_min_target_reached"


def test_daily_risk_keeps_trading_after_runner_target():
    decision = evaluate_daily_risk(realized_pips=305.0, consecutive_losses=0)

    assert decision.allowed is True
    assert decision.mode == "protect_profit"
    assert decision.reason == "daily_min_target_reached"


def test_daily_risk_blocks_trading_after_max_loss_or_loss_streak():
    max_loss_decision = evaluate_daily_risk(realized_pips=-201.0, consecutive_losses=0)
    streak_decision = evaluate_daily_risk(realized_pips=0.0, consecutive_losses=3)

    assert max_loss_decision.allowed is False
    assert max_loss_decision.mode == "halt"
    assert max_loss_decision.reason == "daily_limits_hit"
    assert streak_decision.allowed is False
    assert streak_decision.mode == "halt"
    assert streak_decision.reason == "daily_limits_hit"


def test_summarize_daily_pips_from_deals_pairs_entries_and_exits():
    deals = [
        _deal(position_id=1, entry=0, type=0, price=4000.0),
        _deal(position_id=1, entry=1, type=1, price=4010.0),
        _deal(position_id=2, entry=0, type=1, price=4020.0),
        _deal(position_id=2, entry=1, type=0, price=4010.0),
    ]

    summary = summarize_daily_pips_from_deals(
        deals,
        magic=202606,
        pip_multiplier=0.1,
        deal_entry_in=0,
        deal_entry_out=1,
        deal_type_buy=0,
        deal_type_sell=1,
    )

    assert summary.realized_pips == pytest.approx(200.0)
    assert summary.closed_positions == 2
    assert summary.consecutive_losses == 0


def test_summarize_daily_pips_filters_symbol_when_deal_symbol_is_available():
    deals = [
        _deal(position_id=1, entry=0, type=0, price=4000.0, symbol="XAUUSD"),
        _deal(position_id=1, entry=1, type=1, price=4010.0, symbol="XAUUSD"),
        _deal(position_id=2, entry=0, type=0, price=1.1000, symbol="EURUSD"),
        _deal(position_id=2, entry=1, type=1, price=1.1200, symbol="EURUSD"),
    ]

    summary = summarize_daily_pips_from_deals(
        deals,
        symbol="XAUUSD",
        magic=202606,
        pip_multiplier=0.1,
        deal_entry_in=0,
        deal_entry_out=1,
        deal_type_buy=0,
        deal_type_sell=1,
    )

    assert summary.realized_pips == pytest.approx(100.0)
    assert summary.closed_positions == 1
