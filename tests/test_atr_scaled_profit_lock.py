"""Tests for ATR-scaled profit-lock (prevents premature BEP in volatile sessions)."""
from collections import namedtuple

import pandas as pd
import pytest

from src.execution import (
    _atr_pips_from_df,
    compute_atr_lock_policy,
    _profit_lock_stop_loss,
)

Tick = namedtuple("Tick", ["bid", "ask"])


def _closed_df(rows):
    df = pd.DataFrame(rows, columns=["High", "Low", "Close"])
    df.attrs["closed_only"] = True  # use every row as a closed candle
    return df


# --- _atr_pips_from_df ---------------------------------------------------

def test_atr_pips_from_constant_range_df():
    # Each candle spans 5.0 in price; pip_multiplier 0.1 => 50 pips ATR.
    df = _closed_df([[2005.0, 2000.0, 2002.0]] * 10)
    assert _atr_pips_from_df(df, pip_multiplier=0.1) == pytest.approx(50.0)


def test_atr_pips_returns_none_on_thin_data():
    assert _atr_pips_from_df(None, 0.1) is None
    assert _atr_pips_from_df(_closed_df([[1.0, 0.0, 0.5]]), 0.1) is None
    assert _atr_pips_from_df(_closed_df([[1, 0, 0.5]] * 3), pip_multiplier=0) is None


# --- compute_atr_lock_policy ---------------------------------------------

def test_policy_quiet_session_falls_back_to_baseline():
    step, gap = compute_atr_lock_policy(
        atr_pips=10.0, base_step_pips=50.0, base_gap_pips=50.0,
        atr_arm_mult=1.5, atr_gap_mult=1.0,
    )
    assert (step, gap) == (50.0, 50.0)


def test_policy_high_volatility_widens_both_distances():
    step, gap = compute_atr_lock_policy(
        atr_pips=60.0, base_step_pips=50.0, base_gap_pips=50.0,
        atr_arm_mult=1.5, atr_gap_mult=1.0,
    )
    assert step == pytest.approx(90.0)   # 1.5 * 60
    assert gap == pytest.approx(60.0)    # 1.0 * 60


# --- _profit_lock_stop_loss integration ----------------------------------

def test_flag_off_keeps_legacy_ladder(monkeypatch):
    """Regression guard: with ATR scaling off, behaviour is unchanged."""
    monkeypatch.setenv("MT5_ATR_SCALED_LOCK", "False")
    monkeypatch.setenv("MT5_PROFIT_LOCK_STEP_PIPS", "50")
    monkeypatch.setenv("MT5_PROFIT_LOCK_GAP_PIPS", "50")
    monkeypatch.setenv("MT5_CUSTOM_STAGED_PROFIT_LOCK", "False")

    entry, pm = 2000.0, 0.1
    tick = Tick(bid=entry + 120 * pm, ask=entry + 120 * pm + 0.02)  # +120 pips
    sl = _profit_lock_stop_loss(entry, 1, tick, pm, spread_buffer=0.2,
                                atr_pips=60.0)  # atr present but flag off => ignored
    # floor(120/50)*50 - 50 = 50 locked pips -> entry + 50 pips
    assert sl == pytest.approx(entry + 50 * pm)


def test_stepped_50pip_ladder_buy(monkeypatch):
    """150p→lock100, 200p→lock150, 250p→lock200 (step=50, gap=50, buy)."""
    monkeypatch.setenv("MT5_PROFIT_LOCK_STEP_PIPS", "50")
    monkeypatch.setenv("MT5_PROFIT_LOCK_GAP_PIPS", "50")
    monkeypatch.setenv("MT5_CUSTOM_STAGED_PROFIT_LOCK", "False")
    monkeypatch.setenv("MT5_ATR_SCALED_LOCK", "False")

    entry, pm = 2000.0, 0.1
    for profit_pips, expected_lock_pips in [(150, 100), (200, 150), (250, 200), (300, 250)]:
        tick = Tick(bid=entry + profit_pips * pm, ask=entry + profit_pips * pm + 0.02)
        sl = _profit_lock_stop_loss(entry, 1, tick, pm, spread_buffer=0.0)
        assert sl == pytest.approx(entry + expected_lock_pips * pm), \
            f"At +{profit_pips}p profit, expected SL at +{expected_lock_pips}p, got {(sl - entry)/pm:.1f}p"


def test_stepped_50pip_ladder_sell(monkeypatch):
    """Mirror of buy test for sell direction."""
    monkeypatch.setenv("MT5_PROFIT_LOCK_STEP_PIPS", "50")
    monkeypatch.setenv("MT5_PROFIT_LOCK_GAP_PIPS", "50")
    monkeypatch.setenv("MT5_CUSTOM_STAGED_PROFIT_LOCK", "False")
    monkeypatch.setenv("MT5_ATR_SCALED_LOCK", "False")

    entry, pm = 2000.0, 0.1
    for profit_pips, expected_lock_pips in [(150, 100), (200, 150), (250, 200)]:
        tick = Tick(bid=entry - profit_pips * pm - 0.02, ask=entry - profit_pips * pm)
        sl = _profit_lock_stop_loss(entry, -1, tick, pm, spread_buffer=0.0)
        assert sl == pytest.approx(entry - expected_lock_pips * pm), \
            f"Sell: at -{profit_pips}p profit, expected SL at -{expected_lock_pips}p"


def test_flag_on_high_atr_gives_more_breathing_room(monkeypatch):
    monkeypatch.setenv("MT5_ATR_SCALED_LOCK", "True")
    monkeypatch.setenv("MT5_PROFIT_LOCK_STEP_PIPS", "50")
    monkeypatch.setenv("MT5_PROFIT_LOCK_GAP_PIPS", "50")
    monkeypatch.setenv("MT5_CUSTOM_STAGED_PROFIT_LOCK", "False")
    monkeypatch.setenv("MT5_ATR_LOCK_ARM_MULT", "1.5")
    monkeypatch.setenv("MT5_ATR_LOCK_GAP_MULT", "1.0")

    entry, pm = 2000.0, 0.1

    # At +50 pips during high ATR (60), step widens to 90 -> no lock yet (breathing).
    tick_50 = Tick(bid=entry + 50 * pm, ask=entry + 50 * pm + 0.02)
    assert _profit_lock_stop_loss(entry, 1, tick_50, pm, 0.2, atr_pips=60.0) is None

    # At +120 pips: floor(120/90)*90 - 60 = 30 locked pips -> entry + 30 pips
    tick_120 = Tick(bid=entry + 120 * pm, ask=entry + 120 * pm + 0.02)
    sl = _profit_lock_stop_loss(entry, 1, tick_120, pm, 0.2, atr_pips=60.0)
    assert sl == pytest.approx(entry + 30 * pm)
