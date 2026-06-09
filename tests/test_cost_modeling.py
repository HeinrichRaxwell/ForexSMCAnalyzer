import os
import pytest
from src.labeler import compute_cost_r


def test_cost_r_basic():
    # spread 0.30 + slippage 0.0, risk 3.0 USD => 0.10 R
    assert compute_cost_r(risk_pips=3.0, spread_usd=0.30, slippage_usd=0.0) == pytest.approx(0.10)


def test_cost_r_small_risk_larger_fraction():
    # risk kecil => cost_R lebih besar
    big = compute_cost_r(risk_pips=10.0, spread_usd=0.30, slippage_usd=0.0)
    small = compute_cost_r(risk_pips=1.0, spread_usd=0.30, slippage_usd=0.0)
    assert small > big
    assert small == pytest.approx(0.30)


def test_cost_r_includes_slippage():
    assert compute_cost_r(risk_pips=2.0, spread_usd=0.30, slippage_usd=0.10) == pytest.approx(0.20)


def test_cost_r_zero_risk_no_div_zero():
    # risk_pips=0 tidak boleh ZeroDivisionError; kembalikan 0.0 (tidak bisa hitung)
    assert compute_cost_r(risk_pips=0.0, spread_usd=0.30, slippage_usd=0.0) == 0.0


def test_cost_r_negative_risk_treated_as_abs():
    assert compute_cost_r(risk_pips=-3.0, spread_usd=0.30, slippage_usd=0.0) == pytest.approx(0.10)


from src.labeler import compute_pnl_relative


def test_pnl_relative_win_subtracts_cost():
    # RR 2:1, risk 3 USD, cost 0.30/3=0.10 => 2.0 - 0.10 = 1.90
    r = compute_pnl_relative(label=1, entry=2000.0, sl=1997.0, tp=2006.0,
                             spread_usd=0.30, slippage_usd=0.0)
    assert r == pytest.approx(1.90)


def test_pnl_relative_loss_is_deeper_than_minus_one():
    # loss => -1.0 - cost
    r = compute_pnl_relative(label=0, entry=2000.0, sl=1997.0, tp=2006.0,
                             spread_usd=0.30, slippage_usd=0.0)
    assert r == pytest.approx(-1.10)


def test_pnl_relative_zero_risk_safe():
    r = compute_pnl_relative(label=1, entry=2000.0, sl=2000.0, tp=2006.0,
                             spread_usd=0.30, slippage_usd=0.0)
    # risk 0 => base R 0, cost 0 => 0.0
    assert r == 0.0
