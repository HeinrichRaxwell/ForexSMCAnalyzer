import numpy as np
import pandas as pd
import pytest
from src.setup_features import (
    rr_ratio, atr_percentile, body_to_range_ratio,
    dist_to_recent_swing_norm, htf_trend_aligned,
)


def test_rr_ratio_buy():
    # entry 2000, sl 1997 (risk 3), tp 2006 (reward 6) => 2.0
    assert rr_ratio(entry=2000.0, sl=1997.0, tp=2006.0) == pytest.approx(2.0)


def test_rr_ratio_zero_risk_safe():
    assert rr_ratio(entry=2000.0, sl=2000.0, tp=2006.0) == 0.0


def test_atr_percentile_midrange():
    series = pd.Series(np.arange(1, 101, dtype=float))  # 1..100
    # current atr = 50 => percentile ~0.49-0.50
    p = atr_percentile(series, current_atr=50.0)
    assert 0.45 <= p <= 0.55


def test_atr_percentile_max():
    series = pd.Series(np.arange(1, 11, dtype=float))
    assert atr_percentile(series, current_atr=100.0) == pytest.approx(1.0)


def test_body_to_range_ratio_full_body():
    # open 2000 close 2010 high 2010 low 2000 => body 10 range 10 => 1.0
    assert body_to_range_ratio(open_=2000.0, high=2010.0, low=2000.0, close=2010.0) == pytest.approx(1.0)


def test_body_to_range_ratio_doji():
    # body 0 range 10 => 0.0
    assert body_to_range_ratio(open_=2005.0, high=2010.0, low=2000.0, close=2005.0) == pytest.approx(0.0)


def test_body_to_range_zero_range_safe():
    assert body_to_range_ratio(open_=2000.0, high=2000.0, low=2000.0, close=2000.0) == 0.0


def test_dist_to_recent_swing_norm():
    # entry 2000, swing 2012, atr 4 => 3.0
    assert dist_to_recent_swing_norm(entry=2000.0, swing_price=2012.0, atr=4.0) == pytest.approx(3.0)


def test_dist_to_recent_swing_zero_atr_safe():
    assert dist_to_recent_swing_norm(entry=2000.0, swing_price=2012.0, atr=0.0) == 0.0


def test_htf_trend_aligned_match():
    assert htf_trend_aligned(direction=1, htf_trend=1) == 1
    assert htf_trend_aligned(direction=-1, htf_trend=-1) == 1


def test_htf_trend_aligned_mismatch():
    assert htf_trend_aligned(direction=1, htf_trend=-1) == 0
    assert htf_trend_aligned(direction=1, htf_trend=0) == 0
