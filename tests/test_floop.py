import pandas as pd
import numpy as np
import pytest
from src.indicators.floop import (
    run_floop_pro,
    calculate_tr,
    calculate_adx,
    calculate_chop,
    calculate_range_filter,
    rma,
    _apply_signal_gates,
    _rolling_percentile_nearest_rank,
    _score_mtf_confluence,
)

def test_floop_indicator_components():
    # Generate synthetic trend data
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=100, freq="1H")
    
    # Generate a simple trending price series
    close = np.linspace(100, 110, 100) + np.random.normal(0, 0.1, 100)
    high = close + 0.5
    low = close - 0.5
    open_val = close - np.random.normal(0, 0.1, 100)
    volume = np.random.randint(100, 500, 100).astype(float)
    
    df = pd.DataFrame({
        'time': dates,
        'Open': open_val,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    # Test TR
    tr = calculate_tr(df)
    assert len(tr) == 100
    assert (tr >= 0).all()
    
    # Test ADX
    adx_val, plus_di, minus_di = calculate_adx(df, len_adx=14)
    assert len(adx_val) == 100
    
    # Test CHOP
    chop = calculate_chop(df, period=14)
    assert len(chop) == 100
    
    # Test Range Filter
    atr = tr.rolling(14).mean().fillna(1.0)
    filt, trend, sig = calculate_range_filter(df['Close'], atr, sensitivity=6, atr_multiplier=0.8)
    assert len(filt) == 100
    assert len(trend) == 100
    assert len(sig) == 100

def test_run_floop_pro_defaults():
    # Generate synthetic trend data
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=100, freq="1H")
    
    # Generate a simple trending price series
    close = np.linspace(100, 110, 100) + np.random.normal(0, 0.1, 100)
    high = close + 0.5
    low = close - 0.5
    open_val = close - np.random.normal(0, 0.1, 100)
    volume = np.random.randint(100, 500, 100).astype(float)
    
    df = pd.DataFrame({
        'time': dates,
        'Open': open_val,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    signals, strength, trend = run_floop_pro(
        df,
        sensitivity=6,
        atr_len=14,
        atr_mult=0.8,
        use_adx=True,
        adx_thresh=20.0,
        use_chop=True,
        chop_thresh=61.8,
        use_cooldown=True,
        cooldown_len=5,
        ema_filter=False
    )
    
    assert len(signals) == 100
    assert len(strength) == 100
    assert len(trend) == 100
    assert signals.isin([-1, 0, 1]).all()
    assert (strength >= 0).all() and (strength <= 14).all()
    assert trend.isin([-1, 0, 1]).all()

def test_rma_matches_pine_wilder_seeded_by_sma():
    src = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = rma(src, 3)

    expected = pd.Series([np.nan, np.nan, 2.0, 2.6666666667, 3.4444444444])
    pd.testing.assert_series_equal(result, expected, check_names=False, check_exact=False, rtol=1e-9)

def test_rolling_percentile_nearest_rank_matches_pine():
    src = pd.Series([10.0, 20.0, 30.0, 40.0])

    result = _rolling_percentile_nearest_rank(src, window=4, percentile=65)

    expected = pd.Series([np.nan, np.nan, np.nan, 30.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_floop_cooldown_resets_only_after_accepted_signal():
    rf_sig = pd.Series([1, 0, 1, 0, 1], dtype=float)
    ema_gate = np.array([True, True, True, True, True])
    adx_trending = pd.Series([False, True, True, True, True])
    chop_clear = pd.Series([True, True, True, True, True])

    long_sig, short_sig, cooldown_clear, chop_gate = _apply_signal_gates(
        rf_sig=rf_sig,
        ema_gate=ema_gate,
        adx_trending=adx_trending,
        chop_clear=chop_clear,
        use_adx=True,
        use_chop=True,
        use_cooldown=True,
        cooldown_len=5,
    )

    assert long_sig.tolist() == [False, False, True, False, False]
    assert short_sig.tolist() == [False, False, False, False, False]
    assert cooldown_clear.tolist() == [True, True, True, False, False]
    assert chop_gate.tolist() == [False, True, True, False, False]

def test_floop_mtf_score_uses_only_pine_scored_timeframes():
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    rf_trend = pd.Series([1.0, -1.0, 1.0], index=idx)
    aligned = pd.Series([1.0, -1.0, 1.0], index=idx)
    extra_aligned = pd.Series([1.0, -1.0, 1.0], index=idx)
    opposite = pd.Series([-1.0, 1.0, -1.0], index=idx)

    score = _score_mtf_confluence(
        rf_trend,
        {
            "M5": aligned,
            "M15": aligned,
            "H1": aligned,
            "H4": aligned,
            "M30": extra_aligned,
            "D1": extra_aligned,
            "M1": opposite,
        },
    )

    assert score.tolist() == [4, 4, 4]
