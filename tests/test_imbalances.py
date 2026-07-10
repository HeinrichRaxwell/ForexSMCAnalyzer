import pytest
import pandas as pd
import numpy as np
from src.smc_detector import detect_fvg_and_ob

def test_detect_fvg_bullish():
    """Test detection of bullish Fair Value Gaps."""
    # Bullish expansion: Candle 0 High is 10, Candle 1 is bullish, Candle 2 Low is 12.
    data = {
        'Open':  [9, 10, 14],
        'High':  [10, 15, 16],
        'Low':   [8, 10, 12],
        'Close': [9.5, 14, 15]
    }
    df = pd.DataFrame(data)
    
    result = detect_fvg_and_ob(df)
    
    assert result['FVG_Type'].iloc[2] == 'BULLISH'
    assert result['FVG_Top'].iloc[2] == 12.0
    assert result['FVG_Bottom'].iloc[2] == 10.0

def test_detect_fvg_bearish():
    """Test detection of bearish Fair Value Gaps."""
    # Bearish expansion: Candle 0 Low is 15, Candle 1 is bearish, Candle 2 High is 13.
    data = {
        'Open':  [16, 15, 12],
        'High':  [17, 15, 13],
        'Low':   [15, 11, 10],
        'Close': [15.5, 12, 11]
    }
    df = pd.DataFrame(data)
    
    result = detect_fvg_and_ob(df)
    
    assert result['FVG_Type'].iloc[2] == 'BEARISH'
    assert result['FVG_Top'].iloc[2] == 15.0
    assert result['FVG_Bottom'].iloc[2] == 13.0

def test_detect_ob_retest_does_not_mark_mitigated():
    """Order Block retest is an entry event; mitigation requires a body close beyond the zone."""
    # 0. Bearish candle (future OB): O=12, H=13, L=10, C=11
    # 1. Bullish break (BOS): O=11, H=18, L=11, C=17 (BOS is triggered here)
    # 2. Bullish candle: O=17, H=19, L=15, C=18
    # 3. Pullback/retest: wick enters the OB, but the candle body does not close beyond OB_Bottom.
    data = {
        'Open':       [12, 11, 17, 18],
        'High':       [13, 18, 19, 18],
        'Low':        [10, 11, 15, 11],
        'Close':      [11, 17, 18, 12],
        'Swing_High': [13, np.nan, np.nan, np.nan],
        'Swing_Low':  [np.nan, 10, np.nan, np.nan],
        'BOS':        [np.nan, 13, np.nan, np.nan],
        'CHoCH':      [np.nan, np.nan, np.nan, np.nan],
        'Trend':      [1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    
    result = detect_fvg_and_ob(df)
    
    # OB should be identified at candle index 1
    assert result['OB_Type'].iloc[1] == 'BULLISH'
    assert result['OB_Top'].iloc[1] == 13.0
    assert result['OB_Bottom'].iloc[1] == 10.0
    
    assert bool(result['OB_Mitigated'].iloc[1]) is False


def test_detect_ob_close_beyond_zone_marks_mitigated_and_creates_breaker():
    """A closed candle beyond the OB invalidates it and creates the breaker block."""
    data = {
        'Open':       [12, 11, 17, 18],
        'High':       [13, 18, 19, 18],
        'Low':        [10, 11, 15, 9],
        'Close':      [11, 17, 18, 9.5],
        'Swing_High': [13, np.nan, np.nan, np.nan],
        'Swing_Low':  [np.nan, 10, np.nan, np.nan],
        'BOS':        [np.nan, 13, np.nan, np.nan],
        'CHoCH':      [np.nan, np.nan, np.nan, np.nan],
        'Trend':      [1, 1, 1, 1]
    }
    df = pd.DataFrame(data)

    result = detect_fvg_and_ob(df)

    assert result['OB_Type'].iloc[1] == 'BULLISH'
    assert bool(result['OB_Mitigated'].iloc[1]) is True
    assert result['BB_Type'].iloc[3] == 'BEARISH'


def test_bearish_order_block_uses_last_bullish_candle_before_drop():
    """Bearish OB must anchor to the final bullish candle before a bearish break."""
    data = {
        'Open':       [100.0, 104.0, 106.0, 105.0, 103.0],
        'High':       [105.0, 107.0, 108.0, 106.0, 104.0],
        'Low':        [99.0, 103.0, 104.0, 100.0, 96.0],
        'Close':      [104.0, 106.0, 105.0, 102.0, 97.0],
        'Swing_High': [np.nan, 107.0, np.nan, np.nan, np.nan],
        'Swing_Low':  [np.nan, np.nan, 104.0, np.nan, np.nan],
        'BOS':        [np.nan, np.nan, np.nan, np.nan, 104.0],
        'CHoCH':      [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Trend':      [-1, -1, -1, -1, -1],
    }
    df = pd.DataFrame(data)

    result = detect_fvg_and_ob(df, symbol="XAUUSD")

    assert result['OB_Type'].iloc[4] == 'BEARISH'
    assert result['OB_Top'].iloc[4] == 107.0
    assert result['OB_Bottom'].iloc[4] == 103.0
    assert result['OB_Fibo_0.5'].iloc[4] == 105.0
