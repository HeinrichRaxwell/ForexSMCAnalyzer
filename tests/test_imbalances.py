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

def test_detect_ob_and_mitigation():
    """Test that Order Blocks are correctly identified and their mitigation is tracked."""
    # 0. Bearish candle (future OB): O=12, H=13, L=10, C=11
    # 1. Bullish break (BOS): O=11, H=18, L=11, C=17 (BOS is triggered here)
    # 2. Bullish candle: O=17, H=19, L=15, C=18
    # 3. Pullback (Mitigation): O=18, H=18, L=12, C=13 (Low 12 goes below OB_Top 13)
    data = {
        'Open':       [12, 11, 17, 18],
        'High':       [13, 18, 19, 18],
        'Low':        [10, 11, 15, 11],
        'Close':      [11, 17, 18, 13],
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
    
    # Mitigation check: candle index 3 Low is 12, which is <= OB_Top (13.0)
    assert result['OB_Mitigated'].iloc[1] == True
