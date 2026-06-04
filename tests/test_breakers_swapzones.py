import pytest
import pandas as pd
import numpy as np
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_snr_and_swapzones

def test_breaker_blocks_detection():
    # Construct a series of prices that forms a Bullish OB and then breaks below it
    df = pd.DataFrame({
        'Open':  [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 102.0, 98.0,  99.0],
        'High':  [101.0, 102.5, 103.0, 104.5, 105.5, 106.0, 105.0, 103.0, 99.5, 100.5],
        'Low':   [99.0,  100.5, 101.5, 102.5, 103.5, 104.0, 103.0, 101.0, 97.0,  98.0],
        'Close': [100.5, 102.0, 102.5, 104.0, 105.0, 104.5, 103.5, 101.5, 97.5,  99.0]
    })
    
    # Manually inject BOS/CHoCH signals to trigger OB
    # A bullish OB is formed by the last bearish candle before a bullish break.
    # Let's say index 1 is a bearish candle (Close < Open)
    df.loc[1, 'Open'] = 102.0
    df.loc[1, 'Close'] = 101.0
    df.loc[1, 'Low'] = 100.5
    df.loc[1, 'High'] = 102.5
    
    # Index 2 is a bullish break candle (Close > Open)
    df.loc[2, 'Open'] = 101.0
    df.loc[2, 'Close'] = 103.0
    df.loc[2, 'Low'] = 100.8
    df.loc[2, 'High'] = 103.2
    
    # Set structures
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan
    df['BOS'] = np.nan
    df['CHoCH'] = np.nan
    df['Trend'] = 1
    
    # Inject a Bullish BOS at index 2 (breaking index 1 high)
    df.loc[2, 'BOS'] = 102.5
    
    df = detect_fvg_and_ob(df, symbol="XAUUSD")
    
    # Index 2 should have created a Bullish OB based on index 1 wicks:
    # OB_Top = 102.5, OB_Bottom = 100.5
    assert df['OB_Type'].iloc[2] == 'BULLISH'
    assert df['OB_Top'].iloc[2] == 102.5
    assert df['OB_Bottom'].iloc[2] == 100.5
    
    # Now, index 8 close is 97.5, which breaks below OB_Bottom (100.5)
    # This should create a Bearish Breaker Block at index 8!
    assert df['BB_Type'].iloc[8] == 'BEARISH'
    assert df['BB_Top'].iloc[8] == 102.5
    assert df['BB_Bottom'].iloc[8] == 100.5
    assert df['OB_Mitigated'].iloc[2] == True

def test_swapzones_detection():
    # Construct a series of prices that forms a Swing High and then closes above it
    df = pd.DataFrame({
        'Open':  [100.0, 101.0, 102.0, 101.0, 100.0, 101.0, 103.5, 104.0, 103.0, 102.0],
        'High':  [101.0, 103.0, 102.5, 102.0, 101.0, 102.5, 104.5, 105.0, 104.0, 103.0],
        'Low':   [99.0,  100.5, 101.5, 100.5, 99.5,  100.0, 103.0, 103.5, 102.0, 101.0],
        'Close': [100.5, 102.0, 101.0, 101.5, 100.5, 102.0, 104.0, 104.5, 102.5, 101.5]
    })
    
    # Add Swing points manually
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan
    
    # High at index 1 is 103.0, which is a Swing High
    df.loc[1, 'Swing_High'] = 103.0
    
    df = detect_snr_and_swapzones(df)
    
    # At index 6, price closes at 104.0, which is above the Swing High of 103.0
    # This should create a Swap Support zone at index 6!
    assert df['Swap_Type'].iloc[6] == 'SUPPORT'
    assert df['Swap_Level'].iloc[6] == 103.0
    
    # Index 9 price is Low=101.0, High=103.0, which touches Swap_Level (103.0)
    # This should mark the swap zone at index 6 as mitigated
    assert df['Swap_Mitigated'].iloc[6] == True
