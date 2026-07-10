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


def test_breaker_retest_does_not_mark_mitigated_without_closed_invalidation():
    df = pd.DataFrame({
        'Open':  [100.0, 102.0, 101.0, 103.0, 102.0, 101.0],
        'High':  [101.0, 102.5, 103.2, 103.5, 102.8, 103.0],
        'Low':   [99.0, 100.5, 100.8, 99.0, 100.0, 99.8],
        'Close': [100.5, 101.0, 103.0, 100.0, 101.8, 101.5],
        'Swing_High': [np.nan]*6,
        'Swing_Low': [np.nan]*6,
        'BOS': [np.nan, np.nan, 102.5, np.nan, np.nan, np.nan],
        'CHoCH': [np.nan]*6,
        'Trend': [1]*6,
    })

    result = detect_fvg_and_ob(df, symbol="XAUUSD")

    assert result['BB_Type'].iloc[3] == 'BEARISH'
    assert bool(result['BB_Mitigated'].iloc[3]) is False

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
    
    # Index 9 retests the swap level but does not close back below the support zone.
    assert bool(df['Swap_Mitigated'].iloc[6]) is False


def test_swapzone_close_beyond_zone_marks_mitigated():
    df = pd.DataFrame({
        'Open':  [100.0, 101.0, 102.0, 101.0, 100.0, 101.0, 103.5, 104.0, 103.0, 101.0],
        'High':  [101.0, 103.0, 102.5, 102.0, 101.0, 102.5, 104.5, 105.0, 104.0, 102.0],
        'Low':   [99.0,  100.5, 101.5, 100.5, 99.5,  100.0, 103.0, 103.5, 102.0, 99.0],
        'Close': [100.5, 102.0, 101.0, 101.5, 100.5, 102.0, 104.0, 104.5, 102.5, 100.0]
    })
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan
    df.loc[1, 'Swing_High'] = 103.0

    result = detect_snr_and_swapzones(df)

    assert result['Swap_Type'].iloc[6] == 'SUPPORT'
    assert bool(result['Swap_Mitigated'].iloc[6]) is True

def test_supply_demand_zones_detection():
    from src.smc_detector import detect_supply_demand_zones
    # Construct a series of prices that forms a Rally-Base-Rally (RBR) -> Demand zone
    df = pd.DataFrame({
        'Open':  [100.0, 100.0, 110.0, 111.0, 115.0],
        'High':  [101.0, 111.0, 112.0, 122.0, 118.0],
        'Low':   [99.0,  99.0,  109.0, 110.0, 108.0],
        'Close': [100.5, 110.0, 111.0, 121.0, 114.0]
    })
    
    df = detect_supply_demand_zones(df, symbol="XAUUSD")
    
    # Index 3 is the expansion candle of Rally-Base-Rally.
    # So index 3 should have SD_Type = 'DEMAND_RBR'
    assert df['SD_Type'].iloc[3] == 'DEMAND_RBR'
    assert df['SD_Top'].iloc[3] == 112.0
    assert df['SD_Bottom'].iloc[3] == 109.0
    # Check that it calculates SL and Fibo levels correctly
    assert df['SD_SL'].iloc[3] == 109.0 - (20 * 0.1) # XAUUSD pip multiplier is 0.1, buffer is 2.0


def test_supply_demand_retest_does_not_mark_mitigated_without_closed_invalidation():
    from src.smc_detector import detect_supply_demand_zones

    df = pd.DataFrame({
        'Open':  [100.0, 100.0, 110.0, 111.0, 115.0, 111.0],
        'High':  [101.0, 111.0, 112.0, 122.0, 118.0, 112.0],
        'Low':   [99.0,  99.0,  109.0, 110.0, 108.0, 109.5],
        'Close': [100.5, 110.0, 111.0, 121.0, 114.0, 110.5],
    })

    result = detect_supply_demand_zones(df, symbol="XAUUSD")

    assert result['SD_Type'].iloc[3] == 'DEMAND_RBR'
    assert bool(result['SD_Mitigated'].iloc[3]) is False


def test_supply_demand_close_beyond_zone_marks_mitigated():
    from src.smc_detector import detect_supply_demand_zones

    df = pd.DataFrame({
        'Open':  [100.0, 100.0, 110.0, 111.0, 115.0, 111.0],
        'High':  [101.0, 111.0, 112.0, 122.0, 118.0, 112.0],
        'Low':   [99.0,  99.0,  109.0, 110.0, 108.0, 106.0],
        'Close': [100.5, 110.0, 111.0, 121.0, 114.0, 108.0],
    })

    result = detect_supply_demand_zones(df, symbol="XAUUSD")

    assert result['SD_Type'].iloc[3] == 'DEMAND_RBR'
    assert bool(result['SD_Mitigated'].iloc[3]) is True
