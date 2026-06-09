import pytest
import pandas as pd
import numpy as np
from src.smc_detector import detect_fvg_and_ob, detect_snr_and_swapzones, detect_indecision_candles

def test_bullish_fvg_fibo_xauusd():
    """Test Fibonacci calculations for a Bullish FVG using XAUUSD (pip size 0.1)."""
    # Candle 0 (i-2): Low is 8.0, High is 10.0
    # Candle 1 (i-1): Bullish candle (Close 14.0 > Open 10.0)
    # Candle 2 (i): Low is 12.0, High is 16.0
    data = {
        'Open':  [9.0, 10.0, 14.0],
        'High':  [10.0, 15.0, 16.0],
        'Low':   [8.0, 10.0, 12.0],
        'Close': [9.5, 14.0, 15.0]
    }
    df = pd.DataFrame(data)
    
    result = detect_fvg_and_ob(df, symbol="XAUUSD")
    
    # Assert FVG type
    assert result['FVG_Type'].iloc[2] == 'BULLISH'
    
    # Assert Fibonacci Levels
    expected_fibo_1_0 = 10.0  # Low of Candle 1 (index i-1)
    expected_fibo_0_0 = 15.0  # High of Candle 1 (index i-1)
    expected_fibo_0_5 = 12.5  # 15.0 - 0.5 * (15.0 - 10.0)
    expected_fibo_0_618 = 11.91 # 15.0 - 0.618 * (15.0 - 10.0)
    expected_sl = 8.0        # 10.0 - 2.0 (XAUUSD buffer = 20 * 0.1, boundary = High of Candle 0 = 10.0)
    
    assert np.isclose(result['FVG_Fibo_1.0'].iloc[2], expected_fibo_1_0)
    assert np.isclose(result['FVG_Fibo_0.0'].iloc[2], expected_fibo_0_0)
    assert np.isclose(result['FVG_Fibo_0.5'].iloc[2], expected_fibo_0_5)
    assert np.isclose(result['FVG_Fibo_0.618'].iloc[2], expected_fibo_0_618)
    assert np.isclose(result['FVG_SL'].iloc[2], expected_sl)
 
def test_bullish_fvg_fibo_jpy():
    """Test Fibonacci calculations for a Bullish FVG using a JPY pair (pip size 0.01)."""
    data = {
        'Open':  [9.0, 10.0, 14.0],
        'High':  [10.0, 15.0, 16.0],
        'Low':   [8.0, 10.0, 12.0],
        'Close': [9.5, 14.0, 15.0]
    }
    df = pd.DataFrame(data)
    
    result = detect_fvg_and_ob(df, symbol="USDJPY")
    
    expected_sl = 9.8        # 10.0 - 0.2 (USDJPY buffer = 20 * 0.01, boundary = High of Candle 0 = 10.0)
    assert np.isclose(result['FVG_SL'].iloc[2], expected_sl)
 
def test_bullish_fvg_fibo_forex_standard():
    """Test Fibonacci calculations for a Bullish FVG using standard forex (pip size 0.0001)."""
    data = {
        'Open':  [9.0, 10.0, 14.0],
        'High':  [10.0, 15.0, 16.0],
        'Low':   [8.0, 10.0, 12.0],
        'Close': [9.5, 14.0, 15.0]
    }
    df = pd.DataFrame(data)
    
    result = detect_fvg_and_ob(df, symbol="EURUSD")
    
    expected_sl = 9.998      # 10.0 - 0.002 (EURUSD buffer = 20 * 0.0001, boundary = High of Candle 0 = 10.0)
    assert np.isclose(result['FVG_SL'].iloc[2], expected_sl)
 
def test_bearish_fvg_fibo_xauusd():
    """Test Fibonacci calculations for a Bearish FVG using XAUUSD (pip size 0.1)."""
    # Candle 0 (i-2): Low is 15.0, High is 17.0
    # Candle 1 (i-1): Bearish candle (Close 12.0 < Open 15.0)
    # Candle 2 (i): Low is 10.0, High is 13.0
    data = {
        'Open':  [16.0, 15.0, 12.0],
        'High':  [17.0, 15.0, 13.0],
        'Low':   [15.0, 11.0, 10.0],
        'Close': [15.5, 12.0, 11.0]
    }
    df = pd.DataFrame(data)
    
    result = detect_fvg_and_ob(df, symbol="XAUUSD")
    
    # Assert FVG type
    assert result['FVG_Type'].iloc[2] == 'BEARISH'
    
    # Assert Fibonacci Levels
    expected_fibo_1_0 = 15.0  # High of Candle 1 (index i-1)
    expected_fibo_0_0 = 11.0  # Low of Candle 1 (index i-1)
    expected_fibo_0_5 = 13.0  # 11.0 + 0.5 * (15.0 - 11.0)
    expected_fibo_0_618 = 13.472 # 11.0 + 0.618 * (15.0 - 11.0)
    expected_sl = 17.0        # 15.0 + 2.0 (XAUUSD buffer = 20 * 0.1, boundary = Low of Candle 0 = 15.0)
    
    assert np.isclose(result['FVG_Fibo_1.0'].iloc[2], expected_fibo_1_0)
    assert np.isclose(result['FVG_Fibo_0.0'].iloc[2], expected_fibo_0_0)
    assert np.isclose(result['FVG_Fibo_0.5'].iloc[2], expected_fibo_0_5)
    assert np.isclose(result['FVG_Fibo_0.618'].iloc[2], expected_fibo_0_618)
    assert np.isclose(result['FVG_SL'].iloc[2], expected_sl)

def test_bullish_fvg_fibo_large_candle2():
    """Test FVG Fibonacci calculations for large Candle 2 (no fallback)."""
    data = {
        'Open':  [9.0, 10.0, 29.0],
        'High':  [10.0, 30.0, 32.0],
        'Low':   [8.0, 10.0, 12.0],
        'Close': [9.5, 29.0, 30.0]
    }
    df = pd.DataFrame(data)
    
    result = detect_fvg_and_ob(df, symbol="XAUUSD")
    
    assert result['FVG_Type'].iloc[2] == 'BULLISH'
    
    # Expected levels (since Candle 1 range is 20.0 = 200 pips, and fallback is disabled)
    expected_fibo_1_0 = 10.0  # Low of Candle 1
    expected_fibo_0_0 = 30.0  # High of Candle 1
    expected_fibo_0_5 = 20.0  # 30.0 - 0.5 * 20.0
    expected_fibo_0_618 = 17.64 # 30.0 - 0.618 * 20.0
    expected_sl = 8.0        # 10.0 - 2.0 (High of Candle 0 - buffer)
    
    assert np.isclose(result['FVG_Fibo_1.0'].iloc[2], expected_fibo_1_0)
    assert np.isclose(result['FVG_Fibo_0.0'].iloc[2], expected_fibo_0_0)
    assert np.isclose(result['FVG_Fibo_0.5'].iloc[2], expected_fibo_0_5)
    assert np.isclose(result['FVG_Fibo_0.618'].iloc[2], expected_fibo_0_618)
    assert np.isclose(result['FVG_SL'].iloc[2], expected_sl)


def test_ob_fibo_1candle():
    """Test 1-candle Fibonacci calculations for a Bullish OB."""
    df = pd.DataFrame({
        'Open':  [100.0, 102.0, 101.0, 103.0],
        'High':  [101.0, 102.5, 103.2, 104.0],
        'Low':   [99.0,  100.5, 100.8, 102.0],
        'Close': [100.5, 101.0, 103.0, 103.5]
    })
    
    # OB candle is at index 1: Close 101.0 < Open 102.0. High = 102.5, Low = 100.5
    # Trigger candle is at index 2: Close 103.0 > Open 101.0 (Bullish BOS/CHoCH)
    
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan
    df['BOS'] = np.nan
    df['CHoCH'] = np.nan
    df['Trend'] = 1
    
    # Manually inject BOS at index 2
    df.loc[2, 'BOS'] = 102.5
    
    result = detect_fvg_and_ob(df, symbol="XAUUSD")
    
    assert result['OB_Type'].iloc[2] == 'BULLISH'
    # 1-candle OB wicks: High = 102.5, Low = 100.5
    # Fibo levels:
    # 0.50 of OB candle = 102.5 - 0.5 * (102.5 - 100.5) = 101.5
    # 0.618 of OB candle = 102.5 - 0.618 * (102.5 - 100.5) = 101.264
    # SL = Low - buffer (100.5 - 2.0 = 98.5)
    # TP (Fibo 0.0) = High of trigger candle (103.2)
    
    assert np.isclose(result['OB_Fibo_0.5'].iloc[2], 101.5)
    assert np.isclose(result['OB_Fibo_0.618'].iloc[2], 101.264)
    assert np.isclose(result['OB_SL'].iloc[2], 98.5)
    assert np.isclose(result['OB_Fibo_0.0'].iloc[2], 103.2)


def test_swapzone_fibo_1candle():
    """Test 1-candle Fibonacci calculations for a Support Swapzone (broken Resistance)."""
    df = pd.DataFrame({
        'Open':  [100.0, 101.0, 102.0, 101.0, 100.0, 101.0, 103.5],
        'High':  [101.0, 103.0, 102.5, 102.0, 101.0, 102.5, 104.5],
        'Low':   [99.0,  100.5, 101.5, 100.5, 99.5,  100.0, 103.0],
        'Close': [100.5, 102.0, 101.0, 101.5, 100.5, 102.0, 104.0]
    })
    
    # Swing high at index 1: High = 103.0, Low = 100.5
    # Breakout candle at index 6: Close = 104.0 > 103.0 (breaks resistance)
    
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan
    df.loc[1, 'Swing_High'] = 103.0
    
    result = detect_snr_and_swapzones(df, symbol="XAUUSD")
    
    assert result['Swap_Type'].iloc[6] == 'SUPPORT'
    assert result['Swap_Level'].iloc[6] == 103.0
    
    # 1-candle zone of Swing High at index 1: High = 103.0, Low = 100.5
    # Fibo levels:
    # 0.50 of Swing candle = 103.0 - 0.5 * (103.0 - 100.5) = 101.75
    # 0.618 of Swing candle = 103.0 - 0.618 * (103.0 - 100.5) = 101.455
    # SL = Low - buffer (100.5 - 2.0 = 98.5)
    # TP (Fibo 0.0) = High of breakout candle (104.5)
    
    assert np.isclose(result['Swap_Fibo_0.5'].iloc[6], 101.75)
    assert np.isclose(result['Swap_Fibo_0.618'].iloc[6], 101.455)
    assert np.isclose(result['Swap_SL'].iloc[6], 98.5)
    assert np.isclose(result['Swap_Fibo_0.0'].iloc[6], 104.5)


def test_indecision_candle_fibo():
    """Test Fibonacci calculations for a Bearish Indecision Candle setup."""
    df = pd.DataFrame({
        'Open':  [100.0, 102.0, 102.1, 101.0],
        'High':  [101.0, 104.0, 102.5, 99.5],
        'Low':   [99.0,  100.0, 101.0, 98.0],
        'Close': [100.5, 102.1, 101.5, 98.5]
    })
    
    # Candle 1 (index 1) is an Indecision Candle:
    # Open = 102.0, Close = 102.1. Body = 0.1
    # High = 104.0, Low = 100.0. Range = 4.0
    # Body/Range = 0.1 / 4.0 = 2.5% <= 25% (Indecision)
    
    # Candle 3 (index 3) is a Bearish Breakout:
    # Close = 98.5 < Low of index 1 (100.0)
    
    result = detect_indecision_candles(df, body_ratio=0.25, symbol="XAUUSD")
    
    assert result['IC_Type'].iloc[3] == 'BEARISH'
    # 1-candle Indecision wicks: High = 104.0, Low = 100.0
    # Fibo levels:
    # 0.50 of IC candle = 100.0 + 0.5 * (104.0 - 100.0) = 102.0
    # 0.618 of IC candle = 100.0 + 0.618 * (104.0 - 100.0) = 102.472
    # SL = High + buffer (104.0 + 2.0 = 106.0)
    # TP (Fibo 0.0) = Low of breakout candle (98.0)
    
    assert np.isclose(result['IC_Fibo_0.5'].iloc[3], 102.0)
    assert np.isclose(result['IC_Fibo_0.618'].iloc[3], 102.472)
    assert np.isclose(result['IC_SL'].iloc[3], 106.0)
    assert np.isclose(result['IC_Fibo_0.0'].iloc[3], 98.0)



