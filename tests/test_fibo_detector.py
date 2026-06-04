import pytest
import pandas as pd
import numpy as np
from src.smc_detector import detect_fvg_and_ob

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
    expected_fibo_1_0 = 8.0  # Low of Candle 0
    expected_fibo_0_0 = 16.0 # High of Candle 2
    expected_fibo_0_5 = 12.0 # 16.0 - 0.5 * (16.0 - 8.0)
    expected_fibo_0_618 = 11.056 # 16.0 - 0.618 * (16.0 - 8.0)
    expected_sl = 6.0        # 8.0 - 2.0 (XAUUSD buffer = 20 * 0.1)
    
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
    
    expected_sl = 7.8        # 8.0 - 0.2 (USDJPY buffer = 20 * 0.01)
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
    
    expected_sl = 7.998      # 8.0 - 0.002 (EURUSD buffer = 20 * 0.0001)
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
    expected_fibo_1_0 = 17.0  # High of Candle 0
    expected_fibo_0_0 = 10.0  # Low of Candle 2
    expected_fibo_0_5 = 13.5  # 10.0 + 0.5 * (17.0 - 10.0)
    expected_fibo_0_618 = 14.326 # 10.0 + 0.618 * (17.0 - 10.0)
    expected_sl = 19.0        # 17.0 + 2.0 (XAUUSD buffer = 20 * 0.1)
    
    assert np.isclose(result['FVG_Fibo_1.0'].iloc[2], expected_fibo_1_0)
    assert np.isclose(result['FVG_Fibo_0.0'].iloc[2], expected_fibo_0_0)
    assert np.isclose(result['FVG_Fibo_0.5'].iloc[2], expected_fibo_0_5)
    assert np.isclose(result['FVG_Fibo_0.618'].iloc[2], expected_fibo_0_618)
    assert np.isclose(result['FVG_SL'].iloc[2], expected_sl)
