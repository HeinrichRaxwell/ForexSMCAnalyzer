import pytest
import pandas as pd
from src.rejection_detector import detect_rejection_at_level

def test_bullish_rejection_success():
    # Lower shadow = 1.080 - 1.000 = 0.080
    # Total range = 1.120 - 1.000 = 0.120
    # Lower shadow / Total range = 0.080 / 0.120 = 2/3 (0.667) >= 0.5
    # Touch check: Low=1.000 <= 1.020 <= body_max=1.100 (True)
    df = pd.DataFrame([{
        'Open': 1.100,
        'High': 1.120,
        'Low': 1.000,
        'Close': 1.080
    }])
    assert detect_rejection_at_level(df, entry_level=1.020, direction=1) is True

def test_bullish_rejection_no_touch():
    df = pd.DataFrame([{
        'Open': 1.100,
        'High': 1.120,
        'Low': 1.000,
        'Close': 1.080
    }])
    assert detect_rejection_at_level(df, entry_level=0.990, direction=1) is False
    assert detect_rejection_at_level(df, entry_level=1.110, direction=1) is False

def test_bullish_rejection_small_wick():
    # Lower shadow = 1.080 - 1.060 = 0.020
    # Total range = 1.120 - 1.060 = 0.060
    # Lower shadow / Total range = 0.020 / 0.060 = 1/3 (0.333) < 0.5
    df = pd.DataFrame([{
        'Open': 1.100,
        'High': 1.120,
        'Low': 1.060,
        'Close': 1.080
    }])
    assert detect_rejection_at_level(df, entry_level=1.070, direction=1) is False

def test_bearish_rejection_success():
    # Upper shadow = 1.120 - 1.040 = 0.080
    # Total range = 1.120 - 1.000 = 0.120
    # Upper shadow / Total range = 0.080 / 0.120 = 2/3 (0.667) >= 0.5
    # Touch check: body_min=1.020 <= 1.100 <= High=1.120 (True)
    df = pd.DataFrame([{
        'Open': 1.020,
        'High': 1.120,
        'Low': 1.000,
        'Close': 1.040
    }])
    assert detect_rejection_at_level(df, entry_level=1.100, direction=-1) is True

def test_bearish_rejection_no_touch():
    df = pd.DataFrame([{
        'Open': 1.020,
        'High': 1.120,
        'Low': 1.000,
        'Close': 1.040
    }])
    assert detect_rejection_at_level(df, entry_level=1.130, direction=-1) is False
    assert detect_rejection_at_level(df, entry_level=1.010, direction=-1) is False

def test_bearish_rejection_small_wick():
    # Upper shadow = 1.060 - 1.040 = 0.020
    # Total range = 1.060 - 1.000 = 0.060
    # Upper shadow / Total range = 0.020 / 0.060 = 1/3 (0.333) < 0.5
    df = pd.DataFrame([{
        'Open': 1.020,
        'High': 1.060,
        'Low': 1.000,
        'Close': 1.040
    }])
    assert detect_rejection_at_level(df, entry_level=1.050, direction=-1) is False

def test_rejection_lookback_limit():
    # 6 candles
    # Candle at index 0 (oldest) has a valid rejection at level 1.020
    # Candle at indices 1 to 5 have no touch at 1.020
    candles = [
        {'Open': 1.100, 'High': 1.120, 'Low': 1.000, 'Close': 1.080}, # Rejection here
        {'Open': 1.100, 'High': 1.120, 'Low': 1.050, 'Close': 1.080},
        {'Open': 1.100, 'High': 1.120, 'Low': 1.050, 'Close': 1.080},
        {'Open': 1.100, 'High': 1.120, 'Low': 1.050, 'Close': 1.080},
        {'Open': 1.100, 'High': 1.120, 'Low': 1.050, 'Close': 1.080},
        {'Open': 1.100, 'High': 1.120, 'Low': 1.050, 'Close': 1.080},
    ]
    df = pd.DataFrame(candles)
    
    # Lookback 5 checks indices 1-5 (no rejection found)
    assert detect_rejection_at_level(df, entry_level=1.020, direction=1, lookback=5) is False
    
    # Lookback 6 checks indices 0-5 (rejection found at index 0)
    assert detect_rejection_at_level(df, entry_level=1.020, direction=1, lookback=6) is True

def test_rejection_empty_and_invalid():
    # Empty
    df_empty = pd.DataFrame()
    assert detect_rejection_at_level(df_empty, entry_level=1.020, direction=1) is False
    
    # Range <= 0
    df_flat = pd.DataFrame([{
        'Open': 1.000,
        'High': 1.000,
        'Low': 1.000,
        'Close': 1.000
    }])
    assert detect_rejection_at_level(df_flat, entry_level=1.000, direction=1) is False
