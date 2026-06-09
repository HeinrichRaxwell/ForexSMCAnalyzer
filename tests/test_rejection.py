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

def test_bullish_engulfing_rejection():
    # Candle 0 touches level 2340 but has no wick rejection
    # Candle 1 is bullish and engulfs candle 0
    df = pd.DataFrame([
        {'Open': 2342.0, 'High': 2343.0, 'Low': 2339.5, 'Close': 2341.0}, # Bearish touch at 2340
        {'Open': 2340.5, 'High': 2345.0, 'Low': 2340.0, 'Close': 2344.0}  # Bullish engulfing (close 2344 > open 2342)
    ])
    assert detect_rejection_at_level(df, entry_level=2340.0, direction=1, lookback=2) is True

def test_bearish_engulfing_rejection():
    # Candle 0 touches level 2350
    # Candle 1 is bearish and engulfs candle 0
    df = pd.DataFrame([
        {'Open': 2348.0, 'High': 2351.0, 'Low': 2347.0, 'Close': 2349.5}, # Bullish touch at 2350
        {'Open': 2349.8, 'High': 2350.0, 'Low': 2344.0, 'Close': 2345.0}  # Bearish engulfing (close 2345 < open 2348)
    ])
    assert detect_rejection_at_level(df, entry_level=2350.0, direction=-1, lookback=2) is True

def test_double_touch_rejection():
    # Candle 0 touches level 2340
    # Candle 1 is far (no touch) and closes above 2340 (no body break)
    # Candle 2 touches level 2340 again
    df = pd.DataFrame([
        {'Open': 2342.0, 'High': 2343.0, 'Low': 2339.5, 'Close': 2341.0}, # Touch
        {'Open': 2341.5, 'High': 2344.0, 'Low': 2341.0, 'Close': 2343.0}, # No touch, body above
        {'Open': 2343.0, 'High': 2344.0, 'Low': 2339.8, 'Close': 2342.0}  # Touch again
    ])
    assert detect_rejection_at_level(df, entry_level=2340.0, direction=1, lookback=3) is True

def test_psychological_price_helpers():
    from src.rejection_detector import get_nearest_psychological_level, is_near_psychological_level
    # Check nearest multiples of 5
    assert get_nearest_psychological_level(2338.2) == 2340.0
    assert get_nearest_psychological_level(2342.3) == 2340.0
    assert get_nearest_psychological_level(2342.6) == 2345.0
    
    # Check near psychological levels within 10 pips (for XAUUSD, pip=0.1, threshold=1.0 USD)
    assert is_near_psychological_level(2340.8, "XAUUSD") is True
    assert is_near_psychological_level(2339.2, "XAUUSD") is True
    assert is_near_psychological_level(2341.5, "XAUUSD") is False
    assert is_near_psychological_level(2344.2, "XAUUSD") is True

