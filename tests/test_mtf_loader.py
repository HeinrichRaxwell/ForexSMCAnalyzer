import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import functions from main.py
from src.main import extract_active_htf_fvgs, generate_synthetic_data

def test_extract_active_htf_fvgs_bullish_active():
    """Test that an unmitigated bullish HTF FVG is correctly identified as active."""
    # Bullish FVG occurs at index 2 (Low[2] > High[0] and Close[1] > Open[1])
    # FVG_Top = 15.0, FVG_Bottom = 11.0
    data = {
        'time': pd.date_range(start="2026-06-01", periods=5, freq="1h"),
        'Open':  [10.0, 11.0, 14.0, 16.0, 15.0],
        'High':  [11.0, 15.0, 17.0, 18.0, 16.0],
        'Low':   [9.0, 10.0, 15.0, 14.0, 13.0],
        'Close': [10.5, 14.0, 16.0, 15.0, 14.0],
        'FVG_Type': [None, None, 'BULLISH', None, None],
        'FVG_Top': [np.nan, np.nan, 15.0, np.nan, np.nan],
        'FVG_Bottom': [np.nan, np.nan, 11.0, np.nan, np.nan]
    }
    df = pd.DataFrame(data)
    
    active_fvgs = extract_active_htf_fvgs(df)
    assert len(active_fvgs) == 1
    assert active_fvgs[0]['type'] == 'BULLISH'
    assert active_fvgs[0]['top'] == 15.0
    assert active_fvgs[0]['bottom'] == 11.0

def test_extract_active_htf_fvgs_bullish_mitigated():
    """Test that a mitigated bullish HTF FVG (closed below FVG_Bottom) is identified as inactive."""
    # Bullish FVG occurs at index 2, FVG_Bottom = 11.0.
    # At index 4, the Close is 10.0 (below FVG_Bottom).
    data = {
        'time': pd.date_range(start="2026-06-01", periods=5, freq="1h"),
        'Open':  [10.0, 11.0, 14.0, 13.0, 12.0],
        'High':  [11.0, 15.0, 17.0, 14.0, 13.0],
        'Low':   [9.0, 10.0, 15.0, 11.0, 9.5],
        'Close': [10.5, 14.0, 16.0, 12.0, 10.0],
        'FVG_Type': [None, None, 'BULLISH', None, None],
        'FVG_Top': [np.nan, np.nan, 15.0, np.nan, np.nan],
        'FVG_Bottom': [np.nan, np.nan, 11.0, np.nan, np.nan]
    }
    df = pd.DataFrame(data)
    
    active_fvgs = extract_active_htf_fvgs(df)
    assert len(active_fvgs) == 0

def test_extract_active_htf_fvgs_bearish_active():
    """Test that an unmitigated bearish HTF FVG is correctly identified as active."""
    # Bearish FVG occurs at index 2 (Low[0] > High[2] and Close[1] < Open[1])
    # FVG_Top = 15.0, FVG_Bottom = 11.0
    data = {
        'time': pd.date_range(start="2026-06-01", periods=5, freq="1h"),
        'Open':  [16.0, 15.0, 12.0, 10.0, 11.0],
        'High':  [17.0, 15.0, 11.0, 12.0, 13.0],
        'Low':   [15.0, 11.0, 9.0,  9.0,  10.0],
        'Close': [15.5, 12.0, 10.0, 11.0, 12.0],
        'FVG_Type': [None, None, 'BEARISH', None, None],
        'FVG_Top': [np.nan, np.nan, 15.0, np.nan, np.nan],
        'FVG_Bottom': [np.nan, np.nan, 11.0, np.nan, np.nan]
    }
    df = pd.DataFrame(data)
    
    active_fvgs = extract_active_htf_fvgs(df)
    assert len(active_fvgs) == 1
    assert active_fvgs[0]['type'] == 'BEARISH'
    assert active_fvgs[0]['top'] == 15.0
    assert active_fvgs[0]['bottom'] == 11.0

def test_extract_active_htf_fvgs_bearish_mitigated():
    """Test that a mitigated bearish HTF FVG (closed above FVG_Top) is identified as inactive."""
    # Bearish FVG occurs at index 2, FVG_Top = 15.0.
    # At index 4, the Close is 16.0 (above FVG_Top).
    data = {
        'time': pd.date_range(start="2026-06-01", periods=5, freq="1h"),
        'Open':  [16.0, 15.0, 12.0, 14.0, 15.0],
        'High':  [17.0, 15.0, 11.0, 16.0, 17.0],
        'Low':   [15.0, 11.0, 9.0,  13.0, 14.0],
        'Close': [15.5, 12.0, 10.0, 15.0, 16.0],
        'FVG_Type': [None, None, 'BEARISH', None, None],
        'FVG_Top': [np.nan, np.nan, 15.0, np.nan, np.nan],
        'FVG_Bottom': [np.nan, np.nan, 11.0, np.nan, np.nan]
    }
    df = pd.DataFrame(data)
    
    active_fvgs = extract_active_htf_fvgs(df)
    assert len(active_fvgs) == 0

def test_ltf_setup_prioritization():
    """Test the overlap logic that tags lower timeframe setups as htf_prioritized."""
    # Active HTF FVGs list
    active_htf_fvgs = [
        {'timeframe': 'H1', 'type': 'BULLISH', 'bottom': 2340.0, 'top': 2350.0},
        {'timeframe': 'H4', 'type': 'BEARISH', 'bottom': 2360.0, 'top': 2370.0}
    ]
    
    # 1. Bullish setup inside Bullish HTF FVG -> should be prioritized
    setup_1 = {
        'direction': 1,  # Bullish
        'entry_price': 2345.0,
        'setup_type': 0  # FVG
    }
    
    # 2. Bullish setup outside Bullish HTF FVG -> should NOT be prioritized
    setup_2 = {
        'direction': 1,
        'entry_price': 2355.0,
        'setup_type': 0
    }
    
    # 3. Bearish setup inside Bearish HTF FVG -> should be prioritized
    setup_3 = {
        'direction': -1,  # Bearish
        'entry_price': 2365.0,
        'setup_type': 1  # OB
    }
    
    # 4. Bullish setup inside Bearish HTF FVG -> should NOT be prioritized (wrong direction)
    setup_4 = {
        'direction': 1,
        'entry_price': 2365.0,
        'setup_type': 1
    }
    
    setups = [setup_1, setup_2, setup_3, setup_4]
    
    # Apply prioritization logic (mimicking src/main.py)
    for setup in setups:
        setup['htf_prioritized'] = False
        setup['matching_htf_fvgs'] = []
        for htf_fvg in active_htf_fvgs:
            is_same_direction = (setup['direction'] == 1 and htf_fvg['type'] == 'BULLISH') or \
                                (setup['direction'] == -1 and htf_fvg['type'] == 'BEARISH')
            if is_same_direction:
                entry = setup['entry_price']
                if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                    setup['htf_prioritized'] = True
                    setup['matching_htf_fvgs'].append(htf_fvg)
                    
    assert setup_1['htf_prioritized'] is True
    assert len(setup_1['matching_htf_fvgs']) == 1
    assert setup_1['matching_htf_fvgs'][0]['timeframe'] == 'H1'
    
    assert setup_2['htf_prioritized'] is False
    assert len(setup_2['matching_htf_fvgs']) == 0
    
    assert setup_3['htf_prioritized'] is True
    assert len(setup_3['matching_htf_fvgs']) == 1
    assert setup_3['matching_htf_fvgs'][0]['timeframe'] == 'H4'
    
    assert setup_4['htf_prioritized'] is False
    assert len(setup_4['matching_htf_fvgs']) == 0

def test_synthetic_data_generation():
    """Test that synthetic data is generated correctly and contains the expected length and columns."""
    df = generate_synthetic_data(60, seed=100)
    assert len(df) == 60
    assert 'Open' in df.columns
    assert 'High' in df.columns
    assert 'Low' in df.columns
    assert 'Close' in df.columns
