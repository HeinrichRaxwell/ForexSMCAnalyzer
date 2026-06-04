import pytest
import pandas as pd
import numpy as np
from src.smc_detector import detect_fvg_and_ob
from src.main import get_active_setups, find_dynamic_tp
from src.rejection_detector import detect_rejection_at_level

def test_fvg_double_options_generation():
    # Construct a dataframe with a Bullish FVG at index 2 (last candle, so unmitigated)
    # FVG zone: between Candle 0 High (10.0) and Candle 2 Low (12.0)
    # Fibo 1.0 (Low of Candle 0) = 8.0
    # Fibo 0.0 (High of Candle 2) = 16.0
    # Fvg SL = 8.0 - 20 * 0.1 = 6.0
    # Fibo 0.5 = 16.0 - 0.5 * (16.0 - 8.0) = 12.0
    # Fibo 0.618 = 16.0 - 0.618 * (16.0 - 8.0) = 11.056
    data = [
        {'time': '2026-06-01 09:00:00', 'Open': 9.0, 'High': 10.0, 'Low': 8.0, 'Close': 9.5, 'Volume': 100, 'Trend': 1, 'OB_Type': None, 'OB_Mitigated': False},
        {'time': '2026-06-01 09:15:00', 'Open': 10.0, 'High': 15.0, 'Low': 10.0, 'Close': 14.0, 'Volume': 100, 'Trend': 1, 'OB_Type': None, 'OB_Mitigated': False},
        {'time': '2026-06-01 09:30:00', 'Open': 13.0, 'High': 16.0, 'Low': 12.0, 'Close': 15.0, 'Volume': 100, 'Trend': 1, 'OB_Type': None, 'OB_Mitigated': False},
    ]
    df = pd.DataFrame(data)
    
    # Run detection
    df = detect_fvg_and_ob(df, symbol="XAUUSD")
    
    # Run active setups extraction
    setups = get_active_setups(df)
    
    # We expect 2 setups: Option A and Option B for the FVG at index 2
    fvg_setups = [s for s in setups if s['setup_type'] == 0]
    assert len(fvg_setups) == 2
    
    opt_a = [s for s in fvg_setups if 'Option A' in s['option_name']][0]
    opt_b = [s for s in fvg_setups if 'Option B' in s['option_name']][0]
    
    # Option A (Midpoint): Entry = Fibo 0.5 (12.5), SL = FVG_SL (8.0), TP 1 = Fibo 0.0 (15.0)
    assert np.isclose(opt_a['entry_price'], 12.5)
    assert np.isclose(opt_a['sl_price'], 8.0)
    assert np.isclose(opt_a['tp_price'], 15.0)
    assert opt_a['rejection_confirmed'] is False # No touch yet
    
    # Option B (Golden Pocket): Entry = Fibo 0.618 (11.91), SL = FVG_SL (8.0), TP 1 = Fibo 0.0 (15.0)
    assert np.isclose(opt_b['entry_price'], 11.91)
    assert np.isclose(opt_b['sl_price'], 8.0)
    assert np.isclose(opt_b['tp_price'], 15.0)
    assert opt_b['rejection_confirmed'] is False # No touch yet

def test_rejection_detection_at_fibo_levels():
    # Construct a dataframe where we manually check rejection on the last candle
    # Touch at 12.0 (Fibo 0.5)
    # Candle 3: Low = 11.0, High = 15.0, body min = 13.0, Open/Close = 13.0/14.0
    # Lower shadow = 13.0 - 11.0 = 2.0. Total range = 15.0 - 11.0 = 4.0. Lower shadow ratio = 0.5 >= 0.5.
    data = [
        {'Open': 13.0, 'High': 15.0, 'Low': 11.0, 'Close': 14.0}
    ]
    df = pd.DataFrame(data)
    
    # Bullish rejection at 12.0 should be True
    assert detect_rejection_at_level(df, entry_level=12.0, direction=1) is True
    # Bullish rejection at 11.056 should be True
    assert detect_rejection_at_level(df, entry_level=11.056, direction=1) is True
    # Bullish rejection at 14.5 should be False (above body_max)
    assert detect_rejection_at_level(df, entry_level=14.5, direction=1) is False

def test_dynamic_tp_opposite_structure():
    # Setup dataframe containing a Bearish FVG as opposite target
    # Index 2: Bearish FVG: Low of Candle 0 (15.0) > High of Candle 2 (13.0)
    # FVG bottom = 13.0 (Bearish FVG)
    data = [
        {'time': '2026-06-01 09:00:00', 'Open': 16.0, 'High': 17.0, 'Low': 15.0, 'Close': 15.5, 'OB_Type': None, 'OB_Mitigated': False},
        {'time': '2026-06-01 09:15:00', 'Open': 15.0, 'High': 15.0, 'Low': 11.0, 'Close': 12.0, 'OB_Type': None, 'OB_Mitigated': False},
        {'time': '2026-06-01 09:30:00', 'Open': 12.0, 'High': 13.0, 'Low': 10.0, 'Close': 11.0, 'OB_Type': None, 'OB_Mitigated': False},
    ]
    df = pd.DataFrame(data)
    df = detect_fvg_and_ob(df, symbol="XAUUSD")
    
    # Let's verify Bearish FVG is detected
    assert df['FVG_Type'].iloc[2] == 'BEARISH'
    fvg_bottom = df['FVG_Bottom'].iloc[2] # Should be 13.0
    assert np.isclose(fvg_bottom, 13.0)
    
    # Now check if find_dynamic_tp finds this Bearish FVG for a Bullish setup with entry < 13.0
    tp_dynamic = find_dynamic_tp(df, entry_price=12.0, direction=1)
    assert np.isclose(tp_dynamic, 13.0)
    
    # If entry_price is above 13.0, it shouldn't match (since it must be above entry_price for Buy)
    tp_dynamic_higher = find_dynamic_tp(df, entry_price=14.0, direction=1)
    assert tp_dynamic_higher is None
