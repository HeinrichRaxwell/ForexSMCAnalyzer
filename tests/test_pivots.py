import pandas as pd
import numpy as np
import pytest
from src.indicators.pivots import calculate_daily_pivots, align_daily_pivots, get_pivot_features_at_idx

def test_daily_pivots():
    # Generate mock daily data
    dates = pd.date_range(start="2026-06-01", periods=5, freq="1D")
    high = [100.0, 105.0, 110.0, 108.0, 106.0]
    low = [90.0, 95.0, 100.0, 98.0, 96.0]
    close = [95.0, 100.0, 105.0, 102.0, 100.0]
    
    df_d1 = pd.DataFrame({
        'time': dates,
        'High': high,
        'Low': low,
        'Close': close
    })
    
    pivots_df = calculate_daily_pivots(df_d1)
    
    assert len(pivots_df) == 5
    assert 'pivot_PP' in pivots_df.columns
    assert 'pivot_R1' in pivots_df.columns
    assert 'pivot_S1' in pivots_df.columns
    
    # Row 0 pivots should be NaN (shifted)
    assert pd.isna(pivots_df['pivot_PP'].iloc[0])
    
    # Row 1 pivots should be based on Row 0 daily bar: H=100, L=90, C=95
    # PP = (100 + 90 + 95) / 3 = 95.0
    # R1 = 2 * 95 - 90 = 100
    # S1 = 2 * 95 - 100 = 90
    assert np.isclose(pivots_df['pivot_PP'].iloc[1], 95.0)
    assert np.isclose(pivots_df['pivot_R1'].iloc[1], 100.0)
    assert np.isclose(pivots_df['pivot_S1'].iloc[1], 90.0)

def test_align_daily_pivots():
    # Daily data
    dates_d1 = pd.date_range(start="2026-06-01", periods=3, freq="1D")
    df_d1 = pd.DataFrame({
        'time': dates_d1,
        'High': [100.0, 105.0, 110.0],
        'Low': [90.0, 95.0, 100.0],
        'Close': [95.0, 100.0, 105.0]
    })
    
    # Lower timeframe data (hourly) for June 2nd
    dates_ltf = pd.date_range(start="2026-06-02 00:00:00", periods=5, freq="1h")
    df_ltf = pd.DataFrame({
        'time': dates_ltf,
        'High': [98.0] * 5,
        'Low': [94.0] * 5,
        'Close': [96.0] * 5
    })
    
    df_aligned = align_daily_pivots(df_ltf, df_d1)
    
    assert len(df_aligned) == 5
    assert 'pivot_PP' in df_aligned.columns
    
    # Since they are on June 2nd, their daily pivots should be calculated from June 1st D1 bar:
    # PP = 95.0, R1 = 100.0, S1 = 90.0
    assert (df_aligned['pivot_PP'] == 95.0).all()
    assert (df_aligned['pivot_R1'] == 100.0).all()
    assert (df_aligned['pivot_S1'] == 90.0).all()

def test_get_pivot_features():
    df = pd.DataFrame({
        'pivot_PP': [95.0],
        'pivot_R1': [100.0],
        'pivot_R2': [105.0],
        'pivot_R3': [110.0],
        'pivot_R4': [115.0],
        'pivot_S1': [90.0],
        'pivot_S2': [85.0],
        'pivot_S3': [80.0],
        'pivot_S4': [75.0]
    })
    
    # Entry price is 97.0
    # dist_to_pp = (97.0 - 95.0) / 95.0 = 2/95
    # nearest level is R1 (100.0). dist = abs(97 - 100) / 97 = 3/97
    features = get_pivot_features_at_idx(df, 0, 97.0)
    
    assert np.isclose(features['dist_entry_to_pp'], 2.0 / 95.0)
    assert np.isclose(features['dist_entry_to_nearest_pivot'], 2.0 / 97.0)
