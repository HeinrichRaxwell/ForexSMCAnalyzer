import pytest
import pandas as pd
import numpy as np
from src.smc_detector import detect_swing_points

def test_detect_swing_points_basic():
    """Test swing points detection on a simple, known dataset."""
    # Create a simple trend with a peak at index 3 (value 10) and a trough at index 7 (value 2)
    data = {
        'High': [5, 6, 7, 10, 8, 7, 6, 5, 4, 3, 2],
        'Low':  [3, 4, 5,  8, 6, 5, 4, 2, 3, 1, 2]
    }
    df = pd.DataFrame(data)
    
    # Run detector with window=5 (2 left, 2 right)
    result = detect_swing_points(df, window=5)
    
    # Assert columns exist
    assert 'Swing_High' in result.columns
    assert 'Swing_Low' in result.columns
    
    # Peak at index 3: High[3]=10 is max of indices 1, 2, 3, 4, 5 ([6, 7, 10, 8, 7])
    assert result['Swing_High'].iloc[3] == 10
    # Other values near should be NaN for Swing_High
    assert pd.isna(result['Swing_High'].iloc[2])
    assert pd.isna(result['Swing_High'].iloc[4])
    
    # Trough at index 7: Low[7]=2 is min of indices 5, 6, 7, 8, 9 ([5, 4, 2, 3, 1])
    # Wait, indices 5 to 9 are Lows: [5, 4, 2, 3, 1]. The min is index 9 (value 1).
    # Since index 9 is near the edge (len is 11, half window is 2, index 9 is at boundary len - half - 1),
    # let's look at the window around index 7: 5,6,7,8,9 -> min is 1 at index 9. So index 7 is NOT a swing low in window 5.
    # Let's check index 9: window is 7,8,9,10. But i goes from 2 to 8. So index 9 is never evaluated as a center.
    # Let's construct a cleaner data set where peak and trough are fully inside the loop range.
    
def test_detect_swing_points_clear_peaks():
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    highs = [1, 2, 3, 2, 1, 5, 1, 2, 10, 2, 1, 1]
    lows =  [1, 1, 0, 1, 1, 1, 1, 0,  2, 1, 1, 1]
    df = pd.DataFrame({'High': highs, 'Low': lows})
    
    result = detect_swing_points(df, window=5)
    
    # Swing High at index 2 (val 3): window 0..4, highs = [1, 2, 3, 2, 1]. 3 is max.
    assert result['Swing_High'].iloc[2] == 3
    # Swing High at index 5 (val 5): window 3..7, highs = [2, 1, 5, 1, 2]. 5 is max.
    assert result['Swing_High'].iloc[5] == 5
    # Swing High at index 8 (val 10): window 6..10, highs = [1, 2, 10, 2, 1]. 10 is max.
    assert result['Swing_High'].iloc[8] == 10
    
    # Swing Low at index 2 (val 0): window 0..4, lows = [1, 1, 0, 1, 1]. 0 is min.
    assert result['Swing_Low'].iloc[2] == 0
    # Swing Low at index 7 (val 0): window 5..9, lows = [1, 1, 0, 2, 1]. 0 is min.
    assert result['Swing_Low'].iloc[7] == 0
