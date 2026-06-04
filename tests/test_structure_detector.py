import pytest
import pandas as pd
import numpy as np
from src.smc_detector import detect_structures

def test_detect_structures():
    """Test BOS and CHoCH detection on a custom structured dataframe."""
    # Construct a sequence of events
    # We will provide Swing_High and Swing_Low directly.
    data = {
        'Close':      [10, 12, 14, 11, 15,  9,  8,  5,  6,  4, 12],
        'Swing_High': [14, np.nan, np.nan, np.nan, 15, np.nan, np.nan, np.nan, np.nan, np.nan, 12],
        'Swing_Low':  [np.nan, 10, np.nan, 11, np.nan, np.nan, np.nan, 5, np.nan, 4, np.nan]
    }
    df = pd.DataFrame(data)
    
    result = detect_structures(df)
    
    # Assert columns
    assert 'BOS' in result.columns
    assert 'CHoCH' in result.columns
    assert 'Trend' in result.columns
    
    # Trace execution:
    # idx 0: Close 10, Swing_High 14, Swing_Low NaN. last_high becomes 14. Trend = 1.
    # idx 1: Close 12, Swing_Low 10. last_low becomes 10. Trend = 1.
    # idx 2: Close 14. Close not > 14 (not > last_high). Close not < 10. Trend = 1.
    # idx 3: Close 11. Swing_Low 11. Close not > 14, not < 10. last_low becomes 11. Trend = 1.
    # idx 4: Close 15. Close > last_high (15 > 14). Bullish BOS detected! BOS level = 14. last_high reset to None. Swing_High 15. last_high becomes 15. Trend = 1.
    assert result['BOS'].iloc[4] == 14
    assert result['Trend'].iloc[4] == 1
    
    # idx 5: Close 9. Close < last_low (9 < 11). Bearish CHoCH detected! CHoCH level = 11. Trend shifts to -1. last_low reset to None.
    assert result['CHoCH'].iloc[5] == 11
    assert result['Trend'].iloc[5] == -1
    
    # idx 6: Close 8. Trend = -1. last_low is None, last_high is 15.
    # idx 7: Close 5. Swing_Low 5. last_low becomes 5. Trend = -1.
    # idx 8: Close 6. Close not < 5, close not > 15. Trend = -1.
    # idx 9: Close 4. Close < last_low (4 < 5). Bearish BOS detected! BOS level = 5. last_low reset to None. Swing_Low 4. last_low becomes 4. Trend = -1.
    assert result['BOS'].iloc[9] == 5
    assert result['Trend'].iloc[9] == -1
    
    # idx 10: Close 12. Close > last_high (12 > 15 is False. Wait, last_high is 15. Close is 12. So no CHoCH).
    # Let's verify Close > last_high works for Bullish CHoCH.
    # If we change index 10 Close to 16, Close > 15 -> CHoCH = 15, Trend = 1.
    
def test_detect_structures_bullish_choch():
    data = {
        'Close':      [10, 8, 7, 12],
        'Swing_High': [10, np.nan, np.nan, np.nan],
        'Swing_Low':  [np.nan, 8, np.nan, np.nan]
    }
    df = pd.DataFrame(data)
    # Set starting trend to -1 (Bearish)
    # We will modify the function or just set starting trend to bearish.
    # Wait, the starting trend in detect_structures is hardcoded to 1 (Bullish).
    # How does it become Bearish? It becomes Bearish when price breaks last_low.
    # So we need to first trigger a Bearish CHoCH, then a Bullish CHoCH.
    
    # Let's construct a sequence:
    # 1. Start Bullish.
    # 2. Break low -> Bearish CHoCH (Trend becomes Bearish).
    # 3. Establish a new High.
    # 4. Break high -> Bullish CHoCH (Trend becomes Bullish).
    data = {
        'Close':      [10, 12,  8, 11, 14],
        'Swing_High': [12, np.nan, np.nan, 11, np.nan],
        'Swing_Low':  [np.nan, 10, np.nan, np.nan, np.nan]
    }
    # idx 0: Close 10, Swing_Low 10. last_low=10.
    # idx 1: Close 12, Swing_High 12. last_high=12.
    # idx 2: Close 8. Close < 10 (last_low). Bearish CHoCH = 10. Trend becomes -1.
    # idx 3: Close 11, Swing_High 11. last_high=11. Trend is -1.
    # idx 4: Close 14. Close > 11 (last_high). Bullish CHoCH = 11. Trend becomes 1.
    df = pd.DataFrame(data)
    result = detect_structures(df)
    
    assert result['CHoCH'].iloc[2] == 10
    assert result['Trend'].iloc[2] == -1
    assert result['CHoCH'].iloc[4] == 11
    assert result['Trend'].iloc[4] == 1
