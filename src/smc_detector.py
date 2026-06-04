import numpy as np
import pandas as pd

def detect_swing_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Detect Swing Highs (peaks) and Swing Lows (troughs) within a sliding window.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns ['High', 'Low'].
        window (int): The window size for identifying swing points (must be odd, e.g., 5).
        
    Returns:
        pd.DataFrame: A copy of the DataFrame with new columns 'Swing_High' and 'Swing_Low'.
    """
    df = df.copy()
    half = window // 2
    
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan
    
    for i in range(half, len(df) - half):
        high_range = df['High'].iloc[i - half : i + half + 1]
        low_range = df['Low'].iloc[i - half : i + half + 1]
        
        # Check if the middle element is the maximum in the window for high
        if df['High'].iloc[i] == high_range.max():
            df.at[df.index[i], 'Swing_High'] = df['High'].iloc[i]
            
        # Check if the middle element is the minimum in the window for low
        if df['Low'].iloc[i] == low_range.min():
            df.at[df.index[i], 'Swing_Low'] = df['Low'].iloc[i]
            
    return df
