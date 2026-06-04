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

def detect_structures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect BOS (Break of Structure) and CHoCH (Change of Character).
    
    Args:
        df (pd.DataFrame): DataFrame containing columns ['Close', 'Swing_High', 'Swing_Low'].
        
    Returns:
        pd.DataFrame: A copy of the DataFrame with new columns 'BOS', 'CHoCH', and 'Trend'.
    """
    df = df.copy()
    df['BOS'] = np.nan
    df['CHoCH'] = np.nan
    df['Trend'] = 1  # 1: Bullish, -1: Bearish
    
    last_high = None
    last_low = None
    current_trend = 1
    
    for idx, row in df.iterrows():
        # 1. Check for breaks of existing structure first (before updating with current candle's swing points)
        if current_trend == 1:
            # Bullish BOS: price closes above last swing high
            if last_high is not None and row['Close'] > last_high:
                df.at[idx, 'BOS'] = last_high
                last_high = None
            # Bearish CHoCH: price closes below last swing low (trend reversal)
            elif last_low is not None and row['Close'] < last_low:
                df.at[idx, 'CHoCH'] = last_low
                current_trend = -1
                last_low = None
        else:  # Bearish trend
            # Bearish BOS: price closes below last swing low
            if last_low is not None and row['Close'] < last_low:
                df.at[idx, 'BOS'] = last_low
                last_low = None
            # Bullish CHoCH: price closes above last swing high (trend reversal)
            elif last_high is not None and row['Close'] > last_high:
                df.at[idx, 'CHoCH'] = last_high
                current_trend = 1
                last_high = None
                
        # 2. Record trend for this candle
        df.at[idx, 'Trend'] = current_trend
        
        # 3. Update active swing points if new ones are detected on this candle
        if not pd.isna(row['Swing_High']):
            last_high = row['Swing_High']
        if not pd.isna(row['Swing_Low']):
            last_low = row['Swing_Low']
            
    return df

