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

def get_pip_multiplier(symbol: str) -> float:
    """
    Get the pip multiplier for a given symbol to calculate price values from pips.
    """
    symbol_upper = symbol.upper()
    if "JPY" in symbol_upper:
        return 0.01
    elif "XAUUSD" in symbol_upper or "GOLD" in symbol_upper:
        return 0.1
    else:
        return 0.0001

def detect_fvg_and_ob(df: pd.DataFrame, symbol: str = "XAUUSD") -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVG) and Order Blocks (OB) with mitigation tracking.
    
    Args:
        df (pd.DataFrame): DataFrame containing candle data and structure signals.
        symbol (str): Symbol name (e.g., "XAUUSD", "EURUSD").
        
    Returns:
        pd.DataFrame: DataFrame with FVG and OB columns.
    """
    df = df.copy()
    
    # 1. Initialize FVG columns
    df['FVG_Type'] = None
    df['FVG_Top'] = np.nan
    df['FVG_Bottom'] = np.nan
    df['FVG_Fibo_0.0'] = np.nan
    df['FVG_Fibo_0.5'] = np.nan
    df['FVG_Fibo_0.618'] = np.nan
    df['FVG_Fibo_1.0'] = np.nan
    df['FVG_SL'] = np.nan
    
    # 2. Initialize OB columns
    df['OB_Type'] = None
    df['OB_Top'] = np.nan
    df['OB_Bottom'] = np.nan
    df['OB_Mitigated'] = False
    
    active_obs = []
    
    # Ensure dependencies exist
    if 'BOS' not in df.columns or 'CHoCH' not in df.columns:
        if 'Swing_High' not in df.columns:
            df = detect_swing_points(df)
        df = detect_structures(df)
        
    for i in range(len(df)):
        # --- A. FVG Detection (3-candle relationship) ---
        if i >= 2:
            # Bullish FVG: High[i-2] < Low[i] and Candle i-1 is bullish
            if df['High'].iloc[i-2] < df['Low'].iloc[i] and df['Close'].iloc[i-1] > df['Open'].iloc[i-1]:
                df.at[df.index[i], 'FVG_Type'] = 'BULLISH'
                df.at[df.index[i], 'FVG_Top'] = df['Low'].iloc[i]
                df.at[df.index[i], 'FVG_Bottom'] = df['High'].iloc[i-2]
                
                # Check Candle 2 range in pips
                pip_multiplier = get_pip_multiplier(symbol)
                candle2_range_pips = (df['High'].iloc[i-1] - df['Low'].iloc[i-1]) / pip_multiplier
                buffer = 20 * pip_multiplier
                
                if candle2_range_pips > 150:
                    # Draw Fibo on FVG gap
                    fibo_1_0 = df['High'].iloc[i-2]
                    fibo_0_0 = df['Low'].iloc[i]
                else:
                    # Draw Fibo on Candle 2 wicks
                    fibo_1_0 = df['Low'].iloc[i-1]
                    fibo_0_0 = df['High'].iloc[i-1]
                
                fibo_0_5 = fibo_0_0 - 0.5 * (fibo_0_0 - fibo_1_0)
                fibo_0_618 = fibo_0_0 - 0.618 * (fibo_0_0 - fibo_1_0)
                fvg_sl = fibo_1_0 - buffer
                
                df.at[df.index[i], 'FVG_Fibo_0.0'] = fibo_0_0
                df.at[df.index[i], 'FVG_Fibo_0.5'] = fibo_0_5
                df.at[df.index[i], 'FVG_Fibo_0.618'] = fibo_0_618
                df.at[df.index[i], 'FVG_Fibo_1.0'] = fibo_1_0
                df.at[df.index[i], 'FVG_SL'] = fvg_sl
                
            # Bearish FVG: Low[i-2] > High[i] and Candle i-1 is bearish
            elif df['Low'].iloc[i-2] > df['High'].iloc[i] and df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:
                df.at[df.index[i], 'FVG_Type'] = 'BEARISH'
                df.at[df.index[i], 'FVG_Top'] = df['Low'].iloc[i-2]
                df.at[df.index[i], 'FVG_Bottom'] = df['High'].iloc[i]
                
                # Check Candle 2 range in pips
                pip_multiplier = get_pip_multiplier(symbol)
                candle2_range_pips = (df['High'].iloc[i-1] - df['Low'].iloc[i-1]) / pip_multiplier
                buffer = 20 * pip_multiplier
                
                if candle2_range_pips > 150:
                    # Draw Fibo on FVG gap
                    fibo_1_0 = df['Low'].iloc[i-2]
                    fibo_0_0 = df['High'].iloc[i]
                else:
                    # Draw Fibo on Candle 2 wicks
                    fibo_1_0 = df['High'].iloc[i-1]
                    fibo_0_0 = df['Low'].iloc[i-1]
                
                fibo_0_5 = fibo_0_0 + 0.5 * (fibo_1_0 - fibo_0_0)
                fibo_0_618 = fibo_0_0 + 0.618 * (fibo_1_0 - fibo_0_0)
                fvg_sl = fibo_1_0 + buffer
                
                df.at[df.index[i], 'FVG_Fibo_0.0'] = fibo_0_0
                df.at[df.index[i], 'FVG_Fibo_0.5'] = fibo_0_5
                df.at[df.index[i], 'FVG_Fibo_0.618'] = fibo_0_618
                df.at[df.index[i], 'FVG_Fibo_1.0'] = fibo_1_0
                df.at[df.index[i], 'FVG_SL'] = fvg_sl
                
        # --- B. OB Detection ---
        has_bos = not pd.isna(df['BOS'].iloc[i])
        has_choch = not pd.isna(df['CHoCH'].iloc[i])
        
        if has_bos or has_choch:
            trend = df['Trend'].iloc[i]
            if trend == 1:
                # Bullish break: look backward for the last bearish candle (Close < Open)
                ob_idx = None
                for j in range(i - 1, -1, -1):
                    if df['Close'].iloc[j] < df['Open'].iloc[j]:
                        ob_idx = j
                        break
                if ob_idx is not None:
                    df.at[df.index[i], 'OB_Type'] = 'BULLISH'
                    df.at[df.index[i], 'OB_Top'] = df['High'].iloc[ob_idx]
                    df.at[df.index[i], 'OB_Bottom'] = df['Low'].iloc[ob_idx]
                    active_obs.append({
                        'index': i,
                        'ob_index': ob_idx,
                        'type': 'BULLISH',
                        'top': df['High'].iloc[ob_idx],
                        'bottom': df['Low'].iloc[ob_idx]
                    })
            else:
                # Bearish break: look backward for the last bullish candle (Close > Open)
                ob_idx = None
                for j in range(i - 1, -1, -1):
                    if df['Close'].iloc[j] > df['Open'].iloc[j]:
                        ob_idx = j
                        break
                if ob_idx is not None:
                    df.at[df.index[i], 'OB_Type'] = 'BEARISH'
                    df.at[df.index[i], 'OB_Top'] = df['High'].iloc[ob_idx]
                    df.at[df.index[i], 'OB_Bottom'] = df['Low'].iloc[ob_idx]
                    active_obs.append({
                        'index': i,
                        'ob_index': ob_idx,
                        'type': 'BEARISH',
                        'top': df['High'].iloc[ob_idx],
                        'bottom': df['Low'].iloc[ob_idx]
                    })
                    
        # --- C. OB Mitigation Check ---
        still_active = []
        for ob in active_obs:
            if i <= ob['index']:
                still_active.append(ob)
                continue
                
            mitigated = False
            if ob['type'] == 'BULLISH':
                if df['Low'].iloc[i] <= ob['top']:
                    mitigated = True
            else:  # BEARISH OB
                if df['High'].iloc[i] >= ob['bottom']:
                    mitigated = True
                    
            if mitigated:
                df.at[df.index[ob['index']], 'OB_Mitigated'] = True
            else:
                still_active.append(ob)
                
        active_obs = still_active
        
    return df


