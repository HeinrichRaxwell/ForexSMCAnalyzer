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
    
    # 3. Initialize Breaker Block (BB) columns
    df['BB_Type'] = None
    df['BB_Top'] = np.nan
    df['BB_Bottom'] = np.nan
    df['BB_Mitigated'] = False
    
    active_obs = []
    active_breakers = []
    
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
                    
        # --- C. OB Mitigation Check & Breaker Block Creation ---
        still_active = []
        for ob in active_obs:
            if i <= ob['index']:
                still_active.append(ob)
                continue
                
            mitigated = False
            broken = False
            if ob['type'] == 'BULLISH':
                if df['Low'].iloc[i] <= ob['top']:
                    mitigated = True
                if df['Close'].iloc[i] < ob['bottom']:
                    broken = True
            else:  # BEARISH OB
                if df['High'].iloc[i] >= ob['bottom']:
                    mitigated = True
                if df['Close'].iloc[i] > ob['top']:
                    broken = True
                    
            if broken:
                # OB is broken! It is marked as mitigated, and a Breaker Block flips open
                df.at[df.index[ob['index']], 'OB_Mitigated'] = True
                
                bb_type = 'BEARISH' if ob['type'] == 'BULLISH' else 'BULLISH'
                df.at[df.index[i], 'BB_Type'] = bb_type
                df.at[df.index[i], 'BB_Top'] = ob['top']
                df.at[df.index[i], 'BB_Bottom'] = ob['bottom']
                
                active_breakers.append({
                    'index': i,
                    'type': bb_type,
                    'top': ob['top'],
                    'bottom': ob['bottom']
                })
            else:
                if mitigated:
                    df.at[df.index[ob['index']], 'OB_Mitigated'] = True
                # It is not broken yet, so it remains active for further retests or breaks
                still_active.append(ob)
                
        active_obs = still_active
        
        # --- D. Breaker Block Mitigation Check ---
        still_active_breakers = []
        for bb in active_breakers:
            if i <= bb['index']:
                still_active_breakers.append(bb)
                continue
                
            mitigated = False
            if bb['type'] == 'BULLISH':
                # Mitigated if price touches support
                if df['Low'].iloc[i] <= bb['top']:
                    mitigated = True
                # Invalidated if price closes below
                if df['Close'].iloc[i] < bb['bottom']:
                    mitigated = True
            else:  # BEARISH
                # Mitigated if price touches resistance
                if df['High'].iloc[i] >= bb['bottom']:
                    mitigated = True
                # Invalidated if price closes above
                if df['Close'].iloc[i] > bb['top']:
                    mitigated = True
                    
            if mitigated:
                df.at[df.index[bb['index']], 'BB_Mitigated'] = True
            else:
                still_active_breakers.append(bb)
                
        active_breakers = still_active_breakers
        
    return df

def detect_snr_and_swapzones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect key Support and Resistance levels from Swing Highs and Swing Lows,
    and track Support-Resistance Flips (Swapzones).
    """
    df = df.copy()
    
    # Initialize columns
    df['Swap_Type'] = None
    df['Swap_Level'] = np.nan
    df['Swap_Mitigated'] = False
    
    active_supports = [] # list of float levels
    active_resistances = [] # list of float levels
    active_swapzones = [] # list of dicts with 'index', 'type', 'level'
    
    for i in range(len(df)):
        # 1. Update active supports/resistances if new Swing Point detected on this candle
        if 'Swing_High' in df.columns and not pd.isna(df['Swing_High'].iloc[i]):
            active_resistances.append(float(df['Swing_High'].iloc[i]))
        if 'Swing_Low' in df.columns and not pd.isna(df['Swing_Low'].iloc[i]):
            active_supports.append(float(df['Swing_Low'].iloc[i]))
            
        # 2. Check if price closes beyond any support or resistance (Swapzone trigger)
        close_val = df['Close'].iloc[i]
        
        # Check broken Resistances (Resistance flips to Support)
        broken_resistances = []
        for res in active_resistances:
            if close_val > res:
                df.at[df.index[i], 'Swap_Type'] = 'SUPPORT' # Swap Support
                df.at[df.index[i], 'Swap_Level'] = res
                active_swapzones.append({
                    'index': i,
                    'type': 'SUPPORT',
                    'level': res
                })
                broken_resistances.append(res)
        for res in broken_resistances:
            active_resistances.remove(res)
            
        # Check broken Supports (Support flips to Resistance)
        broken_supports = []
        for sup in active_supports:
            if close_val < sup:
                df.at[df.index[i], 'Swap_Type'] = 'RESISTANCE' # Swap Resistance
                df.at[df.index[i], 'Swap_Level'] = sup
                active_swapzones.append({
                    'index': i,
                    'type': 'RESISTANCE',
                    'level': sup
                })
                broken_supports.append(sup)
        for sup in broken_supports:
            active_supports.remove(sup)
            
        # 3. Check mitigation of active swapzones
        still_active_swaps = []
        for swap in active_swapzones:
            if i <= swap['index']:
                still_active_swaps.append(swap)
                continue
                
            high_val = df['High'].iloc[i]
            low_val = df['Low'].iloc[i]
            level = swap['level']
            
            # Mitigated if touched by high/low
            mitigated = (low_val <= level <= high_val)
            
            if mitigated:
                df.at[df.index[swap['index']], 'Swap_Mitigated'] = True
            else:
                still_active_swaps.append(swap)
        active_swapzones = still_active_swaps
        
    return df


