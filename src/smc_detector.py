import os
import numpy as np
import pandas as pd

def _get_sl_buffer_pips(strategy_name: str) -> float:
    """Gets the Stop Loss buffer in pips for a given strategy from environment variables,
    with a default fallback.
    """
    env_name = f"MT5_{strategy_name.upper()}_SL_BUFFER_PIPS"
    default_map = {
        "OB": 20.0,
        "FVG": 20.0,
        "SWAP": 20.0,
        "BPR": 20.0,
        "IC": 20.0,
        "SND": 20.0,
    }
    fallback = default_map.get(strategy_name.upper(), 20.0)
    try:
        val = os.getenv(env_name)
        if val is not None:
            return float(val)
    except (ValueError, TypeError):
        pass
    return fallback

def is_running_candle(df: pd.DataFrame, idx: int) -> bool:
    """
    Check if the candle at idx is a running candle.
    In live scanning, the last row (index len(df)-1) is the forming candle,
    but only if df.attrs['has_running_candle'] is explicitly True.
    """
    if idx != len(df) - 1:
        return False
    if df.attrs.get("closed_only", False):
        return False
    return bool(df.attrs.get("has_running_candle", False))

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
    
    # If the dataframe has a running candle at the end, we must exclude the last row
    # from being used as a future lookahead candle in the sliding window.
    # This prevents the running candle's live high/low from repainting the swing points of previous closed candles.
    has_running = (
        bool(df.attrs.get("has_running_candle", False))
        and not bool(df.attrs.get("closed_only", False))
    )
    end_offset = half + 1 if has_running else half
    
    for i in range(half, len(df) - end_offset):
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
    
    for idx_pos, (idx, row) in enumerate(df.iterrows()):
        is_running = is_running_candle(df, idx_pos)
        
        if not is_running:
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
        
        if not is_running:
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
    
    # 1. JPY Pairs (e.g. USDJPYm)
    if "JPY" in symbol_upper:
        return 0.01
    # 2. Gold (e.g. XAUUSDm, XAUCNHm)
    elif "XAU" in symbol_upper or "GOLD" in symbol_upper:
        return 0.1
    # 3. Silver (e.g. XAGUSDm)
    elif "XAG" in symbol_upper or "SILVER" in symbol_upper:
        return 0.01
    # 4. Crypto (e.g. BTCUSDm, ETHUSDm)
    elif any(crypto in symbol_upper for crypto in ["BTC", "ETH", "SOL", "LTC", "XRP", "ADA", "DOT", "DOGE", "BCH", "BNB"]):
        if "BTC" in symbol_upper:
            return 1.0
        elif "ETH" in symbol_upper:
            return 0.1
        else:
            return 0.01
    # 5. Oil / Gas (e.g. USOIL, UKOIL, WTI, BRENT)
    elif any(oil in symbol_upper for oil in ["USO", "UKO", "WTI", "BRENT", "XBR", "XTI"]):
        return 0.01
    # 6. Indices (e.g. US30, GER30, SPX500, NAS100)
    elif any(idx in symbol_upper for idx in ["US30", "GER", "SPX", "NAS", "USTEC", "DE30", "HK50"]):
        return 1.0
    # 7. Standard Forex (e.g. EURUSDm, GBPUSDm)
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
    df['OB_Fibo_0.0'] = np.nan
    df['OB_Fibo_0.5'] = np.nan
    df['OB_Fibo_0.618'] = np.nan
    df['OB_Fibo_1.0'] = np.nan
    df['OB_SL'] = np.nan
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
        is_running = is_running_candle(df, i)
        
        # --- A. FVG Detection (3-candle relationship) ---
        if i >= 2 and not is_running:
            # Bullish FVG: High[i-2] < Low[i] and Candle i-1 is bullish
            if df['High'].iloc[i-2] < df['Low'].iloc[i] and df['Close'].iloc[i-1] > df['Open'].iloc[i-1]:
                df.at[df.index[i], 'FVG_Type'] = 'BULLISH'
                df.at[df.index[i], 'FVG_Top'] = df['Low'].iloc[i]
                df.at[df.index[i], 'FVG_Bottom'] = df['High'].iloc[i-2]
                
                # Always draw Fibo on Candle 2 wicks (middle candle)
                pip_multiplier = get_pip_multiplier(symbol)
                buffer = _get_sl_buffer_pips("FVG") * pip_multiplier
                
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
                
                # Always draw Fibo on Candle 2 wicks (middle candle)
                pip_multiplier = get_pip_multiplier(symbol)
                buffer = _get_sl_buffer_pips("FVG") * pip_multiplier
                
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
        
        if (has_bos or has_choch) and not is_running:
            trend = df['Trend'].iloc[i]
            if trend == 1:
                # Bullish break: look backward for the last bearish candle (Close < Open)
                ob_idx = None
                for j in range(i - 1, -1, -1):
                    if df['Close'].iloc[j] < df['Open'].iloc[j]:
                        ob_idx = j
                        break
                if ob_idx is not None:
                    ob_top = float(df['High'].iloc[ob_idx])
                    ob_bottom = float(df['Low'].iloc[ob_idx])
                    buffer = _get_sl_buffer_pips("OB") * get_pip_multiplier(symbol)
                    
                    fibo_0_0 = float(df['High'].iloc[i])
                    fibo_1_0 = ob_bottom
                    fibo_0_5 = ob_top - 0.5 * (ob_top - ob_bottom)
                    fibo_0_618 = ob_top - 0.618 * (ob_top - ob_bottom)
                    ob_sl = ob_bottom - buffer
                    
                    df.at[df.index[i], 'OB_Type'] = 'BULLISH'
                    df.at[df.index[i], 'OB_Top'] = ob_top
                    df.at[df.index[i], 'OB_Bottom'] = ob_bottom
                    df.at[df.index[i], 'OB_Fibo_0.0'] = fibo_0_0
                    df.at[df.index[i], 'OB_Fibo_0.5'] = fibo_0_5
                    df.at[df.index[i], 'OB_Fibo_0.618'] = fibo_0_618
                    df.at[df.index[i], 'OB_Fibo_1.0'] = fibo_1_0
                    df.at[df.index[i], 'OB_SL'] = ob_sl
                    
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
                    ob_top = float(df['High'].iloc[ob_idx])
                    ob_bottom = float(df['Low'].iloc[ob_idx])
                    buffer = _get_sl_buffer_pips("OB") * get_pip_multiplier(symbol)
                    
                    fibo_0_0 = float(df['Low'].iloc[i])
                    fibo_1_0 = ob_top
                    fibo_0_5 = ob_bottom + 0.5 * (ob_top - ob_bottom)
                    fibo_0_618 = ob_bottom + 0.618 * (ob_top - ob_bottom)
                    ob_sl = ob_top + buffer
                    
                    df.at[df.index[i], 'OB_Type'] = 'BEARISH'
                    df.at[df.index[i], 'OB_Top'] = ob_top
                    df.at[df.index[i], 'OB_Bottom'] = ob_bottom
                    df.at[df.index[i], 'OB_Fibo_0.0'] = fibo_0_0
                    df.at[df.index[i], 'OB_Fibo_0.5'] = fibo_0_5
                    df.at[df.index[i], 'OB_Fibo_0.618'] = fibo_0_618
                    df.at[df.index[i], 'OB_Fibo_1.0'] = fibo_1_0
                    df.at[df.index[i], 'OB_SL'] = ob_sl
                    
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
            if i <= ob['index'] or is_running:
                still_active.append(ob)
                continue
                
            broken = False
            if ob['type'] == 'BULLISH':
                if df['Close'].iloc[i] < ob['bottom']:
                    broken = True
            else:  # BEARISH OB
                if df['Close'].iloc[i] > ob['top']:
                    broken = True
                    
            if broken:
                # OB is invalidated by a body close beyond the zone, and a Breaker Block flips open.
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
                # Retests into the OB are entry opportunities, not mitigation.
                still_active.append(ob)
                
        active_obs = still_active
        
        # --- D. Breaker Block Mitigation Check ---
        still_active_breakers = []
        for bb in active_breakers:
            if i <= bb['index'] or is_running:
                still_active_breakers.append(bb)
                continue
                
            mitigated = False
            if bb['type'] == 'BULLISH':
                # Invalidated only if price closes below the flipped zone.
                if df['Close'].iloc[i] < bb['bottom']:
                    mitigated = True
            else:  # BEARISH
                # Invalidated only if price closes above the flipped zone.
                if df['Close'].iloc[i] > bb['top']:
                    mitigated = True
                    
            if mitigated:
                df.at[df.index[bb['index']], 'BB_Mitigated'] = True
            else:
                still_active_breakers.append(bb)
                
        active_breakers = still_active_breakers
        
    return df

def detect_snr_and_swapzones(df: pd.DataFrame, symbol: str = "XAUUSD") -> pd.DataFrame:
    """
    Detect key Support and Resistance levels from Swing Highs and Swing Lows,
    and track Support-Resistance Flips (Swapzones) with 1-candle Fibonacci levels.
    """
    df = df.copy()
    
    # Initialize columns
    df['Swap_Type'] = None
    df['Swap_Level'] = np.nan
    df['Swap_Fibo_0.0'] = np.nan
    df['Swap_Fibo_0.5'] = np.nan
    df['Swap_Fibo_0.618'] = np.nan
    df['Swap_Fibo_1.0'] = np.nan
    df['Swap_SL'] = np.nan
    df['Swap_Mitigated'] = False
    
    active_supports = [] # list of dicts with 'level', 'high', 'low', 'index'
    active_resistances = [] # list of dicts with 'level', 'high', 'low', 'index'
    active_swapzones = [] # list of dicts with 'index', 'type', 'level'
    
    for i in range(len(df)):
        is_running = is_running_candle(df, i)
        
        # 1. Update active supports/resistances if new Swing Point detected on this candle
        if not is_running:
            if 'Swing_High' in df.columns and not pd.isna(df['Swing_High'].iloc[i]):
                active_resistances.append({
                    'level': float(df['Swing_High'].iloc[i]),
                    'high': float(df['High'].iloc[i]),
                    'low': float(df['Low'].iloc[i]),
                    'index': i
                })
            if 'Swing_Low' in df.columns and not pd.isna(df['Swing_Low'].iloc[i]):
                active_supports.append({
                    'level': float(df['Swing_Low'].iloc[i]),
                    'high': float(df['High'].iloc[i]),
                    'low': float(df['Low'].iloc[i]),
                    'index': i
                })
            
        # 2. Check if price closes beyond any support or resistance (Swapzone trigger)
        if not is_running:
            close_val = df['Close'].iloc[i]
            
            # Check broken Resistances (Resistance flips to Support)
            broken_resistances = []
            recorded_any = False
            for res_dict in active_resistances:
                res = res_dict['level']
                if close_val > res:
                    swap_high = res_dict['high']
                    swap_low = res_dict['low']
                    buffer = _get_sl_buffer_pips("SWAP") * get_pip_multiplier(symbol)
                    
                    if not recorded_any:
                        df.at[df.index[i], 'Swap_Type'] = 'SUPPORT' # Swap Support
                        df.at[df.index[i], 'Swap_Level'] = res
                        
                        # Fibo calculation on the swing point candle!
                        # Bullish Swapzone Fibo (S/R flip)
                        fibo_1_0 = swap_low
                        fibo_0_0 = float(df['High'].iloc[i]) # Target TP (breakout candle high)
                        
                        # Entry levels on the 1-candle zone
                        fibo_0_5 = swap_high - 0.5 * (swap_high - swap_low)
                        fibo_0_618 = swap_high - 0.618 * (swap_high - swap_low)
                        swap_sl = swap_low - buffer
                        
                        df.at[df.index[i], 'Swap_Fibo_0.0'] = fibo_0_0
                        df.at[df.index[i], 'Swap_Fibo_0.5'] = fibo_0_5
                        df.at[df.index[i], 'Swap_Fibo_0.618'] = fibo_0_618
                        df.at[df.index[i], 'Swap_Fibo_1.0'] = fibo_1_0
                        df.at[df.index[i], 'Swap_SL'] = swap_sl
                        recorded_any = True
                    
                    active_swapzones.append({
                        'index': i,
                        'type': 'SUPPORT',
                        'level': res,
                        'high': swap_high,
                        'low': swap_low
                    })
                    broken_resistances.append(res_dict)
            for res_dict in broken_resistances:
                active_resistances.remove(res_dict)
                
            # Check broken Supports (Support flips to Resistance)
            broken_supports = []
            recorded_any_sup = False
            for sup_dict in active_supports:
                sup = sup_dict['level']
                if close_val < sup:
                    swap_high = sup_dict['high']
                    swap_low = sup_dict['low']
                    buffer = _get_sl_buffer_pips("SWAP") * get_pip_multiplier(symbol)
                    
                    if not recorded_any_sup:
                        df.at[df.index[i], 'Swap_Type'] = 'RESISTANCE' # Swap Resistance
                        df.at[df.index[i], 'Swap_Level'] = sup
                        
                        # Fibo calculation on the swing point candle!
                        # Bearish Swapzone Fibo (S/R flip)
                        fibo_1_0 = swap_high
                        fibo_0_0 = float(df['Low'].iloc[i]) # Target TP (breakout candle low)
                        
                        # Entry levels on the 1-candle zone
                        fibo_0_5 = swap_low + 0.5 * (swap_high - swap_low)
                        fibo_0_618 = swap_low + 0.618 * (swap_high - swap_low)
                        swap_sl = swap_high + buffer
                        
                        df.at[df.index[i], 'Swap_Fibo_0.0'] = fibo_0_0
                        df.at[df.index[i], 'Swap_Fibo_0.5'] = fibo_0_5
                        df.at[df.index[i], 'Swap_Fibo_0.618'] = fibo_0_618
                        df.at[df.index[i], 'Swap_Fibo_1.0'] = fibo_1_0
                        df.at[df.index[i], 'Swap_SL'] = swap_sl
                        recorded_any_sup = True
                    
                    active_swapzones.append({
                        'index': i,
                        'type': 'RESISTANCE',
                        'level': sup,
                        'high': swap_high,
                        'low': swap_low
                    })
                    broken_supports.append(sup_dict)
            for sup_dict in broken_supports:
                active_supports.remove(sup_dict)
            
        # 3. Check mitigation of active swapzones
        still_active_swaps = []
        for swap in active_swapzones:
            if i <= swap['index'] or is_running:
                still_active_swaps.append(swap)
                continue
                
            close_val = df['Close'].iloc[i]
            
            # Retests are not mitigation; only a closed candle back beyond the
            # flipped swing candle invalidates the setup.
            mitigated = False
            if swap['type'] == 'SUPPORT':
                if close_val < swap['low']:
                    mitigated = True
            else: # RESISTANCE
                if close_val > swap['high']:
                    mitigated = True
            
            if mitigated:
                df.at[df.index[swap['index']], 'Swap_Mitigated'] = True
            else:
                still_active_swaps.append(swap)
        active_swapzones = still_active_swaps
        
    return df

def detect_bpr(df: pd.DataFrame, symbol: str = "XAUUSD") -> pd.DataFrame:
    """
    Detect Balanced Price Ranges (BPR) which are formed when a bullish FVG and a bearish FVG
    overlap in the same price range within a short lookback window (e.g. 15 candles).
    """
    df = df.copy()
    
    # Initialize columns
    df['BPR_Type'] = None
    df['BPR_Top'] = np.nan
    df['BPR_Bottom'] = np.nan
    df['BPR_Fibo_0.0'] = np.nan
    df['BPR_Fibo_0.5'] = np.nan
    df['BPR_Fibo_0.618'] = np.nan
    df['BPR_Fibo_1.0'] = np.nan
    df['BPR_SL'] = np.nan
    df['BPR_Mitigated'] = False
    
    active_bprs = [] # list of dicts with 'index', 'type', 'top', 'bottom'
    
    # Find all FVGs first for lookback check
    # Note: df already has FVG columns populated from detect_fvg_and_ob
    fvg_types = df['FVG_Type'].values if 'FVG_Type' in df.columns else [None] * len(df)
    fvg_tops = df['FVG_Top'].values if 'FVG_Top' in df.columns else [np.nan] * len(df)
    fvg_bottoms = df['FVG_Bottom'].values if 'FVG_Bottom' in df.columns else [np.nan] * len(df)
    
    for i in range(2, len(df)):
        is_running = is_running_candle(df, i)
        
        if not is_running:
            current_type = fvg_types[i]
            if current_type is not None and pd.notna(current_type):
                curr_top = fvg_tops[i]
                curr_bottom = fvg_bottoms[i]
                
                # Look back up to 15 candles for opposite FVG
                lookback = 15
                start_k = max(2, i - lookback)
                for k in range(start_k, i):
                    prev_type = fvg_types[k]
                    if prev_type is not None and pd.notna(prev_type) and prev_type != current_type:
                        prev_top = fvg_tops[k]
                        prev_bottom = fvg_bottoms[k]
                        
                        # Calculate overlap range
                        overlap_bottom = max(curr_bottom, prev_bottom)
                        overlap_top = min(curr_top, prev_top)
                        
                        if overlap_bottom < overlap_top:
                            # Overlap exists! Create a BPR at index i
                            bpr_type = 'BULLISH' if current_type == 'BULLISH' else 'BEARISH'
                            df.at[df.index[i], 'BPR_Type'] = bpr_type
                            df.at[df.index[i], 'BPR_Top'] = overlap_top
                            df.at[df.index[i], 'BPR_Bottom'] = overlap_bottom
                            
                            pip_multiplier = get_pip_multiplier(symbol)
                            buffer = _get_sl_buffer_pips("BPR") * pip_multiplier
                            
                            # BPR entries use the displacement candle that formed the current FVG,
                            # matching FVG fibs instead of pulling fibs across the overlap zone.
                            if bpr_type == 'BULLISH':
                                fibo_1_0 = float(df['Low'].iloc[i-1])
                                fibo_0_0 = float(df['High'].iloc[i-1])
                                fibo_0_5 = fibo_0_0 - 0.5 * (fibo_0_0 - fibo_1_0)
                                fibo_0_618 = fibo_0_0 - 0.618 * (fibo_0_0 - fibo_1_0)
                                bpr_sl = fibo_1_0 - buffer
                            else:
                                fibo_1_0 = float(df['High'].iloc[i-1])
                                fibo_0_0 = float(df['Low'].iloc[i-1])
                                fibo_0_5 = fibo_0_0 + 0.5 * (fibo_1_0 - fibo_0_0)
                                fibo_0_618 = fibo_0_0 + 0.618 * (fibo_1_0 - fibo_0_0)
                                bpr_sl = fibo_1_0 + buffer
                                
                            df.at[df.index[i], 'BPR_Fibo_0.0'] = fibo_0_0
                            df.at[df.index[i], 'BPR_Fibo_0.5'] = fibo_0_5
                            df.at[df.index[i], 'BPR_Fibo_0.618'] = fibo_0_618
                            df.at[df.index[i], 'BPR_Fibo_1.0'] = fibo_1_0
                            df.at[df.index[i], 'BPR_SL'] = bpr_sl
                            
                            active_bprs.append({
                                'index': i,
                                'type': bpr_type,
                                'top': overlap_top,
                                'bottom': overlap_bottom
                            })
                            break # Only pair with the most recent opposite FVG in range
                        
        # 3. Check mitigation of active BPRs
        still_active_bprs = []
        for bpr in active_bprs:
            if i <= bpr['index'] or is_running:
                still_active_bprs.append(bpr)
                continue
                
            close_val = df['Close'].iloc[i]
            low_val = df['Low'].iloc[i]
            high_val = df['High'].iloc[i]
            
            mitigated = False
            if bpr['type'] == 'BULLISH':
                # Mitigated ONLY if price closes below the BPR bottom (broken by body candle)
                if close_val < bpr['bottom']:
                    mitigated = True
            else: # BEARISH
                # Mitigated ONLY if price closes above the BPR top (broken by body candle)
                if close_val > bpr['top']:
                    mitigated = True
                    
            if mitigated:
                df.at[df.index[bpr['index']], 'BPR_Mitigated'] = True
            else:
                still_active_bprs.append(bpr)
        active_bprs = still_active_bprs
        
    return df


def detect_indecision_candles(df: pd.DataFrame, body_ratio: float = 0.25, symbol: str = "XAUUSD") -> pd.DataFrame:
    """
    Detects Indecision Candles (Doji, Spinning Top) that are broken by subsequent price action,
    and calculates their Fibonacci levels.
    """
    df = df.copy()
    
    df['IC_Type'] = None
    df['IC_Top'] = np.nan
    df['IC_Bottom'] = np.nan
    df['IC_Fibo_0.0'] = np.nan
    df['IC_Fibo_0.5'] = np.nan
    df['IC_Fibo_0.618'] = np.nan
    df['IC_Fibo_1.0'] = np.nan
    df['IC_SL'] = np.nan
    df['IC_Mitigated'] = False
    
    pip_multiplier = get_pip_multiplier(symbol)
    buffer = _get_sl_buffer_pips("IC") * pip_multiplier
    
    active_ics = [] # list of dicts: {'index', 'type', 'top', 'bottom'}
    
    for i in range(len(df)):
        is_running = is_running_candle(df, i)
        
        if not is_running:
            # Check if candle i triggers a breakout of a previous indecision candle k
            close_i = df['Close'].iloc[i]
            
            # Search backward for an indecision candle k that hasn't been broken yet
            for k in range(i - 1, max(-1, i - 6), -1):
                body_k = abs(df['Close'].iloc[k] - df['Open'].iloc[k])
                range_k = df['High'].iloc[k] - df['Low'].iloc[k]
                if range_k <= 0 or (body_k / range_k) > body_ratio:
                    continue
                    
                # Check if any candle between k and i closed outside k's range
                already_broken = False
                for m in range(k + 1, i):
                    if df['Close'].iloc[m] > df['High'].iloc[k] or df['Close'].iloc[m] < df['Low'].iloc[k]:
                        already_broken = True
                        break
                if already_broken:
                    continue
                    
                high_k = df['High'].iloc[k]
                low_k = df['Low'].iloc[k]
                
                if close_i > high_k:
                    # Bullish Breakout!
                    df.at[df.index[i], 'IC_Type'] = 'BULLISH'
                    df.at[df.index[i], 'IC_Top'] = high_k
                    df.at[df.index[i], 'IC_Bottom'] = low_k
                    
                    fibo_1_0 = low_k
                    fibo_0_0 = high_k
                    fibo_0_5 = high_k - 0.5 * (high_k - low_k)
                    fibo_0_618 = high_k - 0.618 * (high_k - low_k)
                    ic_sl = low_k - buffer
                    
                    df.at[df.index[i], 'IC_Fibo_0.0'] = fibo_0_0
                    df.at[df.index[i], 'IC_Fibo_0.5'] = fibo_0_5
                    df.at[df.index[i], 'IC_Fibo_0.618'] = fibo_0_618
                    df.at[df.index[i], 'IC_Fibo_1.0'] = fibo_1_0
                    df.at[df.index[i], 'IC_SL'] = ic_sl
                    
                    active_ics.append({
                        'index': i,
                        'type': 'BULLISH',
                        'top': high_k,
                        'bottom': low_k
                    })
                    break
                    
                elif close_i < low_k:
                    # Bearish Breakout!
                    df.at[df.index[i], 'IC_Type'] = 'BEARISH'
                    df.at[df.index[i], 'IC_Top'] = high_k
                    df.at[df.index[i], 'IC_Bottom'] = low_k
                    
                    fibo_1_0 = high_k
                    fibo_0_0 = low_k
                    fibo_0_5 = low_k + 0.5 * (high_k - low_k)
                    fibo_0_618 = low_k + 0.618 * (high_k - low_k)
                    ic_sl = high_k + buffer
                    
                    df.at[df.index[i], 'IC_Fibo_0.0'] = fibo_0_0
                    df.at[df.index[i], 'IC_Fibo_0.5'] = fibo_0_5
                    df.at[df.index[i], 'IC_Fibo_0.618'] = fibo_0_618
                    df.at[df.index[i], 'IC_Fibo_1.0'] = fibo_1_0
                    df.at[df.index[i], 'IC_SL'] = ic_sl
                    
                    active_ics.append({
                        'index': i,
                        'type': 'BEARISH',
                        'top': high_k,
                        'bottom': low_k
                    })
                    break
                
        # Check mitigation of active IC zones
        still_active = []
        for ic in active_ics:
            if i <= ic['index'] or is_running:
                still_active.append(ic)
                continue
                
            close_val = df['Close'].iloc[i]
            
            mitigated = False
            if ic['type'] == 'BULLISH':
                if close_val < ic['bottom']:
                    mitigated = True # Invalidated by body close beyond the IC zone
            else: # BEARISH
                if close_val > ic['top']:
                    mitigated = True # Invalidated by body close beyond the IC zone
                    
            if mitigated:
                df.at[df.index[ic['index']], 'IC_Mitigated'] = True
            else:
                still_active.append(ic)
        active_ics = still_active
        
    return df

def detect_supply_demand_zones(df: pd.DataFrame, symbol: str = "XAUUSD") -> pd.DataFrame:
    """
    Detect Supply and Demand zones based on classical Price Action patterns:
    - RBR (Rally Base Rally) -> Demand Zone
    - DBR (Drop Base Rally) -> Demand Zone
    - DBD (Drop Base Drop) -> Supply Zone
    - RBD (Rally Base Drop) -> Supply Zone
    """
    df = df.copy()
    
    # Initialize columns
    df['SD_Type'] = None
    df['SD_Top'] = np.nan
    df['SD_Bottom'] = np.nan
    df['SD_Fibo_0.0'] = np.nan
    df['SD_Fibo_0.5'] = np.nan
    df['SD_Fibo_0.618'] = np.nan
    df['SD_Fibo_1.0'] = np.nan
    df['SD_SL'] = np.nan
    df['SD_Mitigated'] = False
    
    if len(df) < 5:
        return df
        
    # Calculate body size and its average
    body = (df['Close'] - df['Open']).abs()
    avg_body = body.rolling(window=14, min_periods=1).mean()
    
    active_zones = [] # list of dicts
    
    for i in range(2, len(df)):
        is_running = is_running_candle(df, i)
        
        if not is_running:
            # Calculate recent candles state
            body_i2 = abs(df['Close'].iloc[i-2] - df['Open'].iloc[i-2])
            body_i1 = abs(df['Close'].iloc[i-1] - df['Open'].iloc[i-1])
            body_i = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            
            avg_b_i = avg_body.iloc[i] if not pd.isna(avg_body.iloc[i]) else 1.0
            
            # Rally / Drop definition (strong moves)
            is_rally_i2 = (df['Close'].iloc[i-2] > df['Open'].iloc[i-2]) and (body_i2 > 1.0 * avg_b_i)
            is_drop_i2 = (df['Close'].iloc[i-2] < df['Open'].iloc[i-2]) and (body_i2 > 1.0 * avg_b_i)
            
            is_rally_i = (df['Close'].iloc[i] > df['Open'].iloc[i]) and (body_i > 1.0 * avg_b_i)
            is_drop_i = (df['Close'].iloc[i] < df['Open'].iloc[i]) and (body_i > 1.0 * avg_b_i)
            
            # Base definition (consolidation/indecision)
            is_base_i1 = (body_i1 < 0.8 * avg_b_i)
            
            sd_type = None
            if is_base_i1:
                if is_rally_i2 and is_rally_i:
                    sd_type = 'DEMAND_RBR'
                elif is_drop_i2 and is_rally_i:
                    sd_type = 'DEMAND_DBR'
                elif is_drop_i2 and is_drop_i:
                    sd_type = 'SUPPLY_DBD'
                elif is_rally_i2 and is_drop_i:
                    sd_type = 'SUPPLY_RBD'
                    
            if sd_type is not None:
                top_val = float(df['High'].iloc[i-1])
                bottom_val = float(df['Low'].iloc[i-1])
                
                buffer = _get_sl_buffer_pips("SND") * get_pip_multiplier(symbol)
                
                if 'DEMAND' in sd_type:
                    # Demand Zone: Fibo calculation
                    fibo_1_0 = bottom_val
                    fibo_0_0 = float(df['High'].iloc[i]) # High of expansion candle
                    fibo_0_5 = top_val - 0.5 * (top_val - bottom_val)
                    fibo_0_618 = top_val - 0.618 * (top_val - bottom_val)
                    sl_val = bottom_val - buffer
                else:
                    # Supply Zone: Fibo calculation
                    fibo_1_0 = top_val
                    fibo_0_0 = float(df['Low'].iloc[i]) # Low of expansion candle
                    fibo_0_5 = bottom_val + 0.5 * (top_val - bottom_val)
                    fibo_0_618 = bottom_val + 0.618 * (top_val - bottom_val)
                    sl_val = top_val + buffer
                    
                df.at[df.index[i], 'SD_Type'] = sd_type
                df.at[df.index[i], 'SD_Top'] = top_val
                df.at[df.index[i], 'SD_Bottom'] = bottom_val
                df.at[df.index[i], 'SD_Fibo_0.0'] = fibo_0_0
                df.at[df.index[i], 'SD_Fibo_0.5'] = fibo_0_5
                df.at[df.index[i], 'SD_Fibo_0.618'] = fibo_0_618
                df.at[df.index[i], 'SD_Fibo_1.0'] = fibo_1_0
                df.at[df.index[i], 'SD_SL'] = sl_val
                
                active_zones.append({
                    'index': i,
                    'type': sd_type,
                    'top': top_val,
                    'bottom': bottom_val,
                    'fibo_0.0': fibo_0_0,
                    'fibo_0.5': fibo_0_5,
                    'fibo_0.618': fibo_0_618,
                    'fibo_1.0': fibo_1_0,
                    'sl': sl_val
                })
            
        # Update mitigation of active zones
        close_val = float(df['Close'].iloc[i])
        
        still_active = []
        for zone in active_zones:
            if zone['index'] == i or is_running:
                still_active.append(zone)
                continue
                
            mitigated = False
            if 'DEMAND' in zone['type']:
                if close_val < zone['bottom']:
                    mitigated = True # Invalidated by body close beyond demand
            else:
                if close_val > zone['top']:
                    mitigated = True # Invalidated by body close beyond supply
                    
            if mitigated:
                df.at[df.index[zone['index']], 'SD_Mitigated'] = True
            else:
                still_active.append(zone)
                
        active_zones = still_active
        
    return df
