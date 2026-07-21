import pandas as pd
import numpy as np

def calculate_daily_pivots(df_d1: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily pivot levels (Classic formula) using the previous day's High, Low, Close.
    
    Args:
        df_d1 (pd.DataFrame): Daily OHLCV DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame containing daily pivot levels.
    """
    df = df_d1.copy()
    
    # We shift by 1 to use the previous day's metrics for today's pivots
    df['prev_High'] = df['High'].shift(1)
    df['prev_Low'] = df['Low'].shift(1)
    df['prev_Close'] = df['Close'].shift(1)
    
    pp = (df['prev_High'] + df['prev_Low'] + df['prev_Close']) / 3.0
    pvt_range = df['prev_High'] - df['prev_Low']
    
    # Classic Support & Resistance formulas
    r1 = 2.0 * pp - df['prev_Low']
    r2 = pp + pvt_range
    r3 = df['prev_High'] + 2.0 * (pp - df['prev_Low'])
    r4 = pp + pvt_range * 3.0
    
    s1 = 2.0 * pp - df['prev_High']
    s2 = pp - pvt_range
    s3 = df['prev_Low'] - 2.0 * (df['prev_High'] - pp)
    s4 = pp - pvt_range * 3.0
    
    pivots_df = pd.DataFrame({
        'time': df['time'],
        'pivot_PP': pp,
        'pivot_R1': r1,
        'pivot_R2': r2,
        'pivot_R3': r3,
        'pivot_R4': r4,
        'pivot_S1': s1,
        'pivot_S2': s2,
        'pivot_S3': s3,
        'pivot_S4': s4
    })
    return pivots_df

def align_daily_pivots(df_ltf: pd.DataFrame, df_d1: pd.DataFrame) -> pd.DataFrame:
    """
    Align daily pivots onto a lower timeframe DataFrame by date lookup.
    
    Args:
        df_ltf (pd.DataFrame): Lower timeframe DataFrame (e.g., M15, H1).
        df_d1 (pd.DataFrame): Daily timeframe DataFrame.
        
    Returns:
        pd.DataFrame: The df_ltf with daily pivot columns added.
    """
    df_ltf = df_ltf.copy()
    pivots_df = calculate_daily_pivots(df_d1)
    
    # Format time column to datetime
    df_ltf['time_dt'] = pd.to_datetime(df_ltf['time'])
    pivots_df['time_dt'] = pd.to_datetime(pivots_df['time'])
    
    # Get date parts
    df_ltf['date_only'] = df_ltf['time_dt'].dt.date
    pivots_df['date_only'] = pivots_df['time_dt'].dt.date
    
    # Drop timestamp columns to avoid duplicates and join by date
    pivots_df = pivots_df.drop(columns=['time', 'time_dt']).set_index('date_only')
    
    # Join pivots onto lower timeframe
    df_ltf = df_ltf.join(pivots_df, on='date_only')
    
    # Clean up temporary columns
    df_ltf.drop(columns=['time_dt', 'date_only'], inplace=True)
    
    # Forward fill any gaps (e.g., weekends or early history)
    pivot_cols = ['pivot_PP', 'pivot_R1', 'pivot_R2', 'pivot_R3', 'pivot_R4', 'pivot_S1', 'pivot_S2', 'pivot_S3', 'pivot_S4']
    df_ltf[pivot_cols] = df_ltf[pivot_cols].ffill()
    
    return df_ltf

def get_pivot_features_at_idx(df: pd.DataFrame, idx: int, entry_price: float) -> dict:
    """
    Get pivot level proximity features at a specific row index.
    
    Args:
        df (pd.DataFrame): Lower timeframe DataFrame with pivot columns.
        idx (int): The row index of the setup.
        entry_price (float): Trade entry price level.
        
    Returns:
        dict: Proximity features dictionary.
    """
    p_cols = ['pivot_PP', 'pivot_R1', 'pivot_R2', 'pivot_R3', 'pivot_R4', 'pivot_S1', 'pivot_S2', 'pivot_S3', 'pivot_S4']
    
    # Verify columns exist and are not NaN at the target index
    has_pivots = all(col in df.columns for col in p_cols) and not pd.isna(df['pivot_PP'].iloc[idx])
    
    if has_pivots:
        pp = df['pivot_PP'].iloc[idx]
        levels = [df[col].iloc[idx] for col in p_cols]
        
        dist_to_pp = (entry_price - pp) / pp
        dist_to_nearest = min(abs(entry_price - lvl) for lvl in levels) / entry_price
    else:
        dist_to_pp = 0.0
        dist_to_nearest = 0.0
        
    return {
        'dist_entry_to_pp': dist_to_pp,
        'dist_entry_to_nearest_pivot': dist_to_nearest
    }

def draw_pivots_on_mt5(symbol: str, df_d1: pd.DataFrame):
    """
    Calculate daily pivots and save them to the MT5 Terminal Common Files folder
    as 'pivot_levels.json' so that a custom MQL5 indicator can draw them on MT5.
    """
    pivots_df = calculate_daily_pivots(df_d1)
    if pivots_df.empty:
        return
        
    latest_pivots = pivots_df.iloc[-1]
    if pd.isna(latest_pivots['pivot_PP']):
        if len(pivots_df) >= 2:
            latest_pivots = pivots_df.iloc[-2]
            
    if pd.isna(latest_pivots['pivot_PP']):
        return
        
    # Prepare JSON data
    pivot_data = {
        'pivot_PP': float(latest_pivots['pivot_PP']),
        'pivot_R1': float(latest_pivots['pivot_R1']),
        'pivot_R2': float(latest_pivots['pivot_R2']),
        'pivot_R3': float(latest_pivots['pivot_R3']),
        'pivot_R4': float(latest_pivots['pivot_R4']),
        'pivot_S1': float(latest_pivots['pivot_S1']),
        'pivot_S2': float(latest_pivots['pivot_S2']),
        'pivot_S3': float(latest_pivots['pivot_S3']),
        'pivot_S4': float(latest_pivots['pivot_S4']),
    }
    
    # Write to MT5 Terminal Common Files directory AND all local instance MQL5/Files directories
    import glob
    import json
    import os

    appdata = os.environ.get('APPDATA')
    if appdata:
        common_files_dir = os.path.join(appdata, "MetaQuotes", "Terminal", "Common", "Files")
        file_paths = [os.path.join(common_files_dir, "pivot_levels.json")]

        # Also write to all local instance MQL5/Files folders for MQL5 indicators reading without FILE_COMMON flag
        local_files_dirs = glob.glob(os.path.join(appdata, "MetaQuotes", "Terminal", "*", "MQL5", "Files"))
        for local_dir in local_files_dirs:
            file_paths.append(os.path.join(local_dir, "pivot_levels.json"))

        for fp in file_paths:
            try:
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w") as f:
                    json.dump(pivot_data, f, indent=4)
            except Exception as e:
                print(f"[Pivots Visualizer Error] Failed to write {fp}: {e}")
        print(f"[Pivots Visualizer] Successfully wrote daily pivot levels to {len(file_paths)} MT5 file paths.")

def detect_pivot_rejection_setups_at_idx(df: pd.DataFrame, idx: int, symbol: str = "XAUUSD") -> list:
    """
    Detect if there is a fresh pivot level rejection at the specified index.
    Returns a list of setup dicts (usually max 1 setup).
    """
    p_cols = ['pivot_PP', 'pivot_R1', 'pivot_R2', 'pivot_R3', 'pivot_R4', 'pivot_S1', 'pivot_S2', 'pivot_S3', 'pivot_S4']
    if not all(col in df.columns for col in p_cols):
        return []
        
    # Skip if NaNs
    if pd.isna(df['pivot_PP'].iloc[idx]):
        return []
        
    open_val = float(df['Open'].iloc[idx])
    high_val = float(df['High'].iloc[idx])
    low_val = float(df['Low'].iloc[idx])
    close_val = float(df['Close'].iloc[idx])
    
    total_range = high_val - low_val
    if total_range <= 0:
        return []
        
    # Helper to get pip multiplier
    try:
        from src.smc_detector import get_pip_multiplier
        pip_mult = get_pip_multiplier(symbol)
    except ImportError:
        pip_mult = 0.1 if "XAU" in symbol.upper() else 0.0001
        
    buffer = 20.0 * pip_mult
    
    # Define support and resistance levels at this index
    pp = df['pivot_PP'].iloc[idx]
    s_levels = {
        'PP': pp,
        'S1': df['pivot_S1'].iloc[idx],
        'S2': df['pivot_S2'].iloc[idx],
        'S3': df['pivot_S3'].iloc[idx],
        'S4': df['pivot_S4'].iloc[idx]
    }
    r_levels = {
        'PP': pp,
        'R1': df['pivot_R1'].iloc[idx],
        'R2': df['pivot_R2'].iloc[idx],
        'R3': df['pivot_R3'].iloc[idx],
        'R4': df['pivot_R4'].iloc[idx]
    }
    
    setups = []
    from src.rejection_detector import get_nearest_psychological_level
    
    # 1. Check Bullish (Buy) Rejections on Support levels
    bullish_candidates = []
    tolerance = 20.0 * pip_mult
    for name, lvl in s_levels.items():
        if pd.isna(lvl):
            continue
        # Check touch with tolerance (low can be slightly above the level, or level can go slightly into the body)
        body_max = max(open_val, close_val)
        if (low_val - tolerance) <= lvl <= (body_max + tolerance):
            lower_shadow = min(open_val, close_val) - low_val
            if lower_shadow / total_range >= 0.5:
                # Calculate distance to Low to find the closest level
                dist = abs(lvl - low_val)
                bullish_candidates.append((dist, lvl, name))
                
    if bullish_candidates:
        # Pick the level closest to Low
        bullish_candidates.sort(key=lambda x: x[0])
        _, best_lvl, best_name = bullish_candidates[0]
        
        entry = get_nearest_psychological_level(best_lvl)
        sl = min(best_lvl, entry) - buffer
        tp = entry + 2.0 * (entry - sl)
        
        setups.append({
            'direction': 1,
            'pivot_level': best_lvl,
            'pivot_name': best_name,
            'entry_price': entry,
            'sl_price': sl,
            'tp_price': tp,
            'option_name': f'Pivot Buy ({best_name})'
        })
        
    # 2. Check Bearish (Sell) Rejections on Resistance levels
    bearish_candidates = []
    for name, lvl in r_levels.items():
        if pd.isna(lvl):
            continue
        # Check touch with tolerance (high can be slightly below the level)
        body_min = min(open_val, close_val)
        if (body_min - tolerance) <= lvl <= (high_val + tolerance):
            upper_shadow = high_val - max(open_val, close_val)
            if upper_shadow / total_range >= 0.5:
                # Calculate distance to High to find the closest level
                dist = abs(lvl - high_val)
                bearish_candidates.append((dist, lvl, name))
                
    if bearish_candidates:
        # Pick the level closest to High
        bearish_candidates.sort(key=lambda x: x[0])
        _, best_lvl, best_name = bearish_candidates[0]
        
        entry = get_nearest_psychological_level(best_lvl)
        sl = max(best_lvl, entry) + buffer
        tp = entry - 2.0 * (sl - entry)
        
        setups.append({
            'direction': -1,
            'pivot_level': best_lvl,
            'pivot_name': best_name,
            'entry_price': entry,
            'sl_price': sl,
            'tp_price': tp,
            'option_name': f'Pivot Sell ({best_name})'
        })
        
    return setups
