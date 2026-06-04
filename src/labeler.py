import os
import sys
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Add project root to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_bpr

def get_killzone(hour: int) -> int:
    """
    Get session killzone code relative to MT5 server time:
    1 for London (~09:00-12:00 MT5 time / 07:00-10:00 GMT)
    2 for NY (~14:00-17:00 MT5 time / 12:00-15:00 GMT)
    3 for Asia (~02:00-07:00 MT5 time / 00:00-05:00 GMT)
    0 for none.
    """
    if 9 <= hour < 12:
        return 1
    elif 14 <= hour < 17:
        return 2
    elif 2 <= hour < 7:
        return 3
    else:
        return 0

def get_timeframe_delta(df: pd.DataFrame) -> pd.Timedelta:
    """Estimate timeframe delta from the first two candles."""
    if len(df) >= 2:
        return df['time'].iloc[1] - df['time'].iloc[0]
    return pd.Timedelta(minutes=15) # Default fallback

def resolve_ambiguity_with_ticks(symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp, direction: int, entry: float, sl: float, tp: float, is_filled: bool) -> tuple:
    """
    Fetch tick data from MT5 for the specified time range and simulate the exact path.
    Returns: (is_filled_now, resolved_outcome)
    resolved_outcome is 1.0 (win), 0.0 (loss), or None (still open/no tick data).
    """
    if mt5 is None:
        return is_filled, None # Fallback: let caller handle candle-level logic
        
    # Make sure MT5 is initialized
    if not mt5.initialize():
        return is_filled, None
        
    # Try to resolve symbol suffix
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    active_sym = None
    for sym in symbols_to_try:
        mt5.symbol_select(sym, True)
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 1)
        if rates is not None and len(rates) > 0:
            active_sym = sym
            break
            
    if not active_sym:
        return is_filled, None
        
    dt_from = start_time.to_pydatetime()
    dt_to = end_time.to_pydatetime()
    
    ticks = mt5.copy_ticks_range(active_sym, dt_from, dt_to, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return is_filled, None

    # Verify price scale compatibility (avoid mixing synthetic test data with real ticks)
    first_tick_price = ticks[0]['bid']
    if abs(first_tick_price - entry) > entry * 0.5:
        return is_filled, None
        
    filled = is_filled
    for tick in ticks:
        price = tick['bid']
        ask_price = tick['ask']
        
        if not filled:
            if direction == 1: # Buy
                if ask_price <= entry:
                    filled = True
            else: # Sell
                if price >= entry:
                    filled = True
                    
            if not filled:
                if direction == 1:
                    if price <= sl: # Invalidated before entry
                        return False, None
                    if price >= tp: # Mitigated before entry
                        return False, None
                else:
                    if price >= sl: # Invalidated
                        return False, None
                    if price <= tp: # Mitigated
                        return False, None
                        
        if filled:
            if direction == 1:
                if price <= sl:
                    return True, 0.0 # Loss
                if price >= tp:
                    return True, 1.0 # Win
            else:
                if price >= sl:
                    return True, 0.0 # Loss
                if price <= tp:
                    return True, 1.0 # Win
                    
    return filled, None

def simulate_trade(df: pd.DataFrame, start_idx: int, direction: int, sl: float, tp: float, entry: float = None, symbol: str = "XAUUSD") -> float:
    """
    Simulate trade entry and outcome using tick data for ambiguous candles.
    If entry is None, assumes immediate entry at start_idx (for backward compatibility).
    
    Returns:
        1.0 if Win (TP hit first)
        0.0 if Loss (SL hit first)
        None if trade doesn't resolve by the end of the dataframe
    """
    if entry is None:
        # Backward compatible mode (no entry trigger required, assumes filled immediately)
        for j in range(start_idx, len(df)):
            high = df['High'].iloc[j]
            low = df['Low'].iloc[j]
            
            if direction == 1:  # Bullish (Buy)
                is_sl_hit = (low <= sl)
                is_tp_hit = (high >= tp)
                if is_sl_hit and is_tp_hit:
                    return 0.0
                elif is_sl_hit:
                    return 0.0
                elif is_tp_hit:
                    return 1.0
            else:  # Bearish (Sell)
                is_sl_hit = (high >= sl)
                is_tp_hit = (low <= tp)
                if is_sl_hit and is_tp_hit:
                    return 0.0
                elif is_sl_hit:
                    return 0.0
                elif is_tp_hit:
                    return 1.0
        return None

    # Full retracement & tick resolution simulation
    tf_delta = get_timeframe_delta(df)
    filled = False
    
    # Initialize connection once if needed
    mt5_active = False
    if mt5 is not None:
        mt5_active = mt5.initialize()
        
    for j in range(start_idx, len(df)):
        high = df['High'].iloc[j]
        low = df['Low'].iloc[j]
        candle_time = df['time'].iloc[j]
        
        if not filled:
            # Check entry conditions
            if direction == 1: # Buy
                has_entry = (low <= entry)
                has_sl = (low <= sl)
                has_tp = (high >= tp)
                
                if has_entry and (has_sl or has_tp):
                    outcome = None
                    if mt5_active:
                        filled, outcome = resolve_ambiguity_with_ticks(symbol, candle_time, candle_time + tf_delta, direction, entry, sl, tp, False)
                    if outcome is not None:
                        return outcome
                    if has_sl:
                        return 0.0
                    if has_tp:
                        return 1.0
                elif has_entry:
                    filled = True
                elif has_sl or has_tp:
                    return None
            else: # Sell
                has_entry = (high >= entry)
                has_sl = (high >= sl)
                has_tp = (low <= tp)
                
                if has_entry and (has_sl or has_tp):
                    outcome = None
                    if mt5_active:
                        filled, outcome = resolve_ambiguity_with_ticks(symbol, candle_time, candle_time + tf_delta, direction, entry, sl, tp, False)
                    if outcome is not None:
                        return outcome
                    if has_sl:
                        return 0.0
                    if has_tp:
                        return 1.0
                elif has_entry:
                    filled = True
                elif has_sl or has_tp:
                    return None
                    
        # If filled
        if filled:
            if direction == 1:
                is_sl_hit = (low <= sl)
                is_tp_hit = (high >= tp)
                
                if is_sl_hit and is_tp_hit:
                    outcome = None
                    if mt5_active:
                        _, outcome = resolve_ambiguity_with_ticks(symbol, candle_time, candle_time + tf_delta, direction, entry, sl, tp, True)
                    if outcome is not None:
                        return outcome
                    return 0.0
                elif is_sl_hit:
                    return 0.0
                elif is_tp_hit:
                    return 1.0
            else:
                is_sl_hit = (high >= sl)
                is_tp_hit = (low <= tp)
                
                if is_sl_hit and is_tp_hit:
                    outcome = None
                    if mt5_active:
                        _, outcome = resolve_ambiguity_with_ticks(symbol, candle_time, candle_time + tf_delta, direction, entry, sl, tp, True)
                    if outcome is not None:
                        return outcome
                    return 0.0
                elif is_sl_hit:
                    return 0.0
                elif is_tp_hit:
                    return 1.0

                    
    return None


def label_smc_setups(df: pd.DataFrame, buffer: float = 0.5, symbol: str = "XAUUSD") -> pd.DataFrame:
    """
    Load historical data, run SMC detection algorithms, and simulate trade setups
    to label them with outcomes.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV.
        buffer (float): Distance in USD to add to SL (below Low for Buy, above High for Sell).
        symbol (str): Symbol name (e.g., "XAUUSD", "EURUSD").
        
    Returns:
        pd.DataFrame: A DataFrame of labeled setups with features.
    """
    df = df.copy()
    
    # Ensure time is datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Estimate timeframe
    tf_delta = get_timeframe_delta(df)
    timeframe_minutes = int(tf_delta.total_seconds() / 60)
    
    # Run SMC detectors
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=symbol)
    df = detect_bpr(df, symbol=symbol)
    
    # Calculate ATR_14
    close_prev = df['Close'].shift(1).fillna(df['Open'])
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            np.abs(df['High'] - close_prev),
            np.abs(df['Low'] - close_prev)
        )
    )
    df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    setups = []
    
    for i in range(len(df)):
        # Skip if ATR_14 is NaN
        if pd.isna(df['ATR_14'].iloc[i]):
            continue
            
        t_val = df['time'].iloc[i]
        hour_val = int(t_val.hour)
        day_of_week_val = int(t_val.dayofweek)
        trend_val = int(df['Trend'].iloc[i])
        killzone_val = get_killzone(hour_val)
        atr_val = df['ATR_14'].iloc[i]
        
        # 1. Check FVG Setup
        fvg_type = df['FVG_Type'].iloc[i] if 'FVG_Type' in df.columns else None
        if pd.notna(fvg_type) and fvg_type is not None:
            fvg_top = df['FVG_Top'].iloc[i]
            fvg_bottom = df['FVG_Bottom'].iloc[i]
            fvg_sl = df['FVG_SL'].iloc[i]
            
            if fvg_type == 'BULLISH':
                direction = 1
                fvg_width = df['Low'].iloc[i] - df['High'].iloc[i-2]
            else: # BEARISH
                direction = -1
                fvg_width = df['Low'].iloc[i-2] - df['High'].iloc[i]
                
            for entry_col in ['FVG_Fibo_0.5', 'FVG_Fibo_0.618']:
                entry = df[entry_col].iloc[i]
                sl = fvg_sl
                tp = df['FVG_Fibo_0.0'].iloc[i]
                
                risk_pips = abs(entry - sl)
                    
                label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'timeframe': timeframe_minutes,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 0,  # FVG
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': fvg_width,
                        'relative_fvg_width': fvg_width / atr_val,
                        'label': int(label)
                    })
                    
        # 2. Check OB Setup
        ob_type = df['OB_Type'].iloc[i] if 'OB_Type' in df.columns else None
        if pd.notna(ob_type) and ob_type is not None:
            ob_top = df['OB_Top'].iloc[i]
            ob_bottom = df['OB_Bottom'].iloc[i]
            ob_sl = df['OB_SL'].iloc[i]
            direction = 1 if ob_type == 'BULLISH' else -1
                
            for entry_col in ['OB_Fibo_0.5', 'OB_Fibo_0.618']:
                entry = df[entry_col].iloc[i]
                sl = ob_sl
                tp = df['OB_Fibo_0.0'].iloc[i]
                risk_pips = abs(entry - sl)
                
                label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'timeframe': timeframe_minutes,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # OB
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'label': int(label)
                    })

        # 3. Check BPR Setup
        bpr_type = df['BPR_Type'].iloc[i] if 'BPR_Type' in df.columns else None
        if pd.notna(bpr_type) and bpr_type is not None:
            bpr_top = df['BPR_Top'].iloc[i]
            bpr_bottom = df['BPR_Bottom'].iloc[i]
            bpr_sl = df['BPR_SL'].iloc[i]
            direction = 1 if bpr_type == 'BULLISH' else -1
            
            for entry_col in ['BPR_Fibo_0.5', 'BPR_Fibo_0.618']:
                entry = df[entry_col].iloc[i]
                sl = bpr_sl
                tp = df['BPR_Fibo_0.0'].iloc[i]
                risk_pips = abs(entry - sl)
                
                label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'timeframe': timeframe_minutes,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 0,  # Treat as FVG for ML
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': abs(bpr_top - bpr_bottom),
                        'relative_fvg_width': abs(bpr_top - bpr_bottom) / atr_val,
                        'label': int(label)
                    })
                    
    return pd.DataFrame(setups)

def main():
    print("=== Trade Simulation & Dataset Labeling Engine ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_path = os.path.join(data_dir, 'labeled_setups.csv')
    
    # We look for timeframe-specific files first
    tf_files = {
        '15': 'historical_xauusdm_15.csv',
        '30': 'historical_xauusdm_30.csv',
        '1h': 'historical_xauusdm_1h.csv',
        '4h': 'historical_xauusdm_4h.csv',
        '1d': 'historical_xauusdm_1d.csv'
    }
    
    all_labeled_dfs = []
    
    # Try loading files for each timeframe
    files_processed = 0
    for tf, fname in tf_files.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            print(f"Loading historical data for TF {tf} from {fpath}...")
            df = pd.read_csv(fpath)
            print(f"Loaded {len(df)} candles. Simulating setups...")
            labeled_df = label_smc_setups(df)
            print(f"Generated {len(labeled_df)} labeled setups.")
            all_labeled_dfs.append(labeled_df)
            files_processed += 1
            
    # Fall back to historical_xauusdm.csv if no timeframe files exist
    if files_processed == 0:
        fallback_path = os.path.join(data_dir, 'historical_xauusdm.csv')
        if os.path.exists(fallback_path):
            print(f"No timeframe-specific files found. Falling back to default: {fallback_path}")
            df = pd.read_csv(fallback_path)
            labeled_df = label_smc_setups(df)
            all_labeled_dfs.append(labeled_df)
        else:
            print(f"Error: No historical data files found in {data_dir}")
            return
            
    # Combine all setups
    combined_df = pd.concat(all_labeled_dfs, ignore_index=True)
    
    if combined_df.empty:
        print("No valid resolved setups detected across any timeframe.")
        return
        
    total_trades = len(combined_df)
    wins = (combined_df['label'] == 1).sum()
    losses = (combined_df['label'] == 0).sum()
    winrate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
    
    print("\n--- Simulation Summary ---")
    print(f"Total resolved setups: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Raw SMC Winrate (Tick-Accurate Retracement): {winrate:.2f}%")
    print("--------------------------\n")
    
    # Save the output CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Combined labeled setups saved to: {output_path}")

if __name__ == "__main__":
    main()

