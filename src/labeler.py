import os
import sys
import numpy as np
import pandas as pd

# Add project root to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob

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

def simulate_trade(df: pd.DataFrame, start_idx: int, direction: int, sl: float, tp: float) -> float:
    """
    Simulate the trade from start_idx onwards to find whether it hits TP or SL first.
    
    Returns:
        1.0 if Win (TP hit first)
        0.0 if Loss (SL hit first)
        None if trade doesn't resolve by the end of the dataframe
    """
    for j in range(start_idx, len(df)):
        high = df['High'].iloc[j]
        low = df['Low'].iloc[j]
        
        if direction == 1:  # Bullish (Buy)
            # If both are hit in the same candle, treat it as a Loss (0) to be conservative
            is_sl_hit = (low <= sl)
            is_tp_hit = (high >= tp)
            if is_sl_hit and is_tp_hit:
                return 0.0
            elif is_sl_hit:
                return 0.0
            elif is_tp_hit:
                return 1.0
        else:  # Bearish (Sell)
            # If both are hit in the same candle, treat it as a Loss (0) to be conservative
            is_sl_hit = (high >= sl)
            is_tp_hit = (low <= tp)
            if is_sl_hit and is_tp_hit:
                return 0.0
            elif is_sl_hit:
                return 0.0
            elif is_tp_hit:
                return 1.0
                
    return None

def label_smc_setups(df: pd.DataFrame, buffer: float = 0.5) -> pd.DataFrame:
    """
    Load historical data, run SMC detection algorithms, and simulate trade setups
    to label them with outcomes.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV.
        buffer (float): Distance in USD to add to SL (below Low for Buy, above High for Sell).
        
    Returns:
        pd.DataFrame: A DataFrame of labeled setups with features.
    """
    df = df.copy()
    
    # Ensure time is datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Run SMC detectors
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df)
    
    # Calculate ATR_14
    close_prev = df['Close'].shift(1).fillna(df['Open'])
    # True Range = max(High - Low, abs(High - Close_prev), abs(Low - Close_prev))
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
        
        # 1. Check FVG Setup
        fvg_type = df['FVG_Type'].iloc[i] if 'FVG_Type' in df.columns else None
        if pd.notna(fvg_type) and fvg_type is not None:
            fvg_top = df['FVG_Top'].iloc[i]
            fvg_bottom = df['FVG_Bottom'].iloc[i]
            
            if fvg_type == 'BULLISH':
                direction = 1
                entry = fvg_top
                sl = fvg_bottom - buffer
                tp = entry + (entry - sl) * 2
            elif fvg_type == 'BEARISH':
                direction = -1
                entry = fvg_bottom
                sl = fvg_top + buffer
                tp = entry - (sl - entry) * 2
            else:
                direction = None
                
            if direction is not None:
                label = simulate_trade(df, i + 1, direction, sl, tp)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 0,  # FVG
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': (entry - sl) if direction == 1 else (sl - entry),
                        'atr_14': df['ATR_14'].iloc[i],
                        'trend': trend_val,
                        'killzone': killzone_val,
                        'label': int(label)
                    })
                    
        # 2. Check OB Setup
        ob_type = df['OB_Type'].iloc[i] if 'OB_Type' in df.columns else None
        if pd.notna(ob_type) and ob_type is not None:
            ob_top = df['OB_Top'].iloc[i]
            ob_bottom = df['OB_Bottom'].iloc[i]
            
            if ob_type == 'BULLISH':
                direction = 1
                entry = ob_top
                sl = ob_bottom - buffer
                tp = entry + (entry - sl) * 2
            elif ob_type == 'BEARISH':
                direction = -1
                entry = ob_bottom
                sl = ob_top + buffer
                tp = entry - (sl - entry) * 2
            else:
                direction = None
                
            if direction is not None:
                label = simulate_trade(df, i + 1, direction, sl, tp)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # OB
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': (entry - sl) if direction == 1 else (sl - entry),
                        'atr_14': df['ATR_14'].iloc[i],
                        'trend': trend_val,
                        'killzone': killzone_val,
                        'label': int(label)
                    })
                    
    return pd.DataFrame(setups)

def main():
    print("=== Trade Simulation & Dataset Labeling Engine ===")
    
    # Check directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'historical_xauusdm.csv')
    output_path = os.path.join(base_dir, 'data', 'labeled_setups.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Historical data file not found at {data_path}")
        return
        
    print(f"Loading historical data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} candles.")
    
    print("Running SMC setup detection and trade simulation (labeling)...")
    labeled_df = label_smc_setups(df)
    
    if labeled_df.empty:
        print("No valid resolved setups detected.")
        return
        
    total_trades = len(labeled_df)
    wins = (labeled_df['label'] == 1).sum()
    losses = (labeled_df['label'] == 0).sum()
    winrate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
    
    print("\n--- Simulation Summary ---")
    print(f"Total resolved setups: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Raw SMC Winrate: {winrate:.2f}%")
    print("--------------------------\n")
    
    # Save the output CSV
    labeled_df.to_csv(output_path, index=False)
    print(f"Labeled setups saved to: {output_path}")

if __name__ == "__main__":
    main()
