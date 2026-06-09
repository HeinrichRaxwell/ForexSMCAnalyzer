import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import (
    detect_swing_points, 
    detect_structures, 
    detect_fvg_and_ob, 
    detect_snr_and_swapzones, 
    detect_bpr, 
    detect_indecision_candles
)
from src.labeler import get_killzone
from src.inference import predict_setup_probability

def evaluate_setup(s, df):
    # Extract features matching model trainer
    features = {
        'timeframe': 240, # H4
        'hour': s['hour'],
        'day_of_week': s['day_of_week'],
        'setup_type': s['setup_type'],
        'direction': s['direction'],
        'entry_price': s['entry_price'],
        'sl_price': s['sl_price'],
        'tp_price': s['tp_price'],
        'risk_pips': s['risk_pips'],
        'atr_14': s['atr_14'],
        'trend': s['trend'],
        'relative_risk': s['relative_risk'],
        'killzone': s['killzone'],
        'fvg_width': s['fvg_width'],
        'relative_fvg_width': s['relative_fvg_width']
    }
    prob = predict_setup_probability(features)
    return prob, features

def main():
    if not connect_mt5():
        print("Error: Could not connect to MT5.")
        return
        
    symbol = "XAUUSDm"
    df = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    if df is None or df.empty:
        print("Error: Failed to fetch H4 data.")
        return
        
    # Run SMC detectors
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=symbol)
    df = detect_snr_and_swapzones(df, symbol=symbol)
    df = detect_bpr(df, symbol=symbol)
    df = detect_indecision_candles(df, symbol=symbol)
    
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
    df['time'] = pd.to_datetime(df['time'])
    
    # Check 1: Thu 28 May 2026 (04:00 - 20:00)
    print("\n==================================================")
    print("ANALYSIS FOR THURSDAY, 28 MAY 2026 (H4)")
    print("==================================================")
    start_may28 = pd.to_datetime("2026-05-28 00:00:00")
    end_may28 = pd.to_datetime("2026-05-28 23:59:59")
    may28_df = df[(df['time'] >= start_may28) & (df['time'] <= end_may28)]
    print(may28_df[['time', 'Open', 'High', 'Low', 'Close', 'FVG_Type', 'BPR_Type', 'BPR_Top', 'BPR_Bottom', 'IC_Type', 'IC_Top', 'IC_Bottom']])
    
    # Check 2: Wed 3 June 2026 (00:00 - 23:59)
    print("\n==================================================")
    print("ANALYSIS FOR WEDNESDAY, 3 JUNE 2026 (H4)")
    print("==================================================")
    start_june3 = pd.to_datetime("2026-06-03 00:00:00")
    end_june3 = pd.to_datetime("2026-06-03 23:59:59")
    june3_df = df[(df['time'] >= start_june3) & (df['time'] <= end_june3)]
    print(june3_df[['time', 'Open', 'High', 'Low', 'Close', 'FVG_Type', 'BPR_Type', 'IC_Type', 'IC_Top', 'IC_Bottom', 'IC_Mitigated']])
    
    # Get active setups evaluated
    from src.main import get_active_setups
    setups = get_active_setups(df)
    
    print("\n==================================================")
    print("EVALUATING SPECIFIC SETUPS")
    print("==================================================")
    
    # Filter setups in the range of May 28 and June 3
    interest_start = pd.to_datetime("2026-05-27 00:00:00")
    interest_end = pd.to_datetime("2026-06-05 12:00:00")
    
    found_any = False
    for s in setups:
        setup_time = pd.to_datetime(s['time'])
        if setup_time >= interest_start and setup_time <= interest_end:
            # We want to identify the BPR or IC setups
            option_name = s.get('option_name', '')
            if 'BPR' in option_name or 'IC' in option_name or 'Doji' in option_name:
                found_any = True
                prob, features = evaluate_setup(s, df)
                print(f"Setup at {s['time']}: {option_name}")
                print(f"  Direction: {'BUY' if s['direction'] == 1 else 'SELL'}, Entry: {s['entry_price']:.3f}, SL: {s['sl_price']:.3f}, TP: {s['tp_price']:.3f}")
                print(f"  Confidence: {prob:.2%}")
                # We check if it triggers/mitigates
                print(f"  Status (threshold 0.50): {'PASS (Trade Placed)' if prob >= 0.50 else 'Filtered (Low Confidence)'}")
                print(f"  Status (threshold 0.70): {'PASS (Trade Placed)' if prob >= 0.70 else 'Filtered (Low Confidence)'}")
                
    if not found_any:
        print("No active BPR or IC setups in this date range found in get_active_setups.")

if __name__ == '__main__':
    main()
