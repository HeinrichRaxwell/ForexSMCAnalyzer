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
    detect_bpr
)

def main():
    if not connect_mt5():
        print("Error: Could not connect to MT5.")
        return
        
    symbol = "XAUUSDm"
    df = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    
    # Run SMC detectors
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=symbol)
    
    # Print columns of interest for index 209 to 216
    df['time'] = pd.to_datetime(df['time'])
    
    target_start = pd.to_datetime("2026-05-27 12:00:00")
    target_end = pd.to_datetime("2026-05-29 00:00:00")
    
    sub_df = df[(df['time'] >= target_start) & (df['time'] <= target_end)]
    
    print("=== Raw OHLCV and FVG Detection around May 28th ===")
    for idx, row in sub_df.iterrows():
        print(f"Index {idx} | Time: {row['time']}")
        print(f"  O: {row['Open']:.3f}, H: {row['High']:.3f}, L: {row['Low']:.3f}, C: {row['Close']:.3f}, V: {row['Volume']}")
        print(f"  FVG Type: {row['FVG_Type']}, FVG Top: {row['FVG_Top']:.3f}, FVG Bottom: {row['FVG_Bottom']:.3f}")
        
    # Let's check the BPR logic step by step for index 214 (May 28 12:00)
    print("\n=== Executing BPR Overlap Check ===")
    # Find index of 12:00 candle
    idx_12 = df[df['time'] == pd.to_datetime("2026-05-28 12:00:00")].index[0]
    curr_type = df['FVG_Type'].iloc[idx_12]
    curr_top = df['FVG_Top'].iloc[idx_12]
    curr_bottom = df['FVG_Bottom'].iloc[idx_12]
    
    print(f"Current FVG at index {idx_12} ({df['time'].iloc[idx_12]}): Type={curr_type}, Top={curr_top:.3f}, Bottom={curr_bottom:.3f}")
    
    # Look back 15 candles
    for k in range(idx_12 - 15, idx_12):
        prev_time = df['time'].iloc[k]
        prev_type = df['FVG_Type'].iloc[k]
        prev_top = df['FVG_Top'].iloc[k]
        prev_bottom = df['FVG_Bottom'].iloc[k]
        
        if prev_type is not None and pd.notna(prev_type):
            print(f"  Lookback index {k} ({prev_time}): Type={prev_type}, Top={prev_top:.3f}, Bottom={prev_bottom:.3f}")
            
            # Check overlap
            overlap_bottom = max(curr_bottom, prev_bottom)
            overlap_top = min(curr_top, prev_top)
            print(f"    Overlap range: {overlap_bottom:.3f} to {overlap_top:.3f}")
            if overlap_bottom < overlap_top:
                print("    -> Overlap EXISTS!")
            else:
                print("    -> No overlap.")

if __name__ == '__main__':
    main()
