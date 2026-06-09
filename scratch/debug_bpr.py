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
    detect_fvg_and_ob
)

def main():
    if not connect_mt5():
        print("Error: Could not connect to MT5.")
        return
        
    symbol = "XAUUSDm"
    df = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=symbol)
    
    # Let's run detect_bpr logic here manually with print statements
    df = df.copy()
    df['BPR_Type'] = None
    df['BPR_Top'] = np.nan
    df['BPR_Bottom'] = np.nan
    
    fvg_types = df['FVG_Type'].values
    fvg_tops = df['FVG_Top'].values
    fvg_bottoms = df['FVG_Bottom'].values
    
    idx_12 = df[df['time'] == pd.to_datetime("2026-05-28 16:00:00")].index[0]
    print(f"Target index: {idx_12} | Time: {df['time'].iloc[idx_12]}")
    print(f"FVG Type: {fvg_types[idx_12]}, Top: {fvg_tops[idx_12]}, Bottom: {fvg_bottoms[idx_12]}")
    
    # Look back check
    lookback = 15
    start_k = max(2, idx_12 - lookback)
    for k in range(start_k, idx_12):
        prev_type = fvg_types[k]
        if prev_type is not None and pd.notna(prev_type):
            print(f"  Check index {k} ({df['time'].iloc[k]}): Type={prev_type}, Top={fvg_tops[k]}, Bottom={fvg_bottoms[k]}")
            
            # Check overlap
            overlap_bottom = max(fvg_bottoms[idx_12], fvg_bottoms[k])
            overlap_top = min(fvg_tops[idx_12], fvg_tops[k])
            print(f"    Overlap range: {overlap_bottom} to {overlap_top}")
            if overlap_bottom < overlap_top:
                print(f"    -> OVERLAP EXISTS! Type mismatch: {prev_type != fvg_types[idx_12]}")
                if prev_type != fvg_types[idx_12]:
                    print("    -> Setting BPR!")
                    bpr_type = 'BULLISH' if fvg_types[idx_12] == 'BULLISH' else 'BEARISH'
                    df.at[df.index[idx_12], 'BPR_Type'] = bpr_type
                    df.at[df.index[idx_12], 'BPR_Top'] = overlap_top
                    df.at[df.index[idx_12], 'BPR_Bottom'] = overlap_bottom
                    break

if __name__ == '__main__':
    main()
