import os
import sys
import pandas as pd
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
        print("Error connecting to MT5")
        return
        
    df = fetch_historical_data("XAUUSDm", mt5.TIMEFRAME_H4, 250)
    df = detect_swing_points(df)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol="XAUUSD")
    df = detect_bpr(df, symbol="XAUUSD")
    
    df['time'] = pd.to_datetime(df['time'])
    idx_16 = df[df['time'] == pd.to_datetime("2026-05-28 16:00:00")].index[0]
    
    print(f"Candle at index {idx_16} ({df['time'].iloc[idx_16]}):")
    print(f"  FVG Type: {df['FVG_Type'].iloc[idx_16]}")
    print(f"  BPR Type: {df['BPR_Type'].iloc[idx_16]}")
    print(f"  BPR Top: {df['BPR_Top'].iloc[idx_16]}")
    print(f"  BPR Bottom: {df['BPR_Bottom'].iloc[idx_16]}")
    
if __name__ == '__main__':
    main()
