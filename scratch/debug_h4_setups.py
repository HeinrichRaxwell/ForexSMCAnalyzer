import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob
from src.main import get_active_setups

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
        
    symbol = "XAUUSD"
    # Fetch 250 candles (same as scanner_worker)
    h4_df = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    
    h4_df = detect_swing_points(h4_df)
    h4_df = detect_structures(h4_df)
    h4_df = detect_fvg_and_ob(h4_df, symbol=symbol)
    
    # Print columns
    print(f"Total candles fetched: {len(h4_df)}")
    
    # Find all rows with FVG
    fvg_rows = h4_df[h4_df['FVG_Type'].notna()]
    print("\n--- ALL DETECTED FVGs IN H4 (250 candles) ---")
    for idx, row in fvg_rows.iterrows():
        print(f"Index: {idx}, Time: {row['time']}, Type: {row['FVG_Type']}, Top: {row['FVG_Top']:.3f}, Bottom: {row['FVG_Bottom']:.3f}")
        
    # Get active setups
    setups = get_active_setups(h4_df)
    print("\n--- ACTIVE FVG SETUPS IN H4 ---")
    for s in setups:
        if s['setup_type'] == 0:
            print(f"Index: {s['index']}, Time: {s['time']}, Opt: {s['option_name']}, Entry: {s['entry_price']:.3f}, SL: {s['sl_price']:.3f}")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
