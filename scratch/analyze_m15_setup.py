import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob

def main():
    if not connect_mt5():
        print("Failed to initialize MT5")
        return
        
    symbol = "XAUUSD"
    import MetaTrader5 as mt5
    
    print("Fetching M15 historical data (150 candles)...")
    m15_df = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 150)
    if m15_df is None or m15_df.empty:
        print("Failed to fetch M15 data")
        mt5.shutdown()
        return
        
    print("\nRecent M15 Candles on 2026-06-05:")
    m15_df['time'] = pd.to_datetime(m15_df['time'])
    
    # Filter candles on June 5, 2026
    day_df = m15_df[m15_df['time'] >= '2026-06-05 00:00:00']
    print(day_df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']].to_string())
    
    # Run detectors
    m15_df = detect_swing_points(m15_df, window=5)
    m15_df = detect_structures(m15_df)
    m15_df = detect_fvg_and_ob(m15_df, symbol=symbol)
    
    # Check if there are any FVGs detected in the whole DataFrame
    print("\n=== All detected FVGs in M15 dataframe ===")
    fvg_cols = [c for c in m15_df.columns if 'FVG' in c]
    print(f"FVG Columns: {fvg_cols}")
    
    # Let's see if any row has FVG_Top or FVG_Bottom not NaN
    if 'FVG_Top' in m15_df.columns:
        detected_fvgs = m15_df[m15_df['FVG_Top'].notna()]
        if not detected_fvgs.empty:
            print(detected_fvgs[['time', 'FVG_Type', 'FVG_Top', 'FVG_Bottom', 'FVG_Fibo_0.5', 'FVG_Fibo_0.618']].to_string())
        else:
            print("No rows have FVG_Top.")
    else:
        print("No FVG columns exist after running detector.")
        
    mt5.shutdown()

if __name__ == "__main__":
    main()
