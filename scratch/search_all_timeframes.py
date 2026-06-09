import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob

def main():
    if not connect_mt5():
        print("Failed to initialize MT5")
        return
        
    symbol = "XAUUSD"
    import MetaTrader5 as mt5
    
    timeframes = {
        'M5': (mt5.TIMEFRAME_M5, 300),
        'M15': (mt5.TIMEFRAME_M15, 150),
        'M30': (mt5.TIMEFRAME_M30, 100),
        'H1': (mt5.TIMEFRAME_H1, 50)
    }
    
    print("Searching for any FVG created between 01:00 and 02:00 broker time on June 5, 2026...")
    
    for tf_name, (tf_code, lookback) in timeframes.items():
        df = fetch_historical_data(symbol, tf_code, lookback)
        if df is None or df.empty:
            continue
            
        df = detect_swing_points(df, window=5)
        df = detect_structures(df)
        df = detect_fvg_and_ob(df, symbol=symbol)
        
        df['time'] = pd.to_datetime(df['time'])
        
        # Filter for June 5, 2026, between 01:00:00 and 02:00:00
        start_t = '2026-06-05 00:30:00'
        end_t = '2026-06-05 02:30:00'
        
        mask = (df['time'] >= start_t) & (df['time'] <= end_t)
        tf_fvgs = df[mask & df['FVG_Top'].notna()]
        
        if not tf_fvgs.empty:
            print(f"\n=== Detected FVGs on {tf_name} in target range ===")
            print(tf_fvgs[['time', 'FVG_Type', 'FVG_Top', 'FVG_Bottom', 'FVG_Fibo_0.5', 'FVG_Fibo_0.618']].to_string())
            
            # Let's print the candles around the FVG to see the context
            for idx in tf_fvgs.index:
                print(f"\nContext candles for FVG at index {idx} ({tf_name}):")
                print(df.loc[idx-2:idx+1, ['time', 'Open', 'High', 'Low', 'Close', 'Volume']])
        else:
            print(f"No FVGs detected on {tf_name} in this range.")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
