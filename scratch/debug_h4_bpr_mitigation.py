import os
import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_snr_and_swapzones, detect_bpr

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
        
    symbol = "XAUUSD"
    df_h4 = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    
    df_h4 = detect_swing_points(df_h4)
    df_h4 = detect_structures(df_h4)
    df_h4 = detect_fvg_and_ob(df_h4, symbol=symbol)
    df_h4 = detect_bpr(df_h4, symbol=symbol)
    
    # Let's locate the BPR at index 215 (or around 2026-05-28)
    bpr_idx = None
    for idx, row in df_h4.iterrows():
        if "2026-05-28" in str(row['time']):
            bpr_idx = idx
            print(f"\nFound candle on 2026-05-28:")
            print(f"Index: {idx} | Time: {row['time']}")
            print(f"Open: {row['Open']} | High: {row['High']} | Low: {row['Low']} | Close: {row['Close']}")
            print(f"BPR Type: {row['BPR_Type']} | BPR Top: {row['BPR_Top']} | BPR Bottom: {row['BPR_Bottom']}")
            print(f"BPR Fibo 0.5: {row['BPR_Fibo_0.5']} | BPR Fibo 0.618: {row['BPR_Fibo_0.618']}")
            
    if bpr_idx is not None:
        print("\nSubsequent candles:")
        # Look at the candles following the BPR
        for j in range(bpr_idx, min(len(df_h4), bpr_idx + 15)):
            row = df_h4.iloc[j]
            # Check if this candle mitigates the BPR (for BULLISH, close < BPR Bottom)
            is_mitigated = False
            bpr_bottom = df_h4['BPR_Bottom'].iloc[bpr_idx]
            close_val = row['Close']
            if close_val < bpr_bottom:
                is_mitigated = True
            print(f"Index: {j} | Time: {row['time']} | High: {row['High']:.3f} | Low: {row['Low']:.3f} | Close: {row['Close']:.3f} | Mitigates BPR? {is_mitigated}")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
