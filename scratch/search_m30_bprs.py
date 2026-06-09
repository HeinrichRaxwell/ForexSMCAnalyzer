import os
import sys
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_bpr

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
    symbol = "XAUUSD"
    # Fetch a larger M30 dataset to cover late May 2026
    df_m30 = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 2000)
    df_m30 = detect_swing_points(df_m30)
    df_m30 = detect_structures(df_m30)
    df_m30 = detect_fvg_and_ob(df_m30, symbol=symbol)
    df_m30 = detect_bpr(df_m30, symbol=symbol)
    
    print("\n--- DETECTED BPRs ON M30 (LATE MAY / EARLY JUNE 2026) ---")
    bprs = df_m30[df_m30['BPR_Type'].notna()]
    for idx, row in bprs.iterrows():
        if "2026-05-28" in str(row['time']) or "2026-06-02" in str(row['time']) or "2026-06-03" in str(row['time']):
            print(f"Index: {idx} | Time: {row['time']} | Type: {row['BPR_Type']} | Top: {row['BPR_Top']:.3f} | Bottom: {row['BPR_Bottom']:.3f} | Mitigated: {row['BPR_Mitigated']}")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
