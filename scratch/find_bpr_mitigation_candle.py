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
    df_h4 = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    df_h4 = detect_swing_points(df_h4)
    df_h4 = detect_structures(df_h4)
    df_h4 = detect_fvg_and_ob(df_h4, symbol=symbol)
    df_h4 = detect_bpr(df_h4, symbol=symbol)
    
    bpr_bottom = df_h4['BPR_Bottom'].iloc[215]
    bpr_top = df_h4['BPR_Top'].iloc[215]
    print(f"BPR detected at index 215 (Time: {df_h4['time'].iloc[215]}), Bottom: {bpr_bottom}, Top: {bpr_top}")
    
    for i in range(216, len(df_h4)):
        close_val = df_h4['Close'].iloc[i]
        low_val = df_h4['Low'].iloc[i]
        high_val = df_h4['High'].iloc[i]
        
        # Check mitigation (close below bottom for Bullish BPR)
        if close_val < bpr_bottom:
            print(f"MITIGATION EVENT: Candle {i} ({df_h4['time'].iloc[i]}) closed at {close_val} which is below BPR Bottom {bpr_bottom}")
            break
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
