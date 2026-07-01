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
    
    m15_df = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 30)
    
    m15_df = detect_swing_points(m15_df, window=5)
    m15_df = detect_structures(m15_df)
    m15_df = detect_fvg_and_ob(m15_df, symbol=symbol)
    
    print("\nLast 25 M15 Candles:")
    print(m15_df[['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'FVG_Type', 'FVG_Top', 'FVG_Bottom']].tail(25).to_string())
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
