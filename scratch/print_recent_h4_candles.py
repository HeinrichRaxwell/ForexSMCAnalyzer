import os
import sys
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
    symbol = "XAUUSD"
    df_h4 = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    
    print("Printing H4 candles from 2026-06-02 onwards:")
    for idx, row in df_h4.iterrows():
        if "2026-06-02" in str(row['time']) or "2026-06-03" in str(row['time']) or "2026-06-04" in str(row['time']) or "2026-06-05" in str(row['time']):
            print(f"Index: {idx} | Time: {row['time']} | Open: {row['Open']:.3f} | High: {row['High']:.3f} | Low: {row['Low']:.3f} | Close: {row['Close']:.3f}")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
