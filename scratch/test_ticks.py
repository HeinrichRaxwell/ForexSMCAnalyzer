import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

def main():
    print("Connecting to MT5...")
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
        
    print("MT5 Connected.")
    symbol = "XAUUSD"
    # Try alternate symbols
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    active_sym = None
    for sym in symbols_to_try:
        mt5.symbol_select(sym, True)
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 1)
        if rates is not None and len(rates) > 0:
            active_sym = sym
            break
            
    if not active_sym:
        print("XAUUSD symbol not found.")
        mt5.shutdown()
        return
        
    print(f"Active symbol: {active_sym}")
    
    # Try fetching ticks for the last 1 hour
    date_to = datetime.now()
    date_from = date_to - timedelta(hours=1)
    print(f"Fetching ticks from {date_from} to {date_to}...")
    ticks = mt5.copy_ticks_range(active_sym, date_from, date_to, mt5.COPY_TICKS_ALL)
    
    if ticks is None:
        print("Failed to fetch ticks, error:", mt5.last_error())
    else:
        print(f"Successfully fetched {len(ticks)} ticks.")
        if len(ticks) > 0:
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            print(df.head())
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
