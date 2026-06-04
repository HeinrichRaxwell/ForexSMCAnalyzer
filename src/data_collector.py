import os
import sys
import argparse
import MetaTrader5 as mt5
import pandas as pd

# Add project root to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data

def get_mt5_timeframe(tf) -> int:
    """
    Map standard timeframes to MT5 constants.
    
    Args:
        tf: Can be an integer or a string.
        
    Returns:
        int: The MT5 timeframe constant.
    """
    # Create timeframe mapping
    tf_map = {
        1: mt5.TIMEFRAME_M1,
        2: mt5.TIMEFRAME_M2,
        3: mt5.TIMEFRAME_M3,
        4: mt5.TIMEFRAME_M4,
        5: mt5.TIMEFRAME_M5,
        6: mt5.TIMEFRAME_M6,
        10: mt5.TIMEFRAME_M10,
        12: mt5.TIMEFRAME_M12,
        15: mt5.TIMEFRAME_M15,
        20: mt5.TIMEFRAME_M20,
        30: mt5.TIMEFRAME_M30,
        60: mt5.TIMEFRAME_H1,
        120: mt5.TIMEFRAME_H2,
        180: mt5.TIMEFRAME_H3,
        240: mt5.TIMEFRAME_H4,
        360: mt5.TIMEFRAME_H6,
        480: mt5.TIMEFRAME_H8,
        720: mt5.TIMEFRAME_H12,
        1440: mt5.TIMEFRAME_D1,
        "1": mt5.TIMEFRAME_M1,
        "5": mt5.TIMEFRAME_M5,
        "15": mt5.TIMEFRAME_M15,
        "30": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    
    if isinstance(tf, str):
        tf_lower = tf.lower()
        if tf_lower in tf_map:
            return tf_map[tf_lower]
        try:
            tf = int(tf)
        except ValueError:
            pass
            
    if tf in tf_map:
        return tf_map[tf]
        
    # Default fallback
    return tf

def download_bulk_data(symbol="XAUUSD", timeframe=15, num_candles=50000, output_dir="data") -> str:
    """
    Connect to MT5, fetch historical candle data and save to a CSV file.
    
    Args:
        symbol (str): The symbol to fetch (e.g. 'XAUUSD').
        timeframe (int/str): The timeframe to fetch.
        num_candles (int): The number of candles to fetch.
        output_dir (str): The folder where the CSV will be saved.
        
    Returns:
        str: Path to the saved CSV file.
    """
    print("Initializing MT5 connection for bulk download...")
    if not connect_mt5():
        raise RuntimeError("Failed to connect to MetaTrader 5 terminal.")
        
    try:
        # Resolve timeframe
        tf_constant = get_mt5_timeframe(timeframe)
        
        # Check active symbol variations (like in data_loader.py)
        symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
        active_symbol = None
        
        for sym in symbols_to_try:
            mt5.symbol_select(sym, True)
            rates = mt5.copy_rates_from_pos(sym, tf_constant, 0, 1)
            if rates is not None and len(rates) > 0:
                active_symbol = sym
                break
                
        if active_symbol is None:
            active_symbol = symbol
            
        print(f"Active symbol resolved to: {active_symbol}")
        print(f"Fetching {num_candles} candles for {active_symbol} on timeframe constant {tf_constant}...")
        
        # Fetch data
        df = fetch_historical_data(active_symbol, tf_constant, num_candles)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Build output file name
        output_file = os.path.join(output_dir, f"historical_{active_symbol.lower()}.csv")
        
        # Save DataFrame
        df.to_csv(output_file, index=False)
        print(f"Successfully downloaded {len(df)} bars and saved to: {output_file}")
        
        return output_file
        
    finally:
        mt5.shutdown()
        print("MT5 connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download bulk historical data from MT5.")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol to download (default: XAUUSD)")
    parser.add_argument("--timeframe", type=str, default="15", help="Timeframe (default: 15 or 15m)")
    parser.add_argument("--candles", type=int, default=50000, help="Number of candles to download (default: 50000)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory (default: data)")
    
    args = parser.parse_args()
    
    try:
        download_bulk_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            num_candles=args.candles,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Error downloading bulk data: {e}")
        import sys
        sys.exit(1)
