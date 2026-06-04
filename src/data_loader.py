import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def connect_mt5() -> bool:
    """
    Initialize connection to the MetaTrader 5 terminal.
    
    Returns:
        bool: True if connection is successful, False otherwise.
    """
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        return False
    return True

def fetch_historical_data(symbol: str, timeframe: int, num_candles: int) -> pd.DataFrame:
    """
    Fetch historical candlestick (OHLCV) data from MT5 and format columns.
    Tries common broker variations (e.g., symbol suffix 'm' for Exness).
    
    Args:
        symbol (str): The trading symbol (e.g., 'XAUUSD').
        timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M15).
        num_candles (int): The number of candles to fetch.
        
    Returns:
        pd.DataFrame: Formatted DataFrame with columns ['time', 'Open', 'High', 'Low', 'Close', 'Volume'].
        
    Raises:
        ValueError: If data fetching fails or returned data is empty.
    """
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    rates = None
    active_symbol = None
    
    for sym in symbols_to_try:
        # Select the symbol in Market Watch first
        mt5.symbol_select(sym, True)
        rates = mt5.copy_rates_from_pos(sym, timeframe, 0, num_candles)
        if rates is not None and len(rates) > 0:
            active_symbol = sym
            print(f"Successfully fetched data for symbol: {sym}")
            break
            
    if rates is None or len(rates) == 0:
        raise ValueError(f"Failed to fetch historical data for symbol: {symbol} (tried variations: {symbols_to_try})")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(
        columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        },
        inplace=True
    )
    
    # Return formatted columns
    return df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]
