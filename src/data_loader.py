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
    
    Args:
        symbol (str): The trading symbol (e.g., 'XAUUSD').
        timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M15).
        num_candles (int): The number of candles to fetch.
        
    Returns:
        pd.DataFrame: Formatted DataFrame with columns ['time', 'Open', 'High', 'Low', 'Close', 'Volume'].
        
    Raises:
        ValueError: If data fetching fails or returned data is empty.
    """
    rates = mt5.copy_rates_from_now(symbol, timeframe, num_candles)
    if rates is None or len(rates) == 0:
        raise ValueError(f"Failed to fetch historical data for symbol: {symbol}")
    
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
