import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def connect_mt5() -> bool:
    """
    Initialize connection to the MetaTrader 5 terminal and login to the specific account if credentials exist.
    
    Returns:
        bool: True if connection is successful, False otherwise.
    """
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        return False
        
    login_id = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    if login_id and password and server:
        try:
            login_num = int(login_id)
            if mt5.login(login=login_num, password=password, server=server):
                print(f"[MT5 Connection] Successfully logged in to account #{login_num} on server {server}")
            else:
                print(f"[MT5 Connection] Login failed for account #{login_num} on server {server}, error code =", mt5.last_error())
                return False
        except ValueError:
            print("[MT5 Connection] Invalid MT5_LOGIN format in .env, must be an integer.")
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
    
    # copy_rates_from_pos(pos=0) always includes the currently-forming candle
    # as the LAST row. Mark this so smc_detector.py can exclude it from lookahead
    # windows and prevent swing point repainting.
    df.attrs["has_running_candle"] = True
    
    # Return formatted columns
    return df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]

def fetch_historical_data_range(symbol: str, timeframe: int, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """
    Fetch historical candlestick (OHLCV) data from MT5 within a specific date range.
    Tries common broker variations (e.g., symbol suffix 'm' for Exness).
    
    Args:
        symbol (str): The trading symbol (e.g., 'XAUUSD').
        timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M15).
        date_from (datetime): Start date.
        date_to (datetime): End date.
        
    Returns:
        pd.DataFrame: Formatted DataFrame with columns ['time', 'Open', 'High', 'Low', 'Close', 'Volume'].
        
    Raises:
        ValueError: If data fetching fails or returned data is empty.
    """
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    rates = None
    active_symbol = None
    
    for sym in symbols_to_try:
        mt5.symbol_select(sym, True)
        rates = mt5.copy_rates_range(sym, timeframe, date_from, date_to)
        if rates is not None and len(rates) > 0:
            active_symbol = sym
            print(f"Successfully fetched range data for symbol: {sym}")
            break
            
    if rates is None or len(rates) == 0:
        raise ValueError(f"Failed to fetch historical data for symbol: {symbol} in range {date_from} to {date_to}")
    
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
    
    # copy_rates_range fetches by date range (historical). The live candle is
    # only included if date_to is in the future. Mark conservatively as False
    # since this is used for historical analysis and backtesting.
    df.attrs["has_running_candle"] = False
    
    return df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]

