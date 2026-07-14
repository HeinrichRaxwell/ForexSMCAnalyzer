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
            
    save_active_account_info()
    return True


def get_current_account_login() -> int | None:
    """Query the currently logged in MT5 account number, returning None if not connected."""
    try:
        acc = mt5.account_info()
        if acc is not None:
            login = getattr(acc, "login", None)
            if login:
                return int(login)
    except Exception:
        pass
    return None


def get_active_account_login() -> int | None:
    """Get the active login ID: either from active connection or from data/active_account.json."""
    login = get_current_account_login()
    if login:
        return login
    
    # Try reading from active_account.json
    try:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "active_account.json")
        if os.path.exists(path):
            import json
            with open(path, "r") as f:
                info = json.load(f)
                return info.get("login")
    except Exception:
        pass
    return None


def save_active_account_info():
    """Query current MT5 account info and write it to data/active_account.json."""
    import json
    try:
        acc = mt5.account_info()
        if acc is not None:
            info = {
                "login": getattr(acc, "login", None),
                "name": getattr(acc, "name", None),
                "server": getattr(acc, "server", None),
                "trade_mode": getattr(acc, "trade_mode", None), # 0 = DEMO, 1 = CONTEST, 2 = REAL
                "balance": getattr(acc, "balance", 0.0),
                "currency": getattr(acc, "currency", "USD"),
            }
            info["is_real"] = getattr(acc, "trade_mode", 0) == 2 or "real" in str(getattr(acc, "server", "")).lower()
            
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "active_account.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(info, f, indent=4)
            print(f"[MT5 Connection] Logged active account info: {info['login']} ({'REAL' if info['is_real'] else 'DEMO'})")
    except Exception as e:
        print(f"[MT5 Connection] Failed to log active account info: {e}")

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

