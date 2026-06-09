import pandas as pd
import numpy as np

def get_nearest_psychological_level(price: float) -> float:
    """
    Get the nearest psychological level ending in 0 or 5.
    Examples: 2338.2 -> 2340.0, 2342.3 -> 2340.0, 2342.6 -> 2345.0
    """
    return 5.0 * round(price / 5.0)

def is_near_psychological_level(price: float, symbol: str = "XAUUSD") -> bool:
    """
    Check if the price is within 10 pips of a psychological level ending in 0 or 5.
    For XAUUSD, pip size is 0.1, so 10 pips = 1.0 USD.
    """
    try:
        from src.smc_detector import get_pip_multiplier
        pip_mult = get_pip_multiplier(symbol)
    except ImportError:
        # Fallback if import fails
        symbol_upper = symbol.upper()
        if "JPY" in symbol_upper:
            pip_mult = 0.01
        elif "XAUUSD" in symbol_upper or "GOLD" in symbol_upper:
            pip_mult = 0.1
        else:
            pip_mult = 0.0001
            
    threshold = 10.0 * pip_mult
    nearest = get_nearest_psychological_level(price)
    return abs(price - nearest) <= threshold

def detect_rejection_at_level(df: pd.DataFrame, entry_level: float, direction: int, lookback: int = 5, symbol: str = "XAUUSD") -> bool:
    """
    Evaluates the last `lookback` candles of the dataframe to check if price has touched 
    the `entry_level` and printed a valid rejection (pinbar, engulfing, or double touch).
    
    Args:
        df (pd.DataFrame): DataFrame containing columns ['Open', 'High', 'Low', 'Close'].
        entry_level (float): The price level to check.
        direction (int): 1 for Bullish (Buy) rejection, -1 for Bearish (Sell) rejection.
        lookback (int): Number of recent candles to evaluate.
        symbol (str): Symbol name.
        
    Returns:
        bool: True if rejection is confirmed, False otherwise.
    """
    if df.empty:
        return False
        
    # Ensure lookback doesn't exceed dataframe size
    lookback = min(lookback, len(df))
    last_candles = df.tail(lookback).copy()
    
    # Reset index to allow relative positioning
    last_candles = last_candles.reset_index(drop=True)
    n = len(last_candles)
    
    # 1. Check Pinbar Rejection (wick rejection)
    # Any single candle touching the level with a wick >= 50% of the range
    for i in range(n):
        open_val = float(last_candles.at[i, 'Open'])
        high_val = float(last_candles.at[i, 'High'])
        low_val = float(last_candles.at[i, 'Low'])
        close_val = float(last_candles.at[i, 'Close'])
        
        total_range = high_val - low_val
        if total_range <= 0:
            continue
            
        if direction == 1:
            body_max = max(open_val, close_val)
            if low_val <= entry_level <= body_max:
                lower_shadow = min(open_val, close_val) - low_val
                if lower_shadow / total_range >= 0.5:
                    return True
        elif direction == -1:
            body_min = min(open_val, close_val)
            if body_min <= entry_level <= high_val:
                upper_shadow = high_val - max(open_val, close_val)
                if upper_shadow / total_range >= 0.5:
                    return True

    # 2. Check Engulfing Confirmation
    # Find candles that touch/sweep the level, and see if the next candle is an engulfing candle
    # in the trade direction.
    for i in range(n - 1):
        open_val = float(last_candles.at[i, 'Open'])
        high_val = float(last_candles.at[i, 'High'])
        low_val = float(last_candles.at[i, 'Low'])
        close_val = float(last_candles.at[i, 'Close'])
        
        # Check if candle i touches the level
        touched = False
        if direction == 1:
            body_max = max(open_val, close_val)
            if low_val <= entry_level <= body_max:
                touched = True
        else:
            body_min = min(open_val, close_val)
            if body_min <= entry_level <= high_val:
                touched = True
                
        if touched:
            # Look at next candle (i+1) for engulfing confirmation
            open_next = float(last_candles.at[i+1, 'Open'])
            high_next = float(last_candles.at[i+1, 'High'])
            low_next = float(last_candles.at[i+1, 'Low'])
            close_next = float(last_candles.at[i+1, 'Close'])
            
            if direction == 1:
                # Bullish Engulfing:
                # - Candle i is bearish: close_val <= open_val
                # - Candle i+1 is bullish: close_next > open_next
                # - Body of i+1 engulfs body of i: close_next >= open_val and open_next <= close_val
                #   Or close_next closes above high_val of candle i
                if close_val <= open_val and close_next > open_next:
                    if (close_next >= open_val and open_next <= close_val) or (close_next > high_val):
                        return True
            else:
                # Bearish Engulfing:
                # - Candle i is bullish: close_val >= open_val
                # - Candle i+1 is bearish: close_next < open_next
                # - Body of i+1 engulfs body of i: close_next <= open_val and open_next >= close_val
                #   Or close_next closes below low_val of candle i
                if close_val >= open_val and close_next < open_next:
                    if (close_next <= open_val and open_next >= close_val) or (close_next < low_val):
                        return True

    # 3. Check Double Touch (Double Bottom/Top)
    # At least two candles touch the level (at least 2 bars apart) without a body close breaking the level
    touches = []
    for i in range(n):
        open_val = float(last_candles.at[i, 'Open'])
        high_val = float(last_candles.at[i, 'High'])
        low_val = float(last_candles.at[i, 'Low'])
        close_val = float(last_candles.at[i, 'Close'])
        
        if direction == 1:
            body_max = max(open_val, close_val)
            if low_val <= entry_level <= body_max:
                touches.append(i)
        else:
            body_min = min(open_val, close_val)
            if body_min <= entry_level <= high_val:
                touches.append(i)
                
    if len(touches) >= 2:
        for t_idx in range(len(touches) - 1):
            i1 = touches[t_idx]
            i2 = touches[t_idx + 1]
            if i2 - i1 >= 2:
                # Verify that no candle body in between broke the level
                broken = False
                for mid in range(i1 + 1, i2):
                    op = float(last_candles.at[mid, 'Open'])
                    cl = float(last_candles.at[mid, 'Close'])
                    if direction == 1:
                        # Bullish: candle body closes below entry_level
                        if min(op, cl) < entry_level:
                            broken = True
                            break
                    else:
                        # Bearish: candle body closes above entry_level
                        if max(op, cl) > entry_level:
                            broken = True
                            break
                if not broken:
                    return True

    return False
