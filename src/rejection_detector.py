import pandas as pd

def detect_rejection_at_level(df: pd.DataFrame, entry_level: float, direction: int, lookback: int = 5) -> bool:
    """
    Evaluates the last `lookback` candles of the dataframe to check if price has touched 
    the `entry_level` and printed a wick rejection.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns ['Open', 'High', 'Low', 'Close'].
        entry_level (float): The price level to check.
        direction (int): 1 for Bullish (Buy) rejection, -1 for Bearish (Sell) rejection.
        lookback (int): Number of recent candles to evaluate.
        
    Returns:
        bool: True if at least one candle in the lookback period shows rejection at the level, False otherwise.
    """
    if df.empty:
        return False
        
    last_candles = df.tail(lookback)
    
    for idx, row in last_candles.iterrows():
        open_val = float(row['Open'])
        high_val = float(row['High'])
        low_val = float(row['Low'])
        close_val = float(row['Close'])
        
        total_range = high_val - low_val
        if total_range <= 0:
            continue
            
        if direction == 1:
            # Bullish rejection check:
            # - The candle must touch the level: Low <= entry_level <= max(Open, Close)
            body_max = max(open_val, close_val)
            if low_val <= entry_level <= body_max:
                lower_shadow = min(open_val, close_val) - low_val
                if lower_shadow / total_range >= 0.5:
                    return True
        elif direction == -1:
            # Bearish rejection check:
            # - The candle must touch the level: min(Open, Close) <= entry_level <= High
            body_min = min(open_val, close_val)
            if body_min <= entry_level <= high_val:
                upper_shadow = high_val - max(open_val, close_val)
                if upper_shadow / total_range >= 0.5:
                    return True
                    
    return False
