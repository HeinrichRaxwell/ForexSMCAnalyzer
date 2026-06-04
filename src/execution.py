import os
import sys
import MetaTrader5 as mt5

def get_active_broker_symbol(symbol: str) -> str:
    """
    Find the matching symbol name supported by the active broker terminal.
    """
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    for sym in symbols_to_try:
        info = mt5.symbol_info(sym)
        if info is not None:
            return sym
    return symbol # Fallback

def get_active_trade_count(symbol: str, magic: int) -> int:
    """
    Get the total count of active positions and pending orders for this magic number.
    """
    positions = mt5.positions_get(symbol=symbol, magic=magic)
    pos_count = len(positions) if positions is not None else 0
    
    orders = mt5.orders_get(symbol=symbol, magic=magic)
    ord_count = len(orders) if orders is not None else 0
    
    return pos_count + ord_count

def execute_trade_for_setup(setup: dict, base_symbol: str = "XAUUSD") -> tuple:
    """
    Sends a pending limit order to MT5 for the given setup.
    
    Args:
        setup (dict): The setup dictionary with keys: timeframe, direction, entry_price, sl_price, tp_price, option_name.
        base_symbol (str): The default symbol (e.g., 'XAUUSD').
        
    Returns:
        (int, str): (ticket_id, success_message) or (None, error_message)
    """
    # Check if execution is enabled in .env
    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").lower() == "true"
    if not execute_enabled:
        return None, "Auto-execution disabled (MT5_EXECUTE_TRADES=False in .env)"
        
    # Check allowed timeframes
    allowed_tfs_str = os.getenv("MT5_ALLOWED_TIMEFRAMES", "M5,M15,M30,H1,H4,D1")
    allowed_tfs = [tf.strip() for tf in allowed_tfs_str.split(",")]
    tf = setup.get("timeframe", "M15")
    if tf not in allowed_tfs:
        return None, f"Timeframe {tf} disabled in .env"
        
    # Get active broker symbol
    symbol = get_active_broker_symbol(base_symbol)
    magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
    
    # Check concurrent trades limit
    max_concurrent = int(os.getenv("MT5_MAX_CONCURRENT_TRADES", "3"))
    current_trades = get_active_trade_count(symbol, magic)
    if current_trades >= max_concurrent:
        return None, f"Max limit reached ({current_trades}/{max_concurrent})"
        
    # Check duplicate entry price (within 0.10 USD/points for Gold)
    entry_price_val = float(setup["entry_price"])
    orders = mt5.orders_get(symbol=symbol, magic=magic)
    if orders is not None:
        for o in orders:
            if abs(o.price_open - entry_price_val) < 0.10:
                return None, f"Duplicate pending order already exists at price {o.price_open:.3f}"
                
    positions = mt5.positions_get(symbol=symbol, magic=magic)
    if positions is not None:
        for p in positions:
            if abs(p.price_open - entry_price_val) < 0.10:
                return None, f"Duplicate position already exists at price {p.price_open:.3f}"
        
    # 1. Determine lot size
    lot = 0.01
    opt_name = setup.get("option_name", "")
    if "Midpoint" in opt_name or "0.5" in opt_name:
        lot_str = os.getenv("MT5_LOT_SIZE_OPTION_A", "0.01")
        try:
            lot = float(lot_str)
        except ValueError:
            lot = 0.01
    elif "GoldenPocket" in opt_name or "0.618" in opt_name:
        lot_str = os.getenv("MT5_LOT_SIZE_OPTION_B", "0.01")
        try:
            lot = float(lot_str)
        except ValueError:
            lot = 0.01
            
    # 2. Determine order type
    # 1 for Bullish (Buy Limit), -1 for Bearish (Sell Limit)
    direction = setup.get("direction", 1)
    if direction == 1:
        order_type = mt5.ORDER_TYPE_BUY_LIMIT
    else:
        order_type = mt5.ORDER_TYPE_SELL_LIMIT
        
    # Ensure symbol is selected in Market Watch
    mt5.symbol_select(symbol, True)
    
    # Get symbol info for digit formatting
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None, f"Failed to get symbol info for {symbol}"
        
    digits = symbol_info.digits
    
    entry_price = round(float(setup["entry_price"]), digits)
    sl_price = round(float(setup["sl_price"]), digits)
    tp_price = round(float(setup["tp_price"]), digits)
    
    comment = f"SMC {setup.get('timeframe', 'M15')} {opt_name[:15]}"
    
    # Try different filling types (RETURN, IOC, FOK)
    filling_types = [
        mt5.ORDER_FILLING_RETURN,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK
    ]
    
    last_error = ""
    for fill in filling_types:
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill,
        }
        
        # Send order
        res = mt5.order_send(request)
        if res is None:
            last_error = "mt5.order_send returned None (check MT5 connection)"
            continue
            
        if res.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
            print(f"[Execution Engine] Order successfully placed: Ticket {res.order} using filling {fill}")
            return res.order, f"PENDING ORDER PLACED: Ticket #{res.order} ({lot} lot @ {entry_price:.3f})"
        else:
            last_error = f"retcode={res.retcode}, comment='{res.comment}'"
            print(f"[Execution Engine] Try filling={fill} failed: {last_error}")
            
    return None, f"Failed to place order: {last_error}"
