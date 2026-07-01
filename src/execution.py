import os
import sys
import MetaTrader5 as mt5
import pandas as pd
import numpy as np


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

def validate_market_indicators(symbol: str, tf_str: str, direction: int) -> tuple:
    """
    Validate current market indicators (RSI, Stoch RSI, EMA, Volume) before placing a trade.
    Returns:
        (bool, str): (is_valid, reason)
    """
    from src.data_loader import fetch_historical_data
    
    # 1. Map timeframe string to MT5 timeframe
    mapping = {
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    tf = mapping.get(tf_str, mt5.TIMEFRAME_M15)
    
    # 2. Fetch history (recent 100 candles)
    df = fetch_historical_data(symbol, tf, 100)
    if df is None or df.empty or len(df) < 30:
        return True, "Insufficient historical data to validate indicators"
        
    close = df['Close']
    
    # 3. Calculate RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 0.00001)
    rsi = 100 - (100 / (1 + rs))
    curr_rsi = rsi.iloc[-1]
    
    # 4. Calculate Stochastic RSI (14)
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, 0.00001)
    curr_stoch_rsi = stoch_rsi.iloc[-1]
    
    # 5. Volume check
    curr_vol = df['Volume'].iloc[-1]
    avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
    
    # Validation checks
    # A. Volume spike check (market reaction too violent)
    if curr_vol > 3.5 * avg_vol:
        return False, f"Abnormal volume spike detected ({curr_vol:.0f} > 3.5x avg {avg_vol:.0f})"
        
    # B. RSI & Stoch RSI Extreme checks (Disabled for pending limit orders)
    # if direction == 1:  # Buy
    #     if curr_rsi < 20.0:
    #         return False, f"RSI is extremely oversold ({curr_rsi:.1f} < 20), price dumping too fast"
    #     if curr_rsi > 70.0:
    #         return False, f"RSI is overbought ({curr_rsi:.1f} > 70)"
    #     if curr_stoch_rsi > 0.85:
    #         return False, f"Stoch RSI is overbought ({curr_stoch_rsi:.2f} > 0.85)"
    # else:  # Sell
    #     if curr_rsi > 80.0:
    #         return False, f"RSI is extremely overbought ({curr_rsi:.1f} > 80), price pumping too fast"
    #     if curr_rsi < 30.0:
    #         return False, f"RSI is oversold ({curr_rsi:.1f} < 30)"
    #     if curr_stoch_rsi < 0.15:
    #         return False, f"Stoch RSI is oversold ({curr_stoch_rsi:.2f} < 0.15)"
    pass
            
    return True, f"Indicators valid: RSI={curr_rsi:.1f}, StochRSI={curr_stoch_rsi:.2f}, Volume={curr_vol:.0f}"

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
    allowed_tfs_str = os.getenv("MT5_ALLOWED_TIMEFRAMES", "M15,M30,H1,H4,D1")
    allowed_tfs = [tf.strip() for tf in allowed_tfs_str.split(",")]
    tf = setup.get("timeframe", "M15")
    if tf not in allowed_tfs:
        return None, f"Timeframe {tf} disabled in .env"
        
    # Get active broker symbol
    symbol = get_active_broker_symbol(base_symbol)
    magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
    
    # Concurrent trades limit removed per user request
        
    # Check duplicate entry price (within 0.15 USD/points for Gold)
    entry_price_val = float(setup["entry_price"])
    orders = mt5.orders_get(symbol=symbol, magic=magic)
    if orders is not None:
        for o in orders:
            if abs(o.price_open - entry_price_val) < 0.15:
                return None, f"Duplicate pending order already exists at price {o.price_open:.3f}"
                
    positions = mt5.positions_get(symbol=symbol, magic=magic)
    if positions is not None:
        for p in positions:
            if abs(p.price_open - entry_price_val) < 0.15:
                return None, f"Duplicate position already exists at price {p.price_open:.3f}"
        
    # 1. Determine lot size
    lot = 0.01
    opt_name = setup.get("option_name", "")
    if "Midpoint" in opt_name or "0.5" in opt_name or "Option A" in opt_name:
        lot_str = os.getenv("MT5_LOT_SIZE_OPTION_A", "0.01")
        try:
            lot = float(lot_str)
        except ValueError:
            lot = 0.01
    elif "GoldenPocket" in opt_name or "0.618" in opt_name or "Option B" in opt_name:
        lot_str = os.getenv("MT5_LOT_SIZE_OPTION_B", "0.02")
        try:
            lot = float(lot_str)
        except ValueError:
            lot = 0.02
            
    # 2. Determine order type
    # 1 for Bullish (Buy Limit), -1 for Bearish (Sell Limit)
    direction = setup.get("direction", 1)
    if direction == 1:
        order_type = mt5.ORDER_TYPE_BUY_LIMIT
    else:
        order_type = mt5.ORDER_TYPE_SELL_LIMIT
        
    # Ensure symbol is selected in Market Watch
    mt5.symbol_select(symbol, True)
    
    # Check entry price distance to market price (prevent spamming far orders)
    tick = mt5.symbol_info_tick(symbol)
    if tick is not None:
        current_price = tick.ask if direction == 1 else tick.bid
        price_diff = abs(entry_price_val - current_price)
        max_dist = float(os.getenv("MT5_MAX_PENDING_DISTANCE_USD", "20.0"))
        if tf == 'H4':
            max_dist = max(max_dist, 100.0)  # Allow up to 100.0 USD for H4 setups
        elif tf == 'H1':
            max_dist = max(max_dist, 60.0)  # Allow up to 60.0 USD for H1 setups
        elif tf == 'M30':
            max_dist = max(max_dist, 30.0)  # Allow up to 30.0 USD for M30 setups
        elif tf == 'D1':
            max_dist = max(max_dist, 200.0)  # Allow up to 200.0 USD for D1 setups
            
        if price_diff > max_dist:
            return None, f"price is too far from market ({price_diff:.2f} USD > {max_dist} USD limit)"
            
        # Price is close! Now check current market indicators before placing order
        is_valid_mkt, mkt_reason = validate_market_indicators(symbol, tf, direction)
        if not is_valid_mkt:
            return None, f"Market indicators check failed: {mkt_reason}"
            
    # Get symbol info for digit formatting
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None, f"Failed to get symbol info for {symbol}"
        
    digits = symbol_info.digits
    
    entry_price = round(float(setup["entry_price"]), digits)
    sl_price = round(float(setup["sl_price"]), digits)
    
    # Set the safety hard TP on the MT5 server to the furthest target (TP 3 / 1:4 RR).
    # The bot's background worker will manage soft TP close at TP 1 or TP 2 dynamically.
    safety_tp_price = setup.get("tp3_price", setup.get("tp2_price", setup["tp_price"]))
    tp_price = round(float(safety_tp_price), digits)
    
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


def execute_market_order_for_setup(setup: dict, base_symbol: str = "XAUUSD") -> tuple:
    """
    Sends a market buy/sell order (Instant Execution) to MT5 for the given setup.
    
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
    allowed_tfs_str = os.getenv("MT5_ALLOWED_TIMEFRAMES", "M15,M30,H1,H4,D1")
    allowed_tfs = [tf.strip() for tf in allowed_tfs_str.split(",")]
    tf = setup.get("timeframe", "M15")
    if tf not in allowed_tfs:
        return None, f"Timeframe {tf} disabled in .env"
        
    # Get active broker symbol
    symbol = get_active_broker_symbol(base_symbol)
    magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
    
    # 1. Determine lot size
    lot = 0.01
    opt_name = setup.get("option_name", "")
    if "Midpoint" in opt_name or "0.5" in opt_name or "Option A" in opt_name:
        lot_str = os.getenv("MT5_LOT_SIZE_OPTION_A", "0.01")
        try:
            lot = float(lot_str)
        except ValueError:
            lot = 0.01
    elif "GoldenPocket" in opt_name or "0.618" in opt_name or "Option B" in opt_name:
        lot_str = os.getenv("MT5_LOT_SIZE_OPTION_B", "0.02")
        try:
            lot = float(lot_str)
        except ValueError:
            lot = 0.02
            
    # 2. Determine order type
    # 1 for Bullish (Buy), -1 for Bearish (Sell)
    direction = setup.get("direction", 1)
    
    # Ensure symbol is selected in Market Watch
    mt5.symbol_select(symbol, True)
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, "Failed to get current price tick from MT5"
        
    if direction == 1:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        
    # Get symbol info for digit formatting
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None, f"Failed to get symbol info for {symbol}"
        
    digits = symbol_info.digits
    
    price_formatted = round(float(price), digits)
    sl_price = round(float(setup["sl_price"]), digits)
    safety_tp_price = setup.get("tp3_price", setup.get("tp2_price", setup["tp_price"]))
    tp_price = round(float(safety_tp_price), digits)
    
    comment = f"SMC Mkt {setup.get('timeframe', 'M15')} {opt_name[:12]}"
    
    # Try different filling types (RETURN, IOC, FOK)
    filling_types = [
        mt5.ORDER_FILLING_RETURN,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK
    ]
    
    last_error = ""
    for fill in filling_types:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price_formatted,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_filling": fill,
        }
        
        # Send order
        res = mt5.order_send(request)
        if res is None:
            last_error = "mt5.order_send returned None"
            continue
            
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[Execution Engine] Market Order successfully placed: Position Ticket {res.deal} (Order Ticket {res.order}) using filling {fill}")
            ticket = res.order if res.order else res.deal
            return ticket, f"MARKET ORDER PLACED: Ticket #{ticket} ({lot} lot @ {price_formatted:.3f})"
        else:
            last_error = f"retcode={res.retcode}, comment='{res.comment}'"
            print(f"[Execution Engine] Try market filling={fill} failed: {last_error}")
            
    return None, f"Failed to place market order: {last_error}"


def modify_position_sltp(ticket: int, symbol: str, sl: float, tp: float) -> bool:
    """Modify the Stop Loss and Take Profit of an active position."""
    import MetaTrader5 as mt5
    # Get digits for formatting
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[Execution Engine] Failed to get symbol info for {symbol} to modify SL/TP")
        return False
    digits = symbol_info.digits
    sl = round(sl, digits)
    tp = round(tp, digits)
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": float(sl),
        "tp": float(tp)
    }
    
    res = mt5.order_send(request)
    if res is None:
        print(f"[Execution Engine] Modify SL/TP for position #{ticket} returned None")
        return False
        
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[Execution Engine] Modify SL/TP for position #{ticket} successful (SL: {sl:.3f}, TP: {tp:.3f})")
        return True
    else:
        print(f"[Execution Engine] Modify SL/TP for position #{ticket} failed: retcode={res.retcode}, comment='{res.comment}'")
        return False


def close_position(ticket: int, symbol: str, volume: float = None) -> bool:
    """Close an active position or partial close if volume is specified."""
    import MetaTrader5 as mt5
    positions = mt5.positions_get(ticket=ticket)
    if positions is None or len(positions) == 0:
        print(f"[Execution Engine] Position #{ticket} not found to close")
        return False
        
    pos = positions[0]
    broker_symbol = pos.symbol
    
    # Determine close volume
    close_vol = float(volume) if volume is not None else float(pos.volume)
    # Ensure close volume does not exceed position volume
    close_vol = min(close_vol, float(pos.volume))
    
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    
    # Get current price
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is None:
        print(f"[Execution Engine] Failed to get price tick to close position #{ticket}")
        return False
    price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
    
    # Try different filling types
    filling_types = [
        mt5.ORDER_FILLING_RETURN,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK
    ]
    
    for fill in filling_types:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "volume": close_vol,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": f"SMC Bot Soft TP ({close_vol:.2f} lot)",
            "type_filling": fill,
        }
        
        res = mt5.order_send(request)
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[Execution Engine] Position #{ticket} successfully closed/partially closed: {close_vol:.2f} lot using filling {fill}")
            return True
            
    print(f"[Execution Engine] Failed to close position #{ticket} using all filling types")
    return False


def load_sent_signals() -> dict:
    """Load the registry of already alerted signals (internal helper to avoid circular imports)."""
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sent_signals.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _last_closed_trend(df: pd.DataFrame):
    """Return the latest closed-candle trend, ignoring the active candle when present."""
    if df is None or df.empty or 'Trend' not in df.columns:
        return None

    trends = pd.to_numeric(df['Trend'], errors='coerce').dropna()
    if trends.empty:
        return None

    closed_trends = trends.iloc[:-1] if len(trends) >= 2 else trends
    if closed_trends.empty:
        return None
    return int(closed_trends.iloc[-1])


def _consecutive_closed_trend_count(df: pd.DataFrame, trend_value: int) -> int:
    """Count how many latest closed candles share the requested trend value."""
    if df is None or df.empty or 'Trend' not in df.columns:
        return 0

    trends = pd.to_numeric(df['Trend'], errors='coerce').dropna()
    if trends.empty:
        return 0

    closed_trends = trends.iloc[:-1] if len(trends) >= 2 else trends
    count = 0
    for value in reversed(closed_trends.tolist()):
        if int(value) != int(trend_value):
            break
        count += 1
    return count


def should_emergency_exit_on_reversal(
    df_tf: pd.DataFrame,
    setup_timeframe: str,
    direction: int,
    h1_trend=None,
    h4_trend=None,
) -> bool:
    """
    Confirm opposite CHoCH before closing a live trade.
    LTF trades need two closed opposite candles or HTF confirmation to avoid noise exits.
    """
    opposite_direction = -int(direction)
    latest_closed_trend = _last_closed_trend(df_tf)
    if latest_closed_trend != opposite_direction:
        return False

    timeframe = str(setup_timeframe).upper()
    if timeframe in {"H1", "H4", "D1"}:
        return True

    consecutive_opposite = _consecutive_closed_trend_count(df_tf, opposite_direction)
    if consecutive_opposite >= 2:
        return True

    return h1_trend == opposite_direction or h4_trend == opposite_direction


def manage_active_trades(symbol: str, magic: int, timeframes_data: dict):
    """
    Scans all active positions for the magic number and applies:
    1. Break Even Point (BEP) when price reaches 1:1 Risk-to-Reward.
    2. Structural Trailing Stop (SMC strategy 2) based on Swing Lows/Highs.
    3. Soft TP execution at TP 1 (Fibo TP) with Partial Close (Option B) if HTF trend is confluent, 
       or full close if HTF trend is not aligned.
    4. Emergency trend reversal exit (opposite CHoCH/Trend reversal) on the setup timeframe.
    """
    import MetaTrader5 as mt5
    from src.smc_detector import get_pip_multiplier
    from src.telegram_bot import send_telegram_alert
    
    broker_symbol = get_active_broker_symbol(symbol)
    positions = mt5.positions_get(symbol=broker_symbol, magic=magic)
    if positions is None or len(positions) == 0:
        return
        
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is None:
        return
        
    pip_multiplier = get_pip_multiplier(symbol)
    spread_buffer = 2.0 * pip_multiplier # 2 pips buffer (0.20 USD for Gold)
    
    # Check Higher Timeframe Trends for confluence
    h1_trend = _last_closed_trend(timeframes_data['H1']) if 'H1' in timeframes_data else None
    h4_trend = _last_closed_trend(timeframes_data['H4']) if 'H4' in timeframes_data else None
    
    for p in positions:
        ticket = p.ticket
        entry_price = p.price_open
        current_sl = p.sl
        current_tp = p.tp
        current_vol = p.volume
        direction = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
        
        # We parse the timeframe and option type from the position comment (if set by bot)
        comment = p.comment
        tf = "M30" # default fallback
        if "H4" in comment:
            tf = "H4"
        elif "H1" in comment:
            tf = "H1"
        elif "D1" in comment:
            tf = "D1"
        elif "M15" in comment:
            tf = "M15"
            
        is_option_b = "Option B" in comment or "0.618" in comment or "GoldenPocket" in comment
        
        # 1. Look up original Stop Loss, TP1, and TP2 from sent signals registry
        original_sl = current_sl
        tp1 = 0.0
        tp2 = 0.0
        sent_signals = load_sent_signals()
        for sig_key, sig_data in sent_signals.items():
            if sig_data.get('ticket_a') == ticket or sig_data.get('ticket_b') == ticket:
                if sig_data.get('ticket_a') == ticket:
                    feat = sig_data.get('features_0.5', {})
                    original_sl = feat.get('sl_price', current_sl)
                    tp1 = feat.get('tp_price', 0.0)
                    tp2 = feat.get('tp2_price', 0.0)
                else:
                    feat = sig_data.get('features_0.618', {})
                    original_sl = feat.get('sl_price', current_sl)
                    tp1 = feat.get('tp_price', 0.0)
                    tp2 = feat.get('tp2_price', 0.0)
                break
                
        # Fallback values if registry lookup fails
        if original_sl == 0:
            original_sl = current_sl
            
        df_tf = timeframes_data.get(tf)
        
        # --- A. Emergency Reversal Exit (opposite CHoCH/Trend reversal) ---
        if df_tf is not None and not df_tf.empty and 'Trend' in df_tf.columns:
            if should_emergency_exit_on_reversal(df_tf, tf, direction, h1_trend, h4_trend):
                # Reversal detected on setup timeframe! Close the trade immediately.
                print(f"[Execution Engine] Confirmed trend reversal (opposite CHoCH) detected on {tf} for position #{ticket}. Closing deal.")
                success = close_position(ticket, broker_symbol)
                if success:
                    try:
                        send_telegram_alert(
                            f"🚨 <b>[SMC Trade Manager] Emergency Exit!</b>\n\n"
                            f"Timeframe: {tf}\n"
                            f"Posisi #{ticket} ({'BUY' if direction == 1 else 'SELL'}) ditutup lebih cepat di harga pasar "
                            f"karena terdeteksi pembalikan trend (Opposite CHoCH terjadi pada candle terakhir)."
                        )
                    except Exception:
                        pass
                    continue
                    
        # --- B. Soft TP Logic at TP 1 (Fibo TP) ---
        if tp1 > 0.0:
            is_tp1_reached = False
            if direction == 1: # Buy
                if tick.bid >= tp1:
                    is_tp1_reached = True
            else: # Sell
                if tick.ask <= tp1:
                    is_tp1_reached = True
                    
            if is_tp1_reached:
                # Check structure quality (H1 or H4 trend confluence)
                is_structure_good = (h1_trend == direction) or (h4_trend == direction)
                
                # If Option A or structure is not confluent (bad), close 100%
                if not is_option_b or not is_structure_good:
                    print(f"[Execution Engine] TP 1 reached for #{ticket}. Closing 100% (Confluence={is_structure_good}, OptB={is_option_b}).")
                    success = close_position(ticket, broker_symbol)
                    if success:
                        try:
                            reason = "Struktur HTF kurang mendukung untuk hold sisa" if not is_structure_good else "Option A selalu ditutup penuh di TP 1"
                            send_telegram_alert(
                                f"🎯 <b>[SMC Trade Manager] TP 1 Hit! (Sapu Bersih)</b>\n\n"
                                f"Timeframe: {tf}\n"
                                f"Posisi #{ticket} ({'BUY' if direction == 1 else 'SELL'}) ditutup penuh di level Fibo TP 1: <code>{tp1:.3f}</code>.\n"
                                f"• Alasan: <i>{reason}</i>."
                            )
                        except Exception:
                            pass
                        continue
                else:
                    # Option B and structure is GOOD! Do partial close if not already done
                    # (we check if current volume is still original 0.02 or larger)
                    if current_vol >= 0.02:
                        print(f"[Execution Engine] TP 1 reached for Option B #{ticket}. Performing 50% partial close.")
                        success_partial = close_position(ticket, broker_symbol, volume=0.01)
                        if success_partial:
                            # Move SL of remaining portion to BEP and TP to TP2 (Dynamic TP)
                            new_sl = entry_price + spread_buffer * direction
                            new_tp = tp2 if tp2 > 0 else (entry_price + (entry_price - original_sl) * 3 if direction == 1 else entry_price - (original_sl - entry_price) * 3)
                            success_modify = modify_position_sltp(ticket, broker_symbol, new_sl, new_tp)
                            
                            try:
                                send_telegram_alert(
                                    f"🎯 <b>[SMC Trade Manager] TP 1 Hit! (Partial Close)</b>\n\n"
                                    f"Timeframe: {tf}\n"
                                    f"Posisi Option B #{ticket} ({'BUY' if direction == 1 else 'SELL'}) mengenai Fibo TP 1: <code>{tp1:.3f}</code>.\n"
                                    f"• <b>Tindakan:</b> Ditutup 50% (0.01 lot) untuk mengunci profit.\n"
                                    f"• <b>Sisa 0.01 Lot:</b> Dibiarkan jalan ke TP 2 (Struktur): <code>{new_tp:.3f}</code> dengan SL digeser ke BEP: <code>{new_sl:.3f}</code>."
                                )
                            except Exception:
                                pass
                            continue
                            
        # --- C. Risk-to-Reward 1:1 Break Even (BEP) check ---
        # (This is a fallback if price did not hit TP 1 yet, but reached 1:1 RR)
        is_already_be = False
        if direction == 1:
            if current_sl >= entry_price:
                is_already_be = True
        else:
            if current_sl > 0 and current_sl <= entry_price:
                is_already_be = True
                
        if not is_already_be and original_sl != 0:
            initial_risk = abs(entry_price - original_sl)
            if direction == 1:
                target_price = entry_price + initial_risk
                if tick.bid >= target_price:
                    new_sl = entry_price + spread_buffer
                    success = modify_position_sltp(ticket, broker_symbol, new_sl, current_tp)
                    if success:
                        current_sl = new_sl
                        is_already_be = True
            else:
                target_price = entry_price - initial_risk
                if tick.ask <= target_price:
                    new_sl = entry_price - spread_buffer
                    success = modify_position_sltp(ticket, broker_symbol, new_sl, current_tp)
                    if success:
                        current_sl = new_sl
                        is_already_be = True
                        
        # --- D. SMC-based Structural Trailing (Strategy 2) ---
        if df_tf is not None and not df_tf.empty:
            buffer = 2.0 * pip_multiplier
            if direction == 1: # Buy
                swing_lows = df_tf['Swing_Low'].dropna()
                if not swing_lows.empty:
                    recent_swing_low = swing_lows.iloc[-1]
                    new_sl = recent_swing_low - buffer
                    # Ensure SL only moves UP and is below current market Bid
                    if new_sl > current_sl and new_sl < tick.bid:
                        modify_position_sltp(ticket, broker_symbol, new_sl, current_tp)
            else: # Sell
                swing_highs = df_tf['Swing_High'].dropna()
                if not swing_highs.empty:
                    recent_swing_high = swing_highs.iloc[-1]
                    new_sl = recent_swing_high + buffer
                    # Ensure SL only moves DOWN and is above current market Ask
                    if (current_sl == 0 or new_sl < current_sl) and new_sl > tick.ask:
                        modify_position_sltp(ticket, broker_symbol, new_sl, current_tp)
