import os
import sys
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

sys.path.append("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_snr_and_swapzones, detect_bpr, get_pip_multiplier
from src.execution import get_active_broker_symbol, load_sent_signals

def main():
    if not connect_mt5():
        print("Failed to initialize MT5")
        return
        
    symbol = "XAUUSD"
    magic = 202606
    
    broker_symbol = get_active_broker_symbol(symbol)
    positions = mt5.positions_get(symbol=broker_symbol, magic=magic)
    if positions is None or len(positions) == 0:
        print("No active positions found on MT5.")
        mt5.shutdown()
        return
        
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is None:
        print("Failed to get tick.")
        mt5.shutdown()
        return
        
    print("Fetching H1 & M30 & H4 to evaluate trailing swing points...")
    timeframes_data = {}
    timeframes_data['M30'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 400)
    timeframes_data['H1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
    timeframes_data['H4'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    
    for tf in timeframes_data:
        df = timeframes_data[tf]
        df = detect_swing_points(df, window=5)
        timeframes_data[tf] = df
        
    pip_multiplier = get_pip_multiplier(symbol)
    spread_buffer = 2.0 * pip_multiplier
    
    print(f"\n=== Diagnostics for {len(positions)} Active Position(s) ===")
    for p in positions:
        ticket = p.ticket
        entry_price = p.price_open
        current_sl = p.sl
        current_tp = p.tp
        direction = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
        
        # Original SL Lookup
        original_sl = current_sl
        sent_signals = load_sent_signals()
        found_in_sent = False
        for sig_key, sig_data in sent_signals.items():
            if sig_data.get('ticket_a') == ticket or sig_data.get('ticket_b') == ticket:
                if sig_data.get('ticket_a') == ticket:
                    original_sl = sig_data['features_0.5']['sl_price']
                else:
                    original_sl = sig_data['features_0.618']['sl_price']
                found_in_sent = True
                break
                
        if original_sl == 0:
            original_sl = current_sl
            
        print(f"\nPosition Ticket #{ticket}:")
        print(f"  Comment: '{p.comment}'")
        print(f"  Direction: {'BUY' if direction == 1 else 'SELL'}")
        print(f"  Entry Price: {entry_price:.3f}")
        print(f"  Current SL: {current_sl:.3f}")
        print(f"  Current TP: {current_tp:.3f}")
        print(f"  Original SL: {original_sl:.3f} (Found in registry: {found_in_sent})")
        print(f"  Current Market Price: Bid={tick.bid:.3f}, Ask={tick.ask:.3f}")
        
        # BEP Check
        is_already_be = False
        if direction == 1:
            if current_sl >= entry_price:
                is_already_be = True
        else:
            if current_sl > 0 and current_sl <= entry_price:
                is_already_be = True
                
        print(f"  Already at BEP: {is_already_be}")
        if not is_already_be and original_sl != 0:
            initial_risk = abs(entry_price - original_sl)
            target_price = entry_price + initial_risk if direction == 1 else entry_price - initial_risk
            price_for_check = tick.bid if direction == 1 else tick.ask
            distance_to_target = target_price - price_for_check if direction == 1 else price_for_check - target_price
            
            print(f"  Initial Risk: {initial_risk:.3f} USD")
            print(f"  BEP Trigger Price (1:1 R:R): {target_price:.3f}")
            print(f"  Distance to BEP target: {distance_to_target:.3f} USD")
            if (direction == 1 and tick.bid >= target_price) or (direction == -1 and tick.ask <= target_price):
                print(f"  -> BEP Target MET! Moving SL to BEP: {entry_price + spread_buffer:.3f}")
            else:
                print(f"  -> BEP Target NOT met yet.")
                
        # Trailing Stop Check
        comment = p.comment
        tf = "M30"
        if "H4" in comment:
            tf = "H4"
        elif "H1" in comment:
            tf = "H1"
        elif "D1" in comment:
            tf = "D1"
        elif "M15" in comment:
            tf = "M15"
            
        print(f"  Evaluating structural trailing on {tf}...")
        df_tf = timeframes_data.get(tf)
        if df_tf is not None and not df_tf.empty:
            buffer = 2.0 * pip_multiplier
            if direction == 1:
                swing_lows = df_tf['Swing_Low'].dropna()
                if not swing_lows.empty:
                    recent_swing_low = swing_lows.iloc[-1]
                    new_sl = recent_swing_low - buffer
                    print(f"    Most recent confirmed Swing Low on {tf}: {recent_swing_low:.3f}")
                    print(f"    Proposed trailing SL (Swing Low - buffer): {new_sl:.3f}")
                    print(f"    Check new_sl ({new_sl:.3f}) > current_sl ({current_sl:.3f}): {new_sl > current_sl}")
                    print(f"    Check new_sl ({new_sl:.3f}) < current_bid ({tick.bid:.3f}): {new_sl < tick.bid}")
                    if new_sl > current_sl and new_sl < tick.bid:
                        print("    -> Trailing Stop condition met! SL can be trailed.")
                    else:
                        print("    -> Trailing Stop condition NOT met.")
                else:
                    print(f"    No confirmed Swing Lows found on {tf}.")
            else:
                swing_highs = df_tf['Swing_High'].dropna()
                if not swing_highs.empty:
                    recent_swing_high = swing_highs.iloc[-1]
                    new_sl = recent_swing_high + buffer
                    print(f"    Most recent confirmed Swing High on {tf}: {recent_swing_high:.3f}")
                    print(f"    Proposed trailing SL (Swing High + buffer): {new_sl:.3f}")
                    print(f"    Check new_sl ({new_sl:.3f}) < current_sl ({current_sl:.3f}): {current_sl == 0 or new_sl < current_sl}")
                    print(f"    Check new_sl ({new_sl:.3f}) > current_ask ({tick.ask:.3f}): {new_sl > tick.ask}")
                    if (current_sl == 0 or new_sl < current_sl) and new_sl > tick.ask:
                        print("    -> Trailing Stop condition met! SL can be trailed.")
                    else:
                        print("    -> Trailing Stop condition NOT met.")
                else:
                    print(f"    No confirmed Swing Highs found on {tf}.")
        else:
            print(f"    No data found for timeframe {tf}.")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
