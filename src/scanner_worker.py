import os
import sys
import time
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_snr_and_swapzones, detect_bpr, get_pip_multiplier
from src.labeler import get_killzone
from src.inference import predict_setup_probability, process_mt5_history_feedback
from src.rejection_detector import detect_rejection_at_level
from src.main import find_dynamic_tp, extract_active_htf_fvgs, get_active_setups, plot_smc_chart
from src.telegram_bot import send_telegram_alert
from src.execution import execute_trade_for_setup

# Storage for sent signal signatures
SENT_SIGNALS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sent_signals.json")

def load_sent_signals() -> dict:
    """Load the registry of already alerted signals."""
    if os.path.exists(SENT_SIGNALS_FILE):
        try:
            with open(SENT_SIGNALS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_sent_signals(sent_dict: dict):
    """Save the registry of alerted signals to disk."""
    os.makedirs(os.path.dirname(SENT_SIGNALS_FILE), exist_ok=True)
    with open(SENT_SIGNALS_FILE, "w") as f:
        json.dump(sent_dict, f, indent=4)

def prune_invalid_pending_orders(symbol: str, magic: int, active_high_confidence_setups: list):
    """
    Cancel pending orders on the MT5 account that are no longer active, 
    mitigated, or too far away from the current price.
    """
    import MetaTrader5 as mt5
    from src.execution import get_active_broker_symbol
    broker_symbol = get_active_broker_symbol(symbol)
    orders = mt5.orders_get(symbol=broker_symbol, magic=magic)
    if orders is None or len(orders) == 0:
        return
        
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is None:
        return
    current_price = tick.ask if len(orders) > 0 else tick.bid
    
    # We build a set of (entry_price, order_type) for the current active high-confidence setups
    # order_type: 2 = Buy Limit, 3 = Sell Limit
    active_keys = set()
    for s in active_high_confidence_setups:
        o_type = 2 if s['direction'] == 1 else 3
        active_keys.add((round(s['entry_price'], 3), o_type))
        
    cancelled_tickets = []
    for o in orders:
        o_price = round(o.price_open, 3)
        o_type = o.type
        
        is_still_valid = (o_price, o_type) in active_keys
        price_diff = abs(o.price_open - tick.last) if tick.last > 0 else abs(o.price_open - current_price)
        is_too_far = price_diff > 30.0
        
        if not is_still_valid or is_too_far:
            reason = "structure mitigated/invalid" if not is_still_valid else "too far from market (>30 USD)"
            print(f"[Risk Management] Cancelling zombie/invalid pending order #{o.ticket} ({reason}).")
            
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": o.ticket,
            }
            res = mt5.order_send(request)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[Risk Management] Order #{o.ticket} successfully cancelled.")
                cancelled_tickets.append((o.ticket, reason))
                
    if cancelled_tickets:
        try:
            lines = [f"🧹 <b>[Risk Management] Cleaned up {len(cancelled_tickets)} zombie pending orders:</b>"]
            for ticket, reason in cancelled_tickets:
                lines.append(f"• Order #{ticket} ({reason})")
            send_telegram_alert("\n".join(lines))
        except Exception:
            pass

def run_scan(symbol: str, confidence_threshold: float):
    """Run a single scan cycle across all timeframes and send new signals to Telegram."""
    print(f"\n--- Starting Scan Cycle for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # 1. Try to connect to MT5 Exness terminal
    import MetaTrader5 as mt5
    if not connect_mt5():
        print("[Scanner Error] Failed to connect to MetaTrader 5 terminal. Skipping cycle.")
        return
        
    # 1.5. Run feedback loop to process MT5 history outcomes and retrain model
    try:
        new_feedback_count = process_mt5_history_feedback()
        if new_feedback_count > 0:
            print(f"[Feedback Loop] Learned {new_feedback_count} new outcomes and retrained model!")
            send_telegram_alert(f"🔄 <b>Bot AI telah mempelajari {new_feedback_count} kekalahan/kemenangan baru dari akun Anda dan melakukan retraining otomatis!</b>")
    except Exception as e:
        print(f"[Feedback Loop Error] {e}")
        
    timeframes_data = {}
    mt5_active = True
    
    try:
        # Fetch multi-timeframe data with expanded lookback bars
        print("Fetching multi-timeframe data from MT5...")
        timeframes_data['D1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 100)
        timeframes_data['H4'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
        timeframes_data['H1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
        timeframes_data['M30'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 400)
        timeframes_data['M15'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 500)
        timeframes_data['M5'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M5, 500)
    except Exception as e:
        print(f"[Scanner Error] Error loading data from MT5: {e}")
        import MetaTrader5 as mt5
        mt5.shutdown()
        return
    
    # 2. Run SMC detection algorithms on all timeframes
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        df_tf = detect_swing_points(df_tf, window=5)
        df_tf = detect_structures(df_tf)
        df_tf = detect_fvg_and_ob(df_tf, symbol=symbol)
        df_tf = detect_snr_and_swapzones(df_tf)
        df_tf = detect_bpr(df_tf, symbol=symbol)
        
        # Calculate ATR_14
        close_prev = df_tf['Close'].shift(1).fillna(df_tf['Open'])
        tr = np.maximum(
            df_tf['High'] - df_tf['Low'],
            np.maximum(
                np.abs(df_tf['High'] - close_prev),
                np.abs(df_tf['Low'] - close_prev)
            )
        )
        df_tf['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
        timeframes_data[tf_name] = df_tf
        
    # 3. Extract active HTF FVGs for hierarchy prioritization
    active_fvgs_by_tf = {}
    for tf_name in ['M5', 'M15', 'M30', 'H1', 'H4', 'D1']:
        active_fvgs_by_tf[tf_name] = extract_active_htf_fvgs(timeframes_data[tf_name])
        
    # Extract setups
    all_setups = []
    for tf_name in ['D1', 'H4', 'H1', 'M30', 'M15', 'M5']:
        tf_setups = get_active_setups(timeframes_data[tf_name])
        for s in tf_setups:
            s['timeframe'] = tf_name
            all_setups.append(s)
            
    # 4. Multi-Timeframe Alignment, Suppression, and Rejection Checks
    tf_weights = {'M5': 0.5, 'M15': 1, 'M30': 2, 'H1': 3, 'H4': 4, 'D1': 5}
    tf_minutes_map = {'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    
    for setup in all_setups:
        setup['htf_prioritized'] = False
        setup['matching_htf_fvgs'] = []
        setup['suppressed'] = False
        setup['htf_conflict_reason'] = ""
        setup_tf = setup['timeframe']
        
        for htf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
            if tf_weights[htf_name] > tf_weights[setup_tf]:
                # HTF Prioritization (same direction)
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_same = (setup['direction'] == 1 and htf_fvg['type'] == 'BULLISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BEARISH')
                    if is_same:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['htf_prioritized'] = True
                            fvg_info = htf_fvg.copy()
                            fvg_info['timeframe'] = htf_name
                            setup['matching_htf_fvgs'].append(fvg_info)
                            
                # Conflict Suppression (opposite direction)
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_opp = (setup['direction'] == 1 and htf_fvg['type'] == 'BEARISH') or \
                             (setup['direction'] == -1 and htf_fvg['type'] == 'BULLISH')
                    if is_opp:
                        setup['suppressed'] = True
                        setup['htf_conflict_reason'] = f"Opposite active {htf_name} FVG"
                        break
                        
        # Check Rejection on M15 for higher timeframe setups
        if setup_tf not in ['M15', 'M5']:
            m15_df = timeframes_data.get('M15')
            if m15_df is not None and not m15_df.empty:
                rej_confirmed = detect_rejection_at_level(m15_df, setup['entry_price'], setup['direction'], lookback=15)
                setup['rejection_confirmed'] = rej_confirmed

    # 5. Model Inference & Notification Dispatch
    sent_signals = load_sent_signals()
    signals_sent_this_cycle = 0
    active_high_confidence = []
    
    for setup in all_setups:
        is_scalp = False
        if setup['suppressed']:
            is_scalp = True
            # Convert suppressed setup to a counter-trend scalp setup (30-50 pips TP)
            pip_unit = get_pip_multiplier(symbol)
            if setup['direction'] == 1:
                setup['tp_price'] = setup['entry_price'] + 30 * pip_unit
                setup['tp2_price'] = setup['entry_price'] + 50 * pip_unit
            else:
                setup['tp_price'] = setup['entry_price'] - 30 * pip_unit
                setup['tp2_price'] = setup['entry_price'] - 50 * pip_unit
            setup['tp3_price'] = setup['tp2_price']
            setup['option_name'] += " [Scalp 30-50p]"
            
        features = {
            'timeframe': tf_minutes_map[setup['timeframe']],
            'hour': setup['hour'],
            'day_of_week': setup['day_of_week'],
            'setup_type': setup['setup_type'],
            'direction': setup['direction'],
            'entry_price': setup['entry_price'],
            'sl_price': setup['sl_price'],
            'tp_price': setup['tp_price'],
            'risk_pips': setup['risk_pips'],
            'atr_14': setup['atr_14'],
            'trend': setup['trend'],
            'relative_risk': setup['relative_risk'],
            'killzone': setup['killzone'],
            'fvg_width': setup['fvg_width'],
            'relative_fvg_width': setup['relative_fvg_width']
        }
        
        # Check setup age in bars to avoid executing/alerting stale historical setups (SMC logic)
        tf_df = timeframes_data[setup['timeframe']]
        setup_age_bars = len(tf_df) - 1 - setup['index']
        tf_name = setup['timeframe']
        
        if tf_name == 'M5':
            max_age_bars = 12   # 1 hour lookback
        elif tf_name == 'M15':
            max_age_bars = 16   # 4 hours lookback
        elif tf_name == 'M30':
            max_age_bars = 24   # 12 hours lookback
        elif tf_name == 'H1':
            max_age_bars = 120  # 5 days lookback
        elif tf_name == 'H4':
            max_age_bars = 200  # ~1 month lookback
        elif tf_name == 'D1':
            max_age_bars = 100  # ~3 months lookback
        else:
            max_age_bars = 20
            
        if setup_age_bars > max_age_bars:
            continue
            
        try:
            prob = predict_setup_probability(features)
        except Exception as e:
            print(f"Error predicting setup probability: {e}")
            continue
            
        if prob >= confidence_threshold:
            setup['probability'] = prob
            setup['status'] = "HIGH CONFIDENCE SIGNAL"
            active_high_confidence.append(setup)
            
            # Generate unique signature for this signal to avoid duplicates
            setup_time_str = str(setup['time'])
            setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
            dir_name = "BULL" if setup['direction'] == 1 else "BEAR"
            
            # Key format: TF_SetupName_Direction_Price_Time
            sig_key = f"{setup['timeframe']}_{setup_name}_{dir_name}_{setup['entry_price']:.3f}_{setup_time_str.replace(' ', '_')}"
            
            if sig_key in sent_signals:
                continue # Already sent
                
            print(f"[New Signal Detected] {setup['timeframe']} {dir_name} {setup_name} | Win Prob: {prob:.2%}")
            
            # 6. Generate setup chart
            tf_df = timeframes_data[setup['timeframe']]
            tf_setups = [setup]
            title = f"{symbol} {setup['timeframe']} - {dir_name} {setup_name} Signal"
            image_filename = f"temp_alert_{setup['timeframe']}.png"
            
            try:
                plot_smc_chart(tf_df, title=title, active_setups=tf_setups, output_filename=image_filename)
            except Exception as e:
                print(f"Failed to generate chart image: {e}")
                image_filename = None
                
            # 7. Format Telegram alert message using HTML tags
            direction_emoji = "🟢 BUY (LONG)" if setup['direction'] == 1 else "🔴 SELL (SHORT)"
            setup_type_desc = "Order Block" if setup['setup_type'] == 1 else "Fair Value Gap"
            if "BPR" in setup.get('option_name', ''):
                setup_type_desc = "Balanced Price Range"
            elif "Breaker" in setup.get('option_name', ''):
                setup_type_desc = "Breaker Block"
            elif "Swapzone" in setup.get('option_name', ''):
                setup_type_desc = "Swapzone (S/R)"
                
            rej_status = "✅ Confirmed (Wick)" if setup.get('rejection_confirmed', False) else "❌ No Wick Conf"
            htf_prior_status = "✅ YES" if setup['htf_prioritized'] else "❌ NO"
            
            # 6.5. Auto-execute trade on MT5 if enabled in .env
            ticket_id, exec_msg = execute_trade_for_setup(setup, symbol)
            if ticket_id:
                exec_status = f"✅ PENDING ORDER PLACED (Ticket #{ticket_id})"
            elif "disabled" in exec_msg:
                exec_status = "⚠️ Monitoring Only (Disabled in .env)"
            else:
                exec_status = f"❌ FAILED ({exec_msg})"
            
            # Format HTF matching string
            htf_align_str = ""
            if setup['matching_htf_fvgs']:
                matching_desc = ", ".join([f"{f['timeframe']} FVG ({f['bottom']:.3f}-{f['top']:.3f})" for f in setup['matching_htf_fvgs']])
                htf_align_str = f"<b>Matched HTF Structure:</b>\n• {matching_desc}\n"
            
            if is_scalp:
                header_title = f"🚨 <b>COUNTER-TREND SCALP SIGNAL ({symbol})</b> 🚨"
                tp_header = "🎯 <b>TAKE PROFIT TARGETS (Scalp Mode):</b>"
            else:
                header_title = f"🚨 <b>HIGH CONFIDENCE SMC SIGNAL ({symbol})</b> 🚨"
                tp_header = "🎯 <b>TAKE PROFIT TARGETS:</b>"
                
            msg = (
                f"{header_title}\n\n"
                f"<b>Timeframe:</b> {setup['timeframe']}\n"
                f"<b>Direction:</b> {direction_emoji}\n"
                f"<b>Setup:</b> {setup_type_desc} ({setup.get('option_name', 'Golden Zone')})\n\n"
                f"🎯 <b>AI Success Score:</b> <code>{prob:.2%}</code>\n"
                f"📱 <b>HTF Priority:</b> {htf_prior_status}\n"
                f"🕯️ <b>LTF Rejection:</b> {rej_status}\n"
                f"🤖 <b>Auto-Trading:</b> <code>{exec_status}</code>\n\n"
                f"📈 <b>ENTRY LEVELS:</b>\n"
                f"• <b>Entry:</b> <code>{setup['entry_price']:.3f}</code>\n"
                f"• <b>Stop Loss (SL):</b> <code>{setup['sl_price']:.3f}</code>\n\n"
                f"{tp_header}\n"
                f"• <b>TP 1:</b> <code>{setup['tp_price']:.3f}</code>\n"
                f"• <b>TP 2:</b> <code>{setup['tp2_price']:.3f}</code>\n"
                f"• <b>TP 3:</b> <code>{setup['tp3_price']:.3f}</code>\n\n"
                f"{htf_align_str}"
                f"<i>Sent automatically by Forex SMC AI Analyzer.</i>"
            )
            
            # Send Telegram notification
            success = send_telegram_alert(msg, image_filename)
            
            # Clean up temporary chart file
            if image_filename and os.path.exists(image_filename):
                try:
                    os.remove(image_filename)
                except Exception:
                    pass
                    
            if success:
                sent_signals[sig_key] = {
                    'time_sent': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'timeframe': setup['timeframe'],
                    'direction': dir_name,
                    'type': setup_name,
                    'price': setup['entry_price'],
                    'probability': prob,
                    'ticket_id': ticket_id,
                    'features': features
                }
                signals_sent_this_cycle += 1
                
    if signals_sent_this_cycle > 0:
        save_sent_signals(sent_signals)
        print(f"Sent {signals_sent_this_cycle} new alerts this cycle.")
    else:
        print("No new trade signals triggered this cycle.")
        
    # 8. Clean up invalid/old pending orders from MT5
    execute_enabled = os.getenv("MT5_EXECUTE_TRADES", "False").lower() == "true"
    if execute_enabled:
        try:
            magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
            prune_invalid_pending_orders(symbol, magic, active_high_confidence)
        except Exception as e:
            print(f"[Scanner Error] Error during pending orders pruning: {e}")
            
    # Free MT5 connection at the very end of the cycle
    import MetaTrader5 as mt5
    mt5.shutdown()
    print("--- Scan Cycle Finished ---")

def main():
    parser = argparse.ArgumentParser(description="Forex SMC Scanner background worker with Telegram Alerts.")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol (default: XAUUSD)")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold to alert (default: 0.60)")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=5, help="Scan interval in minutes (default: 5)")
    
    args = parser.parse_args()
    
    # Verify environment file variables exist
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id or token.startswith("YOUR_") or chat_id.startswith("YOUR_"):
        print("\n[WARNING] Telegram credentials are not configured in your .env file.")
        print("Alerts will print in the console but will NOT be sent to Telegram.")
        print("Please configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable alerts.\n")
        
    if args.loop:
        print(f"Starting background worker loop. Scanning every {args.interval} minutes...")
        try:
            while True:
                run_scan(args.symbol, args.threshold)
                print(f"Sleeping for {args.interval} minutes...")
                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\nScanner stopped by user.")
    else:
        # Run once
        run_scan(args.symbol, args.threshold)

if __name__ == "__main__":
    main()
