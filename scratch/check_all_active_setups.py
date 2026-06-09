import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import (
    detect_swing_points, 
    detect_structures, 
    detect_fvg_and_ob,
    detect_snr_and_swapzones,
    detect_bpr,
    detect_indecision_candles
)
from src.labeler import get_killzone
from src.inference import predict_setup_probability
from src.main import get_active_setups, find_dynamic_tp
from src.execution import get_active_broker_symbol, validate_market_indicators

def main():
    if not connect_mt5():
        print("Error: Could not connect to MT5.")
        return
        
    symbol = "XAUUSDm"
    magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Error: Could not get tick for {symbol}")
        return
    current_price = tick.bid
    print(f"Current XAUUSDm Price: {current_price:.3f} (Bid) / {tick.ask:.3f} (Ask)")
    
    # Check MT5 pending orders on account
    orders = mt5.orders_get(symbol=symbol, magic=magic)
    print(f"\nActive MT5 Pending Orders for magic {magic}:")
    if orders is None or len(orders) == 0:
        print("  None")
    else:
        for o in orders:
            o_type = "BUY LIMIT" if o.type == mt5.ORDER_TYPE_BUY_LIMIT else "SELL LIMIT"
            print(f"  Ticket #{o.ticket} | Type: {o_type} | Price: {o.price_open:.3f} | SL: {o.price_sl:.3f} | TP: {o.price_tp:.3f} | Dist: {abs(o.price_open - current_price):.2f} USD")
            
    # Load and scan multi-timeframe data
    timeframes = {
        'M15': (mt5.TIMEFRAME_M15, 200),
        'M30': (mt5.TIMEFRAME_M30, 200),
        'H1': (mt5.TIMEFRAME_H1, 200),
        'H4': (mt5.TIMEFRAME_H4, 200)
    }
    
    print("\nScanning active setups across timeframes...")
    for tf_name, (mt5_tf, num_bars) in timeframes.items():
        df = fetch_historical_data(symbol, mt5_tf, num_bars)
        if df is None or df.empty:
            continue
            
        df = detect_swing_points(df)
        df = detect_structures(df)
        df = detect_fvg_and_ob(df, symbol=symbol)
        df = detect_snr_and_swapzones(df, symbol=symbol)
        df = detect_bpr(df, symbol=symbol)
        df = detect_indecision_candles(df, symbol=symbol)
        
        # Calculate ATR_14
        close_prev = df['Close'].shift(1).fillna(df['Open'])
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - close_prev),
                np.abs(df['Low'] - close_prev)
            )
        )
        df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
        
        setups = get_active_setups(df)
        
        tf_setups = []
        for s in setups:
            # Check strategy name
            opt_name = s['option_name']
            if "OB" in opt_name:
                strat = "OB"
            elif "BPR" in opt_name:
                strat = "BPR"
            elif "IC" in opt_name:
                strat = "IC"
            elif "Swap" in opt_name:
                strat = "Swapzone"
            else:
                strat = "FVG"
            
            s['strategy'] = strat
            s['timeframe'] = tf_name
            tf_setups.append(s)
            
        if not tf_setups:
            continue
            
        print(f"\n--- Timeframe: {tf_name} (Found {len(tf_setups)} active option setups) ---")
        
        # Sort setups by distance to current price
        tf_setups.sort(key=lambda s: abs(s['entry_price'] - current_price))
        
        for s in tf_setups:
            # Rebuild features for prediction
            features = {
                'timeframe': 15 if tf_name == 'M15' else (30 if tf_name == 'M30' else (60 if tf_name == 'H1' else 240)),
                'hour': s['hour'],
                'day_of_week': s['day_of_week'],
                'setup_type': s['setup_type'],
                'direction': s['direction'],
                'entry_price': s['entry_price'],
                'sl_price': s['sl_price'],
                'tp_price': s['tp_price'],
                'risk_pips': s['risk_pips'],
                'atr_14': s['atr_14'],
                'trend': s['trend'],
                'relative_risk': s['relative_risk'],
                'killzone': s['killzone'],
                'fvg_width': s['fvg_width'],
                'relative_fvg_width': s['relative_fvg_width']
            }
            try:
                prob = predict_setup_probability(features)
                dist = abs(s['entry_price'] - current_price)
                
                # Check distance limit
                max_dist = 20.0
                if tf_name == 'H4':
                    max_dist = 60.0
                elif tf_name == 'H1':
                    max_dist = 40.0
                elif tf_name == 'M30':
                    max_dist = 30.0
                
                dist_ok = dist <= max_dist
                
                # Check indicators
                is_valid_mkt = "N/A"
                if dist_ok:
                    is_valid_mkt_bool, mkt_reason = validate_market_indicators(symbol, tf_name, s['direction'])
                    is_valid_mkt = "VALID" if is_valid_mkt_bool else f"FAILED ({mkt_reason})"
                
                print(f"  Setup: {s['option_name']} | Dir: {'BUY' if s['direction'] == 1 else 'SELL'} | Entry: {s['entry_price']:.3f} | Dist: {dist:.2f} USD")
                print(f"    Prob (Confidence): {prob:.2%} (Threshold: 50%)")
                print(f"    Distance Check: {'PASS' if dist_ok else f'FAILED (Too far, limit {max_dist} USD)'}")
                print(f"    Market Indicators: {is_valid_mkt}")
                
            except Exception as e:
                print(f"  Error evaluating {s['option_name']}: {e}")

if __name__ == '__main__':
    main()
