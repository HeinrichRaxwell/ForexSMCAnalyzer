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
from src.main import get_active_setups
from src.execution import validate_market_indicators

def main():
    if not connect_mt5():
        print("Error: Could not connect to MT5.")
        return
        
    symbol = "XAUUSDm"
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return
    current_price = tick.bid
    print(f"Current XAUUSDm Price: {current_price:.3f}\n")
    
    timeframes = {
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1
    }
    
    for tf_name, mt5_tf in timeframes.items():
        df = fetch_historical_data(symbol, mt5_tf, 250)
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
            
        print(f"=== Timeframe: {tf_name} (Found {len(tf_setups)} active options) ===")
        # Filter for setups within 40 USD of current price to keep it readable
        tf_setups = [s for s in tf_setups if abs(s['entry_price'] - current_price) <= 40.0]
        
        for s in tf_setups:
            features = {
                'timeframe': 30 if tf_name == 'M30' else 60,
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
            prob = predict_setup_probability(features)
            dist = abs(s['entry_price'] - current_price)
            
            max_dist = 30.0 if tf_name == 'M30' else 40.0
            dist_ok = dist <= max_dist
            
            is_valid_mkt = "N/A"
            if dist_ok:
                is_valid_mkt_bool, mkt_reason = validate_market_indicators(symbol, tf_name, s['direction'])
                is_valid_mkt = "VALID" if is_valid_mkt_bool else f"FAILED ({mkt_reason})"
                
            print(f"  Setup: {s['option_name']} | Dir: {'BUY' if s['direction'] == 1 else 'SELL'} | Entry: {s['entry_price']:.3f} | Dist: {dist:.2f} USD")
            print(f"    Prob (Confidence): {prob:.2%} (Threshold: 50%)")
            print(f"    Distance Check: {'PASS' if dist_ok else f'FAILED (Too far)'}")
            print(f"    Market Indicators: {is_valid_mkt}")
        print()

if __name__ == '__main__':
    main()
