import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, get_pip_multiplier
from src.main import get_active_setups
from src.inference import predict_setup_probability
from src.scanner_worker import is_good_fvg

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
        
    symbol = "XAUUSD"
    # Fetch data
    h4_df = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 100)
    h1_df = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 150)
    d1_df = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 50)
    
    # Process D1 for trend
    d1_df = detect_swing_points(d1_df)
    d1_df = detect_structures(d1_df)
    
    timeframes_data = {'D1': d1_df, 'H4': h4_df, 'H1': h1_df}
    
    # Process H4
    h4_df = detect_swing_points(h4_df)
    h4_df = detect_structures(h4_df)
    h4_df = detect_fvg_and_ob(h4_df, symbol=symbol)
    
    # Calculate ATR_14 for H4
    close_prev = h4_df['Close'].shift(1).fillna(h4_df['Open'])
    tr = np.maximum(
        h4_df['High'] - h4_df['Low'],
        np.maximum(
            np.abs(h4_df['High'] - close_prev),
            np.abs(h4_df['Low'] - close_prev)
        )
    )
    h4_df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    timeframes_data['H4'] = h4_df
    
    # Process H1
    h1_df = detect_swing_points(h1_df)
    h1_df = detect_structures(h1_df)
    h1_df = detect_fvg_and_ob(h1_df, symbol=symbol)
    close_prev_h1 = h1_df['Close'].shift(1).fillna(h1_df['Open'])
    tr_h1 = np.maximum(
        h1_df['High'] - h1_df['Low'],
        np.maximum(
            np.abs(h1_df['High'] - close_prev_h1),
            np.abs(h1_df['Low'] - close_prev_h1)
        )
    )
    h1_df['ATR_14'] = tr_h1.rolling(window=14, min_periods=1).mean()
    timeframes_data['H1'] = h1_df
    
    # Get H4 active setups
    h4_setups = get_active_setups(h4_df)
    print("--- H4 ACTIVE SETUPS ---")
    for s in h4_setups:
        if s['setup_type'] == 0:  # FVG
            s['timeframe'] = 'H4'
            idx = s['index']
            # Calculate probability
            features = {
                'timeframe': 240,
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
            except Exception as e:
                prob = 0.0
            
            is_good, reason = is_good_fvg(h4_df, idx, s, symbol, timeframes_data)
            print(f"Index: {idx}, Time: {s['time']}, Dir: {'BULL' if s['direction'] == 1 else 'BEAR'}, Opt: {s['option_name']}")
            print(f"  Entry: {s['entry_price']:.3f}, SL: {s['sl_price']:.3f}, TP: {s['tp_price']:.3f}")
            print(f"  Prob: {prob:.2%}, Width: {s['fvg_width']:.2f} USD")
            print(f"  Quality Filter: {'PASS' if is_good else 'REJECTED - ' + reason}")
            
    # Get H1 active setups
    h1_setups = get_active_setups(h1_df)
    print("\n--- H1 ACTIVE SETUPS ---")
    for s in h1_setups:
        if s['setup_type'] == 0:
            s['timeframe'] = 'H1'
            idx = s['index']
            features = {
                'timeframe': 60,
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
            except Exception:
                prob = 0.0
            print(f"Index: {idx}, Time: {s['time']}, Dir: {'BULL' if s['direction'] == 1 else 'BEAR'}, Opt: {s['option_name']}")
            print(f"  Entry: {s['entry_price']:.3f}, SL: {s['sl_price']:.3f}, TP: {s['tp_price']:.3f}")
            print(f"  Prob: {prob:.2%}, Width: {s['fvg_width']:.2f} USD")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
