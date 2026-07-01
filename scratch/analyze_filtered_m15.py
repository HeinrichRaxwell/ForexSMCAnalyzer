import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob
from src.inference import predict_setup_probability
from src.main import get_active_setups
from src.scanner_worker import is_good_fvg

def main():
    if not connect_mt5():
        print("Failed to initialize MT5")
        return
        
    symbol = "XAUUSD"
    import MetaTrader5 as mt5
    
    # Fetch data
    d1_df = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 100)
    h4_df = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    h1_df = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
    m15_df = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 500)
    
    # Process
    d1_df = detect_swing_points(d1_df, window=5)
    d1_df = detect_structures(d1_df)
    d1_df = detect_fvg_and_ob(d1_df, symbol=symbol)
    
    h4_df = detect_swing_points(h4_df, window=5)
    h4_df = detect_structures(h4_df)
    h4_df = detect_fvg_and_ob(h4_df, symbol=symbol)
    
    h1_df = detect_swing_points(h1_df, window=5)
    h1_df = detect_structures(h1_df)
    h1_df = detect_fvg_and_ob(h1_df, symbol=symbol)
    
    m15_df = detect_swing_points(m15_df, window=5)
    m15_df = detect_structures(m15_df)
    m15_df = detect_fvg_and_ob(m15_df, symbol=symbol)
        
    # Calculate ATR_14
    close_prev = m15_df['Close'].shift(1).fillna(m15_df['Open'])
    tr = np.maximum(
        m15_df['High'] - m15_df['Low'],
        np.maximum(
            np.abs(m15_df['High'] - close_prev),
            np.abs(m15_df['Low'] - close_prev)
        )
    )
    m15_df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    setups = get_active_setups(m15_df)
    
    print("\nLooking for Bearish FVG at 2026-06-05 00:15...")
    target_time = "2026-06-05 00:15:00"
    
    timeframes_data = {'D1': d1_df, 'H4': h4_df, 'H1': h1_df, 'M15': m15_df}
    
    for s in setups:
        s_time_str = str(s['time'])
        if target_time in s_time_str:
            print(f"\nFOUND: {s['option_name']} at {s['time']}")
            print(f"Entry Price: {s['entry_price']:.3f} | SL: {s['sl_price']:.3f} | TP1: {s['tp_price']:.3f}")
            
            # Check Quality Filter
            is_good, reason = is_good_fvg(m15_df, s['index'], s, symbol, timeframes_data)
            print(f"Quality Filter: {'PASS' if is_good else 'REJECT'} - {reason}")
            
            # Predict probability
            features = {
                'timeframe': 15,
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
            print(f"AI Success Score: {prob:.2%}")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
