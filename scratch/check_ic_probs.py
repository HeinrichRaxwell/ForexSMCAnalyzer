import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.main import get_active_setups
from src.inference import predict_setup_probability
from src.smc_detector import (
    detect_swing_points, detect_structures, detect_fvg_and_ob,
    detect_supply_demand_zones, detect_indecision_candles
)

def main():
    connect_mt5()
    df = fetch_historical_data('XAUUSD', mt5.TIMEFRAME_M30, 200)
    df = detect_swing_points(df)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df)
    df = detect_supply_demand_zones(df)
    df = detect_indecision_candles(df)
    
    # Calculate ATR_14
    close_prev = df['Close'].shift(1).fillna(df['Open'])
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(np.abs(df['High'] - close_prev), np.abs(df['Low'] - close_prev))
    )
    df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    setups = get_active_setups(df, symbol='XAUUSD')
    print("=== Active M30 IC Setups today (June 5, 2026) ===")
    found = False
    for s in setups:
        if 'IC' in s['option_name']:
            found = True
            features = {
                'timeframe': 30,
                'hour': int(s['hour']),
                'day_of_week': int(s['day_of_week']),
                'setup_type': int(s['setup_type']),
                'direction': int(s['direction']),
                'entry_price': float(s['entry_price']),
                'sl_price': float(s['sl_price']),
                'tp_price': float(s['tp_price']),
                'risk_pips': float(s['risk_pips']),
                'atr_14': float(s['atr_14']),
                'trend': int(s['trend']),
                'relative_risk': float(s['relative_risk']),
                'killzone': int(s['killzone']),
                'fvg_width': float(s['fvg_width']),
                'relative_fvg_width': float(s['relative_fvg_width']),
                'near_psychological_level': int(s['near_psychological_level'])
            }
            prob = predict_setup_probability(features)
            print(f"Time Sent: {s['time']}")
            print(f"Option Name: {s['option_name']}")
            print(f"Entry: {s['entry_price']:.3f} | SL: {s['sl_price']:.3f} | TP: {s['tp_price']:.3f}")
            print(f"Win Probability: {prob:.2%}")
            print(f"Rejection Confirmed: {s['rejection_confirmed']}")
            print("-" * 50)
    if not found:
        print("No active M30 IC setups found in the latest 200 candles.")

if __name__ == "__main__":
    main()
