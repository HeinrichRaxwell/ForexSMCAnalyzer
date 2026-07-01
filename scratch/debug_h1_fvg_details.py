import os
import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.append("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_snr_and_swapzones, detect_bpr
from src.labeler import get_killzone
from src.inference import predict_setup_probability
from src.rejection_detector import detect_rejection_at_level
from src.main import find_dynamic_tp, extract_active_htf_fvgs, get_active_setups
from src.execution import validate_market_indicators, get_active_broker_symbol

def debug_details():
    if not connect_mt5():
        print("Failed to connect.")
        return
        
    symbol = "XAUUSD"
    active_symbol = get_active_broker_symbol(symbol)
    print(f"Active Broker Symbol: {active_symbol}")
    
    tick = mt5.symbol_info_tick(active_symbol)
    if tick is None:
        print("Failed to get tick info.")
        mt5.shutdown()
        return
    current_price = tick.bid
    print(f"Current Price of Gold (Bid): {current_price}")
    
    # Load H1 data
    df = fetch_historical_data(active_symbol, mt5.TIMEFRAME_H1, 300)
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=active_symbol)
    
    close_prev = df['Close'].shift(1).fillna(df['Open'])
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(np.abs(df['High'] - close_prev), np.abs(df['Low'] - close_prev))
    )
    df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    setups = get_active_setups(df)
    
    print("\n--- DETAILED ACTIVE H1 BEARISH FVG SETUPS ---")
    h1_bear_fvgs = [s for s in setups if s['setup_type'] == 0 and s['direction'] == -1]
    
    if not h1_bear_fvgs:
        print("No active H1 Bearish FVG setups found!")
    else:
        for s in h1_bear_fvgs:
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
            prob = predict_setup_probability(features)
            dist = abs(s['entry_price'] - current_price)
            
            # Indicator validation
            is_valid_ind, ind_reason = validate_market_indicators(active_symbol, "H1", -1)
            
            print(f"\nSetup: {s['option_name']} at {s['time']}")
            print(f"  Entry Price: {s['entry_price']:.3f} | SL: {s['sl_price']:.3f} | TP 1: {s['tp_price']:.3f}")
            print(f"  Distance to Market: {dist:.3f} USD (Max Allowed: 20.0 USD)")
            print(f"  ML Win Probability: {prob:.2%}")
            print(f"  Indicator Check: {'PASSED' if is_valid_ind else 'FAILED'} ({ind_reason})")
            
    mt5.shutdown()

if __name__ == "__main__":
    debug_details()
