import os
import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_snr_and_swapzones, detect_bpr
from src.labeler import get_killzone
from src.inference import predict_setup_probability
from src.rejection_detector import detect_rejection_at_level
from src.main import find_dynamic_tp, extract_active_htf_fvgs, get_active_setups

def check_setups():
    if not connect_mt5():
        print("Failed to connect.")
        return
        
    symbol = "XAUUSD"
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    active_symbol = symbol
    for sym in symbols_to_try:
        if mt5.symbol_info(sym) is not None:
            active_symbol = sym
            break
            
    print(f"Active symbol: {active_symbol}")
    
    timeframes_data = {}
    timeframe_map = {
        'D1': mt5.TIMEFRAME_D1,
        'H4': mt5.TIMEFRAME_H4,
        'H1': mt5.TIMEFRAME_H1,
        'M30': mt5.TIMEFRAME_M30,
        'M15': mt5.TIMEFRAME_M15
    }
    
    # Fetch data
    for tf_name, tf_const in timeframe_map.items():
        # D1 needs 50, H4 100, H1 150, M30 200, M15 200
        num_candles = 50 if tf_name == 'D1' else (100 if tf_name == 'H4' else (150 if tf_name == 'H1' else 200))
        df_tf = fetch_historical_data(active_symbol, tf_const, num_candles)
        
        # Run detection
        df_tf = detect_swing_points(df_tf, window=5)
        df_tf = detect_structures(df_tf)
        df_tf = detect_fvg_and_ob(df_tf, symbol=active_symbol)
        df_tf = detect_snr_and_swapzones(df_tf)
        df_tf = detect_bpr(df_tf, symbol=active_symbol)
        
        close_prev = df_tf['Close'].shift(1).fillna(df_tf['Open'])
        tr = np.maximum(
            df_tf['High'] - df_tf['Low'],
            np.maximum(np.abs(df_tf['High'] - close_prev), np.abs(df_tf['Low'] - close_prev))
        )
        df_tf['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
        timeframes_data[tf_name] = df_tf

    mt5.shutdown()
    
    # Extract active HTF FVGs
    active_fvgs_by_tf = {}
    for tf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
        active_fvgs_by_tf[tf_name] = extract_active_htf_fvgs(timeframes_data[tf_name])
        
    # Get setups
    all_setups = []
    tf_weights = {'M15': 1, 'M30': 2, 'H1': 3, 'H4': 4, 'D1': 5}
    tf_minutes_map = {'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    
    for tf_name in ['D1', 'H4', 'H1', 'M30', 'M15']:
        tf_setups = get_active_setups(timeframes_data[tf_name])
        print(f"\nTimeframe {tf_name} - Found {len(tf_setups)} raw active setups")
        for s in tf_setups:
            s['timeframe'] = tf_name
            all_setups.append(s)
            
    # Process alignments
    for setup in all_setups:
        setup['htf_prioritized'] = False
        setup['matching_htf_fvgs'] = []
        setup['suppressed'] = False
        setup['htf_conflict_reason'] = ""
        setup_tf = setup['timeframe']
        
        for htf_name in ['M30', 'H1', 'H4', 'D1']:
            if tf_weights[htf_name] > tf_weights[setup_tf]:
                # HTF Prioritization
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
                            
                # Conflict Suppression
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_opp = (setup['direction'] == 1 and htf_fvg['type'] == 'BEARISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BULLISH')
                    if is_opp:
                        setup['suppressed'] = True
                        setup['htf_conflict_reason'] = f"Opposite active {htf_name} FVG"
                        break
                        
    # Run XGBoost inference
    print("\n" + "="*80)
    print("DETAILED SETUPS ANALYSIS WITH ML AND ALIGNMENT:")
    print("="*80)
    
    count_notified = 0
    for setup in all_setups:
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
        
        try:
            prob = predict_setup_probability(features)
        except Exception as e:
            print(f"Error predicting probability for {setup['timeframe']}: {e}")
            continue
            
        direction_str = "BUY" if setup['direction'] == 1 else "SELL"
        setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
        if "BPR" in setup['option_name']:
            setup_name = "BPR"
            
        print(f"TF: {setup['timeframe']:<3} | {direction_str:<4} {setup_name:<4} ({setup['option_name'][:18]}) | Prob: {prob:.2%} | Suppressed: {setup['suppressed']} ({setup['htf_conflict_reason']})")
        if prob >= 0.60 and not setup['suppressed']:
            count_notified += 1
            
    print(f"\nTotal setups that would be notified (Prob >= 60% & not suppressed): {count_notified}")

if __name__ == "__main__":
    check_setups()
