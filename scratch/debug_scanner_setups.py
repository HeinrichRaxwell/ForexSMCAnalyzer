import os
import sys
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

sys.path.append("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_snr_and_swapzones, detect_bpr
from src.main import get_active_setups, extract_active_htf_fvgs
from src.rejection_detector import detect_rejection_at_level
from src.inference import predict_setup_probability
from src.scanner_worker import is_good_fvg

def main():
    if not connect_mt5():
        print("Failed to initialize MT5")
        return
        
    symbol = "XAUUSD"
    print("Fetching MTF data...")
    timeframes_data = {}
    timeframes_data['D1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 100)
    timeframes_data['H4'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    timeframes_data['H1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
    timeframes_data['M30'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 400)
    timeframes_data['M15'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 500)
    
    # Run SMC detection
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        df_tf = detect_swing_points(df_tf, window=5)
        df_tf = detect_structures(df_tf)
        df_tf = detect_fvg_and_ob(df_tf, symbol=symbol)
        df_tf = detect_snr_and_swapzones(df_tf)
        df_tf = detect_bpr(df_tf, symbol=symbol)
        
        # ATR_14
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
        
    active_fvgs_by_tf = {}
    for tf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
        active_fvgs_by_tf[tf_name] = extract_active_htf_fvgs(timeframes_data[tf_name])
        
    all_setups = []
    for tf_name in ['D1', 'H4', 'H1', 'M30', 'M15']:
        tf_setups = get_active_setups(timeframes_data[tf_name])
        for s in tf_setups:
            if s['setup_type'] == 0 and ('Option A' in s['option_name'] or 'Option B' in s['option_name']):
                s['timeframe'] = tf_name
                all_setups.append(s)
                
    tf_weights = {'M15': 1, 'M30': 2, 'H1': 3, 'H4': 4, 'D1': 5}
    tf_minutes_map = {'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    
    # 4. Multi-Timeframe Alignment and Suppression
    for setup in all_setups:
        setup['htf_prioritized'] = False
        setup['matching_htf_fvgs'] = []
        setup['suppressed'] = False
        setup['htf_conflict_reason'] = ""
        setup_tf = setup['timeframe']
        
        for htf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
            if tf_weights[htf_name] > tf_weights[setup_tf]:
                # Prioritization
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
                            
                # Conflict Suppression - only if entry is inside the opposite HTF FVG
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_opp = (setup['direction'] == 1 and htf_fvg['type'] == 'BEARISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BULLISH')
                    if is_opp:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['suppressed'] = True
                            setup['htf_conflict_reason'] = f"Entry inside opposite active {htf_name} FVG"
                            break
                        
        if setup_tf not in ['M15']:
            m15_df = timeframes_data.get('M15')
            if m15_df is not None and not m15_df.empty:
                rej_confirmed = detect_rejection_at_level(m15_df, setup['entry_price'], setup['direction'], lookback=15)
                setup['rejection_confirmed'] = rej_confirmed

    # Overlap Suppression
    OVERLAP_PROXIMITY_USD = 15.0
    all_setups.sort(key=lambda s: (-tf_weights[s['timeframe']], s['entry_price']))
    claimed_zones = []
    
    for setup in all_setups:
        if setup['suppressed']:
            continue
        setup_tf = setup['timeframe']
        setup_weight = tf_weights[setup_tf]
        direction = setup['direction']
        entry = setup['entry_price']
        opt_key = "A" if ("Option A" in setup.get('option_name', '') or "Midpoint" in setup.get('option_name', '') or "0.5" in setup.get('option_name', '')) else "B"
        
        overlaps_htf = False
        for (claimed_dir, claimed_entry, claimed_weight, claimed_opt) in claimed_zones:
            if claimed_weight > setup_weight:
                if abs(entry - claimed_entry) <= OVERLAP_PROXIMITY_USD:
                    setup['suppressed'] = True
                    setup['htf_conflict_reason'] = f"Overlaps with {['','M15','M30','H1','H4','D1'][claimed_weight]} FVG @ {claimed_entry:.3f}"
                    overlaps_htf = True
                    break
        
        if not overlaps_htf:
            claimed_zones.append((direction, entry, setup_weight, opt_key))

    print("\n=== Active Setups Analyzed ===")
    for s in all_setups:
        # We are interested in H1 setups or setups near the current price
        print(f"Time: {s['time']} | TF: {s['timeframe']} | Dir: {'BULL' if s['direction']==1 else 'BEAR'} | Opt: {s.get('option_name')} | Entry: {s['entry_price']:.3f} | Suppressed: {s['suppressed']} | Reason: {s.get('htf_conflict_reason', 'None')}")
        if s['timeframe'] == 'H1':
            # Check quality
            is_good, reason = is_good_fvg(timeframes_data['H1'], s['index'], s, symbol, timeframes_data)
            print(f"  -> Quality Check: {is_good} | Reason: {reason}")
            # Check ML prob
            features = {
                'timeframe': tf_minutes_map[s['timeframe']],
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
                print(f"  -> ML Win Prob: {prob:.2%}")
            except Exception as e:
                print(f"  -> ML error: {e}")
                
    mt5.shutdown()

if __name__ == "__main__":
    main()
