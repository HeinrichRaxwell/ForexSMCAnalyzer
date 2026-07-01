import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob
from src.main import get_active_setups, extract_active_htf_fvgs
from src.scanner_worker import is_good_fvg

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
        
    symbol = "XAUUSD"
    h4_df = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 100)
    h1_df = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 150)
    d1_df = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 50)
    
    d1_df = detect_swing_points(d1_df)
    d1_df = detect_structures(d1_df)
    
    h4_df = detect_swing_points(h4_df)
    h4_df = detect_structures(h4_df)
    h4_df = detect_fvg_and_ob(h4_df, symbol=symbol)
    
    close_prev = h4_df['Close'].shift(1).fillna(h4_df['Open'])
    tr = np.maximum(
        h4_df['High'] - h4_df['Low'],
        np.maximum(np.abs(h4_df['High'] - close_prev), np.abs(h4_df['Low'] - close_prev))
    )
    h4_df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    h1_df = detect_swing_points(h1_df)
    h1_df = detect_structures(h1_df)
    h1_df = detect_fvg_and_ob(h1_df, symbol=symbol)
    close_prev_h1 = h1_df['Close'].shift(1).fillna(h1_df['Open'])
    tr_h1 = np.maximum(
        h1_df['High'] - h1_df['Low'],
        np.maximum(np.abs(h1_df['High'] - close_prev_h1), np.abs(h1_df['Low'] - close_prev_h1))
    )
    h1_df['ATR_14'] = tr_h1.rolling(window=14, min_periods=1).mean()
    
    timeframes_data = {'D1': d1_df, 'H4': h4_df, 'H1': h1_df}
    
    # Check if H4 Index 99 is active
    h4_setups = get_active_setups(h4_df)
    h4_found = False
    for s in h4_setups:
        if s['setup_type'] == 0 and s['index'] == 99:
            print(f"H4 Setup (Index 99) is ACTIVE: {s['option_name']} at {s['entry_price']:.3f}")
            h4_found = True
    if not h4_found:
        print("H4 Setup (Index 99) is NOT active (mitigated or not found)")
        
    # Check if H1 Index 147 is active
    h1_setups = get_active_setups(h1_df)
    h1_found = False
    for s in h1_setups:
        if s['setup_type'] == 0 and s['index'] == 147:
            print(f"H1 Setup (Index 147) is ACTIVE: {s['option_name']} at {s['entry_price']:.3f}")
            h1_found = True
    if not h1_found:
        print("H1 Setup (Index 147) is NOT active (mitigated or not found)")

    # Run the same overlap and suppression check from scanner_worker
    active_fvgs_by_tf = {}
    for tf_name in ['H1', 'H4', 'D1']:
        active_fvgs_by_tf[tf_name] = extract_active_htf_fvgs(timeframes_data[tf_name])
        
    all_setups = []
    for tf_name in ['H4', 'H1']:
        tf_setups = get_active_setups(timeframes_data[tf_name])
        for s in tf_setups:
            if s['setup_type'] == 0 and ('Option A' in s['option_name'] or 'Option B' in s['option_name']):
                s['timeframe'] = tf_name
                all_setups.append(s)
                
    tf_weights = {'H1': 3, 'H4': 4, 'D1': 5}
    OVERLAP_PROXIMITY_USD = 15.0
    all_setups.sort(key=lambda s: (-tf_weights[s['timeframe']], s['entry_price']))
    
    claimed_zones = []
    print("\n--- Overlap and Suppression Results ---")
    for setup in all_setups:
        setup['suppressed'] = False
        setup_tf = setup['timeframe']
        setup_weight = tf_weights[setup_tf]
        direction = setup['direction']
        entry = setup['entry_price']
        opt_key = "A" if ("Option A" in setup.get('option_name', '') or "0.5" in setup.get('option_name', '')) else "B"
        
        overlaps_htf = False
        for (claimed_dir, claimed_entry, claimed_weight, claimed_opt) in claimed_zones:
            if claimed_weight > setup_weight:
                if abs(entry - claimed_entry) <= OVERLAP_PROXIMITY_USD:
                    setup['suppressed'] = True
                    setup['htf_conflict_reason'] = f"Overlaps with {['','','','H1','H4','D1'][claimed_weight]} FVG @ {claimed_entry:.3f}"
                    overlaps_htf = True
                    break
        
        if not overlaps_htf:
            claimed_zones.append((direction, entry, setup_weight, opt_key))
            
        if setup['index'] in [99, 147]:
            print(f"Setup {setup['timeframe']} Index {setup['index']} ({setup['option_name']}):")
            print(f"  Entry: {setup['entry_price']:.3f}")
            print(f"  Suppressed: {setup['suppressed']}")
            if setup['suppressed']:
                print(f"  Reason: {setup.get('htf_conflict_reason')}")
                
    mt5.shutdown()

if __name__ == "__main__":
    main()
