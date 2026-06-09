import os
import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_snr_and_swapzones, detect_bpr, detect_indecision_candles
from src.main import get_active_setups, extract_active_htf_fvgs
from src.inference import predict_setup_probability
from src.execution import get_active_broker_symbol, validate_market_indicators

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
        
    symbol = "XAUUSD"
    broker_symbol = get_active_broker_symbol(symbol)
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is not None:
        current_price = (tick.bid + tick.ask) / 2
        print(f"Current XAUUSD Market Price: {current_price:.2f}")
    else:
        print("Failed to get current tick")
        return
        
    timeframes = ['M15', 'M30', 'H1', 'H4', 'D1']
    tf_minutes_map = {'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    tf_weights = {'M15': 1, 'M30': 2, 'H1': 3, 'H4': 4, 'D1': 5}
    
    timeframes_data = {}
    for tf_name in timeframes:
        if tf_name == 'D1':
            df = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 100)
        elif tf_name == 'H4':
            df = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
        elif tf_name == 'H1':
            df = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
        elif tf_name == 'M30':
            df = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 400)
        else:
            df = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 500)
            
        df = detect_swing_points(df)
        df = detect_structures(df)
        df = detect_fvg_and_ob(df, symbol=symbol)
        df = detect_snr_and_swapzones(df, symbol=symbol)
        df = detect_bpr(df, symbol=symbol)
        df = detect_indecision_candles(df, symbol=symbol)
        
        close_prev = df['Close'].shift(1).fillna(df['Open'])
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - close_prev),
                np.abs(df['Low'] - close_prev)
            )
        )
        df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
        timeframes_data[tf_name] = df
        
    # Extract active HTF FVGs
    active_fvgs_by_tf = {}
    for tf_name in timeframes:
        active_fvgs_by_tf[tf_name] = extract_active_htf_fvgs(timeframes_data[tf_name])
        
    all_setups = []
    for tf_name in timeframes:
        tf_setups = get_active_setups(timeframes_data[tf_name])
        for s in tf_setups:
            s['timeframe'] = tf_name
            if 'OB' in s['option_name']:
                s['strategy'] = 'OB'
            elif 'BPR' in s['option_name']:
                s['strategy'] = 'BPR'
            elif 'IC' in s['option_name']:
                s['strategy'] = 'IC'
            elif 'Swap' in s['option_name']:
                s['strategy'] = 'Swapzone'
            else:
                s['strategy'] = 'FVG'
            all_setups.append(s)
            
    # HTF Prioritization & suppression check
    for setup in all_setups:
        setup['htf_prioritized'] = False
        setup['suppressed'] = False
        setup['htf_conflict_reason'] = ""
        setup_tf = setup['timeframe']
        
        for htf_name in timeframes:
            if tf_weights[htf_name] > tf_weights[setup_tf]:
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_same = (setup['direction'] == 1 and htf_fvg['type'] == 'BULLISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BEARISH')
                    if is_same:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['htf_prioritized'] = True
                            
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_opp = (setup['direction'] == 1 and htf_fvg['type'] == 'BEARISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BULLISH')
                    if is_opp:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['suppressed'] = True
                            setup['htf_conflict_reason'] = f"Opposite HTF FVG ({htf_name})"
                            
    print("\n--- ACTIVE UNMITIGATED SETUPS IN SYSTEM ---")
    for s in all_setups:
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
        except Exception:
            prob = 0.0
            
        dist = abs(s['entry_price'] - current_price)
        
        # Max distance limits
        max_dist = 20.0
        if s['timeframe'] == 'H4':
            max_dist = 60.0
        elif s['timeframe'] == 'H1':
            max_dist = 40.0
        elif s['timeframe'] == 'M30':
            max_dist = 30.0
        elif s['timeframe'] == 'D1':
            max_dist = 150.0
            
        dist_ok = dist <= max_dist
        
        status_ok = "OK"
        if s['suppressed']:
            status_ok = f"SUPPRESSED ({s['htf_conflict_reason']})"
        elif not dist_ok:
            status_ok = f"TOO_FAR ({dist:.1f} > {max_dist})"
            
        print(f"TF: {s['timeframe']} | {s['option_name']} | Dir: {'BULL' if s['direction']==1 else 'BEAR'} | Entry: {s['entry_price']:.2f} | Dist: {dist:.2f} | Prob: {prob:.1%} | Status: {status_ok}")
        
    mt5.shutdown()

if __name__ == "__main__":
    main()
