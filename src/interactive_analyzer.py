import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# Add project root to python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Force spread filtering for interactive diagnostics; live entry quality remains telemetry-only.
os.environ["MT5_ENFORCE_SPREAD_FILTER"] = "True"
os.environ["MT5_ENFORCE_ENTRY_GATE"] = "False"



from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import (
    detect_swing_points,
    detect_structures,
    detect_fvg_and_ob,
    detect_snr_and_swapzones,
    detect_bpr,
    get_pip_multiplier,
    detect_indecision_candles,
    detect_supply_demand_zones,
)
from src.inference import predict_setup_probability
from src.rejection_detector import detect_rejection_at_level
from src.main import find_dynamic_tp, get_active_setups
from src.indicators.knn_classifier import run_knn_classifier, calculate_knn_probability_at_bar
from src.indicators.volume_clusters import calculate_volume_clusters
from src.entry_quality_gate import (
    evaluate_entry_quality,
    build_oscillator_context,
    build_spread_context,
)
from src.execution import get_active_broker_symbol

def apply_smc_detectors(df: pd.DataFrame, symbol: str, closed_only: bool = True) -> pd.DataFrame:
    """Run all structural detectors to find BOS, CHoCH, OB, FVG, BPR, IC, SnD."""
    is_source_closed_only = bool(getattr(df, "attrs", {}).get("closed_only", False))
    if not is_source_closed_only:
        df.attrs["has_running_candle"] = True

    df = detect_swing_points(df)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=symbol)
    df = detect_snr_and_swapzones(df)
    df = detect_bpr(df)
    df = detect_indecision_candles(df)
    df = detect_supply_demand_zones(df)
    
    if closed_only:
        df = df.iloc[:-1].copy()
        df.attrs["closed_only"] = True
    elif is_source_closed_only:
        df.attrs["closed_only"] = True
        
    return df

def get_strategy_name(option_name: str) -> str:
    if "OB" in option_name:
        return "OB"
    elif "BPR" in option_name:
        return "BPR"
    elif "IC" in option_name:
        return "IC"
    elif "Swap" in option_name:
        return "Swapzone"
    elif "Breaker" in option_name:
        return "Breaker"
    elif "SND" in option_name:
        return "SND"
    elif "Pivot" in option_name:
        return "Pivot"
    else:
        return "FVG"

def get_live_spread_context(symbol: str):
    """Build spread context from live MT5 tick and symbol precision."""
    broker_symbol = get_active_broker_symbol(symbol)
    tick = mt5.symbol_info_tick(broker_symbol)
    info = mt5.symbol_info(broker_symbol)
    if tick is None or info is None:
        return None

    digits = getattr(info, "digits", None)
    point = getattr(info, "point", None)
    if not point:
        point = 10 ** -int(digits) if digits is not None else 0.001

    return build_spread_context(
        bid=getattr(tick, "bid", None),
        ask=getattr(tick, "ask", None),
        point=point,
        digits=digits,
    )

def analyze_symbol_setups(symbol: str, confidence_threshold: float = 0.50):
    print(f"\n==================================================")
    print(f"[SCAN] ANALYZING SYMBOL: {symbol.upper()}")
    print(f"==================================================")
    
    # 1. Connect to MT5
    if not connect_mt5():
        print("[ERROR] Failed to connect to MT5.")
        return
        
    broker_symbol = get_active_broker_symbol(symbol)
    print(f"Broker symbol matched: {broker_symbol}")
    
    # 2. Get Live Price and Spread info
    tick = mt5.symbol_info_tick(broker_symbol)
    info = mt5.symbol_info(broker_symbol)
    if tick is None or info is None:
        print(f"[ERROR] Failed to fetch live tick/info for {broker_symbol}. Symbol might not be available/selected.")
        mt5.shutdown()
        return
        
    pip_multiplier = get_pip_multiplier(symbol)
    point = getattr(info, "point", 0.00001)
    digits = getattr(info, "digits", 5)
    
    bid = tick.bid
    ask = tick.ask
    spread_price = ask - bid
    spread_pips = spread_price / pip_multiplier if pip_multiplier > 0 else 0.0
    
    print(f"[DATA] Market Price: Bid = {bid:.{digits}f} | Ask = {ask:.{digits}f}")
    print(f"[DATA] Spread: {spread_pips:.1f} pips ({spread_price:.{digits}f} points)")
    
    # 3. Load historical OHLCV data
    timeframes_data = {}
    print("[WAIT] Fetching historical data (D1, H4, H1, M30, M15, M5, M1)...")
    try:
        timeframes_data['D1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 100)
        timeframes_data['H4'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
        timeframes_data['H1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 300)
        timeframes_data['M30'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 400)
        timeframes_data['M15'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 500)
        timeframes_data['M5'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M5, 500)
        timeframes_data['M1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M1, 500)
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        mt5.shutdown()
        return

    d1_pivot_source = timeframes_data.get('D1')
    
    # 4. Detect SMC Structures & indicators
    print("[WAIT] Processing SMC structures and FLoOP/KNN indicators...")
    tf_trends = {}
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        df_tf = apply_smc_detectors(df_tf, symbol=symbol, closed_only=True)
        # ATR 14
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

        # Calculate FLoOP
        from src.indicators.floop import calculate_atr, calculate_range_filter
        try:
            df_tf_copy = df_tf.copy()
            df_tf_copy['time'] = pd.to_datetime(df_tf_copy['time'])
            df_tf_copy.set_index('time', inplace=True)
            atr_floop = calculate_atr(df_tf_copy, 14)
            _, trend_floop, _ = calculate_range_filter(df_tf_copy['Close'], atr_floop, sensitivity=6, atr_multiplier=0.8)
            tf_trends[tf_name] = pd.Series(trend_floop, index=df_tf_copy.index)
        except Exception as e:
            tf_trends[tf_name] = None
            
    # Pre-calculate KNN and Volume Profile
    tf_knn_data = {}
    tf_vp_data = {}
    for tf_name, df_tf in timeframes_data.items():
        try:
            pc1, pc2, pc3, pc4, target_clean = run_knn_classifier(
                df_tf,
                atr_period=10, factor=2.0,
                k_neighbors=10, sampling_window_size=1000, momentum_window=10,
                normalizing_window_size=1000,
                lazy=True
            )
            t_last = len(df_tf) - 1
            knn_up, knn_down = calculate_knn_probability_at_bar(
                t_last, pc1.values, pc2.values, pc3.values, pc4.values, target_clean.values,
                k=10, sampling_window=1000, stride=10
            )
            tf_knn_data[tf_name] = (knn_up, knn_down)
        except Exception:
            tf_knn_data[tf_name] = (0.0, 0.0)
            
        try:
            clusters_data = calculate_volume_clusters(
                df_tf, lookback=200, k=5, iterations=20, rows=20
            )
            tf_vp_data[tf_name] = clusters_data
        except Exception:
            tf_vp_data[tf_name] = {}

    # Extract setups
    all_setups = []
    for tf_name in ['D1', 'H4', 'H1', 'M30', 'M15']:
        tf_setups = get_active_setups(timeframes_data[tf_name], symbol=symbol, tf_trends=tf_trends, df_d1=d1_pivot_source)
        for s in tf_setups:
            s['timeframe'] = tf_name
            s['strategy'] = get_strategy_name(s['option_name'])
            s['symbol'] = symbol
            if s['strategy'] in ['FVG', 'OB', 'BPR', 'IC', 'SND', 'Pivot', 'Swapzone', 'Breaker']:
                all_setups.append(s)

    # MTF Alignment & Priorities
    tf_weights = {'M15': 1, 'M30': 2, 'H1': 3, 'H4': 4, 'D1': 5}
    tf_minutes_map = {'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    
    # Extract active HTF FVGs
    from src.main import extract_active_htf_fvgs
    active_fvgs_by_tf = {}
    for tf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
        active_fvgs_by_tf[tf_name] = extract_active_htf_fvgs(timeframes_data[tf_name])

    for setup in all_setups:
        setup['htf_prioritized'] = False
        setup['matching_htf_fvgs'] = []
        setup['suppressed'] = False
        setup['htf_conflict_reason'] = ""
        setup_tf = setup['timeframe']
        
        for htf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
            if tf_weights[htf_name] > tf_weights[setup_tf]:
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_same = (setup['direction'] == 1 and htf_fvg['type'] == 'BULLISH') or \
                              (setup['direction'] == -1 and htf_fvg['type'] == 'BEARISH')
                    if is_same:
                        if htf_fvg['bottom'] <= setup['entry_price'] <= htf_fvg['top']:
                            setup['htf_prioritized'] = True
                            setup['matching_htf_fvgs'].append(f"{htf_name} FVG")
                            
        # Extract features for prediction
        tf = setup['timeframe']
        knn_up_tf, knn_down_tf = tf_knn_data.get(tf, (0.0, 0.0))
        knn_prob_sig = knn_up_tf if setup['direction'] == 1 else knn_down_tf
        knn_prob_opp = knn_down_tf if setup['direction'] == 1 else knn_up_tf
        
        clusters_data_tf = tf_vp_data.get(tf, {})
        dist_entry_to_poc = 0.0
        dist_entry_to_nearest_poc = 0.0
        if clusters_data_tf and 'current_poc' in clusters_data_tf:
            curr_poc = clusters_data_tf['current_poc']
            entry = setup['entry_price']
            dist_entry_to_poc = (entry - curr_poc) / curr_poc if curr_poc > 0 else 0.0
            
            pocs = clusters_data_tf.get('pocs', [])
            if pocs:
                dist_entry_to_nearest_poc = min(abs(entry - poc) for poc in pocs) / entry

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
            'relative_fvg_width': setup['relative_fvg_width'],
            'near_psychological_level': setup['near_psychological_level'],
            'knn_prob_sig': knn_prob_sig,
            'knn_prob_opp': knn_prob_opp,
            'dist_entry_to_poc': dist_entry_to_poc,
            'dist_entry_to_nearest_poc': dist_entry_to_nearest_poc,
            'dist_entry_to_pp': setup.get('dist_entry_to_pp', 0.0),
            'dist_entry_to_nearest_pivot': setup.get('dist_entry_to_nearest_pivot', 0.0),
            'floop_signal': setup['floop_signal'],
            'floop_strength': setup['floop_strength'],
            'floop_trend': setup.get('floop_trend', 0),
            'floop_trend_aligned': 1 if setup.get('floop_trend', 0) == setup['direction'] else 0
        }
        
        setup['features'] = features
        try:
            setup['probability'] = predict_setup_probability(features)
        except Exception as e:
            print(f"[Warning] Failed to predict probability: {e}")
            setup['probability'] = 0.5

    # 5. Evaluate and Print active setups
    if not all_setups:
        print("[INFO] No active SMC setups found on D1, H4, H1, M30, or M15 charts.")
        mt5.shutdown()
        return

    print(f"\n[INFO] Active Setups Found: {len(all_setups)}")
    print("-" * 75)
    
    spread_ctx = get_live_spread_context(symbol)
    
    for idx, setup in enumerate(all_setups):
        tf = setup['timeframe']
        strat = setup['strategy']
        dir_label = "BUY" if setup['direction'] == 1 else "SELL"
        entry_price = setup['entry_price']
        sl = setup['sl_price']
        tp = setup['tp_price']
        prob = setup['probability']
        
        # Evaluate Entry Quality Gate
        tf_df = timeframes_data.get(tf)
        gate_decision = evaluate_entry_quality(
            setup,
            strategy=strat,
            probability=prob,
            accept_threshold=confidence_threshold,
            spread=spread_ctx,
            oscillator=build_oscillator_context(tf_df),
        )
        
        status = "ENTRY APPROVED" if gate_decision.allowed else "SKIP"
        reason = gate_decision.reason
        
        print(f"Setup #{idx+1}: {tf} | {strat} | {dir_label}")
        print(f"  +- Price Levels: Entry = {entry_price:.{digits}f} | SL = {sl:.{digits}f} | TP = {tp:.{digits}f}")
        print(f"  +- Win Probability (ML): {prob:.2%}")
        if gate_decision.spread_r is not None:
            print(f"  +- Spread/Risk Ratio: {gate_decision.spread_r:.2%}")
        print(f"  +- STATUS: {status} ({reason})")
        print("-" * 75)
        
    mt5.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Interactive Forex/Crypto SMC Analyzer")
    parser.add_argument("symbol", nargs="?", type=str, help="Specific trading symbol to analyze (optional)")
    parser.add_argument("--threshold", type=float, default=0.40, help="Confidence threshold (default: 0.40)")
    args = parser.parse_args()
    
    if args.symbol:
        analyze_symbol_setups(args.symbol, args.threshold)
    else:
        print("==================================================")
        print(" Welcome to the Interactive SMC Market Analyzer ")
        print("==================================================")
        print("Type a symbol name to analyze (e.g. EURUSD, GBPUSD, BTCUSD, XAUUSD).")
        print("Type 'exit' or 'quit' to close the program.")
        print("-" * 50)
        
        while True:
            try:
                sym_input = input("\nEnter symbol to analyze: ").strip()
                if not sym_input:
                    continue
                if sym_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                analyze_symbol_setups(sym_input, args.threshold)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
