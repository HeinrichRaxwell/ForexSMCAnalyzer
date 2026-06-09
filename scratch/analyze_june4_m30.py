import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
import MetaTrader5 as mt5

# Add project root to python path
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
from src.main import get_active_setups
from src.inference import predict_setup_probability

def main():
    if not connect_mt5():
        print("Error: Could not connect to MT5.")
        return
        
    symbol = "XAUUSDm"
    timeframe = mt5.TIMEFRAME_M30
    num_candles = 500 # 250 hours, covers ~10 days
    
    print(f"Fetching {num_candles} candles of {symbol} from MT5...")
    df = fetch_historical_data(symbol, timeframe, num_candles)
    if df is None or df.empty:
        print("Error: Failed to fetch data from MT5.")
        return
        
    print(f"Fetched {len(df)} rows.")
    
    # Run SMC detectors
    df = detect_swing_points(df, window=5)
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
    
    # Convert time to datetime and filter for June 4, 2026
    df['time'] = pd.to_datetime(df['time'])
    
    # Let's inspect all candles on June 4, 2026
    start_time = pd.to_datetime("2026-06-04 15:00:00")
    end_time = pd.to_datetime("2026-06-05 06:00:00")
    
    june4_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    print("\n=== Live Candles around June 4th 15:00 to June 5th 06:00 ===")
    print(june4_df[['time', 'Open', 'High', 'Low', 'Close', 'IC_Type', 'IC_Top', 'IC_Bottom', 'IC_Mitigated']])
    
    # Let's check get_active_setups output for M30 timeframe on the full df
    setups = get_active_setups(df)
    
    print("\n=== Active IC (Indecision Candle) Setups in M30 ===")
    ic_setups = [s for s in setups if "IC" in s.get('option_name', '')]
    if not ic_setups:
        print("No active IC setups found in get_active_setups.")
    else:
        for s in ic_setups:
            setup_time = pd.to_datetime(s['time'])
            if setup_time >= start_time and setup_time <= end_time:
                print(f"Found IC setup at {s['time']}: {s['option_name']}, Dir: {s['direction']}, Entry: {s['entry_price']:.3f}, SL: {s['sl_price']:.3f}, TP: {s['tp_price']:.3f}")
                
    # Check if there are any setups around June 4th that were evaluated
    print("\n=== Running Model Evaluation for setups around June 4th ===")
    for s in setups:
        setup_time = pd.to_datetime(s['time'])
        if setup_time >= start_time and setup_time <= end_time:
            # Extract features matching model trainer
            features = {
                'timeframe': 30, # M30
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
                print(f"Setup at {s['time']} ({s['option_name']}):")
                print(f"  Direction: {'BUY' if s['direction'] == 1 else 'SELL'}, Entry: {s['entry_price']:.3f}, SL: {s['sl_price']:.3f}")
                print(f"  Features: Risk Pips: {s['risk_pips']:.3f}, Trend: {s['trend']}, Killzone: {s['killzone']}")
                print(f"  Confidence: {prob:.2%}")
                if prob >= 0.70:
                    print("  Status: >>> HIGH CONFIDENCE - SHOULD ENTRY <<<")
                else:
                    print("  Status: Filtered (Low Confidence)")
            except Exception as e:
                print(f"  Error predicting probability: {e}")
                
    # Search sent_signals.json
    sent_path = "data/sent_signals.json"
    if os.path.exists(sent_path):
        try:
            with open(sent_path, "r") as f:
                sent_data = json.load(f)
            print("\n=== Matching Signatures in sent_signals.json ===")
            matched = False
            for key, val in sent_data.items():
                if "2026-06-04" in key or "2026-06-04" in val.get('time_sent', ''):
                    print(f"Key: {key}")
                    print(f"  Time sent: {val.get('time_sent')}, Timeframe: {val.get('timeframe')}, Direction: {val.get('direction')}")
                    matched = True
            if not matched:
                print("No signals found in sent_signals.json for June 4, 2026.")
        except Exception as e:
            print(f"Error reading sent_signals.json: {e}")

if __name__ == '__main__':
    main()
