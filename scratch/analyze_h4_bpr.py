import os
import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# Add project root to python path
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
    print(f"Active broker symbol: {broker_symbol}")
    
    tick = mt5.symbol_info_tick(broker_symbol)
    if tick is not None:
        print(f"Current Market Price: Bid={tick.bid}, Ask={tick.ask}, Last={tick.last}")
    else:
        print("Failed to get symbol tick info")
        
    # Fetch data
    print("Fetching H4 historical data...")
    df_h4 = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    print(f"Fetched {len(df_h4)} H4 candles. Last candle time: {df_h4['time'].iloc[-1]}")
    
    # Process indicators
    df_h4 = detect_swing_points(df_h4)
    df_h4 = detect_structures(df_h4)
    df_h4 = detect_fvg_and_ob(df_h4, symbol=symbol)
    df_h4 = detect_snr_and_swapzones(df_h4, symbol=symbol)
    df_h4 = detect_bpr(df_h4, symbol=symbol)
    df_h4 = detect_indecision_candles(df_h4, symbol=symbol)
    
    close_prev = df_h4['Close'].shift(1).fillna(df_h4['Open'])
    tr = np.maximum(
        df_h4['High'] - df_h4['Low'],
        np.maximum(
            np.abs(df_h4['High'] - close_prev),
            np.abs(df_h4['Low'] - close_prev)
        )
    )
    df_h4['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    print("\n--- DETECTED BPRs ON H4 ---")
    bpr_mask = df_h4['BPR_Type'].notna()
    bprs_found = df_h4[bpr_mask]
    print(f"Found {len(bprs_found)} BPR setups historically in this range.")
    
    for idx, row in bprs_found.iterrows():
        print(f"\nIndex: {idx} | Time: {row['time']}")
        print(f"  Type: {row['BPR_Type']} | Top: {row['BPR_Top']} | Bottom: {row['BPR_Bottom']}")
        print(f"  Fibo 0.5: {row['BPR_Fibo_0.5']} | Fibo 0.618: {row['BPR_Fibo_0.618']}")
        print(f"  Mitigated: {row['BPR_Mitigated']}")
        
    # Get active setups from H4
    h4_setups = get_active_setups(df_h4)
    h4_bpr_setups = [s for s in h4_setups if 'BPR' in s['option_name']]
    print(f"\nFound {len(h4_bpr_setups)} active (unmitigated) BPR setups in H4 data loader.")
    
    for s in h4_bpr_setups:
        print(f"\nSetup: {s['option_name']}")
        print(f"  Time: {s['time']} | Index: {s['index']}")
        print(f"  Direction: {s['direction']} (1=Bullish, -1=Bearish)")
        print(f"  Entry: {s['entry_price']} | SL: {s['sl_price']} | TP: {s['tp_price']}")
        
        # Calculate features and predict probability
        features = {
            'timeframe': 240, # H4
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
            print(f"  Win Probability: {prob:.2%}")
        except Exception as e:
            print(f"  Probability calculation failed: {e}")
            prob = 0.0
            
        # Check distance
        if tick is not None:
            curr_price = tick.ask if s['direction'] == 1 else tick.bid
            diff = abs(s['entry_price'] - curr_price)
            print(f"  Current price distance: {diff:.2f} USD")
            
            # Check maximum distance allowed
            max_dist = 60.0 # H4 limit
            if diff > max_dist:
                print(f"  ❌ SKIPPED: Too far from market ({diff:.2f} USD > {max_dist} USD limit)")
            else:
                print(f"  ✅ Within distance limit (<={max_dist} USD)")
                
                # Check indicators
                is_valid_mkt, mkt_reason = validate_market_indicators(symbol, 'H4', s['direction'])
                print(f"  Market Indicators validation: {'PASSED' if is_valid_mkt else 'FAILED'}")
                print(f"    Reason: {mkt_reason}")
                
    # Also fetch and check D1 trend and other timeframes if needed
    print("\nShutting down MT5...")
    mt5.shutdown()

if __name__ == "__main__":
    main()
