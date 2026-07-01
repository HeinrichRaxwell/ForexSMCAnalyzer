import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import (
    detect_swing_points, 
    detect_structures, 
    detect_fvg_and_ob,
    detect_bpr
)
from src.labeler import get_killzone
from src.inference import predict_setup_probability

def main():
    if not connect_mt5():
        print("Error connecting to MT5")
        return
        
    df = fetch_historical_data("XAUUSDm", mt5.TIMEFRAME_H4, 250)
    df = detect_swing_points(df)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol="XAUUSD")
    df = detect_bpr(df, symbol="XAUUSD")
    
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
    df['time'] = pd.to_datetime(df['time'])
    
    idx_16 = df[df['time'] == pd.to_datetime("2026-05-28 16:00:00")].index[0]
    
    print(f"BPR Details at index {idx_16} ({df['time'].iloc[idx_16]}):")
    bpr_type = df['BPR_Type'].iloc[idx_16]
    bpr_top = float(df['BPR_Top'].iloc[idx_16])
    bpr_bottom = float(df['BPR_Bottom'].iloc[idx_16])
    bpr_fibo_0_5 = float(df['BPR_Fibo_0.5'].iloc[idx_16])
    bpr_fibo_0_618 = float(df['BPR_Fibo_0.618'].iloc[idx_16])
    bpr_fibo_0_0 = float(df['BPR_Fibo_0.0'].iloc[idx_16])
    bpr_sl = float(df['BPR_SL'].iloc[idx_16])
    bpr_mitigated = df['BPR_Mitigated'].iloc[idx_16]
    
    print(f"  Type: {bpr_type}")
    print(f"  Range: {bpr_bottom:.3f} to {bpr_top:.3f}")
    print(f"  Fibo 0.5: {bpr_fibo_0_5:.3f} | Fibo 0.618: {bpr_fibo_0_618:.3f}")
    print(f"  SL: {bpr_sl:.3f} | TP 1: {bpr_fibo_0_0:.3f}")
    print(f"  Mitigated in final df: {bpr_mitigated}")
    
    # Let's find exactly when it got mitigated
    mitigation_idx = None
    for j in range(idx_16 + 1, len(df)):
        low_val = df['Low'].iloc[j]
        high_val = df['High'].iloc[j]
        close_val = df['Close'].iloc[j]
        
        mitigated = False
        # BULLISH BPR check
        # Under new rule: mitigated if Low <= Fibo 0.382 of BPR range
        bpr_fibo_0_382 = bpr_top - 0.382 * (bpr_top - bpr_bottom)
        if low_val <= bpr_fibo_0_382:
            mitigated = True
        if close_val < bpr_bottom:
            mitigated = True
            
        if mitigated:
            mitigation_idx = j
            break
            
    if mitigation_idx is not None:
        print(f"  Mitigated at {df['time'].iloc[mitigation_idx]} (index {mitigation_idx})")
        print(f"    Mitigating candle: Low={df['Low'].iloc[mitigation_idx]:.3f}, High={df['High'].iloc[mitigation_idx]:.3f}")
    else:
        print("  Still unmitigated in final df.")
        
    # Evaluate model confidence
    t_val = df['time'].iloc[idx_16]
    hour_val = int(t_val.hour)
    day_of_week_val = int(t_val.dayofweek)
    trend_val = int(df['Trend'].iloc[idx_16])
    killzone_val = get_killzone(hour_val)
    atr_val = df['ATR_14'].iloc[idx_16]
    
    direction = 1 # BULLISH
    risk_pips_a = abs(bpr_fibo_0_5 - bpr_sl)
    features_a = {
        'timeframe': 240,
        'hour': hour_val,
        'day_of_week': day_of_week_val,
        'setup_type': 0, # FVG for BPR
        'direction': direction,
        'entry_price': bpr_fibo_0_5,
        'sl_price': bpr_sl,
        'tp_price': bpr_fibo_0_0,
        'risk_pips': risk_pips_a,
        'atr_14': atr_val,
        'trend': trend_val,
        'relative_risk': risk_pips_a / atr_val,
        'killzone': killzone_val,
        'fvg_width': abs(bpr_top - bpr_bottom),
        'relative_fvg_width': abs(bpr_top - bpr_bottom) / atr_val
    }
    
    risk_pips_b = abs(bpr_fibo_0_618 - bpr_sl)
    features_b = {
        'timeframe': 240,
        'hour': hour_val,
        'day_of_week': day_of_week_val,
        'setup_type': 0,
        'direction': direction,
        'entry_price': bpr_fibo_0_618,
        'sl_price': bpr_sl,
        'tp_price': bpr_fibo_0_0,
        'risk_pips': risk_pips_b,
        'atr_14': atr_val,
        'trend': trend_val,
        'relative_risk': risk_pips_b / atr_val,
        'killzone': killzone_val,
        'fvg_width': abs(bpr_top - bpr_bottom),
        'relative_fvg_width': abs(bpr_top - bpr_bottom) / atr_val
    }
    
    prob_a = predict_setup_probability(features_a)
    prob_b = predict_setup_probability(features_b)
    
    print(f"\nModel probabilities at creation:")
    print(f"  Option A (0.50): {prob_a:.2%}")
    print(f"  Option B (0.618): {prob_b:.2%}")
    print(f"  Max Confidence: {max(prob_a, prob_b):.2%}")

if __name__ == '__main__':
    main()
