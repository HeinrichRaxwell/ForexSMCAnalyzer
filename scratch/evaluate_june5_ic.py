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
    detect_snr_and_swapzones, 
    detect_bpr, 
    detect_indecision_candles
)
from src.labeler import get_killzone
from src.inference import predict_setup_probability

def main():
    if not connect_mt5():
        print("Error: Could not connect to MT5.")
        return
        
    symbol = "XAUUSDm"
    df = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 500)
    
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
    
    # Find index 492 (2026-06-05 01:30:00)
    df['time'] = pd.to_datetime(df['time'])
    target_time = pd.to_datetime("2026-06-05 01:30:00")
    
    row_idx = df[df['time'] == target_time].index
    if len(row_idx) == 0:
        print(f"Error: Candle at {target_time} not found.")
        return
        
    idx = row_idx[0]
    print(f"Found target candle at index {idx}, time: {df['time'].iloc[idx]}")
    
    # Check if IC is detected at this index
    ic_type = df['IC_Type'].iloc[idx]
    if ic_type is None or pd.isna(ic_type):
        print("No IC setup detected at this candle index.")
        return
        
    print(f"IC Type: {ic_type}")
    ic_top = float(df['IC_Top'].iloc[idx])
    ic_bottom = float(df['IC_Bottom'].iloc[idx])
    ic_fibo_0_5 = float(df['IC_Fibo_0.5'].iloc[idx])
    ic_fibo_0_618 = float(df['IC_Fibo_0.618'].iloc[idx])
    ic_fibo_0_0 = float(df['IC_Fibo_0.0'].iloc[idx])
    ic_sl = float(df['IC_SL'].iloc[idx])
    
    t_val = df['time'].iloc[idx]
    hour_val = int(t_val.hour)
    day_of_week_val = int(t_val.dayofweek)
    trend_val = int(df['Trend'].iloc[idx])
    killzone_val = get_killzone(hour_val)
    atr_val = df['ATR_14'].iloc[idx]
    
    print(f"IC Info:")
    print(f"  Range: {ic_bottom:.3f} to {ic_top:.3f}")
    print(f"  Entry 0.5: {ic_fibo_0_5:.3f}, Entry 0.618: {ic_fibo_0_618:.3f}")
    print(f"  SL: {ic_sl:.3f}, TP 1: {ic_fibo_0_0:.3f}")
    
    # Let's rebuild the features for both Option A and Option B
    direction = -1 # BEARISH
    risk_pips_a = abs(ic_fibo_0_5 - ic_sl)
    features_a = {
        'timeframe': 30,
        'hour': hour_val,
        'day_of_week': day_of_week_val,
        'setup_type': 1, # OB for IC
        'direction': direction,
        'entry_price': ic_fibo_0_5,
        'sl_price': ic_sl,
        'tp_price': ic_fibo_0_0,
        'risk_pips': risk_pips_a,
        'atr_14': atr_val,
        'trend': trend_val,
        'relative_risk': risk_pips_a / atr_val,
        'killzone': killzone_val,
        'fvg_width': 0.0,
        'relative_fvg_width': 0.0
    }
    
    risk_pips_b = abs(ic_fibo_0_618 - ic_sl)
    features_b = {
        'timeframe': 30,
        'hour': hour_val,
        'day_of_week': day_of_week_val,
        'setup_type': 1, # OB for IC
        'direction': direction,
        'entry_price': ic_fibo_0_618,
        'sl_price': ic_sl,
        'tp_price': ic_fibo_0_0,
        'risk_pips': risk_pips_b,
        'atr_14': atr_val,
        'trend': trend_val,
        'relative_risk': risk_pips_b / atr_val,
        'killzone': killzone_val,
        'fvg_width': 0.0,
        'relative_fvg_width': 0.0
    }
    
    prob_a = predict_setup_probability(features_a)
    prob_b = predict_setup_probability(features_b)
    
    print(f"\nModel probabilities at creation:")
    print(f"  Option A (0.50): {prob_a:.2%}")
    print(f"  Option B (0.618): {prob_b:.2%}")
    
    max_prob = max(prob_a, prob_b)
    print(f"  Max Confidence: {max_prob:.2%}")
    if max_prob >= 0.70:
        print("  Status: HIGH CONFIDENCE setup!")
    else:
        print("  Status: FILTERED due to low confidence (<0.70)")
        
    # Check if there were execution issues
    close_price = df['Close'].iloc[idx]
    print(f"\nBreakout candle closed at: {close_price:.3f}")
    dist_a = abs(ic_fibo_0_5 - close_price)
    dist_b = abs(ic_fibo_0_618 - close_price)
    print(f"Distance from Close to Entry A (0.5): {dist_a:.2f} USD")
    print(f"Distance from Close to Entry B (0.618): {dist_b:.2f} USD")
    
    # Check mitigation candle specifically
    if idx + 1 < len(df):
        next_time = df['time'].iloc[idx + 1]
        next_high = df['High'].iloc[idx + 1]
        print(f"\nNext candle at {next_time}: High={next_high:.3f}, Low={df['Low'].iloc[idx + 1]:.3f}")
        if next_high >= ic_bottom:
            print(f"  -> Mitigated on next candle because High ({next_high:.3f}) >= IC Bottom ({ic_bottom:.3f})")

if __name__ == '__main__':
    main()
