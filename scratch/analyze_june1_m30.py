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
    # Load from the M30 history file which has June 1st data
    csv_path = "data/historical_xauusdm_30.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Run SMC detectors
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol="XAUUSD")
    df = detect_snr_and_swapzones(df, symbol="XAUUSD")
    df = detect_bpr(df, symbol="XAUUSD")
    df = detect_indecision_candles(df, symbol="XAUUSD")
    
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
    
    # Filter for June 1, 2026
    df['time'] = pd.to_datetime(df['time'])
    
    start_time = pd.to_datetime("2026-06-01 00:00:00")
    end_time = pd.to_datetime("2026-06-01 08:00:00")
    
    june1_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    print("\n=== Candles around June 1st 01:00 ===")
    print(june1_df[['time', 'Open', 'High', 'Low', 'Close', 'IC_Type', 'IC_Top', 'IC_Bottom', 'IC_Mitigated']])
    
    # Let's find any setup created in this range
    for idx in june1_df.index:
        ic_type = df['IC_Type'].iloc[idx]
        if ic_type is not None and pd.notna(ic_type):
            print(f"\nFound IC Setup at {df['time'].iloc[idx]} (index {idx}):")
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
            
            direction = 1 if ic_type == 'BULLISH' else -1
            
            # Feature dicts
            features_a = {
                'timeframe': 30,
                'hour': hour_val,
                'day_of_week': day_of_week_val,
                'setup_type': 1, # OB for IC
                'direction': direction,
                'entry_price': ic_fibo_0_5,
                'sl_price': ic_sl,
                'tp_price': ic_fibo_0_0,
                'risk_pips': abs(ic_fibo_0_5 - ic_sl),
                'atr_14': atr_val,
                'trend': trend_val,
                'relative_risk': abs(ic_fibo_0_5 - ic_sl) / atr_val,
                'killzone': killzone_val,
                'fvg_width': 0.0,
                'relative_fvg_width': 0.0
            }
            
            features_b = {
                'timeframe': 30,
                'hour': hour_val,
                'day_of_week': day_of_week_val,
                'setup_type': 1,
                'direction': direction,
                'entry_price': ic_fibo_0_618,
                'sl_price': ic_sl,
                'tp_price': ic_fibo_0_0,
                'risk_pips': abs(ic_fibo_0_618 - ic_sl),
                'atr_14': atr_val,
                'trend': trend_val,
                'relative_risk': abs(ic_fibo_0_618 - ic_sl) / atr_val,
                'killzone': killzone_val,
                'fvg_width': 0.0,
                'relative_fvg_width': 0.0
            }
            
            prob_a = predict_setup_probability(features_a)
            prob_b = predict_setup_probability(features_b)
            
            print(f"  Type: {ic_type}")
            print(f"  Range: {ic_bottom:.3f} to {ic_top:.3f}")
            print(f"  Fibo 0.5: {ic_fibo_0_5:.3f} | Fibo 0.618: {ic_fibo_0_618:.3f}")
            print(f"  Model Confidence A: {prob_a:.2%}")
            print(f"  Model Confidence B: {prob_b:.2%}")
            
            # Let's find exactly which candle mitigated this IC
            mitigation_idx = None
            for j in range(idx + 1, len(df)):
                low_val = df['Low'].iloc[j]
                high_val = df['High'].iloc[j]
                close_val = df['Close'].iloc[j]
                
                mitigated = False
                if ic_type == 'BULLISH':
                    if low_val <= ic_top:
                        mitigated = True
                    if close_val < ic_bottom:
                        mitigated = True
                else:
                    if high_val >= ic_bottom:
                        mitigated = True
                    if close_val > ic_top:
                        mitigated = True
                        
                if mitigated:
                    mitigation_idx = j
                    break
            
            if mitigation_idx is not None:
                mit_time = df['time'].iloc[mitigation_idx]
                mit_low = df['Low'].iloc[mitigation_idx]
                mit_high = df['High'].iloc[mitigation_idx]
                print(f"  Mitigated at {mit_time} (index {mitigation_idx})")
                print(f"    Mitigating candle: Low={mit_low:.3f}, High={mit_high:.3f}")
            else:
                print("  Still unmitigated in the dataset.")

if __name__ == '__main__':
    main()
