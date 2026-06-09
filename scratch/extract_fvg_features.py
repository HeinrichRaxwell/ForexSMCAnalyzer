import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob
from src.inference import predict_setup_probability
from src.labeler import get_killzone

def main():
    if not connect_mt5():
        print("Failed to initialize MT5")
        return
        
    symbol = "XAUUSD"
    import MetaTrader5 as mt5
    
    m15_df = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 500)
    
    m15_df = detect_swing_points(m15_df, window=5)
    m15_df = detect_structures(m15_df)
    m15_df = detect_fvg_and_ob(m15_df, symbol=symbol)
    
    # Calculate ATR_14
    close_prev = m15_df['Close'].shift(1).fillna(m15_df['Open'])
    tr = np.maximum(
        m15_df['High'] - m15_df['Low'],
        np.maximum(
            np.abs(m15_df['High'] - close_prev),
            np.abs(m15_df['Low'] - close_prev)
        )
    )
    m15_df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    m15_df['time'] = pd.to_datetime(m15_df['time'])
    idx = m15_df[m15_df['time'] == '2026-06-05 00:15:00'].index[0]
    
    # Extract features for index 139 (which corresponds to 00:15:00)
    row = m15_df.iloc[idx]
    
    # The setup timeframe trend
    trend = row['Trend']
    
    # Entry options
    entry_05 = row['FVG_Fibo_0.5']
    entry_0618 = row['FVG_Fibo_0.618']
    sl = row['FVG_SL']
    tp = row['FVG_Fibo_0.0'] # TP1
    
    # Risk pips
    risk_pips_05 = abs(entry_05 - sl) / 0.1 # 0.1 for Gold
    risk_pips_0618 = abs(entry_0618 - sl) / 0.1
    
    # FVG width
    fvg_width = abs(row['FVG_Top'] - row['FVG_Bottom'])
    atr = row['ATR_14']
    
    killzone = get_killzone(row['time'].hour)
    
    features_05 = {
        'timeframe': 15,
        'hour': row['time'].hour,
        'day_of_week': row['time'].weekday(),
        'setup_type': 0, # FVG
        'direction': -1, # BEARISH
        'entry_price': entry_05,
        'sl_price': sl,
        'tp_price': tp,
        'risk_pips': risk_pips_05,
        'atr_14': atr,
        'trend': trend,
        'relative_risk': risk_pips_05 / atr if atr > 0 else 1.0,
        'killzone': killzone,
        'fvg_width': fvg_width,
        'relative_fvg_width': fvg_width / atr if atr > 0 else 0.4
    }
    
    features_0618 = features_05.copy()
    features_0618['entry_price'] = entry_0618
    features_0618['risk_pips'] = risk_pips_0618
    features_0618['relative_risk'] = risk_pips_0618 / atr if atr > 0 else 1.0
    
    prob_05 = predict_setup_probability(features_05)
    prob_0618 = predict_setup_probability(features_0618)
    
    print(f"\nSMC M15 Bearish FVG at {row['time']}:")
    print(f"FVG Zone: {row['FVG_Bottom']:.3f} - {row['FVG_Top']:.3f} (Width: {fvg_width:.3f} USD)")
    print(f"ATR 14: {atr:.3f} USD | Trend: {'BULL' if trend == 1 else 'BEAR'}")
    print(f"Option A (0.50): Entry = {entry_05:.3f} | SL = {sl:.3f} | TP1 = {tp:.3f}")
    print(f"  AI Success Probability: {prob_05:.2%}")
    print(f"Option B (0.618): Entry = {entry_0618:.3f} | SL = {sl:.3f} | TP1 = {tp:.3f}")
    print(f"  AI Success Probability: {prob_0618:.2%}")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
