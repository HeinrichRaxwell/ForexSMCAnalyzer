import os
import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob

def check_h1_sequence():
    if not connect_mt5():
        print("Failed to connect.")
        return
        
    symbol = "XAUUSD"
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    active_symbol = symbol
    for sym in symbols_to_try:
        if mt5.symbol_info(sym) is not None:
            active_symbol = sym
            break
            
    df = fetch_historical_data(active_symbol, mt5.TIMEFRAME_H1, 100)
    mt5.shutdown()
    
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=active_symbol)
    
    # Locate index 71 (28 bars ago)
    setup_idx = 71
    setup = df.iloc[setup_idx]
    
    fvg_type = setup['FVG_Type']
    if pd.isna(fvg_type) or fvg_type is None:
        print("No FVG found at index 71 in this run. Current DF length is:", len(df))
        return
        
    fibo_0_5 = setup['FVG_Fibo_0.5']
    fibo_0_618 = setup['FVG_Fibo_0.618']
    fibo_0_0 = setup['FVG_Fibo_0.0']
    sl_val = setup['FVG_SL']
    
    print(f"\n=== DETAILED SEQUENCE ANALYSIS FOR H1 FVG AT INDEX {setup_idx} ({setup['time']}) ===")
    print(f"Type: {fvg_type}")
    print(f"Entry 0.5 (Option A): {fibo_0_5:.3f}")
    print(f"Entry 0.618 (Option B): {fibo_0_618:.3f}")
    print(f"TP (Fibo 0.0): {fibo_0_0:.3f}")
    print(f"SL Level: {sl_val:.3f}")
    print("-" * 60)
    
    # Trace candles after index 71
    entered_0_5 = False
    entered_0_618 = False
    
    for i in range(setup_idx + 1, len(df)):
        candle = df.iloc[i]
        time_str = candle['time'].strftime('%Y-%m-%d %H:%M')
        low = candle['Low']
        high = candle['High']
        close = candle['Close']
        bars_after = i - setup_idx
        
        # Check Entry
        triggered_0_5 = False
        triggered_0_618 = False
        if not entered_0_5 and high >= fibo_0_5:
            entered_0_5 = True
            triggered_0_5 = True
        if not entered_0_618 and high >= fibo_0_618:
            entered_0_618 = True
            triggered_0_618 = True
            
        print(f"Bar +{bars_after} | {time_str} | Low={low:.3f} | High={high:.3f} | Close={close:.3f}")
        
        if triggered_0_5 or triggered_0_618:
            triggers = []
            if triggered_0_5: triggers.append("Fibo 0.5")
            if triggered_0_618: triggers.append("Fibo 0.618")
            print(f"  >>> [ENTRY TRIGGERED] {', '.join(triggers)}")
            
        # Check TP & SL if entered
        if entered_0_5 or entered_0_618:
            # Check if SL hit
            if high >= sl_val:
                print(f"  >>> [STOP LOSS HIT] at {high:.3f} (SL was {sl_val:.3f})")
                break
            # Check if TP hit
            if low <= fibo_0_0:
                pips_gain = abs(fibo_0_5 - fibo_0_0) * 10
                print(f"  >>> [TAKE PROFIT HIT] at {low:.3f} (TP was {fibo_0_0:.3f}). Profit: {pips_gain:.1f} pips!")
                break

if __name__ == "__main__":
    check_h1_sequence()
