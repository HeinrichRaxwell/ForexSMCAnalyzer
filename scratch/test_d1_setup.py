import os
import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob

def check_d1_setup():
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
            
    # Fetch 50 D1 bars
    df = fetch_historical_data(active_symbol, mt5.TIMEFRAME_D1, 50)
    mt5.shutdown()
    
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=active_symbol)
    
    print("\nScanning D1 candles for May 2026 FVGs:")
    for i in range(len(df)):
        candle = df.iloc[i]
        time_str = candle['time'].strftime('%Y-%m-%d')
        fvg_type = candle['FVG_Type']
        
        # Look for the setup around mid-May (specifically 2026-05-15)
        if pd.notna(fvg_type) and fvg_type is not None and "2026-05" in time_str:
            bars_ago = len(df) - 1 - i
            print(f"\n[D1 FVG FOUND] Index: {i} | Time: {time_str} | {bars_ago} bars ago")
            print(f"Type: {fvg_type}")
            print(f"FVG Zone boundaries: Bottom={candle['FVG_Bottom']:.3f}, Top={candle['FVG_Top']:.3f}")
            print(f"Fibo 1.0 (Candle 2 Boundary): {candle['FVG_Fibo_1.0']:.3f}")
            print(f"Fibo 0.5 (Midpoint Entry): {candle['FVG_Fibo_0.5']:.3f}")
            print(f"Fibo 0.618 (GP Entry): {candle['FVG_Fibo_0.618']:.3f}")
            print(f"Fibo 0.0 (Target/TP 1): {candle['FVG_Fibo_0.0']:.3f}")
            print(f"SL Level: {candle['FVG_SL']:.3f}")
            
            # Trace the outcome in subsequent D1 bars
            touched_0_5 = False
            touched_0_618 = False
            hit_tp = False
            hit_sl = False
            max_high_reached = -99999.0
            
            for j in range(i + 1, len(df)):
                c_j = df.iloc[j]
                low_j = c_j['Low']
                high_j = c_j['High']
                
                fibo_0_5 = candle['FVG_Fibo_0.5']
                fibo_0_618 = candle['FVG_Fibo_0.618']
                fibo_0_0 = candle['FVG_Fibo_0.0']
                sl_val = candle['FVG_SL']
                
                if fvg_type == 'BULLISH':
                    if low_j <= fibo_0_5:
                        touched_0_5 = True
                    if low_j <= fibo_0_618:
                        touched_0_618 = True
                    if low_j <= sl_val:
                        hit_sl = True
                        break
                    if high_j >= fibo_0_0:
                        hit_tp = True
                else: # BEARISH
                    if high_j >= fibo_0_5:
                        touched_0_5 = True
                    if high_j >= fibo_0_618:
                        touched_0_618 = True
                    if high_j > max_high_reached:
                        max_high_reached = high_j
                    if high_j >= sl_val:
                        hit_sl = True
                        break
                    if low_j <= fibo_0_0:
                        hit_tp = True
            
            print(f"Outcome analysis:")
            print(f"  - Touched Fibo 0.5? {touched_0_5}")
            print(f"  - Touched Fibo 0.618? {touched_0_618} (Max High reached during retest: {max_high_reached:.3f})")
            print(f"  - Reached target Fibo 0.0 (TP)? {hit_tp}")
            print(f"  - Reached SL? {hit_sl}")
            
            pips_gain = abs(candle['FVG_Fibo_0.5'] - candle['FVG_Fibo_0.0']) * 10
            print(f"  - Calculated Pip potential (0.5 to 0.0): {pips_gain:.1f} pips")

if __name__ == "__main__":
    check_d1_setup()
