import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

sys.path.append("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")
os.chdir("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, get_pip_multiplier
from src.main import get_active_setups

def check_m30():
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
            
    print(f"Active symbol: {active_symbol}")
    
    # Get latest M30 data
    df = fetch_historical_data(active_symbol, mt5.TIMEFRAME_M30, 400)
    df = detect_swing_points(df)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=active_symbol)
    
    # Find FVG setups
    print("\nSearch for FVG at 2026-06-04 01:30:00 in M30:")
    target_time = pd.to_datetime("2026-06-04 01:30:00")
    matching_rows = df[pd.to_datetime(df['time']) == target_time]
    if matching_rows.empty:
        print("Candle at 01:30:00 not found.")
    else:
        for idx in matching_rows.index:
            # Let's check if FVG was created on this candle or nearby candles
            # In smc_detector, FVG columns are populated on the 3rd candle of the pattern (candle 3, index i)
            # Let's inspect rows around idx to see which row has the FVG
            for check_idx in range(idx - 2, idx + 5):
                if check_idx < 0 or check_idx >= len(df):
                    continue
                row = df.iloc[check_idx]
                if pd.notna(row['FVG_Type']):
                    # Check mitigation
                    mitigated = False
                    fvg_top = row['FVG_Top']
                    fvg_bottom = row['FVG_Bottom']
                    mit_time = None
                    for j in range(check_idx + 1, len(df)):
                        if row['FVG_Type'] == 'BULLISH':
                            if df['Close'].iloc[j] < fvg_bottom:
                                mitigated = True
                                mit_time = df['time'].iloc[j]
                                break
                        else:
                            if df['Close'].iloc[j] > fvg_top:
                                mitigated = True
                                mit_time = df['time'].iloc[j]
                                break
                    print(f"Candle Time: {row['time']} | Index: {check_idx} | Type: {row['FVG_Type']} | Top: {row['FVG_Top']:.3f} | Bottom: {row['FVG_Bottom']:.3f} | Mitigated: {mitigated} (at {mit_time} if yes)")

            
    tick = mt5.symbol_info_tick(active_symbol)
    print(f"\nCurrent Market Price: Bid={tick.bid:.3f}, Ask={tick.ask:.3f}")
    
    mt5.shutdown()

if __name__ == "__main__":
    check_m30()
