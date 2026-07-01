import os
import sys
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_bpr, get_pip_multiplier
from src.main import get_active_setups
from src.scanner_worker import is_good_fvg

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
    symbol = "XAUUSD"
    df_h4 = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 250)
    df_h4 = detect_swing_points(df_h4)
    df_h4 = detect_structures(df_h4)
    df_h4 = detect_fvg_and_ob(df_h4, symbol=symbol)
    
    # Add ATR_14
    import numpy as np
    close_prev = df_h4['Close'].shift(1).fillna(df_h4['Open'])
    tr = np.maximum(
        df_h4['High'] - df_h4['Low'],
        np.maximum(
            np.abs(df_h4['High'] - close_prev),
            np.abs(df_h4['Low'] - close_prev)
        )
    )
    df_h4['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    setups = get_active_setups(df_h4)
    fvg_setups = [s for s in setups if s['option_name'].startswith('Option')]
    
    print("\n--- ACTIVE H4 FVGs AND QUALITY FILTERS ---")
    
    # We group by index
    seen_indices = set()
    for s in fvg_setups:
        idx = s['index']
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        
        # Check FVG type
        fvg_type = df_h4['FVG_Type'].iloc[idx]
        fvg_top = df_h4['FVG_Top'].iloc[idx]
        fvg_bottom = df_h4['FVG_Bottom'].iloc[idx]
        
        print(f"\nIndex: {idx} | Time: {df_h4['time'].iloc[idx]}")
        print(f"  Type: {fvg_type} | Top: {fvg_top:.3f} | Bottom: {fvg_bottom:.3f}")
        print(f"  Fvg Width: {s['fvg_width']:.3f} USD")
        
        # Run quality filter
        is_good, reason = is_good_fvg(df_h4, idx, s, symbol, {'H4': df_h4})
        print(f"  Quality Filter Result: {'PASSED' if is_good else 'FAILED'}")
        print(f"    Reason: {reason}")
        
    mt5.shutdown()

if __name__ == "__main__":
    main()
