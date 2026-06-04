import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob
from src.labeler import get_killzone
from src.inference import predict_setup_probability

def generate_synthetic_data(num_candles=100, seed=42) -> pd.DataFrame:
    """
    Generate synthetic candlestick data representing simulated market structures.
    This acts as a fallback when MT5 terminal is not active.
    """
    print(f"Generating synthetic XAUUSD data (seed={seed})...")
    np.random.seed(seed)
    time = pd.date_range(start="2026-06-01", periods=num_candles, freq="15min")
    
    # Generate a trend with clear structure: Bullish expansion, pullback, bearish reversal
    prices = [2330.0]
    for i in range(1, num_candles):
        if i < 40:
            # Bullish phase
            change = np.random.normal(0.5, 1.2)
        elif i < 70:
            # Bearish correction
            change = np.random.normal(-0.8, 1.0)
        else:
            # Reversal & consolidation
            change = np.random.normal(0.2, 1.5)
        prices.append(prices[-1] + change)
        
    df = pd.DataFrame(index=time)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time'}, inplace=True)
    
    df['Close'] = prices
    df['Open'] = df['Close'].shift(1).fillna(2329.0)
    
    # Create wicks
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.exponential(0.5, size=num_candles)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.exponential(0.5, size=num_candles)
    df['Volume'] = np.random.randint(100, 1000, size=num_candles)
    
    # Inject a couple of artificial extreme expansion candles to guarantee FVGs
    # Bullish FVG candle at index 20
    df.loc[19, 'High'] = 2341.0
    df.loc[19, 'Close'] = 2340.5
    df.loc[20, 'Open'] = 2340.5
    df.loc[20, 'High'] = 2351.0
    df.loc[20, 'Low'] = 2340.0
    df.loc[20, 'Close'] = 2350.0
    df.loc[21, 'Open'] = 2350.0
    df.loc[21, 'Low'] = 2345.0  # Candle 19 High < Candle 21 Low (2341.0 < 2345.0) -> Bullish FVG!
    
    # Inject a guaranteed active Bullish FVG near the end of the dataset
    if num_candles > 30:
        idx_base = num_candles - 5
        df.loc[idx_base - 2, 'High'] = 2360.0
        df.loc[idx_base - 2, 'Close'] = 2359.5
        df.loc[idx_base - 1, 'Open'] = 2359.5
        df.loc[idx_base - 1, 'High'] = 2375.0
        df.loc[idx_base - 1, 'Low'] = 2359.0
        df.loc[idx_base - 1, 'Close'] = 2374.0
        df.loc[idx_base, 'Open'] = 2374.0
        df.loc[idx_base, 'Low'] = 2365.0
        df.loc[idx_base, 'Close'] = 2370.0
        
        # Subsequent candles shouldn't close below FVG_Bottom (2360.0)
        for k in range(idx_base + 1, num_candles):
            df.loc[k, 'Open'] = 2370.0
            df.loc[k, 'High'] = 2373.0
            df.loc[k, 'Low'] = 2366.0
            df.loc[k, 'Close'] = 2368.0
            
    return df


def plot_smc_chart(df: pd.DataFrame, title: str = "XAUUSD M15 - SMC/ICT Analysis", active_setups=None):
    """
    Plot a premium, dark-themed TradingView style chart displaying candlesticks
    along with detected Swing Points, BOS, CHoCH, FVGs, and Order Blocks.
    """
    print("Plotting SMC structures...")
    # Setup dark theme style
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#131722')
    ax.set_facecolor('#131722')
    
    ax.grid(color='#2a2e39', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Set axis limits with margin
    ax.set_xlim(-1, len(df))
    price_min = df['Low'].min() - 2.0
    price_max = df['High'].max() + 2.0
    ax.set_ylim(price_min, price_max)
    
    for idx, row in df.iterrows():
        # Draw wick
        ax.plot([idx, idx], [row['Low'], row['High']], color='#787b86', linewidth=1.2, zorder=2)
        
        # Draw body
        is_bullish = row['Close'] >= row['Open']
        body_color = '#089981' if is_bullish else '#f23645'
        bottom = min(row['Open'], row['Close'])
        height = abs(row['Close'] - row['Open'])
        if height == 0:
            height = 0.05
            
        rect = patches.Rectangle((idx - 0.3, bottom), 0.6, height, 
                                 facecolor=body_color, edgecolor=body_color, zorder=3)
        ax.add_patch(rect)
        
        # 1. Swing High & Swing Low markers
        if not pd.isna(row['Swing_High']):
            ax.scatter(idx, row['Swing_High'] + 0.4, color='#f23645', marker='v', s=45, zorder=4)
            ax.text(idx, row['Swing_High'] + 0.8, f"SH", color='#f97316', fontsize=7, ha='center', fontweight='bold')
            
        if not pd.isna(row['Swing_Low']):
            ax.scatter(idx, row['Swing_Low'] - 0.4, color='#089981', marker='^', s=45, zorder=4)
            ax.text(idx, row['Swing_Low'] - 1.2, f"SL", color='#10b981', fontsize=7, ha='center', fontweight='bold')
            
        # 2. Break of Structure (BOS)
        if not pd.isna(row['BOS']):
            # Draw dotted line extending slightly back
            start_x = max(0, idx - 8)
            ax.plot([start_x, idx], [row['BOS'], row['BOS']], color='#f59e0b', linestyle=':', linewidth=1.5, zorder=1)
            ax.text(idx, row['BOS'] + 0.2, "BOS", color='#f59e0b', fontsize=8, fontweight='bold', ha='right')
            
        # 3. Change of Character (CHoCH)
        if not pd.isna(row['CHoCH']):
            # Draw dashed line
            start_x = max(0, idx - 8)
            ax.plot([start_x, idx], [row['CHoCH'], row['CHoCH']], color='#8b5cf6', linestyle='--', linewidth=1.5, zorder=1)
            ax.text(idx, row['CHoCH'] + 0.2, "CHoCH", color='#8b5cf6', fontsize=8, fontweight='bold', ha='right')
            
        # 4. Fair Value Gaps (FVG) Shading
        if 'FVG_Type' in df.columns and row['FVG_Type'] is not None:
            is_bull_fvg = row['FVG_Type'] == 'BULLISH'
            fvg_color = '#10b981' if is_bull_fvg else '#ef4444'
            # Render a shaded rectangle spanning from candle i-2 to i
            rect_fvg = patches.Rectangle((idx - 2, row['FVG_Bottom']), 2, row['FVG_Top'] - row['FVG_Bottom'],
                                         facecolor=fvg_color, alpha=0.15, edgecolor='none', zorder=1)
            ax.add_patch(rect_fvg)
            # Add small text label inside FVG
            label_y = (row['FVG_Top'] + row['FVG_Bottom']) / 2
            ax.text(idx - 1, label_y, "FVG", color=fvg_color, fontsize=7, alpha=0.7, ha='center', va='center')
            
        # 5. Order Blocks (OB) Shading
        if 'OB_Type' in df.columns and row['OB_Type'] is not None:
            is_bull_ob = row['OB_Type'] == 'BULLISH'
            ob_color = '#3b82f6' if is_bull_ob else '#d97706'
            label = "Bull OB" if is_bull_ob else "Bear OB"
            # Draw OB band extending forward
            width = min(15, len(df) - idx)
            rect_ob = patches.Rectangle((idx, row['OB_Bottom']), width, row['OB_Top'] - row['OB_Bottom'],
                                        facecolor=ob_color, alpha=0.1, edgecolor=ob_color, 
                                        linestyle='--', linewidth=0.8, zorder=1)
            ax.add_patch(rect_ob)
            
            # Show if OB is mitigated
            if row['OB_Mitigated']:
                label += " (Mitigated)"
                text_color = '#9ca3af'
            else:
                text_color = ob_color
                
            ax.text(idx + 0.2, row['OB_Top'] + 0.2, label, color=text_color, fontsize=7, fontweight='bold')
            
    # Chart titles and styling
    ax.set_title(title, color='#e5e7eb', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(colors='#9ca3af', labelsize=9)
    
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#2a2e39')
    ax.spines['left'].set_color('#2a2e39')
    
    ax.set_ylabel("Price (USD)", color='#9ca3af', fontsize=10, labelpad=10)
    ax.set_xlabel("Bars (Timeline)", color='#9ca3af', fontsize=10, labelpad=10)
    
    # 6. Plot Active ML-Filtered Signals
    if active_setups:
        # Draw high confidence signals on the chart
        for setup in active_setups:
            if setup['status'] == "HIGH CONFIDENCE SIGNAL":
                idx = setup['index']
                line_color = '#089981' if setup['direction'] == 1 else '#f23645'
                
                # Draw vertical dotted line at the detection candle
                ax.axvline(x=idx, color=line_color, linestyle=':', alpha=0.7, linewidth=1.5)
                
                # Draw horizontal lines for Entry, SL, and TP
                end_x = min(idx + 15, len(df) - 1)
                x_range = [idx, end_x]
                
                # Entry (Blue-ish)
                ax.plot(x_range, [setup['entry_price'], setup['entry_price']], color='#2962ff', linestyle='-', linewidth=1.5, zorder=4)
                # SL (Red)
                ax.plot(x_range, [setup['sl_price'], setup['sl_price']], color='#f23645', linestyle='--', linewidth=1.2, zorder=4)
                # TP (Green)
                ax.plot(x_range, [setup['tp_price'], setup['tp_price']], color='#089981', linestyle='--', linewidth=1.2, zorder=4)
                
                # Annotation labels on the chart
                setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
                dir_label = "BUY" if setup['direction'] == 1 else "SELL"
                ax.text(idx + 0.5, setup['entry_price'] + 0.2, f"★ ML {dir_label} {setup_name} ({setup['probability']:.1%})",
                        color='white', fontsize=7.5, fontweight='bold', 
                        bbox=dict(facecolor='#2962ff', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'), zorder=5)

        # Draw a summary box of all active signals at the bottom left
        text_lines = ["ACTIVE SMC SIGNALS (ML FILTERED)"]
        text_lines.append("-" * 35)
        for setup in active_setups:
            setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
            dir_label = "BULL" if setup['direction'] == 1 else "BEAR"
            is_high = setup['status'] == "HIGH CONFIDENCE SIGNAL"
            status_text = "PASS" if is_high else "FILTERED"
            text_lines.append(f"{setup_name} {dir_label} | Prob: {setup['probability']:.1%} | {status_text}")
            
        if len(active_setups) == 0:
            text_lines.append("No active setups found.")
            
        text_str = "\n".join(text_lines)
        ax.text(0.02, 0.05, text_str, transform=ax.transAxes, color='#e5e7eb', fontsize=8, fontfamily='monospace',
                bbox=dict(facecolor='#1e222d', alpha=0.85, edgecolor='#2a2e39', boxstyle='round,pad=0.5'), zorder=5)
                
    plt.tight_layout()
    output_filename = "xauusd_smc_analysis.png"
    plt.savefig(output_filename, dpi=180, facecolor='#131722')
    plt.close()
    print(f"SMC analysis visualization successfully saved as '{output_filename}'")
    
def get_active_setups(df: pd.DataFrame, buffer: float = 0.5):
    """
    Identify active (unmitigated) SMC setups (OBs and FVGs) from the detected structures.
    """
    active_setups = []
    
    # 1. OB Setups
    for i in range(len(df)):
        ob_type = df['OB_Type'].iloc[i]
        if pd.notna(ob_type) and ob_type is not None:
            # Check if it remains unmitigated (OB_Mitigated is False)
            if not df['OB_Mitigated'].iloc[i]:
                ob_top = df['OB_Top'].iloc[i]
                ob_bottom = df['OB_Bottom'].iloc[i]
                t_val = pd.to_datetime(df['time'].iloc[i])
                hour_val = int(t_val.hour)
                day_of_week_val = int(t_val.dayofweek)
                trend_val = int(df['Trend'].iloc[i])
                killzone_val = get_killzone(hour_val)
                atr_val = df['ATR_14'].iloc[i] if 'ATR_14' in df.columns else 1.0
                if pd.isna(atr_val):
                    atr_val = 1.0
                
                if ob_type == 'BULLISH':
                    direction = 1
                    entry = ob_top
                    sl = ob_bottom - buffer
                    tp = entry + (entry - sl) * 2
                else:
                    direction = -1
                    entry = ob_bottom
                    sl = ob_top + buffer
                    tp = entry - (sl - entry) * 2
                
                risk_pips = (entry - sl) if direction == 1 else (sl - entry)
                
                active_setups.append({
                    'index': i,
                    'time': df['time'].iloc[i],
                    'hour': hour_val,
                    'day_of_week': day_of_week_val,
                    'setup_type': 1,  # OB
                    'direction': direction,
                    'entry_price': entry,
                    'sl_price': sl,
                    'tp_price': tp,
                    'risk_pips': risk_pips,
                    'atr_14': atr_val,
                    'trend': trend_val,
                    'killzone': killzone_val
                })
                
    # 2. FVG Setups
    for i in range(len(df)):
        fvg_type = df['FVG_Type'].iloc[i]
        if pd.notna(fvg_type) and fvg_type is not None:
            fvg_top = df['FVG_Top'].iloc[i]
            fvg_bottom = df['FVG_Bottom'].iloc[i]
            
            # Check mitigation from i+1 to end of df
            mitigated = False
            for j in range(i + 1, len(df)):
                if fvg_type == 'BULLISH':
                    if df['Low'].iloc[j] <= fvg_top:
                        mitigated = True
                        break
                elif fvg_type == 'BEARISH':
                    if df['High'].iloc[j] >= fvg_bottom:
                        mitigated = True
                        break
            
            if not mitigated:
                t_val = pd.to_datetime(df['time'].iloc[i])
                hour_val = int(t_val.hour)
                day_of_week_val = int(t_val.dayofweek)
                trend_val = int(df['Trend'].iloc[i])
                killzone_val = get_killzone(hour_val)
                atr_val = df['ATR_14'].iloc[i] if 'ATR_14' in df.columns else 1.0
                if pd.isna(atr_val):
                    atr_val = 1.0
                    
                if fvg_type == 'BULLISH':
                    direction = 1
                    entry = fvg_top
                    sl = fvg_bottom - buffer
                    tp = entry + (entry - sl) * 2
                else:
                    direction = -1
                    entry = fvg_bottom
                    sl = fvg_top + buffer
                    tp = entry - (sl - entry) * 2
                
                risk_pips = (entry - sl) if direction == 1 else (sl - entry)
                
                active_setups.append({
                    'index': i,
                    'time': df['time'].iloc[i],
                    'hour': hour_val,
                    'day_of_week': day_of_week_val,
                    'setup_type': 0,  # FVG
                    'direction': direction,
                    'entry_price': entry,
                    'sl_price': sl,
                    'tp_price': tp,
                    'risk_pips': risk_pips,
                    'atr_14': atr_val,
                    'trend': trend_val,
                    'killzone': killzone_val
                })
                
    return active_setups

def extract_active_htf_fvgs(df: pd.DataFrame) -> list:
    """
    Extract active (unmitigated) FVGs on a higher timeframe (HTF).
    A Bullish HTF FVG is active if subsequent price has not closed below FVG_Bottom.
    A Bearish HTF FVG is active if subsequent price has not closed above FVG_Top.
    """
    active_fvgs = []
    if 'FVG_Type' not in df.columns:
        return active_fvgs
        
    for i in range(len(df)):
        fvg_type = df['FVG_Type'].iloc[i]
        if pd.notna(fvg_type) and fvg_type is not None:
            fvg_top = df['FVG_Top'].iloc[i]
            fvg_bottom = df['FVG_Bottom'].iloc[i]
            
            # Check mitigation from i+1 to end of df
            mitigated = False
            for j in range(i + 1, len(df)):
                if fvg_type == 'BULLISH':
                    if df['Close'].iloc[j] < fvg_bottom:
                        mitigated = True
                        break
                elif fvg_type == 'BEARISH':
                    if df['Close'].iloc[j] > fvg_top:
                        mitigated = True
                        break
            
            if not mitigated:
                active_fvgs.append({
                    'index': i,
                    'time': df['time'].iloc[i],
                    'type': fvg_type,
                    'top': fvg_top,
                    'bottom': fvg_bottom
                })
    return active_fvgs

def main():
    print("=== SMC/ICT AUTO-ANALYZER CORE ENGINE ===")
    
    # Try to connect to MT5 Exness terminal
    mt5_active = False
    symbol = "XAUUSD"
    timeframes_data = {}
    
    try:
        import MetaTrader5 as mt5
        if connect_mt5():
            print("Connected successfully to MetaTrader 5!")
            print(f"Fetching multi-timeframe data for {symbol} from MT5...")
            timeframes_data['D1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_D1, 50)
            timeframes_data['H4'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H4, 100)
            timeframes_data['H1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_H1, 150)
            timeframes_data['M30'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M30, 200)
            timeframes_data['M15'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M15, 200)
            mt5_active = True
            mt5.shutdown()
        else:
            print("MetaTrader 5 terminal not running. Switching to fallback mode.")
    except Exception as e:
        print(f"Could not connect to MT5. Error: {e}")
        print("Switching to fallback mode.")
        
    if not mt5_active:
        # Fallback to simulated data
        print("Generating synthetic multi-timeframe data...")
        timeframes_data['D1'] = generate_synthetic_data(50, seed=42)
        timeframes_data['H4'] = generate_synthetic_data(100, seed=43)
        timeframes_data['H1'] = generate_synthetic_data(150, seed=44)
        timeframes_data['M30'] = generate_synthetic_data(200, seed=45)
        timeframes_data['M15'] = generate_synthetic_data(200, seed=46)
        
    # Run SMC detection algorithms on all timeframes
    print("Running SMC structure detection algorithms on all timeframes...")
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        df_tf = detect_swing_points(df_tf, window=5)
        df_tf = detect_structures(df_tf)
        df_tf = detect_fvg_and_ob(df_tf, symbol=symbol)
        
        # Calculate ATR_14
        close_prev = df_tf['Close'].shift(1).fillna(df_tf['Open'])
        tr = np.maximum(
            df_tf['High'] - df_tf['Low'],
            np.maximum(
                np.abs(df_tf['High'] - close_prev),
                np.abs(df_tf['Low'] - close_prev)
            )
        )
        df_tf['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
        timeframes_data[tf_name] = df_tf
        
    # Extract active HTF FVGs (from H1, H4, and D1)
    active_htf_fvgs = []
    for tf_name in ['H1', 'H4', 'D1']:
        tf_fvgs = extract_active_htf_fvgs(timeframes_data[tf_name])
        for fvg in tf_fvgs:
            fvg['timeframe'] = tf_name
            active_htf_fvgs.append(fvg)
            
    print(f"Detected {len(active_htf_fvgs)} active Higher Timeframe (HTF) FVGs.")
    
    # Identify active setups on lower timeframes (M15 and M30)
    active_setups_m15 = get_active_setups(timeframes_data['M15'])
    for s in active_setups_m15:
        s['timeframe'] = 'M15'
        
    active_setups_m30 = get_active_setups(timeframes_data['M30'])
    for s in active_setups_m30:
        s['timeframe'] = 'M30'
        
    all_ltf_setups = active_setups_m15 + active_setups_m30
    
    # Check if the setup entry price falls inside any active HTF FVG of the same direction
    for setup in all_ltf_setups:
        setup['htf_prioritized'] = False
        setup['matching_htf_fvgs'] = []
        
        for htf_fvg in active_htf_fvgs:
            # Check same direction
            is_same_direction = (setup['direction'] == 1 and htf_fvg['type'] == 'BULLISH') or \
                                (setup['direction'] == -1 and htf_fvg['type'] == 'BEARISH')
            if is_same_direction:
                entry = setup['entry_price']
                if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                    setup['htf_prioritized'] = True
                    setup['matching_htf_fvgs'].append(htf_fvg)
                    
    # Query model predictions for each setup
    filtered_setups_with_prob = []
    for setup in all_ltf_setups:
        features = {
            'hour': setup['hour'],
            'day_of_week': setup['day_of_week'],
            'setup_type': setup['setup_type'],
            'direction': setup['direction'],
            'entry_price': setup['entry_price'],
            'sl_price': setup['sl_price'],
            'tp_price': setup['tp_price'],
            'risk_pips': setup['risk_pips'],
            'atr_14': setup['atr_14'],
            'trend': setup['trend'],
            'killzone': setup['killzone']
        }
        try:
            prob = predict_setup_probability(features)
            setup['probability'] = prob
            setup['status'] = "HIGH CONFIDENCE SIGNAL" if prob >= 0.70 else "FILTERED (Low Confidence)"
        except Exception as e:
            print(f"Error predicting setup probability: {e}")
            setup['probability'] = 0.0
            setup['status'] = "ERROR"
            
        filtered_setups_with_prob.append(setup)
        
    # Print results in terminal
    print("\n" + "="*95)
    print("                    ACTIVE SMC TRADE SIGNALS & ML FILTERING")
    print("="*95)
    print(f"{'Time':<20} | {'TF':<4} | {'Type':<5} | {'Dir':<8} | {'Entry':<8} | {'SL':<8} | {'TP':<8} | {'Win Prob':<8} | {'HTF Prior':<9} | Status")
    print("-"*95)
    for setup in filtered_setups_with_prob:
        setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
        dir_name = "Bullish" if setup['direction'] == 1 else "Bearish"
        prior_str = "YES" if setup['htf_prioritized'] else "NO"
        print(f"{str(setup['time']):<20} | {setup['timeframe']:<4} | {setup_name:<5} | {dir_name:<8} | {setup['entry_price']:.3f} | {setup['sl_price']:.3f} | {setup['tp_price']:.3f} | {setup['probability']:.2%} | {prior_str:<9} | {setup['status']}")
    print("="*95 + "\n")
    
    # Print prioritized setups clearly
    prioritized_setups = [s for s in filtered_setups_with_prob if s['htf_prioritized']]
    if prioritized_setups:
        print("*"*95)
        print("                    PRIORITIZED MULTI-TIMEFRAME (HTF) SETUPS")
        print("*"*95)
        for setup in prioritized_setups:
            setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
            dir_name = "Bullish" if setup['direction'] == 1 else "Bearish"
            matching_desc = ", ".join([f"{f['timeframe']} FVG ({f['bottom']:.3f}-{f['top']:.3f})" for f in setup['matching_htf_fvgs']])
            print(f"* {setup['timeframe']} {dir_name} {setup_name} at {setup['entry_price']:.3f} matched HTF: {matching_desc} (Win Prob: {setup['probability']:.2%})")
        print("*"*95 + "\n")
        
    # Display statistics
    df_m15 = timeframes_data['M15']
    num_bos = df_m15['BOS'].notna().sum()
    num_choch = df_m15['CHoCH'].notna().sum()
    num_fvg = df_m15['FVG_Type'].notna().sum() if 'FVG_Type' in df_m15.columns else 0
    num_ob = df_m15['OB_Type'].notna().sum() if 'OB_Type' in df_m15.columns else 0
    
    print("\n--- M15 Detection Summary ---")
    print(f"Total Candles Analyzed: {len(df_m15)}")
    print(f"Break of Structure (BOS) signals: {num_bos}")
    print(f"Change of Character (CHoCH) signals: {num_choch}")
    print(f"Fair Value Gaps (FVG) detected: {num_fvg}")
    print(f"Order Blocks (OB) created: {num_ob}")
    print(f"Active Setups Found: {len(filtered_setups_with_prob)}")
    print(f"High Confidence Signals: {sum(1 for s in filtered_setups_with_prob if s['status'] == 'HIGH CONFIDENCE SIGNAL')}")
    print("-------------------------\n")
    
    # Generate visualization (only for M15 setups to avoid index mismatch on M15 chart)
    chart_title = "XAUUSD M15 - Exness Live Data" if mt5_active else "XAUUSD M15 - Simulated Market Structure"
    m15_filtered_setups = [s for s in filtered_setups_with_prob if s.get('timeframe') == 'M15']
    plot_smc_chart(df_m15, title=chart_title, active_setups=m15_filtered_setups)
    print("Phase 2 complete! Signal Filtering Engine & Self-Learning Loop successfully integrated with MTF.")

if __name__ == "__main__":
    main()
