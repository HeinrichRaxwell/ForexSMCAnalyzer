import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5, fetch_historical_data
from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_snr_and_swapzones, detect_bpr, detect_indecision_candles, detect_supply_demand_zones
from src.labeler import get_killzone
from src.inference import predict_setup_probability
from src.rejection_detector import detect_rejection_at_level, is_near_psychological_level
from src.indicators.knn_classifier import run_knn_classifier, calculate_knn_probability_at_bar
from src.indicators.volume_clusters import calculate_volume_clusters

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


def plot_smc_chart(df: pd.DataFrame, title: str = "XAUUSD - SMC/ICT Analysis", active_setups=None, output_filename="xauusd_smc_analysis.png"):
    """
    Plot a premium, dark-themed TradingView style chart displaying candlesticks
    along with detected Swing Points, BOS, CHoCH, FVGs, and Order Blocks.
    """
    print("Plotting SMC structures...")
    # Setup dark theme style
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#131722')
    ax.set_facecolor('#131722')
    
    ax.grid(color='#2a2e39', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Determine the visible range (zoom in on the last 60 candles for clarity)
    visible_candles = 60
    start_idx = max(0, len(df) - visible_candles)
    
    # Calculate price limits based only on the visible range
    df_visible = df.iloc[start_idx:]
    price_min = df_visible['Low'].min()
    price_max = df_visible['High'].max()
    price_range = price_max - price_min
    
    # Add a 10% margin to the top and bottom of the price limits
    price_min = price_min - 0.10 * price_range if price_range > 0 else price_min - 5.0
    price_max = price_max + 0.10 * price_range if price_range > 0 else price_max + 5.0
    
    # Set axis limits with margin on the right for labels
    ax.set_xlim(start_idx - 1, len(df) + 12)
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
            
            # Check mitigation from idx+1 to end of df (closing-basis check to allow retracement entries)
            mitigated = False
            for j in range(idx + 1, len(df)):
                if is_bull_fvg:
                    if df['Close'].iloc[j] < row['FVG_Bottom']:
                        mitigated = True
                        break
                else:
                    if df['Close'].iloc[j] > row['FVG_Top']:
                        mitigated = True
                        break
            
            if mitigated:
                # Mitigated: draw only small block at creation, low opacity, no label
                rect_fvg = patches.Rectangle((idx - 2, row['FVG_Bottom']), 2, row['FVG_Top'] - row['FVG_Bottom'],
                                             facecolor=fvg_color, alpha=0.04, edgecolor='none', zorder=1)
                ax.add_patch(rect_fvg)
            else:
                # Active (Unmitigated): extend all the way to the right edge with extra margin
                width = (len(df) + 4) - (idx - 2)
                rect_fvg = patches.Rectangle((idx - 2, row['FVG_Bottom']), width, row['FVG_Top'] - row['FVG_Bottom'],
                                             facecolor=fvg_color, alpha=0.10, edgecolor=fvg_color, linestyle=':', linewidth=0.5, zorder=1)
                ax.add_patch(rect_fvg)
                
                # Show Fibo levels (0.5 and 0.618 zones) on the chart for active FVG
                fibo_0_5_val = row.get('FVG_Fibo_0.5', np.nan)
                fibo_0_618_val = row.get('FVG_Fibo_0.618', np.nan)
                if pd.notna(fibo_0_5_val) and pd.notna(fibo_0_618_val):
                    rect_fib = patches.Rectangle((idx - 2, min(fibo_0_5_val, fibo_0_618_val)), width, abs(fibo_0_5_val - fibo_0_618_val),
                                                 facecolor='#eab308', alpha=0.08, edgecolor='none', zorder=1)
                    ax.add_patch(rect_fib)
                    
                # Add text label ONLY for active FVG
                label_y = (row['FVG_Top'] + row['FVG_Bottom']) / 2
                ax.text(idx + 1, label_y, "Active FVG", color=fvg_color, fontsize=6.5, alpha=0.7, ha='left', va='center', fontweight='bold')
            
        # 5. Order Blocks (OB) Shading
        if 'OB_Type' in df.columns and row['OB_Type'] is not None:
            is_bull_ob = row['OB_Type'] == 'BULLISH'
            ob_color = '#3b82f6' if is_bull_ob else '#d97706'
            
            if row['OB_Mitigated']:
                # Mitigated: draw only small block at creation, low opacity, no label
                rect_ob = patches.Rectangle((idx, row['OB_Bottom']), 2, row['OB_Top'] - row['OB_Bottom'],
                                            facecolor=ob_color, alpha=0.03, edgecolor='none', zorder=1)
                ax.add_patch(rect_ob)
            else:
                # Active (Unmitigated): extend all the way to the right with extra margin
                width = (len(df) + 4) - idx
                rect_ob = patches.Rectangle((idx, row['OB_Bottom']), width, row['OB_Top'] - row['OB_Bottom'],
                                            facecolor=ob_color, alpha=0.08, edgecolor=ob_color, 
                                            linestyle='--', linewidth=0.6, zorder=1)
                ax.add_patch(rect_ob)
                
                # Print label only for active OB
                label = "Bull OB" if is_bull_ob else "Bear OB"
                ax.text(idx + 0.5, row['OB_Top'] + 0.1, label, color=ob_color, fontsize=6.5, fontweight='bold', alpha=0.8)
            
        # 5.5. Balanced Price Ranges (BPR) Shading
        if 'BPR_Type' in df.columns and pd.notna(row.get('BPR_Type')) and row['BPR_Type'] is not None:
            bpr_color = '#d946ef'  # Magenta for BPR
            
            if row['BPR_Mitigated']:
                # Mitigated: draw small block, low opacity, no label
                rect_bpr = patches.Rectangle((idx - 2, row['BPR_Bottom']), 2, row['BPR_Top'] - row['BPR_Bottom'],
                                             facecolor=bpr_color, alpha=0.03, edgecolor='none', zorder=1)
                ax.add_patch(rect_bpr)
            else:
                # Active (Unmitigated): extend to the right with extra margin
                width = (len(df) + 4) - (idx - 2)
                rect_bpr = patches.Rectangle((idx - 2, row['BPR_Bottom']), width, row['BPR_Top'] - row['BPR_Bottom'],
                                             facecolor=bpr_color, alpha=0.08, edgecolor=bpr_color, 
                                             linestyle='-.', linewidth=0.6, zorder=1)
                ax.add_patch(rect_bpr)
                
                # Print label only for active BPR
                bpr_lbl = f"{'Bull' if row['BPR_Type'] == 'BULLISH' else 'Bear'} BPR"
                ax.text(idx + 0.5, row['BPR_Top'] + 0.1, bpr_lbl, color=bpr_color, fontsize=6.5, fontweight='bold', alpha=0.8)
            
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
                
                # Draw horizontal lines for Entry, SL, TP 1, and TP 2 extending to right margin
                end_x = len(df) + 3
                x_range = [idx, end_x]
                
                # Entry (Blue-ish)
                ax.plot(x_range, [setup['entry_price'], setup['entry_price']], color='#2962ff', linestyle='-', linewidth=1.5, zorder=4)
                ax.text(end_x + 0.4, setup['entry_price'], 'Entry', color='#2962ff', fontsize=8, fontweight='bold', va='center')
                
                # SL (Red)
                ax.plot(x_range, [setup['sl_price'], setup['sl_price']], color='#f23645', linestyle='--', linewidth=1.2, zorder=4)
                ax.text(end_x + 0.4, setup['sl_price'], 'SL', color='#f23645', fontsize=8, fontweight='bold', va='center')
                
                # TP 1 (Green)
                ax.plot(x_range, [setup['tp_price'], setup['tp_price']], color='#089981', linestyle='--', linewidth=1.2, zorder=4)
                ax.text(end_x + 0.4, setup['tp_price'], 'TP 1', color='#089981', fontsize=8, fontweight='bold', va='center')
                
                # TP 2 (Cyan/Teal)
                ax.plot(x_range, [setup['tp2_price'], setup['tp2_price']], color='#0ea5e9', linestyle='-.', linewidth=1.2, zorder=4)
                ax.text(end_x + 0.4, setup['tp2_price'], 'TP 2 (Dyn)', color='#0ea5e9', fontsize=8, fontweight='bold', va='center')

                # TradingView-style shaded position zones (Green for Profit, Red for Loss)
                rect_tp = patches.Rectangle((idx, min(setup['entry_price'], setup['tp_price'])), 
                                            end_x - idx, 
                                            abs(setup['tp_price'] - setup['entry_price']),
                                            facecolor='#089981', alpha=0.15, edgecolor='none', zorder=1)
                ax.add_patch(rect_tp)
                
                rect_sl = patches.Rectangle((idx, min(setup['entry_price'], setup['sl_price'])), 
                                            end_x - idx, 
                                            abs(setup['sl_price'] - setup['entry_price']),
                                            facecolor='#f23645', alpha=0.15, edgecolor='none', zorder=1)
                ax.add_patch(rect_sl)
                
                # If FVG setup, show the Fibo levels 0.5 and 0.618 and the zone
                if setup['setup_type'] == 0:
                    setup_idx = setup['index']
                    fibo_0_5_val = df['FVG_Fibo_0.5'].iloc[setup_idx]
                    fibo_0_618_val = df['FVG_Fibo_0.618'].iloc[setup_idx]
                    
                    if pd.notna(fibo_0_5_val) and pd.notna(fibo_0_618_val):
                        ax.plot(x_range, [fibo_0_5_val, fibo_0_5_val], color='#eab308', linestyle=':', linewidth=1.0, alpha=0.8, zorder=4)
                        ax.plot(x_range, [fibo_0_618_val, fibo_0_618_val], color='#eab308', linestyle=':', linewidth=1.0, alpha=0.8, zorder=4)
                        ax.fill_between(x_range, fibo_0_5_val, fibo_0_618_val, color='#eab308', alpha=0.08, zorder=3)
                        ax.text(idx + 1, fibo_0_5_val, 'Fibo 0.5', color='#eab308', fontsize=6.5, va='bottom')
                        ax.text(idx + 1, fibo_0_618_val, 'Fibo 0.618', color='#eab308', fontsize=6.5, va='bottom')
                
                # Annotation labels on the chart
                setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
                dir_label = "BUY" if setup['direction'] == 1 else "SELL"
                opt_info = f" ({setup['option_name']})" if setup['setup_type'] == 0 else ""
                ax.text(idx + 0.5, setup['entry_price'] + 0.2, f"★ ML {dir_label} {setup_name}{opt_info} ({setup['probability']:.1%})",
                        color='white', fontsize=7.5, fontweight='bold', 
                        bbox=dict(facecolor='#2962ff', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'), zorder=5)

        # Draw a summary box of all active signals at the bottom left
        text_lines = ["ACTIVE SMC SIGNALS (ML FILTERED)"]
        text_lines.append("-" * 50)
        for setup in active_setups:
            setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
            dir_label = "BULL" if setup['direction'] == 1 else "BEAR"
            is_high = setup['status'] == "HIGH CONFIDENCE SIGNAL"
            status_text = "PASS" if is_high else "FILTERED"
            rej_status = "Rej: Y" if setup.get('rejection_confirmed', False) else "Rej: N"
            
            opt_lbl = ""
            if setup['setup_type'] == 0:
                opt_lbl = " (Mid)" if "Midpoint" in setup['option_name'] else " (GP)"
                
            text_lines.append(f"{setup_name}{opt_lbl} {dir_label} | Prob: {setup['probability']:.1%} | {rej_status} | {status_text}")
            
        if len(active_setups) == 0:
            text_lines.append("No active setups found.")
            
        text_str = "\n".join(text_lines)
        ax.text(0.02, 0.05, text_str, transform=ax.transAxes, color='#e5e7eb', fontsize=8, fontfamily='monospace',
                bbox=dict(facecolor='#1e222d', alpha=0.85, edgecolor='#2a2e39', boxstyle='round,pad=0.5'), zorder=5)
                
    plt.tight_layout()
    plt.savefig(output_filename, dpi=180, facecolor='#131722')
    plt.close()
    print(f"SMC analysis visualization successfully saved as '{output_filename}'")

def find_dynamic_tp(df: pd.DataFrame, entry_price: float, direction: int) -> float:
    """
    Finds the dynamic Take Profit (TP 2) by searching for the first unmitigated
    opposite structure in the dataframe.
    """
    levels = []
    
    # 1. Gather all active (unmitigated) opposite structures
    for k in range(len(df)):
        # Check Order Blocks
        ob_type = df['OB_Type'].iloc[k]
        if pd.notna(ob_type) and ob_type is not None:
            if not df['OB_Mitigated'].iloc[k]:
                if direction == 1 and ob_type == 'BEARISH':
                    levels.append(float(df['OB_Bottom'].iloc[k]))
                elif direction == -1 and ob_type == 'BULLISH':
                    levels.append(float(df['OB_Top'].iloc[k]))
                    
        # Check Fair Value Gaps
        fvg_type = df['FVG_Type'].iloc[k]
        if pd.notna(fvg_type) and fvg_type is not None:
            # Check mitigation from k+1 to the end of the dataframe
            mitigated = False
            fvg_top = float(df['FVG_Top'].iloc[k])
            fvg_bottom = float(df['FVG_Bottom'].iloc[k])
            for j in range(k + 1, len(df)):
                if fvg_type == 'BULLISH':
                    if df['Close'].iloc[j] < fvg_bottom:
                        mitigated = True
                        break
                elif fvg_type == 'BEARISH':
                    if df['Close'].iloc[j] > fvg_top:
                        mitigated = True
                        break
            if not mitigated:
                if direction == 1 and fvg_type == 'BEARISH':
                    levels.append(fvg_bottom)
                elif direction == -1 and fvg_type == 'BULLISH':
                    levels.append(fvg_top)
                    
    # 2. Filter levels above/below entry_price based on direction
    tp_dynamic = None
    if direction == 1:
        # For Buy: lowest opposite level above entry_price
        valid_levels = [lvl for lvl in levels if lvl > entry_price]
        if valid_levels:
            tp_dynamic = min(valid_levels)
    else:
        # For Sell: highest opposite level below entry_price
        valid_levels = [lvl for lvl in levels if lvl < entry_price]
        if valid_levels:
            tp_dynamic = max(valid_levels)
            
    return tp_dynamic
    
def get_active_setups(df: pd.DataFrame, buffer: float = 0.5, symbol: str = "XAUUSD", tf_trends: dict = None, df_d1: pd.DataFrame = None):
    """
    Identify active (unmitigated) SMC setups (OBs and FVGs) from the detected structures.
    """
    df = df.copy()
    
    # Align daily pivots if df_d1 is provided
    if df_d1 is not None:
        from src.indicators.pivots import align_daily_pivots
        try:
            df = align_daily_pivots(df, df_d1)
        except Exception as e:
            print(f"Error aligning daily pivots in get_active_setups: {e}")

    # Calculate FLOOP Pro signals
    from src.indicators.floop import run_floop_pro
    htf_trend_series = None
    mtf_trends_list = None
    if tf_trends is not None:
        htf_trend_series = tf_trends.get('H4')
        if htf_trend_series is None:
            htf_trend_series = tf_trends.get('4h')
        mtf_trends_list = tf_trends
        
    floop_signals, floop_strengths, floop_trends = run_floop_pro(
        df,
        sensitivity=6,
        atr_len=14,
        atr_mult=0.8,
        use_adx=True,
        adx_thresh=20.0,
        use_chop=True,
        chop_thresh=61.8,
        use_cooldown=True,
        cooldown_len=5,
        ema_filter=False,
        htf_trend_series=htf_trend_series,
        mtf_trends=mtf_trends_list
    )
    df['floop_signal'] = floop_signals
    df['floop_strength'] = floop_strengths
    df['floop_trend'] = floop_trends
    
    active_setups = []
    
    # 1. OB Setups
    for i in range(len(df)):
        ob_type = df['OB_Type'].iloc[i]
        if pd.notna(ob_type) and ob_type is not None:
            # Check if it remains unmitigated (OB_Mitigated is False)
            if not df['OB_Mitigated'].iloc[i]:
                t_val = pd.to_datetime(df['time'].iloc[i])
                hour_val = int(t_val.hour)
                day_of_week_val = int(t_val.dayofweek)
                trend_val = int(df['Trend'].iloc[i])
                killzone_val = get_killzone(hour_val)
                atr_val = df['ATR_14'].iloc[i] if 'ATR_14' in df.columns else 1.0
                if pd.isna(atr_val):
                    atr_val = 1.0
                
                direction = 1 if ob_type == 'BULLISH' else -1
                
                # Fibo levels for OB
                fibo_0_5 = float(df['OB_Fibo_0.5'].iloc[i])
                fibo_0_618 = float(df['OB_Fibo_0.618'].iloc[i])
                fibo_0_0 = float(df['OB_Fibo_0.0'].iloc[i])
                ob_sl = float(df['OB_SL'].iloc[i])
                
                # --- Option A: Midpoint (Fibo 0.5) ---
                entry_a = fibo_0_5
                sl_a = ob_sl
                tp_a = fibo_0_0
                risk_pips_a = abs(entry_a - sl_a)
                rejection_confirmed_a = detect_rejection_at_level(df, entry_a, direction)
                
                tp_dynamic_a = find_dynamic_tp(df, entry_a, direction)
                tp2_a = tp_dynamic_a if tp_dynamic_a is not None else (entry_a + risk_pips_a * 3 if direction == 1 else entry_a - risk_pips_a * 3)
                tp3_a = entry_a + risk_pips_a * 4 if direction == 1 else entry_a - risk_pips_a * 4
                
                active_setups.append({
                    'index': i,
                    'time': df['time'].iloc[i],
                    'hour': hour_val,
                    'day_of_week': day_of_week_val,
                    'setup_type': 1,  # OB
                    'direction': direction,
                    'entry_price': entry_a,
                    'sl_price': sl_a,
                    'tp_price': tp_a,
                    'tp2_price': tp2_a,
                    'tp3_price': tp3_a,
                    'risk_pips': risk_pips_a,
                    'atr_14': atr_val,
                    'trend': trend_val,
                    'relative_risk': risk_pips_a / atr_val,
                    'killzone': killzone_val,
                    'fvg_width': 0.0,
                    'relative_fvg_width': 0.0,
                    'option_name': 'OB Midpoint 0.5',
                    'rejection_confirmed': rejection_confirmed_a
                })
                
                # --- Option B: Golden Pocket (Fibo 0.618) ---
                entry_b = fibo_0_618
                sl_b = ob_sl
                tp_b = fibo_0_0
                risk_pips_b = abs(entry_b - sl_b)
                rejection_confirmed_b = detect_rejection_at_level(df, entry_b, direction)
                
                tp_dynamic_b = find_dynamic_tp(df, entry_b, direction)
                tp2_b = tp_dynamic_b if tp_dynamic_b is not None else (entry_b + risk_pips_b * 3 if direction == 1 else entry_b - risk_pips_b * 3)
                tp3_b = entry_b + risk_pips_b * 4 if direction == 1 else entry_b - risk_pips_b * 4
                
                active_setups.append({
                    'index': i,
                    'time': df['time'].iloc[i],
                    'hour': hour_val,
                    'day_of_week': day_of_week_val,
                    'setup_type': 1,  # OB
                    'direction': direction,
                    'entry_price': entry_b,
                    'sl_price': sl_b,
                    'tp_price': tp_b,
                    'tp2_price': tp2_b,
                    'tp3_price': tp3_b,
                    'risk_pips': risk_pips_b,
                    'atr_14': atr_val,
                    'trend': trend_val,
                    'relative_risk': risk_pips_b / atr_val,
                    'killzone': killzone_val,
                    'fvg_width': 0.0,
                    'relative_fvg_width': 0.0,
                    'option_name': 'OB GoldenPocket 0.618',
                    'rejection_confirmed': rejection_confirmed_b
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
                    if df['Close'].iloc[j] < fvg_bottom:
                        mitigated = True
                        break
                elif fvg_type == 'BEARISH':
                    if df['Close'].iloc[j] > fvg_top:
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
                    
                direction = 1 if fvg_type == 'BULLISH' else -1
                if direction == 1:
                    fvg_width = df['Low'].iloc[i] - df['High'].iloc[i-2]
                else:
                    fvg_width = df['Low'].iloc[i-2] - df['High'].iloc[i]
                
                # Retrieve Fibonacci levels computed during smc_detector
                fibo_0_5 = float(df['FVG_Fibo_0.5'].iloc[i])
                fibo_0_618 = float(df['FVG_Fibo_0.618'].iloc[i])
                fibo_0_0 = float(df['FVG_Fibo_0.0'].iloc[i])
                fvg_sl = float(df['FVG_SL'].iloc[i])
                
                # --- Option A: Midpoint (Fibo 0.5) ---
                entry_a = fibo_0_5
                sl_a = fvg_sl
                tp_a = fibo_0_0
                risk_pips_a = (entry_a - sl_a) if direction == 1 else (sl_a - entry_a)
                rejection_confirmed_a = detect_rejection_at_level(df, entry_a, direction)
                
                tp_dynamic_a = find_dynamic_tp(df, entry_a, direction)
                if tp_dynamic_a is not None:
                    tp2_a = tp_dynamic_a
                else:
                    tp2_a = entry_a + (entry_a - sl_a) * 3 if direction == 1 else entry_a - (sl_a - entry_a) * 3
                
                # Extended Target (TP 3) at 1:4 RR
                tp3_a = entry_a + (entry_a - sl_a) * 4 if direction == 1 else entry_a - (sl_a - entry_a) * 4
                
                active_setups.append({
                    'index': i,
                    'time': df['time'].iloc[i],
                    'hour': hour_val,
                    'day_of_week': day_of_week_val,
                    'setup_type': 0,  # FVG
                    'direction': direction,
                    'entry_price': entry_a,
                    'sl_price': sl_a,
                    'tp_price': tp_a,    # TP 1
                    'tp2_price': tp2_a,  # TP 2 Dynamic
                    'tp3_price': tp3_a,  # TP 3 Extended
                    'risk_pips': risk_pips_a,
                    'atr_14': atr_val,
                    'trend': trend_val,
                    'relative_risk': risk_pips_a / atr_val,
                    'killzone': killzone_val,
                    'fvg_width': fvg_width,
                    'relative_fvg_width': fvg_width / atr_val,
                    'option_name': 'Option A (Midpoint)',
                    'rejection_confirmed': rejection_confirmed_a
                })
                
                # --- Option B: Golden Pocket (Fibo 0.618) ---
                entry_b = fibo_0_618
                sl_b = fvg_sl
                tp_b = fibo_0_0
                risk_pips_b = (entry_b - sl_b) if direction == 1 else (sl_b - entry_b)
                rejection_confirmed_b = detect_rejection_at_level(df, entry_b, direction)
                
                tp_dynamic_b = find_dynamic_tp(df, entry_b, direction)
                if tp_dynamic_b is not None:
                    tp2_b = tp_dynamic_b
                else:
                    tp2_b = entry_b + (entry_b - sl_b) * 3 if direction == 1 else entry_b - (sl_b - entry_b) * 3
                
                # Extended Target (TP 3) at 1:4 RR
                tp3_b = entry_b + (entry_b - sl_b) * 4 if direction == 1 else entry_b - (sl_b - entry_b) * 4
                
                active_setups.append({
                    'index': i,
                    'time': df['time'].iloc[i],
                    'hour': hour_val,
                    'day_of_week': day_of_week_val,
                    'setup_type': 0,  # FVG
                    'direction': direction,
                    'entry_price': entry_b,
                    'sl_price': sl_b,
                    'tp_price': tp_b,    # TP 1
                    'tp2_price': tp2_b,  # TP 2 Dynamic
                    'tp3_price': tp3_b,  # TP 3 Extended
                    'risk_pips': risk_pips_b,
                    'atr_14': atr_val,
                    'trend': trend_val,
                    'relative_risk': risk_pips_b / atr_val,
                    'killzone': killzone_val,
                    'fvg_width': fvg_width,
                    'relative_fvg_width': fvg_width / atr_val,
                    'option_name': 'Option B (Golden Pocket)',
                    'rejection_confirmed': rejection_confirmed_b
                })
                
    # 3. Breaker Block Setups
    if 'BB_Type' in df.columns:
        for i in range(len(df)):
            bb_type = df['BB_Type'].iloc[i]
            if pd.notna(bb_type) and bb_type is not None:
                if not df['BB_Mitigated'].iloc[i]:
                    bb_top = df['BB_Top'].iloc[i]
                    bb_bottom = df['BB_Bottom'].iloc[i]
                    t_val = pd.to_datetime(df['time'].iloc[i])
                    hour_val = int(t_val.hour)
                    day_of_week_val = int(t_val.dayofweek)
                    trend_val = int(df['Trend'].iloc[i])
                    killzone_val = get_killzone(hour_val)
                    atr_val = df['ATR_14'].iloc[i] if 'ATR_14' in df.columns else 1.0
                    if pd.isna(atr_val):
                        atr_val = 1.0
                        
                    direction = 1 if bb_type == 'BULLISH' else -1
                    if direction == 1:
                        entry = bb_bottom  # Retest level
                        sl = entry - buffer
                        tp = entry + (entry - sl) * 2
                    else:
                        entry = bb_top
                        sl = entry + buffer
                        tp = entry - (sl - entry) * 2
                        
                    risk_pips = (entry - sl) if direction == 1 else (sl - entry)
                    rejection_confirmed = detect_rejection_at_level(df, entry, direction)
                    
                    tp_dynamic = find_dynamic_tp(df, entry, direction)
                    tp2 = tp_dynamic if tp_dynamic is not None else (entry + risk_pips * 3 if direction == 1 else entry - risk_pips * 3)
                    tp3 = entry + risk_pips * 4 if direction == 1 else entry - risk_pips * 4
                    
                    active_setups.append({
                        'index': i,
                        'time': df['time'].iloc[i],
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # Treat as OB for ML
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'tp2_price': tp2,
                        'tp3_price': tp3,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'option_name': f'Breaker ({bb_type})',
                        'rejection_confirmed': rejection_confirmed
                    })
                    
    # 4. Swapzone Setups
    if 'Swap_Type' in df.columns:
        for i in range(len(df)):
            swap_type = df['Swap_Type'].iloc[i]
            if pd.notna(swap_type) and swap_type is not None:
                if not df['Swap_Mitigated'].iloc[i]:
                    swap_level = df['Swap_Level'].iloc[i]
                    t_val = pd.to_datetime(df['time'].iloc[i])
                    hour_val = int(t_val.hour)
                    day_of_week_val = int(t_val.dayofweek)
                    trend_val = int(df['Trend'].iloc[i])
                    killzone_val = get_killzone(hour_val)
                    atr_val = df['ATR_14'].iloc[i] if 'ATR_14' in df.columns else 1.0
                    if pd.isna(atr_val):
                        atr_val = 1.0
                        
                    direction = 1 if swap_type == 'SUPPORT' else -1
                    entry = swap_level
                    sl = entry - buffer if direction == 1 else entry + buffer
                    tp = entry + (entry - sl) * 2 if direction == 1 else entry - (sl - entry) * 2
                    
                    risk_pips = (entry - sl) if direction == 1 else (sl - entry)
                    rejection_confirmed = detect_rejection_at_level(df, entry, direction)
                    
                    tp_dynamic = find_dynamic_tp(df, entry, direction)
                    tp2 = tp_dynamic if tp_dynamic is not None else (entry + risk_pips * 3 if direction == 1 else entry - risk_pips * 3)
                    tp3 = entry + risk_pips * 4 if direction == 1 else entry - risk_pips * 4
                    
                    active_setups.append({
                        'index': i,
                        'time': df['time'].iloc[i],
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # Treat as OB for ML
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'tp2_price': tp2,
                        'tp3_price': tp3,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'option_name': f'Swapzone ({swap_type})',
                        'rejection_confirmed': rejection_confirmed
                    })
                    
    # 5. BPR Setups
    if 'BPR_Type' in df.columns:
        for i in range(len(df)):
            bpr_type = df['BPR_Type'].iloc[i]
            if pd.notna(bpr_type) and bpr_type is not None:
                if not df['BPR_Mitigated'].iloc[i]:
                    bpr_top = df['BPR_Top'].iloc[i]
                    bpr_bottom = df['BPR_Bottom'].iloc[i]
                    t_val = pd.to_datetime(df['time'].iloc[i])
                    hour_val = int(t_val.hour)
                    day_of_week_val = int(t_val.dayofweek)
                    trend_val = int(df['Trend'].iloc[i])
                    killzone_val = get_killzone(hour_val)
                    atr_val = df['ATR_14'].iloc[i] if 'ATR_14' in df.columns else 1.0
                    if pd.isna(atr_val):
                        atr_val = 1.0
                        
                    direction = 1 if bpr_type == 'BULLISH' else -1
                    
                    # Fibo levels for BPR
                    fibo_0_5 = float(df['BPR_Fibo_0.5'].iloc[i])
                    fibo_0_618 = float(df['BPR_Fibo_0.618'].iloc[i])
                    fibo_0_0 = float(df['BPR_Fibo_0.0'].iloc[i])
                    bpr_sl = float(df['BPR_SL'].iloc[i])
                    
                    # --- Option A: Midpoint (Fibo 0.5) ---
                    entry_a = fibo_0_5
                    sl_a = bpr_sl
                    tp_a = fibo_0_0
                    risk_pips_a = abs(entry_a - sl_a)
                    rejection_confirmed_a = detect_rejection_at_level(df, entry_a, direction)
                    
                    tp_dynamic_a = find_dynamic_tp(df, entry_a, direction)
                    tp2_a = tp_dynamic_a if tp_dynamic_a is not None else (entry_a + risk_pips_a * 3 if direction == 1 else entry_a - risk_pips_a * 3)
                    tp3_a = entry_a + risk_pips_a * 4 if direction == 1 else entry_a - risk_pips_a * 4
                    
                    active_setups.append({
                        'index': i,
                        'time': df['time'].iloc[i],
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 0,  # Treat as FVG for ML to leverage FVG features
                        'direction': direction,
                        'entry_price': entry_a,
                        'sl_price': sl_a,
                        'tp_price': tp_a,
                        'tp2_price': tp2_a,
                        'tp3_price': tp3_a,
                        'risk_pips': risk_pips_a,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips_a / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': abs(bpr_top - bpr_bottom),
                        'relative_fvg_width': abs(bpr_top - bpr_bottom) / atr_val,
                        'option_name': 'BPR Midpoint 0.5',
                        'rejection_confirmed': rejection_confirmed_a
                    })
                    
                    # --- Option B: Golden Pocket (Fibo 0.618) ---
                    entry_b = fibo_0_618
                    sl_b = bpr_sl
                    tp_b = fibo_0_0
                    risk_pips_b = abs(entry_b - sl_b)
                    rejection_confirmed_b = detect_rejection_at_level(df, entry_b, direction)
                    
                    tp_dynamic_b = find_dynamic_tp(df, entry_b, direction)
                    tp2_b = tp_dynamic_b if tp_dynamic_b is not None else (entry_b + risk_pips_b * 3 if direction == 1 else entry_b - risk_pips_b * 3)
                    tp3_b = entry_b + risk_pips_b * 4 if direction == 1 else entry_b - risk_pips_b * 4
                    
                    active_setups.append({
                        'index': i,
                        'time': df['time'].iloc[i],
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 0,  # Treat as FVG for ML to leverage FVG features
                        'direction': direction,
                        'entry_price': entry_b,
                        'sl_price': sl_b,
                        'tp_price': tp_b,
                        'tp2_price': tp2_b,
                        'tp3_price': tp3_b,
                        'risk_pips': risk_pips_b,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips_b / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': abs(bpr_top - bpr_bottom),
                        'relative_fvg_width': abs(bpr_top - bpr_bottom) / atr_val,
                        'option_name': 'BPR GoldenPocket 0.618',
                        'rejection_confirmed': rejection_confirmed_b
                    })
                    
    # 6. Indecision Candle (IC) Setups
    if 'IC_Type' in df.columns:
        for i in range(len(df)):
            ic_type = df['IC_Type'].iloc[i]
            if pd.notna(ic_type) and ic_type is not None:
                if not df['IC_Mitigated'].iloc[i]:
                    t_val = pd.to_datetime(df['time'].iloc[i])
                    hour_val = int(t_val.hour)
                    day_of_week_val = int(t_val.dayofweek)
                    trend_val = int(df['Trend'].iloc[i]) if 'Trend' in df.columns else 1
                    killzone_val = get_killzone(hour_val)
                    atr_val = df['ATR_14'].iloc[i] if 'ATR_14' in df.columns else 1.0
                    if pd.isna(atr_val):
                        atr_val = 1.0
                        
                    direction = 1 if ic_type == 'BULLISH' else -1
                    fibo_0_5 = float(df['IC_Fibo_0.5'].iloc[i])
                    fibo_0_618 = float(df['IC_Fibo_0.618'].iloc[i])
                    fibo_0_0 = float(df['IC_Fibo_0.0'].iloc[i])
                    ic_sl = float(df['IC_SL'].iloc[i])
                    
                    # --- Option A: Midpoint (Fibo 0.5) ---
                    entry_a = fibo_0_5
                    sl_a = ic_sl
                    tp_a = fibo_0_0
                    risk_pips_a = abs(entry_a - sl_a)
                    rejection_confirmed_a = detect_rejection_at_level(df, entry_a, direction)
                    
                    tp_dynamic_a = find_dynamic_tp(df, entry_a, direction)
                    tp2_a = tp_dynamic_a if tp_dynamic_a is not None else (entry_a + risk_pips_a * 3 if direction == 1 else entry_a - risk_pips_a * 3)
                    tp3_a = entry_a + risk_pips_a * 4 if direction == 1 else entry_a - risk_pips_a * 4
                    
                    active_setups.append({
                        'index': i,
                        'time': df['time'].iloc[i],
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # OB
                        'direction': direction,
                        'entry_price': entry_a,
                        'sl_price': sl_a,
                        'tp_price': tp_a,
                        'tp2_price': tp2_a,
                        'tp3_price': tp3_a,
                        'risk_pips': risk_pips_a,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips_a / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'option_name': 'IC Midpoint 0.5',
                        'rejection_confirmed': rejection_confirmed_a
                    })
                    
                    # --- Option B: Golden Pocket (Fibo 0.618) ---
                    entry_b = fibo_0_618
                    sl_b = ic_sl
                    tp_b = fibo_0_0
                    risk_pips_b = abs(entry_b - sl_b)
                    rejection_confirmed_b = detect_rejection_at_level(df, entry_b, direction)
                    
                    tp_dynamic_b = find_dynamic_tp(df, entry_b, direction)
                    tp2_b = tp_dynamic_b if tp_dynamic_b is not None else (entry_b + risk_pips_b * 3 if direction == 1 else entry_b - risk_pips_b * 3)
                    tp3_b = entry_b + risk_pips_b * 4 if direction == 1 else entry_b - risk_pips_b * 4
                    
                    active_setups.append({
                        'index': i,
                        'time': df['time'].iloc[i],
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # Treat as OB for ML
                        'direction': direction,
                        'entry_price': entry_b,
                        'sl_price': sl_b,
                        'tp_price': tp_b,
                        'tp2_price': tp2_b,
                        'tp3_price': tp3_b,
                        'risk_pips': risk_pips_b,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips_b / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'option_name': 'IC GoldenPocket 0.618',
                        'rejection_confirmed': rejection_confirmed_b
                    })
                    
    # 7. Supply & Demand (SND) Setups
    if 'SD_Type' in df.columns:
        for i in range(len(df)):
            sd_type = df['SD_Type'].iloc[i]
            if pd.notna(sd_type) and sd_type is not None:
                if not df['SD_Mitigated'].iloc[i]:
                    sd_top = df['SD_Top'].iloc[i]
                    sd_bottom = df['SD_Bottom'].iloc[i]
                    t_val = pd.to_datetime(df['time'].iloc[i])
                    hour_val = int(t_val.hour)
                    day_of_week_val = int(t_val.dayofweek)
                    trend_val = int(df['Trend'].iloc[i]) if 'Trend' in df.columns else 1
                    killzone_val = get_killzone(hour_val)
                    atr_val = df['ATR_14'].iloc[i] if 'ATR_14' in df.columns else 1.0
                    if pd.isna(atr_val):
                        atr_val = 1.0
                        
                    direction = 1 if 'DEMAND' in sd_type else -1
                    fibo_0_5 = float(df['SD_Fibo_0.5'].iloc[i])
                    fibo_0_618 = float(df['SD_Fibo_0.618'].iloc[i])
                    fibo_0_0 = float(df['SD_Fibo_0.0'].iloc[i])
                    sd_sl = float(df['SD_SL'].iloc[i])
                    
                    # --- Option A: Midpoint (Fibo 0.5) ---
                    entry_a = fibo_0_5
                    sl_a = sd_sl
                    tp_a = fibo_0_0
                    risk_pips_a = abs(entry_a - sl_a)
                    rejection_confirmed_a = detect_rejection_at_level(df, entry_a, direction)
                    
                    tp_dynamic_a = find_dynamic_tp(df, entry_a, direction)
                    tp2_a = tp_dynamic_a if tp_dynamic_a is not None else (entry_a + risk_pips_a * 3 if direction == 1 else entry_a - risk_pips_a * 3)
                    tp3_a = entry_a + risk_pips_a * 4 if direction == 1 else entry_a - risk_pips_a * 4
                    
                    active_setups.append({
                        'index': i,
                        'time': df['time'].iloc[i],
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # Treat as OB for ML
                        'direction': direction,
                        'entry_price': entry_a,
                        'sl_price': sl_a,
                        'tp_price': tp_a,
                        'tp2_price': tp2_a,
                        'tp3_price': tp3_a,
                        'risk_pips': risk_pips_a,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips_a / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'option_name': f'SND Midpoint 0.5 ({sd_type})',
                        'rejection_confirmed': rejection_confirmed_a
                    })
                    
                    # --- Option B: Golden Pocket (Fibo 0.618) ---
                    entry_b = fibo_0_618
                    sl_b = sd_sl
                    tp_b = fibo_0_0
                    risk_pips_b = abs(entry_b - sl_b)
                    rejection_confirmed_b = detect_rejection_at_level(df, entry_b, direction)
                    
                    tp_dynamic_b = find_dynamic_tp(df, entry_b, direction)
                    tp2_b = tp_dynamic_b if tp_dynamic_b is not None else (entry_b + risk_pips_b * 3 if direction == 1 else entry_b - risk_pips_b * 3)
                    tp3_b = entry_b + risk_pips_b * 4 if direction == 1 else entry_b - risk_pips_b * 4
                    
                    active_setups.append({
                        'index': i,
                        'time': df['time'].iloc[i],
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # Treat as OB for ML
                        'direction': direction,
                        'entry_price': entry_b,
                        'sl_price': sl_b,
                        'tp_price': tp_b,
                        'tp2_price': tp2_b,
                        'tp3_price': tp3_b,
                        'risk_pips': risk_pips_b,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips_b / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'option_name': f'SND GoldenPocket 0.618 ({sd_type})',
                        'rejection_confirmed': rejection_confirmed_b
                    })
                    
    # 8. Pivot Rejection Setups
    from src.indicators.pivots import detect_pivot_rejection_setups_at_idx
    # Look back over the last 10 bars to find any active/open pivot rejection setups
    lookback_scan = min(10, len(df))
    start_scan_idx = len(df) - lookback_scan
    
    for k in range(start_scan_idx, len(df)):
        pivot_setups = detect_pivot_rejection_setups_at_idx(df, k, symbol=symbol)
        for ps in pivot_setups:
            direction = ps['direction']
            entry = ps['entry_price']
            sl = ps['sl_price']
            tp = ps['tp_price']
            
            # Check if this setup is still active (has not resolved to Win or Loss)
            is_active = True
            if k < len(df) - 1:
                from src.labeler import simulate_trade
                outcome = simulate_trade(df, k + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if outcome is not None:
                    # Already resolved, not active
                    is_active = False
                    
            if is_active:
                t_val = pd.to_datetime(df['time'].iloc[k])
                hour_val = int(t_val.hour)
                day_of_week_val = int(t_val.dayofweek)
                trend_val = int(df['Trend'].iloc[k]) if 'Trend' in df.columns else 1
                killzone_val = get_killzone(hour_val)
                atr_val = df['ATR_14'].iloc[k] if 'ATR_14' in df.columns else 1.0
                if pd.isna(atr_val):
                    atr_val = 1.0
                risk_pips = abs(entry - sl)
                
                # Pivot setups detected on candle k already confirmed rejection
                rejection_confirmed = True
                
                tp_dynamic = find_dynamic_tp(df, entry, direction)
                tp2 = tp_dynamic if tp_dynamic is not None else (entry + risk_pips * 3 if direction == 1 else entry - risk_pips * 3)
                tp3 = entry + risk_pips * 4 if direction == 1 else entry - risk_pips * 4
                
                active_setups.append({
                    'index': k,
                    'time': df['time'].iloc[k],
                    'hour': hour_val,
                    'day_of_week': day_of_week_val,
                    'setup_type': 2,  # 2: Pivot Rejection
                    'direction': direction,
                    'entry_price': entry,
                    'sl_price': sl,
                    'tp_price': tp,
                    'tp2_price': tp2,
                    'tp3_price': tp3,
                    'risk_pips': risk_pips,
                    'atr_14': atr_val,
                    'trend': trend_val,
                    'relative_risk': risk_pips / atr_val,
                    'killzone': killzone_val,
                    'fvg_width': 0.0,
                    'relative_fvg_width': 0.0,
                    'option_name': ps['option_name'],
                    'rejection_confirmed': rejection_confirmed
                })
                    
    # Tag setups with psychological price proximity status and FLOOP Pro features
    from src.indicators.pivots import get_pivot_features_at_idx
    for s in active_setups:
        s['near_psychological_level'] = int(is_near_psychological_level(s['entry_price'], symbol))
        idx = s['index']
        s['floop_signal'] = int(df['floop_signal'].iloc[idx]) if 'floop_signal' in df.columns else 0
        s['floop_strength'] = float(df['floop_strength'].iloc[idx]) if 'floop_strength' in df.columns else 0.0
        s['floop_trend'] = int(df['floop_trend'].iloc[idx]) if 'floop_trend' in df.columns else 0
        
        p_feat = get_pivot_features_at_idx(df, idx, s['entry_price'])
        s['dist_entry_to_pp'] = p_feat['dist_entry_to_pp']
        s['dist_entry_to_nearest_pivot'] = p_feat['dist_entry_to_nearest_pivot']
        
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
            timeframes_data['M5'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M5, 300)
            timeframes_data['M1'] = fetch_historical_data(symbol, mt5.TIMEFRAME_M1, 300)
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
        timeframes_data['M5'] = generate_synthetic_data(300, seed=47)
        timeframes_data['M1'] = generate_synthetic_data(300, seed=48)
        
    # Run SMC detection algorithms on all timeframes
    print("Running SMC structure detection algorithms on all timeframes...")
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        df_tf = detect_swing_points(df_tf, window=5)
        df_tf = detect_structures(df_tf)
        df_tf = detect_fvg_and_ob(df_tf, symbol=symbol)
        df_tf = detect_snr_and_swapzones(df_tf, symbol=symbol)
        df_tf = detect_bpr(df_tf, symbol=symbol)
        df_tf = detect_indecision_candles(df_tf, symbol=symbol)
        df_tf = detect_supply_demand_zones(df_tf, symbol=symbol)
        
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
        
    # Pre-calculate trends for FLOOP Pro MTF/HTF
    from src.indicators.floop import calculate_atr, calculate_range_filter
    tf_trends = {}
    for tf_name in timeframes_data:
        df_tf = timeframes_data[tf_name]
        try:
            df_tf_copy = df_tf.copy()
            df_tf_copy['time'] = pd.to_datetime(df_tf_copy['time'])
            df_tf_copy.set_index('time', inplace=True)
            
            atr_floop = calculate_atr(df_tf_copy, 14)
            _, trend_floop, _ = calculate_range_filter(df_tf_copy['Close'], atr_floop, sensitivity=6, atr_multiplier=0.8)
            tf_trends[tf_name] = pd.Series(trend_floop, index=df_tf_copy.index)
        except Exception as e:
            print(f"Error calculating RF trend for TF {tf_name}: {e}")
            tf_trends[tf_name] = None

    # Extract active HTF FVGs (from H1, H4, and D1)
    active_htf_fvgs = []
    for tf_name in ['H1', 'H4', 'D1']:
        tf_fvgs = extract_active_htf_fvgs(timeframes_data[tf_name])
        for fvg in tf_fvgs:
            fvg['timeframe'] = tf_name
            active_htf_fvgs.append(fvg)
            
    print(f"Detected {len(active_htf_fvgs)} active Higher Timeframe (HTF) FVGs.")
    
    # Identify active setups on ALL timeframes
    all_setups = []
    for tf_name in ['D1', 'H4', 'H1', 'M30', 'M15']:
        tf_setups = get_active_setups(timeframes_data[tf_name], symbol=symbol, tf_trends=tf_trends, df_d1=timeframes_data.get('D1'))
        for s in tf_setups:
            s['timeframe'] = tf_name
            all_setups.append(s)
            
    # Extract active FVGs for all timeframes to check alignment
    active_fvgs_by_tf = {}
    for tf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
        active_fvgs_by_tf[tf_name] = extract_active_htf_fvgs(timeframes_data[tf_name])
        
    # Timeframe weights to check hierarchy
    tf_weights = {'M15': 1, 'M30': 2, 'H1': 3, 'H4': 4, 'D1': 5}
       # Check if the setup entry price falls inside any active HTF FVG of the same direction,
    # check for HTF trend conflicts, and re-check rejection on LTF (M15) for HTF setups.
    tf_minutes_map = {'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
    
    for setup in all_setups:
        setup['htf_prioritized'] = False
        setup['matching_htf_fvgs'] = []
        setup['suppressed'] = False
        setup['htf_conflict_reason'] = ""
        setup_tf = setup['timeframe']
        
        # Look for same-direction HTF FVGs for prioritization, and opposite-direction for suppression
        for htf_name in ['M30', 'H1', 'H4', 'D1']:
            if tf_weights[htf_name] > tf_weights[setup_tf]:
                # 1. Prioritization
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_same_direction = (setup['direction'] == 1 and htf_fvg['type'] == 'BULLISH') or \
                                         (setup['direction'] == -1 and htf_fvg['type'] == 'BEARISH')
                    if is_same_direction:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['htf_prioritized'] = True
                            fvg_info = htf_fvg.copy()
                            fvg_info['timeframe'] = htf_name
                            setup['matching_htf_fvgs'].append(fvg_info)
                            
                # 2. Conflict Suppression - only if entry is inside the opposite HTF FVG
                for htf_fvg in active_fvgs_by_tf[htf_name]:
                    is_opposite_direction = (setup['direction'] == 1 and htf_fvg['type'] == 'BEARISH') or \
                                             (setup['direction'] == -1 and htf_fvg['type'] == 'BULLISH')
                    if is_opposite_direction:
                        entry = setup['entry_price']
                        if entry >= htf_fvg['bottom'] and entry <= htf_fvg['top']:
                            setup['suppressed'] = True
                            setup['htf_conflict_reason'] = f"Entry inside opposite active {htf_name} FVG"
                            break
                        
        # Check Rejection on lower timeframes (M15, M5, M1) for the setup
        rej_confirmed = False
        rej_tf_source = "None"
        
        # 1. Check on M5 (lookback 30 candles)
        m5_df = timeframes_data.get('M5')
        if m5_df is not None and not m5_df.empty:
            if detect_rejection_at_level(m5_df, setup['entry_price'], setup['direction'], lookback=30):
                rej_confirmed = True
                rej_tf_source = "M5"
                
        # 2. Check on M1 (lookback 90 candles) if not already confirmed on M5
        if not rej_confirmed:
            m1_df = timeframes_data.get('M1')
            if m1_df is not None and not m1_df.empty:
                if detect_rejection_at_level(m1_df, setup['entry_price'], setup['direction'], lookback=90):
                    rej_confirmed = True
                    rej_tf_source = "M1"
                    
        # 3. Fallback to M15 (lookback 15 candles) if not confirmed on M5/M1 and setup is on timeframe higher than M15
        if not rej_confirmed and setup_tf != 'M15':
            m15_df = timeframes_data.get('M15')
            if m15_df is not None and not m15_df.empty:
                if detect_rejection_at_level(m15_df, setup['entry_price'], setup['direction'], lookback=15):
                    rej_confirmed = True
                    rej_tf_source = "M15"
                    
        setup['rejection_confirmed'] = rej_confirmed
        setup['rejection_source'] = rej_tf_source
                            
    # Pre-calculate KNN and Volume Profile data for each timeframe to use for setup features
    print("Pre-calculating KNN and Volume Profile features for live signals...")
    tf_knn_data = {}
    tf_vp_data = {}
    for tf_name, df_tf in timeframes_data.items():
        # KNN
        try:
            pc1, pc2, pc3, pc4, target_clean = run_knn_classifier(
                df_tf,
                atr_period=10, factor=2.0,
                k_neighbors=10, sampling_window_size=1000, momentum_window=10,
                normalizing_window_size=1000,
                lazy=True
            )
            # Evaluate at the very last bar (live candle)
            t_last = len(df_tf) - 1
            knn_up, knn_down = calculate_knn_probability_at_bar(
                t_last, pc1.values, pc2.values, pc3.values, pc4.values, target_clean.values,
                k=10, sampling_window=1000, stride=10
            )
            tf_knn_data[tf_name] = (knn_up, knn_down)
        except Exception as e:
            print(f"Error computing live KNN for TF {tf_name}: {e}")
            tf_knn_data[tf_name] = (0.0, 0.0)
            
        # Volume profile
        try:
            clusters_data = calculate_volume_clusters(
                df_tf, lookback=200, k=5, iterations=20, rows=20
            )
            tf_vp_data[tf_name] = clusters_data
        except Exception as e:
            print(f"Error computing live Volume Clusters for {tf_name}: {e}")
            tf_vp_data[tf_name] = {}

    # Query model predictions for each setup
    filtered_setups_with_prob = []
    for setup in all_setups:
        setup_tf = setup['timeframe']
        knn_up, knn_down = tf_knn_data.get(setup_tf, (0.0, 0.0))
        knn_prob_sig = knn_up if setup['direction'] == 1 else knn_down
        knn_prob_opp = knn_down if setup['direction'] == 1 else knn_up
        
        clusters_data = tf_vp_data.get(setup_tf, {})
        dist_entry_to_poc = 0.0
        dist_entry_to_nearest_poc = 0.0
        if clusters_data and 'current_poc' in clusters_data:
            curr_poc = clusters_data['current_poc']
            entry = setup['entry_price']
            dist_entry_to_poc = (entry - curr_poc) / curr_poc if curr_poc > 0 else 0.0
            
            pocs = clusters_data.get('pocs', [])
            if pocs:
                dist_entry_to_nearest_poc = min(abs(entry - poc) for poc in pocs) / entry

        features = {
            'timeframe': tf_minutes_map[setup['timeframe']],
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
            'relative_risk': setup['relative_risk'],
            'killzone': setup['killzone'],
            'fvg_width': setup['fvg_width'],
            'relative_fvg_width': setup['relative_fvg_width'],
            'near_psychological_level': setup['near_psychological_level'],
            'knn_prob_sig': knn_prob_sig,
            'knn_prob_opp': knn_prob_opp,
            'dist_entry_to_poc': dist_entry_to_poc,
            'dist_entry_to_nearest_poc': dist_entry_to_nearest_poc,
            'dist_entry_to_pp': setup.get('dist_entry_to_pp', 0.0),
            'dist_entry_to_nearest_pivot': setup.get('dist_entry_to_nearest_pivot', 0.0),
            'floop_signal': setup['floop_signal'],
            'floop_strength': setup['floop_strength'],
            'floop_trend': setup.get('floop_trend', 0),
            'floop_trend_aligned': 1 if setup.get('floop_trend', 0) == setup['direction'] else 0
        }
        try:
            prob = predict_setup_probability(features)
            setup['probability'] = prob
            
            if setup['suppressed']:
                setup['status'] = "FILTERED (HTF Conflict)"
            else:
                setup['status'] = "HIGH CONFIDENCE SIGNAL" if prob >= 0.70 else "FILTERED (Low Confidence)"
        except Exception as e:
            print(f"Error predicting setup probability: {e}")
            setup['probability'] = 0.0
            setup['status'] = "ERROR"
            
        filtered_setups_with_prob.append(setup)
        
    # Print results in terminal
    print("\n" + "="*160)
    print("                                                         ACTIVE SMC TRADE SIGNALS & ML FILTERING")
    print("="*160)
    print(f"{'Time':<16} | {'TF':<3} | {'Type':<4} | {'Dir':<7} | {'Entry Option (Price)':<30} | {'SL':<8} | {'TP 1':<8} | {'TP 2 (Dyn)':<10} | {'TP 3 (Ext)':<10} | {'Win Prob':<8} | {'HTF Prior':<9} | {'Rej Conf':<8} | Status")
    print("-"*160)
    for setup in filtered_setups_with_prob:
        setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
        dir_name = "Bullish" if setup['direction'] == 1 else "Bearish"
        prior_str = "YES" if setup['htf_prioritized'] else "NO"
        rej_str = "YES" if setup.get('rejection_confirmed', False) else "NO"
        entry_opt_str = f"{setup['option_name']} ({setup['entry_price']:.3f})"
        
        time_str = str(setup['time'])
        if len(time_str) >= 16:
            time_str = time_str[:16]
            
        print(f"{time_str:<16} | {setup['timeframe']:<3} | {setup_name:<4} | {dir_name:<7} | {entry_opt_str:<30} | {setup['sl_price']:.3f} | {setup['tp_price']:.3f} | {setup['tp2_price']:.3f} | {setup['tp3_price']:.3f} | {setup['probability']:.2%} | {prior_str:<9} | {rej_str:<8} | {setup['status']}")
    print("="*160 + "\n")
    
    # Print prioritized setups clearly
    prioritized_setups = [s for s in filtered_setups_with_prob if s['htf_prioritized']]
    if prioritized_setups:
        print("*"*160)
        print("                                                    PRIORITIZED MULTI-TIMEFRAME (HTF) SETUPS")
        print("*"*160)
        for setup in prioritized_setups:
            setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
            dir_name = "Bullish" if setup['direction'] == 1 else "Bearish"
            matching_desc = ", ".join([f"{f['timeframe']} FVG ({f['bottom']:.3f}-{f['top']:.3f})" for f in setup['matching_htf_fvgs']])
            rej_str = "Confirmed" if setup.get('rejection_confirmed', False) else "No Rejection"
            print(f"* {setup['timeframe']} {dir_name} {setup_name} | {setup['option_name']} at {setup['entry_price']:.3f} | SL: {setup['sl_price']:.3f} | TP 1: {setup['tp_price']:.3f} | TP 2: {setup['tp2_price']:.3f} | TP 3: {setup['tp3_price']:.3f} | Rej: {rej_str} | matched HTF: {matching_desc} (Win Prob: {setup['probability']:.2%})")
        print("*"*160 + "\n")
        
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
    
    # Generate visualizations for all timeframes
    print("Generating visualizations for all timeframes...")
    artifact_dir = os.path.join(os.environ.get('APPDATA', ''), 'gemini', 'antigravity-cli', 'brain', 'aade0c14-67d6-4b69-a8a6-5834a430a34c')
    if not os.path.exists(artifact_dir):
        artifact_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'brain', 'aade0c14-67d6-4b69-a8a6-5834a430a34c')
    
    import shutil
    
    # Save standard xauusd_smc_analysis.png as M15 for compatibility
    chart_title_m15 = f"XAUUSD M15 - {'Exness Live' if mt5_active else 'Simulated'} Market Structure"
    m15_setups = [s for s in filtered_setups_with_prob if s.get('timeframe') == 'M15']
    plot_smc_chart(timeframes_data['M15'], title=chart_title_m15, active_setups=m15_setups, output_filename="xauusd_smc_analysis.png")
    if os.path.exists(artifact_dir):
        try:
            shutil.copy("xauusd_smc_analysis.png", os.path.join(artifact_dir, "xauusd_smc_analysis.png"))
        except Exception as e:
            print(f"Error copying general chart: {e}")

    for tf_name in ['D1', 'H4', 'H1', 'M30', 'M15']:
        tf_df = timeframes_data[tf_name]
        tf_setups = [s for s in filtered_setups_with_prob if s.get('timeframe') == tf_name]
        tf_title = f"XAUUSD {tf_name} - {'Exness Live' if mt5_active else 'Simulated'} Market Structure"
        tf_filename = f"xauusd_smc_analysis_{tf_name}.png"
        
        plot_smc_chart(tf_df, title=tf_title, active_setups=tf_setups, output_filename=tf_filename)
        
        # Copy to artifact folder
        if os.path.exists(artifact_dir):
            try:
                shutil.copy(tf_filename, os.path.join(artifact_dir, tf_filename))
                print(f"Saved and copied {tf_filename} to artifacts.")
            except Exception as e:
                print(f"Error copying {tf_filename}: {e}")
                
    print("Phase 2 complete! Signal Filtering Engine & Self-Learning Loop successfully integrated with MTF and BPR.")

if __name__ == "__main__":
    main()
