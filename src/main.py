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

def generate_synthetic_data(num_candles=100) -> pd.DataFrame:
    """
    Generate synthetic candlestick data representing simulated market structures.
    This acts as a fallback when MT5 terminal is not active.
    """
    print("Generating synthetic XAUUSD data for visualization...")
    np.random.seed(42)
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

def main():
    print("=== SMC/ICT AUTO-ANALYZER CORE ENGINE ===")
    
    # Try to connect to MT5 Exness terminal
    mt5_active = False
    try:
        import MetaTrader5 as mt5
        if connect_mt5():
            print("Connected successfully to MetaTrader 5!")
            # Check symbol list
            symbol = "XAUUSD"
            # Try fetching M15 timeframe (200 candles)
            print(f"Fetching last 150 candles of {symbol} from MT5...")
            df = fetch_historical_data(symbol, 15, 150)
            mt5_active = True
            mt5.shutdown()
        else:
            print("MetaTrader 5 terminal not running. Switching to fallback mode.")
    except Exception as e:
        print(f"Could not connect to MT5. Error: {e}")
        print("Switching to fallback mode.")
        
    if not mt5_active:
        # Fallback to simulated data
        df = generate_synthetic_data(120)
        
    # Run SMC detection algorithms
    print("Running SMC structure detection algorithms...")
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df)
    
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
    
    # Identify active setups
    active_setups = get_active_setups(df)
    
    # Query model predictions for each setup
    filtered_setups_with_prob = []
    for setup in active_setups:
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
    print("\n" + "="*80)
    print("                    ACTIVE SMC TRADE SIGNALS & ML FILTERING")
    print("="*80)
    print(f"{'Time':<20} | {'Type':<5} | {'Dir':<8} | {'Entry':<8} | {'SL':<8} | {'TP':<8} | {'Win Prob':<8} | Status")
    print("-"*80)
    for setup in filtered_setups_with_prob:
        setup_name = "OB" if setup['setup_type'] == 1 else "FVG"
        dir_name = "Bullish" if setup['direction'] == 1 else "Bearish"
        print(f"{str(setup['time']):<20} | {setup_name:<5} | {dir_name:<8} | {setup['entry_price']:.3f} | {setup['sl_price']:.3f} | {setup['tp_price']:.3f} | {setup['probability']:.2%} | {setup['status']}")
    print("="*80 + "\n")
    
    # Display statistics
    num_bos = df['BOS'].notna().sum()
    num_choch = df['CHoCH'].notna().sum()
    num_fvg = df['FVG_Type'].notna().sum() if 'FVG_Type' in df.columns else 0
    num_ob = df['OB_Type'].notna().sum() if 'OB_Type' in df.columns else 0
    
    print("\n--- Detection Summary ---")
    print(f"Total Candles Analyzed: {len(df)}")
    print(f"Break of Structure (BOS) signals: {num_bos}")
    print(f"Change of Character (CHoCH) signals: {num_choch}")
    print(f"Fair Value Gaps (FVG) detected: {num_fvg}")
    print(f"Order Blocks (OB) created: {num_ob}")
    print(f"Active Setups Found: {len(filtered_setups_with_prob)}")
    print(f"High Confidence Signals: {sum(1 for s in filtered_setups_with_prob if s['status'] == 'HIGH CONFIDENCE SIGNAL')}")
    print("-------------------------\n")
    
    # Generate visualization
    chart_title = "XAUUSD M15 - Exness Live Data" if mt5_active else "XAUUSD M15 - Simulated Market Structure"
    plot_smc_chart(df, title=chart_title, active_setups=filtered_setups_with_prob)
    print("Phase 2 complete! Signal Filtering Engine & Self-Learning Loop successfully integrated.")

if __name__ == "__main__":
    main()
