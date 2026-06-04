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

def plot_smc_chart(df: pd.DataFrame, title: str = "XAUUSD M15 - SMC/ICT Analysis"):
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
    
    plt.tight_layout()
    output_filename = "xauusd_smc_analysis.png"
    plt.savefig(output_filename, dpi=180, facecolor='#131722')
    plt.close()
    print(f"SMC analysis visualization successfully saved as '{output_filename}'")
    
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
            # mt5.TIMEFRAME_M15 = 15
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
    print("-------------------------\n")
    
    # Generate visualization
    chart_title = "XAUUSD M15 - Exness Live Data" if mt5_active else "XAUUSD M15 - Simulated Market Structure"
    plot_smc_chart(df, title=chart_title)
    print("Phase 1 complete! SMC Engine successfully validated.")

if __name__ == "__main__":
    main()
