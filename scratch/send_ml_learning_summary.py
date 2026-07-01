import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add root project directory to system path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.telegram_bot import send_telegram_alert

def generate_and_send_summary():
    labeled_path = os.path.join(base_dir, "data", "labeled_setups.csv")
    if not os.path.exists(labeled_path):
        print(f"Error: Labeled setups file not found at {labeled_path}")
        return False
        
    df = pd.read_csv(labeled_path)
    
    # Prune to latest 1000 for stats if needed
    if len(df) > 1000:
        df = df.iloc[-1000:].copy()
        
    total_setups = len(df)
    
    # Classify setups
    wins = df[df['label'] == 1]
    losses = df[df['label'] == 0]
    
    total_wins = len(wins)
    total_losses = len(losses)
    
    winrate = (total_wins / total_setups * 100) if total_setups > 0 else 0.0
    
    # Distinguish between full loss and mitigated/early cut-loss (CHoCH)
    # Mitigated loss is when pnl_relative > -0.5
    if 'pnl_relative' in df.columns:
        mitigated_losses = losses[losses['pnl_relative'] > -0.5]
        full_losses = losses[losses['pnl_relative'] <= -0.5]
    else:
        mitigated_losses = pd.DataFrame()
        full_losses = losses
        
    total_mitigated = len(mitigated_losses)
    total_full_losses = len(full_losses)
    
    # Calculate average PnL Relative
    avg_pnl = df['pnl_relative'].mean() if 'pnl_relative' in df.columns else 0.0
    
    # Get details of setups learned
    setup_counts = df['setup_type'].value_counts()
    fvg_count = setup_counts.get(0, 0)
    ob_count = setup_counts.get(1, 0)
    
    direction_counts = df['direction'].value_counts()
    buy_count = direction_counts.get(1, 0)
    sell_count = direction_counts.get(-1, 0)
    
    # Active Models Info
    model_xgb_path = os.path.join(base_dir, "models", "smc_xgb_classifier.joblib")
    model_lgb_path = os.path.join(base_dir, "models", "smc_lgb_classifier.joblib")
    
    model_status = "❌ Belum Terlatih"
    if os.path.exists(model_xgb_path) and os.path.exists(model_lgb_path):
        model_status = "✅ Aktif (XGBoost & LightGBM Ensemble)"
        
    # Format current date
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare message for Telegram
    msg = (
        f"📊 <b>[MLOps Brain Report] Apa Yang Telah Dipelajari AI</b> 📊\n\n"
        f"Status Otak AI: <b>{model_status}</b>\n"
        f"Update Terakhir: <code>{current_time_str}</code>\n\n"
        f"📈 <b>Statistik Database Latih (Windowing 1000 Setup):</b>\n"
        f"• Total Setup Teranalisis: <b>{total_setups} setups</b>\n"
        f"• Winrate Rata-rata Setup: <b>{winrate:.2f}%</b>\n"
        f"• Rata-rata PnL Relative: <b>{avg_pnl:+.2f} R</b>\n"
        f"• Setup FVG (Fair Value Gap): <b>{fvg_count}</b>\n"
        f"• Setup OB (Order Block): <b>{ob_count}</b>\n"
        f"• Setup Arah BUY: <b>{buy_count}</b> | Arah SELL: <b>{sell_count}</b>\n\n"
        f"🧠 <b>Sistem Bobot Gaji & Hukuman (Sample Weighting):</b>\n"
        f"1️⃣ <b>🏆 Profit Besar (Win):</b>\n"
        f"   - Jumlah: <b>{total_wins} setups</b>\n"
        f"   - Bobot Latih: <code>2.00</code> (Prioritas Tinggi)\n"
        f"   - <i>AI dipaksa meniru setup sukses ini.</i>\n\n"
        f"2️⃣ <b>🛡️ Cut-Loss CHoCH (Mitigated Loss):</b>\n"
        f"   - Jumlah: <b>{total_mitigated} setups</b>\n"
        f"   - Bobot Latih: <code>0.50</code> (Reinforcement Positif)\n"
        f"   - <i>AI diajarkan bahwa keluar lebih awal adalah keputusan cerdas untuk meminimalkan risiko.</i>\n\n"
        f"3️⃣ <b>💀 Loss Konyol (Full Loss):</b>\n"
        f"   - Jumlah: <b>{total_full_losses} setups</b>\n"
        f"   - Bobot Latih: <code>1.50</code> (Penalty Standar)\n"
        f"   - <i>AI diajarkan menghindari area dengan setup gagal ini.</i>\n\n"
        f"🚨 <i>Sistem MLOps Anda sekarang membatasi data latih maksimal 1000 setups terbaru agar bot tetap sensitif dengan market regime saat ini.</i>"
    )
    
    print("Sending Telegram Alert...")
    success = send_telegram_alert(msg)
    if success:
        print("Telegram alert sent successfully.")
    else:
        print("Failed to send Telegram alert.")
    return success

if __name__ == "__main__":
    generate_and_send_summary()
