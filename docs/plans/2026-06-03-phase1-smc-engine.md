# Phase 1: SMC Detection Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Membangun core engine berbasis Python untuk mengunduh data XAUUSD secara real-time dari MetaTrader 5 Exness dan mendeteksi struktur SMC/ICT (Swing High/Low, BOS, CHoCH, FVG, dan Order Blocks) dengan akurasi 100% secara matematis.

**Architecture:** Engine ini menggunakan pemrograman fungsional/objek dengan library `pandas` dan `numpy` untuk memproses data OHLCV historis. Deteksi struktur dilakukan secara bertahap mulai dari Swing Points, diikuti oleh transisi struktur (BOS/CHoCH), dan penandaan area imbalance (FVG) serta supply/demand (Order Blocks).

**Tech Stack:** Python 3.10+, MetaTrader5 Library, Pandas, NumPy, Matplotlib/Plotly (untuk visualisasi debug).

---

### Task 1: Setup Project Environment & MT5 Connection

**Files:**
- Create: `forex-smc-analyzer/requirements.txt`
- Create: `forex-smc-analyzer/src/data_loader.py`
- Test: `forex-smc-analyzer/tests/test_data_loader.py`

**Step 1: Write requirements.txt**
Tulis library yang dibutuhkan untuk Fase 1.
```text
MetaTrader5>=5.0.45
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pytest>=7.0.0
python-dotenv>=1.0.0
```

**Step 2: Implement data_loader.py**
Buat modul untuk inisialisasi koneksi ke MT5 Exness dan fungsi mengunduh data candlestick.
```python
import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def connect_mt5():
    """Menghubungkan ke terminal MetaTrader 5 yang sedang berjalan."""
    if not mt5.initialize():
        print("Inisialisasi MT5 gagal, error code =", mt5.last_error())
        return False
    return True

def fetch_historical_data(symbol, timeframe, num_candles):
    """
    Mengambil data candlestick historis dari MT5.
    timeframe: mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, dll.
    """
    rates = mt5.copy_rates_from_now(symbol, timeframe, num_candles)
    if rates is None or len(rates) == 0:
        raise ValueError(f"Gagal mengambil data untuk {symbol}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    return df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]
```

**Step 3: Write tests for connection**
Buat test sederhana di `tests/test_data_loader.py` untuk memverifikasi koneksi. (Catatan: Tes ini membutuhkan terminal MT5 aktif).

---

### Task 2: Swing Points Detection

**Files:**
- Create: `forex-smc-analyzer/src/smc_detector.py`
- Test: `forex-smc-analyzer/tests/test_swing_detector.py`

**Step 1: Implement detect_swing_points**
Mendeteksi Swing High (puncak lokal) dan Swing Low (lembah lokal) dengan window filter (misal 5 candle: candle tengah harus lebih tinggi/rendah dari 2 candle kiri dan 2 candle kanan).
```python
import numpy as np
import pandas as pd

def detect_swing_points(df, window=5):
    """
    Mendeteksi Swing High dan Swing Low.
    df: DataFrame dengan kolom High & Low
    window: Jumlah candle untuk konfirmasi swing (harus ganjil, misal 5)
    """
    df = df.copy()
    half = window // 2
    
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan
    
    for i in range(half, len(df) - half):
        high_range = df['High'].iloc[i - half : i + half + 1]
        low_range = df['Low'].iloc[i - half : i + half + 1]
        
        # Cek jika index tengah adalah nilai tertinggi/terendah
        if df['High'].iloc[i] == high_range.max():
            df.loc[df.index[i], 'Swing_High'] = df['High'].iloc[i]
            
        if df['Low'].iloc[i] == low_range.min():
            df.loc[df.index[i], 'Swing_Low'] = df['Low'].iloc[i]
            
    return df
```

**Step 2: Test swing detection**
Tulis uji coba unit test dengan data dummy untuk memastikan algoritum mendeteksi puncak dan lembah secara tepat.

---

### Task 3: Market Structure (BOS & CHoCH) Detection

**Files:**
- Modify: `forex-smc-analyzer/src/smc_detector.py`
- Test: `forex-smc-analyzer/tests/test_structure_detector.py`

**Step 1: Implement detect_structures**
Mendeteksi Break of Structure (BOS) dan Change of Character (CHoCH) berdasarkan pergerakan harga yang menembus Swing Points sebelumnya dengan *candle body close* (bukan shadow/wick).
```python
def detect_structures(df):
    """
    Mendeteksi BOS dan CHoCH.
    Menyimpan history swing high/low terakhir untuk memeriksa penembusan (break).
    """
    df = df.copy()
    df['BOS'] = np.nan
    df['CHoCH'] = np.nan
    
    last_high = None
    last_low = None
    current_trend = 1 # 1: Bullish, -1: Bearish
    
    # Loop untuk melacak struktur pasar secara dinamis
    for idx, row in df.iterrows():
        # Update swing point terakhir yang terdeteksi
        if not pd.isna(row['Swing_High']):
            last_high = row['Swing_High']
        if not pd.isna(row['Swing_Low']):
            last_low = row['Swing_Low']
            
        # Cek penembusan untuk Bullish Trend
        if current_trend == 1 and last_high is not None:
            if row['Close'] > last_high:
                # Harga break swing high terakhir
                df.at[idx, 'BOS'] = last_high
                last_high = None # Reset swing high yang sudah ter-break
                
        # Cek penembusan untuk Bearish Trend (Reversal / CHoCH)
        if current_trend == 1 and last_low is not None:
            if row['Close'] < last_low:
                # Tren berubah dari Bullish ke Bearish (CHoCH)
                df.at[idx, 'CHoCH'] = last_low
                current_trend = -1
                last_low = None
                
        # Cek penembusan untuk Bearish Trend (BOS Bearish)
        if current_trend == -1 and last_low is not None:
            if row['Close'] < last_low:
                df.at[idx, 'BOS'] = last_low
                last_low = None
                
        # Cek penembusan untuk Bullish Reversal (CHoCH Bullish)
        if current_trend == -1 and last_high is not None:
            if row['Close'] > last_high:
                df.at[idx, 'CHoCH'] = last_high
                current_trend = 1
                last_high = None
                
    return df
```

---

### Task 4: Fair Value Gaps (FVG) & Order Blocks (OB)

**Files:**
- Modify: `forex-smc-analyzer/src/smc_detector.py`
- Test: `forex-smc-analyzer/tests/test_imbalances.py`

**Step 1: Implement detect_fvg_and_ob**
Deteksi FVG (imbalance 3 candle) dan Order Blocks (candle sebelum ekspansi kuat). OB yang belum disentuh oleh harga berikutnya ditandai sebagai *Mitigated = False*.
```python
def detect_fvg(df):
    """
    Bullish FVG: High Candle i-2 < Low Candle i
    Bearish FVG: Low Candle i-2 > High Candle i
    """
    df = df.copy()
    df['FVG_Type'] = None
    df['FVG_Top'] = np.nan
    df['FVG_Bottom'] = np.nan
    
    for i in range(2, len(df)):
        # Bullish FVG
        if df['High'].iloc[i-2] < df['Low'].iloc[i] and df['Close'].iloc[i-1] > df['Open'].iloc[i-1]:
            df.at[df.index[i], 'FVG_Type'] = 'BULLISH'
            df.at[df.index[i], 'FVG_Top'] = df['Low'].iloc[i]
            df.at[df.index[i], 'FVG_Bottom'] = df['High'].iloc[i-2]
            
        # Bearish FVG
        elif df['Low'].iloc[i-2] > df['High'].iloc[i] and df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:
            df.at[df.index[i], 'FVG_Type'] = 'BEARISH'
            df.at[df.index[i], 'FVG_Top'] = df['Low'].iloc[i-2]
            df.at[df.index[i], 'FVG_Bottom'] = df['High'].iloc[i]
            
    return df
```

---

### Task 5: Integration & Visualization Script

**Files:**
- Create: `forex-smc-analyzer/src/main.py`

**Step 1: Implement main.py**
Script utama yang menghubungkan seluruh modul: inisialisasi MT5 -> unduh data XAUUSD -> jalankan detektor SMC -> visualisasikan menggunakan Matplotlib dengan plot garis BOS/CHoCH, kotak FVG, dan penanda swing points.
