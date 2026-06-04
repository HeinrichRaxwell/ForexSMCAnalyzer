# Phase 2: Machine Learning Filter & Self-Learning Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Membangun lapisan kecerdasan buatan (Machine Learning) menggunakan XGBoost untuk menyaring setup SMC dari Phase 1 agar menghasilkan winrate tinggi (>75%) melalui proses labeling trade historis, ekstrasi fitur pasar, dan sistem pelatihan ulang mandiri (self-learning loop).

**Architecture:** Model klasifikasi biner dilatih menggunakan data XAUUSD historis 1-2 tahun ke belakang. Setiap setup SMC disimulasikan sebagai posisi trading dengan aturan Stop Loss (SL) dan Take Profit (TP) tertentu; jika menyentuh TP dulu maka dilabeli `1` (Win), dan jika menyentuh SL dulu dilabeli `0` (Loss). XGBoost akan memprediksi probabilitas keberhasilan setup baru berdasarkan fitur-fitur kondisi pasar saat entry.

**Tech Stack:** Scikit-Learn, XGBoost, Joblib, Pandas, NumPy, MetaTrader5.

---

### Task 1: Historical Data Collector (1-2 Years)

**Files:**
- Create: `forex-smc-analyzer/src/data_collector.py`
- Test: `forex-smc-analyzer/tests/test_data_collector.py`

**Step 1: Implement data_collector.py**
Fungsi untuk mengunduh data candlestick M15 historis (misal 50,000 bar atau 2 tahun ke belakang) dari MT5 Exness dan menyimpannya sebagai file CSV di `data/historical_xauusd.csv`.
```python
import os
import MetaTrader5 as mt5
import pandas as pd
from src.data_loader import connect_mt5, fetch_historical_data

def download_bulk_data(symbol="XAUUSD", timeframe=15, num_candles=50000, output_dir="data"):
    """
    Mengunduh data candle historis skala besar dari MT5 Exness.
    """
    if not connect_mt5():
        raise RuntimeError("Gagal menghubungkan ke terminal MT5.")
        
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"historical_{symbol.lower()}.csv")
    
    print(f"Mengunduh {num_candles} candle {symbol}...")
    df = fetch_historical_data(symbol, timeframe, num_candles)
    
    df.to_csv(filepath, index=False)
    print(f"Data historis berhasil disimpan di {filepath} (Size: {df.shape[0]} bar)")
    mt5.shutdown()
    return filepath
```

---

### Task 2: Trade Simulation & Dataset Labeling Engine

**Files:**
- Create: `forex-smc-analyzer/src/labeler.py`
- Test: `forex-smc-analyzer/tests/test_labeler.py`

**Step 1: Implement Trade Simulator & Labeler**
Mendeteksi setup SMC pada data historis, mensimulasikan order entri pada FVG atau OB terdekat dengan rasio Risk:Reward 1:2 (SL di bawah OB/Swing point, TP = 2x SL), melacak bar ke depan sampai menyentuh TP (Label: `1`) atau SL (Label: `0`).
```python
import pandas as pd
import numpy as np

def label_smc_setups(df):
    """
    Mensimulasikan trade pada setiap setup yang terdeteksi untuk membuat target label (0 atau 1).
    """
    df = df.copy()
    setups = []
    
    for i in range(len(df) - 50): # Sisakan ruang di depan untuk pengecekan TP/SL
        row = df.iloc[i]
        
        # Cek jika ada setup Bullish OB/FVG
        is_bullish_setup = False
        entry_price = np.nan
        sl_price = np.nan
        
        # Jika terdeteksi Bullish OB baru
        if 'OB_Type' in df.columns and row['OB_Type'] == 'BULLISH':
            is_bullish_setup = True
            entry_price = row['OB_Top']       # Entry di batas atas OB
            sl_price = row['OB_Bottom'] - 0.5  # SL di bawah OB dengan buffer
            
        elif 'FVG_Type' in df.columns and row['FVG_Type'] == 'BULLISH':
            is_bullish_setup = True
            entry_price = row['FVG_Top']       # Entry di batas atas FVG
            sl_price = row['FVG_Bottom'] - 0.5 # SL di bawah batas FVG
            
        if is_bullish_setup and entry_price > sl_price:
            risk = entry_price - sl_price
            tp_price = entry_price + (risk * 2) # RR 1:2
            
            # Cari bar berikutnya apakah menyentuh TP dulu atau SL dulu
            outcome = None
            for j in range(i + 1, len(df)):
                future_low = df['Low'].iloc[j]
                future_high = df['High'].iloc[j]
                
                # Cek jika menyentuh SL dulu
                if future_low <= sl_price:
                    outcome = 0
                    break
                # Cek jika menyentuh TP dulu
                if future_high >= tp_price:
                    outcome = 1
                    break
                    
            if outcome is not None:
                # Ekstrak fitur pasar saat entri
                features = {
                    'time_of_day': row['time'].hour if isinstance(row['time'], pd.Timestamp) else 12,
                    'day_of_week': row['time'].dayofweek if isinstance(row['time'], pd.Timestamp) else 0,
                    'spread_fvg': (row['FVG_Top'] - row['FVG_Bottom']) if not pd.isna(row['FVG_Top']) else 0,
                    'trend_direction': row['Trend'] if 'Trend' in row else 1,
                    'label': outcome
                }
                setups.append(features)
                
    return pd.DataFrame(setups)
```

---

### Task 3: Feature Engineering & XGBoost Classifier Training

**Files:**
- Create: `forex-smc-analyzer/src/model_trainer.py`
- Test: `forex-smc-analyzer/tests/test_model_trainer.py`

**Step 1: Implement Feature Engineering & Model Training**
Membaca dataset berlabel, membagi data menjadi Train & Test set, melatih model **XGBoost Classifier**, mengoptimasi tingkat presisi (precision - meminimalkan false signal), dan menyimpan model.
```python
import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score

def train_xgboost_filter(labeled_data_path="data/labeled_setups.csv", model_dir="models"):
    """
    Melatih XGBoost Classifier untuk memprediksi probabilitas keberhasilan setup SMC.
    """
    df = pd.read_csv(labeled_data_path)
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Inisialisasi model XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Pengecekan performa
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    print(f"XGBoost Model Precision: {precision * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    # Simpan model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "smc_xgb_classifier.joblib")
    joblib.dump(model, model_path)
    print(f"Model berhasil disimpan di: {model_path}")
    return model_path
```

---

### Task 4: Signal Filtering Engine with Confidence & Self-Learning Loop

**Files:**
- Create: `forex-smc-analyzer/src/inference.py`
- Modify: `forex-smc-analyzer/src/main.py`

**Step 1: Implement Predictor & Signal Filter**
Modul untuk mengevaluasi setup baru secara real-time. Jika probabilitas sukses model XGBoost < 80% (threshold confidence tinggi), maka sinyal difilter/dibuang.
```python
import joblib
import pandas as pd
import numpy as np

def predict_setup_probability(features_dict, model_path="models/smc_xgb_classifier.joblib"):
    """
    Memprediksi skor probabilitas sukses (winrate) untuk setup baru.
    """
    if not os.path.exists(model_path):
        # Jika model belum dilatih, kembalikan default confidence 100% (non-filtered)
        return 1.0
        
    model = joblib.load(model_path)
    # Konversi dictionary fitur ke dataframe baris tunggal
    df_features = pd.DataFrame([features_dict])
    
    # Prediksi probabilitas sukses (class 1)
    proba = model.predict_proba(df_features)[0][1]
    return float(proba)
```

**Step 2: Self-Learning Feedback Loop Script**
Membuat mekanisme retraining periodik. Jika ada sinyal riil yang dikirim (misal ke Telegram) kemudian trade menyentuh SL/TP, kita meng-append hasil riil tersebut ke file data latih dan melatih ulang XGBoost secara otomatis.
