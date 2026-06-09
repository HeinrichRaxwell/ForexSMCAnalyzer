# 🤖 Forex SMC Analyzer & Machine Learning Filter (XAUUSDm)

Sistem asisten trading algoritmik berbasis **Smart Money Concepts (SMC)**, **Pinescript Volume Momentum (FLoOP Pro)**, dan **Machine Learning Ensemble (XGBoost & LightGBM)** untuk menyaring, mengeksekusi, dan mempelajari transaksi secara otomatis pada instrumen **Gold (XAUUSD)** melalui integrasi **MetaTrader 5 (MT5)**.

---

## 📌 Pendahuluan & Visi Bot
Pasar emas (XAUUSD) terkenal dengan likuiditas tinggi, manipulasi harga (*liquidity sweeps*), dan pergeseran tren yang cepat. Menggunakan sistem rule-based SMC saja seringkali menghasilkan *false breakout* karena noise volatilitas. 

Bot ini diciptakan untuk memecahkan masalah tersebut dengan menggabungkan **analisis struktural presisi (SMC)**, **konfirmasi volume institusional (FLoOP Pro)**, dan **kecerdasan buatan (Machine Learning)** sebagai filter probabilitas. Bot ini tidak hanya mengeksekusi order, tetapi juga **belajar dari kesalahannya sendiri secara real-time** melalui Feedback Loop MLOps langsung dari akun MetaTrader 5 Anda.

---

## 🛠️ Arsitektur Utama Bot (Core Engine)

Sistem ini terbagi menjadi 4 lapisan utama yang bekerja secara sinergis:

```
[ PASAR LIVE XAUUSD (MT5) ]
            │
            ▼ (1. Multi-Timeframe SMC Engine)
 [Swing, BOS/CHoCH, OB, FVG, Rejections]
            │
            ▼ (2. Volume & Momentum Filter - FLoOP Pro)
  [Pinescript Volume, KNN, POC Volume Profile]
            │
            ▼ (3. Machine Learning Filter Layer)
[Ensemble XGBoost + LightGBM (Probabilitas >= 70%)]
            │
  ┌─────────┴─────────┐
  ▼ (Sinyal Lolos)    ▼ (Sinyal Lemah)
[Eksekusi MT5 Live]   [FILTERED (Diabaikan)]
  │
  └─────────┬─────────┘
            ▼ (4. MLOps Self-Learning Loop)
[Database Latih (Windowing 1000)] <──> [Champion vs Challenger Gate (Accuracy)]
```

---

## 📂 Struktur Direktori & Fungsionalitas File

Berikut adalah pemetaan detail fungsi seluruh berkas yang ada di dalam proyek:

### 📁 1. Folder `src/` (Core Logic)
*   [`src/smc_detector.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/smc_detector.py):  
    Mengimplementasikan seluruh logika pendeteksian pola SMC. Berisi fungsi `detect_swing_points` untuk melacak titik fractal tinggi/rendah, `detect_structures` untuk BOS dan CHoCH, serta `detect_fvg_and_ob` untuk menemukan Order Block, Fair Value Gap (FVG), dan menghitung koordinat Fibonacci pada area tersebut.
*   [`src/rejection_detector.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/rejection_detector.py):  
    Berisi fungsi `detect_rejection_at_level` untuk memverifikasi apakah harga telah melakukan sentuhan (*touch*) dan meninggalkan ekor (*wick*) penolakan sebesar $\ge 50\%$ di area entry setup sebelum pending order diisi.
*   [`src/model_trainer.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/model_trainer.py):  
    Melatih model ensemble XGBoost dan LightGBM. Bertanggung jawab atas pembagian data latih/uji (80/20), perhitungan berat sampel (*Sample Weighting*), data windowing 6 bulan, pemotongan dataset ke 1000 setups terbaru, serta gerbang validasi Champion vs Challenger.
*   [`src/inference.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/inference.py):  
    Menggunakan model `.joblib` untuk memprediksi probabilitas sinyal live. Skrip ini juga melacak transaksi closed dari MT5 history, memproses perhitungan `pnl_relative` aktual, menyinkronkan database latih, dan menyusun teks penjelasan teknis untuk notifikasi Telegram.
*   [`src/execution.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/execution.py):  
    Mengendalikan interaksi langsung dengan MetaTrader 5 API. Mengatur pengiriman pending order limit, market order instan, penentuan lot dinamis berdasarkan resiko akun, modifikasi trailing Stop Loss, serta pembatalan pending order yang tidak lagi valid (*mitigated/invalidated*).
*   [`src/scanner_worker.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/scanner_worker.py):  
    Loop utama background worker yang berjalan tanpa henti. Setiap menit, ia menghubungkan MT5, mensinkronisasikan riwayat transaksi (feedback loop), memicu retraining, melakukan pemindaian multi-timeframe, menyaring sinyal lewat ML, dan mengeksekusi order.
*   [`src/labeler.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/labeler.py):  
    Melakukan pelabelan transaksi pada data historis CSV. Menghasilkan target `label` (Win=1, Loss=0) dan mengukur 26 fitur entry pasar secara simulasi (termasuk ATR, killzone, data volume, dll.) untuk disimpan ke `data/labeled_setups.csv`.
*   [`src/data_loader.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/data_loader.py):  
    Mengatur inisialisasi koneksi MetaTrader 5 dan mengunduh data lilin (OHLCV) historis broker untuk setiap timeframe.
*   [`src/data_collector.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/data_collector.py):  
    Alat CLI (argparse) untuk mengunduh lilin historis dalam skala besar (misal 50.000 bar) dari MT5 untuk keperluan inisialisasi database latih awal.
*   [`src/backtester.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/backtester.py):  
    Melakukan pengujian simulasi trading historis (backtest) secara offline menggunakan data CSV untuk mengevaluasi winrate kasar vs winrate terfilter ML.
*   [`src/main.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/main.py):  
    Skrip konsol visualisasi terpadu. Memetakan grafik candlestick lengkap dengan penanda level BOS/CHoCH, FVG Fibo levels, Order Blocks, dan menyimpan grafiknya sebagai file gambar PNG.
*   [`src/telegram_bot.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/telegram_bot.py):  
    Wrapper API Telegram untuk mengirim pesan teks HTML dan visualisasi grafik PNG ke Telegram.

### 📁 2. Folder `src/indicators/` (Indikator Kustom FLoOP Pro)
*   [`src/indicators/floop.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/indicators/floop.py):  
    Diterjemahkan dari TradingView Pinescript. Menghitung tren volume institusional (Bullish/Bearish) dan mengukur kekuatan momentum volume breakout.
*   [`src/indicators/knn_classifier.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/indicators/knn_classifier.py):  
    Melakukan klasifikasi terdekat (KNN) secara offline pada harga untuk memprediksi arah pergerakan pasar jangka pendek.
*   [`src/indicators/pivots.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/indicators/pivots.py):  
    Menghitung level Pivot Point harian (Standard, Fibonacci, Classic, Woodie) sebagai support/resistance dinamis institusi.
*   [`src/indicators/volume_clusters.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/src/indicators/volume_clusters.py):  
    Mengelompokkan data harga-volume menggunakan K-Means Clustering untuk memetakan level Point of Control (POC) volume profil pasar.

### 📁 3. Folder `tests/` (Test Suite Komprehensif)
*   [`tests/test_fibo_detector.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/tests/test_fibo_detector.py): Menguji kalkulasi level Fibonacci pada FVG.
*   [`tests/test_rejection.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/tests/test_rejection.py): Memverifikasi deteksi candle wick rejection 50% di key level.
*   [`tests/test_model_trainer.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/tests/test_model_trainer.py): Menguji retraining, Data Windowing, dan Champion vs Challenger.
*   [`tests/test_inference.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/tests/test_inference.py): Menguji kesesuaian prediksi probabilitas ensemble.
*   [`tests/test_active_trade_management.py`](file:///C:/Users/WINDOWS%2011%20PRO/forex-smc-analyzer/tests/test_active_trade_management.py): Memverifikasi penutupan dan trailing proteksi order.
*   *Dan 15 file test unit lainnya untuk menjamin keandalan sistem.*

---

## 📐 2. Detail Strategi Deteksi SMC & Level Fibonacci FVG

Deteksi Fair Value Gap (FVG) didasarkan pada ketidakseimbangan harga dalam pola 3 lilin berturut-turut (Candle 1, Candle 2, Candle 3):

```
BULLISH FVG (BUY SETUP)                  BEARISH FVG (SELL SETUP)

   Candle 3 (High)  ───┬───                 Candle 1 (Low)   ───┬───
                       │ [GAP]                                  │ [GAP]
   Candle 1 (Low)   ───┴───                 Candle 3 (High)  ───┴───
```

### A. Formula Batas FVG & Fibonacci Candle 2
Fibonacci digambar di sepanjang tubuh Candle 2 (middle candle) sebagai penentu area diskon entri:
*   **Untuk Bullish FVG:**
    *   $\text{FVG\_Bottom (Candle 1 High)} = \text{High}_{i-2}$
    *   $\text{FVG\_Top (Candle 3 Low)} = \text{Low}_{i}$
    *   $\text{Fibo}_{1.0} = \text{Low}_{i-1}$ (Low Candle 2)
    *   $\text{Fibo}_{0.0} = \text{High}_{i-1}$ (High Candle 2)
    *   $\text{Fibo}_{0.5} \text{ (Midpoint Entry)} = \text{Fibo}_{0.0} - 0.5 \times (\text{Fibo}_{0.0} - \text{Fibo}_{1.0})$
    *   $\text{Fibo}_{0.618} \text{ (Golden Pocket Entry)} = \text{Fibo}_{0.0} - 0.618 \times (\text{Fibo}_{0.0} - \text{Fibo}_{1.0})$
    *   $\text{Stop Loss (SL)} = \text{FVG\_Bottom} - \text{Buffer}$
*   **Untuk Bearish FVG:**
    *   $\text{FVG\_Top (Candle 1 Low)} = \text{Low}_{i-2}$
    *   $\text{FVG\_Bottom (Candle 3 High)} = \text{High}_{i}$
    *   $\text{Fibo}_{1.0} = \text{High}_{i-1}$ (High Candle 2)
    *   $\text{Fibo}_{0.0} = \text{Low}_{i-1}$ (Low Candle 2)
    *   $\text{Fibo}_{0.5} \text{ (Midpoint Entry)} = \text{Fibo}_{0.0} + 0.5 \times (\text{Fibo}_{1.0} - \text{Fibo}_{0.0})$
    *   $\text{Fibo}_{0.618} \text{ (Golden Pocket Entry)} = \text{Fibo}_{0.0} + 0.618 \times (\text{Fibo}_{1.0} - \text{Fibo}_{0.0})$
    *   $\text{Stop Loss (SL)} = \text{FVG\_Top} + \text{Buffer}$

### B. Nilai Buffer Dinamis
Buffer ditentukan berdasarkan ukuran pip simbol instrumen:
$$\text{Buffer} = 20 \text{ pips} \times \text{PipMultiplier}$$
*   Untuk **XAUUSD / Gold**: $\text{PipMultiplier} = 0.1 \implies \text{Buffer} = 2.0 \text{ USD}$.
*   Untuk JPY Pairs: $\text{PipMultiplier} = 0.01 \implies \text{Buffer} = 0.20 \text{ USD}$.
*   Untuk Forex Major: $\text{PipMultiplier} = 0.0001 \implies \text{Buffer} = 0.0020 \text{ USD}$.

### C. Opsi Eksekusi Order (Limit Setup)
Bot mengirimkan dua opsi order limit ke pasar untuk memaksimalkan peluang *risk-to-reward* (R:R):
1.  **Option A (Midpoint):**
    *   **Entry:** $\text{Fibo}_{0.5}$
    *   **Take Profit (TP 1):** $\text{Fibo}_{0.0}$ (Invalidation level)
    *   **Stop Loss (SL):** $\text{FVG\_SL}$
2.  **Option B (Golden Pocket):**
    *   **Entry:** $\text{Fibo}_{0.618}$
    *   **Take Profit (TP 1):** $\text{Fibo}_{0.0}$
    *   **Stop Loss (SL):** $\text{FVG\_SL}$
    *   *Catatan:* Opsi B memberikan R:R yang jauh lebih sempit sehingga profitabilitas relatif per R-multiple sangat tinggi.

---

## ⚡ 3. Strategi Konfirmasi Rejection & Multi-Timeframe (MTF)

### A. Logika Rejection di Key Level
Sebelum eksekusi pending order disentuh, harga harus menunjukkan penolakan (*wick rejection*) pada area level entri pada 5 bar terakhir.
*   **Bullish Rejection (Touch & Wick Upward):**
    *   Syarat Sentuh: $\text{Low} \le \text{Entry Level} \le \max(\text{Open}, \text{Close})$
    *   Shadow Bawah: $\text{LowerShadow} = \min(\text{Open}, \text{Close}) - \text{Low}$
    *   Range Total: $\text{TotalRange} = \text{High} - \text{Low}$
    *   Logika Penolakan: $\frac{\text{LowerShadow}}{\text{TotalRange}} \ge 0.5 \quad (50\% \text{ Rejection Wick})$
*   **Bearish Rejection (Touch & Wick Downward):**
    *   Syarat Sentuh: $\min(\text{Open}, \text{Close}) \le \text{Entry Level} \le \text{High}$
    *   Shadow Atas: $\text{UpperShadow} = \text{High} - \max(\text{Open}, \text{Close})$
    *   Range Total: $\text{TotalRange} = \text{High} - \text{Low}$
    *   Logika Penolakan: $\frac{\text{UpperShadow}}{\text{TotalRange}} \ge 0.5 \quad (50\% \text{ Rejection Wick})$

### B. Prioritas Multi-Timeframe (MTF Alignment)
1.  Bot memantau timeframe **H1, H4, dan D1** untuk mencari area FVG unmitigated yang aktif.
2.  Saat setup terdeteksi pada timeframe eksekusi (**M15 / M30**):
    *   Bot mengecek apakah $\text{Entry Price}$ setup M15/M30 berada di dalam koordinat $\text{FVG\_Bottom}$ hingga $\text{FVG\_Top}$ dari FVG timeframe besar (H1/H4/D1) dengan arah tren yang sama.
    *   Jika **Ya**, setup dilabeli sebagai **`htf_prioritized = True`** (Sinyal prioritas institusi super).

---

## 📊 4. Integrasi Volume Momentum (FLoOP Pro & POC)

Diadopsi dari TradingView Pinescript ke Python:

### A. Tren & Sinyal FLoOP
*   Bot menghitung momentum volume saat entry. Jika $\text{FLoOP Trend} \neq \text{Direction}$, setup ditolak oleh filter kualitas dasar karena melawan arah aliran dana institusi besar.
*   `floop_strength` mewakili volume breakout. Nilai $> 5.0$ menandakan momentum breakout yang solid.

### B. K-Means Volume Profile (POC)
*   200 candle terakhir dipisah ke dalam $k=5$ cluster harga terbobot volume transaksi.
*   Cluster harga dengan volume terbesar dideklarasikan sebagai **Point of Control (POC)**.
*   Jarak entri terhadap POC dihitung:
    $$\text{dist\_entry\_to\_poc} = \frac{|\text{Entry} - \text{POC}|}{\text{POC}}$$
*   Semakin dekat jarak entri ke POC ($\text{dist\_entry\_to\_poc} \le 0.002$), semakin tinggi kualitas setup karena didukung oleh konsentrasi transaksi institusi.

---

## 🤖 5. Filter Machine Learning & Probabilitas

### Ensemble Classifier
Model terdiri dari gabungan **XGBoost** dan **LightGBM**.
*   **Input Fitur:** 26 fitur entry pasar (Jam trading, volatilitas ATR, data FLoOP, rasio risiko ATR, kedekatan Pivot & POC).
*   **Anti Data Leakage:** Kolom `pnl_relative` dan target `label` dikeluarkan secara ketat dari fitur latihan untuk mencegah **Data Leakage**.
*   **Ambang Batas Kelulusan:** Sinyal disaring dengan ambang batas probabilitas optimal (misal $\ge 70\%$) sebagai sinyal High Confidence.

---

## 🧠 6. Siklus MLOps & Self-Learning (Feedback Loop)

### A. Aturan Umpan Balik & Pembobotan Sampel (Sample Weighting)
Saat posisi close di MT5, profit/loss bersih dihitung dan diubah menjadi `pnl_relative` aktual:
$$\text{pnl\_relative} = \frac{\text{Harga Close} - \text{Harga Entry}}{\text{Harga Entry} - \text{Harga SL}} \quad (\text{Skala R-Multiple})$$
Setiap trade dimasukkan ke `labeled_setups.csv` dengan bobot latih berikut:
1.  **🏆 WIN / Profit Besar (Label 1):** Bobot = **`2.00`** (Memaksa AI meniru pola sukses).
2.  **🛡️ Mitigated / Early Cut-Loss (Label 0, PnL > -0.5):** Bobot = **`0.50`** (Reward atas tindakan meminimalkan risiko saat arah tren berbalik).
3.  **💀 Full Loss / Loss Konyol (Label 0, PnL <= -0.5):** Bobot = **`1.50`** (Hukuman agar AI mempelajari pola kegagalan).

### B. Champion vs Challenger Validation Gate & Dynamic Retraining
*   **Data Split:** 80% Training Data, 20% Testing Data.
*   **Kriteria Promosi:** Challenger (model baru) menggantikan Champion (model lama) **hanya jika** Akurasi Ujian Challenger $\ge$ Akurasi Ujian Champion. Jika tidak, challenger dibuang.
*   **Data Windowing:** Data berusia $> 6$ bulan otomatis dihapus, dan dataset latih dibatasi maksimal **1000 setups terbaru** agar bot selalu sensitif pada kondisi volatilitas market paling segar (*current market regime*).
*   **Konfigurasi Retraining Dinamis (via `.env`):**
    *   `ML_RETRAIN_THRESHOLD`: Menentukan jumlah trade terakumulasi sebelum retraining dipicu. 
        *   Set ke `1` untuk mode agresif: setiap 1 trade close di MT5 langsung dipelajari dan model di-retrain instan.
        *   Set ke `5` (atau lebih) untuk mode konservatif: menunda retraining hingga terkumpul minimal 5 transaksi baru.
    *   `ML_RETRAIN_ON_WEEKEND`: Jika diset ke `true`, bot akan otomatis melakukan retraining pada hari Sabtu/Minggu (saat market libur) tanpa memperdulikan apakah threshold jumlah trade sudah terpenuhi. Ini memastikan otak AI selalu disegarkan secara matang sebelum sesi perdagangan hari Senin dimulai.

---

## 📱 7. Notifikasi Telegram & Explainability

Setiap kali posisi ditutup di MT5, AI akan menganalisis parameter entry dan mengidentifikasi penyebab hasil transaksi (Win, Loss, atau Cut-loss) berdasarkan logika Pinescript & SMC:

```
🏆 [SMC Trade Closed] Option A (0.50) Selesai 🏆

• Ticket: #82713098
• Timeframe: M15
• Setup: BULL (Fair Value Gap)
• Entry Price: 2385.120
• SL | TP: 2383.620 | 2388.120
• Hasil Posisi: 🟢 WIN (PROFIT)
• Net PnL Riil: +300,000 IDR
• PnL Relative: +2.00 R

🏆 Penyebab Profit: Trade berhasil mencapai TP karena didukung oleh keselarasan struktur SMC dan momentum volume FLoOP dan rebound harga pada level POC/Pivot broker.

🧠 Pelajaran AI (Bobot Latih = 2.00):
AI memperkuat memori kesuksesan setup ini. Model mempelajari bahwa kombinasi fitur teknis ini (terutama keselarasan volume FLoOP & SMC) adalah pola pemenang. AI akan memprioritaskan setup serupa dalam scanning pasar berikutnya.
```

---

## 🚀 8. Instalasi & Konfigurasi Operasional

### A. Prasyarat Sistem
*   Windows OS (disarankan Windows 10/11)
*   Python 3.10+
*   Terminal MetaTrader 5 (Terhubung ke akun broker aktif, misal: Exness/XM)
*   Aktifkan menu *Allow Algo Trading* dan *Allow WebRequest* pada MetaTrader 5.

### B. Instalasi Dependensi
Clone repository ini dan jalankan perintah instalasi berikut:
```bash
pip install -r requirements.txt
```

### C. Konfigurasi Environment (`.env`)
Buat berkas `.env` pada root direktori proyek Anda:
```env
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID
MT5_LOGIN=YOUR_MT5_LOGIN_ID
MT5_PASSWORD=YOUR_MT5_PASSWORD
MT5_SERVER=YOUR_MT5_SERVER_NAME
```

### D. Menjalankan Unit Test
Pastikan seluruh sistem berjalan dengan benar (100% Passed):
```bash
pytest
```

### E. Retraining Manual Otak AI
Untuk melatih ulang model XGBoost & LightGBM menggunakan database latih Anda saat ini:
```bash
python src/model_trainer.py
```

### F. Menjalankan Bot Live (Scanner Worker)
Jalankan perintah berikut untuk mengaktifkan bot pemantau pasar live:
```bash
python src/scanner_worker.py --symbol XAUUSD --loop --interval 1 --threshold 0.50
```
*   `--symbol`: Simbol target broker (misal: `XAUUSD` or `XAUUSDm`).
*   `--loop`: Berjalan secara terus-menerus.
*   `--interval`: Interval pemindaian pasar (dalam menit).
*   `--threshold`: Batas probabilitas ML minimal untuk memicu alert/order (misal: `0.50` sampai `0.70`).
