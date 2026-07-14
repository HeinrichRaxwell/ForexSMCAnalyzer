# Forex SMC Analyzer — XAUUSD Trading Bot

Bot trading otomatis untuk **Gold (XAUUSD)** berbasis Smart Money Concepts (SMC), multi-timeframe analysis, dan Machine Learning (XGBoost + LightGBM). Bot ini mendeteksi Fair Value Gap, Order Block, dan Balanced Price Range, lalu mengeksekusi order langsung ke MetaTrader 5 secara real-time.

---

## Cara Kerja Singkat

```
Market Live XAUUSD
        |
        v  -- Scan tiap 1-5 menit (M30, H1, H4, D1)
  SMC Detector  (FVG, OB, BPR, Swing High/Low, BOS/CHoCH)
        |
        v  -- Filter volume & momentum
  FLoOP Pro + KNN + Volume Profile POC
        |
        v  -- Filter probabilitas
  ML Ensemble  (XGBoost + LightGBM, threshold >= 50%)
        |
   Lolos?  ---Tidak---> Shadow Tracker (dipantau offline)
        |
       Ya
        |
        v  -- Eksekusi langsung
  MetaTrader 5  (Pending limit order, SL/TP otomatis)
        |
        v  -- Setelah trade close
  Feedback Loop  (bot pelajari hasilnya, retrain model sendiri)
```

Bot juga punya sistem **Price Watch Zone** — setelah scan, semua zona yang terdeteksi disimpan. Antara scan berikutnya, bot tetap cek harga tiap sepersekian detik. Kalau harga masuk ke zona, langsung eksekusi tanpa nunggu scan ulang.

---

## Sebelum Mulai

Yang kamu butuhkan:

- Windows 10 atau 11
- Python 3.10 ke atas
- MetaTrader 5 — sudah terhubung ke akun broker (Exness, XM, atau broker lain yang support MT5)
- Akun Telegram (untuk notifikasi bot)

---

## 1. Setup Telegram Bot

Bot ini mengirim notifikasi ke Telegram setiap kali ada sinyal, order dieksekusi, atau trade selesai.

**Langkah membuat bot Telegram:**

1. Buka Telegram, cari **@BotFather**
2. Kirim pesan `/newbot`
3. BotFather akan tanya nama bot — isi terserah, misal `SMC Gold Bot`
4. Setelah itu tanya username bot — harus diakhiri `bot`, misal `smc_gold_xauusd_bot`
5. BotFather akan kasih **token** seperti ini:
   ```
   1234567890:AAFbkUXJdVSf0Xp4H_8ZtaOncZYmJMpI_Yo
   ```
   Simpan token ini, nanti dimasukkan ke `.env`

**Cari Chat ID kamu:**

1. Cari **@userinfobot** di Telegram
2. Kirim `/start`
3. Bot akan balas dengan info akun kamu, termasuk `Id: 1234567890`
4. Itu Chat ID kamu — simpan juga

**Aktifkan bot kamu:**

Sebelum bot bisa kirim pesan ke kamu, kamu harus mulai percakapan dulu:
1. Cari username bot yang baru dibuat tadi di Telegram
2. Tekan **Start**

---

## 2. Setup MetaTrader 5

**Izinkan trading otomatis di MT5:**

1. Buka MetaTrader 5
2. Menu **Tools** > **Options** > tab **Expert Advisors**
3. Centang:
   - `Allow automated trading`
   - `Allow DLL imports`
4. Klik OK

**Aktifkan Auto Trading di toolbar:**

Di toolbar atas MT5, pastikan tombol **Auto Trading** (ikon robot) berwarna hijau/aktif.

**Cari nama server broker kamu:**

1. Di MT5, klik **File** > **Open an account**
2. Atau lihat di pojok kanan bawah MT5 — ada tulisan nama server, misal `Exness-MT5Real8`
3. Nama ini nanti diisi di `.env` (field `MT5_SERVER`)

**Cari login dan password:**

- Login number biasanya dikirim email saat daftar broker
- Password adalah password trading kamu (bukan password login website broker)

---

## 3. Instalasi Bot

**Clone repository:**

```bash
git clone https://github.com/HeinrichRaxwell/ForexSMCAnalyzer.git
cd ForexSMCAnalyzer
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Setup konfigurasi:**

Copy file template dan isi dengan data kamu:

```bash
copy .env.example .env
```

Buka file `.env` dengan text editor (Notepad, VS Code, dll), lalu isi:

```env
# Telegram
TELEGRAM_BOT_TOKEN=1234567890:AAFbkUXJdVSf0Xp4H_8ZtaOncZYmJMpI_Yo
TELEGRAM_CHAT_ID=1234567890

# MT5 — isi sesuai akun broker kamu
MT5_LOGIN=12345678
MT5_PASSWORD=password_trading_kamu
MT5_SERVER=Exness-MT5Real8

# Aktifkan trading otomatis (set True kalau sudah siap live)
MT5_EXECUTE_TRADES=False

# Magic number — angka unik buat bot kamu, ganti kalau mau
MT5_MAGIC_NUMBER=202610
```

> Untuk konfigurasi lain seperti lot size, max trades, trailing stop, dll — sudah ada defaultnya di `.env.example` dengan penjelasan tiap baris.

---

## 4. Jalankan Bot

**Test dulu tanpa eksekusi order (mode monitor):**

```bash
python -m src.scanner_worker --symbol XAUUSD --loop --interval 5 --threshold 0.50
```

Bot akan scan pasar dan kirim notifikasi sinyal ke Telegram kamu, tapi tidak eksekusi order. Pastikan notifikasi masuk dulu sebelum aktifkan trading.

**Aktifkan trading otomatis:**

Setelah yakin sinyal masuk, ubah di `.env`:
```env
MT5_EXECUTE_TRADES=True
```

Lalu jalankan dengan realtime reaction (deteksi entry lebih cepat):

```bash
python -m src.scanner_worker --symbol XAUUSD --loop --interval 5 --threshold 0.50 --realtime-reaction --tick-interval 0.1 --min-reaction-move 0.10
```

Parameter yang bisa diatur:

| Parameter | Default | Keterangan |
|---|---|---|
| `--symbol` | `XAUUSD` | Simbol di MT5. Cek nama persis di Market Watch (bisa `XAUUSDm`, dll) |
| `--interval` | `5` | Interval full scan dalam menit |
| `--threshold` | `0.50` | Confidence minimum ML untuk eksekusi (0.50 = 50%) |
| `--realtime-reaction` | off | Aktifkan pemantauan zona real-time antara scan |
| `--tick-interval` | `1.0` | Seberapa sering cek harga (detik). `0.1` = tiap 100ms |
| `--min-reaction-move` | `0.10` | Minimum pergerakan harga untuk trigger market order ($) |

---

## 5. Cara Bot Trading

Bot memakai strategi **dual entry Fibonacci** di setiap zona yang terdeteksi:

**Option A — Midpoint (Fib 0.5):**
- Entry di titik tengah candle 2 FVG
- Lot lebih kecil
- Stop Loss di bawah/atas zona FVG

**Option B — Golden Pocket (Fib 0.618):**
- Entry lebih dalam ke zona
- Lot lebih besar (karena lebih dekat SL, RR lebih bagus)
- Stop Loss sama dengan Option A

**Take Profit:**
- Default ke Fib 0 (awal swing candle 2) — ini sekitar 100-250 pips untuk setup M30/H1

**Trailing Stop (khusus XAUUSD):**

| Profit | Aksi SL |
|---|---|
| < 80 pips | Tidak ada perubahan, biarkan trade jalan |
| 80 – 149 pips | SL digeser ke BEP + spread buffer (trade jadi risk-free) |
| >= 150 pips | Trailing ladder tiap 50 pip. 150 pips = lock 50 pip, 200 pips = lock 100 pip, dst |

**Confidence Swap:**

Kalau slot order sudah penuh tapi ada sinyal baru dengan confidence lebih tinggi, bot otomatis cancel order lama (yang confidence lebih rendah) dan ganti dengan sinyal baru. Order yang dicancel masuk ke Shadow Tracker untuk dipantau hasilnya.

---

## 6. Notifikasi Telegram yang Dikirim Bot

Bot mengirim beberapa jenis pesan:

- **Sinyal baru terdeteksi** — lengkap dengan entry, SL, TP, confidence, dan info confluence
- **Order dieksekusi** — konfirmasi order berhasil masuk ke MT5
- **Trade close** — hasil trade, PnL, dan pelajaran yang dipetik AI
- **Retraining** — pemberitahuan kalau bot baru belajar dari trade terbaru
- **WatchZone Hit** — kalau harga masuk zona yang dipantau antara scan (entry lebih cepat)
- **Order Eviction** — kalau slot penuh dan order lama diganti yang confidence lebih tinggi

---

## 7. Konfigurasi Penting di `.env`

Beberapa setting yang paling sering perlu diubah:

**Lot size:**
```env
MT5_LOT_SIZE_OPTION_A=0.01   # Lot untuk Fib 0.5 entry
MT5_LOT_SIZE_OPTION_B=0.01   # Lot untuk Fib 0.618 entry
```

Atau aktifkan dynamic lot (otomatis sesuai balance):
```env
MT5_DYNAMIC_LOT_ENABLED=True
MT5_DYNAMIC_LOT_BASE_BALANCE_USD=100    # Balance awal
MT5_DYNAMIC_LOT_BALANCE_STEP_USD=50    # Setiap naik $50, lot naik 1 step
MT5_DYNAMIC_LOT_BASE_LOT=0.01
MT5_DYNAMIC_LOT_STEP_LOT=0.01
MT5_DYNAMIC_LOT_MAX=0.10               # Maksimal lot
```

**Risk management:**
```env
MT5_MAX_CONCURRENT_TRADES=6    # Maksimal trade aktif bersamaan
MT5_MAX_PENDING_ORDERS=8       # Maksimal pending order
MT5_ALLOWED_TIMEFRAMES=M30,H1,H4,D1   # Timeframe yang diizinkan eksekusi
```

**ML confidence:**
```env
ML_ACCEPT_THRESHOLD=0.50       # Minimum confidence untuk kirim sinyal/eksekusi
ML_RETRAIN_THRESHOLD=10        # Retrain setelah berapa trade close
```

---

## 8. Struktur File

```
ForexSMCAnalyzer/
├── src/
│   ├── scanner_worker.py        # Main loop — jalankan ini
│   ├── execution.py             # Kirim order ke MT5, trailing stop
│   ├── smc_detector.py          # Deteksi FVG, OB, BPR, Swing
│   ├── inference.py             # Prediksi ML, feedback loop
│   ├── model_trainer.py         # Retrain XGBoost & LightGBM
│   ├── entry_quality_gate.py    # Filter RSI8, Stochastic, spread
│   ├── realtime_reaction_watcher.py   # Deteksi entry cepat antar scan
│   ├── price_watch_zones.py     # Pre-register zona, cek tiap tick
│   ├── shadow_tracker.py        # Monitor sinyal yang tidak dieksekusi
│   ├── telegram_bot.py          # Kirim notifikasi ke Telegram
│   ├── data_loader.py           # Ambil data OHLCV dari MT5
│   └── indicators/
│       ├── floop.py             # FLoOP Pro volume momentum
│       ├── knn_classifier.py    # KNN directional classifier
│       ├── pivots.py            # Daily pivot points
│       └── volume_clusters.py  # K-Means volume profile (POC)
├── models/
│   ├── smc_lgb_classifier.joblib    # Model LightGBM terlatih
│   ├── smc_xgb_classifier.joblib    # Model XGBoost terlatih
│   └── confidence_calibrator.joblib # Kalibrasi probabilitas
├── data/
│   ├── labeled_setups.csv       # Database training AI
│   ├── sent_signals.json        # State sinyal yang sudah dikirim
│   └── shadow_signals.json      # Pantauan sinyal yang tidak dieksekusi
├── tests/                       # Unit tests
├── .env.example                 # Template konfigurasi
├── requirements.txt
└── README.md
```

---

## 9. Perintah Tambahan

**Retrain model manual:**
```bash
python -m src.model_trainer
```

**Lihat grafik analisis:**
```bash
python -m src.main --symbol XAUUSD --timeframe M30
```

**Jalankan unit test:**
```bash
pytest
```

**Backtest menggunakan data historis:**
```bash
python -m src.backtester
```

---

## 10. Troubleshooting

**Bot tidak connect ke MT5:**
- Pastikan MetaTrader 5 sudah dibuka dan login
- Cek apakah Auto Trading aktif (tombol robot di toolbar)
- Kalau pakai VPS, pastikan MT5 berjalan di user session yang sama dengan bot

**Sinyal tidak masuk Telegram:**
- Cek token dan chat ID di `.env`
- Pastikan kamu sudah `/start` bot Telegram kamu
- Coba jalankan `python -m src.telegram_bot` untuk test kirim pesan

**Bot error saat start:**
- Jalankan `pip install -r requirements.txt` ulang
- Pastikan Python versi 3.10 ke atas

**Order tidak masuk padahal ada sinyal:**
- Cek apakah `MT5_EXECUTE_TRADES=True` di `.env`
- Cek log di console — biasanya ada pesan alasan kenapa tidak dieksekusi (spread terlalu lebar, terlalu banyak order aktif, dll)
- Cek apakah simbol di `--symbol` sama persis dengan yang ada di Market Watch MT5

---

## 11. Analisis Winrate WatchZone & Integrasi Pivot MT5

### A. Laporan Kinerja WatchZone (Hasil Riil Juli 2026)
Analisis riwayat trading riil (276 total closed trades) menunjukkan hasil sebagai berikut:

| Tipe Eksekusi | Total Trades | Winrate | Net Profit | Karakteristik & Rekomendasi |
| :--- | :---: | :---: | :---: | :--- |
| **Standard Limit (Pending)** | 27 | **62.96%** | **+$102.38** | Sangat stabil karena antri pasif di level Fib presisi. |
| **WatchZone (Instant Market)** | 249 | **54.21%** | **-$108.07** | Frekuensi sangat tinggi. Sangat gacor pada OB, namun terseret pada FVG. |

#### **Breakdown Performa WatchZone Berdasarkan Timeframe & Strategi:**
* **Order Block (OB) via WatchZone (Super Gacor):**
  * **M30 OB**: Winrate **93.33%** (15 Trades | Profit +$150.65)
  * **H1 OB**: Winrate **77.78%** (9 Trades | Profit +$65.65)
  * *Rekomendasi*: OB terbukti memiliki akurasi pantulan instan tertinggi saat disentuh harga.
* **Fair Value Gap (FVG) via WatchZone (Risiko Tinggi):**
  * **M30 FVG**: Winrate **51.85%** (Profit -$203.13)
  * **H1 FVG**: Winrate **41.02%** (Profit -$85.65)
  * *Penyebab*: FVG sering langsung ditembus dalam saat market *trending* kuat atau *news*.
* **Daily Timeframe (D1) via WatchZone (Tidak Akurat):**
  * Winrate **39.43%** (71 Trades | Profit -$17.24) karena zona D1 terlalu lebar untuk eksekusi instan tanpa konfirmasi.

---

### B. Integrasi Visualisasi Pivot & SnR di MT5
Sistem visualisasi garis SnR dan Pivot di MetaTrader 5 bekerja dengan cara berikut:
1. **Python Bot**: Menghitung level Pivot Point (PP), Support (S1-S4), dan Resistance (R1-R4) secara periodik dari data D1.
2. **File Ekspor**: Data level ditulis sebagai file terenkripsi JSON ke direktori global MT5 Common:
   `AppData\Roaming\MetaQuotes\Terminal\Common\Files\pivot_levels.json`
3. **MQL5 Indicator**: Indikator kustom di dalam MT5 membaca file `.json` tersebut dan menggambar garis horizontal secara dinamis di chart kamu.
   * *Troubleshooting*: Jika garis SnR/Pivot tidak muncul di terminal MT5 kamu, pastikan **Indikator Pivot Kustom** sudah ditarik (*drag*) dari panel **Navigator** ke chart `XAUUSDm` kamu saat ini.

---

### C. Update Pengamanan & Aturan Main Baru (Juli 2026)
Untuk mengunci profit dan meminimalkan kerugian saat terjadi fluktuasi news, pengamanan berikut telah diimplementasikan:

1. **WatchZone Terbatas Khusus OB & BPR**:
   * Sistem WatchZone sekarang **hanya diizinkan** melakukan entri instan untuk strategi **OB** (Order Block) dan **BPR** (Balanced Price Range) karena akurasinya yang sangat tinggi.
   * Strategi **FVG** hanya diizinkan melalui **Standard Limit (Pending Order)** agar bot mendapatkan harga diskon terdalam (Fib 0.5/0.618) dan membatasi risiko SL.
2. **Enforcement Kebijakan Strategi**:
   * WatchZone sekarang mendeteksi dan memblokir strategi yang masuk daftar hitam di `.env` (seperti `IC`, `SND`, `Swapzone`, `Pivot`).
3. **Proteksi Clustering & Over-exposure**:
   * Menambahkan pemeriksaan jarak posisi aktif searah. Bot tidak akan membuka posisi baru jika sudah ada transaksi aktif dalam rentang **30 pips** untuk timeframe yang sama (`MT5_SAME_TF_PROXIMITY_PIPS`) atau **15 pips** secara umum (`MT5_PENDING_PROXIMITY_PIPS`).
4. **D1 Dinonaktifkan**:
   * Timeframe `D1` telah dihapus dari daftar trading (`MT5_ALLOWED_TIMEFRAMES=M30,H1,H4`) untuk menghindari entri berskala besar yang lambat dan berisiko tinggi.

---

## Catatan

- Bot ini dibuat untuk XAUUSD (Gold). Bisa dipakai untuk pair lain tapi parameter pip multiplier dan trailing stop belum dioptimalkan untuk pair lain.
- Model ML yang disertakan sudah dilatih dengan data nyata. Makin banyak trade yang masuk, makin pintar modelnya (self-learning otomatis).
- Jangan lupa set `MT5_EXECUTE_TRADES=False` dulu saat pertama coba, pantau sinyal dulu beberapa hari sebelum aktifkan trading.
