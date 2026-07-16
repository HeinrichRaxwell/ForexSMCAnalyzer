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
MT5_MAX_PENDING_ORDERS=6       # Maksimal pending order; harus <= MT5_MAX_CONCURRENT_TRADES
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
│   ├── dashboard.py             # UI Dashboard Streamlit interaktif
│   ├── dashboard_data.py        # Pengolahan data telemetri untuk dashboard
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

**Jalankan dashboard Streamlit:**
```bash
streamlit run src/dashboard.py
```
*Dashboard ini menyediakan visualisasi chart interaktif bergaya TradingView, telemetri model AI, status shadow tracking, serta analisis performa winrate secara real-time.*

**Lihat grafik analisis static (Matplotlib):**
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

## 11. Performance Evidence And Downloads

This repository publishes downloadable, reproducible performance evidence in
[`reports/`](reports/README.md). It does not treat historical results as a
profit guarantee.

### Standard Limit Real-Tick Replay

[`standard_limit_real_tick_may2026.csv`](reports/standard_limit_real_tick_may2026.csv)
contains a MT5 bid/ask tick replay for XAUUSDm from 1 May through 7 June 2026.
The run used a 0.50 model threshold, weighted A/B sizing, and one concurrent
structure. All report rows have 100% required tick-day coverage.

The result is a historical test of **standard pending-limit** execution. A
winrate is calculated only from resolved trades; skipped setups are recorded in
the `missed` field and must not be interpreted as wins. The CSV carries the
timeframe, strategy, wins, losses, drawdown, and coverage fields for independent
review.

| Timeframe | FVG | OB | BB | Swapzone | BPR | IC | Combined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M15 | 80.65% (75W/18L) | 81.61% (71W/16L) | 10.00% (1W/9L) | 57.07% (109W/82L) | 77.50% (31W/9L) | 92.69% (203W/16L) | 85.33% (285W/49L) |
| M30 | 72.22% (39W/15L) | 73.47% (36W/13L) | 16.67% (1W/5L) | 52.00% (52W/48L) | 62.50% (10W/6L) | 87.80% (108W/15L) | 76.30% (132W/41L) |
| H1 | 77.78% (21W/6L) | 76.00% (19W/6L) | n/a | 54.00% (27W/23L) | 85.71% (6W/1L) | 84.21% (64W/12L) | 78.35% (76W/21L) |
| H4 | 72.73% (8W/3L) | 75.00% (3W/1L) | 0.00% (0W/1L) | 88.89% (8W/1L) | 0.00% (0W/2L) | 75.00% (12W/4L) | 73.08% (19W/7L) |
| D1 | 50.00% (2W/2L) | n/a | n/a | 66.67% (2W/1L) | 100.00% (1W/0L) | 100.00% (1W/0L) | 57.14% (4W/3L) |

`n/a` means no resolved trade, not a zero-loss or zero-profit result. Small
samples, especially H4 and D1, are not sufficient to establish reliability.

### WatchZone Forward Evidence

[`forward_test_trades.csv`](reports/forward_test_trades.csv) is the trade-level
export from closed MT5 forward-test positions. It includes open/close time in
WIB, raw MT5 entry/exit comments, entry/exit price, PnL, result, and the planned SL/TP only when
the ticket can be matched to a saved source signal. Blank planned levels mean
the source signal was not retained, not that the trade had no protection.

[`forward_test_summary.csv`](reports/forward_test_summary.csv) aggregates those
trades by entry type, timeframe, and strategy. WatchZone evidence is **real
forward-test history**, not yet a reconstructed real-tick backtest: an exact
replay needs historical zone registration, first tick hit, and fresh M1/M5
rejection confirmation for each event.

MT5 comments identify the execution compactly: pending limits use
`SMC <TF> <Strategy> <A/B>` and instant WatchZone entries use
`SMC <TF> <Strategy> Mkt <A/B>`. Telegram WatchZone alerts include zone, hit
price, entry, SL, TP, confidence, and ticket. Enable
`TELEGRAM_EVENT_LOG_ENABLED=True` to export future secret-free alert delivery
events to [`telegram_delivery_events.csv`](reports/telegram_delivery_events.csv).

| Timeframe | Strategy | Closed Trades | Winrate | Net PnL (USD) |
| --- | --- | ---: | ---: | ---: |
| M30 | BPR | 1 (1W/0L) | 100.00% | +9.64 |
| M30 | FVG | 56 (30W/26L) | 53.57% | -167.25 |
| M30 | IC | 21 (15W/6L) | 71.43% | +17.72 |
| M30 | OB | 15 (13W/2L) | 86.67% | +113.03 |
| H1 | FVG | 39 (15W/24L) | 38.46% | -98.40 |
| H1 | IC | 29 (20W/9L) | 68.97% | -63.99 |
| H1 | OB | 16 (11W/5L) | 68.75% | +78.87 |
| H4 | FVG | 8 (6W/2L) | 75.00% | +10.87 |
| H4 | IC | 2 (1W/1L) | 50.00% | +0.74 |
| H4 | OB | 6 (2W/4L) | 33.33% | -20.12 |
| H4 | SND | 3 (3W/0L) | 100.00% | +15.40 |
| H4 | Unknown | 13 (8W/5L) | 61.54% | -32.44 |
| D1 | Unknown | 71 (28W/43L) | 39.44% | -17.24 |

There are no closed M15 WatchZone trades in the current public export. `Unknown`
means the historical trade could not be matched back to its saved source signal;
it is retained instead of being silently assigned to a strategy.

For spreadsheet users, download
[`forward_test_report.xlsx`](reports/forward_test_report.xlsx). It contains an
overview, the standard-limit real-tick matrix, forward trades, and forward
summary in separate sheets.

### Daily Publication

On the machine that refreshes the local MT5 exports, run:

```powershell
.\scripts\publish_daily_reports.ps1
```

The command rebuilds the files in `reports/`, commits only public report
artifacts and supporting documentation, then pushes to `main`. It deliberately
does not publish `.env`, account state, raw signal state, raw tick cache, or
the local `scratch/` source exports.

---

## 12. Integrasi Visualisasi Pivot & SnR di MT5
Sistem visualisasi garis SnR dan Pivot di MetaTrader 5 bekerja dengan cara berikut:
1. **Python Bot**: Menghitung level Pivot Point (PP), Support (S1-S4), dan Resistance (R1-R4) secara periodik dari data D1.
2. **File Ekspor**: Data level ditulis sebagai file terenkripsi JSON ke direktori global MT5 Common:
   `AppData\Roaming\MetaQuotes\Terminal\Common\Files\pivot_levels.json`
3. **MQL5 Indicator**: Indikator kustom di dalam MT5 membaca file `.json` tersebut dan menggambar garis horizontal secara dinamis di chart kamu.
   * *Troubleshooting*: Jika garis SnR/Pivot tidak muncul di terminal MT5 kamu, pastikan **Indikator Pivot Kustom** sudah ditarik (*drag*) dari panel **Navigator** ke chart `XAUUSDm` kamu saat ini.

---

### Update Pengamanan & Aturan Main Baru (Juli 2026)
Untuk mengunci profit dan meminimalkan kerugian saat terjadi fluktuasi news, pengamanan berikut telah diimplementasikan:

1. **WatchZone untuk OB, BPR, dan FVG**:
   * Sistem WatchZone mengizinkan entri instan untuk **OB**, **BPR**, **FVG**, dan Pivot sesuai live policy. Setiap entry tetap memerlukan rejection confirmation, reversal guard, dan market safety dari closed-bar volume, RSI8, dan Stochastic sebelum market order dikirim.
   * FVG juga tetap dapat menggunakan **Standard Limit (Pending Order)** saat rejection belum terkonfirmasi, sehingga entry dapat menunggu retracement ke Fib 0.5/0.618.
2. **Enforcement Kebijakan Strategi**:
   * WatchZone sekarang mendeteksi dan memblokir strategi yang masuk daftar hitam di `.env` (seperti `IC`, `SND`, `Swapzone`, `Pivot`).
3. **Proteksi Clustering & Over-exposure**:
   * Satu magic number hanya boleh memiliki satu eksposur searah gabungan (posisi dan pending order) lewat `MT5_MAX_SAME_DIRECTION_TRADES=1`. Scanner juga memakai lock per magic agar worker simbol berbeda tidak menggandakan eksposur.
4. **D1 Dinonaktifkan**:
   * Timeframe `D1` telah dihapus dari daftar trading (`MT5_ALLOWED_TIMEFRAMES=M30,H1,H4`) untuk menghindari entri berskala besar yang lambat dan berisiko tinggi.

---

## Catatan

- Bot ini dibuat untuk XAUUSD (Gold). Bisa dipakai untuk pair lain tapi parameter pip multiplier dan trailing stop belum dioptimalkan untuk pair lain.
- Model ML yang disertakan sudah dilatih dengan data nyata. Makin banyak trade yang masuk, makin pintar modelnya (self-learning otomatis).
- Jangan lupa set `MT5_EXECUTE_TRADES=False` dulu saat pertama coba, pantau sinyal dulu beberapa hari sebelum aktifkan trading.
- Backtest dan forward test adalah bukti historis, bukan jaminan profit atau rekomendasi trading.
