# ML Phase 6 Entry Rejection Hardening - 2026-06-07

Dokumen ini dibuat sebelum Phase 6 dikerjakan supaya konteks tidak hilang jika session putus.

## Tujuan

Phase 6 memperkuat entry logic live scanner:

- memastikan strategy tambahan yang sudah ada di code benar-benar masuk live scanner jika memang tersedia,
- membuat logic market order vs pending order lebih eksplisit dan testable,
- menambah visibility calibration per strategy asli,
- merapikan risiko test pollution MetaTrader5 supaya hasil test tidak misleading,
- mendokumentasikan rule rejection/key level/volume/confluence yang benar-benar dipakai.

Phase 6 tidak menjanjikan bot perfect, next trade profit, atau profit konsisten tanpa loss. Targetnya adalah entry lebih jelas, lebih terukur, dan lebih mudah diverifikasi.

## Status Sebelum Phase 6

Audit sebelumnya menemukan strategy entry yang tersedia:

- `FVG`
- `OB`
- `BPR`
- `IC` / Indecision Candle
- `Breaker`
- `Swapzone`
- `Pivot`
- `SND` / Supply Demand

Tambahan indikator dari Pine/TradingView translation yang ditemukan:

- `src/indicators/floop.py`
- `src/indicators/knn_classifier.py`
- `src/indicators/pivots.py`
- `src/indicators/volume_clusters.py`

Hasil test sebelum Phase 6:

```text
strategy/indicator/inference focused tests: 42 passed, 11 warnings
test_scanner_market_orders.py alone: 3 passed
full suite after Phase 5: 119 passed, 38 warnings
```

Catatan penting:

- `test_scanner_market_orders.py` pernah gagal jika digabung setelah `test_inference.py`.
- Root cause terlihat sebagai test pollution: `test_inference.py` mengganti `sys.modules["MetaTrader5"]` dengan fake `SimpleNamespace`, sehingga patch `MetaTrader5.symbol_info` pada test lain bisa kena module fake tanpa method lengkap.
- Ini belum membuktikan live entry salah, tapi test isolation harus diperbaiki supaya audit berikutnya dapat dipercaya.

## Temuan Entry Logic

### Rejection

`detect_rejection_at_level()` dipakai untuk melihat reaksi candle di level entry:

- FVG/OB/BPR/IC/SND memakai rejection sebagai confluence.
- Pivot rejection dianggap sudah confirmed karena setup-nya memang lahir dari rejection candle di pivot level.
- Scanner juga mengecek rejection LTF:
  - M5,
  - M1,
  - fallback M15 untuk setup timeframe lebih tinggi.

Target Phase 6: rule market order harus jelas:

- Kalau price sedang berada di entry zone,
- rejection confirmed,
- level valid,
- confidence lolos threshold,
- maka boleh market order.
- Kalau belum ada rejection/price belum masuk zone, fallback ke pending limit order.

### Key Level

Key level yang sudah masuk:

- pivot level dan psychological level,
- FVG/OB/BPR/IC/SND structure level,
- Swapzone support/resistance flip,
- Breaker Block retest,
- HTF active FVG priority/conflict.

### Volume

Volume yang sudah masuk:

- FVG quality filter memakai buyer/seller volume pressure pada displacement candle.
- Execution market indicators menolak abnormal volume spike.
- Volume clusters / POC proximity dipakai sebagai feature dan confluence text.

Target Phase 6: tidak overclaim bahwa volume logic perfect. Yang bisa diklaim hanya: volume checks ada dan akan diverifikasi lewat tests/calibration.

## Planned Work

### Task 1 - SND Live Scanner Activation

Masalah:

- `get_active_setups()` sudah punya SND block.
- `smc_detector.py` sudah punya `detect_supply_demand_zones()`.
- `scanner_worker.run_scan()` terlihat belum memanggil `detect_supply_demand_zones()` di pipeline detector live.

Target:

- Import `detect_supply_demand_zones` di `scanner_worker.py`.
- Panggil setelah `detect_indecision_candles()` atau sebelum `get_active_setups()`.
- Tambah test yang gagal dulu: scanner detector pipeline harus memanggil SND detector.

### Task 2 - Market Entry Decision Helper

Masalah:

- Logic market order vs pending order ada inline di `scanner_worker.py`.
- Ini sulit dites langsung dan rawan beda logic untuk dual vs single.

Target:

- Tambah helper pure function, misalnya `should_market_enter_setup(setup, current_price, entry_buffer=0.5)`.
- BUY market order jika rejection confirmed dan current price berada di zona entry valid:
  - dari sekitar SL+buffer sampai entry+buffer untuk single,
  - dual tetap mempertahankan pilihan 0.5/0.618.
- SELL market order jika rejection confirmed dan current price berada di zona entry valid:
  - dari entry-buffer sampai SL-buffer.
- Jika rejection belum confirmed, return false.
- Tambah tests untuk BUY, SELL, dan no-rejection fallback.

### Task 3 - Strategy Calibration Breakdown

Masalah:

- Calibration report sudah punya `setup_types`, tapi strategy asli seperti `BPR`, `IC`, `SND`, `Pivot`, `Swapzone`, `Breaker` bisa tercampur ke `setup_type` 0/1/2.

Target:

- Pastikan labeled/shadow rows punya strategy metadata jika tersedia.
- `calibration_report.py` menambah breakdown `strategies` dari kolom `strategy` atau fallback metadata lain.
- Tambah test report grouping by strategy.

### Task 4 - MT5 Test Isolation

Masalah:

- Fake MT5 module dari test bisa bocor ke test lain.

Target:

- Rapikan test inference agar fake module dikembalikan atau fungsi terkait meng-import module dengan aman.
- Tambah/ubah test supaya focused combined suite tidak gagal.

## Verification Plan

Focused tests:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m pytest tests\test_scanner_market_orders.py tests\test_inference.py -q
& '.\.venv\Scripts\python.exe' -m pytest tests\test_fibo_detector.py tests\test_imbalances.py tests\test_breakers_swapzones.py tests\test_rejection.py tests\test_floop.py tests\test_knn_classifier.py tests\test_pivots.py tests\test_volume_clusters.py tests\test_inference.py tests\test_scanner_market_orders.py -q
& '.\.venv\Scripts\python.exe' -m pytest tests\test_calibration_report.py -q
```

Compile:

```powershell
& '.\.venv\Scripts\python.exe' -m py_compile src\scanner_worker.py src\main.py src\smc_detector.py src\calibration_report.py src\execution.py src\inference.py
```

Full:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m src.calibration_report
& '.\.venv\Scripts\python.exe' -m src.rollout_status --threshold 0.50
```

## Manual Trading Policy Setelah Phase 6

Market order boleh dianggap valid hanya jika:

- confidence lolos threshold,
- price berada di entry zone,
- rejection LTF/keylevel valid,
- SL dan TP tersedia,
- `.env` memang mengaktifkan live execution,
- rollout status tidak `BLOCKED`.

Pending order dipakai jika:

- confidence lolos threshold,
- setup valid,
- price belum berada di zona market entry,
- atau rejection belum cukup untuk instant entry.

Jika ada keraguan, pakai one-shot scanner dulu:

```powershell
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50
```

## Yang Selesai Di Phase 6

Phase 6 sudah dikerjakan dan diverifikasi pada 2026-06-07.

Perubahan utama:

- `src/scanner_worker.py`
  - menambah `apply_smc_detectors()` untuk menjalankan pipeline detector live secara eksplisit,
  - memasukkan `detect_supply_demand_zones()` ke scanner live,
  - menambah `should_market_enter_setup()` untuk single setup market order,
  - menambah `choose_dual_market_entry_option()` untuk dual layer 0.5 / 0.618,
  - mengganti logic inline market-vs-pending order dengan helper yang bisa dites.
- `src/calibration_report.py`
  - menambah report breakdown `strategies`,
  - menambah fallback strategy untuk dataset lama yang belum punya kolom `strategy`.
- `src/shadow_tracker.py`
  - resolved shadow signal sekarang menyimpan `strategy` ke CSV untuk data baru.
- `src/labeler.py`
  - historical labeled setup baru sekarang menyimpan `strategy` asli: `FVG`, `OB`, `BPR`, `Swapzone`, `IC`, `SND`, `Pivot`.
- `src/model_trainer.py`
  - `strategy` dimasukkan ke `NON_FEATURE_COLUMNS` supaya metadata strategy tidak bocor menjadi feature model.
- Tests:
  - `tests/test_scanner_entry_decisions.py`,
  - update `tests/test_calibration_report.py`,
  - update `tests/test_shadow_tracker.py`,
  - update isolation behavior di `tests/test_inference.py`.

## Status Rejection / Market Order

Status sekarang:

- Rejection sudah dipakai sebagai syarat market order.
- Market order hanya boleh jalan jika:
  - confidence lolos threshold,
  - `rejection_confirmed=True`,
  - harga saat ini berada di zona entry valid,
  - SL dan TP tersedia,
  - live execution memang aktif.
- Kalau rejection belum confirm atau harga belum masuk zona, scanner fallback ke pending/limit order.

Rule yang dites:

- BUY single: `sl_price + 0.5 <= current_price <= entry_price + 0.5`.
- SELL single: `entry_price - 0.5 <= current_price <= sl_price - 0.5`.
- Dual setup:
  - layer 0.5 diprioritaskan ketika harga dekat entry utama,
  - layer 0.618 dipakai ketika harga lebih dalam di zona,
  - tanpa rejection tidak ada instant market order.

Catatan penting:

- Ini bukan "perfect" dan tidak menjamin next trade profit.
- Yang bisa diklaim: logic entry/rejection sekarang eksplisit, dites, dan punya fallback pending order.
- Volume/confluence sudah masuk lewat FVG quality filter, execution market checks, KNN/FLOOP/pivots, dan volume cluster features, tapi tetap harus divalidasi dari forward live data.

## Verification Result Terakhir

Command yang sudah jalan:

```powershell
& '.\.venv\Scripts\python.exe' -m py_compile src\calibration_report.py src\shadow_tracker.py src\model_trainer.py src\labeler.py src\scanner_worker.py
& '.\.venv\Scripts\python.exe' -m pytest tests\test_calibration_report.py tests\test_shadow_tracker.py tests\test_model_trainer.py tests\test_scanner_entry_decisions.py tests\test_scanner_market_orders.py -q
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m src.calibration_report
& '.\.venv\Scripts\python.exe' -m src.rollout_status --threshold 0.50
```

Hasil:

- compile: pass,
- focused suite: `37 passed, 27 warnings`,
- full suite: `127 passed, 38 warnings`,
- rollout status threshold 0.50: `READY`,
- live execution: `True`.

Warnings yang tersisa:

- pandas `FutureWarning` di test FLOOP karena freq `1H`,
- sklearn `UndefinedMetricWarning` di dummy model tests.

Keduanya bukan failure entry/rejection.

## Calibration Snapshot Terbaru

Overall:

- samples: 1020,
- winrate: 36.47%,
- expectancy: -0.0R,
- profit factor: 0.99,
- max drawdown: 81.38R,
- max consecutive losses: 33.

Threshold 0.50:

- samples: 326,
- winrate: 95.09%,
- expectancy: 1.53R,
- max drawdown: 2.0R,
- max consecutive losses: 2.

Recommendation:

- threshold: `0.50`,
- reason: `lowest_threshold_meeting_rules`.

Strategy breakdown saat ini memakai fallback untuk dataset lama:

- `FVG_OR_BPR`: 388 samples, winrate 50.26%, expectancy 0.07R,
- `OB_OR_SWAPZONE_IC_SND`: 492 samples, winrate 31.10%, expectancy 0.07R,
- `PIVOT_REJECTION`: 140 samples, winrate 17.14%, expectancy -0.49R.

Dataset lama belum bisa memisahkan BPR/IC/SND/Swapzone secara sempurna karena kolom `strategy` dulu belum disimpan. Mulai data baru, labeler dan shadow resolver akan menyimpan `strategy` asli.

## Manual PowerShell Step

Masuk folder project:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
```

Cek dulu rollout:

```powershell
& '.\.venv\Scripts\python.exe' -m src.calibration_report
& '.\.venv\Scripts\python.exe' -m src.rollout_status --threshold 0.50
```

One-shot scan:

```powershell
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50
```

Loop live hanya setelah sadar bahwa `.env` sekarang `MT5_EXECUTE_TRADES=True`, artinya scanner bisa place real order:

```powershell
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50 --loop
```

Mode lebih aman untuk observasi adalah ubah dulu `.env`:

```text
MT5_EXECUTE_TRADES=False
```

lalu jalankan scanner untuk monitoring/shadow learning tanpa eksekusi real.

## Next Phase Yang Masuk Akal

Phase berikutnya sebaiknya bukan menambah indikator lagi, tapi forward validation:

- jalankan scanner monitoring saat market open,
- biarkan shadow tracker mengumpulkan signal di bawah threshold juga,
- regenerate calibration setelah data baru masuk,
- cek apakah threshold 0.50 tetap READY,
- kalau ada loss, lihat cluster penyebabnya dari report: timeframe, strategy, hour, killzone, direction, dan confidence bucket.

Bot boleh dibuat makin pintar dari data, tapi tidak boleh dianggap 100% pasti profit. Sistem yang sehat adalah sistem yang bisa menolak setup jelek, belajar dari outcome, dan tetap punya risk control saat salah.
