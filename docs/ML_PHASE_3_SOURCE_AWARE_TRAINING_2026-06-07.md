# ML Phase 3 Source-Aware Training - 2026-06-07

Phase 3 membuat trainer membaca data real trade dan resolved shadow signal secara bersamaan, tetapi dengan bobot berbeda.

Tujuan utamanya: bot bisa belajar dari signal yang sebelumnya tidak dieksekusi karena confidence di bawah threshold, tanpa membuat data virtual/shadow mengalahkan data real trade.

## Yang Diimplementasikan

- `src/model_trainer.py`
  - `load_training_dataset()`
  - `prepare_training_features()`
  - `calculate_sample_weights()`
  - `get_shadow_sample_weight()`
- `train_xgboost_filter()` sekarang otomatis mencari shadow label di:
  - `data/shadow_labeled_setups.csv`
- Shadow sample hanya dipakai saat training, tidak dicampur balik ke:
  - `data/labeled_setups.csv`
- Champion-vs-Challenger gate dibuat fail-closed:
  - Jika evaluasi champion error, challenger ditolak.
  - Model lama tidak ditimpa.

## Rule Bobot Training

Bobot dasar masih memakai logic outcome existing:

- Win: `2.0`
- Loss mitigated atau cut-loss kecil: `0.5`
- Full loss: `1.5`

Untuk shadow sample, bobot dasar dikalikan:

```text
ML_SHADOW_SAMPLE_WEIGHT=0.35
```

Jika env tidak diset, default adalah `0.35`.
Nilai diklem antara `0.0` sampai `1.0`.

Contoh:

- Real win: `2.0`
- Shadow win: `2.0 * 0.35 = 0.70`
- Real full loss: `1.5`
- Shadow full loss: `1.5 * 0.35 = 0.525`

Artinya shadow tetap bisa menggeser model jika polanya berulang dan valid, tetapi tidak lebih dominan dari real closed trade.

## Anti-Leakage

Kolom outcome/source berikut tidak dipakai sebagai fitur model:

- `label`
- `time`
- `pnl_relative`
- `sample_source`
- `signal_id`
- `confidence`
- `accept_threshold`
- `resolved_at`
- `result`
- metadata status/ticket/tracking lain

Ini penting agar model tidak belajar dari informasi post-trade atau metadata shadow yang tidak tersedia secara fair saat live inference.

## Status Data Saat Phase 3 Dibuat

Saat cek lokal:

- `data/labeled_setups.csv`: `1000` real rows
- `data/shadow_labeled_setups.csv`: `20` resolved shadow rows

Jumlah shadow ini masih kecil. Itu cukup untuk membuktikan pipeline bekerja, tetapi belum cukup untuk menyimpulkan edge market.

## Verifikasi

Focused trainer tests:

```text
8 passed, 27 warnings
```

Compile check:

```text
py_compile src/model_trainer.py src/shadow_tracker.py src/scanner_worker.py: pass
```

Full test suite:

```text
108 passed, 38 warnings
```

Warnings berasal dari metric sklearn pada dummy test data dan FutureWarning pandas di test FLOOP; tidak menggagalkan test.

## Retrain Non-Live Setelah Phase 3

Command:

```powershell
& '.\.venv\Scripts\python.exe' -m src.model_trainer
```

Hasil penting:

```text
Loaded 1000 real setups.
[Source-Aware Training] Loaded 20 resolved shadow setups.
[Source-Aware Training] Combined dataset: 1000 real + 20 shadow setups.
```

Challenger-vs-Champion:

```text
Champion accuracy: 92.00%
Challenger accuracy: 83.00%
Champion winrate lolos filter: 95.24%
Challenger winrate lolos filter: 79.10%
Status: Challenger REJECTED, Champion remains active.
```

Ini hasil yang benar secara safety: model baru yang lebih buruk tidak menimpa model aktif.

## Manual PowerShell

Dari PowerShell:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
```

Cek test:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Retrain otak ML tanpa menjalankan order live:

```powershell
& '.\.venv\Scripts\python.exe' -m src.model_trainer
```

One-shot scan tanpa loop:

```powershell
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50
```

Loop scanner saat market sudah buka:

```powershell
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50 --loop --interval 5
```

Stop loop:

```powershell
Ctrl+C
```

Catatan penting: `.env` saat ini mengaktifkan live execution. Scanner loop bisa mengirim order real kalau sinyal lolos rule dan threshold.

## Batasan

- Ini membuat bot lebih pintar secara pipeline learning, bukan membuat bot perfect.
- Confidence tidak seharusnya dipaksa menuju 100%. Confidence 100% yang sering muncul justru perlu dicurigai sebagai overfit.
- Shadow sample baru `20` row, jadi belum cukup untuk menaikkan atau menurunkan threshold secara agresif.
- Phase 4 masih diperlukan untuk calibration report per confidence bucket, timeframe, strategy, winrate, expectancy, dan drawdown.
