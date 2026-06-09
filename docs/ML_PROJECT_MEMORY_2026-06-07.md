# Forex SMC Analyzer Project Memory - 2026-06-07

Dokumen ini adalah ringkasan cepat kalau session putus.

## Project

Path:

```text
C:\Users\WINDOWS 11 PRO\forex-smc-analyzer
```

Tujuan user:

- ML trading bot membaca setup SMC/ICT style,
- menerima signal di atas threshold,
- tetap memantau signal di bawah threshold sebagai shadow data,
- belajar dari TP/SL,
- entry market order hanya saat rejection/key level/price zone valid,
- pending order saat belum ada reaksi market yang cukup.

Peringatan:

- `.env` sekarang punya `MT5_EXECUTE_TRADES=True`.
- Scanner loop bisa kirim order real.
- Jangan klaim perfect atau next trade pasti profit.

## Phase Summary

Phase 1:

- Shadow tracking aktif untuk semua signal di bawah threshold, termasuk confidence di bawah 30%.
- Tujuan: signal yang tidak dieksekusi tetap dipantau sebagai data belajar.

Phase 2:

- Shadow resolver membaca candle setelah signal.
- Outcome TP/SL ditulis ke `data/shadow_labeled_setups.csv`.

Phase 3:

- Trainer membaca real data plus shadow data.
- Shadow sample diberi bobot lebih kecil, default `0.35`.
- Outcome metadata yang berpotensi leakage dipisah dari feature training.

Phase 4:

- Calibration report dibuat di `src/calibration_report.py`.
- Output: `data/calibration_report.json`.
- Report berisi threshold, buckets, timeframe, hour, killzone, setup type, direction, source.

Phase 5:

- Rollout gate dibuat di `src/rollout_status.py`.
- Threshold 0.50 dicek dari calibration metrics.
- Policy doc: `docs/ML_PHASE_5_ROLLOUT_POLICY_2026-06-07.md`.

Phase 6:

- Rejection/entry hardening selesai.
- Live scanner sekarang menjalankan SND detector.
- Market order helper:
  - `should_market_enter_setup()`,
  - `choose_dual_market_entry_option()`.
- Calibration report sekarang punya `strategies`.
- Labeler dan shadow resolver sekarang menyimpan metadata `strategy` untuk data baru.
- Policy doc: `docs/ML_PHASE_6_ENTRY_REJECTION_HARDENING_2026-06-07.md`.

## Latest Verification

Terakhir dijalankan:

```powershell
& '.\.venv\Scripts\python.exe' -m py_compile src\calibration_report.py src\shadow_tracker.py src\model_trainer.py src\labeler.py src\scanner_worker.py
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m src.calibration_report
& '.\.venv\Scripts\python.exe' -m src.rollout_status --threshold 0.50
```

Hasil:

- full suite: `127 passed, 38 warnings`,
- calibration report regenerated,
- rollout threshold 0.50: `READY`,
- live execution: `True`,
- warning: scanner loop bisa place real order.

## Latest Calibration

Overall:

- samples: 1020,
- winrate: 36.47%,
- expectancy: -0.0R,
- profit factor: 0.99,
- max drawdown: 81.38R.

Threshold 0.50:

- samples: 326,
- winrate: 95.09%,
- expectancy: 1.53R,
- max drawdown: 2.0R.

Interpretasi:

- Threshold 0.50 lolos rollout gate dari data saat ini.
- Overall dataset masih jelek karena low-confidence buckets banyak loss.
- Jangan turunkan threshold tanpa lihat report.

## Rejection Entry Rule

Market order:

- confidence lolos threshold,
- `rejection_confirmed=True`,
- current price ada di entry zone,
- SL/TP valid.

Pending order:

- confidence lolos threshold,
- setup valid,
- rejection belum confirm atau current price belum masuk instant-entry zone.

BUY single:

```text
sl_price + 0.5 <= current_price <= entry_price + 0.5
```

SELL single:

```text
entry_price - 0.5 <= current_price <= sl_price - 0.5
```

Dual:

- 0.5 layer prioritas jika harga dekat entry utama,
- 0.618 layer jika harga masuk lebih dalam,
- tanpa rejection, tidak market order.

## Manual Commands

Masuk project:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
```

Cek health:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m src.calibration_report
& '.\.venv\Scripts\python.exe' -m src.rollout_status --threshold 0.50
```

One-shot scanner:

```powershell
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50
```

Live loop:

```powershell
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50 --loop
```

Gunakan live loop hanya jika siap dengan real order karena `.env` live execution aktif.

## Next Work

Prioritas berikutnya:

1. Jalankan monitoring saat market open.
2. Kumpulkan shadow outcomes baru.
3. Regenerate calibration report.
4. Lihat loss cluster dari strategy/timeframe/hour/killzone.
5. Baru adjust threshold atau logic, jangan berdasarkan satu trade.
