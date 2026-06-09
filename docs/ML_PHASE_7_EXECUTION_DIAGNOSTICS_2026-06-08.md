# ML Phase 7 Execution Diagnostics - 2026-06-08

Dokumen ini dibuat supaya konteks Phase 7 tidak hilang kalau session putus.

## Trigger

User melihat rejection di Daily Pivot, tapi bot tidak market order. Audit data menunjukkan beberapa Pivot signal terbaru memang masuk shadow karena confidence masih di bawah threshold 0.50.

Contoh yang ditemukan di `data/shadow_signals.json`:

- `M15_Pivot_SINGLE_BEAR_4335.000_2026-06-08_11:30:00`
  - strategy: `Pivot`
  - option: `Pivot Sell (PP)`
  - confidence: sekitar `0.084`
  - threshold: `0.50`
  - filtered reason: `below_accept_threshold`
- `H1_Pivot_SINGLE_BULL_4320.000_2026-06-08_14:00:00`
  - strategy: `Pivot`
  - option: `Pivot Buy (S1)`
  - confidence: sekitar `0.266`
  - threshold: `0.50`
  - filtered reason: `below_accept_threshold`

Jadi untuk kasus ini, tidak market order adalah expected behavior karena threshold belum lolos.

## Bug Yang Diperbaiki

Ada bug potensial di scanner registry:

- low-confidence setup disimpan ke `sent_signals.json` dengan `is_low_confidence=True`,
- kalau setup yang sama kemudian naik menjadi high-confidence, cabang duplicate registry bisa `continue` tanpa order karena ticket masih `None`,
- akibatnya shadow tracking bisa mengunci live execution untuk setup yang sebenarnya sudah lolos threshold.

Fix:

- tambah `should_promote_low_confidence_record()`,
- jika record low-confidence belum punya ticket dan belum outcome recorded, scanner menghapus registry lama dan fall-through ke jalur eksekusi high-confidence normal,
- berlaku untuk single dan dual setup.

## Dashboard Phase 7

Dashboard sekarang menambah diagnostic read-only:

- source `shadow_registry` untuk low-confidence row di `sent_signals.json`,
- `Execution Decision Diagnostics`,
- `Key Level / Pivot Diagnostics`,
- decision label:
  - `shadow_monitoring`,
  - `live_ticket`,
  - `accepted_no_ticket`,
  - `needs_live_promotion_review`,
  - `registry_missing_confidence`,
  - `below_threshold_registry`.

Tujuan tabel ini adalah menjawab kenapa signal tidak order tanpa harus membuka JSON manual.

## Market Order Rule Tetap

Market order hanya boleh terjadi kalau:

- setup masih aktif,
- confidence `>= ML_ACCEPT_THRESHOLD`,
- rejection confirmed,
- current price berada di entry zone,
- SL/TP valid,
- `.env` mengaktifkan `MT5_EXECUTE_TRADES=True`,
- MT5 tidak menolak order karena duplicate, distance, timeframe, symbol, atau market indicator check.

Pivot rejection yang confidence-nya `0.08`, `0.17`, atau `0.26` tetap shadow, bukan market order, karena threshold user adalah `0.50`.

## Model Stale Warning

Warning:

```text
Model smc_xgb_classifier.joblib is older than latest labeled/shadow data.
Model smc_lgb_classifier.joblib is older than latest labeled/shadow data.
```

Artinya model masih bisa dipakai, tapi file model aktif lebih tua dari data label/shadow terbaru. Ini bisa terjadi karena belum retrain, atau retrain sudah jalan tetapi champion gate mempertahankan model lama. Cara refresh yang benar:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m src.model_trainer
& '.\.venv\Scripts\python.exe' -m src.calibration_report
```

Jangan artikan warning ini sebagai scanner rusak. Ini freshness warning.

Update Phase 7:

- Retrain manual sudah dijalankan.
- Challenger ditolak oleh champion gate karena performa lebih lemah dari model lama.
- Model joblib tidak dioverwrite, jadi warning timestamp masih bisa muncul.
- Ini lebih aman daripada memaksa model baru yang lebih buruk.

## Verification Plan

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m pytest tests\test_scanner_entry_decisions.py -q
& '.\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& '.\.venv\Scripts\python.exe' -m compileall -q src tests
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Hasil aktual Phase 7:

- `tests/test_scanner_entry_decisions.py`: `14 passed`
- `tests/test_dashboard_data.py`: `38 passed`
- targeted execution/dashboard suite: `79 passed`
- compile all `src` and `tests`: exit code 0
- full suite: `201 passed, 38 warnings`
- rollout threshold 0.50: `READY`
- dashboard HTTP check: `200 OK`

## Total Phase

Sampai dokumen ini:

- Dashboard track: 7 phase.
- ML/live-trading hardening track: 7 phase.

Phase 7 bukan akhir mutlak kalau forward-test menemukan bug baru, tapi ini menutup diagnostic gap untuk Pivot/Rejection execution path.
