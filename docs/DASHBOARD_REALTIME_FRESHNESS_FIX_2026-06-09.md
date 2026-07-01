# Dashboard Realtime Freshness Fix - 2026-06-09

## Tujuan

Memastikan dashboard tidak lagi menampilkan warning stale model yang misleading, readiness gate memakai data forward terbaru, dan Streamlit auto-refresh tanpa user refresh manual.

## Root Cause

- `smc_xgb_classifier.joblib` dan `smc_lgb_classifier.joblib` bisa terlihat lebih tua dari labeled/shadow data karena champion/challenger gate menolak challenger dan mempertahankan champion lama.
- Kondisi itu bukan otomatis berarti retraining rusak jika `data/learning_status.json` menunjukkan `last_train_time` sudah mereview data terbaru.
- Proses Streamlit sempat masih berjalan dari import lama sebelum `src/dashboard_data.py` terbaru dimuat.
- Accepted registry lama punya banyak `outcome_*_recorded=True`, tetapi belum menyimpan `result/status/pnl/close_reason` per leg, sehingga dashboard sebelumnya bisa menampilkan accepted sebagai open semua.

## Perubahan Kode

- `src/dashboard_data.py`
  - Model freshness sekarang membaca `learning_status.last_train_time`.
  - Warning stale model disuppress jika latest data sudah direview oleh retrain terbaru.
  - Model detail memakai status `reviewed` ketika champion lama sengaja dipertahankan.
  - Snapshot membawa `forward_summary`.
  - Accepted/shadow forward summary memisahkan `tp`, `sl`, `expired`, `bep_profit`, `protected_profit`, `profit_not_tp_verified`, `breakeven`, `resolved_unclassified`, dan `open`.
  - `flatten_sent_signals()` membaca outcome per leg dari `result_a/result_b`, `status_a/status_b`, `pnl_relative_a/pnl_relative_b`, `net_profit_a/net_profit_b`, `close_price_a/close_price_b`, `close_reason_a/close_reason_b`, dan `exit_category_a/exit_category_b`.

- `src/dashboard.py`
  - Auto-refresh dashboard awalnya dibuat 5 detik, lalu dinaikkan ke 15 detik agar tidak terlalu agresif saat user membaca tabel panjang.
  - Patch scroll-restore JS ternyata masih bisa mental ke atas karena tetap memakai browser full reload.
  - Auto-refresh sekarang diganti ke Streamlit `st.fragment(run_every="15s")`, jadi data dashboard rerun dari Python tanpa `window.location.reload()`.
  - Sidebar filter punya fragment sendiri; body dashboard punya fragment sendiri. Ini menjaga filter tetap update tanpa memanggil `st.sidebar` dari dalam body fragment.
  - Streamlit cache TTL mengikuti interval auto-refresh.
  - Tabel signal menampilkan detail close outcome: `exit_category`, `close_price`, `close_reason`, `pnl_relative`, `net_profit`.

- `src/inference.py`
  - `process_mt5_history_feedback()` sekarang menulis outcome ringkas ke `data/sent_signals.json` saat MT5 history menutup trade.
  - Field baru untuk single trade: `status`, `result`, `exit_category`, `pnl_relative`, `net_profit`, `close_price`, `close_reason`, `resolved_at`.
  - Field baru untuk dual leg: suffix `_a` dan `_b`, misalnya `result_a`, `pnl_relative_b`, `exit_category_a`.
  - BEP/protected profit tidak dipaksa menjadi TP; kategori detail tetap disimpan.

## State Terverifikasi

- Dashboard HTTP: `200 OK` di `http://localhost:8501`.
- Warning dashboard model freshness: `[]`.
- Readiness:
  - `Health Checks`: ready
  - `Model Freshness`: ready
  - `Retraining`: ready
  - `Forward Evidence`: caution
- Forward evidence terbaru:
  - Accepted: `total=601`, `resolved_unclassified=521`, `open=80`
  - Shadow: `total=340`, `tp=139`, `sl=62`, `open=139`, `winrate_pct=69.15`
  - `Resolved Forward`: `201`
- Overall masih `CAUTION` karena risk gate sah: `Max consecutive losses is high: 33.`

## Verification

- `python -m py_compile src\dashboard.py src\dashboard_data.py src\inference.py`
- `python -m pytest tests\test_dashboard_data.py -q` -> `42 passed`
- Setelah scroll-preserve patch: `python -m pytest tests\test_dashboard_data.py -q` -> `43 passed`
- Setelah soft-refresh fragment patch: `python -m pytest tests\test_dashboard_data.py -q` -> `43 passed`
- Setelah soft-refresh fragment patch: `python -m pytest -q` -> `211 passed, 35 warnings`
- `python -m pytest tests\test_inference.py tests\test_model_trainer.py -q` -> `21 passed, 33 warnings`
- `python -m pytest -q` -> `210 passed, 35 warnings`
- `python -m src.rollout_status --threshold 0.50` -> `READY`, dengan warning `MT5_EXECUTE_TRADES=True; scanner loop can place real orders.`

## Cara Run Manual PowerShell

Dashboard:

```powershell
cd "C:\Users\WINDOWS 11 PRO\forex-smc-analyzer"
& ".\.venv\Scripts\streamlit.exe" run src\dashboard.py --server.port 8501 --server.headless true
```

Live scanner:

```powershell
cd "C:\Users\WINDOWS 11 PRO\forex-smc-analyzer"
& ".\.venv\Scripts\python.exe" -m src.main
```

## Catatan Penting

Dashboard bisa makin realtime membaca file lokal, tetapi tidak menjamin profit. Forward test demo tetap wajib dipakai untuk validasi market live sebelum akun real.

## Formula QA Split

- Dashboard sempat menampilkan `Overall Formula Status = BLOCKED` karena status core formula bot dan full PineScript parity dicampur.
- Current split:
  - `Core Formula Status`: `READY`
  - `PineScript Parity`: `BLOCKED`
  - `Formula Areas`: `9`
  - `PineScripts`: `5`
- Penyebab PineScript parity blocked:
  - `Machine Learning RSI AI Classification & Ranking` belum full port sebagai modul Python khusus.
  - `Multi-Timeframe Volume Profiles` belum full port untuk VAH/VAL/POC value-area behavior.
- Formula targeted suite terverifikasi: `60 passed, 2 warnings`.
- Full suite setelah split: `212 passed, 35 warnings`.
