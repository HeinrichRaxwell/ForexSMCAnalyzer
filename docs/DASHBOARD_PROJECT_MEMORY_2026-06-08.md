# Dashboard Project Memory - 2026-06-08

## Objective

Create a complete local dashboard for the Forex SMC Analyzer so the owner can monitor:

- all accepted signals
- all shadow signals below threshold
- all strategy/timeframe performance
- confidence calibration
- learning/retraining state
- trade manager behavior
- backtest vs forward-test evidence

The dashboard must be evidence-first and must not claim guaranteed profit.

## Current Project State

Project path:

`C:\Users\WINDOWS 11 PRO\forex-smc-analyzer`

Important data files found:

- `data/sent_signals.json`
- `data/shadow_signals.json`
- `data/labeled_setups.csv`
- `data/shadow_labeled_setups.csv`
- `data/calibration_report.json`
- `data/learning_status.json`
- `data/backtest_simulation_results.csv`
- `data/real_tick_backtest_*.csv`

Important source files:

- `src/scanner_worker.py`
- `src/shadow_tracker.py`
- `src/inference.py`
- `src/model_trainer.py`
- `src/calibration_report.py`
- `src/rollout_status.py`
- `src/execution.py`
- `src/real_tick_backtester.py`

Current requirements do not include Streamlit. Phase 1 dashboard plan should add:

`streamlit>=1.35.0`

## Important Bot Rules To Show In Dashboard

### Confidence

Live accept threshold is controlled by:

`ML_ACCEPT_THRESHOLD`

Training max window is controlled by:

`ML_TRAINING_MAX_SETUPS`

Current user preference:

`ML_TRAINING_MAX_SETUPS=5000`

### Shadow Learning

Signals below live threshold are tracked in `shadow_signals.json`.

Resolved shadow outcomes are written to `shadow_labeled_setups.csv`.

Shadow data is source-aware and should stay visibly separate from real accepted trade data.

### CHoCH Cutloss

Current hardened rule in `src/execution.py`:

- M15/M30 emergency close requires two closed opposite candles, or HTF confirmation.
- H1/H4/D1 can emergency close after one closed opposite candle.
- Candle in progress is ignored.

Dashboard should show this rule in the Trade Manager tab to avoid confusion.

## Dashboard Tabs

1. Command Center
2. Live Signal Monitor
3. Trade Manager
4. AI Learning
5. Confidence Calibration
6. Strategy Performance
7. Backtest vs Forward Test

## Phase 1 Implementation Shape

Create:

- `src/dashboard_data.py`
- `src/dashboard.py`
- `tests/test_dashboard_data.py`

Modify:

- `requirements.txt`

Do not modify live trading logic for dashboard Phase 1.

Do not call MT5 order functions.

Do not retrain from dashboard.

## Verification Commands

Use these after implementation:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Manual launch command:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
.\.venv\Scripts\streamlit.exe run src\dashboard.py
```

## Current Design Documents

- `docs/superpowers/specs/2026-06-08-forex-smc-dashboard-design.md`
- `docs/superpowers/plans/2026-06-08-forex-smc-dashboard.md`

## Phase 1 Implementation Completed

Completed on 2026-06-08.

Files implemented:

- `src/dashboard_data.py`
- `src/dashboard.py`
- `tests/test_dashboard_data.py`

File modified:

- `requirements.txt` now includes `streamlit>=1.35.0`

Dashboard behavior:

- Local Streamlit dashboard.
- Read-only: no MT5 order call, no retrain button, no Telegram send, no file deletion.
- Loads accepted signals from `data/sent_signals.json`.
- Loads all shadow signals, including below-threshold confidence, from `data/shadow_signals.json`.
- Shows learning state from `data/learning_status.json`.
- Shows calibration and threshold performance from `data/calibration_report.json`.
- Shows model inventory and warning when model files are older than latest labeled/shadow data.
- Shows CHoCH emergency exit rule and confirms active candle is ignored by the trading logic.
- Shows backtest file inventory and forward shadow outcome summary.

Verification completed:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Latest verification result:

- Dashboard data tests: `8 passed`
- Full suite: `168 passed, 38 warnings`
- Dashboard HTTP check: `http://localhost:8501` returned `200 OK`

## Dashboard Import Fix

Issue found after Phase 1:

```text
ModuleNotFoundError: No module named 'src'
```

Cause:

- Streamlit can launch `src/dashboard.py` with `sys.path` starting at the `src` folder.
- In that launch mode, Python cannot resolve `from src.dashboard_data import ...` unless the project root is also on `sys.path`.

Fix:

- `src/dashboard.py` now inserts the project root into `sys.path` before importing `src.dashboard_data`.
- `tests/test_dashboard_data.py` includes a regression test that reproduces this Streamlit-style launch path.

Updated verification result after this fix:

- Dashboard data tests: `9 passed`
- Full suite: `169 passed, 38 warnings`

Current running dashboard:

- URL: `http://localhost:8501`
- Started with Streamlit process PID `29760`

Manual relaunch:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
.\.venv\Scripts\streamlit.exe run src\dashboard.py --server.port 8501 --server.headless true
```

Important caveat:

Dashboard Phase 1 monitors evidence. It does not prove guaranteed profit and does not make the strategy perfect. Use it for demo forward-test tracking, calibration review, and learning data visibility.

## Phase 2 Implementation Completed

Completed on 2026-06-08.

Design and plan docs:

- `docs/superpowers/specs/2026-06-08-forex-smc-dashboard-phase-2-design.md`
- `docs/superpowers/plans/2026-06-08-forex-smc-dashboard-phase-2.md`

Files updated:

- `src/dashboard_data.py`
- `src/dashboard.py`
- `tests/test_dashboard_data.py`

Phase 2 behavior:

- Accepted dual-leg signals now preserve per-leg features for drilldown:
  - `features_0.5`
  - `features_0.618`
- Shadow signals preserve `features` for drilldown.
- Live Signal Monitor now has signal detail selection and feature JSON.
- AI Learning now has source-aware confidence bucket summary.
- Command Center now has health checks for important data/model/config files.
- Backtest vs Forward now has accepted-vs-shadow forward evidence summary.

New data helpers:

- `assign_confidence_bucket`
- `build_signal_detail`
- `build_confidence_bucket_summary`
- `build_dashboard_health_checks`
- `summarize_forward_evidence`

Updated verification result after Phase 2:

- Dashboard data tests: `15 passed`
- Full suite: `175 passed, 38 warnings`
- Dashboard HTTP check: `http://localhost:8501` returned `200 OK`

## Phase 2 Import Cache Incident

Issue reported after Phase 2:

```text
ImportError: cannot import name 'build_confidence_bucket_summary' from 'src.dashboard_data'
```

Root cause:

- `src/dashboard_data.py` already contained `build_confidence_bucket_summary`.
- Direct venv import worked.
- Multiple old Streamlit dashboard processes were still running and one browser session was attached to a process/module cache from before Phase 2.

Fix performed:

- Stopped old Streamlit dashboard process chain.
- Started one fresh Streamlit dashboard process on port `8501`.
- Confirmed direct imports and dashboard HTTP response.

Updated verification after restart:

- `from src.dashboard_data import build_confidence_bucket_summary`: passed
- `from src.dashboard import cached_data`: passed
- Dashboard HTTP check: `http://localhost:8501` returned `200 OK`
- Full suite: `175 passed, 38 warnings`

Operational rule:

- After adding new dashboard imports or helper functions, restart the Streamlit server instead of relying on hot reload.

Safety status:

- Dashboard remains read-only.
- No MT5 order call added.
- No retrain button added.
- No `.env` writer added.
- No Telegram send added.
- No file deletion added.

Recommended next dashboard phase:

- Dashboard Phase 3 should be safe operations only after a separate spec:
  - export reports
  - scanner log viewer
  - optional manual refresh button
  - optional retrain command preview, not execution by default

## Phase 3 Implementation Completed

Completed on 2026-06-08.

Design and plan docs:

- `docs/superpowers/specs/2026-06-08-forex-smc-dashboard-phase-3-design.md`
- `docs/superpowers/plans/2026-06-08-forex-smc-dashboard-phase-3.md`

Files updated:

- `src/dashboard_data.py`
- `src/dashboard.py`
- `tests/test_dashboard_data.py`

Phase 3 behavior:

- Command Center now includes safe manual PowerShell command previews:
  - start live scanner
  - start dashboard
  - run calibration report
  - run dashboard tests
  - preview retrain command
- Command previews are copy-only. The dashboard does not execute scanner, retrain, tests, Telegram, MT5 orders, or `.env` writes.
- Command Center now includes a manual dashboard cache refresh button.
- Command Center now shows process hygiene notes for stale Streamlit import/cache issues.
- Live Signal Monitor can download the current filtered signal table as CSV.
- Live Signal Monitor can download selected signal detail as JSON.
- AI Learning can download confidence bucket summary as CSV.
- Backtest vs Forward can download forward evidence as CSV.
- Backtest vs Forward can download a JSON dashboard report containing snapshot, accepted signals, shadow signals, confidence summary, health checks, forward evidence, and explicit non-claims.
- Backtest vs Forward now includes `data/*.log` inventory and safe log tail viewer.

New data helpers:

- `build_report_export_payload`
- `build_log_inventory`
- `read_log_tail`
- `build_safe_command_previews`
- `build_dashboard_process_notes`

Safety status after Phase 3:

- Dashboard remains read-only.
- No MT5 order call added.
- No retrain execution button added.
- No `.env` writer added.
- No Telegram send added.
- No file deletion added.
- No labeled dataset mutation added.
- Log tail reader rejects paths outside `data/*.log`.

Verification completed:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Latest verification result:

- Dashboard data tests: `20 passed`
- Compile check: passed for `src/dashboard_data.py` and `src/dashboard.py`
- Full suite: `180 passed, 38 warnings`
- Direct Phase 3 imports: passed
- Dashboard HTTP check: `http://localhost:8501` returned `200 OK`

Current running dashboard:

- URL: `http://localhost:8501`
- Fresh Streamlit process started with PID `23500`

Manual dashboard relaunch:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
.\.venv\Scripts\streamlit.exe run src\dashboard.py --server.port 8501 --server.headless true
```

Manual live scanner start:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
.\.venv\Scripts\python.exe -m src.main
```

Operational rule:

- After adding new dashboard imports or helper functions, stop old Streamlit dashboard processes and start a fresh dashboard server.
- Do not use dashboard command previews as proof of profitability. They are operational commands only.

Important caveat:

Dashboard Phase 3 improves monitoring and operational safety. It does not prove guaranteed profit, a perfect bot, or next-trade profit.

## Phase 4 Implementation Completed

Completed on 2026-06-08.

Design and plan docs:

- `docs/superpowers/specs/2026-06-08-forex-smc-dashboard-phase-4-design.md`
- `docs/superpowers/plans/2026-06-08-forex-smc-dashboard-phase-4.md`

Files updated:

- `src/dashboard_data.py`
- `src/dashboard.py`
- `tests/test_dashboard_data.py`

Phase 4 behavior:

- Command Center now shows a read-only readiness gate:
  - overall status
  - latest training data timestamp
  - resolved forward evidence count
  - component checklist with status, reasons, and manual action
- Command Center now shows model freshness detail:
  - model modified time
  - latest labeled/shadow data time
  - model age
  - data lag
  - stale/fresh status
- AI Learning now shows retraining readiness:
  - status
  - `new_trades_since_last_train`
  - `ML_RETRAIN_THRESHOLD`
  - total labeled rows
  - model stale flag
  - manual action text
- Backtest vs Forward now shows a forward readiness gate:
  - status
  - resolved forward trades
  - minimum required resolved trades
  - profit factor
  - max consecutive losses
  - recommended threshold
  - manual action text

New data helpers:

- `build_model_freshness_details`
- `build_retraining_readiness`
- `build_forward_readiness_gate`
- `build_dashboard_readiness_report`

Readiness status meanings:

- `ready`: no blocking issue detected by that checklist.
- `caution`: data exists but needs manual review or refresh.
- `blocked`: required evidence, file, or data condition is missing/insufficient.

Safety status after Phase 4:

- Dashboard remains read-only.
- No MT5 order call added.
- No retrain execution button added.
- No `.env` writer added.
- No Telegram send added.
- No file deletion added.
- No labeled dataset mutation added.
- No Streamlit subprocess execution added.

Verification completed:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Latest verification result:

- Dashboard data tests: `26 passed`
- Compile check: passed for `src/dashboard_data.py` and `src/dashboard.py`
- Full suite: `186 passed, 38 warnings`
- Direct Phase 4 imports: passed
- Dashboard HTTP check: `http://localhost:8501` returned `200 OK`

Current running dashboard:

- URL: `http://localhost:8501`
- Fresh Streamlit process started with PID `2852`

Important caveat:

Dashboard Phase 4 readiness is a review checklist. It does not prove guaranteed profit, a perfect bot, or next-trade profit.

## Phase 5 Implementation Completed

Completed on 2026-06-08.

Design and plan docs:

- `docs/superpowers/specs/2026-06-08-forex-smc-dashboard-phase-5-design.md`
- `docs/superpowers/plans/2026-06-08-forex-smc-dashboard-phase-5.md`

Files updated:

- `src/dashboard_data.py`
- `src/dashboard.py`
- `tests/test_dashboard_data.py`

Phase 5 behavior:

- Command Center now includes a System QA panel.
- System QA shows source/test/dashboard/docs/data inventory.
- System QA shows a feature coverage matrix:
  - dashboard phase documentation
  - code/test inventory
  - runtime readiness
  - data/model health
- System QA shows dashboard phase documentation inventory for Phase 1 through Phase 5.
- System QA shows copy-only verification commands:
  - dashboard data tests
  - compile all `src` and `tests`
  - full pytest suite
  - dashboard HTTP check
  - dashboard import check

New data helpers:

- `build_phase_document_inventory`
- `build_code_inventory`
- `build_verification_checklist`
- `build_feature_coverage_matrix`

Safety status after Phase 5:

- Dashboard remains read-only.
- No MT5 order call added.
- No retrain execution button added.
- No scanner execution button added.
- No `.env` writer added.
- No Telegram send added.
- No file deletion added.
- No labeled dataset mutation added.
- No Streamlit subprocess execution added.

Verification completed:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m compileall -q src tests
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Latest verification result:

- Dashboard data tests: `30 passed`
- Compile all `src` and `tests`: passed
- Full suite: `190 passed, 38 warnings`
- Direct Phase 5 imports: passed
- Dashboard HTTP check: `http://localhost:8501` returned `200 OK`

Current running dashboard:

- URL: `http://localhost:8501`
- Fresh Streamlit process started with PID `8204`

Important caveat:

Phase 5 proves the checks above passed in this workspace. It does not prove guaranteed profit, perfect strategy logic, broker fill quality, market execution quality, or next-trade profit.

## Phase 6 Implementation Completed

Completed on 2026-06-08.

Design and plan docs:

- `docs/superpowers/specs/2026-06-08-forex-smc-dashboard-phase-6-design.md`
- `docs/superpowers/plans/2026-06-08-forex-smc-dashboard-phase-6.md`

Files updated:

- `src/dashboard_data.py`
- `src/dashboard.py`
- `tests/test_dashboard_data.py`

Phase 6 behavior:

- Strategy Performance now includes Formula QA.
- Formula QA shows overall formula status.
- Formula QA shows strategy formula inventory:
  - SMC/Fibonacci structures
  - rejection logic
  - pivot classic/rejection
  - FLoOP Pro
  - KNN SuperTrend
  - volume clusters
  - scanner entry decisions
  - active trade management
  - Telegram signal formatting
- Formula QA shows PineScript translation matrix for all five `PineScripts/*.txt` files.
- Formula QA shows formula verification checklist with copy-only pytest commands.

New data helpers:

- `build_strategy_formula_inventory`
- `build_pinescript_translation_matrix`
- `build_formula_verification_checklist`
- `build_formula_qa_report`

Known formula/PineScript caveats:

- `Floop PRO.txt`: core formula/pivots/filter behavior represented in Python and covered by tests.
- `AI-SuperTrend (KNN Machine Learning).txt`: core KNN probability engine exists, but not full TradingView visual/signal-state parity.
- `Clusters Volume Profile [LuxAlgo].txt`: core cluster/POC numeric features exist, but not full TradingView visual profile objects.
- `Machine Learning RSI AI Classification & Ranking (Zeiierman).txt`: not full port.
- `Multi-Timeframe Volume Profiles.txt`: not full port.

Safety status after Phase 6:

- Dashboard remains read-only.
- No MT5 order call added.
- No retrain execution button added.
- No scanner execution button added.
- No `.env` writer added.
- No Telegram send added.
- No file deletion added.
- No labeled dataset mutation added.
- No Streamlit subprocess execution added.

Verification completed:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_fibo_detector.py tests\test_floop.py tests\test_pivots.py tests\test_rejection.py tests\test_imbalances.py tests\test_breakers_swapzones.py tests\test_knn_classifier.py tests\test_volume_clusters.py tests\test_scanner_entry_decisions.py tests\test_scanner_market_orders.py tests\test_active_trade_management.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m compileall -q src tests
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Latest verification result:

- Dashboard data tests: `34 passed`
- Formula targeted suite: `57 passed, 2 warnings`
- Compile all `src` and `tests`: passed
- Full suite: `194 passed, 38 warnings`
- Direct Phase 6 imports: passed
- Dashboard HTTP check: `http://localhost:8501` returned `200 OK`

Current running dashboard:

- URL: `http://localhost:8501`
- Fresh Streamlit process started with PID `12536`

Phase count status:

- Dashboard track: Phase 1 through Phase 6 completed.
- Core ML/trading track documented separately: 7 phases total in the ML roadmap docs.

Important caveat:

Phase 6 proves targeted formula tests passed and shows formula/translation evidence. It does not prove guaranteed profit, perfect PineScript bar-by-bar parity, full ML RSI/MTF Volume Profile port, broker fill quality, or next-trade profit.

## Phase 7 Implementation Completed

Phase 7 fokus ke execution-path diagnostics dan Pivot/Rejection market-order hardening.

Trigger:

- User melihat rejection di Daily Pivot tetapi tidak ada market order.
- Audit data menunjukkan Pivot signal terbaru masuk shadow karena confidence di bawah threshold 0.50.
- Warning model stale masih muncul karena labeled/shadow data lebih baru daripada model joblib.

Yang dikerjakan:

- `src/scanner_worker.py`
  - menambah `should_promote_low_confidence_record()`,
  - memperbaiki jalur single dan dual setup supaya low-confidence registry record tidak mengunci eksekusi jika setup yang sama nanti lolos threshold,
  - promosi hanya terjadi kalau record low-confidence belum punya ticket dan belum outcome recorded.
- `src/dashboard_data.py`
  - low-confidence row dari `sent_signals.json` sekarang source-nya `shadow_registry`,
  - menambah `build_execution_decision_diagnostics()`,
  - menambah key-level context: pivot, volume POC, support/resistance flip, supply/demand,
  - phase inventory default dinaikkan sampai Phase 7.
- `src/dashboard.py`
  - menambah table `Execution Decision Diagnostics`,
  - menambah table `Key Level / Pivot Diagnostics`,
  - tetap read-only, tanpa order/retrain button.
- Tests:
  - menambah scanner promotion tests,
  - menambah dashboard diagnostic tests.
- Docs:
  - `docs/superpowers/specs/2026-06-08-forex-smc-dashboard-phase-7-design.md`,
  - `docs/superpowers/plans/2026-06-08-forex-smc-dashboard-phase-7.md`,
  - `docs/ML_PHASE_7_EXECUTION_DIAGNOSTICS_2026-06-08.md`.

Interpretasi penting:

- Pivot rejection yang confidence 0.08, 0.17, atau 0.26 tidak akan order saat threshold 0.50. Itu expected.
- Kalau Pivot/Rejection nanti confidence >= 0.50, masih harus lolos rule market order: rejection confirmed, price dalam entry zone, SL/TP valid, execution enabled, dan MT5 tidak menolak order.
- Warning stale model bukan berarti scanner rusak; artinya model belum retrain memakai labeled/shadow data terbaru.

Verification completed after Phase 7:

- `tests/test_scanner_entry_decisions.py`: `14 passed`
- `tests/test_dashboard_data.py`: `38 passed`
- targeted execution/dashboard suite: `79 passed`
- compile all `src` and `tests`: exit code 0
- full suite: `201 passed, 38 warnings`
- `src.calibration_report`: regenerated
- `src.rollout_status --threshold 0.50`: `READY`
- dashboard HTTP check: `200 OK`

Retrain result:

- Manual `src.model_trainer` ran successfully.
- Challenger rejected by champion gate.
- Model files were not overwritten, so top-level stale-model warnings can remain.
- Dashboard readiness now explains this as possible champion retention instead of treating it as a broken scanner.
