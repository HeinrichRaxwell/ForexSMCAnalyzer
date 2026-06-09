# Forex SMC Dashboard Phase 4 Design - 2026-06-08

## Objective

Extend the local Streamlit dashboard with a read-only readiness gate for demo forward-testing. Phase 4 turns existing evidence into explicit `ready`, `caution`, or `blocked` decisions so the owner can see why the bot is or is not ready for manual next steps.

The dashboard remains evidence-first. It must not claim guaranteed profit, a perfect bot, or next-trade profit.

## Phase 4 Scope

Phase 4 adds four capabilities:

1. Model freshness details, including model age versus latest labeled/shadow data age.
2. Retraining readiness, based on `ML_RETRAIN_THRESHOLD`, `new_trades_since_last_train`, stale models, and available labeled rows.
3. Forward-test readiness gate, based on accepted/shadow resolved evidence, calibration recommendation, profit factor, and max consecutive losses.
4. A combined dashboard readiness report that lists status, severity, reasons, and manual next actions.

These features are advisory. They do not change strategy entry logic, threshold policy, risk management, model training, Telegram behavior, or order execution.

## Data Layer

`src/dashboard_data.py` remains the owner of pure helper functions. Phase 4 adds:

- `build_model_freshness_details(latest_data_time, model_inventory, now)`: returns one row per model with stale flag, data lag, and age.
- `build_retraining_readiness(snapshot, latest_data_time, model_inventory)`: returns status, reasons, and manual action text.
- `build_forward_readiness_gate(calibration_report, forward_summary, minimum_resolved)`: returns status, reasons, and metrics for demo forward-test promotion review.
- `build_dashboard_readiness_report(...)`: combines model, retrain, forward evidence, and health checks into one report.

Status rules:

- `ready`: no blocking issue detected.
- `caution`: evidence exists but needs manual review or refresh.
- `blocked`: a required file/data/model condition is missing or insufficient.

## Dashboard UI

`src/dashboard.py` keeps the existing seven tabs. Phase 4 enriches existing sections:

- Command Center:
  - overall readiness status
  - readiness checklist table
  - model freshness detail table
  - retraining readiness panel
- AI Learning:
  - retraining readiness detail near learning metrics
- Backtest vs Forward:
  - forward-test readiness gate next to forward evidence summary

The UI should keep reasons visible in tables instead of hiding them behind hover. Commands remain copy-only text.

## Safety Rules

Dashboard Phase 4 is still read-only:

- no MT5 order calls
- no retrain execution button
- no `.env` write
- no Telegram send
- no file deletion
- no labeled dataset mutation
- no subprocess execution from Streamlit

The dashboard can recommend manual commands, but execution stays in PowerShell.

## Verification

Required checks:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Dashboard restart and HTTP check:

```powershell
Start-Process -FilePath 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\streamlit.exe' -ArgumentList 'run','src\dashboard.py','--server.port','8501','--server.headless','true' -WorkingDirectory 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer' -WindowStyle Hidden -PassThru
Invoke-WebRequest -Uri 'http://localhost:8501' -UseBasicParsing -TimeoutSec 10
```

## Non-Claims

Readiness status is not a profit guarantee. It is a checklist for evidence quality, model freshness, and operational state before manual action.
