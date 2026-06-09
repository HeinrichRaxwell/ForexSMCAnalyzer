# Forex SMC Dashboard Phase 5 Design - 2026-06-08

## Objective

Extend the local Streamlit dashboard with a read-only System QA and Feature Coverage layer. Phase 5 makes the dashboard show what has evidence, what was verified, what commands should be run manually, and which dashboard phases have spec/plan/memory coverage.

This phase is about evidence quality. It does not claim that trading is guaranteed profitable or that every future market condition is covered.

## Phase 5 Scope

Phase 5 adds four capabilities:

1. Dashboard phase document inventory for Phase 1 through Phase 5.
2. Test/source inventory showing how many source and test files are present.
3. Copy-only verification checklist for compile, test, dashboard import, and HTTP checks.
4. Feature coverage matrix that summarizes dashboard/core evidence areas and their current status.

These features improve confidence in operations and maintenance. They do not change strategy logic, model logic, order execution, Telegram behavior, risk management, or training.

## Data Layer

`src/dashboard_data.py` remains the owner of pure helper functions. Phase 5 adds:

- `build_phase_document_inventory(base_dir)`: returns rows for dashboard phases and whether spec/plan/memory evidence exists.
- `build_code_inventory(base_dir)`: counts source files, test files, dashboard files, docs, and data files.
- `build_verification_checklist(base_dir)`: returns copy-only commands for broad verification.
- `build_feature_coverage_matrix(...)`: returns one row per major feature area with status, evidence, and manual action.

Status rules:

- `ready`: evidence exists and no obvious missing dependency is detected.
- `caution`: evidence exists but needs manual review or data freshness attention.
- `blocked`: required file, doc, or evidence is missing.

## Dashboard UI

`src/dashboard.py` keeps the same seven tabs. Phase 5 enriches Command Center:

- System QA summary metrics.
- Feature coverage matrix.
- Phase document inventory.
- Verification checklist command table.
- Code/test inventory.

The dashboard must keep the checklist copy-only. It must not run pytest, compile, trainer, scanner, Telegram, MT5, or shell commands from Streamlit.

## Safety Rules

Dashboard Phase 5 is still read-only:

- no MT5 order calls
- no retrain execution button
- no scanner execution button
- no `.env` write
- no Telegram send
- no file deletion
- no labeled dataset mutation
- no subprocess execution from Streamlit

## Verification

Required checks:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m compileall -q src tests
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Dashboard restart and HTTP check:

```powershell
Start-Process -FilePath 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\streamlit.exe' -ArgumentList 'run','src\dashboard.py','--server.port','8501','--server.headless','true' -WorkingDirectory 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer' -WindowStyle Hidden -PassThru
Invoke-WebRequest -Uri 'http://localhost:8501' -UseBasicParsing -TimeoutSec 10
```

## Non-Claims

Phase 5 verification proves only the checks that were actually run. It cannot prove guaranteed profit, perfect strategy logic, broker execution quality, market fill quality, or next-trade outcome.
