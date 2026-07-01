# Forex SMC Dashboard Phase 6 Design - 2026-06-08

## Objective

Extend the local Streamlit dashboard with Strategy Formula QA evidence. Phase 6 makes formula coverage visible: which strategy modules exist, which tests cover them, which PineScript translations are full/core/partial/missing, and which targeted verification commands should be run manually.

This phase prevents overclaiming. It does not change trading formulas unless a failing test exposes a concrete bug.

## Phase 6 Scope

Phase 6 adds four capabilities:

1. Strategy formula inventory for SMC/Fibo, rejection, pivots, FLoOP, KNN SuperTrend, volume clusters, Telegram signal text, scanner decision logic, and active trade management.
2. PineScript translation matrix for the five files under `PineScripts/`.
3. Formula verification checklist with copy-only targeted pytest commands.
4. Formula QA status report that highlights full coverage, partial ports, and missing full ports.

These features improve truthfulness and maintainability. They do not execute MT5 orders, retrain models, edit `.env`, send Telegram messages, or change strategy entry logic.

## Data Layer

`src/dashboard_data.py` remains the owner of pure helper functions. Phase 6 adds:

- `build_strategy_formula_inventory(base_dir)`: returns source/test/doc evidence per formula area.
- `build_pinescript_translation_matrix(base_dir)`: returns one row per PineScript with Python module mapping and status.
- `build_formula_verification_checklist(base_dir)`: returns copy-only targeted commands for formula-related tests.
- `build_formula_qa_report(strategy_inventory, pinescript_matrix)`: returns overall formula QA status and reason rows.

Status meanings:

- `ready`: source and test evidence exists and no known missing full port is attached.
- `caution`: core/partial port exists, but not full TradingView parity.
- `blocked`: required source/test evidence is missing or a PineScript is not fully ported where the bot may rely on it.

## Dashboard UI

`src/dashboard.py` keeps the same seven tabs. Phase 6 enriches Strategy Performance:

- Formula QA status metric.
- Strategy formula inventory table.
- PineScript translation matrix.
- Formula verification checklist with command preview.

The dashboard must present partial/missing ports plainly. It must not say formulas are perfect when evidence says a script is only core/partial or not full port.

## Safety Rules

Dashboard Phase 6 is still read-only:

- no MT5 order calls
- no retrain execution button
- no scanner execution button
- no `.env` write
- no Telegram send
- no file deletion
- no labeled dataset mutation
- no subprocess execution from Streamlit

## Verification

Required targeted formula checks:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_fibo_detector.py tests\test_floop.py tests\test_pivots.py tests\test_rejection.py tests\test_imbalances.py tests\test_breakers_swapzones.py tests\test_knn_classifier.py tests\test_volume_clusters.py tests\test_scanner_entry_decisions.py tests\test_scanner_market_orders.py tests\test_active_trade_management.py -q
```

Required broad checks:

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

Formula QA can show code/test/translation evidence. It cannot prove guaranteed profit, perfect PineScript bar-by-bar parity without TradingView OHLCV comparison, broker fill quality, or next-trade outcome.
