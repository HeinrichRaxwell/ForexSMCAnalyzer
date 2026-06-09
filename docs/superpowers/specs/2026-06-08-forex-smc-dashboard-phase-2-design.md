# Forex SMC Dashboard Phase 2 Design - 2026-06-08

## Objective

Extend the local Streamlit dashboard from Phase 1 into a stronger evidence monitor for demo forward-testing. The dashboard remains read-only and must not execute MT5 orders, retrain models, edit `.env`, delete files, or send Telegram messages.

## Phase 2 Scope

Phase 2 adds four capabilities:

1. Signal detail drilldown
2. Confidence and shadow-learning analytics
3. Operational health checks
4. Backtest vs forward-test summary

These features improve observability. They do not change strategy entry logic, threshold policy, risk management, or model training.

## Data Layer

`src/dashboard_data.py` remains the owner of file-safe parsing and computed data. Phase 2 adds pure helper functions that can be tested without Streamlit:

- `assign_confidence_bucket(confidence)`: maps confidence values into stable buckets from `0.00-0.30` through `0.90-1.00`.
- `build_signal_detail(signal_id, accepted_signals, shadow_signals)`: finds one accepted or shadow signal and returns a structured detail payload.
- `build_confidence_bucket_summary(signals)`: groups accepted and shadow signals by source, bucket, timeframe, strategy, status, and result.
- `build_dashboard_health_checks(base_dir, now)`: checks important project files, model files, and data staleness.
- `summarize_forward_evidence(accepted_signals, shadow_signals)`: summarizes accepted and shadow forward outcomes without mixing them into one performance claim.

Flattened signal rows must preserve per-leg feature dictionaries:

- accepted `0.5` leg uses `features_0.5`
- accepted `0.618` leg uses `features_0.618`
- shadow signal uses `features`

## Dashboard UI

`src/dashboard.py` keeps the same seven tabs. Phase 2 enriches existing tabs instead of adding a new navigation system:

- Command Center: health check table and warnings.
- Live Signal Monitor: signal selector with detail panel and feature JSON.
- AI Learning: confidence bucket summary for accepted and shadow signals.
- Strategy Performance: source-aware bucket and strategy/timeframe summaries.
- Backtest vs Forward: compact accepted-vs-shadow forward summary plus existing backtest file inventory.

The UI should keep essential values visible in tables and metrics. Hover-only explanations are not enough.

## Safety Rules

Dashboard Phase 2 is still read-only:

- no MT5 order calls
- no retrain button
- no `.env` write
- no Telegram send
- no file deletion
- no hidden mutation of labeled datasets

Any future operational button belongs to Dashboard Phase 3 and must have an explicit safety spec.

## Verification

Required checks:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Dashboard HTTP check:

```powershell
Invoke-WebRequest -Uri 'http://localhost:8501' -UseBasicParsing -TimeoutSec 10
```

## Non-Claims

Dashboard Phase 2 must not say the bot is perfect or that the next trade will profit. It should make evidence easier to inspect so demo forward-testing decisions are less blind.
