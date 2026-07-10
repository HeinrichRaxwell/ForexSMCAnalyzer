# VPS Real-Money Runbook - 2026-06-16

This bot should run live MT5 execution only on Windows VPS with the broker MT5 terminal installed and logged in. Linux VPS is fine for dashboard/reporting, but live `MetaTrader5` order execution needs the Windows terminal session.

## Current Safety Defaults

- `MT5_EXECUTE_TRADES=False`
- `MT5_REQUIRE_ROLLOUT_READY=True`
- `MT5_MAX_CONCURRENT_TRADES=1`
- `MT5_DAILY_GOVERNOR_ENABLED=True`
- `MT5_ENFORCE_ENTRY_GATE=False`
- `MT5_LIVE_STRATEGY_ALLOWLIST=FVG_OR_BPR,OB_OR_SWAPZONE_IC_SND`

Entry quality is telemetry-only. It must not block execution. Real-money readiness is controlled by the rollout preflight and risk governor.

## VPS Setup

1. Install Windows VPS, broker MT5 terminal, Python, Git, and Visual C++ runtime if required by ML packages.
2. Log into MT5, enable algo trading, add the broker symbol to Market Watch, and keep the terminal running.
3. Copy the repo, `.env`, `data/`, and `models/` from the local machine.
4. Create the venv and install requirements:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. Verify tests and artifacts:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rollout_status.py tests\test_scanner_entry_decisions.py tests\test_scanner_market_orders.py tests\test_live_risk_governor.py -q
.\.venv\Scripts\python.exe -m py_compile src\rollout_status.py src\scanner_worker.py src\execution.py src\live_risk_governor.py
```

6. Regenerate report after any `.env` live-policy change:

```powershell
.\.venv\Scripts\python.exe -m src.calibration_report --output data\calibration_report.json
```

7. Run real-money preflight:

```powershell
.\.venv\Scripts\python.exe -m src.rollout_status --profile real-money --threshold 0.50
```

Only if this prints `Status: READY`, set `MT5_EXECUTE_TRADES=True`.

## Start Scanner

```powershell
.\.venv\Scripts\python.exe -m src.scanner_worker --symbol XAUUSD --threshold 0.50 --loop --interval 1
```

If `MT5_EXECUTE_TRADES=True` and the real-money preflight is blocked, scanner startup now fails fast with `[Scanner Guard] Real-money rollout preflight BLOCKED`.

## Current Local Status

The latest local report still blocks real-money because live-policy metrics do not meet real-money drawdown/streak/expectancy requirements. That is intentional: VPS-ready means the bot refuses unsafe live execution instead of forcing real orders.
