# Forex SMC Dashboard Phase 3 Design - 2026-06-08

## Objective

Extend the local Streamlit dashboard into a safe operations console for demo forward-testing. Phase 3 makes it easier to inspect evidence, export reports, read logs, and copy the exact PowerShell commands needed to run scanner/dashboard/calibration tasks.

The dashboard remains read-only. Phase 3 does not place MT5 orders, retrain models, edit `.env`, delete files, or send Telegram messages.

## Phase 3 Scope

Phase 3 adds five capabilities:

1. Report export payloads for dashboard snapshot, signal evidence, confidence buckets, health checks, and forward evidence.
2. Scanner/backtest log inventory and safe tail viewer for `.log` files under `data/`.
3. Safe command previews for manual PowerShell execution.
4. Dashboard process hygiene notes so stale Streamlit import/cache issues are visible.
5. Manual cache refresh control using Streamlit cache clearing only.

These features improve operations and diagnostics. They do not change strategy entry logic, threshold policy, risk management, Telegram behavior, model training, or order execution.

## Data Layer

`src/dashboard_data.py` remains the owner of pure data helpers. Phase 3 adds tested helpers:

- `build_report_export_payload(...)`: builds JSON-safe report text from already loaded dashboard data.
- `build_log_inventory(base_dir)`: lists `.log` files in `data/` with size, modified time, and relative path.
- `read_log_tail(path, base_dir, max_lines)`: returns the last lines from a project log file and rejects paths outside the project.
- `build_safe_command_previews(base_dir)`: returns exact manual PowerShell commands for scanner, dashboard, calibration report, tests, and retrain preview.
- `build_dashboard_process_notes()`: returns read-only instructions for restarting Streamlit after dashboard helper/import changes.

All helpers must be deterministic and testable without MT5, Telegram, Streamlit, network, or live market access.

## Dashboard UI

`src/dashboard.py` keeps the same seven tabs. Phase 3 enriches existing areas:

- Command Center:
  - Safe command preview table.
  - Process hygiene notes for stale Streamlit servers.
  - Manual refresh button that clears dashboard cache and reruns Streamlit.
- Live Signal Monitor:
  - Download current filtered signal table as CSV.
  - Download selected signal detail as JSON.
- AI Learning:
  - Download confidence bucket summary as CSV.
- Backtest vs Forward:
  - Download forward evidence and dashboard report JSON.
  - Log inventory and tail viewer for `data/*.log`.

The UI should keep warnings and safety status explicit. Any command shown is copy-only text; the dashboard must not execute scanner/retrain/test commands.

## Safety Rules

Dashboard Phase 3 is still read-only:

- no MT5 order calls
- no retrain execution button
- no `.env` write
- no Telegram send
- no file deletion
- no mutation of labeled datasets
- no subprocess execution from Streamlit

The retrain command can be shown as a preview only, because the owner may want to run it manually after reviewing forward-test evidence.

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

Dashboard Phase 3 must not claim the bot is perfect, guaranteed profitable, or that the next trade will profit. It should help the owner monitor forward-test evidence, model freshness, logs, and manual operating steps with less guesswork.
