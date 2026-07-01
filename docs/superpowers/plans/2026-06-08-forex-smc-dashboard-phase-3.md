# Forex SMC Dashboard Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe read-only operations tooling to the Streamlit dashboard: exports, log viewer, command previews, refresh, and process hygiene guidance.

**Architecture:** Keep filesystem parsing, export payload construction, log safety, and command text generation in `src/dashboard_data.py`. Keep `src/dashboard.py` focused on rendering and downloads. Tests target pure helpers before UI wiring.

**Tech Stack:** Python, pandas, Streamlit, pytest.

---

## File Structure

- Modify: `tests/test_dashboard_data.py`
  - Add Phase 3 regression tests for export payloads, log inventory/tail safety, command previews, and process notes.
- Modify: `src/dashboard_data.py`
  - Add pure helper functions for Phase 3 operations data.
- Modify: `src/dashboard.py`
  - Render command previews, process notes, cache refresh, downloads, and log viewer.
- Modify: `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`
  - Record Phase 3 implementation, safety status, verification, and relaunch instructions.

## Task 1: Report Export Payload

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write the failing test**

```python
def test_report_export_payload_is_json_safe_and_source_aware():
    payload = build_report_export_payload(
        snapshot={"counts": {"accepted_signals": 1}, "env": {"ML_ACCEPT_THRESHOLD": "0.50"}},
        accepted_signals=[{"signal_id": "a", "source": "accepted", "confidence": 0.62}],
        shadow_signals=[{"signal_id": "s", "source": "shadow", "confidence": 0.32}],
        confidence_summary=pd.DataFrame([{"source": "shadow", "confidence_bucket": "0.30-0.40", "signal_count": 1}]),
        health_checks=[{"name": "data/sent_signals.json", "status": "ok"}],
        forward_summary={"accepted": {"total": 1}, "shadow": {"total": 1}},
        generated_at=datetime(2026, 6, 8, 16, 30, 0),
    )

    decoded = json.loads(payload["json"])

    assert payload["filename"] == "forex_smc_dashboard_report_2026-06-08_16-30-00.json"
    assert decoded["snapshot"]["counts"]["accepted_signals"] == 1
    assert decoded["signals"]["accepted"][0]["source"] == "accepted"
    assert decoded["signals"]["shadow"][0]["source"] == "shadow"
    assert decoded["non_claims"]["guaranteed_profit"] is False
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_report_export_payload_is_json_safe_and_source_aware -q
```

Expected: import error for `build_report_export_payload`.

- [ ] **Step 3: Implement minimal code**

Add `build_report_export_payload` that returns:

```python
{
    "filename": "forex_smc_dashboard_report_<timestamp>.json",
    "json": "<pretty JSON string>",
}
```

The JSON must include `generated_at`, `snapshot`, `signals.accepted`, `signals.shadow`, `confidence_summary`, `health_checks`, `forward_summary`, and `non_claims`.

- [ ] **Step 4: Run test to verify it passes**

Run the same targeted test and confirm pass.

## Task 2: Log Inventory And Safe Tail

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing tests**

```python
def test_log_inventory_lists_data_logs_newest_first(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    old_log = data_dir / "old_stdout.log"
    new_log = data_dir / "new_stdout.log"
    old_log.write_text("old\n", encoding="utf-8")
    new_log.write_text("new\n", encoding="utf-8")
    os.utime(old_log, (datetime(2026, 6, 8, 10, 0, 0).timestamp(), datetime(2026, 6, 8, 10, 0, 0).timestamp()))
    os.utime(new_log, (datetime(2026, 6, 8, 11, 0, 0).timestamp(), datetime(2026, 6, 8, 11, 0, 0).timestamp()))

    logs = build_log_inventory(base_dir=tmp_path)

    assert [row["name"] for row in logs] == ["new_stdout.log", "old_stdout.log"]
    assert logs[0]["relative_path"] == "data/new_stdout.log"


def test_read_log_tail_rejects_paths_outside_project(tmp_path):
    outside = tmp_path.parent / "outside.log"
    outside.write_text("secret\n", encoding="utf-8")

    tail = read_log_tail(outside, base_dir=tmp_path)

    assert tail["status"] == "rejected"
    assert tail["lines"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_log_inventory_lists_data_logs_newest_first tests\test_dashboard_data.py::test_read_log_tail_rejects_paths_outside_project -q
```

Expected: import errors for log helpers.

- [ ] **Step 3: Implement minimal code**

Add:

```python
def build_log_inventory(base_dir=BASE_DIR):
    ...

def read_log_tail(path, base_dir=BASE_DIR, max_lines=120):
    ...
```

Only allow files inside `<base_dir>/data` with suffix `.log`. Return a status object instead of raising for missing/rejected files.

- [ ] **Step 4: Run tests to verify pass**

Run the same targeted tests and confirm pass.

## Task 3: Safe Command Previews And Process Notes

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing tests**

```python
def test_safe_command_previews_are_copy_only_and_project_scoped(tmp_path):
    commands = build_safe_command_previews(base_dir=tmp_path)
    names = {command["name"] for command in commands}

    assert {"Start live scanner", "Start dashboard", "Run calibration report", "Run dashboard tests", "Preview retrain command"}.issubset(names)
    assert all(command["execution"] == "copy_only" for command in commands)
    assert all(str(tmp_path) in command["command"] for command in commands)


def test_dashboard_process_notes_are_read_only():
    notes = build_dashboard_process_notes()

    assert notes["mode"] == "read_only"
    assert "restart" in " ".join(notes["notes"]).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_safe_command_previews_are_copy_only_and_project_scoped tests\test_dashboard_data.py::test_dashboard_process_notes_are_read_only -q
```

Expected: import errors for command/process helpers.

- [ ] **Step 3: Implement minimal code**

Add fixed command preview records:

- Start live scanner: `.\.venv\Scripts\python.exe -m src.main`
- Start dashboard: `.\.venv\Scripts\streamlit.exe run src\dashboard.py --server.port 8501 --server.headless true`
- Run calibration report: `.\.venv\Scripts\python.exe -m src.calibration_report`
- Run dashboard tests: `.\.venv\Scripts\python.exe -m pytest tests\test_dashboard_data.py -q`
- Preview retrain command: `.\.venv\Scripts\python.exe -m src.model_trainer`

Each row must have `execution="copy_only"` and `safety` text.

- [ ] **Step 4: Run tests to verify pass**

Run the same targeted tests and confirm pass.

## Task 4: Streamlit UI Wiring

**Files:**

- Modify: `src/dashboard.py`

- [ ] **Step 1: Import Phase 3 helpers**

Import:

```python
build_dashboard_process_notes,
build_log_inventory,
build_report_export_payload,
build_safe_command_previews,
read_log_tail,
```

- [ ] **Step 2: Add cached data outputs**

In `cached_data()`, compute `log_inventory`, `command_previews`, and `process_notes`. Return them with the existing tuple.

- [ ] **Step 3: Add Command Center operations panel**

Render:

- cache refresh button using `st.cache_data.clear()` and `st.rerun()`
- command preview table
- process hygiene notes

- [ ] **Step 4: Add downloads and log viewer**

Render:

- CSV download for filtered signal table
- JSON download for selected signal detail
- CSV download for confidence bucket summary
- JSON report download in Backtest vs Forward
- log inventory table and tail viewer in Backtest vs Forward

- [ ] **Step 5: Compile dashboard**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
```

Expected: exit code 0.

## Task 5: Verification, Restart, And Memory

**Files:**

- Modify: `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`

- [ ] **Step 1: Run targeted dashboard tests**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

- [ ] **Step 2: Run compile check**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
```

- [ ] **Step 3: Run full suite**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

- [ ] **Step 4: Restart Streamlit cleanly**

Find and stop only Streamlit dashboard processes, then start one fresh server:

```powershell
Start-Process -FilePath 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\streamlit.exe' -ArgumentList 'run','src\dashboard.py','--server.port','8501','--server.headless','true' -WorkingDirectory 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer' -WindowStyle Hidden -PassThru
```

- [ ] **Step 5: HTTP check**

```powershell
Invoke-WebRequest -Uri 'http://localhost:8501' -UseBasicParsing -TimeoutSec 10
```

- [ ] **Step 6: Update dashboard memory**

Append Phase 3 completion status, safety status, verification output, and the exact manual launch command to `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`.

## Self-Review

Spec coverage:

- Report exports: Tasks 1 and 4.
- Log inventory and tail viewer: Tasks 2 and 4.
- Safe command previews: Tasks 3 and 4.
- Streamlit process hygiene and cache refresh: Tasks 3 and 4.
- Read-only safety: no task adds order execution, retraining execution, Telegram send, `.env` writes, dataset mutation, or file deletion.

Placeholder scan:

- No `TBD` or `TODO` implementation markers.
- Retraining is explicitly command-preview only.
