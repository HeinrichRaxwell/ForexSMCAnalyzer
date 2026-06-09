# Forex SMC Dashboard Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only readiness gate to the dashboard so demo forward-test, model freshness, and retraining status are visible as actionable evidence.

**Architecture:** Implement all readiness decisions as pure helpers in `src/dashboard_data.py`. Render the resulting rows and summaries in `src/dashboard.py`. Keep tests focused on helper behavior and run the full suite before restarting Streamlit.

**Tech Stack:** Python, pandas, Streamlit, pytest.

---

## File Structure

- Modify: `tests/test_dashboard_data.py`
  - Add tests for model freshness details, retraining readiness, forward readiness, and combined readiness report.
- Modify: `src/dashboard_data.py`
  - Add Phase 4 readiness helpers.
- Modify: `src/dashboard.py`
  - Render readiness panels in Command Center, AI Learning, and Backtest vs Forward.
- Modify: `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`
  - Record Phase 4 scope, behavior, verification, and safety status.

## Task 1: Model Freshness Details

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write the failing test**

```python
def test_model_freshness_details_flags_stale_models():
    latest_data = datetime(2026, 6, 8, 16, 0, 0)
    model_time = datetime(2026, 6, 8, 12, 0, 0)

    rows = build_model_freshness_details(
        latest_data_time=latest_data,
        model_inventory=[{"name": "smc_xgb_classifier.joblib", "modified_at": model_time}],
        now=datetime(2026, 6, 8, 17, 0, 0),
    )

    assert rows[0]["status"] == "stale"
    assert rows[0]["data_lag_hours"] == 4.0
    assert rows[0]["model_age_hours"] == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_model_freshness_details_flags_stale_models -q
```

Expected: import error for `build_model_freshness_details`.

- [ ] **Step 3: Implement minimal code**

Add `build_model_freshness_details` that returns one row per model with:

- `name`
- `status`
- `modified_at`
- `latest_data_time`
- `data_lag_hours`
- `model_age_hours`
- `detail`

- [ ] **Step 4: Run test to verify pass**

Run the same targeted test and confirm pass.

## Task 2: Retraining Readiness

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing tests**

```python
def test_retraining_readiness_cautions_when_model_stale_even_without_new_trade_count():
    snapshot = {
        "counts": {"real_labeled_rows": 10, "shadow_labeled_rows": 5},
        "learning": {"new_trades_since_last_train": 0},
        "env": {"ML_RETRAIN_THRESHOLD": "5"},
    }

    result = build_retraining_readiness(
        snapshot=snapshot,
        latest_data_time=datetime(2026, 6, 8, 16, 0, 0),
        model_inventory=[{"name": "smc_xgb_classifier.joblib", "modified_at": datetime(2026, 6, 8, 12, 0, 0)}],
    )

    assert result["status"] == "caution"
    assert result["new_trades_since_last_train"] == 0
    assert any("newer than model" in reason for reason in result["reasons"])


def test_retraining_readiness_ready_when_threshold_is_reached():
    snapshot = {
        "counts": {"real_labeled_rows": 10, "shadow_labeled_rows": 5},
        "learning": {"new_trades_since_last_train": 6},
        "env": {"ML_RETRAIN_THRESHOLD": "5"},
    }

    result = build_retraining_readiness(snapshot=snapshot, latest_data_time=None, model_inventory=[])

    assert result["status"] == "ready"
    assert result["manual_action"] == "Run model trainer, then regenerate calibration report."
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_retraining_readiness_cautions_when_model_stale_even_without_new_trade_count tests\test_dashboard_data.py::test_retraining_readiness_ready_when_threshold_is_reached -q
```

Expected: import error for `build_retraining_readiness`.

- [ ] **Step 3: Implement minimal code**

Rules:

- `blocked` when total labeled rows are zero.
- `ready` when `new_trades_since_last_train >= ML_RETRAIN_THRESHOLD`.
- `caution` when latest data is newer than any model.
- `ready` otherwise means no retrain needed right now.

- [ ] **Step 4: Run tests to verify pass**

Run the same targeted tests and confirm pass.

## Task 3: Forward Readiness Gate

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing tests**

```python
def test_forward_readiness_gate_blocks_when_resolved_evidence_is_low():
    result = build_forward_readiness_gate(
        calibration_report={"overall": {"profit_factor": 1.2, "max_consecutive_losses": 3}},
        forward_summary={"accepted": {"tp": 1, "sl": 0}, "shadow": {"tp": 1, "sl": 0}},
        minimum_resolved=5,
    )

    assert result["status"] == "blocked"
    assert result["resolved_forward_trades"] == 2


def test_forward_readiness_gate_ready_with_enough_evidence_and_clean_risk():
    result = build_forward_readiness_gate(
        calibration_report={
            "overall": {"profit_factor": 1.4, "max_consecutive_losses": 4},
            "recommendation": {"threshold": "0.50", "expectancy_r": 1.1},
        },
        forward_summary={"accepted": {"tp": 4, "sl": 2}, "shadow": {"tp": 4, "sl": 1}},
        minimum_resolved=10,
    )

    assert result["status"] == "ready"
    assert result["recommended_threshold"] == "0.50"
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_forward_readiness_gate_blocks_when_resolved_evidence_is_low tests\test_dashboard_data.py::test_forward_readiness_gate_ready_with_enough_evidence_and_clean_risk -q
```

Expected: import error for `build_forward_readiness_gate`.

- [ ] **Step 3: Implement minimal code**

Rules:

- Count resolved forward trades as accepted `tp+sl` plus shadow `tp+sl`.
- `blocked` when resolved count is below `minimum_resolved`.
- `caution` when profit factor is below `1.10` or max consecutive losses exceeds `10`.
- `ready` when evidence and calibration risk rules pass.

- [ ] **Step 4: Run tests to verify pass**

Run the same targeted tests and confirm pass.

## Task 4: Combined Readiness Report

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing test**

```python
def test_dashboard_readiness_report_uses_worst_status():
    report = build_dashboard_readiness_report(
        snapshot={
            "counts": {"real_labeled_rows": 10, "shadow_labeled_rows": 5},
            "learning": {"new_trades_since_last_train": 0},
            "env": {"ML_RETRAIN_THRESHOLD": "5"},
        },
        latest_data_time=None,
        model_inventory=[],
        calibration_report={"overall": {"profit_factor": 1.0, "max_consecutive_losses": 2}},
        forward_summary={"accepted": {"tp": 1, "sl": 0}, "shadow": {"tp": 0, "sl": 0}},
        health_checks=[{"name": "data/sent_signals.json", "status": "ok"}],
        minimum_forward_resolved=5,
    )

    assert report["overall_status"] == "blocked"
    assert any(row["component"] == "Forward Evidence" for row in report["checks"])
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_dashboard_readiness_report_uses_worst_status -q
```

Expected: import error for `build_dashboard_readiness_report`.

- [ ] **Step 3: Implement minimal code**

Combine retraining, forward evidence, model freshness, and health checks. Worst status order is:

```text
blocked > caution > ready
```

- [ ] **Step 4: Run test to verify pass**

Run the same targeted test and confirm pass.

## Task 5: Streamlit UI Wiring

**Files:**

- Modify: `src/dashboard.py`

- [ ] **Step 1: Import Phase 4 helpers**

Import:

```python
build_dashboard_readiness_report,
build_model_freshness_details,
```

- [ ] **Step 2: Add cached data outputs**

In `cached_data()`, compute:

- `latest_data_time` from labeled and shadow labeled frames
- `model_freshness_details`
- `readiness_report`

- [ ] **Step 3: Render Command Center readiness**

Add:

- overall readiness status metric
- readiness checks table
- model freshness detail table

- [ ] **Step 4: Render learning and forward gate details**

Add:

- retraining readiness details in AI Learning
- forward readiness details in Backtest vs Forward

- [ ] **Step 5: Compile**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
```

Expected: exit code 0.

## Task 6: Verification, Restart, And Memory

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

Stop only dashboard Streamlit processes and start one fresh server on port 8501.

- [ ] **Step 5: HTTP check**

```powershell
Invoke-WebRequest -Uri 'http://localhost:8501' -UseBasicParsing -TimeoutSec 10
```

- [ ] **Step 6: Update dashboard memory**

Append Phase 4 completion status, safety status, verification output, and manual commands to `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`.

## Self-Review

Spec coverage:

- Model freshness details: Tasks 1 and 5.
- Retraining readiness: Tasks 2 and 5.
- Forward-test readiness gate: Tasks 3 and 5.
- Combined report: Tasks 4 and 5.
- Read-only safety: no task adds MT5 order, retrain execution, Telegram send, `.env` write, subprocess execution, file deletion, or dataset mutation.

Placeholder scan:

- No deferred implementation markers.
- Any manual action remains text-only guidance.
