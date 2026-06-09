# Forex SMC Dashboard Phase 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only Strategy Formula QA evidence to the dashboard.

**Architecture:** Add pure formula inventory and PineScript matrix helpers in `src/dashboard_data.py`, test them first, then render the evidence in `src/dashboard.py` under Strategy Performance.

**Tech Stack:** Python, pandas, Streamlit, pytest.

---

## File Structure

- Modify: `tests/test_dashboard_data.py`
  - Add tests for formula inventory, PineScript matrix, formula verification checklist, and formula QA report.
- Modify: `src/dashboard_data.py`
  - Add Phase 6 formula QA helpers.
- Modify: `src/dashboard.py`
  - Render formula QA panels in Strategy Performance.
- Modify: `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`
  - Record Phase 6 scope, safety, verification, and known formula caveats.

## Task 1: Strategy Formula Inventory

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write the failing test**

```python
def test_strategy_formula_inventory_marks_ready_when_source_and_tests_exist(tmp_path):
    (tmp_path / "src" / "indicators").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "indicators" / "floop.py").write_text("def x(): pass", encoding="utf-8")
    (tmp_path / "tests" / "test_floop.py").write_text("def test_x(): assert True", encoding="utf-8")

    rows = build_strategy_formula_inventory(base_dir=tmp_path)
    floop = next(row for row in rows if row["formula_area"] == "FLoOP Pro")

    assert floop["status"] == "ready"
    assert "src/indicators/floop.py" in floop["source_files"]
    assert "tests/test_floop.py" in floop["test_files"]
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_strategy_formula_inventory_marks_ready_when_source_and_tests_exist -q
```

Expected: import error for `build_strategy_formula_inventory`.

- [ ] **Step 3: Implement minimal code**

Add formula areas with source/test evidence:

- SMC Fibonacci Structures
- Rejection Logic
- Pivot Classic/Rejection
- FLoOP Pro
- KNN SuperTrend
- Volume Clusters
- Scanner Entry Decisions
- Active Trade Management
- Telegram Signal Formatting

- [ ] **Step 4: Run test to verify pass**

Run the same targeted test and confirm pass.

## Task 2: PineScript Translation Matrix

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing test**

```python
def test_pinescript_translation_matrix_marks_missing_full_ports(tmp_path):
    pine_dir = tmp_path / "PineScripts"
    pine_dir.mkdir()
    (pine_dir / "Machine Learning RSI  AI Classification & Ranking (Zeiierman).txt").write_text("pine", encoding="utf-8")

    rows = build_pinescript_translation_matrix(base_dir=tmp_path)
    ml_rsi = next(row for row in rows if row["pinescript"] == "Machine Learning RSI AI Classification & Ranking")

    assert ml_rsi["status"] == "blocked"
    assert "Belum full port" in ml_rsi["detail"]
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_pinescript_translation_matrix_marks_missing_full_ports -q
```

Expected: import error for `build_pinescript_translation_matrix`.

- [ ] **Step 3: Implement minimal code**

Return five PineScript rows with known mapping:

- FLoOP Pro: `ready`
- AI SuperTrend KNN: `caution`
- Clusters Volume Profile LuxAlgo: `caution`
- Machine Learning RSI AI Classification & Ranking: `blocked`
- Multi-Timeframe Volume Profiles: `blocked`

- [ ] **Step 4: Run test to verify pass**

Run the same targeted test and confirm pass.

## Task 3: Formula Verification Checklist And QA Report

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing tests**

```python
def test_formula_verification_checklist_is_copy_only(tmp_path):
    rows = build_formula_verification_checklist(base_dir=tmp_path)

    assert any("test_floop.py" in row["command"] for row in rows)
    assert all(row["execution"] == "copy_only" for row in rows)


def test_formula_qa_report_uses_worst_status():
    report = build_formula_qa_report(
        strategy_inventory=[{"formula_area": "FLoOP Pro", "status": "ready"}],
        pinescript_matrix=[{"pinescript": "Multi-Timeframe Volume Profiles", "status": "blocked", "detail": "Belum full port"}],
    )

    assert report["overall_status"] == "blocked"
    assert any(row["status"] == "blocked" for row in report["checks"])
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_formula_verification_checklist_is_copy_only tests\test_dashboard_data.py::test_formula_qa_report_uses_worst_status -q
```

Expected: import errors for formula checklist/report helpers.

- [ ] **Step 3: Implement minimal code**

Add copy-only commands and worst-status report using existing `_worst_status`.

- [ ] **Step 4: Run tests to verify pass**

Run the same targeted tests and confirm pass.

## Task 4: Streamlit UI Wiring

**Files:**

- Modify: `src/dashboard.py`

- [ ] **Step 1: Import Phase 6 helpers**

Import:

```python
build_formula_qa_report,
build_formula_verification_checklist,
build_pinescript_translation_matrix,
build_strategy_formula_inventory,
```

- [ ] **Step 2: Add cached data outputs**

In `cached_data()`, compute formula inventory, PineScript matrix, formula verification checklist, and formula QA report.

- [ ] **Step 3: Render Strategy Performance Formula QA**

Render:

- Formula QA overall status
- Strategy formula inventory
- PineScript translation matrix
- Formula verification checklist and command preview

- [ ] **Step 4: Compile**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
```

Expected: exit code 0.

## Task 5: Verification, Restart, And Memory

**Files:**

- Modify: `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`

- [ ] **Step 1: Run dashboard tests**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

- [ ] **Step 2: Run formula targeted tests**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_fibo_detector.py tests\test_floop.py tests\test_pivots.py tests\test_rejection.py tests\test_imbalances.py tests\test_breakers_swapzones.py tests\test_knn_classifier.py tests\test_volume_clusters.py tests\test_scanner_entry_decisions.py tests\test_scanner_market_orders.py tests\test_active_trade_management.py -q
```

- [ ] **Step 3: Run compileall**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m compileall -q src tests
```

- [ ] **Step 4: Run full suite**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

- [ ] **Step 5: Restart dashboard and HTTP check**

Stop only dashboard Streamlit processes, start one fresh server, then check `http://localhost:8501`.

- [ ] **Step 6: Update memory**

Append Phase 6 completion, verification output, total phase status, and formula caveats to `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`.

## Self-Review

Spec coverage:

- Strategy formula inventory: Tasks 1 and 4.
- PineScript matrix: Tasks 2 and 4.
- Formula verification checklist: Tasks 3 and 4.
- Formula QA report: Tasks 3 and 4.
- Targeted formula verification: Task 5.
- Read-only safety: no task executes MT5, scanner, trainer, Telegram, shell commands, `.env` writes, file deletion, or dataset mutation from Streamlit.

Placeholder scan:

- No deferred implementation markers.
- Missing/partial PineScript ports are explicit and not hidden.
