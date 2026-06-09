# Forex SMC Dashboard Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only System QA and Feature Coverage evidence to the dashboard.

**Architecture:** Implement QA inventory and checklist helpers as pure functions in `src/dashboard_data.py`, then render them in Command Center from `src/dashboard.py`. Tests cover helper behavior before UI wiring.

**Tech Stack:** Python, pandas, Streamlit, pytest.

---

## File Structure

- Modify: `tests/test_dashboard_data.py`
  - Add tests for phase document inventory, code inventory, verification checklist, and feature coverage matrix.
- Modify: `src/dashboard_data.py`
  - Add Phase 5 QA helpers.
- Modify: `src/dashboard.py`
  - Render QA summary in Command Center.
- Modify: `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`
  - Record Phase 5 scope, safety, verification, and dashboard status.

## Task 1: Phase Document Inventory

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write the failing test**

```python
def test_phase_document_inventory_reports_specs_and_plans(tmp_path):
    specs = tmp_path / "docs" / "superpowers" / "specs"
    plans = tmp_path / "docs" / "superpowers" / "plans"
    specs.mkdir(parents=True)
    plans.mkdir(parents=True)
    (specs / "2026-06-08-forex-smc-dashboard-phase-2-design.md").write_text("spec", encoding="utf-8")
    (plans / "2026-06-08-forex-smc-dashboard-phase-2.md").write_text("plan", encoding="utf-8")
    (tmp_path / "docs" / "DASHBOARD_PROJECT_MEMORY_2026-06-08.md").write_text("## Phase 2 Implementation Completed", encoding="utf-8")

    rows = build_phase_document_inventory(base_dir=tmp_path, max_phase=2)
    phase_2 = next(row for row in rows if row["phase"] == "Phase 2")

    assert phase_2["status"] == "ready"
    assert phase_2["has_spec"] is True
    assert phase_2["has_plan"] is True
    assert phase_2["has_memory"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_phase_document_inventory_reports_specs_and_plans -q
```

Expected: import error for `build_phase_document_inventory`.

- [ ] **Step 3: Implement minimal code**

Add `build_phase_document_inventory(base_dir=BASE_DIR, max_phase=5)` that checks specs, plans, and dashboard memory.

- [ ] **Step 4: Run test to verify pass**

Run the same targeted test and confirm pass.

## Task 2: Code Inventory And Verification Checklist

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing tests**

```python
def test_code_inventory_counts_source_and_tests(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "src" / "dashboard.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "src" / "dashboard_data.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "tests" / "test_dashboard_data.py").write_text("def test_ok(): assert True", encoding="utf-8")
    (tmp_path / "docs" / "a.md").write_text("doc", encoding="utf-8")
    (tmp_path / "data" / "a.json").write_text("{}", encoding="utf-8")

    inventory = build_code_inventory(base_dir=tmp_path)

    assert inventory["source_files"] == 2
    assert inventory["test_files"] == 1
    assert inventory["dashboard_files"] == 2
    assert inventory["docs_files"] == 1
    assert inventory["data_files"] == 1


def test_verification_checklist_is_copy_only_and_includes_broad_checks(tmp_path):
    rows = build_verification_checklist(base_dir=tmp_path)
    names = {row["name"] for row in rows}

    assert {"Dashboard data tests", "Compile all src/tests", "Full pytest suite", "Dashboard HTTP check"}.issubset(names)
    assert all(row["execution"] == "copy_only" for row in rows)
    assert all(str(tmp_path) in row["command"] or "localhost:8501" in row["command"] for row in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_code_inventory_counts_source_and_tests tests\test_dashboard_data.py::test_verification_checklist_is_copy_only_and_includes_broad_checks -q
```

Expected: import errors for `build_code_inventory` and `build_verification_checklist`.

- [ ] **Step 3: Implement minimal code**

Add both helpers with deterministic rows and copy-only command previews.

- [ ] **Step 4: Run tests to verify pass**

Run the same targeted tests and confirm pass.

## Task 3: Feature Coverage Matrix

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing test**

```python
def test_feature_coverage_matrix_blocks_when_phase_docs_missing():
    rows = build_feature_coverage_matrix(
        phase_inventory=[{"phase": "Phase 1", "status": "blocked", "detail": "missing"}],
        code_inventory={"source_files": 2, "test_files": 1, "dashboard_files": 2},
        readiness_report={"overall_status": "ready"},
        health_checks=[{"name": "data/sent_signals.json", "status": "ok"}],
    )

    phase_row = next(row for row in rows if row["feature_area"] == "Dashboard Phase Documentation")
    assert phase_row["status"] == "blocked"
    assert "missing" in phase_row["evidence"]
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_feature_coverage_matrix_blocks_when_phase_docs_missing -q
```

Expected: import error for `build_feature_coverage_matrix`.

- [ ] **Step 3: Implement minimal code**

Add rows for:

- Dashboard Phase Documentation
- Code/Test Inventory
- Runtime Readiness
- Data/Model Health

- [ ] **Step 4: Run test to verify pass**

Run the same targeted test and confirm pass.

## Task 4: Streamlit UI Wiring

**Files:**

- Modify: `src/dashboard.py`

- [ ] **Step 1: Import Phase 5 helpers**

Import:

```python
build_code_inventory,
build_feature_coverage_matrix,
build_phase_document_inventory,
build_verification_checklist,
```

- [ ] **Step 2: Add cached data outputs**

In `cached_data()`, compute phase inventory, code inventory, verification checklist, and feature coverage matrix.

- [ ] **Step 3: Render Command Center QA panel**

Render:

- System QA summary metrics
- Feature coverage matrix
- Phase document inventory
- Verification checklist
- Code inventory

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

- [ ] **Step 2: Compile all source and tests**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m compileall -q src tests
```

- [ ] **Step 3: Run full suite**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

- [ ] **Step 4: Restart dashboard and HTTP check**

Stop only dashboard Streamlit processes, start one fresh server, then check `http://localhost:8501`.

- [ ] **Step 5: Update memory**

Append Phase 5 completion and verification details to `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`.

## Self-Review

Spec coverage:

- Phase document inventory: Tasks 1 and 4.
- Code/test inventory: Tasks 2 and 4.
- Verification checklist: Tasks 2 and 4.
- Feature coverage matrix: Tasks 3 and 4.
- Read-only safety: no task executes shell, MT5, Telegram, trainer, scanner, `.env` writes, file deletion, or dataset mutation from Streamlit.

Placeholder scan:

- No deferred implementation markers.
- All verification commands remain explicit and manual in the dashboard.
