# Forex SMC Dashboard Phase 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable execution diagnostics for Pivot/Rejection/key-level setups and prevent low-confidence registry rows from blocking later live execution.

**Architecture:** Keep execution decisions in `src/scanner_worker.py`, dashboard parsing/diagnostics in `src/dashboard_data.py`, and Streamlit rendering in `src/dashboard.py`. The UI stays read-only and only explains decisions from stored data.

**Tech Stack:** Python, pytest, pandas, Streamlit, MetaTrader5 integration through existing scanner/execution modules.

---

### Task 1: Scanner Shadow-to-Live Promotion

**Files:**
- Modify: `src/scanner_worker.py`
- Test: `tests/test_scanner_entry_decisions.py`

- [x] **Step 1: Write the failing test**

Add tests that call `should_promote_low_confidence_record()` for single and dual records.

- [x] **Step 2: Verify RED**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_scanner_entry_decisions.py::test_low_confidence_registry_record_can_promote_to_live_single_execution -q
```

Expected: import failure because the helper does not exist.

- [x] **Step 3: Implement helper and scanner fall-through**

Add `should_promote_low_confidence_record(sig_data, ticket_fields)` and use it before the existing duplicate-signal branch for dual and single setups.

- [x] **Step 4: Verify GREEN**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_scanner_entry_decisions.py -q
```

Expected: all tests pass.

### Task 2: Dashboard Execution Diagnostics

**Files:**
- Modify: `src/dashboard_data.py`
- Modify: `src/dashboard.py`
- Test: `tests/test_dashboard_data.py`

- [x] **Step 1: Write failing tests**

Add tests for low-confidence registry source, Pivot shadow threshold reason, and accepted-no-ticket diagnostics.

- [x] **Step 2: Verify RED**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py::test_execution_diagnostics_explain_pivot_shadow_below_threshold -q
```

Expected: import failure because `build_execution_decision_diagnostics` does not exist.

- [x] **Step 3: Implement data helper**

Add `build_execution_decision_diagnostics()` and key-level context inference to `src/dashboard_data.py`.

- [x] **Step 4: Render diagnostics**

Add read-only execution diagnostics and key-level/pivot diagnostics tables to Live Signal Monitor in `src/dashboard.py`.

- [x] **Step 5: Verify GREEN**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: all dashboard data tests pass.

### Task 3: Documentation and Final Verification

**Files:**
- Create: `docs/superpowers/specs/2026-06-08-forex-smc-dashboard-phase-7-design.md`
- Create: `docs/superpowers/plans/2026-06-08-forex-smc-dashboard-phase-7.md`
- Create: `docs/ML_PHASE_7_EXECUTION_DIAGNOSTICS_2026-06-08.md`
- Modify: `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`

- [x] **Step 1: Document root cause and limits**

Record that observed Pivot rows were below threshold and therefore correctly stayed shadow, while scanner promotion had a concrete bug for later high-confidence promotion.

- [x] **Step 2: Run broad verification**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m compileall -q src tests
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Actual:

- targeted execution/dashboard suite: `79 passed`
- compile all `src` and `tests`: exit code 0
- full suite: `201 passed, 38 warnings`

- [x] **Step 3: Refresh model freshness if needed**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m src.model_trainer
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m src.calibration_report
```

Actual:

- `src.model_trainer` ran successfully.
- Challenger was rejected by champion gate: old champion accuracy/winrate beat the new challenger.
- Model joblib timestamps stayed old by design because model files were not overwritten.
- `src.calibration_report` regenerated successfully with `sample_count=1170`, `profit_factor=1.06`, and `max_consecutive_losses=33`.
- Dashboard readiness now explains that the stale model warning can persist when champion validation retains the older model.
