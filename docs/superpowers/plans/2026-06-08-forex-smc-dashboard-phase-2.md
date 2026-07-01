# Forex SMC Dashboard Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only dashboard Phase 2 analytics: signal drilldown, confidence bucket summaries, health checks, and forward evidence summaries.

**Architecture:** Keep all computations in `src/dashboard_data.py` and keep `src/dashboard.py` as rendering only. Tests target pure data helpers so dashboard behavior is stable without launching Streamlit.

**Tech Stack:** Python, pandas, Streamlit, pytest.

---

## File Structure

- Modify: `src/dashboard_data.py`
  - Add confidence bucket helpers, detail builder, health checks, and forward summary.
- Modify: `tests/test_dashboard_data.py`
  - Add regression tests for Phase 2 helper behavior.
- Modify: `src/dashboard.py`
  - Render the new helper outputs in existing tabs.
- Modify: `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`
  - Record Phase 2 implementation and verification status.

## Task 1: Preserve Signal Features And Detail Lookup

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `build_signal_detail` and assert:

```python
def test_signal_detail_finds_accepted_leg_features():
    accepted = flatten_sent_signals({
        "sig-a": {
            "time_sent": "2026-06-08 10:00:00",
            "timeframe": "H1",
            "direction": "BULL",
            "type": "FVG",
            "price_0.5": 2300.0,
            "price_0.618": 2298.0,
            "probability_0.5": 0.61,
            "probability_0.618": 0.72,
            "ticket_a": 111,
            "ticket_b": 222,
            "features_0.5": {"atr_14": 10.5, "floop_strength": 6.0},
            "features_0.618": {"atr_14": 11.5, "floop_strength": 7.0},
        }
    })

    detail = build_signal_detail("sig-a_0.618", accepted, [])

    assert detail["signal"]["signal_id"] == "sig-a_0.618"
    assert detail["signal"]["leg"] == "0.618"
    assert detail["features"] == {"atr_14": 11.5, "floop_strength": 7.0}
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: import error for `build_signal_detail` or missing feature fields.

- [ ] **Step 3: Implement minimal code**

Add feature preservation in `flatten_sent_signals` and `flatten_shadow_signals`. Add:

```python
def build_signal_detail(signal_id, accepted_signals, shadow_signals):
    for row in [*(accepted_signals or []), *(shadow_signals or [])]:
        if row.get("signal_id") == signal_id:
            features = row.get("features") if isinstance(row.get("features"), dict) else {}
            return {"signal": {key: value for key, value in row.items() if key != "features"}, "features": features}
    return None
```

- [ ] **Step 4: Verify GREEN**

Run the dashboard data tests and confirm pass.

## Task 2: Confidence Bucket Summary

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing tests**

Add tests for `assign_confidence_bucket` and `build_confidence_bucket_summary`:

```python
def test_confidence_bucket_summary_counts_results():
    signals = [
        {"source": "shadow", "confidence": 0.29, "timeframe": "M15", "strategy": "FVG", "status": "resolved", "result": "tp"},
        {"source": "shadow", "confidence": 0.35, "timeframe": "M15", "strategy": "FVG", "status": "resolved", "result": "sl"},
        {"source": "accepted", "confidence": 0.61, "timeframe": "H1", "strategy": "BPR", "status": None, "result": None},
    ]

    summary = build_confidence_bucket_summary(signals)

    low = summary[(summary["source"] == "shadow") & (summary["confidence_bucket"] == "0.00-0.30")].iloc[0]
    assert low["signal_count"] == 1
    assert low["tp_count"] == 1
    assert low["winrate_pct"] == 100.0
```

- [ ] **Step 2: Verify RED**

Run dashboard data tests and confirm missing helper failure.

- [ ] **Step 3: Implement minimal code**

Use fixed bucket labels:

```text
0.00-0.30, 0.30-0.40, 0.40-0.50, 0.50-0.60, 0.60-0.70, 0.70-0.80, 0.80-0.90, 0.90-1.00, unknown
```

Group by `source`, `confidence_bucket`, `timeframe`, and `strategy`.

- [ ] **Step 4: Verify GREEN**

Run dashboard data tests and confirm pass.

## Task 3: Health Checks And Forward Summary

**Files:**

- Modify: `tests/test_dashboard_data.py`
- Modify: `src/dashboard_data.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert missing files are reported and forward summary stays source-aware:

```python
def test_dashboard_health_checks_reports_missing_files(tmp_path):
    checks = build_dashboard_health_checks(base_dir=tmp_path, now=datetime(2026, 6, 8, 12, 0, 0))
    missing = [check for check in checks if check["status"] == "missing"]
    assert any(check["name"] == "data/sent_signals.json" for check in missing)


def test_forward_evidence_summary_keeps_sources_separate():
    accepted = [{"source": "accepted", "result": "tp"}, {"source": "accepted", "result": "sl"}]
    shadow = [{"source": "shadow", "result": "tp"}, {"source": "shadow", "status": "open"}]

    summary = summarize_forward_evidence(accepted, shadow)

    assert summary["accepted"]["total"] == 2
    assert summary["accepted"]["tp"] == 1
    assert summary["shadow"]["open"] == 1
```

- [ ] **Step 2: Verify RED**

Run dashboard data tests and confirm missing helper failure.

- [ ] **Step 3: Implement minimal code**

`build_dashboard_health_checks` checks required dashboard files and model files. `summarize_forward_evidence` counts `total`, `tp`, `sl`, `expired`, `open`, and `winrate_pct` for accepted and shadow independently.

- [ ] **Step 4: Verify GREEN**

Run dashboard data tests and confirm pass.

## Task 4: Streamlit UI Wiring

**Files:**

- Modify: `src/dashboard.py`

- [ ] **Step 1: Wire imports**

Import:

```python
from src.dashboard_data import (
    build_confidence_bucket_summary,
    build_dashboard_health_checks,
    build_signal_detail,
    summarize_forward_evidence,
)
```

- [ ] **Step 2: Add cached data outputs**

Return `health_checks` and `forward_summary` from `cached_data()`.

- [ ] **Step 3: Render Phase 2 views**

Add:

- health check table in Command Center
- signal detail selector in Live Signal Monitor
- confidence bucket summary in AI Learning
- forward evidence metrics/table in Backtest vs Forward

- [ ] **Step 4: Compile**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
```

Expected: exit code 0.

## Task 5: Verification And Memory

**Files:**

- Modify: `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`

- [ ] **Step 1: Run targeted tests**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

- [ ] **Step 2: Run full suite**

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

- [ ] **Step 3: Check dashboard HTTP if server is running**

```powershell
C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe -Command "Invoke-WebRequest -Uri 'http://localhost:8501' -UseBasicParsing -TimeoutSec 10 | Select-Object StatusCode,StatusDescription"
```

- [ ] **Step 4: Update memory doc**

Append Phase 2 completion status, verification output, and relaunch command to `docs/DASHBOARD_PROJECT_MEMORY_2026-06-08.md`.

## Self-Review

Spec coverage:

- Signal detail drilldown: Tasks 1 and 4.
- Confidence and shadow-learning analytics: Tasks 2 and 4.
- Operational health checks: Tasks 3 and 4.
- Backtest vs forward summary: Tasks 3 and 4.
- Read-only safety: no task adds order, retrain, Telegram, env-write, or delete actions.

Placeholder scan:

- No `TBD` or `TODO` implementation markers.
- Deferred operations are explicitly assigned to future Dashboard Phase 3.
