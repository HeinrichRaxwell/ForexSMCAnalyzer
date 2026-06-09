# Forex SMC Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local read-only Streamlit dashboard for monitoring Forex SMC Analyzer signals, learning, calibration, trade management, and backtest vs forward-test evidence.

**Architecture:** Keep data parsing separate from UI. `src/dashboard_data.py` owns file reading, normalization, flattening, and metrics. `src/dashboard.py` owns Streamlit rendering and user filters. Tests target the data layer so the dashboard remains reliable even when optional files are missing.

**Tech Stack:** Python, pandas, Streamlit, pytest, existing CSV/JSON artifacts.

---

## File Structure

- Create: `src/dashboard_data.py`
  - Responsibility: read project files, normalize records, compute metrics, expose a dashboard snapshot.
- Create: `src/dashboard.py`
  - Responsibility: Streamlit UI, tabs, filters, tables, charts, warnings.
- Create: `tests/test_dashboard_data.py`
  - Responsibility: test missing-file handling, signal flattening, learning metrics, calibration parsing, and model freshness warnings.
- Modify: `requirements.txt`
  - Add `streamlit>=1.35.0`.

## Task 1: Dashboard Data Loader

**Files:**

- Create: `src/dashboard_data.py`
- Test: `tests/test_dashboard_data.py`

- [ ] **Step 1: Write failing tests for safe file loading**

Add this to `tests/test_dashboard_data.py`:

```python
import json
import pandas as pd

from src.dashboard_data import (
    load_csv_safe,
    load_json_safe,
)


def test_load_json_safe_returns_default_for_missing_file(tmp_path):
    result = load_json_safe(tmp_path / "missing.json", default={})
    assert result == {}


def test_load_csv_safe_returns_empty_frame_for_missing_file(tmp_path):
    result = load_csv_safe(tmp_path / "missing.csv")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_load_json_safe_reads_existing_file(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert load_json_safe(path, default={}) == {"ok": True}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: import failure because `src.dashboard_data` does not exist.

- [ ] **Step 3: Implement safe loaders**

Create `src/dashboard_data.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def load_json_safe(path: str | Path, default: Any = None) -> Any:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return default
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_csv_safe(path: str | Path) -> pd.DataFrame:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(resolved)
    except Exception:
        return pd.DataFrame()
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: all tests pass.

## Task 2: Flatten Accepted And Shadow Signals

**Files:**

- Modify: `src/dashboard_data.py`
- Modify: `tests/test_dashboard_data.py`

- [ ] **Step 1: Write failing tests for signal flattening**

Append to `tests/test_dashboard_data.py`:

```python
from src.dashboard_data import flatten_sent_signals, flatten_shadow_signals


def test_flatten_sent_signals_emits_dual_legs():
    data = {
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
            "outcome_a_recorded": True,
            "outcome_b_recorded": False,
        }
    }

    rows = flatten_sent_signals(data)

    assert len(rows) == 2
    assert rows[0]["source"] == "accepted"
    assert rows[0]["signal_id"] == "sig-a_0.5"
    assert rows[0]["leg"] == "0.5"
    assert rows[0]["confidence"] == 0.61
    assert rows[1]["leg"] == "0.618"
    assert rows[1]["ticket_id"] == 222


def test_flatten_shadow_signals_keeps_confidence_and_result():
    data = {
        "shadow-a": {
            "signal_id": "shadow-a",
            "status": "resolved",
            "result": "tp",
            "symbol": "XAUUSD",
            "timeframe": "M15",
            "strategy": "FVG",
            "direction_name": "BEAR",
            "leg": "0.5",
            "entry_price": 2300.0,
            "sl_price": 2308.0,
            "tp_price": 2288.0,
            "confidence": 0.37,
            "accept_threshold": 0.50,
            "created_at": "2026-06-08 10:00:00",
        }
    }

    rows = flatten_shadow_signals(data)

    assert rows == [{
        "source": "shadow",
        "signal_id": "shadow-a",
        "status": "resolved",
        "result": "tp",
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "strategy": "FVG",
        "direction": "BEAR",
        "leg": "0.5",
        "entry_price": 2300.0,
        "sl_price": 2308.0,
        "tp_price": 2288.0,
        "confidence": 0.37,
        "accept_threshold": 0.50,
        "ticket_id": None,
        "time": None,
        "created_at": "2026-06-08 10:00:00",
        "latest_seen_at": None,
        "resolved_at": None,
    }]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: functions not defined.

- [ ] **Step 3: Implement signal flattening**

Append to `src/dashboard_data.py`:

```python
def _base_signal_row(source: str, signal_id: str, payload: dict) -> dict:
    return {
        "source": source,
        "signal_id": signal_id,
        "status": payload.get("status"),
        "result": payload.get("result"),
        "symbol": payload.get("symbol", "XAUUSD"),
        "timeframe": payload.get("timeframe"),
        "strategy": payload.get("type") or payload.get("strategy"),
        "direction": payload.get("direction") or payload.get("direction_name"),
        "entry_price": payload.get("entry_price"),
        "sl_price": payload.get("sl_price"),
        "tp_price": payload.get("tp_price"),
        "accept_threshold": payload.get("accept_threshold"),
        "time": payload.get("time"),
        "created_at": payload.get("created_at") or payload.get("time_sent"),
        "latest_seen_at": payload.get("latest_seen_at"),
        "resolved_at": payload.get("resolved_at"),
    }


def flatten_sent_signals(sent_signals: dict) -> list[dict]:
    rows: list[dict] = []
    for signal_id, payload in (sent_signals or {}).items():
        if "probability_0.5" in payload or "probability_0.618" in payload:
            rows.append({
                **_base_signal_row("accepted", f"{signal_id}_0.5", payload),
                "leg": "0.5",
                "entry_price": payload.get("price_0.5"),
                "confidence": payload.get("probability_0.5"),
                "ticket_id": payload.get("ticket_a"),
                "outcome_recorded": payload.get("outcome_a_recorded") or payload.get("outcome_recorded"),
            })
            rows.append({
                **_base_signal_row("accepted", f"{signal_id}_0.618", payload),
                "leg": "0.618",
                "entry_price": payload.get("price_0.618"),
                "confidence": payload.get("probability_0.618"),
                "ticket_id": payload.get("ticket_b"),
                "outcome_recorded": payload.get("outcome_b_recorded") or payload.get("outcome_recorded"),
            })
        else:
            rows.append({
                **_base_signal_row("accepted", signal_id, payload),
                "leg": payload.get("leg", "single"),
                "entry_price": payload.get("price") or payload.get("entry_price"),
                "confidence": payload.get("probability"),
                "ticket_id": payload.get("ticket_id"),
                "outcome_recorded": payload.get("outcome_recorded"),
            })
    return rows


def flatten_shadow_signals(shadow_signals: dict) -> list[dict]:
    rows: list[dict] = []
    for signal_id, payload in (shadow_signals or {}).items():
        rows.append({
            **_base_signal_row("shadow", payload.get("signal_id", signal_id), payload),
            "leg": payload.get("leg"),
            "confidence": payload.get("confidence"),
            "ticket_id": payload.get("ticket_id"),
        })
    return rows
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: tests pass.

## Task 3: Dashboard Snapshot Metrics

**Files:**

- Modify: `src/dashboard_data.py`
- Modify: `tests/test_dashboard_data.py`

- [ ] **Step 1: Write failing test for snapshot metrics**

Append to `tests/test_dashboard_data.py`:

```python
from src.dashboard_data import build_snapshot_from_frames


def test_build_snapshot_from_frames_counts_learning_and_signals():
    accepted = [{"source": "accepted", "strategy": "FVG", "timeframe": "H1", "confidence": 0.62}]
    shadow = [{"source": "shadow", "strategy": "FVG", "timeframe": "M15", "confidence": 0.31, "result": "tp"}]
    real_df = pd.DataFrame({"label": [1, 0, 1], "pnl_relative": [2.0, -1.0, 1.5]})
    shadow_df = pd.DataFrame({"label": [1], "confidence": [0.31], "result": ["tp"]})
    learning = {"new_trades_since_last_train": 2, "last_train_time": "2026-06-08 11:51:21"}
    calibration = {"overall": {"sample_count": 4, "winrate_pct": 75.0}}

    snapshot = build_snapshot_from_frames(
        accepted_signals=accepted,
        shadow_signals=shadow,
        real_labeled=real_df,
        shadow_labeled=shadow_df,
        learning_status=learning,
        calibration_report=calibration,
        env_values={"ML_ACCEPT_THRESHOLD": "0.50", "ML_TRAINING_MAX_SETUPS": "5000"},
        model_inventory=[],
        warnings=[],
    )

    assert snapshot["counts"]["accepted_signals"] == 1
    assert snapshot["counts"]["shadow_signals"] == 1
    assert snapshot["counts"]["real_labeled_rows"] == 3
    assert snapshot["counts"]["shadow_labeled_rows"] == 1
    assert snapshot["learning"]["new_trades_since_last_train"] == 2
    assert snapshot["calibration"]["overall_winrate_pct"] == 75.0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: `build_snapshot_from_frames` missing.

- [ ] **Step 3: Implement snapshot builder**

Append to `src/dashboard_data.py`:

```python
def build_snapshot_from_frames(
    *,
    accepted_signals: list[dict],
    shadow_signals: list[dict],
    real_labeled: pd.DataFrame,
    shadow_labeled: pd.DataFrame,
    learning_status: dict,
    calibration_report: dict,
    env_values: dict,
    model_inventory: list[dict],
    warnings: list[str],
) -> dict:
    overall = (calibration_report or {}).get("overall", {})
    return {
        "counts": {
            "accepted_signals": len(accepted_signals),
            "shadow_signals": len(shadow_signals),
            "real_labeled_rows": int(len(real_labeled)),
            "shadow_labeled_rows": int(len(shadow_labeled)),
        },
        "learning": {
            "new_trades_since_last_train": learning_status.get("new_trades_since_last_train"),
            "last_train_time": learning_status.get("last_train_time"),
        },
        "calibration": {
            "sample_count": overall.get("sample_count"),
            "overall_winrate_pct": overall.get("winrate_pct"),
            "profit_factor": overall.get("profit_factor"),
            "max_consecutive_losses": overall.get("max_consecutive_losses"),
        },
        "env": env_values,
        "models": model_inventory,
        "warnings": warnings,
    }
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: tests pass.

## Task 4: Project Snapshot Loader

**Files:**

- Modify: `src/dashboard_data.py`
- Modify: `tests/test_dashboard_data.py`

- [ ] **Step 1: Write failing test for model freshness warning**

Append to `tests/test_dashboard_data.py`:

```python
from datetime import datetime, timedelta
from src.dashboard_data import build_model_freshness_warnings


def test_model_freshness_warning_when_data_newer_than_model():
    newer_data = datetime(2026, 6, 8, 12, 0, 0)
    older_model = datetime(2026, 6, 6, 5, 0, 0)

    warnings = build_model_freshness_warnings(
        latest_data_time=newer_data,
        model_inventory=[{"name": "smc_xgb_classifier.joblib", "modified_at": older_model}],
    )

    assert warnings == ["Model smc_xgb_classifier.joblib is older than latest labeled/shadow data."]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: `build_model_freshness_warnings` missing.

- [ ] **Step 3: Implement freshness helper and full project loader**

Append to `src/dashboard_data.py`:

```python
def read_env_values(path: str | Path = ".env") -> dict:
    resolved = resolve_project_path(path)
    keys = {
        "ML_ACCEPT_THRESHOLD",
        "ML_TRAINING_MAX_SETUPS",
        "ML_RETRAIN_THRESHOLD",
        "ML_RETRAIN_ON_WEEKEND",
        "MT5_EXECUTE_TRADES",
        "MT5_MAGIC_NUMBER",
    }
    values = {}
    if not resolved.exists():
        return values
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in keys:
            values[key] = value.strip().strip('"').strip("'")
    return values


def get_model_inventory(model_dir: str | Path = "models") -> list[dict]:
    resolved = resolve_project_path(model_dir)
    rows = []
    for name in ["smc_xgb_classifier.joblib", "smc_lgb_classifier.joblib"]:
        path = resolved / name
        if path.exists():
            rows.append({
                "name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "modified_at": pd.Timestamp(path.stat().st_mtime, unit="s").to_pydatetime(),
            })
    return rows


def latest_frame_time(*frames: pd.DataFrame):
    latest = None
    for frame in frames:
        if frame is None or frame.empty or "time" not in frame.columns:
            continue
        parsed = pd.to_datetime(frame["time"], errors="coerce").dropna()
        if parsed.empty:
            continue
        value = parsed.max().to_pydatetime()
        latest = value if latest is None or value > latest else latest
    return latest


def build_model_freshness_warnings(latest_data_time, model_inventory: list[dict]) -> list[str]:
    if latest_data_time is None:
        return []
    warnings = []
    for model in model_inventory:
        modified_at = model.get("modified_at")
        if modified_at is not None and modified_at < latest_data_time:
            warnings.append(f"Model {model.get('name')} is older than latest labeled/shadow data.")
    return warnings


def load_dashboard_snapshot(base_dir: str | Path = BASE_DIR) -> dict:
    global BASE_DIR
    previous_base = BASE_DIR
    BASE_DIR = Path(base_dir)
    try:
        sent = load_json_safe("data/sent_signals.json", default={}) or {}
        shadow = load_json_safe("data/shadow_signals.json", default={}) or {}
        real_labeled = load_csv_safe("data/labeled_setups.csv")
        shadow_labeled = load_csv_safe("data/shadow_labeled_setups.csv")
        learning = load_json_safe("data/learning_status.json", default={}) or {}
        calibration = load_json_safe("data/calibration_report.json", default={}) or {}
        accepted_rows = flatten_sent_signals(sent)
        shadow_rows = flatten_shadow_signals(shadow)
        models = get_model_inventory("models")
        latest_data = latest_frame_time(real_labeled, shadow_labeled)
        warnings = build_model_freshness_warnings(latest_data, models)
        return build_snapshot_from_frames(
            accepted_signals=accepted_rows,
            shadow_signals=shadow_rows,
            real_labeled=real_labeled,
            shadow_labeled=shadow_labeled,
            learning_status=learning,
            calibration_report=calibration,
            env_values=read_env_values(".env"),
            model_inventory=models,
            warnings=warnings,
        )
    finally:
        BASE_DIR = previous_base
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: tests pass.

## Task 5: Streamlit UI

**Files:**

- Create: `src/dashboard.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add Streamlit dependency**

Append this line to `requirements.txt`:

```text
streamlit>=1.35.0
```

- [ ] **Step 2: Create Streamlit dashboard**

Create `src/dashboard.py`:

```python
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard_data import (
    flatten_sent_signals,
    flatten_shadow_signals,
    load_dashboard_snapshot,
    load_json_safe,
)


st.set_page_config(page_title="Forex SMC Analyzer Dashboard", layout="wide")


@st.cache_data(ttl=15)
def cached_data():
    snapshot = load_dashboard_snapshot()
    accepted = pd.DataFrame(flatten_sent_signals(load_json_safe("data/sent_signals.json", default={}) or {}))
    shadow = pd.DataFrame(flatten_shadow_signals(load_json_safe("data/shadow_signals.json", default={}) or {}))
    calibration = load_json_safe("data/calibration_report.json", default={}) or {}
    return snapshot, accepted, shadow, calibration


snapshot, accepted_df, shadow_df, calibration = cached_data()

st.title("Forex SMC Analyzer")
st.caption("Read-only operational dashboard for signals, learning, calibration, and forward-test evidence.")

if snapshot["warnings"]:
    for warning in snapshot["warnings"]:
        st.warning(warning)

counts = snapshot["counts"]
env = snapshot["env"]
learning = snapshot["learning"]
cal = snapshot["calibration"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accepted Signals", counts["accepted_signals"])
col2.metric("Shadow Signals", counts["shadow_signals"])
col3.metric("Real Labeled", counts["real_labeled_rows"])
col4.metric("Shadow Labeled", counts["shadow_labeled_rows"])

tab_command, tab_signals, tab_trade, tab_learning, tab_calibration, tab_strategy, tab_backtest = st.tabs([
    "Command Center",
    "Live Signal Monitor",
    "Trade Manager",
    "AI Learning",
    "Confidence Calibration",
    "Strategy Performance",
    "Backtest vs Forward",
])

with tab_command:
    st.subheader("Bot Configuration")
    st.json(env)
    st.subheader("Learning Status")
    st.write(learning)
    st.subheader("Model Inventory")
    st.dataframe(pd.DataFrame(snapshot["models"]), use_container_width=True)

with tab_signals:
    st.subheader("Accepted Signals")
    st.dataframe(accepted_df, use_container_width=True)
    st.subheader("Shadow Signals")
    st.dataframe(shadow_df, use_container_width=True)

with tab_trade:
    st.subheader("Trade Manager Rules")
    st.write("M15/M30 emergency CHoCH close requires two closed opposite candles or H1/H4 confirmation.")
    st.write("H1/H4/D1 emergency CHoCH close can use one closed opposite candle.")
    st.write("The candle currently forming is ignored for emergency CHoCH decisions.")
    if not accepted_df.empty:
        cols = [c for c in ["signal_id", "timeframe", "strategy", "direction", "leg", "entry_price", "sl_price", "tp_price", "ticket_id", "outcome_recorded"] if c in accepted_df.columns]
        st.dataframe(accepted_df[cols], use_container_width=True)

with tab_learning:
    st.subheader("AI Learning")
    st.metric("New Trades Since Last Retrain", learning.get("new_trades_since_last_train"))
    st.metric("Last Retrain Time", learning.get("last_train_time"))
    st.metric("Training Max Setups", env.get("ML_TRAINING_MAX_SETUPS"))

with tab_calibration:
    st.subheader("Calibration Overall")
    st.write(cal)
    thresholds = calibration.get("thresholds", {}) if isinstance(calibration, dict) else {}
    if thresholds:
        st.subheader("Threshold Performance")
        st.dataframe(pd.DataFrame.from_dict(thresholds, orient="index"), use_container_width=True)

with tab_strategy:
    st.subheader("Strategy Breakdown")
    combined = pd.concat([accepted_df, shadow_df], ignore_index=True, sort=False)
    if not combined.empty and "strategy" in combined.columns:
        st.dataframe(combined.groupby(["source", "strategy", "timeframe"]).size().reset_index(name="signals"), use_container_width=True)
    else:
        st.info("No strategy data loaded.")

with tab_backtest:
    st.subheader("Backtest vs Forward Test")
    st.write("Phase 1 shows local data readiness. Backtest summary charts can be added after the dashboard data layer is stable.")
```

- [ ] **Step 3: Compile dashboard files**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m py_compile src\dashboard_data.py src\dashboard.py
```

Expected: exit code 0.

## Task 6: Verification And Launch

**Files:**

- Verify only.

- [ ] **Step 1: Run dashboard data tests**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run full suite**

Run:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Launch dashboard**

Run:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
.\.venv\Scripts\streamlit.exe run src\dashboard.py
```

Expected: Streamlit starts and prints a local URL, usually `http://localhost:8501`.

If `streamlit.exe` is missing, install dependencies first:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Network approval may be required for dependency install.

## Self-Review

Spec coverage:

- Command Center: Task 5.
- Live Signal Monitor: Tasks 2 and 5.
- Trade Manager rules: Task 5.
- AI Learning: Tasks 3 and 5.
- Confidence Calibration: Tasks 3 and 5.
- Strategy Performance: Task 5.
- Backtest vs Forward: Task 5 includes explicit Phase 1 readiness text; deeper backtest charts are deferred to Phase 2 by design.

No missing-detail markers are used as implementation instructions. Deferred Phase 2 scope is explicitly described as not part of Phase 1.
