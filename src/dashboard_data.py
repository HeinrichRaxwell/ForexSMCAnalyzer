from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIDENCE_BUCKETS = [
    (0.00, 0.30, "0.00-0.30"),
    (0.30, 0.40, "0.30-0.40"),
    (0.40, 0.50, "0.40-0.50"),
    (0.50, 0.60, "0.50-0.60"),
    (0.60, 0.70, "0.60-0.70"),
    (0.70, 0.80, "0.70-0.80"),
    (0.80, 0.90, "0.80-0.90"),
    (0.90, 1.01, "0.90-1.00"),
]


def resolve_project_path(path: str | Path, base_dir: str | Path = BASE_DIR) -> Path:
    import os
    path = Path(path)
    if path.is_absolute():
        resolved = path
    else:
        resolved = Path(base_dir) / path
        
    filename = resolved.name
    if filename == "sent_signals.json":
        active_path = Path(base_dir) / "data" / "active_account.json"
        if active_path.exists():
            try:
                with open(active_path, "r") as f:
                    info = json.load(f)
                    login = info.get("login")
                    if login:
                        name, ext = os.path.splitext(filename)
                        segregated_path = resolved.parent / f"{name}_{login}{ext}"
                        if segregated_path.exists():
                            return segregated_path
            except Exception:
                pass
    return resolved


def load_json_safe(path: str | Path, default: Any = None, base_dir: str | Path = BASE_DIR) -> Any:
    resolved = resolve_project_path(path, base_dir)
    if not resolved.exists():
        return default
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_csv_safe(path: str | Path, base_dir: str | Path = BASE_DIR) -> pd.DataFrame:
    resolved = resolve_project_path(path, base_dir)
    if not resolved.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(resolved)
    except Exception:
        return pd.DataFrame()


def _base_signal_row(source: str, signal_id: str, payload: dict[str, Any]) -> dict[str, Any]:
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


def _leg_value(payload: dict[str, Any], field: str, suffix: str, default: Any = None) -> Any:
    key = f"{field}{suffix}"
    return payload.get(key, default)


def _leg_status(payload: dict[str, Any], suffix: str, outcome_recorded: Any) -> Any:
    status = _leg_value(payload, "status", suffix, payload.get("status"))
    if status:
        return status
    result = _leg_value(payload, "result", suffix, payload.get("result"))
    if outcome_recorded and result in (None, "", "open"):
        return "resolved_unclassified"
    return status


def _leg_outcome_fields(payload: dict[str, Any], suffix: str, outcome_recorded: Any) -> dict[str, Any]:
    return {
        "status": _leg_status(payload, suffix, outcome_recorded),
        "result": _leg_value(payload, "result", suffix, payload.get("result")),
        "pnl_relative": _leg_value(payload, "pnl_relative", suffix, payload.get("pnl_relative")),
        "net_profit": _leg_value(payload, "net_profit", suffix, payload.get("net_profit")),
        "close_price": _leg_value(payload, "close_price", suffix, payload.get("close_price")),
        "close_reason": _leg_value(payload, "close_reason", suffix, payload.get("close_reason")),
        "exit_category": _leg_value(payload, "exit_category", suffix, payload.get("exit_category")),
        "resolved_at": _leg_value(payload, "resolved_at", suffix, payload.get("resolved_at")),
    }


def flatten_sent_signals(sent_signals: dict[str, dict[str, Any]] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_id, payload in (sent_signals or {}).items():
        source = "shadow_registry" if payload.get("is_low_confidence") is True else "accepted"
        if "probability_0.5" in payload or "probability_0.618" in payload:
            midpoint_recorded = payload.get("outcome_a_recorded") or payload.get("outcome_recorded")
            golden_pocket_recorded = payload.get("outcome_b_recorded") or payload.get("outcome_recorded")
            midpoint = {
                **_base_signal_row(source, f"{signal_id}_0.5", payload),
                **_leg_outcome_fields(payload, "_a", midpoint_recorded),
                "leg": "0.5",
                "entry_price": payload.get("price_0.5"),
                "confidence": payload.get("probability_0.5"),
                "ticket_id": payload.get("ticket_a"),
                "outcome_recorded": midpoint_recorded,
                "is_low_confidence": payload.get("is_low_confidence", False),
            }
            golden_pocket = {
                **_base_signal_row(source, f"{signal_id}_0.618", payload),
                **_leg_outcome_fields(payload, "_b", golden_pocket_recorded),
                "leg": "0.618",
                "entry_price": payload.get("price_0.618"),
                "confidence": payload.get("probability_0.618"),
                "ticket_id": payload.get("ticket_b"),
                "outcome_recorded": golden_pocket_recorded,
                "is_low_confidence": payload.get("is_low_confidence", False),
            }
            if isinstance(payload.get("features_0.5"), dict):
                midpoint["features"] = payload.get("features_0.5")
            if isinstance(payload.get("features_0.618"), dict):
                golden_pocket["features"] = payload.get("features_0.618")
            rows.append(midpoint)
            rows.append(golden_pocket)
        else:
            outcome_recorded = payload.get("outcome_recorded")
            row = {
                **_base_signal_row(source, signal_id, payload),
                **_leg_outcome_fields(payload, "", outcome_recorded),
                "leg": payload.get("leg", "single"),
                "entry_price": payload.get("price") or payload.get("entry_price"),
                "confidence": payload.get("probability") or payload.get("confidence"),
                "ticket_id": payload.get("ticket_id"),
                "outcome_recorded": outcome_recorded,
                "is_low_confidence": payload.get("is_low_confidence", False),
            }
            if isinstance(payload.get("features"), dict):
                row["features"] = payload.get("features")
            rows.append(row)
    return rows


def flatten_shadow_signals(shadow_signals: dict[str, dict[str, Any]] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_id, payload in (shadow_signals or {}).items():
        row = {
            **_base_signal_row("shadow", payload.get("signal_id", signal_id), payload),
            "leg": payload.get("leg"),
            "confidence": payload.get("confidence"),
            "ticket_id": payload.get("ticket_id"),
        }
        if isinstance(payload.get("features"), dict):
            row["features"] = payload.get("features")
        rows.append(row)
    return rows


def build_signal_detail(
    signal_id: str,
    accepted_signals: list[dict[str, Any]] | None,
    shadow_signals: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    for row in [*(accepted_signals or []), *(shadow_signals or [])]:
        if row.get("signal_id") == signal_id:
            features = row.get("features") if isinstance(row.get("features"), dict) else {}
            signal = {key: value for key, value in row.items() if key != "features"}
            return {"signal": signal, "features": features}
    return None


def assign_confidence_bucket(confidence: Any) -> str:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "unknown"
    if pd.isna(value):
        return "unknown"
    value = max(0.0, min(1.0, value))
    for lower, upper, label in CONFIDENCE_BUCKETS:
        if lower <= value < upper:
            return label
    return "unknown"


def build_confidence_bucket_summary(signals: list[dict[str, Any]] | None) -> pd.DataFrame:
    rows = []
    for signal in signals or []:
        rows.append({
            "source": signal.get("source") or "unknown",
            "confidence_bucket": assign_confidence_bucket(signal.get("confidence")),
            "timeframe": signal.get("timeframe") or "unknown",
            "strategy": signal.get("strategy") or "unknown",
            "status": signal.get("status") or "open",
            "result": signal.get("result") or "open",
        })
    if not rows:
        return pd.DataFrame(columns=[
            "source",
            "confidence_bucket",
            "timeframe",
            "strategy",
            "signal_count",
            "tp_count",
            "sl_count",
            "expired_count",
            "open_count",
            "winrate_pct",
        ])

    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["source", "confidence_bucket", "timeframe", "strategy"], dropna=False)
    summary = grouped.agg(
        signal_count=("result", "count"),
        tp_count=("result", lambda values: int((values == "tp").sum())),
        sl_count=("result", lambda values: int((values == "sl").sum())),
        expired_count=("result", lambda values: int((values == "expired").sum())),
        open_count=("result", lambda values: int((values == "open").sum())),
    ).reset_index()
    resolved = summary["tp_count"] + summary["sl_count"]
    summary["winrate_pct"] = [
        round((tp / total) * 100, 2) if total else None
        for tp, total in zip(summary["tp_count"], resolved)
    ]
    return summary.sort_values(["source", "confidence_bucket", "timeframe", "strategy"]).reset_index(drop=True)


def _format_percent_value(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _infer_key_level_context(signal: dict[str, Any]) -> str:
    strategy = str(signal.get("strategy") or "").lower()
    features = signal.get("features") if isinstance(signal.get("features"), dict) else {}
    nearest_pivot = _read_float_value(features.get("dist_entry_to_nearest_pivot"))
    nearest_poc = _read_float_value(features.get("dist_entry_to_nearest_poc"))

    contexts: list[str] = []
    if "pivot" in strategy or (nearest_pivot is not None and nearest_pivot <= 0.002):
        contexts.append("pivot")
    if nearest_poc is not None and nearest_poc <= 0.002:
        contexts.append("volume_poc")
    if "swapzone" in strategy:
        contexts.append("support_resistance_flip")
    if "snd" in strategy or "supply" in strategy or "demand" in strategy:
        contexts.append("supply_demand")
    return ", ".join(contexts) if contexts else "none"


def build_execution_decision_diagnostics(
    signals: list[dict[str, Any]] | None,
    default_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal in signals or []:
        confidence = _read_float_value(signal.get("confidence"))
        threshold = _read_float_value(signal.get("accept_threshold"), default_threshold)
        source = signal.get("source") or "unknown"
        ticket_id = signal.get("ticket_id")
        is_shadow_source = source in {"shadow", "shadow_registry"} or signal.get("is_low_confidence") is True

        if confidence is None:
            decision = "registry_missing_confidence"
            reason = "No confidence value was stored for this registry row."
        elif is_shadow_source and confidence < threshold:
            decision = "shadow_monitoring"
            reason = (
                f"Confidence {_format_percent_value(confidence)} below accept threshold "
                f"{_format_percent_value(threshold)}; no MT5 order."
            )
        elif is_shadow_source and confidence >= threshold:
            decision = "needs_live_promotion_review"
            reason = "Shadow/low-confidence registry row now meets threshold; scanner should promote it on the next live pass if the setup is still active."
        elif ticket_id not in (None, "", 0):
            decision = "live_ticket"
            reason = f"Live ticket recorded: {ticket_id}."
        elif confidence >= threshold:
            decision = "accepted_no_ticket"
            reason = "Confidence passed threshold but no ticket is stored; check MT5_EXECUTE_TRADES, order logs, and execution skip reason."
        else:
            decision = "below_threshold_registry"
            reason = (
                f"Confidence {_format_percent_value(confidence)} below accept threshold "
                f"{_format_percent_value(threshold)}."
            )

        rows.append({
            "signal_id": signal.get("signal_id"),
            "source": source,
            "timeframe": signal.get("timeframe"),
            "strategy": signal.get("strategy"),
            "direction": signal.get("direction"),
            "confidence": confidence,
            "accept_threshold": threshold,
            "ticket_id": ticket_id,
            "status": signal.get("status"),
            "result": signal.get("result"),
            "key_level_context": _infer_key_level_context(signal),
            "decision": decision,
            "reason": reason,
        })
    return rows


def _file_health_row(path: Path, name: str, now: datetime, stale_after_hours: int | None) -> dict[str, Any]:
    if not path.exists():
        return {
            "name": name,
            "status": "missing",
            "size_bytes": 0,
            "modified_at": None,
            "age_hours": None,
            "detail": "File is missing.",
        }

    modified_at = datetime.fromtimestamp(path.stat().st_mtime)
    age_hours = round((now - modified_at).total_seconds() / 3600, 2)
    if stale_after_hours is None:
        status = "ok"
        detail = f"File is present. Last modified {age_hours} hours ago."
    else:
        status = "stale" if age_hours > stale_after_hours else "ok"
        detail = f"Last updated {age_hours} hours ago."
    return {
        "name": name,
        "status": status,
        "size_bytes": path.stat().st_size,
        "modified_at": modified_at,
        "age_hours": age_hours,
        "detail": detail,
    }


def build_dashboard_health_checks(
    base_dir: str | Path = BASE_DIR,
    now: datetime | None = None,
    stale_after_hours: int = 24,
) -> list[dict[str, Any]]:
    base = Path(base_dir)
    current_time = now or datetime.now()
    required_files = [
        ("data/sent_signals.json", stale_after_hours),
        ("data/shadow_signals.json", stale_after_hours),
        ("data/labeled_setups.csv", stale_after_hours),
        ("data/shadow_labeled_setups.csv", stale_after_hours),
        ("data/calibration_report.json", stale_after_hours),
        ("data/learning_status.json", stale_after_hours),
        ("models/smc_xgb_classifier.joblib", None),
        ("models/smc_lgb_classifier.joblib", None),
        (".env", None),
    ]
    return [
        _file_health_row(base / relative_path, relative_path, current_time, max_age_hours)
        for relative_path, max_age_hours in required_files
    ]


def _summarize_signal_source(signals: list[dict[str, Any]] | None) -> dict[str, Any]:
    rows = signals or []
    tp = sum(1 for row in rows if row.get("result") == "tp")
    sl = sum(1 for row in rows if row.get("result") in {"sl", "full_loss", "cut_loss_early"})
    expired = sum(1 for row in rows if row.get("result") == "expired")
    bep_profit = sum(1 for row in rows if row.get("result") == "bep_profit")
    protected_profit = sum(1 for row in rows if row.get("result") == "protected_profit")
    profit_not_tp = sum(1 for row in rows if row.get("result") == "profit_not_tp_verified")
    breakeven = sum(1 for row in rows if row.get("result") == "breakeven")
    open_count = sum(
        1
        for row in rows
        if (
            row.get("result") in (None, "", "open")
            and not str(row.get("status") or "").startswith("resolved")
        )
    )
    unclassified = sum(
        1
        for row in rows
        if (
            row.get("result") in (None, "", "open")
            and str(row.get("status") or "").startswith("resolved")
        )
    )
    resolved = tp + sl
    return {
        "total": len(rows),
        "tp": tp,
        "sl": sl,
        "expired": expired,
        "bep_profit": bep_profit,
        "protected_profit": protected_profit,
        "profit_not_tp_verified": profit_not_tp,
        "breakeven": breakeven,
        "resolved_unclassified": unclassified,
        "open": open_count,
        "winrate_pct": round((tp / resolved) * 100, 2) if resolved else None,
    }


def summarize_forward_evidence(
    accepted_signals: list[dict[str, Any]] | None,
    shadow_signals: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    return {
        "accepted": _summarize_signal_source(accepted_signals),
        "shadow": _summarize_signal_source(shadow_signals),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict("records")]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat(sep=" ")
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, pd.DataFrame)) else False:
        return None
    return value


def build_report_export_payload(
    *,
    snapshot: dict[str, Any],
    accepted_signals: list[dict[str, Any]],
    shadow_signals: list[dict[str, Any]],
    confidence_summary: pd.DataFrame,
    health_checks: list[dict[str, Any]],
    forward_summary: dict[str, Any],
    generated_at: datetime | None = None,
) -> dict[str, str]:
    report_time = generated_at or datetime.now()
    payload = {
        "generated_at": report_time,
        "snapshot": snapshot,
        "signals": {
            "accepted": accepted_signals,
            "shadow": shadow_signals,
        },
        "confidence_summary": confidence_summary,
        "health_checks": health_checks,
        "forward_summary": forward_summary,
        "non_claims": {
            "guaranteed_profit": False,
            "perfect_bot": False,
            "next_trade_profit": False,
        },
    }
    stamp = report_time.strftime("%Y-%m-%d_%H-%M-%S")
    return {
        "filename": f"forex_smc_dashboard_report_{stamp}.json",
        "json": json.dumps(_json_safe(payload), indent=2, sort_keys=True),
    }


def build_log_inventory(base_dir: str | Path = BASE_DIR) -> list[dict[str, Any]]:
    base = Path(base_dir)
    data_dir = base / "data"
    rows: list[dict[str, Any]] = []
    if not data_dir.exists():
        return rows
    for path in data_dir.glob("*.log"):
        stat = path.stat()
        rows.append({
            "name": path.name,
            "relative_path": str(path.relative_to(base)).replace("\\", "/"),
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 2),
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
        })
    return sorted(rows, key=lambda row: row["modified_at"], reverse=True)


def _is_allowed_log_path(path: Path, base_dir: Path) -> bool:
    try:
        data_dir = (base_dir / "data").resolve()
        resolved = path.resolve()
        resolved.relative_to(data_dir)
    except ValueError:
        return False
    return resolved.suffix.lower() == ".log"


def read_log_tail(
    path: str | Path,
    base_dir: str | Path = BASE_DIR,
    max_lines: int = 120,
) -> dict[str, Any]:
    base = Path(base_dir)
    resolved = resolve_project_path(path, base)
    if not _is_allowed_log_path(resolved, base):
        return {"status": "rejected", "path": str(path), "lines": [], "text": ""}
    if not resolved.exists():
        return {"status": "missing", "path": str(resolved), "lines": [], "text": ""}

    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-max(1, int(max_lines)):]
    return {
        "status": "ok",
        "path": str(resolved),
        "lines": tail,
        "text": "\n".join(tail),
    }


def build_safe_command_previews(base_dir: str | Path = BASE_DIR) -> list[dict[str, str]]:
    base = Path(base_dir)
    python_exe = base / ".venv" / "Scripts" / "python.exe"
    streamlit_exe = base / ".venv" / "Scripts" / "streamlit.exe"
    commands = [
        (
            "Start live scanner",
            "Start the scanner loop manually from PowerShell.",
            f"cd '{base}'; & '{python_exe}' -m src.main",
            "Manual scanner start only. Check demo account and execution flags first.",
        ),
        (
            "Start dashboard",
            "Start one fresh Streamlit dashboard server.",
            f"cd '{base}'; & '{streamlit_exe}' run src\\dashboard.py --server.port 8501 --server.headless true",
            "Dashboard is read-only; restart after dashboard helper/import changes.",
        ),
        (
            "Run calibration report",
            "Regenerate the calibration report after resolved outcomes are available.",
            f"cd '{base}'; & '{python_exe}' -m src.calibration_report",
            "Offline report generation; review output before changing thresholds.",
        ),
        (
            "Run dashboard tests",
            "Run the dashboard data regression tests.",
            f"cd '{base}'; & '{python_exe}' -m pytest tests\\test_dashboard_data.py -q",
            "Validation command only.",
        ),
        (
            "Preview retrain command",
            "Manual retrain command preview after forward evidence review.",
            f"cd '{base}'; & '{python_exe}' -m src.model_trainer",
            "Preview only in dashboard. Do not retrain blindly from UI.",
        ),
    ]
    return [
        {
            "name": name,
            "purpose": purpose,
            "command": command,
            "execution": "copy_only",
            "safety": safety,
        }
        for name, purpose, command, safety in commands
    ]


def build_dashboard_process_notes() -> dict[str, Any]:
    return {
        "mode": "read_only",
        "notes": [
            "Restart Streamlit after adding dashboard imports or helper functions.",
            "Stop only dashboard Streamlit processes to avoid killing scanner/backtest jobs.",
            "If an import error persists in the browser, start one fresh dashboard server on port 8501.",
        ],
    }


def build_snapshot_from_frames(
    *,
    accepted_signals: list[dict[str, Any]],
    shadow_signals: list[dict[str, Any]],
    real_labeled: pd.DataFrame,
    shadow_labeled: pd.DataFrame,
    learning_status: dict[str, Any],
    calibration_report: dict[str, Any],
    env_values: dict[str, Any],
    model_inventory: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    overall = (calibration_report or {}).get("overall", {})
    live_accepted = [row for row in accepted_signals if row.get("source") == "accepted"]
    forward_summary = summarize_forward_evidence(accepted_signals, shadow_signals)
    return {
        "counts": {
            "accepted_signals": len(live_accepted),
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
        "forward_summary": forward_summary,
        "warnings": warnings,
    }


def read_env_values(path: str | Path = ".env", base_dir: str | Path = BASE_DIR) -> dict[str, str]:
    resolved = resolve_project_path(path, base_dir)
    keys = {
        "ML_ACCEPT_THRESHOLD",
        "ML_TRAINING_MAX_SETUPS",
        "ML_RETRAIN_THRESHOLD",
        "ML_RETRAIN_ON_WEEKEND",
        "MT5_EXECUTE_TRADES",
        "MT5_MAGIC_NUMBER",
    }
    values: dict[str, str] = {}
    if not resolved.exists():
        return values

    for line in resolved.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in keys:
            values[key] = value.strip().strip('"').strip("'")
    return values


def get_model_inventory(model_dir: str | Path = "models", base_dir: str | Path = BASE_DIR) -> list[dict[str, Any]]:
    resolved = resolve_project_path(model_dir, base_dir)
    rows: list[dict[str, Any]] = []
    for name in ["smc_xgb_classifier.joblib", "smc_lgb_classifier.joblib"]:
        path = resolved / name
        if path.exists():
            stat = path.stat()
            rows.append({
                "name": name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime),
            })
    return rows


def latest_frame_time(*frames: pd.DataFrame) -> datetime | None:
    latest: datetime | None = None
    candidate_columns = ["time", "created_at", "latest_seen_at", "resolved_at"]
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for column in candidate_columns:
            if column not in frame.columns:
                continue
            parsed = pd.to_datetime(frame[column], errors="coerce").dropna()
            if parsed.empty:
                continue
            value = parsed.max().to_pydatetime()
            latest = value if latest is None or value > latest else latest
    return latest


def _last_train_datetime(learning_status: dict[str, Any] | None) -> datetime | None:
    if not isinstance(learning_status, dict):
        return None
    parsed = pd.to_datetime(learning_status.get("last_train_time"), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _latest_data_reviewed_by_retrain(
    latest_data_time: datetime | None,
    learning_status: dict[str, Any] | None,
) -> bool:
    last_train_time = _last_train_datetime(learning_status)
    return (
        latest_data_time is not None
        and last_train_time is not None
        and last_train_time >= latest_data_time
    )


def build_model_freshness_warnings(
    latest_data_time: datetime | None,
    model_inventory: list[dict[str, Any]],
    learning_status: dict[str, Any] | None = None,
) -> list[str]:
    if latest_data_time is None:
        return []
    if _latest_data_reviewed_by_retrain(latest_data_time, learning_status):
        return []

    warnings: list[str] = []
    for model in model_inventory:
        modified_at = model.get("modified_at")
        if modified_at is not None and modified_at < latest_data_time:
            warnings.append(f"Model {model.get('name')} is older than latest labeled/shadow data.")
    return warnings


def build_model_freshness_details(
    latest_data_time: datetime | None,
    model_inventory: list[dict[str, Any]],
    learning_status: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    current_time = now or datetime.now()
    reviewed_by_retrain = _latest_data_reviewed_by_retrain(latest_data_time, learning_status)
    rows: list[dict[str, Any]] = []
    for model in model_inventory or []:
        modified_at = model.get("modified_at")
        is_stale = (
            latest_data_time is not None
            and modified_at is not None
            and modified_at < latest_data_time
            and not reviewed_by_retrain
        )
        is_reviewed = (
            latest_data_time is not None
            and modified_at is not None
            and modified_at < latest_data_time
            and reviewed_by_retrain
        )
        data_lag_hours = None
        if is_stale or is_reviewed:
            data_lag_hours = round((latest_data_time - modified_at).total_seconds() / 3600, 2)
        model_age_hours = None
        if modified_at is not None:
            model_age_hours = round((current_time - modified_at).total_seconds() / 3600, 2)
        if is_stale:
            status = "stale"
            detail = "Latest labeled/shadow data is newer than this model."
        elif is_reviewed:
            status = "reviewed"
            detail = "Latest labeled/shadow data was reviewed by the latest retrain; champion model may have been retained."
        else:
            status = "fresh"
            detail = "Model is not older than latest labeled/shadow data."
        rows.append({
            "name": model.get("name"),
            "status": status,
            "modified_at": modified_at,
            "latest_data_time": latest_data_time,
            "data_lag_hours": data_lag_hours,
            "model_age_hours": model_age_hours,
            "detail": detail,
        })
    return rows


def _read_int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _any_model_stale(
    latest_data_time: datetime | None,
    model_inventory: list[dict[str, Any]],
    learning_status: dict[str, Any] | None = None,
) -> bool:
    if latest_data_time is None:
        return False
    if _latest_data_reviewed_by_retrain(latest_data_time, learning_status):
        return False
    return any(
        model.get("modified_at") is not None and model.get("modified_at") < latest_data_time
        for model in model_inventory or []
    )


def build_retraining_readiness(
    *,
    snapshot: dict[str, Any],
    latest_data_time: datetime | None,
    model_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = snapshot.get("counts", {})
    learning = snapshot.get("learning", {})
    env = snapshot.get("env", {})
    real_rows = _read_int_value(counts.get("real_labeled_rows"), 0)
    shadow_rows = _read_int_value(counts.get("shadow_labeled_rows"), 0)
    total_labeled_rows = real_rows + shadow_rows
    new_trades = _read_int_value(learning.get("new_trades_since_last_train"), 0)
    threshold = _read_int_value(env.get("ML_RETRAIN_THRESHOLD"), 5)
    stale_model = _any_model_stale(latest_data_time, model_inventory, learning)
    last_train_time = pd.to_datetime(learning.get("last_train_time"), errors="coerce")
    model_times = [
        model.get("modified_at")
        for model in model_inventory or []
        if model.get("modified_at") is not None
    ]
    latest_model_time = max(model_times) if model_times else None
    reviewed_latest_data = _latest_data_reviewed_by_retrain(latest_data_time, learning)
    retrain_after_model = (
        latest_model_time is not None
        and not pd.isna(last_train_time)
        and last_train_time.to_pydatetime() > latest_model_time
    )

    reasons: list[str] = []
    if total_labeled_rows <= 0:
        reasons.append("No labeled or shadow-labeled rows are available for training.")
        status = "blocked"
        manual_action = "Collect and resolve labeled/shadow outcomes before retraining."
    elif new_trades >= threshold:
        reasons.append(f"New resolved trades since last train reached {new_trades}/{threshold}.")
        status = "ready"
        manual_action = "Run model trainer, then regenerate calibration report."
    elif stale_model:
        if retrain_after_model:
            reasons.append("Latest labeled/shadow data is newer than model files, but last retrain ran after the model timestamp; champion validation may have retained the older model.")
            manual_action = "Review trainer output. Keeping the champion is safer than overwriting it with a weaker challenger."
        else:
            reasons.append("Latest labeled/shadow data is newer than model files.")
            manual_action = "Review new outcomes; retrain manually if the data quality is acceptable."
        status = "caution"
    elif reviewed_latest_data:
        reasons.append("Latest labeled/shadow data has already been reviewed by the latest retrain; champion validation may have retained the active model.")
        status = "ready"
        manual_action = "No retrain needed right now."
    else:
        reasons.append(f"Retrain threshold not reached yet: {new_trades}/{threshold}.")
        status = "ready"
        manual_action = "No retrain needed right now."

    return {
        "status": status,
        "reasons": reasons,
        "manual_action": manual_action,
        "new_trades_since_last_train": new_trades,
        "retrain_threshold": threshold,
        "total_labeled_rows": total_labeled_rows,
        "model_stale": stale_model,
    }


def _read_float_value(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolved_count(summary: dict[str, Any] | None) -> int:
    row = summary or {}
    return (
        _read_int_value(row.get("tp"), 0)
        + _read_int_value(row.get("sl"), 0)
        + _read_int_value(row.get("bep_profit"), 0)
        + _read_int_value(row.get("protected_profit"), 0)
        + _read_int_value(row.get("profit_not_tp_verified"), 0)
        + _read_int_value(row.get("breakeven"), 0)
    )


def build_forward_readiness_gate(
    *,
    calibration_report: dict[str, Any],
    forward_summary: dict[str, dict[str, Any]],
    minimum_resolved: int = 20,
) -> dict[str, Any]:
    overall = calibration_report.get("overall", {}) if isinstance(calibration_report, dict) else {}
    recommendation = calibration_report.get("recommendation", {}) if isinstance(calibration_report, dict) else {}
    accepted_resolved = _resolved_count((forward_summary or {}).get("accepted"))
    shadow_resolved = _resolved_count((forward_summary or {}).get("shadow"))
    resolved_forward_trades = accepted_resolved + shadow_resolved
    profit_factor = _read_float_value(overall.get("profit_factor"))
    max_consecutive_losses = _read_int_value(overall.get("max_consecutive_losses"), 0)

    reasons: list[str] = []
    status = "ready"
    manual_action = "Continue demo forward-test monitoring and review before real-account promotion."
    if resolved_forward_trades < minimum_resolved:
        status = "blocked"
        reasons.append(f"Resolved forward evidence is low: {resolved_forward_trades}/{minimum_resolved}.")
        manual_action = "Keep scanner running on demo until more TP/SL outcomes are resolved."
    else:
        if profit_factor is None or profit_factor < 1.10:
            status = "caution"
            reasons.append(f"Calibration profit factor needs review: {profit_factor}.")
        if max_consecutive_losses > 10:
            status = "caution"
            reasons.append(f"Max consecutive losses is high: {max_consecutive_losses}.")
        if not reasons:
            reasons.append("Forward evidence count and calibration risk checks passed.")

    return {
        "status": status,
        "reasons": reasons,
        "manual_action": manual_action,
        "resolved_forward_trades": resolved_forward_trades,
        "minimum_resolved": minimum_resolved,
        "accepted_resolved": accepted_resolved,
        "shadow_resolved": shadow_resolved,
        "profit_factor": profit_factor,
        "max_consecutive_losses": max_consecutive_losses,
        "recommended_threshold": recommendation.get("threshold"),
        "recommended_expectancy_r": recommendation.get("expectancy_r"),
    }


def _worst_status(statuses: list[str]) -> str:
    rank = {"ready": 0, "caution": 1, "blocked": 2}
    return max(statuses or ["ready"], key=lambda status: rank.get(status, 0))


def build_dashboard_readiness_report(
    *,
    snapshot: dict[str, Any],
    latest_data_time: datetime | None,
    model_inventory: list[dict[str, Any]],
    calibration_report: dict[str, Any],
    forward_summary: dict[str, dict[str, Any]],
    health_checks: list[dict[str, Any]],
    minimum_forward_resolved: int = 20,
) -> dict[str, Any]:
    missing_health = [row for row in health_checks or [] if row.get("status") == "missing"]
    stale_health = [row for row in health_checks or [] if row.get("status") == "stale"]
    if missing_health:
        health_status = "blocked"
        health_reasons = [f"Missing required file: {row.get('name')}" for row in missing_health]
    elif stale_health:
        health_status = "caution"
        health_reasons = [f"Stale file: {row.get('name')}" for row in stale_health]
    else:
        health_status = "ready"
        health_reasons = ["Required dashboard files are present."]

    model_details = build_model_freshness_details(
        latest_data_time,
        model_inventory,
        learning_status=snapshot.get("learning", {}),
    )
    stale_models = [row for row in model_details if row.get("status") == "stale"]
    reviewed_models = [row for row in model_details if row.get("status") == "reviewed"]
    model_status = "caution" if stale_models else "ready"
    model_reasons = (
        [f"{row.get('name')} is older than latest labeled/shadow data." for row in stale_models]
        if stale_models
        else (
            [f"{row.get('name')} was reviewed by the latest retrain; champion may be retained." for row in reviewed_models]
            if reviewed_models
            else ["Model files are not older than latest labeled/shadow data."]
        )
    )

    retraining = build_retraining_readiness(
        snapshot=snapshot,
        latest_data_time=latest_data_time,
        model_inventory=model_inventory,
    )
    forward_gate = build_forward_readiness_gate(
        calibration_report=calibration_report,
        forward_summary=forward_summary,
        minimum_resolved=minimum_forward_resolved,
    )
    checks = [
        {
            "component": "Health Checks",
            "status": health_status,
            "reasons": " | ".join(health_reasons),
            "manual_action": "Fix missing/stale data files before relying on the dashboard." if health_status != "ready" else "No manual action needed.",
        },
        {
            "component": "Model Freshness",
            "status": model_status,
            "reasons": " | ".join(model_reasons),
            "manual_action": "Retrain manually after reviewing new labeled/shadow outcomes." if model_status == "caution" else "No manual action needed.",
        },
        {
            "component": "Retraining",
            "status": retraining["status"],
            "reasons": " | ".join(retraining["reasons"]),
            "manual_action": retraining["manual_action"],
        },
        {
            "component": "Forward Evidence",
            "status": forward_gate["status"],
            "reasons": " | ".join(forward_gate["reasons"]),
            "manual_action": forward_gate["manual_action"],
        },
    ]
    return {
        "overall_status": _worst_status([row["status"] for row in checks]),
        "checks": checks,
        "model_freshness": model_details,
        "retraining": retraining,
        "forward_gate": forward_gate,
        "non_claims": {
            "guaranteed_profit": False,
            "perfect_bot": False,
            "next_trade_profit": False,
        },
    }


def build_phase_document_inventory(
    base_dir: str | Path = BASE_DIR,
    max_phase: int = 7,
) -> list[dict[str, Any]]:
    base = Path(base_dir)
    specs_dir = base / "docs" / "superpowers" / "specs"
    plans_dir = base / "docs" / "superpowers" / "plans"
    memory_path = base / "docs" / "DASHBOARD_PROJECT_MEMORY_2026-06-08.md"
    memory_content = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
    rows: list[dict[str, Any]] = []
    for phase_number in range(1, max_phase + 1):
        if phase_number == 1:
            spec_pattern = "*forex-smc-dashboard-design.md"
            plan_pattern = "*forex-smc-dashboard.md"
        else:
            spec_pattern = f"*forex-smc-dashboard-phase-{phase_number}-design.md"
            plan_pattern = f"*forex-smc-dashboard-phase-{phase_number}.md"
        has_spec = specs_dir.exists() and any(specs_dir.glob(spec_pattern))
        has_plan = plans_dir.exists() and any(plans_dir.glob(plan_pattern))
        has_memory = f"Phase {phase_number} Implementation Completed" in memory_content
        status = "ready" if has_spec and has_plan and has_memory else "blocked"
        missing = []
        if not has_spec:
            missing.append("spec")
        if not has_plan:
            missing.append("plan")
        if not has_memory:
            missing.append("memory")
        rows.append({
            "phase": f"Phase {phase_number}",
            "status": status,
            "has_spec": has_spec,
            "has_plan": has_plan,
            "has_memory": has_memory,
            "detail": "Documentation evidence present." if status == "ready" else f"Missing: {', '.join(missing)}",
        })
    return rows


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob(pattern) if item.is_file())


def build_code_inventory(base_dir: str | Path = BASE_DIR) -> dict[str, int]:
    base = Path(base_dir)
    src_dir = base / "src"
    tests_dir = base / "tests"
    docs_dir = base / "docs"
    data_dir = base / "data"
    dashboard_files = sum(
        1
        for name in ["dashboard.py", "dashboard_data.py"]
        if (src_dir / name).exists()
    )
    return {
        "source_files": _count_files(src_dir, "*.py"),
        "test_files": _count_files(tests_dir, "test_*.py"),
        "dashboard_files": dashboard_files,
        "docs_files": _count_files(docs_dir, "*.md"),
        "data_files": sum(
            _count_files(data_dir, pattern)
            for pattern in ["*.json", "*.csv", "*.log", "*.pid"]
        ),
    }


def build_verification_checklist(base_dir: str | Path = BASE_DIR) -> list[dict[str, str]]:
    base = Path(base_dir)
    python_exe = base / ".venv" / "Scripts" / "python.exe"
    rows = [
        (
            "Dashboard data tests",
            "Run dashboard helper and import regression tests.",
            f"cd '{base}'; & '{python_exe}' -m pytest tests\\test_dashboard_data.py -q",
        ),
        (
            "Compile all src/tests",
            "Compile every Python source and test file.",
            f"cd '{base}'; & '{python_exe}' -m compileall -q src tests",
        ),
        (
            "Full pytest suite",
            "Run the full repository test suite.",
            f"cd '{base}'; & '{python_exe}' -m pytest -q",
        ),
        (
            "Dashboard HTTP check",
            "Verify the local Streamlit dashboard responds on port 8501.",
            "Invoke-WebRequest -Uri 'http://localhost:8501' -UseBasicParsing -TimeoutSec 10",
        ),
        (
            "Dashboard import check",
            "Verify Phase 5 dashboard helpers import from Python.",
            f"cd '{base}'; & '{python_exe}' -c \"from src.dashboard_data import build_feature_coverage_matrix; print('imports ok')\"",
        ),
    ]
    return [
        {
            "name": name,
            "purpose": purpose,
            "command": command,
            "execution": "copy_only",
        }
        for name, purpose, command in rows
    ]


def build_feature_coverage_matrix(
    *,
    phase_inventory: list[dict[str, Any]],
    code_inventory: dict[str, int],
    readiness_report: dict[str, Any],
    health_checks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    phase_status = _worst_status([row.get("status", "ready") for row in phase_inventory or []])
    missing_phase_details = [
        f"{row.get('phase')}: {row.get('detail')}"
        for row in phase_inventory or []
        if row.get("status") != "ready"
    ]
    test_files = _read_int_value(code_inventory.get("test_files"), 0)
    source_files = _read_int_value(code_inventory.get("source_files"), 0)
    dashboard_files = _read_int_value(code_inventory.get("dashboard_files"), 0)
    code_status = "ready" if source_files > 0 and test_files > 0 and dashboard_files >= 2 else "blocked"
    health_status = _worst_status([row.get("status", "ready") for row in health_checks or []])
    health_issues = [
        f"{row.get('name')}: {row.get('status')}"
        for row in health_checks or []
        if row.get("status") != "ok"
    ]
    readiness_status = readiness_report.get("overall_status", "ready")
    return [
        {
            "feature_area": "Dashboard Phase Documentation",
            "status": phase_status,
            "evidence": " | ".join(missing_phase_details) if missing_phase_details else f"{len(phase_inventory or [])} dashboard phases have documentation evidence.",
            "manual_action": "Fill missing spec/plan/memory docs." if phase_status != "ready" else "No manual action needed.",
        },
        {
            "feature_area": "Code/Test Inventory",
            "status": code_status,
            "evidence": f"{source_files} source files, {test_files} test files, {dashboard_files} dashboard files.",
            "manual_action": "Add missing tests or dashboard files." if code_status != "ready" else "Run verification checklist after edits.",
        },
        {
            "feature_area": "Runtime Readiness",
            "status": readiness_status,
            "evidence": f"Readiness report overall status: {readiness_status}.",
            "manual_action": "Review readiness gate reasons." if readiness_status != "ready" else "Continue monitoring.",
        },
        {
            "feature_area": "Data/Model Health",
            "status": health_status,
            "evidence": " | ".join(health_issues) if health_issues else f"{len(health_checks or [])} health checks are ok.",
            "manual_action": "Fix missing/stale data or model files." if health_status != "ready" else "No manual action needed.",
        },
    ]


def _relative_existing_files(base: Path, relative_paths: list[str]) -> list[str]:
    return [
        relative_path
        for relative_path in relative_paths
        if (base / relative_path).exists()
    ]


def build_strategy_formula_inventory(base_dir: str | Path = BASE_DIR) -> list[dict[str, Any]]:
    base = Path(base_dir)
    formula_specs = [
        {
            "formula_area": "SMC Fibonacci Structures",
            "source_candidates": ["src/smc_detector.py"],
            "test_candidates": [
                "tests/test_fibo_detector.py",
                "tests/test_imbalances.py",
                "tests/test_breakers_swapzones.py",
            ],
            "detail": "FVG, OB, BPR, IC, SND, swapzone Fibonacci levels and structure detectors.",
        },
        {
            "formula_area": "Rejection Logic",
            "source_candidates": ["src/rejection_detector.py"],
            "test_candidates": ["tests/test_rejection.py"],
            "detail": "Pinbar, engulfing, and double-touch rejection detection near entry/key levels.",
        },
        {
            "formula_area": "Pivot Classic/Rejection",
            "source_candidates": ["src/indicators/pivots.py"],
            "test_candidates": ["tests/test_pivots.py"],
            "detail": "Daily classic pivots and pivot rejection setups.",
        },
        {
            "formula_area": "FLoOP Pro",
            "source_candidates": ["src/indicators/floop.py"],
            "test_candidates": ["tests/test_floop.py"],
            "detail": "FLoOP ATR/range filter, MTF score, ADX/CHOP/cooldown gates.",
        },
        {
            "formula_area": "KNN SuperTrend",
            "source_candidates": ["src/indicators/knn_classifier.py"],
            "test_candidates": ["tests/test_knn_classifier.py"],
            "detail": "AI SuperTrend KNN probability engine.",
        },
        {
            "formula_area": "Volume Clusters",
            "source_candidates": ["src/indicators/volume_clusters.py"],
            "test_candidates": ["tests/test_volume_clusters.py"],
            "detail": "K-Means volume clusters and POC features.",
        },
        {
            "formula_area": "Scanner Entry Decisions",
            "source_candidates": ["src/scanner_worker.py", "src/main.py"],
            "test_candidates": [
                "tests/test_scanner_entry_decisions.py",
                "tests/test_scanner_market_orders.py",
                "tests/test_scanner_shadow_signals.py",
            ],
            "detail": "Signal filtering, market/pending order decisions, shadow tracking, and HTF/rejection logic.",
        },
        {
            "formula_area": "Active Trade Management",
            "source_candidates": ["src/execution.py"],
            "test_candidates": ["tests/test_active_trade_management.py"],
            "detail": "Closed-candle CHoCH/emergency exit behavior and trade management.",
        },
        {
            "formula_area": "Telegram Signal Formatting",
            "source_candidates": ["src/telegram_bot.py", "src/scanner_worker.py"],
            "test_candidates": ["tests/test_telegram.py", "tests/test_telegram_signal_messages.py"],
            "detail": "Professional signal text/photo message formatting.",
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in formula_specs:
        source_files = _relative_existing_files(base, spec["source_candidates"])
        test_files = _relative_existing_files(base, spec["test_candidates"])
        status = "ready" if source_files and test_files else "blocked"
        missing = []
        if not source_files:
            missing.append("source")
        if not test_files:
            missing.append("test")
        rows.append({
            "formula_area": spec["formula_area"],
            "status": status,
            "source_files": ", ".join(source_files),
            "test_files": ", ".join(test_files),
            "detail": spec["detail"] if status == "ready" else f"Missing {', '.join(missing)} evidence.",
        })
    return rows


def build_pinescript_translation_matrix(base_dir: str | Path = BASE_DIR) -> list[dict[str, Any]]:
    base = Path(base_dir)
    pine_dir = base / "PineScripts"
    rows = [
        {
            "pinescript": "FLoOP Pro",
            "file_pattern": "Floop PRO*.txt",
            "python_mapping": "src/indicators/floop.py; src/indicators/pivots.py",
            "status": "ready",
            "detail": "Core FLoOP formula, classic pivots, filters, and cooldown are represented in Python.",
        },
        {
            "pinescript": "AI SuperTrend KNN Machine Learning",
            "file_pattern": "AI-SuperTrend*.txt",
            "python_mapping": "src/indicators/knn_classifier.py",
            "status": "caution",
            "detail": "Core probability engine exists, but full TradingView visual/signal-state behavior is not fully ported.",
        },
        {
            "pinescript": "Clusters Volume Profile LuxAlgo",
            "file_pattern": "Clusters Volume Profile*.txt",
            "python_mapping": "src/indicators/volume_clusters.py",
            "status": "caution",
            "detail": "Core cluster/POC numeric features exist, but full TradingView visual profile objects are not ported.",
        },
        {
            "pinescript": "Machine Learning RSI AI Classification & Ranking",
            "file_pattern": "Machine Learning RSI*Ranking*.txt",
            "python_mapping": "",
            "status": "blocked",
            "detail": "Belum full port. Python does not replicate this ML RSI script as a dedicated module.",
        },
        {
            "pinescript": "Multi-Timeframe Volume Profiles",
            "file_pattern": "Multi-Timeframe Volume Profiles*.txt",
            "python_mapping": "",
            "status": "blocked",
            "detail": "Belum full port. Python does not yet provide full MTF VAH/VAL/POC value-area behavior.",
        },
    ]
    for row in rows:
        row["pinescript_file_present"] = pine_dir.exists() and any(pine_dir.glob(row["file_pattern"]))
        mapped_files = [item.strip() for item in row["python_mapping"].split(";") if item.strip()]
        row["python_files_present"] = bool(mapped_files) and all((base / item).exists() for item in mapped_files)
    return rows


def build_formula_verification_checklist(base_dir: str | Path = BASE_DIR) -> list[dict[str, str]]:
    base = Path(base_dir)
    python_exe = base / ".venv" / "Scripts" / "python.exe"
    rows = [
        (
            "Formula targeted suite",
            "Run targeted strategy/formula tests.",
            "tests\\test_fibo_detector.py tests\\test_floop.py tests\\test_pivots.py tests\\test_rejection.py tests\\test_imbalances.py tests\\test_breakers_swapzones.py tests\\test_knn_classifier.py tests\\test_volume_clusters.py tests\\test_scanner_entry_decisions.py tests\\test_scanner_market_orders.py tests\\test_active_trade_management.py",
        ),
        (
            "FLoOP formula tests",
            "Run FLoOP Pro indicator parity and component tests.",
            "tests\\test_floop.py",
        ),
        (
            "Pivot and rejection tests",
            "Run pivot classic/rejection and LTF rejection formula tests.",
            "tests\\test_pivots.py tests\\test_rejection.py",
        ),
        (
            "SMC/Fibonacci tests",
            "Run SMC structure, fibo, imbalance, breaker/swapzone tests.",
            "tests\\test_fibo_detector.py tests\\test_imbalances.py tests\\test_breakers_swapzones.py",
        ),
        (
            "Scanner decision tests",
            "Run scanner entry/market/shadow decision tests.",
            "tests\\test_scanner_entry_decisions.py tests\\test_scanner_market_orders.py tests\\test_scanner_shadow_signals.py",
        ),
    ]
    return [
        {
            "name": name,
            "purpose": purpose,
            "command": f"cd '{base}'; & '{python_exe}' -m pytest {test_args} -q",
            "execution": "copy_only",
        }
        for name, purpose, test_args in rows
    ]


def build_formula_qa_report(
    *,
    strategy_inventory: list[dict[str, Any]],
    pinescript_matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    for row in strategy_inventory or []:
        checks.append({
            "component": row.get("formula_area", "unknown"),
            "status": row.get("status", "blocked"),
            "evidence": row.get("detail", ""),
            "manual_action": "Run targeted formula tests after edits." if row.get("status") == "ready" else "Add missing formula source/test evidence.",
        })
    for row in pinescript_matrix or []:
        checks.append({
            "component": row.get("pinescript", "unknown"),
            "status": row.get("status", "blocked"),
            "evidence": row.get("detail", ""),
            "manual_action": "Full PineScript parity needs explicit implementation and bar-by-bar comparison." if row.get("status") != "ready" else "No manual action needed unless TradingView parity changes.",
        })
    core_status = _worst_status([row.get("status", "blocked") for row in strategy_inventory or []])
    pinescript_parity_status = _worst_status([row.get("status", "blocked") for row in pinescript_matrix or []])
    return {
        "overall_status": _worst_status([row["status"] for row in checks]),
        "core_status": core_status,
        "pinescript_parity_status": pinescript_parity_status,
        "checks": checks,
        "non_claims": {
            "guaranteed_profit": False,
            "perfect_formula_parity": False,
            "next_trade_profit": False,
        },
    }


def load_dashboard_snapshot(base_dir: str | Path = BASE_DIR) -> dict[str, Any]:
    sent = load_json_safe("data/sent_signals.json", default={}, base_dir=base_dir) or {}
    shadow = load_json_safe("data/shadow_signals.json", default={}, base_dir=base_dir) or {}
    real_labeled = load_csv_safe("data/labeled_setups.csv", base_dir=base_dir)
    shadow_labeled = load_csv_safe("data/shadow_labeled_setups.csv", base_dir=base_dir)
    learning = load_json_safe("data/learning_status.json", default={}, base_dir=base_dir) or {}
    calibration = load_json_safe("data/calibration_report.json", default={}, base_dir=base_dir) or {}
    accepted_rows = flatten_sent_signals(sent)
    shadow_rows = flatten_shadow_signals(shadow)
    models = get_model_inventory("models", base_dir=base_dir)
    latest_data = latest_frame_time(real_labeled, shadow_labeled)
    warnings = build_model_freshness_warnings(latest_data, models, learning_status=learning)

    return build_snapshot_from_frames(
        accepted_signals=accepted_rows,
        shadow_signals=shadow_rows,
        real_labeled=real_labeled,
        shadow_labeled=shadow_labeled,
        learning_status=learning,
        calibration_report=calibration,
        env_values=read_env_values(".env", base_dir=base_dir),
        model_inventory=models,
        warnings=warnings,
    )
