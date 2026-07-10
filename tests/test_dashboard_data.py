import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.dashboard_data import (
    assign_confidence_bucket,
    build_confidence_bucket_summary,
    build_code_inventory,
    build_dashboard_process_notes,
    build_dashboard_health_checks,
    build_dashboard_readiness_report,
    build_feature_coverage_matrix,
    build_forward_readiness_gate,
    build_formula_qa_report,
    build_formula_verification_checklist,
    build_log_inventory,
    build_model_freshness_details,
    build_execution_decision_diagnostics,
    build_phase_document_inventory,
    build_pinescript_translation_matrix,
    build_report_export_payload,
    build_retraining_readiness,
    build_safe_command_previews,
    build_signal_detail,
    build_model_freshness_warnings,
    build_snapshot_from_frames,
    build_strategy_formula_inventory,
    build_verification_checklist,
    flatten_sent_signals,
    flatten_shadow_signals,
    load_dashboard_snapshot,
    load_csv_safe,
    load_json_safe,
    read_log_tail,
    summarize_forward_evidence,
)


def test_load_json_safe_returns_default_for_missing_file(tmp_path):
    result = load_json_safe(tmp_path / "missing.json", default={})
    assert result == {}


def test_load_csv_safe_returns_empty_frame_for_missing_file(tmp_path):
    result = load_csv_safe(tmp_path / "missing.csv")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_dashboard_uses_streamlit_fragment_instead_of_browser_reload():
    dashboard_source = (Path(__file__).resolve().parents[1] / "src" / "dashboard.py").read_text(encoding="utf-8")

    assert "@st.fragment" in dashboard_source
    assert "run_every=f\"{REFRESH_SECONDS}s\"" in dashboard_source
    assert "location.reload" not in dashboard_source
    assert "_install_auto_refresh" not in dashboard_source


def test_dashboard_emergency_exit_rules_match_same_timeframe_mitigation_policy():
    dashboard_source = (Path(__file__).resolve().parents[1] / "src" / "dashboard.py").read_text(encoding="utf-8")

    assert "M15/M30: close only after two closed opposite candles on the setup timeframe." in dashboard_source
    assert "Early mitigation is separate" in dashboard_source
    assert "or H1/H4 confirmation" not in dashboard_source


def test_load_json_safe_reads_existing_file(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert load_json_safe(path, default={}) == {"ok": True}


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


def test_flatten_sent_signals_reads_dual_leg_outcome_fields():
    rows = flatten_sent_signals({
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
            "status_a": "resolved",
            "result_a": "tp",
            "pnl_relative_a": 2.0,
            "net_profit_a": 100.0,
            "close_price_a": 2310.0,
            "close_reason_a": "DEAL_REASON_TP",
            "exit_category_a": "tp_profit",
            "outcome_a_recorded": True,
        }
    })

    midpoint = rows[0]
    golden_pocket = rows[1]
    assert midpoint["status"] == "resolved"
    assert midpoint["result"] == "tp"
    assert midpoint["pnl_relative"] == 2.0
    assert midpoint["net_profit"] == 100.0
    assert midpoint["close_price"] == 2310.0
    assert midpoint["close_reason"] == "DEAL_REASON_TP"
    assert midpoint["exit_category"] == "tp_profit"
    assert golden_pocket["result"] is None


def test_flatten_sent_signals_marks_low_confidence_registry_source():
    rows = flatten_sent_signals({
        "M15_Pivot_SINGLE_BEAR_4335.000_2026-06-08_11:30:00": {
            "time_sent": "2026-06-08 11:31:00",
            "timeframe": "M15",
            "direction": "BEAR",
            "type": "Pivot",
            "price": 4335.0,
            "probability": 0.08,
            "is_low_confidence": True,
        }
    })

    assert rows[0]["source"] == "shadow_registry"
    assert rows[0]["is_low_confidence"] is True
    assert rows[0]["strategy"] == "Pivot"


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


def test_model_freshness_warning_when_data_newer_than_model():
    newer_data = datetime(2026, 6, 8, 12, 0, 0)
    older_model = datetime(2026, 6, 6, 5, 0, 0)

    warnings = build_model_freshness_warnings(
        latest_data_time=newer_data,
        model_inventory=[{"name": "smc_xgb_classifier.joblib", "modified_at": older_model}],
    )

    assert warnings == ["Model smc_xgb_classifier.joblib is older than latest labeled/shadow data."]


def test_model_freshness_warning_suppressed_when_retrain_reviewed_latest_data():
    latest_data = datetime(2026, 6, 8, 12, 0, 0)
    older_model = datetime(2026, 6, 6, 5, 0, 0)

    warnings = build_model_freshness_warnings(
        latest_data_time=latest_data,
        model_inventory=[{"name": "smc_xgb_classifier.joblib", "modified_at": older_model}],
        learning_status={"last_train_time": "2026-06-08 12:30:00"},
    )

    assert warnings == []


def test_load_dashboard_snapshot_reads_project_files_without_global_path_mutation(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / ".env").write_text(
        "ML_ACCEPT_THRESHOLD=0.50\n"
        "ML_TRAINING_MAX_SETUPS=5000\n"
        "IGNORED_SECRET=secret\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "sent_signals.json").write_text(
        json.dumps({
            "sig-a": {
                "time_sent": "2026-06-08 10:00:00",
                "timeframe": "H1",
                "direction": "BULL",
                "type": "FVG",
                "price_0.5": 2300.0,
                "probability_0.5": 0.61,
            }
        }),
        encoding="utf-8",
    )
    (tmp_path / "data" / "shadow_signals.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "data" / "learning_status.json").write_text(
        json.dumps({"new_trades_since_last_train": 3, "last_train_time": "2026-06-08 12:30:00"}),
        encoding="utf-8",
    )
    (tmp_path / "data" / "calibration_report.json").write_text(
        json.dumps({"overall": {"sample_count": 2, "winrate_pct": 50.0}}),
        encoding="utf-8",
    )
    (tmp_path / "data" / "labeled_setups.csv").write_text(
        "time,label\n2026-06-08 12:00:00,1\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "shadow_labeled_setups.csv").write_text("time,label\n", encoding="utf-8")
    model_path = tmp_path / "models" / "smc_xgb_classifier.joblib"
    model_path.write_text("model", encoding="utf-8")
    old_timestamp = datetime(2026, 6, 6, 5, 0, 0).timestamp()
    os.utime(model_path, (old_timestamp, old_timestamp))

    snapshot = load_dashboard_snapshot(base_dir=tmp_path)

    assert snapshot["counts"]["accepted_signals"] == 2
    assert snapshot["counts"]["real_labeled_rows"] == 1
    assert snapshot["env"] == {"ML_ACCEPT_THRESHOLD": "0.50", "ML_TRAINING_MAX_SETUPS": "5000"}
    assert snapshot["learning"]["new_trades_since_last_train"] == 3
    assert snapshot["warnings"] == []
    assert snapshot["forward_summary"]["accepted"]["total"] == 2
    assert snapshot["forward_summary"]["shadow"]["total"] == 0


def test_dashboard_script_imports_when_launched_from_src_path():
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    command = (
        "import os, runpy, sys; "
        f"root={str(project_root)!r}; "
        f"src={str(src_dir)!r}; "
        "sys.path=[src] + [p for p in sys.path if p not in ('', root)]; "
        "os.chdir(root); "
        "runpy.run_path(os.path.join(src, 'dashboard.py'), run_name='__main__')"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=project_root.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


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


def test_assign_confidence_bucket_uses_stable_boundaries():
    assert assign_confidence_bucket(None) == "unknown"
    assert assign_confidence_bucket(0.29) == "0.00-0.30"
    assert assign_confidence_bucket(0.30) == "0.30-0.40"
    assert assign_confidence_bucket(0.50) == "0.50-0.60"
    assert assign_confidence_bucket(1.0) == "0.90-1.00"


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

    accepted = summary[(summary["source"] == "accepted") & (summary["confidence_bucket"] == "0.60-0.70")].iloc[0]
    assert accepted["signal_count"] == 1
    assert accepted["open_count"] == 1


def test_execution_diagnostics_explain_pivot_shadow_below_threshold():
    diagnostics = build_execution_decision_diagnostics([
        {
            "source": "shadow",
            "signal_id": "M15_Pivot_SINGLE_BEAR_4335.000_2026-06-08_11:30:00",
            "timeframe": "M15",
            "strategy": "Pivot",
            "direction": -1,
            "confidence": 0.084,
            "accept_threshold": 0.50,
            "ticket_id": None,
            "features": {"dist_entry_to_nearest_pivot": 0.0003},
        }
    ])

    row = diagnostics[0]
    assert row["decision"] == "shadow_monitoring"
    assert row["key_level_context"] == "pivot"
    assert "below accept threshold" in row["reason"]


def test_execution_diagnostics_flags_high_confidence_without_ticket():
    diagnostics = build_execution_decision_diagnostics([
        {
            "source": "accepted",
            "signal_id": "H1_Pivot_SINGLE_BULL_4320.000_2026-06-08_14:00:00",
            "timeframe": "H1",
            "strategy": "Pivot",
            "direction": 1,
            "confidence": 0.71,
            "accept_threshold": 0.50,
            "ticket_id": None,
        }
    ])

    assert diagnostics[0]["decision"] == "accepted_no_ticket"
    assert "passed threshold" in diagnostics[0]["reason"]


def test_dashboard_health_checks_reports_missing_files(tmp_path):
    checks = build_dashboard_health_checks(base_dir=tmp_path, now=datetime(2026, 6, 8, 12, 0, 0))

    missing = [check for check in checks if check["status"] == "missing"]

    assert any(check["name"] == "data/sent_signals.json" for check in missing)
    assert any(check["name"] == "models/smc_xgb_classifier.joblib" for check in missing)


def test_dashboard_health_checks_reports_stale_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    signal_path = data_dir / "sent_signals.json"
    signal_path.write_text("{}", encoding="utf-8")
    old_timestamp = datetime(2026, 6, 1, 12, 0, 0).timestamp()
    os.utime(signal_path, (old_timestamp, old_timestamp))

    checks = build_dashboard_health_checks(
        base_dir=tmp_path,
        now=datetime(2026, 6, 8, 12, 0, 0),
        stale_after_hours=24,
    )

    sent_check = next(check for check in checks if check["name"] == "data/sent_signals.json")
    assert sent_check["status"] == "stale"


def test_dashboard_health_checks_do_not_age_static_model_or_env_files(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()
    for relative_path in [
        "data/sent_signals.json",
        "data/shadow_signals.json",
        "data/labeled_setups.csv",
        "data/shadow_labeled_setups.csv",
        "data/calibration_report.json",
        "data/learning_status.json",
        "models/smc_xgb_classifier.joblib",
        "models/smc_lgb_classifier.joblib",
        ".env",
    ]:
        path = tmp_path / relative_path
        path.write_text("{}", encoding="utf-8")
        old_timestamp = datetime(2026, 6, 1, 12, 0, 0).timestamp()
        os.utime(path, (old_timestamp, old_timestamp))

    checks = build_dashboard_health_checks(
        base_dir=tmp_path,
        now=datetime(2026, 6, 8, 12, 0, 0),
        stale_after_hours=24,
    )

    by_name = {row["name"]: row for row in checks}
    assert by_name["models/smc_xgb_classifier.joblib"]["status"] == "ok"
    assert by_name["models/smc_lgb_classifier.joblib"]["status"] == "ok"
    assert by_name[".env"]["status"] == "ok"
    assert by_name["data/sent_signals.json"]["status"] == "stale"


def test_forward_evidence_summary_keeps_sources_separate():
    accepted = [{"source": "accepted", "result": "tp"}, {"source": "accepted", "result": "sl"}]
    shadow = [{"source": "shadow", "result": "tp"}, {"source": "shadow", "status": "open"}]

    summary = summarize_forward_evidence(accepted, shadow)

    assert summary["accepted"]["total"] == 2
    assert summary["accepted"]["tp"] == 1
    assert summary["accepted"]["winrate_pct"] == 50.0
    assert summary["shadow"]["total"] == 2
    assert summary["shadow"]["open"] == 1


def test_report_export_payload_is_json_safe_and_source_aware():
    payload = build_report_export_payload(
        snapshot={"counts": {"accepted_signals": 1}, "env": {"ML_ACCEPT_THRESHOLD": "0.50"}},
        accepted_signals=[{"signal_id": "a", "source": "accepted", "confidence": 0.62}],
        shadow_signals=[{"signal_id": "s", "source": "shadow", "confidence": 0.32}],
        confidence_summary=pd.DataFrame([{
            "source": "shadow",
            "confidence_bucket": "0.30-0.40",
            "signal_count": 1,
        }]),
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


def test_log_inventory_lists_data_logs_newest_first(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    old_log = data_dir / "old_stdout.log"
    new_log = data_dir / "new_stdout.log"
    old_log.write_text("old\n", encoding="utf-8")
    new_log.write_text("new\n", encoding="utf-8")
    old_time = datetime(2026, 6, 8, 10, 0, 0).timestamp()
    new_time = datetime(2026, 6, 8, 11, 0, 0).timestamp()
    os.utime(old_log, (old_time, old_time))
    os.utime(new_log, (new_time, new_time))

    logs = build_log_inventory(base_dir=tmp_path)

    assert [row["name"] for row in logs] == ["new_stdout.log", "old_stdout.log"]
    assert logs[0]["relative_path"] == "data/new_stdout.log"


def test_read_log_tail_rejects_paths_outside_project(tmp_path):
    outside = tmp_path.parent / "outside.log"
    outside.write_text("secret\n", encoding="utf-8")

    tail = read_log_tail(outside, base_dir=tmp_path)

    assert tail["status"] == "rejected"
    assert tail["lines"] == []


def test_safe_command_previews_are_copy_only_and_project_scoped(tmp_path):
    commands = build_safe_command_previews(base_dir=tmp_path)
    names = {command["name"] for command in commands}

    assert {
        "Start live scanner",
        "Start dashboard",
        "Run calibration report",
        "Run dashboard tests",
        "Preview retrain command",
    }.issubset(names)
    assert all(command["execution"] == "copy_only" for command in commands)
    assert all(str(tmp_path) in command["command"] for command in commands)


def test_dashboard_process_notes_are_read_only():
    notes = build_dashboard_process_notes()

    assert notes["mode"] == "read_only"
    assert "restart" in " ".join(notes["notes"]).lower()


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


def test_model_freshness_details_marks_reviewed_when_retrain_after_latest_data():
    latest_data = datetime(2026, 6, 8, 16, 0, 0)
    model_time = datetime(2026, 6, 8, 12, 0, 0)

    rows = build_model_freshness_details(
        latest_data_time=latest_data,
        model_inventory=[{"name": "smc_xgb_classifier.joblib", "modified_at": model_time}],
        learning_status={"last_train_time": "2026-06-08 16:30:00"},
        now=datetime(2026, 6, 8, 17, 0, 0),
    )

    assert rows[0]["status"] == "reviewed"
    assert "reviewed by the latest retrain" in rows[0]["detail"]


def test_retraining_readiness_cautions_when_model_stale_even_without_new_trade_count():
    snapshot = {
        "counts": {"real_labeled_rows": 10, "shadow_labeled_rows": 5},
        "learning": {"new_trades_since_last_train": 0},
        "env": {"ML_RETRAIN_THRESHOLD": "5"},
    }

    result = build_retraining_readiness(
        snapshot=snapshot,
        latest_data_time=datetime(2026, 6, 8, 16, 0, 0),
        model_inventory=[{
            "name": "smc_xgb_classifier.joblib",
            "modified_at": datetime(2026, 6, 8, 12, 0, 0),
        }],
    )

    assert result["status"] == "caution"
    assert result["new_trades_since_last_train"] == 0
    assert any("newer than model" in reason for reason in result["reasons"])


def test_retraining_readiness_explains_recent_retrain_with_retained_champion():
    snapshot = {
        "counts": {"real_labeled_rows": 10, "shadow_labeled_rows": 5},
        "learning": {
            "new_trades_since_last_train": 0,
            "last_train_time": "2026-06-08 22:02:36",
        },
        "env": {"ML_RETRAIN_THRESHOLD": "5"},
    }

    result = build_retraining_readiness(
        snapshot=snapshot,
        latest_data_time=datetime(2026, 6, 8, 21, 0, 0),
        model_inventory=[{
            "name": "smc_xgb_classifier.joblib",
            "modified_at": datetime(2026, 6, 6, 5, 0, 0),
        }],
    )

    assert result["status"] == "ready"
    assert any("champion" in reason.lower() for reason in result["reasons"])


def test_retraining_readiness_ready_when_threshold_is_reached():
    snapshot = {
        "counts": {"real_labeled_rows": 10, "shadow_labeled_rows": 5},
        "learning": {"new_trades_since_last_train": 6},
        "env": {"ML_RETRAIN_THRESHOLD": "5"},
    }

    result = build_retraining_readiness(
        snapshot=snapshot,
        latest_data_time=None,
        model_inventory=[],
    )

    assert result["status"] == "ready"
    assert result["manual_action"] == "Run model trainer, then regenerate calibration report."


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


def test_phase_document_inventory_reports_specs_and_plans(tmp_path):
    specs = tmp_path / "docs" / "superpowers" / "specs"
    plans = tmp_path / "docs" / "superpowers" / "plans"
    specs.mkdir(parents=True)
    plans.mkdir(parents=True)
    (specs / "2026-06-08-forex-smc-dashboard-phase-2-design.md").write_text("spec", encoding="utf-8")
    (plans / "2026-06-08-forex-smc-dashboard-phase-2.md").write_text("plan", encoding="utf-8")
    memory = tmp_path / "docs" / "DASHBOARD_PROJECT_MEMORY_2026-06-08.md"
    memory.write_text("## Phase 2 Implementation Completed", encoding="utf-8")

    rows = build_phase_document_inventory(base_dir=tmp_path, max_phase=2)
    phase_2 = next(row for row in rows if row["phase"] == "Phase 2")

    assert phase_2["status"] == "ready"
    assert phase_2["has_spec"] is True
    assert phase_2["has_plan"] is True
    assert phase_2["has_memory"] is True


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

    assert {
        "Dashboard data tests",
        "Compile all src/tests",
        "Full pytest suite",
        "Dashboard HTTP check",
    }.issubset(names)
    assert all(row["execution"] == "copy_only" for row in rows)
    assert all(str(tmp_path) in row["command"] or "localhost:8501" in row["command"] for row in rows)


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


def test_pinescript_translation_matrix_marks_missing_full_ports(tmp_path):
    pine_dir = tmp_path / "PineScripts"
    pine_dir.mkdir()
    (pine_dir / "Machine Learning RSI  AI Classification & Ranking (Zeiierman).txt").write_text("pine", encoding="utf-8")

    rows = build_pinescript_translation_matrix(base_dir=tmp_path)
    ml_rsi = next(row for row in rows if row["pinescript"] == "Machine Learning RSI AI Classification & Ranking")

    assert ml_rsi["status"] == "blocked"
    assert "Belum full port" in ml_rsi["detail"]


def test_formula_verification_checklist_is_copy_only(tmp_path):
    rows = build_formula_verification_checklist(base_dir=tmp_path)

    assert any("test_floop.py" in row["command"] for row in rows)
    assert all(row["execution"] == "copy_only" for row in rows)


def test_formula_qa_report_uses_worst_status():
    report = build_formula_qa_report(
        strategy_inventory=[{"formula_area": "FLoOP Pro", "status": "ready"}],
        pinescript_matrix=[{
            "pinescript": "Multi-Timeframe Volume Profiles",
            "status": "blocked",
            "detail": "Belum full port",
        }],
    )

    assert report["overall_status"] == "blocked"


def test_formula_qa_report_separates_core_formula_from_pinescript_parity():
    report = build_formula_qa_report(
        strategy_inventory=[
            {"formula_area": "FLoOP Pro", "status": "ready"},
            {"formula_area": "Rejection Entry", "status": "ready"},
        ],
        pinescript_matrix=[{
            "pinescript": "Multi-Timeframe Volume Profiles",
            "status": "blocked",
            "detail": "Belum full port",
        }],
    )

    assert report["core_status"] == "ready"
    assert report["pinescript_parity_status"] == "blocked"
    assert report["overall_status"] == "blocked"
    assert any(row["status"] == "blocked" for row in report["checks"])
