from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_data import (
    BASE_DIR,
    build_code_inventory,
    build_confidence_bucket_summary,
    build_dashboard_health_checks,
    build_dashboard_process_notes,
    build_dashboard_readiness_report,
    build_execution_decision_diagnostics,
    build_feature_coverage_matrix,
    build_formula_qa_report,
    build_formula_verification_checklist,
    build_log_inventory,
    build_model_freshness_details,
    build_phase_document_inventory,
    build_pinescript_translation_matrix,
    build_report_export_payload,
    build_safe_command_previews,
    build_signal_detail,
    build_strategy_formula_inventory,
    build_verification_checklist,
    flatten_sent_signals,
    flatten_shadow_signals,
    load_csv_safe,
    load_dashboard_snapshot,
    load_json_safe,
    latest_frame_time,
    read_log_tail,
    summarize_forward_evidence,
)


REFRESH_SECONDS = 15
SIGNAL_COLUMNS = [
    "source",
    "signal_id",
    "status",
    "result",
    "symbol",
    "timeframe",
    "strategy",
    "direction",
    "leg",
    "exit_category",
    "entry_price",
    "sl_price",
    "tp_price",
    "close_price",
    "close_reason",
    "confidence",
    "pnl_relative",
    "net_profit",
    "accept_threshold",
    "ticket_id",
    "outcome_recorded",
    "is_low_confidence",
    "time",
    "created_at",
    "latest_seen_at",
    "resolved_at",
]


st.set_page_config(page_title="Forex SMC Analyzer Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #d7dde5;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: #ffffff;
    }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    .small-note { color: #586574; font-size: 0.88rem; }
    .rule-line {
        border-left: 4px solid #2f7d68;
        padding: 0.55rem 0.8rem;
        margin: 0.4rem 0;
        background: #f6f8fa;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _display(value: Any, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in str(value))
    return safe.strip("_") or "signal"


def _records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    for column in SIGNAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame["event_time"] = pd.NaT
    for column in ["created_at", "time", "latest_seen_at", "resolved_at"]:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        frame["event_time"] = frame["event_time"].fillna(parsed)
    return frame.sort_values("event_time", ascending=False, na_position="last")


def _safe_options(frame: pd.DataFrame, column: str) -> list[Any]:
    if frame.empty or column not in frame.columns:
        return []
    values = frame[column].dropna().unique().tolist()
    return sorted(values, key=lambda item: str(item))


def _table_frame(frame: pd.DataFrame) -> pd.DataFrame:
    view = frame.copy()
    for column in view.select_dtypes(include=["object"]).columns:
        view[column] = view[column].map(lambda value: "" if value is None or pd.isna(value) else str(value))
    return view


def _combined_signal_frame(accepted: pd.DataFrame, shadow: pd.DataFrame) -> pd.DataFrame:
    signal_frames = [frame.dropna(axis=1, how="all") for frame in [accepted, shadow] if not frame.empty]
    combined = pd.concat(signal_frames, ignore_index=True, sort=False) if signal_frames else pd.DataFrame()
    return combined.reindex(columns=[*SIGNAL_COLUMNS, "event_time"])


def _filter_default(key: str, options: list[Any]) -> list[Any]:
    if key not in st.session_state:
        return options
    current = st.session_state.get(key)
    if not isinstance(current, list):
        return options
    return [value for value in current if value in options]


def render_filter_sidebar(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.header("Filters")
        st.info("No signals loaded.")
        return

    st.header("Filters")
    sources = _safe_options(frame, "source")
    timeframes = _safe_options(frame, "timeframe")
    strategies = _safe_options(frame, "strategy")
    statuses = _safe_options(frame, "status")
    results = _safe_options(frame, "result")

    st.multiselect("Source", sources, default=_filter_default("filter_sources", sources), key="filter_sources")
    st.multiselect("Timeframe", timeframes, default=_filter_default("filter_timeframes", timeframes), key="filter_timeframes")
    st.multiselect("Strategy", strategies, default=_filter_default("filter_strategies", strategies), key="filter_strategies")
    st.multiselect("Status", statuses, default=_filter_default("filter_statuses", statuses), key="filter_statuses")
    st.multiselect("Result", results, default=_filter_default("filter_results", results), key="filter_results")
    st.slider(
        "Confidence",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
        key="filter_confidence",
    )


def _filter_signals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    selected_sources = st.session_state.get("filter_sources", [])
    selected_timeframes = st.session_state.get("filter_timeframes", [])
    selected_strategies = st.session_state.get("filter_strategies", [])
    selected_statuses = st.session_state.get("filter_statuses", [])
    selected_results = st.session_state.get("filter_results", [])
    min_confidence, max_confidence = st.session_state.get("filter_confidence", (0.0, 1.0))

    filtered = frame.copy()
    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]
    if selected_timeframes:
        filtered = filtered[filtered["timeframe"].isin(selected_timeframes)]
    if selected_strategies:
        filtered = filtered[filtered["strategy"].isin(selected_strategies)]
    if selected_statuses and "status" in filtered.columns:
        filtered = filtered[filtered["status"].isna() | filtered["status"].isin(selected_statuses)]
    if selected_results and "result" in filtered.columns:
        filtered = filtered[filtered["result"].isna() | filtered["result"].isin(selected_results)]
    return filtered[
        filtered["confidence"].isna()
        | ((filtered["confidence"] >= min_confidence) & (filtered["confidence"] <= max_confidence))
    ]


def _calibration_frame(calibration: dict[str, Any], key: str) -> pd.DataFrame:
    section = calibration.get(key, {}) if isinstance(calibration, dict) else {}
    if not isinstance(section, dict) or not section:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(section, orient="index")
    frame.index.name = key[:-1] if key.endswith("s") else key
    return frame.reset_index()


def _aggregate_labeled(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty or "label" not in frame.columns:
        return pd.DataFrame()
    available = [column for column in group_cols if column in frame.columns]
    if not available:
        return pd.DataFrame()
    work = frame.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    grouped = work.groupby(available, dropna=False).agg(
        sample_count=("label", "count"),
        winrate_pct=("label", lambda values: round(float(values.mean()) * 100, 2)),
    )
    if "pnl_relative" in work.columns:
        pnl_grouped = work.groupby(available, dropna=False)["pnl_relative"].mean().round(3)
        grouped["avg_pnl_relative"] = pnl_grouped
    return grouped.reset_index().sort_values("sample_count", ascending=False)


def _backtest_inventory() -> pd.DataFrame:
    data_dir = BASE_DIR / "data"
    rows = []
    for pattern in ["backtest_simulation_results.csv", "real_tick_backtest_*.csv"]:
        for path in data_dir.glob(pattern):
            rows.append({
                "file": path.name,
                "size_kb": round(path.stat().st_size / 1024, 2),
                "modified_at": pd.Timestamp(path.stat().st_mtime, unit="s"),
            })
    return pd.DataFrame(rows).sort_values("modified_at", ascending=False) if rows else pd.DataFrame()


@st.cache_data(ttl=REFRESH_SECONDS)
def cached_data():
    snapshot = load_dashboard_snapshot()
    accepted_records = flatten_sent_signals(load_json_safe("data/sent_signals.json", default={}) or {})
    shadow_records = flatten_shadow_signals(load_json_safe("data/shadow_signals.json", default={}) or {})
    accepted = _records_frame(accepted_records)
    shadow = _records_frame(shadow_records)
    labeled = load_csv_safe("data/labeled_setups.csv")
    shadow_labeled = load_csv_safe("data/shadow_labeled_setups.csv")
    calibration = load_json_safe("data/calibration_report.json", default={}) or {}
    backtest = load_csv_safe("data/backtest_simulation_results.csv")
    real_tick_files = _backtest_inventory()
    bucket_summary = build_confidence_bucket_summary([*accepted_records, *shadow_records])
    try:
        default_threshold = float(snapshot.get("env", {}).get("ML_ACCEPT_THRESHOLD", 0.50))
    except (TypeError, ValueError):
        default_threshold = 0.50
    execution_diagnostics = build_execution_decision_diagnostics(
        [*accepted_records, *shadow_records],
        default_threshold=default_threshold,
    )
    health_checks = build_dashboard_health_checks()
    forward_summary = snapshot.get("forward_summary") or summarize_forward_evidence(accepted_records, shadow_records)
    log_inventory = build_log_inventory()
    command_previews = build_safe_command_previews()
    process_notes = build_dashboard_process_notes()
    latest_data_time = latest_frame_time(labeled, shadow_labeled)
    model_freshness_details = build_model_freshness_details(
        latest_data_time,
        snapshot["models"],
        learning_status=snapshot.get("learning", {}),
    )
    readiness_report = build_dashboard_readiness_report(
        snapshot=snapshot,
        latest_data_time=latest_data_time,
        model_inventory=snapshot["models"],
        calibration_report=calibration,
        forward_summary=forward_summary,
        health_checks=health_checks,
        minimum_forward_resolved=20,
    )
    phase_inventory = build_phase_document_inventory()
    code_inventory = build_code_inventory()
    verification_checklist = build_verification_checklist()
    feature_coverage = build_feature_coverage_matrix(
        phase_inventory=phase_inventory,
        code_inventory=code_inventory,
        readiness_report=readiness_report,
        health_checks=health_checks,
    )
    strategy_formula_inventory = build_strategy_formula_inventory()
    pinescript_translation_matrix = build_pinescript_translation_matrix()
    formula_verification_checklist = build_formula_verification_checklist()
    formula_qa_report = build_formula_qa_report(
        strategy_inventory=strategy_formula_inventory,
        pinescript_matrix=pinescript_translation_matrix,
    )
    return (
        snapshot,
        accepted,
        shadow,
        labeled,
        shadow_labeled,
        calibration,
        backtest,
        real_tick_files,
        bucket_summary,
        execution_diagnostics,
        health_checks,
        forward_summary,
        log_inventory,
        command_previews,
        process_notes,
        latest_data_time,
        model_freshness_details,
        readiness_report,
        phase_inventory,
        code_inventory,
        verification_checklist,
        feature_coverage,
        strategy_formula_inventory,
        pinescript_translation_matrix,
        formula_verification_checklist,
        formula_qa_report,
    )


def render_metric_row(snapshot: dict[str, Any], calibration: dict[str, Any]) -> None:
    counts = snapshot["counts"]
    learning = snapshot["learning"]
    env = snapshot["env"]
    threshold = env.get("ML_ACCEPT_THRESHOLD", "0.50")
    threshold_report = calibration.get("thresholds", {}).get(str(threshold), {}) if isinstance(calibration, dict) else {}

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accepted", counts["accepted_signals"])
    col2.metric("Shadow", counts["shadow_signals"])
    col3.metric("Training Rows", counts["real_labeled_rows"] + counts["shadow_labeled_rows"])
    col4.metric("Tuned Threshold WR", _display(threshold_report.get("winrate_pct"), "%"))
    col5.metric("Since Retrain", _display(learning.get("new_trades_since_last_train")))


def render_signal_table(frame: pd.DataFrame, height: int = 440) -> None:
    columns = [column for column in SIGNAL_COLUMNS if column in frame.columns]
    st.dataframe(_table_frame(frame[columns]), width="stretch", height=height)


st.title("Forex SMC Analyzer")
st.caption(f"Local read-only dashboard. Live soft-refresh: {REFRESH_SECONDS}s, no browser reload.")

@st.fragment(run_every=f"{REFRESH_SECONDS}s")
def render_dashboard_filters():
    (
        _snapshot,
        accepted_df,
        shadow_df,
        *_unused_dashboard_data,
    ) = cached_data()
    combined_df = _combined_signal_frame(accepted_df, shadow_df)
    render_filter_sidebar(combined_df)


@st.fragment(run_every=f"{REFRESH_SECONDS}s")
def render_dashboard_body():
    (
        snapshot,
        accepted_df,
        shadow_df,
        labeled_df,
        shadow_labeled_df,
        calibration_report,
        backtest_df,
        backtest_files,
        bucket_summary_df,
        execution_diagnostics,
        health_checks,
        forward_summary,
        log_inventory,
        command_previews,
        process_notes,
        latest_data_time,
        model_freshness_details,
        readiness_report,
        phase_inventory,
        code_inventory,
        verification_checklist,
        feature_coverage,
        strategy_formula_inventory,
        pinescript_translation_matrix,
        formula_verification_checklist,
        formula_qa_report,
    ) = cached_data()
    signal_frames = [frame.dropna(axis=1, how="all") for frame in [accepted_df, shadow_df] if not frame.empty]
    combined_df = pd.concat(signal_frames, ignore_index=True, sort=False) if signal_frames else pd.DataFrame()
    combined_df = combined_df.reindex(columns=[*SIGNAL_COLUMNS, "event_time"])
    filtered_df = _filter_signals(combined_df)
    
    
    for warning in snapshot["warnings"]:
        st.warning(warning)
    
    render_metric_row(snapshot, calibration_report)
    
    tabs = st.tabs([
        "Command Center",
        "Live Signal Monitor",
        "Trade Manager",
        "AI Learning",
        "Confidence Calibration",
        "Strategy Performance",
        "Backtest vs Forward",
    ])
    
    with tabs[0]:
        st.subheader("System QA")
        qa_cols = st.columns(4)
        qa_cols[0].metric("Source Files", _display(code_inventory.get("source_files")))
        qa_cols[1].metric("Test Files", _display(code_inventory.get("test_files")))
        qa_cols[2].metric("Dashboard Files", _display(code_inventory.get("dashboard_files")))
        qa_cols[3].metric("Data Files", _display(code_inventory.get("data_files")))
        feature_frame = pd.DataFrame(feature_coverage)
        if not feature_frame.empty:
            st.dataframe(feature_frame, width="stretch", hide_index=True)
    
        st.subheader("Phase Documentation")
        phase_frame = pd.DataFrame(phase_inventory)
        if not phase_frame.empty:
            st.dataframe(phase_frame, width="stretch", hide_index=True)
    
        st.subheader("Verification Checklist")
        verification_frame = pd.DataFrame(verification_checklist)
        if not verification_frame.empty:
            st.dataframe(verification_frame, width="stretch", hide_index=True)
            selected_verification = st.selectbox(
                "Verification Command",
                verification_frame["name"].tolist(),
                key="verification_command_selector",
            )
            verification_command = verification_frame[verification_frame["name"] == selected_verification].iloc[0]
            st.code(verification_command["command"], language="powershell")
    
        st.subheader("Readiness Gate")
        readiness_cols = st.columns(3)
        readiness_cols[0].metric("Overall Status", str(readiness_report.get("overall_status", "unknown")).upper())
        readiness_cols[1].metric("Latest Training Data", _display(latest_data_time))
        readiness_cols[2].metric("Resolved Forward", _display(readiness_report.get("forward_gate", {}).get("resolved_forward_trades")))
        readiness_frame = pd.DataFrame(readiness_report.get("checks", []))
        if not readiness_frame.empty:
            st.dataframe(readiness_frame, width="stretch", hide_index=True)
    
        freshness_frame = pd.DataFrame(model_freshness_details)
        if not freshness_frame.empty:
            st.subheader("Model Freshness Detail")
            st.dataframe(freshness_frame, width="stretch", hide_index=True)
    
        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Configuration")
            env_frame = pd.DataFrame(
                [{"key": key, "value": value} for key, value in snapshot["env"].items()]
            )
            st.dataframe(env_frame, width="stretch", hide_index=True)
    
            st.subheader("Learning State")
            learning_cols = st.columns(3)
            learning_cols[0].metric("New Since Retrain", _display(snapshot["learning"].get("new_trades_since_last_train")))
            learning_cols[1].metric("Last Retrain", _display(snapshot["learning"].get("last_train_time")))
            learning_cols[2].metric("Max Setups", _display(snapshot["env"].get("ML_TRAINING_MAX_SETUPS")))
    
        with right:
            st.subheader("Model Inventory")
            model_frame = pd.DataFrame(snapshot["models"])
            st.dataframe(model_frame, width="stretch", hide_index=True)
            st.subheader("Data Counts")
            st.dataframe(pd.DataFrame([snapshot["counts"]]), width="stretch", hide_index=True)
    
        st.subheader("Health Checks")
        health_frame = pd.DataFrame(health_checks)
        if not health_frame.empty:
            st.dataframe(health_frame, width="stretch", hide_index=True)
        else:
            st.info("No health check rows loaded.")
    
        st.subheader("Operations")
        if st.button("Refresh Dashboard Data", key="refresh_dashboard_data"):
            st.cache_data.clear()
            st.rerun()
    
        command_frame = pd.DataFrame(command_previews)
        if not command_frame.empty:
            st.dataframe(command_frame, width="stretch", hide_index=True)
            selected_command_name = st.selectbox(
                "Command Preview",
                command_frame["name"].tolist(),
                key="command_preview_selector",
            )
            selected_command = command_frame[command_frame["name"] == selected_command_name].iloc[0]
            st.code(selected_command["command"], language="powershell")
    
        for note in process_notes.get("notes", []):
            st.markdown(f'<div class="rule-line">{note}</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.subheader("Signals")
        if filtered_df.empty:
            st.info("No signals match the active filters.")
        else:
            signal_export_columns = [column for column in SIGNAL_COLUMNS if column in filtered_df.columns]
            st.download_button(
                "Download Filtered Signals CSV",
                data=_table_frame(filtered_df[signal_export_columns]).to_csv(index=False).encode("utf-8"),
                file_name="forex_smc_filtered_signals.csv",
                mime="text/csv",
                key="download_filtered_signals_csv",
            )
    
            chart_cols = st.columns([1, 1])
            with chart_cols[0]:
                by_source = filtered_df.groupby("source").size().reset_index(name="signals")
                st.bar_chart(by_source, x="source", y="signals")
            with chart_cols[1]:
                bucketed = filtered_df.dropna(subset=["confidence"]).copy()
                if not bucketed.empty:
                    bucketed["confidence_bucket"] = pd.cut(
                        bucketed["confidence"],
                        bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                        include_lowest=True,
                    ).astype(str)
                    st.bar_chart(bucketed.groupby("confidence_bucket").size().reset_index(name="signals"), x="confidence_bucket", y="signals")
            render_signal_table(filtered_df)
    
            st.subheader("Execution Decision Diagnostics")
            diagnostics_df = pd.DataFrame(execution_diagnostics)
            if diagnostics_df.empty:
                st.info("No execution diagnostic rows loaded.")
            else:
                diagnostic_cols = [
                    "source",
                    "signal_id",
                    "timeframe",
                    "strategy",
                    "direction",
                    "confidence",
                    "accept_threshold",
                    "ticket_id",
                    "key_level_context",
                    "decision",
                    "reason",
                ]
                st.dataframe(
                    _table_frame(diagnostics_df[[col for col in diagnostic_cols if col in diagnostics_df.columns]]),
                    width="stretch",
                    height=320,
                    hide_index=True,
                )
                key_level_diagnostics = diagnostics_df[
                    diagnostics_df["key_level_context"].fillna("none").astype(str) != "none"
                ]
                if not key_level_diagnostics.empty:
                    st.subheader("Key Level / Pivot Diagnostics")
                    st.dataframe(
                        _table_frame(key_level_diagnostics[[col for col in diagnostic_cols if col in key_level_diagnostics.columns]]),
                        width="stretch",
                        height=260,
                        hide_index=True,
                    )
    
            st.subheader("Signal Detail")
            signal_ids = filtered_df["signal_id"].dropna().astype(str).tolist()
            selected_signal_id = st.selectbox("Signal", signal_ids, index=0) if signal_ids else None
            if selected_signal_id:
                detail = build_signal_detail(
                    selected_signal_id,
                    accepted_df.to_dict("records"),
                    shadow_df.to_dict("records"),
                )
                if detail:
                    detail_cols = st.columns([1.1, 1])
                    with detail_cols[0]:
                        signal_frame = pd.DataFrame([
                            {"field": key, "value": value}
                            for key, value in detail["signal"].items()
                            if key != "event_time"
                        ])
                        st.dataframe(_table_frame(signal_frame), width="stretch", hide_index=True, height=360)
                    with detail_cols[1]:
                        st.json(detail["features"] or {})
                        st.download_button(
                            "Download Signal Detail JSON",
                            data=json.dumps(detail, indent=2, default=str).encode("utf-8"),
                            file_name=f"forex_smc_signal_{_safe_filename(selected_signal_id)}.json",
                            mime="application/json",
                            key=f"download_signal_detail_{_safe_filename(selected_signal_id)}",
                        )
    
    with tabs[2]:
        st.subheader("Active Trade View")
        active = accepted_df.copy()
        if not active.empty and "outcome_recorded" in active.columns:
            active = active[(active["ticket_id"].notna()) & (active["outcome_recorded"] != True)]
        render_signal_table(active if not active.empty else accepted_df.head(50), height=360)
    
        st.subheader("Emergency Exit Rules")
        st.markdown('<div class="rule-line">M15/M30: close only after two closed opposite candles or H1/H4 confirmation.</div>', unsafe_allow_html=True)
        st.markdown('<div class="rule-line">H1/H4/D1: one closed opposite candle can trigger emergency CHoCH exit.</div>', unsafe_allow_html=True)
        st.markdown('<div class="rule-line">The candle currently forming is ignored for emergency CHoCH decisions.</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        st.subheader("Learning Pipeline")
        cols = st.columns(4)
        cols[0].metric("Real Labeled", snapshot["counts"]["real_labeled_rows"])
        cols[1].metric("Shadow Labeled", snapshot["counts"]["shadow_labeled_rows"])
        cols[2].metric("New Since Retrain", _display(snapshot["learning"].get("new_trades_since_last_train")))
        cols[3].metric("Training Window", _display(snapshot["env"].get("ML_TRAINING_MAX_SETUPS")))
    
        st.subheader("Retraining Readiness")
        retraining_readiness = readiness_report.get("retraining", {})
        retrain_cols = st.columns(3)
        retrain_cols[0].metric("Status", str(retraining_readiness.get("status", "unknown")).upper())
        retrain_cols[1].metric("New Trades", _display(retraining_readiness.get("new_trades_since_last_train")))
        retrain_cols[2].metric("Threshold", _display(retraining_readiness.get("retrain_threshold")))
        st.dataframe(pd.DataFrame([{
            "reasons": " | ".join(retraining_readiness.get("reasons", [])),
            "manual_action": retraining_readiness.get("manual_action"),
            "total_labeled_rows": retraining_readiness.get("total_labeled_rows"),
            "model_stale": retraining_readiness.get("model_stale"),
        }]), width="stretch", hide_index=True)
    
        if not shadow_labeled_df.empty:
            shadow_view = shadow_labeled_df.copy()
            if "confidence" in shadow_view.columns:
                shadow_view["confidence_bucket"] = pd.cut(
                    pd.to_numeric(shadow_view["confidence"], errors="coerce"),
                    bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                    include_lowest=True,
                ).astype(str)
                learned = _aggregate_labeled(shadow_view, ["confidence_bucket", "timeframe"])
                st.dataframe(learned, width="stretch", hide_index=True)
        else:
            st.info("No shadow labeled rows loaded.")
    
        st.subheader("Signal Confidence Buckets")
        if bucket_summary_df.empty:
            st.info("No confidence bucket summary loaded.")
        else:
            st.dataframe(bucket_summary_df, width="stretch", hide_index=True)
            st.download_button(
                "Download Confidence Buckets CSV",
                data=bucket_summary_df.to_csv(index=False).encode("utf-8"),
                file_name="forex_smc_confidence_buckets.csv",
                mime="text/csv",
                key="download_confidence_buckets_csv",
            )
            bucket_totals = bucket_summary_df.groupby(["source", "confidence_bucket"], dropna=False)["signal_count"].sum().reset_index()
            st.bar_chart(bucket_totals, x="confidence_bucket", y="signal_count", color="source")
    
    with tabs[4]:
        st.subheader("Overall Calibration")
        st.dataframe(pd.DataFrame([calibration_report.get("overall", {})]), width="stretch", hide_index=True)
    
        thresholds = _calibration_frame(calibration_report, "thresholds")
        buckets = _calibration_frame(calibration_report, "buckets")
        if not thresholds.empty:
            st.subheader("Thresholds")
            st.dataframe(thresholds, width="stretch", hide_index=True)
            if "winrate_pct" in thresholds.columns:
                st.line_chart(thresholds, x="threshold", y="winrate_pct")
        if not buckets.empty:
            st.subheader("Confidence Buckets")
            st.dataframe(buckets, width="stretch", hide_index=True)
    
    with tabs[5]:
        st.subheader("Formula QA")
        formula_cols = st.columns(4)
        formula_cols[0].metric("Core Formula Status", str(formula_qa_report.get("core_status", "unknown")).upper())
        formula_cols[1].metric("PineScript Parity", str(formula_qa_report.get("pinescript_parity_status", "unknown")).upper())
        formula_cols[2].metric("Formula Areas", len(strategy_formula_inventory))
        formula_cols[3].metric("PineScripts", len(pinescript_translation_matrix))
    
        formula_checks = pd.DataFrame(formula_qa_report.get("checks", []))
        if not formula_checks.empty:
            st.dataframe(formula_checks, width="stretch", hide_index=True)
    
        st.subheader("Strategy Formula Inventory")
        formula_inventory_frame = pd.DataFrame(strategy_formula_inventory)
        if formula_inventory_frame.empty:
            st.info("No formula inventory loaded.")
        else:
            st.dataframe(formula_inventory_frame, width="stretch", hide_index=True)
    
        st.subheader("PineScript Translation Matrix")
        pinescript_frame = pd.DataFrame(pinescript_translation_matrix)
        if pinescript_frame.empty:
            st.info("No PineScript translation rows loaded.")
        else:
            st.dataframe(pinescript_frame, width="stretch", hide_index=True)
    
        st.subheader("Formula Verification Checklist")
        formula_verification_frame = pd.DataFrame(formula_verification_checklist)
        if not formula_verification_frame.empty:
            st.dataframe(formula_verification_frame, width="stretch", hide_index=True)
            selected_formula_command = st.selectbox(
                "Formula Verification Command",
                formula_verification_frame["name"].tolist(),
                key="formula_verification_command_selector",
            )
            formula_command = formula_verification_frame[formula_verification_frame["name"] == selected_formula_command].iloc[0]
            st.code(formula_command["command"], language="powershell")
    
        st.subheader("Signal Mix")
        if combined_df.empty:
            st.info("No accepted or shadow signals loaded.")
        else:
            mix = combined_df.groupby(["source", "strategy", "timeframe"], dropna=False).size().reset_index(name="signals")
            st.dataframe(mix.sort_values("signals", ascending=False), width="stretch", hide_index=True)
    
        st.subheader("Labeled Performance")
        real_perf = _aggregate_labeled(labeled_df, ["timeframe", "setup_type", "direction"])
        shadow_perf = _aggregate_labeled(shadow_labeled_df, ["timeframe", "setup_type", "direction"])
        perf_tabs = st.tabs(["Real Labels", "Shadow Labels"])
        with perf_tabs[0]:
            st.dataframe(real_perf, width="stretch", hide_index=True)
        with perf_tabs[1]:
            st.dataframe(shadow_perf, width="stretch", hide_index=True)
    
    with tabs[6]:
        st.subheader("Backtest Files")
        st.dataframe(backtest_files, width="stretch", hide_index=True)
    
        st.subheader("Forward Evidence Summary")
        forward_frame = pd.DataFrame([
            {"source": source, **metrics}
            for source, metrics in forward_summary.items()
        ])
        st.dataframe(forward_frame, width="stretch", hide_index=True)
    
        st.subheader("Forward Readiness Gate")
        forward_gate = readiness_report.get("forward_gate", {})
        forward_gate_cols = st.columns(4)
        forward_gate_cols[0].metric("Status", str(forward_gate.get("status", "unknown")).upper())
        forward_gate_cols[1].metric("Resolved", _display(forward_gate.get("resolved_forward_trades")))
        forward_gate_cols[2].metric("Min Required", _display(forward_gate.get("minimum_resolved")))
        forward_gate_cols[3].metric("Profit Factor", _display(forward_gate.get("profit_factor")))
        st.dataframe(pd.DataFrame([{
            "reasons": " | ".join(forward_gate.get("reasons", [])),
            "manual_action": forward_gate.get("manual_action"),
            "recommended_threshold": forward_gate.get("recommended_threshold"),
            "max_consecutive_losses": forward_gate.get("max_consecutive_losses"),
        }]), width="stretch", hide_index=True)
        st.download_button(
            "Download Forward Evidence CSV",
            data=forward_frame.to_csv(index=False).encode("utf-8"),
            file_name="forex_smc_forward_evidence.csv",
            mime="text/csv",
            key="download_forward_evidence_csv",
        )
    
        report_payload = build_report_export_payload(
            snapshot=snapshot,
            accepted_signals=accepted_df.to_dict("records"),
            shadow_signals=shadow_df.to_dict("records"),
            confidence_summary=bucket_summary_df,
            health_checks=health_checks,
            forward_summary=forward_summary,
        )
        st.download_button(
            "Download Dashboard Report JSON",
            data=report_payload["json"].encode("utf-8"),
            file_name=report_payload["filename"],
            mime="application/json",
            key="download_dashboard_report_json",
        )
    
        st.subheader("Backtest Simulation")
        if backtest_df.empty:
            st.info("No backtest_simulation_results.csv loaded.")
        else:
            st.dataframe(backtest_df.tail(200), width="stretch", height=360)
    
        st.subheader("Forward Evidence")
        forward = shadow_df.copy()
        if not forward.empty:
            forward_summary = forward.groupby(["status", "result"], dropna=False).size().reset_index(name="signals")
            st.dataframe(forward_summary, width="stretch", hide_index=True)
        else:
            st.info("No shadow forward signals loaded.")
    
        st.subheader("Logs")
        log_frame = pd.DataFrame(log_inventory)
        if log_frame.empty:
            st.info("No data log files found.")
        else:
            st.dataframe(log_frame, width="stretch", hide_index=True)
            selected_log = st.selectbox(
                "Log File",
                log_frame["relative_path"].tolist(),
                key="log_file_selector",
            )
            tail_lines = st.slider(
                "Tail Lines",
                min_value=20,
                max_value=300,
                value=120,
                step=20,
                key="log_tail_lines",
            )
            log_tail = read_log_tail(selected_log, max_lines=tail_lines)
            if log_tail["status"] == "ok":
                st.text_area("Log Tail", value=log_tail["text"], height=360)
            else:
                st.warning(f"Log read status: {log_tail['status']}")

with st.sidebar:
    render_dashboard_filters()
render_dashboard_body()
