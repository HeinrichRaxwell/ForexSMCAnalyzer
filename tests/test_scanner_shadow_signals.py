import json
import os
import sys
from unittest.mock import patch

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scanner_worker import (
    get_accept_threshold,
    process_existing_shadow_outcomes,
    register_entry_gate_filtered_lead,
    register_low_confidence_lead,
    register_shadow_candidate,
)


def _setup(**overrides):
    data = {
        "time": "2026-06-07 10:00:00",
        "index": 42,
        "direction": 1,
        "entry_price": 2300.0,
        "sl_price": 2295.0,
        "tp_price": 2310.0,
        "features": {
            "timeframe": 15,
            "direction": 1,
            "entry_price": 2300.0,
            "sl_price": 2295.0,
            "tp_price": 2310.0,
        },
    }
    data.update(overrides)
    return data


def test_get_accept_threshold_defaults_to_fifty_percent(monkeypatch):
    monkeypatch.delenv("ML_ACCEPT_THRESHOLD", raising=False)
    monkeypatch.setenv("MT5_EXECUTE_TRADES", "False")

    assert get_accept_threshold(None) == 0.50


def test_get_accept_threshold_uses_env_when_cli_threshold_missing(monkeypatch):
    monkeypatch.setenv("ML_ACCEPT_THRESHOLD", "0.62")

    assert get_accept_threshold(None) == 0.62


def test_get_accept_threshold_prefers_cli_over_env(monkeypatch):
    monkeypatch.setenv("ML_ACCEPT_THRESHOLD", "0.62")
    monkeypatch.setenv("MT5_EXECUTE_TRADES", "False")

    assert get_accept_threshold(0.55) == 0.55


def test_get_accept_threshold_does_not_add_hidden_live_minimum_when_execution_enabled(monkeypatch):
    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.delenv("ML_LIVE_MIN_THRESHOLD", raising=False)

    assert get_accept_threshold(0.40) == 0.40


def test_get_accept_threshold_uses_configured_live_minimum(monkeypatch):
    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("ML_LIVE_MIN_THRESHOLD", "0.65")

    assert get_accept_threshold(0.60) == 0.65


def test_get_accept_threshold_falls_back_on_invalid_env(monkeypatch):
    monkeypatch.setenv("ML_ACCEPT_THRESHOLD", "not-a-number")
    monkeypatch.setenv("MT5_EXECUTE_TRADES", "False")

    assert get_accept_threshold(None) == 0.50


def test_register_shadow_candidate_records_zero_confidence_single_without_order_ticket(tmp_path):
    shadow_path = tmp_path / "shadow_signals.json"

    changed = register_shadow_candidate(
        sig_key="M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(),
        probability=0.0,
        shadow_signals_file=str(shadow_path),
        now="2026-06-07 10:01:00",
    )

    assert changed is True
    data = json.loads(shadow_path.read_text())
    record = data["M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00"]
    assert record["confidence"] == 0.0
    assert record["status"] == "open"
    assert record["ticket_id"] is None
    assert record["filtered_reason"] == "below_accept_threshold"


def test_register_shadow_candidate_ignores_live_accepted_confidence(tmp_path):
    shadow_path = tmp_path / "shadow_signals.json"

    changed = register_shadow_candidate(
        sig_key="M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(),
        probability=0.50,
        shadow_signals_file=str(shadow_path),
        now="2026-06-07 10:01:00",
    )

    assert changed is False
    assert not shadow_path.exists()


def test_register_shadow_candidate_can_force_high_confidence_entry_gate_filter(tmp_path):
    shadow_path = tmp_path / "shadow_signals.json"

    changed = register_shadow_candidate(
        sig_key="M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.60,
        opt=_setup(),
        probability=0.72,
        shadow_signals_file=str(shadow_path),
        now="2026-06-07 10:01:00",
        force=True,
        filtered_reason="entry_gate_oscillator_overbought_buy",
    )

    assert changed is True
    data = json.loads(shadow_path.read_text())
    record = data["M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00"]
    assert record["confidence"] == 0.72
    assert record["accept_threshold"] == 0.60
    assert record["filtered_reason"] == "entry_gate_oscillator_overbought_buy"


def test_register_entry_gate_filtered_lead_does_not_shadow_above_threshold(tmp_path):
    """Entry-gate-filtered lead with prob >= threshold must NOT be shadow-tracked (bug fix)."""
    shadow_path = tmp_path / "shadow_signals.json"
    sent_signals = {}
    lead = {
        "is_dual": False,
        "timeframe": "M15",
        "strategy": "BPR",
        "direction": -1,
        "max_prob": 0.73,
        "opt": _setup(direction=-1, entry_price=2300.0, sl_price=2305.0, tp_price=2290.0),
    }

    changed = register_entry_gate_filtered_lead(
        lead=lead,
        sent_signals=sent_signals,
        symbol="XAUUSD",
        timeframe="M15",
        strategy="BPR",
        direction_name="BEAR",
        accept_threshold=0.60,
        filtered_reason="entry_gate_oscillator_oversold_sell",
        shadow_signals_file=str(shadow_path),
        now="2026-06-07 10:01:00",
    )

    assert changed is False
    assert sent_signals == {}
    assert not shadow_path.exists()


def test_register_low_confidence_lead_writes_sent_registry_and_shadow_store(tmp_path):
    shadow_path = tmp_path / "shadow_signals.json"
    sent_signals = {}
    lead = {
        "is_dual": False,
        "timeframe": "M15",
        "strategy": "FVG",
        "direction": 1,
        "max_prob": 0.0,
        "opt": _setup(),
    }

    changed = register_low_confidence_lead(
        lead=lead,
        sent_signals=sent_signals,
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        shadow_signals_file=str(shadow_path),
        now="2026-06-07 10:01:00",
    )

    assert changed is True
    assert lead["opt"]["status"] == "FILTERED (Low Confidence)"
    assert list(sent_signals) == ["M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00"]
    assert sent_signals["M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00"]["is_low_confidence"] is True

    shadow_data = json.loads(shadow_path.read_text())
    shadow = shadow_data["M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00"]
    assert shadow["confidence"] == 0.0
    assert shadow["status"] == "open"
    assert shadow["ticket_id"] is None


def test_register_low_confidence_dual_lead_writes_two_shadow_legs(tmp_path):
    shadow_path = tmp_path / "shadow_signals.json"
    sent_signals = {}
    lead = {
        "is_dual": True,
        "timeframe": "H1",
        "strategy": "FVG",
        "direction": -1,
        "max_prob": 0.49,
        "opt_a": _setup(direction=-1, entry_price=2300.0, sl_price=2305.0, tp_price=2290.0),
        "opt_b": _setup(direction=-1, entry_price=2298.0, sl_price=2305.0, tp_price=2290.0),
        "prob_a": 0.12,
        "prob_b": 0.49,
    }

    changed = register_low_confidence_lead(
        lead=lead,
        sent_signals=sent_signals,
        symbol="XAUUSD",
        timeframe="H1",
        strategy="FVG",
        direction_name="BEAR",
        accept_threshold=0.50,
        shadow_signals_file=str(shadow_path),
        now="2026-06-07 10:01:00",
    )

    assert changed is True
    sig_key = "H1_FVG_DUAL_BEAR_2300.000_2298.000_2026-06-07_10:00:00"
    assert list(sent_signals) == [sig_key]
    assert sent_signals[sig_key]["is_low_confidence"] is True

    shadow_data = json.loads(shadow_path.read_text())
    assert set(shadow_data) == {f"{sig_key}_0.5", f"{sig_key}_0.618"}
    assert shadow_data[f"{sig_key}_0.5"]["confidence"] == 0.12
    assert shadow_data[f"{sig_key}_0.618"]["confidence"] == 0.49
    assert all(record["ticket_id"] is None for record in shadow_data.values())


def test_process_existing_shadow_outcomes_passes_fetched_timeframes_to_resolver(tmp_path):
    timeframes_data = {
        "M15": pd.DataFrame([{"time": "2026-06-07 10:15:00", "High": 111.0, "Low": 99.0}]),
        "H1": pd.DataFrame([{"time": "2026-06-07 11:00:00", "High": 112.0, "Low": 98.0}]),
    }
    shadow_path = tmp_path / "shadow_signals.json"
    labeled_path = tmp_path / "shadow_labeled_setups.csv"

    with patch(
        "src.scanner_worker.process_shadow_signal_outcomes",
        return_value={"resolved_count": 1, "expired_count": 0, "labeled_rows_appended": 1},
    ) as mock_process, patch(
        "src.inference.check_and_trigger_retraining",
        return_value={"retrained": True, "status": "ACCEPTED"},
    ) as mock_retrain:
        result = process_existing_shadow_outcomes(
            timeframes_data,
            shadow_signals_file=str(shadow_path),
            shadow_labeled_data_path=str(labeled_path),
            now="2026-06-07 10:31:00",
        )

    assert result["resolved_count"] == 1
    assert result["retrain_result"] == {"retrained": True, "status": "ACCEPTED"}
    mock_retrain.assert_called_once_with(1)
    mock_process.assert_called_once_with(
        timeframes_data,
        shadow_signals_file=str(shadow_path),
        shadow_labeled_data_path=str(labeled_path),
        now="2026-06-07 10:31:00",
    )


def test_process_existing_shadow_outcomes_does_not_retrain_without_new_labeled_rows(tmp_path):
    timeframes_data = {
        "M15": pd.DataFrame([{"time": "2026-06-07 10:15:00", "High": 111.0, "Low": 99.0}]),
    }

    with patch(
        "src.scanner_worker.process_shadow_signal_outcomes",
        return_value={"resolved_count": 0, "expired_count": 0, "labeled_rows_appended": 0},
    ), patch("src.inference.check_and_trigger_retraining") as mock_retrain:
        result = process_existing_shadow_outcomes(timeframes_data)

    assert result["retrain_result"] is None
    mock_retrain.assert_not_called()


# ---------------------------------------------------------------------------
# Bug regression: force=True shadow-pollution fix
# https://github.com/haidar/forex-smc-analyzer/issues/XXX
# ---------------------------------------------------------------------------

def _make_single_lead(prob, entry_price=1900.0, tp_price=1920.0, sl_price=1890.0):
    opt = {
        "time": "2026-07-01 10:00:00",
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "direction": 1,
        "index": 0,
        "option_name": "test_opt",
        "features": {"timeframe": 60, "direction": 1},
        "filtered_reason": "strategy_not_allowlisted",
    }
    return {"is_dual": False, "opt": opt, "max_prob": prob}


def test_above_threshold_not_shadow_tracked(tmp_path):
    """Signal with confidence above accept_threshold must NOT be shadow-tracked."""
    shadow_file = str(tmp_path / "shadow_signals.json")
    sent_signals = {}
    lead = _make_single_lead(prob=0.70)  # above threshold=0.50

    register_entry_gate_filtered_lead(
        lead=lead,
        sent_signals=sent_signals,
        symbol="XAUUSD",
        timeframe="H1",
        strategy="IC",
        direction_name="SHORT",
        accept_threshold=0.50,
        shadow_signals_file=shadow_file,
    )

    assert not os.path.exists(shadow_file), (
        "Above-threshold signal (conf=0.70, threshold=0.50) must not create shadow file"
    )


def test_below_threshold_is_shadow_tracked(tmp_path):
    """Signal with confidence below accept_threshold MUST be shadow-tracked."""
    shadow_file = str(tmp_path / "shadow_signals.json")
    sent_signals = {}
    lead = _make_single_lead(prob=0.40)  # below threshold=0.50

    register_entry_gate_filtered_lead(
        lead=lead,
        sent_signals=sent_signals,
        symbol="XAUUSD",
        timeframe="H1",
        strategy="IC",
        direction_name="SHORT",
        accept_threshold=0.50,
        shadow_signals_file=shadow_file,
    )

    assert os.path.exists(shadow_file), "Below-threshold signal must be shadow-tracked"
    with open(shadow_file) as f:
        data = json.load(f)
    assert len(data) == 1, f"Expected 1 shadow signal, got {len(data)}"


def test_at_threshold_not_shadow_tracked(tmp_path):
    """Signal exactly at threshold is NOT below threshold — must not be shadow-tracked."""
    shadow_file = str(tmp_path / "shadow_signals.json")
    sent_signals = {}
    lead = _make_single_lead(prob=0.50)  # exactly at threshold=0.50

    register_entry_gate_filtered_lead(
        lead=lead,
        sent_signals=sent_signals,
        symbol="XAUUSD",
        timeframe="H1",
        strategy="IC",
        direction_name="SHORT",
        accept_threshold=0.50,
        shadow_signals_file=shadow_file,
    )

    assert not os.path.exists(shadow_file), (
        "Signal at exactly threshold=0.50 must not create shadow file"
    )
