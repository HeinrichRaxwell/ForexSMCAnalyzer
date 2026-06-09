import json
import os
import sys
from unittest.mock import patch

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scanner_worker import (
    get_accept_threshold,
    process_existing_shadow_outcomes,
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

    assert get_accept_threshold(None) == 0.50


def test_get_accept_threshold_uses_env_when_cli_threshold_missing(monkeypatch):
    monkeypatch.setenv("ML_ACCEPT_THRESHOLD", "0.62")

    assert get_accept_threshold(None) == 0.62


def test_get_accept_threshold_prefers_cli_over_env(monkeypatch):
    monkeypatch.setenv("ML_ACCEPT_THRESHOLD", "0.62")

    assert get_accept_threshold(0.55) == 0.55


def test_get_accept_threshold_falls_back_on_invalid_env(monkeypatch):
    monkeypatch.setenv("ML_ACCEPT_THRESHOLD", "not-a-number")

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
