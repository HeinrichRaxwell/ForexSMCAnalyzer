import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shadow_tracker import (
    build_shadow_signal_records,
    load_shadow_signals,
    process_shadow_signal_outcomes,
    resolve_shadow_record,
    should_shadow_signal,
    upsert_shadow_signals,
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


def test_shadow_signal_tracks_every_confidence_below_live_threshold_by_default(monkeypatch):
    monkeypatch.delenv("ML_SHADOW_MIN_CONFIDENCE", raising=False)

    assert should_shadow_signal(0.0, accept_threshold=0.50) is True
    assert should_shadow_signal(0.12, accept_threshold=0.50) is True
    assert should_shadow_signal(0.4999, accept_threshold=0.50) is True
    assert should_shadow_signal(0.50, accept_threshold=0.50) is False
    assert should_shadow_signal(0.75, accept_threshold=0.50) is False


def test_shadow_signal_optional_minimum_is_configurable(monkeypatch):
    monkeypatch.setenv("ML_SHADOW_MIN_CONFIDENCE", "0.30")

    assert should_shadow_signal(0.2999, accept_threshold=0.50) is False
    assert should_shadow_signal(0.30, accept_threshold=0.50) is True
    assert should_shadow_signal(0.49, accept_threshold=0.50) is True


def test_build_single_shadow_signal_record_is_open_and_non_executable():
    records = build_shadow_signal_records(
        signal_id="M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(),
        probability=0.0,
        now="2026-06-07 10:01:00",
    )

    assert len(records) == 1
    record = records[0]
    assert record["signal_id"] == "M15_FVG_SINGLE_BULL_2300.000_2026-06-07_10:00:00"
    assert record["source"] == "shadow"
    assert record["status"] == "open"
    assert record["result"] is None
    assert record["label"] is None
    assert record["confidence"] == 0.0
    assert record["accept_threshold"] == 0.50
    assert record["filtered_reason"] == "below_accept_threshold"
    assert record["entry_price"] == 2300.0
    assert record["sl_price"] == 2295.0
    assert record["tp_price"] == 2310.0
    assert record["ticket_id"] is None
    assert record["features"]["timeframe"] == 15


def test_build_dual_shadow_records_tracks_each_entry_separately():
    records = build_shadow_signal_records(
        signal_id="H1_FVG_DUAL_BEAR_2300.000_2298.000_2026-06-07_10:00:00",
        symbol="XAUUSD",
        timeframe="H1",
        strategy="FVG",
        direction_name="BEAR",
        accept_threshold=0.50,
        opt_a=_setup(direction=-1, entry_price=2300.0, sl_price=2305.0, tp_price=2290.0),
        probability_a=0.11,
        opt_b=_setup(direction=-1, entry_price=2298.0, sl_price=2305.0, tp_price=2290.0),
        probability_b=0.37,
        now="2026-06-07 10:01:00",
    )

    assert [record["leg"] for record in records] == ["0.5", "0.618"]
    assert [record["signal_id"] for record in records] == [
        "H1_FVG_DUAL_BEAR_2300.000_2298.000_2026-06-07_10:00:00_0.5",
        "H1_FVG_DUAL_BEAR_2300.000_2298.000_2026-06-07_10:00:00_0.618",
    ]
    assert [record["confidence"] for record in records] == [0.11, 0.37]
    assert [record["entry_price"] for record in records] == [2300.0, 2298.0]
    assert all(record["status"] == "open" for record in records)
    assert all(record["ticket_id"] is None for record in records)


def test_upsert_shadow_signals_is_idempotent_and_preserves_resolved_outcome(tmp_path):
    shadow_path = tmp_path / "shadow_signals.json"
    records = build_shadow_signal_records(
        signal_id="sig-1",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(),
        probability=0.25,
        now="2026-06-07 10:01:00",
    )

    assert upsert_shadow_signals(records, shadow_signals_file=str(shadow_path)) is True
    assert upsert_shadow_signals(records, shadow_signals_file=str(shadow_path)) is False
    assert len(load_shadow_signals(str(shadow_path))) == 1

    data = load_shadow_signals(str(shadow_path))
    data["sig-1"]["status"] = "resolved"
    data["sig-1"]["result"] = "tp"
    data["sig-1"]["label"] = 1
    shadow_path.write_text(json.dumps(data, indent=4))

    refreshed_records = build_shadow_signal_records(
        signal_id="sig-1",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(entry_price=2301.0),
        probability=0.26,
        now="2026-06-07 10:02:00",
    )
    assert upsert_shadow_signals(refreshed_records, shadow_signals_file=str(shadow_path)) is True

    final_data = load_shadow_signals(str(shadow_path))
    assert len(final_data) == 1
    assert final_data["sig-1"]["status"] == "resolved"
    assert final_data["sig-1"]["result"] == "tp"
    assert final_data["sig-1"]["label"] == 1
    assert final_data["sig-1"]["latest_seen_at"] == "2026-06-07 10:02:00"


def _candles(rows):
    return pd.DataFrame(rows)


def test_resolve_shadow_record_buy_tp_after_entry():
    record = build_shadow_signal_records(
        signal_id="buy-tp",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(time="2026-06-07 10:00:00", direction=1, entry_price=100.0, sl_price=95.0, tp_price=110.0),
        probability=0.25,
        now="2026-06-07 10:00:30",
    )[0]
    candles = _candles([
        {"time": "2026-06-07 10:15:00", "Open": 102.0, "High": 104.0, "Low": 99.0, "Close": 103.0},
        {"time": "2026-06-07 10:30:00", "Open": 103.0, "High": 111.0, "Low": 101.0, "Close": 110.0},
    ])

    resolved, changed = resolve_shadow_record(record, candles, now="2026-06-07 10:31:00")

    assert changed is True
    assert resolved["status"] == "resolved"
    assert resolved["result"] == "tp"
    assert resolved["label"] == 1
    assert resolved["pnl_relative"] == 2.0
    assert resolved["triggered_at"] == "2026-06-07 10:15:00"
    assert resolved["resolved_at"] == "2026-06-07 10:30:00"


def test_resolve_shadow_record_buy_sl_after_entry():
    record = build_shadow_signal_records(
        signal_id="buy-sl",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(time="2026-06-07 10:00:00", direction=1, entry_price=100.0, sl_price=95.0, tp_price=110.0),
        probability=0.25,
        now="2026-06-07 10:00:30",
    )[0]
    candles = _candles([
        {"time": "2026-06-07 10:15:00", "Open": 101.0, "High": 103.0, "Low": 99.0, "Close": 100.0},
        {"time": "2026-06-07 10:30:00", "Open": 100.0, "High": 102.0, "Low": 94.0, "Close": 95.0},
    ])

    resolved, changed = resolve_shadow_record(record, candles, now="2026-06-07 10:31:00")

    assert changed is True
    assert resolved["status"] == "resolved"
    assert resolved["result"] == "sl"
    assert resolved["label"] == 0
    assert resolved["pnl_relative"] == -1.0


def test_resolve_shadow_record_sell_tp_after_entry():
    record = build_shadow_signal_records(
        signal_id="sell-tp",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BEAR",
        accept_threshold=0.50,
        opt=_setup(time="2026-06-07 10:00:00", direction=-1, entry_price=100.0, sl_price=105.0, tp_price=90.0),
        probability=0.25,
        now="2026-06-07 10:00:30",
    )[0]
    candles = _candles([
        {"time": "2026-06-07 10:15:00", "Open": 98.0, "High": 101.0, "Low": 96.0, "Close": 97.0},
        {"time": "2026-06-07 10:30:00", "Open": 97.0, "High": 99.0, "Low": 89.0, "Close": 90.0},
    ])

    resolved, changed = resolve_shadow_record(record, candles, now="2026-06-07 10:31:00")

    assert changed is True
    assert resolved["status"] == "resolved"
    assert resolved["result"] == "tp"
    assert resolved["label"] == 1
    assert resolved["pnl_relative"] == 2.0


def test_resolve_shadow_record_same_candle_tp_and_sl_uses_conservative_sl():
    record = build_shadow_signal_records(
        signal_id="same-candle",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(time="2026-06-07 10:00:00", direction=1, entry_price=100.0, sl_price=95.0, tp_price=110.0),
        probability=0.25,
        now="2026-06-07 10:00:30",
    )[0]
    candles = _candles([
        {"time": "2026-06-07 10:15:00", "Open": 100.0, "High": 111.0, "Low": 94.0, "Close": 105.0},
    ])

    resolved, changed = resolve_shadow_record(record, candles, now="2026-06-07 10:16:00")

    assert changed is True
    assert resolved["status"] == "resolved"
    assert resolved["result"] == "sl"
    assert resolved["label"] == 0
    assert resolved["pnl_relative"] == -1.0


def test_resolve_shadow_record_expires_after_max_bars_without_entry_or_exit():
    record = build_shadow_signal_records(
        signal_id="expired",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(time="2026-06-07 10:00:00", direction=1, entry_price=100.0, sl_price=95.0, tp_price=110.0),
        probability=0.25,
        now="2026-06-07 10:00:30",
    )[0]
    candles = _candles([
        {"time": "2026-06-07 10:15:00", "Open": 103.0, "High": 104.0, "Low": 101.0, "Close": 103.0},
        {"time": "2026-06-07 10:30:00", "Open": 103.0, "High": 104.0, "Low": 101.0, "Close": 103.0},
    ])

    resolved, changed = resolve_shadow_record(record, candles, max_bars=2, now="2026-06-07 10:31:00")

    assert changed is True
    assert resolved["status"] == "expired"
    assert resolved["result"] == "expired"
    assert resolved["label"] is None
    assert resolved["resolved_at"] == "2026-06-07 10:30:00"


def test_process_shadow_signal_outcomes_updates_json_and_appends_labeled_csv_once(tmp_path):
    shadow_path = tmp_path / "shadow_signals.json"
    labeled_path = tmp_path / "shadow_labeled_setups.csv"
    records = build_shadow_signal_records(
        signal_id="process-buy-tp",
        symbol="XAUUSD",
        timeframe="M15",
        strategy="FVG",
        direction_name="BULL",
        accept_threshold=0.50,
        opt=_setup(time="2026-06-07 10:00:00", direction=1, entry_price=100.0, sl_price=95.0, tp_price=110.0),
        probability=0.25,
        now="2026-06-07 10:00:30",
    )
    upsert_shadow_signals(records, shadow_signals_file=str(shadow_path))
    candles_by_timeframe = {
        "M15": _candles([
            {"time": "2026-06-07 10:15:00", "Open": 102.0, "High": 104.0, "Low": 99.0, "Close": 103.0},
            {"time": "2026-06-07 10:30:00", "Open": 103.0, "High": 111.0, "Low": 101.0, "Close": 110.0},
        ])
    }

    first = process_shadow_signal_outcomes(
        candles_by_timeframe,
        shadow_signals_file=str(shadow_path),
        shadow_labeled_data_path=str(labeled_path),
        now="2026-06-07 10:31:00",
    )
    second = process_shadow_signal_outcomes(
        candles_by_timeframe,
        shadow_signals_file=str(shadow_path),
        shadow_labeled_data_path=str(labeled_path),
        now="2026-06-07 10:32:00",
    )

    assert first["resolved_count"] == 1
    assert first["expired_count"] == 0
    assert first["labeled_rows_appended"] == 1
    assert second["resolved_count"] == 0
    assert second["labeled_rows_appended"] == 0

    shadow_data = load_shadow_signals(str(shadow_path))
    assert shadow_data["process-buy-tp"]["status"] == "resolved"
    assert shadow_data["process-buy-tp"]["result"] == "tp"

    labeled_df = pd.read_csv(labeled_path)
    assert len(labeled_df) == 1
    assert labeled_df.iloc[0]["signal_id"] == "process-buy-tp"
    assert labeled_df.iloc[0]["sample_source"] == "shadow"
    assert labeled_df.iloc[0]["strategy"] == "FVG"
    assert labeled_df.iloc[0]["label"] == 1
    assert labeled_df.iloc[0]["pnl_relative"] == 2.0
    for column in [
        "rr_ratio", "atr_percentile", "body_to_range_ratio",
        "dist_to_recent_swing", "htf_trend_aligned", "confluence_score",
        "order_type", "reaction_strength",
    ]:
        assert column in labeled_df.columns
        assert pd.notna(labeled_df.iloc[0][column])
    assert labeled_df.iloc[0]["rr_ratio"] == 2.0
    assert labeled_df.iloc[0]["order_type"] == 0
