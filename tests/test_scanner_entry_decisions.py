import pandas as pd

import src.scanner_worker as scanner_worker
from src.scanner_worker import (
    apply_smc_detectors,
    choose_dual_market_entry_option,
    choose_dual_recovery_execution_mode,
    choose_recovery_execution_mode,
    send_recovery_alert_with_chart,
    should_promote_low_confidence_record,
    should_market_enter_setup,
)


def _setup(**overrides):
    data = {
        "direction": 1,
        "entry_price": 100.0,
        "sl_price": 90.0,
        "tp_price": 120.0,
        "rejection_confirmed": True,
    }
    data.update(overrides)
    return data


def test_should_market_enter_buy_when_rejection_confirmed_inside_entry_zone():
    setup = _setup(direction=1, entry_price=100.0, sl_price=90.0)

    assert should_market_enter_setup(setup, current_price=95.0) is True


def test_should_market_enter_sell_when_rejection_confirmed_inside_entry_zone():
    setup = _setup(direction=-1, entry_price=100.0, sl_price=110.0)

    assert should_market_enter_setup(setup, current_price=105.0) is True


def test_should_market_enter_requires_rejection_confirmation():
    setup = _setup(direction=1, entry_price=100.0, sl_price=90.0, rejection_confirmed=False)

    assert should_market_enter_setup(setup, current_price=95.0) is False


def test_should_market_enter_rejects_price_outside_entry_zone():
    setup = _setup(direction=1, entry_price=100.0, sl_price=90.0)

    assert should_market_enter_setup(setup, current_price=101.0) is False
    assert should_market_enter_setup(setup, current_price=90.0) is False


def test_choose_dual_market_entry_option_buy_selects_correct_layer():
    opt_a = _setup(direction=1, entry_price=100.0, sl_price=90.0)
    opt_b = _setup(direction=1, entry_price=98.0, sl_price=90.0)

    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=99.0) == "a"
    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=95.0) == "b"
    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=101.0) is None


def test_choose_dual_market_entry_option_sell_selects_correct_layer():
    opt_a = _setup(direction=-1, entry_price=100.0, sl_price=110.0)
    opt_b = _setup(direction=-1, entry_price=102.0, sl_price=110.0)

    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=101.0) == "a"
    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=105.0) == "b"
    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=99.0) is None


def test_recovery_execution_uses_market_when_price_returns_to_rejection_zone():
    setup = _setup(direction=1, entry_price=100.0, sl_price=90.0, rejection_confirmed=True)

    assert choose_recovery_execution_mode(setup, current_price=96.0) == "market"
    assert choose_recovery_execution_mode(setup, current_price=101.0) == "pending"


def test_low_confidence_registry_record_can_promote_to_live_single_execution():
    record = {
        "is_low_confidence": True,
        "ticket_id": None,
        "reentries_count": 0,
    }

    assert should_promote_low_confidence_record(record, ("ticket_id",)) is True


def test_low_confidence_registry_record_does_not_promote_when_ticket_exists():
    record = {
        "is_low_confidence": True,
        "ticket_id": 123456,
    }

    assert should_promote_low_confidence_record(record, ("ticket_id",)) is False


def test_low_confidence_registry_record_can_promote_to_live_dual_execution():
    record = {
        "is_low_confidence": True,
        "ticket_a": None,
        "ticket_b": None,
    }

    assert should_promote_low_confidence_record(record, ("ticket_a", "ticket_b")) is True


def test_dual_recovery_execution_uses_same_layer_priority_as_new_market_entry():
    opt_a = _setup(direction=1, entry_price=100.0, sl_price=90.0)
    opt_b = _setup(direction=1, entry_price=98.0, sl_price=90.0)

    assert choose_dual_recovery_execution_mode(opt_a, opt_b, current_price=99.0, option="a") == "market"
    assert choose_dual_recovery_execution_mode(opt_a, opt_b, current_price=99.0, option="b") == "skip"
    assert choose_dual_recovery_execution_mode(opt_a, opt_b, current_price=95.0, option="a") == "skip"
    assert choose_dual_recovery_execution_mode(opt_a, opt_b, current_price=95.0, option="b") == "market"
    assert choose_dual_recovery_execution_mode(opt_a, opt_b, current_price=101.0, option="a") == "pending"


def test_apply_smc_detectors_includes_supply_demand_detector(monkeypatch):
    calls = []

    def marker(name):
        def _inner(df, *args, **kwargs):
            calls.append(name)
            return df
        return _inner

    import src.scanner_worker as scanner_worker

    monkeypatch.setattr(scanner_worker, "detect_swing_points", marker("swing"))
    monkeypatch.setattr(scanner_worker, "detect_structures", marker("structure"))
    monkeypatch.setattr(scanner_worker, "detect_fvg_and_ob", marker("fvg_ob"))
    monkeypatch.setattr(scanner_worker, "detect_snr_and_swapzones", marker("swapzone"))
    monkeypatch.setattr(scanner_worker, "detect_bpr", marker("bpr"))
    monkeypatch.setattr(scanner_worker, "detect_indecision_candles", marker("ic"))
    monkeypatch.setattr(scanner_worker, "detect_supply_demand_zones", marker("snd"))

    df = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5]})

    result = apply_smc_detectors(df, symbol="XAUUSD")

    assert result is df
    assert calls == ["swing", "structure", "fvg_ob", "swapzone", "bpr", "ic", "snd"]


def _bpr_forms_on_last_candle_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-06-09 09:00", periods=6, freq="15min"),
            "Open": [102.0, 99.0, 96.0, 96.0, 98.0, 101.0],
            "High": [105.0, 100.0, 98.0, 99.0, 102.0, 103.0],
            "Low": [100.0, 95.0, 94.0, 94.0, 97.0, 101.0],
            "Close": [103.0, 96.0, 95.0, 98.0, 101.0, 102.0],
            "Volume": [100, 110, 120, 130, 140, 150],
        }
    )


def test_apply_smc_detectors_closed_only_ignores_bpr_on_forming_last_candle():
    df = _bpr_forms_on_last_candle_df()

    result = apply_smc_detectors(df, symbol="XAUUSD", closed_only=True)

    assert result["BPR_Type"].dropna().empty


def test_apply_smc_detectors_closed_only_detects_bpr_after_next_candle_opens():
    df = _bpr_forms_on_last_candle_df()
    df.loc[len(df)] = {
        "time": pd.Timestamp("2026-06-09 10:30"),
        "Open": 102.0,
        "High": 102.2,
        "Low": 101.5,
        "Close": 101.8,
        "Volume": 160,
    }

    result = apply_smc_detectors(df, symbol="XAUUSD", closed_only=True)

    assert result["BPR_Type"].iloc[5] == "BULLISH"


def test_recovery_alert_generates_chart_and_sends_photo(monkeypatch):
    df = pd.DataFrame({"Close": [1.0, 2.0]})
    setup = _setup(index=42, timeframe="M15", strategy="FVG")
    expected_image = "temp_alert_M15_recovery_single_42.png"
    calls = {}
    removed = []

    def fake_plot(tf_df, title, active_setups, output_filename):
        calls["plot"] = {
            "df": tf_df,
            "title": title,
            "setups": active_setups,
            "image": output_filename,
        }

    def fake_send(message, image_path=None):
        calls["send"] = {"message": message, "image": image_path}
        return True

    monkeypatch.setattr(scanner_worker, "plot_smc_chart", fake_plot)
    monkeypatch.setattr(scanner_worker, "send_telegram_alert", fake_send)
    monkeypatch.setattr(scanner_worker.os.path, "exists", lambda path: path == expected_image)
    monkeypatch.setattr(scanner_worker.os, "remove", lambda path: removed.append(path))

    success = send_recovery_alert_with_chart(
        "Recovery message",
        timeframes_data={"M15": df},
        timeframe="M15",
        symbol="XAUUSD",
        direction_name="BULL",
        strategy="FVG",
        setups=[setup],
        image_suffix="recovery_single",
    )

    assert success is True
    assert calls["plot"]["df"] is df
    assert "XAUUSD M15" in calls["plot"]["title"]
    assert "Recovery" in calls["plot"]["title"]
    assert calls["plot"]["setups"] == [setup]
    assert calls["plot"]["image"] == expected_image
    assert calls["send"] == {"message": "Recovery message", "image": expected_image}
    assert removed == [expected_image]


def test_recovery_alert_falls_back_to_text_when_chart_generation_fails(monkeypatch):
    calls = {}

    def fake_plot(*args, **kwargs):
        raise RuntimeError("chart failed")

    def fake_send(message, image_path=None):
        calls["send"] = {"message": message, "image": image_path}
        return True

    monkeypatch.setattr(scanner_worker, "plot_smc_chart", fake_plot)
    monkeypatch.setattr(scanner_worker, "send_telegram_alert", fake_send)

    success = send_recovery_alert_with_chart(
        "Recovery message",
        timeframes_data={"M15": pd.DataFrame({"Close": [1.0]})},
        timeframe="M15",
        symbol="XAUUSD",
        direction_name="BULL",
        strategy="FVG",
        setups=[_setup(index=11)],
        image_suffix="recovery_single",
    )

    assert success is True
    assert calls["send"] == {"message": "Recovery message", "image": None}
