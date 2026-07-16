import pandas as pd

import src.scanner_worker as scanner_worker
from src.entry_quality_gate import OscillatorContext, SpreadContext
from src.scanner_worker import (
    assert_rollout_ready_for_live,
    apply_smc_detectors,
    choose_dual_market_entry_option,
    choose_dual_recovery_execution_mode,
    evaluate_live_entry_gate,
    is_price_too_far_execution,
    recovery_failure_action,
    record_recovery_failure,
    is_live_entry_timeframe,
    choose_recovery_execution_mode,
    configure_console_encoding,
    enforce_recovery_strategy_policy,
    send_recovery_alert_with_chart,
    should_place_pending_setup,
    should_promote_low_confidence_record,
    should_retry_unfilled_watch_record,
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


def test_live_entry_timeframes_exclude_m15_but_keep_higher_timeframes():
    assert is_live_entry_timeframe("M15") is False

    for timeframe in ("M30", "H1", "H4", "D1"):
        assert is_live_entry_timeframe(timeframe) is True


def test_should_market_enter_buy_when_rejection_confirmed_inside_entry_zone():
    setup = _setup(direction=1, entry_price=100.0, sl_price=90.0)

    assert should_market_enter_setup(setup, current_price=99.8) is True


def test_should_market_enter_sell_when_rejection_confirmed_inside_entry_zone():
    setup = _setup(direction=-1, entry_price=100.0, sl_price=110.0)

    assert should_market_enter_setup(setup, current_price=100.2) is True


def test_should_market_enter_blocks_when_emergency_reversal_is_already_active():
    setup = _setup(direction=-1, entry_price=4085.4325, sl_price=4092.193)
    timeframes_data = {"M15": pd.DataFrame({"Trend": [1, 1, 1]})}

    assert (
        should_market_enter_setup(
            setup,
            current_price=4085.44,
            timeframe="M15",
            timeframes_data=timeframes_data,
        )
        is False
    )


def test_should_place_pending_blocks_when_trade_manager_would_immediately_exit():
    setup = _setup(direction=1, entry_price=4177.682908, sl_price=4133.245)
    timeframes_data = {"H4": pd.DataFrame({"Trend": [1, 1, -1, -1, -1]})}

    assert (
        should_place_pending_setup(
            setup,
            timeframe="H4",
            timeframes_data=timeframes_data,
        )
        is False
    )
    assert setup["pending_entry_blocked_reason"] == "immediate_emergency_reversal"


def test_should_place_pending_allows_when_trade_manager_would_not_exit():
    setup = _setup(direction=1, entry_price=4180.0645, sl_price=4159.714)
    timeframes_data = {"M30": pd.DataFrame({"Trend": [1, 1, 1, -1]})}

    assert (
        should_place_pending_setup(
            setup,
            timeframe="M30",
            timeframes_data=timeframes_data,
        )
        is True
    )


def test_should_market_enter_requires_rejection_confirmation():
    setup = _setup(direction=1, entry_price=100.0, sl_price=90.0, rejection_confirmed=False)

    assert should_market_enter_setup(setup, current_price=95.0) is False


def test_should_market_enter_rejects_price_outside_entry_zone():
    setup = _setup(direction=1, entry_price=100.0, sl_price=90.0)

    assert should_market_enter_setup(setup, current_price=101.0) is False
    assert should_market_enter_setup(setup, current_price=90.0) is False


def test_watch_zone_refresh_rejection_uses_fresh_m5_confirmation(monkeypatch):
    import src.scanner_worker as scanner_worker

    fresh_frame = pd.DataFrame({"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5]})
    monkeypatch.setattr(scanner_worker, "fetch_historical_data", lambda *args: fresh_frame)
    monkeypatch.setattr(scanner_worker, "detect_rejection_at_level", lambda *args, **kwargs: True)

    confirmed, source = scanner_worker.refresh_watch_zone_rejection(
        "XAUUSD", _setup(timeframe="M30")
    )

    assert confirmed is True
    assert source == "M5"


def test_should_market_enter_waits_for_reclaim_after_deep_sweep():
    setup = _setup(direction=1, entry_price=4079.34189, sl_price=4073.562)

    assert should_market_enter_setup(setup, current_price=4077.811) is False


def test_choose_dual_market_entry_option_buy_selects_correct_layer():
    opt_a = _setup(direction=1, entry_price=100.0, sl_price=90.0)
    opt_b = _setup(direction=1, entry_price=98.0, sl_price=90.0)

    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=99.0) == "a"
    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=97.7) == "b"
    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=101.0) is None


def test_choose_dual_market_entry_option_buy_waits_for_reclaim_after_deep_sweep():
    opt_a = _setup(direction=1, entry_price=4080.5095, sl_price=4073.562)
    opt_b = _setup(direction=1, entry_price=4079.34189, sl_price=4073.562)

    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=4077.811) is None


def test_choose_dual_market_entry_option_sell_selects_correct_layer():
    opt_a = _setup(direction=-1, entry_price=100.0, sl_price=110.0)
    opt_b = _setup(direction=-1, entry_price=102.0, sl_price=110.0)

    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=101.0) == "a"
    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=102.3) == "b"
    assert choose_dual_market_entry_option(opt_a, opt_b, current_price=99.0) is None


def test_choose_dual_market_entry_option_blocks_when_emergency_reversal_is_already_active():
    opt_a = _setup(direction=-1, entry_price=4085.4325, sl_price=4092.193)
    opt_b = _setup(direction=-1, entry_price=4086.556, sl_price=4092.193)
    timeframes_data = {"M15": pd.DataFrame({"Trend": [1, 1, 1]})}

    assert (
        choose_dual_market_entry_option(
            opt_a,
            opt_b,
            current_price=4085.44,
            timeframe="M15",
            timeframes_data=timeframes_data,
        )
        is None
    )


def test_recovery_execution_uses_market_when_price_returns_to_rejection_zone():
    setup = _setup(direction=1, entry_price=100.0, sl_price=90.0, rejection_confirmed=True)

    assert choose_recovery_execution_mode(setup, current_price=99.8) == "market"
    assert choose_recovery_execution_mode(setup, current_price=101.0) == "pending"


def test_low_confidence_registry_record_can_promote_to_live_single_execution():
    record = {
        "is_low_confidence": True,
        "ticket_id": None,
        "reentries_count": 0,
    }

    assert should_promote_low_confidence_record(record, ("ticket_id",)) is True


def test_price_too_far_execution_message_is_detected_for_watch_retry():
    assert is_price_too_far_execution(
        "price is too far from market (123.63 USD > 30.0 USD limit)"
    ) is True
    assert is_price_too_far_execution("Market indicators check failed") is False


def test_recovery_failure_classifies_permanent_and_deferred_execution_states():
    assert recovery_failure_action("Auto-execution disabled (MT5_EXECUTE_TRADES=False in .env)") == "blocked"
    assert recovery_failure_action("Live strategy policy blocked pending order: entry_policy_not_allowlisted:WatchZone:M30:FVG") == "blocked"
    assert recovery_failure_action("max same-direction trades reached (1/1) for XAUUSDm") == "deferred"
    assert recovery_failure_action("price is too far from market (300 pips > 200 pips limit)") == "price_watch"
    assert recovery_failure_action("MT5 order failed: retcode=10030") == "retry"


def test_recovery_failure_does_not_consume_retries_for_deferred_or_blocked_states():
    record = {}
    action = record_recovery_failure(
        record,
        "max concurrent trades reached (6/6) for XAUUSDm",
        1,
        2,
        message_key="message",
        retries_key="retries",
        outcome_key="done",
    )

    assert action == "deferred"
    assert record == {
        "message": "max concurrent trades reached (6/6) for XAUUSDm",
        "watch_status": "execution_deferred",
    }

    action = record_recovery_failure(
        record,
        "Live strategy policy blocked pending order: entry_policy_not_allowlisted:WatchZone:M30:FVG",
        1,
        2,
        message_key="message",
        retries_key="retries",
        outcome_key="done",
    )

    assert action == "blocked"
    assert record["done"] is True
    assert record["watch_status"] == "execution_blocked"


def test_console_encoding_configuration_never_requires_utf8_console_support():
    class ReconfigurableStream:
        def __init__(self):
            self.arguments = None

        def reconfigure(self, **kwargs):
            self.arguments = kwargs

    class PlainStream:
        pass

    stream = ReconfigurableStream()

    assert configure_console_encoding((stream, PlainStream())) == 1
    assert stream.arguments == {"errors": "backslashreplace"}


def test_recovery_rechecks_current_strategy_policy(monkeypatch):
    monkeypatch.setenv("MT5_LIVE_STRATEGY_BLOCKLIST", "")
    monkeypatch.setenv("MT5_STANDARD_LIMIT_STRATEGY_BLOCKLIST", "H4:OB")
    record = {}

    allowed, reason = enforce_recovery_strategy_policy(
        record,
        strategy="OB",
        setup={"strategy": "OB", "timeframe": "H4"},
        probability=0.60,
        timeframe="H4",
        outcome_keys=("outcome_a_recorded", "outcome_b_recorded", "outcome_recorded"),
        message_keys=("watch_last_execution_message_0.5", "watch_last_execution_message_0.618"),
    )

    assert allowed is False
    assert "entry_policy_blocked:Standard Limit:H4:OB" in reason
    assert record["watch_status"] == "execution_blocked"
    assert record["outcome_a_recorded"] is True
    assert record["outcome_b_recorded"] is True
    assert record["outcome_recorded"] is True


def test_price_too_far_watch_record_can_retry_until_ticket_or_outcome_exists():
    record = {
        "watch_reason": "watch_price_too_far",
        "ticket_id": None,
        "outcome_recorded": False,
    }

    assert should_retry_unfilled_watch_record(record, ("ticket_id",)) is True

    record["ticket_id"] = 123456
    assert should_retry_unfilled_watch_record(record, ("ticket_id",)) is False

    record["ticket_id"] = None
    record["outcome_recorded"] = True
    assert should_retry_unfilled_watch_record(record, ("ticket_id",)) is False


def test_scanner_guard_blocks_live_execution_when_rollout_preflight_fails(monkeypatch, tmp_path):
    report_path = tmp_path / "calibration_report.json"
    env_path = tmp_path / ".env"
    report_path.write_text(
        """
        {
          "live_policy": {
            "allowed_timeframes": ["M30", "H1"],
            "blocked_strategies": ["Pivot", "PIVOT_REJECTION", "SND", "Swapzone"],
            "allowed_strategies": ["FVG_OR_BPR"],
            "thresholds": {
              "0.50": {
                "sample_count": 120,
                "expectancy_r": 0.10,
                "max_drawdown_r": 12.0,
                "profit_factor": 1.10,
                "max_consecutive_losses": 8
              }
            },
            "sources": {
              "real": {"expectancy_r": -0.10, "profit_factor": 0.90}
            }
          }
        }
        """,
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "MT5_EXECUTE_TRADES=True",
                "MT5_REQUIRE_ROLLOUT_READY=True",
                "MT5_MAX_CONCURRENT_TRADES=1",
                "MT5_ALLOWED_TIMEFRAMES=M30,H1",
                "MT5_LIVE_STRATEGY_ALLOWLIST=FVG_OR_BPR",
                "MT5_LIVE_STRATEGY_BLOCKLIST=Pivot,SND,Swapzone",
                "MT5_ENFORCE_ENTRY_GATE=False",
                "MT5_DAILY_GOVERNOR_ENABLED=True",
                "ML_LIVE_MIN_THRESHOLD=0.50",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_REQUIRE_ROLLOUT_READY", "True")

    ready, message = assert_rollout_ready_for_live(
        0.50,
        report_path=str(report_path),
        env_path=str(env_path),
    )

    assert ready is False
    assert "expectancy_r" in message


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
    assert choose_dual_recovery_execution_mode(opt_a, opt_b, current_price=95.0, option="a") == "pending"
    assert choose_dual_recovery_execution_mode(opt_a, opt_b, current_price=95.0, option="b") == "pending"
    assert choose_dual_recovery_execution_mode(opt_a, opt_b, current_price=97.7, option="a") == "skip"
    assert choose_dual_recovery_execution_mode(opt_a, opt_b, current_price=97.7, option="b") == "market"
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


def test_live_entry_gate_keeps_fvg_eligible_with_good_spread_and_oscillator(monkeypatch):
    monkeypatch.delenv("ML_ENTRY_BUY_CONFIDENCE_BONUS", raising=False)
    monkeypatch.setattr(
        scanner_worker,
        "get_live_spread_context",
        lambda symbol: SpreadContext(spread_points=280, spread_price=0.280, point=0.001, digits=3),
    )
    monkeypatch.setattr(
        scanner_worker,
        "build_oscillator_context",
        lambda df: OscillatorContext(rsi_8=52.0, stoch_rsi_k=0.50, stoch_rsi_d=0.48),
    )
    setup = _setup(features={"rr_ratio": 2.0})

    decision = evaluate_live_entry_gate(
        setup,
        strategy="FVG",
        probability=0.63,
        accept_threshold=0.50,
        symbol="XAUUSD",
        timeframe="M15",
        timeframes_data={"M15": pd.DataFrame({"Close": [1.0, 2.0]})},
    )

    assert decision.allowed is True
    assert setup["entry_gate"]["filtered_reason"] == "entry_gate_pass"


def test_live_entry_gate_records_rejection_without_blocking_execution(monkeypatch):
    monkeypatch.setattr(
        scanner_worker,
        "get_live_spread_context",
        lambda symbol: SpreadContext(spread_points=250, spread_price=0.250, point=0.001, digits=3),
    )
    monkeypatch.setattr(
        scanner_worker,
        "build_oscillator_context",
        lambda df: OscillatorContext(rsi_8=28.0, stoch_rsi_k=0.12, stoch_rsi_d=0.15),
    )
    setup = _setup(direction=-1, entry_price=100.0, sl_price=105.0, tp_price=90.0, features={"rr_ratio": 2.0})

    decision = evaluate_live_entry_gate(
        setup,
        strategy="BPR",
        probability=0.72,
        accept_threshold=0.50,
        symbol="XAUUSD",
        timeframe="M15",
        timeframes_data={"M15": pd.DataFrame({"Close": [1.0, 2.0]})},
    )

    assert decision.allowed is True
    assert decision.filtered_reason == "entry_gate_observer_only"
    assert "would have blocked" in decision.reason
    assert setup["entry_gate"]["allowed"] is True
    assert setup["entry_gate"]["enforced"] is False
    assert setup["entry_gate"]["would_have_allowed"] is False
    assert setup["entry_gate"]["filtered_reason"] == "entry_gate_oscillator_oversold_sell"


def test_live_entry_gate_ignores_enforcement_env(monkeypatch):
    monkeypatch.setenv("MT5_ENFORCE_ENTRY_GATE", "True")
    monkeypatch.setattr(
        scanner_worker,
        "get_live_spread_context",
        lambda symbol: SpreadContext(spread_points=250, spread_price=0.250, point=0.001, digits=3),
    )
    monkeypatch.setattr(
        scanner_worker,
        "build_oscillator_context",
        lambda df: OscillatorContext(rsi_8=28.0, stoch_rsi_k=0.12, stoch_rsi_d=0.15),
    )
    setup = _setup(direction=-1, entry_price=100.0, sl_price=105.0, tp_price=90.0, features={"rr_ratio": 2.0})

    decision = evaluate_live_entry_gate(
        setup,
        strategy="BPR",
        probability=0.72,
        accept_threshold=0.50,
        symbol="XAUUSD",
        timeframe="M15",
        timeframes_data={"M15": pd.DataFrame({"Close": [1.0, 2.0]})},
    )

    assert decision.allowed is True
    assert decision.filtered_reason == "entry_gate_observer_only"
    assert setup["entry_gate"]["allowed"] is True
    assert setup["entry_gate"]["enforced"] is False
    assert setup["entry_gate"]["would_have_allowed"] is False


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
