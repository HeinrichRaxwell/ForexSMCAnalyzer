import pandas as pd
import pytest

from src.entry_quality_gate import (
    OscillatorContext,
    SpreadContext,
    build_oscillator_context,
    build_spread_context,
    evaluate_entry_quality,
    points_to_price,
)


def _setup(**overrides):
    data = {
        "direction": 1,
        "entry_price": 4300.0,
        "sl_price": 4295.0,
        "tp_price": 4310.0,
        "rejection_confirmed": True,
        "htf_prioritized": True,
        "features": {
            "rr_ratio": 2.0,
            "floop_trend_aligned": 1,
            "confluence_score": 2.0,
        },
    }
    data.update(overrides)
    return data


def test_exness_three_digit_points_convert_to_price():
    assert points_to_price(300, point=0.001) == pytest.approx(0.300)
    assert points_to_price(250, point=0.001) == pytest.approx(0.250)


def test_build_spread_context_reads_live_bid_ask_as_points_and_price():
    spread = build_spread_context(bid=4300.100, ask=4300.400, point=0.001, digits=3)

    assert spread.spread_price == pytest.approx(0.300)
    assert spread.spread_points == pytest.approx(300)
    assert spread.digits == 3


def test_fvg_remains_live_eligible_when_confidence_spread_and_oscillators_are_good(monkeypatch):
    monkeypatch.delenv("ML_ENTRY_BASE_THRESHOLD", raising=False)
    monkeypatch.delenv("ML_ENTRY_BUY_CONFIDENCE_BONUS", raising=False)

    decision = evaluate_entry_quality(
        _setup(),
        strategy="FVG",
        probability=0.61,
        accept_threshold=0.50,
        spread=SpreadContext(spread_points=280, spread_price=0.280, point=0.001, digits=3),
        oscillator=OscillatorContext(rsi_8=51.0, stoch_rsi_k=0.52, stoch_rsi_d=0.50),
    )

    assert decision.allowed is True
    assert decision.required_confidence == pytest.approx(0.50)
    assert decision.filtered_reason == "entry_gate_pass"


def test_bpr_remains_live_eligible_when_confidence_spread_and_oscillators_are_good(monkeypatch):
    monkeypatch.delenv("ML_ENTRY_BASE_THRESHOLD", raising=False)
    monkeypatch.delenv("ML_ENTRY_BUY_CONFIDENCE_BONUS", raising=False)

    decision = evaluate_entry_quality(
        _setup(),
        strategy="BPR",
        probability=0.62,
        accept_threshold=0.50,
        spread=SpreadContext(spread_points=290, spread_price=0.290, point=0.001, digits=3),
        oscillator=OscillatorContext(rsi_8=48.0, stoch_rsi_k=0.42, stoch_rsi_d=0.45),
    )

    assert decision.allowed is True
    assert decision.required_confidence == pytest.approx(0.50)


def test_core_strategy_respects_runtime_threshold_without_hidden_sixty_percent_floor(monkeypatch):
    monkeypatch.delenv("ML_ENTRY_BASE_THRESHOLD", raising=False)
    monkeypatch.delenv("ML_ENTRY_BUY_CONFIDENCE_BONUS", raising=False)

    decision = evaluate_entry_quality(
        _setup(),
        strategy="FVG",
        probability=0.45,
        accept_threshold=0.40,
        spread=SpreadContext(spread_points=280, spread_price=0.280, point=0.001, digits=3),
        oscillator=OscillatorContext(rsi_8=51.0, stoch_rsi_k=0.52, stoch_rsi_d=0.50),
    )

    assert decision.allowed is True
    assert decision.required_confidence == pytest.approx(0.40)
    assert decision.filtered_reason == "entry_gate_pass"


def test_blocks_sell_when_rsi8_and_stoch_rsi_are_oversold():
    decision = evaluate_entry_quality(
        _setup(
            direction=-1,
            entry_price=4300.0,
            sl_price=4305.0,
            tp_price=4290.0,
            htf_prioritized=False,
            features={"rr_ratio": 2.0, "floop_trend_aligned": 0, "confluence_score": 0.0},
        ),
        strategy="FVG",
        probability=0.74,
        accept_threshold=0.50,
        spread=SpreadContext(spread_points=250, spread_price=0.250, point=0.001, digits=3),
        oscillator=OscillatorContext(rsi_8=27.0, stoch_rsi_k=0.12, stoch_rsi_d=0.15),
    )

    assert decision.allowed is False
    assert decision.filtered_reason == "entry_gate_oscillator_oversold_sell"


def test_blocks_buy_when_rsi8_and_stoch_rsi_are_overbought():
    decision = evaluate_entry_quality(
        _setup(direction=1, htf_prioritized=False, features={"rr_ratio": 2.0, "floop_trend_aligned": 0, "confluence_score": 0.0}),
        strategy="BPR",
        probability=0.74,
        accept_threshold=0.50,
        spread=SpreadContext(spread_points=250, spread_price=0.250, point=0.001, digits=3),
        oscillator=OscillatorContext(rsi_8=74.0, stoch_rsi_k=0.88, stoch_rsi_d=0.86),
    )

    assert decision.allowed is False
    assert decision.filtered_reason == "entry_gate_oscillator_overbought_buy"


def test_htf_support_allows_oversold_sell_with_reason_metadata():
    setup = _setup(
        direction=-1,
        entry_price=4300.0,
        sl_price=4305.0,
        tp_price=4290.0,
        htf_prioritized=True,
        features={"rr_ratio": 2.0, "floop_trend_aligned": 1, "confluence_score": 2.0},
    )

    decision = evaluate_entry_quality(
        setup,
        strategy="FVG",
        probability=0.74,
        accept_threshold=0.50,
        spread=SpreadContext(spread_points=250, spread_price=0.250, point=0.001, digits=3),
        oscillator=OscillatorContext(rsi_8=27.0, stoch_rsi_k=0.12, stoch_rsi_d=0.15),
    )

    assert decision.allowed is True
    assert decision.filtered_reason == "entry_gate_pass_htf_supported_oscillator_extreme"
    assert decision.required_confidence == pytest.approx(0.70)


def test_wide_spread_is_metadata_not_a_skip_reason():
    decision = evaluate_entry_quality(
        _setup(),
        strategy="FVG",
        probability=0.80,
        accept_threshold=0.50,
        spread=SpreadContext(spread_points=420, spread_price=0.420, point=0.001, digits=3),
        oscillator=OscillatorContext(rsi_8=50.0, stoch_rsi_k=0.50, stoch_rsi_d=0.50),
    )

    assert decision.allowed is True
    assert decision.filtered_reason == "entry_gate_pass"


def test_high_spread_r_is_metadata_not_a_skip_reason():
    decision = evaluate_entry_quality(
        _setup(entry_price=4300.0, sl_price=4299.0),
        strategy="FVG",
        probability=0.80,
        accept_threshold=0.50,
        spread=SpreadContext(spread_points=300, spread_price=0.300, point=0.001, digits=3),
        oscillator=OscillatorContext(rsi_8=50.0, stoch_rsi_k=0.50, stoch_rsi_d=0.50),
    )

    assert decision.allowed is True
    assert decision.filtered_reason == "entry_gate_pass"
    assert decision.spread_r == pytest.approx(0.30)


def test_very_high_confidence_can_pass_oscillator_extreme_without_htf_support():
    decision = evaluate_entry_quality(
        _setup(
            direction=-1,
            entry_price=4300.0,
            sl_price=4305.0,
            tp_price=4290.0,
            htf_prioritized=False,
            features={"rr_ratio": 2.0, "floop_trend_aligned": 0, "confluence_score": 0.0},
        ),
        strategy="FVG",
        probability=0.84,
        accept_threshold=0.50,
        spread=SpreadContext(spread_points=250, spread_price=0.250, point=0.001, digits=3),
        oscillator=OscillatorContext(rsi_8=27.0, stoch_rsi_k=0.12, stoch_rsi_d=0.15),
    )

    assert decision.allowed is True
    assert decision.filtered_reason == "entry_gate_pass_oscillator_extreme_high_confidence"
    assert decision.required_confidence == pytest.approx(0.80)


def test_build_oscillator_context_uses_rsi8_and_stoch_rsi_933():
    df = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105, 106, 105, 104, 103, 102, 101, 102, 103, 104, 105, 106]})

    context = build_oscillator_context(df)

    assert context.rsi_8 is not None
    assert 0.0 <= context.rsi_8 <= 100.0
    assert context.stoch_rsi_k is not None
    assert 0.0 <= context.stoch_rsi_k <= 1.0
    assert context.stoch_rsi_d is not None
    assert 0.0 <= context.stoch_rsi_d <= 1.0
