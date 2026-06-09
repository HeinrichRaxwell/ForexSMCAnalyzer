from src.scanner_worker import (
    format_dual_signal_message,
    format_single_signal_message,
)


def _base_setup(**overrides):
    setup = {
        "direction": 1,
        "entry_price": 2300.125,
        "sl_price": 2294.500,
        "tp_price": 2310.250,
        "tp2_price": 2318.750,
        "tp3_price": 2328.000,
    }
    setup.update(overrides)
    return setup


def test_format_single_signal_message_uses_professional_sections():
    msg = format_single_signal_message(
        symbol="XAUUSD",
        timeframe="M15",
        direction=1,
        setup_desc="Fair Value Gap 0.500 Entry",
        probability=0.7345,
        confidence_threshold=0.50,
        setup=_base_setup(),
        execution_status="Market order active (ticket #12345)",
        htf_priority_status="Confirmed",
        rejection_status="Confirmed (M5)",
        confluences=[
            "Fair Value Gap M15 (Model confidence: 73.5%)",
            "Supported by FLOOP Pro Trend Filter",
        ],
        htf_matches=[{"timeframe": "H4", "bottom": 2298.0, "top": 2305.0}],
    )

    assert "<b>SMC Trade Signal - XAUUSD</b>" in msg
    assert "<b>Signal</b>" in msg
    assert "<b>Model Confidence</b>" in msg
    assert "<b>Execution</b>" in msg
    assert "<b>Levels</b>" in msg
    assert "<b>Confluence</b>" in msg
    assert "<b>HTF Match</b>" in msg
    assert "Model confidence" in msg
    assert "AI Success Score" not in msg
    assert "HIGH CONFIDENCE" not in msg
    assert "Market order active (ticket #12345)" in msg


def test_format_dual_signal_message_keeps_fib_lot_plan_clear():
    opt_a = _base_setup(entry_price=2300.500)
    opt_b = _base_setup(entry_price=2298.618)

    msg = format_dual_signal_message(
        symbol="XAUUSD",
        timeframe="H1",
        direction=-1,
        setup_desc="Fair Value Gap (Dual Fibonacci Entry)",
        probability_a=0.6123,
        probability_b=0.7789,
        confidence_threshold=0.50,
        opt_a=opt_a,
        opt_b=opt_b,
        execution_status_a="Pending order placed (ticket #111)",
        execution_status_b="Pending order placed (ticket #222)",
        htf_priority_status="Confirmed",
        rejection_status="Confirmed (M15/H1)",
        confluences=["Fair Value Gap H1 (Model confidence: 77.9%)"],
        htf_matches=[],
    )

    assert "0.500 entry (0.01 lot): <code>61.23%</code>" in msg
    assert "0.618 entry (0.02 lot): <code>77.89%</code>" in msg
    assert "Entry 0.500 (0.01 lot): <code>2300.500</code>" in msg
    assert "Entry 0.618 (0.02 lot): <code>2298.618</code>" in msg
    assert "AI Success Score" not in msg
