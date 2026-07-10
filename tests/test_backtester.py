import pandas as pd

from src.backtester import (
    DEFAULT_BACKTEST_CONCURRENCIES,
    DEFAULT_BACKTEST_THRESHOLDS,
    build_model_feature_frame,
    generate_all_setups,
)
from src.smc_detector import detect_bpr, detect_fvg_and_ob


def test_build_model_feature_frame_fills_missing_expected_features_in_order():
    features = [
        {"hour": 8, "direction": 1, "entry_price": 2300.0},
        {"hour": 9, "direction": -1, "entry_price": 2310.0, "knn_prob_sig": 0.42},
    ]
    expected = ["hour", "direction", "entry_price", "knn_prob_sig", "floop_signal"]

    frame = build_model_feature_frame(features, expected)

    assert frame.columns.tolist() == expected
    assert frame["knn_prob_sig"].tolist() == [0.0, 0.42]
    assert frame["floop_signal"].tolist() == [0.0, 0.0]
    assert frame.dtypes.apply(lambda dtype: pd.api.types.is_numeric_dtype(dtype)).all()


def test_default_backtest_thresholds_include_live_threshold():
    assert 0.50 in DEFAULT_BACKTEST_THRESHOLDS


def test_default_backtest_concurrencies_include_live_setting():
    assert 3 in DEFAULT_BACKTEST_CONCURRENCIES


def test_backtester_fvg_setups_start_after_confirmation_candle():
    df = pd.DataFrame({
        "time": pd.date_range("2026-06-01 09:00:00", periods=4, freq="15min"),
        "Open":  [9.0, 10.0, 14.0, 14.5],
        "High":  [10.0, 15.0, 16.0, 15.2],
        "Low":   [8.0, 10.0, 12.0, 13.0],
        "Close": [9.5, 14.0, 15.0, 14.8],
        "ATR_14": [1.0, 1.0, 1.0, 1.0],
        "Trend": [1, 1, 1, 1],
    })
    df = detect_fvg_and_ob(df, symbol="XAUUSD")

    fvg_setups = [setup for setup in generate_all_setups(df) if setup["strategy"] == "FVG"]

    assert fvg_setups
    assert {setup["active_from_index"] for setup in fvg_setups} == {4}


def test_backtester_bpr_setups_start_after_confirmation_candle():
    df = pd.DataFrame({
        "time": pd.date_range("2026-06-01 09:00:00", periods=7, freq="15min"),
        "Open":  [102.0, 99.0, 96.0, 96.0, 98.0, 101.0, 102.0],
        "High":  [105.0, 100.0, 98.0, 99.0, 102.0, 103.0, 102.5],
        "Low":   [100.0, 95.0, 94.0, 94.0, 97.0, 101.0, 101.5],
        "Close": [103.0, 96.0, 95.0, 98.0, 101.0, 102.0, 102.2],
        "ATR_14": [1.0] * 7,
        "Trend": [1] * 7,
    })
    df = detect_bpr(detect_fvg_and_ob(df, symbol="XAUUSD"), symbol="XAUUSD")

    bpr_setups = [setup for setup in generate_all_setups(df) if setup["strategy"] == "BPR"]

    assert bpr_setups
    assert {setup["active_from_index"] for setup in bpr_setups} == {7}
