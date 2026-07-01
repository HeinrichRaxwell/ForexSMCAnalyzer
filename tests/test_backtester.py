import pandas as pd

from src.backtester import (
    DEFAULT_BACKTEST_CONCURRENCIES,
    DEFAULT_BACKTEST_THRESHOLDS,
    build_model_feature_frame,
)


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
