import pandas as pd

from src.real_tick_backtester import (
    calculate_tick_coverage,
    filter_setups_for_run,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    trim_candles_to_range,
)


def test_parse_csv_strings_trims_and_drops_empty_values():
    assert parse_csv_strings(" FVG, OB,,COMBINED ") == ["FVG", "OB", "COMBINED"]


def test_parse_csv_floats_parses_numbers():
    assert parse_csv_floats("50, 100.5") == [50.0, 100.5]


def test_parse_csv_ints_parses_numbers():
    assert parse_csv_ints("1, 3,5") == [1, 3, 5]


def test_filter_setups_for_run_filters_strategy_and_threshold():
    setups = [
        {"strategy": "FVG", "probability": 0.49},
        {"strategy": "FVG", "probability": 0.50},
        {"strategy": "OB", "probability": 0.90},
    ]

    assert filter_setups_for_run(setups, "FVG", 0.50) == [setups[1]]
    assert filter_setups_for_run(setups, "COMBINED", 0.50) == [setups[1], setups[2]]


def test_calculate_tick_coverage_uses_candle_days_not_weekends_only():
    ticks = pd.DataFrame({"time": pd.to_datetime(["2026-06-01 01:00:00"])})
    candle_times = pd.to_datetime(["2026-06-01 00:00:00", "2026-06-02 00:00:00"])

    coverage = calculate_tick_coverage(
        ticks=ticks,
        missing_cache_days=["2026-06-03"],
        candle_times=candle_times,
    )

    assert coverage["tick_days_required"] == 2
    assert coverage["tick_days_with_ticks"] == 1
    assert coverage["tick_days_missing"] == 1
    assert coverage["tick_coverage_pct"] == 50.0
    assert coverage["is_real_tick_complete"] is False


def test_trim_candles_to_range_removes_bars_outside_requested_window():
    candles = pd.DataFrame(
        {
            "time": pd.to_datetime([
                "2026-01-01 20:00:00",
                "2026-01-02 00:00:00",
                "2026-01-02 04:00:00",
                "2026-01-03 00:00:00",
            ]),
            "Open": [1, 2, 3, 4],
        }
    )

    trimmed = trim_candles_to_range(candles, pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03"))

    assert trimmed["time"].tolist() == [
        pd.Timestamp("2026-01-02 00:00:00"),
        pd.Timestamp("2026-01-02 04:00:00"),
    ]
