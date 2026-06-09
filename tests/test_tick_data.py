from datetime import datetime

import numpy as np
import pandas as pd

from src.tick_data import (
    build_tick_frame,
    download_ticks_day,
    iter_days,
    load_ticks_range_from_cache,
    read_tick_cache,
    tick_cache_path,
)


def test_iter_days_returns_inclusive_day_boundaries():
    days = list(iter_days(datetime(2026, 1, 1, 12), datetime(2026, 1, 3, 1)))

    assert days == [
        datetime(2026, 1, 1),
        datetime(2026, 1, 2),
        datetime(2026, 1, 3),
    ]


def test_tick_cache_path_uses_symbol_and_date(tmp_path):
    path = tick_cache_path(tmp_path, "XAUUSDm", datetime(2026, 1, 2))

    assert path == tmp_path / "XAUUSDm" / "2026-01-02.csv.gz"


def test_build_tick_frame_normalizes_mt5_tick_array():
    ticks = np.array(
        [
            (1704067200, 2300.1, 2300.3, 0.0, 10, 1704067200123, 2, 10.0),
            (1704067201, 2300.2, 2300.4, 0.0, 11, 1704067201456, 2, 11.0),
        ],
        dtype=[
            ("time", "i8"),
            ("bid", "f8"),
            ("ask", "f8"),
            ("last", "f8"),
            ("volume", "i8"),
            ("time_msc", "i8"),
            ("flags", "i8"),
            ("volume_real", "f8"),
        ],
    )

    frame = build_tick_frame(ticks)

    assert frame.columns.tolist() == ["time", "time_msc", "bid", "ask", "last", "volume", "flags", "volume_real"]
    assert frame["time"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00.123"),
        pd.Timestamp("2024-01-01 00:00:01.456"),
    ]
    assert frame["bid"].tolist() == [2300.1, 2300.2]
    assert frame["ask"].tolist() == [2300.3, 2300.4]


def test_load_ticks_range_from_cache_filters_to_requested_window(tmp_path):
    day = datetime(2026, 1, 2)
    path = tick_cache_path(tmp_path, "XAUUSDm", day)
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": pd.to_datetime([
                "2026-01-02 00:00:00.000",
                "2026-01-02 00:00:01.000",
                "2026-01-02 00:00:02.000",
            ]),
            "time_msc": [1767312000000, 1767312001000, 1767312002000],
            "bid": [2300.0, 2301.0, 2302.0],
            "ask": [2300.2, 2301.2, 2302.2],
            "last": [0.0, 0.0, 0.0],
            "volume": [1, 1, 1],
            "flags": [2, 2, 2],
            "volume_real": [1.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False, compression="gzip")

    frame, missing_days = load_ticks_range_from_cache(
        "XAUUSDm",
        pd.Timestamp("2026-01-02 00:00:00.500"),
        pd.Timestamp("2026-01-02 00:00:02.000"),
        cache_dir=tmp_path,
    )

    assert missing_days == []
    assert frame["bid"].tolist() == [2301.0]


def test_load_ticks_range_from_cache_marks_corrupt_cache_as_missing(tmp_path):
    day = datetime(2026, 1, 2)
    path = tick_cache_path(tmp_path, "XAUUSDm", day)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial gzip")

    frame, missing_days = load_ticks_range_from_cache(
        "XAUUSDm",
        pd.Timestamp("2026-01-02 00:00:00"),
        pd.Timestamp("2026-01-02 23:59:59"),
        cache_dir=tmp_path,
    )

    assert frame.empty
    assert missing_days == ["2026-01-02"]


def test_download_ticks_day_redownloads_corrupt_cache(tmp_path):
    class FakeMT5:
        COPY_TICKS_ALL = 0

        def copy_ticks_range(self, symbol, day_start, day_end, flags):
            return np.array(
                [(1767312000, 2300.0, 2300.2, 0.0, 1, 1767312000000, 2, 1.0)],
                dtype=[
                    ("time", "i8"),
                    ("bid", "f8"),
                    ("ask", "f8"),
                    ("last", "f8"),
                    ("volume", "i8"),
                    ("time_msc", "i8"),
                    ("flags", "i8"),
                    ("volume_real", "f8"),
                ],
            )

    day = datetime(2026, 1, 2)
    path = tick_cache_path(tmp_path, "XAUUSDm", day)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial gzip")

    result = download_ticks_day("XAUUSDm", day, cache_dir=tmp_path, mt5_module=FakeMT5())

    assert result["cached"] is False
    assert result["ticks"] == 1
    assert read_tick_cache(path)["bid"].tolist() == [2300.0]
