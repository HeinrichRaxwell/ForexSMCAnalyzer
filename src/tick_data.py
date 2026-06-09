import os
import gzip
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TICK_CACHE_DIR = BASE_DIR / "data" / "ticks"
TICK_COLUMNS = ["time", "time_msc", "bid", "ask", "last", "volume", "flags", "volume_real"]
CACHE_READ_ERRORS = (EOFError, OSError, gzip.BadGzipFile, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError)


def _as_day(value) -> datetime:
    ts = pd.Timestamp(value).to_pydatetime()
    return datetime(ts.year, ts.month, ts.day)


def iter_days(start, end):
    day = _as_day(start)
    end_day = _as_day(end)
    while day <= end_day:
        yield day
        day += timedelta(days=1)


def tick_cache_path(cache_dir, symbol: str, day) -> Path:
    return Path(cache_dir) / str(symbol) / f"{_as_day(day).date().isoformat()}.csv.gz"


def build_tick_frame(ticks) -> pd.DataFrame:
    if ticks is None or len(ticks) == 0:
        return pd.DataFrame(columns=TICK_COLUMNS)

    frame = pd.DataFrame(ticks)
    for column in TICK_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0

    if "time_msc" in frame.columns and pd.to_numeric(frame["time_msc"], errors="coerce").fillna(0).gt(0).any():
        frame["time"] = pd.to_datetime(frame["time_msc"], unit="ms", errors="coerce")
    else:
        frame["time"] = pd.to_datetime(frame["time"], unit="s", errors="coerce")

    frame = frame[TICK_COLUMNS].dropna(subset=["time"]).copy()
    frame = frame.sort_values(["time_msc", "time"], kind="mergesort").reset_index(drop=True)
    return frame


def read_tick_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=TICK_COLUMNS)
    frame = pd.read_csv(path, compression="gzip")
    if frame.empty:
        return pd.DataFrame(columns=TICK_COLUMNS)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    return frame[TICK_COLUMNS].dropna(subset=["time"]).reset_index(drop=True)


def write_tick_cache(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        pd.DataFrame(columns=TICK_COLUMNS).to_csv(path, index=False, compression="gzip")
        return
    frame[TICK_COLUMNS].to_csv(path, index=False, compression="gzip")


def load_ticks_range_from_cache(symbol: str, start, end, cache_dir=DEFAULT_TICK_CACHE_DIR):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frames = []
    missing_days = []

    for day in iter_days(start_ts, end_ts):
        path = tick_cache_path(cache_dir, symbol, day)
        if not path.exists():
            missing_days.append(day.date().isoformat())
            continue
        try:
            cached = read_tick_cache(path)
        except CACHE_READ_ERRORS:
            missing_days.append(day.date().isoformat())
            continue
        if not cached.empty:
            frames.append(cached)

    if not frames:
        return pd.DataFrame(columns=TICK_COLUMNS), missing_days

    frame = pd.concat(frames, ignore_index=True)
    frame = frame[(frame["time"] >= start_ts) & (frame["time"] < end_ts)]
    frame = frame.sort_values(["time_msc", "time"], kind="mergesort").reset_index(drop=True)
    return frame, missing_days


def resolve_active_symbol(symbol: str, mt5_module=None) -> str:
    mt5_module = mt5_module or mt5
    if mt5_module is None:
        raise RuntimeError("MetaTrader5 package is not available.")

    for candidate in [symbol, f"{symbol}m", f"{symbol}.", "GOLD"]:
        mt5_module.symbol_select(candidate, True)
        rates = mt5_module.copy_rates_from_pos(candidate, mt5_module.TIMEFRAME_M1, 0, 1)
        if rates is not None and len(rates) > 0:
            return candidate
    raise RuntimeError(f"No active MT5 symbol found for {symbol}.")


def download_ticks_day(symbol: str, day, cache_dir=DEFAULT_TICK_CACHE_DIR, mt5_module=None, force=False) -> dict:
    mt5_module = mt5_module or mt5
    if mt5_module is None:
        raise RuntimeError("MetaTrader5 package is not available.")

    day_start = _as_day(day)
    day_end = day_start + timedelta(days=1)
    path = tick_cache_path(cache_dir, symbol, day_start)
    if path.exists() and not force:
        try:
            cached = read_tick_cache(path)
            return {"symbol": symbol, "day": day_start.date().isoformat(), "path": str(path), "ticks": len(cached), "cached": True}
        except CACHE_READ_ERRORS:
            pass

    ticks = mt5_module.copy_ticks_range(symbol, day_start, day_end, mt5_module.COPY_TICKS_ALL)
    frame = build_tick_frame(ticks)
    write_tick_cache(frame, path)
    return {"symbol": symbol, "day": day_start.date().isoformat(), "path": str(path), "ticks": len(frame), "cached": False}


def download_ticks_range(
    symbol: str,
    start,
    end,
    cache_dir=DEFAULT_TICK_CACHE_DIR,
    mt5_module=None,
    force=False,
    verbose=False,
) -> pd.DataFrame:
    mt5_module = mt5_module or mt5
    if mt5_module is None:
        raise RuntimeError("MetaTrader5 package is not available.")

    initialized_here = False
    if hasattr(mt5_module, "initialize") and not os.environ.get("PYTEST_CURRENT_TEST"):
        initialized_here = bool(mt5_module.initialize())

    try:
        for day in iter_days(start, end):
            result = download_ticks_day(symbol, day, cache_dir=cache_dir, mt5_module=mt5_module, force=force)
            if verbose:
                source = "cached" if result["cached"] else "downloaded"
                print(f"Ticks {result['day']}: {result['ticks']} ({source})", flush=True)
    finally:
        if initialized_here and hasattr(mt5_module, "shutdown"):
            mt5_module.shutdown()

    frame, _ = load_ticks_range_from_cache(symbol, start, end, cache_dir=cache_dir)
    return frame
