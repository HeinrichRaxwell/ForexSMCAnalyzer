from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "smc_xgb_classifier.joblib"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "real_tick_backtest_results.csv"
DEFAULT_TICK_CACHE_DIR = BASE_DIR / "data" / "ticks"

DEFAULT_TIMEFRAMES = ["M15", "M30", "H1", "H4", "D1"]
DEFAULT_STRATEGIES = ["FVG", "OB", "BB", "Swapzone", "BPR", "IC", "COMBINED"]
TIMEFRAME_ALIASES = {
    "M15": "15",
    "M30": "30",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in parse_csv_strings(value)]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in parse_csv_strings(value)]


def filter_setups_for_run(setups: list[dict], strategy: str, threshold: float) -> list[dict]:
    filtered = []
    for setup in setups:
        if strategy != "COMBINED" and setup.get("strategy") != strategy:
            continue
        if float(setup.get("probability", 0.5)) < float(threshold):
            continue
        filtered.append(setup)
    return filtered


def _date_strings(values) -> set[str]:
    if values is None:
        return set()
    dates = pd.to_datetime(values, errors="coerce")
    return {ts.date().isoformat() for ts in dates.dropna()}


def calculate_tick_coverage(
    ticks: pd.DataFrame,
    missing_cache_days: list[str] | None,
    candle_times,
    start=None,
    end=None,
) -> dict:
    required_days = _date_strings(candle_times)
    tick_days = _date_strings(ticks["time"]) if ticks is not None and not ticks.empty and "time" in ticks.columns else set()
    missing_required_days = sorted(required_days - tick_days)
    with_ticks = len(required_days & tick_days)
    required = len(required_days)
    coverage_pct = (with_ticks / required * 100.0) if required else 0.0
    missing_cache = sorted(set(missing_cache_days or []))

    requested_days = 0
    if start is not None and end is not None:
        start_day = pd.Timestamp(start).normalize()
        end_day = pd.Timestamp(end).normalize()
        requested_days = int((end_day - start_day).days) + 1

    return {
        "tick_days_requested": requested_days,
        "tick_days_required": required,
        "tick_days_with_ticks": with_ticks,
        "tick_days_missing": len(missing_required_days),
        "tick_missing_days": ";".join(missing_required_days),
        "tick_missing_cache_files": ";".join(missing_cache),
        "tick_coverage_pct": round(coverage_pct, 6),
        "is_real_tick_complete": required > 0 and len(missing_required_days) == 0,
    }


def prepare_smc_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    import numpy as np

    from src.smc_detector import (
        detect_bpr,
        detect_fvg_and_ob,
        detect_indecision_candles,
        detect_snr_and_swapzones,
        detect_structures,
        detect_swing_points,
    )

    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)

    frame = detect_swing_points(frame, window=5)
    frame = detect_structures(frame)
    frame = detect_fvg_and_ob(frame, symbol=symbol)
    frame = detect_snr_and_swapzones(frame, symbol=symbol)
    frame = detect_bpr(frame, symbol=symbol)
    frame = detect_indecision_candles(frame, symbol=symbol)

    close_prev = frame["Close"].shift(1).fillna(frame["Open"])
    true_range = np.maximum(
        frame["High"] - frame["Low"],
        np.maximum(abs(frame["High"] - close_prev), abs(frame["Low"] - close_prev)),
    )
    frame["ATR_14"] = true_range.rolling(window=14, min_periods=1).mean()
    frame["ATR_14"] = frame["ATR_14"].ffill().bfill().fillna(1.0)
    return frame


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        return None

    import joblib

    return joblib.load(model_path)


def score_setups_with_model(setups: list[dict], model) -> list[dict]:
    if not setups:
        return setups

    if model is None or not hasattr(model, "predict_proba") or not hasattr(model, "feature_names_in_"):
        for setup in setups:
            setup["probability"] = float(setup.get("probability", 0.5))
        return setups

    from src.backtester import build_model_feature_frame

    expected_features = list(model.feature_names_in_)
    features = [setup.get("features", {}) for setup in setups]
    feature_frame = build_model_feature_frame(features, expected_features)
    probabilities = model.predict_proba(feature_frame)[:, 1]
    for setup, probability in zip(setups, probabilities):
        setup["probability"] = float(probability)
    return setups


def build_scored_setups(df: pd.DataFrame, symbol: str, sizing: str, model) -> list[dict]:
    from src.backtester import generate_all_setups

    if sizing == "weighted":
        setups = generate_all_setups(df, symbol=symbol, lot_size_05=0.01, lot_size_0618=0.02)
    else:
        setups = generate_all_setups(df, symbol=symbol, lot_size_05=0.01, lot_size_0618=0.01)
    return score_setups_with_model(setups, model)


def trim_candles_to_range(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = frame[(frame["time"] >= start_ts) & (frame["time"] < end_ts)]
    return frame.sort_values("time", kind="mergesort").reset_index(drop=True)


def fetch_candles_for_timeframe(symbol: str, timeframe_name: str, start: datetime, end: datetime) -> pd.DataFrame:
    from src.data_collector import get_mt5_timeframe
    from src.data_loader import fetch_historical_data_range

    normalized = timeframe_name.upper()
    if normalized not in TIMEFRAME_ALIASES:
        raise ValueError(f"Unsupported timeframe: {timeframe_name}")

    timeframe = get_mt5_timeframe(TIMEFRAME_ALIASES[normalized])
    candles = fetch_historical_data_range(symbol, timeframe, start, end)
    return trim_candles_to_range(candles, start, end)


def build_timeframe_setups(
    symbol: str,
    timeframes: list[str],
    start: datetime,
    end: datetime,
    sizing_configs: list[str],
    model,
) -> dict[tuple[str, str], dict]:
    prepared = {}
    for timeframe in timeframes:
        print(f"Fetching candles: {symbol} {timeframe} {start} -> {end}", flush=True)
        candles = fetch_candles_for_timeframe(symbol, timeframe, start, end)
        if candles.empty:
            print(f"No candles returned for {timeframe}; skipping.", flush=True)
            continue
        print(f"Detecting setups: {timeframe} candles={len(candles)}", flush=True)
        detected = prepare_smc_frame(candles, symbol)
        for sizing in sizing_configs:
            print(f"Building setups: {timeframe} sizing={sizing}", flush=True)
            prepared[(timeframe.upper(), sizing)] = {
                "candles": detected,
                "setups": build_scored_setups(detected, symbol, sizing, model),
            }
            print(
                f"Prepared setups: {timeframe} sizing={sizing} count={len(prepared[(timeframe.upper(), sizing)]['setups'])}",
                flush=True,
            )
    return prepared


def run_real_tick_backtest(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframes: list[str],
    strategies: list[str],
    thresholds: list[float],
    capitals: list[float],
    max_concurrencies: list[int],
    sizing_configs: list[str],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    tick_cache_dir: str | Path = DEFAULT_TICK_CACHE_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    download_ticks: bool = False,
    force_ticks: bool = False,
) -> pd.DataFrame:
    from src.tick_backtester import run_tick_simulation
    from src.tick_data import download_ticks_range, load_ticks_range_from_cache, resolve_active_symbol

    import MetaTrader5 as mt5

    start = pd.Timestamp(start).to_pydatetime()
    end = pd.Timestamp(end).to_pydatetime()
    timeframes = [tf.upper() for tf in timeframes]
    sizing_configs = [sizing.lower() for sizing in sizing_configs]
    output_path = Path(output_path)
    tick_cache_dir = Path(tick_cache_dir)

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        active_symbol = resolve_active_symbol(symbol, mt5_module=mt5)
        print(f"Resolved active symbol: {active_symbol}", flush=True)
        model = load_model(model_path)
        if model is None:
            print("ML model not found; using neutral 0.50 probabilities.", flush=True)
        else:
            print(f"Loaded ML model: {model_path}", flush=True)
        timeframe_setups = build_timeframe_setups(
            symbol=active_symbol,
            timeframes=timeframes,
            start=start,
            end=end,
            sizing_configs=sizing_configs,
            model=model,
        )
    finally:
        mt5.shutdown()

    if download_ticks:
        print(f"Downloading/loading ticks for {active_symbol} from {start} to {end}...", flush=True)
        ticks = download_ticks_range(
            active_symbol,
            start,
            end,
            cache_dir=tick_cache_dir,
            mt5_module=mt5,
            force=force_ticks,
            verbose=True,
        )
        missing_cache_days = []
    else:
        print(f"Loading ticks from cache for {active_symbol} from {start} to {end}...", flush=True)
        ticks, missing_cache_days = load_ticks_range_from_cache(
            active_symbol,
            start,
            end,
            cache_dir=tick_cache_dir,
        )

    results = []
    for (timeframe, sizing), payload in timeframe_setups.items():
        candles = payload["candles"]
        all_setups = payload["setups"]
        print(f"Simulating {timeframe} {sizing}: {len(all_setups)} scored setups...", flush=True)
        coverage = calculate_tick_coverage(
            ticks=ticks,
            missing_cache_days=missing_cache_days,
            candle_times=candles["time"],
            start=start,
            end=end,
        )
        for capital in capitals:
            for strategy in strategies:
                for max_concurrent in max_concurrencies:
                    for threshold in thresholds:
                        filtered = filter_setups_for_run(all_setups, strategy, threshold)
                        simulation = run_tick_simulation(
                            ticks=ticks,
                            setups=filtered,
                            starting_capital=capital,
                            contract_size=100.0,
                            max_concurrent=max_concurrent,
                        )
                        results.append({
                            "symbol": active_symbol,
                            "requested_symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "sizing_config": sizing,
                            "capital": float(capital),
                            "ml_threshold": float(threshold),
                            "max_concurrent": int(max_concurrent),
                            "setup_count": len(filtered),
                            "total_resolved": simulation["total_resolved"],
                            "wins": simulation["wins"],
                            "losses": simulation["losses"],
                            "missed": simulation["missed"],
                            "winrate": simulation["winrate"],
                            "final_balance": simulation["final_balance"],
                            "max_dd_usd": simulation["max_drawdown_usd"],
                            "max_dd_pct": simulation["max_drawdown_pct"],
                            "blown": simulation["blown"],
                            **coverage,
                        })

    results_frame = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(output_path, index=False)
    return results_frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MT5 bid/ask real-tick SMC backtest matrix.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--start", default=None, help="Optional start datetime/date, overrides --days.")
    parser.add_argument("--end", default=None, help="Optional end datetime/date, default is now.")
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--thresholds", default="0.50")
    parser.add_argument("--capitals", default="50,100")
    parser.add_argument("--max-concurrent", default="3")
    parser.add_argument("--sizing", default="equal,weighted")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--tick-cache-dir", default=str(DEFAULT_TICK_CACHE_DIR))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--download-ticks", action="store_true")
    parser.add_argument("--force-ticks", action="store_true")
    return parser


def parse_date_range(args) -> tuple[datetime, datetime]:
    end = pd.Timestamp(args.end).to_pydatetime() if args.end else datetime.now()
    start = pd.Timestamp(args.start).to_pydatetime() if args.start else end - timedelta(days=int(args.days))
    return start, end


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    start, end = parse_date_range(args)
    results = run_real_tick_backtest(
        symbol=args.symbol,
        start=start,
        end=end,
        timeframes=parse_csv_strings(args.timeframes),
        strategies=parse_csv_strings(args.strategies),
        thresholds=parse_csv_floats(args.thresholds),
        capitals=parse_csv_floats(args.capitals),
        max_concurrencies=parse_csv_ints(args.max_concurrent),
        sizing_configs=parse_csv_strings(args.sizing),
        output_path=args.output,
        tick_cache_dir=args.tick_cache_dir,
        model_path=args.model_path,
        download_ticks=args.download_ticks,
        force_ticks=args.force_ticks,
    )
    print(f"Saved real-tick backtest results to: {args.output}")
    print(f"Rows: {len(results)}")
    if not results.empty:
        complete = bool(results["is_real_tick_complete"].all())
        min_coverage = float(results["tick_coverage_pct"].min())
        print(f"Tick coverage complete: {complete} (min coverage {min_coverage:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
