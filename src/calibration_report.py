import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.model_trainer import (
    load_training_dataset,
    prepare_training_features,
    calculate_sample_weights,
    make_xgb_model,
    make_lgb_model,
)
from src.calibrator import fit_calibrator, save_calibrator
from src.live_trade_policy import normalize_strategy_name

from sklearn.model_selection import TimeSeriesSplit


DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
BUCKET_EDGES = [0.00, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
SETUP_TYPE_STRATEGY_FALLBACK = {
    0: "FVG_OR_BPR",
    1: "OB_OR_SWAPZONE_IC_SND",
    2: "PIVOT_REJECTION",
}
DEFAULT_LIVE_TIMEFRAMES = {"M30", "H1", "H4", "D1"}
DEFAULT_LIVE_BLOCKED_STRATEGIES = {"Pivot", "PIVOT_REJECTION", "SND", "Swapzone"}
TIMEFRAME_ALIASES = {
    "15": "M15",
    "15.0": "M15",
    "M15": "M15",
    "30": "M30",
    "30.0": "M30",
    "M30": "M30",
    "60": "H1",
    "60.0": "H1",
    "1H": "H1",
    "H1": "H1",
    "240": "H4",
    "240.0": "H4",
    "4H": "H4",
    "H4": "H4",
    "1440": "D1",
    "1440.0": "D1",
    "1D": "D1",
    "D1": "D1",
}
TIMEFRAME_SORT_ORDER = {"M15": 0, "M30": 1, "H1": 2, "H4": 3, "D1": 4}


def _resolve_project_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _load_env_values(path=".env") -> dict:
    env_path = _resolve_project_path(path)
    values = {}
    if not env_path or not os.path.exists(env_path):
        return values
    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            values[key.strip()] = raw_value.strip().strip("\"'")
    return values


def normalize_live_timeframe(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    normalized = str(value).strip().upper()
    return TIMEFRAME_ALIASES.get(normalized, normalized)


def _live_policy_env_value(env_values: dict, name: str, default: str | None = None):
    if name in env_values:
        return env_values[name]
    return os.getenv(name, default)


def _live_policy_allowed_timeframes(env_values: dict) -> set[str]:
    configured = _csv_set(_live_policy_env_value(env_values, "MT5_ALLOWED_TIMEFRAMES", None))
    if not configured:
        return set(DEFAULT_LIVE_TIMEFRAMES)
    return {tf for tf in (normalize_live_timeframe(item) for item in configured) if tf}


def _live_policy_blocked_strategies(env_values: dict) -> set[str]:
    configured = _csv_set(_live_policy_env_value(env_values, "MT5_LIVE_STRATEGY_BLOCKLIST", None))
    if not configured:
        return set(DEFAULT_LIVE_BLOCKED_STRATEGIES)
    blocked = set(configured)
    if "Pivot" in blocked:
        blocked.add("PIVOT_REJECTION")
    return blocked


def _live_policy_allowed_strategies(env_values: dict) -> set[str]:
    return _csv_set(_live_policy_env_value(env_values, "MT5_LIVE_STRATEGY_ALLOWLIST", None))


def _sort_timeframes(timeframes: set[str]) -> list[str]:
    return sorted(timeframes, key=lambda tf: (TIMEFRAME_SORT_ORDER.get(tf, 99), tf))


def assign_confidence_bucket(confidence) -> str:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "unscored"
    if not np.isfinite(value):
        return "unscored"

    value = max(0.0, min(value, 1.0))
    for idx in range(len(BUCKET_EDGES) - 1):
        lower = BUCKET_EDGES[idx]
        upper = BUCKET_EDGES[idx + 1]
        if lower <= value < upper or (upper == 1.0 and value <= upper):
            return f"{lower:.2f}-{upper:.2f}"
    return "unscored"


def _load_active_models(model_dir="models", model_path=None):
    if model_path is None:
        model_dir = _resolve_project_path(model_dir)
        model_path = os.path.join(model_dir, "smc_xgb_classifier.joblib")
    else:
        model_path = _resolve_project_path(model_path)
        model_dir = os.path.dirname(model_path)

    if not os.path.exists(model_path):
        return None, None

    model_xgb = joblib.load(model_path)
    lgb_path = os.path.join(model_dir, "smc_lgb_classifier.joblib")
    model_lgb = None
    if os.path.exists(lgb_path):
        try:
            model_lgb = joblib.load(lgb_path)
        except Exception as exc:
            print(f"[Calibration Warning] Failed to load LightGBM model: {exc}")
    return model_xgb, model_lgb


def _build_model_matrix(df: pd.DataFrame, model_xgb):
    X_all, _ = prepare_training_features(df)
    expected_features = list(getattr(model_xgb, "feature_names_in_", X_all.columns))
    for feature in expected_features:
        if feature not in X_all.columns:
            X_all[feature] = 0.0
    return X_all[expected_features]


def _predict_active_model_confidence(df: pd.DataFrame, model_dir="models", model_path=None):
    model_xgb, model_lgb = _load_active_models(model_dir=model_dir, model_path=model_path)
    if model_xgb is None:
        return None

    try:
        X = _build_model_matrix(df, model_xgb)
        probs_xgb = model_xgb.predict_proba(X)[:, 1]
        if model_lgb is None:
            return probs_xgb
        try:
            probs_lgb = model_lgb.predict_proba(X)[:, 1]
            return (probs_xgb + probs_lgb) / 2
        except Exception as exc:
            print(f"[Calibration Warning] Failed to score LightGBM model: {exc}")
            return probs_xgb
    except Exception as exc:
        print(f"[Calibration Warning] Failed to score active model: {exc}")
        return None


def _time_sorted_positions(df: pd.DataFrame) -> np.ndarray:
    """Return positional order (0..n-1) that sorts rows by time, NaT last, stable."""
    if "time" in df.columns:
        times = pd.to_datetime(df["time"], errors="coerce")
        # argsort with mergesort is stable; NaT sorts last under numpy datetime64
        return np.argsort(times.values, kind="mergesort")
    return np.arange(len(df))


def walk_forward_oof_confidence(df: pd.DataFrame, n_splits: int = 5, embargo: int = 0):
    """Out-of-fold confidence via expanding-window walk-forward validation.

    Each fold trains the same ensemble used in production (make_xgb_model +
    make_lgb_model, identical sample weights) on strictly EARLIER rows, then
    predicts the held-out later rows. No row is ever scored by a model that saw
    it during training, and no future information leaks backward. Rows in the
    initial training-only block (before the first test fold) stay NaN/unscored.

    This is the honest replacement for scoring the full dataset with the active
    model, which scored ~80% of rows in-sample and produced inflated winrates.
    """
    X_all, y_all = prepare_training_features(df)
    n = len(df)
    oof = np.full(n, np.nan, dtype=float)

    # Need enough rows for a meaningful time-series split.
    if n < (n_splits + 1) * 2:
        print(f"[Walk-Forward] Only {n} rows; too few for {n_splits}-fold OOF. Leaving unscored.")
        return pd.Series(oof, index=df.index)

    order = _time_sorted_positions(df)          # positional order over original rows
    X_sorted = X_all.iloc[order]
    y_sorted = y_all.iloc[order]
    sorted_index_labels = df.index[order]       # original index labels in time order

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=embargo)
    fold = 0
    for train_pos, test_pos in splitter.split(X_sorted):
        if len(train_pos) == 0 or len(test_pos) == 0:
            continue
        fold += 1
        X_tr = X_sorted.iloc[train_pos]
        y_tr = y_sorted.iloc[train_pos]
        train_index_labels = sorted_index_labels[train_pos]
        weights = calculate_sample_weights(df, train_index_labels)

        model_xgb = make_xgb_model()
        model_xgb.fit(X_tr, y_tr, sample_weight=weights)
        try:
            model_lgb = make_lgb_model()
            model_lgb.fit(X_tr, y_tr, sample_weight=weights)
        except Exception as exc:
            print(f"[Walk-Forward Warning] LightGBM fit failed on fold {fold}: {exc}")
            model_lgb = None

        X_te = X_sorted.iloc[test_pos]
        probs = model_xgb.predict_proba(X_te)[:, 1]
        if model_lgb is not None:
            try:
                probs = (probs + model_lgb.predict_proba(X_te)[:, 1]) / 2
            except Exception as exc:
                print(f"[Walk-Forward Warning] LightGBM predict failed on fold {fold}: {exc}")

        oof[order[test_pos]] = probs

    scored = int(np.isfinite(oof).sum())
    print(f"[Walk-Forward] Scored {scored}/{n} rows out-of-fold across {fold} folds "
          f"({n - scored} initial rows left unscored).")
    return pd.Series(oof, index=df.index)


def score_outcome_dataset(
    labeled_data_path="data/labeled_setups.csv",
    shadow_labeled_data_path=None,
    model_dir="models",
    model_path=None,
    mode="walk_forward",
    n_splits=5,
    embargo=0,
) -> pd.DataFrame:
    """Attach a `confidence` column to the labeled dataset for reporting.

    mode="walk_forward" (default): honest out-of-fold scoring. Each row is
        scored by a model trained only on earlier rows. Use this for any
        performance/winrate claim.
    mode="active_model": score every row with the saved champion model. This
        scores most rows IN-SAMPLE and inflates winrates; kept only for
        debugging / inspecting what the live model currently predicts.
    """
    df = load_training_dataset(
        labeled_data_path=labeled_data_path,
        shadow_labeled_data_path=shadow_labeled_data_path,
    )
    df = df.copy()

    if mode == "walk_forward":
        oof = walk_forward_oof_confidence(df, n_splits=n_splits, embargo=embargo)
        df["confidence"] = oof.values
        df["confidence_source"] = np.where(
            pd.Series(oof.values).notna(), "walk_forward_oof", "unscored"
        )
    elif mode == "active_model":
        active_model_probs = _predict_active_model_confidence(df, model_dir=model_dir, model_path=model_path)
        if active_model_probs is not None:
            df["confidence"] = active_model_probs
            df["confidence_source"] = "active_model"
        else:
            if "confidence" not in df.columns:
                df["confidence"] = np.nan
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
            df["confidence_source"] = np.where(df["confidence"].notna(), "stored", "missing")
    else:
        raise ValueError(f"Unknown scoring mode: {mode!r} (expected 'walk_forward' or 'active_model').")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    if "pnl_relative" not in df.columns:
        df["pnl_relative"] = np.where(df["label"] == 1, 1.0, -1.0)
    df["pnl_relative"] = pd.to_numeric(df["pnl_relative"], errors="coerce")
    df["confidence_bucket"] = df["confidence"].apply(assign_confidence_bucket)
    return df.reset_index(drop=True)


def fit_calibrator_from_scored(scored_df: pd.DataFrame, min_samples: int = 50):
    """Fit isotonic calibrator from honest out-of-fold scored rows only."""
    df = scored_df.copy()
    confidence = pd.to_numeric(df.get("confidence"), errors="coerce")
    labels = pd.to_numeric(df.get("label"), errors="coerce")
    mask = confidence.notna() & labels.isin([0, 1])
    if "confidence_source" in df.columns:
        mask &= df["confidence_source"].astype(str).eq("walk_forward_oof")
    if int(mask.sum()) < int(min_samples):
        return None
    return fit_calibrator(
        confidence[mask].to_numpy(),
        labels[mask].astype(int).to_numpy(),
        min_samples=min_samples,
    )


def _round_float(value, digits=2):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return round(float(value), digits)


def _max_consecutive_losses(labels) -> int:
    max_streak = 0
    current_streak = 0
    for label in labels:
        if int(label) == 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return int(max_streak)


def _metrics_for_group(group: pd.DataFrame) -> dict:
    sample_count = int(len(group))
    if sample_count == 0:
        return {
            "sample_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "winrate_pct": 0.0,
            "expectancy_r": 0.0,
            "max_drawdown_r": 0.0,
            "avg_confidence": None,
            "avg_win_r": None,
            "avg_loss_r": None,
            "profit_factor": None,
            "max_consecutive_losses": 0,
        }

    ordered_group = group.copy()
    if "time" in ordered_group.columns:
        ordered_group["_time_sort"] = pd.to_datetime(ordered_group["time"], errors="coerce")
        ordered_group = ordered_group.sort_values("_time_sort", kind="mergesort").drop(columns=["_time_sort"])

    labels = group["label"].astype(int)
    pnl = pd.to_numeric(group["pnl_relative"], errors="coerce")
    ordered_pnl = pd.to_numeric(ordered_group["pnl_relative"], errors="coerce").fillna(0.0)
    equity = ordered_pnl.cumsum()
    running_peak = equity.cummax().clip(lower=0.0)
    max_drawdown = (running_peak - equity).max()
    wins = int((labels == 1).sum())
    losses = int((labels == 0).sum())
    win_pnl = pnl[labels == 1]
    loss_pnl = pnl[labels == 0]
    gross_profit = win_pnl[win_pnl > 0].sum()
    gross_loss = abs(loss_pnl[loss_pnl < 0].sum())
    profit_factor = None if gross_loss == 0 else gross_profit / gross_loss
    ordered_labels = ordered_group["label"].astype(int).tolist()
    return {
        "sample_count": sample_count,
        "win_count": wins,
        "loss_count": losses,
        "winrate_pct": _round_float((wins / sample_count) * 100.0),
        "expectancy_r": _round_float(pnl.mean()),
        "max_drawdown_r": _round_float(max_drawdown),
        "avg_confidence": _round_float(pd.to_numeric(group["confidence"], errors="coerce").mean()),
        "avg_win_r": _round_float(win_pnl.mean()) if not win_pnl.empty else None,
        "avg_loss_r": _round_float(loss_pnl.mean()) if not loss_pnl.empty else None,
        "profit_factor": _round_float(profit_factor),
        "max_consecutive_losses": _max_consecutive_losses(ordered_labels),
    }


def _group_metrics(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {}
    result = {}
    for key, group in df.groupby(column, dropna=False):
        result[str(key)] = _metrics_for_group(group)
    return result


def _ensure_strategy_column(df: pd.DataFrame) -> pd.DataFrame:
    if "strategy" not in df.columns:
        df["strategy"] = np.nan

    fallback = pd.Series("UNKNOWN", index=df.index)
    if "setup_type" in df.columns:
        setup_type = pd.to_numeric(df["setup_type"], errors="coerce")
        fallback = setup_type.map(SETUP_TYPE_STRATEGY_FALLBACK).fillna("UNKNOWN")

    strategy = df["strategy"].replace("", np.nan)
    df["strategy"] = strategy.fillna(fallback)
    if "setup_type" in df.columns:
        setup_type = pd.to_numeric(df["setup_type"], errors="coerce")
        df.loc[setup_type == 2, "strategy"] = "PIVOT_REJECTION"
    return df


def _threshold_metrics(df: pd.DataFrame, thresholds=None) -> dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    result = {}
    scored_confidence = pd.to_numeric(df["confidence"], errors="coerce")
    for threshold in thresholds:
        passed = df[scored_confidence >= threshold]
        result[f"{threshold:.2f}"] = _metrics_for_group(passed)
    return result


def _build_live_policy_report(df: pd.DataFrame, thresholds=None, env_values=None) -> dict:
    env_values = env_values if env_values is not None else _load_env_values(".env")
    allowed_timeframes = _live_policy_allowed_timeframes(env_values)
    blocked_strategies = _live_policy_blocked_strategies(env_values)
    allowed_strategies = _live_policy_allowed_strategies(env_values)

    live_df = df.copy()
    live_df["_live_timeframe"] = live_df.get("timeframe", pd.Series(index=live_df.index, dtype=object)).apply(
        normalize_live_timeframe
    )
    live_df["_live_strategy"] = live_df.apply(
        lambda row: normalize_strategy_name(row.get("strategy"), {"setup_type": row.get("setup_type")}),
        axis=1
    )
    live_df["_raw_strategy"] = live_df["strategy"].astype(str)

    timeframe_mask = live_df["_live_timeframe"].isin(allowed_timeframes)
    strategy_mask = ~(live_df["_live_strategy"].isin(blocked_strategies) | live_df["_raw_strategy"].isin(blocked_strategies))
    if allowed_strategies:
        strategy_mask &= (live_df["_live_strategy"].isin(allowed_strategies) | live_df["_raw_strategy"].isin(allowed_strategies))
    live_mask = timeframe_mask & strategy_mask
    filtered = live_df[live_mask].drop(columns=["_live_timeframe", "_live_strategy", "_raw_strategy"])

    excluded_timeframe = int((~timeframe_mask).sum())
    excluded_strategy = int((timeframe_mask & ~strategy_mask).sum())

    return {
        "allowed_timeframes": _sort_timeframes(allowed_timeframes),
        "blocked_strategies": sorted(blocked_strategies),
        "allowed_strategies": sorted(allowed_strategies),
        "excluded_counts": {
            "timeframe": excluded_timeframe,
            "strategy": excluded_strategy,
            "total": int((~live_mask).sum()),
        },
        "overall": _metrics_for_group(filtered),
        "thresholds": _threshold_metrics(filtered, thresholds=thresholds),
        "timeframes": _group_metrics(filtered, "timeframe"),
        "strategies": _group_metrics(filtered, "strategy"),
        "sources": _group_metrics(filtered, "sample_source"),
    }


def recommend_threshold(
    report: dict,
    min_samples: int = 50,
    min_expectancy_r: float = 1.0,
    max_drawdown_r: float = 3.0,
) -> dict:
    thresholds = report.get("thresholds", {})
    if not thresholds:
        return {
            "threshold": None,
            "sample_count": 0,
            "expectancy_r": None,
            "max_drawdown_r": None,
            "reason": "no_threshold_metrics",
        }

    sorted_items = sorted(thresholds.items(), key=lambda item: float(item[0]))
    for threshold, metrics in sorted_items:
        sample_count = int(metrics.get("sample_count", 0) or 0)
        expectancy = metrics.get("expectancy_r")
        max_drawdown = metrics.get("max_drawdown_r")
        if expectancy is None or max_drawdown is None:
            continue
        if (
            sample_count >= min_samples
            and float(expectancy) >= min_expectancy_r
            and float(max_drawdown) <= max_drawdown_r
        ):
            return {
                "threshold": threshold,
                "sample_count": sample_count,
                "expectancy_r": expectancy,
                "max_drawdown_r": max_drawdown,
                "reason": "lowest_threshold_meeting_rules",
            }

    fallback_threshold, fallback_metrics = sorted_items[-1]
    return {
        "threshold": fallback_threshold,
        "sample_count": int(fallback_metrics.get("sample_count", 0) or 0),
        "expectancy_r": fallback_metrics.get("expectancy_r"),
        "max_drawdown_r": fallback_metrics.get("max_drawdown_r"),
        "reason": "no_threshold_met_all_rules",
    }


def build_calibration_report(
    scored_df: pd.DataFrame,
    output_path="data/calibration_report.json",
    thresholds=None,
    scoring_mode="walk_forward",
    live_policy_env=None,
) -> dict:
    df = scored_df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    if "pnl_relative" not in df.columns:
        df["pnl_relative"] = np.where(df["label"] == 1, 1.0, -1.0)
    df["pnl_relative"] = pd.to_numeric(df["pnl_relative"], errors="coerce")
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    if "confidence_bucket" not in df.columns:
        df["confidence_bucket"] = df["confidence"].apply(assign_confidence_bucket)
    if "sample_source" not in df.columns:
        df["sample_source"] = "real"
    df = _ensure_strategy_column(df)

    scored_mask = pd.to_numeric(df["confidence"], errors="coerce").notna()
    meta = {
        "scoring_mode": scoring_mode,
        "is_out_of_sample": scoring_mode == "walk_forward",
        "total_rows": int(len(df)),
        "scored_rows": int(scored_mask.sum()),
        "unscored_rows": int((~scored_mask).sum()),
        "warning": (
            None
            if scoring_mode == "walk_forward"
            else "IN-SAMPLE scoring: ~80% of rows were seen by the model during "
            "training. Winrates here are inflated and must NOT be used to set a "
            "live threshold. Re-run with mode='walk_forward' for honest numbers."
        ),
    }

    report = {
        "meta": meta,
        "overall": _metrics_for_group(df),
        "thresholds": _threshold_metrics(df, thresholds=thresholds),
        "buckets": _group_metrics(df, "confidence_bucket"),
        "timeframes": _group_metrics(df, "timeframe"),
        "hours": _group_metrics(df, "hour"),
        "killzones": _group_metrics(df, "killzone"),
        "setup_types": _group_metrics(df, "setup_type"),
        "strategies": _group_metrics(df, "strategy"),
        "directions": _group_metrics(df, "direction"),
        "sources": _group_metrics(df, "sample_source"),
    }
    report["live_policy"] = _build_live_policy_report(df, thresholds=thresholds, env_values=live_policy_env)
    report["recommendation"] = recommend_threshold(report)

    if output_path:
        output_path = _resolve_project_path(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate ML calibration/performance report.")
    parser.add_argument("--labeled-data", default="data/labeled_setups.csv")
    parser.add_argument("--shadow-labeled-data", default=None)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default="data/calibration_report.json")
    parser.add_argument(
        "--mode",
        choices=["walk_forward", "active_model"],
        default="walk_forward",
        help="walk_forward (default): honest out-of-fold scoring. "
             "active_model: scores most rows IN-SAMPLE; inflated winrates, debug only.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=0,
                        help="Rows to skip between train and test fold (time-series gap).")
    args = parser.parse_args()

    if args.mode == "active_model":
        print("[Calibration] WARNING: mode=active_model scores ~80% of rows IN-SAMPLE. "
              "Winrates are inflated and must NOT be used for go-live decisions.")

    scored = score_outcome_dataset(
        labeled_data_path=args.labeled_data,
        shadow_labeled_data_path=args.shadow_labeled_data,
        model_dir=args.model_dir,
        model_path=args.model_path,
        mode=args.mode,
        n_splits=args.n_splits,
        embargo=args.embargo,
    )
    if args.mode == "walk_forward":
        calibrator = fit_calibrator_from_scored(scored)
        calibrator_path = os.path.join(_resolve_project_path(args.model_dir), "confidence_calibrator.joblib")
        if calibrator is not None:
            save_calibrator(calibrator, calibrator_path)
            print(f"[Calibration] Isotonic calibrator saved to {calibrator_path}")
        else:
            print("[Calibration] Not enough walk-forward OOF rows to fit calibrator; skipped.")
    report = build_calibration_report(scored, output_path=args.output, scoring_mode=args.mode)
    print(f"[Calibration] Report written to {_resolve_project_path(args.output)}")
    print(f"[Calibration] scoring_mode={args.mode}")
    print(json.dumps(report["overall"], indent=4))


if __name__ == "__main__":
    main()
