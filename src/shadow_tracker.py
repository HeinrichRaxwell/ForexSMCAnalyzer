import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SHADOW_SIGNALS_FILE = os.path.join(BASE_DIR, "data", "shadow_signals.json")
DEFAULT_SHADOW_LABELED_DATA_FILE = os.path.join(BASE_DIR, "data", "shadow_labeled_setups.csv")

def get_current_account_login() -> int | None:
    try:
        import MetaTrader5 as mt5
        acc = mt5.account_info()
        if acc is not None:
            login = getattr(acc, "login", None)
            if login:
                return int(login)
    except Exception:
        pass
    return None

def get_shadow_signals_file() -> str:
    login = get_current_account_login()
    filename = f"shadow_signals_{login}.json" if login else "shadow_signals.json"
    return os.path.join(BASE_DIR, "data", filename)

def get_shadow_labeled_data_file() -> str:
    login = get_current_account_login()
    filename = f"shadow_labeled_setups_{login}.csv" if login else "shadow_labeled_setups.csv"
    return os.path.join(BASE_DIR, "data", filename)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(sep=" ")
        except TypeError:
            return value.isoformat()
    return value


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        print(f"[Shadow Tracker] Invalid {name}={raw_value!r}; using {default}.")
        return default


def should_shadow_signal(probability: float, accept_threshold: float, min_confidence: float = None) -> bool:
    """Return True for candidate signals below live threshold that should be tracked virtually."""
    try:
        prob = float(probability)
        threshold = float(accept_threshold)
    except (TypeError, ValueError):
        return False

    if not np.isfinite(prob) or not np.isfinite(threshold):
        return False

    if min_confidence is None:
        min_confidence = _read_float_env("ML_SHADOW_MIN_CONFIDENCE", 0.0)
    try:
        min_conf = float(min_confidence)
    except (TypeError, ValueError):
        min_conf = 0.0

    return min_conf <= prob < threshold


def load_shadow_signals(shadow_signals_file: str = None) -> dict:
    if shadow_signals_file is None:
        shadow_signals_file = get_shadow_signals_file()
    path = _resolve_path(shadow_signals_file)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_shadow_signals(shadow_signals: dict, shadow_signals_file: str = None):
    if shadow_signals_file is None:
        shadow_signals_file = get_shadow_signals_file()
    path = _resolve_path(shadow_signals_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_json_safe(shadow_signals), f, indent=4)


def _build_one_record(
    signal_id: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction_name: str,
    accept_threshold: float,
    opt: dict,
    probability: float,
    now: str,
    leg: str = "single",
    filtered_reason: str = "below_accept_threshold",
) -> dict:
    return _to_json_safe({
        "signal_id": signal_id,
        "source": "shadow",
        "status": "open",
        "result": None,
        "label": None,
        "symbol": symbol,
        "time": str(opt.get("time", "")),
        "timeframe": timeframe,
        "strategy": strategy,
        "direction": opt.get("direction"),
        "direction_name": direction_name,
        "leg": leg,
        "setup_index": opt.get("index"),
        "option_name": opt.get("option_name", ""),
        "entry_price": opt.get("entry_price"),
        "sl_price": opt.get("sl_price"),
        "tp_price": opt.get("tp_price"),
        "confidence": float(probability),
        "accept_threshold": float(accept_threshold),
        "filtered_reason": opt.get("filtered_reason", filtered_reason),
        "created_at": now,
        "latest_seen_at": now,
        "resolved_at": None,
        "ticket_id": None,
        "features": opt.get("features", {}),
    })


def build_shadow_signal_records(
    signal_id: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction_name: str,
    accept_threshold: float,
    opt: dict = None,
    probability: float = None,
    opt_a: dict = None,
    probability_a: float = None,
    opt_b: dict = None,
    probability_b: float = None,
    now: str = None,
    filtered_reason: str = "below_accept_threshold",
) -> list:
    """Build normalized shadow records for single or dual-entry candidate signals."""
    now = now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if opt is not None:
        return [
            _build_one_record(
                signal_id=signal_id,
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                direction_name=direction_name,
                accept_threshold=accept_threshold,
                opt=opt,
                probability=probability,
                now=now,
                filtered_reason=filtered_reason,
            )
        ]

    records = []
    if opt_a is not None:
        records.append(
            _build_one_record(
                signal_id=f"{signal_id}_0.5",
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                direction_name=direction_name,
                accept_threshold=accept_threshold,
                opt=opt_a,
                probability=probability_a,
                now=now,
                leg="0.5",
                filtered_reason=filtered_reason,
            )
        )
    if opt_b is not None:
        records.append(
            _build_one_record(
                signal_id=f"{signal_id}_0.618",
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                direction_name=direction_name,
                accept_threshold=accept_threshold,
                opt=opt_b,
                probability=probability_b,
                now=now,
                leg="0.618",
                filtered_reason=filtered_reason,
            )
        )
    return records


def upsert_shadow_signals(records: list, shadow_signals_file: str = None) -> bool:
    """Insert or refresh shadow records by signal_id without overwriting resolved outcomes."""
    if shadow_signals_file is None:
        shadow_signals_file = get_shadow_signals_file()
    shadow_signals = load_shadow_signals(shadow_signals_file)
    changed = False

    for record in records:
        signal_id = str(record["signal_id"])
        existing = shadow_signals.get(signal_id)
        safe_record = _to_json_safe(record)

        if existing is None:
            shadow_signals[signal_id] = safe_record
            changed = True
            continue

        merged = dict(existing)
        preserved_status = existing.get("status")
        preserved_result = existing.get("result")
        preserved_label = existing.get("label")
        preserved_resolved_at = existing.get("resolved_at")

        merged.update(safe_record)
        merged["created_at"] = existing.get("created_at", safe_record.get("created_at"))

        if preserved_status == "resolved":
            merged["status"] = preserved_status
            merged["result"] = preserved_result
            merged["label"] = preserved_label
            merged["resolved_at"] = preserved_resolved_at

        if merged != existing:
            shadow_signals[signal_id] = merged
            changed = True

    if changed:
        save_shadow_signals(shadow_signals, shadow_signals_file)
    return changed


def _time_to_string(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _candidate_candles_after_signal(record: dict, candles: pd.DataFrame) -> pd.DataFrame:
    if candles is None or candles.empty:
        return pd.DataFrame()
    if "time" not in candles.columns:
        return candles.copy()

    df = candles.copy()
    df["_time_parsed"] = pd.to_datetime(df["time"], errors="coerce")
    signal_time = pd.to_datetime(record.get("time"), errors="coerce")
    if pd.notna(signal_time):
        df = df[df["_time_parsed"] > signal_time]
    df = df.sort_values("_time_parsed", kind="mergesort")
    return df.drop(columns=["_time_parsed"])


def _hit_flags(record: dict, candle: dict) -> tuple:
    direction = int(record.get("direction", 0))
    entry = float(record.get("entry_price"))
    sl = float(record.get("sl_price"))
    tp = float(record.get("tp_price"))
    high = float(candle.get("High"))
    low = float(candle.get("Low"))

    if direction == 1:
        entry_hit = low <= entry
        sl_hit = low <= sl
        tp_hit = high >= tp
    elif direction == -1:
        entry_hit = high >= entry
        sl_hit = high >= sl
        tp_hit = low <= tp
    else:
        entry_hit = False
        sl_hit = False
        tp_hit = False
    return entry_hit, sl_hit, tp_hit


def _pnl_relative(record: dict, result: str) -> float:
    if result == "sl":
        return -1.0

    direction = int(record.get("direction", 0))
    entry = float(record.get("entry_price"))
    sl = float(record.get("sl_price"))
    tp = float(record.get("tp_price"))
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    reward = (tp - entry) if direction == 1 else (entry - tp)
    return round(reward / risk, 6)


def _mark_shadow_resolved(record: dict, result: str, resolved_at: str, now: str) -> dict:
    updated = dict(record)
    updated["status"] = "resolved"
    updated["result"] = result
    updated["label"] = 1 if result == "tp" else 0
    updated["pnl_relative"] = _pnl_relative(record, result)
    updated["resolved_at"] = resolved_at
    updated["latest_seen_at"] = now
    return _to_json_safe(updated)


def resolve_shadow_record(record: dict, candles: pd.DataFrame, max_bars: int = None, now: str = None) -> tuple:
    """Resolve one open shadow signal against future candles using conservative candle rules."""
    if record.get("status") in {"resolved", "expired"}:
        return record, False

    now = now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = _candidate_candles_after_signal(record, candles)
    if df.empty:
        return record, False

    triggered = bool(record.get("triggered_at"))
    triggered_at = record.get("triggered_at")
    inspected = 0

    for _, row in df.iterrows():
        inspected += 1
        candle = row.to_dict()
        candle_time = _time_to_string(candle.get("time", now))
        entry_hit, sl_hit, tp_hit = _hit_flags(record, candle)

        if not triggered:
            if not entry_hit:
                if max_bars is not None and inspected >= max_bars:
                    updated = dict(record)
                    updated["status"] = "expired"
                    updated["result"] = "expired"
                    updated["label"] = None
                    updated["resolved_at"] = candle_time
                    updated["latest_seen_at"] = now
                    return _to_json_safe(updated), True
                continue
            triggered = True
            triggered_at = candle_time

        if sl_hit and tp_hit:
            updated = _mark_shadow_resolved(record, "sl", candle_time, now)
            updated["triggered_at"] = triggered_at
            return updated, True
        if sl_hit:
            updated = _mark_shadow_resolved(record, "sl", candle_time, now)
            updated["triggered_at"] = triggered_at
            return updated, True
        if tp_hit:
            updated = _mark_shadow_resolved(record, "tp", candle_time, now)
            updated["triggered_at"] = triggered_at
            return updated, True

        if max_bars is not None and inspected >= max_bars:
            updated = dict(record)
            updated["status"] = "expired"
            updated["result"] = "expired"
            updated["label"] = None
            updated["triggered_at"] = triggered_at
            updated["resolved_at"] = candle_time
            updated["latest_seen_at"] = now
            return _to_json_safe(updated), True

    if triggered and triggered_at and record.get("triggered_at") != triggered_at:
        updated = dict(record)
        updated["triggered_at"] = triggered_at
        updated["latest_seen_at"] = now
        return _to_json_safe(updated), True
    return record, False


def _shadow_label_row(record: dict) -> dict:
    features = record.get("features") or {}
    entry = record.get("entry_price")
    sl = record.get("sl_price")
    tp = record.get("tp_price")
    try:
        risk = abs(float(entry) - float(sl))
        reward = abs(float(tp) - float(entry))
        derived_rr = reward / risk if risk > 0.0 else 0.0
    except (TypeError, ValueError):
        derived_rr = 0.0

    direction = features.get("direction", record.get("direction"))
    floop_trend = features.get("floop_trend", 0)
    try:
        htf_aligned = 1 if int(direction) == int(floop_trend) and int(floop_trend) != 0 else 0
    except (TypeError, ValueError):
        htf_aligned = 0

    row = {
        "signal_id": record.get("signal_id"),
        "sample_source": "shadow",
        "time": record.get("time"),
        "timeframe": features.get("timeframe", record.get("timeframe")),
        "strategy": record.get("strategy"),
        "hour": features.get("hour"),
        "day_of_week": features.get("day_of_week"),
        "setup_type": features.get("setup_type"),
        "direction": record.get("direction"),
        "entry_price": record.get("entry_price"),
        "sl_price": record.get("sl_price"),
        "tp_price": record.get("tp_price"),
        "risk_pips": features.get("risk_pips"),
        "atr_14": features.get("atr_14"),
        "trend": features.get("trend"),
        "relative_risk": features.get("relative_risk"),
        "killzone": features.get("killzone"),
        "fvg_width": features.get("fvg_width"),
        "relative_fvg_width": features.get("relative_fvg_width"),
        "near_psychological_level": features.get("near_psychological_level"),
        "knn_prob_sig": features.get("knn_prob_sig"),
        "knn_prob_opp": features.get("knn_prob_opp"),
        "dist_entry_to_poc": features.get("dist_entry_to_poc"),
        "dist_entry_to_nearest_poc": features.get("dist_entry_to_nearest_poc"),
        "dist_entry_to_pp": features.get("dist_entry_to_pp"),
        "dist_entry_to_nearest_pivot": features.get("dist_entry_to_nearest_pivot"),
        "floop_signal": features.get("floop_signal"),
        "floop_strength": features.get("floop_strength"),
        "floop_trend": features.get("floop_trend"),
        "floop_trend_aligned": features.get("floop_trend_aligned"),
        "rr_ratio": features.get("rr_ratio", derived_rr),
        "atr_percentile": features.get("atr_percentile", 0.0),
        "body_to_range_ratio": features.get("body_to_range_ratio", 0.0),
        "dist_to_recent_swing": features.get("dist_to_recent_swing", 0.0),
        "htf_trend_aligned": features.get("htf_trend_aligned", htf_aligned),
        "confluence_score": features.get("confluence_score", 0),
        "order_type": features.get("order_type", 0),
        "reaction_strength": features.get("reaction_strength", 0.0),
        "confidence": record.get("confidence"),
        "accept_threshold": record.get("accept_threshold"),
        "filtered_reason": record.get("filtered_reason"),
        "resolved_at": record.get("resolved_at"),
        "result": record.get("result"),
        "pnl_relative": record.get("pnl_relative"),
        "label": record.get("label"),
    }
    return _to_json_safe(row)


def _load_labeled_signal_ids(shadow_labeled_data_path: str) -> set:
    path = _resolve_path(shadow_labeled_data_path)
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, usecols=["signal_id"])
    except Exception:
        return set()
    return set(str(v) for v in df["signal_id"].dropna().tolist())


def append_shadow_labeled_rows(rows: list, shadow_labeled_data_path: str = DEFAULT_SHADOW_LABELED_DATA_FILE) -> int:
    if not rows:
        return 0

    path = _resolve_path(shadow_labeled_data_path)
    existing_ids = _load_labeled_signal_ids(path)
    new_rows = [row for row in rows if str(row.get("signal_id")) not in existing_ids]
    if not new_rows:
        return 0

    df_new = pd.DataFrame(new_rows)
    if os.path.exists(path):
        try:
            existing_columns = list(pd.read_csv(path, nrows=0).columns)
            if existing_columns:
                for col in existing_columns:
                    if col not in df_new.columns:
                        df_new[col] = np.nan
                df_new = df_new[existing_columns]
        except Exception:
            pass

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_new.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return len(df_new)


def process_shadow_signal_outcomes(
    candles_by_timeframe: dict,
    shadow_signals_file: str = None,
    shadow_labeled_data_path: str = None,
    max_bars: int = None,
    now: str = None,
) -> dict:
    if shadow_signals_file is None:
        shadow_signals_file = get_shadow_signals_file()
    if shadow_labeled_data_path is None:
        shadow_labeled_data_path = get_shadow_labeled_data_file()
    shadow_signals = load_shadow_signals(shadow_signals_file)
    if not shadow_signals:
        return {"resolved_count": 0, "expired_count": 0, "labeled_rows_appended": 0}

    now = now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated = False
    resolved_rows = []
    resolved_count = 0
    expired_count = 0

    for signal_id, record in list(shadow_signals.items()):
        if record.get("status") in {"resolved", "expired"}:
            continue
        timeframe = record.get("timeframe")
        candles = candles_by_timeframe.get(timeframe)
        resolved_record, changed = resolve_shadow_record(record, candles, max_bars=max_bars, now=now)
        if not changed:
            continue

        shadow_signals[signal_id] = resolved_record
        updated = True
        if resolved_record.get("status") == "resolved":
            resolved_count += 1
            if resolved_record.get("label") is not None:
                resolved_rows.append(_shadow_label_row(resolved_record))
        elif resolved_record.get("status") == "expired":
            expired_count += 1

    appended = append_shadow_labeled_rows(resolved_rows, shadow_labeled_data_path)
    if updated:
        save_shadow_signals(shadow_signals, shadow_signals_file)

    return {
        "resolved_count": resolved_count,
        "expired_count": expired_count,
        "labeled_rows_appended": appended,
    }
