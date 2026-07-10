import os
import sys
from datetime import datetime
import hashlib
import json

import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv

load_dotenv()

# Add project root to python path if not present
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.calibrator import apply_calibrator, load_calibrator


_CALIBRATOR_CACHE = {}


def _resolve_project_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def apply_confidence_calibration(raw_prob, calibrator_path="models/confidence_calibrator.joblib"):
    """Map raw ensemble probability to calibrated confidence, with identity fallback."""
    calibrator_path = _resolve_project_path(calibrator_path)
    if calibrator_path not in _CALIBRATOR_CACHE:
        _CALIBRATOR_CACHE[calibrator_path] = load_calibrator(calibrator_path)
    calibrated = apply_calibrator(_CALIBRATOR_CACHE[calibrator_path], np.array([float(raw_prob)]))
    return float(calibrated[0])


def _build_inference_matrix(features, expected_features):
    """Build numeric inference matrix matching trained model feature order.

    Missing model features are filled with zero so older live signal payloads remain
    inferable after new feature columns are added during retraining.
    """
    df = pd.DataFrame(features)
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0.0
    X = df[list(expected_features)].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return X


def predict_setup_probability(features_dict, model_path="models/smc_xgb_classifier.joblib", calibrator_path=None):
    """
    Load the trained models (XGBoost & LightGBM) and predict the ensemble probability of success.
    
    Args:
        features_dict (dict or list of dicts): Setup features dictionary or list of dictionaries.
        model_path (str): Path to the trained joblib model.
        
    Returns:
        float or list of floats: Probability of success (label = 1).
    """
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE_DIR, model_path)
    if calibrator_path is None:
        calibrator_path = os.path.join(os.path.dirname(model_path), "confidence_calibrator.joblib")
        
    lgb_model_path = os.path.join(os.path.dirname(model_path), "smc_lgb_classifier.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model_xgb = joblib.load(model_path)
    
    model_lgb = None
    if os.path.exists(lgb_model_path):
        try:
            model_lgb = joblib.load(lgb_model_path)
        except Exception as e:
            print(f"[Inference Warning] Failed to load LightGBM model: {e}")
            
    is_list = isinstance(features_dict, list)
    if not is_list:
        features_dict = [features_dict]
        
    expected_features = list(model_xgb.feature_names_in_)
    X = _build_inference_matrix(features_dict, expected_features)
    
    # Predict probabilities
    probs_xgb = model_xgb.predict_proba(X)[:, 1]
    
    if model_lgb is not None:
        try:
            probs_lgb = model_lgb.predict_proba(X)[:, 1]
            probs = (probs_xgb + probs_lgb) / 2
        except Exception as e:
            print(f"[Inference Warning] Error during LightGBM prediction: {e}")
            probs = probs_xgb
    else:
        probs = probs_xgb

    probs = np.array([apply_confidence_calibration(prob, calibrator_path=calibrator_path) for prob in probs])
        
    if is_list:
        return probs.tolist()
    else:
        return float(probs[0])

def update_feedback_data(new_trades_list, labeled_data_path="data/labeled_setups.csv"):
    """
    Append new trades (with their features and actual outcomes) to the labeled dataset CSV.
    
    Args:
        new_trades_list (dict or list of dicts): A single trade dict or list of trade dicts.
        labeled_data_path (str): Path to the labeled CSV dataset.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(labeled_data_path):
        labeled_data_path = os.path.join(base_dir, labeled_data_path)
        
    if isinstance(new_trades_list, dict):
        new_trades_list = [new_trades_list]
        
    df_new = pd.DataFrame(new_trades_list)
    _fill_legacy_feedback_features(df_new)
    audit_columns = [
        'close_price',
        'close_reason',
        'net_profit',
        'manager_exit_trigger',
        'manager_exit_timeframe',
        'manager_exit_detail',
        'manager_exit_recorded_at',
    ]
    
    # Standard columns order for labeled_setups.csv
    columns_order = [
        'time', 'timeframe', 'hour', 'day_of_week', 'setup_type', 'direction', 
        'entry_price', 'sl_price', 'tp_price', 'risk_pips', 'atr_14', 
        'trend', 'relative_risk', 'killzone', 'fvg_width', 'relative_fvg_width',
        'near_psychological_level', 'knn_prob_sig', 'knn_prob_opp', 
        'dist_entry_to_poc', 'dist_entry_to_nearest_poc', 
        'dist_entry_to_pp', 'dist_entry_to_nearest_pivot',
        'floop_signal', 'floop_strength', 'floop_trend', 'floop_trend_aligned', 'pnl_relative', 'label'
    ]
    
    if os.path.exists(labeled_data_path):
        try:
            existing_columns = list(pd.read_csv(labeled_data_path, nrows=0).columns)
            if existing_columns:
                columns_order = existing_columns
                missing_audit_columns = [
                    column for column in audit_columns
                    if column in df_new.columns and column not in columns_order
                ]
                if missing_audit_columns:
                    existing_df = pd.read_csv(labeled_data_path)
                    for column in missing_audit_columns:
                        existing_df[column] = np.nan
                    columns_order = columns_order + missing_audit_columns
                    existing_df = existing_df[columns_order]
                    existing_df.to_csv(labeled_data_path, index=False)
        except Exception as e:
            print(f"[Feedback Loop] Warning: failed to read existing feedback CSV header: {e}")
    else:
        columns_order = columns_order + [
            column for column in audit_columns
            if column in df_new.columns and column not in columns_order
        ]

    # Ensure all columns exist
    for col in columns_order:
        if col not in df_new.columns:
            df_new[col] = np.nan
            
    df_new = df_new[columns_order]
    
    header = not os.path.exists(labeled_data_path)
    os.makedirs(os.path.dirname(labeled_data_path), exist_ok=True)
    
    df_new.to_csv(labeled_data_path, mode='a', header=header, index=False)
    print(f"Appended {len(df_new)} new trades to {labeled_data_path}")


def _safe_series_number(df, column, default=0.0):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors='coerce').fillna(default)


def _fill_legacy_feedback_features(df):
    """Populate new model feature columns for older live-feedback payloads."""
    entry = _safe_series_number(df, 'entry_price')
    sl = _safe_series_number(df, 'sl_price')
    tp = _safe_series_number(df, 'tp_price')
    risk = (entry - sl).abs()
    reward = (tp - entry).abs()
    rr = pd.Series(np.where(risk > 0.0, reward / risk, 0.0), index=df.index, dtype=float)
    floop_trend = _safe_series_number(df, 'floop_trend').astype(int)

    defaults = {
        'rr_ratio': rr,
        'atr_percentile': 0.0,
        'body_to_range_ratio': 0.0,
        'dist_to_recent_swing': 0.0,
        'htf_trend_aligned': (
            (_safe_series_number(df, 'direction').astype(int) == floop_trend)
            & (floop_trend != 0)
        ).astype(int),
        'confluence_score': 0,
        'order_type': 0,
        'reaction_strength': 0.0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
        elif isinstance(default, pd.Series):
            df[column] = df[column].where(df[column].notna(), default)
        else:
            df[column] = df[column].fillna(default)


def _feedback_key_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _feedback_key_number(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value).strip()


def _feedback_row_key(row: dict) -> tuple:
    """Build a stable key for one setup outcome so backfills do not double-append."""
    return (
        _feedback_key_text(row.get('time', '')),
        _feedback_key_number(row.get('timeframe', '')),
        _feedback_key_number(row.get('direction', '')),
        _feedback_key_number(row.get('entry_price', '')),
        _feedback_key_number(row.get('sl_price', '')),
        _feedback_key_number(row.get('tp_price', '')),
    )


def _load_existing_feedback_keys(labeled_data_path: str) -> set:
    if not os.path.exists(labeled_data_path):
        return set()
    try:
        df_existing = pd.read_csv(labeled_data_path)
    except Exception:
        return set()
    required_cols = {'time', 'timeframe', 'direction', 'entry_price', 'sl_price', 'tp_price'}
    if not required_cols.issubset(df_existing.columns):
        return set()
    return {_feedback_row_key(row) for row in df_existing.to_dict('records')}

def trigger_auto_retrain():
    """
    Retrain the XGBoost model with the updated dataset.
    """
    from src.model_trainer import train_xgboost_filter
    print("Triggering automatic model retraining...")
    return train_xgboost_filter()

def find_matching_manual_deal(symbol: str, direction: int, entry_price: float, sig_time_str: str) -> int:
    """Search MT5 deal history for a manual entry matching this signal."""
    # DEACTIVATED: Do not learn from manual trades to avoid importing human emotions/panic into AI.
    return None
    
    try:
        sig_time = datetime.strptime(sig_time_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        sig_time = datetime.now() - timedelta(hours=2)
        
    from_date = sig_time - timedelta(hours=24)
    to_date = sig_time + timedelta(hours=24)
    
    deals = mt5.history_deals_get(from_date, to_date)
    if deals is None or len(deals) == 0:
        return None
        
    for d in deals:
        # Check if it is an entry deal (DEAL_ENTRY_IN) and same symbol
        if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_IN:
            # Check direction (0 = Buy, 1 = Sell in MT5 deal type)
            d_dir = 1 if d.type == mt5.DEAL_TYPE_BUY else -1
            if d_dir == direction:
                # Check price proximity (within 1.5 USD for Gold)
                if abs(d.price - entry_price) <= 1.5:
                    print(f"[Feedback Loop] Found matching manual entry deal: Ticket #{d.position_id} @ {d.price:.3f}")
                    return d.position_id
    return None


def _get_retrain_threshold(default: int = 1) -> int:
    """Read the auto-retrain threshold from env, falling back to one closed trade."""
    raw_value = os.getenv("ML_RETRAIN_THRESHOLD", str(default))
    try:
        threshold = int(raw_value)
    except (TypeError, ValueError):
        print(f"[Feedback Loop] Invalid ML_RETRAIN_THRESHOLD={raw_value!r}; using {default}.")
        threshold = default
    return max(1, threshold)


def _is_weekend_retrain_enabled(default: bool = True) -> bool:
    raw_value = os.getenv("ML_RETRAIN_ON_WEEKEND")
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _retrain_alert_event_key(stats: dict, max_setups: int) -> str:
    """Build a stable key so the same MLOps result is not announced repeatedly."""
    def metric(value):
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return value

    payload = {
        "status": stats.get("status"),
        "dataset_size": stats.get("dataset_size"),
        "real_dataset_size": stats.get("real_dataset_size"),
        "shadow_dataset_size": stats.get("shadow_dataset_size"),
        "test_size": stats.get("test_size"),
        "eval_threshold": metric(stats.get("eval_threshold")),
        "old_accuracy": metric(stats.get("old_accuracy")),
        "new_accuracy": metric(stats.get("new_accuracy")),
        "old_winrate": metric(stats.get("old_winrate")),
        "new_winrate": metric(stats.get("new_winrate")),
        "old_passed_count": stats.get("old_passed_count"),
        "new_passed_count": stats.get("new_passed_count"),
        "max_setups": max_setups,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _should_send_retrain_alert(status: dict, stats: dict, max_setups: int, now: datetime = None) -> tuple:
    """Return whether this retrain alert should be sent and metadata to persist."""
    now = now or datetime.now()
    event_key = _retrain_alert_event_key(stats, max_setups)
    last_key = status.get("last_retrain_telegram_event_key")
    if last_key == event_key:
        return False, {
            "last_retrain_telegram_event_key": event_key,
            "last_retrain_telegram_suppressed_at": now.strftime('%Y-%m-%d %H:%M:%S'),
        }
    return True, {
        "last_retrain_telegram_event_key": event_key,
        "last_retrain_telegram_sent_at": now.strftime('%Y-%m-%d %H:%M:%S'),
    }


def check_and_trigger_retraining(new_trades_count: int, status_file: str = None):
    """
    Check if we should retrain the model.
    Retrain when accumulated trades reach ML_RETRAIN_THRESHOLD (default: 1)
    or when it is the weekend.
    """
    from datetime import datetime
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if status_file is None:
        status_file = os.path.join(base_dir, "data", "learning_status.json")
    
    status = {"new_trades_since_last_train": 0, "last_train_time": ""}
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                status = json.load(f)
        except Exception:
            pass
            
    status["new_trades_since_last_train"] = status.get("new_trades_since_last_train", 0) + new_trades_count
    
    weekend_enabled = _is_weekend_retrain_enabled(default=True)
    is_weekend = weekend_enabled and datetime.now().weekday() in [5, 6]
    accumulated = status["new_trades_since_last_train"]
    threshold = _get_retrain_threshold(default=1)
    
    should_retrain = is_weekend or accumulated >= threshold
    result = {
        "new_trades_count": new_trades_count,
        "new_trades_since_last_train": accumulated,
        "threshold": threshold,
        "is_weekend": is_weekend,
        "weekend_enabled": weekend_enabled,
        "should_retrain": should_retrain,
        "retrained": False,
        "status": "DEFERRED",
        "stats": None,
        "error": None,
    }
    
    print(f"[Feedback Loop] Accumulated {accumulated} new trades since last retrain. (Weekend: {is_weekend})")
    
    if should_retrain:
        print(f"[Feedback Loop] Retraining conditions met (Accumulated >= {threshold} or Weekend). Running retrain...")
        try:
            model_xgb, stats = trigger_auto_retrain()
            status["new_trades_since_last_train"] = 0
            status["last_train_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            result["new_trades_since_last_train"] = 0
            result["retrained"] = True
            result["status"] = stats.get("status", "RETRAINED") if stats else "RETRAINED"
            result["stats"] = stats
            print("[Feedback Loop] Model retrained successfully. Accumulator reset.")
            
            # Send Telegram Alert
            if stats:
                from src.model_trainer import get_training_max_setups
                from src.telegram_bot import send_telegram_alert
                max_setups = get_training_max_setups()
                status_emoji = "✅" if stats['status'] == 'ACCEPTED' else "⚠️"
                status_text = "Diterima & Di-promote" if stats['status'] == 'ACCEPTED' else "Ditolak (Akurasi Turun)"
                test_size = int(stats.get('test_size', 0) or 0)
                eval_threshold = float(stats.get('eval_threshold', 0.50) or 0.50)
                old_passed_count = int(stats.get('old_passed_count', 0) or 0)
                new_passed_count = int(stats.get('new_passed_count', 0) or 0)
                msg = (
                    f"🤖 <b>[MLOps Auto-Retrain] Otak AI Diupdate</b> 🤖\n\n"
                    f"• <b>Dataset Latih:</b> {stats['dataset_size']} trade terbaru\n"
                    f"• <b>Active Window Cap:</b> {max_setups} setup terbaru\n"
                    f"• <b>Basis Metric:</b> Holdout test 20% | threshold {eval_threshold:.2f} | sample {test_size}\n"
                    f"• <b>Champion Test Accuracy:</b> {stats.get('old_accuracy', 0.0):.2f}%\n"
                    f"• <b>Challenger Test Accuracy:</b> {stats.get('new_accuracy', 0.0):.2f}%\n"
                    f"• <b>Champion Test Winrate @ {eval_threshold:.2f}:</b> {stats.get('old_winrate', 0.0):.2f}% ({old_passed_count}/{test_size} lolos)\n"
                    f"• <b>Challenger Test Winrate @ {eval_threshold:.2f}:</b> {stats.get('new_winrate', 0.0):.2f}% ({new_passed_count}/{test_size} lolos)\n"
                    f"• <b>Status Kelulusan:</b> {status_emoji} <b>{status_text}</b>\n\n"
                    f"<i>Metric ini dihitung dari labeled/shadow dataset, bukan jaminan winrate live berikutnya. "
                    f"AI menggunakan sample weighting untuk membedakan TP kuat, BEP/proteksi, dan loss.</i>"
                )
                should_send_alert, alert_status = _should_send_retrain_alert(status, stats, max_setups)
                status.update(alert_status)
                if should_send_alert:
                    try:
                        send_telegram_alert(msg)
                    except Exception as te:
                        print(f"Failed to send retraining alert: {te}")
                else:
                    print("[Feedback Loop] Retrain Telegram alert suppressed; same MLOps result already announced.")
        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = str(e)
            print(f"[Feedback Loop] Auto-retraining error: {e}")
    else:
        print(f"[Feedback Loop] Retraining deferred. Accumulating more trades (needs >= {threshold} trades or weekend).")
        
    try:
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(status_file, "w") as f:
            json.dump(status, f, indent=4)
    except Exception as e:
        result["error"] = str(e)
        print(f"[Feedback Loop] Error saving learning status: {e}")

    return result


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_mt5_deal_reason(mt5_module, reason_code):
    if reason_code is None:
        return "UNKNOWN"

    reason_names = [
        "DEAL_REASON_CLIENT",
        "DEAL_REASON_MOBILE",
        "DEAL_REASON_WEB",
        "DEAL_REASON_EXPERT",
        "DEAL_REASON_SL",
        "DEAL_REASON_TP",
        "DEAL_REASON_SO",
    ]
    try:
        reason_int = int(reason_code)
    except (TypeError, ValueError):
        return str(reason_code).upper()

    for attr_name in reason_names:
        attr_value = getattr(mt5_module, attr_name, None)
        if attr_value is not None and int(attr_value) == reason_int:
            return attr_name.replace("DEAL_REASON_", "")
    return str(reason_int)


def classify_trade_close_outcome(features: dict, label: int, pnl_relative: float) -> dict:
    """Classify the real exit path without assuming every positive PnL hit TP."""
    features = features or {}
    pnl_r = _safe_float(pnl_relative, 2.0 if label == 1 else -1.0)
    entry_price = _safe_float(features.get("entry_price"))
    close_price = _safe_float(features.get("close_price"))
    tp_price = _safe_float(features.get("tp_price"))
    direction = int(_safe_float(features.get("direction"), 1) or 1)
    close_reason = str(features.get("close_reason") or "UNKNOWN").upper()
    net_profit = _safe_float(features.get("net_profit"), 0.0)

    reached_tp_by_price = False
    if entry_price is not None and close_price is not None and tp_price is not None:
        tp_distance = abs(tp_price - entry_price)
        price_tolerance = max(tp_distance * 0.02, 0.01)
        if direction == 1:
            reached_tp_by_price = close_price >= (tp_price - price_tolerance)
        else:
            reached_tp_by_price = close_price <= (tp_price + price_tolerance)

    reason_is_tp = close_reason in {"TP", "DEAL_REASON_TP"}
    reason_is_sl = close_reason in {"SL", "DEAL_REASON_SL"}
    is_positive = label == 1 or (net_profit is not None and net_profit > 0)

    if is_positive:
        if reason_is_tp or reached_tp_by_price or pnl_r >= 0.90:
            return {"category": "tp_profit", "weight": 2.0, "close_reason": close_reason}
        if pnl_r <= 0.10:
            return {"category": "bep_profit", "weight": 0.5, "close_reason": close_reason}
        if reason_is_sl:
            return {"category": "protected_profit", "weight": 1.0, "close_reason": close_reason}
        if pnl_r < 0.75:
            return {"category": "protected_profit", "weight": 1.0, "close_reason": close_reason}
        return {"category": "profit_not_tp_verified", "weight": 1.5, "close_reason": close_reason}

    if abs(pnl_r) <= 0.05:
        return {"category": "breakeven", "weight": 0.25, "close_reason": close_reason}
    if pnl_r > -0.5:
        return {"category": "cut_loss_early", "weight": 0.5, "close_reason": close_reason}
    return {"category": "full_loss", "weight": 1.5, "close_reason": close_reason}


def format_trade_outcome_status(close_outcome: dict, label: int, pnl_relative: float):
    category = (close_outcome or {}).get("category")
    if category == "tp_profit":
        return "🏆", "WIN TP (PROFIT)", "🟢"
    if category == "bep_profit":
        return "🛡️", "BEP+ / SL PROTECT", "🟡"
    if category == "protected_profit":
        return "🛡️", "PROTECTED PROFIT", "🟢"
    if category == "profit_not_tp_verified":
        return "🟢", "PROFIT (NON-TP)", "🟢"
    if category == "breakeven":
        return "⚪", "BEP / NYARIS IMPAS", "⚪"
    if category == "cut_loss_early":
        return "🛡️", "CUT-LOSS EARLY", "🟡"
    return "💀", "LOSS (RUGI)", "🔴"


def _registry_result_from_close_outcome(close_outcome: dict, label: int) -> str:
    category = (close_outcome or {}).get("category")
    if category == "tp_profit":
        return "tp"
    if category in {"bep_profit", "protected_profit", "profit_not_tp_verified", "breakeven", "cut_loss_early"}:
        return category
    if category == "full_loss":
        return "sl"
    return "tp" if int(label) == 1 else "sl"


def _outcome_field_suffix(out_field: str) -> str:
    if out_field == "outcome_a_recorded":
        return "_a"
    if out_field == "outcome_b_recorded":
        return "_b"
    return ""


def _store_trade_outcome_in_registry(
    sig_data: dict,
    out_field: str,
    *,
    label: int,
    net_profit: float,
    pnl_relative: float,
    close_price,
    close_reason: str,
    close_outcome: dict,
) -> bool:
    suffix = _outcome_field_suffix(out_field)
    updates = {
        f"status{suffix}": "resolved",
        f"result{suffix}": _registry_result_from_close_outcome(close_outcome, label),
        f"exit_category{suffix}": (close_outcome or {}).get("category"),
        f"pnl_relative{suffix}": pnl_relative,
        f"net_profit{suffix}": net_profit,
        f"close_price{suffix}": close_price,
        f"close_reason{suffix}": close_reason,
    }
    resolved_at_key = f"resolved_at{suffix}"
    if not sig_data.get(resolved_at_key):
        updates[resolved_at_key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    changed = False
    for key, value in updates.items():
        if sig_data.get(key) != value:
            sig_data[key] = value
            changed = True
    return changed


def analyze_trade_outcome_reason(features: dict, label: int, pnl_relative: float) -> str:
    """
    Analyze the technical features of a trade setup and generate a human-readable reason
    explaining why it succeeded or failed, and what the AI learned from it.
    """
    if not features:
        return "Detail fitur entry tidak ditemukan untuk analisis teknis."
        
    direction_val = features.get('direction', 1)
    setup_type_val = "Order Block" if features.get('setup_type', 0) == 1 else "Fair Value Gap"
    trend = features.get('trend', 1)
    floop_trend = features.get('floop_trend', 0)
    floop_strength = features.get('floop_strength', 0.0)
    dist_nearest_pivot = features.get('dist_entry_to_nearest_pivot', 0.0)
    dist_nearest_poc = features.get('dist_entry_to_nearest_poc', 0.0)
    atr = features.get('atr_14', 1.0)
    risk_pips = features.get('risk_pips', 1.0)
    
    # 1. Technical Analysis of the Entry State
    reasons = []
    
    # Check structure alignment
    struct_aligned = (trend == direction_val)
    if struct_aligned:
        reasons.append(f"selaras dengan tren struktur pasar utama (SMC Trend: {'BULLISH' if trend == 1 else 'BEARISH'})")
    else:
        reasons.append(f"berlawanan dengan tren struktur pasar utama (SMC Trend: {'BULLISH' if trend == 1 else 'BEARISH'})")
        
    # Check FLoOP volume alignment
    volume_aligned = (floop_trend == direction_val)
    if volume_aligned:
        reasons.append(f"volume institusi selaras dengan arah entri (FLoOP Trend: {'BULLISH' if floop_trend == 1 else 'BEARISH'})")
    elif floop_trend != 0:
        reasons.append(f"volume institusi berlawanan dengan arah entri (FLoOP Trend: {'BULLISH' if floop_trend == 1 else 'BEARISH'})")
        
    if floop_strength > 5.0:
        reasons.append(f"momentum volume FLoOP solid (kekuatan {floop_strength:.1f})")
    else:
        reasons.append(f"momentum volume FLoOP lemah saat entry (kekuatan {floop_strength:.1f})")
        
    # Check proximity to Key Levels
    if dist_nearest_poc < 0.002:
        reasons.append("dieksekusi dekat dengan area konsentrasi volume tertinggi (POC)")
    if dist_nearest_pivot < 0.002:
        reasons.append("dieksekusi dekat dengan level Pivot harian broker")
        
    # Check risk ratio
    relative_risk = risk_pips / atr if atr > 0 else 0.0
    if relative_risk < 0.1:
        reasons.append(" Stop Loss sangat ketat sehingga rentan noise")

    close_outcome = classify_trade_close_outcome(features, label, pnl_relative)
    outcome_category = close_outcome.get("category")
    outcome_weight = close_outcome.get("weight", 1.0)
    close_reason = close_outcome.get("close_reason", "UNKNOWN")
    manager_exit_trigger = features.get("manager_exit_trigger")
    manager_exit_timeframe = features.get("manager_exit_timeframe")
    manager_exit_detail = features.get("manager_exit_detail")

    # 2. Match with Outcome to form analysis text
    analysis_text = ""
    lesson_text = ""
    
    if label == 1:
        # Success factors
        success_factors = []
        if struct_aligned and volume_aligned:
            success_factors.append("keselarasan struktur SMC dan momentum volume FLoOP")
        if dist_nearest_poc < 0.002 or dist_nearest_pivot < 0.002:
            success_factors.append("rebound harga pada level POC/Pivot broker")
        if floop_strength > 5.0:
            success_factors.append("kekuatan momentum volume pasar yang memadai")
            
        if not success_factors:
            success_factors.append("keselarasan probabilitas machine learning pada parameter entry")
            
        factors_str = " dan ".join(success_factors)
        if outcome_category == "tp_profit":
            analysis_text = f"🏆 <b>Penyebab Profit TP:</b> Trade berhasil mencapai area TP karena didukung oleh {factors_str}."
            lesson_text = (
                f"🧠 <b>Pelajaran AI (Bobot Latih = {outcome_weight:.2f}):</b>\n"
                f"AI memperkuat memori setup ini sebagai winner berkualitas karena exit selaras dengan target utama. "
                f"Model akan menaikkan prioritas setup serupa saat struktur, level, dan momentum kembali cocok."
            )
        elif outcome_category == "bep_profit":
            analysis_text = (
                f"🛡️ <b>Penyebab BEP+ / Profit Proteksi:</b> Posisi ditutup hijau kecil "
                f"({pnl_relative:+.2f} R) melalui exit proteksi/SL bergerak (Reason: {close_reason}), bukan karena TP utama tersentuh. "
                f"Entry masih dibantu oleh {factors_str}, tetapi follow-through market belum cukup kuat sampai target."
            )
            lesson_text = (
                f"🧠 <b>Pelajaran AI (Bobot Latih = {outcome_weight:.2f}):</b>\n"
                f"AI menyimpan setup ini sebagai profit defensif, bukan winner murni. "
                f"Model belajar bahwa pola ini boleh dihargai karena menjaga modal, tetapi confidence tidak dinaikkan sekuat trade yang benar-benar TP."
            )
        elif outcome_category == "protected_profit":
            analysis_text = (
                f"🛡️ <b>Penyebab Protected Profit:</b> Trade keluar profit sebelum TP utama "
                f"({pnl_relative:+.2f} R, Reason: {close_reason}). Faktor entry yang membantu: {factors_str}. "
                f"Ini lebih tepat dibaca sebagai proteksi profit/trailing exit, bukan validasi penuh target."
            )
            lesson_text = (
                f"🧠 <b>Pelajaran AI (Bobot Latih = {outcome_weight:.2f}):</b>\n"
                f"AI memberi reward sedang. Setup ini positif secara risk management, tetapi masih perlu lebih banyak bukti "
                f"sebelum dianggap pola yang kuat untuk mengejar TP penuh."
            )
        else:
            analysis_text = (
                f"🟢 <b>Penyebab Profit Non-TP:</b> Trade ditutup profit ({pnl_relative:+.2f} R), "
                f"namun exit belum terverifikasi sebagai TP utama. Faktor entry yang mendukung: {factors_str}."
            )
            lesson_text = (
                f"🧠 <b>Pelajaran AI (Bobot Latih = {outcome_weight:.2f}):</b>\n"
                f"AI tetap mencatat hasil ini sebagai profit, tetapi bobotnya lebih konservatif karena jalur exit bukan TP yang jelas."
            )
    else:
        # Failure cases
        if outcome_category == "breakeven":
            analysis_text = (
                f"⚪ <b>Penyebab BEP / Nyaris Impas:</b> Posisi ditutup sangat dekat area entry "
                f"({pnl_relative:+.2f} R, Reason: {close_reason}). Setup belum cukup memberi bukti arah kuat maupun kegagalan penuh."
            )
            lesson_text = (
                f"🧠 <b>Pelajaran AI (Bobot Latih = {outcome_weight:.2f}):</b>\n"
                f"AI memberi bobot sangat kecil karena hasilnya netral. Data ini berguna untuk membaca noise, "
                f"tetapi tidak boleh menaikkan confidence seperti TP."
            )
        elif outcome_category == "cut_loss_early":
            # Cut-loss
            analysis_text = (
                f"🛡️ <b>Penyebab Cut-Loss Early:</b> Sinyal struktur pasar berbalik arah (CHoCH) "
                f"atau momentum volume berputar di timeframe kecil sebelum SL utama tersentuh. "
                f"Bot memotong kerugian lebih awal guna menjaga ketahanan modal."
            )
            lesson_text = (
                f"🧠 <b>Pelajaran AI (Bobot Latih = {outcome_weight:.2f}):</b>\n"
                f"AI memberikan bobot rendah (0.50) sebagai reward proteksi risiko. "
                f"Model mempelajari bahwa tindakan keluar lebih cepat saat arah trend berbalik "
                f"sangat efektif menjaga ketahanan drawdown akun trading."
            )
        else:
            # Full Loss
            failure_factors = []
            if not struct_aligned:
                failure_factors.append("melawan arah tren struktur pasar utama")
            if not volume_aligned and floop_trend != 0:
                failure_factors.append("tekanan volume institusional FLoOP yang berlawanan")
            if floop_strength <= 3.0:
                failure_factors.append("kurangnya kekuatan dorongan volume pasar saat entry")
            if relative_risk < 0.1:
                failure_factors.append("jarak stop loss terlalu sempit terhadap volatilitas market")
                
            if not failure_factors:
                failure_factors.append("adanya noise manipulasi pasar jangka pendek di luar estimasi ML")
                
            factors_str = " dan ".join(failure_factors)
            analysis_text = f"💀 <b>Penyebab Loss:</b> Trade menyentuh SL penuh akibat {factors_str}."
            lesson_text = (
                f"🧠 <b>Pelajaran AI (Bobot Latih = {outcome_weight:.2f}):</b>\n"
                f"AI memberikan penalti bobot latih standar (1.50) pada kegagalan ini. "
                f"Model mempelajari kegagalan ini dan akan menurunkan tingkat probabilitas kelulusan "
                f"pada setup sejenis di masa depan untuk meminimalkan loss konyol."
            )
            
    manager_exit_text = ""
    if manager_exit_trigger:
        manager_exit_text = f"\n\n📌 <b>Trigger Exit Manager:</b> {manager_exit_trigger}"
        if manager_exit_timeframe:
            manager_exit_text += f" ({manager_exit_timeframe})"
        if manager_exit_detail:
            manager_exit_text += f" - {manager_exit_detail}"

    return f"{analysis_text}{manager_exit_text}\n\n{lesson_text}"


def process_mt5_history_feedback(sent_signals_file="data/sent_signals.json", labeled_data_path="data/labeled_setups.csv", return_details=False):
    """
    Query MT5 history to check outcomes of sent orders, record wins/losses,
    and trigger retraining if new outcomes are recorded.
    Sends Telegram notifications when a position closes.
    """
    import json
    import os
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta
    from src.telegram_bot import send_telegram_alert
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(sent_signals_file):
        sent_signals_file = os.path.join(base_dir, sent_signals_file)
    if not os.path.isabs(labeled_data_path):
        labeled_data_path = os.path.join(base_dir, labeled_data_path)
        
    empty_result = {"feedback_count": 0, "retrain_result": None}
    if not os.path.exists(sent_signals_file):
        return empty_result if return_details else 0
        
    try:
        with open(sent_signals_file, "r") as f:
            sent_signals = json.load(f)
    except Exception:
        return empty_result if return_details else 0
        
    # Check if MT5 is initialized
    mt5_initialized = True
    if not mt5.initialize():
        mt5_initialized = False
        
    if not mt5_initialized:
        print("[Feedback Loop] Error: MT5 terminal connection not available.")
        return empty_result if return_details else 0
        
    feedback_trades = []
    existing_feedback_keys = _load_existing_feedback_keys(labeled_data_path)
    updated = False
    
    from src.execution import get_active_broker_symbol
    broker_symbol = get_active_broker_symbol("XAUUSD")
    
    for sig_key, sig_data in sent_signals.items():
        already_recorded_signal = sig_data.get('outcome_recorded', False)
            
        timeframe = sig_data.get('timeframe', 'M30')
        direction = sig_data.get('direction', 'BULL')
        dir_val = 1 if direction == 'BULL' else -1
        
        # We handle two cases: Old single ticket format, and new dual ticket format
        tickets_to_check = [] # List of tuples: (ticket, option_key, outcome_field)
        
        if 'ticket_id' in sig_data:
            ticket_id = sig_data.get('ticket_id')
            if ticket_id is None:
                entry_price = sig_data['features']['entry_price']
                manual_ticket = find_matching_manual_deal(broker_symbol, dir_val, entry_price, sig_data.get('time_sent', ''))
                if manual_ticket:
                    sig_data['ticket_id'] = manual_ticket
                    ticket_id = manual_ticket
                    updated = True
            tickets_to_check.append((ticket_id, 'Single', 'outcome_recorded'))
        else:
            ticket_a = sig_data.get('ticket_a')
            ticket_b = sig_data.get('ticket_b')
            
            # Check if ticket_a was None (rejected/not placed), search for manual entry
            if ticket_a is None:
                entry_price = sig_data.get('price_0.5')
                if entry_price:
                    manual_ticket = find_matching_manual_deal(broker_symbol, dir_val, entry_price, sig_data.get('time_sent', ''))
                    if manual_ticket and manual_ticket != ticket_b:
                        sig_data['ticket_a'] = manual_ticket
                        ticket_a = manual_ticket
                        sig_data['is_manual_a'] = True
                        updated = True
                        
            # Check if ticket_b was None (rejected/not placed), search for manual entry
            if ticket_b is None:
                entry_price = sig_data.get('price_0.618')
                if entry_price:
                    manual_ticket = find_matching_manual_deal(broker_symbol, dir_val, entry_price, sig_data.get('time_sent', ''))
                    if manual_ticket and manual_ticket != ticket_a:
                        sig_data['ticket_b'] = manual_ticket
                        ticket_b = manual_ticket
                        sig_data['is_manual_b'] = True
                        updated = True
                        
            if ticket_a is not None and (already_recorded_signal or not sig_data.get('outcome_a_recorded', False)):
                tickets_to_check.append((ticket_a, 'Option A (0.50)', 'outcome_a_recorded'))
            if ticket_b is not None and (already_recorded_signal or not sig_data.get('outcome_b_recorded', False)):
                tickets_to_check.append((ticket_b, 'Option B (0.618)', 'outcome_b_recorded'))
                
        for ticket, opt_name, out_field in tickets_to_check:
            if ticket is None:
                if not sig_data.get(out_field, False):
                    sig_data[out_field] = True
                    updated = True
                continue
                
            # 1. Check if still active pending order
            active_orders = mt5.orders_get(ticket=ticket)
            if active_orders is not None and len(active_orders) > 0:
                continue  # Still pending, skip
                
            # 2. Check if still open position
            active_positions = mt5.positions_get(ticket=ticket)
            if active_positions is not None and len(active_positions) > 0:
                continue  # Still open position, skip
                
            # 3. Not pending, not open. Check if there are deals in history
            deals = mt5.history_deals_get(position=ticket)
            if deals is None or len(deals) == 0:
                if already_recorded_signal or sig_data.get(out_field, False):
                    continue
                # Order was cancelled or expired without being filled
                sig_data[out_field] = True
                updated = True
                print(f"[Feedback Loop] Signal {sig_key} ({opt_name} Ticket #{ticket}) was cancelled/expired.")
                try:
                    send_telegram_alert(
                        f"🧹 <b>[Risk Management] Pending Order Dicabut</b> 🧹\n\n"
                        f"Order pending limit berikut telah dibatalkan (mitigated/expired/too far).\n"
                        f"• <b>Timeframe:</b> {timeframe}\n"
                        f"• <b>Setup:</b> {direction} ({opt_name})\n"
                        f"• <b>Ticket:</b> #{ticket}"
                    )
                except Exception as e:
                    print(f"Failed to send telegram cancel alert: {e}")
                continue
                
            # 4. Closed! Calculate net profit
            total_profit = sum([d.profit for d in deals if d.position_id == ticket])
            commission = sum([d.commission for d in deals if d.position_id == ticket])
            swap = sum([d.swap for d in deals if d.position_id == ticket])
            net_profit = total_profit + commission + swap
            
            label = 1 if net_profit > 0 else 0
            
            # Retrieve features for this specific option
            features = None
            if opt_name == 'Option A (0.50)':
                features = sig_data.get('features_0.5')
            elif opt_name == 'Option B (0.618)':
                features = sig_data.get('features_0.618')
            else:
                features = sig_data.get('features')

            features_dict = None
            close_price = None
            close_reason = "UNKNOWN"
            pnl_relative = 2.0 if label == 1 else -1.0
            duplicate_feedback = False

            if features:
                features_dict = features.copy()
                features_dict['label'] = label
                features_dict['time'] = sig_data.get('time_sent', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                
                # Calculate actual close price and broker exit reason from deals.
                for d in deals:
                    if d.entry == mt5.DEAL_ENTRY_OUT:
                        close_price = d.price
                        close_reason = _format_mt5_deal_reason(mt5, getattr(d, "reason", None))
                        break
                        
                entry_price = features_dict.get('entry_price')
                sl_price = features_dict.get('sl_price')
                if close_price and entry_price and sl_price:
                    risk_price = abs(entry_price - sl_price)
                    if risk_price > 0:
                        if dir_val == 1: # Buy
                            pnl_relative = (close_price - entry_price) / risk_price
                        else: # Sell
                            pnl_relative = (entry_price - close_price) / risk_price

                suffix = _outcome_field_suffix(out_field)
                for field_name in (
                    "manager_exit_trigger",
                    "manager_exit_timeframe",
                    "manager_exit_detail",
                    "manager_exit_recorded_at",
                ):
                    registry_value = sig_data.get(f"{field_name}{suffix}")
                    if registry_value is not None:
                        features_dict[field_name] = registry_value
                            
                features_dict['pnl_relative'] = pnl_relative
                features_dict['close_price'] = close_price
                features_dict['close_reason'] = close_reason
                features_dict['net_profit'] = net_profit
                feedback_key = _feedback_row_key(features_dict)
                if feedback_key not in existing_feedback_keys:
                    feedback_trades.append(features_dict)
                    existing_feedback_keys.add(feedback_key)
                    print(f"[Feedback Loop] Recorded outcome for signal {sig_key} ({opt_name}): {'WIN' if label == 1 else 'LOSS'} (Net Profit: {net_profit:,.2f}, PnL Relative: {pnl_relative:.2f})")
                elif already_recorded_signal or sig_data.get(out_field, False):
                    duplicate_feedback = True

            close_outcome = classify_trade_close_outcome(features_dict if features_dict else {}, label, pnl_relative)
            if _store_trade_outcome_in_registry(
                sig_data,
                out_field,
                label=label,
                net_profit=net_profit,
                pnl_relative=pnl_relative,
                close_price=close_price,
                close_reason=close_reason,
                close_outcome=close_outcome,
            ):
                updated = True

            if duplicate_feedback:
                continue

            if not sig_data.get(out_field, False):
                sig_data[out_field] = True
                updated = True
            elif already_recorded_signal:
                continue
            
            # Send Telegram Alert
            emoji, outcome_str, color_str = format_trade_outcome_status(close_outcome, label, pnl_relative)
            
            # Check if it was a manual entry
            is_manual = False
            if opt_name == 'Option A (0.50)':
                is_manual = sig_data.get('is_manual_a', False)
            elif opt_name == 'Option B (0.618)':
                is_manual = sig_data.get('is_manual_b', False)
            
            manual_tag = " (Manual Entry)" if is_manual else ""
            
            account_info_fn = getattr(mt5, "account_info", None)
            account_info = account_info_fn() if account_info_fn is not None else None
            currency = getattr(account_info, "currency", "USD") or "USD"
            
            # Generate detailed technical analysis explanation
            analysis_details = ""
            if features_dict:
                try:
                    analysis_details = analyze_trade_outcome_reason(features_dict, label, pnl_relative)
                except Exception as ae:
                    print(f"Error generating analysis details: {ae}")
                    analysis_details = "Terjadi kesalahan saat menyusun analisis teknis."
            else:
                analysis_details = "Detail fitur entry tidak ditemukan untuk analisis teknis."
                
            setup_type_str = "Order Block" if features_dict and features_dict.get('setup_type', 0) == 1 else "Fair Value Gap"
            
            msg = (
                f"{emoji} <b>[SMC Trade Closed] {opt_name}{manual_tag} Selesai</b> {emoji}\n\n"
                f"• <b>Ticket:</b> #{ticket}\n"
                f"• <b>Timeframe:</b> {timeframe}\n"
                f"• <b>Setup:</b> {direction} ({setup_type_str})\n"
                f"• <b>Entry Price:</b> <code>{features_dict.get('entry_price', 0.0) if features_dict else 0.0:.3f}</code>\n"
                f"• <b>SL | TP:</b> <code>{features_dict.get('sl_price', 0.0) if features_dict else 0.0:.3f} | {features_dict.get('tp_price', 0.0) if features_dict else 0.0:.3f}</code>\n"
                f"• <b>Close Price:</b> <code>{features_dict.get('close_price', 0.0) if features_dict else 0.0:.3f}</code> ({features_dict.get('close_reason', 'UNKNOWN') if features_dict else 'UNKNOWN'})\n"
                f"• <b>Hasil Posisi:</b> {color_str} <b>{outcome_str}</b>\n"
                f"• <b>Net PnL Riil:</b> <code>{net_profit:+,.2f} {currency}</code>\n"
                f"• <b>PnL Relative:</b> <code>{pnl_relative:+.2f} R</code>\n\n"
                f"{analysis_details}\n\n"
                f"<i>Database latih diperbarui & retraining otomatis dipicu.</i>"
            )
            try:
                send_telegram_alert(msg)
            except Exception as e:
                print(f"Failed to send trade closed Telegram alert: {e}")
                
        # If it's a dual model, check if both options are recorded
        if 'ticket_a' in sig_data or 'ticket_b' in sig_data:
            a_done = sig_data.get('outcome_a_recorded', False) or ('ticket_a' not in sig_data) or (sig_data.get('ticket_a') is None)
            b_done = sig_data.get('outcome_b_recorded', False) or ('ticket_b' not in sig_data) or (sig_data.get('ticket_b') is None)
            if a_done and b_done:
                sig_data['outcome_recorded'] = True
                updated = True
                
    retrain_result = None
    if feedback_trades:
        # Save feedback data
        update_feedback_data(feedback_trades, labeled_data_path)
        # Check and trigger retraining based on threshold
        retrain_result = check_and_trigger_retraining(len(feedback_trades))
            
    if updated:
        # Save signals registry back
        with open(sent_signals_file, "w") as f:
            json.dump(sent_signals, f, indent=4)
            
    if return_details:
        return {"feedback_count": len(feedback_trades), "retrain_result": retrain_result}
    return len(feedback_trades)
