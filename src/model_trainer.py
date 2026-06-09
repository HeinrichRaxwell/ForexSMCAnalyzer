import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from dotenv import load_dotenv
import joblib

# Add project root to python path if not present
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
load_dotenv()

NON_FEATURE_COLUMNS = [
    'label',
    'time',
    'pnl_relative',
    'sample_source',
    'strategy',
    'signal_id',
    'confidence',
    'accept_threshold',
    'resolved_at',
    'result',
    'source',
    'status',
    'created_at',
    'latest_seen_at',
    'filtered_reason',
    'ticket_id',
    'triggered_at',
]


def _resolve_project_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _default_shadow_labeled_path(labeled_data_path):
    return os.path.join(os.path.dirname(labeled_data_path), "shadow_labeled_setups.csv")


def _read_float_env(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        print(f"[Trainer Warning] Invalid {name}={raw_value!r}; using {default}.")
        return default


def _read_int_env(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        print(f"[Trainer Warning] Invalid {name}={raw_value!r}; using {default}.")
        return default


def _normalize_labeled_frame(df, sample_source):
    if df.empty:
        normalized = df.copy()
        normalized['sample_source'] = sample_source
        return normalized

    if 'label' not in df.columns:
        raise ValueError("Training dataset must contain a label column.")

    normalized = df.copy()
    normalized['label'] = pd.to_numeric(normalized['label'], errors='coerce')
    normalized = normalized[normalized['label'].isin([0, 1])].copy()
    normalized['label'] = normalized['label'].astype(int)
    normalized['sample_source'] = sample_source
    return normalized


def _load_real_labeled_data(labeled_data_path):
    if not os.path.exists(labeled_data_path):
        raise FileNotFoundError(f"Labeled setups file not found at {labeled_data_path}")
    return _normalize_labeled_frame(pd.read_csv(labeled_data_path), "real")


def _load_shadow_labeled_data(shadow_labeled_data_path):
    if not shadow_labeled_data_path or not os.path.exists(shadow_labeled_data_path):
        return pd.DataFrame()
    shadow_df = _normalize_labeled_frame(pd.read_csv(shadow_labeled_data_path), "shadow")
    if not shadow_df.empty:
        print(f"[Source-Aware Training] Loaded {len(shadow_df)} resolved shadow setups from {shadow_labeled_data_path}.")
    return shadow_df


def load_training_dataset(labeled_data_path="data/labeled_setups.csv", shadow_labeled_data_path=None):
    """Load real labeled setups plus optional resolved shadow setups for training."""
    labeled_data_path = _resolve_project_path(labeled_data_path)
    if shadow_labeled_data_path is None:
        shadow_labeled_data_path = _default_shadow_labeled_path(labeled_data_path)
    shadow_labeled_data_path = _resolve_project_path(shadow_labeled_data_path)

    real_df = _load_real_labeled_data(labeled_data_path)
    shadow_df = _load_shadow_labeled_data(shadow_labeled_data_path)
    if shadow_df.empty:
        return real_df.reset_index(drop=True)
    return pd.concat([real_df, shadow_df], ignore_index=True, sort=False).reset_index(drop=True)


def prepare_training_features(df):
    """Return numeric model features and binary target without outcome/source metadata leakage."""
    X = df.drop(columns=NON_FEATURE_COLUMNS, errors='ignore').copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all').fillna(0.0)
    y = df['label'].astype(int)
    return X, y


def calculate_sample_weights(df, row_indexes=None, shadow_sample_weight=None):
    """Build training weights from outcome severity, then down-weight shadow rows."""
    selected = df.loc[row_indexes].copy() if row_indexes is not None else df.copy()
    sample_weights = np.ones(len(selected))

    if 'pnl_relative' in selected.columns:
        pnl_values = pd.to_numeric(selected['pnl_relative'], errors='coerce').values
        labels = selected['label'].astype(int).values
        for idx_row in range(len(selected)):
            pnl_val = pnl_values[idx_row]
            if labels[idx_row] == 1:
                if pd.notna(pnl_val) and pnl_val <= 0.10:
                    sample_weights[idx_row] = 0.5
                elif pd.notna(pnl_val) and pnl_val < 0.90:
                    sample_weights[idx_row] = 1.0
                else:
                    sample_weights[idx_row] = 2.0
            else:
                if pd.notna(pnl_val) and pnl_val > -0.5:
                    sample_weights[idx_row] = 0.5
                else:
                    sample_weights[idx_row] = 1.5

    if shadow_sample_weight is None:
        shadow_sample_weight = get_shadow_sample_weight()
    shadow_sample_weight = max(0.0, min(float(shadow_sample_weight), 1.0))

    if 'sample_source' in selected.columns:
        sources = selected['sample_source'].fillna("real").astype(str).str.lower().values
        sample_weights[sources == "shadow"] *= shadow_sample_weight

    return sample_weights


def make_xgb_model():
    """Single source of truth for the XGBoost hyperparameters used in training and OOF scoring."""
    return XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )


def make_lgb_model():
    """Single source of truth for the LightGBM hyperparameters used in training and OOF scoring."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )


def get_shadow_sample_weight():
    return max(0.0, min(_read_float_env("ML_SHADOW_SAMPLE_WEIGHT", 0.35), 1.0))


def get_training_max_setups():
    return max(1, _read_int_env("ML_TRAINING_MAX_SETUPS", 1000))


def _apply_data_windowing(df):
    time_sort_col = None
    if 'time' in df.columns:
        try:
            df = df.copy()
            df['time_parsed'] = pd.to_datetime(df['time'])
            time_sort_col = 'time_parsed'
            max_time = df['time_parsed'].max()
            if pd.notna(max_time):
                cutoff_time = max_time - pd.Timedelta(days=180)
                before_count = len(df)
                df = df[df['time_parsed'] >= cutoff_time].copy()
                after_count = len(df)
                if before_count - after_count > 0:
                    print(f"[Data Windowing] Removed {before_count - after_count} setups older than 6 months (cutoff: {cutoff_time.strftime('%Y-%m-%d')}).")
        except Exception as e:
            print(f"[Data Windowing Warning] Failed to apply date-based filter: {e}")
            time_sort_col = None

    max_setups = get_training_max_setups()
    if len(df) > max_setups:
        print(f"[Data Windowing] Limiting dataset to the latest {max_setups} setups (removing {len(df) - max_setups} older setups to prevent overfitting on outdated market regimes).")
        if time_sort_col is not None and time_sort_col in df.columns:
            df = df.sort_values(time_sort_col, kind='mergesort').iloc[-max_setups:].copy()
        else:
            df = df.iloc[-max_setups:].copy()

    if 'time_parsed' in df.columns:
        df = df.drop(columns=['time_parsed'])
    return df.reset_index(drop=True)


def train_xgboost_filter(labeled_data_path="data/labeled_setups.csv", model_dir="models", shadow_labeled_data_path=None):
    """
    Train an XGBoost classifier to filter out low-probability SMC trade setups.
    
    Args:
        labeled_data_path (str): Path to the labeled CSV dataset.
        model_dir (str): Directory where the trained model should be saved.
        shadow_labeled_data_path (str): Optional path to resolved shadow setup labels.
        
    Returns:
        XGBClassifier: The trained XGBoost model.
    """
    labeled_data_path = _resolve_project_path(labeled_data_path)
    if shadow_labeled_data_path is None:
        shadow_labeled_data_path = _default_shadow_labeled_path(labeled_data_path)
    shadow_labeled_data_path = _resolve_project_path(shadow_labeled_data_path)
    model_dir = _resolve_project_path(model_dir)
        
    print(f"Loading labeled data from {labeled_data_path}...")
    real_df = _load_real_labeled_data(labeled_data_path)
    print(f"Loaded {len(real_df)} real setups.")
    real_df = _apply_data_windowing(real_df)
        
    try:
        real_df.drop(columns=['sample_source'], errors='ignore').to_csv(labeled_data_path, index=False)
        print(f"Pruned dataset written back to: {labeled_data_path}")
    except Exception as e:
        print(f"Warning: Failed to save pruned dataset back to CSV: {e}")

    shadow_df = _load_shadow_labeled_data(shadow_labeled_data_path)
    if shadow_df.empty:
        df = real_df.reset_index(drop=True)
    else:
        df = pd.concat([real_df, shadow_df], ignore_index=True, sort=False).reset_index(drop=True)
        print(f"[Source-Aware Training] Combined dataset: {len(real_df)} real + {len(shadow_df)} shadow setups.")
        df = _apply_data_windowing(df)
    real_count = int((df.get('sample_source', pd.Series(dtype=str)) == 'real').sum())
    shadow_count = int((df.get('sample_source', pd.Series(dtype=str)) == 'shadow').sum())
    
    X, y = prepare_training_features(df)
    
    print(f"Features: {list(X.columns)}")
    print(f"Target: label")
    
    # Split the data into training and test sets (80% train, 20% test) stratified by target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    sample_weights_train = calculate_sample_weights(df, X_train.index)
    
    # Train the XGBClassifier
    print("Training XGBoost Classifier with sample weights...")
    model_xgb = make_xgb_model()
    model_xgb.fit(X_train, y_train, sample_weight=sample_weights_train)

    # Train the LGBMClassifier
    print("Training LightGBM Classifier with sample weights...")
    model_lgb = make_lgb_model()
    model_lgb.fit(X_train, y_train, sample_weight=sample_weights_train)
    
    # Evaluate model metrics
    y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
    y_prob_lgb = model_lgb.predict_proba(X_test)[:, 1]
    y_prob = (y_prob_xgb + y_prob_lgb) / 2
    
    y_pred = (y_prob >= 0.5).astype(int)
    
    print("\n" + "="*40)
    print("        CLASSIFICATION REPORT (XGB)       ")
    print("="*40)
    print(classification_report(y_test, model_xgb.predict(X_test)))
    
    print("\n" + "="*40)
    print("        CLASSIFICATION REPORT (LGBM)      ")
    print("="*40)
    print(classification_report(y_test, model_lgb.predict(X_test)))
    
    print("\n" + "="*40)
    print("        CLASSIFICATION REPORT (Ensemble)   ")
    print("="*40)
    print(classification_report(y_test, y_pred))
    
    print("\n" + "="*60)
    print("     WINRATE AT DIFFERENT CONFIDENCE THRESHOLDS (Ensemble)     ")
    print("="*60)
    
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85]
    total_test = len(y_test)
    
    for t in thresholds:
        passed_idx = y_prob >= t
        passed_count = np.sum(passed_idx)
        filtered_count = total_test - passed_count
        
        if passed_count > 0:
            winrate = np.mean(y_test[passed_idx]) * 100
        else:
            winrate = 0.0
            
        print(f"Threshold: {t:.2f} | Winrate: {winrate:6.2f}% | Passed: {passed_count:4d}/{total_test} | Filtered: {filtered_count:4d}")
    print("="*60 + "\n")
    
    xgb_model_path = os.path.join(model_dir, "smc_xgb_classifier.joblib")
    lgb_model_path = os.path.join(model_dir, "smc_lgb_classifier.joblib")
    
    # Calculate stats
    eval_threshold = 0.5
    new_passed_idx = y_prob >= 0.5
    new_passed_count = int(np.sum(new_passed_idx))
    new_winrate = np.mean(y_test[new_passed_idx]) * 100 if np.sum(new_passed_idx) > 0 else 0.0
    
    new_pred = (y_prob >= 0.5).astype(int)
    new_accuracy = accuracy_score(y_test, new_pred) * 100
    
    stats_dict = {
        'dataset_size': len(df),
        'real_dataset_size': real_count,
        'shadow_dataset_size': shadow_count,
        'shadow_sample_weight': get_shadow_sample_weight(),
        'feature_columns': list(X.columns),
        'test_size': int(len(y_test)),
        'eval_threshold': eval_threshold,
        'old_passed_count': new_passed_count, # Fallback
        'new_passed_count': new_passed_count,
        'old_winrate': new_winrate, # Fallback
        'new_winrate': new_winrate,
        'old_accuracy': new_accuracy, # Fallback
        'new_accuracy': new_accuracy,
        'status': 'ACCEPTED'
    }
    
    # Champion vs Challenger validation gate
    should_save = True
    if os.path.exists(xgb_model_path) and os.path.exists(lgb_model_path):
        try:
            print("\n[MLOps] Comparing Challenger (new) vs Champion (old) model...")
            old_xgb = joblib.load(xgb_model_path)
            old_lgb = joblib.load(lgb_model_path)
            
            # Evaluate old models on the new test set
            old_xgb_prob = old_xgb.predict_proba(X_test)[:, 1]
            old_lgb_prob = old_lgb.predict_proba(X_test)[:, 1]
            old_prob = (old_xgb_prob + old_lgb_prob) / 2
            
            old_passed_idx = old_prob >= 0.5
            old_passed_count = int(np.sum(old_passed_idx))
            old_winrate = np.mean(y_test[old_passed_idx]) * 100 if np.sum(old_passed_idx) > 0 else 0.0
            
            old_pred = (old_prob >= 0.5).astype(int)
            old_accuracy = accuracy_score(y_test, old_pred) * 100
            
            stats_dict['old_passed_count'] = old_passed_count
            stats_dict['old_winrate'] = old_winrate
            stats_dict['old_accuracy'] = old_accuracy
            
            print(f"[MLOps] Champion (Old) Test Accuracy: {old_accuracy:.2f}% | Winrate Lolos Filter: {old_winrate:.2f}%")
            print(f"[MLOps] Challenger (New) Test Accuracy: {new_accuracy:.2f}% | Winrate Lolos Filter: {new_winrate:.2f}%")
            
            # Kelulusan berdasarkan akurasi otak baru (Challenger) vs otak lama (Champion)
            if new_accuracy < old_accuracy:
                print("[MLOps] Challenger REJECTED: New model accuracy is lower than old model. Keeping Champion.")
                stats_dict['status'] = 'REJECTED'
                should_save = False
                model_xgb = old_xgb
            else:
                print("[MLOps] Challenger ACCEPTED: New model accuracy meets or exceeds old model. Promoting to Champion.")
                stats_dict['status'] = 'ACCEPTED'
        except Exception as e:
            print(f"[MLOps Warning] Error during Champion vs Challenger comparison: {e}")
            print("[MLOps] Challenger REJECTED: Champion validation gate failed. Keeping Champion.")
            stats_dict['status'] = 'REJECTED_GATE_ERROR'
            stats_dict['gate_error'] = str(e)
            should_save = False

    # Save the trained models only if accepted
    if should_save:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model_xgb, xgb_model_path)
        joblib.dump(model_lgb, lgb_model_path)
        print(f"Trained XGBoost model saved to: {xgb_model_path}")
        print(f"Trained LightGBM model saved to: {lgb_model_path}")
    else:
        print("[MLOps] Model files were NOT overwritten. Champion remains active.")
        
    return model_xgb, stats_dict

if __name__ == "__main__":
    train_xgboost_filter()
