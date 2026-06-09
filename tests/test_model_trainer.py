import os
import sys
import pandas as pd
import numpy as np
import pytest
import joblib

# Add project root to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import (
    _apply_data_windowing,
    calculate_sample_weights,
    load_training_dataset,
    prepare_training_features,
    train_xgboost_filter,
)


class RejectingChampionModel:
    def predict_proba(self, X):
        raise RuntimeError("forced champion gate failure")


def test_train_xgboost_filter(tmp_path):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a small dummy dataframe with realistic columns and enough samples for train/test split
    # Since test_size=0.2, 50 samples will give 40 train and 10 test samples.
    # We make sure there is stratification by including both 0 and 1 labels.
    data = {
        'time': pd.date_range(start="2026-06-01", periods=50, freq="15min").strftime("%Y-%m-%d %H:%M:%S"),
        'hour': np.random.randint(0, 24, size=50),
        'day_of_week': np.random.randint(0, 7, size=50),
        'setup_type': np.random.randint(0, 2, size=50),
        'direction': np.random.choice([-1, 1], size=50),
        'entry_price': np.random.uniform(2300, 2350, size=50),
        'sl_price': np.random.uniform(2300, 2350, size=50),
        'tp_price': np.random.uniform(2300, 2350, size=50),
        'risk_pips': np.random.uniform(1.0, 10.0, size=50),
        'atr_14': np.random.uniform(1.0, 5.0, size=50),
        'trend': np.random.choice([-1, 1], size=50),
        'killzone': np.random.randint(0, 4, size=50),
        'label': np.random.choice([0, 1], size=50)
    }
    
    # Force some variation to avoid issues during stratification or split
    data['label'][:25] = 0
    data['label'][25:] = 1
    
    df = pd.DataFrame(data)
    
    # Save to temp csv
    labeled_data_path = tmp_path / "labeled_setups.csv"
    df.to_csv(labeled_data_path, index=False)
    
    # Run the trainer
    model_dir = tmp_path / "models"
    model, stats = train_xgboost_filter(labeled_data_path=str(labeled_data_path), model_dir=str(model_dir))
    
    # Assert model was returned and file is saved
    assert model is not None
    assert stats is not None
    assert stats['dataset_size'] == 50
    model_path = model_dir / "smc_xgb_classifier.joblib"
    assert os.path.exists(model_path)
    
    # Assert model can be loaded and predict
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None
    
    # Verify predictions run on feature columns (excluding label and time)
    X_test = df.drop(columns=['label', 'time'], errors='ignore')
    preds = loaded_model.predict(X_test)
    assert len(preds) == len(df)
    
    # Check predictions of probability
    probs = loaded_model.predict_proba(X_test)[:, 1]
    assert len(probs) == len(df)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_model_trainer_data_windowing_and_challenger(tmp_path):
    # Set random seed
    np.random.seed(42)
    
    # 1. Test Data Windowing based on Date (remove data > 6 months)
    # We will insert 20 old trades (1 year ago) and 40 new trades (recent)
    dates_old = pd.date_range(start="2025-01-01", periods=20, freq="1d").strftime("%Y-%m-%d %H:%M:%S")
    dates_new = pd.date_range(start="2026-05-01", periods=40, freq="1d").strftime("%Y-%m-%d %H:%M:%S")
    all_dates = list(dates_old) + list(dates_new)
    
    data = {
        'time': all_dates,
        'hour': np.random.randint(0, 24, size=60),
        'day_of_week': np.random.randint(0, 7, size=60),
        'setup_type': np.random.randint(0, 2, size=60),
        'direction': np.random.choice([-1, 1], size=60),
        'entry_price': np.random.uniform(2300, 2350, size=60),
        'sl_price': np.random.uniform(2300, 2350, size=60),
        'tp_price': np.random.uniform(2300, 2350, size=60),
        'risk_pips': np.random.uniform(1.0, 10.0, size=60),
        'atr_14': np.random.uniform(1.0, 5.0, size=60),
        'trend': np.random.choice([-1, 1], size=60),
        'killzone': np.random.randint(0, 4, size=60),
        'pnl_relative': np.random.uniform(-1.0, 3.0, size=60),
        'label': np.random.choice([0, 1], size=60)
    }
    
    # Force stratification
    data['label'][:30] = 0
    data['label'][30:] = 1
    
    df = pd.DataFrame(data)
    labeled_data_path = tmp_path / "labeled_setups_window.csv"
    df.to_csv(labeled_data_path, index=False)
    
    # Run retraining
    model_dir = tmp_path / "models_window"
    model, stats = train_xgboost_filter(labeled_data_path=str(labeled_data_path), model_dir=str(model_dir))
    
    # The 20 old trades should be removed by the 6-month window filter. Total remaining should be 40
    # Wait, the threshold for train/test split might need enough samples. With 40 samples it will work fine (test size 20% = 8, train = 32)
    assert stats['dataset_size'] == 40
    
    # Verify the saved CSV has been pruned (only 40 lines remains + header)
    pruned_df = pd.read_csv(labeled_data_path)
    assert len(pruned_df) == 40
    
    # 2. Test Champion vs Challenger Gate
    # First training is completed, Champion is now saved.
    # Now let's train again with the same data - should result in ACCEPTED or similar performance
    _, stats2 = train_xgboost_filter(labeled_data_path=str(labeled_data_path), model_dir=str(model_dir))
    assert 'status' in stats2
    assert stats2['status'] in ['ACCEPTED', 'REJECTED']


def test_window_limit_keeps_latest_setups_by_timestamp_not_csv_order(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_TRAINING_MAX_SETUPS", "1000")
    np.random.seed(42)

    recent_m15 = pd.DataFrame({
        'time': pd.date_range(start="2026-06-01", periods=100, freq="15min").strftime("%Y-%m-%d %H:%M:%S"),
        'timeframe': 15,
    })
    recent_m30 = pd.DataFrame({
        'time': pd.date_range(start="2026-06-02", periods=100, freq="30min").strftime("%Y-%m-%d %H:%M:%S"),
        'timeframe': 30,
    })
    older_h1 = pd.DataFrame({
        'time': pd.date_range(start="2026-01-01", periods=1000, freq="1h").strftime("%Y-%m-%d %H:%M:%S"),
        'timeframe': 60,
    })
    df = pd.concat([recent_m15, recent_m30, older_h1], ignore_index=True)
    rows = len(df)
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
    df['setup_type'] = np.arange(rows) % 2
    df['direction'] = np.where(np.arange(rows) % 2 == 0, 1, -1)
    df['entry_price'] = 2300.0 + (np.arange(rows) % 50)
    df['sl_price'] = df['entry_price'] - df['direction'] * 5.0
    df['tp_price'] = df['entry_price'] + df['direction'] * 10.0
    df['risk_pips'] = 5.0
    df['atr_14'] = 7.5
    df['trend'] = df['direction']
    df['killzone'] = np.arange(rows) % 4
    df['pnl_relative'] = np.where(np.arange(rows) % 2 == 0, 2.0, -1.0)
    df['label'] = np.arange(rows) % 2

    labeled_data_path = tmp_path / "labeled_setups_ordered_by_source.csv"
    model_dir = tmp_path / "models"
    df.to_csv(labeled_data_path, index=False)

    _, stats = train_xgboost_filter(labeled_data_path=str(labeled_data_path), model_dir=str(model_dir))

    pruned_df = pd.read_csv(labeled_data_path)
    assert stats['dataset_size'] == 1000
    assert len(pruned_df) == 1000
    assert {15, 30}.issubset(set(pruned_df['timeframe']))
    assert pd.to_datetime(pruned_df['time']).max() == pd.Timestamp("2026-06-04 01:30:00")


def test_window_limit_can_be_configured_from_env(monkeypatch):
    monkeypatch.setenv("ML_TRAINING_MAX_SETUPS", "1200")
    rows = 1300
    df = pd.DataFrame({
        'time': pd.date_range(start="2026-06-01", periods=rows, freq="15min").strftime("%Y-%m-%d %H:%M:%S"),
        'label': np.arange(rows) % 2,
        'entry_price': 2300.0 + np.arange(rows),
    })

    windowed = _apply_data_windowing(df)

    assert len(windowed) == 1200
    assert pd.to_datetime(windowed['time']).min() == pd.Timestamp("2026-06-02 01:00:00")


def test_load_training_dataset_combines_real_and_shadow_without_polluting_real_csv(tmp_path):
    real_df = pd.DataFrame({
        'time': pd.date_range(start="2026-06-01", periods=4, freq="15min").strftime("%Y-%m-%d %H:%M:%S"),
        'hour': [8, 9, 10, 11],
        'day_of_week': [0, 0, 0, 0],
        'setup_type': [1, 1, 0, 0],
        'direction': [1, -1, 1, -1],
        'entry_price': [2300.0, 2301.0, 2302.0, 2303.0],
        'sl_price': [2295.0, 2306.0, 2297.0, 2308.0],
        'tp_price': [2310.0, 2291.0, 2312.0, 2293.0],
        'risk_pips': [5.0, 5.0, 5.0, 5.0],
        'atr_14': [6.0, 6.1, 6.2, 6.3],
        'trend': [1, -1, 1, -1],
        'killzone': [1, 1, 2, 2],
        'pnl_relative': [2.0, -1.0, 1.8, -0.8],
        'label': [1, 0, 1, 0],
    })
    shadow_df = pd.DataFrame({
        'signal_id': ['shadow-win', 'shadow-loss', 'shadow-invalid'],
        'sample_source': ['shadow', 'shadow', 'shadow'],
        'time': pd.date_range(start="2026-06-02", periods=3, freq="15min").strftime("%Y-%m-%d %H:%M:%S"),
        'hour': [12, 13, 14],
        'day_of_week': [1, 1, 1],
        'setup_type': [1, 0, 1],
        'direction': [1, -1, 1],
        'entry_price': [2310.0, 2311.0, 2312.0],
        'sl_price': [2305.0, 2316.0, 2307.0],
        'tp_price': [2320.0, 2301.0, 2322.0],
        'risk_pips': [5.0, 5.0, 5.0],
        'atr_14': [6.4, 6.5, 6.6],
        'trend': [1, -1, 1],
        'killzone': [2, 3, 3],
        'confidence': [0.34, 0.22, 0.49],
        'accept_threshold': [0.50, 0.50, 0.50],
        'result': ['tp', 'sl', 'expired'],
        'pnl_relative': [2.0, -1.0, None],
        'label': [1, 0, None],
    })
    real_path = tmp_path / "labeled_setups.csv"
    shadow_path = tmp_path / "shadow_labeled_setups.csv"
    real_df.to_csv(real_path, index=False)
    shadow_df.to_csv(shadow_path, index=False)

    combined = load_training_dataset(str(real_path), str(shadow_path))

    assert len(combined) == 6
    assert combined['sample_source'].value_counts().to_dict() == {'real': 4, 'shadow': 2}
    assert set(combined['signal_id'].dropna()) == {'shadow-win', 'shadow-loss'}
    assert pd.read_csv(real_path).columns.tolist() == real_df.columns.tolist()


def test_prepare_training_features_drops_shadow_metadata_and_keeps_numeric_features(tmp_path):
    real_path = tmp_path / "labeled_setups.csv"
    shadow_path = tmp_path / "shadow_labeled_setups.csv"
    pd.DataFrame({
        'time': ['2026-06-01 08:00:00', '2026-06-01 08:15:00'],
        'hour': [8, 8],
        'direction': [1, -1],
        'entry_price': [2300.0, 2301.0],
        'pnl_relative': [2.0, -1.0],
        'label': [1, 0],
    }).to_csv(real_path, index=False)
    pd.DataFrame({
        'signal_id': ['shadow-a', 'shadow-b'],
        'sample_source': ['shadow', 'shadow'],
        'time': ['2026-06-01 08:30:00', '2026-06-01 08:45:00'],
        'hour': [9, 9],
        'direction': [1, -1],
        'entry_price': [2302.0, 2303.0],
        'confidence': [0.31, 0.44],
        'accept_threshold': [0.50, 0.50],
        'result': ['tp', 'sl'],
        'pnl_relative': [2.0, -1.0],
        'label': [1, 0],
    }).to_csv(shadow_path, index=False)

    combined = load_training_dataset(str(real_path), str(shadow_path))
    X, y = prepare_training_features(combined)

    assert set(y.tolist()) == {0, 1}
    assert 'sample_source' not in X.columns
    assert 'signal_id' not in X.columns
    assert 'confidence' not in X.columns
    assert 'accept_threshold' not in X.columns
    assert 'pnl_relative' not in X.columns
    assert X.columns.tolist() == ['hour', 'direction', 'entry_price']


def test_calculate_sample_weights_applies_shadow_multiplier(monkeypatch):
    df = pd.DataFrame({
        'label': [1, 0, 1, 0],
        'pnl_relative': [2.0, -1.0, 2.0, -1.0],
        'sample_source': ['real', 'real', 'shadow', 'shadow'],
    })
    monkeypatch.setenv("ML_SHADOW_SAMPLE_WEIGHT", "0.25")

    weights = calculate_sample_weights(df, df.index)

    assert weights.tolist() == [2.0, 1.5, 0.5, 0.375]


def test_calculate_sample_weights_downweights_bep_plus_wins():
    df = pd.DataFrame({
        'label': [1, 1, 1, 0],
        'pnl_relative': [2.0, 0.026, 0.45, -0.02],
        'sample_source': ['real', 'real', 'real', 'real'],
    })

    weights = calculate_sample_weights(df, df.index)

    assert weights.tolist() == [2.0, 0.5, 1.0, 0.5]


def test_load_training_dataset_allows_missing_shadow_file(tmp_path):
    real_path = tmp_path / "labeled_setups.csv"
    pd.DataFrame({
        'time': ['2026-06-01 08:00:00', '2026-06-01 08:15:00'],
        'hour': [8, 8],
        'direction': [1, -1],
        'entry_price': [2300.0, 2301.0],
        'label': [1, 0],
    }).to_csv(real_path, index=False)

    combined = load_training_dataset(str(real_path), str(tmp_path / "missing_shadow.csv"))

    assert len(combined) == 2
    assert combined['sample_source'].tolist() == ['real', 'real']


def test_champion_gate_failure_does_not_overwrite_existing_models(tmp_path):
    np.random.seed(42)
    rows = 50
    df = pd.DataFrame({
        'time': pd.date_range(start="2026-06-01", periods=rows, freq="15min").strftime("%Y-%m-%d %H:%M:%S"),
        'hour': np.random.randint(0, 24, size=rows),
        'day_of_week': np.random.randint(0, 7, size=rows),
        'setup_type': np.random.randint(0, 2, size=rows),
        'direction': np.random.choice([-1, 1], size=rows),
        'entry_price': np.random.uniform(2300, 2350, size=rows),
        'sl_price': np.random.uniform(2300, 2350, size=rows),
        'tp_price': np.random.uniform(2300, 2350, size=rows),
        'risk_pips': np.random.uniform(1.0, 10.0, size=rows),
        'atr_14': np.random.uniform(1.0, 5.0, size=rows),
        'trend': np.random.choice([-1, 1], size=rows),
        'killzone': np.random.randint(0, 4, size=rows),
        'pnl_relative': np.random.uniform(-1.0, 3.0, size=rows),
        'label': [0] * 25 + [1] * 25,
    })
    labeled_data_path = tmp_path / "labeled_setups.csv"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    df.to_csv(labeled_data_path, index=False)
    joblib.dump(RejectingChampionModel(), model_dir / "smc_xgb_classifier.joblib")
    joblib.dump(RejectingChampionModel(), model_dir / "smc_lgb_classifier.joblib")

    _, stats = train_xgboost_filter(labeled_data_path=str(labeled_data_path), model_dir=str(model_dir))

    assert stats['status'] == 'REJECTED_GATE_ERROR'
    assert isinstance(joblib.load(model_dir / "smc_xgb_classifier.joblib"), RejectingChampionModel)
    assert isinstance(joblib.load(model_dir / "smc_lgb_classifier.joblib"), RejectingChampionModel)
