import os
import sys
import pandas as pd
import numpy as np
import pytest
import joblib
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import predict_setup_probability, update_feedback_data, trigger_auto_retrain
from src.model_trainer import train_xgboost_filter

def test_predict_setup_probability(tmp_path):
    # Train a dummy model first
    np.random.seed(42)
    data = {
        'time': pd.date_range(start="2026-06-01", periods=10, freq="15min").strftime("%Y-%m-%d %H:%M:%S"),
        'hour': np.random.randint(0, 24, size=10),
        'day_of_week': np.random.randint(0, 7, size=10),
        'setup_type': np.random.randint(0, 2, size=10),
        'direction': np.random.choice([-1, 1], size=10),
        'entry_price': np.random.uniform(2300, 2350, size=10),
        'sl_price': np.random.uniform(2300, 2350, size=10),
        'tp_price': np.random.uniform(2300, 2350, size=10),
        'risk_pips': np.random.uniform(1.0, 10.0, size=10),
        'atr_14': np.random.uniform(1.0, 5.0, size=10),
        'trend': np.random.choice([-1, 1], size=10),
        'killzone': np.random.randint(0, 4, size=10),
        'label': np.random.choice([0, 1], size=10)
    }
    # Force some variation to avoid issues during stratification
    data['label'][:5] = 0
    data['label'][5:] = 1
    
    df = pd.DataFrame(data)
    labeled_data_path = tmp_path / "labeled_setups.csv"
    df.to_csv(labeled_data_path, index=False)
    
    model_dir = tmp_path / "models"
    train_xgboost_filter(labeled_data_path=str(labeled_data_path), model_dir=str(model_dir))
    
    model_path = model_dir / "smc_xgb_classifier.joblib"
    
    # Test predict_setup_probability with single dict
    single_setup = {
        'hour': 8,
        'day_of_week': 2,
        'setup_type': 1,
        'direction': 1,
        'entry_price': 2315.5,
        'sl_price': 2310.0,
        'tp_price': 2326.5,
        'risk_pips': 5.5,
        'atr_14': 2.3,
        'trend': 1,
        'killzone': 0
    }
    
    prob = predict_setup_probability(single_setup, model_path=str(model_path))
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    
    # Test predict_setup_probability with list of dicts
    multiple_setups = [single_setup, single_setup]
    probs = predict_setup_probability(multiple_setups, model_path=str(model_path))
    assert isinstance(probs, list)
    assert len(probs) == 2
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in probs)

def test_update_feedback_data(tmp_path):
    labeled_data_path = tmp_path / "test_feedback.csv"
    
    trade = {
        'time': '2026-06-04 12:00:00',
        'hour': 12,
        'day_of_week': 3,
        'setup_type': 0,
        'direction': -1,
        'entry_price': 2345.0,
        'sl_price': 2348.0,
        'tp_price': 2339.0,
        'risk_pips': 3.0,
        'atr_14': 1.8,
        'trend': -1,
        'killzone': 2,
        'label': 1
    }
    
    # Update first time (creates file)
    update_feedback_data(trade, labeled_data_path=str(labeled_data_path))
    assert os.path.exists(labeled_data_path)
    
    df = pd.read_csv(labeled_data_path)
    assert len(df) == 1
    assert list(df.columns) == [
        'time', 'timeframe', 'hour', 'day_of_week', 'setup_type', 'direction', 
        'entry_price', 'sl_price', 'tp_price', 'risk_pips', 'atr_14', 
        'trend', 'relative_risk', 'killzone', 'fvg_width', 'relative_fvg_width', 'label'
    ]
    assert df.iloc[0]['hour'] == 12
    assert df.iloc[0]['label'] == 1
    
    # Update second time (appends file)
    trades_list = [trade, trade]
    update_feedback_data(trades_list, labeled_data_path=str(labeled_data_path))
    df_updated = pd.read_csv(labeled_data_path)
    assert len(df_updated) == 3

def test_trigger_auto_retrain():
    with patch('src.model_trainer.train_xgboost_filter') as mock_train:
        trigger_auto_retrain()
        mock_train.assert_called_once()
