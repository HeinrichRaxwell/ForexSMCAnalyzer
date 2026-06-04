import os
import sys
import pandas as pd
import numpy as np
import pytest
import joblib

# Add project root to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import train_xgboost_filter

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
    model = train_xgboost_filter(labeled_data_path=str(labeled_data_path), model_dir=str(model_dir))
    
    # Assert model was returned and file is saved
    assert model is not None
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
