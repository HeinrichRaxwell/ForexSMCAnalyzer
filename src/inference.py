import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add project root to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def predict_setup_probability(features_dict, model_path="models/smc_xgb_classifier.joblib"):
    """
    Load the trained model and predict the probability of success for a given setup.
    
    Args:
        features_dict (dict or list of dicts): Setup features dictionary or list of dictionaries.
        model_path (str): Path to the trained joblib model.
        
    Returns:
        float or list of floats: Probability of success (label = 1).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(model_path):
        model_path = os.path.join(base_dir, model_path)
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model = joblib.load(model_path)
    
    is_list = isinstance(features_dict, list)
    if not is_list:
        features_dict = [features_dict]
        
    df = pd.DataFrame(features_dict)
    
    # Ensure correct feature names and order
    expected_features = list(model.feature_names_in_)
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected features: {missing_features}")
        
    # Reorder columns to match feature_names_in_
    X = df[expected_features]
    
    # Predict probabilities for class 1 (success)
    probs = model.predict_proba(X)[:, 1]
    
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
    
    # Standard columns order for labeled_setups.csv
    columns_order = [
        'time', 'hour', 'day_of_week', 'setup_type', 'direction', 
        'entry_price', 'sl_price', 'tp_price', 'risk_pips', 'atr_14', 
        'trend', 'relative_risk', 'killzone', 'fvg_width', 'relative_fvg_width', 'label'
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

def trigger_auto_retrain():
    """
    Retrain the XGBoost model with the updated dataset.
    """
    from src.model_trainer import train_xgboost_filter
    print("Triggering automatic model retraining...")
    return train_xgboost_filter()
