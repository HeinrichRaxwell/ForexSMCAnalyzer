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
        'time', 'timeframe', 'hour', 'day_of_week', 'setup_type', 'direction', 
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

def process_mt5_history_feedback(sent_signals_file="data/sent_signals.json", labeled_data_path="data/labeled_setups.csv"):
    """
    Query MT5 history to check outcomes of sent orders, record wins/losses,
    and trigger retraining if new outcomes are recorded.
    
    Returns:
        int: Number of newly recorded trade outcomes.
    """
    import json
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(sent_signals_file):
        sent_signals_file = os.path.join(base_dir, sent_signals_file)
    if not os.path.isabs(labeled_data_path):
        labeled_data_path = os.path.join(base_dir, labeled_data_path)
        
    if not os.path.exists(sent_signals_file):
        return 0
        
    try:
        with open(sent_signals_file, "r") as f:
            sent_signals = json.load(f)
    except Exception:
        return 0
        
    # Check if MT5 is initialized
    mt5_initialized = True
    if not mt5.initialize():
        mt5_initialized = False
        
    if not mt5_initialized:
        print("[Feedback Loop] Error: MT5 terminal connection not available.")
        return 0
        
    feedback_trades = []
    updated = False
    
    # Query history from 3 days ago to 1 day in future
    from_date = datetime.now() - timedelta(days=3)
    to_date = datetime.now() + timedelta(days=1)
    
    for sig_key, sig_data in sent_signals.items():
        ticket_id = sig_data.get('ticket_id')
        if ticket_id is None or sig_data.get('outcome_recorded', False):
            continue
            
        # 1. Check if still active pending order
        active_orders = mt5.orders_get(ticket=ticket_id)
        if active_orders is not None and len(active_orders) > 0:
            continue  # Still pending, skip
            
        # 2. Check if still open position
        active_positions = mt5.positions_get(ticket=ticket_id)
        if active_positions is not None and len(active_positions) > 0:
            continue  # Still open position, skip
            
        # 3. Not pending, not open. Check if there are deals in history
        # (This history check is relative to position ticket id)
        deals = mt5.history_deals_get(position=ticket_id)
        if deals is None or len(deals) == 0:
            # Order was cancelled or expired without being filled
            sig_data['outcome_recorded'] = True
            updated = True
            print(f"[Feedback Loop] Signal {sig_key} (Ticket #{ticket_id}) was cancelled/expired.")
            continue
            
        # 4. Closed! Calculate net profit
        # (Note: profit can be in account currency, e.g. IDR, USD, etc. We just need > 0 for Win)
        total_profit = sum([d.profit for d in deals if d.position_id == ticket_id])
        label = 1 if total_profit > 0 else 0
        
        # Check if features are stored
        features = sig_data.get('features')
        if features:
            features_dict = features.copy()
            features_dict['label'] = label
            # Map time format
            features_dict['time'] = sig_data.get('time_sent', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            feedback_trades.append(features_dict)
            print(f"[Feedback Loop] Recorded outcome for signal {sig_key}: {'WIN' if label == 1 else 'LOSS'} (Profit: ${total_profit:.2f})")
            
        sig_data['outcome_recorded'] = True
        updated = True
        
    if feedback_trades:
        # Save feedback data
        update_feedback_data(feedback_trades, labeled_data_path)
        # Retrain model
        try:
            trigger_auto_retrain()
        except Exception as e:
            print(f"[Feedback Loop] Auto-retraining error: {e}")
            
    if updated:
        # Save signals registry back
        with open(sent_signals_file, "w") as f:
            json.dump(sent_signals, f, indent=4)
            
    return len(feedback_trades)

