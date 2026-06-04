import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# Add project root to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_xgboost_filter(labeled_data_path="data/labeled_setups.csv", model_dir="models"):
    """
    Train an XGBoost classifier to filter out low-probability SMC trade setups.
    
    Args:
        labeled_data_path (str): Path to the labeled CSV dataset.
        model_dir (str): Directory where the trained model should be saved.
        
    Returns:
        XGBClassifier: The trained XGBoost model.
    """
    # Resolve paths relative to project root if they are relative
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isabs(labeled_data_path):
        labeled_data_path = os.path.join(base_dir, labeled_data_path)
        
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(base_dir, model_dir)
        
    print(f"Loading labeled data from {labeled_data_path}...")
    if not os.path.exists(labeled_data_path):
        raise FileNotFoundError(f"Labeled setups file not found at {labeled_data_path}")
        
    df = pd.read_csv(labeled_data_path)
    print(f"Loaded {len(df)} setups.")
    
    # Separate features and target
    X = df.drop(columns=['label', 'time'], errors='ignore')
    y = df['label']
    
    print(f"Features: {list(X.columns)}")
    print(f"Target: label")
    
    # Split the data into training and test sets (80% train, 20% test) stratified by target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Train the XGBClassifier
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Evaluate model metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*40)
    print("        CLASSIFICATION REPORT        ")
    print("="*40)
    print(classification_report(y_test, y_pred))
    
    print("\n" + "="*60)
    print("     WINRATE AT DIFFERENT CONFIDENCE THRESHOLDS     ")
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
    
    # Save the trained model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "smc_xgb_classifier.joblib")
    joblib.dump(model, model_path)
    print(f"Trained model saved successfully to: {model_path}")
    
    return model

if __name__ == "__main__":
    train_xgboost_filter()
