import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    labeled_data_path = "data/labeled_setups.csv"
    model_path = "models/smc_xgb_classifier.joblib"
    lgb_model_path = "models/smc_lgb_classifier.joblib"
    
    if not os.path.exists(labeled_data_path):
        print(f"Error: {labeled_data_path} not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    df = pd.read_csv(labeled_data_path)
    model_xgb = joblib.load(model_path)
    model_lgb = None
    if os.path.exists(lgb_model_path):
        model_lgb = joblib.load(lgb_model_path)
        
    # Standard split to match trainer evaluation
    features = list(model_xgb.feature_names_in_)
    X = df[features]
    y = df['label']
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Predict probabilities
    probs_xgb = model_xgb.predict_proba(X_test)[:, 1]
    if model_lgb is not None:
        probs_lgb = model_lgb.predict_proba(X_test)[:, 1]
        probs = (probs_xgb + probs_lgb) / 2
    else:
        probs = probs_xgb
        
    # Calculate for 0.60, 0.50, 0.40, and 0.30
    thresholds = [0.70, 0.60, 0.50, 0.40, 0.30]
    
    print("=== MODEL PERFORMANCE COMPARISON (TEST SET) ===")
    print(f"Total Test Set Setups: {len(y_test)}")
    print("-" * 65)
    
    for t in thresholds:
        passed_indices = np.where(probs >= t)[0]
        passed_count = len(passed_indices)
        
        if passed_count > 0:
            wins = np.sum(y_test.iloc[passed_indices] == 1)
            winrate = (wins / passed_count) * 100
        else:
            winrate = 0.0
            
        print(f"Threshold: {t:.2f} | Winrate: {winrate:.2f}% | Trades Passed: {passed_count}/{len(y_test)} | Filtered: {len(y_test) - passed_count}")

if __name__ == '__main__':
    main()
