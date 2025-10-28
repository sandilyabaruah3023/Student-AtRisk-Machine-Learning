# src/evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from load_data import load_oulad
from preprocess import merge_core
from features import simple_feature_engineer
from pathlib import Path

def evaluate_model(model_path: str, data_dir: str = "../OULAD"):
    model_art = joblib.load(model_path)
    model = model_art['model']
    feat_list = model_art.get('features', None)

    dfs = load_oulad(data_dir)
    merged = merge_core(dfs)
    X, y = simple_feature_engineer(merged)

    # Align columns in case of missing columns
    if feat_list:
        for c in feat_list:
            if c not in X.columns:
                X[c] = 0
        X = X[feat_list]

    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    import sys
    model_p = sys.argv[1] if len(sys.argv) > 1 else "../models/rf_at_risk.joblib"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "../OULAD"
    evaluate_model(model_p, data_dir)
