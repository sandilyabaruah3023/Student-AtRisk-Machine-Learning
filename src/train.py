# src/train.py
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

from load_data import load_oulad
from preprocess import merge_core
from features import simple_feature_engineer

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_pipeline(data_dir: str = "../OULAD", random_state: int = 42):
    dfs = load_oulad(data_dir)
    merged = merge_core(dfs)
    X, y = simple_feature_engineer(merged)

    # remove rows with no features
    if X.shape[1] == 0:
        raise RuntimeError("No features created. Inspect feature engineering.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    model_path = MODEL_DIR / "rf_at_risk.joblib"
    joblib.dump({
        "model": clf,
        "features": X.columns.tolist(),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }, model_path)

    print("Saved model and test data to", model_path)
    return model_path

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../OULAD"
    train_pipeline(data_dir)
