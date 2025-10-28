# src/predict.py
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "processed" / "processed_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# -----------------------------
# Load test split
# -----------------------------
def load_test_split():
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH)
    label_col = [c for c in df.columns if c.lower().strip() in ["label", "class"]][0]

    # Keep original labels for reporting
    orig_labels = sorted(df[label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(orig_labels)}

    # Encode target
    y = df[label_col].map(class_to_idx).astype(int).values
    X = df.drop(columns=[label_col]).astype(np.float32).values

    SEED = 42
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )

    # Load same scaler from training
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_test_s = scaler.transform(X_test)

    return X_test_s, y_test, orig_labels   # return original string labels

# -----------------------------
# Prediction with all models
# -----------------------------
def main():
    print("[INFO] Loading scaler + models ...")
    X_test_s, y_test, class_names = load_test_split()

    # Load models
    bilstm = tf.keras.models.load_model(MODELS_DIR / "bilstm_cnn.keras")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")
    meta = joblib.load(MODELS_DIR / "meta_classifier.joblib")

    # Prepare BiLSTM input shape
    te_seq = np.expand_dims(X_test_s, axis=-1)

    # Predictions
    print("[INFO] Running predictions on TEST split only ...")

    bilstm_pred = np.argmax(bilstm.predict(te_seq, verbose=0), axis=1)
    rf_pred = rf.predict(X_test_s)
    xgb_pred = xgb.predict(X_test_s)

    # Meta classifier needs stacked probabilities
    rf_proba = rf.predict_proba(X_test_s)
    xgb_proba = xgb.predict_proba(X_test_s)
    bilstm_proba = bilstm.predict(te_seq, verbose=0)
    stack_test = np.hstack([rf_proba, xgb_proba, bilstm_proba])
    meta_pred = meta.predict(stack_test)

    # Reports
    print("\n=== BiLSTM+CNN ===")
    print(classification_report(y_test, bilstm_pred, target_names=class_names, zero_division=0))

    print("\n=== Random Forest ===")
    print(classification_report(y_test, rf_pred, target_names=class_names, zero_division=0))

    print("\n=== XGBoost ===")
    print(classification_report(y_test, xgb_pred, target_names=class_names, zero_division=0))

    print("\n=== Meta-Classifier ===")
    print(classification_report(y_test, meta_pred, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    main()
