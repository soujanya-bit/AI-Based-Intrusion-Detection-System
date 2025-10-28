import os
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils as kutils

warnings.filterwarnings("ignore")

# -----------------------------
# Paths & seed
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root (backend/)
DATA_PATH = ROOT / "processed" / "processed_dataset.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
for p in [REPORTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Utilities
# -----------------------------
def find_label_col(df: pd.DataFrame) -> str:
    """Find the label column robustly (handles weird spacing/case)."""
    norm = {c: str(c).strip().lower() for c in df.columns}
    inv = {v: k for k, v in norm.items()}
    for key in ["label", " class", "attack", "labels"]:
        if key in inv:
            return inv[key]
    # Fallback: try exact 'label' ignoring case/space
    for c in df.columns:
        if str(c).strip().lower() == "label":
            return c
    raise KeyError("Could not find a label column. Checked common variants.")

def make_stats_table(y_true, y_pred, labels_order):
    """Return dict of metrics including macro specificity."""
    # Per-class confusion matrices for specificity (TN/(TN+FP))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    # cm[i,i] = TP_i; row i = actual i; col j = predicted j
    # For class i:
    #   TP = cm[i,i]
    #   FN = sum_j cm[i,j] - TP
    #   FP = sum_j cm[j,i] - TP
    #   TN = total - TP - FN - FP
    totals = cm.sum()
    specificities = []
    for i in range(len(labels_order)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = totals - TP - FN - FP
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificities.append(spec)
    macro_specificity = float(np.mean(specificities))

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall(Sensitivity)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Specificity": macro_specificity,
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, cm

def plot_confusion(cm, classes, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_roc_all(y_true, y_proba, class_names, title, save_path):
    """y_true: integer labels; y_proba: (n_samples, n_classes) probabilities."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))
    # macro + micro
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average ROC (AUC = {roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average ROC (AUC = {roc_auc['macro']:.3f})")

    # per-class
    for i, cname in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], lw=1, label=f"{cname} (AUC = {roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_table_as_csv(metrics_dict, path):
    df = pd.DataFrame(metrics_dict).T
    df.to_csv(path, index=True)

# -----------------------------
# Model builders
# -----------------------------
def build_bilstm_cnn(input_features: int, n_classes: int):
    """1D CNN + BiLSTM for tabular-as-sequence (features as time steps)."""
    inp = layers.Input(shape=(input_features, 1))
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# -----------------------------
# Training script
# -----------------------------
def main():
    print("[INFO] Loading processed CSV ...")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded shape: {df.shape}")

    # Robust label col & cleaning
    label_col = find_label_col(df)
    # Canonicalize class names
    df[label_col] = df[label_col].astype(str).str.strip()

    # Drop any leftover non-numeric columns (they shouldn't exist after preprocessing, but just in case)
    non_numeric = [c for c in df.columns if c != label_col and not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        print(f"[WARNING] Dropping non-numeric columns found at training: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # Remove any rows with infs/NaNs (safety)
    df = df.replace([np.inf, -np.inf], np.nan)
    before = df.shape[0]
    df = df.dropna(axis=0)
    after = df.shape[0]
    if before != after:
        print(f"[INFO] Dropped {before - after} rows containing NaN/Inf before training.")

    # Encode labels to integers
    class_names = sorted(df[label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y = df[label_col].map(class_to_idx).astype(int).values
    X = df.drop(columns=[label_col]).astype(np.float32).values

    print(f"[INFO] Classes ({len(class_names)}): {class_names}")

    # Split: train/val/test = 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    print(f"[INFO] Split -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ------------------ BiLSTM+CNN ------------------
    print("[INFO] Training BiLSTM+CNN ...")
    n_features = X_train_s.shape[1]
    n_classes = len(class_names)

    # reshape to (samples, time_steps=n_features, channels=1)
    tr_seq = np.expand_dims(X_train_s, axis=-1)
    va_seq = np.expand_dims(X_val_s, axis=-1)
    te_seq = np.expand_dims(X_test_s, axis=-1)

    nn_model = build_bilstm_cnn(n_features, n_classes)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    history = nn_model.fit(
        tr_seq, y_train,
        validation_data=(va_seq, y_val),
        epochs=60,
        batch_size=1024,
        callbacks=[es, rlrop],
        verbose=2
    )

    # Convergence plots (loss & accuracy)
    plt.figure(figsize=(7,5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("BiLSTM+CNN Convergence (Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "bilstm_cnn_convergence_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("BiLSTM+CNN Convergence (Accuracy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "bilstm_cnn_convergence_accuracy.png", dpi=200)
    plt.close()

    # Evaluate NN
    nn_proba = nn_model.predict(te_seq, verbose=0)
    nn_pred = np.argmax(nn_proba, axis=1)

    print("[INFO] BiLSTM+CNN Classification Report:")
    print(classification_report(y_test, nn_pred, target_names=class_names, zero_division=0))
    nn_metrics, nn_cm = make_stats_table(y_test, nn_pred, list(range(n_classes)))
    plot_confusion(nn_cm, class_names, "Confusion Matrix - BiLSTM+CNN",
                   REPORTS_DIR / "cm_bilstm_cnn.png")
    plot_roc_all(y_test, nn_proba, class_names, "ROC - BiLSTM+CNN",
                 REPORTS_DIR / "roc_bilstm_cnn.png")

    # Save model
    nn_model.save(MODELS_DIR / "bilstm_cnn.keras")

    # ------------------ Random Forest ------------------
    print("[INFO] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED,
        class_weight=None,  # dataset is balanced enough by preprocessing
    )
    rf.fit(X_train_s, y_train)
    rf_proba = rf.predict_proba(X_test_s)
    rf_pred = np.argmax(rf_proba, axis=1)

    print("[INFO] Random Forest Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=class_names, zero_division=0))
    rf_metrics, rf_cm = make_stats_table(y_test, rf_pred, list(range(n_classes)))
    plot_confusion(rf_cm, class_names, "Confusion Matrix - Random Forest",
                   REPORTS_DIR / "cm_random_forest.png")
    plot_roc_all(y_test, rf_proba, class_names, "ROC - Random Forest",
                 REPORTS_DIR / "roc_random_forest.png")

    # Save RF
    import joblib
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    # ------------------ XGBoost ------------------
    print("[INFO] Training XGBoost ...")
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        tree_method="hist",
        learning_rate=0.1,
        n_estimators=400,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=SEED,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    eval_set = [(X_train_s, y_train), (X_val_s, y_val)]
    xgb.fit(X_train_s, y_train, eval_set=eval_set, verbose=False)
    # Convergence analysis for XGB (eval metric vs trees)
    results = xgb.evals_result()
    train_logloss = results["validation_0"]["mlogloss"]
    val_logloss = results["validation_1"]["mlogloss"]
    plt.figure(figsize=(7,5))
    plt.plot(train_logloss, label="train_mlogloss")
    plt.plot(val_logloss, label="val_mlogloss")
    plt.xlabel("Boosting Round")
    plt.ylabel("mlogloss")
    plt.title("XGBoost Convergence (mlogloss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "xgb_convergence_mlogloss.png", dpi=200)
    plt.close()

    xgb_proba = xgb.predict_proba(X_test_s)
    xgb_pred = np.argmax(xgb_proba, axis=1)

    print("[INFO] XGBoost Classification Report:")
    print(classification_report(y_test, xgb_pred, target_names=class_names, zero_division=0))
    xgb_metrics, xgb_cm = make_stats_table(y_test, xgb_pred, list(range(n_classes)))
    plot_confusion(xgb_cm, class_names, "Confusion Matrix - XGBoost",
                   REPORTS_DIR / "cm_xgboost.png")
    plot_roc_all(y_test, xgb_proba, class_names, "ROC - XGBoost",
                 REPORTS_DIR / "roc_xgboost.png")

    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")

    # ------------------ Meta-classifier (Stacking) ------------------
    # Train on validation set using RF + XGB probabilities as features.
    print("[INFO] Training Meta-Classifier (Logistic Regression on stacked probs) ...")
    rf_val_proba = rf.predict_proba(X_val_s)
    xgb_val_proba = xgb.predict_proba(X_val_s)
    stack_val = np.hstack([rf_val_proba, xgb_val_proba])  # shape: (n_val, 2*n_classes)

    meta = LogisticRegression(
        multi_class="multinomial",
        max_iter=200,
        n_jobs=None,
        random_state=SEED
    )
    meta.fit(stack_val, y_val)

    rf_test_proba = rf.predict_proba(X_test_s)
    xgb_test_proba = xgb.predict_proba(X_test_s)
    stack_test = np.hstack([rf_test_proba, xgb_test_proba])
    meta_proba = kutils.to_categorical(meta.predict(stack_test), num_classes=n_classes)
    # We also want probabilities; approximate using predict_proba if available
    if hasattr(meta, "predict_proba"):
        meta_proba = meta.predict_proba(stack_test)
    meta_pred = np.argmax(meta_proba, axis=1)

    print("[INFO] Meta-Classifier Classification Report:")
    print(classification_report(y_test, meta_pred, target_names=class_names, zero_division=0))
    meta_metrics, meta_cm = make_stats_table(y_test, meta_pred, list(range(n_classes)))
    plot_confusion(meta_cm, class_names, "Confusion Matrix - Meta-Classifier",
                   REPORTS_DIR / "cm_meta.png")
    plot_roc_all(y_test, meta_proba, class_names, "ROC - Meta-Classifier",
                 REPORTS_DIR / "roc_meta.png")

    joblib.dump(meta, MODELS_DIR / "meta_classifier.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # ------------------ Final table ------------------
    metrics_table = {
        "BiLSTM+CNN": nn_metrics,
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
        "Meta-Classifier": meta_metrics,
    }
    print("[INFO]\nFinal Metrics Table:")
    df_metrics = pd.DataFrame(metrics_table).T
    pd.set_option("display.max_colwidth", None)
    print(df_metrics.round(6))

    save_table_as_csv(metrics_table, REPORTS_DIR / "metrics_table.csv")

    # Save per-model classification reports as text
    with open(REPORTS_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("=== BiLSTM+CNN ===\n")
        f.write(classification_report(y_test, nn_pred, target_names=class_names, zero_division=0))
        f.write("\n\n=== Random Forest ===\n")
        f.write(classification_report(y_test, rf_pred, target_names=class_names, zero_division=0))
        f.write("\n\n=== XGBoost ===\n")
        f.write(classification_report(y_test, xgb_pred, target_names=class_names, zero_division=0))
        f.write("\n\n=== Meta-Classifier ===\n")
        f.write(classification_report(y_test, meta_pred, target_names=class_names, zero_division=0))

    # Also save mapping of indices to names
    with open(REPORTS_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"index_to_class": {i: c for c, i in class_to_idx.items()}}, f, indent=2)

    print(f"[INFO] Saved reports to: {REPORTS_DIR}")
    print(f"[INFO] Saved models to: {MODELS_DIR}")

if __name__ == "__main__":
    main()
