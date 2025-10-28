# reporting_and_analysis.py
# ------------------------------------------------------------
# Generates reports & graphs for all models using saved files
# ------------------------------------------------------------
import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from itertools import cycle
import shap
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
MODEL_DIR = "models"
DATA_PATH = "processed/processed_dataset.csv"   # path to your dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# ---------------- Load Artifacts ----------------
print("--- Loading models & artifacts ---")
bilstm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "bilstm_cnn.keras"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.joblib"))
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
meta = joblib.load(os.path.join(MODEL_DIR, "meta_classifier.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

with open(os.path.join(MODEL_DIR, "feature_names.json"), "r") as f:
    feature_names = json.load(f)

with open(os.path.join(MODEL_DIR, "training_history.json"), "r") as f:
    history = json.load(f)

# ---------------- Prepare Test Data ----------------
print("--- Preparing dataset ---")
df = pd.read_csv(DATA_PATH)
label_col = [c for c in df.columns if c.lower().strip() in ["label", "class"]][0]
class_names = sorted(df[label_col].unique().tolist())

# Encode labels
class_to_idx = {c: i for i, c in enumerate(class_names)}
y = df[label_col].map(class_to_idx).astype(int).values
X = df.drop(columns=[label_col]).astype(np.float32).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Scale
X_test_s = scaler.transform(X_test)

# BiLSTM input
X_test_seq = np.expand_dims(X_test_s, axis=-1)

# ---------------- Predictions ----------------
print("--- Running predictions ---")
proba_bilstm = bilstm.predict(X_test_seq, verbose=0)
proba_xgb = xgb.predict_proba(X_test_s)
proba_rf = rf.predict_proba(X_test_s)

stacked = np.hstack([proba_bilstm, proba_xgb, proba_rf])
proba_meta = meta.predict_proba(stacked)
pred_meta = meta.predict(stacked)

# Reports
report_bilstm = classification_report(y_test, np.argmax(proba_bilstm, axis=1), target_names=class_names, output_dict=True, zero_division=0)
report_xgb = classification_report(y_test, np.argmax(proba_xgb, axis=1), target_names=class_names, output_dict=True, zero_division=0)
report_rf = classification_report(y_test, np.argmax(proba_rf, axis=1), target_names=class_names, output_dict=True, zero_division=0)
report_meta = classification_report(y_test, pred_meta, target_names=class_names, output_dict=True, zero_division=0)

# ---------------- Graphs ----------------
def plot_performance_comparison():
    metrics = ['precision', 'recall', 'f1-score']
    models = ['CNN+BiLSTM', 'XGBoost', 'Random Forest', 'Meta-Classifier']
    scores = {
        'CNN+BiLSTM': [report_bilstm['weighted avg'][m] for m in metrics],
        'XGBoost': [report_xgb['weighted avg'][m] for m in metrics],
        'Random Forest': [report_rf['weighted avg'][m] for m in metrics],
        'Meta-Classifier': [report_meta['weighted avg'][m] for m in metrics]
    }
    x = np.arange(len(metrics))
    width = 0.2
    plt.figure(figsize=(10,6))
    for i, model in enumerate(models):
        plt.bar(x + i*width - width*1.5, scores[model], width, label=model)
    plt.xticks(x, [m.title() for m in metrics]); plt.ylabel("Score")
    plt.title("Overall Model Performance (Weighted Avg)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "performance_comparison.png"))
    plt.close()

def plot_confusion_matrix():
    cm = confusion_matrix(y_test, pred_meta)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Meta-Classifier)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.close()

def plot_multiclass_roc():
    y_bin = label_binarize(y_test, classes=range(len(class_names)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba_meta[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10,8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], lw=2, color=color,
                 label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],'k--'); plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Meta-Classifier)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"))
    plt.close()

def plot_convergence():
    plt.figure(figsize=(10,6))
    plt.plot(history['accuracy'], label="Train Acc")
    plt.plot(history['val_accuracy'], label="Val Acc")
    plt.plot(history['loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Val Loss")
    plt.title("CNN+BiLSTM Training Convergence")
    plt.xlabel("Epochs"); plt.ylabel("Value")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "convergence_analysis.png"))
    plt.close()

def plot_shap_summary():
    print("⏳ Running SHAP...")
    X_test_df = pd.DataFrame(X_test_s, columns=feature_names)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test_df.sample(1000, random_state=42))
    shap.summary_plot(shap_values, X_test_df.sample(1000, random_state=42),
                      class_names=class_names, show=False)
    plt.title("Feature Importance (SHAP on XGBoost)")
    plt.savefig(os.path.join(MODEL_DIR, "shap_summary.png"), bbox_inches="tight")
    plt.close()

# ---------------- Run All ----------------
if __name__ == "__main__":
    plot_performance_comparison()
    plot_confusion_matrix()
    plot_multiclass_roc()
    plot_convergence()
    try:
        plot_shap_summary()
    except Exception as e:
        print("❌ SHAP failed:", e)
    print("✅ All reports saved in models/ folder.")
