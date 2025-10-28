import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import load_model
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress the UserWarning about feature names in StandardScaler
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ---------------- Class Mapping ----------------
# This list is complete and correct, based on your classes.json file.
classes = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest",
    "DoS slowloris", "FTP-Patator", "Heartbleed", "Infiltration", "PortScan",
    "SSH-Patator", "Web Attack - Brute Force", "Web Attack - Sql Injection",
    "Web Attack - XSS"
]

# ---------------- NEW: Function to Plot Confusion Matrix ----------------
def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """
    Generates and saves a confusion matrix plot.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Create a dataframe for a better heatmap
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    
    # Show the plot
    plt.show()

# ---------------- Loading Data ----------------
print("🔹 Loading test dataset...")
df = pd.read_csv("processed/cicids2017_test_sample.csv")

X_test = df.drop("label", axis=1)
y_test = df["label"]

print(f"✅ Shapes -> X_test: {X_test.shape}, y_test: {y_test.shape}")

# Filter the class list to only include classes present in the test set for plotting
active_classes = sorted(list(y_test.unique()))
print(f"🔹 Active classes in this test set: {active_classes}")


# ---------------- Scaling ----------------
print("🔹 Scaling features...")
scaler = load("models/scaler.joblib")
X_test_scaled = scaler.transform(X_test)

# ---------------- Loading Models ----------------
print("🔹 Loading models...")
xgb = load("models/xgboost.joblib")
meta_clf = load("models/meta_classifier.joblib")
dl_model = load_model("models/bilstm_cnn.keras")
rf = load("models/random_forest.joblib")

# ---------------- Test BiLSTM-CNN ----------------
print("\n🚀 Testing BiLSTM-CNN...")
X_test_seq = np.expand_dims(X_test_scaled, axis=-1)
y_pred_dl = dl_model.predict(X_test_seq, verbose=0).argmax(axis=1)
y_pred_dl_labels = [classes[i] for i in y_pred_dl]

print(f"✅ BiLSTM-CNN Accuracy: {accuracy_score(y_test, y_pred_dl_labels):.4f}")
print("🔹 Classification Report (BiLSTM-CNN):")
print(classification_report(y_test, y_pred_dl_labels, zero_division=0))
plot_confusion_matrix(y_test, y_pred_dl_labels, active_classes, "BiLSTM-CNN")


# ---------------- Test XGBoost ----------------
print("\n🚀 Testing XGBoost...")
y_pred_xgb = xgb.predict(X_test_scaled)
y_pred_xgb_labels = [classes[i] for i in y_pred_xgb]

print(f"✅ XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb_labels):.4f}")
print("🔹 Classification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb_labels, zero_division=0))
plot_confusion_matrix(y_test, y_pred_xgb_labels, active_classes, "XGBoost")


# ---------------- Test Meta Classifier ----------------
print("\n🚀 Testing Meta Classifier...")
rf_proba = rf.predict_proba(X_test_scaled)
xgb_proba = xgb.predict_proba(X_test_scaled)
meta_features = np.hstack([rf_proba, xgb_proba])
y_pred_meta = meta_clf.predict(meta_features)
y_pred_meta_labels = [classes[i] for i in y_pred_meta]

print(f"✅ Meta Classifier Accuracy: {accuracy_score(y_test, y_pred_meta_labels):.4f}")
print("🔹 Classification Report (Meta Classifier):")
print(classification_report(y_test, y_pred_meta_labels, zero_division=0))
plot_confusion_matrix(y_test, y_pred_meta_labels, active_classes, "Meta Classifier")