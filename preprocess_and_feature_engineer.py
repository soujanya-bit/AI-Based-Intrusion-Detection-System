import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "processed_dataset.csv")
SCALER_FILE = os.path.join(PROCESSED_DIR, "scaler.joblib")
ENCODER_FILE = os.path.join(PROCESSED_DIR, "label_encoder.joblib")
FEATURES_FILE = os.path.join(PROCESSED_DIR, "feature_names.csv")

CSV_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

def load_data():
    print("[INFO] Loading CSV files...")
    dfs = []
    for file in CSV_FILES:
        file_path = os.path.join(DATA_DIR, file)
        print(f"[INFO] Loading {file_path} ...")
        try:
            dfs.append(pd.read_csv(file_path))
        except UnicodeDecodeError:
            print(f"[WARNING] UTF-8 failed for {file_path}, retrying with latin1...")
            dfs.append(pd.read_csv(file_path, encoding="latin1"))
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined dataset shape: {df.shape}")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Starting preprocessing...")
    df.columns = df.columns.str.strip().str.lower()
    drop_cols = [c for c in ["flow id", "source ip", "destination ip", "timestamp"] if c in df.columns]
    if drop_cols:
        print(f"[INFO] Dropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols, errors="ignore")
    df = df.drop_duplicates()
    print(f"[INFO] After dropping duplicates: {df.shape}")
    if "label" not in df.columns:
        raise ValueError("ERROR: 'label' column not found in dataset!")
    df["label"] = df["label"].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna() # Optionally, use df.fillna(0) instead to keep rare samples
    # Convert all except "label" to numeric
    numeric_cols = df.columns.drop("label")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    print("[INFO] Preprocessing complete.")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Starting feature engineering...")
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols, errors="ignore")
    print("[INFO] Feature engineering complete.")
    return df

def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Checking class balance...")
    print("[INFO] Class distribution before balancing:\n", df["label"].value_counts())
    X = df.drop(columns=["label"])
    y = df["label"]
    if "BENIGN" in y.unique():
        rus = RandomUnderSampler(sampling_strategy={"BENIGN": 200000}, random_state=42)
        X, y = rus.fit_resample(X, y)
    class_counts = Counter(y)
    sampling_strategy = {}
    for cls, count in class_counts.items():
        if count < 1000:
            sampling_strategy[cls] = 1000
        else:
            sampling_strategy[cls] = count
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    df_balanced = pd.DataFrame(X_res, columns=X.columns)
    df_balanced["label"] = y_res
    print("[INFO] Class distribution after balancing:\n", df_balanced["label"].value_counts())
    return df_balanced

def scale_and_encode(df: pd.DataFrame):
    print("[INFO] Scaling features and encoding labels...")
    feature_cols = [c for c in df.columns if c != "label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, SCALER_FILE)
    print(f"[INFO] Saved scaler: {SCALER_FILE}")
    le = LabelEncoder()
    y_encoded = le.fit_transform(df["label"])
    joblib.dump(le, ENCODER_FILE)
    print(f"[INFO] Saved label encoder: {ENCODER_FILE}")
    pd.Series(feature_cols).to_csv(FEATURES_FILE, index=False, header=False)
    print(f"[INFO] Saved feature column names: {FEATURES_FILE}")
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    df_scaled["label"] = y_encoded
    return df_scaled, le.classes_

def main():
    df = load_data()
    df = preprocess(df)
    df = feature_engineering(df)
    df = balance_classes(df)
    df_scaled, class_names = scale_and_encode(df)
    df_scaled.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved balanced and scaled dataset to {OUTPUT_FILE}")
    print(f"[INFO] Classes (label encoding order): {class_names}")

if __name__ == "__main__":
    main()
