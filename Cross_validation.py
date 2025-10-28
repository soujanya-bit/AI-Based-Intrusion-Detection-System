import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def cross_validate_model(model, X, y, class_names, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold = 1
    for train_index, val_index in skf.split(X, y):
        print(f"\n--- Fold {fold} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))
        fold += 1

def load_and_prepare_data(file_path, label_column):
    # Load CSV dataset
    df = pd.read_csv(file_path)

    # Separate features and labels
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Handle missing values with imputation (if any)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Feature scaling (recommended for many models, especially neural nets)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    class_names = le.classes_

    return X_scaled, y_encoded, class_names

if __name__ == "__main__":
    # Example file and label column - change these to your dataset paths and label
    dataset_path = "processed/processed_dataset.csv"
    label_col = "label"  # Adjust if label column name differs

    # Load and prepare data
    X, y, class_names = load_and_prepare_data(dataset_path, label_col)

    # Define or load a model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Execute cross-validation
    cross_validate_model(model, X, y, class_names, folds=5)
