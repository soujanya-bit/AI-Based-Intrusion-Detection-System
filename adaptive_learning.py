from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

# Suppose 'streamed_batches' yields batches of your data as DataFrames
def streamed_batches():
    for i in range(5):  # Simulate 5 real-time batches
        df = pd.read_csv(f"new_data_batch_{i}.csv")  # Simulate new data
        yield df

# Set up scaler and label encoder
scaler = StandardScaler()
le = LabelEncoder()

clf = SGDClassifier(loss="log_loss", random_state=42)

classes = None  # Will be set after first batch, for partial_fit

for i, batch_df in enumerate(streamed_batches()):
    X = batch_df.drop("label", axis=1)
    y = batch_df["label"]

    # Fit scaler/encoder on first batch, update thereafter
    if i == 0:
        X_scaled = scaler.fit_transform(X)
        y_enc = le.fit_transform(y)
        classes = np.unique(y_enc)
    else:
        X_scaled = scaler.transform(X)
        y_enc = le.transform(y)

    # Adaptively train (partial_fit = online learning)
    clf.partial_fit(X_scaled, y_enc, classes=classes)

    # Optional: Evaluate on current batch
    preds = clf.predict(X_scaled)
    print(f"Batch {i} accuracy: {(preds == y_enc).mean():.4f}")
