import pandas as pd

# Load your processed CICIDS2017 test dataset file (CSV or pickle)
test_df = pd.read_csv("processed/processed_dataset.csv")  # or your actual test data path

# Randomly sample 1000 rows for quick testing
test_sample = test_df.sample(n=1000, random_state=42)

# Save sample to a new CSV
test_sample.to_csv("processed/cicids2017_test_sample.csv", index=False)

print("Saved CICIDS2017 test sample for quick evaluation.")
