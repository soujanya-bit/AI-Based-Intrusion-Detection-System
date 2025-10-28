import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
INPUT_FILE = "data/malicious_sequences.csv"
OUTPUT_DIR = "models/visualizations"
TIME_STEPS = 10
# These labels must match the order of features used in training
FEATURE_LABELS = [
    'Flow Duration', 'Total Fwd Pkts', 'Total Bwd Pkts',
    'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Flow Bytes/s',
    'Flow Pkts/s', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Pkt Len Mean'
]

# --- Main Execution ---
if __name__ == '__main__':
    try:
        # Load the flattened data and reshape it back into sequences
        data = np.loadtxt(INPUT_FILE, delimiter=",")
        num_features = data.shape[1]
        X = data.reshape(-1, TIME_STEPS, num_features)
        print(f"Loaded {X.shape[0]} malicious sequences from '{INPUT_FILE}'.")
    except FileNotFoundError:
        print(f"Error: '{INPUT_FILE}' not found.")
        print("Run 'capture_and_predict.py' on a PCAP containing attacks to generate it.")
        exit()
    except ValueError as e:
        print(f"Error reshaping data: {e}. Ensure TIME_STEPS is correct.")
        exit()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, seq in enumerate(X):
        fig, ax = plt.subplots(figsize=(12, 7))
        for i in range(num_features):
            ax.plot(seq[:, i], label=FEATURE_LABELS[i], marker='o', linestyle='--')
        
        ax.set_title(f"Malicious Sequence #{idx+1} Feature Visualization")
        ax.set_xlabel("Time Step in Sequence")
        ax.set_ylabel("Normalized Value")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        output_path = os.path.join(OUTPUT_DIR, f"malicious_sequence_{idx+1}.png")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {output_path}")