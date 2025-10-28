import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scapy.all import rdpcap, IP, UDP, Raw
from tensorflow import keras
import joblib
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import random
import time
import json
from feature_extractor import extract_flow_features

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pcap', 'pcapng'}
PREDICTION_THRESHOLD = 0.5
TIME_STEPS = 10
SHAP_JSON_PATH = "models/shap_top_features.json"
SHAP_PLOT_PATH = "models/visualizations/shap_summary.png"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# --- Load model/scaler ---
try:
    scaler = joblib.load('models/scaler.pkl')
    model = keras.models.load_model('models/lstm_cnn_model.keras')
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model or scaler not loaded. Please run 'preprocess_and_evaluate.py' first.")
    scaler = None
    model = None

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_feature_dataframe(df):
    """Run model predictions and return results."""
    print(f"🧠 Analyzing {len(df)} flows...")
    if df.empty:
        raise ValueError('No valid flows were provided for analysis.')

    feature_names = scaler.get_feature_names_out()
    df_reindexed = df.reindex(columns=feature_names, fill_value=0)
    df_reindexed.replace([np.inf, -np.inf], 0, inplace=True)
    X_scaled = scaler.transform(df_reindexed).astype(np.float32)

    if len(X_scaled) < TIME_STEPS:
        raise ValueError(f'Not enough data for a full sequence. Found {len(X_scaled)} flows, but need at least {TIME_STEPS}.')

    X_seq = np.array([X_scaled[i:i + TIME_STEPS] for i in range(len(X_scaled) - TIME_STEPS + 1)])
    predictions = model.predict(X_seq, verbose=0)

    results = []
    for i, p in enumerate(predictions):
        label = 'Attack' if p[0] > PREDICTION_THRESHOLD else 'BENIGN'
        flow_data = df.iloc[i + TIME_STEPS - 1].to_dict()
        results.append({
            'id': i + 1,
            'label': label,
            'confidence': f"{p[0]:.4f}",
            'flow_data': {k: f"{v:.2f}" for k, v in flow_data.items()}
        })
    print("✅ Analysis complete.")
    return results, X_seq, feature_names

def generate_shap_from_df(X_seq, feature_names):
    """Generate SHAP values for sequence data (flattened for KernelExplainer)."""
    os.makedirs("models/visualizations", exist_ok=True)
    print("🔍 Generating SHAP values... (KernelExplainer)")

    # Flatten 3D (samples, timesteps, features) → 2D (samples, timesteps*features)
    num_samples, time_steps, num_features = X_seq.shape
    X_flat = X_seq.reshape(num_samples, time_steps * num_features)

    # Create "expanded" feature names for flattened array
    expanded_feature_names = [
        f"{fname}_t{t+1}" for t in range(time_steps) for fname in feature_names
    ]

    # Use a smaller background sample for speed
    background_size = min(50, len(X_flat))
    background = X_flat[np.random.choice(X_flat.shape[0], background_size, replace=False)]

    # KernelExplainer
    explainer = shap.KernelExplainer(lambda x: model.predict(x.reshape(-1, time_steps, num_features)), background)
    shap_values = explainer.shap_values(background, nsamples=100)

    # Mean importance across flattened features
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    total_importance = np.sum(mean_shap_values)

    shap_importance = sorted(
        zip(expanded_feature_names, mean_shap_values),
        key=lambda x: x[1],
        reverse=True
    )

    shap_top_features = [
    {
        "feature": name,
        "importance": round(float(val) / total_importance * 100, 2)
    }
    for name, val in shap_importance
]


    with open(SHAP_JSON_PATH, "w") as f:
        json.dump(shap_top_features, f, indent=4)
    print(f"✅ SHAP JSON saved to {SHAP_JSON_PATH}")

    # Save summary plot
    shap.summary_plot(shap_values, background, feature_names=expanded_feature_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=300)
    plt.close()
    print(f"📊 SHAP summary plot saved to {SHAP_PLOT_PATH}")

# --- API Routes ---
@app.route('/predict', methods=['POST'])
def predict_pcap():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded on the server.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            print(f"📄 Analyzing uploaded file: {filename}")
            packets = rdpcap(filepath)
            df = extract_flow_features(packets)
            results, X_seq, feature_names = analyze_feature_dataframe(df)

            # Auto-generate SHAP
            generate_shap_from_df(X_seq, feature_names)

            return jsonify({'results': results, 'filename': filename})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type.'}), 400

def generate_benign_features():
    print("🔄 Generating benign traffic for simulation...")
    num_benign_flows = 50
    benign_packets = []
    start_time = time.time()
    duration = 3 
    for i in range(num_benign_flows):
        client = f"172.16.0.{random.randint(1,10)}"
        server = f"10.10.10.{random.randint(1,10)}"
        pkt1 = IP(src=client, dst=server) / UDP(dport=53) / Raw(load="areyou-there.com")
        pkt1.time = start_time + random.uniform(0, duration) 
        pkt2 = IP(src=server, dst=client) / UDP(sport=53) / Raw(load="yes-i-am-here.com")
        pkt2.time = pkt1.time + random.uniform(0.05, 0.1)
        benign_packets.extend([pkt1, pkt2])
    
    benign_packets.sort(key=lambda p: p.time)
    return extract_flow_features(benign_packets)

def generate_injected_attack_features():
    print("🔄 Generating injected attack features...")
    attack_flows = []
    
    for _ in range(15):  # DDoS
        attack_flows.append({
            'Flow Duration': random.uniform(0.001, 0.1), 'Total Fwd Packets': random.randint(200, 600),
            'Total Backward Packets': 0, 'Fwd Packet Length Mean': random.uniform(1000, 1400),
            'Bwd Packet Length Mean': 0, 'Flow Bytes/s': 25000000.0, 'Flow Packets/s': 2500000.0,
            'Fwd IAT Mean': random.uniform(0.00001, 0.0001), 'Bwd IAT Mean': 0,
            'Packet Length Mean': random.uniform(1000, 1400)
        })
        
    for _ in range(15):  # Port Scan
        attack_flows.append({
            'Flow Duration': random.uniform(0.0001, 0.01), 'Total Fwd Packets': 1,
            'Total Backward Packets': 1, 'Fwd Packet Length Mean': random.uniform(20, 40),
            'Bwd Packet Length Mean': random.uniform(20, 40), 'Flow Bytes/s': random.uniform(6000, 30000),
            'Flow Packets/s': random.uniform(600, 3000), 'Fwd IAT Mean': 0, 'Bwd IAT Mean': 0,
            'Packet Length Mean': random.uniform(20, 40)
        })

    for _ in range(10):  # Brute Force
        attack_flows.append({
            'Flow Duration': random.uniform(2.0, 5.0), 'Total Fwd Packets': random.randint(10, 20),
            'Total Backward Packets': random.randint(5, 15), 'Fwd Packet Length Mean': random.uniform(10, 30),
            'Bwd Packet Length Mean': random.uniform(10, 30), 'Flow Bytes/s': random.uniform(100, 500),
            'Flow Packets/s': random.uniform(5, 15), 'Fwd IAT Mean': random.uniform(100000, 200000),
            'Bwd IAT Mean': random.uniform(100000, 200000), 'Packet Length Mean': random.uniform(10, 30)
        })
        
    return pd.DataFrame(attack_flows)

@app.route('/simulate', methods=['POST'])
def simulate_and_analyze():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded on the server.'}), 500

    try:
        print("\n🚀 Starting new simulation...")
        benign_df = generate_benign_features()
        attack_df = generate_injected_attack_features()
        
        combined_df = pd.concat([benign_df, attack_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        
        filename = "Simulated Attack Scenario"
        results, X_seq, feature_names = analyze_feature_dataframe(combined_df)

        # Auto-generate SHAP
        generate_shap_from_df(X_seq, feature_names)

        return jsonify({'results': results, 'filename': filename})

    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    try:
        with open('models/feature_importance.json', 'r') as f:
            importance_data = json.load(f)
        return jsonify(importance_data)
    except FileNotFoundError:
        return jsonify({'error': 'Feature importance file not found. Please retrain the model.'}), 500

@app.route('/shap/top-features', methods=['GET'])
def get_shap_top_features():
    try:
        with open(SHAP_JSON_PATH, 'r') as f:
            shap_data = json.load(f)
        return jsonify(shap_data)
    except FileNotFoundError:
        return jsonify({'error': 'SHAP feature file not found. Please run prediction or simulation first.'}), 500

@app.route('/shap/summary-plot', methods=['GET'])
def get_shap_summary_plot():
    try:
        return send_file(SHAP_PLOT_PATH, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({'error': 'SHAP summary plot not found. Please run prediction or simulation first.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
