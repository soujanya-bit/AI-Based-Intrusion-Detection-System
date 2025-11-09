# AI-Based-Intrusion-Detection-System
# Overview

This project presents an AI-driven Intrusion Detection System (IDS) designed to detect malicious network activity using deep learning.
It leverages a hybrid LSTM–CNN architecture to capture both temporal and spatial features from network traffic data, improving detection accuracy and robustness against modern cyber threats.

The model is trained and evaluated on the CICIDS2017 dataset, a comprehensive benchmark dataset widely used for network intrusion research.

# Objectives

To develop an intelligent IDS capable of detecting attacks in encrypted and non-encrypted network traffic.

To combine Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for sequential pattern learning.

To achieve high accuracy and low false-positive rates across multiple intrusion types.

# System Architecture
Data Preprocessing → Feature Scaling → LSTM–CNN Model → Classification → Evaluation


1. Data Preprocessing:

Loaded CICIDS2017 dataset and merged daily CSVs.

Encoded categorical features (e.g., protocol type, service).

Applied normalization and one-hot encoding.

Handled class imbalance using oversampling (SMOTE).

2. Model Architecture:

CNN Layers: Capture spatial correlations among network features.

LSTM Layers: Model sequential dependencies in traffic patterns.

Dense Layers: Perform final multi-class classification.

# Technologies Used

Python 3.10

TensorFlow / Keras

Scikit-learn, Pandas, NumPy

Matplotlib / Seaborn (for visualization)

CICIDS2017 Dataset

Evaluation Metrics

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

ROC-AUC

Achieved an overall accuracy of ~99% on the CICIDS2017 test split, outperforming several baseline models (DNN, standalone CNN, standalone LSTM).

Key Features

Hybrid LSTM–CNN architecture for deep temporal-spatial feature learning.

Modular pipeline for data cleaning, feature selection, and model training.

Support for binary and multi-class intrusion detection.

Visual dashboards for model performance and class-level insights.
