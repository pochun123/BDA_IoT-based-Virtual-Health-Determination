# rf_model_wrapper.py（對應你 RF 訓練流程的預測 wrapper）

import numpy as np
import pickle
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft, fftfreq
import pywt
import json
import sys

# ===== 特徵擷取函數（6維） =====
def extract_features(signal, fs=4880):
    features = {}
    features['RMS'] = np.sqrt(np.mean(signal**2))
    features['Kurtosis'] = kurtosis(signal)

    fft_vals = np.abs(fft(signal))[:len(signal)//2]
    fft_freq = fftfreq(len(signal), d=1/fs)[:len(signal)//2]
    features['RMSF'] = np.sqrt(np.sum((fft_freq**2) * (fft_vals**2)) / (np.sum(fft_vals**2) + 1e-12))
    features['Spectral_Kurtosis'] = kurtosis(fft_vals)
    features['Spectral_Entropy'] = entropy(fft_vals / (np.sum(fft_vals) + 1e-12))
    features['Spectral_Energy'] = np.sum(fft_vals**2)  # Total spectral energy
    return np.array(list(features.values()))

# ===== 單筆預測主流程 =====
def pred_rf(file_path, scaler, clf):
    data = np.loadtxt(file_path, skiprows=1)  # 預期為 1D 序列
    if data.ndim == 2 and data.shape[1] == 3:
        signal = data.mean(axis=1)
    else:
        signal = data.flatten()
    
    feats = extract_features(signal)
    feature_names = ["RMS", "Kurtosis", "RMSF", "Spectral_Kurtosis", "Spectral_Entropy", "Spectral_Energy"]
    X = pd.DataFrame(feats.reshape(1, -1), columns=feature_names)
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    pred_health = int(clf.predict(X_scaled)[0])
    prob = float(clf.predict_proba(X_scaled)[0, 1] * 100)

    return {
        "Health_Prediction": pred_health,
        "Health_Probability": round(prob, 2)
    }


