import streamlit as st
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from CNN_package import CNN3D, CNN3D_con_coord, pred_con_cnn, pred_dis_cnn
from RF_package import pred_rf, extract_features

# 設定網頁標題與圖示
st.set_page_config(page_title="機器手臂振動監測中心", layout="wide")
# --- 主畫面佈局 ---
st.title("🏭 機器手臂振動即時監測")
st.write("請上傳機器手臂振動數據檔案（.txt 或 .csv），系統將自動分析傳動軸負荷。")


@st.cache_resource
def load_rf_assets(pos):
    """載入 RF 模型與對應的 6 維 Scaler"""
    model_path = os.path.join("models", f"rf_clf_{pos}.pkl")
    scaler_path = os.path.join("models", f"rf_scaler_{pos}.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_resource
def load_cnn_assets(pos):
    # 分類型
    dis_m_path = f"models/cnn_dis_{pos}.pth"
    dis_model = CNN3D()
    dis_model.load_state_dict(torch.load(dis_m_path, map_location="cpu"))
    dis_model.eval()
    
    # 連續型
    con_m_path = f"models/cnn_con_{pos}.pth"
    con_model = CNN3D_con_coord()
    con_model.load_state_dict(torch.load(con_m_path, map_location="cpu"))
    con_model.eval()
    
    scaler = joblib.load(f"models/cnn_scaler_{pos}.pkl")
    
    return dis_model, con_model, scaler

# --- 側邊欄：模型控制中心 ---
with st.sidebar:
    st.header("🛠️ 模型控制")
    
    # 1. 選擇感測器位置 (對應 Xa, Xb, Ya, Yb)
    sensor_location = st.selectbox(
        "選擇感測器位置",
        options=["Xa", "Xb", "Ya", "Yb"],
        format_func=lambda x: {
            "Xa": "馬達側 - 水平 (Xa)",
            "Xb": "惰輪側 - 水平 (Xb)",
            "Ya": "馬達側 - 垂直 (Ya)",
            "Yb": "惰輪側 - 垂直 (Yb)"
        }[x]
    )
    
    # 2. 選擇預測模型 ( RF or CNN)
    analysis_type = st.radio(
        "選擇預測模型",
        options=["Random Forest", "CNN"],
        help="RF 使用 6 項核心統計特徵；CNN 使用 PSD/CSD 深度學習模型"
    )

    st.divider()
    st.info(f"當前載入模型路徑：\n`models/{sensor_location}...`")


uploaded_file = st.file_uploader("上傳振動數據檔案 (.txt 或 .csv)", type=["txt", "csv"])
if uploaded_file:
    try:
        temp_path = "temp_signal.txt"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        data = np.loadtxt(uploaded_file, skiprows=1)
        # 統一處理三軸數據
        signal = data.mean(axis=1) if (data.ndim == 2 and data.shape[1] == 3) else data.flatten()
        
        
        col1, col2 = st.columns(2)
        # --- 分支 1: RF ---
        if analysis_type == "Random Forest":
            
            # 載入 6 維模型
            clf, scaler = load_rf_assets(sensor_location)
            
            model = pred_rf(temp_path, scaler, clf)
            health_pred = model["Health_Prediction"]
            health_prob = model["Health_Probability"]
            features = extract_features(signal).flatten()

            with col1:
                st.subheader("RF 分析結果")
                #  0=異常, 1=正常
                if health_pred == 1 and health_prob >= 60:
                    st.success(" 手臂狀態：✅正常")
                elif health_pred == 1 and 50 <= health_prob < 60:
                    st.warning(" 手臂狀態：⚠️需注意")
                else:
                    st.error(" 手臂狀態：❌異常")
                
                with st.expander("查看 6 維核心特徵"):
                    st.write(pd.DataFrame([features], columns=["RMS", "Kurtosis", "RMSF", "Spectral_Kurtosis", "Spectral_Entropy", "Spectral_Energy"]).T)
                with col2:
                    st.subheader("警示燈號")
                    if health_pred == 1 and health_prob >= 60:
                        light = "🟢"
                    elif health_pred == 1 and 50 <= health_prob < 60:
                        light = "🟠"  
                    else:
                        light = "🔴"
                    st.markdown(f"<h1 style='text-align: center;'>{light}</h1>", unsafe_allow_html=True)
        # --- 分支 2: CNN ---
        elif analysis_type == "CNN":
            
            # 取得該方向對應的標籤 (X 或 Y)
            label_type = sensor_location[0] 
            dis_model, con_model, scaler = load_cnn_assets(sensor_location)
    

            with col1:
                st.subheader("CNN 分析結果")
                    
                # 類別型模型
                cnn_dis = pred_dis_cnn(temp_path, scaler, dis_model, label=label_type)
                # 連續型模型
                cnn_con = pred_con_cnn(temp_path, scaler, con_model, label=label_type)
                thresholds = {
                    "X": {"target": 80, "norm": 5, "warn": 15},  # 正常範圍 ±5, 注意範圍 ±15
                    "Y": {"target": 260, "norm": 10, "warn": 20} # 正常範圍 ±10, 注意範圍 ±20
                }

                if label_type in thresholds:
                    config = thresholds[label_type]
                    target = config["target"]
                    pred_class = cnn_dis[0]
                    pred_value = cnn_con[0]
    
                    diff = abs(pred_value - target)
    

                if pred_class == target and diff <= config["norm"]:
                    status = "success"
                    msg = "✅正常"
                elif pred_class == target and diff <= config["warn"]:
                    status = "warning"
                    msg = "⚠️需注意"
                else:
                    status = "error"
                    msg = "❌異常"
    
                display_text = f"預測負荷: {pred_class} Unit (手臂狀態:{msg})"
                getattr(st, status)(display_text)
                
            with col2:
                st.subheader("警示燈號")
    
                cnn_val = cnn_con[0]
    
                if label_type == "X":
                    if 75 <= cnn_val <= 85:
                        light = "🟢"
                    elif 65 < cnn_val < 75 or 85 < cnn_val < 95:
                        light = "🟠"
                    else: 
                        light = "🔴"
            
                elif label_type == "Y":
                    if 250 <= cnn_val <= 270:
                        light = "🟢"
                    elif 240 < cnn_val < 250 or 270 < cnn_val < 280:
                        light = "🟠"
                    else:
                        light = "🔴"
    
                st.markdown(f"<h1 style='text-align: center;'>{light}</h1>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ 診斷過程發生錯誤: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.divider()
 
        st.subheader("📈 原始訊號預覽")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(signal[:1000], color ='#1f77b4', linewidth=0.8)
        ax.set_title(f"Vibration Signal Preview ({sensor_location})")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
