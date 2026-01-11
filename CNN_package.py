import os
import numpy as np
import pandas as pd
from scipy import signal
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import pickle

def save_scaler(scaler, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def read_data(folder_path):
    data_dict = {}
    subfolders_or_files = sorted(os.listdir(folder_path))

    for item in subfolders_or_files:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            # 處理子資料夾
            files = sorted([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
            for i, file_name in enumerate(files):
                file_path = os.path.join(item_path, file_name)
                data = np.loadtxt(file_path, skiprows=1)
                key = f"data_{item}_{i + 1}"
                data_dict[key] = data
        elif os.path.isfile(item_path):
            # 處理直接放在主資料夾中的檔案
            data = np.loadtxt(item_path, skiprows=1)
            key = f"data_root_{item}"
            data_dict[key] = data

    return data_dict


def read_txt(file_path):
    data_dict = {}
    if os.path.isfile(file_path):
        data = np.loadtxt(file_path, skiprows=1)
        key = os.path.splitext(os.path.basename(file_path))[0]  # e.g. "datafile1.txt" -> "datafile1"
        data_dict[key] = data
    else:
        raise ValueError("The provided path is not a valid file.")
    return data_dict

def tune_data(data_dict, min_len=16175, target_len=16175):
    cleaned_dict = {}

    for key, data in data_dict.items():
        if isinstance(data, np.ndarray):
            m = data.shape[0]
            if m < min_len:
                # print(f"移除變數：{key}（長度為 {m}）")
                continue  
            elif m > target_len:
                data = data[:target_len, :] 
            cleaned_dict[key] = data  
    return cleaned_dict

def PSD_CSD_predict(data_dict, scaler, nperseg = 256, mask = None, label = 'X'):
    #計算每個變數的PSD和CSD，並利用訓練資料的平均和標準差將其標準化。
    Pxx_list, Pyy_list, Pzz_list = [], [], []
    Pxy_list, Pxz_list, Pyz_list = [], [], []
    N = len(data_dict)
    if label == 'X':
        fs = 4000
    else:
        fs = 4200
    for var_name, data in data_dict.items():
        f, P = signal.welch(data[:, 0], fs=fs, nperseg=nperseg)
        if mask:
            freq_mask = (f >= mask[0]) & (f <= mask[1])
            P = P[freq_mask]
        Pxx_list.append(P)
        _, P = signal.welch(data[:, 1], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pyy_list.append(P)

        _, P = signal.welch(data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pzz_list.append(P)

        _, P = signal.csd(data[:, 0], data[:, 1], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pxy_list.append(np.abs(P))

        _, P = signal.csd(data[:, 0], data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pxz_list.append(np.abs(P))

        _, P = signal.csd(data[:, 1], data[:, 2], fs=fs, nperseg=nperseg)
        if mask:
            P = P[freq_mask]
        Pyz_list.append(np.abs(P))

    Pxx = np.stack(Pxx_list)
    Pyy = np.stack(Pyy_list)
    Pzz = np.stack(Pzz_list)
    Pxy = np.stack(Pxy_list)
    Pxz = np.stack(Pxz_list)
    Pyz = np.stack(Pyz_list)
    Pxx_norm = (Pxx - scaler['Pxx']['mean']) / scaler['Pxx']['std']
    Pyy_norm = (Pyy - scaler['Pyy']['mean']) / scaler['Pyy']['std']
    Pzz_norm = (Pzz - scaler['Pzz']['mean']) / scaler['Pzz']['std']
    Pxy_norm = (Pxy - scaler['Pxy']['mean']) / scaler['Pxy']['std']
    Pxz_norm = (Pxz - scaler['Pxz']['mean']) / scaler['Pxz']['std']
    Pyz_norm = (Pyz - scaler['Pyz']['mean']) / scaler['Pyz']['std']
    X = np.zeros((N, Pxx.shape[1], 3, 3))
    X[:, :, 0, 0] = Pxx_norm
    X[:, :, 1, 1] = Pyy_norm
    X[:, :, 2, 2] = Pzz_norm
    X[:, :, 1, 0] = X[:, :, 0, 1] = Pxy_norm
    X[:, :, 2, 0] = X[:, :, 0, 2] = Pxz_norm
    X[:, :, 2, 1] = X[:, :, 1, 2] = Pyz_norm
    return X

def add_coordconv_channels(data):
    #將頻率、行、列的坐標添加到資料中，並將其標準化到[-1, 1]的範圍內
    N, _, D, H, W = data.size()
    
    # Normalize x/y/freq to [-1, 1]
    z_range = torch.linspace(-1, 1, D)  # frequency
    y_range = torch.linspace(-1, 1, H)  # row
    x_range = torch.linspace(-1, 1, W)  # col

    # Create meshgrid in (D, H, W) order
    zz, yy, xx = torch.meshgrid(z_range, y_range, x_range, indexing='ij')  # (D, H, W)

    # Reshape to (1, 1, D, H, W) then expand to (N, 1, D, H, W)
    zz = zz.unsqueeze(0).unsqueeze(0).expand(N, -1, -1, -1, -1)
    yy = yy.unsqueeze(0).unsqueeze(0).expand(N, -1, -1, -1, -1)
    xx = xx.unsqueeze(0).unsqueeze(0).expand(N, -1, -1, -1, -1)

    # Concatenate with input → (N, 4, D, H, W)
    return torch.cat([data, zz, yy, xx], dim=1)

class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(2,0,0))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(32)
        self.pool = nn.MaxPool3d((2, 1, 1))

        self.fc1 = nn.LazyLinear(64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.drop_fc = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNN3D_con_coord(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(4, 16, kernel_size=(7, 3, 3), padding=(2, 0, 0))
        #self.bn1 = nn.GroupNorm(4, 16) 
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(5, 1, 1), padding=(1, 0, 0))
        #self.bn2 = nn.GroupNorm(4, 32)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((2, 1, 1))
        self.pool_2 = nn.AvgPool3d((2, 1, 1))
        #self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.LazyLinear(256)  
        self.bn_fc1 = nn.BatchNorm1d(256)  
        self.drop_fc = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)  
        self.fc4 = nn.Linear(16, 1)  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool_2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        #x = self.gap(x) 
        x = F.relu(self.bn_fc1(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def pred_dis_cnn(path, scaler, model_obj, label, folder = False):
    if folder:   
        data = read_data(path)
    else:
        data = read_txt(path)
    if label == 'X':
        data_dict = tune_data(data, 20000, 20000)
    elif label == 'Y':
        data_dict = tune_data(data, 21000, 21000)

    X = PSD_CSD_predict(data_dict, scaler, label = label)
    X = X.reshape(-1, 1, X.shape[1], X.shape[2], X.shape[3])
    X= torch.from_numpy(X).float()
    data_set = TensorDataset(X)
    
    if folder:
        test_loader = DataLoader(data_set, batch_size = 16, shuffle=False)
    else:
        test_loader = DataLoader(data_set, batch_size = 1, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for (x, ) in test_loader:
            x = x.to('cpu').float()
            output = model_obj(x)
            pred = output.argmax(dim=1)
            preds.append(pred.cpu().numpy())
    if label == 'X':
        reverse_map = {0: 65, 1: 80, 2: 95, 3: 130}
    elif label == 'Y':
        reverse_map = {0: 220, 1: 260, 2: 300, 3: 380}
    preds = np.concatenate(preds).tolist()
    mapped_preds = [reverse_map[p] for p in preds]
    return mapped_preds
def pred_con_cnn(path, scaler, model_obj, label, folder = False):
    if folder:   
        data = read_data(path)
    else:
        data = read_txt(path)
    if label == 'X':
        data_dict = tune_data(data, 20000, 20000)
    elif label == 'Y':
        data_dict = tune_data(data, 21000, 21000)
    X = PSD_CSD_predict(data_dict, scaler, label = label)
    X= torch.from_numpy(X).float()
    X = X.reshape(-1, 1, X.shape[1], X.shape[2], X.shape[3])
    X = add_coordconv_channels(X)
    data_set = TensorDataset(X)
    if folder:
        test_loader = DataLoader(data_set, batch_size = 16, shuffle=False)
    else:
        test_loader = DataLoader(data_set, batch_size = 1, shuffle=False)
    device = torch.device("cpu")
    preds = []
    with torch.no_grad():
        for (x, ) in test_loader:
            x = x.to('cpu').float()

            y_pred = model_obj(x)
            if label == 'Y':
                y_pred_real = y_pred * (380 - 220) + 220
            else:
                y_pred_real = y_pred * (130 - 65) + 65
            preds.append(y_pred_real.cpu().numpy())
    return [round(float(x), 2) for x in np.concatenate(preds).flatten()]