import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset, look_back=1):
    """
    建立監督式學習資料集 (X, y)
    """
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

def scale_data(data):
    """資料正規化 (0~1)"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 確保輸入是 2D array
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def difference_data(df):
    """
    對資料進行差分處理 (用於 VAR/VECM 等需要平穩性資料的模型)
    """
    return df.diff().dropna()