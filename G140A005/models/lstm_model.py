import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from utils.dataset import create_dataset, scale_data

def run_lstm(series, look_back=60, epochs=10, batch_size=1):
    """
    執行 LSTM 模型 (已加入防呆機制)
    :param series: 價格序列
    """
    data_values = series.values
    
    # --- 防呆機制：檢查資料量是否足夠 ---
    n_samples = len(data_values)
    if n_samples < 20:
        raise ValueError(f"資料筆數過少 ({n_samples}筆)，無法進行 LSTM 訓練，請至少提供 20 筆以上資料。")
    
    # 如果資料少於 look_back，強制將 look_back 縮小為資料長度的 1/3，確保有足夠資料訓練
    if n_samples <= look_back + 10:
        old_look_back = look_back
        look_back = int(n_samples / 3)
        print(f"警告: 資料不足，自動將 look_back 從 {old_look_back} 調整為 {look_back}")

    # 資料正規化
    scaled_data, scaler = scale_data(data_values)
    
    # 建立訓練資料集
    X, y = create_dataset(scaled_data, look_back)
    
    #再一次檢查 X 是否為空
    if len(X) == 0:
         raise ValueError(f"訓練集為空，請減少 look_back 參數或增加資料量。")

    # LSTM 需要輸入格式為 [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # 建立 LSTM 模型
    model = Sequential()
    # 簡化模型以適應小資料量，避免過度擬合
    model.add(LSTM(50, return_sequences=False, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # 訓練模型
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # 預測未來一步
    last_sequence = scaled_data[-look_back:]
    last_sequence = last_sequence.reshape((1, look_back, 1))
    predicted_scaled = model.predict(last_sequence)
    
    # 反正規化回原始股價
    predicted_price = scaler.inverse_transform(predicted_scaled)
    
    return predicted_price[0][0]