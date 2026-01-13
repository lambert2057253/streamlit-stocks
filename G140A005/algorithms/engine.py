import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# 在 StockPredictor 類別中修改 run_lstm 方法
def run_lstm(self, steps=7):
    # 1. 取得收盤價數據
    data = self.data['Close'].values.reshape(-1, 1)
    
    # 2. 數據量檢查：LSTM 需要至少 60 天歷史數據作為窗格
    if len(data) <= 60:
        raise ValueError(f"數據量不足！目前僅有 {len(data)} 筆，LSTM 需要至少 61 筆資料才能運行。")

    # 3. 資料正規化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 4. 準備訓練集 (Sliding Window)
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    # 轉換為 Numpy 並調整為 Keras 要求的 3D 形狀: (樣本數, 時間步, 特徵數)
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.array(y) # 確保 y 是 Numpy Array

    # 5. 建立模型 (修正 Keras 3 的警告，使用 Input 層)
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # 6. 訓練模型
    # 修正：確保 y 與 X 的第一維度相同
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    # 7. 滾動預測未來
    last_batch = scaled_data[-60:].reshape(1, 60, 1)
    preds = []
    
    for _ in range(steps):
        # 預測並取得數值
        curr_pred = model.predict(last_batch, verbose=0)
        preds.append(curr_pred[0, 0])
        
        # 更新窗格：推入預測值，彈出最舊值
        new_val = curr_pred.reshape(1, 1, 1)
        last_batch = np.append(last_batch[:, 1:, :], new_val, axis=1)
            
    # 8. 轉回原始價格
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()