import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Flatten, Conv1D, MaxPooling1D
from utils.dataset import create_dataset, scale_data

def prepare_dl_data(series, look_back):
    """
    智慧型資料準備：若資料不足，自動縮短 look_back
    """
    data_values = series.values
    n_samples = len(data_values)
    
    # --- 自動調整 look_back 機制 ---
    # 訓練至少需要 look_back + 1 (預測目標) + 1 (緩衝)
    if n_samples <= look_back + 5:
        old_look_back = look_back
        # 強制調整為資料長度的一半，或至少留 3 筆資料
        look_back = max(1, int(n_samples / 2) - 1)
        print(f"注意: 資料筆數 ({n_samples}) 少於設定的回測天數。自動調整 look_back: {old_look_back} -> {look_back}")

    scaled_data, scaler = scale_data(data_values)
    X, y = create_dataset(scaled_data, look_back)
    
    # 最後一道防線
    if len(X) == 0:
        raise ValueError(f"資料極度不足 ({n_samples}筆)，無法建立任何訓練樣本。")
        
    return X, y, scaled_data, scaler

# 1. 多層感知器 (MLP)
def run_mlp(series, look_back=20, epochs=10):
    X, y, scaled_data, scaler = prepare_dl_data(series, look_back)
    
    # 關鍵修正：使用實際生成的 X 形狀 (actual_look_back)
    actual_look_back = X.shape[1]
    
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=actual_look_back))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    
    last_seq = scaled_data[-actual_look_back:].reshape(1, actual_look_back)
    pred = model.predict(last_seq)
    return scaler.inverse_transform(pred)[0][0]

# 2. 迴圈神經網路 (RNN)
def run_rnn(series, look_back=20, epochs=10):
    X, y, scaled_data, scaler = prepare_dl_data(series, look_back)
    actual_look_back = X.shape[1]
    
    X = X.reshape((X.shape[0], actual_look_back, 1))
    
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=(actual_look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    
    last_seq = scaled_data[-actual_look_back:].reshape(1, actual_look_back, 1)
    pred = model.predict(last_seq)
    return scaler.inverse_transform(pred)[0][0]

# 3. 長短期記憶網路 (LSTM)
def run_lstm_model(series, look_back=60, epochs=10):
    X, y, scaled_data, scaler = prepare_dl_data(series, look_back)
    actual_look_back = X.shape[1]
    
    X = X.reshape((X.shape[0], actual_look_back, 1))
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(actual_look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    
    last_seq = scaled_data[-actual_look_back:].reshape(1, actual_look_back, 1)
    pred = model.predict(last_seq)
    return scaler.inverse_transform(pred)[0][0]

# 4. 自迴歸 LSTM (AR-LSTM)
def run_ar_lstm(series, look_back=60, epochs=10):
    X, y, scaled_data, scaler = prepare_dl_data(series, look_back)
    actual_look_back = X.shape[1]
    
    X = X.reshape((X.shape[0], actual_look_back, 1))
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(actual_look_back, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    
    last_seq = scaled_data[-actual_look_back:].reshape(1, actual_look_back, 1)
    pred = model.predict(last_seq)
    return scaler.inverse_transform(pred)[0][0]

# 5. 卷積神經網路 (CNN)
def run_cnn(series, look_back=60, epochs=10):
    X, y, scaled_data, scaler = prepare_dl_data(series, look_back)
    actual_look_back = X.shape[1]
    
    X = X.reshape((X.shape[0], actual_look_back, 1))
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(actual_look_back, 1)))
    model.add(MaxPooling1D(pool_size=2)) # 若資料太短，池化層可能會報錯，這裡保留但建議資料至少要 >4 筆
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    
    last_seq = scaled_data[-actual_look_back:].reshape(1, actual_look_back, 1)
    pred = model.predict(last_seq)
    return scaler.inverse_transform(pred)[0][0]