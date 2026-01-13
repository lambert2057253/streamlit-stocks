import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

# 1. 自迴歸 (AR)
def run_ar(series, steps=5):
    # AR 是 ARIMA(p,0,0) 的特例
    model = ARIMA(series, order=(5, 0, 0))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

# 2. 移動平均 (MA)
def run_ma(series, steps=5):
    # MA 是 ARIMA(0,0,q) 的特例
    model = ARIMA(series, order=(0, 0, 5))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

# 3. 自迴歸移動平均 (ARMA)
def run_arma(series, steps=5):
    # ARMA 是 ARIMA(p,0,q) 的特例
    model = ARIMA(series, order=(2, 0, 2))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

# 4. 自迴歸綜合移動平均 (ARIMA)
def run_arima_model(series, steps=5):
    model = ARIMA(series, order=(5, 1, 2))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

# 5. 季節自迴歸綜合移動平均 (SARIMA)
def run_sarima(series, steps=5):
    # seasonal_order=(P,D,Q,s) s=12 代表月週期，若是日資料可設 5 (一週) 或 20 (一月)
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=steps)

# 6. 帶有外源迴歸量的 SARIMA (SARIMAX)
def run_sarimax(df, steps=5):
    # 需要 'Close' 作為目標，'Volume' 作為外生變數 (Exogenous)
    endog = df['Close']
    exog = df['Volume']
    
    model = SARIMAX(endog, exog=exog, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    
    # 預測時需要未來的 exog 值，這裡簡單假設未來 Volume 維持最後一天的數值
    last_vol = exog.iloc[-1]
    exog_forecast = [last_vol] * steps
    
    return model_fit.forecast(steps=steps, exog=exog_forecast)

# 7. 向量自迴歸 (VAR) - 多變量
def run_var(df, steps=5):
    # 使用 Close 和 Volume 兩個變數
    data = df[['Close', 'Volume']]
    # VAR 通常需要平穩資料，這裡做一次差分
    data_diff = data.diff().dropna()
    
    model = VAR(data_diff)
    model_fit = model.fit(maxlags=5) # 自動選擇 lag
    
    # 預測差分值
    lag_order = model_fit.k_ar
    forecast_diff = model_fit.forecast(data_diff.values[-lag_order:], steps=steps)
    
    # 還原差分 (簡單還原 Close)
    last_close = df['Close'].iloc[-1]
    forecast_close = []
    current_val = last_close
    for i in range(steps):
        # forecast_diff[:, 0] 是 Close 的變化量
        current_val += forecast_diff[i, 0] 
        forecast_close.append(current_val)
        
    return pd.Series(forecast_close, index=pd.RangeIndex(start=len(df), stop=len(df)+steps))

# 8. 向量誤差校正 (VECM) - 多變量
def run_vecm(df, steps=5):
    # VECM 用於有共整合關係的變數
    data = df[['Close', 'Open', 'High', 'Low']] 
    
    model = VECM(data, k_ar_diff=5, coint_rank=1, deterministic="ci")
    model_fit = model.fit()
    
    forecast = model_fit.predict(steps=steps)
    # forecast[:, 0] 對應 Close
    return pd.Series(forecast[:, 0])