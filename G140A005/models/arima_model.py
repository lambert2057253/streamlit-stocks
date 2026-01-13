from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def run_arima(series, order=(5, 1, 0), steps=5):
    """
    執行 ARIMA 模型並預測未來天數
    :param series: 價格序列 (Pandas Series)
    :param order: ARIMA 參數 (p, d, q)
    :param steps: 預測未來幾天
    """
    # 建立模型
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    
    # 預測
    forecast_result = model_fit.forecast(steps=steps)
    return forecast_result