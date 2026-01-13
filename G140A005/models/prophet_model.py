from prophet import Prophet
import pandas as pd

def run_prophet(df, periods=30):
    """
    執行 Prophet 模型
    :param df: 必須包含 Index(Date) 和 Close
    :param periods: 預測未來幾天
    """
    # Prophet 需要特定的欄位名稱: ds (日期), y (數值)
    prophet_df = df.reset_index()[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']
    
    m = Prophet()
    m.fit(prophet_df)
    
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)