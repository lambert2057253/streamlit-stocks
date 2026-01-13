import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 匯入自定義模組
from data.loader import load_data
from indicators.trend import calculate_sma, calculate_ema, calculate_bollinger_bands
from indicators.momentum import calculate_rsi, calculate_macd

# 匯入統計模型
from models.statistical_models import (
    run_ar, run_ma, run_arma, run_arima_model, 
    run_sarima, run_sarimax, run_var, run_vecm
)
from models.prophet_model import run_prophet

# 匯入深度學習模型
from models.dl_models import (
    run_mlp, run_rnn, run_lstm_model, run_ar_lstm, run_cnn
)

st.set_page_config(page_title="股票全方位分析系統", layout="wide")
st.title("📈 股票技術分析與預測平台 - G140A005")

# --- 側邊欄：1. 資料匯入 ---
st.sidebar.header("1. 資料匯入")
uploaded_file = st.sidebar.file_uploader("上傳 CSV (TWSE 格式)\n需台灣證卷交易所csv格式", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"讀取成功！資料筆數: {len(df)}")

        # --- 側邊欄：2. 技術指標設定 ---
        st.sidebar.header("2. 技術指標")
        show_sma = st.sidebar.checkbox("顯示 SMA (均線)")
        show_rsi = st.sidebar.checkbox("顯示 RSI (相對強弱指標)")
        
        # --- 主要圖表區域 ---
        # 如果勾選 RSI，則建立上下兩個子圖；否則只建一個
        if show_rsi:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, subplot_titles=('K線與均線', 'RSI 強弱指標'),
                                row_heights=[0.7, 0.3])
        else:
            fig = go.Figure()

        # 1. 繪製 K 線圖
        candlestick = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                     low=df['Low'], close=df['Close'], name='K線')
        
        if show_rsi:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)

        # 2. 處理 SMA 邏輯
        if show_sma:
            window = st.sidebar.slider("SMA 週期", 5, 60, 20)
            df['SMA'] = calculate_sma(df['Close'], window)
            sma_trace = go.Scatter(x=df.index, y=df['SMA'], name=f'SMA {window}', line=dict(color='orange'))
            if show_rsi:
                fig.add_trace(sma_trace, row=1, col=1)
            else:
                fig.add_trace(sma_trace)

        # 3. 處理 RSI 邏輯 (繪製在第二個子圖)
        if show_rsi:
            rsi_window = st.sidebar.slider("RSI 週期", 5, 30, 5)
            df['RSI'] = calculate_rsi(df['Close'], rsi_window)
            
            # RSI 曲線
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            
            # 加入 70/30 超買超賣基準線
            fig.add_shape(type="line", x0=df.index[0], y0=70, x1=df.index[-1], y1=70,
                          line=dict(color="red", width=1, dash="dot"), row=2, col=1)
            fig.add_shape(type="line", x0=df.index[0], y0=30, x1=df.index[-1], y1=30,
                          line=dict(color="green", width=1, dash="dot"), row=2, col=1)
            fig.update_yaxes(range=[0, 100], row=2, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False, title_text="市場數據分析")
        st.plotly_chart(fig, use_container_width=True, key="main_chart")

        # --- 側邊欄：3. 預測模型選擇 ---
        st.sidebar.header("3. 預測模型選擇")
        model_category = st.sidebar.selectbox("選擇模型類別", ["無", "統計模型 (Statistical)", "深度學習 (Deep Learning)", "Prophet"])
        
        if model_category != "無":
            selected_model = None
            if model_category == "統計模型 (Statistical)":
                selected_model = st.sidebar.selectbox("選擇具體模型", ["AR (自迴歸)", "MA (移動平均)", "ARMA", "ARIMA", "SARIMA", "SARIMAX", "VAR", "VECM"])
                steps = st.sidebar.slider("預測未來天數", 1, 14, 7)
            elif model_category == "深度學習 (Deep Learning)":
                selected_model = st.sidebar.selectbox("選擇具體模型", ["MLP", "RNN", "LSTM", "AR-LSTM", "CNN"])
                epochs = st.sidebar.slider("訓練 Epochs", 10, 100, 20)
                look_back = st.sidebar.slider("回測天數 (Look Back)", 5, 60, 20)
            elif model_category == "Prophet":
                periods = st.sidebar.slider("預測天數", 5, 60, 30)

            if st.button("開始執行預測"):
                try:
                    with st.spinner("模型運算中..."):
                        fig_res = go.Figure()
                        
                        # 歷史數據
                        fig_res.add_trace(go.Scatter(x=df.index, y=df['Close'], name='歷史數據', line=dict(color='blue')))

                        if model_category == "統計模型 (Statistical)":
                            if "AR (" in selected_model: result = run_ar(df['Close'], steps)
                            elif "MA (" in selected_model: result = run_ma(df['Close'], steps)
                            elif "ARMA" in selected_model: result = run_arma(df['Close'], steps)
                            elif "ARIMA" in selected_model: result = run_arima_model(df['Close'], steps)
                            elif "SARIMA" in selected_model: result = run_sarima(df['Close'], steps)
                            elif "SARIMAX" in selected_model: result = run_sarimax(df, steps)
                            elif "VAR" in selected_model: result = run_var(df, steps)
                            elif "VECM" in selected_model: result = run_vecm(df, steps)
                            
                            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
                            # 預測線連同最後一筆數據
                            plot_x = [df.index[-1]] + list(future_dates)
                            plot_y = [df['Close'].iloc[-1]] + list(result)
                            fig_res.add_trace(go.Scatter(x=plot_x, y=plot_y, name='未來預測', line=dict(color='red', dash='dash')))

                        elif model_category == "深度學習 (Deep Learning)":
                            if "MLP" in selected_model: pred = run_mlp(df['Close'], look_back, epochs)
                            elif "RNN" in selected_model: pred = run_rnn(df['Close'], look_back, epochs)
                            elif "LSTM" in selected_model: pred = run_lstm_model(df['Close'], look_back, epochs)
                            elif "AR-LSTM" in selected_model: pred = run_ar_lstm(df['Close'], look_back, epochs)
                            elif "CNN" in selected_model: pred = run_cnn(df['Close'], look_back, epochs)
                            
                            next_date = df.index[-1] + pd.Timedelta(days=1)
                            fig_res.add_trace(go.Scatter(x=[df.index[-1], next_date], y=[df['Close'].iloc[-1], pred], 
                                                         name='預測下一日', line=dict(color='red', width=4)))
                            st.metric("預測下個交易日價格", f"{pred:.2f}")

                        elif model_category == "Prophet":
                            forecast = run_prophet(df, periods)
                            fig_res.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prophet 預測', line=dict(color='green')))

                        st.plotly_chart(fig_res, use_container_width=True, key="res_chart")
                except Exception as e:
                    st.error(f"執行出錯: {e}")
    else:
        st.error("CSV 格式不符，請檢查數據結構。")
else:
    st.info("👋 歡迎！請先從側邊欄上傳 CSV 數據檔案開始分析。請至[台灣證卷交易所下載](https://www.twse.com.tw/zh/trading/historical/stock-day.html) CSV , 否則清洗數據會失敗")