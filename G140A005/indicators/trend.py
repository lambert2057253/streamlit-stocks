import pandas as pd

def calculate_sma(series, window=5):
    """計算簡單移動平均 (SMA)"""
    return series.rolling(window=window).mean()

def calculate_ema(series, window=5):
    """計算指數移動平均 (EMA)"""
    return series.ewm(span=window, adjust=False).mean()

def calculate_bollinger_bands(series, window=20, num_std=2):
    """計算布林通道"""
    sma = calculate_sma(series, window)
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band