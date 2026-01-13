import pandas as pd
import io

def parse_roc_date(date_str):
    """
    將民國年字串 (ex: '114/12/01') 轉換為西元年 datetime 物件
    """
    try:
        if pd.isna(date_str):
            return None
        parts = date_str.split('/')
        if len(parts) != 3:
            return None
        year = int(parts[0]) + 1911
        month = int(parts[1])
        day = int(parts[2])
        return pd.Timestamp(year=year, month=month, day=day)
    except:
        return None

def clean_number(value):
    """
    移除逗號並轉換為浮點數
    """
    if isinstance(value, str):
        # 移除逗號
        value = value.replace(',', '')
        # 處理漲跌價差中的 + - 符號 (有時候會黏在數字上)
        value = value.replace('+', '').replace('X', '') # X有時代表除權息相關標記
    try:
        return float(value)
    except ValueError:
        return 0.0

def load_data(uploaded_file):
    """
    讀取 Streamlit 上傳的 CSV 檔案並進行清洗
    """
    if uploaded_file is None:
        return None

    # 台灣證交所下載的 CSV 通常是 Big5 或 CP950 編碼，但也可能是 UTF-8
    # 這裡先嘗試讀取，跳過第一行標題 ("114年12月..."), 從第二行開始當作 Header
    try:
        df = pd.read_csv(uploaded_file, encoding='cp950', header=1)
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', header=1)

    # 重新命名欄位以利程式處理 (對應您的資料結構)
    # 原始: "日期","成交股數","成交金額","開盤價","最高價","最低價","收盤價","漲跌價差","成交筆數","註記"
    # 注意：有時候結尾會有空白欄位，這裡做正規化
    df.columns = [c.strip() for c in df.columns]
    
    rename_map = {
        '日期': 'Date',
        '成交股數': 'Volume',
        '開盤價': 'Open',
        '最高價': 'High',
        '最低價': 'Low',
        '收盤價': 'Close'
    }
    
    df = df.rename(columns=rename_map)

    # 確保必要欄位存在
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return None

    # 1. 處理日期 (民國 -> 西元)
    df['Date'] = df['Date'].apply(parse_roc_date)
    
    # 2. 處理數值 (移除逗號)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = df[col].apply(clean_number)

    # 3. 設定索引並排序
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date').sort_index()

    return df