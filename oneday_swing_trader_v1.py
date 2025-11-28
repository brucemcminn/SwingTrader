import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

def get_history(symbol, period="6mo"):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True)
    if df.empty:
        return None
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    # RSI 14
    df["rsi"] = ta.rsi(df["Close"], length=14)

    # Supertrend 10,3 like you’re using on TradingView
    st = ta.supertrend(df["High"], df["Low"], df["Close"],
                       length=10, multiplier=3.0)
    # pandas_ta typically gives columns like:
    # 'SUPERT_10_3.0', 'SUPERTd_10_3.0', 'SUPERTl_10_3.0'
    df["st_value"] = st["SUPERT_10_3.0"]
    df["st_dir"]   = st["SUPERTd_10_3.0"]  # +1 = uptrend (green), -1 = downtrend (red)

    return df
