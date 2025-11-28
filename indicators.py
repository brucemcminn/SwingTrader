# indicators.py

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-style ATR."""
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    return atr_val


def supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute Supertrend (like TradingView 10,3).
    Returns df with added columns:
        'st'     - supertrend line
        'st_dir' - +1 for uptrend (green), -1 for downtrend (red)
    """
    df = df.copy()
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    atr_val = atr(high, low, close, length=length)

    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr_val
    basic_lb = hl2 - multiplier * atr_val

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    for i in range(1, len(df)):
        # Upper band
        if (basic_ub.iloc[i] > final_ub.iloc[i - 1]) or (close.iloc[i - 1] < final_ub.iloc[i - 1]):
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i - 1]

        # Lower band
        if (basic_lb.iloc[i] < final_lb.iloc[i - 1]) or (close.iloc[i - 1] > final_lb.iloc[i - 1]):
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i - 1]

    st = pd.Series(index=df.index, dtype=float)
    st_dir = pd.Series(index=df.index, dtype=int)

    # start arbitrary in uptrend
    st_dir.iloc[0] = 1
    st.iloc[0] = np.nan

    for i in range(1, len(df)):
        if close.iloc[i] > final_ub.iloc[i - 1]:
            st_dir.iloc[i] = 1
        elif close.iloc[i] < final_lb.iloc[i - 1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = st_dir.iloc[i - 1]

        if st_dir.iloc[i] == 1:
            st.iloc[i] = final_lb.iloc[i]
        else:
            st.iloc[i] = final_ub.iloc[i]

    df["st"] = st
    df["st_dir"] = st_dir

    return df