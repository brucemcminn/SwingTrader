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
    rsi_val = 100 - (100 / (1 + rs))

    return rsi_val


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
    Simple Supertrend implementation.

    Returns a copy of df with two extra columns:
      - 'st'     : supertrend line (float)
      - 'st_dir' : +1 for uptrend, -1 for downtrend (int)
    """
    df = df.copy()

    # Convert columns to flat 1-D NumPy arrays no matter how yfinance/pandas packs them
    high_arr = np.asarray(df["High"], dtype="float64").reshape(-1)
    low_arr = np.asarray(df["Low"], dtype="float64").reshape(-1)
    close_arr = np.asarray(df["Close"], dtype="float64").reshape(-1)

    n = len(df)
    if n == 0:
        df["st"] = np.nan
        df["st_dir"] = 0
        return df

    # Midpoint of high/low
    hl2 = (high_arr + low_arr) / 2.0

    # Use Pandas rolling to get a simple ATR-like band, but start from 1-D arrays
    high_s = pd.Series(high_arr)
    low_s = pd.Series(low_arr)

    band_range = (
        high_s.rolling(length).max()
        - low_s.rolling(length).min()
    ).rolling(length).mean().to_numpy()

    upperband = hl2 + multiplier * band_range
    lowerband = hl2 - multiplier * band_range

    st = np.full(n, np.nan, dtype="float64")
    direction = np.zeros(n, dtype="int64")

    # Seed first value
    st[0] = hl2[0]
    direction[0] = 1

    for i in range(1, n):
        prev_st = st[i - 1]
        curr_close = close_arr[i]

        ub = upperband[i]
        lb = lowerband[i]

        # If bands are NaN early on, fall back to previous ST
        if np.isnan(ub):
            ub = prev_st
        if np.isnan(lb):
            lb = prev_st

        # If close or prev_st is NaN, just carry forward
        if np.isnan(curr_close) or np.isnan(prev_st):
            st[i] = prev_st
            direction[i] = direction[i - 1]
            continue

        # Pure scalar comparison
        if curr_close > prev_st:
            # Uptrend
            st[i] = max(lb, prev_st)
            direction[i] = 1
        else:
            # Downtrend
            st[i] = min(ub, prev_st)
            direction[i] = -1

    df["st"] = st
    df["st_dir"] = direction
    return df
