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


def _rma(series: pd.Series, length: int) -> pd.Series:
    """
    Wilder's RMA (used by TradingView for ATR).
    """
    return series.ewm(alpha=1 / length, adjust=False).mean()


def supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend implementation closely matching TradingView's built-in version.

    Expects columns: 'High', 'Low', 'Close'
    Adds columns:
      - 'st'     : Supertrend line
      - 'st_dir' : +1 for uptrend (green), -1 for downtrend (red)
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    # True Range
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR using RMA (Wilder)
    atr = _rma(tr, length)

    # Basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    n = len(df)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    direction = pd.Series(index=df.index, dtype="int64")

    # Initialize direction
    if n == 0:
        out = df.copy()
        out["st"] = np.nan
        out["st_dir"] = 0
        return out

    direction.iloc[0] = 1  # start as uptrend by convention

    # Iterate over bars to refine bands and determine trend
    for i in range(1, n):
        prev = i - 1

        # Refine upper band
        if (upper_band.iloc[i] < final_upper.iloc[prev]) or (close.iloc[prev] > final_upper.iloc[prev]):
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[prev]

        # Refine lower band
        if (lower_band.iloc[i] > final_lower.iloc[prev]) or (close.iloc[prev] < final_lower.iloc[prev]):
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[prev]

        # Direction flip logic
        if direction.iloc[prev] == -1 and close.iloc[i] > final_upper.iloc[i]:
            direction.iloc[i] = 1
        elif direction.iloc[prev] == 1 and close.iloc[i] < final_lower.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[prev]

    # Supertrend line = lower band in uptrend, upper band in downtrend
    st = pd.Series(index=df.index, dtype="float64")
    st[direction == 1] = final_lower[direction == 1]
    st[direction == -1] = final_upper[direction == -1]

    out = df.copy()
    out["st"] = st
    out["st_dir"] = direction
    return out

