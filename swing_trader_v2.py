import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ---------- Indicator helpers ----------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder-style RSI using an EMA approximation.
    """
    delta = series.diff()

    gain = (delta.where(delta > 0, 0.0))
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df: pd.DataFrame, length: int = 10) -> pd.Series:
    """
    Average True Range (ATR).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = true_range.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    return atr_val


def supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    SuperTrend(10, 3) implementation returning:
      - 'st': supertrend line
      - 'st_dir': +1 for uptrend (green), -1 for downtrend (red)
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    atr_val = atr(df, length)

    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st = pd.Series(index=df.index, dtype=float)
    st_dir = pd.Series(index=df.index, dtype=int)

    final_upper = pd.Series(index=df.index, dtype=float)
    final_lower = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            final_upper.iloc[i] = upper_band.iloc[i]
            final_lower.iloc[i] = lower_band.iloc[i]
            st.iloc[i] = np.nan
            st_dir.iloc[i] = 1  # arbitrary start
            continue

        # Final upper band
        if (upper_band.iloc[i] < final_upper.iloc[i - 1]) or (close.iloc[i - 1] > final_upper.iloc[i - 1]):
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Final lower band
        if (lower_band.iloc[i] > final_lower.iloc[i - 1]) or (close.iloc[i - 1] < final_lower.iloc[i - 1]):
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Direction + chosen band
        if st.iloc[i - 1] == final_upper.iloc[i - 1]:
            # Previously in downtrend
            if close.iloc[i] > final_upper.iloc[i]:
                st_dir.iloc[i] = 1
                st.iloc[i] = final_lower.iloc[i]
            else:
                st_dir.iloc[i] = -1
                st.iloc[i] = final_upper.iloc[i]
        else:
            # Previously in uptrend
            if close.iloc[i] < final_lower.iloc[i]:
                st_dir.iloc[i] = -1
                st.iloc[i] = final_upper.iloc[i]
            else:
                st_dir.iloc[i] = 1
                st.iloc[i] = final_lower.iloc[i]

    out = pd.DataFrame(index=df.index)
    out["st"] = st
    out["st_dir"] = st_dir
    return out


# ---------- Signal logic ----------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = rsi(df["Close"], length=14)
    st_df = supertrend(df, length=10, multiplier=3.0)
    df["st"] = st_df["st"]
    df["st_dir"] = st_df["st_dir"]
    return df


def find_long_signal(df: pd.DataFrame,
                     rsi_oversold_level: float = 35.0,
                     rsi_lookback: int = 30) -> Tuple[bool, dict]:
    """
    Long setup:
      - In the last `rsi_lookback` bars, RSI went below `rsi_oversold_level`
      - SuperTrend flipped from -1 to +1 on the latest bar
    Returns (signal_bool, info_dict).
    """
    if len(df) < rsi_lookback + 2:
        return False, {}

    recent = df.iloc[-rsi_lookback:]
    last = df.iloc[-1]
    prev = df.iloc[-2]

    recent_min_rsi = recent["rsi"].min()

    rsi_condition = recent_min_rsi <= rsi_oversold_level
    st_flip_condition = (prev["st_dir"] == -1) and (last["st_dir"] == 1)

    signal = bool(rsi_condition and st_flip_condition)

    info = {
        "last_close": float(last["Close"]),
        "st_line": float(last["st"]),
        "recent_min_rsi": float(recent_min_rsi),
        "last_rsi": float(last["rsi"]),
        "st_flip": bool(st_flip_condition),
    }

    # Suggested stop: max(supertrend line, 4% below close)
    stop_price = min(last["Close"] * 0.96, last["st"])
    info["stop_price"] = float(stop_price)
    info["risk_pct"] = float((last["Close"] - stop_price) / last["Close"] * 100.0)

    return signal, info


# ---------- Data download + scan ----------

def fetch_ohlc(ticker: str, interval: str = "1d", lookback_days: int = 365) -> pd.DataFrame:
    """
    Download OHLC data using yfinance.
    Uses `period` instead of explicit start/end dates to avoid timezone issues.
    """
    # yfinance understands things like "365d", "730d" etc.
    period = f"{lookback_days}d"

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({interval}).")

    return df


def scan_universe(tickers: List[str],
                  interval: str = "1d",
                  label: str = "") -> None:
    print(f"\n=== Scanning {label or interval} ===")
    for t in tickers:
        try:
            df = fetch_ohlc(t, interval=interval)
            df = compute_indicators(df)
            signal, info = find_long_signal(df)

            if signal:
                print(f"\n*** LONG SIGNAL: {t} ({interval}) ***")
                print(f"  Last close:      {info['last_close']:.2f}")
                print(f"  RSI (last/min):  {info['last_rsi']:.1f} / {info['recent_min_rsi']:.1f}")
                print(f"  ST line:         {info['st_line']:.2f}")
                print(f"  Stop suggestion: {info['stop_price']:.2f} (~{info['risk_pct']:.1f}% risk)")
            else:
                # Optional: print “watchlist” conditions
                if info:  # will be non-empty if df had enough rows
                    if info["recent_min_rsi"] <= 40:
                        print(f"- WATCH {t} ({interval}): "
                              f"recent RSI low {info['recent_min_rsi']:.1f}, ST flip: {info['st_flip']}")

        except Exception as e:
            print(f"! Error processing {t} ({interval}): {e}")


if __name__ == "__main__":
    # Universe to scan – tweak as you like
    daily_tickers = [
        "GLD", "GDX", "GDXJ", "SLV", "SILJ",
        "XLE", "XBI", "SMH", "FCX", "CCJ", "INDA", "EWZ"
    ]

    four_hour_tickers = [
        "GDX", "GDXJ", "URA", "SILJ", "SLV", "XBI", "SMH"
    ]

    # 1-day swing candidates
    scan_universe(daily_tickers, interval="1d", label="Daily (1D)")

    # 4-hour swing candidates
    scan_universe(four_hour_tickers, interval="4h", label="4H")
