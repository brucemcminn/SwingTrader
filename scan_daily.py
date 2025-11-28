# scan_daily.py

import pandas as pd
import yfinance as yf

from indicators import rsi, supertrend
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols


def get_sp500_symbols():
    """Get S&P 500 constituents from NASDAQ trader list."""
    df = get_nasdaq_symbols()
    # Filter for S&P 500 (flagged by Y in ETF field for SP500 index)
    sp500 = df[df["ETF"] == "N"]["Symbol"].tolist()
    return sp500


def download_history(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return None
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = rsi(df["Close"], length=14)
    df = supertrend(df, length=10, multiplier=3.0)
    return df


def is_long_candidate(df: pd.DataFrame) -> bool:
    """Your first-pass daily long rules."""
    if len(df) < 40:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = last["Close"]
    rsi_val = last["rsi"]
    prev_rsi = prev["rsi"]
    st_val = last["st"]
    st_dir = last["st_dir"]

    # 1) Supertrend green (uptrend)
    if st_dir != 1:
        return False

    # 2) Price above ST line
    if close <= st_val:
        return False

    # 3) RSI in pullback zone and curling up
    if not (40 <= rsi_val <= 60):
        return False
    if rsi_val <= prev_rsi:
        return False

    # 4) Candle not an ugly bearish bar (close not in bottom 40% of range)
    high = last["High"]
    low = last["Low"]
    if high > low:
        pos = (close - low) / (high - low)
        if pos < 0.4:
            return False

    # 5) Optional: liquidity filter – avg volume
    avg_vol = df["Volume"].tail(20).mean()
    if avg_vol < 1_000_000:  # 1M shares/day
        return False

    return True


def main():
    symbols = get_sp500_symbols()
    print(f"Scanning {len(symbols)} S&P 500 symbols...")

    candidates = []

    for sym in symbols:
        try:
            df = download_history(sym)
            if df is None:
                continue
            df = add_indicators(df)

            if is_long_candidate(df):
                last = df.iloc[-1]
                candidates.append(
                    {
                        "symbol": sym,
                        "close": round(float(last["Close"]), 2),
                        "rsi": round(float(last["rsi"]), 2),
                        "st": round(float(last["st"]), 2),
                    }
                )
        except Exception as e:
            # Be tolerant of random yfinance/API weirdness
            print(f"Error for {sym}: {e}")

    if not candidates:
        print("No candidates today with current rules.")
        return

    result = pd.DataFrame(candidates).sort_values("rsi")
    print("\n=== Daily Long Candidates (sorted by RSI) ===")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()