# scan_daily_sp500.py

import pandas as pd
import yfinance as yf
import traceback

from indicators import rsi, supertrend


def get_sp500_symbols() -> list[str]:
    df = pd.read_csv("sp500.csv")
    return df["SYMBOL"].astype(str).str.strip().tolist()


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

    # NEW: flatten MultiIndex columns like ('Close', 'A') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Make sure we pass a 1-D Series to rsi()
    close_col = out["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]

    out["rsi"] = rsi(close_col, length=14)
    out = supertrend(out, length=10, multiplier=3.0)
    return out


def is_long_candidate(df: pd.DataFrame) -> bool:
    """Your first-pass daily long rules."""
    if len(df) < 40:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Force everything to plain Python scalars
    try:
        close = float(last["Close"])
        rsi_val = float(last["rsi"])
        prev_rsi = float(prev["rsi"])
        st_val = float(last["st"])
        st_dir = int(last["st_dir"])
        high = float(last["High"])
        low = float(last["Low"])
    except Exception:
        return False

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
        except Exception:
            print(f"\n=== Full traceback for {sym} ===")
            traceback.print_exc()
            break

    if not candidates:
        print("No candidates today with current rules.")
        return

    result = pd.DataFrame(candidates).sort_values("rsi")
    print("\n=== Daily Long Candidates (sorted by RSI) ===")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
