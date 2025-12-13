#!/usr/bin/env python3
"""
Build ATR-based trading universe from NASDAQ-listed symbols only,
using a local nasdaqlisted.csv file (pipe-delimited).

Inputs:
- nasdaqlisted.csv

Outputs:
- atr_universe_full.csv : all symbols with ATR and ATR%
- atr_top500.csv        : final universe, ranked by ATR%, with pinned ETFs
"""

import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------- CONFIG ----------
BASE_DIR = Path(".")
NASDAQ_FILE = BASE_DIR / "NASDAQ_list.csv"

FULL_OUT_FILE = BASE_DIR / "atr_universe_full.csv"
TOP_OUT_FILE = BASE_DIR / "atr_top500.csv"

LOOKBACK_PERIOD = "90d"
ATR_LENGTH = 14

MIN_PRICE = 5.0
MIN_BARS = ATR_LENGTH + 5

# Always-include ETFs (broad + precious metals)
ALWAYS_INCLUDE = [
    "SPY",  # S&P 500
    "QQQ",  # NASDAQ-100
    "IWM",  # Russell 2000
    "GLD",  # Gold
    "IAU",  # Gold
    "SGOL", # Gold
    "PHYS", # Gold (closed-end but trades like ETF)
    "SLV",  # Silver
]


# ---------- HELPERS ----------

def build_raw_universe_from_nasdaq() -> pd.DataFrame:
    """
    Read nasdaqlisted.csv (pipe-delimited) and return a cleaned universe
    with a 'Symbol' column.
    """
    if not NASDAQ_FILE.exists():
        raise FileNotFoundError(f"Missing {NASDAQ_FILE}. Put nasdaqlisted.csv in {BASE_DIR}")

    print(f"Reading NASDAQ symbols from {NASDAQ_FILE}")
    df = pd.read_csv(NASDAQ_FILE, sep="|")

    # Normalize column names and symbol
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" not in df.columns:
        raise ValueError("nasdaqlisted.csv must have a 'Symbol' column")

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()

    # Drop test issues if column present
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] != "Y"]

    # Optional: filter out obvious non-common stuff (rights, units, warrants, prefs etc.)
    if "Security Name" in df.columns:
        junk_keywords = ["Warrant", "Warrants", "Unit", "Units", "Right", "Rights",
                         "Preferred", "Preference", "Depositary Share", "Notes"]
        pat = "|".join(junk_keywords)
        df = df[~df["Security Name"].str.contains(pat, case=False, na=False)]

    df = df.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    print(f"Raw NASDAQ universe size after filters: {len(df)} symbols")
    return df[["Symbol"]]


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = ATR_LENGTH) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def fetch_ohlcv(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            period=LOOKBACK_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            return None

        return df
    except Exception:
        return None


def build_atr_universe():
    raw_universe = build_raw_universe_from_nasdaq()
    symbols = raw_universe["Symbol"].tolist()

    rows = []
    total = len(symbols)
    print(f"Computing ATR for up to {total} NASDAQ symbols...")

    for i, sym in enumerate(symbols, start=1):
        if i % 200 == 0:
            print(f"  Processed {i}/{total} symbols...")

        df = fetch_ohlcv(sym)
        if df is None or len(df) < MIN_BARS:
            continue

        last = df.iloc[-1]
        try:
            close = float(last["Close"])
        except Exception:
            continue

        if close < MIN_PRICE:
            continue

        df["ATR"] = atr(df)
        last = df.iloc[-1]
        try:
            atr_val = float(last["ATR"])
        except Exception:
            continue

        if atr_val <= 0 or close <= 0:
            continue

        atr_pct = atr_val / close

        rows.append(
            {
                "symbol": sym,
                "close": close,
                "atr": atr_val,
                "atr_pct": atr_pct,
            }
        )

        # Small pause to be polite to Yahoo
        time.sleep(0.01)

    if not rows:
        print("No symbols passed ATR universe filters.")
        return

    full_df = pd.DataFrame(rows)
    full_df = full_df.sort_values("atr_pct", ascending=False).reset_index(drop=True)

    full_df.to_csv(FULL_OUT_FILE, index=False)
    print(f"Saved full ATR universe ({len(full_df)} symbols) to {FULL_OUT_FILE}")

    # Build ATR top 500 + pinned ETFs
    top_df = full_df.head(500).copy()
    top_symbols = set(top_df["symbol"].tolist())

    for pinned in ALWAYS_INCLUDE:
        if pinned not in top_symbols:
            row = full_df[full_df["symbol"] == pinned]
            if not row.empty:
                top_df = pd.concat([top_df, row], ignore_index=True)
                top_symbols.add(pinned)
            else:
                # If not present in full_df (e.g. NYSE-only ETF),
                # still add a bare row so the scanner sees it
                top_df = pd.concat(
                    [
                        top_df,
                        pd.DataFrame(
                            [{"symbol": pinned, "close": None, "atr": None, "atr_pct": None}]
                        ),
                    ],
                    ignore_index=True,
                )
                top_symbols.add(pinned)

    top_df = (
        top_df.sort_values("atr_pct", ascending=False)
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )

    final_universe = top_df[["symbol"]].rename(columns={"symbol": "SYMBOL"})
    final_universe.to_csv(TOP_OUT_FILE, index=False)
    print(f"Saved ATR-top NASDAQ universe ({len(final_universe)} symbols) to {TOP_OUT_FILE}")


def main():
    try:
        build_atr_universe()
    except Exception:
        print("Error while building ATR universe:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
