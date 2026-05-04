"""5-min → 30-min resampler + loader for Research 33.

Pulls 5-minute OHLCV from `backtest_data/market_data.db` and groups
6→1 anchored at 09:15 IST per session, dropping any trailing partial
group (e.g. the lone 15:15 → 15:30 partial). NSE session yields 12 full
30-min candles per trading day.

All times kept naive (DB stores naive timestamps in IST).
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd

DB_PATH = (
    Path(__file__).resolve().parents[3] / "backtest_data" / "market_data.db"
)

INTRADAY_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "HINDUNILVR",
]

BT_START = "2024-01-01"
BT_END = "2026-04-30"

CANDLES_PER_30MIN = 6  # 6 × 5min = 30min


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"market_data.db not found at {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


@lru_cache(maxsize=128)
def load_5min(symbol: str, start: str = BT_START, end: str = BT_END) -> pd.DataFrame:
    sql = """
        SELECT date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = '5minute'
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    with _connect() as con:
        df = pd.read_sql(sql, con, params=(symbol, start, end + " 23:59:59"))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def to_30min(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Group 6 consecutive 5-min candles into one 30-min candle, anchored
    per-session at 09:15 IST. Trailing partials dropped. Returns empty
    dataframe if input is empty.
    """
    if df_5min.empty:
        return df_5min

    df = df_5min.copy()
    df["_day"] = df.index.normalize()

    rows: list[dict] = []
    for _, day_df in df.groupby("_day", sort=True):
        day_df = day_df.sort_index()
        n_complete = (len(day_df) // CANDLES_PER_30MIN) * CANDLES_PER_30MIN
        for i in range(0, n_complete, CANDLES_PER_30MIN):
            grp = day_df.iloc[i : i + CANDLES_PER_30MIN]
            rows.append({
                "date": grp.index[0],
                "open": float(grp["open"].iloc[0]),
                "high": float(grp["high"].max()),
                "low": float(grp["low"].min()),
                "close": float(grp["close"].iloc[-1]),
                "volume": int(grp["volume"].sum()),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


@lru_cache(maxsize=128)
def load_30min(symbol: str, start: str = BT_START, end: str = BT_END) -> pd.DataFrame:
    """Convenience: 5-min → 30-min for a symbol."""
    return to_30min(load_5min(symbol, start, end))


if __name__ == "__main__":
    for sym in INTRADAY_STOCKS:
        d5 = load_5min(sym)
        d30 = to_30min(d5)
        if d30.empty:
            print(f"{sym:12s} EMPTY")
            continue
        print(
            f"{sym:12s} 5m={len(d5):>6}  30m={len(d30):>5}  "
            f"range={d30.index.min().date()} -> {d30.index.max().date()}"
        )
