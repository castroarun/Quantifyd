"""Data loader for the short-options signal sweep.

Pulls 5-min and daily OHLC from `backtest_data/market_data.db` for:
- NIFTY index (`NIFTY50`)
- 10 large-caps with intraday history

Intentionally minimal: connects directly to SQLite, returns pandas DataFrames.
Caches per-process so a multi-symbol run only hits the DB once per symbol.

All times are kept naive (assume IST already; the DB stores naive timestamps).
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Project layout: research/29_short_options_signal_sweep/scripts/data_loader.py
# DB at:        backtest_data/market_data.db (3 levels up from scripts/)
DB_PATH = (
    Path(__file__).resolve().parents[3] / "backtest_data" / "market_data.db"
)

INTRADAY_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "HINDUNILVR",
]

INDEX_SYMBOL = "NIFTY50"

# Backtest period — clipped so daily T-1 is always available for CPR
BT_START = "2024-03-04"
BT_END = "2026-03-19"


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"market_data.db not found at {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


@lru_cache(maxsize=64)
def load_5min(symbol: str, start: str = BT_START, end: str = BT_END) -> pd.DataFrame:
    """Load 5-minute OHLCV for a symbol. Indexed by timestamp."""
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


@lru_cache(maxsize=64)
def load_daily(symbol: str, start: str = "2023-03-20", end: str = BT_END) -> pd.DataFrame:
    """Load daily OHLCV for a symbol. Indexed by date (normalized to midnight)."""
    sql = """
        SELECT date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = 'day'
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    with _connect() as con:
        df = pd.read_sql(sql, con, params=(symbol, start, end))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    return df


def resample(df_5min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 5-min OHLC to 10-min or 15-min by grouping consecutive candles.

    Groups are anchored to start-of-session (09:15) per day so candle
    boundaries stay consistent across days. Trailing partial groups
    (e.g., the lone 15:25 candle when grouping by 3) are dropped to
    avoid distorting indicators that assume a fixed-duration candle.
    """
    if timeframe == "5min":
        return df_5min
    n = {"10min": 2, "15min": 3}[timeframe]

    if df_5min.empty:
        return df_5min

    # Group within each session date. df_5min is indexed by timestamps that
    # already carry the session date (e.g., 2024-03-04 09:15:00).
    df = df_5min.copy()
    df["_day"] = df.index.normalize()

    rows: list[dict] = []
    for _, day_df in df.groupby("_day", sort=True):
        day_df = day_df.sort_index()
        n_complete = (len(day_df) // n) * n  # drop trailing partials
        for i in range(0, n_complete, n):
            grp = day_df.iloc[i : i + n]
            rows.append(
                {
                    "date": grp.index[0],
                    "open": grp["open"].iloc[0],
                    "high": grp["high"].max(),
                    "low": grp["low"].min(),
                    "close": grp["close"].iloc[-1],
                    "volume": grp["volume"].sum(),
                }
            )
    out = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    return out


def session_dates(df_5min: pd.DataFrame) -> list[pd.Timestamp]:
    """Sorted list of unique trading dates present in a 5-min dataframe."""
    if df_5min.empty:
        return []
    return sorted(df_5min.index.normalize().unique().tolist())


def slice_session(df_5min: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Return only the 5-min candles belonging to a given trading day."""
    day = pd.Timestamp(day).normalize()
    mask = df_5min.index.normalize() == day
    return df_5min.loc[mask].sort_index()


if __name__ == "__main__":
    # Quick sanity check
    nifty = load_5min(INDEX_SYMBOL, BT_START, BT_END)
    nifty_d = load_daily(INDEX_SYMBOL)
    print(f"NIFTY50  5min rows={len(nifty):>6}  range={nifty.index.min()} -> {nifty.index.max()}")
    print(f"NIFTY50  daily rows={len(nifty_d):>6}  range={nifty_d.index.min()} -> {nifty_d.index.max()}")
    print(f"Session count: {len(session_dates(nifty))}")
    for sym in INTRADAY_STOCKS[:3]:
        df = load_5min(sym, BT_START, BT_END)
        print(f"{sym:12s} 5min rows={len(df):>6}")
    # Resample sanity
    one_day = slice_session(nifty, session_dates(nifty)[10])
    print(f"\nOne day 5min: {len(one_day)} candles -> 15min resample: "
          f"{len(resample(one_day, '15min'))} candles, 10min: {len(resample(one_day, '10min'))}")
