"""Canonical data loader for research/81 — the ONLY way strategies read bars.

Encodes the Phase-0 audit's cleaning rules (reports/00_data_audit.md):
  - drop rows with nonpositive prices (2,550 such 5-min rows found)
  - drop OHLC-inconsistent rows (H < max(O,C), L > min(O,C), H < L)
  - intraday: keep only 09:15-15:29 bars (drops muhurat/junk); drop sessions
    with < MIN_BARS_SESSION bars (partial/muhurat sessions)
  - corporate-action guard: flag overnight |gap| > threshold dates on daily

Reads market_data.db read-only. In-process LRU-ish cache per (symbol, tf).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "backtest_data" / "market_data.db"

MIN_BARS_SESSION = 30          # sessions with fewer 5-min bars are muhurat/partial
SESSION_START, SESSION_END = "09:15", "15:29"

_cache: dict[tuple, pd.DataFrame] = {}
_CACHE_MAX = 64


def _connect():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)


def list_symbols(timeframe: str = "5minute", min_first: str | None = None,
                 max_first: str | None = None) -> list[str]:
    """Symbols on a timeframe, optionally filtered by their earliest bar date."""
    con = _connect()
    try:
        rows = con.execute(
            "SELECT symbol, MIN(date) FROM market_data_unified "
            "WHERE timeframe=? GROUP BY symbol", (timeframe,)).fetchall()
    finally:
        con.close()
    out = []
    for sym, mn in rows:
        d = mn[:10]
        if min_first and d < min_first:
            continue
        if max_first and d > max_first:
            continue
        out.append(sym)
    return sorted(out)


def load_bars(symbol: str, timeframe: str = "5minute",
              start: str | None = None, end: str | None = None,
              clean: bool = True) -> pd.DataFrame:
    """OHLCV DataFrame indexed by DatetimeIndex, audit-clean by default."""
    key = (symbol, timeframe, start, end, clean)
    if key in _cache:
        return _cache[key]

    con = _connect()
    try:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe=? ORDER BY date",
            con, params=(symbol, timeframe))
    finally:
        con.close()
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(str(end) + " 23:59:59")]

    if clean and len(df):
        pos = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
        sane = ((df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9)
                & (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9)
                & (df["high"] >= df["low"] - 1e-9))
        df = df[pos & sane]
        if timeframe != "day" and len(df):
            tm = df.index.strftime("%H:%M")
            df = df[(tm >= SESSION_START) & (tm <= SESSION_END)]
            if timeframe == "5minute" and len(df):
                per_sess = df.groupby(df.index.date).size()
                good = set(per_sess[per_sess >= MIN_BARS_SESSION].index)
                df = df[[d in good for d in df.index.date]]

    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = df
    return df


def resample_intraday(df5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Session-aware 5-min → N-min bars, aligned to 09:15 within each session."""
    if df5.empty:
        return df5
    sess = pd.Series(df5.index.normalize(), index=df5.index)
    mins_in = ((df5.index - df5.index.normalize()).total_seconds() // 60
               - (9 * 60 + 15)).astype(int)
    bucket = mins_in // minutes
    g = df5.groupby([sess, bucket])
    out = g.agg(open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"))
    # index = timestamp of bucket start
    idx = [s + pd.Timedelta(minutes=9 * 60 + 15 + b * minutes)
           for s, b in out.index]
    out.index = pd.DatetimeIndex(idx, name="date")
    return out.sort_index()


def to_daily(df5: pd.DataFrame) -> pd.DataFrame:
    """Session daily bars from 5-min (index = session date)."""
    if df5.empty:
        return df5
    g = df5.groupby(df5.index.normalize())
    out = g.agg(open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"))
    out.index.name = "date"
    return out


def ca_gap_flags(daily: pd.DataFrame, threshold: float = 0.25) -> pd.DatetimeIndex:
    """Dates whose overnight |close/prev_close - 1| > threshold — corporate-action
    or crash suspects. Strategies must not enter across these; symbols with many
    flags should be excluded from cash-equity results."""
    if daily.empty or len(daily) < 2:
        return pd.DatetimeIndex([])
    gap = (daily["close"] / daily["close"].shift(1) - 1).abs()
    return daily.index[gap > threshold]


def chronological_splits(index_like, is_frac: float = 0.6, val_frac: float = 0.2):
    """(is, val, oos) boundary timestamps for a bar index — brief section 4.1.
    OOS is touched ONCE per finalized system; log every touch in the study state."""
    n = len(index_like)
    return (index_like[int(n * is_frac) - 1],
            index_like[int(n * (is_frac + val_frac)) - 1])
