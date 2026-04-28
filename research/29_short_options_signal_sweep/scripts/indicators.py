"""Technical indicators for the short-options signal sweep.

All functions are pure (no I/O). Most operate on a single 5-min DataFrame
already produced by `data_loader.load_5min()` or `data_loader.resample()`.

Conventions
-----------
- RSI uses Wilder's smoothing (EWM with alpha = 1/period).
- EMA uses span-style smoothing (`ewm(span=period, adjust=False)`).
- VWAP resets each session (anchored at session open).
- CPR is computed from the **previous trading day's daily** OHLC.
  Two "CPR direction" conventions are exposed:
    * `priceCPR`  : where today's price sits relative to today's CPR
                    (above TC = bullish, below BC = bearish, inside = neutral).
    * `cprDelta`  : where today's CPR pivot sits relative to yesterday's
                    CPR pivot (higher = bullish day, lower = bearish day).
- OR15 = high/low of the first 15 minutes of session (09:15-09:30 IST),
  i.e. the three 5-min candles 09:15, 09:20, 09:25. The 09:30 candle is
  the first POST-OR candle and is excluded from OR computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Momentum / trend
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# VWAP (intraday, session-anchored)
# ---------------------------------------------------------------------------

def vwap_intraday(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP. Returns NaN where volume is missing/zero.

    Index instruments (NIFTY) typically have volume == 0; in that case
    VWAP returns all-NaN and Strategy F (which depends on VWAP) is
    automatically inapplicable.
    """
    if df.empty:
        return pd.Series(dtype=float)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    day = df.index.normalize()
    cum_pv = pv.groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    return (cum_pv / cum_v.replace(0.0, np.nan)).rename("vwap")


# ---------------------------------------------------------------------------
# CPR
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CPRLevels:
    pivot: float
    top: float       # max(TC, BC)
    bottom: float    # min(TC, BC)
    width: float     # top - bottom

    @property
    def is_narrow(self) -> bool:
        # caller decides; placeholder so we can extend later
        return False


def cpr_from_prev(prev_high: float, prev_low: float, prev_close: float) -> CPRLevels:
    pivot = (prev_high + prev_low + prev_close) / 3.0
    bc = (prev_high + prev_low) / 2.0
    tc = 2.0 * pivot - bc
    top, bottom = (tc, bc) if tc >= bc else (bc, tc)
    return CPRLevels(pivot=pivot, top=top, bottom=bottom, width=top - bottom)


def daily_cpr_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Given a daily OHLC dataframe, return a per-trading-day CPR table.

    Each row is keyed by the **trading day** the CPR applies to (i.e. row N's
    CPR is computed from row N-1's H/L/C). The first row is dropped.

    Columns: pivot, top, bottom, width, prev_pivot.
    `prev_pivot` is the CPR pivot of the day before today's CPR — used by
    the `cprDelta` direction convention.
    """
    if daily_df.empty or len(daily_df) < 2:
        return pd.DataFrame()

    df = daily_df.sort_index().copy()
    rows: list[dict] = []
    prev_pivots: list[float] = []
    today_pivots: list[float] = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur_day = df.index[i]
        levels = cpr_from_prev(prev["high"], prev["low"], prev["close"])
        rows.append(
            {
                "date": cur_day,
                "pivot": levels.pivot,
                "top": levels.top,
                "bottom": levels.bottom,
                "width": levels.width,
            }
        )
        today_pivots.append(levels.pivot)
        prev_pivots.append(rows[-2]["pivot"] if rows[:-1] else np.nan)

    out = pd.DataFrame(rows).set_index("date")
    # prev_pivot = pivot of the trading day before this row's trading day
    out["prev_pivot"] = out["pivot"].shift(1)
    return out


def cpr_direction(
    *,
    convention: Literal["priceCPR", "cprDelta"],
    cpr: CPRLevels,
    price: float | None = None,
    prev_pivot: float | None = None,
) -> Literal["bullish", "bearish", "neutral"]:
    """Return CPR-implied direction under one of two conventions."""
    if convention == "priceCPR":
        if price is None:
            raise ValueError("priceCPR convention requires price=")
        if price > cpr.top:
            return "bullish"
        if price < cpr.bottom:
            return "bearish"
        return "neutral"

    if convention == "cprDelta":
        if prev_pivot is None or np.isnan(prev_pivot):
            return "neutral"
        if cpr.pivot > prev_pivot:
            return "bullish"
        if cpr.pivot < prev_pivot:
            return "bearish"
        return "neutral"

    raise ValueError(f"unknown CPR convention: {convention}")


# ---------------------------------------------------------------------------
# Opening Range (OR15)
# ---------------------------------------------------------------------------

OR15_END_TIME = time(9, 30)
OR15_START_TIME = time(9, 15)


def or15_levels(session_df: pd.DataFrame) -> tuple[float, float] | None:
    """Return (or_high, or_low) for a single session's 5-min dataframe.

    OR15 = high/low across candles whose timestamp is in [09:15, 09:30),
    i.e. the 09:15, 09:20, and 09:25 candles. Returns None if the session
    is missing OR candles (truncated half-day).
    """
    if session_df.empty:
        return None
    times = session_df.index.time
    mask = (times >= OR15_START_TIME) & (times < OR15_END_TIME)
    or_slice = session_df.loc[mask]
    if len(or_slice) < 3:  # need full 09:15, 09:20, 09:25
        return None
    return float(or_slice["high"].max()), float(or_slice["low"].min())


# ---------------------------------------------------------------------------
# Day-running extremes (high-so-far, low-so-far)
# ---------------------------------------------------------------------------

def running_extremes(session_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-candle running day-high and day-low of the session up to AND
    INCLUDING that candle. Used by Path C (post-12 day-extreme break) — when
    evaluating whether the *current* candle breaks the day's high/low, the
    relevant level is the prior candle's running extreme. Caller should
    `.shift(1)` the columns when comparing intra-candle.
    """
    if session_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=session_df.index)
    out["day_high_so_far"] = session_df["high"].cummax()
    out["day_low_so_far"] = session_df["low"].cummin()
    return out


# ---------------------------------------------------------------------------
# Gap
# ---------------------------------------------------------------------------

def session_gap(session_df: pd.DataFrame, prev_close: float) -> float:
    """Signed gap = (today's open - prev_close) / prev_close. Returns NaN if
    the session has no 09:15 candle.
    """
    if session_df.empty or np.isnan(prev_close) or prev_close == 0:
        return float("nan")
    first = session_df.iloc[0]
    return float((first["open"] - prev_close) / prev_close)


if __name__ == "__main__":
    # Quick sanity smoke
    from data_loader import load_5min, load_daily, INDEX_SYMBOL, slice_session, session_dates

    nifty = load_5min(INDEX_SYMBOL)
    nifty_d = load_daily(INDEX_SYMBOL)
    cpr_table = daily_cpr_table(nifty_d)
    print(f"CPR table rows: {len(cpr_table)}, head:")
    print(cpr_table.head(3))

    day = session_dates(nifty)[10]
    s = slice_session(nifty, day)
    or_levels = or15_levels(s)
    print(f"\nDay {day.date()}  rows={len(s)}  OR15={or_levels}")

    rsi_series = rsi(s["close"], 14)
    print(f"RSI(14) tail: {rsi_series.tail(3).to_dict()}")

    # gap
    if day in cpr_table.index:
        # find prev close = daily close on the trading day before `day`
        prev_close = nifty_d.loc[nifty_d.index < day, "close"].iloc[-1]
        print(f"Gap: {session_gap(s, prev_close):.4%}")

    extremes = running_extremes(s)
    print(f"Running extremes (last 3):\n{extremes.tail(3)}")
