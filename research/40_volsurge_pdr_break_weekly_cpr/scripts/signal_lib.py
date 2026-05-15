"""Pure signal functions for the VOLSURGE + PDR-break + narrow-weekly-CPR system.

NO I/O here — every function takes pandas frames / scalars and returns
frames / scalars. The runner (`run_volsurge_sweep.py`) wires these together.

The signal logic faithfully encodes the 10 examples in
`VOLSURGE_PDR_BREAK_WEEKLY_CPR_INTRADAY_SWEEP_STATUS.md`:

  ex#1/2/3  ZYDUSLIFE/BIOCON/LAURUSLABS  long, vol-surge + PDR break, uptrend
  ex#4      KALYANKJIL                   SHORT mirror -> symmetric trend-aligned
  ex#5      VOLTAS 13-Apr                wide CPR + no vol -> NO-TRADE (AND gate)
  ex#6      VOLTAS 27-Apr                weak vol -> correctly rejected (failure)
  ex#7      TMPV   week-of 08-Apr        trigger = opening TF candle of ANY day
                                          inside the narrow-CPR week (not day 1)
  ex#8      TMPV   30-Mar                clean directional candle MANDATORY
                                          (broke range but bullish bar -> skip)
  ex#9      TMPV   19-Jan                FULL confluence still FAILED
                                          -> judge on expectancy, never 1 trade
  ex#10     TMPV   15-May                big candle INSIDE prior WEEK range
                                          -> range break must escape the WEEK
                                          range, not merely the prior day.

Locked rules turned into code below:

  * weekly_cpr(daily_df)  -> per-(year,week) width% from PRIOR week H/L/C.
  * daily_trend(daily_df, mode) -> 'up'/'down'/'flat' per date, as of D-1.
  * resample_5m(df, tf)   -> session-anchored OHLCV resample (drops trailing
                              partial group; mirrors research/29 resample()).
  * is_clean_candle(...)  -> directional body / colour / close-zone gate (ex#8).
  * range_escape(...)     -> close must clear BOTH prior-day AND prior-WEEK
                              extreme in the trade direction (ex#10).
  * volume_surge(...)     -> trigger vol >= k * same-slot trailing baseline
                              (ex#5/6 mandatory).
  * clear_room(...)       -> optional residual S/R-headroom filter (ex#6/9
                              hypothesis), on/off grid axis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

SESSION_OPEN = pd.Timestamp("09:15:00").time()


# ---------------------------------------------------------------------------
# Weekly CPR  (prev-week H/L/C -> this week pivot/bc/tc, width%)
# ---------------------------------------------------------------------------

def weekly_cpr(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Per ISO (year, week) weekly CPR computed from the PRIOR week's H/L/C.

    Formula (services/cpr_covered_call_service.py ~360):
        P  = (H + L + C) / 3
        BC = (H + L) / 2
        TC = 2P - BC
        width      = |TC - BC|
        width_pct  = width / P * 100

    Returns a DataFrame indexed by (iso_year, iso_week) of the CURRENT week,
    with columns: pivot, bc, tc, top, bottom, width, width_pct,
    prev_week_high, prev_week_low, prev_week_close.

    A week is only present if its PRIOR week had >= 3 daily bars (mirrors the
    >=3-day guard in cpr_covered_call_service._calculate_weekly_cpr).
    """
    if daily_df is None or daily_df.empty or len(daily_df) < 6:
        return pd.DataFrame()

    df = daily_df.sort_index().copy()
    iso = df.index.isocalendar()
    df["_iy"] = iso["year"].values
    df["_iw"] = iso["week"].values

    # Aggregate each week's H/L/C (close = last bar of that week).
    wk = df.groupby(["_iy", "_iw"]).agg(
        wk_high=("high", "max"),
        wk_low=("low", "min"),
        wk_close=("close", "last"),
        n=("close", "size"),
    )
    wk = wk.sort_index()

    rows = []
    keys = list(wk.index)
    for i in range(1, len(keys)):
        prev = wk.loc[keys[i - 1]]
        cur_key = keys[i]
        if prev["n"] < 3:
            continue
        ph, pl, pc = float(prev["wk_high"]), float(prev["wk_low"]), float(prev["wk_close"])
        pivot = (ph + pl + pc) / 3.0
        bc = (ph + pl) / 2.0
        tc = 2.0 * pivot - bc
        top, bottom = (tc, bc) if tc >= bc else (bc, tc)
        width = top - bottom
        width_pct = (width / pivot * 100.0) if pivot > 0 else np.nan
        rows.append({
            "iso_year": int(cur_key[0]),
            "iso_week": int(cur_key[1]),
            "pivot": pivot, "bc": bc, "tc": tc,
            "top": top, "bottom": bottom,
            "width": width, "width_pct": width_pct,
            "prev_week_high": ph, "prev_week_low": pl, "prev_week_close": pc,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index(["iso_year", "iso_week"])
    return out


def cpr_for_date(cpr_tbl: pd.DataFrame, day: pd.Timestamp) -> Optional[dict]:
    """Look up the weekly-CPR row that governs trading day `day`."""
    if cpr_tbl is None or cpr_tbl.empty:
        return None
    iso = pd.Timestamp(day).isocalendar()
    key = (int(iso.year), int(iso.week))
    if key not in cpr_tbl.index:
        return None
    return cpr_tbl.loc[key].to_dict()


# ---------------------------------------------------------------------------
# Daily trend (side selector, evaluated as of D-1)
# ---------------------------------------------------------------------------

def daily_trend(daily_df: pd.DataFrame, mode: str) -> pd.Series:
    """Per trading-date trend label in {'up','down','flat'}.

    The label for date D is computed from data AVAILABLE AT THE CLOSE OF D
    (the runner then uses the D-1 label as the side selector for D, so there
    is no look-ahead).

    modes:
      'sma50'  : close > SMA50  -> up ; close < SMA50  -> down ; else flat
      'sma200' : close > SMA200 -> up ; close < SMA200 -> down ; else flat
      'hh20'   : close == 20-day rolling max -> up ;
                 close == 20-day rolling min -> down ; else flat
                 (20-day higher-high / lower-low breakout proxy)
    """
    if daily_df is None or daily_df.empty:
        return pd.Series(dtype=object)
    df = daily_df.sort_index()
    close = df["close"].astype(float)

    if mode == "sma50":
        ma = close.rolling(50, min_periods=50).mean()
        up, down = close > ma, close < ma
    elif mode == "sma200":
        ma = close.rolling(200, min_periods=200).mean()
        up, down = close > ma, close < ma
    elif mode == "hh20":
        hh = df["high"].rolling(20, min_periods=20).max()
        ll = df["low"].rolling(20, min_periods=20).min()
        up = close >= hh
        down = close <= ll
    else:
        raise ValueError(f"unknown daily-trend mode: {mode!r}")

    out = pd.Series("flat", index=df.index, dtype=object)
    out[up.fillna(False)] = "up"
    out[down.fillna(False) & ~up.fillna(False)] = "down"
    # rows where the MA is NaN (insufficient history) -> 'flat' (no trade)
    return out


def trend_to_direction(label: str) -> Optional[str]:
    """Map a daily-trend label to a tradable side. flat -> no trade."""
    if label == "up":
        return "long"
    if label == "down":
        return "short"
    return None


# ---------------------------------------------------------------------------
# Resample 5-min -> {5,10,15,30,60} min, session-anchored
# ---------------------------------------------------------------------------

_TF_GROUP = {"5min": 1, "10min": 2, "15min": 3, "30min": 6, "60min": 12}


def resample_5m(df_5min: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample a 5-min OHLCV frame to `tf`, anchored to the session open.

    Mirrors research/29 data_loader.resample(): groups N consecutive 5-min
    candles WITHIN each trading day (so 09:15 is always a group boundary,
    consistent across days), summing volume and dropping the trailing
    partial group (e.g. the lone 15:25 when N does not divide the session).
    """
    if df_5min is None or df_5min.empty:
        return df_5min if df_5min is not None else pd.DataFrame()
    if tf not in _TF_GROUP:
        raise ValueError(f"unknown timeframe: {tf!r}")
    n = _TF_GROUP[tf]
    if n == 1:
        return df_5min.sort_index()

    df = df_5min.sort_index().copy()
    day = df.index.normalize()
    rows: list[dict] = []
    for _, gday in df.groupby(day, sort=True):
        gday = gday.sort_index()
        n_complete = (len(gday) // n) * n
        for i in range(0, n_complete, n):
            grp = gday.iloc[i:i + n]
            rows.append({
                "date": grp.index[0],
                "open": float(grp["open"].iloc[0]),
                "high": float(grp["high"].max()),
                "low": float(grp["low"].min()),
                "close": float(grp["close"].iloc[-1]),
                "volume": float(grp["volume"].sum()),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def opening_candles(df_tf: pd.DataFrame) -> pd.DataFrame:
    """The first candle of each trading day (ex#7: opening TF candle of ANY
    day inside a narrow-CPR week is the trigger candle)."""
    if df_tf is None or df_tf.empty:
        return pd.DataFrame()
    df = df_tf.sort_index()
    first_idx = df.groupby(df.index.normalize()).head(1).index
    return df.loc[first_idx]


# ---------------------------------------------------------------------------
# Clean directional candle (ex#8 — mandatory gate)
# ---------------------------------------------------------------------------

# strictness presets -> (min body/range, close-zone fraction of range)
CLEAN_PRESETS = {
    "loose": (0.40, 0.40),   # body >= 40% of range, close in outer 40%
    "strict": (0.60, 0.25),  # body >= 60% of range, close in outer 25%
}


def is_clean_candle(o: float, h: float, l: float, c: float,
                    direction: str, b_min: float, zone: float) -> bool:
    """True iff the trigger candle is a decisive bar IN the trade direction.

    LONG  : green (c > o), real body (|c-o|/range >= b_min), close in the
            TOP `zone` fraction of the candle's range.
    SHORT : red   (c < o), real body, close in the BOTTOM `zone` fraction.

    Rejects dojis, pins, and opposite-colour bars even if the close breaches
    the broken range (ex#8 TMPV 30-Mar: closed below range but bullish bar
    -> NO-TRADE).
    """
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    if body / rng < b_min:
        return False
    if direction == "long":
        if c <= o:                       # must be green
            return False
        # close within the top `zone` of the range
        return (c - l) / rng >= (1.0 - zone)
    elif direction == "short":
        if c >= o:                       # must be red
            return False
        return (h - c) / rng >= (1.0 - zone)
    return False


# ---------------------------------------------------------------------------
# Range escape (ex#10 — must clear BOTH prior-day AND prior-WEEK extreme)
# ---------------------------------------------------------------------------

def range_escape(trigger_close: float, prev_day_hi: float, prev_day_lo: float,
                  prev_week_hi: float, prev_week_lo: float,
                  direction: str) -> bool:
    """True iff the trigger candle's CLOSE genuinely escapes the established
    range — clearing the prior DAY *and* the prior WEEK extreme.

    ex#10 (TMPV 15-May): a big-volume first candle that stays inside the prior
    WEEK's range is NOT a breakout. So a long break requires
    close > max(PrevDayHigh, PrevWeekHigh); a short break requires
    close < min(PrevDayLow, PrevWeekLow).
    """
    if direction == "long":
        ref = max(prev_day_hi, prev_week_hi)
        return trigger_close > ref
    elif direction == "short":
        ref = min(prev_day_lo, prev_week_lo)
        return trigger_close < ref
    return False


# ---------------------------------------------------------------------------
# Volume surge (ex#5 / ex#6 — mandatory gate)
# ---------------------------------------------------------------------------

def volume_surge(trigger_vol: float, baseline: float, k: float) -> bool:
    """True iff trigger-candle volume >= k * baseline.

    baseline = trailing average of the SAME intraday slot's volume (i.e. the
    opening candle's volume averaged over the prior N opening candles), so we
    compare like-for-like (an opening 30m bar vs prior opening 30m bars), not
    against a flat all-day mean. ex#5 (VOLTAS wide-CPR, weak vol -> skip a
    winner) and ex#6 (VOLTAS 27-Apr, weak vol -> correctly reject a failure)
    make this gate mandatory.
    """
    if baseline is None or baseline <= 0 or np.isnan(baseline):
        return False
    return float(trigger_vol) >= k * float(baseline)


def slot_baseline(opening_vols: pd.Series, idx: int, n: int = 20) -> Optional[float]:
    """Trailing mean of the prior `n` opening-candle volumes (strictly before
    position `idx`). Returns None if fewer than `min(n, 5)` priors exist."""
    if idx <= 0:
        return None
    lo = max(0, idx - n)
    window = opening_vols.iloc[lo:idx]
    if len(window) < min(n, 5):
        return None
    v = float(window.mean())
    return v if v > 0 else None


# ---------------------------------------------------------------------------
# Optional: clear room ahead (ex#6 / ex#9 failure hypothesis) — grid axis
# ---------------------------------------------------------------------------

def clear_room(entry_price: float, direction: str, atr_daily: Optional[float],
                recent_swing_high: Optional[float],
                recent_swing_low: Optional[float],
                r_atr: float = 1.0) -> bool:
    """True iff there is NO opposing structural level within `r_atr * ATR`
    of entry in the trade direction.

    Both VOLTAS-27-Apr (ex#6) and TMPV-19-Jan (ex#9) failures broke straight
    INTO a nearby opposing S/R shelf. The 'clear room' axis tests whether
    requiring open headroom in the trade direction improves expectancy.

    If ATR is unavailable we cannot judge headroom -> return True (i.e. the
    filter is a no-op rather than silently blocking every trade).
    """
    if atr_daily is None or atr_daily <= 0 or np.isnan(atr_daily):
        return True
    margin = r_atr * atr_daily
    if direction == "long":
        if recent_swing_high is None or np.isnan(recent_swing_high):
            return True
        return (recent_swing_high - entry_price) >= margin
    elif direction == "short":
        if recent_swing_low is None or np.isnan(recent_swing_low):
            return True
        return (entry_price - recent_swing_low) >= margin
    return True


def swing_levels(daily_df: pd.DataFrame, as_of: pd.Timestamp,
                 lookback: int = 20) -> tuple[Optional[float], Optional[float]]:
    """Nearest opposing swing high/low from the prior `lookback` daily bars
    strictly before `as_of` (used by clear_room)."""
    if daily_df is None or daily_df.empty:
        return None, None
    prior = daily_df.loc[daily_df.index < pd.Timestamp(as_of).normalize()]
    if prior.empty:
        return None, None
    prior = prior.tail(lookback)
    return float(prior["high"].max()), float(prior["low"].min())
