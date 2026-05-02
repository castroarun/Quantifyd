"""CCRB — CPR-Compression Range Breakout — signal generator.

For each (symbol x timeframe), walk session bars and emit a signal on the
FIRST fresh transition past prev_day_high (long) / prev_day_low (short),
ONLY on days that pass the daily-bar setup filter:

  today_cpr_width / today_open <= today_narrow_threshold
  AND yesterday_ctx in {W, N, W_OR_N, W_AND_N} qualifies

Designed for the streaming runner — yields Signal dataclasses, no I/O,
no DataFrame side-effects on its inputs.

This module deliberately precomputes the daily-setup-mask once per
(symbol x today_narrow_threshold x yesterday_wide_thresh x yesterday_narrow_range_thresh)
group and reuses it across direction/volume cells, since the daily filter
is independent of those axes.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import time as dtime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

# Reuse research/29 helpers
SCRIPTS_29 = Path(__file__).resolve().parents[2] / "29_short_options_signal_sweep" / "scripts"
sys.path.insert(0, str(SCRIPTS_29))
from data_loader import resample, slice_session  # noqa: E402

ENTRY_OPEN_TIME = dtime(9, 20)   # give CPR + first OR a moment
ENTRY_CUTOFF_TIME = dtime(14, 0)  # last entry-eligible bar START


# ---------------------------------------------------------------------------
# CPR helper
# ---------------------------------------------------------------------------

def _cpr(prev_high: float, prev_low: float, prev_close: float) -> tuple[float, float, float, float]:
    """Return (pivot, top_central, bottom_central, width)."""
    pivot = (prev_high + prev_low + prev_close) / 3.0
    bc = (prev_high + prev_low) / 2.0
    tc = 2.0 * pivot - bc
    top, bottom = (tc, bc) if tc >= bc else (bc, tc)
    return pivot, top, bottom, abs(top - bottom)


# ---------------------------------------------------------------------------
# Daily-setup table
# ---------------------------------------------------------------------------

def daily_setup_table(daily: pd.DataFrame) -> pd.DataFrame:
    """For each trading day with at least 2 priors, compute:
       - today_open (= today's daily open)
       - today_cpr_width, today_cpr_width_pct (vs today_open)
       - prev_day_high/low/close/open
       - prev_cpr_width, prev_cpr_width_pct (vs prev_open)
       - prev_range, prev_range_pct (prev_high - prev_low / prev_open)

    today_cpr_width is computed from prev_day_high/low/close (standard CPR).
    prev_cpr_width is computed from the day-before-prev's H/L/C.

    Indexed by today's trading date (normalized).
    """
    if daily.empty or len(daily) < 3:
        return pd.DataFrame()
    df = daily.sort_index().copy()
    rows = []
    # i is "today"; need i-1 (prev) and i-2 (day-before-prev)
    for i in range(2, len(df)):
        today = df.iloc[i]
        prev = df.iloc[i - 1]
        ppv = df.iloc[i - 2]
        # today's CPR comes from prev's HLC
        _, _, _, today_w = _cpr(prev["high"], prev["low"], prev["close"])
        # prev's CPR comes from ppv's HLC
        _, _, _, prev_w = _cpr(ppv["high"], ppv["low"], ppv["close"])
        prev_range = prev["high"] - prev["low"]
        rows.append({
            "date": df.index[i].normalize(),
            "today_open": float(today["open"]),
            "today_cpr_width": float(today_w),
            "today_cpr_width_pct": float(today_w / today["open"]) if today["open"] > 0 else float("nan"),
            "prev_open": float(prev["open"]),
            "prev_high": float(prev["high"]),
            "prev_low": float(prev["low"]),
            "prev_close": float(prev["close"]),
            "prev_cpr_width": float(prev_w),
            "prev_cpr_width_pct": float(prev_w / prev["open"]) if prev["open"] > 0 else float("nan"),
            "prev_range": float(prev_range),
            "prev_range_pct": float(prev_range / prev["open"]) if prev["open"] > 0 else float("nan"),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("date")
    return out


def qualifying_dates(
    setup: pd.DataFrame,
    *,
    today_narrow: float,
    yesterday_ctx: str,           # "W", "N", "W_OR_N", "W_AND_N"
    yesterday_wide_thresh: float,
    yesterday_narrow_range_thresh: float,
) -> set[pd.Timestamp]:
    """Set of trading-day Timestamps that pass the daily-bar filter."""
    if setup.empty:
        return set()
    today_ok = setup["today_cpr_width_pct"] <= today_narrow
    wide_ok = setup["prev_cpr_width_pct"] >= yesterday_wide_thresh
    narrow_ok = setup["prev_range_pct"] <= yesterday_narrow_range_thresh

    if yesterday_ctx == "W":
        ctx_ok = wide_ok
    elif yesterday_ctx == "N":
        ctx_ok = narrow_ok
    elif yesterday_ctx == "W_OR_N":
        ctx_ok = wide_ok | narrow_ok
    elif yesterday_ctx == "W_AND_N":
        ctx_ok = wide_ok & narrow_ok
    else:
        raise ValueError(f"unknown yesterday_ctx: {yesterday_ctx}")

    mask = today_ok & ctx_ok
    return set(setup.index[mask].tolist())


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class CCRBSignal:
    symbol: str
    cohort: str
    timeframe: str
    variant_tag: str
    direction: str            # "long" / "short"
    date: pd.Timestamp        # session date (normalized)
    signal_time: pd.Timestamp
    entry_price: float
    prev_day_high: float
    prev_day_low: float
    prev_day_close: float
    today_open: float
    today_cpr_width_pct: float
    yesterday_cpr_width_pct: float
    yesterday_range_pct: float
    bar_volume: float
    vol_ratio: float
    bar_open: float
    bar_high: float
    bar_low: float
    bar_close: float


# ---------------------------------------------------------------------------
# 20-day same-bar-position volume average — for volume-confirm filter
# ---------------------------------------------------------------------------

def build_bar_pos_vol_avg(df_tf: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """For each (intraday) bar, return the trailing 20-session average
    volume of the SAME bar position (e.g., 09:35 vs prior 20 sessions'
    09:35). Returns a Series aligned with df_tf.index, NaN where lookback
    isn't satisfied or volume is zero across the board.
    """
    if df_tf.empty or "volume" not in df_tf.columns:
        return pd.Series(dtype=float, index=df_tf.index)
    s = df_tf.copy()
    s["_t"] = s.index.time
    s["_d"] = s.index.normalize()
    # Pivot: rows = day, cols = bar_time, values = volume
    pv = s.pivot_table(index="_d", columns="_t", values="volume", aggfunc="last")
    # Trailing-20 mean (excluding today)
    avg = pv.shift(1).rolling(lookback, min_periods=lookback).mean()
    # Map back to original index
    out = pd.Series(np.nan, index=df_tf.index, dtype=float)
    for ts in df_tf.index:
        d = ts.normalize()
        t = ts.time()
        try:
            v = avg.at[d, t]
            if pd.notna(v):
                out.at[ts] = float(v)
        except KeyError:
            pass
    return out


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def ccrb_signals(
    df5: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    symbol: str,
    cohort: str,
    timeframe: str,
    today_narrow: float,
    yesterday_ctx: str,
    yesterday_wide_thresh: float,
    yesterday_narrow_range_thresh: float,
    vol_mode: str,                  # "off", "vm1.5", "vm2.0"
    direction: str,                 # "long" / "short"
    setup_table: Optional[pd.DataFrame] = None,
    qualifying_set: Optional[set[pd.Timestamp]] = None,
    df_tf_cache: Optional[pd.DataFrame] = None,
    bar_pos_avg_cache: Optional[pd.Series] = None,
) -> Iterator[CCRBSignal]:
    """Yield CCRBSignal for each qualifying day where a fresh range break
    fires intraday between 09:20 and 14:00 IST.

    Pre-computed inputs (`setup_table`, `qualifying_set`, `df_tf_cache`,
    `bar_pos_avg_cache`) are accepted to let the runner share work across
    the variant grid (e.g., direction long/short scan the same intraday
    bars for the same qualifying-day set).
    """
    if df5.empty:
        return
    if setup_table is None:
        setup_table = daily_setup_table(daily)
    if setup_table.empty:
        return
    if qualifying_set is None:
        qualifying_set = qualifying_dates(
            setup_table,
            today_narrow=today_narrow,
            yesterday_ctx=yesterday_ctx,
            yesterday_wide_thresh=yesterday_wide_thresh,
            yesterday_narrow_range_thresh=yesterday_narrow_range_thresh,
        )
    if not qualifying_set:
        return

    df_tf = df_tf_cache if df_tf_cache is not None else (
        df5 if timeframe == "5min" else resample(df5, timeframe)
    )
    if df_tf.empty:
        return

    # Volume thresholds
    if vol_mode == "off":
        vmult = None
    elif vol_mode == "vm1.5":
        vmult = 1.5
    elif vol_mode == "vm2.0":
        vmult = 2.0
    else:
        raise ValueError(f"unknown vol_mode: {vol_mode}")

    bar_pos_avg = bar_pos_avg_cache
    if vmult is not None and bar_pos_avg is None:
        bar_pos_avg = build_bar_pos_vol_avg(df_tf, lookback=20)

    # Use 4-decimal tags so 0.0030 / 0.0040 / 0.0050 are distinguishable.
    variant_tag = (
        f"t{today_narrow:.4f}"
        f"_ctx{yesterday_ctx}"
        f"_w{yesterday_wide_thresh:.4f}"
        f"_n{yesterday_narrow_range_thresh:.4f}"
        f"_{timeframe}"
        f"_{vol_mode}"
        f"_{direction[0]}"
    )

    # Group bars by trading day for fast day-walk
    df_tf = df_tf.sort_index()
    by_day = df_tf.groupby(df_tf.index.normalize(), sort=True)

    for day, day_df in by_day:
        if day not in qualifying_set:
            continue
        if day not in setup_table.index:
            continue
        st = setup_table.loc[day]
        prev_high = st["prev_high"]
        prev_low = st["prev_low"]

        # Walk bars 09:20 .. 14:00. Track previous bar's close for fresh-transition check.
        # First fresh transition past prev_high (long) / prev_low (short) wins.
        prev_close: Optional[float] = None

        for ts, bar in day_df.iterrows():
            t = ts.time()
            if t < ENTRY_OPEN_TIME:
                prev_close = float(bar["close"])
                continue
            if t > ENTRY_CUTOFF_TIME:
                break

            bar_close = float(bar["close"])
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_vol = float(bar.get("volume", 0.0))

            # Need prev_close for fresh-transition test. If None (first bar
            # in window), seed with bar_open as "before bar start" proxy.
            ref = prev_close if prev_close is not None else bar_open

            fresh_long = (bar_close > prev_high) and (ref <= prev_high)
            fresh_short = (bar_close < prev_low) and (ref >= prev_low)

            triggered = (direction == "long" and fresh_long) or (
                direction == "short" and fresh_short
            )

            if not triggered:
                prev_close = bar_close
                continue

            # Volume filter (if any)
            vol_ratio = float("nan")
            if vmult is not None:
                avg_v = bar_pos_avg.at[ts] if (bar_pos_avg is not None and ts in bar_pos_avg.index) else np.nan
                if pd.isna(avg_v) or avg_v <= 0:
                    # Cannot evaluate -> reject
                    prev_close = bar_close
                    continue
                vol_ratio = bar_vol / avg_v
                if vol_ratio < vmult:
                    prev_close = bar_close
                    continue

            yield CCRBSignal(
                symbol=symbol,
                cohort=cohort,
                timeframe=timeframe,
                variant_tag=variant_tag,
                direction=direction,
                date=day,
                signal_time=ts,
                entry_price=bar_close,
                prev_day_high=float(prev_high),
                prev_day_low=float(prev_low),
                prev_day_close=float(st["prev_close"]),
                today_open=float(st["today_open"]),
                today_cpr_width_pct=float(st["today_cpr_width_pct"]),
                yesterday_cpr_width_pct=float(st["prev_cpr_width_pct"]),
                yesterday_range_pct=float(st["prev_range_pct"]),
                bar_volume=bar_vol,
                vol_ratio=vol_ratio,
                bar_open=bar_open,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_close=bar_close,
            )
            # One signal per (day, direction). Done with this day.
            break

            # (unreachable)


if __name__ == "__main__":
    # Quick sanity smoke
    from data_loader import load_5min, load_daily

    sym = "RELIANCE"
    df5 = load_5min(sym, "2024-01-01", "2024-12-31")
    daily = load_daily(sym, "2023-06-01", "2024-12-31")
    print(f"{sym}: 5m rows={len(df5)}, daily rows={len(daily)}")
    setup = daily_setup_table(daily)
    print(f"setup table rows: {len(setup)}")
    if not setup.empty:
        print(setup.head(3))

    q = qualifying_dates(setup, today_narrow=0.005, yesterday_ctx="W_OR_N",
                         yesterday_wide_thresh=0.005, yesterday_narrow_range_thresh=0.009)
    print(f"qualifying days (loosest): {len(q)}")

    n = 0
    for s in ccrb_signals(
        df5, daily,
        symbol=sym, cohort="A", timeframe="15min",
        today_narrow=0.005, yesterday_ctx="W_OR_N",
        yesterday_wide_thresh=0.005, yesterday_narrow_range_thresh=0.009,
        vol_mode="off", direction="long",
    ):
        n += 1
        if n <= 3:
            print(f"  {s.date.date()} {s.signal_time.time()} entry={s.entry_price:.2f} pdh={s.prev_day_high:.2f}")
    print(f"long signals fired: {n}")
