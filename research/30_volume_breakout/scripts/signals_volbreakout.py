"""Signal generator: volume-confirmed first-candle gap/breakout.

Hypothesis (from a Godfrey Phillips chart): the day after a strong close,
the first 5-min or first 15-min candle closes above the prior day's high
WITH a clear volume spike, followed by a strong intraday momentum run.

This module produces, per (stock, timeframe, variant, direction), a stream
of `VBSignal` objects -- one per session at most. The signal candle is the
FIRST candle of the day on the chosen timeframe (e.g. 09:15-09:20 for 5min,
09:15-09:30 for 15min).

Signal rules
------------
LONG (breakout):
- It's the first candle of the session
- close > prev_day_high
- volume > vol_mult * avg(first_bar_volume, 20 prior sessions on the same tf)
- (optional) today_open > prev_day_close * (1 + gap_pct)
- (optional) RSI(14) on 5-min at signal_time >= 60   -- note: for 15-min
  signals we still use the 5-min RSI snapshot at the signal-candle close
  for consistency with how downstream walk-forward operates.

SHORT (breakdown): mirror of long.
- close < prev_day_low
- volume > vol_mult * avg(first_bar_volume, 20 prior sessions)
- (optional) today_open < prev_day_close * (1 - gap_pct)
- (optional) RSI(14) <= 40
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
import pandas as pd

# Reuse the research/29 framework for data + indicators
SCRIPTS_29 = Path(__file__).resolve().parents[2] / "29_short_options_signal_sweep" / "scripts"
sys.path.insert(0, str(SCRIPTS_29))

from data_loader import slice_session  # noqa: E402
from indicators import rsi as rsi_func  # noqa: E402

Direction = Literal["long", "short"]


@dataclass(frozen=True)
class VBSignal:
    symbol: str
    timeframe: str          # "5min" or "15min"
    variant: str            # human-readable variant tag
    direction: Direction
    date: pd.Timestamp
    signal_time: pd.Timestamp   # candle close timestamp = end of first bar
    signal_price: float         # first-bar close
    first_bar_open: float
    first_bar_high: float
    first_bar_low: float
    first_bar_close: float
    first_bar_volume: float
    vol_avg_20d: float
    vol_ratio: float
    prev_day_high: float
    prev_day_low: float
    prev_day_close: float
    gap_pct: float
    rsi_at_signal: Optional[float]
    extras: dict


def vol_breakout_signals(
    df_5min: pd.DataFrame,
    df_first_bars_tf: pd.DataFrame,   # first-bar-of-session DataFrame on the selected tf
    daily: pd.DataFrame,
    *,
    vol_mult: float,
    gap_pct: Optional[float],         # None = filter off; else minimum |gap| ratio
    use_rsi: bool,
    rsi_low: float = 40.0,
    rsi_high: float = 60.0,
    rsi_period: int = 14,
    direction: Direction,
    symbol: str,
    timeframe: str,
    rsi_precomputed: Optional[pd.Series] = None,  # optional: precomputed 5-min RSI series
) -> Iterator[VBSignal]:
    """Yield at most one signal per session.

    `df_first_bars_tf` must be a DataFrame indexed by day-normalized timestamp
    with columns: ts (session-tagged timestamp at end of first bar), open, high,
    low, close, volume. It contains EXACTLY ONE row per session — the first
    candle of the day at the given timeframe.

    The volume average is the trailing 20-session mean of `volume` ON THIS
    SAME first-bar series (so 5min uses 5min first-bar averages and 15min
    uses 15min first-bar averages).
    """
    if df_first_bars_tf.empty or daily.empty:
        return

    gap_tag = "off" if gap_pct is None else f"{gap_pct:.3f}"
    rsi_tag = f"rsi{int(rsi_low)}_{int(rsi_high)}" if use_rsi else "norsi"
    variant = f"{direction[:1]}_vm{vol_mult}_gap{gap_tag}_{rsi_tag}"

    # 5-min RSI for the RSI-confirm filter
    if use_rsi:
        rsi_full = rsi_precomputed if rsi_precomputed is not None else rsi_func(df_5min["close"], rsi_period)
    else:
        rsi_full = None

    daily_sorted = daily.sort_index()

    # rolling mean of volume across first bars
    fb = df_first_bars_tf.sort_index().copy()
    fb["vol_avg_20d"] = fb["volume"].shift(1).rolling(20, min_periods=10).mean()

    for day, row in fb.iterrows():
        # Need at least 10 prior sessions for the volume baseline
        v_avg = row["vol_avg_20d"]
        if pd.isna(v_avg) or v_avg <= 0:
            continue
        # Need prior daily close/high/low
        prior = daily_sorted.loc[daily_sorted.index < day]
        if prior.empty:
            continue
        prev_close = float(prior["close"].iloc[-1])
        prev_high = float(prior["high"].iloc[-1])
        prev_low = float(prior["low"].iloc[-1])
        if prev_close <= 0:
            continue

        first_open = float(row["open"])
        first_high = float(row["high"])
        first_low = float(row["low"])
        first_close = float(row["close"])
        first_volume = float(row["volume"])
        signal_ts = pd.Timestamp(row["ts"])

        gap = (first_open - prev_close) / prev_close
        vol_ratio = first_volume / float(v_avg)

        # Volume confirm
        if vol_ratio < vol_mult:
            continue

        # Directional rules
        if direction == "long":
            if first_close <= prev_high:
                continue
            if gap_pct is not None and gap < gap_pct:
                continue
        else:  # short
            if first_close >= prev_low:
                continue
            if gap_pct is not None and gap > -gap_pct:
                continue

        # RSI gate (use 5-min RSI at signal_ts)
        rsi_v: Optional[float] = None
        if use_rsi:
            if rsi_full is None or signal_ts not in rsi_full.index:
                continue
            rv = float(rsi_full.loc[signal_ts])
            if np.isnan(rv):
                continue
            if direction == "long" and rv < rsi_high:
                continue
            if direction == "short" and rv > rsi_low:
                continue
            rsi_v = rv

        yield VBSignal(
            symbol=symbol,
            timeframe=timeframe,
            variant=variant,
            direction=direction,
            date=pd.Timestamp(day),
            signal_time=signal_ts,
            signal_price=first_close,
            first_bar_open=first_open,
            first_bar_high=first_high,
            first_bar_low=first_low,
            first_bar_close=first_close,
            first_bar_volume=first_volume,
            vol_avg_20d=float(v_avg),
            vol_ratio=vol_ratio,
            prev_day_high=prev_high,
            prev_day_low=prev_low,
            prev_day_close=prev_close,
            gap_pct=gap,
            rsi_at_signal=rsi_v,
            extras={},
        )


def build_first_bars(df_5min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Build a per-session 'first bar' DataFrame on the requested timeframe.

    timeframe in {"5min", "15min"}. Returns a DataFrame indexed by session date
    (normalized to midnight) with columns: ts, open, high, low, close, volume.

    For 5min: the first 09:15 candle of each session.
    For 15min: aggregate the first three 5min candles (09:15, 09:20, 09:25)
        into a single bar; ts = end of last constituent (i.e., the 09:25 bar
        timestamp). If any of those three are missing, skip the session.
    """
    if df_5min.empty:
        return df_5min
    if timeframe == "5min":
        from datetime import time as dtime
        mask = df_5min.index.time == dtime(9, 15)
        first = df_5min.loc[mask].copy()
        first["ts"] = first.index
        first.index = first.index.normalize()
        return first[["ts", "open", "high", "low", "close", "volume"]]

    if timeframe == "15min":
        from datetime import time as dtime
        rows = []
        # We need exactly 09:15, 09:20, 09:25 candles.
        targets = (dtime(9, 15), dtime(9, 20), dtime(9, 25))
        df = df_5min.copy()
        df["_day"] = df.index.normalize()
        for day, day_df in df.groupby("_day", sort=True):
            day_df = day_df.sort_index()
            day_times = set(day_df.index.time)
            if not all(t in day_times for t in targets):
                continue
            grp = day_df.loc[[t in targets for t in day_df.index.time]].iloc[:3]
            if len(grp) < 3:
                continue
            rows.append({
                "_idx": pd.Timestamp(day),
                "ts": grp.index[-1],   # close of 09:25 bar = end of first 15-min
                "open": float(grp["open"].iloc[0]),
                "high": float(grp["high"].max()),
                "low": float(grp["low"].min()),
                "close": float(grp["close"].iloc[-1]),
                "volume": float(grp["volume"].sum()),
            })
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows).set_index("_idx")
        out.index.name = "date"
        return out[["ts", "open", "high", "low", "close", "volume"]]

    raise ValueError(f"unsupported timeframe: {timeframe}")


if __name__ == "__main__":
    # Smoke test
    from data_loader import load_5min, load_daily
    sym = "RELIANCE"
    df5 = load_5min(sym, "2024-03-01", "2026-03-25")
    daily = load_daily(sym, "2023-03-01", "2026-03-25")
    print(f"{sym}: 5min rows={len(df5)}, daily rows={len(daily)}")
    fb5 = build_first_bars(df5, "5min")
    fb15 = build_first_bars(df5, "15min")
    print(f"first-bar 5min: {len(fb5)}, first-bar 15min: {len(fb15)}")
    # Long, vm=2, gap_pct=0.003, no RSI
    sigs = list(vol_breakout_signals(
        df5, fb5, daily,
        vol_mult=2.0, gap_pct=0.003, use_rsi=False,
        direction="long", symbol=sym, timeframe="5min",
    ))
    print(f"LONG signals (vm=2, gap=0.3%): {len(sigs)}")
    for s in sigs[:5]:
        print(f"  {s.date.date()}  vol_ratio={s.vol_ratio:.2f}  gap={s.gap_pct*100:.2f}%  "
              f"close={s.signal_price:.1f}  prev_high={s.prev_day_high:.1f}")
