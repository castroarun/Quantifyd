"""Signal generators for the short-options signal sweep.

Each generator takes a precomputed per-symbol context (5-min OHLC + daily CPR
table + indicators) and yields **at most one signal per trading day** for
its path's window.

A signal is a `Signal` dataclass with everything Phase-1 needs to log a
trajectory and everything Phase-2 needs to evaluate exit policies.

Currently implemented:
- `path_a_signals`  — Path A: early breakout (09:35/09:40/09:45) with RSI confirm.

Stubs for Paths B/C/D and Strategies E/F live below — to be filled in
once the smoke test on Path A passes review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from data_loader import slice_session
from indicators import (
    cpr_direction,
    cpr_from_prev,
    daily_cpr_table,
    ema as ema_func,
    or15_levels,
    rsi as rsi_func,
    running_extremes,
    session_gap,
    vwap_intraday,
)

Direction = Literal["long", "short"]


@dataclass(frozen=True)
class Signal:
    path: str               # "A" / "B" / "C" / "D" / "E" / "F"
    variant: str            # human-readable variant tag, e.g. "gap0.5_rsi40_60"
    symbol: str
    timeframe: str          # "5min" / "10min" / "15min"
    date: pd.Timestamp      # trading date (normalized)
    signal_time: pd.Timestamp  # candle close timestamp that triggered
    direction: Direction
    signal_price: float     # close of the trigger candle
    level_for_T4: float     # the level the price re-crossing it would invalidate
                            # (OR level for A; signal-candle open for B/D; day extreme for C; etc.)
    extras: dict            # path-specific diagnostic fields (RSI value, gap, etc.)


# ---------------------------------------------------------------------------
# Path A — early breakout + RSI confirm (NIFTY)
# ---------------------------------------------------------------------------

PATH_A_CHECK_TIMES = (time(9, 35), time(9, 40), time(9, 45))


def path_a_signals(
    df_5min: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    gap_threshold: float | None = 0.005,  # |gap| <= 0.5% to pass; None = off
    rsi_low: float = 40.0,
    rsi_high: float = 60.0,
    rsi_period: int = 14,
    symbol: str = "NIFTY50",
    timeframe: str = "5min",
) -> Iterator[Signal]:
    """Yield at most one Path A signal per session.

    Rules
    -----
    - Skip the day if `gap_threshold` is set and `|gap| > gap_threshold`.
    - OR15 = high/low of 09:15, 09:20, 09:25 candles.
    - For 5-min candles closing at 09:35, 09:40, 09:45 (in that order):
        * long  iff close > OR_high AND RSI(14) >= rsi_high
        * short iff close < OR_low  AND RSI(14) <= rsi_low
    - The first candle (in time order) that triggers wins; later candles
      that day are ignored. (One signal per day per variant.)
    """
    if df_5min.empty or daily.empty:
        return
    variant = f"gap{'off' if gap_threshold is None else f'{gap_threshold:.3f}'}_rsi{int(rsi_low)}_{int(rsi_high)}"

    # Pre-compute RSI on the entire 5-min series (Wilder smoothing carries
    # across days — that's fine for an intraday RSI driven by recent action).
    rsi_full = rsi_func(df_5min["close"], rsi_period)

    daily_sorted = daily.sort_index()
    days = sorted(df_5min.index.normalize().unique())

    for day in days:
        sess = slice_session(df_5min, day)
        if sess.empty:
            continue

        # gap filter (need prev daily close)
        prev_close_rows = daily_sorted.loc[daily_sorted.index < day]
        if prev_close_rows.empty:
            continue
        prev_close = float(prev_close_rows["close"].iloc[-1])
        gap = session_gap(sess, prev_close)
        if gap_threshold is not None and (np.isnan(gap) or abs(gap) > gap_threshold):
            continue

        or_levels = or15_levels(sess)
        if or_levels is None:
            continue
        or_high, or_low = or_levels

        # iterate the three candidate post-OR candles
        sess_times = sess.index.time
        for t in PATH_A_CHECK_TIMES:
            mask = sess_times == t
            if not mask.any():
                continue
            row = sess.loc[mask].iloc[0]
            ts = sess.index[mask][0]
            close = float(row["close"])
            r = float(rsi_full.loc[ts]) if ts in rsi_full.index else float("nan")
            if np.isnan(r):
                continue
            if close > or_high and r >= rsi_high:
                yield Signal(
                    path="A",
                    variant=variant,
                    symbol=symbol,
                    timeframe=timeframe,
                    date=pd.Timestamp(day),
                    signal_time=ts,
                    direction="long",
                    signal_price=close,
                    level_for_T4=or_high,
                    extras={"or_high": or_high, "or_low": or_low, "rsi": r, "gap": gap},
                )
                break  # first signal of the day wins
            if close < or_low and r <= rsi_low:
                yield Signal(
                    path="A",
                    variant=variant,
                    symbol=symbol,
                    timeframe=timeframe,
                    date=pd.Timestamp(day),
                    signal_time=ts,
                    direction="short",
                    signal_price=close,
                    level_for_T4=or_low,
                    extras={"or_high": or_high, "or_low": or_low, "rsi": r, "gap": gap},
                )
                break


# ---------------------------------------------------------------------------
# Stubs — implement after Path A smoke test passes
# ---------------------------------------------------------------------------

def path_b_signals(
    df_5min: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    rsi_low: float = 40.0,
    rsi_high: float = 60.0,
    rsi_period: int = 14,
    start_time: time = time(11, 0),
    symbol: str = "NIFTY50",
    timeframe: str = "5min",
) -> Iterator[Signal]:
    """Path B — post-11:00 RSI zone-entry. One signal per session (first
    candle whose close drives RSI from outside the zone (rsi_low, rsi_high)
    into either tail). The candle's open is recorded as the level whose
    re-cross would invalidate the signal (used by exit policy T4).

    Note: `daily` is unused but kept in the signature for a uniform
    runner dispatch.
    """
    if df_5min.empty:
        return
    variant = (
        f"rsi{int(rsi_low)}_{int(rsi_high)}_from{start_time.strftime('%H%M')}"
    )
    rsi_full = rsi_func(df_5min["close"], rsi_period)

    days = sorted(df_5min.index.normalize().unique())
    for day in days:
        sess = slice_session(df_5min, day)
        if sess.empty:
            continue
        in_zone_prev = False  # RSI in zone on the immediately prior candle
        first_obs = True
        for ts, row in sess.iterrows():
            r = float(rsi_full.loc[ts]) if ts in rsi_full.index else float("nan")
            if np.isnan(r):
                continue
            now_in_zone = (r >= rsi_high) or (r <= rsi_low)
            # update the "previous-candle in-zone" tracker for any candle
            # before the trigger window — we just want the rolling state.
            if ts.time() < start_time:
                in_zone_prev = now_in_zone
                first_obs = False
                continue
            # within window: detect entry into zone (was out, now in)
            if first_obs:
                # First candle of the day actually observed in window — we
                # don't have a confirmed prior-candle state. Treat the first
                # observation conservatively: if already in zone, skip
                # (we've missed the entry); if outside, set baseline.
                in_zone_prev = now_in_zone
                first_obs = False
                continue
            crossed_in = now_in_zone and not in_zone_prev
            if crossed_in:
                close = float(row["close"])
                direction: Direction = "long" if r >= rsi_high else "short"
                yield Signal(
                    path="B",
                    variant=variant,
                    symbol=symbol,
                    timeframe=timeframe,
                    date=pd.Timestamp(day),
                    signal_time=ts,
                    direction=direction,
                    signal_price=close,
                    level_for_T4=float(row["open"]),  # signal-candle open
                    extras={"rsi": r, "candle_open": float(row["open"])},
                )
                break  # one signal per day
            in_zone_prev = now_in_zone


# ---------------------------------------------------------------------------
# Path C — post-12:00 day-extreme break + range compression (NIFTY)
# ---------------------------------------------------------------------------

def path_c_signals(
    df_5min: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    range_threshold: float | None = 0.006,  # range/open <= 0.6% to pass; None = off
    use_rsi: bool = False,
    rsi_low: float = 40.0,
    rsi_high: float = 60.0,
    rsi_period: int = 14,
    start_time: time = time(12, 0),
    end_time: time = time(15, 15),
    symbol: str = "NIFTY50",
    timeframe: str = "5min",
) -> Iterator[Signal]:
    """Path C — post-12:00 day-extreme break + range compression.

    For each session, in window [start_time, end_time], the first candle whose
    close exceeds the day's running high (long) or breaks the day's running
    low (short) AND for which the day's range so far / day's open is below
    `range_threshold` triggers a signal. RSI gate optional.
    """
    if df_5min.empty:
        return
    rng_tag = "off" if range_threshold is None else f"{range_threshold:.3f}"
    rsi_tag = f"rsi{int(rsi_low)}_{int(rsi_high)}" if use_rsi else "norsi"
    variant = f"rng{rng_tag}_{rsi_tag}"

    rsi_full = rsi_func(df_5min["close"], rsi_period)
    days = sorted(df_5min.index.normalize().unique())
    for day in days:
        sess = slice_session(df_5min, day)
        if sess.empty:
            continue
        day_open = float(sess["open"].iloc[0])
        if day_open == 0:
            continue
        # running extremes prior to this candle
        ext = running_extremes(sess)
        prev_high = ext["day_high_so_far"].shift(1)
        prev_low = ext["day_low_so_far"].shift(1)
        # running range
        running_high_inc = ext["day_high_so_far"]
        running_low_inc = ext["day_low_so_far"]

        sess_times = sess.index.time
        in_window = (sess_times >= start_time) & (sess_times <= end_time)
        for ts, row in sess.loc[in_window].iterrows():
            ph = prev_high.loc[ts]
            pl = prev_low.loc[ts]
            if pd.isna(ph) or pd.isna(pl):
                continue
            close = float(row["close"])
            # range up to but not including this candle (use prev_high/prev_low)
            range_so_far = float(ph - pl)
            range_pct = range_so_far / day_open
            if range_threshold is not None and range_pct > range_threshold:
                continue
            r = float(rsi_full.loc[ts]) if ts in rsi_full.index else float("nan")
            direction: Direction | None = None
            level: float | None = None
            if close > ph:
                if use_rsi and (np.isnan(r) or r < rsi_high):
                    pass
                else:
                    direction = "long"
                    level = float(ph)
            elif close < pl:
                if use_rsi and (np.isnan(r) or r > rsi_low):
                    pass
                else:
                    direction = "short"
                    level = float(pl)
            if direction is not None:
                yield Signal(
                    path="C",
                    variant=variant,
                    symbol=symbol,
                    timeframe=timeframe,
                    date=pd.Timestamp(day),
                    signal_time=ts,
                    direction=direction,
                    signal_price=close,
                    level_for_T4=level,
                    extras={
                        "rsi": r if not np.isnan(r) else None,
                        "range_pct": range_pct,
                        "day_high_so_far": float(ph),
                        "day_low_so_far": float(pl),
                    },
                )
                break  # one signal per day


# ---------------------------------------------------------------------------
# Path D — post-12:00 RSI drift + CPR alignment (NIFTY)
# ---------------------------------------------------------------------------

def path_d_signals(
    df_5min: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    cpr_convention: Literal["priceCPR", "cprDelta"] = "priceCPR",
    rsi_low: float = 40.0,
    rsi_high: float = 60.0,
    rsi_period: int = 14,
    start_time: time = time(12, 0),
    end_time: time = time(15, 15),
    symbol: str = "NIFTY50",
    timeframe: str = "5min",
) -> Iterator[Signal]:
    """Path D — post-12:00 RSI drift + CPR alignment.

    Long when RSI >= rsi_high AND CPR direction = bullish.
    Short when RSI <= rsi_low AND CPR direction = bearish.
    First qualifying candle in window.
    """
    if df_5min.empty or daily.empty:
        return
    variant = f"{cpr_convention}_rsi{int(rsi_low)}_{int(rsi_high)}"

    cpr_table = daily_cpr_table(daily)
    if cpr_table.empty:
        return
    rsi_full = rsi_func(df_5min["close"], rsi_period)
    days = sorted(df_5min.index.normalize().unique())

    for day in days:
        if day not in cpr_table.index:
            continue
        cpr_row = cpr_table.loc[day]
        cpr_levels = cpr_from_prev(0, 0, 0)  # placeholder, will rebuild from row
        # rebuild a CPRLevels-like object from cpr_table row
        from indicators import CPRLevels as _CPR  # local import to avoid circular
        cpr_levels = _CPR(
            pivot=float(cpr_row["pivot"]),
            top=float(cpr_row["top"]),
            bottom=float(cpr_row["bottom"]),
            width=float(cpr_row["width"]),
        )
        prev_pivot = float(cpr_row["prev_pivot"]) if not pd.isna(cpr_row["prev_pivot"]) else None

        sess = slice_session(df_5min, day)
        if sess.empty:
            continue
        sess_times = sess.index.time
        in_window = (sess_times >= start_time) & (sess_times <= end_time)
        for ts, row in sess.loc[in_window].iterrows():
            r = float(rsi_full.loc[ts]) if ts in rsi_full.index else float("nan")
            if np.isnan(r):
                continue
            close = float(row["close"])
            cprdir = cpr_direction(
                convention=cpr_convention,
                cpr=cpr_levels,
                price=close if cpr_convention == "priceCPR" else None,
                prev_pivot=prev_pivot if cpr_convention == "cprDelta" else None,
            )
            direction: Direction | None = None
            if r >= rsi_high and cprdir == "bullish":
                direction = "long"
            elif r <= rsi_low and cprdir == "bearish":
                direction = "short"
            if direction is not None:
                yield Signal(
                    path="D",
                    variant=variant,
                    symbol=symbol,
                    timeframe=timeframe,
                    date=pd.Timestamp(day),
                    signal_time=ts,
                    direction=direction,
                    signal_price=close,
                    level_for_T4=float(row["open"]),  # signal-candle open
                    extras={
                        "rsi": r,
                        "cpr_pivot": cpr_levels.pivot,
                        "cpr_top": cpr_levels.top,
                        "cpr_bottom": cpr_levels.bottom,
                        "cpr_dir": cprdir,
                    },
                )
                break


# ---------------------------------------------------------------------------
# Strategy E — first-candle Open=Low / Open=High break (HELD STOCKS)
# ---------------------------------------------------------------------------

def strategy_e_signals(
    df_intraday: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    filter_mode: Literal["base", "cpr", "rsi", "cpr_rsi"] = "base",
    rsi_low: float = 40.0,
    rsi_high: float = 60.0,
    rsi_period: int = 14,
    cpr_convention: Literal["priceCPR", "cprDelta"] = "priceCPR",
    eq_tolerance: float = 0.0,  # absolute pts — Open == Low / Open == High strict by default
    symbol: str = "STOCK",
    timeframe: str = "5min",
) -> Iterator[Signal]:
    """Strategy E — first-candle Open=Low/High break.

    Day's first candle:
      - Open == Low  -> long bias (trigger when subsequent candle CLOSES > 1st-candle high)
      - Open == High -> short bias (trigger when subsequent candle CLOSES < 1st-candle low)
    Filter variants:
      - base     : just bias + breakout
      - cpr      : also require CPR direction alignment (long w/ bullish, short w/ bearish)
      - rsi      : also require RSI >= rsi_high (long) or <= rsi_low (short) at trigger
      - cpr_rsi  : both
    First qualifying candle in the day after the first candle.
    """
    if df_intraday.empty:
        return
    variant = f"{filter_mode}_{timeframe}"
    if filter_mode in ("rsi", "cpr_rsi"):
        variant = f"{filter_mode}_rsi{int(rsi_low)}_{int(rsi_high)}_{timeframe}"

    rsi_full = rsi_func(df_intraday["close"], rsi_period) if filter_mode in ("rsi", "cpr_rsi") else None
    cpr_table = daily_cpr_table(daily) if filter_mode in ("cpr", "cpr_rsi") and not daily.empty else None

    days = sorted(df_intraday.index.normalize().unique())
    for day in days:
        sess = slice_session(df_intraday, day)
        if len(sess) < 2:
            continue
        first = sess.iloc[0]
        first_open = float(first["open"])
        first_high = float(first["high"])
        first_low = float(first["low"])
        bias: Direction | None = None
        # Open == Low (within tolerance)
        if abs(first_open - first_low) <= eq_tolerance:
            bias = "long"
        elif abs(first_open - first_high) <= eq_tolerance:
            bias = "short"
        if bias is None:
            continue

        # CPR setup
        cprdir = None
        cpr_levels = None
        prev_pivot_v = None
        if cpr_table is not None and day in cpr_table.index:
            from indicators import CPRLevels as _CPR
            row = cpr_table.loc[day]
            cpr_levels = _CPR(
                pivot=float(row["pivot"]),
                top=float(row["top"]),
                bottom=float(row["bottom"]),
                width=float(row["width"]),
            )
            prev_pivot_v = float(row["prev_pivot"]) if not pd.isna(row["prev_pivot"]) else None

        # walk subsequent candles
        for ts, row in sess.iloc[1:].iterrows():
            close = float(row["close"])
            triggered = False
            level: float | None = None
            if bias == "long" and close > first_high:
                triggered = True
                level = first_high
            elif bias == "short" and close < first_low:
                triggered = True
                level = first_low
            if not triggered:
                continue

            # Filters
            r = None
            if rsi_full is not None and ts in rsi_full.index:
                rv = float(rsi_full.loc[ts])
                r = rv if not np.isnan(rv) else None
            if filter_mode in ("rsi", "cpr_rsi"):
                if r is None:
                    continue
                if bias == "long" and r < rsi_high:
                    continue
                if bias == "short" and r > rsi_low:
                    continue
            if filter_mode in ("cpr", "cpr_rsi"):
                if cpr_levels is None:
                    continue
                cprdir = cpr_direction(
                    convention=cpr_convention,
                    cpr=cpr_levels,
                    price=close if cpr_convention == "priceCPR" else None,
                    prev_pivot=prev_pivot_v if cpr_convention == "cprDelta" else None,
                )
                if bias == "long" and cprdir != "bullish":
                    continue
                if bias == "short" and cprdir != "bearish":
                    continue

            yield Signal(
                path="E",
                variant=variant,
                symbol=symbol,
                timeframe=timeframe,
                date=pd.Timestamp(day),
                signal_time=ts,
                direction=bias,
                signal_price=close,
                level_for_T4=float(level),
                extras={
                    "first_open": first_open,
                    "first_high": first_high,
                    "first_low": first_low,
                    "rsi": r,
                    "cpr_dir": cprdir,
                },
            )
            break


# ---------------------------------------------------------------------------
# Strategy F — EMA(9) cross VWAP + CPR alignment (HELD STOCKS, 5-min only)
# ---------------------------------------------------------------------------

def strategy_f_signals(
    df_5min: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    ema_period: int = 9,
    cpr_convention: Literal["priceCPR", "cprDelta"] = "priceCPR",
    symbol: str = "STOCK",
    timeframe: str = "5min",
) -> Iterator[Signal]:
    """Strategy F — first session EMA(9) cross of VWAP with CPR alignment.

    Long: EMA(9) crosses ABOVE VWAP AND price > CPR top (priceCPR) or
          today's CPR pivot > yesterday's (cprDelta).
    Short: EMA(9) crosses BELOW VWAP AND price < CPR bottom or pivot lower.

    EMA and VWAP are computed per-session (anchored). VWAP requires non-zero
    volume — for instruments with no volume (indices) this generator yields
    nothing.
    """
    if df_5min.empty or daily.empty:
        return
    variant = f"{cpr_convention}"

    cpr_table = daily_cpr_table(daily)
    if cpr_table.empty:
        return

    days = sorted(df_5min.index.normalize().unique())
    for day in days:
        if day not in cpr_table.index:
            continue
        sess = slice_session(df_5min, day)
        if sess.empty or len(sess) < ema_period + 2:
            continue
        # Per-session EMA and VWAP
        ema_s = ema_func(sess["close"], ema_period)
        vwap_s = vwap_intraday(sess)
        if vwap_s.isna().all():
            continue  # no volume => no VWAP => skip

        from indicators import CPRLevels as _CPR
        row = cpr_table.loc[day]
        cpr_levels = _CPR(
            pivot=float(row["pivot"]),
            top=float(row["top"]),
            bottom=float(row["bottom"]),
            width=float(row["width"]),
        )
        prev_pivot_v = float(row["prev_pivot"]) if not pd.isna(row["prev_pivot"]) else None

        diff = ema_s - vwap_s
        prev_diff = diff.shift(1)
        for ts in sess.index:
            d_now = diff.loc[ts]
            d_prev = prev_diff.loc[ts]
            if pd.isna(d_now) or pd.isna(d_prev):
                continue
            close = float(sess.loc[ts, "close"])
            direction: Direction | None = None
            if d_prev <= 0 and d_now > 0:
                # cross above
                cprdir = cpr_direction(
                    convention=cpr_convention,
                    cpr=cpr_levels,
                    price=close if cpr_convention == "priceCPR" else None,
                    prev_pivot=prev_pivot_v if cpr_convention == "cprDelta" else None,
                )
                if cprdir == "bullish":
                    direction = "long"
            elif d_prev >= 0 and d_now < 0:
                cprdir = cpr_direction(
                    convention=cpr_convention,
                    cpr=cpr_levels,
                    price=close if cpr_convention == "priceCPR" else None,
                    prev_pivot=prev_pivot_v if cpr_convention == "cprDelta" else None,
                )
                if cprdir == "bearish":
                    direction = "short"
            if direction is not None:
                yield Signal(
                    path="F",
                    variant=variant,
                    symbol=symbol,
                    timeframe=timeframe,
                    date=pd.Timestamp(day),
                    signal_time=ts,
                    direction=direction,
                    signal_price=close,
                    level_for_T4=float(vwap_s.loc[ts]),  # VWAP at signal — re-cross invalidates
                    extras={
                        "ema": float(ema_s.loc[ts]),
                        "vwap": float(vwap_s.loc[ts]),
                        "ema_minus_vwap": float(d_now),
                        "cpr_dir": "bullish" if direction == "long" else "bearish",
                    },
                )
                break


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import INDEX_SYMBOL, load_5min, load_daily

    nifty_5m = load_5min(INDEX_SYMBOL, "2024-03-04", "2024-04-30")
    nifty_d = load_daily(INDEX_SYMBOL)

    sigs = list(
        path_a_signals(
            nifty_5m,
            nifty_d,
            gap_threshold=0.005,
            rsi_low=40,
            rsi_high=60,
            symbol=INDEX_SYMBOL,
        )
    )
    print(f"Path A: {len(sigs)} signals over {len(set(nifty_5m.index.normalize()))} sessions")
    for s in sigs[:10]:
        print(
            f"  {s.date.date()}  {s.signal_time.time()}  {s.direction:5s}  "
            f"close={s.signal_price:.1f}  OR=({s.extras['or_low']:.1f},{s.extras['or_high']:.1f})  "
            f"RSI={s.extras['rsi']:.1f}  gap={s.extras['gap']*100:.2f}%"
        )
