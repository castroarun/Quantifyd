"""Prev-day range breakout — trade simulator (intraday + multi-day swing).

A breakout is one event per stock per day (first 5-min close beyond the prev-day
range). For that event we simulate every (stop-loss x target x hold) combination
and return each result in R-multiples. Compression gating is applied by the
caller (it's a cheap per-day boolean), so this module is pure trade mechanics.

Conventions: long = +1, short = -1. R = |entry - initial_SL|. Stop is checked
before target on same-bar ties (conservative). Swing carries an open position
onto subsequent DAILY bars; an overnight gap through the stop/target fills at the
next day's open.
"""
from __future__ import annotations

import numpy as np

SL_DEFS = ("BOX", "HALF", "ATR", "BAR")
TARGETS = ("1R", "2R", "3R", "MM", "TRAIL")
HOLDS = ("INTRA", "SWING")
ATR_SL_MULT = 1.5
TRAIL_MULT = 1.0      # trail 1R behind the run extreme
MAXDAYS = 10          # swing time stop (trading days held after entry day)


def sl_price(sl_def, direction, entry, pdl, pdh, prevrange, datr, bar_lo, bar_hi):
    long = direction > 0
    if sl_def == "BOX":
        return pdl if long else pdh
    if sl_def == "HALF":
        return entry - 0.5 * prevrange if long else entry + 0.5 * prevrange
    if sl_def == "ATR":
        return entry - ATR_SL_MULT * datr if long else entry + ATR_SL_MULT * datr
    if sl_def == "BAR":
        return bar_lo if long else bar_hi
    raise ValueError(sl_def)


def _target(tgt_kind, direction, entry, R, prevrange):
    if tgt_kind == "MM":
        return entry + prevrange if direction > 0 else entry - prevrange
    if tgt_kind in ("1R", "2R", "3R"):
        k = int(tgt_kind[0])
        return entry + k * R if direction > 0 else entry - k * R
    return None  # TRAIL


def simulate(direction, entry, SL, tgt_kind, prevrange,
             d5h, d5l, d5c, daily_after, hold):
    """Return (r_gross, hold_days) or None. r_gross in R; cost applied by caller.

    d5h/d5l/d5c: entry-day 5-min OHLC arrays from the bar AFTER entry to EOD.
    daily_after: list of (o,h,l,c) daily bars strictly after the entry day.
    """
    R = abs(entry - SL)
    if R <= 0:
        return None
    long = direction > 0
    target = _target(tgt_kind, direction, entry, R, prevrange)
    trailing = tgt_kind == "TRAIL"
    trail = SL
    run = entry

    def hit_long(lo, hi, op=None):
        """Return exit price if stopped/targeted this bar, else None. op=open for gaps."""
        nonlocal trail, run
        if op is not None:  # overnight gap check at open
            if trailing:
                if op <= trail:
                    return op
            else:
                if op <= SL:
                    return op
                if target is not None and op >= target:
                    return op
        run = max(run, hi)
        if trailing:
            trail = max(trail, run - TRAIL_MULT * R)
            if lo <= trail:
                return trail
        else:
            if lo <= SL:               # stop first
                return SL
            if target is not None and hi >= target:
                return target
        return None

    def hit_short(lo, hi, op=None):
        nonlocal trail, run
        if op is not None:
            if trailing:
                if op >= trail:
                    return op
            else:
                if op >= SL:
                    return op
                if target is not None and op <= target:
                    return op
        run = min(run, lo)
        if trailing:
            trail = min(trail, run + TRAIL_MULT * R)
            if hi >= trail:
                return trail
        else:
            if hi >= SL:
                return SL
            if target is not None and lo <= target:
                return target
        return None

    hit = hit_long if long else hit_short

    # --- intraday phase (entry day, bars after entry) ---
    last_close = entry
    for i in range(len(d5c)):
        ex = hit(d5l[i], d5h[i])
        if ex is not None:
            return ((ex - entry) if long else (entry - ex)) / R, 0
        last_close = d5c[i]

    if hold == "INTRA":
        return ((last_close - entry) if long else (entry - last_close)) / R, 0

    # --- swing phase (subsequent daily bars) ---
    for d, (o, h, l, c) in enumerate(daily_after[:MAXDAYS], start=1):
        ex = hit(l, h, op=o)
        if ex is not None:
            return ((ex - entry) if long else (entry - ex)) / R, d
        last_close = c
    return ((last_close - entry) if long else (entry - last_close)) / R, \
        min(len(daily_after), MAXDAYS)
