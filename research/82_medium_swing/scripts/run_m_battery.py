"""research/82 M-battery: medium-swing 5-15 session holds, long + short.

Pre-registration: MEDIUM_SWING_82_STUDY_STATE.md (repo root), section 3.
Reuses research/81 engine + screen harness. REQUIRES ENGINE_MAX_TIME_STOP=15.

Run (VPS): cd /home/arun/quantifyd/research/82_medium_swing &&
  ENGINE_MAX_TIME_STOP=15 /home/arun/quantifyd/venv/bin/python3 scripts/run_m_battery.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

assert os.getenv("ENGINE_MAX_TIME_STOP") == "15", "set ENGINE_MAX_TIME_STOP=15"

R81 = Path(__file__).resolve().parents[2] / "81_swing_edge_discovery"
for p in (str(R81), str(R81 / "scripts"), str(R81.parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

from screen_common import (STUDY, atr14, clip_is, run_screen)  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
TS_GRID = (5, 10, 15)


def _stops(df, d, k=2.5):
    a = atr14(df)
    return (df["close"] - k * a) if d == 1 else (df["close"] + k * a), a


# ---- M1: Donchian at medium holds ----
M1_GRID = [(N, d, ts) for N in (20, 55) for d in (1, -1) for ts in TS_GRID]


def m1_label(cell):
    N, d, ts = cell
    return f"M1_N{N}_{'L' if d == 1 else 'S'}_ts{ts}"


def m1_entries(df, cell, allowed):
    N, d, ts = cell
    stop, a = _stops(df, d)
    if d == 1:
        m = df["close"] > df["high"].rolling(N).max().shift(1)
    else:
        m = df["close"] < df["low"].rolling(N).min().shift(1)
    m &= a.notna() & allowed
    sig = clip_is(df.index[m])
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


# ---- M2: deep-z reversion at medium holds ----
M2_GRID = [(z, d, ts) for z in (2.0, 2.5) for d in (1, -1) for ts in TS_GRID]


def m2_label(cell):
    z, d, ts = cell
    return f"M2_z{z}_{'L' if d == 1 else 'S'}_ts{ts}"


def m2_entries(df, cell, allowed):
    z_thr, d, ts = cell
    sma20 = df["close"].rolling(20).mean()
    sd20 = df["close"].rolling(20).std()
    sma200 = df["close"].rolling(200).mean()
    stop, a = _stops(df, d)
    z = (df["close"] - sma20) / sd20
    if d == 1:
        m = (z <= -z_thr) & (df["close"] > sma200)
    else:
        m = (z >= z_thr) & (df["close"] < sma200)
    m &= a.notna() & sma200.notna() & (sd20 > 0) & allowed
    sig = clip_is(df.index[m])
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": sma20.loc[sig]}, index=sig)


# ---- M3: short-specific setups ----
M3_GRID = [(kind, ts) for kind in ("breakdown", "pullfade") for ts in TS_GRID]


def m3_label(cell):
    kind, ts = cell
    return f"M3_{kind}_S_ts{ts}"


def m3_entries(df, cell, allowed):
    kind, ts = cell
    sma20 = df["close"].rolling(20).mean()
    sma200 = df["close"].rolling(200).mean()
    stop, a = _stops(df, -1)
    if kind == "breakdown":
        m = (df["close"] < df["low"].rolling(55).min().shift(1)) \
            & (df["close"] < sma200)
    else:  # bear pullback-fade: close crosses below SMA20 while SMA20<SMA200
        below = df["close"] < sma20
        m = below & ~below.shift(1).fillna(False) & (sma20 < sma200)
    m &= a.notna() & sma200.notna() & allowed
    sig = clip_is(df.index[m])
    return pd.DataFrame({"direction": -1, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


# ---- M4: prev-week range break at medium holds ----
M4_GRID = [(d, ts) for d in (1, -1) for ts in TS_GRID]


def m4_label(cell):
    d, ts = cell
    return f"M4_PW{'H_L' if d == 1 else 'L_S'}_ts{ts}"


def m4_entries(df, cell, allowed):
    d, ts = cell
    wk = df.resample("W").agg(high=("high", "max"), low=("low", "min"))
    hi = wk["high"].shift(1).reindex(df.index, method="ffill")
    lo = wk["low"].shift(1).reindex(df.index, method="ffill")
    rng = hi - lo
    if d == 1:
        m = df["close"] > hi
        stop = hi - 0.33 * rng
    else:
        m = df["close"] < lo
        stop = lo + 0.33 * rng
    m &= hi.notna() & (rng > 0) & allowed
    sig = clip_is(df.index[m])
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    for exp, grid, lab, build in (
            ("m1", M1_GRID, m1_label, m1_entries),
            ("m2", M2_GRID, m2_label, m2_entries),
            ("m3", M3_GRID, m3_label, m3_entries),
            ("m4", M4_GRID, m4_label, m4_entries)):
        print(f"\n########## {exp.upper()} ##########", flush=True)
        run_screen(exp, RESULTS, grid, lab, build)
    print("\nM-BATTERY COMPLETE", flush=True)
