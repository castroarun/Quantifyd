"""EXP-D1: end-of-day strength/weakness carrying into the next 2-4 sessions.

G0: institutional order flow spreads across days; a close pinned to the
day's extreme with a >=1% move suggests unfinished flow (continuation).
CLV = (close - low) / (high - low) in [0, 1].
Long: CLV >= thr AND day return >= +1%. Short: CLV <= 1-thr AND ret <= -1%.
GRID LOCKED: thr {0.8, 0.9} x dir {L,S} x ts {2,4} = 16 cells.
Stop FIXED 2.5xATR14. No target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from screen_common import STUDY, atr14, clip_is, run_screen

EXP = "d1"
RESULTS = STUDY / "experiments" / "D1_eod_strength_daily" / "results"
GRID = [(thr, d, ts) for thr in (0.8, 0.9) for d in (1, -1) for ts in (2, 4)]
K_ATR = 2.5
MIN_MOVE = 0.01


def cell_label(cell):
    thr, d, ts = cell
    return f"clv{thr}_{'L' if d == 1 else 'S'}_ts{ts}"


def build_entries(df, cell, allowed):
    thr, d, ts = cell
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = (df["close"] - df["low"]) / rng
    ret1 = df["close"] / df["close"].shift(1) - 1
    a = atr14(df)
    if d == 1:
        m = (clv >= thr) & (ret1 >= MIN_MOVE)
        stop = df["close"] - K_ATR * a
    else:
        m = (clv <= 1 - thr) & (ret1 <= -MIN_MOVE)
        stop = df["close"] + K_ATR * a
    m &= a.notna() & clv.notna() & allowed
    sig = clip_is(df.index[m])
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


if __name__ == "__main__":
    run_screen(EXP, RESULTS, GRID, cell_label, build_entries)
