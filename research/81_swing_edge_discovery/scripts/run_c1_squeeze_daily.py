"""EXP-C1: volatility squeeze -> expansion breakout, daily F&O, 2-4d hold.

G0: post-compression expansion attracts breakout flow; vol regimes cluster.
Squeeze = BB(20,2) relative width in its lowest decile/quintile of trailing
252d. Signal: yesterday squeezed AND today closes beyond the band.
GRID LOCKED: squeeze_pct {0.10, 0.20} x dir {L,S} x ts {2,4} = 16 cells.
Stop FIXED 2.0xATR14. No target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from screen_common import STUDY, atr14, clip_is, run_screen

EXP = "c1"
RESULTS = STUDY / "experiments" / "C1_squeeze_daily" / "results"
GRID = [(q, d, ts) for q in (0.10, 0.20) for d in (1, -1) for ts in (2, 4)]
K_ATR = 2.0


def cell_label(cell):
    q, d, ts = cell
    return f"sq{int(q*100)}_{'L' if d == 1 else 'S'}_ts{ts}"


def build_entries(df, cell, allowed):
    q, d, ts = cell
    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    width = (upper - lower) / mid
    wrank = width.rolling(252).rank(pct=True)
    squeezed_y = wrank.shift(1) <= q
    a = atr14(df)
    if d == 1:
        m = squeezed_y & (df["close"] > upper)
        stop = df["close"] - K_ATR * a
    else:
        m = squeezed_y & (df["close"] < lower)
        stop = df["close"] + K_ATR * a
    m &= a.notna() & wrank.notna() & allowed
    sig = clip_is(df.index[m])
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


if __name__ == "__main__":
    run_screen(EXP, RESULTS, GRID, cell_label, build_entries)
