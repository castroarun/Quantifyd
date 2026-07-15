"""EXP-B2: deeper-z reversion follow-up to B1 (pre-registered BEFORE run).

Justified by B1's monotone dose-response (net rises z2.0->z2.5). Target FIXED
sma20 and stop FIXED 2.5xATR (B1-dominant choices — conditioning noted in the
ledger; the multiple-testing discount for family B covers B1+B2 jointly).
GRID LOCKED: z {2.5, 3.0} x dir {L,S} x ts {2,4} = 8 cells.
Also prints per-year detail for the family-best cells (regime concentration).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from screen_common import (STUDY, atr14, clip_is, run_screen, per_year_detail)

EXP = "b2"
RESULTS = STUDY / "experiments" / "B2_deep_z_daily" / "results"
GRID = [(z, d, ts) for z in (2.5, 3.0) for d in (1, -1) for ts in (2, 4)]
K_ATR = 2.5


def cell_label(cell):
    z, d, ts = cell
    return f"z{z}_{'L' if d == 1 else 'S'}_sma20_ts{ts}"


def build_entries(df, cell, allowed):
    z_thr, d, ts = cell
    sma20 = df["close"].rolling(20).mean()
    sd20 = df["close"].rolling(20).std()
    sma200 = df["close"].rolling(200).mean()
    a = atr14(df)
    z = (df["close"] - sma20) / sd20
    if d == 1:
        m = (z <= -z_thr) & (df["close"] > sma200)
        stop = df["close"] - K_ATR * a
    else:
        m = (z >= z_thr) & (df["close"] < sma200)
        stop = df["close"] + K_ATR * a
    m &= a.notna() & sma200.notna() & (sd20 > 0) & allowed
    sig = clip_is(df.index[m])
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": sma20.loc[sig]}, index=sig)


if __name__ == "__main__":
    agg = run_screen(EXP, RESULTS, GRID, cell_label, build_entries)
    if len(agg):
        for c in agg.head(2)["cell"]:
            per_year_detail(EXP, RESULTS, c)
    # regime check on B1's best too
    b1r = STUDY / "experiments" / "B1_zscore_daily" / "results"
    import shutil
    src = b1r / "b1_trades.csv"
    if src.exists():
        tmp = RESULTS / "b1_trades.csv"
        if not tmp.exists():
            shutil.copy(src, tmp)
        per_year_detail("b1", RESULTS, "z2.5_S_sma20_ts2")
