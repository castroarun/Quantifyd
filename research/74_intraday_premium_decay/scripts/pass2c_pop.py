"""research/72 pass 2 (c) — is the 15:15 IV pop tradeable? Buy ATM straddle @15:10, sell @15:15,
net of theta, by DTE. Also the tight 1-min pop (15:14->15:15). Reads saved CSVs (no DB)."""
import numpy as np, pandas as pd
from pathlib import Path

OUT = Path("/home/arun/quantifyd/research/74_intraday_premium_decay/results")
M = pd.read_csv(OUT / "pass2_day_meta.csv", dtype={"day": str})
R = pd.read_csv(OUT / "atm_minute_paths.csv", dtype={"min": str})
R = R[(R.spot > 20000) & (R.spot < 28000) & (R.prem > 0)]
piv = R.pivot_table(index="day", columns="min", values="prem", aggfunc="mean")
ivp = R.pivot_table(index="day", columns="min", values="iv", aggfunc="mean")

print("(c) POP TRADE — long ATM straddle, buy @15:10 sell @15:15; Rs per lot=65, net of theta")
for lbl, dd in [("ALL", M.day), ("DTE 0-1", M[M.dte <= 1].day), ("DTE 2+", M[M.dte >= 2].day)]:
    idx = [d for d in dd if d in piv.index]
    p = piv.reindex(idx)
    if "15:10" not in p.columns or "15:15" not in p.columns:
        print("  %-9s minutes missing" % lbl)
        continue
    chg = ((p["15:15"] - p["15:10"]).dropna()) * 65
    iv2 = ivp.reindex(p.index)
    ivc = (iv2["15:15"] - iv2["15:10"]).dropna() if {"15:10", "15:15"} <= set(iv2.columns) else pd.Series([np.nan])
    line = "  %-9s buy15:10/sell15:15  mean Rs%+.0f  med Rs%+.0f  win%%=%.0f  IVd%+.2f  n=%d" % (
        lbl, chg.mean(), chg.median(), 100 * (chg > 0).mean(), np.nanmean(ivc), len(chg))
    print(line)
    if "15:14" in p.columns:
        t = ((p["15:15"] - p["15:14"]).dropna()) * 65
        if len(t):
            print("            tight 15:14->15:15 pop: mean Rs%+.0f  med Rs%+.0f  win%%=%.0f  n=%d" % (
                t.mean(), t.median(), 100 * (t > 0).mean(), len(t)))
