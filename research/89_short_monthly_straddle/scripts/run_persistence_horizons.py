#!/usr/bin/env python3
"""
research/89 — Calm PERSISTENCE by holding horizon. Directly answers: 'we don't need
the stock calm all month, just a better probability of calmness over a SHORTER hold'.

For calm entries (rv_rank<=0.30), measure P(the calm persists) over H = 3/5/10/15/20
trading days, and how much of the eventual 20-day move has already accrued by day H
(the 'premium erosion vs gamma risk' trade-off). Pure daily data.
"""
import os, sys, math, sqlite3, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

HORIZONS = [3, 5, 10, 15, 20]
START = "2010-01-01"
RESULTS = Path(__file__).resolve().parent.parent / "results"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def fwd_rv(close, i, H):
    seg = close.iloc[i:i+H+1]
    if len(seg) < 3: return np.nan
    return np.log(seg/seg.shift(1)).dropna().std()*math.sqrt(252)

def main():
    conn = sqlite3.connect(E.db_path())
    universe = E.TIER_A + E.DEFAULT_STOCKS
    rows = []
    for sym in universe:
        df = E.load_daily(sym, conn)
        if df.empty or len(df) < 400: continue
        df = df[df.index >= START]
        feat = E.add_features(df); close = feat["close"]
        n = len(feat)
        for i in range(n - max(HORIZONS) - 1):
            rank = feat["rv_rank"].iloc[i]; rv0 = feat["rv20"].iloc[i]
            if not (np.isfinite(rank) and np.isfinite(rv0)): continue
            if rank > 0.30: continue                     # calm entries only
            if abs(feat["gap"].iloc[i]) > 0.20: continue
            rec = dict(sym=sym, tier="A" if sym in E.TIER_A else "C", rv0=rv0)
            move20 = abs(close.iloc[i+20]/close.iloc[i]-1) if i+20 < n else np.nan
            for H in HORIZONS:
                fr = fwd_rv(close, i, H)
                rec[f"stay_{H}"] = (fr <= rv0*1.25) if np.isfinite(fr) else np.nan
                mH = abs(close.iloc[i+H]/close.iloc[i]-1)
                rec[f"movefrac_{H}"] = (mH/move20) if (np.isfinite(move20) and move20>0) else np.nan
            rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS/"persistence_horizons.csv", index=False)
    log(f"calm entries: {len(d)}")
    for tier_label, sub in [("ALL", d), ("A_index", d[d.tier=="A"]), ("C_stocks", d[d.tier=="C"])]:
        if sub.empty: continue
        log(f"\n===== {tier_label}  (calm entries n={len(sub)}) =====")
        log("Horizon |  P(calm persists) | mean % of 20d-move already done by then")
        for H in HORIZONS:
            p = sub[f"stay_{H}"].mean()*100
            mf = sub[f"movefrac_{H}"].mean()*100
            log(f"  {H:2d} td  |      {p:5.1f}%        |   {mf:5.1f}%")
    log(f"\nDONE -> results/persistence_horizons.csv")

if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING); main()
