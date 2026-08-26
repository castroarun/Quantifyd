#!/usr/bin/env python3
"""research/127 Phase B3 analyzer — do tight stops help C1 over the full history?"""
import math
from pathlib import Path
import numpy as np, pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
COST = 0.005

def prep(fn):
    d = pd.read_csv(RESULTS / fn)
    d = d[(d.atm_vol >= 100) & (d.wing_vol_min >= 10)].copy()
    d["net"] = d["gross_pct"] - COST * d["turnover_pct"]
    return d

b3 = prep("phase_b3_trades.csv")
c1 = prep("phase_b2_trades.csv")
c1 = c1[c1.config == "C1_E45X21W7K25_noSL"].assign(config="NO_SL")
df = pd.concat([b3, c1], ignore_index=True)

def line(d, label):
    if len(d) == 0: return f"{label:10s} n=0"
    p = d["net"]
    t = p.mean()/(p.std(ddof=1)/math.sqrt(len(p))) if len(p) > 1 else float("nan")
    stops = (d.exit_reason == "stop").mean() * 100
    return (f"{label:10s} n={len(d):4d}  net={p.mean()*100:+.3f}%S0  t={t:+.2f}"
            f"  win={100*(p>0).mean():.1f}%  p05={np.percentile(p,5)*100:+.2f}%"
            f"  p01={np.percentile(p,1)*100:+.2f}%  stopped={stops:.0f}%  "
            f"hold={d['hold_days'].mean():.0f}d")

print("="*112)
print("PHASE B3 — tight stops on C1, full liquid history (net @0.5% cost)")
print("="*112)
for c in ["SL125","SL150","SL175","NO_SL"]:
    print(line(df[df.config == c], c))

print("\n--- per-year net, SL125 vs NO_SL (dense era) ---")
for y in range(2021, 2027):
    a = df[(df.config == "SL125") & (df.year == y)]
    b = df[(df.config == "NO_SL") & (df.year == y)]
    print(f"  {y}: SL125 {a['net'].mean()*100:+.3f}% (n={len(a)})   "
          f"NO_SL {b['net'].mean()*100:+.3f}% (n={len(b)})")

print("\n--- where the stop fires: outcome of stopped trades vs their no-stop twin ---")
sl = df[df.config == "SL125"]
ns = df[df.config == "NO_SL"].set_index(["symbol","expiry"])
stopped = sl[sl.exit_reason == "stop"]
twins = []
for _, t in stopped.iterrows():
    try:
        tw = ns.loc[(t["symbol"], t["expiry"])]
        twins.append((t["net"], float(tw["net"])))
    except KeyError:
        pass
if twins:
    a = np.array(twins)
    better = (a[:,0] > a[:,1]).mean()
    print(f"  {len(a)} stopped trades matched to no-stop twins: stop was BETTER in "
          f"{better*100:.0f}% of them; avg stopped {a[:,0].mean()*100:+.2f}% vs "
          f"held {a[:,1].mean()*100:+.2f}%")
