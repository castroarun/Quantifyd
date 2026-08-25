#!/usr/bin/env python3
"""research/127 — IV-rank / IV-percentile / VRP gate splits on Phase A trades."""
import math
from pathlib import Path
import numpy as np, pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
tr = pd.read_csv(RESULTS / "phase_a_trades.csv")
iv = pd.read_csv(RESULTS / "iv_daily.csv")
iv["date"] = pd.to_datetime(iv["date"])
iv = iv.sort_values(["symbol", "date"])
# causal IV rank vs own trailing 252 obs (exclude today)
iv["iv_rank"] = (iv.groupby("symbol")["iv"]
                 .transform(lambda s: s.rolling(252).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)))
tr["entry_date"] = pd.to_datetime(tr["entry_date"])
m = tr.merge(iv[["symbol","date","iv","iv_rank"]],
             left_on=["symbol","entry_date"], right_on=["symbol","date"], how="left")
m["vrp"] = m["iv"] / m["rv20"]

LIQ = m[(m.atm_vol >= 100) & (m.wing_vol_min >= 10)].copy()

def line(d, label):
    if len(d) == 0: return f"{label:30s} n=0"
    p = d["gross_pct"]
    t = p.mean()/(p.std(ddof=1)/math.sqrt(len(p))) if len(p) > 1 else float("nan")
    return (f"{label:30s} n={len(d):4d}  gross={p.mean()*100:+.3f}%S0  t={t:+.2f}"
            f"  win={100*(p>0).mean():.1f}%  p05={np.percentile(p,5)*100:+.2f}%")

print(f"Phase A trades matched to IV: {m['iv'].notna().mean()*100:.0f}%  "
      f"(iv_rank available: {m['iv_rank'].notna().mean()*100:.0f}%)")
print(f"LIQUID sample n={len(LIQ)}\n")
print("--- IV level (entry ATM IV) ---")
for lo, hi in [(0,0.25),(0.25,0.35),(0.35,0.50),(0.50,2)]:
    print(line(LIQ[(LIQ.iv >= lo) & (LIQ.iv < hi)], f"IV {lo:.2f}-{hi:.2f}"))
print("\n--- IV RANK (own trailing 252d) ---")
for lo, hi in [(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.01)]:
    print(line(LIQ[(LIQ.iv_rank >= lo) & (LIQ.iv_rank < hi)], f"IVR {lo:.2f}-{hi:.2f}"))
print("\n--- VRP = IV / RV20 (premium richness) ---")
for lo, hi in [(0,0.9),(0.9,1.1),(1.1,1.4),(1.4,99)]:
    print(line(LIQ[(LIQ.vrp >= lo) & (LIQ.vrp < hi)], f"VRP {lo:.1f}-{hi:.1f}"))
print("\n--- combos ---")
print(line(LIQ[(LIQ.iv_rank > 0.5)], "IVR>0.5 (rich vs own history)"))
print(line(LIQ[(LIQ.iv_rank > 0.5) & (LIQ.vrp > 1.1)], "IVR>0.5 & VRP>1.1"))
print(line(LIQ[(LIQ.iv_rank > 0.5) & (LIQ.rv_rank < 0.5)], "IVR>0.5 & RV-rank<0.5 (rich IV, calm stock)"))
print(line(LIQ[(LIQ.iv_rank < 0.25)], "IVR<0.25 (cheap — expect worst)"))

print("\n--- per-year of best combo (if it beats no-gate) ---")
best = LIQ[(LIQ.iv_rank > 0.5) & (LIQ.rv_rank < 0.5)]
for y, d in best.groupby("year"): print(line(d, str(y)))
