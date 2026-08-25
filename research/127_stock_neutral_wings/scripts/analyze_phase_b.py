#!/usr/bin/env python3
"""research/127 Phase B analyzer — per-config comparison on the liquidity-clean sample."""
import math
from pathlib import Path
import numpy as np, pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
df = pd.read_csv(RESULTS / "phase_b_trades.csv")
LIQ = df[(df.atm_vol >= 100) & (df.wing_vol_min >= 10)].copy()
COST = 0.005  # 0.5% of premium turnover (slippage+txn proxy); sweep below

def line(d, label, cost=COST):
    if len(d) == 0: return f"{label:10s} n=0"
    net = d["gross_pct"] - cost*d["turnover_pct"]
    g = d["gross_pct"]
    tn = net.mean()/(net.std(ddof=1)/math.sqrt(len(net))) if len(net) > 1 else float("nan")
    return (f"{label:10s} n={len(d):5d}  gross={g.mean()*100:+.3f}%  net={net.mean()*100:+.3f}%S0"
            f"  t(net)={tn:+.2f}  win={100*(net>0).mean():.1f}%  p05={np.percentile(net,5)*100:+.2f}%"
            f"  credit={d['credit_pct'].mean()*100:.2f}%  hold={d['hold_days'].mean():.0f}d")

ORDER = ["E30","E40","BASE","E50","E60","X10","X15","X28","W3","W5?","W7","W10",
         "SL150","SL300","SLnone","TPnone","K2.5","K5"]
print("="*118)
print(f"PHASE B — {len(df)} rows total; LIQUID sample (atm_vol>=100 & wing_vol_min>=10): {len(LIQ)} rows")
print(f"cost model: net = gross − {COST*100:.2f}% × premium-turnover")
print("="*118)
print("\n--- DTE_entry axis (exit 21, W5, SL200/TP50, ATM) ---")
for c in ["E30","E40","BASE","E50","E60"]: print(line(LIQ[LIQ.config==c], c))
print("\n--- DTE_exit axis ---")
for c in ["X10","X15","BASE","X28"]: print(line(LIQ[LIQ.config==c], c))
print("\n--- Wing width axis ---")
for c in ["W3","BASE","W7","W10"]: print(line(LIQ[LIQ.config==c], c))
print("\n--- Stop/target axis ---")
for c in ["SL150","BASE","SL300","SLnone","TPnone"]: print(line(LIQ[LIQ.config==c], c))
print("\n--- Short-strike offset axis ---")
for c in ["BASE","K2.5","K5"]: print(line(LIQ[LIQ.config==c], c))

print("\n--- cost sensitivity on BASE (liquid) ---")
for c in [0.0025, 0.005, 0.01]:
    print(line(LIQ[LIQ.config=="BASE"], f"c={c*100:.2f}%", cost=c))

print("\n--- per-year (net) for BASE and best-by-t config ---")
b = LIQ[LIQ.config=="BASE"]
t_by = {}
for c in LIQ.config.unique():
    d = LIQ[LIQ.config==c]; net = d["gross_pct"]-COST*d["turnover_pct"]
    if len(d) > 50: t_by[c] = net.mean()/(net.std(ddof=1)/math.sqrt(len(net)))
best = max(t_by, key=t_by.get)
print(f"best t(net): {best} ({t_by[best]:+.2f})")
for cfg in dict.fromkeys(["BASE", best]):
    print(f"  [{cfg}]")
    for y, d in LIQ[LIQ.config==cfg].groupby("year"): print("   ", line(d, str(y)))
