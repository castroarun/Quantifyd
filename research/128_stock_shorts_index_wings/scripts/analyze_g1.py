#!/usr/bin/env python3
"""research/128 G1 analyzer — index wings vs stock wings vs naked, same entries."""
import math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
R127 = HERE.parent.parent / "127_stock_neutral_wings" / "results"

g1 = pd.read_csv(R / "g1_trades.csv")
g1 = g1[g1.atm_vol >= 100].copy()

c1 = pd.read_csv(R127 / "phase_b2_trades.csv")
c1 = c1[(c1.config == "C1_E45X21W7K25_noSL") & (c1.atm_vol >= 100) & (c1.wing_vol_min >= 10)].copy()
c1["net_pct"] = c1["gross_pct"] - 0.005 * c1["turnover_pct"]
c1["config"] = "C1_STOCKW"
c1["wing_debit_pct"] = c1["wing_debit_pct"]
df = pd.concat([g1, c1[["config","symbol","expiry","year","net_pct","gross_pct",
                        "wing_debit_pct","exit_reason"]]], ignore_index=True)

def line(d, label):
    if len(d) == 0: return f"{label:12s} n=0"
    p = d["net_pct"]
    t = p.mean()/(p.std(ddof=1)/math.sqrt(len(p))) if len(p) > 1 else float("nan")
    return (f"{label:12s} n={len(d):4d}  net={p.mean()*100:+.3f}%S0  t={t:+.2f}"
            f"  win={100*(p>0).mean():.1f}%  p05={np.percentile(p,5)*100:+.2f}%"
            f"  p01={np.percentile(p,1)*100:+.2f}%  wing_cost={d['wing_debit_pct'].mean()*100:.2f}%")

print("="*112)
print("research/128 G1 — same entries: NAKED vs NIFTY/BNF wings (3/5/7%) vs C1 stock wings")
print("="*112)
for c in ["NAKED","NW7","NW5","NW3","C1_STOCKW"]:
    print(line(df[df.config == c], c))

print("\n--- matched-trade comparison (same symbol+expiry present in all configs) ---")
key = ["symbol","expiry"]
sets = [set(map(tuple, df[df.config == c][key].values)) for c in ["NAKED","NW5","C1_STOCKW"]]
common = set.intersection(*sets)
print(f"common trades: {len(common)}")
for c in ["NAKED","NW7","NW5","NW3","C1_STOCKW"]:
    d = df[df.config == c]
    d = d[d[key].apply(tuple, axis=1).isin(common)]
    print(line(d, c))

print("\n--- worst 1% months test: per-trade net in the 5 worst SYSTEMIC months vs idiosyncratic hits ---")
d5 = df[df.config == "NW5"].copy(); dc = df[df.config == "C1_STOCKW"].copy()
for tag, dd in [("NW5", d5), ("C1_STOCKW", dc)]:
    w = dd.nsmallest(10, "net_pct")[["symbol","entry_date" if "entry_date" in dd else "expiry","net_pct"]] \
        if "entry_date" in dd else dd.nsmallest(10, "net_pct")[["symbol","expiry","net_pct"]]
    print(tag, "worst-10:", [(r[0], f"{r[-1]*100:+.1f}%") for r in w.values])

print("\n--- per-year (dense era), NW5 vs C1 ---")
for y in range(2021, 2027):
    a = df[(df.config == "NW5") & (df.year == y)]["net_pct"]
    b = df[(df.config == "C1_STOCKW") & (df.year == y)]["net_pct"]
    print(f"  {y}: NW5 {a.mean()*100:+.3f}% (n={len(a)})   C1 {b.mean()*100:+.3f}% (n={len(b)})")
