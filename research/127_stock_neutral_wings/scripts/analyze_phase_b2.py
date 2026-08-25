#!/usr/bin/env python3
"""research/127 Phase B2 analyzer — composite configs, with and without the VRP gate."""
import math
from pathlib import Path
import numpy as np, pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
tr = pd.read_csv(RESULTS / "phase_b2_trades.csv")
iv = pd.read_csv(RESULTS / "iv_daily.csv")
iv["date"] = pd.to_datetime(iv["date"]); iv = iv.sort_values(["symbol","date"])
iv["iv_rank"] = (iv.groupby("symbol")["iv"]
                 .transform(lambda s: s.rolling(252).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)))
# causal rv20 from iv file is absent; recompute VRP using credit-side proxy is wrong —
# join rv20 from phase A? Not per-config. Use daily spot: rv from iv_daily is not there,
# so approximate VRP with iv / trailing-20d realized vol computed from spot in DB.
import sqlite3, sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "89_short_monthly_straddle" / "scripts"))
import engine as E
conn = sqlite3.connect(E.db_path())
rv = []
for s in tr["symbol"].unique():
    d = E.load_daily(s, conn)
    if d.empty: continue
    lr = np.log(d["close"]/d["close"].shift())
    lr = lr.where(lr.abs() <= 0.25)
    r = lr.rolling(20).std()*math.sqrt(252)
    rv.append(pd.DataFrame({"symbol": s, "date": d.index, "rv20": r.values}))
rv = pd.concat(rv)
tr["entry_date"] = pd.to_datetime(tr["entry_date"])
m = (tr.merge(iv[["symbol","date","iv","iv_rank"]], left_on=["symbol","entry_date"],
              right_on=["symbol","date"], how="left")
       .merge(rv, left_on=["symbol","entry_date"], right_on=["symbol","date"], how="left"))
m["vrp"] = m["iv"] / m["rv20"]
LIQ = m[(m.atm_vol >= 100) & (m.wing_vol_min >= 10)].copy()

def line(d, label, cost=0.005):
    if len(d) == 0: return f"{label:26s} n=0"
    net = d["gross_pct"] - cost*d["turnover_pct"]
    t = net.mean()/(net.std(ddof=1)/math.sqrt(len(net))) if len(net) > 1 else float("nan")
    return (f"{label:26s} n={len(d):4d}  gross={d['gross_pct'].mean()*100:+.3f}%  "
            f"net={net.mean()*100:+.3f}%S0  t={t:+.2f}  win={100*(net>0).mean():.1f}%  "
            f"p05={np.percentile(net,5)*100:+.2f}%")

print(f"PHASE B2 — {len(tr)} rows; liquid {len(LIQ)}; IV matched {LIQ['iv'].notna().mean()*100:.0f}%")
print("\n--- composites, liquid, NO gate (net @0.5% turnover cost) ---")
for c in sorted(LIQ.config.unique()):
    print(line(LIQ[LIQ.config == c], c))
print("\n--- composites + VRP>1.1 gate ---")
G = LIQ[LIQ.vrp > 1.1]
for c in sorted(G.config.unique()):
    print(line(G[G.config == c], c))
print("\n--- best composite per-year, no gate vs VRP>1.1 ---")
tb = {}
for c in G.config.unique():
    d = G[G.config == c]
    if len(d) > 60:
        net = d["gross_pct"] - 0.005*d["turnover_pct"]
        tb[c] = net.mean()/(net.std(ddof=1)/math.sqrt(len(net)))
best = max(tb, key=tb.get)
print(f"best gated composite: {best} (t={tb[best]:+.2f})")
for tag, dd in [("no-gate", LIQ[LIQ.config == best]), ("VRP>1.1", G[G.config == best])]:
    print(f"  [{best} | {tag}]")
    for y, d in dd.groupby("year"):
        print("   ", line(d, str(y)))
print("\n--- gated best composite: cost sensitivity ---")
for c_ in [0.0025, 0.005, 0.0075, 0.01]:
    print(line(G[G.config == best], f"cost {c_*100:.2f}%", cost=c_))
print("\n--- gated best composite: per-symbol (n>=8) ---")
d = G[G.config == best].copy()
d["net"] = d["gross_pct"] - 0.005*d["turnover_pct"]
g = d.groupby("symbol")["net"].agg(["count","mean"])
print(g[g["count"] >= 8].sort_values("mean", ascending=False).to_string())
