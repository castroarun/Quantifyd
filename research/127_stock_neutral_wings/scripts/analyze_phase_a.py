#!/usr/bin/env python3
"""research/127 Phase A analyzer — pooled verdict, cost sweep, liquidity + gate splits."""
import math
from pathlib import Path
import numpy as np, pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
df = pd.read_csv(RESULTS / "phase_a_trades.csv")

def stats(d, label, cost=0.0):
    if len(d) == 0: return f"{label:34s}  n=0"
    pnl = d["gross_pct"] - cost * d["turnover_pct"]
    m, s, n = pnl.mean(), pnl.std(ddof=1), len(pnl)
    t = m / (s / math.sqrt(n)) if s > 0 and n > 1 else float("nan")
    return (f"{label:34s}  n={n:5d}  mean={m*100:+.3f}%S0  t={t:+.2f}  "
            f"win={100*(pnl>0).mean():.1f}%  p05={np.percentile(pnl,5)*100:+.2f}%  "
            f"p95={np.percentile(pnl,95)*100:+.2f}%")

print("="*110)
print(f"PHASE A — {len(df)} trades, {df['symbol'].nunique()} symbols, "
      f"{df['entry_date'].min()} -> {df['entry_date'].max()}")
print("="*110)
print("\n--- POOLED, cost sweep (cost = c x turnover; slippage+taxes proxy) ---")
for c in [0.0, 0.0025, 0.005, 0.01]:
    print(stats(df, f"cost {c*100:.2f}% of turnover", c))
be = (df["gross_pct"].mean()) / (df["turnover_pct"].mean()) if df["turnover_pct"].mean() > 0 else float("nan")
print(f"break-even cost rate: {be*100:.3f}% of turnover  "
      f"(avg turnover {df['turnover_pct'].mean()*100:.1f}%S0)")

print("\n--- LIQUIDITY buckets (gross) ---")
for v in [0, 50, 100, 200, 500]:
    print(stats(df[df.atm_vol >= v], f"atm_vol >= {v}"))
print(stats(df[(df.atm_vol >= 100) & (df.wing_vol_min >= 10)], "atm_vol>=100 & wing_vol_min>=10"))

print("\n--- PER YEAR (gross, all) ---")
for y, d in df.groupby("year"):
    print(stats(d, str(y)))

print("\n--- EXIT REASONS ---")
print(df.groupby("exit_reason")["gross_pct"].agg(["count", "mean", lambda x: (x>0).mean()]))

print("\n--- MECHANISM: corr(gross, maxmove/be_width) ---")
ok = df[(df.be_width_pct > 0)].copy()
ok["move_ratio"] = ok["maxmove_pct"] / ok["be_width_pct"]
print("corr =", round(ok["move_ratio"].corr(ok["gross_pct"]), 3))

LIQ = df[df.atm_vol >= 100].copy()   # gates evaluated on the liquid sample only
print(f"\n--- GATE SPLITS on liquid sample (atm_vol>=100, n={len(LIQ)}) — gross ---")
num = {"hv30_rank": [("low<0.33", lambda d: d.hv30_rank < 0.33), ("high>0.67", lambda d: d.hv30_rank > 0.67)],
       "rv_rank":   [("calm<0.33", lambda d: d.rv_rank < 0.33), ("hot>0.67", lambda d: d.rv_rank > 0.67)],
       "adx14":     [("ADX<25", lambda d: d.adx14 < 25), ("ADX>=25", lambda d: d.adx14 >= 25)],
       "chop14":    [("CHOP>61.8", lambda d: d.chop14 > 61.8), ("CHOP<38", lambda d: d.chop14 < 38)],
       "rsi14":     [("RSI 40-60", lambda d: (d.rsi14 >= 40) & (d.rsi14 <= 60)), ("RSI outside", lambda d: (d.rsi14 < 40) | (d.rsi14 > 60))],
       "bb_bw_rank":[("squeeze<0.3", lambda d: d.bb_bw_rank < 0.3), ("wide>0.7", lambda d: d.bb_bw_rank > 0.7)],
       "atr_ratio": [("contracting<1", lambda d: d.atr_ratio < 1.0), ("expanding>=1", lambda d: d.atr_ratio >= 1.0)],
       "trend_dist_atr": [("ranging<=2ATR", lambda d: d.trend_dist_atr <= 2.0), ("trending>2ATR", lambda d: d.trend_dist_atr > 2.0)],
       "cpr_width_pct": [("narrow<0.5%", lambda d: d.cpr_width_pct < 0.5), ("wide>=0.5%", lambda d: d.cpr_width_pct >= 0.5)],
       "nr7":       [("NR7", lambda d: d.nr7 == 1)],
       "inside_day":[("inside day", lambda d: d.inside_day == 1)]}
for col, conds in num.items():
    d0 = LIQ.dropna(subset=[col]) if col in LIQ else LIQ
    for label, fn in conds:
        print(stats(d0[fn(d0)], f"{col}: {label}"))

print("\n--- TOP/BOTTOM symbols on liquid sample (n>=15) ---")
g = LIQ.groupby("symbol")["gross_pct"].agg(["count", "mean"])
g = g[g["count"] >= 15].sort_values("mean", ascending=False)
print(pd.concat([g.head(10), g.tail(5)]).to_string())
