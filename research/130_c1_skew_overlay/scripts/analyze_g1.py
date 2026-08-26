#!/usr/bin/env python3
"""research/130 G1 analyzer — does the added credit spread pay, and do the
MA/RSI/stoch inputs improve the direction choice?"""
import math
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
df = pd.read_csv(R / "g1_overlay.csv")
for c in ["ps_net_pct", "cs_net_pct"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def line(x, label):
    x = x.dropna()
    if len(x) < 20: return f"{label:26s} n={len(x)}"
    t = x.mean()/(x.std(ddof=1)/math.sqrt(len(x)))
    return (f"{label:26s} n={len(x):4d}  net={x.mean()*100:+.3f}%S0  t={t:+.2f}"
            f"  win={100*(x>0).mean():.1f}%  p05={np.percentile(x,5)*100:+.2f}%"
            f"  p01={np.percentile(x,1)*100:+.2f}%")

print("="*104)
print(f"research/130 G1 — EXTRA credit-spread unit on each C1 trade ({len(df)} trades)")
print("="*104)
print("--- always-on overlays ---")
print(line(df["ps_net_pct"], "ALWAYS bull put spread"))
print(line(df["cs_net_pct"], "ALWAYS bear call spread"))

print("\n--- direction from the indicator inputs (bull-state -> PS, bear-state -> CS) ---")
for gate in ["sma200", "ema2050", "rsi50", "stochkd"]:
    gated = df["ps_net_pct"].where(df[gate] == 1, df["cs_net_pct"])
    print(line(gated, f"{gate}-directed"))
    print(line(df.loc[df[gate] == 1, "ps_net_pct"], f"  PS when {gate} bull"))
    print(line(df.loc[df[gate] == 0, "cs_net_pct"], f"  CS when {gate} bear"))

print("\n--- per-year, ALWAYS_PS ---")
for y, d in df.groupby("year"):
    x = d["ps_net_pct"].dropna()
    if len(x) > 10:
        print(f"  {y}: {x.mean()*100:+.3f}% (n={len(x)}, win {100*(x>0).mean():.0f}%)")

print("\n--- combined book: C1 + ALWAYS_PS vs C1 alone (same trades) ---")
c1 = pd.read_csv(Path(__file__).resolve().parent.parent.parent /
                 "127_stock_neutral_wings" / "results" / "phase_b2_trades.csv")
c1 = c1[(c1.config == "C1_E45X21W7K25_noSL") & (c1.atm_vol >= 100) & (c1.wing_vol_min >= 10)].copy()
c1["c1_net"] = c1["gross_pct"] - 0.005 * c1["turnover_pct"]
m = df.merge(c1[["symbol", "expiry", "c1_net"]], on=["symbol", "expiry"], how="inner")
m["combo"] = m["c1_net"] + m["ps_net_pct"]
print(line(m["c1_net"], "C1 alone"))
print(line(m["combo"].dropna(), "C1 + ALWAYS_PS"))
