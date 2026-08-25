#!/usr/bin/env python3
"""research/127 Phase D analyzer — robustness verdicts on C1."""
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

b2 = prep("phase_b2_trades.csv"); c1 = b2[b2.config == "C1_E45X21W7K25_noSL"].copy()
pd_ = prep("phase_d_trades.csv")

def line(d, label):
    if len(d) == 0: return f"{label:34s} n=0"
    p = d["net"]
    t = p.mean()/(p.std(ddof=1)/math.sqrt(len(p))) if len(p) > 1 else float("nan")
    return (f"{label:34s} n={len(d):4d}  net={p.mean()*100:+.3f}%S0  t={t:+.2f}"
            f"  win={100*(p>0).mean():.1f}%  p05={np.percentile(p,5)*100:+.2f}%")

print("="*100)
print(f"PHASE D — robustness on C1 (net @{COST*100:.1f}% turnover cost, liquid sample)")
print("="*100)
print(line(c1, "C1 reference"))

print("\n--- 1. SUPER-WINNER GUARD (drop top-3 contributing symbols) ---")
contrib = c1.groupby("symbol")["net"].sum().sort_values(ascending=False)
top3 = list(contrib.head(3).index)
print("top-3 contributors:", top3)
print(line(c1[~c1.symbol.isin(top3)], "C1 without top-3"))
top5 = list(contrib.head(5).index)
print(line(c1[~c1.symbol.isin(top5)], "C1 without top-5"))
pos_share = (c1.groupby("symbol")["net"].mean() > 0).mean()
print(f"breadth: {pos_share*100:.0f}% of symbols have positive mean net "
      f"({c1['symbol'].nunique()} symbols)")

print("\n--- 2. OOS ERA SPLITS ---")
c1["ed_"] = pd.to_datetime(c1["entry_date"])
print(line(c1[c1.ed_ < "2024-01-01"], "2016-2023 (older era)"))
print(line(c1[c1.ed_ >= "2024-01-01"], "2024-2026 (modern era)"))
print(line(c1[(c1.ed_ >= "2021-01-01") & (c1.ed_ < "2025-01-01")], "2021-2024 (ex the strong 25/26)"))

print("\n--- 3. LIQUIDITY THRESHOLD SENSITIVITY (re-filter from full C1 rows) ---")
full = pd.read_csv(RESULTS / "phase_b2_trades.csv")
full = full[full.config == "C1_E45X21W7K25_noSL"].copy()
full["net"] = full["gross_pct"] - COST*full["turnover_pct"]
for av, wv in [(50, 1), (100, 10), (200, 10), (500, 25)]:
    print(line(full[(full.atm_vol >= av) & (full.wing_vol_min >= wv)], f"atm_vol>={av} & wing>={wv}"))

print("\n--- 4. PARAMETER NEIGHBORHOOD (want: plateau, no lone peak) ---")
for c in ["N_X18","N_X24","N_W6","N_W8","N_K2","N_K3"]:
    print(line(pd_[pd_.config == c], c))

print("\n--- 5. DTE-WINDOW PLACEBO (same structure at 35/55 entry) ---")
for c in ["P_E35","P_E55"]:
    print(line(pd_[pd_.config == c], c))

print("\n--- 6. ENTRY-LAG (enter next session after anchor) ---")
print(line(pd_[pd_.config == "LAG1"], "LAG1 (next-session entry)"))

print("\n--- 7. MULTIPLE-TESTING NOTE ---")
n_cfg = 17 + 5 + 9
print(f"configs tried across B/B2/D: ~{n_cfg}. Bonferroni-ish deflation: a t of 5.0 "
      f"at n_cfg={n_cfg} retains ~t_eff 3.5-4; C1 t must stay >3 after guards above to pass G3.")
