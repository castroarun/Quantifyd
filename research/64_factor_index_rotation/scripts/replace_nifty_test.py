"""
"Replace the Nifty ETF in the 3-asset GTAA book with 1 or 2 factor indices" (user request).
Book = [equity sleeve] + Gold + Nasdaq, equal-weight & inverse-vol, monthly, net 20bps,
2016-26. Sleeve = each single factor, Nifty (baseline), and the best 2-factor pairs.

Consistent basis: factor + Nifty sleeves use PRICE-return INDEX series (fair ranking
across factors); Gold/Nasdaq are ETF prices. Run on VPS.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / "results"
sys.path.insert(0, str(ROOT / "research" / "63_gtaa_etf_rotation" / "scripts"))
from gtaa_engine import load_monthly_closes, _metrics

fac = pd.read_csv(RES / "factor_monthly_closes.csv", index_col=0, parse_dates=True)  # Mom/Qual/Value/LowVol/Alpha/Nifty50
etf = load_monthly_closes(["GOLDBEES", "MON100"]).rename(columns={"GOLDBEES": "Gold", "MON100": "Nasdaq"})
panel = fac.join(etf, how="outer").sort_index()
start = panel[["Gold", "Nasdaq", "Momentum"]].dropna().index.min()
rets = panel.loc[start:].pct_change()
print(f"window: {panel.loc[start:].index.min().date()} -> {panel.index.max().date()} ({len(rets.dropna())} mo)\n")


def book(assets, mode="equal", cost=20.0, lb=12):
    R = rets[assets].dropna(); prev = pd.Series(0.0, index=assets); net = []; idx = R.index
    for i in range(len(idx) - 1):
        t1 = idx[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[i-lb+1:i+1].std(); iv = 1/vol.replace(0, np.nan); w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4))); prev = w
    return _metrics(pd.Series(dict(net)).sort_index())


SINGLE = ["Nifty50", "Momentum", "Quality", "Value", "LowVol", "Alpha"]
PAIRS = [["Momentum", "LowVol"], ["Momentum", "Quality"], ["Momentum", "Alpha"],
         ["Momentum", "Value"], ["Alpha", "LowVol"], ["Quality", "LowVol"]]

rows = []
print("=== REPLACE NIFTY WITH 1 FACTOR: {sleeve, Gold, Nasdaq} ===")
for s in SINGLE:
    for mode in ("equal", "invvol"):
        m = book([s, "Gold", "Nasdaq"], mode)
        tag = "  <= baseline" if s == "Nifty50" else ""
        rows.append((f"{s}+Gold+Nasdaq", mode, m))
        print(f"  {s:9s} {mode:6s} CAGR={m['cagr']*100:5.1f}% DD={m['max_drawdown']*100:6.1f}% Calmar={m['calmar']:.2f} Sharpe={m['sharpe']:.2f}{tag}")

print("\n=== REPLACE NIFTY WITH 2 FACTORS: {f1, f2, Gold, Nasdaq} ===")
for pr in PAIRS:
    for mode in ("equal", "invvol"):
        m = book(pr + ["Gold", "Nasdaq"], mode)
        rows.append((f"{'+'.join(pr)}+Gold+Nasdaq", mode, m))
        print(f"  {'+'.join(pr):16s} {mode:6s} CAGR={m['cagr']*100:5.1f}% DD={m['max_drawdown']*100:6.1f}% Calmar={m['calmar']:.2f} Sharpe={m['sharpe']:.2f}")

print("\n=== TOP 10 by Calmar ===")
for name, mode, m in sorted(rows, key=lambda x: -x[2]['calmar'])[:10]:
    print(f"  {name:30s} {mode:6s} CAGR={m['cagr']*100:5.1f}% DD={m['max_drawdown']*100:6.1f}% Calmar={m['calmar']:.2f}")

import csv
with open(RES / "replace_nifty.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["book", "mode", "cagr", "max_dd", "calmar", "sharpe"])
    for name, mode, m in rows:
        w.writerow([name, mode, round(m['cagr']*100, 2), round(m['max_drawdown']*100, 2), round(m['calmar'], 3), round(m['sharpe'], 3)])
print("\nwrote results/replace_nifty.csv")
