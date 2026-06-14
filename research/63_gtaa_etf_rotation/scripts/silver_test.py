"""
Add SILVERBEES to the GTAA ETF book (user request). Silver lists ~2022-02 so the 4-asset
window is short (~4.3y) — compare 4-asset vs 3-asset over the SAME window + correlations.
Run on VPS: venv/bin/python research/63_gtaa_etf_rotation/scripts/silver_test.py
"""
import numpy as np, pandas as pd
from gtaa_engine import load_monthly_closes, _metrics

p = load_monthly_closes(["NIFTYBEES", "GOLDBEES", "MON100", "SILVERBEES"])
p = p.rename(columns={"GOLDBEES": "Gold", "MON100": "Nasdaq", "SILVERBEES": "Silver", "NIFTYBEES": "Nifty"})
p4 = p.loc[p.dropna().index.min():]          # common window incl. silver (2022+)
r4 = p4.pct_change().dropna()
yrs = len(r4) / 12
print(f"Common window (with Silver): {p4.index.min().date()} -> {p4.index.max().date()} ({len(r4)} months, {yrs:.1f}y)\n")


def book(rets, assets, mode="equal", cost=20.0, lb=12):
    R = rets[assets]
    prev = pd.Series(0.0, index=assets); net = []
    idx = R.index
    for i in range(len(idx) - 1):
        t1 = idx[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[i-lb+1:i+1].std(); iv = 1/vol.replace(0, np.nan)
            w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4)))
        prev = w
    s = pd.Series(dict(net)).sort_index(); m = _metrics(s)
    return m

print("Per-asset (this window):")
for c in p4.columns:
    mm = _metrics(r4[c]); print(f"  {c:7s} CAGR={mm['cagr']*100:6.1f}%  vol={r4[c].std()*np.sqrt(12)*100:5.1f}%  DD={mm['max_drawdown']*100:6.1f}%")

print("\nCorrelation (monthly returns):")
print(r4.corr().round(2).to_string())

print("\nEqual-weight, monthly rebalance, net 20bps (SAME window):")
for label, assets in [("3-asset (Nifty/Gold/Nasdaq)", ["Nifty", "Gold", "Nasdaq"]),
                       ("4-asset (+Silver)", ["Nifty", "Gold", "Nasdaq", "Silver"]),
                       ("Gold+Silver+Nasdaq (no equity)", ["Gold", "Silver", "Nasdaq"])]:
    m = book(r4, assets, "equal")
    print(f"  {label:34s} CAGR={m['cagr']*100:5.1f}% DD={m['max_drawdown']*100:6.1f}% Calmar={m['calmar']:.2f} Sharpe={m['sharpe']:.2f}")

print("\nInverse-vol, monthly rebalance, net 20bps (SAME window):")
for label, assets in [("3-asset (Nifty/Gold/Nasdaq)", ["Nifty", "Gold", "Nasdaq"]),
                       ("4-asset (+Silver)", ["Nifty", "Gold", "Nasdaq", "Silver"])]:
    m = book(r4, assets, "invvol")
    print(f"  {label:34s} CAGR={m['cagr']*100:5.1f}% DD={m['max_drawdown']*100:6.1f}% Calmar={m['calmar']:.2f} Sharpe={m['sharpe']:.2f}")
