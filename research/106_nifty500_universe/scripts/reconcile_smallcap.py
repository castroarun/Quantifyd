"""Reconcile my traded-value smallcap PROXY against the REAL Nifty Smallcap 250 ETF (HDFCSML250).

The regime switch trades on "are smallcaps leading?". If the proxy disagrees with the real index, the
switch is trading on a mis-measured signal. HDFCSML250 exists only from 2023-02, but that is exactly the
window where the proxy (says largecaps lead in 2026) conflicts with Arun's read (says smallcaps lead).
"""
import importlib.util, sqlite3
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH, EXCLUDE = m75.BENCH, m75.EXCLUDE
close, tv = m75.load()

con = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")
real = pd.Series({pd.Timestamp(d): c for d, c in con.execute(
    "SELECT date,close FROM market_data_unified WHERE symbol='HDFCSML250' AND timeframe='day' AND close>0")}
).sort_index()
nb = close[BENCH].ffill()
print(f"HDFCSML250 (real Nifty Smallcap 250 ETF): {real.index[0].date()} .. {real.index[-1].date()}, {len(real)} days\n")

# ── REAL smallcap-vs-largecap RS ──
common = real.index.intersection(nb.index)
rs_real = (real[common] / nb[common]).dropna()
rs_real_m = rs_real.resample("ME").last()
real6 = rs_real_m / rs_real_m.shift(6) - 1

# ── MY PROXY (ranks 301-500 vs 1-100 by traded value), rebuilt monthly ──
idx = close.index[(close.index >= pd.Timestamp("2022-01-01"))]
ME = [pd.Timestamp(x) for x in sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values))]
LC, SC, prev = [], [], None
for d in ME:
    adv = tv.loc[:d].tail(126).median().dropna()
    adv = adv[[s for s in adv.index if s not in EXCLUDE and s != BENCH]]
    rank = adv.sort_values(ascending=False)
    if prev is not None:
        for bucket, store in ((rank.index[:100], LC), (rank.index[300:500], SC)):
            r = []
            for s in bucket:
                if s in close.columns:
                    a, b = close.at[prev, s], close.at[d, s]
                    if pd.notna(a) and pd.notna(b) and a > 0:
                        r.append(b / a - 1)
            store.append((d, float(np.mean(r)) if r else 0.0))
    prev = d
lci = (1 + pd.Series(dict(LC))).cumprod(); sci = (1 + pd.Series(dict(SC))).cumprod()
rs_prox = sci / lci
prox6 = rs_prox / rs_prox.shift(6) - 1

# ── compare ──
j = pd.DataFrame({"real6": real6, "prox6": prox6}).dropna()
if len(j) > 3:
    agree = ((j.real6 > 0) == (j.prox6 > 0)).mean() * 100
    corr = j.real6.corr(j.prox6)
    print(f"Overlap: {len(j)} months ({j.index[0].date()} .. {j.index[-1].date()})")
    print(f"  Direction agreement (both say 'smallcaps leading' or not): {agree:.0f}%")
    print(f"  Correlation of the 6m RS momentum series:                  {corr:+.2f}\n")
    print(f"{'month':>10} {'REAL 6m RS':>12} {'PROXY 6m RS':>13} {'real says':>12} {'proxy says':>12} {'agree':>7}")
    for d, r in j.tail(20).iterrows():
        rs_ = "SMALL" if r.real6 > 0 else "LARGE"
        ps_ = "SMALL" if r.prox6 > 0 else "LARGE"
        print(f"{d.strftime('%Y-%m'):>10} {r.real6*100:>11.1f}% {r.prox6*100:>12.1f}% "
              f"{rs_:>12} {ps_:>12} {('yes' if rs_==ps_ else 'NO'):>7}")
else:
    print("insufficient overlap")

# ── what does each say RIGHT NOW ──
print("\n=== CURRENT READ ===")
if len(real6.dropna()):
    v = real6.dropna().iloc[-1]
    print(f"  REAL (HDFCSML250 vs NIFTYBEES, 6m): {v*100:+.1f}% -> smallcaps {'LEADING' if v>0 else 'LAGGING'}")
if len(prox6.dropna()):
    v = prox6.dropna().iloc[-1]
    print(f"  PROXY (tv ranks 301-500 vs 1-100):  {v*100:+.1f}% -> smallcaps {'LEADING' if v>0 else 'LAGGING'}")
# 12m too
r12 = rs_real_m / rs_real_m.shift(12) - 1
p12 = rs_prox / rs_prox.shift(12) - 1
if len(r12.dropna()):
    print(f"  REAL 12m:  {r12.dropna().iloc[-1]*100:+.1f}%")
if len(p12.dropna()):
    print(f"  PROXY 12m: {p12.dropna().iloc[-1]*100:+.1f}%")
