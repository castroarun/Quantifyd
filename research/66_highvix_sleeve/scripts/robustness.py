"""research/65+66 — ROBUSTNESS (walk-forward halves + per-year + net-of-cost) for the two new sleeves,
and the DEFINED-RISK (stopped) high-VIX version. Daily NIFTY+VIX, causal. Cost = 0.05%/round-trip
(NIFTY future/ETF). Low-VIX uses 2% trail + daily VIX exit (intraday G3b is strictly better)."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
C = n.close; r = C.pct_change().fillna(0).values; Cp = C.shift(1).values
vxp = vix.reindex(n.index, method="ffill").shift(1).values
vx5 = vix.reindex(n.index, method="ffill").shift(6).values     # VIX ~5 sessions ago
yr = np.array(n.index.year); N = len(C); COST = 0.0005

def sim(entry, exitc, trail):
    pos = np.zeros(N); trades = np.zeros(N); inpos = False; peak = 0.0
    for i in range(1, N):
        if inpos:
            peak = max(peak, Cp[i])
            if exitc[i] or (trail and Cp[i] < peak*(1-trail)):
                inpos = False; trades[i] = 1            # exit (cost)
        if not inpos and entry[i]:
            inpos = True; peak = Cp[i]; trades[i] = 1   # entry (cost)
        pos[i] = 1.0 if inpos else 0.0
    pnl = pos*r - trades*COST
    return pos, pnl

lo_entry = (vxp < 13); lo_exit = (vxp >= 13)
hi_entry = (vxp > 22) & (vxp < vx5); hi_exit = (vxp <= 22)
sleeves = {
    "low-VIX long (2% trail)": sim(lo_entry, lo_exit, 0.02),
    "high-VIX long (VIX-fall, NO stop)": sim(hi_entry, hi_exit, None),
    "high-VIX long (VIX-fall, 5% stop=defined-risk)": sim(hi_entry, hi_exit, 0.05),
}
def stats(pnl, mask):
    p = pnl[mask];
    if p.sum() == 0 and (p != 0).sum() == 0: return (0, 0, 0, 0)
    eq = np.cumprod(1+p); dd = (eq/np.maximum.accumulate(eq)-1).min()
    dep = int((p != 0).sum()); shp = p.mean()/p.std()*np.sqrt(252) if p.std() > 0 else 0
    return (eq[-1]-1)*100, shp, dd*100, dep
h1 = yr <= 2020; h2 = yr >= 2021
print(f"{'sleeve':46s} | {'2015-20 tot/Sh/DD':>20} | {'2021-26 tot/Sh/DD':>20}  (net of 0.05%/trade)")
for name, (pos, pnl) in sleeves.items():
    a = stats(pnl, h1); b = stats(pnl, h2)
    print(f"{name:46s} | {a[0]:5.0f}% {a[1]:4.2f} {a[2]:5.0f}% | {b[0]:5.0f}% {b[1]:4.2f} {b[2]:5.0f}%")
print("\nper-year total% (net):")
for name, (pos, pnl) in sleeves.items():
    py = {y: ((np.prod(1+pnl[yr == y])-1)*100) for y in range(2015, 2027)}
    print(f"  {name:46s} " + " ".join(f"{py[y]:+5.0f}" for y in range(2015, 2027)))
print("  years:" + " "*44 + " ".join(f"{y%100:5d}" for y in range(2015, 2027)))
print("\nWALK-FORWARD = does the edge hold in BOTH halves (no params fitted; thresholds are external).")
