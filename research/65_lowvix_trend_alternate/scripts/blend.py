"""research/65 G2 — blended book: REAL fly ₹ (V2 AlgoTest, VIX≥13) + REAL low-VIX long (NIFTY, VIX<13)
on one ₹8L pool. Does filling the idle (VIX<13) windows with the melt-up long smooth the book / lift
return vs fly-alone (idle in debt)? Cached NIFTY+VIX daily."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
C = n.close; r = C.pct_change()
vxp = vix.reindex(n.index, method="ffill").shift(1)
pos = (vxp < 13).astype(float)                 # long NIFTY when prior-close VIX<13
pnl = (pos * r).fillna(0); yr = n.index.year
lvret = {y: ((1+g).prod()-1)*100 for y, g in pnl.groupby(yr)}
lvdays = {y: int(g.sum()) for y, g in pos.groupby(yr)}
# REAL V2 fly per-year ₹ (VIX>=13, from the published V2 study heatmap)
fly = {2019: 10930, 2020: 166432, 2021: 57973, 2022: 201312, 2023: 62331, 2024: 293918, 2025: 104912, 2026: -17698}
CAP = 800000; DEBT = 0.065
print(f"Pool ₹{CAP:,}. Low-VIX long = ₹8L NOTIONAL NIFTY on VIX<13 days. Fly = V2 AlgoTest real ₹.\n")
print(f"{'year':5s} {'fly ₹(real)':>12} {'lowVIX days':>11} {'lowVIX ret%':>11} {'lowVIX ₹':>10} {'BLENDED ₹':>11} {'fly+debt ₹':>11}")
eq_fly = [0.0]; eq_blend = [0.0]; tf = tb = 0
for y in range(2019, 2027):
    fr = fly.get(y, 0); lr = lvret.get(y, 0.0); ld = lvdays.get(y, 0)
    lv_rs = CAP*lr/100
    # fly-alone alternative: idle cash (the ~120-160 non-fly low-VIX days) in debt
    debt_rs = CAP*DEBT*(ld/252)
    blend = fr + lv_rs; flydebt = fr + debt_rs
    tf += flydebt; tb += blend
    eq_fly.append(eq_fly[-1]+flydebt); eq_blend.append(eq_blend[-1]+blend)
    print(f"{y:5d} {fr:+12,} {ld:11d} {lr:+10.1f}% {lv_rs:+10,.0f} {blend:+11,.0f} {flydebt:+11,.0f}")
def calmar(eq):
    e = np.array(eq); peak = np.maximum.accumulate(e); dd = (e-peak)
    maxdd = -dd.min() if len(dd) else 0
    ann = (e[-1])/ (len(e)-1)
    return ann, maxdd
af, df = calmar(eq_fly); ab, db = calmar(eq_blend)
print(f"\nTOTAL  fly+debt {tf:+,.0f}  |  BLENDED {tb:+,.0f}  |  lift {tb-tf:+,.0f}")
print(f"annual ₹ avg: fly+debt {af:,.0f} (maxDD ₹{df:,.0f})  |  blended {ab:,.0f} (maxDD ₹{db:,.0f})")
print("\nKEY: the low-VIX long is biggest in the years the fly is THIN (low-VIX years) — does it fill them?")
