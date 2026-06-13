"""research/64 P6 — INTRA-HOLD conditional calm survival. Given a fly entered at day-0 (band = entry
+-2%) has stayed calm through day k (k=3,4), what is P(calm holds the next 1-2 days), and do the days-
elapsed BEHAVIOUR features (current distance to the band, closest approach so far, path range/chop,
drift) refine that? Cached NIFTY daily; calm = 2% move-stop not hit. Also the compression+VIX subset."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H, L, C = n.high, n.low, n.close; clv = C.values; prevC = C.shift(1)
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
tr = pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1); atr14 = rma(tr, 14)
lo14, hi14 = L.rolling(14).min(), H.rolling(14).max(); piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
gv = pd.DataFrame(index=n.index)
gv["vix"] = vix.reindex(n.index, method="ffill")
gv["atr"] = atr14/C*100; gv["cpr"] = (tc-bc).abs()/C*100; gv["st"] = 100*(C-lo14)/(hi14-lo14)
gv = gv.shift(1)
gate = ((gv.atr < 1.1).astype(int) + (gv.cpr < 0.16).astype(int) + (gv.st > 65).astype(int) >= 2) & (gv.vix >= 13) & (gv.vix <= 22)

rows = []
for i in range(1, len(clv)-5):
    es = clv[i-1]
    m = [(clv[i+dd]-es)/es*100 for dd in range(5)]   # signed move from entry at each of the 5 days
    rows.append([i] + m)
A = pd.DataFrame(rows, columns=["i", "d1", "d2", "d3", "d4", "d5"])
mv = A[["d1", "d2", "d3", "d4", "d5"]].values; ab = np.abs(mv)
calm = {k: ab[:, :k].max(axis=1) < 2.0 for k in (1, 2, 3, 4, 5)}
A["gate"] = gate.iloc[A.i.values].values

def P(num, den): return (num & den).sum() / den.sum() if den.sum() else float("nan")
print(f"N={len(A)} windows")
print("\n=== unconditional calm ===")
print("  calm3 %.1f%%  calm4 %.1f%%  calm5 %.1f%%" % (calm[3].mean()*100, calm[4].mean()*100, calm[5].mean()*100))
print("\n=== CONDITIONAL survival (the core answer) ===")
print("  P(calm day4 | calm thru 3)      = %.1f%%" % (P(calm[4], calm[3])*100))
print("  P(calm day5 | calm thru 3)      = %.1f%%  (survive next 2)" % (P(calm[5], calm[3])*100))
print("  P(calm day5 | calm thru 4)      = %.1f%%  (survive next 1)" % (P(calm[5], calm[4])*100))
print("  -- per-day hazard once calm: ~%.0f%% breach per extra day" % ((1-P(calm[5], calm[4]))*100))

# feature conditioning at day-3 close, among those calm thru 3
sub3 = calm[3]
f3 = pd.DataFrame({
    "calm4": calm[4][sub3], "calm5": calm[5][sub3],
    "dist3": ab[sub3, 2],                       # how far from entry at day-3 close (buffer used)
    "maxabs3": ab[sub3, :3].max(axis=1),        # closest approach to the band so far
    "range3": mv[sub3, :3].max(axis=1) - mv[sub3, :3].min(axis=1),  # path chop
    "drift3": mv[sub3, 2],                       # signed position
})
print("\n=== at DAY-3 close (among calm-thru-3, n=%d): P(stay calm thru day5) by buffer used ===" % sub3.sum())
for feat, lab in [("dist3", "distance from entry at day-3 (% of the 2% buffer used)"),
                  ("maxabs3", "closest approach to band so far"),
                  ("range3", "path range/chop (days1-3)")]:
    s = f3.copy(); s["q"] = pd.qcut(s[feat], 5, labels=False, duplicates="drop")
    g = s.groupby("q").agg(calm5=("calm5", "mean"), lo=(feat, "min"), hi=(feat, "max"), nn=(feat, "size"))
    print(f"  by {lab}:")
    for q, r in g.iterrows():
        print(f"     {feat} {r.lo:.2f}-{r.hi:.2f}%  ->  P(calm5)= {r.calm5*100:4.0f}%  (n={int(r.nn)})")

# at day-4 close among calm thru 4
sub4 = calm[4]
f4 = pd.DataFrame({"calm5": calm[5][sub4], "dist4": ab[sub4, 3]})
print("\n=== at DAY-4 close (among calm-thru-4, n=%d): P(stay calm day5) by distance at day-4 ===" % sub4.sum())
s = f4.copy(); s["q"] = pd.qcut(s["dist4"], 5, labels=False, duplicates="drop")
g = s.groupby("q").agg(calm5=("calm5", "mean"), lo=("dist4", "min"), hi=("dist4", "max"), nn=("dist4", "size"))
for q, r in g.iterrows():
    print(f"     dist4 {r.lo:.2f}-{r.hi:.2f}%  ->  P(calm5)= {r.calm5*100:4.0f}%  (n={int(r.nn)})")

# gated subset sanity
g3 = calm[3] & A.gate.values
print("\n=== gated (compression+VIX) subset: P(calm5|calm3)= %.0f%% (n=%d) vs all %.0f%% ===" % (
    P(calm[5], g3)*100, g3.sum(), P(calm[5], calm[3])*100))
