"""research/65 — FULL always-on book on ONE pool: neutral fly + bull jade + bear jade (VIX 13-22, 5-day
holds) + low-VIX long w/ 2% trail (VIX<13). Per-year deployment by sleeve + idle. Cached NIFTY+VIX."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H, L, C = n.high, n.low, n.close; r = C.pct_change().fillna(0).values; clv = C.values; Cp = C.shift(1).values
prevC = C.shift(1); rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
atr = rma(pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1), 14)
lo14, hi14 = L.rolling(14).min(), H.rolling(14).max(); piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
fp = pd.DataFrame(index=n.index); fp["a"] = atr/C*100; fp["c"] = (tc-bc).abs()/C*100; fp["s"] = 100*(C-lo14)/(hi14-lo14); fp = fp.shift(1)
vxp = vix.reindex(n.index, method="ffill").shift(1).values; ret = (C.pct_change()*100).values; yr = np.array(n.index.year)
neutral = ((((fp.a < 1.1).astype(int)+(fp.c < 0.16).astype(int)+(fp.s > 65).astype(int)) >= 2)).fillna(False).values
mid = lambda i: (vxp[i] >= 13) and (vxp[i] <= 22)
N = len(C); HOLD = 5
state = "flat"; hold_left = 0; peak = 0.0
sleeve = np.array(["idle"]*N, dtype=object); lvpnl = np.zeros(N)
for i in range(1, N):
    if np.isnan(vxp[i]):
        sleeve[i] = "idle"; continue
    if state == "flat":
        if vxp[i] < 13:
            state = "lowvix"; peak = Cp[i]
        elif mid(i) and (neutral[i] or ret[i] > 0.5 or ret[i] < -0.5):
            state = "fly5"; hold_left = HOLD
    if state == "lowvix":
        peak = max(peak, Cp[i])
        if (vxp[i] >= 13) or (Cp[i] < peak*0.98):    # exit on VIX>=13 or 2% trail
            state = "flat"; sleeve[i] = "lowvix" if False else "idle"
            # note: on exit day we are flat
        else:
            sleeve[i] = "lowvix"; lvpnl[i] = r[i]
    elif state == "fly5":
        sleeve[i] = "fly_jade"; hold_left -= 1
        if hold_left <= 0: state = "flat"
d = pd.DataFrame({"y": yr, "sl": sleeve, "lv": lvpnl})
g = d.groupby("y").agg(sess=("y", "size"), lowvix=("sl", lambda s: (s == "lowvix").sum()),
                       flyjade=("sl", lambda s: (s == "fly_jade").sum()), idle=("sl", lambda s: (s == "idle").sum()))
g["deployed"] = g.lowvix + g.flyjade; g["util%"] = (g.deployed/g.sess*100).round(0)
lvret = {y: (np.prod(1+gg.lv.values)-1)*100 for y, gg in d.groupby("y")}
print(f"{'year':5s} {'sess':>5} {'lowVIX-long':>11} {'fly+jade':>9} {'idle':>6} {'DEPLOYED':>9} {'util%':>6} {'lowVIX ret%':>11}")
for y in range(2015, 2027):
    if y not in g.index: continue
    row = g.loc[y]
    print(f"{y:5d} {int(row.sess):5d} {int(row.lowvix):11d} {int(row.flyjade):9d} {int(row.idle):6d} {int(row.deployed):9d} {row['util%']:5.0f}% {lvret.get(y,0):+10.1f}%")
av = g.mean()
print(f"\nAVG/yr: low-VIX long {av.lowvix:.0f}d | fly+jade {av.flyjade:.0f}d | idle {av.idle:.0f}d | DEPLOYED {av.deployed:.0f}d ({av.deployed/av.sess*100:.0f}%)")
print("Idle now = mostly VIX>22 (the chaos regime neither sleeve covers).")
