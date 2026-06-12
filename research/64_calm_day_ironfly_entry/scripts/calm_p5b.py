"""research/64 P5b — does direction become predictable AFTER a compression squeeze?
Tests the squeeze->breakout follow-through: when a low-vol day is followed by a breakout, does it
sustain? If yes -> a tradeable directional trigger; if ~drift -> direction stays unpredictable. Cached."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H, L, C, O = n.high, n.low, n.close, n.open; prevC = C.shift(1)
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
tr = pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1); atr14 = rma(tr, 14)
ma20 = C.rolling(20).mean(); std20 = C.rolling(20).std()
lo14, hi14 = L.rolling(14).min(), H.rolling(14).max()
piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
f = pd.DataFrame(index=n.index)
f["atr_pct"] = atr14/C*100
f["bb_width"] = 4*std20/ma20*100
f["cpr_d"] = (tc-bc).abs()/C*100
f["stoch"] = 100*(C-lo14)/(hi14-lo14)
f["ret1"] = C.pct_change()*100         # today's move (for follow-through we use the day AFTER entry)
f = f.shift(1)   # causal entry-time features
clv = C.values; HH = 5
end_ret = np.full(len(C), np.nan); maxab = np.full(len(C), np.nan); day1 = np.full(len(C), np.nan)
for i in range(1, len(C)-HH):
    es = clv[i-1]; path = (clv[i:i+HH]-es)/es
    end_ret[i] = (clv[i+HH-1]-es)/es*100; maxab[i] = np.max(np.abs(path))*100
    day1[i] = (clv[i]-es)/es*100         # the FIRST day's realised move after entry
f["end_ret"] = end_ret; f["maxab"] = maxab; f["day1"] = day1
d = f.dropna().copy(); N = len(d)
comp = ((d.atr_pct < 1.1).astype(int) + (d.cpr_d < 0.16).astype(int) + (d.stoch > 65).astype(int)) >= 2
print(f"N={N}  compressed days={comp.mean()*100:.0f}%")

# 1. does a squeeze precede BIGGER moves? (move-rate compressed vs not)
print("\n=== 1. move-rate: compressed vs non-compressed (H=5) ===")
for thr in (1.5, 2.0, 3.0):
    mc = (d[comp].end_ret.abs() >= thr).mean(); mn = (d[~comp].end_ret.abs() >= thr).mean()
    print(f"  P(|move|>={thr}%): compressed {mc*100:4.0f}%  vs  non-compressed {mn*100:4.0f}%")

# 2. squeeze -> breakout FOLLOW-THROUGH: condition on day-1 move sign+size, does the 5d end same side?
print("\n=== 2. follow-through: after a COMPRESSED day, if day-1 moves >X%, does the 5d window end same direction? ===")
print(f"{'day-1 trigger':22s} {'n':>5} {'P(end same side)':>16} {'P(end>=1.5% same side)':>22}")
cmp = d[comp]
for thr in (0.3, 0.5, 0.75, 1.0):
    for side, lab in [(1, f"up >{thr}%"), (-1, f"down >{thr}%")]:
        if side == 1: sub = cmp[cmp.day1 > thr]
        else: sub = cmp[cmp.day1 < -thr]
        if len(sub) < 20: continue
        same = ((np.sign(sub.end_ret) == side)).mean()
        same_strong = (((np.sign(sub.end_ret) == side) & (sub.end_ret.abs() >= 1.5))).mean()
        print(f"  compressed + {lab:14s} {len(sub):5d} {same*100:15.0f}% {same_strong*100:21.0f}%")

# 3. baseline (NON-compressed) follow-through for comparison
print("\n=== 3. same follow-through but NON-compressed (is the squeeze special?) ===")
ncmp = d[~comp]
for thr in (0.5, 1.0):
    for side, lab in [(1, f"up >{thr}%"), (-1, f"down >{thr}%")]:
        sub = ncmp[ncmp.day1 > thr] if side == 1 else ncmp[ncmp.day1 < -thr]
        if len(sub) < 20: continue
        same = (np.sign(sub.end_ret) == side).mean()
        print(f"  non-comp + {lab:14s} {len(sub):5d}  P(end same side)={same*100:.0f}%")
