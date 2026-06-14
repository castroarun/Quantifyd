"""COMBINED coverage: with all 3 systems (neutral fly + bull jade + bear jade), how many days/year is
capital DEPLOYED vs IDLE? Two models:
  POOL  = one capital pool, take any signal when flat (priority neutral>bull>bear), hold 5. (1x capital)
  UNION = 3 independent books (1x each), union of in-trade days. (3x capital, max coverage)
Cached NIFTY+VIX daily."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H, L, C = n.high, n.low, n.close; prevC = C.shift(1)
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
tr = pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1); atr14 = rma(tr, 14)
lo14, hi14 = L.rolling(14).min(), H.rolling(14).max(); piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
vx = vix.reindex(n.index, method="ffill"); ret = C.pct_change()*100
fp = pd.DataFrame(index=n.index)
fp["atr"] = (atr14/C*100); fp["cpr"] = ((tc-bc).abs()/C*100); fp["st"] = 100*(C-lo14)/(hi14-lo14); fp = fp.shift(1)
vxp = vx.shift(1)
neutral = ((((fp.atr < 1.1).astype(int)+(fp.cpr < 0.16).astype(int)+(fp.st > 65).astype(int)) >= 2) & (vxp >= 13) & (vxp <= 22)).fillna(False).values
bull = ((vx >= 13) & (vx <= 22) & (ret > 0.5)).fillna(False).values
bear = ((vx >= 13) & (vx <= 22) & (ret < -0.5)).fillna(False).values
N = len(C); HOLD = 5; yr = np.array(n.index.year)

def book_intrade(trig):
    it = np.zeros(N, bool); i = 0
    while i < N-HOLD:
        if trig[i]: it[i:min(i+HOLD, N)] = True; i += HOLD
        else: i += 1
    return it
nit, bit, beit = book_intrade(neutral), book_intrade(bull), book_intrade(bear)
union = nit | bit | beit

# single pool: take any signal when flat, priority neutral>bull>bear
pool = np.zeros(N, bool); i = 0
while i < N-HOLD:
    if neutral[i] or bull[i] or bear[i]:
        pool[i:min(i+HOLD, N)] = True; i += HOLD
    else: i += 1

d = pd.DataFrame({"y": yr, "neutral": nit, "pool": pool, "union": union})
g = d.groupby("y").agg(sessions=("y", "size"), neutral=("neutral", "sum"), pool=("pool", "sum"), union=("union", "sum"))
g["pool_idle"] = g.sessions - g.pool
g["union_idle"] = g.sessions - g.union
print("in-trade days / year:")
print(g.assign(neutral_pct=(g.neutral/g.sessions*100).round(0), pool_pct=(g.pool/g.sessions*100).round(0),
               union_pct=(g.union/g.sessions*100).round(0))[["sessions", "neutral", "pool", "union", "pool_idle", "union_idle", "neutral_pct", "pool_pct", "union_pct"]].to_string())
print(f"\nAVG/yr: neutral-only in-trade {g.neutral.mean():.0f} ({g.neutral.mean()/g.sessions.mean()*100:.0f}%)")
print(f"        POOL (1x capital, all 3) in-trade {g.pool.mean():.0f} ({g.pool.mean()/g.sessions.mean()*100:.0f}%) -> IDLE {g.pool_idle.mean():.0f} days/yr ({g.pool_idle.mean()/g.sessions.mean()*100:.0f}%)")
print(f"        UNION (3x capital)       in-trade {g.union.mean():.0f} ({g.union.mean()/g.sessions.mean()*100:.0f}%) -> IDLE {g.union_idle.mean():.0f} days/yr ({g.union_idle.mean()/g.sessions.mean()*100:.0f}%)")

# (B) — of neutral entries, how many hit the day-3 'near-band, convert' state (alive + drift 1.4-2%)
clv = C.values; ne = []; i = 0
while i < N-HOLD:
    if neutral[i]: ne.append(i); i += HOLD
    else: i += 1
conv = []
for i in ne:
    es = clv[i-1]; alive3 = np.max(np.abs(clv[i:i+3]-es)/es) < 0.02
    drift3 = abs(clv[i+2]-es)/es*100
    conv.append((yr[i], alive3 and (1.4 <= drift3 < 2.0)))
dc = pd.DataFrame(conv, columns=["y", "nb"])
gc = dc.groupby("y").agg(entries=("nb", "size"), convert=("nb", "sum")); gc["convert%"] = (gc.convert/gc.entries*100).round(0)
print("\n=== (B) neutral entries hitting the day-3 NEAR-BAND convert state ===")
print(gc.to_string())
print(f"  AVG: {gc.convert.mean():.0f} convert-states/yr out of {gc.entries.mean():.0f} neutral entries ({gc.convert.sum()/gc.entries.sum()*100:.0f}%)")
