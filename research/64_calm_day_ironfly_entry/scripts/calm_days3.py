"""Days/year & selectivity for ALL three systems (one-trade-at-a-time, 5-day hold).
Neutral fly: compression gate + VIX 13-22 (prior-close, causal).
Bull jade : VIX 13-22 AND today closes UP >0.5% (day-1 confirmation).
Bear jade : VIX 13-22 AND today closes DOWN >0.5%. Cached NIFTY+VIX daily."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H, L, C = n.high, n.low, n.close; prevC = C.shift(1)
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
tr = pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1); atr14 = rma(tr, 14)
lo14, hi14 = L.rolling(14).min(), H.rolling(14).max(); piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
vx = vix.reindex(n.index, method="ffill")
ret = C.pct_change()*100
fp = pd.DataFrame(index=n.index)
fp["atr"] = (atr14/C*100); fp["cpr"] = ((tc-bc).abs()/C*100); fp["st"] = 100*(C-lo14)/(hi14-lo14); fp = fp.shift(1)
vix_prior = vx.shift(1)
neutral = (((fp.atr < 1.1).astype(int)+(fp.cpr < 0.16).astype(int)+(fp.st > 65).astype(int)) >= 2) & (vix_prior >= 13) & (vix_prior <= 22)
bull = (vx >= 13) & (vx <= 22) & (ret > 0.5)
bear = (vx >= 13) & (vx <= 22) & (ret < -0.5)
HOLD = 5; idx = list(n.index)

def sim(trig):
    trig = trig.fillna(False).values
    entry = np.zeros(len(idx), bool); intr = np.zeros(len(idx), bool); i = 0
    while i < len(idx)-HOLD:
        if trig[i]:
            entry[i] = True; intr[i:min(i+HOLD, len(idx))] = True; i += HOLD
        else: i += 1
    return trig, entry, intr

out = {}
for name, t in [("Neutral fly", neutral), ("Bull jade", bull), ("Bear jade", bear)]:
    tg, en, it = sim(t)
    d = pd.DataFrame({"y": n.index.year, "tg": tg, "en": en, "it": it})
    g = d.groupby("y").agg(trig=("tg", "sum"), entries=("en", "sum"), intrade=("it", "sum"))
    out[name] = g
    print(f"\n=== {name} (one-trade-at-a-time, 5-day hold) ===")
    print(g.to_string())
    print(f"  AVG/yr: trigger-days {g.trig.mean():.0f} | entries {g.entries.mean():.0f} | in-trade {g.intrade.mean():.0f} (~{g.intrade.mean()/250*100:.0f}% of yr)")
print("\n=== SUMMARY (avg/yr) ===")
for name, g in out.items():
    print(f"  {name:12s} trigger {g.trig.mean():4.0f} | entries {g.entries.mean():3.0f} | in-trade {g.intrade.mean():4.0f} days (~{g.intrade.mean()/250*100:.0f}%)")
