"""research/64 P8 — predict a day-4/5 BREACH from price patterns at day-3 close (beyond drift).
Among flies still calm through day-3, which day-3 features flag an impending 2% breach on day-4/5?
And crucially — within the LOW-drift (seemingly-safe) group, does anything still warn? Cached NIFTY daily."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H = n.high.values; L = n.low.values; C = n.close.values
vx = vix.reindex(n.index, method="ffill").values
prevC = pd.Series(C).shift(1).values
tr = np.maximum.reduce([H - L, np.abs(H - prevC), np.abs(L - prevC)])
atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
N = len(C); rows = []
for i in range(20, N - 5):
    es = C[i-1]; m = [(C[i+dd]-es)/es*100 for dd in range(5)]
    if max(abs(m[0]), abs(m[1]), abs(m[2])) >= 2:    # keep only calm-through-day-3
        continue
    j = i + 2                                         # day-3 close
    piv = (H[j]+L[j]+C[j])/3; bcv = (H[j]+L[j])/2; cpr_d3 = abs((2*piv-bcv)-bcv)/C[j]*100
    rows.append([
        abs(m[2]),                                            # drift3 (known)
        (H[j]-L[j])/es*100,                                   # day-3 candle range (expansion)
        (max(H[i],H[i+1],H[j])-min(L[i],L[i+1],L[j]))/es*100, # hold range days1-3
        atr[j]/atr[i-1],                                      # ATR expansion ratio vs entry
        cpr_d3,                                               # day-3 CPR width
        vx[j]-vx[i-1],                                        # VIX change over the hold
        abs(m[2])-abs(m[1]),                                  # acceleration (|move| growing?)
        1 if max(abs(m[3]), abs(m[4])) >= 2 else 0,           # OUTCOME: breach on day-4/5
    ])
d = pd.DataFrame(rows, columns=["drift3", "d3range", "holdrange3", "atr_ratio", "cpr_d3", "dvix", "accel", "breach"])
print(f"calm-through-3 n={len(d)}  breach-by-day5 rate = {d.breach.mean()*100:.1f}%\n")
print("=== ALL calm-thru-3: P(breach by day5) by feature quintile (lo→hi) ===")
for f in ["drift3", "d3range", "holdrange3", "atr_ratio", "cpr_d3", "dvix", "accel"]:
    s = d.copy(); s["q"] = pd.qcut(s[f].rank(method="first"), 5, labels=False)
    g = s.groupby("q").breach.mean()*100
    print(f"  {f:11s} {[round(x) for x in g.tolist()]}   spread {round(g.iloc[-1]-g.iloc[0]):+d}pp")
low = d[d.drift3 < 0.6]
print(f"\n=== WITHIN LOW-drift (<0.6%, the 'safe' group) n={len(low)} breach {low.breach.mean()*100:.0f}% — does anything still warn? ===")
for f in ["d3range", "holdrange3", "atr_ratio", "cpr_d3", "dvix", "accel"]:
    s = low.copy(); s["q"] = pd.qcut(s[f].rank(method="first"), 4, labels=False)
    g = s.groupby("q").breach.mean()*100
    print(f"  {f:11s} {[round(x) for x in g.tolist()]}   spread {round(g.iloc[-1]-g.iloc[0]):+d}pp")
