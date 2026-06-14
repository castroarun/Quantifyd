"""Per-year days + WIN-RATIO for all 3 systems (one-trade-at-a-time, 5-day hold). Win-ratio is
price-action (credible). ₹ P&L/DD need real premiums (AlgoTest) — NOT modelled here. Cached daily."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
H, L, C = n.high, n.low, n.close; clv = C.values; prevC = C.shift(1)
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
tr = pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1); atr14 = rma(tr, 14)
lo14, hi14 = L.rolling(14).min(), H.rolling(14).max(); piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
vx = vix.reindex(n.index, method="ffill")
ret = C.pct_change()*100
fp = pd.DataFrame(index=n.index)
fp["atr"] = (atr14/C*100); fp["cpr"] = ((tc-bc).abs()/C*100); fp["st"] = 100*(C-lo14)/(hi14-lo14); fp = fp.shift(1)
vxp = vx.shift(1)
neutral = ((((fp.atr < 1.1).astype(int)+(fp.cpr < 0.16).astype(int)+(fp.st > 65).astype(int)) >= 2) & (vxp >= 13) & (vxp <= 22)).fillna(False).values
bull = ((vx >= 13) & (vx <= 22) & (ret > 0.5)).fillna(False).values
bear = ((vx >= 13) & (vx <= 22) & (ret < -0.5)).fillna(False).values
vv = vx.values; HOLD = 5; N = len(clv); yr = n.index.year
straddle = lambda v: 0.8*(v/100)*np.sqrt(HOLD/252)*100
def jbull(r, v):
    cr = 0.42*straddle(v)+0.30-0.18*straddle(v); return cr+min(0.0, r+2.0)-min(max(r-1.0, 0.0), 1.5)+max(-4.0-r, 0.0)
def jbear(r, v):
    cr = 0.42*straddle(v)+0.30-0.18*straddle(v); return cr-max(r-2.0, 0.0)-min(max(-1.0-r, 0.0), 1.5)+max(r-4.0, 0.0)

def run(trig, kind):
    recs = []; i = 1
    while i < N-HOLD:
        if trig[i]:
            if kind == "neutral":
                es = clv[i-1]; win = np.max(np.abs(clv[i:i+HOLD]-es)/es) < 0.02
            else:
                es = clv[i]; r = (clv[i+HOLD-1]-es)/es*100
                pay = jbull(r, vv[i]) if kind == "bull" else jbear(r, vv[i])
                win = pay > 0
            recs.append((yr[i], win)); i += HOLD
        else: i += 1
    d = pd.DataFrame(recs, columns=["y", "win"])
    return d.groupby("y").agg(entries=("win", "size"), wins=("win", "sum"))

for name, t, k in [("Neutral fly", neutral, "neutral"), ("Bull jade", bull, "bull"), ("Bear jade", bear, "bear")]:
    g = run(t, k); g["win%"] = (g.wins/g.entries*100).round(0)
    print(f"\n=== {name} — entries & win-ratio by year ===")
    print(g[["entries", "win%"]].to_string())
    print(f"  AVG: entries {g.entries.mean():.0f}/yr | win-ratio {g.wins.sum()/g.entries.sum()*100:.0f}%")
