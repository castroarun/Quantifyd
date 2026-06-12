"""research/64 P2b — parsimony: independent axes only. Naive 11-feature composite vs a 4-axis
orthogonal composite (vol-rep + CPR + stoch + gap); out-of-sample AUC; best parsimonious gate across
horizons H=3/5/8 and per-year. Walk-forward (z/thresholds from TRAIN). Read-only."""
import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect
import datetime as dt
pd.set_option("display.width", 200)

ak = os.environ.get("KITE_API_KEY")
try:
    import config; ak = ak or getattr(config, "KITE_API_KEY", None)
except Exception: pass
tj = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
at = tj.get("access_token") if isinstance(tj, dict) else tj
ak = ak or (tj.get("api_key") if isinstance(tj, dict) else None)
kite = KiteConnect(api_key=ak); kite.set_access_token(at)


def daily(token, start=dt.date(2015, 1, 1)):
    rows, s = [], start
    while s < dt.date.today():
        e = min(s + dt.timedelta(days=380), dt.date.today())
        try: rows += kite.historical_data(token, s, e, "day")
        except Exception as ex: print("fail", ex)
        s = e + dt.timedelta(days=1)
    seen = {r["date"].date(): r for r in rows}
    df = pd.DataFrame([seen[k] for k in sorted(seen)])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date")[["open", "high", "low", "close"]].astype(float)


n = daily(256265); vix = daily(264969)["close"].rename("vix")
H, L, C, O = n.high, n.low, n.close, n.open; prevC = C.shift(1)
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
tr = pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1); atr14 = rma(tr, 14)
ma20 = C.rolling(20).mean(); ret = C.pct_change(); low14, high14 = L.rolling(14).min(), H.rolling(14).max()
piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
f = pd.DataFrame(index=n.index)
f["vix"] = vix.reindex(n.index, method="ffill")
f["atr14_pct"] = atr14/C
f["cpr_width_d"] = (tc-bc).abs()/C
f["stoch_k"] = 100*(C-low14)/(high14-low14)
f["gap_prev"] = (O-prevC).abs()/prevC
f = f.shift(1)
clv = C.values
for Hh in (3, 5, 8):
    o = np.full(len(C), np.nan)
    for i in range(1, len(C)-Hh):
        es = clv[i-1]; o[i] = 0.0 if np.max(np.abs(clv[i:i+Hh]-es)/es) >= 0.02 else 1.0
    f[f"calm_{Hh}"] = pd.Series(o, index=n.index)
d = f.dropna().copy(); N = len(d)
tr_idx, te_idx = d.index[:int(N*0.6)], d.index[int(N*0.6):]
trn, tst = d.loc[tr_idx], d.loc[te_idx]
HIGH = {"stoch_k"}


def auc(y, s):  # rank-based (Mann-Whitney)
    y = np.asarray(y); s = np.asarray(s); r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y)-n1
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)


def zscore(cols):
    mu, sd = trn[cols].mean(), trn[cols].std()
    def sc(fr):
        z = (fr[cols]-mu)/sd
        for c in cols:
            if c in HIGH: z[c] = -z[c]
        return -z.mean(axis=1)
    return sc


NAIVE = ["vix", "atr14_pct", "cpr_width_d", "stoch_k", "gap_prev"]   # (vol rep + 4 axes; vix~atr redundant)
ORTH = ["atr14_pct", "cpr_width_d", "stoch_k", "gap_prev"]           # one vol rep + 3 independent axes
print(f"N={N} base calm5 train={trn.calm_5.mean():.1%} test={tst.calm_5.mean():.1%}")
for name, cols in [("naive(vix+atr+cpr+stoch+gap)", NAIVE), ("ORTHOGONAL(atr+cpr+stoch+gap)", ORTH)]:
    sc = zscore(cols); d["s"] = sc(d); t = d.loc[te_idx].copy()
    t["q"] = pd.qcut(t["s"], 5, labels=[1, 2, 3, 4, 5])
    g = (t.groupby("q").calm_5.mean()*100).round(1)
    print(f"\n[{name}]  test AUC={auc(tst.calm_5, sc(tst)):.3f}")
    print("  calm_5 by score quintile (test):", g.to_dict())

# best parsimonious gate: low-vol AND narrow-CPR AND high-stoch (thresholds from TRAIN)
print("\n=== parsimonious gate variants (thresholds=TRAIN; evaluated on TEST) ===")
thr = {"atr14_pct": trn.atr14_pct.quantile(0.5), "cpr_width_d": trn.cpr_width_d.quantile(0.5),
       "stoch_k": trn.stoch_k.quantile(0.5), "vix": trn.vix.quantile(0.5)}
def mask(fr, cs):
    m = pd.Series(True, index=fr.index)
    for c in cs: m &= (fr[c] >= thr[c]) if c in HIGH else (fr[c] <= thr[c])
    return m
for cs in (["atr14_pct"], ["atr14_pct", "cpr_width_d"], ["atr14_pct", "cpr_width_d", "stoch_k"], ["vix", "cpr_width_d", "stoch_k"]):
    m = mask(tst, cs); base = tst.calm_5.mean()
    print(f"  {'+'.join(cs):38s} calm5={tst[m].calm_5.mean():.1%}  cov={m.mean():.0%}  (base {base:.0%}, lift {(tst[m].calm_5.mean()-base)*100:+.0f}pp, n={int(m.sum())})")

# the chosen gate across horizons + per-year (full sample, gate thresholds from train)
gate = mask(d, ["atr14_pct", "cpr_width_d", "stoch_k"])
print("\n=== gate [low-ATR ∧ narrow-CPR ∧ high-stoch] across horizons (full sample) ===")
for Hh in (3, 5, 8):
    print(f"  H={Hh}: gated calm={d[gate][f'calm_{Hh}'].mean():.1%}  vs base={d[f'calm_{Hh}'].mean():.1%}  (cov {gate.mean():.0%})")
print("\n=== per-year calm_5: gated vs base ===")
yr = pd.DataFrame({"y": d.index.year, "calm": d.calm_5.values, "g": gate.values})
for y, sub in yr.groupby("y"):
    gg = sub[sub.g == 1]
    print(f"  {y}: base {sub.calm.mean():.0%} (n={len(sub)})  gated {gg.calm.mean():.0%} (n={len(gg)})" if len(gg) else f"  {y}: base {sub.calm.mean():.0%}")
