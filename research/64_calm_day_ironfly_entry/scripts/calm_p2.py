"""research/64 P2 — deepest combination/composite work on the calm-day signal.
Reuses P1 pipeline (NIFTY+VIX daily from Kite, causal features, calm_H outcome), then:
  A correlation/redundancy  B threshold->coverage curves  C conditional lift
  D best 2-/3-way AND gates (walk-forward)  E composite compression score (walk-forward)
  F multivariate logistic (train/test AUC + coefs)  G EV proxy
No look-ahead; thresholds chosen on TRAIN only for walk-forward. Read-only."""
import sys, json, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect
import datetime as dt
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

ak = os.environ.get("KITE_API_KEY")
try:
    import config; ak = ak or getattr(config, "KITE_API_KEY", None)
except Exception:
    pass
tj = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
at = tj.get("access_token") if isinstance(tj, dict) else tj
ak = ak or (tj.get("api_key") if isinstance(tj, dict) else None)
kite = KiteConnect(api_key=ak); kite.set_access_token(at)


def daily(token, start=dt.date(2015, 1, 1)):
    rows, s = [], start
    while s < dt.date.today():
        e = min(s + dt.timedelta(days=380), dt.date.today())
        try: rows += kite.historical_data(token, s, e, "day")
        except Exception as ex: print("chunk fail", s, ex)
        s = e + dt.timedelta(days=1)
    seen = {r["date"].date(): r for r in rows}
    df = pd.DataFrame([seen[k] for k in sorted(seen)])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date")[["open", "high", "low", "close"]].astype(float)


print("loading ..."); n = daily(256265); vix = daily(264969)["close"].rename("vix")
H, L, C, O = n.high, n.low, n.close, n.open; prevC = C.shift(1)
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
tr = pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1); atr14 = rma(tr, 14)
up = H.diff(); dn = -L.diff()
plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=n.index)
minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=n.index)
pdi = 100*rma(plus, 14)/atr14; mdi = 100*rma(minus, 14)/atr14
ma20, ma50 = C.rolling(20).mean(), C.rolling(50).mean(); std20 = C.rolling(20).std(); ret = C.pct_change()
low14, high14 = L.rolling(14).min(), H.rolling(14).max()
piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc

f = pd.DataFrame(index=n.index)
f["vix"] = vix.reindex(n.index, method="ffill")
f["atr14_pct"] = atr14/C
f["rvol_10"] = ret.rolling(10).std()
f["rvol_20"] = ret.rolling(20).std()
f["donch20_width"] = (H.rolling(20).max()-L.rolling(20).min())/C
f["range_5d"] = (H.rolling(5).max()-L.rolling(5).min())/C
f["bb_width"] = (4*std20)/ma20
f["cpr_width_d"] = (tc-bc).abs()/C
f["gap_prev"] = (O-prevC).abs()/prevC
f["stoch_k"] = 100*(C-low14)/(high14-low14)
f["rng_contraction"] = (H-L)/(H-L).rolling(20).mean()
f = f.shift(1)   # causal

# outcome
clv = C.values; out = np.full(len(C), np.nan)
for i in range(1, len(C)-5):
    es = clv[i-1]; out[i] = 0.0 if np.max(np.abs(clv[i:i+5]-es)/es) >= 0.02 else 1.0
f["calm"] = pd.Series(out, index=n.index)
d = f.dropna().copy()
N = len(d); BASE = d.calm.mean()
SURV = ["vix", "atr14_pct", "rvol_10", "rvol_20", "donch20_width", "range_5d", "bb_width", "cpr_width_d", "gap_prev", "stoch_k", "rng_contraction"]
# calm-direction: stoch_k HIGH=calm; everything else LOW=calm
HIGH_CALM = {"stoch_k"}
print(f"\nN={N}  base calm_5={BASE:.1%}  period {d.index[0].date()}..{d.index[-1].date()}")

# A. correlation/redundancy (spearman)
print("\n=== A. Spearman correlation among survivors (redundancy) ===")
print(d[SURV].corr("spearman").round(2).to_string())

# B. threshold -> calm/coverage curves (each gated in its calm direction at percentile p)
print("\n=== B. single-feature gate: calm-rate @ coverage (keep the calm-direction tail) ===")
print(f"{'feature':16s} " + " ".join(f"p{p:02d}" for p in (20, 30, 40, 50, 60)) + "   (calm% @ that coverage)")
for c in SURV:
    line = f"{c:16s} "
    for p in (20, 30, 40, 50, 60):
        thr = d[c].quantile(p/100 if c not in HIGH_CALM else 1-p/100)
        sub = d[d[c] <= thr] if c not in HIGH_CALM else d[d[c] >= thr]
        line += f"{sub.calm.mean()*100:4.0f} "
    print(line + f"  (cov≈{20}-{60}%)")

# C. conditional lift: within the calmest-tertile of ATR, does each other feature add?
base_sub = d[d.atr14_pct <= d.atr14_pct.quantile(0.5)]   # low-ATR half
print(f"\n=== C. conditional lift (within low-ATR half, calm={base_sub.calm.mean():.1%}, n={len(base_sub)}) ===")
for c in SURV:
    if c == "atr14_pct": continue
    thr = base_sub[c].quantile(0.5 if c not in HIGH_CALM else 0.5)
    sub = base_sub[base_sub[c] <= thr] if c not in HIGH_CALM else base_sub[base_sub[c] >= thr]
    print(f"  +{c:16s} -> calm {sub.calm.mean():.1%} (n={len(sub)}, lift {(sub.calm.mean()-base_sub.calm.mean())*100:+.1f}pp)")

# walk-forward split
tr_idx = d.index[:int(N*0.6)]; te_idx = d.index[int(N*0.6):]
trn, tst = d.loc[tr_idx], d.loc[te_idx]
def gate_mask(frame, combo, thr_src):
    m = pd.Series(True, index=frame.index)
    for c in combo:
        t = thr_src[c]
        m &= (frame[c] >= t) if c in HIGH_CALM else (frame[c] <= t)
    return m

# D. best 2-/3-way AND gates, thresholds = TRAIN tertile, evaluated on TEST (walk-forward)
print(f"\n=== D. AND-gates: thresholds from TRAIN tertile, walk-forward on TEST (train calm={trn.calm.mean():.1%}, test={tst.calm.mean():.1%}) ===")
thr_src = {c: trn[c].quantile(0.66 if c in HIGH_CALM else 0.34) for c in SURV}  # ~calmest third
rows = []
for k in (1, 2, 3):
    for combo in itertools.combinations(SURV, k):
        mtr = gate_mask(trn, combo, thr_src); mte = gate_mask(tst, combo, thr_src)
        if mtr.sum() < 80 or mte.sum() < 60: continue
        rows.append(dict(k=k, combo="+".join(combo),
                         train_calm=round(trn[mtr].calm.mean(), 3), train_cov=round(mtr.mean(), 2),
                         test_calm=round(tst[mte].calm.mean(), 3), test_cov=round(mte.mean(), 2)))
R = pd.DataFrame(rows).sort_values("test_calm", ascending=False)
print("TOP 15 by TEST calm-rate (coverage shown so we don't cheat with tiny gates):")
print(R.head(15).to_string(index=False))
R.to_csv("/tmp/calm_p2_gates.csv", index=False)

# E. composite compression score (z on TRAIN, applied to TEST)
mu, sd = trn[SURV].mean(), trn[SURV].std()
def comp_score(frame):
    z = (frame[SURV]-mu)/sd
    for c in HIGH_CALM: z[c] = -z[c]   # flip so + = more compressed/calm for all
    return -z.mean(axis=1)             # higher = more compressed = expected calmer
d["score"] = comp_score(d)
print("\n=== E. composite COMPRESSION score: calm-rate by quintile (z on TRAIN, applied to TEST) ===")
tst2 = d.loc[te_idx].copy(); tst2["q"] = pd.qcut(d.loc[tr_idx, "score"].rank(), 5, labels=False).reindex(tst2.index)
tst2["q"] = pd.qcut(tst2["score"], 5, labels=[1, 2, 3, 4, 5])
g = tst2.groupby("q").calm.agg(["mean", "count"])
print(g.assign(mean=lambda x: (x["mean"]*100).round(1)).to_string())

# F. multivariate logistic (train/test AUC + standardized coefs)
print("\n=== F. multivariate logistic (L2), train/test ===")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    Xtr = ((trn[SURV]-mu)/sd).values; Xte = ((tst[SURV]-mu)/sd).values
    lr = LogisticRegression(C=0.5, max_iter=1000).fit(Xtr, trn.calm.values)
    auc_tr = roc_auc_score(trn.calm, lr.predict_proba(Xtr)[:, 1])
    auc_te = roc_auc_score(tst.calm, lr.predict_proba(Xte)[:, 1])
    print(f"  AUC train={auc_tr:.3f}  test={auc_te:.3f}  (0.5=no skill)")
    co = pd.Series(lr.coef_[0], index=SURV).sort_values(key=abs, ascending=False)
    print("  standardized coefficients (sign: + => more calm):")
    print(co.round(3).to_string())
except Exception as e:
    print("  sklearn unavailable / failed:", e)

# G. EV proxy: EV = calm*W - (1-calm)*Lstop, for a few credit/stop assumptions (per 10-lot trade)
print("\n=== G. EV proxy per trade (10 lots): EV = calm*W - (1-calm)*Lstop ===")
best_gate_calm = R.head(1)["test_calm"].values[0] if len(R) else BASE
for W, Ls in [(40000, 34000), (60000, 34000), (80000, 34000)]:
    ev_base = BASE*W - (1-BASE)*Ls
    ev_gate = best_gate_calm*W - (1-best_gate_calm)*Ls
    print(f"  W=+{W:,} Lstop=-{Ls:,}:  base(calm {BASE:.0%}) EV={ev_base:,.0f}   best-gate(calm {best_gate_calm:.0%}) EV={ev_gate:,.0f}")
print("\n(Reminder: W = premium captured is ASSUMED; true ₹ optimum needs option premiums / AlgoTest.)")
print("saved /tmp/calm_p2_gates.csv")
