"""V2 iron-fly causal-feature screen.
For each of the 204 baseline trades, compute technical features KNOWN AT 09:20 ENTRY
(prior day / prior completed week / prior completed month only), then test which
separate losers from winners well enough to use as an entry skip-filter on top of VIX>=13.
Read-only. No orders. Run: PYTHONPATH=. venv/bin/python3 /tmp/feature_screen.py
"""
import io, contextlib, json, math
import numpy as np, pandas as pd
from collections import defaultdict
from kiteconnect import KiteConnect
import config

YEARS = 7.3; MARGIN = 958020.0

# ---------- trade set ----------
_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    exec(open("/tmp/cd_data.py").read())            # defines C = [(date,vix,pnl),...]
TR = [(d, v, p) for d, v, p in C if d not in ("2020-03-13", "2020-03-20")]   # ex-COVID, 204

# ---------- daily bars ----------
tok = json.load(open("backtest_data/access_token.json"))
k = KiteConnect(api_key=config.KITE_API_KEY); k.set_access_token(tok["access_token"])
recs = []
for a, b in [("2017-01-01", "2021-06-30"), ("2021-07-01", "2026-06-09")]:
    for c in k.historical_data(256265, a, b, "day"):
        recs.append((c["date"].strftime("%Y-%m-%d"), c["open"], c["high"], c["low"], c["close"]))
df = (pd.DataFrame(recs, columns=["date", "o", "h", "l", "c"])
      .drop_duplicates("date").set_index("date").sort_index())
df.index = pd.to_datetime(df.index)

def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean(); rd = dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + ru/rd.replace(0, np.nan))

def cpr_width(h, l, c):
    piv = (h + l + c)/3; bc = (h + l)/2; tc = 2*piv - bc
    return (tc - bc).abs()

def add_ind(frame):
    f = frame.copy()
    f["rsi"] = rsi(f["c"])
    f["sma20"] = f["c"].rolling(20).mean(); f["sma50"] = f["c"].rolling(50).mean()
    f["sma200"] = f["c"].rolling(200).mean()
    sd = f["c"].rolling(20).std()
    f["bbw"] = (4*sd)/f["sma20"]*100                       # band width % of mid
    f["pctb"] = (f["c"] - (f["sma20"]-2*sd))/(4*sd)*100     # %B 0..100
    f["cprw"] = cpr_width(f["h"], f["l"], f["c"]).shift(0)  # this row's CPR (from its own HLC)
    # ichimoku
    f["tenkan"] = (f["h"].rolling(9).max()+f["l"].rolling(9).min())/2
    f["kijun"] = (f["h"].rolling(26).max()+f["l"].rolling(26).min())/2
    f["spanA"] = ((f["tenkan"]+f["kijun"])/2).shift(26)
    f["spanB"] = ((f["h"].rolling(52).max()+f["l"].rolling(52).min())/2).shift(26)
    return f

dd = add_ind(df)
wk = add_ind(df.resample("W-FRI").agg({"o":"first","h":"max","l":"min","c":"last"}).dropna())
mo = add_ind(df.resample("ME").agg({"o":"first","h":"max","l":"min","c":"last"}).dropna())

di = dd.index; wi = wk.index; mi = mo.index

def last_before(frame_idx, cutoff):
    pos = frame_idx.searchsorted(pd.Timestamp(cutoff)) - 1   # last strictly-before
    return pos

def feats(dstr):
    d = pd.Timestamp(dstr)
    if d not in dd.index: return None
    pos = dd.index.get_loc(d)
    if pos < 1: return None
    pd1 = dd.iloc[pos-1]                                     # prior completed day
    spot = dd.iloc[pos]["o"]                                 # entry-day open (known 09:20)
    F = {}
    # daily
    F["dCPRw_pct"] = pd1["cprw"]/spot*100
    F["dRSI"] = pd1["rsi"]
    F["dBBW"] = pd1["bbw"]; F["dPctB"] = pd1["pctb"]
    F["d_vs_sma50"] = (pd1["c"]-pd1["sma50"])/pd1["sma50"]*100
    F["d_vs_sma200"] = (pd1["c"]-pd1["sma200"])/pd1["sma200"]*100
    F["d_sma50_slope"] = (dd.iloc[pos-1]["sma50"]-dd.iloc[pos-6]["sma50"])/dd.iloc[pos-6]["sma50"]*100
    cloud_hi = max(pd1["spanA"], pd1["spanB"]); cloud_lo = min(pd1["spanA"], pd1["spanB"])
    F["d_ichi"] = 1 if pd1["c"] > cloud_hi else (-1 if pd1["c"] < cloud_lo else 0)
    F["d_cloud_thick"] = abs(pd1["spanA"]-pd1["spanB"])/spot*100
    F["d_inside"] = 1 if (dd.iloc[pos-1]["h"] < dd.iloc[pos-2]["h"] and dd.iloc[pos-1]["l"] > dd.iloc[pos-2]["l"]) else 0
    # weekly (last completed week strictly before entry's week)
    wp = last_before(wi, d - pd.Timedelta(days=d.weekday()))   # before Monday of entry week
    if wp < 2: return None
    w1 = wk.iloc[wp]; w2 = wk.iloc[wp-1]
    F["wCPRw_pct"] = w1["cprw"]/spot*100                      # "this trading week" CPR (from last wk)
    F["wCPRw_prev_pct"] = w2["cprw"]/spot*100                 # prior week's CPR
    F["wCPR_widening"] = F["wCPRw_pct"] - F["wCPRw_prev_pct"]
    F["wRSI"] = w1["rsi"]; F["wBBW"] = w1["bbw"]; F["wPctB"] = w1["pctb"]
    F["w_vs_sma20"] = (w1["c"]-w1["sma20"])/w1["sma20"]*100
    wch = max(w1["spanA"], w1["spanB"]); wcl = min(w1["spanA"], w1["spanB"])
    F["w_ichi"] = 1 if w1["c"] > wch else (-1 if w1["c"] < wcl else 0)
    F["w_inside"] = 1 if (w1["h"] < w2["h"] and w1["l"] > w2["l"]) else 0
    # prior-week range break (where is entry-day open vs last completed week H/L)
    F["pwk_break"] = 1 if spot > w1["h"] else (-1 if spot < w1["l"] else 0)
    F["pwk_pos"] = (spot - w1["l"])/(w1["h"]-w1["l"]+1e-9)    # 0=at PWL,1=at PWH
    # monthly
    mpos = last_before(mi, d.replace(day=1))
    if mpos < 1: return None
    m1 = mo.iloc[mpos]
    F["mRSI"] = m1["rsi"]; F["mCPRw_pct"] = m1["cprw"]/spot*100
    mpiv = (m1["h"]+m1["l"]+m1["c"])/3
    F["m_vs_pivot"] = (spot-mpiv)/mpiv*100
    return F

# build labelled table
data = []
for d, v, p in TR:
    F = feats(d)
    if F is None: continue
    F.update(_date=d, _vix=v, _pnl=p, _yr=d[:4], _mon=int(d[5:7]))
    data.append(F)
T = pd.DataFrame(data)
print(f"Trades with full features: {len(T)} of {len(TR)}")
NUM = [c for c in T.columns if not c.startswith("_") and T[c].nunique() > 6]
CAT = ["d_ichi", "w_ichi", "pwk_break", "d_inside", "w_inside"]

def metrics(sub):
    sub = sub.sort_values("_date")
    eq = 0; peak = -1e18; mdd = 0
    for p in sub["_pnl"]:
        eq += p; peak = max(peak, eq); mdd = min(mdd, eq-peak)
    tot = sub["_pnl"].sum()
    yt = sub.groupby("_yr")["_pnl"].sum()
    cal = (tot/YEARS)/abs(mdd) if mdd else 0
    return dict(n=len(sub), tot=tot, mdd=mdd, cal=cal,
                green=int((yt > 0).sum()), yrs=len(yt), neg=sorted(yt[yt < 0].index))

BASE = "VIX>=13"
B13 = T[T["_vix"] >= 13]
mb = metrics(B13)
print(f"\n=== BASELINE {BASE}: n={mb['n']} tot={mb['tot']:,.0f} Calmar={mb['cal']:.2f} "
      f"MaxDD={mb['mdd']:,.0f} green={mb['green']}/{mb['yrs']} neg={mb['neg']} ===")

print("\n############### UNIVARIATE QUARTILE ATTRIBUTION (on VIX>=13) ###############")
print("feature                quartile mean-pnl (n) | totalP&L | neg-years/total  [Q1..Q4 low->high feature value]")
flagged = []
for col in NUM:
    s = B13[[col, "_pnl", "_yr"]].dropna()
    if len(s) < 40: continue
    try:
        s["q"] = pd.qcut(s[col], 4, labels=[1,2,3,4], duplicates="drop")
    except Exception:
        continue
    means = []; line = f"{col:<20}"
    qstats = {}
    for q in [1,2,3,4]:
        sq = s[s["q"] == q]
        if len(sq) == 0: means.append(0); continue
        yt = sq.groupby("_yr")["_pnl"].sum(); negy = int((yt < 0).sum())
        means.append(sq["_pnl"].mean())
        qstats[q] = (sq["_pnl"].mean(), sq["_pnl"].sum(), len(sq), negy, len(yt))
    # monotonic? spearman-ish sign via corr of feature vs pnl
    corr = s[col].corr(s["_pnl"])
    print(f"\n{col:<20} corr(feat,pnl)={corr:+.2f}")
    for q in [1,2,3,4]:
        if q in qstats:
            mn, tt, nn, ny, ty = qstats[q]
            print(f"     Q{q}: mean {mn:>9,.0f} (n={nn:>2}) tot {tt:>10,.0f}  neg {ny}/{ty}")
    # flag extreme negative bucket
    for q in (1, 4):
        if q in qstats and qstats[q][1] < 0 and qstats[q][3] >= 4:
            other = 4 if q == 1 else 1
            mono = (qstats.get(other, (0,))[0] > qstats[q][0])
            flagged.append((col, q, qstats[q], mono, corr))

print("\n\n############### FLAGGED SKIP-RULES (extreme bucket neg in >=4 yrs) ###############")
for col, q, st, mono, corr in flagged:
    mn, tt, nn, ny, ty = st
    print(f"  SKIP {col} {'bottom' if q==1 else 'top'}-quartile: drops n={nn} totP&L={tt:,.0f} "
          f"neg{ny}/{ty} mono={mono} corr={corr:+.2f}")

print("\n\n############### CATEGORICAL GROUP MEANS (VIX>=13) ###############")
for col in CAT:
    print(f"\n{col}:")
    for val, sg in B13.groupby(col):
        yt = sg.groupby("_yr")["_pnl"].sum()
        print(f"   {col}={val:>2}: n={len(sg):>3} mean {sg['_pnl'].mean():>9,.0f} "
              f"tot {sg['_pnl'].sum():>10,.0f} neg {int((yt<0).sum())}/{len(yt)}")

# ---------- WALK-FORWARD on flagged numeric skip-rules ----------
print("\n\n############### WALK-FORWARD (pick threshold on TRAIN half, apply to TEST) ###############")
def wf(col, side):
    """side='lo' skip below thr, 'hi' skip above thr. Pick thr maximising TRAIN Calmar."""
    res = []
    halves = [("2019-22", lambda y: y <= "2022", "2023-26", lambda y: y >= "2023"),
              ("2023-26", lambda y: y >= "2023", "2019-22", lambda y: y <= "2022")]
    for trn_name, trn, tst_name, tst in halves:
        tr = B13[B13["_yr"].map(trn)].dropna(subset=[col])
        te = B13[B13["_yr"].map(tst)].dropna(subset=[col])
        if len(tr) < 20 or len(te) < 20: return None
        best = None
        for thr in np.quantile(tr[col], np.linspace(0.1, 0.5, 9)):
            kept = tr[tr[col] >= thr] if side == "lo" else tr[tr[col] <= thr]
            if len(kept) < len(tr)*0.5: continue
            m = metrics(kept)
            if best is None or m["cal"] > best[1]: best = (thr, m["cal"])
        if best is None: continue
        thr = best[0]
        kte = te[te[col] >= thr] if side == "lo" else te[te[col] <= thr]
        m0 = metrics(te); m1 = metrics(kte)
        res.append((trn_name, tst_name, thr, m0["cal"], m1["cal"], m0["tot"], m1["tot"], m0["mdd"], m1["mdd"]))
    return res

seen = set()
for col, q, st, mono, corr in flagged:
    if col in seen: continue
    seen.add(col)
    side = "lo" if q == 1 else "hi"
    r = wf(col, side)
    if not r: continue
    print(f"\n  {col} (skip {'below' if side=='lo' else 'above'} thr):")
    for trn, tst, thr, c0, c1, t0, t1, d0, d1 in r:
        print(f"    train {trn} -> thr={thr:.3f} -> TEST {tst}: Calmar {c0:.2f}->{c1:.2f}  "
              f"P&L {t0:,.0f}->{t1:,.0f}  DD {d0:,.0f}->{d1:,.0f}")

print("\nDONE.")
