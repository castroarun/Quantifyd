# -*- coding: utf-8 -*-
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
STUDY = Path("/home/arun/quantifyd/research/147_third_sleeve_archetypes")
sys.path.insert(0, "/home/arun/quantifyd/research/142_bananapatterns_replication/scripts")
import bluesky_replay as br

def yahoo_monthly(path):
    rows = json.load(open(path))
    s = pd.Series({pd.Timestamp(t, unit="s"): v for t, v in rows}).sort_index()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s[~s.index.duplicated(keep="last")]

xau = yahoo_monthly(STUDY / "results" / "yahoo_gc.json")
inr = yahoo_monthly(STUDY / "results" / "yahoo_inr.json")
ref = (xau * inr.reindex(xau.index).ffill()).dropna()

w = br.load_frames("2004-06-01", trail_sma=15)
close, high, open_, athcp, sma15, tv20 = (w[k] for k in ("close","high","open","athcp","sma50","tv20"))
etf = [c for c in close.columns if br.ETF_RE.search(c)]
tv_prev = tv20.shift(1); prev_close = close.shift(1)
elig = tv_prev >= br.TV_FLOOR; elig[etf] = False
score = 2*(close/close.shift(63)-1)+(close/close.shift(126)-1)+(close/close.shift(189)-1)+(close/close.shift(252)-1)
rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
dates = close.index
C,H,O,ATH,S = close.values, high.values, open_.values, athcp.values, sma15.values
RSv, TVv = rs.values, tv_prev.values
days = np.array([i for i,d in enumerate(dates) if str(d.date()) >= "2006-01-01"])
wkz = np.zeros(len(dates), dtype=bool)
gb = close["GOLDBEES"].dropna()
gb_m = gb.resample("ME").last().pct_change()
ref_m = ref.pct_change()
gold_m = pd.concat([ref_m[ref_m.index < "2015-02-01"], gb_m[gb_m.index >= "2015-02-01"]])
gold_m = gold_m[~gold_m.index.duplicated()].sort_index()

ys_all = []
for seed in range(1, 11):
    eq,_,_ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wkz,True,0.0025,
                         stop=0.08,slots=16,size_pct=0.0625,stcg=0.20,cash_yield=0.05)
    oa = pd.Series(np.asarray(eq,dtype=float), index=dates[days])
    mo = oa.resample("ME").last().pct_change().fillna(0)
    mo = mo[mo.index >= "2007-01-01"]
    mg = gold_m.reindex(mo.index).fillna(0)
    bl = (1 + 0.75*mo + 0.25*mg).cumprod()
    out = {}
    for yr, seg in bl.groupby(bl.index.year):
        prev = bl[bl.index.year < yr]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        run_ = pd.concat([pd.Series([base]), seg])
        out[yr] = ((seg.iloc[-1]/base-1)*100, float((run_/run_.cummax()-1).min()*100))
    ys_all.append(out)
for yr in range(2007, 2027):
    rr = [y[yr][0] for y in ys_all if yr in y]
    dd = [y[yr][1] for y in ys_all if yr in y]
    print(f"{yr};{np.median(rr):+.1f};{np.median(dd):.1f}")
print("DONE")
