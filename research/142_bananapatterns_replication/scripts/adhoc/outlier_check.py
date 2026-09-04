# -*- coding: utf-8 -*-
import sys, numpy as np
sys.path.insert(0, "research/142_bananapatterns_replication/scripts")
import bluesky_replay as br
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
wk = np.zeros(len(dates), dtype=bool)
n_yrs = len(days)/247.0
W = 0.0625
res = dict(share_top10=[], share_top1pct=[], n100=[], n200=[], best=[],
           cagr=[], cagr_w100=[], cagr_w50=[], cagr_no_top10=[])
for seed in range(1, 31):
    eq, trades, _ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wk,True,0.0025,
                                stop=0.08,slots=16,size_pct=0.0625)
    eq = np.asarray(eq,dtype=float)
    rets = np.array([t[4]/t[3]-1 for t in trades if t[5] != "open_marked"])
    lg = np.log1p(W*rets)                       # per-trade log contribution to book growth
    tot = lg.sum()
    order = np.argsort(-lg)
    res["share_top10"].append(100*lg[order[:10]].sum()/tot)
    k = max(1, int(0.01*len(lg)))
    res["share_top1pct"].append(100*lg[order[:k]].sum()/tot)
    res["n100"].append(int((rets >= 1.0).sum()))
    res["n200"].append(int((rets >= 2.0).sum()))
    res["best"].append(100*rets.max())
    res["cagr"].append((np.exp(tot/n_yrs)-1)*100)
    for cap, key in ((1.0, "cagr_w100"), (0.5, "cagr_w50")):
        lgc = np.log1p(W*np.minimum(rets, cap))
        res[key].append((np.exp(lgc.sum()/n_yrs)-1)*100)
    lg2 = np.delete(lg, order[:10])
    res["cagr_no_top10"].append((np.exp(lg2.sum()/n_yrs)-1)*100)
def m(k): return float(np.median(res[k]))
print(f"trades/seed ~{len(rets)} | trades >=+100%: median {m('n100'):.0f} | >=+200%: {m('n200'):.0f} | best single: {m('best'):.0f}%")
print(f"top-10 trades share of total 20y growth: median {m('share_top10'):.1f}%")
print(f"top-1% of trades share: median {m('share_top1pct'):.1f}%")
print(f"trade-implied CAGR: {m('cagr'):.1f}%")
print(f"  winners capped at +100%: {m('cagr_w100'):.1f}%")
print(f"  winners capped at  +50%: {m('cagr_w50'):.1f}%")
print(f"  top-10 trades REMOVED  : {m('cagr_no_top10'):.1f}%")
print("DONE", flush=True)
