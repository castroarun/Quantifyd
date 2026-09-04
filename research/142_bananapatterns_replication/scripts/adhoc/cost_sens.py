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
for bps in (0.0025, 0.0040, 0.0060):
    cagrs, dds = [], []
    for seed in range(1, 31):
        eq,_,_ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wk,True,bps,
                             stop=0.08,slots=16,size_pct=0.0625,stcg=0.20,cash_yield=0.05)
        eq = np.asarray(eq,dtype=float)
        cagrs.append(((eq[-1]/eq[0])**(1/n_yrs)-1)*100)
        dds.append(float((eq/np.maximum.accumulate(eq)-1).min()*100))
    print(f"cost {bps*10000:.0f}bps/side: after-tax CAGR med {np.median(cagrs):.1f}% "
          f"[worst {min(cagrs):.1f}] dd {np.median(dds):.1f}% Calmar {np.median(cagrs)/abs(np.median(dds)):.2f}", flush=True)
print("DONE", flush=True)
