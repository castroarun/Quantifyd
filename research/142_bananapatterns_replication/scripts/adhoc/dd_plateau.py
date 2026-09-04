
import sys, numpy as np
sys.path.insert(0, "research/142_bananapatterns_replication/scripts")
import bluesky_replay as br
w = br.load_frames("2004-06-01", trail_sma=20)
close, high, open_, athcp, sma, tv20 = (w[k] for k in ("close","high","open","athcp","sma50","tv20"))
etf = [c for c in close.columns if br.ETF_RE.search(c)]
tv_prev = tv20.shift(1); prev_close = close.shift(1)
elig = tv_prev >= br.TV_FLOOR; elig[etf] = False
score = 2*(close/close.shift(63)-1)+(close/close.shift(126)-1)+(close/close.shift(189)-1)+(close/close.shift(252)-1)
rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
dates = close.index
C,H,O,ATH,S = close.values, high.values, open_.values, athcp.values, sma.values
RSv, TVv = rs.values, tv_prev.values
days = np.array([i for i,d in enumerate(dates) if str(d.date()) >= "2006-01-01"])
nb = close["NIFTYBEES"].dropna()
m08 = (dates[days] >= "2008-01-01") & (dates[days] <= "2009-06-30")
for x in [0.06, 0.08, 0.10, 0.12, 0.15]:
    wk = (nb < (1-x)*nb.rolling(252).max()).shift(1).reindex(dates).ffill().fillna(False).astype(bool).values
    terms, dds, d08 = [], [], []
    for seed in range(1,11):
        eq,_,_ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wk,True,0.0025,stop=0.08,slots=8)
        eq = np.asarray(eq,dtype=float)
        terms.append(eq[-1]/eq[0])
        dd = eq/np.maximum.accumulate(eq)-1
        dds.append(float(dd.min()*100)); d08.append(float(dd[m08].min()*100))
    print(f"DD{int(x*100):2d}: med x{np.median(terms):7.1f} [{min(terms):.0f}..{max(terms):.0f}] "
          f"dd {np.median(dds):.1f}% 2008 {np.median(d08):.1f}% blocked {100*wk[days].mean():.1f}%")
