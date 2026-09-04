
import sys, numpy as np, pandas as pd
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
wk = np.zeros(len(dates), dtype=bool)
ddates = dates[days]

for slots, size in [(16, 0.0625), (20, 0.05)]:
    yr_rets = {}
    for seed in range(1, 31):
        eq,_,_ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wk,True,0.0025,stop=0.08,slots=slots,size_pct=size)
        nav = pd.Series(np.asarray(eq,dtype=float), index=ddates)
        for yr, seg in nav.groupby(nav.index.year):
            prev = nav[nav.index.year < yr]
            base = prev.iloc[-1] if len(prev) else seg.iloc[0]
            yr_rets.setdefault(yr, []).append((seg.iloc[-1]/base - 1)*100)
    print(f"
== slots={slots} @ {size*100:.2f}% (no gate, 30 seeds) ==")
    print(f"{'year':>5s} {'worst':>7s} {'median':>7s} {'best':>7s} {'neg':>5s}")
    for yr in sorted(yr_rets):
        v = np.array(yr_rets[yr])
        print(f"{yr:5d} {v.min():+7.1f} {np.median(v):+7.1f} {v.max():+7.1f} {int((v<0).sum()):3d}/30")
print("DONE", flush=True)
