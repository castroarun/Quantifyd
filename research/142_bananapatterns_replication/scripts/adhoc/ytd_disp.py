
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
wk_dd10 = (nb < 0.9*nb.rolling(252).max()).shift(1).reindex(dates).ffill().fillna(False).astype(bool).values
wk_off = np.zeros(len(dates), dtype=bool)
i26 = int((dates[days] >= "2026-01-01").argmax())
for label, wk, slots, size in [("off_s8", wk_off, 8, 0.1875), ("off_s16", wk_off, 16, 0.0625),
                                ("dd10_s8", wk_dd10, 8, 0.1875), ("dd10_s16", wk_dd10, 16, 0.0625)]:
    ys = []
    for seed in range(1, 31):
        eq,_,_ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wk,True,0.0025,stop=0.08,slots=slots,size_pct=size)
        eq = np.asarray(eq,dtype=float)
        ys.append(float(eq[-1]/eq[i26-1]-1)*100)
    ys = np.array(ys)
    print(f"{label:9s} 2026ytd: median {np.median(ys):+.1f}%  [{ys.min():+.1f}..{ys.max():+.1f}]  "
          f"IQR {np.percentile(ys,75)-np.percentile(ys,25):.1f}pp  negative seeds {int((ys<0).sum())}/30", flush=True)
print("DONE", flush=True)
