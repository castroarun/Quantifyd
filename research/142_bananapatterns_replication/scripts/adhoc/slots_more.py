
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
wk = np.zeros(len(dates), dtype=bool)
i26 = int((dates[days] >= "2026-01-01").argmax())
n_yrs = len(days)/247.0
print(f"{'slots':>5s} {'size':>6s} | {'CAGR med':>8s} {'worst':>6s} {'best':>6s} {'sprd':>5s} | "
      f"{'maxDD med':>9s} | {'2026 med':>8s} {'2026 range':>15s} {'neg':>4s}")
for slots in [8, 16, 20, 24, 32]:
    size = round(1.0/slots, 4)
    cagrs, dds, ys = [], [], []
    for seed in range(1, 31):
        eq,_,_ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wk,True,0.0025,stop=0.08,slots=slots,size_pct=size)
        eq = np.asarray(eq,dtype=float)
        cagrs.append(((eq[-1]/eq[0])**(1/n_yrs)-1)*100)
        dds.append(float((eq/np.maximum.accumulate(eq)-1).min()*100))
        ys.append(float(eq[-1]/eq[i26-1]-1)*100)
    cagrs, ys = np.array(cagrs), np.array(ys)
    print(f"{slots:5d} {size*100:5.2f}% | {np.median(cagrs):8.1f} {cagrs.min():6.1f} {cagrs.max():6.1f} "
          f"{cagrs.max()-cagrs.min():5.1f} | {np.median(dds):9.1f} | {np.median(ys):+8.1f} "
          f"{ys.min():+6.1f}..{ys.max():+6.1f} {int((ys<0).sum()):3d}/30", flush=True)
print("DONE", flush=True)
