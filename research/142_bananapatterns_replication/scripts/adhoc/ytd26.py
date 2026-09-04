
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
def align(raw): return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values
GATES = {
  "DD10": align(nb < 0.9*nb.rolling(252).max()),
  "SMA200": align(nb < nb.rolling(200).mean()),
  "No gate": np.zeros(len(dates), dtype=bool),
}
i26 = int((dates[days] >= "2026-01-01").argmax())
for name, wk in GATES.items():
    ytds = []
    for seed in range(1,11):
        eq,_,_ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wk,True,0.0025,stop=0.08,slots=8)
        eq = np.asarray(eq,dtype=float)
        ytds.append(float(eq[-1]/eq[i26-1]-1)*100)
    ys = sorted(round(y,1) for y in ytds)
    print(f"{name:8s} 2026 YTD per seed: {ys}  -> median {np.median(ytds):+.1f}%")
