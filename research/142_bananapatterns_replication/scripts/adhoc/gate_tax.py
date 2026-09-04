
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

def align(raw): return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values
nb = close["NIFTYBEES"].dropna()
cells = {
  "gate_OFF": np.zeros(len(dates), dtype=bool),
  "NIFTYBEES_SMA200": align(nb < nb.rolling(200).mean()),
  "NIFTYBEES_DD10": align(nb < 0.9*nb.rolling(252).max()),
}
yrs = (dates[days] >= "2026-01-01")
for name, wk in cells.items():
    terms, dds, ytd = [], [], []
    for seed in range(1,11):
        eq,_,_ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wk,True,0.0025,stop=0.08,slots=8,stcg=0.20)
        eq = np.asarray(eq,dtype=float)
        terms.append(eq[-1]/eq[0])
        dds.append(float((eq/np.maximum.accumulate(eq)-1).min()*100))
        ytd.append(float(eq[-1]/eq[yrs.argmax()]-1)*100)
    n = len(days)/247.0
    cagrs = [t**(1/n)-1 for t in terms]
    print(f"{name:22s} AFTER-TAX: med x{np.median(terms):8.1f} [{min(terms):.0f}..{max(terms):.0f}] "
          f"CAGR {np.median(cagrs)*100:.1f}% dd {np.median(dds):.1f}% 2026ytd {np.median(ytd):+.1f}%")
