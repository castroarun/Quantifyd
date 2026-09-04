
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
m20 = (dates[days] >= "2020-01-01") & (dates[days] <= "2020-12-31")

def align(raw): return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values

def don_state(n_lo, n_hi=None):
    """Donchian latch: turns WEAK on an n_lo-day-low close, back OK on an (n_hi or n_lo)-day-high close."""
    lo = nb <= nb.rolling(n_lo).min()
    hi = nb >= nb.rolling(n_hi or n_lo).max()
    state = np.zeros(len(nb), dtype=bool); cur = False
    lov, hiv = lo.values, hi.values
    for i in range(len(nb)):
        if lov[i]: cur = True
        elif hiv[i]: cur = False
        state[i] = cur
    import pandas as pd
    return align(pd.Series(state, index=nb.index))

CELLS = {
  "DON_mid252 (below midline)": align(nb < (nb.rolling(252).max()+nb.rolling(252).min())/2),
  "DON_low63 breakdown": align(nb <= nb.rolling(63).min()),
  "DON_low126 breakdown": align(nb <= nb.rolling(126).min()),
  "DON_latch 63lo/63hi": don_state(63),
  "DON_latch 126lo/63hi": don_state(126, 63),
  "DON_latch 63lo/126hi": don_state(63, 126),
  "DD10 (winner, ref)": align(nb < 0.9*nb.rolling(252).max()),
  "gate_OFF (ref)": None,
}
for name, wk in CELLS.items():
    if wk is None: wk = np.zeros(len(dates), dtype=bool)
    terms, dds, d08, d20 = [], [], [], []
    for seed in range(1,11):
        eq,_,_ = br.simulate(seed,"random",days,dates,C,H,O,ATH,S,RSv,TVv,trig,wk,True,0.0025,stop=0.08,slots=8)
        eq = np.asarray(eq,dtype=float)
        terms.append(eq[-1]/eq[0])
        dd = eq/np.maximum.accumulate(eq)-1
        dds.append(float(dd.min()*100)); d08.append(float(dd[m08].min()*100)); d20.append(float(dd[m20].min()*100))
    print(f"{name:28s} med x{np.median(terms):7.1f} [{min(terms):.0f}..{max(terms):.0f}] "
          f"dd {np.median(dds):.1f}% 2008 {np.median(d08):.1f}% 2020 {np.median(d20):.1f}% blocked {100*wk[days].mean():.1f}%")
