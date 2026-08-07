"""Existing book, no leverage: CURRENT (equal-weight trim every rebalance) vs LET WINNERS RUN
(rotate names on rank only, never trim held weights). Honest per-leg cost (0.15%/leg) so the churn
cost the live book pays is captured. Also tracks realized short-term gains (STCG exposure proxy).
rsblend + buffer22 + Donchian15 + weekly gate, 2006-2026 (+ modern 2014+)."""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
P62 = Path("/home/arun/quantifyd/research/62_momentum_etf_subselect/scripts/62_mom30_subselect.py")
P75 = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
_a = importlib.util.spec_from_file_location("r62", str(P62)); r62 = importlib.util.module_from_spec(_a); _a.loader.exec_module(r62)
_b = importlib.util.spec_from_file_location("mom75", str(P75)); mom75 = importlib.util.module_from_spec(_b); _b.loader.exec_module(mom75)
BENCH = r62.BENCH
START = pd.Timestamp("2006-01-01"); RT_LEG = 0.0015   # 0.15% per leg = 0.30% round-trip
def run_book(close, tv, trim, N=8, buffer=22, gate=100, donchian=15, etf_size=30):
    idx = close.index[close.index >= START]
    me = set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
    iso = idx.isocalendar(); wk = set(pd.Series(idx, index=idx).groupby([iso.year.values, iso.week.values]).last().values)
    nbx = close[BENCH].ffill(); daycash = (1.065)**(1/252)
    roll_low = close.rolling(donchian, min_periods=donchian).min().shift(1)
    cash, held, nav, prev, derisked = 1.0, {}, [], None, False
    st = dict(cost=0.0, stgain=0.0, donch=0, gate=0)   # held[s]=[val,costbasis,buydate]
    def E(): return sum(v[0] for v in held.values())
    def sell(s, d, reason):
        nonlocal cash
        v = held[s][0]; g = v - held[s][1]
        if g > 0 and (d - held[s][2]).days < 365: st['stgain'] += g   # STCG exposure
        cash += v; st['cost'] += v*RT_LEG; del held[s]
    def do_fill(d, px):
        nonlocal cash, held, derisked
        etf = r62.eligible_etf(close, tv, d, "rsblend", etf_size)
        if not etf: return
        top = etf[:N]; buf = set(etf[:buffer])
        for s in list(held):
            if s not in buf: sell(s, d, "rotate")
        keep = [s for s in held if s in buf]; add = [s for s in top if s not in keep]
        target = [s for s in (keep + add)[:N] if pd.notna(px.get(s, np.nan))]
        if not target: return
        if trim:                                        # CURRENT: reset every name to equal weight
            tot = E() + cash; w = tot/len(target)
            for s in target:
                if s in held:
                    d0 = w - held[s][0]
                    if d0 < 0 and (held[s][0]-w) > held[s][1]*0 and (d-held[s][2]).days < 365 and held[s][0] > held[s][1]:
                        st['stgain'] += min(-d0, held[s][0]-held[s][1])   # trimming a winner realizes STCG
                    st['cost'] += abs(d0)*RT_LEG; held[s][0] = w
                else:
                    st['cost'] += w*RT_LEG; held[s] = [w, w, d]
            cash = tot - w*len(target)
        else:                                           # LET WINNERS RUN: keep held weights, fund new from cash
            per = (E() + cash)/N
            for s in target:
                if s not in held:
                    buy = min(per, max(0.0, cash))
                    if buy <= 0: break
                    st['cost'] += buy*RT_LEG; held[s] = [buy, buy, d]; cash -= buy
        derisked = False
    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0: stt[0] += stt[0]*(p1/p0-1.0)
        if cash > 0: cash *= daycash
        if held:                                        # Donchian
            lows = roll_low.loc[d]
            for s in list(held):
                p1 = px.get(s, np.nan); ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln: sell(s, d, "donch"); st['donch'] += 1
        if gate and d in wk:
            b = nbx.loc[:d].dropna(); roff = len(b) >= gate and b.iloc[-1] < b.tail(gate).mean()
            if roff:
                for s in list(held): sell(s, d, "gate")
                derisked = True; st['gate'] += 1
            else: derisked = False
        if d in me and not derisked: do_fill(d, px)
        nav.append((d, E()+cash)); prev = d
    n = pd.DataFrame(nav, columns=["date","nav"]).set_index("date")["nav"]
    def blk(ns):
        yrs = (ns.index[-1]-ns.index[0]).days/365.25
        cagr = (ns.iloc[-1]/ns.iloc[0])**(1/yrs)-1
        dr = ns.pct_change().dropna(); sh = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
        dd = ((ns-ns.cummax())/ns.cummax()).min()
        return cagr*100, dd*100, sh, cagr/abs(dd) if dd<0 else 0, ns.iloc[-1]/ns.iloc[0]
    full = blk(n); mod = blk(n[n.index >= "2014-01-01"])
    return full, mod, st
close, tv = mom75.load()
print("EXISTING BOOK, no leverage, honest per-leg cost. 2006-2026 | modern 2014+\n")
print(f"{'variant':>22} {'CAGR':>6} {'MaxDD':>7} {'Sharpe':>6} {'Calmar':>6} {'mult':>8} | {'mod CAGR':>8} {'mod DD':>7} | {'cost%':>6} {'STCGexp':>8}")
for trim, lbl in [(True, "CURRENT (trim to EW)"), (False, "LET WINNERS RUN")]:
    (c, dd, sh, cal, mult), (mc, mdd, msh, mcal, mm), st = run_book(close, tv, trim)
    print(f"{lbl:>22} {c:>5.1f}% {dd:>6.1f}% {sh:>6.2f} {cal:>6.2f} {mult:>7.0f}x | {mc:>7.1f}% {mdd:>6.1f}% | {st['cost']*100:>5.1f} {st['stgain']*100:>7.0f}", flush=True)
