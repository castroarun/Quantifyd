"""OA + GOLD two-sleeve weight sweep (no TN): find the best proportion.
OA adopted spec (trail-15, 16 slots, no gate), after-tax, cash_yield 5%,
10 seeds; GOLDBEES from 2015 (window 2015->now, stated). Monthly rebalance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R142 = Path('/home/arun/quantifyd/research/142_bananapatterns_replication/scripts')
sys.path.insert(0, str(R142))
import bluesky_replay as br

w = br.load_frames('2004-06-01', trail_sma=15)
close, high, open_, athcp, sma15, tv20 = (w[k] for k in
    ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
etf = [c for c in close.columns if br.ETF_RE.search(c)]
tv_prev = tv20.shift(1)
prev_close = close.shift(1)
elig = tv_prev >= br.TV_FLOOR
elig[etf] = False
score = 2*(close/close.shift(63)-1) + (close/close.shift(126)-1) \
    + (close/close.shift(189)-1) + (close/close.shift(252)-1)
rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
dates = close.index
C, H, O, ATH, S = close.values, high.values, open_.values, athcp.values, sma15.values
RSv, TVv = rs.values, tv_prev.values
days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= '2006-01-01'])
wk = np.zeros(len(dates), dtype=bool)
oa_navs = []
for seed in range(1, 11):
    eq, _, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S, RSv, TVv,
                           trig, wk, True, 0.0025, stop=0.08, slots=16,
                           size_pct=0.0625, stcg=0.20, cash_yield=0.05)
    oa_navs.append(pd.Series(np.asarray(eq, dtype=float), index=dates[days]))
print('OA seeds done', flush=True)
gold = close['GOLDBEES'].dropna()
gold = gold[gold.index >= '2015-01-01']


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1]/nav.iloc[0])**(1/yrs) - 1)*100
    dd = float((nav/nav.cummax()-1).min()*100)
    return cagr, dd


def wdd(nav, a, b):
    seg = nav[(nav.index >= a) & (nav.index <= b)]
    if not len(seg):
        return np.nan
    run_ = seg/seg.cummax()
    return float((run_-1).min()*100)


print(f"{'OA/GOLD':>9} | {'CAGR med [rng]':>22} | {'DD med':>7} | {'Calmar':>6} | "
      f"{'2018w':>6} {'2020w':>6} {'2022H1':>7}")
for wg in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    cs, ds, w18, w20, w22 = [], [], [], [], []
    for oa in oa_navs:
        idx = oa.index.intersection(gold.index)
        mo = oa.loc[idx].resample('ME').last().pct_change().fillna(0)
        mg = gold.loc[idx].resample('ME').last().pct_change().fillna(0)
        bl = (1 + (1-wg)*mo + wg*mg).cumprod()
        c_, d_ = stats(bl)
        cs.append(c_); ds.append(d_)
        w18.append(wdd(bl, '2018-01-01', '2018-12-31'))
        w20.append(wdd(bl, '2020-01-01', '2020-12-31'))
        w22.append(wdd(bl, '2022-01-01', '2022-06-30'))
    cm, dm = np.median(cs), np.median(ds)
    print(f"{100-wg*100:3.0f}/{wg*100:2.0f}   | {cm:6.1f} [{min(cs):.1f}..{max(cs):.1f}] | "
          f"{dm:6.1f}% | {cm/abs(dm):6.2f} | {np.median(w18):6.1f} {np.median(w20):6.1f} "
          f"{np.median(w22):7.1f}", flush=True)
print('window 2015->now (GOLDBEES history); after-tax; medians of 10 OA seeds')
print('DONE', flush=True)
