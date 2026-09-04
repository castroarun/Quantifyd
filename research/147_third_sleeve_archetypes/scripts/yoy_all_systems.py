"""YoY table for the report: OA (adopted spec) / TN (incumbent) / 50-50 / 45-45-10 gold.
Per year: return + intra-year maxDD. OA & blends = median across 10 OA seeds; TN
deterministic (offset 0). After-tax both legs (OA stcg in-sim; TN engine tax on).
Gold = GOLDBEES closes (history from 2015 -> gold blend column starts 2015).
Summary = full-window CAGR / maxDD / Calmar per config (windows stated).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R142 = Path('/home/arun/quantifyd/research/142_bananapatterns_replication/scripts')
R144 = Path('/home/arun/quantifyd/research/144_truenorth_reassessment/scripts')
sys.path.insert(0, str(R142))
sys.path.insert(0, str(R144))
import bluesky_replay as br
import tn_sweep as tn

# ---- TN incumbent nav (after-tax if the engine supports it) ----
ctx = tn.Ctx()
try:
    row = tn.run(ctx, tax=True)
    tn_tax = True
except TypeError:
    row = tn.run(ctx)
    tn_tax = False
tn_nav = row['_nav'].dropna()
print(f'TN nav {tn_nav.index[0].date()}..{tn_nav.index[-1].date()} tax={tn_tax} '
      f'w0 {row["w0_cagr"]}%', flush=True)

# ---- OA adopted spec, 10 seeds, after-tax ----
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
print('OA 10 seeds done', flush=True)

gold = close['GOLDBEES'].dropna()
gold = gold[gold.index >= '2015-01-01']


def yearly(nav):
    out = {}
    for yr, seg in nav.groupby(nav.index.year):
        prev = nav[nav.index.year < yr]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        run_ = pd.concat([pd.Series([base]), seg])
        out[yr] = ((seg.iloc[-1]/base - 1)*100,
                   float((run_/run_.cummax() - 1).min()*100))
    return out


def full(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1]/nav.iloc[0])**(1/yrs) - 1
    dd = (nav/nav.cummax()-1).min()
    return cagr*100, dd*100


def blend(navs, weights):
    """navs: list of Series; monthly-rebalanced blend on the common index."""
    idx = navs[0].index
    for nv in navs[1:]:
        idx = idx.intersection(nv.index)
    ms = [nv.loc[idx].resample('ME').last().pct_change().fillna(0) for nv in navs]
    r = sum(wt*m for wt, m in zip(weights, ms))
    return (1 + r).cumprod()


def med_yearly(nav_list):
    ys = [yearly(nv) for nv in nav_list]
    years = sorted(set().union(*[y.keys() for y in ys]))
    out = {}
    for yr in years:
        rets = [y[yr][0] for y in ys if yr in y]
        dds = [y[yr][1] for y in ys if yr in y]
        out[yr] = (float(np.median(rets)), float(np.median(dds)))
    return out

cfg = {}
cfg['OA'] = med_yearly(oa_navs)
cfg['TN'] = yearly(tn_nav)
b5050 = [blend([oa, tn_nav], [0.5, 0.5]) for oa in oa_navs]
cfg['B5050'] = med_yearly(b5050)
b454510 = [blend([oa, tn_nav, gold], [0.45, 0.45, 0.10]) for oa in oa_navs]
cfg['B454510'] = med_yearly(b454510)

years = sorted(cfg['OA'].keys())
print('\nYEAR;OA_ret;OA_dd;TN_ret;TN_dd;B5050_ret;B5050_dd;GOLD10_ret;GOLD10_dd')
for yr in years:
    cells = []
    for k in ('OA', 'TN', 'B5050', 'B454510'):
        v = cfg[k].get(yr)
        cells += [f'{v[0]:.1f}', f'{v[1]:.1f}'] if v else ['', '']
    print(f'{yr};' + ';'.join(cells))

print('\nSUMMARY (full window per config):')
oa_f = [full(nv) for nv in oa_navs]
print(f"OA 2006-26: CAGR {np.median([x[0] for x in oa_f]):.1f}% "
      f"dd {np.median([x[1] for x in oa_f]):.1f}%")
c_, d_ = full(tn_nav)
print(f'TN {tn_nav.index[0].year}-26: CAGR {c_:.1f}% dd {d_:.1f}% (tax={tn_tax})')
b_f = [full(nv) for nv in b5050]
print(f"B5050: CAGR {np.median([x[0] for x in b_f]):.1f}% dd {np.median([x[1] for x in b_f]):.1f}% "
      f"({b5050[0].index[0].date()}..)")
g_f = [full(nv) for nv in b454510]
print(f"B454510: CAGR {np.median([x[0] for x in g_f]):.1f}% dd {np.median([x[1] for x in g_f]):.1f}% "
      f"({b454510[0].index[0].date()}..)")
print('DONE', flush=True)
