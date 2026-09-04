"""Build GOLD_INR_REF (XAUUSD x USDINR, Stooq daily) to extend gold history
pre-2015; validate vs GOLDBEES on the 2015+ overlap; recompute (a) the
45-45-10 blend's yearly cells 2007-2014 + full-window summary, and (b) the
OA/GOLD two-sleeve frontier on the extended 2007+ window (now containing
2008 AND 2013 — the two years the adoption caveats worry about).
Reference series only: saved to results/gold_inr_ref.csv, NOT into market_data.db.
"""
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path('/home/arun/quantifyd/research/147_third_sleeve_archetypes')
R142 = Path('/home/arun/quantifyd/research/142_bananapatterns_replication/scripts')
R144 = Path('/home/arun/quantifyd/research/144_truenorth_reassessment/scripts')
sys.path.insert(0, str(R142))
sys.path.insert(0, str(R144))


import json


def yahoo_monthly(path):
    rows = json.load(open(path))
    s = pd.Series({pd.Timestamp(t, unit='s'): v for t, v in rows}).sort_index()
    # month-start stamps -> month-end labels (align with resample('ME'))
    s.index = s.index.to_period('M').to_timestamp('M')
    return s[~s.index.duplicated(keep='last')]

xau = yahoo_monthly(STUDY / 'results' / 'yahoo_gc.json')    # COMEX front gold, USD/oz
inr = yahoo_monthly(STUDY / 'results' / 'yahoo_inr.json')   # USDINR
print(f'gold-usd {xau.index[0].date()}..{xau.index[-1].date()} n={len(xau)} (GC=F futures proxy)')
print(f'usd-inr  {inr.index[0].date()}..{inr.index[-1].date()} n={len(inr)}')
ref = (xau * inr.reindex(xau.index).ffill()).dropna()
ref.name = 'gold_inr_ref_monthly'
ref.to_csv(STUDY / 'results' / 'gold_inr_ref.csv')

import bluesky_replay as br
import tn_sweep as tn

ctx = tn.Ctx()
try:
    row = tn.run(ctx, tax=True)
except TypeError:
    row = tn.run(ctx)
tn_nav = row['_nav'].dropna()

w = br.load_frames('2004-06-01', trail_sma=15)
close = w['close']
gb = close['GOLDBEES'].dropna()

# validation on overlap (monthly returns; ref is already month-end)
mr = ref.pct_change().dropna()
mg = gb.resample('ME').last().pct_change().dropna()
common = mr.index.intersection(mg.index)
corr = mr.loc[common].corr(mg.loc[common])
drift = (mg.loc[common] - mr.loc[common]).mean() * 12 * 100
print(f'VALIDATION vs GOLDBEES (monthly, {len(common)} months): corr {corr:.3f}, '
      f'GOLDBEES minus REF annualized drift {drift:+.2f}pp (tracking/expense/roll)')

# spliced gold monthly returns: REF before 2015, GOLDBEES from 2015
ref_m = ref.pct_change()
gb_m = gb.resample('ME').last().pct_change()
gold_m = pd.concat([ref_m[ref_m.index < '2015-02-01'], gb_m[gb_m.index >= '2015-02-01']])
gold_m = gold_m[~gold_m.index.duplicated()].sort_index()

# gold-INR yearly returns for the record (2007-2014 from REF)
print('\nGOLD-INR yearly (REF pre-2015):')
gold_nav = (1 + gold_m.fillna(0)).cumprod()
for yr, seg in gold_nav.groupby(gold_nav.index.year):
    if 2007 <= yr <= 2015:
        prev = gold_nav[gold_nav.index.year < yr]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        print(f'  {yr}: {(seg.iloc[-1]/base-1)*100:+.1f}%')

# OA seeds (adopted spec, after-tax)
high, open_, athcp, sma15, tv20 = (w[k] for k in ('high', 'open', 'athcp', 'sma50', 'tv20'))
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
wkz = np.zeros(len(dates), dtype=bool)
oa_navs = []
for seed in range(1, 11):
    eq, _, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S, RSv, TVv,
                           trig, wkz, True, 0.0025, stop=0.08, slots=16,
                           size_pct=0.0625, stcg=0.20, cash_yield=0.05)
    oa_navs.append(pd.Series(np.asarray(eq, dtype=float), index=dates[days]))
print('OA seeds done', flush=True)


def yearly(nav):
    out = {}
    for yr, seg in nav.groupby(nav.index.year):
        prev = nav[nav.index.year < yr]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        run_ = pd.concat([pd.Series([base]), seg])
        out[yr] = ((seg.iloc[-1]/base-1)*100, float((run_/run_.cummax()-1).min()*100))
    return out


def blend_m(oa, wts):
    idx = oa.index.intersection(tn_nav.index)
    mo = oa.loc[idx].resample('ME').last().pct_change().fillna(0)
    mt = tn_nav.loc[idx].resample('ME').last().pct_change().fillna(0)
    mgold = gold_m.reindex(mo.index).fillna(0)
    r = wts[0]*mo + wts[1]*mt + wts[2]*mgold
    return (1 + r).cumprod()

# (a) 45-45-10 yearly cells 2007-2014 + full summary from 2007
ys = [yearly(blend_m(oa, (0.45, 0.45, 0.10))) for oa in oa_navs]
print('\n45-45-10 with spliced gold, yearly medians:')
for yr in range(2007, 2016):
    rr = [y[yr][0] for y in ys if yr in y]
    dd = [y[yr][1] for y in ys if yr in y]
    print(f'  {yr}: {np.median(rr):+.1f}% ({np.median(dd):.1f}%)')
fulls = []
for oa in oa_navs:
    bl = blend_m(oa, (0.45, 0.45, 0.10))
    bl = bl[bl.index >= '2007-01-01']
    yrs_ = (bl.index[-1]-bl.index[0]).days/365.25
    fulls.append((((bl.iloc[-1]/bl.iloc[0])**(1/yrs_)-1)*100,
                  float((bl/bl.cummax()-1).min()*100)))
cm = np.median([f[0] for f in fulls]); dm = np.median([f[1] for f in fulls])
print(f'45-45-10 FULL 2007->2026: CAGR {cm:.1f}% dd {dm:.1f}% Calmar {cm/abs(dm):.2f}')

# (b) OA/GOLD frontier on 2007+
print('\nOA/GOLD frontier, 2007->now (incl. 2008 & 2013):')
for wg in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    cs, ds = [], []
    for oa in oa_navs:
        mo = oa.resample('ME').last().pct_change().fillna(0)
        mo = mo[mo.index >= '2007-01-01']
        mgold = gold_m.reindex(mo.index).fillna(0)
        bl = (1 + (1-wg)*mo + wg*mgold).cumprod()
        yrs_ = (bl.index[-1]-bl.index[0]).days/365.25
        cs.append(((bl.iloc[-1])**(1/yrs_)-1)*100)
        ds.append(float((bl/bl.cummax()-1).min()*100))
    cm, dm = np.median(cs), np.median(ds)
    print(f'  {100-wg*100:3.0f}/{wg*100:2.0f}: CAGR {cm:5.1f}% [{min(cs):.1f}..{max(cs):.1f}] '
          f'dd {dm:6.1f}% Calmar {cm/abs(dm):.2f}')
print('DONE', flush=True)
