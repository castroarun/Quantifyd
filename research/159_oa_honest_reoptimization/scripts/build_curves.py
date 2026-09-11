# -*- coding: utf-8 -*-
"""Curves and a year-by-year table for the best honest Open Alpha configuration.

The config: buy at the breakout close (~15:10), 75-day trail, 8% stop (inert but kept),
16 slots at 6.25%, funds excluded, 25 bps a side, 5% on idle cash, gated on INDIA VIX
above its own one-year 70th percentile.

Four series, all AFTER TAX so the comparison is like for like:

    OA_gated   the config above, 2016-2026 (VIX starts 2015, so this is its whole life)
    OA_nogate  the same config without the gate, for attribution
    TN         True North incumbent, after tax, from research/144's own NAV file
    IPO        IPO Base on the HONEST next-day entry (not the study's same-day entry,
               which needs the close known at the open)
    NIFTYBEES  the index, as a price series

Written as a tidy CSV of daily NAVs rebased to 1.0 plus a year-by-year table in the house
format - annual return with the intra-year maximum drawdown beneath it, measured from the
running peak of that year's own curve.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/158_oa_arming_width/scripts'))
sys.path.insert(0, str(ROOT / 'research/153_ipo_base/scripts'))
sys.path.insert(0, str(ROOT))
import oa_entry_mechanics as em          # noqa: E402

RES = ROOT / 'research/159_oa_honest_reoptimization/results'
SEEDS = list(range(1, 31))
COST, SLOTS, SIZE, STOP, TRAIL = 0.0025, 16, 0.0625, 0.08, 75
CASH_Y, STCG, LTCG = 0.05, 0.20, 0.125


def vix_gate(index, pct=0.70):
    import sqlite3
    con = sqlite3.connect('file:%s?mode=ro' % (ROOT / 'backtest_data/market_data.db'),
                          uri=True)
    d = pd.read_sql_query("select date, close from market_data_unified where "
                          "symbol='INDIAVIX' and timeframe='day' order by date", con)
    con.close()
    d['date'] = pd.to_datetime(d['date'].str[:10])
    s = d.drop_duplicates('date').set_index('date')['close'].dropna()
    off = (s > s.rolling(252, min_periods=252).quantile(pct)).shift(1)
    return off.reindex(index).ffill().fillna(False).astype(bool).to_numpy()


def oa_curve(w, gated, win):
    close, high, athcp, tv20 = w['close'], w['high'], w['athcp'], w['tv20']
    etf = [c for c in close.columns if em.is_etf(c)]
    tvp, prev = tv20.shift(1), close.shift(1)
    elig = tvp >= em.TV_FLOOR
    elig[etf] = False
    r = {n: close / close.shift(n) - 1 for n in (63, 126, 189, 252)}
    rs = ((2 * r[63] + r[126] + r[189] + r[252]).where(elig)
          .rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev < athcp) & (prev >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = setup & (close > athcp) & athcp.notna()
    dates = close.index
    days = np.array([i for i, d in enumerate(dates)
                     if win[0] <= str(d.date()) <= win[1]])
    garr = vix_gate(dates) if gated else np.zeros(len(dates), bool)
    curves, cagrs = [], []
    for sd in SEEDS:
        eq, trd, _, _ = em.simulate(
            sd, 'random', days, dates, close.values, high.values, w['open'].values,
            athcp.values, w['sma50'].values, rs.values, tvp.values,
            trig.fillna(False).values, garr, True, COST, stop=STOP, slots=SLOTS,
            size_pct=SIZE, fill_close=True, cash_yield=CASH_Y, stcg=STCG, ltcg=LTCG)
        s = pd.Series(eq, index=dates[days])
        curves.append(s / s.iloc[0])
        st, _ = em.stats_from(eq, dates[days], trd, em.CAPITAL)
        cagrs.append(st['cagr'])
    # the MEDIAN-CAGR seed, not the mean of curves: averaging 30 equity paths
    # manufactures a smoother curve than any single book could have run.
    k = int(np.argsort(cagrs)[len(cagrs) // 2])
    return curves[k], float(np.median(cagrs))


def yearly_table(navs):
    """-> DataFrame of year x series, each cell 'ret% (maxdd%)' in the house format."""
    out = {}
    for name, s in navs.items():
        rows = {}
        for y, sub in s.groupby(s.index.year):
            if len(sub) < 20:
                continue
            ret = 100.0 * (sub.iloc[-1] / sub.iloc[0] - 1)
            dd = 100.0 * (sub / sub.cummax() - 1).min()
            rows[y] = (round(float(ret), 1), round(float(dd), 1))
        out[name] = rows
    return out


print('loading frames (trail-%d) ...' % TRAIL, flush=True)
w = em.load_frames('2005-01-01', trail_sma=TRAIL)
dates = w['close'].index

print('OA gated 2016-2026 ...', flush=True)
oa_g, cg = oa_curve(w, True, ('2016-01-01', '2026-08-31'))
print('  median CAGR after tax: %.2f%%' % cg, flush=True)
print('OA ungated 2016-2026 ...', flush=True)
oa_n, cn = oa_curve(w, False, ('2016-01-01', '2026-08-31'))
print('  median CAGR after tax: %.2f%%' % cn, flush=True)

tn = pd.read_csv(ROOT / 'research/144_truenorth_reassessment/results/'
                        'nav_INC_cash_n8_d15_tax1.csv', index_col=0, parse_dates=True)
tn = tn.iloc[:, 0].dropna()
nb = w['close'].get('NIFTYBEES').dropna()

navs = {'OA gated (VIX p70)': oa_g, 'OA no gate': oa_n,
        'TN incumbent': tn, 'NIFTYBEES': nb}
common = max(s.index.min() for s in navs.values())
navs = {k: (v[v.index >= common] / v[v.index >= common].iloc[0])
        for k, v in navs.items()}
df = pd.DataFrame(navs).ffill()
df.to_csv(RES / 'curves_after_tax.csv')
print()
print('common window: %s -> %s' % (df.index[0].date(), df.index[-1].date()))
for k in df:
    s = df[k].dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] ** (1 / yrs) - 1) * 100
    dd = (s / s.cummax() - 1).min() * 100
    print('  %-22s CAGR %6.2f%%  maxDD %7.2f%%  Calmar %5.2f  final %.2fx'
          % (k, cagr, dd, cagr / abs(dd), s.iloc[-1]))

yt = yearly_table({k: df[k].dropna() for k in df})
import json
json.dump({k: {str(y): v for y, v in r.items()} for k, r in yt.items()},
          open(RES / 'yearly_after_tax.json', 'w'), indent=1)
print()
print('year   ' + ''.join('%22s' % k for k in df.columns))
for y in sorted({y for r in yt.values() for y in r}):
    line = '%-7d' % y
    for k in df.columns:
        v = yt[k].get(y)
        line += '%22s' % ('%+.1f%% (%.1f%%)' % v if v else '-')
    print(line)
print()
print('wrote %s and %s' % ('curves_after_tax.csv', 'yearly_after_tax.json'))
