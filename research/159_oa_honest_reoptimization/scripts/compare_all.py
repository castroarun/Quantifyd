# -*- coding: utf-8 -*-
"""All five books on one table and one correlation matrix, after tax, same window.

Series compared:
    OA v2    research/161, ATH base-age breakout, next-open fill both sides
    OA v3    this session, r/142 signal with the trail and VIX gate re-optimised
    TN       True North incumbent, from research/144's own after-tax NAV
    IPO      IPO Base on the HONEST next-day entry (15.0%), not the study's same-day entry
    NIFTY    NIFTYBEES, the index, as a price series

Everything after tax. The window is forced by the shortest series - OA v3's gate needs
INDIA VIX, which starts 2015 - so the common comparison runs 2016 onward and the longer
figures are quoted separately rather than mixed in.

CORRELATION is computed on WEEKLY returns, not daily. Daily correlation between books that
trade different names on different days understates how much they move together in the only
situation that matters, which is a drawdown; weekly is the standard used elsewhere in this
project (r/142 reported both).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/159_oa_honest_reoptimization/results'
R161 = ROOT / 'research/161_ath_base_age_breakout/results'

navs = {}

# --- OA v3 and TN and the index, already built ------------------------------
df = pd.read_csv(RES / 'curves_after_tax.csv', index_col=0, parse_dates=True)
navs['OA v3 (gated)'] = df['OA gated (VIX p70)']
navs['TN incumbent'] = df['TN incumbent']
navs['NIFTYBEES'] = df['NIFTYBEES']

# --- OA v2 ------------------------------------------------------------------
z = np.load(R161 / 'curves161.npz', allow_pickle=True)
print('curves161.npz keys:', list(z.keys())[:8])
# 'WINNER' is the adopted v2 cell; 'bh' is its buy-and-hold reference, and the rest are
# the neighbour and ablation arms kept for their own sweep.
key_nav = 'WINNER' if 'WINNER' in z else None
key_date = 'dates' if 'dates' in z else None
print('  using nav key=%s date key=%s' % (key_nav, key_date))
if key_nav is not None and key_date is not None:
    d = pd.to_datetime([str(x)[:10] for x in z[key_date]])
    arr = z[key_nav]
    if arr.ndim == 2:
        # many seeds: take the seed whose CAGR is the median, never the mean of paths
        fin = arr[:, -1] if arr.shape[0] < arr.shape[1] else arr[-1, :]
        arr = (arr[int(np.argsort(fin)[len(fin) // 2]), :] if arr.shape[0] < arr.shape[1]
               else arr[:, int(np.argsort(fin)[len(fin) // 2])])
    s = pd.Series(np.asarray(arr, dtype=float), index=d).dropna()
    navs['OA v2'] = s / s.iloc[0]
else:
    print('  !! could not identify the curve arrays; OA v2 omitted')

# --- IPO Base, honest next-day entry ----------------------------------------
_ipo = RES / 'ipo_honest_curve.csv'
if _ipo.exists():
    s_ipo = pd.read_csv(_ipo, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    navs['IPO (honest)'] = s_ipo / s_ipo.iloc[0]
else:
    print('  !! ipo_honest_curve.csv not built yet; IPO omitted')

# --- align ------------------------------------------------------------------
start = max(s.dropna().index.min() for s in navs.values())
end = min(s.dropna().index.max() for s in navs.values())
al = pd.DataFrame({k: v for k, v in navs.items()}).loc[start:end].ffill().dropna()
al = al / al.iloc[0]
print()
print('common window: %s -> %s  (%d days)' % (al.index[0].date(), al.index[-1].date(),
                                              len(al)))
print()
yrs = (al.index[-1] - al.index[0]).days / 365.25
print('%-16s %8s %9s %8s %8s' % ('series', 'CAGR', 'maxDD', 'Calmar', 'final'))
summ = {}
for k in al:
    s = al[k]
    cagr = (s.iloc[-1] ** (1 / yrs) - 1) * 100
    dd = (s / s.cummax() - 1).min() * 100
    summ[k] = (cagr, dd, cagr / abs(dd))
    print('%-16s %7.2f%% %8.2f%% %8.2f %7.2fx' % (k, cagr, dd, cagr / abs(dd), s.iloc[-1]))

# --- yearly, house format ----------------------------------------------------
print()
rows = {}
for k in al:
    for y, sub in al[k].groupby(al.index.year):
        if len(sub) < 20:
            continue
        rows.setdefault(y, {})[k] = (100.0 * (sub.iloc[-1] / sub.iloc[0] - 1),
                                     100.0 * (sub / sub.cummax() - 1).min())
cols = [c for c in al.columns]
sysl = [c for c in cols if c != 'NIFTYBEES']          # benchmarks excluded from best-of
print('%-6s' % 'year' + ''.join('%20s' % c for c in cols) + '  %-14s %-14s %-14s'
      % ('BEST CAGR', 'LEAST DD', 'BEST OVERALL'))
for y in sorted(rows):
    r = rows[y]
    line = '%-6d' % y + ''.join(
        '%20s' % ('%+.1f (%.1f)' % r[c] if c in r else '-') for c in cols)
    have = {c: r[c] for c in sysl if c in r}
    if have:
        bc = max(have, key=lambda c: have[c][0])
        ld = max(have, key=lambda c: have[c][1])
        bo = max(have, key=lambda c: have[c][0] + have[c][1])
        line += '  %-14s %-14s %-14s' % (bc, ld, bo)
    print(line)
print()
print('%-6s' % 'FULL' + ''.join('%20s' % ('%.1f (%.1f)' % (summ[c][0], summ[c][1]))
                                for c in cols))

# --- correlation -------------------------------------------------------------
wk = al.resample('W-FRI').last().pct_change().dropna()
print()
print('WEEKLY return correlation (%d weeks):' % len(wk))
c = wk.corr()
print('%-16s' % '' + ''.join('%16s' % k[:14] for k in c.columns))
for k in c.index:
    print('%-16s' % k[:15] + ''.join('%16.2f' % c.loc[k, j] for j in c.columns))

al.to_csv(RES / 'all_systems_after_tax.csv')
json.dump({k: [round(v, 2) for v in summ[k]] for k in summ},
          open(RES / 'all_systems_summary.json', 'w'), indent=1)
print()
print('wrote all_systems_after_tax.csv')
