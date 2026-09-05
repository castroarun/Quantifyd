"""research/152 — head-to-head: is the MYB sleeve a better third sleeve than r/147's
GOLDBEES sleeve? Both judged on the SAME window (GOLDBEES history starts 2015-01) and the
SAME paths (10 OA seeds x 3 TN offsets), paired against the same-path TN+OA 50-50 baseline.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

RES = Path('/home/arun/quantifyd/research/152_multiyear_breakout/results')
R146 = Path('/home/arun/quantifyd/research/146_complementary_third_sleeve/results')
DB = '/home/arun/quantifyd/backtest_data/market_data.db'
OFFSETS = [0, 4, 8]


def stats(nav):
    y = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1
    d = float((nav / nav.cummax() - 1).min())
    return c * 100, d * 100, (c / abs(d) if d < 0 else np.nan)


con = sqlite3.connect(DB)
gold = pd.read_sql("SELECT substr(date,1,10) d, close FROM market_data_unified WHERE "
                   "symbol='GOLDBEES' AND timeframe='day' AND close IS NOT NULL ORDER BY d", con)
con.close()
gold['d'] = pd.to_datetime(gold['d'])
gold = gold.set_index('d')['close']

myb = pd.read_csv(RES / 'myb_equity_seeds.csv', index_col=0, parse_dates=True)
oa = pd.read_csv(R146 / 'oa_navs.csv', index_col=0, parse_dates=True)
tn = {o: pd.read_csv(R146 / f'tn_nav_off{o}.csv', index_col=0, parse_dates=True).iloc[:, 0]
      for o in OFFSETS if (R146 / f'tn_nav_off{o}.csv').exists()}

idx = myb.index.intersection(oa.index).intersection(gold.index)
for v in tn.values():
    idx = idx.intersection(v.index)
print(f'COMMON WINDOW: {idx[0].date()} -> {idx[-1].date()} '
      f'({(idx[-1]-idx[0]).days/365.25:.1f}y)  [GOLDBEES history is the binding constraint]')
cash = pd.Series((1 + 0.05 / 252) ** np.arange(len(idx)), index=idx)


def mr(s):
    return s.loc[idx].resample('ME').last().pct_change().fillna(0)


m_oa = {c: mr(oa[c]) for c in oa.columns}
m_tn = {o: mr(v) for o, v in tn.items()}
m_myb = {c: mr(myb[c]) for c in myb.columns}
m_gold, m_cash = mr(gold), mr(cash)
cols = list(oa.columns)
mcols = list(myb.columns)

med_myb = myb.loc[idx].median(axis=1)
print('\n--- correlations on this window (sleeve vs each leg, daily / monthly) ---')
for nm, sl in (('MYB', med_myb), ('GOLD', gold.loc[idx])):
    for ln, other in [('OA', oa.loc[idx].median(axis=1))] + [(f'TN{o}', v.loc[idx]) for o, v in tn.items()]:
        print(f'  {nm:5s} vs {ln:5s}: {sl.pct_change().corr(other.pct_change()):+.3f} / '
              f'{sl.resample("ME").last().pct_change().corr(other.resample("ME").last().pct_change()):+.3f}')

rows = []
for w3 in (0.0, 0.10, 0.15, 0.20):
    wl = (1 - w3) / 2
    for label, third in (('MYB', m_myb), ('GOLD', m_gold), ('cash-null', m_cash)):
        if w3 == 0 and label != 'MYB':
            continue
        cs, ds, ks, dk = [], [], [], []
        for off in tn:
            for j, c in enumerate(cols):
                t = third[mcols[j % len(mcols)]] if label == 'MYB' else third
                b = (1 + wl * m_oa[c] + wl * m_tn[off] + (w3 * t if w3 else 0)).cumprod()
                base = (1 + .5 * m_oa[c] + .5 * m_tn[off]).cumprod()
                x, bs = stats(b), stats(base)
                cs.append(x[0]); ds.append(x[1]); ks.append(x[2]); dk.append(x[2] - bs[2])
        rows.append(dict(w3=w3, sleeve='TN+OA 50-50 BASELINE' if w3 == 0 else label,
                         cagr=round(float(np.median(cs)), 2),
                         dd=round(float(np.median(ds)), 2),
                         calmar=round(float(np.median(ks)), 2),
                         calmar_worst=round(min(ks), 2),
                         paired_dCalmar=round(float(np.median(dk)), 3),
                         wins=f'{int(np.sum(np.array(dk)>0))}/{len(dk)}'))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / 'myb_vs_gold.csv', index=False)
print('\nDONE')
