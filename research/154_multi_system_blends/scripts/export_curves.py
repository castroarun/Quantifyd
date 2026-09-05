"""Export month-end growth-of-100 curves + drawdown series for the five systems and
NIFTY, as compact JSON for the comparison page. Median path = cross-sectional median
NAV across seeds/offsets at each date (rebased to 100 at each series' own start)."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/arun/quantifyd/research')
OUT = R / '154_multi_system_blends' / 'results'
SRC = {
    'OA':  R / '154_multi_system_blends/results/oa_navs30.csv',
    'TN':  R / '154_multi_system_blends/results/tn_navs12.csv',
    'VCP': R / '151_vcp_breakout/results/vcp_equity_seeds.csv',
    'MYB': R / '152_multiyear_breakout/results/myb_equity_seeds.csv',
    'IPO': R / '153_ipo_base/results/ipo_equity_seeds.csv',
}

series, dds = {}, {}
for name, path in SRC.items():
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime([str(x)[:10] for x in df.index])
    df = df.sort_index().astype(float)
    med = df.median(axis=1)                      # cross-sectional median path
    med = med / med.iloc[0] * 100.0
    m = med.resample('ME').last().dropna()
    series[name] = m
    dds[name] = (m / m.cummax() - 1.0) * 100.0

con = sqlite3.connect('/home/arun/quantifyd/backtest_data/market_data.db')
q = con.execute("SELECT date, close FROM market_data_unified WHERE symbol='NIFTYBEES' "
                "AND timeframe='day' AND date>='2006-01-01' ORDER BY date").fetchall()
con.close()
b = pd.Series({pd.Timestamp(str(d)[:10]): float(c) for d, c in q}).sort_index()
b = b / b.iloc[0] * 100.0
series['NIFTY'] = b.resample('ME').last().dropna()
dds['NIFTY'] = (series['NIFTY'] / series['NIFTY'].cummax() - 1.0) * 100.0

idx = sorted(set().union(*[set(s.index) for s in series.values()]))
payload = {
    'dates': [d.strftime('%Y-%m') for d in idx],
    'curves': {k: [None if d not in v.index else round(float(v[d]), 2) for d in idx]
               for k, v in series.items()},
    'dd': {k: [None if d not in v.index else round(float(v[d]), 2) for d in idx]
           for k, v in dds.items()},
    'note': 'Month-end, growth of Rs 100 from each series own start (MYB starts 2010). '
            'Median path across 30 seeds (12 offsets for TN). After-tax, 25 bps/side, '
            'idle cash 5% p.a. Drawdown measured from each curve running peak.',
}
json.dump(payload, open(OUT / 'curves_five_systems.json', 'w'))
for k, v in series.items():
    print(f'{k:6s} {v.index[0].date()} .. {v.index[-1].date()}  final {v.iloc[-1]:,.0f}  '
          f'maxDD {dds[k].min():.1f}%')
print('wrote curves_five_systems.json')
