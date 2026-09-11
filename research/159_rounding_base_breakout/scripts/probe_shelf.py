"""Probe a SHELF-breakout trigger: prior S bars form a tight consolidation (close range <= R%), the shelf high sits
within A% of the all-time-high close, and today closes above the shelf high on >= K x 20-day median volume."""
import sqlite3, pandas as pd, sys
c = sqlite3.connect('/home/arun/quantifyd/backtest_data/market_data.db')
S_LIST, R, A, K = (15, 20, 30), 0.12, 0.15, 3.0
cases = {'KMEW': ('2025-05-01', '2025-10-15'), 'CENTURYPLY': ('2017-01-01', '2017-06-30'), 'CHOLAHLDNG': ('2025-02-01', '2025-07-15'),
         'JAYSREETEA': ('2009-05-01', '2009-10-31'), 'MONARCH': ('2023-07-01', '2023-12-31'), 'NAM-INDIA': ('2025-03-01', '2025-08-31'),
         'SAPPHIRE': ('2022-07-01', '2022-12-31'), 'COROMANDEL': ('2009-06-01', '2009-12-31')}
for sym, (lo, hi) in cases.items():
    d = pd.read_sql(f"select date,open,close,volume from market_data_unified where symbol='{sym}' and timeframe='day' order by date", c)
    d = d[d.volume > 0].reset_index(drop=True)
    r = d.close.pct_change(); cut = r[r < -0.35].index.max()
    d = (d.loc[cut:] if pd.notna(cut) else d).reset_index(drop=True)
    d['ath'] = d.close.cummax().shift(1)
    d['volx'] = d.volume / d.volume.rolling(20).median().shift(1)
    out = []
    for S in S_LIST:
        shi = d.close.rolling(S).max().shift(1); slo = d.close.rolling(S).min().shift(1)
        tight = (shi - slo) / shi <= R
        near = shi >= d.ath * (1 - A)
        fire = (d.close > shi) & tight & near & (d.volx >= K) & (d.close > d.close.shift(1))
        for i in d.index[fire]:
            if lo <= d.date[i] <= hi:
                out.append((S, d.date[i], round(d.close[i], 2), round(shi[i], 2), f'{(shi[i]-slo[i])/shi[i]*100:.1f}%', round(d.ath[i], 2), f'{(d.close[i]/d.ath[i]-1)*100:+.1f}%', round(d.volx[i], 1)))
    print(f'\n== {sym} ==  (S, date, close, shelf_high, shelf_range, ATH, close_vs_ATH, volx)')
    for o in out: print('  ', o)
    if not out: print('   no shelf breakout in window')
