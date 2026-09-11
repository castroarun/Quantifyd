"""Probe alternative breakout triggers on the two examples: SKFINDIA (Arun says BO = 16-May-2025) and KMEW (Sep-2025)."""
import sqlite3, pandas as pd, sys
c = sqlite3.connect('/home/arun/quantifyd/backtest_data/market_data.db')
for sym, lo, hi in [('SKFINDIA','2024-06-01','2025-08-01'), ('KMEW','2024-11-01','2025-11-15')]:
    d = pd.read_sql(f"select date,open,high,low,close,volume from market_data_unified where symbol='{sym}' and timeframe='day' and date between '{lo}' and '{hi}' order by date", c)
    d = d[d.volume > 0].reset_index(drop=True)
    d['med20'] = d.volume.rolling(20).median().shift(1)
    d['volx'] = (d.volume / d.med20).round(1)
    for n in (40, 60, 100, 250):
        d[f'hi{n}'] = d.close.rolling(n, min_periods=20).max().shift(1)
        d[f'b{n}'] = d.close > d[f'hi{n}']
    # first-day-of-cluster breakouts: close above prior N-day high, not above it yesterday
    print(f'\n===== {sym} =====')
    for n in (40, 60, 100, 250):
        f = d[d[f'b{n}'] & ~d[f'b{n}'].shift(1, fill_value=False)]
        print(f'first close > prior {n}-day high:', [(r.date, r.close, r.volx) for r in f.itertuples()])
    print('volume >= 3x med20 days:', [(r.date, r.close, r.volx) for r in d[d.volx >= 3].itertuples()])
    print('combined: close > 60d high AND vol >= 2x:', [(r.date, r.close, r.volx) for r in d[d.b60 & (d.volx >= 2)].itertuples()][:6])
    w = d.set_index(pd.to_datetime(d.date)).close.resample('W').agg(['min','max']).round(1)
    print(w.to_string())
