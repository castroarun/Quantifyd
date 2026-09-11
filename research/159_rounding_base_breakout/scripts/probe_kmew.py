"""research/159 — inspect KMEW's saucer shape to calibrate detector thresholds. Read-only."""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'
con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
df = pd.read_sql_query(
    "SELECT date,open,high,low,close,volume FROM market_data_unified "
    "WHERE timeframe='day' AND symbol='KMEW' ORDER BY date", con, parse_dates=['date'])
con.close()
df = df[df['volume'] > 0].reset_index(drop=True)
print('rows(vol>0)=', len(df), df['date'].min().date(), df['date'].max().date())

# monthly summary
m = df.set_index('date').resample('ME').agg(
    o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'),
    v=('volume', 'sum'), n=('close', 'size'))
pd.set_option('display.width', 200)
print(m.round(1).to_string())

# OBV
sign = np.sign(df['close'].diff().fillna(0.0))
df['obv'] = (sign * df['volume']).cumsum()

# left rim = post-listing high close; trough = min close
i_rim = int(df['close'][:120].idxmax())
i_tr = int(df['close'].idxmin())
print('\nfirst-120d max close idx=%d date=%s close=%.1f' % (i_rim, df.date[i_rim].date(), df.close[i_rim]))
print('global min close idx=%d date=%s close=%.1f' % (i_tr, df.date[i_tr].date(), df.close[i_tr]))
R = df.close[i_rim]
print('depth from that rim = %.1f%%' % (100 * (R - df.close[i_tr]) / R))

# first close back above R
after = df[df.index > i_tr]
xs = after[after['close'] > R]
if len(xs):
    j = xs.index[0]
    print('first close > rim after trough: idx=%d date=%s close=%.1f' % (j, df.date[j].date(), df.close[j]))
    print('  bars from listing=%d, bars from trough=%d' % (j, j - i_tr))
    if j + 1 < len(df):
        print('  next-day open=%.1f date=%s' % (df.open[j + 1], df.date[j + 1].date()))
# quadratic fit on log close from listing to breakout
for end in [i_tr + 40, i_tr + 60, i_tr + 80, (xs.index[0] if len(xs) else len(df) - 1)]:
    end = min(end, len(df) - 1)
    seg = df.iloc[0:end + 1]
    x = np.arange(len(seg))
    y = np.log(seg['close'].values)
    a, b, c = np.polyfit(x, y, 2)
    yh = a * x**2 + b * x + c
    ss = 1 - ((y - yh)**2).sum() / ((y - y.mean())**2).sum()
    vx = -b / (2 * a)
    print('window 0..%d (n=%d, end=%s): curv=%.3e R2=%.3f vertex_frac=%.2f'
          % (end, len(seg), seg['date'].iloc[-1].date(), a, ss, vx / len(seg)))

# volume halves
n = len(df)
print('\nmedian vol first-half=%.0f second-half=%.0f' % (df.volume[:n//2].median(), df.volume[n//2:].median()))
# 20d median traded value at various points
df['tv'] = df['close'] * df['volume']
print('20d median traded value (Rs cr) at trough=%.2f' % (df['tv'].rolling(20).median()[i_tr] / 1e7))
if len(xs):
    print('20d median traded value (Rs cr) at breakout=%.2f' % (df['tv'].rolling(20).median()[xs.index[0]] / 1e7))
