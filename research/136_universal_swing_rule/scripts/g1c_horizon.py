"""G1c: does the reversion excess keep growing with holding period?

G1b established RSI(2)<10 as a real signal — excess over a date-matched random
entry of +0.091% / +0.104% / +0.117% at 2 / 5 / 10 days, t 9.1 / 6.7 / 5.4 on
312k signals. But ~10 bps is smaller than a 20-30 bps round trip, so it is a
signal and not yet a strategy.

The excess rose monotonically with horizon while cost is paid once. That is the
whole question this run answers: extend to 15 / 20 / 30 days.

  * If the excess keeps climbing toward 30-40 bps, cost stops mattering and this
    becomes a G2 candidate — a real 2-30 day book.
  * If it flattens near 12 bps, the standalone book is dead and the survivor is
    an entry-timing overlay on the momentum book already running.

Three arms, all reported, no cherry-picking:
  A  RSI(2) < 10                          the G1b carrier
  B  RSI(2) < 10 and close > 200-DMA      buy weakness only inside strength
  C  RSI(2) < 5                           does deeper oversold pay more?

Arms B and C are the 'condition harder' route. Each condition is a fitting
opportunity, so they are measured here but nothing is selected on them yet —
that needs held-out data at G3.

Read-only.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logging.disable(logging.WARNING)

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT = ROOT / 'research' / '136_universal_swing_rule' / 'results'
OUT.mkdir(parents=True, exist_ok=True)

START = '2015-01-01'
MIN_HISTORY = 250
MIN_TURNOVER = 5e7
HORIZONS = (2, 5, 10, 15, 20, 30)
SEED = 20260831

rng = np.random.default_rng(SEED)

print('loading daily bars...', flush=True)
conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
df = pd.read_sql_query(
    "SELECT symbol, substr(date,1,10) d, open, close, volume "
    "FROM market_data_unified WHERE timeframe='day' AND date >= ? ORDER BY symbol, date",
    conn, params=(START,))
conn.close()
print(f'  {len(df):,} rows | {df.symbol.nunique()} symbols | {df.d.min()} -> {df.d.max()}',
      flush=True)

df['d'] = pd.to_datetime(df['d'])
df = df.sort_values(['symbol', 'd'])
g = df.groupby('symbol', sort=False)

df['turnover'] = df['close'] * df['volume']
df['turn20'] = g['turnover'].transform(lambda s: s.rolling(20, min_periods=20).median())
df['bars'] = g.cumcount()
df['next_open'] = g['open'].shift(-1)
df['sma200'] = g['close'].transform(lambda s: s.rolling(200, min_periods=200).mean())


def _rsi(s, n=2):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


df['rsi2'] = g['close'].transform(_rsi)

for h in HORIZONS:
    df[f'exit{h}'] = g['open'].shift(-(h + 1))
    df[f'fwd{h}'] = df[f'exit{h}'] / df['next_open'] - 1.0

df = df[(df['bars'] >= MIN_HISTORY) & (df['turn20'] >= MIN_TURNOVER)
        & df['next_open'].notna()].copy()
print(f'  eligible rows: {len(df):,} | {df.symbol.nunique()} symbols\n', flush=True)

ARMS = {
    'A  rsi2<10': df['rsi2'] < 10,
    'B  rsi2<10 & >200dma': (df['rsi2'] < 10) & (df['close'] > df['sma200']),
    'C  rsi2<5': df['rsi2'] < 5,
}


def stats(x):
    x = x.dropna()
    if len(x) < 30:
        return len(x), np.nan
    return len(x), x.mean()


rows = []
for name, mask in ARMS.items():
    print(f'=== {name}: {int(mask.sum()):,} signals ===', flush=True)
    for h in HORIZONS:
        col = f'fwd{h}'
        sig = df.loc[mask, col]
        counts = df.loc[mask].groupby('d').size()
        pool = df[[col, 'd']].dropna()
        draws = []
        for day, k in counts.items():
            same = pool.loc[pool['d'] == day, col].to_numpy()
            if len(same):
                draws.append(rng.choice(same, size=min(k, len(same)), replace=False))
        ctrl = pd.Series(np.concatenate(draws)) if draws else pd.Series(dtype=float)

        n_s, m_s = stats(sig)
        n_c, m_c = stats(ctrl)
        ex = m_s - m_c
        se = np.sqrt(sig.var(ddof=1) / n_s + ctrl.var(ddof=1) / n_c) if n_c else np.nan
        t = ex / se if se == se else np.nan
        rows.append(dict(arm=name, horizon=h, n=n_s, signal_pct=100 * m_s,
                         control_pct=100 * m_c, excess_bps=10000 * ex, t=t))
        print('  {:>2}d  signal {:+6.3f}%  random {:+6.3f}%  |  excess {:+6.1f} bps  '
              't {:5.2f}   (n={:,})'.format(h, 100 * m_s, 100 * m_c, 10000 * ex, t, n_s),
              flush=True)
    print('', flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / 'g1c_horizon.csv', index=False)

print('=== EXCESS NET OF A ROUND TRIP (bps) — does it clear the toll? ===', flush=True)
hdr = '{:<22} {:>5}'.format('arm', 'h')
for b in (20, 30, 50):
    hdr += '{:>12}'.format(f'-{b}bps')
print(hdr, flush=True)
for _, r in res.iterrows():
    line = '{:<22} {:>4}d'.format(r.arm, int(r.horizon))
    for b in (20, 30, 50):
        v = r.excess_bps - b
        line += '{:>+12.1f}'.format(v)
    print(line, flush=True)

print('\n=== is the excess monotonic in horizon? (the shape that matters) ===', flush=True)
for name in ARMS:
    sub = res[res.arm == name].sort_values('horizon')
    seq = sub.excess_bps.tolist()
    mono = all(b >= a - 1.0 for a, b in zip(seq, seq[1:]))
    print('  {:<22} {}  {}'.format(
        name, ' -> '.join(f'{v:+.0f}' for v in seq),
        'MONOTONIC' if mono else 'peaked/irregular'), flush=True)

print(f'\nwrote {OUT / "g1c_horizon.csv"}')
