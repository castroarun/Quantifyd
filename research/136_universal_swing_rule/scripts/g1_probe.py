"""G1: does a universal breakout signal beat a date-matched random entry?

The cheapest test that can kill the idea. No rule mechanics, no stops, no sizing —
just: on the days this signal fires, does the stock do better over the next 2/5/10
days than a randomly chosen liquid stock on the same day?

That control is the whole point. In a market that rose most of the period, any long
signal shows a profit; research/87 and /88 both produced a raw t=10 that dissolved
into drift and survivorship once controls were added. So three numbers are always
reported together: the signal, a date-matched random draw, and the universe drift.

Entry is the NEXT day's open. The signal bar's close is never tradeable.

Read-only. Writes results to research/136_universal_swing_rule/results/.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

logging.disable(logging.WARNING)

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT = ROOT / 'research' / '136_universal_swing_rule' / 'results'
OUT.mkdir(parents=True, exist_ok=True)

START = '2015-01-01'
MIN_HISTORY = 250          # bars before a symbol is eligible
MIN_TURNOVER = 5e7         # ₹5 crore trailing-20d median
LOOKBACKS = (20, 55)
HORIZONS = (2, 5, 10)
SEED = 20260831

rng = np.random.default_rng(SEED)

print('loading daily bars…', flush=True)
conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
df = pd.read_sql_query(
    "SELECT symbol, substr(date,1,10) d, open, high, low, close, volume "
    "FROM market_data_unified WHERE timeframe='day' AND date >= ? ORDER BY symbol, date",
    conn, params=(START,))
conn.close()
print(f'  {len(df):,} rows · {df.symbol.nunique()} symbols · {df.d.min()} → {df.d.max()}',
      flush=True)

df['d'] = pd.to_datetime(df['d'])
df = df.sort_values(['symbol', 'd'])
g = df.groupby('symbol', sort=False)

# ── causal features only: every window looks backwards ─────────────────────
df['turnover'] = df['close'] * df['volume']
df['turn20'] = g['turnover'].transform(lambda s: s.rolling(20, min_periods=20).median())
df['bars'] = g.cumcount()
df['next_open'] = g['open'].shift(-1)

for n in LOOKBACKS:
    # highest close of the trailing n days INCLUDING today -> today is a new n-day high
    df[f'hi{n}'] = g['close'].transform(lambda s, n=n: s.rolling(n, min_periods=n).max())
    df[f'brk{n}'] = (df['close'] >= df[f'hi{n}']) & df[f'hi{n}'].notna()

for h in HORIZONS:
    # entry at next open, exit at the open h days later — both tradeable prices
    df[f'exit{h}'] = g['open'].shift(-(h + 1))
    df[f'fwd{h}'] = df[f'exit{h}'] / df['next_open'] - 1.0

eligible = (df['bars'] >= MIN_HISTORY) & (df['turn20'] >= MIN_TURNOVER) & df['next_open'].notna()
df = df[eligible].copy()
print(f'  eligible rows: {len(df):,} · {df.symbol.nunique()} symbols', flush=True)


def stats(x: pd.Series) -> tuple:
    x = x.dropna()
    if len(x) < 30:
        return len(x), np.nan, np.nan
    return len(x), x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


rows = []
for n in LOOKBACKS:
    sig_mask = df[f'brk{n}']
    print(f'\n=== {n}-day breakout: {sig_mask.sum():,} signals ===', flush=True)

    for h in HORIZONS:
        col = f'fwd{h}'
        sig = df.loc[sig_mask, col]
        universe = df[col]

        # date-matched random control: for each signal, draw a random eligible row
        # from the SAME calendar day. Same days, same pool, no signal.
        counts = df.loc[sig_mask].groupby('d').size()
        pool = df[[col, 'd']].dropna()
        draws = []
        for day, k in counts.items():
            same_day = pool.loc[pool['d'] == day, col].to_numpy()
            if len(same_day):
                draws.append(rng.choice(same_day, size=min(k, len(same_day)), replace=False))
        ctrl = pd.Series(np.concatenate(draws)) if draws else pd.Series(dtype=float)

        n_s, m_s, t_s = stats(sig)
        n_c, m_c, t_c = stats(ctrl)
        n_u, m_u, _ = stats(universe)

        # the number that matters: signal minus its date-matched control
        excess = m_s - m_c
        se = np.sqrt(sig.var(ddof=1) / n_s + ctrl.var(ddof=1) / n_c) if n_c else np.nan
        t_ex = excess / se if se and se == se else np.nan

        rows.append(dict(lookback=n, horizon=h, n_signals=n_s,
                         signal_pct=100 * m_s, control_pct=100 * m_c, drift_pct=100 * m_u,
                         excess_pct=100 * excess, t_signal=t_s, t_excess=t_ex))
        print(f'  {h:>2}d  signal {100*m_s:+6.3f}%  random {100*m_c:+6.3f}%  '
              f'drift {100*m_u:+6.3f}%  |  excess {100*excess:+6.3f}%  '
              f't_excess {t_ex:5.2f}   (n={n_s:,})', flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / 'g1_probe.csv', index=False)

print('\n=== NET OF COSTS (excess, round-trip bps) ===', flush=True)
print(f"{'lookback':>9} {'horizon':>8} " + ''.join(f'{b:>9}bps' for b in (0, 20, 30, 50)))
for _, r in res.iterrows():
    line = f"{int(r.lookback):>9} {int(r.horizon):>8} "
    for b in (0, 20, 30, 50):
        line += f'{r.excess_pct - b / 100.0:>+11.3f}'
    print(line, flush=True)

print('\n=== PER-YEAR excess, 20d breakout / 5d horizon ===', flush=True)
sub = df[df['brk20']].copy()
sub['yr'] = sub['d'].dt.year
for yr, grp in sub.groupby('yr'):
    s = grp['fwd5'].dropna()
    same = df[df['d'].dt.year == yr]['fwd5'].dropna()
    if len(s) > 30:
        print(f'  {yr}  n={len(s):>6,}  signal {100*s.mean():+6.3f}%  '
              f'universe {100*same.mean():+6.3f}%  excess {100*(s.mean()-same.mean()):+6.3f}%',
              flush=True)

print(f'\nwrote {OUT / "g1_probe.csv"}')
