"""G1b: does buying WEAKNESS beat a date-matched random entry?

G1 found buying new highs loses to random by a steady -0.10 to -0.15% over
2-10 days, t -2.9 to -7.2, across 231k signals. A negative that consistent is
not noise - it says the short-horizon effect here is reversion. This inverts
the signal to measure it head-on, with the identical universe, controls and
next-open entry so the two runs are directly comparable.

Original design notes follow.
G1: does a universal breakout signal beat a date-matched random entry?

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
    # lowest close of the trailing n days INCLUDING today -> today is a new n-day low
    df[f'lo{n}'] = g['close'].transform(lambda s, n=n: s.rolling(n, min_periods=n).min())
    df[f'brk{n}'] = (df['close'] <= df[f'lo{n}']) & df[f'lo{n}'].notna()


def _rsi2(s):
    d = s.diff()
    up = d.clip(lower=0).rolling(2).mean()
    dn = (-d.clip(upper=0)).rolling(2).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# A second, independent reversion read so the verdict does not rest on one
# construction: 2-day RSI, the classic short-horizon oversold measure.
df['rsi2'] = g['close'].transform(_rsi2)
df['brk_rsi'] = df['rsi2'] < 10

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

# ── RSI(2) < 10 arm, identical controls ────────────────────────────────────
print("")
print("=== RSI(2) < 10 (oversold): {:,} signals ===".format(int(df['brk_rsi'].sum())), flush=True)
for h in HORIZONS:
    col = 'fwd{}'.format(h)
    sig = df.loc[df['brk_rsi'], col]
    counts = df.loc[df['brk_rsi']].groupby('d').size()
    pool = df[[col, 'd']].dropna()
    draws = []
    for day, k in counts.items():
        same_day = pool.loc[pool['d'] == day, col].to_numpy()
        if len(same_day):
            draws.append(rng.choice(same_day, size=min(k, len(same_day)), replace=False))
    ctrl = pd.Series(np.concatenate(draws)) if draws else pd.Series(dtype=float)
    n_s, m_s, t_s = stats(sig)
    n_c, m_c, _ = stats(ctrl)
    excess = m_s - m_c
    se = np.sqrt(sig.var(ddof=1) / n_s + ctrl.var(ddof=1) / n_c) if n_c else np.nan
    t_ex = excess / se if se and se == se else np.nan
    rows.append(dict(lookback='rsi2<10', horizon=h, n_signals=n_s,
                     signal_pct=100 * m_s, control_pct=100 * m_c, drift_pct=np.nan,
                     excess_pct=100 * excess, t_signal=t_s, t_excess=t_ex))
    print('  {:>2}d  signal {:+6.3f}%  random {:+6.3f}%  |  excess {:+6.3f}%  t_excess {:5.2f}   (n={:,})'
          .format(h, 100 * m_s, 100 * m_c, 100 * excess, t_ex, n_s), flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / 'g1b_reversion.csv', index=False)

print('\n=== NET OF COSTS (excess, round-trip bps) ===', flush=True)
print(f"{'lookback':>9} {'horizon':>8} " + ''.join(f'{b:>9}bps' for b in (0, 20, 30, 50)))
for _, r in res.iterrows():
    line = f"{int(r.lookback):>9} {int(r.horizon):>8} "
    for b in (0, 20, 30, 50):
        line += f'{r.excess_pct - b / 100.0:>+11.3f}'
    print(line, flush=True)

print('\n=== PER-YEAR excess, 20d LOW / 5d horizon ===', flush=True)
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
