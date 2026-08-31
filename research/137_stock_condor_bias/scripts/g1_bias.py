"""G1: does any trend cue shrink the tail a sold option spread cares about?

A condor does not need to be right about direction often. It needs the spread it
sold to expire out of the money. So the number that decides this idea is not
mean forward return — it is whether a bullish signal lowers the probability of a
large FALL, and its bearish mirror lowers the probability of a large RISE.

research/129 asked a version of this of MA, EMA, RSI and stochastic states and
found nothing: no regime state shrank the sold tail, and bear states carried
higher forward drift. SuperTrend is the fifth member of that family and gets the
same test, with its parameters varied and weekly bars included, because a single
favourite setting proving nothing would prove nothing about the family.

Everything is causal: indicators use only closed bars, the forward window starts
the day after the signal, and weekly signals are mapped to daily rows by
forward-fill so a week's state is never known before that week closes.

Read-only.
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
sys.path.insert(0, str(ROOT))   # run from anywhere; services/ lives at the repo root
OUT = ROOT / 'research' / '137_stock_condor_bias' / 'results'
OUT.mkdir(parents=True, exist_ok=True)

START = '2015-01-01'
HORIZON = 21          # one monthly option cycle
FALL_5, FALL_10 = -0.05, -0.10

print('loading F&O universe...', flush=True)
from services.data_manager import FNO_LOT_SIZES

conn = sqlite3.connect(f"file:{ROOT / 'backtest_data' / 'market_data.db'}?mode=ro", uri=True)
syms = tuple(sorted(FNO_LOT_SIZES))
q = ("SELECT symbol, substr(date,1,10) d, open, high, low, close FROM market_data_unified "
     f"WHERE timeframe='day' AND date >= ? AND symbol IN ({','.join('?' * len(syms))}) "
     "ORDER BY symbol, date")
df = pd.read_sql_query(q, conn, params=(START, *syms))
conn.close()

df['d'] = pd.to_datetime(df['d'])
df = df.sort_values(['symbol', 'd']).reset_index(drop=True)
keep = df.groupby('symbol')['close'].transform('size') >= 1000
df = df[keep].copy()
print(f'  {len(df):,} rows | {df.symbol.nunique()} symbols | {df.d.min():%Y-%m-%d} -> {df.d.max():%Y-%m-%d}\n',
      flush=True)


# ── indicators, all trailing ───────────────────────────────────────────────
def atr(g, n):
    h, l, c = g['high'], g['low'], g['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def supertrend_dir(g, period, mult):
    """+1 while price holds above the trailing stop, -1 below. Classic ratchet."""
    a = atr(g, period)
    hl2 = (g['high'] + g['low']) / 2
    upper, lower = hl2 + mult * a, hl2 - mult * a
    c = g['close'].to_numpy()
    up, lo = upper.to_numpy(), lower.to_numpy()
    n = len(c)
    fu, fl = np.full(n, np.nan), np.full(n, np.nan)
    dirn = np.zeros(n)
    for i in range(n):
        if np.isnan(up[i]):
            continue
        if i == 0 or np.isnan(fu[i - 1]):
            fu[i], fl[i], dirn[i] = up[i], lo[i], 1
            continue
        fu[i] = up[i] if (up[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lo[i] if (lo[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if dirn[i - 1] == 1:
            dirn[i] = -1 if c[i] < fl[i] else 1
        else:
            dirn[i] = 1 if c[i] > fu[i] else -1
    return pd.Series(dirn, index=g.index)


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


frames = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.copy()
    c = g['close']

    for per, mult in ((7, 3), (10, 3), (14, 3), (10, 2), (21, 5)):
        g[f'st_{per}_{mult}'] = supertrend_dir(g, per, mult)

    g['ema20'], g['ema50'] = c.ewm(span=20).mean(), c.ewm(span=50).mean()
    g['ema200'] = c.ewm(span=200).mean()
    g['ema_20_50'] = np.where(g.ema20 > g.ema50, 1, -1)
    g['ema_50_200'] = np.where(g.ema50 > g.ema200, 1, -1)
    g['sma50'] = c.rolling(50, min_periods=50).mean()
    g['sma200'] = c.rolling(200, min_periods=200).mean()
    g['px_sma50'] = np.where(c > g.sma50, 1, -1)
    g['px_sma200'] = np.where(c > g.sma200, 1, -1)
    g['rsi50'] = np.where(rsi(c) > 50, 1, -1)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    g['macd'] = np.where(macd - macd.ewm(span=9).mean() > 0, 1, -1)
    hi55 = c.rolling(55, min_periods=55).max()
    lo55 = c.rolling(55, min_periods=55).min()
    pos = (c - lo55) / (hi55 - lo55)
    g['donch55'] = np.where(pos > 0.75, 1, np.where(pos < 0.25, -1, 0))

    # weekly bars -> weekly SuperTrend -> mapped back to daily by forward fill,
    # so a week's state is only ever known once that week has closed
    w = g.set_index('d').resample('W-FRI').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(w) > 60:
        for per, mult in ((7, 3), (10, 3)):
            wd = supertrend_dir(w.reset_index(), per, mult)
            wd.index = w.index
            g[f'wk_st_{per}_{mult}'] = wd.reindex(g['d'], method='ffill').to_numpy()

    g['fwd'] = c.shift(-(HORIZON + 1)) / c.shift(-1) - 1.0   # enter next day, hold 21
    frames.append(g)

df = pd.concat(frames, ignore_index=True)
df = df[df['fwd'].notna()].copy()
print(f'  {len(df):,} rows with a complete {HORIZON}-day forward window\n', flush=True)

SIGNALS = [c for c in df.columns
           if c.startswith(('st_', 'wk_st_', 'ema_', 'px_', 'rsi', 'macd', 'donch'))
           and c not in ('rsi50_raw',)]

uncond = df['fwd']
u_mean = uncond.mean()
u_f5 = (uncond <= FALL_5).mean()
u_f10 = (uncond <= FALL_10).mean()
u_r5 = (uncond >= 0.05).mean()

print('=' * 104)
print(f'Unconditional over {len(uncond):,} stock-days: mean {100*u_mean:+.2f}%  '
      f'P(fall>=5%) {100*u_f5:.1f}%  P(fall>=10%) {100*u_f10:.1f}%  P(rise>=5%) {100*u_r5:.1f}%')
print('=' * 104)
print(f"{'signal':16} {'n bull':>8} {'mean':>7} {'vs unc':>8} {'P(-5%)':>8} {'delta':>7} "
      f"{'P(-10%)':>8} {'delta':>7} {'t':>6}   {'n bear':>8} {'P(+5%)':>8} {'delta':>7}")
print('-' * 104)

rows = []
for sig in SIGNALS:
    bull = df.loc[df[sig] == 1, 'fwd'].dropna()
    bear = df.loc[df[sig] == -1, 'fwd'].dropna()
    if len(bull) < 2000 or len(bear) < 2000:
        continue

    b_mean, b_f5, b_f10 = bull.mean(), (bull <= FALL_5).mean(), (bull <= FALL_10).mean()
    r_r5 = (bear >= 0.05).mean()
    se = np.sqrt(bull.var(ddof=1) / len(bull) + uncond.var(ddof=1) / len(uncond))
    t = (b_mean - u_mean) / se if se else np.nan

    rows.append(dict(signal=sig, n_bull=len(bull), n_bear=len(bear),
                     mean_pct=100 * b_mean, mean_delta=100 * (b_mean - u_mean),
                     p_fall5=100 * b_f5, d_fall5=100 * (b_f5 - u_f5),
                     p_fall10=100 * b_f10, d_fall10=100 * (b_f10 - u_f10),
                     p_rise5_bear=100 * r_r5, d_rise5_bear=100 * (r_r5 - u_r5), t=t))

    print(f'{sig:16} {len(bull):>8,} {100*b_mean:>+6.2f}% {100*(b_mean-u_mean):>+7.2f}% '
          f'{100*b_f5:>7.1f}% {100*(b_f5-u_f5):>+6.1f} {100*b_f10:>7.1f}% '
          f'{100*(b_f10-u_f10):>+6.1f} {t:>6.1f}   {len(bear):>8,} '
          f'{100*r_r5:>7.1f}% {100*(r_r5-u_r5):>+6.1f}', flush=True)

res = pd.DataFrame(rows).sort_values('d_fall5')
res.to_csv(OUT / 'g1_bias.csv', index=False)

print('\n' + '=' * 104)
print('THE CONDOR TEST — bullish signals ranked by how much they cut the adverse tail')
print('(falsification bar set before running: needs -3.0 points or better, and its')
print(' bearish mirror must cut P(rise>=5%) too)')
print('=' * 104)
passed = 0
for _, r in res.iterrows():
    ok = r.d_fall5 <= -3.0 and r.d_rise5_bear <= -3.0
    passed += ok
    print(f"  {r.signal:16} P(fall>=5%) {r.d_fall5:+5.1f} pts   "
          f"bear P(rise>=5%) {r.d_rise5_bear:+5.1f} pts   "
          f"{'PASSES' if ok else ''}")

print(f'\n{passed} of {len(res)} signals clear the bar.')
print(f'wrote {OUT / "g1_bias.csv"}')
