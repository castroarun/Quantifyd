"""
Phase-2 stage 2 (research/93): portfolio replay variants — can the book beat NIFTYBEES?
All replays start 2005-01-07 (benchmark comparability). Trades from g6_trades_<mode>.csv.

Axes (named cells, not a grid):
  exit mode : target | trail_hma44 | trail_donch10
  gate      : NEW entries only when NIFTYBEES weekly close > 40wk SMA
  rank      : slot contention priority — alpha | rr (reward:risk desc; trails: 1/risk)
  slots     : 20 x 5% | 40 x 2.5%

Same-week-exit trades realize at entry (zombie bug fix). Costs 25bps round-trip split.
Prints CAGR/Sharpe/MaxDD/Calmar/exposure per cell + per-year for the best Calmar cell.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
import numpy as np
import pandas as pd
from collections import defaultdict
from hma_weekly_engine import load_daily, to_weekly

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', 'results'))
COST_BPS = 25.0
START = '2005-01-07'

def load_trades(mode):
    out = []
    with open(os.path.join(RESULTS, f'g6_trades_{mode}.csv')) as f:
        for r in csv.DictReader(f):
            if r['entry_date'] < START:
                continue
            out.append(dict(symbol=r['symbol'], entry_date=r['entry_date'],
                            exit_date=r['exit_date'], entry=float(r['entry']),
                            exit=float(r['exit']), stop=float(r['stop']),
                            target=float(r['target']), risk=float(r['risk_pct'])))
    return out

MODES = ('target', 'trail_hma44', 'trail_donch10')
trades_by_mode = {m: load_trades(m) for m in MODES}
all_syms = sorted({t['symbol'] for m in MODES for t in trades_by_mode[m]})
print(f'trades: ' + ', '.join(f'{m}={len(trades_by_mode[m])}' for m in MODES), flush=True)

closes = {}
for i, s in enumerate(all_syms, 1):
    w = to_weekly(load_daily(s))
    closes[s] = w.set_index(pd.to_datetime(w['date']))['close']
    if i % 150 == 0:
        print(f'  closes {i}/{len(all_syms)}', flush=True)

bw = to_weekly(load_daily('NIFTYBEES'))
bser = bw.set_index(pd.to_datetime(bw['date']))['close']
bsma = bser.rolling(40).mean()

weeks = pd.date_range(START, '2026-07-24', freq='W-FRI')

def bench_at(wk):
    i = bser.index.searchsorted(wk, side='right') - 1
    return float(bser.iloc[i]) if i >= 0 else np.nan

def gate_on(wk):
    i = bser.index.searchsorted(wk, side='right') - 1
    if i < 0 or np.isnan(bsma.iloc[i]):
        return True
    return bser.iloc[i] > bsma.iloc[i]

def replay(trades, slots, weight, rank, gate):
    by_entry = defaultdict(list)
    for t in trades:
        by_entry[t['entry_date']].append(t)
    nav = 1.0; positions = {}; rows = []
    taken = skipped = 0
    for wk in weeks:
        d = wk.strftime('%Y-%m-%d')
        for s in [s for s, p in positions.items() if p['t']['exit_date'] == d]:
            p = positions.pop(s)
            nav += p['units'] * (p['t']['exit'] * (1 - COST_BPS / 2e4) - p['mark'])
        for s, p in positions.items():
            ser = closes[s]
            i = ser.index.searchsorted(wk, side='right') - 1
            if i >= 0:
                px = float(ser.iloc[i])
                nav += p['units'] * (px - p['mark']); p['mark'] = px
        cands = by_entry.get(d, [])
        if cands and gate and not gate_on(wk):
            skipped += len(cands); cands = []
        if rank == 'rr':
            cands = sorted(cands, key=lambda t: -((t['target'] - t['entry']) /
                           max(1e-9, t['entry'] - t['stop'])) if t['target'] > t['entry']
                           else -1.0 / max(0.1, t['risk']))
        else:
            cands = sorted(cands, key=lambda t: t['symbol'])
        for t in cands:
            if t['symbol'] in positions:
                continue
            if len(positions) >= slots:
                skipped += 1; continue
            units = nav * weight / (t['entry'] * (1 + COST_BPS / 2e4))
            taken += 1
            if t['exit_date'] == d:
                nav += units * (t['exit'] * (1 - COST_BPS / 2e4) - t['entry'])
                continue
            positions[t['symbol']] = dict(units=units, mark=t['entry'], t=t)
        rows.append((wk, nav, len(positions)))
    df = pd.DataFrame(rows, columns=['date', 'nav', 'n_pos']).set_index('date')
    return df, taken, skipped

def metrics(s):
    s = s.dropna() / s.dropna().iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = s.iloc[-1] ** (1 / yrs) - 1
    ret = s.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(52) if ret.std() > 0 else 0
    dd = (s / s.cummax() - 1).min()
    return cagr * 100, sharpe, dd * 100, (cagr / abs(dd)) if dd else float('inf')

CELLS = [
    ('target', 20, 0.05, 'alpha', False),
    ('target', 20, 0.05, 'rr',    False),
    ('target', 40, 0.025, 'alpha', False),
    ('target', 20, 0.05, 'alpha', True),
    ('target', 20, 0.05, 'rr',    True),
    ('target', 40, 0.025, 'rr',   True),
    ('trail_hma44', 20, 0.05, 'alpha', False),
    ('trail_hma44', 40, 0.025, 'alpha', False),
    ('trail_hma44', 20, 0.05, 'rr',    True),
    ('trail_hma44', 40, 0.025, 'rr',   True),
    ('trail_donch10', 20, 0.05, 'alpha', False),
    ('trail_donch10', 40, 0.025, 'alpha', False),
    ('trail_donch10', 20, 0.05, 'rr',    True),
    ('trail_donch10', 40, 0.025, 'rr',   True),
]

bench = pd.Series([bench_at(w) for w in weeks], index=weeks)
bc, bs, bd, bcal = metrics(bench)
print(f'\nBENCH NIFTYBEES B&H: CAGR {bc:6.2f}% Sharpe {bs:4.2f} MaxDD {bd:6.1f}% Calmar {bcal:4.2f}\n', flush=True)

best = None
print(f'{"mode":14s} {"slots":>5s} {"rank":>5s} {"gate":>4s} | {"CAGR":>7s} {"Sharpe":>6s} {"MaxDD":>7s} {"Calmar":>6s} {"expo":>5s} {"taken":>5s}', flush=True)
navs = {}
for mode, slots, wt, rank, gate in CELLS:
    df, taken, skipped = replay(trades_by_mode[mode], slots, wt, rank, gate)
    c, sh, dd, cal = metrics(df['nav'])
    expo = df['n_pos'].mean() * wt * 100
    key = f'{mode}|{slots}|{rank}|{"G" if gate else "-"}'
    navs[key] = df
    print(f'{mode:14s} {slots:5d} {rank:>5s} {"ON" if gate else "off":>4s} | {c:6.2f}% {sh:6.2f} {dd:6.1f}% {cal:6.2f} {expo:4.0f}% {taken:5d}', flush=True)
    if best is None or cal > best[1]:
        best = (key, cal, df)

print(f'\nBEST by Calmar: {best[0]}', flush=True)
df = best[2]
strat_y = df['nav'].resample('YE').last().pct_change().dropna() * 100
bench_y = bench.resample('YE').last().pct_change().dropna() * 100
print(f'{"year":6s}{"strat":>8s}{"bench":>8s}{"excess":>8s}', flush=True)
for y in strat_y.index:
    b = bench_y.get(y, np.nan)
    print(f'{y.year:<6d}{strat_y[y]:8.1f}{b:8.1f}{strat_y[y]-b:8.1f}', flush=True)
df.to_csv(os.path.join(RESULTS, 'g7_best_nav.csv'))
print('\nbest NAV saved: results/g7_best_nav.csv', flush=True)
