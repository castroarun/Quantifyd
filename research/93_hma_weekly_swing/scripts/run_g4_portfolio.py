"""
G4 portfolio sim (research/93): turn the base-cell signal (8/4/26) into a book.
- 20 slots, 5% of current NAV per entry (fixed-fractional), long-only, no leverage
- entries/exits straight from results/g1_trades.csv (per-symbol non-overlapping)
- weekly mark-to-market from DB weekly closes; slot contention = first-by-symbol
- benchmark: NIFTYBEES buy-and-hold on the overlapping window
Outputs results/g4_nav.csv (date, nav, bench, n_pos) + headline metrics.
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
TRADES = os.path.join(RESULTS, 'g1_trades.csv')
NAV_OUT = os.path.join(RESULTS, 'g4_nav.csv')

SLOTS = 20
WEIGHT = 0.05
COST_BPS = 25.0
CASH_IN_BENCH = os.environ.get('CASH_IN_BENCH', '0') == '1'   # idle cash tracks NIFTYBEES

trades = []
with open(TRADES) as f:
    for r in csv.DictReader(f):
        trades.append(dict(symbol=r['symbol'], entry_date=r['entry_date'],
                           exit_date=r['exit_date'], entry=float(r['entry']),
                           exit=float(r['exit'])))
print(f'{len(trades)} candidate trades', flush=True)

syms = sorted({t['symbol'] for t in trades})
closes = {}
for i, s in enumerate(syms, 1):
    w = to_weekly(load_daily(s))
    closes[s] = w.set_index(pd.to_datetime(w['date']))['close']
    if i % 100 == 0:
        print(f'  loaded weekly closes {i}/{len(syms)}', flush=True)

by_entry = defaultdict(list)
for t in trades:
    by_entry[t['entry_date']].append(t)

bench_ser = None
if CASH_IN_BENCH:
    bw0 = to_weekly(load_daily('NIFTYBEES'))
    bench_ser = bw0.set_index(pd.to_datetime(bw0['date']))['close']

weeks = pd.date_range('2001-01-05', '2026-07-24', freq='W-FRI')
nav = 1.0
positions = {}          # symbol -> dict(shares_value_units, trade)
rows = []
n_taken = n_skipped = 0
prev_bpx = None
for wk in weeks:
    d = wk.strftime('%Y-%m-%d')
    # 0) idle cash tracks the index (variant)
    if CASH_IN_BENCH and bench_ser is not None:
        bi = bench_ser.index.searchsorted(wk, side='right') - 1
        bpx = float(bench_ser.iloc[bi]) if bi >= 0 else None
        if bpx is not None and prev_bpx is not None:
            cash = nav - sum(p['units'] * p['mark'] for p in positions.values())
            nav += cash * (bpx / prev_bpx - 1)
        prev_bpx = bpx
    # 1) exits at this week's known exit price
    for s in [s for s, p in positions.items() if p['t']['exit_date'] == d]:
        p = positions.pop(s)
        nav += p['units'] * (p['t']['exit'] * (1 - COST_BPS / 2e4) - p['mark'])
    # 2) mark to market
    for s, p in positions.items():
        ser = closes[s]
        idx = ser.index.searchsorted(wk, side='right') - 1
        if idx >= 0:
            px = float(ser.iloc[idx])
            nav += p['units'] * (px - p['mark'])
            p['mark'] = px
    # 3) entries dated this week
    for t in sorted(by_entry.get(d, []), key=lambda x: x['symbol']):
        if t['symbol'] in positions:
            continue
        if len(positions) >= SLOTS:
            n_skipped += 1
            continue
        alloc = nav * WEIGHT
        units = alloc / (t['entry'] * (1 + COST_BPS / 2e4))
        n_taken += 1
        if t['exit_date'] == d:      # same-week exit: realize now, never occupy a slot
            nav += units * (t['exit'] * (1 - COST_BPS / 2e4) - t['entry'])
            continue
        positions[t['symbol']] = dict(units=units, mark=t['entry'], t=t)
    rows.append((d, nav, len(positions)))

df = pd.DataFrame(rows, columns=['date', 'nav', 'n_pos'])
df['date'] = pd.to_datetime(df['date'])

# benchmark: NIFTYBEES B&H
bw = to_weekly(load_daily('NIFTYBEES'))
bser = bw.set_index(pd.to_datetime(bw['date']))['close']
df['bench'] = [float(bser.iloc[bser.index.searchsorted(d, side='right') - 1])
               if bser.index.searchsorted(d, side='right') > 0 else np.nan
               for d in df['date']]
df.to_csv(NAV_OUT, index=False)

def metrics(series, label):
    s = series.dropna()
    s = s / s.iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = s.iloc[-1] ** (1 / yrs) - 1
    ret = s.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(52) if ret.std() > 0 else 0
    dd = (s / s.cummax() - 1).min()
    print(f'{label:22s} CAGR {cagr*100:6.2f}%  Sharpe {sharpe:5.2f}  '
          f'MaxDD {dd*100:6.1f}%  Calmar {(cagr/abs(dd)) if dd else float("inf"):5.2f}',
          flush=True)

print(f'\ntaken {n_taken} trades, skipped (slots full) {n_skipped}', flush=True)
print(f'avg positions {df["n_pos"].mean():.1f} / {SLOTS}', flush=True)
d2 = df.set_index('date')
print('\n=== FULL PERIOD (2001-2026, strategy only) ===', flush=True)
metrics(d2['nav'], 'HMA weekly book')
mask = d2['bench'].notna()
print(f'\n=== OVERLAP WITH NIFTYBEES ({d2.index[mask][0].date()} on) ===', flush=True)
metrics(d2.loc[mask, 'nav'], 'HMA weekly book')
metrics(d2.loc[mask, 'bench'], 'NIFTYBEES B&H')
# modern sub-period (survivorship-lighter)
recent = d2.loc[d2.index >= '2019-01-01']
print('\n=== 2019+ ===', flush=True)
metrics(recent['nav'], 'HMA weekly book')
metrics(recent['bench'].dropna(), 'NIFTYBEES B&H')
