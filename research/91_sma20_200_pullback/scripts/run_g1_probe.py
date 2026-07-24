"""
G1 cheap probe (research/91): does ANY gross edge exist for the 20/200-SMA
retrace-break on the deep 5-min names (2015->now)? Baseline params, both
directions, incremental CSV, pooled t-stat. Kill if gross expectancy <= 0.
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
from sma_pullback_engine import run_symbol, summarize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'g1_probe.csv')
OUT = os.path.abspath(OUT)
FIELDS = ['symbol', 'direction', 'n', 'gross_exp_pct', 'net_exp_pct', 'net_exp_R',
          'win_pct', 'pf', 't_stat', 'avg_hold', 'total_net_pct']

# deep 5-min names with real 2015-era history (tradeable; INDIAVIX excluded)
NAMES = ['NIFTY50', 'BANKNIFTY', 'HDFCBANK', 'ICICIBANK', 'RELIANCE', 'INFY',
         'TCS', 'SBIN', 'ITC', 'HINDUNILVR', 'KOTAKBANK', 'BHARTIARTL']
BASE = dict(slope_lb=3, near_pct=0.005, atr_mult=2.5, entry_win=3,
            hold='intraday', cost_bps=5.0)

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {(r['symbol'], r['direction']) for r in csv.DictReader(f)}
    print(f'resume: {len(done)} cells already done')
else:
    with open(OUT, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

pooled = {'long': [], 'short': []}
tasks = [(s, d) for s in NAMES for d in ('long', 'short')]
print(f'G1 probe: {len(tasks)} cells | baseline {BASE}\n', flush=True)

for k, (sym, d) in enumerate(tasks, 1):
    if (sym, d) in done:
        continue
    t0 = time.time()
    try:
        trades, s = run_symbol(sym, '5minute', direction=d, **BASE)
    except Exception as e:
        print(f'[{k}/{len(tasks)}] {sym:11s} {d:5s} ERROR {e}', flush=True); continue
    pooled[d].extend(trades)
    row = {'symbol': sym, 'direction': d, **{f: s.get(f) for f in FIELDS if f not in ('symbol', 'direction')}}
    with open(OUT, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    print(f'[{k}/{len(tasks)}] {sym:11s} {d:5s} {time.time()-t0:4.0f}s | '
          f'n={s.get("n",0):5d} gross={s.get("gross_exp_pct")}% net={s.get("net_exp_pct")}% '
          f'R={s.get("net_exp_R")} win={s.get("win_pct")}% t={s.get("t_stat")}', flush=True)

print('\n=== POOLED (all deep 5-min names, baseline) ===', flush=True)
for d in ('long', 'short'):
    ps = summarize(pooled[d])
    print(f'{d:5s}: {ps}', flush=True)
print('\nGATE: gross expectancy > 0 to proceed to G2. See STATUS-MD.', flush=True)
