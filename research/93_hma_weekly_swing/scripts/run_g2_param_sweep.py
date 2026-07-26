"""
G2 parameter sweep (research/93): neg_bars {5,8,12} x recent_win {2,4,8} x
target_lb {13,26,52} = 27 cells, full universe, setup AND matched random control
per cell. Symbol-outer loop (one DB load per name). Crash-safe: per-symbol
aggregate rows appended immediately; pooled cell table computed at the end
(and recomputable from the CSV alone).
"""
import os, sys, csv, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
import numpy as np
from collections import defaultdict
from hma_weekly_engine import (list_universe, load_daily, to_weekly, add_indicators,
                               backtest, random_control)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', 'results'))
OUT = os.path.join(RESULTS, 'g2_sweep_rows.csv')
CELLS_OUT = os.path.join(RESULTS, 'g2_sweep_cells.csv')

GRID = list(itertools.product((5, 8, 12), (2, 4, 8), (13, 26, 52)))
COST = 25.0
FIELDS = ['symbol', 'neg_bars', 'recent_win', 'target_lb', 'kind',
          'n', 'sum_net', 'sum_sq', 'sum_gross', 'wins']

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {r['symbol'] for r in csv.DictReader(f)}
    print(f'resume: {len(done)} symbols already swept', flush=True)
else:
    with open(OUT, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

syms, _ = list_universe()
print(f'G2 sweep: {len(GRID)} cells x {len(syms)} candidates', flush=True)

t0 = time.time(); n_done = 0
for k, sym in enumerate(syms, 1):
    if sym in done:
        continue
    try:
        df = load_daily(sym)
        if df.empty:
            continue
        w = to_weekly(df)
        if len(w) < 200 or df['close'].median() < 20 or \
           (df['close'] * df['volume']).median() < 1e7:
            continue
        w = add_indicators(w)
        rows = []
        for nb, rw_, tl in GRID:
            tr = backtest(w, neg_bars=nb, recent_win=rw_, target_lb=tl,
                          max_hold=52, cost_bps=COST)
            ct = random_control(w, tr, seed=hash(sym) % (2**31), target_lb=tl,
                                max_hold=52, cost_bps=COST)
            for kind, trades in (('setup', tr), ('control', ct)):
                if not trades:
                    continue
                net = np.array([t['net_pct'] for t in trades])
                gross = np.array([t['gross_pct'] for t in trades])
                rows.append(dict(symbol=sym, neg_bars=nb, recent_win=rw_, target_lb=tl,
                                 kind=kind, n=len(net), sum_net=round(net.sum(), 4),
                                 sum_sq=round((net ** 2).sum(), 4),
                                 sum_gross=round(gross.sum(), 4),
                                 wins=int((net > 0).sum())))
        with open(OUT, 'a', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=FIELDS)
            for r in rows:
                wtr.writerow(r)
    except Exception as e:
        print(f'[{k}] {sym} ERROR {e}', flush=True)
        continue
    n_done += 1
    if n_done % 50 == 0:
        el = time.time() - t0
        print(f'[{k}/{len(syms)}] {n_done} swept, {el/60:.1f} min', flush=True)

# ---- aggregate pooled per-cell table
agg = defaultdict(lambda: dict(n=0, sn=0.0, ss=0.0, sg=0.0, w=0))
with open(OUT) as f:
    for r in csv.DictReader(f):
        key = (r['neg_bars'], r['recent_win'], r['target_lb'], r['kind'])
        a = agg[key]
        a['n'] += int(r['n']); a['sn'] += float(r['sum_net'])
        a['ss'] += float(r['sum_sq']); a['sg'] += float(r['sum_gross'])
        a['w'] += int(r['wins'])

with open(CELLS_OUT, 'w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(['neg_bars', 'recent_win', 'target_lb', 'n_setup', 'setup_net',
                  'setup_gross', 'setup_win', 'setup_t', 'n_ctrl', 'ctrl_net', 'diff'])
    print('\n=== POOLED CELLS (net %/trade, 25bps) ===', flush=True)
    print('nb rw tl | n_set  set_net set_t | n_ctl ctl_net |  diff', flush=True)
    for nb, rw_, tl in GRID:
        s = agg.get((str(nb), str(rw_), str(tl), 'setup'))
        c = agg.get((str(nb), str(rw_), str(tl), 'control'))
        if not s or s['n'] < 30:
            continue
        m = s['sn'] / s['n']
        var = max(s['ss'] / s['n'] - m ** 2, 1e-12)
        tstat = m / np.sqrt(var / s['n'])
        cm = (c['sn'] / c['n']) if c and c['n'] else float('nan')
        diff = m - cm
        wtr.writerow([nb, rw_, tl, s['n'], round(m, 4), round(s['sg'] / s['n'], 4),
                      round(100 * s['w'] / s['n'], 1), round(tstat, 2),
                      c['n'] if c else 0, round(cm, 4), round(diff, 4)])
        print(f'{nb:2d} {rw_:2d} {tl:2d} | {s["n"]:5d} {m:8.3f} {tstat:5.1f} | '
              f'{(c["n"] if c else 0):5d} {cm:7.3f} | {diff:+6.3f}', flush=True)
print('\ndone. cells table: ' + CELLS_OUT, flush=True)
