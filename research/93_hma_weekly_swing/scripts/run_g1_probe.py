"""
G1 cheap probe (research/93): does the weekly HMA30/44 + MACD(21,39,9) + RSI(9/3/21W)
triple-aligned retracement entry have ANY edge on the full daily universe — and does it
beat a same-symbol, year-matched RANDOM-ENTRY control with identical exits?
Incremental CSV, resumable, pooled t-stats. Kill if gross<=0 or setup<=control.
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
import numpy as np
from hma_weekly_engine import list_universe, run_symbol, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', 'results'))
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, 'g1_probe.csv')
TRADES_OUT = os.path.join(RESULTS, 'g1_trades.csv')

SUMFIELDS = ['n', 'gross_exp_pct', 'net_exp_pct', 'net_exp_R', 'win_pct', 'pf',
             't_stat', 'avg_hold_wks', 'total_net_pct']
FIELDS = ['symbol', 'kind'] + SUMFIELDS          # kind = setup | control
TRADEFIELDS = ['symbol', 'entry_date', 'exit_date', 'entry', 'exit', 'stop', 'target',
               'reason', 'hold_wks', 'gross_pct', 'net_pct', 'risk_pct', 'R']

PARAMS = dict(neg_bars=8, recent_win=4, target_lb=26, max_hold=52, cost_bps=25.0)

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {r['symbol'] for r in csv.DictReader(f)}
    print(f'resume: {len(done)} symbols already done', flush=True)
else:
    with open(OUT, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    with open(TRADES_OUT, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=TRADEFIELDS).writeheader()

syms, screens = list_universe()
print(f'G1 probe: {len(syms)} candidate symbols | params {PARAMS} | screens {screens}',
      flush=True)

t0 = time.time()
skipped = {}
n_done = 0
for k, sym in enumerate(syms, 1):
    if sym in done:
        continue
    try:
        trades, s, ctrl = run_symbol(sym, seed=hash(sym) % (2**31), **PARAMS)
    except Exception as e:
        print(f'[{k}/{len(syms)}] {sym:15s} ERROR {e}', flush=True)
        continue
    if trades is None:
        skipped[s] = skipped.get(s, 0) + 1
        continue
    with open(OUT, 'a', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=FIELDS)
        wtr.writerow({'symbol': sym, 'kind': 'setup', **{f_: s.get(f_) for f_ in SUMFIELDS}})
        wtr.writerow({'symbol': sym, 'kind': 'control', **{f_: ctrl.get(f_) for f_ in SUMFIELDS}})
    if trades:
        with open(TRADES_OUT, 'a', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=TRADEFIELDS)
            for t in trades:
                wtr.writerow({'symbol': sym, **t})
    n_done += 1
    if n_done % 50 == 0:
        el = time.time() - t0
        print(f'[{k}/{len(syms)}] {n_done} tested, {el/60:.1f} min, '
              f'{el/max(1,n_done):.1f}s/sym', flush=True)

print(f'\nskipped: {skipped}', flush=True)

# ---- pooled verdict from the trade file (setup) + per-symbol control aggregation
print('\n=== POOLED G1 VERDICT ===', flush=True)
setup_trades = []
with open(TRADES_OUT) as f:
    for r in csv.DictReader(f):
        setup_trades.append(dict(net_pct=float(r['net_pct']), gross_pct=float(r['gross_pct']),
                                 hold_wks=int(r['hold_wks']),
                                 R=float(r['R']) if r['R'] not in ('', 'None', None) else None))
print('SETUP  :', summarize(setup_trades), flush=True)

# control pooled: weighted by n from per-symbol rows
ns, gs, nets = [], [], []
with open(OUT) as f:
    for r in csv.DictReader(f):
        if r['kind'] == 'control' and r['n'] not in ('0', '', None):
            n = int(r['n'])
            if n > 0 and r['gross_exp_pct'] not in ('', 'None'):
                ns.append(n); gs.append(float(r['gross_exp_pct'])); nets.append(float(r['net_exp_pct']))
ns = np.array(ns); gs = np.array(gs); nets = np.array(nets)
if ns.sum() > 0:
    print(f'CONTROL: n={int(ns.sum())} gross_exp={np.average(gs, weights=ns):.4f}% '
          f'net_exp={np.average(nets, weights=ns):.4f}%', flush=True)
print('\nGATE: setup gross>0 AND net(25bps)>0 AND setup>control. See STATUS-MD.', flush=True)
