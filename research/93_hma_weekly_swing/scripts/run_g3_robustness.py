"""
G3 robustness (research/93): is the G1 setup-vs-control edge real?
- regenerates control trades (same per-symbol seeds) and saves them
- Welch t-test on setup-vs-control net expectancy
- per-year setup vs control table
- OOS split (entries <=2018 vs >=2019)
- super-winner guard (drop top-3 symbols by total net)
- exit-reason breakdown + cost sensitivity
Fast (~2 min), reads results/g1_trades.csv + the DB.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
import numpy as np
from collections import defaultdict
from hma_weekly_engine import load_daily, to_weekly, add_indicators, backtest, random_control

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', 'results'))
TRADES = os.path.join(RESULTS, 'g1_trades.csv')
CTRL_OUT = os.path.join(RESULTS, 'g1_control_trades.csv')
PARAMS = dict(neg_bars=8, recent_win=4, target_lb=26, max_hold=52, cost_bps=25.0)

setup = []
with open(TRADES) as f:
    for r in csv.DictReader(f):
        setup.append(r)
syms = sorted({r['symbol'] for r in setup})
print(f'{len(setup)} setup trades across {len(syms)} symbols', flush=True)

# ---- regenerate + save control trades (same seed rule as G1)
ctrl = []
if os.path.exists(CTRL_OUT):
    with open(CTRL_OUT) as f:
        ctrl = list(csv.DictReader(f))
    print(f'loaded {len(ctrl)} cached control trades', flush=True)
else:
    fields = ['symbol', 'entry_date', 'exit_date', 'reason', 'hold_wks',
              'gross_pct', 'net_pct', 'R']
    with open(CTRL_OUT, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    for i, sym in enumerate(syms, 1):
        df = load_daily(sym)
        w = add_indicators(to_weekly(df))
        tr = backtest(w, **PARAMS)
        ct = random_control(w, tr, seed=hash(sym) % (2**31),
                            target_lb=PARAMS['target_lb'], max_hold=PARAMS['max_hold'],
                            cost_bps=PARAMS['cost_bps'])
        with open(CTRL_OUT, 'a', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=fields)
            for t in ct:
                wtr.writerow({'symbol': sym, **{k: t[k] for k in fields if k != 'symbol'}})
        ctrl.extend({'symbol': sym, **{k: str(t[k]) for k in fields if k != 'symbol'}} for t in ct)
        if i % 100 == 0:
            print(f'  control regen {i}/{len(syms)}', flush=True)
    print(f'{len(ctrl)} control trades saved', flush=True)

s_net = np.array([float(r['net_pct']) for r in setup])
c_net = np.array([float(r['net_pct']) for r in ctrl])

def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb)

print('\n=== SETUP vs CONTROL (net, 25bps) ===', flush=True)
print(f'setup  : n={len(s_net)} mean={s_net.mean():.3f}% med={np.median(s_net):.3f}%')
print(f'control: n={len(c_net)} mean={c_net.mean():.3f}% med={np.median(c_net):.3f}%')
print(f'diff   : {s_net.mean()-c_net.mean():+.3f}%/trade  Welch t = {welch(s_net, c_net):.2f}')

# ---- per-year
print('\n=== PER-YEAR (mean net %/trade) ===', flush=True)
sy, cy = defaultdict(list), defaultdict(list)
for r in setup:
    sy[r['entry_date'][:4]].append(float(r['net_pct']))
for r in ctrl:
    cy[r['entry_date'][:4]].append(float(r['net_pct']))
print(f'{"year":6s}{"n_set":>6s}{"setup":>9s}{"n_ctl":>7s}{"control":>9s}{"diff":>8s}')
for y in sorted(set(sy) | set(cy)):
    a, b = sy.get(y, []), cy.get(y, [])
    d = (np.mean(a) - np.mean(b)) if a and b else float('nan')
    print(f'{y:6s}{len(a):6d}{(np.mean(a) if a else float("nan")):9.2f}{len(b):7d}'
          f'{(np.mean(b) if b else float("nan")):9.2f}{d:8.2f}', flush=True)

# ---- OOS split
print('\n=== SUB-PERIOD SPLIT ===', flush=True)
for lab, lo, hi in (('<=2018', '0000', '2018'), ('>=2019', '2019', '9999')):
    a = np.array([float(r['net_pct']) for r in setup if lo <= r['entry_date'][:4] <= hi])
    b = np.array([float(r['net_pct']) for r in ctrl if lo <= r['entry_date'][:4] <= hi])
    if len(a) > 5 and len(b) > 5:
        print(f'{lab}: setup n={len(a)} {a.mean():.3f}% | control n={len(b)} {b.mean():.3f}% '
              f'| diff {a.mean()-b.mean():+.3f}% | Welch t {welch(a, b):.2f}', flush=True)

# ---- super-winner guard
tot = defaultdict(float)
for r in setup:
    tot[r['symbol']] += float(r['net_pct'])
top3 = {s for s, _ in sorted(tot.items(), key=lambda kv: -kv[1])[:3]}
a = np.array([float(r['net_pct']) for r in setup if r['symbol'] not in top3])
print(f'\n=== SUPER-WINNER GUARD === drop {sorted(top3)}')
print(f'ex-top3: n={len(a)} mean={a.mean():.3f}% (vs {s_net.mean():.3f}%) '
      f'| vs control diff {a.mean()-c_net.mean():+.3f}%', flush=True)

# ---- exit reasons + cost sensitivity
from collections import Counter
print('\nexit reasons (setup):', dict(Counter(r['reason'] for r in setup)))
g = np.array([float(r['gross_pct']) for r in setup])
print('cost sensitivity (net %/trade): ' +
      '  '.join(f'{c}bps={g.mean()-c/100:.3f}%' for c in (0, 10, 25, 50)), flush=True)
