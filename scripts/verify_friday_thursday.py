"""Two claims from real_book.py that would drive a live change, checked before I make them.

CLAIM 1 — Friday is the best and safest day in the shadow record, and nobody is
live on it. n=19, net +Rs 286,628, t 2.18, worst only -Rs 10,184. If that net is
one or two days, it is not a finding. So: Friday day by day, and its top-3 share.

CLAIM 2 — a contradiction I have to resolve rather than pick a side of.
  the V1/CSL replay says DTE3 (Thursday) is the BEST cell, t 3.85, zero stops
  the NAS 916 shadow says Thursday is the WORST day, -Rs 139,122, t -0.83
Same days, opposite verdicts. If that is real, it says the day is not the unit —
the CONSTRUCTION is: a held straddle with a wide combined stop earns on Thursday,
while the per-leg-stop-and-trail construction gives it back. That would make
"size up Thursday" true for one book and false for the other, which is a very
different recommendation from either read alone. Checked per construction.

Read-only.
"""
from __future__ import annotations

import json, sqlite3, statistics as st, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
WD2DTE = {'NIFTY': {0: 1, 1: 0, 2: 4, 3: 3, 4: 2}}


def tstat(v):
    return (st.mean(v) / (st.stdev(v) / len(v) ** 0.5)) if len(v) > 2 and st.stdev(v) else 0.0


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


# ── per-construction, per-weekday, NIFTY 916 shadow at 10-lot normalised ───
SYS = [('nas_916_atm', 'nas_916_atm_trading.db'), ('nas_916_atm2', 'nas_916_atm2_trading.db'),
       ('nas_916_atm4', 'nas_916_atm4_trading.db')]
per = {}
for key, db in SYS:
    c = sqlite3.connect(f'file:{ROOT}/backtest_data/{db}?mode=ro', uri=True)
    s = defaultdict(float)
    for d, pnl, lots in c.execute("SELECT trade_date, net_pnl, lots FROM nas_atm_trades "
                                  "WHERE exit_time IS NOT NULL AND net_pnl IS NOT NULL"):
        if d and lots and float(lots) > 0 and date.fromisoformat(d).weekday() < 5:
            s[d] += float(pnl) / float(lots) * 10
    c.close()
    per[key] = dict(s)

print('CLAIM 2 — NIFTY 916 CONSTRUCTIONS BY WEEKDAY (10-lot normalised, live + shadow)')
print(f"  {'system':14}" + ''.join(f'{d:>13}' for d in DOW))
print('  ' + '-' * 79)
for key, s in per.items():
    wd = defaultdict(list)
    for d, v in s.items():
        wd[date.fromisoformat(d).weekday()].append(v)
    print(f'  {key:14}' + ''.join(f'{sum(wd.get(k, [0])):>13,.0f}' for k in range(5)))
pool = defaultdict(float)
for s in per.values():
    for d, v in s.items():
        pool[d] += v
wd = defaultdict(list)
for d, v in pool.items():
    wd[date.fromisoformat(d).weekday()].append(v)
print(f'  {"POOLED":14}' + ''.join(f'{sum(wd.get(k, [0])):>13,.0f}' for k in range(5)))
print(f'  {"n days":14}' + ''.join(f'{len(wd.get(k, [])):>13}' for k in range(5)))
print(f'  {"t":14}' + ''.join(f'{tstat(wd.get(k, [0, 0, 0])):>13.2f}' for k in range(5)))

print('\n  Same weekdays, the V1/CSL held-straddle replay (30% combined stop, 10 lots):')
tr = json.loads((ROOT / 'frontend' / 'public' / 'straddles' / 'v1_sl30.json').read_text())['trades']
rep = defaultdict(list)
for t in tr:
    if t.get('final') is not None:
        rep[date.fromisoformat(t['day'][:10]).weekday()].append(t['final'])
print(f'  {"replay":14}' + ''.join(f'{sum(rep.get(k, [0])):>13,.0f}' for k in range(5)))
print(f'  {"n days":14}' + ''.join(f'{len(rep.get(k, [])):>13}' for k in range(5)))
print(f'  {"t":14}' + ''.join(f'{tstat(rep.get(k, [0, 0, 0])):>13.2f}' for k in range(5)))

print('\n  VERDICT: where the two rows disagree, the day is not the unit — the construction is.')

# ── Friday, day by day ────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('CLAIM 1 — FRIDAY IN THE 916 SHADOW, DAY BY DAY (10-lot normalised, pooled)')
print('=' * 72)
fri = sorted((d, v) for d, v in pool.items() if date.fromisoformat(d).weekday() == 4)
cum = 0
for d, v in fri:
    cum += v
    print(f'  {d}  {v:>11,.0f}   cum {cum:>11,.0f}')
fv = [v for _, v in fri]
t3 = sorted(fv, reverse=True)[:3]
print(f'\n  n {len(fv)} · net {sum(fv):>11,.0f} · mean {sum(fv)/len(fv):>9,.0f} · t {tstat(fv):.2f}')
print(f'  maxDD {dd(fv):>10,.0f} · worst day {min(fv):>10,.0f} · green '
      f'{sum(1 for x in fv if x>0)}/{len(fv)}')
print(f'  top 3 days {sum(t3):>10,.0f} = {100*sum(t3)/sum(fv):.0f}% of the Friday total')
h = len(fv) // 2
print(f'  first half {sum(fv[:h]):>10,.0f} · second half {sum(fv[h:]):>10,.0f}  '
      f'({"both +" if sum(fv[:h])>0 and sum(fv[h:])>0 else "flips"})')

print('\n  For contrast, the SAME pooled construction on the days it IS live:')
for k in (0, 1):
    v = wd.get(k, [])
    t3k = sorted(v, reverse=True)[:3]
    print(f'  {DOW[k]}  n {len(v)} · net {sum(v):>11,.0f} · t {tstat(v):>5.2f} · '
          f'maxDD {dd(v):>10,.0f} · top3 {100*sum(t3k)/sum(v) if sum(v) else 0:.0f}%')

# ── what adding Friday would have done to the real book ───────────────────
print('\n' + '=' * 72)
print('IF FRIDAY HAD BEEN LIVE — the real book with the Friday cell added')
print('=' * 72)
live = defaultdict(float)
matrix = json.loads((ROOT / 'backtest_data' / 'nas_day_matrix.json').read_text())['systems']
for key, db in SYS:
    c = sqlite3.connect(f'file:{ROOT}/backtest_data/{db}?mode=ro', uri=True)
    ld = {int(k) for k, v in (matrix[key].get('dte') or {}).items() if v}
    for d, pnl, lots in c.execute("SELECT trade_date, net_pnl, lots FROM nas_atm_trades "
                                  "WHERE exit_time IS NOT NULL AND net_pnl IS NOT NULL"):
        if d and lots and date.fromisoformat(d).weekday() < 5 \
                and WD2DTE['NIFTY'][date.fromisoformat(d).weekday()] in ld:
            live[d] += float(pnl)
    c.close()
base = [live[d] for d in sorted(live)]
print(f'  NIFTY 916 live today (Mon+Tue)      net {sum(base):>10,.0f} · maxDD {dd(base):>10,.0f} · '
      f'ret/DD {sum(base)/abs(dd(base)):>5.2f} · t {tstat(base):>4.2f} · {len(base)}d')
# Friday at the same real lots the book was running that week
addf = dict(live)
for key, db in SYS:
    c = sqlite3.connect(f'file:{ROOT}/backtest_data/{db}?mode=ro', uri=True)
    for d, pnl, lots in c.execute("SELECT trade_date, net_pnl, lots FROM nas_atm_trades "
                                  "WHERE exit_time IS NOT NULL AND net_pnl IS NOT NULL"):
        if d and lots and date.fromisoformat(d).weekday() == 4:
            addf[d] = addf.get(d, 0.0) + float(pnl)
    c.close()
av = [addf[d] for d in sorted(addf)]
print(f'  + Friday at the same real lots      net {sum(av):>10,.0f} · maxDD {dd(av):>10,.0f} · '
      f'ret/DD {sum(av)/abs(dd(av)):>5.2f} · t {tstat(av):>4.2f} · {len(av)}d')
print('\n  NOTE: the shadow trades at whatever lots the executor recorded that day, and a')
print('  shadow fill is not a real fill — no slippage, no margin gate, no rejection. Treat')
print('  this as the size of the opportunity, not as the P&L it would have banked.')
