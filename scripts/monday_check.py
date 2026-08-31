"""What actually runs on Monday, and is the candidate the same thing?

Arun: "its 30% csl on mondays as i remember, chk whats live on monday and tuesday".
He is right. The numeric keys in csl_paper_config are DTE, not weekday, and for a
Tuesday NIFTY expiry DTE1 is Monday. NAS_COMB20 is configured
{DTE0: sl 25, DTE1: sl 30} at 09:16->15:20 — so Monday runs a 30% combined-premium
stop live, which is the candidate's mechanic.

This checks the live books day by day and compares the candidate against COMB20 on
Mondays alone, where they should be closest.
"""
import json, statistics as st, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
LOTS = 10

d = json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())
recs = [r for r in d['records'] if r.get('pnl') is not None and float(r.get('lots') or 0)]

print('WHAT EACH BOOK ACTUALLY TRADED, BY WEEKDAY (records, not config)')
print(f"{'book':26} " + ' '.join(f'{x:>16}' for x in DOW[:5]))
print('-' * 108)
for b in sorted({r['book'] for r in recs}):
    per = defaultdict(list)
    for r in recs:
        if r['book'] != b:
            continue
        wd = date.fromisoformat(r['day'][:10]).weekday()
        per[wd].append(float(r['pnl']) / float(r['lots']) * LOTS)
    line = f'{b:26} '
    for wd in range(5):
        v = per.get(wd)
        line += f'{(f"{len(v)}d {sum(v):+,.0f}" if v else "-"):>16} ' if v else f'{"-":>16} '
    print(line)

# the candidate
cand = {t['day'][:10]: t['final'] for t in json.loads(
    (ROOT / 'frontend' / 'public' / 'straddles' / 'v1_sl30.json').read_text())['trades']
    if t.get('final') is not None}

comb = {}
for r in recs:
    if r['book'] == 'NAS_COMB20':
        comb[r['day'][:10]] = float(r['pnl']) / float(r['lots']) * LOTS

print('\nCANDIDATE vs NAS_COMB20, MONDAYS ONLY (both 30% combined-premium there)')
mon = [k for k in sorted(set(cand) & set(comb))
       if date.fromisoformat(k).weekday() == 0]
if mon:
    print(f"{'day':12} {'candidate':>12} {'COMB20 live':>13} {'diff':>12}")
    for k in mon:
        print(f'{k:12} {cand[k]:>+12,.0f} {comb[k]:>+13,.0f} {cand[k]-comb[k]:>+12,.0f}')
    cs = [cand[k] for k in mon]; cb = [comb[k] for k in mon]
    print(f"{'TOTAL':12} {sum(cs):>+12,.0f} {sum(cb):>+13,.0f} {sum(cs)-sum(cb):>+12,.0f}")
    if len(mon) >= 3:
        x, y = cs, cb
        mx, my = st.mean(x), st.mean(y)
        num = sum((p-mx)*(q-my) for p, q in zip(x, y))
        den = (sum((p-mx)**2 for p in x) * sum((q-my)**2 for q in y)) ** 0.5
        print(f'\n  correlation on those {len(mon)} Mondays: '
              f'{(num/den if den else 0):.2f}')
else:
    print('  no shared Mondays')

print('\nCANDIDATE, BY WEEKDAY — where does its money come from?')
per = defaultdict(list)
for k, v in cand.items():
    per[date.fromisoformat(k).weekday()].append(v)
for wd in range(5):
    v = per.get(wd)
    if v:
        print(f'  {DOW[wd]}  {len(v):>3} days · net {sum(v):>10,.0f} · '
              f'mean {sum(v)/len(v):>8,.0f} · worst {min(v):>10,.0f}')
