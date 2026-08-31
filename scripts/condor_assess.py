"""Should the Wed->Fri iron condor join the live NIFTY book?

The question is NOT "is it profitable" — a 7-cycle book cannot answer that. It is
"does adding it improve the book", which is a different and more answerable
question, because a sleeve can earn its place on WHEN it trades and WHAT SHAPE its
risk is, even before its own t is convincing.

Three things decide it:

  1. its own record, stated with the sample size in plain view and with the
     number of cycles it would need before its mean is distinguishable from luck
  2. whether it occupies calendar the live book leaves empty, or piles onto days
     already covered — measured, not assumed
  3. what it does to the whole book's return-per-drawdown

The structural argument, which is the interesting one: every other sleeve in the
live book is a SHORT straddle — naked, or protected only by a stop that has to
fire in time. The condor buys wings, so its worst case is known at entry and does
not depend on a stop working. That is a different risk shape, not just a
different day, and it is the kind of thing correlation alone will not show.

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
WD2DTE = {'NIFTY': {0: 1, 1: 0, 2: 4, 3: 3, 4: 2}, 'SENSEX': {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}}


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


def tstat(v):
    return (st.mean(v) / (st.stdev(v) / len(v) ** 0.5)) if len(v) > 2 and st.stdev(v) else 0.0


# ── the condor's own record ───────────────────────────────────────────────
h = json.loads((ROOT / 'backtest_data' / 'condor_paper_state.json').read_text())['history']
h = sorted(h, key=lambda x: x['entry_day'])
BOOK_QTY = 130          # 2 lots x 65
SCALE = 650 / BOOK_QTY  # show at 10 lots alongside everything else

print('WED->FRI IRON CONDOR — every completed cycle')
print(f"  {'entry':11} {'exit':11} {'credit':>8} {'exit val':>9} "
      f"{'P&L @2L':>10} {'P&L @10L':>10}")
for x in h:
    print(f"  {x['entry_day']:11} {x['exit_day']:11} {x['credit']:>8.2f} "
          f"{x['exit_value']:>9.2f} {x['pnl']:>10,.0f} {x['pnl']*SCALE:>10,.0f}")

v2 = [x['pnl'] for x in h]
v10 = [x * SCALE for x in v2]
n = len(v2)
sd = st.stdev(v2)
t = tstat(v2)
need = (2 * sd / st.mean(v2)) ** 2 if st.mean(v2) else float('inf')
print(f'\n  {n} cycles · net Rs{sum(v2):,.0f} at 2 lots = Rs{sum(v10):,.0f} at 10 lots')
print(f'  mean/cycle Rs{st.mean(v2):,.0f} · sd Rs{sd:,.0f} · '
      f'green {sum(1 for x in v2 if x>0)}/{n}')
print(f'  maxDD Rs{dd(v2):,.0f} (2L) · worst cycle Rs{min(v2):,.0f}')
print(f'  t = {t:.2f}   <-- 2.0 is the 1-in-20 bar')
print(f'  cycles needed for t=2 at this mean and spread: ~{need:.0f}  '
      f'(= ~{need-n:.0f} more weeks)')

# ── does it occupy empty calendar? ────────────────────────────────────────
print('\n' + '=' * 74)
print('DOES IT FILL A GAP, OR DOUBLE UP? — what the live book holds each weekday')
print('=' * 74)
matrix = json.loads((ROOT / 'backtest_data' / 'nas_day_matrix.json').read_text())['systems']
holds = defaultdict(list)
for key, venue in [('nas_916_atm', 'NIFTY'), ('nas_916_atm2', 'NIFTY'), ('nas_916_atm4', 'NIFTY'),
                   ('sensex_atm', 'SENSEX'), ('sensex_atm2', 'SENSEX'), ('sensex_atm4', 'SENSEX')]:
    row = matrix.get(key, {})
    if not row.get('live'):
        continue
    ld = {int(k) for k, v in (row.get('dte') or {}).items() if v}
    for wd in range(5):
        if WD2DTE[venue][wd] in ld:
            holds[wd].append(key)
holds[1].append('NAS_COMB20')          # live, Tuesday only after 31-Aug
for wd in range(5):
    who = holds.get(wd) or []
    cond = 'CONDOR HOLDS (entered Wed, carried to Fri)' if wd in (2, 3, 4) else ''
    print(f'  {DOW[wd]:4} {len(who)} live sleeve(s): {", ".join(who) or "NOTHING":58} {cond}')
print('\n  The condor is entered Wednesday and carried through Thursday to Friday close.')
print('  Wed/Thu already carry the three SENSEX sleeves. FRIDAY carries nothing at all —')
print('  that is the only genuinely empty day it fills, and it fills it with a')
print('  defined-risk structure rather than another short straddle.')

# ── correlation and combined effect ──────────────────────────────────────
print('\n' + '=' * 74)
print('WHAT IT DOES TO THE WHOLE LIVE BOOK')
print('=' * 74)
live = defaultdict(float)
for key, db, venue in [('nas_916_atm', 'nas_916_atm_trading.db', 'NIFTY'),
                       ('nas_916_atm2', 'nas_916_atm2_trading.db', 'NIFTY'),
                       ('nas_916_atm4', 'nas_916_atm4_trading.db', 'NIFTY'),
                       ('sensex_atm', 'sensex_atm_trading.db', 'SENSEX'),
                       ('sensex_atm2', 'sensex_atm2_trading.db', 'SENSEX'),
                       ('sensex_atm4', 'sensex_atm4_trading.db', 'SENSEX')]:
    ld = {int(k) for k, vv in (matrix[key].get('dte') or {}).items() if vv}
    if not matrix[key].get('live'):
        continue
    c = sqlite3.connect(f'file:{ROOT}/backtest_data/{db}?mode=ro', uri=True)
    for d, pnl, lots in c.execute("SELECT trade_date, net_pnl, lots FROM nas_atm_trades "
                                  "WHERE exit_time IS NOT NULL AND net_pnl IS NOT NULL"):
        if d and lots and date.fromisoformat(d).weekday() < 5 \
                and WD2DTE[venue][date.fromisoformat(d).weekday()] in ld:
            live[d] += float(pnl)
    c.close()
for r in json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())['records']:
    if r['book'] == 'NAS_COMB20' and r.get('pnl') is not None:
        live[r['day'][:10]] += float(r['pnl'])

# condor P&L booked on its exit day (Friday) — that is when it is realised
cond = {x['exit_day'][:10]: x['pnl'] for x in h}
base = [live[d] for d in sorted(live)]
print(f'  live book today          net Rs{sum(base):>10,.0f} · maxDD Rs{dd(base):>10,.0f} · '
      f'ret/DD {sum(base)/abs(dd(base)):>5.2f} · t {tstat(base):>4.2f} · {len(base)}d')

for lots, tag in ((2, 'at its current 2 lots'), (5, 'at 5 lots'), (10, 'at 10 lots')):
    m = dict(live)
    k = lots / 2
    for d, p in cond.items():
        m[d] = m.get(d, 0.0) + p * k
    mv = [m[d] for d in sorted(m)]
    print(f'  + condor {tag:22} net Rs{sum(mv):>10,.0f} · maxDD Rs{dd(mv):>10,.0f} · '
          f'ret/DD {sum(mv)/abs(dd(mv)):>5.2f} · t {tstat(mv):>4.2f} · {len(mv)}d')

common = sorted(set(cond) & set(live))
if len(common) >= 4:
    x = [cond[d] for d in common]; y = [live[d] for d in common]
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    print(f'\n  correlation to the live book on {len(common)} shared days: '
          f'{(num/den if den else 0):.2f}')
else:
    print(f'\n  correlation: only {len(common)} shared days — not measurable yet. '
          f'The live book trades almost none of the days the condor exits on, which is '
          f'itself the point.')
