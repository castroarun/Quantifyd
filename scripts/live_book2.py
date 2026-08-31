"""Pressure-test the obvious conclusion before acting on it.

live_book.py says: drop the seven losing sleeves and return/DD goes 2.05 -> 4.48.
That is exactly the shape of an in-sample selection artefact — of course removing
whatever lost money improves the record. Before proposing any change to a live
book I have to separate two very different things:

  * a sleeve that lost because of ONE identifiable event (then dropping it is
    curve-fitting to a day that will not repeat in the same place), from
  * a sleeve that bleeds steadily (then it is structural, and dropping it is a
    decision rather than a fit).

So: for every losing sleeve, what does it look like WITHOUT its single worst day,
what is the t on its daily P&L, and how does the book's concentration read.

Also: the whole-book weekday shape, which live_book.py truncated.

Read-only.
"""
from __future__ import annotations

import json, sqlite3, statistics as st, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
LOTS = 10

NAS = [('NAS 916 ATM','nas_916_atm_trading.db'),('NAS 916 ATM2','nas_916_atm2_trading.db'),
       ('NAS 916 ATM4','nas_916_atm4_trading.db'),('NAS ATM','nas_atm_trading.db'),
       ('NAS ATM2','nas_atm2_trading.db'),('NAS ATM4','nas_atm4_trading.db'),
       ('SENSEX ATM','sensex_atm_trading.db'),('SENSEX ATM2','sensex_atm2_trading.db'),
       ('SENSEX ATM4','sensex_atm4_trading.db')]

series = {}
for label, db in NAS:
    p = ROOT / 'backtest_data' / db
    if not p.exists():
        continue
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    cols = {r[1] for r in c.execute('PRAGMA table_info(nas_atm_trades)')}
    col = 'net_pnl' if 'net_pnl' in cols else 'gross_pnl'
    s = defaultdict(float)
    for d, v, lots in c.execute(
            f"SELECT substr(exit_time,1,10), {col}, lots FROM nas_atm_trades "
            f"WHERE exit_time IS NOT NULL AND {col} IS NOT NULL"):
        if d and lots and float(lots) > 0:
            s[d] += float(v) / float(lots) * LOTS
    c.close()
    if s:
        series[label] = dict(s)

for r in json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())['records']:
    if r.get('pnl') is None or not float(r.get('lots') or 0):
        continue
    d = dict(series.get(r['book'], {}))
    d[r['day'][:10]] = d.get(r['day'][:10], 0.0) + float(r['pnl']) / float(r['lots']) * LOTS
    series[r['book']] = d

days = sorted({d for s in series.values() for d in s})
port = defaultdict(float)
for s in series.values():
    for d, x in s.items():
        port[d] += x
pv = [port[d] for d in days]


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


DAYN = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
print('WHOLE BOOK BY WEEKDAY (all 20 sleeves, 10 lots)')
wd = defaultdict(list)
for d in days:
    wd[date.fromisoformat(d).weekday()].append(port[d])
for k in range(5):
    v = wd.get(k)
    if v:
        print(f'  {DAYN[k]}  {len(v):>3}d · net {sum(v):>11,.0f} · mean {sum(v)/len(v):>9,.0f} · '
              f'worst {min(v):>10,.0f} · best {max(v):>10,.0f}')

print('\nTHE BIG DAYS — 76% of the book is three sessions. Which, and driven by what?')
big = sorted(days, key=lambda d: -abs(port[d]))[:6]
for d in sorted(big):
    contrib = sorted(((lab, s[d]) for lab, s in series.items() if d in s),
                     key=lambda kv: -abs(kv[1]))[:4]
    who = ' · '.join(f'{l} {v:+,.0f}' for l, v in contrib)
    print(f'  {d} {DAYN[date.fromisoformat(d).weekday()]}  {port[d]:>11,.0f}   {who}')

print('\nIS EACH LOSING SLEEVE A BLEED, OR ONE BAD DAY?')
print(f"{'sleeve':24} {'days':>5} {'net':>11} {'ex-worst-day':>13} {'t':>6} {'green':>7}")
print('-' * 74)
for lab, s in sorted(series.items(), key=lambda kv: sum(kv[1].values())):
    v = list(s.values())
    if sum(v) >= 0:
        continue
    ex = sorted(v)[1:]
    t = (st.mean(v) / (st.stdev(v) / len(v) ** 0.5)) if len(v) > 2 and st.stdev(v) else 0.0
    g = sum(1 for x in v if x > 0)
    print(f'{lab:24} {len(v):>5} {sum(v):>11,.0f} {sum(ex):>13,.0f} {t:>6.2f} '
          f'{g}/{len(v):>4}')

print('\n  Reading: a sleeve whose ex-worst-day figure is still deeply negative is bleeding.')
print('  One that flips positive lost to a single event, and dropping it is fitting to that day.')

# ── the honest version of the prune ────────────────────────────────────────
print('\nPRUNE TESTS — return/DD of the book under each removal')
base = sum(pv) / abs(dd(pv))
print(f'  {"whole book (20 sleeves)":34} {sum(pv):>11,.0f}  DD {dd(pv):>10,.0f}  ret/DD {base:>5.2f}')


def book_without(drop):
    p = defaultdict(float)
    for lab, s in series.items():
        if lab in drop:
            continue
        for d, x in s.items():
            p[d] += x
    v = [p[d] for d in sorted(p)]
    return sum(v), dd(v), (sum(v) / abs(dd(v)) if dd(v) else float('nan'))


for name, drop in [
    ('drop all 7 losers (in-sample)', {l for l, s in series.items() if sum(s.values()) < 0}),
    ('drop NAS ATM only',             {'NAS ATM'}),
    ('drop NAS ATM + NAS ATM2',       {'NAS ATM', 'NAS ATM2'}),
    ('drop the 4 CSL/COMB losers',    {'NAS_COMB20', 'CSL30F_NIFTY', 'NAS_C20_TRAIL',
                                       'CSL_TIMEB2_NIFTY'}),
    ('drop every NIFTY CSL sleeve',   {l for l in series
                                       if l.startswith(('CSL', 'NAS_C')) and 'SENSEX' not in l}),
]:
    n, d_, r = book_without(drop)
    print(f'  {name:34} {n:>11,.0f}  DD {d_:>10,.0f}  ret/DD {r:>5.2f}  '
          f'{"better" if r > base else "worse"}')

# ── what the NIFTY CSL family actually is ─────────────────────────────────
print('\nTHE NIFTY CSL/COMB FAMILY, POOLED (this is the live COMB book + its shadows)')
fam = defaultdict(float)
for lab, s in series.items():
    if lab.startswith(('CSL', 'NAS_C')) and 'SENSEX' not in lab:
        for d, x in s.items():
            fam[d] += x
fv = [fam[d] for d in sorted(fam)]
print(f'  {len(fv)} days · net {sum(fv):>11,.0f} · maxDD {dd(fv):>11,.0f} · '
      f'green {sum(1 for x in fv if x>0)}/{len(fv)}')
wdf = defaultdict(list)
for d in fam:
    wdf[date.fromisoformat(d).weekday()].append(fam[d])
for k in range(5):
    v = wdf.get(k)
    if v:
        print(f'    {DAYN[k]}  {len(v):>2}d · net {sum(v):>10,.0f} · mean {sum(v)/len(v):>9,.0f}')

print('\nTHE SENSEX CSL FAMILY, POOLED')
fam2 = defaultdict(float)
for lab, s in series.items():
    if 'SENSEX' in lab and lab.startswith('CSL'):
        for d, x in s.items():
            fam2[d] += x
f2 = [fam2[d] for d in sorted(fam2)]
print(f'  {len(f2)} days · net {sum(f2):>11,.0f} · maxDD {dd(f2):>11,.0f} · '
      f'green {sum(1 for x in f2 if x>0)}/{len(f2)}')
