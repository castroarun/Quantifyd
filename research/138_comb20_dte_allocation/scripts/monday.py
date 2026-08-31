"""Why the Monday cell reads positive on the replay and negative in the live book.

Q3 of the confound test showed DTE1 (Monday) at SL30 earning Rs 1,31,845 in the
first half of the recorded window and Rs 12,960 in the second — a 90% collapse
that the pooled figure hides. If that decay is real and ordered in time, then the
replay's Monday profit is a fact about April and May, and the live book — which
only started trading Mondays in August — is sampling the decayed part. That
would reconcile the two records without either being wrong.

Test: Monday chronologically, replay and live side by side, no aggregation.
Plus the live record with dates, including today.

Read-only.
"""
from __future__ import annotations

import json, statistics as st, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
QTY, COST = 650, 160

study = json.loads((ROOT / 'static' / 'app' / 'options_study.json').read_text())


def run(day, sl=30.0):
    bars = [(h, p) for h, p in ((b[0], b[1]) for b in (day.get('series') or []) if b[1])
            if h >= '09:20']
    if len(bars) < 2:
        return None
    ent = bars[0][1]
    thr = (1 + sl / 100.0) * ent
    for h, p in bars:
        if p >= thr:
            return round((ent - p) * QTY - COST)
    return round((ent - bars[-1][1]) * QTY - COST)


print('MONDAY (DTE1) ON THE RECORDED CHAIN, IN TIME ORDER — 30% stop, 10 lots')
mons = [(d['date'], run(d)) for d in sorted(study['days'], key=lambda x: x['date'])
        if d['dte'] == 1 and run(d) is not None]
cum = 0
for d, p in mons:
    cum += p
    print(f'  {d}  {p:>10,.0f}   cum {cum:>10,.0f}')
h = len(mons) // 2
print(f'\n  first {h}: {sum(p for _, p in mons[:h]):>10,.0f}   '
      f'last {len(mons)-h}: {sum(p for _, p in mons[h:]):>10,.0f}')
last6 = [p for _, p in mons[-6:]]
print(f'  most recent 6 Mondays on the chain: {sum(last6):>10,.0f}  '
      f'({sum(1 for x in last6 if x>0)}/6 green)')

# ── the live record, by book, Mondays only, with dates ─────────────────────
print('\nTHE LIVE/PAPER RECORD ON MONDAYS — every NIFTY CSL-family book, 10-lot normalised')
recs = json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())['records']
mon = defaultdict(dict)
for r in recs:
    if r.get('pnl') is None or not float(r.get('lots') or 0):
        continue
    d = r['day'][:10]
    if date.fromisoformat(d).weekday() != 0:
        continue
    if 'SENSEX' in r['book']:
        continue
    mon[d][r['book']] = float(r['pnl']) / float(r['lots']) * 10

books = sorted({b for v in mon.values() for b in v})
print(f"  {'date':12}" + ''.join(f'{b[:13]:>15}' for b in books) + f"{'DAY':>13}")
for d in sorted(mon):
    tot = sum(mon[d].values())
    print(f'  {d:12}' + ''.join(f'{mon[d].get(b, 0):>15,.0f}' for b in books) + f'{tot:>13,.0f}')
tot_by_book = {b: sum(mon[d].get(b, 0) for d in mon) for b in books}
print(f"  {'TOTAL':12}" + ''.join(f'{tot_by_book[b]:>15,.0f}' for b in books)
      + f'{sum(tot_by_book.values()):>13,.0f}')

print('\nAND THE SAME BOOKS ON EVERY OTHER WEEKDAY, for contrast')
oth = defaultdict(float)
cnt = defaultdict(set)
for r in recs:
    if r.get('pnl') is None or not float(r.get('lots') or 0) or 'SENSEX' in r['book']:
        continue
    d = r['day'][:10]
    w = date.fromisoformat(d).weekday()
    oth[w] += float(r['pnl']) / float(r['lots']) * 10
    cnt[w].add(d)
for w in range(5):
    if w in oth:
        print(f'  {["Mon","Tue","Wed","Thu","Fri"][w]}  {len(cnt[w])}d  {oth[w]:>12,.0f}')

print('\nTHURSDAY LIVE (DTE3) — the cell the evidence says to size up')
thu = defaultdict(float)
for r in recs:
    if r.get('pnl') is None or not float(r.get('lots') or 0) or 'SENSEX' in r['book']:
        continue
    d = r['day'][:10]
    if date.fromisoformat(d).weekday() != 3:
        continue
    thu[d] += float(r['pnl']) / float(r['lots']) * 10
for d in sorted(thu):
    print(f'  {d}  {thu[d]:>12,.0f}')
if thu:
    print(f'  {len(thu)} Thursdays · net {sum(thu.values()):>12,.0f} at 10-lot equivalent')
