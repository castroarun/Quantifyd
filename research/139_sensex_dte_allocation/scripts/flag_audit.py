"""Which CSL/COMB books have actually placed REAL orders? Evidence, not flags.

The question: is_live_book() reads only NAS_COMB20 as live, but CSL_TIMEB_SENSEX
and CSL_TIMEB2_LIVE carry comments asserting they are real money. That gap changes
the SENSEX expiry-day risk from Rs18,000 to Rs34,035, so it needs settling on
evidence rather than on which line of the file one believes.

Four independent places the truth would show up, checked in order of how hard they
are to fake:

  1. broker order ids recorded against a book's trades — an order id exists only if
     an order was placed
  2. the executor's own event log, where entries are pushed with a "REAL" tag
  3. the cron log /tmp/csl_paper.log, where every run prints mode=LIVE|paper per book
  4. the published state file's per-record fields

If a book has never produced an order id or a LIVE log line, it has never traded
real money regardless of what any comment says.

Read-only.
"""
from __future__ import annotations

import json, re, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

state = json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())

# ── 1. what fields do records actually carry? ─────────────────────────────
recs = state['records']
keys = sorted({k for r in recs for k in r})
print(f'{len(recs)} records · fields present: {", ".join(keys)}\n')

orderish = [k for k in keys if re.search(r'order|oid|broker|kite|real|live|mode', k, re.I)]
print(f'order/mode-related fields: {orderish or "NONE — records carry no order id at all"}')

by_book = defaultdict(lambda: dict(n=0, oids=0, real=0, days=set()))
for r in recs:
    b = by_book[r['book']]
    b['n'] += 1
    b['days'].add(r['day'][:10])
    for k in orderish:
        v = r.get(k)
        if v and str(v) not in ('0', 'False', 'paper', 'none'):
            b['oids'] += 1
    if str(r.get('mode', '')).lower() == 'live' or r.get('real'):
        b['real'] += 1

print(f"\n{'book':24} {'recs':>5} {'days':>5} {'order-ish set':>14} {'marked real':>12}")
print('-' * 64)
for bk in sorted(by_book):
    b = by_book[bk]
    print(f'{bk:24} {b["n"]:>5} {len(b["days"]):>5} {b["oids"]:>14} {b["real"]:>12}')

# ── 2. the event log ──────────────────────────────────────────────────────
ev = state.get('events') or []
print(f'\n{len(ev)} events in the state file. Sample shape: {ev[0] if ev else "—"}')
real_ev = [e for e in ev if 'REAL' in json.dumps(e).upper() or 'LIVE' in json.dumps(e).upper()]
print(f'events mentioning REAL/LIVE: {len(real_ev)}')
for e in real_ev[-12:]:
    print('   ', json.dumps(e)[:190])

# ── 3. the cron log — the executor prints mode per book every run ─────────
log = Path('/tmp/csl_paper.log')
print(f'\n--- /tmp/csl_paper.log ---')
if not log.exists():
    print('  not present')
else:
    txt = log.read_text(errors='replace').splitlines()
    print(f'  {len(txt)} lines')
    probe = [l for l in txt if 'mode=' in l]
    print(f'  PROBE lines carrying mode=: {len(probe)}')
    seen = {}
    for l in probe:
        m = re.search(r'PROBE\s+(\S+).*mode=(\w+)', l)
        if m:
            seen.setdefault(m.group(1), set()).add(m.group(2))
    for k, v in sorted(seen.items()):
        print(f'    {k:22} modes seen: {sorted(v)}')
    live_lines = [l for l in txt if re.search(r'mode=LIVE|LIVE order|order placed|order_id', l)]
    print(f'\n  lines indicating a LIVE order: {len(live_lines)}')
    for l in live_lines[-15:]:
        print('   ', l[:190])

# ── 4. Thursday-specific: did TimeB SENSEX ever trade on an expiry day? ───
print('\n' + '=' * 74)
print('CSL_TIMEB_SENSEX — its actual record, by weekday')
print('=' * 74)
tb = [r for r in recs if r['book'] == 'CSL_TIMEB_SENSEX']
if not tb:
    print('  no records at all')
for r in sorted(tb, key=lambda x: x['day']):
    d = r['day'][:10]
    print(f'  {d} {DOW[date.fromisoformat(d).weekday()]}  DTE{r.get("dte")}  '
          f'lots {r.get("lots")}  pnl {r.get("pnl")}  '
          f'{" ".join(f"{k}={r[k]}" for k in orderish if k in r)}')

print('\nVERDICT')
print('  A book that has never recorded a broker order id, never logged mode=LIVE, and')
print('  never emitted a REAL event has not traded real money — whatever the comment says.')
