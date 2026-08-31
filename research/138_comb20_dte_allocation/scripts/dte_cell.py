"""Which DTE cells does the 30% combined-premium stop actually earn on?

Framing (Arun, after correcting me twice): the candidate is not a new system —
on matched Mondays it correlates 0.96 with NAS_COMB20, the live book. And
COMB20's Monday cell (DTE1, 30% stop) is where it bleeds: -Rs 56,630 over three
Mondays against +Rs 18,695 on Fridays.

So the question stops being "should we add this" and becomes "is COMB20's day
allocation wrong". The candidate's replay carries a dte field per trade, so the
same mechanic can be sliced by DTE over 92 days instead of COMB20's 10.

Tests, in order:
  1. per-DTE record of the 30% stop across the whole replay
  2. the live COMB20 record per DTE, for comparison on the same axis
  3. what dropping or keeping each cell does to the book's total and drawdown

Read-only.
"""
import json, statistics as st, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
LOTS = 10

tr = json.loads((ROOT / 'frontend' / 'public' / 'straddles' / 'v1_sl30.json').read_text())['trades']
tr = [t for t in tr if t.get('final') is not None]


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


print('THE 30% COMBINED-PREMIUM STOP, BY DTE  (replay, 10 lots, 92 days)')
print(f"{'DTE':>4} {'weekday':>8} {'n':>4} {'net':>11} {'mean/day':>10} {'t':>6} "
      f"{'worst':>10} {'maxDD':>11} {'stopped':>8}")
print('-' * 82)
by = defaultdict(list)
for t in tr:
    by[t.get('dte')].append(t)

order = sorted(by, key=lambda k: (k is None, k))
for k in order:
    g = by[k]
    v = [x['final'] for x in g]
    wd = {DOW[date.fromisoformat(x['day'][:10]).weekday()] for x in g
          if date.fromisoformat(x['day'][:10]).weekday() < 5}
    t_ = (st.mean(v) / (st.stdev(v) / len(v) ** 0.5)) if len(v) > 2 and st.stdev(v) else float('nan')
    stopped = sum(1 for x in g if x.get('stopped'))
    print(f'{str(k):>4} {"/".join(sorted(wd))[:8]:>8} {len(v):>4} {sum(v):>11,.0f} '
          f'{sum(v)/len(v):>10,.0f} {t_:>6.2f} {min(v):>10,.0f} {dd(v):>11,.0f} '
          f'{stopped:>8}')

allv = [t['final'] for t in tr]
print('-' * 82)
print(f'{"ALL":>4} {"":>8} {len(allv):>4} {sum(allv):>11,.0f} {sum(allv)/len(allv):>10,.0f} '
      f'{st.mean(allv)/(st.stdev(allv)/len(allv)**0.5):>6.2f} {min(allv):>10,.0f} '
      f'{dd(allv):>11,.0f} {sum(1 for t in tr if t.get("stopped")):>8}')

# ── the live book on the same axis ─────────────────────────────────────────
d = json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())
comb = [r for r in d['records'] if r['book'] == 'NAS_COMB20'
        and r.get('pnl') is not None and float(r.get('lots') or 0)]
print('\nNAS_COMB20 LIVE, BY DTE (10 lots)')
cb = defaultdict(list)
for r in comb:
    cb[r.get('dte')].append(float(r['pnl']) / float(r['lots']) * LOTS)
for k in sorted(cb, key=lambda x: (x is None, x)):
    v = cb[k]
    print(f'  DTE{k}  {len(v):>2}d · net {sum(v):>10,.0f} · mean {sum(v)/len(v):>9,.0f} · '
          f'worst {min(v):>10,.0f}')

# ── what dropping a cell would do ─────────────────────────────────────────
print('\nDROP-ONE TEST on the replay — remove one DTE, keep the rest')
base_net, base_dd = sum(allv), dd(allv)
print(f'  {"keep all":22} net {base_net:>11,.0f} · maxDD {base_dd:>11,.0f} · '
      f'ret/DD {base_net/abs(base_dd):>5.2f}')
for k in order:
    keep = [t['final'] for t in tr if t.get('dte') != k]
    if not keep:
        continue
    n, w = sum(keep), dd(keep)
    r = n / abs(w) if w else float('nan')
    flag = 'BETTER' if r > base_net / abs(base_dd) else ''
    print(f'  {f"drop DTE{k}":22} net {n:>11,.0f} · maxDD {w:>11,.0f} · '
          f'ret/DD {r:>5.2f}  {flag}')

print('\nKEEP-ONLY TEST — run a single DTE cell')
for k in order:
    keep = [t['final'] for t in tr if t.get('dte') == k]
    if len(keep) < 5:
        continue
    n, w = sum(keep), dd(keep)
    print(f'  {f"only DTE{k}":22} net {n:>11,.0f} · maxDD {w:>11,.0f} · '
          f'ret/DD {(n/abs(w) if w else float("nan")):>5.2f} · n={len(keep)}')
