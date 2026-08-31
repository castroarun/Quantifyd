"""Is the V1 30% CSL variant the same thing as the COMB book already running?

Arun's question, and a fair one: NAS_COMB20 is a live combined-premium-SL book,
the candidate is a combined-premium-SL variant, so a 0.09 correlation looks wrong.

It was. The portfolio I measured against contained nine NAS ATM/ATM2/ATM4 sleeves
and none of the CSL/COMB family — those records live in csl_paper_state.json, not
in the nas_*_trading.db stores. So the candidate was being correlated against a
book that excluded the system most like it.

This pulls the whole combined-premium family in, normalises to 10 lots, and
correlates the candidate against COMB20 directly.
"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
LOTS = 10

d = json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())
books = defaultdict(dict)
for r in d['records']:
    lots = float(r.get('lots') or 0)
    if not lots or r.get('pnl') is None:
        continue
    books[r['book']][r['day'][:10]] = float(r['pnl']) / lots * LOTS

print('COMBINED-PREMIUM FAMILY — all at 10 lots')
print(f"{'book':26} {'days':>5} {'net':>11} {'mean/day':>10} {'worst':>10} {'maxDD':>11}")
print('-' * 78)


def dd_of(vals):
    cum = peak = worst = 0.0
    for x in vals:
        cum += x; peak = max(peak, cum); worst = min(worst, cum - peak)
    return worst


for b, s in sorted(books.items(), key=lambda kv: -sum(kv[1].values())):
    v = [s[k] for k in sorted(s)]
    print(f'{b:26} {len(v):>5} {sum(v):>11,.0f} {sum(v)/len(v):>10,.0f} '
          f'{min(v):>10,.0f} {dd_of(v):>11,.0f}')

fam = defaultdict(float)
for s in books.values():
    for k, v in s.items():
        fam[k] += v
fv = [fam[k] for k in sorted(fam)]
print('-' * 78)
print(f"{'CSL/COMB FAMILY':26} {len(fv):>5} {sum(fv):>11,.0f} {sum(fv)/len(fv):>10,.0f} "
      f"{min(fv):>10,.0f} {dd_of(fv):>11,.0f}")

# the candidate, same normalisation
csl = {t['day'][:10]: t['final'] for t in json.loads(
    (ROOT / 'frontend' / 'public' / 'straddles' / 'v1_sl30.json').read_text())['trades']
    if t.get('final') is not None}


def corr(a, b):
    c = sorted(set(a) & set(b))
    if len(c) < 5:
        return None, len(c)
    x = [a[k] for k in c]; y = [b[k] for k in c]
    mx, my = st.mean(x), st.mean(y)
    num = sum((p - mx) * (q - my) for p, q in zip(x, y))
    den = (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5
    return (num / den if den else 0.0), len(c)


print('\nV1 + 30% CSL vs each combined-premium book it supposedly duplicates')
print(f"{'book':26} {'shared days':>12} {'corr':>7}")
print('-' * 78)
for b, s in sorted(books.items()):
    r, n = corr(csl, s)
    print(f'{b:26} {n:>12} {("%.2f" % r) if r is not None else "too few":>7}')
r, n = corr(csl, fam)
print('-' * 78)
print(f'{"CSL/COMB FAMILY combined":26} {n:>12} {("%.2f" % r) if r is not None else "too few":>7}')

v = list(csl.values())
print(f'\nCandidate itself: {len(v)} days · net {sum(v):,.0f} · '
      f'maxDD {dd_of([csl[k] for k in sorted(csl)]):,.0f} · worst day {min(v):,.0f}')
