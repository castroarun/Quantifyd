"""The live short-premium book as one portfolio, every sleeve on the same size.

Arun's framing: judge overall progress, not the best individual systems — and
mixing real-money with paper-shadow sleeves is fine provided the sizes are put
on a common footing first.

So every trade is reduced to P&L PER LOT (net_pnl / lots) and then re-expressed
at a single notional size. Without that the aggregate is dominated by whichever
sleeve happened to be running the most lots, which is a sizing decision rather
than a result.

One caveat that normalisation cannot remove: a NIFTY lot and a SENSEX lot are
different notional, so per-lot is the unit these books are SIZED in, not equal
risk. The cross-venue comparison is indicative; the within-venue one is exact.

Read-only.
"""
from __future__ import annotations

import json
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOTS = 10          # everything expressed at this size
NIFTY_LOT, SENSEX_LOT = 65, 20

BOOKS = [
    ('NAS 916 ATM',  'nas_916_atm_trading.db',  'nas_atm_trades', 'NIFTY'),
    ('NAS 916 ATM2', 'nas_916_atm2_trading.db', 'nas_atm_trades', 'NIFTY'),
    ('NAS 916 ATM4', 'nas_916_atm4_trading.db', 'nas_atm_trades', 'NIFTY'),
    ('NAS ATM',      'nas_atm_trading.db',      'nas_atm_trades', 'NIFTY'),
    ('NAS ATM2',     'nas_atm2_trading.db',     'nas_atm_trades', 'NIFTY'),
    ('NAS ATM4',     'nas_atm4_trading.db',     'nas_atm_trades', 'NIFTY'),
    ('SENSEX ATM',   'sensex_atm_trading.db',   'nas_atm_trades', 'SENSEX'),
    ('SENSEX ATM2',  'sensex_atm2_trading.db',  'nas_atm_trades', 'SENSEX'),
    ('SENSEX ATM4',  'sensex_atm4_trading.db',  'nas_atm_trades', 'SENSEX'),
]


def daily_per_lot(db: str, table: str) -> tuple[dict, dict]:
    """-> ({day: P&L at LOTS}, {lots: n_trades}) so the sizing is visible, not hidden."""
    p = ROOT / 'backtest_data' / db
    if not p.exists():
        return {}, {}
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    try:
        cols = {r[1] for r in c.execute(f'PRAGMA table_info({table})')}
        pnl = 'net_pnl' if 'net_pnl' in cols else 'gross_pnl'
        out, sizes = defaultdict(float), defaultdict(int)
        for d, v, lots in c.execute(
                f"SELECT substr(exit_time,1,10), {pnl}, lots FROM {table} "
                f"WHERE exit_time IS NOT NULL AND {pnl} IS NOT NULL"):
            if not d:
                continue
            n = float(lots or 0)
            if n <= 0:
                continue                      # cannot normalise what has no size
            out[d] += float(v) / n * LOTS
            sizes[int(n)] += 1
        return dict(out), dict(sizes)
    finally:
        c.close()


series, sizing = {}, {}
for label, db, tab, venue in BOOKS:
    s, sz = daily_per_lot(db, tab)
    if s:
        series[label] = (s, venue)
        sizing[label] = sz

all_days = sorted({d for s, _ in series.values() for d in s})
print(f'{len(series)} sleeves · {len(all_days)} trading days · {all_days[0]} -> {all_days[-1]}')
print(f'every sleeve normalised to {LOTS} lots (P&L / its own lots x {LOTS})\n')

print('=' * 100)
print(f'SLEEVE BY SLEEVE — all at {LOTS} lots')
print('=' * 100)
print(f"{'sleeve':14} {'venue':7} {'lots traded':16} {'days':>5} {'net':>12} {'mean/day':>10} "
      f"{'worst':>11} {'top3':>7}")
print('-' * 100)
rows = []
for label, (s, venue) in sorted(series.items(), key=lambda kv: -sum(kv[1][0].values())):
    v = list(s.values())
    net = sum(v)
    sz = ', '.join(f'{k}L x{n}' for k, n in sorted(sizing[label].items()))
    top3 = sum(sorted(v, reverse=True)[:3])
    share = f'{100*top3/net:.0f}%' if net > 0 else '—'
    rows.append((label, s, venue))
    print(f'{label:14} {venue:7} {sz[:16]:16} {len(v):>5} {net:>12,.0f} {net/len(v):>10,.0f} '
          f'{min(v):>11,.0f} {share:>7}')

port = defaultdict(float)
for label, s, _ in rows:
    for d, v in s.items():
        port[d] += v
pv = [port[d] for d in all_days]
cum = peak = dd = 0.0
for x in pv:
    cum += x; peak = max(peak, cum); dd = min(dd, cum - peak)

print('-' * 100)
print(f"{'PORTFOLIO':14} {'':7} {'':16} {len(pv):>5} {sum(pv):>12,.0f} {sum(pv)/len(pv):>10,.0f} "
      f"{min(pv):>11,.0f}")
sd = st.stdev(pv)
top3 = sorted(pv, reverse=True)[:3]
print(f"\n  max drawdown   {dd:>12,.0f}      t on daily P&L {sum(pv)/len(pv)/(sd/len(pv)**0.5):>6.2f}")
print(f"  daily std dev  {sd:>12,.0f}      green days     "
      f"{sum(1 for x in pv if x>0)}/{len(pv)} ({100*sum(1 for x in pv if x>0)/len(pv):.0f}%)")
print(f"  top 3 days     {sum(top3):>12,.0f}  = {100*sum(top3)/sum(pv):.0f}% of the total")
print(f"  return / DD    {sum(pv)/abs(dd):>12.2f}")

# ── venue split, since a NIFTY lot != a SENSEX lot ─────────────────────────
print()
for venue in ('NIFTY', 'SENSEX'):
    sub = defaultdict(float)
    for label, s, v in rows:
        if v == venue:
            for d, x in s.items():
                sub[d] += x
    if not sub:
        continue
    vv = list(sub.values())
    cum = peak = d2 = 0.0
    for x in [sub[d] for d in sorted(sub)]:
        cum += x; peak = max(peak, cum); d2 = min(d2, cum - peak)
    print(f'  {venue:7} {len(vv):>4} days · net {sum(vv):>11,.0f} · maxDD {d2:>11,.0f} · '
          f'ret/DD {sum(vv)/abs(d2) if d2 else float("nan"):>5.2f}')

# ── candidates, also per-lot ───────────────────────────────────────────────
def corr(a, b):
    common = sorted(set(a) & set(b))
    if len(common) < 8:
        return None, len(common)
    x = [a[d] for d in common]; y = [b[d] for d in common]
    mx, my = st.mean(x), st.mean(y)
    num = sum((p-mx)*(q-my) for p, q in zip(x, y))
    den = (sum((p-mx)**2 for p in x) * sum((q-my)**2 for q in y)) ** 0.5
    return (num/den if den else 0.0), len(common)


V1_LOTS = 10        # both V1 feeds are published at 10 lots already
cands = {}
f = ROOT / 'frontend' / 'public' / 'straddles' / 'v1.json'
if f.exists():
    d = json.loads(f.read_text())
    cands['V1 one-and-done'] = {str(k)[:10]: v['series'][-1][1] / V1_LOTS * LOTS
                                for k, v in (d.get('per_day') or {}).items() if v.get('series')}
f = ROOT / 'frontend' / 'public' / 'straddles' / 'v1_sl30.json'
if f.exists():
    d = json.loads(f.read_text())
    cands['V1 + 30% CSL'] = {t['day'][:10]: t['final'] / V1_LOTS * LOTS
                             for t in d['trades'] if t.get('final') is not None}

print()
print('=' * 100)
print(f'CANDIDATES — also at {LOTS} lots. Does adding one improve the whole book?')
print('=' * 100)
pd_ = {d: port[d] for d in all_days}
base = sum(pv) / abs(dd)
for name, cs in cands.items():
    r, n = corr(cs, pd_)
    merged = dict(pd_)
    for d, x in cs.items():
        merged[d] = merged.get(d, 0.0) + x
    mv = [merged[d] for d in sorted(merged)]
    cum = peak = dm = 0.0
    for x in mv:
        cum += x; peak = max(peak, cum); dm = min(dm, cum - peak)
    ratio = sum(mv) / abs(dm) if dm else float('nan')
    print(f'  {name:18} own net {sum(cs.values()):>11,.0f} ({len(cs)}d) · '
          f'corr {("%.2f" % r) if r is not None else "n/a":>5} · '
          f'combined ret/DD {ratio:>5.2f} vs {base:.2f} · '
          f'{"BETTER" if ratio > base else "worse"}')
