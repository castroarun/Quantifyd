"""The live short-premium portfolio as one book, not eight cards.

Arun's framing: judge overall progress, not the best individual systems. Every
page in the app shows one system at a time, which is exactly how a book that is
really one bet in eight pieces comes to look diversified.

This builds the daily P&L of every NAS-family book that has traded, aligns them
on the calendar, and asks the questions a portfolio owner asks:

  * what does the combined equity curve look like, and what is its drawdown
  * how correlated are the sleeves with each other on DAILY P&L — the number
    that decides whether another one adds diversification or just size
  * how much of the total comes from each sleeve, and how much from its best days
  * where would the V1 + 30% CSL variant sit if it were added

Read-only. Reads each book's own store mode=ro and writes nothing.
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

# (label, db, table, live-or-paper) — the venue matters: NIFTY and SENSEX books
# trade different days by design, so they are not substitutes for each other
BOOKS = [
    ('NAS 916 ATM   (NIFTY)', 'nas_916_atm_trading.db',  'nas_atm_trades', 'live'),
    ('NAS 916 ATM2  (NIFTY)', 'nas_916_atm2_trading.db', 'nas_atm_trades', 'live'),
    ('NAS 916 ATM4  (NIFTY)', 'nas_916_atm4_trading.db', 'nas_atm_trades', 'live'),
    ('NAS ATM       (NIFTY)', 'nas_atm_trading.db',      'nas_atm_trades', 'shadow'),
    ('NAS ATM2      (NIFTY)', 'nas_atm2_trading.db',     'nas_atm_trades', 'shadow'),
    ('NAS ATM4      (NIFTY)', 'nas_atm4_trading.db',     'nas_atm_trades', 'shadow'),
    ('SENSEX ATM',            'sensex_atm_trading.db',   'nas_atm_trades', 'live'),
    ('SENSEX ATM2',           'sensex_atm2_trading.db',  'nas_atm_trades', 'live'),
    ('SENSEX ATM4',           'sensex_atm4_trading.db',  'nas_atm_trades', 'live'),
]


def daily(db: str, table: str) -> dict:
    p = ROOT / 'backtest_data' / db
    if not p.exists():
        return {}
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    try:
        cols = {r[1] for r in c.execute(f'PRAGMA table_info({table})')}
        pnl = 'net_pnl' if 'net_pnl' in cols else 'gross_pnl'
        out = defaultdict(float)
        for d, v in c.execute(
                f"SELECT substr(exit_time,1,10), {pnl} FROM {table} "
                f"WHERE exit_time IS NOT NULL AND {pnl} IS NOT NULL"):
            if d:
                out[d] += float(v)
        return dict(out)
    finally:
        c.close()


series = {}
for label, db, tab, kind in BOOKS:
    s = daily(db, tab)
    if s:
        series[label] = (s, kind)

# The two candidates for inclusion. Both are replays over the recorded chain, not
# live books — kept separate below so they never inflate the portfolio's own record,
# but aligned on the same days so the correlation is measured, not assumed.
def from_feed(fname, key='per_day'):
    f = ROOT / 'frontend' / 'public' / 'straddles' / fname
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    out = {}
    blob = d.get(key) or {}
    days = d.get('days') or []
    items = blob.items() if isinstance(blob, dict) else zip(days, blob)
    for day, v in items:
        ser = v.get('series') if isinstance(v, dict) else None
        if ser:
            out[str(day)[:10]] = float(ser[-1][1])
    return out


CANDIDATES = {
    'V1 one-and-done (naked)': from_feed('v1.json'),
    'V1 + 30% CSL (variant)': {t['day'][:10]: float(t['final'])
                               for t in json.loads(
                                   (ROOT / 'frontend' / 'public' / 'straddles' /
                                    'v1_sl30.json').read_text())['trades']
                               if t.get('final') is not None},
}
CANDIDATES = {k: v for k, v in CANDIDATES.items() if v}

all_days = sorted({d for s, _ in series.values() for d in s})
print(f'{len(series)} books · {len(all_days)} trading days · '
      f'{all_days[0]} -> {all_days[-1]}\n')

print('=' * 92)
print('SLEEVE BY SLEEVE')
print('=' * 92)
print(f"{'book':24} {'kind':7} {'days':>5} {'net':>12} {'mean/day':>10} {'best':>11} "
      f"{'worst':>11} {'top3 share':>11}")
print('-' * 92)
rows = []
for label, (s, kind) in sorted(series.items(), key=lambda kv: -sum(kv[1][0].values())):
    v = list(s.values())
    net = sum(v)
    top3 = sorted(v, reverse=True)[:3]
    share = 100 * sum(top3) / net if net > 0 else float('nan')
    rows.append((label, kind, s, net))
    print(f'{label:24} {kind:7} {len(v):>5} {net:>12,.0f} {net/len(v):>10,.0f} '
          f'{max(v):>11,.0f} {min(v):>11,.0f} {share:>10.0f}%')

# ── the portfolio, as actually held ────────────────────────────────────────
port = defaultdict(float)
for label, kind, s, _ in rows:
    for d, v in s.items():
        port[d] += v
pv = [port[d] for d in all_days]
cum, peak, dd, curve = 0.0, 0.0, 0.0, []
for x in pv:
    cum += x
    peak = max(peak, cum)
    dd = min(dd, cum - peak)
    curve.append(cum)

print('-' * 92)
print(f"{'PORTFOLIO (all sleeves)':24} {'':7} {len(pv):>5} {sum(pv):>12,.0f} "
      f"{sum(pv)/len(pv):>10,.0f} {max(pv):>11,.0f} {min(pv):>11,.0f}")
sd = st.stdev(pv)
print(f"\n  max drawdown        {dd:>12,.0f}")
print(f"  daily std dev       {sd:>12,.0f}")
print(f"  t on daily P&L      {sum(pv)/len(pv)/(sd/len(pv)**0.5):>12.2f}")
print(f"  green days          {sum(1 for x in pv if x > 0)}/{len(pv)} "
      f"({100*sum(1 for x in pv if x>0)/len(pv):.0f}%)")
top3 = sorted(pv, reverse=True)[:3]
print(f"  top 3 days          {sum(top3):>12,.0f}  = {100*sum(top3)/sum(pv):.0f}% of the total")
worst3 = sorted(pv)[:3]
print(f"  worst 3 days        {sum(worst3):>12,.0f}")

# ── correlation: the question that decides whether to add anything ─────────
print()
print('=' * 92)
print('DAILY-P&L CORRELATION — do these sleeves diversify, or are they one bet?')
print('=' * 92)
labels = [r[0] for r in rows]
print(f"{'':24} " + ' '.join(f'{l.split()[0][:6]:>7}' for l in labels))
for a in labels:
    sa = dict(series[a][0])
    line = f'{a:24} '
    for b in labels:
        sb = dict(series[b][0])
        common = sorted(set(sa) & set(sb))
        if len(common) < 8:
            line += f'{"·":>7} '
            continue
        x = [sa[d] for d in common]
        y = [sb[d] for d in common]
        mx, my = st.mean(x), st.mean(y)
        num = sum((p - mx) * (q - my) for p, q in zip(x, y))
        den = (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5
        line += f'{(num/den if den else 0):>7.2f} '
    print(line)

print()
print('=' * 92)
print('THE CANDIDATES — would adding one diversify the book, or just enlarge it?')
print('=' * 92)


def corr(a: dict, b: dict):
    common = sorted(set(a) & set(b))
    if len(common) < 8:
        return None, len(common)
    x = [a[d] for d in common]
    y = [b[d] for d in common]
    mx, my = st.mean(x), st.mean(y)
    num = sum((p - mx) * (q - my) for p, q in zip(x, y))
    den = (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5
    return (num / den if den else 0.0), len(common)


port_daily = {d: port[d] for d in all_days}
for name, cs in CANDIDATES.items():
    v = list(cs.values())
    r, n = corr(cs, port_daily)
    print()
    print(f'  {name}')
    print(f'    own record        {len(v)} days · net {sum(v):>12,.0f} · '
          f'mean/day {sum(v)/len(v):>9,.0f} · worst {min(v):>10,.0f}')
    print(f'    corr to portfolio {("%.2f" % r) if r is not None else "n/a":>12}  '
          f'(on {n} shared days)')
    # what the combined book would look like
    merged = dict(port_daily)
    for d, x in cs.items():
        merged[d] = merged.get(d, 0.0) + x
    mv = [merged[d] for d in sorted(merged)]
    cum = peak = ddm = 0.0
    for x in mv:
        cum += x; peak = max(peak, cum); ddm = min(ddm, cum - peak)
    base_ratio = sum(pv) / abs(dd) if dd else float('nan')
    new_ratio = sum(mv) / abs(ddm) if ddm else float('nan')
    print(f'    combined book     net {sum(mv):>12,.0f} · maxDD {ddm:>12,.0f} · '
          f'return/DD {new_ratio:>6.2f}  (portfolio alone {base_ratio:.2f})')
    print(f'    verdict           {"IMPROVES return/DD" if new_ratio > base_ratio else "does NOT improve return/DD"}')

Path(ROOT / 'research' / 'portfolio_view.json').write_text(json.dumps(
    dict(days=all_days, portfolio=[port[d] for d in all_days],
         sleeves={k: v[0] for k, v in series.items()}), indent=1), encoding='utf-8')
print(f"\nwrote research/portfolio_view.json")
