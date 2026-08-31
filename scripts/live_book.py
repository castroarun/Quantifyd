"""The live short-premium book — NAS suite AND the CSL/COMB family, together.

Every version of this I have produced so far was incomplete. The first read nine
NAS ATM sleeves and no CSL/COMB at all, because those records live in
csl_paper_state.json rather than the nas_*_trading.db stores. The second was
lot-normalised but still missing the family. This is both, on one footing.

All sleeves normalised to 10 lots (P&L / its own lots x 10) per Arun's
instruction, so a sleeve's weight reflects its result rather than how many lots
it happened to be running.

Questions it answers, in the order they matter for a live book:
  1. what is the book actually making, and what did it cost in drawdown
  2. which sleeves carry it and which are dead weight
  3. how much of the profit is a handful of days
  4. how correlated are the sleeves — is this one bet or several
  5. what would pruning the worst sleeves do

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

NAS = [
    ('NAS 916 ATM',  'nas_916_atm_trading.db',  'NIFTY'),
    ('NAS 916 ATM2', 'nas_916_atm2_trading.db', 'NIFTY'),
    ('NAS 916 ATM4', 'nas_916_atm4_trading.db', 'NIFTY'),
    ('NAS ATM',      'nas_atm_trading.db',      'NIFTY'),
    ('NAS ATM2',     'nas_atm2_trading.db',     'NIFTY'),
    ('NAS ATM4',     'nas_atm4_trading.db',     'NIFTY'),
    ('SENSEX ATM',   'sensex_atm_trading.db',   'SENSEX'),
    ('SENSEX ATM2',  'sensex_atm2_trading.db',  'SENSEX'),
    ('SENSEX ATM4',  'sensex_atm4_trading.db',  'SENSEX'),
]

series = {}
for label, db, venue in NAS:
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
        series[label] = (dict(s), venue, 'NAS')

csl = json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())
for r in csl['records']:
    if r.get('pnl') is None or not float(r.get('lots') or 0):
        continue
    lab = r['book']
    ven = 'SENSEX' if 'SENSEX' in lab else 'NIFTY'
    d, _, _ = series.get(lab, ({}, ven, 'CSL'))
    d = dict(d)
    d[r['day'][:10]] = d.get(r['day'][:10], 0.0) + float(r['pnl']) / float(r['lots']) * LOTS
    series[lab] = (d, ven, 'CSL')

days = sorted({d for s, _, _ in series.values() for d in s})


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


print(f'LIVE SHORT-PREMIUM BOOK — {len(series)} sleeves · {len(days)} days · '
      f'{days[0]} → {days[-1]} · all at {LOTS} lots\n')
print(f"{'sleeve':26} {'fam':4} {'venue':7} {'days':>5} {'net':>11} {'mean/day':>9} "
      f"{'worst':>10} {'maxDD':>11}")
print('-' * 92)
rows = []
for lab, (s, ven, fam) in sorted(series.items(), key=lambda kv: -sum(kv[1][0].values())):
    v = [s[k] for k in sorted(s)]
    rows.append((lab, s, ven, fam, sum(v)))
    print(f'{lab:26} {fam:4} {ven:7} {len(v):>5} {sum(v):>11,.0f} {sum(v)/len(v):>9,.0f} '
          f'{min(v):>10,.0f} {dd(v):>11,.0f}')

port = defaultdict(float)
for lab, s, _, _, _ in rows:
    for d, x in s.items():
        port[d] += x
pv = [port[d] for d in days]
print('-' * 92)
print(f"{'WHOLE BOOK':26} {'':4} {'':7} {len(pv):>5} {sum(pv):>11,.0f} "
      f"{sum(pv)/len(pv):>9,.0f} {min(pv):>10,.0f} {dd(pv):>11,.0f}")

sd = st.stdev(pv)
top3 = sorted(pv, reverse=True)[:3]
print(f"\n  overall P&L      {sum(pv):>12,.0f}")
print(f"  max drawdown     {dd(pv):>12,.0f}")
print(f"  return / DD      {sum(pv)/abs(dd(pv)):>12.2f}")
print(f"  t on daily P&L   {sum(pv)/len(pv)/(sd/len(pv)**0.5):>12.2f}")
print(f"  green days       {sum(1 for x in pv if x>0)}/{len(pv)} "
      f"({100*sum(1 for x in pv if x>0)/len(pv):.0f}%)")
print(f"  top 3 days       {sum(top3):>12,.0f}  = {100*sum(top3)/sum(pv):.0f}% of the total")

# ── dead weight ────────────────────────────────────────────────────────────
print('\nLOSING SLEEVES')
losers = [r for r in rows if r[4] < 0]
for lab, s, ven, fam, net in losers:
    print(f'  {lab:26} {net:>11,.0f}')
print(f'  {"total drag":26} {sum(r[4] for r in losers):>11,.0f}')

pruned = defaultdict(float)
for lab, s, _, _, net in rows:
    if net <= 0:
        continue
    for d, x in s.items():
        pruned[d] += x
prd = sorted(pruned)
prv = [pruned[d] for d in prd]
print(f'\n  book WITHOUT the losing sleeves: net {sum(prv):>11,.0f} · '
      f'maxDD {dd(prv):>11,.0f} · ret/DD {sum(prv)/abs(dd(prv)):>5.2f}'
      f'   (whole book {sum(pv)/abs(dd(pv)):.2f})')

# ── correlation among the survivors ────────────────────────────────────────
keep = [r for r in rows if r[4] > 0][:8]
print('\nCORRELATION AMONG PROFITABLE SLEEVES (daily P&L)')
print(f"{'':26} " + ' '.join(f'{l[0][:6]:>7}' for l in keep))
for a in keep:
    line = f'{a[0]:26} '
    for b in keep:
        common = sorted(set(a[1]) & set(b[1]))
        if len(common) < 6:
            line += f'{"·":>7} '
            continue
        x = [a[1][d] for d in common]; y = [b[1][d] for d in common]
        mx, my = st.mean(x), st.mean(y)
        num = sum((p-mx)*(q-my) for p, q in zip(x, y))
        den = (sum((p-mx)**2 for p in x) * sum((q-my)**2 for q in y)) ** 0.5
        line += f'{(num/den if den else 0):>7.2f} '
    print(line)

# ── weekday shape of the whole book ───────────────────────────────────────
print('\nWHOLE BOOK BY WEEKDAY')
wd = defaultdict(list)
for d in days:
    wd[date.fromisoformat(d).weekday()].append(port[d])
for k in range(5):
    v = wd.get(k)
    if v:
        print(f'  {["Mon","Tue","Wed","Thu","Fri"][k]}  {len(v):>3}d · net {sum(v):>11,.0f} · '
              f'mean {sum(v)/len(v):>9,.0f} · worst {min(v):>10,.0f}')
