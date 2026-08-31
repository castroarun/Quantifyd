"""What the LIVE book — real money only — actually did, and what it is leaving on the table.

Every portfolio view I have produced this session pooled live systems with paper
shadows. That was right for comparing constructions on equal footing, which is
what Arun asked for, but it cannot answer "what should change in the live book",
because most of what it measured is not the live book.

The roster, taken from the two files the executors actually gate on rather than
from any doc:

  backtest_data/nas_day_matrix.json   -> live: true and which DTEs are enabled
  research/111/scripts/csl_paper_exec.py BOOKS -> "mode": "live"

which gives, as of today:

  nas_916_atm / atm2 / atm4    NIFTY   live on DTE 0,1  = Tue, Mon
  sensex_atm / atm2 / atm4     SENSEX  live on DTE 0,1  = Thu, Wed
  NAS_COMB20                   NIFTY   live, 2 lots, DTE 0,1 = Tue, Mon

  nas_atm / atm2 / atm4        paper_shadow, all DTEs  -> NOT real money
  every other CSL/COMB book    paper                   -> NOT real money

The nas_*_trading.db tables hold BOTH the live trades and the all-DTE shadow in
one place, so a live trade is identified the same way the executor identifies
it: by the DTE its weekday maps to. Anything on a non-live DTE is shadow.

Two questions:
  1. what has the real-money book made, at its real size, and at what drawdown
  2. the live gates are Mon/Tue on NIFTY and Wed/Thu on SENSEX — what do the
     shadows say about the days nobody is live on

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

# weekday -> DTE, straight from csl_paper_exec.py's NIFTY_MKT / SENSEX_MKT
WD2DTE = {'NIFTY': {0: 1, 1: 0, 2: 4, 3: 3, 4: 2},
          'SENSEX': {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}}

matrix = json.loads((ROOT / 'backtest_data' / 'nas_day_matrix.json').read_text())['systems']

NAS = [('nas_916_atm', 'nas_916_atm_trading.db', 'NIFTY'),
       ('nas_916_atm2', 'nas_916_atm2_trading.db', 'NIFTY'),
       ('nas_916_atm4', 'nas_916_atm4_trading.db', 'NIFTY'),
       ('sensex_atm', 'sensex_atm_trading.db', 'SENSEX'),
       ('sensex_atm2', 'sensex_atm2_trading.db', 'SENSEX'),
       ('sensex_atm4', 'sensex_atm4_trading.db', 'SENSEX'),
       ('nas_atm', 'nas_atm_trading.db', 'NIFTY'),
       ('nas_atm2', 'nas_atm2_trading.db', 'NIFTY'),
       ('nas_atm4', 'nas_atm4_trading.db', 'NIFTY')]


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


def tstat(v):
    return (st.mean(v) / (st.stdev(v) / len(v) ** 0.5)) if len(v) > 2 and st.stdev(v) else 0.0


live_daily, shadow_daily = defaultdict(float), defaultdict(float)
per_sleeve = {}
print('THE LIVE ROSTER, as the executors gate it')
print(f"{'system':16} {'venue':7} {'live':>5} {'live DTEs':>10} {'live days':>10} {'lots':>5}")
print('-' * 62)

for key, db, venue in NAS:
    p = ROOT / 'backtest_data' / db
    if not p.exists():
        continue
    row = matrix.get(key, {})
    is_live = bool(row.get('live'))
    live_dtes = {int(k) for k, v in (row.get('dte') or {}).items() if v} if is_live else set()
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    L, S, lots_seen = defaultdict(float), defaultdict(float), set()
    for d, pnl, lots in c.execute(
            "SELECT trade_date, net_pnl, lots FROM nas_atm_trades "
            "WHERE exit_time IS NOT NULL AND net_pnl IS NOT NULL"):
        if not d or not lots:
            continue
        wd = date.fromisoformat(d).weekday()
        if wd > 4:
            continue
        lots_seen.add(int(lots))
        (L if WD2DTE[venue][wd] in live_dtes else S)[d] += float(pnl)
    c.close()
    days = sorted(L)
    print(f'{key:16} {venue:7} {("yes" if is_live else "no"):>5} '
          f'{",".join(str(x) for x in sorted(live_dtes)) or "-":>10} {len(days):>10} '
          f'{",".join(str(x) for x in sorted(lots_seen)):>5}')
    per_sleeve[key] = (dict(L), dict(S), venue, is_live)
    for d, v in L.items():
        live_daily[d] += v
    for d, v in S.items():
        shadow_daily[d] += v

# COMB20 — real money, at its real 2 lots
comb = defaultdict(float)
for r in json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())['records']:
    if r['book'] != 'NAS_COMB20' or r.get('pnl') is None:
        continue
    comb[r['day'][:10]] += float(r['pnl'])
for d, v in comb.items():
    live_daily[d] += v
print(f'{"NAS_COMB20":16} {"NIFTY":7} {"yes":>5} {"0,1":>10} {len(comb):>10} {2:>5}')

ld = sorted(live_daily)
lv = [live_daily[d] for d in ld]
print(f'\n{"="*70}\nREAL MONEY ONLY — {len(ld)} trading days · {ld[0]} -> {ld[-1]}\n{"="*70}')
print(f'  net P&L          {sum(lv):>12,.0f}')
print(f'  max drawdown     {dd(lv):>12,.0f}')
print(f'  return / DD      {sum(lv)/abs(dd(lv)):>12.2f}')
print(f'  mean / day       {sum(lv)/len(lv):>12,.0f}')
print(f'  t on daily P&L   {tstat(lv):>12.2f}')
print(f'  green days       {sum(1 for x in lv if x>0)}/{len(lv)} '
      f'({100*sum(1 for x in lv if x>0)/len(lv):.0f}%)')
print(f'  best / worst     {max(lv):>12,.0f} / {min(lv):,.0f}')
t3 = sorted(lv, reverse=True)[:3]
print(f'  top 3 days       {sum(t3):>12,.0f}  = {100*sum(t3)/sum(lv):.0f}% of the total')

print('\nREAL-MONEY BOOK BY WEEKDAY  (NIFTY is live Mon/Tue · SENSEX live Wed/Thu · nobody Fri)')
wd = defaultdict(list)
for d in ld:
    wd[date.fromisoformat(d).weekday()].append(live_daily[d])
for k in range(5):
    if k in wd:
        v = wd[k]
        print(f'  {DOW[k]}  {len(v):>3}d · net {v and sum(v):>11,.0f} · mean {sum(v)/len(v):>9,.0f} · '
              f'worst {min(v):>10,.0f} · t {tstat(v):>5.2f}')

print('\nPER-SLEEVE, REAL MONEY ONLY (live DTEs only, real lots)')
print(f"{'system':16} {'days':>5} {'net':>11} {'mean':>9} {'maxDD':>11} {'t':>6}")
print('-' * 62)
for key, (L, S, venue, is_live) in per_sleeve.items():
    if not is_live or not L:
        continue
    v = [L[d] for d in sorted(L)]
    print(f'{key:16} {len(v):>5} {sum(v):>11,.0f} {sum(v)/len(v):>9,.0f} {dd(v):>11,.0f} '
          f'{tstat(v):>6.2f}')
cv = [comb[d] for d in sorted(comb)]
print(f'{"NAS_COMB20":16} {len(cv):>5} {sum(cv):>11,.0f} {sum(cv)/len(cv):>9,.0f} '
      f'{dd(cv):>11,.0f} {tstat(cv):>6.2f}')

# ── the days nobody is live on ─────────────────────────────────────────────
print(f'\n{"="*70}\nTHE DAYS THE LIVE BOOK SITS OUT — what the shadows recorded there\n{"="*70}')
print('  (paper shadow of the same three NIFTY constructions, all DTEs, at their own lots)')
sd = sorted(shadow_daily)
swd = defaultdict(list)
for d in sd:
    swd[date.fromisoformat(d).weekday()].append(shadow_daily[d])
print(f"  {'day':5} {'n':>4} {'net':>12} {'mean':>10} {'worst':>11} {'t':>6}  live?")
for k in range(5):
    if k in swd:
        v = swd[k]
        nlive = 'NIFTY live' if k in (0, 1) else ('SENSEX live' if k in (2, 3) else 'NOBODY LIVE')
        print(f'  {DOW[k]:5} {len(v):>4} {sum(v):>12,.0f} {sum(v)/len(v):>10,.0f} {min(v):>11,.0f} '
              f'{tstat(v):>6.2f}  {nlive}')
