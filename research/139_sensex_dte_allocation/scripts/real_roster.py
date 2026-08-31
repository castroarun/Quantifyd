"""The live roster rebuilt from EVIDENCE, not from the mode flag. The flag was wrong.

I established the live roster earlier from is_live_book() — B.get("mode") == "live"
— which reads only NAS_COMB20 as live. The executor's own event log contradicts that
flatly: CSL_TIMEB_SENSEX and CSL_TIMEB_NIFTY both emit events with source="REAL" and
"[REAL MONEY]" in the message, including fills and P&L.

Evidence beats the flag. An event that says a straddle was sold at 8 lots for real
money, with a cumulative P&L that tracks, is a record of a trade; a missing dict key
is not evidence of its absence. So this rebuilds the roster from what the executor
RECORDED ITSELF DOING, and re-prices the exposure that follows.

Consequences to correct:
  * my "live book = 7 sleeves, Rs2,38,557 over 43 days" OMITTED both TimeB books
  * my SENSEX expiry-day risk of Rs18,000 counted only the 9:16 suite
  * TimeB SENSEX runs 8 lots on Thursday (not the 6 in the config's base "lots"),
    and app.py is explicit that TimeB is NOT covered by the -Rs3,000/lot book stop

Read-only.
"""
from __future__ import annotations

import json, re, sqlite3, statistics as st, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
WD2DTE = {'NIFTY': {0: 1, 1: 0, 2: 4, 3: 3, 4: 2}, 'SENSEX': {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}}

state = json.loads((ROOT / 'backtest_data' / 'csl_paper_state.json').read_text())
events = state.get('events') or []

# ── 1. which books have REAL-money evidence, and at what size ─────────────
real = defaultdict(lambda: dict(entries=0, exits=0, lots=set(), days=set(), pnl=None))
for e in events:
    if e.get('source') != 'REAL':
        continue
    msg = e.get('msg', '')
    if '[REAL MONEY]' not in msg:
        continue
    b = real[e['book']]
    b['days'].add(e['ts'][:10])
    if e.get('type') == 'ENTRY':
        b['entries'] += 1
    if e.get('type') == 'EXIT':
        b['exits'] += 1
    m = re.search(r'\((\d+) lots', msg)
    if m:
        b['lots'].add(int(m.group(1)))
    m = re.search(r'cum ([+-]?\d+)', msg)
    if m:
        b['pnl'] = int(m.group(1))

print('BOOKS WITH REAL-MONEY EVIDENCE IN THE EXECUTOR\'S OWN EVENT LOG')
print(f"  {'book':24} {'entries':>8} {'exits':>6} {'lots seen':>12} {'days':>5} {'cum P&L':>10}")
print('  ' + '-' * 70)
tot = 0
for bk in sorted(real):
    b = real[bk]
    print(f'  {bk:24} {b["entries"]:>8} {b["exits"]:>6} '
          f'{",".join(str(x) for x in sorted(b["lots"])):>12} {len(b["days"]):>5} '
          f'{b["pnl"] if b["pnl"] is not None else "-":>10}')
    tot += b['pnl'] or 0
print(f'  {"":24} {"":>8} {"":>6} {"":>12} {"":>5} {tot:>10}')

src = (ROOT / 'research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py').read_text()
flagged = re.findall(r'"([A-Z0-9_]+)":\s*\{\*\*(?:NIFTY|SENSEX)_MKT[^}]*?"mode":\s*"live"', src)
print(f'\n  books carrying the "mode": "live" FLAG : {flagged}')
print(f'  books with REAL-MONEY EVIDENCE          : {sorted(real)}')
missing = sorted(set(real) - set(flagged))
print(f'\n  *** {len(missing)} book(s) trade real money WITHOUT the flag: {missing}')
print('  is_live_book() is therefore NOT the operative gate for these books, and any')
print('  audit that trusts it (including mine, earlier today) understates live risk.')

# ── 2. re-price SENSEX expiry-day exposure with the real roster ───────────
cfg = json.loads((ROOT / 'backtest_data' / 'csl_paper_config.json').read_text())['books']
study = json.loads((ROOT / 'static' / 'app' / 'sensex_options_study.json').read_text())


def med_prem(dte, hhmm):
    v = [p for d in study['days'] if d['dte'] == dte
         for p in [next((q for h, q in d['series'] if h >= hhmm), None)] if p]
    return st.median(v) if v else None


matrix = json.loads((ROOT / 'backtest_data' / 'nas_day_matrix.json').read_text())['systems']
suite_lots = 0
for k, db in [('sensex_atm', 'sensex_atm_trading.db'), ('sensex_atm2', 'sensex_atm2_trading.db'),
              ('sensex_atm4', 'sensex_atm4_trading.db')]:
    if matrix[k].get('live') and (matrix[k].get('dte') or {}).get('0'):
        c = sqlite3.connect(f'file:{ROOT}/backtest_data/{db}?mode=ro', uri=True)
        r = c.execute('SELECT lots FROM nas_atm_trades WHERE exit_time IS NOT NULL '
                      'ORDER BY id DESC LIMIT 1').fetchone()
        c.close()
        suite_lots += r[0] if r else 0

print('\n' + '=' * 78)
print('SENSEX EXPIRY DAY (Thursday) — DESIGNED RISK, CORRECTED')
print('=' * 78)
suite_risk = suite_lots * 3000.0
print(f'  9:16 suite            {suite_lots} lots x Rs3,000/lot book stop = Rs{suite_risk:>10,.0f}')

tb = (cfg.get('CSL_TIMEB_SENSEX') or {}).get('0')
tb_lots = max(real['CSL_TIMEB_SENSEX']['lots']) if real.get('CSL_TIMEB_SENSEX') else 0
tb_risk = 0.0
if tb and tb_lots:
    prem = med_prem(0, tb.get('entry', '13:00'))
    slc = tb.get('sl')
    pct = 0.50 if slc in (None, 'none') else float(slc) / 100.0
    tb_risk = pct * prem * 20 * tb_lots
    print(f'  CSL_TIMEB_SENSEX      {tb_lots} lots, enters {tb.get("entry")}, median premium '
          f'{prem:.0f} pts, {int(pct*100)}% backstop = Rs{tb_risk:>10,.0f}')
    print(f'                        NOT covered by the book stop (app.py: "COMB / TimeB ... '
          f'are NOT covered")')
print(f'  {"":22}{"":26} TOTAL  Rs{suite_risk + tb_risk:>10,.0f}')
print(f'\n  Previously reported: Rs{suite_risk:,.0f}. Corrected: '
      f'Rs{suite_risk + tb_risk:,.0f} — {(suite_risk+tb_risk)/suite_risk:.1f}x higher.')

# ── 3. the corrected live book P&L ───────────────────────────────────────
print('\n' + '=' * 78)
print('THE CORRECTED LIVE BOOK (adding the two TimeB books I had excluded)')
print('=' * 78)
add = {b: real[b]['pnl'] for b in real if b not in ('NAS_COMB20',)}
print(f'  previously reported live net (7 sleeves, 43 days) .... Rs 238,557')
for b, p in sorted(add.items()):
    print(f'  + {b:36} Rs{p:>10,.0f}')
print(f'  {"":38} {"":>10}')
print(f'  corrected live net .................................. Rs'
      f'{238557 + sum(add.values()):>10,.0f}')
print('\n  (NAS_COMB20 was already included at -Rs10,089.)')
