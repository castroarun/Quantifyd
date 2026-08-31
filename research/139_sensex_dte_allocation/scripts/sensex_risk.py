"""The complete designed risk on SENSEX, every weekday, suite AND the CSL/COMB sleeves.

Arun: "designed (book stop fires) -Rs18,000 im ok with this risk... but are u
talking ab the suite + comb (50%) for sensex included?"

No — that figure was the 9:16 suite ONLY. This adds the CSL/COMB side and states
the total per day. It also has to handle an unresolved question head-on, because
it changes the answer:

  is_live_book() in csl_paper_exec.py requires B.get("mode") == "live". ONLY
  NAS_COMB20 (a NIFTY book) carries that flag. By the code, NO SENSEX CSL/COMB
  book is live. But CSL_TIMEB_SENSEX's own comment says "2026-08-18: 6L->8L REAL
  (notional parity w/ NIFTY TB@8L)".

So the honest answer is two numbers, not one. Both are computed below.

One thing that must not be glossed: the portfolio book stop explicitly covers the
three 9:16 systems ONLY — app.py: "Sleeves (COMB / TimeB) are separate books with
their own combined-premium stops and are NOT covered by the book stop above."
So a live TimeB sleeve's risk does NOT sit under the -Rs3,000/lot expiry cap; it
is bounded only by its own combined-premium stop, applied to ITS entry premium at
ITS entry time.

Read-only.
"""
from __future__ import annotations

import json, re, sqlite3, statistics as st, sys
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
LOT = 20
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
WD2DTE = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}
BACKSTOP = 0.50           # executor's disaster backstop for SL "none"

cfg = json.loads((ROOT / 'backtest_data' / 'csl_paper_config.json').read_text())['books']
src = (ROOT / 'research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py').read_text()
matrix = json.loads((ROOT / 'backtest_data' / 'nas_day_matrix.json').read_text())['systems']
study = json.loads((ROOT / 'static' / 'app' / 'sensex_options_study.json').read_text())

# median combined premium at each entry clock, per DTE — so a % stop can be priced
def med_premium(dte, hhmm):
    vals = []
    for d in study['days']:
        if d['dte'] != dte:
            continue
        p = next((p for h, p in d['series'] if h >= hhmm), None)
        if p:
            vals.append(p)
    return st.median(vals) if vals else None


# ── the SENSEX CSL/COMB roster ────────────────────────────────────────────
books = {}
for m in re.finditer(r'"([A-Z0-9_]+)":\s*\{\*\*SENSEX_MKT[^}]*?"lots":\s*(\d+),\s*"qty":\s*(\d+)([^}]*)',
                     src):
    bk, lots, qty, rest = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    books[bk] = dict(lots=lots, qty=qty, live='"mode": "live"' in rest)

print('SENSEX CSL/COMB BOOKS — flag says live?')
for bk, b in books.items():
    print(f'  {bk:22} lots {b["lots"]:>2}  mode={"LIVE" if b["live"] else "paper"}  '
          f'cells: {",".join("DTE"+k for k in sorted(cfg.get(bk) or {}))}')
print('\n  NOTE: CSL_TIMEB_SENSEX\'s inline comment asserts REAL money ("6L->8L REAL",')
print('  2026-08-18) but carries NO "mode": "live" flag, so is_live_book() reads it as')
print('  PAPER. Unresolved — ops review 2026-09-05. Both readings are priced below.')

# ── suite exposure per day ────────────────────────────────────────────────
suite = {}
for k, db in [('sensex_atm', 'sensex_atm_trading.db'), ('sensex_atm2', 'sensex_atm2_trading.db'),
              ('sensex_atm4', 'sensex_atm4_trading.db')]:
    ld = {int(x) for x, v in (matrix[k].get('dte') or {}).items() if v} if matrix[k].get('live') else set()
    c = sqlite3.connect(f'file:{ROOT}/backtest_data/{db}?mode=ro', uri=True)
    r = c.execute('SELECT lots FROM nas_atm_trades WHERE exit_time IS NOT NULL '
                  'ORDER BY id DESC LIMIT 1').fetchone()
    c.close()
    suite[k] = (ld, r[0] if r else 0)

print('\n' + '=' * 92)
print('DESIGNED WORST CASE PER DAY — what each control caps the loss at')
print('=' * 92)
print(f"  {'day':4} {'DTE':>4} {'suite lots':>11} {'book stop/lot':>14} {'suite risk':>12}"
      f" {'CSL live':>10} {'CSL risk':>11} {'TOTAL':>12}")
print('  ' + '-' * 88)

tot_code = tot_comment = 0
for wd in range(5):
    dte = WD2DTE[wd]
    sl = sum(lots for ld, lots in suite.values() if dte in ld)
    per_lot = 3000.0 if dte == 0 else 1300.0
    srisk = sl * per_lot

    # CSL sleeves with a cell on this DTE
    csl_code = csl_comment = 0
    detail = []
    for bk, b in books.items():
        cell = (cfg.get(bk) or {}).get(str(dte))
        if not cell:
            continue
        lots = cell.get('lots', b['lots'])
        s = cell.get('sl')
        pct = BACKSTOP if s in (None, 'none') else (float(s) / 100.0 if not str(s).startswith('rs') else None)
        prem = med_premium(dte, cell.get('entry', '09:16'))
        if pct is None or prem is None:
            continue
        risk = pct * prem * LOT * lots
        detail.append(f'{bk[:16]} {lots}L@{int(pct*100)}%={risk:,.0f}')
        if b['live']:
            csl_code += risk
        csl_comment += risk if bk in ('CSL_TIMEB_SENSEX', 'CSL_TIMEB2_LIVE') or b['live'] else 0

    tot_code += srisk + csl_code
    tot_comment += srisk + csl_comment
    print(f'  {DOW[wd]:4} {dte:>4} {sl:>11} {per_lot:>14,.0f} {srisk:>12,.0f} '
          f'{csl_code:>10,.0f} {csl_comment:>11,.0f} {srisk+csl_comment:>12,.0f}')
    if detail:
        print(f'        CSL cells: {" · ".join(detail)}')

print('  ' + '-' * 88)
print(f'  {"WEEK":4} {"":>4} {"":>11} {"":>14} {"":>12} {tot_code:>10,.0f} '
      f'{tot_comment:>11,.0f} {tot_comment:>12,.0f}')

print('\n' + '=' * 92)
print('THE ANSWER TO THE QUESTION')
print('=' * 92)
thu_suite = sum(lots for ld, lots in suite.values() if 0 in ld) * 3000.0
print(f'  SENSEX expiry day (Thursday), 9:16 suite only ....... Rs{thu_suite:>10,.0f}')
print(f'    ^ this is the Rs18,000 figure. It does NOT include any COMB/TimeB sleeve.')
print(f'\n  IF the code is right (no SENSEX CSL book is live):')
print(f'    SENSEX expiry-day designed risk ................... Rs{thu_suite:>10,.0f}')
print(f'    whole SENSEX week ................................. Rs{tot_code + sum(sum(lots for ld,lots in suite.values() if WD2DTE[w] in ld)*(3000.0 if WD2DTE[w]==0 else 1300.0) for w in range(5)):>10,.0f}')
print(f'\n  IF the comments are right (TimeB SENSEX is REAL at 6 lots):')
tb = (cfg.get('CSL_TIMEB_SENSEX') or {}).get('0')
if tb:
    lots = tb.get('lots', books['CSL_TIMEB_SENSEX']['lots'])
    prem = med_premium(0, tb.get('entry', '13:00'))
    add = BACKSTOP * prem * LOT * lots
    print(f'    + TimeB SENSEX DTE0: {lots} lots, enters {tb.get("entry")}, median premium '
          f'{prem:.0f} pts, 50% backstop')
    print(f'      = Rs{add:,.0f}  -- and this is NOT under the -Rs3,000/lot book stop')
    print(f'        (app.py: "COMB / TimeB ... are NOT covered by the book stop above")')
    print(f'    SENSEX expiry-day designed risk ................... Rs{thu_suite+add:>10,.0f}')
print('\n  So the answer depends on a flag we have not resolved. That is exactly why the')
print('  2026-09-05 ops review matters: it changes the expiry-day number materially.')
