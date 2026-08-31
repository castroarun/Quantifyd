"""SENSEX per-DTE study — the same question research/138 answered for NIFTY.

Arun: "like we did for Nifty nas + comb portfolio, can we do for sensex? similar
studies and recommendation for each dte".

Construction held IDENTICAL to the NIFTY grid so the two can be read side by side:
sell the ATM straddle at 09:16, hold to 15:20, one COMBINED-premium stop, sliced by
DTE. Built from sensex_options_study.json (92 recorded days, symbol='SENSEX').

Three things that are NOT the same as NIFTY and must be kept straight:

  * SENSEX expiry is THURSDAY, so the weekday map differs:
        Mon=DTE3  Tue=DTE2  Wed=DTE1  Thu=DTE0  Fri=DTE4
  * lot is 20 (NIFTY 65). "10 lots" here is qty 200, NOT the same notional as 10
    NIFTY lots (qty 650). Per-DTE comparison INSIDE SENSEX is exact; against NIFTY
    it is indicative.
  * the live SENSEX CSL books run WINDOWS (TimeB DTE0 13:00-15:20, DTE1 10:30-12:00),
    not the full day. So this grid answers "what does the full-day held straddle do
    per DTE on SENSEX" — the question comparable to NIFTY — and is NOT a model of the
    live TimeB windows. The 9:16 suite (sensex_atm/atm2/atm4) IS a 09:16 book, so the
    grid is directly relevant to the days those trade.

The stopless column is a CONTROL to detect an inert stop, never a candidate
(Arun, 2026-08-31: "having no stop loss cannot be a recommendation"). It is
labelled as such in every table.

Read-only.
"""
from __future__ import annotations

import json, sqlite3, statistics as st, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOT, LOTS = 20, 10
QTY = LOT * LOTS            # 200
COST = 160 * (QTY / 650)    # scale NIFTY's round-trip cost by qty -> ~49
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
WD2DTE = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}
STOPS = [15.0, 20.0, 25.0, 30.0, 40.0, None]     # None = CONTROL only

study = json.loads((ROOT / 'static' / 'app' / 'sensex_options_study.json').read_text())
days = [d for d in study['days'] if d['dte'] <= 4]
print(f"SENSEX ATM straddle — {study['n_days']} recorded days, {len(days)} on the DTE0-4 scale")
print(f"lot {LOT} x {LOTS} lots = qty {QTY} · round-trip cost Rs{COST:.0f}\n")


def run(day, sl):
    bars = [(h, p) for h, p in day['series'] if '09:16' <= h <= '15:20']
    if len(bars) < 5:
        return None, False
    ent = bars[0][1]
    thr = (1 + sl / 100.0) * ent if sl is not None else None
    for h, p in bars:
        if thr is not None and p >= thr:
            return round((ent - p) * QTY - COST), True
    return round((ent - bars[-1][1]) * QTY - COST), False


cells = defaultdict(lambda: defaultdict(list))
for d in days:
    for sl in STOPS:
        pnl, stopped = run(d, sl)
        if pnl is not None:
            cells[d['dte']][sl].append((d['date'], pnl, stopped))


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


def tstat(v):
    return (st.mean(v) / (st.stdev(v) / len(v) ** 0.5)) if len(v) > 2 and st.stdev(v) else 0.0


dtes = sorted(cells)
hdr = ''.join(f"{('SL'+str(int(s)) if s else 'nostop*'):>11}" for s in STOPS)


def grid(title, fn):
    print(f'\n{title}')
    print(f"{'DTE':>4} {'day':>4} {'n':>4}{hdr}")
    print('-' * (13 + 11 * len(STOPS)))
    for k in dtes:
        ds = cells[k][STOPS[0]]
        wd = DOW[date.fromisoformat(ds[0][0]).weekday()]
        line = f'{k:>4} {wd:>4} {len(ds):>4}'
        for s in STOPS:
            line += fn([x[1] for x in cells[k][s]], cells[k][s])
        print(line)


grid('NET P&L BY DTE x STOP  (* stopless = control, never a recommendation)',
     lambda v, g: f'{sum(v):>11,.0f}')
grid('t  (2.0 is the 1-in-20 bar; %d cells tested here)' % (len(dtes) * len(STOPS)),
     lambda v, g: f'{tstat(v):>11.2f}')
grid('MAX DRAWDOWN', lambda v, g: f'{dd(v):>11,.0f}')
grid('STOPS FIRED  (n/N — an all-zero row means the stop is INERT on that day)',
     lambda v, g: f'{sum(1 for x in g if x[2]):>7}/{len(g):<3}')

print('\n' + '=' * 78)
print('STABILITY — does each cell hold in BOTH halves of the sample?')
print('=' * 78)
print(f"{'DTE':>4} {'day':>4} {'stop':>6} {'1st half':>13} {'2nd half':>13} {'verdict':>10}")
print('-' * 60)
for k in dtes:
    for s in (20.0, 30.0):
        rec = sorted(cells[k][s])
        h = len(rec) // 2
        a = sum(x[1] for x in rec[:h]); b = sum(x[1] for x in rec[h:])
        wd = DOW[date.fromisoformat(rec[0][0]).weekday()]
        v = 'both +' if a > 0 and b > 0 else ('both -' if a < 0 and b < 0 else 'flips')
        print(f'{k:>4} {wd:>4} {int(s):>6} {a:>13,.0f} {b:>13,.0f} {v:>10}')

# ── what SENSEX actually runs live, per DTE ───────────────────────────────
print('\n' + '=' * 78)
print('WHAT SENSEX RUNS TODAY, AND WHAT THE GRID SAYS ABOUT THAT DAY')
print('=' * 78)
matrix = json.loads((ROOT / 'backtest_data' / 'nas_day_matrix.json').read_text())['systems']
cfg = json.loads((ROOT / 'backtest_data' / 'csl_paper_config.json').read_text())['books']
live_dtes = {int(k) for k, v in (matrix['sensex_atm'].get('dte') or {}).items() if v}
for wd in range(5):
    k = WD2DTE[wd]
    suite = '9:16 suite LIVE (3 sleeves)' if k in live_dtes else 'suite dark'
    sleeves = [b for b in ('CSL_TIMEB_SENSEX', 'CSL30F_SENSEX', 'CSL30F_SENSEX_WED')
               if str(k) in (cfg.get(b) or {})]
    best = max(cells[k], key=lambda s: tstat([x[1] for x in cells[k][s]]) if s else -9)
    v30 = [x[1] for x in cells[k][30.0]]
    print(f'  {DOW[wd]:4} DTE{k}  {suite:28} CSL: {",".join(s[:14] for s in sleeves) or "-":32}')
    print(f'          grid @SL30: net {sum(v30):>9,.0f} · t {tstat(v30):>5.2f} · '
          f'DD {dd(v30):>9,.0f} · stops {sum(1 for x in cells[k][30.0] if x[2])}/{len(v30)}')

# ── live SENSEX record per DTE, for comparison ───────────────────────────
print('\n' + '=' * 78)
print('THE LIVE SENSEX RECORD PER DTE (9:16 suite, real money, real lots)')
print('=' * 78)
live = defaultdict(lambda: defaultdict(float))
for key, db in [('sensex_atm', 'sensex_atm_trading.db'), ('sensex_atm2', 'sensex_atm2_trading.db'),
                ('sensex_atm4', 'sensex_atm4_trading.db')]:
    ld = {int(x) for x, v in (matrix[key].get('dte') or {}).items() if v}
    c = sqlite3.connect(f'file:{ROOT}/backtest_data/{db}?mode=ro', uri=True)
    for d, pnl, lots in c.execute("SELECT trade_date, net_pnl, lots FROM nas_atm_trades "
                                  "WHERE exit_time IS NOT NULL AND net_pnl IS NOT NULL"):
        if d and lots and date.fromisoformat(d).weekday() < 5:
            k = WD2DTE[date.fromisoformat(d).weekday()]
            live['LIVE' if k in ld else 'shadow'][k] += float(pnl)
    c.close()
for tag in ('LIVE', 'shadow'):
    print(f'  {tag}:')
    for k in sorted(live[tag]):
        wd = [DOW[w] for w in range(5) if WD2DTE[w] == k][0]
        print(f'    DTE{k} {wd}  Rs{live[tag][k]:>10,.0f}')
