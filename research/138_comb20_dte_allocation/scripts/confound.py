"""research/138 phase 2 — is Thursday good, or is 30% good on Thursday?

Phase 1 sliced ONE stop level (30%) by DTE and found Thursday best by a distance:
+Rs 1,55,265 over 18 days, t 3.85, zero stops. But the live book already trades
Thursday — NAS_COMB20_THU, DTE3, at SL 20 and half size. So the phase-1 result
does not on its own say "trade Thursday"; it might be saying "30% is the right
stop on Thursday", or it might be saying "Thursday is good and the stop is
irrelevant". Those imply opposite actions, so they have to be separated.

The separation is cheap because the underlying source — static/app/options_study.json —
holds the FULL untruncated 5-min ATM straddle premium path for every recorded day.
v1_sl30.json is only one stop level rendered from it. Re-render the grid and the
question answers itself.

Mechanic held identical to sl30_journeys.py so the numbers stay comparable:
  sell the ATM straddle at the first bar >= 09:20, exit once the combined premium
  rises SL% above entry, else hold to the last bar. qty 650 (10 lots x 65),
  Rs 160 round-trip cost.

Caveat stated up front: this is the V1 entry clock (09:20, hold to close), not
COMB20's exact 09:16 -> 15:20. On matched Mondays the two correlate 0.96, so it
is a good proxy for the day/stop question — but it is a proxy, and a stop level
chosen here would still want confirming on COMB20's own clock before deploy.

Controls, because a 5x5 grid on ~18 days per cell is exactly where false
positives live:
  * a no-stop column, so "does any stop help at all" is visible
  * first-half / second-half split per cell, so a cell that only works in one
    period is exposed rather than averaged away
  * the count of cells tested is stated, so the t-values can be read against it

Read-only.
"""
from __future__ import annotations

import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

QTY, COST = 650, 160          # 10 lots x 65, round-trip cost — same as sl30_journeys.py
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
STOPS = [15.0, 20.0, 25.0, 30.0, 40.0, None]   # None = hold to close, no stop

study = json.loads((ROOT / 'static' / 'app' / 'options_study.json').read_text())


def run(day, sl):
    """One day at one stop level -> (pnl, stopped). None = no stop."""
    bars = [(b[0], b[1]) for b in (day.get('series') or []) if b[1]]
    bars = [(h, p) for h, p in bars if h >= '09:20']
    if len(bars) < 2:
        return None, False
    ent = bars[0][1]
    thr = (1 + sl / 100.0) * ent if sl is not None else None
    for h, p in bars:
        if thr is not None and p >= thr:
            return round((ent - p) * QTY - COST), True
    return round((ent - bars[-1][1]) * QTY - COST), False


days = sorted(study['days'], key=lambda x: x['date'])
cells = defaultdict(lambda: defaultdict(list))     # dte -> sl -> [(date, pnl, stopped)]
for dy in days:
    for sl in STOPS:
        pnl, stopped = run(dy, sl)
        if pnl is not None:
            cells[dy['dte']][sl].append((dy['date'], pnl, stopped))


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


def tstat(v):
    return (st.mean(v) / (st.stdev(v) / len(v) ** 0.5)) if len(v) > 2 and st.stdev(v) else 0.0


dtes = sorted(cells)
print('THE CONFOUND TEST — net P&L by DTE x stop level  (10 lots, recorded chain)')
print(f"  {len(days)} recorded days · {len(dtes)} DTEs x {len(STOPS)} stops = "
      f"{len(dtes)*len(STOPS)} cells tested\n")
hdr = ''.join(f"{('SL'+str(int(s)) if s else 'no stop'):>12}" for s in STOPS)
print(f"{'DTE':>4} {'day':>4} {'n':>4}{hdr}")
print('-' * (14 + 12 * len(STOPS)))
for k in dtes:
    ds = cells[k][STOPS[0]]
    wd = DOW[__import__('datetime').date.fromisoformat(ds[0][0]).weekday()] if ds else '?'
    line = f'{k:>4} {wd:>4} {len(ds):>4}'
    for s in STOPS:
        line += f'{sum(x[1] for x in cells[k][s]):>12,.0f}'
    print(line)

print('\nSAME GRID AS t  (t>=2 is the 1-in-20 bar; with 30 cells tested expect ~1.5 by luck)')
print(f"{'DTE':>4} {'day':>4}{hdr}")
print('-' * (9 + 12 * len(STOPS)))
for k in dtes:
    ds = cells[k][STOPS[0]]
    wd = DOW[__import__('datetime').date.fromisoformat(ds[0][0]).weekday()] if ds else '?'
    line = f'{k:>4} {wd:>4}'
    for s in STOPS:
        line += f'{tstat([x[1] for x in cells[k][s]]):>12.2f}'
    print(line)

print('\nSAME GRID AS max drawdown')
print(f"{'DTE':>4} {'day':>4}{hdr}")
print('-' * (9 + 12 * len(STOPS)))
for k in dtes:
    ds = cells[k][STOPS[0]]
    wd = DOW[__import__('datetime').date.fromisoformat(ds[0][0]).weekday()] if ds else '?'
    line = f'{k:>4} {wd:>4}'
    for s in STOPS:
        line += f'{dd([x[1] for x in cells[k][s]]):>12,.0f}'
    print(line)

# ── the actual question ────────────────────────────────────────────────────
print('\n' + '=' * 78)
print('Q1  THURSDAY (DTE3): is it the day, or the 30% stop?')
print('=' * 78)
for s in STOPS:
    v = [x[1] for x in cells[3][s]]
    hits = sum(1 for x in cells[3][s] if x[2])
    tag = ('  <- LIVE: NAS_COMB20_THU runs this, at 5 lots' if s == 20.0 else
           '  <- phase-1 headline' if s == 30.0 else '')
    print(f'  SL {str(int(s)) if s else "none":>4}  net {sum(v):>10,.0f}  mean {sum(v)/len(v):>8,.0f}  '
          f't {tstat(v):>5.2f}  DD {dd(v):>10,.0f}  stops {hits}/{len(v)}{tag}')
print('  Reading: if the row is flat across every stop level, Thursday is the day and the')
print('  stop is doing nothing — the phase-1 headline was about Thursday, not about 30%.')

print('\n' + '=' * 78)
print('Q2  MONDAY (DTE1): the cell COMB20 runs at 30%, and where the live book bleeds')
print('=' * 78)
for s in STOPS:
    v = [x[1] for x in cells[1][s]]
    hits = sum(1 for x in cells[1][s] if x[2])
    tag = '  <- LIVE: NAS_COMB20 DTE1 runs this' if s == 30.0 else ''
    print(f'  SL {str(int(s)) if s else "none":>4}  net {sum(v):>10,.0f}  mean {sum(v)/len(v):>8,.0f}  '
          f't {tstat(v):>5.2f}  DD {dd(v):>10,.0f}  stops {hits}/{len(v)}{tag}')

print('\n' + '=' * 78)
print('Q3  STABILITY — does each cell hold up in BOTH halves of the sample?')
print('     (a cell that only works in one half is a period artefact, not an edge)')
print('=' * 78)
print(f"{'DTE':>4} {'day':>4} {'stop':>6} {'1st half':>12} {'2nd half':>12} {'verdict':>12}")
print('-' * 62)
for k in dtes:
    for s in (20.0, 30.0, None):
        rec = sorted(cells[k][s])
        h = len(rec) // 2
        a = sum(x[1] for x in rec[:h]); b = sum(x[1] for x in rec[h:])
        wd = DOW[__import__('datetime').date.fromisoformat(rec[0][0]).weekday()]
        v = 'both +' if a > 0 and b > 0 else ('both -' if a < 0 and b < 0 else 'flips')
        print(f'{k:>4} {wd:>4} {(str(int(s)) if s else "none"):>6} {a:>12,.0f} {b:>12,.0f} {v:>12}')

# ── what the live allocation is actually worth vs alternatives ─────────────
print('\n' + '=' * 78)
print('Q4  THE LIVE NIFTY ALLOCATION vs alternatives, on this same chain')
print('     COMB20 DTE0@25 + DTE1@30, COMB20_FRI DTE2@30, COMB20_THU DTE3@20 (5 lots)')
print('=' * 78)


def book(alloc):
    """alloc: {dte: (stop, lot_scale)} -> pooled daily P&L series."""
    per = defaultdict(float)
    for k, (s, scale) in alloc.items():
        for d, p, _ in cells[k][s]:
            per[d] += p * scale
    v = [per[d] for d in sorted(per)]
    return sum(v), dd(v), tstat(v), len(v)


LIVE = {0: (25.0, 1.0), 1: (30.0, 1.0), 2: (30.0, 1.0), 3: (20.0, 0.5)}
OPTS = [
    ('LIVE today', LIVE),
    ('drop DTE1 (Mon)', {k: v for k, v in LIVE.items() if k != 1}),
    ('DTE3 to full size', {**LIVE, 3: (20.0, 1.0)}),
    ('DTE3 to 30%, full size', {**LIVE, 3: (30.0, 1.0)}),
    ('drop DTE1 + DTE3 full @20', {0: (25.0, 1.0), 2: (30.0, 1.0), 3: (20.0, 1.0)}),
    ('drop DTE1 + DTE3 full @30', {0: (25.0, 1.0), 2: (30.0, 1.0), 3: (30.0, 1.0)}),
    ('DTE0+DTE3 only, @30', {0: (30.0, 1.0), 3: (30.0, 1.0)}),
    ('all five DTEs @30', {k: (30.0, 1.0) for k in dtes}),
]
print(f"{'allocation':30} {'net':>12} {'maxDD':>12} {'ret/DD':>8} {'t':>6} {'days':>5}")
print('-' * 78)
for name, a in OPTS:
    n, d_, t_, nd = book(a)
    print(f'{name:30} {n:>12,.0f} {d_:>12,.0f} {n/abs(d_) if d_ else 0:>8.2f} {t_:>6.2f} {nd:>5}')
