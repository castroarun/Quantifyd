"""SENSEX expiry day — where does the gamma damage actually stop?

research/139 found every stop from 15% to 40% costs ~70% of the DTE0 cell, and
even SL40 still fires on 7 of 18 expiry Thursdays. The live book already runs a
deliberately-wide 50% disaster backstop there. The open question is whether 50%
is the right width or just a round number: this walks the stop out until it stops
doing damage, so the backstop is chosen on evidence.

To be explicit about what is and is not being asked: **the stopless column is a
CONTROL, not a candidate** (Arun: "having no stop loss cannot be a recommendation").
The deliverable is a WIDTH — the narrowest stop that no longer sabotages the cell —
not permission to remove one. A stop that never fires in-sample is still doing its
job: it bounds the tail on the day the sample did not contain.

Also reported, because a width chosen only on mean P&L is half an answer:
  * how many times each width fires (a width that fires often is still in the way)
  * the worst single day at each width (what the stop actually BUYS you)
  * the same walk for NIFTY's expiry day (DTE0/Tuesday) as a cross-check, since
    the live NIFTY book runs SL25 there and the same gamma logic should apply

Read-only.
"""
from __future__ import annotations

import json, statistics as st, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

WIDTHS = [20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, None]


def dd(v):
    cum = peak = w = 0.0
    for x in v:
        cum += x; peak = max(peak, cum); w = min(w, cum - peak)
    return w


def tstat(v):
    return (st.mean(v) / (st.stdev(v) / len(v) ** 0.5)) if len(v) > 2 and st.stdev(v) else 0.0


def walk(title, days, qty, cost, lo='09:16', hi='15:20'):
    print('\n' + '=' * 96)
    print(title)
    print('=' * 96)
    print(f"  {'stop':>8} {'net':>12} {'mean/day':>10} {'t':>6} {'maxDD':>12} "
          f"{'worst day':>12} {'fires':>8} {'vs 50%':>12}")
    print('  ' + '-' * 88)
    res = {}
    for s in WIDTHS:
        vals, fires = [], 0
        for d in days:
            bars = [(h, p) for h, p in d['series'] if lo <= h <= hi]
            if len(bars) < 5:
                continue
            ent = bars[0][1]
            thr = (1 + s / 100.0) * ent if s is not None else None
            hit = None
            for h, p in bars:
                if thr is not None and p >= thr:
                    hit = p
                    break
            if hit is not None:
                vals.append(round((ent - hit) * qty - cost)); fires += 1
            else:
                vals.append(round((ent - bars[-1][1]) * qty - cost))
        res[s] = vals
    base = sum(res[50.0])
    for s in WIDTHS:
        v = res[s]
        fires = 0
        for d, val in zip(days, v):
            pass
        # recount fires cleanly
        fires = 0
        for d in days:
            bars = [(h, p) for h, p in d['series'] if lo <= h <= hi]
            if len(bars) < 5:
                continue
            ent = bars[0][1]
            if s is not None and any(p >= (1 + s / 100.0) * ent for _, p in bars):
                fires += 1
        tag = 'nostop*' if s is None else f'{int(s)}%'
        mark = '   <- LIVE backstop' if s == 50.0 else ''
        print(f"  {tag:>8} {sum(v):>12,.0f} {sum(v)/len(v):>10,.0f} {tstat(v):>6.2f} "
              f"{dd(v):>12,.0f} {min(v):>12,.0f} {fires:>4}/{len(v):<3} "
              f"{sum(v)-base:>+12,.0f}{mark}")
    return res


# ── SENSEX expiry day ─────────────────────────────────────────────────────
sx = json.loads((ROOT / 'static' / 'app' / 'sensex_options_study.json').read_text())
sx0 = [d for d in sx['days'] if d['dte'] == 0]
walk(f'SENSEX EXPIRY DAY (DTE0 / Thursday) — {len(sx0)} days, 10 lots (qty 200), full day',
     sx0, 200, 49)

print('\n  Reading: the column that matters is "fires". A stop that still fires on a third')
print('  of expiry days is being tripped by gamma, not by a real loss. Find the width')
print('  where firing collapses AND the net stops improving — that is the backstop.')

# ── NIFTY expiry day, as a cross-check ───────────────────────────────────
nf = json.loads((ROOT / 'static' / 'app' / 'options_study.json').read_text())
nf0 = [{'series': [(b[0], b[1]) for b in (d.get('series') or []) if b[1]], 'dte': d['dte']}
       for d in nf['days'] if d['dte'] == 0]
nf0 = [d for d in nf0 if d['series']]
walk(f'NIFTY EXPIRY DAY (DTE0 / Tuesday) — {len(nf0)} days, 10 lots (qty 650), full day '
     f'[live NAS_COMB20 runs SL25 here]', nf0, 650, 160)

print('\n  Cross-check purpose: if the gamma effect is real it should appear on BOTH')
print('  venues on their own expiry day. If NIFTY DTE0 shows no such pattern, then the')
print('  SENSEX result is about SENSEX (thinner book, wider spreads), not about expiry.')
