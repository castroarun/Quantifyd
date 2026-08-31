"""Does the 75% recommendation survive the tail? Arun: "risk is going to be very high right".

He is right to push, and the check that matters is not in my 18-day sample at all.
research/118 measured the SENSEX expiry-day tail over 127 DTE0 days:

    worst day approximately -Rs21,500 PER LOT, and 8.7% of DTE0 days worse than -500 pts

My grid's worst day at 75% was -Rs64,399 at 10 lots = **-Rs6,440/lot**. So the
measured tail is **3.3x worse than the worst thing in my sample**, and my sample
almost certainly does not contain a 1-in-127 event at all. Optimising a stop width
on 18 days and calling the widest one "better on drawdown too" is exactly the error
of reading a tail off a sample too small to hold it.

So this asks the question properly:

  1. what does each stop width COST on a normal day (the mean argument I made)
  2. what does each stop width SAVE on a tail day (the argument I skipped)
  3. what is the entry premium actually, so the % widths can be read in rupees
  4. how far did price actually travel on the worst days in the sample

Read-only.
"""
from __future__ import annotations

import json, statistics as st, sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
LOT, LOTS = 20, 10
QTY = LOT * LOTS                       # 200
R118_WORST_PER_LOT = 21500.0           # research/118, 127 DTE0 days

sx = json.loads((ROOT / 'static' / 'app' / 'sensex_options_study.json').read_text())
d0 = [d for d in sx['days'] if d['dte'] == 0]

ents = [d['series'][0][1] for d in d0]
print(f'SENSEX EXPIRY DAY (DTE0) — {len(d0)} days in the recorded sample\n')
print(f'  entry combined premium: median {st.median(ents):.0f} pts · '
      f'mean {st.mean(ents):.0f} · range {min(ents):.0f}-{max(ents):.0f}')

print('\nWHAT EACH STOP WIDTH MEANS IN RUPEES (at the median entry premium)')
med = st.median(ents)
print(f"  {'width':>7} {'pts of adverse move':>21} {'loss/lot':>11} {'loss @10 lots':>15}")
for w in (20, 25, 30, 40, 50, 60, 75, 100):
    pts = med * w / 100.0
    print(f'  {w:>6}% {pts:>21,.0f} {pts*LOT:>11,.0f} {pts*QTY:>15,.0f}')
print(f'\n  For scale, research/118\'s measured worst DTE0 day is Rs{R118_WORST_PER_LOT:,.0f}/lot '
      f'= Rs{R118_WORST_PER_LOT*LOTS:,.0f} at 10 lots,')
print(f'  which is {R118_WORST_PER_LOT/LOT:,.0f} POINTS of adverse move — '
      f'{R118_WORST_PER_LOT/LOT/med*100:,.0f}% of the median entry premium.')
print('  Every width tested here fires FAR below that, so on a true tail day the stop')
print('  does engage. The question is what it costs to engage 25 points later.')

# ── the actual adverse excursion distribution in-sample ───────────────────
print('\nHOW FAR THE COMBINED PREMIUM ACTUALLY TRAVELLED, worst-first')
rows = []
for d in d0:
    ent = d['series'][0][1]
    hi = max(p for _, p in d['series'])
    rows.append((d['date'], ent, hi, 100 * (hi / ent - 1)))
rows.sort(key=lambda r: -r[3])
print(f"  {'date':12} {'entry':>8} {'peak':>8} {'peak rise':>10}   caught by")
for dt, ent, hi, pc in rows[:8]:
    caught = [str(w) + '%' for w in (20, 25, 30, 40, 50, 60, 75, 100) if pc >= w]
    print(f'  {dt:12} {ent:>8.0f} {hi:>8.0f} {pc:>9.1f}%   '
          f'{", ".join(caught) if caught else "none — never breached 20%"}')
print(f'  ... {len(rows)-8} quieter days omitted')

# ── the honest comparison: mean vs tail, 50 vs 75 ────────────────────────
print('\n' + '=' * 76)
print('50% vs 75% — the full trade-off, not just the mean')
print('=' * 76)


def run(days, w):
    out = []
    for d in days:
        ent = d['series'][0][1]
        thr = (1 + w / 100.0) * ent if w else None
        hit = next((p for _, p in d['series'] if thr and p >= thr), None)
        out.append(round((ent - (hit if hit is not None else d['series'][-1][1])) * QTY - 49))
    return out


a, b = run(d0, 50), run(d0, 75)
print(f"  {'':22} {'50% (live)':>14} {'75% (proposed)':>16}")
print('  ' + '-' * 54)
print(f"  {'net':22} {sum(a):>14,.0f} {sum(b):>16,.0f}")
print(f"  {'worst day IN SAMPLE':22} {min(a):>14,.0f} {min(b):>16,.0f}")
print(f"  {'2nd worst':22} {sorted(a)[1]:>14,.0f} {sorted(b)[1]:>16,.0f}")
print(f"  {'3rd worst':22} {sorted(a)[2]:>14,.0f} {sorted(b)[2]:>16,.0f}")
print(f"  {'sum of losing days':22} {sum(x for x in a if x<0):>14,.0f} "
      f"{sum(x for x in b if x<0):>16,.0f}")
print(f"  {'losing days':22} {sum(1 for x in a if x<0):>14} {sum(1 for x in b if x<0):>16}")
mx = med * 0.50 * QTY, med * 0.75 * QTY
print(f"  {'designed loss at stop':22} {-mx[0]:>14,.0f} {-mx[1]:>16,.0f}   "
      f"<- the number that scales with the tail")
print(f"  {'  ...as % of the other':22} {'':>14} {mx[1]/mx[0]:>15.2f}x")

print('\n  THE POINT: the extra Rs1.3L of net at 75% is earned by NOT exiting on days')
print('  that reverted. The price of that is a designed per-event loss 1.5x larger,')
print('  on a day type whose measured worst case (research/118, 127 days) is 3.3x')
print('  anything in these 18. In-sample the wider stop looks better on drawdown too —')
print('  but that is precisely the statistic a short sample cannot measure.')

print('\n  A tail day at each width, using research/118\'s -Rs21,500/lot as the event:')
print(f"  {'width':>7} {'stopped out at':>16} {'vs holding through':>20}")
for w in (50, 60, 75):
    loss = med * w / 100.0 * LOT
    print(f'  {w:>6}% {-loss*LOTS:>16,.0f} {(R118_WORST_PER_LOT-loss)*LOTS:>19,.0f} saved')
print('\n  So all three widths still protect against the measured tail. The difference')
print(f'  between 50% and 75% on such a day is Rs{(med*0.25*LOT)*LOTS:,.0f} at 10 lots —')
print('  real, but an order of magnitude smaller than the tail itself. The stop width')
print('  is NOT what governs tail risk here. SIZE is.')
