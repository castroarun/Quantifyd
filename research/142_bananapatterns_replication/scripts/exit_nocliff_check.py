"""Pre-declared NO-CLIFF check (Arun 2026-09-03): under the adopted no-gate /
16-slot spec, is the incumbent exit (trail-SMA20, -8% stop) within noise of the
local best? Grid: trail {15,20,25} x stop {6%,8%,10%}, 30 seeds, PAIRED deltas
vs the incumbent. Decision rule (declared before running): incumbent stands
unless a cell beats it by >1pp median paired dCAGR on >20/30 seeds.
This is a sanity check, NOT an optimization - re-tuning stays with the Dec-12
joint restudy.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

print('loading frames ...', flush=True)
w = br.load_frames('2004-06-01', trail_sma=20)
close, high, open_, athcp, sma20, tv20 = (w[k] for k in
    ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
etf = [c for c in close.columns if br.ETF_RE.search(c)]
tv_prev = tv20.shift(1)
prev_close = close.shift(1)
elig = tv_prev >= br.TV_FLOOR
elig[etf] = False
score = 2*(close/close.shift(63)-1) + (close/close.shift(126)-1) \
    + (close/close.shift(189)-1) + (close/close.shift(252)-1)
rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
dates = close.index
C, H, O, ATH = close.values, high.values, open_.values, athcp.values
RSv, TVv = rs.values, tv_prev.values
days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= '2006-01-01'])
wk = np.zeros(len(dates), dtype=bool)
n_yrs = len(days) / 247.0

S = {n: close.rolling(n).mean().values for n in (15, 20, 25)}

res = {}
for trail in (15, 20, 25):
    for stop in (0.06, 0.08, 0.10):
        cagrs, dds = [], []
        for seed in range(1, 31):
            eq, _, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH,
                                   S[trail], RSv, TVv, trig, wk, True, 0.0025,
                                   stop=stop, slots=16, size_pct=0.0625)
            eq = np.asarray(eq, dtype=float)
            cagrs.append(((eq[-1]/eq[0])**(1/n_yrs)-1)*100)
            dds.append(float((eq/np.maximum.accumulate(eq)-1).min()*100))
        res[(trail, stop)] = (np.array(cagrs), np.array(dds))
        print(f'trail{trail}/stop{int(stop*100)} done', flush=True)

inc_c, inc_d = res[(20, 0.08)]
print(f'\n{"cell":16s} {"CAGR med":>8s} {"worst":>6s} {"dd med":>7s} | '
      f'{"paired dCAGR":>12s} {"wins":>6s}')
for (trail, stop), (cg, dd) in sorted(res.items()):
    dc = cg - inc_c
    tag = ' <- INCUMBENT' if (trail, stop) == (20, 0.08) else ''
    print(f'trail{trail}/stop{int(stop*100):2d}   {np.median(cg):8.1f} {cg.min():6.1f} '
          f'{np.median(dd):7.1f} | {np.median(dc):+12.2f} {int((dc > 0).sum()):4d}/30{tag}',
          flush=True)
print('\nDecision rule: incumbent stands unless a cell shows median paired dCAGR '
      '>+1.0pp on >20/30 seeds.', flush=True)
pd.DataFrame([dict(trail=t, stop=s, cagr_med=round(float(np.median(cg)), 2),
                   cagr_worst=round(float(cg.min()), 2),
                   dd_med=round(float(np.median(dd)), 1),
                   paired_dcagr_med=round(float(np.median(cg - inc_c)), 2),
                   wins_vs_incumbent=int(((cg - inc_c) > 0).sum()))
              for (t, s), (cg, dd) in sorted(res.items())]).to_csv(
    STUDY / 'results' / 'exit_nocliff_check.csv', index=False)
print('DONE', flush=True)
