# -*- coding: utf-8 -*-
"""research/172 Phase 2 part C - the decisive controls for the HONEST winner.

GATE OVERRIDE, recorded deliberately. The automatic ranking picked
TRAIL10_T63_g0_BK20 (E189 Calmar 0.606) and the pre-registered +0.05 gate then skipped
the nulls. Two reasons that is the wrong place to stop:

  * the automatic winner is a TRAP. Its neighbour at a -15% book kill returns -1.06% CAGR,
    it collapses to -7.47% at 30 bps a side, and one of its twelve start-offsets gives
    Calmar -0.009. It fails the pre-registered plateau and cost clauses.
  * the honest winner, TRAIL20_T63_g0, is stable on both entries and all 24 offsets and
    lifts CAGR by ~1.7pp over 52W OPT. Whether it beats the momentum-matched null is the
    question the whole study turns on, and it costs 60 cheap cells to answer.

So the nulls, the cost ladder and the blend are run for TRAIL20_T63_g0 even though its
Calmar margin over 52W OPT (+0.030 on E189, +0.047 on E252) is inside the skip threshold.
This is disclosed in RESULTS.md, not quietly done.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path('/home/arun/quantifyd/research/172_52wk_channel_n100')
sys.path.insert(0, str(HERE / 'scripts'))
RES = HERE / 'results'

import p172                     # noqa: E402
import bt172 as B               # noqa: E402
import bt172b as B2             # noqa: E402
import run172d as D             # noqa: E402

WIN = 'TRAIL20_T63_g0'
STACK = dict(trail_pct=0.20, time_bars=63, time_min_gain=0.0)


def main():
    P = p172.Panel()
    t0 = time.time()
    days = P.days['full']
    grp = 'time_stop'

    # ---- cost ladder on the honest winner, both entries
    path = RES / 'stops.csv'
    done = D.done_labels(path)
    for cost in (0.0, 30.0, 45.0):
        for tag, L in D.ENTRIES:
            lab = '%s__%s_c%d' % (WIN, tag, int(cost))
            if lab in done:
                continue
            row, _ = D.run_stack(P, WIN, grp, STACK, tag, L, cost=cost)
            D.append(path, row)
            print('  [cost] %s %s %dbps cagr=%6.2f dd=%7.2f calmar=%.3f'
                  % (WIN, tag, int(cost), row['cagr'], row['maxdd'], row['calmar']),
                  flush=True)

    # ---- momentum-matched nulls N3 / N4 on the winner's own exit stack
    npath = RES / 'p2_nulls.csv'
    ndone = D.done_labels(npath)
    up = (P.ST['ST_14_4'] == 1)
    ush = np.zeros_like(up)
    ush[1:] = up[:-1]
    out = {}
    for tag, L in D.ENTRIES:
        trig = D.trig_for(P, L)
        elig = P.eligible(D.UNI, L)
        esh = np.zeros_like(elig)
        esh[1:] = elig[:-1]
        rs = np.where(esh, P.RS252, np.nan)
        with np.errstate(invalid='ignore'):
            med = np.nanmedian(rs, axis=1)
            strong = np.zeros_like(elig)
            strong[1:] = (elig & (P.RS252 >= med[:, None]))[:-1]
        pools = {'N3mom': strong, 'N4trendmom': esh & ush & strong}
        for nm, pool in pools.items():
            key = 'P2%s_%s' % (nm, tag)
            for s in range(1, 31):
                lab = '%s_%02d' % (key, s)
                if lab in ndone:
                    continue
                rng = np.random.default_rng(90000 + s)
                null = np.zeros_like(trig)
                nper = trig.sum(axis=1)
                for i in np.nonzero(nper)[0]:
                    pl = np.nonzero(pool[i])[0]
                    if not len(pl):
                        continue
                    null[i, rng.choice(pl, size=min(int(nper[i]), len(pl)),
                                       replace=False)] = True
                cfg = dict(days=days, slots=20, cost_bps=15.0, gate=None, seed=None,
                           tax=True, exit_arr=None, **STACK)
                r = B2.simulate_stack(P, null, cfg)
                m = B.metrics(r['nav'], r['dates'], r['trades'])
                row = dict(label=lab, stack=key, entry=tag, entry_L=L, group='p2null',
                           cost_bps=15.0, window='full')
                row.update(m)
                D.append(npath, row)
            sub = pd.read_csv(npath)
            sub = sub[sub.label.str.startswith(key)]
            out[key] = dict(n=len(sub), cagr_med=round(float(sub.cagr.median()), 2),
                            cagr_min=round(float(sub.cagr.min()), 2),
                            cagr_max=round(float(sub.cagr.max()), 2),
                            calmar_med=round(float(sub.calmar.median()), 3),
                            calmar_min=round(float(sub.calmar.min()), 3),
                            calmar_max=round(float(sub.calmar.max()), 3))
            print('  [null] %-16s n=%d cagr med %.2f [%.2f..%.2f] calmar med %.3f'
                  % (key, out[key]['n'], out[key]['cagr_med'], out[key]['cagr_min'],
                     out[key]['cagr_max'], out[key]['calmar_med']), flush=True)
    json.dump(out, open(RES / 'p2_null_summary.json', 'w'), indent=1)
    print('DONE p2 part C in %.1f min' % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
