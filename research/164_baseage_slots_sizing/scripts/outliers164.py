# -*- coding: utf-8 -*-
"""research/164 — honest outlier dependence for the shortlist.

research/161's `outlier_all` multiplied every trade's return together, which is not a book
multiple and reads as 1e20x. The right questions are (a) what share of the book's total
RUPEE profit came from its ten best trades, and (b) what happens if those ten events are
DELETED from the event list and the whole book is re-run on all 30 seeds.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
sys.path.insert(0, str(HERE))
import sim164 as S                                                   # noqa: E402

SEEDS = list(range(1, 31))
CELLS = {}


def cfg_of(r):
    return dict(exit='ST_14_4', hard_stop=False, time_stop=0, cost_bps=25.0, gate_ok=None,
                idle_yield=0.05, slots=int(r.slots), slot_pct=float(r.slot_pct),
                select=str(r.select))


def main():
    panel = pickle.load(open(RES / 'panel164.pkl', 'rb'))
    cal = panel.cal
    ev = pd.read_csv(RES / 'events164.csv')
    events = ev.to_dict('records')
    for e in events:
        e['entry_i'] = int(e['entry_i'])
    cells = pd.read_csv(RES / 'cells_full.csv').set_index('label', drop=False)
    rep = json.load(open(RES / 'final164.json'))
    labs = [rep['baseline']] + list(rep['shortlist'])

    out, lines = {}, ['## OA — Base Age · outlier dependence: delete the ten best trades', '',
                      '| cell | total book profit | top-10 trades\' share of it | CAGR (all '
                      'events) | CAGR (top-10 events DELETED, 30 seeds) | cost |',
                      '|---|---|---|---|---|---|']
    for lab in labs:
        r = cells.loc[lab]
        cfg = cfg_of(r)
        ms = int(r.med_seed)
        nav, tr, _, _ = S.simulate(events, panel, cfg, ms)
        t = pd.DataFrame(tr)
        top = t.nlargest(10, 'pnl')
        tot = float(t.pnl.sum())
        share = 100.0 * float(top.pnl.sum()) / tot if tot else np.nan
        kill = {(a, b) for a, b in zip(top.symbol, top.entry_date)}
        ev2 = [e for e in events
               if (e['symbol'], cal[e['entry_i']]) not in kill]
        cg_all = [S.metrics(S.simulate(events, panel, cfg, sd)[0], cal)['cagr'] for sd in SEEDS]
        cg_cut = [S.metrics(S.simulate(ev2, panel, cfg, sd)[0], cal)['cagr'] for sd in SEEDS]
        a, b = float(np.median(cg_all)), float(np.median(cg_cut))
        out[lab] = dict(total_pnl=round(tot), top10_share_pct=round(share, 1),
                        cagr_all=round(a, 2), cagr_drop10=round(b, 2), cost_pp=round(a - b, 2),
                        med_seed=ms, events_removed=len(events) - len(ev2))
        lines.append('| %s | ₹%s | %.1f%% | %.2f%% | %.2f%% | **%+.2f pp** |'
                     % (lab, format(int(tot), ','), share, a, b, b - a))
        print(lines[-1], flush=True)
    lines += ['', '*The ten best trades are identified on the median-seed path, then those ten '
              'EVENTS are removed from the event list and the whole book is re-run on all 30 '
              'seeds — so the slots they occupied are freed for whatever else qualified that '
              'day. That is a fair deletion, not a bookkeeping subtraction.*', '']
    open(RES / 'outliers164.md', 'w', encoding='utf-8').write('\n'.join(lines))
    json.dump(out, open(RES / 'outliers164.json', 'w'), indent=1)
    print('\nwrote outliers164.md')


if __name__ == '__main__':
    main()
