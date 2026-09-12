# -*- coding: utf-8 -*-
"""research/166 — honest outlier dependence for the shortlist: identify the ten best trades
on the median-seed path, DELETE those ten EVENTS from the event list, and re-run the whole
book on all 30 seeds so the slots they occupied are freed for whatever else qualified that
day. Same method as research/164's outliers164.py.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
ROOT = HERE.parent.parent.parent
PANEL_PKL = ROOT / 'research' / '164_baseage_slots_sizing' / 'results' / 'panel164.pkl'
sys.path.insert(0, str(HERE))
import sim166 as S                                                    # noqa: E402

SEEDS = list(range(1, 31))
EXTRA = ['BASE_rand', 'A_unre_m010', 'X_entrs_unre_m010', 'B2_trimD_k150',
         'C_unre010_k150d_f000', 'C_none000_k150d_f025']


def cfg_of(r):
    return dict(exit='ST_14_4', hard_stop=bool(r.get('hard_stop', 0)),
                hard_stop_pct=float(r.get('hard_stop_pct', 0.92)),
                rot_sell_only=bool(r.get('rot_sell_only', 0)), time_stop=0, cost_bps=25.0,
                gate_ok=None, idle_yield=float(r['idle']), slots=int(r['slots']),
                slot_pct=float(r['slot_pct']), select=str(r['select']),
                rot_score=(None if pd.isna(r['rot_score']) else str(r['rot_score'])),
                rot_margin=float(r['rot_margin']),
                rot_max_per_day=int(r['rot_max_per_day']),
                rot_entrant=str(r['rot_entrant']), trim_mult=float(r['trim_mult']),
                trim_when=str(r['trim_when']), min_fill_frac=float(r['min_fill_frac']))


def main():
    panel = pickle.load(open(PANEL_PKL, 'rb'))
    cal = panel.cal
    st = pickle.load(open(RES / 'st166.pkl', 'rb'))
    aux = S.build_aux(panel, st)
    ev = pd.read_csv(RES / 'events166.csv')
    events = ev.to_dict('records')
    for e in events:
        e['entry_i'] = int(e['entry_i'])
    cells = pd.read_csv(RES / 'cells_full.csv').set_index('label', drop=False)
    rep = json.load(open(RES / 'final166.json'))
    labs = []
    for l in [rep['baseline']] + list(rep.get('shortlist', [])) + EXTRA:
        if l in cells.index and l not in labs:
            labs.append(l)

    out, lines = {}, ['## OA — Base Age · outlier dependence: delete the ten best trades', '',
                      '| cell | total book profit | top-10 trades\' share of it | CAGR (all '
                      'events) | CAGR (top-10 events DELETED, 30 seeds) | cost |',
                      '|---|---|---|---|---|---|']
    for lab in labs:
        r = cells.loc[lab]
        cfg = cfg_of(r)
        ms = int(r.med_seed)
        nav, tr, _, _ = S.simulate(events, panel, aux, cfg, ms)
        t = pd.DataFrame(tr)
        top = t.nlargest(10, 'pnl')
        tot = float(t.pnl.sum())
        share = 100.0 * float(top.pnl.sum()) / tot if tot else np.nan
        kill = {(a, b) for a, b in zip(top.symbol, top.entry_date)}
        ev2 = [e for e in events if (e['symbol'], cal[e['entry_i']]) not in kill]
        cg_all = [S.metrics(S.simulate(events, panel, aux, cfg, sd)[0], cal)['cagr']
                  for sd in SEEDS]
        cg_cut = [S.metrics(S.simulate(ev2, panel, aux, cfg, sd)[0], cal)['cagr']
                  for sd in SEEDS]
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
              'day. That is a fair deletion, not a bookkeeping subtraction. Trims count as '
              'realisations here, which is why a trimming cell shows a lower concentration.*',
              '']
    open(RES / 'outliers166.md', 'w', encoding='utf-8').write('\n'.join(lines))
    json.dump(out, open(RES / 'outliers166.json', 'w'), indent=1)
    print('\nwrote outliers166.md')


if __name__ == '__main__':
    main()
