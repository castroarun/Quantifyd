# -*- coding: utf-8 -*-
"""research/164 — slot-count / position-size / contested-slot sweep on the adopted
Open Alpha - Base Age spec.

Usage:
    python3 run164.py --stage=proof                  harness reproduction at 5.5% and 5.0%
    python3 run164.py --stage=scan   --seeds=10      all cells, 10 seeds
    python3 run164.py --stage=full   --seeds=30      all cells, 30 seeds
    python3 run164.py --stage=full   --seeds=30 --only=A_,D_
    python3 run164.py --stage=cost   --seeds=10 --bps=40  cost rung on the shortlist

Writes INCREMENTALLY: one row per (cell, seed) to results/seedstats_<stage>.csv and one
summary row per cell to results/cells_<stage>.csv, both appended the moment a cell
finishes. Already-finished cells are skipped on a restart.
"""
import csv
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
sys.path.insert(0, str(HERE))
import sim164 as S                                                   # noqa: E402

W1 = ('2005-01-03', '2015-12-31')
W2 = ('2016-01-01', '2026-12-31')
BASE_IDLE = 0.05

PANEL = EVENTS = CAL = None


# --------------------------------------------------------------------- the cell grid
def grid():
    """(label, axis, slots, slot_pct, select, idle, cost_bps) with duplicates folded."""
    seen, out = {}, []

    def add(label, axis, slots, pct, sel, idle=BASE_IDLE, bps=25.0):
        key = (slots, round(pct, 6), sel, idle, bps)
        if key in seen:
            seen[key][1].append(label)
            return
        row = dict(label=label, axis=axis, slots=slots, slot_pct=round(pct, 6),
                   select=sel, idle=idle, cost_bps=bps)
        seen[key] = (row, [])
        row['_alias'] = seen[key][1]
        out.append(row)

    # A. concentration, fully invested: slot_pct = 1/slots
    for s in (6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 24, 30):
        add('A_s%02d_eq' % s, 'A', s, 1.0 / s, 'random')
    # B. fixed 6.25% size, deliberate cash buffer
    for s in (8, 10, 12, 14, 16):
        add('B_s%02d_p0625' % s, 'B', s, 0.0625, 'random')
    # C. size independent of slot count (drop anything above 100% invested)
    for s in (10, 16, 20):
        for p in (0.04, 0.05, 0.0625, 0.08, 0.10):
            if s * p <= 1.0 + 1e-9:
                add('C_s%02d_p%04d' % (s, round(p * 10000)), 'C', s, p, 'random')
    # D. who wins a contested slot (coupled to A: fully invested at 1/slots)
    for sel in ('rs', 'ext', 'tv', 'age'):
        for s in (8, 16):
            add('D_%s_s%02d' % (sel, s), 'D', s, 1.0 / s, sel)
    return out


def proof_grid():
    return [dict(label='PROOF_055', axis='P', slots=16, slot_pct=0.0625, select='random',
                 idle=0.055, cost_bps=25.0, _alias=[]),
            dict(label='PROOF_050', axis='P', slots=16, slot_pct=0.0625, select='random',
                 idle=0.05, cost_bps=25.0, _alias=[])]


# --------------------------------------------------------------------- worker
def _one(args):
    cell, seed = args
    cfg = dict(exit='ST_14_4', hard_stop=False, time_stop=0, cost_bps=cell['cost_bps'],
               gate_ok=None, idle_yield=cell['idle'], slots=cell['slots'],
               slot_pct=cell['slot_pct'], select=cell['select'])
    nav, tr, inv, book = S.simulate(EVENTS, PANEL, cfg, seed)
    m = S.metrics(nav, CAL, tr)
    m['invested_pct'] = round(100.0 * float(np.nanmean(inv)), 2)
    m['w1_cagr'] = S.window_cagr(nav, CAL, *W1)
    m['w1_dd'] = S.window_dd(nav, CAL, *W1)
    m['w2_cagr'] = S.window_cagr(nav, CAL, *W2)
    m['w2_dd'] = S.window_dd(nav, CAL, *W2)
    m.update(book)
    # outlier dependence: total multiple with and without the ten best trades
    t = pd.DataFrame(tr)
    m['mult_all'] = round(float((1 + t.ret_pct / 100).prod()), 3)
    m['mult_drop10'] = round(
        float((1 + t.drop(t.nlargest(10, 'ret_pct').index).ret_pct / 100).prod()), 3)
    return seed, nav.astype(np.float32), m


SEED_FIELDS = ['label', 'axis', 'slots', 'slot_pct', 'select', 'idle', 'cost_bps', 'seed',
               'cagr', 'maxdd', 'calmar', 'sharpe', 'trades', 'trades_per_yr', 'win_rate',
               'avg_win', 'avg_loss', 'expectancy', 'max_loss_streak', 'invested_pct',
               'w1_cagr', 'w1_dd', 'w2_cagr', 'w2_dd', 'days_signal', 'days_bind',
               'days_full', 'turned_away', 'cash_refused', 'entries_taken', 'pos_rs_med', 'cap_pct_med', 'cap_pct_p95',
               'cap_over1pct', 'mult_all', 'mult_drop10']
CELL_FIELDS = (['label', 'axis', 'slots', 'slot_pct', 'select', 'idle', 'cost_bps',
                'seeds', 'alias', 'n_events', 'cagr_med', 'cagr_min', 'cagr_max',
                'maxdd_med', 'maxdd_worst', 'calmar_med', 'calmar_min', 'sharpe_med']
               + ['%s_med' % k for k in ('trades', 'trades_per_yr', 'win_rate', 'avg_win',
                                         'avg_loss', 'expectancy', 'max_loss_streak',
                                         'invested_pct', 'w1_cagr', 'w1_dd', 'w2_cagr',
                                         'w2_dd', 'days_bind', 'days_full', 'turned_away',
                                         'cash_refused', 'entries_taken', 'pos_rs_med', 'cap_pct_med', 'cap_pct_p95',
                                         'cap_over1pct', 'mult_all', 'mult_drop10')]
               + ['med_seed', 'secs'])


def main():
    global PANEL, EVENTS, CAL
    stage, seeds_n, workers, only, bps = 'scan', 10, 2, None, None
    idle_override = None
    for a in sys.argv[1:]:
        if a.startswith('--stage='):
            stage = a.split('=', 1)[1]
        elif a.startswith('--seeds='):
            seeds_n = int(a.split('=', 1)[1])
        elif a.startswith('--workers='):
            workers = int(a.split('=', 1)[1])
        elif a.startswith('--only='):
            only = a.split('=', 1)[1].split(',')
        elif a.startswith('--bps='):
            bps = float(a.split('=', 1)[1])
        elif a.startswith('--idle='):
            idle_override = float(a.split('=', 1)[1])
    t0 = time.time()
    seeds = list(range(1, seeds_n + 1))

    PANEL = pickle.load(open(RES / 'panel164.pkl', 'rb'))
    CAL = PANEL.cal
    ev = pd.read_csv(RES / 'events164.csv')
    EVENTS = ev.to_dict('records')
    for e in EVENTS:
        e['entry_i'] = int(e['entry_i'])
    print('panel %d syms / %d days (%s .. %s);  %d events'
          % (len(PANEL.close), PANEL.n, CAL[0], CAL[-1], len(EVENTS)), flush=True)

    cells = proof_grid() if stage == 'proof' else grid()
    if bps is not None:
        for c in cells:
            c['cost_bps'] = bps
            c['label'] = '%s_bps%d' % (c['label'], int(bps))
    if idle_override is not None:
        for c in cells:
            c['idle'] = idle_override
            c['label'] = '%s_idle%03d' % (c['label'], round(idle_override * 1000))
    if only:
        cells = [c for c in cells if any(c['label'].startswith(p) for p in only)]
    print('stage=%s  cells=%d  seeds=%d  workers=%d' % (stage, len(cells), len(seeds),
                                                        workers), flush=True)

    scsv, ccsv = RES / ('seedstats_%s.csv' % stage), RES / ('cells_%s.csv' % stage)
    done = set()
    if ccsv.exists():
        with open(ccsv) as f:
            done = {r['label'] for r in csv.DictReader(f)}
        print('resuming: %d cells already done' % len(done), flush=True)
    else:
        with open(scsv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=SEED_FIELDS).writeheader()
        with open(ccsv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=CELL_FIELDS).writeheader()

    pool = Pool(workers) if workers > 1 else None
    navdir = RES / ('navs_%s' % stage)
    navdir.mkdir(exist_ok=True)
    for ci, cell in enumerate(cells, 1):
        if cell['label'] in done:
            continue
        ct = time.time()
        jobs = [(cell, sd) for sd in seeds]
        res = pool.map(_one, jobs) if pool else [_one(j) for j in jobs]
        res.sort(key=lambda r: r[0])
        navs = np.vstack([r[1] for r in res])
        ms = [r[2] for r in res]
        with open(scsv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SEED_FIELDS, extrasaction='ignore')
            for (sd, _, m) in res:
                row = {k: cell.get(k) for k in
                       ('label', 'axis', 'slots', 'slot_pct', 'select', 'idle', 'cost_bps')}
                row['seed'] = sd
                row.update(m)
                w.writerow(row)
        cg = np.array([m['cagr'] for m in ms], float)
        med_i = int(np.argsort(cg)[len(cg) // 2])
        np.savez_compressed(navdir / ('%s.npz' % cell['label']), navs=navs,
                            seeds=np.array(seeds), med_seed=seeds[med_i],
                            dates=np.array(CAL))
        row = {k: cell.get(k) for k in
               ('label', 'axis', 'slots', 'slot_pct', 'select', 'idle', 'cost_bps')}
        row.update(seeds=len(seeds), alias='|'.join(cell.get('_alias', [])),
                   n_events=len(EVENTS),
                   cagr_med=round(float(np.median(cg)), 2), cagr_min=round(float(cg.min()), 2),
                   cagr_max=round(float(cg.max()), 2),
                   maxdd_med=round(float(np.median([m['maxdd'] for m in ms])), 2),
                   maxdd_worst=round(float(min(m['maxdd'] for m in ms)), 2),
                   calmar_med=round(float(np.median([m['calmar'] for m in ms])), 3),
                   calmar_min=round(float(min(m['calmar'] for m in ms)), 3),
                   sharpe_med=round(float(np.median([m['sharpe'] for m in ms])), 3),
                   med_seed=seeds[med_i], secs=round(time.time() - ct, 1))
        for k in ('trades', 'trades_per_yr', 'win_rate', 'avg_win', 'avg_loss', 'expectancy',
                  'max_loss_streak', 'invested_pct', 'w1_cagr', 'w1_dd', 'w2_cagr', 'w2_dd',
                  'days_bind', 'days_full', 'turned_away', 'cash_refused', 'entries_taken', 'pos_rs_med', 'cap_pct_med',
                  'cap_pct_p95', 'cap_over1pct', 'mult_all', 'mult_drop10'):
            v = [m.get(k, np.nan) for m in ms]
            row['%s_med' % k] = round(float(np.nanmedian(v)), 3)
        with open(ccsv, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=CELL_FIELDS, extrasaction='ignore').writerow(row)
        print('[%2d/%2d] %-16s slots=%2d pct=%.4f sel=%-6s | CAGR %6.2f%% [%5.2f..%5.2f] '
              'DD %7.2f%% Calmar %5.3f inv %5.1f%% bind %4d streak %2d | %.0fs'
              % (ci, len(cells), cell['label'], cell['slots'], cell['slot_pct'],
                 cell['select'], row['cagr_med'], row['cagr_min'], row['cagr_max'],
                 row['maxdd_med'], row['calmar_med'], row['invested_pct_med'],
                 row['days_bind_med'], row['max_loss_streak_med'], row['secs']), flush=True)
    if pool:
        pool.close(); pool.join()
    print('\nstage %s done in %.0fs' % (stage, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
