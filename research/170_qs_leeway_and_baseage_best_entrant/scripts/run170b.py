# -*- coding: utf-8 -*-
"""research/170 Part B -- confirm research/166's post-hoc "highest-RS entrant" pick.

The harness (sim170.py = research/166's sim166.py, byte-identical) and the inputs (the
research/164 panel, research/166's frozen event list and SuperTrend line cache) are reused
unchanged. Only the seed set and the cell list are new.

    python3 run170b.py --stage=fresh  --seeds=30 --seedbase=1000 --workers=2
    python3 run170b.py --stage=r166   --seeds=30 --seedbase=0    --workers=2
    python3 run170b.py --stage=plat_fresh --plateau=rs --seedbase=1000 --workers=2
    python3 run170b.py --stage=cost40_fresh --bps=40 --seedbase=1000 --workers=2

Writes incrementally: one row per (cell, seed) to results/seedstatsB_<stage>.csv and one
summary row per cell to results/cellsB_<stage>.csv. Finished cells are skipped on restart.
"""
import csv
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
ROOT = HERE.parent.parent.parent
PANEL_PKL = ROOT / 'research' / '164_baseage_slots_sizing' / 'results' / 'panel164.pkl'
R166 = ROOT / 'research' / '166_baseage_rotation_and_drift' / 'results'
sys.path.insert(0, str(HERE))
import sim170 as S                                                    # noqa: E402

W1 = ('2005-01-03', '2015-12-31')
W2 = ('2016-01-01', '2026-12-31')
BASE_IDLE = 0.052

PANEL = EVENTS = CAL = AUX = None


def _c(label, axis, **kw):
    row = dict(label=label, axis=axis, slots=16, slot_pct=0.0625, select='random',
               idle=BASE_IDLE, cost_bps=25.0, rot_score=None, rot_margin=0.0,
               rot_max_per_day=1, rot_entrant='tv', trim_mult=0.0, trim_when='month',
               min_fill_frac=0.0, rot_sell_only=0, hard_stop=0, hard_stop_pct=0.92)
    row.update(kw)
    return row


def grid(stage, plateau):
    """The PRE-REGISTERED cell list. Three selection cells (one per entrant priority),
    the incumbent, and the rate-matched random-swap null. The plateau stage adds the two
    margin neighbours of whichever entrant wins -- also pre-registered."""
    if stage.startswith('plat'):
        if plateau not in ('tv', 'rs', 'age'):
            raise SystemExit('--plateau=tv|rs|age required for the plateau stage')
        out = []
        for m in (7.5, 12.5):
            out.append(_c('X_ent%s_unre_m%03d' % (plateau, round(m * 10)), 'SEL_PLAT',
                          rot_score='unreal', rot_margin=m, rot_entrant=plateau))
        return out

    out = [
        # the incumbent: never swap. Not a selection cell -- the thing to beat.
        _c('BASE_rand', 'BASE'),
        # research/166's PRE-REGISTERED rotation winner: the freed slot goes to the most
        # liquid refused entrant.
        _c('A_unre_m010', 'SEL', rot_score='unreal', rot_margin=10.0, rot_entrant='tv'),
        # the POST-HOC pick under confirmation: highest 12-month relative strength.
        _c('X_entrs_unre_m010', 'SEL', rot_score='unreal', rot_margin=10.0,
           rot_entrant='rs'),
        # the third entrant priority: oldest base.
        _c('X_entage_unre_m010', 'SEL', rot_score='unreal', rot_margin=10.0,
           rot_entrant='age'),
        # the rate-matched null: swap a RANDOM holding at the same ~4 swaps a year.
        _c('A_null_p003', 'NULL', rot_score='rand', rot_margin=0.03),
    ]
    if stage.startswith('cost'):
        out = [c for c in out if c['label'] != 'A_null_p003']
    return out


def _one(args):
    cell, seed = args
    cfg = dict(exit='ST_14_4', hard_stop=bool(cell.get('hard_stop', 0)),
               hard_stop_pct=float(cell.get('hard_stop_pct', 0.92)),
               rot_sell_only=bool(cell.get('rot_sell_only', 0)), time_stop=0,
               cost_bps=cell['cost_bps'],
               gate_ok=None, idle_yield=cell['idle'], slots=cell['slots'],
               slot_pct=cell['slot_pct'], select=cell['select'],
               rot_score=cell['rot_score'], rot_margin=cell['rot_margin'],
               rot_max_per_day=cell['rot_max_per_day'], rot_entrant=cell['rot_entrant'],
               trim_mult=cell['trim_mult'], trim_when=cell['trim_when'],
               min_fill_frac=cell['min_fill_frac'])
    nav, tr, inv, book = S.simulate(EVENTS, PANEL, AUX, cfg, seed)
    m = S.metrics(nav, CAL, tr)
    m['invested_pct'] = round(100.0 * float(np.nanmean(inv)), 2)
    m['w1_cagr'] = S.window_cagr(nav, CAL, *W1)
    m['w1_dd'] = S.window_dd(nav, CAL, *W1)
    m['w2_cagr'] = S.window_cagr(nav, CAL, *W2)
    m['w2_dd'] = S.window_dd(nav, CAL, *W2)
    m.update(book)
    t = pd.DataFrame(tr)
    tf = t[t['kind'] == 'FULL']
    m['mult_all'] = round(float((1 + tf.ret_pct / 100).prod()), 3)
    m['mult_drop10'] = round(
        float((1 + tf.drop(tf.nlargest(10, 'ret_pct').index).ret_pct / 100).prod()), 3)
    return seed, nav.astype(np.float32), m


NUM = ['cagr', 'maxdd', 'calmar', 'sharpe', 'trades', 'trades_per_yr', 'win_rate', 'avg_win',
       'avg_loss', 'expectancy', 'max_loss_streak', 'invested_pct', 'w1_cagr', 'w1_dd',
       'w2_cagr', 'w2_dd', 'days_signal', 'days_bind', 'days_full', 'turned_away',
       'cash_refused', 'entries_taken', 'swaps', 'swap_attempts', 'swaps_per_yr', 'trims',
       'trims_per_yr', 'trim_notional', 'partial_fills', 'tax_paid', 'tax_st', 'tax_lt',
       'turnover_x', 'pos_rs_med', 'cap_pct_med', 'cap_pct_p95', 'cap_over1pct',
       'profit_total', 'top10_share', 'top10_share_all', 'mult_all', 'mult_drop10']
CFG = ['label', 'axis', 'slots', 'slot_pct', 'select', 'idle', 'cost_bps', 'rot_score',
       'rot_margin', 'rot_max_per_day', 'rot_entrant', 'trim_mult', 'trim_when',
       'min_fill_frac', 'rot_sell_only', 'hard_stop', 'hard_stop_pct']
SEED_FIELDS = CFG + ['seed'] + NUM
CELL_FIELDS = (CFG + ['seeds', 'seedbase', 'cagr_med', 'cagr_min', 'cagr_max', 'maxdd_med',
                      'maxdd_worst', 'calmar_med', 'calmar_min']
               + ['%s_med' % k for k in NUM] + ['med_seed', 'secs'])


def main():
    global PANEL, EVENTS, CAL, AUX
    stage, seeds_n, workers, bps, seedbase, plateau = 'fresh', 30, 2, None, 1000, None
    for a in sys.argv[1:]:
        if a.startswith('--stage='):
            stage = a.split('=', 1)[1]
        elif a.startswith('--seeds='):
            seeds_n = int(a.split('=', 1)[1])
        elif a.startswith('--workers='):
            workers = int(a.split('=', 1)[1])
        elif a.startswith('--bps='):
            bps = float(a.split('=', 1)[1])
        elif a.startswith('--seedbase='):
            seedbase = int(a.split('=', 1)[1])
        elif a.startswith('--plateau='):
            plateau = a.split('=', 1)[1]
    t0 = time.time()
    seeds = [seedbase + i for i in range(1, seeds_n + 1)]

    PANEL = pickle.load(open(PANEL_PKL, 'rb'))
    CAL = PANEL.cal
    st = pickle.load(open(R166 / 'st166.pkl', 'rb'))
    AUX = S.build_aux(PANEL, st)
    ev = pd.read_csv(R166 / 'events166.csv')
    EVENTS = ev.to_dict('records')
    for e in EVENTS:
        e['entry_i'] = int(e['entry_i'])
    print('panel %d syms / %d days (%s .. %s);  %d events;  ST lines for %d syms'
          % (len(PANEL.close), PANEL.n, CAL[0], CAL[-1], len(EVENTS), len(st)), flush=True)

    cells = grid(stage, plateau)
    if bps is not None:
        for c in cells:
            c['cost_bps'] = bps
            c['label'] = '%s_b%d' % (c['label'], int(bps))
    print('stage=%s cells=%d seeds=%s..%s workers=%d'
          % (stage, len(cells), seeds[0], seeds[-1], workers), flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    scsv, ccsv = RES / ('seedstatsB_%s.csv' % stage), RES / ('cellsB_%s.csv' % stage)
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
    navdir = RES / ('navsB_%s' % stage)
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
                row = {k: cell.get(k) for k in CFG}
                row['seed'] = sd
                row.update(m)
                w.writerow(row)
        cg = np.array([m['cagr'] for m in ms], float)
        med_i = int(np.argsort(cg)[len(cg) // 2])
        np.savez_compressed(navdir / ('%s.npz' % cell['label']), navs=navs,
                            seeds=np.array(seeds), med_seed=seeds[med_i],
                            dates=np.array(CAL))
        row = {k: cell.get(k) for k in CFG}
        row.update(seeds=len(seeds), seedbase=seedbase,
                   cagr_med=round(float(np.median(cg)), 2),
                   cagr_min=round(float(cg.min()), 2), cagr_max=round(float(cg.max()), 2),
                   maxdd_med=round(float(np.median([m['maxdd'] for m in ms])), 2),
                   maxdd_worst=round(float(min(m['maxdd'] for m in ms)), 2),
                   calmar_med=round(float(np.median([m['calmar'] for m in ms])), 3),
                   calmar_min=round(float(min(m['calmar'] for m in ms)), 3),
                   med_seed=seeds[med_i], secs=round(time.time() - ct, 1))
        for k in NUM:
            v = [m.get(k, np.nan) for m in ms]
            try:
                row['%s_med' % k] = round(float(np.nanmedian(np.array(v, dtype=float))), 3)
            except Exception:
                row['%s_med' % k] = np.nan
        with open(ccsv, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=CELL_FIELDS, extrasaction='ignore').writerow(row)
        print('[%2d/%2d] %-22s | CAGR %6.2f%% [%5.2f..%5.2f] DD %7.2f%% Calmar %5.3f | '
              'swaps/yr %5.1f inv %5.1f%% tax %9.0f turn %4.2fx W1 %5.2f W2 %5.2f | %.0fs'
              % (ci, len(cells), cell['label'], row['cagr_med'], row['cagr_min'],
                 row['cagr_max'], row['maxdd_med'], row['calmar_med'],
                 row['swaps_per_yr_med'], row['invested_pct_med'], row['tax_paid_med'],
                 row['turnover_x_med'], row['w1_cagr_med'], row['w2_cagr_med'],
                 row['secs']), flush=True)
    if pool:
        pool.close()
        pool.join()
    print('\nstage %s done in %.0fs' % (stage, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
