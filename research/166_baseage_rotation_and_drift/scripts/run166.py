# -*- coding: utf-8 -*-
"""research/166 -- rotation + drift sweep on the adopted Open Alpha - Base Age spec.

Usage:
    python3 run166.py --stage=proof --seeds=30 --workers=1
    python3 run166.py --stage=scan  --seeds=10 --workers=2
    python3 run166.py --stage=full  --seeds=30 --workers=2 --only=BASE,A_cush
    python3 run166.py --stage=cost40 --seeds=30 --workers=2 --bps=40 --only=<shortlist>

Writes INCREMENTALLY: one row per (cell, seed) to results/seedstats_<stage>.csv and one
summary row per cell to results/cells_<stage>.csv, both appended the moment a cell finishes.
Already-finished cells are skipped on restart.
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
sys.path.insert(0, str(HERE))
import sim166 as S                                                    # noqa: E402

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


def grid(stage):
    out = []
    if stage == 'proof':
        out.append(_c('PROOF_050', 'P', idle=0.050))
        out.append(_c('PROOF_052', 'P', idle=0.052))
        out.append(_c('PROOF_055', 'P', idle=0.055))
        return out

    # ---------------- baselines (not selection cells) ------------------------------
    out.append(_c('BASE_rand', 'BASE'))                      # the incumbent, 5.2% idle cash
    out.append(_c('BASE_tv', 'BASE', select='tv'))           # r/164's live tie-break

    # ---------------- axis A: rotation ---------------------------------------------
    MARG = {'cushion': (0, 5, 10, 25), 'rs': (0, 10, 25, 50),
            'athdist': (0, 10, 20, 30), 'unreal': (0, 5, 10, 20),
            'held': (40, 80, 160, 320)}
    for sc, margins in MARG.items():
        for m in margins:
            out.append(_c('A_%s_m%03d' % (sc[:4], m), 'A', rot_score=sc, rot_margin=float(m)))
    for p in (0.05, 0.15, 0.35):
        out.append(_c('A_null_p%03d' % round(p * 100), 'A_null',
                      rot_score='rand', rot_margin=p))

    # ---------------- axis B: drift / trimming --------------------------------------
    for k in (1.5, 2.0, 3.0):
        out.append(_c('B1_trimM_k%03d' % round(k * 100), 'B',
                      trim_mult=k, trim_when='month'))
    for k in (1.5, 2.0):
        out.append(_c('B2_trimD_k%03d' % round(k * 100), 'B',
                      trim_mult=k, trim_when='demand'))
    for f in (0.25, 0.50, 0.75):
        out.append(_c('B3_fill_f%03d' % round(f * 100), 'B', min_fill_frac=f))
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
CELL_FIELDS = (CFG + ['seeds', 'cagr_med', 'cagr_min', 'cagr_max', 'maxdd_med',
                      'maxdd_worst', 'calmar_med', 'calmar_min']
               + ['%s_med' % k for k in NUM] + ['med_seed', 'secs'])


def main():
    global PANEL, EVENTS, CAL, AUX
    stage, seeds_n, workers, only, bps = 'scan', 10, 1, None, None
    extra = None
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
        elif a.startswith('--extra='):
            extra = a.split('=', 1)[1]
        elif a.startswith('--idle='):
            idle_override = float(a.split('=', 1)[1])
    t0 = time.time()
    seeds = list(range(1, seeds_n + 1))

    PANEL = pickle.load(open(PANEL_PKL, 'rb'))
    CAL = PANEL.cal
    st = pickle.load(open(RES / 'st166.pkl', 'rb'))
    AUX = S.build_aux(PANEL, st)
    ev = pd.read_csv(RES / 'events166.csv')
    EVENTS = ev.to_dict('records')
    for e in EVENTS:
        e['entry_i'] = int(e['entry_i'])
    print('panel %d syms / %d days (%s .. %s);  %d events;  ST lines for %d syms'
          % (len(PANEL.close), PANEL.n, CAL[0], CAL[-1], len(EVENTS), len(st)), flush=True)

    cells = grid(stage)
    if extra:
        cells = extra_grid(extra)
    if bps is not None:
        for c in cells:
            c['cost_bps'] = bps
            c['label'] = '%s_b%d' % (c['label'], int(bps))
    if idle_override is not None:
        for c in cells:
            c['idle'] = idle_override
            c['label'] = '%s_y%03d' % (c['label'], round(idle_override * 1000))
    if only:
        cells = [c for c in cells if any(c['label'].startswith(p) for p in only)]
    print('stage=%s cells=%d seeds=%d workers=%d' % (stage, len(cells), len(seeds), workers),
          flush=True)

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
        row.update(seeds=len(seeds), cagr_med=round(float(np.median(cg)), 2),
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
        print('[%2d/%2d] %-18s | CAGR %6.2f%% [%5.2f..%5.2f] DD %7.2f%% Calmar %5.3f | '
              'swaps/yr %5.1f trims/yr %5.1f cashref %5.0f slotref %5.0f inv %5.1f%% '
              'tax %9.0f turn %4.2fx | %.0fs'
              % (ci, len(cells), cell['label'], row['cagr_med'], row['cagr_min'],
                 row['cagr_max'], row['maxdd_med'], row['calmar_med'],
                 row['swaps_per_yr_med'], row['trims_per_yr_med'], row['cash_refused_med'],
                 row['turned_away_med'], row['invested_pct_med'], row['tax_paid_med'],
                 row['turnover_x_med'], row['secs']), flush=True)
    if pool:
        pool.close(); pool.join()
    print('\nstage %s done in %.0fs' % (stage, time.time() - t0), flush=True)


def extra_grid(spec):
    """Cells added AFTER the pre-registered scan, to execute the pre-registered plateau /
    interaction tests.  Every one of them is disclosed separately in RESULTS.md.
    spec is a comma list of shorthand definitions, e.g.
      rotmax:cushion:10:3   swap up to 3 times a day
      ent:cushion:10:rs     entrant priority by rs252 instead of tv20
      ix:cushion:10:2.0:month:0.5   rotation x month-end trim x partial fill
      plat:cushion:2,15,20  extra margins on one score
    """
    out = []
    for item in spec.split(','):
        parts = item.split(':')
        kind = parts[0]
        if kind == 'rotmax':
            sc, m, k = parts[1], float(parts[2]), int(parts[3])
            out.append(_c('X_max%d_%s_m%03d' % (k, sc[:4], m), 'X', rot_score=sc,
                          rot_margin=m, rot_max_per_day=k))
        elif kind == 'ent':
            sc, m, e = parts[1], float(parts[2]), parts[3]
            out.append(_c('X_ent%s_%s_m%03d' % (e, sc[:4], m), 'X', rot_score=sc,
                          rot_margin=m, rot_entrant=e))
        elif kind == 'plat':
            sc = parts[1]
            for m in [float(x) for x in parts[2].split('|')]:
                out.append(_c('X_plat_%s_m%03d' % (sc[:4], m), 'X', rot_score=sc,
                              rot_margin=m))
        elif kind == 'trimplat':
            for k in [float(x) for x in parts[1].split('|')]:
                out.append(_c('X_trimM_k%03d' % round(k * 100), 'X', trim_mult=k,
                              trim_when='month'))
        elif kind == 'ix':
            sc, m, k, when, f = parts[1], float(parts[2]), float(parts[3]), parts[4], float(parts[5])
            lab = 'C_%s%03d_k%03d%s_f%03d' % (sc[:4], m, round(k * 100), when[0],
                                              round(f * 100))
            out.append(_c(lab, 'C', rot_score=(sc if sc != 'none' else None), rot_margin=m,
                          trim_mult=k, trim_when=when, min_fill_frac=f))
        elif kind == 'sel':
            # CONTROL: change only the contested-slot tie-break, no rotation at all
            out.append(_c('CTRL_sel_%s' % parts[1], 'CTRL', select=parts[1]))
        elif kind == 'null':
            # the rate-matched null: a random holding swapped out with probability p
            for p in [float(x) for x in parts[1].split('|')]:
                out.append(_c('A_null_p%03d' % round(p * 100), 'A_null',
                              rot_score='rand', rot_margin=p))
        elif kind == 'sellonly':
            # CONTROL: sell the weakest on the same trigger but DO NOT buy the entrant --
            # this is a conditional stop-loss, not a rotation
            sc, m = parts[1], float(parts[2])
            out.append(_c('CTRL_sellonly_%s_m%03d' % (sc[:4], m), 'CTRL', rot_score=sc,
                          rot_margin=m, rot_sell_only=1))
        elif kind == 'hs':
            # CONTROL: an UNCONDITIONAL hard stop at the same depth, no rotation
            for pc in [float(x) for x in parts[1].split('|')]:
                out.append(_c('CTRL_hardstop%02d' % round(100 * (1 - pc)), 'CTRL',
                              hard_stop=1, hard_stop_pct=pc))
        elif kind == 'base':
            out.append(_c('BASE_rand', 'BASE'))
            out.append(_c('BASE_tv', 'BASE', select='tv'))
    return out


if __name__ == '__main__':
    main()
