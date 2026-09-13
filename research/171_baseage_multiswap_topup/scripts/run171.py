# -*- coding: utf-8 -*-
"""research/171 -- OA Base Age: swap MORE than one holding a night, and/or redeploy the
proceeds into the EXISTING holdings that are running hardest, instead of into a new entrant.

The harness (`sim171.py`, generated from research/170's `sim170.py` by `patch171.py`) and the
inputs (research/164's panel, research/166's frozen event list and SuperTrend line cache) are
reused unchanged.  Only the cell list and the seed set are new.

    python3 patch171.py --verify                                  # build + prove the no-op
    python3 run171.py --stage=proof  --seeds=30 --seedbase=1000    # harness proof
    python3 run171.py --stage=main   --seeds=30 --seedbase=7000    # the pre-registered grid
    python3 run171.py --stage=follow --seeds=30 --seedbase=7000    # pre-registered follow-ups
    python3 run171.py --stage=cost40 --seeds=30 --seedbase=7000 --bps=40

Writes incrementally: one row per (cell, seed) to results/seedstats_<stage>.csv and one
summary row per cell to results/cells_<stage>.csv.  Finished cells are skipped on restart.
"""
import csv
import json
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
import sim171 as S                                                    # noqa: E402

W1 = ('2005-01-03', '2015-12-31')
W2 = ('2016-01-01', '2026-12-31')
W3 = ('2025-01-01', '2026-12-31')          # REPORTING window only -- never a selection window
BASE_IDLE = 0.052

PANEL = EVENTS = CAL = AUX = None


def _c(label, axis, **kw):
    row = dict(label=label, axis=axis, slots=16, slot_pct=0.0625, select='random',
               idle=BASE_IDLE, cost_bps=25.0, rot_score=None, rot_margin=0.0,
               rot_max_per_day=1, rot_entrant='tv', trim_mult=0.0, trim_when='month',
               min_fill_frac=0.0, rot_sell_only=0, hard_stop=0, hard_stop_pct=0.92,
               rot_dest='entrant', rot_trigger='signal', rot_spill='none',
               topup_rank='rs', topup_split=1, topup_cap=0.0)
    row.update(kw)
    return row


def _rot(label, axis, **kw):
    """A cell built on OA-ROT-1: sell the holding more than 10% under water, entrant = rs252."""
    d = dict(rot_score='unreal', rot_margin=10.0, rot_entrant='rs')
    d.update(kw)
    return _c(label, axis, **d)


CAPS = [('c20', 2.0), ('c30', 3.0), ('c00', 0.0)]
XS = [('rs', 'rs'), ('unre', 'unreal'), ('cush', 'cushion')]


def grid(stage):
    """THE PRE-REGISTERED CELL LIST.  Written into the STATUS doc before the first cell ran."""
    if stage == 'proof':
        # research/170 Part B's own two cells, on its own fresh seed base (1000).
        return [_c('BASE_rand', 'PROOF'),
                _rot('X_entrs_unre_m010', 'PROOF')]

    if stage == 'measure':
        # The eligibility histogram on the INCUMBENT's own path.  `rot_max_per_day = 0` makes
        # the rotation block break before it can fire, so the NAV must equal REF_base exactly
        # while the 10%-under-water counter still records.  A control, not a selection cell.
        return [_rot('CTRL_measure_k0', 'MEASURE', rot_max_per_day=0)]

    if stage == 'main':
        out = [
            # ---- references (not selection cells) --------------------------------------
            _c('REF_base', 'REF'),                       # the incumbent: never swap
            _rot('REF_rot1', 'REF'),                     # OA-ROT-1 exactly as staged live
        ]
        # ---- AXIS A: how many holdings leave per evening -------------------------------
        for k in (2, 3, 4, 6, 99):
            out.append(_rot('A_k%s' % ('all' if k == 99 else k), 'A_K', rot_max_per_day=k))
        for m, tag in ((7.5, 'm075'), (12.5, 'm125')):
            out.append(_rot('A_k2_%s' % tag, 'A_MARGIN', rot_max_per_day=2, rot_margin=m))
        out.append(_rot('A_k2_spillcash', 'A_SPILL', rot_max_per_day=2, rot_spill='cash'))
        out.append(_rot('A_k2_spilltopup', 'A_SPILL', rot_max_per_day=2, rot_spill='topup',
                        topup_rank='rs', topup_split=1, topup_cap=3.0))
        # ---- AXIS B: destination = top up an EXISTING winner ---------------------------
        for k in (1, 2):
            for (xt, xv) in XS:
                for (ct, cv) in CAPS:
                    out.append(_rot('B_S_k%d_%s_%s' % (k, xt, ct), 'B_TOPUP',
                                    rot_max_per_day=k, rot_dest='topup',
                                    topup_rank=xv, topup_split=1, topup_cap=cv))
        # ---- AXIS C: hybrid -------------------------------------------------------------
        for k in (1, 2):
            out.append(_rot('C_hyb_k%d' % k, 'C_HYBRID', rot_max_per_day=k,
                            rot_dest='hybrid', topup_rank='rs', topup_split=1,
                            topup_cap=3.0))
            out.append(_rot('C_els_k%d' % k, 'C_HYBRID', rot_max_per_day=k,
                            rot_trigger='any', rot_spill='topup', topup_rank='rs',
                            topup_split=1, topup_cap=3.0))
        # ---- controls and nulls (NOT selection cells) -----------------------------------
        for k in (1, 2, 3):
            out.append(_rot('CTRL_sellonly_k%d' % k, 'CTRL', rot_max_per_day=k,
                            rot_sell_only=1))
        for k, p in ((2, 0.03), (3, 0.03)):
            out.append(_c('CTRL_null_k%d' % k, 'NULL', rot_score='rand', rot_margin=p,
                          rot_max_per_day=k, rot_entrant='rs'))
        return out

    if stage == 'follow':
        # Pre-registered follow-ups.  `--best=<label>` names the winning Axis-B cell; its
        # rank / cap are read back from results/cells_main.csv so nothing is retyped.
        best = _read_best()
        out = []
        for k in (1, 2):
            out.append(_rot('F_split2_k%d' % k, 'B_SPLIT', rot_max_per_day=k,
                            rot_dest='topup', topup_rank=best['topup_rank'], topup_split=2,
                            topup_cap=float(best['topup_cap'])))
            out.append(_rot('F_anyeve_k%d' % k, 'B_TRIGGER', rot_max_per_day=k,
                            rot_dest='topup', rot_trigger='any',
                            topup_rank=best['topup_rank'], topup_split=1,
                            topup_cap=float(best['topup_cap'])))
        return out

    if stage == 'follow2':
        # Executing the pre-registered plateau and control clauses on the cell that leads the
        # main grid: `C_els_k1` -- entrant when a signal is refused, otherwise top up the
        # strongest holding, one sale an evening.
        base = dict(rot_max_per_day=1, rot_trigger='any', rot_spill='topup',
                    topup_rank='rs', topup_split=1, topup_cap=3.0)
        out = [
            # plateau on the margin -- the pre-registered clause
            _rot('P_els_m075', 'C_PLATEAU', rot_margin=7.5, **base),
            _rot('P_els_m125', 'C_PLATEAU', rot_margin=12.5, **base),
            # which holding gets the money
            _rot('C_els_k1_unre', 'C_RANK', **dict(base, topup_rank='unreal')),
            _rot('C_els_k1_cush', 'C_RANK', **dict(base, topup_rank='cushion')),
            # how concentrated it is allowed to get
            _rot('C_els_k1_c20', 'C_CAP', **dict(base, topup_cap=2.0)),
            _rot('C_els_k1_c00', 'C_CAP', **dict(base, topup_cap=0.0)),
            _rot('C_els_k1_c15', 'C_CAP', **dict(base, topup_cap=1.5)),
            # spread it over the best two rather than all into the best one
            _rot('C_els_k1_split2', 'C_SPLIT', **dict(base, topup_split=2)),
            # ---- THE CONTROLS THAT SEPARATE THE STOP FROM THE TOP-UP (not selection) ----
            # 1. same sale, same trigger, money stays in CASH.  If this reaches the same
            #    Calmar, the top-up is decoration and the edge is an unconditional stop.
            _rot('CTRL_anycash_k1', 'CTRL', rot_max_per_day=1, rot_trigger='any',
                 rot_spill='cash'),
            _rot('CTRL_anycash_k2', 'CTRL', rot_max_per_day=2, rot_trigger='any',
                 rot_spill='cash'),
            # 2. research/166's plain unconditional hard stop, no rotation machinery at all
            _c('CTRL_hardstop10', 'CTRL', hard_stop=1, hard_stop_pct=0.90),
            # 3. rate-matched random swap at the elevated rate this family fires at
            _c('CTRL_null_p007', 'NULL', rot_score='rand', rot_margin=0.07,
               rot_max_per_day=1, rot_entrant='rs'),
        ]
        return out

    if stage == 'follow3':
        # The pre-registered plateau clause, applied to the cushion-ranked variant that leads
        # follow2 -- and one control that separates the ANY-EVENING trigger from the ranking.
        base = dict(rot_max_per_day=1, rot_trigger='any', rot_spill='topup',
                    topup_rank='cushion', topup_split=1, topup_cap=3.0)
        return [
            _rot('P_cush_m075', 'C_PLATEAU', rot_margin=7.5, **base),
            _rot('P_cush_m125', 'C_PLATEAU', rot_margin=12.5, **base),
            _rot('C_cush_c15', 'C_CAP', **dict(base, topup_cap=1.5)),
            _rot('C_cush_c20', 'C_CAP', **dict(base, topup_cap=2.0)),
            _rot('C_cush_c00', 'C_CAP', **dict(base, topup_cap=0.0)),
            _rot('C_cush_k2', 'C_K', **dict(base, rot_max_per_day=2)),
            _rot('C_cush_split2', 'C_SPLIT', **dict(base, topup_split=2)),
            # CONTROL: same ranking, but only on evenings when a signal was actually refused
            _rot('C_cush_sigonly', 'C_TRIGGER', **dict(base, rot_trigger='signal')),
        ]

    if stage == 'plat':
        # margin neighbours of whichever k wins axis A
        best_k = int(_arg('--bestk', '2'))
        return [_rot('P_k%d_m075' % best_k, 'A_PLATEAU', rot_max_per_day=best_k,
                     rot_margin=7.5),
                _rot('P_k%d_m125' % best_k, 'A_PLATEAU', rot_max_per_day=best_k,
                     rot_margin=12.5)]

    if stage.startswith('cost'):
        # re-scoring of the shortlist, not selection
        labels = _arg('--labels', 'REF_base,REF_rot1,A_k2').split(',')
        rows = {}
        for p in sorted(RES.glob('cells_*.csv')):
            if 'cost' in p.name:
                continue
            for r in csv.DictReader(open(p)):
                rows[r['label']] = r
        out = []
        for lb in labels:
            r = rows[lb]
            out.append(_c(lb, 'COST', rot_score=(r['rot_score'] or None),
                          rot_margin=float(r['rot_margin']),
                          rot_max_per_day=int(r['rot_max_per_day']),
                          rot_entrant=r['rot_entrant'], rot_dest=r['rot_dest'],
                          rot_trigger=r['rot_trigger'], rot_spill=r['rot_spill'],
                          topup_rank=r['topup_rank'], topup_split=int(r['topup_split']),
                          topup_cap=float(r['topup_cap']),
                          rot_sell_only=int(r['rot_sell_only']),
                          hard_stop=int(r['hard_stop']),
                          hard_stop_pct=float(r['hard_stop_pct'])))
        return out

    raise SystemExit('unknown stage %s' % stage)


def _arg(flag, default):
    for a in sys.argv[1:]:
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return default


def _read_best():
    lb = _arg('--best', '')
    rows = {r['label']: r for r in csv.DictReader(open(RES / 'cells_main.csv'))}
    if lb not in rows:
        raise SystemExit('--best=<label from cells_main.csv> required; have %d labels'
                         % len(rows))
    return rows[lb]


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
               min_fill_frac=cell['min_fill_frac'],
               rot_dest=cell['rot_dest'], rot_trigger=cell['rot_trigger'],
               rot_spill=cell['rot_spill'], topup_rank=cell['topup_rank'],
               topup_split=cell['topup_split'], topup_cap=cell['topup_cap'])
    nav, tr, inv, book = S.simulate(EVENTS, PANEL, AUX, cfg, seed)
    m = S.metrics(nav, CAL, tr)
    m['invested_pct'] = round(100.0 * float(np.nanmean(inv)), 2)
    for tag, w in (('w1', W1), ('w2', W2), ('w3', W3)):
        m['%s_cagr' % tag] = S.window_cagr(nav, CAL, *w)
        m['%s_dd' % tag] = S.window_dd(nav, CAL, *w)
    m.update(book)
    t = pd.DataFrame(tr)
    tf = t[t['kind'] == 'FULL']
    m['mult_all'] = round(float((1 + tf.ret_pct / 100).prod()), 3)
    m['mult_drop10'] = round(
        float((1 + tf.drop(tf.nlargest(10, 'ret_pct').index).ret_pct / 100).prod()), 3)
    return seed, nav.astype(np.float32), m


NUM = ['cagr', 'maxdd', 'calmar', 'sharpe', 'trades', 'trades_per_yr', 'win_rate', 'avg_win',
       'avg_loss', 'expectancy', 'max_loss_streak', 'invested_pct',
       'w1_cagr', 'w1_dd', 'w2_cagr', 'w2_dd', 'w3_cagr', 'w3_dd',
       'days_signal', 'days_bind', 'days_full', 'turned_away',
       'cash_refused', 'entries_taken', 'swaps', 'swap_attempts', 'swaps_per_yr', 'trims',
       'trims_per_yr', 'trim_notional', 'partial_fills', 'tax_paid', 'tax_st', 'tax_lt',
       'turnover_x', 'pos_rs_med', 'cap_pct_med', 'cap_pct_p95', 'cap_over1pct',
       'profit_total', 'top10_share', 'top10_share_all', 'mult_all', 'mult_drop10',
       'topups', 'topups_per_yr', 'topup_notional', 'spills', 'topup_refused',
       'elig_days', 'elig_total', 'elig_ge1', 'elig_ge2', 'elig_ge3', 'elig_max',
       'elig_ge2_pct', 'elig_mean', 'max_pos_w']
CFG = ['label', 'axis', 'slots', 'slot_pct', 'select', 'idle', 'cost_bps', 'rot_score',
       'rot_margin', 'rot_max_per_day', 'rot_entrant', 'trim_mult', 'trim_when',
       'min_fill_frac', 'rot_sell_only', 'hard_stop', 'hard_stop_pct',
       'rot_dest', 'rot_trigger', 'rot_spill', 'topup_rank', 'topup_split', 'topup_cap']
SEED_FIELDS = CFG + ['seed'] + NUM
CELL_FIELDS = (CFG + ['seeds', 'seedbase', 'cagr_med', 'cagr_min', 'cagr_max', 'maxdd_med',
                      'maxdd_worst', 'calmar_med', 'calmar_min']
               + ['%s_med' % k for k in NUM] + ['med_seed', 'secs'])


def main():
    global PANEL, EVENTS, CAL, AUX
    stage = _arg('--stage', 'main')
    seeds_n = int(_arg('--seeds', '30'))
    workers = int(_arg('--workers', '2'))
    seedbase = int(_arg('--seedbase', '7000'))
    bps = _arg('--bps', '')
    bps = float(bps) if bps else None
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

    cells = grid(stage)
    if bps is not None:
        for c in cells:
            c['cost_bps'] = bps
            c['label'] = '%s_b%d' % (c['label'], int(bps))
    print('stage=%s cells=%d seeds=%s..%s workers=%d'
          % (stage, len(cells), seeds[0], seeds[-1], workers), flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    scsv = RES / ('seedstats_%s.csv' % stage)
    ccsv = RES / ('cells_%s.csv' % stage)
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
        row.update(seeds=len(seeds), seedbase=seedbase,
                   cagr_med=round(float(np.median(cg)), 3),
                   cagr_min=round(float(cg.min()), 2), cagr_max=round(float(cg.max()), 2),
                   maxdd_med=round(float(np.median([m['maxdd'] for m in ms])), 3),
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
        print('[%2d/%2d] %-22s | CAGR %6.2f%% [%5.2f..%5.2f] DD %7.2f%% Cal %5.3f | '
              'sw/yr %5.1f tu/yr %5.1f inv %5.1f%% maxw %4.1f%% tax %8.0f turn %4.2fx '
              'W1 %5.2f W2 %5.2f W3 %6.2f | %.0fs'
              % (ci, len(cells), cell['label'], row['cagr_med'], row['cagr_min'],
                 row['cagr_max'], row['maxdd_med'], row['calmar_med'],
                 row['swaps_per_yr_med'], row['topups_per_yr_med'], row['invested_pct_med'],
                 row['max_pos_w_med'], row['tax_paid_med'], row['turnover_x_med'],
                 row['w1_cagr_med'], row['w2_cagr_med'], row['w3_cagr_med'],
                 row['secs']), flush=True)
    if pool:
        pool.close()
        pool.join()
    print('\nstage %s done in %.0fs' % (stage, time.time() - t0), flush=True)
    if stage == 'proof':
        _check_proof()


PROOF_REQUIRED = {'BASE_rand': (20.945, -34.045, 0.611),
                  'X_entrs_unre_m010': (22.58, -31.78, 0.710)}


def _check_proof():
    rows = {r['label']: r for r in csv.DictReader(open(RES / 'cells_proof.csv'))}
    out = {}
    ok = True
    for lb, (cg, dd, cal) in PROOF_REQUIRED.items():
        r = rows.get(lb)
        if r is None:
            print('PROOF MISSING: %s' % lb)
            ok = False
            continue
        got = (float(r['cagr_med']), float(r['maxdd_med']), float(r['calmar_med']))
        good = (abs(got[0] - cg) < 0.006 and abs(got[1] - dd) < 0.006
                and abs(got[2] - cal) < 0.0006)
        ok = ok and good
        out[lb] = dict(required=[cg, dd, cal], got=list(got), match=good)
        print('PROOF %-20s required %8.3f / %8.3f / %6.3f   got %8.3f / %8.3f / %6.3f   %s'
              % (lb, cg, dd, cal, got[0], got[1], got[2], 'MATCH' if good else 'MISMATCH'))
    json.dump(out, open(RES / 'proof171.json', 'w'), indent=1)
    if not ok:
        raise SystemExit('HARNESS PROOF FAILED - do not run selection cells')
    print('HARNESS PROOF OK - sim171 reproduces research/170 Part B on its own seed base')


if __name__ == '__main__':
    main()
