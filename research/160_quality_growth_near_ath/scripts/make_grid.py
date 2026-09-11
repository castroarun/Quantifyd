# -*- coding: utf-8 -*-
"""research/160 STUDY leg - emit the cell JSON for each gate.

    make_grid.py g1   -> results/grid_g1.json
    make_grid.py g2   -> results/grid_g2.json   (reads the G1 survivors from cells_g1.csv)
    make_grid.py g3   -> results/grid_g3.json   (reads the G2 finalists)

Every cell is a dict of qg_engine.Cell field names. Anything omitted takes the engine
default. The engine skips a cell whose label already sits in the --out CSV, so re-emitting
a superset of a grid is always safe.

WINDOW. 2018-08-01 .. 2026-09-10 everywhere, because that is the first month at which >=80%
of active liquid names have four filed fiscal years (DATA leg 8.4). Earlier dates measure
Screener's page depth, not the screen.
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / 'results'
M = str(RES / 'masks')            # DATA leg masks
MS = str(RES / 'masks_study')     # this leg's Family-B masks

START, END = '2018-08-01', '2026-09-10'
OFF, SEEDS = 12, 30


def base(**kw):
    d = dict(start=START, end=END, entry='rebalance', cadence='monthly', rank='rs',
             slots=15, buffer=1.5, state='near', k=0.90, tv_floor=2.0,
             exits='none', index_gate='none', fill='next_open', cost_bps=25.0,
             tax=True, cash_yield=0.05, offsets=OFF, arms='all')
    d.update(kw)
    return d


def mask(name, study=False):
    return '%s/%s.npz' % (MS if study else M, name)


# the DATA leg's masks worth a G1 cell (arun_strict == g20_mc1000 == de0p2 == q15, so the
# duplicates are not re-run under their alias names)
DATA_MASKS = ['arun_strict', 'no_growth', 'no_roe', 'no_roce', 'no_de', 'no_mcap',
              'growth_only', 'quality_only',
              'g15_mc500', 'g15_mc1000', 'g15_mc2500', 'g20_mc500', 'g20_mc2500',
              'g25_mc1000', 'g30_mc1000', 'de0p5', 'de1p0', 'q12', 'q20',
              'opm_slope_pos', 'opm_steady', 'opm_min', 'opm_rising_q']
STUDY_MASKS = ['b1_noneg', 'b2_noneg_mc1000', 'b3_qual_mc', 'b4_g15_mc',
               'b5_g15_qual_mc', 'b7_g10_qual_mc']


def g1():
    cells = []

    # ---- 1. price-only controls (no fundamental mask at all) ------------------------
    cells.append(base(label='P_base_k90_N15_rs'))
    for k in (0.85, 0.95):
        cells.append(base(label='P_k%d_N15_rs' % int(k * 100), k=k))
    cells.append(base(label='P_newath_N15_rs', state='new_ath'))
    for n in (10, 20, 30):
        cells.append(base(label='P_base_N%d_rs' % n, slots=n))
    for tv in (1.0, 5.0):
        cells.append(base(label='P_base_tv%d_rs' % int(tv), tv_floor=tv))
    cells.append(base(label='P_base_rank_distath', rank='dist_ath'))
    # THE NULL: same universe, same near-ATH state, selection at random
    cells.append(base(label='P_NULL_random_N15', rank='random', offsets=0, seeds=SEEDS))
    # the screenable sub-universe with NO screen applied - the only fair comparator
    for mm in ('fail', 'pass'):
        cells.append(base(label='P_hasdata_%s' % mm, mask=mask('has_data'), mask_missing=mm))

    # ---- 2. fundamentals ONLY (near-ATH removed: k=0 => close >= 0, always true) -----
    for nm, st in [('arun_strict', False), ('growth_only', False), ('quality_only', False),
                   ('b3_qual_mc', True), ('b5_g15_qual_mc', True)]:
        for mm in ('fail', 'pass'):
            cells.append(base(label='F_noATH_%s_%s' % (nm, mm), k=0.0,
                              mask=mask(nm, st), mask_missing=mm))

    # ---- 3. every mask at k=0.90, both missing policies ------------------------------
    for nm in DATA_MASKS:
        for mm in ('fail', 'pass'):
            cells.append(base(label='A_%s_%s' % (nm, mm), mask=mask(nm), mask_missing=mm))
    for nm in STUDY_MASKS:
        for mm in ('fail', 'pass'):
            cells.append(base(label='B_%s_%s' % (nm, mm), mask=mask(nm, True),
                              mask_missing=mm))

    # ---- 4. ranking control: the mask with RS removed --------------------------------
    for nm, st in [('arun_strict', False), ('b5_g15_qual_mc', True)]:
        cells.append(base(label='R_rand_%s_fail' % nm, rank='random', offsets=0,
                          seeds=SEEDS, mask=mask(nm, st), mask_missing='fail'))

    # ---- 5. ladders on the two family leaders ----------------------------------------
    for nm, st, tag in [('arun_strict', False, 'A'), ('b5_g15_qual_mc', True, 'B')]:
        for n in (8, 10, 20, 30):
            cells.append(base(label='%s_%s_N%d' % (tag, nm, n), slots=n,
                              mask=mask(nm, st), mask_missing='fail'))
        for k in (0.85, 0.95):
            cells.append(base(label='%s_%s_k%d' % (tag, nm, int(k * 100)), k=k,
                              mask=mask(nm, st), mask_missing='fail'))
        cells.append(base(label='%s_%s_newath' % (tag, nm), state='new_ath',
                          mask=mask(nm, st), mask_missing='fail'))
        # the only exit that exists in G1: the screen itself turning false
        cells.append(base(label='%s_%s_fundfail' % (tag, nm), exits='fund_fail',
                          mask=mask(nm, st), mask_missing='fail'))
    return cells


def g1b():
    """Supplementary G1 cells the first pass showed were needed: the two masks that actually
    won the Calmar ranking (b7_g10_qual_mc, b3_qual_mc) never got the N / k / rank ladders,
    the liquidity floor turned out to be a live axis, and arun_strict is so thin that its
    honest slot count is below 8."""
    cells = []
    for tv in (3.0, 10.0):
        cells.append(base(label='P_base_tv%d_rs' % int(tv), tv_floor=tv))
    cells.append(base(label='P_hasdata_N30_fail', slots=30, mask=mask('has_data'),
                      mask_missing='fail'))
    cells.append(base(label='P_hasdata_tv5_fail', tv_floor=5.0, mask=mask('has_data'),
                      mask_missing='fail'))
    for nm in ('b7_g10_qual_mc', 'b3_qual_mc'):
        for n in (8, 10, 20, 30):
            cells.append(base(label='B_%s_N%d' % (nm, n), slots=n,
                              mask=mask(nm, True), mask_missing='fail'))
        for k in (0.85, 0.95):
            cells.append(base(label='B_%s_k%d' % (nm, int(k * 100)), k=k,
                              mask=mask(nm, True), mask_missing='fail'))
        cells.append(base(label='B_%s_newath' % nm, state='new_ath',
                          mask=mask(nm, True), mask_missing='fail'))
        for tv in (5.0, 10.0):
            cells.append(base(label='B_%s_tv%d' % (nm, int(tv)), tv_floor=tv,
                              mask=mask(nm, True), mask_missing='fail'))
        cells.append(base(label='B_%s_fundfail' % nm, exits='fund_fail',
                          mask=mask(nm, True), mask_missing='fail'))
        cells.append(base(label='R_rand_%s_fail' % nm, rank='random', offsets=0,
                          seeds=SEEDS, mask=mask(nm, True), mask_missing='fail'))
    # arun_strict is 43% invested at N=15; its honest book is much smaller
    for n in (4, 5, 6):
        cells.append(base(label='A_arun_strict_N%d' % n, slots=n,
                          mask=mask('arun_strict'), mask_missing='fail'))
    for tv in (5.0,):
        cells.append(base(label='A_arun_strict_tv%d' % int(tv), tv_floor=tv,
                          mask=mask('arun_strict'), mask_missing='fail'))
    return cells


# --------------------------------------------------------------------------- G2 ------
# The five books carried forward from G1. The CONTROL (no screen, screenable sub-universe)
# is swept identically to every screened arm, because the question G2 answers is not "does
# an exit help" - it is "does the screen still contribute anything once the book has the
# exit and the gate Arun's process is missing".
G2_BOOKS = [
    ('ctrl', mask('has_data'), 'no screen, screenable sub-universe'),
    ('Astrict', mask('arun_strict'), 'FAMILY A: the screen as written'),
    ('Ag15mc500', mask('g15_mc500'), 'FAMILY A relaxed: growth>15, mcap>500, D/E kept'),
    ('Bb7', mask('b7_g10_qual_mc', True), 'FAMILY B leader: quality + growth>10, no D/E'),
    ('Bb3', mask('b3_qual_mc', True), 'FAMILY B: quality only, no growth, no D/E'),
]
G2_EXITS = ['none', 'fund_fail',
            'sma_trail:20', 'sma_trail:50', 'sma_trail:100', 'sma_trail:200',
            'peak_dd:15', 'peak_dd:20', 'peak_dd:25', 'peak_dd:30',
            'donchian_low:20', 'donchian_low:50',
            'time:12', 'time:24',
            'hard_stop:15', 'hard_stop:20',
            'fund_fail,sma_trail:50', 'sma_trail:50,peak_dd:25', 'sma_trail:100,peak_dd:30']
G2_GATES = [('none', 'block_new', 'g0'),
            ('nifty200sma', 'block_new', 'gN2b'),
            ('niftybees100sma_weekly', 'block_new', 'gB1b')]


def _ex_tag(e):
    return (e.replace('sma_trail:', 'sma').replace('peak_dd:', 'dd')
             .replace('donchian_low:', 'dl').replace('hard_stop:', 'hs')
             .replace('time:', 't').replace('fund_fail', 'ff').replace(',', '+'))


def g2a():
    """Exit bake-off x index gate x the five books. 19 x 3 x 5 = 285 cells, run with
    arms='tax' (one third of the work) because this is a WIDE SCAN - every cell reported
    in the study is re-run with arms='all' in G2c."""
    cells = []
    for bk, mk, _ in G2_BOOKS:
        for ex in G2_EXITS:
            for gate, act, gt in G2_GATES:
                cells.append(base(label='%s_%s_%s' % (bk, _ex_tag(ex), gt),
                                  mask=mk, mask_missing='fail', exits=ex,
                                  index_gate=gate, gate_action=act, arms='tax'))
    return cells


def g2b():
    """Book construction, now that G2a has ranked the exits. The exit shortlist is the union
    of each book's top families (none / sma200 / peak_dd:15 / hard_stop:15 / donchian_low:50)
    and it is re-crossed with the gate at every slot count, because an exit tuned under one
    gate can flip when the gate changes (playbook 5). Wide scan -> arms='tax'."""
    cells = []
    books = [b for b in G2_BOOKS if b[0] in ('ctrl', 'Astrict', 'Ag15mc500', 'Bb7')]
    short = ['none', 'sma_trail:200', 'peak_dd:15', 'hard_stop:15', 'donchian_low:50']
    for bk, mk, _ in books:
        for ex in short:
            for gate, act, gt in G2_GATES[:2]:
                for n in (15, 30):
                    cells.append(base(label='%s_%s_%s_N%d' % (bk, _ex_tag(ex), gt, n),
                                      mask=mk, mask_missing='fail', exits=ex, slots=n,
                                      index_gate=gate, gate_action=act, arms='tax'))
        # the axes G2a held fixed
        for n in (40, 50):
            cells.append(base(label='%s_sma200_g0_N%d' % (bk, n), mask=mk,
                              mask_missing='fail', exits='sma_trail:200', slots=n,
                              arms='tax'))
        for tv in (5.0, 10.0):
            cells.append(base(label='%s_sma200_g0_N30_tv%d' % (bk, int(tv)), mask=mk,
                              mask_missing='fail', exits='sma_trail:200', slots=30,
                              tv_floor=tv, arms='tax'))
        for cad in ('quarterly', 'semiannual'):
            cells.append(base(label='%s_sma200_g0_N30_%s' % (bk, cad), mask=mk,
                              mask_missing='fail', exits='sma_trail:200', slots=30,
                              cadence=cad, arms='tax'))
        cells.append(base(label='%s_sma200_g0_N30_buf1' % bk, mask=mk, mask_missing='fail',
                          exits='sma_trail:200', slots=30, buffer=1.0, arms='tax'))
        cells.append(base(label='%s_sma200_g0_N30_k85' % bk, mask=mk, mask_missing='fail',
                          exits='sma_trail:200', slots=30, k=0.85, arms='tax'))
        # the gate as a liquidator rather than a brake
        for gate, gt in (('nifty200sma', 'gN2L'), ('niftybees100sma_weekly', 'gB1L')):
            cells.append(base(label='%s_sma200_%s_N30' % (bk, gt), mask=mk,
                              mask_missing='fail', exits='sma_trail:200', slots=30,
                              index_gate=gate, gate_action='liquidate_all', arms='tax'))
        # daily entry mechanics - a different book entirely, 30 seeds
        for en in ('first_qualify', 'ath_breakout'):
            cells.append(base(label='%s_%s_N30' % (bk, en), mask=mk, mask_missing='fail',
                              entry=en, slots=30, exits='sma_trail:200',
                              offsets=0, seeds=SEEDS, arms='tax'))
    return cells


# --------------------------------------------------------------------------- G3 ------
# The six finalists G2 left standing. Each is the best cell of its own book, EXCEPT
# 'Astrict_asrun', which is not a best cell at all: it is literally Arun's process - the
# screen as written, near its high, 15 names, no exit, no gate - and it is carried through
# every robustness test precisely because it is the thing he asked about.
W1 = ('2018-08-01', '2022-06-30')
W2 = ('2022-07-01', '2026-09-10')

FINALISTS = [
    ('F_ctrl_best', dict(mask=mask('has_data'), exits='sma_trail:200', slots=30, tv_floor=5.0),
     'NO SCREEN - near-ATH + RS on the screenable universe, 200-SMA trail, 30 names, tv>=5cr'),
    ('F_ctrl_N50', dict(mask=mask('has_data'), exits='sma_trail:200', slots=50),
     'NO SCREEN - the same, 50 names, tv>=2cr (the risk-adjusted price-only book)'),
    ('F_Astrict_asrun', dict(mask=mask('arun_strict'), exits='none', slots=15),
     "FAMILY A - Arun's process exactly as he runs it: the screen as written, 15 names, no exit"),
    ('F_Astrict_best', dict(mask=mask('arun_strict'), exits='peak_dd:15', slots=15),
     'FAMILY A at its best risk-adjusted setting found anywhere in 400+ cells'),
    ('F_Ag15mc500', dict(mask=mask('g15_mc500'), exits='peak_dd:15', slots=15),
     'FAMILY A relaxed - growth>15, mcap>500cr, D/E kept; 15% peak-drawdown exit'),
    ('F_Bb7', dict(mask=mask('b7_g10_qual_mc', True), exits='none', slots=15),
     'FAMILY B - quality + growth>10, no debt test; 15 names, no exit (nothing beat holding)'),
]


def g3main():
    """The six finalists over the full window, arms='all' (gross / net / after-tax) with the
    equity curves and the trade lists dumped for the YoY table, the tearsheet, the blend and
    the outlier test."""
    return [base(label=lb, mask_missing='fail', **kw) for lb, kw, _ in FINALISTS]


def g3rob():
    """Everything the adoption bar demands of a finalist: both sub-windows, the cost ladder,
    the missing-data policy the other way, and the idle-cash sensitivity (a book that is only
    60% invested earns a lot of its CAGR from the 5% cash assumption - set it to zero and the
    strategy has to stand on its own)."""
    cells = []
    for lb, kw, _ in FINALISTS:
        cells.append(base(label=lb + '_W1', start=W1[0], end=W1[1], mask_missing='fail', **kw))
        cells.append(base(label=lb + '_W2', start=W2[0], end=W2[1], mask_missing='fail', **kw))
        for c in (40.0, 60.0):
            cells.append(base(label='%s_cost%d' % (lb, int(c)), cost_bps=c,
                              mask_missing='fail', **kw))
        cells.append(base(label=lb + '_cash0', cash_yield=0.0, mask_missing='fail', **kw))
        cells.append(base(label=lb + '_misspass', mask_missing='pass', **kw))
    return cells


def pairs1():
    """The pre-registered paired test: every mask arm against the IDENTICAL book with no
    screen, on the same 12 offsets. The control is has_data (the screenable sub-universe),
    not the raw universe, so 'could not be screened' is never scored as 'failed the screen'."""
    ctrl = base(label='ctrl', mask=mask('has_data'), mask_missing='fail')
    out = []
    for nm, st in [('arun_strict', False), ('g15_mc500', False), ('g15_mc1000', False),
                   ('growth_only', False), ('quality_only', False), ('no_growth', False),
                   ('b1_noneg', True), ('b2_noneg_mc1000', True), ('b3_qual_mc', True),
                   ('b4_g15_mc', True), ('b5_g15_qual_mc', True), ('b7_g10_qual_mc', True)]:
        out.append(dict(name=nm, control='has_data (no screen)',
                        a=base(label='a_' + nm, mask=mask(nm, st), mask_missing='fail'),
                        b=dict(ctrl)))
    return out


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else 'g1').lower()
    cells = {'g1': g1, 'g1b': g1b, 'g2a': g2a, 'g2b': g2b, 'g3main': g3main, 'g3rob': g3rob, 'pairs1': pairs1}[which]()
    if which.startswith('pairs'):
        out = RES / ('%s.json' % which)
        json.dump(cells, open(out, 'w'), indent=1)
        print('%s: %d pairs -> %s' % (which, len(cells), out))
        return
    labels = [c['label'] for c in cells]
    assert len(labels) == len(set(labels)), 'duplicate labels: %s' % (
        [x for x in labels if labels.count(x) > 1][:5])
    out = RES / ('grid_%s.json' % which)
    json.dump(cells, open(out, 'w'), indent=1)
    print('%s: %d cells -> %s' % (which, len(cells), out))


if __name__ == '__main__':
    sys.exit(main())
