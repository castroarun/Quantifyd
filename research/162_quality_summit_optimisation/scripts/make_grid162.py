# -*- coding: utf-8 -*-
"""research/162 Part A — write the grid JSONs the r/162 engine consumes.

Pre-registration (STATUS section 3): **every selection decision is made on W1**
(2018-08-01 -> 2022-06-30). W2 (2022-07-01 -> 2026-09-10) is computed once, at the end,
for the chosen cells only. So phase 1 and phase 2 run on W1; phase 3 is the G3 package on
the finalists and is the first place W2 appears.

    make_grid162.py <phase> [--trail st_14_4] [--rank rs] [--mask b7_g10_qual_mc]

Phases
  a1  independent axes, one change at a time against the Quality Summit baseline:
      the six ATR-scaled trails x {fund_fail off, on}, the six ranking axes x N in
      {8,10,15}, inverse-vol sizing, plus the baselines and the no-screen control.
  a2  interactions on the phase-1 winners: trail x N, trail x rank, trail x weights,
      the index gate, and the A4 screen dials (growth x k x tv).
  a3  G3 on the finalists: full window + W1 + W2, cost ladder, cash yield off, missing
      policy both ways, all three arms.
"""
import argparse
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
R160M = 'research/160_quality_growth_near_ath/results/masks_study'
R162M = 'research/162_quality_summit_optimisation/results/masks162'
AUX = 'research/162_quality_summit_optimisation/results/aux_162.npz'

FULL = ('2018-08-01', '2026-09-10')
W1 = ('2018-08-01', '2022-06-30')
W2 = ('2022-07-01', '2026-09-10')

B7 = '%s/b7_g10_qual_mc.npz' % R160M
TRAILS = [('st7_3', 'st_trail:7_3'), ('st10_3', 'st_trail:10_3'),
          ('st14_4', 'st_trail:14_4'), ('st20_3', 'st_trail:20_3'),
          ('ch22_2', 'chand:22_2'), ('ch22_3', 'chand:22_3')]
RANKS = ['rs', 'profit_g3', 'opm_slope3', 'z_rs_profit', 'z_rs_opm', 'mcap_desc']


def cell(label, start, end, **kw):
    c = dict(label=label, start=start, end=end, aux=AUX, offsets=12, arms='tax',
             mask=B7, mask_missing='fail')
    c.update(kw)
    return c


def phase_a1():
    g = []
    # ---- A0 baselines: the incumbent and the no-screen control, all three windows ----
    for tag, (s, e) in (('full', FULL), ('W1', W1), ('W2', W2)):
        g.append(cell('A0_QSbase_%s' % tag, s, e, exits='none',
                      arms='all' if tag == 'full' else 'tax'))
        g.append(cell('A0_ctrl_%s' % tag, s, e, exits='none', mask='',
                      arms='all' if tag == 'full' else 'tax'))
    # ---- A1 the ATR-scaled trails, with and without the fundamental-failure exit ----
    for nm, spec in TRAILS:
        g.append(cell('A1_%s' % nm, *W1, exits=spec))
        g.append(cell('A1_%s_ff' % nm, *W1, exits='%s,fund_fail' % spec))
    g.append(cell('A1_ff_only', *W1, exits='fund_fail'))
    # ---- A2 ranking axes inside the qualifying set ----------------------------------
    for r in RANKS:
        for n in (8, 10, 15):
            g.append(cell('A2_%s_N%d' % (r, n), *W1, exits='none', rank=r, slots=n))
    # ---- A3 sizing ------------------------------------------------------------------
    g.append(cell('A3_invvol_N15', *W1, exits='none', weights='invvol'))
    return g


def phase_a2(trails, best_n, mask):
    """Interactions. `trails` is a list of (name, spec); `best_n` the phase-1 slot count."""
    g = []
    # the slot-count axis on its own, and crossed with each surviving trail
    for n in (8, 10, 12, 20, 30):
        g.append(cell('A2b_noexit_N%d' % n, *W1, exits='none', slots=n, mask=mask))
        for nm, spec in trails:
            g.append(cell('A2b_%s_N%d' % (nm, n), *W1, exits=spec, slots=n, mask=mask))
    # the best slot count crossed with every ranking axis and with inverse-vol sizing
    for r in RANKS:
        if r != 'rs':
            g.append(cell('A2c_N%d_%s' % (best_n, r), *W1, exits='none', rank=r,
                          slots=best_n, mask=mask))
    g.append(cell('A2d_N%d_invvol' % best_n, *W1, exits='none', weights='invvol',
                  slots=best_n, mask=mask))
    for nm, spec in trails:
        g.append(cell('A2d_%s_N%d_invvol' % (nm, best_n), *W1, exits=spec,
                      weights='invvol', slots=best_n, mask=mask))
        # index gate, block_new only
        g.append(cell('A2e_%s_N%d_gateB1' % (nm, best_n), *W1, exits=spec, slots=best_n,
                      mask=mask, index_gate='niftybees100sma_weekly',
                      gate_action='block_new'))
        g.append(cell('A2f_ctrl_%s_N%d' % (nm, best_n), *W1, exits=spec, slots=best_n,
                      mask=''))
    g.append(cell('A2e_noexit_N%d_gateB1' % best_n, *W1, exits='none', slots=best_n,
                  mask=mask, index_gate='niftybees100sma_weekly',
                  gate_action='block_new'))
    g.append(cell('A2f_ctrl_noexit_N%d' % best_n, *W1, exits='none', slots=best_n, mask=''))
    # A4 screen dials at the best slot count, with and without the best trail
    dials = {'g10': B7, 'g12': '%s/b6_g12_qual_mc.npz' % R162M,
             'g15': '%s/b5_g15_qual_mc.npz' % R160M}
    for gname, mk in dials.items():
        for k in (0.85, 0.90):
            for tv in (2.0, 5.0):
                g.append(cell('A4_%s_k%d_tv%d_N%d' % (gname, int(k * 100), int(tv), best_n),
                              *W1, exits='none', mask=mk, k=k, tv_floor=tv, slots=best_n))
    return g


def phase_a2b(mask, ctrl):
    """The plateau probe around the phase-2 candidate (near-ATH band k x slot count N),
    then that candidate crossed back through every other axis. A winner whose neighbours
    disagree is noise, so the band is swept before anything is called a result."""
    g = []
    for k in (0.80, 0.825, 0.85, 0.875, 0.90, 0.95):
        for n in (8, 10, 12, 15):
            g.append(cell('A5_k%d_N%d' % (round(k * 1000), n), *W1, exits='none',
                          mask=mask, k=k, slots=n))
    for k in (0.85,):
        for n in (10,):
            tag = 'k%d_N%d' % (round(k * 1000), n)
            g.append(cell('A5_%s_invvol' % tag, *W1, exits='none', mask=mask, k=k,
                          slots=n, weights='invvol'))
            for nm, spec in [('st20_3', 'st_trail:20_3'), ('st10_3', 'st_trail:10_3'),
                             ('st14_4', 'st_trail:14_4')]:
                g.append(cell('A5_%s_%s' % (tag, nm), *W1, exits=spec, mask=mask, k=k,
                              slots=n))
            g.append(cell('A5_%s_st20_3_invvol' % tag, *W1, exits='st_trail:20_3',
                          mask=mask, k=k, slots=n, weights='invvol'))
            g.append(cell('A5_%s_gateB1' % tag, *W1, exits='none', mask=mask, k=k, slots=n,
                          index_gate='niftybees100sma_weekly', gate_action='block_new'))
            for tv in (5.0, 10.0):
                g.append(cell('A5_%s_tv%d' % (tag, int(tv)), *W1, exits='none', mask=mask,
                              k=k, slots=n, tv_floor=tv))
            for r in RANKS:
                if r != 'rs':
                    g.append(cell('A5_%s_%s' % (tag, r), *W1, exits='none', mask=mask,
                                  k=k, slots=n, rank=r))
            # the two controls at the SAME construction: no screen at all, and the
            # screenable sub-universe (four filed fiscal years, nothing else)
            g.append(cell('A5_%s_ctrlnone' % tag, *W1, exits='none', mask='', k=k, slots=n))
            g.append(cell('A5_%s_ctrlhasdata' % tag, *W1, exits='none', mask=ctrl, k=k,
                          slots=n))
            g.append(cell('A5_%s_random' % tag, *W1, exits='none', mask=ctrl, k=k, slots=n,
                          rank='random', offsets=0, seeds=30))
    return g


def phase_a3(finalists):
    """finalists: list of (name, dict-of-cell-overrides)."""
    g = []
    for nm, ov in finalists:
        for tag, (s, e) in (('full', FULL), ('W1', W1), ('W2', W2)):
            g.append(cell('G3_%s_%s' % (nm, tag), s, e, arms='all', **ov))
        for bps in (40.0, 60.0):
            g.append(cell('G3_%s_cost%d' % (nm, int(bps)), *FULL, cost_bps=bps, **ov))
        g.append(cell('G3_%s_cash0' % nm, *FULL, cash_yield=0.0, **ov))
        ov2 = dict(ov)
        if ov2.get('mask', B7):
            ov2['mask_missing'] = 'pass'
            g.append(cell('G3_%s_misspass' % nm, *FULL, **ov2))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase')
    ap.add_argument('--trails', default='st20_3=st_trail:20_3,st10_3=st_trail:10_3')
    ap.add_argument('--best-n', type=int, default=10)
    ap.add_argument('--mask', default=B7)
    ap.add_argument('--finalists', default=None, help='JSON list of [name, overrides]')
    a = ap.parse_args()

    if a.phase == 'a1':
        g = phase_a1()
    elif a.phase == 'a2':
        tr = [tuple(x.split('=', 1)) for x in a.trails.split(',') if x]
        g = phase_a2(tr, a.best_n, a.mask)
    elif a.phase == 'a2b':
        g = phase_a2b(a.mask, '%s/has_data.npz' % R162M)
    elif a.phase == 'a3':
        g = phase_a3(json.load(open(a.finalists)))
    else:
        raise SystemExit('unknown phase %r' % a.phase)

    out = STUDY / 'results' / ('grid_%s.json' % a.phase)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(g, open(out, 'w'), indent=1)
    print('wrote %s: %d cells' % (out, len(g)))


if __name__ == '__main__':
    main()
