# -*- coding: utf-8 -*-
"""research/170 Part A -- build the grid JSONs for qg_engine170.py.

    python3 make_grid170.py proof                  -> gridA_proof.json   (harness proof)
    python3 make_grid170.py a1                     -> gridA_main.json    (42 selection cells)
    python3 make_grid170.py a2 <best_buffer>       -> gridA_band.json    (8 selection cells)
    python3 make_grid170.py val <b1> <b2> <k>      -> gridA_val.json     (validation, not selection)

Every cell is the research/160 Family-B `b7` Quality Summit book with exactly one thing
changed. Idle cash is 5.2% post-tax (Arun's standard) everywhere except the proof cell,
which reproduces research/160's published 5.0% row.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
ROOT = HERE.parent.parent.parent
MASK = str(ROOT / 'research' / '160_quality_growth_near_ath' / 'results' / 'masks_study'
           / 'b7_g10_qual_mc.npz')

W_FULL = ('2018-08-01', '2026-09-10')
W1 = ('2018-08-01', '2022-06-30')
W2 = ('2022-07-01', '2026-09-10')

BUFFERS = [1.0, 1.33, 1.5, 1.67, 2.0, 2.5, 3.0]
SLOTS = [10, 15, 20]
CADENCE = ['monthly', 'quarterly']
KBAND = [0.75, 0.80, 0.85, 0.90, 0.95]


def base(**kw):
    c = dict(start=W_FULL[0], end=W_FULL[1], entry='rebalance', cadence='monthly',
             rank='rs', slots=15, buffer=1.5, retain='strict', state='near', k=0.90,
             tv_floor=2.0, mask=MASK, mask_missing='fail', exits='none',
             index_gate='none', gate_action='block_new', fill='next_open',
             cost_bps=25.0, tax=True, cash_yield=0.052, max_position_pct=0.30,
             offsets=12, seeds=0, arms='all')
    c.update(kw)
    return c


def bl(b):
    return '%03d' % round(b * 100)


def main():
    mode = sys.argv[1]
    out, cells = None, []

    if mode == 'proof':
        # research/160 F_Bb7 published at 5.0% idle cash: 21.19 / -37.07 / 0.58.
        # If this engine does not reproduce it, nothing downstream is trustworthy.
        cells.append(base(label='PROOF_incumbent_y050', cash_yield=0.050))
        cells.append(base(label='PROOF_incumbent_y052', cash_yield=0.052))
        out = RES / 'gridA_proof.json'

    elif mode == 'a1':
        # AXIS A1 -- the rank leeway itself, crossed with book size and cadence.
        for b in BUFFERS:
            for n in SLOTS:
                for cad in CADENCE:
                    cells.append(base(label='A1_b%s_N%02d_%s' % (bl(b), n, cad[:2]),
                                      buffer=b, slots=n, cadence=cad))
        out = RES / 'gridA_main.json'

    elif mode == 'a2':
        # AXIS A2 -- the near-ATH band k, at the deployed leeway and at the leeways A1
        # shortlisted (comma list, e.g. "1.5,1.67,2.5").
        for b in sorted({float(x) for x in sys.argv[2].split(',')} | {1.5}):
            for k in KBAND:
                lab = 'A2_k%03d_b%s' % (round(k * 100), bl(b))
                cells.append(base(label=lab, k=k, buffer=b))
        out = RES / 'gridA_band.json'

    elif mode == 'val':
        # VALIDATION -- not selection cells. Two windows, the cost ladder, the random
        # ranking null, the 'loose' retain probe and the missing-data policy, on the
        # incumbent and on whatever A1/A2 shortlisted.
        finalists = []
        for arg in sys.argv[2:]:
            b, n, cad, k = arg.split(':')
            finalists.append((float(b), int(n), cad, float(k)))
        for (b, n, cad, k) in finalists:
            tag = 'b%s_N%02d_%s_k%03d' % (bl(b), n, cad[:2], round(k * 100))
            kw = dict(buffer=b, slots=n, cadence=cad, k=k)
            cells.append(base(label='V_%s_W1' % tag, start=W1[0], end=W1[1], **kw))
            cells.append(base(label='V_%s_W2' % tag, start=W2[0], end=W2[1], **kw))
            cells.append(base(label='V_%s_c40' % tag, cost_bps=40.0, **kw))
            cells.append(base(label='V_%s_c60' % tag, cost_bps=60.0, **kw))
            cells.append(base(label='V_%s_misspass' % tag, mask_missing='pass', **kw))
            cells.append(base(label='V_%s_cash0' % tag, cash_yield=0.0, **kw))
            # 'loose' retain: keep a name whose RS rank is fine even after it leaves the
            # near-ATH band. A SECOND kind of leeway, and the one that tests whether the
            # rank is the binding constraint at all.
            cells.append(base(label='V_%s_loose' % tag, retain='loose', **kw))
            # the null: same book, same leeway, ranking replaced by a coin toss
            cells.append(base(label='V_%s_randnull' % tag, rank='random', offsets=0,
                              seeds=30, **kw))
        out = RES / 'gridA_val.json'

    else:
        raise SystemExit('unknown mode %r' % mode)

    RES.mkdir(parents=True, exist_ok=True)
    json.dump(cells, open(out, 'w'), indent=1)
    print('%s: %d cells' % (out, len(cells)))


if __name__ == '__main__':
    main()
