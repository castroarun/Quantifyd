# -*- coding: utf-8 -*-
"""research/160 - the live-book reference curves, sliced to THIS study's window.

True North and Open Alpha are the two books real money is in. r/154 saved their daily NAV
ensembles (TN 12 rebalance offsets, OA 30 selection seeds) over 2006-2026; this study's
honest window is 2018-08 -> 2026-09, so they have to be sliced and re-based before they can
sit in the same table as a QG column. The 50-50 pair is built the r/154 way: a PATH is
(OA seed s, TN offset o), 360 of them, blended on daily returns and re-compounded, so the
pair's drawdown is a real path's drawdown and not the drawdown of two medians averaged.

    make_ref_curves.py [--qg <equity.csv> --qgw 0.20]

Writes results/ref_tn.csv, ref_oa.csv, ref_pair.csv and, when --qg is given,
ref_blend<W>.csv - the pair diluted to (1-w) with the QG book at w.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
R154 = ROOT / 'research' / '154_multi_system_blends' / 'results'
RES = Path(__file__).resolve().parents[1] / 'results'
START, END = '2018-08-01', '2026-09-10'


def load(p):
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime([str(x)[:10] for x in df.index])
    return df.sort_index().astype(float)


def slice_rebase(df, idx):
    d = df.reindex(idx).ffill().dropna(how='all')
    return d / d.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qg', default=None)
    ap.add_argument('--qgw', type=float, default=0.20)
    a = ap.parse_args()

    tn, oa = load(R154 / 'tn_navs12.csv'), load(R154 / 'oa_navs30.csv')
    idx = tn.index.intersection(oa.index)
    if a.qg:
        idx = idx.intersection(load(a.qg).index)
    idx = idx[(idx >= pd.Timestamp(START)) & (idx <= pd.Timestamp(END))]
    TN, OA = slice_rebase(tn, idx), slice_rebase(oa, idx)
    TN.to_csv(RES / 'ref_tn.csv')
    OA.to_csv(RES / 'ref_oa.csv')

    rt, ro = TN.pct_change().fillna(0.0).values, OA.pct_change().fillna(0.0).values
    # path (s,o): OA seed-major, TN tiled - exactly r/154's pairing
    RT, RO = np.tile(rt, (1, OA.shape[1])), np.repeat(ro, TN.shape[1], axis=1)
    pair = np.cumprod(1.0 + 0.5 * RO + 0.5 * RT, axis=0)
    cols = ['s%do%d' % (s, o) for s in range(OA.shape[1]) for o in range(TN.shape[1])]
    pd.DataFrame(pair, index=idx, columns=cols).to_csv(RES / 'ref_pair.csv')
    print('TN %s  OA %s  pair %s  window %s..%s'
          % (TN.shape, OA.shape, pair.shape, idx[0].date(), idx[-1].date()))

    if a.qg:
        QG = slice_rebase(load(a.qg), idx)
        rq = QG.pct_change().fillna(0.0).values
        # QG carries 12 offsets like TN, so it tiles the same way
        RQ = np.tile(rq, (1, OA.shape[1])) if rq.shape[1] == TN.shape[1] else \
            np.repeat(rq, (RO.shape[1] // rq.shape[1]), axis=1)
        w = a.qgw
        bl = np.cumprod(1.0 + (1 - w) * (0.5 * RO + 0.5 * RT) + w * RQ, axis=0)
        out = RES / ('ref_blend%d.csv' % int(w * 100))
        pd.DataFrame(bl, index=idx, columns=cols).to_csv(out)
        print('blend at %d%% -> %s' % (int(w * 100), out))


if __name__ == '__main__':
    sys.exit(main())
