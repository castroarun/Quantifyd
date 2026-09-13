# -*- coding: utf-8 -*-
"""research/172 part C - correlation and blend value of the 52W book against the live
mpf books: TN (True North), OA (Open Alpha . Base Age) and IPO Base.

Re-uses research/168's blend engine UNCHANGED (blend_grid.load / blend / metrics), so the
incumbent numbers in this file tie digit-for-digit to the published r/168 study. The 52W
sleeve is added as a 4th leg funded pro-rata out of the incumbent three.

Incumbent baseline (r/168, published basis, 30 paired paths, after tax, 5.2% cash):
    TN+OA 50:50                       CAGR 20.28  DD -26.91  Calmar 0.749
    + IPO-A 25% (the adopted book)    CAGR 21.18  DD -24.01  Calmar 0.885
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/172_52wk_channel_n100/results'
sys.path.insert(0, str(ROOT / 'research/168_three_sleeve_blend/scripts'))
import blend_grid as BG          # noqa: E402

BAR_CALMAR = 0.10                # pre-registered adoption bar
BAR_DD = 2.0


def main():
    idx, S, bench = BG.load()
    z = np.load(RES / 'curves.npz', allow_pickle=True)
    d52 = pd.DatetimeIndex(pd.to_datetime(z['dates']))
    keys = [k for k in z.files if k != 'dates']
    curves = {k: pd.Series(z[k], index=d52) for k in keys}

    # align the 52W curves onto the r/168 common window
    A = {k: (v.reindex(idx).ffill().bfill()).to_numpy(float) for k, v in curves.items()}
    A = {k: v / v[0] for k, v in A.items()}

    # ---------------------------------------------------------------- correlations
    ref = {'TN': S['TN_15'], 'OA BaseAge': S['BA_25'], 'IPO-A': S['A_25']}
    corr = {'daily': {}, 'monthly': {}}
    mi = pd.DatetimeIndex(idx)
    for name, arr in A.items():
        s = pd.Series(arr, index=mi)
        rd = s.pct_change().dropna()
        rm = s.resample('ME').last().pct_change().dropna()
        corr['daily'][name] = {}
        corr['monthly'][name] = {}
        for rn, M in ref.items():
            med = np.median(M, axis=0)
            t = pd.Series(med, index=mi)
            corr['daily'][name][rn] = round(float(rd.corr(t.pct_change().dropna())), 4)
            corr['monthly'][name][rn] = round(
                float(rm.corr(t.resample('ME').last().pct_change().dropna())), 4)
        b = pd.Series(bench, index=mi)
        corr['daily'][name]['NIFTYBEES'] = round(float(rd.corr(b.pct_change().dropna())), 4)
        corr['monthly'][name]['NIFTYBEES'] = round(
            float(rm.corr(b.resample('ME').last().pct_change().dropna())), 4)
    json.dump(corr, open(RES / 'correlations.json', 'w'), indent=1)
    print(json.dumps(corr, indent=1), flush=True)

    # ---------------------------------------------------------------- blends
    bnds = BG.boundaries(idx, 'monthly')
    P = 30
    T = len(idx)

    def tile(a):
        return np.tile(a, (P, 1))

    cand_names = ['52W_OPT', '52W_Spec_A', '52W_OPT_(PIT-100)']
    cands = {k: tile(A[k]) for k in cand_names if k in A}
    cands['CASH'] = S['CASH_25']
    cands['EW_B&H_Nifty100'] = tile(A['EW_B&H_Nifty100'])

    rows = []
    # incumbent: TN 37.5 / OA 37.5 / IPO-A 25  (the adopted r/168 book)
    base_stack = np.stack([S['TN_15'], S['BA_25'], S['A_25']])
    base_w = np.array([0.375, 0.375, 0.25])
    base = BG.blend(base_stack, base_w, bnds)
    bm = BG.metrics(base, idx)
    rows.append(dict(cand='(incumbent TN37.5/OA37.5/IPO25)', w=0,
                     cagr=round(float(np.median(bm['cagr'])), 2),
                     maxdd=round(float(np.median(bm['maxdd'])), 2),
                     calmar=round(float(np.median(bm['calmar'])), 3),
                     cagr_worst=round(float(bm['cagr'].min()), 2),
                     dd_worst=round(float(bm['maxdd'].min()), 2),
                     d_calmar=0.0, d_cagr=0.0, d_dd=0.0, wins=0))
    # also the 50:50 TN+OA pair for reference
    p2 = BG.blend(np.stack([S['TN_15'], S['BA_25']]), np.array([0.5, 0.5]), bnds)
    m2 = BG.metrics(p2, idx)
    rows.append(dict(cand='(TN+OA 50:50)', w=0,
                     cagr=round(float(np.median(m2['cagr'])), 2),
                     maxdd=round(float(np.median(m2['maxdd'])), 2),
                     calmar=round(float(np.median(m2['calmar'])), 3),
                     cagr_worst=round(float(m2['cagr'].min()), 2),
                     dd_worst=round(float(m2['maxdd'].min()), 2),
                     d_calmar=None, d_cagr=None, d_dd=None, wins=None))

    for cn, C in cands.items():
        for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.33):
            rest = 1.0 - w
            stack = np.stack([S['TN_15'], S['BA_25'], S['A_25'], C])
            ww = np.array([0.375 * rest, 0.375 * rest, 0.25 * rest, w])
            nav = BG.blend(stack, ww, bnds)
            m = BG.metrics(nav, idx)
            wins = int((m['calmar'] > bm['calmar']).sum())
            rows.append(dict(cand=cn, w=int(w * 100),
                             cagr=round(float(np.median(m['cagr'])), 2),
                             maxdd=round(float(np.median(m['maxdd'])), 2),
                             calmar=round(float(np.median(m['calmar'])), 3),
                             cagr_worst=round(float(m['cagr'].min()), 2),
                             dd_worst=round(float(m['maxdd'].min()), 2),
                             d_calmar=round(float(np.median(m['calmar'] - bm['calmar'])), 3),
                             d_cagr=round(float(np.median(m['cagr'] - bm['cagr'])), 2),
                             d_dd=round(float(np.median(m['maxdd'] - bm['maxdd'])), 2),
                             wins=wins))
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'blend.csv', index=False)
    pd.set_option('display.width', 250)
    print(df.to_string(index=False), flush=True)

    verdict = {}
    for cn in cand_names:
        sub = df[df.cand == cn]
        if not len(sub):
            continue
        best = sub.loc[sub.d_calmar.idxmax()]
        cash = df[(df.cand == 'CASH') & (df.w == best.w)].iloc[0]
        verdict[cn] = dict(
            best_weight=int(best.w), d_calmar=float(best.d_calmar),
            d_cagr=float(best.d_cagr), d_dd=float(best.d_dd), wins_of_30=int(best.wins),
            cash_null_d_calmar=float(cash.d_calmar),
            beats_cash_null=bool(best.d_calmar > cash.d_calmar),
            clears_bar=bool((best.d_calmar >= BAR_CALMAR or best.d_dd >= BAR_DD)
                            and best.d_cagr >= 0 and best.d_calmar > cash.d_calmar))
    json.dump(verdict, open(RES / 'blend_verdict.json', 'w'), indent=1)
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == '__main__':
    main()
