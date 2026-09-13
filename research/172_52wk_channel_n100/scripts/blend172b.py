# -*- coding: utf-8 -*-
"""research/172 Phase 2 - correlation + 4-sleeve blend for the Phase 2 winner."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/172_52wk_channel_n100/results'
sys.path.insert(0, str(ROOT / 'research/168_three_sleeve_blend/scripts'))
import blend_grid as BG   # noqa: E402

def main():
    idx, S, bench = BG.load()
    z = np.load(RES / 'p2_curves.npz', allow_pickle=True)
    d = pd.DatetimeIndex(pd.to_datetime(z['dates']))
    want = [k for k in z.files if k.startswith('TRAIL20_T63_g0')]
    z1 = np.load(RES / 'curves.npz', allow_pickle=True)
    A = {}
    for k in want:
        v = pd.Series(z1[k] if k in z1.files else z[k], index=d).reindex(idx).ffill().bfill().to_numpy(float)
        A[k] = v / v[0]
    for k in ('52W_OPT',):
        v = pd.Series(z1[k], index=pd.DatetimeIndex(pd.to_datetime(z1['dates']))).reindex(idx).ffill().bfill().to_numpy(float)
        A[k] = v / v[0]
    mi = pd.DatetimeIndex(idx)
    ref = {'TN': S['TN_15'], 'OA BaseAge': S['BA_25'], 'IPO-A': S['A_25']}
    corr = {'daily': {}, 'monthly': {}}
    for name, arr in A.items():
        s = pd.Series(arr, index=mi); rd = s.pct_change().dropna()
        rm = s.resample('ME').last().pct_change().dropna()
        corr['daily'][name] = {}; corr['monthly'][name] = {}
        for rn, M in ref.items():
            t = pd.Series(np.median(M, axis=0), index=mi)
            corr['daily'][name][rn] = round(float(rd.corr(t.pct_change().dropna())), 4)
            corr['monthly'][name][rn] = round(float(rm.corr(t.resample('ME').last().pct_change().dropna())), 4)
        b = pd.Series(bench, index=mi)
        corr['daily'][name]['NIFTYBEES'] = round(float(rd.corr(b.pct_change().dropna())), 4)
        corr['monthly'][name]['NIFTYBEES'] = round(float(rm.corr(b.resample('ME').last().pct_change().dropna())), 4)
    json.dump(corr, open(RES / 'p2_correlations.json', 'w'), indent=1)
    print(json.dumps(corr, indent=1), flush=True)

    bnds = BG.boundaries(idx, 'monthly'); P = 30
    base = BG.blend(np.stack([S['TN_15'], S['BA_25'], S['A_25']]), np.array([.375, .375, .25]), bnds)
    bm = BG.metrics(base, idx)
    rows = [dict(cand='(incumbent TN37.5/OA37.5/IPO25)', w=0,
                 cagr=round(float(np.median(bm['cagr'])), 2), maxdd=round(float(np.median(bm['maxdd'])), 2),
                 calmar=round(float(np.median(bm['calmar'])), 3), d_calmar=0.0, d_cagr=0.0, d_dd=0.0, wins=0)]
    cands = {k: np.tile(A[k], (P, 1)) for k in A}
    cands['CASH'] = S['CASH_25']
    for cn, C in cands.items():
        for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.33):
            rest = 1.0 - w
            nav = BG.blend(np.stack([S['TN_15'], S['BA_25'], S['A_25'], C]),
                           np.array([.375 * rest, .375 * rest, .25 * rest, w]), bnds)
            m = BG.metrics(nav, idx)
            rows.append(dict(cand=cn, w=int(w * 100),
                             cagr=round(float(np.median(m['cagr'])), 2),
                             maxdd=round(float(np.median(m['maxdd'])), 2),
                             calmar=round(float(np.median(m['calmar'])), 3),
                             d_calmar=round(float(np.median(m['calmar'] - bm['calmar'])), 3),
                             d_cagr=round(float(np.median(m['cagr'] - bm['cagr'])), 2),
                             d_dd=round(float(np.median(m['maxdd'] - bm['maxdd'])), 2),
                             wins=int((m['calmar'] > bm['calmar']).sum())))
    df = pd.DataFrame(rows); df.to_csv(RES / 'p2_blend.csv', index=False)
    pd.set_option('display.width', 250); print(df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
