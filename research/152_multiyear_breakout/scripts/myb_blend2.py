"""research/152 — CORRECTED blend stage.

Fix vs the first pass: `blend3.py`-style code computed the TN+OA 50-50 baseline on the
OA-and-TN intersection (2006 ->) while every w3>0 blend ran on the candidate's shorter
window (2010 ->). That put the 2008 crash in the baseline and not in the blends, so the
"DD improvement" was partly a window swap. Here EVERY row - baseline, candidate and
cash-null - is computed on the SAME common window, and A-vs-B is reported PAIRED on the
same (OA seed, TN offset) path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RES = Path('/home/arun/quantifyd/research/152_multiyear_breakout/results')
R146 = Path('/home/arun/quantifyd/research/146_complementary_third_sleeve/results')
W3 = [0.10, 0.15, 0.20, 0.25, 0.33]
OFFSETS = [0, 4, 8]
WINDOWS = {'2011-12': ('2011-01-01', '2012-12-31'), '2013 grind': ('2013-01-01', '2013-12-31'),
           '2015-16': ('2015-08-01', '2016-02-29'), '2018 grind': ('2018-01-01', '2018-10-31'),
           '2020 crash': ('2020-02-01', '2020-04-30'), '2022H1 grind': ('2022-01-01', '2022-06-30'),
           '2024-25': ('2024-01-01', '2025-12-31')}


def stats(nav):
    y = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1
    d = float((nav / nav.cummax() - 1).min())
    return c * 100, d * 100, (c / abs(d) if d < 0 else np.nan)


def wdd(nav, a, b):
    s = nav[(nav.index >= a) & (nav.index <= b)]
    if len(s) < 3:
        return np.nan, np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1) * 100, float((s / s.cummax() - 1).min() * 100)


def main():
    myb = pd.read_csv(RES / 'myb_equity_seeds.csv', index_col=0, parse_dates=True)
    oa = pd.read_csv(R146 / 'oa_navs.csv', index_col=0, parse_dates=True)
    tn = {o: pd.read_csv(R146 / f'tn_nav_off{o}.csv', index_col=0, parse_dates=True).iloc[:, 0]
          for o in OFFSETS if (R146 / f'tn_nav_off{o}.csv').exists()}

    idx = myb.index.intersection(oa.index)
    for v in tn.values():
        idx = idx.intersection(v.index)
    print(f'COMMON WINDOW (all rows use it): {idx[0].date()} -> {idx[-1].date()} '
          f'({(idx[-1]-idx[0]).days/365.25:.1f}y)')
    cash = pd.Series((1 + 0.05 / 252) ** np.arange(len(idx)), index=idx)

    def mret(s):
        return s.loc[idx].resample('ME').last().pct_change().fillna(0)

    m_oa = {c: mret(oa[c]) for c in oa.columns}
    m_tn = {o: mret(v) for o, v in tn.items()}
    m_myb = {c: mret(myb[c]) for c in myb.columns}
    m_cash = mret(cash)

    # correlations on the common window
    print('\n--- correlation of the MYB sleeve (seed median) to each leg, common window ---')
    med = myb.loc[idx].median(axis=1)
    for name, other in [('OA (seed median)', oa.loc[idx].median(axis=1))] + \
                       [(f'TN offset {o}', v.loc[idx]) for o, v in tn.items()]:
        d = med.pct_change().corr(other.pct_change())
        mo = med.resample('ME').last().pct_change().corr(
            other.resample('ME').last().pct_change())
        print(f'  vs {name:20s} daily {d:.3f}   monthly {mo:.3f}')

    oa_cols = list(oa.columns)
    myb_cols = list(myb.columns)
    rows, paired = [], []
    for w3 in [0.0] + W3:
        wl = (1 - w3) / 2
        cs, ds, ks = [], [], []
        cs_c, ds_c, ks_c = [], [], []
        for off in tn:
            for j, c in enumerate(oa_cols):
                third = m_myb[myb_cols[j % len(myb_cols)]]
                b = (1 + wl * m_oa[c] + wl * m_tn[off] + (w3 * third if w3 else 0)).cumprod()
                bc = (1 + wl * m_oa[c] + wl * m_tn[off] + (w3 * m_cash if w3 else 0)).cumprod()
                x = stats(b); y = stats(bc)
                cs.append(x[0]); ds.append(x[1]); ks.append(x[2])
                cs_c.append(y[0]); ds_c.append(y[1]); ks_c.append(y[2])
                if w3:
                    base = (1 + 0.5 * m_oa[c] + 0.5 * m_tn[off]).cumprod()
                    bs = stats(base)
                    paired.append(dict(w3=w3, offset=off, seed=c,
                                       d_cagr=x[0] - bs[0], d_dd=x[1] - bs[1],
                                       d_calmar=x[2] - bs[2],
                                       d_calmar_vs_cash=x[2] - y[2],
                                       d_cagr_vs_cash=x[0] - y[0]))
        rows.append(dict(w3=w3, cand='MYB' if w3 else 'TN+OA 50-50 BASELINE',
                         cagr_med=round(float(np.median(cs)), 2),
                         cagr_min=round(min(cs), 2), dd_med=round(float(np.median(ds)), 2),
                         dd_worst=round(min(ds), 2), calmar_med=round(float(np.median(ks)), 2),
                         calmar_min=round(min(ks), 2)))
        if w3:
            rows.append(dict(w3=w3, cand='cash-null',
                             cagr_med=round(float(np.median(cs_c)), 2),
                             cagr_min=round(min(cs_c), 2),
                             dd_med=round(float(np.median(ds_c)), 2),
                             dd_worst=round(min(ds_c), 2),
                             calmar_med=round(float(np.median(ks_c)), 2),
                             calmar_min=round(min(ks_c), 2)))
    df = pd.DataFrame(rows)
    print('\n--- 3-sleeve blend, ALL rows on the common window (medians over '
          f'{len(tn)} TN offsets x {len(oa_cols)} OA seeds) ---')
    print(df.to_string(index=False))
    df.to_csv(RES / 'blend152_corrected.csv', index=False)

    p = pd.DataFrame(paired)
    print('\n--- PAIRED vs the SAME-PATH TN+OA 50-50 baseline (median delta, win count) ---')
    for w3, g in p.groupby('w3'):
        print(f'  w3={w3:.2f}: dCAGR {g.d_cagr.median():+.2f}pp  dDD {g.d_dd.median():+.2f}pp  '
              f'dCalmar {g.d_calmar.median():+.3f}  Calmar-wins {int((g.d_calmar>0).sum())}/{len(g)}  '
              f'| vs cash-null: dCalmar {g.d_calmar_vs_cash.median():+.3f} '
              f'({int((g.d_calmar_vs_cash>0).sum())}/{len(g)}), '
              f'dCAGR {g.d_cagr_vs_cash.median():+.2f}pp')
    p.to_csv(RES / 'blend152_paired.csv', index=False)

    # per-window behaviour of the sleeve and of the blend at 15%
    print('\n--- per-window: MYB sleeve alone / baseline blend / blend+MYB15% (median) ---')
    wr = []
    for name, (a, b_) in WINDOWS.items():
        sl = wdd(med, a, b_)
        bb, bm = [], []
        for off in tn:
            for j, c in enumerate(oa_cols):
                base = (1 + 0.5 * m_oa[c] + 0.5 * m_tn[off]).cumprod()
                bl = (1 + 0.425 * m_oa[c] + 0.425 * m_tn[off]
                      + 0.15 * m_myb[myb_cols[j % len(myb_cols)]]).cumprod()
                bb.append(wdd(base, a, b_)); bm.append(wdd(bl, a, b_))
        r = dict(window=name,
                 myb_ret=round(sl[0], 1), myb_dd=round(sl[1], 1),
                 base_ret=round(float(np.median([x[0] for x in bb])), 1),
                 base_dd=round(float(np.median([x[1] for x in bb])), 1),
                 blend15_ret=round(float(np.median([x[0] for x in bm])), 1),
                 blend15_dd=round(float(np.median([x[1] for x in bm])), 1))
        wr.append(r)
        print('  ', r)
    pd.DataFrame(wr).to_csv(RES / 'blend152_windows.csv', index=False)
    print('\nBLEND2 DONE')


if __name__ == '__main__':
    main()
