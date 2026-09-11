# -*- coding: utf-8 -*-
"""research/162 Part C — correlation and blend value against the HONEST pair.

research/160 ran this test against the PUBLISHED Open Alpha. research/159 has since shown
that book's headline rests on a same-bar look-ahead fill, so r/160's blend table is
unplaceable and is re-run here against the pair Arun's money is actually in:

    True North  +  Open Alpha · Base Age,  50-50, rebalanced monthly.

True North's ensemble is r/154's 12 rebalance-day offsets (`tn_navs12.csv`); Base Age's is
the 30-seed ensemble this study generated in Part B from r/161's engine, whose no-mask
control reproduces r/161's published row exactly. A PATH is (Base Age seed s in 1..30,
True North offset o in 0..11) -> 360 paths, the r/154 convention. Quality Summit carries 12
rebalance-day offsets, so it tiles across the seeds exactly like True North.

Monthly returns on month-end NAVs, restricted to the overlap, which the 8-year fundamentals
window pins at 2018-08 -> 2026-08. Drawdowns are measured from the running peak of the FULL
curve. The CASH NULL at the same weight is not decoration: a sleeve must beat holding cash
in its place before it is allowed to beat nothing.

**Stated inconsistency:** the True North curve carries idle cash at 6.5% p.a. (r/159's
convention), Base Age at 5.5% (r/161's) and Quality Summit at 5.0% (r/160's). The pair is
therefore flattered by a few tenths of a point relative to the candidate — the bias runs
against the candidate, which is the safe direction, and it is not corrected because the
curves are other studies' artefacts.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
R154 = ROOT / 'research/154_multi_system_blends/results'
RES = Path(__file__).resolve().parents[1] / 'results'

START, END_M = '2018-08-01', '2026-08'
CASH_YIELD = 0.05
WINDOWS = {'2020 crash': ('2020-02', '2020-04'), '2022H1 grind': ('2022-01', '2022-06')}
WEIGHTS = (0.10, 0.20, 0.33)


def load(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime([str(x)[:10] for x in df.index])
    return df.sort_index().astype(float)


def monthly(df):
    m = df.resample('ME').last()
    m.index = m.index.to_period('M')
    return m


def stats(nav, years):
    c = (nav[-1] ** (1.0 / years) - 1.0) * 100.0
    run = np.maximum.accumulate(nav, axis=0)
    dd = (nav / run - 1.0).min(axis=0) * 100.0
    return c, dd, np.where(dd < 0, c / np.abs(dd), np.nan)


def band(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)), float(np.min(x)), float(np.max(x))


def main():
    ba_d = load(RES / 'baseage_navs30.csv')
    tn_d = load(R154 / 'tn_navs12.csv')
    qs = {'QS baseline': load(RES / 'QS_base_equity.csv'),
          'QS best (W1)': load(RES / 'QS_v2_equity.csv')}

    idx_d = ba_d.index.intersection(tn_d.index)
    for v in qs.values():
        idx_d = idx_d.intersection(v.index)
    idx_d = idx_d[idx_d >= pd.Timestamp(START)]

    ba, tn = monthly(ba_d.loc[idx_d]), monthly(tn_d.loc[idx_d])
    idx = ba.index.intersection(tn.index)
    idx = idx[idx <= pd.Period(END_M, 'M')]
    years = (len(idx) - 1) / 12.0
    NSEED, NOFF = ba.shape[1], tn.shape[1]
    NPATH = NSEED * NOFF

    def rmat(df, kind):
        r = df.loc[idx].pct_change().fillna(0.0).values.astype(float)
        return np.repeat(r, NOFF, axis=1) if kind == 'seed' else np.tile(r, (1, NSEED))

    R = {'BA': rmat(ba, 'seed'), 'TN': rmat(tn, 'off'),
         'CASH': np.full((len(idx), NPATH), (1 + CASH_YIELD) ** (1 / 12) - 1.0)}
    for k, v in qs.items():
        R[k] = rmat(monthly(v.loc[idx_d]), 'off')
    cum = lambda r: np.cumprod(1.0 + r, axis=0)                            # noqa: E731

    L = ['# research/162 Part C — correlation and blend value vs True North + Base Age', '',
         'Overlap window **%s .. %s** (%.2f years), monthly returns, %d paths '
         '(Base Age seed x True North offset; Quality Summit tiled across seeds like True '
         'North). Drawdowns from the running peak of the full curve. Cash sleeve %.1f%% '
         'p.a. After tax, 25 bps a side.'
         % (idx[0], idx[-1], years, NPATH, CASH_YIELD * 100), '',
         '**Cash-yield inconsistency, stated:** True North 6.5%, Base Age 5.5%, '
         'Quality Summit 5.0% — the pair is flattered by a few tenths of a point, i.e. the '
         'bias runs AGAINST the candidate.', '']

    # ---- correlations ---------------------------------------------------------------
    L += ['## Return correlation (median across the %d paths)' % NPATH, '',
          '| pair | monthly | daily |', '|---|---:|---:|']
    d_ret = {'BA': ba_d.loc[idx_d].pct_change().fillna(0.0),
             'TN': tn_d.loc[idx_d].pct_change().fillna(0.0)}
    for k, v in qs.items():
        d_ret[k] = v.loc[idx_d].pct_change().fillna(0.0)
    for x, y in [('QS baseline', 'TN'), ('QS baseline', 'BA'), ('QS best (W1)', 'TN'),
                 ('QS best (W1)', 'BA'), ('TN', 'BA')]:
        cm = []
        for p in range(NPATH):
            u, v = R[x][:, p], R[y][:, p]
            if u.std() and v.std():
                cm.append(float(np.corrcoef(u, v)[0, 1]))
        dx, dy = d_ret[x], d_ret[y]
        cd = [float(np.corrcoef(dx.iloc[:, i % dx.shape[1]],
                                dy.iloc[:, i % dy.shape[1]])[0, 1]) for i in range(60)]
        L.append('| %s vs %s | **%.3f** | %.3f |'
                 % (x, y, band(cm)[0], float(np.median(cd))))
    L += ['', 'A complement is normally wanted below ~0.40 monthly (the r/154 bar).', '']

    # ---- blends ----------------------------------------------------------------------
    pair = cum(0.5 * R['BA'] + 0.5 * R['TN'])
    bc, bd, bk = stats(pair, years)
    rows = [('**TN + Base Age 50-50 (the honest pair)**', pair)]
    for nm in list(qs) + ['CASH']:
        for w in WEIGHTS:
            lbl = ('+ CASH at %d%% (the null)' if nm == 'CASH' else '+ %s at %%d%%%%' % nm) \
                % int(w * 100)
            rows.append((lbl, cum((1 - w) * (0.5 * R['BA'] + 0.5 * R['TN']) + w * R[nm])))
    for nm in ('TN', 'BA') + tuple(qs):
        rows.append(('%s standalone' % nm, cum(R[nm])))

    L += ['## Blend value against the honest pair', '',
          '| book | CAGR | [min..max] | MaxDD | Calmar | dCalmar vs pair | paths improved |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name, nav in rows:
        c, d, k = stats(nav, years)
        mc, lo, hi = band(c)
        dk = k - bk
        L.append('| %s | %.2f | [%.2f..%.2f] | %.2f | %.3f | %+.3f | %d/%d |'
                 % (name, mc, lo, hi, float(np.median(d)), float(np.nanmedian(k)),
                    float(np.nanmedian(dk)), int(np.nansum(dk > 0)), NPATH))
    L.append('')

    # ---- stress windows ---------------------------------------------------------------
    keep = [r for r in rows if 'at 20%' in r[0] or 'honest pair' in r[0]]
    L += ['## Stress windows — return % (intra-window drawdown from the full-curve peak)',
          '', '| window | ' + ' | '.join(n.replace('**', '') for n, _ in keep) + ' |',
          '|---|' + '---:|' * len(keep)]
    for wn, (s, e) in WINDOWS.items():
        sel = np.where((idx >= pd.Period(s, 'M')) & (idx <= pd.Period(e, 'M')))[0]
        cs = []
        for _, nav in keep:
            run = np.maximum.accumulate(nav, axis=0)
            dd = (nav[sel] / run[sel] - 1).min(axis=0)
            seg = nav[sel] / nav[sel[0] - 1 if sel[0] else sel[0]]
            cs.append('%+.1f (%.1f)' % (np.median(seg[-1] - 1) * 100, np.median(dd) * 100))
        L.append('| %s | %s |' % (wn, ' | '.join(cs)))
    L.append('')

    (RES / 'partC_blend.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))

    # ---- daily curves the YoY table and the tearsheet consume -------------------------
    reb = lambda d: (d / d.iloc[0])                                        # noqa: E731
    reb(ba_d.loc[idx_d]).to_csv(RES / 'ref_baseage.csv')
    reb(tn_d.loc[idx_d]).to_csv(RES / 'ref_tn.csv')
    rb = ba_d.loc[idx_d].pct_change().fillna(0.0).values
    rt = tn_d.loc[idx_d].pct_change().fillna(0.0).values
    RB, RT = np.repeat(rb, NOFF, axis=1), np.tile(rt, (1, NSEED))
    prd = np.cumprod(1.0 + 0.5 * RB + 0.5 * RT, axis=0)
    pd.DataFrame(prd, index=idx_d,
                 columns=['s%do%d' % (s, o) for s in range(NSEED)
                          for o in range(NOFF)]).to_csv(RES / 'ref_pair.csv')
    z = np.load(RES / 'partB_navs.npz')
    if 'b7_g10_qual_mc|fail' in z.files:
        ov = pd.DataFrame(np.asarray(z['b7_g10_qual_mc|fail']).T,
                          index=pd.to_datetime([str(x)[:10] for x in z['dates']]),
                          columns=['seed%d' % i for i in range(1, 31)])
        ov = ov.loc[ov.index >= pd.Timestamp(START)]
        (ov / ov.iloc[0]).to_csv(RES / 'ref_baseage_b7overlay.csv')
    print('\nwrote ref_tn.csv, ref_baseage.csv, ref_pair.csv, ref_baseage_b7overlay.csv')


if __name__ == '__main__':
    sys.exit(main())
