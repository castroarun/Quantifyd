# -*- coding: utf-8 -*-
"""research/160 G4 - portfolio fit: correlation and blend value against the live pair.

    blend.py <qg_equity.csv> [--name QG] [--out results/g4_blend.md]

Path convention copied verbatim from r/154 `blend_matrix.py` so the numbers are comparable
with the frontier work: a PATH = (OA seed s in 1..30, TN offset o in 0..11) -> 360 paths.
QG carries 12 rebalance-day offsets, so it is tiled across the seeds exactly like TN. Every
A-vs-B number is PAIRED on the path; unpaired medians lie at small n.

Monthly returns, month-end NAVs, restricted to the OVERLAP of all three systems, which the
short fundamentals window pins at 2018-08 -> 2026-08. Drawdowns are measured from the
running peak of the FULL curve, never from a window's own first bar (the r/154 retraction).

The CASH NULL at the same weight is not decoration: r/146 killed a whole sleeve family by
showing that holding cash in its place did the same job. A sleeve must beat cash before it
is allowed to beat nothing.
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
STUDY = Path(__file__).resolve().parents[1]

NSEED, NOFF = 30, 12
NPATH = NSEED * NOFF
END_MONTH = '2026-08'
WINDOWS = {'2020 crash': ('2020-02', '2020-04'), '2022H1 grind': ('2022-01', '2022-06')}


def monthly_nav(daily):
    m = daily.resample('ME').last()
    m.index = m.index.to_period('M')
    return m


def load(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime([str(x)[:10] for x in df.index])
    return monthly_nav(df.sort_index().astype(float))


def stats(nav, years):
    cagr = (nav[-1] ** (1.0 / years) - 1.0) * 100.0
    run = np.maximum.accumulate(nav, axis=0)
    dd = (nav / run - 1.0).min(axis=0) * 100.0
    return cagr, dd, np.where(dd < 0, cagr / np.abs(dd), np.nan)


def band(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)), float(np.min(x)), float(np.max(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('curve')
    ap.add_argument('--name', default='QG')
    ap.add_argument('--out', default=str(STUDY / 'results' / 'g4_blend.md'))
    a = ap.parse_args()

    oa, tn, qg = load(R154 / 'oa_navs30.csv'), load(R154 / 'tn_navs12.csv'), load(a.curve)
    idx = oa.index.intersection(tn.index).intersection(qg.index)
    idx = idx[idx <= pd.Period(END_MONTH, 'M')]
    years = (len(idx) - 1) / 12.0
    L = ['# research/160 G4 - correlation and blend value',
         '',
         'Overlap window **%s .. %s** (%.2f years), monthly returns, 360 paths '
         '(OA seed x TN offset; %s tiled across seeds like TN). Drawdowns from the running '
         'peak of the full curve. Cash sleeve 5%% p.a.' % (idx[0], idx[-1], years, a.name),
         '']

    def rmat(df, kind):
        r = df.loc[idx].pct_change().fillna(0.0).values.astype(np.float64)
        return np.repeat(r, NOFF, axis=1) if kind == 'seed' else np.tile(r, (1, NSEED))

    R = {'OA': rmat(oa, 'seed'), 'TN': rmat(tn, 'off'), a.name: rmat(qg, 'off'),
         'CASH': np.full((len(idx), NPATH), 1.05 ** (1 / 12) - 1.0)}
    cum = lambda r: np.cumprod(1.0 + r, axis=0)                            # noqa: E731

    # ---- correlations, path by path -------------------------------------------------
    L += ['## Monthly return correlation (median across the 360 paths)', '',
          '| pair | median | min | max |', '|---|---:|---:|---:|']
    for x, y in [(a.name, 'TN'), (a.name, 'OA'), ('TN', 'OA')]:
        c = []
        for p in range(NPATH):
            u, v = R[x][:, p], R[y][:, p]
            if u.std() and v.std():
                c.append(float(np.corrcoef(u, v)[0, 1]))
        m, lo, hi = band(c)
        L.append('| %s vs %s | **%.3f** | %.3f | %.3f |' % (x, y, m, lo, hi))
    L.append('')
    L.append('A complement is normally wanted below ~0.40 monthly (the bar r/154 used).')
    L.append('')

    # ---- blends ---------------------------------------------------------------------
    pair = cum(0.5 * R['OA'] + 0.5 * R['TN'])
    rows = [('TN+OA 50-50 (the deployed pair)', pair)]
    for w in (0.10, 0.20, 0.25, 0.33, 0.40):
        rows.append(('+ %s at %d%%' % (a.name, int(w * 100)),
                     cum((1 - w) * (0.5 * R['OA'] + 0.5 * R['TN']) + w * R[a.name])))
        rows.append(('+ CASH at %d%% (the null)' % int(w * 100),
                     cum((1 - w) * (0.5 * R['OA'] + 0.5 * R['TN']) + w * R['CASH'])))
    rows.append(('%s standalone' % a.name, cum(R[a.name])))
    rows.append(('TN standalone', cum(R['TN'])))
    rows.append(('OA standalone', cum(R['OA'])))

    L += ['## Blend value against the deployed TN+OA pair', '',
          '| book | CAGR | [min..max] | MaxDD | Calmar | dCalmar vs pair | paired wins |',
          '|---|---:|---:|---:|---:|---:|---:|']
    bc, bd, bk = stats(pair, years)
    for name, nav in rows:
        c, d, k = stats(nav, years)
        mc, lo, hi = band(c)
        dk = k - bk
        L.append('| %s | %.2f | [%.2f..%.2f] | %.2f | %.3f | %+.3f | %d/%d |'
                 % (name, mc, lo, hi, float(np.median(d)), float(np.nanmedian(k)),
                    float(np.nanmedian(dk)), int(np.nansum(dk > 0)), NPATH))
    L.append('')

    # ---- stress windows --------------------------------------------------------------
    L += ['## Stress windows (return %, drawdown from the full-curve peak)', '',
          '| window | ' + ' | '.join(n for n, _ in rows[:3]) + ' |',
          '|---|' + '---:|' * 3]
    for wn, (s, e) in WINDOWS.items():
        sel = np.where((idx >= pd.Period(s, 'M')) & (idx <= pd.Period(e, 'M')))[0]
        cs = []
        for _, nav in rows[:3]:
            run = np.maximum.accumulate(nav, axis=0)
            dd = (nav[sel] / run[sel] - 1).min(axis=0)
            seg = nav[sel] / nav[sel[0]]
            cs.append('%+.1f (%.1f)' % (np.median(seg[-1] - 1) * 100, np.median(dd) * 100))
        L.append('| %s | %s |' % (wn, ' | '.join(cs)))

    Path(a.out).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    sys.exit(main())
