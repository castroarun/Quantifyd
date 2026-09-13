# -*- coding: utf-8 -*-
"""research/170 -- reporting: Part B paired tests, NAV curves for the figures, and the
house-format YoY tables for both parts.

    report170.py paired_b     -> results/pairedB.md
    report170.py curves       -> results/curveB_*.csv
    report170.py yoy          -> results/yoyA.md/.html, results/yoyB.md/.html
"""
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / 'results'
ROOT = STUDY.parent.parent
DB = ROOT / 'backtest_data' / 'market_data.db'

B_LABELS = [
    ('BASE_rand', 'incumbent (never swap)'),
    ('A_unre_m010', 'swap, entrant = tv20 (r/166 pre-registered)'),
    ('X_entrs_unre_m010', 'swap, entrant = rs252 (the pick under test)'),
    ('X_entage_unre_m010', 'swap, entrant = base age'),
    ('A_null_p003', 'rate-matched RANDOM swap (the null)'),
]


# ------------------------------------------------------------------ paired, Part B ----
def load_seedstats(stage):
    out = defaultdict(dict)
    p = RES / ('seedstatsB_%s.csv' % stage)
    if not p.exists():
        return out
    for r in csv.DictReader(open(p)):
        out[r['label']][int(r['seed'])] = r
    return out


def pdiff(a, b, key):
    ks = sorted(set(a) & set(b))
    d = [float(a[k][key]) - float(b[k][key]) for k in ks]
    return float(np.median(d)), int(sum(1 for x in d if x > 0)), len(d)


def paired_b():
    lines = ['# research/170 Part B -- paired confirmation on fresh seeds', '',
             'Open Alpha - Base Age, 16 slots @ 6.25%, SuperTrend(14,4) close trail, '
             'after tax, 25 bps a side, idle cash 5.2% post-tax, 2005-01-03 -> 2026-09-11. '
             'Deltas are medians of the per-seed difference on the SAME seed.', '']
    for stage, title in [('fresh', 'FRESH seeds 1001-1030 (never used by r/164 or r/166)'),
                         ('r166', "research/166's own seeds 1-30")]:
        ss = load_seedstats(stage)
        plat = load_seedstats('plat_%s' % stage)
        ss.update(plat)
        if not ss:
            continue
        base = ss.get('BASE_rand', {})
        null = ss.get('A_null_p003', {})
        lines += ['## %s' % title, '',
                  '| cell | CAGR | worst seed | MaxDD | Calmar | dCalmar vs incumbent | '
                  'wins | dCAGR vs incumbent | wins | dCalmar vs null | wins | W1 | W2 | '
                  'swaps/yr | tax paid |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        order = [l for l, _ in B_LABELS] + sorted(plat.keys())
        for lab in order:
            if lab not in ss:
                continue
            rows = ss[lab]
            cg = [float(r['cagr']) for r in rows.values()]
            cal = [float(r['calmar']) for r in rows.values()]
            dd = [float(r['maxdd']) for r in rows.values()]
            desc = dict(B_LABELS).get(lab, lab)
            if lab == 'BASE_rand':
                d1 = d2 = d3 = (0.0, 0, len(cg))
            else:
                d1 = pdiff(rows, base, 'calmar')
                d2 = pdiff(rows, base, 'cagr')
                d3 = pdiff(rows, null, 'calmar') if null else (float('nan'), 0, 0)
            lines.append(
                '| %s | %.2f%% | %.2f%% | %.2f%% | %.3f | %+.3f | %d/%d | %+.2fpp | %d/%d '
                '| %+.3f | %d/%d | %.2f%% | %.2f%% | %.1f | Rs %s |'
                % (desc, np.median(cg), min(cg), np.median(dd), np.median(cal),
                   d1[0], d1[1], d1[2], d2[0], d2[1], d2[2], d3[0], d3[1], d3[2],
                   np.median([float(r['w1_cagr']) for r in rows.values()]),
                   np.median([float(r['w2_cagr']) for r in rows.values()]),
                   np.median([float(r['swaps_per_yr']) for r in rows.values()]),
                   format(int(np.median([float(r['tax_paid']) for r in rows.values()])),
                          ',d')))
        lines.append('')
        # window-level paired wins for the shortlist
        lines += ['Per-window paired CAGR wins against the incumbent, same seed:', '',
                  '| cell | W1 2005-2015 | W2 2016-2026 |', '|---|---:|---:|']
        for lab in order:
            if lab not in ss or lab == 'BASE_rand':
                continue
            a1 = pdiff(ss[lab], base, 'w1_cagr')
            a2 = pdiff(ss[lab], base, 'w2_cagr')
            lines.append('| %s | %+.2fpp on %d/%d | %+.2fpp on %d/%d |'
                         % (dict(B_LABELS).get(lab, lab), a1[0], a1[1], a1[2],
                            a2[0], a2[1], a2[2]))
        lines.append('')
    # cost ladder
    lines += ['## Cost ladder, fresh seeds (Calmar)', '',
              '| cell | 25 bps | 40 bps | 60 bps |', '|---|---:|---:|---:|']
    l25 = load_seedstats('fresh')
    l40 = load_seedstats('cost40_fresh')
    l60 = load_seedstats('cost60_fresh')
    for lab, desc in B_LABELS:
        if lab not in l25:
            continue
        def m(d, k):
            r = d.get(k)
            return np.median([float(x['calmar']) for x in r.values()]) if r else float('nan')
        lines.append('| %s | %.3f | %.3f | %.3f |'
                     % (desc, m(l25, lab), m(l40, lab + '_b40'), m(l60, lab + '_b60')))
    (RES / 'pairedB.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('wrote', RES / 'pairedB.md')


# ------------------------------------------------------------------------- curves -----
def curves():
    for stage in ('fresh',):
        d = RES / ('navsB_%s' % stage)
        for lab, _ in B_LABELS:
            f = d / ('%s.npz' % lab)
            if not f.exists():
                continue
            z = np.load(f)
            df = pd.DataFrame(z['navs'].T,
                              index=pd.to_datetime([str(x)[:10] for x in z['dates']]),
                              columns=['seed%d' % s for s in z['seeds']])
            df.to_csv(RES / ('curveB_%s.csv' % lab))
            print('  curveB_%s.csv' % lab)


# ---------------------------------------------------------------------------- YoY -----
def bench_yearly(sym, lo, hi):
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    q = con.execute("select date, close from market_data_unified where symbol=? and "
                    "timeframe='day' and date>=? and date<=? order by date",
                    (sym, lo, hi)).fetchall()
    con.close()
    s = pd.Series({pd.Timestamp(str(d)[:10]): float(c) for d, c in q if c}).sort_index()
    if s.empty:
        return {}, (float('nan'), float('nan'))
    return year_stats(s)


def year_stats(nav):
    peak = nav.cummax()
    dd = nav / peak - 1.0
    out = {}
    for yr, seg in nav.groupby(nav.index.year):
        prev = nav[nav.index.year < yr]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        out[int(yr)] = [round(float(seg.iloc[-1] / base - 1.0) * 100, 2),
                        round(float(dd[dd.index.year == yr].min()) * 100, 2)]
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    return out, (cagr * 100, float(dd.min()) * 100)


def yoy_table(cols, benches, title, note, out_stem):
    """cols: list of (name, {year:[ret,dd]}, (cagr,maxdd)). benches: same, excluded from
    the best-of picks. House format: each cell is the year's return with the intra-year
    max drawdown beneath it; three best-of columns on the right."""
    years = sorted({y for _, d, _ in cols + benches for y in d})
    hdr = ['Year'] + [c[0] for c in cols] + [b[0] for b in benches] + \
          ['BEST CAGR', 'LEAST DD', 'BEST OVERALL']
    md = ['# %s' % title, '', note, '',
          '| ' + ' | '.join(hdr) + ' |',
          '|' + '---|' * len(hdr)]
    html = ['<table><thead><tr>' + ''.join('<th>%s</th>' % h for h in hdr) +
            '</tr></thead><tbody>']
    for y in years:
        cells, picks = [], []
        for name, d, _ in cols + benches:
            v = d.get(y)
            cells.append('—' if v is None else '%+.1f<br><small>(%.1f)</small>'
                         % (v[0], v[1]))
        for name, d, _ in cols:
            if y in d:
                picks.append((name, d[y][0], d[y][1]))
        if picks:
            bc = max(picks, key=lambda p: p[1])[0]
            ld = max(picks, key=lambda p: p[2])[0]
            bo = max(picks, key=lambda p: p[1] + p[2])[0]
        else:
            bc = ld = bo = '—'
        md.append('| %d | %s | %s | %s | %s |'
                  % (y, ' | '.join(c.replace('<br><small>', ' ').replace('</small>', '')
                                   for c in cells), bc, ld, bo))
        html.append('<tr><td>%d</td>%s<td>%s</td><td>%s</td><td>%s</td></tr>'
                    % (y, ''.join('<td>%s</td>' % c for c in cells), bc, ld, bo))
    summ = ['**CAGR / MaxDD**']
    for name, _, s in cols + benches:
        summ.append('%.2f%% / %.1f%%' % (s[0], s[1]))
    md.append('| ' + ' | '.join(summ) + ' | | | |')
    html.append('<tr><td><b>CAGR / MaxDD</b></td>' +
                ''.join('<td><b>%s</b></td>' % s for s in summ[1:]) +
                '<td></td><td></td><td></td></tr></tbody></table>')
    (RES / (out_stem + '.md')).write_text('\n'.join(md) + '\n', encoding='utf-8')
    (RES / (out_stem + '.html')).write_text('\n'.join(html) + '\n', encoding='utf-8')
    print('wrote', out_stem)


def yoy():
    # ---- Part A: from the `yearly` column of the cell CSVs ---------------------------
    rows = {}
    for fn in ('cellsA_main.csv', 'cellsA_band.csv', 'cellsA_proof.csv'):
        p = RES / fn
        if p.exists():
            for r in csv.DictReader(open(p)):
                rows[r['label']] = r

    def col(label, name):
        r = rows[label]
        y = {int(k): v for k, v in json.loads(r['yearly']).items()}
        return (name, y, (float(r['cagr_net_tax']), float(r['maxdd'])))

    A = [col('A1_b100_N15_mo', 'QS no leeway (rank 15)'),
         col('A1_b150_N15_mo', 'QS incumbent (rank 23)'),
         col('A1_b167_N15_mo', 'QS leeway rank 26'),
         col('A1_b250_N15_mo', 'QS leeway rank 38'),
         col('A2_k085_b150', 'QS band k 0.85')]
    bA = []
    for sym, nm in [('NIFTY50', 'NIFTY 50'), ('NIFTYMIDCAP150', 'Midcap 150')]:
        y, s = bench_yearly(sym, '2018-08-01', '2026-09-10')
        bA.append((nm, y, s))
    yoy_table(A, bA, 'research/170 Part A -- Quality Summit rank leeway, year by year',
              'After tax (20% STCG / 12.5% LTCG, Indian FY netting), net of 25 bps a side, '
              'idle cash 5.2% post-tax, medians across 12 rebalance-day offsets. Each cell '
              'is the year return with the intra-year max drawdown beneath it, measured '
              'from the running peak of the FULL curve. Window 2018-08-01 -> 2026-09-10.',
              'yoyA')

    # ---- Part B: from the fresh-seed NAV matrices ------------------------------------
    B, cal = [], None
    for lab, desc in B_LABELS:
        f = RES / 'navsB_fresh' / ('%s.npz' % lab)
        if not f.exists():
            continue
        z = np.load(f)
        idx = pd.to_datetime([str(x)[:10] for x in z['dates']])
        per = [year_stats(pd.Series(z['navs'][i], index=idx)) for i in range(z['navs'].shape[0])]
        years = sorted({y for d, _ in per for y in d})
        med = {y: [round(float(np.median([d[y][0] for d, _ in per if y in d])), 2),
                   round(float(np.median([d[y][1] for d, _ in per if y in d])), 2)]
               for y in years}
        s = (float(np.median([x[1][0] for x in per])),
             float(np.median([x[1][1] for x in per])))
        B.append((desc, med, s))
        cal = idx
    bB = []
    if cal is not None:
        y, s = bench_yearly('NIFTYBEES', str(cal[0])[:10], str(cal[-1])[:10])
        bB.append(('NIFTYBEES', y, s))
    yoy_table(B, bB, 'research/170 Part B -- Base Age best-qualifying entrant, year by year',
              'After tax, net of 25 bps a side, idle cash 5.2% post-tax, medians across 30 '
              'FRESH seeds (1001-1030). Each cell is the year return with the intra-year '
              'max drawdown beneath it, measured from the running peak of the FULL curve. '
              'Window 2005-01-03 -> 2026-09-11.',
              'yoyB')


if __name__ == '__main__':
    {'paired_b': paired_b, 'curves': curves, 'yoy': yoy}[sys.argv[1]]()
