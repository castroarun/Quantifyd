# -*- coding: utf-8 -*-
"""research/171 -- reporting: paired tests against BOTH references, the eligibility table,
the house-format YoY table and the figure.

    report171.py paired   -> results/paired171.md
    report171.py elig     -> results/elig171.md
    report171.py yoy      -> results/yoy171.md / .html
    report171.py fig      -> results/r171.png
"""
import csv
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

DESC = {
    'REF_base': 'INCUMBENT - never swap',
    'REF_rot1': 'OA-ROT-1 - k=1, entrant rs252 (staged live)',
    'A_k2': 'k=2 - sell the two weakest',
    'A_k3': 'k=3',
    'A_k4': 'k=4 (= all eligible capped at 25% of the book)',
    'A_k6': 'k=6',
    'A_kall': 'k=all eligible',
    'A_k2_m075': 'k=2, margin 7.5%',
    'A_k2_m125': 'k=2, margin 12.5%',
    'A_k2_spillcash': 'k=2, 2nd loser sold to CASH when no 2nd entrant',
    'A_k2_spilltopup': 'k=2, 2nd loser sold and TOPS UP the best holding',
    'C_hyb_k1': 'hybrid 50/50 entrant + top-up, k=1',
    'C_hyb_k2': 'hybrid 50/50 entrant + top-up, k=2',
    'C_els_k1': 'entrant if a signal is refused, else top up, k=1',
    'C_els_k2': 'entrant if a signal is refused, else top up, k=2',
    'CTRL_sellonly_k1': 'CONTROL sell-only k=1 (no redeploy)',
    'CTRL_sellonly_k2': 'CONTROL sell-only k=2',
    'CTRL_sellonly_k3': 'CONTROL sell-only k=3',
    'CTRL_null_k2': 'NULL random swap, k=2',
    'CTRL_null_k3': 'NULL random swap, k=3',
    'CTRL_measure_k0': 'CONTROL measure-only (fires nothing)',
}


def bdesc(lab):
    if lab in DESC:
        return DESC[lab]
    if lab.startswith('B_S_'):
        _, _, k, x, c = lab.split('_')
        xn = {'rs': 'rs252', 'unre': 'unrealised', 'cush': 'cushion'}[x]
        cn = {'c20': 'cap 2x', 'c30': 'cap 3x', 'c00': 'no cap'}[c]
        return 'TOP-UP the best holding by %s, %s, %s' % (xn, cn, k)
    if lab.startswith('F_split2'):
        return 'TOP-UP split across the best TWO holdings, %s' % lab.split('_')[-1]
    if lab.startswith('F_anyeve'):
        return 'TOP-UP on ANY evening with an eligible loser, %s' % lab.split('_')[-1]
    if lab.startswith('P_'):
        return 'plateau neighbour %s' % lab
    return lab


def load_seedstats(*stages):
    out = defaultdict(dict)
    for stage in stages:
        p = RES / ('seedstats_%s.csv' % stage)
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            out[r['label']][int(r['seed'])] = r
    return out


def pdiff(a, b, key):
    ks = sorted(set(a) & set(b))
    d = [float(a[k][key]) - float(b[k][key]) for k in ks]
    if not d:
        return float('nan'), 0, 0
    return float(np.median(d)), int(sum(1 for x in d if x > 0)), len(d)


def med(rows, k):
    return float(np.median([float(r[k]) for r in rows.values()]))


def paired():
    ss = load_seedstats('main', 'follow', 'follow2', 'follow3', 'plat', 'measure')
    base, rot1 = ss.get('REF_base', {}), ss.get('REF_rot1', {})
    order = [k for k in DESC if k in ss and not k.startswith('B_')]
    order = ([l for l in ['REF_base', 'REF_rot1'] if l in ss]
             + [l for l in ss if l.startswith('A_')]
             + [l for l in ss if l.startswith('B_')]
             + [l for l in ss if l.startswith('F_')]
             + [l for l in ss if l.startswith('P_')]
             + [l for l in ss if l.startswith('C_')]
             + [l for l in ss if l.startswith('CTRL_')])
    seen, uniq = set(), []
    for l in order:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    lines = [
        '# research/171 -- paired tests, 30 seeds (7001-7030)', '',
        'OA - Open Alpha - Base Age. 16 slots @ 6.25%, SuperTrend(14,4) close trail, next-open '
        'fills, after tax (20% STCG / 12.5% LTCG with Indian FY netting), net of 25 bps a side, '
        'idle cash 5.2% post-tax credited daily, 2005-01-03 -> 2026-09-11. Every delta is the '
        'MEDIAN of the per-seed difference on the SAME seed, with the number of seeds won.', '',
        '| cell | CAGR | worst seed | MaxDD | Calmar | dCal vs INCUMBENT | wins | dCAGR vs '
        'INCUMBENT | wins | dCal vs OA-ROT-1 | wins | dCAGR vs OA-ROT-1 | wins |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for lab in uniq:
        r = ss[lab]
        d1 = (0.0, 0, len(r)) if lab == 'REF_base' else pdiff(r, base, 'calmar')
        d2 = (0.0, 0, len(r)) if lab == 'REF_base' else pdiff(r, base, 'cagr')
        d3 = (0.0, 0, len(r)) if lab == 'REF_rot1' else pdiff(r, rot1, 'calmar')
        d4 = (0.0, 0, len(r)) if lab == 'REF_rot1' else pdiff(r, rot1, 'cagr')
        lines.append('| %s | %.2f%% | %.2f%% | %.2f%% | %.3f | %+.3f | %d/%d | %+.2fpp | '
                     '%d/%d | %+.3f | %d/%d | %+.2fpp | %d/%d |'
                     % (bdesc(lab), med(r, 'cagr'),
                        min(float(x['cagr']) for x in r.values()),
                        med(r, 'maxdd'), med(r, 'calmar'),
                        d1[0], d1[1], d1[2], d2[0], d2[1], d2[2],
                        d3[0], d3[1], d3[2], d4[0], d4[1], d4[2]))
    lines += ['', '## Windows -- paired CAGR against the INCUMBENT, same seed', '',
              'W1 = fit 2005-01 -> 2015-12.  W2 = holdout 2016-01 -> 2026-09.  '
              'W3 = 2025-01 -> 2026-09, the regime Arun will watch live -- a REPORTING '
              'window, never a selection window.', '',
              '| cell | W1 | W2 | W3 | W1-W2 gap |', '|---|---:|---:|---:|---:|']
    for lab in uniq:
        if lab == 'REF_base':
            continue
        r = ss[lab]
        a1, a2, a3 = (pdiff(r, base, 'w1_cagr'), pdiff(r, base, 'w2_cagr'),
                      pdiff(r, base, 'w3_cagr'))
        lines.append('| %s | %+.2fpp on %d/%d | %+.2fpp on %d/%d | %+.2fpp on %d/%d | %+.2fpp |'
                     % (bdesc(lab), a1[0], a1[1], a1[2], a2[0], a2[1], a2[2],
                        a3[0], a3[1], a3[2], med(r, 'w1_cagr') - med(r, 'w2_cagr')))
    lines += ['', '## Book behaviour', '',
              '| cell | swaps/yr | top-ups/yr | avg invested | cash refusals | slot refusals | '
              'turnover xNAV | tax paid (Rs 10L book) | max position weight | win rate | '
              'max losing streak | trades/yr | ten-best-trades share | median position % of '
              'the name tv20 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for lab in uniq:
        r = ss[lab]
        lines.append('| %s | %.1f | %.1f | %.1f%% | %.0f | %.0f | %.2fx | Rs %s | %.1f%% | '
                     '%.1f%% | %.0f | %.1f | %.1f%% | %.3f%% |'
                     % (bdesc(lab), med(r, 'swaps_per_yr'), med(r, 'topups_per_yr'),
                        med(r, 'invested_pct'), med(r, 'cash_refused'),
                        med(r, 'days_full'), med(r, 'turnover_x'),
                        format(int(med(r, 'tax_paid')), ',d'), med(r, 'max_pos_w'),
                        med(r, 'win_rate'), med(r, 'max_loss_streak'),
                        med(r, 'trades_per_yr'), med(r, 'top10_share'),
                        med(r, 'cap_pct_med')))
    # cost ladder if present
    l40 = load_seedstats('cost40')
    l60 = load_seedstats('cost60')
    if l40:
        lines += ['', '## Cost ladder (Calmar)', '',
                  '| cell | 25 bps | 40 bps | 60 bps |', '|---|---:|---:|---:|']
        for lab in uniq:
            if (lab + '_b40') not in l40:
                continue
            def m(d, k):
                return med(d[k], 'calmar') if k in d else float('nan')
            lines.append('| %s | %.3f | %.3f | %.3f |'
                         % (bdesc(lab), med(ss[lab], 'calmar'),
                            m(l40, lab + '_b40'), m(l60, lab + '_b60')))
    (RES / 'paired171.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('wrote', RES / 'paired171.md')


def elig():
    """How often are TWO holdings simultaneously eligible?  Reported before any CAGR."""
    ss = load_seedstats('main', 'measure', 'follow2')
    lines = ['# research/171 -- the eligibility histogram', '',
             'On an evening when at least one qualifying Base Age signal was REFUSED, how many '
             'open positions were simultaneously more than 10% below their average buy price? '
             'Medians across 30 seeds (7001-7030), whole window 2005-01-03 -> 2026-09-11 '
             '(21.7 years). This is a RECORDING inside the engine: it decides nothing and it '
             'draws no random number.', '',
             '| book | refused-signal evenings | >=1 eligible | >=2 eligible | >=3 eligible | '
             'most ever eligible | mean eligible per evening |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for lab in ('CTRL_measure_k0', 'REF_rot1', 'A_k2', 'A_kall'):
        if lab not in ss:
            continue
        r = ss[lab]
        d = med(r, 'elig_days')
        lines.append('| %s | %.0f | %.0f (%.1f%%) | %.0f (%.1f%%) | %.0f (%.1f%%) | %.0f | %.2f |'
                     % (bdesc(lab), d,
                        med(r, 'elig_ge1'), 100.0 * med(r, 'elig_ge1') / d if d else 0,
                        med(r, 'elig_ge2'), 100.0 * med(r, 'elig_ge2') / d if d else 0,
                        med(r, 'elig_ge3'), 100.0 * med(r, 'elig_ge3') / d if d else 0,
                        med(r, 'elig_max'), med(r, 'elig_mean')))
    lines += ['', 'Per YEAR (divide by 21.7):', '',
              '| book | refused-signal evenings / yr | >=2 eligible / yr | swaps actually '
              'fired / yr |', '|---|---:|---:|---:|']
    for lab in ('CTRL_measure_k0', 'REF_rot1', 'A_k2', 'A_kall'):
        if lab not in ss:
            continue
        r = ss[lab]
        lines.append('| %s | %.1f | %.1f | %.1f |'
                     % (bdesc(lab), med(r, 'elig_days') / 21.7,
                        med(r, 'elig_ge2') / 21.7, med(r, 'swaps_per_yr')))
    (RES / 'elig171.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('wrote', RES / 'elig171.md')


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


def _cell_years(lab):
    for stage in ('main', 'follow', 'follow2', 'follow3', 'plat'):
        f = RES / ('navs_%s' % stage) / ('%s.npz' % lab)
        if f.exists():
            break
    else:
        return None
    z = np.load(f)
    idx = pd.to_datetime([str(x)[:10] for x in z['dates']])
    per = [year_stats(pd.Series(z['navs'][i], index=idx)) for i in range(z['navs'].shape[0])]
    years = sorted({y for d, _ in per for y in d})
    m = {y: [round(float(np.median([d[y][0] for d, _ in per if y in d])), 2),
             round(float(np.median([d[y][1] for d, _ in per if y in d])), 2)] for y in years}
    s = (float(np.median([x[1][0] for x in per])), float(np.median([x[1][1] for x in per])))
    return m, s, idx


def yoy_table(cols, benches, title, note, out_stem):
    years = sorted({y for _, d, _ in cols + benches for y in d})
    hdr = ['Year'] + [c[0] for c in cols] + [b[0] for b in benches] + \
          ['BEST CAGR', 'LEAST DD', 'BEST OVERALL']
    md = ['# %s' % title, '', note, '', '| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    html = ['<table><thead><tr>' + ''.join('<th>%s</th>' % h for h in hdr) +
            '</tr></thead><tbody>']
    for y in years:
        cells, picks = [], []
        for name, d, _ in cols + benches:
            v = d.get(y)
            cells.append('—' if v is None else '%+.1f<br><small>(%.1f)</small>' % (v[0], v[1]))
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
    summ = ['**CAGR / MaxDD**'] + ['%.2f%% / %.1f%%' % (s[0], s[1]) for _, _, s in cols + benches]
    md.append('| ' + ' | '.join(summ) + ' | | | |')
    html.append('<tr><td><b>CAGR / MaxDD</b></td>' +
                ''.join('<td><b>%s</b></td>' % s for s in summ[1:]) +
                '<td></td><td></td><td></td></tr></tbody></table>')
    (RES / (out_stem + '.md')).write_text('\n'.join(md) + '\n', encoding='utf-8')
    (RES / (out_stem + '.html')).write_text('\n'.join(html) + '\n', encoding='utf-8')
    print('wrote', out_stem)


YOY_COLS = None          # set on the command line: --cols=lab:Name,lab:Name


def yoy():
    spec = None
    for a in sys.argv[1:]:
        if a.startswith('--cols='):
            spec = a.split('=', 1)[1]
    if not spec:
        raise SystemExit('--cols=label:Name,label:Name required')
    cols, idx = [], None
    for part in spec.split(','):
        lab, nm = part.split(':')
        r = _cell_years(lab)
        if r is None:
            raise SystemExit('no NAV matrix for %s' % lab)
        m, s, idx = r
        cols.append((nm, m, s))
    b = []
    y, s = bench_yearly('NIFTYBEES', str(idx[0])[:10], str(idx[-1])[:10])
    b.append(('NIFTYBEES', y, s))
    yoy_table(cols, b,
              'research/171 -- OA Base Age: how many to swap, and where the money goes',
              'After tax (20% STCG / 12.5% LTCG, Indian FY netting), net of 25 bps a side, '
              'idle cash 5.2% post-tax, medians across 30 seeds (7001-7030). Each cell is the '
              'year return with the intra-year max drawdown beneath it, measured from the '
              'running peak of the FULL curve. Window 2005-01-03 -> 2026-09-11. Benchmarks '
              'are excluded from the best-of picks.', 'yoy171')


def fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = None
    for a in sys.argv[1:]:
        if a.startswith('--cols='):
            spec = a.split('=', 1)[1]
    if not spec:
        raise SystemExit('--cols=label:Name,... required')
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[2.1, 1], sharex=True)
    for part in spec.split(','):
        lab, nm = part.split(':')
        for stage in ('main', 'follow', 'follow2', 'follow3', 'plat'):
            f = RES / ('navs_%s' % stage) / ('%s.npz' % lab)
            if f.exists():
                break
        z = np.load(f)
        idx = pd.to_datetime([str(x)[:10] for x in z['dates']])
        cur = pd.DataFrame(z['navs'].T, index=idx).median(axis=1)
        g = cur / cur.iloc[0] * 100.0
        ax[0].plot(g.index, g.values, lw=1.4, label=nm)
        ax[1].plot(g.index, (g / g.cummax() - 1.0) * 100.0, lw=1.0, label=nm)
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    q = con.execute("select date, close from market_data_unified where symbol='NIFTYBEES' "
                    "and timeframe='day' order by date").fetchall()
    con.close()
    s = pd.Series({pd.Timestamp(str(d)[:10]): float(c) for d, c in q if c}).sort_index()
    s = s[(s.index >= idx[0]) & (s.index <= idx[-1])]
    g = s / s.iloc[0] * 100.0
    ax[0].plot(g.index, g.values, lw=1.2, color='#888', ls='--', label='NIFTYBEES')
    ax[1].plot(g.index, (g / g.cummax() - 1.0) * 100.0, lw=0.9, color='#888', ls='--')
    ax[0].set_yscale('log')
    ax[0].set_title('OA - Base Age: how many holdings to swap, and where the proceeds go\n'
                    'growth of Rs 100, log scale, median of 30 seeds, after tax and 25 bps')
    ax[0].set_ylabel('growth of Rs 100 (log)')
    ax[0].grid(alpha=0.25)
    ax[0].legend(fontsize=8, loc='upper left')
    ax[1].set_ylabel('drawdown %')
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RES / 'r171.png', dpi=120)
    print('wrote', RES / 'r171.png')


if __name__ == '__main__':
    {'paired': paired, 'elig': elig, 'yoy': yoy, 'fig': fig}[sys.argv[1]]()
