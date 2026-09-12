# -*- coding: utf-8 -*-
"""research/168 step 5 - the final adoption arithmetic and the house tables.

Adds the piece the first two passes could not do honestly:

THE RISK-MATCHED CASH NULL. Calmar alone CANNOT adjudicate a cash null - cash has zero
drawdown, so Calmar rises without bound as the cash weight rises (100% cash scores Calmar
infinity). Comparing an IPO blend and a cash blend at the SAME WEIGHT therefore flatters cash
at high weights for a trivial reason. The decision-grade question is:

    at the SAME portfolio drawdown, does the IPO sleeve deliver more CAGR than simply
    holding that much of the book in the arbitrage fund?

So for every IPO weight the cash weight is solved (1% grid) that reproduces the IPO blend's
median max drawdown, and the two are compared on CAGR, paired across the 30 paths.

Also: the recommended cell and its neighbourhood, the per-year house table in the binding
format, the stress windows, and the cost ladder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/168_three_sleeve_blend/scripts'))
import blend_grid as G                                        # noqa: E402

RES = G.RES
PUB = dict(tn=15, ba=25, ipo=25)
HAR = dict(tn=25, ba=25, ipo=25)


def mk(S, third, w_ipo, tn_share, bp, bnd):
    r = (100.0 - w_ipo)
    w = np.array([r * tn_share, r * (1 - tn_share), w_ipo]) / 100.0
    nv = np.stack([S['TN_%d' % bp['tn']], S['BA_%d' % bp['ba']],
                   S['%s_%d' % (third, bp['ipo'])]])
    return G.blend(nv, w, bnd)


def main():
    idx, S, bench = G.load()
    bnd = G.boundaries(idx, 'monthly')
    G.selftest(S, idx, {'monthly': bnd})
    out = {}

    # ───────────────────────── 1. risk-matched cash null
    print('\n=== THE RISK-MATCHED CASH NULL (monthly rebalance, TN:BA 50:50 of remainder) ===')
    print('For each IPO weight, the CASH weight is solved so the two books carry the SAME')
    print('median max drawdown. Then CAGR is compared, paired across the 30 paths.\n')
    cash_grid = {}
    for c in range(0, 101):
        m = G.metrics(mk(S, 'CASH', c, 0.5, PUB, bnd), idx)
        cash_grid[c] = (float(np.median(m['cagr'])), float(np.median(m['maxdd'])),
                        m['cagr'])
    rm = []
    print('%-6s %5s  %8s %8s   %8s %8s %6s   %9s %9s'
          % ('arm', 'w', 'CAGR', 'MaxDD', 'cashW', 'cashCAGR', 'cashDD', 'dCAGR', 'wins'))
    for third in ('INC', 'A'):
        for w in (5, 10, 15, 20, 25, 30, 35, 40, 50):
            m = G.metrics(mk(S, third, w, 0.5, PUB, bnd), idx)
            dd = float(np.median(m['maxdd']))
            cstar = min(cash_grid, key=lambda c: abs(cash_grid[c][1] - dd))
            cc, cdd, carr = cash_grid[cstar]
            d = m['cagr'] - carr
            print('%-6s %4d%%  %7.2f%% %7.2f%%   %7d%% %7.2f%% %6.2f%%   %+8.2f %6d/30'
                  % (third, w, float(np.median(m['cagr'])), dd, cstar, cc, cdd,
                     float(np.median(d)), int((d > 0).sum())))
            rm.append(dict(arm=third, w_ipo=w, cagr=round(float(np.median(m['cagr'])), 2),
                           maxdd=round(dd, 2), cash_w=cstar, cash_cagr=round(cc, 2),
                           cash_dd=round(cdd, 2), d_cagr=round(float(np.median(d)), 2),
                           wins=int((d > 0).sum())))
    pd.DataFrame(rm).to_csv(RES / 'risk_matched_cash_null.csv', index=False)
    out['risk_matched'] = rm

    # ───────────────────────── 2. the recommended cell, both bases, + neighbourhood
    print('\n=== THE RECOMMENDED CELL AND ITS NEIGHBOURHOOD ===')
    print('TN:BA held at the DEPLOYED 50:50 of the remainder. monthly rebalance.')
    print('%-10s %7s %8s %8s %8s %9s %10s %7s %7s  %8s %8s %8s %8s'
          % ('arm', 'w_ipo', 'CAGR', 'CAGRwst', 'MaxDD', 'MaxDDwst', 'Calmar', 'WA', 'WB',
             '2008', '2020H1', '2018', '2022H1'))
    nb = []
    for third in ('INC', 'A'):
        for w in (0, 10, 15, 20, 25, 30, 35, 50):
            nv = mk(S, third, w, 0.5, PUB, bnd)
            m = G.metrics(nv, idx)
            row = dict(arm=third if w else '2-sleeve', w_ipo=w,
                       cagr=round(float(np.median(m['cagr'])), 2),
                       cagr_worst=round(float(m['cagr'].min()), 2),
                       maxdd=round(float(np.median(m['maxdd'])), 2),
                       maxdd_worst=round(float(m['maxdd'].min()), 2),
                       calmar=round(float(np.median(m['calmar'])), 3),
                       wa=round(float(np.median(m['WA 2006-2015_cagr'])), 2),
                       wb=round(float(np.median(m['WB 2016-2026_cagr'])), 2))
            for k in G.WINDOWS:
                row[k] = round(float(np.median(m[k + '_ret'])), 2)
                row[k + '_dd'] = round(float(np.median(m[k + '_dd'])), 2)
            nb.append(row)
            print('%-10s %6d%% %7.2f %8.2f %8.2f %9.2f %10.3f %7.2f %7.2f  %8.2f %8.2f '
                  '%8.2f %8.2f'
                  % (row['arm'], w, row['cagr'], row['cagr_worst'], row['maxdd'],
                     row['maxdd_worst'], row['calmar'], row['wa'], row['wb'],
                     row['2008'], row['2020H1'], row['2018'], row['2022H1']))
    pd.DataFrame(nb).to_csv(RES / 'neighbourhood.csv', index=False)
    out['neighbourhood'] = nb

    # ───────────────────────── 3. the decisive paired comparisons at the chosen weight
    print('\n=== PAIRED at the RECOMMENDED weight (25%), monthly, TN:BA 50:50 of remainder ===')
    def pp(a, b, label):
        ma, mb = G.metrics(a, idx), G.metrics(b, idx)
        dc, dd, dk = ma['cagr'] - mb['cagr'], ma['maxdd'] - mb['maxdd'], \
            ma['calmar'] - mb['calmar']
        print('%-46s dCAGR %+7.3f (%2d/30)  dDD %+7.3f (%2d/30 shallower)  '
              'dCalmar %+7.4f (%2d/30)'
              % (label, np.median(dc), (dc > 0).sum(), np.median(dd), (dd > 0).sum(),
                 np.median(dk), (dk > 0).sum()))
        return dict(label=label, d_cagr=round(float(np.median(dc)), 3),
                    cagr_wins=int((dc > 0).sum()), d_dd=round(float(np.median(dd)), 3),
                    dd_wins=int((dd > 0).sum()), d_calmar=round(float(np.median(dk)), 4),
                    calmar_wins=int((dk > 0).sum()))
    pairs = []
    b2 = mk(S, 'INC', 0, 0.5, PUB, bnd)
    for w in (20, 25, 30, 50):
        pairs.append(pp(mk(S, 'A', w, 0.5, PUB, bnd), b2,
                        'A %d%% minus two-sleeve' % w))
        pairs.append(pp(mk(S, 'INC', w, 0.5, PUB, bnd), b2,
                        'INC %d%% minus two-sleeve' % w))
        pairs.append(pp(mk(S, 'A', w, 0.5, PUB, bnd), mk(S, 'INC', w, 0.5, PUB, bnd),
                        'A %d%% minus INC %d%%' % (w, w)))
    pairs.append(pp(mk(S, 'A', 25, 0.5, PUB, bnd), mk(S, 'INC', 30, 0.5, PUB, bnd),
                    'A 25% minus INC 30% (each near its own optimum)'))
    print('\n--- the same, HARMONISED basis (True North also at 25 bps a side)')
    for w in (25,):
        pairs.append(pp(mk(S, 'A', w, 0.5, HAR, bnd), mk(S, 'INC', 0, 0.5, HAR, bnd),
                        'HARM A %d%% minus two-sleeve' % w))
        pairs.append(pp(mk(S, 'A', w, 0.5, HAR, bnd), mk(S, 'INC', w, 0.5, HAR, bnd),
                        'HARM A %d%% minus INC %d%%' % (w, w)))
    pd.DataFrame(pairs).to_csv(RES / 'paired_final.csv', index=False)
    out['paired_final'] = pairs

    # ───────────────────────── 4. the per-year house table
    cols = {}
    cols['TN'] = ('sleeve', S['TN_15'])
    cols['OA BaseAge'] = ('sleeve', S['BA_25'])
    cols['IPO-INC'] = ('sleeve', S['INC_25'])
    cols['IPO-A'] = ('sleeve', S['A_25'])
    cols['TN+OA 50:50'] = ('blend', mk(S, 'INC', 0, 0.5, PUB, bnd))
    cols['+IPO-INC 25%'] = ('blend', mk(S, 'INC', 25, 0.5, PUB, bnd))
    cols['+IPO-A 25%'] = ('blend', mk(S, 'A', 25, 0.5, PUB, bnd))
    cols['+IPO-A 50%'] = ('blend', mk(S, 'A', 50, 0.5, PUB, bnd))
    cols['+CASH 25%'] = ('blend', mk(S, 'CASH', 25, 0.5, PUB, bnd))
    py, summ = {}, {}
    for nm, (kind, arr) in cols.items():
        m = G.metrics(arr, idx)
        k = int(np.argsort(m['cagr'])[len(m['cagr']) // 2])
        py[nm] = G.peryear(arr[k], idx)
        summ[nm] = dict(cagr=round(float(np.median(m['cagr'])), 2),
                        dd=round(float(np.median(m['maxdd'])), 2),
                        calmar=round(float(np.median(m['calmar'])), 3),
                        cagr_worst=round(float(m['cagr'].min()), 2),
                        dd_worst=round(float(m['maxdd'].min()), 2),
                        drawn_path=k)
    py['NIFTYBEES'] = G.peryear(bench, idx)
    bm = G.metrics(bench[None, :], idx)
    summ['NIFTYBEES'] = dict(cagr=round(float(bm['cagr'][0]), 2),
                             dd=round(float(bm['maxdd'][0]), 2),
                             calmar=round(float(bm['calmar'][0]), 3))
    names = list(py)
    picks = [n for n in names if n != 'NIFTYBEES']
    yrs = sorted(py['TN'])
    lines = ['| year | ' + ' | '.join(names) + ' | BEST CAGR | LEAST DD | BEST OVERALL |',
             '|' + '---|' * (len(names) + 4)]
    for y in yrs:
        cells = []
        for n in names:
            v = py[n].get(y)
            cells.append('%+.1f<br><sub>(%+.1f)</sub>' % v if v else '')
        av = {n: py[n][y] for n in picks if y in py[n]}
        bc = max(av, key=lambda n: av[n][0])
        ld = max(av, key=lambda n: av[n][1])
        bo = max(av, key=lambda n: av[n][0] + av[n][1])
        lines.append('| %d | ' % y + ' | '.join(cells)
                     + ' | %s | %s | %s |' % (bc, ld, bo))
    lines.append('| **full** | ' + ' | '.join(
        '**%.2f**<br><sub>%.1f / %.2f</sub>' % (summ[n]['cagr'], summ[n]['dd'],
                                                summ[n]['calmar']) for n in names)
        + ' | | | |')
    md = '\n'.join(lines)
    open(RES / 'peryear_table.md', 'w').write(md)
    print('\n=== PER-YEAR HOUSE TABLE (markdown written to results/peryear_table.md) ===')
    print(md)
    json.dump(dict(peryear=py, summary=summ, out=out,
                   window=[str(idx[0].date()), str(idx[-1].date())]),
              open(RES / 'final_report.json', 'w'), indent=1)

    # ───────────────────────── 5. cost ladder + correlation of the blend to its legs
    print('\n=== COST LADDER on the recommended blends (all three sleeves move together) ===')
    lad = []
    for label, third, w in (('TN+OA 50:50', 'INC', 0), ('+IPO-INC 25%', 'INC', 25),
                            ('+IPO-A 25%', 'A', 25), ('+IPO-A 50%', 'A', 50)):
        line = '%-16s' % label
        for bps, bp in ((25, PUB), (40, dict(tn=40, ba=40, ipo=40)),
                        (60, dict(tn=60, ba=60, ipo=60))):
            m = G.metrics(mk(S, third, w, 0.5, bp, bnd), idx)
            line += '  %3dbps %6.2f%% / %7.2f%% / %5.3f' % (
                bps, np.median(m['cagr']), np.median(m['maxdd']), np.median(m['calmar']))
            lad.append(dict(blend=label, cost_bps=bps,
                            cagr=round(float(np.median(m['cagr'])), 2),
                            maxdd=round(float(np.median(m['maxdd'])), 2),
                            calmar=round(float(np.median(m['calmar'])), 3)))
        print(line)
    pd.DataFrame(lad).to_csv(RES / 'cost_ladder.csv', index=False)

    print('\n=== CORRELATION: the recommended blend vs its legs (monthly returns) ===')
    rec = mk(S, 'A', 25, 0.5, PUB, bnd)
    k = int(np.argsort(G.metrics(rec, idx)['cagr'])[15])
    fr = pd.DataFrame({'+IPO-A 25% blend': rec[k],
                       'TN': S['TN_15'][k], 'OA BaseAge': S['BA_25'][k],
                       'IPO-A': S['A_25'][k], 'IPO-INC': S['INC_25'][k],
                       'NIFTYBEES': bench}, index=idx)
    print(fr.resample('ME').last().pct_change().dropna().corr().round(3).to_string())
    print('\nfinal_report done')


if __name__ == '__main__':
    main()
