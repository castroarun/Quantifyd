# -*- coding: utf-8 -*-
"""research/160 — the house year-on-year comparison table (binding format, Arun 2026-09-04).

One column per system AND per blend in scope, plus the index benchmarks. Each year-cell is
the annual return with the intra-year MAX DRAWDOWN in a smaller muted line beneath it, in
the same cell. Three best-of columns on the right — BEST CAGR, LEAST DD, BEST OVERALL
(return + dd, i.e. return net of intra-year pain) — from which benchmarks are EXCLUDED. A
summary row carries full-period CAGR / MaxDD / Calmar per column with each column's window.

Conventions (r/154, 2026-09-05):
  * per-year return: the seed/offset MEDIAN of each path's calendar-year return
  * per-year drawdown: the MEDIAN of the worst drawdown experienced DURING that year,
    measured from the RUNNING PEAK OF THE FULL CURVE — never from the year's first bar.
    Slicing the year and taking the drawdown within the slice reported -2.4% where the
    truth was -16.5% (r/154 retraction).

Usage:
  yoy_table.py --curves QG=results/A_equity.csv OA=... --out results/yoy_study
               [--benchmarks NIFTY50,NIFTYMIDCAP150,NIFTYSMLCAP250]
Writes <out>.md, <out>.html and <out>.csv.
"""
from __future__ import annotations

import argparse
import html
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
DEFAULT_BENCH = ['NIFTY50', 'NIFTYMIDCAP150', 'NIFTYSMLCAP250']


def load_curves(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime([str(x)[:10] for x in df.index])
    return df.sort_index().astype(float)


def bench_series(sym, start=None, end=None):
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    q = con.execute("select date, close from market_data_unified where symbol=? and "
                    "timeframe='day' order by date", (sym,)).fetchall()
    con.close()
    if not q:
        return None
    s = pd.Series({pd.Timestamp(str(d)[:10]): float(c) for d, c in q if c}).sort_index()
    # index series carry O=H=L=C rows with zero volume; they are REAL sessions, so nothing
    # is filtered here - see qg_panel.py for why the naive phantom test must not apply.
    if start:
        s = s[s.index >= start]
    if end:
        s = s[s.index <= end]
    return s


def year_stats(nav):
    """year -> (return %, worst drawdown % during the year, measured from the running peak
    of the FULL curve)."""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    out = {}
    for yr, seg in nav.groupby(nav.index.year):
        prev = nav[nav.index.year < yr]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        out[int(yr)] = (float(seg.iloc[-1] / base - 1.0) * 100,
                        float(dd[dd.index.year == yr].min()) * 100)
    return out


def full_stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    mdd = float((nav / nav.cummax() - 1).min() * 100)
    return cagr, mdd, (cagr / abs(mdd) if mdd else np.nan)


def build(curves, bench_syms, out):
    rows, summ, windows, is_bench = {}, {}, {}, {}
    for name, path in curves.items():
        df = load_curves(path)
        per = [year_stats(df[c]) for c in df.columns]
        years = sorted({y for p in per for y in p})
        rows[name] = {y: (float(np.median([p[y][0] for p in per if y in p])),
                          float(np.median([p[y][1] for p in per if y in p])))
                      for y in years}
        fs = [full_stats(df[c]) for c in df.columns]
        summ[name] = tuple(float(np.median([f[i] for f in fs])) for i in range(3))
        windows[name] = '%s to %s (%d path%s)' % (df.index[0].date(), df.index[-1].date(),
                                                  len(df.columns),
                                                  '' if len(df.columns) == 1 else 's')
        is_bench[name] = False

    any_curve = load_curves(next(iter(curves.values())))
    lo, hi = any_curve.index[0], any_curve.index[-1]
    for s in bench_syms:
        ser = bench_series(s, lo, hi)
        if ser is None or len(ser) < 200:
            print('benchmark %s: unavailable in the window, skipped' % s)
            continue
        rows[s] = year_stats(ser)
        summ[s] = full_stats(ser)
        windows[s] = '%s to %s (index)' % (ser.index[0].date(), ser.index[-1].date())
        is_bench[s] = True

    order = [n for n in curves if n in rows] + [s for s in bench_syms if s in rows]
    all_years = sorted({y for n in order for y in rows[n]})
    sysnames = [n for n in order if not is_bench[n]]

    recs = []
    for y in all_years:
        rec = {'year': y}
        avail = []
        for n in order:
            v = rows[n].get(y)
            rec['%s_ret' % n] = None if v is None else round(v[0], 1)
            rec['%s_dd' % n] = None if v is None else round(v[1], 1)
            if v is not None and not is_bench[n]:
                avail.append((n, v))
        if avail:
            rec['best_ret'] = max(avail, key=lambda t: t[1][0])[0]
            rec['least_dd'] = max(avail, key=lambda t: t[1][1])[0]
            rec['best_overall'] = max(avail, key=lambda t: t[1][0] + t[1][1])[0]
        recs.append(rec)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(str(out) + '.csv', index=False)

    # ---------------------------------------------------------------- markdown ----------
    M = ['# Year-on-year: return with intra-year drawdown\n',
         'Each cell is the calendar-year return with the worst intra-year drawdown in '
         'parentheses, measured from the running peak of the FULL curve. Ensemble columns '
         'are medians across paths. Best-of columns exclude the benchmarks.\n',
         '| Year | ' + ' | '.join(order) + ' | BEST CAGR | LEAST DD | BEST OVERALL |',
         '|' + '---|' * (len(order) + 4)]
    for r in recs:
        cells = []
        for n in order:
            rr, dd = r['%s_ret' % n], r['%s_dd' % n]
            cells.append('—' if rr is None else '%+.1f (%.1f)' % (rr, dd))
        M.append('| %d | %s | %s | %s | %s |' % (r['year'], ' | '.join(cells),
                                                 r.get('best_ret', ''), r.get('least_dd', ''),
                                                 r.get('best_overall', '')))
    M.append('| **CAGR / MaxDD / Calmar** | ' + ' | '.join(
        '**%.1f / %.1f / %.2f**' % summ[n] for n in order) + ' | | | |')
    M.append('\n**Windows** (they differ; every number is on its own window)\n')
    for n in order:
        M.append('- `%s` — %s' % (n, windows[n]))
    M.append('\nAll system columns are after-tax (20%% STCG / 12.5%% LTCG, Indian FY '
             'netting), net of costs, idle cash 5%% p.a., medians across the seed/offset '
             'ensemble. Benchmarks are price series, no costs or taxes applied.\n')
    Path(str(out) + '.md').write_text('\n'.join(M), encoding='utf-8')

    # -------------------------------------------------------------------- html ----------
    def cell(n, r):
        rr, dd = r['%s_ret' % n], r['%s_dd' % n]
        if rr is None:
            return '<td class="na">—</td>'
        cls = 'pos' if rr >= 0 else 'neg'
        return ('<td><div class="ret %s">%+.1f</div><div class="dd">(%.1f)</div></td>'
                % (cls, rr, dd))
    H = ['<!doctype html><meta charset="utf-8"><title>YoY — return and intra-year drawdown</title>',
         '<style>body{background:#0e1116;color:#e6edf3;font:14px/1.45 system-ui,sans-serif;'
         'margin:0;padding:24px}h1{font-size:20px;margin:0 0 4px}p.sub{color:#8b949e;'
         'margin:0 0 18px;max-width:70ch}table{border-collapse:collapse;width:100%;'
         'font-variant-numeric:tabular-nums}th,td{padding:6px 10px;text-align:right;'
         'border-bottom:1px solid #21262d}th{color:#8b949e;font-weight:600;text-align:right;'
         'position:sticky;top:0;background:#0e1116}th:first-child,td:first-child{text-align:left}'
         '.bench{color:#8b949e}.ret{font-weight:600}.ret.pos{color:#3fb950}.ret.neg{color:#f85149}'
         '.dd{font-size:11px;color:#8b949e;margin-top:1px}.na{color:#484f58}'
         'tr.summary td,tr.summary th{border-top:2px solid #30363d;font-weight:700;'
         'background:#161b22}.best{color:#e3b341}</style>',
         '<h1>Year-on-year: return with intra-year drawdown</h1>',
         '<p class="sub">Each cell: calendar-year return, with the worst intra-year '
         'drawdown beneath it measured from the running peak of the full curve. Ensemble '
         'columns are medians across paths; system columns are after-tax and net of costs. '
         'Best-of columns exclude benchmarks.</p>',
         '<table><thead><tr><th>Year</th>' +
         ''.join('<th%s>%s</th>' % (' class="bench"' if is_bench[n] else '', html.escape(n))
                 for n in order) +
         '<th class="best">BEST CAGR</th><th class="best">LEAST DD</th>'
         '<th class="best">BEST OVERALL</th></tr></thead><tbody>']
    for r in recs:
        H.append('<tr><td>%d</td>%s<td class="best">%s</td><td class="best">%s</td>'
                 '<td class="best">%s</td></tr>'
                 % (r['year'], ''.join(cell(n, r) for n in order), r.get('best_ret', ''),
                    r.get('least_dd', ''), r.get('best_overall', '')))
    H.append('<tr class="summary"><td>CAGR / MaxDD / Calmar</td>' +
             ''.join('<td>%.1f / %.1f / %.2f</td>' % summ[n] for n in order) +
             '<td></td><td></td><td></td></tr>')
    H.append('</tbody></table><p class="sub" style="margin-top:16px">Windows: ' +
             ' &nbsp;·&nbsp; '.join('<b>%s</b> %s' % (html.escape(n), windows[n])
                                    for n in order) + '</p>')
    Path(str(out) + '.html').write_text('\n'.join(H), encoding='utf-8')
    print('wrote %s.md / .html / .csv' % out)
    for n in order:
        print('  %-28s CAGR %6.2f  MaxDD %7.2f  Calmar %5.2f   %s'
              % (n, summ[n][0], summ[n][1], summ[n][2], windows[n]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curves', nargs='+', required=True, help='NAME=path/to/equity.csv')
    ap.add_argument('--benchmarks', default=','.join(DEFAULT_BENCH))
    ap.add_argument('--out', required=True, help='output stem (no extension)')
    a = ap.parse_args()
    curves = {}
    for spec in a.curves:
        name, _, path = spec.partition('=')
        if not path:
            raise SystemExit('--curves takes NAME=path, got %r' % spec)
        curves[name] = path
    build(curves, [s for s in a.benchmarks.split(',') if s], a.out)


if __name__ == '__main__':
    main()
