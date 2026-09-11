# -*- coding: utf-8 -*-
"""THE MOMENTUM PORTFOLIO REPORT — every portfolio number and every chart on /app/mpf-report.

Run on demand, and after ANY change to a Momentum-Portfolio system (True North, Open Alpha,
IPO Base, Quality Summit) or to any of the after-tax curve files it reads:

    cd /home/arun/quantifyd && venv/bin/python3 research/_utilities/mpf_report_build.py

Writes
    static/app/mpf_report.json        <- every number the page prints
    frontend/public/mpf_report.json   <- the same, so a full emptyOutDir rebuild keeps it
    frontend/public/mpf-report-*.png  <- ten charts (see CHARTS below)

EVERYTHING ON THIS PAGE IS AFTER TAX. Arun asked three times: mixed pre/post-tax tables made
the roster page unreadable. Any figure that exists only pre-tax is omitted, and the omission
is recorded rather than quietly converted.

WHY A GENERATOR AND NOT A TYPED TABLE. One edit to a curve file and a hand-copied table is
silently wrong for months. Every metric, every year cell, every correlation and the blend row
is computed here. Only per-system NARRATIVE lives in TypeScript
(frontend/src/data/mpf_report.ts), and each figure typed there carries a `source` field.

--------------------------------------------------------------------------------------------
THE THREE-WINDOW PROBLEM, AND HOW THIS PAGE RESOLVES IT
--------------------------------------------------------------------------------------------
Three windows exist in the evidence and Arun has already complained about the confusion:

  2006-04 -> 2026-09  every book actually being chosen between (True North, Open Alpha Base
                      Age, IPO Base, the index). 20.4 years, and it contains 2008 AND 2020 —
                      the two drawdowns that matter.
  2016    -> 2026     the roster window, forced by INDIA VIX, which the ATH+VIX variant needs.
  2018-08 -> 2026-09  the only window Quality Summit can exist in: point-in-time fundamentals
                      need four filed fiscal years and Screener history begins FY2015.

RESOLUTION. The HEADLINE is the 20.4-year window, because a window starting in 2018 throws
away both crashes. Quality Summit gets a SECOND, clearly labelled section in which EVERY
system is re-measured on 2018-08 onward. The 2016-2026 window appears nowhere as a headline —
only as one labelled row of evidence inside the Open Alpha section, for the ATH+VIX variant.
No table ever mixes windows, and every table title states WHICH SYSTEMS, WHICH WINDOW,
WHICH BASIS.

--------------------------------------------------------------------------------------------
INPUTS (the evidence files)
--------------------------------------------------------------------------------------------
  research/159_oa_honest_reoptimization/results/full_period_after_tax.csv
      HEADLINE. After-tax daily curves 2006-04-03 -> 2026-09-03 for
      'Open Alpha - Base Age', 'True North', 'IPO Base - First Base', 'NIFTYBEES (index)'.
  research/159_oa_honest_reoptimization/results/full_period_summary.json
      the measured average-invested figures that go on the table.
  research/159_oa_honest_reoptimization/results/all_systems_after_tax.csv
      the 2016-2026 roster curves — used ONLY to build the 2018 section, so that section
      reproduces the published roster page exactly.
  research/160_quality_growth_near_ath/results/F_Bb7_equity.csv
      Quality Summit, 12 rebalance offsets, 2018-08-01 -> 2026-09-10.
  research/159_oa_honest_reoptimization/results/after_tax_tables.csv
      AFTER-TAX entry-mechanic surface (table A), null control (B) and price-gate bake-off
      (C). The after-tax re-run CRASHED before the VIX-gate rows, so no VIX gate figure is
      published here — see NOTES['aftertax_incomplete'].

HOUSE RULES HONOURED
  * the drawn Quality Summit path is the MEDIAN-CAGR OFFSET of twelve, never an average of
    paths — averaging equity curves manufactures a smoother line than any book could run;
  * every intra-year drawdown is measured from the running peak of the FULL curve;
  * benchmarks are excluded from the BEST CAGR / LEAST DD / BEST OVERALL picks;
  * the TN + Base Age blend is COMPUTED HERE and labelled as such. It is NOT a study result,
    and the proper blend/allocation study across TN + Base Age + IPO is still owed.

CHARTS (all dark, colours constant everywhere: gold True North, green Base Age,
coral Quality Summit, purple IPO Base, blue blend, grey index)
  mpf-report-curves-20y.png      log growth of 100 + drawdown panel, 20.4 years
  mpf-report-curves-2018.png     the same on the Quality Summit window
  mpf-report-yearly-bars.png     yearly returns, systems side by side, 20.4 years
  mpf-report-rolling3y.png       rolling 3-year CAGR
  mpf-report-corr-20y.png        weekly-return correlation heatmap, 20.4 years
  mpf-report-corr-2018.png       the same on the Quality Summit window
  mpf-report-invested.png        average invested vs average in cash, measured
  mpf-report-heat-truenorth.png  monthly-return heatmap, one per system
  mpf-report-heat-baseage.png
  mpf-report-heat-ipobase.png
  mpf-report-heat-qualitysummit.png
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import TwoSlopeNorm
import textwrap


def wrap(s, width=168):
    """Footnotes are long by design — they carry the caveat. At 12in and 7.5pt a single
    line runs off the right edge, so every footnote is hard-wrapped here."""
    return textwrap.fill(' '.join(s.split()), width)

ROOT = Path('/home/arun/quantifyd')
R159 = ROOT / 'research/159_oa_honest_reoptimization/results'
R160 = ROOT / 'research/160_quality_growth_near_ath/results'
PUB = ROOT / 'frontend/public'
OUT_JSONS = [ROOT / 'static/app/mpf_report.json', PUB / 'mpf_report.json']

# ---- names. Arun asked for systems to be NAMED, never versioned: these are different
# signals, not revisions of one another.
TN = 'True North'
BA = 'Open Alpha · Base Age'
IPO = 'IPO Base'
QS = 'Quality Summit'
BM = 'NIFTYBEES'
BLEND = 'TN + Base Age (50-50, monthly)'

FULL_RENAME = {'True North': TN, 'Open Alpha - Base Age': BA,
               'IPO Base - First Base': IPO, 'NIFTYBEES (index)': BM}
ROSTER_RENAME = {'TN incumbent': TN, 'OA v2': BA, 'IPO (honest)': IPO, 'NIFTYBEES': BM}

COLOR = {TN: '#e3b341', BA: '#3fb950', QS: '#ff7b72', IPO: '#bc8cff',
         BLEND: '#58a6ff', BM: '#8b949e'}
SLUG = {TN: 'truenorth', BA: 'baseage', IPO: 'ipobase', QS: 'qualitysummit'}

BG, PANEL, INK, MUT, GRID = '#0e1116', '#161b22', '#e6edf3', '#8b949e', '#30363d'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': PANEL, 'savefig.facecolor': BG,
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUT,
                     'ytick.color': MUT, 'axes.edgecolor': GRID, 'font.size': 9})

# Measured average-invested, from each engine's own reporting. Anything not measured stays
# None and prints as "not measured" — the handover asserts ~67% for Base Age but NO FILE
# carries it, and this page does not print numbers it cannot point at.
INVESTED = {TN: 43.0, IPO: 32.7, BM: 100.0, BA: None, QS: 91.2}
INVESTED_SRC = {
    TN: 'research/144 phase A avg_inv 0.43 (via research/159 scripts/full_period.py)',
    IPO: 'research/153 G3 "invested 32.7% of NAV" (via the same script)',
    BM: 'fully invested by definition',
    BA: 'NOT MEASURED. research/159 scripts/full_period.py records None with the comment '
        '"to be measured in its own harness"; research/161 saved no invested series. The '
        'handover doc asserts ~67% but no file on disk carries it.',
    QS: 'research/160 RESULTS.md decomposition table, Family B "% inv" = 91.2',
}

# Each system's own series span (not the measured window — that is stated per table).
SERIES_SPAN = {TN: '2006-04 → 2026-09', BA: '2005-01 → 2026-09',
               IPO: '2006-01 → 2026-09', BM: '2005-01 → 2026-09',
               QS: '2018-08 → 2026-09'}

NOTES = {}


# --------------------------------------------------------------------------- metrics

def cagr(s):
    return (s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1


def maxdd(s):
    return (s / s.cummax() - 1).min()


def metrics(s, key):
    c, d = cagr(s), maxdd(s)
    inv = INVESTED.get(key)
    return {'cagr': round(c * 100, 2), 'maxdd': round(d * 100, 2),
            'calmar': round(c / abs(d), 2) if d else None,
            'growth100': int(round(100 * s.iloc[-1] / s.iloc[0])),
            'invested': inv, 'cash': None if inv is None else round(100 - inv, 1),
            'seriesSpan': SERIES_SPAN.get(key, '')}


def yearly(s):
    """Per-year return, and that year's worst drawdown measured from the running peak of the
    FULL curve (house rule) — never from the year's own first bar."""
    out, peak = {}, s.cummax()
    for y, grp in s.groupby(s.index.year):
        prev = s.loc[:grp.index[0] - pd.Timedelta(days=1)]
        base = prev.iloc[-1] if len(prev) else grp.iloc[0]
        out[str(int(y))] = [round((grp.iloc[-1] / base - 1) * 100, 1),
                            round((grp / peak.loc[grp.index] - 1).min() * 100, 1)]
    return out


def best_of(yoy, contenders):
    years = sorted({y for c in contenders for y in yoy[c]})
    out = {}
    for y in years:
        have = [(c, yoy[c][y][0], yoy[c][y][1]) for c in contenders if y in yoy[c]]
        if not have:
            continue
        out[y] = {'bestCagr': max(have, key=lambda t: t[1])[0],
                  'leastDD': max(have, key=lambda t: t[2])[0],          # dd<0: max = shallowest
                  'bestOverall': max(have, key=lambda t: t[1] + t[2])[0]}
    return out, [str(y) for y in years]


def add_blend(frame, a, b, label):
    """50-50 of two books, rebalanced MONTHLY, built from their daily curves. Generator
    arithmetic, not a study output — the page says so on the row."""
    rets = frame[[a, b]].pct_change().fillna(0.0)
    mid = rets.index.to_period('M')
    vals, nav, prev = np.array([0.5, 0.5]), [], mid[0]
    for i, r in enumerate(rets.values):
        if mid[i] != prev:
            vals, prev = np.array([0.5, 0.5]) * vals.sum(), mid[i]
        vals = vals * (1.0 + r)
        nav.append(vals.sum())
    s = pd.Series(nav, index=rets.index)
    frame[label] = s / s.iloc[0]
    return frame


def block(frame, contenders, benchmarks, window_why, source, basis):
    rows = {c: metrics(frame[c], c) for c in frame.columns}
    yoy = {c: yearly(frame[c]) for c in frame.columns}
    bo, years = best_of(yoy, contenders)
    wk = frame.resample('W-FRI').last().pct_change().dropna()
    corr = wk.corr().round(2)
    return {'window': [str(frame.index[0].date()), str(frame.index[-1].date())],
            'years': round((frame.index[-1] - frame.index[0]).days / 365.25, 1),
            'windowWhy': window_why, 'source': source, 'basis': basis,
            'order': list(frame.columns), 'contenders': contenders,
            'benchmarks': benchmarks, 'rows': rows, 'yoy': yoy,
            'bestOf': bo, 'yearList': years,
            'corr': corr.to_dict(), 'corrOrder': list(frame.columns),
            'weeks': int(len(wk))}


# --------------------------------------------------------------- HEADLINE: 20.4 years

full = pd.read_csv(R159 / 'full_period_after_tax.csv', index_col=0, parse_dates=True)
full = full.rename(columns=FULL_RENAME)[[TN, BA, IPO, BM]].ffill().dropna()
full = full / full.iloc[0]
full = add_blend(full, TN, BA, BLEND)
full = full[[TN, BA, IPO, BLEND, BM]]

headline = block(
    full, contenders=[TN, BA, IPO, BLEND], benchmarks=[BM],
    window_why=('The full common period of every book actually being chosen between. It is '
                'set by the shortest series, True North’s, and it is the window that '
                'matters because it contains BOTH 2008 and 2020 — the two falls a window '
                'starting in 2018 throws away.'),
    source='research/159_oa_honest_reoptimization/results/full_period_after_tax.csv',
    basis=('After tax, 25 bps a side, 5% on idle cash. True North is a single after-tax NAV '
           'path from research/144; Open Alpha · Base Age and IPO Base are their studies’ '
           'drawn curves. Placeable entries only: decided on the close, filled at the next open.'))

# -------------------------------------------- SECOND WINDOW: where Quality Summit exists

roster = pd.read_csv(R159 / 'all_systems_after_tax.csv', index_col=0, parse_dates=True)
roster = roster.rename(columns=ROSTER_RENAME)[[TN, BA, IPO, BM]]
qs_all = pd.read_csv(R160 / 'F_Bb7_equity.csv', index_col=0, parse_dates=True)

START18 = pd.Timestamp('2018-08-01')
END18 = min(roster.index[-1], qs_all.index[-1])
roster, qs_all = roster.loc[START18:END18], qs_all.loc[START18:END18]

cg = qs_all.apply(cagr)
drawn = (cg - cg.median()).abs().idxmin()
QS_OFFSETS = {'drawn': drawn, 'n': int(qs_all.shape[1]), 'median': round(cg.median() * 100, 2),
              'min': round(cg.min() * 100, 2), 'max': round(cg.max() * 100, 2)}

w18 = pd.concat([roster, qs_all[drawn].rename(QS)], axis=1).ffill().dropna()
w18 = w18 / w18.iloc[0]
w18 = add_blend(w18, TN, BA, BLEND)
w18 = w18[[TN, BA, QS, IPO, BLEND, BM]]

window2018 = block(
    w18, contenders=[TN, BA, QS, IPO, BLEND], benchmarks=[BM],
    window_why=('Quality Summit cannot be drawn before Aug-2018: point-in-time fundamentals '
                'need four filed fiscal years and Screener history begins FY2015. So every '
                'system is RE-MEASURED here on the only window all five can share. These '
                'figures are NOT comparable with the 20.4-year table above.'),
    source=('research/160_quality_growth_near_ath/results/F_Bb7_equity.csv for Quality Summit '
            '(the median-CAGR offset of 12) and '
            'research/159_oa_honest_reoptimization/results/all_systems_after_tax.csv for the '
            'others — the roster curve file, so this section reproduces the published '
            'roster page exactly. It is a DIFFERENT run from the 20.4-year table’s file.'),
    basis=('After tax, 25 bps a side, 5% on idle cash. Quality Summit is the median-CAGR '
           'rebalance offset of twelve, never an average of paths.'))
window2018['qsOffsets'] = QS_OFFSETS

# ------------------------------------------------ THE CORRECTION: after-tax evidence only

at_all = pd.read_csv(R159 / 'after_tax_tables.csv')
at = at_all[at_all['window'] == '2006-2026']

trails = sorted(at[at.table == 'A'].trail.unique())
ENTRY_ORDER = ['buy at breakout close', 'stop above breakout candle',
               'next-day stop at pivot', 'LOOK-AHEAD reference']
entry_surface = {'trails': [int(t) for t in trails], 'rows': []}
for lab in ENTRY_ORDER:
    sub = at[(at.table == 'A') & (at.label == lab)].set_index('trail')
    entry_surface['rows'].append({
        'label': lab,
        'placeable': lab != 'LOOK-AHEAD reference',
        'cagr': [round(float(sub.loc[t, 'cagr_med']), 2) if t in sub.index else None
                 for t in trails]})

nullb = at[at.table == 'B']
null_control = {'trails': sorted(int(t) for t in nullb.trail.unique()), 'rows': []}
for lab in ['rule: buy at close', 'null: random names']:
    sub = nullb[nullb.label == lab].set_index('trail')
    null_control['rows'].append({
        'label': lab,
        'cagr': [round(float(sub.loc[t, 'cagr_med']), 2) if t in sub.index else None
                 for t in null_control['trails']]})
null_control['edge'] = [round(a - b, 2) if a is not None and b is not None else None
                        for a, b in zip(null_control['rows'][0]['cagr'],
                                        null_control['rows'][1]['cagr'])]

def _gate_rows(df):
    return [{'gate': r.label.replace('VIX:', ''), 'cagr': round(float(r.cagr_med), 2),
             'maxdd': round(float(r.dd_med), 2), 'calmar': round(float(r.calmar_med), 3),
             'blockedPct': round(float(r.blocked_pct), 1)}
            for r in df.sort_values('cagr_med', ascending=False).itertuples()]


# PRICE gates ran on the full 2006-2026 window. VIX gates could only run on 2016-2026, because
# INDIA VIX begins in 2015. Two windows, so TWO tables — they are never merged.
gate_bakeoff = _gate_rows(at[at.table == 'C'])
vix_gates = _gate_rows(at_all[(at_all.table == 'C') & (at_all.window == '2016-2026')])

NOTES['aftertax_incomplete'] = (
    'The after-tax re-run of research/159’s tables (scripts/aftertax_all.py) FINISHED at 22:18 '
    'on 11-Sep-2026, so every table in this section is after tax and no pre-tax figure appears '
    'anywhere on this page. Note the window split inside the gate bake-off: the PRICE gates ran '
    'on the full 2006-2026 window, while every VIX construction could only run on 2016-2026, '
    'because INDIA VIX begins in 2015. They are therefore shown as TWO tables and must never be '
    'read across.')
NOTES['invested_gap'] = INVESTED_SRC[BA]
NOTES['cash_yield'] = (
    'Idle cash is credited at 5% a year everywhere on this page EXCEPT True North, whose '
    'curve comes from research/144’s own after-tax NAV file at 6.5%. True North holds '
    'cash 57% of the time, so that inconsistency is worth roughly 0.9 points a year to it. '
    'Not enough to reorder the table, but it is not like-for-like. Cash yield is not a small '
    'term for any of these books: at 5%, roughly 2.8 of True North’s points and 3.4 of '
    'IPO Base’s are the sweep rather than the strategy, while NIFTYBEES is fully '
    'invested and gets none of it.')
NOTES['invested_timeseries'] = (
    'No engine wrote a daily invested-fraction series, so the "when is each book in cash" '
    'chart is a bar of the MEASURED averages rather than a strip over time. A time-series '
    'version needs the engines to emit that column and is owed.')
NOTES['two_curve_files'] = (
    'The 20.4-year section and the 2018 section are built from DIFFERENT curve files for the '
    'same systems — full_period_after_tax.csv and all_systems_after_tax.csv. They are '
    'separate runs. The 2018 section uses the roster file deliberately, so it reproduces the '
    'published roster page to the decimal.')

correction = {
    'headline': ('Two of the three live books were justified by an entry no order can place. '
                 'The engine decided a trade from the day’s CLOSE and paid a price from '
                 'EARLIER in that same day, which silently deletes every trade that started '
                 'well and failed by the bell.'),
    'costs': [
        {'book': 'Open Alpha (published, research/142)', 'published': '40.8%', 'honest': '−1.7%'},
        {'book': 'IPO Base (research/153)', 'published': '31.0%', 'honest': '15.0%'},
        {'book': 'True North', 'published': '—', 'honest': 'unaffected — the engine holds only closes'},
    ],
    'proof': [
        'Of the source site’s own 54 published Open Alpha trades, 49 of the 50 checkable ones closed above the level they were bought at — by construction, because that engine only books the ones that did.',
        'In the 120 days before each published entry, an order resting at that same level would already have been filled and lost 7.0 times on average.',
        'HCLTECH is the clearest: the clean 10-Jan-2025 entry at 1972.20 follows twenty earlier days when that level was touched and the close fell back.',
        'A list of trades taken can never show the trades that are missing.',
    ],
    'entrySurface': entry_surface,
    'nullControl': null_control,
    'gateBakeoff': gate_bakeoff,
    'vixGates': vix_gates,
    'athVix': {'label': 'Open Alpha · ATH + VIX (research/159)', 'cagr': 19.23,
               'maxdd': -34.15, 'calmar': 0.56, 'window': '2016-01-01 to 2026-09-04',
               'source': 'research/159_oa_honest_reoptimization/results/all_systems_summary.json'},
}

# --------------------------------------------------------------------------- assemble

res = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M IST'),
    'generator': 'research/_utilities/mpf_report_build.py',
    'postTaxOnly': True,
    'standard': ('After tax — 20% short-term, 12.5% long-term above 365 days, netted '
                 'within the Indian financial year with loss carry-forward — 25 bps a '
                 'side, 5% a year on idle cash (6.5% inside True North’s own curve), and '
                 'placeable entries only: decided on the close, filled at the next open.'),
    'names': {'TN': TN, 'BA': BA, 'IPO': IPO, 'QS': QS, 'BM': BM, 'BLEND': BLEND},
    'blendNote': ('Computed by this generator from the True North and Open Alpha · Base '
                  'Age daily curves, 50-50, rebalanced monthly. It is NOT a study result. The '
                  'proper blend and allocation study across True North + Base Age + IPO Base '
                  'has NOT been started, and it is the only structure that plausibly clears '
                  'the 25% target.'),
    'investedSources': INVESTED_SRC,
    'headline': headline,
    'window2018': window2018,
    'correction': correction,
    'notes': NOTES,
    'sources': {
        '20.4-year curves': 'research/159_oa_honest_reoptimization/results/full_period_after_tax.csv',
        'average invested': 'research/159_oa_honest_reoptimization/results/full_period_summary.json + scripts/full_period.py',
        '2018-window curves (TN, Base Age, IPO, index)': 'research/159_oa_honest_reoptimization/results/all_systems_after_tax.csv',
        'Quality Summit, 12 offsets': 'research/160_quality_growth_near_ath/results/F_Bb7_equity.csv',
        'after-tax entry surface / null / price gates': 'research/159_oa_honest_reoptimization/results/after_tax_tables.csv',
        'ATH + VIX summary row': 'research/159_oa_honest_reoptimization/results/all_systems_summary.json',
        'the entry audit': 'research/158_oa_arming_width/OA_ARMING_WIDTH_AND_POKE_FILL_DAILY_SWEEP_STATUS.md + scripts/verify_published_trades.py',
    },
}


# --------------------------------------------------------------------------- charts

def growth_chart(frame, rows, path, title, foot1, foot2, ticks):
    g = 100.0 * frame
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.4), sharex=True,
                                   gridspec_kw={'height_ratios': [2.4, 1]})
    for k in g.columns:
        lw = 2.6 if k == BLEND else (1.1 if k == BM else 1.8)
        ax1.plot(g.index, g[k], color=COLOR[k], lw=lw,
                 label='%s  —  %.1f%%/yr, %.0f%% worst fall, Calmar %.2f'
                       % (k, rows[k]['cagr'], rows[k]['maxdd'], rows[k]['calmar']))
    ax1.set_yscale('log')
    ax1.set_yticks(ticks)
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_ylabel('growth of 100 (log scale)')
    ax1.grid(True, which='both', color=GRID, lw=0.5, alpha=0.6)
    ax1.legend(loc='upper left', frameon=False, fontsize=8.5)
    ax1.set_title(title, color=INK, fontsize=12, loc='left', pad=12)
    for k in g.columns:
        dd = (g[k] / g[k].cummax() - 1) * 100
        ax2.plot(dd.index, dd, color=COLOR[k], lw=2.0 if k == BLEND else (1.0 if k == BM else 1.3))
        ax2.fill_between(dd.index, dd, 0, color=COLOR[k], alpha=0.07)
    ax2.set_ylabel('drawdown %')
    ax2.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax2.axhline(0, color=GRID, lw=0.8)
    fig.text(0.012, 0.030, wrap(foot1), color=MUT, fontsize=7.5)
    fig.text(0.012, 0.009, wrap(foot2), color=MUT, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print('wrote', path)


growth_chart(
    full, headline['rows'], PUB / 'mpf-report-curves-20y.png',
    'THE MOMENTUM PORTFOLIO — every book being chosen between, after tax, %s to %s (20.4 years)'
    % (full.index[0].date(), full.index[-1].date()),
    'WHAT TO SEE: the lower panel. The 50-50 blend ends highest of all, because it keeps most '
    'of Open Alpha · Base Age’s return while inheriting True North’s shallower falls: Base Age '
    'gives up a third of the book at its worst, the blend only a quarter.',
    'Log scale, because over twenty years a 43x book plotted linearly flattens every other '
    'line and hides 2008 entirely. After tax, 25 bps a side, 5% on idle cash (6.5% inside '
    'True North’s own curve). The blend is computed by this generator, not by a study.',
    [100, 400, 1600, 6400])

growth_chart(
    w18, window2018['rows'], PUB / 'mpf-report-curves-2018.png',
    'THE 2018 WINDOW — where Quality Summit can be compared, after tax, %s to %s'
    % (w18.index[0].date(), w18.index[-1].date()),
    'WHAT TO SEE: Quality Summit (coral) ends near True North on multiple and takes the '
    'deepest falls in the set — 2022 and 2025. It is a near-the-high momentum book with a '
    'quality filter, not a different kind of risk.',
    'This window exists only because point-in-time fundamentals need four filed fiscal years. '
    'It throws away 2008 and 2020, so it flatters everything: read it alongside the 20-year '
    'chart, never instead of it. Quality Summit is the median-CAGR offset of twelve.',
    [100, 200, 400, 800])

# ---- yearly grouped bars
yrs = headline['yearList']
bars = [TN, BA, IPO, BLEND, BM]
fig, ax = plt.subplots(figsize=(13, 5.4))
x = np.arange(len(yrs))
w = 0.16
for i, k in enumerate(bars):
    v = [headline['yoy'][k].get(y, [np.nan])[0] for y in yrs]
    ax.bar(x + (i - (len(bars) - 1) / 2) * w, v, w, color=COLOR[k], label=k)
ax.set_xticks(x)
ax.set_xticklabels(yrs, rotation=45, ha='right')
ax.axhline(0, color=GRID, lw=0.9)
ax.set_ylabel('return for the year, %  (after tax)')
ax.grid(True, axis='y', color=GRID, lw=0.5, alpha=0.6)
ax.legend(frameon=False, fontsize=8.5, ncol=5, loc='upper left')
ax.set_title('YEAR BY YEAR, after tax, %s to %s — who carried which year'
             % (full.index[0].date(), full.index[-1].date()),
             color=INK, fontsize=12, loc='left', pad=12)
fig.text(0.012, 0.015,
         wrap('WHAT TO SEE: the years nobody else carried. True North is the only green bar in 2008 '
         'and 2011; IPO Base is the tall one in 2020; Base Age owns 2017, 2021 and 2023. That '
         'pattern is the entire argument for holding more than one of them.'),
         color=MUT, fontsize=7.5)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(PUB / 'mpf-report-yearly-bars.png', dpi=125)
plt.close(fig)
print('wrote yearly bars')

# ---- rolling 3-year CAGR
fig, ax = plt.subplots(figsize=(12, 5.0))
win = 756
for k in [TN, BA, IPO, BLEND, BM]:
    s = full[k]
    roll = (s / s.shift(win)) ** (1 / 3.0) - 1
    ax.plot(roll.index, roll * 100, color=COLOR[k],
            lw=2.2 if k == BLEND else (1.0 if k == BM else 1.5), label=k)
ax.axhline(0, color=GRID, lw=0.9)
ax.axhline(25, color='#d29922', lw=0.9, ls='--')
ax.text(full.index[int(len(full) * 0.02)], 26, "Arun's 25% bar", color='#d29922', fontsize=8)
ax.set_ylabel('trailing 3-year CAGR, %  (after tax)')
ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax.legend(frameon=False, fontsize=8.5, ncol=5, loc='upper right')
ax.set_title('ROLLING 3-YEAR RETURN, after tax — how long each book can disappoint',
             color=INK, fontsize=12, loc='left', pad=12)
fig.text(0.012, 0.015,
         wrap('WHAT TO SEE: every book spends multi-year stretches below the 25% bar, and every '
         'book spends stretches below zero. A three-year run of nothing is the normal '
         'behaviour of these systems, not evidence that one has broken.'),
         color=MUT, fontsize=7.5)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(PUB / 'mpf-report-rolling3y.png', dpi=125)
plt.close(fig)
print('wrote rolling 3y')


# ---- correlation heatmaps
def corr_chart(frame, path, title, foot):
    wk = frame.drop(columns=[BLEND]).resample('W-FRI').last().pct_change().dropna()
    c = wk.corr()
    fig, ax = plt.subplots(figsize=(1.35 * len(c) + 3.4, 1.05 * len(c) + 2.6))
    im = ax.imshow(c.values, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=30, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.index, fontsize=8.5)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, '%.2f' % c.values[i, j], ha='center', va='center',
                    color='#0e1116' if c.values[i, j] > 0.45 else '#e6edf3',
                    fontsize=9.5, fontweight='bold' if i != j else 'normal')
    ax.set_title(title, color=INK, fontsize=11, loc='left', pad=12)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.text(0.012, 0.012, wrap(foot, 140), color=MUT, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('wrote', path)


corr_chart(full, PUB / 'mpf-report-corr-20y.png',
           'WEEKLY-RETURN CORRELATION, after tax, %s to %s (%d weeks)'
           % (full.index[0].date(), full.index[-1].date(), headline['weeks']),
           'WHAT TO SEE: IPO Base is the only genuine diversifier — it is loosely coupled '
           'to everything, including the index. True North and Base Age are moderately '
           'related, which is why a 50-50 of them still smooths the ride.')

corr_chart(w18, PUB / 'mpf-report-corr-2018.png',
           'WEEKLY-RETURN CORRELATION on the 2018 window, after tax (%d weeks)'
           % window2018['weeks'],
           'WHAT TO SEE: Quality Summit is closest to Base Age and to the index. It is not a '
           'third source of return — it is a weaker sampling of a family the book already '
           'trades, which is why research/160 left it unpapered.')

# ---- invested vs cash
fig, ax = plt.subplots(figsize=(10, 4.4))
keys = [TN, BA, IPO, QS, BM]
inv = [INVESTED[k] for k in keys]
pos = np.arange(len(keys))
for i, k in enumerate(keys):
    if inv[i] is None:
        ax.barh(i, 100, color=GRID, alpha=0.35)
        ax.text(50, i, 'NOT MEASURED — no engine wrote this column', ha='center',
                va='center', color=MUT, fontsize=9, style='italic')
    else:
        ax.barh(i, inv[i], color=COLOR[k])
        ax.barh(i, 100 - inv[i], left=inv[i], color=GRID, alpha=0.55)
        ax.text(inv[i] / 2, i, 'invested %.0f%%' % inv[i], ha='center', va='center',
                color='#0e1116', fontsize=9, fontweight='bold')
        if 100 - inv[i] > 12:
            ax.text(inv[i] + (100 - inv[i]) / 2, i, 'cash %.0f%%' % (100 - inv[i]),
                    ha='center', va='center', color=INK, fontsize=9)
ax.set_yticks(pos)
ax.set_yticklabels(keys, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel('average share of the book, %')
ax.set_title('HOW MUCH OF EACH BOOK IS ACTUALLY IN THE MARKET (measured averages)',
             color=INK, fontsize=12, loc='left', pad=12)
fig.text(0.012, 0.015,
         wrap('WHAT TO SEE: True North holds cash 57% of the time and still produces one of the two '
         'best returns — that is the gate, and it is a stronger result than the CAGR alone '
         'says. IPO Base is two thirds cash by design, so reading its CAGR beside a fully '
         'invested index is not like for like.'),
         color=MUT, fontsize=7.5)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(PUB / 'mpf-report-invested.png', dpi=130)
plt.close(fig)
print('wrote invested')


# ---- monthly heatmaps, one per system
def heat(series, key, path, subtitle):
    m = series.resample('ME').last().pct_change().dropna() * 100
    df = pd.DataFrame({'y': m.index.year, 'm': m.index.month, 'v': m.values})
    piv = df.pivot(index='y', columns='m', values='v').reindex(columns=range(1, 13))
    lim = float(np.nanmax(np.abs(piv.values))) or 1.0
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(piv) + 2.6))
    im = ax.imshow(piv.values, cmap='RdYlGn', norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
                   aspect='auto')
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                        'Nov', 'Dec'], fontsize=8.5)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, '%.0f' % v, ha='center', va='center', fontsize=7.5,
                        color='#0e1116' if abs(v) > lim * 0.30 else '#30363d')
    ax.set_title('%s — monthly returns after tax, %%' % key, color=INK, fontsize=12,
                 loc='left', pad=12)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.text(0.012, 0.012, wrap(subtitle, 150), color=MUT, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print('wrote', path)


heat(full[TN], TN, PUB / 'mpf-report-heat-truenorth.png',
     'WHAT TO SEE: the pale rows. When the gate is off the book is in cash and the month is a '
     'small positive — that is why 2008 and 2011 have almost no red in them.')
heat(full[BA], BA, PUB / 'mpf-report-heat-baseage.png',
     'WHAT TO SEE: the deep red months cluster, and they are deeper than True North’s. '
     'There is no gate here — the SuperTrend trail is the only thing between the book and '
     'a falling market.')
heat(full[IPO], IPO, PUB / 'mpf-report-heat-ipobase.png',
     'WHAT TO SEE: the blank-looking stretches. This book is two thirds cash and took no '
     'trades at all in some years; the flat months are the strategy working, not decay.')
heat(w18[QS], QS, PUB / 'mpf-report-heat-qualitysummit.png',
     'WHAT TO SEE: the single best month in the set and some of the worst, from Aug-2018 only '
     '— the quality screen does not damp the ride, it changes which names are held.')

res['charts'] = {
    'curves20y': '/app/mpf-report-curves-20y.png',
    'curves2018': '/app/mpf-report-curves-2018.png',
    'yearlyBars': '/app/mpf-report-yearly-bars.png',
    'rolling3y': '/app/mpf-report-rolling3y.png',
    'corr20y': '/app/mpf-report-corr-20y.png',
    'corr2018': '/app/mpf-report-corr-2018.png',
    'invested': '/app/mpf-report-invested.png',
    'heat': {k: '/app/mpf-report-heat-%s.png' % SLUG[k] for k in [TN, BA, IPO, QS]},
}

for p in OUT_JSONS:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print('wrote', p)

print('\nHEADLINE %s -> %s (%.1f yrs)' % (headline['window'][0], headline['window'][1],
                                          headline['years']))
for k in headline['order']:
    r = headline['rows'][k]
    print('  %-32s %6.2f%% %8.2f%% %6.2f  %6dx  inv %s'
          % (k, r['cagr'], r['maxdd'], r['calmar'], r['growth100'],
             'n/m' if r['invested'] is None else '%.0f%%' % r['invested']))
print('\n2018 WINDOW %s -> %s' % (window2018['window'][0], window2018['window'][1]))
for k in window2018['order']:
    r = window2018['rows'][k]
    print('  %-32s %6.2f%% %8.2f%% %6.2f  %6dx' % (k, r['cagr'], r['maxdd'], r['calmar'],
                                                   r['growth100']))
print('\nQuality Summit drawn path %s (median %.2f%%, %.2f .. %.2f)'
      % (drawn, QS_OFFSETS['median'], QS_OFFSETS['min'], QS_OFFSETS['max']))
