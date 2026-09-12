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
  research/163_.../results/cash052/full_period_after_tax_cash052.csv
      HEADLINE. After-tax daily curves 2006-04-03 -> 2026-09-03 for
      'Open Alpha - Base Age', 'True North', 'IPO Base - First Base', 'NIFTYBEES (index)'.
      This is research/159's full_period_after_tax.csv with EVERY cash-holding column
      replaced by a 5.2%-idle-cash re-run of that book's own engine. NIFTYBEES holds no
      cash and is byte-identical. Override with --curves-dir.
  research/159_oa_honest_reoptimization/results/full_period_summary.json
      the measured average-invested figures that go on the table.
  research/163_.../results/cash052/all_systems_after_tax_cash052.csv
      the 2016-2026 roster curves, the same columns swapped — used ONLY to build the 2018
      section, so that section still reproduces the published roster page's construction.
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
coral Quality Summit, purple IPO Base, muted slate DASHED blend, grey index).
The books being compared are always the heaviest strokes; the blend and the index are
deliberately secondary. See LEAD / LW / line_kw below — that is one place, honoured by
every line chart on the page.
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
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
from matplotlib.colors import TwoSlopeNorm
import textwrap


def wrap(s, width=168):
    """Footnotes are long by design — they carry the caveat. At 12in and 7.5pt a single
    line runs off the right edge, so every footnote is hard-wrapped here."""
    return textwrap.fill(' '.join(s.split()), width)

ROOT = Path('/home/arun/quantifyd')
R159 = ROOT / 'research/159_oa_honest_reoptimization/results'
R160 = ROOT / 'research/160_quality_growth_near_ath/results'
R163 = ROOT / 'research/163_mpf_cash_yield_harmonisation/results'
C52 = R163 / 'cash052'
PUB = ROOT / 'frontend/public'
OUT_JSONS = [ROOT / 'static/app/mpf_report.json', PUB / 'mpf_report.json']

# ---- WHERE THE TWO CURVE FILES COME FROM.
# research/163 re-ran every book on this page at 5.2% idle cash — the ARBITRAGE-FUND rate
# after 20% short-term tax at 2025-26 cash-futures spreads. Two passes got here:
#   12-Sep-2026 18:47  True North (research/144, 6.5%) and Base Age (research/161, 5.5%)
#                      were brought onto a common 5.0%;
#   12-Sep-2026 23:xx  all five books moved 5.0% -> 5.2%, which also brought IPO Base,
#                      Open Alpha · ATH + VIX and Quality Summit onto the same rate.
# `--curves-dir` points the generator at a different folder. To rebuild the page's curves at
# the previous 5.0% basis (the prose and the ATH+VIX row still read 5.2%):
#   --curves-dir research/163_mpf_cash_yield_harmonisation/results \
#   --full-period-csv full_period_after_tax_cash05.csv \
#   --roster-csv all_systems_after_tax_cash05.csv
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument('--curves-dir', default=str(C52),
                 help='folder holding full_period_after_tax*.csv and all_systems_after_tax*.csv')
_ap.add_argument('--full-period-csv', default='full_period_after_tax_cash052.csv')
_ap.add_argument('--roster-csv', default='all_systems_after_tax_cash052.csv')
ARGS = _ap.parse_args()
CURVES = Path(ARGS.curves_dir)
FULL_CSV = CURVES / ARGS.full_period_csv
ROSTER_CSV = CURVES / ARGS.roster_csv
for _p in (FULL_CSV, ROSTER_CSV):
    if not _p.exists():
        raise SystemExit('curve file missing: %s' % _p)

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
         BLEND: '#6e8b9e', BM: '#8b949e'}
SLUG = {TN: 'truenorth', BA: 'baseage', IPO: 'ipobase', QS: 'qualitysummit'}

# ---- LINE WEIGHT IS THE ARGUMENT, not decoration (redrawn 12-Sep-2026).
# The chart previously drew the 50-50 blend thickest, in bright light blue, which buried
# True North and Base Age — the two lines the page is actually comparing. The books being
# compared now LEAD; the blend is a thin muted dashed line, secondary by construction; the
# index is the thinnest grey. Nothing is 3px. Quality Summit leads on the 2018 window,
# where it is one of the books being compared, and does not appear on the 20-year one.
# Colour-blind check: gold / green / purple / coral separate on lightness as well as hue,
# and the two lines a red-green reader could confuse — Base Age green and Quality Summit
# coral — never share a chart with the blend in a similar tone, which is why the blend was
# moved off blue-that-reads-as-teal and onto a desaturated slate, and is dashed as well.
LEAD = {TN, BA, QS}
LW = {k: (1.9 if k in LEAD else 1.25) for k in COLOR}
LW[IPO] = 1.35
LW[BLEND] = 1.2
LW[BM] = 1.0
DASH = {BLEND: (4, 2.2)}


def line_kw(k, scale=1.0):
    """One place that decides how any series is stroked, on every chart on this page."""
    kw = dict(color=COLOR[k], lw=LW[k] * scale, zorder=3 if k in LEAD else 2)
    if k in DASH:
        kw['dashes'] = DASH[k]
    if k == BM:
        kw['alpha'] = 0.85
    return kw

BG, PANEL, INK, MUT, GRID = '#0e1116', '#161b22', '#e6edf3', '#8b949e', '#30363d'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': PANEL, 'savefig.facecolor': BG,
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUT,
                     'ytick.color': MUT, 'axes.edgecolor': GRID, 'font.size': 9})

# Measured average-invested, from each engine's own reporting. Anything not measured stays
# None and prints as "not measured" — the handover asserts ~67% for Base Age but NO FILE
# carries it, and this page does not print numbers it cannot point at.
INVESTED = {TN: 43.0, IPO: 32.7, BM: 100.0, BA: 72.9, QS: 91.2}
INVESTED_SRC = {
    TN: 'research/144 phase A avg_inv 0.43 (via research/159 scripts/full_period.py)',
    IPO: 'research/153 G3 "invested 32.7% of NAV" (via the same script)',
    BM: 'fully invested by definition',
    BA: 'MEASURED 12-Sep-2026 in research/163: research/161’s engine re-run with the daily '
        'invested fraction (market value of open positions / NAV) recorded, 30 seeds. '
        'Median 72.91%, band 72.74–73.03% at the 5.2% cash rate (72.89%, 72.73–73.05% at '
        '5.0% — the rate barely touches it) — a very tight band, because the fraction is set '
        'by how often the 16 slots are full, not by which names win them. The daily series '
        'for the drawn seed is research/163_mpf_cash_yield_harmonisation/results/cash052/'
        'baseage_invested_daily_052.csv. NOTE: the handover doc asserted ~67% from memory '
        'with no source file; that figure is superseded by this measurement.',
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

full = pd.read_csv(FULL_CSV, index_col=0, parse_dates=True)
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
    source='research/163_mpf_cash_yield_harmonisation/results/cash052/'
           'full_period_after_tax_cash052.csv — research/159’s full_period_after_tax.csv with '
           'True North, Open Alpha · Base Age and IPO Base each replaced by a 5.2%-idle-cash '
           're-run of that book’s own engine. NIFTYBEES holds no cash and is byte-identical '
           'to research/159’s file.',
    basis=('After tax, 25 bps a side, 5.2% a year on idle cash for EVERY book, accrued daily '
           'and not taxed again. True North is a single after-tax NAV path from research/144’s '
           'engine; Open Alpha · Base Age is one seed of 30 from research/161’s engine, held '
           'at the seed the 5.0% page drew; IPO Base is its study’s drawn curve. Placeable '
           'entries only: decided on the close, filled at the next open.'))

# -------------------------------------------- SECOND WINDOW: where Quality Summit exists

roster = pd.read_csv(ROSTER_CSV, index_col=0, parse_dates=True)
roster = roster.rename(columns=ROSTER_RENAME)[[TN, BA, IPO, BM]]
qs_all = pd.read_csv(C52 / 'F_Bb7_equity_cash052.csv', index_col=0, parse_dates=True)

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
    source=('research/163_mpf_cash_yield_harmonisation/results/cash052/F_Bb7_equity_cash052.csv '
            'for Quality Summit — research/160’s F_Bb7 finalist cell re-run at 5.2% idle cash '
            'on its own frozen panel (the 5.0% re-run reproduced research/160’s file exactly '
            'first), the median-CAGR offset of 12 — and '
            'research/163_mpf_cash_yield_harmonisation/results/cash052/'
            'all_systems_after_tax_cash052.csv for the others — research/159’s roster curve '
            'file with True North, Base Age and IPO Base swapped for their 5.2%-idle-cash '
            're-runs, so this section still reproduces the roster page’s construction. It is '
            'a DIFFERENT run from the 20.4-year table’s file.'),
    basis=('After tax, 25 bps a side, 5.2% a year on idle cash for every book, accrued daily '
           'and not taxed again. Quality Summit is the median-CAGR rebalance offset of '
           'twelve, never an average of paths.'))
window2018['qsOffsets'] = QS_OFFSETS

# ------------------------------------------------ THE CORRECTION: after-tax evidence only

# These three tables are their OWN 30-seed re-runs of the Open Alpha engine, not slices of
# the curve files. research/163 re-ran them at 5.2% too (scripts/aftertax_all_052.py, which is
# research/159's aftertax_all.py with CASH_Y moved and its output redirected). If that re-run
# is not complete, the page falls back to research/159's 5.0% tables AND SAYS SO in the
# caption — a mixed basis is acceptable only when it is stated.
_AT52 = C52 / 'after_tax_tables_cash052.csv'
_at52 = pd.read_csv(_AT52) if _AT52.exists() else None
_at50 = pd.read_csv(R159 / 'after_tax_tables.csv')
AT_COMPLETE = _at52 is not None and len(_at52) >= len(_at50)
at_all = _at52 if AT_COMPLETE else _at50
AT_YIELD = '5.2%' if AT_COMPLETE else '5.0%'
AT_SRC = (str(_AT52).replace('/home/arun/quantifyd/', '') if AT_COMPLETE
          else 'research/159_oa_honest_reoptimization/results/after_tax_tables.csv')
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
    'Every table in this section is after tax and no pre-tax figure appears anywhere on this '
    'page. Note the window split inside the gate bake-off: the PRICE gates ran on the full '
    '2006-2026 window, while every VIX construction could only run on 2016-2026, because '
    'INDIA VIX begins in 2015. They are therefore shown as TWO tables and must never be read '
    'across. IDLE CASH IN THIS SECTION: ' + AT_YIELD + ' a year, from ' + AT_SRC + '.'
    + ('' if AT_COMPLETE else
       ' THIS SECTION IS THEREFORE ON A DIFFERENT CASH RATE FROM THE TABLES ABOVE, which '
       'credit 5.2%. These are 30-seed medians of the same engine at 5.0%; the cash rate is '
       'worth about 0.04 pp a year to a book this heavily invested, so the ORDERING of the '
       'rows — which is all this section is for — is unaffected. The 5.2% re-run is '
       'research/163 scripts/aftertax_all_052.py and can be finished at any time.'))
NOTES['invested_gap'] = INVESTED_SRC[BA]
NOTES['cash_yield'] = (
    'EVERY book on this page credits idle cash at 5.2% a year, POST-TAX, accrued daily — the '
    'cash yield is credited to the cash balance each bar and is never passed through the '
    'capital-gains settlement, which touches realised equity gains only. '
    'WHERE 5.2% COMES FROM. The rule is that idle cash sits in the best post-tax cash '
    'instrument, not in a savings account. That instrument is an ARBITRAGE FUND: it is taxed '
    'as equity — 20% short-term on units churned inside a year, 12.5% long-term beyond a '
    'year, and about 0.25% exit load if redeemed inside a month — and at 2025-26 cash-futures '
    'spreads it yields roughly 6.5% pre-tax, which lands at about 5.2% post-tax. The '
    'alternative is a liquid ETF, taxed at slab: LIQUIDCASE, LIQUIDADD and LIQUIDBETF in '
    'market_data.db realised 5.4–5.5% pre-tax in 2025 and about 5.0% annualised in 2026, '
    'which at a 30% slab is only about 3.5% post-tax. The operating rule that follows is '
    'BULK IN THE ARBITRAGE FUND, A LIQUID-ETF BUFFER for money needed at the next open, '
    'because arbitrage redemptions settle T+1. 5.2% is a FLAT ASSUMPTION, not a measured '
    'realised yield, and it is REVIEWED ON 15-DEC-2026. '
    'HOW THE PAGE GOT HERE. On 12-Sep-2026 research/163 re-ran all five books twice. First '
    'True North (research/144 had assumed 6.5%) and Open Alpha · Base Age (research/161 had '
    'assumed 5.5%) came onto a common 5.0%: True North lost 0.97 points of CAGR because it '
    'holds cash 57% of the time, Base Age 0.34. Then every book moved 5.0% → 5.2%: True '
    'North 18.56% → 18.69%, Base Age 19.93% → 19.99%, IPO Base 15.10% → 15.26% on the '
    '20.4-year window. Each re-run first REPRODUCED its own published curve at the old yield '
    'before the yield was touched. '
    'WHY THE MOVES ARE THE SIZE THEY ARE. Twenty extra basis points are earned only on the '
    'share of a book that is in cash, so the gain is about (1 − invested) × 0.2 points a '
    'year: 0.11 for True North at 43% invested, 0.14 for IPO Base at 32%, 0.05 for Base Age '
    'at 73%, 0.02 for Quality Summit at 91%. Every book’s measured move agreed with that '
    'arithmetic within its own path noise. '
    'AND CASH YIELD IS NOT A SMALL TERM for these books overall: roughly 2.9 of True North’s '
    'points and 3.5 of IPO Base’s are the sweep rather than the strategy, while NIFTYBEES is '
    'fully invested and gets none of it.')
NOTES['path_redraw'] = (
    'ONE HONEST CAVEAT ON THE 5.0% → 5.2% MOVE. Three of these books draw ONE path out of an '
    'ensemble (30 seeds for Base Age and for ATH + VIX, 12 rebalance offsets for Quality '
    'Summit). Changing the cash rate changes the cash balance, which changes INTEGER SHARE '
    'COUNTS, which changes whether a particular buy is affordable, which re-draws every later '
    'selection in that path. That re-draw is worth up to ±2 points of CAGR on a single path — '
    'an order of magnitude more than the 0.02–0.16 points the cash rate itself is worth. So '
    'the drawn path can move further, or the other way, without anything being wrong. Two '
    'things were done about it. The drawn seed is HELD at the one the 5.0% page drew, so the '
    'curve files differ only by the cash rate wherever that is possible; and the consistency '
    'test is run PAIRED, path index by path index, across the whole ensemble, against the '
    '(1 − invested) × 0.2 arithmetic. All five books passed. The one row where the re-draw is '
    'plainly visible is Open Alpha · ATH + VIX: its drawn path moved +1.20 points while the '
    'cash rate is worth +0.04 to it and its 30-seed paired median moved −0.15 ± 0.21. Read '
    'that row’s band, not its point.')
NOTES['invested_timeseries'] = (
    'The "when is each book in cash" chart is a bar of MEASURED averages rather than a strip '
    'over time. Open Alpha · Base Age now has a daily invested series — measured in '
    'research/163 and saved as cash052/baseage_invested_daily_052.csv — but True North, IPO '
    'Base and Quality Summit still only report a window average, so a time-series version is '
    'still owed and needs those three engines to emit the column.')
NOTES['two_curve_files'] = (
    'The 20.4-year section and the 2018 section are built from DIFFERENT curve files for the '
    'same systems — full_period_after_tax_cash052.csv and all_systems_after_tax_cash052.csv, '
    'both derived from research/159’s two originals. They are separate runs. The 2018 section '
    'uses the roster file deliberately, so it reproduces the published roster page’s '
    'construction.')

# ---- the Open Alpha · ATH + VIX summary row, re-measured at 5.2% idle cash.
# COMPUTED, not typed: research/163 scripts/oa_vix_cash052.py re-ran research/159's adopted
# gate cell at 5.0% (reproducing the published 19.23 / -34.15 / 0.56 exactly, on
# compare_all.py's own aligned index) and then at 5.2%. This is the one row on the page where
# the single-path re-draw described in NOTES['path_redraw'] is larger than the cash effect, so
# the 30-seed band is published beside the point and the row says why.
_AV = json.load(open(C52 / 'athvix_summary_cash052.json'))
_ATHVIX_ROW = {
    'label': 'Open Alpha · ATH + VIX (research/159)',
    'cagr': _AV['cagr_052'], 'maxdd': _AV['maxdd_052'], 'calmar': _AV['calmar_052'],
    'window': _AV['window'],
    'cagrAt5': _AV['cagr_050'], 'maxddAt5': _AV['maxdd_050'], 'calmarAt5': _AV['calmar_050'],
    'seedBand': [_AV['aligned_band_052']['cagr_min'], _AV['aligned_band_052']['cagr_med'],
                 _AV['aligned_band_052']['cagr_max']],
    'pathNote': ('Drawn path, seed %d, held at the seed the 5.0%% page drew. It moved %+.2f '
                 'points when the cash rate went from 5.0%% to 5.2%%, but the cash rate is '
                 'worth only %+.2f to a book %.0f%% invested: the rest is the share-count '
                 're-draw. Across all 30 seeds, paired, the move is %+.2f points and the '
                 '5.2%% CAGR band is %.1f–%.1f%% with a median of %.1f%%. Read the band.'
                 % (_AV['drawn_seed'], _AV['drawn_seed_redraw_pp'],
                    _AV['cagr_delta_rule_of_thumb_pp'], _AV['invested_median_pct'],
                    _AV['aligned_paired_delta_med_pp'],
                    _AV['aligned_band_052']['cagr_min'], _AV['aligned_band_052']['cagr_max'],
                    _AV['aligned_band_052']['cagr_med'])),
    'source': ('research/163_mpf_cash_yield_harmonisation/results/cash052/'
               'athvix_summary_cash052.json — research/159’s cell, its own engine, 5.2% idle '
               'cash; the 5.0% re-run reproduced all_systems_summary.json exactly first'),
}

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
    'athVix': dict(_ATHVIX_ROW),
}

# --------------------------------------------------------------------------- assemble

res = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M IST'),
    'generator': 'research/_utilities/mpf_report_build.py',
    'postTaxOnly': True,
    'standard': ('After tax — 20% short-term, 12.5% long-term above 365 days, netted '
                 'within the Indian financial year with loss carry-forward — 25 bps a '
                 'side, 5.2% a year on idle cash for every book, accrued daily and not taxed '
                 'again, and placeable entries only: decided on the close, filled at the '
                 'next open.'),
    'cashYield': {
        'rate': 5.2, 'basis': 'post-tax', 'instrument': 'arbitrage fund',
        'preTax': 6.5,
        'why': ('Arbitrage funds carry equity taxation — 20% short-term on units churned '
                'inside a year, 12.5% long-term beyond a year, ~0.25% exit load inside a '
                'month — so ~6.5% pre-tax at 2025-26 cash-futures spreads is ~5.2% post-tax.'),
        'alternative': ('A liquid ETF is taxed at slab: LIQUIDCASE / LIQUIDADD / LIQUIDBETF '
                        'realised 5.4–5.5% pre-tax in 2025 and ~5.0% annualised in 2026, '
                        'which is only ~3.5% post-tax at a 30% slab.'),
        'operatingRule': ('Bulk in the arbitrage fund, a liquid-ETF buffer for money needed '
                          'at the next open — arbitrage redemptions settle T+1.'),
        'status': 'flat assumption, not a measured realised yield',
        'reviewDue': '2026-12-15',
    },
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
        '20.4-year curves': str(FULL_CSV).replace('/home/arun/quantifyd/', ''),
        'True North at 5.2% idle cash': 'research/163_mpf_cash_yield_harmonisation/results/cash052/tn_nav_INC_cash_n8_d15_tax1_cash052.csv (research/144’s engine, its own study used 6.5%)',
        'Open Alpha · Base Age at 5.2% idle cash': 'research/163_mpf_cash_yield_harmonisation/results/cash052/ba_nav_winner_cash052.csv (research/161’s engine, its own study used 5.5%; 30 seeds, drawn seed held at the one the 5.0% page drew)',
        'IPO Base at 5.2% idle cash': 'research/163_mpf_cash_yield_harmonisation/results/cash052/ipo_honest_curve_cash052.csv (research/153’s engine via research/159’s honest-entry transform; 30 seeds, median-CAGR seed)',
        'average invested': 'research/159_oa_honest_reoptimization/results/full_period_summary.json + scripts/full_period.py; Base Age measured in research/163 (cash052/baseage_invested_daily_052.csv)',
        '2018-window curves (TN, Base Age, IPO, index)': str(ROSTER_CSV).replace('/home/arun/quantifyd/', ''),
        'Quality Summit, 12 offsets at 5.2% idle cash': 'research/163_mpf_cash_yield_harmonisation/results/cash052/F_Bb7_equity_cash052.csv (research/160’s F_Bb7 cell on its own frozen panel)',
        'after-tax entry surface / null / price gates': AT_SRC + ' (idle cash ' + AT_YIELD + ')',
        'ATH + VIX summary row': 'research/163_mpf_cash_yield_harmonisation/results/cash052/athvix_summary_cash052.json',
        'the cash-rate re-runs': 'research/163_mpf_cash_yield_harmonisation/MPF_CASH_YIELD_5P2_DAILY_RUN_STATUS.md + scripts/{tn,ba,ipo,oa_vix,qs}_cash052.py, each gated on reproducing its own 5.0% curve first',
        'the entry audit': 'research/158_oa_arming_width/OA_ARMING_WIDTH_AND_POKE_FILL_DAILY_SWEEP_STATUS.md + scripts/verify_published_trades.py',
    },
}


# --------------------------------------------------------------------------- charts

def growth_chart(frame, rows, path, title, foot1, foot2, ticks):
    """Growth of 100 on a log axis, with a drawdown panel underneath.

    DRAWN WEEKLY, MEASURED DAILY. Twenty years of daily closes on a log axis is a hairy
    line: five of them overlap into noise. The growth panel is resampled to Friday closes
    for DRAWING ONLY — every number in `rows`, every legend figure and the drawdown panel
    itself come from the daily series, untouched.

    The drawdown panel keeps its daily resolution (it must show the true depth) and has NO
    fills: five translucent filled series on top of one another was unreadable. Only the
    index keeps a very faint fill, as the reference the others are read against.
    """
    g = 100.0 * frame
    gw = g.resample('W-FRI').last().ffill()          # drawing only
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.8), sharex=True,
                                   gridspec_kw={'height_ratios': [2.0, 1.2]})
    for k in g.columns:
        ax1.plot(gw.index, gw[k],
                 label='%s  —  %.1f%%/yr, Calmar %.2f'
                       % (k, rows[k]['cagr'], rows[k]['calmar']),
                 **line_kw(k))
    ax1.set_yscale('log')
    ax1.set_yticks(ticks)
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_ylabel('growth of 100 (log scale)')
    ax1.grid(True, which='both', color=GRID, lw=0.5, alpha=0.5)
    ax1.set_title(title, color=INK, fontsize=12, loc='left', pad=12)
    # The first years are empty at the top left on a log growth chart, so the legend sits
    # there without covering a curve; it is checked against the drawn maximum below.
    leg_loc = 'upper left'
    early = gw.iloc[:int(len(gw) * 0.45)]
    if float(early.max().max()) > float(gw.max().max()) * 0.30:
        leg_loc = 'lower right'
    ax1.legend(loc=leg_loc, frameon=False, fontsize=8.5, labelspacing=0.35)

    for k in g.columns:
        dd = (g[k] / g[k].cummax() - 1) * 100
        kw = line_kw(k)
        kw['lw'] = max(0.85, kw['lw'] * 0.60)        # thin: five daily series overlap here
        ax2.plot(dd.index, dd, **kw)
        if k == BM:
            ax2.fill_between(dd.index, dd, 0, color=COLOR[k], alpha=0.10, lw=0, zorder=1)
    ax2.set_ylabel('drawdown %  (daily)')
    ax2.grid(True, color=GRID, lw=0.5, alpha=0.5)
    ax2.axhline(0, color=GRID, lw=0.8)
    # Footnotes are laid out from the BOTTOM UP so a long one cannot collide with the one
    # above it or run off the canvas — they used to do both.
    w1, w2 = wrap(foot1), wrap(foot2)
    lh = 7.5 * 1.45 / (8.8 * 72)                     # one text line as a fraction of the figure
    y2 = 0.010
    y1 = y2 + (w2.count('\n') + 1) * lh + 0.006
    fig.text(0.012, y1, w1, color=MUT, fontsize=7.5, va='bottom', linespacing=1.45)
    fig.text(0.012, y2, w2, color=MUT, fontsize=7.5, va='bottom', linespacing=1.45)
    fig.tight_layout(rect=(0, y1 + (w1.count('\n') + 1) * lh + 0.008, 1, 1))
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print('wrote', path)


growth_chart(
    full, headline['rows'], PUB / 'mpf-report-curves-20y.png',
    'THE MOMENTUM PORTFOLIO — every book being chosen between, after tax, %s to %s (20.4 years)'
    % (full.index[0].date(), full.index[-1].date()),
    # computed, not typed, so the sentence cannot go stale when a curve file changes
    ('WHAT TO SEE: the lower panel. The 50-50 blend keeps almost all of Open Alpha · Base Age’s '
     'return — %.1f%% a year against %.1f%% — while inheriting True North’s shallower falls: '
     'Base Age gives up %.0f%% of the book at its worst, the blend %.0f%%. That is the best '
     'return-per-unit-of-fall on the chart, Calmar %.2f against %.2f for True North and %.2f '
     'for Base Age.'
     % (headline['rows'][BLEND]['cagr'], headline['rows'][BA]['cagr'],
        abs(headline['rows'][BA]['maxdd']), abs(headline['rows'][BLEND]['maxdd']),
        headline['rows'][BLEND]['calmar'], headline['rows'][TN]['calmar'],
        headline['rows'][BA]['calmar'])),
    'Log scale, because over twenty years a 30x book plotted linearly flattens every other '
    'line and hides 2008 entirely. The growth panel is DRAWN WEEKLY and MEASURED DAILY — '
    'Friday closes only, to keep five twenty-year lines legible; every figure in the legend '
    'and the whole drawdown panel come from the daily series. After tax, 25 bps a side, 5.2% '
    'a year post-tax on idle cash for every book — the arbitrage-fund rate; NIFTYBEES holds '
    'no cash and gets none of it. The blend is a thin dashed line because it is this '
    'generator’s own arithmetic, not a study result.',
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
    'chart, never instead of it. Quality Summit is the median-CAGR offset of twelve. After '
    'tax, 25 bps a side, 5.2% a year post-tax on idle cash for every book. Growth panel drawn '
    'weekly, measured daily; drawdown panel daily.',
    [100, 200, 400, 800])

# ---- yearly grouped bars
yrs = headline['yearList']
bars = [TN, BA, IPO, BLEND, BM]
fig, ax = plt.subplots(figsize=(13, 5.4))
x = np.arange(len(yrs))
w = 0.16
for i, k in enumerate(bars):
    v = [headline['yoy'][k].get(y, [np.nan])[0] for y in yrs]
    # same hierarchy as the line charts: the books being compared are solid, the blend and
    # the index are outlined so they read as context rather than as contenders.
    if k == BLEND:                       # secondary: outlined, so it reads as derived
        ax.bar(x + (i - (len(bars) - 1) / 2) * w, v, w, label=k, color='none',
               edgecolor=COLOR[k], linewidth=1.1)
    elif k == BM:                        # context: filled but faded right back
        ax.bar(x + (i - (len(bars) - 1) / 2) * w, v, w, label=k, color=COLOR[k],
               alpha=0.40, linewidth=0)
    else:                                # the books being compared
        ax.bar(x + (i - (len(bars) - 1) / 2) * w, v, w, label=k, color=COLOR[k],
               linewidth=0)
ax.set_xticks(x)
ax.set_xticklabels(yrs, rotation=45, ha='right')
ax.axhline(0, color=GRID, lw=0.9)
ax.set_ylabel('return for the year, %  (after tax)')
ax.grid(True, axis='y', color=GRID, lw=0.5, alpha=0.6)
ax.legend(frameon=False, fontsize=8.5, ncol=5, loc='upper left')
ax.set_title('YEAR BY YEAR, after tax, %s to %s — who carried which year'
             % (full.index[0].date(), full.index[-1].date()),
             color=INK, fontsize=12, loc='left', pad=12)
fig.text(0.012, 0.012,
         wrap('WHAT TO SEE: the years nobody else carried. IPO Base is the only book that finishes '
         '2008 and 2011 above water at all, and it is the tall bar in 2020; True North is the '
         'one that barely moves in 2018 and 2022, when the other two are deep red; Base Age '
         'owns 2017, 2021 and 2023. That pattern is the entire argument for holding more than '
         'one of them.'),
         color=MUT, fontsize=7.5, va='bottom', linespacing=1.45)
fig.tight_layout(rect=(0, 0.105, 1, 1))
fig.savefig(PUB / 'mpf-report-yearly-bars.png', dpi=125)
plt.close(fig)
print('wrote yearly bars')

# ---- rolling 3-year CAGR
fig, ax = plt.subplots(figsize=(12, 5.0))
win = 756
for k in [TN, BA, IPO, BLEND, BM]:
    s = full[k]
    roll = (s / s.shift(win)) ** (1 / 3.0) - 1
    ax.plot(roll.index, roll * 100, label=k, **line_kw(k))
ax.axhline(0, color=GRID, lw=0.9)
ax.axhline(25, color='#d29922', lw=0.9, ls='--')
# in AXES x / DATA y: the rolling series only begins three years in, so a date taken from the
# full frame put this label off the left edge, on top of the y-axis title.
_bt = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.text(0.012, 25.9, "Arun's 25% bar", color='#d29922', fontsize=8, transform=_bt)
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
    """A small square heatmap — sized to the matrix, not to the page (rewritten 12-Sep-2026).

    It used to render a 4x4 at near-full page width with a colourbar, which made each cell
    enormous and pushed the title off the right edge. Now: about 5 inches square for a 4x4,
    no colourbar (every cell is annotated, so the bar told the reader nothing and only added
    width), the always-1.00 diagonal greyed out so the eye goes to the off-diagonal pairs
    that are the actual content, and a FIXED 0-to-1 scale on both heatmaps so the two can be
    compared. Red means "moves with the others" — bad for diversification.
    """
    wk = frame.drop(columns=[BLEND]).resample('W-FRI').last().pct_change().dropna()
    c = wk.corr()
    n = len(c)
    side = 0.72 * n + 2.15                      # 4x4 -> 5.0in, 5x5 -> 5.8in, 6x6 -> 6.5in
    fig, ax = plt.subplots(figsize=(side, side * 0.90), constrained_layout=True)
    m = np.array(c.values, dtype=float)
    np.fill_diagonal(m, np.nan)                 # the diagonal is not information
    im = ax.imshow(m, cmap='RdYlGn_r', vmin=0, vmax=1)
    im.cmap.set_bad(PANEL)
    ax.set_xticks(range(n))
    ax.set_xticklabels([s.replace('Open Alpha · ', '') for s in c.columns],
                       rotation=30, ha='right', fontsize=7.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s.replace('Open Alpha · ', '') for s in c.index], fontsize=7.5)
    ax.tick_params(length=0)
    for i in range(n):
        for j in range(n):
            v = c.values[i, j]
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', color=GRID, fontsize=8)
            else:
                ax.text(j, i, '%.2f' % v, ha='center', va='center',
                        color='#0e1116' if v > 0.45 else '#e6edf3',
                        fontsize=8, fontweight='bold')
    ax.set_title(title, color=INK, fontsize=9.5, loc='left', pad=8)
    # `foot` is deliberately NOT drawn inside the figure any more: at this size it collided
    # with the rotated tick labels. The page caption carries it (MpfReport.tsx), which is
    # also where the week count and the scale explanation now live.
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print('wrote', path)


# Titles are short so they fit the small figure; the week count and the window live in the
# page caption instead.
corr_chart(full, PUB / 'mpf-report-corr-20y.png',
           'WEEKLY-RETURN CORRELATION — 20.4 years',
           'Scale fixed 0 to 1; red = moves with the others. IPO Base is the only genuine '
           'diversifier. True North and Base Age are moderately related, which is why a '
           '50-50 of them still smooths the ride.')

corr_chart(w18, PUB / 'mpf-report-corr-2018.png',
           'WEEKLY-RETURN CORRELATION — the 2018 window',
           'Same 0-to-1 scale as the 20-year heatmap, so the two can be read across. '
           'Quality Summit is closest to Base Age and to the index — not a third source of '
           'return, a weaker sampling of a family the book already trades.')

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
         'invested index is not like for like. Open Alpha · Base Age was measured on '
         '12-Sep-2026 (research/163, 30 seeds, median 72.9%, band 72.7-73.1%) and is no '
         'longer a gap; the handover doc’s unsourced 67% is superseded.'),
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
