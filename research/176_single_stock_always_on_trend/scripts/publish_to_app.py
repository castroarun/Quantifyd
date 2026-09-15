# -*- coding: utf-8 -*-
"""Insert the research/176 study entry at the top of frontend/src/data/backtests.ts."""
import io
import re

P = '/home/arun/quantifyd/frontend/src/data/backtests.ts'
ANCHOR = 'export const BACKTEST_STUDIES: BacktestStudy[] = ['
SLUG = 'single-stock-alwayson-trend-research176'

VERDICT = (
    'NO EDGE. A per-stock always-on trend system does not beat simply holding the same stock - on any '
    'timeframe, in either window half, on any of the three names Arun asked about. Nothing is adopted and '
    'no live book is touched.\\n\\n'
    '## This is research/48 again, on five times the data\\n\\n'
    'research/48 found a SuperTrend always-on book on RECLTD 15-min that looked superb and died the moment it '
    'was run across 381 names. Its one unanswered objection was that the 15-minute window covered only 2.3 '
    'years. research/176 closes that: **146 liquid F&amp;O names, 11.5 years of intraday data resampled from '
    '5-minute bars, and 20.7 years of daily bars** - 327,840 per-name cells across SuperTrend, EMA crossover '
    'and the MST master(7,5) + child(7,2) pair Arun traded by hand on MARUTI. **The result gets worse with a '
    'longer window, not better.**\\n\\n'
    '## The pre-registered gate, and how far short it falls\\n\\n'
    'The bar was written into the STATUS doc before the first cell ran: the best cell must beat **buy-and-hold '
    'of the same stock** on at least 55% of the basket, in BOTH window halves.\\n\\n'
    '| Timeframe | best cell | full | 2006-15 / 2015-20 | recent half |\\n|---|---|---|---|---|\\n'
    '| daily 2006-2026 | EMA(9,21) long/flat | **0.432** | 0.526 | **0.329** |\\n'
    '| 60-min 2015-2026 | EMA(50,200) long/flat | **0.336** | 0.479 | **0.233** |\\n'
    '| 30-min 2015-2026 | EMA(50,200) long/flat | **0.295** | - | - |\\n\\n'
    'Not one cell on any timeframe in any window reaches 0.55, and the recent half is the worse of the two '
    'everywhere - so this is not even a "the edge used to exist" story. The decay is **monotone in turnover**: '
    'daily beats 60-min beats 30-min, which is research/56 reproduced on stocks.\\n\\n'
    '## The long/short arm is a catastrophe, and the short leg dies for the fourth time\\n\\n'
    'Arun described the system as futures long AND short on the master flip. Daily, long/short, best cell: '
    'beat-rate **0.062**, median CAGR **-3.4%** against buy-and-hold **+12.7%**. His own ST(7,3) long/short '
    'returns a median **-5.77%** across the basket. Tested alone, the short leg has **negative median '
    'expectancy per trade on every family and every timeframe** (daily -0.019, 60-min -0.004, 30-min -0.004) '
    'and beats the stock on 1.4% to 8.2% of names. That reproduces research/81, research/82 and research/83 '
    'on a new construction. **Drop the short leg.** Arun’s long/short MST posture on HDFCBANK returns -9.70% a '
    'year at a -91.2% drawdown.\\n\\n'
    '## The three names he asked about\\n\\n'
    '| name | buy and hold | best of 40 cells | cells beating B&amp;H | ST(7,3) long/flat | MST 7,5 / 7,2 long/flat |\\n'
    '|---|---|---|---|---|---|\\n'
    '| MARUTI | +15.46% (dd -63.7%) | EMA(9,21) +19.25% (dd -34.6%) | **3 / 40** | +12.54% | +13.17% |\\n'
    '| RELIANCE | +14.07% (dd -68.9%) | ST(7,5) +10.95% | **0 / 40** | +8.06% (dd -63.7%) | +10.95% |\\n'
    '| HDFCBANK | +15.67% (dd -56.0%) | EMA(50,200) +13.29% | **0 / 40** | +10.86% | +6.84% (dd -59.1%) |\\n\\n'
    'On 60-minute bars: MARUTI 11/40, RELIANCE 4/40, HDFCBANK 0/40. MARUTI’s three winning daily cells, drawn '
    'from 5,840 name-by-cell draws, is what a null looks like.\\n\\n'
    '## The SuperTrend surface is a flat failing plane, not a spike\\n\\n'
    'Daily, long/flat, share of 146 names beating buy-and-hold, by period x ATR multiplier: **0.151 to 0.342**, '
    'with no structure. Long/short, the same 24 cells: **0.007 to 0.048** - the best SuperTrend long/short '
    'configuration in the grid beats the stock on seven of 146 names. A uniform failure is stronger evidence '
    'than a single bad cell.\\n\\n'
    '## The one thing that IS real - and why it still does not help\\n\\n'
    'A **block-permutation null** was built for this study: take the posture series the rule actually produced, '
    'cut it into spells, and shuffle the spell lengths within each posture class. Time in market, trade count, '
    'cost bill and both run-length distributions survive exactly; only WHEN the long spells happen is '
    'destroyed. 200 draws per name per cell.\\n\\n'
    '**The rule beats its own matched shuffle on 69.9% of names for CAGR and 73.3% for Calmar.** The timing '
    'skill is real. What it buys is **drawdown, not return** - median -51.8% against buy-and-hold’s -75.7%, '
    'while median CAGR is 11.3% against 12.7%. And the risk edge decays: Calmar-beat-vs-buy-and-hold runs '
    '0.659 in 2006-15 and **0.521 in 2016-26** - a coin flip in the recent decade, with ST(7,3) (0.404) and the '
    'MST pair (0.452) outright losers.\\n\\n'
    '## Where the beat lives - and why you cannot pick it in advance\\n\\n'
    '| the stock’s own buy-and-hold CAGR was | n | trend beats it | median excess |\\n|---|---|---|---|\\n'
    '| below 0% | 4 | **75.0%** | **+6.47pp** |\\n| 0-10% | 46 | 56.5% | +2.21pp |\\n'
    '| 10-20% | 79 | **35.4%** | -2.02pp |\\n| above 20% | 24 | 41.7% | -2.55pp |\\n\\n'
    'The rule is a **loss-avoider**: it wins on the names that fell and loses on the names that rose. Which '
    'name will fall over the next twenty years is exactly what you do not know when you choose it.\\n\\n'
    '## Arun’s literal MST machine, lot-stacking included\\n\\n'
    'Master 7,5 sets the regime; each child 7,2 flip in the master’s direction adds a lot up to five; a master '
    'reversal closes everything and waits to re-arm. One lot = 20% of capital, so this is a **ramp, not '
    'leverage**. Equal-weight over the three names, daily, 2006-2026:\\n\\n'
    '| variant | CAGR | MaxDD | Calmar |\\n|---|---|---|---|\\n'
    '| stacked long/flat | **+6.86%** | **-14.8%** | 0.464 |\\n| stacked long/short | +4.25% | -15.0% | 0.284 |\\n'
    '| single-unit long/flat | +11.46% | -27.9% | 0.410 |\\n| single-unit long/short | **-2.00%** | -71.6% | -0.028 |\\n'
    '| **buy and hold** | **+17.71%** | -54.7% | 0.324 |\\n\\n'
    'The stacking is the best drawdown tool in the whole study - it cuts the book’s worst loss from -54.7% to '
    '-14.8% and 2008 from -45.1% to -9.4%. It is also the lowest-returning positive arm: **6.86% a year, below '
    'NIFTY 50’s 8.79% and barely above the 5.2% post-tax cash the project already earns on idle balances.** A '
    'ramp that is only fully invested after four confirmations is a de-levering device wearing a trading '
    'system’s clothes.\\n\\n'
    '## The deciding test: plain cash beats it as a sleeve\\n\\n'
    'research/134 established that a directional complement is judged on the blend, not standalone. Against the '
    'live short-vol book (C1 stock winged strangles + the 45-DTE NIFTY straddle, 75 common months 2019-05 to '
    '2026-07, standalone **+21.29% CAGR, -10.39% DD, Calmar 2.05**), change in blend Calmar:\\n\\n'
    '| sleeve | corr | 10% | 20% | 30% | 40% |\\n|---|---|---|---|---|---|\\n'
    '| **plain CASH at 5.2%** | - | +0.09 | +0.21 | +0.36 | **+0.54** |\\n'
    '| **RELIANCE buy and hold** | -0.17 | +0.27 | +0.35 | **+0.41** | +0.10 |\\n'
    '| ARUN3 / EMA(10,30) | -0.18 | +0.13 | +0.14 | +0.11 | +0.07 |\\n'
    '| ARUN3 / EMA(9,21) | -0.15 | +0.11 | +0.10 | +0.05 | -0.02 |\\n'
    '| ARUN3 / MST 7,5 / 7,2 | -0.15 | -0.07 | -0.22 | -0.44 | -0.66 |\\n'
    '| HDFCBANK / MST 7,5 / 7,2 | -0.08 | -0.27 | -0.55 | -0.87 | -1.21 |\\n\\n'
    '**Every directional sleeve is beaten by doing nothing with the money.** And where a directional sleeve '
    'does beat cash at low weight it is **buy-and-hold of RELIANCE**, not any trend rule applied to it. The '
    'pre-registered bar required "+0.10 Calmar or -2pp drawdown AT EQUAL OR BETTER RETURN" - every blend in the '
    'table cuts CAGR (21.29% to 19.65% at only 10% weight), so nothing clears it on either leg. This is '
    'research/134’s conclusion, reproduced independently at single-stock level: **the diversifier is plain long '
    'equity, and trend timing on top of it hurts.**\\n\\n'
    '## The options expression was never opened, by pre-registration\\n\\n'
    'The plan committed to the "options selling hedge on the child ST" leg only if the futures signal survived. '
    'It survived nothing. Two further reasons it would not have rescued anything: **research/56** already ran '
    'exactly this construction on the index and the always-on credit book returned -Rs 17k to -Rs 62k per six '
    'weeks against a ~10 bps per-flip break-even; and **research/150** killed five option structures built on '
    'high-win-rate signals, because an overlay changes the payoff SHAPE and does not create expectancy. It '
    'cannot rescue a posture whose underlying expectancy is already below buy-and-hold.\\n\\n'
    '## What it changes\\n\\n'
    'Operationally, nothing. What it adds to the shelf: **the single-stock always-on trend line is CLOSED** - '
    'cite this study with research/48 rather than re-running it; a **reusable block-permutation null** that is '
    'the cleanest separator this project has of "the rule works" from "being long works"; and a fourth '
    'confirmation that **stops and ramps buy drawdown, never return** (research/172, research/174). If Arun '
    'wants a non-index directional book, the answer stands from research/134 and is now confirmed on single '
    'stocks: own the equity, do not time it - and that equity is already owned through True North, Open Alpha '
    'and IPO Base.'
)

ENTRY = """  {
    slug: '__SLUG__',
    title: 'Always-on directional trend on one or a few stocks - SuperTrend, EMA crossover and the MST master+child pair, on 146 names across daily, 60-min and 30-min bars (research/176)',
    verdict:
      '__VERDICT__',
    status: 'COMPLETE',
    date: '2026-09-15',
    cardBlurb:
      'Arun’s hand-traded MARUTI system - master SuperTrend(7,5), child SuperTrend(7,2), futures long/short - put on a 146-name basket over 20.7 years of daily and 11.5 years of intraday data. It does not beat holding the stock, anywhere. 0 of 40 daily cells beat buy-and-hold on RELIANCE, 0 of 40 on HDFCBANK.',
    cardStats: [
      { label: 'Best beat-rate vs holding the stock', value: '0.432' },
      { label: 'Pre-registered gate', value: '0.55' },
      { label: 'Name-cells tested', value: '327,840' },
    ],

    system: {
      intro:
        'A vectorised trend engine written fresh for this study. NOTHING is reused from services/maruthi_*.py - that live algo was disabled on 2026-03-25 with nine critical bugs, and one of them (SuperTrend computed on spot while trading the future) is a modelling question this study answers explicitly rather than inherits.',
      rows: [
        { k: 'SuperTrend', v: 'Band-locked to the TradingView convention - the upper band only moves down and the lower band only moves up while unbroken. Wilder ATR. Period {7,10,14,21} x multiplier {1.5,2,2.5,3,4,5} = 24 cells.' },
        { k: 'EMA crossover', v: '(5,20) (9,21) (10,30) (20,50) (21,55) (50,100) (50,200) = 7 cells.' },
        { k: 'MST master + child', v: 'Master SuperTrend(7,m) sets the regime, child SuperTrend(7,c) arms the entry inside it; flat between a master flip and the first child confirmation. m in {4,5,6} x c in {1.5,2,2.5} = 9 cells. Arun’s own cell is master 7,5 / child 7,2.' },
        { k: 'Direction policies', v: 'long/flat, long/short, and short-only - the last so the short leg can be judged on its own rather than hidden inside a long/short average.' },
        { k: 'Fill', v: 'Honest = signal on the bar close, filled at the NEXT bar’s open. A same-bar-close fill is carried as a labelled reference arm and is worth +0.35pp of CAGR and +0.034 of beat-rate on daily - never enough to move a verdict.' },
        { k: 'Instrument', v: 'The underlying cash series is used as a proxy for the front-month future, with a 5 bps monthly roll charge on top of the round trip. This FLATTERS the always-on book by understating roll slippage, and the verdict is negative anyway.' },
        { k: 'Bars', v: 'Daily straight from the database; 60-min and 30-min RESAMPLED from 5-minute bars anchored at 09:15, which is what makes an 11.5-year intraday window possible where research/48 had 2.3.' },
        { k: 'Nulls', v: 'Buy-and-hold of the same name (primary); a block-permutation null preserving time-in-market, trade count and both run-length distributions, 200 draws per name per cell; and a plain cash sleeve at the blend stage.' },
      ],
    },

    conditions: {
      intro: 'Futures P&L is business income in India, not capital gains, so every figure here is PRE-TAX and labelled as such. The after-tax comparison is made only at the blend stage.',
      rows: [
        { k: 'Universe', v: 'The 322 symbols carrying 5-minute bars from Feb-2015 to 2026 - the liquid F&O panel built in research/81 - filtered to a 20-day median traded value of at least Rs 10 crore (153 names), less 7 names carrying an unexplained overnight price step (146 names in every headline figure).' },
        { k: 'Split-adjustment defect', v: 'market_data.db is NOT retroactively split-adjusted. A scan of the panel flagged 14 of 321 names with an overnight gap outside 0.65x to 1.55x; the 7 inside the liquid basket are excluded. The flagged names and dates are in results/universe.csv.' },
        { k: 'Windows', v: 'Daily 2006-01-01 to 2026-09-15 (20.7 years), split 2006-2015 / 2016-2026. Intraday 2015-02 to 2026, split 2015-2020 / 2021-2026. Both halves must pass.' },
        { k: 'Costs', v: '10 / 20 / 40 bps round trip in every table, plus a 60 bps book-level rung. Switches per year carried in every row - the 30-minute cells run 16 to 128 a year.' },
        { k: 'Idle cash', v: '5.2% post-tax, the project standard, credited whenever the book is flat.' },
        { k: 'Pre-registered bar', v: 'Locked in the STATUS doc before the first cell: G1a beat-rate at least 0.55 in BOTH halves; G1b neighbours at least 80% of the best cell; G1c beats the matched null on at least 60% of names; G2 still clears at 40 bps; G3 the short leg positive alone; G4 +0.10 blend Calmar or -2pp drawdown at equal-or-better return AND beats the cash-null; G5 options only if G1-G4 pass.' },
        { k: 'What was NOT tested', v: 'The options expression (gated on G1-G4, which failed - and research/56 and research/150 have already killed that family four times); 15-minute and 5-minute bars (30/60-min are monotonically worse than daily, so faster is a dead direction); pyramiding beyond 5 lots; per-name parameter selection, which IS the research/48 overfit this study exists to avoid.' },
      ],
    },

    comparisons: [
      {
        title: 'The pre-registered gate, on every timeframe',
        caption: 'Share of the 146 names whose net CAGR at 20 bps beats that same name’s buy-and-hold. Best long/flat cell shown. The gate was 0.55 in BOTH halves.',
        columns: ['Timeframe', 'Best cell', 'Full window', 'First half', 'Recent half', 'At 40 bps', 'Median CAGR', 'Median buy-and-hold'],
        rows: [
          ['Daily 2006-2026', 'EMA(9,21)', '0.432', '0.526', '0.329', '0.363', '+11.49%', '+12.72%'],
          ['60-min 2015-2026', 'EMA(50,200)', '0.336', '0.479', '0.233', '0.267', '+7.99%', '+11.43%'],
          ['30-min 2015-2026', 'EMA(50,200)', '0.295', '-', '-', '0.212', '+7.40%', '+11.43%'],
        ],
        heatmap: true,
      },
      {
        title: 'Arun’s own configurations across the basket - daily, next-open fills, 20 bps',
        caption: 'vs B&H is the median difference in CAGR against holding the same stock. Calmar-beat is the share of names where the rule’s Calmar exceeds the stock’s.',
        columns: ['Cell', 'Policy', 'beat-rate 20bps', 'beat-rate 40bps', 'Median CAGR', 'vs buy-and-hold', 'Calmar-beat', 'Switches/yr'],
        rows: [
          ['SuperTrend(7,3)', 'long/flat', '0.322', '0.267', '+10.54%', '-2.91pp', '0.534', '6'],
          ['SuperTrend(7,3)', 'long/short', '0.034', '0.034', '-5.77%', '-19.20pp', '0.034', '6'],
          ['MST 7,5 / 7,2', 'long/flat', '0.322', '0.301', '+10.10%', '-2.28pp', '0.555', '3'],
          ['MST 7,5 / 7,2', 'long/short', '0.027', '0.027', '-3.59%', '-17.82pp', '0.027', '3'],
          ['EMA(20,50)', 'long/flat', '0.377', '0.356', '+10.77%', '-2.67pp', '0.541', '5'],
          ['EMA(50,200)', 'long/flat', '0.329', '0.315', '+10.29%', '-2.40pp', '0.562', '1'],
        ],
      },
      {
        title: 'The short leg, judged on its own',
        caption: 'Short-only postures, next-open fills, 20 bps. Median expectancy per trade is negative on every family and every timeframe - the fourth independent kill of equity shorts in this project after research/81, 82 and 83.',
        columns: ['Timeframe', 'Best short cell', 'beat-rate', 'Median CAGR', 'Median buy-and-hold', 'Median expectancy/trade'],
        rows: [
          ['Daily', 'EMA(10,30)', '0.014', '-8.30%', '+12.72%', '-0.0189'],
          ['60-min', 'EMA(9,21)', '0.082', '-11.23%', '+11.43%', '-0.0036'],
          ['30-min', 'EMA(20,50)', '0.082', '-10.81%', '+11.43%', '-0.0041'],
        ],
      },
      {
        title: 'The block-permutation null - real timing skill, worthless return',
        caption: 'Daily, long/flat, 146 names, 200 draws each. The null keeps time-in-market, trade count and both run-length distributions and destroys only WHEN the long spells happen. 0.50 would be no skill.',
        columns: ['Cell', 'Time in market', 'Rule CAGR', 'Null CAGR', 'Buy-and-hold CAGR', 'Beats own null (CAGR)', 'Rule Calmar', 'Null Calmar', 'B&H Calmar', 'Beats own null (Calmar)'],
        rows: [
          ['EMA(9,21)', '54.9%', '11.29%', '8.21%', '12.72%', '69.9%', '0.221', '0.132', '0.174', '73.3%'],
          ['EMA(10,30)', '55.7%', '10.95%', '8.51%', '12.72%', '63.7%', '0.204', '0.137', '0.174', '67.8%'],
          ['MST 7,5 / 7,2', '59.4%', '9.99%', '9.45%', '12.72%', '59.6%', '0.187', '0.146', '0.174', '64.4%'],
          ['SuperTrend(7,3)', '54.3%', '10.37%', '8.85%', '12.72%', '55.5%', '0.191', '0.143', '0.174', '60.3%'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Blend against the live short-vol book - change in blend Calmar',
        caption: 'C1 stock winged strangles + the 45-DTE NIFTY straddle, equal risk, 75 common months 2019-05 to 2026-07. The book alone: +21.29% CAGR, -10.39% drawdown, Calmar 2.05. Every blend in this table LOWERS CAGR, so none clears the "at equal or better return" clause.',
        columns: ['Sleeve', 'Correlation', '10%', '20%', '30%', '40%'],
        rows: [
          ['Plain CASH at 5.2% - the null', '-', '+0.09', '+0.21', '+0.36', '+0.54'],
          ['RELIANCE buy and hold', '-0.17', '+0.27', '+0.35', '+0.41', '+0.10'],
          ['3-name book / EMA(10,30) long/flat', '-0.18', '+0.13', '+0.14', '+0.11', '+0.07'],
          ['3-name book / EMA(9,21) long/flat', '-0.15', '+0.11', '+0.10', '+0.05', '-0.02'],
          ['HDFCBANK / SuperTrend(7,3) long/flat', '-0.20', '+0.14', '+0.34', '+0.14', '-0.34'],
          ['3-name book / MST 7,5 / 7,2', '-0.15', '-0.07', '-0.22', '-0.44', '-0.66'],
          ['HDFCBANK / MST 7,5 / 7,2', '-0.08', '-0.27', '-0.55', '-0.87', '-1.21'],
        ],
        highlightRows: [0, 1],
      },
    ],

    results: {
      metrics: [
        { label: 'Best beat-rate vs holding the stock', value: '0.432', hint: 'daily EMA(9,21) long/flat; gate was 0.55', tone: 'neg' },
        { label: 'Recent half', value: '0.329', hint: 'the same cell, 2016-2026 - worse, not better', tone: 'neg' },
        { label: 'Best book found', value: '13.88% / -28.0%', hint: '3-name equal-weight, daily EMA(9,21) long/flat, Calmar 0.495' },
        { label: 'Holding the same three names', value: '17.71% / -54.7%', hint: 'Calmar 0.324 - more return, worse path' },
        { label: 'The deployed TN+OA pair', value: 'Calmar 1.68', hint: 'the best thing found here is less than a third as good' },
        { label: 'Cells tested', value: '327,840', hint: '40 signals x 3 policies x 2 fills x 3 windows x 146 names x 3 timeframes' },
      ],
      tables: [
        {
          title: 'Year by year - the 3-name book against every variant and the benchmark',
          caption: 'MARUTI + RELIANCE + HDFCBANK equal-weight, daily bars, next-open fills, 20 bps round trip, idle cash 5.2%, pre-tax. Each cell is the annual return; the figure beneath is that year’s maximum drawdown.',
          columns: ['Year', 'EMA(9,21) long/flat', 'ST(7,3) long/flat', 'ST(7,3) long/short', 'MST stacked long/flat', 'Buy and hold', 'NIFTY 50'],
          rows: [
            ['2006', '+31.7% (-16.6%)', '+44.0% (-16.7%)', '+18.3% (-22.1%)', '+7.3% (-7.3%)', '+68.9% (-25.3%)', '-'],
            ['2007', '+51.3% (-11.3%)', '+43.4% (-11.8%)', '+17.4% (-15.8%)', '+28.7% (-5.3%)', '+63.1% (-15.1%)', '-'],
            ['2008', '-18.6% (-25.5%)', '-31.8% (-36.3%)', '-33.1% (-40.8%)', '-9.4% (-14.8%)', '-45.1% (-54.0%)', '-'],
            ['2009', '+68.4% (-14.0%)', '+65.1% (-14.2%)', '+18.8% (-17.9%)', '+9.6% (-2.6%)', '+117.6% (-14.1%)', '-'],
            ['2010', '+9.7% (-8.2%)', '+0.6% (-11.4%)', '-14.9% (-20.7%)', '+1.6% (-3.1%)', '+10.8% (-11.7%)', '-'],
            ['2011', '-15.4% (-16.0%)', '-13.5% (-14.0%)', '-8.8% (-24.1%)', '+3.4% (-1.6%)', '-26.0% (-26.4%)', '-24.9% (-26.2%)'],
            ['2012', '+28.9% (-10.0%)', '+23.9% (-6.8%)', '-1.3% (-13.4%)', '+4.2% (-1.4%)', '+48.4% (-16.0%)', '+27.7% (-13.8%)'],
            ['2013', '+0.8% (-14.7%)', '-2.9% (-17.5%)', '-20.0% (-26.8%)', '+4.5% (-2.3%)', '+8.9% (-19.1%)', '+6.8% (-14.6%)'],
            ['2014', '+32.7% (-4.9%)', '+35.9% (-4.1%)', '+28.2% (-5.1%)', '+20.5% (-2.6%)', '+39.8% (-9.5%)', '+31.4% (-6.5%)'],
            ['2015', '+9.5% (-12.5%)', '+1.8% (-12.3%)', '-19.8% (-26.5%)', '+12.6% (-6.4%)', '+23.0% (-11.6%)', '-4.1% (-16.0%)'],
            ['2016', '+24.6% (-4.2%)', '+20.5% (-5.3%)', '+22.2% (-10.8%)', '+3.8% (-1.5%)', '+12.6% (-15.6%)', '+3.0% (-12.5%)'],
            ['2017', '+48.2% (-5.3%)', '+43.9% (-5.5%)', '+18.0% (-16.6%)', '+30.8% (-4.2%)', '+71.1% (-5.3%)', '+28.6% (-4.1%)'],
            ['2018', '+1.5% (-9.4%)', '+2.4% (-7.5%)', '-5.5% (-16.5%)', '+0.0% (-8.1%)', '+3.2% (-19.0%)', '+3.2% (-14.6%)'],
            ['2019', '+8.4% (-11.2%)', '+8.2% (-11.5%)', '-5.6% (-15.4%)', '+5.8% (-2.6%)', '+18.8% (-16.4%)', '+12.0% (-11.4%)'],
            ['2020', '+32.6% (-10.3%)', '+39.0% (-10.3%)', '+43.9% (-23.1%)', '+7.6% (-4.4%)', '+18.8% (-41.0%)', '+14.9% (-38.4%)'],
            ['2021', '+8.2% (-10.6%)', '+0.9% (-11.9%)', '-11.8% (-20.9%)', '+3.4% (-4.4%)', '+7.7% (-12.5%)', '+24.1% (-10.1%)'],
            ['2022', '-2.0% (-13.3%)', '-7.6% (-13.7%)', '-28.9% (-31.2%)', '+0.6% (-4.7%)', '+11.7% (-16.1%)', '+4.3% (-16.5%)'],
            ['2023', '+11.6% (-5.3%)', '+11.1% (-3.7%)', '+3.5% (-7.9%)', '+3.5% (-1.8%)', '+12.1% (-7.9%)', '+20.0% (-7.1%)'],
            ['2024', '-2.4% (-10.1%)', '-0.1% (-10.9%)', '-8.7% (-21.8%)', '+2.2% (-4.5%)', '+2.1% (-13.4%)', '+8.8% (-10.9%)'],
            ['2025', '+10.5% (-9.8%)', '+14.8% (-7.1%)', '-3.8% (-10.4%)', '+7.2% (-0.9%)', '+31.9% (-8.6%)', '+10.5% (-8.7%)'],
            ['2026 YTD', '-10.5% (-11.5%)', '-4.6% (-6.7%)', '+9.7% (-10.2%)', '+1.5% (-1.7%)', '-23.6% (-25.7%)', '-11.5% (-15.2%)'],
            ['FULL 2006-2026', '+13.88% / -28.0% / Cal 0.495', '+11.86% / -40.5% / Cal 0.293', '-1.03% / -57.9% / Cal -0.018', '+6.86% / -14.8% / Cal 0.464', '+17.71% / -54.7% / Cal 0.324', '+8.79% / -38.4% / Cal 0.229'],
          ],
          highlightRows: [21],
        },
        {
          title: 'Where the beat lives - and why it is not selectable in advance',
          caption: 'EMA(9,21) long/flat, daily, grouped by what the STOCK itself did over the same twenty years.',
          columns: ['The stock’s own buy-and-hold CAGR', 'Names', 'Trend beats it', 'Median excess'],
          rows: [
            ['Below 0%', '4', '75.0%', '+6.47pp'],
            ['0 to 10%', '46', '56.5%', '+2.21pp'],
            ['10 to 20%', '79', '35.4%', '-2.02pp'],
            ['Above 20%', '24', '41.7%', '-2.55pp'],
          ],
        },
        {
          title: 'Cost ladder on the best book found',
          caption: '3-name equal-weight, daily EMA(9,21) long/flat. Costs are NOT what kills it on daily - holding the stock is. Costs are what kill the intraday arms, whose beat-rates roughly halve between 20 and 40 bps.',
          columns: ['Round-trip cost', 'CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['10 bps', '+14.44%', '-27.7%', '0.521'],
            ['20 bps', '+13.88%', '-28.0%', '0.495'],
            ['40 bps', '+12.75%', '-28.8%', '0.443'],
            ['60 bps', '+11.64%', '-29.6%', '0.394'],
            ['Buy and hold (reference)', '+17.71%', '-54.7%', '0.324'],
          ],
        },
        {
          title: 'The Calmar edge is not stable either',
          caption: 'Share of 146 names where the rule’s Calmar beats the stock’s, daily, long/flat, by window.',
          columns: ['Cell', 'Full 2006-2026', '2006-2015', '2016-2026'],
          rows: [
            ['EMA(9,21)', '0.637', '0.659', '0.521'],
            ['EMA(10,30)', '0.596', '0.600', '0.575'],
            ['SuperTrend(7,3)', '0.534', '0.600', '0.404'],
            ['MST 7,5 / 7,2', '0.555', '0.563', '0.452'],
          ],
        },
      ],
      charts: [
        {
          src: '/app/research176-alwayson-trend.png',
          caption: 'Top: growth of the 3-name equal-weight book under each variant, log scale, against buy-and-hold and NIFTY 50. Middle: the drawdown panel. Bottom left: the SuperTrend plateau map - share of 146 names beating buy-and-hold, a flat failing plane rather than a spike. Bottom right: the best cell on each timeframe against the pre-registered 0.55 gate.',
        },
      ],
    },

    winners: [
      {
        config: 'There is no winner. The best construction found loses to holding the same three stocks.',
        summary:
          'The best cell anywhere in 327,840 draws - a 3-name equal-weight book running EMA(9,21) long/flat on daily bars - returns 13.88% at a -28.0% drawdown. Buy-and-hold of the same three names returns 17.71%. The trend rule buys a smaller drawdown with 3.8 points of annual return, and the deployed True North + Open Alpha pair does that trade far better at Calmar 1.68.',
        metrics: [
          { k: 'Best book', v: '3-name EMA(9,21) long/flat: 13.88% CAGR, -28.0% DD, Calmar 0.495' },
          { k: 'Its null', v: 'Buy-and-hold the same names: 17.71% CAGR, -54.7% DD, Calmar 0.324' },
          { k: 'Pre-registered gate', v: 'beat-rate 0.55 in both halves - achieved 0.432 full, 0.329 recent' },
          { k: 'As a sleeve', v: 'Beaten by a plain cash sleeve at every weight, and by RELIANCE buy-and-hold' },
        ],
        rejected: [
          'Long/short on any rule or timeframe - median CAGR -3.4% daily and -12.0% on 60-min against buy-and-hold +12.7%',
          'Short-only - negative median expectancy per trade on every family and timeframe',
          'SuperTrend(7,3) on RELIANCE, the exact idea in the ask - +8.06% against the stock’s +14.07%, at a -63.7% drawdown, and 0 of 40 cells on that name beat holding it',
          'The MST lot-stack as a return system - 6.86% a year, below NIFTY 50 and barely above the 5.2% cash standard',
          'Faster timeframes - 60-min and 30-min are monotonically worse than daily, so 15-min and 5-min were not run',
          'The options expression - pre-registered as gated on the futures signal, which failed; research/56 and research/150 have killed that family four times already',
        ],
      },
    ],

    caveats: [
      'P&L is computed on the underlying CASH series as a proxy for the front-month future, with a 5 bps monthly roll charge. This is Maruthi live-bug #9, deliberately inherited and disclosed. A real futures book pays more roll friction than modelled, so the proxy FLATTERS the always-on system - and the verdict is negative anyway.',
      'market_data.db is not retroactively split-adjusted. 14 of 321 panel names carry an unexplained overnight step; the 7 inside the liquid basket are excluded from every headline figure, and the flagged names and dates are published in results/universe.csv.',
      'Survivorship: the universe is the 5-minute panel as it exists today, so names delisted before 2015 are absent. This biases the study IN FAVOUR of buy-and-hold’s opponent being flattered - and buy-and-hold still won, so the bias cannot rescue the finding.',
      'Intraday bars are resampled from 5-minute data with 30/60-minute bins anchored at 09:15. Spread and market impact inside a bar are modelled only through the cost ladder, not tick by tick.',
      'The blend uses research/134’s 75-month combined short-vol series (2019-05 to 2026-07), which is much shorter than the daily equity window and covers one broad regime.',
      'Futures P&L in India is business income, not capital gains, so all standalone figures are pre-tax. The equity books quoted for comparison (True North, Open Alpha) are after-tax, which makes the gap between them slightly WIDER than shown, not narrower.',
      'Multiple testing: 327,840 name-cells were evaluated. The single apparent winner - EMA(9,21) on MARUTI - is one of 5,840 daily name-by-cell draws, and the SuperTrend plateau map is published in full precisely so the discovery can be discounted.',
    ],

    githubLinks: [
      { label: 'research/176 scripts', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/176_single_stock_always_on_trend/scripts' },
      { label: 'RESULTS.md', href: 'https://github.com/castroarun/Quantifyd/blob/main/research/176_single_stock_always_on_trend/results/RESULTS.md' },
    ],
    projectPaths: [
      'research/176_single_stock_always_on_trend/SINGLE_STOCK_ALWAYS_ON_TREND_MULTITF_SWEEP_STATUS.md',
      'research/176_single_stock_always_on_trend/scripts/engine.py',
      'research/176_single_stock_always_on_trend/scripts/stage0_universe.py',
      'research/176_single_stock_always_on_trend/scripts/stage1_daily_basket.py',
      'research/176_single_stock_always_on_trend/scripts/stage2_intraday_basket.py',
      'research/176_single_stock_always_on_trend/scripts/stage3_nulls.py',
      'research/176_single_stock_always_on_trend/scripts/stage4_book_and_blend.py',
      'research/176_single_stock_always_on_trend/scripts/stage5_mst_stacking.py',
      'research/176_single_stock_always_on_trend/scripts/stage6_yoy_and_chart.py',
      'research/176_single_stock_always_on_trend/results/RESULTS.md',
      'research/176_single_stock_always_on_trend/results/yoy_table.md',
      'research/176_single_stock_always_on_trend/results/universe.csv',
    ],
  },
"""
ENTRY = ENTRY.replace("__SLUG__", SLUG).replace("__VERDICT__", VERDICT)


def main():
    s = io.open(P, encoding='utf-8').read()
    if SLUG in s:
        print('already present, nothing to do')
        return
    i = s.index(ANCHOR) + len(ANCHOR)
    s = s[:i] + '\n' + ENTRY + s[i:].lstrip('\n')
    io.open(P, 'w', encoding='utf-8').write(s)
    print('inserted research/176 entry')


if __name__ == '__main__':
    main()
