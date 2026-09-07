"""r/156 publish - append the BacktestStudy entry to frontend/src/data/backtests.ts.

Idempotent: if the slug is already present the file is left untouched.
Also copies the factsheet PNG into frontend/public/.
"""
import shutil
from pathlib import Path

ROOT = Path("/home/arun/quantifyd")
TS = ROOT / "frontend/src/data/backtests.ts"
RES = ROOT / "research/156_sector_rotation/results"
PUB = ROOT / "frontend/public"
SLUG = "sector-trend-rotation-research156"

ENTRY = r"""  {
    slug: 'sector-trend-rotation-research156',
    title: 'Can sector trend build a better book? - rotation across sectors, and sector-gated stock picking',
    verdict:
      'NO EDGE for sector rotation, and NO ADDED VALUE for the sector filter. Two separate questions were asked and both failed, for cleanly separable reasons. BRANCH A - ALLOCATING ACROSS SECTORS. Of 1,440 rotation configurations on the nine real NSE sector indices (8 trend signals x 5 sector counts x 4 weightings x 3 rebalance clocks x 3 gates, each on 4 rebalance-day offsets = 5,760 runs), ZERO clear the pre-registered bar of 20% CAGR after tax at Calmar 1.0. The best-CAGR cell returns 16.3% CAGR at -36.9% drawdown (Calmar 0.44); the best-Calmar cell returns 13.5% at -21.5% (0.63), and its drawdown protection comes from a NIFTY500-above-200-SMA cash gate rather than from any sector choice. Over the identical window, equal-weighting all nine sectors returns 14.0% (Calmar 0.32), NIFTY 500 buy-and-hold 13.2% (0.35), and the plain Midcap 150 index 18.0% (0.41). Doing nothing but holding the Midcap 150 beats every rotation cell we built, on return AND on risk-adjusted return. The signal is not absent, it is worthless: against a 500-draw random-sector null rotating on the same clock and paying the same costs and taxes, the best momentum configuration sits at the 94th to 100th percentile of the random distribution at every sector count. Momentum genuinely ranks sectors better than chance - it just ranks them worth less than the diversification the ranking destroys. That is the research/63 lesson, diversification beat selection, repeating on a new asset class. BRANCH B - SECTOR AS A UNIVERSE FILTER. This is where the big numbers were. Holding the momentum leaders inside the top-ranked industries returns 32.5% CAGR after tax at -38.3% drawdown (Calmar 0.85) over 2016-2026, which clears the 20% bar with room. The pre-registered decomposition kills it. Running the IDENTICAL stock rule over the full universe with NO sector filter at all returns 32.9% at -39.3% (Calmar 0.84) - statistically the same book - and on paired offsets the sector-filtered version wins on CAGR in only 5 of 16 paired comparisons. All the return comes from the stock momentum ranking, which beats random stocks inside the same sectors on 91-100% of 100 draws. None of it comes from the sector layer. Restricting to random sectors costs about 8pp of CAGR; restricting to momentum-ranked sectors recovers almost exactly that 8pp and stops. The sector layer is a round trip to nowhere. A DATA FINDING THAT MATTERS BEYOND THIS STUDY. The wide 20-industry cross-section built from stock constituents LOOKS strongly predictive (momentum information coefficient t = 3.6 to 5.6, top-minus-bottom spread 8-14% a year). It is an artefact. Validated against the real sector indices, every synthetic basket out-drifts its real index by +4 to +14 percentage points of CAGR per year - for scale, research/154 accepted its gold reconstruction at +0.5pp. On the same eight sectors over the same window the real indices score t = 0.9 to 1.4 while the synthetic baskets score 2.6 to 2.9, and with industry labels RANDOMLY SHUFFLED the synthetic panel still produces t around 1.0 to 1.9 with a 95th percentile of 3.0 to 3.5. Most of the apparent sector momentum is stock-level momentum among survivors, aggregated into equal-weight baskets and mislabelled as a sector effect. COMPLEMENT VALUE: NONE. Against the deployed True North 40 / Open Alpha 40 / IPO 20 book on each candidate own window, every candidate at every weight from 5% to 30% loses CAGR and is beaten by a plain cash sleeve at the same weight on Calmar. Correlations to the incumbents run 0.41 to 0.54 daily - above the pre-registered 0.40 ceiling for both legs, in every case, and about as correlated to the incumbents as the incumbents are to each other. Sector-flavoured Indian equity is Indian equity. This CONFIRMS AND EXTENDS research/147 rather than contradicting it: that study killed sector rotation on a single cell, and reproduced here after tax the same cell returns 8.9% CAGR at -56.0% drawdown, Calmar 0.16, the worst book in the study. The sweep says the kill was not an artefact of one cell - the whole design space is dead.',
    status: 'COMPLETE',
    date: '2026-09-07',
    cardBlurb:
      'Arun asked whether the sector indices can be read for trend - ride the leaders in proportions, or drill into their leader stocks for a curated book. Twelve signal families, 1,440 rotation configurations, a 500-draw random-sector null, and a three-way decomposition of the stock branch. Nothing clears the bar: the best rotation cell loses to simply holding the Midcap 150, and the sector-gated stock book is indistinguishable from the same stock rule with no sector filter at all.',
    cardStats: [
      { label: 'Verdict', value: 'NO EDGE (rotation) - NO ADDED VALUE (sector filter)' },
      { label: 'Configs clearing the bar', value: '0 of 1,440' },
      { label: 'Sector-gated vs no filter at all', value: '32.5% / -38.3% / 0.85  vs  32.9% / -39.3% / 0.84' },
    ],

    systemRules: {
      intro:
        'Arun asked for no bias to existing systems, and that was honoured as an instruction about inherited DESIGN. Nothing here comes from Open Alpha (all-time-high close trigger, relative strength 70, 16 slots, -8% stop, 15-SMA trail) or True North (Nifty-200 top-8, 100-SMA weekly gate, 15-day Donchian). Twelve trend and strength families were written from first principles and the data was allowed to choose. Measurement discipline was kept - costs, taxes, offset ensembles, paired comparison, null controls and plateau checks are how a fresh look avoids fooling itself, not house style. A PATH here is a rebalance-day OFFSET, not a seed: these books are deterministic rank-based books, so their path ensemble is the 4 offsets, and every figure is a median with the worst offset stated.',
      sharedCoreTitle: 'The two books, and the mechanics common to both',
      sharedCore: [
        { k: 'Branch A - rotation across sectors', v: 'At each rebalance, rank the nine real NSE sector indices, hold the top N, weight them, hold to the next rebalance. N of 1 to 5; weighting equal, rank-weighted, inverse-volatility or signal-proportional; clock monthly, quarterly or fortnightly; gate none, own-absolute-momentum-positive-else-cash, or NIFTY500-above-200-SMA-else-cash.' },
        { k: 'Branch B - sector as a universe filter', v: 'At each rebalance, take the top K industries by sector momentum, then hold the best stocks INSIDE them by a within-sector stock signal, equal-weighted across slots. K of 2 to 5; slots 10, 15 or 20; stock signal 126-day momentum, 12-1 momentum, risk-adjusted 126-day momentum or distance from the 252-day high; clock monthly or quarterly.' },
        { k: 'The twelve signal families', v: 'Absolute momentum at 21/42/63/126/189/252 days; 12-1 momentum skipping the last month; risk-adjusted momentum; trend-persistence vote; distance from the rolling high; distance from the 50/100/200-day average; ACCELERATION (short-horizon momentum minus its long-horizon share - the to-be-trending candidate); cross-sectional reversal; volatility-scaled momentum; constituent BREADTH and breadth CHANGE; and a low-volatility control.' },
        { k: 'Why relative strength is a gate and not a ranking', v: 'Ranking by return minus the NIFTY 500 return over the same lookback is RANK-IDENTICAL to ranking by absolute return, because the benchmark term is common to every sector. So relative strength enters the design as a gate - hold only sectors beating the benchmark - never as a separate ranking axis. Stated because Arun named relative strength specifically.' },
        { k: 'Execution', v: 'Signal computed on the close of the rebalance day, traded at the next close. 25 bps per side on traded notional, with a ladder to 40 and 60 on every finalist.' },
        { k: 'Tax and idle cash', v: '20% short-term / 12.5% long-term capital gains with Indian financial-year loss netting settled on the first trading day on or after 1 April, tracked on FIFO lots. Idle cash compounds at 5% a year whenever a gate is in cash. These are short-holding books: tax costs 2.7 to 3.2 percentage points of CAGR.' },
        { k: 'Drawdown convention', v: 'Every sub-window drawdown is measured from the running peak of the FULL curve, never from the window first bar - the research/154 correction.' },
      ],
      riskLayer: {
        title: 'The pre-registered gates, written into the STATUS document before the first run',
        caption:
          'Fixed in advance and applied literally. The falsification condition was also pre-registered: if the best surviving configuration advantage over the equal-weight null is smaller than the spread across rebalance-day offsets, the finding is noise regardless of headline CAGR.',
        columns: ['Gate', 'Criterion', 'Outcome'],
        rows: [
          ['G1 - does sector leadership exist', 'Monthly rank IC vs forward 1-month return with abs(t) at least 2.0 on both asset sets, or 2.5 on the real indices, AND monotone terciles', 'FAIL on the real indices - best abs(t) across 168 tests is 2.33, and 1.46 at the 1-month horizon'],
          ['G2 - branch A standalone', 'Offset-median CAGR at least 20%, worst offset at least 18%, Calmar at least 1.0, beating equal-weight, NIFTY 500, the 95th percentile of a 500-draw random-sector null and a cash null', 'FAIL - 0 of 1,440 configurations'],
          ['G3 - branch B decomposition', 'Must beat, on PAIRED offsets, the same stock rule with no sector filter, random stocks in the same sectors, and the same rule in random sectors', 'FAIL on the first - wins 5 of 16 paired comparisons against the no-filter control'],
          ['Complement bar', '+0.10 Calmar or -2pp drawdown at equal-or-better CAGR after tax, beating the cash null at the same weight, correlation below 0.40 to both live legs', 'FAIL on all three - best is +0.04 Calmar, loses to cash, correlation 0.41 to 0.54'],
        ],
        highlightRows: [1, 2],
      },
    },

    system: {
      intro:
        'Two questions, run as two branches, with the falsification tests that decide whether the wide-panel evidence can be trusted at all. Roughly 20,300 cells were evaluated in total, disclosed so any single number can be discounted for multiple testing - though there is no winner here to discount.',
      rows: [
        { k: 'P1 - does leadership exist', v: '388 information-coefficient cells: 28 signal specifications x forward horizons of 1 and 3 months x two asset sets x three windows. Rank IC, its t-statistic, hit rate and top-minus-bottom tercile spread.' },
        { k: 'P1b - the falsification tests', v: 'About 864 cells. (A) The same eight sectors over the same window, real index versus synthetic basket. (B) Each basket own full-sample drift removed - deliberately look-ahead, used only as a diagnostic. (C) Industry labels randomly shuffled, 50 draws, rebuilding the baskets each time.' },
        { k: 'P2 - branch A sweep', v: '1,440 configurations x 4 offsets on the real sector indices = 5,760 runs, and the same again on the synthetic panel, plus 24 equal-weight nulls, the buy-and-hold benchmarks, a cash null and a 2,500-run random-sector null.' },
        { k: 'P4 - branch B and its controls', v: '768 sector-gated stock books plus 3,204 paired control runs (no-sector-filter, random stocks in the same sectors at 100 draws, random sectors at 100 draws).' },
        { k: 'P5 - robustness', v: 'Finalist re-runs across offsets, cost ladder 25/40/60 with and without tax, two halves plus the 2018, 2020 and 2022H1 stress windows, per-year table.' },
        { k: 'P6 - portfolio fit', v: 'Daily and monthly correlation to True North, Open Alpha and the IPO sleeve, plus a blend weight sweep from 5% to 30% over 12 PAIRED paths against a cash null, every block computed on the candidate own window.' },
        { k: 'Ranking metric, fixed in advance', v: 'Offset-ensemble median after-tax Calmar, with median CAGR at least 20% and worst offset at least 18% as hard filters.' },
      ],
    },

    conditions: {
      intro:
        'The binding constraint is the sample: nine tradeable sector indices over 11.7 years. About 140 monthly rebalances, one crash (2020), the 2018 and 2022H1 grinds, and no 2008. Book windows start January 2016 after a 260-day signal warm-up. Given research/64 found Kite Quality, LowVol and Commodities index series CORRUPT, the same integrity checks were run here before anything else.',
      rows: [
        { k: 'The nine sector indices are clean', v: 'NIFTYAUTO, NIFTYIT, NIFTYENERGY, NIFTYFINSRV, NIFTYFMCG, NIFTYMETAL, NIFTYPHARMA, NIFTYPSUBANK, NIFTYREALTY: 2,894 to 2,895 bars from 01-Jan-2015, zero phantom holiday rows, at most one missing day against the NIFTY 50 calendar, no split-scale steps, and at most four days with an absolute move above 12% - all real events.' },
        { k: 'What is missing', v: 'NIFTYMEDIA, NIFTYINFRA, NIFTYPVTBANK, NIFTYCONSUMPTION and NIFTYCOMMODITIES are NOT in the database and were not used. BANKNIFTY has history from 2011 but is a SUBSET of NIFTYFINSRV, so it was excluded from the cross-section rather than double-counting the same bet.' },
        { k: 'The synthetic 20-industry panel, and why it is barred from headlines', v: 'Built from the 500-symbol / 20-industry map in the official Nifty 200, Midcap 150 and Smallcap 250 CSVs to widen the cross-section and reach 2008. Validated against the real sector indices over the 2015-2026 overlap: correlations 0.74 to 0.96 daily, but every basket out-drifts its real index by +4 to +14 percentage points of CAGR a year (IT +12.1, Pharma +13.7, Auto +12.2, FMCG +11.9, Metals +11.6, Realty +11.2, Financials +4.1, Energy +5.3). Shape tracks, level does not. Reported as a diagnostic only.' },
        { k: 'The survivorship signature is visible in the wedge itself', v: 'The industries with the SMALLEST drift wedge - Financial Services and Oil, Gas and Consumable Fuels - are the ones whose index membership is most stable. That is exactly what a survivorship explanation predicts, and it is why the wide panel cannot be read as sector economics.' },
        { k: 'Split adjustment', v: 'The database is not retroactively split-adjusted, so any single-day move beyond 40% was excluded from basket construction and from stock eligibility. Distance-from-high signals were run with that guard in place.' },
        { k: 'NaN discipline', v: 'All rolling statistics are computed on the per-series dropna and re-aligned, never on a union-index frame - the failure mode that silently disabled research/142 SMA-200 gate for months.' },
        { k: 'Window end', v: '29-Aug-2026, the last full week, so a live partial candle cannot contaminate the final bar. Data snapshot: market_data.db on the VPS. Read-only throughout; no reconstructed series was written to any database.' },
      ],
    },

    comparisons: [
      {
        title: 'G1 - the information coefficient on the REAL sector indices says nothing is there',
        caption:
          'Monthly rank information coefficient against the forward 1-month return, full window, nine real NSE sector indices. A maximum abs(t) of 2.33 across 168 correlated tests is what pure noise looks like. Spread is the top-minus-bottom tercile difference, annualised.',
        columns: ['Signal', 'IC t-stat (1m)', 'Tercile spread /yr', 'Monotone', 'IC t-stat (3m)'],
        rows: [
          ['63-day momentum', '1.46', '+8.22%', 'yes', '1.65'],
          ['ACCELERATION (the to-be-trending candidate)', '1.37', '+7.09%', 'yes', '1.79'],
          ['126-day risk-adjusted momentum', '1.03', '+4.76%', 'yes', '2.11'],
          ['200-SMA distance', '1.27', '+5.65%', 'no', '1.43'],
          ['126-day momentum', '0.82', '+5.31%', 'no', '1.20'],
          ['12-1 momentum (252 skip 21)', '-0.50', '+0.31%', 'no', '0.28'],
          ['Distance from the 252-day high', '-0.16', '-2.42%', 'no', '0.60'],
          ['Low volatility (252d) - the non-momentum control', '-0.66', '-3.96%', 'no', '-0.83'],
        ],
        heatmap: false,
      },
      {
        title: 'The falsification tests - same eight sectors, same window, real index versus synthetic basket',
        caption:
          'Only the CONSTRUCTION differs. The synthetic equal-weight baskets of current index members deliver 2 to 6 times the spread of the real cap-weighted indices for the identical sectors over the identical dates.',
        columns: ['Signal', 'REAL index IC t', 'REAL spread /yr', 'SYNTHETIC IC t', 'SYNTHETIC spread /yr'],
        rows: [
          ['63-day momentum', '1.44', '+6.72%', '2.74', '+15.53%'],
          ['126-day momentum', '0.92', '+2.59%', '2.56', '+14.29%'],
          ['200-SMA distance', '1.30', '+5.80%', '2.61', '+15.98%'],
          ['126-day risk-adjusted momentum', '1.14', '+5.89%', '1.92', '+8.54%'],
          ['63-day risk-adjusted momentum', '0.84', '+4.73%', '2.89', '+12.53%'],
          ['12-1 momentum', '0.33', '+2.50%', '1.47', '+6.90%'],
          ['252-day momentum', '0.67', '+6.04%', '1.62', '+6.99%'],
          ['ACCELERATION', '1.44', '+9.34%', '1.02', '+6.91%'],
        ],
        heatmap: false,
      },
      {
        title: 'The shuffled-label null - random groupings of the same stocks reproduce most of the sector signal',
        caption:
          'Industry labels randomly permuted across the 500 stocks (preserving industry sizes), baskets rebuilt, 50 draws. If this were sector economics the signal should vanish; instead a large part of it survives.',
        columns: ['Signal', 'Shuffled IC t: mean', '5th pct', '95th pct', 'max', 'REAL-label IC t'],
        rows: [
          ['126-day momentum', '1.61', '0.04', '3.12', '3.71', '5.22'],
          ['126-day risk-adjusted momentum', '1.88', '0.61', '3.52', '4.25', '5.59'],
          ['63-day risk-adjusted momentum', '1.34', '-0.23', '2.93', '3.11', '4.99'],
          ['200-SMA distance', '1.47', '0.16', '3.00', '3.43', '4.55'],
          ['12-1 momentum', '1.90', '0.38', '3.38', '3.77', '4.24'],
          ['63-day momentum', '1.03', '-0.46', '2.30', '3.21', '4.18'],
          ['252-day momentum', '1.83', '0.26', '3.27', '3.45', '3.59'],
          ['ACCELERATION', '-0.68', '-1.94', '0.66', '1.30', '0.22'],
        ],
        heatmap: false,
      },
      {
        title: 'Branch A - the rotation sweep against every required null',
        caption:
          'After 25 bps per side, after Indian financial-year-netted tax, idle cash 5% a year. Offset-ensemble medians over 4 rebalance-day offsets, window January 2016 to August 2026. Zero of 1,440 configurations clear the pre-registered bar.',
        columns: ['Book', 'CAGR (median)', 'Worst offset', 'MaxDD', 'Calmar'],
        rows: [
          ['Best by CAGR - 200-SMA distance, top 5 of 9, fortnightly, ungated', '16.3%', '15.8%', '-36.9%', '0.44'],
          ['Best by Calmar - 63d momentum, top 3, monthly, NIFTY500 200-SMA cash gate', '13.5%', '9.6%', '-21.5%', '0.63'],
          ['Equal-weight all nine sectors (the research/63 null)', '14.0%', '13.8%', '-43.3%', '0.32'],
          ['research/147 SECROT cell reproduced (top-2, 126d momentum, monthly)', '8.9%', '8.3%', '-56.0%', '0.16'],
          ['NIFTY 500 buy-and-hold', '13.2%', '-', '-38.3%', '0.35'],
          ['Midcap 150 buy-and-hold - beats EVERY rotation cell', '18.0%', '-', '-44.2%', '0.41'],
          ['Smallcap 250 buy-and-hold', '15.8%', '-', '-60.8%', '0.26'],
          ['Cash at 5%', '4.9%', '-', '0.0%', '-'],
        ],
        highlightRows: [5],
      },
      {
        title: 'The random-sector null - momentum ranks sectors better than chance, and it is not worth the concentration',
        caption:
          '500 random draws per sector count, same monthly clock, same costs and taxes as the real rule. This is the most precise finding in the study: rotating at random destroys 5 to 10 percentage points against equal-weight, and rotating on momentum recovers most of it and finishes level with simply holding everything.',
        columns: ['Sectors held', 'Random median CAGR', 'Random 95th pct', 'Best momentum config', 'Percentile of the best'],
        rows: [
          ['1', '4.04%', '13.07%', '12.79%', '94.4th'],
          ['2', '6.01%', '11.11%', '12.51%', '98.0th'],
          ['3', '7.13%', '11.69%', '14.08%', '99.4th'],
          ['4', '8.15%', '11.11%', '14.84%', '100.0th'],
          ['5', '9.01%', '11.40%', '14.97%', '100.0th'],
        ],
        heatmap: false,
      },
      {
        title: 'Branch B - the decomposition that kills the sector filter',
        caption:
          'Best book: top 5 industries by 126-day momentum, then the 10 best stocks inside them by risk-adjusted 126-day momentum, quarterly, on the 500-name universe with a Rs 5 crore 20-day-median traded-value floor. Every control is PAIRED on the same rebalance-day offsets.',
        columns: ['Arm', 'CAGR', 'MaxDD', 'Calmar', 'What it proves'],
        rows: [
          ['The sector-gated book', '32.5%', '-38.3%', '0.85', 'Clears the 20% bar - then the controls arrive'],
          ['CTRL_NOSECT - same stock rule, FULL universe, no sector filter', '32.9%', '-39.3%', '0.84', 'The sector layer adds NOTHING. The book wins on CAGR in 5 of 16 paired comparisons'],
          ['CTRL_RNDSTK - random stocks inside the same top-K industries (100 draws)', '18-22% median', '-', '0.36-0.54', 'The STOCK ranking does all the work - the book beats 91-100% of draws'],
          ['CTRL_RNDSEC - same stock rule inside K RANDOM industries (100 draws)', '19-24% median', '-', '0.43-0.58', 'Right sectors beat wrong sectors - but both lose to not restricting at all'],
          ['Sector ranking taken from the REAL indices instead of the synthetic baskets', '14.4-17.8% median', '-', '0.31-0.35', 'When the sector signal comes from real tradeable prices, the whole branch degrades'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Correlation to the deployed sleeves - every candidate breaches the 0.40 ceiling',
        caption:
          'Median paths, daily / monthly returns, 2016-2026. For reference, Open Alpha to True North is 0.421 (research/154). These candidates are about as correlated to the incumbents as the incumbents are to each other.',
        columns: ['Candidate', 'vs True North', 'vs Open Alpha', 'vs IPO base'],
        rows: [
          ['Sector rotation (best CAGR)', '0.435 / 0.330', '0.462 / 0.550', '0.288 / 0.306'],
          ['Sector rotation (best Calmar)', '0.495 / 0.382', '0.428 / 0.471', '0.284 / 0.299'],
          ['Equal-weight all nine sectors', '0.372 / 0.326', '0.436 / 0.559', '0.252 / 0.295'],
          ['Sector-gated stock book', '0.413 / 0.443', '0.539 / 0.619', '0.329 / 0.235'],
          ['research/147 SECROT', '0.414 / 0.298', '0.433 / 0.519', '0.252 / 0.247'],
        ],
        heatmap: false,
      },
      {
        title: 'Blend value at 10% weight - and the cash null beats every candidate',
        caption:
          'True North 40 / Open Alpha 40 / IPO 20 baseline, 12 PAIRED paths, every block computed on the CANDIDATE OWN WINDOW - the baseline and the cash null are re-run on the identical index, because mixing windows is the error research/152 was caught on. Sweeping 5% to 30% changes nothing: the Calmar gain never exceeds +0.05 and CAGR falls monotonically.',
        columns: ['Candidate', 'Baseline, same window', 'Candidate at 10%', 'CASH NULL at 10%', 'Verdict'],
        rows: [
          ['Sector rotation (best CAGR)', '35.45% / -18.45% / 1.99', '33.67% / -17.60% / 1.99', '32.23% / -16.33% / 2.04', 'loses CAGR, no Calmar gain, cash wins'],
          ['Sector rotation (best Calmar)', '35.54% / -18.45% / 1.99', '33.15% / -18.28% / 1.87', '32.30% / -16.33% / 2.05', 'worse on every axis'],
          ['Equal-weight nine sectors', '35.54% / -18.45% / 1.99', '33.48% / -17.78% / 1.92', '32.30% / -16.33% / 2.05', 'cash wins'],
          ['Sector-gated stock book', '36.15% / -18.45% / 2.03', '35.99% / -18.01% / 2.04', '32.84% / -16.33% / 2.08', '+0.01 Calmar, cash wins'],
          ['Same stocks, no sector filter', '36.15% / -18.45% / 2.03', '36.09% / -17.55% / 2.07', '32.84% / -16.33% / 2.08', '+0.04 Calmar, cash still wins'],
          ['research/147 SECROT', '35.54% / -18.45% / 1.99', '32.91% / -18.14% / 1.85', '32.30% / -16.33% / 2.05', 'worse on every axis'],
        ],
        heatmap: false,
      },
    ],

    results: {
      metrics: [
        { label: 'Configs clearing the bar', value: '0 of 1,440', tone: 'neg' },
        { label: 'Best rotation CAGR', value: '16.3%', hint: 'after tax, -36.9% DD, Calmar 0.44' },
        { label: 'Midcap 150 buy-and-hold', value: '18.0%', hint: 'beats every rotation cell, Calmar 0.41', tone: 'pos' },
        { label: 'Sector-gated stocks', value: '32.5% / 0.85', hint: 'clears the 20% bar' },
        { label: 'Same stocks, NO sector filter', value: '32.9% / 0.84', hint: 'the sector layer adds nothing', tone: 'neg' },
        { label: 'Correlation to the live legs', value: '0.41 - 0.54', hint: 'bar was below 0.40', tone: 'neg' },
        { label: 'Best blend Calmar gain', value: '+0.04', hint: 'bar was +0.10, and cash beats it', tone: 'neg' },
        { label: 'Cells evaluated', value: '~20,300', hint: 'disclosed for multiple testing' },
      ],
      tables: [
        {
          title: 'Year by year, house format - annual return with the intra-year drawdown beneath it',
          caption:
            'After tax, net of 25 bps per side, offset-ensemble medians. Common window 31-Mar-2016 to 28-Aug-2026. Every drawdown is measured from the running peak of the FULL curve, never from the year first bar. Benchmarks are excluded from the best-of picks. READ THE LAST ROW WITH THE WINDOW IN MIND: Open Alpha at 44.4% and True North at 24.1% are their 2016-2026 figures, not their long-run numbers (34.90% and 19.91%). This decade flattered every equity book, sector rotation included.',
          columns: ['Year', 'Sector rot. (best Calmar)', 'Sector rot. (best CAGR)', 'r/147 SECROT', 'Equal-wt 9 sectors', 'Sector-gated stocks', 'Same stocks, NO filter', 'True North (LIVE)', 'Open Alpha (LIVE)', 'Deployed TN40/OA40/IPO20', 'NIFTY 500', 'Midcap 150', 'BEST CAGR', 'LEAST DD', 'BEST OVERALL'],
          rows: [
            ['2016', '+15.8 (-9.1)', '+12.1 (-12.4)', '+21.4 (-13.1)', '+11.5 (-11.6)', '+11.5 (-18.5)', '+20.7 (-21.1)', '+32.2 (-6.8)', '+27.7 (-8.0)', '+34.0 (-6.3)', '+8.2 (-12.0)', '+13.2 (-14.5)', 'Deployed', 'Deployed', 'Deployed'],
            ['2017', '+30.8 (-6.4)', '+38.0 (-9.4)', '+23.9 (-10.4)', '+34.5 (-7.7)', '+58.3 (-13.8)', '+99.1 (-14.7)', '+39.2 (-8.7)', '+104.4 (-18.1)', '+68.2 (-8.0)', '+35.9 (-8.4)', '+54.3 (-9.9)', 'Open Alpha', 'Sector rot. (Calmar)', 'Open Alpha'],
            ['2018', '-10.1 (-18.5)', '-4.6 (-14.9)', '-7.4 (-19.0)', '-7.2 (-17.2)', '-10.9 (-24.6)', '-7.1 (-24.6)', '-5.6 (-23.1)', '-18.5 (-19.3)', '-9.3 (-17.4)', '-3.4 (-15.8)', '-13.3 (-24.0)', 'Sector rot. (CAGR)', 'Sector rot. (CAGR)', 'Sector rot. (CAGR)'],
            ['2019', '+0.4 (-19.1)', '+6.2 (-13.3)', '-5.4 (-26.1)', '+1.9 (-20.5)', '+13.7 (-22.5)', '+36.5 (-22.8)', '-0.7 (-21.2)', '+4.7 (-24.5)', '+4.5 (-16.4)', '+7.7 (-12.7)', '-0.3 (-25.6)', 'No-filter stocks', 'Sector rot. (CAGR)', 'No-filter stocks'],
            ['2020', '+26.7 (-20.9)', '+25.6 (-36.9)', '-9.2 (-56.0)', '+14.7 (-43.3)', '+54.8 (-30.3)', '+45.9 (-32.1)', '+57.4 (-23.5)', '+117.4 (-14.9)', '+87.9 (-11.3)', '+16.7 (-38.3)', '+24.4 (-44.2)', 'Open Alpha', 'Deployed', 'Open Alpha'],
            ['2021', '+32.8 (-14.8)', '+29.2 (-14.5)', '+23.5 (-28.2)', '+35.5 (-11.6)', '+230.6 (-9.6)', '+114.0 (-15.7)', '+59.9 (-13.6)', '+151.3 (-9.5)', '+88.6 (-6.4)', '+30.2 (-10.0)', '+46.8 (-10.5)', 'Sector-gated stocks', 'Deployed', 'Sector-gated stocks'],
            ['2022', '+3.5 (-21.5)', '+11.4 (-21.9)', '+3.3 (-29.6)', '+8.4 (-19.3)', '-0.7 (-38.3)', '-1.0 (-39.3)', '+12.0 (-17.8)', '+4.4 (-20.3)', '+10.7 (-12.9)', '+3.0 (-18.5)', '+3.0 (-21.6)', 'True North', 'Deployed', 'Deployed'],
            ['2023', '+25.5 (-20.9)', '+36.7 (-12.8)', '+18.6 (-24.7)', '+32.4 (-12.3)', '+107.1 (-26.9)', '+90.1 (-26.8)', '+41.0 (-11.0)', '+44.6 (-19.9)', '+48.9 (-13.7)', '+25.8 (-11.2)', '+43.7 (-10.4)', 'Sector-gated stocks', 'True North', 'Sector-gated stocks'],
            ['2024', '+23.6 (-9.3)', '+18.0 (-12.3)', '+27.2 (-11.7)', '+16.4 (-11.2)', '-7.4 (-28.7)', '+14.2 (-22.0)', '+24.6 (-17.3)', '+63.9 (-11.3)', '+50.8 (-10.9)', '+15.2 (-10.9)', '+23.8 (-11.0)', 'Open Alpha', 'Sector rot. (Calmar)', 'Open Alpha'],
            ['2025', '-3.4 (-19.2)', '-1.8 (-26.8)', '+6.0 (-17.4)', '+5.0 (-21.7)', '+10.1 (-33.2)', '-8.1 (-31.6)', '+5.1 (-18.0)', '+11.4 (-25.1)', '+7.5 (-14.8)', '+6.7 (-18.8)', '+5.4 (-21.1)', 'Open Alpha', 'Deployed', 'Deployed'],
            ['2026 (to Aug)', '+4.4 (-9.3)', '+6.0 (-19.9)', '+0.1 (-15.5)', '+0.8 (-16.0)', '-2.4 (-20.7)', '+2.5 (-24.1)', '+5.9 (-10.9)', '+37.1 (-21.7)', '+30.4 (-9.0)', '-1.4 (-16.2)', '+5.5 (-14.0)', 'Open Alpha', 'Deployed', 'Deployed'],
            ['FULL PERIOD - CAGR / MaxDD / Calmar', '13.5% / -21.5% / 0.63', '16.2% / -36.9% / 0.44', '8.9% / -56.0% / 0.16', '14.0% / -43.3% / 0.32', '32.5% / -38.3% / 0.85', '32.9% / -39.3% / 0.84', '24.1% / -23.5% / 1.03', '44.4% / -25.1% / 1.77', '36.9% / -17.4% / 2.12', '13.2% / -38.3% / 0.35', '18.0% / -44.2% / 0.41', '-', '-', '-'],
          ],
          highlightRows: [11],
        },
        {
          title: 'Stress windows - both halves agree, so nothing here is a single-regime artefact',
          caption:
            'Offset medians. Drawdown measured from the running peak of the full curve. The best-Calmar rotation book gentle 2020 is the NIFTY 500 200-SMA cash gate firing, not a sector call - and that same gate already sits inside True North.',
          columns: ['Book', '2018 grind', '2020 crash', '2022H1 grind', 'H1 total return', 'H2 total return'],
          rows: [
            ['Sector rotation (best CAGR)', '-4.7% (-15.3 DD)', '-15.6% (-37.0)', '-12.3% (-21.9)', '+123.5%', '+123.8%'],
            ['Sector rotation (best Calmar)', '-10.2% (-20.9)', '-7.6% (-21.6)', '-11.4% (-22.7)', '+114.8%', '+73.3%'],
            ['Equal-weight nine sectors', '-6.7% (-17.2)', '-22.8% (-43.3)', '-10.7% (-19.3)', '+97.8%', '+105.9%'],
            ['Sector-gated stock book', '-10.3% (-24.6)', '-6.2% (-30.7)', '-28.0% (-39.4)', '+368.4%', '+308.6%'],
            ['Same stocks, no sector filter', '-7.7% (-25.6)', '-12.5% (-32.1)', '-27.3% (-40.3)', '+627.1%', '+144.9%'],
            ['research/147 SECROT', '-7.3% (-19.2)', '-34.8% (-56.1)', '-18.1% (-32.5)', '+46.4%', '+64.3%'],
          ],
          heatmap: false,
        },
        {
          title: 'Cost and tax sensitivity - nothing is close enough to the bar for an assumption to rescue it',
          caption: 'Offset medians, CAGR / MaxDD / Calmar. Tax costs 2.7 to 3.2 percentage points of CAGR: these are short-holding books taxed at 20% throughout.',
          columns: ['Book', '25 bps, taxed', '40 bps, taxed', '60 bps, taxed', '25 bps, UNTAXED'],
          rows: [
            ['Best CAGR', '16.3% / -37.0% / 0.44', '15.4% / 0.40', '14.2% / 0.36', '19.0% / 0.51'],
            ['Best Calmar', '13.3% / -24.0% / 0.56', '11.9% / 0.47', '10.1% / 0.37', '16.5% / 0.88'],
            ['research/147 SECROT', '9.2% / -56.1% / 0.16', '7.9% / 0.14', '6.3% / 0.11', '11.2% / 0.20'],
          ],
          heatmap: false,
        },
      ],
      charts: [
        {
          src: '/app/sector-rotation-research156.png',
          caption:
            'Growth of Rs 100 on a log scale with a drawdown panel beneath, 31-Mar-2016 to 28-Aug-2026, after tax and 25 bps per side, offset-ensemble medians. The sector-rotation books track the index benchmarks; the two stock books (with and without the sector filter) are nearly indistinguishable from each other; the deployed True North / Open Alpha / IPO blend sits above all of them at a fraction of the drawdown.',
        },
      ],
    },

    winners: [
      {
        config: 'There is no winner - the honest outcome is the passive benchmark',
        summary:
          'The single most useful sentence to take away: over this window, holding the Midcap 150 index returned 18.0% CAGR at Calmar 0.41, and NOT ONE of the 1,440 sector-rotation configurations beat it on both measures. When a 1,440-cell sweep ceiling sits below a passive benchmark, there is nothing to discount for multiple testing - the family is simply dead.',
        metrics: [
          { k: 'Configurations clearing the pre-registered bar', v: '0 of 1,440' },
          { k: 'Best rotation cell', v: '16.3% CAGR / -36.9% / Calmar 0.44' },
          { k: 'Midcap 150 buy-and-hold, same window', v: '18.0% / -44.2% / 0.41' },
          { k: 'Falsification condition', v: 'Met with room - the best book beats equal-weight by 2.3pp of CAGR while its own offsets span 15.8% to 17.2%' },
        ],
        rejected: [
          'To-be-trending / early detection: ACCELERATION scores t = 1.37 at one month and 1.79 at three on the real indices - the best of the early-detection family and still not significant. Breadth CHANGE scores t = 0.36. Nothing anticipates leadership; the momentum family only confirms it, and even that confirmation is weak.',
          'Cross-sectional reversal is the exact mirror of momentum by construction and is negative wherever momentum is positive - a useful sanity check that the engine is sound, and no independent signal.',
          'Low volatility as a non-momentum control: t = -0.66 at one month, spread -3.96% a year. Nothing there either.',
          'Ranking sectors on the REAL indices for branch B is materially WORSE than ranking on the synthetic baskets (14.4-17.8% CAGR versus 20.8-23.0%). When the sector signal comes from real tradeable prices the branch degrades - the construction artefact showing up downstream.',
        ],
      },
    ],

    caveats: [
      'THE SAMPLE IS SMALL AND IT IS THE BINDING CONSTRAINT. Nine tradeable sector indices over 11.7 years is about 140 monthly rebalances with a nine-name cross-section. That is enough to say no clear signal is detectable; it is NOT enough to prove none exists. A modest true edge could hide inside this noise. The finding is that nothing detectable-and-tradeable is here, not that sector rotation is impossible in principle.',
      'THE SYNTHETIC 20-INDUSTRY PANEL IS SEVERELY SURVIVORSHIP-BIASED AND IS BARRED FROM EVERY HEADLINE. Membership is today Nifty-500 union applied backwards; the measured drift wedge is +4 to +14 percentage points of CAGR a year against the real indices. It is reported because the falsification tests built on it are the most informative part of the study, not because its performance numbers mean anything.',
      'BRANCH B RUNS ON A SURVIVORSHIP-BIASED STOCK UNIVERSE TOO - the same 500 current index members. The absolute 32% CAGR is inflated by that. The DECOMPOSITION is much less affected, because the book and all three controls draw from the identical universe, and the decomposition is the finding.',
      'THE WINDOW FLATTERED EVERYTHING. 2016-2026 was a strong decade for Indian equity: Open Alpha prints 44.4% CAGR on it against 34.90% over its full history, and True North 24.1% against 19.91%. Comparisons within the table are fair because every column shares the window, but no level in it should be read as a forward expectation.',
      'BRANCH A IS NOT STRAIGHTFORWARDLY TRADEABLE EVEN IF IT HAD PASSED. None of the nine sector indices is directly investable. Liquid futures exist only on BANKNIFTY (a subset of one of them, and excluded here); sector ETFs exist for a few and are thin; several sectors have no tradeable wrapper at all. A real implementation would be constituent baskets - which is branch B, which the decomposition already killed.',
      'NO 2008. The real sector series start in 2015, so the one crash in the window is 2020, which India recovered from in a V. A slow grinding bear of the 2008 or 2011 kind is untested for this family.',
      'MULTIPLE TESTING: about 20,300 cells. Normally that demands a heavy discount on any winner. Here it cuts the other way - the SWEEP CEILING sits below the passive benchmarks, which is the cleanest form a negative result can take.',
      'WHAT WOULD CHANGE THIS VERDICT, in priority order: (1) point-in-time sector index constituents, which would let branch B be re-asked without survivorship; (2) back-filling the real sector indices to their 2005 inception, which NSE publishes and our database lacks - it would add 2008 and roughly double the sample, and it is a data-acquisition task rather than a modelling one; (3) non-price sector inputs such as earnings revisions or sector flows, none of which we hold.',
      'NOT TESTED, AND WHY: intraday or weekly sector timing (the intraday line is closed by research/109 and /110 - 58 constructions, none clearing the roughly 10 bps cost floor); long-short sector pairs (no shorting infrastructure, and research/147 killed index trend long-short when the 2020 V-recovery caught it short); options overlays (no liquid sector options in India beyond BANKNIFTY, and five prior kills of the structure-on-a-weak-signal family in research/129 and /150); macro or fundamental sector inputs (not in our data). One legitimate cheap follow-up was NOT run: using sector rank as a GATE on the existing books - though the branch-B decomposition, which shows the sector filter is return-neutral on a momentum stock book, is the strongest available prior against it.',
      'NOTHING WAS DEPLOYED. No live or paper engine, crontab, sizing, gate or spec was touched. Read-only access to market_data.db throughout, and no reconstructed series was written to any database.',
    ],
    githubLinks: [{ label: 'research/156 (repo)', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/156_sector_rotation' }],
    projectPaths: [
      'research\\156_sector_rotation\\SECTOR_TREND_ROTATION_DAILY_SWEEP_STATUS.md',
      'research\\156_sector_rotation\\results\\RESULTS.md',
    ],
  },
"""


def main():
    txt = TS.read_text(encoding="utf-8")
    if SLUG in txt:
        print("already published, skipping TS edit")
    else:
        marker = "];\n\nexport function getStudy"
        assert marker in txt, "anchor not found"
        txt = txt.replace(marker, ENTRY + marker, 1)
        TS.write_text(txt, encoding="utf-8")
        print(f"inserted {SLUG}; file now {len(txt.splitlines())} lines")
    PUB.mkdir(parents=True, exist_ok=True)
    src = RES / "sector_rotation_research156.png"
    dst = PUB / "sector-rotation-research156.png"
    shutil.copy(src, dst)
    print("copied chart ->", dst)


if __name__ == "__main__":
    main()
