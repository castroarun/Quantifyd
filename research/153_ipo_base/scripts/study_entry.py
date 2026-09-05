# -*- coding: utf-8 -*-
"""Insert the research/153 BacktestStudy entry into frontend/src/data/backtests.ts.
Run ON THE VPS (the laptop copy of backtests.ts is stale). Idempotent."""
from pathlib import Path

P = Path("/home/arun/quantifyd/frontend/src/data/backtests.ts")
s = P.read_text(encoding="utf-8")
if "ipo-base-breakout-research153" in s:
    print("already present")
    raise SystemExit(0)

ENTRY = r"""  {
    slug: 'ipo-base-breakout-research153',
    title: 'IPO Base Breakout (bananapatterns) - the third sleeve that finally clears the bar, once the listing dates are rebuilt from scratch',
    verdict:
      'The hard part of this study was not the strategy, it was the data. We have no listing-date table, and the obvious proxy - the first row a symbol has in our database - is only 70% accurate. It fails three ways, and one of them would have invalidated the whole study: bulk DATA-ONBOARDING WAVES. 451 symbols begin on 2005-01-03, 95 on 2015-01-01, 45 on 2026-08-17, 41 on 2026-04-20 and 15 on 2025-05-26 - and that last batch contains ABB, a company listed in the 1990s. Left alone, ABB is a 2025 IPO and its next breakout is an IPO base breakout. A second defect: DELHIVERY carries eight rows at Rs 5-11 from 2016, weeks apart on 150-500 shares, before its real 2022-05-24 listing at Rs 536 - a different instrument on the same ticker, and a 93x price jump sitting inside what a base window would measure. A vetted listing table was built to fix all three (reject start days shared by 8 or more symbols; strip leading junk rows by price jump, date gap and dust volume; require a listing-day volume or range fingerprint) and it was validated before use: 48 of 48 known NSE IPOs accepted, the listing date exact to within three days for 47 of them, and 0 of 12 known long-listed onboardings wrongly accepted. Only then was a single backtest run. WHAT THE SCREEN IS. A newly listed stock builds its first consolidation and breaks out of it. We swept the definition rather than matching the site, because the panel dials were never legible: 256 signal-geometry cells and 384 exit-and-book cells, each a seed ensemble on two windows, 680 cells in all. 207 of 256 and 383 of 384 clear a positive after-cost per-trade expectancy in BOTH windows - this is a broad plateau, not a peak. THE ANSWER TO THE OBVIOUS OBJECTION. How can the site apply an IBD relative-strength filter of 70 to a six-month-old listing when the score needs 252 trading days? It cannot: the strict filter returns literally zero signals below a 12-month age band, and every short-window substitute we built made the book worse. This screen is pure price structure. TWO OF THE SITE OWN DIALS ARE WRONG. Their Trail 30-week setting is the worst exit we tested (Calmar 0.49 against 0.99 for a 20-day trail) and their Breakout close entry costs 14 percentage points of CAGR against a buy-stop at the pivot, losing on 30 of 30 paired seeds. Their Take +25% dial, by contrast, is the single best thing in the study and helps in every geometry. THE VERDICT. Adopted spec: listed within 6 months, 25-day base no deeper than 30%, buy-stop at the base high, -8% close stop, exit below the 20-day SMA, +25% take-profit, 8 slots at 18.75%. Standalone, 30 seeds, after Indian tax and 25 bps a side, 2006 to Sep-2026: 31.03% CAGR [28.82..33.44], worst seed 28.82%, drawdown -20.88%, Calmar 1.50, 32.6 trades a year, and it keeps 4.89% per trade with each seed ten best trades deleted. Against a date-matched random-entry null it wins on 29 of 30 paired seeds by +0.96pp per trade. As a THIRD SLEEVE at 20% weight beside True North and Open Alpha it delivers +1.13pp of CAGR, -3.63pp of drawdown and +0.56 of Calmar against the deployed pair, at correlations of 0.16 daily to Open Alpha and 0.18 to True North - lower than the correlation between the two legs already running - and it beats a plain-cash sleeve at the same weight by 5.6pp of CAGR. Every leg of the pre-registered adoption bar is met with room. The honest caveats are that the entire edge lives in getting filled AT the pivot, and that the book earned nothing but the cash yield in 2013 and 2014 because the Indian IPO pipeline was shut.',
    status: 'COMPLETE',
    date: '2026-09-05',
    cardBlurb:
      'A fourth bananapatterns screen on the research/142 engine. Most of the work was rebuilding listing dates the database does not have - bulk download waves were masquerading as IPOs. What survives is the first third-sleeve candidate to clear the pre-registered bar on every leg.',
    cardStats: [
      { label: 'Verdict', value: 'STRATEGY candidate - third sleeve' },
      { label: 'Standalone (30 seeds)', value: '31.03% CAGR / -20.88% DD / Calmar 1.50' },
      { label: 'Blend at 20%', value: '+0.56 Calmar, -3.63pp DD vs the deployed pair' },
    ],
    system: {
      intro: 'The engine extends the research/142 decoded bananapatterns engine, reusing its validated common mechanics unchanged: IBD-weighted RS percentile, 20-day median traded value of at least Rs 5 cr at t-1, ETFs excluded, buy-stop AT the pivot filled max(pivot, open), close-based hard stop and close-based moving-average trail, and a random-selection seed ensemble because the book always has more qualifying signals than slots.',
      sharedCoreTitle: 'Shared core - the decoded bananapatterns mechanics plus the IPO-base signal',
      sharedCore: [
        { k: 'The signal', v: 'A stock listed within the last N months (swept 3 / 6 / 12 / 24) closes above the highest close of the last L trading days (swept 15 / 25 / 40 / 60), where that base is no deeper than D from pivot to lowest low (swept 20 / 30 / 40 / 60%), and it was not already above the pivot on the previous close.' },
        { k: 'Universe gate', v: 'Only symbols in the VETTED listing table (1,293 accepted listings, 2006-2026, of which 786 ever become young and liquid). All price rows before the vetted listing date are MASKED out of the panel, so no base window can contain a pre-listing artefact.' },
        { k: 'Relative strength', v: 'OFF, and that is a finding rather than a shortcut. Four policies were tested; the strict 252-day IBD score returns ZERO signals for any age band at or below 12 months, and a 3-month-return substitute costs about 8pp of CAGR by starving the book.' },
        { k: 'Exits (tested jointly with entry, never alone)', v: '-7 / -8 / -10% close stop; trail on a close below SMA 20 / 30 / 50 / 150 (150 is the site Trail 30-week); +25% take-profit on and off.' },
        { k: 'Sizing', v: 'Slots x size of 5x30 / 8x18.75 / 10x15 / 16x6.25% of NAV. The site risk-per-trade dial is ALGEBRAICALLY IDENTICAL to fixed-fraction sizing when the stop is a fixed percentage: size% = risk% / stop%, so 1.5% risk over an 8% stop IS 18.75%. It is a sizing dial wearing a risk-management label.' },
        { k: 'Adopted spec', v: 'Listed within 6 months, 25-day base, depth at most 30%, no RS filter, buy-stop at the base high filled max(pivot, open), -8% close stop, exit on a close below SMA-20, +25% take-profit, 8 slots at 18.75% of NAV, NO market gate.' },
      ],
      riskLayer: {
        title: 'The three age bands actually run (each a 30-seed ensemble, after tax)',
        columns: ['Sleeve', 'Recency of listing', 'Base', 'Trail', 'Slots', 'Character'],
        rows: [
          ['NARROW', 'within 3 months', '25 days, depth <= 30%', 'SMA-20 + 25% TP', '8 x 18.75%', 'Purest IPO base; only 19.6% invested'],
          ['MID (adopted)', 'within 6 months', '25 days, depth <= 30%', 'SMA-20 + 25% TP', '8 x 18.75%', 'Best blend value; 32.7% invested'],
          ['WIDE', 'within 24 months', '15 days, depth <= 30%', 'SMA-20 + 25% TP', '8 x 18.75%', 'No longer an IPO base - converges on Open Alpha'],
        ],
        highlightRows: [1],
      },
    },
    conditions: {
      intro: 'Pre-registered in the STATUS document BEFORE the first backtest, including the falsification criteria and the complement bar.',
      rows: [
        { k: 'Ranking metric', v: 'Median after-tax CAGR on W2, gated on positive per-trade expectancy net of 25 bps a side in BOTH windows and at least 4 trades a year.' },
        { k: 'Complement bar', v: '+0.10 Calmar OR -2pp drawdown at greater-than-or-equal CAGR against the deployed TN+OA 50-50 book, after tax, robust across seeds and offsets, correlation below about 0.40 to BOTH legs, and beating a plain-cash sleeve at the same weight.' },
        { k: 'Windows', v: 'W1 2020-2025 (the site window) and W2 2006-01 to 2026-09-04. Both must pass.' },
        { k: 'Path dependence', v: '10 random-selection seeds to scan, 30 seeds for any adoption decision, reported as median [min..max] plus the worst seed. Every A-vs-B is PAIRED on the same seed.' },
        { k: 'Scale of the search', v: '256 + 384 + about 40 control runs = 680 cells, disclosed so any single winner is discounted as best-of-680.' },
        { k: 'Costs / tax / cash', v: '25 bps a side headline with a 25 / 40 / 60 ladder; 20% STCG and 12.5% LTCG netted across the Indian FINANCIAL year with loss carry-forward, settled 1 April; idle cash at 5% a year.' },
      ],
    },
    comparisons: [
      {
        title: 'The listing-date proxy had to be rebuilt before anything could be tested',
        columns: ['Defect', 'Evidence', 'Fix', 'Result'],
        rows: [
          ['Bulk data-onboarding waves read as IPOs', '451 symbols start 2005-01-03, 95 on 2015-01-01, 45 on 2026-08-17, 41 on 2026-04-20, 15 on 2025-05-26 (includes ABB, listed in the 1990s)', 'Reject any start day shared by 8 or more symbols (real waves carry 12-451; genuine multi-IPO days carry 2-6)', '0 of 12 known onboardings wrongly accepted'],
          ['Pre-listing junk rows on a reused ticker', 'DELHIVERY: 8 rows at Rs 5-11 from 2016 on 150-500 shares before the real Rs 536 listing - a 93x jump. Also FUSION (97x), LATENTVIEW, SBICARD, STARHEALTH, MAZDOCK', 'Strip leading rows to the last of a price jump above 3x or below 1/3 in the first 250 rows, a date gap over 30 days, or volume under 5,000 shares in the first 60 rows - then MASK them from the panel', 'Listing date exact to within 3 days for 47 of 48'],
          ['No check that a listing looks like one', 'Known-IPO day-1 volume is a median 15x the next 20 days median; onboardings are about 1x', 'Accept if day-1 volume ratio is at least 1.5 OR day-1 high-low range is at least 8%', '48 of 48 known NSE IPOs accepted'],
        ],
      },
      {
        title: 'How could the site apply RS >= 70 to a six-month-old listing? It cannot.',
        columns: ['RS policy', 'Signals (age <= 12 months)', 'Median W2 CAGR across matched cells', 'Read'],
        rows: [
          ['STRICT - 252-day IBD score >= 70 required', '0', 'n/a', 'Mathematically impossible below a 12-month age band'],
          ['RELAXED - apply where computable, pass where not', 'identical to OFF below 12 months', '19.9%', 'Not a filter at all for this screen'],
          ['SHORT - 3-month return percentile >= 70', '1,010 of 2,322', '11.9%', 'Costs about 8pp of CAGR, mostly by starving the book'],
          ['OFF (adopted)', '2,322', '19.8%', 'The screen is pure price structure'],
        ],
        highlightRows: [3],
      },
      {
        title: 'The age band is a clean dose-response - and it decides what you actually own (30 seeds, 2006 to Sep-2026, after tax, 25 bps, 5% idle cash)',
        columns: ['Sleeve', 'Age band', 'CAGR median [min..max]', 'MaxDD (worst seed)', 'Calmar', 'Trades/yr', 'Invested', 'Edge over date-matched null'],
        rows: [
          ['NARROW', 'listed <= 3 months', '24.10 [22.97..24.87]', '-13.80 (-17.68)', '1.74', '19.4', '19.6%', '+2.08pp per trade'],
          ['MID - ADOPTED', 'listed <= 6 months', '31.03 [28.82..33.44]', '-20.88 (-23.23)', '1.50', '32.6', '32.7%', '+0.96pp per trade'],
          ['WIDE', 'listed <= 24 months, 15-day base', '35.99 [32.40..42.29]', '-33.34 (-40.49)', '1.08', '61.7', '59.5%', '+0.97pp per trade'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Two of the site own dials are the worst settings we tested (median W2 Calmar across all 384 exit-and-book cells)',
        columns: ['Dial', 'Setting', 'Calmar', 'Read'],
        rows: [
          ['Trail', 'SMA-20', '0.99', 'Best. Monotone: 20 beats 30 beats 50 beats 150'],
          ['Trail', 'SMA-30', '0.80', ''],
          ['Trail', 'SMA-50', '0.74', ''],
          ['Trail', 'SMA-150 = the site Trail 30-week', '0.49', 'The single worst exit in the study'],
          ['Take profit', '+25% ON', '0.93', 'Helps in EVERY geometry, trail and slot count - the best dial found'],
          ['Take profit', 'OFF', '0.69', ''],
          ['Hard stop', '7% / 8% / 10%', '0.78 / 0.75 / 0.75', 'Inert - the trail binds first'],
        ],
      },
      {
        title: 'Robustness battery - adopted spec, W2 2006 to Sep-2026, 30 seeds unless stated',
        columns: ['Test', 'Result', 'Read'],
        rows: [
          ['30-seed ensemble', '31.03% [28.82..33.44] CAGR, DD -20.88% (worst seed -23.23%), Calmar 1.50', 'Tight band; 32.6 trades a year'],
          ['Cost ladder 25 / 40 / 60 bps a side', '31.03% / 28.77% / 25.76% CAGR', 'About 1.8pp of CAGR per +15 bps a side. Survives 60 bps'],
          ['Market gate ON (NIFTYBEES below its 200-DMA)', '-5.23pp CAGR, loses on 0 of 30 paired seeds', 'Rejected - the same result Open Alpha reached'],
          ['FILL: buy-stop at the pivot vs the signal-day close', '31.03% vs 16.98% - a gap of 14.08pp, close-fill loses on 0 of 30 seeds', 'THE critical dependency. The whole edge lives in the entry price'],
          ['Date-matched random-entry null', 'Real beats a random young-and-liquid name on the same days by +1.75pp CAGR and +0.96pp per trade; real wins 29 of 30 paired seeds', 'The base-and-breakout mechanics genuinely add value'],
          ['Cohort drift null', 'Equal-weight hold of every young-and-liquid name: 17.8% CAGR at -82.6% drawdown, gross', 'The cohort drifts up; the screen contribution is the risk control'],
          ['Delete each seed ten best trades', 'Per-trade return 5.39% to 4.89% (4.39% net of a 50 bps round trip); the top-10 trades are only 11% of the summed return', 'Not a lottery-ticket book'],
          ['W1 2020-2025 (the site window)', '44.57% CAGR / -20.78% DD / Calmar 2.18', 'Both windows pass; the recent one is far more generous'],
        ],
      },
      {
        title: 'Portfolio fit - 3-sleeve blend, monthly rebalanced, after tax, medians over 10 OA seeds x 3 TN offsets x 10 IPO seeds',
        columns: ['Weighting', 'CAGR', 'MaxDD', 'Calmar', '2018 grind DD', '2020 crash DD', '2022H1 grind DD'],
        rows: [
          ['TN+OA 50-50 - the DEPLOYED baseline', '27.14%', '-16.42%', '1.65', '-11.26%', '-1.98%', '-9.07%'],
          ['plus IPO 10% (45/45/10)', '27.72%', '-14.44%', '1.92', '-10.61%', '-1.75%', '-8.86%'],
          ['plus IPO 20% (40/40/20) - the candidate', '28.27%', '-12.79%', '2.21', '-9.82%', '-2.14%', '-8.75%'],
          ['plus IPO 33%', '28.91%', '-13.05%', '2.21', '-9.30%', '-2.67%', '-8.60%'],
          ['CASH-NULL at 10%', '24.91%', '-14.49%', '1.72', '-9.94%', '-1.70%', '-8.08%'],
          ['CASH-NULL at 20%', '22.67%', '-12.64%', '1.79', '-8.65%', '-1.43%', '-7.07%'],
        ],
        highlightRows: [2],
      },
      {
        title: 'Against the incumbent third-sleeve candidate - gold (research/147) - on the window BOTH can be measured on (2015-01 to 2026-09)',
        columns: ['Blend', 'CAGR', 'MaxDD', 'Calmar', 'Read'],
        rows: [
          ['TN+OA 50-50 baseline', '29.63%', '-16.10%', '1.84', 'n/a'],
          ['plus GOLDBEES 10% (research/147)', '28.36%', '-13.37%', '2.12', 'Gold buys Calmar by LOWERING return'],
          ['plus GOLDBEES 20%', '27.03%', '-10.54%', '2.56', 'Same shape, more of it'],
          ['plus IPO Base 10%', '30.84%', '-14.24%', '2.17', 'Buys Calmar while RAISING return'],
          ['plus IPO Base 20%', '32.01%', '-12.76%', '2.51', 'Best CAGR-and-Calmar combination of the three-sleeve set'],
          ['4-SLEEVE 40 OA / 40 TN / 10 gold / 10 IPO', '29.05%', '-11.55%', '2.52', 'EXPLORATORY - a single cell, not a swept result'],
        ],
        highlightRows: [4],
      },
      {
        title: 'Capacity - position as a share of the held name 20-day median traded value (held names: median Rs 17.0 cr a day)',
        columns: ['Portfolio', 'Sleeve at 10%', 'Position', 'Median % of daily traded value', 'p90', 'p99'],
        rows: [
          ['Rs 1 cr', 'Rs 10 L', 'Rs 1.88 L', '0.11%', '0.29%', '0.37%'],
          ['Rs 5 cr', 'Rs 50 L', 'Rs 9.38 L', '0.55%', '1.45%', '1.86%'],
          ['Rs 10 cr', 'Rs 1 cr', 'Rs 18.75 L', '1.10%', '2.91%', '3.73%'],
          ['Rs 50 cr', 'Rs 5 cr', 'Rs 93.75 L', '5.52%', '14.54%', '18.63%'],
        ],
      },
      {
        title: 'Year by year - the flat years are the story (median across 30 seeds, after tax; intra-year max drawdown in brackets)',
        columns: ['Year', 'IPO Base sleeve', 'Open Alpha', 'True North', 'TN+OA 50-50', 'TN+OA+IPO 40/40/20'],
        rows: [
          ['2006', '+50.3 (-11.4)', '+19.0 (-12.5)', '+9.0 (-11.6)', '+16.1 (-3.8)', '+22.5 (-3.9)'],
          ['2007', '+97.2 (-12.1)', '+143.0 (-13.4)', '+53.6 (-15.6)', '+89.7 (-7.1)', '+87.5 (-6.1)'],
          ['2008', '+4.2 (-12.5)', '-9.1 (-14.9)', '-14.1 (-19.6)', '+0.6 (-2.6)', '+1.0 (-1.7)'],
          ['2009', '+3.1 (-3.8)', '+58.0 (-11.2)', '+53.8 (-9.9)', '+55.8 (-4.0)', '+43.9 (-3.0)'],
          ['2010', '+28.5 (-9.0)', '+20.0 (-9.1)', '+4.1 (-15.4)', '+21.1 (-3.9)', '+21.5 (-2.7)'],
          ['2011', '+13.3 (-6.9)', '-1.6 (-11.3)', '-11.7 (-13.9)', '-0.8 (-4.5)', '+1.8 (-3.4)'],
          ['2012', '+2.5 (-3.9)', '+13.2 (-7.8)', '+20.6 (-6.8)', '+15.3 (-4.7)', '+12.7 (-3.7)'],
          ['2013', '+5.1 (0.0) CASH ONLY', '+15.0 (-7.5)', '+0.8 (-10.8)', '+10.6 (-1.9)', '+9.4 (-1.4)'],
          ['2014', '+5.0 (0.0) CASH ONLY', '+84.3 (-10.0)', '+46.3 (-11.9)', '+61.4 (-4.9)', '+48.6 (-3.9)'],
          ['2015', '+13.9 (-6.7)', '+22.5 (-10.9)', '-6.4 (-12.4)', '+5.2 (-7.1)', '+6.9 (-5.4)'],
          ['2016', '+43.8 (-9.3)', '+17.2 (-11.7)', '+31.0 (-6.2)', '+27.7 (-1.9)', '+33.1 (-1.5)'],
          ['2017', '+58.2 (-7.5)', '+108.0 (-17.2)', '+32.2 (-8.5)', '+69.9 (-0.5)', '+66.0 (-0.3)'],
          ['2018', '-1.9 (-18.3)', '-11.5 (-18.5)', '-5.7 (-22.7)', '-10.1 (-11.3)', '-7.1 (-9.8)'],
          ['2019', '+14.2 (-8.0)', '+9.1 (-9.2)', '-2.1 (-8.6)', '+4.6 (-6.4)', '+6.5 (-5.2)'],
          ['2020', '+87.7 (-7.3)', '+123.2 (-11.5)', '+62.3 (-6.4)', '+81.7 (-2.0)', '+80.0 (-2.1)'],
          ['2021', '+41.2 (-12.2)', '+178.2 (-7.6)', '+58.6 (-12.5)', '+107.0 (0.0)', '+97.1 (0.0)'],
          ['2022', '+14.3 (-15.1)', '+11.4 (-18.8)', '+7.4 (-9.1)', '+10.1 (-9.1)', '+12.7 (-8.8)'],
          ['2023', '+72.4 (-16.3)', '+54.0 (-11.3)', '+37.4 (-9.0)', '+55.7 (-1.7)', '+59.6 (-3.2)'],
          ['2024', '+78.8 (-13.8)', '+77.3 (-12.0)', '+18.5 (-18.2)', '+23.4 (-8.7)', '+30.8 (-6.8)'],
          ['2025', '-1.8 (-20.2)', '+20.1 (-19.0)', '+5.4 (-10.8)', '+19.1 (-4.0)', '+16.8 (-4.5)'],
          ['2026 YTD', '+67.4 (-9.2)', '+35.2 (-21.8)', '+5.2 (-6.2)', '+16.3 (-5.5)', '+25.6 (-3.0)'],
          ['FULL PERIOD CAGR / MaxDD / Calmar', '31.48% / -20.88% / 1.51', '33.57% / -26.38% / 1.27', '19.48% / -24.98% / 0.78', '27.14% / -16.42% / 1.65', '28.27% / -12.79% / 2.21'],
        ],
      },
    ],
    results: {
      metrics: [
        { label: 'Listing-table validation', value: '48/48 recall, 0/12 leaks', tone: 'pos', hint: 'built and tested before any backtest ran' },
        { label: 'Standalone (30 seeds, after tax)', value: '31.03% / -20.88% / Calmar 1.50', tone: 'pos', hint: 'worst seed 28.82%' },
        { label: 'Cells clearing the expectancy gate', value: '207/256 and 383/384', tone: 'pos', hint: 'a plateau, not a peak' },
        { label: 'Edge over a date-matched null', value: '+0.96pp per trade, 29/30 seeds', tone: 'pos' },
        { label: 'Correlation to the deployed legs', value: '0.16 OA / 0.18 TN daily', tone: 'pos', hint: 'lower than OA-to-TN at 0.42' },
        { label: 'Blend at 20% weight', value: '+1.13pp CAGR, -3.63pp DD, +0.56 Calmar', tone: 'pos' },
        { label: 'Fill dependence', value: '-14.08pp CAGR if filled at the close', tone: 'neg', hint: 'the whole edge is the entry price' },
        { label: 'Dead years', value: '2013 and 2014 earned only the cash yield', tone: 'neg', hint: 'the IPO pipeline was shut' },
      ],
      tables: [],
      charts: [
        { src: '/app/ipo_base_research153.png', caption: 'Growth of Rs 100 (log) and drawdown, 2006 to Sep-2026, after Indian tax and 25 bps a side: the IPO Base sleeve (median of 30 selection seeds) against the deployed TN+OA 50-50 book, the 40/40/20 candidate blend, and NIFTY 50 / Midcap 150 / Smallcap 250. The long flat stretch from 2009 to 2015 is real and is the honest cost of this strategy - the Indian IPO pipeline supplied 8 to 17 usable listings a year in 2012-2014 against 80 to 182 in 2021-2025.' },
      ],
    },
    winners: [
      {
        config: 'ADOPTED CANDIDATE - IPO-Base MID at 10-20% of the book, pending Arun sign-off and a paper soak',
        summary: 'Every leg of the pre-registered complement bar is met with room, and the choice between age bands was made BY the bar rather than after seeing the outcome: the narrow 3-month sleeve improves Calmar more per unit of weight but costs a hair of CAGR (27.08 against a 27.14 baseline at 10%), so it fails the greater-than-or-equal-CAGR leg. The 6-month band passes both legs cleanly.',
        metrics: [
          { k: 'Standalone, 30 seeds, after tax', v: '31.03% CAGR [28.82..33.44], -20.88% DD, Calmar 1.50, 32.6 trades a year, 49.0% win rate, +5.39% mean per trade' },
          { k: 'Blend value at 20% weight', v: '+1.13pp CAGR, -3.63pp drawdown, +0.56 Calmar vs the deployed TN+OA pair' },
          { k: 'Beats the cash-null', v: 'yes - +5.60pp of CAGR for +0.15pp of drawdown at the same 20% weight' },
          { k: 'Capacity', v: 'comfortable to about a Rs 10 cr portfolio (1.1% of daily traded value); starts to bind near Rs 50 cr' },
        ],
        rejected: [
          'The market gate (NIFTYBEES below its 200-DMA) - costs 5.23pp of CAGR and loses on 0 of 30 paired seeds',
          'The site Trail 30-week (SMA-150) - the worst exit tested, Calmar 0.49 against 0.99 for SMA-20',
          'The site Breakout close entry - costs 14.08pp of CAGR against a buy-stop at the pivot, on 0 of 30 seeds',
          'Every relative-strength filter - the strict 252-day score is mathematically undefined for these names and short-window substitutes starve the book',
          'The WIDE 24-month band - highest CAGR at 35.99%, but at that age the screen is no longer an IPO base and its profile converges on Open Alpha (33.8% / -27.3%), which we already run',
        ],
      },
    ],
    caveats: [
      'THE ENTIRE EDGE LIVES IN THE ENTRY PRICE. A buy-stop filled at the pivot returns 31.03% CAGR; filling at the signal-day close returns 16.98%, and the close loses on 30 of 30 paired seeds. Live, this requires a working buy-stop or GTT at the pivot on every candidate every day. This is the same lesson research/142 learned the hard way (x536 against x14.4).',
      'NO REPLICATION GATE WAS RUN. The site IPO-Base panel dials and its published headline numbers were never legible, so every setting here is a swept axis rather than a match. Two of the dials the site does expose - Trail 30-week and Breakout close - are the worst settings we tested, so their published figures cannot be assumed comparable to these. The engine is built so a replication gate is a one-command run when the dials arrive.',
      'MULTI-YEAR DEAD ZONES ARE STRUCTURAL. 2013 and 2014 returned exactly the idle-cash yield because the book took no trades at all. This strategy is a function of the IPO pipeline, which is a policy and market-cycle variable rather than a price series, and an operator must be prepared to do nothing for years at a time.',
      'REGIME CONCENTRATION. 2020-2026 supplies a large share of both the trades and the return; the site own window (2020-2025) prints 44.57% CAGR against 31.03% for the full period. Expect the forward number nearer the full-period figure, or below it.',
      'SURVIVORSHIP is measured and small inside the database but is not zero and biases upward. The database RETAINS dead series rather than purging them - 9.0% of all symbols end more than 90 days early, and a post-2010 sample of those ended a median -42.9% from their peak. Inside the traded cohort, 41 of 334 names later go stale; they are 5.1% of trades and returned a mean +2.70% against +5.52% for the rest, a drag the backtest does pay. The unmeasurable residual is IPOs that died before ever being onboarded to Kite at all.',
      'THE LISTING TABLE IS VALIDATED ON A 60-NAME TEST SET (48 known IPOs plus 12 known onboardings), not on all 1,293 accepted listings. A systematic error in the untested remainder is possible. Separately, the 2025 cohort apparent 48.6% death rate is a FEED-FRESHNESS artefact rather than delisting - those series stop in identical batches on 2026-02-17, 2026-05-07 and 2026-05-15, i.e. symbols dropped from the nightly refresh list.',
      'MULTIPLE TESTING: 680 cells. The plateau is broad and the MEDIAN cell clears the expectancy gate, not merely the best one, but the headline cell numbers should still be discounted for best-of-680.',
      'The four-sleeve result (40 OA / 40 TN / 10 gold / 10 IPO at 29.05% / -11.55% / Calmar 2.52) is a SINGLE EXPLORATORY CELL on a favourable 2015+ window, not a swept finding. It shares the dated review already registered for the research/152 four-sleeve question and must be run with a GOLD-ONLY null, because the real question is what this adds ON TOP OF gold.',
      'Point-in-time discipline: the Rs 5 cr liquidity floor and the RS score are computed causally at t-1, but the symbol UNIVERSE is today Kite coverage applied to the past.',
      'NOTHING WAS DEPLOYED. No live engine, crontab, sizing or spec was touched by this study.',
    ],
    githubLinks: [{ label: 'research/153 (repo)', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/153_ipo_base' }],
    projectPaths: [
      'research\\153_ipo_base\\IPO_BASE_BREAKOUT_DAILY_SWEEP_STATUS.md',
      'research\\153_ipo_base\\results\\RESULTS.md',
    ],
  },
"""

marker = "\n];\n\nexport function getStudy"
i = s.rindex(marker)
s = s[:i] + "\n" + ENTRY + s[i + 1:]
P.write_text(s, encoding="utf-8")
print("inserted; file now", len(s.splitlines()), "lines")
