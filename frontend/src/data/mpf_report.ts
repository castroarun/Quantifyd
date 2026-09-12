/**
 * THE MOMENTUM PORTFOLIO REPORT — the narrative half of /app/mpf-report.
 *
 * DIVISION OF LABOUR, and it is deliberate:
 *   - every NUMBER on the page (CAGR / drawdown / Calmar / growth of 100 / average invested,
 *     the year tables, the correlations, the blend row, the entry surface, the null control,
 *     the gate bake-off) is COMPUTED by research/_utilities/mpf_report_build.py into
 *     static/app/mpf_report.json and fetched by the page. None of it is typed here.
 *   - every study figure quoted in PROSE below carries a `source` naming the file it came
 *     from, so any line on the page can be audited back to an evidence file in one step.
 *
 * BINDING EDITORIAL RULES ON THIS PAGE (Arun, 11-Sep-2026, each raised more than once):
 *   - POST-TAX ONLY. Any figure that exists only pre-tax is omitted and the omission noted.
 *   - Every table states WHICH SYSTEMS, WHICH WINDOW, WHICH BASIS.
 *   - Systems are NAMED, never versioned: "Open Alpha · Base Age", "Open Alpha · ATH + VIX".
 *     They are different signals, not revisions of one another.
 *   - Test period, average invested and average in cash go ON the table, with the cash-yield
 *     assumption stated as a caveat.
 *   - Questions are answered in Q&A form.
 *
 * Full provenance, including the three-window problem and every place two sources disagreed,
 * is in research/160_quality_growth_near_ath/MPF_REPORT_PAGE_BUILD_STATUS.md §8.
 *
 * The Strategies register (frontend/src/data/strategies.ts) remains the register of record.
 * Nothing here overrides it, and nothing here was changed in it.
 */

export type SystemKind = 'live' | 'candidate' | 'research' | 'paper';

export interface KV {
  k: string;
  v: string;
}

export interface RejectedRow {
  name: string;
  numbers: string;
  why: string;
}

export interface SystemReport {
  /** must match a key in mpf_report.json headline.rows / window2018.rows */
  key: string;
  name: string;
  kind: SystemKind;
  statusLabel: string;
  accent: 'gold' | 'green' | 'coral' | 'purple';
  /** size as traded, from the register — or why there is none */
  size: string;
  /** the one-line rule */
  rule: string;
  /** which window this system appears on */
  windowNote: string;
  /** an alert line shown at the top of the section, when there is one */
  alert?: string;
  /** chart key in mpf_report.json charts.heat */
  heatKey?: string;

  rules: KV[];
  mechanics: KV[];
  evidence: KV[];
  distinctiveTitle: string;
  distinctive: string;
  caveats: string[];
  links: { label: string; href: string; kind: 'study' | 'status' | 'dashboard' | 'register' }[];
  rejected?: { title: string; caption: string; rows: RejectedRow[] };
}

export const REPORT_DATE = '11 Sep 2026';

export const HIGHLIGHTS: string[] = [
  'No single book reaches 25% after tax over the full twenty years, and the year table shows why a blend might: True North is the only green bar in 2008 and 2011, IPO Base is the tall one in 2020, and Base Age owns 2017, 2021 and 2023.',
  'True North holds cash 57% of the time and still produces one of the two best returns. That is the gate doing the work, and it is a stronger result than the CAGR alone says.',
  'IPO Base is the only genuine diversifier in the book — loosely coupled to everything, including the index — which is why it survives the halving of its published number and still earns a place.',
];

export const SYSTEMS: SystemReport[] = [
  // ------------------------------------------------------------------ TRUE NORTH
  {
    key: 'True North',
    name: 'True North',
    kind: 'live',
    statusLabel: 'LIVE · real money',
    accent: 'gold',
    size: '₹7,69,000',
    windowNote: 'Measured on the full 20.4-year window and again on the 2018 window.',
    heatKey: 'True North',
    rule:
      'Rank the Nifty 200 by blended momentum and hold the top 8, only while the weekly index gate holds; Donchian trailing stop per name, month-end rebalance with a top-22 buffer.',
    rules: [
      { k: 'Universe', v: 'NSE Nifty 200, point-in-time' },
      { k: 'Selection', v: 'Blended 6-month + 12-month relative strength against NIFTYBEES, top 8, with a top-22 buffer to suppress churn' },
      { k: 'Gate', v: 'Weekly index gate — the whole drawdown story sits here, not in the stop' },
      { k: 'Stop', v: 'Daily Donchian trailing stop per name' },
      { k: 'Rebalance', v: 'Month-end; exits run FIRST, then the refill. Rotate-only — winners are never trimmed' },
      { k: 'Top-ups', v: 'Immediate equal top-up on a deposit; broker-quantity assert before any sell' },
      { k: 'Idle cash', v: 'Swept into CASHIETF above a 3% reserve, released before the rebalance buys' },
    ],
    mechanics: [
      { k: 'Trigger', v: 'Month-end ranking on closing prices. There is no intraday trigger and no breakout event' },
      { k: 'Fill', v: 'Closing prices only. The engine holds no opens and no highs, so a decide-from-the-close / pay-from-the-open mismatch is STRUCTURALLY IMPOSSIBLE. That is why the 11-Sep-2026 audit found it clean' },
      { k: 'Exit', v: 'Falls out of the top 22 at a month end, or the gate flips the whole book to cash' },
      { k: 'Stop', v: '15-day Donchian low on the close, per name, checked daily' },
      { k: 'Slots', v: '8, equal weight' },
      { k: 'Cadence', v: 'Month-end rebalance · daily stop check · weekly (Friday) gate check' },
      { k: 'Liquidity', v: 'Point-in-time traded-value universe, ETFs excluded' },
      { k: 'Gate detail', v: 'NIFTYBEES below its 100-day SMA on a Friday close → LIQUIDATE the whole book to cash. It does not merely block new buys' },
    ],
    evidence: [
      { k: 'Own study window', v: '20.9% after-tax CAGR at −23.7%, Calmar 0.88, on 2012 → 3-Sep-2026 — the study’s primary window WA (that study credited idle cash at 6.5%, so it reads well above the rows on this page, which credit 5.2% — the arbitrage-fund rate after tax). Source: research/144_truenorth_reassessment/results/RESULTS.md §(b)' },
      { k: 'Robustness band', v: '12 rebalance-day offsets: median 20.7% [14.9 .. 25.1], drawdown median −25.1%, WORST OFFSET −28.3%' },
      { k: 'Both sub-windows', v: 'W1 (2016-06 → 2019-12) 13.6% median · W2 (2020 → now) 27.3% median — positive in both' },
      { k: 'Cells disclosed', v: '71-cell gate bake-off + 27-cell action/frequency sweep + 240-run slots × exits sweep, all ranked after tax' },
      { k: 'Null / control', v: 'The bake-off is itself the control: 5 gate series × 14 constructions plus no-gate. The inherited gate won its own bake-off' },
      { k: 'Entry audit', v: 'CLEAN — 11-Sep-2026. Every live dial matches the study: 8 slots, 22 buffer, 200 universe, NIFTYBEES 100-SMA weekly gate to cash, Donchian-15, 0.3% round trip, 6.5% cash' },
    ],
    distinctiveTitle: 'The 100-SMA gate is the whole risk story',
    distinctive:
      'Remove the gate and the same book returns MORE — 23.9% against 20.9% — at a −46.5% fall instead of −23.7%, so Calmar collapses from 0.88 to 0.51. The gate is not a return filter that happens to help in crashes; it IS the product. And it had to be that gate: 71 cells of alternatives were tested and no other series and no other construction beat NIFTYBEES-SMA100 on drawdown-constrained return. Only a NIFTYBEES-based gate protects 2008 at all, because the index series in the database begin in 2011. The cash that gate parks is also why this book sits out 57% of the time and still finishes near the top.',
    caveats: [
      'TRUE NORTH IS A SINGLE PATH while Open Alpha · Base Age and IPO Base are their studies’ drawn curves and Quality Summit is a 12-offset median. That is not identical treatment, and a like-for-like ensemble re-run of True North is OWED before the columns are compared to the decimal.',
      'ITS CASH YIELD WAS 6.5% UNTIL 12-SEP-2026 — it is 5.2% now, like every other book here. research/144 assumed 6.5%, and this book holds cash 57% of the time, so the assumption alone was worth about a point a year to it. research/163 re-ran the identical cell on research/144’s own engine twice on 12-Sep-2026, each time reproducing the published curve at the old yield first: 6.5% → 5.0% cost 0.97 points of CAGR (19.53% → 18.56% on the 20.4-year window) and 1.3 points of drawdown, then 5.0% → 5.2% gave 0.13 back (18.56% → 18.69%), which is what (1 − 43% invested) × 0.2 points predicts. Those are the numbers the tables on this page now show; the study’s own page still quotes the 6.5% figures. 5.2% is the ARBITRAGE-FUND rate after 20% short-term tax at 2025-26 cash-futures spreads — and this book, sitting in cash 57% of the time, is the one with the most riding on where that cash actually sits. Moving it into an arbitrage fund with a liquid-ETF buffer is an OWED OPERATIONAL ACTION, not a modelled one.',
      'Survivorship: market_data.db keeps only 102 stopped series in 2,158 over eleven years, fewer than NSE actually delisted. Pressure is upward on every arm here, benchmarks included.',
      'market_data.db is NOT retroactively split-adjusted. Smaller exposure for a Nifty-200 momentum book than for an all-time-high screen, but not zero.',
      'Residual phantom rows (OHLC = previous close, volume 0) exist on 2025-03-18 and 2024-01-15 for small-caps. Benign for a Nifty-200 system; flagged for a future purge.',
      'NOT re-verified in the entry audit: survivorship, the relative-strength formula, and the line-by-line implementation. Only the buy/sell mechanics were.',
    ],
    links: [
      { label: 'Full study — True North reassessment (research/144)', href: '/app/backtest/truenorth-reassessment-research144', kind: 'study' },
      { label: 'STATUS — research/144_truenorth_reassessment/TRUENORTH_MOMENTUM_DAILY_SWEEP_STATUS.md', href: '/app/backtest/truenorth-reassessment-research144', kind: 'status' },
      { label: 'Live dashboard — the True North book', href: '/portfolio?tab=tn', kind: 'dashboard' },
      { label: 'Register entry — Strategies index', href: '/strategies', kind: 'register' },
    ],
  },

  // ------------------------------------------------------- OPEN ALPHA · BASE AGE
  {
    key: 'Open Alpha · Base Age',
    name: 'Open Alpha · Base Age',
    kind: 'candidate',
    statusLabel: 'CANDIDATE · not deployed',
    accent: 'green',
    size:
      'No size — never papered, no order has ever been placed on this signal. The LIVE Open Alpha book (₹4,46,348 in RA6610) runs the OLD published spec and ITS BUYING IS PAUSED.',
    windowNote: 'Measured on the full 20.4-year window and again on the 2018 window.',
    heatKey: 'Open Alpha · Base Age',
    alert:
      'BUYING ON THE LIVE OPEN ALPHA BOOK IS PAUSED, since 11-Sep-2026, for two separate reasons. (1) services/oa_entry.py arms the INVERSE of the designed signal: it selects close < pivot — names that have NOT broken out — and rests a stop at the high, so it can never pick a name on its breakout day. (2) Even corrected, the published rule is not placeable at all: the study decided from the close and paid from the same day’s open. Both entry crons are commented out (backup /tmp/mpf/ct.bak.20260911-111828). Exits, stops, the trail, marks and reconcile all still run. DO NOT restore the entry jobs before BOTH the inverted condition and the stale ETF filter in that file are fixed.',
    rule:
      'Buy the first close above an all-time high that is at least 60 bars old and at least 20% above the base beneath it; fill at the next open; trail out on a SuperTrend(14,4) close, no hard stop.',
    rules: [
      { k: 'Universe', v: 'All NSE cash dailies; funds excluded by instrument NAME, not by guessing from the ticker' },
      { k: 'Signal', v: 'The first close above the prior all-time-high close — the running maximum of closes strictly before today' },
      { k: 'Base age', v: 'At least 60 trading bars between the bar that set the previous high and today. 40 bars is indistinguishable: this is a plateau, not a peak' },
      { k: 'Base depth', v: 'The stock must have fallen at least 20% below that previous high somewhere inside the gap' },
      { k: 'Volume', v: 'NONE. Tested at 2×, 3× and 5× the prior 20-bar median and REJECTED' },
      { k: 'Shape', v: 'NONE. The rounding-base / saucer requirement was tested and is strongly negative — 5.6 trades a year' },
      { k: 'Book', v: '16 slots at 6.25% of NAV, CNC, random draw among same-day signals when there are more signals than free slots' },
      { k: 'Re-arm', v: '60 bars after an event fires, so one run of new highs does not emit a cluster' },
    ],
    mechanics: [
      { k: 'Trigger', v: 'Decided on the CLOSE of the breakout day' },
      { k: 'Fill', v: 'The NEXT day’s open, on BOTH legs. No second condition selects the trades that worked — the engine was read line by line for the same defect and is clean' },
      { k: 'Exit', v: 'SuperTrend(14,4) on the close, filled at the next open' },
      { k: 'Stop', v: 'NONE. No hard stop — the trail is the only exit' },
      { k: 'Slots', v: '16 at 6.25%' },
      { k: 'Cadence', v: 'Event-driven, about 32 trades a year' },
      { k: 'Liquidity', v: '20-day median traded value ≥ ₹2 cr; survives the stricter ₹5 cr floor at 20.62%' },
      { k: 'Gate', v: 'NONE — no market gate and no VIX gate. The VIX gate belongs to the ATH + VIX variant below, not to this one' },
    ],
    evidence: [
      { k: 'Own study window', v: '21.26% after-tax CAGR at −34.80%, Calmar 0.618, on 3-Jan-2005 → 11-Sep-2026, 30-seed median (that study credited idle cash at 5.5%, so it reads a little above the rows on this page, which credit 5.2% — the arbitrage-fund rate after tax). Source: research/161_ath_base_age_breakout/results/RESULTS.md §1' },
      { k: 'Robustness band', v: '30 seeds: 19.87 .. 21.89%. WORST PATH 19.87% — a whisker BELOW the pre-registered 20% floor. On a median reading it passes; anyone who meant "every path clears 20%" should read it as a fail by 0.13pp' },
      { k: 'Both sub-windows', v: 'pre-2016 19.33% at −32.45% · 2016+ 23.24% at −32.43% — index-beating in both' },
      { k: 'Null control', v: 'A date-matched random-entry control on the same days returns 12.11%. The rule beats it by +9.15pp' },
      { k: 'Trade profile', v: '49.2% win rate, +37.2% average win against −11.6% average loss, expectancy +12.4% a trade, 31.7 trades a year, longest losing streak 14' },
      { k: 'Cost ladder', v: '25 / 40 / 60 bps → 21.26 / 20.34 / 19.35% — a shallow slope, this is a low-turnover book' },
      { k: 'Idle cash', v: 'Credited at 5.2% post-tax on this page — the arbitrage-fund rate (research/161’s own study used 5.5%; re-running it at 5.0% cost 0.34 points on the 20.4-year window, and the move back up to 5.2% returned 0.06, matching (1 − invested) × 0.2). Measured 72.9% invested over 30 seeds, band 72.7–73.0%, so about a quarter of the book earns the sweep. With no cash carry at all the study returned 19.27% against 21.26%' },
      { k: 'Cells disclosed', v: '864 (6 base ages × 3 depths × 4 volume levels × 2 shapes × 6 exits), 810 completed. The winner is reported as a plateau, not a spike' },
    ],
    distinctiveTitle: 'The exit is worth +11.85pp; base age is worth +2.6pp and −6.8pp of drawdown',
    distinctive:
      'Swapping Open Alpha’s own 15-day-SMA-plus-8%-stop pair for a SuperTrend(14,4) trail on the SAME plain all-time-high entries is worth +11.85 points a year — 6.81% becomes 18.66%. Only then does the entry filter earn: requiring the previous high to be at least 60 bars old and the base at least 20% deep adds a further +2.6 points and takes 6.8 points OFF the drawdown, 18.66% → 21.26% and −41.59% → −34.80%. Anyone reading this as "base age is the discovery" has it backwards: the trail is the big win, and age is a real, smaller second improvement that mostly buys drawdown. Volume confirmation is the mirror image — it makes each trade better (expectancy +12.4% → +15.1% at long base ages) and the book worse (down to 14.8% at 5×), because it throws away 40 to 60% of the opportunities a 16-slot book needs. A good filter for a discretionary trader picking a handful of names and a bad one for a mechanical book; both halves of that sentence are true at once.',
    caveats: [
      'OUTLIER DEPENDENCE IS SEVERE. Removing ten trades out of 687 collapses compounded growth by a factor of 133,000. This book makes its money in a few names, and the better entry did not fix it.',
      'AVERAGE INVESTED WAS NEVER MEASURED for this book. research/159’s full_period.py records it as None with the note "to be measured in its own harness", and research/161 saved no invested series. The handover doc asserts roughly 67% but no file on disk carries it, so this page prints "not measured" rather than a number it cannot point at.',
      'NOT A DEPLOYMENT RECOMMENDATION. No paper book, no live engine, no order has ever been placed on this spec. The portfolio-fit test against the honest Open Alpha curve is the next gate and it is dated 26-Sep-2026.',
      'Survivorship is sharp for a pattern that requires a new all-time high — the names that never came back are exactly the ones missing from the universe.',
      'market_data.db is NOT retroactively split-adjusted. The all-time high is computed only from bars after the last day-over-day fall worse than −35%, so genuine highs on names that split are discarded along with the fakes.',
      '864 cells disclosed; discount the winner for multiple testing. The plateau is the finding — the peak is only where it happens to be highest.',
      'The 2018-window column is KINDER to this book than the 20-year one, because 2018 onward excludes the stretch where it returns 19.33%. Read both, never one.',
    ],
    links: [
      { label: 'Full study — base-age breakout (research/161)', href: '/app/backtest/ath-base-age-breakout-research161', kind: 'study' },
      { label: 'STATUS — research/161_ath_base_age_breakout/ATH_BASE_AGE_VOLUME_BREAKOUT_DAILY_SWEEP_STATUS.md', href: '/app/backtest/ath-base-age-breakout-research161', kind: 'status' },
      { label: 'Live dashboard — the Open Alpha book (OLD spec, buying paused)', href: '/portfolio?tab=oa', kind: 'dashboard' },
      { label: 'Register entry — Strategies index', href: '/strategies', kind: 'register' },
    ],
    rejected: {
      title: 'The other two Open Alphas — evidence only, never candidates',
      caption:
        'Neither appears in the headline table, deliberately. One is superseded on every axis and exists only as the correction’s evidence; the other cannot be placed at all. They are here so the archive is not lost.',
      rows: [
        {
          name: 'Open Alpha · ATH + VIX (research/159) — this session’s repair of the published rule',
          numbers: '19.2% / −34.1% / Calmar 0.56 — AFTER TAX, on the 2016-2026 window ONLY',
          why:
            'Buy at the breakout close, 75-day trail, and enter only when INDIA VIX is above its own one-year 70th percentile. It is a very large improvement on what the live book actually runs, and it is beaten by Base Age on every axis. Its window is 2016-2026 and cannot be extended, because INDIA VIX does not exist before 2015 in our data — which also means the gate is fitted on its only window and can never be tested out of sample. It is the least-tested component anywhere on this page. Keep it as the correction’s evidence; do not present it as a candidate.',
        },
        {
          name: 'Open Alpha (published, research/142) — the spec the live book actually runs',
          numbers: '40.8% published → −1.7% once filled the way an order actually fills',
          why:
            'The engine counted a trade only when the day CLOSED above the breakout level, but paid the price available at that day’s open. Buying on this book is paused. Note also that research/142’s published figures still carry the 221-fund universe contamination in their recent years, and a re-run is owed before they are cited again.',
        },
        {
          name: 'Same-day abort on a failed breakout',
          numbers: '−21.2% with slot recycling, −13.3% without',
          why:
            'Arun’s proposal to cut a breakout that fails intraday. The toll is paid hundreds of times a year and it is far larger than the losses it avoids.',
        },
      ],
    },
  },

  // ------------------------------------------------------------------ QUALITY SUMMIT
  {
    key: 'Quality Summit',
    name: 'Quality Summit',
    kind: 'research',
    statusLabel: 'RESEARCH ONLY · not papered',
    accent: 'coral',
    size: 'No size. Not in the Strategies register, not papered, no order has ever been placed.',
    windowNote:
      'Appears ONLY on the 2018 window. It cannot exist earlier: point-in-time fundamentals need four filed fiscal years and Screener history begins FY2015.',
    heatKey: 'Quality Summit',
    rule:
      'Hold the 15 highest relative-strength names that are within 10% of their all-time-high close AND pass a loose point-in-time quality screen; rebalance monthly, fill at the next open, no exit rule.',
    rules: [
      { k: 'Universe', v: '2,158 NSE cash names, funds excluded by instrument NAME not by ticker' },
      { k: 'Price state', v: 'Close ≥ 0.90 × the causal all-time-high close (the running maximum restarts on any one-day −40% split-shaped collapse)' },
      { k: 'Quality screen', v: 'No negative sales or net profit in the last three FILED years · three-year average ROE > 15 · ROCE > 15 (lenders judged on ROE alone) · three-year sales AND profit growth > 10 · market cap > ₹1,000 cr. NO debt-to-equity test' },
      { k: 'Point-in-time', v: 'Only fiscal years FILED by the decision date — Indian FY end plus a four-month filing lag. Screener panel, 2,131 of 2,158 names covered' },
      { k: 'Liquidity', v: '20-day median traded value ≥ ₹2 cr at the decision close' },
      { k: 'Selection', v: 'Relative-strength rank among the qualifying set, top 15 equal weight, hysteresis buffer 1.5×N' },
      { k: 'Exit', v: 'NONE. Hold while the name stays in the top of the ranked set; it drops out at a rebalance' },
      { k: 'What Arun actually types', v: 'This is his own screener.in query with TWO dials relaxed — growth from 20% to 10%, and the debt-to-equity test removed. Left as written those two dials cost 11.8 points of CAGR' },
    ],
    mechanics: [
      { k: 'Trigger', v: 'A monthly rebalance date, decided on the close. There is no event and no breakout' },
      { k: 'Fill', v: 'The next open. The engine never reads a high or a low, so the trigger/fill trap is structurally impossible' },
      { k: 'Exit', v: 'Rotation only — a name leaves when it falls out of the ranked set at a rebalance' },
      { k: 'Stop', v: 'NONE. 285 exit-by-gate cells were run and no exit beat simply holding' },
      { k: 'Slots', v: '15, equal weight, about 91% invested' },
      { k: 'Cadence', v: 'Monthly, about 61 trades a year' },
      { k: 'Liquidity', v: '₹2 cr 20-day median traded value' },
      { k: 'Gate', v: 'NONE. A NIFTY 200-SMA gate raises Calmar only by parking the book in cash, not by improving it' },
    ],
    evidence: [
      { k: 'Own study window', v: '21.19% after-tax CAGR at −37.1%, Calmar 0.58, 91% invested, on 1-Aug-2018 → 10-Sep-2026 (that study credited idle cash at 5.0%; the rows on this page credit 5.2%, worth about +0.02 a year to a book this heavily invested). Source: research/160_quality_growth_near_ath/results/RESULTS.md' },
      { k: 'Robustness band', v: '12 rebalance offsets: median 21.05% on the common window, range 17.45 .. 23.50%. WORST PATH 17.45%' },
      { k: 'Both sub-windows', v: 'W1 (2018-08 → 2022-06, contains the 2020 crash) 20.11% · W2 (2022-07 → 2026-09, contains the smallcap boom) 20.95% — unusually stable' },
      { k: 'Null control', v: 'Picking names AT RANDOM from the same liquid near-the-high universe returns 14.82%. The relative-strength RANKING is worth +7.70pp; the quality screen is worth −1.32pp' },
      { k: 'Trade profile', v: '46% win rate, +27.9% average win against −10.8% average loss, about 61 trades a year' },
      { k: 'Cost ladder', v: '19.10% at 60 bps' },
      { k: 'Cells disclosed', v: '588 (G1 133 · G2 413 · G3 42, plus 12 paired A-vs-B comparisons)' },
      { k: 'The pre-registered bar', v: 'TWELVE fundamental masks tested against +2pp CAGR or +0.15 Calmar on 8 of 12 paired offsets. ZERO passed. The best of them — this one — buys +0.106 Calmar and gives back 1.32 points of CAGR' },
      { k: 'Blend value', v: 'Added to the deployed pair at 10 / 20 / 33%, Calmar falls 2.37 → 2.23 → 2.02 → 1.69, monotonically. Holding plain CASH in its place beats it on 360 of 360 paths' },
    ],
    distinctiveTitle:
      'The quality screen costs 1.3 points and removes 11 points of drawdown — and the screen as written starves the book to 10.8%',
    distinctive:
      'Against the IDENTICAL book with no screen at all on the same universe, the loose quality screen gives up 1.32 points of return a year and takes about 11 points off the drawdown — a real trade, and it still fails the pre-registered bar of +0.15 Calmar. The screen as Arun actually types it, with three-year growth above 20 and debt-to-equity at or below 0.2, is a different animal: it can only fill 43% of a fifteen-slot book and returns 10.8% after tax, BELOW the Midcap 150 index at 16.41%, and three of those eleven points are the cash yield on the 57% of the book it could not fill. Set the idle-cash assumption to zero and it returns 7.74%. The replication gate found this before any backtest did: the written screen picks 8 of the 69 stocks he actually holds, while "near its all-time high" ALONE picks 42 of them.',
    caveats: [
      'THE WINDOW CANNOT BE EXTENDED — 8.1 years, and it contains the 2023-25 smallcap boom while throwing away 2008 and 2020. Every figure for this book is measured on a kinder period than the others’ headline window.',
      'Screener shows figures as they stand TODAY. The four-month filing lag controls when a year becomes visible; it cannot undo a later restatement.',
      'TWO OF THE SCREEN’S CRITERIA ARE INERT. ROCE never rejects a name that ROE has not already rejected, and debt-to-equity ≤ 0.2 removes the entire financial sector by construction, because Screener carries no Borrowings row for lenders.',
      'Read the % invested next to every CAGR. A thinly invested book’s return is the idle-cash yield wearing a strategy’s name.',
      'Delete its ten best trades out of 213 and the trade-level compounding proxy falls from 98.5× to 0.32× — below one.',
      'Survivorship pressure is upward on every arm, benchmarks included. The random null is the control that neutralises it.',
      'The database is not retroactively split-adjusted; the all-time-high running maximum restarts on 152 detected events, which also fires on genuine crashes and makes the near-the-high state EASIER to satisfy for those names.',
    ],
    links: [
      { label: 'Full study — Arun’s screener query tested (research/160)', href: '/app/backtest/quality-growth-near-ath-research160', kind: 'study' },
      { label: 'STATUS — research/160_quality_growth_near_ath/QUALITY_GROWTH_NEAR_ATH_DAILY_SWEEP_STATUS.md', href: '/app/backtest/quality-growth-near-ath-research160', kind: 'status' },
      { label: 'No dashboard — research only, never papered', href: '', kind: 'dashboard' },
      { label: 'No register entry — not a system we run', href: '', kind: 'register' },
    ],
  },

  // ------------------------------------------------------------------ IPO BASE
  {
    key: 'IPO Base',
    name: 'IPO Base',
    kind: 'paper',
    statusLabel: 'LIVE PAPER · arms on the first deposit',
    accent: 'purple',
    size: '₹10,00,000 notional — arms for real money on the first Capital Desk deposit',
    windowNote: 'Measured on the full 20.4-year window and again on the 2018 window.',
    heatKey: 'IPO Base',
    alert:
      'ITS PARAMETERS HAVE NOT BEEN RE-OPTIMISED. IPO Base’s 680-cell sweep was scored against the same look-ahead entry that broke Open Alpha’s, and for Open Alpha the parameter surface INVERTED once the entry was made placeable. So this book’s 20-SMA trail and +25% target are suspect on exactly the same grounds. The re-optimisation is the top owed item and it is NOT STARTED.',
    rule:
      'A recently listed stock closes above the highest close of its last 25 bars, from a base no deeper than 30% → buy-stop AT the pivot the next day; −8% close stop, +25% target, exit below the 20-SMA; 8 slots at 18.75%, no market gate.',
    rules: [
      { k: 'Universe', v: 'NSE equities with a VETTED listing date (research/153 table, 1,353 accepted), ETFs excluded, all pre-listing rows masked' },
      { k: 'Age band', v: 'Listed within 6 months AND at least 60 bars — 60, not the spec’s 25: the study’s own harness only admitted stocks with 60+ bars, so 60 is what was validated' },
      { k: 'Liquidity', v: '20-day median traded value ≥ ₹5 cr at t−1' },
      { k: 'Signal', v: 'Pivot = the highest close of the last 25 bars; base depth ≤ 30%; not already extended; close > pivot' },
      { k: 'Exits', v: 'Stop at close ≤ 0.92× buy → target at close ≥ 1.25× buy → close below the 20-SMA (the entry bar is exempt)' },
      { k: 'Book', v: '8 slots at 18.75% of equity, 25 bps a side, NO market gate — it lost 30 of 30 seeds' },
      { k: 'Tie-break', v: 'Highest 20-day traded value first — PRE-REGISTERED, not backtested: the study drew lots across 30 seeds' },
      { k: 'Data guard', v: 'A single-day close move ≤ −40% is treated as a split or bonus: the position is HELD and alerted, never stopped out' },
    ],
    mechanics: [
      { k: 'Trigger', v: 'Close above the 25-bar pivot' },
      { k: 'Fill', v: 'The NEXT day, a buy-stop resting AT the pivot, filled at max(pivot, open). THE LIVE ENGINE IS CORRECT and always was — it is the STUDY that filled on the signal day' },
      { k: 'Exit', v: 'Close below the 20-day SMA' },
      { k: 'Stop', v: '−8% on the close; target +25% on the close' },
      { k: 'Slots', v: '8 at 18.75%' },
      { k: 'Cadence', v: 'Event-driven, with long idle stretches by design — the sleeve is 33% invested on average and took no trades at all in 2013-14' },
      { k: 'Liquidity', v: '₹5 cr 20-day median traded value' },
      { k: 'Gate', v: 'NONE' },
    ],
    evidence: [
      { k: 'Own study window', v: '15.00% after-tax CAGR at −37.55%, Calmar 0.40, 2006 → Sep-2026, 30 seeds — the entry the LIVE engine actually places (that study credited idle cash at 5.0%; the rows on this page credit 5.2%, worth +0.16 a year to a book only 32% invested — the largest cash-rate effect of any book here). Source: research/158_oa_arming_width/OA_ARMING_WIDTH_AND_POKE_FILL_DAILY_SWEEP_STATUS.md §3' },
      { k: 'The correction', v: 'The study headline reproduced at 31.48% and was published at 31.03%. Its close-fill control gives 17.49%. The live engine’s honest next-day entry gives 15.00% — the published number HALVES' },
      { k: 'Why it survives', v: '98.5% of same-day signals survive as reachable next-day fills, so the loss is the ENTRY PRICE, not missed trades. The book survives at about half strength rather than dying' },
      { k: 'Against the index', v: 'NIFTYBEES over the same window 11.5% at −59.7%. IPO Base still beats it on return AND on drawdown' },
      { k: 'Diversification', v: 'The only genuine diversifier in the set — loosely coupled to every other book and to the index alike. See the correlation heatmaps above' },
      { k: 'Cells disclosed', v: '680 (research/153 adopted spec) — ALL OF THEM SCORED ON THE LOOK-AHEAD ENTRY, and therefore suspect' },
      { k: 'The years that earn', v: 'It is the tall bar in 2020 and it earns again in 2023, in years when Open Alpha was flat or negative' },
    ],
    distinctiveTitle: 'Next-day entry halves the published number — to 15.0%, and it still clears',
    distinctive:
      'Alone among the three books, IPO Base was never traded wrongly: services/ipo_paper.py always waited and bought the next morning. It was the STUDY that bought on the signal day. So nothing had to be paused and nothing had to be repaired — the only thing that changed on 11-Sep-2026 was the number the book is allowed to claim, from 31.0% to 15.0%. It still beats the index on both return and drawdown, it is the only real diversifier in the portfolio, and the honest read on it is its Calmar rather than its CAGR, because it sits about two thirds in cash by design.',
    caveats: [
      'THE RE-OPTIMISATION IS NOT STARTED. Its 680-cell sweep was scored on the look-ahead entry exactly as Open Alpha’s was, and for Open Alpha the trail surface INVERTED when the entry was corrected. The trail and the +25% target are suspect until a staged re-run is done: exit economics (trail 10-50 × target {+25%, +50%, none} × stop {6, 8, 10, none}), then base geometry, then a null control and a gate bake-off, after tax throughout.',
      'RENAMED SYMBOLS GO STALE SILENTLY, AND THIS BOOK IS THE MOST EXPOSED. LOTUSDEV is absent from the instrument dump — the tradeable symbol is LOTUSDEV-BE — so the nightly refresh asks for the dead name, gets nothing, and treats it as "no new bars". Its data is frozen 126 days. Ten of eleven stale young names are missing from the dump and six freeze on the same day, a batch series migration. IPO Base trades exactly the young, thin names NSE moves to trade-for-trade. NOT FIXED.',
      'IPO BASE SITS ABOUT TWO THIRDS IN CASH by design, so comparing its CAGR with a fully invested index is not like for like. Its Calmar is the fairer read and its multi-year dead zones are structural, not decay.',
      'The published spec records a 25-bar floor but the study’s panel loader admitted only symbols with 60+ bars, so the published 31.03% was earned on stocks aged roughly 3-6 months. The live book was set to 60 to match what was validated; whether 25 is tradeable is an open review dated 15-Dec-2026.',
      'The ENTIRE edge is getting filled AT the pivot — taking the signal-day close instead costs 14.08pp of CAGR and loses on 30 of 30 paired seeds. The soak review on 15-Oct-2026 exists to measure exactly that.',
      'The live book breaks ties by highest 20-day traded value while the backtest drew lots across a 28.82-33.44% spread. A pre-registered deviation, not a validated choice.',
      'Its own-window drawdown is −37.55% in the r/158 audit and −35.86% on the roster curve file: two honest runs over slightly different spans. The audit figure is quoted here; both are in the provenance table.',
    ],
    links: [
      { label: 'Full study — IPO Base breakout (research/153)', href: '/app/backtest/ipo-base-breakout-research153', kind: 'study' },
      { label: 'The entry audit that halved it (research/158) — on the roster page', href: '/app/backtest/mpf-honest-entries-roster-2026-09', kind: 'status' },
      { label: 'Live paper dashboard — the IPO Base book', href: '/portfolio?tab=ipo', kind: 'dashboard' },
      { label: 'Register entry — Strategies index', href: '/strategies', kind: 'register' },
    ],
  },
];

/** Section 4 — the dated reviews in the Ops & Review Centre that touch these books. */
export const REVIEWS: { title: string; due: string; status: string; what: string }[] = [
  {
    title: 'Open Alpha · Base Age — RS ≥ 70 test, reconcile with the honest re-optimisation, paper-book call',
    due: '26 Sep 2026',
    status: 'PENDING',
    what:
      'It cleared all five pre-registered criteria standalone, but its portfolio-fit test could not be run on the day it landed, because the old Open Alpha curve was unusable as a benchmark. Now that an honest curve exists, run the blend test. PASS CRITERION: correlation below 0.40 to the honest Open Alpha AND +0.10 Calmar at some weight. Also test RS ≥ 70 — the one axis from Open Alpha’s spec the study never swept — and decide SEPARATELY whether the SuperTrend(14,4) exit should be proposed as a change to the live book, which would be its own study.',
  },
  {
    title: 'Open Alpha — automated execution soak: did the machine trade the spec?',
    due: '9 Oct 2026',
    status: 'PENDING',
    what:
      'From 8-Sep-2026 the live book places its own exits and arms its own entries. Check that every exit raised was placed and filled with nothing resting overnight; that every freed slot was refilled; that reconcile applied every tagged fill and the book matches the broker exactly; and that realised P&L per closed trade is sane. FAIL on any of these means going back to alert-only. NOTE: buying has been PAUSED since 11-Sep-2026, so this review will have far less evidence than planned.',
  },
  {
    title: 'Quality Summit — the one open question: quality as an OVERLAY inside Open Alpha’s entries',
    due: '10 Oct 2026',
    status: 'PENDING',
    what:
      'research/160 killed the standalone book in both families and found it dilutive to the live pair at every weight. The single question the correlation does NOT answer: does a LOOSE quality gate improve Open Alpha’s OWN entries as an overlay? The point-in-time Screener panel and 37 causal masks now exist, so this costs a day. Measure against the HONEST Open Alpha curve, never the old one. PASS CRITERION: ≥ +2pp after-tax CAGR or ≥ +0.15 Calmar on ≥ 8 of 12 paired paths. If it fails, close the quality-screen line permanently.',
  },
  {
    title: 'IPO Base paper soak — fill quality against the pivot',
    due: '15 Oct 2026',
    status: 'SCHEDULED',
    what:
      'The book went to paper on 6-Sep-2026 and runs the adopted spec forward on real prices. PASS CRITERION, pre-registered in research/153: modelled versus actual fill within 0.5% of the pivot, and a miss rate under 15%. That is the only thing the soak needs to answer, because the entire edge is the fill. Also report realised selection against the 30-seed band. EXPECT LONG IDLE STRETCHES — that is the strategy working, not decay.',
  },
  {
    title: 'Open Alpha — measure the gap-ceiling deviation',
    due: '8 Nov 2026',
    status: 'PENDING',
    what:
      'A known, accepted deviation: the study fills entries at max(pivot, open) with NO ceiling, but Kite refuses SL-M via the API and the exchange caps a stop-limit’s spread at roughly 3%. A breakout that gaps more than about 3% above its pivot will NOT fill where the backtest took it — and gap-ups are exactly where breakout edges concentrate. Count armed-but-unfilled entry orders and the gap on each, then replay with a 3% fill ceiling to price what the constraint costs.',
  },
  {
    title: 'IPO Base — is the 25-bar floor in the spec actually tradeable?',
    due: '15 Dec 2026',
    status: 'PENDING',
    what:
      'The adopted spec records min_bars 25, but the study’s harness built its panel with n ≥ 60, so no stock with fewer than 60 daily bars was EVER shown to the backtest. The live book was set to 60 to match what was validated. The wider band may well be better, but it is untested — and that belongs in a study, not a live book.',
  },
  {
    title: 'Quality Summit — refresh the point-in-time Screener panel before any reuse',
    due: '1 Feb 2027',
    status: 'PENDING',
    what:
      'The panel is now the shared store backtest_data/fundamentals.db, fetched 11-Sep-2026 and stopping at 2026-09-01. Screener figures are RESTATED rather than as-reported and the free page depth is about twelve fiscal years, so the earliest usable month walks forward every August as a new fiscal year is filed. Do NOT reuse a stale panel for a live decision.',
  },
];

/** Section 4 — what is owed, in the priority the entry-audit session left it in. */
export const OWED: { title: string; state: string; what: string }[] = [
  {
    title: '1 · IPO Base re-optimisation on the honest entry',
    state: 'NOT STARTED — top priority',
    what:
      'Its 680-cell sweep was scored on the look-ahead entry, exactly like Open Alpha’s. Given the trail surface INVERTED for Open Alpha, IPO Base’s 20-SMA trail and +25% target are suspect. Staged plan: (a) exit economics — trail 10-50 × target {+25%, +50%, none} × stop {6, 8, 10, none}; (b) base geometry — L, depth, age band; then a null control and a gate bake-off, after tax throughout.',
  },
  {
    title: '2 · services/oa_entry.py still has the inverted condition AND the old ETF filter',
    state: 'NOT FIXED — entry crons commented out',
    what:
      'Backup of the crontab at /tmp/mpf/ct.bak.20260911-111828. Do NOT restore the entry jobs before both are fixed. This is engine work and belongs in its own change with its own STATUS doc and an after-15:40 deploy — never as a side effect of a report page.',
  },
  {
    title: '3 · The rename / refresh defect',
    state: 'NOT FIXED',
    what:
      'scripts/refresh_daily_universe.py should resolve each symbol against the instrument dump and follow or raise a rename, rather than returning quietly when the dead name yields nothing. IPO Base is the most exposed book.',
  },
  {
    title: '4 · research/142’s published figures carry the fund contamination',
    state: 'RE-RUN OWED',
    what:
      'The ETF filter matched ticker spellings written before the 2023-25 gold and silver wave, so 221 funds — EGOLD, GROWWGOLD, TATAGOLD, ESILVER, MON100, MAFANG, ICICIB22 and about 180 sector and index funds — were reaching an EQUITY momentum book. Fixed going forward via backtest_data/etf_exclusions.json, built from the instrument’s long NAME. research/142’s published numbers still carry it in their recent years.',
  },
  {
    title: '5 · True North’s curve is a single path',
    state: 'RE-RUN OWED',
    what:
      'Every other book here is an ensemble. A like-for-like re-run of True North under the same ensemble treatment is owed before the columns are compared to the decimal.',
  },
  {
    title: '6 · Research-number collision on 159',
    state: 'OPEN',
    what:
      'Two folders are numbered 159 — 159_oa_honest_reoptimization and 159_rounding_base_breakout. The former should renumber, but it is referenced from the roster page, its own STATUS doc and several scripts.',
  },
  {
    title: '7 · Blend and allocation work across True North + Base Age + IPO Base',
    state: 'NOT STARTED — and it is the only structure that plausibly clears 25%',
    what:
      'No single book reaches 25% after tax. The 50-50 blend row on this page is this report generator’s own arithmetic, not a study: it is a first look, not an allocation. A real study would sweep weights, rebalance frequency and the cash rule, on paired paths, after tax — and it has not been started.',
  },
  {
    title: '8 · The after-tax re-run of the entry-mechanic tables is incomplete',
    state: 'CRASHED',
    what:
      'research/159 scripts/aftertax_all.py stopped with a ValueError before it reached the VIX-gate rows. Tables A, B and the PRICE-gate rows of C completed after tax and are published on this page. Every VIX-gate figure exists only pre-tax, so — post-tax-only being binding — none is printed here.',
  },
  {
    title: '9 · There is no pre-registered soak criterion for a Base Age paper book',
    state: 'OPEN',
    what:
      'Nothing on disk defines what would count as the book tracking its study. If the 26-Sep answer is yes, that criterion has to be written BEFORE the book starts, exactly as IPO Base got one.',
  },
];
