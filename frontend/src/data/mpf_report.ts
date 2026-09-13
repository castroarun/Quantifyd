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

export const REPORT_DATE = '13 Sep 2026';

export const HIGHLIGHTS: string[] = [
  'No single book reaches 25% after tax over the full twenty years, and neither does the blend. research/168 settled the allocation on 12-Sep-2026 — True North 37.5 / Base Age 37.5 / IPO Base 25, monthly — and what the third sleeve buys is a shallower fall: 21.18% after tax at −24.01% against the old pair’s 20.28% at −26.91%, better on both counts on 30 of 30 paired paths (study medians).',
  'True North holds cash 57% of the time and still produces one of the two best returns. That is the gate doing the work, and it is a stronger result than the CAGR alone says.',
  'IPO Base changed on 12-Sep-2026, and for a reason worse than a weak number. Measured on the entry the book actually places, its old rules LOST to picking names at random. One dial fixed it — a 50-day trail instead of a 20-day one — and the re-fitted book is both the better standalone system and the better blend sleeve, even though it is slightly more correlated to the other two.',

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
      { k: 'Selection', v: 'Relative-strength rank among the qualifying set, top 15 equal weight. RANK LEEWAY: a holding is KEPT while it still ranks within 23 (1.5 × 15, rounded up); a name slipping to 16th, 20th or 23rd at a rebalance stays. It is sold only when it ranks 24th or worse, or leaves the near-high band, the liquidity floor or the quality screen. research/170 swept leeways from 15 to 45 and none beat 23; only 14.5% of all sales are rank sales, the other 85.5% are band, liquidity or screen exits' },
      { k: 'Exit', v: 'No stop, no trail. A holding leaves only at a rebalance, and only when it ranks 24th or worse (leeway to rank 23) or no longer qualifies (near-high band, liquidity, quality screen)' },
      { k: 'What Arun actually types', v: 'This is his own screener.in query with TWO dials relaxed — growth from 20% to 10%, and the debt-to-equity test removed. Left as written those two dials cost 11.8 points of CAGR' },
    ],
    mechanics: [
      { k: 'Trigger', v: 'A monthly rebalance date, decided on the close. There is no event and no breakout' },
      { k: 'Fill', v: 'The next open. The engine never reads a high or a low, so the trigger/fill trap is structurally impossible' },
      { k: 'Exit', v: 'Rotation only — a name leaves at a rebalance when it ranks 24th or worse (leeway to rank 23) or stops qualifying. research/170: 85.5% of sales are the name leaving the band, the liquidity floor or the screen; only 14.5% are rank sales' },
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
    kind: 'live',
    statusLabel: 'LIVE · real money since 8-Sep-2026 · re-fitted spec from 12-Sep-2026',
    accent: 'purple',
    size: '₹2,28,711 real money, funded through the Capital Desk. HARD CAPACITY CAP about ₹20–25L: at ₹1cr a typical position would be most of a day’s volume in these names',
    windowNote: 'Measured on the full 20.4-year window and again on the 2018 window — both on the RE-FITTED spec.',
    heatKey: 'IPO Base',
    alert:
      'THE SPEC CHANGED ON 12-SEP-2026, AND THE BOOK WAS DEAD FOR FOUR SESSIONS BEFORE THAT. research/167 measured the old rules on the entry the book actually places and they lost to a random-entry control; the live book now runs the re-fit (trail SMA-50, stop 10%, NIFTYBEES<SMA-150 entry gate). Separately, a 3-Sep commit had dropped two constants that only the live branch reads, so from the first real deposit on 8-Sep every nightly run crashed and no exit was checked on 9, 10 or 11 Sep. Restored on 12-Sep. The one open position would have been held on all three days under either rule set.',
    rule:
      'A recently listed stock closes above the highest close of its last 25 bars, from a base no deeper than 30% → buy-stop AT the pivot the next day, filled only if the day’s high reaches it; −10% close stop, +25% target, exit below the 50-SMA; 8 slots at 18.75%; no new entries while NIFTYBEES closes below its 150-day average.',
    rules: [
      { k: 'Universe', v: 'NSE equities with a VETTED listing date (research/153 table, 1,353 accepted), funds excluded by instrument long name, all pre-listing rows masked' },
      { k: 'Age band', v: 'Listed within 6 months AND at least 25 bars on the signal day — the floor Spec A was validated at. The book ran 60 from 6 to 13 Sep on a misreading of the harness; research/169 measured that version at about half the return' },
      { k: 'Liquidity', v: '20-day median traded value ≥ ₹5 cr at t−1' },
      { k: 'Signal', v: 'Pivot = the highest close of the last 25 bars; base depth ≤ 30%; not already extended; close > pivot' },
      { k: 'Market gate', v: 'NEW 12-Sep-2026: no new entries while NIFTYBEES closes below its 150-day average. Holdings keep their own exits' },
      { k: 'Exits', v: 'CHANGED 12-Sep-2026: stop at close ≤ 0.90× buy → target at close ≥ 1.25× buy → close below the 50-SMA (entry bar exempt). Was 0.92× and the 20-SMA' },
      { k: 'Book', v: '8 slots at 18.75% of equity, 25 bps a side — re-confirmed on the honest entry against 5, 10 and 16 slots' },
      { k: 'Tie-break', v: 'Highest 20-day traded value first — PRE-REGISTERED, not backtested: the study drew lots across 30 seeds' },
      { k: 'Data guard', v: 'A single-day close move ≤ −40% is treated as a split or bonus: the position is HELD and alerted, never stopped out' },
    ],
    mechanics: [
      { k: 'Trigger', v: 'Close above the 25-bar pivot' },
      { k: 'Fill', v: 'The NEXT day, a buy-stop resting AT the pivot, filled at max(pivot, open) — and only if the day’s high reached the pivot. That last clause was missing from the live engine until 12-Sep-2026' },
      { k: 'Exit', v: 'Close below the 50-day SMA' },
      { k: 'Stop', v: '−10% on the close; target +25% on the close' },
      { k: 'Slots', v: '8 at 18.75%' },
      { k: 'Cadence', v: 'Event-driven, 18.9 trades a year held a median 37 days, with long idle stretches by design — about 36% invested on average' },
      { k: 'Liquidity', v: '₹5 cr 20-day median traded value' },
      { k: 'Gate', v: 'NIFTYBEES below its 150-day average blocks new entries' },
    ],
    evidence: [
      { k: 'Re-fitted, own study', v: '21.80% after tax at −26.6% median drawdown (−32.9% worst seed), Calmar 0.819, 30 seeds, 2006 → Sep-2026, idle cash 5.0%. Seed band 20.83–23.19. Source: research/167 results/stage9_adoption.csv' },
      { k: 'Re-fitted, this page’s basis', v: 'At 5.2% idle cash on this page’s 20.4-year window the 30-seed median is 22.08%, band 21.09–23.49; the drawn path is the median-CAGR seed. Source: research/168 ipo_navs_cash052.npz, arm A_25bps_y52' },
      { k: 'The old spec, honestly measured', v: '14.90% after tax at −38.6%, and it LOST to a date-matched random-entry control: 14.90 vs 15.11, real rules winning 14 of 30 paired seeds, 8 of 30 with a gate. The published 31.03% was a same-bar look-ahead fill. Source: research/167 §3' },
      { k: 'Why the trail', v: 'Edge over random, in points of CAGR at trail 10 / 15 / 20 / 30 / 40 / 50 / 60 / 75 / 100: +0.04 / −0.71 / −0.10 / +1.26 / +3.08 / +4.78 / +2.05 / +0.43 / +0.31. Zero or negative across the old spec’s whole region, unanimous 30 of 30 from 30 to 75. CORRECTED 13-Sep-2026 by research/169: that control ran on a panel with gaps that handicapped the random arm; on a clean panel the edge at trail 50 is +2.25 points, holding for 2006–2015 and tying random young names in 2016–2026' },
      { k: 'In the portfolio', v: 'research/168: the re-fit is worth more to the blend than the old spec at EVERY weight from 5% to 50%, 30 of 30 paired paths; the old spec earns no place at any weight. Against arbitrage-fund cash at equal risk it adds 2.52 points of CAGR at a 25% weight, 30 of 30' },
      { k: 'Recommended weight', v: '25% of the book — True North 37.5 / Base Age 37.5 / IPO Base 25, monthly: 21.18% after tax, −24.01%, Calmar 0.885 (30-path medians). Adopted on the Capital Desk on 13-Sep-2026, with the live book switched to the validated 25-bar floor in the same change' },
      { k: 'Diversification', v: 'Still the loosest-coupled book, but the re-fit is a little MORE correlated than the old spec (monthly 0.348 to True North and 0.329 to Base Age, against 0.259 and 0.319) — and still the better blend sleeve. Pairwise correlation was the wrong screen' },
      { k: 'Cells disclosed', v: '~350 in research/167 (read the 21.80% as 19–22%), plus the research/168 weight grid' },
    ],
    distinctiveTitle: 'The old rules picked no better than chance. One dial — the trail — made it a real edge',
    distinctive:
      'research/153 published 31.0%. Its fill was unplaceable, and measured on the entry this book actually uses the same rules returned 14.90% — and lost to drawing names at random from the same universe. A 25-bar base breakout in a young stock is an edge only if the winner is given room to run: at a 20-day trail the rules add nothing over chance, at a 50-day trail they add almost five points a year and win every paired run. The re-fit is live, and research/168 then showed it is also the better blend sleeve, so the question left is the weight, not the rules.',
    caveats: [
      'THE EDGE LIVES IN THE FIRST FEW WEEKS AFTER LISTING, AND IT IS NARROW. research/169: at a 25-bar floor the rules return 21.80% after tax; at 40 bars they LOSE to random young names on 27 of 30 runs; at 60 bars they return 11.57%. The live book ran 60 by mistake from 6 to 13-Sep-2026 and now runs 25. Every other universe tried — Nifty 50 through all stocks — fails its random control.',
      '2008 IS THE HONEST BLACK MARK. The re-fit loses 10.3% in 2008 where the old spec made +0.4%, because the fast 20-day trail that costs seven points a year in normal times is exactly what sidestepped that crash. At a 25% blend weight the gap shrinks to about 2.5 points (research/168), and plain cash at the same weight would give the same cushion. This book is not a crash hedge.',
      'THE TRAIL AND THE GATE WERE BOTH CHOSEN AFTER SEEING THE DATA. ~350 cells were scored, so read the 21.80% as 19–22%. The only out-of-sample evidence is the 2006–2015 / 2016–2026 split, which it passes strongly. There is no held-out period.',
      'CAPACITY IS THE BINDING CONSTRAINT. At ₹10L the MEDIAN position is 1.56% of the name’s own 20-day traded value and the 90th percentile 9.05% (research/167 had labelled the median as the 90th percentile); at ₹1cr the 90th percentile is about 90% of a day’s volume. A 25% weight is fundable today but binds once the whole book passes about ₹85L.',
      'RENAMED SYMBOLS GO STALE SILENTLY, AND THIS BOOK IS THE MOST EXPOSED. LOTUSDEV is absent from the instrument dump — the tradeable symbol is LOTUSDEV-BE — so the nightly refresh asks for the dead name and treats nothing as "no new bars". Ten of eleven stale young names are missing from the dump. NOT FIXED.',
      'A CRASHING CRON WRITES A TRACEBACK TO A FILE NOBODY READS. That is how four sessions of a live book passed unnoticed. The same shape exists on every paper and real book; a health check is registered for 19-Sep-2026 and the alerting fix is still owed.',
      'IPO BASE SITS ABOUT TWO THIRDS IN CASH by design, so its Calmar is the fairer read against a fully invested index, and multi-year dead zones are structural rather than decay.',
      'The live book breaks ties by highest 20-day traded value while the backtest drew lots across 30 seeds. A pre-registered deviation, not a validated choice.',
    ],
    links: [
      { label: 'Full study — IPO Base re-measured and re-fitted (research/167)', href: '/app/backtest/ipo-base-honest-reopt-research167', kind: 'study' },
      { label: 'The three-sleeve blend that set the weight (research/168)', href: 'https://github.com/castroarun/Quantifyd/blob/main/research/168_three_sleeve_blend/results/RESULTS.md', kind: 'study' },
      { label: 'The original study, superseded (research/153)', href: '/app/backtest/ipo-base-breakout-research153', kind: 'status' },
      { label: 'Live dashboard — the IPO Base book', href: '/portfolio?tab=ipo', kind: 'dashboard' },
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
    state: 'DONE 12-Sep-2026 — deployed to the live book',
    what:
      'research/167. The old spec, measured on the entry the book places, lost to a random-entry control (14 of 30). Re-fitted on ~350 cells: trail SMA-20 → SMA-50, stop 8% → 10%, plus a NIFTYBEES<SMA-150 entry gate — 21.80% after tax, −26.6% median drawdown, beating the control 30 of 30. Base geometry and 8-slot sizing were re-confirmed unchanged. Live since 12-Sep-2026, with the open position’s stop re-based; its bar floor was corrected from 60 to the validated 25 on 13-Sep-2026 (research/169). Published: /app/backtest/ipo-base-honest-reopt-research167.',
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
    state: 'DONE 12-Sep-2026 — weights adopted on the Capital Desk 13-Sep-2026',
    what:
      'research/168. Recommended TN 37.5 / Base Age 37.5 / IPO Base 25, monthly: 21.18% after tax, −24.01%, Calmar 0.885, against the 50-50 pair’s 20.28%, −26.91%, 0.749 — better on both on 30 of 30 paired paths. It does not reach 25%. The Calmar surface peaks at a 45–60% IPO weight, reported but not recommended because of capacity and the lack of a held-out period. Adopted on the Capital Desk on 13-Sep-2026. Still open: research/168 also found the two-sleeve pair itself prefers a True North tilt (85:15, Calmar 0.826) — a review dated 26-Sep-2026.',
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
