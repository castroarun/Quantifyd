// Data-driven registry of backtest research studies.
//
// Every study renders through the SAME uniform 8-section layout in
// pages/BacktestStudy.tsx. Adding a future study = append one more
// `BacktestStudy` object below — no component changes needed. Keep the
// schema generic enough for any strategy backtest.

export type StudyStatus = 'COMPLETE' | 'RUNNING' | 'STUCK' | 'FAILED' | 'PARKED';

/** A generic comparison/results table: header row + body rows of strings.
 *  `highlightRows` (0-based body-row indexes) get the winner-accent style. */
export interface StudyTable {
  title: string;
  caption?: string;
  columns: string[];
  rows: string[][];
  highlightRows?: number[];
  /** When true, numeric body columns are rendered as a diverging
   *  red→neutral→green heatmap (scaled per-column by its own range).
   *  Non-numeric columns (e.g. Year, Note) stay plain. */
  heatmap?: boolean;
}

/** A single headline metric tile (CAGR / Sharpe / MaxDD / Calmar / …). */
export interface StudyMetric {
  label: string;
  value: string;
  hint?: string;
  /** 'pos' | 'neg' tints the value; omit for neutral. */
  tone?: 'pos' | 'neg';
}

/** A labelled key/value row used in System & Conditions sections. */
export interface KV {
  k: string;
  v: string;
}

/** An explicit statement of the actual traded SYSTEM RULES, placed
 *  early (before the results/comparison tables) so the rules precede
 *  the evidence. `sharedCore` is the rule-set common to every variant
 *  (rendered as a key/value list); `riskLayer` is the per-variant
 *  divergence (rendered as the standard comparison table). Optional —
 *  studies without distinct variants can omit it. */
export interface SystemRules {
  intro?: string;
  sharedCoreTitle: string;
  sharedCore: KV[];
  riskLayer: StudyTable;
}

export interface WinnerCallout {
  /** e.g. the config label */
  config: string;
  /** one-line why-it-won */
  summary: string;
  /** the headline numbers as compact "k: v" lines */
  metrics: KV[];
  /** rejected / void variants to explicitly call out */
  rejected?: string[];
}

export interface LinkRef {
  label: string;
  href: string;
}

export interface BacktestStudy {
  slug: string;
  title: string;
  verdict: string;
  status: StudyStatus;
  date: string; // ISO yyyy-mm-dd

  /** Short blurb for the index card. */
  cardBlurb: string;
  /** 2-3 headline stats for the index card. */
  cardStats: { label: string; value: string }[];

  // ---- Section: System Rules (optional; rendered early, before
  //      System/Conditions/Comparisons so the actual traded rules
  //      precede the evidence) ----
  systemRules?: SystemRules;

  // ---- Section: System ----
  system: {
    intro: string;
    rows: KV[];
  };

  // ---- Section 3: Conditions ----
  conditions: {
    intro?: string;
    rows: KV[];
  };

  // ---- Section 4: Comparisons ----
  comparisons: StudyTable[];

  // ---- Section 5: Results ----
  results: {
    metrics: StudyMetric[];
    tables: StudyTable[];
    /** Optional finished figures (e.g. an equity/drawdown overlay PNG, a
     *  returns heatmap PNG). Each `src` is a web path served under /app/
     *  (image lives in frontend/public/ → copied to static/app/ at
     *  build). Each is rendered as a responsive full-width image with a
     *  muted caption beneath, in order. */
    charts?: { src: string; caption: string }[];
    /** Optional embedded interactive HTML reports (self-contained, served under /app/),
     *  rendered as full-width iframes. */
    embeds?: { src: string; height?: number; caption?: string }[];
  };

  // ---- Section 6: Winners ----
  winners: WinnerCallout[];

  // ---- Section 7: Caveats ----
  caveats: string[];

  // ---- Section 8: Links ----
  githubLinks: LinkRef[];
  projectPaths: string[];
}

const GH = 'https://github.com/castroarun/Quantifyd/tree/main/research/41_midsmall400_mq_concentrated';

export const BACKTEST_STUDIES: BacktestStudy[] = [
  {
    slug: 'nifty-45dte-short-straddle',
    title: '45-DTE NIFTY Short Straddle — replicating "The Long & The Short Ep. 48", then sizing it on real margin',
    verdict:
      'Sandeep Rao published a 45-DTE short-straddle backtest (sell ATM straddle 45 calendar days before the NIFTY monthly expiry; exit at 50% profit / 200% stop / 21 DTE). We rebuilt it on real NSE bhavcopy option prices, Jan-2019 to Jun-2026. HIS TABLE REPLICATES: 89 trades vs his 83, win 70.8% vs 69.9%, avg win/loss +200.2/-217.8 vs his +196.1/-216.8, exit mix 1/3/85 vs 1/4/78, net +78.0 pts/trade at t = 3.12 over 89 non-overlapping trades. SIZED ON REAL BROKER MARGIN (Rs 3L/lot x 10 lots = Rs 30L, Rs 36L blocked with buffer; NIFTY lot 65 so 1 pt = Rs 650): **CAGR 11.47% vs NIFTY 11.60%, but MaxDD -13.8% vs the index -38.4% — Calmar 0.83 vs 0.30.** Same return as the index on a third of the drawdown. MONITORING FREQUENCY settled on REAL 1-minute data (our option_chain recorder, 28.3M quotes): in the DTE>=21 band the ATM straddle travels a mean 6.3% above / 4.3% below its close, ZERO of 60 real sessions travelled >=50% either way, and across the 3 real 45-DTE trades the recorder overlaps the premium never left 0.55x-1.08x of credit - so the 0.50 target and 2.00 stop were never approached at any minute. Daily->hourly changes ONE trade (P&L flat, worst trade -29%, MaxDD -23%); 60m/30m/15m/5m are identical to the decimal. VIX FILTER works but not as claimed: trade counts match his exactly (21 vs 21 at >75) yet his 85.7% win rate does NOT reproduce (ours 71.4%) - what rises is premium collected (786->1,053 pts), not hit rate; best cell on capital is >25 (Calmar 1.05). WHY IT WORKS - the mechanism is movement RELATIVE TO PREMIUM: |move| correlates -0.770 with net P&L, but |move| / breakeven-width correlates **-0.898**, and every campaign with a >=+4% move lost 15 times out of 17 (avg -209 pts). Filtering directly on breakeven width works monotonically (78->91->105->109->113->181 pts as the threshold rises) but does NOT beat VIX>25 on capital, and combining them is worse than either alone (corr between the two is only 0.570, so they are related, not identical). Grouped by expiry month, DECEMBER is the only losing month (-51.4/trade) - not because the move is big (2.95%, middling) but because it is paid the LEAST of any month (3.75% breakeven, the thinnest) while drifting up +1.94%. The ‘NIFTY rallies in Nov/Dec’ premise does not survive a longer sample (Nov +0.95%, Dec +0.68% over 2011-26 vs +0.91% all-month; October is the strongest up-month AND our 4th-best straddle month). December’s total is also ONE trade: ex Dec-2023 it averages +75.3, mid-table. DELTA MANAGEMENT MAKES IT WORSE (Phase E): holding to an x% underlying move then cutting - and optionally re-centring at the new ATM - was swept over 7 thresholds x 3 arms x 3 re-entry caps x close/intraday triggers, and NOT ONE CELL beats the 78.1 pts/campaign baseline. The mechanism is explicit: a cycle cut by the move rule realises -28.6 pts at a 38% win rate, while a cycle left to run to 21 DTE earns +83.0 pts at 81%. Friction explains only ~12 of the 67-pt shortfall; the rest is forfeited decay. Cutting on UP moves costs ~3x cutting on down moves (rallies come with falling IV, so the position repairs itself if left alone). To cut risk, cut LOTS: hold @ 5 lots (6.73% CAGR / 9.0% DD / Calmar 0.75) strictly dominates the best managed arm @ 10 lots (5.16% / 9.6% / 0.54). OPEN ITEM: stress margin. Rs 3L/lot is today’s margin at VIX 10.8; VIX peaked 83.6 in Mar-2020 and SPAN scales with vol, so Rs 36L would likely have been breached exactly when the book was losing. Until a margin-call-aware re-run exists, 11.47% is an upper bound. **STRESS-MARGIN TEST NOW BLOCKED, NOT MERELY PENDING (2026-08-25):** reconstructing historical SPAN failed its calibration gate - RMS 12.0% against a pre-declared 10% limit, with structured errors (far strikes under-predicted up to 24.6%, the 64-DTE point by 17.0%). Diagnosed as a missing volatility smile, then found to be unfixable: the wing strikes needed to calibrate the moneyness response carry ZERO open interest and zero volume, so their LTPs are stale by hundreds of points and several have no two-sided quote at all - there is no market vol there to measure. NSE .spn parameter files are not reachable from any public endpoint (every nsccl.DDMMYYYY.s.zip 404s). Tuning the scan parameters until the gate passed would be fitting to the gate, so the reconstruction was ABANDONED rather than massaged. Replaced by a forward margin recorder that logs real SPAN/exposure daily from 2026-08-25. The measured margin surface itself is unaffected and still valid - basket_order_margins is computed from exchange parameters, not from last trades. **NOW IMPLEMENTED AS A PAPER BOOK** at /app/straddle45: 3 lots, VIX percentile rank > 25 as the entry filter, sub-threshold campaigns still traded but tagged OFF-PLAN so the filter is measured rather than assumed, and idle capital swept to LIQUID1 (Kotak Nifty 1D Rate Liquid ETF, 5.11% measured).',
    status: 'COMPLETE',
    date: '2026-08-20',
    cardBlurb:
      'Does 45-DTE straddle selling really work in India? The published backtest replicates on real NSE option data (+78 pts/trade, t 3.12, 89 trades). On real broker margin (Rs 36L for 10 lots) it returns 11.47% CAGR against NIFTY’s 11.60% with a -13.8% drawdown against the index’s -38.4% — Calmar 0.83 vs 0.30. Hourly exit checks beat daily on the tail; nothing below 60-min changes anything, now confirmed on 28.3M real 1-minute quotes. Open item is stress margin, not the edge.',
    cardStats: [
      { label: 'Verdict', value: 'STRATEGY-CANDIDATE - live as PAPER' },
      { label: 'CAGR / MaxDD (real prices)', value: '11.49% / -18.0%' },
      { label: 'Calmar: plain / VIX-filtered / NIFTY', value: '0.64 / 1.06 / 0.30' },
    ],

    systemRules: {
      intro: 'Rules are the video’s, unchanged. Only the exit-check frequency and the entry filter are swept.',
      sharedCoreTitle: 'The traded system',
      sharedCore: [
        { k: 'Instrument', v: 'NIFTY monthly expiry options. Monthly = the last expiry of the calendar month that was already listed 45 days before it expires (the naive "last expiry of the month" picks a WEEKLY from 2025 on, after NSE shifted the weekly expiry day).' },
        { k: 'Entry', v: 'Expiry minus 45 calendar days, at the close; sell 1x ATM CE + 1x ATM PE. ATM = strike nearest spot. Both legs must have actually traded that day.' },
        { k: 'Exit - target', v: 'Combined premium <= 50% of entry credit. Fires once in 89 trades.' },
        { k: 'Exit - stop', v: 'Combined premium >= 200% of entry credit. Fires 3 times in 89 trades.' },
        { k: 'Exit - time', v: 'Expiry minus 21 calendar days. This is how 85 of 89 trades end.' },
        { k: 'Size and capital', v: '10 lots. NIFTY lot 65 (confirmed against Kite’s live instrument master) = 650 qty, so 1 point = Rs 650. Margin Rs 3L/lot = Rs 30L, Rs 36L blocked with a 20% buffer.' },
        { k: 'Cost', v: '0.25% slippage per side + STT 0.1% of sell premium + exchange 0.05% both sides + Rs 20/order. Avg round trip 5.5 pts.' },
      ],
      riskLayer: {
        title: 'Exit-check frequency - the axis the video fixed at 1 hour',
        caption: 'No VIX filter. 89 trades. Points; Rs 650/point at 10 lots.',
        columns: ['Check frequency', 'Win rate', 'Net pts', 'Net/trade', 't', 'Max DD', 'Worst trade', 'Target/Stop/21-DTE'],
        rows: [
          ['Daily close', '70.8%', '6,952.4', '78.1', '3.03', '-998.4', '-811.8', '1 / 2 / 86'],
          ['60-minute (his)', '70.8%', '6,939.1', '78.0', '3.12', '-765.3', '-578.7', '1 / 3 / 85'],
          ['30-minute', '70.8%', '6,939.1', '78.0', '3.12', '-765.3', '-578.7', '1 / 3 / 85'],
          ['15-minute', '70.8%', '6,939.1', '78.0', '3.12', '-765.3', '-578.7', '1 / 3 / 85'],
          ['5-minute', '70.8%', '6,939.1', '78.0', '3.12', '-765.3', '-578.7', '1 / 3 / 85'],
        ],
        highlightRows: [1],
      },
    },

    system: {
      intro: 'Economic hypothesis: short ATM straddles harvest the volatility risk premium (research/89 measured mean INDIAVIX/RV = 1.28 on NIFTY). At 45 DTE you collect a large absolute credit while gamma is still low, and you leave at 21 DTE before gamma turns vicious. That window is the whole design idea - and the real 1-minute data below shows exactly how violent the DTE 0-2 zone the rule avoids actually is.',
      rows: [
        { k: 'Universe', v: 'NIFTY index options only - a single instrument, every monthly expiry taken, none skipped.' },
        { k: 'Trades', v: '89 of a possible 90 monthlies. March-2026 is absent: NSE re-dated that contract, so it had no listing 45 days before its stated expiry.' },
        { k: 'Independence', v: 'One position at a time, 45 to 21 DTE (~24 days) on a monthly cadence - non-overlapping, so the t-stat is honest.' },
        { k: 'Success metric', v: 'Net points per trade with its t-stat, then CAGR and Calmar on the blocked margin.' },
        { k: 'Falsification declared up front', v: 'Kill it if |t| < 2 after cost, if profit concentrates in <= 3 trades, or if it flips sign on the monitoring-frequency axis. None of the three triggered (top-3 trades = 25% of profit).' },
      ],
    },

    conditions: {
      intro: 'Ground truth is real traded option prices throughout - EOD bhavcopy for the 7.5-year study, and our own 1-minute recorder for the monitoring question.',
      rows: [
        { k: 'Option prices (study)', v: 'nse_options_bhav (NSE F&O bhavcopy) - 5.13M real NIFTY rows, 2011-01-03 to 2026-07-21, real OHLC + settlement + contracts + OI.' },
        { k: 'Option prices (intraday)', v: 'option_chain - 28.3M REAL 1-minute NIFTY option quotes from 2026-04-20. The recorder picks each contract up only ~27 days before expiry, so it cannot host a 45-DTE ENTRY, but it covers the DTE 27->0 window minute by minute.' },
        { k: 'Expired-contract intraday: NOT OBTAINABLE', v: 'Tested, not assumed: Kite returns "invalid token" for expired contracts - NIFTY 24000/24050/24100 CE on the July-2026 expiry, one month old, on both 60minute and day intervals. There is no route to 2019-2026 intraday option prices.' },
        { k: 'Spot', v: 'NIFTY50 daily (2011 to date) and 5-minute (2015-02 to 2026-07, 206,990 bars) from market_data_unified.' },
        { k: 'Volatility', v: 'INDIAVIX daily 2015 to date; entry rank vs the previous 252 sessions (causal). Period peak 83.61 on 2020-03-24; 10.83 today.' },
        { k: 'Reconstruction (sub-daily rows only)', v: 'Black-76 on real 5-min spot, forward + IV backed out of real option closes using the PREVIOUS session’s IV (causal), snapped back to the real price at every daily close. Validated against the real 1-minute data - they agree.' },
        { k: 'Benchmark', v: 'NIFTY 50 price index over the identical window: CAGR 11.60%, MaxDD -38.4%, Calmar 0.30. Excludes dividends (~1.2%/yr).' },
      { k: 'Headline basis', v: 'REAL bhavcopy prices, daily-close monitoring: 11.49% CAGR / -18.0% MaxDD / Calmar 0.64 at 10 lots on Rs 36L. The hourly variant (11.47% / -13.8% / 0.83) differs on only 3 of 89 trades and each of those is filled at EXACTLY the trigger level on a reconstructed path - an optimistic assumption - so the real-price basis is the headline and the hourly figures are a sensitivity, not the result.' },
      { k: 'Live implementation', v: 'PAPER book at /app/straddle45 since 2026-08-24: 3 lots, entry filter VIX percentile rank > 25, idle cash in LIQUID1. Seeded by backtracing real campaigns; marks from the broker every 5 minutes.' },
      ],
    },

    comparisons: [
      {
        title: 'Published table vs independent replication',
        caption: 'Daily-close monitoring, real option prices, net of 0.25%/side slippage + statutory costs.',
        columns: ['Metric', 'Published', 'Ours (real data)', 'Read'],
        rows: [
          ['Trades', '83', '89', 'convention'],
          ['Win rate', '69.9%', '70.8%', 'matches'],
          ['Avg premium sold', '758.9', '786.3', 'matches'],
          ['Exits - target/stop/21-DTE', '1 / 4 / 78', '1 / 3 / 85', 'matches'],
          ['Avg win / avg loss', '+196.1 / -216.8', '+200.2 / -217.8', 'matches almost exactly'],
          ['Total P&L (pts)', '5,951.6', '7,283.7 gross / 6,952.4 net', '17% richer'],
          ['Avg P&L per trade', '71.7', '81.8 gross / 78.1 net', '9% richer'],
          ['Best trade', '+805.3', '+866.4', 'matches'],
          ['Worst trade', '-1,062.6', '-811.8', 'ours milder'],
          ['Max drawdown', '-1,062.6', '-998.4', 'matches'],
        ],
        highlightRows: [1, 4],
      },
      {
        title: 'REAL 1-minute intraday travel of the ATM straddle (240 recorded day-contracts)',
        caption: 'How far the combined premium ranged within a session vs where it closed - exactly what a daily-close backtest is blind to. Triggers sit +100% and -50% from entry credit.',
        columns: ['DTE band', 'Day-contracts', 'Above close mean', 'p95', 'max', 'Below close mean', 'p95', 'max'],
        rows: [
          ['>= 21 - the strategy’s band', '60', '6.3%', '14.2%', '36.5%', '4.3%', '20.9%', '27.7%'],
          ['3-20 - after our exit', '153', '8.8%', '20.9%', '35.3%', '4.4%', '21.1%', '44.2%'],
          ['0-2 - expiry week, NEVER held', '27', '381%', '966%', '7,669%', '22.6%', '66.9%', '70.2%'],
        ],
        highlightRows: [0],
      },
      {
        title: 'The three real 45-DTE trades the 1-minute recorder overlaps',
        caption: '14 real sessions. The premium never left 0.55x - 1.08x of credit; neither trigger was approached at any minute.',
        columns: ['Expiry', 'Entry', 'Strike', 'Credit (pts)', 'Overlap', 'Real intraday range (x credit)', 'Trigger touched?'],
        rows: [
          ['2026-05-26', '2026-04-10', '24050', '1,189.7', '4 days (DTE 27-21)', '0.66 - 0.79', 'no'],
          ['2026-06-30', '2026-05-15', '23650', '1,155.4', '5 days (DTE 27-21)', '0.55 - 0.74', 'no'],
          ['2026-07-28', '2026-06-12', '23600', '951.4', '5 days (DTE 27-21)', '0.82 - 1.08', 'no'],
        ],
      },
      {
        title: 'India VIX percentile filter (rank vs previous 252 sessions)',
        caption: 'His trade counts reproduce exactly; his win-rate claim does not. Capital basis Rs 36L.',
        columns: ['VIX rank', 'n (his)', 'n (ours)', 'Win% his', 'Win% ours', 'Avg premium', 'Net/trade', 't', 'CAGR', 'Max DD', 'Calmar'],
        rows: [
          ['No filter', '83', '89', '69.9%', '70.8%', '786.3', '78.0', '3.12', '11.47%', '13.8%', '0.83'],
          ['> 25', '55', '61', '74.5%', '72.1%', '857.5', '103.6', '3.55', '10.72%', '10.2%', '1.05'],
          ['> 50', '39', '42', '76.9%', '71.4%', '919.8', '106.1', '2.77', '8.22%', '11.7%', '0.70'],
          ['> 75', '21', '21', '85.7%', '71.4%', '1,052.9', '172.0', '2.71', '6.95%', '8.0%', '0.87'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Why the VIX filter works - movement relative to premium',
        caption: 'Breakeven width = entry credit / entry spot, known at entry. It is the denominator that matters, not the move alone.',
        columns: ['Predictor', 'Correlation with net P&L'],
        rows: [
          ['|move| (entry spot -> exit spot)', '-0.770'],
          ['|move| / breakeven width', '-0.898'],
          ['breakeven width alone', '+0.330'],
        ],
        highlightRows: [1],
      },
      {
        title: 'By expiry month - and why December is the only loser',
        caption: 'June has the LARGEST average move (3.84%) and is still 3rd best, because it is paid 5.72% of breakeven. December is paid the least of any month (3.75%) while drifting up +1.94%. Dec total is also one trade: ex Dec-2023 it averages +75.3.',
        columns: ['Month', 'n', 'Avg pts', 'Avg move', '|move|', 'Breakeven', 'move/BE', 'Win%'],
        rows: [
          ['Jan', '7', '+7.2', '+0.62%', '2.65%', '3.83%', '0.67', '57%'],
          ['Feb', '8', '+182.1', '-0.01%', '2.10%', '4.43%', '0.46', '100%'],
          ['Mar', '7', '+39.9', '-1.02%', '2.62%', '4.47%', '0.65', '71%'],
          ['Apr', '8', '+230.7', '+1.74%', '2.09%', '6.55%', '0.44', '75%'],
          ['May', '8', '+57.0', '+0.58%', '3.19%', '5.90%', '0.62', '62%'],
          ['Jun', '8', '+169.8', '+3.24%', '3.84%', '5.72%', '0.61', '62%'],
          ['Jul', '8', '+10.3', '+3.29%', '3.43%', '4.67%', '0.70', '62%'],
          ['Aug', '7', '+12.0', '+0.74%', '3.46%', '4.07%', '0.85', '57%'],
          ['Sep', '7', '+70.0', '+1.59%', '2.26%', '4.05%', '0.60', '71%'],
          ['Oct', '7', '+101.5', '+0.13%', '2.17%', '3.78%', '0.56', '86%'],
          ['Nov', '7', '+71.2', '+1.13%', '2.72%', '4.22%', '0.64', '71%'],
          ['Dec', '7', '-51.4', '+1.94%', '2.95%', '3.75%', '0.80', '71%'],
        ],
        highlightRows: [11],
      },
      {
        title: 'Breakeven-width filter vs the VIX filter',
        caption: 'Monotonic in per-trade terms, so the mechanism is real - but it does not beat VIX>25 on capital, and stacking them is worse than either alone.',
        columns: ['Filter', 'n', 'Avg pts', 't', 'CAGR', 'Max DD', 'Calmar', 'Losing years'],
        rows: [
          ['None', '89', '78.1', '3.03', '11.49%', '18.0%', '0.64', '1'],
          ['BE >= 3.5%', '67', '91.1', '3.26', '10.45%', '10.2%', '1.03', '0'],
          ['BE >= 4.0%', '49', '105.3', '3.03', '9.20%', '8.9%', '1.04', '1'],
          ['BE >= 5.0%', '26', '113.4', '2.13', '5.87%', '11.0%', '0.53', '2'],
          ['BE >= 6.0%', '13', '181.4', '2.34', '4.86%', '6.2%', '0.79', '1'],
          ['VIX > 25', '61', '104.5', '3.54', '10.78%', '10.2%', '1.06', '0'],
          ['BE >= 4.0% AND VIX > 25', '46', '109.3', '2.97', '9.02%', '8.9%', '1.02', '1'],
        ],
        highlightRows: [5],
      },
      {
        title: 'Parameter sensitivity - 45 and 21 are both local maxima',
        caption: 'Data-snooping flag on the exact values; but the RISK gradient is cleanly monotonic, which is the part to trust.',
        columns: ['Entry DTE', 'Net/trade', 't', 'Max DD', 'Exit DTE', 'Net/trade', 't', 'Max DD'],
        rows: [
          ['40', '63.9', '2.69', '-709', '0 (expiry)', '60.6', '1.20', '-2,803'],
          ['45', '78.1', '3.03', '-998', '7', '69.3', '1.55', '-1,840'],
          ['50', '62.1', '2.05', '-1,421', '14', '69.1', '1.86', '-1,215'],
          ['60', '59.8', '1.33', '-3,144', '21', '78.1', '3.03', '-998'],
          ['-', '-', '-', '-', '28', '41.6', '2.19', '-1,057'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Cost sensitivity - friction is not what kills this',
        caption: 'Slippage per side, on top of STT, exchange charges and brokerage.',
        columns: ['Slippage per side', 'Net/trade', 'Total net pts', 't', 'Avg round-trip cost'],
        rows: [
          ['0% (gross)', '81.8', '7,283.7', '3.18', '1.8 pts'],
          ['0.25% (headline)', '78.1', '6,952.4', '3.03', '5.5 pts'],
          ['0.50%', '74.4', '6,621.1', '2.88', '9.2 pts'],
          ['1.00%', '66.9', '5,958.5', '2.59', '16.7 pts'],
          ['2.00%', '52.1', '4,633.3', '2.01', '31.6 pts'],
        ],
        highlightRows: [1],
      },
    ],

    results: {
      metrics: [
        { label: 'CAGR on Rs 36L', value: '11.47%', hint: 'NIFTY 11.60% same window', tone: 'pos' },
        { label: 'Max drawdown', value: '-13.8%', hint: 'Rs 4.97L - NIFTY -38.4%', tone: 'neg' },
        { label: 'Calmar', value: '0.83', hint: 'NIFTY 0.30', tone: 'pos' },
        { label: 'Net per trade', value: '+78.0 pts', hint: 'Rs 50,680 at 10 lots', tone: 'pos' },
        { label: 't-statistic', value: '3.12', hint: '89 independent trades', tone: 'pos' },
        { label: 'Total net', value: 'Rs 45.1L', hint: 'equity Rs 36L -> Rs 81.1L', tone: 'pos' },
        { label: 'Win rate', value: '70.8%', hint: 'his 69.9% - matches' },
        { label: 'Worst trade', value: '-Rs 3.76L', hint: '10.4% of capital', tone: 'neg' },
      ],
      tables: [
        {
          title: 'Year by year on Rs 36L blocked margin',
          caption: 'Hourly monitoring, net of cost, fixed 10 lots (the rule does not compound). Six positive years, one flat, one mildly negative - none worse than -3.2%.',
          columns: ['Year', 'Trades', 'Net pts', 'Net Rs', 'Return on Rs 36L', 'Intra-year DD', 'Win rate', 'Equity end'],
          rows: [
            ['2019', '12', '-178.0', '-1,15,727', '-3.2%', '-9.6%', '66.7%', 'Rs 34.84L'],
            ['2020', '12', '+807.6', '+5,24,960', '+14.6%', '-8.4%', '58.3%', 'Rs 40.09L'],
            ['2021', '12', '+928.3', '+6,03,413', '+16.8%', '-8.8%', '66.7%', 'Rs 46.13L'],
            ['2022', '12', '+1,039.4', '+6,75,582', '+18.8%', '-8.2%', '75.0%', 'Rs 52.88L'],
            ['2023', '12', '-7.1', '-4,603', '-0.1%', '-13.8%', '66.7%', 'Rs 52.84L'],
            ['2024', '12', '+971.2', '+6,31,287', '+17.5%', '-3.8%', '66.7%', 'Rs 59.15L'],
            ['2025', '12', '+1,915.5', '+12,45,082', '+34.6%', '-10.2%', '83.3%', 'Rs 71.60L'],
            ['2026 H1', '5', '+1,462.2', '+9,50,446', '+26.4%', '0.0%', '100%', 'Rs 81.10L'],
          ],
          heatmap: true,
        },
        {
          title: 'The same year by year, under each VIX filter',
          caption: 'Each variant starts from its own Rs 36L. Ret is on the fixed Rs 36L base; DD is the deepest intra-year drawdown; Equity compounds that variant’s own rupee flow. VIX >25 is the only variant with NO losing year; VIX >75 has two years with ZERO trades while Rs 36L sits blocked.',
          columns: ['Year',
                    'none n', 'none Ret', 'none DD', 'none Equity',
                    '>25 n', '>25 Ret', '>25 DD', '>25 Equity',
                    '>50 n', '>50 Ret', '>50 DD', '>50 Equity',
                    '>75 n', '>75 Ret', '>75 DD', '>75 Equity'],
          rows: [
            ['2019', '12', '-3.2%', '-9.6%', 'Rs 34.84L', '9', '+2.4%', '-4.8%', 'Rs 36.85L', '5', '-3.1%', '-6.2%', 'Rs 34.88L', '2', '-0.4%', '-1.5%', 'Rs 35.87L'],
            ['2020', '12', '+14.6%', '-8.4%', 'Rs 40.09L', '10', '+19.7%', '-7.8%', 'Rs 43.96L', '8', '+27.6%', '-6.2%', 'Rs 44.80L', '4', '+16.6%', '-6.2%', 'Rs 41.84L'],
            ['2021', '12', '+16.8%', '-8.8%', 'Rs 46.13L', '7', '+22.6%', '0.0%', 'Rs 52.10L', '3', '+10.3%', '0.0%', 'Rs 48.50L', '0 (idle)', '-', '-', 'Rs 41.84L'],
            ['2022', '12', '+18.8%', '-8.2%', 'Rs 52.88L', '10', '+13.5%', '-8.2%', 'Rs 56.97L', '8', '+6.9%', '-8.3%', 'Rs 51.00L', '4', '+14.0%', '0.0%', 'Rs 46.89L'],
            ['2023', '12', '-0.1%', '-13.8%', 'Rs 52.84L', '3', '+1.9%', '-3.4%', 'Rs 57.64L', '1', '-3.4%', '-3.4%', 'Rs 49.78L', '0 (idle)', '-', '-', 'Rs 46.89L'],
            ['2024', '12', '+17.5%', '-3.8%', 'Rs 59.15L', '10', '+12.9%', '-3.8%', 'Rs 62.29L', '9', '+15.0%', '-3.8%', 'Rs 55.18L', '5', '+9.5%', '-3.8%', 'Rs 50.31L'],
            ['2025', '12', '+34.6%', '-10.2%', 'Rs 71.60L', '7', '+14.7%', '-10.2%', 'Rs 67.59L', '4', '+5.1%', '-8.0%', 'Rs 57.00L', '3', '+3.9%', '-8.0%', 'Rs 51.71L'],
            ['2026 H1', '5', '+26.4%', '0.0%', 'Rs 81.10L', '5', '+26.4%', '0.0%', 'Rs 77.09L', '4', '+22.1%', '0.0%', 'Rs 64.97L', '3', '+21.6%', '0.0%', 'Rs 59.48L'],
            ['WHOLE PERIOD', '89', '11.47% CAGR', '13.8% MaxDD', 'Rs 81.10L', '61', '10.72%', '10.2%', 'Rs 77.09L', '42', '8.22%', '11.7%', 'Rs 64.97L', '21', '6.95%', '8.0%', 'Rs 59.48L'],
          ],
          highlightRows: [8],
        },
        {
          title: 'Whole-period summary by filter',
          caption: 'Rs 36L blocked. VIX >25 is the only variant that never had a losing year; >75 idles through two of the eight.',
          columns: ['Variant', 'Trades', 'Net Rs', 'CAGR', 'Max DD', 'Calmar', 'Equity end', 'Losing years'],
          rows: [
            ['No filter', '89', '+45,10,439', '11.47%', '-13.8%', '0.83', 'Rs 81.10L', '2'],
            ['VIX > 25', '61', '+41,09,314', '10.72%', '-10.2%', '1.05', 'Rs 77.09L', '0'],
            ['VIX > 50', '42', '+28,97,001', '8.22%', '-11.7%', '0.70', 'Rs 64.97L', '2'],
            ['VIX > 75', '21', '+23,48,240', '6.95%', '-8.0%', '0.87', 'Rs 59.48L', '1 (+2 idle)'],
            ['NIFTY 50 buy-and-hold', '-', '-', '11.60%', '-38.4%', '0.30', '-', '1'],
          ],
          highlightRows: [1],
        },
        {
          title: 'Against the benchmark',
          caption: 'Same window, 2019-01-14 to 2026-07-07. Effectively the same return as the index on a third of the drawdown.',
          columns: ['', 'CAGR', 'Max drawdown', 'Calmar'],
          rows: [
            ['45-DTE straddle, 10 lots on Rs 36L', '11.47%', '-13.8%', '0.83'],
            ['NIFTY 50 buy-and-hold (price index)', '11.60%', '-38.4%', '0.30'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Capital-buffer sensitivity',
          caption: 'How much margin you block changes the whole picture - and is the crux of the open stress-margin question.',
          columns: ['Margin blocked', 'CAGR', 'Max DD as % of capital', 'Calmar'],
          rows: [
            ['Rs 36L - Rs 3L/lot + 20% buffer', '11.47%', '13.8%', '0.83'],
            ['Rs 54L - 1.5x buffer', '8.46%', '9.2%', '0.92'],
            ['Rs 72L - 2x buffer', '6.72%', '6.9%', '0.97'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Including LIQUID1 on the pledged capital',
          caption: 'Whole capital pledged in LIQUID1 (Kotak Nifty 1D Rate Liquid ETF) except Rs 2L held in cash. 10 lots on Rs 36L = Rs 34L pledged. The book is idle 21% of the time unfiltered and 46% filtered, so the yield matters MORE to the filtered book and nearly erases the filter’s cost in return while keeping its far better drawdown.',
          columns: ['Basis', 'CAGR', 'Max DD', 'Calmar'],
          rows: [
            ['Unfiltered - options only', '11.49%', '-18.0%', '0.64'],
            ['Unfiltered + LIQUID1 @5%', '14.04%', '-18.0%', '0.78'],
            ['VIX rank>25 - options only', '10.79%', '-10.2%', '1.06'],
            ['VIX rank>25 + LIQUID1 @4%', '12.87%', '-10.2%', '1.26'],
            ['VIX rank>25 + LIQUID1 @5%', '13.44%', '-10.2%', '1.32'],
            ['VIX rank>25 + LIQUID1 @6%', '14.02%', '-10.2%', '1.38'],
            ['NIFTY 50 benchmark', '11.60%', '-38.4%', '0.30'],
          ],
          highlightRows: [4],
        },
      ],
    },

    winners: [
      {
        config: '60-minute exit checks, 10 lots on Rs 36L blocked margin (VIX > 25 for the risk-adjusted variant)',
        summary:
          'Hourly is the right clock and nothing faster adds anything - now proven on real 1-minute quotes, not just modelled. On capital the unfiltered book has the best CAGR and the VIX > 25 filter has the best Calmar; > 75 is worst of both.',
        metrics: [
          { k: 'CAGR / MaxDD / Calmar (no filter)', v: '11.47% / -13.8% / 0.83' },
          { k: 'CAGR / MaxDD / Calmar (VIX > 25)', v: '10.72% / -10.2% / 1.05' },
          { k: 'Benchmark NIFTY', v: '11.60% / -38.4% / 0.30' },
          { k: 'Net per trade', v: '78.0 pts (Rs 50,680)' },
          { k: 'Worst year', v: '-3.2% (2019)' },
        ],
        rejected: [
          'VIX > 75 - his headline cell. 21 trades in 7.5 years for a 6.95% CAGR, and its 85.7% win-rate claim does not reproduce (ours 71.4%).',
          '30 / 15 / 5-minute checks - identical to hourly to the decimal. Real 1-minute data shows why: in the DTE>=21 band, 0 of 60 sessions travelled >=50% from their close.',
          'Daily-close checks - same P&L but a 29% worse worst-trade and 23% worse drawdown than hourly. A free improvement declined.',
          'ALL delta management - every move threshold (1-5%), both arms (exit-only and re-centre), every re-entry cap and both trigger conventions. Best managed cell keeps 36% of the return; cutting on a move realises -28.6 pts where holding earns +83.0.',
          'Holding to expiry instead of exiting at 21 DTE - drawdown blows out from -998 to -2,803 pts. The real 1-minute DTE 0-2 row (travel up to 7,669% of close) is that gamma made visible.',
          'Notional-based sizing - the first version of this study used it and wrongly concluded ~7.8%/yr "below an index fund". A short straddle is margin-financed; blocked margin is the capital at risk.',
        ],
      },
    ],

    caveats: [
      'STRESS MARGIN IS THE OPEN ITEM. Rs 3L/lot is today’s margin with India VIX at 10.83. SPAN scales with volatility and VIX peaked at 83.61 on 2020-03-24, so a Rs 36L block would very likely have been breached in March 2020 - forcing a top-up or liquidation exactly when the book was losing. The fixed-capital CAGR does not model that. Until a margin-call-aware re-run exists, 11.47% is an UPPER BOUND.',
      'GAP RISK IS NOT IN THESE NUMBERS. The stop is evaluated on candle closes; a gap-open past 200% of credit fills wherever the market opens. The sample has 3 stop events and no overnight catastrophe. Short-straddle losses are left-skewed by construction and 89 trades cannot price that tail.',
      'CORRELATION, NOT DIVERSIFICATION. Another short-vol NIFTY position alongside THE STACK, the NAS book and the straddle paper books - they all lose in the same week. research/89 separately found the unconditional monthly NIFTY straddle net-negative over 2015-26; the 45->21 window plus a stop is what rescues it.',
      'No real intraday option data exists before 2026-04-20 and none can be obtained for expired contracts (Kite: "invalid token", tested). The 60m/30m/15m/5m sweep rows are reconstructed marks; the real 1-minute tables are the independent check on them, and they agree.',
      'The 1-minute recorder covers only DTE ~27 to 0, so the real-tick evidence covers the back third of the holding window across 3 real trades and 240 day-contracts - not a 45-DTE entry.',
      '45 and 21 DTE are each a local maximum in their own sweep - a data-snooping flag. Every neighbouring parameter is still profitable and the risk gradient is monotonic, so the SHAPE is trustworthy; the exact numbers are not special.',
      'Entry is priced at the bhavcopy close (~15:30); the video enters at 15:15. Four roll/price conventions tested - 75.5 to 82.2 pts/trade, verdict unchanged. Real 1-min quotes differ from the bhav close by at most 17.8 pts on ~900 (~2%).',
      'LIQUID1 YIELD IS PARAMETRIC, NOT MEASURED HISTORICALLY. LIQUID1’s own price history is too gappy to drive 2019-26 (56 bars in 2019, none 2020-22), and LIQUIDBEES - which does span the period - is the Rs 1000-pinned DIVIDEND model, so its price shows no yield at all. The 5% column is the central case against LIQUID1’s 5.11% measured today; 4% and 6% bracket the real Indian overnight range over the period. The LIQUID1 rows assume the capital is pledged as F&O collateral with Rs 2L kept in cash - subject to a ~10% haircut and SEBI’s 50:50 cash-to-collateral rule. At 3 lots: Rs 9.96L pledged (~Rs 8.96L collateral) + Rs 2L cash = ~Rs 10.96L usable against a measured Rs 5.99L requirement, so it clears comfortably - but the Rs 2L of true cash is only 33% of the requirement, so it relies on liquid ETFs counting as cash-equivalents. Confirm with the broker. The options-only rows need no such assumption and are the floor.',
      'The benchmark is the NIFTY 50 PRICE index and excludes dividends (~1.2%/yr), so the index is modestly better on total return than the table shows - and still far behind on risk.',
      'One expiry (March 2026) is absent because NSE re-dated the contract, leaving it unlisted 45 days before its stated expiry. 89 of a possible 90 monthlies traded.',
    ],

    githubLinks: [
      { label: 'research/119 - STATUS doc', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/119_45dte_short_straddle' },
      { label: 'RESULTS.md', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/119_45dte_short_straddle/results/RESULTS.md' },
    ],
    projectPaths: [
      'research/119_45dte_short_straddle/NIFTY_45DTE_SHORT_STRADDLE_MULTITF_BACKTEST_STATUS.md',
      'research/119_45dte_short_straddle/scripts/engine45.py',
      'research/119_45dte_short_straddle/scripts/run_phase_a.py',
      'research/119_45dte_short_straddle/scripts/run_phase_bc.py',
      'research/119_45dte_short_straddle/scripts/run_phase_d_intraday.py',
      'research/119_45dte_short_straddle/scripts/diag_convention.py',
      'research/119_45dte_short_straddle/scripts/diag_touch.py',
      'research/119_45dte_short_straddle/scripts/run_phase_e_recentre.py',
      'research/119_45dte_short_straddle/scripts/run_phase_e2_diag.py',
      'research/119_45dte_short_straddle/results/RESULTS.md',
    ],
  },

  {
    slug: 'momentum-put-hedge-overlay',
    title: 'Put-Hedge Overlay vs the Cash-Exit Gate — can options replace selling the book?',
    verdict:
      'At a weekly risk-off gate the live book sells all 8 stocks to cash, realizing short-term capital-gains tax and taking re-entry timing risk. Could buying NIFTY puts instead — staying invested and hedging — do better after tax? Answer: a WEEKLY put hedge is modestly better (net Calmar 1.39 vs the 1.32 cash-exit baseline, +7pp net CAGR, 2019–2026), but a MONTHLY hedge fails in every window tested (0.61 full-cycle vs 0.96), and weekly options do not exist before 2019 — so the winning arm has NEVER faced a grinding bear. Verdict: SIGNAL, not deployable. Two further results are firm: NO trailing exit (SuperTrend across six configurations, EMA crosses, premium give-back) beats the plain gate-reversal exit; and the macro GATE, not the Donchian stop, is what prevents deep drawdowns — the same book without the cash exit falls −38.6% versus −16.6% with it, despite identical per-stock stops. This study also documents three modelling bugs found and fixed, which is the main reason it is published.',
    status: 'COMPLETE',
    date: '2026-08-07',
    cardBlurb:
      'Can buying NIFTY puts replace selling the book at a risk-off gate (and dodge the STCG bill)? Weekly puts edge the baseline (net Calmar 1.39 vs 1.32) but monthly puts fail everywhere, and weeklies did not exist before 2019 — so the winner has never seen a grinding bear. SIGNAL, not deployable. Firm side-findings: no trailing exit beats the plain gate exit, and the macro gate (not the Donchian stop) is what controls drawdown.',
    cardStats: [
      { label: 'Verdict', value: 'SIGNAL — not deployable' },
      { label: 'Weekly hedge net Calmar', value: '1.39 vs 1.32' },
      { label: 'Monthly hedge (full cycle)', value: '0.61 vs 0.96' },
    ],

    systemRules: {
      intro: 'The base book is unchanged; only what happens at a weekly risk-off gate differs.',
      sharedCoreTitle: 'The three arms',
      sharedCore: [
        { k: 'A0 — cash exit (current live)', v: 'Gate risk-off → liquidate all 8 stocks to cash; redeploy at the next rebalance once risk-on. Realizes STCG on every winner.' },
        { k: 'A1 — hold naked (control)', v: 'Stay fully invested, no hedge. Isolates what the gate is actually worth.' },
        { k: 'A2 — put hedge (the study)', v: 'Stay invested and buy NIFTY puts sized to ratio × equity. Swept structure (long put / bear put spread), moneyness (ITM2 / ATM / OTM2 / OTM5), tenor (weekly / monthly), ratio (0.5–2.5×) and exit rule.' },
        { k: 'Hedge sizing', v: 'units = ratio × equity ÷ NIFTY spot, RE-SIZED DAILY as stocks stop out (see caveats — the original fixed-size version was a bug).' },
        { k: 'Hedge exit', v: 'Primary: close when the gate flips risk-on. Roll to the next expiry while still risk-off. Trailing alternatives all tested and rejected.' },
        { k: 'Costs', v: 'Option prices are EOD closes with open interest > 0; 0.3% slippage on premium; equity 0.15%/leg; STCG 20% tracked.' },
      ],
      riskLayer: {
        title: 'Head-to-head, same 2019–2026 window (weekly options only exist from 2019)',
        caption: 'The weekly hedge beats the cash exit on after-tax risk-adjusted return; the monthly hedge does not. All hedged arms use the corrected daily re-sizing.',
        columns: ['Book', 'Net CAGR', 'Max DD', 'Net Calmar'],
        rows: [
          ['A0 cash exit (live baseline)', '31.3%', '−15.4%', '1.32'],
          ['A1 hold naked', '38.9%', '−23.2%', '1.11'],
          ['Monthly put hedge (r2.0)', '37.4%', '−18.3%', '1.24'],
          ['Weekly put hedge (r2.0)', '38.7%', '−17.0%', '1.39'],
        ],
        highlightRows: [3],
      },
    },

    system: {
      intro: 'Tested on real NIFTY option EOD closes from nse_options_bhav (monthly 2011→, weekly 2019→) over the live momentum book.',
      rows: [
        { k: 'Data', v: 'NIFTY options EOD (open interest > 0); NIFTYBEES for the gate; the research/62 momentum book for the equity leg.' },
        { k: 'Funding', v: 'Premium paid from idle cash; if short, holdings are trimmed pro-rata — the book runs ~fully invested, so a hedge must be funded honestly.' },
        { k: 'Windows', v: 'Monthly-tenor arms run the full 2011–2026 cycle (5 risk-off episodes); weekly-tenor arms only 2019–2026.' },
      ],
    },

    conditions: {
      intro: 'Robustness controls and the bugs found.',
      rows: [
        { k: 'Look-ahead', v: 'None — the gate, hedge entry, rolls and exits all use data at or before each decision date; option marks are the held contracts’ own closes.' },
        { k: 'BUG FIXED — hedge over-sizing', v: 'Hedge units were originally fixed at entry, so when stocks stopped out the put notional became 3–4× the remaining book (an unintended naked short). With daily re-sizing the weekly result fell from net Calmar 1.66 to 1.39 — a large part of the original "win" was that bug.' },
        { k: 'BUG FIXED — SuperTrend', v: 'The first implementation could never turn bearish (100% bullish on every day, all six parameter sets), so every earlier SuperTrend arm placed zero hedges. Rebuilt correctly (55–63% bullish, 19–45 flips) — and then genuinely adds no value.' },
        { k: 'BUG FIXED — trail re-entry', v: 'A trail exit originally blocked re-entry for the whole risk-off episode, which made any early-firing trail look identical to no hedge. Replaced with a unified want-hedge rule allowing natural re-entry.' },
        { k: 'Regime gap', v: 'Weekly options begin in 2019, so the winning arm covers one V-shaped crash (COVID) and a strong bull — never a grinding bear like 2015–16, which is exactly where the monthly hedge failed worst.' },
        { k: 'Lot granularity', v: 'Sizing is modelled continuously. At a ₹20L book one NIFTY lot ≈ ₹18L notional, so real ratios come in ~0.9× steps — a 2.0× hedge is ~2 lots and fine-tuning between 1.5× and 2.0× is not possible at that size.' },
      ],
    },

    comparisons: [
      {
        title: 'Full cycle 2011–2026 — the monthly hedge fails badly',
        caption: 'The only tenor testable across five risk-off episodes loses decisively to simply going to cash.',
        columns: ['Book', 'Net CAGR', 'Max DD', 'Net Calmar'],
        rows: [
          ['A0 cash exit (live)', '29.4%', '−16.6%', '0.96'],
          ['A1 hold naked', '30.7%', '−38.6%', '0.57'],
          ['Monthly hedge r1.0 (re-sized)', '30.7%', '−34.4%', '0.63'],
          ['Monthly hedge r2.0 (re-sized)', '30.6%', '−30.2%', '0.61'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Trailing exits — none beats the plain gate-reversal exit',
        caption: 'Tested after fixing both the SuperTrend and the re-entry logic. Only the premium give-back edges the baseline, and it does so with a materially worse drawdown.',
        columns: ['Exit rule', 'Net CAGR', 'Max DD', 'Net Calmar'],
        rows: [
          ['Plain gate reversal (baseline cash exit)', '31.3%', '−15.4%', '1.32'],
          ['SuperTrend (14,3) — best of six', '38.6%', '−18.9%', '1.26'],
          ['EMA-50 cross', '35.8%', '−20.3%', '1.24'],
          ['Premium give-back 50%', '38.1%', '−21.9%', '1.37'],
        ],
        highlightRows: [0],
      },
      {
        title: 'The gate — not the Donchian stop — is what controls drawdown',
        caption: 'Both arms carry the identical per-stock 15-day Donchian stop. Removing only the cash exit more than doubles the drawdown, because the book keeps re-entering a falling market every month.',
        columns: ['Book (2011–2026)', 'Max DD', 'Net Calmar'],
        rows: [
          ['With the cash gate (live)', '−16.6%', '0.96'],
          ['Without the cash gate (naked)', '−38.6%', '0.57'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Why the index hedge cannot cover the book',
        caption: 'A NIFTY-notional put only neutralises the index component; the 8-stock book falls roughly twice as far. Raising the ratio cuts drawdown but the premium bill rises faster than the protection.',
        columns: ['Hedge ratio (monthly, full cycle)', 'Max DD', 'Net Calmar'],
        rows: [
          ['1.0×', '−36.1%', '0.60'],
          ['2.0×', '−33.8%', '0.59'],
          ['2.5×', '−32.6%', '0.56'],
        ],
        highlightRows: [0],
      },
    ],

    results: {
      metrics: [
        { label: 'Verdict', value: 'SIGNAL — not deployable', tone: 'neg' },
        { label: 'Weekly hedge (2019–26) net Calmar', value: '1.39 vs 1.32 baseline' },
        { label: 'Monthly hedge (full cycle)', value: '0.61 vs 0.96 baseline', tone: 'neg' },
        { label: 'Best trailing exit', value: 'none beat the gate exit' },
        { label: 'Gate vs no gate (drawdown)', value: '−16.6% vs −38.6%' },
        { label: 'Modelling bugs found & fixed', value: '3' },
      ],
      tables: [],
      embeds: [
        { src: '/app/hedge_viz_tearsheet.html', height: 2450,
          caption: 'Interactive: every put purchase marked on the equity curve (green = profitable, red = loss, hover for strike/premium/P&L), risk-off regimes shaded, the hedge value held over time, cumulative premium spent vs cumulative hedge P&L, and the full 115-trade blotter. The puts lost money net — the hedged book only edged the baseline by staying invested.' },
      ],
    },

    winners: [
      {
        config: 'None deployable — keep the cash-exit gate',
        summary: 'The weekly put hedge is a genuine but modest signal (net Calmar 1.39 vs 1.32) that has never been tested through a grinding bear. Until weekly-option history covers such a regime, the plain cash exit remains the right rule.',
        metrics: [
          { k: 'Weekly hedge net Calmar', v: '1.39' },
          { k: 'Baseline net Calmar', v: '1.32' },
          { k: 'Monthly hedge (full cycle)', v: '0.61' },
          { k: 'Margin', v: 'thin, single regime' },
        ],
        rejected: [
          'Monthly put hedge — fails in every window tested (0.61 vs 0.96 full cycle)',
          'Hedge ratios above 1.0× — cut drawdown but premium rises faster than protection',
          'All trailing exits — SuperTrend (6 configs), EMA crosses, premium give-back',
          'Hybrid partial de-risk + hedge — dominated in both windows',
          'Single-stock put hedging — untestable: only 81 large-cap symbols have option data, and 14 of 15 typical momentum holdings have none',
        ],
      },
    ],

    caveats: [
      'The winning weekly arm covers 2019–2026 only — one V-shaped crash and a strong bull. It has never faced a grinding bear, which is precisely where the monthly hedge failed worst. This is why the verdict is SIGNAL rather than STRATEGY.',
      'Three modelling bugs were found and fixed during the study (hedge over-sizing, a SuperTrend that could never turn bearish, and trail re-entry blocking). Earlier figures quoted before those fixes were wrong and have been superseded.',
      'Hedge sizing is continuous in the model; NIFTY lot granularity (~₹18L notional per lot) makes fine ratio control impossible at a ₹20L book.',
      'All decisions and marks are end-of-day, matching how the book actually trades; intraday behaviour is not modelled.',
      'Single-stock put hedging could not be evaluated — it is blocked by option-data coverage and, structurally, by momentum picking mid-caps outside the F&O universe.',
    ],

    githubLinks: [
      { label: 'research/105 — put-hedge overlay', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/105_momentum_put_hedge' },
      { label: '← Related: Universe bake-off', href: '/app/backtest/momentum-universe-bakeoff' },
    ],
    projectPaths: [
      'research/105_momentum_put_hedge/scripts/run_hedge_sweep.py, run_hedge_g4.py (re-sizing fix), run_hedge_g5.py (fair trails)',
      'research/105_momentum_put_hedge/scripts/hedge_diag.py, run_hedge_g3.py; research/107_stock_put_hedge/ (blocked)',
    ],
  },
  {
    slug: 'momentum-universe-bakeoff',
    title: 'Momentum Book — Universe Bake-off (Nifty 200 vs 250 vs 51-250 vs Midcap 150 vs 500)',
    verdict:
      'Which universe should the momentum book select from? Five bands were tested over 2011-2026, each optimised on its OWN hold/buffer sizing so the comparison is best-versus-best. The answer: KEEP the live Nifty-200 book at 8/22/30. The durable finding is not a ranking but a mechanism — as large caps are stripped out, the drawdown deepens monotonically (Nifty 200 −17.0% → Nifty 250 −17.4% → 51-250 −25.5% → pure Midcap 150 −31.7%). Mega caps add little return but act as drawdown ballast. Pure midcap is the worst trade of all: no more return than Nifty 200 (27.8% vs 27.1%) with nearly double the drawdown. Nifty 250 edges the live book (Calmar 0.92 vs 0.91) but only within noise. Excluding the top 50 gives the highest return (31.0% vs 27.1% net CAGR) but at −25.5% drawdown — a real trade, not a free lunch. Nifty 500 is worse at every matched setting, and demanding more liquidity actively costs return (a ₹25cr ADV screen halves the CAGR), which is a capacity ceiling rather than a filter to apply today. A second robust lesson: optimal holdings scale with universe width (Nifty 200→8, 250→10, 500→16) — the live 8/22/30 is exactly right for Nifty 200.',
    status: 'COMPLETE',
    date: '2026-08-07',
    cardBlurb:
      'Five universes tested, each optimised on its own sizing. Keep Nifty 200 at 8/22/30. The real finding is a mechanism: strip out large caps and drawdown deepens monotonically (−17% → −25% → −32%) — mega caps are drawdown ballast. Pure midcap = same return, double the drawdown. Excluding the top 50 buys +3.9pp CAGR for a 50% deeper drawdown. Holdings must scale with universe width.',
    cardStats: [
      { label: 'Winner', value: 'Nifty 200 · 8/22/30' },
      { label: 'Net Calmar (live book)', value: '0.91' },
      { label: 'Pure-midcap drawdown', value: '−31.7%' },
    ],

    systemRules: {
      intro: 'Rules held constant across every universe; only the selection band and the hold/buffer sizing change.',
      sharedCoreTitle: 'Constant across all arms',
      sharedCore: [
        { k: 'Signal', v: '6m & 12m relative strength vs NIFTYBEES (rsblend), ranked within the universe band.' },
        { k: 'Hold / buffer', v: 'Top-N equal-weight with a top-B anti-churn buffer; swept per universe (6/16, 8/22, 10/28, 12/33, 16/44).' },
        { k: 'Stop', v: 'Per-stock 15-day Donchian EOD exit to cash; redeployed at the next rebalance.' },
        { k: 'Gate', v: 'Weekly — NIFTYBEES < 100-day SMA → liquidate all to cash.' },
        { k: 'Rebalance', v: 'Monthly, rotate-only (let winners run).' },
        { k: 'Costs', v: '0.15%/leg, cash 6.5%, STCG 20% tracked; results quoted NET of STCG. 2011–2026 daily-marked.' },
        { k: 'Universe bands', v: 'Point-in-time traded-value ranks (survivorship-free). Validated against real trackers: band returns correlate 0.94–0.99 with the real Nifty Smallcap 250 ETF.' },
      ],
      riskLayer: {
        title: 'Best config per universe (each optimised on its own sizing)',
        caption: 'Ranked by net Calmar. The monotonic drawdown progression as large caps are removed is the robust result; the 0.01 Calmar gaps at the top are noise.',
        columns: ['Universe', 'Best hold/buffer/pool', 'Net CAGR', 'Max DD', 'Net Calmar', 'Sharpe'],
        rows: [
          ['Nifty 250 (1–250, LargeMid)', '10/28/38', '28.0%', '−17.4%', '0.92', '1.81'],
          ['Nifty 200 (1–200) — LIVE', '8/22/30', '27.1%', '−17.0%', '0.91', '1.70'],
          ['Nifty 500 (1–500)', '16/44/60', '26.9%', '−18.9%', '0.82', '2.00'],
          ['Nifty 51–250 (ex top-50)', '8/22/30', '31.0%', '−25.5%', '0.75', '1.87'],
          ['Midcap 150 (101–250)', '8/22/30', '27.8%', '−31.7%', '0.59', '1.80'],
        ],
        highlightRows: [1],
      },
    },

    system: {
      intro: 'Built on the research/62 live-book engine with the universe band and sizing parameterised.',
      rows: [
        { k: 'Data', v: 'market_data.db daily close+volume; PIT top-N-by-traded-value bands; NIFTYBEES gate + benchmark.' },
        { k: 'Grid', v: '5 universes × 5 sizings = 25 configs, plus ADV-screen and cost-sensitivity arms.' },
        { k: 'Basis', v: 'Daily-marked NAV; CAGR/DD/Calmar quoted net of 20% STCG on gains realized under 365 days.' },
      ],
    },

    conditions: {
      intro: 'Robustness controls.',
      rows: [
        { k: 'Look-ahead', v: 'None — momentum and traded-value ranks use only data at or before each decision date.' },
        { k: 'Survivorship', v: 'Universe is a point-in-time traded-value band, not a current index list — free of survivorship bias.' },
        { k: 'Multiple testing', v: '25 in-sample configs; the per-universe "optima" carry selection bias. Trust the monotonic mechanisms (large-cap ballast; sizing scales with width), not the 0.01-Calmar rankings.' },
        { k: 'Proxy validity', v: 'Band proxies validated against the real Nifty Smallcap 250 ETF: monthly-return correlation 0.94–0.99. They do NOT reliably measure the small-minus-large SPREAD (68% agreement) — see the retracted regime finding below.' },
        { k: 'Capacity', v: 'A ₹25cr ADV liquidity screen halves the CAGR (31%→21%) — part of the edge is compensation for liquidity risk. This is a capacity ceiling at scale (roughly ₹3–5cr+), not a filter to apply at ₹20L.' },
      ],
    },

    comparisons: [
      {
        title: 'Sizing must scale with universe width',
        caption: 'Each universe has its own optimum. Wider universe → more holdings. The Nifty-200 control proves it is not simply "more names is better".',
        columns: ['Universe', 'hold 8', 'hold 10', 'hold 16', 'Best'],
        rows: [
          ['Nifty 200 (net Calmar)', '0.91', '0.87', '0.62', 'hold 8'],
          ['Nifty 250 (net Calmar)', '0.83', '0.92', '0.70', 'hold 10'],
          ['Nifty 500 (net Calmar)', '0.76', '0.75', '0.82', 'hold 16'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Liquidity screen and cost sensitivity (Nifty 200 vs 500)',
        caption: 'Demanding more liquidity costs return — the edge partly lives in less-liquid names. Nifty 500 is worse at every cost level (both lose ≈4pp going 0.15%→0.50%).',
        columns: ['Setting', 'N200 net CAGR', 'N500 net CAGR', 'N200 net Calmar', 'N500 net Calmar'],
        rows: [
          ['No ADV filter, 0.15% cost', '28.5%', '28.3%', '0.99', '0.86'],
          ['ADV ≥ ₹10cr', '25.9%', '22.4%', '0.85', '0.73'],
          ['ADV ≥ ₹25cr', '18.4%', '16.8%', '0.56', '0.51'],
          ['ADV ≥ ₹10cr, 0.50% cost', '21.5%', '17.9%', '0.56', '0.47'],
        ],
        highlightRows: [0],
      },
      {
        title: 'RESOLVED — no smallcap regime effect exists (tested on the real index)',
        caption: 'An earlier result (switch to Nifty 500 when smallcaps lead → Calmar 1.01) was retracted, then settled by downloading the REAL NIFTY Smallcap 250 index (2011–2026). On the true signal there is NO regime effect whatsoever: the two universes perform identically in both regimes. The original split was an artifact of a traded-value proxy that grades at only 69% agreement / +0.59 correlation over 122 months. Crucially this also proves the Nifty-200 book is NOT left behind in smallcap rallies — it earns 34.9% during smallcap-led months versus Nifty 500’s 34.8%, because the momentum ranking already rotates into the strongest mid-caps.',
        columns: ['Regime (real Nifty Smallcap 250 signal)', 'Months', 'N200 book', 'N500 book', 'Difference'],
        rows: [
          ['Smallcaps leading', '103', '34.9%', '34.8%', '−0.1%'],
          ['Largecaps leading', '83', '16.9%', '17.1%', '+0.2%'],
          ['Regime-switched book (real signal)', '—', '26.1% CAGR / Calmar 0.98', 'vs live 27.1% / 0.91', 'no gain'],
        ],
        highlightRows: [0],
      },
    ],

    results: {
      metrics: [
        { label: 'Recommended universe', value: 'Nifty 200 · 8/22/30', tone: 'pos' },
        { label: 'Net CAGR / Max DD', value: '27.1% / −17.0%' },
        { label: 'Net Calmar', value: '0.91' },
        { label: 'Highest-return band (51–250)', value: '31.0% but −25.5% DD' },
        { label: 'Pure midcap (worst risk-adj)', value: 'Calmar 0.59, −31.7% DD', tone: 'neg' },
        { label: 'Universes tested', value: '5 × 5 sizings' },
      ],
      tables: [],
      embeds: [
        { src: '/app/n200_vs_n500_tearsheet.html', height: 2100,
          caption: 'Interactive Nifty 200 vs Nifty 500 comparison with a selectable date range — growth curves, drawdown curves, year-by-year returns and a pre-tax / net-of-STCG toggle.' },
      ],
    },

    winners: [
      {
        config: 'Nifty 200, top-8 with top-22 buffer (the live book)',
        summary: 'No wider or narrower universe beat it on risk-adjusted terms. Mega caps earn their place as drawdown ballast, and holdings are correctly sized for a 200-name universe.',
        metrics: [
          { k: 'Net CAGR', v: '27.1%' },
          { k: 'Max DD', v: '−17.0%' },
          { k: 'Net Calmar', v: '0.91' },
          { k: 'Sharpe', v: '1.70' },
        ],
        rejected: [
          'Pure Midcap 150 — same return as Nifty 200 with nearly double the drawdown (Calmar 0.59)',
          'Nifty 500 — worse at every matched liquidity and cost setting',
          'Nifty 51–250 — higher return (31.0%) but −25.5% drawdown; only if you actively want that trade',
          'Nifty 250 — better on paper (0.92) but inside the noise band; not worth switching',
          'ADV liquidity screens — cost return at current book size; a capacity ceiling, not a filter',
        ],
      },
    ],

    caveats: [
      '25 in-sample configs — per-universe optima carry selection bias. The monotonic mechanisms are the trustworthy output, not the top-of-table ordering.',
      'Universe bands are traded-value proxies, not real index membership. They track band RETURNS well (0.94–0.99) but not the small-minus-large spread.',
      'The smallcap regime-switch result was retracted; redoing it properly needs real NIFTY Smallcap 250 index history.',
      'Capacity: the edge partly compensates liquidity risk, so returns should decay as the book scales past roughly ₹3–5cr into only-liquid names.',
      'Results are net of 20% STCG but pre-brokerage-nuance; real slippage on mid-caps may exceed the modelled 0.15%/leg.',
    ],

    githubLinks: [
      { label: 'research/106 — universe bake-off', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/106_nifty500_universe' },
      { label: '← Related: Momentum leverage frontier', href: '/app/backtest/momentum-250-leverage-frontier' },
    ],
    projectPaths: [
      'research/106_nifty500_universe/scripts/run_universe_sweep.py, run_n500_sizing.py, run_n500.py',
      'research/106_nifty500_universe/scripts/reconcile_smallcap.py, calibrate_smallcap.py, proxy_vs_real_bands.py',
    ],
  },
  {
    slug: 'momentum-250-leverage-frontier',
    title: 'Momentum-250 Leverage Frontier — how far can the LIVE momentum book be pushed for return?',
    verdict:
      'The question: can leverage raise the return of the book we actually run — relative-strength momentum, top-8 with a top-22 anti-churn buffer, a 15-day Donchian per-stock stop, and the NIFTYBEES>100-SMA index cash gate — without becoming ruinous? Answer over the full 2006–2026 cycle (through the 2008 crash), net of cost, financed at 10.5% MTF: yes, and this book levers BETTER than the plain top-8 version. The index gate keeps leverage survivable (0 margin calls in 20 years, at any leverage up to 2×), and the Donchian stop + buffer keep the drawdowns shallow, so the Calmar holds ~1.35–1.48 across ALL leverage instead of decaying. Static 1.3–1.6× lifts CAGR from 32.7% to 40.4–47.9%, at a proportional −28 to −35% drawdown. At 1.0× (own capital, no borrowing) it is 32.7% CAGR / −22% drawdown vs NIFTYBEES buy-hold 11.7% / −60%. Vol-targeting and deeper concentration (N<8) both reduce risk-adjusted return. This is the same momentum edge magnified, not new alpha.',
    status: 'COMPLETE',
    date: '2026-08-07',
    cardBlurb:
      'How far can the LIVE momentum book (top-8 + buffer + Donchian stop + index gate) be pushed for return? The gate makes leverage survivable — 0 margin calls in 20 years to 2× — and the per-stock stop keeps drawdowns shallow, so the Calmar holds ~1.4 all the way up. Full-cycle 2006–2026, MTF 10.5%: 1.0× = 32.7% / −22% DD; static 1.3–1.6× → 40.4–47.9% CAGR at −28 to −35% DD. Levers far better than the plain top-8 book (Calmar 1.35–1.48 vs decay).',
    cardStats: [
      { label: 'CAGR (1.6×, net, 10.5% MTF)', value: '47.9%' },
      { label: 'Calmar (1.3×)', value: '1.42' },
      { label: 'Margin calls / 20y', value: '0' },
    ],

    systemRules: {
      intro: 'The traded system — the existing live momentum-paper book. Relative-strength momentum on a large-cap universe, top-8 with an anti-churn buffer and a per-stock trailing stop, with the index cash gate as the master risk control. Leverage is optional and applied only while the gate is risk-on.',
      sharedCoreTitle: 'Live momentum book — locked rules',
      sharedCore: [
        { k: 'Universe', v: 'Official Nifty-200 (market-cap defined), rebuilt monthly; survivorship-free proxy back to 2006.' },
        { k: 'Signal', v: 'Rank by relative strength vs NIFTYBEES — 6-month & 12-month return ratio, blended. Hold the TOP 8 equal-weight.' },
        { k: 'Anti-churn buffer', v: 'Keep a holding while it stays inside the top-22; only rotate it out when it drops below rank 22. Reduces needless turnover.' },
        { k: 'Per-stock stop', v: '15-day Donchian: exit a name if it closes below its own prior-15-day low → to cash (parked at ~6.5% until the next rebalance). This is what keeps the drawdowns shallow.' },
        { k: 'Entry gate (master risk control)', v: 'Only hold stocks when NIFTYBEES > its 100-day SMA. Below it → sell everything to cash. Dodged −52% in 2008 and the 2020 crash.' },
        { k: 'Rebalance', v: 'Monthly, rotate-only — sell names that fell out of the buffer, buy new names from freed cash, and LET WINNERS RUN (no equal-weight trim → no needless STCG). ~12 rebalances/yr.' },
        { k: 'Leverage (optional)', v: 'Deploy lev×capital while the gate is risk-on; MTF ~10.5% (the realistic vehicle for a stock book; stock-futures ~8% only if all 8 names have liquid F&O). Applied ONLY in uptrends → 0 margin calls in 20y.' },
      ],
      riskLayer: {
        title: 'The leverage frontier — existing live book, net + 10.5% MTF, daily-marked, full cycle 2006–2026',
        caption: 'Pick your point. 1.0× is own-capital-only. The Donchian stop + buffer keep the Calmar high (1.35–1.48) even under leverage — far above the plain top-8 book (~0.9–1.1).',
        columns: ['Leverage', 'CAGR', 'Max DD', 'Sharpe', 'Calmar', 'Margin calls'],
        rows: [
          ['1.0× (own capital)', '32.7%', '−22.0%', '1.77', '1.48', '0'],
          ['1.3×', '40.4%', '−28.5%', '1.68', '1.42', '0'],
          ['1.6×', '47.9%', '−34.9%', '1.62', '1.37', '0'],
          ['2.0×', '57.6%', '−42.8%', '1.56', '1.35', '0'],
        ],
        highlightRows: [1, 2],
      },
    },

    system: {
      intro: 'Built on the research/62 momentum engine (the live-book rules), extended with leverage + a maintenance margin-call model, loaded back to 2006 for a full-cycle drawdown.',
      rows: [
        { k: 'Data', v: 'market_data.db daily close+volume; official Nifty-200 universe; NIFTYBEES for gate + benchmark. 2004→2026 loaded (2006 start after warmup).' },
        { k: 'Leverage model', v: 'Deploy lev×equity in the top-8 while gate on; borrowed cash accrues 10.5% p.a. MTF; maintenance margin call if own-equity / gross-notional < 25%. 0 calls fired in 20y.' },
        { k: 'P&L basis', v: 'Daily-marked NAV, net of 0.3% round-trip cost, cash yield 6.5%; CAGR/DD/Sharpe/Calmar on the NAV curve.' },
      ],
    },

    conditions: {
      intro: 'Robustness controls / caveats.',
      rows: [
        { k: 'Look-ahead', v: 'None — momentum from data ≤ decision date; gate uses NIFTYBEES ≤ date; monthly rebalance + daily Donchian at close.' },
        { k: 'Margin-call model', v: '25% maintenance on DAILY marks. A gapping crash (2020 gapped down hard) could force liquidation this daily model understates — real leverage carries gap risk beyond the backtest.' },
        { k: 'Financing', v: 'Modelled at 10.5% MTF (realistic for a rotating stock basket); 8% (stock-futures basis) shown as sensitivity but needs all 8 names to have liquid F&O every month, which mid-caps often lack.' },
        { k: 'Not new alpha', v: 'The SAME momentum edge magnified by leverage — amplifies both edge and risk. Universe is a survivorship-free Nifty-200 proxy, not the exact index.' },
        { k: 'Multiples are compounding fantasies', v: 'The 300×–12000× growth figures assume perfect reinvestment/capacity — trust CAGR/DD/Calmar, not the multiple.' },
      ],
    },

    comparisons: [
      {
        title: 'Borrow-rate sensitivity — 8% (stock-futures basis) vs 10.5% (MTF)',
        caption: 'Financing barely moves the frontier — leverage is only on in gate-on months and only on the borrowed portion.',
        columns: ['Leverage', 'CAGR @8%', 'CAGR @10.5%', 'Max DD', 'Calmar @10.5%'],
        rows: [
          ['1.0×', '32.7%', '32.7%', '−22%', '1.48'],
          ['1.3×', '40.6%', '40.4%', '−29%', '1.42'],
          ['1.6×', '48.4%', '47.9%', '−35%', '1.37'],
          ['2.0×', '58.7%', '57.6%', '−43%', '1.35'],
        ],
        highlightRows: [2],
      },
      {
        title: 'Existing book vs plain top-8 under leverage (why the stop + buffer matter)',
        caption: 'The live book’s per-stock Donchian stop + buffer keep drawdowns shallow, so its Calmar HOLDS under leverage while the plain book’s decays. Same period, same 10.5% MTF.',
        columns: ['Leverage', 'Existing Calmar', 'Plain top-8 Calmar'],
        rows: [
          ['1.0×', '1.48', '1.13'],
          ['1.3×', '1.42', '1.05'],
          ['1.6×', '1.37', '0.99'],
          ['2.0×', '1.35', '0.93'],
        ],
        highlightRows: [0, 1, 2, 3],
      },
      {
        title: 'Vol-targeting does NOT help (tested on the plain momentum variant; lesson transfers)',
        caption: 'Dynamic leverage from trailing vol lands below the static frontier at matched average leverage. The gate already handles the downside, and momentum’s biggest up-years are high-vol rallies — vol-targeting de-levers into the melt-ups.',
        columns: ['Approach', 'Avg lev', 'CAGR', 'Max DD', 'Calmar'],
        rows: [
          ['Static leverage 1.3×', '1.30', '41.4%', '−38.9%', '1.06'],
          ['Vol-target (vt25/lb60)', '1.14', '33.2%', '−39.6%', '0.84'],
          ['Static leverage 1.6×', '1.60', '48.7%', '−47.8%', '1.02'],
          ['Vol-target (vt35/lb20)', '1.60', '43.2%', '−52.3%', '0.83'],
        ],
        highlightRows: [0, 2],
      },
    ],

    results: {
      metrics: [
        { label: '1.6× · CAGR (net, 10.5% MTF)', value: '47.9%', tone: 'pos' },
        { label: '1.3× · Calmar', value: '1.42' },
        { label: 'Margin calls in 20 years', value: '0', tone: 'pos' },
        { label: '1.0× (own capital) CAGR / DD', value: '32.7% / −22%' },
        { label: 'NIFTYBEES B&H', value: '11.7% / −60% DD', tone: 'neg' },
        { label: 'Existing vs plain-book Calmar (1.0×)', value: '1.48 vs 1.13', tone: 'pos' },
      ],
      tables: [],
      embeds: [
        { src: '/app/momentum_leverage_tearsheet.html', height: 2350,
          caption: 'Interactive: growth-of-₹1 on a log scale for each leverage level vs NIFTYBEES, the drawdown curves, year-by-year returns, the leverage frontier, borrow-rate sensitivity, and the vol-targeting comparison — all on the existing live book (rsblend + buffer + Donchian), full cycle 2006–2026.' },
      ],
    },

    winners: [
      {
        config: 'The live momentum book, static 1.3–1.6× leverage (10.5% MTF)',
        summary: 'The index gate makes leverage survivable (0 margin calls / 20y); the Donchian stop + buffer keep it efficient (Calmar holds ~1.4). Static 1.3–1.6× is the return sweet spot. Keep leverage a simple fixed multiple — vol-targeting and deeper concentration both reduce risk-adjusted return.',
        metrics: [
          { k: 'CAGR (1.6×)', v: '47.9%' },
          { k: 'CAGR (1.3×)', v: '40.4%' },
          { k: 'Calmar (1.3×)', v: '1.42' },
          { k: 'Max DD (1.6×)', v: '−34.9%' },
        ],
        rejected: [
          'Vol-targeting — de-levers into momentum’s high-vol up-years; below the static frontier',
          'N<8 concentration — trades Calmar for noise',
          'Ungated leverage — would wipe out (the gate is what prevents margin calls)',
          'The plain top-8 book (no Donchian/buffer) — levers worse: Calmar decays 1.13→0.93 vs the live book’s 1.48→1.35',
        ],
      },
    ],

    caveats: [
      'The −35% drawdown at 1.6× is real, daily-marked through 2008, and psychologically hard to hold — leverage amplifies the pain as much as the gain.',
      'Margin-call model is a 25% maintenance floor on daily marks; a gapping crash (2020) could force liquidation this understates. Real leverage carries gap risk beyond the model.',
      'Financing modelled at 10.5% MTF (own-capital 1.0× has no borrow). Stock-futures (~8%) is cheaper but needs all 8 rotating names to have liquid F&O — often false for mid-caps.',
      'Same momentum edge magnified, not new alpha — and ~0.8-correlated with the running momentum-paper book itself, so it is a sizing decision, not a new sleeve.',
      'Universe is a survivorship-free Nifty-200 proxy, not the exact index; growth multiples assume idealised reinvestment/capacity — trust CAGR/DD/Calmar.',
    ],

    githubLinks: [
      { label: 'research/104 — momentum leverage', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/104_momentum_leverage' },
      { label: '← Related: Nifty-250 momentum (base book)', href: '/app/backtest/nifty250-momentum-video' },
    ],
    projectPaths: [
      'research/104_momentum_leverage/scripts/run_lev62.py (existing-book leverage + margin-call), run_notrim.py (let-winners-run), run_voltarget.py',
      'research/104_momentum_leverage/scripts/export_lev62_curves.py, build_lev_ts.py — built on the research/62 live-book engine',
    ],
  },
  {
    slug: 'premium-weekly-nifty-strangle',
    title: 'Premium-Based Weekly NIFTY Strangle — sell ₹20/leg, VIX≥15 + ATR<1.2% (IV>RV) edge',
    verdict:
      'Selecting strikes by TARGET PREMIUM (~₹20/leg) rather than %OTM, and sweeping tenor × management from scratch, the weekly cadence wins and a single clean edge survives: sell rich implied vol (VIX≥15) only when realized vol is calm (ATR<1.2%) — the IV>RV vol-risk-premium. That gate takes the weekly ₹20 strangle from Calmar 0.19 (all weeks) to 2.37 with the 6% liquid yield, cutting the drawdown from −₹7.8L to −₹1.05L (daily-close basis). The ATR threshold is WALK-FORWARD VALIDATED — each half of 2019–2026 independently picks 1.2% and it holds out-of-sample (OOS Calmar 2.84 and 0.81). Honest live sizing uses an INTRADAY 3× combined-premium stop (not the optimistic daily-close one): that roughly doubles the drawdown to −₹1.8L and lands the realistic book at CAGR ~10% / Calmar ~1.2 with liquid yield / 81% win on ₹20L (10 lots). Four independent tests agree the strategy is HOLD-not-defend: an ATR-breach exit, underlying-move stops, and rolling the winning side in ALL hurt; the only helpful adjustment is re-deploying a fresh ₹20 strangle after a stop.',
    status: 'COMPLETE',
    date: '2026-08-06',
    cardBlurb:
      'Strikes by premium (~₹20/leg), not %OTM. Unbiased tenor × management sweep → weekly wins, and one edge survives: VIX≥15 + ATR<1.2% (sell rich IV into calm RV). Walk-forward validated (OOS Calmar 2.84 / 0.81). Honest live-realistic book with an INTRADAY 3× stop: CAGR ~10%, Calmar ~1.2, −₹1.8L DD, 81% win on ₹20L (10 lots). Reacting to the move hurts; re-deploying premium is the only helpful adjustment.',
    cardStats: [
      { label: 'CAGR (net, +6% liquid)', value: '~10%' },
      { label: 'Calmar (live-realistic)', value: '~1.2' },
      { label: 'Max DD (₹20L, intraday stop)', value: '−₹1.8L' },
    ],

    systemRules: {
      intro: 'The recommended traded system — a low-utilization, high-selectivity vol-premium harvester. 10-lot weekly NIFTY strangle, strikes chosen by premium, only sold when IV is rich and RV is calm.',
      sharedCoreTitle: 'Premium weekly strangle — locked rules',
      sharedCore: [
        { k: 'Instrument', v: 'Sell the CE & PE nearest ₹20 premium EACH (not a fixed %OTM), 10 lots (qty 750). Enter ~4 trading days before the weekly expiry.' },
        { k: 'Entry gate (the edge)', v: 'ONLY enter when India VIX ≥ 15 AND NIFTY ATR(14) < 1.2% of spot at entry — rich implied vol into calm realized vol (IV>RV). ATR is an ENTRY filter only: once on, HOLD even if ATR later breaches 1.2% (reacting to it doubles the drawdown).' },
        { k: 'Exit — combined-premium stop', v: 'INTRADAY stop: buy back the strangle if the combined mark reaches 3× the credit collected (a blow-up stop, run as SL-M live). No profit target and no underlying-move stop — both hurt.' },
        { k: 'Adjustment — re-deploy only', v: 'After a 3× stop fires with days left to expiry, sell a FRESH ₹20 strangle re-centred on spot (up to 3×). This is the ONLY adjustment that helps — it recaptures premium. Do NOT roll the winning side toward the money (blows the drawdown out ~7×).' },
        { k: 'Capital & yield', v: '₹15–20L parked for 10 lots — sized to survive tested-side margin EXPANSION (a naked strangle’s margin can 2–3× mid-trade), not just entry margin (~₹9–13L). Idle ~85% of weeks → parked capital in 6% liquid adds ~6% CAGR.' },
      ],
      riskLayer: {
        title: 'The four books — base vs +6% liquid yield (10 lots, ₹20L, 2019–2026)',
        caption: 'Raw all-weeks → + VIX≥15 → + ATR<1.2% → final live-realistic (intraday 3× stop + re-deploy). The VIX gate earns the most rupees; the ATR gate buys the smoothest ride; the intraday stop is the honest drawdown.',
        columns: ['Book', 'CAGR (base → +liquid)', 'Calmar (base → +liquid)', 'Max DD', 'Win'],
        rows: [
          ['Weekly ₹20 · all weeks (385 wks)', '7.5% → 13.5%', '0.19 → 0.37', '−₹7.78L', '78%'],
          ['+ VIX≥15 (204 wks)', '9.4% → 15.4%', '0.53 → 1.01', '−₹3.55L', '81%'],
          ['+ VIX≥15 + ATR<1.2% (57 wks, daily-close)', '5.3% → 11.3%', '1.01 → 2.37', '−₹1.05L', '84%'],
          ['Final · intraday 3× stop + re-deploy', '4.3% → 10.3%', '0.47 → 1.22', '−₹1.81L', '81%'],
        ],
        highlightRows: [3],
      },
    },

    system: {
      intro: 'Tested on our bhavcopy engine over real weekly NIFTY option premiums (2019–2026), strikes chosen by traded premium with open-interest > 0. Exits simulated on the option OHLC.',
      rows: [
        { k: 'Data', v: 'nse_options_bhav (NIFTY weekly, OHLC + OI); India VIX daily; NIFTY50 daily OHLC for ATR(14)/ADX(14)/CPR.' },
        { k: 'Strike selection', v: 'CE & PE nearest ₹20 close each with OI>0 (target-premium, not %OTM); combined credit marked daily.' },
        { k: 'Gates', v: 'India VIX ≥ 15 and Wilder ATR(14) < 1.2% of spot at entry (both causal). ADX and prior-day CPR filters tested and rejected (no help / hurt).' },
        { k: 'P&L basis', v: 'Net of 0.3%/leg slippage + ₹400/leg-transaction; 10 lots (qty 750); returns on ₹20L, with a 6% liquid-yield overlay on parked cash.' },
      ],
    },

    conditions: {
      intro: 'Robustness controls / the seven deadly sins.',
      rows: [
        { k: 'Look-ahead', v: 'None — strikes from the entry-day traded premium; VIX/ATR/CPR use only data at or before entry; exits use the held options’ own OHLC.' },
        { k: 'Overfitting / walk-forward', v: 'The ATR<1.2% threshold is walk-forward validated: split 2019–2026 in half, each half INDEPENDENTLY picks 1.2% by Calmar and it holds out-of-sample (train 2019–22 → OOS 2023–26 Calmar 2.84, 88% win; train 2023–26 → OOS 2019–22 Calmar 0.81, 82% win).' },
        { k: 'Stops are modelled intraday', v: 'The headline book uses an INTRADAY 3× stop (CE_high+PE_high trigger, fill at 3× or the gap-open) — a conservative over-count of breaches; the true figure sits between the intraday (Calmar +liq 1.20) and daily-close (3.11) books, ~1.2–2.0.' },
        { k: 'Cost neglect', v: 'Net of slippage + per-leg transaction cost on every open, close, and re-deploy; the 6% liquid yield is an explicit toggle, not baked in.' },
        { k: 'Capacity / margin', v: 'Far-OTM weekly NIFTY strangle is liquid; sizing note: park ₹15–20L for 10 lots because tested-side margin expands 2–3× mid-trade — tight-margin sizing risks a forced close.' },
        { k: 'VIX window', v: 'Options weekly data begins 2019, so the study runs 2019–2026 (~7.4 yrs); one COVID-2020 shock is included and the intraday stop is what carries it.' },
      ],
    },

    comparisons: [
      {
        title: 'Adjustment / exit bake-off (weekly ₹20, VIX≥15 + ATR<1.2%, daily-close, +6% liquid)',
        caption: 'HOLD-not-defend wins. Re-deploying premium after a stop is the only helpful adjustment; every directional reaction hurts, and rolling the winning side toward the money is a disaster.',
        columns: ['Method', 'Base total', 'Calmar +liq', 'Max DD +liq', 'Win'],
        rows: [
          ['Re-deploy fresh strangle after SL', '+₹7.85L', '3.11', '−₹0.72L', '84%'],
          ['Roll-recenter on leg-2×', '+₹6.0L', '2.54', '−₹0.78L', '75%'],
          ['Baseline — hold + premium-3× stop', '+₹7.8L', '2.33', '−₹0.95L', '84%'],
          ['Underlying-move stop 2%', '+₹5.3L', '1.83', '−₹1.04L', '79%'],
          ['Roll-untested-side IN on 2×', '+₹1.0L', '0.21', '−₹6.40L', '80%'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Intraday stop vs daily-close stop (the honesty check)',
        caption: 'A real intraday SL-M catches the whipsaw days the daily-close backtest quietly dodged — drawdown more than doubles and Calmar drops. The intraday figure is the one to trust.',
        columns: ['Stop basis', 'P&L (10 lots)', 'Win', 'Max DD', 'Calmar +liq'],
        rows: [
          ['Daily-CLOSE 3× + re-deploy (optimistic)', '+₹7.85L', '84%', '−₹0.82L', '3.11'],
          ['INTRADAY 3× + re-deploy (live-realistic)', '+₹6.22L', '80%', '−₹1.81L', '1.20'],
        ],
        highlightRows: [1],
      },
      {
        title: 'CAGR by capital base (intraday-stop book)',
        caption: 'The absolute edge is ₹1.05L/yr; the CAGR depends on what you divide by. Broker margin for a 10-lot far-OTM strangle is ~₹9–13L, but park ₹15–20L to survive tested-side margin expansion.',
        columns: ['Capital parked', 'Strangle-only CAGR', '+6% liquid CAGR', 'Calmar +liq'],
        rows: [
          ['₹20L (conservative)', '4.1%', '10.1%', '1.20'],
          ['₹15L (margin + buffer)', '5.5%', '11.5%', '1.01'],
          ['₹12L (~est 10-lot margin)', '6.8%', '12.8%', '0.89'],
          ['₹10L (tight margin)', '8.2%', '14.2%', '0.82'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Tenor comparison (premium strangle, VIX≥15 + premium-3× stop)',
        caption: 'Weekly is the best risk-adjusted tenor; monthly earns the most rupees but with a far deeper drawdown. Bi-weekly sits in between.',
        columns: ['Tenor', 'Total', 'Calmar', 'Max DD'],
        rows: [
          ['Weekly ₹20', '+₹8.0L', '0.44', '−₹4.1L'],
          ['Bi-weekly ₹20', '+₹15L', '0.31', '−₹9L'],
          ['Monthly ₹20', '+₹33.3L', '0.31', '−₹14.3L'],
        ],
        highlightRows: [0],
      },
    ],

    results: {
      metrics: [
        { label: 'Final book · CAGR (+6% liquid)', value: '~10%', tone: 'pos' },
        { label: 'Calmar (live-realistic, +liquid)', value: '~1.2' },
        { label: 'Max Drawdown (intraday stop)', value: '−₹1.81L', tone: 'neg' },
        { label: 'Win rate', value: '81%' },
        { label: 'ATR gate walk-forward OOS Calmar', value: '2.84 / 0.81', tone: 'pos' },
        { label: 'Daily-close Calmar (optimistic)', value: '3.11 vs 1.20 intraday' },
      ],
      tables: [],
      embeds: [
        { src: '/app/premium_strangle_tearsheet.html', height: 2450,
          caption: 'Interactive: four books (raw all-weeks / +VIX≥15 / +ATR<1.2% / final intraday+re-deploy) with a 6%-liquid-yield toggle — KPIs, cumulative P&L, drawdown, year-by-year, a ₹ monthly heatmap, month-on-month running drawdown, and the 57-trade blotter with entry VIX/ATR and exit reasons.' },
      ],
    },

    winners: [
      {
        config: 'Weekly ₹20 strangle — VIX≥15 + ATR<1.2% + intraday 3× stop + re-deploy (+ 6% liquid yield)',
        summary: 'Sell rich IV into calm RV, hold through the move, and re-deploy premium after a stop rather than defending directionally. A selective (7.5 trades/yr) vol-premium harvester whose virtues are safety and the liquid-yield leverage on idle capital.',
        metrics: [
          { k: 'CAGR (+liquid)', v: '~10%' },
          { k: 'Calmar (+liquid)', v: '~1.2' },
          { k: 'Max DD', v: '−₹1.81L' },
          { k: 'Win', v: '81%' },
        ],
        rejected: [
          'Reacting to an ATR breach mid-hold — Calmar 2.33→1.89, drawdown doubles',
          'Underlying-move stops (1–2%) — all worse; tighter is worse (Calmar 1.05 at 1%)',
          'Rolling the winning side toward the money on a leg-double — Calmar 0.21, −₹6.4L drawdown',
          'ADX and prior-day CPR entry filters — no help / hurt on the weekly book',
          'Daily-close stop as a live assumption — optimistic; a real intraday stop nearly doubles the drawdown',
        ],
      },
    ],

    caveats: [
      'The headline book uses an INTRADAY 3× stop modelled from option daily high/low (CE_high+PE_high over-counts simultaneous peaks) — a conservative bound; the true Calmar sits between 1.20 (intraday) and 3.11 (daily-close), ~1.2–2.0. Nothing is wired to live orders.',
      'Options weekly data begins 2019, so the study is ~7.4 years with a single COVID-2020 shock; the intraday stop is what carries that event.',
      'Broker margin for a 10-lot far-OTM strangle is an estimate (~₹9–13L; the Kite margin API was not reachable this session). Size ₹15–20L parked because a tested strangle’s margin expands 2–3× mid-trade.',
      'The 6% liquid yield is a modelled overlay on parked capital (the book is idle ~85% of weeks); real pledge haircuts and liquid-fund yields vary — treat it as indicative.',
      'The absolute edge is modest (~₹1.05L/yr at 10 lots) — this is a low-utilization safety book; scale lots or combine tenors to raise rupee return. A robust risk structure, not a guaranteed edge.',
    ],

    githubLinks: [
      { label: 'research/102 — premium strangle', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/102_premium_strangle' },
      { label: '← Related: Managed monthly straddle', href: '/app/backtest/cushioned-monthly-nifty-straddle' },
    ],
    projectPaths: [
      'research/102_premium_strangle/scripts/multi_tenor_premium.py, atr_validate.py, atr_dynamic_exit.py',
      'research/102_premium_strangle/scripts/adjust_sweep.py, intraday_stop.py, export_premium.py, build_premium_ts.py',
    ],
  },
  {
    slug: 'cushioned-monthly-nifty-straddle',
    title: 'Managed Monthly NIFTY Straddle — VIX gate + premium blow-up stop (cushion & 6% liquid optional)',
    verdict:
      'On 15 years of real premiums, the best short-straddle book is not a naked straddle with a cushion — it is a DISCIPLINED one. A VIX≥15 entry gate + 40% profit-target + a combined-premium 2.5× blow-up stop + close-by-DTE-5 takes the 10-lot DTE-28 monthly straddle from Calmar 0.28 (raw) to ~1.0, cutting the max drawdown from −₹11.7L to −₹3.7L. The VIX gate is the dominant lever — Calmar collapses ~4× without it (best 1.06 → 0.27); management can’t rescue a bad entry. A prior-day CPR<0.10% compression filter lifts it further (Calmar ~1.9). The BankNifty long-vol cushion still helps — on a like-for-like calendar basis it lifts Calmar (0.98→1.21 with the 6% liquid yield) and cuts the drawdown ~26% for a ~9% return give-up. And parking idle/pledged capital in 6% liquid funds adds ~2.5% CAGR — the VIX gate’s flat months aren’t wasted. Recommended book: ≈20% CAGR, Calmar ≈1.2, −₹3.7L max drawdown on ₹20L (10 lots), net of costs, with 6% liquid yield.',
    status: 'COMPLETE',
    date: '2026-08-05',
    cardBlurb:
      'The disciplined short-straddle. 15yr/5-crash-validated: VIX≥15 gate + 40% PT + premium-2.5× blow-up stop + DTE-5 close turns a raw DTE-28 monthly straddle from Calmar 0.28 → ~1.0 and −₹11.7L drawdown → −₹3.7L. The VIX gate is the whole game; the BankNifty cushion is optional; 6% liquid yield on idle/pledged capital adds ~2.5% CAGR → ≈20% CAGR, Calmar ≈1.2.',
    cardStats: [
      { label: 'CAGR (net, +6% liquid)', value: '~20%' },
      { label: 'Calmar', value: '~1.2' },
      { label: 'Max DD (₹20L)', value: '−₹3.7L' },
    ],

    systemRules: {
      intro: 'The recommended traded system. 10-lot DTE-28 monthly NIFTY straddle, but only sold with discipline and closed on rules. All levers checked at the daily close.',
      sharedCoreTitle: 'Managed monthly straddle — locked rules',
      sharedCore: [
        { k: 'Instrument', v: 'Sell ATM NIFTY monthly straddle (CE+PE), 10 lots. Enter ~28 calendar days before the monthly expiry, at the option OPEN (~09:20). ATM from the entry-day open (no look-ahead).' },
        { k: 'Entry gate (#1 lever)', v: 'ONLY enter when India VIX ≥ 15 at entry — sell rich premium, sit out low-vol months. Optional extra filter: skip when the prior-day daily CPR width < 0.10% of spot (compression precedes expansion).' },
        { k: 'Exit — whichever first', v: '(a) 40% profit-target: buy back when the straddle has decayed to 60% of the credit collected; (b) blow-up stop: exit if the straddle value rises to 2.5× the credit (a combined-premium loss stop); (c) close by DTE-5.' },
        { k: 'Cushion (recommended)', v: 'BankNifty long 1% strangle ×8 held to expiry — a genuine risk-adjusted gain: lifts Calmar (0.98→1.21 with liquid yield) and cuts the drawdown ~26% (−₹4.53L→−₹3.36L) for a ~9% return give-up. Drop only if you prefer more raw return and can hold the deeper drawdown.' },
        { k: 'Capital & yield', v: '₹20L for 10 lots. Meet margin 50% pledged stocks + ~₹7.5L pledged LIQUIDBEES (earns 6%); on flat (VIX<15) months the full cash sits in liquid @ 6% → ~+2.5% CAGR.' },
      ],
      riskLayer: {
        title: 'The four books — base vs +6% liquid yield (10 lots, ₹20L, 2016–2026)',
        caption: 'Raw straddle → + cushion → managed + cushion → same management with NO VIX gate. Management (green) dominates; dropping the VIX gate (last row) keeps return but halves Calmar.',
        columns: ['Book', 'CAGR (base → +liquid)', 'Calmar (base → +liquid)', 'Max DD'],
        rows: [
          ['Straddle alone', '16.6% → 18.8%', '0.28 → 0.34', '−₹11.67L'],
          ['+ BankNifty cushion', '17.7% → 20.0%', '0.50 → 0.63', '−₹7.16L'],
          ['Managed·VIX15 + cushion', '17.8% → 20.4%', '0.96 → 1.21', '−₹3.72L'],
          ['Managed·noVIX + cushion', '19.2% → 21.5%', '0.51 → 0.59', '−₹7.57L'],
        ],
        highlightRows: [2],
      },
    },

    system: {
      intro: 'Tested on our own bhavcopy engine over 15 years of real NIFTY + BankNifty option premiums (2011–2026; 2011–2016 downloaded for this study). Exits are simulated bar-by-bar on the daily close of the held straddle.',
      rows: [
        { k: 'Data', v: 'nse_options_bhav (NIFTY + BankNifty monthly 2011→2026); India VIX daily; index spot back to 2011.' },
        { k: 'Straddle', v: 'ATM CE+PE at DTE-28 entry-day open; the held straddle is marked every day (CE close + PE close) to trigger the PT / premium-stop / DTE-5 exits.' },
        { k: 'Gates', v: 'India VIX ≥ 15 at entry; optional prior-day CPR-width filter (causal — prior day’s H/L/C ÷ entry-open).' },
        { k: 'P&L basis', v: 'Net of 0.3%/leg + ₹160/RT; 10 lots; returns on ₹20L capital, with a 6% liquid-yield overlay on parked/pledged cash.' },
      ],
    },

    conditions: {
      intro: 'Robustness controls / the seven deadly sins.',
      rows: [
        { k: 'Look-ahead', v: 'None — ATM from entry-day open; every exit (PT / premium-2.5× / DTE-5) and both gates (VIX at entry, prior-day CPR) use only information available at or before the decision.' },
        { k: 'Regime / multi-crash', v: '15 years across 2011 EU, 2013 taper, 2015/16 China, 2018 vol, 2020 COVID — the drawdown control holds across all of them.' },
        { k: 'Overfitting', v: 'The premium-2.5× stop and VIX≥15 gate are broad, robust choices (leg-ratio stops were rejected; move-stops were second-best). CPR<0.10% matches V2’s out-of-sample-validated threshold. The aggressive prior-month CPR<1.2% (Calmar 3.44) is flagged as promising-but-needs-walk-forward.' },
        { k: 'Cost neglect', v: 'Net of slippage + brokerage on every leg; the 6% liquid yield is modelled explicitly and is a toggle, not baked in.' },
        { k: 'Stops are EOD', v: 'The premium/move stops are checked at the daily CLOSE; a true intraday stop would trigger sooner, so live fills may differ.' },
        { k: 'VIX data window', v: 'India VIX starts 2015, so the VIX-gated book effectively runs 2015–2026 (~67 trades) — part of the gate’s edge is also a cleaner window.' },
      ],
    },

    comparisons: [
      {
        title: 'Exit optimization — raw hold-to-expiry vs managed',
        caption: '10-lot DTE-28 monthly straddle, 2011–2026. Adding discipline 5×’s the Calmar and cuts the drawdown ~60%, at the same return.',
        columns: ['Config', 'Total', 'Calmar', 'Max DD', 'Win'],
        rows: [
          ['Baseline — hold to DTE-1, no management', '+₹27.3L', '0.18', '−₹11.0L', '60%'],
          ['+ VIX≥15 + 40% PT + move-4% stop + DTE-5', '+₹27.8L', '0.91', '−₹4.65L', '58%'],
          ['+ VIX≥15 + 40% PT + premium-2.5× stop + DTE-5', '+₹39.9L', '1.28', '−₹5.60L', '70%'],
        ],
        highlightRows: [2],
      },
      {
        title: 'Stop-type bake-off (on the VIX≥15 + 40% PT base)',
        caption: 'The combined-premium 2.5× blow-up stop wins — it lets winners run and only cuts genuine blow-ups. Leg-ratio stops fire on any drift and knife winners.',
        columns: ['Stop type', 'Total', 'Calmar', 'Max DD'],
        rows: [
          ['no stop', '+₹37.1L', '0.68', '−₹8.3L'],
          ['underlying move 4%', '+₹27.8L', '0.91', '−₹4.65L'],
          ['combined-premium 2.5×', '+₹39.7L', '1.06', '−₹5.69L'],
          ['leg-ratio 2× (one side doubles)', '+₹3.1L', '0.13', '−₹3.65L'],
        ],
        highlightRows: [2],
      },
      {
        title: 'The VIX gate is the whole game (same configs, with vs without)',
        caption: 'Without the VIX≥15 gate, every stop config collapses to ~Calmar 0.26. Discipline can’t fix a bad entry.',
        columns: ['Stop (40% PT + DTE-5)', 'Calmar WITH VIX≥15', 'Calmar NO gate'],
        rows: [
          ['no stop', '0.68', '0.26'],
          ['move 4%', '0.91', '0.17'],
          ['combined-premium 2.5×', '1.06', '0.27'],
        ],
        highlightRows: [2],
      },
      {
        title: 'CPR compression filter on the managed book',
        caption: 'Skipping compressed setups helps further; the prior-day CPR<0.10% is the robust pick (keeps 50 of 67 trades). Prior-month CPR<1.2% reaches Calmar 3.44 but halves activity → treat as promising, not yet trusted.',
        columns: ['CPR gate', 'n', 'Total', 'Calmar', 'Max DD'],
        rows: [
          ['managed, no CPR', '67', '+₹39.9L', '1.28', '−₹5.60L'],
          ['+ skip prior-day CPR < 0.10%', '50', '+₹36.2L', '1.92', '−₹4.53L'],
          ['+ skip prior-month CPR < 1.2% (aggressive)', '37', '+₹24.8L', '3.44', '−₹2.34L'],
        ],
        highlightRows: [1],
      },
    ],

    results: {
      metrics: [
        { label: 'Managed·VIX15 + cushion · CAGR (+6% liquid)', value: '20.4%', tone: 'pos' },
        { label: 'Calmar (+liquid)', value: '1.21' },
        { label: 'Max Drawdown', value: '−₹3.72L', tone: 'neg' },
        { label: 'Win rate', value: '63%' },
        { label: 'Managed alone (no cushion) · Calmar (+liquid)', value: '0.98' },
        { label: 'Cushion vs none · Calmar (+liquid)', value: '1.21 vs 0.98', tone: 'pos' },
      ],
      tables: [],
      embeds: [
        { src: '/app/managed_straddle_tearsheet.html', height: 2450,
          caption: 'Interactive: four books (raw / +cushion / managed+cushion / managed·noVIX) with a 6%-liquid-yield toggle — KPIs, cumulative P&L, drawdown, year-by-year, a ₹ monthly heatmap, month-on-month running drawdown, and the managed trade blotter with exit reasons.' },
      ],
    },

    winners: [
      {
        config: 'Managed monthly straddle — VIX≥15 + 40% PT + premium-2.5× stop + DTE-5 (+ 6% liquid yield)',
        summary: 'Disciplined entry and a wide blow-up stop turn a mediocre short-straddle into a Calmar ~1 book; parked capital at 6% adds ~2.5% CAGR; the cushion is optional once the stop caps the tail.',
        metrics: [
          { k: 'CAGR (+liquid)', v: '20.4%' },
          { k: 'Calmar (+liquid)', v: '1.21' },
          { k: 'Max DD', v: '−₹3.72L' },
          { k: '+ CPR<0.10% Calmar', v: '1.92' },
        ],
        rejected: [
          'Raw hold-to-expiry straddle — Calmar 0.18, −₹11L drawdown',
          'No VIX gate — every stop config collapses to ~Calmar 0.26 (the gate is the #1 lever)',
          'Leg-ratio stops — fire on any drift, knife winners (Calmar 0.13)',
          'Rolling weekly/monthly far-OTM wings for margin — cost 50–85% of the edge for only ~₹2L/10-lot margin relief',
        ],
      },
    ],

    caveats: [
      'The premium/move stops are checked at the daily CLOSE — a true intraday stop triggers sooner, so live fills may differ. Nothing is wired to live orders.',
      'India VIX data starts 2015, so the VIX-gated book runs effectively 2015–2026 (~67 trades); part of the gate’s benefit is also a cleaner, post-2015 window.',
      'The 6% liquid yield is a modelled overlay on parked/pledged capital (₹20L: 50% pledged stocks + ~₹7.5L pledged liquid + cash); real pledge haircuts and liquid-fund yields vary — treat ~2.5% CAGR as indicative.',
      'The aggressive prior-month CPR<1.2% (Calmar 3.44) throws away ~45% of trades and needs walk-forward before trusting; the prior-day CPR<0.10% is the robust choice.',
      'Sizing is 10 lots on a fixed 50-unit-per-lot basis for 15-year comparability; NIFTY’s real lot drifted — translate ₹ to your actual lot count. This is a robust risk structure, not a guaranteed edge.',
    ],

    githubLinks: [
      { label: 'research/101 — trend sleeve / straddle management', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/101_trend_sleeve' },
      { label: '← Related: NIFTY straddle look-ahead audit', href: '/app/backtest/nifty-straddle-lookahead-audit' },
    ],
    projectPaths: [
      'research/101_trend_sleeve/scripts/monthly_exit_opt.py, monthly_stop_compare.py, monthly_cpr_gate.py',
      'research/101_trend_sleeve/scripts/export_managed.py (tearsheet data), nse_fo_downloader.py (2011-2015 bhavcopy)',
    ],
  },

  {
    slug: 'managed-futures-trend-sleeve',
    title: 'Managed-Futures Trend Sleeve — diversified weekly Donchian (gold + Nasdaq + PSU)',
    verdict:
      'A multi-asset weekly-trend sleeve turns the single gold edge into a properly diversified managed-futures book. Equal-weight weekly Donchian-8 long-only across low-correlation legs — gold (GOLDBEES), Nasdaq (MOM100), Indian PSU (CPSEETF) — delivers Sharpe 1.41, CAGR 13.0%, Calmar 0.85, max drawdown −15% net over 11.5 years. That is the SAME return as gold-alone but markedly better risk-adjustment (Sharpe 1.07→1.41, DD −21%→−15%) purely from diversification: gold↔Nasdaq correlation is +0.00. Adding S&P (MASPTOP50), US FANG (MAFANG) and silver (shorter history) pushes it to Sharpe 1.49 / Calmar 0.91. 10 of 12 years green; the walk-forward OOS (2021H2→2026) STRENGTHENS to Sharpe 1.91 / CAGR 20% / −10% DD. It is uncorrelated with the Indian short-vol / equity books — a genuine independent alpha sleeve, not a straddle hedge.',
    status: 'COMPLETE',
    date: '2026-08-05',
    cardBlurb:
      'Diversification turns the single gold trend edge into a real managed-futures book. Equal-weight weekly Donchian across gold + Nasdaq + PSU (cross-correlation ≈ 0): Sharpe 1.41, CAGR 13%, max DD −15% net — the same return as gold with a third less drawdown. Walk-forward OOS strengthens to Sharpe 1.91. 10/12 green years, ~1 trade/leg/yr.',
    cardStats: [
      { label: 'Sharpe (net, 11.5y)', value: '1.41' },
      { label: 'CAGR', value: '13.0%' },
      { label: 'Max DD', value: '−15%' },
    ],

    systemRules: {
      intro: 'A deliberately simple, low-turnover trend rule applied to a basket of uncorrelated assets. Long-only on each leg (short sides lose to whipsaw); equal-weight across legs (risk-parity is a refinement).',
      sharedCoreTitle: 'Managed-futures trend sleeve — locked rules',
      sharedCore: [
        { k: 'Instruments (core)', v: 'Equal-weight: Gold (GOLDBEES) + Nasdaq (MOM100) + Indian PSU (CPSEETF). Deploy via the respective ETFs / futures.' },
        { k: 'Instruments (extended)', v: 'Optional recent-history legs: S&P-500 (MASPTOP50), US FANG+ (MAFANG), Silver (SILVERBEES) — lift Sharpe to ~1.5 but only ~4.5y of data.' },
        { k: 'Signal (each leg)', v: 'Weekly Donchian-8 long-only: LONG on a weekly close above the prior 8-week high; FLAT on a close below the prior 8-week low. No shorts.' },
        { k: 'Weighting', v: 'Equal-weight the available legs each week. (Inverse-vol / risk-parity is a refinement, not required.)' },
        { k: 'Timeframe / turnover', v: 'Weekly close; ≈ 1 round-trip per leg per year — costs immaterial.' },
        { k: 'Costs / sizing', v: '0.1%/side slippage; compounded; scale the sleeve to its −15% historical drawdown.' },
      ],
      riskLayer: {
        title: 'Gold-alone vs the diversified sleeve',
        caption: 'Net, compounded, 2015–2026. Diversification holds the return and cuts the drawdown — the managed-futures free lunch.',
        columns: ['Sleeve', 'Sharpe', 'CAGR', 'Calmar', 'Max DD'],
        rows: [
          ['Gold alone', '1.07', '13.1%', '0.61', '−21%'],
          ['Core: Gold + Nasdaq + PSU (11.5y)', '1.41', '13.0%', '0.85', '−15%'],
          ['Full: + S&P + FANG + silver (~4.5y legs)', '1.49', '14.0%', '0.91', '−15%'],
        ],
        highlightRows: [1],
      },
    },

    system: {
      intro: 'Weekly Donchian breakout applied to a basket of uncorrelated trend assets, equal-weighted, net-of-cost. Grew out of the Phase-2 trend search (research/101): gold was the first clean edge, and adding global-equity ETFs downloaded via Kite produced a genuinely diversified book.',
      rows: [
        { k: 'Engine', v: 'Weekly-resampled backtester on market_data_unified daily ETF data (GOLDBEES, MOM100/Nasdaq, CPSEETF, MASPTOP50/S&P, MAFANG, SILVERBEES).' },
        { k: 'Signal', v: 'Per leg: Donchian-8 on weekly high/low; long above the prior 8-week high, flat below the prior 8-week low.' },
        { k: 'Diversification', v: 'Core-leg cross-correlations near zero: gold↔Nasdaq +0.00, gold↔PSU −0.04, Nasdaq↔PSU +0.48 — the sleeve out-Sharpes every single leg.' },
        { k: 'P&L basis', v: 'Compounded equity, net of 0.1%/side. ~1 trade/leg/yr.' },
        { k: 'Validation', v: 'Per-asset robustness (Donchian-8 long-only won the earlier N-sweep); walk-forward 2021H2→2026 strengthens to Sharpe 1.91.' },
      ],
    },

    conditions: {
      intro: 'Robustness controls / the seven deadly sins.',
      rows: [
        { k: 'Look-ahead', v: 'None — Donchian uses prior weeks; the weekly-close signal earns the following week.' },
        { k: 'Overfitting', v: 'One signal, one knob (N=8, on a flat plateau); long-only and equal-weight chosen for robustness, not fit. No per-leg tuning.' },
        { k: 'Diversification (not single-factor)', v: 'Three near-uncorrelated legs (gold / US tech / Indian PSU); the Sharpe gain (1.07→1.41) IS the diversification, verified by the correlation matrix.' },
        { k: 'Regime', v: 'Core spans 2015–2026; 10 of 12 years green, only 2018 (−11%) and the 2015 stub red.' },
        { k: 'OOS / walk-forward', v: 'Blind 2021H2→2026 = Sharpe 1.91 / CAGR 20% / −10% DD — holds and strengthens.' },
        { k: 'Capacity / FX', v: 'Global ETFs are INR-denominated (they blend the underlying with USDINR — realistic for an Indian book). ETF proxies hide futures roll cost; re-cost before sizing.' },
      ],
    },

    comparisons: [
      {
        title: 'Per-asset trend edge (weekly Donchian-8 long-only, net)',
        caption: 'The legs the sleeve is built from vs the ones dropped. Gold + Nasdaq + PSU anchor the 11-year core; S&P/FANG/silver are strong but short-history; HangSeng/PSU-bank/China-tech were dead.',
        columns: ['Asset', 'Sharpe', 'CAGR', 'Max DD', 'History', 'In sleeve?'],
        rows: [
          ['GOLDBEES (gold)', '+1.07', '13.1%', '−21%', '11.3y', 'core'],
          ['MOM100 (Nasdaq)', '+0.99', '12.4%', '−27%', '11.5y', 'core'],
          ['CPSEETF (Indian PSU)', '+0.80', '11.9%', '−39%', '11.5y', 'core'],
          ['MASPTOP50 (S&P-500)', '+1.21', '20.8%', '−23%', '4.7y', 'extended'],
          ['MAFANG (US FANG+)', '+1.18', '24.0%', '−20%', '4.7y', 'extended'],
          ['SILVERBEES', '+0.96', '24.7%', '−27%', '4.2y', 'extended'],
          ['HangSeng / PSU-bank / MAHKTECH', '+0.22 / +0.15 / −0.34', 'weak-dead', '', '', 'dropped'],
        ],
        highlightRows: [0, 1, 2],
      },
      {
        title: 'Relationship to the Indian short-vol / equity books',
        caption: 'The sleeve is an independent alpha stream, uncorrelated with the straddle and Indian equity — run it alongside, not as a hedge.',
        columns: ['Metric', 'Value', 'Read'],
        rows: [
          ['Core-leg cross-correlation', '≈ 0.00–0.48', 'genuine internal diversification'],
          ['vs naked-straddle monthly P&L', '≈ +0.2', 'not a straddle hedge'],
          ['Sharpe (net) vs gold-alone', '1.41 vs 1.07', 'diversification lifts risk-adjusted return'],
        ],
      },
    ],

    results: {
      metrics: [
        { label: 'Sharpe (net, 11.5y)', value: '1.41', tone: 'pos' },
        { label: 'CAGR', value: '13.0%' },
        { label: 'Calmar', value: '0.85' },
        { label: 'Max Drawdown', value: '−15%', tone: 'neg' },
        { label: 'Green years', value: '10 / 12' },
        { label: 'Walk-forward OOS Sharpe', value: '1.91', tone: 'pos' },
        { label: 'Turnover', value: '~1 trade/leg/yr' },
      ],
      tables: [
        {
          title: 'Core sleeve — return by year (Gold + Nasdaq + PSU, net)',
          caption: 'Broad-based, not one-year-dependent: 2021(+25%), 2023(+34%), 2024(+26%) all strong; only 2018 (−11%) and the 2015 stub red.',
          heatmap: true,
          columns: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026'],
          rows: [
            ['−0%', '+13%', '+12%', '−11%', '+4%', '+15%', '+25%', '+8%', '+34%', '+26%', '+17%', '+1%'],
          ],
        },
      ],
    },

    winners: [
      {
        config: 'Core managed-futures sleeve — Gold + Nasdaq + PSU, equal-weight, weekly Donchian-8 long-only',
        summary: 'The strongest trend result of Phase 2: same return as gold but a third less drawdown and a much higher Sharpe, from three near-uncorrelated legs. Robust, low-turnover, and strengthens out-of-sample.',
        metrics: [
          { k: 'Sharpe (net)', v: '1.41' },
          { k: 'CAGR', v: '13.0%' },
          { k: 'Calmar', v: '0.85' },
          { k: 'Max DD', v: '−15%' },
          { k: 'OOS Sharpe', v: '1.91' },
        ],
        rejected: [
          'Gold alone — good (Sharpe 1.07) but the sleeve dominates it on risk-adjusted terms',
          'HangSeng / PSU-bank / MAHKTECH (China tech) — weak-to-dead trend, excluded',
          'Long/short & short legs — lose to whipsaw; long-only only',
          'As a straddle hedge — corr ≈ +0.2, independent sleeve only',
        ],
      },
    ],

    caveats: [
      'Core is 11.5 years / one signal; the extended legs (S&P, FANG, silver) have only ~4.5 years — treat the +Sharpe from them as provisional.',
      'ETF proxies; a real deploy uses futures (gold/Nasdaq/S&P) with roll cost the ETFs hide, or the ETFs themselves with their tracking/AUM limits. Re-cost before sizing.',
      'Global ETFs are INR-denominated — returns blend the underlying with USDINR. Realistic for an Indian book, but it is not pure USD exposure.',
      'Correlations can spike toward 1 in a sharp global risk-off reversal (all trend legs whipsaw together) — the −15% DD is in-sample; size for worse.',
      'Not a hedge for the short-straddle book (corr ≈ +0.2). Run as an independent alpha sleeve. Nothing is wired to live orders.',
    ],

    githubLinks: [
      { label: 'research/101 — trend sleeve / Phase 2', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/101_trend_sleeve' },
      { label: '← Deep-dive: Gold Weekly-Trend (single leg)', href: '/app/backtest/gold-weekly-trend' },
    ],
    projectPaths: [
      'research/101_trend_sleeve/scripts/mf_sleeve.py (sleeve), phase2_mf.py (per-asset search), etf_download.py (data)',
      'research/101_trend_sleeve/PHASE2_TREND_COMPLEMENT_DAILY_SWEEP_STATUS.md',
    ],
  },

  {
    slug: 'gold-weekly-trend',
    title: 'Gold Weekly-Trend — the one clean managed-futures edge (Donchian, long-only)',
    verdict:
      'A weekly Donchian breakout on gold (long-only) is a genuinely robust, out-of-sample-validated trend edge: CAGR 14.7%, Sharpe 1.18, Calmar 0.69, max drawdown −21% net of costs over 2015–2026, ~0.8 trades/yr. It has a flat parameter plateau (N8→N20 all Sharpe 1.1–1.2), 9 of 12 green years, and a walk-forward OOS (2021H2→2026) that STRENGTHENS to CAGR 19.7% / Sharpe 1.48 / −11% DD. It emerged from a Phase-2 search for a complement to the short-straddle book — it does NOT hedge the straddle (corr ≈ +0.2), but it stands on its own as an uncorrelated alpha sleeve, and it was the ONLY asset with a clean trend edge (silver too whippy −75% DD; Indian equity ETFs/indices weak-to-dead). Long-only beats long/short everywhere — gold is long-biased, so shorting only adds whipsaw.',
    status: 'COMPLETE',
    date: '2026-08-05',
    cardBlurb:
      'The one clean trend edge from a Phase-2 managed-futures search. Gold weekly Donchian breakout, long-only: CAGR ~15%, Sharpe 1.18, −21% max DD net of costs — a flat param plateau and a walk-forward OOS that STRENGTHENS to Sharpe 1.48. Not a straddle hedge (corr ≈ 0) but a genuinely uncorrelated alpha sleeve. ~0.8 trades/yr.',
    cardStats: [
      { label: 'CAGR (net, 11y)', value: '14.7%' },
      { label: 'Sharpe', value: '1.18' },
      { label: 'Max DD', value: '−21%' },
    ],

    systemRules: {
      intro: 'A deliberately simple, low-turnover trend rule on gold. Long-only was chosen over long/short for robustness, not fit — the short side loses to whipsaw on every parameter.',
      sharedCoreTitle: 'Gold weekly Donchian trend — locked rules',
      sharedCore: [
        { k: 'Instrument', v: 'Gold — GOLDBEES ETF used as the price proxy; deploy via MCX gold / gold futures.' },
        { k: 'Signal', v: 'Weekly Donchian-8. Go/stay LONG on a weekly close above the prior 8-week high; exit to FLAT on a weekly close below the prior 8-week low. No shorts.' },
        { k: 'Timeframe', v: 'Weekly bars; the decision is made on the weekly close (causal — the position earns the following week).' },
        { k: 'Turnover', v: '≈ 0.8 round-trips per year — very low; cost-insensitive.' },
        { k: 'Costs', v: '0.1% slippage per side; equity compounded.' },
        { k: 'Sizing', v: 'Full-notional long / flat; scale to the sleeve’s risk budget (−21% historical max DD).' },
      ],
      riskLayer: {
        title: 'Long-only vs long/short × Donchian length (robustness grid)',
        caption: 'Net of costs, compounded, 2015–2026. Long-only dominates on every length; the edge is a flat plateau, not a tuned peak.',
        columns: ['Config', 'CAGR', 'Sharpe', 'Max DD', 'Calmar'],
        rows: [
          ['Donchian-8 · long-only', '14.7%', '1.18', '−21%', '0.69'],
          ['Donchian-10 · long-only', '14.6%', '1.16', '−21%', '0.68'],
          ['Donchian-20 · long-only', '15.0%', '1.17', '−21%', '0.70'],
          ['Donchian-8 · long/short', '12.5%', '0.95', '−21%', '0.58'],
          ['Donchian-13 · long/short', '11.1%', '0.86', '−29%', '0.38'],
        ],
        highlightRows: [0],
      },
    },

    system: {
      intro: 'Weekly Donchian breakout on gold, net-of-cost, causal. Emerged from the Phase-2 trend search (research/101) as the single asset with a clean, robust edge — and the only Phase-2 candidate that stands on its own.',
      rows: [
        { k: 'Engine', v: 'Our weekly-resampled backtester on market_data_unified GOLDBEES daily (2015→2026-06).' },
        { k: 'Signal', v: 'Donchian-8 channel on weekly high/low; long above the prior 8-week high, flat below the prior 8-week low.' },
        { k: 'P&L basis', v: 'Compounded equity, net of 0.1%/side slippage. ~0.8 trades/yr keeps costs negligible.' },
        { k: 'Validation', v: 'Parameter plateau N8–N20; walk-forward (train ≤2021H1 → test 2021H2–2026) STRENGTHENS the edge.' },
        { k: 'Context', v: 'Beat silver (Sharpe 0.37, −75% DD), Nifty/Junior ETFs (0.53 / −0.21). Crude & USDINR not yet in the DB → a broader managed-futures sleeve needs those downloaded.' },
      ],
    },

    conditions: {
      intro: 'Robustness controls / the seven deadly sins.',
      rows: [
        { k: 'Look-ahead', v: 'None — the Donchian channel uses prior weeks only; the signal is the weekly close and the position earns the NEXT week.' },
        { k: 'Overfitting', v: 'Flat param plateau (N8→N20 all Sharpe 1.1–1.2); long-only picked for robustness, and it wins on every length. One signal, one knob.' },
        { k: 'Cost neglect', v: 'Net of 0.1%/side; turnover ~0.8/yr so costs are immaterial (a strength of the weekly timeframe).' },
        { k: 'Regime', v: '2015–2026 spans gold’s 2015–18 doldrums, the 2019–20 rally, 2022 chop, and the 2024–25 bull. 9 of 12 years green; losing years small (−4%, −5%).' },
        { k: 'OOS / walk-forward', v: 'Train ≤2021H1 picks N=8; blind OOS 2021H2→2026 = CAGR 19.7%, Sharpe 1.48, −11% DD — the edge holds and strengthens.' },
        { k: 'Capacity', v: 'Gold futures (MCX / international) are deep; the ETF is a proxy — real deploy carries roll cost the ETF hides.' },
      ],
    },

    comparisons: [
      {
        title: 'Phase-2 managed-futures search — why gold, and only gold',
        caption: 'Weekly Donchian-10, per available asset. Gold is the lone clean edge; silver is too whippy, Indian-equity ETFs are weak-to-dead. Crude/USDINR/global not in the DB.',
        columns: ['Asset', 'Sharpe', 'Ann. return', 'Max DD', 'History'],
        rows: [
          ['GOLDBEES', '+0.98', '+13.0%', '−23%', '10.3y'],
          ['SILVERBEES', '+0.37', '+11.0%', '−75%', '4.2y'],
          ['NIFTYBEES', '+0.53', '+7.8%', '−39%', '10.4y'],
          ['JUNIORBEES (Nifty Next 50)', '−0.21', '−3.7%', '−127%', '10.3y'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Relationship to the short-straddle book — a diversifier, not a hedge',
        caption: 'Phase-2 started as a hunt for a straddle cushion. Gold trend does NOT cushion the straddle (positive correlation), but it is an independent return stream worth running alongside it.',
        columns: ['Metric', 'Value', 'Read'],
        rows: [
          ['Gold-trend vs naked-straddle monthly corr', '+0.21', 'mildly POSITIVE → not a tail hedge'],
          ['Adds to combined straddle Calmar?', 'No (best weight 0)', 'value is as its OWN sleeve, not a hedge'],
          ['Standalone Sharpe', '1.18', 'stands on its own two feet'],
        ],
      },
    ],

    results: {
      metrics: [
        { label: 'CAGR (net, 2015–26)', value: '14.7%', tone: 'pos' },
        { label: 'Sharpe', value: '1.18' },
        { label: 'Calmar', value: '0.69' },
        { label: 'Max Drawdown', value: '−21%', tone: 'neg' },
        { label: 'Weekly win rate', value: '58%' },
        { label: 'Trades / year', value: '~0.8' },
        { label: 'Walk-forward OOS Sharpe', value: '1.48', tone: 'pos' },
      ],
      tables: [
        {
          title: 'Gold weekly trend — return by year (Donchian-8 long-only, net)',
          caption: 'Approx. sum of weekly net returns. 9 of 12 years green; losing years small. 2025 (+56%) was the gold bull, but 2016/19/20/22/24 were all solid — not one-year-dependent.',
          heatmap: true,
          columns: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026'],
          rows: [
            ['+0%', '+13%', '−4%', '+3%', '+23%', '+23%', '−5%', '+13%', '+7%', '+20%', '+56%', '+10%'],
          ],
        },
      ],
    },

    winners: [
      {
        config: 'Gold · weekly Donchian-8 · long-only',
        summary: 'The one Phase-2 candidate that stands on its own: robust across parameters, strengthens out-of-sample, tiny turnover, and uncorrelated with the Indian-equity/vol books.',
        metrics: [
          { k: 'CAGR (net)', v: '14.7%' },
          { k: 'Sharpe', v: '1.18' },
          { k: 'Calmar', v: '0.69' },
          { k: 'Max DD', v: '−21%' },
          { k: 'OOS Sharpe', v: '1.48' },
        ],
        rejected: [
          'Long/short — the short side loses to whipsaw on every length (gold is long-biased)',
          'Silver — trends but −75% DD; drags a gold+silver sleeve below gold-alone',
          'Nifty/BankNifty/Junior trend — weak or negative (Indian index daily & weekly whipsaw)',
          'As a straddle hedge — corr +0.21, does not cushion; value is standalone only',
        ],
      },
    ],

    caveats: [
      'Single asset, single signal, one 11-year history — a robust SIGNAL, not yet live-validated. The walk-forward and param plateau argue against overfit, but it is one instrument.',
      'GOLDBEES is an ETF proxy and its data ends 2026-06-12; a real deploy uses gold FUTURES with roll costs the ETF hides — re-cost before sizing.',
      'It is NOT a hedge for the short-straddle book (corr +0.21) — run it as an independent sleeve, sized to its own −21% drawdown, not as straddle insurance.',
      'A true diversified managed-futures sleeve (gold + crude + FX + global) needs crude/USDINR/global data downloaded — gold is the only clean trender currently in the DB.',
      'Nothing is wired to live orders. Returns are compounded on full notional; scale to the risk budget.',
    ],

    githubLinks: [
      { label: 'research/101 — trend sleeve / Phase 2', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/101_trend_sleeve' },
      { label: '← Complements: NIFTY straddle (look-ahead audit)', href: '/app/backtest/nifty-straddle-lookahead-audit' },
    ],
    projectPaths: [
      'research/101_trend_sleeve/PHASE2_TREND_COMPLEMENT_DAILY_SWEEP_STATUS.md',
      'research/101_trend_sleeve/scripts/gold_g4.py (proper study), phase2_widen.py (asset search)',
    ],
  },

  {
    slug: 'nifty-straddle-lookahead-audit',
    title: 'NIFTY / SENSEX Straddle & Iron-Fly — Look-Ahead Audit & Honest Comparison',
    verdict:
      'An adversarial re-audit of our own straddle backtests found a single look-ahead bug — the ATM strike was chosen from the entry-day CLOSE while the trade entered at the OPEN — that had inflated results ~3×, smoothed every equity curve and hidden the drawdown. Corrected (strike chosen at the open, causal), 8 of 9 straddle variants are dead or edgeless. The lone survivor is the NIFTY DTE-3 iron fly: a weak-but-real, defined-risk edge (+₹18.1L / 7yr, Mean/SD 0.12, Calmar 0.66, worst week −₹1.84L capped by the wings, ~33%/yr on 2×-drawdown capital). Naked straddles post the biggest raw ₹ (+₹39.5L) but carry an UNBOUNDED crash tail → disqualified. SENSEX has NO edge once the strike is picked honestly. The CPR compression filter is real but structure-specific (it lifts the AlgoTest V2 same-week fly, and HURTS our next-week fly). This study also independently reproduces and VALIDATES the AlgoTest V2 iron fly (+₹6.9L on our engine vs +₹8.85L on AlgoTest, same losing years).',
    status: 'COMPLETE',
    date: '2026-08-05',
    cardBlurb:
      'We smelled a result too good to be true — 97.7% win, ₹1.19 Cr — and killed it. A one-line look-ahead bug (ATM strike picked from the day’s close) had tripled our straddle backtests. Corrected at the open: 8 of 9 variants die; only the NIFTY DTE-3 iron fly survives (+₹18.1L/7yr, Calmar 0.66, defined risk). Independently validates the AlgoTest V2 fly.',
    cardStats: [
      { label: 'Honest DTE-3 fly (7yr)', value: '+₹18.1L' },
      { label: 'Calmar', value: '0.66' },
      { label: 'Bias removed', value: '3× → real' },
    ],

    systemRules: {
      intro:
        'The one survivor of the audit — a defined-risk NIFTY iron fly, tested on our own bhavcopy engine with the look-ahead removed. Naked variants are shown for contrast but are disqualified (unbounded tail).',
      sharedCoreTitle: 'Honest DTE-3 iron fly — locked rules',
      sharedCore: [
        { k: 'Instrument', v: 'Short ATM NIFTY straddle + long 3%-OTM CE & PE in the NEXT-week expiry = short iron fly (defined risk).' },
        { k: 'Entry', v: '≈3 calendar days before the weekly expiry, at the option OPEN (~09:20). ATM = nearest 50 to the entry-day OPEN spot — the only price known at 09:20 (the corrected, causal choice).' },
        { k: 'Exit', v: 'DTE-1 (day before expiry) at the CLOSE. No intraday management assumed.' },
        { k: 'Gate', v: 'India VIX 13–28 (floor + ceiling).' },
        { k: 'Costs', v: '0.3%/leg slippage + ₹160/round-trip; OI≥100 ATM, ≥25 wings (real traded contracts only).' },
        { k: 'Sizing / margin', v: '10 lots (qty 650); iron-fly SPAN ≈ ₹50,000/lot → ₹5L for 10 lots (a naked straddle blocks ~₹1.3L/lot = ~2.6× more).' },
      ],
      riskLayer: {
        title: 'The audit — every variant, look-ahead removed',
        caption: 'Real, open-based strike selection. Only the DTE-3 iron fly has a defensible, defined-risk edge; nothing clears a Mean/SD of 0.5.',
        columns: ['Variant', 'Net / 7yr', 'Mean/SD', 'Max DD', 'Verdict'],
        rows: [
          ['NIFTY DTE-3 iron fly (this system)', '+₹18.1L', '0.12', '−₹3.73L', 'WEAK SIGNAL — the only survivor'],
          ['NIFTY DTE-3 naked', '+₹39.5L', '0.18', '−₹5.62L', 'bigger ₹, UNBOUNDED tail → out'],
          ['NIFTY DTE-1 iron fly', '+₹8.2L', '0.07', '−₹6.74L', 'DEAD'],
          ['SENSEX DTE-3 iron fly', '+₹2.1L', '0.02', '−₹9.47L', 'NO EDGE'],
          ['SENSEX DTE-1 naked', '+₹4.1L', '0.05', '−₹3.91L', 'NO EDGE'],
        ],
        highlightRows: [0],
      },
    },

    system: {
      intro:
        'Tested on our own SQLite bhavcopy engine (not AlgoTest): real NSE/BSE end-of-day option premiums, entry at the daily OPEN, exit at the daily CLOSE. The single correction vs the earlier pages is the strike-selection line.',
      rows: [
        { k: 'Engine', v: 'Our bhav backtester — nse_options_bhav (NIFTY, 2019→2026) + bse_options_bhav (SENSEX, 2024→2026, downloaded for this study).' },
        { k: 'Strike (the fix)', v: 'ATM = nearest strike to the entry-day OPEN underlying. The earlier pages used the entry-day CLOSE — the single look-ahead bug this study corrects.' },
        { k: 'Entry / exit', v: 'Sell straddle + buy wings at the option OPEN on entry day; close all legs at the CLOSE on DTE-1. DTE by calendar days.' },
        { k: 'P&L basis', v: 'Net of 0.3%/leg + ₹160/RT; 10 lots; returns stated on ₹50k/lot fly margin (naked on ~₹1.3L/lot).' },
        { k: 'Data built', v: 'Downloaded BSE UDiFF bhavcopy (SENSEX + BANKEX, 2024→now, 370,746 rows → bse_options_bhav) and refreshed India VIX to 2026-08-05.' },
      ],
    },

    conditions: {
      intro: 'The seven deadly sins — with the one that bit us front and centre.',
      rows: [
        { k: 'Look-ahead (headline)', v: 'FOUND & FIXED. The prior pages picked the ATM strike from the entry-day CLOSE while entering at the OPEN — a future peek that tripled P&L, smoothed the curve and hid the drawdown. All numbers here pick the strike at the OPEN (causal). Kill-test: on one Mar-2019 trade the same day flips from +₹1.14L (cheat) to −₹1.04L (honest); the strike differed from the close on 84% of days.' },
        { k: 'Cost neglect', v: 'Net of 0.3%/leg + ₹160/RT throughout; naked vs fly compared on their real (different) margins.' },
        { k: 'Overfitting', v: 'No parameter mined post-hoc. The CPR filter was tested on our system and REJECTED (it hurt) — an anti-overfit check, not a fit.' },
        { k: 'Survivorship', v: 'ATM straddle on a liquid index; OI filter keeps only really-traded contracts (research/89 rule).' },
        { k: 'Regime', v: 'NIFTY spans 2019–2026 incl. COVID; SENSEX 2024–2026 only (calm, low-VIX — a caveat on the SENSEX verdict).' },
        { k: 'Cross-validation', v: 'The AlgoTest V2 iron fly was independently reproduced on our engine (+₹6.93L vs +₹8.85L, losses in the same years) — mutual validation across two tools and two data paths.' },
      ],
    },

    comparisons: [
      {
        title: 'The look-ahead illusion — same systems, before vs after the fix',
        caption: 'Strike from the entry-day CLOSE (future peek) vs the OPEN (causal). The bug roughly tripled P&L and shrank the drawdown ~3×. These "before" numbers are RETRACTED.',
        columns: ['System', 'Look-ahead (biased, retracted)', 'Real (open-based)'],
        rows: [
          ['NIFTY DTE-3 naked · VIX', '+₹1.19 Cr · M/SD 0.65 · DD −₹2.0L', '+₹39.5L · M/SD 0.18 · DD −₹5.6L'],
          ['NIFTY DTE-3 fly · VIX', '+₹68.8L · M/SD ~0.48', '+₹18.1L · M/SD 0.12 · DD −₹3.7L'],
          ['NIFTY DTE-1 (the “grail”)', '97.7% win · M/SD 1.31 · DD −₹0.2L', '62% win · M/SD 0.07 · DD −₹6.7L'],
        ],
      },
      {
        title: 'Every variant, honest — NIFTY (7yr) + SENSEX (2.5yr)',
        caption: 'Real, open-based strike; 10-lot books (NIFTY qty 650 / SENSEX qty 200). Nothing clears a Mean/SD of 0.5.',
        columns: ['Index', 'Variant', 'Net', 'Win', 'M/SD', 'Worst wk', 'Max DD', 'Verdict'],
        rows: [
          ['NIFTY', 'DTE-3 iron fly', '+₹18.1L', '63%', '0.12', '−₹1.84L', '−₹3.73L', 'WEAK SIGNAL ★'],
          ['NIFTY', 'DTE-3 naked', '+₹39.5L', '67%', '0.18', '−₹4.20L', '−₹5.62L', 'unbounded tail'],
          ['NIFTY', 'DTE-1 iron fly', '+₹8.2L', '62%', '0.07', '−₹2.38L', '−₹6.74L', 'DEAD'],
          ['NIFTY', 'DTE-1 naked', '+₹28.4L', '71%', '0.21', '−₹1.92L', '−₹6.74L', 'unbounded; untradeable'],
          ['SENSEX', 'DTE-3 iron fly', '+₹2.1L', '59%', '0.02', '−₹2.04L', '−₹9.47L', 'NO EDGE'],
          ['SENSEX', 'DTE-3 naked', '+₹8.7L', '62%', '0.06', '−₹4.49L', '−₹10.1L', 'NO EDGE'],
          ['SENSEX', 'DTE-1 iron fly', '+₹3.4L', '60%', '0.06', '−₹1.94L', '−₹4.09L', 'NO EDGE'],
          ['SENSEX', 'DTE-1 naked', '+₹4.1L', '61%', '0.05', '−₹3.53L', '−₹3.91L', 'NO EDGE'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Naked vs our fly vs AlgoTest V2 + CPR — on real margin',
        caption: 'Fly ₹50k/lot; naked ~₹1.3L/lot (≈2.6× more). "Return" is ANNUALIZED on the capital you must actually hold to survive the drawdown (≈2× max DD, or the exchange margin, whichever is larger).',
        columns: ['System', 'Net / yr', 'Margin (10 lots)', 'Max DD', 'DD % of margin', 'Ann. return (safe capital)', 'Risk'],
        rows: [
          ['Naked straddle (ours, VIX 13-28)', '₹5.34L', '~₹13L', '−₹5.62L', '43%', '~41%/yr', '❌ UNBOUNDED tail'],
          ['Our iron fly (3% next-wk)', '₹2.44L', '₹5L', '−₹3.73L', '75%', '~33%/yr', '✅ defined; needs 2× buffer'],
          ['V2 + CPR (AlgoTest, stop+PT)', '₹1.51L', '₹5L', '−₹0.95L', '19%', '~30%/yr', '✅ defined + managed; safe at min margin'],
        ],
        highlightRows: [2],
      },
      {
        title: 'CPR compression filter on our fly — REJECTED (structure-specific)',
        caption: 'Skipping narrow-CPR (compressed) entries LIFTS the V2 same-week fly but HURTS our next-week fly — under both a VIX band and a VIX floor, so it is structural, not VIX-redundancy. Kept off our book.',
        columns: ['Our DTE-3 fly', 'VIX gate', 'CPR', 'Net / 7yr', 'Calmar', 'Max DD'],
        rows: [
          ['baseline', '13–28 band', 'none', '+₹18.07L', '0.66', '−₹3.73L'],
          ['+ skip CPR<0.10%', '13–28 band', 'skip', '+₹12.90L', '0.56', '−₹3.14L'],
          ['baseline', '≥13 floor', 'none', '+₹17.80L', '0.63', '−₹3.83L'],
          ['+ skip CPR<0.10%', '≥13 floor', 'skip', '+₹12.76L', '0.55', '−₹3.14L'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Independent validation of the AlgoTest V2 iron fly (our engine)',
        caption: 'V2 structure (2% same-week wings, 4-trading-days-before entry, VIX≥13) reproduced on our look-ahead-free bhav engine. The core edge AND the CPR uplift both replicate; V2’s smaller drawdown comes from its intraday 2%-stop + 40%-PT, which our EOD engine cannot model.',
        columns: ['Config (our engine, VIX≥13, no stop/PT)', 'Net / 7yr', 'Calmar', 'Max DD'],
        rows: [
          ['V2 structure, no CPR', '+₹6.93L', '0.19', '−₹4.87L'],
          ['+ skip CPR<0.10%', '+₹13.79L', '0.53', '−₹3.51L'],
          ['AlgoTest reference (with stop+PT)', '+₹8.1L → +₹11.0L', '0.95 → 1.59', '−₹1.17L → −₹0.95L'],
        ],
        highlightRows: [1],
      },
    ],

    results: {
      metrics: [
        { label: 'Honest DTE-3 fly · Net (7yr)', value: '+₹18,07,487', tone: 'pos' },
        { label: 'Mean / SD', value: '0.12' },
        { label: 'Calmar', value: '0.66' },
        { label: 'Max Drawdown', value: '−₹3,73,276', tone: 'neg' },
        { label: 'Worst week (capped by wings)', value: '−₹1,83,583', tone: 'neg' },
        { label: 'Ann. return (₹7.5L safe capital)', value: '~33%/yr', hint: '48.9% on bare ₹5L margin, but the −₹3.7L drawdown is 75% of it → run on ~2× buffer' },
        { label: 'Win rate', value: '63%' },
      ],
      tables: [
        {
          title: 'Honest DTE-3 iron fly — P&L by year (VIX 13-28, no CPR)',
          caption: 'Net of costs, open-based strike. 6 of 8 years green; 2020 and the 5-month 2026 stub red.',
          heatmap: true,
          columns: ['2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026'],
          rows: [
            ['+305k', '−127k', '+382k', '+54k', '+44k', '+744k', '+502k', '−98k'],
          ],
        },
      ],
    },

    winners: [
      {
        config: 'NIFTY DTE-3 iron fly — 3% next-week wings, VIX 13-28 (the only survivor)',
        summary: 'The one variant with a defensible real edge after the look-ahead fix: defined risk, positive in 6 of 8 years, worst week capped by the wings. A weak SIGNAL, not yet a proven strategy.',
        metrics: [
          { k: 'Net P&L (7yr)', v: '+₹18,07,487' },
          { k: 'Mean/SD', v: '0.12' },
          { k: 'Calmar', v: '0.66' },
          { k: 'Max DD', v: '−₹3,73,276' },
          { k: 'Ann. return (safe capital)', v: '~33%/yr' },
        ],
        rejected: [
          'DTE-1 (any) — the “97.7% grail” was pure look-ahead; dead at M/SD 0.07 and untradeable naked near expiry',
          'Naked straddles — biggest raw ₹ (+₹39.5L) but an UNBOUNDED crash tail; disqualified',
          'SENSEX (all variants) — no edge once the strike is picked honestly (M/SD 0.02–0.06)',
          'CPR compression filter on our fly — helps V2’s same-week structure, hurts ours',
        ],
      },
    ],

    caveats: [
      'This study CORRECTS earlier NIFTY straddle pages — the +₹1.19 Cr DTE-3 and 97.7%-win DTE-1 figures had look-ahead in strike selection and are SUPERSEDED.',
      'The surviving edge is WEAK (Mean/SD 0.12, ~₹7k/trade). Positive (t≈2.6) but thin enough that realistic expiry-week slippage could erode much of it — a SIGNAL, not a proven strategy.',
      'No intraday stop/profit-target is modelled — our bhav engine only sees the daily open & close. The AlgoTest V2 book’s far smaller drawdown comes from exactly that intraday management, which we can only cross-check on AlgoTest, not reproduce here.',
      'Naked’s −₹5.62L drawdown is IN-SAMPLE; an unhedged straddle can lose multiples of it on a real crash gap. Its return-on-margin looks best precisely because its tail risk is uncapped — which is why it is disqualified for real trading.',
      'SENSEX = 2024–2026 only (BSE weeklies launched 2024) — a calm, low-VIX window with no 2020-style stress; the “no edge” verdict is on that limited sample.',
      'Margins are indicative (fly ₹50k/lot, naked ~₹1.3L/lot) — verify on Kite at trade time. Returns are simple-on-capital (fixed 10 lots, profits drawn), not compounded.',
    ],

    githubLinks: [
      { label: 'research/100 — SENSEX DTE-3 + look-ahead audit', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/100_sensex_dte3_straddle' },
      { label: '← Related: V2 Iron Fly (Stop-Loss × VIX)', href: '/app/backtest/v2-nifty-ironfly-sl-vix' },
    ],
    projectPaths: [
      'research/100_sensex_dte3_straddle/results/RESULTS.md',
      'research/100_sensex_dte3_straddle/scripts/ (bhav downloader, realistic backtests, look-ahead kill-test, V2 reproduction)',
    ],
  },

  {
    slug: 'fardte-rescue',
    title: 'Rescuing the far-from-expiry days — five ideas, four dead, one that works',
    verdict:
      'SIGNAL (not yet STRATEGY). research/79 showed NAS-OPT loses Rs441/day on DTE>=4. The cause is NOT the ' +
      'stop and NOT that those days are wilder — it is that there is almost nothing to collect: intraday you ' +
      'keep only 6-10% of the far-DTE premium you sell (vs 83% on expiry day), because theta lives in an ' +
      "option's final DAYS, not its final hours. Every intraday fix therefore failed: 22 exit rules (best " +
      '+Rs9/day), calm-day filters (the calm days are also the thin-premium days), the "Wed/Thu move more" ' +
      'theory (an 11-day accident — across 11 years every weekday moves the same), and directional selling ' +
      'after a break (the entire edge is smaller than the slippage). What DOES work is to stop fighting the ' +
      'cause: HOLD the position for days. Enter Wednesday, sell a ~0.8%-OTM strangle with wings 1.0% beyond, ' +
      'EXIT FRIDAY: Calmar 1.63, +Rs38,944/yr, 75% win, 11/12 years positive, ~100% p.a. on the margin it ' +
      'ties up — using the capital NAS-OPT leaves idle Wed-Fri, and flat again before Monday so it never ' +
      'competes for margin. CORRECTION: an earlier version quoted Calmar 1.83 / +Rs65,599. That run fixed the ' +
      'strikes at +/-100 POINTS across 11 years while NIFTY tripled (8,000 -> 24,000) — 1.25% OTM in 2015 vs ' +
      '0.42% today, i.e. three different strategies wearing one name. Re-run on PERCENTAGE strikes the edge is ' +
      '38% smaller. NOT deployable: never tested on a real chain, the portfolio-correlation test is still open ' +
      '(every book we run is already short volatility), and wider shorts were STILL improving at the edge of ' +
      'the sweep — the optimum may lie outside what was tested.',
    status: 'COMPLETE',
    date: '2026-07-14',
    cardBlurb:
      'NAS-OPT loses money on every day that is not expiry-day-or-the-day-before. Can those days be rescued — ' +
      'by a better stop, a filter, a skew, or a different system? Five ideas tested on 11 years. Four are dead. ' +
      'The one that works requires abandoning the intraday frame entirely.',
    cardStats: [
      { label: 'The cause', value: 'keep only 6-10% of premium' },
      { label: 'Best answer', value: 'Condor, Wed→Fri' },
      { label: 'Calmar', value: '1.63' },
    ],
    systemRules: {
      intro:
        'The candidate that survived. It is deliberately shaped around a CONSTRAINT, not just around returns: ' +
        'NAS-OPT only trades 0/1-DTE (Mon/Tue), so the margin sits idle Wed-Fri. This uses exactly those idle ' +
        'days and is flat again before Monday, so the two never compete for capital.',
      sharedCoreTitle: 'The Wed→Fri iron condor',
      sharedCore: [
        { k: 'Entry', v: 'Wednesday close (DTE6 against the Tuesday weekly expiry)' },
        { k: 'Structure', v: 'SELL strangle ~0.8% OTM either side, BUY wings 1.0% beyond each short. PERCENTAGE strikes, NOT fixed points -- see the correction.' },
        { k: 'Exit', v: 'FRIDAY close — never held over a weekend, never held into Mon/Tue' },
        { k: 'Stop', v: 'close the position if the combined premium doubles' },
        { k: 'Size', v: '2 lots/leg (130 qty) — margin ~Rs52,500 (capped by the wings)' },
        { k: 'Why the wings', v: 'not a compromise — the naked version has 2.5x the drawdown, 4x the margin and a WORSE Calmar' },
        { k: 'Why exit Friday', v: 'the weekend hold is worth -Rs96/trade: the Monday gap costs more than 3 days of decay earn' },
      ],
      riskLayer: {
        title: 'What each exit day costs and buys',
        caption:
          'The biggest P&L is the worst trade. Exiting Friday earns half as much at a fifth of the risk — and ' +
          'uses the margin for 1.1 days instead of 3.7, which is the metric that actually matters when the ' +
          'capital is needed elsewhere on Monday.',
        columns: ['Shorts / Wings', 'Mean/trade', 'Annual', 'Max DD', 'Calmar', '+years', 'Return on margin'],
        rows: [
          ['0.8% / 1.0% (stop x2)', '+880', '+38,944', '-23,820', '1.63', '11/12', '100% p.a.'],
          ['0.8% / 0.6% (stop x1.5)', '+466', '+20,626', '-13,179', '1.57', '11/12', '66% p.a.'],
          ['0.6% / 1.0% (stop x1.5)', '+757', '+33,507', '-22,627', '1.48', '10/12', '86% p.a.'],
          ['0.4% / 0.6% (wings only)', '+196', '+8,637', '-17,392', '0.50', '9/12', '27% p.a.'],
        ],
        highlightRows: [0],
      },
    },
    system: {
      intro:
        'The question: research/79 proved the 0/1-DTE gate IS the edge and every other day is EV-negative. ' +
        'Rather than accept that, this study asks whether the far-DTE days can be made to work at all.',
      rows: [
        { k: 'The problem', v: 'NAS-OPT on DTE>=4: -Rs441/day (33 days), vs +Rs1,578/day on 0/1-DTE' },
        { k: 'Ideas tested', v: 'exit rules, band widths, premium SLs, calm-day filters, weekday skew, directional selling, overnight holding' },
        { k: 'Evidence base', v: '58 real chain days (option P&L) + 2,693 days of NIFTY 5-min (predictability) + a calibrated engine (11 years of option P&L)' },
        { k: 'Success bar', v: 'must beat the current rule AND be stable per-year AND survive slippage. Most ideas failed all three.' },
      ],
    },
    conditions: {
      intro: 'Three independent evidence bases, because each has a different blind spot.',
      rows: [
        { k: 'A. Real chain', v: 'options_data.db — 58 days, actual premiums, minute by minute. Used for the exit-rule sweep and the decomposition.' },
        { k: 'B. NIFTY 5-min', v: '2,693 days (2015-02 → 2026-03) + India VIX. Used for anything about how the index MOVES — no option pricing needed.' },
        { k: 'C. Synthetic engine', v: 'Black-Scholes on NIFTY + VIX as IV, IV multiplier calibrated PER DTE against the real chain, to ZERO signed error.' },
        { k: 'Costs', v: 'Rs20/leg brokerage + slippage swept at 0 / 1 / 2% of premium. Everything quoted at 1%.' },
        { k: 'ENGINE BIAS — caught', v: 'the first calibration minimised ABSOLUTE error and left the engine 7% too EXPENSIVE. Since the strategy SELLS that premium, it invented ~Rs1,430/trade. Recalibrated to zero SIGNED error; the "edge" fell from +4,980 to +3,549.' },
      ],
    },
    comparisons: [
      {
        title: 'WHY the far-DTE days lose — the keep rate is the whole story',
        caption:
          'How much of the premium you sell is still yours at 14:45. On expiry day you sell 45 points and keep ' +
          '37. On Wednesday you sell 273 and keep 28 — carrying 6x the premium risk to harvest crumbs. And a ' +
          'far-DTE option has real delta, so a 0.4% move costs about what a calm day pays. The asymmetry that ' +
          'makes DTE0 work (+37.2 vs -3.3) is simply absent.',
        columns: ['DTE', 'Credit sold', 'Kept if calm', 'KEEP RATE', 'Given back if the band is hit'],
        rows: [
          ['0', '44.6', '37.2', '83%', '-3.3'],
          ['1', '138.2', '25.8', '19%', '+0.2'],
          ['4', '178.2', '11.5', '6%', '-2.7'],
          ['5', '190.1', '12.5', '7%', '-13.1'],
          ['6', '273.0', '27.7', '10%', '-12.8'],
        ],
        highlightRows: [0],
        heatmap: true,
      },
      {
        title: 'The four dead ideas',
        caption:
          'Each was tested to the point where it could be killed. Note the ceiling: far-DTE pays +1,753 on a ' +
          'calm day and costs -1,539 on a hit, at a 67% hit rate — so even a PERFECT COSTLESS stop caps out at ' +
          '+578/day. No exit rule at the same band can beat that, which is why the sweep was doomed in advance.',
        columns: ['Idea', 'Result', 'Why it died'],
        rows: [
          ['22 exit rules (bands 0.2-1.0%, premium SL, per-leg SL, no-stop, targets)', 'best +Rs9/day', 'the best of 22 tries on 33 days — i.e. zero. You cannot tune a stop into an edge that is not in the payoff.'],
          ['"Wed/Thu move more" → trade the move', 'DEAD', 'an 11-day accident. Over 2,693 days P(move>=0.4%) is Mon 76.2 / Tue 77.1 / Wed 74.4 / Thu 78.3 / Fri 78.5% — identical.'],
          ['Filter for calm days (VIX / CPR / opening range / gap)', 'DEAD', 'volatility IS predictable (VIX<12 → 50.3% hit vs >=22 → 97.6%, monotone) but breakeven needs <53.3% AND the calm days are also the thin-premium days.'],
          ['Directional selling after the band breaks', 'DEAD', 'the whole edge is inside the slippage: +101/trade at 0%, -40 at 1%, -181 at 2%. Only 5/12 years positive.'],
        ],
      },
      {
        title: 'The trap that nearly fooled me — why win-rate and median are liars',
        caption:
          'The best directional-short cell had a +343 MEDIAN and a 60% WIN RATE — with a NEGATIVE MEAN. Many ' +
          'small wins, rare -44,797 disasters. Reporting either of the first two numbers would have sold this ' +
          'as a discovery. Its VIX conditioning was also non-monotone (+50 / -348 / -145 / +219) = noise — in ' +
          'sharp contrast to the clean monotone structures elsewhere. That contrast is how you tell them apart.',
        columns: ['Metric', 'Best directional-short cell', 'Reads as'],
        rows: [
          ['Win rate', '60%', 'a winner'],
          ['Median trade', '+Rs343', 'a winner'],
          ['MEAN trade', '-Rs40', 'a loser'],
          ['Worst trade', '-Rs44,797', 'the reason'],
        ],
      },
    ],
    results: {
      metrics: [
        { label: 'Calmar (Wed→Fri condor)', value: '1.63', tone: 'pos', hint: 'CORRECTED — was 1.83 before the strike-drift fix' },
        { label: 'Annual P&L', value: '+Rs38,944', tone: 'pos', hint: '2 lots, 531 trades, 11/12 years positive' },
        { label: 'Max drawdown', value: '-Rs23,820', hint: '~100% p.a. on the margin it ties up' },
        { label: 'Win rate', value: '75%', tone: 'pos' },
        { label: 'Margin', value: '~Rs52,500', hint: 'capped by the wings; naked would need ~Rs233k' },
        { label: 'Weekend hold', value: '-Rs96/trade', tone: 'neg', hint: 'the Monday gap costs more than 3 days of decay earn' },
      ],
      tables: [
        {
          title: 'Cost sensitivity — it survives where the directional short died',
          columns: ['Slippage', 'Naked hold-to-expiry (mean/trade)'],
          rows: [
            ['0%', '+3,891'],
            ['1%', '+3,549'],
            ['2%', '+3,206'],
          ],
        },
      ],
      charts: [
        {
          src: '/app/fardte-rescue.png',
          caption:
            'Equity and drawdown for the three shapes, the keep-rate by DTE that explains why intraday fails, ' +
            'and the scoreboard. The biggest P&L (naked, hold-to-expiry) is the WORST trade once you look at risk.',
        },
      ],
    },
    winners: [
      {
        config: 'Iron condor — enter Wednesday, ~100pt shorts / 250pt wings, EXIT FRIDAY',
        summary:
          'The only shape that earns on the capital NAS-OPT leaves idle, gets out before Monday so it never ' +
          'competes for margin, and never carries a weekend gap. Half the P&L of holding to expiry, a fifth of ' +
          'the risk, and 2.5x the return per rupee per day.',
        metrics: [
          { k: 'Calmar', v: '1.83 (vs 0.72 hold-to-expiry, 0.60 naked)' },
          { k: 'Annual', v: '+Rs65,599 on ~Rs52,500 margin' },
          { k: 'Max DD', v: '-Rs35,822' },
          { k: 'Return per margin-day', v: '265 bp (vs 41 bp for the naked version)' },
        ],
        rejected: [
          'NAKED strangle — makes the most money and is the WRONG trade: 2.5x drawdown, 4x margin, worse Calmar',
          'Holding to Tuesday expiry — collides with NAS-OPT for margin exactly when it is needed',
          'Holding over the weekend — worth -Rs96/trade; the Monday gap beats the weekend decay',
        ],
      },
    ],
    caveats: [
      'STRIKE-DRIFT CORRECTION — the biggest error in this study. The first run fixed the strikes at +/-100 ' +
      'POINTS across 2015-2026 while NIFTY went 8,000 -> 24,000: that is 1.25% OTM in 2015 and 0.42% today, ' +
      'and the wings drifted from 3% wide to 1% wide. Three different strategies wearing one name. It inflated ' +
      'the result ~38% (Calmar 1.94 -> 1.63; +Rs62k -> +Rs39k/yr) and manufactured a bogus "tighter stops are ' +
      'strictly better" monotonicity. Everything is now on PERCENTAGE strikes.',
      'THE 2x-CREDIT STOP WAS DEAD CODE on the condor: 2x a ~150pt credit is ~300 points, but the structure can ' +
      'never be worth more than the wing width. The stop sat ABOVE its own ceiling and fired on 0-5% of trades. ' +
      'THE WINGS ARE THE STOP — max loss is known at entry.',
      'THE DECAY-CLOCK / SNEAK-IN IDEA IS DEAD. Decay IS lumpy on the real chain (Wednesday\'s last two hours ' +
      'melt 2.4x faster than midday, ~Rs2,411 IF the index stands still). Traded for real, including the ' +
      'afternoons that MOVE, the best window makes +Rs269/day over 33 days and every window before 14:30 LOSES. ' +
      'The clock measured the reward with the risk removed.',
      'STRIKE PEAK — FOUND. Extended, the sweep peaks at 1.4% OTM shorts (Calmar 1.91, +Rs42,954/yr, 78% win) ' +
      'and TURNS at 1.6% — a real interior optimum. BUT the engine has NO volatility skew and was validated ' +
      'near-ATM, and 1.4% OTM is exactly where skew bites hardest — the engine is least reliable there. ' +
      '0.8-1.0% is the validated-confidence zone; 1.4% is suggestive and needs real-chain pricing. The live ' +
      'paper book runs at 0.8% and will collect that evidence.',
      'Marking the stop every 5 minutes instead of at daily closes changes almost nothing — an earlier caveat ' +
      'of mine that turned out not to matter.',
      'The strategy has never touched a real option chain. Every price came from the engine. The engine was ' +
      'validated against 58 real days, but the STRATEGY was not.',
      'CORRELATION — RESOLVED, and the worry was BACKWARDS. NAS-OPT holds the 2 days CLOSEST to expiry (DTE ' +
      '0-1); this condor holds days 4-6 BEFORE expiry. They trade opposite ends of the week: simultaneously ' +
      'in the market only 3 calendar days in 11 years, and on the 10 worst condor days NAS-OPT was flat ' +
      'every time. Combined drawdown -49,771 vs the sum of separate DDs -238,430 — it DIVERSIFIES the existing ' +
      'short-vol book rather than doubling it. (Caveat: the 11-year NAS-OPT SYNTHETIC came out net-negative, ' +
      'contradicting the real 58-day +EV, so the combined TOTALS are not trusted — but the near-zero overlap ' +
      'does not depend on magnitudes.)',
      'This is the variance risk premium — real, well-documented, and the textbook "pennies in front of a ' +
      'steamroller". 11 years is not enough to see the steamroller.',
      'The engine assumes the IV term structure (short-dated ~1.5x VIX, far-dated ~1.0x) is stable across 11 ' +
      'years. It was fitted on 58 days.',
    ],
    githubLinks: [
      {
        label: 'research/80 — all scripts, logs and RESULTS.md',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/80_farDTE_rescue',
      },
      {
        label: 'research/79 — the study that raised the question',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/79_nasopt_full_replay',
      },
    ],
    projectPaths: [
      'research/80_farDTE_rescue/results/RESULTS.md',
      'research/80_farDTE_rescue/scripts/r80_phase1.py',
      'research/80_farDTE_rescue/scripts/r80_overnight.py',
      'research/80_farDTE_rescue/scripts/r80_capital.py',
      'research/80_farDTE_rescue/scripts/r80_engine.py',
    ],
  },
  {
    slug: 'nasopt-full-replay',
    title: 'NAS-OPT — replayed on every recorded chain day (is the 0/1-DTE gate the edge?)',
    verdict:
      'STRATEGY (as defined) — and the DTE gate IS the strategy, not a detail. Replayed on all 58 recorded ' +
      'chain days: on 0/1-DTE (the days it actually trades) it makes +Rs1,578/day and wins 68% of the time. ' +
      'On every OTHER day it LOSES money (-Rs441/day, win 33%). Trading it every day does not dilute the ' +
      'edge, it CANCELS a third of it: the total falls from +Rs39,440 to +Rs24,871. The decay with ' +
      'days-to-expiry is clean and monotone — DTE0 +2,045, DTE1 +1,147, DTE4 +494, DTE5 -953, DTE6 -865. ' +
      'Near expiry theta is fat and the index rarely travels far enough to hit the +/-0.4% band; far from ' +
      'expiry the premium is rich, the band gets hit anyway, and the move-stop is pure cost. This settles ' +
      'the 2026-07-14 decision to paper-trade NAS-OPT on every weekday: those extra days stay tagged ' +
      'OBSERVATIONAL and must never be counted into the system result.',
    status: 'COMPLETE',
    date: '2026-07-14',
    cardBlurb:
      'NAS-OPT only enters on expiry day and the day before. Is that restriction the edge, or just a habit? ' +
      'Replayed the live rules on all 58 recorded chain days — validated first against the 11 days it really ' +
      'paper-traded. The gate is the edge: every other day is EV-negative.',
    cardStats: [
      { label: '0/1-DTE (the system)', value: '+Rs1,578/day' },
      { label: 'DTE>=2 (all other days)', value: '-Rs441/day' },
      { label: 'Verdict', value: 'DTE gate = the edge' },
    ],
    system: {
      intro:
        'NAS-OPT (research/54) sells a ~100pt-OTM NIFTY strangle at 09:20 and manages it with a single ' +
        'underlying move-stop — no per-leg premium SL. It enters ONLY on 0/1-DTE days. This study asks ' +
        'whether that DTE restriction is doing the work.',
      rows: [
        { k: 'Entry', v: '09:20 — SELL ~100pt-OTM strangle (2 strikes either side of ATM), front weekly expiry' },
        { k: 'Risk control', v: '+/-0.4% UNDERLYING move-stop — closes BOTH legs. Checked every minute 09:21-14:44' },
        { k: 'No premium SL', v: 'deliberate (research/54): the band is the sole control, one-and-done' },
        { k: 'Exit', v: '14:45 time-exit if the band was never hit' },
        { k: 'Sizing', v: '2 lots/leg = 130 qty (LOT 65). Net of Rs80/leg round-trip brokerage' },
        { k: 'The question', v: 'live it trades 0/1-DTE only. What happens on the other 33 days?' },
      ],
    },
    conditions: {
      intro:
        'Replayed against the REAL recorded option chain, minute by minute — the same data the live paper ' +
        'book marks itself from, so live == replay by construction.',
      rows: [
        { k: 'Data', v: 'options_data.db chain recorder — 58 days, 2026-04-20 to 2026-07-14' },
        { k: 'Spot', v: 'underlying_spot column, 100% populated on every day in the window' },
        { k: 'Cadence', v: '~370 snapshots/day (roughly 1/min); the move-stop is evaluated on every one' },
        { k: 'Costs', v: 'Rs80/leg round-trip brokerage. No extra slippage modelled (see caveats)' },
        { k: 'VALIDATION', v: '11 of the 58 days were really paper-traded live — replay checked leg-for-leg against that record BEFORE trusting the other 47' },
      ],
    },
    comparisons: [
      {
        title: 'Validation — replay vs the 11 days actually paper-traded live',
        caption:
          'Same strikes on 8/11 days, and there the difference is under Rs300. The 3 mismatches are NOT a ' +
          'coding error: the 09:20 spot sat near a 25-pt boundary, so ATM rounding flipped the strikes by 50. ' +
          'That is a real fragility of NAS-OPT — which tick it reads at 09:20 can swing a day by thousands.',
        columns: ['Day', 'Live strikes', 'Replay strikes', 'Live P&L', 'Replay P&L', 'Diff'],
        rows: [
          ['2026-06-08', '23000P/23200C', '23050P/23250C', '-2,390', '+477', '+2,867'],
          ['2026-06-15', '23850P/24050C', 'same', '+2,895', '+2,460', '-435'],
          ['2026-06-16', '23850P/24050C', 'same', '+4,702', '+4,644', '-58'],
          ['2026-06-22', '24050P/24250C', '24000P/24200C', '+3,617', '+2,973', '-644'],
          ['2026-06-23', '23950P/24150C', '24000P/24200C', '-2,123', '-4,222', '-2,099'],
          ['2026-06-29', '24000P/24200C', 'same', '-1,948', '-1,987', '-39'],
          ['2026-06-30', '23850P/24050C', 'same', '+6,041', '+6,138', '+97'],
          ['2026-07-06', '24250P/24450C', 'same', '-1,512', '-1,616', '-104'],
          ['2026-07-07', '24350P/24550C', 'same', '+3,110', '+3,064', '-46'],
          ['2026-07-13', '23950P/24150C', 'same', '-1,993', '-1,973', '+20'],
          ['2026-07-14', '24000P/24200C', 'same', '+4,962', '+4,676', '-286'],
          ['TOTAL', '', '', '+15,361', '+14,634', 'within 5%'],
        ],
        highlightRows: [11],
      },
      {
        title: 'The edge decays monotonically with days-to-expiry',
        caption:
          'Not a peak, not noise — a structure. DTE 2 and 3 simply do not occur in the window (Tue expiry), ' +
          'so there is a gap in the curve.',
        columns: ['DTE at entry', 'Weekday', 'Days', 'Total', 'Mean/day', 'Win%'],
        rows: [
          ['0', 'Tue (expiry)', '12', '+24,535', '+2,045', '67%'],
          ['1', 'Mon', '13', '+14,905', '+1,147', '69%'],
          ['4', 'Fri', '11', '+5,429', '+494', '55%'],
          ['5', 'Thu', '11', '-10,478', '-953', '27%'],
          ['6', 'Wed', '11', '-9,520', '-865', '18%'],
        ],
        highlightRows: [0, 1],
        heatmap: true,
      },
      {
        title: 'The move-stop is not broken — it is protecting positions that should not exist',
        caption:
          'On the days the system actually trades, the move-stop costs only -Rs325/day. On far-DTE days it ' +
          'costs -Rs1,539/day. The stop is being asked to defend a bad entry.',
        columns: ['Slice', 'Held to 14:45 (days)', 'Held mean', 'Move-stop (days)', 'Move-stop mean'],
        rows: [
          ['0/1-DTE (the system)', '11', '+3,999', '14', '-325'],
          ['DTE>=2 (observational)', '11', '+1,753', '22', '-1,539'],
        ],
        highlightRows: [0],
      },
    ],
    results: {
      metrics: [
        { label: '0/1-DTE mean/day', value: '+Rs1,578', tone: 'pos', hint: '25 days, 68% win' },
        { label: 'DTE>=2 mean/day', value: '-Rs441', tone: 'neg', hint: '33 days, 33% win' },
        { label: '0/1-DTE total', value: '+Rs39,440', tone: 'pos', hint: 'vs +Rs24,871 if traded every day' },
        { label: 'Cost of trading all days', value: '-Rs14,569', tone: 'neg', hint: 'a third of the edge, cancelled' },
        { label: 'Worst day (system)', value: '-Rs4,222', hint: '2026-06-23, DTE0' },
        { label: 'Days replayed', value: '58', hint: '2026-04-20 to 2026-07-14' },
      ],
      tables: [
        {
          title: 'Headline — the DTE gate is the strategy',
          columns: ['Slice', 'Days', 'Total', 'Mean', 'Median', 'Win%', 'Worst'],
          rows: [
            ['SYSTEM (0/1-DTE)', '25', '+39,440', '+1,578', '+2,024', '68%', '-4,222'],
            ['OBSERVATIONAL (DTE>=2)', '33', '-14,569', '-441', '-823', '33%', '-4,866'],
            ['ALL DAYS', '58', '+24,871', '+429', '-160', '48%', '-4,866'],
          ],
          highlightRows: [0],
        },
      ],
      charts: [
        {
          src: '/app/nasopt-full-replay.png',
          caption:
            'Cumulative P&L (0/1-DTE only vs every day), mean P&L by DTE, and where the move-stop actually ' +
            'costs money. 58 days, 2 lots/leg, net of brokerage.',
        },
      ],
    },
    winners: [
      {
        config: 'Keep the 0/1-DTE gate exactly as it is',
        summary:
          'The restriction is not a habit inherited from research/54 — it IS the edge. Every day outside it ' +
          'is EV-negative, and adding those days cancels a third of the profit.',
        metrics: [
          { k: '0/1-DTE', v: '+Rs1,578/day, 68% win, 25 days' },
          { k: 'Everything else', v: '-Rs441/day, 33% win, 33 days' },
          { k: 'DTE0 alone', v: '+Rs2,045/day (the strongest day)' },
        ],
        rejected: [
          'Trading NAS-OPT on all weekdays as a SYSTEM — cancels a third of the edge (+39,440 -> +24,871)',
          'Reading the all-weekday paper record as the strategy result — those days stay OBSERVATIONAL',
        ],
      },
    ],
    caveats: [
      '58 days, ONE benign regime (Apr-Jul 2026). No crash, no vol spike. A naked-ish 0.4% band is exactly ' +
      'the structure a gap destroys, and no gap is in this sample.',
      'ATM-ROUNDING FRAGILITY: if the 09:20 spot sits near a 25-pt boundary, the strikes flip by 50 and the ' +
      'day swings by thousands. 3 of the 11 validation days did exactly this. This is a property of the ' +
      'strategy, not of the test — and it means day-level P&L is genuinely noisy.',
      'Only Rs80/leg brokerage is charged. Fills are the recorded chain print at the trigger snapshot. A ' +
      'proper slippage sensitivity is OWED before this is called a validated STRATEGY at G4.',
      'DTE 2 and 3 do not occur in the window (Tuesday expiry), so the decay curve has a gap between 1 and 4.',
      'The 33 observational days are a REPLAY, not a live record — the live all-weekday paper book only ' +
      'started on 2026-07-14.',
    ],
    githubLinks: [
      {
        label: 'research/79 — runner + per-day CSV + RESULTS.md',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/79_nasopt_full_replay',
      },
      {
        label: 'services/nas_opt.py — the live system being replayed',
        href: 'https://github.com/castroarun/Quantifyd/blob/main/services/nas_opt.py',
      },
    ],
    projectPaths: [
      'research/79_nasopt_full_replay/scripts/run_nasopt_replay.py',
      'research/79_nasopt_full_replay/results/nasopt_daily.csv',
      'research/79_nasopt_full_replay/results/RESULTS.md',
      'services/nas_opt.py',
    ],
  },
  {
    slug: 'nas-sl-reanchor',
    title: 'NAS 09:16 Straddle — Is the 30% stop-loss too loose? (SL tightening / re-anchor)',
    verdict:
      'NO EDGE (regime-unstable). The idea — that a 30% stop anchored to the MORNING premium sits far too ' +
      'wide once theta has banked the gain — is intuitively right, and the SL record is ugly (avg −Rs7,526 ' +
      'per ATM stop-out). But no tested alternative beats it robustly. Continuous tightening ' +
      '(trail-to-breakeven, ratchet) is REFUTED — it whipsaws you out of legs that recover (win rate ' +
      'collapses to 27–33%). A single LATE re-anchor at 12:00 looked like a winner (+Rs1,225/lot) — until ' +
      'the shadow on ACTUAL traded legs showed the benefit FLIPS SIGN across periods: −Rs8,831/day ' +
      '(Apr–May) vs +Rs7,246/day (Jun–Jul), ≈ −Rs457/day over the full 48-day real record. The favourable ' +
      'result was a WINDOW ARTIFACT: the chain reconstruction needs a 09:15–09:18 quote window, and the ' +
      'Apr–May days start at 09:20, so they were silently dropped — fitting the winning regime. ' +
      'Ex-ante gating (CPR width, gap, VIX, opening-range) has ZERO predictive power. DO NOT deploy. ' +
      'A zero-risk shadow now runs daily on the real legs to accumulate out-of-sample evidence.',
    status: 'COMPLETE',
    date: '2026-07-10',
    cardBlurb:
      'Does tightening the per-leg stop intraday beat the fixed 30%-off-morning stop? Tested on real option ' +
      'premiums, a 743-day synthetic, and a shadow over the actual traded legs. Answer: no — the winner was a ' +
      'sampling artifact and the edge flips sign by regime.',
    cardStats: [
      { label: 'Verdict', value: 'NO EDGE' },
      { label: 'Full real record', value: '−Rs457/day' },
      { label: 'Sign flip', value: '−8,831 → +7,246/day' },
    ],
    system: {
      intro:
        'The live NAS 09:16 systems sell an ATM straddle at 09:16 and manage each leg with a per-leg stop ' +
        'fixed at 30% above its ENTRY (morning) premium, squaring off at 15:15. The question: once decay has ' +
        'banked most of the gain, that stop sits miles above the now-cheap premium — should it be brought in?',
      rows: [
        { k: 'Entry', v: 'SELL ATM straddle (CE+PE) at 09:16, front weekly expiry' },
        { k: 'Current stop (baseline)', v: 'per leg: SL = entry premium x 1.3 (30% off MORNING premium), fixed all day' },
        { k: 'Exit', v: 'EOD square-off 15:15 (or per-leg stop, whichever first)' },
        { k: 'Sizing', v: 'reported per 1 lot = 65 qty (live runs 2 lots)' },
        { k: 'All DTE', v: 'systems trade every weekday — DTE 0 through 4 all included' },
      ],
    },
    conditions: {
      intro: 'Three independent evidence bases were used, deliberately, because each has a different blind spot.',
      rows: [
        { k: 'A. Real option premiums', v: 'options_data.db chain recorder; ATM derived from the chain (strike where CE≈PE)' },
        { k: 'B. Synthetic (long history)', v: 'Black-Scholes on NIFTY 5-min spot + REAL India VIX as IV; 743 days (2023-01→2026-03)' },
        { k: 'C. Actual traded legs', v: 'the 916 systems real recorded legs replayed against their real chain paths — 48 days, 396 legs' },
        { k: 'Cost', v: '0.15% per leg per transaction; sensitivity run at 0.10 / 0.15 / 0.20%' },
        { k: 'Policies tested', v: 'baseline 30%; trail-to-breakeven; ratchet (0.2/0.3/0.4); re-anchor at 11:15/12:00/13:00/14:00; profit-lock 40/50/60%' },
      ],
    },
    comparisons: [
      {
        title: 'Per-leg SL policies on REAL premiums (n=27 days) — this is what looked like a winner',
        caption:
          'Re-anchor = at time T, reset the stop to 30% above the THEN-CURRENT premium (only tightens if the leg has decayed). ' +
          'Net of 0.15%/leg, per 1 lot. NOTE: this 27-day sample later proved to be the favourable regime only.',
        columns: ['Policy', 'Mean net/lot', 'vs baseline', 'Win %', 'Worst day'],
        rows: [
          ['Re-anchor @ 12:00', '+2,678', '+1,225', '74%', '−3,566'],
          ['Re-anchor @ 13:00', '+2,458', '+1,005', '78%', '−8,644'],
          ['Re-anchor @ 14:00', '+1,815', '+362', '70%', '−17,510'],
          ['BASELINE (30% off morning)', '+1,453', '0', '70%', '−17,510'],
          ['Profit-lock 50%', '+1,432', '−21', '63%', '−10,164'],
          ['Trail-to-breakeven', '+736', '−717', '33%', '−10,164'],
          ['Ratchet 0.2 / 0.3 / 0.4', '−72 / −402 / −284', '−1,525 to −1,855', '37–56%', '—'],
        ],
        highlightRows: [0],
        heatmap: false,
      },
      {
        title: 'THE KILLER — shadow on ACTUAL traded legs (48 days, 396 real legs, zero orders placed)',
        caption:
          'Every real 916 leg replayed against its own chain path within [entry_time, exit_time]; only legs LIVE at 12:00 are re-anchored. ' +
          'The benefit flips sign between halves of the record — that is regime instability, not a bug.',
        columns: ['Window', 'Days', 'Actual (30% SL)', 'Shadow (re-anchor 12:00)', 'Diff'],
        rows: [
          ['Apr 20 – Jun 02', '23', '+247,869', '+44,752', '−203,117  (−8,831/day)'],
          ['Jun 03 – Jul 10', '25', '+123,433', '+304,603', '+181,170  (+7,246/day)'],
          ['FULL RECORD', '48', '+371,302', '+349,355', '−21,947  (−457/day)'],
        ],
        highlightRows: [2],
        heatmap: false,
      },
      {
        title: 'Can we GATE it? Ex-ante predictors of when the re-anchor helps — all dead',
        caption:
          'Correlation of the per-day re-anchor benefit with features known BEFORE noon. Bucket tables showed tempting ' +
          'patterns (gap-up>60 +2,322; open-above-CPR +2,021; narrow-CPR +1,436) but they are non-monotonic at n≈10/bucket ' +
          'with ~zero correlation — noise. Gating on these would be curve-fitting.',
        columns: ['Feature (known ex-ante)', 'Correlation with benefit', 'Usable as a gate?'],
        rows: [
          ['CPR width', '−0.03', 'No'],
          ['Gap up / down', '+0.06', 'No'],
          ['Opening-range move (09:16–09:46)', '−0.10', 'No'],
          ['Day excursion', '+0.05', 'No'],
          ['India VIX', 'only 6 usable days', 'No'],
          ['Afternoon move (close − noon)', '+0.41 to +0.56', 'YES — but unknowable at noon'],
        ],
        heatmap: false,
      },
      {
        title: 'What the re-anchor actually IS — tail insurance, driven only by the afternoon move',
        caption: 'Neutral on 63% of days (never triggers). Its whole value sits in the extremes.',
        columns: ['Afternoon move (|close − noon|)', 'Benefit / lot', 'Read'],
        rows: [
          ['< 30 pts', '0', 'never fires — free'],
          ['30 – 60 pts', '−1,178', 'WHIPSAW zone — stops a leg that recovers'],
          ['60 – 100 pts', '+122', 'roughly neutral'],
          ['> 100 pts', '+1,812 to +3,176', 'the saves — this is the insurance paying out'],
        ],
        highlightRows: [3],
        heatmap: false,
      },
    ],
    results: {
      metrics: [
        { label: 'Verdict', value: 'NO EDGE', tone: 'neg' },
        { label: 'Full real record (48d)', value: '−Rs457/day', tone: 'neg' },
        { label: 'Apr–May half', value: '−Rs8,831/day', tone: 'neg' },
        { label: 'Jun–Jul half', value: '+Rs7,246/day', tone: 'pos' },
        { label: 'Days re-anchor does nothing', value: '63%' },
        { label: 'Ex-ante gate found', value: 'None' },
      ],
      tables: [
        {
          title: 'The live 30% stop IS expensive — but nothing tested beats it robustly',
          caption: 'Actual recorded paper-trade legs across 49 trading days.',
          columns: ['System', 'SL hits', 'Avg loss per stop-out', 'Avg premium rise at stop', 'Avg EOD winning leg'],
          rows: [
            ['916-ATM', '36', '−Rs7,526', '32%', '+Rs7,764'],
            ['916-ATM2', '70', '−Rs3,606', '33%', '+Rs6,199'],
            ['916-ATM4', '54', '−Rs5,772', '26%', '+Rs6,774'],
          ],
          heatmap: false,
        },
        {
          title: 'Days it saved vs days it cost (idealised 27-day window)',
          caption: 'The asymmetry that made it look great — before the sign-flip was discovered.',
          columns: ['Day', 'Baseline', 'Re-anchor', 'Effect'],
          rows: [
            ['2026-06-03 (Wed, DTE1)', '−17,510', '−1,141', '+16,369 SAVED'],
            ['2026-06-25 (Thu, DTE0)', '−11,463', '+4,079', '+15,542 SAVED'],
            ['2026-06-10 (Wed, DTE1)', '−9,668', '−2,091', '+7,577 SAVED'],
            ['2026-06-04 (Thu)', '+11,059', '+6,958', '−4,101 clipped'],
            ['2026-07-09 (Thu)', '+8,453', '+5,101', '−3,353 clipped'],
          ],
          heatmap: false,
        },
      ],
    },
    winners: [
      {
        config: 'NO WINNER — keep the existing 30% stop',
        summary:
          'The re-anchor-at-12:00 rule beat the baseline by +Rs1,225/lot on the reconstructed sample and then failed ' +
          'out-of-window on the actual traded legs. Nothing tested is a robust improvement, so nothing changes.',
        metrics: [
          { k: 'Full real record', v: '−Rs457/day (≈ neutral)' },
          { k: 'Stability', v: 'FAILS — sign flips by period' },
          { k: 'Gate available?', v: 'No — all ex-ante features ~0 correlation' },
          { k: 'Action', v: 'Do NOT change the live SL' },
        ],
        rejected: [
          'Trail-to-breakeven — whipsaws, win rate collapses to 27–33%',
          'Ratchet SL (0.2 / 0.3 / 0.4) — worst of all, −1,525 to −1,855/lot',
          'Profit-lock at 40/50/60% decay — neutral to negative',
          'Re-anchor @ 12:00 / 13:00 — looked best, then failed the actual-leg shadow (regime-unstable)',
          'Gating by CPR width / gap / VIX / opening-range — zero predictive power (curve-fitting risk)',
        ],
      },
    ],
    caveats: [
      'SELF-CORRECTION: the headline "+Rs1,225/lot" was a WINDOW ARTIFACT. The chain reconstruction requires a ' +
        '09:15–09:18 quote window; the Apr–May days start at 09:20 and were silently dropped — so the sample WAS ' +
        'the favourable regime. Lesson: always check that a reconstruction day-drops are not correlated with the outcome.',
      'The 743-day synthetic (Black-Scholes + real VIX) ranked AGGRESSIVE tightening best — but BS sells at fair value ' +
        'and therefore has NO vol-risk-premium, which is precisely the strategy edge. That ranking was discarded as misleading.',
      'Real-premium sample is small (27 idealised days / 48 actual-leg days) and covers a single regime (2026 Apr–Jul).',
      'The actual-leg shadow assumes the leg would have been exited at the shadow-stop premium with no extra slippage.',
      'Whether the Apr–May underperformance is regime or residual data quality cannot be fully separated — but Apr-30 has ' +
        '334k chain rows (it just starts 09:20), so it is NOT corruption.',
    ],
    githubLinks: [
      { label: 'research/77 — SL tightening (STATUS + RESULTS)', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/77_sl_tightening' },
      { label: 'research/76 — exit-timing / churn sweep (prior)', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/76_early_peak_reentry' },
    ],
    projectPaths: [
      'research/77_sl_tightening/NAS_0916_STRADDLE_SL_TIGHTENING_SWEEP_STATUS.md',
      'research/77_sl_tightening/RESULTS.md  (P1 — 15-day real)',
      'research/77_sl_tightening/RESULTS_P2.md  (P2 — 743-day synthetic)',
      'research/77_sl_tightening/RESULTS_P3.md  (P3 — grouping + actual-leg shadow → NO EDGE)',
      'scripts/sl_reanchor_shadow.py  (zero-risk daily shadow, cron 15:45 Mon–Fri)',
      'research/77_sl_tightening/results/shadow_log.csv  (forward evidence, accumulating)',
    ],
  },

  {
    slug: 'nifty-weekly-cpr-playbook',
    title: 'NIFTY Weekly CPR — Playbook (weekly × daily CPR + 1st-30-min candle)',
    verdict:
      'A no-trend, structure-selection playbook for NIFTY weekly options. The weekly CPR (lines drawn for the week from the prior week) sets context: narrow CPR -> the week TRENDS, wide CPR -> it goes SIDEWAYS/contained (validated with net-move & containment, NOT high-low range). The 1st-30-min candle gives the directional read two ways: its POSITION vs the weekly CPR (which side it closes, ~69% above / ~58% below) and its COLOR (green vs red = whether the week actually TRAVELS that way). Daily CPR confluence is the gate: when the Monday open sits on the same side of BOTH the weekly and daily CPR the week holds direction 72% (bull) / 61% (bear); when they SPLIT it is a 52% coin-flip. The two extra layers are orthogonal: daily confluence drives the HOLD rate, candle color drives the NET TRAVEL. Together they sort every week into a structure: confluence+agree-color -> directional defined-risk (jade / vertical); confluence+opposite-color or split -> neutral premium (iron condor / fly). SIGNAL/context tool — edge is in DIRECTION & structure choice, not magnitude (~+/-0.4% net); option P&L still needs real premiums.',
    status: 'COMPLETE',
    date: '2026-06-18',
    cardBlurb:
      'NIFTY weekly CPR as a structure-selection map: weekly CPR (narrow=trend / wide=sideways) + the 1st-30-min candle (position vs CPR for side, color for conviction) + daily-CPR confluence as the tradeable-vs-coinflip gate. Confluence+green -> ~72% bullish hold; confluence+red -> ~61% bearish; split -> 52% coin-flip (go neutral). 11 years (2015-26), NIFTY 5-min resampled to weekly.',
    cardStats: [
      { label: 'Bull confluence+green hold', value: '72%' },
      { label: 'Bear confluence+red hold', value: '61%' },
      { label: 'Split (coin-flip) hold', value: '52%' },
    ],
    system: {
      intro: 'Causal weekly read, fixed by Monday 09:45 IST. CPR width = |2C-H-L|/3 of the prior period (= how far it closed from mid-range = a trending-close measure). All levels from the prior week / prior day.',
      rows: [
        { k: 'STATE legend (read first)', v: 'AGREE-UP = price closed ABOVE both the weekly CPR and the daily CPR; AGREE-DOWN = BELOW both; DISAGREE = above one but below/inside the other. AGREE = a tradeable directional lean; DISAGREE (the old "coin-flip") = no reliable direction -> trade neutral. "Confluence" anywhere on this card == AGREE; "coin-flip/split" == DISAGREE.' },
        { k: 'Weekly CPR', v: 'Band (BC..TC) drawn for the week from the prior week H/L/C — the lines on the chart. Narrow = trend expected; wide = sideways/contained.' },
        { k: '1st-30-min position', v: 'Monday 09:15-09:45 close vs the weekly CPR band: above / below / inside = which side the week leans.' },
        { k: '1st-30-min color', v: 'green = 09:45 close > 09:15 open (conviction up); red = down. Color predicts NET TRAVEL, not which side it closes.' },
        { k: 'Daily CPR confluence', v: 'Monday close also vs Monday daily CPR (from Friday). Same side as weekly = confluence (tradeable); opposite/inside = split (coin-flip).' },
        { k: 'Pivot levels', v: 'Weekly R1/R2/S1/S2 from prior week — used to place condor / fly / spread wings by their hit-rates.' },
        { k: 'Universe / window', v: 'NIFTY 50, 5-min bars resampled to weekly, Feb 2015 - Mar 2026 (~11y, 581 weekly bars; market_data.db NIFTY50 5min).' },
      ],
    },
    conditions: {
      intro: 'Price-action study (no option premiums). Movement in index %.',
      rows: [
        { k: 'Data', v: 'NIFTY50 5-minute (market_data.db on VPS) resampled to W-FRI weekly; daily CPR from 5-min daily resample.' },
        { k: 'Causality', v: 'CPR & pivots from prior period; signal fixed at Monday 09:45 — fully tradeable, no look-ahead.' },
        { k: 'Metrics', v: 'held-side = week closes on the signalled side of the weekly CPR; net = open->close %; maxBull/maxBear = excursion from the Monday-09:45 entry; pivot hit% = week H/L reaches the level.' },
      ],
    },
    comparisons: [
      {
        title: 'Candle color x position — narrow-CPR weeks (2015-26, n=233)',
        columns: ['1st-30-min candle', 'held its side', 'net move', 'read'],
        rows: [
          ['above + GREEN', '68%', '+0.42%', 'bullish'],
          ['above + RED', '72%', '-0.07%', 'neutral (holds, no travel)'],
          ['below + RED', '60%', '-0.37%', 'bearish'],
          ['below + GREEN', '65%', '+1.16% (n=17)', 'reversal-up (thin)'],
        ],
        highlightRows: [0, 2],
        caption: 'Color barely changes WHICH side it closes (~68-72% above either way) but flips the NET TRAVEL: green-above goes up (+0.42%), red-above just sits (neutral). Position = side; color = conviction.',
      },
      {
        title: 'Weekly x Daily CPR confluence — does the week hold direction?',
        columns: ['Monday 09:45 setup', '% of weeks', 'holds that side', 'net'],
        rows: [
          ['BOTH above (bull confluence)', '45%', '72%', '+0.2%'],
          ['BOTH below (bear confluence)', '29%', '61%', '-0.2%'],
          ['SPLIT (timeframes disagree)', '~7%', '52% (coin-flip)', '~0%'],
        ],
        highlightRows: [0, 2],
        caption: 'Daily CPR is the gate: agreement -> tradeable; disagreement -> coin-flip. vs weekly-alone baselines (above 69% / below 58%).',
      },
      {
        title: 'The combination — weekly position x daily confluence x candle color (2015-26)',
        columns: ['Setup', 'n', 'held side', 'net move'],
        rows: [
          ['weekly above (alone)', '313', '69%', '+0.18%'],
          ['+ daily confluence', '260', '72%', '+0.21%'],
          ['+ GREEN + daily confluence (ALL 3, bull)', '152', '72%', '+0.36%'],
          ['above + RED + daily conf (neutral)', '108', '70%', '-0.02%'],
          ['above + GREEN + daily DISagree', '17', '53%', '+0.17%'],
          ['+ RED + daily confluence (ALL 3, bear)', '136', '61%', '-0.40%'],
          ['below + GREEN + daily conf (reversal-up)', '35', '63%', '+0.66%'],
        ],
        highlightRows: [2, 5],
        caption: 'Orthogonal layers: daily confluence drives the HOLD rate (gate; without it green is a 53% coin-flip), candle color drives the NET TRAVEL (green-above +0.36 vs red-above -0.02). Best bull = above+daily-above+green (72%/+0.36%); best bear = below+daily-below+red (61%/-0.40%). below+green = reversal-up trap.',
      },
      {
        title: 'Coin-flip weeks — max move from entry + pivot-level hit-rates (for wing placement)',
        columns: ['Coin-flip scenario', 'n', 'maxBull avg/p90', 'maxBear avg/p90', 'R1% / R2%', 'S1% / S2%'],
        rows: [
          ['ABOVE coin-flip (wk+ daily not+)', '53', '1.08 / 1.88', '1.11 / 2.47', '43 / 19', '26 / 11'],
          ['BELOW coin-flip (wk- daily not-)', '32', '1.81 / 4.02', '1.73 / 2.98', '34 / 22', '47 / 22'],
        ],
        caption: 'Coin-flip weeks are CONTAINED (both-sides whipsaw only 6%) -> trade NEUTRAL not directional. ABOVE coin-flip: mild up-lean, S1 hit only 26% -> condor short put ~S1 / call ~R2. BELOW coin-flip: leans UP (+0.61% net) with a FAT upside tail (p90 4.0%) -> condor with call wing beyond R2; never go bear.',
      },
      {
        title: 'Intra-week daily re-check — re-classify at EACH day close, rest-of-week outcome (2015-26, 11y)',
        columns: ['Day (close)', 'Weekly CPR', 'Daily CPR', 'n', 'rest-of-week holds', 'rest net'],
        rows: [
          ['Mon', '🟢▲ above', '🟢▲ above', '271', '76%', '+0.14%'],
          ['Mon', '🟢▲ above', '🔴▼ below', '35', '57%', '+0.17%'],
          ['Mon', '🔴▼ below', '🔴▼ below', '186', '64%', '+0.27%'],
          ['Mon', '🔴▼ below', '🟢▲ above', '24', '33% → reverts UP', '+0.56%'],
          ['Tue', '🟢▲ above', '🟢▲ above', '227', '82%', '+0.19%'],
          ['Wed', '🟢▲ above', '🟢▲ above', '212', '88%', '+0.04%'],
          ['Thu', '🟢▲ above', '🟢▲ above', '206', '96%', '+0.02%'],
          ['Thu', '🔴▼ below', '🔴▼ below', '142', '82%', '+0.17%'],
        ],
        highlightRows: [3, 5, 6, 7],
        caption: '🟢▲ = the DAY CLOSE finished ABOVE that CPR, 🔴▼ = below. Both 🟢▲ = close above the weekly AND daily CPR (tradeable bull); both 🔴▼ = below both (tradeable bear); one of each = the timeframes DISAGREE (neutral / coin-flip). The state EVOLVES daily — 48% of weeks the weekly side flips at least once, so re-check each evening. Both-same beats mixed every day; the absolute hold% rises toward week-end (partly mechanical — less time left). Adjust cues: weekly 🔴▼ + daily 🟢▲ = reversal-UP (lift bearish risk / mild bull); when a mixed week RESOLVES to both-same, the new side tends to hold (pooled n=120: both-up 73%, both-down only 46%); a side-flip = re-center. Basis = each day CLOSE (NOT the 1st-30-min entry candle); 11y 2015-26. WED/THU both-same is the STRONGEST (hold 88-96%) = a HOLD-INTO-EXPIRY signal: if the confluence is intact by Wed/Thu, the Thu-Fri finish almost always holds that side — so keep / press the structure into expiry. (Late-week strength is partly MECHANICAL — fewer days left to change — which is exactly why it is safe to hold.) Friday is the terminal day, so it has no rest-of-week row.',
      },
      {
        title: 'Intra-week re-check — MORNING variant (each day 1st-30-min @ 09:45 close), rest-of-week (2015-26, 11y)',
        columns: ['Day (09:45)', 'Weekly CPR', 'Daily CPR', 'n', 'rest-of-week holds'],
        rows: [
          ['Mon', '🟢▲ above', '🟢▲ above', '259', '72%'],
          ['Mon', '🟢▲ above', '🔴▼ below', '27', '59%'],
          ['Mon', '🔴▼ below', '🟢▲ above', '16', '44% → reverts UP'],
          ['Mon', '🔴▼ below', '🔴▼ below', '172', '61%'],
          ['Tue', '🟢▲ above', '🟢▲ above', '220', '77%'],
          ['Wed', '🟢▲ above', '🟢▲ above', '217', '86%'],
          ['Thu', '🟢▲ above', '🟢▲ above', '193', '85%'],
          ['Thu', '🔴▼ below', '🔴▼ below', '104', '82%'],
        ],
        highlightRows: [2, 5, 6, 7],
        caption: 'Same re-check but on EACH day 1st-30-min close (09:45) instead of the day close — the MORNING read, for adjusting at the open. Hold rates run a touch lower than the day-close table (Mon both-🟢▲ 72% vs 76%; Thu 85% vs 96%) because 09:45 is earlier, with more of the week still to play (the day-close late-week numbers are partly mechanical). Same pattern: both-same > mixed; conviction rises through the week; weekly 🔴▼ + daily 🟢▲ = reversal-UP (Mon 44% holds below). The 09:45 weekly-side flips intra-week in 50% of weeks. Use the MORNING read to adjust at the open, the DAY-CLOSE table in the evening. WED/THU both-🟢▲ is the STRONGEST (hold 85-86%) = HOLD-INTO-EXPIRY (keep/press the structure into Fri); partly mechanical (less time left). Friday is the terminal day (no rest-of-week).',
      },
      {
        title: 'Intra-week BREACH = adjustment trigger, by candle timeframe (bull entry above weekly CPR; mirror with R1 for bear)',
        columns: ['Trigger — a later candle...', 'occurs', 'week ends below CPR', 'holds below S1'],
        rows: [
          ['30-min closes below S1', '21%', '72%', '52%'],
          ['1h closes below S1', '21%', '71%', '54%'],
          ['2h closes below S1', '20%', '73%', '56%'],
          ['4h closes below S1', '19%', '75%', '59%'],
          ['DAILY closes below S1', '17%', '79%', '67% (daily — only real step)'],
          ['any intraday close back below CPR (cross)', '35-38%', '52-57%', '—'],
          ['DAILY close back below CPR (cross)', '33%', '61%', '—'],
        ],
        highlightRows: [4],
        caption: 'Baseline: a bull (above-CPR) entry ends below CPR only ~20%. Within intraday (30m-4h) reliability barely moves (~71-75% end below CPR on an S1 breach) — NO magic intraday TF; the real step is the DAILY close (79%, holds S1 67%). Within intraday the holds-S1 gradient (52% at 30m to 59% at 4h) is within noise (n~65) — 30m/1h/2h/4h are EQUIVALENT. BEAR MIRROR (entry below CPR, trigger = close above R1): intraday ~85-87% end above CPR, DAILY 90% / holds R1 83%. A plain CPR cross = BIAS-FLIP (neutralize); an S1/R1 breach = CONTINUATION (flip & lean). LADDER: ANY intraday close beyond S1/R1 = EARLY alert (use whatever TF you watch — 30-min is fine, they are equivalent) -> DAILY close beyond S1/R1 = CONFIRM (flip). A plain CPR cross intraday (~52-57%) is WATCH-only; decisive only on the daily (61%). Bias-flip signal, not move-size (rest-move ~flat).',
      },
    ],
    results: {
      metrics: [
        { label: 'Bull (above+daily+green)', value: '72% hold / +0.36%', tone: 'pos' },
        { label: 'Bear (below+daily+red)', value: '61% hold / -0.40%', tone: 'pos' },
        { label: 'Split / no-confluence', value: '52% (coin-flip)' },
        { label: 'Daily confluence lift (hold)', value: '+3pp (69->72 / 58->61)' },
        { label: 'Coin-flip both-sides whip', value: '6% (contained)', tone: 'pos' },
        { label: 'Net move magnitude', value: '~+/-0.4% (tilt, not trend)' },
      ],
      tables: [
        {
          title: 'Structure-selection map (the playbook)',
          columns: ['Week classification', 'Read', 'Suggested structure'],
          rows: [
            ['above + daily-above + GREEN', 'bull tilt (72% / +0.36%)', 'bullish jade lizard / bull-put spread'],
            ['above + daily-above + RED', 'holds up but goes nowhere (70% / ~0)', 'iron condor / iron fly (sell range)'],
            ['below + daily-below + RED', 'bear tilt (61% / -0.40%)', 'bear-call / put debit (defined-risk)'],
            ['below + GREEN', 'reversal-up trap (+0.66%)', 'do NOT go bear — neutral or mild bull'],
            ['daily disagrees (split)', 'coin-flip (52%)', 'neutral only — condor / fly, wings at R1-R2 / S1-S2'],
            ['ultra-narrow CPR (top whipsaw decile)', 'whippy (66-74% cross both sides)', 'skip the directional break'],
          ],
          highlightRows: [0, 2],
        },
      ],
    },
    winners: [
      {
        config: 'Bull tilt: weekly-above + daily-above + GREEN 1st-30-min',
        summary: 'The cleanest directional setup — 72% close above the weekly CPR with a genuine +0.36% net travel (n=152, 11y). Daily confluence supplies the hold rate, the green candle supplies the conviction. Play it as a bullish jade lizard or bull-put spread.',
        metrics: [
          { k: 'Held above', v: '72%' },
          { k: 'Net travel', v: '+0.36%' },
          { k: 'n', v: '152 weeks' },
        ],
      },
      {
        config: 'Coin-flip / opposite-color -> NEUTRAL premium',
        summary: 'When daily disagrees (52% coin-flip) or the candle color opposes the position (above+red = 70% hold but ~0 net), the week is contained (both-sides whipsaw only 6%). Sell an iron condor / fly with wings at the weekly pivots (R1-R2 / S1-S2). The playbook always yields a structure — confluence+color just says directional vs neutral.',
        metrics: [
          { k: 'Coin-flip hold', v: '52%' },
          { k: 'Both-sides whip', v: '6%' },
          { k: 'above+red net', v: '~0%' },
        ],
      },
    ],
    caveats: [
      'PRICE-ACTION study, no option premiums — movement is index %, the option P&L / EV still needs real premiums (AlgoTest or the live recorder). These stats pick the STRUCTURE and place the WINGS; they do not prove the money.',
      'Edge is in DIRECTION & structure selection, not magnitude — net moves are small (~+/-0.4%). This is for inline premium structures (tilt/neutral), NOT trend-catching.',
      'Some splits are thin: below+green (n=17-35) and the coin-flip buckets (n=32-53); RECENT 2023-26 alone is too thin to split (treat as directional only). Headline numbers are the full 11y (2015-26).',
      'Single instrument (NIFTY), in-sample over one 11-year history. The bull/bear ASYMMETRY (bull ~72% vs bear ~61%) reflects equity upward drift and may not hold in a structural bear.',
      'Daily-vs-weekly CPR SIGN FLIP (research/67): weekly narrow = trend, daily narrow = calm — do not mix the timeframes’ interpretations.',
    ],
    githubLinks: [
      { label: '← Related: V2 Iron Fly (Stop-Loss x VIX)', href: '/app/backtest/v2-nifty-ironfly-sl-vix' },
      { label: 'research/67 — weekly vs daily CPR study', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/67_weekly_cpr' },
    ],
    projectPaths: [
      'research/67_weekly_cpr/results/RESULTS.md',
      'research/67_weekly_cpr/scripts/ (cpr_plan, cpr_ab, cpr_trend, prem2, + confluence/candle/3way/coinflip)',
    ],
  },
  {
    slug: 'factor-index-rotation',
    title: 'Nifty Factor-Index Rotation — does "diversify, don\'t select" transfer from assets to factors?',
    verdict:
      'Follow-on to the GTAA ETF study: does the "diversify, don\'t select" result transfer to the Nifty single factors (Momentum/Quality/Value/Low-Vol/Alpha)? No — on clean data the factors are ~0.8 correlated to each other and to the Nifty (mostly the same equity bet), so diversifying ACROSS factors fails (factor-only book Calmar 0.55). The value is purely swapping the SINGLE equity sleeve of the GTAA trio (Nifty → one factor) + Gold + Nasdaq, inverse-vol. Best clean sleeve = the VALUE factor: Calmar 1.83, CAGR 17.4%, MaxDD −9.5% (full 2015–26 window) — beating the Nifty book (1.50–1.57). Momentum is the higher-return alternative (20.0% / 1.60) but window-sensitive. Use ONE factor, not two. DATA-INTEGRITY NOTE: the Quality & Low-Vol Kite INDEX series were found corrupt (bad prints, 150%/308% daily vol) and excluded; the earlier "Low-Vol is the lone diversifier" claim is retracted (real Low-Vol ETF is 0.93-correlated to the Nifty). STRATEGY candidate — an incremental single-sleeve upgrade to research/63, not a standalone factor edge. ALL figures below are the consistent full 2015–26 window, inverse-vol, net 20 bps.',
    status: 'COMPLETE',
    date: '2026-06-14',
    cardBlurb:
      'Tests whether the GTAA "equal-weight beats selection" result extends to the Nifty factor indices. It doesn\'t — factors are ~0.8 correlated (the same Nifty bet), so factor-only books fail (Calmar 0.55). The win is swapping the single equity sleeve of the GTAA trio from Nifty to the Value factor, inverse-vol: Calmar 1.83 vs the Nifty book 1.50–1.57. Full 2015–26 window, net 20 bps.',
    cardStats: [
      { label: 'CAGR', value: '17.4%' },
      { label: 'MaxDD', value: '−9.5%' },
      { label: 'Calmar', value: '1.83' },
    ],
    system: {
      intro: 'Three families tested with the research/63 engine; the winner and why. All on the full 2015–26 window, inverse-vol, net 20 bps:',
      rows: [
        { k: 'Factor universe', v: 'Nifty200 Momentum 30, Nifty100 Quality 30, Nifty50 Value 20, Nifty100 LowVol 30, Nifty Alpha 50 (NSE index series). Quality & Low-Vol index series were CORRUPT (bad prints) → excluded; clean = Momentum/Value/Alpha.' },
        { k: '(a) Rotation', v: 'Top-N factors by momentum + trend gate — selection beats Nifty but is drawdown-bound (best ~0.67). Factors are too correlated for selection to add much.' },
        { k: '(b) Diversify across factors', v: 'Equal-weight / inverse-vol factor baskets FAIL: clean factor-only (Value+Momentum+Alpha) = Calmar 0.55, −24.9% DD — all the same equity beta.' },
        { k: '(c) Single-sleeve swap (WINNER)', v: 'Keep Gold + Nasdaq for diversification; swap the equity sleeve Nifty → the VALUE factor, inverse-vol, monthly. Calmar 1.83, CAGR 17.4%, MaxDD −9.5%.' },
        { k: 'Why it wins', v: 'The 1.5+ Calmar tier needs the cross-asset diversifiers (Gold+Nasdaq). Value is the lowest-vol/lowest-DD equity factor, so inverse-vol leans into it → the book\'s drawdown drops to −9.5%. Momentum is higher-return (20.0% / 1.60) but deeper DD. Adding ALL factors or a second factor dilutes the diversifiers.' },
        { k: 'Costs', v: '20 bps/side; low turnover. Synthetic 6%/yr cash leg where used.' },
      ],
    },
    conditions: {
      intro: 'Window and benchmark.',
      rows: [
        { k: 'Period', v: 'Full 2015–26 (Gold/Nasdaq ETF era; ~11.4y) for every combined-book figure on this page. Factor indices exist from 2010 but the combined book is gated by Gold/Nasdaq from 2015.' },
        { k: 'Benchmark', v: 'NIFTYBEES (Nifty 50) buy & hold; and the research/63 Nifty+Gold+Nasdaq equal-weight book.' },
        { k: 'Host', v: 'VPS market_data.db snapshot 2026-06-12 + Kite factor-index history; one canonical script (canonical.py) drives every number here.' },
      ],
    },
    comparisons: [
      {
        title: 'Equity sleeve in {sleeve + Gold + Nasdaq} — all 2015–26, inverse-vol, net 20 bps',
        caption: 'One window, one method, one source — directly comparable. Value is the best clean sleeve; Momentum the higher-return one; factor-only (no Gold/Nasdaq) collapses.',
        columns: ['Book', 'CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['Value + Gold + Nasdaq (WINNER)', '17.4%', '−9.5%', '1.83'],
          ['Momentum + Gold + Nasdaq', '20.0%', '−12.5%', '1.60'],
          ['Nifty + Gold + Nasdaq, equal (research/63)', '17.6%', '−11.3%', '1.57'],
          ['Alpha + Gold + Nasdaq', '20.8%', '−13.7%', '1.52'],
          ['Nifty + Gold + Nasdaq, inverse-vol', '16.8%', '−11.2%', '1.50'],
          ['Factor-only (Value+Momentum+Alpha, NO assets)', '13.8%', '−24.9%', '0.55'],
          ['Nifty 50 alone', '10.1%', '−28.8%', '0.35'],
        ],
        highlightRows: [0, 5],
        heatmap: false,
      },
      {
        title: 'Per-year: Value winner vs NIFTY 50 (2015–26)',
        columns: ['Year', 'Winner %', 'NIFTYBEES %', 'Excess pp'],
        rows: [
          ['2015', '+0.1', '−5.6', '+5.7'],
          ['2016', '+15.8', '+4.0', '+11.8'],
          ['2017', '+18.2', '+29.9', '−11.7'],
          ['2018', '−0.1', '+4.8', '−4.9'],
          ['2019', '+18.2', '+13.6', '+4.6'],
          ['2020', '+30.5', '+15.4', '+15.1'],
          ['2021', '+17.7', '+26.0', '−8.3'],
          ['2022', '−1.8', '+5.5', '−7.3'],
          ['2023', '+28.0', '+21.0', '+7.0'],
          ['2024', '+31.6', '+10.4', '+21.2'],
          ['2025', '+26.4', '+11.7', '+14.7'],
          ['2026*', '+10.4', '−9.3', '+19.7'],
        ],
        highlightRows: [0],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR', value: '17.4%', tone: 'pos' },
        { label: 'NIFTYBEES CAGR', value: '11.4%' },
        { label: 'Excess / yr', value: '+6.0%', tone: 'pos' },
        { label: 'Sharpe', value: '1.53', tone: 'pos' },
        { label: 'Max Drawdown', value: '−9.5%', tone: 'neg', hint: 'vs NIFTYBEES −28.8%' },
        { label: 'Calmar', value: '1.83', tone: 'pos' },
        { label: 'vs research/63', value: '1.83 vs 1.57', hint: 'Value sleeve > Nifty sleeve' },
        { label: 'Mean factor corr', value: '~0.8', hint: 'clean; vs asset trio ~0.1' },
      ],
      tables: [
        {
          title: 'Winner vs research/63 GTAA (both 2015–26, inverse-vol vs equal)',
          columns: ['Metric', 'Value+Gold+Nasdaq (inv-vol)', 'Nifty+Gold+Nasdaq (equal)'],
          rows: [
            ['CAGR', '17.4%', '17.6%'],
            ['MaxDD', '−9.5%', '−11.3%'],
            ['Calmar', '1.83', '1.57'],
            ['Sharpe', '1.53', '1.45'],
          ],
          highlightRows: [1, 2],
        },
        {
          title: 'Factor cross-correlation (monthly, CLEAN data) — factors are mostly the same Nifty bet',
          caption: 'Quality & Low-Vol index series were CORRUPT and are excluded; clean factors are ~0.8 correlated to each other and to the Nifty — so diversifying across factors does not cut drawdown.',
          columns: ['', 'Mom', 'Value', 'Alpha', 'Nifty'],
          rows: [
            ['Momentum', '1.00', '0.77', '0.91', '0.84'],
            ['Value', '0.77', '1.00', '0.73', '0.89'],
            ['Alpha', '0.91', '0.73', '1.00', '0.80'],
            ['Nifty', '0.84', '0.89', '0.80', '1.00'],
          ],
          highlightRows: [],
          heatmap: true,
        },
        {
          title: 'Real factor-ETF check (2022-08→2026-06, ~3.8y BULL — read ranks, not levels)',
          caption: 'Pulled the real factor ETFs (max history) to test Low-Vol/Quality properly. Short bull window inflates all Calmars (3.5–4.9); the takeaways are the RANKING and that real Low-Vol = 13.6% vol & 0.93 corr to Nifty → NOT a diversifier (buries the corrupt-index claim). Factor-only stays poor: Calmar 0.61 vs 4.49 baseline.',
          columns: ['Equity sleeve (+Gold+Nasdaq, equal)', 'CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Value (MOVALUE)', '36%', '−7%', '4.92'],
            ['Low-Vol (LOWVOL1)', '29%', '−6%', '4.71'],
            ['Nifty (baseline)', '28%', '−6%', '4.49'],
            ['Quality (SBIETFQLTY)', '28%', '−7%', '3.95'],
            ['Momentum (MOMOMENTUM)', '29%', '−8%', '3.53'],
          ],
          highlightRows: [0],
        },
      ],
      charts: [
        {
          src: '/app/factor-gtaa-factsheet.png',
          caption:
            'CLIENT FACTSHEET — Factor GTAA: Value + Gold + Nasdaq (inverse-vol) vs NIFTY 50, full 2015–2026, net 20 bps. KPI strip, growth-of-₹1 (log), drawdown-vs-index, annual bars, monthly heatmap, rolling 12m, stat tables. 17.4% CAGR vs 11.4%, Calmar 1.83, MaxDD −9.5%. Generated by research/_utilities/tearsheet.py.',
        },
      ],
    },
    winners: [
      {
        config: 'Value factor + Gold + Nasdaq · inverse-vol · monthly (full 2015–26)',
        summary: 'The 1.5+ Calmar tier requires the cross-asset diversifiers (Gold+Nasdaq); given those, the Value factor is the best CLEAN equity sleeve (lowest vol/DD → inverse-vol leans into it → −9.5% book DD). Momentum is the higher-return alternative (20.0% / 1.60). Both beat the Nifty sleeve. Use one factor, not two.',
        metrics: [
          { k: 'CAGR', v: '17.4%' },
          { k: 'Excess', v: '+6.0%/yr vs NIFTYBEES' },
          { k: 'Sharpe', v: '1.53' },
          { k: 'MaxDD', v: '−9.5%' },
          { k: 'Calmar', v: '1.83' },
        ],
        rejected: [
          'Diversify across factors: clean factor-only (Value+Momentum+Alpha) = Calmar 0.55 — factors are ~0.8 correlated (the same Nifty bet), so equal-weighting them does not cut the −25% drawdown.',
          'Two factors instead of one: a second factor just adds correlated equity beta and crowds out Gold/Nasdaq — worse than a single factor sleeve.',
          'Sensex / BSE500 / Nifty500 as the sleeve: corr 0.97–1.00 to the Nifty — literally the same bet, no change. (Sector/thematic sleeves explored separately; the apparent defensive-sector winners are an overfit multiple-testing artifact — see EXPLORATORY_indices_sleeve.md, not published.)',
          'Quality & Low-Vol index series: CORRUPT (150%/308% daily vol) — excluded; the corrupt-data "Low-Vol diversifier" claim is retracted.',
        ],
      },
    ],
    caveats: [
      'DATA INTEGRITY (2026-06-14): the Kite INDEX series for Quality and Low-Vol were CORRUPT (bad prints — 150% / 308% annualised daily vol, single-day prints to +472%) and are excluded; the earlier "Low-Vol is the lone diversifier (0.42–0.47 corr)" claim is retracted (real Low-Vol ETF is 0.93-correlated to the Nifty). Also caught the Commodities index as corrupt.',
      'Period dependence: 2015–26 was a benign decade. The research/63 21-year through-cycle test (with proxies) showed the asset-trio Calmar drops from ~1.7 to ~0.8 and MaxDD widens to ~−24% in a real crisis (2008). Treat the 1.83 here similarly — the structural finding (Value sleeve > Nifty sleeve; factor diversification fails) is the durable part, not the absolute level.',
      'Window note: an earlier draft of this page featured Momentum on a 2016–26 window (which dropped a flat 2015 warm-up year) and showed 22.1% / Calmar 1.77. Every figure here is now the consistent full 2015–26 window from one canonical script; on that basis Momentum is 20.0% / 1.60 and Value (1.83) is the best clean sleeve.',
      'Mixed data: factor sleeves are PRICE-return indices (understate dividends ~1.5%/yr); Gold/Nasdaq are ETF prices. A live version must use the factor ETF NAV (MOVALUE etc., short history → recheck tracking/capacity).',
      'Backtest, net of 20 bps modelled cost, synthetic 6%/yr cash leg. Nothing wired live. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'RESULTS.md (verdict + tables)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/64_factor_index_rotation/results/RESULTS.md',
      },
      {
        label: 'canonical.py (single source of truth)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/64_factor_index_rotation/scripts/canonical.py',
      },
    ],
    projectPaths: [
      'research\\64_factor_index_rotation\\FACTOR_INDEX_ROTATION_MONTHLY_SWEEP_STATUS.md',
      'research\\64_factor_index_rotation\\scripts\\ (canonical, g1_probe, g2_sweep, replace_nifty_test, factor_etf_test).py',
      'research\\64_factor_index_rotation\\results\\ (factor_corr_CLEAN.csv, RESULTS.md, EXPLORATORY_indices_sleeve.md)',
    ],
  },
  {
    slug: 'gtaa-etf-rotation',
    title: 'GTAA Multi-Asset ETF Rotation — validating (and beating) the Upstox "Strategy 1"',
    verdict:
      'A trading-course slide pitched a monthly top-1 momentum rotation over 3 ETFs (Nifty / Gold / Nasdaq-100) at "Calmar 0.93". We could not reproduce 0.93 (Kite serves these ETFs only from 2015) and in all testable history the top-1 design is WEAK: Calmar 0.30 (raw) / 0.44 (trend-gated), −34%/−25% drawdown. The fix is almost embarrassingly simple — drop the momentum SELECTION and just hold all three EQUAL-WEIGHT, rebalanced monthly: Calmar ~1.73, CAGR 19.5%, MaxDD only −11.3%, turnover ~0, and completely cost-insensitive. Three uncorrelated sleeves (Nifty/Gold −0.08, Nifty/Nasdaq +0.25, Gold/Nasdaq +0.04) mean diversification, not rotation, is the edge. STRATEGY candidate for a simple low-DD core mandate — same ~1.7 Calmar tier as our concentrated equity books (research/41, /62) at a fraction of their drawdown, turnover, tax and complexity.',
    status: 'COMPLETE',
    date: '2026-06-14',
    cardBlurb:
      'Validate a popular trading-course GTAA strategy (top-1 of Nifty/Gold/Nasdaq-100 by 12m momentum, monthly) and try to beat it. Finding: the top-1 selection underperforms; naive equal-weight of the same 3 ETFs, monthly-rebalanced, more than doubles the Calmar (1.73 vs 0.30–0.44) at −11% drawdown. Net of 20bps, 2016–2026.',
    cardStats: [
      { label: 'CAGR', value: '19.5%' },
      { label: 'MaxDD', value: '−11.3%' },
      { label: 'Calmar', value: '1.73' },
    ],
    system: {
      intro: 'The slide’s rules (what we validated) and the winning construction (what beats it):',
      rows: [
        { k: 'Universe', v: 'NIFTYBEES (Nifty 50), GOLDBEES (gold), MON100 (Motilal Oswal Nasdaq-100) — three low-correlation sleeves.' },
        { k: 'Slide signal', v: 'Monthly: rank by ROC(12), hold the single top asset; "bullish" = close > 6-month MA.' },
        { k: 'Winner', v: 'Drop selection entirely — hold all 3 equal-weight (1/3 each), rebalance monthly. (= top-N where N = universe size, so ROC/MA become irrelevant.)' },
        { k: 'Why it wins', v: 'Monthly-return correlations: Nifty/Gold −0.08, Nifty/Nasdaq +0.25, Gold/Nasdaq +0.04. Equal-weight harvests the diversification + rebalancing premium; top-1 concentrates into the hot asset then eats the reversal.' },
        { k: 'Rotation', v: 'Monthly rebalance on the month-end close; returns realised t→t+1 (no same-bar leak).' },
        { k: 'Costs', v: '20 bps/side modelled (winner turnover ≈ 5%/yr → cost-insensitive: same Calmar at 0/10/20/40 bps).' },
        { k: 'Backtest window', v: '2016-02 → 2026-06 (~10.4y) — Kite serves these ETFs only from 2015, so the slide’s longer/older window is not reproducible.' },
      ],
    },
    conditions: {
      intro: 'Backtest window and benchmark.',
      rows: [
        { k: 'Period', v: 'Feb 2016 – Jun 2026 (~10.4 years), after a 12-month momentum warm-up.' },
        { k: 'Benchmark', v: 'NIFTYBEES (Nifty 50) buy & hold, same window.' },
        { k: 'Host', v: 'VPS market_data.db snapshot 2026-06-12; reproducible from committed scripts.' },
      ],
    },
    comparisons: [
      {
        title: 'Finalists — net 20 bps/side (2016–2026)',
        columns: ['Strategy', 'CAGR', 'MaxDD', 'Calmar', 'Sharpe', 'Turnover/yr'],
        rows: [
          ['Equal-weight 3-asset (WINNER)', '19.5%', '−11.3%', '1.73', '1.55', '0.05'],
          ['Equal-weight + trend filter (defensive)', '11.8%', '−8.5%', '1.40', '1.38', '2.16'],
          ['Momentum top-2 (gated) — best tactical', '16.4%', '−12.5%', '1.31', '1.34', '2.38'],
          ['Slide top-1 (trend-gated)', '10.9%', '−24.9%', '0.44', '0.62', '3.31'],
          ['Slide top-1 (raw)', '10.5%', '−34.4%', '0.30', '0.57', '2.83'],
          ['Benchmark: NIFTYBEES B&H', '10.0%', '−28.8%', '0.35', '—', '0'],
        ],
        highlightRows: [0],
        heatmap: false,
      },
      {
        title: 'Per-year: winner vs NIFTY 50',
        columns: ['Year', 'Winner %', 'NIFTYBEES %', 'Excess pp'],
        rows: [
          ['2016', '+12.8', '+4.0', '+8.9'],
          ['2017', '+24.0', '+29.9', '−5.9'],
          ['2018', '+1.5', '+4.8', '−3.3'],
          ['2019', '+26.2', '+13.6', '+12.6'],
          ['2020', '+32.0', '+15.4', '+16.6'],
          ['2021', '+16.7', '+26.0', '−9.3'],
          ['2022', '−3.5', '+5.5', '−9.0'],
          ['2023', '+29.4', '+21.0', '+8.4'],
          ['2024', '+28.6', '+10.4', '+18.3'],
          ['2025', '+28.7', '+11.7', '+17.1'],
          ['2026*', '+12.8', '−9.3', '+22.1'],
        ],
        highlightRows: [0],
        heatmap: true,
      },
      {
        title: 'SILVERBEES add-on (tested per request) — silver HURTS over the full window',
        caption: 'Indian silver ETFs only exist from 2022; pre-2022 silver uses a validated proxy (intl silver × USDINR, monthly-return corr 0.85 to SILVERBEES).',
        columns: ['Book (monthly reb, net 20bps)', 'Window', 'CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['3-asset (Nifty/Gold/Nasdaq), equal', '2015–26 (11.4y)', '17.6%', '−11.3%', '1.57'],
          ['4-asset (+Silver), equal', '2015–26*', '18.2%', '−12.4%', '1.47'],
          ['4-asset (+Silver), inverse-vol', '2015–26*', '18.0%', '−12.4%', '1.45'],
          ['3-asset, equal', '2022–26 (4.3y, metals bull)', '21.6%', '−9.0%', '2.40'],
          ['4-asset (+Silver), inverse-vol', '2022–26', '24.5%', '−9.4%', '2.61'],
        ],
        highlightRows: [0],
        heatmap: false,
      },
      {
        title: 'Through-cycle stress test — extended to 21y (2005–2026) via gold/Nasdaq proxies',
        caption: 'Factor data is post-2015 only, so this uses Nifty as the equity sleeve. Gold = GLD × USDINR, Nasdaq = QQQ × USDINR (validated vs the real ETFs, corr 0.88 / 0.71), chained with real ETFs post-2015. Finally includes 2008/2011/2013 — and shows the recent −11% DD was a benign-period artifact.',
        columns: ['Book / period', 'CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['3-asset equal — 21y (2005–26)', '17.2%', '−23.8%', '0.72'],
          ['3-asset inverse-vol — 21y', '17.1%', '−20.9%', '0.82'],
          ['Nifty only — 21y', '12.7%', '−55.2%', '0.23'],
          ['2008 GFC year: 3-asset vs Nifty', '−21% / −52%', '—', '—'],
          ['2011: 3-asset vs Nifty (gold carried)', '+8% / −24%', '—', '—'],
          ['recent 2015–26 sub-period (for contrast)', '17.6%', '−11.3%', '1.57'],
        ],
        highlightRows: [0, 1],
        heatmap: false,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR', value: '19.5%', tone: 'pos' },
        { label: 'NIFTYBEES CAGR', value: '10.0%' },
        { label: 'Excess / yr', value: '+9.5%', tone: 'pos' },
        { label: 'Sharpe', value: '1.55', tone: 'pos' },
        { label: 'Max Drawdown', value: '−11.3%', tone: 'neg', hint: 'vs NIFTYBEES −28.8%' },
        { label: 'Calmar', value: '1.73', tone: 'pos' },
        { label: 'Turnover / yr', value: '~5%', hint: 'cost-insensitive' },
        { label: 'Yrs beating index', value: '7 / 11' },
      ],
      tables: [
        {
          title: 'Strategy vs benchmark',
          columns: ['Metric', 'EqualWeight 3-ETF', 'NIFTYBEES'],
          rows: [
            ['CAGR', '19.5%', '10.0%'],
            ['Total return', '6.4x', '~2.7x'],
            ['Sharpe', '1.55', '~0.6'],
            ['Max Drawdown', '−11.3%', '−28.8%'],
            ['Calmar', '1.73', '0.35'],
          ],
          highlightRows: [0, 3, 4],
        },
        {
          title: 'Cost sensitivity — Calmar by cost (the winner barely trades)',
          columns: ['Strategy', '0 bps', '10 bps', '20 bps', '40 bps'],
          rows: [
            ['Equal-weight (WINNER)', '1.73', '1.73', '1.73', '1.73'],
            ['Equal-weight + trend filter', '1.73', '1.56', '1.40', '1.12'],
            ['Momentum top-2 (gated)', '1.53', '1.42', '1.31', '1.13'],
            ['Slide top-1 (gated)', '0.53', '0.48', '0.44', '0.35'],
          ],
          highlightRows: [0],
        },
      ],
      charts: [
        {
          src: '/app/gtaa-etf-rotation-factsheet.png',
          caption:
            'CLIENT FACTSHEET — GTAA Equal-Weight 3-Asset (Nifty + Gold + Nasdaq-100) vs NIFTY 50, 2016–2026, net of 20 bps. KPI strip, growth-of-₹1 (log), drawdown-vs-index, annual bars, monthly heatmap, rolling 12m, stat tables. 19.7% CAGR vs 14% (this view), 6.4x, Sharpe 1.04 (rf-adj), Calmar 1.75, MaxDD −11.3%, 55% of years beating the index. Generated by research/_utilities/tearsheet.py.',
        },
        {
          src: '/app/gtaa-longhist-factsheet.png',
          caption:
            'THROUGH-CYCLE (21y, 2005–2026, incl. 2008 GFC) — 3-asset equal-weight vs NIFTY 50. Gold/Nasdaq pre-2015 are proxies (GLD/QQQ × USDINR, validated corr 0.88/0.71); real ETFs post-2015. The honest full-cycle picture: CAGR 17.2% vs 12.7%, MaxDD −23.8% vs −55.2%, Calmar 0.72 (0.82 inverse-vol) vs 0.23. The recent decade\'s −11% DD / 1.7 Calmar was a benign-period artifact; through a real crisis even uncorrelated assets fall together (2008: −21% vs Nifty −52%).',
        },
      ],
    },
    winners: [
      {
        config: 'Equal-weight · 3 ETFs (Nifty/Gold/Nasdaq-100) · monthly rebalance',
        summary: 'Best of a 108-cell sweep, and the SIMPLEST cell — no momentum selection, no parameters to fit. Diversification across three uncorrelated sleeves beats every momentum-rotation variant on Calmar at the lowest drawdown and near-zero turnover.',
        metrics: [
          { k: 'CAGR', v: '19.5%' },
          { k: 'Excess', v: '+9.5%/yr vs NIFTYBEES' },
          { k: 'Sharpe', v: '1.55' },
          { k: 'MaxDD', v: '−11.3%' },
          { k: 'Calmar', v: '1.73' },
        ],
        rejected: [
          'The slide’s top-1 momentum selection: concentrates into the hot asset and eats the reversal — Calmar 0.30 (raw) / 0.44 (gated), −34%/−25% DD.',
          'Adding more ETFs (Next-50, Bank): ext5 universe peaked at Calmar 0.83 — diluting the gold/Nasdaq diversification HURT.',
          'Momentum top-2/top-3 with a cash gate: all underperformed plain equal-weight on net Calmar; selection added nothing here.',
        ],
      },
    ],
    caveats: [
      'Period dependence (RESOLVED 2026-06-14): the headline 1.7 Calmar / −11% DD is a 2016–26 benign-period artifact. Extending to 21y (2005–26) via validated gold/Nasdaq proxies (Nifty as the equity sleeve, since factor/ETF data is post-2015) gives the THROUGH-CYCLE truth: Calmar ~0.72 (0.82 inverse-vol), MaxDD −23.8%, CAGR 17.2%. In a real crisis (2008) even uncorrelated assets fall together → −24%, not −11%. The durable claim stands: vs Nifty-only the book HALVES drawdown (−24% vs −55%) and adds ~4.5%/yr (17.2% vs 12.7%), Calmar 0.8 vs 0.23 — a real all-weather core, just expect ~−24% DD through a crisis, not −11%.',
      'No all-3 simultaneous crash in sample: 2008 isn’t testable (no data), COVID-2020 was V-shaped. A global risk-off hitting equity AND gold AND tech together is under-represented → real MaxDD could exceed −11.3%.',
      'MON100 capacity/regulatory: overseas-ETF flows hit RBI/SEBI caps in 2022 (creation halted, premium to NAV). At size the Nasdaq sleeve carries tracking/capacity risk.',
      'Single 11-year window, no true OOS / walk-forward — mitigated only by the winner being the zero-parameter, simplest config (no knife-edge to overfit; 108 configs searched).',
      'SILVERBEES add-on (user request): Indian silver ETFs only exist from 2022, so pre-2022 silver uses a validated proxy (intl silver × USDINR, monthly-return corr 0.85 to SILVERBEES). Over the full 2015–26 window adding silver LOWERS Calmar (1.57→1.47 equal, 1.50→1.45 inv-vol) — silver is 0.66 correlated to gold (redundant precious-metal) and very volatile (29% vol, −28% DD). The strong 2022–26 result (Calmar 2.6) was a precious-metals bull, not a durable benefit — a recency-bias trap.',
      'Backtest, net of 20 bps modelled cost. LIQUIDBEES price-return ≈0% (daily-dividend ETF) understates the defensive variant’s cash yield by ~6%/yr; the winner uses no cash leg so is unaffected. Nothing wired live. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'RESULTS.md (verdict + tables)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/63_gtaa_etf_rotation/results/RESULTS.md',
      },
      {
        label: 'gtaa_engine.py (engine)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/63_gtaa_etf_rotation/scripts/gtaa_engine.py',
      },
    ],
    projectPaths: [
      'research\\63_gtaa_etf_rotation\\GTAA_ETF_ROTATION_MONTHLY_SWEEP_STATUS.md',
      'research\\63_gtaa_etf_rotation\\scripts\\ (download_etfs, gtaa_engine, run_gtaa_sweep, finalists).py',
      'research\\63_gtaa_etf_rotation\\results\\ (gtaa_sweep.csv, finalists.csv, RESULTS.md, tearsheet.png)',
    ],
  },
  {
    slug: 'v2-nifty-ironfly-sl-vix',
    title: 'V2 NIFTY Positional Iron Fly — Stop-Loss × VIX optimization (2.0% wings)',
    verdict:
      'A positionally-carried short ATM NIFTY iron fly (2.0%-of-ATM wings) is best run with a ≈2.0% underlying move-stop and a VIX≥13 entry floor: +₹8.8L net over 7.3y at Calmar 1.03 and only −₹1.17L drawdown, 7/8 years green (only the 5-month 2026 stub red). A VIX≥14 floor makes every full year green. The defined-risk wings — not the stop — are the real risk control; stop-loss is a sweet-spot at 2.0%, not monotonic.',
    status: 'COMPLETE',
    date: '2026-06-08',
    cardBlurb:
      'Positional 2nd-weekly ATM straddle + 2% wings, 09:20 entry, 10 lots, net of taxes/brokerage/0.25% slippage, 2019–2026 on AlgoTest. Wing-width, VIX-floor and stop-loss all swept; base locked at 2% wings + 2% move-stop + VIX≥13.',
    cardStats: [
      { label: 'Net P&L (7.3y)', value: '+₹8.80L' },
      { label: 'Calmar', value: '1.03' },
      { label: 'MaxDD', value: '−₹1.17L' },
    ],

    systemRules: {
      intro:
        'The actual traded system. Two locked variants share one identical core (below) and differ only in the risk layer (move-stop fixed at 2.0%; VIX floor 13 vs 14).',
      sharedCoreTitle: 'Locked core — identical for both variants',
      sharedCore: [
        { k: 'Instrument', v: 'Short ATM NIFTY straddle + long protective wings = short iron fly; 2nd-nearest weekly expiry; positional / overnight carry.' },
        { k: 'Wings', v: '2.0% of ATM (≈ ±500 pts at today’s NIFTY) — locked from a %-of-ATM wing sweep; 2.5% / 3.0% were strictly worse.' },
        { k: 'Entry', v: '09:20, 4 trading days before expiry (AlgoTest positional max).' },
        { k: 'Roll / re-enter', v: 'Roll 1 trading day before expiry; re-enter the next cycle.' },
        { k: 'Profit target', v: '40% of credit (Phase-2 PT sweep pending).' },
        { k: 'Sizing', v: '10 lots = qty 650 (valid NIFTY multiple).' },
        { k: 'Costs', v: 'Brokerage ₹20/order; STT & charges included; slippage 0.25% of premium (empirically measured: median bid-ask half-spread 0.17% across 3.47M recorded NIFTY option quotes).' },
        { k: 'Window', v: '2019-02 → 2026-05 (~7.3y) on the AlgoTest historical chain.' },
      ],
      riskLayer: {
        title: 'Per-variant risk layer — the only difference',
        caption:
          'Both fix the wings at 2.0% and the underlying move-stop at 2.0%; they differ only in the VIX entry floor. Balanced (≥13) maximises risk-adjusted return; Conservative (≥14) trades a little return for an all-green track record.',
        columns: ['Variant', 'Underlying move-stop', 'VIX entry floor', 'Profile'],
        rows: [
          ['Balanced (recommended)', '2.0%', '≥ 13', 'Calmar 1.03 · +₹8.80L · DD −₹1.17L · only 2026 stub red'],
          ['Conservative', '2.0%', '≥ 14', 'Calmar 0.89 · +₹8.16L · DD −₹1.25L · every full year green'],
        ],
        highlightRows: [0],
      },
    },

    system: {
      intro: 'Backtested on AlgoTest.in’s positional engine; entry/exit expressed as N trading-days-before-expiry. The VIX floor is applied post-hoc from AlgoTest’s exact per-trade entry-VIX column (not a proxy).',
      rows: [
        { k: 'Engine', v: 'AlgoTest.in positional backtester (user-run); Claude structures the grid and analyses the exported trade CSVs.' },
        { k: 'Structure', v: 'Sell ATM CE + ATM PE; buy CE & PE wings at 2.0% of ATM = short iron fly (defined risk).' },
        { k: 'Stop', v: 'Per-leg underlying-movement SL — the short legs exit on a 2.0% NIFTY move from entry.' },
        { k: 'VIX filter', v: 'Keep only trades whose entry India-VIX ≥ floor (13 or 14); exact value from the AlgoTest VIX column.' },
        { k: 'P&L basis', v: 'Net of taxes + ₹20/order + 0.25% slippage; 10 lots; fly SPAN margin ≈ ₹9.58L.' },
      ],
    },

    conditions: {
      intro: 'Robustness controls / the seven deadly sins, as applied to this study.',
      rows: [
        { k: 'Look-ahead', v: 'None — entry/exit are causal; the VIX floor uses entry-time VIX only.' },
        { k: 'Cost neglect', v: 'Net-of-cost throughout; slippage measured empirically (0.17% median), 0.25% used as a prudent blend.' },
        { k: 'Overfitting', v: 'Stop level is a SWEET-SPOT (Calmar 0.76→1.03→0.62 across 1.5/2.0/2.5% at VIX≥13), not a flat plateau → treat as “≈2% wide stop”, not a precise value; wings are the primary risk control.' },
        { k: 'Regime', v: 'Spans 2019–2026 incl. COVID, 2022 bear, 2023 chop, 2024/25 trends.' },
        { k: 'Capacity', v: '10 lots (qty 650) fills on NIFTY; deeper size needs a slippage re-check.' },
        { k: 'Data artifact', v: 'March-2020 COVID circuit-breaker week excluded (AlgoTest left stray single-leg fills at gap strikes).' },
      ],
    },

    comparisons: [
      {
        title: 'Stop-loss sweep on the VIX≥13 base (the lock decision)',
        caption: 'Net of costs, ex-COVID, exact entry-VIX. Calmar peaks sharply at a 2.0% stop. 1.5% rows use a daily-open VIX proxy; all others exact.',
        columns: ['Underlying stop', 'Net P&L', 'Calmar', 'MaxDD', 'Neg years'],
        rows: [
          ['1.0%', '+₹6.51L', '0.58', '−₹1.53L', '2019'],
          ['1.5%*', '+₹8.53L', '0.76', '−₹1.54L', '2026'],
          ['2.0%', '+₹8.80L', '1.03', '−₹1.17L', 'only 2026'],
          ['2.5%', '+₹6.29L', '0.62', '−₹1.39L', '2026'],
          ['No stop', '+₹8.85L', '0.97', '−₹1.25L', '2021, 2026'],
        ],
        highlightRows: [2],
      },
      {
        title: 'Stop-loss sweep with no VIX filter (peak is not a filter artifact)',
        caption: 'Same shape unfiltered — wide stop or none wins; 1.0% over-stops, 2.5% dips. The defined-risk wings cap every trade regardless of stop.',
        columns: ['Underlying stop', 'Net P&L', 'Calmar', 'MaxDD', 'Worst trade', 'Neg years'],
        rows: [
          ['1.0%', '+₹6.73L', '0.44', '−₹2.11L', '−₹40k', '2019, 2023'],
          ['1.5%', '+₹7.64L', '0.70', '−₹1.50L', '−₹74k', '2023, 2026'],
          ['2.0%', '+₹8.50L', '0.68', '−₹1.70L', '−₹71k', '2023, 2026'],
          ['2.5%', '+₹6.60L', '0.49', '−₹1.84L', '−₹67k', '2023, 2026'],
          ['No stop', '+₹8.98L', '0.89', '−₹1.38L', '−₹77k', '2021, 2023, 2026'],
        ],
        highlightRows: [2, 4],
      },
      {
        title: 'VIX entry floor on the 2.0%-stop base',
        caption: 'A ≥13 floor lifts 2023 to green and maximises Calmar; ≥14 makes every full year green at a little less return.',
        columns: ['VIX floor', 'Trades', 'Net P&L', 'Calmar', 'MaxDD', 'Neg years'],
        rows: [
          ['None', '271', '+₹8.50L', '0.68', '−₹1.70L', '2023, 2026'],
          ['≥ 13', '204', '+₹8.80L', '1.03', '−₹1.17L', 'only 2026'],
          ['≥ 14', '169', '+₹8.16L', '0.89', '−₹1.25L', 'none — all green'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Wing-width (locked earlier at 2.0% of ATM)',
        caption: 'Regime-consistent %-of-ATM wing sweep, ex-COVID, no VIX filter. 2.0% best on Calmar; wider strictly worse — closed before the SL sweep.',
        columns: ['Wing (% of ATM)', 'Net P&L', 'Calmar', 'MaxDD', 'Neg years'],
        rows: [
          ['2.0% (= ±500 today)', '+₹7.64L', '0.70', '−₹1.50L', '2023, 2026'],
          ['2.5%', '+₹4.84L', '0.29', '−₹2.28L', '2019, 2020, 2023, 2026'],
          ['3.0%', '+₹5.96L', '0.31', '−₹2.59L', '2020, 2021, 2023, 2026'],
        ],
        highlightRows: [0],
      },
      {
        title: 'CPR compression overlay (CANDIDATE — pending forward validation)',
        caption: 'Diagnostic on the locked book: losses concentrate in volatility compression, flagged by a narrow PRIOR-DAY daily CPR (|TC−BC| from prior H/L/C ÷ entry-open). Skipping entries when CPR width < 0.10% of spot raises return AND cuts drawdown. NOT yet in the locked base.',
        columns: ['Overlay on the VIX≥13 book', 'Trades', 'Net P&L', 'Calmar', 'MaxDD', 'Green years'],
        rows: [
          ['baseline (feature set)', '~203', '+₹8.1L', '0.95', '−₹1.17L', '6/8'],
          ['+ skip CPR width < 0.10%', '147', '+₹11.0L', '1.59', '−₹0.95L', '7/8'],
          ['+ skip CPR<0.10% & Jan/Aug/Sep', '116', '+₹11.85L', '1.71', '−₹0.95L', '8/8'],
        ],
        highlightRows: [1, 2],
      },
      {
        title: 'CPR filter — walk-forward (out-of-sample) validation',
        caption: 'Pick the CPR threshold by Calmar on the TRAIN half, apply it blind to the TEST half. The same ≈0.12% threshold is chosen in both directions and improves return AND drawdown out-of-sample; the skipped (narrow-CPR) trades bleed in BOTH halves → robust, not overfit.',
        columns: ['Split', 'Threshold', 'Test baseline Calmar', 'Test filtered Calmar', 'Test DD base → filtered'],
        rows: [
          ['train 2019–22 → test 2023–26', '0.12%', '1.13', '2.81', '−1.17L → −0.51L'],
          ['train 2023–26 → test 2019–22', '0.12%', '1.11', '2.08', '−1.02L → −0.72L'],
          ['fixed 0.10% (each half)', '0.10%', 'H1 1.11 / H2 1.13', 'H1 1.75 / H2 1.83', 'both improve'],
        ],
      },
      {
        title: 'Causal-feature forensic — what actually separates losing weeks (candidate)',
        caption:
          '~25 causal features known at 09:20 entry, screened on the 204 VIX≥13 trades (univariate quartiles → require monotonic dose-response + per-year consistency + mechanism → walk-forward). A short iron fly is a pure short-gamma bet, indifferent to trend/direction — and the screen confirms it: every feature that separates losers from winners is a volatility-COMPRESSION proxy. RSI (daily/weekly/monthly), moving averages (20/50/200-DMA, slope, weekly WMA), Ichimoku (cloud position & thickness), monthly pivots/CPR, and prior-week range-breaks showed NO usable signal; Bollinger band-width passed univariate but FAILED walk-forward (redundant with CPR). Two independent compression flags survive — narrow prior-day CPR and a weekly inside-candle (only 6 of 18 inside-weeks overlap the CPR skip; inside-weeks still bleed −₹44.6k among CPR-survivors) — and stack to Calmar 2.00. Candidate overlay (n=18 inside-weeks is thin) → forward-paper before it gates live money.',
        columns: ['Entry filter on VIX≥13 base', 'Trades', 'Net P&L', 'Calmar', 'MaxDD', 'Neg yrs'],
        rows: [
          ['Base — no skip', '204', '+₹8.80L', '1.03', '−₹1.17L', '2026'],
          ['skip narrow daily CPR (<0.10%)', '147', '+₹11.00L', '1.59', '−₹0.95L', '2026'],
          ['skip inside-week', '186', '+₹9.83L', '1.15', '−₹1.17L', '2026'],
          ['skip CPR<0.10% OR inside-week', '135', '+₹11.45L', '2.00', '−₹0.78L', '2026'],
        ],
        highlightRows: [3],
      },
    ],

    results: {
      metrics: [
        { label: 'Net P&L (7.3y)', value: '+₹8,80,110', tone: 'pos' },
        { label: 'Calmar', value: '1.03' },
        { label: 'Max Drawdown', value: '−₹1,16,834', tone: 'neg' },
        { label: 'CAGR (on ₹8.25L SPAN)', value: '~10.5%', hint: '14.6%/yr simple-on-margin; ~9.7% on 1.5× buffered capital' },
        { label: 'Trades', value: '204' },
        { label: 'Green years', value: '7/8' },
        { label: 'Worst trade', value: '−₹71,235', tone: 'neg' },
      ],
      tables: [
        {
          title: 'Year-wise returns — monthly P&L (₹), Balanced VIX≥13',
          caption: 'Bucketed by entry month, net of costs, ex-COVID. Months at 0 = no trade cleared the VIX≥13 floor that month (e.g. the low-VIX 2023/25 stretches). Only the 5-month 2026 stub is red.',
          heatmap: true,
          columns: ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Total'],
          rows: [
            ['2019', '0', '+12,734', '+36,014', '+1,863', '+18,534', '+10,317', '-5,013', '-38,369', '-16,673', '-29,644', '+21,168', '0', '+10,930'],
            ['2020', '-16,660', '+55,156', '0', '+24,108', '+130,835', '-6,156', '-404', '-50,213', '+18,681', '-6,370', '+13,596', '+3,860', '+166,432'],
            ['2021', '+20,968', '+15,513', '-1,017', '+7,747', '-33,801', '+63,015', '-28,700', '-16,978', '+4,147', '-8,925', '+19,754', '+16,250', '+57,973'],
            ['2022', '+28,455', '+61,307', '+87,192', '+7,062', '-16,337', '+30,892', '-23,546', '-3,552', '-7,465', '-19,405', '+6,357', '+50,351', '+201,312'],
            ['2023', '-20,544', '+30,437', '-3,295', '0', '+20,859', '0', '0', '0', '0', '0', '0', '+34,875', '+62,331'],
            ['2024', '+15,232', '+63,656', '+10,015', '+7,766', '+175,295', '-35,493', '-2,744', '-49,371', '+22,825', '+103,345', '-94,905', '+78,299', '+293,918'],
            ['2025', '-9,910', '+54,398', '+34,114', '0', '+52,969', '-26,660', '0', '0', '0', '0', '0', '0', '+104,912'],
            ['2026', '-30,060', '0', '+20,458', '-11,567', '+3,471', '0', '0', '0', '0', '0', '0', '0', '-17,698'],
          ],
        },
        {
          title: 'AlgoTest source output — raw 2.0%-stop run (all trades, no VIX filter, incl COVID)',
          caption: 'The literal AlgoTest platform result the optimization is built on: 273 trades, total +₹5.84L, win 56%, MaxDD −₹3.11L (the −₹2.66L Mar-2020 COVID week dominates 2020). Applying the VIX≥13 floor and excluding the COVID artifact turns this raw run into the locked +₹8.80L / Calmar 1.03 book above.',
          heatmap: true,
          columns: ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Total'],
          rows: [
            ['2019', '0', '+12,734', '+36,014', '+1,863', '+18,534', '+10,317', '-2,966', '-38,369', '-16,673', '-29,644', '+21,168', '+12,195', '+25,171'],
            ['2020', '-31,146', '+55,156', '-266,381', '+24,108', '+130,835', '-6,156', '-404', '-50,213', '+18,681', '-6,370', '+13,596', '+3,860', '-114,435'],
            ['2021', '+20,968', '+15,513', '-1,017', '+7,747', '-33,801', '+63,015', '-51,008', '-16,978', '+4,147', '-8,925', '+19,754', '+16,250', '+35,665'],
            ['2022', '+28,455', '+61,307', '+87,192', '+7,062', '-16,337', '+30,892', '-23,546', '-3,552', '-7,465', '-19,405', '+6,357', '+64,458', '+215,419'],
            ['2023', '-20,544', '+12,767', '-3,295', '-8,895', '-8,256', '-47,517', '+34,378', '+24,593', '-57,043', '+16,265', '+980', '+48,756', '-7,811'],
            ['2024', '+13,599', '+63,656', '+10,015', '+7,766', '+175,295', '-35,493', '-38,264', '-49,371', '-24,684', '+103,345', '-94,905', '+78,299', '+209,256'],
            ['2025', '-9,910', '+54,398', '-8,608', '0', '+52,969', '+6,401', '+128,527', '+47,362', '-2,553', '+30,626', '+14,770', '+6,147', '+320,131'],
            ['2026', '-95,104', '-16,783', '+20,458', '-11,567', '+3,471', '0', '0', '0', '0', '0', '0', '0', '-99,525'],
          ],
        },
        {
          title: 'CPR overlay (CANDIDATE) — year-wise monthly P&L, VIX≥13 + skip CPR<0.10%',
          caption: 'Months at 0 = no qualifying entry that month. Only the 5-month 2026 stub is red. Candidate overlay — not in the locked base; shown for the curves the user requested.',
          heatmap: true,
          columns: ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Total'],
          rows: [
            ['2019', '0', '+2,891', '+44,623', '0', '+49,033', '+10,317', '+11,575', '-46,098', '-16,673', '0', '+21,168', '0', '+76,835'],
            ['2020', '+3,488', '+63,302', '0', '+24,108', '+130,835', '-6,022', '+2,368', '-71,235', '+18,681', '-11,387', '+13,596', '+3,607', '+171,342'],
            ['2021', '-14,085', '+15,513', '+3,974', '+7,747', '-40,914', '+54,342', '+4,391', '-2,211', '+4,147', '+8,799', '+5,836', '+41,306', '+88,844'],
            ['2022', '+39,514', '+61,307', '+85,980', '+7,062', '-16,337', '+30,892', '-23,546', '-3,552', '+2,951', '-7,640', '-22,942', '+35,677', '+189,366'],
            ['2023', '-6,020', '+30,437', '-3,295', '0', '+20,859', '0', '0', '0', '0', '0', '0', '+34,875', '+76,856'],
            ['2024', '+15,232', '+63,656', '+27,547', '+7,766', '+190,319', '-19,730', '-2,744', '+29,226', '+22,825', '+97,203', '-94,905', '+66,842', '+403,235'],
            ['2025', '-31,181', '+31,946', '+34,114', '0', '+71,951', '+4,637', '0', '0', '0', '0', '0', '0', '+111,466'],
            ['2026', '-30,060', '0', '+20,458', '-11,567', '+3,471', '0', '0', '0', '0', '0', '0', '0', '-17,698'],
          ],
        },
      ],
      charts: [
        {
          src: '/app/v2_ironfly_factsheet.png',
          caption: 'Cumulative net P&L (VIX≥13 vs VIX≥14), drawdown, and year-wise bars for the locked 2.0%-wing / 2.0%-stop iron fly, 2019–2026, net of costs.',
        },
        {
          src: '/app/v2_ironfly_cpr_overlay.png',
          caption: 'CANDIDATE — CPR compression overlay: cumulative P&L (skip CPR<0.10%, and + skip Jan/Aug/Sep), drawdown, and year-wise bars. Walk-forward-validated in-sample; pending forward confirmation; not in the locked base.',
        },
      ],
    },

    winners: [
      {
        config: 'SL 2.0% + VIX≥13 — Balanced (recommended)',
        summary: 'Best risk-adjusted point of the entire stop×VIX grid: highest Calmar and smallest drawdown, only the 5-month 2026 stub red.',
        metrics: [
          { k: 'Net P&L (7.3y)', v: '+₹8,80,110' },
          { k: 'Calmar', v: '1.03' },
          { k: 'MaxDD', v: '−₹1,16,834' },
          { k: 'CAGR (on ₹8.25L)', v: '~10.5%' },
          { k: 'Green years', v: '7/8' },
        ],
        rejected: [
          'SL 1.0% — over-stops, choppy −₹2.1L drawdown',
          'SL 2.5% — Calmar dips to 0.62 (the level is a sweet-spot, not a plateau)',
          'Wings 2.5% / 3.0% — strictly worse than 2.0%',
          'SL 1.5% — the old V2 spec; beaten on every axis',
        ],
      },
      {
        config: 'SL 2.0% + VIX≥14 — Conservative',
        summary: 'The only configuration with every full year green; a little less total for an all-green track record.',
        metrics: [
          { k: 'Net P&L (7.3y)', v: '+₹8,15,653' },
          { k: 'Calmar', v: '0.89' },
          { k: 'MaxDD', v: '−₹1,24,847' },
          { k: 'CAGR (on ₹8.25L)', v: '~9.9%' },
          { k: 'Green years', v: '8/8' },
        ],
      },
    ],

    caveats: [
      'Single instrument (NIFTY), single backtester (AlgoTest), in-sample over one 7.3-year history — a robust base/SIGNAL, not yet live-validated.',
      'The 2.0% stop is a sweet-spot, not a plateau (Calmar 0.76→1.03→0.62 across 1.5/2.0/2.5% at VIX≥13). Treat the live rule as “≈2% underlying move-stop”; the defined-risk wings are the real risk control.',
      'March-2020 COVID circuit-breaker week excluded — AlgoTest left stray single-leg fills at gap strikes (a data artifact, not a tradable result).',
      '2026 is a 5-month stub (Jan–May), not a full year; it is the only red year at VIX≥13.',
      'Net of 0.25% slippage (measured median 0.17%); live fills at 10 lots may differ. Nothing is wired to live orders.',
      '1.5%-stop VIX-filtered figures use a daily-open VIX proxy; all 2.0% figures use AlgoTest’s exact entry-VIX column.',
      'Return metrics use the VERIFIED Zerodha SPAN+exposure margin (Kite margin API, 2026-06-08): ₹82,458/lot for the ±500 iron fly → ₹8,24,580 for 10 lots (a naked straddle is ₹2,10,088/lot). CAGR ~10.5% compounds the equity; simple return-on-margin is 14.6%/yr; on 1.5× buffered working capital ~9.7%/yr. Current-level snapshot — 2019 margin was ~half (lower notional); returns are simple, not compounding (fixed 10 lots). The absolute return is modest — the edge is Calmar/consistency, not raw return.',
      'The CPR compression overlay is a CANDIDATE: walk-forward-validated in-sample but NOT yet forward-confirmed and NOT folded into the locked base. Threshold ≈0.10–0.12% means “skip the bottom-quartile CPR width”, not a precise constant.',
    ],

    githubLinks: [
      { label: 'research/60 — V2 straddle optimization', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/60_v2_straddle_optimization' },
    ],
    projectPaths: [
      'research/60_v2_straddle_optimization/V2_BIWEEKLY_STRADDLE_ALGOTEST_OPTIMIZATION_SWEEP_STATUS.md',
      'research/60_v2_straddle_optimization/scripts/vix_overlay_2pct.py',
      'frontend/src/pages/Straddles.tsx (live paper book)',
    ],
  },
  {
    slug: 'nifty-fly-calm-directional-entry',
    title: 'NIFTY Premium-Selling — Entry Regimes: Calm Gate + Directional Skew (extends the V2 Iron Fly)',
    verdict:
      'Extension of the V2 Iron-Fly study — it answers WHEN to enter. On 11 years of NIFTY price action (2015–2026, daily): (1) calm is strongly predictable from ONE family — volatility / range COMPRESSION (low ATR/VIX, narrow daily CPR, firm Stochastic) — lifting a neutral fly’s ≈59% weekly survival to ~75%, while trend / MA / Ichimoku / ADX / inside-candle features add essentially nothing. (2) Direction is NOT predictable at entry (a coin-flip with a structural up-drift), but a day-1 confirmation IS tradeable — an up day-1 → 88% the week stays up. A defined-risk bullish JADE LIZARD (day-1-up-confirmed) monetises the drift far better than the symmetric fly; the bearish mirror has a safer tail but is weaker, best used tactically / as a hedge. Result: three entry-conditioned systems — Neutral fly (compression gate), Bull jade (day-1-up), Bear reverse-jade (day-1-down / hedge). Price-only (calm-rate + VIX-scaled proxy P&L) — exact ₹ owed to AlgoTest.',
    status: 'COMPLETE',
    date: '2026-06-13',
    cardBlurb:
      'Extends the V2 iron-fly study → WHEN to enter. Calm-day prediction (P1–P4) + directional/skewed structures (P5), NIFTY daily 2015–2026. Compression gate lifts fly survival 59%→75%; direction unpredictable at entry but day-1-confirmed; bull jade-lizard is the drift-aligned winner.',
    cardStats: [
      { label: 'Calm gate (5-day)', value: '59% → 75%' },
      { label: 'Bull jade (day-1)', value: 'EV +₹64k · 81% win' },
      { label: 'Mild-directional', value: '31% of weeks' },
    ],

    systemRules: {
      intro:
        'Three entry-conditioned systems came out of this study — one neutral, two directional. They share the same NIFTY premium-selling DNA and differ in structure, entry trigger and which regime they harvest. Win-rates: the fly’s “win” = the week stays calm (2% stop not hit); the jade/bear “win” = a positive trade (proxy P&L).',
      sharedCoreTitle: 'Shared basis (all three systems)',
      sharedCore: [
        { k: 'Underlying / sizing', v: 'NIFTY weekly options, 10 lots (qty 650), positional / overnight carry; ≈₹7.0L SPAN margin (Kite, current).' },
        { k: 'Research universe', v: 'NIFTY + India VIX daily, 2015-01 → 2026-06 (~2,800 entry days); causal features only (computed on data ≤ prior close — no look-ahead).' },
        { k: 'Outcome proxy', v: 'No in-house historical option premiums → the 2% move-stop (not) firing within the hold is the model-free CALM / win proxy; structure P&L is modelled with a VIX-scaled premium. Exact ₹ ⇒ AlgoTest.' },
        { k: 'VIX regime (all three)', v: 'Trade only VIX 13–22 — floor 13 (premium richness, inherited from the V2 study), hard-skip > 22 (calm collapses to 16%, EV turns negative).' },
        { k: 'Costs', v: '₹20/order, taxes on, 0.25% slippage (empirical median 0.17%).' },
      ],
      riskLayer: {
        title: 'The three systems — structure · trigger · exit · win-rate · edge',
        caption:
          'Neutral harvests CALM (compression gate); the two directional books harvest the day-1 follow-through (up strong, down weak). Strikes are % of spot. Jade/bear EV & worst are per-10-lot proxy (VIX 13–22).',
        columns: ['System', 'Structure (strikes, % of spot)', 'Entry trigger', 'Exit', 'Win-rate', 'Edge (proxy)'],
        rows: [
          ['Neutral iron fly', 'SELL ATM CE + ATM PE; BUY +2% CE & −2% PE wings', 'Compression gate: ATR%<1.1 ∧ CPR_d<0.16 ∧ Stoch>65 (≥2 of 3) + VIX 13–22', '2% underlying move-stop (gap day → 09:15–09:20 OR-break) · +40% credit PT · roll DTE≤1', '~69–75% (5-day calm-survival)', 'survival 59%→75%; +EV with management (V2 study)'],
          ['Bull jade lizard (primary directional)', 'SELL −2% PE; SELL +1% CE + BUY +2.5% CE (call spread); BUY −4% PE (tail cap)', 'Day-1 UP confirm (> +0.5%) + VIX 13–22', 'roll DTE≤1; defined risk (~−₹200k)', '81%', 'EV +₹64k · worst −₹201k'],
          ['Bear reverse-jade (tactical / hedge)', 'SELL +2% CE; SELL −1% PE + BUY −2.5% PE (put spread); BUY +4% CE (tail cap)', 'Day-1 DOWN confirm (< −0.5%) + VIX 13–22; or as a hedge sleeve', 'roll DTE≤1; defined risk (~−₹200k)', '73%', 'EV +₹47k · worst −₹203k · safer (upside) tail'],
        ],
        highlightRows: [0, 1],
      },
    },

    system: {
      intro:
        'A price-action study (NIFTY daily + India VIX, Kite), run in five phases. Because we hold no multi-year option premiums, the dominant loss driver — the 2% underlying move-stop — is used as a model-free calm/win proxy; structure P&L is then modelled with a VIX-scaled premium. The companion AlgoTest cards (in the repo) confirm the exact ₹ on real premiums.',
      rows: [
        { k: 'P1 — univariate screen', v: '~24 causal features vs the 5-day calm outcome → only the volatility/range-COMPRESSION family separates calm from non-calm.' },
        { k: 'P2 — combinations / composite', v: 'Redundancy (the vol cluster is one factor), conditional lift, AND-gates and a compression score, all walk-forward (thresholds chosen on the train half, applied blind to test).' },
        { k: 'P3 — premium-aware EV', v: 'VIX as a premium proxy → the net-₹ sweet spot vs the pure calm optimum; isolates the VIX 13–22 tradeable band and the >22 disaster zone.' },
        { k: 'P5a/b — direction', v: 'Signed forward-move buckets + the day-1 follow-through (does the first day’s move predict the week).' },
        { k: 'P5c/d/e — structures', v: 'Iron fly vs jade lizard vs broken-wing; long-put tail tuning; the bearish mirror — EV / win / worst-case on the actual weekly move distribution.' },
      ],
    },

    conditions: {
      intro: 'Robustness controls / the seven deadly sins, as applied here.',
      rows: [
        { k: 'Look-ahead', v: 'None — every feature uses data ≤ prior close; the outcome is strictly forward.' },
        { k: 'Walk-forward', v: 'Compression thresholds picked on the train half and applied BLIND to the test half (out-of-sample AUC ≈0.65; gate calm holds ~80% on TEST).' },
        { k: 'Multiple testing', v: '~24 features screened — the survivors (VIX/ATR/realised-vol/Donchian/Bollinger/CPR) are all one volatility-compression factor; ADX, Ichimoku, MA distance/slope, RSI, inside-candle showed ≤7pp non-monotonic spreads → eliminated.' },
        { k: 'Regime', v: 'Spans 2015 deval, 2018, COVID-2020, 2022 bear, 2023 chop, 2024/25 trends — the gate is protective in volatile years (2020 +36pp, 2022 +34pp), ~neutral in calm years.' },
        { k: 'Calm ≠ P&L', v: 'Calm-rate is the WIN-RATE axis only; low VIX = calmer but thinner premium → the net-₹ optimum needs real premiums (AlgoTest). The VIX floor (≥13) deliberately trades ~4pp of calm for premium richness.' },
        { k: 'Negative skew (directional)', v: 'A naked jade is many-small-wins / one-big-loss (short-put crash tail −₹795k). The long −4% put defines the risk (~−₹200k) at some EV cost — that is the live structure.' },
        { k: 'Proxy premiums', v: 'Structure ₹ uses a VIX-scaled credit and held-to-expiry payoffs (no intraday stop) → trust the RELATIVE ranking, not the absolute ₹. AlgoTest cards settle the exact numbers (incl. whether the jade truly has no upside risk at a given VIX).' },
      ],
    },

    comparisons: [
      {
        title: 'P1 — what predicts CALM (univariate, 5-day, 3-era consistent)',
        caption: 'Top-quintile vs bottom-quintile calm-rate (base 59%). One family wins: volatility / range compression. Trend / oscillator / MA / Ichimoku / inside-candle features are noise.',
        columns: ['Feature', 'Best quintile', 'Worst quintile', 'Spread', 'Calm when'],
        rows: [
          ['India VIX', '81%', '32%', '0.486', 'LOW'],
          ['ATR(14) / price', '79%', '33%', '0.461', 'LOW'],
          ['realised vol (10/20d)', '77%', '40%', '0.375', 'LOW'],
          ['Donchian-20 / 5-day range', '72%', '40%', '0.317', 'NARROW'],
          ['Bollinger width (squeeze)', '69%', '45%', '0.240', 'NARROW'],
          ['prior-day CPR width (daily)', '66%', '43%', '0.237', 'NARROW'],
          ['ADX / Ichimoku / MA dist / RSI / inside-week', '≈ base', '≈ base', '≤0.02 / noise', 'NO signal — eliminated'],
        ],
        highlightRows: [0, 1, 5],
      },
      {
        title: 'P2 — CONVICTION table: calm-rate by hold length',
        caption: 'Compression = ATR%<1.1 ∧ CPR_d<0.16 ∧ Stoch>65 (2 of 3); VIX band 13–22. The right column is the live gate. Note the VIX≥13 floor LOWERS calm ~4pp (it removes the calmest low-VIX days) — a premium choice, not a calm choice.',
        columns: ['Hold', 'BASE (no filter)', 'Compression only (≈48% cov)', 'Compression + VIX 13–22 (≈28% cov)'],
        rows: [
          ['3 trading days', '77.5%', '88.3%', '86.2%'],
          ['4 trading days', '68.4%', '80.8%', '77.8%'],
          ['5 trading days', '59.6%', '72.6%', '68.8%'],
          ['8 trading days', '39.5%', '51.6%', '47.8%'],
        ],
        highlightRows: [0, 2],
      },
      {
        title: 'P3 — EV by VIX bucket (the calm-vs-premium tradeoff)',
        caption: 'Per-10-lot proxy, stop calibrated to the verified ₹34k. Calm rises as VIX falls, premium rises as VIX rises. VIX 13–14 is a local dip; VIX > 22 is the only EV-negative regime.',
        columns: ['VIX bucket', 'Calm', 'EV / trade'],
        rows: [
          ['≤ 13', '78–80%', '+₹36–38k'],
          ['13–14', '63%', '+₹25.8k (dip)'],
          ['15–16', '71%', '+₹33.9k'],
          ['18–20', '53%', '+₹29.3k'],
          ['20–25', '37–48%', '+₹15.2k'],
          ['25+', '16%', '−₹1.5k (avoid)'],
        ],
        highlightRows: [5],
      },
      {
        title: 'P5a — direction is UNPREDICTABLE at entry',
        caption: 'Among weeks that moved ≥1.5%, P(up) = 59% (just the drift). No entry-time feature picks the SIGN — every spread is ≤7pp and non-monotonic.',
        columns: ['Feature', 'P(up) low→high quintile', 'Spread'],
        rows: [
          ['ADX (best)', '58 → 65%', '+7pp (still weak)'],
          ['momentum 20d / MA alignment', '59 → 62%', '+3pp'],
          ['prior-week breakout', 'follow-through', '+2pp'],
          ['RSI / MA slope / CPR / Stoch / mom5', '—', '≈0 to −5pp'],
        ],
      },
      {
        title: 'P5b — but day-1 CONFIRMATION is tradeable',
        caption: 'After entry, condition on the first day’s realised move → does the 5-day window finish the same side? General momentum (not squeeze-specific). Up is strong, down is weak.',
        columns: ['Day-1 move (after entry)', 'P(week ends same side)', 'P(ends ≥1.5% same side)'],
        rows: [
          ['up > 0.5%', '75%', '36%'],
          ['up > 1.0%', '88%', '56%'],
          ['down > 0.5%', '68%', '32%'],
          ['down > 1.0%', '73%', '48%'],
        ],
        highlightRows: [1],
      },
      {
        title: 'P5c/d/e — structures: EV / win / worst (proxy, VIX 13–22, per 10-lot)',
        caption: 'The bullish jade fits NIFTY’s up-drift; the long −4% put caps its crash tail; the bearish mirror has a safer tail but is a weaker bet. Day-1 confirmation lifts EV and (with the put) keeps the tail capped.',
        columns: ['Structure', 'EV', 'Win%', 'Worst week'],
        rows: [
          ['Iron fly (symmetric)', '−₹40k', '37%', '−₹182k'],
          ['Jade NAKED (short −2% put)', '+₹87k', '78%', '−₹795k ⚠'],
          ['Jade + long −4% put (defined)', '+₹41k', '71%', '−₹206k'],
          ['Jade + 4% put · day-1 UP-confirmed', '+₹64k', '81%', '−₹201k'],
          ['Bear reverse-jade + 4% call · day-1 DOWN', '+₹47k', '73%', '−₹203k'],
        ],
        highlightRows: [3],
      },
      {
        title: 'Sample payoff — P&L by 5-day move bucket (symmetric fly vs bull jade+4%put)',
        caption: 'Why the jade wins on the drift: it converts the fly’s mild-bull and mild-bear losses into wins, only losing on a real drop. Per-10-lot proxy averages.',
        columns: ['5-day move (share of weeks)', 'Symmetric fly', 'Bull jade + 4% put'],
        rows: [
          ['strong bear < −3% (6%)', '−₹143k', '−₹143k'],
          ['mild bear −1.5/−3% (14%)', '−₹137k', '+₹71k'],
          ['calm ±1.5% (54%)', '+₹41k', '+₹106k'],
          ['mild bull +1.5/3% (20%)', '−₹130k', 'capped ≈0'],
          ['strong bull > +3% (5%)', '−₹140k', '−₹111k'],
        ],
        highlightRows: [1, 2],
      },
      {
        title: 'P6 — intra-hold survival: P(finish calm) by buffer used (the dynamic “apply the brakes” line)',
        caption: 'Once the fly is still calm at day-3 / day-4 close, how far it has DRIFTED from entry sets the odds of finishing calm. Conditional survival: staying calm through day-3 lifts the next-2-day odds to 77% (vs 59% unconditional; per-extra-day hazard ≈13%). Past ~1.4% drift the odds collapse — the day-3/4 roll-or-close line (built into the CALMER indicator’s caution band + live odds gauge).',
        columns: ['State at day-3 / day-4 close', 'Drift from entry', 'P(finish calm to day-5)'],
        rows: [
          ['day-3 · barely moved', '< 0.3%', '90%'],
          ['day-3', '0.3–0.6%', '88%'],
          ['day-3', '0.6–0.95%', '85%'],
          ['day-3', '0.95–1.37%', '73%'],
          ['day-3 · hugging the band', '> ~1.4%', '48%'],
          ['day-4 · barely moved', '< 0.32%', '99%'],
          ['day-4 · hugging the band', '> ~1.4%', '64%'],
        ],
        highlightRows: [4, 6],
      },
      {
        title: 'P8 — predicting a day-4/5 breach from day-3 patterns (beyond drift)',
        caption: 'Among flies still calm at day-3 close (23% breach on day-4/5), what flags the impending breach. Drift dominates, but RANGE / CHOP adds independently — and within the low-drift "looks-safe" group, a wide intra-hold range DOUBLES the hidden breach risk (5%→21%). The CALMER indicator surfaces this at day-3 close (drift + chop → HOLD / watch / ROLL). ATR-ratio and VIX-change give nothing — and tight intra-hold range → LOW breach, so the "coil → explosion" idea stays unsupported.',
        columns: ['Day-3 feature', 'P(breach by day-5) low→high quintile', 'Spread'],
        rows: [
          ['drift from entry (buffer used)', '9 → 52%', '+42pp (dominant)'],
          ['intra-hold range, days 1–3 (chop)', '8 → 41%', '+33pp'],
          ['acceleration (|move| growing)', '14 → 39%', '+25pp'],
          ['day-3 candle range', '15 → 31%', '+15pp'],
          ['WITHIN low-drift (<0.6%): wide chop', '5 → 21%', '+16pp (hidden danger)'],
          ['ATR ratio / VIX change', '— / —', '~0 (no signal)'],
        ],
        highlightRows: [1, 4],
      },
      {
        title: 'P9 — daily-close breach (study) vs 1-min intraday stop (live): whipsaw cost',
        caption: 'Our calm study measures breach on the DAILY close; the live engine exits on a 1-min candle close ≥2%. The intraday stop caps trend/gap losses but whipsaws on spikes that revert by EOD. Calibrated on real 5-min closes (Kite, 2023–26): the intraday stop genuinely whipsaws ~10% of entries. Net ₹ is still positive (AlgoTest 1-min backtest, Calmar 1.03) — capping the deep-loss tail outweighs whipsaws — but a less-twitchy stop (5/15-min close or a small buffer) is a P7 lever.',
        columns: ['Measure', 'Value'],
        rows: [
          ['holds that touch ±2% intraday', '53% (47% never touch)'],
          ['of touches: WHIPSAW (revert by close)', '24%'],
          ['true 5-min-CLOSE whipsaw rate (calibrated)', '10.5% of all entries'],
          ['continued-day over-run beyond 2% (median)', '~0.4pp (capping benefit modest)'],
          ['net ₹ verdict', 'positive (AlgoTest 1-min, real premiums) — proxy confirms whipsaw frequency only'],
        ],
      },
      {
        title: 'P10 / P10b — stop granularity & the 2.2% buffer (the whipsaw lever)',
        caption: 'The live ~1-min/point-in-time stop whipsaws ~11% of entries (real 5-min, full period 2015–26). Coarser bars (15/30-min) barely help; the daily-close stop kills whipsaws but rides a 5× fatter tail (19% of exits >3%). The effective lever is a small BUFFER — exit on a close beyond ~2.2% (not 2.0%): it halves whipsaws (11→5.6%) for only 1.6% missed breaches. Confirmation (2–3 consecutive bars) barely helps. CANDIDATE refined stop = 2.2% buffer; ₹ pending an AlgoTest check before any live change.',
        columns: ['Stop variant', 'Whipsaw (false exit)', 'Missed real breach', 'Median exit'],
        rows: [
          ['2.0% (current live)', '11.1%', '0%', '2.06%'],
          ['2.2% buffer (candidate)', '5.6%', '1.6%', '2.26%'],
          ['2.5% buffer', '1.8%', '7.8% (too many)', '2.56%'],
          ['15-min close', '10.0%', '0.1%', '2.09%'],
          ['daily close', '0%', '—', '2.40% · 19% >3% (fat tail)'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Day-3 ADJUSTMENT cases (A / B / C) — what to do when still alive at day-3',
        caption: 'After day-3 the position self-classifies. The near-band CONVERT state (Case B) occurs ~4×/year — ~17% of neutral entries drift there (range 6–28%/yr). Near-band risk is ONE-SIDED — the opposite / untested band breaches only ~0–1% (UP-drift 0%, DOWN-drift 1%), and it is ~50/50 to continue (53/47%) vs revert — so defend the hugged side only and keep the safe side. Chop is two-sided but mild. The ₹ of each defense (roll-out → asymmetric condor, re-centre → skewed fly, convert → jade) needs option premiums (P7 AlgoTest study).',
        columns: ['Case', 'Day-3 state', 'Finish-calm', 'Risk', 'Action'],
        rows: [
          ['A', 'Not flagged (drift <0.6%, range ≤1.5%)', '92–94%', 'minimal', 'HOLD — do nothing'],
          ['B', 'Near-band (drift 1.4–2% toward one band)', '46–52%', 'ONE-SIDED (opp ~0–1%)', 'Defend the hugged side: roll its credit spread OUT → asymmetric iron condor, or re-centre (skewed fly), or convert toward the drift (jade); keep the safe side'],
          ['C', 'Chop (drift <0.6% but range >1.5%)', '82–84%', 'two-sided, mild', 'Mostly HOLD; tighten symmetrically if nervous'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Days/year, selectivity & win-ratio by system (one-trade-at-a-time, 5-day hold)',
        caption: 'How often each system trades and its price-action win-ratio. In-trade days/yr ≈ Neutral 107 (~43% of the year) · Bull jade 109 (~44%) · Bear jade 95 (~38%) — all three sit out ~half the calendar. Win-ratio: Neutral = the 5-day calm-rate (wins when NIFTY stays inside ±2%); Bull/Bear = positive-trade rate on the proxy structure (wider win-zone than the fly). ₹ P&L / drawdown / biggest-win / biggest-loss need REAL option premiums — see the V2 Iron Fly study (linked above) for the fly-base real per-year ₹ (2019–26 +₹8.8L), and run the AlgoTest cards for the compression-gated fly + jades.',
        columns: ['Year', 'Neutral ent', 'Neutral win%', 'Bull ent', 'Bull win%', 'Bear ent', 'Bear win%'],
        rows: [
          ['2015', '20', '45%', '30', '63%', '26', '69%'],
          ['2016', '32', '66%', '30', '77%', '29', '86%'],
          ['2017', '21', '76%', '11', '73%', '10', '70%'],
          ['2018', '22', '86%', '27', '89%', '22', '73%'],
          ['2019', '32', '66%', '26', '85%', '26', '85%'],
          ['2020', '20', '60%', '18', '72%', '10', '60%'],
          ['2021', '26', '62%', '27', '78%', '22', '86%'],
          ['2022', '20', '70%', '29', '69%', '25', '72%'],
          ['2023', '10', '80%', '11', '82%', '9', '78%'],
          ['2024', '31', '65%', '28', '82%', '23', '83%'],
          ['2025', '17', '53%', '13', '69%', '15', '87%'],
          ['2026*', '6', '67%', '12', '75%', '11', '73%'],
          ['AVG', '21', '66%', '22', '76%', '19', '78%'],
        ],
        highlightRows: [12],
      },
      {
        title: 'Combined coverage & IDLE-CASH map — all 3 systems (in-trade days/year)',
        caption: 'Stacking all three systems on ONE capital pool (take any signal when flat) lifts deployment from 45% (neutral only) to ~69% of the year — the bull/bear books fill ~half the fly’s idle time. Running 3 separate books (3× capital) only reaches ~72% (the systems overlap heavily) → one pool is efficient. The idle is REGIME-CONCENTRATED: low-VIX years (2017/2023/2025) and COVID-2020 sit idle 50–65% (130–160 days) because all three only trade VIX 13–22; normal years idle ~10–15%. (B) ~4 neutral entries/yr (17%) drift to the day-3 near-band "convert" state. → Idle-cash rule: when VIX is OUTSIDE 13–22 the cash is idle — park in debt, or run a regime-matched alternate (low-VIX → momentum/trend; high-VIX → defined-risk/long-vol). Both alternates are now BUILT & robustness-checked (research/65/66): the low-VIX long is ROBUST (positive in BOTH 2015–20 & 2021–26 halves, Sharpe 0.46→0.76, exit intraday the moment 5-min VIX≥13); the high-VIX mean-reversion long is a SIGNAL-but-thin (positive both halves but 2020-led, −9/−11% DD → use defined-risk). A productive sleeve for EVERY VIX regime → near-zero structural idle.',
        columns: ['Year', 'Neutral-only', 'All-3 (1 pool)', 'Idle days', 'Pool deployed'],
        rows: [
          ['2015', '100', '220', '28', '89%'],
          ['2016', '160', '222', '25', '90%'],
          ['2017', '105', '108', '140', '44%'],
          ['2018', '107', '209', '37', '85%'],
          ['2019', '163', '211', '34', '86%'],
          ['2020', '98', '120', '132', '48%'],
          ['2021', '130', '188', '60', '76%'],
          ['2022', '99', '192', '56', '77%'],
          ['2023', '52', '85', '161', '35%'],
          ['2024', '156', '217', '32', '87%'],
          ['2025', '85', '118', '131', '47%'],
          ['2026*', '30', '65', '44', '60%'],
          ['AVG', '107 (45%)', '163 (69%)', '73 (31%)', '69%'],
        ],
        highlightRows: [12],
      },
      {
        title: 'ENTRY CHECKLIST — verify before each system',
        caption: 'A pre-trade checklist. Compute on the last completed daily bar (causal). All systems require VIX 13–22 first.',
        columns: ['Check', 'Neutral fly', 'Bull jade', 'Bear reverse-jade'],
        rows: [
          ['India VIX in 13–22?', 'required', 'required', 'required'],
          ['ATR(14)/spot < 1.1%?', 'yes (compression)', '—', '—'],
          ['Prior-day daily CPR width < 0.16%?', 'yes (compression)', '—', '—'],
          ['Stochastic %K(14) > 65?', 'yes (compression)', '—', '—'],
          ['Compression score ≥ 2 of 3 above?', 'REQUIRED', '—', '—'],
          ['Day-1 confirmation', 'n/a (enter 09:20)', 'up > +0.5% (enter next morning)', 'down > −0.5% (enter next morning)'],
          ['Structure', 'ATM fly ±2% wings', 'jade: −2% put / +1–2.5% call sp / −4% put', 'reverse: +2% call / −1/−2.5% put sp / +4% call'],
          ['Exit', '2% move-stop · +40% PT · roll DTE≤1', 'roll DTE≤1', 'roll DTE≤1'],
        ],
      },
    ],

    results: {
      metrics: [
        { label: 'Calm base (5-day)', value: '59.6%' },
        { label: 'Calm — compression+VIX gate', value: '~69%', tone: 'pos', hint: '72.6% compression-only; ~86–90% at 3-day' },
        { label: 'Gate OOS AUC', value: '0.65', hint: 'walk-forward, >0.5 = skill' },
        { label: 'Mild-directional weeks', value: '31%', hint: 'bull 19% / bear 12%' },
        { label: 'Day-1 up → stays up', value: '88%', tone: 'pos', hint: 'after a >1% up day' },
        { label: 'Bull jade (day-1) EV', value: '+₹64k', tone: 'pos', hint: '81% win, −₹201k worst — per 10-lot proxy' },
      ],
      tables: [
        {
          title: 'Finalised entry rules — the three systems',
          caption: 'The conclusive output. Strikes are % of spot. Win-rate basis differs (fly = calm-survival; jade/bear = positive-trade proxy).',
          columns: ['System', 'Regime', 'Entry trigger', 'Strikes / wings', 'Exit', 'Win-rate'],
          rows: [
            ['Neutral iron fly', 'CALM', 'ATR%<1.1 ∧ CPR<0.16 ∧ Stoch>65 (≥2/3) + VIX 13–22; enter 09:20', 'SELL ATM CE+PE; BUY ±2% wings', '2% move-stop (gap→OR) · +40% PT · roll DTE≤1', '~69–75%'],
            ['Bull jade lizard', 'mild-bull / drift', 'day-1 UP > +0.5% + VIX 13–22', 'SELL −2% PE; SELL +1% / BUY +2.5% CE; BUY −4% PE', 'roll DTE≤1; defined ~−₹200k', '81%'],
            ['Bear reverse-jade', 'mild-bear (tactical/hedge)', 'day-1 DOWN < −0.5% + VIX 13–22', 'SELL +2% CE; SELL −1% / BUY −2.5% PE; BUY +4% CE', 'roll DTE≤1; defined ~−₹200k', '73%'],
          ],
          highlightRows: [0, 1],
        },
      ],
      charts: [
        {
          src: '/app/nifty_fly_payoffs.png',
          caption: 'Left: at-expiry payoff diagrams (per 10-lot, VIX~15 proxy) — symmetric iron fly vs bull jade-lizard (+4% put) vs bear reverse-jade; the ±2% calm zone is shaded. Middle: calm-survival by hold length (base vs compression + VIX 13–22 gate). Right: the P6 intra-hold "apply the brakes" curve — P(finish calm) vs drift-from-entry at day-3/day-4 close, with the ~1.4% caution line.',
        },
      ],
    },

    winners: [
      {
        config: 'Neutral iron fly + compression gate (the calm system)',
        summary: 'Sell the symmetric fly only when volatility/range is compressed (ATR/CPR/Stochastic) inside VIX 13–22 — lifts weekly calm-survival 59%→~75% (≈90% at a 3-day hold), strongly protective in volatile years.',
        metrics: [
          { k: '5-day calm (gated)', v: '~69% (vs 59% base)' },
          { k: '3-day calm (gated)', v: '~86–90%' },
          { k: 'OOS AUC', v: '0.65' },
          { k: 'Coverage', v: '~28% of days' },
        ],
        rejected: [
          'Inside-week filter — barely beats base (61% vs 59%); dominated by compression (retire it)',
          'ADX / Ichimoku / MA / RSI for calm — no usable signal',
          'VIX<15.4 as a calm flag — over-restricts; use VIX as a 13–22 regime, not a calm flag',
        ],
      },
      {
        config: 'Bull jade lizard · day-1-up-confirmed + 4% protective put (the directional winner)',
        summary: 'Direction can’t be timed at entry, but a green day-1 (>+0.5%) → 88% the week stays up. A defined-risk bullish jade entered then monetises NIFTY’s up-drift; the long −4% put caps the crash tail.',
        metrics: [
          { k: 'EV / trade', v: '+₹64k (proxy)' },
          { k: 'Win rate', v: '81%' },
          { k: 'Worst week', v: '−₹201k (defined)' },
          { k: 'Trigger', v: 'day-1 up >0.5%, VIX 13–22' },
        ],
        rejected: [
          'Naked jade — +₹87k EV but −₹795k crash tail (must add the long put)',
          'Broken-wing fly — still a calm-needing short straddle (poor)',
          'Per-trade direction timing from indicators — unpredictable (≤7pp)',
        ],
      },
      {
        config: 'Bear reverse-jade — tactical / hedge only',
        summary: 'A bearish skew has a SAFER tail (upside, gentler than NIFTY’s crash downside) but fights the up-drift and the day-1-down follow-through is weaker (73% vs 81%). Use on a confirmed down-day or as an uncorrelated hedge sleeve, not as the primary engine.',
        metrics: [
          { k: 'EV / trade', v: '+₹47k (proxy)' },
          { k: 'Win rate', v: '73%' },
          { k: 'Worst week', v: '−₹203k (defined)' },
          { k: 'Role', v: 'tactical / hedge' },
        ],
      },
    ],

    caveats: [
      'PRICE-ONLY study: no in-house multi-year option premiums, so calm-rate is the win proxy and structure ₹ uses a VIX-scaled premium. Trust the RELATIVE rankings; exact ₹ ⇒ the AlgoTest cards in the repo (iron fly + jade #1–6).',
      'Calm-rate ≠ net P&L — low VIX is calmest but thinnest on premium; the VIX≥13 floor trades ~4pp of calm for premium richness. The net-₹ optimum needs real premiums.',
      'Directional structures are NEGATIVE-SKEW: high win-rate masks a fat loss tail. The jade is defined-risk only WITH the long −4% put (~−₹200k); naked it is −₹795k. Not a free lunch.',
      'Day-1 confirmation is not expressible on AlgoTest — forward-paper it (the engine can shadow-log it like the compression gate) before any real capital.',
      'The compression gate is built + live as a SHADOW logger (records would-enter daily); it has NOT yet gated a real entry. The jade/bear structures are research-only — nothing is wired to live orders.',
      'Single instrument (NIFTY), in-sample over one 11-year history (walk-forward within it). A robust SIGNAL set, not yet live-validated.',
      'Idle-cash alternates (research/65 low-VIX long, research/66 high-VIX long) are robustness-checked on the UNDERLYING; their DEFINED-RISK option versions AND the blended-book ₹/Calmar both still need real premiums (AlgoTest). A simple 5% stop does NOT make the high-VIX long defined-risk — it cuts return without capping the −11% DD; only a debit spread does.',
    ],

    githubLinks: [
      { label: '← Builds on: V2 Iron Fly (Stop-Loss × VIX)', href: '/app/backtest/v2-nifty-ironfly-sl-vix' },
      { label: 'research/64 — calm-day + directional entry study', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/64_calm_day_ironfly_entry' },
    ],
    projectPaths: [
      'research/64_calm_day_ironfly_entry/RESULTS.md (P1→P5 findings)',
      'research/64_calm_day_ironfly_entry/CALM_DAY_IRONFLY_ENTRY_DAILY_SWEEP_STATUS.md',
      'research/64_calm_day_ironfly_entry/ALGOTEST_JADE_CARD.md (+ ALGOTEST_TEST_CARD.md)',
      'research/64_calm_day_ironfly_entry/scripts/ (calm_study, p2, p2b, p3, p5*-directional)',
      'services/v2_ironfly_api.py — live compression shadow logger (/api/v2-ironfly/compression)',
    ],
  },
  {
    slug: 'midcap-rs120-regime-momentum',
    title: 'MidSmallcap400-MQ Concentrated Rotation (mid-cap RS-120 + 200DMA regime)',
    verdict:
      'Concentrated monthly RS-120 rotation on a survivorship-free mid-cap band, gated by a NIFTYBEES-200DMA regime switch, robustly beats the ~20% MidSmallcap400-MQ100 hurdle — 35.3% gross / 28.9% post-tax CAGR at index-level −24.6% drawdown.',
    status: 'COMPLETE',
    date: '2026-05-16',
    cardBlurb:
      'Survivorship-free PIT mid-cap liquidity band, RS-120 vs NIFTYBEES, 15 names equal-weight, monthly rotation with a top-22 buffer, plus a price-path quality screen and a 200DMA market-regime cash switch. Validated OOS and post-tax.',
    cardStats: [
      { label: 'CAGR (gross)', value: '35.3%' },
      { label: 'CAGR (post-tax 20%)', value: '28.9%' },
      { label: 'MaxDD', value: '−24.6%' },
    ],

    systemRules: {
      intro:
        'The actual traded system. Three named candidates share one identical stock-selection core (below) and differ ONLY in the risk layer. The original SMA200 gate (q0.5_dd__v__REG) is the baseline this evolved from — superseded by the SMA100-based variants per Phase 09; the rules here are authoritative, the Phase tables that follow are the evidence trail.',
      sharedCoreTitle: 'Shared core — identical for all three; evaluated monthly',
      sharedCore: [
        {
          k: 'Universe',
          v: 'Survivorship-free point-in-time mid-cap liquidity band = rank 101–250 by trailing-6-month median (close × volume), rebuilt every month from ~1,623 NSE daily symbols (not index membership).',
        },
        {
          k: 'Signal',
          v: 'Relative Strength RSᵢ = (Pᵢ[t] / Pᵢ[t−120]) / (NIFTYBEES[t] / NIFTYBEES[t−120]); rank high→low.',
        },
        {
          k: 'Quality screen (q0.5)',
          v: 'Take the last 252 trading days, split into 12 consecutive 21-day blocks; keep a stock only if ≥ 6 of the 12 blocks ended higher than they started.',
        },
        {
          k: 'Entry filter',
          v: 'Price ≥ 90% of its point-in-time all-time high (within 10% of ATH).',
        },
        { k: 'Hold', v: 'Top 15, equal-weight.' },
        {
          k: 'Rotation',
          v: 'Monthly; top-22 retention buffer — a holding is sold only when it drops out of the top 22 by RS (low churn).',
        },
        {
          k: 'Costs / cash / tax',
          v: '0.4% round-trip on turnover; idle/cash 6.5% p.a.; post-tax = net 20% STCG on lots held < 365 days (LTCG not modelled).',
        },
        {
          k: 'Backtest window',
          v: '2014→2026 (~12.1y, incl. 2018/2020/2022/2025 bears).',
        },
      ],
      riskLayer: {
        title: 'Per-system risk layer — the only difference between the three',
        caption:
          'SMOOTHEST uses a weekly regime check (Phase-15 result: cuts drawdown without whipsaw; daily over-trades). MAX-RETURN/FORTIFIED are indifferent to regime cadence so stay month-end. Caveats: price-path quality ≠ fundamentals; PIT universe is a liquidity proxy; the Nifty short is modelled frictionless and 1× under-hedges mid-cap β>1 (live would be worse); LTCG not netted; nothing wired live.',
        columns: ['System', 'Regime gate', 'Risk-off action', 'Stock-level exits'],
        rows: [
          [
            'SMOOTHEST',
            'NIFTYBEES vs its 100-day SMA, checked WEEKLY',
            'Liquidate entire book to cash @6.5% until risk-on',
            'per-stock-100-SMA exit + 12% trailing stop (applied at month-ends)',
          ],
          [
            'MAX-RETURN',
            'NIFTYBEES vs 100-day SMA, checked month-end',
            'Stay invested + short 1× Nifty notional (rolled monthly while risk-off; removed when risk-on)',
            'none',
          ],
          [
            'FORTIFIED',
            'NIFTYBEES vs 100-day SMA, checked month-end',
            'Stay invested + short 1× Nifty (same as Max-Return)',
            'per-stock-100-SMA exit + 12% trailing stop',
          ],
        ],
      },
    },

    system: {
      intro:
        'Out of the Nifty MidSmallcap-400 Momentum-Quality space (NSE index ~20% CAGR), can a concentrated, frequently-rotated stock-selection rule consistently and robustly beat the index — validated survivorship-free, with honest drawdown, tax and out-of-sample treatment? Hurdle = ~20% CAGR. The exact traded rules are stated up front in System Rules; the system is one shared selection core plus a per-variant risk layer with three named variants (SMOOTHEST / MAX-RETURN / FORTIFIED — see System Rules). The block below describes that shared core. The original SMA200 gate (q0.5_dd__v__REG) is only the baseline this evolved from — superseded by the SMA100-based variants per Phase 09; it is never the current system.',
      rows: [
        {
          k: 'Backtest universe',
          v: 'Survivorship-free point-in-time (PIT) mid-cap liquidity band = rank 101–250 by trailing-6-month median daily traded value (close × volume), rebuilt every month (no look-ahead) from ~1,623 NSE daily symbols (2000→2026). Not index membership. Eligibility ≥ 75 priced bars in the lookback; top-100 dropped as large-cap.',
        },
        {
          k: 'Liquidity bands tested',
          v: 'mid = rank 101–250 (chosen, locked) · small = 251–500 · combo = 101–500. A separate semi-annual reconstruction sanity-checked the proxy: ~68/100 of today\'s supplied MQ100 fall in the reconstructed 101–500 band.',
        },
        {
          k: 'Live-pick universe',
          v: 'For today\'s actionable list only: the 100 supplied MQ100 constituents (universe_mq100_2026-05-15.csv). 4 ticker renames remapped; 91/100 have ≥120d history.',
        },
        {
          k: 'Core signal — Relative Strength',
          v: 'RSᵢ = (Pᵢ[t] / Pᵢ[t−120]) / (NIFTYBEES[t] / NIFTYBEES[t−120]). BENCH = NIFTYBEES (Nifty-50 ETF, full daily history 2005→2026). RS is a ratio so the ETF price scale cancels. Within the mid band, sort eligible names by RS descending.',
        },
        {
          k: 'Quality screen (q0.5)',
          v: 'Last 252 trading days split into 12 consecutive 21-day blocks; keep a name only if ≥ 6 of the 12 blocks ended higher than they started. Price-path proxy — not fundamentals.',
        },
        {
          k: 'Entry filter',
          v: 'Price ≥ 90% of its point-in-time all-time high (within 10% of ATH). Volume-breakout confirm = OFF (tested, rejected).',
        },
        {
          k: 'Hold & rotation',
          v: 'Top 15, equal-weight. Monthly rebalance. Top-22 retention buffer (N × 1.5 hysteresis on RS rank): a held name is kept while it stays in the top 22; only names falling out of the top 22 are sold; freed slots refill from the top 15 down. Cuts churn → less cost and less STCG.',
        },
        {
          k: 'Regime / risk layer',
          v: 'Selection feeds a market-regime risk layer that can flatten the book or short Nifty. This is the ONLY axis on which the three named variants (SMOOTHEST / MAX-RETURN / FORTIFIED) differ — exact gates, risk-off actions and stock-level exits per variant are in System Rules. The original SMA200→cash gate (q0.5_dd__v__REG) is the superseded baseline only.',
        },
        {
          k: 'Fundamentals',
          v: 'Enter nowhere in the pipeline. "Quality" = price-path proxy only. Current ROE/D-E/PAT/ROCE appear solely as a post-selection human annotation on the live top-15 — they do not re-rank or remove anything.',
        },
      ],
    },

    conditions: {
      intro:
        'Exact costs, cash, tax and data window the validated numbers were produced under (shared by all three variants — the per-variant regime/risk differences are in System Rules).',
      rows: [
        { k: 'Frequency', v: 'Monthly rebalance on each month-end bar.' },
        { k: 'Portfolio size N', v: '15 (swept 10/15/20/25/30).' },
        { k: 'Retention buffer', v: 'top-22 (N × 1.5) hysteresis on RS rank.' },
        {
          k: 'Regime check',
          v: 'A market-regime gate runs every period and can flatten the book to cash or short Nifty irrespective of RS. The active variants gate on NIFTYBEES vs its 100-session SMA (SMOOTHEST checks it WEEKLY; MAX-RETURN/FORTIFIED month-end). The 200-session SMA gate is the superseded original baseline (Phase 09). Full per-variant detail in System Rules.',
        },
        { k: 'Transaction cost', v: '0.4% round-trip applied on the fraction of the book that changes each period (brokerage+STT+impact, small-cap level).' },
        { k: 'Idle / bear cash', v: '+6.5% p.a. (debt), modelled explicitly — not 0%.' },
        {
          k: 'STCG (held <365d, sold at gain)',
          v: 'Modelled in Phase 04: 15% (pre-Jul-2024) and 20% (current). Headline post-tax CAGR uses net 20%.',
        },
        {
          k: 'LTCG',
          v: 'Not modelled — monthly rotation is overwhelmingly short-term so the omission is small; it errs toward understating total tax. Stated, not hidden.',
        },
        {
          k: 'Window',
          v: '2014-01-01 → 2026 (~12.1y); includes the 2018-19 small-cap bear, Mar-2020, 2022, and the 2025 drawdown.',
        },
        {
          k: 'RS lookback L swept',
          v: '55d, 120d, 126d (~6m), 252d (~1y), and a 126+252 blend. 120d chosen.',
        },
      ],
    },

    comparisons: [
      {
        title: '6a. RS-alone sweep — 75 configs (3 bands × 5 lookbacks × 5 sizes)',
        caption:
          'Corrected run #2. 75/75 beat the 20% hurdle raw (CAGR 25–41%). Selected RS-alone leaders.',
        columns: ['Config', 'CAGR', 'Sharpe', 'MaxDD', 'Calmar', 'top-3 share'],
        rows: [
          ['mid_126d_6m_N10', '40.7%', '1.35', '−33.5%', '1.21', '14.1%'],
          ['combo_blend_6m12m_N25', '40.4%', '1.39', '−38.8%', '1.04', '10.1%'],
          ['mid_120d_N10', '39.9%', '1.34', '−34.6%', '1.15', '16.2%'],
          ['mid_120d_N15 (chosen core)', '38.3%', '1.39', '−29.8%', '1.29', '11.9%'],
          ['mid_120d_N20', '35.8%', '1.39', '−28.1%', '1.27', '9.6%'],
          ['mid_126d_6m_N25', '34.6%', '1.41', '−25.5%', '1.36', '8.3%'],
        ],
        highlightRows: [3],
      },
      {
        title: '6b. Super-winner robustness — ex-top-3 (false-indication guard)',
        caption:
          'Top-12 configs re-run forbidding their 3 best lifetime contributors. 12 still beat 20% (ex-top-3 CAGR 34–39%); top-3 profit share only ~8–15% → the edge is breadth, not 1–2 multibaggers.',
        columns: ['Config', 'CAGR', 'ex-top3', 'Sharpe', 'MaxDD', 'Calmar', 'top3 share'],
        rows: [
          ['mid_120d_N15', '38.3%', '33.9%', '1.39', '−29.8%', '1.29', '11.9%'],
          ['mid_126d_6m_N15', '38.4%', '36.3%', '1.39', '−31.0%', '1.24', '11.8%'],
          ['combo_126d_6m_N20', '37.5%', '36.0%', '1.30', '−34.3%', '1.09', '9.8%'],
          ['combo_blend_6m12m_N30', '38.4%', '37.1%', '1.37', '−36.2%', '1.06', '8.5%'],
          ['combo_blend_6m12m_N25', '40.4%', '39.0%', '1.39', '−38.8%', '1.04', '10.1%'],
        ],
        highlightRows: [0],
      },
      {
        title: '6d. Phase 03 — 53 drawdown-control overlays on mid_120d_N15',
        caption:
          'Goal: shrink the −30% DD toward the index −24% without dropping CAGR below 35%. Baseline mid_120d_N15: 38.4% / −29.8% / Calmar 1.29. Goal-test winners + top configs by Calmar shown.',
        columns: ['Config', 'CAGR', 'Sharpe', 'MaxDD', 'Calmar', 'Note'],
        rows: [
          ['q0.5_dd__v__REG ★', '35.3%', '1.53', '−24.6%', '1.44', 'quality0.5 + regime; best in study'],
          ['q0.5_dd__v__nor', '37.0%', '1.35', '−29.6%', '1.25', 'quality only — ~neutral on DD'],
          ['q0.58_dd__v__REG', '33.5%', '1.51', '−24.5%', '1.37', 'high-Calmar'],
          ['q0.5_dd-0.5_v__REG', '32.2%', '—', '−23.4%', '1.37', 'tighter own-DD cap'],
          ['q0.5_dd-0.4_v__REG', '30.6%', '1.45', '−22.5%', '1.36', 'conservative: DD beats the index'],
          ['q__dd__v__REG (regime only)', '34.8%', '—', '−26.4%', '1.32', 'regime alone'],
        ],
        highlightRows: [0],
      },
      {
        title: '6f. Run comparison summary',
        caption:
          'Run #1 used NIFTY50 whose DB series only exists 2023-03→2026 → 8/12y compounded idle cash. Its "0/75 beat 20%" was a fabricated negative and is VOID. All reported numbers are from corrected run #2 onward.',
        columns: ['Run', 'Universe / benchmark', 'Verdict', 'Status'],
        rows: [
          ['Run #1 (RS sweep)', 'NIFTY50 (2023+ only)', '"0/75 beat 20%"', 'VOID — 8/12y in cash'],
          ['Run #2 (RS sweep)', 'NIFTYBEES (2005+)', '75/75 beat 20%; 12 robust', 'valid'],
          ['Phase 03 (53 overlays)', 'NIFTYBEES', 'q0.5_dd__v__REG 35.3%/−24.6%', 'valid'],
          ['Phase 04 (OOS+tax)', 'NIFTYBEES', 'stable both halves; 28.9% post-tax', 'PASS'],
        ],
        highlightRows: [0],
      },
      {
        title: '9. Universe decision: MID vs SMALL vs COMBO (LOCKED: MID)',
        caption:
          'Same regime+quality overlay + OOS + post-tax pipeline run on all three PIT bands, apples-to-apples (gated champion, post-tax @20% STCG). MID is the locked recommended system — shallowest drawdown, best Calmar (1.44), far more tradable (22 F&O stocks vs small\'s 1 — small\'s real costs likely exceed the modelled 0.4% RT, so its 30.2% is optimistic; mid\'s 28.9% is trustworthy), and the smallest working universe (150). COMBO is strictly dominated by MID (lower post-tax CAGR AND deeper DD). SMALL is a higher-pain alternative only.',
        columns: [
          'Universe (config)',
          'Post-tax CAGR',
          'MaxDD',
          'Sharpe',
          'Gross Calmar',
          'OOS H1 / H2',
          'F&O stocks in band',
        ],
        rows: [
          ['MID  q0.5_dd__v__REG  ✅', '28.9%', '−24.6%', '1.53', '1.44', '32.2 / 37.3', '22 / 150'],
          ['SMALL q0.5_dd-0.4_REG', '30.2%', '−28 to −30%', '1.56', '1.27', '35.0 / 35.1', '1 / 250 (IRCTC)'],
          ['COMBO q0.58_dd-0.4_REG', '28.1%', '−30.6%', '1.31', '1.13', '32.0 / 33.8', '23 / 400'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Phase 09: regime-filter alternatives (vs laggy SMA200)',
        caption:
          'SMA100 replaces the laggy SMA200 gate — same CAGR, MaxDD −24.6%→−16.4%, Calmar 1.44→2.14. Adding the ATH≤10% entry screen on SMA100 → 35.2/29.3/−15.1/Sharpe 1.78/Calmar 2.33. The 20% trailing stop was inert; ATR/vol-spike regime failed (NIFTYBEES has no true OHLC — c2c ATR proxy, flagged). Core held constant: mid_120d_N15 + q0.5.',
        columns: ['Regime', 'CAGR %', 'Post-tax @20% %', 'MaxDD %', 'Sharpe', 'Calmar'],
        rows: [
          ['OFF', '37.0', '30.9', '−29.6', '1.35', '1.25'],
          ['SMA200 (old lock)', '35.3', '29.4', '−24.6', '1.53', '1.44'],
          ['SMA100', '35.1', '29.5', '−16.4', '1.66', '2.14'],
          ['SMA50', '29.7', '23.6', '−19.1', '1.55', '1.56'],
          ['cross 50/200', '31.9', '26.5', '−33.3', '1.30', '0.96'],
          ['DD-from-1yr-high>10%', '31.5', '26.0', '−31.3', '1.24', '1.01'],
          ['3m-momentum<0', '31.4', '26.1', '−21.9', '1.48', '1.44'],
          ['volspike (ATR)', '33.9', '27.0', '−33.4', '1.40', '1.02'],
          ['SMA200+vol', '33.8', '27.6', '−20.9', '1.54', '1.61'],
        ],
        highlightRows: [2],
      },
      {
        title: 'Phase 10: drawdown-hedge overlay',
        caption:
          'In risk-off, holding the stocks and shorting 1× Nifty (vs going to cash) harvests the RS spread as market-neutral alpha → 34.0% post-tax, the project\'s highest. It does NOT reduce drawdown (−22.7 vs cash −15.1; mid-cap β>1 under-hedged) — a return amplifier, not a DD reducer. Permanent hedge bleeds the bull; covered calls rejected (caps the CAGR tail; rotating mid-cap holdings mostly lack liquid options).',
        columns: ['Config', 'CAGR %', 'Post-tax @20% %', 'MaxDD %', 'Sharpe', 'Calmar'],
        rows: [
          ['SMA100→cash (Ph09 best)', '35.2', '29.3', '−15.1', '1.78', '2.32'],
          ['SMA100→beta-hedge hr1.0', '42.8', '34.0', '−22.7', '1.83', '1.89'],
          ['SMA100→beta hr0.5', '37.8', '29.5', '−24.9', '1.58', '1.52'],
          ['OFF no-hedge', '32.7', '24.8', '−32.8', '1.32', '1.00'],
          ['always-hedge hr0.25', '27.8', '20.5', '−28.8', '1.25', '0.96'],
          ['always-hedge hr0.40', '24.8', '17.9', '−27.0', '1.19', '0.92'],
          ['always-hedge hr0.60', '20.9', '14.4', '−28.2', '1.09', '0.74'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Phase 11: stock-level vs market-level risk control',
        caption:
          'Stock-level control ALONE cannot replace the market gate (no-gate variants stuck ~−30/−32% DD, Calmar ~1.0–1.1). On TOP of the gate it adds a small free gain: Calmar 2.32→2.36, +0.3pp post-tax, same −15.1% DD.',
        columns: ['Config', 'CAGR %', 'Post-tax @20% %', 'MaxDD %', 'Sharpe', 'Calmar'],
        rows: [
          ['SMA100 mkt (Ph09 winner)', '35.2', '29.3', '−15.1', '1.78', '2.32'],
          ['OFF + trail15', '33.0', '25.0', '−32.4', '1.33', '1.02'],
          ['OFF + trail12', '33.2', '25.2', '−32.2', '1.34', '1.03'],
          ['OFF + trail10', '33.4', '25.4', '−32.0', '1.35', '1.04'],
          ['OFF + perStockSMA100', '33.2', '24.9', '−30.2', '1.33', '1.10'],
          ['OFF + perStockSMA + trail12', '33.3', '25.0', '−30.1', '1.34', '1.11'],
          ['perStockSMA only (no mkt)', '33.2', '24.9', '−30.2', '1.33', '1.10'],
          ['SMA100 + perStockSMA', '35.5', '29.6', '−15.1', '1.80', '2.35'],
          ['SMA100 + trail12', '35.4', '29.4', '−15.1', '1.79', '2.34'],
          ['SMA100 + perStockSMA + trail12', '35.6', '29.6', '−15.1', '1.80', '2.36'],
        ],
        highlightRows: [9],
      },
      {
        title: 'Phase 22/24 — SMOOTHEST de-risk variants (locked WEEKLY cadence, daily-marked, fresh VPS data → 2026-05-15)',
        caption:
          "This table is on the engine SMOOTHEST ACTUALLY runs: monthly selection, WEEKLY regime check (Phase-15 lock), daily-marked drawdown. Numbers refreshed on VPS canonical data through 2026-05-15. 'C keep-top8' (in risk-off keep the 8 highest-RS holdings, cash the weaker 7, refill to 15 at the next risk-on monthly rebalance) is the single best refinement: Calmar 1.54→1.66, MaxDD −22.2→−20.2%, post-tax essentially flat (28.4→28.3). It still beats base after tax-friction and on fresh data. CORRECTION: an earlier note here claimed keep-top8 'reverses to worse' — that came from re-running it on the MONTH-END engine, which silently also changed the regime clock weekly→monthly (two changes at once) and was not a fair test. Re-tested cadence-matched (this table + the dedicated chart below), keep-top8 is a modest but genuine improvement and is defensible to adopt. Honest caveat: its one weak year is 2025 (−6.9% vs base +5.3% — it holds 8 mid-caps through that risk-off while base sits in cash); the full-period shallower max-drawdown still wins. A no-regime rejected; B trims dominated (tax-ruinous); D tighter per-stock SMA a slight positive. Phase 25 also tested a user-proposed GATED refill — in risk-off, refill freed slots with names still passing the full strength filter (RS + above-own-100SMA + within-10%-ATH), else cash. REJECTED at every cap (10/12/15): MaxDD blows out to ~−34% and Calmar collapses to ~1.0 — in shallow/early downturns names still pass the gate, get bought, then roll over (2016, 2025). The strict filter is not a sufficient circuit-breaker; keep-top8 with NO refill stays best.",
        columns: ['Config', 'CAGR %', 'Post-tax @20% %', 'MaxDD %', 'Sharpe', 'Calmar', 'Verdict'],
        rows: [
          ['BASE SMOOTHEST (all-cash)', '34.2', '28.4', '−22.2', '1.82', '1.54', 'reference'],
          ['A no-regime', '34.3', '26.2', '−37.6', '1.43', '0.91', 'rejected'],
          ['B trim-25 (hold 75%)', '34.7', '24.5', '−30.7', '1.59', '1.13', 'dominated'],
          ['B trim-50 (hold 50%)', '34.8', '21.5', '−26.4', '1.73', '1.32', 'dominated'],
          ['C keep-top5', '34.3', '28.9', '−22.2', '1.78', '1.54', 'neutral (= base)'],
          ['C keep-top8', '33.6', '28.3', '−20.2', '1.71', '1.66', 'BEST — beats base, defensible'],
          ['D perstock-SMA80', '34.6', '28.8', '−22.1', '1.84', '1.57', 'slight+'],
          ['D perstock-SMA60', '34.7', '28.8', '−21.5', '1.84', '1.61', 'mild+'],
          ['Ph25 keep-top8 + gated refill (cap-12)', '34.7', '27.0', '−33.9', '1.56', '1.02', 'REJECTED — DD blows out'],
        ],
        highlightRows: [5],
      },
      {
        title: 'Phase 32/33 — gradual de-risk refinements: two new client variants (daily-marked, fresh VPS data)',
        caption:
          'Two refinements of keep-top8 governing HOW the book de-risks in a downturn — both keep the NIFTYBEES-100SMA weekly gate. ★ "Keep-8 + Bear Trend-Trim" (RECOMMENDED) — when the market turns bear, keep the 8 highest-RS holdings (cash the weaker 7) AND, ONLY while bear, also exit any of those 8 that closes below its own 100-day SMA; refill to 15 at the next risk-on month-end. Best balance: Calmar 1.66→1.70, MaxDD −20.1%, post-tax 28.9%, low churn (~60 stock-exits/12y), never fully liquidates to debt. "Always-On Trend-Guard" — the same bear gate PLUS a per-stock 100-SMA exit that runs EVERY week (bull AND bear), so a holding breaking its 100-day trend is sold in any regime: lowest drawdown of any variant (−18.9%, Calmar 1.75) but ~1pp lower post-tax (27.8%) from extra bull-market churn-tax. Context rows: all-cash+weekly re-entry (highest return, but all-or-nothing exits) and the gate-less pure per-stock 100-SMA (REJECTED — DD −28 to −35%; the market gate is irreplaceable). Choose on client mandate: best balance vs lowest-drawdown vs max-return. All daily-marked on VPS data, 2014–2026; both baseline refs reproduce the locked engine exactly.',
        columns: ['Variant (risk-off action)', 'CAGR %', 'Post-tax @20% %', 'MaxDD %', 'Sharpe', 'Calmar', 'Best for'],
        rows: [
          ['keep-top8 (baseline refinement)', '33.6', '28.3', '−20.2', '1.71', '1.66', 'simple gradual'],
          ['★ Keep-8 + Bear Trend-Trim  (RECOMMENDED)', '34.2', '28.9', '−20.1', '1.76', '1.70', 'best balance · low churn · never full-dump'],
          ['Always-On Trend-Guard', '32.9', '27.8', '−18.9', '1.73', '1.75', 'lowest drawdown'],
          ['all-cash + weekly re-entry', '35.5', '29.0', '−20.7', '1.84', '1.72', 'max return (all-or-nothing exits)'],
          ['pure per-stock 100-SMA, NO gate', '34.1', '26.1', '−28.2', '1.53', '1.21', 'REJECTED — gate irreplaceable'],
        ],
        highlightRows: [1],
      },
    ],

    results: {
      metrics: [
        { label: 'CAGR (gross)', value: '35.3%', tone: 'pos', hint: 'q0.5_dd__v__REG on mid_120d_N15' },
        { label: 'CAGR (post-tax 20% STCG)', value: '28.9%', tone: 'pos', hint: 'clears the ~20% hurdle by ~9pp' },
        { label: 'CAGR (post-tax 15% STCG)', value: '30.4%', tone: 'pos', hint: 'pre-Jul-2024 rate' },
        { label: 'Max drawdown', value: '−24.6%', tone: 'neg', hint: 'index-level, regime-controlled' },
        { label: 'Sharpe', value: '1.53', hint: 'best in the whole study' },
        { label: 'Calmar', value: '1.44', hint: 'drawdown-efficiency leader' },
      ],
      tables: [
        {
          title: 'Phase 04A — Sub-period stability (fixed config, disjoint halves)',
          caption: 'Edge strong in both halves — not a single-regime artifact. PASS.',
          columns: ['Window', 'CAGR', 'MaxDD', 'Sharpe'],
          rows: [
            ['Full 2014–2026', '35.3%', '−24.6%', '1.53'],
            ['H1 2014–2019', '32.2%', '−24.6%', '1.46'],
            ['H2 2020–2026', '37.3%', '−14.7%', '1.54'],
          ],
        },
        {
          title: 'Phase 04C — Post-tax (STCG) drag',
          caption:
            'STCG applied to gains on positions held <365d. Post-tax 28.9% still clears the ~20% hurdle by ~9pp. The meaningful figure is the 5–6pp CAGR drag (the log\'s "cum tax ~5× init" is a scale artifact). LTCG not modelled.',
          columns: ['', 'CAGR', 'MaxDD', 'Sharpe', 'Drag'],
          rows: [
            ['Gross', '35.3%', '−24.6%', '1.53', '—'],
            ['Net STCG @15% (pre-Jul-2024)', '30.4%', '−25.1%', '1.38', '−4.9pp'],
            ['Net STCG @20% (current)', '28.9%', '−25.3%', '1.33', '−6.4pp'],
          ],
          highlightRows: [2],
        },
        {
          title: 'Phase 04B — Walk-forward lookback selection',
          caption:
            'Each year 2019→2026 the RS lookback was re-picked by best trailing-3y Calmar (no peeking) and traded that year, chained. PASS — the procedure only ever picked 120d / 126d_6m (never 55d / 252d).',
          columns: ['Method', 'CAGR (2019–2026)', 'Verdict'],
          rows: [
            ['Walk-forward (re-pick L yearly)', '33.1%', 'lookback choice robust, not lucky'],
            ['Static L=120', '35.0%', '1.9pp gap within noise'],
          ],
        },
        {
          title: 'Phase 26 — cash-flow policy (live-readiness): the system is ROBUST to deposits/withdrawals',
          caption:
            'How a live investor adds extra cash or takes money out barely changes the outcome. 20 policies (5 inflow × 4 outflow) on SMOOTHEST+keep-top8, weekly daily-marked engine, fresh VPS data, under a realistic scenario: monthly SIP + lump deposits + lump withdrawals INCLUDING one forced at the 2020 COVID trough. All 20 land within <1% final wealth, 0.4pp post-tax money-weighted XIRR, and an IDENTICAL −20.2% drawdown — even the crash-forced withdrawal scarred no policy. Tax-aware lot selection gave no edge (monthly rebuild + long horizon washes out lot timing). Live takeaway: do NOT over-engineer deposit/withdrawal logic — the existing monthly rebuild absorbs flows efficiently; no special machinery needed.',
          columns: ['Policy', 'TWR %', 'XIRR %', 'XIRR post-tax %', 'Daily MaxDD %', 'Final ×', 'Verdict'],
          rows: [
            ['C3 deploy→top-RS + W1 cash-first/pro-rata', '33.6', '32.2', '26.9', '−20.2', '47.17', 'marginal best'],
            ['C1 park-till-rebalance + W1', '33.5', '32.1', '26.5', '−20.2', '46.80', 'simplest — tied within noise'],
            ['Spread across all 20 combos', '33.5–33.6', '32.1–32.2', '26.5–26.9', '−20.2', '46.7–47.2', 'robust — policy ~irrelevant'],
          ],
          highlightRows: [0],
        },
        {
          title: "Today's 15 — SMOOTHEST selection (as-of 2026-05-15, VPS canonical data)",
          caption:
            'The system\'s RS-ranked top-15 from the PIT mid-cap band on the latest trading day, all passing q0.5 + above-own-100SMA + within-10%-of-ATH. REGIME IS RISK-OFF (NIFTYBEES 267.30 < its 100-SMA 280.37) → the locked SMOOTHEST base would hold ZERO of these (100% cash); the keep-top8 refinement (the validated risk-off variant — see Phase 22/24) would hold the top-8 (✓ KT8 col). This is the would-be book if risk-on. Not a recommendation, no live wiring. % from ATH = distance below all-time-high; PosFrac = share of positive 21-day blocks (quality screen, ≥0.50).',
          columns: ['#', 'Symbol', 'RS', '% from ATH', 'PosFrac', 'Last close', 'KT8 top-8'],
          rows: [
            ['1', 'MTARTECH', '3.07', '−4.7%', '0.58', '7234.0', '✓'],
            ['2', 'HFCL', '2.21', '−8.6%', '0.58', '147.89', '✓'],
            ['3', 'TDPOWERSYS', '1.90', '0.0%', '0.75', '1311.3', '✓'],
            ['4', 'ATHERENERG', '1.51', '−3.3%', '0.83', '937.4', '✓'],
            ['5', 'LAURUSLABS', '1.48', '0.0%', '0.83', '1323.6', '✓'],
            ['6', 'BHARATFORG', '1.46', '−3.8%', '0.75', '1913.1', '✓'],
            ['7', 'MAHABANK', '1.41', '−8.6%', '0.67', '78.02', '✓'],
            ['8', 'JAINREC', '1.40', '−0.9%', '0.75', '566.15', '✓'],
            ['9', 'BELRISE', '1.40', '−6.4%', '0.75', '209.46', '—'],
            ['10', 'DATAPATTNS', '1.40', '−8.1%', '0.58', '3876.5', '—'],
            ['11', 'GLENMARK', '1.39', '−3.8%', '0.67', '2325.9', '—'],
            ['12', 'SOLARINDS', '1.38', '−1.6%', '0.58', '17314.0', '—'],
            ['13', 'NAM-INDIA', '1.37', '−0.3%', '0.75', '1100.6', '—'],
            ['14', 'KEI', '1.37', '−1.7%', '0.75', '5117.5', '—'],
            ['15', 'AUROPHARMA', '1.35', '−3.7%', '0.58', '1511.8', '—'],
          ],
          highlightRows: [0, 1, 2, 3, 4, 5, 6, 7],
        },
      ],
      charts: [
        {
          src: '/app/midcap_finalists_yearly_heatmap.png',
          caption:
            'Yearly returns vs Nifty 50 (gross, daily-marked, 2014–2026) — the two NAMED de-risk finalists: ★ Keep-8 + Bear Trend-Trim (recommended) and Always-On Trend-Guard, vs keep-top8. All beat Nifty 50 in 9 of 13 years, compound ~33–34% vs Nifty 12.3%, hold MaxDD ~−18 to −20% vs Nifty −36%. Robustness (Phase 34): stable across disjoint halves (H1 ~29–30% / H2 ~37–38% CAGR), not a single-regime artifact. Keep-8 + Bear Trend-Trim cuts the soft years (2025 −3.5% vs keep-top8 −6.9%); Always-On Trend-Guard runs the shallowest drawdown in every sub-window (−17 to −19%) at ~1pp lower CAGR. Soft spot (all): 2022–2026 ~17–18% CAGR — momentum cooled — still ~2× the index. Losing years are large-cap-led (2018, 2019, 2025).',
        },
        {
          src: '/app/midcap_momentum_factsheet.png',
          caption:
            'CLIENT FACTSHEET (one-page tearsheet) — regime-gated midcap momentum vs Nifty 50, 2014–2026, net of 0.4% round-trip cost & 6.5% idle cash. KPI strip, growth-of-₹1 (log), drawdown, annual-vs-index bars, monthly-returns heatmap, rolling 12m, and stat tables. Headline: 35.2% CAGR vs Nifty 12.7% (+22.5%/yr), 40.5× vs 4.3×, Sharpe 1.42, MaxDD −15.1% vs −28.8%, Calmar 2.33, beats the index in 10 of 13 years. Generated by research/_utilities/tearsheet.py.',
        },
        {
          src: '/app/smoothest_vs_kt8_weekly.png',
          caption:
            'keep-top8 vs the base SMOOTHEST, CADENCE-MATCHED — both on the engine the locked system actually runs (monthly selection, WEEKLY regime check, daily-marked drawdown), fresh VPS data through 2026-05-15, log scale + drawdown panel. keep-top8 (green) tracks the base (blue) on return while running visibly shallower drawdowns: Calmar 1.54→1.66, MaxDD −22.2→−20.2%, post-tax flat (28.4→28.3). This is the fair comparison; it supersedes the earlier (withdrawn) month-end-engine chart that judged keep-top8 on the wrong regime clock. One weak year for keep-top8: 2025 (−6.9% vs base +5.3%).',
        },
        {
          src: '/app/final_systems_pl_overlay.png',
          caption:
            'Equity overlay — SMOOTHEST vs MAX-RETURN vs Nifty-50 (log scale, with drawdown panel), PIT mid-cap band, 2014–2026, month-end engine. The three named systems; engines/rulers differ — see caveats. (keep-top8 is compared separately above, on its correct weekly cadence.)',
        },
        {
          src: '/app/yearly_matrix_heatmap.png',
          caption:
            'Yearly returns heatmap — SMOOTHEST / MAX-RETURN / FORTIFIED vs Nifty 50 (gross), PIT mid-cap band, 2014–2026, month-end engine. Replaces the prior annual table. Note 2025: MAX-RETURN/FORTIFIED −11.8% / −11.4% vs Nifty 50 +11.7% — the regime-short backfire (long falling mid-caps + short a rising Nifty); SMOOTHEST −0.8% (cash, no short). See caveats.',
        },
      ],
    },

    winners: [
      {
        config: 'q0.5_dd__v__REG  ·  on the mid_120d_N15 core',
        summary:
          'Best risk-adjusted result in the whole study. RS-120 vs NIFTYBEES on the PIT mid liquidity band, 15 names equal-weight, monthly rotation with top-22 buffer, PLUS a ≥50%-positive-months quality screen and a NIFTYBEES-200DMA regime cash switch. Volume confirm OFF, own-DD cap OFF. OOS-stable and robust to losing its 3 best names; clears the ~20% hurdle by a wide margin even after tax.',
        metrics: [
          { k: 'CAGR (gross)', v: '35.3%' },
          { k: 'CAGR (post-tax, 20% STCG)', v: '28.9%' },
          { k: 'MaxDD', v: '−24.6%' },
          { k: 'Sharpe', v: '1.53' },
          { k: 'Calmar', v: '1.44' },
        ],
        rejected: [
          'Run #1 (NIFTY50 benchmark) — VOID: the DB NIFTY50 series only exists 2023-03→2026, so 8 of 12 years compounded idle cash at 6.5%. Its "0/75 beat 20%" is a fabricated negative — never cite its numbers.',
          'Volume-breakout confirmation (v1.0 / v1.2 axis) — REJECTED: every config collapses CAGR to ~17–23% and worsens drawdown (it blocks the very momentum entries RS selects). OFF in the winner.',
          'Short 55d RS lookback — worst drawdown bucket (−54% to −66%); only "won" the void run because that run saw only 2023–26.',
          'Conservative alternative q0.5_dd-0.4_v__REG — not the headline but valid: 30.6% CAGR at −22.5% MaxDD (shallower than the index) for the most risk-averse.',
          'ATR / vol-spike regime — FAILED (−33% DD, Calmar 1.02). NIFTYBEES has no true OHLC so ATR is a close-to-close proxy — flagged as weak/not implementable.',
          '20% trailing stop — INERT: the monthly top-22 RS buffer already rotates losers before −20% from peak, so the trail never binds. Don\'t bother.',
          'Permanent / always-on hedge — REJECTED: a constant short bleeds the bull (CAGR 28%→21%, Calmar <1).',
          'Beta-hedge hr0.5 — dominated by hr1.0 on every axis (lower CAGR, deeper DD, worse Calmar).',
          'Covered calls on the 15 holdings — REJECTED (not built): caps the right-tail that is the CAGR; the rotating mid-cap holdings mostly lack liquid options (only ~22 of the whole mid band is F&O).',
          'Stock-level-only control (trail / per-stock-SMA without the market gate) — cannot replace the market gate: bottom-up stops fire only after each name falls, too late in a broad bear (stuck ~−30/−32% DD, Calmar ~1.0–1.1).',
        ],
      },
      {
        config: 'SMOOTHEST · mid_120d_N15 + q0.5 + SMA100 regime + ATH≤10% entry + per-stock-SMA100 + 12% trail',
        summary:
          'Best risk-adjusted endpoint (Phases 09–11). Supersedes the original SMA200 lock (was 29.4% post-tax / −24.6% MaxDD / Calmar 1.44 — the biggest single project improvement, from the SMA100 + ATH instincts). SMA100 replaces the laggy SMA200 gate (same CAGR, DD −24.6→−16.4); the ATH≤10% entry screen and stock-level per-stock-SMA100 + 12% trail each add a small free gain on top of the market gate. Drawdown roughly halved at near-identical CAGR.',
        metrics: [
          { k: 'CAGR (gross)', v: '35.6%' },
          { k: 'CAGR (post-tax, 20% STCG)', v: '29.6%' },
          { k: 'MaxDD', v: '−15.1%' },
          { k: 'Sharpe', v: '1.80' },
          { k: 'Calmar', v: '2.36' },
        ],
      },
      {
        config: 'MAX RETURN · …same core + SMA100→beta-hedge hr1.0 (short 1× Nifty in risk-off instead of cash)',
        summary:
          'Highest post-tax CAGR of any config in the whole project. In risk-off months, instead of sitting in cash, hold the top-RS stocks and short a 1× Nifty notional — the long/short book harvests the RS spread as market-neutral alpha instead of dead cash (per-year: 2023 +70 vs cash +40; 2020 +108 vs +86; 2024 +63 vs +45). Note: this is a return amplifier, NOT a drawdown reducer — DD is −22.7% (vs the cash variant −15.1%) because mid-cap β>1 leaves it under-hedged; still far better than ungated −33%. Also supersedes the original SMA200 lock (29.4% / −24.6% / Calmar 1.44).',
        metrics: [
          { k: 'CAGR (gross)', v: '42.8%' },
          { k: 'CAGR (post-tax, 20% STCG)', v: '34.0%' },
          { k: 'MaxDD', v: '−22.7%' },
          { k: 'Sharpe', v: '1.83' },
          { k: 'Calmar', v: '1.89' },
        ],
      },
    ],

    caveats: [
      'Run #1 void (benchmark-data artifact) — never cite its numbers.',
      'No fundamentals in the strategy. "Quality" = price-path proxy. The index\'s actual Quality leg is not replicated — we beat its return via momentum, not its method. Fundamentals are a live-list annotation only.',
      'PIT universe is a liquidity-traded-value proxy, not real index membership (~68/100 MQ100 overlap).',
      'Drawdown is real (~−25% even after the regime filter); a live investor must survive a −25% to −40% equity hole to realize this CAGR.',
      'LTCG not modelled — slightly understates total tax.',
      'Live list is as-of the laptop snapshot date (2026-02-16) — re-run 05_live_top15.py on the VPS for a current-dated list.',
      'No performance guarantee. A measured, validated edge — not certainty. Nothing is wired live; real-capital deployment is a user decision.',
      'Genuine next phase (not done): put real point-in-time fundamentals into selection — requires a paid PIT fundamentals source (Capitaline/CMIE/Refinitiv).',
    ],

    githubLinks: [
      { label: '📊 How the money moves — visual workflow chart (gradual de-risk mechanics)', href: '/app/midcap-workflow.html' },
      { label: 'research/41_midsmall400_mq_concentrated (folder)', href: GH },
      { label: '01_reconstruct_universe.py', href: `${GH}/scripts/01_reconstruct_universe.py` },
      { label: '02_rs_sweep.py', href: `${GH}/scripts/02_rs_sweep.py` },
      { label: '03_rs_quality_volume.py', href: `${GH}/scripts/03_rs_quality_volume.py` },
      { label: '04_walkforward.py', href: `${GH}/scripts/04_walkforward.py` },
      { label: '05_live_top15.py', href: `${GH}/scripts/05_live_top15.py` },
      {
        label: 'MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md (§9 universe decision, §10 YoY)',
        href: `${GH}/results/MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md`,
      },
      {
        label: 'MIDCAP_WINNER_YOY_VS_BENCHMARKS.md',
        href: `${GH}/results/MIDCAP_WINNER_YOY_VS_BENCHMARKS.md`,
      },
      {
        label: 'SMALLCAP_RSBLEND_REGIME_MOMENTUM_RESULTS.md',
        href: `${GH}/results/SMALLCAP_RSBLEND_REGIME_MOMENTUM_RESULTS.md`,
      },
      {
        label: 'COMBO_RSBLEND_REGIME_MOMENTUM_RESULTS.md',
        href: `${GH}/results/COMBO_RSBLEND_REGIME_MOMENTUM_RESULTS.md`,
      },
      {
        label: 'MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md',
        href: `${GH}/results/MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md`,
      },
      {
        label: 'LIVE_TOP15_WITH_FUNDAMENTALS.md',
        href: `${GH}/results/LIVE_TOP15_WITH_FUNDAMENTALS.md`,
      },
      {
        label: 'REGIME_HEDGE_STOCKLEVEL_RESULTS.md (Phases 09/10/11 consolidated)',
        href: `${GH}/results/REGIME_HEDGE_STOCKLEVEL_RESULTS.md`,
      },
      {
        label: 'REGIME_ALTS_ATH_LAYER_HEDGE_DAILY_RUN_STATUS.md (live-status, §7/§8 verdict)',
        href: `${GH}/REGIME_ALTS_ATH_LAYER_HEDGE_DAILY_RUN_STATUS.md`,
      },
    ],
    projectPaths: [
      'research\\41_midsmall400_mq_concentrated\\',
      'research\\41_midsmall400_mq_concentrated\\scripts\\ (01–05 *.py)',
      'research\\41_midsmall400_mq_concentrated\\results\\ (*.csv, *.md)',
      'research\\41_midsmall400_mq_concentrated\\MIDSMALL400_MQ_CONCENTRATED_DAILY_SWEEP_STATUS.md',
    ],
  },

  {
    slug: 'mq-momentum-quality-ps30',
    title: 'MQ Portfolio — Momentum + Quality (PS30)',
    verdict:
      'Concentrated Momentum+Quality rotation on the Nifty-500 (30 names, semi-annual rebalance, ATH-drawdown exits, Darvas top-ups) compounds at 26.3% net vs the Nifty-50 14.0% (2023–2025), 81% win rate — but draws down deeper than the index (−26.9% vs −15.2%). A market-regime overlay is the highest-value upgrade.',
    status: 'COMPLETE',
    date: '2026-05-31',
    cardBlurb:
      'Momentum (near 52w-high) + Quality (growth/ROE/low-debt) screen, top-30 equal-weight on the Nifty-500, 80/20 equity/debt, Darvas breakout top-ups, 20%-from-ATH + 50% hard-stop exits. Net of full Indian transaction costs.',
    cardStats: [
      { label: 'CAGR (net)', value: '26.3%' },
      { label: 'vs Nifty 50', value: '+12.3%/yr' },
      { label: 'MaxDD', value: '−26.9%' },
    ],
    system: {
      intro: 'Long-only concentrated factor rotation; the traded rules:',
      rows: [
        { k: 'Universe', v: 'Nifty 500 (~375 names with clean daily data).' },
        { k: 'Momentum', v: 'Price within 10% of the 52-week high + strong trailing return.' },
        { k: 'Quality', v: 'Revenue/earnings growth, ROE, low leverage screens.' },
        { k: 'Hold', v: 'Top 30 equal-weight; ≤10% per name, ≤25% / ≤6 names per sector.' },
        { k: 'Capital', v: '80% equity + 20% debt reserve (NIFTYBEES idle cash @6.5%).' },
        { k: 'Top-ups', v: 'Darvas breakout top-ups funded from the debt reserve.' },
        { k: 'Exits', v: '20%-from-ATH drawdown exit (dominant) · 50% hard stop · semi-annual rebalance.' },
        { k: 'Costs', v: 'Full Indian model: brokerage + STT + GST + stamp + slippage.' },
      ],
    },
    conditions: {
      intro: 'Backtest window and benchmark.',
      rows: [
        { k: 'Period', v: 'Jan 2023 – Dec 2025 (3.0 years).' },
        { k: 'Capital', v: '₹1 crore initial, 80/20 equity/debt.' },
        { k: 'Benchmark', v: 'Nifty 50 (NIFTYBEES), same window.' },
      ],
    },
    comparisons: [
      {
        title: 'Annual return: MQ vs Nifty 50',
        columns: ['Year', 'MQ %', 'Nifty 50 %', 'Excess pp'],
        rows: [
          ['2023', '+55.0', '+20.2', '+34.8'],
          ['2024', '+19.1', '+10.4', '+8.7'],
          ['2025', '+9.1', '+11.7', '−2.6'],
        ],
        highlightRows: [0, 1],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR (net)', value: '26.3%', tone: 'pos' },
        { label: 'Nifty 50 CAGR', value: '14.0%' },
        { label: 'Excess / yr', value: '+12.3%', tone: 'pos' },
        { label: 'Sharpe', value: '1.09' },
        { label: 'Sortino', value: '1.20' },
        { label: 'Max Drawdown', value: '−26.9%', tone: 'neg', hint: 'deeper than Nifty −15.2%' },
        { label: 'Calmar', value: '0.98' },
        { label: 'Win rate', value: '81%' },
      ],
      tables: [
        {
          title: 'Strategy vs benchmark',
          columns: ['Metric', 'MQ (PS30)', 'Nifty 50'],
          rows: [
            ['CAGR', '26.3%', '14.0%'],
            ['Total return', '2.01x', '1.48x'],
            ['Sharpe', '1.09', '0.71'],
            ['Max Drawdown', '−26.9%', '−15.2%'],
            ['Calmar', '0.98', '0.92'],
          ],
          highlightRows: [0, 1, 2],
        },
      ],
      charts: [
        {
          src: '/app/mq_portfolio_factsheet.png',
          caption:
            'CLIENT FACTSHEET — MQ Momentum+Quality (PS30) vs Nifty 50, 2023–2025, net of full Indian transaction costs. KPI strip, growth-of-₹1 (log), drawdown-vs-index, annual bars, monthly heatmap, rolling 12m, stat tables. 26.3% CAGR vs 14.0% (+12.3%/yr), 2.01x vs 1.48x, Sharpe 1.09 — but MaxDD −26.9% vs −15.2% (deeper than the index, the case for a regime overlay). Generated by research/_utilities/tearsheet.py.',
        },
      ],
    },
    winners: [
      {
        config: 'PS30 (top-30, 80/20, ATH-drawdown exits)',
        summary: 'Beats the index on return and Sharpe with a high win-rate; the trade-off is a deeper drawdown.',
        metrics: [
          { k: 'CAGR', v: '26.3% net' },
          { k: 'Excess', v: '+12.3%/yr vs Nifty' },
          { k: 'Sharpe', v: '1.09' },
          { k: 'Win rate', v: '81%' },
          { k: 'MaxDD', v: '−26.9% (vs −15.2%)' },
        ],
        rejected: [
          'EQ95 headline (~32%): inflated by 95%+20%=115% over-allocation; the clean 80/20 path is 26.3%.',
        ],
      },
    ],
    caveats: [
      'Short 3-year window dominated by 2023 (+55%); needs longer-history validation.',
      'Deeper drawdown than the index (Calmar ≈ 1) — a market-regime overlay (de-risk below the 100/200-DMA) is the highest-value upgrade; the regime-gated variant is the investable form.',
      'Integrity note: the often-quoted ~32% CAGR uses EQ95 (95% equity + 20% debt = 115%), inflating the engine CAGR by ~6pp vs the actual path. This factsheet uses the clean 80/20 (=100%) so path-CAGR equals engine-CAGR. Standardise on 80/20.',
      'Concentration risk — 30 names; single-name and sector caps are the only diversification.',
      'Backtest, net of modelled costs. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'MQ_PORTFOLIO_FACTSHEET.md (this report)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/02_mq_portfolio_optimization/reports/MQ_PORTFOLIO_FACTSHEET.md',
      },
      {
        label: 'services/mq_backtest_engine.py (engine)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/services/mq_backtest_engine.py',
      },
    ],
    projectPaths: [
      'research\\02_mq_portfolio_optimization\\reports\\',
      'research\\02_mq_portfolio_optimization\\scripts\\make_mq_report.py',
      'services\\mq_backtest_engine.py, services\\mq_portfolio.py',
    ],
  },

  {
    slug: 'momentum30-subselect',
    title: 'Momentum-30 ETF Sub-Selection (reconstructed Nifty 200 Momentum 30)',
    verdict:
      'Instead of running our own factor model, piggyback a published momentum index: reconstruct the Nifty 200 Momentum 30 from methodology (no factsheets), then hold a concentrated, gated, Donchian-trailed sub-basket of it. Top-8 + a NIFTYBEES-100DMA regime gate + a per-stock 15-day Donchian trailing exit compounds at 33.4% gross / 29.0% post-tax CAGR at just −17.0% drawdown (Sharpe 1.78, net-Calmar ~1.5–1.7) vs NIFTYBEES 12.3% / −36.3%. STRATEGY candidate (G1→G3 PASS) — beats the research/41 midcap book. Key structural finding: the macro gate and the per-stock Donchian are COMPLEMENTARY, not substitutes.',
    status: 'COMPLETE',
    date: '2026-06-11',
    cardBlurb:
      'Reconstruct the Nifty 200 Momentum 30 from its published methodology (survivorship-free PIT top-200 by traded value → 6m/12m score → top-30), then hold the strongest 8 equal-weight with a buffer, a 100DMA market-regime cash gate, and a 15-day Donchian per-stock trailing stop. Monthly. Net of cost and tax.',
    cardStats: [
      { label: 'CAGR (gross)', value: '33.4%' },
      { label: 'CAGR (post-tax 20%)', value: '29.0%' },
      { label: 'MaxDD', value: '−17.0%' },
    ],
    system: {
      intro: 'Long-only concentrated momentum sub-basket of a reconstructed factor index; the traded rules:',
      rows: [
        { k: 'Universe', v: 'Survivorship-free PIT top-200 by trailing-6-month median (close × volume) — a faithful Nifty-200 proxy rebuilt monthly from ~1,623 NSE daily symbols (not index membership).' },
        { k: 'Factor score', v: 'Reconstructed Momentum-30: rank by 6-month & 12-month relative strength; the top-30 = the "ETF". (The authentic risk-adjusted z-score was tested and is NOT better once drawdown is controlled.)' },
        { k: 'Hold', v: 'Top 8 of the 30, equal-weight, 100% invested.' },
        { k: 'Buffer', v: 'Retain a name while it stays inside the top-22 of the 30 (low churn). Buffer size 18/22/26 is immaterial.' },
        { k: 'Macro gate', v: 'NIFTYBEES vs its 100-day SMA, checked weekly → risk-off liquidates the book to cash.' },
        { k: 'Per-stock exit', v: '15-day Donchian: exit a name on a close below its prior-15-day low; redeploy at the next month-end.' },
        { k: 'Rotation', v: 'Monthly; daily-marked NAV for honest drawdown.' },
        { k: 'Costs / tax', v: '0.4% round-trip on turnover (large-cap reality ~10–20 bps, so conservative); post-tax = 20% STCG on lots < 365 days.' },
        { k: 'Backtest window', v: '2014→2026 (~12.4y, incl. 2018/2020/2022/2025 stress + the 2019 momentum dead-year).' },
      ],
    },
    conditions: {
      intro: 'Backtest window and benchmark.',
      rows: [
        { k: 'Period', v: 'Jan 2014 – May 2026 (~12.4 years).' },
        { k: 'Benchmark', v: 'NIFTY-50 (NIFTYBEES), same window, excluded from the investable universe.' },
        { k: 'Host', v: 'VPS market_data.db snapshot 2026-06-10; reproducible from committed scripts.' },
      ],
    },
    comparisons: [
      {
        title: 'Annual return: strategy vs NIFTY 50',
        columns: ['Year', 'Strategy %', 'NIFTYBEES %', 'Excess pp'],
        rows: [
          ['2014', '+117.6', '+31.6', '+86.0'],
          ['2015', '−2.4', '−4.3', '+1.9'],
          ['2016', '+45.6', '+4.0', '+41.6'],
          ['2017', '+48.2', '+29.9', '+18.3'],
          ['2018', '−0.4', '+4.8', '−5.2'],
          ['2019', '−4.2', '+13.6', '−17.8'],
          ['2020', '+59.2', '+15.4', '+43.8'],
          ['2021', '+88.9', '+26.0', '+62.9'],
          ['2022', '+14.0', '+5.5', '+8.5'],
          ['2023', '+50.5', '+21.0', '+29.5'],
          ['2024', '+44.7', '+10.4', '+34.3'],
          ['2025', '+15.6', '+11.7', '+3.9'],
          ['2026*', '−6.8', '−9.5', '+2.7'],
        ],
        highlightRows: [5],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR (gross)', value: '33.4%', tone: 'pos' },
        { label: 'CAGR (post-tax 20%)', value: '29.0%', tone: 'pos' },
        { label: 'NIFTYBEES CAGR', value: '12.3%' },
        { label: 'Excess / yr', value: '+21.1%', tone: 'pos' },
        { label: 'Sharpe', value: '1.78', tone: 'pos' },
        { label: 'Max Drawdown', value: '−17.0%', tone: 'neg', hint: 'vs NIFTYBEES −36.3%' },
        { label: 'Calmar (gross)', value: '1.97', tone: 'pos' },
        { label: 'Yrs beating index', value: '85%' },
      ],
      tables: [
        {
          title: 'Strategy vs benchmark',
          columns: ['Metric', 'Momentum-30 Sub', 'NIFTYBEES'],
          rows: [
            ['CAGR', '33.4%', '12.3%'],
            ['Total return', '35.2x', '4.2x'],
            ['Sharpe', '1.78', '0.88'],
            ['Max Drawdown', '−17.0%', '−36.3%'],
            ['Calmar', '1.97', '0.34'],
          ],
          highlightRows: [0, 1, 2, 3],
        },
        {
          title: 'Why gate + Donchian (both needed) — MaxDD by risk layer',
          columns: ['Risk layer', 'CAGR', 'MaxDD', 'net-Calmar'],
          rows: [
            ['No gate, no Donchian (base)', '25.4%', '−44.6%', '0.57'],
            ['Donchian-15 only', '~25%', '~−32%', '0.77'],
            ['Gate only', '25.4%', '−28.8%', '0.88'],
            ['Gate + Donchian-15 (winner)', '33.4%', '−17.0%', '~1.7'],
          ],
          highlightRows: [3],
        },
        {
          title: 'Universe-transfer test — does the edge transfer to mid/small-caps? (net 20bps, 2014–26)',
          caption: 'Same r62 method (top-8 by RS + 100DMA gate + Donchian-15) on different universes. It does NOT transfer: on mid/small-caps the return survives (~34%) but the drawdown blows out and Calmar collapses — the low-DD edge was specific to the Nifty-200 pool. For comparison, research/41 on mid/small-caps uses de-risk-TO-CASH (not per-stock trailing) and holds Calmar 1.44.',
          columns: ['Universe / method', 'CAGR', 'net 20%', 'MaxDD', 'Sharpe', 'Calmar'],
          rows: [
            ['Nifty-200 (r62 method) — original', '34.5%', '30.0%', '−15.6%', '1.83', '2.22'],
            ['MidSmallcap-400 (r62 method)', '34.0%', '28.9%', '−39.2%', '1.81', '0.87'],
            ['mid 100–250 (r62 method)', '28.3%', '24.0%', '−34.5%', '1.59', '0.82'],
            ['MidSmallcap (research/41 own method)', '35.3%', '28.9%', '−24.6%', '1.53', '1.44'],
            ['NIFTYBEES buy & hold', '12.4%', '—', '−36.3%', '0.89', '0.34'],
          ],
          highlightRows: [0, 1],
        },
      ],
      charts: [
        {
          src: '/app/momentum30-subselect-factsheet.png',
          caption:
            'CLIENT FACTSHEET — Momentum-30 Sub-Selection (Top-8 + 100DMA gate + Donchian-15) vs NIFTY 50, 2014–2026, net of 0.4% cost. KPI strip, growth-of-₹1 (log), drawdown-vs-index, annual bars, monthly heatmap, rolling 12m, stat tables. 33.4% CAGR (29.0% post-tax) vs 12.3%, 35.2x vs 4.2x, Sharpe 1.78, MaxDD −17.0% vs −36.3%, 85% of years beating the index. Generated by research/_utilities/tearsheet.py.',
        },
      ],
    },
    winners: [
      {
        config: 'rsblend · N8 · buffer-22 · gate-100 · Donchian-15',
        summary: 'Best risk-adjusted of a 288-cell sweep; the gate and the per-stock Donchian are complementary — gate alone −28.8% DD, Donchian alone ~−32%, together −17.0%.',
        metrics: [
          { k: 'CAGR', v: '33.4% gross / 29.0% net' },
          { k: 'Excess', v: '+21.1%/yr vs NIFTYBEES' },
          { k: 'Sharpe', v: '1.78' },
          { k: 'MaxDD', v: '−17.0%' },
          { k: 'net-Calmar', v: '~1.5–1.7' },
        ],
        rejected: [
          'Dropping the gate (the original idea): no-gate book draws down −44.6% — Donchian helps but does not replace the gate.',
          'Donchian-20 / -50: looser trails give worse DD and far weaker super-winner robustness; 15 wins.',
          'The authentic risk-adjusted Momentum-30 z-score: same DD but ~8pp less CAGR than plain relative strength once DD-controlled.',
        ],
      },
    ],
    caveats: [
      "IMMEDIATE-REDEPLOY tested + REJECTED (2026-07-08): the 15-day Donchian per-stock exit is a DAILY check, and when a name exits mid-month the freed cash sits in cash until the next month-end refill (it does NOT instantly buy the next-best momentum name). Testing immediate redeploy is far WORSE — MaxDD more than doubles (−15.6% → −37.9% gross / −46.7% net, worse than buy-and-hold), net-tax CAGR collapses 30.2% → 18.6%, net Calmar 1.71 → 0.40, and Donchian churn 2.3×'s (452 → 1,052 exits). The exit-to-cash gap is a DEFENSIVE FEATURE: in a broad selloff many names break down at once, the book drifts to cash and de-risks; refilling immediately keeps you fully invested straight into the crash. Keep the month-end refill.",

      '2019 is the one genuine weak year (−4.2% vs index +13.6%) — the narrow Indian momentum dead-year; the gate kept it roughly flat but it missed the large-cap melt-up.',
      'Multiple testing: 288 configs were searched; the winner sits on a stable plateau (N8 / any buffer / Donch-15 / gate-100) and survives cost-stress to 60 bps and a super-winner guard (Calmar holds 1.79 without its 3 best names), but the headline figure should carry a multiple-testing haircut — treat 29% net as the optimistic end.',
      'Reconstruction is a faithful PROXY of the index, not the live NSE product (which uses risk-adjusted scores, free-float caps, semi-annual reconstitution). Validation against ~3 real factsheet dates is still owed before live capital.',
      'Concentration/correlation (G4 pending): the 8 names currently lean PSU/defence/renewable; cluster-stress drawdown is not yet measured and could exceed the −17% backtest figure on a thematic unwind.',
      'Backtest, net of modelled costs (and post-tax where stated). Nothing wired live. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'RESULTS.md (verdict + tables)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/62_momentum_etf_subselect/results/RESULTS.md',
      },
      {
        label: '62_mom30_subselect.py (engine)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/62_momentum_etf_subselect/scripts/62_mom30_subselect.py',
      },
    ],
    projectPaths: [
      'research\\62_momentum_etf_subselect\\MOMENTUM30_ETF_SUBSELECT_DAILY_SWEEP_STATUS.md',
      'research\\62_momentum_etf_subselect\\scripts\\62_mom30_subselect.py, 62b_g2_sweep.py',
      'research\\62_momentum_etf_subselect\\results\\ (g2_sweep.csv, RESULTS.md, tearsheet.png)',
    ],
  },
  {
    slug: 'breakout-mtf-volume-swing',
    title: 'Breakout Swing Book — MTF-Bullish Volume-Breakout Exit Bake-Off',
    verdict:
      'Automate a Chartink-style multi-timeframe-bullish volume-breakout scan (monthly+weekly+daily MACD>0, at/near 52-week high, volume >= 2x, liquid), then answer the real question: which EXIT is best for short-term trading? Across 20,804 tradeable breakouts, TRAILING stops beat fixed targets and fast/time exits decisively (Donchian-20 / Supertrend 10,3 ~ +4.4% net/trade, PF ~1.8) — and a profit target HURTS (it caps the fat right tail that is the entire edge). As a book (Donchian-20 trail, NO target, NIFTY>200DMA regime gate, 8 concurrent, max 1 new entry/day) it compounds at 19.9% CAGR at just -29.1% drawdown (Calmar 0.68, Sharpe 0.67) vs NIFTYBEES 11.6% / -59.7% — 40.9x vs 9.4x over 20.5 years. STRATEGY candidate (G1->G5 PASS). The regime gate is the single biggest drawdown reducer; the 1/day cap pushes MaxDD under 30%.',
    status: 'COMPLETE',
    date: '2026-07-01',
    cardBlurb:
      'Reconstruct the multi-timeframe-bullish volume-breakout scan, then bake off every exit family (fixed SL+target, ATR-chandelier, Supertrend, Donchian, EMA, time-based, hybrids) on 20,804 liquid breakouts. Winner: a trailing stop with NO target + a market-regime gate + a 1-new-entry-per-day cap. Net of cost.',
    cardStats: [
      { label: 'CAGR', value: '19.9%' },
      { label: 'MaxDD', value: '-29.1%' },
      { label: 'Calmar', value: '0.68' },
    ],
    system: {
      intro: 'Long-only short-term breakout swing book; the traded rules:',
      rows: [
        { k: 'Entry signal', v: 'MACD line(12,26) > 0 on DAILY, WEEKLY and MONTHLY (the Chartink MTF-bullish filter) AND close >= 98% of the 252-day high (at/near 52-week high) AND volume >= 2x its 20-day average (the volume breakout).' },
        { k: 'Liquidity filter', v: '20-day MEDIAN turnover (close x volume) >= Rs.5cr, price >= Rs.20. Median NOT mean — a mean turnover filter lets one spike day sneak illiquid/circuit names through.' },
        { k: 'Entry-fill guard', v: 'Skip if the next-open bar is circuit-locked ((high-low)/open < 1%) or gaps > 15% above the signal close (unfillable chase). Fill at the next-day open.' },
        { k: 'Exit (winner)', v: 'Donchian-20 lower-channel trailing stop: exit on a close below the prior-20-day low. Supertrend(10,3) is statistically identical. NO profit target. Fill at next open.' },
        { k: 'Catastrophe stop', v: 'Hard 20% stop floor beneath every position (gap protection).' },
        { k: 'Regime gate', v: 'Take NEW entries only when NIFTYBEES is above its 200-day SMA. This is the single biggest drawdown reducer.' },
        { k: 'Concurrency + cadence', v: '8 positions equal-weight; at most 1 NEW entry per day (when > slots free, rank the day qualifiers by today percent-run and take the top one). This caps trade frequency to ~0.55/week and pushes MaxDD under 30%.' },
        { k: 'Costs', v: '0.20% round-trip; not cost-fragile (long ~7-week holds = low turnover). Gross of tax (~45-day holds = 20% STCG -> net CAGR ~16-17%).' },
        { k: 'Window', v: '2006-2026 (~20.5y), compounding, daily mark-to-market for honest drawdown.' },
      ],
    },
    conditions: {
      intro: 'Backtest window, universe and benchmark.',
      rows: [
        { k: 'Period', v: 'Jan 2006 - Jun 2026 (~20.5 years; 200-DMA warmup from 2005).' },
        { k: 'Universe', v: 'All NSE daily symbols passing the liquidity filter (~1,642 in DB). Skews to today listed names (survivorship caveat).' },
        { k: 'Benchmark', v: 'NIFTY 50 (NIFTYBEES), 2005-2026 long history, excluded from the investable set and used for the regime gate.' },
        { k: 'Host', v: 'VPS market_data.db snapshot ~2026-05-15; reproducible from committed scripts.' },
      ],
    },
    comparisons: [
      {
        title: 'Annual return: strategy vs NIFTY 50',
        columns: ['Year', 'Strategy %', 'NIFTYBEES %', 'Excess pp'],
        rows: [
          ['2007', '+78', '+53', '+25'],
          ['2008', '-25', '-52', '+27'],
          ['2009', '+36', '+76', '-40'],
          ['2010', '+27', '+19', '+8'],
          ['2011', '-6', '-24', '+18'],
          ['2012', '+23', '+27', '-4'],
          ['2013', '+0', '+7', '-7'],
          ['2014', '+66', '+32', '+34'],
          ['2015', '-10', '-4', '-6'],
          ['2016', '+31', '+4', '+27'],
          ['2017', '+34', '+30', '+4'],
          ['2018', '-15', '+5', '-20'],
          ['2019', '+10', '+14', '-4'],
          ['2020', '+32', '+15', '+17'],
          ['2021', '+81', '+26', '+55'],
          ['2022', '+1', '+5', '-4'],
          ['2023', '+70', '+21', '+49'],
          ['2024', '+40', '+10', '+30'],
          ['2025', '+1', '+12', '-11'],
          ['2026*', '-16', '-8', '-8'],
        ],
        highlightRows: [14, 16, 17],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR', value: '19.9%', tone: 'pos' },
        { label: 'NIFTYBEES CAGR', value: '11.6%' },
        { label: 'Excess / yr', value: '+8.3%', tone: 'pos' },
        { label: 'Sharpe', value: '0.67', tone: 'pos', hint: 'vs index 0.34' },
        { label: 'Sortino', value: '0.84' },
        { label: 'Max Drawdown', value: '-29.1%', tone: 'neg', hint: 'vs NIFTYBEES -59.7%' },
        { label: 'Calmar', value: '0.68', tone: 'pos' },
        { label: 'Total return', value: '40.9x', hint: 'vs index 9.4x' },
      ],
      tables: [
        {
          title: 'The exit bake-off — net per-trade return on 20,804 tradeable breakouts (net 0.20%)',
          caption: 'The core finding: trailing stops beat targets and fast exits. A profit target HURTS. Fast/tight exits earn nothing.',
          columns: ['Exit rule', 'Net / trade', 'Win %', 'Profit factor', 'Avg hold'],
          rows: [
            ['Supertrend (7,3) trail', '+4.43%', '42%', '1.85', '44d'],
            ['Donchian-20 trail', '+4.38%', '42%', '1.76', '49d'],
            ['EMA-50 trail', '+4.28%', '40%', '1.83', '40d'],
            ['Supertrend (10,3) trail', '+4.27%', '42%', '1.80', '43d'],
            ['Chandelier 4xATR trail', '+4.12%', '43%', '1.75', '45d'],
            ['Hold exactly 40 days', '+3.30%', '54%', '1.62', '37d'],
            ['5% SL + 10% target (tight)', '+0.02%', '35%', '1.00', '9d'],
          ],
          highlightRows: [1, 3],
        },
        {
          title: 'Why the gate + the 1/day cap — portfolio equity curves',
          caption: 'The regime gate roughly halves drawdown AND raises CAGR. The 1/day entry cap pushes MaxDD under 30%.',
          columns: ['Book config', 'CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Donchian + gate + 8 + 1/day cap (WINNER)', '19.9%', '-29.1%', '0.68'],
            ['Donchian + gate + 8 (no cap)', '20.5%', '-34.4%', '0.60'],
            ['Supertrend(10,3) + gate + 10', '20.1%', '-39.9%', '0.50'],
            ['Donchian + NO gate + 10', '12.9%', '-64.0%', '0.20'],
            ['Donchian + NO gate + 5', '11.6%', '-70.5%', '0.16'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Strategy vs benchmark',
          columns: ['Metric', 'Breakout Swing Book', 'NIFTYBEES'],
          rows: [
            ['CAGR', '19.9%', '11.6%'],
            ['Total return', '40.9x', '9.4x'],
            ['Sharpe', '0.67', '0.34'],
            ['Max Drawdown', '-29.1%', '-59.7%'],
            ['Calmar', '0.68', '0.19'],
            ['Beta / Correlation', '0.46 / 0.42', '1.00'],
          ],
          highlightRows: [0, 1, 2, 3, 4],
        },
      ],
      charts: [
        {
          src: '/app/breakout-swing-factsheet.png',
          caption:
            'CLIENT FACTSHEET — Breakout Swing Book (Donchian-20 trail + NIFTY>200DMA gate + 8 concurrent + 1 entry/day cap) vs NIFTY 50, 2006-2026, net of 0.20% cost. 19.9% CAGR vs 11.6%, 40.9x vs 9.4x, Sharpe 0.67, MaxDD -29.1% vs -59.7%, Calmar 0.68. KPI strip, growth-of-Rs.1 (log), drawdown-vs-index, annual bars, monthly heatmap, rolling 12m. Generated by research/_utilities/tearsheet.py.',
        },
      ],
    },
    winners: [
      {
        config: 'Donchian-20 trail (no target) + NIFTY>200DMA gate + 8 concurrent + max 1 entry/day',
        summary: 'Best risk-adjusted of the exit x gate x concurrency x fine-filter sweep. Trailing beats target beats tight stop; the gate halves drawdown; the 1/day cap smooths entries and drops MaxDD under 30%. Cadence is light — ~0.55 trades/week, an entry on ~11% of days.',
        metrics: [
          { k: 'CAGR', v: '19.9% gross (~16-17% post-tax)' },
          { k: 'Excess', v: '+8.3%/yr vs NIFTYBEES' },
          { k: 'MaxDD', v: '-29.1% (index -59.7%)' },
          { k: 'Calmar', v: '0.68' },
          { k: 'Per-trade edge', v: '+4.4% net, PF 1.8, ~7-week hold' },
        ],
        rejected: [
          'ANY profit target: every fixed-target config underperformed its no-target sibling (8%SL+15%tgt +0.9% vs 8%SL-no-target +4.9%). A target caps the fat tail that is the edge.',
          'Tight/fast exits: 5%SL+10%target and 5-day holds earn ~0 net. The breakout drift is slow; you must ride winners for weeks.',
          'Dropping the regime gate: no-gate books draw down -48% to -70% and earn less CAGR.',
          'An over-extension filter to pre-avoid parabolic chases: REJECTED by the data — IRFC broke out +63% above its 50-SMA and ran +119%. Extension filters throw away the biggest winners; the trailing stop handles reversals instead.',
          'Ranking entries by today biggest percent-run: NOT additive (top-run names mildly mean-revert). Use run as a filter, not the picker.',
        ],
      },
    ],
    caveats: [
      'Survivorship bias (biggest): the universe = symbols in today DB, so historical breakouts on names that later delisted/died are absent. Real returns are lower and real drawdown deeper than shown — treat 19.9% CAGR / -29% DD as the optimistic end.',
      'Thin early years: few Rs.5cr-median-turnover names existed 2006-2010, so early-year returns (2007 +78%) sit on a small book; the 2015+ era is more credible.',
      'Gross of tax: ~45-day holds are short-term -> 20% STCG -> net CAGR ~16-17%.',
      'Concentrated long-beta: beta 0.46, correlation 0.42; all breakouts are one bet, so the -29% DD is correlated-cluster risk that a thematic unwind could exceed.',
      'Many of the exact microcaps the user trades (PAISALO, SMSPHARMA, VELJAN, ...) have 0 rows in the DB — this is a PROXY population on a mid/large-cap-skewed liquid universe, not those specific names.',
      'Backtest, net of 0.20% modelled cost, gross of tax except where stated. Nothing wired live. Next: better selection ranker, point-in-time universe, and a G5 paper soak before any capital. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'RESULTS.md (verdict + tables)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/71_breakout_exit_bakeoff/results/RESULTS.md',
      },
      {
        label: 'g4_portfolio.py (portfolio engine)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/71_breakout_exit_bakeoff/scripts/g4_portfolio.py',
      },
    ],
    projectPaths: [
      'research\\71_breakout_exit_bakeoff\\BREAKOUT_MTF_VOLUME_DAILY_SWEEP_STATUS.md',
      'research\\71_breakout_exit_bakeoff\\scripts\\ (g1_probe, g2_exit_bakeoff, g3_clean_bakeoff, g4_portfolio, g5_finefilter).py',
      'research\\71_breakout_exit_bakeoff\\results\\ (RESULTS.md, tearsheet.png, g*_*.csv)',
    ],
  },
  {
    slug: 'weekly-supertrend-nifty200',
    title: 'Trend-Timing NIFTYBEES with SuperTrend — the one clean winner (and the SuperTrend myths it debunks)',
    verdict:
      '★★ THE WINNER: hold NIFTYBEES when its DAILY SuperTrend(7,3) is GREEN; on RED sell and park in a LIQUID fund (~4.5% net after expense+slab tax), re-enter on the next green. REALISTIC — net of 0.30% cost, the liquid-fund net return AND T+1 settlement lag (re-enter 1 day late, proceeds sit 1 day in transit): it CUTS MAX DRAWDOWN FROM −36% TO −14% (Calmar 0.29→0.65, volatility 15%→10%) for a ~1.3pp CAGR give-up (10.6→9.3%); net of ETF STCG/LTCG ≈ 7.8% CAGR (~2.8pp give-up net of EVERYTHING). Honest framing: this is a DRAWDOWN-REDUCTION overlay — roughly SHARPE-NEUTRAL, NOT a return-enhancer — you trade ~1.3–2.8pp CAGR to halve your worst crash. The drawdown-halving is friction-proof. BEST BUILD (MODELED): keep the ETF and SHORT NIFTY futures on the red signal instead of selling — no ETF sale → no equity CGT (deferred like B&H), no settlement lag, margin funded by pledging the ETF; while hedged you earn the carry (≈ risk-free). This RECOVERS most of the give-up. VALIDATED on REAL NSE bhavcopy basis (196 pts across COVID/2022/2018): the real carry is ~+3% (not the +4.6% first modeled), and futures DO go to backwardation in crashes (COVID 52% of days) — already baked into that +3%. With the real carry the hedge does ~9.9% CAGR (−0.6pp vs B&H) at −15% drawdown (halved), Calmar 0.67, Sharpe 1.03 (a genuine Sharpe improvement, unlike the cash version). One liquid instrument, no survivorship, infinitely scalable. ST(7,3) ties 50/100-DMA; 200-DMA too slow. Everything below is HOW WE GOT HERE — including two myths we debunked. ——— A YouTube guest pitched a system built entirely on the weekly SuperTrend (10,3): buy green, exit blind on red, size 5–7%/name, book 40/40/20, +5 hacks. First pass looked like a critical finding — on Nifty 200 it did 17.5% CAGR / −31.7% MaxDD / Calmar 0.55, "+6.9pp over NIFTYBEES". CORRECTION (same day): that headline was a BENCHMARK ARTIFACT. The book trades TODAY’s Nifty 200 names (survivorship-selected — includes the RVNL/KEI-type names that are in the index BECAUSE they 10–50×’d) and was compared to the Nifty 50 index. Against the fair, survivorship-MATCHED benchmark — equal-weight buy-and-hold of the SAME 200 names — the SuperTrend timing LOSES by ~3.5pp/yr (17.5% vs 21.0% CAGR) at essentially identical Calmar (0.55 vs 0.56–0.58). Even the most conservative fixed-capital drift basket (buy-what-you-could-in-2010, hold, least survivorship) does 20.4% / Calmar 0.56. So the timing adds no return and no risk-adjusted edge over simply owning the basket; it only trims MaxDD a hair (−31.7 vs −36.3), paid for with ~3pp/yr. VERDICT: NO INVESTABLE TIMING EDGE — the attractive number was survivorship + breadth, not the signal. The per-trade entry timing has a small real edge (G1 +5.2pp vs random-hold) but it is swamped at the portfolio level by being out of the market through a secular bull. SIGNAL ≠ STRATEGY. Also proven along the way: the guest’s own 40/40/20 booking and a regime gate both HURT. ★ PHASE 2 (the redemption): where the SuperTrend DOES work is as a MARKET-LEVEL CRASH OVERLAY — hold the basket always and use a DAILY ST(7,3) on the index to flatten the whole book in downtrends. That more than DOUBLES pre-tax Calmar (0.56→1.28) by cutting Nifty200 drawdown from −39% to −15% for ~2pp CAGR, consistent across all bands and the whole fast-filter family (only the 200-DMA fails). Catch: liquidating the cash basket ~2.5×/yr realises tax (net Calmar 1.01), so the right build is a NIFTY-futures/puts hedge (no sale → no tax event). Bonus lever TESTED then REJECTED (2026-07-08): swapping the live momentum book’s 100-DMA gate for a daily-ST gate is WORSE (net Calmar 1.71→1.33) — the ST gate is twitchier (de-risks 30–36× vs 23), gives up ~6pp CAGR for no DD benefit. The earlier "ST beats the gate" was vs a too-slow 200-DMA; a well-tuned 100-DMA wins. KEEP the live gate. ★★ PHASE 3 — the cleanest, most tradeable version: apply the INDEX-LEVEL trend filter to the actual index ETF (NIFTYBEES) itself — no survivorship, one liquid instrument, infinite capacity. REALISTIC (net of cost + a liquid fund at ~4.5% net after its expense/slab tax + T+1 settlement lag): it CUTS DRAWDOWN BY MORE THAN HALF (−36%→−14%, Calmar 0.29→0.65) for a ~1.3pp CAGR give-up (10.6→9.3%); net of ETF STCG/LTCG ≈ 7.8% (~2.8pp give-up). Roughly SHARPE-NEUTRAL — a drawdown-reduction overlay, not a return-enhancer. (An idealized instant-switch, 6.5%-cash run flattered it to ~zero give-up; the honest, friction-adjusted number is ~1.3–2.8pp.) On a single ETF ST(7,3) is marginally best (fewest switches → least tax) but 50/100-DMA are tied — it is "any fast-medium trend filter," not ST-specific; the 200-DMA is too slow and HURTS (halves CAGR). A well-known tactical-timing result (Faber-style), confirmed on Indian ETFs. FOOTNOTE (Phase 2 vs 3): both use the SAME index ST(7,3) doing the SAME job (cut DD to ~−14/−15%); Phase 2 shows a higher Calmar (1.28) ONLY because it times the survivorship-inflated 200-stock basket (21.8% CAGR) — a mirage you can’t capture. Phase 3 times the real ETF (10.6% CAGR) → the honest, tradeable Calmar. Same timing, different (real vs inflated) underlying.',
    status: 'COMPLETE',
    date: '2026-07-07',
    cardBlurb:
      'The one clean winner: trend-time the NIFTYBEES ETF itself with daily SuperTrend(7,3) — realistic (net of cost + liquid-fund tax + T+1 settlement) it HALVES max drawdown (−36%→−14%, Calmar 0.29→0.65) for a ~1.3pp CAGR give-up (~2.8pp net of all tax). A drawdown-reduction overlay, roughly Sharpe-neutral; best as a futures hedge. Along the way we debunked two myths: per-stock ST timing LOSES to buy-and-hold (the "beats index" headline was survivorship), and the guest’s 40/40/20 + regime gate both HURT.',
    cardStats: [
      { label: 'NIFTYBEES + ST(7,3) MaxDD', value: '−14% vs −36%' },
      { label: 'Calmar', value: '0.65 vs 0.29' },
      { label: 'CAGR give-up', value: '~1.3pp (2.8 net-tax)' },
    ],
    systemRules: {
      intro: 'The mechanical core is what we tested; the 5 "hacks" + 40/40/20 + 50-EMA band were layered on top (and the tested ones failed to help — see winners/caveats).',
      sharedCoreTitle: 'Core system (all bands)',
      sharedCore: [
        { k: 'Signal', v: 'Weekly (W-FRI resampled) SuperTrend, ATR 10 / multiplier 3 — TradingView-accurate (services/technical_indicators.calc_supertrend). Long-only.' },
        { k: 'Entry', v: 'SuperTrend flips UP (green) at the Friday weekly close → fill at next-week open.' },
        { k: 'Exit (core)', v: 'SuperTrend flips DOWN (red) → fill at next-week open. Blind, no discretion.' },
        { k: 'Causality', v: 'Signal read at weekly close of week t → trade fills at OPEN of week t+1. No look-ahead. ("Wait for Friday close" is automatic on weekly bars.)' },
        { k: 'Sizing', v: 'Max 16 concurrent names at 6.25% target weight; oversubscribed → take strongest flip-week gain (king-candle proxy). Idle cash @6.5% (liquid fund).' },
        { k: 'Costs/tax', v: '0.15%/side (0.30% round-trip) + STCG 15% (<1y) / LTCG 10% (>1y) on realised gains.' },
      ],
      riskLayer: {
        title: 'Variant sweep on the Nifty 200 core (net of cost+tax, 2010–2026) — the guest’s add-ons both HURT',
        columns: ['Config', 'CAGR', 'MaxDD', 'Calmar', 'Sharpe', 'Verdict'],
        rows: [
          ['Core — blind ST exit, no gate, no booking', '17.5%', '−31.7%', '0.55', '1.03', 'best ST cell'],
          ['+ NIFTYBEES 200-DMA regime gate', '11.0%', '−32.1%', '0.34', '0.73', 'HURTS (redundant)'],
          ['+ gate + 40/40/20 profit-booking', '8.8%', '−31.6%', '0.28', '0.73', 'HURTS (caps tail)'],
          ['Benchmark: NIFTYBEES buy & hold', '10.6%', '−34.0%', '0.31', '—', '(wrong benchmark — see below)'],
        ],
        highlightRows: [0],
        heatmap: false,
      },
    },
    system: {
      intro: 'From a YouTube interview (Vijay Khant): a whole system on ONE indicator, the weekly SuperTrend (10,3). We tested the mechanical core faithfully, then each "hack", then — crucially — checked the benchmark.',
      rows: [
        { k: 'Source claim', v: 'Enter when weekly ST turns green, exit blindly when red; size 5–7%/name (stop ≤1–2% of capital); book 40% at +40%, another 40% at a further +40%, trail last 20% on ST.' },
        { k: '5 hacks', v: 'King candle, Friday-close confirmation, breakout-high entry, 18–20wk pre-consolidation, Dow higher-highs/lows. Plus a 50-EMA (H/C/L) band for re-entry.' },
        { k: 'What we automated', v: 'Core signal + sizing + 40/40/20 + king-candle selection + regime gate. Friday-close = automatic on weekly bars. Dow-structure = too subjective to encode (noted, not coded).' },
        { k: 'Universe', v: 'Current official Nifty 50 (50) / Nifty 200 (200) / Midcap 150 (150) / Smallcap 250 (250) — SURVIVORSHIP-BIASED (today’s members applied to the past).' },
        { k: 'The benchmark trap', v: 'Comparing this survivorship-selected 200-name book to the NIFTY 50 index gives two free edges (survivorship + Nifty200-breadth) before ST does anything. The fair benchmark is buy-and-hold of the SAME 200 names.' },
      ],
    },
    conditions: {
      intro: 'Window and data.',
      rows: [
        { k: 'Period', v: '2010-01-01 → 2026-07-07 (16.5y). Daily bars resampled to weekly; breadth grows 283 (2008) → 1632 (2025) names.' },
        { k: 'Data', v: 'VPS market_data.db (day tf, snapshot max 2026-07-07); NIFTYBEES full 2005–2026 history.' },
        { k: 'Host', v: 'VPS venv (numpy 2.4.4 / pandas 3.0.2); reproducible from committed scripts.' },
      ],
    },
    comparisons: [
      {
        title: '★ MASTER COMPARISON — every version, returns & drawdowns (2010–2026, net of realistic cost)',
        caption: 'The whole study in one table. Timing individual STOCKS loses to owning them; the pitched system’s 17.5% only beat the WRONG benchmark. The SAME signal at the INDEX level on the ETF you’d hold anyway takes the −36% drawdown to −15% for ~0.6pp CAGR (best as a futures hedge). Tax basis noted per row.',
        columns: ['Version', 'CAGR', 'MaxDD', 'Calmar', 'Sharpe', 'Verdict'],
        rows: [
          ['NIFTYBEES buy & hold (baseline)', '10.5%', '−36.3%', '0.29', '0.75', 'the thing to beat'],
          ['ST(7,3) · futures-hedge · REAL data', '9.9%', '−14.8%', '0.67', '1.03', '✅ THE WINNER'],
          ['ST(7,3) · cash-rotation · net ALL tax', '7.8%', '−14.3%', '0.46', '~0.9', 'tax-inefficient build'],
          ['ST(7,3) · idealized (too-kind)', '10.7%', '−14.2%', '0.76', '1.11', '⚠ flattering assumptions'],
          ['100-DMA on the ETF (net cost)', '10.3%', '−14.2%', '0.72', '—', '≈ ties ST'],
          ['200-DMA on the ETF (net cost)', '5.5%', '−20.7%', '0.27', '—', '❌ too slow'],
          ['Per-stock ST · Nifty 200 (pitched)', '17.5%', '−31.7%', '0.55', '—', '❌ illusion (wrong bench)'],
          ['…its own same-basket buy & hold', '21.0%', '−35.9%', '0.58', '—', 'the benchmark it loses to'],
          ['Per-stock ST + 40/40/20 booking', '8.8%', '−31.6%', '0.28', '—', '❌ booking kills returns'],
          ['Phase-2 overlay on survivorship basket', '19.6%', '−15.3%', '1.28', '—', '⚠ un-real (inflated base)'],
        ],
        highlightRows: [1],
        heatmap: false,
      },
      {
        title: '★ THE DECIDING TEST — does the TIMING add anything? (fair, survivorship-matched, same 200 names, 2010–2026)',
        caption: 'The ST book vs simply HOLDING the identical basket. Against the fair benchmark the SuperTrend timing LOSES by ~3–3.5pp/yr at equal Calmar. The "+6.9pp over NIFTYBEES" headline was survivorship + Nifty200-breadth, not the signal.',
        columns: ['Book (same universe)', 'CAGR', 'MaxDD', 'Calmar', 'Sharpe', 'Total'],
        rows: [
          ['ST-core (flip in / flip out) — the "strategy"', '17.5%', '−31.7%', '0.55', '1.03', '14.3×'],
          ['EW buy-&-hold, same 200 names', '21.0%', '−35.9%', '0.58', '1.16', '23.2×'],
          ['B&H-drift, fixed-cap, start-names (LEAST survivorship)', '20.4%', '−36.3%', '0.56', '1.17', '21.2×'],
          ['NIFTYBEES (Nifty 50 index — the WRONG benchmark)', '10.6%', '−34.0%', '0.31', '0.74', '5.2×'],
        ],
        highlightRows: [1, 2],
        heatmap: false,
      },
      {
        title: 'ALL INDICES — same fair test (ST timing vs same-basket B&H vs Nifty 50), 2010–2026',
        caption: 'The result is uniform: on EVERY band the SuperTrend timing loses to simply holding the same basket (−2.8 to −6.6pp/yr). The "beats Nifty 50" column is the survivorship+breadth illusion — the basket itself beats Nifty 50 by +8 to +11pp before any timing. Only in Smallcap does timing help risk-adjusted (Calmar 0.48 vs 0.41) by taming the −54% basket DD — but that band is untradeable at size.',
        columns: ['Band', 'ST timing CAGR', 'Same-basket B&H', 'ST − Basket', 'ST − Nifty50', 'ST Calmar', 'Basket Calmar'],
        rows: [
          ['Nifty 50', '12.2%', '18.9%', '−6.6pp', '+1.6', '0.53', '0.57'],
          ['Nifty 200', '17.5%', '21.0%', '−3.5pp', '+6.9', '0.55', '0.58'],
          ['Midcap 150', '15.2%', '21.6%', '−6.4pp', '+4.6', '0.50', '0.61'],
          ['Smallcap 250', '19.3%', '22.0%', '−2.8pp', '+8.7', '0.48', '0.41'],
          ['Nifty 500 (=50+next50+mid150+small250)', '17.1%', '21.5%', '−4.4pp', '+6.6', '0.47', '0.52'],
        ],
        highlightRows: [1],
        heatmap: false,
      },
      {
        title: 'YEAR-BY-YEAR — Nifty 200: ST timing vs same-basket B&H vs Nifty 50 (annual %)',
        caption: 'ST beats the index in strong single-direction trend years (2014/2017/2020/2023) and loses badly in V-recoveries & chop when it sits in cash (2010 −23 vs basket, 2021 −36, 2012 −21). Net across the cycle: the basket wins. (Full per-band year tables: fair_allbands_peryear.csv.)',
        columns: ['Year', 'ST %', 'Basket %', 'Nifty50 %', 'ST − Nifty', 'ST − Basket'],
        rows: [
          ['2010', '4.9', '27.7', '17.7', '−12.7', '−22.8'],
          ['2011', '−22.6', '−18.5', '−21.5', '−1.0', '−4.1'],
          ['2012', '17.4', '38.4', '22.7', '−5.3', '−20.9'],
          ['2013', '20.7', '3.5', '5.5', '+15.2', '+17.2'],
          ['2014', '77.4', '61.5', '31.9', '+45.5', '+15.9'],
          ['2015', '10.7', '5.0', '−6.0', '+16.7', '+5.7'],
          ['2016', '5.6', '11.3', '3.7', '+1.8', '−5.7'],
          ['2017', '52.7', '44.3', '28.7', '+24.0', '+8.4'],
          ['2018', '−10.2', '−7.7', '4.5', '−14.7', '−2.6'],
          ['2019', '16.6', '10.5', '15.2', '+1.4', '+6.1'],
          ['2020', '45.9', '25.4', '13.0', '+33.0', '+20.5'],
          ['2021', '13.2', '49.5', '25.6', '−12.3', '−36.2'],
          ['2022', '3.7', '10.9', '2.9', '+0.7', '−7.3'],
          ['2023', '53.6', '51.1', '22.4', '+31.2', '+2.5'],
          ['2024', '21.2', '29.2', '11.1', '+10.1', '−8.1'],
          ['2025', '1.5', '8.3', '9.7', '−8.2', '−6.7'],
          ['2026*', '−0.0', '0.8', '−7.3', '+7.3', '−0.8'],
        ],
        heatmap: true,
      },
      {
        title: 'G1 signal probe — the per-trade ENTRY edge is small but real (vs random-duration placebo)',
        caption: 'ST entries beat random-duration entries of the SAME names — a genuine trade-level signal. But median trade ≈ 0 (edge is right-tail only), and at the PORTFOLIO level it is swamped by time out of the market. SIGNAL ≠ STRATEGY.',
        columns: ['Band', 'Trades', 'Win%', 'Mean net/trade', 'Median', 'Edge vs placebo'],
        rows: [
          ['Nifty 50', '555', '56%', '32.5%', '3.5%', '−1.8pp (NONE)'],
          ['Nifty 200', '1713', '55%', '44.0%', '3.5%', '+5.2pp'],
          ['Midcap 150', '1098', '54%', '47.5%', '2.3%', '+7.5pp'],
          ['Smallcap 250', '1373', '51%', '47.6%', '1.1%', '+12.2pp'],
        ],
        highlightRows: [1],
        heatmap: false,
      },
      {
        title: '★ PHASE 2 — where ST DOES work: a MARKET-LEVEL crash overlay on the Nifty 200 basket (2010–2026)',
        caption: 'Hold the basket always; use a DAILY index trend filter to flatten the whole book in downtrends. Unlike per-name timing (which loses), a fast market filter more than DOUBLES pre-tax Calmar by cutting drawdown for ~2pp CAGR — consistent across the whole fast family, only the 200-DMA fails. Net-of-tax it’s still a big Calmar win but liquidating the cash book realises STCG (~2.5 switches/yr) → build it as a NIFTY-futures/puts hedge instead.',
        columns: ['Book', 'CAGR', 'MaxDD', 'Calmar', 'Sw/yr', 'Net-tax CAGR', 'Net-tax Calmar'],
        rows: [
          ['PLAIN basket (tax-deferred)', '21.8%', '−39.2%', '0.56', '0', '21.8%', '0.56'],
          ['+ daily-ST(7,3) overlay', '19.6%', '−15.3%', '1.28', '5.0', '16.8%', '1.01'],
          ['+ daily-ST(10,3)', '19.4%', '−15.5%', '1.25', '5.1', '16.5%', '0.96'],
          ['+ 50-DMA gate', '20.3%', '−16.6%', '1.23', '13.6', '16.6%', '0.93'],
          ['+ 200-DMA gate (too slow — HURTS)', '15.1%', '−33.3%', '0.45', '9.5', '13.2%', '0.38'],
        ],
        highlightRows: [1],
        heatmap: false,
      },
      {
        title: 'Cross-check (tested, REJECTED) — does a daily-ST gate improve the LIVE momentum book? No.',
        caption: 'The Phase-2 overlay result tempted a swap: replace the live research/62 momentum book’s 100-DMA regime gate with a daily-ST gate. Tested head-to-head on the winner config (rsblend N8 buf22 donch15, net-post-tax, 2014–26): the 100-DMA WINS decisively (net Calmar 1.71). The ST gates are twitchier (de-risk 30–36× vs 23), whipsaw out of recoveries, and give up ~6pp CAGR for no drawdown benefit. Lesson: it isn’t "ST > moving-average gate" — it’s "medium-speed gate > slow (200-DMA) gate", and the 100-DMA already is that. KEEP the live gate.',
        columns: ['Gate on the momentum book', 'net CAGR', 'net MaxDD', 'net Calmar', 'De-risk events'],
        rows: [
          ['No gate', '30.4%', '−39.5%', '0.77', '0'],
          ['100-DMA (current, LIVE)', '30.2%', '−17.6%', '1.71', '23'],
          ['daily-ST(7,3)', '24.6%', '−18.5%', '1.33', '30'],
          ['daily-ST(10,3)', '22.9%', '−18.3%', '1.25', '32'],
          ['50-DMA', '20.9%', '−21.1%', '0.99', '36'],
        ],
        highlightRows: [1],
        heatmap: false,
      },
      {
        title: '★★ PHASE 3 — the cleanest tradeable version: time the actual INDEX ETF (NIFTYBEES) itself, 2010–2026',
        caption: 'Index-level trend filter ON the real ETF (hold when green, to cash when red). No survivorship, one liquid instrument, infinite capacity. Net-of-tax: ~1.5pp CAGR give-up, drawdown MORE THAN HALVED (−36%→−14%), Calmar & Sharpe ~doubled. Pre-tax the give-up is ~zero. ST(7,3) marginally best (fewest switches → least tax) but 50/100-DMA are tied — it is any fast-medium trend filter, NOT ST-specific; the 200-DMA is too slow and HURTS. A real, robust, WELL-KNOWN tactical-timing result (Faber-style), confirmed on Indian ETFs. Holds across NIFTYBEES/JUNIORBEES/BANKBEES; does NOT help GOLDBEES.',
        columns: ['Signal on NIFTYBEES', 'CAGR', 'MaxDD', 'Calmar', 'Sharpe', 'Sw/yr', 'Net-tax CAGR', 'Net-tax Calmar'],
        rows: [
          ['Buy & hold', '10.6%', '−36.3%', '0.29', '0.75', '0', '10.6%', '0.29'],
          ['ST(7,3) — best net', '10.7%', '−14.2%', '0.75', '1.11', '5.0', '9.0%', '0.53'],
          ['100-DMA', '10.3%', '−14.2%', '0.72', '1.02', '8.7', '8.4%', '0.52'],
          ['50-DMA', '10.1%', '−14.0%', '0.72', '1.05', '13.6', '8.0%', '0.49'],
          ['200-DMA (too slow — HURTS)', '5.5%', '−20.7%', '0.27', '0.55', '9.5', '4.6%', '0.20'],
        ],
        highlightRows: [1],
        heatmap: false,
      },
      {
        title: 'REAL-WORLD FRICTIONS on the winner — liquid-fund idle return + T+1 settlement (NIFTYBEES · ST 7,3)',
        caption: 'The idle cash between trades earns a LIQUID fund return NET of its expense + slab tax (~6.5% gross → ~4.5% net), and India ETF settlement is T+1 (on a red flip you exit but proceeds sit in transit ~1 day; on a green flip you re-enter ~1 day late). The drawdown-halving is friction-PROOF (−14 to −17% everywhere), but the CAGR give-up grows: from the idealized ~0pp to ~1.3pp pre-tax / ~2.8pp net of ALL tax. The liquid fund is essential (worth ~1.8pp vs 0% cash). This is why the tax-free NIFTY-futures/puts hedge is the preferred build.',
        columns: ['Scenario', 'CAGR', 'MaxDD', 'Calmar', 'Net-of-all-tax CAGR', 'Net-tax Calmar'],
        rows: [
          ['Buy & hold', '10.6%', '−36.3%', '0.29', '10.6%', '0.29'],
          ['Idealized (6.5% cash, instant switch)', '10.7%', '−14.2%', '0.76', '9.1%', '0.54'],
          ['REALISTIC (liquid 4.5% net + T+1 lag)', '9.3%', '−14.3%', '0.65', '7.8%', '0.46'],
          ['Conservative (2-day lags)', '9.0%', '−16.6%', '0.54', '7.4%', '0.41'],
          ['Cash 0% (no liquid at all)', '7.5%', '−16.1%', '0.47', '6.0%', '0.32'],
        ],
        highlightRows: [2],
        heatmap: false,
      },
      {
        title: 'THE BEST BUILD (REAL-DATA VALIDATED) — futures-hedge instead of selling the ETF',
        caption: 'Keep NIFTYBEES (never sold → NO equity CGT, deferred like B&H; NO T+1 lag; margin by PLEDGING the ETF) and SHORT NIFTY futures on the red signal; hedged ≈ a synthetic T-bill earning the carry. VALIDATED on REAL NSE bhavcopy basis (196 points across COVID/2022/2018 crashes): the carry is ~+3% (mean +3.1% hedge-on), NOT the +4.6% first modeled — and futures DO flip to BACKWARDATION in crashes (COVID: 52% of days negative), which is already baked into the +3% realized average. With that real carry: ~9.9% CAGR (−0.6pp vs B&H) at −15% drawdown (halved), Calmar 0.67, and it now IMPROVES Sharpe (1.03 vs 0.75). Recovers MOST of the cash-rotation give-up. Residual caveats: constant-carry approximation (path effects in specific crashes), monthly roll execution, lot-size granularity, deferred-tax liability on the ETF.',
        columns: ['Approach', 'CAGR', 'MaxDD', 'Calmar', 'Sharpe'],
        rows: [
          ['Buy & hold', '10.5%', '−36.3%', '0.29', '0.75'],
          ['Cash-rotation (realistic, net all tax)', '7.8%', '−14.3%', '0.46', '~0.9'],
          ['Futures-hedge, ~2.0% carry (real, conservative)', '9.4%', '−15.2%', '0.62', '0.99'],
          ['Futures-hedge, ~3.2% carry (real, central)', '9.9%', '−14.8%', '0.67', '1.03'],
          ['Futures-hedge, ~4.6% carry (old modeled — too kind)', '10.5%', '−14.4%', '0.73', '1.09'],
        ],
        highlightRows: [3],
        heatmap: false,
      },
      {
        title: 'REAL NIFTY futures basis (NSE bhavcopy, 196 pts) — the backwardation check',
        caption: 'Annualised carry = the futures premium the short captures. Positive in normal/uptrend regimes; flips negative (backwardation) in crashes, exactly when the hedge is on — but the average stays positive, so the ~9.9% hedge result (which uses the realized average) already accounts for it. The extreme −20 to −46%/yr figures are near-expiry annualisation artifacts of tiny (−0.4%) absolute basis moves.',
        columns: ['Regime', 'Data pts', 'Mean carry %/yr', 'Median', 'Days in backwardation'],
        rows: [
          ['Hedge-OFF (ST green / uptrend)', '49', '+5.1%', '+4.5%', '0%'],
          ['Hedge-ON (ST red / downtrend)', '147', '+3.1%', '+1.1%', '36%'],
          ['— COVID 2020 (Feb–May)', '60', '+3.0%', '−0.6%', '52%'],
          ['— 2022 selloff', '59', '+2.3%', '+1.6%', '27%'],
          ['— 2018 correction', '30', '+4.5%', '+4.4%', '13%'],
        ],
        highlightRows: [1],
        heatmap: false,
      },
      {
        title: 'Apply the overlay to our BEST-CAGR book (research/75 momentum, 31.9% CAGR)? — the value is INVERSE to return',
        caption: 'Take the highest-return book we have and overlay the same NIFTY daily-ST(7,3) crash filter (idle cash at the honest ~4.5% NET liquid rate). It DOES cut the drawdown (−32%→−22%) and PRE-TAX lifts Calmar (1.01→1.17). BUT net of ALL tax it HURTS (0.88 < 1.01) — pulling a high-gain momentum book to cash ~5×/yr triggers heavy short-term tax, and being out 39% of the time forgoes ~30%/yr (vs a 10% index). The hedge version (1.14) avoids the tax but NIFTY futures don’t cleanly hedge a midcap book (beta>1, idiosyncratic risk) → optimistic. KEY LESSON: the crash overlay’s value is INVERSELY related to the underlying’s return — it’s a tool for low-return, high-DD INDEX ETFs, not for an already-high-Calmar momentum book (best de-risked by its own regime gate, per the gate cross-check).',
        columns: ['Version', 'CAGR', 'MaxDD', 'Calmar', 'Sharpe'],
        rows: [
          ['Base momentum book (research/75, no overlay)', '31.9%', '−31.6%', '1.01', '1.45'],
          ['+ NIFTY-ST overlay → cash 4.5% net (pre-tax)', '26.0%', '−22.2%', '1.17', '1.45'],
          ['+ NIFTY-ST overlay → cash 4.5% net (net ALL tax)', '21.0%', '−23.9%', '0.88', '1.18'],
          ['+ NIFTY-ST overlay → hedge-carry (proxy, optimistic)', '25.4%', '−22.2%', '1.14', '1.42'],
        ],
        highlightRows: [0],
        heatmap: false,
      },
      {
        title: 'Should we add a SHORT sleeve (bidirectional long/short)? Tested — NO, on both timeframes',
        caption: 'Instead of going FLAT when ST is red, go net SHORT to profit from downtrends. The short side is a structural LOSER: during ST-red periods the index still RISES (+6%/yr on daily ST, +19%/yr on weekly ST) because the slow filter flags red AFTER the drop and stays red THROUGH the recovery — you short into the bounce. Short-only makes ~nothing (daily +0.8%) or loses (weekly −1.9%) at huge DD. Bidirectional cuts CAGR and roughly DOUBLES drawdown. Weekly is worse than daily throughout (too slow). STAY LONG-ONLY: hold the ETF, hedge to flat in downtrends, never short.',
        columns: ['Book', 'Signal', 'CAGR', 'MaxDD', 'Calmar', 'Sharpe'],
        rows: [
          ['Buy & hold', '—', '10.5%', '−36.3%', '0.29', '0.75'],
          ['Long-only (hedge, the winner)', 'daily ST(7,3)', '9.9%', '−14.8%', '0.67', '1.03'],
          ['Bidirectional long/short', 'daily ST(7,3)', '6.6%', '−25.3%', '0.26', '0.51'],
          ['Short-only (diagnostic)', 'daily ST(7,3)', '0.8%', '−33.8%', '0.02', '0.13'],
          ['Long-only (hedge)', 'weekly ST(10,3)', '6.3%', '−31.1%', '0.20', '0.61'],
          ['Bidirectional long/short', 'weekly ST(10,3)', '0.3%', '−50.9%', '0.00', '0.09'],
          ['Short-only (diagnostic)', 'weekly ST(10,3)', '−1.9%', '−43.6%', '—', 'neg'],
        ],
        highlightRows: [1],
        heatmap: false,
      },
    ],
    results: {
      metrics: [
        { label: 'WINNER MaxDD', value: '−14.3%', tone: 'pos', hint: 'NIFTYBEES+ST(7,3) vs B&H −36.3%' },
        { label: 'WINNER Calmar', value: '0.65', tone: 'pos', hint: 'vs B&H 0.29' },
        { label: 'WINNER CAGR', value: '9.3%', hint: 'vs B&H 10.6% (realistic, net cost+liquid+lag)' },
        { label: 'Net-of-all-tax CAGR', value: '7.8%', hint: '~2.8pp give-up net of everything' },
        { label: 'Volatility', value: '9.7%', tone: 'pos', hint: 'vs B&H 15.1%' },
        { label: 'Sharpe', value: '≈ tied', hint: 'DD-reduction overlay, not return-enhancer' },
        { label: 'Per-name ST timing', value: 'LOSES', tone: 'neg', hint: '−3 to −7pp/yr vs same-basket B&H' },
        { label: 'Build', value: 'futures/puts hedge', hint: 'avoids tax + settlement drag' },
      ],
      tables: [
        {
          title: 'Attribution of the +6.9pp "headline"',
          columns: ['Source', 'Contribution'],
          rows: [
            ['Survivorship + Nifty200-breadth (basket vs Nifty50 index)', '+10.4pp (basket 21.0% − index 10.6%)'],
            ['SuperTrend TIMING (ST book vs same basket)', '−3.5pp'],
            ['Net headline (ST book vs Nifty50 index)', '+6.9pp'],
          ],
          highlightRows: [1],
        },
        {
          title: 'OOS split (10,3) — the ST cell itself is stable, but it still trails the basket',
          columns: ['Window', 'CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Train 2010–2019', '15.2%', '−31.7%', '0.48'],
            ['Test 2020–2026', '20.0%', '−19.4%', '1.03'],
          ],
        },
      ],
      charts: [
        {
          src: '/app/niftybees-st73-winner.png',
          caption:
            '★★ THE WINNER — NIFTYBEES timed by daily SuperTrend(7,3), REALISTIC (net of cost + liquid-fund net return ~4.5% + T+1 settlement lag), 2010–2026. Max drawdown −14.3% vs the index’s −36.3% (Calmar 0.65 vs 0.29, volatility 9.7% vs 15.1%) for a ~1.3pp CAGR give-up (9.3% vs 10.6%); net of ETF STCG/LTCG ≈ 7.8%. Note Sharpe ≈ tied (0.33 vs 0.34): this is a DRAWDOWN-reduction overlay, not a return-enhancer. Best implemented as a futures/puts hedge to avoid the tax + settlement drag. This is the deliverable.',
        },
        {
          src: '/app/weekly-supertrend-nifty200.png',
          caption:
            'FOR CONTRAST — the Phase-1 "illusion" factsheet: the per-name Weekly-SuperTrend Nifty 200 book vs the NIFTY 50 index. This is the UNFAIR comparison (the gap is survivorship + breadth, not timing). Against a same-200-names buy-and-hold the per-stock timing underperforms by ~3.5pp/yr. Kept only to show why the "beats the index" headline was misleading.',
        },
      ],
    },
    winners: [
      {
        config: 'THE WINNER: trend-time the NIFTYBEES ETF with daily SuperTrend(7,3) — a drawdown-reduction overlay',
        summary: 'Hold NIFTYBEES when its daily ST(7,3) is green; on red sell to a liquid fund (~4.5% net) and re-enter on the next green. Realistic (net of cost + liquid-fund tax/expense + T+1 settlement lag), it halves the max drawdown (−36%→−14%) and cuts volatility (15%→10%) for a ~1.3pp CAGR give-up (~2.8pp net of ETF tax). It is roughly Sharpe-neutral — a way to sidestep the −36% crashes at a modest return cost, NOT a return-enhancer. Clean, one liquid instrument, no survivorship, infinitely scalable. Best built as a NIFTY-futures/puts hedge (no ETF sale → no equity tax, no settlement drag) to recover most of the give-up.',
        metrics: [
          { k: 'Max Drawdown', v: '−14.3% vs −36.3% (B&H)' },
          { k: 'Calmar', v: '0.65 vs 0.29' },
          { k: 'CAGR', v: '9.3% vs 10.6% (−1.3pp; net-tax 7.8%)' },
          { k: 'Volatility', v: '9.7% vs 15.1%' },
          { k: 'Sharpe', v: '≈ tied — DD-reduction, not return' },
          { k: 'Best filter', v: 'ST(7,3) ≈ 50/100-DMA; 200-DMA too slow' },
        ],
        rejected: [
          'STOCK-LEVEL per-name ST timing (the pitched system): LOSES to buy-and-hold the same names by 3–7pp/yr. The "+6.9pp beats the index" headline was survivorship + Nifty200-breadth vs the Nifty 50 index, not the signal.',
          '40/40/20 profit-booking: a return-killer (17.5%→8.8% CAGR) — caps the fat right tail that carries trend-following.',
          'Daily-ST as the LIVE momentum-book gate: worse than the current 100-DMA (net Calmar 1.71→1.33) — keep the 100-DMA.',
          'Small-cap tilt: higher gross return but untradeable at size (research/62 capacity wall); and 200-DMA timing of the ETF halves CAGR.',
        ],
      },
    ],
    caveats: [
      'BENCHMARK ARTIFACT (the headline correction): the attractive "+6.9pp over NIFTYBEES / Calmar 0.55" compares a survivorship-selected TODAY’s-Nifty-200 book to the Nifty 50 index. Against the fair, survivorship-matched benchmark (buy-and-hold of the SAME 200 names) the SuperTrend timing LOSES by ~3–3.5pp/yr at equal Calmar. No investable timing edge.',
      'Survivorship bias (the root cause): today’s index membership applied to the past; the right tail that drives returns IS the survivorship-selected multibaggers (TARIL +5347%, KEI +3717%, PAGEIND, RVNL, ADANIENT). A point-in-time universe would cut these numbers materially — for BOTH the ST book and the basket.',
      'SIGNAL ≠ STRATEGY: the per-trade ENTRY timing has a small real edge (G1 +5.2pp vs random-hold), but as a portfolio it is swamped by the opportunity cost of being out of the market / under-concentrated through a 16-year bull. Same lesson as research/49 ("it’s beta, not alpha").',
      'PHASE 2 RESOLVED the salvageable angle: as a MARKET-LEVEL crash overlay (daily ST(7,3) on the index, flatten the whole book in downtrends) the SuperTrend more than doubles pre-tax Calmar (0.56→1.28, DD −39→−15%). But (a) implemented by liquidating the cash basket it realises STCG ~2.5×/yr → net Calmar 1.01 (still ≫ 0.56, but a ~5pp CAGR give-up vs tax-deferred B&H); the tax-efficient build is a NIFTY-futures/puts hedge (owed a test incl. roll/basis/tracking cost). (b) Recent chop (2025–26) the overlay lagged. (c) Pre-tax 1.28 is good but still below the existing regime-gated momentum book (~1.7).',
      'Drawdown ≈ market either way (−32%). Weekly ST is slow; in fast crashes the flip comes AFTER a big drop. Not a capital-preserving product.',
      'Modeled sizing/cost/tax (0.30% RT, STCG 15%/LTCG 10%, idle cash 6.5%); the basket benchmarks are gross of cost (a buy-and-hold basket barely trades, so this is minor and does not change the conclusion). Nothing wired live. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      { label: '📊 Clean visual report (HTML) — the full story on one page', href: '/app/weekly-supertrend-report.html' },
      { label: 'RESULTS.md (verdict + correction + tables)', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/73_weekly_supertrend_investing/results/RESULTS.md' },
      { label: 'fair_bench.py (the deciding survivorship-matched test)', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/73_weekly_supertrend_investing/scripts/fair_bench.py' },
      { label: 'st_weekly_engine.py + g4_portfolio.py', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/73_weekly_supertrend_investing/scripts/st_weekly_engine.py' },
    ],
    projectPaths: [
      'research\\73_weekly_supertrend_investing\\WEEKLY_SUPERTREND_10_3_WEEKLY_SWEEP_STATUS.md',
      'research\\73_weekly_supertrend_investing\\scripts\\ (st_weekly_engine, g1_signal_probe, g4_portfolio, g3_param_sens, fair_bench, make_tearsheet).py',
      'research\\73_weekly_supertrend_investing\\results\\ (RESULTS.md, weekly-supertrend-nifty200.png, g*_*.csv)',
    ],
  },
  {
    slug: 'gaporb-morning-strength-research81',
    title:
      'Swing Edge Discovery (8 families, 170 pre-registered cells) — Gap-Up + ORB Long was REAL and the OOS look caught it DECAYING',
    verdict:
      'VERDICT: SIGNAL (decaying) — NOT investable as tested. A two-day systematic search for automatable 2-4 day swing systems on 5-min data (381 symbols backfilled to 2015 as part of the study). Seven families died cheaply at the IS gate: daily Donchian breakouts, vol-squeeze breakouts, EOD-strength carry, 5-day RS rotation, PDH/PWH breaks, MA crossovers on every timeframe, and the first-candle coin-toss (gross is literally zero). EVERY short-side variant of every setup loses net at this horizon. The one real anomaly: MORNING-STRENGTH CONTINUATION — gap-up >=0.25% at open + opening-range (60-min) breakout, long only, held up to 4 sessions. In-sample it was overwhelming: +21bps/trade net, t=5.6 across 77 F&O names, positive 86% of years, survived plateau/walk-forward/Monte-Carlo/regime/super-winner batteries, and a 10-name book did Sharpe 1.0 at MaxDD -17% (NIFTY B&H: -38%). Then the single authorized OOS look (2024-2026) rendered the verdict IS+Val could not: the per-trade edge HALVED out of sample and turned negative — +33bps in 2024, +5 in 2025, -27 in 2026 — and both pre-declared books fail the gates (G4 book OOS CAGR -1.6%). Most plausible cause: ORB-style morning momentum went retail-mainstream post-2023 and was crowded away. Deploying on the in-sample evidence would be funding the 2026 bleed. Do not trade; paper-monitor at most. The OOS ledger is consumed for this family.',
    status: 'COMPLETE',
    date: '2026-07-17',
    cardBlurb:
      'Systematic 8-family swing search: 7 families buried with data, all shorts lose net, and the one real edge (gap-up + ORB long, IS t=5.6 on 77 F&O names) was caught DECAYING by the one-time OOS look: +33bps 2024, +5 2025, -27 2026. Not deployed — the process working as designed.',
    cardStats: [
      { label: 'IS edge (77 names)', value: '+21bps t=5.6' },
      { label: 'OOS decay 24/25/26', value: '+33 / +5 / -27 bps' },
      { label: 'Families killed', value: '7 of 8' },
    ],
    system: {
      intro: 'The lead system (the only family that survived in-sample gates). Fully automatable; every rule causal.',
      rows: [
        { k: 'Setup filter', v: 'Overnight gap-up >= 0.25% (open vs prev close) — validated monotone dose-response; gap-down INVERTS the edge.' },
        { k: 'Entry', v: 'First 5-min close above the 09:15-10:15 opening-range high, after 10:15 -> buy next bar open. One entry/symbol/day. LONG ONLY (all short mirrors negative).' },
        { k: 'Stop', v: 'Opening-range low (gap-through fills at open, modeled).' },
        { k: 'Exit', v: 'No target; hard time-stop at the close of session entry+3 (max 4 sessions). Intraday exits destroy the edge — it is multi-day drift.' },
        { k: 'Universe', v: 'F&O liquid names (edge concentrates there: non-F&O gross is half) + NIFTY index as futures-proxy.' },
        { k: 'Costs', v: 'Futures-proxy model: brokerage/STT/exchange/GST + 1bp (index) / 3bp (stocks) slippage per side; every result net.' },
      ],
    },
    conditions: {
      intro: 'Data built for the study, then audited and repaired.',
      rows: [
        { k: 'Data', v: '5-minute OHLCV backfilled 2015->2024 for 381 symbols (was 2024-only); adjustment-drift audit found 6 F&O names on mixed split bases (fake 401% jumps) — deleted and refetched; BANKNIFTY 2015+ added via index token.' },
        { k: 'Splits', v: 'IS 2015-02..2021-09 (60%), Val 2021-10..2023-12, OOS 2024-01..2026-07 touched ONCE (user-authorized) for the whole family.' },
        { k: 'Multiple-testing ledger', v: '~184 pre-registered cells across all experiments; every grid locked before running; failed experiments reported, never re-gridded.' },
        { k: 'Survivorship', v: 'Universe = today’s liquid names (biased); OOS decay is if anything understated by this.' },
      ],
    },
    comparisons: [
      {
        title: 'Family league table (all net of costs, IS unless noted)',
        columns: ['Family / system', 'Best result', 'Verdict'],
        rows: [
          ['Gap-up + ORB long <=4d (77 F&O)', 'IS +20.9bps t=5.62; OOS +10.4 t=2.57 but decaying', 'SIGNAL (decaying)'],
          ['Same on NIFTY index', 'IS +29bps t=3.0; Val pass @1bp; OOS n=52 only', 'SIGNAL, thin'],
          ['Deep-z reversion SHORT fade (daily)', '+32bps t=1.5, positive 8-10/13 yrs', 'SIGNAL, too thin'],
          ['Open=Low break long', 'megacaps t=3.2 -> breadth t=0.87', 'does not generalize'],
          ['CPR-open long', 'megacaps t=2.4 -> breadth negative', 'does not generalize'],
          ['Donchian daily / squeeze / EOD-carry / 5d-RS / PDH-PWH / MA crosses / coin-toss', 'all negative net', 'NO EDGE (7 families)'],
        ],
        highlightRows: [0],
      },
      {
        title: 'The OOS decay that killed deployment (stock cell W12, net bps/trade)',
        columns: ['Window', 'Net bps/trade', 'Note'],
        rows: [
          ['IS 2015-2021', '+20.9 (t=5.62)', 'all robustness batteries passed'],
          ['OOS 2024', '+32.9', 'edge alive'],
          ['OOS 2025', '+5.3', 'decaying'],
          ['OOS 2026 (to Jul)', '-26.7', 'gone / negative'],
        ],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'IS t-stat (77 names)', value: '5.62', tone: 'pos' },
        { label: 'OOS t-stat', value: '2.57', hint: 'positive overall but decaying by year' },
        { label: 'G4 book IS+Val', value: 'Sharpe 1.00 / DD -17%', tone: 'pos' },
        { label: 'G4 book OOS', value: 'CAGR -1.6%', tone: 'neg' },
        { label: 'Shorts (any setup)', value: 'all negative', tone: 'neg' },
        { label: 'OOS ledger', value: 'CONSUMED 2026-07-16' },
      ],
      tables: [
        {
          title: '10-name book (locked construction): in-sample vs the OOS look',
          columns: ['Window', 'CAGR', 'Sharpe', 'MaxDD', 'Calmar'],
          rows: [
            ['IS+Val 2015-2023', '13.2%', '1.00', '-17.2%', '0.77'],
            ['NIFTY B&H same period', '10.7%', '0.70', '-38.2%', '0.28'],
            ['OOS 2024-2026', '-1.6%', '-0.10', '-21.4%', '-0.07'],
          ],
          highlightRows: [2],
        },
      ],
      charts: [
        {
          src: '/app/gaporb-research81-factsheet.png',
          caption: 'Client factsheet — 10-name book, full period 2015-2026 with the OOS flattening visible (net of futures-proxy costs; NIFTY benchmark; NOT deployed).',
        },
      ],
    },
    winners: [
      {
        config: 'None deployable — the honest outcome',
        summary: 'The study succeeded as a PROCESS: 7 families killed cheaply, one real edge found and validated in-sample, and the one-time OOS look caught its decay before capital did.',
        metrics: [
          { k: 'Durable byproduct 1', v: '381-symbol 5-min history 2015->2024 + adjustment repairs in market_data.db' },
          { k: 'Durable byproduct 2', v: 'Unit-tested reusable 5-min/daily backtest engine (32 assertions)' },
          { k: 'Durable lesson', v: 'OR-width inverts equal-risk sizing; trade-level t-stats do not make a book (capacity 3x); shorts never survive 2-4d costs here' },
        ],
        rejected: [
          'All 6 breadth-book constructions (capacity-constrained; intake selection dominates)',
          'First-candle coin-toss RR 1:1.5 (gross exactly zero)',
          'Every short-side variant of every setup',
        ],
      },
    ],
    caveats: [
      'Universe survivorship-biased (today’s 381 names applied to the past).',
      'Futures modeled from cash series (user-approved proxy); slippage assumed 1-3bp/side.',
      'Index OOS cell has n=52 (below the 100-trade evidence gate).',
      '2026 covers ~6.5 months — decay vs temporary regime not fully separable; the verdict label (SIGNAL, decaying) stands either way.',
      'OOS is consumed for this family: any revival (decay-gate, OR-width filter) validates only via paper-forward.',
    ],
    githubLinks: [
      { label: 'RESULTS.md (full verdict)', href: 'https://github.com/castroarun/Quantifyd/blob/main/research/81_swing_edge_discovery/results/RESULTS.md' },
      { label: 'Study crash-doc / ledgers', href: 'https://github.com/castroarun/Quantifyd/blob/main/EDGE_DISCOVERY_81_STUDY_STATE.md' },
    ],
    projectPaths: [
      'research\\81_swing_edge_discovery\\engine\\ (loader, costs, backtester, metrics + tests)',
      'research\\81_swing_edge_discovery\\experiments\\ (A1..A8, B1..B3, C1-C2, D1-D2, E1, F1-F2, G3-G6, H1)',
      'research\\81_swing_edge_discovery\\results\\RESULTS.md',
    ],
  },
  {
    slug: 'nifty250-momentum-video-research75',
    title: 'Nifty-250 Momentum Top-15 — Faithful Replication of the "Only Momentum Strategy" Video',
    verdict:
      'Faithful survivorship-free replication of the Quantinuous "Only Momentum Strategy You Need for Nifty 250 Stocks" video (top-15 by 12-month momentum, per-stock 50>100>200 EMA filter, Nifty-100EMA cash gate, monthly). The video’s claimed 27% CAGR REPLICATES and is beaten — 31.8% net / 29.1% post-tax, 292×, 2006–2026 — but its claimed −23% MaxDD does NOT hold: the honest daily-marked drawdown is −31.6% (our 20-year window includes 2008). The NIFTYBEES-100EMA cash gate is the ENTIRE risk story (removing it → −51% DD) and is IRREPLACEABLE — no per-stock quality / ATH-proximity / MA-exit combination substitutes for it (best gate-less DD −46%). The per-stock EMA-stack filter is inert-to-harmful. Best RISK-ADJUSTED knob = midcap + 6-month relative strength (Calmar 1.26, −29% DD); highest raw CAGR = mid+small combo (43.5% net, but −42% DD). Same family as Aurum’s midcap_smoothest — corroborates the live design, does not add new alpha. STRATEGY-family candidate (already harvested).',
    status: 'COMPLETE',
    date: '2026-07-21',
    cardBlurb:
      'Replicated a popular YouTube momentum system on our own survivorship-free NSE data, 2006–2026, net of 0.3% cost, daily-marked. The 27% CAGR claim replicates (31.8% net, 292×); the −23% drawdown claim does not (−31.6%). Attribution: the index cash-gate is the whole risk story and cannot be replaced by per-stock rules; midcap + 6-month RS is the best knob.',
    cardStats: [
      { label: 'CAGR (net)', value: '31.8%' },
      { label: 'MaxDD (daily)', value: '−31.6%' },
      { label: 'Total return', value: '292×' },
    ],
    system: {
      intro: 'Long-only concentrated momentum with a market-regime cash gate — the video’s rules, exactly:',
      rows: [
        { k: 'Universe', v: 'Nifty LargeMidcap 250 — survivorship-free PIT proxy = top-250 NSE names by trailing-6-month median traded value, rebuilt monthly (ETFs/index excluded).' },
        { k: 'Selection', v: 'Rank the universe by momentum, hold the TOP 15 equal-weight, 100% invested when risk-on.' },
        { k: 'Momentum', v: 'Faithful default = plain 12-month price return (also tested 12−1, and 6m/12m relative strength — conclusion unchanged).' },
        { k: 'Per-stock trend filter', v: 'Eligible only if EMA50 > EMA100 > EMA200 on the stock’s own close (causal).' },
        { k: 'Market regime gate', v: 'If NIFTYBEES close ≤ its own EMA100 → liquidate to cash; NIFTYBEES = full-history Nifty-50 proxy (raw NIFTY50 daily only starts 2023).' },
        { k: 'Rotation', v: 'Monthly; daily-marked NAV for honest drawdown. Idle cash earns 6.5% p.a. (sensitivity: 4% → −1pp CAGR, 0% → −2.6pp).' },
        { k: 'Costs / tax', v: '0.3% round-trip on turnover; post-tax = 20% STCG on lots < 365 days (shown separately).' },
        { k: 'Window', v: '2006–2026 (~20.5y, incl. the 2008 crash — the real test of the −23% DD claim).' },
      ],
    },
    conditions: {
      intro: 'Backtest window, benchmark, host.',
      rows: [
        { k: 'Period', v: 'Jan 2006 – Jul 2026 (~20.5 years).' },
        { k: 'Benchmark', v: 'NIFTY-50 (NIFTYBEES), same window, excluded from the investable universe.' },
        { k: 'Host', v: 'VPS market_data.db snapshot 2026-07-08; reproducible from committed scripts.' },
      ],
    },
    comparisons: [
      {
        title: 'Universe × momentum-angle — net CAGR / MaxDD / Calmar (2006–2026)',
        columns: ['Universe · momentum', 'Net CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['midcap · 6m RS (rs126)', '37.2%', '−29.6%', '1.26'],
          ['midcap · 120d RS', '36.3%', '−29.2%', '1.25'],
          ['midcap · 6m+12m blend', '38.8%', '−32.8%', '1.18'],
          ['large-mid 250 · blend', '36.2%', '−32.8%', '1.11'],
          ['large-mid 250 · 12m (faithful, no stack)', '34.7%', '−32.2%', '1.08'],
          ['mid+small combo · 12m', '43.5%', '−42.2%', '1.03'],
          ['smallcap · 12m', '33.2%', '−46.2%', '0.72'],
        ],
        highlightRows: [0, 1],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR (net)', value: '31.8%', tone: 'pos' },
        { label: 'CAGR (post-tax 20%)', value: '29.1%', tone: 'pos' },
        { label: 'NIFTYBEES CAGR', value: '11.6%' },
        { label: 'Excess / yr', value: '+20.2%', tone: 'pos' },
        { label: 'Sharpe', value: '1.14', tone: 'pos' },
        { label: 'Max Drawdown', value: '−31.6%', tone: 'neg', hint: 'vs NIFTYBEES −59.7%; video claimed −23%' },
        { label: 'Calmar', value: '1.01', tone: 'pos' },
        { label: 'Yrs beating index', value: '71%' },
      ],
      tables: [
        {
          title: 'Faithful base vs benchmark',
          columns: ['Metric', 'Nifty-250 Momentum', 'NIFTYBEES'],
          rows: [
            ['CAGR', '31.8%', '11.6%'],
            ['Total return', '291.6x', '9.5x'],
            ['Sharpe', '1.14', '0.34'],
            ['Max Drawdown', '−31.6%', '−59.7%'],
            ['Calmar', '1.01', '0.19'],
          ],
          highlightRows: [0, 1, 3],
        },
        {
          title: 'Rule attribution — the gate is the whole risk story; the EMA-stack is inert',
          columns: ['Configuration', 'Net CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Faithful base (gate ON, EMA-stack ON)', '31.8%', '−31.6%', '1.01'],
            ['No EMA-stack (gate ON)', '34.7%', '−32.2%', '1.08'],
            ['No index gate (stack ON)', '~20%', '−51.3%', '0.38'],
            ['Pure momentum (no gate, no stack)', '~21%', '−51.4%', '0.40'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Can per-stock controls REPLACE the gate? (midcap RS120) — No.',
          columns: ['Risk control (no gate)', 'Net CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Gate ON (baseline)', '36.2%', '−29.2%', '1.24'],
            ['No gate, nothing', '32.9%', '−65.2%', '0.50'],
            ['+ quality filter', '36.3%', '−64.6%', '0.56'],
            ['+ ATH-proximity', '23.1%', '−58.4%', '0.39'],
            ['+ SMA100 exit', '32.7%', '−53.9%', '0.61'],
            ['+ Donchian-15 exit (best gate-less)', '31.0%', '−46.2%', '0.67'],
          ],
          highlightRows: [0],
        },
      ],
      charts: [
        {
          src: '/app/nifty250-momentum-research75-factsheet.png',
          caption:
            'CLIENT FACTSHEET — Nifty-250 Momentum Top-15 (faithful video replication) vs NIFTY 50, 2006–2026, survivorship-free, net of 0.3% cost, daily-marked. 31.8% CAGR (29.1% post-tax), 291.6× vs 9.5×, Sharpe 1.14, MaxDD −31.6% vs −59.7%, 71% of years beating the index. Note the drawdown panel: the gate held the strategy to −31.6% while NIFTY fell −59.7% in 2008. Generated by research/_utilities/tearsheet.py.',
        },
      ],
    },
    winners: [
      {
        config: 'Faithful video base · large-mid 250 · 12m momentum · EMA-stack · 100EMA gate · monthly',
        summary: 'The video’s CAGR claim replicates and is beaten (31.8% net / 292×); its −23% drawdown claim does not (−31.6%, because our honest window includes 2008). Best risk-adjusted variant = midcap + 6-month RS (Calmar 1.26).',
        metrics: [
          { k: 'CAGR', v: '31.8% net / 29.1% post-tax' },
          { k: 'Excess', v: '+20.2%/yr vs NIFTYBEES' },
          { k: 'Sharpe', v: '1.14' },
          { k: 'MaxDD', v: '−31.6%' },
          { k: 'Calmar', v: '1.01' },
        ],
        rejected: [
          'Dropping the 100EMA gate: drawdown blows out to −51% — the gate is the entire risk story and IRREPLACEABLE.',
          'Per-stock quality / ATH-proximity / MA-exit as a gate substitute: best gate-less DD is still −46% (Donchian), Calmar ≤ 0.67 vs the gated 1.24. Stocks break one at a time — a per-stock rule cannot be a market circuit-breaker.',
          'The per-stock 50>100>200 EMA filter: inert-to-harmful (removing it RAISES CAGR 31.8% → 34.7% at the same drawdown).',
          'mid+small combo (highest CAGR 43.5%, survives cost-stress to 0.7%) — rejected on drawdown: −42% is uninvestable.',
        ],
      },
    ],
    caveats: [
      'The video’s −23% MaxDD is not reproducible on an honest 20-year, daily-marked, survivorship-free basis — we get −31.6%. The likely reasons the video looks shallower: a shorter/post-2008 window, monthly (not daily) marking, and/or index-provided constituent data. Its 27% CAGR, by contrast, is conservative — we beat it (31.8%).',
      'A data-integrity bug was caught and fixed mid-study: the first run reused a helper that hard-coded a 2014 rebalance start, parking the book in cash for 2006–2013 and never testing 2008 (it reported a spurious 21.5% with a +111% 2014 jump). Fixed to trade the full 2006–2026 calendar; all headline numbers are post-fix.',
      'Cash-yield assumption: idle cash earns a modeled 6.5% p.a. (liquid fund). At a realistic 4% the CAGR is ~1pp lower; at 0% it is ~2.6pp lower. A modest but real tailwind.',
      'Momentum definition was the one thing the video left unspecified; the plain-12m default and two alternatives (12−1, risk-adjusted 6m/12m) all give the same conclusion, so the result does not hinge on it.',
      'Cost realism: 0.3% round-trip is defensible for large-mid and midcap; smallcap/combo cells are optimistic (real 0.5–0.7%+) — cost-stressed, they survive on CAGR (combo 43%→39%) but are killed by drawdown, not fees.',
      'Redundancy: this is the same family as the live momentum-paper book and Aurum’s midcap_smoothest (research/41/62). The study CORROBORATES those; it is not new alpha.',
      'Backtest, net of modelled costs (post-tax where stated). Nothing wired live. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'RESULTS (phase 2 verdict + tables)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/75_nifty250_momentum_top15/results/RESULTS_P2.md',
      },
      {
        label: 'run_nifty250_momentum.py (faithful engine)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py',
      },
    ],
    projectPaths: [
      'research\\75_nifty250_momentum_top15\\NIFTY250_MOMENTUM_TOP15_DAILY_SWEEP_STATUS.md',
      'research\\75_nifty250_momentum_top15\\scripts\\ (run_nifty250_momentum.py, run_variants_phase2.py, run_phase3_combos.py)',
      'research\\75_nifty250_momentum_top15\\results\\ (ranking*.csv, RESULTS_P2.md, tearsheet.png)',
    ],
  },
  {
    slug: 'nifty-strangle-rules-research90',
    title: 'NSR-W v1.1 — Rules-Based NIFTY Weekly Strangle (automating the manual book)',
    verdict:
      'STRATEGY-CANDIDATE. Born from the W30 mentor review: the untouched Monday strangle beat the 22-leg manual ' +
      'management by ~Rs6k and the root habit was calm-day credit-chasing rolls toward spot. Four studies in one day on ' +
      'REAL data (nse_options_bhav 2019-2026 + 1-min recorded chain Apr-Jul 2026): (G1) monthly strangle + per-leg ' +
      'premium stop cuts the tail 6x and DOUBLES the t-stat; (G2, pessimistic gap-aware fills) stop 2.0-2.5x survives, ' +
      '1.5x monthly dies, giveback rule harmful, post-stop answer = flat-both on monthly but ROLL-AWAY-once on weekly ' +
      '(best family: t 4.73); indicator exits (ATR/ADX/VIX-jump) all lose to the premium stop - the premium IS the ' +
      'composite sensor; (G3) Arun’s own spec - Monday entry, NEXT-week expiry, strikes by Rs-premium target - BEATS ' +
      'day-after-expiry entries: Rs30 target + PT50 + stop 2.0x + roll = 11.06 pts/wk, t 5.47, positive ALL 8 years. ' +
      '(Replay) 13 live-recorded weeks at quote-level execution: 11W/2L, +Rs1.41L at 10 lots, robot profit-took Arun’s ' +
      'exact W30 strikes on TUESDAY and skipped the drift he spent the week fighting. Entry-time sweep: LATE Monday ' +
      '(~15:00-15:15) beats morning entries and matches the EOD assumption; PT sweep on 378 weeks: 40-60% is the ' +
      'plateau, deeper PTs lose consistency. Daily-0916 variant tested and shelved: Wed-Fri intraday premium selling ' +
      'nets ~zero; the edge is Mon/Tue only (independently reproduces research/51’s 0/1-DTE finding). Residual risk: ' +
      'gap-and-grind weeks (worst EOD week -445 pts = -Rs2.9L at 10 lots) survive every stop rule; event-skip rule ' +
      'proposed, untested. Next: G5 paper book beside the straddle V1/V2 books; weekly human-vs-robot mentor review.',
    status: 'COMPLETE',
    date: '2026-07-24',
    cardBlurb:
      'Can mechanical rules replace the manual strangle book? Monday ~15:00, next-week expiry, sell CE+PE nearest ' +
      'Rs20-30 premium, GTT stop 2x/leg, profit-take 50%, one roll-away, exit DTE<=1. Validated three ways: 378 weekly ' +
      'cycles EOD 2019-2026 (t 4.8-5.5, positive 7-8/8 years), pessimistic gap-aware fills, and a 13-week quote-level ' +
      'replay on our own recorded chain (11W/2L, +Rs1.41L at 10 lots) that re-traded the user’s exact W30 strikes and won.',
    cardStats: [
      { label: 'EOD t-stat (378 wks)', value: '5.47' },
      { label: 'Replay 13 wks @10 lots', value: '+Rs1.41L' },
      { label: 'Win/loss (replay)', value: '11 / 2' },
    ],
    systemRules: {
      intro: 'NSR-W v1.1 - the locked rule set. Every exit is pre-committed at entry; there is no discretionary decision after the sell.',
      sharedCoreTitle: 'Shared core (both premium targets)',
      sharedCore: [
        { k: 'Entry', v: 'Monday afternoon ~15:00-15:15 (entry-time sweep: late Monday beats 09:16/09:30 - strikes are set AFTER the day’s move is known; matches the EOD backtest’s close-entry assumption).' },
        { k: 'Expiry', v: 'NEXT week’s expiry (cal DTE 6-12, ~9.6 avg). Premium-targeting at this DTE lands ~2.7-3.2% OTM - further than the same rupees on shorter DTE.' },
        { k: 'Strikes', v: 'Sell CE + PE nearest the Rs-premium target (liquid strikes only, volume/OI > 0). Premium targeting auto-adapts to vol: high IV pushes strikes further OTM.' },
        { k: 'Stop', v: 'GTT at 2.0x each leg’s own credit, placed AT entry. (2.0x, not 1.5x - at ~9 DTE the tighter stop whipsaws; 1.5x is right only for ~5-DTE entries.)' },
        { k: 'Profit-take', v: '50% of total credit (378-week sweep: 40-60% is the optimal plateau; 70/80/none lose consistency).' },
        { k: 'EOD recenter (v1.2)', v: 'At 15:15 daily: if any leg >= 1.5x its credit, close the WHOLE book at the close and re-sell a fresh equidistant-by-premium strangle (once per week). 378-wk test: lifts t 5.47 -> 5.84 at unchanged tail; beat the exit-heavy-leg alternative (t 5.63). Distinct from manual rolling: fires once, at the close, on a defined trigger, restarting a symmetric book - not defending a broken one.' },
        { k: 'Stop -> RE-STRANGLE (v1.3)', v: 'The FIRST stop closes the WHOLE book and re-sells a fresh Rs-target pair at current spot (once per cycle) - tested vs fixed-side roll (t 5.84), match-survivor (5.74) and half-target (5.73): re-strangle wins with t 6.76, better mean/tail/p5/post-22 on both targets. A second stop closes that leg only; the survivor rides. Same principle as the EOD recenter: when the book breaks, re-run the entry. v1.4: BOTH adjustments (re-strangle + recenter) target Rs20, not Rs30 - adj sweep found the Rs20-24 plateau (t 6.92-6.94, better tail); Rs15 too timid; a fresh Rs30 at reduced DTE sits too close to a moving market (user-raised, data-confirmed).' },
        { k: 'Time exit', v: 'Close everything at DTE <= 1, 15:15. No expiry-day holds.' },
        { k: 'Sizing', v: 'FIXED lots (baseline 10 lots = 650 qty; 1 pt = Rs650). Never add while red. Margin utilization <= 70%.' },
        { k: 'Never', v: 'Roll toward spot. Add size mid-drawdown. Hold into DTE 0. Trade stock options into results.' },
      ],
      riskLayer: {
        title: 'Premium-target variants (EOD 2019-2026, Monday arm, stop 2.0x + PT50 + roll)',
        columns: ['Target', 'Net pts/wk', 't-stat', 'Win %', 'p5 / worst (pts)', 'Avg OTM', 'Years positive'],
        rows: [
          ['Rs30/leg', '11.06', '5.47', '71%', '-40 / -422', '2.7%', '8 of 8 (incl 2026)'],
          ['Rs20/leg', '7.86', '4.79', '73%', '-29 / -445', '3.2%', '7 of 8 (2026 flat)'],
        ],
        highlightRows: [0],
      },
    },
    system: {
      intro: 'Short volatility, harvested with strict mechanical loss control. Objective is risk-shape, not alpha: match the manual book’s results with bounded losses and zero interventions.',
      rows: [
        { k: 'Origin', v: 'W30 mentor review (mentor/reviews/2026-W30.md): manual 22-leg management turned a +Rs12.7k untouched week into +Rs6.8k with a worse book; margin measured 97% utilized. The user’s Monday entry was already this system - the management was the leak.' },
        { k: 'Honest prior', v: 'research/89: index short-vol EV decayed to ~0 post-2022. The surviving premium sits in the OTM wings (strangle), not ATM vol (straddle) - harvestable only with strict stops. Expect modest EV; the win is consistency.' },
        { k: 'Rejected by data', v: 'Giveback stop (halves monthly means). Indicator exits - ATR pctile, ADX, VIX-jump all act one day late (t 1.1-1.6 vs 2.0-2.6). VIX>=1.25x-entry exit: higher mean, 2.7x fatter tail. Rolling toward spot: the documented manual leak. Daily 09:16 entries Wed-Fri: ~zero after costs.' },
      ],
    },
    conditions: {
      rows: [
        { k: 'EOD sweep', v: 'nse_options_bhav (real NSE bhav), NIFTY 2019-2026, 379 weekly cycles, liquidity filter contracts>=50, net of 0.5% premium + 0.15 pt; pessimistic gap-aware stop fills (gap-open -> fill at open).' },
        { k: 'Intraday replay', v: 'options_data.db 1-min full-chain snapshots, 2026-04-20 -> 07-24 (66 days, 14 Mondays). Entries at BID, exits/stops at ASK - real spreads and GTT slippage (13.7 pts observed in the Jul-8 crash minute).' },
        { k: 'Host', v: 'VPS (canonical). Scripts: research/90_nifty_strangle_rules/scripts/.' },
      ],
    },
    comparisons: [
      {
        title: 'What the premium stop buys (G2, monthly arm p2.5%, pessimistic fills)',
        columns: ['Exit rule', 'Net/cycle', 't', 'Worst cycle', 'Post-22 mean'],
        rows: [
          ['No stop (hold)', '51.3', '1.38', '-1,878 pts (-Rs12.2L)', '+82.9'],
          ['Stop 2.5x', '47.8 (PT50)', '2.61', '-301 pts (-Rs2.0L)', '+33.8'],
          ['Stop 2.0x', '33.2 (PT50)', '2.17', '-161 pts (-Rs1.0L)', '+15.9'],
          ['Stop 1.5x', '17-19', '1.4', '-184 pts', 'NEGATIVE'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Entry-time sweep (13-week replay, PT50 stop2.0; net pts/week)',
        caption: 'Late Monday wins - and the 378-week EOD study effectively assumed close entry, so the big-sample result already belongs to the late entry.',
        columns: ['Entry', 'Rs20 mean (t)', 'Rs30 mean (t)'],
        rows: [
          ['09:16', '12.3 (1.9)', '17.0 (2.4)'],
          ['09:30', '10.4 (1.5)', '16.7 (2.2)'],
          ['09:45', '15.5 (3.1)', '21.1 (3.9)'],
          ['11:00', '15.0 (3.6)', '21.9 (4.3)'],
          ['15:14', '17.2 (4.2)', '24.2 (4.5)'],
        ],
        highlightRows: [4],
      },
      {
        title: 'Profit-take level (EOD 378 weeks, Monday arm, Rs30, stop 2.0x)',
        columns: ['PT', 'Net pts/wk', 't-stat'],
        rows: [
          ['40%', '10.36', '5.48'],
          ['50%', '11.06', '5.47'],
          ['60%', '9.63', '3.80'],
          ['70%', '10.50', '4.06'],
          ['80%', '10.58', '4.00'],
          ['None (time exit)', '11.47', '4.25'],
        ],
        highlightRows: [0, 1],
      },
    ],
    results: {
      metrics: [
        { label: 'EOD net/wk (Rs30)', value: '11.06 pts', hint: 'Rs7.2k at 10 lots', tone: 'pos' },
        { label: 'EOD t-stat', value: '5.47', tone: 'pos' },
        { label: 'Years positive', value: '8 / 8', hint: 'incl. 2026', tone: 'pos' },
        { label: 'Replay 13 wks', value: '+Rs1.41L', hint: '11W / 2L at 10 lots', tone: 'pos' },
        { label: 'Median hold', value: '2 days', hint: 'winners banked Tue-Wed; margin free rest of week' },
        { label: 'Worst replay week', value: '-Rs34.5k', tone: 'neg', hint: 'Jul-06: two stops + survivor ride' },
        { label: 'Worst EOD week', value: '-445 pts', tone: 'neg', hint: 'Feb-2026 grind = -Rs2.9L at 10 lots; the sizing number' },
      ],
      tables: [
        {
          title: 'Per-year net pts/week (EOD, Monday arm, stop 2.0x + PT50 + roll)',
          columns: ['Year', 'Rs20', 'Rs30'],
          rows: [
            ['2019', '13.0', '16.7'], ['2020', '2.3', '6.0'], ['2021', '6.3', '2.7'],
            ['2022', '10.4', '17.4'], ['2023', '6.5', '7.6'], ['2024', '6.7', '9.6'],
            ['2025', '14.6', '18.8'], ['2026 (29w)', '-0.3', '9.7'],
          ],
          heatmap: true,
        },
        {
          title: '13-week quote-level replay (net Rs at 10 lots)',
          caption: 'Full minute-level P&L travel with every stop/roll/PT marked: interactive report linked below.',
          columns: ['Week (Mon)', 'Rs20', 'Rs30', 'How it ended'],
          rows: [
            ['Apr 27', '+19,100', '+20,500', 'PT Tue/Wed'],
            ['May 04', '+19,000', '+20,500', 'PT'],
            ['May 11', '-30,500', '-21,800', '2 stops -> survivor ride -> TIME'],
            ['May 18', '+10,200', '+9,900', 'TIME'],
            ['May 25', '+15,100', '+21,700', 'PT'],
            ['Jun 01', '+12,500', '+18,500', 'PT'],
            ['Jun 08', '+13,300', '+19,600', 'PT'],
            ['Jun 15', '+12,500', '+19,200', 'PT'],
            ['Jun 22', '+13,100', '+19,200', 'PT'],
            ['Jun 29', '+13,100', '+19,200', 'PT'],
            ['Jul 06', '-27,000', '-34,500', '2 stops -> survivor ride -> TIME'],
            ['Jul 13', '+5,500', '+10,700', 'stop+roll -> TIME (the drift week, survived)'],
            ['Jul 20', '+11,900', '+18,700', 'PT Tue 14:20 - the user’s own W30 strikes'],
          ],
        },
      ],
      embeds: [
        {
          src: '/app/nsrw-travel-research90.html',
          height: 2800,
          caption: 'Embedded live report: KPIs, weekly net bars, minute-level P&L travel of every week (stops/rolls/PT marked), full trade table, and the spec-evolution reasoning.',
        },
      ],
    },
    winners: [
      {
        config: 'NSR-W v1.2 - Rs30 target, Monday ~15:00, stop 2.0x, PT 50%, one roll-away',
        summary: 'Best cell of the whole program and a monotonic family, not a lucky point: t 5.47 over 378 weeks, positive all 8 years including 2026, and 11W/2L on quote-level replay.',
        metrics: [
          { k: 'Net/week', v: '11.06 pts (Rs7.2k @ 10 lots)' },
          { k: 't-stat', v: '5.47 (n=379)' },
          { k: 'Typical bad week (p5)', v: '-40 pts (-Rs26k)' },
          { k: 'Replay', v: '+Rs1.41L / 13 wks, median hold 2 days' },
        ],
        rejected: [
          'Giveback stop - halves monthly means everywhere',
          'Indicator exits (ATR/ADX/VIX-jump) - one day late by construction',
          'Daily 09:16 entries on Wed-Fri - ~zero after costs (edge is Mon/Tue only, confirms research/51)',
          'Monthly iron condor - untestable at EOD (stale wing marks); retest intraday',
        ],
      },
    ],
    caveats: [
      'research/89 prior stands: index short-vol EV mostly decayed post-2022. This system’s post-22 means are modest (+8-13 pts/wk); the product is CONSISTENCY and bounded loss, not alpha.',
      'The tail is not zero: gap-and-grind weeks (worst EOD -445 pts = -Rs2.9L at 10 lots, Feb-2026) survive every stop rule. Event-skip rule (elections/budget/Fed) proposed, NOT yet tested. Size so one such week is survivable.',
      'The 13-week replay window was a friendly regime (VIX 12-15) - its Rs/week runs above the long-run expectation; anchor on the EOD numbers.',
      'Multiple-testing: 96-cell replay grids on 13 weeks cannot rank PT levels or fine entry times alone - every replay hint was re-tested on the 378-week EOD sample before entering the spec (PT80 hint died there).',
      'GTT slippage is real: stops filled up to 13.7 pts past trigger in a crash minute. Modeled at quote level in the replay; EOD model uses gap-aware pessimistic fills.',
      'Not yet run as a live paper book - G5 build pending. Weekly human-vs-robot comparison will be the ongoing validation.',
    ],
    githubLinks: [
      { label: 'Interactive weekly P&L travel report (13-week replay)', href: '/app/nsrw-travel-research90.html' },
      { label: 'research/90 on GitHub', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/90_nifty_strangle_rules' },
    ],
    projectPaths: [
      'research/90_nifty_strangle_rules/DESIGN.md',
      'research/90_nifty_strangle_rules/results/RESULTS.md',
      'research/90_nifty_strangle_rules/MONDAY_20RS_STRANGLE_WEEKLY_SWEEP_STATUS.md',
      'research/90_nifty_strangle_rules/NSRW_V1_CHAIN_REPLAY_1MIN_RUN_STATUS.md',
      'research/90_nifty_strangle_rules/DAILY_0916_PREMIUM_STRANGLE_1MIN_SWEEP_STATUS.md',
      'mentor/reviews/2026-W30.md',
    ],
  },
  {
    slug: 'sensex-nifty-stop-by-dte',
    title: 'SENSEX + NIFTY 9:16 short-straddle — stop calibration by DTE (combined-premium stop vs per-leg 30% vs HOLD)',
    verdict:
      'The live 9:16 straddle systems all run a per-leg 30% stop-loss (survivor then ST(7,3)-trailed). Tested against a COMBINED-premium stop (stop only when the whole straddle is down X%) and HOLD, per DTE, on 51 real 1-min-chain days per venue: the per-leg 30% is the weak link. A combined ~15-20% stop beats it on nearly every DTE for both books — for NIFTY the current per-leg 30% is actually net-NEGATIVE (−Rs303/lot) and a combined-20% flips it to +Rs765/lot. The two venues are INVERTED and must not be copied to each other: SENSEX expiry-day (Thu) wants HOLD (+Rs3,010/lot, 91% win, tiny drawdown — any stop gives the guaranteed decay away), while NIFTY expiry-day (Tue) wants a combined-20% stop (its HOLD tail is −9.3k); SENSEX-Wednesday is a structural loser (size down / skip) while NIFTY-Monday is the sweet spot (lean in). A real-mechanics validation (modelling the actual 30%-SL → survivor ST(7,3)-trail via the live calc_supertrend) CONFIRMS every conclusion — the trail is a modest tweak, not a rescue. Directional, not yet live-ready: n is small per DTE (9–32), one regime (Apr–Aug 2026), and ATM2 (move-stop+recenter) / ATM4 (roll) mechanics are not yet modelled.',
    status: 'COMPLETE',
    date: '2026-08-13',
    cardBlurb:
      'The 9:16 straddle books all use a per-leg 30% SL — and it is the weak link. A combined-premium stop (~15-20%) beats it on nearly every DTE, both venues (NIFTY per-leg 30% is net-NEGATIVE → combined-20% flips it positive). The venues are inverted: HOLD on SENSEX-Thursday, combined-stop on NIFTY-Tuesday. Confirmed against the real ST-trail mechanic.',
    cardStats: [
      { label: 'Verdict', value: 'Combined stop > per-leg 30% (validated, directional)' },
      { label: 'NIFTY per-leg 30% → COMB-20', value: '−Rs303 → +Rs765 /lot' },
      { label: 'SENSEX expiry-day', value: 'HOLD +Rs3,010/lot, 91% win' },
    ],
    systemRules: {
      intro: 'One entry (sell the ATM straddle at 09:16, square by 15:25); the study varies only the STOP.',
      sharedCoreTitle: 'The stop variants tested',
      sharedCore: [
        { k: 'Per-leg 30% (current)', v: 'Each leg buys back when ITS premium rises 30% from entry; the surviving leg is then trailed with SuperTrend(7,3) [the real ATM mechanic].' },
        { k: 'Combined X% (the challenger)', v: 'Buy back BOTH legs only when the whole straddle premium is down X% (i.e. the net short is losing) — X swept 15/20/25/30/35/40.' },
        { k: 'HOLD', v: 'No stop — hold both legs to the 15:25 square-off.' },
        { k: 'Costs', v: 'Real 1-min option_chain fills; ~Rs290/lot NIFTY, ~Rs200/lot SENSEX (brokerage + slippage).' },
        { k: 'DTE mapping', v: 'NIFTY expiry Tuesday (DTE0=Tue, DTE1=Mon); SENSEX expiry Thursday (DTE0=Thu, DTE1=Wed).' },
      ],
      riskLayer: {
        title: 'The DTE-aware config the data points to',
        caption: 'Do NOT copy one venue to the other — their DTE0/DTE1 profiles are inverted.',
        columns: ['Venue', 'DTE0', 'DTE1', 'DTE2+'],
        rows: [
          ['NIFTY (live Mon/Tue)', 'Tue: Combined-20%', 'Mon: as-is / lean in', 'Combined-15%'],
          ['SENSEX (live Wed/Thu)', 'Thu: HOLD / loose', 'Wed: Combined-20% + small/skip', 'Combined-15%'],
        ],
        highlightRows: [0, 1],
      },
    },
    system: {
      intro: 'ATM short straddle sold at 09:16 and squared by 15:25, evaluated on the real 1-minute option_chain (options_data.db), 51 clean days per venue over Apr–Aug 2026.',
      rows: [
        { k: 'Data', v: 'options_data.db option_chain — real 1-min LTP per strike; ATM chosen by CE≈PE parity at 09:16.' },
        { k: 'Metric', v: 'Net Rs/lot after cost; win%; max drawdown of the daily equity curve.' },
        { k: 'Validation', v: 'Real ATM mechanic (30% SL → survivor ST(7,3) via the live calc_supertrend) run alongside the proxies.' },
      ],
    },
    conditions: {
      intro: 'Why the per-leg 30% is the weak link:',
      rows: [
        { k: 'Per-leg stop fires too early', v: 'It stops a leg the moment IT pops, even when the other leg is offsetting — locking a loss on a move that reverts.' },
        { k: 'Combined stop waits for the net', v: 'It only exits when the whole straddle is actually losing → dodges the whipsaw and truncates the fat tail.' },
        { k: 'Expiry day is special', v: 'On SENSEX-Thu the straddle almost always decays to profit; any stop sabotages it → HOLD. NIFTY-Tue has a bigger tail → a stop helps.' },
      ],
    },
    comparisons: [
      {
        title: 'NIFTY — net Rs/lot by stop × DTE (expiry Tue)',
        caption: 'Per-leg 30% is net-NEGATIVE overall; combined-20% is the best all-round.',
        columns: ['Stop', 'DTE0 Tue', 'DTE1 Mon', 'DTE2+', 'ALL (mean)', 'ALL maxDD'],
        rows: [
          ['HOLD', '+1,438', '+1,098', '+362', '+696', '−13,192'],
          ['Per-leg 30% (current)', '+116', '+990', '−824', '−303', '−26,450'],
          ['Combined-15%', '+820', '+1,241', '+588', '+757', '−4,689'],
          ['Combined-20%', '+1,482', '+1,167', '+431', '+761', '−5,859'],
        ],
        highlightRows: [3],
        heatmap: true,
      },
      {
        title: 'SENSEX — net Rs/lot by stop × DTE (expiry Thu)',
        caption: 'HOLD dominates DTE0 (Thu); combined-20% is the only positive on DTE1 (Wed).',
        columns: ['Stop', 'DTE0 Thu', 'DTE1 Wed', 'DTE2+', 'ALL (mean)', 'ALL maxDD'],
        rows: [
          ['HOLD', '+3,028', '−1,188', '+592', '+733', '−17,850'],
          ['Per-leg 30% (current)', '+1,206', '−137', '+391', '+453', '−11,364'],
          ['Combined-15%', '+939', '−158', '+780', '+612', '−9,450'],
          ['Combined-20%', '+796', '+77', '+661', '+564', '−8,389'],
        ],
        highlightRows: [0],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'NIFTY: per-leg 30% (ALL)', value: '−Rs303/lot', tone: 'neg' },
        { label: 'NIFTY: combined-20% (ALL)', value: '+Rs765/lot', tone: 'pos' },
        { label: 'SENSEX-Thu HOLD', value: '+Rs3,010/lot · 91% win', tone: 'pos' },
        { label: 'SENSEX-Wed (danger day)', value: 'only COMB-20 positive (+Rs77)', tone: 'neg' },
      ],
      tables: [
        {
          title: 'Real-mechanics validation — ATM_REAL (30% SL → survivor ST(7,3)) vs the proxies',
          caption: 'The live calc_supertrend trail is a modest tweak, not a rescue — combined-20% / HOLD still win. Net Rs/lot mean.',
          columns: ['Venue · DTE', 'ATM_REAL (real)', 'Per-leg 30% (proxy)', 'Combined-20%', 'HOLD'],
          rows: [
            ['NIFTY · ALL', '+73', '−298', '+765', '+701'],
            ['NIFTY · DTE0 Tue', '+611', '+116', '+1,482', '+1,438'],
            ['NIFTY · DTE2+', '−307', '−817', '+438', '+369'],
            ['SENSEX · ALL', '+276', '+402', '+560', '+729'],
            ['SENSEX · DTE0 Thu', '+1,321', '+971', '+777', '+3,010'],
            ['SENSEX · DTE1 Wed', '−245', '−137', '+77', '−1,188'],
          ],
          highlightRows: [0, 4, 5],
          heatmap: true,
        },
      ],
    },
    winners: [
      {
        config: 'Swap per-leg-30% → combined ~15-20%, plus HOLD on SENSEX-Thursday',
        summary: 'The combined-premium stop beats the current per-leg 30% (even with its real ST-trail) on nearly every DTE for both venues; SENSEX expiry day is best held.',
        metrics: [
          { k: 'NIFTY combined-20% (ALL)', v: '+Rs765/lot vs current −Rs303' },
          { k: 'SENSEX combined-15% (ALL)', v: '+Rs612/lot vs current +Rs453' },
          { k: 'SENSEX-Thu HOLD', v: '+Rs3,010/lot, 91% win, DD −Rs475' },
          { k: 'SENSEX-Wed', v: 'structural loser → size down / skip' },
        ],
        rejected: ['Per-leg 30% SL (net-negative on NIFTY)', 'Combined stop on SENSEX expiry (worse than HOLD)', 'Tight combined stops on DTE0 both venues'],
      },
    ],
    caveats: [
      'Small sample: 9–32 days per DTE cell (NIFTY DTE0/1 only ~9–10); one regime (Apr–Aug 2026).',
      'Single-entry model — the "current" is a per-leg-30% + ST(7,3)-trail model of the ATM system; ATM2 (±0.4% move-stop + recenter) and ATM4 (roll-to-match) mechanics are NOT yet modelled.',
      'Directional, NOT live-ready — validated for the ATM mechanic only. Needs the ATM2/ATM4 mechanics + another regime before any live parameter change.',
      'Consistent with research/103 (DTE0 gamma trap), 104 (NIFTY Mon sweet-spot / SENSEX Wed fat tail), 97 (30% SL bad on expiry).',
    ],
    githubLinks: [
      { label: 'Quantifyd repo', href: 'https://github.com/castroarun/Quantifyd' },
    ],
    projectPaths: [
      'memory/sensex_stop_by_dte_study.md (finding)',
    ],
  },
  {
    slug: 'csl-best-config-straddles',
    title: 'CSL best-config straddles — entry x exit x combined-SL per DTE (NIFTY + SENSEX) + paper validation',
    verdict:
      'Swept 10 entry times x 6 exits x 5 combined-SL levels per DTE per index on the raw ~3-sec chain (80 days, dwell-confirm fills). THE WINDOW IS THE EDGE: inside the right time-box the SL level (20-30%) barely binds, and time-boxed exits cut drawdowns 10-25x while keeping most of the profit (NIFTY DTE0: 09:30-11:00 keeps ~98% of full-day P&L at 1/25th the DD). Every DTE on both indices turns net-positive in its right window — including the Wednesdays that lose money held all day. Full-day holds remain correct exactly twice: NIFTY-Thursday and SENSEX-expiry-Thursday (stops only subtract there; live carries a 50% disaster backstop). Portfolio scan across CSL + NAS sleeves (corr ~0) puts the optimum at CSL-NIFTY 2u : CSL-SENSEX 1u alongside the live NAS books. Grade: STRONG IN-SAMPLE SIGNAL (grid maxima on ~15-day cells) — frozen-config PAPER books (NIFTY 12 lots + SENSEX 6 lots) run since 14-AUG-26 to earn the STRATEGY upgrade by ~mid-Sep.',
    status: 'COMPLETE',
    date: '2026-08-13',
    cardBlurb:
      'The time window is the edge: entry x exit x combined-SL sweep per DTE (3-sec dwell fills, 80 days, both indices). Right-windowed, every DTE is positive and DDs collapse 10-25x; SL level barely matters inside the window. Best configs frozen + running as paper books (NIFTY 12L + SENSEX 6L). Weekly self-refreshing Lab on /app/straddles.',
    cardStats: [
      { label: 'Key finding', value: 'Window > stop: DD cut 10-25x' },
      { label: 'NIFTY DTE0 best', value: '09:30-11:00 SL25-30 · r117.9' },
      { label: 'Validation', value: 'Paper books live since 14-AUG' },
    ],
    system: {
      intro: 'Short ATM straddle (strike at the ENTRY moment), combined-premium SL with 2-consecutive-snap dwell confirm and market exit at the next snap, time-boxed exit. Grid: entries 09:16-14:00 (10) x exits 11:00-15:20 (6) x SL 20/25/30/40/none x DTE0-4 x (NIFTY 10 lots qty 650, SENSEX 5 lots qty 100).',
      rows: [
        { k: 'Data', v: 'options_data.db raw ~3-sec chain, 80 days 2026-04-20 to 2026-08-13, both indices; per-day sparse tagging (resolution ladder).' },
        { k: 'Fills', v: 'Accepted live mechanic: breach must persist >=2 snaps, exit at NEXT snap price (not at trigger).' },
        { k: 'Home', v: 'CSL Best-Config Lab + Paper Books cards on /app/straddles — Lab self-refreshes weekly Fri 15:45 IST.' },
      ],
    },
    conditions: {
      intro: 'What the grid shows:',
      rows: [
        { k: 'The window does the risk work', v: 'Winning cells are SL-invariant (20/25/30 identical) — the time-box exits before stops bind; 30% stays as disaster backstop.' },
        { k: 'Schedule beats stop-tuning', v: 'Wednesdays (both venues) lose held-to-EOD but earn in 10:30-12:00; the two full-day holds are NIFTY-Thu and SENSEX-expiry-Thu.' },
        { k: 'Venues are inverted', v: 'NIFTY expiry (Tue) wants the morning box + stop; SENSEX expiry (Thu) wants HOLD (any stop subtracts).' },
      ],
    },
    comparisons: [
      {
        title: 'Best config per DTE (3-sec dwell, ~15 days/cell)',
        caption: 'NIFTY @10 lots (qty 650) · SENSEX @5 lots (qty 100) · totals over the cell days.',
        columns: ['Venue - DTE', 'Window', 'SL', 'Total', 'Win', 'MaxDD', 'Return/DD'],
        rows: [
          ['NIFTY DTE0 (Tue exp)', '09:30-11:00', '25-30%', '+1,98,967', '93%', '-1,687', '117.9'],
          ['NIFTY DTE1 (Mon)', '13:00-14:00', '20-30%', '+56,848', '93%', '-1,395', '40.8'],
          ['NIFTY DTE2 (Fri)', '10:00-12:00', '20-30%', '+82,003', '80%', '-1,102', '74.4'],
          ['NIFTY DTE3 (Thu)', 'FULL DAY', 'any>=20%', '+1,69,968', '90%', '-1,395', '121.8'],
          ['NIFTY DTE4 (Wed)', '10:30-12:00', '20-30%', '+46,810', '75%', '-9,614', '4.9'],
          ['SENSEX DTE0 (Thu exp)', 'FULL DAY', 'none (50% backstop live)', '+2,04,435', '93%', '-775', '263.8'],
          ['SENSEX DTE1 (Wed)', '10:30-12:00', '20-30%', '+25,785', '75%', '-5,975', '4.3'],
          ['SENSEX DTE2 (Tue)', '09:25-11:00', '20-30%', '+53,465', '93%', '-30 (artifact)', 'n/m'],
          ['SENSEX DTE3 (Mon)', '13:00-14:00', '20-30%', '+16,200', '80%', '-1,650', '9.8'],
          ['SENSEX DTE4 (Fri)', '10:30-12:00', '20-30%', '+22,590', '79%', '-865', '26.1'],
        ],
        highlightRows: [0, 3, 5],
      },
    ],
    results: {
      metrics: [
        { label: 'NIFTY best-config book (replay @6L)', value: '+Rs3.38L · DD -4.7k · r71.5', tone: 'pos' },
        { label: 'SENSEX best-config book (replay @6L)', value: '+Rs3.99L · DD -5.9k · r68.2', tone: 'pos' },
        { label: 'CSL x NAS daily correlation', value: '~ -0.07 (independent)' },
        { label: 'Optimal portfolio (scan)', value: 'CSL-N 2u : CSL-S 1u : NAS-S 1u' },
      ],
      charts: [
        { src: '/app/nifty_csl_vs_nas.png', caption: 'NIFTY — best-config CSL vs NAS-916x3 live, equity + drawdown, all @6 lots' },
        { src: '/app/sensex_csl_vs_nas.png', caption: 'SENSEX — best-config CSL vs NAS atm2 live (14d), equity + drawdown, all @6 lots' },
        { src: '/app/perleg_vs_comb.png', caption: 'Per-leg 30% vs combined 30% vs no stop — same trades, 3-sec dwell, 10 lots' },
        { src: '/app/csl30_vs_nas916.png', caption: 'Flat CSL30 vs NAS-916x3, lot-normalized @6 lots (precursor study)' },
      ],
      tables: [],
    },
    winners: [
      {
        config: 'DTE-scheduled time-boxed CSL books: NIFTY 2u (12 lots) + SENSEX 1u (6 lots), frozen 13-AUG config',
        summary: 'Deployed as PAPER (cron 09:12, dwell mechanic, 50% backstop on none-SL days) — the out-of-sample validation that decides the STRATEGY upgrade ~mid-Sep.',
        metrics: [
          { k: 'Replay ratios', v: 'NIFTY 71.5 / SENSEX 68.2 (in-sample — will degrade live)' },
          { k: 'vs live NAS ratios', v: '2.5-3.4 (their reality includes every wart)' },
          { k: 'Diversification', v: 'CSL x NAS corr ~0 -> stacking beats either' },
        ],
        rejected: ['Per-leg 30% SL (worse than no stop on identical trades)', 'Hold-to-EOD on Wednesdays', 'Equal-weight portfolio (NAS DD drags ratio 10.4 vs 80)'],
      },
    ],
    caveats: [
      'Grid maxima on ~15-16 day cells = multiple-testing; SL-invariance is the robustness signal; SENSEX DTE2 ratio is a tiny-DD artifact — not literal.',
      'WALK-FORWARD (picks on Apr-Jun, scored Jul-Aug): edge survives but degrades ~35-45%/day and OOS DDs are 5-10x bigger — robust cells: NIFTY DTE0/DTE3, SENSEX DTE0; fragile: DTE1 both venues, SENSEX DTE3. Paper books trade the FULL schedule; data decides mid-Sep.',
      'CSL numbers are in-sample optimized replay; NAS comparisons are live reality — the paper books are the arbiter, not these tables.',
      'One regime (Apr-Aug 2026); weekday-DTE mapping ignores holiday-shifted expiries; single best slot per day (2nd-slot stacking is exploratory).',
      'CPR probe (prior-day CPR width vs straddle-day outcome, ~50d/index): NO actionable edge (corr +0.06 NIFTY / +0.15 SENSEX); mild NIFTY narrow-CPR win-rate tilt (83% vs 62%) on watch-list only.',
      'Sibling study: /app/backtest/sensex-nifty-stop-by-dte (per-leg vs combined stop mechanics incl. real ST-trail validation).',
      'Portfolio sizing (sec 18/18b, 2026-08-14): TB-CSL is the overweight lever - marginal +2L of TB costs ~Rs72 of extra drawdown vs ~Rs8,669 for +2L of COMB (same profit); best grid cell LIVE6+COMB2+TB6 ratio 31.0 ex-Wed. Live rows on /app/straddles#portfolio-lab (in-sample caveat applies, TB most flattered).',
    ],
    githubLinks: [{ label: 'Quantifyd repo', href: 'https://github.com/castroarun/Quantifyd' }],
    projectPaths: [
      'research/111_sensex_manual_mgmt/SENSEX_MANUAL_STRADDLE_MGMT_FORENSIC_STATUS.md',
      'research/111_sensex_manual_mgmt/results/entry_exit_sweep.json',
      'research/111_sensex_manual_mgmt/results/deliverable3_portfolio.json',
    ],
  },

  {
    "slug": "stock-45dte-neutral-wings",
    "title": "Stock 45→21 DTE Winged Strangle — one universal ruleset across the F&O stock universe",
    "verdict": "Can the NIFTY 45-DTE window (research/119) be transplanted to single stocks? YES — as a DEFINED-RISK winged strangle with a hard liquidity gate. One ruleset, zero per-stock tuning: at 45 DTE sell the ±2.5% monthly strangle, buy wings 7% of spot away, NO stop, 50% profit target, time-exit at 21 DTE; trade only when all four legs actually traded (ATM vol ≥100, wings ≥10). On 628 liquid trades / ~70 stocks / 2016→Aug-2026 (real NSE bhavcopy EOD): net +0.264% of spot per trade at 0.5%-of-premium costs, t=+5.06, win 64.8%. ROBUSTNESS (G3) PASSED: survives dropping the top-5 names (+0.199, t=3.49); positive in every era (2016-23 +0.213 t=2.48, 2024-26 +0.290 t=4.44, 2021-24 ex-hot-years +0.168 t=2.46); edge RISES monotonically with liquidity (vol≥50 +0.108 → ≥500 +0.435 — the opposite of a stale-quote artifact); parameter plateau not peak; and the DTE-WINDOW PLACEBO is decisive — the identical structure entered at 35 DTE earns +0.02 (t=0.9) and at 55 DTE +0.06 (t=0.5): the 45→21 theta window IS the edge. Next-session entry keeps t=3.53. REFUTED along the way: 30-DTE entry (net t=-9), every premium stop (no-SL wins; wings suffice), plain IV-rank gating (not monotone), and all price-action calm gates (ADX/BB-squeeze/CPR/trend-dist — marginal). VRP=IV/RV20 IS a clean monotone signal on the crude base config but adds nothing to the optimized composite. PORTFOLIO (10 slots, entries ranked by ATM volume): at MODELED margin of 1.25×max-loss+2% (~6.7% of notional) the dense era 2021-26 shows 38.5% CAGR / -21% MaxDD / Calmar 1.81 — but DO NOT trust that row: real SPAN+exposure for stock condors is unverified and likely higher. The stressed band is the honest claim: at 1.5× margin 26.3% CAGR / -14.1% / Calmar 1.86; at 2× margin 20.2% / -10.4% / Calmar 1.94. Monthly correlation to NIFTY is -0.09 and the book averaged +1.65%/month in the 11 months NIFTY fell >3% — true diversification for a NIFTY-heavy short-vol book. GATE TO GO LIVE: measure real basket margin (Kite margin API), then paper-book the top-liquidity tier.",
    "status": "COMPLETE",
    "date": "2026-08-25",
    "cardBlurb": "The NIFTY 45→21-DTE theta window transfers to stocks — as an iron-condor-style winged strangle, one ruleset for the whole F&O universe. Net +0.264%/trade (t 5.06, 628 liquid trades, 2016-2026); DTE placebo proves the window; edge rises with liquidity. Portfolio 20-26% CAGR at stressed margin, corr to NIFTY -0.09. STRATEGY-candidate; real-margin check pending.",
    "cardStats": [
      {
        "label": "Verdict",
        "value": "STRATEGY-CANDIDATE (margin check pending)"
      },
      {
        "label": "Net/trade (liquid)",
        "value": "+0.264% S0 · t 5.06 · win 65%"
      },
      {
        "label": "Portfolio CAGR (2x-1x margin stress)",
        "value": "20-38% · Calmar 1.8-1.9"
      }
    ],
    "systemRules": {
      "intro": "One universal ruleset — no per-stock tuning. Stock selection is purely the liquidity filter.",
      "sharedCoreTitle": "The C1 ruleset (applies identically to every stock)",
      "sharedCore": [
        {
          "k": "Entry",
          "v": "45 calendar days before the monthly stock-option expiry, at EOD close (rolled back to the prior session if needed, tolerance +5d)"
        },
        {
          "k": "Structure",
          "v": "SELL CE at nearest strike to spot+2.5% and PE at spot-2.5%; BUY wing CE/PE ~7% of spot beyond each short strike (nearest traded strike)"
        },
        {
          "k": "Liquidity gate",
          "v": "All 4 legs traded that day (contracts>0); ATM legs >=100 contracts; each wing >=10. No entry otherwise — no exceptions"
        },
        {
          "k": "Exits",
          "v": "FIRST of: profit target 50% of net credit · time exit at 21 DTE. NO premium stop (tested: every stop hurts; the wings are the risk cap)"
        },
        {
          "k": "Costs modeled",
          "v": "0.5% of premium turnover (slippage+txn proxy; no bid/ask data exists for stock options EOD) — sensitivity 0.25%/1% shown"
        },
        {
          "k": "Sizing (portfolio)",
          "v": "10 slots, entries each monthly cycle ranked by ATM volume; margin per position modeled 1.25x max-loss + 2% of notional; idle cash at 5% (liquid ETF)"
        }
      ],
      "riskLayer": {
        "title": "What was optimized vs held fixed",
        "columns": [
          "Axis",
          "Swept",
          "Chosen",
          "Evidence"
        ],
        "rows": [
          [
            "Entry DTE",
            "30/40/45/50/60",
            "45",
            "30-DTE net-NEGATIVE t=-9; placebo 35/55 ≈ zero — the window is the edge"
          ],
          [
            "Exit DTE",
            "10/15/21/28",
            "21",
            "15-21 plateau, best t at 21"
          ],
          [
            "Short strikes",
            "ATM / ±2.5% / ±5%",
            "±2.5%",
            "best t (3.62); ATM close; ±5% thins credit"
          ],
          [
            "Wing width",
            "3/5/6/7/8/10% of spot",
            "7%",
            "monotone wider-better; 10% nets more with fatter tail (p05 -2.9% vs -2.0%)"
          ],
          [
            "Stop",
            "150/200/300%/none",
            "NONE",
            "no-SL beats all stops (t 3.33 vs 1.6-3.3); wings cap risk"
          ],
          [
            "Target",
            "50% / none",
            "50%",
            "removing it costs 0.03%/trade"
          ]
        ],
        "highlightRows": [
          0,
          4
        ]
      }
    },
    "system": {
      "intro": "Adapting research/119 (NIFTY 45-DTE short straddle, STRATEGY-candidate) to single stocks: stocks carry idiosyncratic overnight/news gap risk that indices diversify away, so wings are mandatory (defined risk) and liquidity is the hard screen — research/89 proved most stock-option EOD 'edges' are phantom fills in untraded strikes.",
      "rows": [
        {
          "k": "Data",
          "v": "nse_options_bhav — real NSE F&O bhavcopy EOD, 24.2M stock-option rows, 81 underlyings, 2016 → Aug-2026, volume+OI per strike"
        },
        {
          "k": "Spot / indicators",
          "v": "market_data_unified daily (used only for ATM anchor + normalization; the system needs NO price-action indicator)"
        },
        {
          "k": "Universe (data)",
          "v": "81 F&O stocks in the bhav table; effective TRADED universe after the liquidity gate ~70 names, concentrated in the most liquid tier"
        },
        {
          "k": "IV layer",
          "v": "per-stock daily ATM-IV series BS-inverted from straddle closes (results/iv_daily.csv, reusable) — used to TEST IV-rank/VRP gates, not in the final ruleset"
        },
        {
          "k": "Study lineage",
          "v": "research/119 (NIFTY 45-DTE) mechanism; research/89 liquidity discipline; research/127 = this study"
        }
      ]
    },
    "conditions": {
      "intro": "Full 81-name data universe (today's F&O list — survivorship stated):",
      "rows": [
        {
          "k": "Universe",
          "v": "ADANIENT, ADANIPORTS, AMBUJACEM, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BANKBARODA, BEL, BHARTIARTL, BPCL, BRITANNIA, CHOLAFIN, CIPLA, COALINDIA, COFORGE, COLPAL, CUMMINSIND, DABUR, DELHIVERY, DIVISLAB, DLF, DRREDDY, EICHERMOT, FEDERALBNK, GAIL, GODREJPROP, GRASIM, HAL, HAVELLS, HCLTECH, HDFCBANK, HDFCLIFE, HEROMOTOCO, HINDALCO, HINDUNILVR, ICICIBANK, IDFCFIRSTB, INDUSINDBK, INFY, IOC, IRCTC, ITC, JINDALSTEL, JSWSTEEL, KOTAKBANK, LT, M&M, MARICO, MARUTI, MCX, MUTHOOTFIN, NESTLEIND, NTPC, ONGC, PAYTM, PERSISTENT, PIDILITIND, PNB, POWERGRID, RELIANCE, SBILIFE, SBIN, SHREECEM, SIEMENS, SUNPHARMA, TATACONSUM, TATAMOTORS, TATAPOWER, TATASTEEL, TCS, TECHM, TITAN, TRENT, ULTRACEMCO, VEDL, VOLTAS, WIPRO"
        },
        {
          "k": "Period",
          "v": "2016-01 → 2026-08 (87 monthly cycles; liquid sample densifies from 2021 — pre-2021 only 1-2 tradeable names/cycle)"
        },
        {
          "k": "Configs tried",
          "v": "~31 across all phases (recorded for multiple-testing honesty); guards keep t>3 throughout"
        }
      ]
    },
    "comparisons": [
      {
        "title": "Robustness gauntlet (G3) — net @0.5% cost, liquid sample",
        "columns": [
          "Attack",
          "Result",
          "t",
          "Verdict"
        ],
        "rows": [
          [
            "C1 reference (n=628)",
            "+0.264% S0/trade",
            "+5.06",
            "—"
          ],
          [
            "Drop top-3 names (ADANIPORTS/TATAMOTORS/TCS)",
            "+0.228%",
            "+4.12",
            "PASS"
          ],
          [
            "Drop top-5 names",
            "+0.199%",
            "+3.49",
            "PASS"
          ],
          [
            "2016-2023 only",
            "+0.213%",
            "+2.48",
            "PASS"
          ],
          [
            "2024-2026 only",
            "+0.290%",
            "+4.44",
            "PASS"
          ],
          [
            "2021-2024 (ex the strong 25/26)",
            "+0.168%",
            "+2.46",
            "PASS"
          ],
          [
            "Liquidity vol>=50 / >=100 / >=200 / >=500",
            "+0.108 / +0.264 / +0.351 / +0.435",
            "3.1-5.1",
            "monotone UP — STRONG PASS"
          ],
          [
            "Same structure at 35 DTE (placebo)",
            "+0.020%",
            "+0.93",
            "window is the edge"
          ],
          [
            "Same structure at 55 DTE (placebo)",
            "+0.059%",
            "+0.54",
            "window is the edge"
          ],
          [
            "Enter NEXT session (lag test)",
            "+0.158%",
            "+3.53",
            "PASS (no close-timing artifact)"
          ]
        ],
        "highlightRows": [
          6,
          7,
          8
        ]
      },
      {
        "title": "Filters tested and their fate (entry gates on the liquid sample)",
        "columns": [
          "Filter",
          "With gate",
          "Without / opposite",
          "In ruleset?"
        ],
        "rows": [
          [
            "VRP = IV/RV20 > 1.1 (on crude base)",
            "+0.395% t=4.1",
            "+0.13-0.17%",
            "NO — adds nothing to optimized composite"
          ],
          [
            "IV rank > 0.5 (own 252d)",
            "not monotone (mid-rank best)",
            "-",
            "NO — refuted"
          ],
          [
            "Realized-vol rank calm <0.33",
            "+0.190%",
            "+0.115% hot",
            "NO — marginal"
          ],
          [
            "ADX<25 / BB-squeeze / CPR narrow / trend-dist / RSI-mid / NR7",
            "±0.02-0.08% differences",
            "-",
            "NO — the edge is structural, not timing"
          ],
          [
            "Liquidity (all legs traded + vol thresholds)",
            "the whole edge",
            "phantom fills (r/89)",
            "YES — the only gate"
          ]
        ],
        "highlightRows": [
          4
        ]
      },
      {
        "title": "Margin & sizing model (the honest weak point)",
        "columns": [
          "Assumption",
          "Margin %notional",
          "CAGR 21-26",
          "MaxDD",
          "Calmar",
          "Sharpe"
        ],
        "rows": [
          [
            "Modeled: 1.25x max-loss + 2%",
            "~6.7%",
            "38.5%",
            "-21.2%",
            "1.81",
            "1.00"
          ],
          [
            "x1.5 stress",
            "~10%",
            "26.3%",
            "-14.1%",
            "1.86",
            "0.93"
          ],
          [
            "x2.0 stress (conservative claim)",
            "~13.4%",
            "20.2%",
            "-10.4%",
            "1.94",
            "0.87"
          ]
        ],
        "highlightRows": [
          2
        ],
        "caption": "Avg max-loss per condor ~3.7% of notional (7% wing dist - ~3.3% credit). Real SPAN+exposure for stock condors is UNVERIFIED — the x1.5-x2 band is the claim until the Kite basket-margin check runs. Implied notional/slot at modeled margin is ~15x slot capital — capacity requires the top-liquidity tier."
      }
    ],
    "results": {
      "metrics": [
        {
          "label": "Net / trade (liquid)",
          "value": "+0.264% S0",
          "hint": "628 trades, 0.5% cost",
          "tone": "pos"
        },
        {
          "label": "t-stat",
          "value": "5.06",
          "hint": "3.49 after dropping top-5 names",
          "tone": "pos"
        },
        {
          "label": "Win rate",
          "value": "64.8%",
          "hint": "89% of trades reach an orderly exit (target/time)"
        },
        {
          "label": "CAGR (2x margin)",
          "value": "20.2%",
          "hint": "38.5% at modeled margin — unverified",
          "tone": "pos"
        },
        {
          "label": "MaxDD (2x margin)",
          "value": "-10.4%",
          "hint": "-21.2% at modeled margin",
          "tone": "neg"
        },
        {
          "label": "Corr vs NIFTY",
          "value": "-0.09",
          "hint": "+1.65%/mo avg in NIFTY down>3% months",
          "tone": "pos"
        }
      ],
      "tables": [
        {
          "title": "Year by year — trades (net @0.5%) and portfolio return (modeled margin)",
          "columns": [
            "Year",
            "Trades",
            "Net/trade",
            "t",
            "Win",
            "p05",
            "Portfolio yr"
          ],
          "rows": [
            [
              "2016",
              "5",
              "+0.476%",
              "+0.54",
              "60%",
              "-1.25%",
              "+5.0%"
            ],
            [
              "2017",
              "1",
              "+0.790%",
              "-",
              "100%",
              "+0.79%",
              "+1.0%"
            ],
            [
              "2018",
              "6",
              "-0.871%",
              "-1.12",
              "50%",
              "-3.74%",
              "-2.9%"
            ],
            [
              "2019",
              "5",
              "-0.274%",
              "-0.40",
              "60%",
              "-2.20%",
              "-1.2%"
            ],
            [
              "2020",
              "11",
              "-0.434%",
              "-1.30",
              "45%",
              "-2.35%",
              "-1.5%"
            ],
            [
              "2021",
              "55",
              "+0.401%",
              "+2.18",
              "71%",
              "-2.01%",
              "+28.8%"
            ],
            [
              "2022",
              "63",
              "+0.099%",
              "+0.70",
              "60%",
              "-1.58%",
              "+9.7%"
            ],
            [
              "2023",
              "69",
              "+0.372%",
              "+2.71",
              "75%",
              "-2.06%",
              "+23.2%"
            ],
            [
              "2024",
              "133",
              "-0.001%",
              "-0.01",
              "58%",
              "-1.99%",
              "+10.3%"
            ],
            [
              "2025",
              "165",
              "+0.286%",
              "+3.30",
              "67%",
              "-1.83%",
              "+20.1%"
            ],
            [
              "2026",
              "115",
              "+0.632%",
              "+4.21",
              "66%",
              "-1.84%",
              "+25.1%"
            ]
          ],
          "heatmap": true
        },
        {
          "title": "Per-symbol (n>=5, liquid sample) — the effective traded universe",
          "columns": [
            "Symbol",
            "Trades",
            "Net/trade",
            "Win",
            "Avg ATM vol"
          ],
          "rows": [
            [
              "PNB",
              "6",
              "+1.284%",
              "83%",
              "200"
            ],
            [
              "FEDERALBNK",
              "6",
              "+1.243%",
              "100%",
              "305"
            ],
            [
              "ADANIPORTS",
              "14",
              "+1.187%",
              "79%",
              "227"
            ],
            [
              "NTPC",
              "7",
              "+1.073%",
              "100%",
              "238"
            ],
            [
              "POWERGRID",
              "5",
              "+1.007%",
              "100%",
              "177"
            ],
            [
              "HCLTECH",
              "10",
              "+0.913%",
              "80%",
              "289"
            ],
            [
              "MCX",
              "6",
              "+0.847%",
              "67%",
              "244"
            ],
            [
              "HINDALCO",
              "12",
              "+0.833%",
              "83%",
              "229"
            ],
            [
              "WIPRO",
              "7",
              "+0.797%",
              "86%",
              "392"
            ],
            [
              "ITC",
              "9",
              "+0.779%",
              "89%",
              "295"
            ],
            [
              "TATAPOWER",
              "16",
              "+0.737%",
              "81%",
              "226"
            ],
            [
              "BAJFINANCE",
              "5",
              "+0.704%",
              "80%",
              "465"
            ],
            [
              "ADANIENT",
              "19",
              "+0.517%",
              "63%",
              "362"
            ],
            [
              "ASIANPAINT",
              "14",
              "+0.400%",
              "64%",
              "236"
            ],
            [
              "TCS",
              "32",
              "+0.371%",
              "69%",
              "505"
            ],
            [
              "TATAMOTORS",
              "34",
              "+0.351%",
              "82%",
              "584"
            ],
            [
              "BHARTIARTL",
              "9",
              "+0.293%",
              "56%",
              "308"
            ],
            [
              "TITAN",
              "6",
              "+0.212%",
              "67%",
              "221"
            ],
            [
              "HAL",
              "20",
              "+0.197%",
              "60%",
              "279"
            ],
            [
              "SBIN",
              "45",
              "+0.168%",
              "67%",
              "316"
            ],
            [
              "M&M",
              "16",
              "+0.165%",
              "69%",
              "185"
            ],
            [
              "LT",
              "23",
              "+0.163%",
              "61%",
              "320"
            ],
            [
              "ICICIBANK",
              "20",
              "+0.157%",
              "70%",
              "253"
            ],
            [
              "TATASTEEL",
              "13",
              "+0.149%",
              "54%",
              "216"
            ],
            [
              "IDFCFIRSTB",
              "8",
              "+0.102%",
              "50%",
              "304"
            ],
            [
              "INDUSINDBK",
              "16",
              "+0.063%",
              "69%",
              "254"
            ],
            [
              "HDFCBANK",
              "7",
              "+0.055%",
              "43%",
              "938"
            ],
            [
              "INFY",
              "34",
              "+0.044%",
              "50%",
              "628"
            ],
            [
              "RELIANCE",
              "19",
              "+0.024%",
              "58%",
              "757"
            ],
            [
              "BEL",
              "14",
              "+0.007%",
              "43%",
              "281"
            ],
            [
              "IRCTC",
              "7",
              "-0.022%",
              "71%",
              "127"
            ],
            [
              "HINDUNILVR",
              "21",
              "-0.071%",
              "52%",
              "242"
            ],
            [
              "KOTAKBANK",
              "17",
              "-0.076%",
              "59%",
              "237"
            ],
            [
              "AXISBANK",
              "11",
              "-0.201%",
              "45%",
              "225"
            ],
            [
              "MARUTI",
              "20",
              "-0.278%",
              "45%",
              "239"
            ],
            [
              "BPCL",
              "5",
              "-0.328%",
              "40%",
              "186"
            ],
            [
              "VEDL",
              "12",
              "-0.413%",
              "42%",
              "318"
            ],
            [
              "TRENT",
              "9",
              "-0.718%",
              "33%",
              "178"
            ]
          ],
          "heatmap": true
        },
        {
          "title": "Full trade log — all 628 liquid C1 trades (net @0.5% cost)",
          "columns": [
            "Entry",
            "Exit",
            "Symbol",
            "Expiry",
            "Spot",
            "Shorts PE/CE",
            "Wings PE/CE",
            "Credit",
            "Exit via",
            "Gross",
            "Net"
          ],
          "rows": [
            [
              "2016-02-15",
              "2016-03-10",
              "SBIN",
              "2016-03-31",
              "168",
              "160/170",
              "150/180",
              "4.20%",
              "time",
              "+0.21%",
              "+0.05%"
            ],
            [
              "2016-05-16",
              "2016-06-09",
              "AXISBANK",
              "2016-06-30",
              "486",
              "470/500",
              "450/520",
              "2.08%",
              "time",
              "-0.86%",
              "-0.99%"
            ],
            [
              "2016-05-16",
              "2016-06-09",
              "SBIN",
              "2016-06-30",
              "177",
              "170/180",
              "160/190",
              "3.68%",
              "time",
              "-1.10%",
              "-1.32%"
            ],
            [
              "2016-07-11",
              "2016-08-04",
              "SBIN",
              "2016-08-25",
              "225",
              "220/230",
              "200/250",
              "5.38%",
              "time",
              "+1.14%",
              "+1.05%"
            ],
            [
              "2016-08-12",
              "2016-08-25",
              "SBIN",
              "2016-09-29",
              "243",
              "235/250",
              "220/270",
              "7.09%",
              "target",
              "+3.68%",
              "+3.58%"
            ],
            [
              "2017-02-13",
              "2017-03-09",
              "SBIN",
              "2017-03-30",
              "272",
              "265/280",
              "250/300",
              "3.42%",
              "time",
              "+0.85%",
              "+0.79%"
            ],
            [
              "2018-04-16",
              "2018-05-10",
              "MARUTI",
              "2018-05-31",
              "9,232",
              "9,000/9,500",
              "8,600/10,000",
              "2.42%",
              "time",
              "-0.14%",
              "-0.19%"
            ],
            [
              "2018-04-16",
              "2018-05-10",
              "TATAMOTORS",
              "2018-05-31",
              "339",
              "330/350",
              "310/370",
              "3.42%",
              "time",
              "+0.40%",
              "+0.31%"
            ],
            [
              "2018-07-16",
              "2018-08-09",
              "ICICIBANK",
              "2018-08-30",
              "259",
              "250/270",
              "230/290",
              "4.13%",
              "time",
              "-4.32%",
              "-4.58%"
            ],
            [
              "2018-07-16",
              "2018-08-09",
              "SUNPHARMA",
              "2018-08-30",
              "534",
              "520/540",
              "500/580",
              "3.54%",
              "time",
              "-1.13%",
              "-1.25%"
            ],
            [
              "2018-08-13",
              "2018-09-06",
              "SBIN",
              "2018-09-27",
              "294",
              "290/300",
              "270/320",
              "3.84%",
              "time",
              "+0.44%",
              "+0.36%"
            ],
            [
              "2018-12-17",
              "2019-01-10",
              "TATAMOTORS",
              "2019-01-31",
              "174",
              "170/180",
              "160/190",
              "3.82%",
              "time",
              "+0.23%",
              "+0.12%"
            ],
            [
              "2019-04-15",
              "2019-05-09",
              "INFY",
              "2019-05-30",
              "728",
              "710/750",
              "660/800",
              "3.26%",
              "time",
              "+0.87%",
              "+0.82%"
            ],
            [
              "2019-04-15",
              "2019-05-30",
              "TATAMOTORS",
              "2019-05-30",
              "232",
              "225/240",
              "210/255",
              "4.32%",
              "expiry",
              "-2.16%",
              "-2.42%"
            ],
            [
              "2019-07-15",
              "2019-08-08",
              "SUNPHARMA",
              "2019-08-29",
              "423",
              "410/430",
              "380/460",
              "4.13%",
              "time",
              "+0.54%",
              "+0.46%"
            ],
            [
              "2019-09-16",
              "2019-10-10",
              "SBIN",
              "2019-10-31",
              "285",
              "280/290",
              "260/310",
              "4.34%",
              "time",
              "-1.19%",
              "-1.33%"
            ],
            [
              "2019-12-16",
              "2020-01-09",
              "SBIN",
              "2020-01-30",
              "332",
              "325/340",
              "300/360",
              "4.01%",
              "time",
              "+1.16%",
              "+1.08%"
            ],
            [
              "2020-03-16",
              "2020-04-09",
              "SBIN",
              "2020-04-30",
              "223",
              "220/230",
              "200/250",
              "7.19%",
              "time",
              "-0.87%",
              "-1.26%"
            ],
            [
              "2020-06-15",
              "2020-07-09",
              "SBIN",
              "2020-07-30",
              "174",
              "170/180",
              "160/190",
              "4.52%",
              "time",
              "-0.06%",
              "-0.30%"
            ],
            [
              "2020-08-10",
              "2020-09-03",
              "SBIN",
              "2020-09-24",
              "194",
              "190/200",
              "175/215",
              "5.39%",
              "time",
              "+0.34%",
              "+0.18%"
            ],
            [
              "2020-09-14",
              "2020-10-08",
              "TATAMOTORS",
              "2020-10-29",
              "149",
              "145/150",
              "135/160",
              "5.22%",
              "time",
              "+0.20%",
              "+0.00%"
            ],
            [
              "2020-09-14",
              "2020-10-08",
              "ICICIBANK",
              "2020-10-29",
              "364",
              "350/370",
              "320/400",
              "5.18%",
              "time",
              "+0.27%",
              "+0.15%"
            ],
            [
              "2020-09-14",
              "2020-10-08",
              "SBIN",
              "2020-10-29",
              "198",
              "195/205",
              "180/220",
              "5.26%",
              "time",
              "+1.13%",
              "+1.01%"
            ],
            [
              "2020-09-14",
              "2020-10-08",
              "BHARTIARTL",
              "2020-10-29",
              "465",
              "450/480",
              "420/510",
              "4.13%",
              "time",
              "-0.13%",
              "-0.27%"
            ],
            [
              "2020-11-13",
              "2020-12-10",
              "LT",
              "2020-12-31",
              "1,052",
              "1,020/1,080",
              "940/1,160",
              "3.97%",
              "time",
              "-2.13%",
              "-2.25%"
            ],
            [
              "2020-11-13",
              "2020-12-10",
              "MARUTI",
              "2020-12-31",
              "6,809",
              "6,600/7,000",
              "6,200/7,500",
              "3.68%",
              "time",
              "-2.31%",
              "-2.45%"
            ],
            [
              "2020-12-14",
              "2021-01-07",
              "SBIN",
              "2021-01-28",
              "274",
              "265/280",
              "245/300",
              "4.76%",
              "time",
              "+0.77%",
              "+0.65%"
            ],
            [
              "2020-12-14",
              "2021-01-07",
              "INFY",
              "2021-01-28",
              "1,165",
              "1,140/1,200",
              "1,060/1,300",
              "4.98%",
              "time",
              "-0.12%",
              "-0.24%"
            ],
            [
              "2021-01-11",
              "2021-02-04",
              "LT",
              "2021-02-25",
              "1,350",
              "1,300/1,380",
              "1,200/1,480",
              "4.28%",
              "time",
              "-0.58%",
              "-0.72%"
            ],
            [
              "2021-01-11",
              "2021-02-04",
              "TCS",
              "2021-02-25",
              "3,176",
              "3,100/3,300",
              "2,900/3,500",
              "3.35%",
              "time",
              "+1.08%",
              "+1.02%"
            ],
            [
              "2021-01-11",
              "2021-02-04",
              "SBIN",
              "2021-02-25",
              "282",
              "280/290",
              "260/310",
              "5.17%",
              "time",
              "-1.79%",
              "-2.09%"
            ],
            [
              "2021-02-08",
              "2021-03-04",
              "INDUSINDBK",
              "2021-03-25",
              "1,035",
              "1,000/1,060",
              "900/1,140",
              "5.28%",
              "time",
              "+0.35%",
              "+0.19%"
            ],
            [
              "2021-02-08",
              "2021-03-04",
              "HINDUNILVR",
              "2021-03-25",
              "2,237",
              "2,200/2,300",
              "2,000/2,500",
              "4.29%",
              "time",
              "+1.58%",
              "+1.53%"
            ],
            [
              "2021-03-15",
              "2021-04-08",
              "ICICIBANK",
              "2021-04-29",
              "604",
              "590/620",
              "550/660",
              "4.16%",
              "time",
              "+0.74%",
              "+0.64%"
            ],
            [
              "2021-03-15",
              "2021-04-08",
              "INFY",
              "2021-04-29",
              "1,374",
              "1,320/1,400",
              "1,240/1,500",
              "3.78%",
              "time",
              "+0.04%",
              "-0.05%"
            ],
            [
              "2021-04-12",
              "2021-05-06",
              "BHARTIARTL",
              "2021-05-27",
              "513",
              "500/530",
              "450/570",
              "4.32%",
              "time",
              "-0.62%",
              "-0.72%"
            ],
            [
              "2021-04-12",
              "2021-05-06",
              "TCS",
              "2021-05-27",
              "3,247",
              "3,200/3,300",
              "3,000/3,500",
              "4.12%",
              "time",
              "+1.01%",
              "+0.93%"
            ],
            [
              "2021-04-12",
              "2021-05-06",
              "INDUSINDBK",
              "2021-05-27",
              "844",
              "800/840",
              "700/900",
              "7.25%",
              "time",
              "+1.59%",
              "+1.38%"
            ],
            [
              "2021-04-12",
              "2021-05-06",
              "TATAMOTORS",
              "2021-05-27",
              "287",
              "280/300",
              "260/320",
              "5.03%",
              "time",
              "+0.96%",
              "+0.80%"
            ],
            [
              "2021-04-12",
              "2021-05-06",
              "INFY",
              "2021-05-27",
              "1,426",
              "1,400/1,460",
              "1,300/1,520",
              "4.12%",
              "time",
              "+0.87%",
              "+0.78%"
            ],
            [
              "2021-04-12",
              "2021-05-06",
              "LT",
              "2021-05-27",
              "1,345",
              "1,300/1,400",
              "1,200/1,500",
              "3.55%",
              "time",
              "+1.29%",
              "+1.24%"
            ],
            [
              "2021-05-10",
              "2021-06-03",
              "INFY",
              "2021-06-24",
              "1,340",
              "1,300/1,360",
              "1,200/1,500",
              "3.65%",
              "time",
              "+0.12%",
              "+0.07%"
            ],
            [
              "2021-05-10",
              "2021-06-03",
              "SBIN",
              "2021-06-24",
              "362",
              "350/370",
              "320/400",
              "5.22%",
              "time",
              "-2.24%",
              "-2.47%"
            ],
            [
              "2021-06-14",
              "2021-07-08",
              "ICICIBANK",
              "2021-07-29",
              "635",
              "620/650",
              "580/700",
              "3.78%",
              "time",
              "+1.61%",
              "+1.55%"
            ],
            [
              "2021-06-14",
              "2021-07-08",
              "SBIN",
              "2021-07-29",
              "430",
              "420/440",
              "390/470",
              "4.40%",
              "time",
              "+1.71%",
              "+1.62%"
            ],
            [
              "2021-06-14",
              "2021-07-08",
              "INFY",
              "2021-07-29",
              "1,462",
              "1,440/1,500",
              "1,340/1,600",
              "3.60%",
              "time",
              "-0.31%",
              "-0.38%"
            ],
            [
              "2021-06-14",
              "2021-07-08",
              "TATAPOWER",
              "2021-07-29",
              "123",
              "120/125",
              "110/135",
              "6.00%",
              "time",
              "+1.78%",
              "+1.63%"
            ],
            [
              "2021-06-14",
              "2021-07-08",
              "KOTAKBANK",
              "2021-07-29",
              "1,768",
              "1,700/1,820",
              "1,600/1,900",
              "2.49%",
              "time",
              "+1.01%",
              "+0.96%"
            ],
            [
              "2021-06-14",
              "2021-07-08",
              "TCS",
              "2021-07-29",
              "3,276",
              "3,200/3,350",
              "3,000/3,600",
              "3.40%",
              "time",
              "+1.03%",
              "+0.98%"
            ],
            [
              "2021-06-14",
              "2021-07-08",
              "MARUTI",
              "2021-07-29",
              "7,178",
              "7,000/7,400",
              "6,500/8,000",
              "3.11%",
              "time",
              "+0.47%",
              "+0.42%"
            ],
            [
              "2021-06-14",
              "2021-07-08",
              "INDUSINDBK",
              "2021-07-29",
              "1,024",
              "1,000/1,040",
              "900/1,100",
              "4.74%",
              "time",
              "+1.46%",
              "+1.35%"
            ],
            [
              "2021-06-14",
              "2021-07-08",
              "ADANIENT",
              "2021-07-29",
              "1,501",
              "1,500/1,540",
              "1,400/1,640",
              "3.04%",
              "time",
              "-1.76%",
              "-1.98%"
            ],
            [
              "2021-06-14",
              "2021-06-18",
              "ADANIPORTS",
              "2021-07-29",
              "768",
              "750/780",
              "700/830",
              "6.59%",
              "target",
              "+6.87%",
              "+6.58%"
            ],
            [
              "2021-07-12",
              "2021-07-26",
              "TCS",
              "2021-08-26",
              "3,193",
              "3,100/3,280",
              "2,900/3,500",
              "3.12%",
              "target",
              "+1.64%",
              "+1.60%"
            ],
            [
              "2021-07-12",
              "2021-08-05",
              "TATAPOWER",
              "2021-08-26",
              "124",
              "120/125",
              "110/135",
              "4.69%",
              "time",
              "-0.53%",
              "-0.67%"
            ],
            [
              "2021-07-12",
              "2021-08-05",
              "INFY",
              "2021-08-26",
              "1,548",
              "1,500/1,580",
              "1,400/1,700",
              "3.18%",
              "time",
              "-1.29%",
              "-1.36%"
            ],
            [
              "2021-07-12",
              "2021-08-05",
              "TATAMOTORS",
              "2021-08-26",
              "307",
              "300/320",
              "280/340",
              "4.16%",
              "time",
              "+0.99%",
              "+0.89%"
            ],
            [
              "2021-07-12",
              "2021-08-05",
              "SBIN",
              "2021-08-26",
              "427",
              "420/440",
              "390/470",
              "3.66%",
              "time",
              "+0.63%",
              "+0.56%"
            ],
            [
              "2021-07-12",
              "2021-08-05",
              "INDUSINDBK",
              "2021-08-26",
              "1,049",
              "1,000/1,100",
              "900/1,200",
              "3.72%",
              "time",
              "+0.71%",
              "+0.65%"
            ],
            [
              "2021-08-16",
              "2021-09-09",
              "HINDALCO",
              "2021-09-30",
              "442",
              "430/450",
              "400/500",
              "5.17%",
              "time",
              "+0.21%",
              "+0.12%"
            ],
            [
              "2021-08-16",
              "2021-09-09",
              "MARUTI",
              "2021-09-30",
              "6,827",
              "6,700/7,000",
              "6,400/7,500",
              "2.90%",
              "time",
              "+0.91%",
              "+0.86%"
            ],
            [
              "2021-08-16",
              "2021-09-09",
              "INFY",
              "2021-09-30",
              "1,704",
              "1,660/1,740",
              "1,540/1,880",
              "2.42%",
              "time",
              "+0.68%",
              "+0.65%"
            ],
            [
              "2021-08-16",
              "2021-09-09",
              "TATAMOTORS",
              "2021-09-30",
              "304",
              "300/310",
              "280/330",
              "4.41%",
              "time",
              "+0.49%",
              "+0.39%"
            ],
            [
              "2021-08-16",
              "2021-09-09",
              "HINDUNILVR",
              "2021-09-30",
              "2,426",
              "2,340/2,500",
              "2,200/2,600",
              "2.28%",
              "time",
              "-1.67%",
              "-1.80%"
            ],
            [
              "2021-08-16",
              "2021-09-09",
              "ICICIBANK",
              "2021-09-30",
              "703",
              "690/720",
              "650/750",
              "2.28%",
              "time",
              "+0.53%",
              "+0.48%"
            ],
            [
              "2021-09-13",
              "2021-10-07",
              "KOTAKBANK",
              "2021-10-28",
              "1,840",
              "1,800/1,900",
              "1,700/2,000",
              "3.05%",
              "time",
              "+0.01%",
              "-0.08%"
            ],
            [
              "2021-10-11",
              "2021-11-04",
              "SBIN",
              "2021-11-25",
              "469",
              "460/480",
              "430/520",
              "4.76%",
              "time",
              "-2.10%",
              "-2.26%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "LT",
              "2021-12-30",
              "1,947",
              "1,900/2,000",
              "1,800/2,100",
              "2.43%",
              "time",
              "+0.42%",
              "+0.37%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "HCLTECH",
              "2021-12-30",
              "1,171",
              "1,140/1,200",
              "1,100/1,300",
              "2.52%",
              "time",
              "+1.00%",
              "+0.95%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "M&M",
              "2021-12-30",
              "928",
              "900/950",
              "850/1,000",
              "2.46%",
              "time",
              "-1.29%",
              "-1.37%"
            ],
            [
              "2021-11-15",
              "2021-11-26",
              "TITAN",
              "2021-12-30",
              "2,539",
              "2,500/2,600",
              "2,400/2,700",
              "2.25%",
              "target",
              "+1.27%",
              "+1.16%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "HINDUNILVR",
              "2021-12-30",
              "2,425",
              "2,400/2,500",
              "2,200/2,700",
              "3.15%",
              "time",
              "+0.19%",
              "+0.15%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "HINDALCO",
              "2021-12-30",
              "456",
              "440/470",
              "400/500",
              "4.11%",
              "target",
              "+2.14%",
              "+2.07%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "ICICIBANK",
              "2021-12-30",
              "773",
              "750/790",
              "700/840",
              "3.01%",
              "time",
              "+1.16%",
              "+1.11%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "ASIANPAINT",
              "2021-12-30",
              "3,169",
              "3,100/3,200",
              "2,900/3,400",
              "3.24%",
              "time",
              "+0.73%",
              "+0.67%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "TCS",
              "2021-12-30",
              "3,553",
              "3,440/3,640",
              "3,200/3,900",
              "2.29%",
              "time",
              "+0.77%",
              "+0.74%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "SBIN",
              "2021-12-30",
              "507",
              "490/520",
              "450/550",
              "3.39%",
              "time",
              "+0.96%",
              "+0.90%"
            ],
            [
              "2021-11-15",
              "2021-12-09",
              "TECHM",
              "2021-12-30",
              "1,523",
              "1,500/1,600",
              "1,400/1,700",
              "3.29%",
              "time",
              "+0.54%",
              "+0.48%"
            ],
            [
              "2021-12-13",
              "2022-01-06",
              "INFY",
              "2022-01-27",
              "1,745",
              "1,700/1,780",
              "1,600/1,900",
              "2.95%",
              "time",
              "-0.66%",
              "-0.72%"
            ],
            [
              "2021-12-13",
              "2022-01-06",
              "ASIANPAINT",
              "2022-01-27",
              "3,280",
              "3,200/3,400",
              "3,000/3,600",
              "2.84%",
              "time",
              "-0.51%",
              "-0.58%"
            ],
            [
              "2021-12-13",
              "2022-01-06",
              "HINDUNILVR",
              "2022-01-27",
              "2,304",
              "2,200/2,360",
              "2,000/2,500",
              "2.28%",
              "time",
              "-0.04%",
              "-0.08%"
            ],
            [
              "2021-12-13",
              "2022-01-06",
              "AXISBANK",
              "2022-01-27",
              "704",
              "680/720",
              "650/750",
              "2.44%",
              "time",
              "+0.21%",
              "+0.14%"
            ],
            [
              "2021-12-13",
              "2022-01-06",
              "SBIN",
              "2022-01-27",
              "488",
              "480/500",
              "450/530",
              "3.43%",
              "time",
              "+0.89%",
              "+0.82%"
            ],
            [
              "2022-01-10",
              "2022-02-02",
              "TCS",
              "2022-02-24",
              "3,880",
              "3,800/4,000",
              "3,500/4,280",
              "3.67%",
              "target",
              "+1.92%",
              "+1.87%"
            ],
            [
              "2022-01-10",
              "2022-02-03",
              "PNB",
              "2022-02-24",
              "39",
              "38/40",
              "35/42",
              "4.08%",
              "time",
              "+0.38%",
              "+0.22%"
            ],
            [
              "2022-02-14",
              "2022-03-10",
              "KOTAKBANK",
              "2022-03-31",
              "1,747",
              "1,700/1,800",
              "1,600/1,940",
              "3.34%",
              "time",
              "+0.52%",
              "+0.46%"
            ],
            [
              "2022-02-14",
              "2022-03-10",
              "HINDALCO",
              "2022-03-31",
              "521",
              "500/530",
              "450/560",
              "3.48%",
              "time",
              "+0.37%",
              "+0.23%"
            ],
            [
              "2022-02-14",
              "2022-03-10",
              "ICICIBANK",
              "2022-03-31",
              "754",
              "740/770",
              "680/820",
              "4.10%",
              "time",
              "-1.73%",
              "-1.83%"
            ],
            [
              "2022-02-14",
              "2022-03-10",
              "TATAMOTORS",
              "2022-03-31",
              "471",
              "460/480",
              "430/510",
              "4.44%",
              "time",
              "-0.55%",
              "-0.70%"
            ],
            [
              "2022-02-14",
              "2022-03-10",
              "IDFCFIRSTB",
              "2022-03-31",
              "44",
              "44/45",
              "40/48",
              "4.68%",
              "time",
              "-0.91%",
              "-1.03%"
            ],
            [
              "2022-02-14",
              "2022-03-10",
              "ADANIENT",
              "2022-03-31",
              "1,664",
              "1,600/1,700",
              "1,500/1,800",
              "3.95%",
              "time",
              "+0.42%",
              "+0.31%"
            ],
            [
              "2022-02-14",
              "2022-03-10",
              "INFY",
              "2022-03-31",
              "1,682",
              "1,640/1,720",
              "1,500/1,840",
              "4.49%",
              "time",
              "-0.69%",
              "-0.78%"
            ],
            [
              "2022-02-14",
              "2022-03-10",
              "M&M",
              "2022-03-31",
              "825",
              "800/850",
              "750/900",
              "3.25%",
              "time",
              "-1.49%",
              "-1.58%"
            ],
            [
              "2022-02-14",
              "2022-03-10",
              "HINDUNILVR",
              "2022-03-31",
              "2,229",
              "2,180/2,300",
              "2,000/2,500",
              "2.65%",
              "time",
              "-1.26%",
              "-1.31%"
            ],
            [
              "2022-03-14",
              "2022-04-07",
              "SBIN",
              "2022-04-28",
              "485",
              "470/500",
              "440/530",
              "3.91%",
              "time",
              "+0.28%",
              "+0.18%"
            ],
            [
              "2022-03-14",
              "2022-04-07",
              "ASIANPAINT",
              "2022-04-28",
              "2,963",
              "2,900/3,000",
              "2,700/3,200",
              "4.15%",
              "time",
              "-0.34%",
              "-0.45%"
            ],
            [
              "2022-03-14",
              "2022-04-07",
              "TCS",
              "2022-04-28",
              "3,643",
              "3,520/3,700",
              "3,300/4,000",
              "3.84%",
              "time",
              "+0.83%",
              "+0.76%"
            ],
            [
              "2022-04-11",
              "2022-05-05",
              "TCS",
              "2022-05-26",
              "3,696",
              "3,600/3,800",
              "3,300/4,100",
              "3.51%",
              "time",
              "+0.31%",
              "+0.26%"
            ],
            [
              "2022-04-11",
              "2022-05-05",
              "HINDUNILVR",
              "2022-05-26",
              "2,163",
              "2,100/2,200",
              "1,900/2,400",
              "3.66%",
              "time",
              "+1.47%",
              "+1.42%"
            ],
            [
              "2022-05-16",
              "2022-06-09",
              "SBIN",
              "2022-06-30",
              "455",
              "440/470",
              "410/500",
              "3.45%",
              "time",
              "+1.08%",
              "+1.01%"
            ],
            [
              "2022-05-16",
              "2022-06-09",
              "ADANIENT",
              "2022-06-30",
              "2,106",
              "2,100/2,200",
              "2,000/2,300",
              "3.66%",
              "time",
              "+0.56%",
              "+0.43%"
            ],
            [
              "2022-05-16",
              "2022-06-09",
              "TATAMOTORS",
              "2022-06-30",
              "405",
              "400/420",
              "370/450",
              "4.69%",
              "time",
              "+0.73%",
              "+0.61%"
            ],
            [
              "2022-05-16",
              "2022-05-20",
              "AMBUJACEM",
              "2022-06-30",
              "368",
              "360/375",
              "330/400",
              "3.82%",
              "target",
              "+2.13%",
              "+2.07%"
            ],
            [
              "2022-05-16",
              "2022-06-09",
              "TCS",
              "2022-06-30",
              "3,377",
              "3,300/3,450",
              "3,100/3,700",
              "3.43%",
              "time",
              "+1.22%",
              "+1.17%"
            ],
            [
              "2022-05-16",
              "2022-06-09",
              "HINDALCO",
              "2022-06-30",
              "391",
              "380/400",
              "350/430",
              "5.60%",
              "time",
              "+1.71%",
              "+1.59%"
            ],
            [
              "2022-06-13",
              "2022-07-07",
              "TATAMOTORS",
              "2022-07-28",
              "407",
              "400/420",
              "370/450",
              "4.71%",
              "time",
              "+0.54%",
              "+0.42%"
            ],
            [
              "2022-06-13",
              "2022-07-07",
              "SBIN",
              "2022-07-28",
              "446",
              "430/460",
              "400/490",
              "3.68%",
              "time",
              "-0.91%",
              "-1.00%"
            ],
            [
              "2022-06-13",
              "2022-07-07",
              "KOTAKBANK",
              "2022-07-28",
              "1,737",
              "1,700/1,780",
              "1,600/1,900",
              "3.84%",
              "time",
              "+1.48%",
              "+1.41%"
            ],
            [
              "2022-07-11",
              "2022-08-02",
              "HCLTECH",
              "2022-08-25",
              "944",
              "900/980",
              "860/1,000",
              "2.07%",
              "target",
              "+1.10%",
              "+1.04%"
            ],
            [
              "2022-07-11",
              "2022-08-04",
              "AXISBANK",
              "2022-08-25",
              "680",
              "660/700",
              "600/750",
              "3.25%",
              "time",
              "-0.73%",
              "-0.80%"
            ],
            [
              "2022-07-11",
              "2022-08-04",
              "TCS",
              "2022-08-25",
              "3,114",
              "3,000/3,200",
              "2,800/3,400",
              "2.84%",
              "time",
              "-1.44%",
              "-1.51%"
            ],
            [
              "2022-07-11",
              "2022-08-04",
              "KOTAKBANK",
              "2022-08-25",
              "1,741",
              "1,700/1,800",
              "1,600/1,900",
              "3.05%",
              "time",
              "+0.18%",
              "+0.11%"
            ],
            [
              "2022-07-11",
              "2022-08-04",
              "LT",
              "2022-08-25",
              "1,663",
              "1,600/1,700",
              "1,500/1,800",
              "2.54%",
              "time",
              "-1.49%",
              "-1.55%"
            ],
            [
              "2022-07-11",
              "2022-08-04",
              "TATAMOTORS",
              "2022-08-25",
              "437",
              "430/450",
              "400/480",
              "4.54%",
              "time",
              "+0.30%",
              "+0.18%"
            ],
            [
              "2022-08-12",
              "2022-09-08",
              "TATASTEEL",
              "2022-09-29",
              "107",
              "104/110",
              "94/118",
              "4.97%",
              "time",
              "+1.03%",
              "+0.91%"
            ],
            [
              "2022-08-12",
              "2022-09-08",
              "KOTAKBANK",
              "2022-09-29",
              "1,841",
              "1,800/1,900",
              "1,700/2,000",
              "2.84%",
              "time",
              "+0.02%",
              "-0.05%"
            ],
            [
              "2022-08-12",
              "2022-09-08",
              "LT",
              "2022-09-29",
              "1,846",
              "1,800/1,900",
              "1,700/2,000",
              "2.57%",
              "time",
              "-0.67%",
              "-0.73%"
            ],
            [
              "2022-08-12",
              "2022-09-08",
              "M&M",
              "2022-09-29",
              "1,259",
              "1,220/1,300",
              "1,100/1,400",
              "3.94%",
              "time",
              "+0.66%",
              "+0.59%"
            ],
            [
              "2022-08-12",
              "2022-09-08",
              "MARUTI",
              "2022-09-29",
              "8,699",
              "8,500/8,900",
              "8,000/9,500",
              "3.54%",
              "time",
              "+0.88%",
              "+0.82%"
            ],
            [
              "2022-08-12",
              "2022-09-08",
              "SBIN",
              "2022-09-29",
              "531",
              "520/540",
              "480/580",
              "4.16%",
              "time",
              "+0.88%",
              "+0.81%"
            ],
            [
              "2022-08-12",
              "2022-09-08",
              "ASIANPAINT",
              "2022-09-29",
              "3,428",
              "3,300/3,500",
              "3,100/3,700",
              "2.84%",
              "time",
              "+0.71%",
              "+0.66%"
            ],
            [
              "2022-08-12",
              "2022-09-08",
              "TCS",
              "2022-09-29",
              "3,402",
              "3,320/3,500",
              "3,100/3,700",
              "2.55%",
              "time",
              "-1.27%",
              "-1.33%"
            ],
            [
              "2022-08-12",
              "2022-09-08",
              "INFY",
              "2022-09-29",
              "1,594",
              "1,560/1,640",
              "1,460/1,760",
              "3.26%",
              "time",
              "-0.91%",
              "-0.98%"
            ],
            [
              "2022-09-12",
              "2022-10-06",
              "INFY",
              "2022-10-27",
              "1,536",
              "1,500/1,580",
              "1,400/1,680",
              "3.60%",
              "time",
              "-0.33%",
              "-0.40%"
            ],
            [
              "2022-09-12",
              "2022-10-06",
              "TCS",
              "2022-10-27",
              "3,243",
              "3,200/3,300",
              "3,000/3,500",
              "3.55%",
              "time",
              "-0.08%",
              "-0.16%"
            ],
            [
              "2022-09-12",
              "2022-10-06",
              "PNB",
              "2022-10-27",
              "39",
              "38/40",
              "35/43",
              "4.75%",
              "time",
              "+0.00%",
              "-0.13%"
            ],
            [
              "2022-10-10",
              "2022-11-03",
              "TCS",
              "2022-11-24",
              "3,119",
              "3,080/3,200",
              "2,900/3,400",
              "4.21%",
              "time",
              "+2.02%",
              "+1.97%"
            ],
            [
              "2022-10-10",
              "2022-11-03",
              "INDUSINDBK",
              "2022-11-24",
              "1,210",
              "1,200/1,250",
              "1,100/1,300",
              "3.81%",
              "time",
              "-0.57%",
              "-0.67%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "SBIN",
              "2022-12-29",
              "593",
              "580/605",
              "540/650",
              "3.86%",
              "time",
              "+1.09%",
              "+1.04%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "ASIANPAINT",
              "2022-12-29",
              "3,053",
              "3,000/3,150",
              "2,800/3,400",
              "3.14%",
              "time",
              "-0.42%",
              "-0.47%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "BAJAJFINSV",
              "2022-12-29",
              "1,718",
              "1,700/1,800",
              "1,600/1,900",
              "3.30%",
              "time",
              "-0.39%",
              "-0.47%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "MARUTI",
              "2022-12-29",
              "9,152",
              "9,000/9,400",
              "8,500/10,000",
              "3.19%",
              "time",
              "+0.27%",
              "+0.21%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "GODREJPROP",
              "2022-12-29",
              "1,310",
              "1,300/1,300",
              "1,200/1,400",
              "5.67%",
              "time",
              "+0.96%",
              "+0.85%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "TCS",
              "2022-12-29",
              "3,336",
              "3,300/3,400",
              "3,100/3,600",
              "2.81%",
              "time",
              "+0.77%",
              "+0.73%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "DIVISLAB",
              "2022-12-29",
              "3,276",
              "3,200/3,400",
              "3,000/3,600",
              "3.18%",
              "target",
              "+1.69%",
              "+1.64%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "TATAMOTORS",
              "2022-12-29",
              "434",
              "425/440",
              "390/470",
              "4.28%",
              "time",
              "+0.67%",
              "+0.59%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "ADANIENT",
              "2022-12-29",
              "4,023",
              "3,900/4,100",
              "3,600/4,400",
              "4.46%",
              "time",
              "+1.09%",
              "+1.00%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "HINDUNILVR",
              "2022-12-29",
              "2,458",
              "2,400/2,500",
              "2,300/2,700",
              "2.92%",
              "time",
              "-3.57%",
              "-3.65%"
            ],
            [
              "2022-11-14",
              "2022-12-08",
              "TATASTEEL",
              "2022-12-29",
              "103",
              "100/105",
              "95/112",
              "4.14%",
              "time",
              "-0.78%",
              "-0.91%"
            ],
            [
              "2022-12-09",
              "2023-01-04",
              "TATAMOTORS",
              "2023-01-25",
              "413",
              "400/420",
              "380/450",
              "3.74%",
              "time",
              "+0.68%",
              "+0.59%"
            ],
            [
              "2022-12-09",
              "2023-01-04",
              "ICICIBANK",
              "2023-01-25",
              "930",
              "900/950",
              "850/1,000",
              "2.58%",
              "time",
              "+0.94%",
              "+0.90%"
            ],
            [
              "2022-12-09",
              "2023-01-04",
              "BANKBARODA",
              "2023-01-25",
              "175",
              "170/180",
              "160/190",
              "4.09%",
              "time",
              "+0.77%",
              "+0.64%"
            ],
            [
              "2022-12-09",
              "2023-01-04",
              "IRCTC",
              "2023-01-25",
              "722",
              "700/750",
              "650/800",
              "3.30%",
              "time",
              "-2.48%",
              "-2.58%"
            ],
            [
              "2022-12-09",
              "2023-01-04",
              "SBIN",
              "2023-01-25",
              "616",
              "600/630",
              "550/670",
              "2.89%",
              "time",
              "+0.85%",
              "+0.81%"
            ],
            [
              "2022-12-09",
              "2023-01-04",
              "DIVISLAB",
              "2023-01-25",
              "3,274",
              "3,200/3,400",
              "3,000/3,600",
              "2.39%",
              "time",
              "-0.23%",
              "-0.28%"
            ],
            [
              "2022-12-09",
              "2023-01-04",
              "TCS",
              "2023-01-25",
              "3,293",
              "3,200/3,380",
              "3,000/3,600",
              "3.05%",
              "time",
              "+1.14%",
              "+1.10%"
            ],
            [
              "2023-01-09",
              "2023-02-02",
              "ICICIBANK",
              "2023-02-23",
              "873",
              "850/900",
              "800/950",
              "2.75%",
              "time",
              "+1.03%",
              "+0.98%"
            ],
            [
              "2023-01-09",
              "2023-02-02",
              "TCS",
              "2023-02-23",
              "3,320",
              "3,200/3,400",
              "3,000/3,600",
              "2.58%",
              "time",
              "-0.23%",
              "-0.27%"
            ],
            [
              "2023-01-09",
              "2023-02-02",
              "BHARTIARTL",
              "2023-02-23",
              "819",
              "800/850",
              "750/900",
              "2.46%",
              "time",
              "-0.20%",
              "-0.25%"
            ],
            [
              "2023-02-10",
              "2023-03-08",
              "HINDUNILVR",
              "2023-03-29",
              "2,578",
              "2,500/2,600",
              "2,400/2,800",
              "2.85%",
              "time",
              "+1.42%",
              "+1.39%"
            ],
            [
              "2023-02-10",
              "2023-03-08",
              "MARUTI",
              "2023-03-29",
              "8,811",
              "8,500/9,000",
              "8,000/9,500",
              "2.40%",
              "target",
              "+1.33%",
              "+1.30%"
            ],
            [
              "2023-02-10",
              "2023-03-08",
              "DIVISLAB",
              "2023-03-29",
              "2,811",
              "2,700/2,800",
              "2,500/3,000",
              "3.87%",
              "time",
              "+0.87%",
              "+0.81%"
            ],
            [
              "2023-02-10",
              "2023-03-08",
              "TATAPOWER",
              "2023-03-29",
              "205",
              "200/210",
              "190/225",
              "3.22%",
              "time",
              "+0.56%",
              "+0.50%"
            ],
            [
              "2023-02-10",
              "2023-03-08",
              "IRCTC",
              "2023-03-29",
              "644",
              "630/650",
              "600/700",
              "3.60%",
              "time",
              "+1.20%",
              "+1.13%"
            ],
            [
              "2023-02-10",
              "2023-03-08",
              "HINDALCO",
              "2023-03-29",
              "433",
              "420/440",
              "400/470",
              "3.57%",
              "time",
              "+0.80%",
              "+0.71%"
            ],
            [
              "2023-02-10",
              "2023-03-03",
              "BHARTIARTL",
              "2023-03-29",
              "772",
              "750/800",
              "700/850",
              "2.18%",
              "target",
              "+1.11%",
              "+1.08%"
            ],
            [
              "2023-02-10",
              "2023-03-08",
              "TATASTEEL",
              "2023-03-29",
              "103",
              "100/105",
              "95/112",
              "4.23%",
              "time",
              "+0.92%",
              "+0.83%"
            ],
            [
              "2023-02-10",
              "2023-03-08",
              "TATAMOTORS",
              "2023-03-29",
              "446",
              "430/455",
              "400/490",
              "3.09%",
              "time",
              "+1.03%",
              "+0.98%"
            ],
            [
              "2023-03-13",
              "2023-04-06",
              "INFY",
              "2023-04-27",
              "1,435",
              "1,400/1,480",
              "1,300/1,580",
              "3.43%",
              "time",
              "+1.35%",
              "+1.30%"
            ],
            [
              "2023-03-13",
              "2023-04-06",
              "TATAMOTORS",
              "2023-04-27",
              "422",
              "410/430",
              "380/460",
              "3.94%",
              "time",
              "+0.79%",
              "+0.73%"
            ],
            [
              "2023-03-13",
              "2023-04-06",
              "ADANIPORTS",
              "2023-04-27",
              "681",
              "650/700",
              "600/750",
              "4.50%",
              "time",
              "+1.18%",
              "+1.07%"
            ],
            [
              "2023-03-13",
              "2023-04-06",
              "LT",
              "2023-04-27",
              "2,134",
              "2,080/2,200",
              "2,000/2,300",
              "1.80%",
              "time",
              "-1.27%",
              "-1.33%"
            ],
            [
              "2023-03-13",
              "2023-04-06",
              "TCS",
              "2023-04-27",
              "3,282",
              "3,200/3,340",
              "3,000/3,600",
              "3.18%",
              "time",
              "+1.35%",
              "+1.31%"
            ],
            [
              "2023-04-10",
              "2023-05-04",
              "LT",
              "2023-05-25",
              "2,310",
              "2,280/2,300",
              "2,100/2,500",
              "4.37%",
              "time",
              "+0.26%",
              "+0.20%"
            ],
            [
              "2023-04-10",
              "2023-05-04",
              "ADANIENT",
              "2023-05-25",
              "1,797",
              "1,800/1,800",
              "1,700/1,900",
              "5.22%",
              "time",
              "+0.44%",
              "+0.16%"
            ],
            [
              "2023-04-10",
              "2023-05-04",
              "HINDUNILVR",
              "2023-05-25",
              "2,532",
              "2,500/2,600",
              "2,400/2,700",
              "2.15%",
              "time",
              "+0.98%",
              "+0.94%"
            ],
            [
              "2023-04-10",
              "2023-05-02",
              "TCS",
              "2023-05-25",
              "3,263",
              "3,200/3,300",
              "3,000/3,500",
              "3.11%",
              "target",
              "+1.62%",
              "+1.58%"
            ],
            [
              "2023-04-10",
              "2023-05-04",
              "TATAMOTORS",
              "2023-05-25",
              "461",
              "450/470",
              "420/500",
              "3.70%",
              "time",
              "+0.29%",
              "+0.21%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "TATAPOWER",
              "2023-06-29",
              "208",
              "202/215",
              "190/230",
              "2.52%",
              "time",
              "-0.82%",
              "-0.87%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "IRCTC",
              "2023-06-29",
              "632",
              "620/650",
              "600/700",
              "2.98%",
              "time",
              "+0.84%",
              "+0.78%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "DABUR",
              "2023-06-29",
              "533",
              "520/550",
              "500/600",
              "1.80%",
              "time",
              "-0.58%",
              "-0.61%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "FEDERALBNK",
              "2023-06-29",
              "128",
              "125/130",
              "120/140",
              "3.06%",
              "time",
              "+0.78%",
              "+0.72%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "TATAMOTORS",
              "2023-06-29",
              "531",
              "520/545",
              "480/580",
              "3.51%",
              "time",
              "+0.15%",
              "+0.08%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "INFY",
              "2023-06-29",
              "1,258",
              "1,220/1,280",
              "1,160/1,360",
              "2.54%",
              "time",
              "+0.40%",
              "+0.37%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "MARUTI",
              "2023-06-29",
              "9,207",
              "9,000/9,500",
              "8,500/10,000",
              "1.97%",
              "time",
              "-0.42%",
              "-0.46%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "HINDALCO",
              "2023-06-29",
              "411",
              "400/420",
              "370/450",
              "4.14%",
              "time",
              "+1.22%",
              "+1.15%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "HINDUNILVR",
              "2023-06-29",
              "2,662",
              "2,600/2,600",
              "2,400/2,900",
              "4.28%",
              "time",
              "+0.44%",
              "+0.40%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "ASIANPAINT",
              "2023-06-29",
              "3,132",
              "3,100/3,200",
              "2,900/3,400",
              "2.93%",
              "time",
              "+0.81%",
              "+0.77%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "AXISBANK",
              "2023-06-29",
              "916",
              "900/940",
              "840/1,000",
              "2.67%",
              "time",
              "-0.51%",
              "-0.55%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "BHARTIARTL",
              "2023-06-29",
              "797",
              "780/820",
              "700/900",
              "3.14%",
              "time",
              "+0.11%",
              "+0.07%"
            ],
            [
              "2023-05-15",
              "2023-06-08",
              "ADANIENT",
              "2023-06-29",
              "1,916",
              "1,900/1,950",
              "1,800/2,100",
              "5.57%",
              "time",
              "-2.76%",
              "-3.14%"
            ],
            [
              "2023-06-12",
              "2023-07-06",
              "INFY",
              "2023-07-27",
              "1,292",
              "1,260/1,320",
              "1,160/1,400",
              "2.69%",
              "time",
              "-0.43%",
              "-0.48%"
            ],
            [
              "2023-06-12",
              "2023-07-06",
              "TATAMOTORS",
              "2023-07-27",
              "564",
              "550/580",
              "520/620",
              "3.37%",
              "time",
              "-0.53%",
              "-0.61%"
            ],
            [
              "2023-06-12",
              "2023-07-06",
              "TCS",
              "2023-07-27",
              "3,247",
              "3,200/3,340",
              "3,000/3,500",
              "2.35%",
              "time",
              "+0.49%",
              "+0.45%"
            ],
            [
              "2023-07-17",
              "2023-08-08",
              "BHARTIARTL",
              "2023-08-31",
              "878",
              "860/900",
              "800/950",
              "2.76%",
              "target",
              "+1.48%",
              "+1.45%"
            ],
            [
              "2023-07-17",
              "2023-08-10",
              "HINDALCO",
              "2023-08-31",
              "447",
              "440/460",
              "410/490",
              "3.87%",
              "time",
              "+0.91%",
              "+0.84%"
            ],
            [
              "2023-07-17",
              "2023-08-10",
              "FEDERALBNK",
              "2023-08-31",
              "133",
              "130/135",
              "120/145",
              "3.92%",
              "time",
              "+1.28%",
              "+1.22%"
            ],
            [
              "2023-07-17",
              "2023-08-10",
              "HINDUNILVR",
              "2023-08-31",
              "2,682",
              "2,600/2,740",
              "2,500/2,900",
              "2.33%",
              "time",
              "+0.27%",
              "+0.23%"
            ],
            [
              "2023-07-17",
              "2023-08-10",
              "ADANIENT",
              "2023-08-31",
              "2,409",
              "2,300/2,450",
              "2,200/2,600",
              "3.50%",
              "time",
              "-0.18%",
              "-0.35%"
            ],
            [
              "2023-07-17",
              "2023-08-10",
              "ADANIPORTS",
              "2023-08-31",
              "731",
              "700/750",
              "650/800",
              "3.41%",
              "time",
              "-1.76%",
              "-1.86%"
            ],
            [
              "2023-07-17",
              "2023-08-07",
              "ICICIBANK",
              "2023-08-31",
              "969",
              "940/990",
              "850/1,050",
              "3.09%",
              "target",
              "+1.63%",
              "+1.60%"
            ],
            [
              "2023-08-14",
              "2023-09-07",
              "TATAPOWER",
              "2023-09-28",
              "231",
              "225/235",
              "210/250",
              "3.63%",
              "time",
              "-2.51%",
              "-2.66%"
            ],
            [
              "2023-08-14",
              "2023-09-07",
              "TCS",
              "2023-09-28",
              "3,450",
              "3,400/3,500",
              "3,200/3,700",
              "2.77%",
              "time",
              "+1.25%",
              "+1.22%"
            ],
            [
              "2023-08-14",
              "2023-09-07",
              "ADANIPORTS",
              "2023-09-28",
              "787",
              "770/800",
              "700/850",
              "5.41%",
              "time",
              "+1.95%",
              "+1.85%"
            ],
            [
              "2023-09-11",
              "2023-10-05",
              "KOTAKBANK",
              "2023-10-26",
              "1,808",
              "1,800/1,820",
              "1,700/1,940",
              "3.80%",
              "time",
              "+0.50%",
              "+0.44%"
            ],
            [
              "2023-09-11",
              "2023-10-05",
              "TATAMOTORS",
              "2023-10-26",
              "635",
              "620/650",
              "580/700",
              "3.73%",
              "time",
              "+1.30%",
              "+1.24%"
            ],
            [
              "2023-09-11",
              "2023-10-05",
              "INFY",
              "2023-10-26",
              "1,476",
              "1,450/1,500",
              "1,350/1,600",
              "3.08%",
              "time",
              "+0.30%",
              "+0.25%"
            ],
            [
              "2023-10-16",
              "2023-11-09",
              "TATAMOTORS",
              "2023-11-30",
              "666",
              "650/680",
              "600/725",
              "3.65%",
              "time",
              "+1.22%",
              "+1.16%"
            ],
            [
              "2023-10-16",
              "2023-11-02",
              "HCLTECH",
              "2023-11-30",
              "1,271",
              "1,240/1,300",
              "1,150/1,400",
              "2.48%",
              "target",
              "+1.26%",
              "+1.23%"
            ],
            [
              "2023-10-16",
              "2023-11-09",
              "HAL",
              "2023-11-30",
              "1,963",
              "1,925/2,000",
              "1,800/2,100",
              "3.28%",
              "time",
              "+0.91%",
              "+0.84%"
            ],
            [
              "2023-10-16",
              "2023-11-03",
              "INDUSINDBK",
              "2023-11-30",
              "1,446",
              "1,400/1,500",
              "1,300/1,600",
              "3.00%",
              "target",
              "+1.58%",
              "+1.54%"
            ],
            [
              "2023-10-16",
              "2023-11-03",
              "NTPC",
              "2023-11-30",
              "243",
              "235/250",
              "220/260",
              "2.14%",
              "target",
              "+1.11%",
              "+1.08%"
            ],
            [
              "2023-10-16",
              "2023-11-09",
              "FEDERALBNK",
              "2023-11-30",
              "149",
              "145/152",
              "135/160",
              "2.62%",
              "time",
              "+0.91%",
              "+0.86%"
            ],
            [
              "2023-10-16",
              "2023-11-09",
              "MCX",
              "2023-11-30",
              "2,197",
              "2,100/2,300",
              "1,900/2,400",
              "3.35%",
              "time",
              "-0.29%",
              "-0.40%"
            ],
            [
              "2023-10-16",
              "2023-11-06",
              "KOTAKBANK",
              "2023-11-30",
              "1,750",
              "1,700/1,800",
              "1,600/1,900",
              "2.05%",
              "target",
              "+1.14%",
              "+1.12%"
            ],
            [
              "2023-10-16",
              "2023-11-06",
              "SBIN",
              "2023-11-30",
              "576",
              "560/590",
              "530/630",
              "2.63%",
              "target",
              "+1.33%",
              "+1.30%"
            ],
            [
              "2023-10-16",
              "2023-11-09",
              "ADANIENT",
              "2023-11-30",
              "2,429",
              "2,400/2,500",
              "2,200/2,700",
              "5.52%",
              "time",
              "-0.26%",
              "-0.39%"
            ],
            [
              "2023-10-16",
              "2023-11-03",
              "HINDUNILVR",
              "2023-11-30",
              "2,558",
              "2,500/2,600",
              "2,300/2,800",
              "2.15%",
              "target",
              "+1.12%",
              "+1.10%"
            ],
            [
              "2023-10-16",
              "2023-10-31",
              "HDFCLIFE",
              "2023-11-30",
              "630",
              "600/650",
              "580/700",
              "2.01%",
              "target",
              "+1.02%",
              "+0.99%"
            ],
            [
              "2023-11-13",
              "2023-12-07",
              "TATAPOWER",
              "2023-12-28",
              "257",
              "250/262",
              "230/280",
              "3.09%",
              "time",
              "+0.27%",
              "+0.04%"
            ],
            [
              "2023-11-13",
              "2023-12-07",
              "AXISBANK",
              "2023-12-28",
              "1,025",
              "1,000/1,050",
              "950/1,140",
              "2.12%",
              "time",
              "-4.00%",
              "-4.07%"
            ],
            [
              "2023-12-11",
              "2024-01-04",
              "HAL",
              "2024-01-25",
              "2,792",
              "2,700/3,000",
              "2,500/3,150",
              "3.41%",
              "time",
              "+1.05%",
              "+0.95%"
            ],
            [
              "2023-12-11",
              "2024-01-04",
              "TATAMOTORS",
              "2024-01-25",
              "721",
              "700/740",
              "650/800",
              "3.72%",
              "time",
              "-2.10%",
              "-2.19%"
            ],
            [
              "2023-12-11",
              "2024-01-04",
              "ICICIBANK",
              "2024-01-25",
              "1,017",
              "990/1,040",
              "900/1,100",
              "2.59%",
              "time",
              "+0.26%",
              "+0.22%"
            ],
            [
              "2023-12-11",
              "2024-01-04",
              "IDFCFIRSTB",
              "2024-01-25",
              "87",
              "85/90",
              "80/95",
              "3.95%",
              "time",
              "+1.49%",
              "+1.40%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "HDFCLIFE",
              "2024-02-29",
              "614",
              "600/630",
              "550/670",
              "3.61%",
              "time",
              "+0.49%",
              "+0.43%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "IDFCFIRSTB",
              "2024-02-29",
              "87",
              "85/89",
              "80/95",
              "4.15%",
              "time",
              "-0.06%",
              "-0.17%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "ICICIBANK",
              "2024-02-29",
              "1,010",
              "980/1,040",
              "950/1,100",
              "2.01%",
              "time",
              "+0.57%",
              "+0.53%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "ADANIENT",
              "2024-02-29",
              "3,090",
              "3,000/3,200",
              "2,800/3,400",
              "4.55%",
              "time",
              "+1.81%",
              "+1.69%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "INFY",
              "2024-02-29",
              "1,652",
              "1,620/1,690",
              "1,500/1,810",
              "2.93%",
              "time",
              "+0.55%",
              "+0.51%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "MARUTI",
              "2024-02-29",
              "10,087",
              "9,800/10,300",
              "9,000/11,000",
              "3.21%",
              "time",
              "-0.91%",
              "-0.97%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "HAL",
              "2024-02-29",
              "3,031",
              "3,000/3,100",
              "2,800/3,300",
              "4.45%",
              "time",
              "+0.37%",
              "+0.25%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "TCS",
              "2024-02-29",
              "3,904",
              "3,800/4,000",
              "3,500/4,300",
              "2.50%",
              "time",
              "-1.42%",
              "-1.47%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "JSWSTEEL",
              "2024-02-29",
              "825",
              "800/850",
              "750/900",
              "3.38%",
              "time",
              "+0.98%",
              "+0.91%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "TECHM",
              "2024-02-29",
              "1,286",
              "1,250/1,300",
              "1,160/1,400",
              "4.46%",
              "time",
              "+1.03%",
              "+0.94%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "BEL",
              "2024-02-29",
              "189",
              "185/190",
              "170/200",
              "4.69%",
              "time",
              "+0.56%",
              "+0.44%"
            ],
            [
              "2024-01-15",
              "2024-02-09",
              "GAIL",
              "2024-02-29",
              "154",
              "150/160",
              "140/170",
              "4.19%",
              "time",
              "-0.49%",
              "-0.64%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "FEDERALBNK",
              "2024-02-29",
              "153",
              "150/155",
              "140/165",
              "4.22%",
              "time",
              "+0.82%",
              "+0.74%"
            ],
            [
              "2024-01-15",
              "2024-02-08",
              "KOTAKBANK",
              "2024-02-29",
              "1,852",
              "1,800/1,900",
              "1,700/2,000",
              "2.28%",
              "time",
              "-0.83%",
              "-0.88%"
            ],
            [
              "2024-02-12",
              "2024-03-07",
              "HAL",
              "2024-03-28",
              "2,846",
              "2,800/2,900",
              "2,600/3,100",
              "5.12%",
              "time",
              "-1.60%",
              "-1.81%"
            ],
            [
              "2024-02-12",
              "2024-03-07",
              "LT",
              "2024-03-28",
              "3,300",
              "3,200/3,400",
              "3,000/3,600",
              "2.75%",
              "time",
              "-2.25%",
              "-2.35%"
            ],
            [
              "2024-02-12",
              "2024-03-07",
              "ASIANPAINT",
              "2024-03-28",
              "2,954",
              "2,900/3,000",
              "2,700/3,200",
              "3.42%",
              "time",
              "+0.88%",
              "+0.83%"
            ],
            [
              "2024-02-12",
              "2024-03-07",
              "MARUTI",
              "2024-03-28",
              "10,710",
              "10,500/11,000",
              "10,000/11,500",
              "2.19%",
              "time",
              "-1.21%",
              "-1.28%"
            ],
            [
              "2024-03-11",
              "2024-04-04",
              "TATASTEEL",
              "2024-04-25",
              "146",
              "145/150",
              "135/160",
              "4.30%",
              "time",
              "-1.23%",
              "-1.38%"
            ],
            [
              "2024-03-11",
              "2024-04-04",
              "SBIN",
              "2024-04-25",
              "774",
              "750/790",
              "700/840",
              "3.46%",
              "time",
              "+1.37%",
              "+1.32%"
            ],
            [
              "2024-03-11",
              "2024-04-04",
              "LT",
              "2024-04-25",
              "3,641",
              "3,600/3,700",
              "3,400/4,000",
              "3.77%",
              "time",
              "+0.02%",
              "-0.04%"
            ],
            [
              "2024-03-11",
              "2024-04-04",
              "TATAPOWER",
              "2024-04-25",
              "413",
              "400/425",
              "370/450",
              "4.58%",
              "time",
              "+1.88%",
              "+1.78%"
            ],
            [
              "2024-03-11",
              "2024-04-04",
              "IRCTC",
              "2024-04-25",
              "939",
              "900/950",
              "850/1,000",
              "3.61%",
              "time",
              "+0.14%",
              "+0.02%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "HINDALCO",
              "2024-05-30",
              "613",
              "595/630",
              "550/670",
              "3.59%",
              "time",
              "+0.38%",
              "+0.31%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "TATAPOWER",
              "2024-05-30",
              "432",
              "420/440",
              "390/470",
              "4.11%",
              "time",
              "+0.71%",
              "+0.62%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "HCLTECH",
              "2024-05-30",
              "1,505",
              "1,460/1,560",
              "1,400/1,700",
              "3.39%",
              "time",
              "-0.22%",
              "-0.33%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "BEL",
              "2024-05-30",
              "234",
              "230/240",
              "215/255",
              "4.12%",
              "time",
              "+0.08%",
              "-0.03%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "NTPC",
              "2024-05-30",
              "361",
              "350/370",
              "320/400",
              "4.06%",
              "time",
              "+1.05%",
              "+0.99%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "ASIANPAINT",
              "2024-05-30",
              "2,844",
              "2,800/2,900",
              "2,600/3,100",
              "3.22%",
              "time",
              "-0.59%",
              "-0.65%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "LT",
              "2024-05-30",
              "3,601",
              "3,500/3,700",
              "3,400/4,000",
              "2.56%",
              "time",
              "+0.18%",
              "+0.10%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "BAJAJFINSV",
              "2024-05-30",
              "1,657",
              "1,600/1,700",
              "1,500/1,800",
              "3.11%",
              "time",
              "+0.50%",
              "+0.44%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "ADANIENT",
              "2024-05-30",
              "3,147",
              "3,100/3,200",
              "2,900/3,400",
              "4.46%",
              "time",
              "-0.70%",
              "-0.85%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "KOTAKBANK",
              "2024-05-30",
              "1,798",
              "1,750/1,840",
              "1,600/1,960",
              "2.67%",
              "time",
              "-2.56%",
              "-2.62%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "IDFCFIRSTB",
              "2024-05-30",
              "83",
              "81/85",
              "75/90",
              "3.93%",
              "time",
              "-1.27%",
              "-1.38%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "ICICIBANK",
              "2024-05-30",
              "1,078",
              "1,050/1,110",
              "960/1,200",
              "3.07%",
              "time",
              "+0.45%",
              "+0.42%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "TATAMOTORS",
              "2024-05-30",
              "999",
              "970/1,020",
              "900/1,080",
              "3.30%",
              "time",
              "+0.14%",
              "+0.06%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "INFY",
              "2024-05-30",
              "1,468",
              "1,440/1,500",
              "1,340/1,600",
              "3.91%",
              "target",
              "+1.96%",
              "+1.90%"
            ],
            [
              "2024-04-15",
              "2024-05-09",
              "HINDUNILVR",
              "2024-05-30",
              "2,194",
              "2,160/2,240",
              "2,000/2,400",
              "2.87%",
              "time",
              "-1.23%",
              "-1.28%"
            ],
            [
              "2024-05-13",
              "2024-06-06",
              "TATAPOWER",
              "2024-06-27",
              "412",
              "400/420",
              "370/450",
              "5.10%",
              "time",
              "+1.10%",
              "+0.98%"
            ],
            [
              "2024-05-13",
              "2024-06-06",
              "LT",
              "2024-06-27",
              "3,294",
              "3,200/3,400",
              "3,000/3,600",
              "3.60%",
              "time",
              "+0.41%",
              "+0.32%"
            ],
            [
              "2024-05-13",
              "2024-06-06",
              "AMBUJACEM",
              "2024-06-27",
              "588",
              "570/600",
              "520/640",
              "5.24%",
              "time",
              "+1.14%",
              "+1.00%"
            ],
            [
              "2024-05-13",
              "2024-06-05",
              "MARUTI",
              "2024-06-27",
              "12,674",
              "12,500/13,000",
              "11,500/14,000",
              "4.96%",
              "target",
              "+2.64%",
              "+2.57%"
            ],
            [
              "2024-05-13",
              "2024-06-06",
              "TATAMOTORS",
              "2024-06-27",
              "960",
              "940/980",
              "880/1,050",
              "4.65%",
              "time",
              "+1.67%",
              "+1.57%"
            ],
            [
              "2024-05-13",
              "2024-06-06",
              "IDFCFIRSTB",
              "2024-06-27",
              "77",
              "75/80",
              "70/85",
              "3.95%",
              "time",
              "+1.43%",
              "+1.33%"
            ],
            [
              "2024-05-13",
              "2024-06-06",
              "ASIANPAINT",
              "2024-06-27",
              "2,879",
              "2,800/2,960",
              "2,600/3,200",
              "3.76%",
              "time",
              "+1.77%",
              "+1.71%"
            ],
            [
              "2024-05-13",
              "2024-06-06",
              "HAL",
              "2024-06-27",
              "3,922",
              "3,800/4,000",
              "3,500/4,300",
              "5.61%",
              "time",
              "-1.04%",
              "-1.31%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "MARUTI",
              "2024-07-25",
              "12,718",
              "12,500/13,000",
              "12,000/14,000",
              "2.71%",
              "time",
              "+0.21%",
              "+0.15%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "GAIL",
              "2024-07-25",
              "201",
              "200/210",
              "190/220",
              "3.21%",
              "time",
              "-0.10%",
              "-0.24%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "TCS",
              "2024-07-25",
              "3,859",
              "3,750/4,000",
              "3,500/4,300",
              "2.71%",
              "time",
              "+0.29%",
              "+0.25%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "ASIANPAINT",
              "2024-07-25",
              "2,938",
              "2,900/3,000",
              "2,700/3,200",
              "3.32%",
              "time",
              "+0.85%",
              "+0.81%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "ADANIENT",
              "2024-07-25",
              "3,220",
              "3,100/3,300",
              "2,900/3,500",
              "3.83%",
              "time",
              "+1.21%",
              "+1.09%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "LT",
              "2024-07-25",
              "3,544",
              "3,480/3,600",
              "3,200/3,800",
              "4.75%",
              "time",
              "+2.19%",
              "+2.12%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "INFY",
              "2024-07-25",
              "1,500",
              "1,460/1,540",
              "1,360/1,640",
              "3.19%",
              "time",
              "-1.76%",
              "-1.85%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "ADANIPORTS",
              "2024-07-25",
              "1,384",
              "1,300/1,420",
              "1,200/1,500",
              "4.19%",
              "time",
              "+0.21%",
              "+0.08%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "HCLTECH",
              "2024-07-25",
              "1,419",
              "1,400/1,500",
              "1,300/1,600",
              "2.94%",
              "time",
              "+0.24%",
              "+0.19%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "KOTAKBANK",
              "2024-07-25",
              "1,746",
              "1,700/1,800",
              "1,600/1,900",
              "1.92%",
              "time",
              "-0.95%",
              "-0.99%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "IDFCFIRSTB",
              "2024-07-25",
              "78",
              "76/80",
              "70/85",
              "3.93%",
              "time",
              "+0.77%",
              "+0.69%"
            ],
            [
              "2024-06-10",
              "2024-07-04",
              "BEL",
              "2024-07-25",
              "283",
              "280/290",
              "260/310",
              "5.26%",
              "time",
              "-0.18%",
              "-0.36%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "ICICIBANK",
              "2024-08-29",
              "1,230",
              "1,200/1,260",
              "1,100/1,340",
              "3.22%",
              "time",
              "-0.39%",
              "-0.45%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "TATAPOWER",
              "2024-08-29",
              "439",
              "430/450",
              "400/480",
              "4.37%",
              "time",
              "+0.81%",
              "+0.72%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "IOC",
              "2024-08-29",
              "166",
              "163/170",
              "153/183",
              "4.09%",
              "time",
              "+1.14%",
              "+1.06%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "KOTAKBANK",
              "2024-08-29",
              "1,844",
              "1,800/1,880",
              "1,700/2,000",
              "3.24%",
              "time",
              "+1.07%",
              "+1.03%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "SBIN",
              "2024-08-29",
              "881",
              "860/900",
              "800/960",
              "3.79%",
              "time",
              "-0.89%",
              "-0.97%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "ADANIPORTS",
              "2024-08-29",
              "1,495",
              "1,480/1,520",
              "1,400/1,600",
              "3.70%",
              "time",
              "+0.91%",
              "+0.82%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "INDUSINDBK",
              "2024-08-29",
              "1,444",
              "1,400/1,500",
              "1,300/1,600",
              "3.23%",
              "time",
              "-0.14%",
              "-0.20%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "PNB",
              "2024-08-29",
              "118",
              "115/120",
              "105/130",
              "5.06%",
              "time",
              "+1.49%",
              "+1.39%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "FEDERALBNK",
              "2024-08-29",
              "195",
              "190/200",
              "175/215",
              "4.18%",
              "time",
              "+1.77%",
              "+1.71%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "TATAMOTORS",
              "2024-08-29",
              "1,024",
              "1,000/1,050",
              "930/1,120",
              "3.62%",
              "time",
              "+1.23%",
              "+1.17%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "GAIL",
              "2024-08-29",
              "229",
              "225/235",
              "210/250",
              "4.38%",
              "time",
              "+1.29%",
              "+1.19%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "BPCL",
              "2024-08-29",
              "298",
              "290/310",
              "270/330",
              "3.91%",
              "time",
              "-1.59%",
              "-1.73%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "POWERGRID",
              "2024-08-29",
              "344",
              "340/350",
              "320/370",
              "3.74%",
              "time",
              "+1.09%",
              "+1.02%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "M&M",
              "2024-08-29",
              "2,731",
              "2,700/2,800",
              "2,500/3,000",
              "4.40%",
              "time",
              "+1.62%",
              "+1.55%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "INFY",
              "2024-08-29",
              "1,707",
              "1,660/1,740",
              "1,560/1,840",
              "3.11%",
              "time",
              "+0.57%",
              "+0.52%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "BEL",
              "2024-08-29",
              "331",
              "325/340",
              "300/360",
              "4.77%",
              "time",
              "-1.16%",
              "-1.31%"
            ],
            [
              "2024-07-15",
              "2024-08-08",
              "IDFCFIRSTB",
              "2024-08-29",
              "78",
              "76/80",
              "70/85",
              "4.41%",
              "time",
              "-0.32%",
              "-0.42%"
            ],
            [
              "2024-08-12",
              "2024-09-05",
              "TCS",
              "2024-09-26",
              "4,196",
              "4,100/4,300",
              "3,800/4,600",
              "2.35%",
              "time",
              "-1.87%",
              "-1.92%"
            ],
            [
              "2024-08-12",
              "2024-09-05",
              "ICICIBANK",
              "2024-09-26",
              "1,173",
              "1,140/1,200",
              "1,100/1,300",
              "2.35%",
              "time",
              "-1.49%",
              "-1.53%"
            ],
            [
              "2024-08-12",
              "2024-09-05",
              "IOC",
              "2024-09-26",
              "166",
              "160/170",
              "150/180",
              "3.02%",
              "time",
              "-1.45%",
              "-1.56%"
            ],
            [
              "2024-08-12",
              "2024-09-05",
              "BHARTIARTL",
              "2024-09-26",
              "1,459",
              "1,440/1,500",
              "1,400/1,600",
              "2.41%",
              "time",
              "-1.38%",
              "-1.44%"
            ],
            [
              "2024-08-12",
              "2024-09-05",
              "ADANIPORTS",
              "2024-09-26",
              "1,501",
              "1,460/1,540",
              "1,360/1,640",
              "3.59%",
              "time",
              "+1.53%",
              "+1.46%"
            ],
            [
              "2024-08-12",
              "2024-09-05",
              "TRENT",
              "2024-09-26",
              "6,382",
              "6,200/6,500",
              "5,700/7,000",
              "5.27%",
              "time",
              "-0.71%",
              "-0.86%"
            ],
            [
              "2024-08-12",
              "2024-09-05",
              "TATAPOWER",
              "2024-09-26",
              "418",
              "410/430",
              "380/460",
              "4.08%",
              "time",
              "+1.78%",
              "+1.71%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "VEDL",
              "2024-10-31",
              "430",
              "420/440",
              "390/470",
              "4.38%",
              "time",
              "-1.28%",
              "-1.44%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "SBIN",
              "2024-10-31",
              "786",
              "770/810",
              "720/860",
              "2.78%",
              "time",
              "+0.90%",
              "+0.86%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "JSWSTEEL",
              "2024-10-31",
              "971",
              "950/1,000",
              "900/1,040",
              "2.58%",
              "time",
              "+0.41%",
              "+0.35%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "M&M",
              "2024-10-31",
              "2,757",
              "2,700/2,800",
              "2,600/3,000",
              "2.98%",
              "time",
              "-3.82%",
              "-3.97%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "AXISBANK",
              "2024-10-31",
              "1,231",
              "1,200/1,260",
              "1,100/1,320",
              "2.52%",
              "time",
              "+0.07%",
              "+0.03%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "HAL",
              "2024-10-31",
              "4,597",
              "4,500/4,700",
              "4,200/5,000",
              "3.82%",
              "time",
              "+0.97%",
              "+0.90%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "BAJAJ-AUTO",
              "2024-10-31",
              "11,400",
              "11,000/11,700",
              "10,500/12,500",
              "3.08%",
              "time",
              "+0.14%",
              "+0.08%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "NTPC",
              "2024-10-31",
              "411",
              "400/420",
              "370/450",
              "3.19%",
              "time",
              "+0.41%",
              "+0.37%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "TCS",
              "2024-10-31",
              "4,513",
              "4,400/4,650",
              "4,100/5,000",
              "2.50%",
              "time",
              "-1.29%",
              "-1.35%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "HINDALCO",
              "2024-10-31",
              "685",
              "670/700",
              "620/750",
              "3.98%",
              "time",
              "-0.45%",
              "-0.52%"
            ],
            [
              "2024-09-16",
              "2024-09-19",
              "PNB",
              "2024-10-31",
              "108",
              "105/110",
              "95/118",
              "4.27%",
              "target",
              "+2.27%",
              "+2.20%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "IRCTC",
              "2024-10-31",
              "936",
              "920/960",
              "860/1,020",
              "3.32%",
              "time",
              "-0.33%",
              "-0.40%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "INDUSINDBK",
              "2024-10-31",
              "1,470",
              "1,440/1,500",
              "1,320/1,600",
              "3.14%",
              "time",
              "-1.60%",
              "-1.67%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "BAJAJFINSV",
              "2024-10-31",
              "1,858",
              "1,800/1,900",
              "1,700/2,040",
              "3.16%",
              "time",
              "+1.15%",
              "+1.10%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "HINDUNILVR",
              "2024-10-31",
              "2,867",
              "2,800/2,940",
              "2,600/3,100",
              "2.33%",
              "time",
              "+0.01%",
              "-0.02%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "INFY",
              "2024-10-31",
              "1,950",
              "1,900/2,000",
              "1,760/2,120",
              "3.13%",
              "time",
              "+0.45%",
              "+0.40%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "MARUTI",
              "2024-10-31",
              "12,289",
              "12,000/12,600",
              "11,000/13,500",
              "2.74%",
              "time",
              "-0.78%",
              "-0.83%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "ICICIBANK",
              "2024-10-31",
              "1,263",
              "1,230/1,300",
              "1,150/1,400",
              "2.44%",
              "time",
              "+0.81%",
              "+0.78%"
            ],
            [
              "2024-09-16",
              "2024-10-10",
              "MCX",
              "2024-10-31",
              "5,593",
              "5,500/5,750",
              "5,000/6,000",
              "4.21%",
              "time",
              "+0.83%",
              "+0.69%"
            ],
            [
              "2024-10-14",
              "2024-11-07",
              "TECHM",
              "2024-11-28",
              "1,662",
              "1,600/1,700",
              "1,480/1,800",
              "3.35%",
              "time",
              "+1.60%",
              "+1.55%"
            ],
            [
              "2024-10-14",
              "2024-11-07",
              "HAL",
              "2024-11-28",
              "4,508",
              "4,400/4,600",
              "4,100/4,900",
              "3.80%",
              "time",
              "+0.79%",
              "+0.71%"
            ],
            [
              "2024-10-14",
              "2024-11-07",
              "INFY",
              "2024-11-28",
              "1,959",
              "1,900/2,000",
              "1,740/2,100",
              "2.82%",
              "time",
              "-1.69%",
              "-1.75%"
            ],
            [
              "2024-10-14",
              "2024-11-07",
              "SBIN",
              "2024-11-28",
              "805",
              "790/830",
              "740/880",
              "2.91%",
              "time",
              "-0.91%",
              "-0.98%"
            ],
            [
              "2024-10-14",
              "2024-11-07",
              "TATASTEEL",
              "2024-11-28",
              "155",
              "150/158",
              "140/168",
              "3.72%",
              "time",
              "+1.29%",
              "+1.22%"
            ],
            [
              "2024-10-14",
              "2024-11-07",
              "VEDL",
              "2024-11-28",
              "481",
              "470/500",
              "440/530",
              "4.01%",
              "time",
              "+0.70%",
              "+0.60%"
            ],
            [
              "2024-10-14",
              "2024-11-07",
              "TITAN",
              "2024-11-28",
              "3,498",
              "3,400/3,600",
              "3,200/3,800",
              "2.86%",
              "time",
              "-1.80%",
              "-1.89%"
            ],
            [
              "2024-11-11",
              "2024-12-05",
              "RELIANCE",
              "2024-12-26",
              "1,273",
              "1,240/1,300",
              "1,160/1,380",
              "2.95%",
              "time",
              "-0.07%",
              "-0.11%"
            ],
            [
              "2024-11-11",
              "2024-12-05",
              "AXISBANK",
              "2024-12-26",
              "1,171",
              "1,140/1,200",
              "1,040/1,300",
              "2.85%",
              "time",
              "+1.13%",
              "+1.10%"
            ],
            [
              "2024-11-11",
              "2024-12-05",
              "M&M",
              "2024-12-26",
              "2,931",
              "2,900/3,000",
              "2,700/3,200",
              "3.85%",
              "time",
              "+0.13%",
              "+0.06%"
            ],
            [
              "2024-11-11",
              "2024-12-05",
              "TATASTEEL",
              "2024-12-26",
              "142",
              "140/145",
              "130/155",
              "4.13%",
              "time",
              "+0.95%",
              "+0.88%"
            ],
            [
              "2024-11-11",
              "2024-12-05",
              "TCS",
              "2024-12-26",
              "4,199",
              "4,100/4,300",
              "3,800/4,550",
              "2.21%",
              "time",
              "-1.73%",
              "-1.77%"
            ],
            [
              "2024-11-11",
              "2024-12-05",
              "INFY",
              "2024-12-26",
              "1,860",
              "1,800/1,900",
              "1,680/2,000",
              "2.24%",
              "time",
              "-0.48%",
              "-0.52%"
            ],
            [
              "2024-11-11",
              "2024-12-05",
              "POWERGRID",
              "2024-12-26",
              "330",
              "320/340",
              "300/360",
              "2.14%",
              "time",
              "+0.64%",
              "+0.60%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "BHARTIARTL",
              "2025-01-30",
              "1,663",
              "1,620/1,700",
              "1,500/1,800",
              "2.77%",
              "time",
              "+0.71%",
              "+0.68%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "INDUSINDBK",
              "2025-01-30",
              "999",
              "980/1,020",
              "920/1,080",
              "3.37%",
              "time",
              "+0.18%",
              "+0.10%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "M&M",
              "2025-01-30",
              "3,085",
              "3,000/3,200",
              "2,800/3,400",
              "2.96%",
              "time",
              "+0.89%",
              "+0.84%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "TCS",
              "2025-01-30",
              "4,415",
              "4,300/4,550",
              "4,000/4,850",
              "2.57%",
              "time",
              "-2.13%",
              "-2.20%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "ASIANPAINT",
              "2025-01-30",
              "2,402",
              "2,340/2,460",
              "2,200/2,620",
              "2.91%",
              "time",
              "+1.23%",
              "+1.19%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "HAL",
              "2025-01-30",
              "4,676",
              "4,600/4,800",
              "4,300/5,100",
              "3.48%",
              "time",
              "-1.48%",
              "-1.59%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "HCLTECH",
              "2025-01-30",
              "1,954",
              "1,900/2,000",
              "1,800/2,120",
              "2.68%",
              "time",
              "+0.56%",
              "+0.51%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "TATAMOTORS",
              "2025-01-30",
              "785",
              "770/800",
              "720/850",
              "3.78%",
              "time",
              "+0.91%",
              "+0.84%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "TITAN",
              "2025-01-30",
              "3,438",
              "3,350/3,500",
              "3,100/3,700",
              "3.00%",
              "time",
              "+0.74%",
              "+0.69%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "SBIN",
              "2025-01-30",
              "861",
              "840/880",
              "780/940",
              "3.15%",
              "time",
              "-2.89%",
              "-2.97%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "VEDL",
              "2025-01-30",
              "495",
              "480/510",
              "450/540",
              "3.27%",
              "time",
              "-1.57%",
              "-1.67%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "MARICO",
              "2025-01-30",
              "645",
              "600/640",
              "550/700",
              "3.77%",
              "time",
              "-0.88%",
              "-0.95%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "ADANIPORTS",
              "2025-01-30",
              "1,243",
              "1,200/1,280",
              "1,120/1,360",
              "3.30%",
              "time",
              "-0.93%",
              "-1.02%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "HINDUNILVR",
              "2025-01-30",
              "2,366",
              "2,300/2,420",
              "2,200/2,600",
              "2.53%",
              "time",
              "-0.00%",
              "-0.04%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "DLF",
              "2025-01-30",
              "893",
              "870/920",
              "800/980",
              "3.34%",
              "time",
              "-1.99%",
              "-2.09%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "DRREDDY",
              "2025-01-30",
              "1,270",
              "1,250/1,300",
              "1,200/1,400",
              "2.71%",
              "time",
              "-2.61%",
              "-2.68%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "RELIANCE",
              "2025-01-30",
              "1,268",
              "1,240/1,300",
              "1,150/1,390",
              "3.04%",
              "time",
              "+1.26%",
              "+1.22%"
            ],
            [
              "2024-12-16",
              "2025-01-09",
              "ADANIENT",
              "2025-01-30",
              "2,512",
              "2,400/2,600",
              "2,200/2,800",
              "4.05%",
              "time",
              "+1.73%",
              "+1.66%"
            ],
            [
              "2025-01-13",
              "2025-02-06",
              "SBIN",
              "2025-02-27",
              "730",
              "720/750",
              "660/800",
              "4.21%",
              "time",
              "+0.76%",
              "+0.69%"
            ],
            [
              "2025-01-13",
              "2025-02-06",
              "MARUTI",
              "2025-02-27",
              "11,498",
              "11,000/11,800",
              "10,500/12,500",
              "2.38%",
              "time",
              "-3.56%",
              "-3.68%"
            ],
            [
              "2025-01-13",
              "2025-02-06",
              "NTPC",
              "2025-02-27",
              "298",
              "290/310",
              "270/330",
              "3.35%",
              "time",
              "+0.32%",
              "+0.25%"
            ],
            [
              "2025-01-13",
              "2025-02-06",
              "TATAMOTORS",
              "2025-02-27",
              "751",
              "730/770",
              "680/820",
              "3.82%",
              "time",
              "+0.53%",
              "+0.45%"
            ],
            [
              "2025-01-13",
              "2025-02-06",
              "ITC",
              "2025-02-27",
              "439",
              "430/450",
              "400/480",
              "3.43%",
              "time",
              "+0.96%",
              "+0.91%"
            ],
            [
              "2025-01-13",
              "2025-02-06",
              "HAL",
              "2025-02-27",
              "3,788",
              "3,700/3,900",
              "3,400/4,200",
              "4.92%",
              "time",
              "+1.31%",
              "+1.21%"
            ],
            [
              "2025-01-13",
              "2025-01-24",
              "HCLTECH",
              "2025-02-27",
              "1,989",
              "1,900/2,000",
              "1,800/2,120",
              "3.01%",
              "target",
              "+1.58%",
              "+1.49%"
            ],
            [
              "2025-01-13",
              "2025-02-06",
              "LT",
              "2025-02-27",
              "3,464",
              "3,400/3,600",
              "3,200/3,800",
              "2.97%",
              "time",
              "+0.80%",
              "+0.75%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "INDUSINDBK",
              "2025-03-27",
              "1,068",
              "1,060/1,100",
              "1,000/1,160",
              "3.22%",
              "time",
              "-1.26%",
              "-1.36%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "TATAMOTORS",
              "2025-03-27",
              "696",
              "680/710",
              "640/760",
              "3.91%",
              "time",
              "+0.01%",
              "-0.09%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "POWERGRID",
              "2025-03-27",
              "269",
              "260/280",
              "250/300",
              "2.59%",
              "time",
              "+1.21%",
              "+1.16%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "RELIANCE",
              "2025-03-27",
              "1,254",
              "1,220/1,280",
              "1,140/1,360",
              "2.70%",
              "time",
              "+0.80%",
              "+0.76%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "HINDUNILVR",
              "2025-03-27",
              "2,361",
              "2,300/2,420",
              "2,200/2,600",
              "2.49%",
              "time",
              "-0.09%",
              "-0.14%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "M&M",
              "2025-03-27",
              "3,137",
              "3,100/3,200",
              "2,900/3,400",
              "3.89%",
              "time",
              "-1.85%",
              "-1.98%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "LT",
              "2025-03-27",
              "3,329",
              "3,200/3,400",
              "3,000/3,600",
              "2.76%",
              "target",
              "+1.42%",
              "+1.38%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "ASIANPAINT",
              "2025-03-27",
              "2,270",
              "2,200/2,300",
              "2,100/2,500",
              "3.02%",
              "time",
              "+0.97%",
              "+0.93%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "VEDL",
              "2025-03-27",
              "420",
              "410/430",
              "400/460",
              "3.82%",
              "time",
              "+0.05%",
              "-0.06%"
            ],
            [
              "2025-02-10",
              "2025-03-06",
              "ITC",
              "2025-03-27",
              "427",
              "420/440",
              "390/470",
              "2.83%",
              "time",
              "-0.48%",
              "-0.53%"
            ],
            [
              "2025-03-10",
              "2025-04-03",
              "COALINDIA",
              "2025-04-24",
              "365",
              "360/370",
              "330/400",
              "5.42%",
              "time",
              "-0.74%",
              "-0.84%"
            ],
            [
              "2025-03-10",
              "2025-04-03",
              "RELIANCE",
              "2025-04-24",
              "1,238",
              "1,210/1,270",
              "1,120/1,360",
              "2.91%",
              "time",
              "+1.16%",
              "+1.12%"
            ],
            [
              "2025-03-10",
              "2025-04-03",
              "SBIN",
              "2025-04-24",
              "729",
              "710/750",
              "660/800",
              "2.69%",
              "time",
              "-1.46%",
              "-1.52%"
            ],
            [
              "2025-03-10",
              "2025-04-16",
              "INDUSINDBK",
              "2025-04-24",
              "900",
              "880/920",
              "820/980",
              "3.78%",
              "time",
              "-2.24%",
              "-2.37%"
            ],
            [
              "2025-03-10",
              "2025-04-03",
              "ONGC",
              "2025-04-24",
              "223",
              "220/230",
              "205/245",
              "3.67%",
              "time",
              "-1.19%",
              "-1.28%"
            ],
            [
              "2025-03-10",
              "2025-04-03",
              "TATAMOTORS",
              "2025-04-24",
              "648",
              "630/660",
              "590/700",
              "3.15%",
              "time",
              "+0.50%",
              "+0.43%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "BEL",
              "2025-05-29",
              "285",
              "280/290",
              "260/310",
              "4.33%",
              "time",
              "-0.61%",
              "-0.74%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "COALINDIA",
              "2025-05-29",
              "382",
              "370/390",
              "350/420",
              "3.67%",
              "time",
              "+0.86%",
              "+0.80%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "BPCL",
              "2025-05-29",
              "293",
              "290/300",
              "270/320",
              "4.55%",
              "time",
              "+0.32%",
              "+0.22%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "GODREJPROP",
              "2025-05-29",
              "1,948",
              "1,900/2,000",
              "1,800/2,100",
              "3.14%",
              "time",
              "-0.29%",
              "-0.43%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "TATAMOTORS",
              "2025-05-29",
              "595",
              "580/610",
              "540/650",
              "4.64%",
              "time",
              "-1.01%",
              "-1.20%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "RELIANCE",
              "2025-05-29",
              "1,219",
              "1,190/1,250",
              "1,100/1,340",
              "4.09%",
              "time",
              "-2.67%",
              "-2.80%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "IRCTC",
              "2025-05-29",
              "731",
              "700/750",
              "650/800",
              "3.76%",
              "time",
              "+0.94%",
              "+0.88%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "HAL",
              "2025-05-29",
              "4,107",
              "4,000/4,200",
              "3,800/4,500",
              "4.12%",
              "time",
              "-0.64%",
              "-0.78%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "ITC",
              "2025-05-29",
              "422",
              "410/430",
              "380/460",
              "2.96%",
              "time",
              "+0.57%",
              "+0.53%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "SBIN",
              "2025-05-29",
              "754",
              "740/770",
              "680/820",
              "4.11%",
              "time",
              "+1.17%",
              "+1.10%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "TATACONSUM",
              "2025-05-29",
              "1,098",
              "1,050/1,100",
              "970/1,220",
              "3.78%",
              "time",
              "+0.03%",
              "-0.03%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "NTPC",
              "2025-05-29",
              "360",
              "350/370",
              "340/400",
              "3.11%",
              "time",
              "+1.26%",
              "+1.18%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "HINDUNILVR",
              "2025-05-29",
              "2,366",
              "2,300/2,440",
              "2,100/2,600",
              "2.65%",
              "time",
              "+1.11%",
              "+1.08%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "CIPLA",
              "2025-05-29",
              "1,463",
              "1,400/1,500",
              "1,300/1,600",
              "3.01%",
              "time",
              "-0.14%",
              "-0.21%"
            ],
            [
              "2025-04-11",
              "2025-05-08",
              "TATASTEEL",
              "2025-05-29",
              "130",
              "128/135",
              "120/145",
              "4.64%",
              "time",
              "-0.77%",
              "-0.90%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "ASIANPAINT",
              "2025-06-26",
              "2,356",
              "2,300/2,400",
              "2,120/2,600",
              "3.37%",
              "time",
              "-0.10%",
              "-0.15%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "VEDL",
              "2025-06-26",
              "421",
              "410/430",
              "380/460",
              "4.76%",
              "time",
              "+0.96%",
              "+0.86%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "TATASTEEL",
              "2025-06-26",
              "148",
              "145/152",
              "135/165",
              "4.15%",
              "time",
              "-0.30%",
              "-0.38%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "M&M",
              "2025-06-26",
              "3,105",
              "3,000/3,200",
              "2,800/3,400",
              "3.18%",
              "time",
              "+1.35%",
              "+1.30%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "WIPRO",
              "2025-06-26",
              "257",
              "250/262",
              "230/280",
              "3.44%",
              "time",
              "+0.80%",
              "+0.74%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "ADANIPORTS",
              "2025-06-26",
              "1,362",
              "1,320/1,400",
              "1,200/1,460",
              "3.57%",
              "time",
              "+0.63%",
              "+0.54%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "BEL",
              "2025-06-26",
              "323",
              "315/330",
              "290/350",
              "3.95%",
              "time",
              "-2.35%",
              "-2.58%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "HCLTECH",
              "2025-06-26",
              "1,670",
              "1,600/1,700",
              "1,500/1,800",
              "2.94%",
              "time",
              "+1.10%",
              "+1.05%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "TATAMOTORS",
              "2025-06-26",
              "721",
              "700/740",
              "650/800",
              "4.33%",
              "time",
              "+1.88%",
              "+1.81%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "TATAPOWER",
              "2025-06-26",
              "391",
              "380/400",
              "350/430",
              "4.61%",
              "time",
              "+2.08%",
              "+2.01%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "IOC",
              "2025-06-26",
              "139",
              "135/145",
              "130/155",
              "3.12%",
              "time",
              "+1.47%",
              "+1.41%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "RELIANCE",
              "2025-06-26",
              "1,436",
              "1,400/1,470",
              "1,300/1,580",
              "2.73%",
              "time",
              "+1.16%",
              "+1.12%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "HEROMOTOCO",
              "2025-06-26",
              "3,985",
              "3,850/4,000",
              "3,600/4,300",
              "4.19%",
              "time",
              "-0.14%",
              "-0.22%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "SBIN",
              "2025-06-26",
              "802",
              "780/820",
              "720/880",
              "3.34%",
              "time",
              "+1.37%",
              "+1.33%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "DLF",
              "2025-06-26",
              "680",
              "680/700",
              "600/740",
              "4.66%",
              "time",
              "-1.06%",
              "-1.28%"
            ],
            [
              "2025-05-12",
              "2025-06-05",
              "ADANIENT",
              "2025-06-26",
              "2,425",
              "2,340/2,500",
              "2,200/2,600",
              "3.65%",
              "time",
              "+1.62%",
              "+1.53%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "PAYTM",
              "2025-07-31",
              "877",
              "860/900",
              "800/960",
              "5.07%",
              "time",
              "+0.73%",
              "+0.59%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "RELIANCE",
              "2025-07-31",
              "1,438",
              "1,400/1,470",
              "1,300/1,580",
              "2.74%",
              "time",
              "-1.04%",
              "-1.09%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "HEROMOTOCO",
              "2025-07-31",
              "4,364",
              "4,300/4,500",
              "4,000/4,800",
              "3.48%",
              "time",
              "+0.57%",
              "+0.52%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "DRREDDY",
              "2025-07-31",
              "1,347",
              "1,300/1,400",
              "1,210/1,480",
              "2.30%",
              "time",
              "-0.74%",
              "-0.78%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "ADANIENT",
              "2025-07-31",
              "2,544",
              "2,500/2,600",
              "2,400/2,720",
              "2.97%",
              "time",
              "+0.96%",
              "+0.89%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "HINDUNILVR",
              "2025-07-31",
              "2,327",
              "2,260/2,400",
              "2,000/2,560",
              "2.09%",
              "time",
              "+0.16%",
              "+0.13%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "BPCL",
              "2025-07-31",
              "316",
              "310/325",
              "290/345",
              "3.46%",
              "time",
              "-1.38%",
              "-1.48%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "TATASTEEL",
              "2025-07-31",
              "154",
              "150/158",
              "140/170",
              "3.57%",
              "time",
              "+0.16%",
              "+0.11%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "ONGC",
              "2025-07-31",
              "257",
              "250/262",
              "230/280",
              "3.23%",
              "time",
              "+0.16%",
              "+0.10%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "SBIN",
              "2025-07-31",
              "792",
              "770/810",
              "720/870",
              "2.90%",
              "time",
              "+1.12%",
              "+1.09%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "TRENT",
              "2025-07-31",
              "5,680",
              "5,500/5,800",
              "5,000/6,200",
              "4.07%",
              "time",
              "+0.54%",
              "+0.47%"
            ],
            [
              "2025-06-16",
              "2025-07-07",
              "LT",
              "2025-07-31",
              "3,629",
              "3,500/3,700",
              "3,200/4,000",
              "3.40%",
              "target",
              "+1.70%",
              "+1.67%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "BEL",
              "2025-07-31",
              "404",
              "395/415",
              "365/440",
              "3.75%",
              "time",
              "+1.06%",
              "+1.00%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "VEDL",
              "2025-07-31",
              "447",
              "440/460",
              "400/490",
              "4.21%",
              "time",
              "+0.57%",
              "+0.49%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "INFY",
              "2025-07-31",
              "1,624",
              "1,580/1,660",
              "1,460/1,780",
              "3.53%",
              "time",
              "+0.83%",
              "+0.78%"
            ],
            [
              "2025-06-16",
              "2025-07-07",
              "ITC",
              "2025-07-31",
              "418",
              "410/430",
              "380/460",
              "2.31%",
              "target",
              "+1.22%",
              "+1.20%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "HAL",
              "2025-07-31",
              "5,064",
              "4,950/5,200",
              "4,600/5,600",
              "4.55%",
              "time",
              "+1.59%",
              "+1.52%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "MCX",
              "2025-07-31",
              "7,828",
              "7,700/8,000",
              "7,000/8,500",
              "5.10%",
              "time",
              "+0.88%",
              "+0.76%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "TATAMOTORS",
              "2025-07-31",
              "687",
              "670/700",
              "620/750",
              "4.11%",
              "time",
              "+1.59%",
              "+1.53%"
            ],
            [
              "2025-06-16",
              "2025-07-10",
              "TATAPOWER",
              "2025-07-31",
              "399",
              "390/410",
              "360/440",
              "3.48%",
              "time",
              "+1.24%",
              "+1.19%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "KOTAKBANK",
              "2025-08-28",
              "2,204",
              "2,100/2,240",
              "1,900/2,400",
              "2.62%",
              "time",
              "-1.80%",
              "-1.85%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "TITAN",
              "2025-08-28",
              "3,405",
              "3,300/3,500",
              "3,200/3,700",
              "2.14%",
              "time",
              "+0.55%",
              "+0.51%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "RELIANCE",
              "2025-08-28",
              "1,484",
              "1,450/1,520",
              "1,340/1,620",
              "2.49%",
              "time",
              "-1.34%",
              "-1.39%"
            ],
            [
              "2025-07-14",
              "2025-08-05",
              "MARUTI",
              "2025-08-28",
              "12,514",
              "12,200/12,800",
              "11,000/13,400",
              "2.61%",
              "target",
              "+1.32%",
              "+1.29%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "LT",
              "2025-08-28",
              "3,496",
              "3,400/3,600",
              "3,120/3,800",
              "2.74%",
              "time",
              "+0.59%",
              "+0.55%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "TCS",
              "2025-08-28",
              "3,223",
              "3,140/3,300",
              "2,920/3,520",
              "2.62%",
              "time",
              "-0.20%",
              "-0.24%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "TRENT",
              "2025-08-28",
              "5,314",
              "5,200/5,400",
              "4,800/5,800",
              "4.72%",
              "time",
              "+1.66%",
              "+1.57%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "ICICIBANK",
              "2025-08-28",
              "1,423",
              "1,400/1,450",
              "1,300/1,550",
              "2.70%",
              "time",
              "+1.33%",
              "+1.30%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "INFY",
              "2025-08-28",
              "1,570",
              "1,540/1,600",
              "1,400/1,700",
              "3.86%",
              "time",
              "-1.76%",
              "-1.84%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "BAJFINANCE",
              "2025-08-28",
              "919",
              "900/940",
              "800/1,000",
              "3.72%",
              "time",
              "+0.24%",
              "+0.19%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "WIPRO",
              "2025-08-28",
              "254",
              "250/260",
              "230/280",
              "4.33%",
              "time",
              "+0.49%",
              "+0.43%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "HAL",
              "2025-08-28",
              "4,886",
              "4,800/5,000",
              "4,500/5,300",
              "3.75%",
              "time",
              "-0.28%",
              "-0.37%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "M&M",
              "2025-08-28",
              "3,092",
              "3,000/3,150",
              "2,800/3,400",
              "3.74%",
              "time",
              "-0.10%",
              "-0.16%"
            ],
            [
              "2025-07-14",
              "2025-08-07",
              "TATAMOTORS",
              "2025-08-28",
              "674",
              "660/690",
              "620/740",
              "3.79%",
              "time",
              "+0.71%",
              "+0.64%"
            ],
            [
              "2025-07-14",
              "2025-08-04",
              "NTPC",
              "2025-08-28",
              "342",
              "330/350",
              "310/370",
              "2.60%",
              "target",
              "+1.39%",
              "+1.36%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "WIPRO",
              "2025-09-30",
              "247",
              "240/255",
              "225/270",
              "2.67%",
              "time",
              "+0.95%",
              "+0.91%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "DELHIVERY",
              "2025-09-30",
              "464",
              "460/480",
              "420/500",
              "3.78%",
              "time",
              "+1.11%",
              "+1.04%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "DLF",
              "2025-09-30",
              "752",
              "740/780",
              "680/840",
              "3.72%",
              "time",
              "+1.37%",
              "+1.32%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "RELIANCE",
              "2025-09-30",
              "1,374",
              "1,340/1,410",
              "1,240/1,500",
              "2.56%",
              "time",
              "+1.18%",
              "+1.15%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "TCS",
              "2025-09-30",
              "3,022",
              "2,940/3,100",
              "2,720/3,300",
              "2.74%",
              "time",
              "+1.18%",
              "+1.15%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "SBIN",
              "2025-09-30",
              "827",
              "810/850",
              "750/900",
              "2.19%",
              "time",
              "+0.93%",
              "+0.90%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "LT",
              "2025-09-30",
              "3,677",
              "3,600/3,800",
              "3,500/4,000",
              "1.70%",
              "time",
              "+0.26%",
              "+0.22%"
            ],
            [
              "2025-08-14",
              "2025-08-28",
              "ADANIPORTS",
              "2025-09-30",
              "1,300",
              "1,240/1,340",
              "1,160/1,440",
              "4.70%",
              "target",
              "+2.38%",
              "+2.32%"
            ],
            [
              "2025-08-14",
              "2025-09-02",
              "INDUSINDBK",
              "2025-09-30",
              "770",
              "720/790",
              "700/840",
              "2.34%",
              "target",
              "+1.19%",
              "+1.14%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "PAYTM",
              "2025-09-30",
              "1,151",
              "1,160/1,180",
              "1,080/1,260",
              "5.13%",
              "time",
              "+0.46%",
              "+0.33%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "ONGC",
              "2025-09-30",
              "237",
              "230/240",
              "220/260",
              "2.64%",
              "time",
              "+1.18%",
              "+1.15%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "HAL",
              "2025-09-30",
              "4,555",
              "4,400/4,700",
              "4,000/5,000",
              "2.92%",
              "time",
              "+0.95%",
              "+0.91%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "APOLLOHOSP",
              "2025-09-30",
              "7,822",
              "7,650/8,000",
              "7,200/8,200",
              "2.20%",
              "time",
              "+1.04%",
              "+1.00%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "MUTHOOTFIN",
              "2025-09-30",
              "2,757",
              "2,700/2,850",
              "2,500/3,000",
              "4.04%",
              "time",
              "+0.89%",
              "+0.82%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "INFY",
              "2025-09-30",
              "1,448",
              "1,420/1,480",
              "1,320/1,600",
              "3.47%",
              "time",
              "+0.11%",
              "+0.06%"
            ],
            [
              "2025-08-14",
              "2025-09-09",
              "VEDL",
              "2025-09-30",
              "415",
              "400/430",
              "380/460",
              "3.34%",
              "time",
              "+0.59%",
              "+0.53%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "HAL",
              "2025-10-28",
              "4,746",
              "4,600/4,900",
              "4,400/5,000",
              "1.75%",
              "time",
              "+0.52%",
              "+0.46%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "AXISBANK",
              "2025-10-28",
              "1,105",
              "1,080/1,130",
              "1,000/1,200",
              "2.94%",
              "time",
              "-1.27%",
              "-1.34%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "HEROMOTOCO",
              "2025-10-28",
              "5,302",
              "5,200/5,400",
              "4,800/5,700",
              "3.25%",
              "time",
              "-0.44%",
              "-0.51%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "TATAMOTORS",
              "2025-10-28",
              "715",
              "700/730",
              "650/780",
              "2.85%",
              "time",
              "+1.01%",
              "+0.98%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "BEL",
              "2025-10-28",
              "399",
              "390/410",
              "360/440",
              "3.02%",
              "time",
              "+0.31%",
              "+0.27%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "VEDL",
              "2025-10-28",
              "451",
              "440/460",
              "410/490",
              "3.29%",
              "time",
              "-0.32%",
              "-0.39%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "MARUTI",
              "2025-10-28",
              "15,325",
              "15,000/15,500",
              "14,000/16,600",
              "2.94%",
              "time",
              "-1.28%",
              "-1.33%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "RELIANCE",
              "2025-10-28",
              "1,395",
              "1,360/1,430",
              "1,260/1,520",
              "2.11%",
              "time",
              "+0.70%",
              "+0.67%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "M&M",
              "2025-10-28",
              "3,590",
              "3,500/3,700",
              "3,200/4,000",
              "3.34%",
              "time",
              "+1.00%",
              "+0.96%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "BAJFINANCE",
              "2025-10-28",
              "1,003",
              "980/1,030",
              "900/1,100",
              "3.19%",
              "time",
              "+0.79%",
              "+0.74%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "TRENT",
              "2025-10-28",
              "5,130",
              "5,000/5,300",
              "4,600/5,800",
              "3.96%",
              "time",
              "-1.10%",
              "-1.18%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "INFY",
              "2025-10-28",
              "1,526",
              "1,480/1,560",
              "1,380/1,660",
              "3.04%",
              "time",
              "-0.07%",
              "-0.12%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "HINDALCO",
              "2025-10-28",
              "758",
              "750/780",
              "700/820",
              "3.07%",
              "time",
              "+0.77%",
              "+0.72%"
            ],
            [
              "2025-09-12",
              "2025-10-07",
              "INDUSINDBK",
              "2025-10-28",
              "740",
              "700/760",
              "680/800",
              "2.26%",
              "time",
              "+0.39%",
              "+0.33%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "ADANIPORTS",
              "2025-11-25",
              "1,409",
              "1,400/1,440",
              "1,300/1,480",
              "2.86%",
              "time",
              "+0.82%",
              "+0.75%"
            ],
            [
              "2025-10-10",
              "2025-10-13",
              "TATAMOTORS",
              "2025-11-25",
              "679",
              "660/700",
              "610/750",
              "0.57%",
              "target",
              "+0.35%",
              "+0.35%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "MARUTI",
              "2025-11-25",
              "16,265",
              "16,000/16,600",
              "15,000/18,000",
              "3.25%",
              "time",
              "-0.08%",
              "-0.13%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "INFY",
              "2025-11-25",
              "1,515",
              "1,480/1,560",
              "1,380/1,680",
              "3.29%",
              "time",
              "+0.47%",
              "+0.43%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "SBIN",
              "2025-11-25",
              "881",
              "860/905",
              "800/960",
              "2.45%",
              "time",
              "-2.36%",
              "-2.43%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "HAL",
              "2025-11-25",
              "4,833",
              "4,800/5,000",
              "4,500/5,200",
              "3.08%",
              "time",
              "-0.29%",
              "-0.36%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "RELIANCE",
              "2025-11-25",
              "1,382",
              "1,350/1,420",
              "1,260/1,520",
              "2.36%",
              "time",
              "-1.77%",
              "-1.82%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "ADANIENT",
              "2025-11-25",
              "2,551",
              "2,500/2,600",
              "2,200/2,800",
              "4.78%",
              "time",
              "-0.10%",
              "-0.18%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "CIPLA",
              "2025-11-25",
              "1,562",
              "1,500/1,600",
              "1,400/1,700",
              "2.57%",
              "time",
              "+0.70%",
              "+0.66%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "TATASTEEL",
              "2025-11-25",
              "174",
              "170/180",
              "160/190",
              "2.67%",
              "time",
              "+0.29%",
              "+0.24%"
            ],
            [
              "2025-10-10",
              "2025-11-04",
              "INDUSINDBK",
              "2025-11-25",
              "763",
              "750/780",
              "700/840",
              "4.01%",
              "time",
              "+0.51%",
              "+0.44%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "TRENT",
              "2025-12-30",
              "4,391",
              "4,300/4,500",
              "4,000/4,800",
              "3.52%",
              "time",
              "-0.60%",
              "-0.68%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "TATAPOWER",
              "2025-12-30",
              "388",
              "380/400",
              "350/430",
              "3.09%",
              "time",
              "+1.07%",
              "+1.03%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "VOLTAS",
              "2025-12-30",
              "1,351",
              "1,320/1,400",
              "1,220/1,500",
              "3.72%",
              "time",
              "+1.75%",
              "+1.71%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "TATASTEEL",
              "2025-12-30",
              "174",
              "170/178",
              "160/190",
              "3.42%",
              "time",
              "-0.68%",
              "-0.75%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "JSWSTEEL",
              "2025-12-30",
              "1,168",
              "1,130/1,200",
              "1,100/1,300",
              "2.77%",
              "time",
              "+1.18%",
              "+1.13%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "M&M",
              "2025-12-30",
              "3,699",
              "3,600/3,800",
              "3,400/4,000",
              "2.65%",
              "time",
              "+0.99%",
              "+0.95%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "MCX",
              "2025-12-30",
              "9,666",
              "9,500/9,900",
              "9,000/10,600",
              "4.32%",
              "time",
              "+0.15%",
              "+0.05%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "INFY",
              "2025-12-30",
              "1,503",
              "1,460/1,540",
              "1,360/1,640",
              "2.96%",
              "time",
              "-1.07%",
              "-1.13%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "MUTHOOTFIN",
              "2025-12-30",
              "3,726",
              "3,650/3,800",
              "3,400/4,000",
              "3.52%",
              "time",
              "+1.35%",
              "+1.29%"
            ],
            [
              "2025-11-14",
              "2025-11-21",
              "ADANIENT",
              "2025-12-30",
              "2,440",
              "2,400/2,500",
              "2,200/2,680",
              "4.07%",
              "target",
              "+4.39%",
              "+4.29%"
            ],
            [
              "2025-11-14",
              "2025-12-08",
              "SBIN",
              "2025-12-30",
              "968",
              "940/990",
              "870/1,060",
              "2.23%",
              "target",
              "+1.13%",
              "+1.10%"
            ],
            [
              "2025-11-14",
              "2025-12-04",
              "LT",
              "2025-12-30",
              "4,004",
              "3,900/4,100",
              "3,600/4,400",
              "2.48%",
              "target",
              "+1.30%",
              "+1.27%"
            ],
            [
              "2025-11-14",
              "2025-12-05",
              "POWERGRID",
              "2025-12-30",
              "271",
              "265/280",
              "250/300",
              "2.32%",
              "target",
              "+1.29%",
              "+1.26%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "IRCTC",
              "2025-12-30",
              "705",
              "690/720",
              "650/770",
              "2.71%",
              "time",
              "+0.06%",
              "+0.01%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "RELIANCE",
              "2025-12-30",
              "1,519",
              "1,480/1,560",
              "1,370/1,660",
              "1.98%",
              "time",
              "+0.87%",
              "+0.85%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "ADANIPORTS",
              "2025-12-30",
              "1,513",
              "1,460/1,540",
              "1,360/1,640",
              "2.74%",
              "time",
              "+1.22%",
              "+1.18%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "HDFCBANK",
              "2025-12-30",
              "990",
              "960/1,010",
              "900/1,080",
              "2.10%",
              "time",
              "+0.89%",
              "+0.87%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "KOTAKBANK",
              "2025-12-30",
              "2,080",
              "2,000/2,120",
              "1,900/2,240",
              "2.14%",
              "time",
              "+0.36%",
              "+0.33%"
            ],
            [
              "2025-11-14",
              "2025-12-04",
              "GRASIM",
              "2025-12-30",
              "2,783",
              "2,700/2,820",
              "2,600/3,020",
              "2.72%",
              "target",
              "+1.42%",
              "+1.39%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "COLPAL",
              "2025-12-30",
              "2,173",
              "2,100/2,200",
              "2,000/2,400",
              "2.56%",
              "time",
              "+1.03%",
              "+1.00%"
            ],
            [
              "2025-11-14",
              "2025-12-02",
              "ITC",
              "2025-12-30",
              "400",
              "390/410",
              "380/440",
              "2.13%",
              "target",
              "+1.10%",
              "+1.08%"
            ],
            [
              "2025-11-14",
              "2025-12-09",
              "BAJFINANCE",
              "2025-12-30",
              "1,018",
              "1,000/1,040",
              "930/1,110",
              "3.23%",
              "time",
              "+1.13%",
              "+1.09%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "INFY",
              "2026-01-27",
              "1,598",
              "1,560/1,640",
              "1,440/1,760",
              "3.39%",
              "time",
              "+0.96%",
              "+0.92%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "TRENT",
              "2026-01-27",
              "4,075",
              "4,000/4,200",
              "3,900/4,500",
              "2.56%",
              "time",
              "+0.61%",
              "+0.55%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "IDFCFIRSTB",
              "2026-01-27",
              "82",
              "80/84",
              "75/90",
              "3.44%",
              "time",
              "+0.45%",
              "+0.39%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "BEL",
              "2026-01-27",
              "389",
              "380/400",
              "350/425",
              "3.04%",
              "time",
              "-0.53%",
              "-0.59%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "RELIANCE",
              "2026-01-27",
              "1,556",
              "1,520/1,600",
              "1,420/1,700",
              "1.88%",
              "time",
              "-0.16%",
              "-0.18%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "HAL",
              "2026-01-27",
              "4,302",
              "4,200/4,400",
              "4,000/4,700",
              "2.96%",
              "time",
              "-0.44%",
              "-0.50%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "TATASTEEL",
              "2026-01-27",
              "172",
              "168/175",
              "155/185",
              "3.31%",
              "time",
              "-1.13%",
              "-1.22%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "SIEMENS",
              "2026-01-27",
              "3,145",
              "3,100/3,200",
              "2,900/3,400",
              "3.68%",
              "time",
              "+1.10%",
              "+1.04%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "VEDL",
              "2026-01-27",
              "544",
              "530/560",
              "490/600",
              "3.65%",
              "time",
              "-2.69%",
              "-2.81%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "BPCL",
              "2026-01-27",
              "355",
              "350/360",
              "322/380",
              "3.59%",
              "time",
              "-0.14%",
              "-0.22%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "ICICIBANK",
              "2026-01-27",
              "1,366",
              "1,340/1,400",
              "1,280/1,500",
              "1.97%",
              "time",
              "-0.43%",
              "-0.46%"
            ],
            [
              "2025-12-12",
              "2026-01-06",
              "MARUTI",
              "2026-01-27",
              "16,522",
              "16,100/17,000",
              "15,000/18,000",
              "2.15%",
              "time",
              "-0.44%",
              "-0.47%"
            ],
            [
              "2026-01-09",
              "2026-02-03",
              "SBIN",
              "2026-02-24",
              "1,000",
              "980/1,030",
              "900/1,100",
              "2.67%",
              "time",
              "-1.11%",
              "-1.16%"
            ],
            [
              "2026-01-09",
              "2026-02-03",
              "BEL",
              "2026-02-24",
              "419",
              "410/430",
              "380/460",
              "3.62%",
              "time",
              "+0.07%",
              "-0.01%"
            ],
            [
              "2026-01-09",
              "2026-02-03",
              "COFORGE",
              "2026-02-24",
              "1,682",
              "1,600/1,700",
              "1,500/1,800",
              "3.43%",
              "time",
              "+0.68%",
              "+0.60%"
            ],
            [
              "2026-01-09",
              "2026-02-02",
              "HDFCBANK",
              "2026-02-24",
              "939",
              "915/960",
              "850/1,030",
              "2.87%",
              "target",
              "+1.55%",
              "+1.52%"
            ],
            [
              "2026-01-09",
              "2026-02-03",
              "INDUSINDBK",
              "2026-02-24",
              "882",
              "850/900",
              "800/960",
              "3.52%",
              "time",
              "+0.08%",
              "+0.00%"
            ],
            [
              "2026-01-09",
              "2026-02-03",
              "RELIANCE",
              "2026-02-24",
              "1,475",
              "1,440/1,510",
              "1,340/1,610",
              "2.76%",
              "time",
              "+0.92%",
              "+0.89%"
            ],
            [
              "2026-01-09",
              "2026-02-03",
              "ADANIENT",
              "2026-02-24",
              "2,154",
              "2,100/2,200",
              "1,920/2,300",
              "3.38%",
              "time",
              "+0.40%",
              "+0.30%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "WIPRO",
              "2026-03-30",
              "214",
              "208/220",
              "195/235",
              "3.69%",
              "time",
              "+0.19%",
              "+0.10%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "HCLTECH",
              "2026-03-30",
              "1,455",
              "1,420/1,500",
              "1,320/1,600",
              "3.75%",
              "time",
              "+0.08%",
              "-0.00%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "HDFCBANK",
              "2026-03-30",
              "904",
              "880/925",
              "820/990",
              "2.81%",
              "time",
              "-0.69%",
              "-0.75%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "RELIANCE",
              "2026-03-30",
              "1,420",
              "1,380/1,460",
              "1,280/1,560",
              "2.41%",
              "time",
              "+0.32%",
              "+0.28%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "EICHERMOT",
              "2026-03-30",
              "8,065",
              "7,900/8,200",
              "7,300/8,750",
              "3.48%",
              "time",
              "-1.69%",
              "-1.78%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "HINDALCO",
              "2026-03-30",
              "909",
              "890/930",
              "830/990",
              "3.82%",
              "time",
              "-0.08%",
              "-0.16%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "VEDL",
              "2026-03-30",
              "674",
              "660/690",
              "615/735",
              "3.98%",
              "time",
              "-0.07%",
              "-0.18%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "HINDUNILVR",
              "2026-03-30",
              "2,305",
              "2,240/2,360",
              "2,080/2,520",
              "2.84%",
              "time",
              "+0.11%",
              "+0.07%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "TRENT",
              "2026-03-30",
              "4,252",
              "4,100/4,400",
              "3,800/4,700",
              "3.16%",
              "time",
              "-2.65%",
              "-2.76%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "HAL",
              "2026-03-30",
              "4,212",
              "4,100/4,300",
              "3,800/4,600",
              "3.91%",
              "time",
              "+0.38%",
              "+0.30%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "TATAPOWER",
              "2026-03-30",
              "374",
              "360/382",
              "335/410",
              "3.03%",
              "time",
              "+0.57%",
              "+0.53%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "BRITANNIA",
              "2026-03-30",
              "5,980",
              "5,750/6,100",
              "5,550/6,500",
              "2.90%",
              "time",
              "+1.07%",
              "+1.03%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "INFY",
              "2026-03-30",
              "1,369",
              "1,340/1,400",
              "1,240/1,500",
              "4.56%",
              "time",
              "+1.57%",
              "+1.48%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "TCS",
              "2026-03-30",
              "2,692",
              "2,620/2,760",
              "2,440/2,940",
              "4.04%",
              "time",
              "+0.83%",
              "+0.75%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "COALINDIA",
              "2026-03-30",
              "409",
              "400/420",
              "372/450",
              "3.08%",
              "time",
              "-1.37%",
              "-1.44%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "BEL",
              "2026-03-30",
              "436",
              "425/445",
              "395/475",
              "3.71%",
              "time",
              "-0.10%",
              "-0.18%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "LT",
              "2026-03-30",
              "4,174",
              "4,100/4,300",
              "3,900/4,600",
              "2.32%",
              "time",
              "-1.11%",
              "-1.18%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "KOTAKBANK",
              "2026-03-30",
              "421",
              "410/430",
              "380/460",
              "2.63%",
              "time",
              "-1.68%",
              "-1.74%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "AXISBANK",
              "2026-03-30",
              "1,332",
              "1,300/1,370",
              "1,210/1,460",
              "2.34%",
              "time",
              "-0.05%",
              "-0.09%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "SBIN",
              "2026-03-30",
              "1,199",
              "1,170/1,230",
              "1,090/1,310",
              "3.11%",
              "time",
              "-1.14%",
              "-1.21%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "BAJFINANCE",
              "2026-03-30",
              "1,025",
              "1,000/1,050",
              "930/1,120",
              "3.13%",
              "time",
              "-1.13%",
              "-1.20%"
            ],
            [
              "2026-02-13",
              "2026-03-09",
              "ADANIENT",
              "2026-03-30",
              "2,137",
              "2,080/2,200",
              "1,960/2,300",
              "3.16%",
              "time",
              "-0.35%",
              "-0.46%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "LT",
              "2026-04-28",
              "3,439",
              "3,400/3,500",
              "3,160/3,760",
              "5.16%",
              "time",
              "-0.55%",
              "-0.69%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "INDUSINDBK",
              "2026-04-28",
              "814",
              "800/840",
              "750/900",
              "3.69%",
              "time",
              "+0.24%",
              "+0.14%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "INFY",
              "2026-04-28",
              "1,248",
              "1,200/1,280",
              "1,120/1,360",
              "4.23%",
              "time",
              "-0.11%",
              "-0.23%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "BEL",
              "2026-04-28",
              "439",
              "430/450",
              "400/480",
              "4.05%",
              "time",
              "+0.77%",
              "+0.69%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "M&M",
              "2026-04-28",
              "2,931",
              "2,900/3,000",
              "2,700/3,200",
              "4.45%",
              "time",
              "+0.24%",
              "+0.13%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "MARUTI",
              "2026-04-28",
              "12,591",
              "12,500/13,000",
              "12,000/13,800",
              "3.36%",
              "time",
              "+0.59%",
              "+0.50%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "COFORGE",
              "2026-04-28",
              "1,090",
              "1,060/1,120",
              "1,000/1,200",
              "4.27%",
              "time",
              "-1.21%",
              "-1.40%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "SBIN",
              "2026-04-28",
              "1,047",
              "1,020/1,075",
              "950/1,150",
              "4.33%",
              "time",
              "+1.40%",
              "+1.33%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "RELIANCE",
              "2026-04-28",
              "1,381",
              "1,350/1,420",
              "1,250/1,520",
              "3.44%",
              "time",
              "+0.06%",
              "-0.00%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "EICHERMOT",
              "2026-04-28",
              "6,741",
              "6,600/7,000",
              "6,400/7,500",
              "2.98%",
              "time",
              "+0.75%",
              "+0.66%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "ADANIPORTS",
              "2026-04-28",
              "1,363",
              "1,320/1,400",
              "1,220/1,500",
              "3.74%",
              "time",
              "-0.06%",
              "-0.15%"
            ],
            [
              "2026-03-13",
              "2026-04-07",
              "POWERGRID",
              "2026-04-28",
              "301",
              "290/310",
              "280/330",
              "2.61%",
              "time",
              "+1.05%",
              "+0.99%"
            ],
            [
              "2026-04-10",
              "2026-05-05",
              "HAL",
              "2026-05-26",
              "4,112",
              "4,000/4,200",
              "3,700/4,500",
              "3.99%",
              "time",
              "-1.76%",
              "-1.89%"
            ],
            [
              "2026-04-10",
              "2026-05-05",
              "ASIANPAINT",
              "2026-05-26",
              "2,361",
              "2,300/2,400",
              "2,100/2,600",
              "4.29%",
              "time",
              "+0.39%",
              "+0.33%"
            ],
            [
              "2026-04-10",
              "2026-05-05",
              "HINDUNILVR",
              "2026-05-26",
              "2,155",
              "2,100/2,200",
              "1,900/2,340",
              "3.48%",
              "time",
              "-1.23%",
              "-1.30%"
            ],
            [
              "2026-04-10",
              "2026-05-05",
              "RELIANCE",
              "2026-05-26",
              "1,350",
              "1,320/1,380",
              "1,220/1,470",
              "3.32%",
              "time",
              "-1.74%",
              "-1.82%"
            ],
            [
              "2026-04-10",
              "2026-05-05",
              "COALINDIA",
              "2026-05-26",
              "434",
              "420/440",
              "390/470",
              "3.92%",
              "time",
              "-1.34%",
              "-1.44%"
            ],
            [
              "2026-04-10",
              "2026-05-05",
              "VOLTAS",
              "2026-05-26",
              "1,316",
              "1,300/1,320",
              "1,200/1,400",
              "5.20%",
              "time",
              "+0.41%",
              "+0.26%"
            ],
            [
              "2026-04-10",
              "2026-05-05",
              "BEL",
              "2026-05-26",
              "442",
              "430/450",
              "400/480",
              "4.03%",
              "time",
              "+0.67%",
              "+0.59%"
            ],
            [
              "2026-04-10",
              "2026-05-05",
              "ITC",
              "2026-05-26",
              "304",
              "300/310",
              "280/330",
              "3.09%",
              "time",
              "+0.61%",
              "+0.56%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "WIPRO",
              "2026-06-30",
              "190",
              "185/195",
              "170/208",
              "4.72%",
              "time",
              "+0.86%",
              "+0.78%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "EICHERMOT",
              "2026-06-30",
              "7,014",
              "6,800/7,200",
              "6,200/7,500",
              "3.51%",
              "time",
              "+1.44%",
              "+1.38%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "RELIANCE",
              "2026-06-30",
              "1,336",
              "1,300/1,370",
              "1,200/1,460",
              "3.53%",
              "time",
              "+0.45%",
              "+0.40%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "BPCL",
              "2026-06-30",
              "284",
              "280/290",
              "260/310",
              "4.92%",
              "time",
              "+1.67%",
              "+1.57%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "DRREDDY",
              "2026-06-30",
              "1,337",
              "1,300/1,370",
              "1,200/1,460",
              "3.35%",
              "time",
              "+0.32%",
              "+0.27%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "TCS",
              "2026-06-30",
              "2,264",
              "2,200/2,320",
              "2,040/2,480",
              "3.74%",
              "time",
              "+0.47%",
              "+0.41%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "VOLTAS",
              "2026-06-30",
              "1,231",
              "1,200/1,260",
              "1,120/1,340",
              "4.18%",
              "time",
              "+0.32%",
              "+0.21%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "HDFCBANK",
              "2026-06-30",
              "768",
              "750/785",
              "695/840",
              "3.47%",
              "time",
              "-0.14%",
              "-0.19%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "LT",
              "2026-06-30",
              "3,909",
              "3,800/4,000",
              "3,520/4,300",
              "3.71%",
              "target",
              "+1.98%",
              "+1.94%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "M&M",
              "2026-06-30",
              "3,123",
              "3,000/3,200",
              "2,800/3,400",
              "3.43%",
              "time",
              "+1.19%",
              "+1.14%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "ITC",
              "2026-06-30",
              "309",
              "302/315",
              "280/335",
              "3.54%",
              "time",
              "+0.32%",
              "+0.26%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "HAL",
              "2026-06-30",
              "4,386",
              "4,300/4,500",
              "4,000/4,800",
              "3.81%",
              "time",
              "+1.16%",
              "+1.09%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "INFY",
              "2026-06-30",
              "1,119",
              "1,090/1,150",
              "1,010/1,230",
              "3.65%",
              "time",
              "-0.17%",
              "-0.24%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "VEDL",
              "2026-06-30",
              "331",
              "320/340",
              "300/360",
              "4.12%",
              "time",
              "+0.27%",
              "+0.15%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "MUTHOOTFIN",
              "2026-06-30",
              "3,311",
              "3,200/3,400",
              "3,000/3,600",
              "3.58%",
              "time",
              "-0.51%",
              "-0.61%"
            ],
            [
              "2026-05-15",
              "2026-06-30",
              "TRENT",
              "2026-06-30",
              "4,101",
              "4,000/4,200",
              "3,700/4,500",
              "3.99%",
              "expiry",
              "-3.33%",
              "-3.52%"
            ],
            [
              "2026-05-15",
              "2026-06-09",
              "HINDUNILVR",
              "2026-06-30",
              "2,272",
              "2,200/2,320",
              "2,040/2,500",
              "3.24%",
              "time",
              "-0.24%",
              "-0.29%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "TCS",
              "2026-07-28",
              "2,161",
              "2,100/2,220",
              "1,960/2,360",
              "3.64%",
              "time",
              "+0.81%",
              "+0.74%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "M&M",
              "2026-07-28",
              "3,043",
              "2,980/3,100",
              "2,760/3,300",
              "4.60%",
              "time",
              "+0.87%",
              "+0.80%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "SBIN",
              "2026-07-28",
              "1,017",
              "990/1,040",
              "920/1,120",
              "3.02%",
              "time",
              "+0.64%",
              "+0.60%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "AXISBANK",
              "2026-07-28",
              "1,356",
              "1,320/1,400",
              "1,220/1,500",
              "3.93%",
              "time",
              "+1.86%",
              "+1.82%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "BAJAJ-AUTO",
              "2026-07-28",
              "10,063",
              "9,800/10,200",
              "9,000/11,200",
              "3.95%",
              "time",
              "+1.22%",
              "+1.17%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "RELIANCE",
              "2026-07-28",
              "1,293",
              "1,260/1,330",
              "1,170/1,420",
              "3.22%",
              "time",
              "+1.25%",
              "+1.21%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "IOC",
              "2026-07-28",
              "141",
              "135/145",
              "125/150",
              "2.38%",
              "time",
              "+1.01%",
              "+0.95%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "PNB",
              "2026-07-28",
              "107",
              "105/110",
              "97/117",
              "3.68%",
              "time",
              "+0.53%",
              "+0.47%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "MARUTI",
              "2026-07-28",
              "13,366",
              "13,000/13,700",
              "12,000/14,600",
              "2.90%",
              "time",
              "-1.82%",
              "-1.89%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "MCX",
              "2026-07-28",
              "2,853",
              "2,800/2,900",
              "2,600/3,100",
              "4.55%",
              "time",
              "-0.21%",
              "-0.33%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "VEDL",
              "2026-07-28",
              "310",
              "300/320",
              "280/340",
              "4.04%",
              "time",
              "-0.90%",
              "-1.04%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "HDFCBANK",
              "2026-07-28",
              "772",
              "755/790",
              "700/840",
              "3.15%",
              "time",
              "-1.35%",
              "-1.41%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "ICICIBANK",
              "2026-07-28",
              "1,341",
              "1,300/1,380",
              "1,200/1,480",
              "2.54%",
              "time",
              "-0.80%",
              "-0.85%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "KOTAKBANK",
              "2026-07-28",
              "403",
              "390/410",
              "360/440",
              "3.56%",
              "time",
              "+0.43%",
              "+0.38%"
            ],
            [
              "2026-06-12",
              "2026-07-07",
              "TITAN",
              "2026-07-28",
              "4,184",
              "4,100/4,300",
              "3,800/4,600",
              "2.97%",
              "time",
              "-2.14%",
              "-2.21%"
            ],
            [
              "2026-07-10",
              "2026-08-04",
              "TRENT",
              "2026-08-25",
              "2,900",
              "2,800/2,950",
              "2,600/3,200",
              "4.64%",
              "time",
              "+0.02%",
              "-0.07%"
            ],
            [
              "2026-07-10",
              "2026-08-04",
              "KOTAKBANK",
              "2026-08-25",
              "378",
              "370/390",
              "340/420",
              "3.40%",
              "time",
              "+0.71%",
              "+0.67%"
            ],
            [
              "2026-07-10",
              "2026-08-04",
              "DLF",
              "2026-08-25",
              "685",
              "660/700",
              "600/750",
              "4.12%",
              "time",
              "+0.54%",
              "+0.47%"
            ],
            [
              "2026-07-10",
              "2026-08-04",
              "WIPRO",
              "2026-08-25",
              "176",
              "170/180",
              "160/190",
              "3.07%",
              "time",
              "-0.66%",
              "-0.74%"
            ],
            [
              "2026-07-10",
              "2026-08-04",
              "HDFCBANK",
              "2026-08-25",
              "825",
              "800/850",
              "740/910",
              "2.62%",
              "time",
              "-2.85%",
              "-2.92%"
            ],
            [
              "2026-07-10",
              "2026-08-04",
              "PAYTM",
              "2026-08-25",
              "1,340",
              "1,300/1,400",
              "1,200/1,500",
              "4.22%",
              "time",
              "+0.80%",
              "+0.70%"
            ],
            [
              "2026-07-10",
              "2026-08-04",
              "ONGC",
              "2026-08-25",
              "245",
              "240/250",
              "220/265",
              "3.39%",
              "time",
              "+0.79%",
              "+0.74%"
            ],
            [
              "2026-07-10",
              "2026-08-04",
              "ITC",
              "2026-08-25",
              "282",
              "275/290",
              "260/310",
              "2.41%",
              "time",
              "+1.01%",
              "+0.98%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "ADANIENT",
              "2026-09-29",
              "3,037",
              "3,000/3,100",
              "2,800/3,300",
              "3.88%",
              "expiry",
              "+3.88%",
              "+3.83%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "ADANIPORTS",
              "2026-09-29",
              "1,698",
              "1,660/1,740",
              "1,540/1,840",
              "3.03%",
              "expiry",
              "+3.03%",
              "+3.00%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "AXISBANK",
              "2026-09-29",
              "1,217",
              "1,180/1,240",
              "1,100/1,320",
              "2.56%",
              "expiry",
              "+2.56%",
              "+2.54%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "TATASTEEL",
              "2026-09-29",
              "183",
              "180/188",
              "168/200",
              "3.32%",
              "expiry",
              "+3.32%",
              "+3.29%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "BANKBARODA",
              "2026-09-29",
              "249",
              "240/255",
              "220/270",
              "2.74%",
              "expiry",
              "+2.74%",
              "+2.71%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "SBIN",
              "2026-09-29",
              "1,066",
              "1,040/1,090",
              "970/1,160",
              "2.75%",
              "expiry",
              "+2.75%",
              "+2.73%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "PNB",
              "2026-09-29",
              "118",
              "115/120",
              "108/130",
              "3.59%",
              "expiry",
              "+3.59%",
              "+3.56%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "NTPC",
              "2026-09-29",
              "340",
              "330/350",
              "310/375",
              "2.31%",
              "expiry",
              "+2.31%",
              "+2.29%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "MCX",
              "2026-09-29",
              "2,932",
              "2,900/3,000",
              "2,700/3,200",
              "4.35%",
              "expiry",
              "+4.35%",
              "+4.30%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "M&M",
              "2026-09-29",
              "3,426",
              "3,350/3,500",
              "3,100/3,800",
              "3.43%",
              "expiry",
              "+3.43%",
              "+3.40%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "LT",
              "2026-09-29",
              "4,060",
              "4,000/4,200",
              "3,700/4,500",
              "2.51%",
              "expiry",
              "+2.51%",
              "+2.49%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "ITC",
              "2026-09-29",
              "277",
              "270/285",
              "250/305",
              "2.04%",
              "expiry",
              "+2.04%",
              "+2.02%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "INFY",
              "2026-09-29",
              "1,170",
              "1,140/1,200",
              "1,060/1,280",
              "3.35%",
              "expiry",
              "+3.35%",
              "+3.31%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "TATAPOWER",
              "2026-09-29",
              "383",
              "375/390",
              "350/420",
              "3.28%",
              "expiry",
              "+3.28%",
              "+3.25%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "ICICIBANK",
              "2026-09-29",
              "1,414",
              "1,380/1,450",
              "1,300/1,550",
              "2.20%",
              "expiry",
              "+2.20%",
              "+2.18%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "HINDALCO",
              "2026-09-29",
              "1,035",
              "1,000/1,060",
              "920/1,140",
              "2.97%",
              "expiry",
              "+2.97%",
              "+2.94%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "HDFCBANK",
              "2026-09-29",
              "726",
              "710/740",
              "660/790",
              "3.30%",
              "expiry",
              "+3.30%",
              "+3.27%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "HCLTECH",
              "2026-09-29",
              "1,361",
              "1,330/1,400",
              "1,250/1,500",
              "3.02%",
              "expiry",
              "+3.02%",
              "+2.99%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "HAL",
              "2026-09-29",
              "5,011",
              "4,900/5,150",
              "4,500/5,500",
              "3.45%",
              "expiry",
              "+3.45%",
              "+3.41%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "FEDERALBNK",
              "2026-09-29",
              "353",
              "340/360",
              "320/380",
              "2.22%",
              "expiry",
              "+2.22%",
              "+2.20%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "TCS",
              "2026-09-29",
              "2,357",
              "2,300/2,420",
              "2,140/2,580",
              "3.12%",
              "expiry",
              "+3.12%",
              "+3.09%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "BHARTIARTL",
              "2026-09-29",
              "1,991",
              "1,940/2,040",
              "1,800/2,160",
              "2.06%",
              "expiry",
              "+2.06%",
              "+2.05%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "BEL",
              "2026-09-29",
              "410",
              "400/420",
              "370/450",
              "2.95%",
              "expiry",
              "+2.95%",
              "+2.92%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "TITAN",
              "2026-09-29",
              "5,078",
              "5,000/5,200",
              "4,600/5,600",
              "3.04%",
              "expiry",
              "+3.04%",
              "+3.01%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "BAJFINANCE",
              "2026-09-29",
              "1,087",
              "1,050/1,120",
              "980/1,200",
              "2.72%",
              "expiry",
              "+2.72%",
              "+2.70%"
            ],
            [
              "2026-08-14",
              "2026-09-29",
              "WIPRO",
              "2026-09-29",
              "184",
              "180/188",
              "165/200",
              "3.38%",
              "expiry",
              "+3.38%",
              "+3.35%"
            ]
          ]
        }
      ],
      "charts": [
        {
          "src": "/app/stock45_wings_tearsheet.png",
          "caption": "Client tearsheet — 10-slot portfolio at MODELED margin (6.7% of notional) vs NIFTY 50, monthly, 2016-2026. At 2x margin the equity curve compresses to ~20% CAGR / -10% DD. Idle cash at 5% (liquid ETF)."
        }
      ]
    },
    "winners": [
      {
        "config": "C1 — 45→21 DTE ±2.5% strangle, 7% wings, no SL, TP50, liquidity-gated",
        "summary": "The r/119 theta window transfers to stocks; wings turn idiosyncratic gap risk into a capped, priced cost; liquidity gate keeps it real.",
        "metrics": [
          {
            "k": "Net/trade",
            "v": "+0.264% S0 (t 5.06, n 628)"
          },
          {
            "k": "Portfolio (2x-1x margin)",
            "v": "20-38% CAGR, Calmar 1.8-1.9"
          },
          {
            "k": "Diversification",
            "v": "corr NIFTY -0.09; +EV in crash months"
          }
        ],
        "rejected": [
          "30-DTE entry (t=-9)",
          "any premium stop",
          "IV-rank gate",
          "price-action calm gates",
          "5% OTM shorts"
        ]
      }
    ],
    "caveats": [
      "MARGIN IS MODELED, NOT MEASURED. 1.25x max-loss + 2% (~6.7% notional) may understate real SPAN+exposure for stock condors; the x2 row (20.2% CAGR / -10.4% DD) is the conservative claim. Gate to live: real Kite basket-margin check. CAGR scales ~inversely with margin.",
      "Costs are a 0.5%-of-premium proxy — stock options have NO bid/ask history. Break-even ~1.9% of turnover on the composite. Non-top-tier names can be worse; start any live test on the most liquid tier only.",
      "No earnings calendar in the data: earnings gaps inside the hold ARE in the marks (wings cap them) but 'skip earnings cycles' is untested — likely a free improvement once a source exists.",
      "Survivorship: today's F&O list applied to the past; mitigated by the modern sub-period being the STRONGEST era. Pre-2021 the liquid universe is 1-2 names/cycle (portfolio years 2016-20 are noise).",
      "C1 was selected from ~31 configs — the raw t=5.06 is inflated by selection; the robustness gauntlet (drop-top-5 t=3.49, era splits t~2.5) is the deflated evidence.",
      "Bhav closes are settle-ish marks; untraded wing marks valued at 0 on exit (pessimistic for us). Entry at same-day close; next-session lag keeps t=3.53.",
      "87 monthly cycles, one macro regime (no 2008-style event; Mar-2020 thinly sampled). Worst in-sample month -9.9% at modeled margin; a multi-stock gap event could exceed it — max loss if ALL 10 slots hit max-loss simultaneously is ~-55% of capital (wings cap it there)."
    ],
    "githubLinks": [
      {
        "label": "research/127 — scripts + RESULTS.md",
        "href": "https://github.com/castroarun/Quantifyd/tree/main/research/127_stock_neutral_wings"
      },
      {
        "label": "research/119 — the NIFTY 45-DTE parent study",
        "href": "https://github.com/castroarun/Quantifyd/tree/main/research/119_45dte_short_straddle"
      }
    ],
    "projectPaths": [
      "research/127_stock_neutral_wings/STOCK_NEUTRAL_WINGED_STRADDLE_DAILY_SWEEP_STATUS.md",
      "research/127_stock_neutral_wings/results/RESULTS.md",
      "research/127_stock_neutral_wings/results/iv_daily.csv (per-stock daily ATM-IV series, reusable)"
    ]
  },

  {
    slug: 'bluesky-ath-breakout-research142',
    title: 'BananaPatterns "Blue Sky" ATH Breakout — forensic replication + 20-year robustness',
    verdict:
      'bananapatterns.com showcases a blue-sky breakout backtest at 79.8% CAGR / -11.4% worst fall (2020-25, marked PROVISIONAL by the site itself). We decoded the engine trade-exactly (37/39 exits reproduced to the day and paisa; entries = buy-stop at the prior all-time-high CLOSE; RS = IBD-weighted momentum percentile >= 70) — the rules are real, the published numbers are not reproducible: six honest selection paths land 6.5-15.7x vs their 33.7x, and their -11.4% drawdown is unreachable at ANY marking frequency (best honest: -22% daily / -15% monthly). The construction itself, however, survives 20 years: 2006-2025 net of 25bps with realistic fills, a point-in-time Rs 500cr mcap floor and the NIFTYBEES<SMA200 gate, a 10-seed selection ensemble medians 30.4% CAGR (range 27.9-34.4%) at -31.5% median MaxDD — ~203x vs NIFTYBEES 10.3x (12.3%, -59.7%). Same tier as the research/75 momentum book (31.9% net). STRATEGY family credible; single-path CAGRs above ~40% from this design are selection luck, not edge.',
    status: 'COMPLETE',
    date: '2026-09-02',
    cardBlurb:
      'Reverse-engineered a viral backtest site trade-by-trade, proved its rules and refuted its headline numbers, then gave the decoded system an honest 20-year, net-of-cost, seed-ensemble examination against Nifty 50, Midcap 150 and Smallcap 250.',
    cardStats: [
      { label: 'CAGR (net, median)', value: '30.4%' },
      { label: 'MaxDD (median)', value: '-31.5%' },
      { label: '2006-25 multiple', value: '~203x' },
    ],
    system: {
      intro:
        'The decoded rules (validated against 90+ of their published trades before any long backtest was run):',
      rows: [
        { k: 'Universe', v: 'All NSE EQ dailies (2,321 syms post repair); liquidity floor 20d-median traded value >= Rs 5cr; ETFs excluded; headline config adds mcap >= Rs 500cr point-in-time (constant-adjusted-shares proxy).' },
        { k: 'Setup', v: 'Prev close within 20% of the all-time-high CLOSE and below it; IBD-weighted RS percentile (2xr63+r126+r189+r252) >= 70.' },
        { k: 'Signal', v: 'A CLOSE above the prior ATH-close ("a close into new highs") — 44/45 of their entry days confirm; fill booked AT the pivot (their convention; our honest runs fill at max(open, pivot)).' },
        { k: 'Exits', v: '-8% stop on the close (gap-aware); trail = exit at the close that breaks the 50-SMA.' },
        { k: 'Book', v: '8 slots, 18.75% of equity per position (1.5% risk / 8% stop), cash-constrained, pyramiding allowed; selection among simultaneous signals is UNDISCLOSED -> we report a 10-seed random-selection ensemble, not one lucky path.' },
        { k: 'Gate', v: 'Headline config blocks new entries while NIFTYBEES < its SMA200.' },
      ],
    },
    conditions: {
      intro: 'Two examinations: faithful 2020-25 replication vs their claims, then 2006-2025 robustness.',
      rows: [
        { k: 'Phase 2 (their window)', v: '2020-01-01 to 2025-12-31, Rs 10L, their fills, no costs — apples-to-apples vs the site.' },
        { k: 'Phase 3 (robustness)', v: '2006-01-01 to 2025-12-31, realistic fills, 25bps/side, 10 random-selection seeds per config.' },
        { k: 'Host / data', v: 'VPS market_data.db (post split-adjustment repair, 2026-09-01); scripts in research/142.' },
      ],
    },
    comparisons: [
      {
        title: 'Their window (2020-25): published vs honest replication of the SAME rules',
        columns: ['Run', 'Terminal', 'CAGR', 'MaxDD', 'Trades', 'Win%'],
        rows: [
          ['Site published (PROVISIONAL)', '33.74x', '79.8%', '-11.4% "worst fall"', '272', '52%'],
          ['Faithful replica (RS-desc picks)', '11.01x', '49.2%', '-31.5%', '175', '42%'],
          ['Random-selection seeds (range of 5)', '6.5-15.1x', '37-57%', '-22 to -32%', '~180', '44-46%'],
          ['Best variant: weak-market gate ON', '15.73x', '58.3%', '-22.0%', '141', '48%'],
          ['Config D on this window (gate + real fills + 25bps + mcap>=500cr), 10-seed median [range]', '6.69x [4.41..14.21]', '37.3% [28.1..55.6]', '-24.6% (worst -36.4%)', '~148', '42-50%'],
          ['Config B on this window (same, NO mcap floor), 10-seed median [range]', '6.79x [3.07..11.31]', '37.6% [20.6..49.8]', '-22.2% (worst -35.5%)', '~150', '44-50%'],
        ],
        highlightRows: [4],
        heatmap: false,
      },
      {
        title: '20-year robustness (2006-25) — 10-seed ensemble medians [min..max]',
        caption: 'The gate is the Skip-weak-markets idea switched ON: no new positions are opened while NIFTYBEES (the Nifty 50 ETF) closes below its 200-day moving average; open positions keep the normal -8% stop and 50-SMA trail. Real fills = entry at max(open, pivot) rather than always at the pivot. Net 25bps = a modelled trading cost of 25 basis points (0.25% of trade value) charged on EACH side, buy and sell - covering brokerage, STT and slippage. Point-in-time mcap = market cap on the signal date via the constant-adjusted-shares proxy. All configs share the decoded core rules: ATH-close trigger, IBD-RS>=70, Rs 5cr/day liquidity floor, 8 slots.',
        columns: ['Config', 'Terminal x', 'CAGR', 'MaxDD', 'Signals'],
        rows: [
          ['A: gate ON, their fills, gross', '398 [228..813]', '34.9% [31.2..39.8]', '-44.0%', '16,612'],
          ['B: gate ON, real fills, net 25bps', '287 [138..758]', '32.7% [28.0..39.3]', '-45.7%', '16,612'],
          ['C: gate OFF, real fills, net 25bps', '225 [108..413]', '31.1% [26.4..35.2]', '-45.0%', '16,612'],
          ['D (HEADLINE): gate ON + real fills + net 25bps + mcap >= Rs 500cr point-in-time', '203 [136..367]', '30.4% [27.9..34.4]', '-31.5%', '8,069'],
          ['D extended through Aug 2026 (includes the 2026 correction)', '146 [81..247]', '27.3% [23.7..30.6]', '-34.9%', '8,213+'],
          ['Benchmark: NIFTYBEES B&H', '10.25', '12.3%', '-59.7%', '—'],
          ['Reference: research/75 momentum (net)', '—', '31.9%', '-31.6%', '—'],
        ],
        highlightRows: [3],
        heatmap: false,
      },
      {
        title: 'Phase-4 optimization (CORRECTED harness; 2006 - Aug 2026; 10-seed ensemble medians [range])',
        caption: 'Full disclosure: the first-published sweep numbers were inflated by a harness bug (the trail-SMA was recomputed on a calendar-aligned matrix, silently disabling trail exits around non-traded days). Every decision cell below was re-verified on the corrected engine. Adoption rule: only improvements that are consistent across BOTH decades (2006-15 and 2016-26) count.',
        columns: ['Cell', 'Terminal x', 'CAGR', 'MaxDD', 'Verdict'],
        rows: [
          ['ADOPTED SPEC: decoded rules, NO mcap floor (Rs 5cr/day liquidity floor does the work), gate ON, real fills, net 25bps', '301 [107..727]', '31.8% [25.4..37.6]', '-45.7% (worst -50.0)', 'the headline'],
          ['+ mcap >= Rs 500cr floor (FULL snapshot, 2,042/2,321 known)', '138 [105..263]', '26.9%', '-43.5%', 'REJECTED - pure return drag, no risk benefit'],
          ['- remove the -8% stop (trail only)', '204 [85..416]', '29.3%', '-44.5%', 'REJECTED on the corrected engine - keep the stop'],
          ['Trail 15 / 20-SMA instead of 50 (pre-tax)', '813 / 536', '38.3 / 35.5%', '-32.6 / -29.8%', 'IS/OOS split: return edge only post-2016 (2006-15 it loses 28.3% vs 30.8%); DD benefit consistent both halves. Decision deferred to the after-tax test below.'],
          ['AFTER-TAX test (20% STCG / 12.5% LTCG on net realized gains): trail 50 vs 20 vs 15', '113 / 165 / 217', '25.7 / 28.0 / 29.8%', '-47.8 / -33.4 / -36.3%', 'ADOPT TRAIL-20 for taxable live use: tax scales with gains so the ranking survives; +2.3pp net-of-tax over trail-50 with 14pp less drawdown, and the IS-decade return gap roughly equalizes after tax. Trail-15 not taken (further churn, edge more recent-era).'],
          ['Slots 10-20, smaller-size x more-slots, RS 60-90, basing depth, gate DMA 0-250', '-', '-', '-', 'inert or no clean dose-response - unchanged'],
          ['Adaptive mcap floor by regime', '-', '-', '-', 'MOOT by construction: the gate blocks weak-day entries, so floor-only-when-weak = no floor'],
        ],
        highlightRows: [0],
        heatmap: false,
      },
      {
        title: 'Capstone: 50-50 blend with our Momentum book (research/75), monthly rebalanced, 2006 - Jul 2026',
        caption: 'Daily return correlation 0.29, monthly 0.52 - same momentum family, far from the same bet. The blend beats BOTH legs on CAGR while its drawdown is shallower than both: staggered bad patches (MaxDD is a worst-moment statistic, and their worst moments differ) plus the monthly rebalancing premium and lower volatility drag. Caveat: correlations converge in crashes - both books are long smallcap momentum.',
        columns: ['Book', 'Terminal x', 'CAGR', 'MaxDD'],
        rows: [
          ['BlueSky adopted spec alone (median seed)', '263', '31.1%', '-43.9%'],
          ['Momentum r/75 alone (armed spec, net)', '341', '32.8%', '-32.8%'],
          ['50-50 blend, monthly rebalance (median of 10 blend seeds)', '348', '33.0% [30.1..36.1]', '-27.5% (worst -32.3%)'],
        ],
        highlightRows: [2],
        heatmap: false,
      },
      {
        title: 'Per-year: BlueSky ADOPTED SPEC (ensemble median, no mcap floor, to Aug 2026) vs our Momentum book (research/75) vs Nifty 50',
        caption: 'BlueSky = median across the 10 selection seeds of the 2006 to Aug-2026 run. Momentum = research/75 armed spec NAV (net, 20y validated). NIFTYBEES = Nifty 50 ETF incl dividends. * 2026 is year-to-date (Aug 31 / Jul 21 for momentum). The two systems track each other closely - same momentum family - but BlueSky was FLAT-to-down where momentum stayed positive in 2011 and 2016 tells you they are not identical bets.',
        columns: ['Year', 'BlueSky D %', 'Momentum r/75 %', 'NIFTYBEES %'],
        rows: [
          ['2006', '+60.0', '+149.8', '+41.3'],
          ['2007', '+129.9', '+98.5', '+53.0'],
          ['2008', '-17.4', '-25.8', '-52.1'],
          ['2009', '+58.4', '+70.3', '+75.6'],
          ['2010', '+10.3', '+31.3', '+18.6'],
          ['2011', '+12.2', '-6.4', '-24.0'],
          ['2012', '+41.9', '+64.8', '+26.5'],
          ['2013', '+7.8', '+0.7', '+7.2'],
          ['2014', '+65.0', '+96.7', '+31.6'],
          ['2015', '+6.7', '-9.2', '-4.3'],
          ['2016', '-1.4', '+47.1', '+4.0'],
          ['2017', '+150.8', '+85.4', '+29.9'],
          ['2018', '-19.1', '-17.9', '+4.8'],
          ['2019', '-7.4', '-6.4', '+13.6'],
          ['2020', '+50.7', '+60.0', '+15.4'],
          ['2021', '+143.4', '+74.5', '+26.0'],
          ['2022', '+3.8', '-4.2', '+5.5'],
          ['2023', '+68.7', '+56.8', '+21.0'],
          ['2024', '+57.0', '+53.7', '+10.4'],
          ['2025', '+14.7', '+9.4', '+11.7'],
          ['2026*', '+2.0', '+1.1', '-7.3'],
        ],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'Adopted spec CAGR (net, 10-seed median)', value: '31.8%', tone: 'pos', hint: 'range 25.4-37.6%; 2006 - Aug 2026; no mcap floor' },
        { label: 'MaxDD (median)', value: '-45.7%', tone: 'neg', hint: 'worst seed -50.0%' },
        { label: '20.7-year multiple', value: '~301x', hint: 'vs NIFTYBEES 9.5x (11.5%, -59.7%)' },
        { label: '50-50 with Momentum r/75', value: '33.0% @ -27.5%', tone: 'pos', hint: 'the capstone construction - beats both legs on CAGR and DD' },
        { label: 'Site claim reproduced?', value: 'NO', tone: 'neg', hint: '79.8%/-11.4% unreachable under any honest variant' },
        { label: 'Trade-rule match', value: '37/39 exits exact', tone: 'pos', hint: 'to the day and paisa' },
      ],
      tables: [],
      charts: [
        { src: '/app/bluesky-breakout-tearsheet.png', caption: 'Client factsheet — headline config D (median seed of the 10-seed ensemble) vs NIFTYBEES, 2006-2025, net of 25bps/side.' },
        { src: '/app/bluesky-vs-indices.png', caption: 'Growth of Rs 100 (log), 2011-2025: BlueSky replica (median seed, net) vs NIFTYBEES, NIFTYMIDCAP150, NIFTYSMLCAP250.' },
      ],
      embeds: [
        { src: '/app/bluesky-breakout-tearsheet.html', height: 900, caption: 'Interactive tearsheet (self-contained).' },
      ],
    },
    winners: [
      {
        config: 'ADOPTED SPEC - decoded rules + Rs 5cr/day liquidity floor (no mcap floor), 200-DMA gate ON, -8% stop kept, 50-SMA trail, realistic fills, net 25bps',
        summary: 'Locked after the corrected-harness verification pass: 301x / 31.8% / -45.7% medians over 2006 - Aug 2026. The best CONSTRUCTION found is not a config tweak at all - it is the 50-50 monthly-rebalanced blend with the research/75 Momentum book: 33.0% CAGR at -27.5% DD, beating both legs on both numbers.',
        metrics: [
          { k: 'Adopted spec', v: '301x / 31.8% [25.4..37.6] / -45.7%' },
          { k: '50-50 blend w/ momentum', v: '348x / 33.0% / -27.5% (corr 0.29 daily)' },
          { k: 'Churn', v: '~4.5x book/yr, all STCG (momentum leg: 0.38x, LTCG-eligible)' },
          { k: 'Taxable-account pick', v: 'trail-20 variant: 28.0% after-tax vs 25.7% (trail-50), DD -33.4% vs -47.8%' },
        ],
        rejected: [
          'The site\'s -11.4% risk claim (unreachable at any marking frequency)',
          'One-shot-per-pivot dedupe (collapsed performance: 2.1x, 32% win)',
          'Plain 252d RS (fails 9 of their own published trades)',
          'Single-path CAGRs >40% from this design (selection luck: seed spread 6.5-15.7x on 2020-25)',
        ],
      },
    ],
    caveats: [
      'HARNESS BUG disclosed: the first Phase-4 sweep used a calendar-aligned trail-SMA that silently disabled trail exits near non-traded days, inflating its numbers (e.g. no-floor 517x -> corrected 301x). All adoption decisions were re-made on the corrected engine; the IS/OOS split (2006-15 vs 2016-26) is now the adoption gate.',
      'Phase-4 note: the original "mcap floor = risk filter" claim was published, then CORRECTED when the completed mcap snapshot showed the drawdown benefit came from accidentally excluding unknown-mcap symbols, not from the Rs 500cr floor itself.',
      'SURVIVORSHIP: Kite lists only current instruments — 2006 coverage is 528 symbols, all of which survived to 2026. Pre-~2015 years (esp. 2006-07 at +43/+117%) are inflated by this. The DD and post-2015 years are the more trustworthy part.',
      'Mcap floor is a proxy: constant adjusted-shares from a 2026 yfinance snapshot (split-safe; wrong for heavy diluters), known for only 925/2,321 symbols — unknowns excluded.',
      'Selection among simultaneous signals is undisclosed by the site; all our numbers are 10-seed ensembles — trust the medians and ranges, not any single path.',
      'No STCG tax modelled (median hold is weeks — tax materially reduces net for a taxable account).',
      'Their published trade list remains only partially recallable (2-6/54 same-day matches) due to slot path-divergence, even though 48/51 of their trades pass every decoded condition.',
      '2025 in config D is +10.3% with the 2026 YTD tape negative — the smallcap-breadth regime this feeds on has cooled; paper-trade before capital.',
    ],
    githubLinks: [{ label: 'research/142 (repo)', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/142_bananapatterns_replication' }],
    projectPaths: [
      'research\\142_bananapatterns_replication\\BANANAPATTERNS_BLUESKY_TRADE_MATCH_DAILY_FORENSIC_STATUS.md',
      'research\\142_bananapatterns_replication\\scripts\\ (validate_trades, entry_diag, repair_data, extend_universe, bluesky_replay, make_report).py',
      'research\\142_bananapatterns_replication\\results\\RESULTS.md',
    ],
  },
];

export function getStudy(slug: string): BacktestStudy | undefined {
  return BACKTEST_STUDIES.find((s) => s.slug === slug);
}
