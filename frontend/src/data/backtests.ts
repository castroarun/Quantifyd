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
            'Yearly returns vs Nifty 50 (gross, daily-marked, 2014–2026) — the live de-risk finalists (all-cash + WEEKLY re-entry, keep-top8) plus the all-cash base. Both beat Nifty 50 in 9–10 of 13 years, compound ~34–35% vs Nifty 12.3%, hold MaxDD ~−20% vs Nifty −36%. Robustness (Phase 30): stable across disjoint halves (H1 ~30–31% / H2 ~37–40% CAGR); soft spots are large-cap-led years (2018, 2019, 2025 trail the index — both finalists were NEGATIVE in 2025) and the 2022–2026 third (~17–19%, weakest but still ~2× the index). The named refinements "Keep-8 + Bear Trend-Trim" (recommended) and "Always-On Trend-Guard" sit on the same selection core.',
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
];

export function getStudy(slug: string): BacktestStudy | undefined {
  return BACKTEST_STUDIES.find((s) => s.slug === slug);
}
