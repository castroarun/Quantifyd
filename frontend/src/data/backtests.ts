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
];

export function getStudy(slug: string): BacktestStudy | undefined {
  return BACKTEST_STUDIES.find((s) => s.slug === slug);
}
