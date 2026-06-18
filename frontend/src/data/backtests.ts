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
    slug: 'nifty-weekly-cpr-playbook',
    title: 'NIFTY Weekly CPR — Directional Playbook (weekly × daily CPR + 1st-30-min candle)',
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
