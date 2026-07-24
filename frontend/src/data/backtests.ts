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
        { k: 'Roll', v: 'ONE roll-away allowed: re-sell the stopped side at the same Rs-target from current spot. A post-roll stop closes THAT leg only; the surviving leg rides with its own stop (this recovered ~27 pts in the worst replay week).' },
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
];

export function getStudy(slug: string): BacktestStudy | undefined {
  return BACKTEST_STUDIES.find((s) => s.slug === slug);
}
