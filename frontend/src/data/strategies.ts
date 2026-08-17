/**
 * THE STRATEGIES REGISTER — register of record for every system we run.
 *
 * Binding rule (.claude/CLAUDE.md): whenever a system goes live, goes to paper,
 * is parked, changes size, or has ANY rule changed, this file is updated in the
 * same commit — status, size, the rule line, the studies, and a dated changeLog
 * entry. A rule change that is not reflected here is not considered deployed.
 *
 * Grouping is by whose money is at risk, not by asset class: that is the
 * question the page exists to answer.
 *
 * `dayPnlFeed` is optional and only set where the app already has a live feed;
 * a system without one shows "—" in the register and is read on its dashboard.
 */

export type SystemStatus = 'live' | 'paper' | 'parked';

/** Which live feed fills the Day column. Wired in Strategies.tsx. */
export type DayPnlFeed = 'orb' | 'strangle' | 'nas-nifty' | 'nas-sensex';

export interface StudyRef {
  /** Published study — resolves to /app/backtest/<slug>. */
  slug: string;
  title: string;
  /** Verdict label, only where the study states one plainly. */
  verdict?: 'STRATEGY' | 'STRATEGY-CANDIDATE' | 'SIGNAL' | 'NO EDGE' | 'CONCLUDED';
}

export interface StrategySystem {
  id: string;
  name: string;
  /** Venue + structure, one short line. */
  subtitle: string;
  status: SystemStatus;
  /** Position size as traded: lots + qty, or capital. */
  size: string;
  /** When this status started. */
  since: string;
  /** The rule in one line — what the register shows. */
  rule: string;
  /** The rules as they run, for the spec pane. */
  rules: Array<[string, string]>;
  /** Source of truth doc for those rules (repo path — not a URL). */
  rulesDoc?: string;
  /** In-app dashboard route. */
  dashboard?: string;
  studies: StudyRef[];
  /** Why a study slot is empty, when it is. */
  studyGap?: string;
  changeLog: Array<{ date: string; text: string }>;
  dayPnlFeed?: DayPnlFeed;
  /** Anything the row would mislead without. */
  note?: string;
}

export const STATUS_LABEL: Record<SystemStatus, string> = {
  live: 'Live · real money',
  paper: 'Paper · validating',
  parked: 'Parked · not trading',
};

export const REGISTER_UPDATED = '17 Aug 2026';

export const SYSTEMS: StrategySystem[] = [
  // ------------------------------------------------------------------ LIVE
  {
    id: 'nas-nifty',
    name: 'NAS Suite · NIFTY',
    subtitle: 'NFO weeklies · short ATM straddle · ATM / ATM2 / ATM4 + 09:16 one-shot',
    status: 'live',
    size: '2 lots · qty 130 per arm',
    since: '07 Jul 2026',
    rule:
      'Short ATM straddle at 09:16 (squeeze arms cascade later); per-leg SL 30%, ATM2 on a ₹2,500/lot rupee stop; 9:16-book stop −₹1,300/lot; square off 15:15.',
    rules: [
      ['Universe', 'NIFTY weekly options, ATM strike at entry'],
      ['Entry', '09:16 one-shot; ATR-squeeze arms cascade up to 5 entries, one per candle'],
      ['Per-leg stop', '30% of leg premium'],
      ['ATM2 stop', '₹2,500 per lot — rupee-denominated, DTE-agnostic'],
      ['Book stop', '−₹1,300 per lot, proportional across open legs'],
      ['Naked survivor', 'SuperTrend(7,3) intrabar trail, 3-tick confirm'],
      ['Exit', '15:15 square-off, EOD backstop after'],
      ['Kill switch', 'POST /api/nas/kill-switch → whole suite to paper'],
    ],
    rulesDoc: 'docs/LABS_AND_JOBS_REFERENCE.md + research/52_nas_optimization',
    dashboard: '/nas',
    dayPnlFeed: 'nas-nifty',
    studies: [
      { slug: 'nasopt-full-replay', title: 'NAS-OPT replayed on every recorded chain day — is the 0/1-DTE gate the edge?', verdict: 'STRATEGY' },
      { slug: 'nas-sl-reanchor', title: 'Is the 30% stop too loose? SL tightening / re-anchoring', verdict: 'NO EDGE' },
      { slug: 'fardte-rescue', title: 'Rescuing the far-from-expiry days — five ideas, four dead, one that works', verdict: 'SIGNAL' },
    ],
    changeLog: [
      { date: '17 Aug 2026', text: 'ST-trail confirm-counter bug fixed (counter lived in a throwaway local, never reached 3/3).' },
      { date: '14 Aug 2026', text: 'Kite MARKET-order rejection handled; sleeves fall back to paper safely.' },
      { date: '07 Jul 2026', text: '09:16 arms armed with real money.' },
    ],
  },
  {
    id: 'nas-sensex',
    name: 'NAS Suite · SENSEX',
    subtitle: 'BFO weeklies · short ATM straddle · ATM / ATM2 / ATM4',
    status: 'live',
    size: '3 lots · qty 60 per arm',
    since: '05 Aug 2026',
    rule:
      'Same 09:16 entry on BFO: SuperTrend(7,3) trail clamped at breakeven, venue stop −₹11,700, TP +₹15,003. DTE-aware — hold through Thursday expiry, size down Wednesday.',
    rules: [
      ['Universe', 'SENSEX weekly options, ATM strike at entry'],
      ['Entry', '09:16, parallel with the NIFTY arms'],
      ['Trail', 'SuperTrend(7,3) on BFO, clamped at breakeven'],
      ['Venue stop', '−₹11,700 for the venue book'],
      ['Take profit', '+₹15,003 — SENSEX fades rather than trends'],
      ['DTE policy', 'DTE0 (Thu) hold — a stop sabotages expiry decay; DTE1 (Wed) is the fat-tail day, size down'],
      ['Exit', '15:15 square-off, EOD backstop after'],
    ],
    rulesDoc: 'research/111_sensex_manual_mgmt + research/90_nas_portfolio_stop',
    dashboard: '/nas',
    dayPnlFeed: 'nas-sensex',
    studies: [
      { slug: 'sensex-nifty-stop-by-dte', title: 'Stop calibration by DTE (combined vs per-leg, real 1-min chain)' },
      { slug: 'nifty-straddle-lookahead-audit', title: 'Straddle & iron-fly look-ahead audit — honest comparison' },
    ],
    changeLog: [
      { date: '13 Aug 2026', text: 'DTE study: Thursday = hold (no stop), Wednesday = tight combined-20%, DTE2+ keeps per-leg 30%.' },
      { date: '05 Aug 2026', text: 'Scaled to 3 lots; ST(7,3) trail + venue TP added.' },
    ],
    note: 'SENSEX weeklies only exist from 2024 — every SENSEX verdict rests on a calm, low-VIX sample.',
  },
  {
    id: 'csl-timeb-nifty',
    name: 'TB-CSL · NIFTY',
    subtitle: 'NFO · time-boxed short straddle · CSL_TIMEB_NIFTY',
    status: 'live',
    size: '6 lots · qty 390',
    since: '14 Aug 2026',
    rule:
      'Entry window, exit and combined-SL frozen per DTE from the Best-Config Lab; the Friday sweep re-reads the window but never moves the live config on its own.',
    rules: [
      ['Universe', 'NIFTY weekly ATM straddle'],
      ['Entry', 'Per-DTE entry time, frozen from the Best-Config Lab'],
      ['Stop', 'Per-DTE combined-SL from the same frozen config'],
      ['Exit', 'Time-boxed exit per DTE'],
      ['Config policy', 'Friday sweep is informational — a config change is a deliberate commit, never automatic'],
      ['Margin gate', 'Skips entry when broker margin is short of the requirement + headroom'],
    ],
    rulesDoc: 'research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py (BOOKS)',
    dashboard: '/straddles',
    studies: [
      { slug: 'csl-best-config-straddles', title: 'CSL best-config straddles — entry × exit × combined-SL per DTE' },
    ],
    changeLog: [
      { date: '17 Aug 2026', text: 'First 6-lot REAL window (13:00 entry) observed.' },
      { date: '14 Aug 2026', text: 'Promoted to real money at 6 lots (qty 390).' },
    ],
  },
  {
    id: 'nas-comb20',
    name: 'NAS_COMB20',
    subtitle: 'NFO · combined-20% stop sleeve',
    status: 'live',
    size: '2 lots · qty 130',
    since: '14 Aug 2026',
    rule:
      'One 09:16 straddle stopped on combined premium +20% rather than per leg — the DTE-1 arm of the stop-calibration study, running with real money.',
    rules: [
      ['Universe', 'NIFTY weekly ATM straddle'],
      ['Entry', '09:16, one straddle'],
      ['Stop', 'Combined premium +20% (both legs together)'],
      ['Exit', 'Time exit per config, or the combined stop'],
      ['Why live', 'Combined-SL won on DTE1 in the DTE study where per-leg 30% did not'],
    ],
    rulesDoc: 'research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py (BOOKS)',
    dashboard: '/straddles',
    studies: [
      { slug: 'sensex-nifty-stop-by-dte', title: 'Stop calibration by DTE — where combined-SL beats per-leg' },
    ],
    changeLog: [
      { date: '17 Aug 2026', text: 'Account-level book stop bought back COMB\'s CE without COMB knowing — unified per-system ledger pending (TODO 2026-08-24).' },
      { date: '14 Aug 2026', text: 'Armed with real money at 2 lots.' },
    ],
    note: 'No per-book pause exists yet: halting this sleeve currently means killing the CSL daemon.',
  },
  {
    id: 'momentum-3l',
    name: 'Momentum ₹3L',
    subtitle: 'NSE cash · Nifty 200 momentum, hold 30',
    status: 'live',
    size: '₹3,00,000',
    since: '24 Jul 2026',
    rule:
      'Rank by blended momentum and hold 30 names, only while the weekly index gate holds; Donchian trailing stop per name, monthly rebalance with a buffer.',
    rules: [
      ['Universe', 'NSE Nifty 200'],
      ['Selection', 'Blended momentum rank, top 30'],
      ['Gate', 'Weekly index gate — the whole drawdown story sits here, not in the stop'],
      ['Stop', 'Daily Donchian trailing stop per name'],
      ['Rebalance', 'Monthly, with a buffer to suppress churn'],
      ['Top-ups', 'Immediate equal top-up on deposit; broker-qty assert before any sell'],
    ],
    rulesDoc: 'services/momentum_paper.py + research/75_nifty250_momentum_top15',
    dashboard: '/momentum-paper',
    studies: [
      { slug: 'nifty250-momentum-video-research75', title: 'Nifty-250 Momentum Top-15 — faithful survivorship-free replication' },
      { slug: 'momentum-universe-bakeoff', title: 'Universe bake-off — Nifty 200 vs 250 vs 51-250 vs Midcap 150 vs 500' },
      { slug: 'momentum-250-leverage-frontier', title: 'Leverage frontier — how far can this book be pushed?' },
      { slug: 'momentum-put-hedge-overlay', title: 'Put-hedge overlay vs the cash-exit gate', verdict: 'SIGNAL' },
    ],
    changeLog: [
      { date: '17 Aug 2026', text: 'Moved under Holdings in the sidebar — it is real money, not a paper book.' },
      { date: '16 Aug 2026', text: 'Top-up ledger corruption fixed and deployed (INSERT OR REPLACE wiped qty/cost/entry_date).' },
      { date: '24 Jul 2026', text: 'Funded with real money at ₹3L.' },
    ],
  },

  // ----------------------------------------------------------------- PAPER
  {
    id: 'csl-timeb-sensex',
    name: 'TB-CSL · SENSEX',
    subtitle: 'BFO · time-boxed short straddle · CSL_TIMEB_SENSEX',
    status: 'paper',
    size: '6 lots · qty 120',
    since: '14 Aug 2026',
    rule: 'The NIFTY book\'s twin on BFO, same frozen per-DTE config. Waiting on 6-lot evidence before it earns real money.',
    rules: [
      ['Universe', 'SENSEX weekly ATM straddle'],
      ['Entry / stop / exit', 'Per-DTE frozen config, same lab as the NIFTY twin'],
      ['Promotion gate', 'Needs its own REAL-window evidence — the NIFTY result does not transfer'],
    ],
    rulesDoc: 'research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py (BOOKS)',
    dashboard: '/straddles',
    studies: [
      { slug: 'csl-best-config-straddles', title: 'CSL best-config straddles — entry × exit × combined-SL per DTE' },
    ],
    changeLog: [{ date: '14 Aug 2026', text: 'Sized to 6 lots on paper alongside the NIFTY promotion.' }],
  },
  {
    id: 'csl30f',
    name: 'CSL30F · NIFTY / SENSEX',
    subtitle: 'Control arm · fixed 30% per-leg stop',
    status: 'paper',
    size: 'NIFTY 2 lots · SENSEX 3 lots',
    since: '14 Aug 2026',
    rule: 'The uniform 30% per-leg stop the DTE study is trying to beat — kept running so every change has a baseline to be measured against.',
    rules: [
      ['Universe', 'NIFTY + SENSEX weekly ATM straddle'],
      ['Entry', '09:16'],
      ['Stop', 'Fixed 30% per leg, every DTE'],
      ['Role', 'Control arm — never optimised, deliberately'],
    ],
    rulesDoc: 'research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py (BOOKS)',
    dashboard: '/straddles',
    studies: [{ slug: 'sensex-nifty-stop-by-dte', title: 'Stop calibration by DTE — this is the arm being beaten' }],
    changeLog: [{ date: '14 Aug 2026', text: 'Running as the fixed-30% control alongside the live sleeves.' }],
  },
  {
    id: 'nas-c20-mgmt',
    name: 'NAS_C20_TRAIL / _SHIFT',
    subtitle: 'Post-entry management arms on COMB20',
    status: 'paper',
    size: '2 lots each',
    since: '14 Aug 2026',
    rule: 'COMB20 with a trail, and COMB20 that shifts the untested strike — isolating whether management adds anything once the stop is already right.',
    rules: [
      ['Base', 'NAS_COMB20 (combined 20% stop)'],
      ['TRAIL arm', 'Trails the combined position once in profit'],
      ['SHIFT arm', 'Shifts the untested leg\'s strike instead of trailing'],
      ['Read', 'Both are measured against the plain COMB20 sleeve, not against each other'],
    ],
    rulesDoc: 'research/111_sensex_manual_mgmt/scripts/csl_mgmt_replay.py',
    dashboard: '/straddles',
    studies: [{ slug: 'csl-best-config-straddles', title: 'CSL best-config lab — where management arms are scored' }],
    changeLog: [{ date: '14 Aug 2026', text: 'Both management arms started on paper.' }],
  },
  {
    id: 'ha-paper',
    name: 'HA 2-Green ₹20L',
    subtitle: 'NSE cash · 30-min Heikin-Ashi break · 81 sleeves',
    status: 'paper',
    size: '₹20,00,000',
    since: '21 Jul 2026',
    rule: 'Two green Heikin-Ashi bars with no lower wick, entry on the break. First full survivor of the research/81–86 sweep; 81 sleeves run in parallel.',
    rules: [
      ['Universe', 'NSE cash, F&O names'],
      ['Signal', 'Two consecutive green HA bars, no lower wick, on 30-min'],
      ['Entry', 'Break of the signal bar'],
      ['Sleeves', '81 parameter sleeves in parallel — the soak IS the test'],
      ['Review', 'Soak review due ~Oct 2026'],
    ],
    rulesDoc: 'services/ha_paper.py + research/86_heikin_patterns',
    dashboard: '/ha-paper',
    studies: [],
    studyGap: 'research/86 written up but not yet published to /app/backtest.',
    changeLog: [{ date: '21 Jul 2026', text: 'Paper book started at ₹20L, 81 sleeves.' }],
  },
  {
    id: 'fnoms-paper',
    name: 'F&O Multi-Signal ₹20L',
    subtitle: 'NSE cash · several signals sharing one book',
    status: 'paper',
    size: '₹20,00,000',
    since: '02 Aug 2026',
    rule: 'Several independent entry signals funded from one book, to see which survive together rather than alone.',
    rules: [
      ['Universe', 'F&O cash names'],
      ['Signals', 'Multiple independent entry signals, one shared capital pool'],
      ['Read', 'The question is interaction and capital contention, not per-signal edge'],
    ],
    rulesDoc: 'services/ (multi-signal paper book)',
    dashboard: '/fnoms-paper',
    studies: [],
    studyGap: 'No published study yet — the book is the first pass.',
    changeLog: [{ date: '02 Aug 2026', text: 'Paper book started at ₹20L.' }],
  },
  {
    id: 'breakout-paper',
    name: 'Breakout ₹10L',
    subtitle: 'NSE cash · daily breakout swing',
    status: 'paper',
    size: '₹10,00,000',
    since: '14 Jun 2026',
    rule: 'At most one new breakout a day, only while NIFTY holds its 200-DMA and fewer than 8 names are held; trailing stop, never a profit target.',
    rules: [
      ['Universe', 'NSE cash'],
      ['Signal', 'MTF-bullish volume breakout'],
      ['Gate', 'NIFTY above its 200-DMA — decisive in the exit bake-off'],
      ['Position cap', '≤1 new entry per day, ≤8 held'],
      ['Exit', 'Trailing stop; a fixed profit target was strictly worse'],
    ],
    rulesDoc: 'services/breakout_paper.py + research/71_breakout_exit_bakeoff',
    dashboard: '/breakout-paper',
    studies: [{ slug: 'breakout-mtf-volume-swing', title: 'Breakout swing book — MTF-bullish volume-breakout exit bake-off' }],
    changeLog: [{ date: '14 Jun 2026', text: 'Paper book started at ₹10L.' }],
    note: 'Backtest is survivorship-optimistic — the paper book is the honest read.',
  },
  {
    id: 'orb-paper',
    name: 'ORB Revival ₹10L',
    subtitle: 'NSE cash · multi-day ORB signal',
    status: 'paper',
    size: '₹10,00,000',
    since: '10 Aug 2026',
    rule: 'The multi-day ORB signal that survived the post-mortem — deliberately not the intraday variant that killed the old live book.',
    rules: [
      ['Universe', 'NSE cash'],
      ['Signal', 'Multi-day opening-range breakout (+16–22 bps 2024–26, in-sample)'],
      ['Explicitly not', 'The intraday ORB variant — negative in every era tested'],
      ['Gate', 'Revival runs on paper only until it earns promotion'],
    ],
    rulesDoc: 'research/89_orb_reassessment',
    dashboard: '/orb-paper',
    studies: [],
    studyGap: 'research/89 reassessment not yet published to /app/backtest.',
    changeLog: [{ date: '10 Aug 2026', text: 'Gated paper revival started at ₹10L.' }],
  },
  {
    id: 'ohol-paper',
    name: 'OHOL 1-Lot',
    subtitle: 'NIFTY · first-candle open-high / open-low',
    status: 'paper',
    size: '1 lot · qty 65',
    since: '10 Aug 2026',
    rule: 'First 5-min candle prints open = high or open = low, trade the continuation. Held to a 1-lot probe on purpose.',
    rules: [
      ['Universe', 'NIFTY'],
      ['Signal', 'First 5-min candle with open = high (short) or open = low (long)'],
      ['Size', 'Deliberately 1 lot — no OHLCV-derived intraday construction has cleared the ~10 bps cost floor'],
      ['Exit', 'Intraday, same session'],
    ],
    rulesDoc: 'research/109_intraday_stocks + research/110_intraday_altinfo',
    dashboard: '/ohol-paper',
    studies: [],
    studyGap: 'research/109 + 110 concluded NO EDGE for the family; this probe is the one exception being watched.',
    changeLog: [{ date: '10 Aug 2026', text: '1-lot paper probe started alongside the ORB revival.' }],
  },
  {
    id: 'nwv',
    name: 'NWV',
    subtitle: 'NIFTY · weekly jade lizard / iron condor',
    status: 'paper',
    size: '1 lot · qty 65',
    since: '27 Jul 2026',
    rule: 'Never roll. Exit the threatened side on a close beyond S1/R2 — the only rule from the study that carried a real t-stat.',
    rules: [
      ['Universe', 'NIFTY weekly options'],
      ['Structure', 'Jade lizard or iron condor'],
      ['Adjustment', 'Never roll — rolling was a disaster in the study'],
      ['Exit', 'Threatened side out on a close beyond S1/R2 (t = 2.48)'],
      ['Directional variants', 'NO EDGE — the structure is not a directional expression'],
    ],
    rulesDoc: 'services/nwv_trade.py + research/94_nwv_jl_ic',
    dashboard: '/nwv',
    studies: [],
    studyGap: 'research/94 not yet published to /app/backtest.',
    changeLog: [{ date: '27 Jul 2026', text: 'Paper book deployed with the never-roll rule.' }],
  },
  {
    id: 'straddle-v1v2',
    name: 'Straddle V1 / V2',
    subtitle: 'NIFTY · positional short straddle & iron fly',
    status: 'paper',
    size: '10 lots · qty 650',
    since: '24 Jun 2026',
    rule: 'V2 is a positional short straddle with ±500 wings; V1 is naked and unhedged, kept only as the baseline V2 is measured against.',
    rules: [
      ['Universe', 'NIFTY weekly / positional options'],
      ['V2', 'Short ATM straddle + ±500 wings (≈2.0% of ATM), SL × VIX tuned'],
      ['V1', 'Naked short straddle — no stop, no wings, unbounded risk by design'],
      ['Entry regime', 'Calm gate + directional skew studied separately'],
      ['Wing caveat', 'The held-wing recorder backtest was invalid (stale far-OTM quotes) — only overnight wings are trustworthy'],
    ],
    rulesDoc: 'research/58_intraday_recenter_straddle + research/60_v2_optimization',
    dashboard: '/straddles',
    studies: [
      { slug: 'v2-nifty-ironfly-sl-vix', title: 'V2 NIFTY positional iron fly — stop-loss × VIX optimization' },
      { slug: 'nifty-fly-calm-directional-entry', title: 'Entry regimes — calm gate + directional skew' },
      { slug: 'nifty-straddle-lookahead-audit', title: 'Look-ahead audit — every variant, honestly re-scored' },
    ],
    changeLog: [{ date: '24 Jun 2026', text: 'V1/V2 paper books running at 10 lots.' }],
  },
  {
    id: 'orb-cash',
    name: 'ORB Cash',
    subtitle: 'NSE cash · OR15 breakout, 15 stocks',
    status: 'paper',
    size: 'paper only',
    since: '17 Aug 2026',
    rule: 'OR15 breakout with VWAP / RSI / CPR filters, 09:14–15:20. Back on in paper mode with a paper-aware margin gate so a near-zero balance no longer blocks every signal.',
    rules: [
      ['Universe', '15 NSE cash names'],
      ['Signal', 'OR15 breakout + VWAP / RSI / CPR filters'],
      ['Session', '09:14 to 15:20'],
      ['Margin gate', 'Paper-aware — a low broker balance no longer blocks paper signals or alerts per skip'],
      ['Promotion gate', 'Stays paper; the multi-day revival book is the funded candidate'],
    ],
    rulesDoc: 'research/89_orb_reassessment',
    dashboard: '/orb',
    dayPnlFeed: 'orb',
    studies: [],
    studyGap: 'research/89 post-mortem not yet published to /app/backtest.',
    changeLog: [
      { date: '17 Aug 2026', text: 'Resumed daily PAPER trading; margin gate made paper-aware.' },
      { date: '10 Aug 2026', text: 'Switched off — noisy per-skip alerts from the margin gate.' },
      { date: '17 Apr 2026', text: 'Went live; later post-mortem showed it traded an unvalidated intraday variant.' },
    ],
  },
  {
    id: 'orb-index',
    name: 'ORB Index',
    subtitle: 'NIFTY · delta-skewed short strangle · 10 variants',
    status: 'paper',
    size: '1 lot per variant',
    since: '—',
    rule: 'ORB break on the index → delta-skewed short strangle (PE −0.22, CE +0.10), across 5/15/30/45/60-min OR windows plus RSI / calm / CPR-against filters.',
    rules: [
      ['Universe', 'NIFTY index options'],
      ['Signal', 'Opening-range break on the index'],
      ['Structure', 'Delta-skewed short strangle — PE −0.22, CE +0.10'],
      ['Variants', '10: OR window × filter (RSI, calm, CPR-against)'],
      ['Status', 'Recording paper trades; no arm has cleared the cost floor'],
    ],
    rulesDoc: 'services/ (strangle recorder)',
    dashboard: '/strangle',
    dayPnlFeed: 'strangle',
    studies: [],
    studyGap: 'No arm has earned a write-up yet.',
    changeLog: [{ date: '—', text: 'Running as a recorder across all 10 variants.' }],
  },

  // ---------------------------------------------------------------- PARKED
  {
    id: 'kc6',
    name: 'KC6',
    subtitle: 'Nifty 500 · Keltner mean reversion',
    status: 'parked',
    size: 'unfunded',
    since: '—',
    rule: 'Close below KC(6, 1.3 ATR) lower while above the 200-SMA; standing SELL LIMIT at the KC mid. Scheduler still runs, strategy is not being pursued.',
    rules: [
      ['Universe', 'Nifty 500'],
      ['Entry', 'Close < KC(6, 1.3 ATR) lower AND close > SMA(200)'],
      ['Primary exit', 'Standing SELL LIMIT at the KC6 mid, placed each morning'],
      ['Other exits', '5% SL · 15% TP · 15-day max hold'],
      ['Crash filter', 'Universe ATR ratio ≥ 1.3× blocks all new entries'],
      ['Why parked', 'Not currently being pursued; the six scheduled jobs still run'],
    ],
    rulesDoc: 'docs/KC6-SESSION-HANDOFF.md',
    dashboard: '/kc6',
    studies: [],
    studyGap: '20-year backtest exists (2,482 trades, PF 1.70) but is not published to /app/backtest.',
    changeLog: [{ date: '—', text: 'Parked per project instructions; migration to /app/kc6 also parked.' }],
    note: 'Journal shows KC6 rows tagged LIVE from the dashboard\'s own mode flag — confirm before reading them as real money.',
  },
  {
    id: 'maruthi',
    name: 'Maruthi dual-SuperTrend',
    subtitle: 'MARUTI · single-name positional',
    status: 'parked',
    size: 'disabled',
    since: '25 Mar 2026',
    rule: 'Dual SuperTrend on MARUTI. Nine correctness bugs found 25 Mar 2026 — do not re-enable until every one is closed.',
    rules: [
      ['Universe', 'MARUTI only'],
      ['Signal', 'Dual SuperTrend'],
      ['Why parked', '9 critical correctness bugs, none closed'],
      ['Re-enable gate', 'All 9 bugs fixed and re-verified'],
    ],
    rulesDoc: 'docs/ (Maruthi design + bug list)',
    studies: [],
    studyGap: 'Single-name study; the REC SuperTrend basket validation killed the same pattern as an overfit.',
    changeLog: [{ date: '25 Mar 2026', text: 'Disabled after 9 correctness bugs were found.' }],
  },
];

/** Labs and research pages that carry no capital — listed so nothing is silently dropped. */
export const LAB_PAGES: Array<{ name: string; to: string; what: string }> = [
  { name: 'MST', to: '/mst', what: 'Multi-signal tracker lab' },
  { name: 'N500M', to: '/n500m', what: 'Nifty 500 momentum lab' },
  { name: 'I75WR', to: '/intraday75wr', what: 'Intraday high-win-rate lab' },
  { name: 'Pairs', to: '/pair-trading', what: 'Pair-trading lab' },
  { name: 'EOD', to: '/eod-breakout', what: 'EOD breakout scan' },
  { name: 'Opt Study', to: '/options-study', what: 'Options decay / CPR / candle aggregates' },
];
