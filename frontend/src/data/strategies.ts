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

export const REGISTER_UPDATED = '24 Aug 2026';

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
      ['ATM4 roll', 'First SL rolls to a premium-matched strike; rolled-leg SL = max(price_x, roll premium) × 1.3 — never tighter than the survivor anchor or its own 30% (research/113)'],
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
      { date: '18 Aug 2026', text: 'ATM4 rolled-leg SL re-anchored to max(price_x, roll premium) × 1.3 (research/113 MAXV: old 1.3×roll-premium rule was the churniest variant, 32% restop; max() keeps the survivor anchor without going tight on overshoot rolls).' },
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
    size: '2 lots · qty 40 per arm',
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
      { date: '18 Aug 2026', text: 'Wednesday flipped to PAPER (matrix dte1 off): the per-leg mechanic ≈ −₹137/lot on Wed with a −₹17k/lot p05 tail — TB-SENSEX\'s real 10:30-12:00 window carries Wednesday instead. Thursday (DTE0) stays real.' },
      { date: '18 Aug 2026', text: 'Suite resized 3 → 2 lots/system — notional parity with the NIFTY book (₹31L vs ₹31.5L per leg; 3 lots was 148%). Restores the portfolio-stop study sizing.' },
      { date: '18 Aug 2026', text: 'ATM4 rolled-leg SL re-anchored to max(price_x, roll premium) × 1.3 (shared executor with the NIFTY suite — research/113).' },
      { date: '13 Aug 2026', text: 'DTE study: Thursday = hold (no stop), Wednesday = tight combined-20%, DTE2+ keeps per-leg 30%.' },
      { date: '05 Aug 2026', text: 'Scaled to 3 lots; ST(7,3) trail + venue TP added.' },
    ],
    note: 'SENSEX weeklies only exist from 2024 — every SENSEX verdict rests on a calm, low-VIX sample.',
  },
  {
    id: 'csl-timeb-nifty',
    name: 'NIFTY TimeB (TB-CSL)',
    subtitle: 'NFO · time-boxed short straddle · CSL_TIMEB_NIFTY',
    status: 'live',
    size: '8 lots · qty 520 (10 lots from 24-Aug)',
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
      ['Days', 'Tue(DTE0) 09:30-11:00 SL25 + Fri(DTE2) 10:00-12:00 SL20. MONDAY IS BACK LIVE from 25-Aug as a separate book (CSL_TIMEB_NIFTY_MON_AM, 8 lots, 09:16-11:16, Rs1,000/lot rupee stop) - Arun override: r/124 makes it the best Monday cell but it FAILS the shuffle null (p=0.376, n=18). The old 13:00-14:00 Monday cell stays PAPER'],
    ],
    rulesDoc: 'research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py (BOOKS)',
    dashboard: '/straddles',
    studies: [
      { slug: 'csl-best-config-straddles', title: 'CSL best-config straddles — entry × exit × combined-SL per DTE' },
    ],
    changeLog: [
      { date: '25 Aug 2026', text: 'Monday returns live in a NEW book, CSL_TIMEB_NIFTY_MON_AM: 09:16-11:16, 8 lots, Rs1,000/lot rupee stop. The r/124 re-run on measured costs made this the best Monday cell (median +Rs6,920/day @8L, win 88.9%, R:R@p95 1:1.0, stop-invariant across every arm) and the rupee stop caps its worst day best (-15,752 vs -20,464 unstopped, -28,496 at SL20). USER OVERRIDE: the cell still fails the label-shuffle null at p=0.376 on 18 Mondays, so it is not statistically distinguishable from mined noise - Arun chose live after seeing that. The old 13:00-14:00 Monday cell continues on paper. Engine change: the CSL daemon gained rupee-per-lot stop support (rsN), additive - percent and none arms unchanged.' },
      { date: '23 Aug 2026', text: 'Monday dropped from live trading — the window atlas (r/122) condemned it by a third independent route (R:R@p95 1:11.8, modelled P(loss) 52%; r/120 and r/121 agree). Friday was briefly dropped in the same decision, then KEPT on its atlas KEEP verdict (93% win, 1:6.9). The Monday cell continues on paper (CSL_TIMEB_NIFTY_MON, 8L) so the November re-run has live-shaped evidence. Same day: Thursday TIMEB SENSEX 10-lot bump declined — stays 8 lots.' },
      { date: '19 Aug 2026', text: 'Thursday split out into CSL_TIMEB_NIFTY_THU at 3 lots (entry 09:25, exit 15:20, SL20): DTE3 is the grid\'s 2nd-best NIFTY cell (mean ₹16,956, 91%) but full size collides with SENSEX\'s best day at current capital. Main book dark on Thu. 8→10 lots scheduled 24-Aug.' },
      { date: '17 Aug 2026', text: 'First 6-lot REAL window (13:00 entry) observed.' },
      { date: '14 Aug 2026', text: 'Promoted to real money at 6 lots (qty 390).' },
    ],
  },
  {
    id: 'nas-comb20',
    name: 'NIFTY COMB (NAS_COMB20)',
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
      ['Days', 'Mon/Tue/Fri live (DTE1/DTE0/DTE2). NIFTY is OFF on Thursdays from 27-Aug - the DTE3 cell runs on PAPER as NAS_COMB20_THU (5 lots). Wednesday has no cell.'],
    ],
    rulesDoc: 'research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py (BOOKS)',
    dashboard: '/straddles',
    studies: [
      { slug: 'sensex-nifty-stop-by-dte', title: 'Stop calibration by DTE — where combined-SL beats per-leg' },
    ],
    changeLog: [
      { date: '27 Aug 2026', text: 'NIFTY OFF on Thursdays live - paper only. Reverses the 19-20 Aug Option-B merge (commit e9aa381) that folded NIFTY-Thu into this book as a 5-lot DTE3 cell after the separate 3-lot Thursday book was retired. Arun call: Thursday is SENSEX expiry and its best day of the week, and NIFTY at DTE3 was competing for the same margin. NIFTY live lots on Thursday go 5 to 0. The cell continues as NAS_COMB20_THU (paper, 5L, 09:16-15:20 SL20) so the DTE3 evidence that motivated Option B (grid mean ~Rs16,956 at 91%) keeps accumulating and this is revisited on data rather than on one bad session. Ops review 2026-10-30.' },
      { date: '17 Aug 2026', text: 'Account-level book stop bought back COMB\'s CE without COMB knowing — unified per-system ledger pending (TODO 2026-08-24).' },
      { date: '14 Aug 2026', text: 'Armed with real money at 2 lots.' },
    ],
    note: 'No per-book pause exists yet: halting this sleeve currently means killing the CSL daemon.',
  },
  {
    id: 'momentum-3l',
    name: 'Momentum-30',
    subtitle: 'NSE cash · Nifty 200 momentum, hold top 8',
    status: 'live',
    size: '₹7,69,000',
    since: '24 Jul 2026',
    rule:
      'Rank by blended momentum and hold the top 8, only while the weekly index gate holds; Donchian trailing stop per name, monthly rebalance with a top-22 buffer.',
    rules: [
      ['Universe', 'NSE Nifty 200'],
      ['Selection', 'Blended 6m+12m RS vs NIFTYBEES, top 8 (top-22 buffer suppresses churn)'],
      ['Gate', 'Weekly index gate — the whole drawdown story sits here, not in the stop'],
      ['Stop', 'Daily Donchian trailing stop per name'],
      ['Rebalance', 'Month-end, exits run FIRST then the refill; rotate-only, winners are never trimmed'],
      ['Top-ups', 'Immediate equal top-up on deposit (research/112); broker-qty assert before any sell'],
      ['Idle cash', 'Swept into CASHIETF (~5.2%) above a 3% reserve; released before the rebalance buys'],
    ],
    rulesDoc: 'services/momentum_paper.py + research/62 (momentum30-subselect)',
    dashboard: '/momentum-paper',
    studies: [
      { slug: 'nifty250-momentum-video-research75', title: 'Nifty-250 Momentum Top-15 — faithful survivorship-free replication' },
      { slug: 'momentum-universe-bakeoff', title: 'Universe bake-off — Nifty 200 vs 250 vs 51-250 vs Midcap 150 vs 500' },
      { slug: 'momentum-250-leverage-frontier', title: 'Leverage frontier — how far can this book be pushed?' },
      { slug: 'momentum-put-hedge-overlay', title: 'Put-hedge overlay vs the cash-exit gate', verdict: 'SIGNAL' },
    ],
    changeLog: [
      { date: '26 Aug 2026', text: 'Month-end order fixed: the Donchian exit check now runs BEFORE the rebalance refills, and the rebalance refuses to buy any name already below its stop. Month-end is now read off the trading calendar, not just weekends.' },
      { date: '26 Aug 2026', text: 'Live P&L chart now compares against Nifty 50 / 500 / Midcap 150 / Smallcap 250, with deposits removed (time-weighted) — the old card counted deposits as performance and read +155.85%.' },
      { date: '20 Aug 2026', text: 'Daily EOD run made self-healing: it re-runs on boot or at 15:25 if its 15:05 firing was lost, and alerts if it was missed after the close.' },
      { date: '19 Aug 2026', text: 'Size ₹6,00,000 → ₹7,69,000 (deposit, deployed as an immediate equal top-up).' },
      { date: '17 Aug 2026', text: 'Size ₹4,00,000 → ₹6,00,000. Idle cash now swept to CASHIETF, deliberately not LIQUIDCASE (Arun holds 17,276 of those, pledged).' },
      { date: '14 Aug 2026', text: 'Size ₹3,00,000 → ₹4,00,000.' },
      { date: '17 Aug 2026', text: 'Moved under Holdings in the sidebar — it is real money, not a paper book.' },
      { date: '16 Aug 2026', text: 'Top-up ledger corruption fixed and deployed (INSERT OR REPLACE wiped qty/cost/entry_date).' },
      { date: '24 Jul 2026', text: 'Funded with real money at ₹3L.' },
    ],
  },

  // ----------------------------------------------------------------- PAPER
  {
    id: 'csl-timeb-sensex',
    name: 'TIMEB SENSEX',
    subtitle: 'BFO · time-boxed short straddle · CSL_TIMEB_SENSEX',
    status: 'live',
    size: '8 lots · qty 160',
    since: '19 Aug 2026',
    rule: 'Time-boxed SENSEX straddle windows, REAL from 19-Aug at 8 lots (notional parity with NIFTY TB@8L). The Wed 10:30-12:00 window (+₹1,612/day at study size) is the venue\'s only earning construction — it replaces the suite\'s real Wednesday.',
    rules: [
      ['Universe', 'SENSEX weekly ATM straddle'],
      ['Entry / stop / exit', 'Per-DTE frozen config, same lab as the NIFTY twin (Wed DTE1: 10:30-12:00 SL20)'],
      ['Margin gate', 'Marketable-LIMIT + 1.3x headroom check at entry; paper-fallback if short'],
      ['Days', 'Wed(DTE1) + Thu(DTE0) ONLY — its Mon/Tue/Fri cells are the grid\'s weakest (₹1,080–3,564) and collide with NIFTY margin'],
      ['Thursday stops', 'NO per-leg stop on expiry day (research/114); book stop −₹3,000/lot, take-profit ₹4,000/lot'],
      ['Wednesday', 'LIVE from 26-Aug (user, 20-Aug): per-leg 30% kept, book stop −₹1,300/lot, take-profit ₹1,667/lot'],
    ],
    rulesDoc: 'research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py (BOOKS)',
    dashboard: '/straddles',
    studies: [
      { slug: 'csl-best-config-straddles', title: 'CSL best-config straddles — entry × exit × combined-SL per DTE' },
    ],
    changeLog: [
      { date: '20 Aug 2026', text: 'Wednesday (DTE1) switched to LIVE on the user\'s instruction. research/118 supports it: Wednesday is the calmest weekday over 1,354 days (1 catastrophic day in 125, the fewest of any) and the −₹16,502 day that had scared us off sits inside a 46-day bucket earning +105.7 pts/day at 80% win. Per-leg 30% retained as status quo pending its own study.' },
      { date: '20 Aug 2026', text: 'Thursday rules rebuilt on research/114: per-leg 30% stop OFF on expiry day (it turned +2,630/lot/day at 92% win into −227 at 25%), take-profit widened ₹1,667→₹4,000/lot (the old one kept only 57% of the edge), book stop widened −₹1,300→−₹3,000/lot on DTE0. The 50% disaster backstop stays — 12 benign Thursdays cannot price the tail.' },
      { date: '19 Aug 2026', text: 'Live days restricted to Wed+Thu (weak Mon/Tue/Fri cells removed — data-driven venue split). 8→10 lots scheduled 24-Aug.' },
      { date: '18 Aug 2026', text: 'REAL from 19-Aug at 8 lots (6L→8L, notional parity with NIFTY TB@8L). Deploy doc: docs/CSL_TIMEB_SENSEX_LIVE_DEPLOY_STATUS.md.' },
      { date: '14 Aug 2026', text: 'Sized to 6 lots on paper alongside the NIFTY promotion.' },
    ],
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

  {
    id: 'n500m',
    name: 'N500M',
    subtitle: 'NSE cash intraday · Nifty 500 momentum · per-stock CCRB + vol-breakout',
    status: 'paper',
    size: '27 stocks · 30 per-stock rules',
    since: '08 May 2026',
    rule:
      'Per-stock rules only: a symbol trades a signal (vol-breakout / CCRB) on the timeframe and exit policy its own backtest promoted. Entry in the 09:15-09:25 window, ATR-based SL, square off at EOD.',
    rules: [
      ['Universe', 'Nifty 500, 27 symbols that cleared per-stock promotion'],
      ['Signal', 'Volume breakout (volbo) or CCRB, per symbol — 5 / 10 / 15-min as promoted'],
      ['Entry', 'Signal window 09:15-09:25 after a pre-market setup check'],
      ['Stop', 'ATR-based SL per rule; target / trail per the promoted exit policy'],
      ['Exit policies', 'T_NO (EOD only) · T_R_TARGET_1.0R · T_STEP_TRAIL'],
      ['Jobs', 'precompute + data refresh + scan + monitor, Mon-Fri'],
      ['Record', '31 closed trades over 25 sessions (08 May → 17 Aug 2026), 58% win, +₹13,852'],
    ],
    rulesDoc: 'services/n500m_configs.py (per-stock promoted rules) + services/n500m_scanner.py',
    dashboard: '/n500m',
    studies: [],
    studyGap: 'Per-stock promotion came from the intraday sweeps (research/109 + 110), which concluded NO EDGE for the family as a whole — this book is the surviving per-stock subset and has no published study of its own.',
    changeLog: [
      { date: '17 Aug 2026', text: 'Confirmed running: 1 trade today (COCHINSHIP short, +₹681 EOD). Moved into Paper Books in the sidebar.' },
      { date: '08 May 2026', text: 'Paper book started; PAPER mode set 07 May.' },
    ],
    note: 'The page shows today only — the 25-session history lives in n500m_positions and is not surfaced yet.',
  },
  {
    id: 'i75wr',
    name: 'I75WR',
    subtitle: 'NSE cash intraday · 3 configs (Diamond Short, Long-TC, Long-MR)',
    status: 'paper',
    size: '₹3,00,000 per config · ₹3,000 risk/trade',
    since: '17 Aug 2026',
    rule:
      'Three parallel configs over the same three systems, differing only in cost assumptions and TP/SL: A original (TP 0.5 / SL 1.5), B cost-resilient (TP 2.0 / SL 1.5), C continuous scan.',
    rules: [
      ['Universe', 'NSE cash intraday'],
      ['Systems', 'Diamond Short (09:45 single scan) · Long-TC (09:15-10:30, 5-min) · Long-MR (11:15-13:15, 5-min)'],
      ['Config A / B', 'Same systems, TP 0.5 / SL 1.5 vs TP 2.0 / SL 1.5 — the cost-resilience test'],
      ['Config C', 'Continuous scan 09:30-15:00'],
      ['Risk', '₹3,000 per trade · ₹9,000 daily loss limit · max 5 concurrent across configs'],
      ['Exit', 'TP/SL audit at 15:15, EOD square-off 15:25'],
    ],
    rulesDoc: 'services/intraday_75wr/config_{a,b,c}.py',
    dashboard: '/intraday75wr',
    studies: [],
    studyGap: 'No published study — the three configs are themselves the experiment.',
    changeLog: [
      { date: '17 Aug 2026', text: 'All three configs switched from off to PAPER (persisted in intraday_75wr_mode_overrides.json). No trades recorded before this.' },
    ],
    note: 'Jobs were registered all along but every config sat at mode=off, so the book has an empty history to date.',
  },
  {
    id: 'pairs',
    name: 'Pairs',
    subtitle: 'F&O futures · 6-pair cohort · z-score mean reversion',
    status: 'paper',
    size: '₹10,00,000 · ₹6,000 risk per pair',
    since: '17 Aug 2026',
    rule:
      'Trade the spread of a cohort pair when its z-score stretches, sized ₹6,000 risk per pair (₹3,000 a leg), max 5 of 6 pairs open. Alpha/beta re-fit on a rolling 252 days.',
    rules: [
      ['Universe', '6-pair cohort from F&O futures, refreshed quarterly'],
      ['Signal', 'Spread z-score against a rolling 252-day alpha/beta fit'],
      ['Sizing', '₹6,000 risk per pair — ₹3,000 per leg'],
      ['Concurrency', 'Max 5 of 6 pairs open'],
      ['Costs', '0.03% per side per leg modelled; 0.10% stress case reported in paper mode'],
      ['Job', 'Daily scan 16:00 IST'],
    ],
    rulesDoc: 'config.py PAIR_TRADING_DEFAULTS + services/pair_trading/',
    dashboard: '/pair-trading',
    studies: [],
    studyGap: 'No published study yet.',
    changeLog: [
      { date: '17 Aug 2026', text: 'Switched from off to PAPER; PAIR_TRADING_DEFAULTS.enabled flipped to True so the change survives a restart (toggle-mode only patches the running process).' },
    ],
  },

  {
    id: 'stock-wings',
    name: 'Stock Winged Strangle',
    subtitle: 'F&O stocks \u2014 45 \u2192 21 DTE \u00b12.5% strangle + 7% wings, one ruleset',
    status: 'paper',
    size: '\u20b920L / 10 slots (paper)',
    since: '25 Aug 2026',
    rule: 'At 45 DTE on the monthly stock expiry sell the \u00b12.5% strangle and buy wings ~7% away; no stop; close at 50% of credit or 21 DTE. Liquidity (all 4 legs traded, shorts \u2265100, wings \u226510) is the only stock filter; slots ranked by option volume.',
    rules: [
      ['Universe', 'Every F&O stock \u2014 one universal ruleset, zero per-stock tuning; the liquidity gate does the selecting'],
      ['Entry', 'Monthly expiry \u2212 45 calendar days at EOD close: sell CE @ spot+2.5% / PE @ spot\u22122.5% (nearest traded strikes)'],
      ['Wings', 'Buy CE/PE ~7% of spot beyond the shorts \u2014 crash cap for stock-specific overnight gaps; wing width chosen by sweep (wider-monotone)'],
      ['Exits', '50% of net credit, or time exit at 21 DTE. NO premium stop \u2014 every stop tested hurts; the wings are the risk cap'],
      ['Sizing', '10 slots \u00d7 \u20b92L; lots to ~\u20b920L notional/slot on a 10%-of-notional margin ESTIMATE \u2014 real SPAN check owed'],
      ['Cadence', 'REAL quotes: 5s live marks; day mark-of-record = the real 15:29:30 close snapshot; exits evaluate on it once daily at the close (the tested cadence). New cycles enter LIVE at ~15:26 on real quotes. Bhavcopy = backfill only'],
      ['Why paper', 'Study is STRATEGY-candidate (G3 passed); gates to real money: real basket margin, live cost/slippage evidence, earnings-skip test'],
    ],
    rulesDoc: 'research/127_stock_neutral_wings/STOCK_NEUTRAL_WINGED_STRADDLE_DAILY_SWEEP_STATUS.md',
    dashboard: '/stock-wings',
    studies: [
      {
        slug: 'stock-45dte-neutral-wings',
        title: 'Stock 45\u219221 DTE winged strangle \u2014 one ruleset across the F&O universe',
        verdict: 'STRATEGY-CANDIDATE',
      },
    ],
    changeLog: [
      { date: '26 Aug 2026', text: 'Switched to REAL options data end-to-end (Arun): 5s live ticks, day mark-of-record = real 15:29:30 snapshot, daily exits evaluated on it, LIVE entries at ~15:26 on real quotes + today-volume gate; real Kite basket margins persisted. Bhavcopy demoted to backfill.' },
      { date: '25 Aug 2026', text: 'PAPER book live: services/stock_wings_paper.py (cron 16:50 + 20:30) seeded from 01-Jun \u2014 18 replayed closes, 10 open Sep-29 positions; page at /app/stock-wings. Study published same day: net +0.264%/trade t 5.06, portfolio 20\u201326% CAGR at stressed margin, corr NIFTY \u22120.09.' },
    ],
  },

  // ---------------------------------------------------------------- PARKED
  {
    id: 'straddle45',
    name: '45-DTE Straddle',
    subtitle: 'NIFTY monthly \u2014 short ATM straddle, 45 \u2192 21 DTE',
    status: 'paper',
    size: '3 lots (195 qty)',
    since: '24 Aug 2026',
    rule: 'Sell the ATM straddle 45 calendar days before the NIFTY monthly expiry; close at 21 DTE, or earlier on a 50% target / 200% stop. No executor \u2014 study and sizing only.',
    rules: [
      ['Instrument', 'NIFTY monthly expiry \u2014 the last expiry of the month already listed 45 days out (from 2025 a weekly can expire AFTER the monthly)'],
      ['Entry', 'Expiry \u2212 45 calendar days at the close; sell 1\u00d7 ATM CE + 1\u00d7 ATM PE, both legs must have traded that day'],
      ['Target', 'Combined premium \u2264 50% of entry credit \u2014 fires once in 89 trades'],
      ['Stop', 'Combined premium \u2265 200% of entry credit \u2014 fires 2\u20133 times in 89 trades'],
      ['Time exit', 'Expiry \u2212 21 calendar days \u2014 how 85 of 89 trades end, and the whole design'],
      ['Monitoring', 'Hourly candle closes; nothing below 60-min changes a single trade'],
      ['Do NOT delta-manage', 'Every move-threshold exit and re-centring scheme tested is worse than holding \u2014 cycles cut on a move realise \u221228.6 pts, cycles left alone earn +83.0'],
      ['Sizing', 'Capital = 3%-adverse-move margin (\u20b92.69L/lot) + 2\u00d7 MaxDD. At 3 lots that is \u20b912.0L'],
      ['Why parked', 'Passed G3 robustness; owes a stress-margin test before real money \u2014 SPAN inflates in the same event that drives the drawdown'],
    ],
    rulesDoc: 'research/119_45dte_short_straddle/NIFTY_45DTE_SHORT_STRADDLE_MULTITF_BACKTEST_STATUS.md',
    dashboard: '/straddle45',
    studies: [
      {
        slug: 'nifty-45dte-short-straddle',
        title: '45-DTE NIFTY short straddle \u2014 replication, monitoring frequency, VIX filter, delta management',
        verdict: 'STRATEGY-CANDIDATE',
      },
    ],
    changeLog: [
      { date: '24 Aug 2026', text: 'Registered. Control-room page added at /app/straddle45 with lot config (default 3), payoff, per-DTE margin and the study links. Nothing armed \u2014 no executor exists.' },
      { date: '23 Aug 2026', text: 'Delta management REFUTED (Phase E): 7 move thresholds \u00d7 3 arms \u00d7 3 re-entry caps \u00d7 close/intraday triggers, not one cell beats holding. To cut risk, cut lots.' },
      { date: '20 Aug 2026', text: 'Published study: the video table replicates on real bhavcopy (+78.1 pts/trade, t 3.03, 89 campaigns). Two own-errors corrected \u2014 notional sizing and NIFTY lot 75\u219265.' },
    ],
    note: 'NOT ARMED and not funded. Margin measured live from Kite on 24 Aug 2026: \u20b92.24L/lot at 45 DTE, rising to \u20b92.69L once spot is 3% away \u2014 and a \u22653% move happens in 66% of campaigns, so the book is sized on the stressed number, not the entry number.',
  },
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
      ['Why parked', 'Not currently being pursued; enabled=False today, but the six scheduled jobs still run'],
      ['Never real money', 'Every KC6 order ever placed is status PAPER / PAPER_TARGET with kite_order_id NULL'],
    ],
    rulesDoc: 'docs/KC6-SESSION-HANDOFF.md',
    dashboard: '/kc6',
    studies: [],
    studyGap: '20-year backtest exists (2,482 trades, PF 1.70) but is not published to /app/backtest.',
    changeLog: [
      { date: '19 Aug 2026', text: 'Audit: it was NOT dormant — 5 paper cycles between 23 Jul and 17 Aug (net −₹4,202, 67% win, last exit 17 Aug), all simulated. config.enabled is False now with no open positions.' },
      { date: '—', text: 'Parked per project instructions; migration to /app/kc6 also parked.' },
    ],
    note: 'The journal tags KC6 rows mode=LIVE (hardcoded in services/journal/sources/kc6_source.py:66), so ₹5,528 of PAPER P&L currently sits in the live ledger — fix with the journal live-only work.',
  },
  {
    id: 'mst',
    name: 'MST',
    subtitle: 'NIFTY options · debit-spread structure engine with pyramiding',
    status: 'paper',
    size: '1 lot per leg (65)',
    since: '18 Aug 2026',
    rule:
      'Stochastic-triggered debit spread on NIFTY with pyramiding to level 2. Back on PAPER after a 3-month stop; LIVE stays barred until the May-2026 incident causes are fixed.',
    rules: [
      ['Universe', 'NIFTY options, spread width 200, debit OTM offset 50'],
      ['Trigger', 'Stochastic (14, 3) with 80 / 20 bands on 30-min bars, ATR(21) sizing'],
      ['Pyramid', 'Up to level 2; exits on overbought/oversold thresholds 70 / 30'],
      ['Guards', 'Min 6 DTE at entry · min credit ₹1,000/lot · abort on leg rejection'],
      ['Live barred', 'Paper only until the 2026-05-15 causes are closed: tick-pipeline freeze, spurious credit_too_low=0/lot rolls, rejected real-leg closes'],
      ['Watch', 'Any 0.00-priced leg or repeated credit_too_low roll in mst_events means the freeze is back — that is what stopped it in May'],
    ],
    rulesDoc: 'services/mst_engine.py + config.py MST_DEFAULTS',
    dashboard: '/mst',
    studies: [],
    studyGap: 'No published study.',
    changeLog: [
      { date: '18 Aug 2026', text: 'PAPER mode activates at the 09:00 pre-open restart — engine boots FLAT for the first time since May.' },
      { date: '17 Aug 2026', text: 'Stale legs cleared so it can run: 2 legs on the expired 19 May weekly (both REAL, paper_mode=0) + 4 legs priced 0.00 from the frozen-tick credit_too_low rolls. Rows kept, marked CLOSED, P&L left NULL. config.py enabled -> True, live still barred.' },
      { date: '15 May 2026', text: 'Disabled after the incident: tick pipeline froze 07 May 14:45, spurious credit_too_low rolls, rejected real-leg closes.' },
      { date: '07 May 2026', text: '10 positions recorded on one day, then nothing.' },
    ],
    note: 'The two real legs from 07 May (BUY 24450 CE @266.80 / SELL 24650 CE @173.65, 65 qty each) expired 19 May with no exit ever recorded — their real-money outcome is in the broker statement, not in this app.',
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
  { name: 'EOD', to: '/eod-breakout', what: 'EOD breakout scan' },
  { name: 'Opt Study', to: '/options-study', what: 'Options decay / CPR / candle aggregates' },
  { name: 'Backtest', to: '/backtest', what: 'Published studies' },
];
