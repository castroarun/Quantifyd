"""OPS & REVIEW CENTER registry -> static/app/straddles/ops_center.json
THE standing registry of (a) every lab job with schedule + manual trigger, and
(b) every periodic re-assessment / review item with its due date — rendered as the
Operations & Review Center section on /app/straddles (#ops-center).

BINDING CONVENTION (also in .claude/CLAUDE.md): any new lab, analyzer job, or
periodic-review obligation created in ANY session MUST be registered here (and in
docs/LABS_AND_JOBS_REFERENCE.md). Reviews carry due dates so the page shows DUE badges.
Runs daily in the 15:40 regen (cheap) so due-flags stay current."""
import json
from datetime import date, datetime
from pathlib import Path

Q = Path("/home/arun/quantifyd")
OUTS = [Q / "static/app/straddles/ops_center.json", Q / "frontend/public/straddles/ops_center.json"]

GROUPS = [
    ("45-DTE NIFTY straddle LIVE book (research/119)", [
    ("straddle45 LIVE executor (research/119)", "*/2 15:18-15:30 Mon-Fri cron (flock)",
     "REAL MONEY. 45-DTE NIFTY ATM short straddle, 3 lots NRML, VIX-rank>25 entry filter, "
     "TP 50%/SL 200%/21-DTE time exit, decided ONCE in the 15:20-15:29 window to match the "
     "backtested close-basis rule. Both-legs-or-neither, reconciles against kite.positions() "
     "before acting, verifies every fill. Unarmed unless STRADDLE45_LIVE=1. "
     "KILL: touch backtest_data/straddle45_KILL",
     "cd /home/arun/quantifyd && ./venv/bin/python3 services/straddle45_live.py status"),
    ]),
    ("CSL-60 DTE-0 straddle PAPER book (research/136)", [
        ("csl60_paper mark", "every minute 09:00-15:59 Mon-Fri cron (flock; acts only on NIFTY expiry days)",
         "Paper-trades the AlgoTest study's rank-1 system: expiry-day 09:16 ATM straddle, per-leg 60% SL, "
         "trail-to-breakeven, 15:15 exit, 10 lots. Reads recorded 1-min option_chain only; no orders, no engine. "
         "Renders in the NAS Trade Book (NIFTY CSL60 - DTE-0). Deploy doc: "
         "research/136_nifty_csl_portfolio/CSL60_DTE0_PAPER_1MIN_DEPLOY_STATUS.md",
         "cd /home/arun/quantifyd && ./venv/bin/python3 services/csl60_paper.py mark"),
    ]),
    ("Options data capture (feeds every options study)", [
        ("option 1-min OHLC recorder", "15:35 Mon-Fri cron (flock)",
         "Captures 1-MINUTE OHLC (high/low, not just an LTP poll) for NIFTY/BANKNIFTY/SENSEX nearest-2-expiry contracts, ~540/day. MUST run daily: Kite refuses historical data for EXPIRED tokens ('invalid token'), so a missed day is lost forever. Unlocks stop-trigger verification and per-leg MAE/MFE, which the LTP-poll option_chain cannot provide. Read-only vs broker; no engine touched. Deploy doc: OPTIONS_OHLC_RECORDER_1MIN_DEPLOY_STATUS.md",
         "cd /home/arun/quantifyd && set -a && . ./.env && set +a && ./venv/bin/python3 scripts/record_option_1min_ohlc.py"),
    ]),
    ("Stock winged strangle PAPER book (research/127)", [
        ("bhav stock daily download", "16:20 Mon-Fri cron (flock)",
         "extends nse_options_bhav with the day's F&O STOCK bhavcopy (idempotent, resumes by date); feeds the paper book and any stock-options research",
         "cd /home/arun/quantifyd && ./venv/bin/python3 research/89_short_monthly_straddle/scripts/download_nse_bhav_stocks.py"),
        ("stock_wings_paper seed+mark", "16:50 + 20:30 Mon-Fri cron (flock)",
         "45->21 DTE +/-2.5% strangle + 7% wings on F&O stocks, Rs20L/10 slots PAPER: opens new 45-DTE cycles, marks/exits from bhav closes, publishes /app/stock_wings_paper.json for /app/stock-wings",
         "cd /home/arun/quantifyd && ./venv/bin/python3 services/stock_wings_paper.py seed"),
        ("stock_wings_paper live marks", "*/5 09:00-15:55 Mon-Fri cron (flock)",
         "DISPLAY ONLY: re-prices the four legs of every open position from live Kite quotes and pulls the REAL basket margin (wings sent first), so /app/stock-wings ticks during the session. Evaluates NO exit and writes NO position row - the target and stop still resolve on the EOD bhav close, as backtested",
         "cd /home/arun/quantifyd && set -a && . ./.env && set +a && ./venv/bin/python3 services/stock_wings_paper.py live"),
    ]),
    ("Expiry-Afternoon Lab (research/125)", [
        ("expiry_lab_assessment", "Tue+Thu 16:05 IST cron",
         "re-runs the DTE0 afternoon sweep, re-scores frozen slots (TimeB2 live Tue 13:15-14:30 CSL30 8L), flags DRIFT/WEAK, appends permanent run history to /app/straddles#expiry-lab",
         "venv/bin/python3 research/125_expiry_afternoon_straddle/scripts/expiry_lab_assessment.py"),
        ("TimeB2 standing book (CSL_TIMEB2_LIVE)", "daemon 09:12 cron, Tuesdays (NIFTY DTE0) only",
         "NIFTY Tue 13:15->14:30 combined-SL30 8L REAL - formalized 25-Aug post-close; trades land in the CSL day records; one-shot runner retired (25-Aug precedent -2,990)",
         "grep CSL_TIMEB2_LIVE /tmp/csl_paper.log"),
    ]),

    ("Daily auto-analysis (15:42 regen chain)", [
        ("Whole regen chain", "15:42 Mon-Fri (moved outside the 15:40 F&O window, user 2026-08-16)", "V1/V2 cards + leaderboard + SL30 + backfill + baseline + portfolio lab (~80 min)",
         "./research/58_intraday_recenter_straddle/scripts/regen_straddles.sh"),
        ("strategy_rankings", "in chain", "Strategy Leaderboard (grades, Corr-book, Period)",
         "PYTHONPATH=. venv/bin/python3 research/58_intraday_recenter_straddle/scripts/strategy_rankings.py"),
        ("csl_paper_backfill", "in chain (~75 min)", "BACKTEST day-curves for paper books; live records always win; avoid 09:00-15:40",
         "setsid nohup venv/bin/python3 research/111_sensex_manual_mgmt/scripts/csl_paper_backfill.py > /tmp/csl_backfill.log 2>&1 &"),
        ("nas_baseline", "in chain", "NAS suite day P&L with REAL/PAPER per-day tags",
         "venv/bin/python3 research/111_sensex_manual_mgmt/scripts/nas_baseline.py"),
        ("portfolio_lab", "in chain", "Options Portfolio Lab: stack rows, corr matrix, curves, source mix",
         "venv/bin/python3 research/111_sensex_manual_mgmt/scripts/portfolio_lab.py"),
    ]),
    ("Weekly re-analysis & re-assessment", [
        ("entry_exit_sweep", "Fri 15:45", "TB-CSL Best-Config Lab regen (informational; frozen config never auto-moves)",
         "setsid nohup venv/bin/python3 -u research/111_sensex_manual_mgmt/scripts/entry_exit_sweep.py > /tmp/eesweep.log 2>&1 &"),
        ("stack_reassessment", "Fri 16:35", "SYSTEM re-assessment: corr-drift, per-DTE shifts, TB windows vs sweep, sizing revalidation, live-vs-model (panel in Portfolio Lab)",
         "venv/bin/python3 research/111_sensex_manual_mgmt/scripts/stack_reassessment.py"),
    ]),
    ("45-DTE straddle PAPER book (research/119)", [
        ("straddle45_paper mark", "*/5 09:00-15:59 Mon-Fri", "Marks the open 45-DTE straddle from the broker's live LTP (falls back to the 1-min recorder, then last EOD close) and republishes /app/straddle45_paper.json",
         "set -a; . .env; set +a; venv/bin/python3 services/straddle45_paper.py mark"),
        ("straddle45_paper seed", "16:20 Mon-Fri", "Opens a new campaign when an entry date (expiry-45d) is reached; closes any campaign that hit target/stop/21-DTE",
         "set -a; . .env; set +a; venv/bin/python3 services/straddle45_paper.py seed"),
        ("margin_recorder", "16:05 Mon-Fri", "Records REAL margin daily: account SPAN/exposure, a reference 1-lot ATM straddle on the front monthly, and the book's own open position, against NIFTY spot + India VIX and its 252-session rank. Replaces the ABANDONED historical SPAN reconstruction - builds true margin-vs-vol history from 2026-08-25 forward",
         "set -a; . .env; set +a; venv/bin/python3 services/margin_recorder.py"),
        ("download_nse_bhav (daily)", "16:10 Mon-Fri", "NSE F&O bhavcopy for the last 5 sessions. NEW - it had NO cron, which is why bhav sat stale from 2026-07-21 to 2026-08-24 and every EOD-priced book silently aged",
         "venv/bin/python3 download_nse_bhav.py --start $(date -d '5 days ago' +%Y-%m-%d) --end $(date +%Y-%m-%d)"),
    ]),
    ("Intraday execution & monitoring (market hours)", [
        ("csl_paper_exec", "09:12 cron", "The 7 CSL books (NAS_COMB20 + CSL_TIMEB_NIFTY REAL). Safe dry-run only while cron copy runs",
         "venv/bin/python3 research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py --probe"),
        ("nas_live_guardian", "*/5 min", "Hunts live failure classes; ALL-CLEAR/FAIL verdicts", "tail -20 /tmp/nas_guardian.log"),
        ("nas_alert_feed + laptop watcher", "*/1 min + 30s poll", "Desktop popups for every book (REAL/PAPER tagged)", "tail /tmp/nas_alert_feed.log"),
        ("integrity watchdog / fail-rejected / MTM dump", "*/5, */2, */1 min", "pipeline freeze email, rejected-order sweep, intraday MTM snapshots", ""),
        ("SL + portfolio-stop monitors", "in-app 10s", "per-leg SLs, rupee stop, -1300/lot venue stop + trail/TP",
         "journalctl -u quantifyd --since '10 min ago' | grep -i monitor"),
    ]),
    ("EOD analyzers", [
        ("nas_analyzer (RAG report)", "15:45", "daily book verdict -> /app/reports", "venv/bin/python3 scripts/nas_analyzer.py"),
        ("options_outlier_scan", "15:47", "outlier/drift scan -> /app/reports", "venv/bin/python3 scripts/options_outlier_scan.py"),
        ("EOD snapshots + GitHub backup", "15:42 / 16:00", "state snapshots; repo backup", ""),
    ]),
    ("Manual-only study scripts (auto-include new live days)", [
        ("comb_tb_overweight_grid", "on demand", "sec-18b sizing grid: is the deployed cell still right?",
         "venv/bin/python3 research/111_sensex_manual_mgmt/scripts/comb_tb_overweight_grid.py"),
        ("per_dte_elimination_check", "on demand", "per-DTE re-ranking: eliminate weak DTE before replacing a system",
         "venv/bin/python3 research/111_sensex_manual_mgmt/scripts/per_dte_elimination_check.py"),
        ("nas_suite_csl_replay / csl_mgmt_replay / sleeve_pstop_test", "on demand", "suite-vs-CSL arms, TRAIL/SHIFT arms, portfolio-overlay test (~2 min each)", ""),
    ]),
    ("Sleeves dividend engine (True North + Open Alpha)", [
        ("dividend_declare", "19:15 Mon-Fri cron (idempotent)",
         "Quarterly HWM dividend declarations for both sleeves (adopted policy research/142 "
         "dividend_sim_v2 E: 25% of new profit above the flow-adjusted HWM, payout capped at "
         "last dividend +7.5%/qtr, surplus to a 6%-p.a. equalization reserve that bridges dry "
         "quarters; capital never invaded, positions never force-sold). Acts only within 12 days "
         "after a calendar quarter end and never re-declares a quarter, so the daily run is a "
         "no-op most nights. Fires the intimation email/desktop alert with the Console "
         "withdrawal amount. State: 'dividend' in bluesky_paper_state.json / mp_state. "
         "UI: /app/sleeves Dividends card.",
         "cd /home/arun/quantifyd && venv/bin/python scripts/dividend_declare.py"),
    ]),
    ("Momentum Portfolio report (/app/mpf-report)", [
        ("mpf_report_build", "on demand / after any mpf system change",
         "Regenerates EVERY number and all ten charts on /app/mpf-report - the single report "
         "page for True North, Open Alpha Base Age, IPO Base and Quality Summit. Reads the "
         "after-tax curve files (r/159 full_period_after_tax.csv and all_systems_after_tax.csv, "
         "r/160 F_Bb7_equity.csv, r/159 after_tax_tables.csv) and writes static/app/mpf_report.json "
         "+ frontend/public/mpf-report-*.png. Read-only over research results; touches no DB, no "
         "engine, no live state. RUN IT whenever a book's rules, size or status changes, or when "
         "any of those curve files is re-run - otherwise the page shows the previous evidence. "
         "A frontend rebuild is needed after it for the PNGs to reach static/app. Build doc: "
         "research/160_quality_growth_near_ath/MPF_REPORT_PAGE_BUILD_STATUS.md",
         "cd /home/arun/quantifyd && venv/bin/python3 research/_utilities/mpf_report_build.py && "
         "export PATH=$HOME/.nvm/versions/node/v20.20.2/bin:$PATH && cd frontend && npm run build"),
    ]),
    ("Kill / pause levers", [
        ("Freeze flag", "instant", "blocks ALL order placement (suite + sleeves)", "touch backtest_data/nas_manual_freeze.flag"),
        ("Master mode", "instant", "whole stack to paper", "echo '{\"mode\": \"paper\"}' > backtest_data/nas_master_mode.json"),
        ("Suite kill-switch", "instant", "suite to paper via API", "curl -X POST http://127.0.0.1:5000/api/nas/kill-switch"),
        ("Per-book live flag", "next morning", "remove \"mode\": \"live\" from the book in csl_paper_exec.py BOOKS", ""),
    ]),
]

# Periodic reviews / re-assessments — THE calendar. status: PENDING | SCHEDULED | PARKED
REVIEWS = [
    ('Capital Desk ONE-OFF REBALANCE 14-Sep-2026 to TN 37.5 / OA 37.5 / IPO 25 - did it run, and did IPO pick up its cash at 18:45?', '2026-09-14', 'PENDING', 'Approved by Arun 13-Sep-2026 ("fix IPO, move all Monday"). A detached one-shot, scripts/deferred_rebalance_20260914.sh, wakes 09:45 IST 14-Sep, re-checks the date and a 09:40-14:30 window, and runs scripts/rebalance_mpf_20260914.py --execute. Planned on 13-Sep values: True North -Rs 2,53,615 (Rs 27,578 idle cash + ~Rs 2.26L CASHIETF redeemed), Open Alpha +Rs 41,265, IPO +Rs 2,12,350; recomputed on live values at run time and refused if more than 15% away. It overrides, for this transfer only, the 05-Sep rule that True North is never sold down to rebalance. CHECK: (1) logs/rebalance_mpf_20260914.json status "done" and the .done marker present; (2) True North capital fell by the transfer and its CASHIETF units match the broker; (3) Open Alpha capital and cash rose by its share; (4) after the 18:45 IPO run, backtest_data/ipo_paper_state.json capital equals allocation_targets.json ipo_funded and its cash rose by the IPO share (the FUNDING SYNC fix deployed 13-Sep - before it, a deposit into an already-live IPO book never reached its buying cash); (5) the Capital Desk drift panel reads 37.5 / 37.5 / 25. If the log says "stopped", its reason line names the state it left. Revert of the IPO fix: /tmp/ipo_paper.pre_fundsync.py.'),
    ("OA . Base Age - the unconditional -10% hard stop has now surfaced THREE times as "
     "the cheapest drawdown lever. Does it deserve its own study with its own bar?",
     "2027-03-13", "PENDING",
     "RAISED 13-Sep-2026 by research/171. A plain -10% stop, with NO rotation machinery "
     "at all, returns 20.79% after tax at -30.46% drawdown and Calmar 0.683, against the "
     "incumbent Base Age book 20.91% / -35.16% / 0.606 (30 seeds 7001-7030, 25 bps, 5.2% "
     "idle cash, 2005-2026). It beats the incumbent on Calmar on 30 of 30 paired seeds "
     "and is excluded ONLY by the pre-registered CAGR-eligibility clause, which it misses "
     "by 0.12pp. research/166 found it first (caveat 4, Calmar 0.677), research/171 found "
     "it twice more independently (0.683 as a hard stop, 0.692 routed through the "
     "rotation path), and it supplies two thirds of the risk-adjusted gain of the best "
     "cell research/171 could build. WHAT TO DECIDE on the due date, alongside the "
     "OA-ROT-1 live review: whether Arun wants a lower-drawdown Base Age at all. If yes, "
     "it needs its own STATUS doc and its own bar (a drawdown bar, not a Calmar bar - it "
     "is insurance with a premium, not an edge) rather than being rediscovered a fourth "
     "time as a by-product. Evidence: "
     "research/171_baseage_multiswap_topup/results/RESULTS.md, the decomposition table."),
    ('research/169 follow-up - IPO Base at the validated 25-bar floor: are the entries 25-60 sessions after listing filling at the pivot, inside capacity?', '2026-10-15', 'PENDING', "DECIDED 13-Sep-2026: MIN_BARS 60 -> 25 and Capital Desk targets 40/40/20 -> 37.5/37.5/25, in one change (research/167_ipo_base_honest_reopt/IPO_BASE_MIN_BARS_25_DAILY_DEPLOY_STATUS.md). The 60 rested on reading research/153's row-count-today filter as a bars-at-signal rule; research/167's own engine returns 21.80% at 25 bars and 11.57% at 60, and inside the 37.5/37.5/25 book the 60-bar version costs 2.46pp on 30 of 30 paths. WHAT TO CHECK on the due date: (1) every IPO entry since 13-Sep - its age in sessions at the signal, its fill against the pivot, and its size as a share of the name's 20-day median traded value; research/167's 90th percentile is 9.05% at Rs 10L (the 1.56% first quoted was the median); (2) whether any entry fell in the 25-40 session band, where the edge is strongest, or 40-60, where research/169 found it loses to random (27 of 30 at a 40-bar floor); (3) that the market gate blocked entries while NIFTYBEES sat below its 150-day average. Aligns with the existing 15-Oct-2026 soak review. Evidence: research/169_ipo_rules_universe_transplant/results/RESULTS.md Q5-Q6."),
    ("OA \u00b7 Base Age - rotation rule OA-ROT-1 ADOPTED: re-test it on the LIVE entry queue "
     "after six months of real-money operation",
     "2027-03-13", "PENDING",
     "ADOPTED into the staged conversion 13-Sep-2026; review the first live swaps. "
     "Arun read research/170 Part B and adopted the rule the same morning, overriding that study's own non-adoption verdict; research/165 built it into the staged live conversion behind its own OFF switch OA_ROT1 in services/oa_real.py. OA_RULESET is still 'legacy', so nothing is running yet. Replication gate PASS: 8,488 of 8,488 rotation decisions across six engine paths identical to research/170's own engine on fire/no-fire, on the name sold and on the name bought; the gate also caught a real defect first - rs252 must be read on the study's NIFTYBEES master calendar, not on 252 of a symbol's own bars. Rule text, gate and runbook: research/165_oa_baseage_live_conversion/OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md. "
     "THE ORIGINAL FINDING, unchanged: "
     "research/170 (13-Sep-2026) CONFIRMED research/166's post-hoc pick on 30 seeds it had "
     "never been run on: when the Base Age book is full and a qualifying signal arrives, "
     "selling the holding more than 10% under water and giving its slot to the refused "
     "entrant with the HIGHEST 252-day relative strength returns 22.58% after tax at a "
     "-31.78% drawdown (Calmar 0.710) against the incumbent's 20.95% / -34.05% / 0.611. "
     "Paired: +0.105 Calmar on 30/30 fresh seeds, +0.094 on research/166's own seeds, "
     "+0.096 on all 60 paths pooled - against a pre-registered bar of +0.100. It beats a "
     "rate-matched random swap on 30/30, wins both pre-registered windows on 30/30, and "
     "holds its advantage at 40 and 60 bps. research/170 DID NOT ADOPT IT: the bar was written "
     "down before the run and the pooled figure misses it by 0.004, the book had never "
     "traded live, and the rule needed an evening scorer plus a two-leg next-open order "
     "that did not exist. Arun overrode the first reason on 13-Sep-2026 and research/165 built "
     "the third; the second is still true and is what this review exists for. Rule text: OA-ROT-1 in "
     "research/170_qs_leeway_and_baseage_best_entrant/results/RESULTS.md. "
     "WHAT TO VERIFY ON THE DUE DATE, in order: (0) THE SWAP RATE, which is now the first "
     "question rather than an assumption: research/170 measures ~4 swaps a year and this "
     "criterion was written against that, but research/165's 400-session walk of the LIVE "
     "code produced ~12 a year and its 60-day walk ~46. Record the real rate, and look for "
     "churn chains - a name swapped in and swapped out again within ten sessions. If the "
     "live rate is far above 4, RE-READ this criterion rather than quietly re-scaling it. "
     "(0b) P&L ATTRIBUTION of both legs of every swap: what the sold name did after it "
     "was sold and what the entrant did. Over the only window research/165 could walk, the "
     "rule LOST on all three arms (-Rs 23,082 from the live book over 400 sessions). "
     "(1) does the LIVE entry queue show the "
     "shape the rule needs - qualifying signals refused while a holding sits more than 10% "
     "under water? If it does NOT, the rule is "
     "inapplicable regardless of the backtest and should be switched off with OA_ROT1=False "
     "and the reason recorded, and this review "
     "closes. (2) If it does, re-run research/170's five Part-B cells on the LIVE event log "
     "plus the extended history and apply the SAME +0.10 paired Calmar bar, UNCHANGED - do "
     "not move the bar because three evaluations have landed just under it. (3) Report the "
     "tax and tradeability cost alongside: the rule raises tax 25%, drops the win rate from "
     "48.9% to 47.3%, lengthens the worst losing streak from 14 to 15, and raises the share "
     "of profit coming from the ten best realisations from 37.5% to 43.7%. "
     "PASS = either a clean drop with the reason recorded, or a re-run that clears +0.10 on "
     "live-informed evidence and is then raised as its own strategy change with its own "
     "STATUS doc and an after-15:40 deploy. Study: /app/backtest/"
     "qs-leeway-and-baseage-best-entrant-research170"),
    ('IPO Base post-outage health check - did the book actually run every session for a week after the 12-Sep spec change?', '2026-09-19', 'PENDING', "On 12-Sep-2026 the IPO Base book was found DEAD for four trading sessions. Commit 3829ad71 (3-Sep, 'IPO Base marks intraday') rewrote the constants block in services/ipo_paper.py and silently dropped two lines, IPO_TAG and SEEN_ORDERS. Both are read ONLY inside the `mode == live` branch, so nothing failed while the sleeve was on paper - and then it was funded with real money on 8-Sep and every nightly cycle and every reconcile crashed on a NameError from that moment. Four sessions with no exit evaluation and no buy-stops armed on a real-money book. Nothing alerted: the cron redirected the traceback into /tmp/ipo_paper.log and no monitor reads it. Re-checked afterwards, KISSHT would have been HELD on all three missed sessions under both the old and the new exit rules, so no exit was actually missed - that was luck, not design. WHAT TO VERIFY on the due date: (1) backtest_data/ipo_paper_state.json last_run advanced on every trading day since 12-Sep; (2) the nav curve has one point per trading day with no gaps; (3) /tmp/ipo_paper.log and /tmp/ipo_reconcile.log contain no traceback; (4) spec_version reads r167-A and every open position's stop equals 0.90x its fill. THE BROADER FIX IS STILL OPEN and matters more than this one book: a crashing cron on a live book should raise an alert rather than write a traceback to a file nobody reads. Every *_paper.py and *_real.py cron shares that shape."),
    ("Momentum Portfolio - idle cash instrument: pick the arbitrage fund, add the "
     "liquid-ETF buffer, measure the realised post-tax yield",
     "2026-12-15", "PENDING",
     "Every book on /app/mpf-report credits idle cash at 5.2% a year POST-TAX (research/163, "
     "12-Sep-2026). That 5.2% is an ASSUMPTION, not a measurement: an arbitrage fund yields "
     "roughly 6.5% pre-tax at 2025-26 cash-futures spreads and carries EQUITY taxation - 20% "
     "short-term on units churned inside a year, 12.5% long-term beyond a year, ~0.25% exit "
     "load inside a month - which lands at about 5.2% post-tax. The alternative, a liquid ETF, "
     "is taxed at slab: LIQUIDCASE / LIQUIDADD / LIQUIDBETF in market_data.db realised 5.4-5.5% "
     "pre-tax in 2025 and ~5.0% annualised in 2026, i.e. only ~3.5% post-tax at a 30% slab. "
     "Operating rule assumed by the page: BULK IN THE ARBITRAGE FUND, a LIQUID-ETF BUFFER for "
     "money needed at the next open, because arbitrage redemptions settle T+1. "
     "WHY IT MATTERS: cash yield is not a rounding term for these books. True North sits in "
     "cash 57% of the time and IPO Base 68%, so roughly 2.9 and 3.5 points of their CAGR are "
     "the sweep rather than the strategy; each 0.2 points of cash rate is worth (1 - invested) "
     "x 0.2 pp a year to a book. TASKS BY THIS DATE: "
     "(0) OWED BY ARUN, OPERATIONAL, NOT A MODEL CHANGE - move True North's idle cash out of "
     "its current liquid instrument into an ARBITRAGE FUND, keeping a liquid-ETF buffer sized "
     "for the gate's re-entry: the 100-SMA weekly gate LIQUIDATES the whole book and then "
     "re-buys 8 names, so the buffer must cover a full re-entry within T+1 of a redemption, or "
     "the redemption must be placed the day the gate signals. Record the fund chosen and the "
     "date in the Capital Desk / True North dashboard note. NO EXECUTOR CHANGE - do not touch "
     "services/*; this is manual cash management. "
     "(1) Name the actual instrument(s) held and the buffer size. "
     "(2) Measure the REALISED post-tax yield actually earned on the idle balance since the "
     "switch, with its source (broker/AMC statement), not a quoted headline yield. "
     "(3) Re-state the page's cash line as that measured number. "
     "PASS = the report's cash line reads a MEASURED number with its source named. If the "
     "measured rate differs from 5.2% by more than 0.5 points, RE-RUN the curves via "
     "research/163_mpf_cash_yield_harmonisation/scripts/{tn,ba,ipo,oa_vix,qs}_cash052.py with "
     "the yield changed and build_inputs_052.py + check_cash052.py behind them, then "
     "regenerate with research/_utilities/mpf_report_build.py. Each script reproduces its own "
     "published curve at the old yield before changing it, so the re-run is self-gating. "
     "Evidence: research/163_mpf_cash_yield_harmonisation/"
     "MPF_CASH_YIELD_5P2_DAILY_RUN_STATUS.md"),
    ('research/167+168 - IPO Base: the blend test is DONE and says ADOPT THE RE-FIT AT 25%. Arun owes the funding call', '2026-09-26', 'PENDING', "ANSWERED 13-Sep-2026 by research/168, which ran the 3-sleeve blend this review was waiting for. THE ANSWER: the re-fit (Spec A) is worth more to the portfolio than the incumbent on 30 of 30 paired paths at EVERY weight from 5% to 50%, on both cost bases, under monthly rebalancing and under pure drift, and in both halves of the window - +1.73pp of blend CAGR and +0.054 Calmar at a 25% weight, +2.42pp / +0.107 at 35%, +3.45pp / +0.267 at 50%. r/167's correlation-based worry was BACKWARDS and the inference from it is retracted: the refit IS the worse pairwise diversifier (monthly 0.348 to True North, 0.329 to OA Base Age, against the incumbent's 0.259 and 0.319) and is still far the better blend sleeve, because the correlation rise is swamped by the return improvement. The INCUMBENT sleeve fails the pre-registered bar at every weight: it costs blend CAGR on 0 of 30 paths at every weight, never reaches +0.10 Calmar (best +0.089 at 30%), and at the SAME portfolio drawdown it is worth only +1.17pp of CAGR over plain cash at 25%, +1.25pp at its best, and LOSES at 50%. The 2008 concern washes out: the 10.7pp standalone gap becomes 2.5pp at blend weight and BOTH arms improve on the two-sleeve book's -22.3%, while plain cash at the same weight gives the same 2008 cushion the incumbent does for free. RECOMMENDED WEIGHT: True North 37.5% / OA Base Age 37.5% / IPO-A 25%, monthly - 21.18% CAGR after tax [worst path 19.17%] / -24.01% drawdown [worst -26.39%] / Calmar 0.885, against the two-sleeve book's 20.28 / -26.91 / 0.749. 20% is the floor at which the bar clears on every path; 35% is where the refit's edge over the incumbent also clears the pre-registered magnitude; 25% is the capacity-aware choice, fundable on a book up to Rs 80-100L because the sleeve itself caps at ~Rs 20-25L. The Calmar surface actually peaks at a 45-60% weight and the unconstrained simplex wants TN 45 / OA 0 / IPO 55 - NOT recommended (capacity, no held-out period, and it deletes a live book). WHAT IS OWED ON THIS DATE, BY ARUN, NOT BY RESEARCH: the funding decision - what weight the IPO sleeve is actually given, and whether the other two move to make room. Evidence: research/168_three_sleeve_blend/results/RESULTS.md. ORIGINAL r/167 CONTEXT FOLLOWS. ""research/167 (12-Sep-2026) re-measured the ADOPTED r/153 IPO spec on an entry an order can actually place, and it is NO EDGE: 14.90% after tax, -38.6% drawdown, and it LOSES to a date-matched random-entry null run on the same days with the same fill rule (14.90 vs 15.11, real wins 14 of 30 paired seeds; gated, 8 of 30). r/153's own null passed only because it was run on the close-fill arm rather than the live next-day-stop arm. Re-fitted, ONE dial carries the whole edge and it is the trail: real-minus-null is zero or negative at trail <=20, the incumbent's own region, and +4.91pp at trail 50, unanimous 30 of 30 across the whole 30-75 band and reproducing at two independent stop values. Spec A - trail SMA-50, stop 10%, target +25%, plus a NEW no-new-entries-while-NIFTYBEES-is-below-its-150-day-SMA gate - gives 21.80% after tax [20.83..23.19 across 30 seeds], -26.6% median drawdown, Calmar 0.819, against the incumbent's 14.90 / -38.6 / 0.386. The gate is insurance with no premium: a coin flip on return (14 of 30) that removes 17.8pp of drawdown on 30 of 30 paths. TWO THINGS BLOCK ADOPTION. (1) The 3-sleeve blend against True North + OA Base Age was never run, and that is the question that decides it - the refit RAISES correlation to the other books (0.282 to Base Age and 0.256 to True North, against the incumbent's 0.245 and 0.211), so it can be worth less to the portfolio while being worth more standalone. (2) The refit is 10.7pp WORSE than the incumbent in 2008, -10.3% against +0.4%, because the fast SMA-20 trail that costs 7pp a year in normal times is exactly what sidestepped that crash. Read before deciding: 21.80% should be discounted to 19-22% for the ~350 cells scored, the worst-seed drawdown is -32.9% not -26.6%, and capacity caps this sleeve at roughly Rs 20-25L permanently. Decide on the same date as the Base Age paper-book call so the two are settled together."),
    ("research/168 - the TWO-SLEEVE weights themselves: does True North deserve a bigger "
     "share than Open Alpha . Base Age?",
     "2026-09-26", "PENDING",
     "A finding that fell out of research/168 and is NOT what that study was asked: the "
     "two-sleeve True North + OA Base Age book does not prefer the deployed 50:50. On the full "
     "231-combination weight simplex (2006-04-03 to 2026-09-03, 30 paired paths, after tax, "
     "5.2% post-tax idle cash), TN 85 : BA 15 scores Calmar 0.826 against 50:50's 0.749, and "
     "EVERY top simplex cell pushes OA Base Age toward zero (the best three-sleeve cells hold "
     "OA at 0-15%). The mechanism is visible in the sleeve table: the two books earn almost the "
     "same CAGR (True North 19.63%, OA Base Age 19.92%) but OA Base Age carries 8 points more "
     "drawdown (-34.05% vs -25.99%) and is the more index-correlated of the two (0.484 to "
     "NIFTYBEES vs 0.394), so the blend keeps the return and sheds the risk by tilting to True "
     "North. WHY THIS IS NOT YET ACTIONABLE: it is an in-sample Calmar surface on one window, "
     "it would shrink a live book that has twenty years of its own evidence, r/164 and r/166 "
     "are still re-deriving Base Age's slots / sizing / drift, and research/168 deliberately "
     "held the deployed 50:50 ratio FIXED so that study's own question (the IPO weight) was "
     "answered on its own. WHAT TO DO BY THIS DATE: decide whether to open a study on the "
     "TN:OA ratio. If yes it needs its own STATUS doc and its own pre-registered bar, paired "
     "across True North's 12 rebalance-day offsets and Base Age's 30 seeds, with BOTH halves of "
     "the window required to agree and a cash-null at the weight being shifted - the same bar "
     "r/168 used. PASS = either a dated study folder exists, or this is recorded as "
     "deliberately declined with the reason. Evidence: "
     "research/168_three_sleeve_blend/results/extend_grid.csv.gz and section 8 of "
     "research/168_three_sleeve_blend/results/RESULTS.md"),
    ("research/168 - IPO Base weight: re-check the 25% recommendation against the live book's "
     "own fills before funding more than 25%",
     "2026-11-28", "PENDING",
     "research/168 recommends funding the re-fitted IPO Base at 25% of the three-sleeve book "
     "(True North 37.5 / OA Base Age 37.5 / IPO-A 25) and says 35% is justified by the same "
     "evidence IF the live book tracks the model. This review is the gate on that step. THE "
     "MODEL TO BEAT, from r/167 Spec A at 5.2% post-tax idle cash: 22.08% CAGR [worst seed "
     "21.09%], -26.57% median drawdown [worst seed -32.76%], 18.9 trades a year, median hold 37 "
     "days, 49.0% win rate, +6.24% net expectancy a trade. WHAT TO VERIFY: (1) fills within "
     "about 0.5% of the modelled max(pivot, open) on the next-day buy-stop, measured trade by "
     "trade, not in aggregate; (2) the MISS rate - how many armed buy-stops never triggered - "
     "against the model's ~1.5%, which also settles the known paper-branch defect that books a "
     "fill without checking the day's high reached the pivot; (3) trades a year and median hold "
     "in the live sample against 18.9 and 37 days; (4) that the NIFTYBEES < 150-day-SMA gate "
     "fired when it should have, with the dates. WHY IT MATTERS FOR THE WEIGHT: the whole "
     "3-sleeve case rests on IPO-A's return, not on its diversification - it is the worse "
     "pairwise diversifier of the two arms - so a live shortfall in expectancy cuts the "
     "justified weight roughly proportionally. CAPACITY IS THE OTHER CAP: the sleeve's p90 "
     "position is 9.0% of the name's 20-day median traded value on a Rs 10L book, so it does "
     "not scale past roughly Rs 20-25L and 25% of the book is only fundable while the book is "
     "under about Rs 80-100L. PASS = a trade-by-trade fill comparison exists and either "
     "supports the step to 35% or records the measured shortfall and the weight it implies. "
     "Evidence: research/168_three_sleeve_blend/results/RESULTS.md section 10, "
     "research/167_ipo_base_honest_reopt/results/stage9_adoption.csv"),
    ("research/164 - Open Alpha - Base Age: test POSITION DRIFT, the first-order constraint the "
     "slot study exposed",
     "2026-10-10", "PENDING",
     "research/164 (12-Sep-2026) re-fitted the slot count and position size of Open Alpha - Base "
     "Age, which had been INHERITED from the old Open Alpha's look-ahead-scored sweep and never "
     "tested. VERDICT: 16 slots at 6.25% SURVIVES - the Calmar curve does hump at 9-12 slots "
     "(0.629/0.669/0.670/0.633 vs the incumbent 0.601, worth +1.1 to +1.4pp CAGR on 25-27 of 30 "
     "paired seeds and 29-30/30 in the fit window, on a plateau, surviving 40 and 60 bps), but "
     "that is below the pre-registered bar of +0.10 Calmar or +2pp CAGR, and 0 of 32 cells cleared "
     "it. The bar's blind spots argue the other way: concentrating 16->10 slots lifts the ten "
     "best trades' share of total profit from 35.7% to 53.4% and doubles the capacity footprint "
     "(median position 0.43%->0.77% of the held name's own 20-day traded value, 33%->46% of trades "
     "above 1%, ten times larger again on a Rs 1 crore book). WHY THIS REVIEW EXISTS: the study's "
     "biggest finding is that the slot count is the SECOND-order knob. At every slot count the "
     "commonest reason a qualifying signal is not taken is that the book has no CASH, not that it "
     "has no SLOT - 3,619 qualifying events produce 688 entries, 977 refused for want of a slot "
     "and 1,955 refused for want of cash - because the book never trims a winner, so a few bloated "
     "positions absorb 95% of NAV while slots sit nominally free. TO DO BY THIS DATE: test "
     "position drift on the same harness (research/164_baseage_slots_sizing/scripts/sim164.py, "
     "which reproduces research/161 exactly) - trim a bloated winner back toward its target weight, "
     "and/or size the next entry to available cash instead of skipping it. Do NOT re-open the slot "
     "count until that is answered: the slot curve measured in r/164 is the curve of a book that "
     "cannot fill its own slots. Secondary: give the contested-slot liquidity tie-break its own "
     "test - ranking same-day candidates by 20-day traded value beat the random draw at both slot "
     "counts (+2.60pp at 8 slots on 30/30 seeds, +0.78pp at 16 on 29/30) in both windows, is free, "
     "needs no new data and is deterministic, which matters because a LIVE book cannot draw a seed "
     "and the spec gives the operator no written tie-break today. Evidence: "
     "research/164_baseage_slots_sizing/results/RESULTS.md"),

    ("research/160 quality-growth near ATH - quality as an OVERLAY inside Open Alpha's entries - DONE 12-Sep-2026, FAILED",
     None, "DONE",
     "CLOSED EARLY (due was 2026-10-10) by research/162 Part B, run against the HONEST book - "
     "Open Alpha V2.0 / Base Age (r/161), whose no-mask control reproduces its published "
     "21.26% / -34.80% / Calmar 0.618 exactly, so the r/159 look-ahead problem that blocked this "
     "test is no longer in the way. Each eligibility mask was applied to ENTRIES only: a candidate "
     "new-ATH close is dropped unless its symbol passes the screen on the SIGNAL day; exits, sizing, "
     "slot contention and the 60-bar re-arm untouched; 30 seeds; both missing-data policies; two "
     "windows. PASS CRITERION WAS >= +2pp after-tax CAGR or >= +0.15 Calmar on >= 8 of 12 paired "
     "paths. RESULT: NOT ONE SCREEN WINS ON A SINGLE SEED OUT OF THIRTY on return, in either window, "
     "under either policy. Control 26.57% CAGR / -26.57% DD / Calmar 0.999 (2018-08->2026-09, 87% "
     "invested); + b7 (the loose gate this review named) 16.79%, paired -9.71pp on 0/30 and -0.248 "
     "Calmar on 0/30; + quality-only 15.02% (-11.28pp); + growth-only 20.77% (-5.80pp, Calmar +0.014 "
     "on 19/30, still a fail); + the screen as Arun wrote it 8.80% (-17.77pp) at 18.8% invested. "
     "MECHANISM: starvation, not selection - the screen cuts qualifying signals from 3,619 to 468 "
     "(b7) or 76 (strict) and the invested fraction from 87% to 63% or 19%. THE QUALITY-SCREEN LINE "
     "IS CLOSED PERMANENTLY, as this review's own text instructed on a fail. Evidence: "
     "research/162_quality_summit_optimisation/results/RESULTS.md and "
     "/app/backtest/quality-summit-optimisation-research162"),
    ("research/162 - re-open the Quality Summit optimisation ONLY when the holdout has grown a year",
     "2027-09-12", "PENDING",
     "r/162 Part A found a real-looking improvement to Quality Summit - keep the b7 screen, widen the "
     "near-ATH band from k=0.90 to 0.85, cut the book from 15 names to 10, size inverse-volatility - "
     "worth +6.06pp CAGR and +0.282 Calmar on 12 of 12 rebalance offsets in the FIT window "
     "(2018-08 -> 2022-06), on a verified 24-cell plateau, surviving the cost ladder. The "
     "pre-registered HOLDOUT (2022-07 -> 2026-09) returned -3.48pp on 3 of 12 offsets and -0.152 "
     "Calmar on 1 of 12; its holdout CAGR sits 9.22pp below its fit CAGR against a 4pp limit written "
     "down before the run, and deleting its ten best trades takes its compounding proxy BELOW ONE. "
     "NOT ADOPTED. The honest caveat is that the holdout is only four years and is dominated by the "
     "2023-25 smallcap boom, so the failure could in principle be regime rather than overfit. "
     "RE-CHECK WITH A YEAR MORE DATA - and only with a year more data; re-running it on the same "
     "window is holdout mining. PASS CRITERION, unchanged: >= +0.15 Calmar OR >= +2pp CAGR at no "
     "worse drawdown, on >= 8 of 12 offsets, in BOTH windows, with the extended holdout no more than "
     "4pp below the fit window. If it fails again, close the Quality Summit optimisation line "
     "permanently. Cost: about two hours - every derived cache, mask and grid is committed. "
     "Evidence: research/162_quality_summit_optimisation/results/RESULTS.md and "
     "/app/backtest/quality-summit-optimisation-research162"),
    ("research/160 - refresh the point-in-time Screener panel before any reuse",
     "2027-02-01", "PENDING",
     "The panel is now the shared store backtest_data/fundamentals.db (tables screener_annual, screener_quarterly, screener_top_ratios, symbol_map, features_pit_monthly; loader research/160_quality_growth_near_ath/scripts/load_fundamentals_db.py, idempotent INSERT OR REPLACE) as well as research/160_quality_growth_near_ath/results/ (features_pit_monthly.csv.gz + masks/ + masks_study/); fetched 11-Sep-2026 and stops at 2026-09-01. REFRESH = screener_fetch_full.py (delete or age-out the cache first, ~2.25 h) -> build_pit_panel.py -> load_fundamentals_db.py. Screener figures are RESTATED, not as-reported, and the free page depth is about 12 fiscal years, so the panel's earliest usable month walks forward every August as a new FY is filed. Before any future study reuses it, re-run scripts/screener_fetch_full.py (about 2.25 h, resumable) then build_pit_panel.py, build_masks.py and build_study_masks.py. Do NOT reuse a stale panel for a live decision."),
    ("Open Alpha - Base Age LIVE conversion: first two weeks - did the live fills, exits and cash "
     "refusals track the study?",
     "2026-09-26", "PENDING",
     "Repurposed 12-Sep-2026 from the paper-book call: Arun ruled the live Open Alpha book CONVERTS to "
     "Base Age (research/165 staged the code behind OA_RULESET, default legacy). At this review compare "
     "actual AMO fills against the next-open assumption, SuperTrend(14,4) exits against the dry-run lines, "
     "and the count of entries refused for cash against the research/164 base rate (1,955 of 3,619 events). "
     "Also: is the 09:20 equity_executor OA top-up off the OA book. "
     "ADDED 13-Sep-2026, the OA-ROT-1 swap rule: (a) HOW MANY SWAPS in two weeks - research/170 expects ~4 a year, research/165's walk of the live code produced ~12; (b) HOW MANY REFUSALS WERE CONVERTED - refused-for-cash / no-free-slot lines followed by a swap, against those that were not; (c) did any swapped-IN name get swapped OUT again within ten sessions (a churn chain); (d) P&L attribution of both legs of every swap; (e) is the entry-sizing NAV-basis question still open - plan() sizes the slot off cost-plus-cash NAV while the study and rot1_pick() use the marked NAV, which is the difference between PAYTM x20 and x21. "
     "ADDED 13-Sep-2026, research/171 (swap-more-than-one / top-up instead of a new "
     "entrant): BOTH ideas are NO EDGE and nothing changed. What this review should "
     "now ALSO check, because research/171 measured it: on how many evenings did TWO "
     "holdings sit more than 10% under water at the same time while a signal was "
     "refused? Over 21.7 years that happens 0.5 times a YEAR once OA-ROT-1 is running "
     "(11 occasions in the whole history) because the rule keeps removing the loser. If "
     "the live queue shows it materially more often than that, research/171 axis A "
     "should be re-opened on the live event log. Do NOT read a second eligible loser as "
     "a missed opportunity: on the full history, taking it was worth -0.21pp of CAGR and "
     "-0.003 of Calmar, and a real loss at 40 bps. Evidence: "
     "research/171_baseage_multiswap_topup/results/RESULTS.md. "
     "Runbooks: "
     "research/165_oa_baseage_live_conversion/OA_BASE_AGE_LIVE_CONVERSION_DEPLOY_STATUS.md section 9 and OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md section 9."),
    ("Open Alpha - Base Age day-one check after the flip",
     "2026-09-15", "PENDING",
     "Did the 18:50 job run both legs (exit confirmation then entries), which AMO order type did Kite "
     "accept (MARKET or the LIMIT fallback, per /tmp/oa_entry.log), did any exit or entry fill at the open, "
     "and did the 09:20 equity_executor stay off the OA book. If the switch was not flipped, mark N/A and "
     "carry to the next session. "
     "ADDED 13-Sep-2026: the log must also show an OA-ROT-1 block, even if it only says there was nothing to swap - if that block is missing, the swap code did not run and the flip is only half applied. If a swap fired, check BOTH legs are in kite.orders() tagged OA-ROT1-SELL and OA-ROT1-BUY with the SELL first, that reconcile labelled the sale rot1_swap_out rather than rule_exit, that the name sold was the deepest loss in the book AND worse than -10%, and that the name bought was the highest-rs252 refused signal. Anything off: set OA_ROT1 = False, which stops the swap and leaves Base Age running."),
    ("research/159 rounding-base shelf breakout - re-check the post-2016-only verdict",
     "2027-03-12", "PENDING",
     "r/159 was killed as SIGNAL-not-STRATEGY on two legs: it misses the 20% CAGR floor "
     "(14.60% after tax, 30-seed median) and it FAILS the pre-2016 window (8.64% vs NIFTYBEES 12.68%) "
     "while making 20.98% from 2016 on. The pre-2016 window is also the thinnest (12-36 events/yr vs "
     "143 in 2023), so the failure may be a sample-size artefact rather than a regime. Re-check with "
     "six more months of live data: has the event rate held up, and does the out-of-sample stretch "
     "since Sep-2026 look like the 2016+ regime or the pre-2016 one? PASS CRITERION: only revisit the "
     "verdict if out-of-sample expectancy per trade stays above +8% AND correlation to Open Alpha has "
     "fallen below 0.40. Otherwise close it permanently."),
    ("straddle45 LIVE - first-fire review (order path never exercised)",
     "2026-09-12", "PENDING",
     "The executor ships fully tested UP TO order placement; the fill path itself is "
     "first exercised on the first live entry. After the first fire (or the first SKIP), "
     "read /tmp/straddle45_live.log + straddle45_live.db events, reconcile fills against "
     "the broker, and confirm both legs went on together at the expected strike. Three "
     "date bugs were caught in pre-flight - assume more may exist in the untested path."),
    ("Open Alpha - automated execution soak: did the machine trade the spec?",
     "2026-10-09", "PENDING",
     "From 08-Sep-2026 Open Alpha places its own exits (15:18) and arms its own entries "
     "(18:50) - before that it was manual-assisted and had NO entry scanner at all, so a "
     "freed slot could never refill. One month of live evidence, then check: (1) every "
     "exit the 15:18 check raised was actually PLACED and FILLED, with no OA-EXIT order "
     "left resting overnight; (2) every freed slot was refilled by the next session's "
     "scan, so the book sits at 16/16 rather than drifting down; (3) reconcile applied "
     "every tagged fill - book positions and cash match the broker exactly; (4) realised "
     "P&L per closed trade is sane (a cost-rate bug reported -419,999,000 on a Rs1,000 "
     "trade before tests caught it). FAIL on any of these = go back to alert-only while "
     "it is fixed."),

    ("Open Alpha - measure the gap-ceiling deviation",
     "2026-11-08", "PENDING",
     "KNOWN, ACCEPTED DEVIATION from r/142: the study fills entries at max(pivot, open) "
     "with NO ceiling, but Kite refuses SL-M via API ('market orders without market "
     "protection') and the exchange caps a stop-limit's limit-to-trigger spread per scrip "
     "(Rs 8.80 on CUPID, ~3%). So a breakout that gaps more than ~3% above its pivot will "
     "NOT fill where the backtest took it. This is not a rounding matter - gap-ups are "
     "where breakout edges concentrate, so the live book may be systematically missing its "
     "best trades. TASK: count armed-but-unfilled OA-ENTRY orders and the gap size on each, "
     "then replay r/142 with a 3% fill ceiling to price what the constraint actually costs. "
     "If material, the options are a market-protection order type or accepting a lower "
     "expected return than the study advertises - state which."),
    ("Sector indices - back-fill to 2005 inception, then re-ask research/156",
     "2027-03-07", "PENDING",
     "research/156 found NO EDGE in sector rotation and NO ADDED VALUE in sector-gated stock "
     "picking, but the sample is the binding constraint: the 9 real NSE sector indices start "
     "2015-01-01 in market_data.db, giving 11.7 years, one crash (2020) and no 2008. NSE "
     "publishes these series back to ~2005. TASK: back-fill NIFTYAUTO/IT/ENERGY/FINSRV/FMCG/"
     "METAL/PHARMA/PSUBANK/REALTY to inception, then re-run research/156's p1_ic.py and "
     "p2_rotation.py unchanged (both are resume-safe). PASS CRITERION for reopening the line: "
     "any signal family reaching abs(IC t) >= 2.5 on the real indices over the LONGER window "
     "with monotone terciles, AND a rotation config clearing 20% CAGR / Calmar 1.0 after tax. "
     "Otherwise the NO EDGE verdict stands permanently and this review closes. Study: "
     "/app/backtest/sector-trend-rotation-research156."),
    ("CSL-60 DTE-0 paper soak - live-vs-study profile check",
     "2026-11-30", "SCHEDULED",
     "After ~12 live expiry Tuesdays, compare the paper book (services/csl60_paper.py, days table) "
     "against the study profile (r/136 rank-1: WR 63.6%, mean +8,059/day, worst -52k, streak<=6 at 10 lots). "
     "PASS: live WR and mean inside the study's per-12-day sampling band; FAIL: stop or re-study. "
     "Study: /app/straddle-study."),
    ("IPO Base paper soak - fill quality against the pivot",
     "2026-10-15", "SCHEDULED",
     "The IPO Base book went to paper on 2026-09-06 (services/ipo_paper.py, /app/ipo-paper), "
     "running research/153's adopted spec forward on real prices. PASS CRITERION, "
     "pre-registered in research/153: modelled vs actual fill within 0.5% of the pivot and a "
     "miss rate under 15%. This is the only thing the soak needs to answer, because the entire "
     "edge is the fill - taking the signal-day close instead costs 14.08pp of CAGR and loses on "
     "30 of 30 paired seeds. ALSO REPORT: realised selection against the 30-seed band, since "
     "the live book breaks ties by highest 20-day traded value while the backtest drew lots "
     "(28.82-33.44% spread). EXPECT LONG IDLE STRETCHES - the sleeve is 32.7% invested on "
     "average and took no trades at all in 2013-14; that is the strategy working, not decay, "
     "and research/155 already tested redeploying the cash and rejected it. The book arms for "
     "real money on the first deposit routed to it from the Capital Desk."),
    ("IPO Base - is the 25-bar floor in the spec actually tradeable?",
     "2026-12-15", "PENDING",
     "research/153's adopted spec records min_bars 25, but the study's own harness "
     "(ipo_replay.Ctx) builds its panel with 'where n >= 60', so no stock with fewer than 60 "
     "daily bars was EVER shown to the backtest. The published 31.03% CAGR was therefore earned "
     "on stocks aged roughly 3-6 months, not 25 days to 6 months. Found 2026-09-06 by "
     "reconciling services/ipo_paper.py against ipo_replay over 34 trading days: at min_bars=25 "
     "the two agreed on only 75% of signals and EVERY disagreement was the live engine seeing a "
     "genuine recent IPO the study could not (INDOMIM 27 bars, LASERPOWER 37, CORDELIA 48, "
     "TURTLEMINT 50, VAML and VEDPOWER 59). The live book was set to 60 to match what was "
     "actually validated; at 60 the engines agree on 20 of 21 signals, and the single gap "
     "(KISSHT) is the study admitting a stock on its TOTAL bar count rather than its count at "
     "the time - a look-ahead the live engine does not repeat. THE QUESTION: is the 25-60 bar "
     "band tradeable? Those were real breakouts on real IPOs and the sleeve is starved of "
     "candidates, so this is worth a proper study - rebuild Ctx with n >= 25 and re-run the "
     "adopted spec across both bands, 30 seeds, after tax. PASS = the wider band clears the "
     "same Calmar bar without a worse worst-seed. Until then the live book stays at 60."),
    ("research/155 - IPO sleeve idle-cash policy re-check",
     "2027-03-31", "SCHEDULED",
     "research/155 CONCLUDED: leave the IPO sleeve's idle cash in cash (5% p.a.). Arun's forward-visibility proposal was built and tested properly - 114 cells x 30 paired paths, every pull-back friction charged (25/40/60 bps both ways, tax on the realised gain with FY netting, T+1 settlement, pro-rata/LIFO/FIFO lot policy). The premise is CONFIRMED (25 bars plus a 25-day base make a 2-day-old listing ineligible for about five weeks, so the next 25 sessions are fully visible with NO look-ahead) and the mechanism WORKS (0 missed entries, about 30 pull-backs in 20 years) - but it can only touch 2.7% of the portfolio, because the sleeve is 20% of the blend, is 67.3% cash, and the candidate pool is empty on only 19.0% of days. It buys +0.105pp blend CAGR (30/30 paths) but only +0.006 Calmar (21/30) against a pre-registered bar of +0.10 on at least 26 of 30, and the advantage is gone by 40 bps and negative by 60. Continuous redeployment is worse: -0.375 Calmar on 30/30 and correlation to Open Alpha 0.21 -> 0.90. REVISIT ONLY IF (a) the IPO sleeve's blend weight exceeds 30%, or (b) the IPO pipeline has been in drought for more than 12 consecutive months. Then re-run research/155 phase 3b at the then-current weights; PASS = the gated arm clears +0.10 blend Calmar on at least 26 of 30 paired paths AND survives the 40 bps rung. Artifacts: research/155_ipo_cash_redeployment/results/{RESULTS.md, paths.csv, cost_ladder.csv, static_tilt_null.csv}. Published at /app/backtest/ipo-idle-cash-redeployment-research155."),
    ("research/154 six-sleeve blend - allocation decision (2 satellites) + the r/146-153 drawdown re-audit",
     "2026-10-15", "PENDING",
     "research/154 enumerated every combination of the six sleeves (OA, TN, VCP, MYB, IPO, GOLD) - 8,172 cells, all PAIRED across 360 paths (30 OA seeds x 12 TN rebalance-day offsets) - and produced three things Arun must act on. (1) A RETRACTION THAT COMES FIRST: the deployed TN+OA pair's worst drawdown in twenty years is the 2008 crash at -16.5% (monthly marks) / -17.15% (daily), NOT the -2.4% that research/146 and /151 reported. Those studies measured the 2008 window starting 2008-01-01, i.e. AFTER the December-2007 peak, so the drawdown from that peak was invisible. ACTION OWED AT THIS REVIEW: re-audit every per-window drawdown figure in research/146 through /153 for the same window-start artefact, and re-open the crash-alpha candidates that r/146 rejected on the (now false) basis that the pair has no crash tail. (2) THE ALLOCATION DECISION: 197 of 1,767 enumerated weight vectors clear the pre-registered bar on ALL THREE panels (median CAGR at least the pair's, and beating the pair, a cash null AND an IPO beta-matched null on >=288/360 paired paths each) - a broad CONTIGUOUS plateau, not a peak. With operational constraints applied (keep both live books, cap the never-traded sleeve at 20%, cap gold at 20%) the best vector is OA 40 / TN 25 / IPO 20 / GOLD 15 = 28.21% CAGR / -10.77% MaxDD / Calmar 2.61 against the deployed pair's 27.74% / -17.01% / 1.68 on 2006-04 to 2026-08, winning 360/360 paths against the pair, 360/360 against cash and 358/360 against the beta-matched null; its 2008 is +7.3% at a -7.5% drawdown. The DEPLOYABLE-TODAY step, using no unproven sleeve, is OA 60 / TN 15 / GOLD 25 = 28.02% / -13.31% / Calmar 2.095. NOTE research/147's 45/45/10 is NOT admitted - it fails CAGR-at-least-the-pair by 1.13pp on the 2006+ window; gold pays for its drawdown cut with return unless Open Alpha's weight rises to fund it. (3) TWO KILLS TO RECORD: VCP is Open Alpha (87.0% of OA's signals are VCP signals, 48.6% holding-day overlap, correlation 0.749) and MYB shares 90.2% of its signals with VCP - retire both permanently. IPO is the opposite: 0.0% signal AND 0.0% holding-day overlap with OA - not one shared symbol-day in sixteen years. PRE-CONDITION ON ANY IPO WEIGHT: it has never traded, live or paper, and it is invested only 19.6% of NAV (zero trades in 2013 and in 2014), so beyond about 20% weight its extra Calmar is indistinguishable from de-levering. The paper-soak criterion already registered under the research/153 review applies unchanged and must clear BEFORE any rupee. ALSO OWED BEFORE DEPLOYMENT: a quarterly-rebalance sensitivity, because blend-level rebalancing turnover and its tax are NOT modelled. Artifacts: research/154_multi_system_blends/results/RESULTS.md, p1_correlations.csv, p2_subsets.csv, p3_weights.csv, p5_windows.csv, p6_overlap.csv, p7_frontier_OA_TN_IPO_GOLD.csv, p8_daily_marked.csv, p8_yoy.csv, gold_nav.csv (a REBUILT daily gold-in-rupee reference, zero missing months, monthly correlation 0.878 to real GOLDBEES - replaces research/147's cached monthly series which was missing 40 of its 274 months). Published at /app/backtest/multi-system-blends-research154."),
    ("Flask HTTP wedge - outbound connection leak exhausts the gunicorn thread pool",
     "2026-09-19", "PENDING",
     "2026-09-05 (Sat): every /app page and /api route timed out while systemd showed "
     "quantifyd ACTIVE and APScheduler jobs kept ticking every 10s - so nothing alerted. "
     "Cause: gunicorn runs ONE worker with 16 gthreads; the worker had accumulated ~79 "
     "established outbound HTTPS connections (API calls with no timeout that never close). "
     "Once all 16 threads were parked on dead sockets the app served no requests while "
     "looking healthy. Restart cleared it instantly (2ms page responses, conns 79->5); "
     "uptime at failure was ~24h since the 04-Sep 15:45 restart, so treat it as a slow leak "
     "that WILL recur. TWO fixes owed, neither done: (a) explicit timeouts on every outbound "
     "call made inside a request handler (and a session with connection pooling/recycling), "
     "(b) the health watchdog must probe a real HTTP ROUTE - the existing integrity watchdog "
     "checks the process, which was alive throughout and reported nothing. Until then: if "
     "pages hang, restart is the remedy (respect the 15:40 IST rule). Re-check by 19-Sep "
     "whether it recurred and whether the timeout fix has been scheduled."),
    ("research/153 IPO Base - adoption call at 10-20%, then paper-soak decision",
     "2026-10-15", "PENDING",
     "research/153 IPO-Base MID cleared EVERY leg of the pre-registered third-sleeve bar: at "
     "20% weight beside TN+OA it adds +1.13pp CAGR, -3.63pp drawdown and +0.56 Calmar "
     "(27.14/-16.42/1.65 -> 28.27/-12.79/2.21), at correlation 0.16 daily to Open Alpha and "
     "0.18 to True North (LOWER than OA-to-TN at 0.42), and it beats a plain-cash sleeve at the "
     "same weight by +5.60pp of CAGR. Standalone 30 seeds 2006-2026 after tax: 31.03% CAGR "
     "[28.82..33.44], worst seed 28.82%, MaxDD -20.88%, Calmar 1.50, 32.6 trades/yr at "
     "+4.89%/trade net. ARUN TO DECIDE ADOPTION - nothing is deployed. PASS CRITERION IF "
     "ADOPTED: a G5 paper soak whose criterion is pre-registered BEFORE it starts - actual fills "
     "within 0.5% of the modelled pivot and a miss rate under 15% - because the ENTIRE edge "
     "lives in the entry price: filling at the signal-day close instead of the pivot buy-stop "
     "costs -14.08pp of CAGR and loses on 30 of 30 paired seeds. Two operator caveats to restate "
     "at the review: the book earned ONLY the idle-cash yield in 2013 and 2014 (no trades at all "
     "- the IPO pipeline supplied 8-17 usable listings a year in 2012-14 against 80-182 in "
     "2021-25), and capacity is comfortable to about a Rs 10 cr portfolio but binds near Rs 50 "
     "cr. IF NOT ADOPTED, record why and close. Artifacts: "
     "research/153_ipo_base/results/RESULTS.md; listing_dates.csv (a VETTED NSE listing-date "
     "table, 1,293 listings 2006-2026, validated at 48/48 recall and 0/12 leaks - REUSABLE by "
     "any future study that needs listing dates); ipo_equity_seeds.csv; ipo_adopted_spec.json. "
     "Published at /app/backtest/ipo-base-breakout-research153."),
    ("BananaPatterns screen family - re-open ONLY on new evidence (research/151 VCP verdict: NO EDGE)",
     "2027-03-05", "SCHEDULED",
     "research/151 killed the site VCP screen: their published trades contain NO volatility contraction (pivot ages 1-157 bars, 11/37 bases with zero contractions), the null control shows a SHORTER pivot lookback always scores better (2-day null Calmar 2.63 vs 30-day 1.28) so the pattern subtracts value, correlation to the live Open Alpha book is 0.749 daily / 0.759 monthly (bar <0.40), and the best blend weight adds +0.033 Calmar against a +0.10 bar while LOSING to a plain cash sleeve. RE-OPEN CRITERION (pre-registered): only if the site publishes an explicit, reproducible VCP definition (contraction count, tightness ratio, base depth, volume condition) OR supplies a trade list that our 30-day closing-high reconstruction fails to explain in a NEW way. Absent that, cite this study and decline. Artifacts: research/151_vcp_breakout/results/RESULTS.md, p1d_family_scan.csv, p6g_cells.csv, vcp_adopted_spec.json, vcp_equity_seeds.csv; published at /app/backtest/vcp-breakout-research151."),
    ("Four-sleeve study TN/OA/GOLD/MYB - pre-registered weight grid (research/152 "
     "exploratory probe: 80/10/10 = Calmar 2.43 vs 2.08 gold-only, +0.628 paired on 30/30)",
     "2026-11-30", "DONE",
     "DELIVERED BY research/154 on 2026-09-05 - both questions in this review are answered, "
     "see the new research/154 entry below. Original text follows. "
     "research/152 found the multi-year-breakout sleeve (MYB) and r/147's gold sleeve fail in "
     "DIFFERENT windows (MYB earns in the 2018 grind and loses 2022H1; gold is flat-to-positive "
     "in both), so a 4-sleeve book beat either on an exploratory, POST-HOC weight search. That "
     "probe is a hypothesis, not a finding. Before any allocation change: pre-register the weight "
     "grid, the ranking metric and the adoption bar; run BOTH windows; 30 OA seeds x 3 TN offsets "
     "paired against the same-path TN+OA 50-50 baseline; include a cash-null AND a gold-only null "
     "(the relevant question is what MYB adds ON TOP OF gold, not on top of nothing). Note MYB "
     "FAILED r/152's own correlation condition (0.426 daily / 0.535 monthly to OA vs bar <0.40) - "
     "the 4-sleeve study must justify overriding that or drop it. THIRD CANDIDATE ADDED 2026-09-05: research/153 IPO-Base sleeve is now the strongest complement measured - it PASSES the correlation leg MYB failed (0.16 daily to OA, 0.18 to TN) and its own exploratory 4-sleeve cell (40 OA / 40 TN / 10 gold / 10 IPO) scored 29.05% / -11.55% / Calmar 2.52 on 2015+. The weight grid for this review must therefore span TN / OA / GOLD / MYB / IPO, with the gold-only null as the binding comparison. Artifacts: "
     "research/152_multiyear_breakout/results/RESULTS.md, four_sleeve_exploratory.csv, "
     "myb_vs_gold.csv, myb_equity_seeds.csv. SECOND QUESTION IN THE SAME REVIEW: research/152's "
     "post-hoc 50-50 pair check found MYB+OA at 28.71% / -14.5% / Calmar 1.98 vs the DEPLOYED "
     "TN+OA at 26.16% / -16.1% / 1.56 over 2010-2026 (month-end NAV, 30 paths). Do NOT act on "
     "it as it stands: that window excludes 2008, which is the True North gate's entire "
     "documented case (r/144) and is untestable for a multi-year-high screen because the "
     "database holds only 527 symbols from 2005. Answer it with a pre-registered bar, a "
     "gate-matched comparison, and a crash-era proxy - or leave TN alone."),
    ("N500M paper book judgement (research/148 audit: t=1.40 at 10bps floor, "
     "promotion-shrinkage 1.33->0.62%/tr - consistent with selection-on-noise)",
     "2027-03-31", "SCHEDULED",
     "Judge at n>=100 closed trades or 2027-03-31, whichever first. PASS bar "
     "(pre-registered, research/148): net-of-10bps expectancy > 0 with t>=2. "
     "Until then: paper only, NO real-money conversation. Artifacts: "
     "research/148_paper_books_audit/results/RESULTS.md."),
    ("I75WR paper book judgement (research/148: n=8, one sub-system C on one "
     "symbol - no verdict possible; A/B have not fired, check their scanners)",
     "2026-12-31", "SCHEDULED",
     "Judge at n>=40 closed trades or 2026-12-31, whichever first. PASS bar "
     "(pre-registered): net expectancy > 0 with t>=2 AND every config with "
     ">=10 trades individually non-negative net. Artifacts: research/148."),
    ("Open Alpha FULL RESTUDY: joint gate x entry x exit/SL optimization (Arun 2026-09-03)",
     "2026-12-12", "SCHEDULED",
     "The 2026-09-03 gate bake-off (research/142 GATE_BAKEOFF_DAILY_SWEEP_STATUS.md) varied "
     "the gate ALONE with entry/exit frozen; but the -8% stop and trail-20 were originally "
     "tuned under the old SMA200 gate, so gate x exit interactions are untested. Arun: do a "
     "complete restudy and reassessment - best COMBO of gate (DD10 family + finalists), "
     "entry (pivot buy-stop variants), and exit/SL (stop levels, trail lengths, ATR trails) "
     "- plus the 16-slot @6.25% sizing found to cut seed spread 6.7x->2.5x. Run AFTER the "
     "Dec-5 paper-soak review so live-fill evidence informs the entry modeling. Guard "
     "multiple testing: joint grid is large - pre-register the ranking metric and use the "
     "two-window + plateau + worst-seed + PAIRED-BY-SEED discipline (the 30-seed paired test "
     "overturned the 10-seed DD10 median verdict: gate costs -1.6pp/yr on 20/30 seeds, its "
     "value is purely 2008 insurance 30/30; 16-slot no-gate was the free win: +0.1pp median, "
     "worst seed +1.7pp, 2026 dispersion [+4.9..+37.7] with 0/30 negative). ALSO IN SCOPE "
     "(Arun 2026-09-03): (a) model idle-cash CASHIETF yield ~6% in the sim - current engine "
     "holds cash at ZERO, understating all configs and flattering no-gate vs gated; (b) "
     "SELECTION-ALPHA mining: the signal flow far exceeds the book - study which qualifying "
     "breakouts became the big runners, find common causal features (RS, volume surge, base "
     "tightness, sector...), build a ranked-selection rule, validate walk-forward both "
     "directions; lookahead/overfit risk is highest here - causal features only, OOS gates."),
    ("First live dividend declaration (True North + Open Alpha) - verify the 2026-Q3 run",
     "2026-10-01", "SCHEDULED",
     "The 19:15 dividend_declare cron should fire its first real declarations on 30-Sep/01-Oct. "
     "Verify: both books declared exactly once (check ledger via /api/sleeves/dividends), the "
     "HWM was flow-adjusted for any deposits, the outflow left cash/CASHIETF only, the "
     "intimation notice rendered, and the Console withdrawal amount is sensible. Both books "
     "are young - a skipped/zero declaration (NAV below contributed capital) is the CORRECT "
     "outcome, not a failure."),
    ("Stock wings (r/127): REAL basket-margin check - the G4 gate", "2026-09-05", "PENDING",
     "The study sizes on a MODELED margin (1.25x max-loss + 2%, ~6.7% of notional; paper book uses a 10% estimate). Measure real SPAN+exposure via Kite basket_order_margins on a live C1 structure (e.g. the open HDFCBANK/INFY Sep-29 condors) and re-price the CAGR claim: 38.5% at modeled vs 20.2% at 2x. Until measured, quote the 2x row."),
    ("Stock wings (r/127): paper-vs-study tracking review after ~3 cycles", "2026-11-25", "PENDING",
     "Compare realised net per cycle vs the study's +0.264%-of-spot/trade, fill reproducibility at bhav closes, and whether the liquidity gate admits ~10 candidates/cycle as seeded (26 candidates on 2026-08-14). Also revisit earnings-skip once an earnings-date source exists. /app/stock-wings"),
    ("Monday AM live cell (CSL_TIMEB_NIFTY_MON_AM) - keep or kill after 4 live Mondays", "2026-09-22", "PENDING",
     "Went live 2026-08-25 at 8 lots, 09:16-11:16, Rs1,000/lot rupee stop, as an Arun OVERRIDE against r/124 (best Monday cell on measured costs, median +6,920@8L/88.9% win, but FAILS the label-shuffle null p=0.376 on n=18). Judge on 4 real Mondays: realised median vs +6,920, how often the rupee stop fires (modelled 5.6%), realised worst day vs the modelled -15,752, and whether entering 8 lots at 09:16 on the SAME strike as the 6-lot suite + 2-lot COMB shows the r/126 Arm C concentration effect. Paper control: CSL_TIMEB_NIFTY_MON (13:00-14:00)."),
    ("TimeB2 (expiry-Tue afternoon straddle) live-vs-model review after 4 Tuesdays", "2026-09-22", "SCHEDULED",
     "research/125 slot went live 2026-08-25 (8L one-shot); judge live fills vs model in /app/straddles#expiry-lab live_vs_model; also decide SENSEX Thu 13:30-15:00 CSL30 deployment then"),

    ("45-DTE straddle: stress-margin test - RECONSTRUCTION ABANDONED, now forward-recording",
     "2026-11-30", "BLOCKED",
     "ATTEMPTED 2026-08-25 and FAILED its own gate: the SPAN reconstruction hit RMS 12.0% against a pre-declared 10% limit, with structured errors (far strikes under-predicted up to 24.6%, 64-DTE by 17.0%). Diagnosed as a missing vol smile, then found unfixable - the wing strikes carry ZERO open interest and zero volume, so their LTPs are stale by hundreds of points and several have no two-sided quote at all; there is no market vol there to calibrate against. NSE .spn parameter files are not reachable from any public endpoint. Tuning scan parameters until the gate passed would be fitting to the gate, so it was abandoned rather than massaged. REPLACED BY: services/margin_recorder.py, logging real SPAN/exposure daily from 2026-08-25. Re-assess 2026-11-30 once ~3 months of real margin-vs-vol history exists. INTERIM BOUND: Rs 11.96L at 3 lots survives ~1.95x margin inflation (measured requirement Rs 5.99L today); whether Mar-2020 exceeded that is NOT known and must not be asserted."),
    ("45-DTE straddle: paper-vs-study tracking review",
     "2026-11-30", "PENDING",
     "After ~3 completed live-paper campaigns, compare realised net/campaign against the study's +78.1 pts and check the entry/exit fills are reproducible. /app/straddle45"),
    ("DONE 2026-08-23: Monday dropped from live TimeB NIFTY; paper twin studies on; Friday KEPT", None, "PARKED",
     "Arun after r/122: live TimeB NIFTY = Tue DTE0 + Fri DTE2. Monday condemned by three studies (R:R@p95 1:11.8, P(loss) 52%) -> dropped live; Friday initially dropped too, then KEPT on its atlas KEEP verdict (93% win, 1:6.9). CSL_TIMEB_NIFTY_MON (paper, 8L) keeps the Monday cell trading for the 2026-11 re-run; weekly reassessment covers it. SAME DAY: Thu TIMEB SENSEX 10L bump DECLINED - stays 8 lots (p95 tail ~Rs38k at 8L was the deciding number)."),
    ("Re-run r/121 non-expiry filters at ~28 days per cell", "2026-11-14", "PENDING",
     "r/121 ran at ~16 days per cell and found nothing: 540 skip rules produced 10 winners vs ~27 expected by chance, and placebo_noise (a Gaussian random number) beat 97% of random skips. Re-run once each cell has ~28 recorded days. Do NOT act on any filter before then."),
    ("~~MON_NIFTY_DTE1 size decision - cut 8 lots to 3, or drop the cell~~ RESOLVED 2026-08-23", None, "PARKED",
     "RESOLVED 2026-08-23 - DROPPED from live, not downsized. r/124 then swept 3,014 window x stop combinations and found nothing clears 1:3 with workable odds; the Monday cell continues on paper (CSL_TIMEB_NIFTY_MON) for the Nov re-run."),
    ("DATA RULE (r/121): INDIAVIX daily bars are unusable for overnight shocks", None, "PARKED",
     "INDIAVIX daily carries open(d)==close(d-1) on 82.5% of rows, so any overnight VIX shock computed from them is structurally zero and yields a confident FALSE NULL. Rebuild from the 5-minute series (4.1% degenerate). Also: for MAX EXCURSION inside a fixed window, 5-min bars equal 1-min exactly (0 differing rows over 4,068 SENSEX window-days) - the no-5-min rule bites on the PATH only. Third frozen-chain holiday found: 2026-05-28 (adds to r/120s 05-01 and 06-26)."),
    ("CSL_TIMEB2_NIFTY Friday leg - answered: no Friday cell", "2026-09-05", "PENDING",
     "r/120: the TIMEB2 shape (13:00-14:00 SL25) returns -521/lot on Fridays with a -3,786 worst day. Do not add a Friday cell."),
    ("COMB on Friday - weakest component of the stack (r/120 byproduct)", "2026-09-12", "PENDING",
     "r/120: COMB Friday earns +191/lot with a -5,853 worst day, and +656 without a single day. A dedicated Friday review of COMB is a cheaper win than any new slot. Not previously registered."),
    ("DATA RULE (r/120): reject frozen-chain holidays in every options backtest", None, "PARKED",
     "The chain recorder polls on exchange holidays and stores a frozen chain, so 2026-05-01 and 2026-06-26 booked cost-only losses on every window and read as real losing Fridays. Guard: a day with <50 distinct spot prints is not a trading day - reject it. Also note market_data_unified has NO NIFTY 1-minute series (NIFTY50 is 5-min only, ends 2026-07-17); SENSEX 1-min is the only trustworthy long intraday clock."),
    ("SENSEX-Thu backstop LEVEL (not shape) - r/116 byproduct, n=4", "2026-09-11", "PENDING",
     "r/116 NO_DEFENCE control: the defence costs Rs4,913 pooled and buys a worst day of -6,667 vs -16,527 - worth keeping. But that premium is paid almost entirely on SENSEX expiry Thursday: 4 backstop firings cost Rs28,059 for Rs370 of tail improvement. Independently reproduces r/114 at a different stop level on a different construction. Level question with n=4 - needs its own study, do NOT change off this."),
    ("SETTLED (research/119): the 9:16 suite and the CSL books WILL pick different strikes", None, "PARKED",
     "The suite re-snaps to the synthetic forward, the CSL daemon uses spot-ATM. They diverge on ~31% of NIFTY and ~48% of SENSEX mornings - it is a cost-of-carry basis, widening with DTE. Tested: forward-snapping the CSL books earns -65/lot/day and does NOT reduce directional risk (there is none to reduce - these books are short GAMMA, not delta). Both rules stay. Do not re-open on the next divergence."),
    ("Wednesday per-leg stop: G2 study on the 1-min chain (r/118 voided the basis)", "2026-08-27", "PENDING",
     "r/118: Wednesday is the CALMEST weekday (0.8% catastrophic, fewest of any) and the -16,502 day sits inside a 46-day bucket earning +105.7 pts/day at 80% win. The Wed per-leg 30% is now status quo, not evidence. Also: r/118 found bse_options_bhav (290k rows, 2024-2026, real BSE option prices) - use it."),
    ("Re-size SENSEX Thursday for its REAL tail (r/118)", "2026-08-26", "PENDING",
     "DTE0 is the fattest-tailed slot: 34% losers, 8.7% worse than -500pts, worst ~-21,500/lot over 127 days - not the 92%/-127 the 12-day window suggested. Book stop -3,000/lot is the cap; confirm sizing is right before 27-Aug."),
    ("Verify SENSEX Thursday no-stop behaviour live (research/114 deploy)", "2026-08-28", "PENDING",
     "first Thursday under the new rules 27-Aug: legs must carry NO_LEG_SL and never re-arm at breakeven, TP fires only at 4,000/lot, book stop at 3,000/lot; confirm the straddle actually holds to the time exit"),
    ("~~research/117 VIX-shock study - launch and report~~ DONE 2026-08-23", None, "PARKED",
     "DONE 2026-08-23 - folded into research/121. INDIAVIX daily bars proved unusable for overnight shocks (open==prev close on 82.5% of rows); the VIX/OI/PCR angle was tested inside r/121 and found NO EDGE against a random-skip null."),
    ("~~research/116 ratcheting vs static backstop - run and decide~~ DONE 2026-08-22", None, "PARKED",
     "DONE 2026-08-22 - STATIC defence is optimal. Every ratchet/breakeven-clamp/peak-giveback variant gave back more than it saved; r/121 and r/122 concur that tightening manufactures losses. No change deployed."),
    ("~~research/114 SENSEX-Thursday rule change - decide and deploy~~ DONE 2026-08-20, deployed with Arun sign-off: per-leg 30% OFF on SENSEX Thursday (leg_sl_disabled_dtes), TP raised 1,667 -> 4,000/lot DTE-aware, book stop -1,300 -> -3,000 Thu, 50% disaster backstop kept.", None, "PARKED",
     "DONE 2026-08-20, deployed with Arun sign-off: per-leg 30% OFF on SENSEX Thursday (leg_sl_disabled_dtes), TP raised 1,667 -> 4,000/lot DTE-aware, book stop -1,300 -> -3,000 Thu, 50% disaster backstop kept."),
    ("~~SENSEX venue TP Rs1,667/lot - re-test says it COSTS money vs holding to 15:15~~ DONE 2026-08-20", None, "PARKED",
     "DONE 2026-08-20 - resolved by the r/114 deploy: TP is now DTE-aware (Rs4,000/lot on Thursday expiry, Rs1,667 other days) in nas_portfolio_stop.py."),
    ("CSL30F_SENSEX_WED live override vs study - keep or kill", "2026-09-18", "PENDING",
     "user 20-Aug: Wed full-day SENSEX COMB live at 3L AGAINST the -571/day 64% cell (verdict Q4 windows-only); judge on 4 live Wednesdays (26-Aug on)"),
    ("Monday 24-Aug: bump TB-NIFTY and TB-SENSEX 8 -> 10 lots (1.0x plan)", "2026-08-24", "PENDING",
     "edit BOOKS lots/qty in csl_paper_exec.py + backfill before 09:12; scale-up page row-1 target assumes it. r/122 ATLAS VERDICTS now apply: bump is EARNED on Tue (R:R@p95 1:1.5) and Thu (1:3.1 but p95 tail Rs47k/p99 Rs70k at 10L - one 1-in-20 Thursday costs ~3 median Thursdays; weigh before 10L), NOT earned on Mon (1:11.8, DOWNSIZE/DROP per r/120+121+122), marginal on Wed (1:2.9) and Fri (1:6.9). RESOLVED 23-Aug: Mon DROPPED live (paper twin), Thu bump DECLINED (stays 8L, Arun weighed the p95 tail). Still open: Tue 8->10L (atlas says earned, 1:1.5) - needs explicit Arun sign-off before editing lots."),
    ("Weekly: append the weeks actual P&L/DD to /app/scaleup vs the 1.0x target", "every Friday evening", "SCHEDULED",
     "frontend/src/data/scaleup.ts WEEKS row; numbers from the stack reassessment + SENSEX day records; adjust targets only via the reassessment"),
    ("Verify ATM4 SURV roll-stop live behaviour (research/113 deploy)", "2026-08-28", "PENDING",
     "next live roll: log must show rolled-leg SL = max(price_x, roll prem) x 1.3; confirm no noise re-stop; deployed 2026-08-18 after close (MAXV, Arun refinement)"),
    ("Build unified per-system position ledger (portfolio-stop/COMB/manual reconcile)", "2026-08-24", "PENDING",
     "2026-08-17 incident: portfolio stop covered COMB CE at account level (COMB not in its system list, no tag) -> COMB book desynced from broker; had to kill the whole CSL daemon to stop COMB 15:20 phantom exit. BUILD: one tagged ledger every actor reconciles through + broker-qty assert before exit buy + per-book pause (no daemon-wide kill)."),
    ("~~Verify ST-trail intrabar exit fires (confirm-counter fix a792136)~~ DONE 2026-08-24", None, "PARKED",
     "DONE 2026-08-24 - CONFIRMED LIVE: both naked survivors (916-ATM and 916-ATM4 24350CE) exited ST_EXIT @ 15.70 today. The counter advances and the trail fires; the stuck-at-1/3 bug is gone."),
    ("~~Verify first REAL sleeve fills + suite Monday~~ DONE 2026-08-21", None, "PARKED",
     "Morning after 09:16: /tmp/csl_paper.log ENTER [LIVE], Kite CSL_* fills, REAL popups, guardian clean | CLOSED: real sleeve fills observed all week - TimeB NIFTY booked +1,205 / +2,206 / +3,399 and TimeB SENSEX +668 / +2,180 / +229 / +4,534, all tagged CSL_* in the orderbook with fills at the expected marketable-LIMIT levels."),
    ("~~Execute TB6 resize~~ DONE 2026-08-16 (other session, commit aa09496; verified)", None, "PARKED",
     "executor 6L/390, backfill matched, lab per-record normalization correct, deployed row 14L r30.9"),
    ("Suite FRIDAY (DTE2) review - keep live or revert", "2026-08-28", "SCHEDULED",
     "per-leg mechanic measured net-negative DTE2+ by stop-by-DTE study; weigh real Fridays (first: +1,259 on 14-AUG)"),
    ("TB-CSL step to 10L decision - DROPPED for now (user 2026-08-25)", None, "PARKED",
     "sec-18b ladder paused at 8L: precondition (real windows track model for a week) is unverifiable until the Thu/Fri 20-21 Aug external-close fills are reconciled (Console tradebook export pending). Re-open at the 15-Sep paper-vs-model checkpoint or when the unified position ledger lands."),
    ("TB-CSL 8L live-behavior validation", "2026-09-12", "SCHEDULED",
     "real SL-hit days should land near model (-12-15k/6L worst), fills near backfill; else downsize"),
    ("SHIFT vs COMB slot race review", "2026-09-12", "SCHEDULED",
     "NAS_C20_SHIFT paper days vs its model; if tracking, SHIFT challenges COMB for the sleeve slot"),
    ("Sleeves paper-vs-model FULL checkpoint (STRATEGY upgrade / re-freeze / kill)", "2026-09-15", "SCHEDULED",
     "the research/111 gate: live days vs backfill expectations at study lots. AGENDA also: Thursday book-structure cleanup - TB-NIFTY and COMB coincide Thu (same full-day SL20 trade, 10L across two labels; exposure = the studied 10-lot sweep cell, sec-19c chat 19-AUG) - decide whether one book should carry Thursday for clean accounting (economics unchanged)"),
    ("All-stack portfolio-overlay re-test on real intraday MTM", "2026-09-26", "SCHEDULED",
     "sec-17 rerun once nas_mtm snapshots span ~6 weeks (started 2026-08-11)"),
    ("Weekly: review stack_reassessment DRIFT flags", "every Friday evening", "SCHEDULED",
     "panel in Portfolio Lab; DRIFT = review, nothing auto-changes"),
    ("TB2 second-slots paper book review (merge into TB-CSL or drop)", "2026-09-05", "SCHEDULED",
     "CSL_TIMEB2_NIFTY 2L PAPER since 08-18 (Mon 10:00-12:00 + Tue 13:00-14:00 SL25); merge as 2nd slots if tracking sweep model (Mon r46 / Tue r265 combined, in-sample)"),
    ("~~TB-SENSEX first REAL window verify~~ DONE 2026-08-21", "2026-08-19", "PARKED",
     "SUPERSEDED SCOPE 18-AUG: TB-SENSEX 8L REAL + suite Wednesday->PAPER (doc updated); verify Wed 10:30 first REAL window at qty 160, suite paper-mode logs, BFO slippage; THURSDAY margin decision pending user (doc sec 3)"),
    ("SENSEX Wednesday exposure review (suite 9L real on the venue's fat-tail day)", "2026-09-04", "SCHEDULED",
     "studies: Wed p05 -17k/lot, prescription = window+tight-CSL or size-down/skip; weigh 2-3 more live Wednesdays of suite vs TB-window results"),
    ("SENSEX-Thu SL re-decision at n>=4 live Thursdays (study=none+backstop vs decided SL40)", "2026-09-11", "SCHEDULED",
     "verdict handoff Q2: compare live Thu results vs the none/SL40 cells; Arun re-decides with live evidence"),
    ("Scale-step decision at n=4 live Thursdays (SENSEX-Thu concentration vs 1.25x ladder)", "2026-09-11", "SCHEDULED",
     "verdict handoff Q5: prefer SENSEX-Thu step IF live tracks lab; never scale off the 100%-win in-sample cell directly"),
    ("SENSEX stack version (NIFTY-first doctrine)", None, "PARKED",
     "design the SENSEX equivalent once the NIFTY stack has a clean month"),
]

today = date.today()
reviews_out = []
for title, due, status, note in REVIEWS:
    flag = status
    if due and "-" in str(due):
        dd = datetime.strptime(due, "%Y-%m-%d").date()
        if dd < today and status != "PARKED": flag = "OVERDUE"
        elif (dd - today).days <= 3 and status != "PARKED": flag = "DUE SOON"
    reviews_out.append({"title": title, "due": due, "status": status, "flag": flag, "note": note})

out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
       "groups": [{"title": t, "jobs": [{"name": a, "schedule": b, "what": c, "cmd": d} for a, b, c, d in rows]}
                  for t, rows in GROUPS],
       "reviews": reviews_out,
       "note": "Registry convention (.claude/CLAUDE.md): every new lab job or periodic review MUST be added here + docs/LABS_AND_JOBS_REFERENCE.md."}
for p in OUTS:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(p, "w"))
    except Exception as e:
        print("write", p, e)
print("ops_center: %d groups, %d reviews (%d due/overdue)" % (
    len(GROUPS), len(reviews_out), sum(1 for r in reviews_out if r["flag"] in ("DUE SOON", "OVERDUE"))))
for r in reviews_out:
    if r["flag"] in ("DUE SOON", "OVERDUE"): print("  !", r["flag"], r["title"])
