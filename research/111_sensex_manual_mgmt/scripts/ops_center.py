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
    ("Kill / pause levers", [
        ("Freeze flag", "instant", "blocks ALL order placement (suite + sleeves)", "touch backtest_data/nas_manual_freeze.flag"),
        ("Master mode", "instant", "whole stack to paper", "echo '{\"mode\": \"paper\"}' > backtest_data/nas_master_mode.json"),
        ("Suite kill-switch", "instant", "suite to paper via API", "curl -X POST http://127.0.0.1:5000/api/nas/kill-switch"),
        ("Per-book live flag", "next morning", "remove \"mode\": \"live\" from the book in csl_paper_exec.py BOOKS", ""),
    ]),
]

# Periodic reviews / re-assessments — THE calendar. status: PENDING | SCHEDULED | PARKED
REVIEWS = [
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
