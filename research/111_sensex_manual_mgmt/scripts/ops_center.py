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
    ("research/116 ratcheting vs static backstop - run and decide", "2026-08-21", "PENDING",
     "Arun 20-Aug: a backstop anchored to entry becomes a huge give-back once the trade wins (TB-SX: 180 pts / ~Rs29k away at +Rs15k open). Test breakeven-clamp, multiplicative ratchet, peak-giveback, time-scaled. Applies to every combined-SL book."),
    ("research/114 SENSEX-Thursday rule change - decide and deploy", "2026-08-21", "PENDING",
     "drop per-leg 30% on SX Thursday + raise/retire the Rs1,667 TP, keep the 50% backstop; needs Arun sign-off, deploy after 15:40; re-run the bake-off when Thursdays double (~Nov)"),
    ("SENSEX venue TP Rs1,667/lot - re-test says it COSTS money vs holding to 15:15", "2026-08-22", "PENDING",
     "tp_validation.py on 34 recorded sessions: TP fired 16 (47%), mean -4,241 per TP day, -67,850 total vs 15:15 hold, worst give-up -26,163. Conflicts with research/91 (SENSEX fades into close). NOT changed live - needs the study sessions verification of method (credit/lots reconstruction) before any rule change."),
    ("CSL30F_SENSEX_WED live override vs study - keep or kill", "2026-09-18", "PENDING",
     "user 20-Aug: Wed full-day SENSEX COMB live at 3L AGAINST the -571/day 64% cell (verdict Q4 windows-only); judge on 4 live Wednesdays (26-Aug on)"),
    ("Monday 24-Aug: bump TB-NIFTY and TB-SENSEX 8 -> 10 lots (1.0x plan)", "2026-08-24", "PENDING",
     "edit BOOKS lots/qty in csl_paper_exec.py + backfill before 09:12; scale-up page row-1 target assumes it"),
    ("Weekly: append the weeks actual P&L/DD to /app/scaleup vs the 1.0x target", "every Friday evening", "SCHEDULED",
     "frontend/src/data/scaleup.ts WEEKS row; numbers from the stack reassessment + SENSEX day records; adjust targets only via the reassessment"),
    ("Verify ATM4 SURV roll-stop live behaviour (research/113 deploy)", "2026-08-28", "PENDING",
     "next live roll: log must show rolled-leg SL = max(price_x, roll prem) x 1.3; confirm no noise re-stop; deployed 2026-08-18 after close (MAXV, Arun refinement)"),
    ("Build unified per-system position ledger (portfolio-stop/COMB/manual reconcile)", "2026-08-24", "PENDING",
     "2026-08-17 incident: portfolio stop covered COMB CE at account level (COMB not in its system list, no tag) -> COMB book desynced from broker; had to kill the whole CSL daemon to stop COMB 15:20 phantom exit. BUILD: one tagged ledger every actor reconciles through + broker-qty assert before exit buy + per-book pause (no daemon-wide kill)."),
    ("Verify ST-trail intrabar exit fires (confirm-counter fix a792136)", "2026-08-21", "PENDING",
     "next naked ATM/ATM4 survivor breaching its trail: log should show 1/3->2/3->3/3 then TRAIL EXIT intrabar, not stuck at 1/3. Deployed+restarted 2026-08-17 15:48."),
    ("Verify first REAL sleeve fills (marketable-LIMIT fix) + suite Monday", "2026-08-17", "PENDING",
     "Morning after 09:16: /tmp/csl_paper.log ENTER [LIVE], Kite CSL_* fills, REAL popups, guardian clean"),
    ("~~Execute TB6 resize~~ DONE 2026-08-16 (other session, commit aa09496; verified)", None, "PARKED",
     "executor 6L/390, backfill matched, lab per-record normalization correct, deployed row 14L r30.9"),
    ("Suite FRIDAY (DTE2) review - keep live or revert", "2026-08-28", "SCHEDULED",
     "per-leg mechanic measured net-negative DTE2+ by stop-by-DTE study; weigh real Fridays (first: +1,259 on 14-AUG)"),
    ("TB-CSL step to 10L decision (after 8L behaves)", "2026-08-24", "SCHEDULED",
     "sec-18b ladder: 6->8 done 08-17 (eff 08-18); step 8->10 if real windows track model for a week"),
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
    ("TB-SENSEX first REAL window verify (if other session deploys tonight)", "2026-08-19", "PENDING",
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
