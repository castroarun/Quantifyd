# Labs, Monitors & Auto-Analysis Jobs — Operations Reference

**Laptop path:** `c:\Users\arunc\Documents\Projects\Covered_Calls\docs\LABS_AND_JOBS_REFERENCE.md` · **VPS:** `/home/arun/quantifyd/docs/LABS_AND_JOBS_REFERENCE.md`
All commands below run on the VPS from `/home/arun/quantifyd` unless marked LAPTOP. Written 2026-08-14.

---

## 1. Daily auto-analysis — the 15:42 regen chain (moved 2026-08-16: all EOD jobs > 15:40; options recorder captures until 15:40) (one cron, six analyzers)

Cron: `42 15 * * 1-5` → `research/58_intraday_recenter_straddle/scripts/regen_straddles.sh`
Re-analyzes everything with the new day's data. Results land on **/app/straddles** within ~80 min.

| Job (in order) | What it refreshes | Where it shows | Manual invoke |
|---|---|---|---|
| v1/v2/variants regen (several scripts inside the sh) | V1/V2 straddle cards, variant lab | /app/straddles top cards | `./research/58_intraday_recenter_straddle/scripts/regen_straddles.sh` (runs the WHOLE chain) |
| `strategy_rankings.py` | **Strategy Leaderboard** (grades, Corr·book, Period) | /app/straddles#leaderboard | `PYTHONPATH=. venv/bin/python3 research/58_intraday_recenter_straddle/scripts/strategy_rankings.py` |
| `sl30_journeys.py` | SL30 card + deep-dive popup data | /app/straddles#sl30-card | `venv/bin/python3 research/58_intraday_recenter_straddle/scripts/sl30_journeys.py` |
| `csl_paper_backfill.py` (~75 min) | BACKTEST day-curves for all paper books (live records always win) | Day P&L curves grid + curve explorer | `setsid nohup venv/bin/python3 research/111_sensex_manual_mgmt/scripts/csl_paper_backfill.py > /tmp/csl_backfill.log 2>&1 &` (avoid 09:00–15:40) |
| `nas_baseline.py` | NAS suite day P&L, REAL/PAPER per-day tags | NAS BASELINE strip in curve explorer | `venv/bin/python3 research/111_sensex_manual_mgmt/scripts/nas_baseline.py` |
| `portfolio_lab.py` | **Options Portfolio Lab** — THE STACK rows, corr matrix, equity/DD curves, source mix | /app/straddles#portfolio-lab | `venv/bin/python3 research/111_sensex_manual_mgmt/scripts/portfolio_lab.py` |

## 2. Weekly auto-analysis

| Job | Cron | What | Manual invoke |
|---|---|---|---|
| `entry_exit_sweep.py` | Fri 15:45 (`45 15 * * 5`) | TB-CSL **Best-Config Lab** regen (entry×exit×SL per DTE). Informational — does NOT move the frozen live book config | `setsid nohup venv/bin/python3 -u research/111_sensex_manual_mgmt/scripts/entry_exit_sweep.py > /tmp/eesweep.log 2>&1 &` |

| `stack_reassessment.py` | Fri 16:35 (`35 16 * * 5`) | **System re-assessment**: corr-drift, per-DTE behavior shifts, TB frozen-windows vs latest sweep, sizing-grid revalidation, live-vs-model tracking → panel in Portfolio Lab | `venv/bin/python3 research/111_sensex_manual_mgmt/scripts/stack_reassessment.py` |

## 3. Intraday execution + monitoring (market hours, Mon–Fri)

| Job | Cron | Role | Manual check |
|---|---|---|---|
| `csl_paper_exec.py` | 09:12 | THE 7 CSL books (NAS_COMB20 + CSL_TIMEB_NIFTY **REAL**, rest paper) | log: `tail -f /tmp/csl_paper.log` · safe dry-run: `venv/bin/python3 research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py --probe` (NEVER run the script without --probe while the cron copy is running) |
| `nas_alert_feed.py` | every 1 min | NAS suite orders → desktop popups (REAL/PAPER tagged) | `venv/bin/python3 scripts/nas_alert_feed.py` · log `/tmp/nas_alert_feed.log` |
| `nas_live_guardian.py` | every 5 min | Hunts live failure classes (stops not firing, churn, P&L misreads) | `set -a && . ./.env && set +a && venv/bin/python3 scripts/nas_live_guardian.py` · log `/tmp/nas_guardian.log` |
| `nas_integrity_watchdog.py` | every 5 min | Pipeline freeze / integrity + email alert | log `/tmp/nas_watchdog.log` |
| `nas_fail_rejected.py` | every 2 min | Failed/rejected order sweeper | log `/tmp/nas_fail_rejected.log` |
| `dump_nas_mtm.py` | every 1 min | Intraday MTM snapshots (feeds future all-stack overlay study) | `logs/dump_nas_mtm.log` |
| portfolio stop / SL monitors | in-app, every 10 s | −₹1,300/lot venue stop, trail/TP, per-leg SLs | `journalctl -u quantifyd --since '10 min ago' \| grep -i monitor` |

## 4. EOD analyzers (after close)

| Job | Cron | What |
|---|---|---|
| `nas_analyzer.py` | 15:45 | Daily NAS RAG report → /app/reports |
| `options_outlier_scan.py` | 15:47 | Options outlier/drift scan → /app/reports |
| `options_study_agg.py` | 15:45 | Opt-Study aggregates (decay/CPR/candles) |
| `snapshot_nas_eod.py` | 15:42 | EOD state snapshots |
| GitHub backup | 16:00 | `backup_to_github_release.sh` |

## 5. Watchers on the LAPTOP

| Watcher | Runs | Manual restart |
|---|---|---|
| `scripts\csl_alert_watcher.pyw` — sticky popups for ALL books (both feeds: CSL + NAS) | auto-start at login (`shell:startup\csl_alerts.bat`), polls every 30 s | `powershell "Get-Process pythonw -EA SilentlyContinue \| Stop-Process -Force; Start-Process 'C:\Users\arunc\AppData\Local\Programs\Python\Python312\pythonw.exe' 'c:\Users\arunc\Documents\Projects\Covered_Calls\scripts\csl_alert_watcher.pyw'"` |
| **nas-live-guardian agent** (Claude) | on demand / periodic review | invoke `/nas-guardian` in Claude Code |

## 6. Manual-only analysis scripts (re-run anytime on fresh data)

All in `research/111_sensex_manual_mgmt/scripts/`, run with `venv/bin/python3` from `/home/arun/quantifyd`:

| Script | Question it answers |
|---|---|
| `per_dte_elimination_check.py` | per-DTE re-ranking of all arms; "eliminate the weak DTE before replacing a system" |
| `sleeve_pstop_test.py` | would a portfolio SL / profit trail help the sleeves? (verdict: no) |
| `nas_suite_csl_replay.py` | suite vs CSL-replacement vs HYBRID, per-DTE, suites + correlations (~2 min) |
| `csl_mgmt_replay.py` | post-CSL management arms: BASE vs TRAIL vs SHIFT (~2 min) |

## 7. Kill / pause levers (for completeness)

| Lever | Effect |
|---|---|
| `POST /api/nas/kill-switch` | suite to paper |
| `touch backtest_data/nas_manual_freeze.flag` | blocks ALL order placement (suite + sleeves) |
| `backtest_data/nas_master_mode.json` → `{"mode":"paper"}` | whole stack (suite + live sleeves) to paper |
| Remove a book's `"mode": "live"` in `csl_paper_exec.py` BOOKS | that sleeve to paper from next morning |

## Option 1-minute OHLC recorder (added 2026-09-01)

| | |
|---|---|
| Job | `scripts/record_option_1min_ohlc.py` |
| Schedule | **15:35 IST, Mon-Fri** (cron, flock `/tmp/opt_1min_ohlc.lock`) |
| Log | `/home/arun/quantifyd/logs/option_1min_ohlc.log` |
| Writes | `backtest_data/options_data.db` -> `option_ohlc` (timeframe `minute`) |
| Scope | NIFTY / BANKNIFTY / SENSEX, nearest 2 expiries, mirrors what the chain recorder tracked that day (~540 contracts, ~140k candles/day, ~5 GB/yr) |
| Deploy doc | `OPTIONS_OHLC_RECORDER_1MIN_DEPLOY_STATUS.md` |

**Why it cannot be skipped:** Kite serves historical candles only for currently-listed
contracts - an expired token returns `InputException: invalid token` (verified 2026-09-01).
**There is no backfill; a day not captured is lost permanently.**

**Why it exists:** `option_chain` is a once-a-minute LTP *poll* and cannot see the
intra-minute high/low that a stop-loss triggers on. That makes stop-based options
backtests unverifiable and maximum-adverse-excursion unmeasurable (proved in
`research/136`). OHLC fixes both.

**Health check:** `SELECT date(date), COUNT(*) FROM option_ohlc GROUP BY 1 ORDER BY 1 DESC LIMIT 5;`
An `invalid token` failure in the log means the job ran too late in the day - move it earlier.

## Sleeves dividend engine — True North + Open Alpha (added 2026-09-03)

| | |
|---|---|
| Job | `scripts/dividend_declare.py` → `services/dividend_engine.py` |
| Schedule | **19:15 IST, Mon-Fri** (idempotent — acts only within 12 days after a calendar quarter end, never re-declares a quarter) |
| Log | `/tmp/dividend_declare.log` |
| State | `dividend` block in `backtest_data/bluesky_paper_state.json` (Open Alpha) and `mp_state` key `dividend` in `momentum_paper.db` (True North) |
| Notices | `services/dividend_notify.py` — registrar-style intimation email + WhatsApp (both dormant until .env keys) + desktop alert feed `/tmp/quantifyd_dividend_alerts.log` |
| UI | `/app/sleeves` Dividends card · `GET /api/sleeves/dividends` · `POST /api/sleeves/dividends/preview` |
| Study | `research/142_bananapatterns_replication/scripts/dividend_sim_v2.py` (adopted variant E) |

**Adopted policy (Arun 2026-09-03):** 25% of new profit above the *flow-adjusted*
high-water mark leaves the book each quarter; the payout is capped at last
dividend +7.5%/qtr (a smooth stepping income line, never a spike); surplus above
the cap banks into a liquid equalization reserve (~6% p.a.) that keeps the line
paying through profitless quarters; if the reserve empties the payout falls and
the line re-bases (an honest cut). Capital is never invaded, positions are never
force-sold (outflow is clipped at cash+CASHIETF), and deposits/withdrawals adjust
the HWM so they never count as profit. Bank payout is manual: the notice carries
the Zerodha Console withdrawal amount (broker APIs cannot push funds to bank).

**Why the trading engines needed no changes:** between record dates both engines
reinvest 100% of booked profits exactly as before; the declaration removes the
entitlement from book cash like a user withdrawal, and each engine sizes off the
smaller NAV at its own next step.


## Paper-book judgement reviews (research/148, added 2026-09-04)

- **N500M** - review 2027-03-31 or n>=100: pass = net-of-10bps expectancy > 0, t>=2. Audit found t=1.40 at the 10bps floor and 53% promotion-shrinkage (expected +1.33%/tr -> live +0.62%): currently consistent with selection-on-noise. Paper only.
- **I75WR** - review 2026-12-31 or n>=40: pass = net expectancy > 0 with t>=2 and every config with >=10 trades non-negative. Only Config C has fired (8 trades, one symbol); check why A/B produce no trades.

## research/151 — BananaPatterns "VCP" screen (review due 2027-03-05)

Verdict **NO EDGE**. The screen reproduces the site exit engine exactly (31/32 trades) but its "volatility contraction pattern" is absent from its own published trades, a null control shows the pattern subtracts value, and the book correlates 0.75 with the live Open Alpha sleeve and loses the blend test to plain cash. Re-open only on a published, reproducible VCP definition. Study: `research/151_vcp_breakout/results/RESULTS.md`; page: `/app/backtest/vcp-breakout-research151`.

### CSL-60 DTE-0 straddle PAPER book (2026-09-07, research/136)
- services/csl60_paper.py — cron every minute 09-15 Mon-Fri; acts only on NIFTY expiry days; log /tmp/csl60_paper.log; manual: ./venv/bin/python3 services/csl60_paper.py mark|show|seed. Renders in NAS Trade Book; review due 2026-11-30 (Ops Center).

## Momentum Portfolio report generator — /app/mpf-report (added 2026-09-11)

Mirrors the Ops & Review Centre entry (`research/111_sensex_manual_mgmt/scripts/ops_center.py`,
group "Momentum Portfolio report (/app/mpf-report)").

| | |
|---|---|
| **Job** | `mpf_report_build` |
| **Schedule** | on demand / after any mpf system change — NOT a cron |
| **Script** | `research/_utilities/mpf_report_build.py` |
| **What** | Regenerates every number and all ten charts on `/app/mpf-report`, the single report page for True North, Open Alpha · Base Age, IPO Base and Quality Summit. Writes `static/app/mpf_report.json`, `frontend/public/mpf_report.json` and `frontend/public/mpf-report-*.png`. |
| **Reads** | `research/163_mpf_cash_yield_harmonisation/results/cash052/{full_period_after_tax_cash052.csv, all_systems_after_tax_cash052.csv, F_Bb7_equity_cash052.csv, athvix_summary_cash052.json}` — since 12-Sep-2026 (evening) these are research/159's and research/160's own curves with **every cash-holding book re-run at 5.2% idle cash**; NIFTYBEES holds no cash and is byte-identical. Plus `research/159_oa_honest_reoptimization/results/after_tax_tables.csv` for the entry-surface / null / gate tables, which the generator swaps automatically for `cash052/after_tax_tables_cash052.csv` once that 5.2% re-run is complete and LABELS the section with whichever rate it used. Read-only: no DB, no engine, no live state. |
| **Manual command** | `cd /home/arun/quantifyd && venv/bin/python3 research/_utilities/mpf_report_build.py` then `export PATH=$HOME/.nvm/versions/node/v20.20.2/bin:$PATH; cd frontend && npm run build`. `--curves-dir` (with `--full-period-csv` / `--roster-csv`) points it at a different curve set — pass `research/163_mpf_cash_yield_harmonisation/results` with the `_cash05` file names to put the curves back on the previous 5.0% basis. To change the cash rate itself, re-run `research/163_mpf_cash_yield_harmonisation/scripts/{tn,ba,ipo,oa_vix,qs}_cash052.py` (each reproduces its own published curve at the old yield before changing it), then `build_inputs_052.py` and `check_cash052.py`. |
| **When to run it** | Whenever a Momentum-Portfolio book changes rules, size or status, or whenever any of the curve files above is re-run. Otherwise the page keeps showing the previous evidence. The frontend rebuild is what moves the PNGs into `static/app/`. |
| **Build doc** | `research/160_quality_growth_near_ath/MPF_REPORT_PAGE_BUILD_STATUS.md` — carries the full provenance table and the three-window resolution. |

**Everything on that page is AFTER TAX**, **every book credits idle cash at 5.2% a year
post-tax, accrued daily**, and every table states which systems, which window and which basis.
The page does not print a figure it cannot point at a file for — a missing value still renders
as a visible gap rather than a zero. Average invested for Open Alpha · Base Age WAS such a gap;
it was measured in research/163 at **72.9%** (30-seed median, daily series in that folder),
which supersedes the ~67% the handover asserted without a source.

**The 5.2% cash rate (set 12-Sep-2026, research/163).** It is the **arbitrage-fund** rate after
tax: arbitrage funds carry equity taxation (20% STCG on units churned inside a year, 12.5% LTCG
beyond a year, ~0.25% exit load inside a month), so ~6.5% pre-tax at 2025-26 cash-futures
spreads is ~5.2% post-tax. A liquid ETF at a 30% slab would be ~3.5% post-tax (LIQUIDCASE /
LIQUIDADD / LIQUIDBETF realised 5.4-5.5% pre-tax in 2025, ~5.0% annualised in 2026). Operating
rule: **bulk in the arbitrage fund, a liquid-ETF buffer** for money needed at the next open,
because arbitrage redemptions settle T+1. It is a **flat assumption, not a measured yield**.
The page reached it in two passes on 12-Sep-2026: True North (6.5%) and Base Age (5.5%) came
onto a common 5.0% first, then all five books moved 5.0% → 5.2%.

**Dated review — 2026-12-15, PENDING** (Ops & Review Centre, top of REVIEWS): *"Momentum
Portfolio - idle cash instrument: pick the arbitrage fund, add the liquid-ETF buffer, measure
the realised post-tax yield."* Task **(0)** is **owed by Arun and is operational, not a model
change**: move True North's idle cash into an arbitrage fund with a liquid-ETF buffer sized for
the gate's re-entry (the 100-SMA weekly gate liquidates all and re-buys 8 names, so the buffer
must cover a full re-entry within T+1 of a redemption, or the redemption must be placed the day
the gate signals), and record the fund and date in the Capital Desk / True North dashboard note.
**No executor change.** PASS = the report's cash line reads a measured number with its source;
if it differs from 5.2% by more than 0.5 points, re-run the curves via the research/163 scripts.

## research/162 — Quality Summit optimisation and the Base Age quality overlay (added 2026-09-12)

Both questions are **answered and closed**; nothing here runs on a schedule.

| Item | State |
|---|---|
| r/160's "quality as an OVERLAY inside Open Alpha's entries" review (was due **2026-10-10**) | **DONE 12-Sep-2026, FAILED.** Not one screen wins on a single seed of thirty on return, either window, either missing-data policy. The quality-screen line is **closed permanently**, as that review's own text instructed on a fail. Ops Centre entry rewritten as a DONE record with the outcome. |
| **NEW dated review — 2027-09-12** | Re-open the Quality Summit optimisation **only when the holdout has grown a year**. r/162 Part A's candidate (b7 screen, near-ATH band k=0.85, ten names, inverse-vol) won the fit window on 12 of 12 offsets and lost the holdout on 1 of 12, 9.22pp below fit against a pre-registered 4pp limit. The holdout is only four years and is dominated by the 2023-25 smallcap boom, so the failure could in principle be regime rather than overfit — but re-running it on the same window is holdout mining. Pass criterion unchanged; if it fails again, close the line permanently. Cost ~2 h; every derived cache, mask and grid is committed. |

Manual re-run (nothing is scheduled):

```bash
cd /home/arun/quantifyd
venv/bin/python3 research/162_quality_summit_optimisation/scripts/build_aux.py        # ~90 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/build_masks162.py   # ~30 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/patch_engine.py     # regenerate qg_engine2.py
venv/bin/python3 research/162_quality_summit_optimisation/scripts/build_panel161.py   # ~45 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/partb_overlay.py    # ~30 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/finalize_a.py       # ~95 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/partc_blend.py
```

## Review 2027-03-13 - OA-ROT-1, the Base Age best-entrant swap (added 13-Sep-2026)
Arun ADOPTED research/170 Part B on 13-Sep-2026: when a qualifying Base Age signal is refused
for a slot or for cash, sell the holding more than 10% under its buy price and buy the refused
signal with the highest 12-month relative strength, both at the next open, at most one a night.
Built into the staged conversion by research/165 behind its own OFF switch `OA_ROT1` in
`services/oa_real.py`; `OA_RULESET` is still `legacy`, so nothing runs yet. Replication gate:
8,488 of 8,488 rotation decisions identical to research/170's engine. The 2027-03-13 review
checks the live SWAP RATE first (~4/yr expected, ~12/yr in research/165's walk of the live
code), then the P&L attribution of both legs, then re-runs research/170's five Part-B cells
against the unchanged +0.10 Calmar bar. The 15-Sep and 26-Sep reviews above carry the swap
checks too. Registered in `ops_center.py` REVIEWS. Runbook:
`research/165_oa_baseage_live_conversion/OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md` section 9.

## Reviews 2026-09-15 and 2026-09-26 - Open Alpha - Base Age LIVE conversion (added 12-Sep-2026)
Live Open Alpha converts to Base Age (research/165, code staged behind `OA_RULESET` in `services/oa_real.py`, default legacy). 15-Sep: day-one check after the flip (18:50 job both legs, AMO type accepted, fills, 09:20 top-up off OA). 26-Sep (repurposed from the paper-book call): first two weeks - fills vs next-open, ST(14,4) exits vs dry-run, cash refusals vs the research/164 base rate. Registered in `ops_center.py` REVIEWS.

## Review 2026-09-19 - IPO Base MIN_BARS 60 vs the validated 25 (research/169, added 13-Sep-2026)
The live IPO book (`services/ipo_paper.py`) runs `MIN_BARS = 60`; research/167 validated Spec A at 25. research/167’s own engine: 21.80% after tax at 25, 11.57% at 60; inside TN/OA/IPO 37.5/37.5/25 the 60-bar book lowers blend CAGR by −2.46pp on 30 of 30 paths. Owed by Arun before the 26-Sep funding call: a recorded MIN_BARS decision (a change is its own STATUS doc, capacity check and after-15:40 deploy). Registered in `ops_center.py` REVIEWS. Evidence: `research/169_ipo_rules_universe_transplant/results/RESULTS.md` Q5-Q6.

## One-shot: Capital Desk rebalance to 37.5 / 37.5 / 25 (Mon 14-Sep-2026)

| Item | Value |
|---|---|
| Launcher | `scripts/deferred_rebalance_20260914.sh` — detached, sleeps to 09:45 IST, re-checks date and 09:40–14:30 window |
| Executor | `scripts/rebalance_mpf_20260914.py --execute` (`--dry [--allow-stale]` plans only) |
| What it does | True North redeems CASHIETF and withdraws; Open Alpha and IPO are credited; all via the Capital Desk |
| Safety | amounts on live values, refused beyond 15% of the approved plan; units sold and verified before any ledger cut; re-dry-run must pay from cash alone; runs once (`logs/rebalance_mpf_20260914.done`) |
| Logs | `logs/deferred_rebalance_20260914.log`, `logs/rebalance_mpf_20260914.json` |
| Approved | Arun, 13-Sep-2026 |
| Review | Ops & Review Centre, due 2026-09-14 |
