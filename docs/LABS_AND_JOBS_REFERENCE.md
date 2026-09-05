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
