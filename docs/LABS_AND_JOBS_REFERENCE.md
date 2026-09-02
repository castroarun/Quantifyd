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

## Expiry-Afternoon Lab (research/125) - added 2026-08-25

| Item | Detail |
|---|---|
| What | DTE0 afternoon straddle slots (NIFTY Tue / SENSEX Thu); winner TimeB2 = Tue 13:15->14:30 CSL30 8L LIVE |
| Page | /app/straddles#expiry-lab (winners, AlgoTest reference, live-vs-model, run history) |
| Job | cron Tue+Thu 16:05 IST: expiry_lab_assessment.py (re-sweep + DRIFT/WEAK flags + history) |
| Manual | venv/bin/python3 research/125_expiry_afternoon_straddle/scripts/expiry_lab_assessment.py |
| Live runner | timeb2_oneshot.py (one-shot, arm pre-13:15 on expiry Tuesday) |
| Review | 2026-09-22: TimeB2 live-vs-model after 4 Tuesdays + SENSEX Thu slot decision |


## Stock winged strangle PAPER book (research/127) — added 2026-08-25

| Job | Schedule | What | Manual command |
|---|---|---|---|
| bhav stock daily download | 16:20 Mon–Fri (flock) | extends `nse_options_bhav` with the day's F&O STOCK bhavcopy (idempotent) | `./venv/bin/python3 research/89_short_monthly_straddle/scripts/download_nse_bhav_stocks.py` |
| stock_wings_paper seed+mark | 16:50 + 20:30 Mon–Fri (flock) | 45→21 DTE ±2.5% strangle + 7% wings on F&O stocks, ₹20L/10 slots PAPER; publishes `/app/stock_wings_paper.json` for `/app/stock-wings` | `./venv/bin/python3 services/stock_wings_paper.py seed` |

Reviews (in Ops Center): real basket-margin check due 2026-09-05; paper-vs-study tracking review due 2026-11-25.
Study: `/app/backtest/stock-45dte-neutral-wings` · Status doc: `research/127_stock_neutral_wings/STOCK_NEUTRAL_WINGED_STRADDLE_DAILY_SWEEP_STATUS.md`


## Straddle Intraday Study lab (research/136) — added 2026-09-01

| Job | Schedule | What | Manual command |
|---|---|---|---|
| AlgoTest archive DB + `/app/straddle-study` | on-demand (no cron) | 16 AlgoTest CSL runs / 21,172 trades (NIFTY SL 10–300%, SENSEX 30/60%) in `backtest_data/algotest_studies.db`; page filters index/SL/DTE/year-range/events and ranks by net/WR/Calmar/Net-DD/PF/t/median/streak | `python3 scripts/load_algotest_studies.py backtest_data/algotest_csv` |

New exports: scp CSVs to `backtest_data/algotest_csv/` and re-run the loader (idempotent, replaces per run).
Study doctrine + verdicts: `research/136_nifty_csl_portfolio/NIFTY_CSL_ATM_STRADDLE_INTRADAY_SWEEP_STATUS.md` §0c–§0e.

## BlueSky ATH-Breakout Paper Book (research/142, G5 soak — since 2026-09-02)

- **What:** Rs 10L EOD paper book on the adopted taxable spec from the BananaPatterns
  replication study: close > prior ATH-close, IBD-RS>=70, Rs 5cr/day TV floor (no mcap
  floor), buy-stop at pivot next day, -8% stop + SMA20 trail, 8 slots, NIFTYBEES 200-DMA
  gate, 25bps modelled. Intended live use: 50-50 monthly blend with the momentum book.
- **Job:** cron 18:40 IST Mon-Fri -> `services/bluesky_paper.py` (log /tmp/bluesky_paper.log)
- **State:** `backtest_data/bluesky_paper_state.json` (lock + atomic writes)
- **UI:** `/app/bluesky-paper` (reads static/app/bluesky_paper.json — no backend restart)
- **Review:** ops-center dated review 2026-12-05 (soak pass criterion pre-registered)
- **Study:** /app/backtest/bluesky-ath-breakout-research142
