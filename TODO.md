# Quantifyd — Project TODO

This file is the **single source of truth for what's pending across sessions**.
Update it when work moves between states. When Arun asks "what's pending",
"what's next", "is anything pending", "where are we" — read this file first.

**Update protocol:**
- New work → add under `Pending`
- Started → move to `In Progress` with date
- Done → move to `Done` with date + commit hash if applicable
- Abandoned → delete entirely (don't accumulate stale "won't do" lines)

---

## Pending (highest-priority first)

### Tri-Sleeve combined book (RS-momentum base + KC6 options + short/hedge)

- **What:** Combine 3 sleeves on ONE Rs.1cr book — (1) the locked
  research/41 MID `q0.5_dd__v__REG` RS-momentum base owns the equity,
  (2) KC6 monetised as credit/debit option spreads, (3) a short/hedge
  sleeve (covered-call vs collar vs systematic-short, decided by backtest).
- **Why:** lift post-tax CAGR and/or cut the base's −24.6% DD without a
  separate capital carve-out. Base alone = 28.9% post-tax / −24.6% / Calmar 1.44.
- **Cross-session note:** the RS base is research/41 (another session,
  COMPLETE/frozen). This work is research/42 — Sleeve 1 only *reads*
  research/41 logic, never edits it.
- **Status / how to resume:** design LOCKED, combined engine not yet
  built, nothing launched. Sole crash-recovery doc:
  `research/42_tri_sleeve_rs_kc6_overlay/TRI_SLEEVE_RS_KC6_HEDGE_DAILY_SWEEP_STATUS.md`
  (§3 spec, §4 ~50-cell grid, §6 resume steps). Next step = build
  `scripts/01_sleeve1_base_replay.py` and pass the Phase-0 fidelity gate
  (must reproduce 35.3% CAGR / −24.6% DD ±0.3pp) before adding overlays.
  Backtests run on VPS (canonical-host rule).

### Restart Flask (local + VPS) to activate I75WR + Pair Trading engines

- **What:** Activate the new `services/intraday_75wr/` package (Configs A/B/C)
  and `services/pair_trading/` package (Config D) — both wired into `app.py`
  with new blueprints + APScheduler cron jobs, both default to PAPER MODE
  (`paper_trading_mode=True, live_trading_enabled=False`).
- **Why:** Backend changes only take effect on Flask restart. Configs sit
  dormant until then. Single-restart picks up both engines + their cron jobs.
- **How (post 15:30 IST Mon-Fri only — CLAUDE.md mandate):**
  ```
  ssh arun@94.136.185.54 'cd /home/arun/quantifyd && git reset --hard origin/main && sudo systemctl restart quantifyd'
  ```
  Local: stop + restart your Flask (`python app.py` or your script).
- **After restart, verify:**
  ```
  curl http://127.0.0.1:5000/api/intraday75wr/state | jq
  curl http://127.0.0.1:5000/api/pair_trading/state | jq
  ```
  Both must show all configs in paper mode. New sidebar entries: **I75WR**
  (between MST and EOD) and **Pairs** (after I75WR). Hard-refresh browser.

### Paper-trade observation period (2-3 weeks) before any live flip

- **What:** Run all 4 configs in PAPER MODE, daily review against backtest
  expectations.
- **Why:** Configs A and B are cost-sensitive — A flips to negative return
  at 0.10%/side slippage. Validate execution quality on real-time tape
  before committing capital.
- **How:**
  - Each evening review `/app/intraday75wr` and `/app/pair-trading` signal
    log + synthetic positions
  - Reconcile against `/app/journal` — paper trades auto-import via
    `services/journal/sync.py`
  - Track actual slippage vs entry/exit-bar prices on each paper signal
  - **Decision rule:** if real slippage exceeds 0.07%/side on the named
    cohorts, default to **Config B** (cost-resilient) NOT Config A when
    going live
  - Log mistake-flags in journal for setups that fired but you wouldn't
    have taken; track missed-signal cases (live setups the engine didn't
    catch)

### Live deploy plan (sequential, reduced capital first)

- **What:** Sequential live-flip after paper-trade validation.
- **Order:** Config B (most cost-resilient) → D (market-neutral, lowest DD)
  → C (low frequency, high PF) → A (only if execution cost stays <0.05%/side)
- **Capital ramp per config:**
  - Week 1-2 live: Rs.2L cap
  - Week 3-4 live: Rs.5L cap
  - Week 5+: Rs.10L full
  Each step contingent on prior-step real-money WR matching paper.
- **API to flip:**
  ```
  curl -X POST .../api/intraday75wr/toggle-mode -d '{"config":"B","mode":"live"}'
  curl -X POST .../api/pair_trading/toggle-mode -d '{"mode":"live"}'
  ```
  Both require BOTH `paper_trading_mode=False` AND `live_trading_enabled=True`
  for real Kite orders (belt-and-suspenders verified by smoke tests).

### App polish — Phase 4 visual refinements

- **What:** Current I75WR + Pair Trading pages are functional but minimal.
  Add charts/sparklines/historical views.
- **Items:**
  - Equity-curve chart on each page (Recharts; currently no chart at all)
  - Per-config historical performance chart (last 30 days) on I75WR
  - Per-pair z-score sparkline (last 60 days) on Pair Trading
  - Trade-by-trade P&L distribution histogram
  - Closed-trades table with link to journal entry
  - Mobile-responsive verification on /app/intraday75wr (cards stack OK on narrow viewports?)
- **When:** After paper-trade observation begins producing real data to chart.

### Trading Journal Phase 2 (deferred from MVP sub-agent)

- **What:** Items the journal MVP sub-agent explicitly deferred:
  - Screenshot upload UI + multipart endpoint (placeholder shown today)
  - 5-min OHLC chart playback on trade page (data is in `market_data_unified`)
  - Kite tradebook reconciliation cron at 16:30 IST + UI page
  - MAE/MFE backfill + slippage tracker
  - NIFTY-regime auto-tagging at trade-open
  - Manual trade entry form (button currently just shows alert)
  - Strategy deep-dive page `/app/journal/strategies/:name`
- **When:** After live trades start flowing; the journal becomes
  load-bearing then.

### Quarterly cohort refresh (mandatory — first refresh due 2026-08-07)

- **What:** Re-screen Pair Trading cohort every 3 months; intraday cohorts every 6 months.
- **Why:** Cointegration breaks on corporate actions; per-stock drift decays as macro shifts.
- **Pair Trading (Config D) — due 2026-08-07:**
  - Re-run `research/39_carry_forward_75wr_quest/scripts/05_pair_universe_screen.py`
  - Replace decayed pairs (cointegration p > 0.05) with newly cointegrated
  - Re-fit alpha/beta on rolling 12-month window
  - Drop pairs whose hedge ratio drifts >2σ from rolling fit
  - Update `PAIR_TRADING_DEFAULTS` in `config.py`
- **Intraday (Configs A/B/C) — due 2026-11-07:**
  - Re-run `research/37_intraday_75wr_quest/scripts/07_per_stock_drift.py`
  - Verify 25 short-diamond + 15 long-MR + 30 long-TC cohorts still produce 60%+ baseline WR per stock
  - Update cohort lists in `results/07_short_diamonds.txt`,
    `11c_long_reversal_diamonds.txt`, `11b_trend_pullback_diamonds.txt`

### Validation cases to verify on first live signal day

- **What:** Two real-world setups Arun flagged live on 2026-05-07.
  When the engines start firing in paper mode, verify these would have
  fired on the relevant historical date:
  - **BOSCHLTD 2026-05-06 → 07** — late-day vol spike + strong-close +
    prev-day-high break + overnight gap-up. Should be caught by sub-agent
    1's BTST detector (research/39).
  - **GODREJCP 2026-05-07** — first-30-min volume breakdown + below
    prev-day-high. NOT covered by current pattern set.
- **Why:** If GODREJCP-style first-30-min breakdowns recur 3+ times in a
  month and our locked engine misses them, queue **Pattern 7 — first-30-min
  range-breakdown** as a follow-up sub-agent.
- **Reference:** `research/39_carry_forward_75wr_quest/CARRY_FORWARD_75WR_DAILY_SWEEP_STATUS.md` section 9.

### CI smoke-test gate

- **What:** Add to CI / pre-commit hook:
  ```
  pytest tests/test_intraday75wr_*.py tests/test_pair_trading.py
  ```
- **Why:** 20/20 smoke tests pass today and verify the paper-mode safety
  lock (live orders require BOTH paper=False AND live=True). A regression
  on these tests would silently break the safety gate. CI catches it.

### Deferred research follow-ups (only if locked configs underperform)

- **PEAD with sector-confluence overlay** — sub-agent 4 hit structural
  ceiling at 65-69% train WR on PEAD alone with real earnings dates.
  Sector-confluence (broader sector also up post-earnings) might push WR
  but is untested. Pursue ONLY if Configs A-D combined underperform paper expectations.
- **Pattern 7 — first-30-min range-breakdown** — see GODREJCP validation
  case above. Pursue ONLY if pattern recurs live and engine misses it.
- **Wider pair-trading universe (Nifty 500 daily)** — current 6-pair Config D
  works on F&O subset. Wider universe → more pairs → diversification but
  also higher cost share + borrow constraints. Pursue ONLY if Config D
  underperforms with the current 6 pairs.
- **Multi-timeframe confluence** — daily breakout + weekly trend +
  sector-leader confirmation. Untested. Low priority.

### Launch Phase F+G research sweeps on VPS

- **What:** Run vol-BO (Phase F) and CCRB (Phase G) sweeps on the 218-stock
  Nifty 500 expanded universe. Both sweeps were partially completed on
  laptop (132/218 vol-BO, 56/218 CCRB) before laptop processes died on
  2026-05-07. Partial CSVs already transferred to VPS — runners will
  resume from skip-set.
- **Why:** Expand the N500M deployable universe beyond the current 27
  stocks to hit Arun's target of 2-3 trades/day (currently ~0.3/day).
- **How:**
  ```
  ssh arun@94.136.185.54 'bash /home/arun/quantifyd/scripts/launch_phase_fg_on_vps.sh'
  ```
  Both run in parallel, single-instance enforced. Logs at
  `/tmp/phase_f_volbo.log` and `/tmp/phase_g_ccrb.log` on VPS. STATUS_MD
  auto-updated by runners at
  `research/34_nifty500_expansion/{VOLBO,CCRB}_RUN_PROGRESS.md`.
- **ETA:** vol-BO ~3-4h, CCRB ~8-12h. Run overnight after market close.
- **After completion:** aggregator will refresh `volbo_leaders.csv` and
  `ccrb_leaders.csv`. The N500M scanner reads these at module-load time —
  next quantifyd restart picks up the expanded universe automatically.

### Verify N500M scanner fires on Friday 2026-05-08 morning

- **What:** Confirm the live-refresh job actually fills today's bars,
  precompute_setup runs at 09:10 IST, and at least one signal evaluates
  to ENTERED or SKIPPED (not silently dropped) by 14:00 IST.
- **How:**
  ```
  ssh arun@94.136.185.54 'curl -s http://127.0.0.1:5000/api/n500m/state | python3 -m json.tool'
  ```
  Check `today_signals` array — should be non-empty by mid-session if any
  of the 27 stocks fire. Even ENTERED:0 + SKIPPED:N counts as proof.
- **If empty after 11:30 IST:** check `journalctl -u quantifyd | grep -i n500m` for scheduler firing + DB freshness via `scripts/_check_market_data_state.py`.

### (Stretch) — N500M scanner trailing-exit logic v2

- **What:** Implement the in-flight trailing logic for `T_CHANDELIER_*`
  and `T_STEP_TRAIL` exit policies in `services/n500m_scanner.py:compute_sl_target`.
  Currently they return only the initial stop with `requires_trailing=True`
  but the executor doesn't update the SL minute-by-minute.
- **Why:** 4 of the top-30 STOCK_CONFIGS use these exits (HDFCAMC CCRB,
  HDFCLIFE CCRB, GODFRYPHLP vol-BO, etc.). Without trailing, they exit
  on initial stop or EOD only — degrades expected Sharpe vs backtest.
- **How:** Extend `n500m_executor.monitor_open_positions()` to recompute
  SL each tick when `position.exit_policy.startswith('T_CHANDELIER')` or
  `'T_STEP_TRAIL'`. Track highest-high (long) / lowest-low (short) since
  entry in DB.

---

## In Progress

_(Move items here when work begins. Include start date.)_

---

## Done — recent (most recent first; trim quarterly)

### 2026-05-15 — research/41 MidSmallcap400-MQ concentrated RS (COMPLETE)

- ✅ **research/41 closed — validated edge that beats the ~20% MQ100 hurdle.**
  Status doc: [research/41_midsmall400_mq_concentrated/MIDSMALL400_MQ_CONCENTRATED_DAILY_SWEEP_STATUS.md](research/41_midsmall400_mq_concentrated/MIDSMALL400_MQ_CONCENTRATED_DAILY_SWEEP_STATUS.md);
  full findings: [MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md](research/41_midsmall400_mq_concentrated/results/MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md);
  full methodology + comparisons: [..._DETAILED_REPORT.md](research/41_midsmall400_mq_concentrated/results/MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md).
  - **Caught + fixed a benchmark bug** that voided run #1: `NIFTY50` daily
    only spans 2023→2026 so 8/12 backtest yrs sat in cash. Fixed →
    `NIFTYBEES` (2005→2026). Run #1 numbers documented as void.
  - **RS-alone:** 75/75 configs beat 20%; 12 robust ex-top-3 (34–39%).
    Pick `mid_120d_N15` = 38.4% CAGR / −29.8% DD / Calmar 1.29.
  - **Phase 03 (drawdown overlays):** winner **`q0.5_dd__v__REG`**
    (quality≥0.5 + SMA200 regime, volume filter rejected) = 35.3% CAGR /
    **−24.6% DD (index-level)** / Sharpe 1.53 / Calmar 1.44.
  - **Phase 04 (OOS + post-tax) PASSED:** edge in both 2014-19 & 2020-26
    halves; walk-forward lookback selection robust (33.1% vs 35.0% static);
    **post-tax 28.9% CAGR @20% STCG** — still ~9pp over the 20% hurdle.
  - **Live deliverable:** [LIVE_TOP15_WITH_FUNDAMENTALS.md](research/41_midsmall400_mq_concentrated/results/LIVE_TOP15_WITH_FUNDAMENTALS.md)
    — 15 picks + web-sourced ROE/D-E/PAT/ROCE; 11/15 fundamentally Strong.
  - Residual: LTCG not netted; PIT universe is liquidity proxy; quality
    leg is price-path not fundamentals; re-run `05_live_top15.py` on VPS
    for a current-dated list. Real-capital use is a user decision (not
    auto-deployed; no live wiring added).

### 2026-05-07 (later — research closure + live engines + UI)

- ✅ **research/37 + 38 + 39 closed; 4 walk-forward-validated configs locked.**
  Final spec: [research/37_intraday_75wr_quest/FINAL_LIVE_SETUP.md](research/37_intraday_75wr_quest/FINAL_LIVE_SETUP.md).
  - Config A: research/37 3-system at TP 0.5/SL 1.5 → 78% WR / PF 1.28 / DD 4.5%
  - Config B: same signals at TP 2.0/SL 1.5 → 53% WR / PF 1.26 / DD 4.5% (cost-resilient)
  - Config C: research/38 multi-bar SHORT bounce at TP 1.5/SL 1.0 → 60% WR / PF 1.69 / DD 3.1%
  - Config D: research/39 6-pair cointegrated pair trading → **78.7% WR / PF 3.57 / DD 0.06%**
  - Total: ~1,895 trades/year combined. The user's original "75% WR + favorable RR + meaningful frequency" target is achieved by Config D (the only mathematically-distinct regime — pair-trading mean-reversion).
- ✅ **Phase 3 Intraday A/B/C live engine built** — `services/intraday_75wr/` package
  (engine_base, config_a/b/c, signal_lib, nifty_regime, db, api). Flask blueprint at
  `/api/intraday75wr/*` (8 endpoints). 7 cron jobs in `app.py`. SQLite at
  `backtest_data/intraday_75wr.db` (5 tables). 10/10 smoke tests pass — verifies
  paper-mode safety lock (real Kite orders require BOTH `paper=False AND live=True`).
- ✅ **Phase 3 Pair Trading Config D paper adapter built** — `services/pair_trading/`
  package (cohort, signal, regime, pair_engine, db, api). Flask blueprint at
  `/api/pair_trading/*` (10 endpoints). Daily 16:00 IST cron job. SQLite at
  `backtest_data/pair_trading.db` (7 tables). 10/10 smoke tests pass.
- ✅ **Phase 4a — Intraday75WR React page** at `/app/intraday75wr` — 3-config cards
  with Off/Paper/Live toggle, day P&L, open positions, recent signals, kill switch.
  Sidebar entry "I75WR" between MST and EOD. Default state: PAPER.
- ✅ **Phase 4b — PairTrading React page** at `/app/pair-trading` — single-toggle
  dashboard, 4-card metrics row, 6-pair grid with z-score gauge per pair, open-positions
  + recent-signals tables, manual scan + kill switch. Sidebar entry "Pairs" right
  after I75WR. Default state: PAPER.
- ✅ **Trading Journal MVP** built earlier this session (sub-agent) — 8-table SQLite
  at `backtest_data/journal.db`, 13-endpoint API at `/api/journal/*`, 4 React pages
  (Calendar, Day, Trade, Insights), seeded with 21 trades from existing strategy DBs.
- ✅ **future_plans.json updated** with both the locked 4-config spec and the
  research/39 carry-forward-75wr-quest WINNER status.
- ✅ **Frontend bundle rebuilt** twice — current hash `index-CSbX8Z55.js` includes
  Journal + I75WR + PairTrading pages. Hard-refresh required on `/app/*` tabs.

### 2026-05-07 (earlier — VPS data infrastructure + N500M)

- ✅ **VPS-canonical data infrastructure** — scp'd 4.85 GB market_data.db
  from laptop to VPS (sha256 byte-identical). All ~17 production code
  references resolve correctly. Fixed hardcoded laptop path in
  `services/intraday_data_bridge.py`. Added VPS-only write guard to
  `services/data_manager.py` (refuses Kite downloads on non-VPS hosts
  unless `ALLOW_LOCAL_DATA_WRITE=1`). Added `QUANTIFYD_HOST=vps` to
  VPS .env. (commits: 41449be, 2f9e03f)
- ✅ **Catchup backfill on VPS** — 5-min: 378/380 stocks current to
  2026-05-07 12:00; daily: 1515/1623 stocks current to 2026-05-07.
  108 daily failures are delisted/illiquid stocks (acceptable).
- ✅ **Live intraday refresh job** — `services/market_data_refresh.py` +
  APScheduler cron every 5 min, 09:15-15:35 IST Mon-Fri. Fetches past
  15 min of 5-min bars for the N500M universe. (commit: 7d9cab0)
- ✅ **Phase F+G migration helpers staged** — `scripts/launch_phase_fg_on_vps.sh`
  with single-instance enforcement, `scripts/transfer_phase_fg_partial_csvs.ps1`
  for the optional partial-CSV pre-step. (commit: d805356)
- ✅ **Laptop replacement recovery** — `docs/LAPTOP-REPLACEMENT-RECOVERY.md`
  + `scripts/pull_market_data_from_vps.py` so a fresh laptop can become
  productive in <30 min with zero data loss.
- ✅ **N500 Momentum live scanner v1** — `services/n500m_{configs,db,scanner,executor}.py`
  + `frontend/src/pages/N500m.tsx` + `/api/n500m/*` routes + APScheduler
  jobs. 3-button OFF/PAPER/LIVE toggle, 30 configs / 27 stocks ranked
  by Sharpe, fail-closed default OFF. End-to-end paper trade verified
  (HAL 5m short 2026-02-04 → +Rs 8,449). (commits: 11ff8b2, 7d9cab0)
- ✅ **VPS quantifyd restart** — picked up `QUANTIFYD_HOST=vps` env var
  + new live-refresh APScheduler job. Mode=PAPER, 30 configs loaded.
- ✅ **Phase D done** — CCRB aggregation surfaced 9 promote candidates
  (HDFCLIFE, ITC, IDFCFIRSTB, SUZLON, RVNL, BHARTIARTL, BAJAJFINSV,
  REDINGTON, CDSL) and 8 robust-on-both names with vol-BO.

### 2026-05-06
- ✅ **Phase C CCRB sweep on top-100** — 100/100 stocks, 876,870 signals,
  in 418 min on laptop.

### 2026-05-05
- ✅ **Phase A backfill** — 380 N500 stocks have 5-min data
  (15 lost to delisting/renames).
- ✅ **Phase B vol-BO sweep on top-100** — 230K signals, RESULTS.md
  with top-15 leaderboard.

---

## Conventions for this file

- One H3 (`###`) per item under `Pending` / `In Progress` / `Done`.
- Each item has a clear **What / Why / How** when it's pending.
- Done items get a one-line summary + date + commit hash.
- Trim `Done` to the last 3-4 weeks; older entries belong in research/* notes.
