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

### 2026-05-07
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
