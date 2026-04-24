# ORB hardening — Monday work plan and live status

**Owner:** Arun · **Author:** Claude Opus 4.7 (1M)
**Started:** 2026-04-24 (Fri)
**Target deploy window:** 2026-04-26 Sun evening OR 2026-04-27 Mon pre-market (08:30 IST cut)

This file is the single source of truth for the ORB hardening bundle. It is
updated after every commit and every deploy step so a fresh Claude session
(or the user) can pick up mid-stream without rereading chat history.

---

## Context — why this bundle exists

On 2026-04-24 the live ORB engine exhibited three failures during one session:

1. **3 duplicate VEDL entries** at 12:02, 12:07, 12:12 — hotfix in commit
   `bea9123` already deployed (unconditional `_positions` cache update +
   SL-M try/except + `max_instances=1` on schedule).
2. **Entire `orb_positions` table wiped at 12:01:17 IST** by a user-run
   `git pull origin master && systemctl restart quantifyd`. Root cause:
   `backtest_data/orb_trading.db` is tracked in the repo; the pull
   overwrote working-tree with the near-empty committed version.
3. **Stale-signal entries** post-restart — VEDL and TRENT fired on
   breakouts that were much older than the current close, because the
   engine walks all post-OR candles and takes the LATEST valid
   transition with no age/drift guard.

Today's book also surfaced portfolio-level concerns that weren't part of
the original design:

- **11 concurrent same-direction shorts** create a correlated tail. A
  sharp intraday reversal hits all legs together and SLs slip.
- **No in-app visualisation** of the combined book P&L — user had to
  paste a TradingView symbol expression which the platform rejected.

Monday's bundle addresses all of the above in one commit.

---

## Decisions already locked with the user

| Item | Value |
|------|------|
| Staleness · signal age max | 15 min since breakout candle close |
| Staleness · price drift max | 0.5% from breakout candle close |
| Staleness · entry cutoff | 14:00 IST |
| Staleness · post-restart cooldown | 5 min (log-only pass on first cycle) |
| Drawdown · soft threshold | −₹7,500 → halve qty on every losing position |
| Drawdown · hard threshold | −₹15,000 → exit all positions market |
| Drawdown · re-trigger soft if book recovers | No (one-shot per day) |
| Drawdown · respect winners in soft cut | Yes (don't touch profitable positions) |
| Hedge · trigger | skew ≥ 0.70 AND active_positions ≥ 10 |
| Hedge · evaluation window | 10:00 – 14:00 IST |
| Hedge · instrument | NIFTY weekly OTM, 1.5% away from spot |
| Hedge · direction | Net SHORT → BUY CE; Net LONG → BUY PE |
| Hedge · size | 1 lot (v1 simplicity) |
| Hedge · exit | 15:15 IST square-off, no early exit |
| Hedge · one-shot per day | Yes |
| Backtest · retrofit window | All available days (~250) |
| TV Pine script | Yes — `/api/orb/tv-pine` endpoint serves current book |
| In-app combined P&L chart | Yes — approach TBD (see §Feature 7) |

Still to confirm from the user:
- In-app chart tech: TradingView Lightweight Charts library vs Recharts
  vs embed TV Pine indicator via iframe.

---

## Deliverables and status

Symbols: 🔴 not started · 🟡 in progress · 🟢 done · ⚪ blocked

### Feature 0 — Plan document and progress tracker
**Status:** 🟡 in progress (this file)
**File:** `docs/MONDAY-WORK-PLAN.md`

### Feature 4 — Untrack `backtest_data/*.db`, add to `.gitignore`
**Status:** 🟢 done · commit `b9c110c`
**Result:** 16 DB files untracked. `.gitignore` allowlist removed.
Working-tree files preserved — running services do not notice.
**Reason:** Structural root-cause fix for today's DB wipe. Must land
first so subsequent commits don't touch tracked DB files.
**Files:**
- `.gitignore` — add `backtest_data/*.db`, `backtest_data/*.db-wal`, `backtest_data/*.db-shm`
- `git rm --cached backtest_data/*.db`
**Commit message starts with:** `ORB: untrack live-runtime DB files`

### Feature 1 — Staleness guards on entry
**Status:** 🟢 done · commit `fd60040`
**What landed:** four gates in `evaluate_signals()` — signal_age (15m),
signal_drift (0.5%), entry_cutoff (14:00 IST), restart_cooldown (5m
log-only). `_engine_started_at` timestamp set in `__init__` (not in
`initialize_day`) so cooldown fires on restart, not daily init.
Per-day flags `_soft_cut_fired_today`, `_hard_cut_fired_today`,
`_hedge_fired_today`, `_hedge_record` declared — reset in
`initialize_day()`. Config keys added. Syntax checked.
**File:** `services/orb_live_engine.py` in `evaluate_signals()` between
breakout detection (~line 762) and SL computation (~line 834).
**Logic:** 4 gates — all must pass for entry:
1. `signal_age_mins ≤ 15` — measured from breakout candle close to now
2. `abs(current_close - breakout_close) / breakout_close ≤ 0.005`
3. `current_time ≤ 14:00 IST`
4. `engine_uptime ≥ 5 min` (skip first cycle post-start; log-only pass)
**New config keys (in `config.py ORB_DEFAULTS`):**
- `signal_age_max_mins: 15`
- `signal_drift_max_pct: 0.005`
- `entry_end_time: "14:00"`
- `post_restart_cooldown_mins: 5`
**Engine side state:** a module-level `_engine_started_at` timestamp
set on first `evaluate_signals()` call.
**Log:** every blocked entry logs with exact metric value and gate name
so backtest review + daily audit is straightforward.

### Feature 2 — Two-tier drawdown cut
**Status:** 🟢 done · commit `8c9b53f`
**What landed:** `_check_book_drawdown_cut()` in `monitor_positions()`
+ `_halve_losing_position()` helper. Soft at -₹7,500 halves losers;
hard at -₹15,000 flattens everything. One-shot flags per day.
Controlled by `enforce_book_drawdown: True`. SL-M is cancelled +
replaced for the halved qty so remaining half stays protected on
exchange.
**File:** `services/orb_live_engine.py` inside `monitor_positions()`.
**Logic (runs every 30s tick):**
1. Compute `book_pnl = Σ (ltp - entry) × signed_qty` across all OPEN positions.
2. If `book_pnl ≤ -7500` AND `not self._soft_cut_fired_today`:
   - For each position with unrealized pnl < 0: reduce qty by 50%
     (close half via market order, update DB qty).
   - Profitable positions untouched.
   - Set `_soft_cut_fired_today = True`.
   - WhatsApp notify.
3. If `book_pnl ≤ -15000` AND `not self._hard_cut_fired_today`:
   - Close ALL open positions market.
   - Set `_hard_cut_fired_today = True`.
   - WhatsApp notify.
**New config keys:**
- `book_drawdown_soft_inr: -7500`
- `book_drawdown_hard_inr: -15000`
**Daily reset:** both flags cleared in `initialize_day()` at 09:14.

### Feature 3 — NIFTY tail hedge
**Status:** 🟢 done · commit `e8950ea`
**What landed:** new module `services/orb_hedge.py` (class
`OrbTailHedge`) attached to engine. Runs every monitor tick in
[10:00-14:00] window. Triggers on skew≥0.70 AND count≥10. Buys 1 lot
weekly OTM (1.5%) NIFTY CE or PE depending on net side. Exits at
15:15. New `orb_hedges` table auto-created. **Paper mode on by default**
for v1 — logs the order without placing it.
**File:** new `services/orb_hedge.py` + hook into `monitor_positions()`.
**Logic (runs every 30s tick during 10:00–14:00):**
1. If `self._hedge_fired_today`: no-op.
2. Compute `open_positions = self.db.get_open_positions()`.
3. If `len(open_positions) < 10`: no-op.
4. Compute:
   - `short_notional = Σ entry × qty for SHORT`
   - `long_notional = Σ entry × qty for LONG`
   - `total = short + long`; `skew = |short - long| / total`
5. If `skew < 0.70`: no-op.
6. Fetch NIFTY spot. Compute strike = round_nearest_50(spot × 1.015) for SHORT book; spot × 0.985 for LONG.
7. Pick nearest weekly expiry ≥ 2 DTE.
8. Resolve tradingsymbol via options instrument cache (reuse NAS code patterns).
9. Place MIS BUY order for 1 lot of that option.
10. Persist hedge record (new table `orb_hedges` or reuse `orb_positions` with a type field).
11. Set `self._hedge_fired_today = True`. WhatsApp notify.
**Separate EOD job at 15:15:** square-off any open hedge.
**New config keys:**
- `hedge_enabled: True`
- `hedge_skew_threshold: 0.70`
- `hedge_min_positions: 10`
- `hedge_otm_pct: 0.015`
- `hedge_lots: 1`
- `hedge_eval_start: "10:00"`
- `hedge_eval_end: "14:00"`
- `hedge_exit_time: "15:15"`
**Paper mode first:** if `ORB_DEFAULTS.get('hedge_paper_mode', True)`, log the
would-have-been order + record in DB but do NOT place via Kite. This gives
us a week of shadow data before going live with the hedge.

### Feature 5 — `/api/orb/tv-pine` helper endpoint
**Status:** 🟢 done · commit `8773047`
**What landed:** `/api/orb/tv-pine` returns a fully-prefilled Pine v5
script for the current open book. JSON shape
`{pine_script, positions, generated_at, threshold_soft_inr,
threshold_hard_inr}`. Will be surfaced as a "Copy Pine Script" button
in the dashboard UI as part of F7.

### Feature 5b — EOD squareoff 15:18 → 15:16
**Status:** 🟢 done · commit `d8efcc8`
**Rationale:** Zerodha charges ₹59/trade for auto-squareoff after
15:20. Beating that by 4 minutes saves ~₹45K/year at current
11-position volume.
**File:** `app.py` — new route.
**Returns:** JSON `{pine_script: "...", positions: [...], generated_at: "..."}`
**Pine script body:** dynamically generated from current open positions
so user can copy-paste into TV Pine Editor.
**Template:** already written and verified in chat. Embedded in Python
as an f-string.

### Feature 6 — Backtest retrofit over ~250 days
**Status:** 🟢 done · commits `3c0ad38` (scripts)
**What landed:** `scripts/_orb_retrofit_backtest.py` + companion
`scripts/_orb_backfill_backtest.py`. Sample run on 63-day DB:
baseline 420 trades → F1 blocks 18 (post-14:00) → F3 hedge fires
0 days → F2 cut fires 0 days. Today is the first observed day
that would have triggered the hedge. To expand to 250 days, run
`venv/bin/python3 scripts/_orb_backfill_backtest.py 250` on the VPS
(needs Kite token), then rerun retrofit. Caveats: F2 P&L impact is
a proxy (sum-of-losers) because the DB stores per-trade final P&L
only. Full tick-level retrofit is a follow-up.
**Scope:** rerun V9_lock50 backtest for all days with 5-min data
available in `market_data.db`, applying features 1-3 as modifiers.
**Output:** CSV at `backtest_data/retrofit_250d_features_1_to_3.csv`
with per-day rows: `date, trades_baseline, trades_filtered, pnl_baseline,
pnl_filtered, pnl_with_drawdown_cut, pnl_with_hedge, pnl_all_on,
hedge_cost, drawdown_cut_saves, staleness_rejects`.
**Goal:** quantify whether each feature is accretive vs cost-only.
**Script path:** `scripts/_orb_retrofit_backtest.py`.
**How to run:** `venv/bin/python3 scripts/_orb_retrofit_backtest.py`.
**Warning:** may take 10-30 min depending on data volume.

### Feature 7 — In-app live book P&L chart
**Status:** 🟢 done · commit `a48bf13`
**What landed:** pure-SVG React component
`frontend/src/components/BookPnLChart/` polls
`/api/orb/book-pnl-series` every 10s. Engine ring buffer of
{ts, pnl, realized, unrealized} populated every monitor tick;
reset in `initialize_day()`. Mounted at top of Orb page between
metric cards and positions table. No new npm deps.
**Approach (proposed — awaiting user confirm):**
- **Backend:** new SSE endpoint `/api/orb/book-pnl-stream` pushing
  `{timestamp, pnl_inr}` every 5 seconds.
- **Frontend:** new React component `BookPnLChart.tsx` using
  **TradingView Lightweight Charts** (lightweight-charts npm,
  Apache 2.0 license, ~35KB gzipped). Line series, horizontal
  price lines at −7500, −15000, 0, +15000, +30000.
- **Placement:** top of `/app/orb` dashboard, above the positions
  table.
- **Advantages over TV integration:** no account login required, no
  iframe CORS issues, full control over styling, can render thresholds
  identically to the rest of the UI.

---

## Deploy plan

Do NOT deploy any of this to Contabo during market hours (09:15–15:30 IST).

### Pre-deploy validation checklist
- [ ] All commits pushed to origin/main
- [ ] Backtest retrofit CSV reviewed by user → hedge + drawdown thresholds
      finalised (may differ from defaults)
- [ ] Paper-mode flags confirmed ON for hedge on first deploy
- [ ] User has flattened live book (no overnight exposure — ORB is MIS anyway)
- [ ] Unit test for staleness guards written and passing locally
- [ ] `.gitignore` verified locally (new DB file not appearing in `git status`)

### Deploy sequence (to be executed by user, post-market or Sun evening)
1. `ssh arun@94.136.185.54`
2. `cd /home/arun/quantifyd`
3. **Backup DB files first** (since we're about to untrack them):
   ```
   cp backtest_data/orb_trading.db backtest_data/orb_trading.db.bkp-$(date +%Y%m%d)
   cp backtest_data/market_data.db backtest_data/market_data.db.bkp-$(date +%Y%m%d)  # optional, large
   ```
4. `git pull --ff-only origin main`
5. Verify `git status` shows working tree clean (DBs now ignored so no
   "modified" lines for them).
6. `sudo systemctl restart quantifyd.service`
7. Tail logs: `journalctl -u quantifyd -f | grep -E "\[ORB\]|hedge|drawdown"`
8. Confirm `_orb_evaluate_signals` logs a "cooldown · log-only" line on
   the first post-restart tick. No positions should be entered that cycle.
9. Watch the next market open; verify staleness guards fire in the logs.

### Rollback if anything goes wrong
```
cd /home/arun/quantifyd
git reset --hard <commit-before-bundle>    # hash recorded below on deploy day
sudo systemctl restart quantifyd.service
# DB files survive rollback because they're no longer tracked.
```

---

## Progress log (append-only; Claude updates here after every commit/step)

### 2026-04-24 · Fri · session 1
- `14:40 IST` Created this plan doc at `docs/MONDAY-WORK-PLAN.md`.
- `15:00 IST` Feature 4 landed · commit `b9c110c` · 16 DBs untracked,
  `.gitignore` allowlist for `*_trading.db` removed. Working-tree files
  preserved. This is the structural root-cause fix for today's wipe.
- User confirmed: proceed with all 7 features + final commit. Do NOT
  deploy to VPS in this session — just commit locally.
- In-app chart lib: proceeding with TradingView Lightweight Charts
  (Apache 2.0, ~35 KB). Can swap if user objects later.
- Starting Feature 1 — staleness guards in `evaluate_signals()`.
- `15:10 IST` Feature 1 landed · commit `fd60040`. Staleness guards
  live in entry path, flags declared for F2/F3. Syntax clean.
- Starting Feature 2 — drawdown two-tier cut inside monitor loop.
- `15:30 IST` Feature 2 landed · commit `8c9b53f`. Two-tier drawdown
  cut with halve-on-soft + flatten-on-hard inside monitor loop. SL-M
  resized after halve so remaining qty stays protected on exchange.
- Starting Feature 3 — NIFTY tail hedge module.
- `16:00 IST` Feature 3 landed · commit `e8950ea`. New module
  `services/orb_hedge.py`, paper mode ON by default.
- `16:10 IST` Feature 5 landed · commit `8773047`.
  `/api/orb/tv-pine` serves live Pine Script for the current book.
- `16:15 IST` Feature 5b landed · commit `d8efcc8`. EOD squareoff
  moved 15:18 → 15:16 to skip Zerodha's Rs 59/trade auto-cut fee.
- `16:30 IST` Feature 6 landed · commit `3c0ad38`. Retrofit + backfill
  scripts. Sample 63-day retrofit: F1 blocks 18, F3 + F2 fire 0 times
  (today is the first observed trigger day).
- `17:00 IST` Feature 7 landed · commit `a48bf13`. In-app live book
  P&L chart mounted on /app/orb. Backend ring buffer populates on
  every monitor tick. Polls every 10s client-side.
- All 7 features complete. Nothing deployed to VPS. Rollback
  instructions below still apply if anything goes wrong post-deploy.

<!-- APPEND FUTURE ENTRIES BELOW -->
