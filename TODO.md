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

### 🔴 TOP: Persist NAS live/paper mode across restart (mode is in-memory only)

- **What:** `/api/nas/master-mode` POST writes the live/paper flag **only to the
  in-process `*_DEFAULTS` dicts** ([app.py:4370-4372](app.py#L4370-L4372)).
  Nothing persists it, and the bootstrap reads `config.py`'s
  `paper_trading_mode=True` default on boot. So **every restart silently
  reverts all 8 NAS variants to PAPER**, even if they were LIVE.
- **Why it's critical (real incident 2026-05-29 ~12:25):** a mid-market restart
  reverted live→paper. The broker still held ~15 live legs, but the now-paper
  app stopped managing them with real orders, **paper-closed the Sq-OTM legs in
  the DB while they stayed open at the broker** (orphan), and a paper squeeze
  re-entry created phantom rows. Recovery required a manual re-flip to live +
  DB re-adopt + phantom cleanup. If unnoticed, the EOD squareoff would have
  "paper-squared" while real shorts sat open overnight-risk.
- **Fix:** persist the master mode to a small state file (e.g.
  `backtest_data/nas_master_mode.json`) on every POST, and have the Flask
  bootstrap read it on startup and apply it to all 8 `*_DEFAULTS` before tickers
  start. Fail safe: if the file is missing/corrupt, default to PAPER. Show the
  resolved mode in the boot log.
- **Related bugs found same session (stage alongside):**
  1. **Cross-variant OTM pooling** — `_check_premium_tick` pools Sq-OTM + 916-OTM
     legs into one comparison ("Subscribed N OTM legs (Sq + 916)") with a single
     global `_adj_triggered`/`_adj_next_direction`, and `_fire_tick_adjustment`
     hard-routes every roll to `NasExecutor` (squeeze). So a 916 leg's premium
     can trigger a roll on the squeeze book, and 916-OTM never gets its own roll.
     Scope the comparison + routing per-strangle/per-variant.
  2. **No-strike → strangle close instead of roll** — when the roll target
     premium has no strike in `[adj_min, adj_max]`, the OTM executor does an
     emergency `close-both` ("No live strike found … closing strangle"). On
     2026-05-29 this repeatedly closed the Sq-OTM instead of rolling. Widen the
     search / pick nearest strike rather than exiting the position.
  3. **Malformed weekly symbol on paper re-entry** — the paper Sq-OTM re-entry at
     12:30 created `NIFTY266224300CE` (16 chars, missing a digit vs the correct
     17-char `NIFTY2660224300CE`) → phantom DB rows that can't resolve a token.
     Audit the strike→tradingsymbol builder used on (re-)entry.
  4. **Cards advertise behavior the code doesn't do** — ATM/ATM2 cards say
     "close BOTH + re-enter, max 5 re-entries" but the code does close-stopped +
     naked-survivor-ST(7,2) (plain ATM) or close-both-no-reenter (ATM2). Decide
     intended behavior (restore cascade re-entry vs fix the card text). [user
     to choose A/B]
  5. **Broken strike selection on squeeze re-entry (highest-impact OTM bug)** —
     on 2026-05-29 the Sq-OTM kept re-entering the SAME strikes (24300CE/23400PE)
     on each fresh squeeze (12:25/12:30/12:35/12:45) instead of premium-based
     re-selecting near ~₹20. As spot drifted, those strikes went deep-OTM
     (24300CE entered @₹3.56) and the no-strike→boundary-exit bug (#2) instantly
     closed them, writing garbage P&L (mismatched exit prices: sold @3.56,
     recorded exit @21.05 → fake −₹2,274). Real money was ~flat (broker
     round-tripped in seconds, daily_pnl=0) but it churned real orders + polluted
     the dashboard (headline −₹4,143 vs curve +₹497 vs daily_pnl 0 — three
     disagreeing P&L calcs). **TEMP STOP applied 2026-05-29 12:55:** Sq-OTM
     `entry_end_time` hot-set to "12:00" via /api/nas/config (blocks entries, no
     restart; reverts to 14:30 on next restart). Fix the entry to premium-select
     fresh strikes every time, and re-enable (reset entry_end_time) once fixed.
  Also: the Day-P&L reporting inconsistency (state.daily_pnl vs closed-trade-sum
  vs MTM curve) needs a single source of truth so the dashboard can't show
  three different numbers.
- **✅ DEPLOYED LIVE 2026-05-29 ~22:48 IST (after close, pull + restart):**
  persist live/paper mode (`262237f`, fixes the revert — **VALIDATED**: boot
  logged "restored persisted mode on boot: 'live'"), no-strike→HOLD not close
  (`033d12d`, bug #2), per-leg premium floor on entry (`2844f6a`, bug #5). Boot
  reconcile orphans=0, all 8 variants FLAT post-EOD, mode=live persisted.
  Sq-OTM `entry_end_time` auto-reverted to 14:30 on restart → re-enabled for
  tomorrow with the fixes in place (no churn expected). **This TOP item is
  RESOLVED** — persist-mode is fixed & proven.
- **Still open (deferred — now non-urgent):** bug #1 cross-variant OTM pooling +
  per-variant executor routing. With #2 (hold) + #5 (no junk entries) + the 90s
  cooldown, the cross-variant trigger no longer drives destructive loops, so this
  is a correctness cleanup — do it as a focused, tested change. Bugs #3 (malformed
  re-entry symbol) and #4 (stale ATM card text / A-vs-B decision) also open.
- **Already live (earlier today):** 916-ATM4 roll mode/ACTIVE (`4eccc96`), OTM roll
  90s cooldown (`560bfa0`), OTM cap 20→40 (`38ddc76`), dashboard strike display
  (`97a1d22`). NAS re-flipped LIVE in-memory; Sq-OTM TEMP-STOPPED via in-memory
  `entry_end_time=12:00`.

### NAS `max_daily_orders` — raise defaults or scale per active strangle

- **What:** Today's incident — Sq OTM hit its `max_daily_orders=20` cap by
  14:44 IST and got gagged from any further orders, **including its own EOD
  squareoff and 14:45 time exit**. 28 orders used today = roughly 14 roll
  cycles × 2 orders each (close + open). Symptom in journal:

  ```
  WARNING:services.nas_ticker:[NAS] Tick adjustment failed: Daily order limit (20) reached
  ```

- **Why this happened today (root cause analysis):**
  1. This morning's PENDING-bypass-guardrail bug let Sq OTM accumulate
     5 strangles instead of the design intent of 1. (Now fixed in commit
     `049d727` — won't recur.)
  2. The cross-leg roll trigger (`_check_premium_tick`) compares the
     globally-most-expensive vs globally-cheapest leg across **all**
     subscribed OTM tokens. With 10 OTM legs (5 strangles × 2 legs each)
     at slightly drifted strikes, the 2.0× ratio condition was *almost
     always true* — every leg-pair was close enough to the threshold that
     tiny tick movements kept firing roll attempts. Each roll = 2 orders.
  3. 20 daily order cap was sized for `max_strangles=1` (i.e. 1 entry +
     ~9 roll cycles = 19 orders). With 5 strangles in play, that budget
     was burned through in ~3 hours.

- **Why the simple fix is risky:** Just bumping defaults (e.g. OTM 20→40)
  treats the symptom. The real failure mode that 20 was protecting
  against is a *runaway adjustment loop* — if the ticker callback bugs
  out and keeps firing rolls forever, the daily cap is the last line of
  defense before the broker rate-limits us or we burn through margin in
  fees. Removing that without a replacement guardrail is unwise.

- **Chosen design — Option A: scale cap by active-strangle count.**

  `effective_cap = base_cap × max(1, active_strangle_count_today)`

  So with `base_cap=20` on OTM: 1 strangle in play = 20-order budget,
  3 strangles = 60, 5 strangles = 100. Same for ATM family (`base_cap=40`).

  Rationale: penalises only the over-accumulation case (today's symptom)
  while preserving the original budget per strangle. From tomorrow with
  `max_strangles=1` properly enforced, normal operation never sees > 1
  strangle per variant, so this scaling never kicks in — but it's there
  as headroom if accumulation ever recurs.

- **Implementation sketch:**
  ```python
  # services/nas_executor.py, _check_guardrails (and same in nas_atm_executor.py)
  base_cap = cfg.get('max_daily_orders', 20)
  strangles_today = self.db.get_today_strangle_count()   # new helper
  effective_cap = base_cap * max(1, strangles_today)
  today_orders = self.db.get_today_order_count()
  if today_orders >= effective_cap:
      return False, (
          f"Daily order limit ({effective_cap}, scaled from base {base_cap} "
          f"× {strangles_today} strangles) reached"
      )
  ```

  The `get_today_strangle_count()` helper queries
  `SELECT COUNT(DISTINCT strangle_id) FROM <positions_table>
   WHERE date(entry_time) = date('now','localtime')` —
  cheap, indexed lookup.

- **Effort:** ~half a day. Two `_check_guardrails` callsites
  (nas_executor.py and nas_atm_executor.py — subclasses inherit).
  One new DB helper in each `*_db.py` (nas_db, nas_atm_db, nas_atm2_db,
  nas_atm4_db). Log the *effective* cap so the gag message is
  self-explanatory in journal.

- **Side note on Option B + C deferred:** Considered separating entry vs
  adjustment counters (B) and tick-rate-limiting (C). Option A chosen for
  simplicity and because it directly maps to the failure mode observed
  today (over-accumulation → too many rolls). Revisit B or C if Option A
  isn't sufficient.

- **Side note for today (operational):** Sq OTM's EOD squareoff at 15:15
  IST will also be blocked by the same cap. User is closing positions
  manually in Kite to compensate.

- **Gate:** No prerequisites. Ship after market close any day.

### NAS ATM SL — 2-tick confirmation before firing (filter bid-ask spike noise)

- **What:** Currently the ATM-family per-leg SL fires on the FIRST tick where
  `live_prem >= sl_price` ([nas_atm_executor.py:check_and_handle_sl line 358-363](services/nas_atm_executor.py#L358-L363)).
  Today's 4 SL fires on the 24000 PE breached SL by tiny margins (0.02 to 0.78
  on a ~Rs 127 premium = 0.02% to 0.6%) — well within bid-ask noise. Today
  happened to be a trending day so the breaches kept moving past SL anyway,
  but on a chop day this exact pattern would generate false exits.

  Mirror the OTM cross-leg roll's existing 2-tick debounce pattern
  ([nas_ticker.py:_check_premium_tick lines 488-548](services/nas_ticker.py#L488-L548))
  and require N consecutive ticks above SL before firing.

- **Why:** Cost of waiting 1 extra tick on a real breakout is ~Rs 0.50-1.50
  slippage per leg (~Rs 30-100 of P&L on a 65 qty NIFTY ATM leg). Benefit is
  not closing prematurely on a single-tick artifact that immediately retreats.
  On chop days the benefit dominates by an order of magnitude.

- **Design — asymmetric confirmation by detection path:**
  - **WebSocket tick path** (`_check_atm_premium_tick` in nas_ticker.py):
    add 2-tick consecutive-breach confirmation using a per-token counter
    that mirrors the existing `_adj_confirm` dict. Reset to 0 on any
    non-breaching tick.
  - **REST poll path** (`_nas_916_sl_monitor` every 10s): keep first-poll
    fire (no confirmation). This path is the safety net when the WS is dead
    — adding a wait there would delay response by 10-30s and defeat the net.

- **Scope:** ATM, ATM2, ATM4 across Sq + 916 (6 of 8 variants total). OTM
  family doesn't have per-leg SL — they already use 2-tick cross-leg roll
  separately. No change needed there.

- **Sketch:**
  ```python
  # nas_ticker.py
  self._atm_sl_confirm: Dict[int, int] = {}   # token -> consecutive breach count

  # Inside the tick callback that checks ATM SL:
  sl = info.get('sl_price')
  if sl and ltp >= sl:
      n = self._atm_sl_confirm.get(token, 0) + 1
      self._atm_sl_confirm[token] = n
      if n >= ATM_SL_CONFIRM_TICKS:   # config-driven, default 2
          ...fire SL exit (existing path)...
          self._atm_sl_confirm[token] = 0
  else:
      self._atm_sl_confirm[token] = 0
  ```

- **Effort:** ~half a day. Three callsites in nas_ticker.py (`_check_atm_premium_tick`,
  `_check_atm2_premium_tick`, `_check_atm4_premium_tick`). Add
  `ATM_SL_CONFIRM_TICKS = 2` constant in `config.py` as a tunable.

- **Side-effect to be mindful of:** For ATM2's `exit_both_on_sl`, the 1-tick
  delay also defers the CE-leg buyback. Usually this is a wash (CE moves
  opposite to PE) but worth a quick post-deploy review of net P&L impact
  across 1-2 weeks of trades.

- **Gate:** Deploy after market close any weekday. No prerequisites.

### DEPLOY AFTER 15:30 TODAY (2026-05-29) — 916-ATM4 roll-leg fix (committed, needs restart)

- **What:** Bug fix already committed to `services/nas_atm4_executor.py` — the
  1st-SL roll's `add_position()` now sets `mode='live'/'paper'` and promotes the
  new leg to `ACTIVE` immediately on successful Kite placement.
- **Why:** Today (live day 1) the 916-ATM4 1st-SL roll fired correctly (sold
  23800PE @88, order 260529150351568 COMPLETE) but the new leg was written
  `status='PENDING', mode=NULL`. The NAS-RECONCILE job only scans
  `WHERE status='PENDING' AND mode='live'`, so it never promoted it → the leg
  sat PENDING, was **unmonitored by the ticker (no SL)**, and was hidden from
  the dashboard card. A live naked short with no stop is the failure this
  prevents. Root cause: [nas_atm4_executor.py:348](services/nas_atm4_executor.py#L348)
  set PENDING in live mode and the `add_position` call omitted `mode`.
- **Today's leg was hand-fixed live:** DB row #44 (23800PE) flipped PENDING→ACTIVE
  at ~10:12 IST; ticker confirmed monitoring 2 ATM4 legs at 10:15. So today is
  safe; the code fix prevents recurrence on the next roll.
- **How to deploy (after 15:30 IST):**
  ```
  git push   # already done from laptop
  ssh arun@94.136.185.54 'cd /home/arun/quantifyd && git reset --hard origin/main && sudo systemctl restart quantifyd'
  ```
- **Scope:** ATM4-only. OTM ([nas_executor.py:204](services/nas_executor.py#L204))
  and ATM ([nas_atm_executor.py:208](services/nas_atm_executor.py#L208)) roll
  paths already set `mode`; ATM2 reuses the ATM entry path. No other variant affected.

### NAS feed watchdog goes blind every morning at open (tz-comparison bug) — fix after close

- **What:** `_latest_candle_dt()` in [services/nas_watchdog.py:95](services/nas_watchdog.py#L95)
  compares `c > latest` (completed candle date vs in-progress candle date)
  *before* the tz-normalization at [lines 105-106](services/nas_watchdog.py#L105-L106).
  At open, startup-loaded historical candles are tz-aware (+05:30) while live
  ticks build tz-naive candles → `can't compare offset-naive and offset-aware
  datetimes` is raised, the function returns `None`, and the watchdog reports
  `latest=None` / `no_candle_data`. Observed live 2026-05-29 09:18; it
  self-healed at 09:20 once live (naive) candles became `completed_candles[-1]`.
- **Why it matters:** During that blind window the watchdog cannot tell whether
  the tick→candle pipeline is actually frozen — and worse, the blind status it
  emits (`no_candle_data`) is **not** an alert trigger: [line 195](services/nas_watchdog.py#L195)
  only emails on `FROZEN`. So a genuine open-bell feed freeze (the exact MST
  failure mode this watchdog exists to catch) would go unalerted for the first
  ~5 min of every session, precisely when live NAS legs are most exposed.
- **Fix (two parts, ~15 min, backend → deploy after 15:30 IST only):**
  1. Normalize tz **before** the comparison in `_latest_candle_dt()` — strip
     tzinfo on both `cur["date"]` and `comp[-1]["date"]` (or convert both to
     naive IST) prior to the `c > latest` check at line 95, not after.
  2. Make `no_candle_data` *during market hours* an alertable state (treat it
     like `FROZEN` for the email path, or add it to the alert condition at
     line 195) so an open-bell freeze actually pages.
- **How to deploy:** after close — `git push` then
  `ssh arun@94.136.185.54 'cd /home/arun/quantifyd && git reset --hard origin/main && sudo systemctl restart quantifyd'`.
- **Discovered:** 2026-05-29, during the NAS live-go readiness check (first
  real-money session). Live legs that morning: 916-OTM 24450CE/23550PE 2L,
  916-ATM 23950CE/23950PE 1L (+ATM2/ATM4).

### NAS Tier 1 — Exchange-side SL-M (port ORB pattern; required before scaling NAS past Rs 25L)

- **What:** Today the per-leg SL on every NAS variant is checked in-memory by the
  WebSocket ticker callback; the BUY exit order is sent *after* `live_prem >= sl_price`
  trips ([nas_executor.py:137](services/nas_executor.py#L137),
  [nas_atm_executor.py:140](services/nas_atm_executor.py#L140)). If Flask, the
  ticker, or the VPS dies during an open position, the short is naked with zero
  exchange-side protection. Port the ORB pattern: after each NAS entry fills,
  immediately place an opposite BUY SL-M on Kite at the trigger price, so the
  exchange itself holds the stop.

- **Why:** Scaling NAS to all 8 systems at full size means up to 28 short NIFTY
  lots across CE+PE on a Mon/Tue/Fri. A daytime process-death plus 2% adverse
  NIFTY move = ~Rs 4-6L drawdown per affected leg with current in-memory-only
  SL. At Rs 1cr deployed this becomes survival-threatening; the exchange-side
  SL is the structural fix. ORB already runs this end-to-end —
  [orb_live_engine.py:1884-1945](services/orb_live_engine.py#L1884-L1945)
  (`_place_exchange_sl_m`) + trail-resize handler at line 1740.

- **Scope split — strategy behavior preserved:**
  - **ATM variants (6 of 8):** per-leg hard SL @ `entry × 1.30` ports directly
    as exchange BUY SL-M with trigger matched 1:1 to the Python value. Python
    check stays as the *secondary* monitor — exchange fires only when Python
    didn't (i.e. when the process is down or slow).
  - **OTM variants (2 of 8 — NAS Squeeze OTM + NAS 9:16 OTM):** the existing
    2-tick cross-leg roll adjustment in
    [nas_ticker.py:_check_premium_tick](services/nas_ticker.py#L457) lines
    488-548 **stays in Python untouched** — the exchange has no concept of
    "if leg A premium is 2× leg B premium, roll leg A." Only add a *wide
    catastrophic* BUY SL-M per leg (e.g. `entry × 3.0`) as fat-finger /
    disaster backstop. Normal roll logic is unaffected.

- **Build steps (~5-7 working days):**
  - Add `sl_order_id` column to all 8 NAS position tables (shared migration helper)
  - Wrap `_place_order` in `nas_executor.py` + `nas_atm_executor.py` +
    `nas_atm2_executor.py` + `nas_atm4_executor.py` to fire a BUY SL-M after
    entry confirmation; track `sl_order_id` in the position row
  - `_close_leg` must CANCEL the SL-M first, then send the exit BUY — and skip
    the BUY if the SL-M already filled (race condition)
  - Adjustment flow: cancel old leg's SL-M → close old → open new → place
    new SL-M (handles both ATM2 cascade re-entry and ATM4 roll-to-match)
  - ATM trail-to-cost: `kite.modify_order(trigger_price=...)` on surviving
    leg's SL-M when Python decides to trail to breakeven
  - Startup reconciliation helper — on Flask boot, read `kite.orders()` +
    `kite.positions()`, match against open NAS positions, rehydrate
    `sl_order_id`s, alert on any drift
  - Smoke test under a `LIVE_DRYRUN=1` flag against the Kite paper account
    before any real-money variant flips

- **Gate:** Required complete + 5 clean sessions with zero reconcile mismatches
  before scaling NAS aggregate capital past Rs 25L.

### NAS Tier 2 — Slippage management (limit-order entries + defined-risk wing)

- **What:** Today NAS uses `MARKET` orders for entries and SL exits
  ([nas_executor.py:148](services/nas_executor.py#L148),
  [nas_atm_executor.py:151](services/nas_atm_executor.py#L151)). Under a vol
  spike — exactly when SL fires — short-option fills can slip 5-15% of
  premium. Three independent improvements, two to build, one explicitly rejected:

  - **2a. Limit-order entries with auto-retry (recommended next build):**
    SELL at `mid_ask + small_buffer` LIMIT, cancel-and-resubmit if unfilled
    in 3-5s, fall back to MARKET after N retries. Saves ~1-3 ticks per leg
    per entry. Shared helper across all 8 variants. Independent of Tier 1
    — can build in parallel.
    Effort: ~2 days.

  - **2b. SL-L on the exchange in place of SL-M — REJECTED.** Caps slippage
    but risks not filling on a gap → naked-short into a worsening move. SL-M
    is the correct primitive for safety; slippage is the price of certainty
    of fill. Do not do this.

  - **2c. Defined-risk overlay (iron condor wings):** SELL the current
    strike + BUY a strike ~Rs 5 further OTM per leg. Converts strangles to
    iron condors; max loss capped to `(wing width × lot − net credit)`
    regardless of slippage. Costs ~Rs 3-5 of premium given up per leg.
    Requires re-backtesting the new credit profile vs current strangle
    Sharpe to verify it still pencils.
    Effort: ~5-7 days (new strike-selection logic + 4-leg order management
    + adjustment rules that respect the wings).

- **Why:** At current scale (Path A, ~Rs 24L peak) slippage is annoying but
  absorbable. At Rs 50L+ a single tail event with bad fills can wipe a month's
  premium income. 2c is the structural fix for tail risk; 2a is the easy
  quality-of-execution win that compounds across every entry.

- **Build order:**
  1. **2a first** — cheap, immediate value, no strategy change, no
     re-backtest needed
  2. **2c after** — gated by capital growth + a fresh backtest comparing
     iron-condor Sharpe against current strangle Sharpe (must still beat
     20% post-cost or it's not worth the wings)

- **Gates:**
  - 2a complete before any single NAS variant's capital exceeds Rs 5L
  - 2c complete + backtested + paper-validated before NAS aggregate capital
    exceeds Rs 50L

### NAS scale-up ladder (binding — rung N+1 requires rung N gate met)

| Rung | Capital ceiling | Prerequisite |
|---|---|---|
| 0 (today) | ~Rs 24L peak (Path A, OTM=2 lots, all 8 systems) | Current state — in-memory SL only |
| 1 | ~Rs 25L | Tier 1 (exchange SL-M) live + 5 clean sessions, zero reconcile mismatches |
| 2 | ~Rs 50L | Tier 2a (limit entries) live + watchdog hardening complete + 2 weeks clean |
| 3 | ~Rs 1cr+ | Tier 2c (iron condor wing) live and backtested; Tier 3 (process resilience) has resolved at least one real drift event |

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

### 🟡 research/43 H-pattern continuation — Phase 3 (threshold sweep / 2-min)

- **What (P1+P2 done 2026-05-29):** intraday H (pole→flag→continuation) on 5-min,
  375 stocks + NIFTY50. P1: raw pattern is breakeven-gross, net-negative after
  6 bps (best breakout|MM net −0.035R). **P2: confluence filters flip it
  net-positive — prev-day-low break is the decisive filter.** Best = **short-only
  breakout|MM + `pdl+trend`**: short-net **+0.060R**, PF 1.03, +1,130 total R over
  32.6k short trades. Verdicts:
  [results/RESULTS.md](research/43_h_pattern_continuation/results/RESULTS.md) +
  [results/RESULTS_P2.md](research/43_h_pattern_continuation/results/RESULTS_P2.md).
- **Why it's not closed:** edge is **thin (PF ~1.03)** — clears net>0 but fails the
  PF>1.2 gate. Real but marginal; a confluence overlay, not a standalone book.
- **How (next, highest value):** detection-threshold sweep (POLE_ATR, RETR_MAX,
  FLAG_MAX, WIDTH_ATR) on the **short-only `pdl+trend` breakout|MM** config — a
  tighter H definition is the likeliest way to push PF >1.2. Add short-only
  win/loss split to the tally so short-book PF is computed exactly (P2 only has
  combined PF).
- **Blocked sub-item:** 2-min entry refinement — **no 2-min data in DB**; needs a
  Kite 2-min download (VPS-only per project rule) before it can be tested.
- **Decision pending from Arun:** pursue Phase 3, or shelve (thin edge)?

### ⚪ research/44 Prev-day range-compression breakout — CONCLUDED (uninvestible standalone)

- **Outcome (P1–P4 done 2026-05-29):** real long high-beta SIGNAL (causal PF ~1.3
  with 20-DMA regime gate) but **uninvestible as a standalone book** — single-factor
  long-beta, −37% MaxDD / Sharpe ~0.4 even regime-filtered. Arun's call: shelve as a
  standalone. Keep the signal for possible use as a HEDGED/DIVERSIFIED sleeve or a
  discretionary long scanner. Verdicts: results/RESULTS*.md. **No further work unless
  revived as a sleeve.**

### 🟢 research/45 Daily anomaly scan — DONE; found a strong edge to build

- **Outcome (2026-05-29):** broad cross-sectional scan of 6 classic anomalies on
  378 liquid names, daily 2000–2026. **Short-term reversal is the standout**:
  perfectly monotonic deciles, LS −0.59%/5d, **t ≈ −9.78**, and naturally
  market-neutral (long losers / short winners) → structurally avoids the correlated
  drawdowns that sank research/43–44. Turn-of-month is a strong simple overlay
  (+0.28%/day on ToM days). Overnight drift huge but hard to trade. Momentum/low-vol/
  52wk-high add little. Verdict: [results/RESULTS.md](research/45_daily_anomaly_scan/results/RESULTS.md).
- **Next → research/46 (recommended build):** short-term mean-reversion, long-short,
  market-neutral. Must test holding period (5/10/21d), quantile width, rank-weighting
  (turnover), and a realistic cost model; add turn-of-month timing overlay. Goal: a
  high-Sharpe, low-DD market-neutral book — the opposite profile of the shelved
  directional systems.

### ⚪ research/46 Short-term reversal long-short — CONCLUDED (cost-bound, not investable)

- **Outcome (2026-05-29):** market-neutral reversal (long losers/short winners),
  daily, 378 names. Strong GROSS (Sharpe ~1.1 near-zero cost) but **break-even
  ~13 bps/side**; at realistic 10 bps only Sharpe 0.36 / +4%/yr (2018+), −24% MaxDD,
  negative in 3 of last 5 yrs (decayed anomaly). Shorting Indian cash overnight is
  structurally hard (SLB/F&O-only). **Not investable standalone.** Verdict:
  [results/RESULTS.md](research/46_st_reversal_longshort/results/RESULTS.md).
- **Synthesis (research/43–46):** net-of-cost high-frequency alpha is scarce in this
  universe — directional intraday dies to cost+correlation; robust cross-sectional
  anomalies are eaten by turnover (reversal) or already harvested (momentum=MQ).
  **Recommended pivot: improve the MQ winner (already 32–48% CAGR) — e.g. a
  market-regime overlay to cut its 27% MaxDD — rather than hunt new HF edges.**

### 🟢 research/47 Regime risk-off options/ETF overlay — DONE (directional)

- **Outcome (2026-05-29):** tested replacing research/41's risk-off "go to cash" with
  options/ETF structures (long put, put/call debit spreads, short put, covered-call
  Nifty ETF) vs cash & short-futures baselines, modeled IV (no historical options
  data → directional). **Cash is near-optimal (Calmar 2.33)** — the gate fires when
  the market is weak, so staying long it loses. **Covered-call ETF is DOMINATED by
  cash** (−22.9% DD vs −15.1%). SHORT futures worst (2025 whipsaw −21%). **Only
  enhancement first thought worth keeping (call-spread re-entry) FAILED the all-years
  test**: across 47 risk-off months it wins only 40%, nets +0.5% total (~0/mo), helps
  7 yrs / hurts 6 (coin flip), and slightly lowers Calmar (2.09 vs cash 2.33). **Final:
  keep PLAIN CASH; no options/ETF overlay reliably beats it** (gate's value is being
  flat; directional overlays add variance for ~0 mean). Verdict + charts:
  [results/RESULTS.md](research/47_regime_options_overlay/results/RESULTS.md).
- **To make decision-grade:** need real Nifty options/IV history. The live
  `option_chain` collector (options_data.db) is already running — ~1yr of it makes
  this testable on real premiums. Optional: add call-spread re-entry to the research/41
  live spec as a small overlay.

### (archived plan) research/44 Phase 3/4 detail

- **What (P1+P2 done 2026-05-29):** prev-day narrow-range → next-day 5-min close
  beyond the range → trend-trade (swing). 379 syms. P1 raw = breakeven (PF 1.003).
  **P2 filters lift it — high-beta is the dominant filter, HTF-trend stacks, and
  LONG-ONLY clears PF>1.2.** Best = **long-only, high-beta (β≥1.2), NR7 prev day,
  prev close>SMA50, break above prev-day high, stop=prev-day low (BOX), 2–3R,
  swing ~3d: PF ~1.24, +0.16R/trade, ~1,700 trades.** PCT25 variant = ~2,900 trades
  at PF ~1.15. Verdict: [results/RESULTS.md](research/44_prevday_range_breakout/results/RESULTS.md).
- **Why it's the best of 3:** beats the H-pattern (PF ~1.09 short-only) and clears
  the PF>1.2 gate long-only. Economically sound (long momentum + market drift).
- **Phase 3 DONE 2026-05-29:** causal trailing-beta (vs synthetic equal-weight
  market index — NIFTY50 daily only starts 2023) **confirms the edge**, monotonic
  in beta (β≥1.4 → PF ~1.29) → real alpha, not look-ahead. **BUT** (a) broad 5-min
  history mostly starts ~2024 so the robust window is thin/bullish; loses in
  pullback years (2021 PF 0.31, 2026 0.74); (b) **naive portfolio is uninvestable
  — 55–87% MaxDD** (one correlated long-beta bet; lower concurrency was worse).
  Verdict: [results/RESULTS_P3.md](research/44_prevday_range_breakout/results/RESULTS_P3.md).
- **How (Phase 4, what would make it real):** (1) **market-regime filter** (only
  long when index > 50-DMA / breadth positive) — the loss years are market-weak,
  so this should both lift PF and slash the DD; (2) de-correlate (one-per-sector /
  gross-exposure cap / hedge with the research/43 short H-pattern); (3) **download
  more 5-min history** (Kite, VPS-only) to test pre-2024; (4) gap-risk control on
  swing holds.
- **Decision pending from Arun:** run Phase 4 (regime filter is the high-value
  test), or shelve as a known signal.

---

## In Progress

_(Move items here when work begins. Include start date.)_

---

## Done — recent (most recent first; trim quarterly)

### 2026-06-03 — NAS paper-days + NAS-OPT paper variant + research/54 (COMPLETE)

**Operating schedule (commit 188b145):** NAS live orders only Mon/Tue/Fri; every other day runs
as PAPER (signals+DB+P&L, no real orders), mode-tagged. DTE gate turned OFF operationally (now a
backtest-study question). config: `live_weekdays=(0,1,4)`, `max_dte_at_entry=None` on both base
dicts; `skip_weekdays` emptied; executors `_place_order`/exit day-aware (`_is_paper`). Deployed
live mid-market (all flat, user-cleared).

**NAS-OPT new paper variant (commits 188b145 backend + 4061e54 dashboard):** research/54's refined
system — 0/1-DTE ~100pt-OTM strangle, 09:20 entry, ±0.4% underlying-move stop (one-and-done), 14:45
exit. `services/nas_opt.py` reads the live options recorder (live == backtest), **paper-only (NO
Kite-order code)**. `register()` adds `/api/nas-opt/state|trades|equity` + entry/monitor/exit jobs.
Dashboard card on `/app/nas` (P&L, win rate, SVG equity curve, today status). `nas_opt_trading.db`
backfilled with 13 backtest trades (+₹20,409, 69% win, maxDD −2,695). RUNNING IN PAPER; first entry
next Mon/Tue.

**research/54 NAS new-systems search (commit d362829, verdict CONCLUDED):** 6 angles tested on the
real recorded NIFTY chain — IV-filter, iron-flies, late-entry, re-entry, directional-skew,
calm-classifier all KILLED; only **~100pt-OTM strangle + move-stop** survived (beats ATM straddle on
net+tail, monotonic). Re-entry HURTS (move-stop one-and-done). Mon(1DTE)+2284/day, Tue(0)+395,
Fri(4)−70 flat, Wed/Thu bleed → Mon/Tue/Fri-live is data-consistent. Report:
`research/54_nas_tune_newsys/results/RESULTS.md`.

**Two follow-ups PARKED (build only when user asks):**
1. **Move-stop upgrade to the LIVE 8 variants** — replace per-leg 1.3× premium stop (whipsaws,
   −₹13,983 on chain) with ±0.4% underlying-move stop (lifts net + bounds tail; 2yr stress −7.9k vs
   no-stop −58.8k). Money-path ticker change (`nas_ticker.py` SL handlers + `entry_spot` capture);
   designed, deploy-later, paper-first. Pair with ~100pt-OTM strikes (research/54 stage4).
2. **Flip NAS-OPT to LIVE** — NOT a toggle; `nas_opt.py` is paper-only by design. Add the Kite
   execution path (orders on entry+exit) behind a paper/live flag + fill read-back + kill switch.
   Build only when user says NAS-OPT paper is working well.


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
