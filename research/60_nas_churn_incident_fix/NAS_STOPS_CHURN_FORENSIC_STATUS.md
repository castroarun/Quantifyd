# NAS Stops / Churn Malfunction — 2026-06-12 Live Incident, Forensic & Fixes

STATUS: RESOLVED · ALL 5 FIXES DEPLOYED + VERIFIED (after 15:30 close 2026-06-12) · STILL FROZEN pending user OK to clear for Mon · Guardian agent built (`/nas-guardian`)

## The Ask
"Stop the code from further executions immediately, I'll manage the trades for the
day. Identify all the issues reported and fix them once and for all."

## What happened (Fri 2026-06-12, ~13:30–13:48 IST, REAL MONEY, all 6 NAS variants live)
NIFTY ran sharply from ~23325 to ~23530. Protective stops malfunctioned:
- Squeeze ATM2/ATM4 23350 CE went 30+ pts PAST their SL (213→228 vs SL 194) and did
  NOT exit. 916-ATM 23450 CE survivor's ST trail did NOT fire. Sq-ATM churned
  (exit a straddle, re-short the same strike ~26s later). Book swung +₹6k → −₹9k.
- Contained 13:48 via **manual-freeze flag** (set_frozen) — blocks ALL code orders
  (entries/exits/adjust/EOD), leaves positions OPEN for manual mgmt. Verified 0
  orders since. Freeze PERSISTS across restarts → system stays dark until cleared.
- User is squaring the open legs manually. EOD auto-square is also blocked by freeze.

## Root causes (forensic, code-cited)
1. **SL detected but not executed.** Ticker logs `SL TICK! ltp>=SL` → calls
   `executor.check_and_handle_sl(positions, live_ltps)`. Executor did
   `live_prem = live_ltps.get(tsym); if live_prem is None: continue` — silently
   SKIPPED the breached leg when the ticker's live_ltps lacked it (subscription
   reshuffle, #4). Reported "no actions taken (threshold not met)". The trigger leg
   never got the REST fetch the *other* legs get.
2. **ST survivor exit not firing.** `_check_atm_st_exit` runs ONLY on 5-min candle
   close, fires only on SuperTrend FLIP (direction==1), and RE-SEEDS 60 fresh
   candles on every roll/re-entry → resets flip detection → premium already above
   the fresh ST never triggers.
3. **Churn.** 15-min `reentry_cooldown_min` not gating the squeeze (ticker
   `has_squeeze`) ST_EXIT→re-enter path → re-shorts same strike ~26s after exit.
4. **Subscription reshuffle.** Re-entering one variant re-subscribes "N legs
   (Sq+916)" and momentarily DROPS other variants' legs → tick-gap → feeds #1.

Common thread: squeeze variants are ticker-ONLY (no 10s poll backstop like 916).

## Fixes — status (all on disk, NONE deployed — deploy after 15:30, frozen until verified)
- [x] **#1a SL-skip** — fetch leg premium via scanner before `continue` (3 executors). `0dbb1d0`.
- [x] **#1b squeeze SL poll backstop** — `_nas_squeeze_sl_monitor` 10s REST poll for
      nas_atm/atm2/atm4 (model on `_nas_916_sl_monitor`). `9c8a63a`. THE key robustness fix.
- [x] **#3 churn cooldown** — ROOT CAUSE = `_pt` date formats didn't match
      `datetime.isoformat()` (T-sep + microseconds) → cooldown blind. Added
      `%Y-%m-%dT%H:%M:%S.%f`. VERIFIED blocks. `9c8a63a`. (06-08 unit test used a
      space-format ts → false pass.)
- [x] **#2 ST exit (complete)** — (a) level-breach `direction==1 or latest_close>st_val`
      (`a963de7`); (b) tick-level: cache `st_val` on each candle-close compute, check it on
      every tick in the `else` branch of `_update_atm{,4}_naked_candle`, exit immediately if
      live premium > cached ST, clear cache on exit so a fresh survivor never uses a stale
      value (`0856fea`). 999999 sentinel preserved. ATM + ATM4. Worst-case lag now seconds.
- [x] **#4 additive subscription** — same-strike legs share ONE token; old exclusion sets
      were triangular so an earlier variant could unsubscribe a later variant's live leg.
      Replaced all exclusions with `_tokens_in_use_by_others()` (excludes ALL sibling maps,
      order-independent). `af05acd`. No variant can drop a sibling's live leg on re-subscribe.

## ALL 5 FIXES DONE (on disk, pushed, NOT deployed). Next: deploy after 15:30 close (freeze stays ON) -> verify SL/ST/cooldown/subscription on a controlled re-arm -> THEN clear freeze.

## Crash recovery (resume without context)
- Freeze flag: `backtest_data/nas_manual_freeze.flag` (present = frozen). Clear ONLY
  after all fixes verified: `python3 -c "from services.nas_kill_switch import clear_frozen; clear_frozen()"`.
- Live snapshot: `./venv/bin/python3 nas_live_snapshot.py` (realized+unrealized; cross-check Kite).
- Deploy after 15:30: `git pull` on VPS → `sudo systemctl restart quantifyd` (freeze stays on).
- Backups: each patched file has `.bak_fix1` etc.

## Files
| File | Purpose |
|---|---|
| this doc | forensic + fix tracker |
| scripts/fix1_sl_skip.py | fix #1a patcher |
| services/nas_atm{,2,4}_executor.py | check_and_handle_sl (SL logic) |
| services/nas_ticker.py | SL handlers, ST exits, subscriptions |
