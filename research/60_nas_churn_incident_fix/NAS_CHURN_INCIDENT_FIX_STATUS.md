# NAS 23200 Churn — Live Incident, Root Cause & Guardrail Fix (2026-06-08)

STATUS: FIX WRITTEN + OFFLINE-TESTED — deploy after 15:30 close (no mid-market restart)

## The Ask
"system is taking lot of trades buying/selling 23200 ce and pe continuously … fix and
test comprehensively, we shud not hv any such operational bugs in live."

## Incident (Mon 2026-06-08, ~09:30–11:10 IST, REAL MONEY)
6 NAS ATM variants armed live churned the 23200 straddle. 45 orders (31 on 23200).
Contained: kill-switch (squared 2) + master-mode OFF + freeze flag + cancelled pending;
3 MIS legs re-armed with 30% SL-M stops; NRML legs left manual. Net day P&L ≈ −₹38k
(incl. a separate non-NAS book −₹27k flagged to user).

## Root cause (evidence-grounded)
- **Order tape:** clockwork close-then-reopen of the full 23200 straddle every 5-min
  candle (BUY both at xx:x0:00, SELL both at xx:x0:13), near-breakeven → brokerage/slippage bleed.
- **Trade DBs:** `nas_atm` = 12 positions, **5 trades all `exit_reason=ST_EXIT`**.
- **THE BUG:** `_check_guardrails` re-entry cap counted only `SL_HIT`:
  `sl_reentries = len([t for t in today_trades if t.get('exit_reason')=='SL_HIT'])`.
  The churn exited via **ST_EXIT** (surviving-leg Supertrend trail) → counter stayed 0 →
  `max_reentries` never tripped → re-entry allowed every candle.
- **Loop:** pinned market (spot ~23200) → ATR squeeze present every candle → scan re-enters
  whenever flat; the ST(7,2) survivor-trail whipsaws on flat premium → ST_EXIT → flat →
  re-enter next candle. **No re-entry cooldown** to break it.
- **Amplifier:** "1 active strangle" guard is PER-VARIANT (per-DB) — 6 ATM variants converge
  on the identical ATM strike with zero cross-variant coordination → exposure/churn stack ~6×.

## Fix (this change)
`services/nas_atm_executor._check_guardrails` (inherited by atm2/atm4 + used by 916_atm*):
1. **Count ALL completed strangle cycles today** (distinct `strangle_id`), not just SL_HIT,
   so `max_reentries` actually caps re-entries regardless of exit reason.
2. **Re-entry cooldown** (`reentry_cooldown_min`, default **15**): block a new strangle within
   N min of the last exit → kills the per-candle loop. 0 = disabled.
config.py: set `reentry_cooldown_min=15` on all 6 ATM variant dicts.

## Tested (offline, `scripts/test_churn_guardrail.py`, stub DB — no live/Kite/restart)
8 guardrail scenarios + subclass-inheritance check (see results/). Replays the exact churn
(5 ST_EXIT cycles) and asserts it is now BLOCKED; proves the old SL_HIT-only counter would
have allowed it; verifies legit re-entry still passes, cap counts mixed exits, cooldown
boundary, and atm2/atm4 inherit the fix.

## Deploy
Markets open (frozen book, safe). Code committed now; takes effect on next restart.
**Deploy after 15:30 IST close** (NO mid-market restart). On deploy: keep master OFF +
freeze until the user re-arms deliberately. Recommended alongside (user decisions, NOT in
this code change): run ONE live ATM variant (not 6 — they're redundant by construction),
and make squeeze entry edge-triggered (new-squeeze) rather than persistence-triggered.

## Files
| File | Purpose |
|---|---|
| `NAS_CHURN_INCIDENT_FIX_STATUS.md` | this doc |
| `scripts/test_churn_guardrail.py` | offline guardrail test |
| `results/test_output.txt` | test run output |
