# V2 Executor + Inside-Week Breakout Sleeve — Locked Build Spec (2026-06-10)

Two paper-first strategies, mirroring the NAS live design (force_paper hatch, master-mode JSON,
watchdog, `/api/<key>/kill-switch`, `/app` route, SSE). Backend deploys after 15:30 IST only.

## 1. V2 short iron fly (the main book)
- **Structure:** SELL ATM CE + SELL ATM PE, BUY ATM+500 CE + BUY ATM−500 PE (symmetric short fly,
  2.0%-of-ATM wings). 2nd-nearest weekly, positional / overnight carry.
- **Entry:** 09:20, ~8 DTE; roll 1 TD before expiry, re-enter.
- **Exits:** 2.0% underlying move-stop, OR +40% credit profit target, OR roll at DTE≤1.
- **Entry gates (ALL must pass):**
  - India VIX ≥ 13 (locked floor).
  - **Combo skip-filter (LIVE):** SKIP entry if prior-day daily CPR width < 0.10% OR last completed
    week was an inside week. (In-sample Calmar 2.00; walk-forward validated; asymmetric-safe.)
  - Shadow-log every would-be-skip with its reason (forward validation ledger).
- **Sizing:** 10 lots = qty 650. Margin ≈ ₹9.6L (₹95.8k/lot).
- **Deploy:** PAPER-first, SHORT (~2-4wk) confirm window verifying CPR + inside-week computation
  day-by-day vs backtest, then promote to live.

## 2. Inside-week breakout sleeve (NEW — paper-only experiment)
Fires ONLY on weeks the V2 filter skipped *because of an inside week* (compression → expansion).
- **Trigger (locked):** first daily CLOSE in the trade week beyond the prior (inside) week's range:
  - close > prior-week HIGH → **UP-break (bullish)**
  - close < prior-week LOW → **DOWN-break (bearish)**
  - no break by week-end → no trade (~20% of cases).
- **Structure per side (evidence-matched synthesis; overridable):**
  - **UP-break → CALL DEBIT SPREAD** (long-gamma, captures the runner): BUY ATM CE, SELL ATM+500 CE.
    Rationale: bull up-break has a real continuation edge (78%, 1.77% MFE; 57-70% reach profit size).
  - **DOWN-break → BROKEN-WING IRON FLY skewed down** (credit, premium + tightly-capped risk; NO
    runner edge to capture): SELL ATM CE + SELL ATM PE, BUY ATM−300 PE (near, tight down cover),
    BUY ATM+700 CE (far). Net credit, slight short-delta lean.
- **Why asymmetric:** G0 probe — bull up-break continues 78%; bear down-break only 43% (NSE up-drift +
  oversold-bounce kill it). Bear side has NO validated edge → it stays PAPER, half-conviction, killable.
- **Status:** brand-new, G0-only (bull) / no-edge (bear). **PAPER-ONLY** until forward data earns it.
  Logs taken trades + outcomes; never risks real money in this sleeve at launch.

## Architecture (mirror NAS)
| Piece | File | Notes |
|---|---|---|
| Pure signals | `services/v2_breakout_signals.py` | inside-week, prior-day CPR, combo skip, breakout trigger, leg builders. Unit-testable, no Kite. |
| Executor | `services/v2_ironfly_executor.py` | basket build, paper/live (`force_paper`+`live_weekdays`), margin via `kite.basket_order_margins`, stop/PT/roll, state. |
| State/DB | `backtest_data/v2_ironfly_state.json` (+ small table) | open legs, closed-today, daily P&L, shadow-skip log. |
| API | `app.py` `/api/v2-ironfly/*`, `/api/v2-breakout/*` | state, stream(SSE), mtm, kill-switch, master-mode. |
| Scheduler | app.py APScheduler | 09:20 entry check, periodic exit/roll check, EOD snapshot. |
| Watchdog | extend `nas_watchdog.py` pattern | daily-loss thresholds, open-leg count, margin. |
| Frontend | `frontend/src/pages/Straddles.tsx` V2 card → `/api/v2-ironfly/state` | replace the paper-logger card with the executor state. |

## Build order
1. `v2_breakout_signals.py` (pure, testable) ← **building now** + smoke test vs market_data.db.
2. `v2_ironfly_executor.py` (Kite plumbing, paper/live, state).
3. API routes + scheduler + master-mode/force_paper wiring.
4. Frontend card wiring + watchdog.
5. Deploy after 15:30 IST; paper soak; promote per plan.
