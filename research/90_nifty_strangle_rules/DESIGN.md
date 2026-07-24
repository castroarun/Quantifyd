# research/90 — NIFTY Rules-Based Short Strangle ("NSR") — G0 Design

**Date:** 2026-07-24 · **Stage:** G0 (idea/design — no compute spent yet)
**Origin:** W30 mentor review (`mentor/reviews/2026-W30.md`). Arun's manual NIFTY
monthly strangle management cost ≈₹6k vs untouched + left a worse book; root habit
= calm-day credit-chasing rolls toward spot. Goal: replace manual management with
mechanical rules so emotions are structurally excluded.

## Objective (stated by Arun, 2026-07-24)

> "Automate strangles rules-based so emotions are left out… aim is not to be 100%
> profitable, but be consistent, minimize losses, control emotions."

**Success is defined as risk-shape, not alpha:** match-or-beat Arun's manual
strangle results with (a) bounded per-trade losses, (b) zero interventions,
(c) margin headroom always ≥30%. Alpha, if any, is a bonus.

## Honest priors (house research — stated up front, unbiased)

| Prior | Source | Implication for NSR |
|---|---|---|
| Index short-vol edge DECAYED ≈0 post-2022 | research/89 (real bhav data) | Expect modest/near-zero raw EV. The win is consistency + removing the manual-management drag, which W30 measured as real and negative. |
| Management (take-25/50% profit) improves risk-shape, not EV | research/89 | Profit-take + stop rules are legitimate objectives here. |
| ±500-pt wings on positional NIFTY fly were net-positive 7yr | research/60 (AlgoTest) | Defined-risk variant (iron condor) must be a grid arm — also solves the 97%-margin problem (wings cut margin ~60-70%). |
| Weekly CPR NARROW → trend week (hostile to short premium); DAILY narrow → calm day | research/67 | CPR gates are the requested entry filter. Test BOTH signs anyway (unbiased; r/67 also logged a sign-flip trap). |
| Expiry-day theta only EV+ at VIX<16 | research/59 | VIX gate arm. |
| ±0.4% underlying move-stop validated (intraday) | research/52/54/68 | Test a daily-close analog for positional (e.g., close beyond short strike − buffer). |
| PANIC gauge tiers (V15/DDhi/Z15) | research/70 | Catastrophic flatten trigger. |
| Options backtests MUST filter real traded volume/OI | research/89 (binding) | NIFTY monthlies/weeklies are liquid, but far strikes pre-2019 may not be — filter anyway. |
| One-and-done: re-entry after stop hurts | research/54 | No same-cycle re-entry arm default. |

## The base (locked before any backtest)

- **Underlying:** NIFTY only (Arun's decision — no stocks).
- **Structure:** short strangle; grid arm for iron condor (buy wings).
- **Sizing:** FIXED lots, never increased intra-trade. No martingale by construction.
- **Entry timing:** fixed day/time (e.g., first trading day after monthly expiry,
  or Monday 09:45 post-opening-range) — no discretion.
- **Strike selection:** fixed rule — %OTM (2.0/2.5/3.0%) or delta bucket (10/15/20Δ)
  via bhav-derived IV.
- **Exits (all pre-committed at entry, first-hit wins):**
  1. Profit-take at X% of credit (grid: 40/50/60)
  2. Per-leg premium stop at Y× credit (grid: 1.5/2.0/2.5)
  3. Giveback stop: open profit retraces 50% from peak
  4. Time exit: DTE ≤ 2 (avoid gamma days; r/74: expiry 15:15 is the worst minute)
  5. PANIC tier ≥ SPIKE → flatten (catastrophic override)
- **Adjustment arms (tested head-to-head, honestly):**
  - A0: NONE — stops only (the W30 counterfactual winner)
  - A1: wing-off on threat (convert to defined-risk, keep strikes)
  - A2: single roll AWAY from spot only, once per cycle, threatened side only
  - (Rolling TOWARD spot is excluded by construction — it is the documented leak.)
- **Entry gates (grid):** weekly CPR width percentile (both signs tested),
  VIX level/percentile, none.
- **Success criterion:** net-of-cost expectancy per trade + max-loss distribution +
  yearly stability, ranked by Calmar-style ratio; gates per playbook
  (per-year positivity, parameter monotonicity, cost sensitivity).

## Data plan

| Need | Source | Coverage |
|---|---|---|
| Daily option prices + IV + volume/OI | `nse_options_bhav` (market_data.db, 30.3M rows) | 2016 → now |
| NIFTY spot 5-min (CPR, PANIC, gaps) | market_data_unified (r/81 collection, token 256265) | 2015 → now |
| VIX daily | Kite historical (token 264969) | ongoing |
| Intraday chain validation (recent) | options_data.db recorder | 2026-04-20 → now |
| Arun's manual baseline | `mentor/` daily capture + tradebooks | 2026-07-24 → now |

Daily-granularity simulation first (stops evaluated on daily highs/lows of leg
premium — conservative both-touch handling documented). Intraday refinement only
if G1 passes.

## Gate plan

- **G1:** daily-granularity sweep 2019→2026 (emphasis post-2022, the no-edge
  regime — if it only works pre-2022, it's dead). Kill if: no config family with
  net-positive post-2022 expectancy AND acceptable max-loss tail.
- **G2:** robustness — per-year table, parameter sensitivity (monotonic > peak),
  cost ×1.5/×2, both-touch pessimistic fills.
- **G4:** tearsheet + publish study to `/app/backtest/...`.
- **G5:** PAPER BOOK — new variant alongside existing straddle V1/V2 paper books
  (`services/` straddle infra, `/app/straddles`), auto-managed on VPS.
  **Weekly mentor review then compares Arun's manual week vs the robot's week —
  this comparison IS the emotional training tool.**
- **G6:** only after paper soak: live with small size, kill-switch, margin cap.

Even if G1 says NO EDGE (plausible per r/89): the fallback deliverable is still
valuable — a **mechanical guardrail overlay for Arun's manual trades** (GTT premium
stops at entry, roll-direction rule, margin cap alerts via the mentor cron),
because W30 proved the manual management drag is real money.

## Open questions (block G1 launch)

1. Expiry focus: monthly (his current book), weekly, or both as grid arms?
2. Baseline size for eventual paper book: 10 lots (₹6.5k/pt) as current?
3. Confirm G1 launch on VPS (runner + STATUS-MD per convention:
   `NIFTY_STRANGLE_RULES_DAILY_SWEEP_STATUS.md`).
