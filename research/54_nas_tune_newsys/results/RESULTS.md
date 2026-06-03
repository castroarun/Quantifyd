# research/54 — NAS Tuning + New Systems: VERDICT

**Verdict label: CONCLUDED** — the path to better NAS systems is **day-selection + stop design**,
not new structures or vol filters. The single actionable upgrade is the **±0.4% underlying-move stop**.

## What was tested (all on the real recorded NIFTY chain, 29 days, net of ₹80/leg)

| Stage | Angle | Result | Label |
|---|---|---|---|
| 1 | **IV-level filter** ("sell rich vol") | corr(IV,P&L)=+0.41 all-days but **−0.14 within 1-DTE** → pure DTE proxy | **NO EDGE** (killed) |
| 2 | **Weekday × DTE map** | Mon(1-DTE) +2,284/day; Tue(0) +395; Fri(4) −70 flat; Wed(6)/Thu(5) bleed | confirms 1-DTE edge |
| 3 | **Defined-risk iron-flies** | cost premium, cut edge to ~0, far wings don't cap intraday tail (−20k) | **NO EDGE** (killed) |

## The winning configuration (real chain, 0/1-DTE)
**Naked ATM straddle + ±0.4% underlying-move stop**: +₹1,412/day, worst day −₹3,260, worst-3 −₹1,850.
- vs no-stop: +705/day but worst −20,284 (unbounded tail).
- vs iron-fly+stop: +586/day (wings just bleed premium; stop already caps the tail).

## TOP ACTIONABLE FINDING
**Replace the live systems' per-leg 1.3× premium stop with a ±0.4% underlying-move stop.**
- Premium stops whipsaw on premium noise: scan shows 1.3× = −₹13,983 vs move-stop positive on the
  same chain. Move-stop triggers on REAL adverse moves → no whipsaw AND bounded tail.
- Validated beyond 29 days: research/52 stress over 2 yrs real NIFTY paths — move-0.4% tightest
  tail −₹7.9k vs no-stop −₹58.8k.
- This is a code change to `nas_executor` / `nas_atm_executor` (today: per-leg 1.3× + naked ST).
  Build + backtest + after-close deploy as its own item.

## Schedule finding (validates the user's directive)
Mon/Tue/Fri-LIVE is data-consistent: it trades Monday (the edge) + Tuesday (positive) + Friday
(breakeven) and excludes Wed/Thu (the two real bleeders) — which now run as paper. (Earlier
"Friday bleeds" call was wrong; Friday/4-DTE is flat, not a bleeder.)

## Open / optional
- Stage 4 (multi-year calm-day classifier, stacked range-day predictor, OOS) — not run; lower
  priority since 1-DTE/Monday is already the clean edge.
- New underlyings (BANKNIFTY/SENSEX recorded 29d each) — out of stated scope (NIFTY options).
- Forward validation accrues now that paper-all-days is enabled (deploy 2026-06-03).

## Caveat (standing)
29 days = SIGNAL, single regime (Apr–Jun 2026), 5-7 obs per weekday cell, no bid/ask slippage.
The DTE gradient + move-stop tail control replicate across research/51/52 independent data → the
DIRECTION is trustworthy; absolute ₹/day will move as the recorder grows.


## Stage 4 addendum (NEW keeper)
**~100pt-OTM strangle + ±0.4% move stop BEATS ATM straddle** (0/1-DTE): monotonic ATM +1,412 -> 1-OTM +1,478 -> 2-OTM +1,570/day, worst -3,260 -> -2,695. Entry 09:20 best (11:00 = -1,445/day, 13:00 = +621). Action: pair the move-stop upgrade with ~100pt-OTM strikes, 09:20 entry. 13-obs SIGNAL.

## FINAL synthesis (6 new angles tested)
| angle | verdict |
|---|---|
| IV-level filter | NO EDGE (DTE proxy) |
| iron-flies | NO EDGE (premium drag) |
| late entry 11:00/13:00 | NO EDGE (09:20 best) |
| intraday re-entry | HURTS (re-sells into trend; one-and-done) |
| directional skew | neutral (drop) |
| ~100pt-OTM strangle + move-stop | KEEPER (beats ATM straddle on net+tail, monotonic) |
| multi-feature calm classifier | NO better than opening-range alone (prior-day feats useless) |

**FINAL refined system:** 1-DTE - ~100pt-OTM strangle - 09:20 entry - +/-0.4% move-stop - ONE-AND-DONE - tight-opening-range(calm) days - exit 14:45 - cross-family. The edge is day-selection + stop + modest-OTM strikes; NOT new structures, vol filters, re-entry, skew, or complex classifiers. Implementation lever = the move-stop upgrade (TODO, money-path build, deploy later, paper-first).