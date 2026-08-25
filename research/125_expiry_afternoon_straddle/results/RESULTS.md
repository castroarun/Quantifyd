# research/125 RESULTS — Expiry-Afternoon Straddle (DTE0 fine time-grid)

**Verdict: SIGNAL (strong, in-sample, n=17/venue). Not yet a STRATEGY — needs paper forward-validation.**

Run 2026-08-25 ~12:00 IST on all 17 NIFTY Tuesdays + 17 SENSEX Thursdays of real ~3-sec
chain data (2026-04-20 onward), spot-ATM straddles, dwell combined-SL, r/123 net costs
(NIFTY 10L basis Rs2,500/rt; SENSEX 5L basis Rs800/rt). Today (25-Aug) excluded.

## The user's old AlgoTest slot (13:45->15:00) is REAL but SUB-OPTIMAL here
Best variant: NIFTY SL40 +Rs320/lot/day 65% ratio 3.1; SENSEX SL20 +Rs635/lot 59% ratio 2.5.
Positive every SL, so the idea survives venue+instrument+era transfer — but ranks far
below the shifted slot.

## Winner: enter ~13:15-13:30, exit 14:30-15:00 — BEFORE the last-hour gamma storm
- NIFTY DTE0: **13:15->14:30, SL30+** (SL30/40/none identical): +Rs800/lot/day, 82% win,
  dd -Rs8,675 @10L, ratio 15.7. Extending to 14:45: +Rs899/lot, ratio 14.0.
- SENSEX DTE0: safest **13:30->14:15 SLnone**: +Rs574/lot, 82%, dd -Rs2,045 @5L, ratio 23.9;
  most rupees **13:30->15:00 SL30**: +Rs1,186/lot, 88%, ratio 9.4.

## Why (calm map, mean |1-min move| as % of 13:00 premium)
Calmest pocket 12:45-13:15 (~1.4%%). Variability rises 3x by 14:45 and peaks ~15:00
(NIFTY 5.9%%/min with POSITIVE drift +0.5%%/min = the r/74 late IV pop). The last 45
minutes are where the AlgoTest slot bleeds its edge; exiting by ~14:30-14:45 keeps the
decay and skips the storm.

## Caveats
- n=17 per venue, one era (Apr-Aug 2026), in-sample grid pick (multiple testing across
  ~350 cells/venue — the top cell is optimistic; neighbors agree though: 13:15/13:30
  entries dominate the board, which is the robustness signal that matters).
- Cost sensitivity: winners survive 1.0-pt slippage (margin of ~Rs1,300/day NIFTY).
- Overlaps: TB2 paper book already trades NIFTY DTE0 13:00->14:00 SL25 — its live-paper
  days are the natural forward validation of this family.

## Recommended next step
Paper the two winner cells (NIFTY Tue 13:15->14:30 SL30; SENSEX Thu 13:30->15:00 SL30)
as config cells on a paper book for 4 expiries, then review vs this model. Today's
(25-Aug) trade, if taken, is manual — executor plans froze at 09:12.

