# research/113 — ATM4 Roll-Leg Stop Calibration — RESULTS

**Verdict: SIGNAL (mechanic upgrade).** On 81 days of real 1-min NIFTY chain
(2026-04-21 → 2026-08-14, 63 actual roll events), the live rule — rolled-leg SL =
1.3 × roll_premium — is confirmed as the churniest variant tested: **32% of rolled legs
re-stop (6% within 15 minutes)**, exactly the failure Arun watched live on 2026-08-18
(roll @12.1, SL 15.7, dead in 6 min). Loosening the rolled leg's stop recovers real
money without fattening the tail.

## Ranking (net ₹/lot(65), 81 days, slippage 0.5pt/leg-side + ₹30/leg-side)

| Variant | Rolled-leg stop | Total | Mean/day | p05 | Win% | Restop% |
|---|---|---|---|---|---|---|
| **P200** | 2.0 × roll_prem | **+160,106** | 1,977 | −1,695 | 84% | 8% |
| P150 | 1.5 × roll_prem | +154,403 | 1,906 | −1,591 | 83% | 21% |
| NOSL | none (EOD only) | +150,850 | 1,862 | −1,695 | 85% | 0% |
| **SURV** | 1.3 × price_x (parity w/ survivor) | +146,396 | 1,807 | **−1,067** | 83% | 22% |
| SQ (live) | 1.3 × roll_prem | +143,362 | 1,770 | −1,272 | 79% | 32% |
| MIN15 | SQ + no-roll if price_x<15 | +143,362 | 1,770 | −1,272 | 79% | 32% |
| F8 / F12 | abs floor 8 / 12 pts | ≈ SQ | | | | |
| **NOROLL** | never roll | **−49,430** | −610 | −4,563 | 38% | — |

## Findings

1. **Rolling itself is strongly validated** — NOROLL loses ₹49k where every rolling
   variant makes ₹140k+. The mechanic earns its keep; only the rolled leg's stop is
   miscalibrated.
2. **The user's diagnosis is correct**: 1 in 3 rolled legs re-stops under the live
   rule; the re-stops are churn, not protection — removing the stop entirely (NOSL)
   *improves* total AND has an identical p05 tail to P200 on this window.
3. **Recommended: SURV — rolled-leg SL = 1.3 × price_x** (the same absolute stop the
   survivor already gets). It strictly dominates the live rule on every headline
   metric (total +₹3.0k, p05 −1,067 vs −1,272, win 83% vs 79%, restop 22% vs 32%)
   and is the **best variant on DTE0 expiry days** (mean 1,282 vs 1,194) — precisely
   today's failure mode. One-line change: `new_sl = price_x * 1.3` instead of
   `new_prem * 1.3` (nas_atm4_executor.py ~line 395). Conceptually clean: both legs
   of the post-roll pair share one absolute risk anchor.
4. **P200 is the max-total alternative** (+12% over live) but fattens p05 to −1,695;
   pick it only if total > tail. NOSL is philosophically unacceptable for a live naked
   short (81 days contained no crash).
5. Floors (F8/F12) and the MIN15 gate almost never bind — roll premiums in the data
   are usually large enough that 30% > 8–12 pts, and price_x < 15 never occurred in
   63 rolls. Today's 12.1 roll was unusually small.
6. Per-DTE stable for all live candidates (no sign flips); SQ↔SURV differ on 20/81
   days with mixed signs — the edge is not one lucky day.

## Sins accounting (the seven)

Look-ahead: none — decisions use the current 1-min snapshot only. Survivorship: n/a
(index options). Overfitting: 9 pre-registered variants, one axis; winner is the
conceptually-motivated one, not a peak. Costs: net-of-cost shown, per-leg so variants
with more orders pay more. Regime: **single 4-month window, no OOS — the honest
caveat**; direction of the finding (tight absolute stops on small premiums = churn) is
mechanically grounded, not curve-fit. Correlation: n/a. Capacity: 2 lots, trivial.
1-min granularity understates intraminute touches for ALL variants; the bias runs
*against* tight stops being seen re-stopping, so SQ's true churn is likely ≥ 32%.

## Next step

Deploy SURV to `nas_atm4_executor.py` (one line + comment) after 15:40 with sign-off,
or paper-shadow it first. Re-check after ~20 more recorded days (the window doubles by
late Sep).


## Addendum (2026-08-18 midday) — MAXV, the deployed rule

Arun's refinement on review: `SL = max(price_x, roll_prem) x 1.3` — pure survivor-parity
(SURV) turns dangerous when the premium match OVERSHOOTS (roll_prem > price_x puts the
1.3 x price_x stop just above entry). Re-ran the sweep with MAXV:

| Variant | Total | Mean/day | p05 | Win% | Restop% |
|---|---|---|---|---|---|
| **MAXV (deployed)** | **+151,866** | 1,875 | **-1,067** | 84% | 19% |
| SURV | +146,396 | 1,807 | -1,067 | 83% | 22% |
| SQ (old live) | +143,362 | 1,770 | -1,272 | 79% | 32% |

MAXV strictly dominates SURV (differs on only 6/63 rolls — the overshoots: saves
+3,705 and +3,559 on 2026-04-30 / 2026-05-21, gives back <600 on four small days) and
is the best risk-adjusted variant tested. **Deployed 2026-08-18** (restart 15:40),
`services/nas_atm4_executor.py` — verify review due 2026-08-28.
