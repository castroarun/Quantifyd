# research/83 — The giveback: can any exit / re-entry policy keep it? → **NO EDGE. HOLD wins.**

**Verdict: NO EDGE. Nothing beats holding to 15:15. Do not change the live exits.**

## The problem is REAL (this part is not in doubt)
Measured on the ACTUAL live book (3x916, normalised to 2 lots, 36 recorded days):

| | |
|---|---|
| mean +30min | **+697** |
| mean +60min | +1,059 |
| **mean PEAK** | **+7,708** |
| mean CLOSE | +1,401 |
| **GIVEBACK (peak→close)** | **−6,308/day = 82% of the peak** |
| peak lands before 11:00 | only **11/36 days (31%)** |

So the user's observation is **half right**: the ~Rs7-9k is real (it is the PEAK), and it really is
given back. But it is **NOT a 30-60min peak** — at 30-60min you are only at ~Rs700-1,000. The peak is
**LATE**. That alone kills "book early": there is nothing to book early.

## Everything tested, PAIRED against HOLD on the same day (n=32, per lot, net 0.15%/leg)

| policy | diff mean | MEDIAN | strip2 | better/worse | 95% CI | |
|---|---|---|---|---|---|---|
| TARGET +1000/lot | +325 | **−210** | **−464** | **7 / 19** | [−813, +1710] | ✗ |
| TARGET +1500/lot | −181 | +0 | −459 | 7 / 14 | [−807, +402] | ✗ |
| TARGET +2000/lot | +24 | +0 | −251 | 7 / 9 | [−557, +587] | ✗ |
| TRAIL arm1000 ret25% | +237 | **−387** | **−546** | **6 / 18** | [−899, +1589] | ✗ |
| EXIT 15:00 | +52 | −28 | −165 | 14 / 16 | [−342, +496] | ✗ |
| **TGT1500 → REENTER 30m** | +129 | +0 | −90 | 11 / 10 | [−252, +567] | ✗ |
| **TGT2000 → REENTER 30m** | +146 | +0 | −66 | 8 / 8 | [−235, +554] | ✗ |

**Every CI includes zero. Nothing is distinguishable from HOLD.**

**RE-ENTRY — the half nobody had tested — is dead too.** research/76 closed-and-stayed-flat; the
user's actual proposal was "close AND re-enter later". A P&L curve cannot price a NEW straddle, so
this was done from the chain. It does not beat holding.

## Two mistakes of mine this study caught (both worth remembering)

1. **The trail (my idea) was outlier noise.** It looked like +1,118/day, 92% win, worst day halved.
   Reality: **median −114**, hurt 18/36 days, and the ENTIRE edge was 2 days (drop them → +32; drop 5
   → −1,289; bootstrap CI [−1663, +4041]). The **92% win rate was the tell, not the evidence** —
   identical to the research/80 directional short (60% win, +343 median, NEGATIVE mean). Clipping
   policies manufacture high win rates by construction.
2. **I first tested "is the mean > 0" instead of "does it beat HOLD".** On that wrong question
   TARGET+1000 came out "REAL" (CI [+72,+969]). Paired against HOLD it is clearly worse on the
   typical day (median −210, hurts 19/32). **HOLD's own mean is only +254 with a CI spanning zero —
   so "beats zero" says nothing.** Always pair against the baseline you would actually run.

## Why no exit rule can work here (the mechanism)
The book is **short gamma** and its decay is **back-loaded** (research/76: red early, late peak).
Any rule that clips the giveback also clips the days that keep running — the same volatility that
takes the Rs7k away is what pays when it does not arrive. You cannot keep one without the other via
a P&L-triggered exit.

## Caveats
- n=32-36 days, one regime. HOLD itself is not significantly > 0 here either — this sample cannot
  prove the book makes money, only that these policies do not improve it.
- Costs 0.15%/leg; exits of a 6-leg book would cost more in practice, which makes the clipping
  policies WORSE, not better.
- Not tested: churn reduction (research/76's actual conclusion — the erosion driver is move-stop
  churn, and its winners were EXIT1500/LULLHOLD *versus the move-stop*, not versus HOLD).

## Decision
**Change nothing.** The live exits stay as they are. The giveback is real but is not harvestable by
a profit target, a trailing stop, or re-entry. If this is revisited, the only untested lever left is
**reducing move-stop churn**, not adding a P&L exit.
