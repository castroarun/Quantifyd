# research/77 P3 — SL re-anchor: day-type grouping + ACTUAL-LEG shadow → **NO VALIDATED EDGE**

**Verdict: NO EDGE (regime-unstable). The "re-anchor the SL at 12:00" idea is NOT validated and must NOT be
deployed. Its benefit FLIPS SIGN between halves of our real record: −₹8,831/day (Apr 20–Jun 02) vs
+₹7,246/day (Jun 03–Jul 10); over the full 48 days of ACTUAL traded legs it is ≈neutral (−₹457/day).
The earlier "+₹1,225/lot" result was a WINDOW ARTIFACT — the ATM reconstruction silently dropped the
Apr–May days (their chain capture starts 09:20, not 09:15), so it was fitted to the favourable half only.
Ex-ante gating (CPR width, CPR, gap, VIX, opening-range) shows ZERO predictive correlation, so we cannot
switch it on only when it works.**

## What the re-anchor IS (mechanism — this part is solid)
Tail insurance, not alpha. On the 27 idealised days: neutral on 17/27 (63% — never triggers), saves
catastrophes, clips some winners.
| Saved | base → re-anchor |
|---|---|
| 06-03 | −17,510 → −1,141 (**+16,369**) |
| 06-25 | −11,463 → +4,079 (**+15,542**) |
| 06-10 | −9,668 → −2,091 (+7,577) |
| Cost | base → re-anchor |
| 06-04 | +11,059 → +6,958 (−4,101) |
| 07-09 | +8,453 → +5,101 (−3,353) |

Driver = the AFTERNOON move (|close − noon|), corr 0.41–0.56:
<30pts → diff 0 (never fires) · **30–60pts → −1,178 (whipsaw zone)** · 60–100 → +122 · **>100 → +1,812..+3,176**

## Ex-ante gating: DOES NOT WORK
corr(diff, ·): **CPR width −0.03 · gap +0.06 · opening-range −0.10 · excursion +0.05** · VIX (only 6 usable days).
Bucket tables show tempting patterns (gap-up>60 +2,322; open-above-CPR +2,021; narrow-CPR +1,436) but they
are non-monotonic with n≈10/bucket and ~zero correlation ⇒ **noise. Gating on these = curve-fitting.**

## The killer: ACTUAL-LEG shadow (real traded legs, real premium paths, 48 days)
`scripts/sl_reanchor_shadow.py` replays every real 916 leg's chain path within [entry_time, exit_time],
re-anchoring only legs LIVE at 12:00. Zero orders.
| Window | Days | Actual (30% SL) | Shadow (re-anchor) | Diff |
|---|---|---|---|---|
| Apr 20 – Jun 02 | 23 | +247,869 | +44,752 | **−203,117 (−8,831/d)** |
| Jun 03 – Jul 10 | 25 | +123,433 | +304,603 | **+181,170 (+7,246/d)** |
| **Full** | **48** | **+371,302** | **+349,355** | **−21,947 (−457/d)** |
Not a data artifact: Apr-30's chain has 334k rows, it just starts 09:20 (late start, not corruption).
The idealised sim and the shadow AGREE inside Jun–Jul; the disagreement is ACROSS PERIODS ⇒ regime instability.

## Self-correction (important)
P1b's "+₹1,225/lot, strongly confirmed" was wrong. The chain-derived-ATM reconstruction requires a
09:15–09:18:30 quote window; Apr–May days start at 09:20 → silently excluded → the sample WAS the winning
regime. Lesson: always verify that a reconstruction's day-drops are not correlated with the outcome.

## What survives
- The 30%-off-MORNING stop being "too loose late in the day" is intuitively right and the SL_HIT record is
  ugly (ATM avg −₹7,526/hit, ATM4 −₹5,772, ATM2 −₹3,606 over 49 paper days) — but **no tested alternative
  beats it robustly.** Trail-to-BE / ratchet were already refuted (whipsaw). Re-anchor is regime-unstable.
- **Infrastructure kept:** `scripts/sl_reanchor_shadow.py` + daily cron (15:45 Mon–Fri) → appends
  `results/shadow_log.csv` on the real legs. We accumulate forward evidence at zero risk.

## Next
1. Let the shadow run for 1–2 months. If Jun–Jul-style behaviour persists out-of-sample, revisit; if it
   keeps flipping, close the idea as CONCLUDED.
2. Do NOT change the live SL on current evidence.
3. If revisited: test whether the whipsaw zone (30–60pt afternoon move) can be avoided with a wider
   re-anchor multiple (e.g. ×1.5 instead of ×1.3) — the only untested dimension.
