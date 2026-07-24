# research/76 P2 — 09:16 ATM straddle: churn / exit-policy sweep

**Verdict: G1 PROBE — SIGNAL-ish direction, NOT validated (n=14). Three actionable, paper-testable
tweaks: (1) exit ~15:00 not 15:15 [+962 vs +683/lot], (2) hold through the 11:15–13:00 lull [+768],
(3) KEEP the 0.4% move-stop — it beats HOLD by halving the worst-day tail. Book-early/re-enter: NOT
supported. DO NOT change live on 14 days.**

## Method
Reconstructed the 09:16 ATM short straddle under 11 exit/adjustment policies, per day, from
options_data.db (spot = underlying_spot TABLE; premiums = option_chain nearest ≤90s). Move-stop =
close+re-open ATM at |spot−entry_spot|≥band%, cap 5. Net of **0.15%/leg per transaction** (a re-center
= 4 leg-txns); cost-sensitivity 0.10/0.15/0.20% run. Per 1 lot (65). Script:
`scripts/churn_exit_sweep.py`. **n = 14 clean days** (2026-04–07; option_chain premiums complete at both
09:16 & 15:15 on only 14 of 55 spot-days — recorder gaps late session = the binding limitation).

## Ranking (mean net/lot, n=14)
MOVE_0.4_EXIT1500 +962 (win71, worst −3476) > MOVE_0.4_LULLHOLD +768 (win71) > MOVE_0.4 +683 (win64,
worst −3925) > MOVE_0.3 +668 (churn2.1) > MOVE_0.4_NOREENTER +431 (win43, med −55 = bad) > HOLD/MOVE_1.0
+355 (worst −8538) > MOVE_0.8 +267 > SL30_NORE +237 > MOVE_0.6 +152 > MOVE_0.5 +60. Cost-sensitivity:
rankings stable across 0.10–0.20%.

## Findings
1. **The give-back is the LAST 15 MIN, not early.** HOLD path (net/lot): 09:31 −163 → 10:16 +238 →
   11:15 +639 → (lull dip) 13:00 +475 → 15:00 **+997** → 15:15 **+392**. Peak at 15:00; ~600/lot lost in
   the final 15 min (the 15:15 square-off catches late moves, e.g. today's 15:04 selloff). Early is RED.
2. **Exit ~15:00** is the biggest, cleanest edge (+962). **Hold-through-lull** second (+768).
3. **0.4% move-stop beats HOLD** (+683 vs +355) by **halving worst-day** (−3925 vs −8538) — tail
   protection > churn cost on average. Earlier "churn eats the edge" was WRONG on the mean; the −31.9k
   10-lot paper churn on 07-09 was a bad-DAY whipsaw, not the average.
4. **Book-early/re-enter NOT supported** — P&L is back-loaded; early exit forfeits theta.
5. Mid bands (0.5/0.6) worse than 0.4 AND 0.3 = **non-monotonic ⇒ noise/overfit ⇒ n too small**.

## Caveats (why this is G1, not G2+)
- **n=14, single recent regime** — non-monotonic bands prove it's noisy. Not enough to change live.
- Reconstructed/idealized fills (0.15% assumption; no rejection/margin realism).
- "Exit 15:00" edge may be partly driven by recent late-selloff days; a trend-up-into-close day would
  see it miss late decay. Needs many more days incl. varied closes.
- The binding fix is DATA: recorder must capture complete chains through 15:15 on more days.

## Next levers (owed)
1. Paper-test "exit 15:00" + "lull-hold" on the LIVE 916 systems (config, low risk) and gather forward days.
2. Re-run when the recorder has ≥40 complete days; check per-DTE + monotonicity + a walk-forward split.
3. Tearsheet only if it survives a bigger sample (currently below G4).
