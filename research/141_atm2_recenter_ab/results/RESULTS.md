# research/141 — ATM2 after the rupee stop: re-center, or close for the day?

## Verdict: **NO EDGE — one-and-done stands. Keep the incumbent.**

*(and a second, louder finding: the arm still running with real money on SENSEX ATM2 is the
worst of the fourteen tested — raised below as a separate strategy change, not acted on here.)*

research/96 replaced ATM2's 0.4% spot-move stop with a ₹2,500/lot rupee MTM stop on evidence, and
in the same change set flipped `move_stop_reenter: True → False` on an assertion. This study
measures the half that was asserted. **The assertion was right, but for the wrong reason** — and the
reason matters, because the same wrong reason is still shaping the SENSEX book.

- Data: `backtest_data/options_data.db :: option_chain`, 1-minute, 2026-04-20 → 2026-09-01.
  88 clean NIFTY days and 88 clean SENSEX days after the frozen-chain and partial-session guards.
- Strike = **forward snap** (`K = round(F/step)·step`, `F = K_ref + CE − PE`, research/132), so no
  re-center inherits an accidental delta from spot-rounding.
- Costs = the **measured** research/122 model: exact Zerodha F&O rate card + slippage of
  **0 pt entry / +0.178 pt time-exit / +6.548 pt forced-stop, per leg-side.**
- All figures are **₹ per lot per day, net**. NIFTY lot 65, SENSEX lot 20.

---

## 0. Reconciliation gate vs research/96 — **PASS**

r/96's own script re-run day-by-day against our reimplementation: **0 of 68 days differ**, for both
the rupee stop and the move stop. Our replica *is* r/96's engine.

| | r/96 published | this study (r/96 method: 68 days, calendar DTE≤1, 2 lots, spot-rounded, one-and-done) |
|---|---|---|
| RUPEE ₹2,500/lot | **+2,153/trade · −6,972 worst · 69% win** | +2,274/trade · **−6,972** worst · **69%** win |
| 0.4% move-stop | **+1,386/trade · −6,887 worst · 62% win** | +1,508/trade · **−6,887** worst · **62%** win |
| FAR bucket (DTE≥2) | −160 / −209 / −98 / +309 | **−160 / −209 / −98 / +309** (identical, all four rules) |

Tails, win rates and the entire far-DTE bucket match **to the rupee**. The whole residual — +121 and
+122 per trade — is **one day: 2026-07-28**, which is r/96's own run date and a calendar-DTE0 expiry
day. Under both rules that day books an identical P&L (no stop fired; it ran to the close), and the
implied r/96 value is ≈ +3,074 against our +6,594 — a gap of 27.1 premium points, i.e. r/96 ran
mid-session and took its "EOD" exit ~27 points before the true 15:15 close, on an expiry day where
that is exactly the remaining decay. **Gate passed; interpretation proceeds.**

---

## 1. Per-arm table — NIFTY (88 days, ₹ per lot per day, net)

| arm | total ₹/lot | mean | median | win% | worst day | p5 | stop-fire% | avg re-centers | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ONE_AND_DONE** (incumbent) | **+18,869** | **+214** | +969 | 64% | **−4,541** | **−3,949** | 28% | 0.00 | 0 |
| RECENTER_1 | +13,685 | +156 | +1,012 | 66% | −8,279 | −7,265 | 28% | 0.25 | 1 |
| RECENTER_2 | +7,793 | +89 | +1,012 | 66% | −11,391 | −7,038 | 28% | 0.30 | 2 |
| RECENTER_3 | +8,493 | +97 | +1,012 | 66% | −10,945 | −7,038 | 28% | 0.31 | 3 |
| RECENTER_5 | +8,493 | +97 | +1,012 | 66% | −10,945 | −7,038 | 28% | 0.31 | 3 |
| RECENTER_2_CD15 | +12,663 | +144 | +1,012 | 66% | −10,945 | −6,949 | 28% | 0.30 | 2 |
| RECENTER_3_CD15 | +12,663 | +144 | +1,012 | 66% | −10,945 | −6,949 | 28% | 0.30 | 2 |
| RECENTER_5_CD15 | +12,663 | +144 | +1,012 | 66% | −10,945 | −6,949 | 28% | 0.30 | 2 |
| RECENTER_5_NOGUARD | +8,493 | +97 | +1,012 | 66% | −10,945 | −7,038 | 28% | 0.31 | 3 |
| MOVESTOP_ONE | −23,078 | −262 | −639 | 35% | −4,875 | −3,114 | 65% | 0.00 | 0 |
| MOVESTOP_RECENTER *(pre-July live)* | −66,057 | **−751** | −122 | 47% | −12,270 | −5,800 | 65% | 1.03 | 4 |
| MOVESTOP_RC1 | −48,660 | −553 | −122 | 47% | −5,648 | −5,205 | 65% | 0.58 | 1 |
| MOVESTOP_RC_CD15 | −64,296 | −731 | −122 | 47% | −10,404 | −5,800 | 65% | 1.02 | 4 |
| *NOSTOP_HOLD (reference)* | *+34,066* | *+387* | *+1,046* | *67%* | *−12,892* | *−5,270* | *0%* | *0.00* | *0* |

**Paired vs the incumbent, same days, Holm-adjusted over the 13 comparisons:**

| arm | Δ mean ₹/lot/day | t | p raw | Holm p | beats incumbent? |
|---|---:|---:|---:|---:|---|
| RECENTER_1 | **−59** | −0.41 | 0.680 | 1.000 | no |
| RECENTER_2 | **−126** | −0.73 | 0.466 | 1.000 | no |
| RECENTER_3 | **−118** | −0.70 | 0.485 | 1.000 | no |
| RECENTER_5 | **−118** | −0.70 | 0.485 | 1.000 | no |
| RECENTER_2/3/5_CD15 | **−71** | −0.46 | 0.648 | 1.000 | no |
| RECENTER_5_NOGUARD | −118 | −0.70 | 0.485 | 1.000 | no |
| MOVESTOP_ONE | −477 | −2.09 | 0.036 | 0.362 | no |
| MOVESTOP_RECENTER | **−965** | −3.94 | 0.000 | **0.001** | no — significantly *worse* |
| MOVESTOP_RC1 | −767 | −3.43 | 0.001 | **0.007** | no — significantly *worse* |
| MOVESTOP_RC_CD15 | −945 | −3.96 | 0.000 | **0.001** | no — significantly *worse* |
| NOSTOP_HOLD | +173 | +0.84 | 0.401 | 1.000 | not after haircut |

**There is no positive plateau.** Every re-center count — 1, 2, 3, 5, with or without a 15-minute
cooldown, with or without the strike-change guard — sits *below* the incumbent. The best of them
(n=1) gives back ₹59/lot/day; the worst gives back ₹126. And the re-center systematically **doubles
to triples the tail**: worst day −4,541 → −8,279 (n=1) → −11,391 (n=2); p5 −3,949 → −7,265.

## 2. Per-arm table — SENSEX (88 days, ₹ per lot per day, net)

| arm | total ₹/lot | mean | median | win% | worst day | p5 | stop-fire% | avg rc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ONE_AND_DONE | −2,620 | −30 | +600 | 59% | **−4,474** | **−3,562** | 36% | 0.00 |
| RECENTER_1 | +9,864 | +112 | +679 | 65% | −8,092 | −6,658 | 36% | 0.34 |
| RECENTER_2 | +4,465 | +51 | +679 | 65% | −11,489 | −6,705 | 36% | 0.43 |
| RECENTER_3 / _5 | +6,201 | +70 | +679 | 65% | −10,969 | −6,705 | 36% | 0.47 |
| RECENTER_2_CD15 | +6,310 | +72 | +679 | 65% | −11,489 | −5,951 | 36% | 0.43 |
| RECENTER_3/5_CD15 | +5,354 | +61 | +679 | 65% | −11,489 | −5,951 | 36% | 0.44 |
| MOVESTOP_ONE | −11,496 | −131 | −583 | 44% | −3,973 | −2,569 | 68% | 0.00 |
| **MOVESTOP_RECENTER — the arm running LIVE today** | **−27,673** | **−314** | +316 | 57% | −8,672 | −5,081 | 68% | 1.14 |
| **MOVESTOP_RC_CD15 — live incl. the 15-min cooldown** | **−29,380** | **−334** | +316 | 57% | −9,499 | −5,081 | 68% | 1.12 |
| MOVESTOP_RC1 | −27,204 | −309 | +22 | 51% | −5,942 | −3,709 | 68% | 0.61 |
| *NOSTOP_HOLD (reference)* | *+43,891* | *+499* | *+892* | *69%* | *−16,347* | *−4,126* | *0%* | *0.00* |

Paired vs one-and-done, Holm over 13: `RECENTER_1` **+142** (t 0.75, Holm 1.000), `RECENTER_2` +81,
`RECENTER_3/5` +100, `RECENTER_*_CD15` +91…+101 — **all positive, none remotely significant, all
Holm p = 1.000.** SENSEX is the venue where re-centering *might* pay, and it still cannot clear the
bar. Every one of them also roughly **doubles the worst day** (−4,474 → −8,092 … −11,489).

---

## 3. The churn cost, priced rather than asserted

This is the line r/96 asserted. Here it is measured. "Extra cycles" = every straddle opened *after*
the first; extra-cycle gross is what those straddles earned before dealing cost.

**NIFTY (lot 65 → a forced stop costs 6.548 pt × 65 = ₹426 per leg-side):**

| arm | extra cycles | extra-cycle GROSS ₹/lot | extra-cycle COST ₹/lot | extra-cycle NET | cost / gross |
|---|---:|---:|---:|---:|---:|
| RECENTER_1 | 22 | +1,150 | 6,334 | **−5,184** | 551% |
| RECENTER_2 | 26 | −2,772 | 8,304 | **−11,076** | 300% |
| RECENTER_3 / _5 | 27 | −1,992 | 8,383 | **−10,375** | 421% |
| RECENTER_*_CD15 | 26 | +1,274 | 7,479 | **−6,205** | 587% |
| MOVESTOP_RECENTER | 91 | +4,414 | 47,392 | **−42,979** | 1,074% |
| MOVESTOP_RC_CD15 | 90 | +6,100 | 47,318 | **−41,218** | 776% |

**The re-centered straddles are gross-flat.** Over 88 days the extra NIFTY cycles earn between
−₹2,772 and +₹1,274 per lot *in total* — statistical noise around zero — while costing ₹6,334 to
₹8,383. **Dealing cost is 300–590% of everything the extra trades produce.** The re-center does not
lose because it picks bad straddles; it loses because it cannot pay for its own round trip.

**SENSEX (lot 20 → a forced stop costs only ₹131 per leg-side):**

| arm | extra cycles | extra-cycle GROSS ₹/lot | extra-cycle COST ₹/lot | extra-cycle NET | cost / gross |
|---|---:|---:|---:|---:|---:|
| RECENTER_1 | 30 | +16,070 | 3,586 | +12,484 | 22% |
| RECENTER_3 / _5 | 41 | +13,587 | 4,766 | +8,821 | 35% |
| **MOVESTOP_RECENTER (live)** | 100 | +3,919 | 20,096 | **−16,177** | 513% |
| **MOVESTOP_RC_CD15 (live)** | 99 | +2,172 | 20,056 | **−17,884** | 923% |

SENSEX's smaller lot makes the same slippage six times cheaper per lot, which is why the rupee-stop
re-center arms are gross-positive there. **But the live SENSEX arm is not a rupee-stop re-center —
it is a move-stop re-center**, which fires nearly twice as often and therefore pays the cost 100
times instead of 30. Its extra cycles produce +₹2,172–3,919 gross against ₹20,056–20,096 of cost.

---

## 4. The mechanism — the re-centered straddle stops out again

| arm | cycle-2+ straddles (NIFTY / SENSEX) | re-stop rate |
|---|---|---|
| RECENTER_1 | 22 / 30 | 27% / 30% |
| RECENTER_2 / _3 / _5 | 26–27 / 38–41 | 30–31% / 29–32% |
| RECENTER_*_CD15 | 26 / 39 | 27% / 31% |
| **MOVESTOP_RECENTER** | 91 / 100 | **55% / 61%** |

Re-center after a rupee stop and the fresh straddle stops out again ~30% of the time. Re-center
after a *0.4% move* stop — the live SENSEX rule — and it stops again **55–61%** of the time, because
a 0.4% move is a low bar that the same trending session clears repeatedly. **The 15-minute cooldown
does not fix this** (61% → 61%): it delays the re-entry without changing the fact that the day is
trending. That is the r/60 churn cascade, priced.

---

## 5. Per trading-DTE — r/96's stated reason is **backwards**

r/96 justified one-and-done as *"re-center adds churn on trending/**expiry** days"*. The data say the
opposite: on expiry days the re-center is the *better* of the two; it is the **far-DTE days** where
it bleeds.

**NIFTY, ₹/lot/day net:**

| trading DTE | n | ONE_AND_DONE | RECENTER_1 | RECENTER_3 | RECENTER_3_CD15 | MOVESTOP_RECENTER |
|---|---:|---:|---:|---:|---:|---:|
| **0 (expiry)** | 18 | +842 | **+1,024** | +1,024 | +1,038 | −19 |
| **1** | 19 | −30 | **+243** | +243 | +243 | −28 |
| 2 | 16 | **+208** | +81 | +81 | +81 | −538 |
| **3+** | 35 | **+27** | −305 | −453 | −341 | −1,616 |

**SENSEX, ₹/lot/day net:**

| trading DTE | n | ONE_AND_DONE | RECENTER_1 | RECENTER_3 | MOVESTOP_RECENTER |
|---|---:|---:|---:|---:|---:|
| 0 | 18 | +74 | **+773** | +531 | +41 |
| 1 | 18 | **−406** | −779 | −829 | −1,553 |
| 2 | 18 | −300 | **+183** | +183 | +17 |
| 3+ | 34 | +258 | +197 | +243 | −22 |

The clean claim from r/96 — "it churns on expiry days" — is **not supported on either venue**. On
NIFTY the re-center is worth +182 and +273 on DTE0 and DTE1 and −178 and −332 on DTE2 and DTE3+.
The right conclusion was reached from the wrong premise. The reason it fails is **cost per round
trip**, which is a function of lot size and stop-fire rate, not of expiry proximity.

The one bucket where re-centering looks attractive — NIFTY near-expiry (trading DTE≤1, n=37) —
still fails the bar: +623 vs +394 (**Δ +229/lot/day, t 0.92, p 0.355, Holm 1.000**), bought with a
worst day that goes from −4,541 to −8,279. n=37 days and one insignificant t is not a mandate to
re-arm a churn mechanism that has a documented live incident behind it.

---

## 6. OOS split

IS = r/96's own day set (≤ 2026-07-28). OOS = after the deploy decision was taken.

| venue / period | n | ONE_AND_DONE | RECENTER_1 | RECENTER_2 | RECENTER_3 | RECENTER_3_CD15 |
|---|---:|---:|---:|---:|---:|---:|
| NIFTY IS | 64 | **+306** | +237 | +145 | +156 | +221 |
| NIFTY OOS | 24 | **−29** | −62 | −62 | −62 | −62 |
| SENSEX IS | 64 | −93 | **+75** | −34 | −7 | −20 |
| SENSEX OOS | 24 | +140 | +212 | **+277** | +277 | +277 |

NIFTY: the incumbent wins in both windows. SENSEX: the re-center wins OOS but not IS. **The sign is
not stable across venue or window** — which is exactly what a null looks like. No arm holds.

---

## 7. The live arm — real recorded money already running this experiment

SENSEX ATM2 kept `move_stop_pct 0.004` + `move_stop_reenter True` (r/96 scope fix, commit
`c95f10a`, 2026-07-29). The two NIFTY ATM2 books ran the same arm *before* 2026-07-28. Three real
ledgers:

| book | window | re-centers | recorded ₹/lot | mean ₹/lot each | re-stopped again |
|---|---|---:|---:|---:|---:|
| SENSEX ATM2 (live now) | 2026-08-04 → 08-27 | 6 | **+3,244** | +541 | **0 / 6 (0%)** |
| NIFTY 916-ATM2 (pre-r/96) | 2026-06-24 → 07-24 | 19 | **−2,521** | −133 | 7 / 19 (37%) |
| NIFTY squeeze-ATM2 (pre-r/96) | 2026-06-24 → 07-24 | 17 | **−9,050** | −532 | 9 / 17 (53%) |
| **combined** | | **42** | **−8,327** | **−198** | **16 / 42 (38%)** |

Detail on the SENSEX live arm (2 lots, DTE0/1 gated): eight days where the move-stop fired; six
were re-centered and **all six ran to 15:15 without a second stop** (+364, +2,114, +1,566, +562,
+534, +2,052 = +₹7,192; extra-cycle cost re-priced on the measured model, ₹451). Two stop-days
(08-25, 09-01) ended without a re-center.

**Read this honestly.** SENSEX's live re-centers have made money — but a 0-for-6 clean run against a
modelled 61% re-stop rate is a small and unusually kind draw, and the NIFTY books, which ran the
identical arm for longer, lost **₹11,571 per lot** across 36 re-centers with a 44% re-stop rate,
including the 2026-07-08 and 07-09 cascades (−₹5,115/lot and −₹2,138/lot in single days) that
became the research/60 churn incident. Pooled across all 42 real re-centers the arm is **−₹198 per
lot per re-center**. Real money agrees with the replay.

---

## 8. Answer to the question, and the recommendation

> After ATM2's ₹2,500/lot rupee stop fires, is closing for the day better or worse than
> re-centering — and if re-centering wins, how many times?

**Closing for the day is better. Re-centering does not win at any n.** On NIFTY every re-center
count loses to the incumbent (−59 to −126 ₹/lot/day) and every one of them roughly doubles the tail.
On SENSEX the re-center's small positive edge (+81 to +142) dies under a family-wise haircut and
does not hold across the IS/OOS split. There is no plateau, no significance, no OOS stability.

**Recommendation 1 — NIFTY ATM2: KEEP one-and-done. No change. No deploy required.**

r/96's decision was correct. It should now be recorded as *measured*, with the correction that its
stated mechanism was wrong: the re-center fails on **round-trip cost against a gross-flat extra
trade**, not on expiry-day trending. This is the fourth independent reproduction of the same result
after research/54, research/60 and research/123.

**Recommendation 2 — SENSEX ATM2: raise as a SEPARATE strategy change (not part of this study).**

SENSEX ATM2 is live on real money running `MOVESTOP_RC_CD15`, which this study ranks **last of the
fourteen arms on its own venue** (−334 ₹/lot/day vs −30 for one-and-done and +499 for no stop at
all), losing −₹17,884/lot to churn across 99 extra cycles. The r/96 scope fix rightly refused to
apply an unvalidated NIFTY rupee stop to SENSEX, but it left the book on the combination this study
finds worst. **This needs its own STATUS-MD, its own SENSEX-calibrated stop, Arun's sign-off, and an
after-15:40 deploy.** Nothing in this study touches it.

**Open question, deliberately not acted on:** `NOSTOP_HOLD` posts the best mean on both venues
(+387 NIFTY, +499 SENSEX) with the fattest tail (−12,892, −16,347) and does not clear the haircut
(Holm 0.362 / 0.688). Whether the ₹2,500/lot stop pays for itself at all is a different question
from the one asked here and deserves its own study before anyone touches a stop.

---

## 9. Honesty / caveats

- **88 recorded chain days per venue** (2026-04-20 → 09-01). Short. The pooled real-money ledger
  (42 re-centers) is the independent check, and it agrees.
- **1-minute cadence.** Intra-minute stop-throughs are unmodelled. This flatters every arm equally
  but flatters the re-center arms slightly more, since they take more stops.
- **09:16 entry only.** The squeeze-entry ATM2 variant is not replayed; its entry time differs.
- **The live SENSEX book is DTE0/1- and gap-gated** by the day matrix; the replay trades every day,
  so the live ledger and the replay are not like-for-like day sets.
- **Multiple testing is controlled** by Holm over 13 arms per venue; the strongest positive result
  anywhere (SENSEX RECENTER_1, +142) has raw p = 0.454 before any haircut.
- Seven deadly sins: no look-ahead (strict forward minute walk); no survivorship (every recorded day
  replayed, guards pre-registered); overfitting controlled by the plateau + Holm + OOS gates; cost
  is measured, not assumed; regime dependence reported per DTE and per IS/OOS window; the venue
  comparison is the correlation check; capacity is unchanged (this changes no size).
