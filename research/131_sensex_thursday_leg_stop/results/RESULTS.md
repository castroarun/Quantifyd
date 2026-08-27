# research/131 — SENSEX DTE0 wide per-leg stop: RESULTS

**Question.** research/114 killed the 30% per-leg stop on SENSEX expiry day and the live config has
carried `leg_sl_disabled_dtes=(0,)` ever since. But r/114 only tested 30% and combined stops. A tight
stop harvests intraday churn; a wide one should only catch a genuine breakaway. **Is there a wide
level that caps a breakaway leg while leaving the expiry-day decay edge intact?**

# VERDICT: **NO EDGE — no per-leg stop at any level. Keep `leg_sl_disabled_dtes=(0,)`.**

**All 33 stop arms lose to HOLD, in both entry sets, under both outer-layer families — 132 arm ×
family × entry-set comparisons, zero winners, zero positive OOS halves.** The loss shrinks
*monotonically* toward zero as the stop widens, which is not a plateau of outperformance but a
monotone approach to the null: the best stop is simply the one that most nearly never fires, and
extrapolating that trend the optimum is a stop at infinity, i.e. HOLD.

And there is a harder, operational reason the answer is no, which the standalone numbers alone would
not have shown: **every level wide enough to stop being a noise-harvester is one the deployed venue
book stop/TP has already pre-empted.** At `RUP8000` the fire rate under the live outer layer is
**0%** and the effect on P&L is **exactly ₹0/lot (t = 0.00)**. The wide leg stop is not merely
unprofitable — it is inert. There is no level that is simultaneously (a) wide enough not to harvest
noise and (b) still able to act before the −₹3,000/lot book stop or the +₹4,000/lot TP.

This is the **fifth independent reproduction** (r/114, r/116, r/121, r/122, r/124) that tightening
SENSEX DTE0 is destructive — and the first to test the wide end of the range, which was the one
untested variant. That gap is now closed.

---

## 1. Data and construction

| | |
|---|---|
| Source | `backtest_data/options_data.db :: option_chain`, real **1-minute** SENSEX chain |
| Sessions | **17 recorded DTE0 sessions**, 2026-04-30 → 2026-08-20, all 17 pass the holiday / partial / thin guards |
| DTE labelling | front expiry == the day itself (**not** weekday) — era-safe across the Fri → Tue → Thu expiry moves |
| Construction | ATM straddle (strike = `round(spot/100)*100`), front expiry, held to 15:15 |
| Sizing | per **1 lot = 20**. 1 premium point = ₹20 |
| Costs | measured outcome-aware model (r/122): forced/stop exit **+6.548 pt** per leg-side, time/EOD **+0.178 pt**, entry 0, plus the exact Zerodha rate card |

**Two entry sets, because the recorder changed.** Before 2026-06-04 the SENSEX chain recorder started
at **09:20**, not 09:16 — which is exactly why r/114 dropped its five May sessions as "no clean
series". They are not bad days; they simply have no 09:16 print.

- **A0920** — uniform 09:20 entry, **all 17** DTE0 sessions. Primary, largest n.
- **B0916** — live-exact 09:16 entry, the **12** sessions that have it. This is *precisely* r/114's day set.

Every arm within a day sees the same minute series, so all comparisons are paired.

---

## 2. r/114 reconciliation — it ties out, and the cost fix makes r/114 *stronger*

Set **B0916** is r/114's exact 12 days; **R114COST** is r/114's own cost model (`SLIP=1.0` pt per
leg-side, `CHG=₹30`/lot).

| Arm | r/114 published | r/131 B0916 + R114COST | Δ |
|---|---|---|---|
| HOLD | +2,630/lot, 92% win, n=12 | **+2,660/lot, 92% win, n=12** | +30 (1.1%) |
| LEG30 (SBOTH) | −227/lot, 25% win | **−207/lot, 25% win** | +20 |

**Reconciled.** Win rates match exactly; means agree to ~1%. The residual is the entry-minute /
ATM-strike spot pick (r/114 took the last unordered row at ≤09:16; r/131 takes the last spot at or
before the entry minute).

**Does the cost change alone move r/114's conclusion? No — it deepens it.** Same 12 days, same rule,
only the cost model swapped:

| Arm | r/114 cost | measured cost | Δ |
|---|---|---|---|
| HOLD | +2,660 | **+2,831** | **+171** |
| LEG30/SBOTH | −207 | **−272** | **−65** |
| **gap** | 2,867 | **3,103** | **+236/lot** |

Because the measured model puts the slippage where it actually is — **in the stop-outs** (6.548 pt vs
0.178 pt) — and HOLD never stops. r/114 used a *flat* 1.0 pt on every leg-side, which over-charged
HOLD and under-charged the stop. Correcting it widens the gap by ₹236/lot. **r/114's headline was
right, and slightly conservative.**

---

## 3. Fire rate — the diagnostic that answers the actual question

The hypothesis was that a wide stop only catches a genuine breakaway. The fire rates say the range
where that becomes true does not exist inside the tradable band.

| Level | pts on a ~150 leg | fire% (A0920, n=17) | fire% standalone → **with live outer layer** | median fire time |
|---|---|---|---|---|
| LEG30 | +30% | **100%** | 100 → 100 | 09:39 |
| LEG40 | +40% | **100%** | 100 → 100 | 09:48 |
| LEG50 | +50% | **100%** | 100 → 94 | 09:51 |
| LEG60 | +60% | 82% | 82 → 76 | 10:20 |
| LEG75 | +75% | 82% | 82 → 76 | 11:57 |
| LEG100 | +100% | 41% | 41 → 41 | 11:17 |
| RUP1500 | 75 pt | **100%** | 100 → 100 | 09:45 |
| RUP2500 | 125 pt | 82% | 82 → 76 | 10:20 |
| RUP4000 | 200 pt | 47% | 47 → 47 | 11:17 |
| RUP6000 | 300 pt | 29% | 29 → **24** | 11:43 |
| RUP8000 | 400 pt | 12% | 12 → **0** | 13:37 |

**The reading:** on expiry day, *a leg doubling is a routine outcome, not a tail event.* A stop at
**100% of entry premium still fires on 41% of expiry sessions** (7 of 17). This is structural, not
bad luck — one leg of a short straddle always rises, and on DTE0 gamma makes the rise violent while
theta makes the other leg collapse faster. There is no percentage level in the tradable band that
distinguishes "breakaway" from "ordinary straddle asymmetry".

The only levels that become genuinely rare are the very wide rupee stops — and those are exactly the
ones the venue book stop/TP reaches first (**RUP8000: 12% → 0%**).

---

## 4. The plateau map — monotone toward the null, no plateau of outperformance

`dMEAN` = mean net ₹/lot minus HOLD's, paired by day. **A0920, STANDALONE, n=17. HOLD = +2,628/lot,
82% win, worst −212.**

| Level | SBOTH | SHOLD | STRAIL |
|---|---|---|---|
| LEG30 | **−3,087** (t −6.23) | −2,149 (t −3.09) | −3,020 (t −4.43) |
| LEG40 | −3,268 | −2,654 | −3,378 |
| LEG50 | −3,351 | −2,864 | **−3,553** (worst arm) |
| LEG60 | −2,746 | −2,823 | −2,956 |
| LEG75 | −2,933 | −3,288 | −3,172 |
| LEG100 | −1,975 | −2,770 | −2,175 |
| RUP1500 | −3,123 | −2,419 | −3,144 |
| RUP2500 | −2,926 | −2,898 | −3,229 |
| RUP4000 | −2,401 | −3,108 | −2,634 |
| RUP6000 | −1,954 | −2,380 | −2,065 |
| RUP8000 | **−867** (t −1.44) | **−801** (t −1.44) | **−801** (t −1.44) |

Same picture on the live-exact 12-day set (**B0916, STANDALONE, HOLD = +2,831/lot, 92% win**): every
arm negative, best `LEG40/SHOLD` at −1,544 (t −1.78), worst `LEG30/SBOTH` at −3,103 (t −4.29).

**Gate 2 (plateau) fails in the specific way that matters:** the numbers do improve smoothly as the
level widens, so the *sign* is stable — but the direction of the monotone is toward **zero from
below**, never above it. The "best" level is the one that fires least. That is the signature of a
rule with no information, only cost.

---

## 5. Family-wise haircut and OOS

**Sign-flip permutation, 10,000 draws, max|t| across all 33 stop arms** (day-level flips applied
jointly, preserving cross-arm correlation):

| Set / family | best arm | dMEAN | t | family-wise p | max\|t\| null p95 / p99 |
|---|---|---|---|---|---|
| A0920 STANDALONE | RUP8000/SBOTH | −867 | −1.44 | 0.769 | 2.84 / 3.62 |
| A0920 VENUE | RUP8000 (all 3) | 0 | 0.00 | 1.000 | 2.90 / 3.73 |
| B0916 STANDALONE | RUP8000/SHOLD | −2,077 | −1.74 | 0.494 | 3.07 / 3.81 |

Nothing comes close to passing. Note the flip side: the *harm* clears the same family-wise bar with
room to spare — `LEG30/SBOTH` at t = −6.23 and `LEG50/STRAIL` at t = −4.99 both exceed the p99
threshold of 3.62. **Tight stops are family-wise significantly destructive; no level is
family-wise significantly helpful.**

**OOS split** (first 8 / last 9 sessions in A0920; 6 / 6 in B0916): **0 of 33 arms is positive in
either half, in either set.** There is no half of the sample, in either construction, in which any
leg stop beat holding. Unanimous.

---

## 6. Survivor treatment — the live ST trail does its job, on damage the stop created

Conditional on a leg having stopped, the treatments rank consistently **SHOLD ≥ STRAIL > SBOTH** on
the mean, but the tails rank the other way (A0920, LEG30):

| Treatment | mean | win% | **worst** |
|---|---|---|---|
| HOLD (no stop at all) | **+2,628** | 82% | **−212** |
| SHOLD (hold the naked survivor to 15:15) | +480 | 71% | **−4,808** |
| STRAIL (live ST(7,3) ceiling, r/128) | −391 | 47% | −3,524 |
| SBOTH (close both together) | −459 | 24% | −2,515 |

**The ST trail that went live 2026-08-26 measurably works as designed:** it truncates the naked
survivor's tail by **₹1,284/lot** (−3,524 vs SHOLD's −4,808) for **₹871/lot** of mean. That is a
real, favourable, quantified property of the deployed trail and it is worth recording. But it is a
repair on damage the leg stop caused in the first place — no survivor treatment rescues any arm, and
the cheapest way to have a small tail here is not to stop the leg at all (HOLD's worst is −212).

`SBOTH` is what r/114's LEG30 did. `SHOLD` mitigates it a little; `STRAIL` mitigates the tail. None
of the three reaches HOLD.

---

## 7. Interaction with the deployed outer layer — the decisive finding

The live DTE0 outer layer (`services/nas_portfolio_stop.py`): book stop **−₹3,000/lot**, TP
**+₹4,000/lot**. Applied to the same 17 sessions, HOLD's exits resolve as:

> **18 BOOK_TP · 10 BOOK_STOP · 6 EOD** (34 leg-exits over 17 sessions, A0920)

**The outer layer already resolves 82% of expiry sessions before 15:15.** That is where the DTE0
protection actually lives. Consequences:

- **HOLD under the venue layer: mean +1,134/lot, 65% win, worst −3,770** (vs +2,628 / 82% / −212
  standalone). The layer costs ₹1,494/lot of mean and buys a bounded worst case. That trade is
  Arun's existing, signed-off choice; this study does not revisit it.
- **Every leg stop still loses under it.** `LEG50/SBOTH` −1,821/lot (t −2.59), `LEG40/SBOTH` −1,774
  (t −2.56), `LEG75/SHOLD` −2,028 (t −2.22).
- **And the wide ones become inert.** `RUP8000` fires on **0 of 17** sessions once the outer layer is
  present; dMEAN **0**, t **0.00**. `RUP6000` drops 29% → 24%. On the 12-day live-exact set,
  `RUP8000` goes 25% → 0% and `RUP6000` 33% → 8%.

So the brief's own test — *"a leg stop that never fires before the book TP does is irrelevant"* — is
met literally. **The wide end of the range is not an untested opportunity; it is a region the
deployed book stop already occupies.**

---

## 8. Worst-day comparison

**2026-06-11** (the session research/122 flagged at −₹54,100 on 10 lots):

| | HOLD | best stop arm | LEG100/SHOLD | RUP8000 |
|---|---|---|---|---|
| Standalone | **+3,502/lot** | LEG30·RUP1500 /SHOLD +1,956 | −703 | −4,490 |
| With venue layer | −3,770/lot | — | — | — |

Two things to say honestly about this day:

1. **On this construction it closes positive.** Held to 15:15, the 09:20 ATM straddle made
   **+₹3,502/lot** on 2026-06-11. Every one of the 33 leg-stop arms did worse. r/122's −₹5,410/lot
   figure is a *different construction and window*; it is not this book's outcome.
2. **What turned 06-11 into a loss was the book stop, not the absence of a leg stop.** With the live
   outer layer the day books −₹3,770/lot, because the combined P&L touched −₹3,000/lot intraday on an
   excursion that fully recovered by the close. That is a finding about the **book stop level**, not
   about leg stops — and it is the natural follow-on study (§10).

**Study's worst HOLD day, standalone: 2026-06-25 at −₹212/lot.** Every leg stop made it worse:
LEG50/SBOTH −265, LEG30/SHOLD −4,808, LEG100/SHOLD −7,270, RUP4000/SHOLD −8,438. Only the levels so
wide they never fired (RUP6000, RUP8000) matched HOLD's −212 — by doing nothing.

---

## 9. Honest caveats

- **n = 17 sessions.** Small. Gate 4 (OOS) is weak by construction; this study can *kill* a level
  convincingly — and it does, unanimously — but it could only ever have called a survivor a
  candidate.
- **This sample is benign, and r/118 says the real DTE0 distribution is not.** A worst standalone
  HOLD day of −₹212/lot over 17 sessions cannot price the tail r/118 measured over 127 real DTE0 days
  (~34% losers, 8.7% worse than −500 pts, worst ≈ −21,500/lot). **What this study can say about that
  tail is narrow but useful: a per-leg stop is not the instrument that catches it**, because the
  levels rare enough to be tail-only never fire before the book stop does (§7).
- **Scope: the plain 09:16/09:20 ATM straddle.** `leg_sl_disabled_dtes=(0,)` is also set on
  SENSEX_ATM4, whose roll mechanics differ. r/113 covers the ATM4 roll-leg stop separately.
- **Venue proxy.** The real book stop is computed across all three SENSEX systems; modelling it on a
  single straddle per-lot makes it fire *later* than the real one on days when siblings are also
  down. So §7 if anything *understates* how much of the field the outer layer already covers.
- **Rate card.** The Zerodha F&O card is applied identically to BFO and NFO, as r/122 did, for
  comparability. Any BSE/NSE transaction-charge difference is second-order against a 6.548 pt
  slippage term.

---

## 10. Recommendation

**No live change. Keep `leg_sl_disabled_dtes=(0,)` on SENSEX_ATM and SENSEX_ATM4.** Nothing here is
deployable and nothing requires an after-15:40 window.

The discomfort that motivated the question — *"no leg-level protection at all on expiry day"* — has a
factual answer that this study can now point at: **the protection is the −₹3,000/lot venue book stop,
which fires on 10 of 17 recorded DTE0 sessions and binds on 82% of them.** DTE0 is not unprotected;
the protection simply sits at the book level rather than the leg level, which is the correct level
for it, because leg-level moves on expiry day carry no information about book-level disaster.

**If Arun wants more DTE0 tail protection, this study says the lever is not a leg stop.** The two
candidates it points to, neither tested here:

1. **The book stop level itself.** 2026-06-11 shows −₹3,000/lot converting a +₹3,502 day into a
   −₹3,770 day on a recovered excursion. A calibration study of the DTE0 book stop (and of position
   size, per r/118's sizing note) is the natural successor and is where the real money is.
2. **A bought far wing** (the r/126 / r/127 family) — a structurally different answer that caps the
   tail without needing to time a stop. Untested on SENSEX DTE0.

**Suggested dated review:** re-run this study once the recorder holds **≥ 40** DTE0 sessions
(≈ 2027-01, at ~1/week), specifically to see whether a real disaster day appears in-sample and
whether the fire-rate/inertness picture in §7 survives it.

---

## 11. Reproduce

```bash
ssh arun@94.136.185.54 && cd /home/arun/quantifyd
nice -n 15 venv/bin/python3 research/131_sensex_thursday_leg_stop/scripts/run_leg_stop_sweep.py
nice -n 15 venv/bin/python3 research/131_sensex_thursday_leg_stop/scripts/analyse_leg_stop.py
```

Outputs: `results/leg_stop_detail.csv` (2,088 rows), `results/leg_stop_summary.csv`,
`results/r114_reconciliation.csv`, `results/analysis.txt` (full report), `results/run.log`.
All small; nothing gitignored. Runtime ≈ 90 s total, single-threaded, DB opened read-only.
