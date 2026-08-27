# research/133 — What combined-premium backstop belongs on the stop-less SENSEX DTE0 straddle?

## VERDICT: **CONCLUDED — KEEP `BACKSTOP = 0.50`. It is the corner of the curve, not a round number. But the rupee cap it delivers is ≈ −₹17,200 at 6 lots, not −₹13,900.**

**No live change is recommended.** The one thing that *should* change is the number Arun
carries in his head for what the backstop actually costs on the bad day.

---

## The one-paragraph answer

Arun's instinct — that ~85 days of recorded options cannot price a disaster — is **correct and
is confirmed quantitatively**: the recorded window's DTE0 afternoons are a *calm slice*, with a
median excursion 78% and a **worst day only 45%** of the multi-year DTE0 sample's. It contains
no disaster at all. But when the multi-year price action is bridged into premium space and the
backstop is priced as insurance across **131 SENSEX DTE0 sessions, 1,356 SENSEX sessions, 375
NIFTY DTE0 sessions and 2,754 NIFTY sessions including COVID**, the answer that comes back is
that **0.50 is already the right level**. The expectation cost of the backstop shrinks
monotonically as the level widens and **crosses zero at L ≈ 0.45–0.60 in every one of the four
independent scopes**; below 0.50 you pay real money for the protection, above 0.50 you buy no
extra expectation and give away tail. 0.50 sits exactly on that corner. What the study *does*
overturn is the arithmetic: overshoot, the 2-poll dwell and the measured +6.548 pt forced-exit
slippage add ~24% to the intended cap, and **38% of DTE0 breaches jump the level inside a single
minute**, so the level is a budget, not a guarantee.

---

## 1. Reconciliation first — does Stage A tie out to r/114? **Yes, exactly.**

Before interpreting anything, the recorded-chain harness was pointed at r/114's own
construction (09:16 entry → 15:15, SENSEX DTE0):

| Arm | r/114 published | r/131 (measured costs, same 12 days) | **r/133, r/114's exact 12 days** | r/133, all 17 DTE0 |
|---|---|---|---|---|
| HOLD | +2,630/lot, 92% win, n=12 | +2,831/lot | **+2,831/lot, 92% win** | +3,029/lot, 94% win |

**Exact tie-out to r/131 to the rupee, and to r/114's win rate.** The residual vs r/114's
published mean is the cost model (r/114's flat 1.0 pt/leg-side vs the measured outcome-aware
model), exactly as r/131 documented. The harness is trustworthy.

Note: r/133's per-leg-30% arm is the *survivor* mechanic (each leg stopped independently, the
other held), not r/114's SBOTH, so it reads +674/lot rather than −227 — the same asymmetry
r/131 found when it showed SURV dominates SBOTH. It is not used in any conclusion here.

---

## 2. The two samples, and why the small one cannot answer the question

### Stage A — recorded 1-minute chain, **n = 17 clean DTE0 sessions** (2026-04-30 → 08-20)

Frozen-chain holidays 2026-05-01 / 05-28 / 06-26 rejected by the <50-distinct-spot-prints
guard; 2026-08-27 rejected as partial. DTE0 derived from the chain (front expiry == the day),
never from the weekday.

| | 13:00 → 15:20, HOLD (the live rule) |
|---|---|
| credit | p25 200.7 / median 236.8 / p75 295.1 pts (today 231.63) |
| net | mean **+1,731/lot**, median +1,684, win 82% |
| at 6 lots | mean **+₹10,388**, worst **−₹31,440** |
| MAE / credit | p50 0.19 · p75 0.43 · p90 0.88 · max 1.15 |

The deployed 0.50 sits at roughly the p84 of this sample — **3 of 17 sessions reached it.**

### Stage B — multi-year index price action, 13:00 → 15:20

| Series | Days | Scope |
|---|---|---|
| SENSEX 1-minute | **1,356** (2021-01-01 → 2026-08-26) | 131 DTE0 (2024 →, the weekly-expiry era) |
| NIFTY50 5-minute | **2,754** (2015-02-02 → 2026-07-17) | 375 DTE0 (2019-02 →) — **contains COVID** |

DTE0 comes from a **built expiry calendar**, not a weekday: era table (SENSEX Fri-2024 → Tue-2025H1
→ Thu-2025-09→; NIFTY Thu → Tue-2025-09→) with a **walk-back to the previous trading day** on
holiday-shifted expiries, and the trading-day set built from the **union** of four series so a
single-symbol data hole cannot shift a label. **Validation:** the calendar reproduces 15 of the
17 chain-derived 2026 DTE0 dates exactly, including the two holiday-shifted Wednesdays
(2026-05-27, and 2026-06-25 ahead of the 06-26 holiday). The two it misses (2026-05-14,
2026-07-09) are absent from SENSEX 1-minute data entirely and so are not in the sample either.
2026-08-26 is labelled DTE0 because 08-27 is not yet in the price DB — one day, immaterial.

### Was the options window representative? **No — and this is the study's first substantive finding.**

| Scope | n | exc p50 | p75 | p90 | p95 | **max** |
|---|---|---|---|---|---|---|
| recorded window 2026-04-20 → 08-26 | 17 | 26.6 bp | 40.7 | 63.0 | 76.3 | **87.4 bp** |
| full SENSEX DTE0 2024 → 2026 | 131 | 34.2 bp | 50.2 | 73.2 | 87.4 | **196.3 bp** |

The recorded window's median is **78%** of the long sample's, its p95 **87%** — and its worst
day is **45%** of the long sample's worst. **The four months of chain contain no disaster at
all.** Arun's premise is right: a disaster level set on this window would be set on a sample
from which disasters are absent by construction.

---

## 3. The bridge — and the point at which the standard bridge becomes unsafe

Everything is done in a dimensionless variable so it is regime-free:

> **R = (max index distance from the strike, inside the window) ÷ (entry credit in points)**

Three routes, per the commission:

| Route | Form | Source |
|---|---|---|
| **B1 linear** | F(R) = **0.331 × R** | the r/122 method reused: b = median (MAE-frac ÷ excursion-bp) = 0.01084 on the 13 recorded days with ≥20 bp of move; × the median credit of 30.5 bp |
| **B2 intrinsic floor** | F(R) = **max(0, R − 1)** | model-free. On expiry day combined premium is never below intrinsic, so a move of R credits away from the strike forces ≥ (R−1) credits of adverse premium |
| **B3 observed** | the recorded worst | Stage A |

**The two cross at R = 1.49.** Below that the fitted slope describes ordinary days better;
**above it the linear bridge understates the loss and only the intrinsic floor is safe** — at
R = 3 the linear route says the straddle is 0.99 credits under water while arithmetic says it is
at least 2.00. This is r/122's *"bridged tails are FLOORS"* warning made quantitative, and it is
why every number below uses **F(R) = max(F_lin, F_intr)**.

**Bridge validation on the 17 recorded days:** the bridge is **conservative on 15 of 17**
(median error +0.09 credits). The two under-statements are 0.39 vs 0.37 and 0.48 vs 0.39 — and
they are not failures in the direction that matters: at a violent excursion the recorded LTP of
a deep-ITM leg is *stale* and prints **below intrinsic** (visibly so on 2026-04-30). The chain
therefore shows a premium you could not actually buy back at; the intrinsic floor is what a real
cover costs.

**Where it is still a floor:** an IV pop on a genuine shock lifts the *extrinsic* on both legs
above what any excursion-to-premium map can see. Every tail figure below should be read as a
**lower bound**, and the conservative of (bridged, observed) is the one quoted.

---

## 4. Fire rate — on both samples

The credit you sell drives the fire rate, so it is carried as an explicit axis rather than
assumed. C25/CMED/C75 freeze the 2026 credit ladder at 25.9 / 30.5 / 39.7 bp of spot; VOL scales
it by trailing-20-day realised vol (**stated honestly: Pearson r = −0.03 in-sample — the vol fit
has no explanatory power inside a 4-month window, so it is a sensitivity, not the primary**).

| Level | cap/lot | cap @6L | **Stage A (n=17)** | SX DTE0 C25 | **SX DTE0 CMED** | SX DTE0 C75 | SX DTE0 VOL | SX all | NF DTE0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | ₹1,465 | ₹8,787 | **8 / 47.1%** | 77.9% | 69.5% | 54.2% | 75.6% | 83.2% | 83.0% |
| 0.30 | ₹1,701 | ₹10,208 | 8 / 47.1% | 66.4% | 56.5% | 47.3% | 65.6% | 74.2% | 75.3% |
| 0.35 | ₹1,938 | ₹11,629 | 7 / 41.2% | 57.3% | 51.1% | 35.9% | 61.1% | 66.4% | 65.7% |
| 0.40 | ₹2,175 | ₹13,050 | 5 / 29.4% | 51.9% | 45.0% | 26.7% | 54.2% | 58.6% | 61.1% |
| **0.50 (live)** | **₹2,649** | **₹15,893** | **3 / 17.6%** | 43.5% | **35.1%** | 17.6% | 40.5% | 46.8% | 49.6% |
| 0.60 | ₹3,122 | ₹18,735 | 3 / 17.6% | 38.2% | 28.2% | 13.0% | 38.2% | 40.9% | 40.2% |
| 0.75 | ₹3,833 | ₹22,998 | 3 / 17.6% | 35.9% | 22.1% | 12.2% | 35.1% | 35.2% | 33.5% |
| 1.00 | ₹5,017 | ₹30,104 | 2 / 11.8% | 23.7% | 14.5% | 6.9% | 27.5% | 27.3% | 27.8% |

**The first hard answer to the commission's question 1.** The deployed 50% level fires on
17.6% of the recorded sample and on **~35–43% of DTE0 sessions over the long run**. By the
pre-registered standard (≤2% = disaster stop, ≥15% = trading stop) **the 50% backstop is a
trading stop wearing the name of a disaster stop.** No level in the tested range is a true
disaster stop: even L = 1.00 fires on ~15–28% of DTE0 afternoons. That is not a defect in the
level — it is what a short ATM straddle on expiry afternoon *is*. A 1× move of the credit is an
ordinary event, not a catastrophe.

Which means the level cannot be judged on its fire rate. It has to be judged on what it does.

---

## 5. Save vs cost — the table the recommendation rests on

For every historical day: HOLD books the terminal intrinsic at 15:20; the backstop books
−(L + overshoot) × credit and stays flat for the rest of the window. Overshoot is taken from the
**measured** Stage-A distribution (median 0.058 credits) and the measured **+6.548 pt per
leg-side** forced-exit slippage is charged on every fire.

### Effect on the book, ₹ per lot (all four scopes)

| L | SENSEX DTE0 (n=131) | SENSEX all (n=1,356) | NIFTY DTE0 (n=375) | NIFTY all (n=2,754) |
|---|---|---|---|---|
| 0.25 | **−672** (t −2.31) | −479 | −806 | −774 |
| 0.30 | −383 | −281 | −630 | −567 |
| 0.35 | −226 | −113 | −400 | −404 |
| 0.40 | −6 | +16 | −346 | −278 |
| **0.50** | **+289** (t 1.71) | **+211** | −121 | −89 |
| 0.60 | +201 | +223 | +16 | −20 |
| 0.75 | +95 | +158 | +83 | +9 |
| 1.00 | +18 | +127 | +38 | −10 |

**Read the shape, not the winner.** The effect improves **monotonically** as the level widens
and then **flattens to approximately zero from 0.50 upward, in all four scopes independently.**
That is precisely r/131's signature — *"a monotone approach to the null"* — and it means:

- **below 0.50 the stop costs real money** (−₹226 to −₹806/lot at 0.25–0.35, and 0.25 is the
  only arm anywhere in the study that is statistically distinguishable from zero at all, with
  t = −2.31 — and it is distinguishable in the *wrong direction*);
- **above 0.50 nothing is bought** — 0.75 and 1.00 add no expectation in any scope;
- **0.50 is the corner**: the tightest level at which the expectation cost has fully decayed.

**This is the seventh independent reproduction** — after r/114, r/116, r/121, r/122, r/124 and
r/131 — that tightening this book is destructive, and the first to establish where the
destruction *stops*.

### What it saves, and what it gives up

SENSEX DTE0, n = 131, at **6 lots**:

| | HOLD (no backstop) | L = 0.40 | **L = 0.50 (live)** | L = 0.60 | L = 1.00 |
|---|---|---|---|---|---|
| fires | — | 71 (54%) | **53 (40%)** | 50 (38%) | 36 (27%) |
| saves / costs | — | 32 / 39 | **27 / 26** | 25 / 25 | 16 / 20 |
| effect on the book | — | −₹37 | **+₹1,732** | +₹1,203 | +₹105 |
| **worst day** | **−₹75,674** | −₹24,792 | **−₹29,829** | −₹34,867 | −₹55,015 |

**At L = 0.50 the backstop converts the worst SENSEX DTE0 afternoon in the sample from
−₹75,674 to −₹29,829 at 6 lots — a 61% reduction in the worst day — while ADDING ₹1,732 of
expectation rather than costing anything.** That is what good insurance looks like: it is
roughly free because on this book the days it fires split almost evenly between rescues (27) and
regrets (26), and the rescues are larger.

No arm clears the family-wise bar. 15 arms are screened; Šidák two-sided 5% needs p < 0.0034
(|t| ≈ 3.30 on Stage A, where the best arm reaches |t| = 0.54; the long sample's best is
t = 1.71). **Nothing here is statistically separable, and nothing is claimed to be.** The case
for 0.50 is the *shape* of the curve reproduced across four independent samples, plus the fact
that it is the incumbent — which is the correct standard for keeping a parameter, not for
changing one.

### The hybrid tested, and rejected on parsimony

A pure percentage means the cap swings with the credit — 2026-08-06's credit of 672 pts would
have put the 50% cap at ₹6,722/lot (₹40,300 at 6 lots). Three hybrid arms were tested:

| arm | SX DTE0 effect | SX DTE0 worst @6L | SX all effect | SX all worst @6L |
|---|---|---|---|---|
| FRAC50 (live) | +₹289 | −₹29,829 | +₹211 | −₹30,808 |
| HYB50_3000 (min 50%, ₹3,000/lot) | +₹66 | −₹23,288 | +₹188 | −₹23,288 |
| HYB60_3500 (min 60%, ₹3,500/lot) | +₹212 | −₹25,653 | +₹227 | −₹25,758 |

HYB60_3500 is marginally better on SENSEX (both scopes) and buys ~₹4,200–5,000 of worst-day at
6 lots for ~₹80/lot of expectation — **but it is worse than FRAC50 on the NIFTY DTE0 tail, it
is nowhere near significant, and it adds a second parameter.** Under the family-wise discipline
this is mining, not a finding. **Recommended only if Arun independently wants a hard rupee
ceiling on high-credit days**; it is not recommended on the evidence.

---

## 6. Does a gap blow through the level? **Yes, more than a third of the time.**

### On the recorded chain (the authoritative measurement, real premium, real mechanic)

| Level | day | level | first touch | overshoot | overshoot % | realised/lot | intended/lot | **excess** |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 2026-04-30 | 356.2 | 406.0 | 49.8 | **14.0%** | −3,260 | −2,375 | **−885** |
| 0.50 | 2026-06-11 | 442.6 | 446.8 | 4.2 | 0.9% | −3,081 | −2,951 | −130 |
| 0.50 | 2026-06-25 | 355.1 | 382.4 | 27.3 | 7.7% | −3,416 | −2,368 | −1,048 |

Across all 39 fires in the study: **overshoot / credit median 0.058, p90 0.210, max 0.315.**
Realised minus intended cap: **median −₹410/lot, worst −₹1,996/lot (−₹11,973 at 6 lots).**

### On the multi-year index sample (SENSEX 1-minute DTE0, minute-by-minute path walk)

| L | breaches | **gapped** (jumped the level inside one minute) | overshoot at the crossing minute, in credits |
|---|---|---|---|
| 0.25 | 65 | 31 — **47.7%** | med 0.09 · p90 0.31 · max 2.07 |
| 0.40 | 60 | 23 — 38.3% | med 0.08 · p90 0.23 · max 1.92 |
| **0.50** | **53** | **20 — 37.7%** | med 0.08 · p90 0.19 · **max 1.82** |
| 0.75 | 46 | 17 — 37.0% | med 0.05 · p90 0.26 · max 1.57 |
| 1.00 | 36 | 9 — 25.0% | med 0.05 · p90 0.31 · max 1.32 |

**Just over a third of L = 0.50 breaches arrive as a jump, not a walk**, and in the worst
historical case the index cleared the level by **1.82 credits in a single minute** — i.e. the
backstop would have been ~2.3 credits under water at the moment it fired, not 0.5. Gapping is
*worse* at tighter levels (47.7% at 0.25), which is a further independent argument against
tightening: a tight level is both more often triggered and more often jumped.

**The backstop is a budget, not a guarantee.** It should be sized on the assumption that the
bad day pays through it.

---

## 7. The true unstopped tail, and the r/118 reconciliation

r/118 measured DTE0 over 127 sessions at ~34% losers with a worst near **−₹21,500/lot**
(= −₹129,000 at 6 lots) — but on a **full-day 09:16 → 15:15** construction. This book holds only
the last 2h20m, so the two are not the same risk and must be compared window-for-window.

| Scope | n | HOLD win | **worst /lot** | **worst @6L** | p01 /lot |
|---|---|---|---|---|---|
| **r/118, DTE0, full day** | 127 | 66% (34% losers) | **≈ −21,500** | ≈ −129,000 | — |
| SENSEX DTE0 13:00→15:20, 2024→2026 | 131 | **61.8%** (38% losers) | −12,612 | −₹75,674 | −11,858 |
| SENSEX all days 13:00→15:20, 2021→2026 | 1,356 | 58.3% | **−20,781** | **−₹124,684** | −13,286 |
| NIFTY all days 13:00→15:20, 2015→2026 (incl COVID) | 2,754 | 54.9% | **−21,644** | **−₹129,862** | −11,754 |
| SENSEX DTE0, worst intraday MTM (not booked) | 131 | — | −18,979 | −₹113,871 | p99 −18,656 |

**They reconcile, and the reconciliation is instructive.** The loser rate matches almost
exactly (38% here vs r/118's 34%). The *worst day* looks 40% smaller — **but only because the
SENSEX DTE0-labelled sample is short**: SENSEX weekly expiries only exist in our data from 2024,
so 131 days that miss the 2021–22 regime and the 2023-02-01 Adani session. As soon as the sample
is deepened — SENSEX all-days back to 2021, or NIFTY back to 2015 with COVID in it — **the
afternoon window alone reaches −₹20,800 to −₹21,600 per lot, the same magnitude as r/118's
full-day figure.**

**Which is right? Both, and the deeper samples are the operative ones.** r/118 is not
over-stating the tail; the short DTE0 window is under-stating it, for exactly the reason Arun
gave. **Plan the book against ≈ −₹21,000/lot = −₹126,000 at 6 lots unstopped**, and note that
even that is a *floor* — it is bridged from index moves and carries no IV-pop term.

Against that, **L = 0.50 caps the modelled worst at −₹4,972/lot on the DTE0 scope and
−₹5,135/lot on the all-days scope — around −₹30,000 at 6 lots.** That is the backstop's real
job, and it does it.

---

## 8. What the level actually costs — the correction Arun should carry

The worked example in the commission — credit 231.63, level 347.44, "caps the loss near
−₹13,900 at 6 lots" — is the **naive** arithmetic. It omits overshoot, the 2-poll dwell, and the
measured forced-exit slippage.

| L | level (combined) | naive /lot | **median /lot** | p90 /lot | **median @6L** | **p90 @6L** |
|---|---|---|---|---|---|---|
| 0.40 | 324.28 | 1,853 | 2,403 | 3,105 | ₹14,417 | ₹18,630 |
| **0.50 (live)** | **347.44** | 2,316 | **2,866** | 3,568 | **−₹17,198** | **−₹21,410** |
| 0.60 | 370.61 | 2,780 | 3,330 | 4,032 | ₹19,978 | ₹24,191 |
| 1.00 | 463.26 | 4,633 | 5,184 | 5,886 | ₹31,102 | ₹35,315 |

**At today's credit the 50% backstop caps the loss at about −₹17,200 at 6 lots on a typical
fire and −₹21,400 on a bad one — roughly 24–54% worse than the −₹13,900 the arithmetic
suggests.** And in the worst gapped case measured on the long sample it would not have held at
all.

---

## 9. Robustness, and the seven deadly sins

| Sin | Control |
|---|---|
| **Look-ahead** | every measurement is intrabar and causal; the vol model uses a trailing-20-day window that **excludes today**; DTE labels come from an expiry calendar, not from what expired |
| **Survivorship** | index-level study, no selection |
| **Overfitting / multiple testing** | 15 arms declared before compute; Šidák bar stated (|t| ≈ 3.30); **no arm clears it and none is claimed to**; the recommendation is *keep the incumbent*, which requires no significance |
| **Cost neglect** | measured outcome-aware model throughout (+6.548 pt/leg-side on a forced exit, +0.178 on time, full Zerodha rate card); the study's headline correction is *itself* a cost finding |
| **Regime dependence** | four independent scopes across two venues and two vol regimes; credit carried as an explicit axis (C25/CMED/C75/VOL); COVID included via NIFTY |
| **Correlation / single factor** | the SENSEX and NIFTY samples agree on the *shape* of the curve independently |
| **Capacity / shortability** | not applicable — this is a 6-lot index-option book, and the study recommends no size change |

**OOS split (Stage A, 2026-06-30 midpoint):** every tight arm (0.25 / 0.30 / 0.35 / 0.40) is
negative in **both halves**. 0.50 and wider are not — H1 −380/lot, H2 exactly 0 (no fires in H2).
Small n, but directionally consistent with the long sample.

**Honest limitations.**
1. **n = 17** on the recorded chain. Nothing on Stage A is significant and nothing should be
   read as such.
2. The **vol credit-model has r = −0.03 in-sample.** It is presented as one rung of a
   sensitivity, never as the primary; the fixed credit ladder is shown alongside it everywhere.
3. **HOLD is modelled at terminal intrinsic** on the long sample, which ignores the residual
   time value still in the straddle at 15:20 (10 minutes to settlement). This *flatters* HOLD,
   so the backstop's measured advantage at 0.50 is if anything understated.
4. The **SENSEX DTE0 sample begins in 2024** — 131 days. The deeper tail is carried by the
   all-days and NIFTY scopes, and that is stated wherever a tail number is quoted.
5. Every bridged tail is a **floor**: no IV-pop term. r/122 already found an observed worst that
   exceeded its own bridged p99.

---

## 10. Recommendation, for Arun's sign-off

**Keep `BACKSTOP = 0.50` in `research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py`.
Change nothing before Thursday 2026-09-03.**

- **0.50 is the corner of the expectation curve** — the tightest level at which the cost of
  carrying the stop has decayed to zero, reproduced independently in four samples. Tightening it
  to 0.40 costs the book and buys only ~₹5,000 of worst-day at 6 lots; tightening to 0.25 costs
  ₹4,034/lot-equivalent at 6 lots and is the one arm in the study that is significantly *bad*.
- **Budget the cap at −₹17,200 at 6 lots**, not −₹13,900 — and −₹21,400 on a bad fire.
- **Accept that it is not a disaster stop.** It fires on ~35–43% of DTE0 afternoons over the
  long run and **38% of those breaches gap through it**. The genuine disaster protection on this
  book is what r/131 already identified: **the −₹3,000/lot venue book stop, plus size.** The
  combined-premium backstop is the second line, and it is correctly placed.
- **The unstopped tail to plan against is ≈ −₹21,000/lot = −₹126,000 at 6 lots**, per the deep
  samples and r/118 — not the −₹75,674 the short DTE0 window suggests.
- If Arun wants a hard rupee ceiling on high-credit days, **HYB60_3500** (exit at the lesser of
  60% of credit and ₹3,500/lot) is the only hybrid worth discussing — it improves the SENSEX
  worst day by ~₹4,200 at 6 lots for ~₹80/lot of expectation. It is **not** recommended on the
  evidence; it does not clear the family-wise bar and it is worse on the NIFTY tail.

**Arun's underlying instinct was right and is now documented: the recorded options window is a
calm slice with no disaster in it, and the multi-year sample must carry the tail. Having done
that, the level it points to is the one already deployed.**

---

## Files

| File | Purpose |
|---|---|
| `scripts/stage_a_chain_backstop.py` | recorded 1-min chain replay, all levels × 2 dwell models, + r/114 reconciliation |
| `scripts/stage_b_longsample.py` | multi-year 13:00→15:20 excursion clock, expiry-calendar DTE labels |
| `scripts/analyse_backstop.py` | bridge, fire rates, save/cost, gap-through, tail, recommendation |
| `results/analysis.txt` | the full numeric output behind every table above |
| `results/stage_a_days.csv`, `stage_a_levels.csv` | per-session and per-arm recorded outcomes |
| `results/stage_b_days.csv` | 4,110 day × window rows across both venues |
| `results/bridge.csv`, `bridge_validation.csv` | the bridge and its 17-day validation |
| `results/fire_rates.csv`, `save_cost.csv`, `long_save_cost.csv` | §4, §5 |
| `results/gap_through.csv`, `gap_through_long.csv` | §6 |
| `results/tail.csv`, `r114_reconciliation.txt` | §7, §1 |
