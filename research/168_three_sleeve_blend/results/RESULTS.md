# research/168 — Three-sleeve blend: is the RE-FITTED IPO Base worth more to the book than the INCUMBENT?

## VERDICT: **STRATEGY — adopt the re-fit, fund it at 25%. The INCUMBENT sleeve does not earn a place in the book at all.**

Three answers, in the order they matter.

**1. Yes — IPO-A is worth more to the portfolio than IPO-INC, unanimously.** At the same weight,
the same rebalance, the same 30 paired paths: **+1.73pp of blend CAGR on 30 of 30 paths and
+0.054 Calmar on 30 of 30** at a 25% weight; **+2.42pp and +0.107 Calmar, 30/30** at 35%;
**+3.45pp and +0.267 Calmar, 30/30** at 50%. The sign is unanimous at **every** weight tested
(5% to 50%), on **both** cost bases, under **monthly rebalancing and under pure drift**. There is
no weight and no convention at which the incumbent is the better portfolio citizen.

**2. r/167's correlation-based worry was backwards, and this retracts the inference drawn from
it.** The refit does have higher pairwise correlation to both live legs (monthly: 0.35 to True
North and 0.33 to OA·BaseAge, against the incumbent's 0.26 and 0.32). It is still the better
blend sleeve by a wide margin, because the correlation rise is swamped by the return improvement.
**Pairwise correlation was the wrong screen for this decision** — which is exactly why r/167
refused to answer the question from it and flagged the blend as the adoption blocker.

**3. The incumbent sleeve fails the pre-registered bar at every weight — and it nearly fails the
cash null.** IPO-INC **costs** the blend CAGR at every weight (0 of 30 paths positive,
−0.32pp at 10% to −2.03pp at 50%) and never reaches +0.10 Calmar (its best is **+0.089** at a 30%
weight). Against plain cash **at the same portfolio drawdown** it is worth only **+1.17pp of CAGR
at a 25% weight, +1.25pp at its best**, and it **loses** to cash at 50%. IPO-INC is insurance with
a premium, and the premium is roughly what an arbitrage fund would have paid you for nothing.
**IPO-A, by contrast, beats cash-at-equal-drawdown by +2.52pp at 25% and +4.79pp at 50%, on
30 of 30 paths at every weight.**

**Recommended weight: 25% IPO-A, i.e. True North 37.5% / OA·BaseAge 37.5% / IPO-A 25%**, monthly
rebalanced, leaving the deployed TN:OA ratio at 50:50 of the remainder. That blend is
**21.18% CAGR after tax [worst path 19.17%] / −24.01% MaxDD [worst path −26.39%] / Calmar 0.885**
over 2006-04-03 → 2026-09-03, against the two-sleeve book's **20.28% / −26.91% / 0.749**.

**The honest qualification on the weight.** The Calmar surface does not peak at 25% — it rises
monotonically to a **broad plateau of ~1.05 at 45–60% IPO-A**, and the unconstrained weight
simplex wants TN 35–45% / OA·BaseAge 0–10% / IPO-A 50–60%. That allocation is **not deployable**
and is not recommended: r/167 measured the sleeve's capacity ceiling at roughly **₹20–25 L**, so a
25% weight funds a book up to ₹80–100 L while a 50% weight caps the whole book at ₹40–50 L; the
surface is entirely in-sample with no held-out period; and it would hollow out a live book that
has twenty years of its own evidence. 25% is the low end of a range in which **every** weight from
20% upward clears the pre-registered bar on **all 30 paths**.

Nothing here was deployed by this study. `services/` and `frontend/` were not touched.

---

## 0. Two corrections this study makes to earlier work

**(a) A bug of mine, caught before it reached the report.** The first pass of the blend engine
measured each rebalance period's returns relative to the rebalance day itself rather than the
previous close, which silently discarded the return of **every rebalance day**. It manufactured a
fake "rebalancing frequency effect": monthly rebalancing appeared to cost **1.8pp of CAGR**
against drift, and quarterly appeared to *earn* +1.4pp. The tell was that the frequency response
was non-monotonic — a frequency beating its neighbours on both sides — and that the move was far
too large for the change made. The fixed engine carries a self-test (a 100% single-sleeve blend
must reproduce that sleeve **exactly** under every frequency) and shows the real effect: monthly
rebalancing is worth **+0.54pp of CAGR and +0.06 Calmar over drift**, monotone in frequency, with
phase dispersion under 0.005 of Calmar. **Any "quarterly beats monthly" reading of an interim
number from this session is retracted.**

**(b) The weight-matched cash null is not decision-grade on Calmar.** Cash has zero drawdown, so
Calmar rises without bound as the cash weight rises — 100% cash scores Calmar **infinity**, and a
45% cash sleeve beats a 45% IPO-A sleeve on Calmar for a purely trivial reason. Comparing a
candidate sleeve and cash **at the same weight** therefore stops being informative above about
30%. This study adds the **risk-matched cash null**: for each IPO weight, the cash weight that
reproduces the blend's median max drawdown is solved on a 1% grid, and the two are compared on
CAGR, paired across the 30 paths. That is the test reported as decisive below.

---

## 1. The idle-cash trap, solved before any blend arithmetic

The four candidate curves were produced at **different** idle-cash rates — r/144 True North at
6.5%, r/161 OA·BaseAge at 5.5%, r/167 IPO at 5.0% — and the three books hold **wildly different
cash** (True North 57%, IPO Base 68%, OA·BaseAge 27%). Blending them as produced would have
tilted the weights toward whichever sleeve was credited the richest rate. Every curve in this
study is measured at **5.2% post-tax idle cash**, the arbitrage-fund standard set by
`research/163_mpf_cash_yield_harmonisation` (review dated 15-Dec-2026).

Three reproduction gates were run before a single blend cell:

| sleeve | gate | result |
|---|---|---|
| True North, 12 offsets | offset 0 at 5.2% must reproduce r/163's `cash052` file | **bit-exact**, 0 of 5,066 rows differ, max rel 2.4e-16 |
| OA·BaseAge, 30 seeds | all 30 paths at 25 bps / 5.2% vs r/163's npz | **bit-exact**, max abs diff 0.000e+00 |
| IPO-INC and IPO-A, 30 seeds each | at 5.0% both must reproduce r/167's stage-9 medians | **exact**: 14.90 / −38.57 and 21.80 / −26.63, delta +0.000pp on both |

Both IPO arms are produced by **one** engine (r/167's fork, which carries the name-based fund
exclusion), so INC vs A is apples-to-apples. Moving IPO from 5.0% to 5.2% is worth +0.16pp of
CAGR to both arms, matching the (1 − invested) × 0.2pp rule of thumb at 31.8% / 36.3% invested.

**One basis is NOT harmonised and it is inherited, not introduced:** r/144's True North runs at
`rt = 0.003`, i.e. **15 bps a side**, while OA·BaseAge and IPO Base both run at **25 bps a side**.
Every conclusion below was therefore re-measured on a **harmonised** basis with True North also at
25 bps. The verdict is identical (§5).

### The sleeves, at 5.2% idle cash, on the common window

Common window **2006-04-03 → 2026-09-03, 5,060 trading days, 20.42 years** — the intersection of
True North (2006-04-03→2026-09-03), OA·BaseAge (2005-01-03→2026-09-11) and IPO
(2006-01-02→2026-09-04).

| sleeve | CAGR (median) | worst path | MaxDD (median) | worst path | Calmar | WA 2006-15 | WB 2016-26 | 2008 | 2020 H1 | 2018 | 2022 H1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| True North (15 bps) | 19.63% | 14.96% | −25.99% | −28.57% | 0.755 | 15.38% | 23.28% | −14.1% | −3.4% | −7.4% | −6.3% |
| True North (25 bps) | 18.41% | 13.74% | −27.60% | −29.21% | 0.656 | 14.28% | 21.97% | −14.3% | −3.8% | −8.6% | −6.5% |
| OA · Base Age | 19.92% | 18.98% | −34.05% | −36.92% | 0.584 | 16.82% | 22.85% | −29.9% | +2.8% | −27.3% | −17.7% |
| **IPO-INC** | 15.09% | 13.27% | −38.52% | −46.38% | 0.381 | 14.58% | 15.73% | **+0.6%** | +28.7% | −7.0% | −14.5% |
| **IPO-A** | 22.08% | 21.09% | −26.57% | −32.76% | 0.826 | 16.61% | 27.30% | −10.1% | +27.4% | −8.7% | −13.3% |
| CASH 5.2% post-tax | 5.11% | 5.11% | 0.00% | 0.00% | — | 5.12% | 5.10% | +5.1% | +2.5% | +5.1% | +2.5% |
| NIFTYBEES | 10.58% | — | −59.71% | — | 0.177 | — | — | −52.1% | — | +4.8% | — |

**Path ensemble.** Blend path `p` (1..30) = IPO seed `p`, OA·BaseAge seed `p`, True North offset
`(p−1) mod 12`. Every comparison in this study is **paired on p**. True North is deterministic
(no seed variance), so its 12 rebalance-day offsets are cycled 2–3 times across the 30 paths —
the 30 paths are therefore not 30 fully independent draws on the True North leg, which is stated
again in the caveats.

---

## 2. The weight curve, end to end

Published basis, monthly rebalance, TN:OA held at 50:50 of the remainder (the deployed ratio).
Medians of 30 paired paths. Every drawdown is measured from the running peak of the **full**
curve (the r/154 convention).

| IPO weight | IPO-A CAGR | MaxDD | Calmar | worst-path CAGR | worst-path DD | IPO-INC CAGR | MaxDD | Calmar |
|---|---|---|---|---|---|---|---|---|
| 0% (two-sleeve) | 20.28% | −26.91% | 0.749 | 17.89% | −30.42% | 20.28% | −26.91% | 0.749 |
| 10% | 20.64% | −25.72% | 0.804 | 18.46% | −28.76% | 19.93% | −25.57% | 0.787 |
| 15% | 20.81% | −25.18% | 0.830 | 18.71% | −27.92% | 19.76% | −24.71% | 0.808 |
| 20% | 21.00% | −24.62% | 0.858 | 18.95% | −27.07% | 19.59% | −23.94% | 0.825 |
| **25%** | **21.18%** | **−24.01%** | **0.885** | **19.17%** | **−26.39%** | 19.40% | −23.26% | 0.837 |
| 30% | 21.30% | −23.42% | 0.912 | 19.39% | −25.80% | 19.18% | −22.58% | **0.849** |
| 35% | 21.42% | −22.81% | 0.944 | 19.60% | −25.20% | 18.96% | −22.39% | 0.842 |
| 50% | 21.71% | −21.16% | **1.019** | 20.14% | −23.41% | 18.19% | −24.31% | 0.747 |

The two shapes are the whole story. **IPO-A raises CAGR and lowers drawdown at the same time, at
every weight** — the Calmar curve is monotone up to a plateau at 45–60% (peak 1.056 at 60% on the
full simplex) and only rolls over above that. **IPO-INC lowers drawdown by giving up return**: its
Calmar peaks at 0.849 at a 30% weight and collapses beyond 40%.

Extending the axis past 33% mattered: 33% was the edge of the first grid and IPO-A was still
improving there, so the first pass had not shown the optimum to be interior. It is — at 45–60%,
on a plateau flat to within 0.01 of Calmar.

---

## 3. The decisive paired comparisons

30 paired paths; "wins" = paths on which the first arm is better. Published basis, monthly
rebalance, TN:OA 50:50 of the remainder.

### 3.1 IPO-A versus IPO-INC, same weight — **the headline**

| weight | Δ CAGR | CAGR wins | Δ MaxDD | DD shallower | Δ Calmar | Calmar wins |
|---|---|---|---|---|---|---|
| 20% | +1.38pp | **30/30** | −0.69pp | 2/30 | +0.034 | **30/30** |
| 25% | +1.73pp | **30/30** | −0.65pp | 2/30 | +0.054 | **30/30** |
| 30% | +2.08pp | **30/30** | −0.31pp | 7/30 | +0.081 | **30/30** |
| 33% | +2.28pp | **30/30** | −0.16pp | 13/30 | +0.095 | **30/30** |
| **35%** | **+2.42pp** | **30/30** | **+0.03pp** | 15/30 | **+0.107** | **30/30** |
| 40% | +2.76pp | **30/30** | +0.36pp | 24/30 | +0.141 | **30/30** |
| 45% | +3.11pp | **30/30** | +1.05pp | 28/30 | +0.193 | **30/30** |
| 50% | +3.45pp | **30/30** | +2.48pp | 29/30 | +0.267 | **30/30** |

**Against the pre-registered bar (b)** — "+0.10 Calmar, or +2pp CAGR at no worse drawdown, on
≥20 of 30 paths" — the honest reading is: **the sign is unanimous at every weight, and the
pre-registered magnitude is first met at a 35% weight** (+0.107 Calmar on 30/30, with drawdown
also equal-or-better). At 20–30% the refit wins on every path but by less than the threshold:
+0.05 to +0.08 Calmar, with the blend's drawdown 0.3–0.7pp *deeper* than the incumbent's. The
refit buys return; the incumbent buys a little more drawdown protection and pays far too much for
it.

### 3.2 Each arm versus the two-sleeve TN + OA·BaseAge baseline

| weight | IPO-A Δ CAGR (wins) | Δ DD | Δ Calmar (wins) | IPO-INC Δ CAGR (wins) | Δ DD | Δ Calmar (wins) |
|---|---|---|---|---|---|---|
| 10% | +0.39pp (30/30) | +1.30pp | +0.053 (30/30) | −0.32pp (**0/30**) | +1.83pp | +0.041 (30/30) |
| 15% | +0.58pp (30/30) | +1.95pp | +0.079 (30/30) | −0.49pp (**0/30**) | +2.72pp | +0.060 (30/30) |
| **20%** | +0.75pp (30/30) | +2.61pp | **+0.107 (30/30)** | −0.68pp (**0/30**) | +3.28pp | +0.075 (30/30) |
| 25% | +0.91pp (30/30) | +3.21pp | **+0.134 (30/30)** | −0.88pp (**0/30**) | +3.91pp | +0.088 (30/30) |
| 30% | +1.05pp (30/30) | +3.84pp | **+0.163 (30/30)** | −1.09pp (**0/30**) | +4.38pp | **+0.089 (30/30)** |
| 35% | +1.19pp (30/30) | +4.44pp | **+0.193 (30/30)** | −1.31pp (**0/30**) | +4.42pp | +0.086 (30/30) |
| 50% | +1.54pp (30/30) | +5.96pp | **+0.268 (30/30)** | −2.03pp (**0/30**) | +3.15pp | +0.008 (16/30) |

**Pre-registered bar (a) — "does the sleeve belong at all?"**
- **IPO-A: CLEARS from a 20% weight**, on 30 of 30 paths, on both legs simultaneously (it adds
  CAGR *and* +0.10 Calmar, which the bar did not even require jointly).
- **IPO-INC: FAILS at every weight.** It never reaches +0.10 Calmar (best +0.089) and it never
  satisfies "at ≥ equal CAGR" — it loses CAGR on **0 of 30** winning paths at every single weight.
  It is a real drawdown-reducer (30/30 at every weight up to 40%) but that is the only thing it is.

### 3.3 The risk-matched cash null — the test that kills the incumbent

For each IPO weight, the **cash** weight reproducing the same median portfolio drawdown is solved
on a 1% grid; the two books are then compared on CAGR, paired across 30 paths.

| arm | IPO weight | blend CAGR | blend MaxDD | matched cash weight | cash-blend CAGR | cash-blend MaxDD | **Δ CAGR at equal risk** | wins |
|---|---|---|---|---|---|---|---|---|
| INC | 10% | 19.93% | −25.57% | 5% | 19.55% | −25.52% | **+0.41pp** | 30/30 |
| INC | 20% | 19.59% | −23.94% | 11% | 18.67% | −23.98% | **+0.92pp** | 30/30 |
| INC | 25% | 19.40% | −23.26% | 14% | 18.22% | −23.22% | **+1.17pp** | 30/30 |
| INC | 30% | 19.18% | −22.58% | 16% | 17.93% | −22.70% | **+1.25pp** | 30/30 |
| INC | 40% | 18.70% | −22.65% | 16% | 17.93% | −22.70% | +0.80pp | 30/30 |
| INC | 50% | 18.19% | −24.31% | 10% | 18.81% | −24.24% | **−0.55pp** | **5/30** |
| **A** | 10% | 20.64% | −25.72% | 4% | 19.70% | −25.80% | **+0.98pp** | 30/30 |
| **A** | 20% | 21.00% | −24.62% | 8% | 19.11% | −24.75% | **+1.92pp** | 30/30 |
| **A** | **25%** | **21.18%** | **−24.01%** | 11% | 18.67% | −23.98% | **+2.52pp** | **30/30** |
| **A** | 35% | 21.42% | −22.81% | 16% | 17.93% | −22.70% | **+3.54pp** | 30/30 |
| **A** | 50% | 21.71% | −21.16% | 22% | 17.04% | −21.13% | **+4.79pp** | 30/30 |

Read the INC rows plainly: **taking the same drawdown risk, the incumbent IPO sleeve adds about
one percentage point of CAGR over simply parking that slice of the book in an arbitrage fund —
and at a 50% weight it adds nothing at all.** That is the same fingerprint that killed the
mean-reversion third sleeve in r/146 and the Quality-Summit blends in r/160 and r/162: the
"diversifier" is mostly de-levering. The refit escapes it by a factor of two to four.

### 3.4 Does the answer survive the conventions?

| check | A 25% vs INC 25% | A 25% vs two-sleeve | INC 25% vs two-sleeve |
|---|---|---|---|
| published basis, monthly | +1.73pp CAGR 30/30, +0.054 Calmar 30/30 | +0.91pp 30/30, +0.134 Calmar 30/30 | −0.88pp 0/30, +0.088 Calmar |
| **harmonised** basis (TN also 25 bps), monthly | +1.72pp 30/30, +0.051 Calmar 30/30 | +1.05pp 30/30, +0.138 Calmar 30/30 | −0.72pp 1/30, +0.090 Calmar |
| **drift** (never rebalanced) | +1.64pp 30/30, +0.094 Calmar 30/30 | +0.75pp 29/30, +0.177 Calmar 30/30 | — |
| **WA 2006–2015** only | — | +0.38pp, 24/30 (16.87 vs 16.48) | −0.25pp, 4/30 (16.24 vs 16.48) |
| **WB 2016–2026** only | — | +1.38pp, **30/30** (25.25 vs 23.97) | −1.41pp, **0/30** (21.92 vs 23.97) |

Both halves agree on both verdicts. The refit helps in WA and WB; the incumbent hurts in WA and
WB. Neither conclusion is a regime artifact, and neither depends on the frictionless-rebalancing
assumption — under pure drift the refit's advantage is, if anything, larger.

---

## 4. The 2008 question r/167 left open

r/167 flagged that the refit is **10.7pp worse than the incumbent in 2008** standalone (−10.3%
versus +0.4%) and warned "if the pair needs a 2008 cushion, this is not it". Measured **inside the
blend**, that gap shrinks to 2.5pp and both arms *improve* on the two-sleeve book:

| 30-path MEDIAN return over the window | 2008 | 2020 H1 | 2018 | 2022 H1 |
|---|---|---|---|---|
| two-sleeve TN + OA·BaseAge 50:50 | **−22.25%** | −0.03% | −17.46% | −11.98% |
| + IPO-INC 25% | **−16.78%** | +7.28% | −14.74% | −12.49% |
| + IPO-A 25% | **−19.28%** | +6.98% | −15.20% | −12.28% |
| + IPO-A 50% | −16.26% | +13.92% | −12.99% | −12.56% |
| + CASH 25% | −16.28% | +5.60% | −14.24% | −11.34% |

So the crash-cushion argument does **not** rescue the incumbent: at blend weight it is worth 2.5pp
of 2008 — and plain cash at the same weight delivers the same 2008 cushion as IPO-INC does
(−16.28% vs −16.78%), for free. Note also that the worst hole in this pair's twenty years is
still 2008, and it is −22.3% on the two-sleeve book: deeper than the −17.0% figure carried for the
old TN + OA·ATH pair, because OA·BaseAge replaced the old Open Alpha.

---

## 5. The per-year house table

Published basis, monthly rebalance, TN:OA 50:50 of the remainder. Each cell is the **annual
return** with the **intra-year max drawdown beneath it, measured from the running peak of the FULL
curve** (r/154). Columns are the **median-CAGR path** of each ensemble; the summary row carries
the **30-path median** CAGR / MaxDD / Calmar. Best-of columns exclude NIFTYBEES. All after tax
(20% STCG / 12.5% LTCG with Indian FY loss netting), net of 25 bps a side, idle cash 5.2%
post-tax.

| year | TN | OA BaseAge | IPO-INC | IPO-A | TN+OA 50:50 | +IPO-INC 25% | +IPO-A 25% | +IPO-A 50% | +CASH 25% | NIFTYBEES | BEST CAGR | LEAST DD | BEST OVERALL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2006 | +8.4<br><sub>(-11.9)</sub> | +8.3<br><sub>(-20.6)</sub> | +52.6<br><sub>(-10.9)</sub> | +69.1<br><sub>(-10.1)</sub> | +8.7<br><sub>(-14.8)</sub> | +18.9<br><sub>(-12.0)</sub> | +22.2<br><sub>(-12.0)</sub> | +32.0<br><sub>(-10.9)</sub> | +7.6<br><sub>(-10.9)</sub> | +16.1<br><sub>(-29.9)</sub> | IPO-A | IPO-A | IPO-A |
| 2007 | +81.6<br><sub>(-15.6)</sub> | +84.7<br><sub>(-10.8)</sub> | +60.0<br><sub>(-11.7)</sub> | +51.6<br><sub>(-15.9)</sub> | +85.0<br><sub>(-11.7)</sub> | +67.6<br><sub>(-10.7)</sub> | +65.4<br><sub>(-12.4)</sub> | +70.7<br><sub>(-12.3)</sub> | +61.9<br><sub>(-8.9)</sub> | +53.0<br><sub>(-14.9)</sub> | TN+OA 50:50 | +CASH 25% | OA BaseAge |
| 2008 | -15.6<br><sub>(-21.5)</sub> | -29.9<br><sub>(-32.6)</sub> | +0.6<br><sub>(-12.7)</sub> | -10.1<br><sub>(-15.8)</sub> | -22.9<br><sub>(-25.7)</sub> | -15.9<br><sub>(-19.9)</sub> | -18.4<br><sub>(-21.4)</sub> | -16.5<br><sub>(-19.6)</sub> | -16.3<br><sub>(-19.0)</sub> | -52.1<br><sub>(-59.7)</sub> | IPO-INC | IPO-INC | IPO-INC |
| 2009 | +52.3<br><sub>(-22.4)</sub> | +62.8<br><sub>(-31.7)</sub> | +1.6<br><sub>(-5.9)</sub> | +4.0<br><sub>(-12.5)</sub> | +58.2<br><sub>(-25.8)</sub> | +53.8<br><sub>(-18.6)</sub> | +54.6<br><sub>(-20.8)</sub> | +30.1<br><sub>(-18.1)</sub> | +43.2<br><sub>(-18.5)</sub> | +75.6<br><sub>(-59.1)</sub> | OA BaseAge | IPO-INC | +IPO-INC 25% |
| 2010 | +5.5<br><sub>(-20.4)</sub> | +15.5<br><sub>(-15.7)</sub> | +12.4<br><sub>(-10.2)</sub> | +21.1<br><sub>(-9.8)</sub> | +10.7<br><sub>(-14.2)</sub> | +13.3<br><sub>(-9.6)</sub> | +15.3<br><sub>(-10.3)</sub> | +16.5<br><sub>(-7.5)</sub> | +9.4<br><sub>(-10.8)</sub> | +18.6<br><sub>(-25.0)</sub> | IPO-A | +IPO-A 50% | IPO-A |
| 2011 | -12.4<br><sub>(-18.6)</sub> | -10.5<br><sub>(-18.6)</sub> | +13.7<br><sub>(-5.5)</sub> | -2.4<br><sub>(-10.2)</sub> | -11.2<br><sub>(-17.7)</sub> | -5.0<br><sub>(-11.3)</sub> | -8.6<br><sub>(-14.7)</sub> | -6.4<br><sub>(-11.6)</sub> | -7.3<br><sub>(-12.6)</sub> | -24.1<br><sub>(-27.3)</sub> | IPO-INC | IPO-INC | IPO-INC |
| 2012 | +22.2<br><sub>(-16.3)</sub> | +28.2<br><sub>(-19.5)</sub> | +0.8<br><sub>(-4.4)</sub> | +11.9<br><sub>(-9.8)</sub> | +25.3<br><sub>(-17.3)</sub> | +21.9<br><sub>(-10.2)</sub> | +25.0<br><sub>(-14.1)</sub> | +18.6<br><sub>(-10.8)</sub> | +20.1<br><sub>(-12.0)</sub> | +26.5<br><sub>(-26.0)</sub> | OA BaseAge | IPO-INC | +IPO-INC 25% |
| 2013 | +0.6<br><sub>(-12.1)</sub> | +4.5<br><sub>(-9.3)</sub> | +5.3<br><sub>(-2.7)</sub> | +4.8<br><sub>(-0.4)</sub> | +2.6<br><sub>(-9.8)</sub> | +1.5<br><sub>(-7.4)</sub> | +1.4<br><sub>(-7.5)</sub> | +2.6<br><sub>(-4.0)</sub> | +3.3<br><sub>(-6.7)</sub> | +7.2<br><sub>(-16.0)</sub> | IPO-INC | IPO-A | IPO-A |
| 2014 | +38.0<br><sub>(-12.0)</sub> | +49.7<br><sub>(-7.3)</sub> | +5.2<br><sub>(+0.0)</sub> | +5.2<br><sub>(+0.0)</sub> | +44.5<br><sub>(-9.3)</sub> | +39.8<br><sub>(-7.2)</sub> | +39.8<br><sub>(-7.2)</sub> | +31.9<br><sub>(-3.7)</sub> | +33.8<br><sub>(-6.9)</sub> | +31.6<br><sub>(-6.2)</sub> | OA BaseAge | IPO-INC | OA BaseAge |
| 2015 | -3.3<br><sub>(-10.9)</sub> | -3.6<br><sub>(-23.5)</sub> | +5.6<br><sub>(-9.5)</sub> | +27.7<br><sub>(-9.6)</sub> | -2.6<br><sub>(-12.4)</sub> | -0.6<br><sub>(-11.9)</sub> | +4.5<br><sub>(-9.9)</sub> | +11.0<br><sub>(-9.5)</sub> | -0.6<br><sub>(-9.0)</sub> | -4.3<br><sub>(-15.0)</sub> | IPO-A | +CASH 25% | IPO-A |
| 2016 | +31.1<br><sub>(-11.8)</sub> | +7.3<br><sub>(-29.5)</sub> | +53.2<br><sub>(-12.7)</sub> | +75.8<br><sub>(-10.6)</sub> | +19.0<br><sub>(-15.9)</sub> | +23.5<br><sub>(-14.8)</sub> | +27.8<br><sub>(-10.3)</sub> | +43.1<br><sub>(-7.2)</sub> | +15.5<br><sub>(-11.4)</sub> | +4.0<br><sub>(-21.6)</sub> | IPO-A | +IPO-A 50% | IPO-A |
| 2017 | +31.7<br><sub>(-10.1)</sub> | +65.2<br><sub>(-13.2)</sub> | +32.2<br><sub>(-9.9)</sub> | +72.3<br><sub>(-10.2)</sub> | +46.3<br><sub>(-8.0)</sub> | +40.4<br><sub>(-6.6)</sub> | +46.8<br><sub>(-6.7)</sub> | +58.1<br><sub>(-6.6)</sub> | +35.1<br><sub>(-5.7)</sub> | +29.9<br><sub>(-8.5)</sub> | IPO-A | +CASH 25% | IPO-A |
| 2018 | -8.3<br><sub>(-23.1)</sub> | -30.5<br><sub>(-35.6)</sub> | -6.1<br><sub>(-20.6)</sub> | -9.1<br><sub>(-19.2)</sub> | -18.4<br><sub>(-28.0)</sub> | -12.6<br><sub>(-21.0)</sub> | -12.9<br><sub>(-21.7)</sub> | -13.5<br><sub>(-21.7)</sub> | -12.8<br><sub>(-20.9)</sub> | +4.8<br><sub>(-14.1)</sub> | IPO-INC | IPO-A | IPO-INC |
| 2019 | -2.7<br><sub>(-26.0)</sub> | +30.1<br><sub>(-35.9)</sub> | +9.5<br><sub>(-13.2)</sub> | +5.4<br><sub>(-15.6)</sub> | +12.8<br><sub>(-28.9)</sub> | +15.1<br><sub>(-21.1)</sub> | +14.0<br><sub>(-22.2)</sub> | +9.4<br><sub>(-21.4)</sub> | +10.9<br><sub>(-21.3)</sub> | +13.6<br><sub>(-10.5)</sub> | OA BaseAge | IPO-INC | IPO-INC |
| 2020 | +66.6<br><sub>(-25.9)</sub> | +48.8<br><sub>(-23.2)</sub> | +69.1<br><sub>(-9.8)</sub> | +74.8<br><sub>(-13.2)</sub> | +58.3<br><sub>(-22.0)</sub> | +60.0<br><sub>(-8.5)</sub> | +62.0<br><sub>(-10.6)</sub> | +67.0<br><sub>(-14.7)</sub> | +43.6<br><sub>(-14.4)</sub> | +15.4<br><sub>(-36.3)</sub> | IPO-A | +IPO-INC 25% | IPO-A |
| 2021 | +61.8<br><sub>(-11.2)</sub> | +84.3<br><sub>(-10.9)</sub> | +2.1<br><sub>(-19.2)</sub> | +52.5<br><sub>(-14.5)</sub> | +74.7<br><sub>(-7.2)</sub> | +46.5<br><sub>(-9.6)</sub> | +66.2<br><sub>(-8.2)</sub> | +61.3<br><sub>(-9.7)</sub> | +54.4<br><sub>(-5.4)</sub> | +26.0<br><sub>(-9.5)</sub> | OA BaseAge | +CASH 25% | OA BaseAge |
| 2022 | +16.1<br><sub>(-14.1)</sub> | -0.8<br><sub>(-24.3)</sub> | -8.7<br><sub>(-32.3)</sub> | +0.9<br><sub>(-21.2)</sub> | +6.2<br><sub>(-16.8)</sub> | -3.4<br><sub>(-21.7)</sub> | -0.3<br><sub>(-18.5)</sub> | +4.0<br><sub>(-17.9)</sub> | +6.2<br><sub>(-12.3)</sub> | +5.5<br><sub>(-16.1)</sub> | TN | +CASH 25% | TN |
| 2023 | +52.3<br><sub>(-11.0)</sub> | +59.8<br><sub>(-15.4)</sub> | +34.8<br><sub>(-36.7)</sub> | +46.5<br><sub>(-11.0)</sub> | +52.7<br><sub>(-10.2)</sub> | +41.6<br><sub>(-18.0)</sub> | +50.6<br><sub>(-10.0)</sub> | +55.4<br><sub>(-6.4)</sub> | +39.6<br><sub>(-7.4)</sub> | +21.0<br><sub>(-9.7)</sub> | OA BaseAge | +IPO-A 50% | +IPO-A 50% |
| 2024 | +24.2<br><sub>(-18.2)</sub> | -1.3<br><sub>(-26.1)</sub> | +1.9<br><sub>(-19.3)</sub> | +13.6<br><sub>(-22.6)</sub> | +15.1<br><sub>(-19.1)</sub> | +15.5<br><sub>(-13.3)</sub> | +14.9<br><sub>(-16.3)</sub> | +12.7<br><sub>(-15.6)</sub> | +12.9<br><sub>(-14.4)</sub> | +10.4<br><sub>(-10.5)</sub> | TN | +IPO-INC 25% | TN |
| 2025 | +4.7<br><sub>(-17.3)</sub> | +2.0<br><sub>(-21.9)</sub> | -27.4<br><sub>(-35.6)</sub> | +1.9<br><sub>(-17.9)</sub> | +4.2<br><sub>(-11.5)</sub> | -3.7<br><sub>(-15.0)</sub> | +4.3<br><sub>(-9.8)</sub> | +2.5<br><sub>(-12.3)</sub> | +4.5<br><sub>(-8.1)</sub> | +11.7<br><sub>(-15.2)</sub> | TN | +CASH 25% | +CASH 25% |
| 2026 | +7.7<br><sub>(-8.7)</sub> | +27.6<br><sub>(-23.6)</sub> | +46.2<br><sub>(-41.8)</sub> | -0.2<br><sub>(-24.0)</sub> | +18.2<br><sub>(-10.7)</sub> | +29.1<br><sub>(-13.2)</sub> | +13.1<br><sub>(-12.1)</sub> | +7.9<br><sub>(-18.2)</sub> | +14.6<br><sub>(-7.8)</sub> | -7.7<br><sub>(-14.8)</sub> | IPO-INC | +CASH 25% | +IPO-INC 25% |
| **full** | **19.63**<br><sub>-26.0 / 0.76</sub> | **19.92**<br><sub>-34.0 / 0.58</sub> | **15.09**<br><sub>-38.5 / 0.38</sub> | **22.08**<br><sub>-26.6 / 0.83</sub> | **20.28**<br><sub>-26.9 / 0.75</sub> | **19.40**<br><sub>-23.3 / 0.84</sub> | **21.18**<br><sub>-24.0 / 0.89</sub> | **21.71**<br><sub>-21.2 / 1.02</sub> | **16.59**<br><sub>-20.3 / 0.83</sub> | **10.58**<br><sub>-59.7 / 0.18</sub> | | | |
Reading the BEST OVERALL column: **IPO-A wins 7 of the 21 years** (2006, 2010, 2013, 2015, 2016,
2017, 2020) and **IPO-INC wins 4** (2008, 2011, 2018, 2019), with the +IPO-INC 25% blend taking
3 more (2009, 2012, 2026). They win in *different kinds* of year: the incumbent owns the pain
years, the refit owns the expansion years. That is the same trade-off the paired tables quantify,
and the blend arithmetic says the expansion years are worth more than the pain years cost.

---

## 6. Cost sensitivity

All three sleeves moved together (True North's published 15 bps a side at the 25 bps column,
40 and 60 bps on every sleeve thereafter). Medians of 30 paired paths.

| blend | 25 bps | 40 bps | 60 bps | CAGR lost, 25→60 |
|---|---|---|---|---|
| TN + OA·BaseAge 50:50 | 20.28% / −26.91% / 0.749 | 18.34% / −28.57% / 0.634 | 16.71% / −29.96% / 0.551 | −3.57pp |
| + IPO-INC 25% | 19.40% / −23.26% / 0.837 | 17.51% / −24.47% / 0.706 | 15.72% / −25.90% / 0.599 | −3.68pp |
| **+ IPO-A 25%** | **21.18% / −24.01% / 0.885** | **19.47% / −25.27% / 0.768** | **17.92% / −26.41% / 0.665** | **−3.26pp** |
| + IPO-A 50% | 21.71% / −21.16% / 1.019 | 20.31% / −22.08% / 0.913 | 18.67% / −23.73% / 0.796 | −3.04pp |

The recommended blend is the **most** cost-tolerant of the three-sleeve options, because IPO-A is
a low-turnover book (18.9 trades a year held a median 37 days). The ranking does not change at any
cost level.

---

## 7. Correlations

Monthly returns, median path, common window.

| | TN | OA·BaseAge | IPO-INC | IPO-A | NIFTYBEES |
|---|---|---|---|---|---|
| TN | 1.000 | 0.442 | 0.259 | 0.348 | 0.394 |
| OA·BaseAge | 0.442 | 1.000 | 0.319 | 0.329 | 0.484 |
| IPO-INC | 0.259 | 0.319 | 1.000 | 0.717 | 0.146 |
| IPO-A | 0.348 | 0.329 | 0.717 | 1.000 | 0.164 |
| NIFTYBEES | 0.394 | 0.484 | 0.146 | 0.164 | 1.000 |

Daily and weekly matrices are in `results/correlations.json`; the ordering is the same at all three
frequencies. IPO-A's correlation to True North is **0.348 monthly / 0.255 weekly**, still well
inside the < ~0.4 complement threshold this project uses, and its correlation to NIFTYBEES (0.164)
is the lowest of the three sleeves. The recommended blend's own correlation to its legs is 0.81
(TN), 0.81 (OA·BaseAge), 0.63 (IPO-A) and 0.52 (NIFTYBEES).

---

## 8. The unconstrained simplex — reported, not recommended

The full 231-combination weight simplex in 5% steps, with every rebalance frequency run over **all
of its phases** (25 frequency/phase combinations, 34,650 cells over two cost bases), ranks by
phase-median Calmar:

| third sleeve | best weights | freq | CAGR | MaxDD | Calmar | phase range |
|---|---|---|---|---|---|---|
| **IPO-A** | TN 45 / OA 0 / **IPO 55** | monthly | 21.63% | −20.57% | **1.063** | single phase |
| IPO-A | TN 35 / OA 5 / IPO 60 | half-yearly | 21.72% | −20.59% | 1.060 | 1.053–1.064 |
| IPO-A | TN 45 / OA 5 / IPO 50 | monthly | 21.63% | −20.61% | 1.060 | single phase |
| IPO-INC | TN 60 / OA 10 / IPO 30 | half-yearly | 19.02% | −22.20% | 0.865 | 0.854–0.886 |
| CASH | TN 0 / OA 0 / CASH 100 | any | 5.11% | 0.00% | **inf** | — |

Harmonised basis: IPO-A's best is TN 25 / OA 10–15 / IPO 60–65, Calmar **1.03**; IPO-INC's best is
**0.822**. Same ordering, same conclusion.

Three reasons the 50–60% cell is not the recommendation: **capacity** (the sleeve caps at ~₹20–25 L,
so 55% caps the whole book at ~₹40 L), **no held-out period** (IPO-A's trail and gate were chosen
on this same window in r/167, and these weights were chosen on it again), and it **deletes
OA·BaseAge**, a live book with twenty years of its own evidence, on the strength of one in-sample
Calmar surface. The surface is also flat: 1.05–1.06 across 45–65%, which is noise-width.

**A separate finding that is NOT this study's question and should not be smuggled into it:** the
two-sleeve book itself prefers a TN tilt — TN 85 : OA 15 scores Calmar 0.826 against 50:50's 0.749,
and every top simplex cell pushes OA·BaseAge toward zero. That is a re-weighting of the live pair
and needs its own study; it is registered as such below. All headline figures in this report hold
the deployed 50:50 ratio fixed precisely so that the IPO question is answered on its own.

---

## 9. Honest caveats — what would make this wrong

1. **Inter-sleeve rebalancing is frictionless.** Moving money between sleeves means realising gains
   and paying cost; none of that is charged. The measured rebalancing premium over drift is only
   **+0.54pp of CAGR**, which is the same order as the friction being ignored, so the
   monthly-versus-drift choice should be read as a wash. It does not affect the verdict: every
   conclusion was re-run under pure drift and held (§3.4).
2. **No held-out period anywhere in the chain.** IPO-A's trail, stop and gate were chosen on
   2006–2026 in r/167; the weights here were chosen on the same window. The only
   out-of-sample-ish evidence is the WA/WB split, which both verdicts pass.
3. **Multiple testing.** 840 + 34,650 cells were scored. The Calmar peak of 1.06 should be read as
   its plateau (1.05–1.06 across a 20-point weight band), and the recommended 25% cell's 0.885 as
   the neighbourhood 0.86–0.91.
4. **The cost bases differ by inheritance** (True North 15 bps a side, the others 25). Everything
   was re-measured harmonised and the verdict is unchanged, but the headline table is on the
   published basis so it ties to `/app/mpf-report`.
5. **The True North leg is not 30 independent draws.** It has no seed variance, so its 12
   rebalance-day offsets are cycled 2–3 times across the 30 paths. Path-to-path dispersion in the
   blend is therefore driven mostly by the two slot-constrained sleeves.
6. **Capacity is the binding constraint on the whole recommendation.** r/167 measured the IPO
   sleeve's p90 position at 9.0% of the name's 20-day median traded value on a ₹10 L book — i.e.
   ~90% of a day's volume at ₹1 cr. The sleeve does not scale past roughly ₹20–25 L, which is what
   caps the weight, not the backtest.
7. **Survivorship and the rename defect** carry straight through from r/167: names never onboarded
   to Kite are unmeasurable, and the `LOTUSDEV` → `LOTUSDEV-BE` rename defect hits this book
   hardest. Those are live-signal risks rather than backtest risks, and they are **still not fixed**.
8. **IPO-A's worst-path standalone drawdown is −32.8%, not −26.6%.** In the recommended blend the
   worst path is −26.4% against the two-sleeve book's −30.4%, so the blend improves the unlucky
   path too — but a reader who sees only medians is being misled about it.
9. **What was NOT tested:** a fourth sleeve (gold, per r/147 / r/154); risk-parity or
   volatility-targeted weights rather than fixed weights; a regime-conditional IPO weight; and
   whether the TN:OA re-weighting the simplex wants survives its own scrutiny.

---

## 10. Recommendation

**Fund the re-fitted IPO Base at 25% of the three-sleeve book: True North 37.5% / OA · Base Age
37.5% / IPO-A 25%, rebalanced monthly.**

- 21.18% CAGR after tax [worst path 19.17%] · −24.01% MaxDD [worst path −26.39%] · Calmar 0.885
- against the two-sleeve baseline's 20.28% / −26.91% / 0.749 — **+0.91pp CAGR and +2.9pp of
  drawdown on 30 of 30 paired paths**
- +2.52pp of CAGR over plain cash **at the same drawdown**, on 30 of 30 paths
- 20% is the floor at which the pre-registered bar clears on every path; 35% is where the refit's
  advantage over the incumbent also clears the pre-registered magnitude; **25% is the
  capacity-aware choice inside that band**, fundable on a book up to ₹80–100 L.
- If a paper soak of the live re-fitted book tracks the model, 35% is the justified next step and
  needs no new research — the evidence for it is in §2 and §3 of this file.

**Do not fund the incumbent spec at any weight.** It fails the pre-registered bar at every weight
and is worth about one percentage point of CAGR over an arbitrage fund at equal risk. This matters
because the re-fitted spec has already been deployed to the live paper book by the parent session —
that deployment is the right call, and this study says the *weight* it should eventually be funded
at is 25%, not the 0–15% a correlation-only reading of r/167 would have suggested.

---

## 11. Files

| File | What |
|---|---|
| `THREE_SLEEVE_BLEND_IPO_WEIGHT_DAILY_SWEEP_STATUS.md` | the live status doc / crash-recovery source |
| `scripts/ipo_arms_cash052.py` | IPO-INC + IPO-A, 30 seeds, 5.2% cash, 25/40/60 bps, reproduction gate vs r/167 |
| `scripts/tn_offsets_cash052.py` | True North, 12 offsets x 4 cost levels, bit-exact gate vs r/163 |
| `scripts/ba_costs_cash052.py` | OA·BaseAge at 40/60 bps, bit-exact gate vs r/163 on all 30 paths |
| `scripts/blend_grid.py` | the 840-cell grid, the blend engine + its self-test, paired comparisons, weight-matched cash null |
| `scripts/blend_extend.py` | the 34,650-cell full simplex with rebalance-PHASE ensembles |
| `scripts/final_report.py` | risk-matched cash null, neighbourhood, per-year house table, cost ladder, correlations |
| `scripts/pin_bar.py` | where the pre-registered bar clears; drift / harmonised / WA-WB robustness |
| `results/ipo_navs_cash052.npz`, `tn_navs_cash052.npz`, `ba_navs_cash052.npz` | every sleeve path used |
| `results/blend_grid.csv`, `extend_grid.csv.gz`, `freq_phase_ensemble.csv` | the sweeps |
| `results/paired_published.csv`, `paired_harmonised.csv`, `paired_final.csv` | paired path counts |
| `results/risk_matched_cash_null.csv` | the test that kills the incumbent |
| `results/neighbourhood.csv`, `cost_ladder.csv`, `correlations.json`, `peryear_table.md`, `final_report.json` | the report tables |

Reproduce, in order (VPS, `/home/arun/quantifyd`, `venv/bin/python3`, ~8 minutes total):
`ipo_arms_cash052.py` → `tn_offsets_cash052.py` → `ba_costs_cash052.py` → `blend_grid.py` →
`blend_extend.py` → `final_report.py` → `pin_bar.py`.

**Reproducibility stamp.** `market_data.db` as of 13-Sep-2026 (VPS, canonical). Sleeve engines:
r/144 `tn_attrib_engine`, r/161 `bt_core` via r/163 `ba_cash05`, r/167 `ipo_honest` / r/153
`ipo_replay`. Idle cash 5.2% post-tax on every sleeve. 25 bps a side (True North 15 bps on the
published basis). After tax: 20% STCG / 12.5% LTCG, Indian FY loss netting, inside each sleeve's
own engine. 30 paired paths. Window 2006-04-03 → 2026-09-03.
