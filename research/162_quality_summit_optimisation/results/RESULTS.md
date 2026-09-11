# research/162 — Quality Summit optimisation, and the quality screen inside Open Alpha · Base Age

**Window 2018-08-01 → 2026-09-10 (8.1 years), split into a FIT window W1 (2018-08-01 →
2022-06-30) and a HOLDOUT W2 (2022-07-01 → 2026-09-10) pre-registered before any cell ran.
193 cells (157 Part A, 36 Part B) plus 14 Part-C blend constructions. After tax (20% STCG /
12.5% LTCG, Indian FY loss-netting), net of 25 bps a side, idle cash 5.0% (Quality Summit) /
5.5% (Base Age), 12 rebalance-day offsets or 30 selection seeds per cell, medians across
paths. Drawdowns measured from the running peak of the full curve.**

---

## The verdicts

| Part | Question | Verdict |
|---|---|---|
| **A** | Can Quality Summit be made to earn more and fall less? | **CONCLUDED — no adoption.** The best improvement found (+6.06pp CAGR and +0.28 Calmar on **12 of 12** offsets in the fit window) **reverses sign in the holdout**: −3.48pp CAGR on 3 of 12, −0.15 Calmar on 1 of 12. The incumbent spec survives unchanged. |
| **B** | Does the quality screen help INSIDE Open Alpha · Base Age? | **NO EDGE.** Every screen loses, on **0 of 30 seeds**, in both windows, under both missing-data policies. The 10-Oct-2026 review registered by r/160 is closed: the answer is no. |
| **C** | Does either book belong in the portfolio? | **DILUTIVE — do not add.** Holding **cash** in its place is better on **360 of 360 paths** at every weight. Monthly correlation to Base Age is **0.717**: it is the same family, sampled worse. |

**Nothing here is recommended for deployment, paper or live. The one operational
consequence is negative: it removes a dated obligation rather than creating one.**

---

## Part A — Quality Summit optimisation: **CONCLUDED, no adoption**

### The incumbent, reproduced bit-identically before anything was changed

Quality Summit is research/160's Family-B `b7` book: close ≥ 0.90 × its own causal
all-time-high close, 20-day median traded value ≥ ₹2 cr, the point-in-time Screener screen
(profitable in the last 3 filed fiscal years, 3-year average ROE > 15%, ROCE > 15% or a
lender, 3-year sales **and** profit growth > 10%, market cap > ₹1,000 cr, **no** debt test),
top 15 by relative strength, rebalanced monthly, next-open fills, **no exit rule**.

r/162's engine is generated from r/160's frozen engine by an auditable patch script
(`scripts/patch_engine.py`, 14 exact-string patches). Run through it, the incumbent returns
**21.19% / −37.07% / Calmar 0.58** — r/160's published row to the second decimal. The
rebuilt eligibility masks reproduce r/160's own coverage index exactly too (`growth_only`
162.6 names mean / 276 in 2026-09; `arun_strict` 26.4 / 46). Nothing downstream rests on a
re-derivation that drifted.

| | after tax | MaxDD (worst path) | Calmar | % invested | trades/yr |
|---|---:|---:|---:|---:|---:|
| **QS incumbent — full window** | **21.19%** | −37.07 (−43.08) | **0.58** | 91.2 | 61.4 |
| QS incumbent — W1 (fit) | 20.11% | −34.43 (−38.14) | 0.61 | 85.1 | 49.2 |
| QS incumbent — W2 (holdout) | 20.95% | −35.54 (−43.48) | 0.59 | 97.3 | 73.1 |

The incumbent is the most *stable* thing in this study: 20.11% in the fit window, 20.95% in
the holdout, a 0.84pp gap. Everything that beat it in the fit window failed to repeat.

### A1 — ATR-scaled trailing exits: the family r/160 never tried, and it does not rescue the book

research/161 found that on all-time-high entries the **exit** was worth +11.85pp of CAGR and
that the exit that did it was a SuperTrend(14,4) close trail. r/160 had only tested
price-level exits (simple-moving-average trails, fixed peak-drawdown stops, Donchian lows,
time stops, hard stops) and concluded that nothing beat holding. This is the missing family,
on the incumbent construction, measured on W1:

| exit on the incumbent book | CAGR W1 | MaxDD | Calmar | % invested | trades/yr |
|---|---:|---:|---:|---:|---:|
| **none (the incumbent)** | **20.11** | −34.43 | **0.61** | 85.1 | 49.2 |
| SuperTrend(20, 3) close trail | 19.14 | −28.15 | 0.67 | 63.6 | 72.6 |
| SuperTrend(10, 3) close trail | 19.28 | −29.06 | 0.65 | 65.1 | 70.2 |
| SuperTrend(14, 4) close trail (r/161's winner) | 18.47 | −33.79 | 0.56 | 72.3 | 59.3 |
| SuperTrend(7, 3) close trail | 17.47 | −29.96 | 0.60 | 64.6 | 72.6 |
| chandelier, 22-day high − 3×ATR14 | 14.46 | −26.91 | 0.52 | 57.0 | 90.1 |
| chandelier, 22-day high − 2×ATR14 | 4.74 | −13.78 | 0.32 | 34.0 | 129.8 |
| `fund_fail` (sell when the name stops passing the screen) | 20.11 | −34.43 | 0.61 | 85.1 | 49.2 |

**Read this the right way.** The best trail buys 6.3 points of drawdown for 1.0 point of
CAGR — a real trade, and worth +0.06 Calmar — but it is nowhere near the pre-registered
+0.15, and it does it partly by **sitting in cash**: the book's invested fraction falls from
85% to 64%, so a chunk of the "improvement" is the 5% idle-cash yield rather than the equity
engine. The chandelier at 2×ATR is a demonstration of the same thing taken to absurdity:
−13.8% drawdown on 34% invested is the drawdown of a cash pile, not of a strategy.

**`fund_fail` does literally nothing.** Every paired row is identical to the row without it.
A name that has passed this screen almost never stops passing it between monthly
rebalances — and when it does, the rebalance ranking has usually dropped it already.

**The trail's value is construction-dependent, which is the trap.** At the incumbent's
15 slots and k=0.90 the SuperTrend(20,3) trail *helps* Calmar. At the concentrated
construction found in A2-A5 it *destroys* the book: 17.06% against 24.48% without it. An
exit tuned under one book is not an exit for a different book — the same interaction r/158
found when a trail that won under one gate lost once the gate was retired.

### A2 — ranking axes: relative strength is the ranking, and nothing fundamental comes close

Every candidate ranking axis, on the incumbent's qualifying set, W1, three slot counts:

| rank | N=8 | N=10 | N=15 |
|---|---:|---:|---:|
| **relative strength (incumbent)** | **21.80** | **21.78** | **20.11** |
| 3-year net-profit growth (`profit_g3`) | 12.71 | 14.07 | 15.57 |
| 3-year operating-margin slope (`opm_slope3`) | 19.99 | 19.80 | 17.49 |
| composite z(RS) + z(profit growth) | 15.49 | 16.83 | 16.92 |
| composite z(RS) + z(margin slope) | 21.47 | 19.03 | 19.42 |
| market cap, largest first (shares-constant **proxy**) | **−0.07** | 2.73 | 7.06 |

Ranking the *same* qualifying names by any fundamental measure destroys between 2 and 22
points of CAGR. Blending a fundamental into the RS score dilutes it rather than sharpening
it. This is r/160's central finding — *the return in this family comes from relative
strength, not from the fundamentals* — re-confirmed from a new direction: it is not only
that the screen adds little; it is that **using the fundamentals to choose among the
survivors actively subtracts.**

### A3 — sizing: inverse volatility is worth about 1 point, and only where the book is already concentrated

Inverse-60-day-volatility targets, capped at 2× and floored at 0.25× the equal-weight
target, re-based so the top-N candidates average exactly equal weight:

| | CAGR W1 | MaxDD | Calmar |
|---|---:|---:|---:|
| incumbent, equal weight | 20.11 | −34.43 | 0.61 |
| incumbent, inverse-vol | 19.81 | −33.68 | 0.63 |
| concentrated (k 0.85, N 10), equal weight | 24.48 | −32.03 | 0.80 |
| concentrated (k 0.85, N 10), inverse-vol | **25.59** | −30.91 | **0.83** |

**The sector-cap axis was dropped, and this is why:** there is no sector field anywhere in
this project's data. `fundamentals.db` carries none, the 2,116 cached Screener pages carry
only the annual / quarterly / top-ratio blocks, and `holdings_meta.db` covers only currently
held names. Faking one from ticker heuristics would have been worse than not running it.

### A4 / A5 — the fit-window winner, its plateau, and the holdout that killed it

The one combination that clears the pre-registered bar **in the fit window** is
**wider band + fewer names**: keep the `b7` screen, relax the near-all-time-high band from
k = 0.90 to **k = 0.85**, cut the book from 15 names to **10**, size inverse-vol. Call it
**QS-v2**.

**The plateau is real** — 24 cells of k × N on W1, so this is not a lone spike:

| k \ N | 8 | 10 | 12 | 15 |
|---|---:|---:|---:|---:|
| 0.80 | 21.75 | 23.44 | 23.68 | 21.43 |
| 0.825 | 22.51 | 23.00 | **24.09** | 22.17 |
| **0.85** | 24.13 | **24.48** | 23.53 | 21.80 |
| 0.875 | 21.65 | 22.31 | 21.92 | 21.35 |
| 0.90 | 21.80 | 21.78 | 21.50 | **20.11** *(incumbent)* |
| 0.95 | 20.03 | 20.53 | 19.81 | 17.53 |

Every neighbour of the winner sits inside ±3pp, and the traded-value dial is flat too
(₹2 cr 24.48, ₹5 cr 24.37; ₹10 cr breaks it at 18.00). It looked like a genuine finding.

**Then the holdout was opened, once, as pre-registered.**

| | W1 (fit) | W2 (holdout) | gap | full window |
|---|---:|---:|---:|---:|
| QS incumbent | 20.11% / −34.43 / 0.61 | 20.95% / −35.54 / 0.59 | **+0.84** | 21.19% / −37.07 / 0.58 |
| **QS-v2** (k 0.85, N 10, inverse-vol) | **25.59% / −30.91 / 0.83** | **16.37% / −40.92 / 0.41** | **−9.22** | 21.43% / −40.93 / 0.53 |
| QS-v2 equal-weight | 24.48% / −32.03 / 0.80 | 14.78% / −42.15 / 0.35 | −9.70 | 20.91% / −41.72 / 0.50 |
| QS-v2 plateau neighbour (k 0.825, N 12) | 24.09% / −32.77 / 0.74 | 16.67% / −42.00 / 0.43 | −7.42 | 21.24% / −41.80 / 0.52 |

The pre-registered rule was: *a cell whose W2 falls more than 4pp below its W1 is declared
not robust.* All three finalists fail it by a wide margin, and they fail it together — so
this is the **construction**, not the sizing detail.

**The paired test, same offset on both sides, is the cleanest statement of it:**

| QS-v2 vs the incumbent | ΔCAGR (median) | CAGR wins | ΔCalmar | Calmar wins | ΔMaxDD | DD wins |
|---|---:|---:|---:|---:|---:|---:|
| **W1 (fit)** | **+6.06pp** | **12/12** | **+0.282** | **12/12** | +4.94pp | 10/12 |
| **W2 (holdout)** | **−3.48pp** | **3/12** | **−0.152** | **1/12** | −4.43pp | 0/12 |
| full window | +0.01pp | 6/12 | −0.055 | 2/12 | −3.93pp | 1/12 |

A 12-of-12 sweep in the fit window and a 1-of-12 rout in the holdout is the signature of a
fit to the 2020-21 leg, not of an edge. Over the whole window QS-v2 buys **0.01 points of
CAGR for 3.9 extra points of drawdown**.

**Two further facts finish it.** QS-v2's trade-level compounding proxy with its **ten best
trades deleted falls to 0.636× — below one, i.e. it loses money without ten names** (332
trades). The incumbent's stands at 355×. And QS-v2's own worst path (19.14% W1) is carried
by a book that, on the holdout, has a 17-trade losing streak.

### Does the SCREEN add value? Yes on drawdown, no on return — at both constructions

Paired against the identical book run on the **screenable sub-universe** (every name with
four filed fiscal years at that date and nothing else required) — the honest control, since
it carries the same coverage:

| | ΔCAGR | wins | ΔCalmar | wins | ΔMaxDD | wins |
|---|---:|---:|---:|---:|---:|---:|
| incumbent construction (k 0.90, N 15) — full window | −1.32pp | 2/12 | **+0.106** | 10/12 | **+12.74pp** | **12/12** |
| candidate construction (k 0.85, N 10) — full window | +4.75pp | 9/12 | **+0.244** | 11/12 | **+16.94pp** | **12/12** |

The first row reproduces r/160's published paired result exactly (−1.32 / +0.106 / 10 of
12). The screen's product is **drawdown insurance, not return** — it takes 13 to 17 points
off the maximum drawdown on every single offset, and pays for it in CAGR. That is a
legitimate product; it is just not the product Arun asked for ("earn more"), and it does not
clear a +0.15 Calmar bar at the construction that survives the holdout.

### Robustness on the finalists

| | 25 bps | 40 bps | 60 bps | 0% cash yield | missing = pass |
|---|---:|---:|---:|---:|---:|
| QS incumbent | 21.19 | 20.25 | 19.10 | 21.01 | 21.19 |
| QS-v2 | 21.43 | 20.54 | 19.19 | 21.15 | 21.43 |

Both survive the cost ladder with the same shallow slope (~1.1pp per 15 bps), neither is a
cash-yield artefact (91-97% invested), and the missing-data policy changes nothing — the
Screener panel covers 2,131 of 2,158 universe names, so this study's coverage bias is small
and measured rather than assumed.

### Tradeability gate — full window, after tax

| book | win % | avg win | avg loss | expectancy/trade | max losing streak | trades/yr | turnover | capacity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| QS incumbent | 46.3 | +27.9% | −10.8% | +7.26% | **17** | 61.4 | 4.2× | 0.7% of held names' median traded value at ₹1 cr |
| QS-v2 | 45.9 | +33.6% | −12.3% | +9.00% | **17** | 38.3 | 3.6× | 1.2% at ₹1 cr |
| screenable-sub-universe control | 42.4 | +32.1% | −13.8% | +5.79% | 16 | 56.4 | 5.1× | 1.4% at ₹1 cr |

A 17-trade losing streak on a book that trades 38-61 times a year is roughly four to six
months of nothing but losers. Multiply the capacity ratios by the real book size before
reading them as a constraint.

---

## Part B — the quality screen INSIDE Open Alpha · Base Age: **NO EDGE**

This is r/160's closing recommendation, registered there as a review dated **10-Oct-2026**
and answered here instead.

**The engine is r/161's, byte-identical** (`bt_core162.py`, md5 verified against
`research/161/scripts/bt_core.py`). The rebuilt panel's no-mask control reproduces r/161's
published winner exactly — **21.26% CAGR, −34.80% drawdown, Calmar 0.618, worst seed 19.87%,
687 trades, 31.7 trades/yr over 2005-2026** — before any masked cell was read.

**The only change** is one line in the event filter: a candidate all-time-high close is
dropped unless its symbol passes the eligibility mask on the **signal day**. Exits, sizing,
slot contention, costs and the 60-bar re-arm are untouched, and the mask is applied *before*
the re-arm, which is r/161's own convention.

### The result, on 2018-08 → 2026-09 (where the fundamentals exist), 30 seeds

| entry filter | CAGR | worst seed | MaxDD | Calmar | % invested | events | trades/yr |
|---|---:|---:|---:|---:|---:|---:|---:|
| **none — Base Age as it stands** | **26.57%** | 23.07 | **−26.57%** | **0.999** | **87.0** | 3,619 | 41.9 |
| + `b7` (Quality Summit's own screen), missing = fail | 16.79% | 16.78 | −22.39% | 0.750 | 63.3 | 468 | 30.0 |
| + `b7`, missing = pass | 16.38% | 16.35 | −21.72% | 0.754 | 63.6 | 1,148 | 30.2 |
| + `b3` (quality only, no growth test), fail | 15.02% | 14.59 | −30.69% | 0.491 | 76.0 | 771 | 35.5 |
| + `b3`, pass | 15.70% | 15.34 | −31.02% | 0.506 | 76.5 | 1,451 | 35.7 |
| + `growth_only` (growth > 20, nothing else), fail | 20.77% | 20.77 | −20.50% | 1.013 | 53.9 | 388 | 24.5 |
| + `growth_only`, pass | 21.60% | 21.59 | −20.95% | 1.031 | 54.4 | 1,068 | 24.8 |
| + `arun_strict` (the screen as written), fail | 8.80% | 8.80 | −10.10% | 0.872 | **18.8** | 76 | 9.2 |
| + `arun_strict`, pass | 11.08% | 11.07 | −12.87% | 0.861 | **24.9** | 756 | 12.0 |

### The paired test — same seed on both sides, against the no-mask control

Pre-registered bar: **≥ +0.10 Calmar or −3pp of drawdown at ≥ equal CAGR, on ≥ 20 of 30
seeds, in BOTH windows.**

| overlay | window | ΔCAGR | CAGR wins | ΔCalmar | Calmar wins | verdict |
|---|---|---:|---:|---:|---:|---|
| `b7`, fail | 2018-08→ | **−9.71pp** | **0/30** | −0.248 | **0/30** | fail |
| `b7`, fail | W1 | −16.16pp | 0/30 | −0.407 | 0/30 | fail |
| `b7`, fail | W2 | −3.58pp | 2/30 | −0.037 | 9/30 | fail |
| `b7`, pass | 2018-08→ | −10.09pp | 0/30 | −0.245 | 0/30 | fail |
| `b3`, fail | 2018-08→ | −11.28pp | 0/30 | −0.478 | 0/30 | fail |
| `growth_only`, fail | 2018-08→ | −5.80pp | 0/30 | +0.014 | 19/30 | fail |
| `growth_only`, pass | 2018-08→ | −4.84pp | 0/30 | +0.039 | 22/30 | fail (W1 is −0.121, 0/30) |
| `arun_strict`, fail | 2018-08→ | −17.77pp | 0/30 | −0.127 | 0/30 | fail |

**Not one screen wins on a single seed out of thirty on return, in any window, under either
missing policy.** `growth_only` is the only overlay that gets within touching distance on
Calmar in the full sub-window (+0.039 on 22 of 30) — and it loses 4.8 points of CAGR to buy
it, and it is negative on Calmar in W1 on 0 of 30. That is not a pass on any reading of the
pre-registered bar.

### Why — and it is not subtle

The screen **starves the book**. Base Age has 16 slots and needs a flow of signals to fill
them; the screen cuts the qualifying events from **3,619 to 468** (`b7`) to **76**
(`arun_strict`), and the invested fraction falls from 87% to 63% to **19%**. What is left
is a smaller sample of the same signal plus a large cash balance. `arun_strict`'s W1 Calmar
of 2.50 is arithmetically true and completely meaningless: it is six trades in four years
against a 6.9%-invested book — the Calmar of a cash pile, which is exactly the artefact
r/160's engine diagnostics were built to catch.

The long-window arm (2005 → 2026) is reported for completeness and is **labelled a coverage
artefact**: the fundamentals panel starts in 2015 and is only usable from Aug-2018, so
thirteen of twenty-one years are decided by the missing policy alone. It says the same thing
anyway — `b7` fail 9.57%, pass 16.46%, against the control's 21.26%.

**The 10-Oct-2026 review is closed with a NO.** The screen is a drawdown filter that costs
too much return to be worth applying to a book whose drawdown is already −26.6%.

---

## Part C — portfolio fit against the honest pair: **DILUTIVE, do not add**

r/160 ran this test against the **published** Open Alpha. r/159 has since shown that book's
headline rests on a same-bar look-ahead fill, so r/160's blend table is unplaceable. Re-run
here against the pair Arun's money is actually in: **True North + Open Alpha · Base Age,
50-50, rebalanced monthly.** 360 paths (Base Age seed × True North offset, the r/154
convention), monthly returns, 2018-08 → 2026-08.

| book | CAGR | [min..max] | MaxDD | Calmar | ΔCalmar vs pair | paths improved |
|---|---:|---:|---:|---:|---:|---:|
| **TN + Base Age 50-50 (the honest pair)** | **24.42** | [19.00..27.37] | **−13.71** | **1.769** | — | — |
| + QS incumbent at 10% | 24.12 | [19.48..27.11] | −13.98 | 1.742 | +0.001 | 181/360 |
| + QS incumbent at 20% | 23.74 | [19.93..26.80] | −15.03 | 1.565 | −0.188 | 70/360 |
| + QS incumbent at 33% | 23.15 | [20.47..26.35] | −17.06 | 1.357 | −0.389 | 27/360 |
| + QS-v2 at 10% | 24.36 | [19.37..27.41] | −13.48 | 1.834 | +0.054 | 237/360 |
| + QS-v2 at 20% | 24.31 | [19.70..27.38] | −14.13 | 1.717 | −0.064 | 125/360 |
| + QS-v2 at 33% | 24.02 | [20.07..27.26] | −17.06 | 1.383 | −0.343 | 8/360 |
| **+ CASH at 10% (the null)** | 22.49 | [17.67..25.11] | −12.21 | 1.826 | **+0.052** | **360/360** |
| **+ CASH at 20% (the null)** | 20.55 | [16.32..22.86] | −10.74 | 1.900 | **+0.118** | **360/360** |
| **+ CASH at 33% (the null)** | 18.04 | [14.55..19.94] | −8.87 | 2.040 | **+0.238** | **360/360** |
| True North standalone | 21.58 | [14.51..25.60] | −16.03 | 1.380 | −0.431 | 0/360 |
| Base Age standalone | 25.40 | [22.28..27.43] | −19.46 | 1.317 | −0.434 | 34/360 |
| QS incumbent standalone | 21.02 | [17.04..23.77] | −31.10 | 0.671 | −1.135 | 0/360 |
| QS-v2 standalone | 21.16 | [17.36..25.56] | −37.03 | 0.574 | −1.163 | 0/360 |

**Cash dominates at every weight, on every path.** Whatever risk reduction a Quality Summit
sleeve supplies, plain cash at the same weight supplies more of it and keeps more Calmar.
That is the same test that killed the whole mean-reversion third-sleeve family in r/146, and
it gives the same answer here.

### Why: it is the same bet

| pair | monthly | daily |
|---|---:|---:|
| QS incumbent vs Base Age | **0.717** | 0.671 |
| QS-v2 vs Base Age | **0.651** | 0.649 |
| QS incumbent vs True North | 0.374 | 0.440 |
| True North vs Base Age | 0.337 | 0.430 |

A complement wants to sit below ~0.40 monthly. Quality Summit sits at **0.72 against Base
Age** — it is a near-all-time-high momentum book being added to a book of all-time-high
breakouts. The two legs already in the pair are 0.337 to each other; the candidate is twice
that to one of them.

### Stress windows — return % (intra-window drawdown from the full-curve peak)

| window | the pair | + QS incumbent 20% | + QS-v2 20% | + CASH 20% |
|---|---:|---:|---:|---:|
| 2020 crash (Feb-Apr) | −10.0 (−10.0) | −10.2 (−11.1) | −9.1 (−10.4) | −7.8 (−7.8) |
| 2022H1 grind (Jan-Jun) | −11.8 (−11.8) | **−14.8 (−14.9)** | −12.2 (−12.6) | −9.1 (−9.1) |

It does not earn in the grind and it does not cushion the crash — it deepens both. The 2022
column is the decisive one: the pair's known weakness is the grind, and this sleeve makes it
3 points worse.

---

## The house year-on-year table

The full table (with the three index benchmarks and the best-of columns) is
`results/yoy162.md` / `.html` / `.csv`. Summary row, each column on its own window:

| | QS incumbent | QS-v2 (W1 fit) | Base Age | Base Age + b7 overlay | True North | TN+BA pair | NIFTYBEES |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CAGR / MaxDD / Calmar** | 21.2 / −37.1 / 0.58 | 21.4 / −40.9 / 0.53 | **26.2 / −26.6 / 0.99** | 16.8 / −22.4 / 0.75 | 21.4 / −21.2 / **1.01** | **24.4 / −19.9 / 1.24** | 10.6 / −36.3 / 0.29 |

**A convention note that matters:** the pair's drawdown reads **−19.9% on daily marks**
(this table) and **−13.71% on monthly marks** (Part C). Both are correct; they are different
measurements. Blend tables in this project are built on monthly returns (r/154 convention),
year tables on daily curves. Quote the daily number when someone asks what the book felt
like to hold.

Two things the year rows show that the summary hides:

- **QS-v2's "improvement" is one leg.** It beats the incumbent in 2021 (+89.1 vs +87.2) and
  2022 (+4.1 vs −19.1) — both inside the fit window — and loses in 2023 (+45.2 vs +59.4),
  2024 (+17.7 vs +35.9) and 2025 (−17.8 vs −16.1), all inside the holdout.
- **The `b7` overlay's best years on Base Age are the years it was barely invested.** Its
  −0.6% in 2018 and its shallow drawdowns are a cash balance, not a filter working.

---

## Caveats — read these before the numbers

1. **Eight years is short and it contains the 2023-25 smallcap boom.** The window cannot be
   extended: Screener serves ~12 fiscal years, so four filed years do not exist for most
   names until FY2018 is filed in Aug-2018 (coverage steps 7% → 87% at that date). The
   holdout W2 is four years and is dominated by one regime; a candidate could fail it for
   regime reasons rather than for being overfit. The pre-registered rule was applied as
   written regardless — but a four-year holdout is a four-year holdout.
2. **The price universe is not point-in-time.** `market_data.db` keeps only 102 stopped
   series in 2,158 (4.7%) across eleven years, fewer than NSE actually delisted. Survivorship
   pressure is **upward on every arm, benchmarks included**. The controls (screenable
   sub-universe; random selection) carry the identical bias, which is why the *paired*
   numbers are the ones quoted.
3. **`market_data.db` is not retroactively split-adjusted.** The near-ATH state restarts its
   `cummax` on a one-day collapse below 0.55×, and this study's new trail frames truncate a
   symbol's series after the last single-day fall worse than −40% (106 symbols affected).
   Both guards also fire on genuine crashes, which makes the near-ATH state *easier* to
   satisfy for those names. Direction stated, not hidden.
4. **The fundamentals are restated, not as-reported.** The filing lag controls *when* a year
   becomes visible; it cannot undo a later restatement. That is the residual look-ahead, and
   it flatters every screened arm — including the ones that lost.
5. **The market-cap ranking axis is a proxy** (r/142's shares-constant snapshot, current
   share count back-projected on historical price). Its −0.07% result is so far from
   everything else that the proxy cannot be the explanation, but it is a proxy.
6. **The SuperTrend and chandelier frames use forward-fill-within-span**, so a missing
   session counts as a repeated bar inside the ATR window rather than being skipped. Holes
   are rare after the phantom-row purge; the effect is immaterial, but it is a choice.
7. **193 cells were run** (157 Part A, 36 Part B) plus 14 Part-C blend constructions.
   Discount the fit-window winner accordingly — which is precisely what the holdout did.
   The number is disclosed so the discount can be applied by the reader, not asserted by us.
8. **The Part-C cash-yield inconsistency is real and uncorrected:** True North's curve carries
   idle cash at 6.5% p.a. (r/159), Base Age at 5.5% (r/161), Quality Summit at 5.0% (r/160).
   The pair is flattered by a few tenths of a point relative to the candidate — the bias runs
   **against** the candidate, which is the safe direction.
9. **What was NOT tested, and why.** A sector cap (no sector data exists anywhere in the
   project). A point-in-time index-membership universe (not reconstructable). Intraday or
   stop-based exits (the engine is close-only by construction, which is also why r/142's
   trigger/fill trap cannot be expressed in it). Screens built from quarterly figures
   (Screener carries ~13 quarters, so they only exist from ~mid-2023).

---

## What this changes

1. **Quality Summit keeps the spec research/160 published.** k = 0.90, 15 names, relative
   strength, monthly, no exit. It was not improved. It is, notably, the most window-stable
   book in the study (20.11% fit / 20.95% holdout), and that stability is the reason nothing
   beat it out of sample.
2. **The 10-Oct-2026 overlay review is closed early, with a NO.** Do not spend a slot on it
   in October. Quality screens do not belong inside Open Alpha · Base Age; they cut its
   signal flow by 87% and take ten points of CAGR with them.
3. **The quality screen's real product is drawdown, not return** — 13 to 17 points off the
   maximum drawdown on 12 of 12 offsets, at a cost of 1.3 points of CAGR. If Arun ever wants
   a lower-drawdown version of a near-ATH momentum book, the screen is the honest way to get
   it. It is not a way to earn more.
4. **Nothing is added to the portfolio.** A cash sleeve beats a Quality Summit sleeve on 360
   of 360 paths at every weight, and the correlation to Base Age (0.72 monthly) says why.
5. **A method note worth keeping.** This study is a clean example of the fit/holdout split
   earning its keep: a 12-of-12 offset sweep with a +6pp CAGR uplift and a verified plateau
   looked like a discovery, and the holdout turned it into a 1-of-12 loss. Without the
   pre-registered W1/W2 split and the 4pp rule written down **before** the run, this would
   have been published as an improvement.

---

## Files

| file | what |
|---|---|
| `scripts/patch_engine.py` → `scripts/qg_engine2.py` | r/160's frozen engine + ATR trails, fundamental/composite ranking axes, inverse-vol sizing (14 auditable patches) |
| `scripts/build_aux.py` → `results/aux_162.npz` | SuperTrend / chandelier exit-signal frames, 60-day volatility, point-in-time ranking frames |
| `scripts/build_masks162.py` → `results/masks162/` | the growth-12 rung, plus `has_data`, `growth_only` and `arun_strict` rebuilt to r/160's own definitions (verified against its INDEX) |
| `scripts/make_grid162.py`, `results/grid_a*.json` | the Part-A grids |
| `results/cells_a.csv` | Part A, 120 swept cells |
| `scripts/finalize_a.py` → `results/partA_final.csv`, `partA_paired.csv`, `partA_outliers.json` | the G3 package: both windows, cost ladder, paired deltas, outlier deletion |
| `scripts/bt_core162.py` | byte-identical copy of r/161's engine (md5 verified) |
| `scripts/build_panel161.py`, `scripts/partb_overlay.py` | Part B: the mask as an entry filter inside Base Age |
| `results/partB_cells.csv`, `partB_paired.csv`, `partB_navs.npz` | Part B results, per-seed paired deltas, NAV ensembles |
| `scripts/partc_blend.py` → `results/partC_blend.md` | correlation + blend value vs True North + Base Age |
| `results/yoy162.{md,html,csv}` | the house year-on-year table |
| `results/r162_compare.png` | log growth of 100, every book + indices, drawdown panel |
| `results/tearsheet_quality_summit_*.png` | the client factsheet for the surviving incumbent |
