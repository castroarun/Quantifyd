# research/159 — Rounding base → shelf breakout near the all-time high

## VERDICT: **SIGNAL, NOT STRATEGY** — and **NO INCREMENTAL VALUE TO THE BOOK**

The pattern Arun drew is **real, reproducible and carries a large per-trade edge**
(+11.45% expectancy per trade, 46.3% win rate, average win +37.9% against average loss
−11.3%). It is not a data artifact: a strictly causal detector reproduces his own KMEW
trade to the day and the rupee, and the shape beats a date-matched control that buys the
same kind of volume thrust near the same all-time highs on the same dates.

It is **not a strategy**, for four independent reasons, and it **should not be added to the
book**, because it dilutes Open Alpha rather than complementing it.

---

## 1. The adoption bar, criterion by criterion

Pre-registered in STATUS §11.1 **before any cell was run**. Best configuration found:
**SuperTrend(14,4) close trail, no hard stop, shelf S=15, volume K=3×, ATH ≥ 0.90,
16 slots @ 6.25% of NAV, ₹10L, 25 bps/side, after tax, 30 seeds, 03-Jan-2005 → 11-Sep-2026.**

| # | Criterion | Result | |
|---|---|---|---|
| 1 | after-tax net CAGR > NIFTYBEES buy-and-hold | **14.60%** vs **12.29%** | **PASS** |
| 2 | max drawdown no worse than NIFTYBEES | **−24.94%** vs **−59.71%** | **PASS** |
| 3 | **≥ 20% after-tax CAGR** (Arun's floor, 30-seed median) | **14.60%** (worst seed 14.57%) | **FAIL** |
| 4 | worst of 30 seeds still beats NIFTYBEES | **14.57%** > 12.29% | **PASS** |
| 5 | **both windows pass** | pre-2016 **8.64%** vs NIFTYBEES **12.68%** | **FAIL** |
| 6 | beats the date-matched near-ATH control | **+4.73pp** (14.60 vs 9.87) | **PASS** |

Two criteria fail. Per the pre-registered labelling rule — *"beats NIFTYBEES but lands below
20% → SIGNAL, not STRATEGY"* — the verdict is **SIGNAL**.

---

## 2. What is genuinely good about it

- **The per-trade edge is large and stable.** Expectancy **+11.45%/trade** over 472 trades
  on the median seed, win rate 46.3%, avg win +37.9% vs avg loss −11.3%.
- **It beats the control that matters.** Against a date-matched "near-ATH volume thrust with
  no saucer and no shelf" control, v3 wins in **10 of 14** exit configurations. The margin
  scales with how much room the exit gives the trade: ST(14,4) **+4.73pp**, ST(10,3) +2.57pp,
  ST(7,3) +2.41pp, EMA50 +1.57pp, Donchian-20 +0.36pp, and it **loses** on the fastest trails
  (Donchian-10 −0.89pp, SMA15 −0.30pp). **The shape's value is that it earns room to run;
  a tight trail throws that away.**
- **It beats random entries comfortably** (CTRL_RND 6.73% vs V3 10.82% on ST(7,3)).
- **Drawdown is less than half the index's** (−24.9% vs −59.7%), and Calmar is ~3× NIFTYBEES
  (0.585 vs 0.206).
- **Costs barely matter**: 14.60 / 14.25 / 13.62% at 25 / 40 / 60 bps — turnover is low
  (21.8 trades/yr), so this is not a cost-fragile edge.

---

## 3. Why it is not a strategy

### 3.1 It is a post-2016 phenomenon
| Window | v3 CAGR | v3 DD | NIFTYBEES CAGR | NIFTYBEES DD |
|---|---|---|---|---|
| pre-2016 | **8.64%** | −24.94% | **12.68%** | −59.71% |
| 2016+ | **20.98%** | −21.06% | 11.86% | −36.34% |

It **loses to the index over the first eleven years** and only clears the 20% floor in the
second window. That is regime dependence, and the pre-registered rule requires both windows.

### 3.2 Extreme outlier dependence
Compounding the median seed's 472 trade returns: **all trades 8.8e11 → top-10 removed 7.0e6**
(a factor of ~125,000) → **winners capped at +50%: 3.4e5**. Ten trades out of 472 carry
essentially the whole result. For contrast, Open Alpha keeps ~90% of its growth rate with its
ten best trades of two decades deleted. **This is a lottery-ticket distribution, not a broad
edge.**

### 3.3 The book cannot be filled, and sizing does not rescue it
889 events over 21.7 years is ~41/year against 16 slots, so the book sits ~40% invested and
the rest earns 5.5% cash. The tell is that the **30-seed CAGR band is 14.57–14.75%** — almost
zero seed variance, because slots are essentially never contended.

Post-hoc (**not pre-registered** — slot count was not in the grid), varying slots only:

| Slots | 4 | 6 | 8 | **10** | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|
| CAGR | 15.09% | 15.87% | 15.76% | **15.96%** | 15.36% | 14.60% | 13.48% |
| MaxDD | −38.8% | −43.5% | −32.6% | −27.2% | −26.4% | −24.9% | −20.6% |

A **plateau at ~15–16%**, never 20%, at any concentration. The 20% floor is not missed by a
sizing choice; the signal does not produce enough opportunity to reach it.

---

## 3.5 The full 2,016-cell sweep — the 20% floor is missed everywhere

The pre-registered grid completed on 11-Sep-2026 (all 2,016 cells, 10-seed scan each).
It does not change the verdict; it removes any remaining doubt about it.

| Question | Answer |
|---|---|
| Cells reaching the **20% CAGR floor** | **0 of 2,016** |
| Best cell in the entire sweep | **14.63%** — `ST(14,4) · no stop · no time stop · shelf 15 · K=3× · ATH≥0.90 · no OBV · no gate` |
| Cells beating NIFTYBEES on **CAGR *and* drawdown** | **22 of 2,016 (1%)** |
| Median cell | **7.34%** · 90th percentile 9.88% · worst 3.72% |

### The exit is a plateau; the entry axes are mostly inert

Median CAGR across every other setting, one axis at a time:

| Axis | Values (median CAGR) | Reading |
|---|---|---|
| **Exit** | **ST(14,4) 9.39** · ST(10,3) 8.06 · ST(7,3) 7.97 · EMA-50 7.91 · Donchian-20 6.76 · Donchian-10 5.56 · SMA-15 5.25 | **A real, monotone plateau: the slower the trail, the better.** All 15 top cells are ST(14,4) |
| ATH proximity | 0.90 → 7.77 · 0.95 → 7.75 · **above ATH → 6.87** | Demanding a *new* high **hurts**; near-the-high is the useful condition, not blue sky itself |
| Volume K | 2× → 7.68 · 3× → 7.21 · 5× → 7.24 | Weak; K mostly trades return for drawdown (K=5 has the best Calmar, 0.618) |
| OBV filter | off 7.94 · **on 6.89** | Costs 1.05pp of CAGR and buys 7.3pp of drawdown — a de-levering filter, not an edge |
| Shelf S | 15 → 7.335 · 20 → 7.350 | **Inert.** The shelf length does not matter once a shelf is required at all |
| Market gate | off 7.355 · on 7.320 | **Inert** — unlike True North, this system gains nothing from a NIFTY-above-100-SMA gate |
| Hard −8% stop | off 7.51 · on 7.21 | Slightly negative on return, mildly positive on drawdown |
| Time stop 120 | off 7.54 · on 7.20 | Negative |

### Neighbourhood of the winner (vary one axis, hold the rest at the winner)

| Axis varied | Neighbours |
|---|---|
| Exit | DON10 5.31 · SMA15 5.23 · DON20 8.98 · EMA50 10.49 · ST(10,3) 10.80 · ST(7,3) 10.82 · **ST(14,4) 14.63** |
| Hard stop | off **14.63** · on 13.33 |
| Time stop | off **14.63** · 120 bars 12.82 |
| Shelf S | 15 **14.63** · 20 13.09 |
| Volume K | 2 → 13.24 · **3 → 14.63** · 5 → 12.74 |
| ATH | **0.90 → 14.63** · 0.95 → 13.53 · 1.00 → 11.80 |
| OBV | off **14.63** · on 10.93 |
| Market gate | off **14.63** · on 12.90 |

**Honest reading of this.** The *exit* result is a plateau and should be trusted: ST(14,4)
wins on the axis median across all 288 of its cells, not just at the winner. The *winner cell
itself* sits about 1.1–1.6pp above its own nearest neighbours on several axes, so some of that
last point of CAGR is selection across 2,016 cells and should be discounted. Neither reading
gets anywhere near 20%: **the best of 2,016 cells is 14.63%, and the 90th percentile cell is
9.88%.**

### 3.4 It dilutes Open Alpha instead of complementing it
Correlation to OA: **daily 0.468, monthly 0.617** — the complement bar is < ~0.4, and this
pattern is by construction a **subset of OA's ATH-breakout entries**.

| Blend (shared window) | CAGR | MaxDD | Calmar |
|---|---|---|---|
| **Open Alpha alone** | **34.90%** | −25.10% | **1.390** |
| OA 90% + v3 10% | 34.24% | −25.02% | 1.368 |
| OA 80% + v3 20% | 33.51% | −24.93% | 1.344 |
| OA 67% + v3 33% | 32.43% | −24.76% | 1.310 |

Every weight is **monotonically worse** than OA alone on both CAGR and Calmar. A mediocre
standalone that lifts the blend is a win; this is a mediocre standalone that **lowers** it.
**On the brief's own rule that is a kill, regardless of the standalone numbers.**

---

## 4. Year by year (after tax, net of 25 bps, 30-seed median; intra-year DD from the full curve's peak)

| Year | v3 book | NIFTYBEES | | Year | v3 book | NIFTYBEES |
|---|---|---|---|---|---|---|
| 2005 | +4.3% (−5.6%) | +32.8% (−14.0%) | | 2016 | −1.7% (−11.8%) | +4.0% (−21.6%) |
| 2006 | +7.2% (−15.8%) | +41.3% (−29.9%) | | 2017 | **+69.5%** (−7.7%) | +29.9% (−8.5%) |
| 2007 | +35.9% (−17.1%) | +53.0% (−14.9%) | | 2018 | −14.6% (−21.1%) | +4.8% (−14.1%) |
| 2008 | **−19.5%** (−24.9%) | **−52.1%** (−59.7%) | | 2019 | +13.5% (−18.9%) | +13.6% (−10.5%) |
| 2009 | +23.2% (−21.5%) | +75.6% (−59.1%) | | 2020 | +25.7% (−10.9%) | +15.4% (−36.3%) |
| 2010 | +8.4% (−9.6%) | +18.6% (−25.0%) | | 2021 | **+86.1%** (−10.0%) | +26.0% (−9.5%) |
| 2011 | −7.5% (−15.8%) | −24.0% (−27.3%) | | 2022 | −5.2% (−18.6%) | +5.5% (−16.1%) |
| 2012 | +5.8% (−15.8%) | +26.5% (−26.0%) | | 2023 | +43.2% (−18.5%) | +21.0% (−9.7%) |
| 2013 | −2.5% (−15.1%) | +7.2% (−16.0%) | | 2024 | +30.5% (−18.2%) | +10.4% (−10.5%) |
| 2014 | +39.0% (−14.9%) | +31.6% (−6.2%) | | 2025 | +13.7% (−14.9%) | +11.7% (−15.2%) |
| 2015 | +14.8% (−9.2%) | −4.3% (−15.0%) | | 2026 | +1.5% (−15.2%) | −9.6% (−14.8%) |
| | | | | **Full** | **14.60% / −24.94% / Calmar 0.585** | **12.29% / −59.71% / 0.206** |

The shape of the record is the finding: **flat-to-poor 2005–2016, then 2017 +69.5% and 2021
+86.1% carrying the whole result.** Note also the genuinely good crash behaviour — 2008
−19.5% against the index's −52.1%, and 2020's −10.9% intra-year against −36.3%.

---

## 5. Tradeability gate

| Metric | Value |
|---|---|
| Trades | 472 (median seed), **21.8/yr** |
| Win rate | **46.3%** |
| Average win / loss | **+37.88% / −11.27%** |
| Expectancy per trade | **+11.45%** |
| Max losing streak | **14** |
| Cost sensitivity | 14.60 / 14.25 / 13.62% at 25 / 40 / 60 bps |
| Capacity | 20-day median traded value ≥ ₹2 cr only — **small-cap heavy, a real capacity wall** |

A 14-trade losing streak on a 21.8-trade-a-year system means roughly **eight months of
nothing but losers** is a normal event. That is very hard to sit through for a book that is
already only 40% invested.

---

## 6. Caveats that travel with every number here

1. **Survivorship.** The universe is symbols present in the DB today; delisted names never
   appear. This bias is unusually sharp for a pattern that *requires* a recovery to an
   all-time high.
2. **Split artifacts.** `market_data.db` is not retroactively split-adjusted. The ATH is
   computed only from bars after the last < −35% one-day move, and 1,399 saucer windows were
   rejected by the split guard — which also discards genuine bases on names that split.
3. **Multiple testing.** The grid is **2,016 cells, all completed**; the winner is discounted
   accordingly. The exit family is a genuine plateau (ST(14,4) leads on the axis median over
   all 288 of its cells), but the winning *cell* sits ~1.1–1.6pp above its own neighbours, so
   part of that last point is selection. **0 of 2,016 cells reach 20%; the median cell is 7.34%.**
4. **One degree of freedom was spent** re-deriving the no-V threshold after v1 failed on
   KMEW (STATUS §3.10 D1), and another aligning the vertex bound (§9.3 D4).
5. **ACCENTMIC-SM, Arun's second example, is absent from the DB** (NSE SME board), so the
   case where SuperTrend(7,3) *held* could never be tested.
6. **The pre-2016 sample is thin** — 12–36 events/year against 143 in 2023 — so the failing
   window is also the least-populated one. That is an explanation, not an excuse: a system
   that cannot trade in half its history is not deployable.
7. **No slippage beyond the 25/40/60 bps ladder**, and no impact model. On ₹2 cr median
   traded value, a ₹10L book taking 6.25% positions is fine; a larger book is not.

---

## 7. What would change the verdict

- **Not** a different exit: the exit family was swept jointly with the entry and ST(14,4) wins
  on a plateau.
- **Not** different sizing: 4→20 slots all land at 13.5–16.0%.
- It would need **more qualifying events per year** (the binding constraint), or evidence
  that the pre-2016 weakness is a data artifact rather than a regime, or a version whose
  correlation to Open Alpha is materially below 0.4 — none of which is in evidence.

**Recommendation: do not deploy, do not paper-trade.** Keep the detector — it is a good
screen and the live-candidate list has standalone value as a watchlist input to the existing
Open Alpha process, which already trades this family better.

---

## ADDENDUM, 11-Sep-2026 18:55 IST — the "dilutes Open Alpha" limb is WEAKENED, not withdrawn

A parallel study, `research/159_oa_honest_reoptimization` (a different session, running the
same evening), found that **Open Alpha’s published ~34.9% CAGR — and research/142’s 40.8% —
rests on a same-bar look-ahead fill**: the signal is `close > pivot` and the fill is
`max(pivot, open)` **on that same bar**, i.e. the entry price is taken from a bar whose close
is what generated the signal. Their own sweep marks that cell `placeable: NO`
(`open_same_REFERENCE`, 43.97% CAGR in their stage A), and their **placeable** short-trail
cells come out **negative** (−2.25% to −2.54% CAGR); with much longer trails their honest
cells reach roughly **18–23%** (their stage A2 `close_same` trail-75 cell: 23.20% CAGR,
−45.4% drawdown, Calmar 0.502, 39.3% win rate — and their stage B figures are **pre-tax**).

**What this does to section 3.4 of this document.** The blend test here compared *this*
book — honest next-open fills on both legs, after tax — against the **research/154 Open Alpha
NAV curve**, which inherits that look-ahead. So the comparison was **an honest book against an
inflated one**, and the conclusion that adding this sleeve "dilutes Open Alpha at every
weight" is **not safe as stated**. Against an honest OA curve, this sleeve’s relative
standing would improve, possibly materially.

**What this does NOT change.** The verdict rests on two failures that never touch Open Alpha:

- **criterion 3** — 14.60% after-tax CAGR against Arun’s **20% floor**, with **0 of 2,016
  cells** reaching it and a 4-to-20-slot plateau at 15–16%; and
- **criterion 5** — the **pre-2016 window** (8.64% against NIFTYBEES’ 12.68%).

Both are measured against NIFTYBEES and against the study’s own sweep, not against Open
Alpha. The outlier dependence (ten trades of 472 carrying the result) and the under-filled
book are likewise independent. **The verdict stands: SIGNAL, not STRATEGY.** What is now
open is only whether it would *complement* an honestly-measured Open Alpha — and that
question is deferred to `research/161_ath_base_age_breakout`, which builds its own honest
in-engine OA proxy rather than reusing the r/154 curve.

**Correlation is unaffected** by the fill assumption in direction: 0.468 daily / 0.617 monthly
is a co-movement measurement, and this pattern remains by construction a **subset** of
all-time-high breakout entries.
