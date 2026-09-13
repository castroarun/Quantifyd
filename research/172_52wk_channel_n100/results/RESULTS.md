# 52W — Buy the 52-week-high close, sell the 52-week-low close, on Nifty 100

# VERDICT: **NO EDGE as written. SIGNAL when optimised — not a STRATEGY. Nothing adopted.**

> **Phase 2 (stop-loss and trailing-stop combinations) did not change the label.** It
> sharpened it: on the best stop stack the momentum-matched null still beats the system on
> return (21–29 of 30 draws) and now **loses to it on risk-adjusted return (27–30 of 30)**.
> The 52-week high carries **no return information and real risk information**. Section 11
> onward.

research/172 · 13-Sep-2026 · window 2006-01-02 → 2026-09-11 (20.7 years) · VPS canonical

**The book is called 52W.** Every figure below is a ₹1,00,00,000 (₹1 crore) NSE-cash
long-only book, 20 slots at 5% of NAV each, next-open fills on both legs, 15 bps per side
(0.10% charges + 0.05% slippage), **after Indian tax** (20% STCG / 12.5% LTCG above 365
days, FY loss-netting settled 1 April, ₹1.25 lakh LTCG exemption), idle cash at 5.2%
post-tax. Where a figure is gross or pre-tax it says so.

---

## 1. The headline

| Book | CAGR (after tax) | MaxDD | Calmar | Sharpe | Trades | Win rate | Expectancy/trade |
|---|---|---|---|---|---|---|---|
| **52W Spec A — the literal rule** (buy close > 252d high close, sell close < 252d low close) | **14.25%** | −46.73% | 0.305 | 0.835 | 144 | 68.1% | +54.68% |
| **52W OPT — the optimised plateau** (same entry, SuperTrend(14,4) exit) | **14.95%** | −26.01% | **0.575** | 0.991 | 949 | 46.0% | +7.58% |
| 52W OPT on a **survivorship-free** universe (PIT top-100 by liquidity) | 10.77% | −32.35% | 0.333 | 0.716 | 1,080 | 42.6% | +4.90% |
| NIFTYBEES (benchmark) | 11.37% | −59.71% | 0.190 | 0.653 | — | — | — |
| Equal-weight buy-and-hold of the **same current Nifty 100** (drift control, pre-tax) | **19.22%** | −58.13% | 0.331 | 0.940 | — | — | — |
| Cash at 5.2% | 5.11% | 0.00% | — | — | — | — | — |

**Read it in one line.** The rule as Arun stated it makes money, but less money than simply
owning the same hundred stocks, and it is beaten by its own random-entry null. Optimising
the exit halves the drawdown and buys a real risk-adjusted improvement — and that improved
book is then beaten, on return, by a null that picks names at random from the top half of
the same universe by 12-month relative strength. **The 52-week high is a low-resolution
momentum proxy. Momentum ranking — which True North already runs — dominates it.**

---

## 2. Q1 — What does the literal spec do, gross / net / after tax, vs benchmark and vs null?

| 52W Spec A arm | CAGR | MaxDD | Calmar |
|---|---|---|---|
| Gross (0 bps, no tax) | 15.24% | −45.90% | 0.332 |
| Net of cost, before tax (15 bps/side) | 15.21% | −46.04% | 0.330 |
| **Net of cost, after tax (the headline)** | **14.25%** | **−46.73%** | **0.305** |
| Same, with idle cash switched off | 13.89% | −48.00% | 0.289 |
| Same-close fill (**LOOK-AHEAD reference arm, not tradeable**) | 14.27% | −43.72% | 0.326 |

Cost is almost irrelevant here — 1.0 trade per slot per year, ₹9.8 lakh of lifetime cost on
a ₹1 crore book. Tax costs a whole percentage point of CAGR (₹2.05 crore paid). The
look-ahead fill is worth nothing, which is unusual and is itself informative: this entry is
so slow that the fill mechanic does not matter (contrast r/142, where the fill was worth 8
CAGR points, and r/153, where it was 14).

**Against the benchmark:** +2.9pp of CAGR over NIFTYBEES with 13pp less drawdown. On its own
that looks like a win.

**Against the controls, it is not:**

| Control (same universe, same book, same costs, same tax) | CAGR | Calmar |
|---|---|---|
| 52W Spec A | 14.25% | 0.305 |
| **Random-entry null, matched fill-for-fill, same 52-week-low exit (30 draws)** | **median 16.47%** [12.90 … 18.52] | **median 0.352** [0.266 … 0.440] |
| Equal-weight buy-and-hold of the same names (pre-tax) | 19.22% | 0.331 |
| The identical entry with **no exit rule at all** (hold to the end) | 13.94% | 0.251 |

The literal spec sits **below the median of its own random null on both CAGR and Calmar**.
The 52-week-low exit is worth +0.31pp of CAGR over never selling at all, and buys that with
₹2 crore of turnover-driven tax. **This is a NO EDGE result and it is not close.**

---

## 3. Q2 — The optimised plateau: what matters and what does not

**252 entry × exit cells, then 220 one-axis cells from the plateau centre, then 30-seed and
12-offset ensembles. ~600 cells disclosed; discount accordingly.** The pre-registered
ranking metric was after-tax Calmar at 15 bps on the full window, and the pre-registered
eligibility clause (CAGR above NIFTYBEES, positive net expectancy, beats the null, beats
the drift control, plateau not peak, both sub-windows positive) was written before any cell
ran.

### 3a. The exit is the whole optimisation. The entry lookback is nearly inert.

Median Calmar across all 12 entry variants, by exit rule:

| Exit | Calmar | CAGR | Trades | Win rate |
|---|---|---|---|---|
| **SuperTrend(14,4) — the winner** | **0.521** | 14.52% | 935 | 45.6% |
| EMA(50) close | 0.487 | 12.51% | 1,361 | 40.1% |
| Donchian close-42 | 0.457 | 14.89% | 826 | 46.4% |
| SuperTrend(10,3) | 0.456 | 12.01% | 1,363 | 41.6% |
| SuperTrend(7,3) | 0.452 | 12.38% | 1,352 | 41.9% |
| ATR(14) × 6.0 trail | 0.444 | 15.18% | 518 | 48.6% |
| Donchian close-126 | 0.392 | 14.69% | 304 | 50.7% |
| **Donchian close-252 (the literal 52-week low)** | **0.318** | 14.26% | 140 | 64.3% |
| Donchian low-252 | 0.280 | 14.01% | 119 | 66.4% |
| SMA(15) close (Open Alpha's old exit) | 0.217 | 5.48% | 3,187 | 37.7% |

SuperTrend(14,4) wins again — the **third independent confirmation** after r/161 (+11.9pp
over OA's 15-SMA/−8% on identical entries) and r/159. The literal 52-week-low exit ranks
**18th of 21**. And a 15-SMA exit, which is fast enough for an all-time-high breakout book,
is catastrophic here (5.5% CAGR): this entry is too slow to be paired with a fast trail.

Median Calmar by entry lookback (close reference): 63 → 0.293, 126 → 0.411, **189 → 0.455**,
252 → 0.422, 378 → 0.448, 504 → 0.418. A **broad flat plateau from 126 to 504 days**, with
only the 63-day cell clearly worse. Median CAGR falls monotonically with lookback
(14.95 → 13.84). Comparing a close-max channel with a high-max channel: close wins at
L ≥ 189, high wins slightly at L ≤ 126, and the gap is ~0.03 of Calmar either way. **Arun's
choice of 52 weeks is neither special nor wrong — it sits in the middle of a plateau that
does not care.**

### 3b. Every other axis, at the plateau centre (L=252, close, ST(14,4), Nifty 100)

| Axis | Result |
|---|---|
| **Slots** | 10 → 0.471, **15 → 0.595**, **20 → 0.575**, 30 → 0.518, 100 → 0.821. The 100-slot cell is the highest Calmar in the whole study and is **disqualified by the pre-registered CAGR clause**: it returns 8.12% and is only 23% invested — it is a de-levered book, not a better one. 15–20 slots is the honest plateau. |
| **Universe** | Nifty 50 → 0.436 (10.21%), Next 50 → 0.527 (13.07%), **Nifty 100 → 0.575 (14.95%)**, Nifty 500 → 0.528 (18.00% CAGR, −34.1% DD). More return down-cap, worse risk — the r/145 shape exactly. |
| **Survivorship control** | **PIT top-100 by liquidity → 0.333 (10.77%); PIT top-50 → 0.267 (7.90%).** See §4. |
| **Index gate** | none → 0.575, NIFTYBEES > 200-SMA → 0.566, > 100-SMA → 0.554. **The gate does not help.** This contradicts r/71 and r/75 for this family, and the reason is visible: SuperTrend(14,4) already exits every holding on a trend break, so the gate has nothing left to remove. |
| **Entry buffer** | 0% → 0.575, +1% → 0.594, +3% → 0.532, within-5%-of-ATH → 0.531, at-a-new-ATH → 0.480. The +1% buffer is a noise-level improvement; the all-time-high variants make it worse, so this is **not** a disguised Open Alpha. |
| **Cost** | 0 / 15 / 30 / 45 bps per side → Calmar 0.603 / 0.575 / 0.548 / 0.519 (CAGR 15.56 / 14.95 / 14.36 / 13.71). About **−0.4pp of CAGR per +15 bps**, a fifth of the slope of a 12×/year book. |
| **Slot contention** | RS-rank (the pre-registered rule) → Calmar 0.575. Random draw, 30 seeds → median **0.588** [0.561 … 0.603]. **The relative-strength ranking adds nothing** — it is inside the noise of an arbitrary draw. |
| **Start-date phase** | 12 monthly start offsets → CAGR median 14.70 [13.84 … 14.95], Calmar 0.570 [0.557 … 0.579]. Extremely tight; path risk is not the problem here. |
| **Sub-windows** | W1 2006-2015: 13.74% / −26.01% / 0.528. W2 2016-2026: 16.10% / −20.97% / 0.768. **Both positive, W2 better.** Passes the split. |

**The optimised spec (52W OPT): Nifty 100, entry = close above the 252-day high close,
exit = SuperTrend(14,4) on the close, both filled at the next open, 20 slots at 5%.**

---

## 4. Q3 — The honest verdict, and the sins

### The three findings that decide it

**(i) Against a matched random-entry null, the optimum wins — until the null is given
momentum.** Four nulls, each matched fill-for-fill on entries per day, all running the same
ST(14,4) exit and the same book, 30 draws each:

| Null | CAGR median [min … max] | Calmar median [min … max] |
|---|---|---|
| **52W OPT (the actual system)** | **14.95%** | **0.575** |
| N1 — a random eligible name | 13.94 [11.91 … 16.29] | 0.395 [0.308 … 0.533] |
| N2 — a random name already in a ST(14,4) **uptrend** | 14.95 [13.36 … 17.09] | 0.445 [0.364 … 0.508] |
| N3 — a random name in the **top half by 252-day relative strength** | **16.79** [15.15 … 18.11] | **0.584** [0.511 … 0.682] |
| N4 — a random name that is both trending and strong | **17.48** [16.66 … 18.83] | 0.579 [0.520 … 0.641] |

The system beats N1 and N2 comfortably. It is **below the entire 30-draw range of N3 and N4
on CAGR** (14.95 vs a worst draw of 15.15 and 16.66) and sits at their **median** on Calmar.
Being at the median of a random control is the definition of no incremental information.

A useful decomposition sits beside it: the same book with **no channel condition at all** —
buy every eligible name the day SuperTrend(14,4) turns up — returns 12.43% at −48.3%, Calmar
0.257. So the channel *does* add something over naive trend entry. It just adds strictly
less than sorting the same universe by 12-month return, which is free.

**(ii) On a survivorship-free universe the advantage disappears.** The official Nifty 50 and
Next 50 lists in this repo are **current** membership, fetched Aug-2026; there is no
point-in-time constituent history anywhere in the project. Substituting the r/169
point-in-time liquidity rank (top 100 by 126-day median traded value, rebuilt monthly, funds
excluded by instrument name):

| Universe | CAGR | MaxDD | Calmar |
|---|---|---|---|
| Current Nifty 100 (survivorship-biased) | 14.95% | −26.01% | 0.575 |
| **PIT top-100 by liquidity (no survivorship)** | **10.77%** | −32.35% | **0.333** |
| PIT top-50 | 7.90% | −29.61% | 0.267 |

**The survivorship premium is ~4.2pp of CAGR and 0.24 of Calmar**, and the honest version
returns **less than NIFTYBEES**. The two universes are not identical constructions — a
liquidity rank is a proxy for index membership, not index membership itself — but the gap is
far too large to be construction noise, and it runs the right way.

**(iii) The economics are lottery-ticket-shaped.** Spec A's ten best trades out of 144 are
**59% of all profit**; cap every winner at +50% and mean return per trade falls from +54.7%
to +19.8%. 52W OPT is better but still concentrated: top-10 share 44.5%, mean return +7.58%
falling to +4.84% capped at +50%. Worst single-position adverse excursion **−33.7%** for
OPT, **−59.2%** for Spec A — which is the honest answer to "how deep does a 52-week-high buy
go before a 52-week low sells it".

### Falsification test, as pre-registered

> *"If the literal Spec A fails to beat NIFTYBEES after tax, AND the optimised plateau's
> advantage over the random null is smaller than its advantage over the drift control, the
> family is written up as NO EDGE / SIGNAL and nothing is proposed for adoption."*

Spec A beats NIFTYBEES (14.25 vs 11.37), so the first clause fails — the family is not
written off entirely. But the optimum loses to the momentum-matched null and loses to the
drift control on return. **The correct label is: the literal rule is NO EDGE; the optimised
rule is a SIGNAL — a genuine drawdown-control mechanism — and not a STRATEGY.**

### The seven deadly sins

| Sin | How it was controlled | Residual |
|---|---|---|
| **Look-ahead** | Channel windows exclude today; every rolling stat shifted one day; both legs fill at the next open. The same-close fill was run once, labelled, and is worth +0.02 CAGR — nothing. | none material |
| **Survivorship** | Named, unfixable for the official lists; quantified with a PIT liquidity proxy, a Nifty 500 arm, and an equal-weight buy-and-hold drift control on the identical name set. | **~4.2pp of CAGR. This is the single largest number in the study.** |
| **Overfitting / multiple testing** | ~600 cells disclosed; metric and adoption bar pre-registered; the winner is the centre of a broad plateau (exit family, lookback 126–504, slots 15–20) and every neighbour agrees within ~10%; W1/W2 both positive. | low — but the *optimum is still not adopted*, so the question is moot |
| **Cost neglect** | 15 bps/side base, 0/15/30/45 ladder, after-tax with FY netting and the ₹1.25 L LTCG exemption, idle cash at 5.2%. Gross and net both reported. | none |
| **Regime dependence** | Per-year table with intra-year drawdown measured **from the running peak of the full curve** (the r/154 convention, not the window's own first bar). W1/W2 split. 2008 and 2020 crash rows, 2018 and 2022 grind rows. | none — 2008 is where the system earns its keep (see §6) |
| **Correlation / single factor** | Daily and monthly correlation to TN, OA·Base Age, IPO-A and NIFTYBEES; 4-sleeve blend against the incumbent; weight-matched cash null. | **fails: 0.69 daily / 0.66 monthly to OA. See §5.** |
| **Capacity / shortability** | Long-only NSE cash on the hundred most liquid names, ₹5 lakh a position against a ₹5 crore liquidity floor (real Nifty-100 turnover is ₹50–500 crore a day). | **none — this is the one thing the system is genuinely good at.** A ₹50–100 crore book is fundable. |

---

## 5. Q4 — Portfolio fit against TN, OA and IPO

Run on the research/168 blend engine **unchanged**, so the incumbent figures tie digit for
digit to the published three-sleeve study. 30 paired paths (IPO seed *p*, OA seed *p*, TN
offset *p* mod 12), monthly rebalance, after tax, 5.2% cash, window 2006-04-03 → 2026-09-03.

### Correlation — it duplicates Open Alpha

| 52W arm | vs TN | vs **OA · Base Age** | vs IPO-A | vs NIFTYBEES |
|---|---|---|---|---|
| 52W OPT, daily | 0.554 | **0.689** | 0.305 | 0.507 |
| 52W OPT, monthly | 0.590 | **0.662** | 0.390 | 0.603 |
| 52W Spec A, daily | 0.385 | 0.554 | 0.252 | 0.724 |

The pre-registered bar was **correlation < 0.40 to both existing legs**. 52W OPT is at
**0.69 to Open Alpha** — it is very nearly the same book. That is the r/145 finding
reproduced: a broad Indian-equity breakout book re-imports the beta OA already harvests.

### Blend value — it fails the bar, and a cash sleeve beats it

Incumbent = the adopted r/168 book, TN 37.5% / OA 37.5% / IPO-A 25%:
**CAGR 21.18%, MaxDD −24.01%, Calmar 0.885.** The candidate is funded pro-rata out of all
three.

| Sleeve added | Weight | Blend CAGR | Blend DD | Blend Calmar | Δ Calmar | Δ CAGR | Δ DD | Paths won /30 |
|---|---|---|---|---|---|---|---|---|
| 52W OPT | 5% | 20.85 | −23.17 | 0.902 | +0.016 | −0.33 | +0.82 | 26 |
| **52W OPT** | **10%** | 20.51 | −22.69 | **0.909** | **+0.033** | −0.67 | +1.61 | 22 |
| 52W OPT | 20% | 19.83 | −22.52 | 0.872 | −0.006 | −1.35 | +1.42 | 12 |
| 52W OPT | 33% | 18.94 | −23.01 | 0.816 | −0.064 | −2.24 | +0.93 | 2 |
| 52W Spec A | 10% | 20.55 | −24.09 | 0.848 | −0.035 | −0.63 | −0.21 | 6 |
| 52W OPT (PIT-100) | 10% | 20.07 | −23.69 | 0.848 | −0.039 | −1.11 | +0.23 | **0** |
| **CASH at 5.2%** | 10% | 19.59 | −21.61 | **0.914** | **+0.028** | −1.59 | +2.49 | **30** |
| **CASH at 5.2%** | 25% | 17.20 | −17.67 | **0.980** | **+0.089** | −3.98 | +6.28 | **30** |
| **CASH at 5.2%** | 33% | 15.92 | −15.50 | **1.029** | **+0.135** | −5.26 | +8.35 | **30** |

**The pre-registered bar was +0.10 Calmar or −2pp drawdown at no worse CAGR.** The best
52W cell reaches **+0.033 Calmar at −0.67pp of CAGR, on 22 of 30 paths** — it clears
neither leg of the bar, and it does it while costing return. Meanwhile a **plain cash
sleeve wins 30 of 30 paths at every weight** and overtakes 52W decisively above 10%. The
survivorship-free version of the candidate loses on **0 of 30 paths**.

This is the same verdict r/146 delivered on mean reversion: what looks like a diversifier is
mostly de-levering, and if de-levering is what you want, an arbitrage fund does it better,
for free, with no correlation and no tax drag.

---

## 6. The YoY house table

Annual return with the intra-year maximum drawdown beneath it, **measured from the running
peak of the full curve** (r/154 convention). After tax, net of costs. Benchmarks excluded
from the best-of picks.

See `results/yoy_table.md` for the rendered table and `results/yoy_data.json` for the data.
The rows that matter:

| | 52W Spec A | 52W OPT | 52W OPT (PIT-100) | Random-entry null (median) | EW B&H Nifty100 | NIFTYBEES |
|---|---|---|---|---|---|---|
| **2008** (crash) | −43.5 (−46.2) | **−22.2 (−25.7)** | −27.4 (−30.2) | −23.0 (−27.2) | −47.7 (−58.1) | −52.1 (−59.7) |
| **2020** (crash) | +13.5 (−30.9) | +27.2 (−13.7) | +24.4 (−25.9) | **+38.0 (−15.5)** | +25.1 (−38.3) | +15.4 (−36.3) |
| **2018** (grind) | +6.2 (−16.7) | −2.9 (−11.5) | −17.4 (−25.1) | −10.1 (−15.9) | −1.9 (−16.1) | +4.8 (−14.1) |
| **2022** (grind) | +5.6 (−23.9) | +4.6 (−21.0) | −12.4 (−32.4) | −5.3 (−26.1) | **+18.3 (−16.0)** | +5.5 (−16.1) |
| **FULL 2006-2026** | 14.25% / −46.7% (Calmar 0.31) | **14.95% / −26.0% (0.57)** | 10.77% / −32.4% (0.33) | 13.35% / −34.6% (0.39) | 19.22% / −58.1% (0.33) | 11.37% / −59.7% (0.19) |

**2008 is where the optimised book earns everything it has.** −22.2% against the index's
−52.1% and the equal-weight basket's −47.7%. A trend exit on a breakout entry is a crash
system. Strip 2008 out and the case collapses — which is exactly what the year-by-year
picks show: 52W OPT takes BEST CAGR in 4 of 21 years and LEAST DD in 8 of 21.

---

## 7. Q5 — Tradeability

| | 52W Spec A | 52W OPT |
|---|---|---|
| Trades | 144 (7.0 / year) | 949 (45.9 / year) |
| Win rate | 68.1% | 46.0% |
| Average win | +87.6% | +25.7% |
| Average loss | −15.5% | −7.9% |
| **Expectancy per trade, net of cost** | **+54.68%** | **+7.58%** |
| Median holding period | **737 days (2.0 years)** | 84 days |
| 95th-percentile holding period | 1,701 days (4.7 years) | 273 days |
| **Max consecutive losers** | 6 | **12** |
| **Worst single-position drawdown (MAE)** | **−59.2%** | −33.7% |
| Median position MAE | −10.3% | −5.2% |
| 5th-percentile position MAE | −34.1% | −14.6% |
| Top-10 trades as share of total profit | **59.0%** | 44.5% |
| Average invested | 93.4% | 71.2% |
| Position size at ₹1 cr / 20 slots | ₹5,00,000 | ₹5,00,000 |
| Lifetime cost paid | ₹9.8 lakh | ₹99.7 lakh |
| Lifetime tax paid | ₹2.05 crore | ₹3.55 crore |

**What a human would actually have to sit through.** Spec A holds a position for two years
on the median and takes it **59% under water at the worst** before the 52-week low finally
sells. It is a high-win-rate book (68%) whose entire profit is ten trades. 52W OPT is the
opposite: it loses 12 times in a row at the worst, wins 46% of the time, and turns the book
over 46 times a year — the payoff shape is completely different even though the CAGRs are
within 0.7pp of each other. Neither is unrunnable; neither is worth running.

---

## 8. What was NOT tested, and why

- **Point-in-time Nifty 100 membership.** It does not exist in this repo and
  niftyindices.com does not publish a reconstructable history. The liquidity-rank proxy is
  the best available substitute and is labelled as a proxy everywhere it appears.
- **Shorts.** Long-only by design. r/83 and r/147 closed the channel-short line on Indian
  equities (V-recoveries eat the short leg).
- **Pyramiding / re-entry into an existing holding.** Excluded from Spec A by Arun's
  wording; r/171 has just concluded that topping up existing winners is the worst
  destination for spare capital on a comparable book.
- **Weekly or monthly bars.** Daily only. A weekly 52-week channel would trade less and is a
  reasonable follow-up, but the null result in §4(i) would have to be overturned first.
- **A capacity test above ₹1 crore.** Not needed — position size is 0.01–0.1% of a day's
  volume in these names.
- **Any deployment.** No live engine was touched, no paper book started, no order placed.

---

## 9. Reproduce

VPS, `/home/arun/quantifyd`, `venv/bin/python`, ~12 minutes total, in this order:

```bash
venv/bin/python -u research/172_52wk_channel_n100/scripts/run172.py  all   # specA, 252-cell grid, 220 axes, robustness
venv/bin/python -u research/172_52wk_channel_n100/scripts/run172b.py all   # nulls at the optimum, start-phase, per-year curves
venv/bin/python -u research/172_52wk_channel_n100/scripts/run172c.py all   # the four adversarial nulls, shortlist, cost ladder
venv/bin/python -u research/172_52wk_channel_n100/scripts/blend172.py      # correlation + 4-sleeve blend vs TN/OA/IPO
venv/bin/python -u research/172_52wk_channel_n100/scripts/report172.py     # tearsheet + curves + YoY table
```

**Reproducibility stamp.** `backtest_data/market_data.db` as of 13-Sep-2026 (VPS,
canonical). Panel 5,890 trading days × 1,490 symbols, 2003-01-01 → 2026-09-11. **202 split
discontinuities detected and back-adjusted** (33 of them on names carrying ≥ ₹5 crore of
daily traded value) — listed in `results/split_events.csv`. **2 phantom holiday dates
dropped** (2014-04-24, 2014-10-15). Funds excluded by instrument name via
`backtest_data/etf_exclusions.json`. Blend engine: research/168 `blend_grid.py`, unmodified.

---

## 10. What this changes

**Nothing operationally.** No book changes size, rules or status. The live register
(`frontend/src/data/strategies.ts`) is untouched by this study because nothing graduated.

**What it adds to the shelf:**

1. **A third independent confirmation that SuperTrend(14,4) is this project's best equity
   trailing exit** (after r/159 and r/161), now demonstrated on an entry family neither of
   those studies used.
2. **A quantified survivorship premium for the official index-constituent CSVs: ~4.2pp of
   CAGR on a 20-year Nifty-100 backtest.** Any future study that screens on the current
   NIFTY50/NIFTYNEXT50 lists should subtract roughly that much before believing its result.
3. **A reusable momentum-matched null.** N3/N4 — drawing the control from the top half of
   the universe by 252-day relative strength — is a much sharper test than a plain random
   draw, and it is what killed this idea. It belongs in the standard control set for every
   future long-equity study.
4. **A new entry on the known-dead-ends list**: *channel / 52-week-high breakout on a
   large-cap Indian universe — SIGNAL, dominated by momentum ranking, correlation 0.69 to
   Open Alpha.*


---

# PHASE 2 — stop-loss and trailing-stop combinations

*Added 13-Sep-2026 on Arun's mid-turn instruction: "u can add some stop loss
variations/trailing SL etc, try different combinations as well."*

Book held at the Phase 1 settings (₹1 crore, 20 slots @ 5%, next-open fills, 15 bps/side,
5.2% idle cash, after tax, 2006-2026). **Only the exit stack varies, and every stack runs
on BOTH entries** — the plateau centre **E189** (189-day close channel) and the literal Spec
A entry **E252** (252-day close channel) — so every comparison is paired on identical
signals. Engine: `scripts/bt172b.py`; `bt172.py` untouched, so Phase 1 stays bit-exact.
~300 cells.

## 11. Q1 — Do initial hard stops help or hurt, and where does the surface turn?

**They do neither. The surface does not turn — it is noise, and the paired entry proves it.**

Initial stop from the entry price, with the literal 52-week-low as the other exit:

| Initial stop | −5% | −8% | −10% | −12% | −15% | −20% | −25% | −30% | none |
|---|---|---|---|---|---|---|---|---|---|
| **Calmar, E189** | 0.385 | **0.408** | 0.324 | 0.349 | 0.372 | 0.319 | 0.326 | 0.321 | 0.326 |
| **Calmar, E252** | 0.330 | 0.301 | 0.317 | 0.293 | 0.296 | 0.314 | 0.315 | 0.316 | 0.305 |
| CAGR, E189 | 14.44 | 16.24 | 13.72 | 14.89 | 16.19 | 13.87 | 14.35 | 14.36 | 14.72 |
| CAGR, E252 | 13.83 | 12.90 | 13.62 | 12.82 | 12.88 | 13.43 | 14.27 | 13.89 | 14.25 |
| MaxDD, E252 | −41.95 | −42.91 | −42.94 | −43.69 | −43.48 | −42.75 | −45.24 | −44.04 | −46.73 |

The E189 row looks like it has a −8% optimum at 0.408. **E252 scores the same cell 0.301 —
below its own no-stop row.** One entry says the best stop in the grid, the other says the
worst. That is the definition of an artifact, and it is exactly why every cell was run on
two entries. ATR-based initial stops behave the same way (1.5×ATR: 0.409 on E189, 0.315 on
E252).

**And the mechanism explains it.** A −30% stop only moves max drawdown from −46.7% to
−44.0%. **A stop on the position does not fix a drawdown made of twenty positions falling
together.** What a tight stop does buy is a collapsing win rate (68.1% with no stop →
35.9% at −8% → 25.8% at −5%) and a longer losing streak (6 → 30), for no risk reduction.

## 12. Q2 — Which trailing family wins, and is it monotonic?

**Trailing wins decisively over fixed-from-entry — the r/71 ordering holds. But the
percentage trail is twin-peaked, not monotonic.**

| Trail from the highest close since entry | −8% | −10% | −12% | −15% | −20% | −25% | −30% |
|---|---|---|---|---|---|---|---|
| **Calmar, E189** | 0.444 | **0.585** | 0.490 | 0.501 | **0.580** | 0.441 | 0.436 |
| **Calmar, E252** | 0.367 | 0.522 | 0.527 | 0.529 | **0.594** | 0.454 | 0.428 |
| CAGR, E189 | 10.69 | 12.19 | 12.27 | 13.75 | **16.01** | 13.80 | 13.01 |
| Trades | 1,690 | 1,230 | 945 | 623 | **350** | 217 | 132 |

Two peaks, at −10% and −20%, with a trough between. **Only the −20% peak survives the
paired entry** (0.580 / 0.594 on both); the −10% peak is E189-only (0.585 vs 0.522). And
−20% is the one carrying the return: 16.01% against 12.19%.

**Chandelier (k × ATR(22) from the highest high) IS monotonic**, and it points the same way:
2× → 0.281, 3× → 0.498, 4× → 0.512. **Wider is better** — the same "slow trail wins" shape
r/159 and r/161 found, and the reason SuperTrend(14,4) beat SuperTrend(7,3) in Phase 1.

Ranking the whole trailing family by Calmar (E189 / E252): 20% trail **0.580 / 0.594** ›
10% trail 0.585 / 0.522 › SuperTrend(14,4) 0.557 / 0.575 › SuperTrend(10,3) 0.556 / 0.552 ›
EMA50 0.516 / 0.525 › chandelier 4× 0.512 / 0.476 › Donchian-close-42 0.497 / 0.496 ›
Donchian-close-63 0.483 / 0.473 › Donchian-close-126 0.409 / 0.404 › **the literal
52-week-low 0.326 / 0.305**.

**So the answer to "trailing ≫ target ≫ tight stop": the first half reproduces (trailing
0.43–0.59 versus fixed-from-entry 0.29–0.41). No target arm was in scope in either phase,
so the middle term is untested here — stated rather than assumed.**

## 13. Q3 — Do combinations beat the best single exit?

**Marginally, and only one combination does: the time stop. Everything else is a wash, an
artifact, or arithmetic.**

| Stack | CAGR E189 / E252 | MaxDD E189 / E252 | Calmar E189 / E252 |
|---|---|---|---|
| **52W STOPPED — 20% trail + exit after 63 days if not up ≥ 0%** | **16.73 / 16.54** | −27.64 / −26.57 | **0.605 / 0.622** |
| 20% trail + 63-day time stop at a +5% bar | 16.43 / 16.26 | −28.84 / −26.94 | 0.570 / 0.604 |
| 20% trail + 126-day time stop | 16.27 / 15.46 | −27.86 / −27.74 | 0.584 / 0.557 |
| 20% trail alone | 16.01 / 15.89 | −27.61 / −26.77 | 0.580 / 0.594 |
| −20% hard stop + 20% trail | 16.01 / 15.89 | — | 0.580 / 0.594 *(identical — see below)* |
| −15% hard stop + SuperTrend(14,4) | 15.14 / 14.94 | −26.96 / −25.82 | 0.562 / 0.579 |
| 52W OPT — SuperTrend(14,4) alone | 15.28 / 14.95 | −27.44 / −26.01 | 0.557 / 0.575 |
| Breakeven at +10% + SuperTrend(14,4) | 15.21 / 14.63 | −26.76 / −26.36 | 0.568 / 0.555 |
| Profit-lock at +50% → −10% trail + ST(14,4) | 14.11 / 14.54 | −25.72 / −22.74 | 0.549 / **0.639** |
| Stop-out blocks re-entry for 63 bars (on −15% + ST(14,4)) | 14.77 / 15.10 | −26.65 / −25.86 | 0.554 / 0.584 |
| Stop-out blocks re-entry **for good** | 12.58 / 12.46 | −25.61 / −27.06 | 0.491 / 0.460 |
| −10% hard stop + Donchian-close-63 | 15.64 / 15.23 | −34.53 / −33.37 | 0.453 / 0.457 |

- **A hard stop wider than the trail is inert by construction.** `−20% hard + −20% trail`
  reproduces `−20% trail` to the digit, because the trail is measured from a peak that is
  never below the entry price, so the hard stop can never be the binding level. Same for
  −15%+−15% and for −15%/−20% with a 3×ATR chandelier. **Three rows of this table are
  arithmetic, not evidence**, and are marked as such.
- **Breakeven moves are a wash** (0.568/0.555 versus 0.557/0.575 for the bare rule).
- **Profit-locks are entry-specific noise**: the best cell scores 0.639 on E252 and 0.549 on
  E189. Do not read the 0.639.
- **Blocking re-entry for good starves the book** — 96 trades in 20.7 years, 8.8% CAGR.
  Blocking for 63 bars is neutral. Spec A's permissiveness was right.
- **The book-level trailing-drawdown kill is a disaster.** At −20% on the winner: **−2.66%
  CAGR, −81.4% drawdown, 15,990 trades.** At −15%: −4.10% and −80.2%. It liquidates,
  re-arms when NIFTYBEES reclaims its 50-SMA, buys back into the same falling tape and is
  killed again, churning the book to death. This is the r/71/r/75 index-gate result in its
  most violent form: **on this family, top-down risk-off machinery destroys value.**
- **The winning combination adds +0.025 / +0.028 of Calmar and +0.7pp of CAGR** over the
  bare 20% trail. Real, consistent across both entries, and small.

### The auto-ranked winner was a trap, and the pre-registered clauses caught it

Ranking E189 Calmar blindly picks `TRAIL10_T63_g0_BK20` at **0.606** — fractionally above
52W STOPPED's 0.605. It fails three pre-registered clauses:

| Clause | 52W STOPPED | The trap |
|---|---|---|
| Plateau (neighbours agree) | −15%/−20% kill both ≈ disaster, but the *stack without any kill* scores 0.605/0.622 | its −15% neighbour returns **−1.06% CAGR** |
| Cost ladder 0/15/30/45 bps | 0.630 / 0.605 / 0.598 / 0.569 | 0.480 / 0.606 / **−0.080** / **−0.186** |
| 12 start-offsets | 0.603 [0.565 … 0.615] | 0.573 [**−0.009** … 0.606] |
| CAGR | 16.73% | 12.19% |

## 14. Robustness of 52W STOPPED

| Ensemble | CAGR median [min … max] | Calmar median [min … max] |
|---|---|---|
| 12 monthly start-offsets, E189 | 16.66% [15.61 … 16.99] | 0.603 [0.565 … 0.615] |
| 12 monthly start-offsets, E252 | 16.59% [15.57 … 16.98] | 0.640 [0.622 … 0.660] |

Cost ladder (0 / 15 / 30 / 45 bps a side): CAGR **17.13 / 16.73 / 16.39 / 15.85** on E189 and
**16.91 / 16.54 / 16.06 / 15.66** on E252; Calmar 0.630 / 0.605 / 0.598 / 0.569 and
0.638 / 0.622 / 0.604 / 0.588. About **−0.4pp of CAGR per +15 bps** — the same shallow slope
as Phase 1, and it survives 45 bps comfortably.

## 15. Q4 — Does any stack change the verdict, or the blend?

**No. And the gate that would have stopped us finding that out was deliberately overridden.**

The pre-registered rule was to skip the nulls and the blend unless the best Phase 2 stack
beat 52W OPT's Calmar of 0.575 by ≥ 0.05. 52W STOPPED clears it by **+0.030 (E189)** and
**+0.047 (E252)** — just inside the threshold, so the automatic gate skipped. That was the
wrong place to stop, for a reason worth recording: the gate exists to save compute, and the
question it was about to skip is the one the whole study turns on. The nulls and the blend
were therefore re-run for 52W STOPPED, costing 186 cheap cells, and **the override is
disclosed in `scripts/run172e.py`'s docstring, in the STATUS log and here — not done
quietly.**

**The momentum-matched nulls, re-run on the winner's own exit stack, 30 draws each:**

| Control (52W STOPPED = 16.73% / Calmar 0.605 on E189; 16.54% / 0.622 on E252) | CAGR median [min … max] | Calmar median | Beats the system on **CAGR** | Beats it on **Calmar** |
|---|---|---|---|---|
| N3 momentum-matched, E189 | **18.16** [15.37 … 20.77] | 0.507 | **24 / 30** | **1 / 30** |
| N4 trend + momentum matched, E189 | **18.71** [16.25 … 20.69] | 0.560 | **29 / 30** | **3 / 30** |
| N3 momentum-matched, E252 | **17.27** [15.12 … 20.01] | 0.489 | **21 / 30** | **0 / 30** |
| N4 trend + momentum matched, E252 | **17.20** [14.96 … 20.10] | 0.530 | **24 / 30** | **2 / 30** |

**This is the single most informative table in the study.** Picking names at random from the
top half of the same universe by 252-day relative strength, and running the *identical* 20%
trail and 63-day time stop, earns **more return** than the 52-week-high entry on 21–29 draws
out of 30 — and does so at a **materially worse drawdown**, losing on Calmar on 27–30 of 30.

So the 52-week high is **not** telling you which stock will go up. It is telling you which
stock's *path* will be smoother. In Phase 1 the system sat at the null's Calmar median
(0.575 against 0.584 / 0.579) — a coin flip. The stop stack is what converts that into a
clean 27–30 of 30 win. **The stops did not create return; they revealed the risk edge that
was already there.**

**Blend, same r/168 engine, 30 paired paths, against TN 37.5 / OA 37.5 / IPO-A 25
(21.18% / −24.01% / Calmar 0.885):**

| Sleeve | w | Blend Calmar | Δ Calmar | Δ CAGR | Δ DD | Paths won /30 |
|---|---|---|---|---|---|---|
| 52W STOPPED (E252) | 5% | 0.912 | +0.027 | −0.24 | +1.00 | 26 |
| **52W STOPPED (E252) — its best cell** | **10%** | 0.918 | **+0.043** | −0.48 | +1.65 | 23 |
| 52W STOPPED (E252) | 25% | 0.868 | −0.012 | −1.22 | +1.13 | 11 |
| 52W OPT (Phase 1) | 10% | 0.909 | +0.033 | −0.67 | +1.61 | 22 |
| **PLAIN CASH at 5.2%** | 15% | 0.932 | **+0.045** | −2.38 | +3.73 | **30** |
| **PLAIN CASH at 5.2%** | 33% | 1.029 | **+0.135** | −5.26 | +8.35 | **30** |

The bar was **+0.10 Calmar or −2pp drawdown at no worse CAGR**. 52W STOPPED reaches **+0.043
at −0.48pp of CAGR on 23 of 30 paths** — an improvement on Phase 1's +0.033, and still a
clear fail. A plain cash sleeve matches it at 15% weight on **30 of 30** and beats it
outright above that. Correlation to Open Alpha is **0.679 daily / 0.639 monthly** (Phase 1:
0.689 / 0.662) — barely moved, still far above the 0.40 bar.

**Verdict unchanged: NO EDGE as written, SIGNAL when optimised, not a STRATEGY. Nothing
adopted.**

## 16. Q5 — Tradeability of 52W STOPPED versus 52W OPT and Spec A

| | 52W Spec A (literal) | 52W OPT (ST 14,4) | **52W STOPPED (20% trail + 63d time stop)** |
|---|---|---|---|
| CAGR after tax | 14.25% | 14.95% | **16.73% / 16.54%** |
| MaxDD | −46.73% | −26.01% | −27.64% / −26.57% |
| Calmar | 0.305 | 0.575 | **0.605 / 0.622** |
| Sharpe | 0.835 | 0.991 | **1.058** |
| Trades | 144 (7.0/yr) | 949 (45.9/yr) | 538 (26.0/yr) |
| Win rate | **68.1%** | 46.0% | **38.1%** |
| Average win | +87.6% | +25.7% | +55.5% |
| Average loss | −15.5% | −7.9% | **−7.9%** |
| Expectancy per trade | +54.68% | +7.58% | +16.24% |
| Median hold | 737 d | 84 d | 98 d |
| p95 hold | 1,701 d | 273 d | 693 d |
| **Max consecutive losers** | 6 | 12 | **21** |
| Worst position drawdown | −59.2% | −33.7% | **−25.6%** |
| Median position drawdown | −10.3% | −5.2% | −6.3% |
| 5th-pct position drawdown | −34.1% | −14.6% | −20.5% |
| Top-10 trades' share of profit | 59.0% | 44.5% | 53.3% |
| Mean trade with winners capped at +50% | +19.8% | +4.84% | +5.51% |
| Average invested | 93.4% | 71.2% | 83.5% |
| Exit mix | 88.9% channel-low, 11.1% end-of-data | 98.8% SuperTrend | **54.3% trailing stop, 42.8% time stop** |

**What it would feel like.** 52W STOPPED is the best book in the study on every headline —
return, Sharpe, Calmar, worst single-position loss — and it has the **worst win rate
(38.1%) and the longest losing streak (21 in a row)** of the three. Two out of every five
positions are sold by a clock rather than by price. It is the hardest of the three to sit
through, and it still is not adoptable.

## 17. Per-year, Phase 2 (return with intra-year drawdown from the full curve's peak)

| Year | 52W STOPPED E189 | 52W OPT | 52W Spec A | NIFTYBEES |
|---|---|---|---|---|
| 2008 | **−21.2 (−25.1)** | −22.2 (−25.7) | −43.5 (−46.2) | −52.1 (−59.7) |
| 2011 | −21.9 (−25.6) | **−12.8 (−19.6)** | −21.8 (−30.9) | −24.0 (−27.3) |
| 2018 | −5.4 (−16.8) | −2.9 (−11.5) | **+6.2 (−16.7)** | +4.8 (−14.1) |
| 2020 | +18.2 (−22.0) | **+27.2 (−13.7)** | +13.5 (−30.9) | +15.4 (−36.3) |
| 2022 | **+18.1 (−18.1)** | +4.6 (−21.0) | +5.6 (−23.9) | +5.5 (−16.1) |
| 2024 | **+40.2 (−12.2)** | +22.4 (−14.6) | +29.2 (−11.6) | +10.4 (−10.5) |
| **FULL** | **16.73 / −27.6 (0.605)** | 14.95 / −26.0 (0.575) | 14.25 / −46.7 (0.305) | 11.37 / −59.7 (0.190) |

Full 21-row table in `results/p2_peryear.csv`. 52W STOPPED wins the 2022 grind (+18.1%
against +4.6%) and loses 2020's V-recovery (+18.2% against +27.2%) — the signature of a
tighter trail: better in a grind, worse in a snapback.

## 18. Phase 2 files

| File | What |
|---|---|
| `scripts/bt172b.py` | the exit-stack simulator (ratcheting levels, book-level kill, re-entry block) |
| `scripts/run172d.py` | phases A / B / C — 65 stacks × 2 entries, time stop, book kill, offsets, cost ladder, per-year |
| `scripts/run172e.py` | the deliberate gate override: nulls N3/N4 and the cost ladder for the honest winner |
| `scripts/blend172b.py` | correlation + 4-sleeve blend for 52W STOPPED |
| `scripts/report172b.py` | the Phase 2 three-panel figure |
| `results/stops.csv` | every Phase 2 cell, one row each, both entries |
| `results/p2_nulls.csv`, `p2_null_summary.json` | 120 momentum-matched null draws |
| `results/p2_blend.csv`, `p2_correlations.json` | blend and correlation |
| `results/p2_peryear.csv`, `p2_outliers.csv`, `p2_summary.json`, `p2_curves.npz` | per-year, outlier dependence, curves |
| `results/p2_gate.json`, `p2_top5.json` | the automatic gate decision that was overridden, and the auto top-5 |
