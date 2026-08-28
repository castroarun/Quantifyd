# research/134 — What to run alongside a book that is entirely short-vol

## VERDICT: **CONCLUDED — the diversifier is plain LONG EQUITY, and you already own it. No new system, and NOT the options structures the ask reached for.**

The request was for a directional, uncorrelated sleeve that profits in extremes so
that trending markets do not turn the neutral book's month into a deep loss. The
data answers it, but not in the shape the question assumed.

---

## The one-paragraph answer

The short-vol book's enemy is the **low-volatility melt-up**, not the crash — in 75
months it has *never* lost money in a NIFTY down-trend, and it made **+7.85%** in the
worst one (April 2020, index −18.4% over 45 days). So a long-put tail hedge, jade
lizard or skewed condor is aimed at the wrong tail, or is itself short vol. What
actually offsets the loss state is **being long the index**: a 30% long-equity sleeve
cuts the worst month from **−9.27% to −4.11%** and max drawdown from **−10.4% to
−7.0%** while *raising* CAGR, and it beats the honest null of simply trading the
neutral book smaller by **+7.1% CAGR at the same worst month**. About **40% of that
edge is genuine diversification** and 60% is the equity premium of a bull decade —
proven by re-running with the sleeve's excess return stripped out, where it still
wins by **+2.9%**. Trend *timing* on top of this actively hurt: plain buy-and-hold
beat every moving-average, Donchian and time-series-momentum variant tested.

---

## 1. Stage A — the problem, measured before any solution

Combined neutral book = C1 stock winged strangles (research/127) + 45-DTE NIFTY
straddle (research/119), equal-risk, 75 common months 2019-05 → 2026-07.

**The two sleeves correlate only +0.32.** The book is not one bet; the existing
diversification between them is real.

**Losses are an up-trend phenomenon:**

| 45-day NIFTY run | n | mean month | worst month |
|---|---|---|---|
| ≤ −5% (down trend) | 7 | **+3.62%** | −1.19% |
| \|run\| < 5% (chop) | 48 | +2.62% | −5.67% |
| ≥ +5% (**up trend**) | 20 | **−1.19%** | **−9.27%** |

corr(book, up-run) **−0.532**; corr(book, down-run) **+0.222**. Nine of the ten
worst months had NIFTY *rising*. The worst (2023-12, −9.27%) came on a **+15.2%**
45-day run; 2020-12 (−5.27%) on **+19.2%**.

Holds on both robustness cuts: dense era 2021+ (down n=5, worst **+0.18%**), and the
**VIX-rank>25 filtered ruleset actually run live** (down n=8, mean +5.40%, worst
**+1.05%** — never lost; up n=19 contributing **+3.35% in total** across all nineteen).

**Mechanism.** A sell-off expands implied vol, so premium sold is rich and
mean-reverts; a melt-up is a low-vol grind that walks through the short call with no
vol spike to compensate. Short vol is not short the market — it is short *surprise*,
and India's upside surprises are the quiet ones.

## 2. Stage B/C — what fixes it

Sleeves are long-only index exposure, cash at 6.5% when out, 10 bps per switch.

**Timing does not help.** Buy-and-hold beat every timed variant:

| Sleeve | CAGR % | MaxDD % | Calmar | switches |
|---|---|---|---|---|
| **NIFTY B&H** | **17.47** | −14.3 | **1.22** | 1 |
| TS-mom 12m | 8.79 | −14.3 | 0.62 | 82 |
| MA50 long/cash | 7.39 | −11.7 | 0.63 | 232 |
| MA200 long/cash | 7.19 | −15.7 | 0.46 | 136 |
| Donchian 20/10 | 6.91 | −11.4 | 0.61 | 142 |
| Donchian 55/20 | 5.50 | −11.9 | 0.46 | 75 |

**The combination, vs the pre-declared size-down null** (shrink the neutral book to
the same worst month, idle cash at 6.5%):

| Weight to NIFTY | CAGR % | MaxDD % | Calmar | worst % | vs size-down |
|---|---|---|---|---|---|
| 0% (neutral alone) | 21.29 | −10.4 | 2.05 | −9.27 | — |
| 10% | 21.13 | −8.7 | 2.43 | −7.55 | **+2.41%** |
| 20% | 20.91 | −7.8 | 2.67 | −5.83 | **+4.78%** |
| **30%** | **20.65** | **−7.0** | **2.96** | **−4.11** | **+7.11%** |
| 40% | 20.34 | −6.2 | 3.29 | −3.05 | **+8.41%** |

**Why the pairing works** — the two legs cover opposite states:

| | up-trend (20) | chop (48) | down-trend (7) |
|---|---|---|---|
| neutral book | **−1.19%** | +2.62% | +3.62% |
| NIFTY B&H sleeve | **+4.03%** | +0.38% | +1.11% |
| **70/30 blend** | **+0.38%** | +1.95% | +2.87% |

The blend's up-trend loss is gone; the cost is ~0.67%/mo of chop return. Worst month
−9.27% → **−4.11%**.

## 3. The controls (all pre-declared before any number was seen)

| Control | Result |
|---|---|
| **C1 size-down null** | **PASSES** — +7.1% CAGR at matched worst month, at 30% |
| **C4 era split** | **PASSES both halves** — early 2019-22 +7.22%, late 2023-26 +6.37% |
| **Equity-premium strip** | **PASSES** — sleeve demeaned to the cash rate, same shape, zero excess return: still **+2.92%** at 30%. This is the honest floor. |
| **Timing (C2-adjacent)** | Gated versions **LOSE** to ungated B&H — the timing was noise |

**Decomposition:** of the +7.1% edge, **+2.9% is diversification** (survives a flat
market) and **~+4.2% is the equity risk premium** of this particular decade. Only the
first number should be relied on if equities go sideways for years.

## 4. What this study rules OUT

- **Long puts / long-put tail hedges.** Insurance against a state the book has never
  lost in across 75 months. research/103 already showed naked long convexity bleeds;
  here it would also be aimed at the wrong tail.
- **Jade lizards, skewed iron condors.** These are themselves net short vol — they
  would *raise* the correlation this study exists to lower.
- **Trend-following systems** (MA, Donchian, TS-momentum) as the sleeve. All beaten
  by simply holding the index over this window.
- **Adding a short/directional-both-ways sleeve.** Down-trends are already the neutral
  book's best state; a short sleeve would fight it.

## 5. Honest limits

1. **Only 7 down-trend months (5 in the dense era).** The sample holds one fast
   V-shaped crash and no slow grinding bear — no 2000-03, no repeated-leg 2008. The
   down-tail is **unmeasured, not proven safe**, and a multi-quarter bear with
   sustained elevated vol would hurt the equity sleeve *and* test the claim that the
   neutral book is safe there. This is the single biggest caveat.
2. **75 months, one market.** NIFTY compounded 17.5%/yr over the window.
3. **NAS is excluded.** 90 days of live history, and an *intraday* failure horizon —
   a sleeve tuned to monthly moves is not evidence about it. It needs its own study.
4. **The C1 stock leg may not be protected by an index sleeve.** research/128 showed
   stock-strangle tails are idiosyncratic. The blend result is book-level; per-leg
   attribution is owed.
5. The demeaned test holds the sleeve's *shape* fixed while removing its mean, which
   is a stylised way to strip the premium, not a real flat-market simulation.

## 6. Recommendation

**Do not build a new system.** Allocate deliberately to long equity you already run —
Momentum-30 (₹20L), Breakout (₹10L), HA-2green (₹20L) are all long-only equity trend
books and are structurally the sleeve this study identifies. The action is
**portfolio weighting, not invention**: size the long-equity books at roughly **25–35%
of combined risk** against the short-vol books, and stop looking for an options
overlay to do a job that plain equity does better and cheaper.

Two caveats on transferring the result: those books are **higher-beta than NIFTY** and
carry their own drawdowns, so the sleeve's realised shape will be noisier than the
index proxy tested here; and the finding is **weighting guidance, not a new edge
claim**. Before acting, the per-leg attribution (limit 4) and a re-run using those
books' own return series in place of the index proxy are owed.

---

**Next:** re-run Stage B/C substituting the actual Momentum-30 / Breakout / HA
monthly series for the NIFTY proxy, and attribute the protection per neutral leg
(C1 stock book vs 45-DTE index book) — the r/128 result says these may differ.
