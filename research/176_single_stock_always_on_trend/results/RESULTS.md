# research/176 — RESULTS

# VERDICT: **NO EDGE.** The single-stock always-on trend system does not beat holding the same stock, on any timeframe, in either window half, on any of the three names Arun asked about. Nothing is adopted; no live book is touched.

**This is research/48 again, confirmed on eleven and a half years of intraday data instead of
two, and extended to twenty-one years of daily data.** The one thing r/48 could not rule out —
that its 15-minute basket kill had only seen the 2024-26 regime — is now ruled out. The result
gets *worse* with a longer window, not better.

---

## The one-paragraph answer

Arun asked whether taking **every** signal from a trend rule on **one or a few** liquid F&O
stocks — SuperTrend, an EMA crossover, or his own MST master(7,5) + child(7,2) pair — could be a
directional book to sit beside the short-vol book. Across **327,840 per-name cells** (40 signal
cells x 3 direction policies x 2 fill conventions x 3 windows x 146 names x 3 timeframes) the best
cell on the best timeframe beats simply holding the same stock on **43.2% of names** against a
pre-registered bar of 55%, and that number *falls* to 32.9% in the recent half. Every fast
timeframe is worse than daily (60-min 33.6%, 30-min 29.5%) — the edge decays monotonically with
turnover, exactly as research/56 predicted. **The long/short arm is a catastrophe**: median CAGR
−3.4% on daily and −12.0% on 60-min against buy-and-hold's +12.7%, beating the stock on 6.2% of
names. **On Arun's own three names, 0 of 40 daily cells beat buy-and-hold on RELIANCE and 0 of 40
on HDFCBANK**; ST(7,3) on RELIANCE returns +8.06% against the stock's +14.07% and takes a −63.7%
drawdown to do it. There *is* one real effect — the rule genuinely times better than a
time-in-market-matched random null on ~70% of names — but it is a **risk** edge, not a return
edge, it degrades in the recent half, and when tested as a sleeve against the live short-vol book
it is beaten by **plain cash** (+0.54 Calmar at 40% weight, versus +0.14 for the best trend book)
and by **buy-and-hold of RELIANCE** (+0.41). The options-expression phase was pre-registered to
open only if the futures signal survived. It did not, so it was not opened.

---

## 1. What was tested, and the bar that was set before it ran

| | |
|---|---|
| Universe | The **322 symbols carrying 5-minute bars from Feb-2015 to 2026** (the liquid F&O panel built in research/81), filtered to a 20-day median traded value of at least ₹10 cr → **153 names**, less **7 names carrying an unexplained overnight price step** → **146 names** in every headline figure |
| Signals | SuperTrend period {7,10,14,21} x multiplier {1.5,2,2.5,3,4,5} = 24 · EMA crossover 7 pairs · **MST master ST(7,m) m∈{4,5,6} + child ST(7,c) c∈{1.5,2,2.5}** = 9 · **40 cells** |
| Direction | long/flat · long/short · short-only (so the short leg can be judged on its own) |
| Fill | honest = **next bar's open**; reference = signal bar's close, always labelled |
| Timeframes | **daily 2006-2026 (20.7 y)**; **60-min and 30-min resampled from 5-min, 2015-2026 (11.5 y)** |
| Cost | 10 / 20 / 40 bps round trip, reported in every table; idle cash at 5.2% post-tax |
| **Pre-registered primary metric** | `beat_rate` = share of the 146 names whose **net CAGR at 20 bps** exceeds that same name's **buy-and-hold CAGR** over the same window |
| **Pre-registered gates** | **G1a** beat_rate ≥ 0.55 in *both* window halves · **G1b** neighbours ≥ 80% of the best cell · **G1c** beats a time-in-market-matched random null on ≥ 60% of names · **G2** still clears G1a at 40 bps · **G3** the short leg earns on its own · **G4** blend value vs the live short-vol book, and beats the cash-null |

The bar was written into the STATUS doc before the first cell ran. It has not been moved.

---

## 2. G1a — the gate, and how far short it falls

Best cell on each timeframe, long/flat, next-open fills, 20 bps, 146 names:

| Timeframe | best cell | beat_rate **full** | **h1** | **h2** | median CAGR | median B&H | median switches/yr |
|---|---|---|---|---|---|---|---|
| **daily** 2006-2026 | EMA(9,21) | **0.432** | 0.526 | **0.329** | +11.49% | +12.72% | 10.5 |
| **60-min** 2015-2026 | EMA(50,200) | **0.336** | 0.479 | **0.233** | +7.99% | +11.43% | 8.8 |
| **30-min** 2015-2026 | EMA(50,200) | **0.295** | — | — | +7.40% | +11.43% | 16.6 |

**Not one cell, on any timeframe, in any window, reaches 0.55.** The gate fails on the full
window and fails again in each half; and the recent half is the worse of the two everywhere, so
this is not a "the edge used to exist" story either.

Arun's own named configurations, daily, across the basket:

| cell | policy | beat20 | beat40 | median CAGR | vs B&H | Calmar-beat | switches/yr |
|---|---|---|---|---|---|---|---|
| **ST(7,3)** | long/flat | 0.322 | 0.267 | +10.54% | **−2.91pp** | 0.534 | 6 |
| **ST(7,3)** | long/short | 0.034 | 0.034 | −5.77% | **−19.20pp** | 0.034 | 6 |
| **MST 7,5 / 7,2** | long/flat | 0.322 | 0.301 | +10.10% | **−2.28pp** | 0.555 | 3 |
| **MST 7,5 / 7,2** | long/short | 0.027 | 0.027 | −3.59% | **−17.82pp** | 0.027 | 3 |
| EMA(20,50) | long/flat | 0.377 | 0.356 | +10.77% | −2.67pp | 0.541 | 5 |
| EMA(50,200) | long/flat | 0.329 | 0.315 | +10.29% | −2.40pp | 0.562 | 1 |

## 3. G1b — plateau: the whole surface is below the bar

The SuperTrend map is not a spike sitting on noise. It is a **flat, uniformly failing plane**,
which is stronger evidence than a single bad cell would be.

Daily, long/flat, beat_rate:

| period \ mult | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 | 5.0 |
|---|---|---|---|---|---|---|
| **7** | 0.185 | 0.274 | 0.342 | 0.322 | 0.329 | 0.322 |
| **10** | 0.178 | 0.253 | 0.322 | 0.288 | 0.288 | 0.342 |
| **14** | 0.151 | 0.240 | 0.274 | 0.247 | 0.315 | 0.295 |
| **21** | 0.171 | 0.212 | 0.281 | 0.253 | 0.281 | 0.281 |

Daily, long/**short**, the same 24 cells: **0.007 to 0.048**. The best SuperTrend long/short
configuration in the grid beats the stock on **seven of 146 names**.

## 4. G3 — the short leg, on its own: dead, for the fourth time in this project

Short-only, next-open, 20 bps:

| Timeframe | best short cell | beat_rate | median CAGR | median B&H | median expectancy/trade |
|---|---|---|---|---|---|
| daily | EMA(10,30) | **0.014** | −8.30% | +12.72% | **−0.0189** |
| 60-min | EMA(9,21) | 0.082 | −11.23% | +11.43% | −0.0036 |
| 30-min | EMA(20,50) | 0.082 | −10.81% | +11.43% | −0.0041 |

Median expectancy per short trade is **negative on every family, every timeframe**. This
reproduces research/81, research/82 and research/83 on a new construction and a longer window.
**Drop the short leg.** Arun's long/short MST posture on HDFCBANK, daily, returns **−9.70% a year
at a −91.2% drawdown**.

## 5. G1c — the one effect that IS real, and why it still does not help

The null: take the posture series the rule actually produced, cut it into spells, and **shuffle
the spell lengths within each posture class**. Time in market, trade count, cost bill and both
run-length distributions are preserved exactly; only *when* the long spells happen is destroyed.
200 draws per (name, cell).

Daily, long/flat, 146 names:

| cell | time in mkt | rule CAGR | null CAGR | B&H CAGR | beats own null (CAGR) | rule Calmar | null Calmar | B&H Calmar | beats own null (Calmar) |
|---|---|---|---|---|---|---|---|---|---|
| **EMA(9,21)** | 54.9% | 11.29% | 8.21% | 12.72% | **69.9%** | **0.221** | 0.132 | 0.174 | **73.3%** |
| EMA(10,30) | 55.7% | 10.95% | 8.51% | 12.72% | 63.7% | 0.204 | 0.137 | 0.174 | 67.8% |
| MST 7,5/7,2 | 59.4% | 9.99% | 9.45% | 12.72% | 59.6% | 0.187 | 0.146 | 0.174 | 64.4% |
| ST(7,3) | 54.3% | 10.37% | 8.85% | 12.72% | 55.5% | 0.191 | 0.143 | 0.174 | 60.3% |

**G1c passes for the fast EMA crossovers.** The rule is genuinely timing, not merely being out of
the market — it beats its own matched shuffle on 70% of names. But what it buys is **drawdown**
(median −51.8% against buy-and-hold's −75.7%), not return. On 60-min the same table holds
(EMA(9,21) beats its null on 68.4%) while the return collapses to **+4.77% against buy-and-hold's
+11.73%** — the timing skill is still there and turnover has eaten all of it.

And the risk edge is not stable. Calmar-beat-vs-buy-and-hold by window, daily, long/flat:

| cell | full | h1 (2006-15) | h2 (2016-26) |
|---|---|---|---|
| EMA(9,21) | 0.637 | 0.659 | **0.521** |
| EMA(10,30) | 0.596 | 0.600 | 0.575 |
| ST(7,3) | 0.534 | 0.600 | **0.404** |
| MST 7,5/7,2 | 0.555 | 0.563 | **0.452** |

In the last decade the best cell is a **coin flip** on risk-adjusted return and ST(7,3) and the
MST pair are outright losers.

## 6. Where the beat lives — and why you cannot select it in advance

EMA(9,21), long/flat, daily, by what the *stock itself* did:

| the name's buy-and-hold CAGR was | n | trend beats B&H | median excess |
|---|---|---|---|
| **below 0%** | 4 | **75.0%** | **+6.47pp** |
| 0-10% | 46 | 56.5% | +2.21pp |
| 10-20% | 79 | **35.4%** | −2.02pp |
| above 20% | 24 | 41.7% | −2.55pp |

The rule is a **loss-avoider**. It wins on the names that fell and loses on the names that rose —
and which name will fall over the next twenty years is precisely the thing you do not know when
you choose it. This is r/48's "winners were all high-vol strong-trenders" finding in its general
form, and research/134's "trend timing on top of long equity hurt" reproduced at single-stock
level.

## 7. Arun's three names, individually

Daily, 2006-2026, next-open, 20 bps, 40 cells each:

| name | buy & hold | best of 40 cells | cells beating B&H | **ST(7,3)** long/flat | **MST 7,5/7,2** long/flat |
|---|---|---|---|---|---|
| **MARUTI** | +15.46% (dd −63.7%) | EMA(9,21) **+19.25%** (dd −34.6%) | **3 / 40** | +12.54% (dd −51.8%) | +13.17% (dd −44.3%) |
| **RELIANCE** | +14.07% (dd −68.9%) | ST(7,5) +10.95% | **0 / 40** | +8.06% (dd −63.7%) | +10.95% (dd −54.2%) |
| **HDFCBANK** | +15.67% (dd −56.0%) | EMA(50,200) +13.29% | **0 / 40** | +10.86% (dd −34.0%) | +6.84% (dd −59.1%) |

60-min, 2015-2026: MARUTI 11/40, RELIANCE 4/40, HDFCBANK **0/40**.

MARUTI's three winning daily cells out of 40, across 146 names and 40 cells — 5,840 draws — is
what a null looks like, not what an edge looks like.

## 8. Arun's literal MST machine, with the lot stacking

Master 7,5 sets the regime, each child 7,2 flip in the master's direction **adds a lot up to
five**, a master reversal closes everything and the book waits to re-arm. One lot = 20% of
capital, so five lots = fully invested — a **ramp**, not leverage.

Daily, 2006-2026, next-open, 20 bps:

| book | variant | CAGR | MaxDD | Calmar |
|---|---|---|---|---|
| ARUN3 equal-weight | **stacked long/flat** | **+6.86%** | **−14.8%** | **0.464** |
| ARUN3 | stacked long/short | +4.25% | −15.0% | 0.284 |
| ARUN3 | single-unit long/flat | +11.46% | −27.9% | 0.410 |
| ARUN3 | single-unit long/short | **−2.00%** | −71.6% | −0.028 |
| ARUN3 | buy & hold | **+17.71%** | −54.7% | 0.324 |

**The stacking is the best drawdown tool in the entire study** — it cuts the three-name book's
worst loss from −54.7% to −14.8%, and 2008 from −45.1% to −9.4%. It is also the **lowest-returning
arm that is still positive**: 6.86% a year, below NIFTY 50's 8.79% and barely above the 5.2%
post-tax cash standard the project already earns on idle balances. A ramp that is only fully
invested after four confirmations is a de-levering device wearing a trading system's clothes.

## 9. G4 — the deciding test: does it add anything to the live short-vol book?

research/134's lesson is that a directional complement is judged on the blend, not standalone.
Blended against the **live short-vol book** (C1 stock winged strangles + the 45-DTE NIFTY
straddle, equal risk, 75 common months 2019-05 → 2026-07; standalone **+21.29% CAGR, −10.39% DD,
Calmar 2.05, worst month −9.27%**):

| sleeve | corr | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| **plain CASH at 5.2%** | — | **+0.09** | **+0.21** | **+0.36** | **+0.54** |
| **RELIANCE buy & hold** | −0.17 | +0.27 | +0.35 | **+0.41** | +0.10 |
| ARUN3 / EMA(10,30) | −0.18 | +0.13 | **+0.14** | +0.11 | +0.07 |
| ARUN3 / EMA(9,21) | −0.15 | +0.11 | +0.10 | +0.05 | −0.02 |
| HDFCBANK / ST(7,3) | −0.20 | +0.14 | +0.34 | +0.14 | −0.34 |
| ARUN3 / MST 7,5/7,2 | −0.15 | −0.07 | −0.22 | −0.44 | −0.66 |
| HDFCBANK / MST 7,5/7,2 | −0.08 | −0.27 | −0.55 | −0.87 | −1.21 |

(figures are the change in blend Calmar versus the short-vol book alone)

**Every single directional sleeve is beaten by doing nothing with the money.** The cash-null
raises Calmar by +0.54 at 40% weight; the best trend book manages +0.14 at 20%. And where a
directional sleeve *does* beat cash at low weight, it is **buy-and-hold of RELIANCE**, not any
trend rule applied to it. The pre-registered G4 bar was "+0.10 Calmar or −2pp drawdown **at equal
or better return**"; **every** blend in the table cuts CAGR (21.29% → 19.65% at only 10%
weight), so no cell clears it on either leg. This is research/134's conclusion — *the diversifier
is plain long equity and trend timing on top of it hurts* — reproduced independently on single
stocks.

## 10. The house YoY table

ARUN3 = MARUTI + RELIANCE + HDFCBANK equal-weight. Each cell: annual return with the intra-year
maximum drawdown beneath. Daily bars, next-open fills, 20 bps round trip, idle cash 5.2%.
Pre-tax (futures P&L is business income in India, not STCG). Full table:
`research/176_single_stock_always_on_trend/results/yoy_table.md`.

| | EMA(9,21) long/flat | ST(7,3) long/flat | ST(7,3) long/short | MST stacked long/flat | **buy & hold** | NIFTY 50 |
|---|---|---|---|---|---|---|
| **2006-2026 CAGR** | +13.88% | +11.86% | **−1.03%** | +6.86% | **+17.71%** | +8.79% |
| **MaxDD** | −28.0% | −40.5% | −57.9% | **−14.8%** | −54.7% | −38.4% |
| **Calmar** | **0.495** | 0.293 | −0.018 | 0.464 | 0.324 | 0.229 |

The honest reading of that row: the best thing this study found is a **Calmar-0.50 book** at
−28% drawdown. The books already running are True North 0.88, Open Alpha 1.24, and the deployed
TN+OA pair **1.68**. Even the best-case construction is less than a third as good as the pair
Arun already owns, before any blend consideration.

Cost ladder on that best book (ARUN3, EMA(9,21), long/flat):

| round-trip cost | CAGR | MaxDD | Calmar |
|---|---|---|---|
| 10 bps | +14.44% | −27.7% | 0.521 |
| 20 bps | +13.88% | −28.0% | 0.495 |
| 40 bps | +12.75% | −28.8% | 0.443 |
| 60 bps | +11.64% | −29.6% | 0.394 |
| **buy & hold** | **+17.71%** | −54.7% | 0.324 |

Costs are not what kills it on daily — **holding the stock is**. Costs *are* what kills the
intraday arms: the 30-minute long/flat cells run 16 to 128 switches a year and their beat_rate
halves between 20 bps and 40 bps (0.295 → 0.212, and EMA(10,30) 0.288 → 0.096).

## 11. G5 — the options expression was never opened, by pre-registration

The plan committed to testing the "options selling hedge on the child ST" leg **only if the
futures signal survived G1-G4**. It did not survive any of them. Two additional reasons it would
not have rescued anything:

- **research/56 already ran exactly this** — the dual-SuperTrend MST/CST regime expressed as
  credit structures on 30-min NIFTY. Gross trend capture was real (Calmar 1.79 gross) and the
  as-specced always-on credit book still returned **−₹17k to −₹62k per 6 weeks**, because the
  break-even is ~10 bps per posture change and a multi-leg options round-trip costs more. The one
  positive variant came from *waiting* — the opposite of always-on.
- **research/150** killed five option structures built on high-win-rate signals: an options
  overlay changes the **payoff shape**, it does not create expectancy. It cannot rescue a posture
  whose underlying expectancy is already below buy-and-hold.

Arun's phrase "*need not be fully directionally trending... even with options system to manage*"
is the hope that structure can compensate for a weak directional read. On the evidence of r/56,
r/129 and r/150 — four independent kills of that family in this project — it cannot.

---

## 12. Seven deadly sins — how each was controlled

| Sin | Control |
|---|---|
| **Look-ahead** | Every posture is decided at a bar close and filled at the **next** bar's open. The signal-close fill is carried only as a labelled reference arm, and it moved the daily long/flat result by +0.35pp of CAGR and +0.034 of beat_rate — never enough to change a verdict. |
| **Survivorship** | The universe is the 5-minute panel as it exists today, so names delisted before 2015 are absent. This biases the study **in favour** of the trend rule's opponent (buy-and-hold) — and buy-and-hold still won, so the bias does not rescue the finding. Stated, not hidden. |
| **Overfitting / multiple testing** | 327,840 name-cells; the plateau map is reported in full and is a flat failing plane, not a spike. The single "winner" — EMA(9,21) on MARUTI — is 1 of 5,840 daily draws. |
| **Cost neglect** | 10 / 20 / 40 bps round trip in every table, plus a 60 bps book-level rung; switches per year carried in every row. |
| **Regime dependence** | Two windows on every timeframe, both reported; the recent half is worse everywhere. |
| **Correlation / single factor** | The blend against the live short-vol book, against the cash-null, and against buy-and-hold of the same names. |
| **Capacity / shortability** | Futures on the 146 most liquid F&O names; capacity is not the binding constraint here, the absent edge is. Single-stock futures roll cost is modelled at 5 bps a month on top of the round trip. |

## 13. Caveats, led rather than buried

- **P&L is computed on the underlying cash series as a proxy for the front-month future** — this
  is Maruthi live-bug #9 (`memory/maruthi_algo_bugs.md`), deliberately inherited and disclosed.
  Real futures carry a basis that decays into expiry and a roll each month. The proxy therefore
  **flatters** the always-on book by omitting roll slippage, and the verdict is negative anyway.
- `market_data.db` **is not retroactively split-adjusted**. 14 of the 321 panel names carry an
  unexplained overnight step outside 0.65x-1.55x; the 7 of those inside the liquid basket are
  excluded from every headline figure. The scan and the flagged dates are in
  `results/universe.csv`.
- Intraday bars are **resampled from 5-minute**, with 30/60-minute bins anchored at 09:15. Spreads
  and market impact inside a bar are modelled only through the cost ladder.
- The blend uses research/134's 75-month combined short-vol series (2019-05 → 2026-07), which is
  shorter than the daily equity window and covers one broad regime.
- **Not tested, and why:** the options expression (pre-registered as gated on G1-G4, which failed);
  15-minute and 5-minute bars (the 30/60-minute trend is monotonically worse than daily, so faster
  is a dead direction); pyramiding beyond 5 lots; per-name parameter selection (that *is* the r/48
  overfit).

---

## 14. What this leaves on the shelf

1. **The single-stock always-on trend line is CLOSED.** Cite this study and research/48 together:
   the family has now failed on 1 name x 2 years intraday (r/48), 381 names x 2 years intraday
   (r/48), and 146 names x 11.5 years intraday plus 20.7 years daily (r/176), across SuperTrend,
   EMA crossover and the MST pair.
2. **The block-permutation null is reusable and sharp.** Preserving time-in-market, trade count
   and both run-length distributions while destroying only the timing is the cleanest way this
   project has found to separate "the rule works" from "being long works". It is what showed the
   trend rule has real skill *and* that the skill is worth nothing here.
3. **A fourth confirmation that stops and ramps buy drawdown, never return.** The MST lot-stack
   cut the three-name drawdown from −54.7% to −14.8% and the return from 17.7% to 6.9%. Same shape
   as research/172 (the 52-week channel "picks smoother paths, not higher returns") and
   research/174 (stops are insurance at a real premium).
4. **If Arun wants a non-index directional book, the answer research/134 gave still stands and is
   now confirmed at single-stock level: own the equity, do not time it** — and the equity he
   should own is already owned, through True North, Open Alpha and IPO Base.
