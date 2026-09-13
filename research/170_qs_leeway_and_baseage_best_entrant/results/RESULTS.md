# research/170 — Quality Summit's rank leeway · and Base Age's best-qualifying entrant

Two unrelated questions Arun asked on the morning of 13-Sep-2026, on two different books.
Read them separately. **Nothing here is deployed.**

| Part | Book | Question | Verdict |
|---|---|---|---|
| **A** | **QS** — Quality Summit (research/160 Family-B `b7`) | Give a slipping holding a rank leeway before selling it? How wide? Does the band `k` move with it? | **CONCLUDED — NO ADOPTION. The premise is wrong and the axis is dead.** The book already keeps a name to **rank 23**, not 15; only **14.5%** of its sales are rank sales at all; and **every** leeway width tested loses to the incumbent on paired evidence. |
| **B** | **OA · Base Age** — Open Alpha on aged deep bases | When the book is full and a signal arrives, which refused entrant gets the freed slot? | **SIGNAL — the post-hoc pick REPLICATED, and still sits exactly on the adoption bar.** Highest-RS entrant wins **60 of 60** paths against the incumbent and **60 of 60** against a random swap, in both windows, at every cost rung — and delivers **+0.096 paired Calmar against a pre-registered +0.10**. **Do not change the live book on this. It is a proposal with a dated review.** |

---

# PART A — QS · the rank leeway

## **VERDICT: CONCLUDED — NO ADOPTION**

**Window 2018-08-01 → 2026-09-10 (8.1 years). 54 selection cells + 48 validation cells.
After tax (20% STCG / 12.5% LTCG, Indian FY netting), net of 25 bps a side, idle cash 5.2%
post-tax credited daily, 12 rebalance-day offsets per cell, medians across offsets.
Drawdowns measured from the running peak of the full curve.**

### Harness proof, run before any selection cell

This study's engine is research/160's `qg_engine.py`, copied and patched by an auditable
11-patch script. Run through it at research/160's own 5.0% idle-cash assumption, the
incumbent returns **21.19% / −37.07% / Calmar 0.58** — research/160's published `F_Bb7` row
to the second decimal. At Arun's current 5.2% standard it returns **21.39% / −36.90% /
0.58**, and that is the baseline every number below is measured against.

## Q&A — Arun's four questions, one at a time

**Q1. Is a holding sold at rank 16 today?**

**No. It is kept, and so is one at rank 23.** The deployed spec already carries a rank
leeway and it is wider than the one Arun proposed. The engine keeps a holding while its
relative-strength rank sits inside `ceil(buffer × N)` of today's qualifying names; at the
published `buffer = 1.5` and `N = 15` that is `ceil(22.5) = 23`. A name is sold on rank only
at **24th or worse**.

**But there is a second exit, and it is the one that actually fires.** A holding is also
sold — at any rank, even 1st — if it stops *qualifying*: below `k = 0.90 ×` its own causal
all-time-high close, below the ₹2 cr traded-value floor, or off the point-in-time `b7`
screen. This study instrumented the engine to separate the two. At the incumbent setting:

| what caused the sale | share of all closed trades |
|---|---:|
| **the state** — the name left the near-ATH band / liquidity / screen | **85.5%** |
| **the rank** — it still qualified, but ranked outside the leeway | **14.5%** |

**Six of every seven sales are not rank sales.** Arun's proposal can only ever touch the
remaining one, which is why the rest of this part reads the way it does.

**Q2. What leeway is best — rank 22, 25, 30?**

**None of them. The incumbent's rank 23 is the best of the seven widths tested, and the
whole axis is flat.** Seven leeway widths at the incumbent construction (N = 15, monthly):

| leeway (keep to rank) | `buffer` | CAGR **pre-tax** | CAGR **after tax** | MaxDD | Calmar | trades/yr | turnover ×NAV | avg hold (days) | % trades held > 365d | tax paid on ₹1 cr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 — no leeway at all | 1.00 | 22.82 | 19.80 | −37.56 | 0.55 | 74.2 | 5.06 | 66 | 0.3% | ₹64.1 L |
| 20 | 1.33 | 24.48 | 21.19 | −37.02 | 0.58 | 64.6 | 4.42 | 78 | 0.6% | ₹70.4 L |
| **23 — the incumbent** | **1.50** | **24.63** | **21.39** | **−36.90** | **0.58** | **61.6** | **4.21** | **81** | **0.7%** | **₹70.3 L** |
| 26 — Arun's "top 25" | 1.67 | 23.95 | 20.76 | −36.58 | 0.57 | 59.8 | 4.06 | 83 | 0.8% | ₹68.3 L |
| 30 | 2.00 | 24.34 | 21.39 | −38.63 | 0.51 | 57.4 | 3.93 | 87 | 1.0% | ₹66.7 L |
| 38 | 2.50 | 24.32 | 21.40 | −40.30 | 0.53 | 54.9 | 3.74 | 91 | 1.2% | ₹65.7 L |
| 45 | 3.00 | 23.96 | 20.70 | −37.91 | 0.57 | 53.6 | 3.59 | 93 | 1.6% | ₹66.0 L |

**Paired against the incumbent on the same 12 offsets, every width loses**, and the
pre-registered bar was ≥ +0.15 Calmar or ≥ +2pp CAGR at no worse drawdown on ≥ 8 of 12:

| candidate | ΔCAGR | CAGR wins | ΔCalmar | Calmar wins | ΔMaxDD | DD wins |
|---|---:|---:|---:|---:|---:|---:|
| rank 15 (no leeway) | −2.02pp | 3/12 | −0.047 | 2/12 | −1.09 | 4/12 |
| rank 20 | −0.15pp | 4/12 | −0.011 | 4/12 | −0.27 | 3/12 |
| **rank 26 — Arun's "top 25"** | **−0.09pp** | **5/12** | **−0.003** | **5/12** | −0.62 | 5/12 |
| rank 30 | −0.39pp | 4/12 | −0.024 | 4/12 | −1.16 | 3/12 |
| rank 38 | −0.57pp | 3/12 | −0.063 | 2/12 | −2.11 | 1/12 |
| rank 45 | −0.84pp | 2/12 | −0.026 | 1/12 | −1.18 | 3/12 |

Arun's own proposal is the closest thing to a tie in the table — a **dead wash**, −0.09pp of
CAGR and −0.003 of Calmar, winning five of twelve offsets. Nothing here is an improvement,
and nothing here is a disaster either; the axis simply does not matter at this construction.

**The one real signal on the axis is at the narrow end, and it is negative.** Removing the
leeway entirely — keep only the top 15 — costs **2.02pp of CAGR on 9 of 12 offsets**. So the
leeway that already exists is earning its keep. Widening it beyond 23 earns nothing more.

**The axis was also crossed with book size and cadence — 42 cells.** At N = 10 and N = 20,
and at quarterly as well as monthly, the same shape holds: the leeway curve is flat with a
shallow top between rank 20 and rank 30, and the best cells in the whole 42 are cells that
change the *cadence and the book size*, not the leeway (N = 10 quarterly, 24.80% after tax
at Calmar 0.58 — the same Calmar as the incumbent, with 5.3 more points of drawdown).

**Q3. What does the leeway buy in tax and churn?**

**Churn: a lot. Tax: essentially nothing — and this is the part of Arun's reasoning that
does not survive contact with the data.**

Going from rank 15 to rank 45, churn falls exactly as he expects: trades **74.2 → 53.6 a
year**, turnover **5.06 → 3.59 × NAV**, average holding period **66 → 93 days**.

But the tax saving he is reaching for requires crossing the **365-day** line, and **this
book never gets close**. The share of closed trades held beyond a year rises only from
**0.3% to 1.6%**. Ninety-eight of every hundred round trips are still short-term at the
widest leeway tested. The rupee tax bill on a ₹1 crore book over 8.1 years moves from
₹70.3 lakh to ₹65.7 lakh — and most of that fall is simply a smaller book making smaller
gains, not a rate change.

**The reason is structural.** A monthly-rebalanced RS book with 15 slots replaces roughly
four names a month. Even with no rank pressure at all, a name has to stay in the top 45 of a
fast-moving momentum ranking *and* inside its own near-ATH band for twelve consecutive
months to reach long-term treatment. Widening the rank band cannot manufacture that,
because the rank is not what is ending most of these holdings. **At quarterly cadence the
long-term share does rise — to 9.7% at rank 60 — and the after-tax CAGR falls to 18.75%.**
You can buy the tax rate on this book. It costs more than it saves.

**Q4. Does the band k = 0.90 change?**

**No. k = 0.90 is the Calmar peak at every leeway width tested, and the one cell that beats
it in the fit window reverses in the holdout — exactly as research/162 found by a different
route.**

| band `k` | leeway 23 | leeway 26 | leeway 38 |
|---|---|---|---|
| 0.75 | 22.35 / −41.72 / **0.53** | 21.91 / −41.85 / 0.53 | 19.03 / −42.58 / 0.45 |
| 0.80 | 20.87 / −40.15 / 0.52 | 21.03 / −39.88 / 0.53 | 18.25 / −39.53 / 0.46 |
| 0.85 | 22.32 / −37.62 / 0.57 | 22.35 / −37.23 / 0.57 | 21.03 / −37.52 / 0.53 |
| **0.90** | **21.39 / −36.90 / 0.58** | 20.76 / −36.58 / 0.57 | 21.40 / −40.30 / 0.53 |
| 0.95 | 17.16 / −37.98 / 0.46 | 17.15 / −37.96 / 0.46 | 16.80 / −38.07 / 0.45 |

*(after-tax CAGR % / MaxDD % / Calmar)*

`k = 0.85` earns more (22.32% vs 21.39%) at the same drawdown, and in the **fit window** it
looks like a genuine finding: **+2.32pp of CAGR and +0.156 of Calmar on 9 of 12 offsets**,
which clears the pre-registered bar outright. **The holdout kills it**: −1.09pp of CAGR on
5 of 12 and −0.063 of Calmar on 4 of 12. research/162 found the same reversal at a different
construction (inverse-vol sizing, N = 10) and called it a fit to the 2020-21 leg. **This is
now the second independent replication of that reversal, and it should be treated as
settled: the 0.90 band is not a parameter worth re-opening.**

There is also a mechanical reason the band and the leeway are not independent: as `k`
loosens, names stop failing the state test and start failing the rank test instead. The
share of rank sales rises from **4.8% at k = 0.95** to **14.5% at k = 0.90** to **50.7% at
k = 0.75**. A leeway only has something to do once the band is loose — and a loose band is
worse.

## The control that settles it: the *other* leeway

Arun's question is about the rank. The instrumentation says the rank is not what ends most
holdings. So this study ran the version of "leeway" that would actually bite: **`retain =
loose` — keep a holding whose rank is fine even after it has fallen out of the near-ATH
band.**

| | CAGR after tax | MaxDD | Calmar | avg hold | % invested |
|---|---:|---:|---:|---:|---:|
| incumbent | **21.39** | **−36.90** | **0.58** | 81 days | 91.3 |
| keep fallen names (`retain = loose`) | 19.88 | −42.35 | **0.47** | 209 days | 99.2 |

**Paired: −1.42pp of CAGR and −0.146 of Calmar, winning 2 of 12 offsets.** Holding on to
names that have left their highs lengthens the average hold from 81 days to 209 — the
tax-rate effect Arun wants — and **costs 5.5 points of drawdown and 1.5 points of return to
get it.** The near-all-time-high requirement is doing protective work. Loosening the grip is
the wrong direction, whichever form the loosening takes.

## Null control

Same book, same leeway, ranking replaced by a coin toss, 30 seeds:

| | after tax | Calmar | % of sales that are rank sales |
|---|---:|---:|---:|
| RS ranking, leeway 23 | **21.39** | **0.58** | 14.5% |
| random ranking, leeway 23 | 11.73 | 0.31 | 43.7% |
| random ranking, leeway 15 | 11.18 | 0.28 | 51.8% |
| random ranking, leeway 38 | 13.53 | 0.36 | 29.5% |

Two things fall out. **The relative-strength ranking is worth +9.7pp of after-tax CAGR** —
consistent with research/160's +7.7pp over random selection, and it remains the entire
engine of this book. And **the leeway helps the random book too** (Calmar 0.28 → 0.31 →
0.36 as it widens), which means whatever small benefit a wider band confers is
**churn reduction, not better selection**. It is not a ranking finding.

## Robustness on the incumbent and the near-misses

| | 25 bps | 40 bps | 60 bps | 0% idle cash | missing = pass | W1 fit | W2 holdout |
|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent (rank 23) | 21.39 / 0.58 | 20.67 / 0.55 | 18.83 / 0.49 | 21.01 | 21.39 | 20.29 / 0.61 | 21.44 / 0.59 |
| rank 26 (Arun's) | 20.76 / 0.57 | 19.82 / 0.53 | 18.10 / 0.47 | 20.72 | 20.76 | 20.74 / 0.60 | 20.20 / 0.55 |
| rank 38 | 21.40 / 0.53 | 20.39 / 0.49 | 18.76 / 0.43 | 20.68 | 21.40 | 20.37 / 0.61 | 19.41 / 0.52 |
| no leeway (rank 15) | 19.80 / 0.55 | 18.56 / 0.49 | 16.57 / 0.43 | 19.22 | 19.80 | 18.93 / 0.54 | 20.11 / 0.56 |

The incumbent dominates at every cost rung. It is also the most *stable* cell in the study —
20.29% in the fit window against 21.44% in the holdout, a **+1.15pp gap**, comfortably inside
the pre-registered 4pp rule, while the two cells that beat it in the fit window (`k = 0.85`
at −2.79pp and N = 10 quarterly at −8.92pp) both deteriorate.

## Tradeability gate — Quality Summit at three leeways

| leeway | win rate | avg win | avg loss | max losing streak | trades/yr | capacity (position ÷ name's own tv20) |
|---|---:|---:|---:|---:|---:|---:|
| rank 15 | 47.2% | +22.5% | −10.0% | 17 | 74.2 | 0.65% |
| **rank 23 (incumbent)** | 46.0% | +27.8% | −10.8% | 17 | 61.6 | 0.71% |
| rank 38 | 45.5% | +31.0% | −11.2% | 17 | 54.9 | 0.66% |

A **17-trade losing streak** is unchanged by the leeway and remains this book's real
tradeability problem for a discretionary operator — the leeway does not fix it.

## Part A YoY — house format

After tax, net of costs, medians across 12 offsets; each cell is the year's return with the
intra-year max drawdown beneath it, measured from the running peak of the full curve.
Benchmarks are excluded from the best-of picks.

| Year | no leeway (15) | **incumbent (23)** | leeway 26 | leeway 38 | band k 0.85 | NIFTY 50 | Midcap 150 | BEST CAGR | LEAST DD | BEST OVERALL |
|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | −12.2 (−18.7) | −12.2 (−18.7) | −12.2 (−18.7) | −12.2 (−18.7) | −13.2 (−21.4) | −4.3 (−14.6) | −5.1 (−19.4) | no leeway | no leeway | no leeway |
| 2019 | +21.8 (−15.9) | +24.3 (−16.2) | +24.4 (−16.2) | +25.3 (−16.2) | +25.5 (−18.0) | +12.0 (−11.4) | −0.3 (−21.0) | k 0.85 | no leeway | leeway 38 |
| 2020 | +32.3 (−33.3) | +39.9 (−33.4) | +37.8 (−33.6) | +35.1 (−34.2) | +29.6 (−32.8) | +14.9 (−38.4) | +24.4 (−40.8) | **incumbent** | k 0.85 | **incumbent** |
| 2021 | +81.1 (−13.7) | +86.9 (−13.9) | +86.5 (−13.6) | +87.6 (−14.0) | +90.6 (−13.4) | +24.1 (−10.1) | +46.8 (−10.4) | k 0.85 | k 0.85 | k 0.85 |
| 2022 | −18.4 (−33.3) | −19.0 (−33.0) | −20.7 (−33.3) | −18.7 (−33.8) | −7.5 (−26.9) | +4.3 (−17.2) | +3.0 (−21.6) | k 0.85 | k 0.85 | k 0.85 |
| 2023 | +52.4 (−29.2) | +58.9 (−27.9) | +58.5 (−28.7) | +65.2 (−27.8) | +61.6 (−19.2) | +20.0 (−9.9) | +43.7 (−10.4) | leeway 38 | k 0.85 | k 0.85 |
| 2024 | +31.3 (−13.9) | +36.0 (−14.1) | +34.7 (−14.0) | +26.9 (−14.3) | +26.9 (−14.1) | +8.8 (−10.9) | +23.8 (−11.0) | **incumbent** | no leeway | **incumbent** |
| 2025 | −17.5 (−36.1) | −16.8 (−34.9) | −17.2 (−34.6) | −21.4 (−37.0) | −17.3 (−37.0) | +10.5 (−15.8) | +5.4 (−21.1) | **incumbent** | leeway 26 | **incumbent** |
| 2026 | +15.0 (−35.6) | +15.7 (−34.4) | +15.1 (−34.5) | +12.9 (−38.5) | +15.5 (−35.1) | −10.2 (−15.2) | +2.8 (−14.1) | **incumbent** | **incumbent** | **incumbent** |
| **CAGR / MaxDD** | 19.80 / −37.6 | **21.39 / −36.9** | 20.76 / −36.6 | 21.40 / −40.3 | 22.32 / −37.6 | 9.38 / −38.4 | 16.41 / −40.8 | | | |

The incumbent takes BEST OVERALL in five of nine years — more than any other column — and
2026 outright. Note that **Quality Summit is not a book anyone is running**: research/160
and research/162 both concluded it is dilutive to the deployed TN+OA pair at every weight,
and this study changes nothing about that.

## What Part A changes

**Nothing operationally.** Quality Summit keeps the spec research/160 published: k = 0.90,
15 names, RS ranking, monthly, leeway to rank 23. The value of this part is three facts that
did not exist yesterday:

1. **The book already does what Arun asked for, and more generously.** Rank 16 is not a sale.
2. **The rank is not the binding constraint — the near-ATH state is** (85.5% of sales).
   Any future question about "holding on longer" on this family should be aimed at the band
   and the state, not at the rank.
3. **This family's holding period cannot be taxed into long-term treatment.** Every route
   tried (wider rank band, quarterly cadence, keeping fallen names) either fails to move the
   long-term share or pays for it in return.

---

# PART B — OA · Base Age: the best-qualifying entrant, confirmed

## **VERDICT: SIGNAL — the post-hoc pick REPLICATED on fresh seeds, and still sits exactly on the bar. Not adopted.**

**Window 2005-01-03 → 2026-09-11 (21.7 years). 5 selection cells. After tax (20% STCG /
12.5% LTCG, Indian FY netting), net of 25 bps a side, idle cash 5.2% post-tax credited daily,
16 slots @ 6.25%, ₹10 lakh book, 30 seeds per cell on each of two independent seed sets.**

### Harness proof, and the seed sets

research/166's harness was copied byte-identical. On research/166's own seeds (1–30) it
reproduces every published row: incumbent **20.98 / −34.05 / 0.613**, the pre-registered
rotation winner **22.68 / −33.52 / 0.673**, the post-hoc entrant cell **22.73 / −31.78 /
0.715**, the rate-matched random null **20.39 / −35.75 / 0.569**. The primary evidence below
is on **seeds 1001–1030**, which no cell in research/164 or research/166 has ever touched.

## Q&A

**Q1. When the book is full and the best signal arrives, should we swap, and which holding
goes?**

**Yes, sell the holding that is more than 10% under water — and give the freed slot to the
refused entrant with the highest 12-month relative strength. On fresh seeds that rule clears
the bar; pooled across both seed sets it misses it by four thousandths.**

| | incumbent (never swap) | entrant = **rs252** (the pick under test) | entrant = tv20 (r/166's pre-registered) | entrant = base age | rate-matched RANDOM swap |
|---|---:|---:|---:|---:|---:|
| **CAGR after tax — FRESH seeds** | 20.95% | **22.58%** | 22.77% | 22.50% | 20.80% |
| worst of 30 fresh seeds | 19.42% | **21.78%** | 21.45% | 20.52% | 18.79% |
| **MaxDD — FRESH seeds** | −34.05% | **−31.78%** | −33.52% | −37.23% | −34.42% |
| **Calmar — FRESH seeds** | 0.611 | **0.710** | 0.671 | 0.605 | 0.603 |
| paired ΔCalmar vs incumbent, **fresh** | — | **+0.105 on 30/30** | +0.058 on 28/30 | −0.018 on 11/30 | +0.001 on 15/30 |
| paired ΔCAGR vs incumbent, **fresh** | — | **+1.64pp on 30/30** | +1.86pp on 30/30 | +1.60pp on 28/30 | +0.00pp on 15/30 |
| paired ΔCalmar vs the **null**, fresh | — | **+0.115 on 30/30** | +0.058 on 27/30 | −0.015 on 14/30 | — |
| paired ΔCalmar vs incumbent, **r/166 seeds** | — | +0.094 on 30/30 | +0.049 on 27/30 | −0.004 on 15/30 | −0.052 on 3/30 |
| **pooled across all 60 paths — ΔCalmar** | — | **+0.096 on 60/60** | +0.051 on 55/60 | −0.013 on 26/60 | −0.037 on 18/60 |
| **pooled across all 60 paths — ΔCAGR** | — | **+1.65pp on 60/60** | +1.77pp on 60/60 | +1.75pp on 58/60 | −0.34pp on 22/60 |
| W1 2005-2015 paired ΔCAGR, fresh | — | +1.26pp on **30/30** | +1.49pp on 30/30 | +1.18pp on 30/30 | −0.37pp on 11/30 |
| W2 2016-2026 paired ΔCAGR, fresh | — | +2.23pp on **30/30** | +2.30pp on 26/30 | +2.04pp on 22/30 | +0.51pp on 18/30 |
| swaps / yr | 0.0 | 4.5 | 4.3 | 4.4 | 3.9 |
| tax paid over the window (₹10 L book) | ₹84.1 L | **₹105.4 L** | ₹111.5 L | ₹108.2 L | ₹86.8 L |

**Two readings, and both belong in the answer.**

*In favour.* It is not a fluke of the seeds it was found on. The rule wins **60 of 60 paths**
on Calmar and **60 of 60** on CAGR against the incumbent, **60 of 60** against a random swap
fired at the same rate, in **both** pre-registered windows, and the advantage **widens
relative to the alternative as costs rise** (see the ladder below). research/166 found this
cell after looking at results; it has now been named in advance and re-run on seeds it had
never seen, and it came back.

*Against.* The pre-registered bar is **+0.10 paired Calmar or +2pp CAGR at no worse
drawdown**, and the pooled figure is **+0.096 and +1.65pp**. Under the rule as written it
**does not clear**, by four thousandths of a Calmar point — the same knife-edge, in the same
direction, that research/166 reported at +0.094. Two independent seed sets have now put this
rule within one hundredth of the threshold from below. *The bar does not resolve this
question, and moving the bar after seeing the number is exactly what the bar exists to
prevent.*

**Q2. Which entrant, and why does it matter?**

**The entrant choice does not change how much the book earns. It changes how far it falls.**
This is the cleanest new fact in the study:

| entrant priority | ΔCAGR vs incumbent (pooled 60) | MaxDD (fresh) | Calmar (fresh) |
|---|---:|---:|---:|
| highest 12-month relative strength (`rs252`) | +1.65pp | **−31.78%** | **0.710** |
| largest traded value (`tv20_cr`) | +1.77pp | −33.52% | 0.671 |
| oldest base (`x_bars`) | +1.75pp | −37.23% | 0.605 |

All three earn the same **+1.6 to +1.8pp**. The entire spread — **5.5 points of drawdown and
0.105 of Calmar** — sits in which refused breakout you buy. **Who leaves the book sets the
return; who enters it sets the drawdown.** The oldest-base entrant actively *loses* to the
incumbent on Calmar (−0.018, winning 11 of 30 fresh seeds) despite earning more, which is
the same "leverage, not selection" failure research/166 found on the sell side.

That spread cuts both ways for the multiple-testing question: a 1-of-3 axis that moves
Calmar by 0.105 is a real lever, and it is also precisely the kind of axis a grid search
exploits. What tips it toward real is the **60/60** consistency and the fact that the rs
variant is the one that *reduces drawdown*, which is not what a return-chasing overfit looks
like.

**Q3. Is the margin a plateau?**

**It is a hump, as research/166 said — but every point on it beats the incumbent.** Fresh
seeds, the rs entrant:

| holding must be under water by | CAGR | MaxDD | Calmar | ΔCalmar vs incumbent |
|---|---:|---:|---:|---:|
| ≥ 7.5% | 21.93% | −32.13% | 0.683 | **+0.072 on 30/30** |
| **≥ 10%** | 22.58% | −31.78% | **0.710** | **+0.105 on 30/30** |
| ≥ 12.5% | 21.71% | −33.26% | 0.645 | +0.026 on 26/30 |

Both immediate neighbours agree in direction and both win ≥ 26 of 30 paths, so the
pre-registered plateau clause passes — but the rule needs its threshold roughly right, and
the working band is narrow (7.5%–12.5%, firing 2.4 to 6.5 swaps a year). Note also that the
7.5% variant earns its edge in **W1** (+2.46pp) and *loses* in W2 (−0.75pp on 9/30), while
the 12.5% variant does the opposite (−0.22pp W1, +1.72pp W2). **Only the 10% margin wins
both windows.** That is a point in its favour and a warning that the band is thin.

**Q4. Does it survive costs and tax?**

**Yes, and the gap is more robust than the incumbent's.** Fresh seeds, Calmar:

| | 25 bps | 40 bps | 60 bps |
|---|---:|---:|---:|
| incumbent | 0.611 | 0.589 | 0.561 |
| **entrant = rs252** | **0.710** | **0.684** | **0.647** |
| entrant = tv20 | 0.671 | 0.631 | 0.590 |
| entrant = base age | 0.605 | 0.569 | 0.534 |

The advantage over the incumbent is +0.099 / +0.095 / +0.086 across the ladder. Tax is
modelled through the FY-netting engine, never haircut: the rule pays **₹105.4 lakh against
the incumbent's ₹84.1 lakh** on a ₹10 lakh book over 21.7 years — **+25%** — and the
+1.65pp of CAGR is what is left after paying it.

**Q5. What does it cost in tradeability and capacity?**

| | incumbent | entrant = rs252 |
|---|---:|---:|
| win rate | 48.9% | 47.3% |
| max losing streak | 14 | 15 |
| trades / yr | 31.8 | 34.4 |
| turnover × NAV | 2.49 | 2.80 |
| profit from the ten best realisations | 37.5% | 43.7% |
| compounding proxy ÷ same with the ten best trades deleted | 1.16 × 10⁵ | 2.16 × 10⁵ |
| median position as % of the held name's own 20-day traded value | 0.43% | 0.50% |
| trades above 1% of the name's tv20 | 33.1% | 37.5% |

All of it moves the wrong way, modestly: you book more small losses on purpose, the worst
streak lengthens by one, and dependence on the ten best names rises. Capacity is unchanged
in substance — on a ₹1 crore book every capacity figure is ten times these, i.e. a median
position of 5.0% of the name's daily traded value, which is executable but no longer trivial.

**Q6. Should the LIVE Base Age book (research/165, staged) adopt it? What changes in the
runbook?**

**No — not on this evidence, and not this week.** Four reasons, three of them research/166's
and unchanged:

- **It misses the pre-registered bar**, pooled, by 0.004 Calmar. That is the rule.
- **The book has never traded live.** research/165 is converting it *now*. Adding a second
  mechanic on day one is the wrong order of operations, and the first live months are the
  only clean read we will ever get of the base spec.
- **It requires live machinery that does not exist**: an evening process that scores every
  holding's unrealised P&L against a −10% threshold, ranks the refused signals by rs252, and
  places two orders at the next open. That is real operational surface, and research/166's
  own finding — that a plain unconditional −10% hard stop reaches Calmar 0.677 with *none*
  of the machinery — remains the cheaper lever for anyone who wants the drawdown down.
- **The blend is untested.** Rotation tilts the held book toward younger positions. Base
  Age's correlation to True North and IPO Base was measured on the un-rotated spec.

**The proposal, written out so it is ready if Arun wants it later** (a proposal, not a
change; no executor file is touched by this study):

> **Rule OA-ROT-1 (proposed, not adopted).** On any session where the Base Age scanner
> produces at least one qualifying entry the book cannot take because no slot is free or
> cash is short: at the 15:18 IST close-proxy check, compute each holding's unrealised
> return against its entry price. If the weakest is **more than 10% below entry**, queue a
> sell of that holding and a buy of the refused entrant with the **highest 252-day relative
> strength**, both as next-open marketable-limit orders. **At most one swap per session.** A
> position entered the same day is not eligible to be swapped out. If the buy cannot be
> funded after the sale, cancel both legs.

**Pre-registered pass criterion for the dated review (2027-03-13, six months of live
operation):** the live entry queue must show the shape the rule needs — signals refused
while a holding sits more than 10% under water, at roughly 4 occurrences a year. If it does,
re-run this part's five cells on the live event log plus the extended history and apply the
**same +0.10 Calmar bar, unchanged**. If the live queue does not show that shape, the rule is
inapplicable regardless of the backtest and should be dropped.

## Part B YoY — house format

After tax, net of costs, medians across 30 **fresh** seeds. Each cell is the year's return
with the intra-year max drawdown beneath it, from the running peak of the full curve.

| Year | incumbent | **entrant rs252** | entrant tv20 | entrant base age | random-swap null | NIFTYBEES | BEST CAGR | LEAST DD | BEST OVERALL |
|---|---|---|---|---|---|---|---|---|---|
| 2005 | +14.1 (−12.3) | +15.0 (−12.3) | +12.8 (−12.7) | +14.9 (−12.5) | +14.1 (−12.4) | +32.8 (−14.0) | **rs252** | **rs252** | **rs252** |
| 2006 | +41.5 (−20.6) | +32.8 (−21.0) | +37.2 (−20.6) | +31.5 (−21.6) | +40.6 (−20.6) | +41.3 (−29.9) | incumbent | incumbent | incumbent |
| 2007 | +84.7 (−10.8) | +82.4 (−9.8) | +82.3 (−9.8) | +82.6 (−9.8) | +79.6 (−11.4) | +53.0 (−14.9) | incumbent | base age | incumbent |
| 2008 | −29.9 (−32.6) | −28.8 (−31.8) | −28.8 (−31.8) | −28.8 (−31.8) | −29.8 (−32.4) | −52.1 (−59.7) | base age | base age | base age |
| 2009 | +62.8 (−31.7) | +66.0 (−30.9) | +66.0 (−30.9) | +66.0 (−30.9) | +62.8 (−31.6) | +75.6 (−59.1) | base age | base age | base age |
| 2010 | +15.5 (−15.7) | +26.8 (−14.5) | +26.8 (−14.5) | +26.8 (−14.5) | +16.5 (−15.7) | +18.6 (−25.0) | tv20 | tv20 | tv20 |
| 2011 | −10.5 (−18.6) | −11.7 (−18.7) | −11.7 (−18.7) | −11.7 (−18.7) | −10.6 (−18.5) | −24.1 (−27.3) | incumbent | null | incumbent |
| 2012 | +28.2 (−19.5) | +29.1 (−19.6) | +33.1 (−19.6) | +33.1 (−19.6) | +28.2 (−19.4) | +26.5 (−26.0) | tv20 | null | tv20 |
| 2013 | +4.4 (−9.3) | +4.4 (−9.3) | +4.7 (−9.3) | +4.4 (−9.6) | +4.5 (−9.3) | +7.2 (−16.0) | tv20 | **rs252** | tv20 |
| 2014 | +49.7 (−7.3) | +59.1 (−7.7) | +58.7 (−8.0) | +58.7 (−8.0) | +50.2 (−7.8) | +31.6 (−6.2) | **rs252** | incumbent | **rs252** |
| 2015 | −3.0 (−22.9) | −3.0 (−22.7) | −5.5 (−23.4) | −5.4 (−23.1) | −3.3 (−22.7) | −4.3 (−15.0) | incumbent | **rs252** | **rs252** |
| 2016 | +7.2 (−28.9) | +7.0 (−28.5) | +7.2 (−29.1) | +7.2 (−29.1) | +6.7 (−28.7) | +4.0 (−21.6) | incumbent | **rs252** | **rs252** |
| 2017 | +61.1 (−12.6) | +64.7 (−12.4) | +61.5 (−13.0) | +64.2 (−13.0) | +62.2 (−12.3) | +29.9 (−8.5) | **rs252** | null | **rs252** |
| 2018 | −27.2 (−33.7) | −24.5 (−30.3) | −27.0 (−33.2) | −32.1 (−37.0) | −28.6 (−33.9) | +4.8 (−14.1) | **rs252** | **rs252** | **rs252** |
| 2019 | +30.1 (−34.0) | +27.2 (−30.6) | +24.8 (−33.5) | +26.2 (−37.2) | +30.0 (−34.3) | +13.6 (−10.5) | incumbent | **rs252** | **rs252** |
| 2020 | +48.5 (−21.1) | +55.8 (−18.1) | +56.0 (−22.9) | +56.0 (−26.5) | +52.3 (−22.3) | +15.4 (−36.3) | base age | **rs252** | **rs252** |
| 2021 | +83.7 (−10.9) | +76.5 (−12.4) | +70.6 (−13.5) | +71.4 (−12.5) | +76.3 (−12.0) | +26.0 (−9.5) | incumbent | incumbent | incumbent |
| 2022 | −4.8 (−27.8) | −2.2 (−24.2) | +6.2 (−23.6) | +8.3 (−23.6) | −3.1 (−24.8) | +5.5 (−16.1) | base age | tv20 | base age |
| 2023 | +51.9 (−19.2) | +50.8 (−18.7) | +44.4 (−14.9) | +50.5 (−14.6) | +51.7 (−19.2) | +21.0 (−9.7) | incumbent | base age | base age |
| 2024 | +4.6 (−24.9) | +18.9 (−22.5) | +19.0 (−20.8) | +22.3 (−22.3) | +8.0 (−21.4) | +10.4 (−10.5) | base age | tv20 | base age |
| 2025 | +2.1 (−17.1) | −1.2 (−19.5) | −1.5 (−18.4) | −0.2 (−19.9) | +3.9 (−19.3) | +11.7 (−15.2) | null | incumbent | incumbent |
| 2026 | +32.7 (−20.3) | +41.8 (−21.6) | +39.1 (−21.4) | +34.9 (−21.4) | +29.4 (−20.9) | −9.4 (−14.8) | **rs252** | incumbent | **rs252** |
| **CAGR / MaxDD** | 20.95 / −34.0 | **22.58 / −31.8** | 22.77 / −33.5 | 22.50 / −37.2 | 20.80 / −34.4 | 12.30 / −59.7 | | | |

**rs252 takes BEST OVERALL in ten of the twenty-two years and LEAST DD in eight** — more
than any other column, and concentrated in the bad years (2018, 2019, 2020), which is the
signature of a drawdown mechanic rather than a return mechanic.

---

## Guarding the seven deadly sins

| Sin | Part A | Part B |
|---|---|---|
| **Look-ahead** | inherited from research/160's engine: every decision on a bar's close, every fill at the next open; the `high`/`low` arrays are never loaded. The new sale-reason split reads the same rank that already decided the sale — it records, it does not decide | inherited from research/166: rotation scores read at `close[i−1]`, both legs fill at `open[i]`, a position bought today cannot be swapped out today |
| **Survivorship** | `market_data.db` keeps only 102 stopped series in 2,158 (4.7%). Pressure is upward on every arm including the benchmarks; the random-ranking null carries the identical bias and is the control for the ranking claim | research/161's universe unchanged: every NSE daily series with ≥ 90 bars, dead names included |
| **Overfitting / multiple testing** | **54 selection cells**, the metric and the adoption bar written into the STATUS doc before the first cell ran and applied unchanged — including to the cells that only beat the incumbent in the fit window | **5 selection cells.** The rule and the three entrant candidates were named in the STATUS doc before anything ran, and the primary evidence is a seed set the cell had never seen. This is a confirmation run, and it is labelled as one in every table |
| **Cost neglect** | 25 / 40 / 60 bps on every finalist; the incumbent dominates at all three | 25 / 40 / 60 bps; the advantage is +0.099 / +0.095 / +0.086 Calmar. Tax through the FY-netting engine, reported in rupees per cell |
| **Regime dependence** | two pre-registered windows on every finalist plus the 4pp rule; the two cells that beat the incumbent in the fit window both fail the holdout | two pre-registered windows per seed; the winning cell takes W1 on 30/30 and W2 on 30/30 fresh seeds |
| **Correlation / single factor** | not re-tested — no entry, exit or universe changed. Quality Summit's 0.62–0.73 monthly correlation to Open Alpha and its dilutive blend verdict (research/160, research/162) stand | not re-tested — same reason. Rotation does tilt the book toward younger positions, an unmeasured change to the blend, and one more reason not to adopt before a live soak |
| **Capacity / shortability** | long-only NSE cash; median position 0.71% of the held name's own tv20 on a ₹1 cr book | long-only; median position 0.43% → 0.50% of tv20 on a ₹10 L book, so 4.3% → 5.0% on ₹1 crore |

## Honest caveats

1. **Part A's window is 8.1 years and cannot be extended.** Screener serves ~12 fiscal
   years; four filed years do not exist for most names until FY2018 is filed in August 2018.
   The window contains the 2023-25 smallcap boom and one real bear leg.
2. **Part B is a confirmation, not a discovery.** The cell being confirmed was found
   post-hoc in research/166. A fresh seed set removes the *seed-selection* component of that
   worry; it cannot remove the fact that this rule was chosen from a family of twenty that
   was itself chosen from three studies of the same book. The 60/60 consistency and the
   random-swap null are the strongest defences available, and they are not proof.
3. **The adoption bar is not resolving the question, and that is uncomfortable.** +0.094,
   +0.096, +0.105 against a bar of +0.100 across three independent evaluations. Anyone
   reading this as "it works" or "it does not work" is reading more than the data supports.
   The correct summary is: **the effect is real and small, and the pre-registered threshold
   happens to sit on top of it.**
4. **`market_data.db` is not retroactively split-adjusted.** Both engines carry their
   inherited defences (research/160's ATH `cummax` restart on a one-day collapse below 0.55×;
   research/161's truncation after a −35% day). Both make the near-ATH state slightly *easier*
   to satisfy for affected names. Direction stated, not hidden.
5. **Part A's tax model is an approximation of the statute** — one netted FY pool with
   per-trade rates and loss carry-forward, not the full STCL/LTCL set-off ordering.
6. **Nothing in either part was soaked on live data.** Part B's event list is research/164's
   frozen 3,619 events; Part A runs on a frozen panel snapshot.
7. **Quality Summit is not a deployed book**, so Part A's conclusion changes no live risk
   either way. Base Age is converting to live under research/165 **with the un-rotated spec**,
   and this study does not alter that.

## Cells disclosed

| | Part A | Part B |
|---|---|---|
| harness proof (not selection) | 2 (5.0% and 5.2% idle cash) | 2 seed sets × the incumbent |
| **selection cells** | **54** — 42 leeway × book size × cadence, 12 band × leeway | **5** — 3 entrant priorities, 2 margin neighbours |
| controls / nulls (not selection) | 6 — 4 random-ranking, 2 `retain = loose` | 2 — incumbent, rate-matched random swap |
| validation re-scorings (not selection) | 42 — W1/W2, 40/60 bps, 0% cash, missing = pass, on 6 finalists | 8 — 40 and 60 bps on the shortlist |
| paired re-runs in-process | 33 pairs × 3 windows | per-seed deltas computed from the stored seed statistics |
| total simulations | ≈ 3,700 (cells × offsets × 3 cost/tax arms) | ≈ 660 (cells × 30 seeds) |

## Files

| file | what |
|---|---|
| `scripts/patch_engine170.py` → `scripts/qg_engine170.py` | Part A's engine, generated from research/160's frozen copy by 11 exact-string patches |
| `scripts/make_grid170.py` | Part A grid JSONs |
| `scripts/paired170a.py` | Part A paired tests + equity dumps |
| `scripts/sim170.py`, `scripts/bt_core.py`, `scripts/run170b.py` | Part B: research/166's harness, copied unchanged, plus this study's runner |
| `scripts/report170.py` | Part B paired tests, NAV curves, YoY tables |
| `results/cellsA_main.csv`, `cellsA_band.csv`, `cellsA_val.csv`, `cellsA_proof.csv` | one row per Part A cell |
| `results/cellsB_*.csv`, `results/seedstatsB_*.csv` | Part B per-cell and per-seed statistics |
| `results/pairedA.md`, `results/pairedB.md` | the pre-registered paired tests |
| `results/yoyA.{md,html}`, `results/yoyB.{md,html}` | the house-format YoY tables |
| `results/r170_partA.png`, `results/r170_partB.png` | log growth curves with drawdown panels |

---

*Written 13-Sep-2026. Part A reproduces research/160's `F_Bb7` row to the second decimal
before any selection cell; Part B reproduces every research/166 row on research/166's own
seeds. Data snapshot: `backtest_data/market_data.db` and `backtest_data/fundamentals.db` as
of 12-Sep-2026.*
