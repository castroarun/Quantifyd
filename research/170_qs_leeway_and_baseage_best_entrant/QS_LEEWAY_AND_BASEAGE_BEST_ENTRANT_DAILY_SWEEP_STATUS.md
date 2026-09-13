# research/170 — Quality Summit rank leeway · and Open Alpha · Base Age's best-qualifying entrant

**STATUS: DONE (sections 1-4 written 13-Sep-2026 08:00 IST before any cell ran; both parts
complete 13-Sep-2026 08:25 IST)**

**Part A verdict: CONCLUDED — NO ADOPTION.** The book already keeps a holding to rank 23,
not 15; only 14.5% of its sales are rank sales at all; every leeway width tested loses to
the incumbent on paired evidence, and the wider leeway buys churn reduction with no
meaningful tax relief (long-term trades rise only 0.3% → 1.6%).

**Part B verdict: SIGNAL — the post-hoc pick REPLICATED and still sits on the bar.** The
highest-RS entrant wins 60 of 60 paths on both CAGR and Calmar against the incumbent and
60 of 60 against a rate-matched random swap, in both windows, at every cost rung — and
delivers +0.096 pooled paired Calmar against a pre-registered +0.100. **Not adopted.**

Full write-up: `results/RESULTS.md`.

Two separate questions, on two different books, asked by Arun on the morning of
13-Sep-2026. They share nothing but this folder. Read them separately.

| Part | Book | Question | Engine | Status |
|---|---|---|---|---|
| **A** | **QS** — Quality Summit (research/160 Family-B `b7`) | Should a holding that slips in the rank be given a grace band before it is sold, and how wide? Does the near-all-time-high band `k` change with it? | research/160 `qg_engine.py`, copied and patched | PLANNED |
| **B** | **OA · Base Age** — Open Alpha on aged deep bases (research/161, live conversion under research/165) | When the book is full and a qualifying signal arrives, the freed slot goes to *which* entrant? research/166 found "the highest-RS one" **after** seeing results. Does it survive on fresh seeds? | research/166 `sim166.py`, copied unchanged | PLANNED |

**Nothing in this study is deployed. Both parts are research only.**

---

# PART A — QS · the rank leeway ("rebalance slot grace") and the band

## A1. The Ask

**What Arun asked (13-Sep-2026):**

> "a stock falling to 16th rank being taken out now — how about giving it a leeway, say
> within top 25 or any optimized numbers, then the stock remains on the rebalance."

**The fact that has to be stated before anything is tested — the premise is not what the
book does.** The Quality Summit spec research/160 published *already carries a rank leeway*,
and it is wider than the one Arun is proposing. The engine keeps a holding while its
relative-strength rank sits inside `ceil(buffer × N)` of today's qualifying names. At the
published `buffer = 1.5` and `N = 15` that is **ceil(22.5) = 23**:

- a holding ranked **16th is kept**, and so is one ranked 23rd;
- it is only sold on rank at **24th or worse**;
- **but** it is also sold — at any rank, even 1st — if it stops *qualifying* at all: if it
  falls below `k = 0.90 ×` its own causal all-time-high close, drops under the ₹2 cr
  20-day-median traded-value floor, or stops passing the point-in-time `b7` screen.

So there are two ways out of this book, and only one of them is a rank. **No rank leeway of
any width can save a name that has left the near-all-time-high band.** Whether Arun's
proposal can do anything at all therefore depends on a number nobody has measured: what
share of monthly sales are rank sales rather than state sales. This study measures it first.

**What is actually being tested:**

1. What fraction of QS's monthly sales are caused by the **rank** (rescuable by a leeway)
   versus by the **state** (not rescuable)?
2. Across rank-leeway widths from "no leeway at all" (keep only the top N) to "keep anything
   in the top 45", at three book sizes and two rebalance cadences — is there a width that
   beats the deployed `buffer = 1.5` on the pre-registered bar?
3. What does a wider leeway buy in **churn and tax**, which is the mechanism Arun is
   reaching for: fewer sales → longer holds → more of the book's gains taxed at the 12.5%
   long-term rate instead of 20% short-term. Reported **after tax and pre-tax**, because the
   leeway is supposed to show up in the gap between them.
4. At the best leeway, does the near-all-time-high band `k` want to move off 0.90?

**Only `buffer = 1.0` and `buffer = 1.5` have ever been run** on this book — 4 cells and 580
cells respectively across research/160's `cells_g1/g2/g3`. research/162 swept `k × N` but
never touched the buffer. The axis is genuinely unexplored.

## A2. The Base — what is being tested

**The book (frozen; every cell below changes exactly one thing about it):**

| element | spec |
|---|---|
| universe | every NSE daily series in `market_data.db`, funds/ETFs excluded via the panel's `is_fund` flag |
| liquidity | 20-day median traded value ≥ **₹2 cr** |
| state | close ≥ **0.90 ×** the name's own **causal** all-time-high close (`cummax` to yesterday) |
| screen | point-in-time Screener mask **`b7_g10_qual_mc`**: profitable in the last 3 filed fiscal years, 3-year average ROE > 15%, ROCE > 15% or a lender, 3-year sales **and** profit growth > 10%, market cap > ₹1,000 cr, **no debt test**. Missing data = ineligible (`fail`); the `pass` policy is re-run on the finalists |
| ranking | relative strength (`score`), descending |
| book size | **N = 15** equal-weight slots, max 30% in one name |
| cadence | **monthly**, on the first session of the month |
| leeway | keep a holding while ranked inside `ceil(buffer × N)`; **buffer = 1.5 → rank 23** |
| retain policy | `strict` — the near-ATH state is required to keep a name, not only to buy it |
| exits | **none**. research/160 and research/162 both found no exit beats holding on this book |
| index gate | none |
| fills | decision on the close, fill at the **next open** |
| costs | **25 bps a side** (40 and 60 bps on the finalists) |
| tax | 20% STCG / 12.5% LTCG above 365 days, Indian FY loss netting, settled 1 April |
| idle cash | **5.2% post-tax, credited daily** (Arun's standard, 12-Sep-2026). research/160 published at 5.0%; the incumbent is re-run at both and the harness proof is at 5.0% |
| capital | ₹1 crore |
| window | **2018-08-01 → 2026-09-10**. It cannot be extended: Screener serves ~12 fiscal years, so four filed years do not exist for most names until FY2018 is filed in Aug-2018 (coverage steps 7% → 87% at that date) |
| paths | **12 rebalance-day offsets**; medians reported with [min..max] and the worst offset |

**Incumbent to beat (research/160 `F_Bb7`, reproduced by this study's engine before any
selection cell runs):** 21.19% after-tax CAGR, −37.07% max drawdown, Calmar 0.58, 91.2%
invested, 61.4 trades/yr, at 5.0% idle cash.

**Success criterion, pre-registered:** rank cells by **after-tax Calmar** (12-offset median)
among cells whose **after-tax CAGR is ≥ the incumbent's**. A cell that raises Calmar only by
sitting in cash is not a candidate; `avg_pct_invested` is reported in every table.

**Adoption bar, pre-registered (all of it must hold):**

1. **paired** against the incumbent on the same 12 offsets: **≥ +0.15 Calmar** *or*
   **≥ +2pp CAGR at no worse drawdown**;
2. on **≥ 8 of 12** offsets;
3. in **both** windows — fit **W1 2018-08-01 → 2022-06-30**, holdout
   **W2 2022-07-01 → 2026-09-10** — and a cell whose W2 CAGR falls **more than 4pp** below
   its W1 CAGR is declared **not robust** (research/162's rule, unchanged);
4. **plateau, not peak** — both immediate neighbours on the leeway axis must agree in
   direction, not merely be "within a few points";
5. survives **40 bps** a side.

**Null control:** the identical book with **random ranking** at the same N and buffer, 30
seeds. If the leeway's benefit shows up in the random-ranked book too, it is not a ranking
finding.

## A3. Plan — Part A grid

| axis | values | cells |
|---|---|---|
| **A1 leeway** `buffer` | 1.0, 1.33, 1.5, 1.67, 2.0, 2.5, 3.0 → at N=15 these are ranks **15 / 20 / 23 / 26 / 30 / 38 / 45** | 7 |
| × book size `N` | 10, 15, 20 | 3 |
| × cadence | monthly, quarterly | 2 |
| | **A1 subtotal** | **42** |
| **A2 band** `k` | 0.75, 0.80, 0.85, 0.90, 0.95 at the best buffer **and** at buffer 1.5, N = 15, monthly | **8 new** (2 already in A1) |
| | **selection cells, total** | **50** (budget 60) |

Validation cells, **not** selection (disclosed separately): harness proof at 5.0%; W1 and W2
for the incumbent and the finalists; 40 and 60 bps on the finalists; the random-ranking null
at two leeways; the `retain = loose` probe (a *second* kind of leeway — keep a name whose
rank is fine even though it has left the near-ATH band — which answers directly whether the
rank is the binding constraint); the `mask_missing = pass` policy on the finalists.

**Extra instrumentation added to this study's copy of the engine** (patch script, exact
string replacements, auditable): the monthly sale reason is split into **`reb_rank`** (the
name still qualifies but ranked outside the leeway) and **`reb_state`** (it no longer
qualifies at all); rupee **tax paid** is accumulated through the FY-netting block; **average
holding period** and the **share of closed trades held beyond 365 days** are recorded. These
are the columns that make Arun's question answerable rather than merely rankable.

---

# PART B — OA · Base Age: swap for the BEST qualifying signal, confirmed honestly

## B1. The Ask

**What Arun asked (13-Sep-2026):**

> "our case: if not any qualifying signal — the best candidate / highest-ranked one."

**What is being tested.** research/166 tested whether a full Open Alpha · Base Age book
should sell a holding to make room for a qualifying breakout it cannot otherwise take. It
found exactly one ranking that works: **sell the holding that is more than 10% under water,
and give its slot to the newcomer** — 22.68% after-tax CAGR, −33.52% drawdown, Calmar 0.673,
against the incumbent's 20.98% / −34.05% / 0.613, on 27 of 30 paired seeds, at ~4.2 swaps a
year. That cell missed the pre-registered adoption bar (+0.049 Calmar against a +0.10 bar)
and **nothing was adopted**.

A **post-hoc** cell did better. Changing *which* refused entrant gets the freed slot from
the most liquid (`tv20_cr`) to the highest 12-month relative strength (`rs252`) lifted the
same rule to **22.73% / −31.78% / Calmar 0.715**, winning **30 of 30** paired seeds on both
CAGR and Calmar, W1 30/30, W2 29/30, with the cost ladder holding at 0.715 / 0.682 / 0.651.
That **clears** the +0.10 Calmar bar. But it was one of three entrant priorities tried
**after** the rotation result was known — a 1-of-3 pick sitting on a 1-of-20 pick — so
research/166 refused to bank it and said so in every table.

**This part exists to settle that one question honestly**: does the highest-RS entrant hold
up when the choice is written down *first* and the book is run on **seeds it has never
seen**?

## B2. The Base

Research/166's harness, unchanged — `sim166.py`, itself research/164's `sim164`, itself
research/161's `bt_core` — copied into this study's `scripts/`. Same frozen event list
(`events166.csv`, 3,619 events), same SuperTrend(14,4) line cache (`st166.pkl`), same panel
(`panel164.pkl`).

| element | spec |
|---|---|
| entry | research/161's adopted Base Age spec: new all-time-high close on a base aged ≥ 40 bars and ≥ 20% deep; next-open fill |
| exit | **SuperTrend(14, 4)** close trail. No stop, no target, no time stop |
| book | **16 slots at 6.25% of NAV**, ₹10 lakh, contested slots broken **at random** (the incumbent's convention) |
| costs | 25 bps a side (40 and 60 on the shortlist) |
| tax | 20% STCG / 12.5% LTCG, Indian FY netting, modelled through the engine — never a haircut |
| idle cash | **5.2% post-tax, credited daily** |
| window | 2005-01-03 → 2026-09-11; **W1 2005-01-03 → 2015-12-31**, **W2 2016-01-01 → 2026-12-31** |
| paths | **30 seeds.** research/166 used seeds **1…30**. This study's primary evidence is **fresh seeds 1001…1030**, which no cell in research/164 or research/166 has ever been run on; every cell is then re-run on 1…30 for comparability |

**The rule, written down before any cell runs:**

> On a day with at least one qualifying Base Age signal the book cannot take (no free slot
> **or** not enough cash), rank the holdings by unrealised return since entry. If the
> weakest is **more than 10% under water**, sell it at the next open and buy, at the same
> open, the refused entrant ranked highest by **X**. At most one swap a day. A position
> bought today cannot be swapped out today.

**X ∈ {`rs252`, `tv20_cr`, `x_bars`}** — 12-month relative strength, 20-day median traded
value in ₹ crore, base age in bars. **Three cells. That is the whole selection question.**

**Adoption bar, pre-registered (research/166's, unchanged):** paired against the incumbent,
**≥ +0.10 Calmar** *or* **≥ +2pp CAGR at no worse drawdown**, on **≥ 20 of 30** seeds, in
**both** windows, surviving **40 bps**, with tax run through the FY engine. Plus, because
this is a confirmation of a post-hoc pick: it must also **beat the rate-matched random-swap
null**, and the **margin plateau must hold** ({7.5%, 10%, 12.5%} under water on the winning
entrant).

## B3. Plan — Part B grid

| cell | what it is | selection? |
|---|---|---|
| `BASE_rand` | the incumbent: never swap | no — baseline |
| `A_unre_m010` | the rule with the **`tv20_cr`** entrant = research/166's pre-registered winner | **yes (1 of 3)** |
| `X_entrs_unre_m010` | the rule with the **`rs252`** entrant = the post-hoc pick under test | **yes (2 of 3)** |
| `X_entage_unre_m010` | the rule with the **`x_bars`** entrant | **yes (3 of 3)** |
| `A_null_p003` | rate-matched random swap, p = 0.03 ≈ 4 swaps/yr | no — null |
| `X_ent<W>_unre_m0075` | margin neighbour, 7.5% under water, winning entrant | **yes (plateau)** |
| `X_ent<W>_unre_m0125` | margin neighbour, 12.5% under water, winning entrant | **yes (plateau)** |

**5 selection cells** (budget 20). Each run on **both** seed sets; the 40 bps ladder on the
shortlist. W1 / W2 statistics are computed inside every run at no extra cost.

**What will be reported:** paired ΔCAGR and ΔCalmar against the incumbent **and** against
the null, seeds won, swaps/yr, rupee tax paid, the share of total profit coming from the ten
best realisations, capacity (median position as a share of the held name's own 20-day traded
value), and an explicit multiple-testing note — this is a confirmation of a post-hoc pick,
and the write-up must say whether it survived on seeds it had never seen.

---

## 5. Status — live log

**Phase: COMPLETE.** Both parts run, reported, published.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-13 08:00 | Folder created, STATUS sections 1-4 written | before any cell |
| 2026-09-13 08:04 | Part A engine generated from research/160's frozen copy | 11 exact-string patches, all matched once |
| 2026-09-13 08:06 | **Part A harness proof PASSES** | 21.19 / −37.07 / 0.58 at 5.0% idle cash = research/160's published `F_Bb7` to the second decimal. At 5.2%: 21.39 / −36.90 / 0.58 — the baseline for this study |
| 2026-09-13 08:08 | Part A axis A1 done, 42 cells | leeway curve flat; **only 14.5% of sales are rank sales at the incumbent setting** |
| 2026-09-13 08:09 | **Part B fresh seeds (1001-1030) done, 5 cells** | rs252 entrant 22.58 / −31.78 / **0.710** vs incumbent 20.95 / −34.05 / 0.611 |
| 2026-09-13 08:11 | **Part B harness proof PASSES** on research/166's seeds 1-30 | every published row reproduced: 20.98 / 22.68 / 22.73 / 20.39 |
| 2026-09-13 08:13 | Part B plateau + 40/60 bps ladders done | 0.683 / 0.710 / 0.645 on the margin; 0.710 / 0.684 / 0.647 on cost |
| 2026-09-13 08:15 | Part A axis A2 done, 15 cells | k = 0.90 is the Calmar peak at every leeway; k = 0.85 wins the fit window and reverses in the holdout (research/162 replicated) |
| 2026-09-13 08:17 | Part B paired analysis | **+0.105 Calmar on 30/30 fresh seeds, +0.094 on research/166's, +0.096 on all 60 pooled** |
| 2026-09-13 08:19 | Part A validation grid done, 48 cells | `retain = loose` −1.42pp CAGR / −0.146 Calmar on 2/12; random-ranking null 11.73% vs 21.39% |
| 2026-09-13 08:22 | Part A paired tests done | every leeway width loses; Arun's "top 25" a dead wash at −0.09pp / −0.003 on 5/12 |
| 2026-09-13 08:25 | YoY tables + figures written; RESULTS.md written | both parts CONCLUDED |

## 6. Crash Recovery — how to resume without Claude

Everything runs on the VPS at `/home/arun/quantifyd/research/170_qs_leeway_and_baseage_best_entrant/`.
Python is `/home/arun/quantifyd/venv/bin/python3`. Nothing here writes to `market_data.db`,
`fundamentals.db`, any live state file, or any `services/` module.

**To see how far Part A got:**

```bash
cd /home/arun/quantifyd/research/170_qs_leeway_and_baseage_best_entrant
wc -l results/cellsA_*.csv          # one row per finished cell, header included
tail -5 results/a_main.log
```

**To resume Part A** (the runner skips every label already present in the CSV):

```bash
cd /home/arun/quantifyd
nice -n 10 venv/bin/python3 research/170_qs_leeway_and_baseage_best_entrant/scripts/qg_engine170.py \
  --panel research/160_quality_growth_near_ath/results/panel_2000.npz \
  --grid  research/170_qs_leeway_and_baseage_best_entrant/results/gridA_main.json \
  --out   research/170_qs_leeway_and_baseage_best_entrant/results/cellsA_main.csv
```

**To see how far Part B got, and resume it:**

```bash
cd /home/arun/quantifyd/research/170_qs_leeway_and_baseage_best_entrant
wc -l results/cellsB_*.csv; tail -5 results/b_fresh.log
cd /home/arun/quantifyd
nice -n 10 venv/bin/python3 research/170_qs_leeway_and_baseage_best_entrant/scripts/run170b.py \
  --stage=fresh --seeds=30 --seedbase=1000 --workers=2
```

**Rebuild the patched Part-A engine from research/160's frozen copy** (idempotent; it
refuses to run if any patch does not match exactly):

```bash
cd /home/arun/quantifyd
venv/bin/python3 research/170_qs_leeway_and_baseage_best_entrant/scripts/patch_engine170.py
```

**Do not touch:** anything under `research/160_…/`, `research/164_…/`,
`research/166_…/` — this study reads those and writes none of them.
**Safe to inspect or delete and regenerate:** everything under
`research/170_qs_leeway_and_baseage_best_entrant/results/`.

## 7. Files

| file | what | committable |
|---|---|---|
| `QS_LEEWAY_AND_BASEAGE_BEST_ENTRANT_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/patch_engine170.py` | generates the Part-A engine from research/160's frozen copy | yes |
| `scripts/qg_engine170.py` | the generated Part-A engine | yes |
| `scripts/make_grid170.py` | builds the Part-A grid JSONs | yes |
| `scripts/run170b.py` | Part-B runner (fresh + research/166 seed sets) | yes |
| `scripts/sim170.py`, `scripts/bt_core.py` | research/166's harness, copied unchanged | yes |
| `scripts/report170.py` | paired tests, YoY tables, figures | yes |
| `results/cellsA_*.csv` | one row per Part-A cell | yes |
| `results/cellsB_*.csv`, `results/seedstatsB_*.csv` | Part-B per-cell and per-seed rows | yes |
| `results/*.log` | run logs | yes (small) |
| `results/navs_*/`, `results/*_equity.csv` | per-path NAV curves | NO — gitignored |
| `results/RESULTS.md` | final findings, both parts | yes |

## 8. Findings

Full write-up with every table: **`results/RESULTS.md`**. Headlines:

### Part A — QS rank leeway: **CONCLUDED, NO ADOPTION**

1. **The premise was wrong.** The deployed spec keeps a holding while ranked **23rd or
   better** (`ceil(1.5 × 15)`), not 15th. A name at rank 16 is not sold today.
2. **The rank is not what ends most holdings.** Instrumenting the engine to split the
   monthly sale reason shows **85.5% of sales are state sales** (the name left the near-ATH
   band, the liquidity floor or the screen) and only **14.5% are rank sales**. A rank leeway
   can only ever touch one sale in seven.
3. **Every leeway width loses, paired.** Rank 15 / 20 / 26 / 30 / 38 / 45 against the
   incumbent's 23: ΔCAGR −2.02 / −0.15 / −0.09 / −0.39 / −0.57 / −0.84 pp, winning 3, 4, 5,
   4, 3, 2 of 12 offsets. Arun's own "top 25" (rank 26) is the closest to a tie — a wash.
4. **Widening does cut churn but cannot cut tax.** Trades 74 → 54 a year, holding period
   66 → 93 days, but the share of trades held beyond 365 days rises only **0.3% → 1.6%**.
   The book's natural holding period is nowhere near the long-term line, and the routes that
   do move it (quarterly cadence, keeping fallen names) cost more than the rate saves.
5. **The other leeway is worse.** `retain = loose` — keep a name whose rank is fine after it
   has left the near-ATH band — costs **−1.42pp CAGR and −0.146 Calmar on 2 of 12 offsets**
   and adds 5.5 points of drawdown. The near-ATH requirement is protective.
6. **k = 0.90 stands.** It is the Calmar peak at every leeway. `k = 0.85` clears the bar in
   the fit window (+2.32pp, +0.156 on 9/12) and reverses in the holdout (−1.09pp, 4/12) —
   the second independent replication of research/162's reversal.
7. **Null:** random ranking at the same leeway returns 11.73% against 21.39%. RS ranking is
   worth **+9.7pp**, and the leeway helps the random book too — so its small benefit is
   churn reduction, not selection.

### Part B — Base Age best-qualifying entrant: **SIGNAL, sits on the bar, NOT ADOPTED**

1. **The post-hoc pick replicated.** On 30 seeds it had never seen, "sell the holding more
   than 10% under water, buy the refused entrant with the highest 252-day relative
   strength" returns **22.58% / −31.78% / Calmar 0.710** against the incumbent's
   **20.95% / −34.05% / 0.611** — paired **+0.105 Calmar on 30/30** and **+1.64pp CAGR on
   30/30**, and **+0.115 Calmar on 30/30 against a rate-matched random swap**.
2. **Pooled across both seed sets (60 paths): +0.096 Calmar on 60/60 and +1.65pp on 60/60.**
   The pre-registered bar is +0.10. Three independent evaluations have now landed at +0.094,
   +0.096 and +0.105. **The effect is real and small, and the threshold sits on top of it.**
3. **Who leaves sets the return; who enters sets the drawdown.** All three entrant
   priorities earn the same +1.6 to +1.8pp of CAGR. The whole spread — 5.5 points of
   drawdown, 0.105 of Calmar — is in the entrant: rs252 −31.78%, tv20 −33.52%, base age
   −37.23%. The oldest-base entrant actually *loses* to the incumbent on Calmar.
4. **Survives costs with the gap widening in relative terms**: Calmar 0.710 / 0.684 / 0.647
   at 25 / 40 / 60 bps against the incumbent's 0.611 / 0.589 / 0.561. Tax is modelled, not
   haircut: ₹105.4 L vs ₹84.1 L on a ₹10 L book over 21.7 years, +25%.
5. **Plateau is a hump, and only the 10% margin wins both windows** (7.5% earns in W1 and
   loses W2; 12.5% does the reverse).
6. **Recommendation: the live Base Age book converting under research/165 goes live
   unchanged.** The rule misses the pre-registered bar, the book has never traded live, and
   the mechanic needs an evening scorer and a two-leg next-open order that does not exist.
   The exact rule text is written out in RESULTS.md as **proposal OA-ROT-1**, with a dated
   review on **2027-03-13** and a pre-registered pass criterion on the live entry queue.
