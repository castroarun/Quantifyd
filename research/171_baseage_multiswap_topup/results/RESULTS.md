# research/171 — OA · Base Age: swap more than one a night? Top up existing winners instead of buying new entrants?

**Nothing here is deployed. The live Base Age book keeps OA-ROT-1 exactly as staged.**

| Axis | What Arun asked | Verdict |
|---|---|---|
| **A — how many leave** | "swap the last 2 ranks instead of 1… an optimized number?" | **NO EDGE. And the optimum is 1.** Two holdings are simultaneously eligible on **0.5 evenings a year** once the rule is running; `k = 3, 4, 6` and "all eligible" are **bit-identical to each other on all 30 seeds** — a third swap in one evening happens on one path in thirty and never again. `k = 2` is a wash against the staged rule and slightly worse (−0.003 Calmar, winning 11 of 30 seeds), and a real loss at 40 bps |
| **B — top up existing winners instead of new entrants** | "instead of new entrants, top up the highest running ones existing within the portfolio" | **NO EDGE. All 18 cells lose risk-adjusted to doing nothing, and none beats the staged rule on a single seed of thirty.** Best of them: Calmar **0.587** against the incumbent's 0.606 and OA-ROT-1's 0.702. The drawdown goes to −36% to −42% and one position reaches **52% of NAV** |
| **C — hybrid** | (the study's own extension) | **SIGNAL — NOT ADOPTED, and it is not Arun's idea.** One construction clears the pre-registered bar (+0.128 Calmar on 30/30), but its top-up leg contributes **nothing** on the evenings Arun described; **100% of its edge comes from selling on evenings when no signal fired** — i.e. from adding an unconditional −10% stop, which research/166 already identified as the cheaper lever and which reaches **0.683–0.692** here with **none** of the machinery |

**Window 2005-01-03 → 2026-09-11 (21.7 years). 51 selection cells, 12 controls, 29 cost
re-scorings, ≈ 2,820 simulations. After tax (20% STCG / 12.5% LTCG, Indian FY loss-netting),
net of 25 bps a side, idle cash 5.2% post-tax credited daily, 16 slots @ 6.25%, ₹10 lakh book,
30 seeds (7001–7030) that no cell in research/164, 166 or 170 has ever touched. All drawdowns
measured from the running peak of the full curve.**

---

## Harness proof, run before any selection cell

`sim171.py` is generated from research/170's `sim170.py` by **12 exact-string patches**, each of
which must match exactly once or the build aborts. research/170's file is never edited.

- **The defaults are a no-op, bit for bit.** research/170's incumbent and OA-ROT-1 cells run
  through both modules on the same seed return an identical NAV array, an identical trade list,
  an identical invested-percentage series and an identical book dictionary (checked on seeds
  1001 and 1017; 690 and 749 trades, 95 and 97 swaps).
- **The published rows reproduce.** On research/170's own fresh seed base (1000), this harness
  returns the incumbent at **20.945 / −34.045 / 0.611** and OA-ROT-1 at **22.58 / −31.78 /
  0.710** — research/170 Part B's published figures to the digit.
- **A second, independent no-op check.** `CTRL_measure_k0` — the full rotation rule with
  `rot_max_per_day = 0`, so the eligibility counter records but nothing can fire — returns
  **20.91 / −35.16 / 0.606**, identical to the incumbent on all 30 new seeds.

---

## The number that decides Axis A, reported before any CAGR

On an evening when at least one qualifying Base Age signal was **refused**, how many open
positions were simultaneously more than 10% below their average buy price? This is a recording
inside the engine: it decides nothing and draws no random number.

| book | refused-signal evenings | ≥ 1 eligible | ≥ 2 eligible | ≥ 3 eligible | most ever | per year with ≥ 2 |
|---|---:|---:|---:|---:|---:|---:|
| the INCUMBENT's own path (never swaps) | 1,396 | 418 (30.0%) | **92 (6.6%)** | 16 (1.1%) | 4 | **4.2** |
| **OA-ROT-1 running (k = 1)** | 1,412 | 97 (6.9%) | **11 (0.8%)** | 2 (0.1%) | 3 | **0.5** |
| k = 2 | 1,408 | 95 (6.7%) | 9 (0.6%) | 2 (0.1%) | 4 | 0.4 |
| k = all eligible | 1,407 | 94 (6.7%) | 9 (0.6%) | 2 (0.1%) | 4 | 0.4 |

**Read the second row.** Once the single-swap rule is actually running it keeps removing the
under-water names, so a *second* one almost never accumulates: **eleven occasions in twenty-one
and a half years — one every two years.** On a book that never swaps there would be 4.2 such
evenings a year; the rule itself destroys its own second opportunity. Three or more eligible at
once happens **twice in the whole history**.

That single table is the answer to "can we find an optimized number?" — and it is why the
`k` column below is flat.

---

## Q&A — Arun's three questions

### **Q1. Should we swap the last TWO ranks instead of one?**

**No. It makes no measurable difference, and what difference it makes is slightly negative.**

| cell | CAGR | worst seed | MaxDD | Calmar | ΔCalmar vs INCUMBENT | ΔCalmar vs OA-ROT-1 | swaps/yr | Calmar at 40 bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **INCUMBENT — never swap** | 20.91% | 20.16% | −35.16% | 0.606 | — | −0.090 on 0/30 | 0.0 | 0.600 |
| **OA-ROT-1 — k = 1 (staged live)** | **22.30%** | **21.37%** | **−31.78%** | **0.702** | **+0.090 on 30/30** | — | 4.5 | **0.684** |
| k = 2 | 22.09% | 20.99% | −31.78% | 0.695 | +0.087 on 30/30 | **−0.003 on 11/30** | 4.5 | 0.671 |
| k = 3 | 22.09% | 20.99% | −31.78% | 0.695 | +0.087 on 30/30 | −0.003 on 11/30 | 4.5 | — |
| k = 4 (= all eligible, capped at 25% of the book) | 22.09% | 20.99% | −31.78% | 0.695 | +0.087 on 30/30 | −0.003 on 11/30 | 4.5 | — |
| k = 6 | 22.09% | 20.99% | −31.78% | 0.695 | +0.087 on 30/30 | −0.003 on 11/30 | 4.5 | — |
| k = all eligible | 22.09% | 20.99% | −31.78% | 0.695 | +0.087 on 30/30 | −0.003 on 11/30 | 4.5 | — |

**`k = 3`, `k = 4`, `k = 6` and "all eligible" are literally the same cell** — every value, on
every one of the thirty seeds, differs by exactly nothing. And `k = 2` differs from them on
**one seed out of thirty**: a third swap in a single evening happens on one path in the whole
ensemble and never again. Arun asked for an *optimum*; the honest answer is that **the axis has
only two distinguishable points, 1 and 2**, and 1 wins.

**What `k = 2` does buy, and what it costs.** Not more trading — a different path. Across the
thirty seeds the *total* swap count against `k = 1` moves by anywhere from **−11 to +7** with a
median of about **+1**: allowing a second sale on the roughly nine evenings in 21.7 years when
two holdings qualify re-routes which names the book owns afterwards, and that re-routing is
worth **−0.21pp of CAGR and −0.003 of Calmar**, winning 11 of 30 paired seeds. A wash at 25 bps.
At **40 bps it becomes a real loss**: 0.671 against OA-ROT-1's 0.684. More churn for no return
is exactly what a cost ladder is for.

**The margin does not rescue it.** At `k = 2`, loosening the trigger to 7.5% gives Calmar 0.692
and tightening it to 12.5% gives 0.625 — the same narrow hump research/166 and research/170
found at `k = 1`, shifted by nothing.

**One `k = 2` variant does beat the staged rule, and it is not a swap.** If the second eligible
loser is sold with **no second entrant to buy** and the money simply sits in cash at 5.2%
(`A_k2_spillcash`), Calmar is **0.707 against OA-ROT-1's 0.702** — +0.007 on 19 of 30 seeds.
That is a wash inside the noise, it collapses at 40 bps (0.683 vs 0.684), and it is a preview
of the real finding in Q3: **on this book the value is in the selling, not in the second buy.**

---

### **Q2. Instead of buying a new entrant, should the money top up the holdings that are running hardest?**

**No — and this is the clearest negative in the study. All eighteen cells lose risk-adjusted to
doing nothing at all, and not one of them beats the staged rule on a single seed out of thirty.**

Eighteen cells: the holding ranked by 12-month relative strength / unrealised return / cushion
above its SuperTrend line, × all-into-the-top-one at `k = 1` or `k = 2`, × a position cap of
2× / 3× / none of the 6.25% target.

| ranked by | cap | CAGR | MaxDD | Calmar | ΔCalmar vs INCUMBENT | ΔCalmar vs OA-ROT-1 | max weight one position reached |
|---|---|---:|---:|---:|---:|---:|---:|
| **rs252** | none | **21.36%** | −35.75% | **0.587** | −0.029 on 10/30 | −0.117 on **0/30** | **52.5%** |
| rs252 | 3× | 21.09% | −41.78% | 0.511 | −0.095 on 0/30 | −0.189 on 0/30 | 32.9% |
| rs252 | 2× | 20.65% | −37.78% | 0.558 | −0.037 on 6/30 | −0.137 on 0/30 | 26.1% |
| unrealised | none | 21.34% | −38.01% | 0.561 | −0.051 on 2/30 | −0.147 on 0/30 | **81.3%** |
| unrealised | 2× | 21.07% | −35.72% | 0.587 | −0.023 on 9/30 | −0.106 on 0/30 | 30.1% |
| cushion | none | 19.42% | −42.01% | 0.463 | −0.154 on 0/30 | −0.253 on 0/30 | 41.1% |
| cushion | 2× | 20.20% | −41.25% | 0.489 | −0.124 on 0/30 | −0.210 on 0/30 | 25.9% |

*(the best cell of each ranking, plus the worst; `k = 2` versions are uniformly worse and are in
`results/paired171.md`)*

**Not one of the eighteen beats OA-ROT-1 on a single seed out of thirty.** The two pre-registered
follow-ups did not save it either: splitting the money across the best **two** holdings gives
0.562, and firing on **any** evening rather than only on refused-signal evenings gives 0.587.

**Why it fails, mechanically.** Three things happen at once and all three are bad:

1. **The slot is spent, not refilled.** A top-up consumes no event and occupies no slot, so the
   book drops to fifteen names and the freed slot sits empty until the next signal. The book
   gives up an independent bet to buy more of one it already owns.
2. **It converts diversification into leverage on one name.** A single position reaches
   **52% of NAV** under the rs252 ranking with no cap and **81%** under the unrealised ranking.
   research/164 warned that concentration inflates the share of profit coming from a handful of
   trades; it does exactly that here — the ten best realisations supply **55.4%** of total book
   profit against the incumbent's **39.3%**.
3. **The drawdown is the price.** −35.8% to −42.0% against the incumbent's −35.2% and
   OA-ROT-1's −31.8%. The extra return, where there is any, is bought with a deeper hole.

**And it re-proves an old result from a new direction.** research/166 found that ranking the
*sell* side by momentum or by cushion is the worst answer, because it keeps the book in the
freshest, most-extended breakouts — "leverage, not selection". Ranking the *buy* side the same
way produces the same failure: **cushion is the worst of the three rankings on both sides.**

---

### **Q3. Is there a mixture that works?**

**One construction clears the pre-registered bar. It should still not be adopted, and the
reason is that it is not doing what the question asked.**

`C_els_k1_cush` — sell the holding more than 10% under water, **one a night**; give the money
to the refused entrant with the highest rs252 if a signal is going begging, and otherwise buy
more of the holding with the **most cushion above its SuperTrend line** (capped at 3× target).

| | CAGR | worst seed | MaxDD | Calmar | ΔCal vs INCUMBENT | ΔCal vs OA-ROT-1 | 25 / 40 / 60 bps |
|---|---:|---:|---:|---:|---:|---:|---|
| INCUMBENT | 20.91% | 20.16% | −35.16% | 0.606 | — | −0.090 on 0/30 | 0.606 / 0.600 / 0.567 |
| OA-ROT-1 (staged) | 22.30% | 21.37% | −31.78% | 0.702 | +0.090 on 30/30 | — | 0.702 / 0.684 / 0.653 |
| `C_els_k1` (top up by rs252) | 23.23% | 22.43% | −31.27% | 0.736 | +0.102 on 30/30 | +0.035 on 21/30 | 0.736 / **0.638** / 0.580 |
| **`C_els_k1_cush` (top up by cushion)** | **24.70%** | **24.03%** | −33.62% | **0.735** | **+0.128 on 30/30** | **+0.030 on 29/30** | 0.735 / **0.702** / 0.643 |
| `C_els_k1_split2` (across the best two) | 23.04% | 22.80% | −31.46% | 0.730 | +0.126 on 30/30 | +0.027 on 30/30 | 0.730 / 0.688 / 0.636 |
| hybrid — 50/50 entrant and top-up | 21.55% | 20.22% | −32.79% | 0.644 | +0.045 on 23/30 | −0.047 on 4/30 | — |

It clears the letter of the pre-registered bar: **+0.128 paired Calmar on 30 of 30 seeds**
(bar: +0.10), **+3.56pp of CAGR on 30 of 30** at a *shallower* drawdown (bar: +2pp at no worse
drawdown), **both windows positive on 30/30**, it survives **40 bps** (+0.102 against the
incumbent), and it beats a rate-matched random swap by a mile (0.735 vs 0.558).

**Four reasons it is still not an adoption, in order of weight.**

**1. Its top-up leg does nothing on the evenings Arun described.** The control that settles it:
run the identical rule with the trigger restricted to **refused-signal evenings only**
(`C_cush_sigonly`) and it returns **22.30% / −31.78% / 0.702** — **OA-ROT-1 to the digit, with
zero top-ups.** With one sale a night, the entrant swap always consumes it, so the top-up never
gets a turn. **Every point of the +0.128 comes from sales made on evenings when no signal fired
at all** — an unconditional −10% stop bolted on to the book, which was never the question.

**2. Most of what the stop earns, a plain stop earns on its own.** Decomposing the 0.735:

| build it up | CAGR | MaxDD | Calmar | machinery needed |
|---|---:|---:|---:|---|
| INCUMBENT | 20.91% | −35.16% | 0.606 | none |
| + unconditional −10% hard stop, money to cash | 20.79% | −30.46% | **0.683** | a price check |
| + the same sale run through the rotation path | 21.08% | −30.47% | **0.692** | a price check |
| + OA-ROT-1's entrant swap on signal evenings | 22.30% | −31.78% | **0.702** | the staged rule |
| + the money redeployed into the best-cushion holding | 24.70% | −33.62% | **0.735** | 8.7 sales + 7.0 top-ups a year, and 40% single-name weight |

**Two-thirds of the risk-adjusted gain over the incumbent is a hard stop that needs no rotation
machinery at all.** research/166 said this in its caveat 4 and excluded the hard stop only
because its CAGR (20.79% here) sits a hair *below* the incumbent's. That is exactly the
pre-registered eligibility clause, and it still applies — but the finding has now reproduced
independently on a third seed set, and it deserves to be looked at in its own right rather than
discovered a fourth time as a by-product.

**3. Ninety-four percent of the return edge lives in the holdout window.** Paired against the
incumbent, `C_els_k1_cush` gains **+0.44pp in W1 (2005–2015)** and **+7.03pp in W2
(2016–2026)**. Both are positive on 30/30, so the letter of the two-window rule passes — but a
sixteen-fold asymmetry between the two halves of the history is a **regime-dependence flag**,
the fifth deadly sin, and the pre-registered 4pp rule only catches the mirror image of it.
By contrast OA-ROT-1, the rule already staged, gains **+1.26pp in W1 and +1.40pp in W2** — the
same edge in both halves, which is what a durable mechanic looks like.

**4. It stops being a sixteen-name book.** A single position reaches **40.8% of NAV** (55.4%
with the cap removed), the ten best realisations supply **57.3%** of total profit against the
incumbent's 39.3%, and the win rate falls from **48.9% to 42.2%**. The tax bill rises from
**₹8.35 lakh to ₹13.32 lakh** per ₹10 lakh of book — **+59%** — and the +3.56pp is what is left
after paying it. On a ₹1 crore book the median position is 5.6% of the held name's own 20-day
traded value, and the topped-up names are several times that.

**The plateau is a hump on one axis and a step on the other, and both are disclosed.**

| margin (cushion variant) | Calmar | vs INCUMBENT |
|---|---:|---:|
| ≥ 7.5% under water | 0.700 | +0.092 on 30/30 |
| **≥ 10%** | **0.735** | **+0.128 on 30/30** |
| ≥ 12.5% | 0.617 | +0.004 on **16/30** — a wash |

| position cap | Calmar | max single weight |
|---|---:|---:|
| 1.5× target | 0.677 | 26.2% |
| 2× | 0.671 | 32.4% |
| **3×** | **0.735** | 40.8% |
| none | 0.720 | 55.4% |

The margin is the same narrow hump this family always has (7.5–12.5%), and the tightening
neighbour degrades to a wash. The cap axis is a **step, not a peak** — anything looser than
3× works and anything tighter does not — which reads less like a tuned parameter and more like
"the effect requires concentration to exist". That is a reason for suspicion, not comfort.

---

## The 2025-01 → 2026-09 window — the regime Arun is about to watch live

Reported because research/165 §8.0 showed the staged rule fires at 5.7–7.9 swaps a year and
**loses** over exactly this stretch. These rows are a slice of the full 21.7-year compounded
curve — **not** the same experiment as research/165's restart from the live book's state, and
the two should not be added together.

| cell | W3 CAGR | paired vs INCUMBENT | seeds won |
|---|---:|---:|---:|
| INCUMBENT | 20.83% | — | — |
| **OA-ROT-1 (staged)** | 20.34% | **+0.10pp** | 15/30 |
| k = 2 | 21.54% | +0.32pp | 15/30 |
| k = all eligible | — | +1.19pp | 16/30 |
| best pure top-up (rs252, no cap) | 17.45% | **−3.09pp** | 9/30 |
| `C_els_k1` (top up by rs252) | 18.02% | **−1.81pp** | 12/30 |
| `C_els_k1_cush` | 28.64% | **+7.76pp** | 26/30 |
| stop to cash (control) | 18.46% | −2.04pp | 10/30 |

**On the whole-curve slice OA-ROT-1 is a coin flip in this window (15 of 30 seeds), not a
loser** — which is a milder reading than research/165's restart-from-live-state experiment gave,
and both belong on the record. Every top-up construction except the cushion variant **loses**
here. The cushion variant's +7.76pp on 26/30 is the same W2-concentrated effect as above, seen
through a shorter lens.

---

## YoY — house format

Full table with intra-year drawdowns and the three best-of columns:
[`results/yoy171.md`](yoy171.md) / `.html`. Summary row:

| | INCUMBENT | OA-ROT-1 k=1 | k=2 | TOP-UP only (best of 18) | stop + TOP-UP by cushion | stop to CASH (control) | NIFTYBEES |
|---|---|---|---|---|---|---|---|
| **CAGR / MaxDD** | 20.91% / −35.2% | **22.30% / −31.8%** | 22.09% / −31.8% | 21.35% / −35.8% | 24.71% / −33.6% | 21.08% / −30.5% | 12.30% / −59.7% |

Two things the year rows say that the summary cannot. **The stop-to-cash control takes LEAST DD
in sixteen of the twenty-two years** — it is insurance, and it is priced like insurance
(+0.09pp of CAGR for −4.7 points of drawdown). And **2025 is one of only three years in which the
INCUMBENT takes BEST OVERALL** (+3.2% against −0.5% to −4.9% for every rotation variant) — the
live book is converting into precisely the stretch in which doing nothing has been the best
answer.

---

## Controls and nulls

| control | what it isolates | CAGR | MaxDD | Calmar |
|---|---|---:|---:|---:|
| measure-only (`rot_max = 0`) | the engine's own no-op | 20.91% | −35.16% | 0.606 — identical to the incumbent |
| sell-only, k = 1 / 2 / 3 | sell the loser, do not redeploy | 21.80 / 21.46 / 21.38% | −36.1% | **0.603 / 0.603 / 0.601** |
| random swap at the matched rate, k = 2 / 3 | is the *choice* of holding doing anything? | 20.06 / 20.42% | −34.6% | 0.571 / 0.571 |
| random swap at the elevated rate (p = 0.07) | the same, at the C-family's ~9 swaps a year | 19.69% | −35.40% | 0.558 |
| unconditional −10% hard stop | the cheap lever | 20.79% | −30.46% | 0.683 |

**research/166's sell-only result reproduces exactly on the new seeds**: selling the loser and
*not* replacing it returns Calmar **0.603**, which is **worse than never swapping at all**
(0.606), at every `k`. The cash must go back to work at the same open or the rule is not a rule.
And a random swap at the same rate is 0.13 of Calmar worse than the ranked one, so *which*
holding leaves is not incidental either. Both controls behave exactly as research/166 said they
would, which is the strongest evidence that this harness is measuring the same thing.

---

## Guarding the seven deadly sins

| Sin | How it is controlled here |
|---|---|
| **Look-ahead** | inherited unchanged from research/166 and research/170: every new rule decides on the **close of the signal bar** (`close[i−1]`) and executes both legs at **`open[i]`**; a position bought or topped up today cannot be sold today (`last_buy_i >= i`). The eligibility histogram is a recording — it reads the score the engine was about to read anyway and draws no random number. The tax-lot split makes a topped-up position **worse** off for tax, never better: shares bought yesterday cannot inherit a two-year-old LTCG clock. Inherited and disclosed: research/161 sizes an entry off a NAV marked at `close[i]` while buying at `open[i]`; kept verbatim so the baseline reproduces and **not** extended to any new rule |
| **Survivorship** | unchanged from research/161: every NSE daily series with ≥ 90 bars, dead names included, no index-membership filter. Inherited residual: `market_data.db` is not retroactively split-adjusted, so `ath_events.py` truncates each series after any day-over-day fall worse than −35% |
| **Overfitting / multiple testing** | **51 selection cells disclosed** (31 pre-registered before the first run, 20 added to execute the pre-registered plateau and control clauses). The ranking metric, both windows, the 4pp rule and the adoption bar were written into the STATUS doc before anything ran and are applied unchanged — including to the one cell that clears them. That cell is flagged in every table as a 1-of-3 ranking pick on top of a 1-of-4 destination pick, found inside a 51-cell grid |
| **Cost neglect** | 25 / 40 / 60 bps on the whole shortlist. `C_els_k1` loses its entire advantage over the staged rule by 40 bps (0.638 vs 0.684); `k = 2` turns from a wash into a loss (0.671 vs 0.684). Tax runs through the FY-netting engine, never a haircut, and is reported in rupees per cell |
| **Regime dependence** | two pre-registered windows per seed plus a third reporting window. The one cell that clears the bar earns **+0.44pp in W1 against +7.03pp in W2** and that asymmetry is stated as a flag, not buried. The staged rule's own +1.26 / +1.40 split is the contrast |
| **Correlation / single factor** | **not re-tested, and that is a gap.** No entry, exit or universe changed, so Base Age's correlation to True North and IPO Base is research/161's and research/154's. But a top-up construction tilts the book toward fewer, larger, more-extended names, which is a real and unmeasured change to the blend — one more reason nothing here is adopted |
| **Capacity / shortability** | long-only NSE cash. Median position rises from **0.426%** of the held name's own 20-day traded value (incumbent) to **0.558%** (`C_els_k1_cush`) on a ₹10 lakh book — ten times those on ₹1 crore. The binding capacity fact is not the median but the **40.8% single-name weight**: at ₹1 crore that is ₹40 lakh in one mid-cap, and the top-up buys it in a single next-open order |

---

## Honest caveats

1. **`k` was never really a free axis.** The engine caps swaps at `k` per evening but each swap
   also needs its own refused entrant, so `k` is bounded by the *smaller* of the eligible losers
   and the refused signals. That is the right model of the live rule — you cannot buy an entrant
   that did not signal — but it means "all eligible" was never reachable and the sweep could not
   have found a large `k` even if one existed. The `spill` variants were built precisely to test
   the unbounded version, and they are the wash reported in Q1.
2. **The eligibility histogram counts holdings with a finite score and an entry before today,
   but does not require a tradeable open price.** It is a frequency statistic, not a fill test;
   the true count of *actionable* pairs is slightly lower than reported, which strengthens the
   Q1 conclusion rather than weakening it.
3. **`C_els_k1_cush` is a post-hoc pick and is labelled as one everywhere.** The any-evening
   trigger came out of Axis C, the cushion ranking was 1 of 3 tried after that, and the 3× cap
   was 1 of 4. Its plateau was then measured — and passed on the margin axis, marginally.
4. **Nothing here was soaked on live data.** The event list is research/164's frozen 3,619
   events; the panel is a snapshot of `market_data.db` as of 12-Sep-2026.
5. **The 2025-26 window is reported two ways and they disagree in tone.** research/165 §8.0 ran
   research/170's engine from the live book's eleven positions over 2025-01-31 → 2026-09-11 and
   found OA-ROT-1 losing on 3 of 30 paths at 7.9 swaps a year. This study's W3 row is a slice of
   the full compounded curve and shows a 15-of-30 coin flip. **Different experiments; the
   restart-from-live-state version is the one that matches what the live book will actually
   do this year**, and it remains the more pessimistic of the two.
6. **The blend was not re-measured.** Every verdict here is standalone. Since nothing is adopted,
   nothing is owed — but a future revival of the top-up idea must clear the blend test first.

---

## What this changes

**Operationally, nothing. The live Base Age book keeps OA-ROT-1 exactly as research/165 staged
it: one swap per evening, the holding more than 10% under water leaves, the refused entrant
with the highest 252-day relative strength takes the slot.**

Three facts that did not exist yesterday:

1. **"Swap two" is not a choice the market offers.** Two eligible losers coincide with two
   refused signals about **once every two years** once the rule is running. `k` is not a
   parameter worth carrying in the runbook, and `rot_max_per_day = 1` should be treated as
   settled rather than tuned.
2. **Redeploying into existing holdings is the worst destination tested on this book** — worse
   than a new entrant on 30 of 30 seeds in all eighteen constructions, and worse than doing
   nothing in sixteen of them. The slot is the scarce resource; spending it to concentrate is
   the opposite of what the book's edge is made of.
3. **The unconditional −10% stop keeps turning up.** Third independent appearance
   (research/166, and twice here), at Calmar 0.683–0.692 with no machinery, failing only the
   CAGR-eligibility clause by about a tenth of a percentage point. If Arun ever wants a
   lower-drawdown Base Age, **that** is the lever — and it should get its own study with its
   own bar, not another footnote.

**Registered for the 26-Sep-2026 review** (the live conversion's own dated check): expect
**~4.5 swaps a year and roughly one occasion every two years where a second holding would also
have qualified**. If the live queue shows two eligible losers materially more often than that,
this study's Axis A should be re-opened on the live event log. **Do not** read a second eligible
loser as a missed opportunity — on twenty-one years of history, taking it was worth −0.21pp.

---

*Written 13-Sep-2026. Harness proved bit-identical to research/170's `sim170.py` at the defaults
and reproducing its published Part B rows on its own seed base before any selection cell ran.
Data snapshot: `backtest_data/market_data.db` as of 12-Sep-2026; event list frozen from
research/164's `events164.csv` (3,619 events). Scripts, per-cell and per-seed statistics under
`research/171_baseage_multiswap_topup/` on the VPS.*

## Cells disclosed

| category | cells | seeds | note |
|---|---:|---:|---|
| harness proof | 2 | 30 | research/170's own two cells on its own seed base — not selection |
| **Axis A — how many leave** | **9** | 30 | 5 values of `k`, 2 margins, 2 spill rules |
| **Axis B — top up an existing winner** | **18** | 30 | 3 rankings × 2 `k` × 3 caps |
| **Axis C — hybrid** | **4** | 30 | 50/50 and entrant-else-top-up, at `k` = 1 and 2 |
| **pre-registered Axis-B follow-ups** | **4** | 30 | split across the best two, any-evening trigger |
| **pre-registered plateau + follow-ups on the Axis-C leader** | **16** | 30 | 8 on the rs252 variant, 8 on the cushion variant |
| **selection cells, total** | **51** | | **budget was 70** |
| controls and nulls (not selection) | 12 | 30 | 2 references, 3 sell-only, 3 random-swap nulls, 2 stop-to-cash, 1 hard stop, 1 measure-only |
| cost re-scorings (not selection) | 29 | 30 | 14 at 40 bps, 15 at 60 bps |

**Total simulations ≈ 2,820.**
