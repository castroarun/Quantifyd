# research/166 — Open Alpha · Base Age: when the book is full and a better signal arrives, should we swap? And should bloated winners be trimmed?

## VERDICT

| axis | what it tests | verdict |
|---|---|---|
| **A — Rotation** | swap the weakest holding out for a qualifying signal the book cannot otherwise take | **SIGNAL — real, but does not clear the adoption bar. Nothing deployed.** |
| **B — Drift / trimming** | trim a position above K× its target weight; size an entry to the cash there is | **NO EDGE** |
| **C — Interaction** | best rotation × best drift | **NO EDGE — subtractive: adding the drift fixes makes rotation worse** |

**OA — Open Alpha · Base Age.** Arun asked whether a full book should rank its positions and
swap a weak one out for a better new signal. The answer, in three lines:

1. **Yes, one version of it genuinely works — and it is not the version anyone would guess.**
   Ranking by momentum, by cushion above the trailing stop, by base age or by time held all
   **lose**. The only ranking that beats the incumbent is the dullest one: **swap out a
   holding that is more than 10% under water**. That earns **+1.68pp of after-tax CAGR at a
   shallower drawdown (−33.52% vs −34.05%), on 30 of 30 paired seeds, in both windows, at
   4.2 swaps a year** — and it beats a rate-matched random swap by **+2.29pp of CAGR and
   +0.104 Calmar**.
2. **It still does not clear the bar that was written down before the run.** The best cell in
   the whole study improves paired Calmar by **+0.094** against a pre-registered bar of
   **+0.10**, and CAGR by **+1.66pp** against a bar of **+2pp**. It misses by six
   thousandths of a Calmar point. And the cell that gets closest needed a *second*,
   post-hoc choice — which of three entrant priorities gets the freed slot — so it is a
   1-of-3 pick sitting on top of a 1-of-20 pick. **Recommendation: the live Base Age book
   converting now under research/165 goes live unchanged.**
3. **Trimming does not work, and the reason is the interesting part.** You *can* abolish the
   cash-refusal problem research/164 found: demand-trimming bloated winners at 1.5× target
   and letting a short entry take what cash there is drives cash refusals from **1,953 to
   67**. It buys **+0.97pp of CAGR and +0.006 of Calmar** — because the refusals do not
   disappear, they **migrate to the slot queue**: slot refusals rise from 978 to 2,796 and
   total refusals barely move (2,931 → 2,863). **Cash and slots are not two constraints.
   They are one constraint wearing two hats**, and the hat it wears is set by how much of
   NAV a few winners happen to hold. That refines research/164's closing claim that "the
   slot count is second-order and position drift is first-order": drift is what *labels* the
   refusal, not what *causes* it.

**And one correction that matters more than anything above.** research/164 recommended
ranking a contested slot by the largest 20-day traded value (tv20), citing −32.67% drawdown
and 0.665 Calmar. A ranked tie-break is **deterministic — it has exactly one path, and no
ensemble**. Re-running it at five idle-cash rates shows its CAGR is stable (21.40–21.76%,
always above the random draw) but **its drawdown is a coin flip**: −32.67, −32.63, **−35.93**,
−32.54, −32.45 at 5.0 / 5.1 / 5.2 / 5.3 / 5.5%. Nothing about the market changes between
those runs — only which entries the book can afford on a handful of days. **The tv20 rule's
CAGR advantage survives. Its Calmar advantage is retracted.** If research/165 is about to
hard-code a tie-break, hard-code it for the CAGR reason, and do not plan on the drawdown.

---

## The numbers that matter

| | incumbent (never swap) | best pre-registered rotation `A_unre_m010` | best rotation, post-hoc entrant `X_entrs_unre_m010` | best drift that fires `B2_trimD_k150` | NIFTYBEES |
|---|---|---|---|---|---|
| after-tax CAGR (30-seed median) | **20.98%** | 22.68% | 22.73% | 21.59% | 12.30% |
| worst of 30 seeds | 20.01% | 21.70% | 21.34% | 20.36% | — |
| MaxDD | **−34.05%** | −33.52% | −31.78% | −34.35% | −59.71% |
| Calmar | **0.613** | 0.673 | **0.715** | 0.629 | 0.21 |
| paired ΔCalmar (30 seeds) | — | +0.049 (27/30) | +0.094 (**30/30**) | +0.019 (19/30) | — |
| paired ΔCAGR | — | +1.68pp (30/30) | +1.66pp (**30/30**) | +0.46pp (21/30) | — |
| W1 2005–2015 CAGR (seeds won) | 19.02% | 20.48% (30/30) | 20.36% (30/30) | 18.56% (5/30) | — |
| W2 2016–2026 CAGR (seeds won) | 22.85% | 24.84% (25/30) | 25.19% (29/30) | 24.55% (23/30) | — |
| swaps / yr | 0.0 | 4.2 | 4.5 | 0.0 | — |
| trims / yr | 0.0 | 0.0 | 0.0 | 3.8 | — |
| turnover ×NAV | 2.48 | 2.74 | 2.78 | 2.62 | — |
| **tax paid over the window** | **₹8,466,719** | ₹11,018,798 | ₹10,896,547 | ₹9,391,322 | — |
| win rate | 49.1% | 47.9% | 47.2% | — | — |
| max losing streak | 14 | 15 | 15 | — | — |
| trades / yr | 31.8 | 34.3 | 34.3 | — | — |
| profit from the ten best realisations | 36.8% | 42.1% | 43.0% | 28.8% | — |
| median position, % of the name's own tv20 | 0.429% | 0.484% | 0.500% | — | — |

Window 2005-01-03 → 2026-09-11. After tax (20% STCG / 12.5% LTCG, Indian FY loss netting),
net of 25 bps a side, **idle cash 5.2% post-tax credited daily**, 16 slots @ 6.25%, ₹10L book,
30 seeds, medians. Capacity is on a ₹10L book — at ₹1 crore every capacity figure is 10×.

---

## Harness proof (done before any selection cell)

`sim166.py` is research/164's `sim164.simulate` — itself research/161's `bt_core.simulate` —
extended only on rotation and drift. With both switched off it must reproduce research/164
exactly. It does, and not merely to the rounded median:

| | CAGR median | worst seed | MaxDD | Calmar | invested |
|---|---|---|---|---|---|
| research/164 published, 5.0% idle cash | 20.94% | 19.81% | −35.50% | 0.601 | 72.9% |
| **this harness at 5.0%** | **20.935%** | **19.81%** | **−35.50%** | **0.601** | **72.9%** |
| research/163's independent 5.2% re-run | 20.975% | 20.010% | −34.05% | 0.6135 | 72.91% |
| **this harness at 5.2% — THE BASELINE** | **20.975%** | **20.010%** | **−34.05%** | **0.6135** | **72.91%** |
| research/161 published, 5.5% idle cash | 21.26% | 19.87% | −34.80% | 0.618 | — |
| **this harness at 5.5%** | **21.26%** | **19.87%** | **−34.80%** | **0.618** | 72.9% |

**Per-seed, not just per-median.** Against research/164's `seedstats_proof.csv` the 30 CAGR,
MaxDD, Calmar, trade-count and win-rate values are **identical, maximum absolute difference
0.000000**. Against research/163's independent 5.2% implementation the 30 per-seed CAGRs are
also identical to 0.000000. Two independently written engines agree exactly. (research/164
printed 20.94 where this prints 20.93 — the underlying median is 20.935 in both; it is a
rounding tie, not a difference.)

The harness also reproduces research/164's **axis D** bit-exactly at 5.0%: tv20 tie-break
21.72 / −32.67 / 0.665, rs252 20.59 / −36.05 / 0.571, base age 21.75 / −37.59 / 0.579 — the
published figures to the digit. That is what makes the tie-break instability above a finding
rather than a bug.

**The one new input**: `st166.pkl`, the SuperTrend(14,4) **line values** (the panel stores only
the exit signal). Self-check: the direction rebuilt from those lines agrees with the panel's
stored `ST_14_4` signal on **5,122,891 of 5,122,891 bars — 100.0000%**.

---

## Q&A — Arun's questions, one at a time

**Q1. When the book is full and a better signal arrives, should we swap?**

**On this evidence: it works, but not by enough to change a live book today.** Rotation is the
strongest thing found across three studies on this book (research/162, research/164 and this
one) — the best cell wins **30 of 30 paired seeds on both CAGR and Calmar, in both windows,
and survives 40 and 60 bps** (Calmar 0.715 → 0.682 → 0.651 while the incumbent goes
0.613 → 0.574 → 0.560). research/164's best candidate managed 26 of 30 and +0.058 Calmar; this
is materially stronger. But the pre-registered bar was **+0.10 Calmar or +2pp CAGR**, and the
best cell delivers **+0.094 and +1.66pp**. It misses. The bar was set high on purpose because
the book is weeks from a real-money decision, and moving it after seeing the result is exactly
how sweeps lie.

**Q2. Ranked by what?**

**By how far under water the holding is — and by nothing else that was tested.** Five ranking
scores, four margins each, 30 seeds, paired:

| rank the weakest holding by… | best cell | CAGR | MaxDD | Calmar | vs incumbent |
|---|---|---|---|---|---|
| **unrealised return since entry** | ≥ 10% under water | **22.68%** | **−33.52%** | **0.673** | **+0.049, 27/30** |
| cushion above its SuperTrend line | ≥ 7.5pp better | 24.08% | −43.82% | 0.550 | −0.066, 0/30 |
| 12-month relative strength | ≥ 25pp better | 23.18% | −41.38% | 0.560 | −0.062, 2/30 |
| distance below its own running high | ≥ 30pp below | 20.77% | −34.05% | 0.609 | −0.001, 8/30 (it fires 0.1×/yr — it is the incumbent) |
| bars held | ≥ 320 bars | 21.24% | −33.81% | 0.621 | +0.009, 18/30 (0.3 swaps/yr — also barely fires) |

The two "obvious" answers — rotate into momentum, rotate out of whatever is closest to its
stop — are the **worst** answers. They raise CAGR a lot (up to 24.08%, +3.05pp) and blow the
drawdown out to **−41% to −44%**, because they keep the book permanently in the freshest,
most-extended breakouts. That is leverage, not selection: Calmar falls in every single one of
those twelve cells, on 0–2 of 30 seeds.

**Margin matters more than the score.** On the winning score the Calmar curve is a narrow hump,
not a shelf — 0.522 (any loss) · 0.543 (≥5%) · **0.673 (≥10%)** · 0.597 (≥20%) — and on the
finer grid 0.504 (≥2.5%) · 0.601 (≥7.5%) · **0.673 (≥10%)** · 0.641 (≥12.5%) · 0.609 (≥15%).
The immediate CAGR neighbours sit within 1.25pp of the peak, so the pre-registered plateau
clause passes by the letter, but the working band is roughly **7.5% to 12.5% under water,
firing 2–6 swaps a year**. Outside it the rule either churns (≥2.5% → 19.8 swaps/yr, Calmar
0.504) or never fires.

**Swap at most once a day.** Allowing two or three swaps a day takes the same rule from Calmar
0.673 to **0.599**.

**Who gets the freed slot matters as much as who leaves it — and that finding is post-hoc.**
Giving it to the highest-RS entrant rather than the most liquid one lifts the same rule from
0.673 to **0.715** at a −31.78% drawdown; giving it to the oldest base drops it to 0.611.
Three variants, one winner, chosen after seeing the rotation result. Discount it accordingly —
that is why the headline recommendation rests on the tv20 version.

**Q3. Is the ranking doing the work, or is it just a stop-loss?**

**The ranking is doing the work, and the buy-side leg is doing most of it.** Two controls
settle it:

| control | what it isolates | CAGR | MaxDD | Calmar |
|---|---|---|---|---|
| the rule | sell the ≥10%-under-water holding, **buy the entrant** | 22.68% | −33.52% | **0.673** |
| **sell-only** | sell it on the same trigger, **do not buy** | 21.81% | −36.27% | **0.603** |
| rate-matched **random** swap (p = 0.03, 4.0 swaps/yr) | swap a random holding at the same rate | 20.39% | −35.74% | **0.569** |
| incumbent | never swap | 20.98% | −34.05% | 0.613 |

Selling the loser and *not* replacing it is **worse than doing nothing** (0.603 < 0.613). So
this is not a stop-loss in disguise — the cash must go straight back into a fresh qualifying
breakout at the same open. And a random swap at the same frequency loses **2.29pp of CAGR and
0.104 of Calmar** to the ranked one, so the choice of *which* holding leaves is not incidental
either.

The honest counterpoint: **a plain unconditional −10% hard stop** (no rotation at all) returns
20.62% / −30.46% / **0.677** — a better Calmar than the rotation rule, on 29 of 30 seeds. It
fails the pre-registered eligibility clause because its CAGR (20.62%) is *below* the baseline:
it buys 3.6pp of drawdown with 0.36pp of return. It is a different product — insurance, not
edge — and if Arun ever wants a lower-drawdown Base Age, that is the cleaner lever and it needs
no rotation machinery at all.

**Q4. What does the swap cost in tax?**

**About ₹2.55 lakh per ₹10 lakh of book over 21.7 years — and it is modelled exactly, not
haircut.** Every swap-out is a real realisation running through the same Indian FY netting
block as any exit (20% STCG, 12.5% LTCG above 365 days, losses carried forward, settled
1 April). The incumbent pays **₹8,466,719** of tax over the window on a ₹10L book; the rotation
rule pays **₹11,018,798** — **+₹2,552,079, up 30.1%**. Turnover rises from 2.48× to 2.74× NAV
a year and trades from 31.8 to 34.3 a year. The +1.68pp of CAGR is what is left **after** all
of that. For contrast, the cushion-ranked rule at 48.8 swaps a year pays **₹16,545,356** —
nearly double the tax — and still ends up with a worse Calmar than doing nothing.

Two second-order costs the tax number does not show, both real: the win rate falls from
**49.1% to 47.9%** (you book more small losses on purpose), the worst losing streak lengthens
from **14 to 15**, and dependence on the ten best realisations rises from **36.8% to 42.1%** of
total profit. Deleting the ten best trades outright costs the incumbent −1.80pp of CAGR and the
rotation rule −2.25pp — so the advantage survives the deletion (19.18% → 20.48%), but it is
slightly more outlier-dependent than what it replaces.

**Q5. Should bloated winners be trimmed?**

**No.** Month-end trimming has no coherent plateau at all — 1.25× gives Calmar 0.595, 1.5×
0.625, 1.75× 0.613, 2× 0.593, 2.5× 0.646, 3× 0.613 — and the "best" of them, 2.5×, wins by
**never firing** (0.1 trims a year; at 3× no position in twenty-one years ever exceeds the
threshold at a month-end). Trimming on demand, only when a signal is about to be refused for
cash, is marginally better and still noise: 1.5× gives **+0.46pp CAGR and +0.019 Calmar on 19
of 30 seeds**. Nothing here is distinguishable from luck, and every trim crystallises a gain on
the book's best-performing name — which is why the demand-trim cell pays **₹9,391,322** of tax
against the incumbent's ₹8,466,719 for four-tenths of a percentage point of return.

**Q6. Does either fix the cash-refusal problem?**

**Completely — and it turns out not to matter.** This is the most useful thing in the study.

| cell | entries taken | refused: no slot | refused: no cash | total refused | CAGR | Calmar |
|---|---|---|---|---|---|---|
| incumbent | 689 | 978 | **1,953** | 2,931 | 20.98% | 0.613 |
| partial fill ≥ 25% of a slot | 746 | 2,378 | 496 | 2,874 | 21.11% | 0.615 |
| demand-trim 1.5× | 720 | 1,673 | 1,223 | 2,896 | 21.59% | 0.629 |
| **both together** | **801** | **2,796** | **67** | **2,863** | 22.05% | 0.617 |

Combining the two abolishes the cash constraint — **1,953 cash refusals become 67** — and the
book takes **112 more entries (689 → 801, +16%)**. But slot refusals rise from 978 to 2,796 and
the **total** number of qualifying signals the book turns away falls only from 2,931 to 2,863.
For all that, Calmar moves **+0.006**.

The mechanism is simple once seen: a Base Age position is held for months, so 16 slots can only
absorb about 700–800 entries in twenty-one years no matter how the money is arranged. Freeing
cash does not create capacity; it just moves the queue. **The binding constraint is slot-time,
and cash vs slots is only the label on the rejection.** research/164 was right that the book's
cash is absorbed by bloated winners, and right that this was untested — but the inference that
fixing drift would unlock the slot question does not survive the test.

**Q7. Should the LIVE Base Age book (converting now under research/165) adopt any of it?**

**No. Convert it exactly as research/161 adopted it: 16 slots at 6.25%, SuperTrend(14,4) close
trail, no stop, no rotation, no trimming.** Four reasons:

- **Nothing cleared the bar written down before the run.** 34 cells were eligible on CAGR; the
  best improved paired Calmar by +0.094 against a +0.10 bar.
- **The best cell is a pick on top of a pick.** The margin was 1 of 20; the entrant priority
  was 1 of 3, chosen after the margin. 57 selection cells were run. An improvement of +0.094
  found at the top of that grid is the size of thing multiple testing manufactures.
- **Rotation makes the book harder to hold, not easier.** Lower win rate, a longer losing
  streak, more dependence on a few names, 30% more tax, and a rule that requires a live
  process to compare every holding's unrealised P&L against a threshold every evening and then
  place two orders at the next open. That is real operational surface for +1.7pp.
- **The book has never traded live.** Adding a second untested mechanic to a system on its
  first day of real money is the wrong order of operations.

**What IS worth acting on right now, and costs nothing:** research/164's tv20 tie-break should
be adopted for its **CAGR** (stable at 21.40–21.76% across every cash rate tested, always above
the random draw's ~21.0%, and it removes the path randomness a live book cannot reproduce) —
but its published **drawdown and Calmar advantage should be struck from the record**, because
it is a single-path artifact that reverses at a 0.2pp change in the idle-cash assumption.

**Re-test date: 2027-03-13** (six months of live Base Age operation). Pass criterion: if the
live book's realised entry queue shows the same shape — signals refused while a holding sits
more than 10% under water — re-run this study's axis A on the live event log plus the extended
history and re-apply the same bar unchanged.

---

## Guarding the seven deadly sins

| Sin | How it is controlled here |
|---|---|
| **Look-ahead** | every new rule decides on the **close of the signal bar** and executes at the **next open**, the same convention the inherited entry and exit use. Rotation scores are read at `close[i-1]`, the swap's sell and buy both fill at `open[i]`; a month-end trim is decided at the month-end close; a demand trim at `close[i-1]`. A position bought today cannot be swapped out today. The SuperTrend line used for the cushion score is the same causal series whose direction the panel already stores (100.0000% agreement on 5.12M bars). Inherited and disclosed: research/161 sizes an entry off a NAV marked at `close[i]` while buying at `open[i]`; kept verbatim so the baseline reproduces, and **not** extended to any new rule. |
| **Survivorship** | unchanged from research/161: every NSE daily series with ≥ 90 bars, dead names included, no index-membership filter. Known residual, inherited: `market_data.db` is not retroactively split-adjusted, so `ath_events.py` truncates each series after any day-over-day fall worse than −35%. |
| **Overfitting / multiple testing** | **57 selection cells disclosed** (38 pre-registered + 19 added to execute the pre-registered plateau and interaction tests), plus 9 controls and 20 re-scorings. Ranking metric, both windows, the 4pp W1→W2 rule and the adoption bar were written into the STATUS doc before the first cell ran and are applied unchanged — including to the cell that misses by 0.006. The winner's post-hoc entrant choice is flagged in every table it appears in. |
| **Cost neglect** | every figure is after tax **and** net of 25 bps a side; the full 40/60 bps ladder is reported for the shortlist, and the ranking is unchanged at every rung (rotation 0.715 / 0.682 / 0.651 vs incumbent 0.613 / 0.574 / 0.560). The extra tax of churn is modelled through the FY-netting engine, never as a haircut, and reported in rupees per cell. |
| **Regime dependence** | two pre-registered windows on every cell, per seed; the full YoY table with intra-year drawdowns measured from the running peak of the **full** curve. The winning rule wins W1 on 30/30 seeds and W2 on 29/30. |
| **Correlation / single factor** | not re-tested: this study changes no entry signal and no exit rule, so the book's correlation with True North and IPO Base is research/161's and research/154's, unchanged by construction. Rotation does tilt the held book toward younger positions — a real, unmeasured change to the blend — which is one more reason not to adopt it before a live soak. |
| **Capacity / shortability** | long-only NSE cash, no shortability issue. Capacity is measured: median position rises from **0.429% to 0.500%** of the held name's own 20-day traded value and the share of trades above 1% from 33.1% to 37.4%, on a ₹10L book — ten times those figures on ₹1 crore. Rotation is modestly worse for capacity, not materially so. |

---

## Honest caveats

1. **The winner misses the bar by 0.006 Calmar.** It would be easy to present this as an
   adoption. It is not one, and the bar was chosen before the numbers existed precisely so
   that this sentence would have to be written.
2. **The entrant-priority axis is post-hoc.** `X_entrs_unre_m010` (Calmar 0.715) exists only
   because three entrant priorities were tried after the rotation result was known. The
   pre-registered version of the same rule is `A_unre_m010` (0.673), and that is the number
   any forward-looking expectation should use.
3. **The plateau is a hump, not a shelf.** Calmar falls from 0.673 to 0.601 one notch below
   the peak margin and to 0.597 two notches above. The rule needs its threshold roughly right.
4. **A −10% hard stop reaches a similar Calmar with none of the machinery**, and was excluded
   only by the pre-registered CAGR clause. Anyone reading this as "rotation is the only way to
   improve risk-adjusted return" would be over-reading it.
5. **`athdist` uses a running maximum over the panel window (from 2005-01-03)**, not each
   symbol's true all-time high, because the panel starts there. For names listed after 2005 it
   is exact; for older names it is a 2005-onward high. That score lost anyway.
6. **Trims are excluded from win rate, average win/loss and losing-streak statistics** (they
   are partial realisations, not round trips) but **included** in turnover, tax and the
   ten-best-realisations share. Stated so the tradeability columns stay comparable to
   research/161's and research/164's.
7. **Nothing here was soaked on live data.** Every figure is a simulation on the frozen
   research/161 event list.

## Next levers

1. **Re-examine the −10% hard stop as a low-drawdown Base Age variant** in its own right, with
   its own bar, rather than as a control. 20.62% / −30.46% / 0.677 on 29 of 30 seeds is a
   legitimate product for someone who cares about the hole more than the height.
2. **Re-run axis A after six months of live operation** (dated review 2027-03-13) on the live
   event queue. If the live book really does sit on under-water positions while signals go
   begging, the same rule should re-clear at least as strongly — and then it clears on
   evidence, not on a grid.
3. **Do not spend more compute on trimming or on partial fills.** Both are settled: they move
   the refusal from one queue to another and change nothing that matters.

---

*Written 13-Sep-2026. Baseline reproduced bit-exactly against research/164 (5.0%) and
research/163 (5.2%). Data snapshot: `backtest_data/market_data.db` as of 12-Sep-2026;
event list frozen from research/164's `events164.csv` (3,619 events). Scripts, per-cell and
per-seed statistics under `research/166_baseage_rotation_and_drift/` on the VPS.*

---

## Cells disclosed

| category | cells | seeds | note |
|---|---|---|---|
| harness proof | 3 | 30 | 5.0 / 5.2 / 5.5% idle cash — not selection |
| **axis A rotation (pre-registered)** | **20** | 30 | 5 scores × 4 margins |
| **axis A nulls (pre-registered)** | **3** | 30 | random swap at p = 0.05 / 0.15 / 0.35 |
| **axis B drift (pre-registered)** | **8** | 30 | month-end trim ×3, demand trim ×2, partial fill ×3 |
| **axis A/B follow-ups executing the pre-registered plateau and interaction tests** | **26** | 30 | margin plateaus on three scores, trim plateau, swaps-per-day, entrant priority, 5 interaction cells |
| baselines | 2 | 30 | random draw and tv20 tie-break at 5.2% |
| **selection cells, total** | **57** | | **budget was 120** |
| controls (not selection) | 9 | 30 | 3 tie-break-only, 2 rate-matched nulls, 1 sell-without-buying, 3 unconditional hard stops |
| re-scorings (not selection) | 74 | 30 | 27 at 40 bps, 27 at 60 bps, 20 tie-break × idle-cash rate |
| cheap 10-seed scans of the same cells | 53 | 10 | superseded by the 30-seed runs |
| outlier deletion re-runs | 8 cells | 61 each | ten best trades deleted, book re-run on all 30 seeds |

Total simulations ≈ **4,800**.
