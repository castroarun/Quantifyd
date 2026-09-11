# research/160 — Quality-growth near the all-time high

**Window 2018-08-01 → 2026-09-10 (8.1 years). 588 cells. After tax (20% STCG / 12.5% LTCG,
Indian FY netting), net of 25 bps a side, idle cash at 5% p.a., 12 rebalance-day offsets or
30 selection seeds per cell, medians across paths. Drawdowns measured from the running peak
of the full curve.**

---

## The verdicts

### Family A — the screen exactly as Arun wrote it: **NO EDGE**

> Sales growth 3y > 20 AND profit growth 3y > 20 AND avg ROE 3y > 15 AND ROCE > 15 AND
> D/E ≤ 0.2 AND price ≥ 0.9 × all-time high AND market cap > ₹1,000 cr; 15 names, no exit.

**10.77% CAGR after tax, −29.1% drawdown, Calmar 0.40, 43% invested.** It does not clear the
25% bar, it does not clear it at any slot count, threshold, exit, gate, cadence or entry
mechanic tested, and it is **beaten by the Midcap 150 index** (16.41%) on its own window.
Against the identical book with no screen at all it loses **11.9 points of CAGR on 12 of 12
rebalance offsets**. It also loses to picking names **at random** from the same liquid,
near-all-time-high universe (14.82%).

Three further facts decide it:

- **At a 0% idle-cash assumption it returns 7.74%.** Three of its eleven points are the
  cash yield on the 57% of the book the screen cannot fill.
- **Delete its ten best trades out of 213 and the trade-level compounding proxy falls from
  98.5× to 0.32× — below one.** Without ten names it loses money.
- **ROCE never rejects anything ROE has not already rejected**, and D/E ≤ 0.2 removes the
  entire financial sector by construction (Screener carries no `Borrowings` row for lenders).
  Two of the seven criteria are doing nothing they appear to be doing.

### Family B — what Arun's own trades say he actually does: **SIGNAL, NOT STRATEGY**

> Near the all-time high, profitable, mcap > ₹1,000 cr, ROE and ROCE > 15, growth > 10, **no
> debt test**; 15 names, RS-ranked, monthly, no exit.

**21.19% CAGR after tax, −37.1% drawdown, Calmar 0.58, 91% invested**, stable across both
sub-windows (W1 20.11%, W2 20.95%) and across the cost ladder (19.10% at 60 bps). It beats
all three indices on return and it is the best fundamental arm in the study — and it still
**misses the 25% bar, misses Calmar 1.0 by a wide margin, and fails the pre-registered
"adds value" test** against the same book with no screen (−1.32pp CAGR, +0.106 Calmar,
bar was +0.15).

### The honest answer to "does it clear 25% after tax?"

**Yes — but only by deleting the screen.** The best cell in 588 is
**`near-ATH + RS, no fundamentals at all, 30 names, 200-SMA trail, tv ≥ ₹5 cr`: 25.88% after
tax, −40.9% drawdown, Calmar 0.61**, W1 23.56% / W2 25.27%, 24.11% at 40 bps, 22.26% at
60 bps. Its plateau is real — N=30 24.35%, N=40 23.62%, N=50 23.62%, tv ≥ ₹2 cr 24.35%,
tv ≥ ₹10 cr 22.30% — so ~23-26% is the ceiling of this family, not a peak.

**It still fails the adoption bar on Calmar (0.61 vs 1.0) and it is not Arun's system.** It
is a plain relative-strength momentum book on liquid names near their highs — which is what
True North and Open Alpha already are.

### Portfolio fit: **DILUTIVE AT EVERY WEIGHT — do not add**

| added to the deployed TN+OA 50-50 pair | CAGR | MaxDD | Calmar | ΔCalmar | paths improved |
|---|---:|---:|---:|---:|---:|
| **the pair alone** | 33.68 | −14.01 | **2.369** | — | — |
| + Family B at 10% | 32.26 | −14.43 | 2.232 | −0.108 | 39/360 |
| + Family B at 20% | 30.99 | −15.64 | 2.017 | −0.355 | 16/360 |
| + Family B at 33% | 29.47 | −17.96 | 1.693 | −0.709 | 0/360 |
| **+ CASH at 20% (the null)** | 27.77 | −10.65 | **2.586** | **+0.212** | **360/360** |
| + CASH at 33% (the null) | 23.97 | −8.50 | 2.794 | +0.427 | 360/360 |

Every weight makes the book worse, monotonically, and **holding cash in its place is better
on every single one of the 360 paths.** The reason is in the correlation: Family B runs
**0.624 monthly against Open Alpha** and the price-only version runs **0.730**. This is not a
complement — it is a weaker sampling of the family Open Alpha already trades (OA alone:
44.80% / −25.76% / Calmar 1.78 on the same window).

---

## The decomposition — where the return actually comes from

| step | CAGR after tax | MaxDD | Calmar | % invested | uplift |
|---|---:|---:|---:|---:|---|
| NIFTY 50 buy-and-hold | 9.38 | −38.4 | 0.24 | 100 | — |
| NIFTY MIDCAP 150 buy-and-hold | 16.41 | −40.8 | 0.40 | 100 | — |
| random selection, liquid + near-ATH (**the null**) | 14.82 | −41.7 | 0.37 | 98.9 | near-ATH alone ≈ +2pp over an index |
| **+ RS ranking** (no screen) | 22.52 | −48.3 | 0.43 | 98.6 | **+7.7pp — this is the engine** |
| + Family B screen (`b7`) | 21.19 | −37.1 | 0.58 | 91.2 | −1.3pp CAGR, +0.15 Calmar |
| + Family A screen (`arun_strict`) | 10.77 | −29.1 | 0.40 | 43.1 | **−11.8pp** |
| + the OPM "steady or rising" step on A | 9.65 | −23.7 | 0.41 | 35.0 | −1.1pp further |
| + slots 30 + tv ≥ ₹5cr + 200-SMA trail, no screen | **25.88** | −40.9 | 0.61 | 95.5 | +3.4pp — book construction |

**Growth is the criterion that binds and the criterion that costs.** Holding everything else
fixed: growth > 15 → 15.43%, > 20 → 10.77%, > 25 → 9.78%, > 30 → 5.94%. Arun's 20% bar sits
on a monotonic downhill slope, not on a plateau.

## The pre-registered paired test — twelve masks, zero pass

Each mask against the identical book with **no screen on the same screenable sub-universe**,
paired across the same 12 offsets. Bar: ≥ +2pp CAGR **or** ≥ +0.15 Calmar on ≥ 8 of 12.

| mask | ΔCAGR | CAGR wins | ΔCalmar | Calmar wins | verdict |
|---|---:|---:|---:|---:|---|
| `arun_strict` (as written) | **−11.90** | **0/12** | −0.048 | 4/12 | no |
| `g15_mc1000` | −8.00 | 0/12 | +0.071 | 8/12 | no |
| `g15_mc500` | −6.46 | 1/12 | +0.101 | 9/12 | no |
| `no_growth` | −6.36 | 1/12 | −0.004 | 6/12 | no |
| `b5_g15_qual_mc` | −4.46 | 1/12 | −0.011 | 6/12 | no |
| `b2_noneg_mc1000` | −3.76 | 1/12 | −0.048 | 2/12 | no |
| `b4_g15_mc` | −2.94 | 2/12 | +0.026 | 7/12 | no |
| `quality_only` | −2.85 | 3/12 | +0.067 | 9/12 | no |
| `growth_only` | −2.68 | 2/12 | +0.015 | 7/12 | no |
| `b1_noneg` | −2.44 | 5/12 | −0.007 | 5/12 | no |
| `b3_qual_mc` | −2.36 | 3/12 | +0.043 | 11/12 | no |
| **`b7_g10_qual_mc`** (the best) | −1.32 | 2/12 | **+0.106** | 10/12 | **no** |

## The missing exit — it turns out not to be missing much

285 cells: 19 exit specifications × 3 index-gate settings × 5 books.

- **On the two books that can stay invested, no exit beats simply holding.** For Family B the
  whole spread from `none` to the best exit is under 1pp of CAGR; the best by Calmar is a 15%
  hard stop at 20.64% (Calmar 0.59) against no exit at 21.19% (0.58).
- On the no-screen control the 200-day SMA trail is worth **+0.06 CAGR and +0.03 Calmar** at
  N=15 — and **+1.8pp of CAGR with −1.2pp of drawdown at N=30**, which is the only place an
  exit earns its keep.
- **The index gate is not the rescue.** NIFTY-200SMA does not appear in the control's top six
  at all. On the thin screened books it raises Calmar by parking the book in cash: Family A
  with a Donchian-20 exit under the gate reaches Calmar 0.55 — at 8.04% CAGR and **27%
  invested**. That is the Calmar of a cash pile.
- This is the same shape r/71 found from the other side: a trailing stop beats a target, and
  a target is worse than nothing.

## Tradeability gate

| book | win % | avg win | avg loss | expectancy/trade | max losing streak | trades/yr | turnover | capacity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Family A as written | 44.9 | +26.3% | −10.0% | +5.77% | 11 | 25.6 | 1.9× | comfortable (thin book) |
| Family B | 46.3 | +27.9% | −10.8% | +7.26% | 17 | 61.4 | 4.2× | comfortable at ₹1 cr |
| no screen, N=30, tv ≥ ₹5cr | 43.7 | +29.8% | −11.0% | +7.00% | 29 | 149.0 | 5.1× | fine at ₹1 cr; check at ₹10 cr+ |

A **29-trade losing streak** on the best-returning book is a real tradeability problem for a
discretionary operator, and it is the number the tradeability gate exists to surface.

## Outlier dependence — every book here is a handful of names

Trade-level, one path, net of costs. `full/ex10` is the compounding proxy divided by the same
proxy with the ten best trades deleted.

| book | trades | mean | win % | top-5% of trades = | full/ex-top-10 |
|---|---:|---:|---:|---:|---:|
| Family A as written | 213 | +4.36% | 47.9 | **85% of all trade return** | 309× |
| Family A relaxed | 326 | +4.27% | 41.7 | 92% | 652× |
| Family B | 520 | +6.02% | 47.5 | 81% | 5,938× |
| no screen, N=30 tv5 | 1,265 | +6.22% | 43.5 | 96% | 167,230× |

Every arm is carried by its tail. That is normal for a momentum book and it is the reason the
drawdowns are what they are — but it means none of these results is a broad, repeatable edge
across many names.

---

## Caveats — read these before the numbers

1. **Eight years is short, and it contains the 2023-25 smallcap boom.** The window cannot be
   extended: Screener serves ~12 fiscal years, so four filed years do not exist for most
   names until FY2018 is filed in August 2018 (coverage steps 7% → 87% at that date). Nothing
   here shows how the screen behaves across a full cycle, because the data does not exist.
2. **The price universe is not point-in-time.** `market_data.db` keeps only 102 stopped
   series in 2,158 (4.7%) across eleven years — fewer than NSE actually delisted or
   suspended. A company that never entered the price database is invisible to this study and
   to its own coverage audit. Survivorship pressure is **upward on every arm, benchmarks
   included**. The random-selection null is the control that neutralises it for the ranking
   claim, because it carries the identical bias.
3. **Screener's own coverage is, unusually, not the problem** — it keeps delisted pages, so
   2,131 of 2,158 universe names are covered and the missing-data policy changes results by
   0.0-1.7pp. Every arm was run both ways regardless.
4. **The fundamentals are restated, not as-reported.** The filing lag controls *when* a year
   becomes visible; it cannot undo a later restatement. That is the residual look-ahead.
5. **`market_data.db` is not retroactively split-adjusted.** The engine restarts the ATH
   `cummax` on any one-day close collapse below 0.55× (152 events logged in
   `results/panel_split_events.csv`), which also restarts on genuine crashes and therefore
   makes the near-ATH state *easier* to satisfy for those names. Direction stated, not hidden.
6. **Weights are not rebalanced between entries** — winners run and the book drifts from
   equal weight.
7. **Tax is an approximation of the statute**: one netted FY pool with per-trade rates and
   loss carry-forward, not the STCL/LTCL set-off ordering.
8. **588 cells were run.** Discount the best cell for multiple testing accordingly; that is
   why plateaus, paired tests and a random null are reported rather than a single winner.
9. **The replication gate is the frame for all of it.** The written screen picks 8 of the 69
   equities Arun actually holds and 3 of the 43 he has been observed buying. A backtest of
   the screen as written is a backtest of a strategy he does not run.

---

## What this changes

1. **Arun's screen, as written, is not the thing that works.** His *practice* — buy near the
   all-time high, keep it if it is profitable and reasonably run — is roughly twice as good
   as the query he believes he is following, and the growth and debt bars are what cost him.
   If he wants to keep a fundamental filter, the honest version is **profitable, ROE and ROCE
   above 15, growth above 10, and no debt test at all**.
2. **The return in this family comes from relative strength, not from the fundamentals.**
   Ranking by RS is worth +7.7pp over random selection in the same near-ATH universe; the
   best fundamental screen is worth −1.3pp and +0.1 Calmar.
3. **Nothing here should be deployed or paper-traded.** It is 0.62-0.73 correlated with Open
   Alpha, it dilutes the live pair at every weight, and a cash sleeve in its place beats it on
   360 of 360 paths.
4. **There is one genuinely useful residual**: the point-in-time Screener panel and the 37
   masks now exist and are reusable. Any future study that wants a *causal* fundamental gate
   — including re-testing whether quality helps **inside Open Alpha's own entries** rather
   than as a standalone book — can use them without re-fetching.

**Recommended next step (not run here):** test `b7`-style quality as an **overlay on Open
Alpha's existing entries**, not as a separate book. That is the only version of this question
the correlation does not already answer.

---

## Files

| file | what |
|---|---|
| `results/cells_g1.csv` | G1 decomposition, 133 cells |
| `results/cells_g2.csv` | G2 exits × gates × book construction, 413 cells |
| `results/cells_g3.csv` | G3 robustness on the six finalists, 42 cells |
| `results/paired_g1.md` | the pre-registered paired test, 12 masks |
| `results/g3_outliers.md` | outlier dependence, trade level |
| `results/g4_blend.md`, `g4_blend_ctrl.md` | correlation + blend value vs TN/OA |
| `results/yoy_study.{md,html,csv}` | the house YoY table |
| `results/qg_compare.png` | log growth of 100, all books + indices, drawdown panel |
| `results/tearsheet_qg_family_b.png` | the client factsheet for the best fundamental book |
| `results/masks_study/*.npz` | the six Family-B masks built by this leg |
