# Quality Summit optimisation — can it earn more and fall less, and does its quality screen help INSIDE Open Alpha · Base Age?

STATUS: **DONE** — all three parts complete, published, nothing adopted.
Verdicts: **Part A CONCLUDED (no adoption)** · **Part B NO EDGE** · **Part C DILUTIVE**.
Full findings: `results/RESULTS.md`. Study page:
http://94.136.185.54:5000/app/backtest/quality-summit-optimisation-research162
Owner: `quant-researcher` agent, research/162. Host: VPS `arun@94.136.185.54`,
`/home/arun/quantifyd`, `venv/bin/python3`. Nothing here is deployed or papered; this study
recommends only.

Last updated: 2026-09-12 00:05 IST

---

## 1. The Ask

**What Arun asked (11-Sep-2026 ~23:35 IST):** *"go"* — on a brief that says: optimise
Quality Summit so it earns more and falls less, and do the overlay test that research/160
registered for 10-Oct-2026 **now** rather than in a month.

**What is actually being tested — three separate questions, three separate verdicts:**

- **Part A — can Quality Summit be made better?** Quality Summit is research/160's Family-B
  `b7` book: *close ≥ 0.9 × its own all-time-high close, 20-day median traded value ≥ ₹2 cr,
  a point-in-time Screener screen (profitable in the last 3 filed fiscal years, 3-year
  average ROE > 15%, ROCE > 15% or a lender, 3-year sales AND profit growth > 10%, market
  cap > ₹1,000 cr, **no** debt/equity test), top 15 by relative strength, rebalanced
  monthly, filled at the next open, no exit rule at all.* It returns **21.19% CAGR after
  tax with a −37.1% drawdown (Calmar 0.58)** on 2018-08-01 → 2026-09-10. r/160 swept 588
  cells of exits, gates, slot counts, screen dials and cadences and found **nothing that
  beat simply holding**. This part asks whether the axes r/160 did **not** try — ATR-scaled
  trailing exits (SuperTrend, chandelier), ranking axes other than relative strength,
  and position sizing / weighting — change that answer.
- **Part B — does the quality screen help INSIDE Open Alpha · Base Age?** r/160's closing
  recommendation, registered as a dated review for 10-Oct-2026. Open Alpha · Base Age is
  research/161's winner: *a new all-time-high CLOSE whose previous all-time high is at
  least 60 bars old and which fell at least 20% below it in between, 16 slots at 6.25% of
  NAV, ₹10 L, next-open fill, SuperTrend(14,4) close trail, no hard stop, liquidity ≥ ₹2 cr*
  — **21.26% after tax / −34.80% / Calmar 0.618** over 2005-2026 on a 30-seed median. The
  question is whether requiring a name to **also** pass a fundamental screen on its signal
  day makes that book better. Entries only; exits untouched.
- **Part C — portfolio fit.** Does either book add anything to the pair Arun's money is
  actually in — True North + Open Alpha · Base Age, 50-50, monthly rebalanced?

**What this is not.** It is not a deployment. Nothing in this study changes an engine, a
live book, a paper book, `frontend/src/data/strategies.ts`, or `/app/mpf-report`. If Part A
produces an adopted new Quality Summit spec it is **reported to Arun for his decision**.

---

## 2. The Base — what is being tested, exactly

### 2.1 The incumbent (the thing to beat), cited from r/160, not re-derived

`b7_g10_qual_mc` mask + `near-ATH k=0.90` + `tv ≥ ₹2 cr` + `rank=rs` + `slots=15` +
`cadence=monthly` + `exits=none` + `index_gate=none` + `fill=next_open` + `cost 25 bps` +
`tax on` + `cash yield 5%` + **12 rebalance-day offsets**, window 2018-08-01 → 2026-09-10:

| | after tax | MaxDD | Calmar | W1 | W2 | % invested | trades/yr | max losing streak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **QS baseline (r/160 b7)** | **21.19%** | −37.1% | **0.58** | 20.11% | 20.95% | 91.2 | 61.4 | 17 |

Also cited, not re-run: 588 cells of SMA trails 20-200, peak-drawdown exits 15-30%,
Donchian lows, time stops, hard stops, two index gates (NIFTY 200-SMA, NIFTYBEES 100-SMA
weekly), N = 8..30, k = 0.85..0.95, tv = ₹1..10 cr, cadence and hysteresis — **none beat
holding**; the strict screen (growth 20, D/E ≤ 0.2) returns 10.8%; the growth threshold is
a monotonic downhill slope, not a plateau; ROCE rejects nothing ROE has not already
rejected. The r/160 blend test was run against the **published** Open Alpha, which r/159
has since shown to rest on a same-bar look-ahead fill — so that blend result is
**unplaceable and is re-run here against Base Age**.

### 2.2 Part A — the axes r/160 did not try

**A1. ATR-scaled trailing exits.** r/160's exit family was entirely price-level based
(simple moving averages, fixed peak-drawdown percentages, Donchian lows). r/161 found that
on all-time-high entries the *exit* was worth +11.85pp of CAGR and the exit that did it was
a **SuperTrend(14,4) close trail**. That family has never been run on Quality Summit.

- SuperTrend close trail at (period, multiplier) ∈ {(7,3), (10,3), (14,4), (20,3)} — exit
  on the close of the day the SuperTrend direction flips to down. Definition copied
  verbatim from r/161's `bt_core.supertrend_dir` so the two studies are comparable.
- Chandelier close trail: exit when close < (22-day highest high − k × ATR14), k ∈ {2, 3}.
  The rolling-22-day form, not the since-entry form, so the line is path-independent.
- Each of the six, **with** and **without** the `fund_fail` exit (sell a name at the next
  rebalance once it stops passing the screen).
- On the `b7` mask, N = 15, k = 0.90, tv ≥ ₹2 cr — i.e. one axis changed at a time.
- The best of those re-checked under the NIFTYBEES 100-SMA weekly gate, `block_new` only.

**A2. Ranking axes inside the qualifying set.** r/160 ranked by relative strength only
(IBD-style 2×r63 + r126 + r189 + r252). Tested here: `rs` (baseline), `profit_g3` (the
3-year net-profit growth rate from the point-in-time panel), `opm_slope3` (the 3-year
operating-margin slope in pp/yr), a cross-sectional composite z-score `z(rs) + z(profit_g3)`,
the composite `z(rs) + z(opm_slope3)`, and `mcap_desc` (largest first, using r/142's
shares-constant market-cap proxy — quoted as a proxy). Each at N ∈ {8, 10, 15}. The two
fundamental ranking fields are read from `backtest_data/fundamentals.db`
`features_pit_monthly`, joined on the same monthly decision dates as the mask and
forward-filled the same way, so they are point-in-time by construction.

**A3. Sizing and caps.** Equal weight (baseline) vs inverse-volatility weights (1/σ over
the trailing 60 sessions, capped at 2× the equal weight and re-normalised). **The
max-25%-per-sector axis is DROPPED**: there is no sector field anywhere in this project's
data — `fundamentals.db` has none, the cached Screener pages carry only the annual /
quarterly / top-ratio blocks, and `holdings_meta.db` covers only currently-held names.
Stated rather than faked.

**A4. Screen-dial neighbours on the winner only.** growth ∈ {10, 12, 15} × k ∈ {0.85, 0.90}
× tv ∈ {₹2 cr, ₹5 cr}. Growth 12 needs a new mask built on the same
`build_study_masks.py` contract; the rest are engine axes.

### 2.3 Part B — the mask as an ENTRY filter inside Base Age

r/161's spec exactly, 30 seeds, 25 bps, after tax, idle cash 5.5%, next-open fill, 16 slots
at 6.25%, ₹10 L, SuperTrend(14,4) close trail, no hard stop, tv ≥ ₹2 cr, X ≥ 60 bars,
depth ≥ 20%, no volume filter, 60-bar re-arm applied after filtering. The **only** change:
a candidate event is dropped unless its symbol passes the eligibility mask on the signal
day (the mask row in force = the most recent 1st-of-month row at or before that date).
Exits, sizing, slot contention and costs are untouched.

Masks tested: **none (the control)**, `b7_g10_qual_mc` (Quality Summit's own),
`b3_qual_mc` (quality only — profitable, mcap > ₹1,000 cr, ROE and ROCE > 15, **no growth
test**), `growth_only` (sales and profit growth > 20, no negatives, nothing else) and
`arun_strict` (the screen exactly as Arun wrote it). Each with **missing = fail** and
**missing = pass**.

Window 2018-08-01 → 2026-09-10, because that is where the fundamentals exist. A second,
clearly labelled arm runs the full 2005 → 2026 window under both missing policies so the
long-window effect is visible — knowing that everything before Aug-2018 is decided by the
missing policy alone and is therefore a coverage artefact, not a fundamentals test.

**Bit-identity requirement:** the no-mask control cell in the r/162 copy of the engine must
reproduce r/161's WINNER row (21.26% median / −34.80% / 0.618, 30 seeds, 2005-2026) before
any masked cell is read.

### 2.4 Part C — portfolio fit

Blend the best Part-A spec and the r/160 `b7` baseline against **True North + Open Alpha ·
Base Age 50-50, monthly rebalanced**, at weights 10 / 20 / 33%, with a **cash null at the
same weight** (5% p.a.). 360 paths = Base Age's 30 seeds × True North's 12 offsets, the
r/154 pairing convention. Monthly returns, window 2018-08 → 2026-08 (the fundamentals
overlap). Per-window rows for the 2020 crash and the 2022H1 grind, drawdowns measured from
the running peak of the **full** curve. Daily and monthly correlation to each leg.

**Known inconsistency, stated up front:** True North's after-tax curve from r/159 carries
its idle cash at 6.5% p.a. while this study's books carry 5.0% (Quality Summit) and 5.5%
(Base Age). The pair is therefore flattered by a few tenths of a point relative to a
candidate sleeve. Not corrected — the curves are other studies' artefacts — but the
direction of the bias is against the candidate, which is the safe direction.

---

## 3. Pre-registration — decided BEFORE any cell ran

**Window.** 2018-08-01 → 2026-09-10. **FIT window W1 = 2018-08-01 → 2022-06-30. HOLDOUT
W2 = 2022-07-01 → 2026-09-10.** Every selection decision is made on **W1 only**. W2 is
computed and reported **once, at the end, for the chosen cells**. A cell whose W2 CAGR
falls more than **4 percentage points below its W1 CAGR** is declared **not robust** and
cannot be recommended, whatever its full-window number says.

**Ranking metric.** After-tax Calmar on W1, subject to (a) after-tax CAGR ≥ the r/160 QS
baseline's W1 CAGR on the same window, and (b) the tradeability gate being *shown* —
maximum losing streak, trades per year, turnover and capacity in the table, every time.

**Adoption bar for a NEW Quality Summit spec (Part A).** Paired across the same 12
rebalance-day offsets against the r/160 `b7` baseline:

> **≥ +0.15 Calmar OR ≥ +2pp CAGR at no worse drawdown, on ≥ 8 of 12 offsets, in W1 AND in
> W2, sitting on a plateau (its immediate parameter neighbours within ±3pp of CAGR), and
> surviving a 40 bps cost.**

**Adoption bar for the overlay on Base Age (Part B).** Paired across the same 30 selection
seeds against the no-mask control:

> **≥ +0.10 Calmar OR −3pp of drawdown at ≥ equal CAGR, on ≥ 20 of 30 seeds, in BOTH
> windows.**

**Adoption bar for a portfolio addition (Part C).** The r/154 / agent standard:
**+0.10 Calmar or −2pp drawdown at ≥ equal CAGR after tax vs the pair, robust across the
360 paths, monthly correlation < ~0.40 to both legs, AND beating the cash null at the same
weight.**

**Cell budget.** Part A ≤ 250 cells, Part B ≤ 60, Part C ≤ 40. The realised counts are
disclosed in RESULTS.md and every headline is discounted for multiple testing accordingly.

**Costs / taxes / cash.** 25 bps a side headline with a 25 / 40 / 60 ladder; 20% STCG /
12.5% LTCG with Indian FY loss-netting; idle cash 5.0% (Quality Summit, r/160's convention)
and 5.5% (Base Age, r/161's convention) — each book keeps its own study's convention so the
comparison to its own baseline is like-for-like, and the difference is stated wherever the
two sit in one table.

---

## 4. Plan — the grid and the cell count

### Part A (target ≤ 250 twelve-offset cells, `arms='tax'` for the scan, `arms='all'` for anything reported)

| Block | Axis | Values | Cells |
|---|---|---|---|
| A0 | baselines | b7 no-exit (the incumbent), no-screen control, both on W1/W2/full | 6 |
| A1 | ATR trails | {ST(7,3), ST(10,3), ST(14,4), ST(20,3), CH(22,2), CH(22,3)} × {fund_fail off, on} | 12 |
| A1b | best trail × index gate | NIFTYBEES 100-SMA weekly, block_new | 2 |
| A1c | trail × N | best 2 trails × N ∈ {10, 15, 20, 30} | 8 |
| A2 | ranking | {rs, profit_g3, opm_slope3, z(rs+profit_g3), z(rs+opm_slope3), mcap_desc} × N ∈ {8, 10, 15} | 18 |
| A2b | best rank × best trail | interaction re-check | 6 |
| A3 | weights | {equal, invvol} × {best 3 specs} | 6 |
| A4 | screen dials on the winner | growth {10, 12, 15} × k {0.85, 0.90} × tv {2, 5} | 12 |
| G3 | robustness on the top 3 | 12-offset band, worst path, W2, cost ladder 25/40/60, top-10-trade deletion, missing policy both ways | ~30 |

Planned ≈ 100 cells; the budget headroom absorbs interaction re-checks. **Interactions are
real** (r/158: a trail that won under one gate lost when the gate was retired), so whenever
one leg of a spec changes the others are re-checked jointly rather than assumed.

### Part B (target ≤ 60 thirty-seed cells)

5 masks (incl. the control) × 2 missing policies × 2 windows (2018-08→ and 2005→) = 20
cells, plus the bit-identity control and a small number of paired re-runs = ≤ 30.

### Part C (target ≤ 40)

2 candidate sleeves × 3 weights × (candidate + cash null) = 12, plus the pair, the
standalones and the two stress windows.

---

## 5. Status — live log

**State header.** Phase: **COMPLETE**. Started 2026-09-11 23:35 IST, finished 2026-09-12
~00:05 IST. 157 Part-A cells, 36 Part-B cells, 14 Part-C blend constructions. No further
compute is owed.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-11 23:17 | VPS reachable, 4 cores, 4.6 GB free RAM | one ssh session at a time |
| 2026-09-11 23:40 | Sections 1-4 written, pre-registration locked | no cell had run |
| 2026-09-11 23:22 | `aux_162.npz` built (89 s, 34 MB) | SuperTrend/chandelier signal frames, 60-day vol, PIT ranking frames; 106 symbols split-truncated |
| 2026-09-11 23:24 | `qg_engine2.py` generated from r/160's engine, 14 patches, compiles clean | r/160's engine untouched |
| 2026-09-11 23:25 | **Engine identity check PASSED** | incumbent = 21.19 / −37.07 / 0.58, r/160's published row to the second decimal |
| 2026-09-11 23:28 | Phase A1 done, 38 cells (W1) | best trail ST(20,3) +0.06 Calmar at −1.0pp CAGR; every fundamental ranking axis LOSES; `fund_fail` has zero effect |
| 2026-09-11 23:32 | Phase A2 done, 41 cells (W1) | N=10 and k=0.85 emerge as the levers; trails hurt once the book is concentrated |
| 2026-09-11 23:36 | Phase A2b done, 40 cells (W1) | 24-cell k×N plateau confirmed; QS-v2 (k 0.85, N 10, inverse-vol) = 25.59% / −30.91 / 0.83 |
| 2026-09-11 23:38 | Masks rebuilt and verified against r/160's INDEX | `growth_only` 162.6/276, `arun_strict` 26.4/46 — exact |
| 2026-09-11 23:40 | **Part B done (27 s)** — control reproduces r/161 exactly (21.26 / −34.80 / 0.618, worst seed 19.87) | every screen loses on 0 of 30 seeds |
| 2026-09-11 23:42 | **Part A G3 done — the HOLDOUT killed the candidate** | W1 +6.06pp on 12/12 → W2 −3.48pp on 3/12; W2 is 9.22pp below W1 vs a 4pp pre-registered limit |
| 2026-09-11 23:43 | Part C done | cash null beats the sleeve on 360/360 paths at every weight; QS vs Base Age monthly correlation 0.717 |
| 2026-09-12 00:00 | YoY table, comparison PNG, tearsheet written | `results/yoy162.*`, `r162_compare.png` |
| 2026-09-12 00:05 | RESULTS.md written, study published, ops review closed | 10-Oct-2026 obligation retired with a NO |

---

## 6. Crash recovery — how Arun resumes without the agent

Everything runs on the VPS at `/home/arun/quantifyd`. Nothing here writes to
`market_data.db`, to any live database, to `services/`, or to the crontab, and no backend
restart is involved at any point.

**What finished.** Each phase writes an incremental, resume-safe CSV:

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/162_quality_summit_optimisation/results
wc -l cells_a*.csv partB_cells.csv 2>/dev/null     # rows = cells completed
tail -3 /tmp/r162_*.log                            # the live logs
```

**Are the jobs alive?** `pgrep -af 'research/162'`. A dead run leaves its CSV intact.

**Resume.** Every runner skips cells whose `label` (Part A) or `cell` (Part B) is already
in the output CSV, so the identical command simply continues:

```bash
cd /home/arun/quantifyd
setsid nohup nice -n 10 venv/bin/python3 -u \
  research/162_quality_summit_optimisation/scripts/qg_engine2.py \
  --panel research/160_quality_growth_near_ath/results/panel_2000.npz \
  --aux research/162_quality_summit_optimisation/results/aux_162.npz \
  --grid research/162_quality_summit_optimisation/results/grid_a1.json \
  --out  research/162_quality_summit_optimisation/results/cells_a.csv \
  > /tmp/r162_a1.log 2>&1 < /dev/null &

setsid nohup nice -n 10 venv/bin/python3 -u \
  research/162_quality_summit_optimisation/scripts/partb_overlay.py \
  > /tmp/r162_b.log 2>&1 < /dev/null &
```

**Rebuild the derived caches if they are missing** (both are pure functions of
`market_data.db` + `fundamentals.db`, both read-only):

```bash
venv/bin/python3 research/162_quality_summit_optimisation/scripts/build_aux.py   # aux_162.npz, ~10 min
venv/bin/python3 research/162_quality_summit_optimisation/scripts/build_panel161.py  # panel161.pkl, ~20 min
```

**Do not touch:** `research/160_quality_growth_near_ath/scripts/*` (r/160's engine is
frozen; r/162 works on its own copy), `research/161_ath_base_age_breakout/results/ath_events.csv`
(inputs, regenerable but slow), `backtest_data/market_data.db`, `backtest_data/fundamentals.db`,
anything under `services/`, `frontend/src/data/strategies.ts`.

**Safe to inspect / delete and regenerate:** everything under
`research/162_quality_summit_optimisation/results/`.

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `QUALITY_SUMMIT_OPTIMISATION_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/build_aux.py` | builds `aux_162.npz`: SuperTrend / chandelier exit-signal frames, 60-day vol, point-in-time ranking frames | yes |
| `scripts/qg_engine2.py` | r/160 engine + ATR trails + new ranking axes + inverse-vol weights | yes |
| `scripts/make_grid162.py` | writes the Part-A grid JSONs | yes |
| `scripts/build_masks162.py` | the growth-12 neighbour mask | yes |
| `scripts/build_panel161.py` | rebuilds r/161's price/exit-signal panel cache | yes |
| `scripts/partb_overlay.py` | Part B: the mask as an entry filter inside Base Age | yes |
| `scripts/partc_blend.py` | Part C: correlation + blend value vs TN + Base Age | yes |
| `scripts/report162.py` | YoY table, curves PNG, tearsheet | yes |
| `results/aux_162.npz` | derived frames (~300 MB) | NO — gitignored |
| `results/panel161.pkl` | r/161 price panel cache (~1 GB) | NO — gitignored |
| `results/cells_a.csv` | Part A, one row per cell | yes |
| `results/partB_cells.csv` | Part B, one row per cell | yes |
| `results/partB_paired.csv` | Part B, per-seed paired deltas | yes |
| `results/partC_blend.md` | Part C tables | yes |
| `results/yoy162.{md,html,csv}` | the house YoY table | yes |
| `results/*.png` | factsheet + curves-vs-indices | yes (also copied to `frontend/public/`) |
| `results/RESULTS.md` | the verdicts | yes |

---

## 8. Findings

**Headline: nothing was adopted, and one dated obligation was retired.**

1. **Part A — CONCLUDED, no adoption.** The best cell in the fit window (keep the `b7`
   screen, widen the near-ATH band to k = 0.85, cut to 10 names, size inverse-vol) beat the
   incumbent by **+6.06pp of CAGR and +0.282 Calmar on 12 of 12 rebalance offsets**, sat on
   a verified 24-cell plateau, and survived the cost ladder. In the **holdout** it lost
   **−3.48pp on 3 of 12 offsets and −0.152 Calmar on 1 of 12**, with W2 running 9.22pp below
   W1 against a pre-registered 4pp limit. Over the full window it buys 0.01 points of CAGR
   for 3.9 extra points of drawdown, and deleting its ten best trades takes its compounding
   proxy **below 1.0**. The incumbent spec stands.
2. **The ATR-trail family r/160 never tried does not rescue the book.** The best,
   SuperTrend(20,3), is worth +0.06 Calmar at −1.0pp of CAGR — and it gets there partly by
   sitting 36% in cash. `fund_fail` changes literally nothing. The trails' value is
   construction-dependent: helpful at 15 slots, destructive at 10.
3. **Every fundamental ranking axis loses**, by 2 to 22 points of CAGR. Relative strength is
   the ranking; using the fundamentals to choose among the survivors actively subtracts.
4. **Part B — NO EDGE, and the 10-Oct-2026 review is closed.** Not one screen wins on a
   single seed out of thirty on return, in any window, under either missing policy. The
   mechanism is starvation: the screen cuts Base Age's qualifying events from 3,619 to 468
   (`b7`) or 76 (`arun_strict`), and the invested fraction from 87% to 63% or 19%.
5. **Part C — DILUTIVE.** Cash at the same weight beats the sleeve on 360 of 360 paths at
   10%, 20% and 33%. Monthly correlation to Base Age is 0.717 — it is the same family.
6. **The screen's real product is drawdown, not return**: 13 to 17 points off the maximum
   drawdown on 12 of 12 offsets, bought with 1.3 points of CAGR. Worth saying plainly,
   because it is a legitimate product and it is not the one that was asked for.
7. **Method note.** This is the clearest example this project has of a pre-registered
   fit/holdout split earning its keep. A 12-of-12 sweep with a plateau would have been
   published as an improvement had the W1/W2 rule not been written down first.
