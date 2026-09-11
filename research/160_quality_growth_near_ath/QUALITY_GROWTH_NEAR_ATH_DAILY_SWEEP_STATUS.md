# Quality-Growth Near All-Time-High — THE STUDY (G1→G4 sweep + publish)

STATUS: **DONE** — G1→G4 complete, 588 cells, published. Verdict in section 9.
Sections 1-4 were written, with the ranking metric and adoption bar pre-registered, before any cell ran.

Owner: STUDY-LEG agent (acting as the project's `quant-researcher`).
Siblings, both **DONE** and consumed read-only, never modified:

- `QUALITY_GROWTH_NEAR_ATH_DATA_LEG_STATUS.md` — the point-in-time Screener panel, the 31
  eligibility masks, the coverage audit and the replication gate against Arun's own book.
- `QUALITY_GROWTH_NEAR_ATH_ENGINE_BUILD_STATUS.md` — the close-only positional backtest
  engine, its five self-tests and its price-only baselines.

Last updated: 2026-09-11 20:50 IST

---

## 1. The Ask

**What Arun asked (his own words, via the orchestrator).** He runs a screener.in query,
checks liquidity and OPM by hand, buys near the all-time high, and holds with **no exit
rule**. He says he beats NIFTY discretionarily. His minimum expectation is **25% CAGR after
tax**, "the max the best". The query:

```
Sales growth 3Years > 20 AND Profit growth 3Years > 20 AND
Average return on equity 3Years > 15 AND Return on capital employed > 15 AND
Debt to equity <= 0.2 AND Current price >= 0.9 * High price all time AND
Market Capitalization > 1000
```

**What is actually being tested.** Three questions, deliberately separated, because the
replication gate says they are not the same question:

1. **Does the screen as written, bought near its high and held, clear 25% CAGR after tax
   over the honest window** (2018-08-01 → 2026-09-10), at 25 bps a side, after 20% STCG /
   12.5% LTCG, with the drawdown and the tradeability that come with it?
2. **Does what Arun actually does** — near-ATH, profitable, growth around 15%, debt
   ignored — do better or worse than the screen he believes he runs?
3. **What is the missing piece of his process?** He has no exit. The engine can price one.
   Which exit, gate and cadence turn a screen into a book, and what does that cost or buy?

**The replication gate already ruled, and it rules the whole write-up.** From
`results/holdings_check.md`, on 69 equities in his real Zerodha history:

| cohort | n | passes the screen as written | passes screen AND near-ATH |
|---|---:|---:|---:|
| Observed buys (first seen after 2026-04-21) | 43 | **3 (7%)** | 2 (5%) |
| All equities held | 69 | **8 (12%)** | 5 (7%) |

and the relaxation ladder on the same names and dates: growth→15% gives 19; growth dropped
gives 27; D/E dropped gives 14; **growth AND D/E both dropped gives 40**; **near-ATH alone,
no fundamentals, gives 42 of 69**.

> **What he actually buys is "near its high and profitable", not "20%+ growth on both lines
> and nearly debt-free".** Therefore this study runs **two labelled families and never
> blurs them**:
>
> - **Family A — the screen as written.** `arun_strict` and its immediate neighbours. A
>   result here is a result about a strategy Arun does **not** currently run.
> - **Family B — what he actually does.** Near-ATH + no negatives + relaxed growth ≥ 15 +
>   no debt/equity test, with the full leave-one-out ladder between A and B.
>
> A Family-B number may never be reported as a validation of Family A, or vice versa.

---

## 2. The Base — what is being simulated, exactly

### 2.1 Inherited, not re-litigated

The engine's causality contract, data hygiene and metric definitions are the ENGINE leg's
and are taken as given: **close-only decisions, next-open fills**, phantom-row purge, split
guard on the ATH `cummax`, `stale_exit_days=60`, tax = 20% STCG / 12.5% LTCG netted within
the Indian FY with loss carry-forward, idle cash at 5% p.a., MaxDD measured from the running
peak of the **full** curve (the r/154 convention, never a within-year slice).

The panel's fundamental masks are the DATA leg's and are taken as given: monthly rows on the
1st, forward-filled; only fiscal years filed by that date; `has_data` = four filed years.

### 2.2 The window, and why it is this one

**Honest window: 2018-08-01 → 2026-09-10 (8.1 years).** Screener serves ~12 fiscal years, so
FY2015 is the earliest year for most names and four filed years do not exist until FY2018 is
usable under the 4-month filing lag. Coverage steps 7% → 87% at exactly 2018-08-01. Anything
earlier measures Screener's page depth, not the screen.

**Two windows, pre-registered:** W1 = 2018-08-01 → 2022-06-30, W2 = 2022-07-01 → 2026-09-10.
Both roughly four years. 2020 (the COVID crash) sits in W1; the 2023-25 smallcap boom sits
entirely in W2.

**Said out loud in every report:** eight years is a short window, it contains one of the
largest smallcap advances in Indian market history (2023-25), and the study cannot show how
this screen behaves across a full cycle because the data does not exist.

### 2.3 Family A — the screen as written

`results/masks/arun_strict.npz`: `sales_g3>20 & profit_g3>20 & roe_avg3>15 & roce>15
(lenders judged on ROE alone) & de<=0.2 & mcap_pit>1000cr & no negatives in the last 3 FY`.
Identical by construction to `g20_mc1000`, `de0p2` and `q15`.

Passes 5 names/month in 2018 rising to 91 in 2025; mean 26.4. **The near-ATH condition cuts
it further**, so `avg_pct_invested` is read next to every CAGR in this family — a thinly
invested book's "CAGR" is the 5% idle-cash yield.

### 2.4 Family B — what he actually does

Six new masks built by **this** leg (`scripts/build_study_masks.py`, writing to
`results/masks_study/`, touching nothing the DATA leg owns), from the DATA leg's own panel,
so the ladder from A to "near-ATH alone" is continuous:

| mask | definition |
|---|---|
| `b1_noneg` | no negative Sales or Net Profit in the last 3 filed FY — "profitable" |
| `b2_noneg_mc1000` | `b1` & `mcap_pit > 1000cr` |
| `b3_qual_mc` | `b2` & `roe_avg3>15` & `roce>15` (lenders: ROE only) — **growth AND D/E dropped**, the 40-of-69 variant |
| `b4_g15_mc` | `b2` & `sales_g3>15` & `profit_g3>15` — relaxed growth, **no** D/E, **no** quality |
| `b5_g15_qual_mc` | `b3` & `sales_g3>15` & `profit_g3>15` — **the Family-B headline**: near-ATH + profitable + growth ≥ 15 + quality, no D/E |
| `b7_g10_qual_mc` | `b3` & growth > 10 — the softer neighbour of `b5` |

`g15_mc1000` (Family B + D/E re-imposed) already exists in the DATA leg's set and is the
bridge cell back to Family A.

Every study mask is False wherever `has_data` is False, exactly like the DATA leg's, and
ships with the same `dates / cols / mask` contract.

### 2.5 Price side, book and costs (the G1 spine)

| Axis | G1 value | Swept at |
|---|---|---|
| entry | `rebalance`, monthly cadence | G2 |
| ranking | `rs` (IBD-style relative strength) | G1 (incl. `random` as the control) |
| slots | 15 | G1 ladder + G2 |
| near-ATH | `close >= 0.90 × causal ATH close` | G1 ladder 0.85 / 0.95 / `new_ath` |
| liquidity | 20-day median traded value ≥ ₹2 cr | G1 ladder 1 / 5 |
| exits | `none` and `fund_fail` | **G2 is the exit phase** |
| index gate | `none` | G2 |
| missing-data policy | **both** `fail` and `pass`, every arm | binding, never one |
| costs | 25 bps/side | G3 ladder 25 / 40 / 60 |
| tax | on (20/12.5, FY-netted) | always |
| ensemble | 12 rebalance-day offsets (30 seeds for daily entries and for random ranking) | always |

### 2.6 Success criterion, ranked metric, adoption bar — PRE-REGISTERED

**Ranking metric for every table: after-tax Calmar at 25 bps/side**, with the tradeability
gate shown in the same table (win rate, avg win %, avg loss %, expectancy net per trade, max
losing streak, trades/yr, capacity, turnover, `avg_pct_invested`).

**A mask arm counts as adding value** if, paired across the 12 offsets against the *same*
book with **no mask**, it wins by **≥ +2pp after-tax CAGR OR ≥ +0.15 Calmar on ≥ 8 of 12
offsets**, AND beats the random-selection null.

**Adoption bar for a final spec** (all five, no partial credit):

1. after-tax CAGR **≥ 25%** AND Calmar **≥ 1.0** over the full window;
2. positive in **both** sub-windows;
3. **plateau** — the parameter neighbours on every swept axis within ±3pp of after-tax CAGR;
4. survives the **cost ladder at 40 bps**;
5. passes the **tradeability gate** — win rate, avg win/loss, expectancy, max losing streak,
   trades/yr and capacity all stated and none disqualifying.

These may be tightened after seeing results. They may **not** be loosened.

---

## 3. Plan — the four gates and their cell counts

| Gate | Question | Cells | Ensemble |
|---|---|---:|---|
| **G1** | Decomposition: index → near-ATH → +RS → +fundamentals → +OPM. Which masks survive? | ~100 | 12 offsets / 30 seeds |
| **G2** | The missing exit, the gate, the cadence and the book size — for survivors and for the no-mask control | ~400-800 | 12 offsets / 30 seeds |
| **G3** | Robustness on the top ~5 per family: two windows, per-year, outliers, cost ladder, missing policy, cash yield, survivorship | ~80 | 12 offsets / 30 seeds |
| **G4** | Portfolio fit: correlation and blend value against True North and Open Alpha | ~40 | r/154 daily curves |

### G1 grid, in full

**Price-only controls (no mask).** near-ATH k=0.90 N=15 RS monthly baseline; k ∈ {0.85,
0.95} and `new_ath`; N ∈ {10, 20}; tv floor ∈ {1, 5}; rank `dist_ath`; **random-selection
null** (rank=random, 30 seeds); `has_data` as a mask both ways — the *screenable
sub-universe with no screen applied*, which is the only fair comparator for a screened arm.

**Fundamentals only, near-ATH removed** (`k=0.0`, so `close >= 0` is always true): the
Family-A mask and the Family-B ladder, both missing policies. This is the arm that says
whether the fundamentals are doing anything at all once the price state is gone.

**Masks at k=0.90**, each run with `mask_missing` = `fail` and = `pass`:
`arun_strict`, `no_growth`, `no_roe`, `no_roce`, `no_de`, `no_mcap`, `growth_only`,
`quality_only`, `g15_mc500`, `g15_mc1000`, `g15_mc2500`, `g20_mc500`, `g20_mc2500`,
`g25_mc1000`, `g30_mc1000`, `de0p5`, `de1p0`, `q12`, `q20`, `opm_slope_pos`, `opm_steady`,
`opm_min`, `opm_rising_q`, plus the six study masks `b1`…`b7`.

**Ranking control**: `arun_strict` and `b5_g15_qual_mc` with `rank=random`, 30 seeds. The
difference against the same mask RS-ranked isolates the mask's contribution from RS's.

**Ladders on the two family leaders**: N ∈ {8, 10, 20, 30} and k ∈ {0.85, 0.95, `new_ath`}.

**Disclosed for the multiple-testing haircut: G1 is ~100 cells.** The ENGINE leg's ~20
self-test cells are validation, not discovery, and are excluded.

### Decomposition table G1 must produce

| step | after-tax CAGR | MaxDD | Calmar | % invested | paired uplift vs the step above |
|---|---|---|---|---|---|
| NIFTY 50 / MIDCAP 150 / SMALLCAP 250 buy-and-hold | | | | 100 | — |
| near-ATH only, random selection (the NULL) | | | | | |
| near-ATH only, RS-ranked | | | | | RS's contribution |
| + fundamentals (Family A / Family B) | | | | | **the screen's contribution** |
| + OPM | | | | | the manual step's contribution |

### G2, G3, G4 — as specified in the brief

G2: exits `{none, fund_fail, sma_trail 20/50/100/200, peak_dd 15/20/25/30, donchian_low
20/50, time 12/24, hard_stop 15/20}` and the sensible combos (`fund_fail`+trail,
trail+`peak_dd`); index gate `{none, nifty200sma, niftybees100sma_weekly}` × action
`{block_new, liquidate_all}`; cadence `{monthly, quarterly, semiannual}`; buffer 1.0 vs 1.5;
N `{10, 15, 20, 30}`; entry `{rebalance, first_qualify (30 seeds), ath_breakout}`. **The
exit winner is re-checked under each gate** — interactions are real (playbook §5).

G3: two windows; per-year with intra-year DD from the full-curve peak; delete-top-10-trades
and winner caps at +50%/+100%; cost ladder 25/40/60; both missing policies; offsets/seeds
median [min..max] and worst path; cash yield 0% vs 5%; the split-scale suspects; the
survivorship statement.

G4: daily and monthly correlation to True North (`r/154 results/tn_navs12.csv`, 12 offsets)
and Open Alpha (`oa_navs30.csv`, 30 seeds) over the **overlapping** 2018-08 → 2026-08 window
only; a 3-sleeve blend weight sweep 10-40% against the TN+OA pair; a **cash null at the same
weight** (the r/146 lesson: a sleeve must beat holding cash in its place).

---

## 4. Files (output map)

All paths relative to `research/160_quality_growth_near_ath/`.

| File | Purpose | Committed? |
|---|---|---|
| `QUALITY_GROWTH_NEAR_ATH_DAILY_SWEEP_STATUS.md` | this file — the sole crash-recovery source | yes |
| `scripts/build_study_masks.py` | the six Family-B masks, from the DATA leg's panel | yes |
| `scripts/make_grid.py` | emits the G1/G2/G3 cell JSONs | yes |
| `scripts/decompose.py` | the G1 decomposition + paired-uplift table | yes |
| `scripts/robustness.py` | G3: windows, outliers, cost ladder, cash yield | yes |
| `scripts/blend.py` | G4: correlation + blend weight sweep vs TN/OA | yes |
| `scripts/report.py` | RESULTS.md, the YoY table, the study-page payload | yes |
| `results/masks_study/*.npz` | the six Family-B masks | yes (small) |
| `results/grid_g1.json` … `grid_g4.json` | the cell definitions actually run | yes |
| `results/cells_g1.csv` … | one row per completed cell, resume-safe | yes if small |
| `results/*_equity.csv` | per-path daily equity curves for the finalists | the small ones |
| `results/RESULTS.md` | **the verdict**, per family | yes |
| `results/yoy_study.{md,html,csv}` | the house YoY table | yes |
| `results/qg_tearsheet.png`, `results/qg_curves.png` | the figures, copied to `frontend/public/` | yes |
| `results/panel_2000.npz`, `results/screener_cache/` | inherited, heavy | **NO — gitignored** |

---

## 5. Status (live log)

**Phase:** DONE.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-11 18:36 | Study leg opened. Doctrine read in full: `quant-researcher.md`, playbook §4/§5/§5A, both sibling STATUS docs, `SELFTEST.md`, `masks/INDEX.csv`, `holdings_check.md`, `coverage_audit.md`, the engine CLI and the r/154 curve files | |
| 2026-09-11 18:45 | **Sections 1-4 written before any cell ran**, ranking metric and adoption bar pre-registered | this file |

## 6. Crash Recovery — resuming without Claude

Everything runs on the VPS at `/home/arun/quantifyd/research/160_quality_growth_near_ath`.
Nothing here writes to `market_data.db`, any live DB, `services/`, or the crontab; no
restart is ever issued.

**Is anything still running?**

```bash
ssh arun@94.136.185.54 "ps -ef | grep qg_engine | grep -v grep"
```

**How far did it get?** One CSV row per completed cell, appended the moment the cell ends:

```bash
ssh arun@94.136.185.54 "cd /home/arun/quantifyd/research/160_quality_growth_near_ath && \
  wc -l results/cells_g1.csv results/cells_g2.csv results/cells_g3.csv 2>/dev/null; \
  tail -3 results/g1.log"
```

**Resume any phase** — the engine skips any cell whose `label` already exists in `--out`,
so re-launching the identical command is always safe and never duplicates work:

```bash
cd /home/arun/quantifyd
setsid nohup nice -n 10 venv/bin/python3 \
  research/160_quality_growth_near_ath/scripts/qg_engine.py \
  --panel research/160_quality_growth_near_ath/results/panel_2000.npz \
  --grid  research/160_quality_growth_near_ath/results/grid_g1.json \
  --out   research/160_quality_growth_near_ath/results/cells_g1.csv \
  > research/160_quality_growth_near_ath/results/g1.log 2>&1 < /dev/null &
```

Substitute `g2`, `g3` for the later phases. To force a cell to re-run, delete its row from
the CSV. **Max two concurrent workers** (engine peak RSS ~500 MB-1.5 GB; the VPS has ~3 GB
free and runs the live trading service), always under `nice`.

**Rebuild the Family-B masks** (seconds, reads the DATA leg's panel read-only):

```bash
venv/bin/python3 research/160_quality_growth_near_ath/scripts/build_study_masks.py
```

**Rebuild the price panel** (~2 min, only if `results/panel_2000.npz` is missing):

```bash
nice -n 10 venv/bin/python3 research/160_quality_growth_near_ath/scripts/qg_panel.py \
  --base-start 2000-01-01 --out research/160_quality_growth_near_ath/results/panel_2000.npz
```

**Never touch:** `backtest_data/market_data.db`, anything under `services/`, the crontab,
the `quantifyd` service, and the DATA leg's files (`build_*.py`, `screener_fetch_full.py`,
`coverage_audit.py`, `holdings_check.py`, `results/screener_cache/`, `results/features_pit*`,
`results/masks/`) or the ENGINE leg's (`qg_engine.py`, `qg_panel.py`, `selftest.py`).

**Safe to delete and regenerate:** everything matching `results/cells_g*.csv`,
`results/grid_g*.json`, `results/masks_study/`, `results/*_equity.csv`, `results/yoy_study*`,
`results/qg_*.png`, `results/RESULTS.md`.

## 7. Findings

*(filled in live as cells complete — see §8 below once G1 returns)*

---

## 8. G1 findings — the decomposition (99 cells, complete)

All figures: 2018-08-01 → 2026-09-10, monthly rebalance, RS ranking, N=15, k=0.90,
tv ≥ ₹2 cr, no exits, no index gate, 25 bps/side, **after tax**, 12 rebalance-day offsets,
median across paths. `inv%` = `avg_pct_invested`.

### 8.1 The headline: the screen as written SUBTRACTS from the price signal

| arm | after-tax CAGR | worst path | MaxDD | Calmar | inv% | tr/yr |
|---|---:|---:|---:|---:|---:|---:|
| NIFTY 50 B&H (window) | *see §8.5* | | | | 100 | — |
| **random selection, same near-ATH universe (the NULL)** | **14.82** | 8.51 | −41.68 | 0.37 | 98.9 | 156.9 |
| near-ATH k=0.90 + RS, **no screen** (raw universe) | 21.24 | 15.00 | −49.49 | 0.43 | 98.7 | 89.5 |
| near-ATH k=0.90 + RS, **no screen, screenable sub-universe** (`has_data`) | **22.52** | 17.34 | −48.27 | 0.43 | 98.6 | 88.0 |
| **+ Family A, the screen as written** (`arun_strict`) | **10.77** | 8.80 | −29.06 | 0.40 | **43.1** | 25.6 |
| + Family A, missing = pass | 12.42 | 10.50 | −29.06 | 0.44 | 47.3 | 32.0 |
| **+ Family B headline** (`b5_g15_qual_mc`) | 17.25 | 13.45 | −39.97 | 0.43 | 82.5 | 53.8 |
| + Family B, quality only, no growth (`b3_qual_mc`) | 20.21 | 16.57 | −40.24 | 0.49 | 97.8 | 69.5 |
| **+ Family B, softest growth bar** (`b7_g10_qual_mc`) | 21.19 | 17.44 | **−37.07** | **0.58** | 91.2 | 61.4 |
| + OPM steady on top of Family A (`opm_slope_pos`) | 9.65 | 8.00 | −23.72 | 0.41 | 35.0 | 21.6 |

**Read it in one line: the tighter the fundamental screen, the worse the book.** The screen
as written costs **−11.8pp of after-tax CAGR** against the identical book with no screen on
the same screenable universe (10.77 vs 22.52), and it **loses to picking names at random
from the near-ATH universe** (14.82). Its drawdown is smaller only because it cannot stay
invested: 43% of the book sits in cash earning 5%.

### 8.2 What RS is worth, and what the screen is worth

| step | after-tax CAGR | uplift |
|---|---:|---:|
| random selection from liquid + near-ATH | 14.82 | — |
| RS ranking on the same set (raw universe) | 21.24 | **+6.4pp — RS is the engine** |
| RS + `arun_strict` | 10.77 | **−11.8pp vs the `has_data` control** |
| RS + `b7_g10_qual_mc` | 21.19 | −1.3pp CAGR, **+0.15 Calmar, +11pp of drawdown saved** |

`arun_strict` with the ranking removed (30 random seeds) returns 8.08%; RS-ranked it returns
10.77%. RS adds +2.7pp even inside the strict screen — but the screen has already thrown away
more than RS can recover.

### 8.3 The near-ATH condition is doing very little on its own

Random selection from the liquid **near-ATH** universe returns 14.82% after tax; the ENGINE
leg's hold-forever control over the same kind of universe returns ~12%. The near-ATH state is
worth a couple of points at most. The two things that move this book are **relative strength**
and, as G2 will test, **the exit and the regime gate that Arun does not have**.

Removing near-ATH entirely (k=0, fundamentals only) gives: `arun_strict` 8.64%,
`b3_qual_mc` 18.73%, `quality_only` 17.42%, `growth_only` 19.43%. So near-ATH adds
+1.5 to +2.1pp on the loose screens and +2.1pp on the strict one — real, small, and nowhere
near the 25% bar by itself.

### 8.4 Which criterion is doing the damage — leave-one-out on Family A

`arun_strict` 10.77 · drop the market-cap floor 11.77 · drop ROCE 10.70 (inert, exactly as
r/158 found) · drop ROE 12.98 · drop D/E 15.02 · **drop growth 15.75** · growth-only 19.40 ·
quality-only 19.15. **Growth is what binds and growth is what costs**, and the threshold
ladder is monotonic in the wrong direction: growth > 15 → 15.43, > 20 → 10.77, > 25 → 9.78,
> 30 → 5.94. Arun's 20% bar sits on a slope, not on a plateau, and the slope runs downhill.

The three OPM readings of his manual "margins steady or rising" step all make it worse:
slope ≥ 0 → 9.65, range ≤ 5pp → 7.45, min ≥ 10pp → 8.58, 8-quarter rising → 5.26 (and that
last one is a recent-window artefact by construction).

### 8.5 Missing-data policy: the coverage bias is small

Every arm was run both ways. The `fail` / `pass` gap is 0.0–1.7pp of CAGR across the whole
grid (`arun_strict` 10.77 / 12.42; `b7` 21.19 / 21.19; `has_data` 22.52 / 22.52), because the
DATA leg covers 98.7% of the universe. This is the first study in this project where the
coverage bias is genuinely immaterial — and it is measured, not assumed.

### 8.6 Live read, before G2

- **No cell in G1 clears the 25% after-tax bar.** The best is 24.69% (price-only, tv ≥ ₹5 cr,
  no fundamentals at all) at a −48.8% drawdown, Calmar 0.51. The screen Arun believes he runs
  returns 10.77%.
- **The only fundamental arm that clears the pre-registered "adds value" bar is the very
  loosest one** — `b7_g10_qual_mc` (profitable, mcap > ₹1,000 cr, ROE and ROCE > 15, growth
  > 10, **no debt test**): +0.15 Calmar and 11pp less drawdown for −1.3pp of CAGR. The paired
  12-offset test is what decides it, and it is running.
- **Slots and liquidity are live axes the first pass under-explored**: N=30 beats N=15
  (22.90 vs 21.24, Calmar 0.54 vs 0.43) and tv ≥ ₹5 cr beats tv ≥ ₹2 cr (24.69 vs 21.24).
  G1b adds those ladders to the mask leaders.
- G1 was **99 cells**; G1b adds **34**. Both counts are disclosed for the multiple-testing
  haircut.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-11 18:52 | Six Family-B masks built (`results/masks_study/`) | b5 passes 26→244 names/month, vs arun_strict's 5→91 — Family B can actually fill a 15-slot book |
| 2026-09-11 18:55 | **G1 launched**, 99 cells | ~6 s per 12-offset cell on the 8-year window |
| 2026-09-11 19:05 | **G1 complete**, 99/99 | findings §8.1-8.6 above |
| 2026-09-11 19:12 | G1b launched (34 supplementary cells) + `pairs1.json` written | N / k / tv / rank ladders on the two Calmar leaders; the paired test |

### 8.7 The pre-registered PAIRED test — NO fundamental mask clears the bar

`results/paired_g1.md`. Every mask arm against the **identical book with no screen** on the
**screenable sub-universe** (`has_data`), paired across the same 12 rebalance-day offsets.
Bar: ≥ +2pp after-tax CAGR **or** ≥ +0.15 Calmar, on ≥ 8 of 12 offsets.

| mask | ΔCAGR median | CAGR wins | ΔCalmar median | Calmar wins | verdict |
|---|---:|---:|---:|---:|---|
| **`arun_strict` (the screen as written)** | **−11.90** | **0/12** | −0.048 | 4/12 | **no** |
| `g15_mc500` | −6.46 | 1/12 | +0.101 | 9/12 | no |
| `g15_mc1000` | −8.00 | 0/12 | +0.071 | 8/12 | no |
| `no_growth` | −6.36 | 1/12 | −0.004 | 6/12 | no |
| `growth_only` | −2.68 | 2/12 | +0.015 | 7/12 | no |
| `quality_only` | −2.85 | 3/12 | +0.067 | 9/12 | no |
| `b1_noneg` | −2.44 | 5/12 | −0.007 | 5/12 | no |
| `b2_noneg_mc1000` | −3.76 | 1/12 | −0.048 | 2/12 | no |
| `b3_qual_mc` | −2.36 | 3/12 | +0.043 | 11/12 | no |
| `b4_g15_mc` | −2.94 | 2/12 | +0.026 | 7/12 | no |
| `b5_g15_qual_mc` | −4.46 | 1/12 | −0.011 | 6/12 | no |
| **`b7_g10_qual_mc` (the best of them)** | −1.32 | 2/12 | **+0.106** | **10/12** | **no** |

**Twelve masks, zero pass.** The screen as written loses on **every single offset** and by
nearly twelve points of CAGR. The best fundamental arm in the whole study — the loosest one,
`b7` — buys about a tenth of a Calmar point and gives back 1.3 points of CAGR, and it misses
the pre-registered bar it was measured against.

### 8.8 G2a — the exit bake-off (285 cells, 19 exits × 3 gate settings × 5 books)

| book | best by after-tax CAGR | best by Calmar | what the exit and the gate are worth |
|---|---|---|---|
| **no screen (`ctrl`)** | `sma_trail:200`, no gate — **22.58%**, DD −47.5, Calmar 0.46 | same | the 200-SMA trail is worth **+0.06 CAGR and +0.03 Calmar** over no exit at all. Every other exit is neutral or negative. **The NIFTY-200SMA gate does not appear in the top six** |
| Family A `arun_strict` | no exit — 10.77%, DD −29.1, Calmar 0.40 | `donchian_low:20` + NIFTY gate — 8.04%, DD −13.1, **Calmar 0.55**, but only **27% invested** | the gate raises Calmar by cutting exposure, not by improving the book |
| Family A `g15_mc500` | no exit — 17.12%, Calmar 0.55 | `donchian_low:50` + NIFTY gate — 13.74%, DD −22.1, **Calmar 0.67** | the only book where the gate genuinely pays: −9.4pp of drawdown for −3.4pp of CAGR |
| Family B `b7` | `time:12` — 21.37% | `hard_stop:15` — 20.64%, Calmar 0.59 | **no exit beats holding.** Every family is within 1pp of `none` |
| Family B `b3` | no exit — 20.21% | `hard_stop:15` — 19.25%, Calmar 0.50 | same |

**The exit Arun is missing turns out not to be missing much.** Across 285 cells the best exit
buys at most a few tenths of a Calmar point on a thin book, and on the two books that can
actually stay invested (`ctrl`, `b7`) **no exit beats simply holding**. This is the same
finding r/71 reached from the other direction — a trailing stop beats a target, and a target
is worse than nothing.

**The index gate is not the rescue either.** On `ctrl` it is absent from the top six; on the
screened books it raises Calmar purely by parking the book in cash. That is a different
result from r/75, where the index-EMA gate was decisive — and the reason is visible in
`avg_pct_invested`: r/75's book was fully invested and this one is not.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-11 19:24 | G1b complete (34 cells) | `P_hasdata_N30` 24.08% / Calmar 0.57 — **the no-screen control at N=30 overtakes every screened arm on both metrics** |
| 2026-09-11 19:30 | **Paired test complete** — 12 masks, **0 pass** | §8.7 |
| 2026-09-11 19:33 | G2a launched, 285 cells, `arms='tax'` (wide scan) | |
| 2026-09-11 19:50 | **G2a complete** | §8.8. No exit beats holding on the investable books; the gate buys Calmar by holding cash |
| 2026-09-11 19:58 | G2b launched, 128 cells — slots 15/30/40/50, tv, cadence, buffer, k, liquidating gates, daily entry mechanics | the remaining book-construction axes |

### 8.9 G2b — book construction, and where the ceiling of this family is

128 more cells: slots 15/30/40/50, liquidity floor, cadence, hysteresis, k, liquidating gates
and the two daily entry mechanics.

**The best cell in the entire study carries no fundamental screen at all:**
`no screen · near-ATH k=0.90 · RS · 30 names · 200-SMA trail · tv ≥ ₹5 cr` —
**25.88% after tax, −40.88% (worst path −45.25%), Calmar 0.61, 95.5% invested, 149 trades/yr.**

The plateau around it is genuine, which is why it is quotable: N=30 tv₹2cr 24.35 · N=40 23.62 ·
N=50 23.62 · N=15 22.58 · tv ₹10cr 22.30 · k=0.85 23.66 · buffer 1.0 23.00 · `first_qualify`
22.12. **So ~22-26% is the ceiling of this family, and the screen is not part of it.**

Best per book: Family B `b7` 21.19% (Calmar 0.58) · Family A relaxed 17.12% (0.55) ·
**Family A as written 10.77% (0.40)**.

### 8.10 G3 — robustness on the six finalists (42 cells, `arms='all'`)

| book | full | W1 2018-08→2022-06 | W2 2022-07→2026-09 | 40 bps | 60 bps | **0% cash yield** | missing=pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Family A as written** | 10.77 | 8.79 *(20.5% inv)* | 12.43 | 10.32 | 9.73 | **7.74** | 12.42 |
| Family A best risk-adj. | 9.71 | 8.02 *(16.8% inv)* | 11.05 | 9.21 | 8.55 | **6.44** | 11.37 |
| Family A relaxed | 15.22 | 11.38 | 18.13 | 14.52 | 13.58 | 12.84 | 15.65 |
| Family B | 21.19 | 20.11 | 20.95 | 20.25 | 19.10 | 21.01 | 21.19 |
| no screen, N=30, ₹5cr | **25.88** | 23.56 | 25.27 | 24.11 | 22.26 | 25.19 | 25.88 |
| no screen, N=50 | 23.62 | 21.37 | 25.17 | 22.17 | 20.34 | 23.27 | 23.62 |

**The cash-yield column is the one that decides Family A.** At a 0% idle-cash assumption the
screen as written returns **7.74%** — three of its eleven points were the 5% yield on the 57%
of the book it could not fill. Family B and the unscreened control barely move, because they
are 91-96% invested.

Every book is positive in both sub-windows; every book survives the cost ladder; the
missing-data policy moves nothing material.

**Outlier dependence** (`results/g3_outliers.md`, trade level, one path):

| book | trades | mean/trade | top 5% of trades = | full ÷ ex-top-10 | ex-top-10 absolute |
|---|---:|---:|---:|---:|---|
| Family A as written | 213 | +4.36% | **85% of all trade return** | 309× | **0.32× — loses money** |
| Family A relaxed | 326 | +4.27% | 92% | 652× | 1.68× |
| Family B | 520 | +6.02% | 81% | 5,938× | 355× |
| no screen, N=30 | 1,265 | +6.22% | 96% | 167,230× | 1.8e9× |

### 8.11 G4 — portfolio fit: dilutive at every weight, and cash beats it on every path

360 paths (OA seed × TN offset, the r/154 convention), overlap window 2018-08 → 2026-08.

| added to the deployed TN+OA 50-50 pair | CAGR | MaxDD | Calmar | ΔCalmar | paths improved |
|---|---:|---:|---:|---:|---:|
| the pair alone | 33.68 | −14.01 | **2.369** | — | — |
| + Family B at 10% | 32.26 | −14.43 | 2.232 | −0.108 | 39/360 |
| + Family B at 20% | 30.99 | −15.64 | 2.017 | −0.355 | 16/360 |
| + Family B at 33% | 29.47 | −17.96 | 1.693 | −0.709 | **0/360** |
| **+ CASH at 20% (the null)** | 27.77 | −10.65 | **2.586** | **+0.212** | **360/360** |
| + CASH at 33% (the null) | 23.97 | −8.50 | 2.794 | +0.427 | 360/360 |

Monthly correlation: **Family B vs Open Alpha 0.624**, price-only version vs OA **0.730**,
Family B vs True North 0.374. This is not a complement — it is a weaker sampling of the family
Open Alpha already trades (OA alone on the same window: 44.80% / −25.76% / Calmar 1.78).

---

## 9. VERDICT

- **Family A — the screen exactly as Arun wrote it: NO EDGE.** 10.77% after tax, −29.1%,
  Calmar 0.40, 43% invested; below the Midcap 150 index; loses to random selection; loses
  −11.90pp on 12 of 12 paired offsets to the same book unscreened; 7.74% with no cash yield;
  loses money without its ten best trades.
- **Family B — what his own trades say he actually does: SIGNAL, NOT STRATEGY.** 21.19%,
  −37.1%, Calmar 0.58, stable in both windows and across the cost ladder — and it still misses
  the 25% bar, misses Calmar 1.0, fails the pre-registered paired test, and dilutes the live book.
- **The 25% bar is reachable only by deleting the screen**, and the book that does it is plain
  relative-strength momentum, which the live book already owns.
- **Nothing deployed, nothing papered.** One dated follow-up registered: quality as an *overlay
  inside Open Alpha's entries* (due 2026-10-10, blocked on `research/159_oa_honest_reoptimization`).

Full write-up: `results/RESULTS.md`. Published: `/app/backtest/quality-growth-near-ath-research160`.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-11 20:07 | **G2b complete** (128 cells) | best cell of the whole study carries NO screen: 25.88% / Calmar 0.61 |
| 2026-09-11 20:12 | **G3 complete** (6 finalists + 36 robustness cells, `arms='all'`) | the 0%-cash-yield column decides Family A: 10.77 → 7.74 |
| 2026-09-11 20:20 | Outlier test, YoY table, tearsheet, comparison chart | Family A ex-top-10 = 0.32× |
| 2026-09-11 20:35 | **G4 complete** — correlation + blend vs TN/OA | dilutive at every weight; cash wins 360/360 |
| 2026-09-11 20:45 | RESULTS.md written; study published to `/app/backtest/quality-growth-near-ath-research160`; frontend rebuilt on the VPS; page + both PNGs return **200** | no backend restart, no service touched |
| 2026-09-11 20:50 | `research/INDEX.md` row 160, `TODO.md` entry, 2 dated reviews in `ops_center.py` REVIEWS | **STATUS: DONE** |
