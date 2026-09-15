# Single-Stock Always-On Directional Trend System — SuperTrend / EMA-cross / MST master+child, futures-first, 324-name 5-min panel 2015-2026 + daily panel 2006-2026

**STATUS: DONE — verdict NO EDGE** — research/176 — opened and closed 2026-09-15

---

## 1. The Ask

**What Arun asked (near-verbatim, from TODO.md queued 2026-09-15):**

> "earlier i used to trade manually on maruti always on... using the mst system in our app... master
> super trend 7,5, child ST 7,2... with futures long/short on mst, options selling hedge on cst. the
> aim is to find some other system apart from nifty/indexes, im inclined towards directional... need
> not be by % move but even with options system to manage, so it need not be fully directionally
> trending... and it can be on a single stock / few stocks - like taking ALL signals instead of
> scanning for signals on stocks.... say we might end up finding supertrend 7,3 all signals works on
> reliance over a good run, or ema crossover on hdfcbank works with either futures or options or both
> or so.... take this up after current task is fully done"

**What we are actually testing.** Is there a **per-name, always-in, every-signal directional trend
system** on one or a few liquid F&O stocks that (a) beats simply **holding that same stock**, (b) is
not a single-name accident — the same rule and its parameter neighbours must work across a **basket**,
(c) survives a **window split**, and (d) **adds value to the live 45-DTE short-vol book** it is meant
to complement. Direction from SuperTrend, EMA crossover, or the MST master(7,5) + child(7,2) pair Arun
traded by hand on MARUTI. Expression: **futures first**; options only if the futures signal survives.

**What "always-on" means here (locked).** The book holds a position whenever the rule has a direction.
Two direction policies are tested as separate arms and never merged:

- **long/flat** — long when the rule is bull, in cash (5.2% post-tax idle yield, the house standard)
  when it is bear.
- **long/short** — long when bull, short when bear, flipping on every signal. This is Arun's literal
  "always on".

The short leg must **earn on its own** (positive net expectancy in isolation, both window halves) or
the long/short arm is dropped — research/81/82/83 killed every short-side equity signal tested here.

---

## 2. The Base — prior art this study must NOT rediscover

This idea has a near-exact precedent that **already failed**, and the study is designed around it.

| # | Study | What it found | What it means here |
|---|---|---|---|
| **48** | REC SuperTrend positional, always-on futures | **THE TRAP.** ST(7,3) always-on on RECLTD 15-min looked superb (+29-39% CAGR, Sharpe 1.1-1.4, every year positive, survived OOS/cost/lag). **Basket validation across 381 F&O names killed it**: beats buy-and-hold on **115/381 (30%)**, median Sharpe **−0.37**, median CAGR **−14.2%**, and on *rising* names only **21/196 (11%)**. Daily: **no** config beat B&H on 80 tested. Verdict NO ROBUST EDGE — lucky single-name overfit. | "ST 7,3 all signals works on RELIANCE" is exactly this shape. **Every cell in research/176 is reported basket-wide, never single-name.** The one thing r/48's basket kill could not rule out: its 15-min window was only **2.3 years (2024-03+)**, one regime. research/176's contribution is the same test over **11.5 years**. |
| **56** | NIFTY 30-min dual-SuperTrend (MST 14/5 + CST 14/2.5), options expression | Gross trend capture is real (Calmar 1.79 gross) but **turnover eats it**: 136 switches/yr, break-even about **10 bps per posture change**, and a multi-leg options round-trip costs more than that — the as-specced always-on credit book was **NO NET EDGE** (−Rs17k to −Rs62k / 6 wk). The only positive variant waited for a pullback inside the regime, i.e. **selectivity, the opposite of always-on**. Params unstable across instruments (NIFTY wants master mult 5, BANKNIFTY 3). | The MST master+child machine has been run once already, on the index, and died on **cost per flip**. research/176 must carry a **switches/year column in every table** and a cost ladder, and must treat parameter instability across names as a first-class result. |
| **134** | Directional diversifier for the short-vol book | The short-vol book's loss state is the **low-vol melt-up** (up-trend months: mean −1.19%, worst −9.27%; it has *never* lost in a NIFTY down-trend). The fix is **plain long equity**, and **trend timing on top of it actively hurt** — buy-and-hold beat every MA, Donchian and time-series-momentum variant tested. | Sets the **deciding metric** (blend vs the live 45-DTE book, not standalone CAGR) **and** the hardest null: if a trend rule cannot beat buy-and-hold of the same name, Arun should just hold the name, which r/134 already recommended. |
| **81 / 82 / 83** | Swing edge discovery; medium swing 5-15d; Turtle on F&O equities | **Every short-side equity signal lost**, at every horizon tested. Turtle N-sizing lost to equal notional (3rd sizing failure). | The long/short arm carries the burden of proof; long/flat is the prior. |
| **135** | Turtle optimisation | Optimisation was **subtractive** — dropping machinery beat adding it; the **plateau test caught the overfit**. | Plateau, not peak, in every sweep. Neighbours must agree. |
| **Maruthi live algo** | `memory/maruthi_algo_bugs.md` | The live MST algo was **DISABLED 2026-03-25 with 9 critical bugs**, incl. #9 *"SuperTrend computed on MARUTI spot but trading MARUTI FUT, about 60pt difference"*. | **No MST production code is reused.** The research engine is written fresh. Bug #9 is a modelling decision research/176 makes explicitly (section 3, Instrument). |

**Economic rationale (G0), stated honestly.** Trend-following on a single equity has a plausible
mechanism — under-reaction to news and flow persistence — but a weak one at single-stock level, where
idiosyncratic gap risk and mean reversion dominate. The counterparty is the liquidity provider and the
mean-reverter, both well capitalised. The prior from r/48 is that the mechanism does **not** clear
costs on the median name. research/176 is therefore run as a **falsification study**: the default
expected outcome is NO EDGE, and the bar is set before the run.

---

## 3. What is being tested — mechanics locked

**Universe.**

- **Daily arm:** names with daily bars from on or before 2006 and a 20-day median traded value at or
  above Rs 25 cr (a futures-liquidity proxy), ETFs excluded via `backtest_data/etf_exclusions.json`.
  Target about 150-250.
- **Intraday arm:** the **324 symbols carrying 5-minute bars 2015-02 to 2026** (verified 15-Sep-2026:
  382 symbols have 5-min, 324 span 2015 to 2026). 15/30/60-minute bars are **resampled from 5-minute**,
  which is why the intraday window is 11.5 years and not r/48's 2.3.
- **The three Arun named — MARUTI, RELIANCE, HDFCBANK — are reported individually in every table**,
  beside the basket distribution, so "does it work on RELIANCE" is answered *and* placed in context.

**Signals (all computed on the same bar series, all causal).**

| Family | Grid | Cells |
|---|---|---|
| SuperTrend, band-locked (TradingView convention) | period in {7, 10, 14, 21} x multiplier in {1.5, 2, 2.5, 3, 4, 5} | 24 |
| EMA crossover (fast/slow) | (5,20) (9,21) (10,30) (20,50) (21,55) (50,100) (50,200) | 7 |
| MST master+child (Arun's) | master ST(7, m) m in {4,5,6} x child ST(7, c) c in {1.5, 2, 2.5}. Master sets regime; child sets entries within it. **Arun's own cell = master 7,5 / child 7,2.** | 9 |
| **Total signal cells** | | **40** |

**Direction policy:** long/flat, long/short → x2.

**Timeframes:** daily, 60-min, 30-min (and 15-min for the survivors only) — the daily and intraday
sweeps are run as separate stages so the cheap one can kill the family first.

**Fill mechanic (enumerate, never assume).**

- **Primary, honest:** signal on bar close → fill at the **next bar's open**.
- **Reference, labeled as optimistic:** fill at the **signal bar's close** (this is the assumption that
  flattered r/48's 15-min numbers; r/48 showed 1-bar lag cut Sharpe 1.15 to 0.53).
- A **2-bar-lag** arm is run on survivors as the execution-fragility probe.

**Instrument and cost (futures, first phase).**

- Signals and P&L are computed on the **underlying cash series**, used as a proxy for the front-month
  future. This is Maruthi bug #9 and is stated as a limitation, not hidden: stock futures trade at a
  small basis to spot which decays to zero at expiry, so a spot-proxy **understates** the cost of an
  always-on book by omitting **roll slippage**. Modelled explicitly: **10 bps per round trip** (futures
  brokerage + STT on sell + exchange/SEBI/GST + 1 tick) **plus a monthly roll charge of 5 bps** for any
  position held across an expiry.
- **Cost ladder in every table: 10 / 20 / 40 bps per round trip.** A 30-min book flips about 130 times
  a year; at 40 bps that is 52% of notional a year in friction, and the ladder is what decides
  deployability (r/56 died exactly here).
- **Idle cash** in the long/flat arm earns **5.2% post-tax** (the house standard, memory
  `idle-cash-standard-5p2-arbitrage.md`).
- **Tax:** futures P&L is business income in India, not STCG. Reported **pre-tax** with that stated;
  the after-tax comparison against the equity books is made at the blend stage only.

**Nulls and controls (each cell carries all three).**

1. **Buy-and-hold of the same name, same window** — the primary null. r/134's finding makes this the
   question that matters: does timing beat holding?
2. **Time-in-market-matched random entry** — random long spells with the same count and the same
   run-length distribution as the rule produced on that name, 200 draws, reported as a percentile.
   This separates "the rule works" from "being long 70% of a bull decade works".
3. **Basket distribution** — the fraction of names on which the cell beats null 1 and null 2, with
   the median and the interquartile range, never a single name.

**Window split.** Daily: 2006-2015 / 2016-2026. Intraday: 2015-2020 / 2021-2026. **Both halves must
pass.** Plus a per-year table for the survivors.

---

## 4. The plan — stages, cell counts, and the PRE-REGISTERED bar

Ranking metric and adoption thresholds are fixed **now, before any cell runs**.

**Pre-registered primary metric:** `beat_rate` = the fraction of basket names on which the cell's
**net** CAGR exceeds that name's **buy-and-hold** CAGR over the same window, at **20 bps** round-trip.
Tie-break: median (cell Calmar minus B&H Calmar) across the basket.

**Pre-registered gates:**

| Gate | Threshold | If it fails |
|---|---|---|
| **G1a — is there a signal at all?** | Best cell's `beat_rate` at or above **55%** of basket names, in **both** window halves | NO EDGE, stop. (r/48's number was 30%.) |
| **G1b — plateau** | The 4 nearest parameter neighbours of the best cell average at or above **80%** of its beat_rate | Spike, not plateau → NO EDGE |
| **G1c — vs random** | Best cell beats its time-in-market-matched random null on at least **60%** of names | It is market exposure, not timing → NO EDGE |
| **G2 — cost** | Still clears G1a at **40 bps** round trip | SIGNAL at best, not deployable |
| **G3 — short leg** | Short-only P&L positive in both halves | Drop long/short; long/flat only |
| **G4 — blend** | Against the live 45-DTE book (research/119 monthly series, r/134 Stage A/B method): **+0.10 Calmar or −2pp drawdown at equal or better return**, and beats the **cash-null** at the same weight | SIGNAL, not a STRATEGY |
| **G5 — options expression** | Only opened if G1-G4 pass. Cites r/56's 10 bps/flip break-even as the bar to clear | — |

**Cell counts.**

| Stage | Grid | Cells |
|---|---|---|
| **0. Data integrity** | split-step scan + phantom-row scan over the panel | — |
| **1. Daily basket sweep** | 40 signals x 2 directions x 2 fills x about 200 names | about 32,000 name-cells |
| **2. Intraday basket sweep** (60-min, 30-min) | 40 x 2 x 2 fills x about 150 names x 2 TFs | about 48,000 name-cells |
| **3. Survivors only** | plateau map, window split, per-year, cost ladder, 2-bar lag, short-leg isolation | — |
| **4. MST stacking** | Arun's literal machine (accumulate to 5 lots, close-all-but-last on master flip) on the survivor names | — |
| **5. Blend vs the live 45-DTE book** | monthly returns, weight sweep 10-40% | — |
| **6. Options expression** | only if 1-5 survive | — |

Multiple-testing disclosure: **80 signal-by-direction combinations** are tested per timeframe. With
about 200 names the study is powered to see a basket-wide effect, but any *single-name* winner must be
discounted as one of about 16,000 draws — which is precisely the r/48 error this design exists to
prevent.

---

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-15 16:45 | Study opened, research/176 | 175 was the last taken number (dividend capture) |
| 2026-09-15 16:50 | Data recon complete | 5-min panel is **382 symbols, 324 spanning 2015-02 to 2026** — far broader than the 10 names in CLAUDE.md. Daily: 2,906 symbols. 60-min: 93 names but **ends 2025-11-07 (stale)**, so 60-min and 30-min bars will be **resampled from 5-min** instead. MARUTI/RELIANCE/HDFCBANK all have full daily + 5-min history. |
| 2026-09-15 16:55 | STATUS doc written (sections 1-4), pre-registered bar locked | Nothing launched yet |
| 2026-09-15 17:00 | Stage 0 done | 322 symbols span the 5-min panel; 153 clear the Rs 10 cr liquidity floor; **14 of 321 carry an unexplained overnight price step** (split-adjustment defect), 7 of them inside the liquid basket -> excluded from all headline figures |
| 2026-09-15 17:05 | Stage 1 (daily, 107,520 name-cells) DONE | **G1a FAILS.** Best long/flat cell EMA(9,21) beat_rate 0.432 (gate 0.55); h1 0.526, h2 0.329. long/short best 0.062, median CAGR -3.4%. short-only best 0.014. |
| 2026-09-15 17:12 | Stage 2 60-min (110,160 cells) DONE | Worse than daily: best 0.336, h2 0.233. Switch rates 9-101/yr. |
| 2026-09-15 17:20 | Stage 2 30-min (110,160 cells) DONE | Worse again: best 0.295, and beat_rate roughly halves at 40 bps. **Monotone decay with turnover** - the r/56 result. |
| 2026-09-15 17:25 | Stage 3 block-permutation nulls DONE (200 draws x 9 cells x 146 names, both timeframes) | **G1c PASSES for the fast EMA crossovers** - EMA(9,21) beats its own time-in-market-matched null on 69.9% of names for CAGR and 73.3% for Calmar. The timing skill is real; it buys drawdown, not return. |
| 2026-09-15 17:35 | Stage 4 book + blend DONE | **G4 FAILS.** Every directional sleeve is beaten by a plain CASH sleeve against the live short-vol book (+0.54 Calmar at 40% vs +0.14 for the best trend book), and by RELIANCE buy-and-hold (+0.41). Every blend cuts CAGR, so no cell clears the "at equal or better return" clause. |
| 2026-09-15 17:45 | Stage 5 MST lot-stacking DONE | The ramp is the best drawdown tool in the study (ARUN3 -14.8% vs B&H -54.7%) and the lowest-returning positive arm (6.86%, below NIFTY's 8.79%). A de-levering device. |
| 2026-09-15 17:55 | Stage 6 YoY + factsheet DONE; RESULTS.md written | **VERDICT: NO EDGE.** G5 (options expression) not opened - pre-registered as gated on G1-G4. |

## 6. Crash recovery

Everything runs on the VPS at `/home/arun/quantifyd`, python `venv/bin/python`.

- **What finished:** each stage writes one row per completed cell to
  `research/176_single_stock_always_on_trend/results/<stage>.csv`. `wc -l` that file.
  Runners **skip cells already present**, so re-launching the same command resumes.
- **Is it alive:** `ps aux | grep 176_` and `tail -f /tmp/r176_<stage>.log`.
- **Resume:** re-run the same launch command; nothing else is needed.
- **Do not touch:** `backtest_data/market_data.db` (read-only, `mode=ro` in every script), anything
  under `services/` (live executors), any `*_paper.json` / `*_state.json`.
- **Safe to inspect:** everything under `research/176_single_stock_always_on_trend/`.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `SINGLE_STOCK_ALWAYS_ON_TREND_MULTITF_SWEEP_STATUS.md` | this file | yes |
| `scripts/engine.py` | bar loading, band-locked SuperTrend, EMA cross, MST pair, always-on P&L, nulls | yes |
| `scripts/stage0_data_integrity.py` | split-step + phantom-row scan | yes |
| `scripts/stage1_daily_basket.py` | daily sweep runner | yes |
| `scripts/stage2_intraday_basket.py` | intraday sweep runner | yes |
| `results/*.csv` | per-cell output | yes if small |
| `results/RESULTS.md` | final verdict | yes |

## 8. Findings

**VERDICT: NO EDGE.** Full write-up: `results/RESULTS.md`. Published at
`/app/backtest/single-stock-alwayson-trend-research176`.

Gate by gate, against the bar locked before the run:

| Gate | Bar | Result | |
|---|---|---|---|
| **G1a** signal exists | beat_rate >= 0.55 both halves | best is **0.432** (daily), 0.336 (60-min), 0.295 (30-min); every h2 is worse than its h1 | **FAIL** |
| **G1b** plateau | neighbours >= 80% of best | the SuperTrend surface is a flat 0.15-0.34 plane - a uniform failure, not a spike | **FAIL (by being uniformly below the bar)** |
| **G1c** vs matched null | beats on >= 60% of names | EMA(9,21) beats its own block-permutation null on **69.9%** (CAGR) and **73.3%** (Calmar) | **PASS** |
| **G2** cost | clears G1a at 40 bps | 0.363 / 0.267 / 0.212 | **FAIL** |
| **G3** short leg | positive on its own | median expectancy per short trade negative on every family and timeframe; beat_rate 0.014-0.082 | **FAIL - drop the short leg** |
| **G4** blend | +0.10 Calmar or -2pp DD at equal return, and beats the cash-null | **plain cash beats every directional sleeve**; every blend cuts CAGR | **FAIL** |
| **G5** options | only if G1-G4 pass | not opened | **N/A** |

The one real finding: the trend rule genuinely times better than a time-in-market-matched random
shuffle, but what it earns is a **smaller drawdown**, not a higher return - and the risk edge
itself decays (Calmar-beat 0.659 in 2006-15, 0.521 in 2016-26). Best construction anywhere in the
study = ARUN3 equal-weight, daily EMA(9,21) long/flat: **13.88% CAGR, -28.0% drawdown, Calmar
0.495**, against buy-and-hold of the same three names at **17.71% / -54.7% / 0.324** and against
the deployed TN+OA pair at Calmar **1.68**.

On Arun's own names: **0 of 40 daily cells beat buy-and-hold on RELIANCE, 0 of 40 on HDFCBANK,
3 of 40 on MARUTI.**
