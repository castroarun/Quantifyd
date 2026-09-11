# New-ATH-Close Breakout — Does the AGE of the Previous High, and Volume Confirmation, Add Anything?

**STATUS: COMPLETE — verdict STRATEGY (candidate). Pre-registration (sections 1-5) was written before any cell ran.**
**Stage gate:** G1 (does a tradeable edge exist in our data, net of cost and tax) on a family we
already know is live-adjacent. **Nothing will be deployed from this study.**
**Research number:** 161 · **Opened:** 11-Sep-2026 18:55 IST · **Owner:** quant-researcher agent
**Canonical copy:** the **VPS** (`/home/arun/quantifyd/research/161_ath_base_age_breakout/`).
The laptop folder is not a git checkout, so the commit lives on the VPS.

---

## 1. Headline

Arun, after seeing research/159 land on **SIGNAL, not STRATEGY**, asked the obvious next
question: *forget the saucer — what if we just take all-time-high closes, and then ask whether
it matters **how long ago** the previous high was set, and whether **volume** has to confirm?*
This study sweeps exactly those two axes, which no study in this repo has swept before, against
an **honest** in-engine plain-ATH baseline.

---

## 2. The Ask

### What Arun asked (verbatim)

> "how can we improve from here on? looking at only ATH closes? ATH closes where the last ATH is
> at least x candles before? Breakout with volumes with or without last ATH being x candles away?"

### What we are actually testing

Across every NSE daily symbol in `market_data_unified`, with strictly causal signals and honest
next-open fills on **both** legs:

> Take the **first close above the prior all-time-high close**. Does requiring the previous ATH to
> be at least **X trading bars old** improve the book? Does requiring a **volume thrust of K×** the
> prior 20-bar median improve it? Do they improve it **together**? And does either of them improve
> it enough to be worth adding to Open Alpha’s entry rule?

**The question behind the question.** X is a proxy for *how much overhead supply has been worked
off and how much boredom has set in* before the breakout. X = 0 means "any new high, including the
fiftieth in a row during a melt-up"; X = 250 means "the stock has gone a year without making a new
high and has just made one". These are very different trades and Arun is right that nobody here has
separated them.

---

## 3. Two facts that change the framing — both must be stated up front

### 3.1 Open Alpha’s published numbers rest on a same-bar look-ahead fill

The parallel study `research/159_oa_honest_reoptimization` (a different session, same evening)
established that Open Alpha’s published **~34.9% CAGR**, and research/142’s **40.8%**, come
from a fill that is **not placeable**: the signal is `close > pivot` and the fill is
`max(pivot, open)` **on that same bar** — the entry price is taken from the very bar whose close
created the signal. Their own sweep labels that cell `placeable: NO`
(`open_same_REFERENCE`, 43.97% CAGR). Their **placeable** short-trail cells come out **negative**
(−2.25% to −2.54% CAGR); their honest long-trail cells reach roughly **18-23%** (stage A2
`close_same` trail-75: 23.20% CAGR, −45.4% DD, Calmar 0.502, 39.3% win rate; their stage B
figures are **pre-tax**, and their study was still writing files at 18:54, so these are quoted as
**theirs, provisional**).

**Consequences, both acted on:**

1. **A dated addendum has been added to research/159’s RESULTS.md and STATUS.** Its blend test
   compared an honest after-tax book against the research/154 Open Alpha curve, which inherits
   that look-ahead — an honest book against an inflated one. The "dilutes Open Alpha at every
   weight" limb is **weakened, not withdrawn**. Its verdict is unaffected: that rested on the
   **20% floor** (0 of 2,016 cells reached it) and the **pre-2016 window**, neither of which
   involves Open Alpha.
2. **This study does NOT use the research/154 Open Alpha curve as a benchmark.** The OA
   comparator here is an **in-engine cell** run through the same book with the same honest fills:
   *plain new-ATH-close breakout, X = 0, no volume filter, OA’s exits (15-SMA close trail plus a
   −8% close stop), liquidity ≥ ₹5 cr.* Same engine, same costs, same tax, same seeds — the only
   fair way to answer "does X or K improve OA’s entry".

### 3.2 What the parallel study does NOT sweep

Their grid is **entry mechanic × trail × stop**, then **slots × RS × gate**. It does **not** sweep
**base age** or **volume confirmation**. Those two are precisely this study’s axes. Beyond the
six exits listed in §4, this study deliberately does **not** re-sweep their axes.

---

## 4. The Base — the system being tested (every threshold locked before running)

**Bar:** daily close, NSE cash. **Direction:** long only. **Causality:** on day `t` only bars dated
≤ `t` are used; every rolling statistic is shifted back one bar.

### 4.1 Pre-processing (identical to research/159, so the two are comparable)

1. Drop zero/NULL-volume rows (Kite phantom holiday rows).
2. **Split guard on history:** `market_data.db` is not retroactively split-adjusted, so a pre-split
   row sits at the old price scale and would fake an unreachable ATH. If the close series contains
   any day-over-day move < **−35%**, the series is truncated to the bars **after the last such
   move**, and the ATH is computed only from those. `hist_bars` and `split_cut` are recorded.
3. Ignore a final partial (intraday) candle. Require ≥ 90 surviving bars.
4. ETFs and index series excluded by name pattern. **SILLYMONKS dropped** (identical series to
   CRESTO — the only true duplicate in the universe, established in research/159).

### 4.2 The trigger

| Component | Rule |
|---|---|
| **Signal** | `close[t] > ATH_prev[t]`, where `ATH_prev` is the running maximum of closes **strictly before** `t` — a genuine **new all-time-high close** |
| **Base age X** | trading bars between the bar that set `ATH_prev` and `t`. Swept: **0 (any), 20, 40, 60, 120, 250** |
| **Base depth** | maximum drawdown from `ATH_prev` reached **inside that gap**. Swept: **any, ≥ 10%, ≥ 20%** |
| **Volume K** | `volume[t] ≥ K ×` the median of the prior 20 bars. Swept: **none, 2×, 3×, 5×** |
| **Saucer shape** | the research/159 v3 base recognition (positive log-quadratic curvature, R² ≥ 0.70, trough centred 0.30-0.70, depth-relative no-V ≥ 0.40, depth 20-70%) required: **off / on** |
| **Gap-through flag** | recorded per event: was `open[t+1] > ATH_prev[t]`, i.e. did the fill gap through the level |
| **Fill** | **next-day open**, both legs |
| **Liquidity** | 20-day median traded value ≥ **₹2 cr** default; the winning region is **re-run at ₹5 cr** for comparability with Open Alpha |
| **Re-arm** | 60 bars after an event fires, so one run of new highs does not emit a cluster |

### 4.3 Exits (swept jointly with the entry, never in isolation)

`ST(14,4)` no stop · `ST(14,4)` + −8% close stop · `ST(10,3)` no stop ·
**`15-SMA close trail` + −8% close stop (Open Alpha’s own)** · `Donchian-20 close` · `50-EMA close`.

### 4.4 The book

Unchanged from research/159 so the two studies are directly comparable: **16 slots at 6.25% of
NAV, ₹10L, NSE cash CNC**, seeded random draw for slot contention, **idle cash 5.5% p.a.**,
**after tax** (20% STCG / 12.5% LTCG above 365 days, Indian FY loss-netting with carry-forward),
costs **25 / 40 / 60 bps per side**.

---

## 5. Plan and cell count (disclosed up front)

**6 (X) × 3 (depth) × 4 (K) × 2 (saucer) × 6 (exit) = 864 cells**, each on a **10-seed** scan,
with the plateau winners re-run on **30 seeds**. The winning region is then re-run at the ₹5 cr
liquidity floor (a further small set, disclosed when it runs). The multiple-testing haircut applies
to **864**; any winner is reported with its neighbourhood along X and K, and a winner whose
neighbours disagree is treated as noise.

**Pre-registered ranking metric:** **after-tax net CAGR**, Calmar as tie-break.

### 5.1 The pre-registered adoption bar

**STRATEGY** requires **all** of:

1. **≥ 20% after-tax net CAGR** (30-seed median; the **worst seed** is stated beside it), **and**
2. beats **NIFTYBEES buy-and-hold** on **CAGR and drawdown** in **both** windows (pre-2016, 2016+), **and**
3. beats the **honest in-engine plain-ATH OA-proxy cell** (§3.1), **and**
4. beats a **date-matched random-entry control**, **and**
5. the winning **X / K sits on a plateau**, not a spike.

**Below 20% but beating the OA proxy → "SIGNAL / improves the OA entry"**, and the study must then
say **plainly whether base age or volume should be added to Open Alpha’s entry rule**. Failing
the OA proxy as well → **NO EDGE**.

### 5.2 Reporting requirements Arun asked for by name

- **Win rate in every table**, alongside average win, average loss, expectancy per trade, max
  losing streak and trades per year.
- **CAGR with and without the 5.5% idle-cash yield** for the headline cells, so the equity
  contribution is visible separately from the cash carry.
- A dedicated **X (rows) × K (columns)** table of CAGR / WR / expectancy at the best exit,
  **with and without** the saucer requirement — this is the direct answer to his question.
- Outlier dependence (top-10 trades removed; winners capped), two windows, cost ladder, seed band.

### 5.3 Falsification plan

The idea is **killed** if no cell beats the honest plain-ATH OA-proxy cell; or if the X and K axes
are **flat** (in which case the honest answer is "age and volume add nothing, do not complicate the
entry rule"); or if the winner is a spike whose neighbours disagree.

---

## 6. Status log

| Time (IST) | Event | Notes |
|---|---|---|
| 11-Sep-2026 18:55 | Folder created; **161 verified free** on the VPS (159 is double-claimed, 160 taken) | market **closed** (18:55 Fri), load 2.91 |
| 11-Sep-2026 18:58 | Parallel study’s `stageA/A2/B.csv` read; the look-ahead finding confirmed from their own `placeable: NO` flag | dated addendum written into research/159 RESULTS.md and STATUS |
| 11-Sep-2026 19:00 | **This document written — sections 1-5, including the adoption bar and the 864-cell count — BEFORE any code ran** | |

---

## 7. Crash recovery

Canonical copy is the VPS. Everything below is runnable as-is.

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd
pgrep -af "ath_events|bt161" || echo "nothing running"
tail -40 /tmp/r161_events.log /tmp/r161_sweep_0.log
```
Detector re-run (idempotent, overwrites its CSV):
```bash
setsid nohup nice -n 10 venv/bin/python -u \
  research/161_ath_base_age_breakout/scripts/ath_events.py \
  > /tmp/r161_events.log 2>&1 < /dev/null &
```
Sweep (3 shards, resumable — each skips cells already in its own CSV):
```bash
for i in 0 1 2; do setsid nohup nice -n 10 venv/bin/python -u \
  research/161_ath_base_age_breakout/scripts/bt161_sweep.py --shard=$i/3 \
  > /tmp/r161_sweep_$i.log 2>&1 < /dev/null & done
```
**Do NOT touch** `backtest_data/market_data.db` in write mode, any `services/*` file, or the
`quantifyd` service. This study touches **no live engine** and restarts **no service**.

---

## 8. Findings — VERDICT: **STRATEGY (candidate)**

All five pre-registered criteria pass. Full write-up in `results/RESULTS.md`; published at
`/app/backtest/ath-base-age-breakout-research161`.

| # | Criterion | Result | |
|---|---|---|---|
| 1 | ≥ 20% after-tax CAGR (30-seed median) | **21.26%** (worst seed **19.87%**) | PASS |
| 2 | Beats NIFTYBEES on CAGR **and** DD in **both** windows | pre-2016 19.33% / −32.45% vs 12.68% / −59.71%; 2016+ 23.24% / −32.43% vs 11.88% / −36.34% | PASS |
| 3 | Beats the honest in-engine OA proxy | 21.26% vs **6.81%** | PASS |
| 4 | Beats a date-matched random control | 21.26% vs **12.11%** | PASS |
| 5 | Winning X on a plateau | X=40 **21.23%** ≈ X=60 **21.26%** | PASS |

### The three answers to Arun

1. **Base age X: YES.** X ≥ 60 bars **with** depth ≥ 20% lifts the book **18.66% → 21.26%**
   and cuts drawdown **−41.59% → −34.80%** (Calmar 0.449 → 0.618).
2. **Volume K: NO.** It raises expectancy per trade (+12.37% → +15.12% at X=250) and makes the
   **book** worse (CAGR → 20.51%, drawdown → −44.14%), because K ≥ 5× discards **60%** of events.
3. **Saucer (research/159 shape): NO.** 5.6 trades/yr, 5.90% CAGR on the identical engine.

### The finding neither of us went looking for

**The exit is worth more than both entry axes combined.** The same plain-ATH entries score
**6.81%** with Open Alpha’s own exits (15-SMA trail + −8% close stop) and **18.66%** with a
SuperTrend(14,4) trail — **+11.85pp from the exit alone**, against **+2.60pp** for the whole
base-age/depth filter. Reported prominently so it is not mistaken for an age result.

### Honest limits
Worst seed **19.87%** is 0.13pp under the floor (the bar is a median with the worst stated);
outlier ratio **133,000×** on removing the top-10 of 687 trades; **−34.8%** drawdown with a
**14-trade** losing streak; and the OA comparator is an **in-engine proxy** resting on a
parallel study that was still running. The portfolio-fit test is deferred and registered in the
ops centre for **26-Sep-2026**.


## 6.1 Run log

| Time (IST) | Event | Notes |
|---|---|---|
| 19:00 | Detector `ath_events.py` written and smoke-tested | CENTURYPLY 09-Jun-2014 shows the shape of the question: prior ATH from **19-Dec-2007**, X = **1,591 bars**, depth **69.5%**, 6.5× volume |
| 19:01 | **Full detection DONE in 39 s — 82,848 new-ATH-close events** across 1,698 liquid symbols | one row per (symbol, new-ATH day), no re-arm and no filtering applied yet |
| 19:02 | Panel built (1,698 symbols, 55 s, cached); 81,629 events land on the calendar | |
| 19:02 | **First 6 cells (plain ATH, no volume filter, ST(14,4))** | see 6.2 — the headline is already visible |
| 19:02 | 3 sweep shards launched `nice -n 10`, load 3.80 | 864 cells × 10 seeds, ~1.8 s/cell → ~9 min |

## 6.2 First finding — the plain-ATH book is far healthier than research/159’s saucer book

Same engine, same fills, same costs, same tax, same 16 slots:

| X (bars since prior ATH) | events | CAGR | MaxDD | Calmar | WR | expectancy/trade | trades/yr |
|---|---|---|---|---|---|---|---|
| 0 (any new ATH) | 10,293 | 18.88% | −41.59% | 0.454 | 47.1% | +10.25% | 37.5 |
| 20 | 7,359 | 18.07% | −35.09% | 0.508 | 45.5% | +9.62% | 37.3 |
| **40** | 5,776 | **20.68%** | −37.11% | 0.558 | 47.8% | **+11.53%** | 35.0 |
| 60 | 4,651 | 18.85% | −32.28% | **0.585** | **49.3%** | +10.20% | 34.3 |
| 120 | 3,000 | 18.03% | −30.97% | 0.582 | 46.3% | +10.37% | 31.9 |
| 250 | 1,806 | 19.02% | −33.05% | 0.576 | 47.0% | **+12.37%** | 25.8 |

Two things stand out immediately, both directly answering Arun’s question:

1. **The binding constraint in research/159 is gone.** That study had 889 events and a book
   ~40% invested; this has **10,293** at X = 0 and **1,806** even at X = 250. The plain-ATH
   book earns **18-21%** where the saucer book earned 14.60%, on the same engine. The saucer
   requirement was **costing** return, not adding it.
2. **X earns its keep through drawdown, not through return.** CAGR is roughly flat across X
   (18.0-20.7%), but **max drawdown improves monotonically** from −41.6% to −31.0%, so
   **Calmar rises 0.454 → 0.585**, and **expectancy per trade rises 10.25% → 12.37%** as
   trades/yr falls 37.5 → 25.8. That is the signature of a filter that removes marginal
   trades rather than one that finds better ones.

These are 10-seed scan numbers at the ₹2 cr floor with one exit; the 30-seed decision cells,
the volume axis, the saucer interaction and the OA-proxy comparison follow.
