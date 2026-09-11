# Rounding-Base (Saucer) + Volume Accumulation → Rim Breakout — Causal Daily Screen Across 2,905 NSE Symbols

**STATUS: DONE for this stage — awaiting Arun’s manual chart verification. No backtest run, nothing deployed.**
**Stage gate:** G0 (hypothesis + implementability) → G1 probe is *identification only*. No sweep, no exits, no CAGR/Calmar. Nothing is deployed.
**Research number:** 159 · **Opened:** 11-Sep-2026 07:42 IST · **Owner:** quant-researcher agent
**Canonical copy:** the **VPS** (`/home/arun/quantifyd/research/159_rounding_base_breakout/`). The laptop folder
`C:\Users\arunc\Documents\Projects\Covered_Calls\research\159_rounding_base_breakout\` is **not** a git checkout
(`git rev-parse` → *not a repository*), so the commit lives on the VPS. Both copies hold the same files.

---

## 1. Headline

Arun described a chart pattern — a **rounding base / saucer** (a smooth, semi-circular price decline and
recovery at a base, rather than a sharp V), with **volume being accumulated** through the right-hand side of
the saucer, followed by a **breakout above the left rim**, then a **trailing exit**. He gave two live examples
(KMEW, ACCENTMIC-SM) and one explicit prohibition: the entry must **not** be identified backwards from a green
close. This task builds the **causal detector** and returns **2-3 examples for him to eyeball on a chart**
before any compute is spent on a real backtest.

---

## 2. The Ask

### What Arun asked (verbatim)

> "semi circle formation at base with volumes being accumated and then the breakout and then we trail with some
> mechanism... supertrend 7,3 works for accent, cuts kmew half way through... pls deploy agents using ur backtest
> instructions to assess and study if such stocks can be picked up, backtest it correctly, the entry process cannot
> be backwards based on a green close on a given day. pls see if u hv understood this, look for a couple of stocks
> with such patterns in the past and just give me the dates and stocks (2-3 options) i will manually verify them 1st"

### His two chart examples (from Kite daily charts with 20/50/200 MAs and an OBV panel)

| | KMEW | ACCENTMIC-SM |
|---|---|---|
| Listed | ~14-Nov-2024 (fresh IPO) | NSE **SME** board |
| Left rim (start of saucer) | ~₹1,200 | ~₹350 |
| Trough | ~₹800, around Mar/Apr-2025 | ~₹175, around Apr-2025 |
| Back to rim | ~₹1,200 by ~Aug-2025 | ~₹350 by ~Oct-2025 |
| Rim-to-rim duration | ~9 months | ~6 months |
| Breakout | ~Sep-2025 | ~Oct-2025 |
| Price now | ₹2,876 (+200% from rim) | ₹720 (+~105%) |
| SuperTrend(7,3) daily | **Stopped him out mid-move** ("cuts kmew half way through") | **Held the whole move** |

### What we are actually testing in THIS task

Across **every symbol with daily data in `market_data_unified`** (2,905 symbols, 03-Jan-2000 → 10-Sep-2026):

> Can a **strictly causal** rule — one that recognises the saucer and **fixes the rim level using only bars up to
> and including the day of recognition**, then enters on the **first close above that pre-fixed level** — find
> events of the kind Arun drew? How many are there, in which years, and do 2-3 of the highest **pattern-quality**
> ones look right to him on a chart?

**Explicitly out of scope for this task** (per Arun: "i will manually verify them 1st"): any exit sweep, any
portfolio construction, any CAGR / Sharpe / Calmar / drawdown figure, any app study page, any deployment.

### Success criterion for THIS task (pre-registered, 11-Sep-2026, before any detector code ran)

1. **Replication check** — the detector must fire on **KMEW** at approximately the Sep-2025 breakout. If it does
   not, the definition is wrong and we say so rather than shipping a different pattern.
2. **Frequency must be sane** — if the screen produces thousands of events per year it is not the rare, tradeable
   formation Arun drew, and the thresholds are too loose; if it produces a handful in 26 years it is untradeable.
   Both outcomes are reported honestly, and both change the design.
3. **Selection of the short-list must be blind to the outcome.** The ranking metric is a **pattern-quality score
   computed only from bars ≤ the breakout day**. Forward returns and the SuperTrend outcome are carried as
   **information columns** and are **forbidden as inputs to selection or ranking** — using them is precisely the
   hindsight Arun forbade.

### Economic hypothesis (G0 — why this should exist, and who is on the other side)

A rounding base is the visible fingerprint of a **change in the marginal holder**. The left rim is where the last
cohort of momentum/IPO buyers gave up; the smooth (not V-shaped) decline says the selling was **patient and
supply-driven** — distribution, lock-in expiry, disappointed allottees — rather than a panic. The long, flat
bottom with **rising OBV** says an accumulator is absorbing that supply *without* moving the price, which is what
a patient buyer with size actually does. By the time price returns to the rim, the willing sellers are exhausted,
so the rim — which was heavy resistance — clears on light effort, and the move that follows has little overhead
supply to fight. **Counterparty:** the trapped left-rim cohort selling back at breakeven, plus the bottom-fishers
who bought the trough and take profit at the rim. **Decay risk:** it is a well-known textbook pattern (O'Neil's
"saucer with handle", Bulkowski's "rounding bottom"); its edge, if any, likely lives in the small/mid-cap and
post-IPO corner where institutional attention is thin — which is exactly where both of Arun's examples sit, and
which is also where **capacity and our data quality are worst**.

---

## 3. The Base — the causal pattern definition (every threshold locked before running)

**Bar:** daily close, NSE cash. **Direction:** long only. **Causality rule:** on every day `t`, only bars with
date ≤ `t` may be touched. The rim level is fixed on the **base-qualify day `q`** and frozen; the entry is the
first close above that frozen level. At no point is a breakout day located first and a base fitted backwards to it.

### 3.0 Pre-processing (applied per symbol, before any statistic)

1. Load the symbol's full daily series ordered by date.
2. **Drop zero/NULL-volume rows** (Kite phantom holiday rows: O=H=L=C=prev close, volume 0 — 3,388 such rows in
   2026 alone, 18,151 in 2020). All rolling statistics are computed on the surviving series, per the brief's
   NaN-robustness rule.
3. **Ignore the final bar if it is a partial (intraday) candle.** Checked: the DB's max daily date is
   **10-Sep-2026** and today is 11-Sep-2026, so the last stored bar is a completed session. No row dropped on
   this account; the guard stays in the code.
4. Require ≥ **90** surviving bars for a symbol to be considered at all.

### 3.1 Window modes (all four evaluated; each event records which mode produced it)

| Mode | Window on day `t` | Left rim `R` |
|---|---|---|
| `L120` | the last **120** bars | highest **close** in the window's **first third** |
| `L180` | the last **180** bars | highest **close** in the window's **first third** |
| `L250` | the last **250** bars | highest **close** in the window's **first third** |
| `VAR` | from the bar of the **highest close in the last 250 bars** (the rim) up to `t` | that highest close |

`VAR` is the mode that matches Arun's "rim to rim" description most literally; it requires a window length
between **90** and **400** bars.

**IPO-age exception (needed for KMEW, declared rather than hidden):** when a symbol has fewer bars of history
than `L` (or fewer than 250 for `VAR`), the window starts at the **first listed bar** and the rim is the highest
close in the available history's first third (or the post-listing high for `VAR`). The window must still contain
≥ 90 bars. Every event carries a boolean **`ipo_short_window`** so these can be counted, inspected or excluded
separately. This exception must never silently widen the general rule.

### 3.2 Base shape tests — all evaluated on the window, using only bars ≤ `t`

| Test | Rule | Rationale / calibration |
|---|---|---|
| **Trough** | lowest close in the window, at fractional position **0.30 – 0.70** of the window | symmetry; a saucer troughs in the middle, a falling knife troughs at the end |
| **Depth** | `(R − trough)/R` between **20%** and **70%** | KMEW is **44.7%**, Arun's ACCENT read is ~50%. Below 20% is a flat range, above 70% is a collapse, not a base |
| **Roundness** | fit `log(close) = a·x² + b·x + c` over the window; require **a > 0** (upward-opening = bowl) and **R² ≥ 0.70** (a per-event flag records whether it also clears 0.60 / 0.80) | KMEW scores **R² 0.817** on the `VAR` window (rim→breakout) and 0.746 when the window is forced to start at the listing bar. **Correction to an earlier note in this doc:** I first wrote that an 0.80 floor would have rejected KMEW — that was based on the listing-anchored window only, and is **retracted**. KMEW clears 0.80 in the mode that actually fires it, so the 0.70 floor is a deliberately loose choice, not one tuned to admit the example |
| **Vertex** | fitted parabola's vertex inside the **middle third** (0.33 – 0.67) of the window | KMEW: vertex at **0.57** |
| **No-V test** (revised — see 3.10) | fraction of window bars closing below `trough + ⅓·(R − trough)` — the **bottom third of the base's vertical range** — must be ≥ **0.40** | a saucer *spends time* at the bottom; a V does not. Threshold derived from geometry, not from an example: a **linear V** spends exactly **0.333** of its bars below that line, an **ideal parabola 0.577**. 0.40 sits between them. KMEW scores **0.451** |
| **Right-side recovery** | close at `t` is within **5% below R** (i.e. `close ≥ 0.95·R`) — the first day this is true is the **base-qualify day `q`** | this is the moment the base is "complete"; **from `q`, `R` is frozen** |

### 3.3 Volume / OBV accumulation (reported always; used as an on/off filter variant)

OBV = cumulative signed volume, `Σ sign(Δclose) · volume`, computed on the de-phantomed series.

| Statistic | Definition |
|---|---|
| `obv_slope_right` | OLS slope of OBV over the **right half** of the base (trough → `q`), normalised by mean daily volume; require **> 0** |
| `obv_gain` | `(OBV(q) − OBV(trough)) / Σvolume(trough→q)` — the fraction of right-half volume that was net accumulation; require **> 0** |
| `vol_ratio` | median daily volume in the right half ÷ median in the left half; require **≥ 1.2** |

KMEW's right-half median volume is **3.6×** its left half (108,627 vs 30,396), so it clears `vol_ratio` easily.
**Filter variant `OBV_ON`** = all three conditions; `OBV_OFF` = none. Events are emitted for both and tagged.

### 3.4 Entry trigger and fill (the part the prohibition is about)

- **Signal:** the **first close strictly above the frozen `R`** on any day after `q`, within a **maximum wait of
  60 trading days**. If 60 bars pass with no close above `R`, the base **expires** and the symbol must re-qualify
  from scratch. If the close falls more than **15%** below `R` while waiting, the base is **voided** early
  (the recovery failed).
- **Fill (a) — next-day open:** buy at the open of the bar after the signal close.
- **Fill (b) — buy-stop at the rim:** a resting stop at `R`, filled at `max(R, next-day open)`.
  Both are reported for every event. research/142 showed a conclusion can flip entirely on this choice
  (×536 vs ×14.4), so neither is assumed.
- **Re-arming:** after an event fires (or a base expires/voids), that symbol is suppressed for **60 trading days**
  so one saucer does not emit a cluster of near-duplicate rows.

### 3.5 Liquidity and universe

- **20-day median traded value (close × volume) at `q` ≥ ₹2 crore.** This is deliberately **looser than our usual
  ₹5 cr** because both of Arun's examples are small caps — KMEW's 20-day median traded value was only **₹1.49 cr**
  at the trough and **₹5.68 cr** by the breakout. Stated as a known capacity risk, not waved away.
- **ETFs and index series excluded** by a name pattern (`*BEES*`, `*ETF*`, `*IETF*`, `NIFTY*`, `*GOLD*`, etc.).
- **Universe:** every symbol with daily rows in `market_data_unified` — **2,905 symbols**, 6,853,766 rows,
  **03-Jan-2000 → 10-Sep-2026**; 2,430 of them have ≥ 250 daily bars.

### 3.6 Split-artifact guard (mandatory — the DB is not retroactively split-adjusted)

`market_data.db` keeps pre-split rows at the **old price scale** (MCX, HEG, NAZARA, CUPID 5×, …). A split inside a
window manufactures a fake cliff that can imitate either the left edge of a saucer or a breakout. Therefore:

> **Reject any window containing a single-day close-to-close move < −35% or > +50%,** and **count** the rejections.

The cost is stated openly: **genuine saucers on symbols that split during the base are missed**, and we cannot tell
how many without a re-fetch.

### 3.7 Exit — information only, not swept in this task

For each event we record, **purely as an information column**, the outcome under **SuperTrend(7, 3)** on daily
closes (period 7, multiplier 3.0, close-based flip), plus raw **+60 / +120 / +250 trading-day** returns from each
fill price. The family to sweep *later* — only if Arun approves the pattern — is {SuperTrend(various), 20-day-low
Donchian close, 50-EMA close, ATR trail}. **None of these columns may touch the short-list ranking.**

### 3.8 Pattern-quality score (the ONLY ranking metric — pre-registered, causal by construction)

Computed at `q` from bars ≤ `q` only. No forward information of any kind enters it.

```
score = 0.30·fit      + 0.20·symmetry + 0.15·depth    + 0.20·accumulation + 0.15·flatness
fit          = clip((R2 − 0.70) / 0.25, 0, 1)
symmetry     = 1 − min(|trough_position − 0.50| / 0.20, 1)
depth        = 1 − min(|depth_pct − 0.375| / 0.25, 1)          # peaks at a 37.5% saucer
accumulation = 0.5·clip((vol_ratio − 1) / 2, 0, 1) + 0.5·clip(obv_gain / 0.5, 0, 1)
flatness     = clip((frac_bars_in_bottom_third_of_range − 0.40) / 0.18, 0, 1)   # 0.40 = V floor, 0.58 = ideal parabola
```

### 3.10 Deviation log — every change made after the spec was locked

The brief allowed refining the definition "only where the data forces it, and log every
deviation". Three changes were made after section 3 was locked; all three were forced by the
**KMEW replication check failing**, and all are recorded here rather than folded in silently.

| # | Change | Why | Effect |
|---|---|---|---|
| **D1** | **No-V test changed from price-relative to depth-relative.** Was: bars within **10% of the trough price** ≥ 15% of the window. Now: bars in the **bottom third of the base's vertical range** ≥ **0.40**. | The original test **rejected KMEW outright** — its only failing gate. A fixed 10%-of-price band is meaningless on a base whose depth is 45%: KMEW scored just **0.082**. The replacement is scale-free and its threshold is set from geometry (linear V = 0.333, parabola = 0.577), not from KMEW. | KMEW now scores **0.451** and fires. The new test is *stricter* on the rest of the population, not looser — it removed several marginal events from the smoke test, including two TATASTEEL rows whose "breakout" was a single +26% day |
| **D2** | **A breakout is now allowed on the base-qualify day itself** (`days_q_to_breakout = 0`). | In fixed-window modes the rim comes from the window's *first third*, so the close on day `q` can already exceed it. Suppressing that day was an artificial miss. Still strictly causal: `R` is fixed from old bars, `close[t]` is known at `t`, and the fill is the **next** bar. | Adds events; each is flagged by `days_q_to_breakout = 0` so they can be excluded. They **are** excluded from the short-list handed to Arun, which requires ≥ 1 bar of separation so the chart shows base-then-breakout |
| **D3** | **Split guard extended to the breakout bar.** | The original guard only covered bars up to `q`, so a split or bad print *on the breakout day* could manufacture the signal — visible as TATASTEEL "breaking out" on a +26% bar. | Breakout bars moving more than +50% in a day are rejected and counted (`bo_split_rejected`); every event also carries `breakout_day_move_pct` so the reader can judge |

No other threshold in section 3 was altered after locking.

### 3.9 Falsification plan (decided now, before seeing results)

The idea is **abandoned or redefined** at this stage if any of the following is true:
(a) the detector cannot fire on KMEW near Sep-2025 under any of the four window modes;
(b) the screen yields > ~500 events per year (it is then a generic recovery screen, not a rare formation);
(c) fewer than ~30 events exist across all 26 years (untestable, let alone tradeable);
(d) Arun looks at the short-listed charts and says they are not the pattern he means.

---

## 4. Plan — what is run, and what is deliberately *not*

| Phase | What | Cost |
|---|---|---|
| **P0** | Data-reality probe: coverage, KMEW/ACCENT presence, phantom rows, partial candle | done, ~1 min |
| **P1** | KMEW shape probe to calibrate thresholds against Arun's own example *before* locking them | done, ~1 min |
| **P2** | Write this STATUS doc (sections 1-4) | done, before any detector code |
| **P3** | Causal detector `detect_rounding_base.py`, per-symbol vectorised, over 2,905 symbols × 4 window modes | target < 10 min |
| **P4** | Event CSV + per-year counts + KMEW replication check | — |
| **P5** | Short-list of 2-3 examples ranked **only** by §3.8 pattern-quality, from different years and sizes | — |
| **P6** | Report to Arun; **stop**. No sweep until he verifies the charts | — |

**Grid for this task (identification only):** 4 window modes × 2 OBV filter states = **8 detector configurations**,
run over 2,905 symbols. All shape thresholds are held at their §3.2 defaults; the R² variants (0.60 / 0.80) are
recorded per event as flags rather than run as separate cells, so the sensitivity is readable without an 8× cost.

**Not run here, on purpose:** exit sweeps, portfolio construction, seed/offset ensembles, cost and tax
arithmetic, correlation and blend tests against True North / Open Alpha, any app study page. Those are G2+ and
are only earned if Arun confirms the pattern.

---

## 5. Status log

**Current phase:** P6 complete — short-list delivered, awaiting Arun’s manual chart verification.
**Nothing was swept, nothing published, no service touched.**

| Date/time (IST) | Event | Notes |
|---|---|---|
| 11-Sep-2026 07:35 | Task opened; brief + playbook read | G0/G1 identification only |
| 11-Sep-2026 07:38 | VPS reachable by key auth, `venv/bin/python` = 3.12.3, DB 31.3 GB | no `sqlite3` CLI on the VPS — all queries via python |
| 11-Sep-2026 07:40 | P0 coverage probe | 2,905 daily symbols, 6,853,766 rows, 03-Jan-2000 → 10-Sep-2026; 2,430 symbols ≥ 250 bars; last bar complete (no partial candle) |
| 11-Sep-2026 07:40 | **KMEW found** (451 bars); **ACCENTMIC-SM absent** (NSE SME board) | second example cannot be replicated |
| 11-Sep-2026 07:41 | P1 KMEW shape probe | rim 05-Dec-2024 @ Rs1,189.7; trough 07-Apr-2025 @ Rs657.6 (−44.7%); first close > rim **16-Sep-2025 @ Rs1,198.5** |
| 11-Sep-2026 07:42 | Thresholds locked; STATUS sections 1-4 written | **before** any detector code |
| 11-Sep-2026 07:55 | P3 detector written, smoke-tested on KMEW/RELIANCE/TATASTEEL | **KMEW produced ZERO events — replication FAILED** |
| 11-Sep-2026 08:00 | Root cause isolated: the **no-V flatness gate** was the single failing test (KMEW 0.082 vs a 0.15 floor) | a price-relative 10% band is meaningless on a 45%-deep base |
| 11-Sep-2026 08:05 | **D1/D2/D3 applied** (see 3.10). No-V test re-derived from geometry (linear V = 0.333, parabola = 0.577 → floor 0.40) | net **stricter**, not looser — it also removed two TATASTEEL events whose "breakout" was a single +26% bar |
| 11-Sep-2026 08:07 | Re-smoke: **KMEW fires 16-Sep-2025 in 3 of 4 modes**, VAR rim Rs1,189.70 | replication PASS, to the day and to the rupee |
| 11-Sep-2026 08:12 | Full run launched on VPS (`/tmp/r159.log`) over 2,492 symbols | ETFs / index series excluded from 2,686 |
| 11-Sep-2026 08:17 | **Run DONE in 305s** | 1,698 raw events; 2,049 bases qualified; 40 expired, 295 voided, **1,251 windows rejected by the split guard** |
| 11-Sep-2026 08:19 | Summary + blind short-list generated | 1,513 de-duplicated events; 837 (55%) pass OBV accumulation |
| 11-Sep-2026 08:22 | **Causality self-audit: 0 violations on 8 invariants** across all 1,698 events | rim < trough ≤ q ≤ breakout < fill; every event clears its own stated thresholds |
| 11-Sep-2026 08:25 | Results copied to laptop; committed on VPS; reported to Arun | **STATUS → DONE for this stage** |

### Live findings during the run

- The detector reproduces **KMEW to the day and to the rupee** — it is not merely "a" saucer
  detector, it recovers Arun’s own example.
- Event frequency lands in the intended rare-but-testable band: **63/year** overall, **35/year**
  once the OBV accumulation filter is applied, across a 2,492-symbol universe.
- **The fill mechanic is not decisive here**, unlike research/142. Within the clean pool the
  next-day open sits a median **+2.2%** above the frozen rim (p10 +0.2%, p90 +6.5%), so fill (a)
  and fill (b) should give similar answers. Across *all* events the median is +4.6% with a fat
  +32.5% p90 tail, driven entirely by the same-day-qualify cases (3.10 D2).
- **The year distribution is the first real warning** — see 8.2.

---

## 6. Crash recovery — how to resume without Claude

Everything below is runnable as-is. **Canonical copy is on the VPS.**

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/159_rounding_base_breakout
```

**Is anything still running?**
```bash
pgrep -af detect_rounding_base.py     # empty = nothing running
tail -40 /tmp/r159.log                # ends with "DONE in <n>s" on success
```

**What finished?**
```bash
wc -l results/rounding_base_events.csv   # 1699 lines = 1 header + 1,698 events
ls -la results/
```

**Re-run from scratch** (idempotent, overwrites the CSV, ~5 minutes):
```bash
cd /home/arun/quantifyd
setsid nohup venv/bin/python -u research/159_rounding_base_breakout/scripts/detect_rounding_base.py > /tmp/r159.log 2>&1 < /dev/null &
sleep 5; pgrep -af detect_rounding_base.py      # confirm the PID is alive
```
The CSV is written **incrementally and flushed per symbol**, so a kill mid-run leaves a valid
partial file. There is no resume flag — a full run is ~5 minutes, so just re-run it.

**Re-generate the summary and short-list from an existing CSV** (seconds, no DB access):
```bash
cd /home/arun/quantifyd
venv/bin/python research/159_rounding_base_breakout/scripts/summarise_events.py
```

**Re-check one symbol’s gates** (why it did or did not fire):
```bash
cd /home/arun/quantifyd
venv/bin/python research/159_rounding_base_breakout/scripts/debug_kmew.py
venv/bin/python research/159_rounding_base_breakout/scripts/detect_rounding_base.py --symbols=KMEW,DIXON
#   ^ writes results/smoke_events.csv; does NOT overwrite the main CSV
```

**Safe to inspect:** everything in `results/` and `scripts/`; the DB **read-only** (every script
opens it with `mode=ro`).
**Do NOT touch:** `backtest_data/market_data.db` in write mode; any `services/*` file; the
`quantifyd` service. This folder touches **no live engine**, and **no service was restarted**.

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `ROUNDING_BASE_BREAKOUT_DAILY_SCREEN_STATUS.md` | this document — sole crash-recovery source | yes |
| `scripts/probe_coverage.py` | P0 data-reality probe (read-only) | yes |
| `scripts/probe_kmew.py` | P1 KMEW shape probe used to calibrate thresholds | yes |
| `scripts/detect_rounding_base.py` | **the causal detector** | yes |
| `scripts/debug_kmew.py` | per-gate trace for one symbol (this found the D1 bug) | yes |
| `scripts/summarise_events.py` | per-year counts, KMEW check, blind short-list | yes |
| `results/rounding_base_events.csv` | **the event table** — 1,698 events × 42 columns (433 KB) | yes |
| `results/summary.txt` | rendered counts + short-list | yes |
| `results/smoke_events.csv` | smoke-test output (`--symbols=`) | no — transient |
| `/tmp/r159.log` (VPS) | run log | no — transient |

---

## 8. Findings

### 8.1 KMEW replication — **PASS**

| | Manual read from Arun’s chart | Detector (causal, `VAR` mode) |
|---|---|---|
| Left rim | ~Rs1,200, Dec-2024 | **Rs1,189.70 on 05-Dec-2024** |
| Trough | ~Rs800, Mar/Apr-2025 | **Rs657.65 on 07-Apr-2025** (depth −44.7%) |
| Base complete (`q`) | — | 15-Sep-2025 |
| Breakout | ~Sep-2025 | **16-Sep-2025**, close Rs1,198.50 |
| Fill (a) next-day open | — | Rs1,191.00 on 17-Sep-2025 |
| Fill (b) buy-stop at rim | — | Rs1,191.00 |
| Shape | — | R² 0.817, vertex 0.56, no-V 0.464, vol-ratio 1.53, OBV filter **pass** |

Fires in **3 of 4 window modes** (`VAR`, `L250`, `L180`) on the **same day**. It needed the
declared IPO-age exception (3.1): KMEW had only 208 bars of history at the breakout.

**[Information only — used nowhere in selection]** +250-bar return from the fill **+144.5%**;
SuperTrend(7,3) exited 23-Feb-2026 for **+30.3%** after 108 bars — which independently
**corroborates Arun’s own complaint** that ST(7,3) "cuts KMEW half way through".

### 8.2 Frequency — and the first real warning

**1,513 de-duplicated events across 793 symbols, 2003 → 2026 — a mean of 63/year**, or **35/year**
with the OBV accumulation filter on. Both falsification bounds (3.9b > 500/yr, 3.9c < 30 total)
are comfortably cleared.

But the distribution is **not uniform**, and this is the finding that should temper any enthusiasm:

| Year | Events | | Year | Events |
|---|---|---|---|---|
| 2023 | **215** | | 2022 | 149 |
| 2025 | **182** | | 2024 | 110 |
| 2009 | **160** | | 2017 | 96 |
| 2020 | 80 | | 2019 | 17 |
| 2021 | 70 | | 2011 | 10 |

**2009 alone — the bounce off the 2008 crash — produced 160 events**, more than the whole of
2010-2016 combined. A rounding base *is* by construction a fall-and-recover shape, so a market
that falls and recovers together stamps the pattern on hundreds of names at once. Much of this
population may therefore be **market beta wearing a saucer costume**. Nothing measured so far
distinguishes the two; only a **date-matched / random-entry null control** at G1 proper can, and
the playbook makes that mandatory (r/87-88 killed two screens in exactly this way).

### 8.3 Causality self-audit — 0 violations

All 1,698 events were re-checked against eight invariants: `rim_date < trough_date ≤
base_qualify_date ≤ breakout_date < fill_date`, `breakout_close > rim_level`, and each event
clearing its own stated depth / R² / no-V thresholds. **Zero violations.** 46% of events carry
`days_q_to_breakout = 0` (3.10 D2) and are **excluded** from the short-list handed to Arun.

### 8.4 Short-list handed to Arun

Chosen **only** by the 3.8 pattern-quality score — one per calendar year and one per liquidity
bucket, drawn from the 318-event clean pool (OBV pass, ≥ 1 bar of separation, real fill,
breakout before Sep-2025). Forward returns and SuperTrend outcomes were computed but **never
consulted**. Full detail in `results/summary.txt`.

| # | Symbol | Left rim | Trough | Breakout | Score |
|---|---|---|---|---|---|
| 1 | **CENTURYPLY** | 19-Oct-2016 @ Rs262.55 | 22-Dec-2016 @ Rs155.10 (−40.9%) | **11-Apr-2017** | 0.823 |
| 2 | **SKFINDIA** | 11-Dec-2024 @ Rs2,336.50 | 13-Mar-2025 @ Rs1,705.90 (−27.0%) | **03-Jul-2025** | 0.790 |
| 3 | **DIXON** | 08-Dec-2022 @ Rs4,164.25 | 22-Feb-2023 @ Rs2,644.20 (−36.5%) | **12-Jun-2023** | 0.682 |
| 4 | **SRF** (addendum, long base) | 08-Sep-2008 @ Rs27.65 | 03-Feb-2009 @ Rs12.70 (−54.1%) | **03-Jun-2009** | 0.754 |

**That the blind ranking produced one big winner, one flat and one clear loser is the point** —
it is evidence the selection really was blind. Had all three been winners, the selection would
itself be suspect.

### 8.5 Caveats that travel with any of these numbers

1. **Survivorship.** The universe is the symbols present in the DB *today*. Delisted names never
   appear — and the bias is unusually sharp for a pattern that *requires* a recovery: the
   companies whose saucer never completed are precisely the ones that went away.
2. **Split artifacts.** The DB is not retroactively split-adjusted. The guard (3.6) rejected
   **1,251 windows** — a large number. It removes false saucers, but it also **discards genuine
   ones on names that split during the base**, and we cannot say how many without a re-fetch.
3. **ACCENTMIC-SM is absent** (NSE SME board; our universe is the main board), so Arun’s second
   example — the one where SuperTrend(7,3) *held* — **cannot be replicated here at all**.
4. **Liquidity floor loosened to Rs2 cr** (vs our usual Rs5 cr) to keep small caps like KMEW in
   scope. Any strategy from this family will meet a **capacity wall**; two of the four
   short-listed events sit under Rs10 cr of 20-day median traded value.
5. **Nothing here is a return estimate.** No costs, no taxes, no slots, no sizing, no market gate,
   no seed or offset ensemble. The `info_*` columns are descriptive only.
6. **Regime loading is unmeasured** (8.2). Until a date-matched null control runs, the honest
   status of this family is *"a real, reproducible chart pattern of unknown value"*.
7. **One degree of freedom has been spent.** The no-V threshold was re-derived after KMEW failed
   (3.10 D1). It was re-derived from geometry rather than fitted to KMEW, and the replacement is
   net stricter — but it is still a choice made after seeing the first attempt fail, and the G1
   multiple-testing haircut must account for it.

### 8.6 What happens next — only on Arun’s word

If he confirms the charts show the pattern he means, the G1-proper sequence is:
**(a)** date-matched and random-entry null controls (does the saucer beat a coin flip on the same
dates, in the same names?); **(b)** the exit bake-off he asked about — SuperTrend(7,3) against
other ST settings, 20-day-low Donchian, 50-EMA and an ATR trail, swept **jointly** with the
entry, never in isolation; **(c)** costs at 25 / 40 / 60 bps and after-tax; **(d)** portfolio
construction with a seed ensemble; **(e)** correlation and blend value against True North and
Open Alpha. **None of that has been run.**
