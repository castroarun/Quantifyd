# Rounding-Base (Saucer) + Volume Accumulation → Rim Breakout — Causal Daily Screen Across 2,905 NSE Symbols

**STATUS: v3 screen DONE — all 9 expected checks pass. Backtest phase PRE-REGISTERED (section 11) and starting.**
**v1 = sections 1-8, v2 = section 9, v3 = section 10, backtest plan = section 11. v1 and v2 are superseded but kept in full for the record.**
**Stage gate:** G0 (hypothesis + implementability) → G1 probe is *identification only*. No sweep, no exits, no CAGR/Calmar. Nothing is deployed.
**Research number:** 159 · **Opened:** 11-Sep-2026 12:45 IST · **Owner:** quant-researcher agent
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
| 11-Sep-2026 12:45 | Task opened; brief + playbook read | G0/G1 identification only |
| 11-Sep-2026 12:48 | VPS reachable by key auth, `venv/bin/python` = 3.12.3, DB 31.3 GB | no `sqlite3` CLI on the VPS — all queries via python |
| 11-Sep-2026 12:50 | P0 coverage probe | 2,905 daily symbols, 6,853,766 rows, 03-Jan-2000 → 10-Sep-2026; 2,430 symbols ≥ 250 bars; last bar complete (no partial candle) |
| 11-Sep-2026 12:50 | **KMEW found** (451 bars); **ACCENTMIC-SM absent** (NSE SME board) | second example cannot be replicated |
| 11-Sep-2026 12:52 | P1 KMEW shape probe | rim 05-Dec-2024 @ Rs1,189.7; trough 07-Apr-2025 @ Rs657.6 (−44.7%); first close > rim **16-Sep-2025 @ Rs1,198.5** |
| 11-Sep-2026 12:55 | Thresholds locked; STATUS sections 1-4 written | **before** any detector code |
| 11-Sep-2026 13:08 | P3 detector written, smoke-tested on KMEW/RELIANCE/TATASTEEL | **KMEW produced ZERO events — replication FAILED** |
| 11-Sep-2026 13:12 | Root cause isolated: the **no-V flatness gate** was the single failing test (KMEW 0.082 vs a 0.15 floor) | a price-relative 10% band is meaningless on a 45%-deep base |
| 11-Sep-2026 13:16 | **D1/D2/D3 applied** (see 3.10). No-V test re-derived from geometry (linear V = 0.333, parabola = 0.577 → floor 0.40) | net **stricter**, not looser — it also removed two TATASTEEL events whose "breakout" was a single +26% bar |
| 11-Sep-2026 13:18 | Re-smoke: **KMEW fires 16-Sep-2025 in 3 of 4 modes**, VAR rim Rs1,189.70 | replication PASS, to the day and to the rupee |
| 11-Sep-2026 13:21 | Full run launched on VPS (`/tmp/r159.log`) over 2,492 symbols | ETFs / index series excluded from 2,686 |
| 11-Sep-2026 13:26 | **Run DONE in 305s** | 1,698 raw events; 2,049 bases qualified; 40 expired, 295 voided, **1,251 windows rejected by the split guard** |
| 11-Sep-2026 13:28 | Summary + blind short-list generated | 1,513 de-duplicated events; 837 (55%) pass OBV accumulation |
| 11-Sep-2026 13:30 | **Causality self-audit: 0 violations on 8 invariants** across all 1,698 events | rim < trough ≤ q ≤ breakout < fill; every event clears its own stated thresholds |
| 11-Sep-2026 13:33 | Results copied to laptop; committed on VPS; reported to Arun | **STATUS → DONE for this stage** |

### Two operational notes recorded during this task

**1. Clock discrepancy — all times in this doc are VPS (authoritative IST).**
The Windows laptop reports a time ~5h50m behind the VPS: a `TZ=Asia/Kolkata date` run locally
returned 07:42 IST while the VPS returned 13:32 IST for the same moment. The timestamps in the
log above were **corrected to the VPS clock** from file mtimes and journal entries. **Anyone
checking the market-hours rule must take the time from the VPS, never from the laptop** — a
laptop reading of "07:42" would wrongly suggest the market was closed when it was in fact
mid-session.

**2. `quantifyd` restarted at 13:28:50 IST — not by this task.**
Observed while verifying that this research had touched nothing: the gunicorn main PID changed
(3660583 → 4060241) at **13:28:50 IST on a trading day**, which is **before the binding 15:40
cutoff**. Evidence that it was not this task: **no `systemctl`, `sudo`, `restart` or deploy
command was issued anywhere in this session** — the work was read-only DB queries, `scp` into
`research/159_*`, and a `git add`/`git commit`, none of which can restart a unit.
`NRestarts=0` means systemd did not auto-restart it either, so it was an explicit external
restart; the only restart cron on the box is `preopen_restart.sh` at 09:00 Mon-Fri, which does
not match. The service came back healthy — scheduler up at 13:28:59, NAS ticker reconnected at
13:29:03, and ST(7,2) monitoring resumed on `NIFTY2691523300PE` at 13:29:04, a ~14-second gap.
**Flagged for Arun**, since a live short-option monitor was in memory at the time.

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


---

# 9. v2 — entry redefined after Arun’s manual check (11-Sep-2026)

## 9.1 What he rejected, and why he is right

Arun checked the v1 short-list against his own charts and **rejected the entry definition**
on SKFINDIA. v1 entered **03-Jul-2025 at Rs2,352**. The real breakout was **16-May-2025 at
Rs2,144.70** — a gap-up on **9.5x** the 20-day median volume that cleared the flat Mar-May
shelf at Rs1,750-1,860. **v1 entered +15% into the move.**

The diagnosis is a conceptual error in v1, not a threshold problem:

> **v1 triggered on the LEFT RIM — the far left lip of the decline. That is the wrong level.
> The breakout Arun trades is out of the BASE CEILING: the top of the consolidation at the
> BOTTOM of the saucer.**

The left rim is not the trigger. It is the **first overhead supply** the trade has to chew
through *after* entry. v1 waited for price to climb all the way back to it and only then
bought — by construction, always late. v2 keeps the left rim purely as an information column
(`dist_to_left_rim_pct`) measuring how much overhead supply still sits above the entry.

The evidence that this is the real defect, not a preference: **81% of v2 entries sit below
the old v1 left rim, a median 14.9% below it.** v2 buys the base; v1 bought the recovery.

## 9.2 The v2 spec (changes from v1 only — everything unlisted is unchanged)

**Kept exactly:** quadratic curvature > 0, R² ≥ 0.70, trough in the middle 30-70%,
depth-relative no-V ≥ 0.40, depth 20-70% measured left-rim-close → trough, split guard,
phantom-row purge, IPO-age exception, liquidity floor Rs2 cr, ETF/index exclusion,
`pattern_quality` as the only ranking metric.

| | v1 | **v2** |
|---|---|---|
| **Base completion** | close back **within 5% of the left rim** | **minimal lift-off**: `close ≥ trough + 0.15 × (rim − trough)`. The base only has to be *recognisable*, not *recovered* |
| **Entry trigger** | first close **above the left rim** | first close **above the highest close of the prior N bars** (the base ceiling), **N = 60** default, with `fired_n40` / `fired_n60` / `fired_n100` recorded per event |
| **Volume gate** | none at the trigger | **volume ≥ K × median volume of the prior 20 bars**, **K = 3** default; the actual `vol_multiple` is recorded so K = 2 / 5 / 9 can be filtered later |
| **Candle** | none | trigger day must be an **up-candle** (`close > prior close`) |
| **Wait window** | 60 bars from q | **120 bars** from q |
| **Fill** | next-day open **and** buy-stop at the rim | **next-day open** |
| **Left rim** | the entry level | **information only** — `dist_to_left_rim_pct` |

Still strictly causal: the trigger compares `close[t]` against the prior N bars and the prior
20-bar volume median, both shifted back one bar. Nothing at or after `t` is consulted.

## 9.3 v2 deviation log

| # | Change | Why | Cost / effect |
|---|---|---|---|
| **D4** | **Vertex bound widened from 0.33-0.67 to 0.30-0.70**, aligning it with the trough-position bound | With the v1 bound the detector **rejected Arun’s own 16-May-2025 SKFINDIA breakout by 0.004** (vertex 0.674 vs the 0.67 ceiling) while `trough_pos` sat at a comfortable 0.664 and R² at 0.903. The two gates test **the same geometric property** — where the bottom of the bowl sits — and having two different tolerances for it was an inconsistency; the fitted vertex is the noisier estimator, drifting monotonically as bars are appended | **This is not cosmetic: D4 admits 47.4% of all de-duplicated v2 events** (2,072 of 4,372). The vertex distribution is piled up against the old boundary because it sweeps through the band as the window grows. Every event carries `vertex_frac`, so the stricter screen is recoverable with `0.33 ≤ vertex_frac ≤ 0.67`. **Disclosed rather than buried — this is the single loosest choice in v2** |
| **D5** | Base expiry is now checked **before** the trigger each bar | The original order let 9 events fire on bar 121 of a stated 120-bar window | Wait window is now a hard ≤ 120 bars; event count moved 6,118 → 6,109 |

## 9.4 Confirmation on the named examples — both exact

| | Arun / coordinator expected | **v2 detector** |
|---|---|---|
| **SKFINDIA** | 16-May-2025 @ Rs2,144.70, 9.5x volume | **16-May-2025, close Rs2,144.70, 9.45x**, base ceiling Rs2,027.30, entry **19-May-2025 @ Rs2,161.80** |
| **KMEW** (K=3) | 19-Aug-2025 @ Rs944.30, 8.8x | **19-Aug-2025, close Rs944.30, 8.78x**, base ceiling Rs912.85, entry **20-Aug-2025 @ Rs937.50** |
| **KMEW** (K≥9) | 12-Sep-2025 @ Rs1,090.55, 29.9x | **12-Sep-2025, close Rs1,090.55, 29.90x**, entry 15-Sep-2025 @ Rs1,127.05 |

**v2 gets a materially better entry than v1 on every example, and the trail outcome improves
with it** (`info_*` columns, descriptive only — not a backtest):

| Symbol | v1 entry | v2 entry | Entry improvement | ST(7,3) v1 → v2 | fwd250 v1 → v2 |
|---|---|---|---|---|---|
| KMEW | Rs1,191.00 | **Rs937.50** | **21.3% cheaper** | +30.3% → **+65.6%** | +144.5% → **+186.4%** |
| SKFINDIA | Rs2,356.20 | **Rs2,161.80** | **8.3% cheaper** | −9.1% → **−1.0%** | −34.1% → −25.3% |
| DIXON | Rs4,207.85 | **Rs3,509.65** | **16.6% cheaper** | −4.6% → **+14.4%** | +158.0% → **+164.0%** |

KMEW is the clearest vindication of his correction: the earlier entry **more than doubles**
what SuperTrend(7,3) captures, from +30% to +66%. His complaint that "ST 7,3 cuts KMEW half
way through" was substantially an **entry** problem, not only a trail problem.

## 9.5 Do the other v1 names survive? Yes, all three — with earlier entries

| Symbol | v1 breakout / entry | **v2 nearest event** | v2 entry | v2 events in total |
|---|---|---|---|---|
| **CENTURYPLY** | 11-Apr-2017 / Rs269.60 | **30-Mar-2017**, close Rs257.30, ceiling Rs250.15, 3.55x | **31-Mar-2017 @ Rs258.00** | 8 |
| **DIXON** | 12-Jun-2023 / Rs4,207.85 | **24-May-2023**, close Rs3,514.30, ceiling Rs3,275.65, 19.11x | **25-May-2023 @ Rs3,509.65** | 4 |
| **SRF** | 03-Jun-2009 / Rs27.20 | **27-Jul-2009**, close Rs28.90, ceiling Rs28.65, 13.29x | **28-Jul-2009 @ Rs29.60** | 7 |

**SRF is the honest exception: v2 enters LATER and ~8.8% higher than v1 did.** SRF’s 2009
recovery was a near-vertical V off the crash low, so the base ceiling was still being made
new every few days and the 60-bar high was not cleared on 3x volume until late July. v2 is
not uniformly better — it is better on saucers with a real shelf, worse on V-recoveries.

## 9.6 Frequency — v2 is far denser than v1, and that matters

| | v1 | **v2** |
|---|---|---|
| Raw events | 1,698 | **6,109** |
| De-duplicated | 1,513 | **4,372** |
| Distinct symbols | 793 | **1,252** |
| Mean per year | 63 | **175** |
| Pass OBV accumulation | 55% | **41%** |

Per-year (de-duplicated): 2025 **738**, 2023 **599**, 2020 **449**, 2022 396, 2026 343,
2024 259, 2017 196, 2009 203 — against 2019 140, 2011 57, 2008 19.

**This is a real caution.** v2 is ~2.9x denser than v1 and the recent years are approaching
the 500/year mark that section 3.9(b) pre-registered as the "this is a generic screen, not a
rare formation" bound — 2025 and 2023 are already past it. Three things drive the density:
the looser base completion (a base arms near the low instead of near the rim), the 120-bar
wait, and D4. The screen is **still not a coin flip** — 1,252 names out of 2,492 over 25
years — but it is no longer the rare formation v1 described, and the null control at G1
proper is now *more* important, not less.

Other v2 distributions:
- **Volume multiple at the trigger:** median **5.4x**, p10 3.3x, p90 19.2x. 55% of events are
  ≥ 5x and 27% are ≥ 9x, so K is a live tightening axis without re-running the screen.
- **Days from base recognition to trigger:** median **21 bars**, p10 2, p90 87 — no event
  fires on day q itself, so base-then-breakout separation is always visible on the chart.
- **Fill slippage**, next-day open vs the trigger close: median **+0.53%**, p10 −0.69%,
  p90 +1.97%. The fill mechanic is not decisive.

## 9.7 v2 causality self-audit — 0 violations across 14 checks

All 6,109 v2 events re-checked: `left_rim < trough ≤ q ≤ trigger < entry`; trigger close
strictly above the base ceiling; volume multiple ≥ K on every event; the `fired_n60` flag set
on every event; depth, R², no-V, trough-position and vertex each inside their stated bounds;
wait within 0-120 bars. **Zero violations.**

## 9.8 v2 short-list — and it is three losers

Selected by `pattern_quality` only, one per year and per liquidity bucket, from the
1,563-event clean pool. Forward returns never consulted.

| # | Symbol | Left rim (supply) | Trough (depth) | Base ceiling | Breakout (vol) | Entry |
|---|---|---|---|---|---|---|
| 1 | **HUHTAMAKI** | 13-Dec-2024 @ Rs305.05 | 28-Feb-2025 @ Rs176.06 (−42.3%) | Rs220.87 | **07-Jul-2025** @ Rs231.73 (5.4x) | 08-Jul-2025 @ **Rs233.00** |
| 2 | **DELTACORP** | 04-Apr-2022 @ Rs333.65 | 16-Jun-2022 @ Rs163.90 (−50.9%) | Rs214.45 | **15-Sep-2022** @ Rs221.50 (3.5x) | 16-Sep-2022 @ **Rs223.80** |
| 3 | **KIRLOSBROS** | 05-Jul-2024 @ Rs2,603.15 | 16-Sep-2024 @ Rs1,613.35 (−38.0%) | Rs2,117.40 | **19-Nov-2024** @ Rs2,187.15 (5.5x) | 21-Nov-2024 @ **Rs2,179.05** |

**All three subsequently lost** (fwd250: −4.9%, −20.9%, −21.9%; ST(7,3): −11.0%, −12.9%,
−7.3%). Stated plainly because the selection was blind — that is what blind selection is for.
It is a caution about the family, and a reminder that v1’s short-list happened to contain
DIXON’s +158% while v2’s top three contain nothing of the sort. **Neither outcome is evidence
of edge in either direction at n = 3.** Only the G1 null control settles that.

Note also that **two of the three top picks (HUHTAMAKI 0.694, DELTACORP 0.698) sit in the band
D4 opened** — the loosest choice in v2 is disproportionately represented at the top of the
ranking, which is worth Arun’s attention when he looks at the charts.

## 9.9 v2 files

| File | Purpose |
|---|---|
| `scripts/detect_rounding_base_v2.py` | the v2 detector (`--symbols=`, `--k=`, `--n=` for probes) |
| `scripts/summarise_events_v2.py` | per-year counts, example confirmation, blind short-list |
| `scripts/make_verify_list_v2.py` | the simplified verification list |
| `scripts/audit_v2.py` | causality self-audit + D4 cost |
| `scripts/debug_skfindia_v2.py` | per-gate day-by-day trace (this found the D4 vertex block) |
| `scripts/probe_triggers.py` | the trigger probe on the two examples |
| `results/rounding_base_events_v2.csv` | **v2 event table** — 6,109 events × 49 columns |
| `results/verify_list_v2.csv` | **4,372 rows**, dd-Mon-yyyy, one per symbol+entry day |
| `results/summary_v2.txt` | rendered v2 summary |
| `results/rounding_base_events.csv`, `results/verify_list.csv`, `results/summary.txt` | **v1, untouched** |

**v2 re-run:** `cd /home/arun/quantifyd && setsid nohup venv/bin/python -u research/159_rounding_base_breakout/scripts/detect_rounding_base_v2.py > /tmp/r159v2.log 2>&1 < /dev/null &` (~3.5 min), then `summarise_events_v2.py`, `make_verify_list_v2.py`, `audit_v2.py`.

## 9.10 What has NOT changed

No backtest, no exit sweep, no CAGR / Sharpe / Calmar / drawdown, no portfolio construction,
no costs or taxes, no correlation or blend test, no app page, nothing deployed, no live engine
or service touched. Forward-return and SuperTrend columns remain **information only** and were
not used to select or rank anything. The v1 caveats in 8.5 all still apply to v2 — survivorship,
split artifacts, ACCENTMIC-SM absent, the Rs2 cr liquidity floor and its capacity wall — plus
the two new ones above: **D4’s 47% share** and **v2’s much higher event density**.


---

# 10. v3 — shelf breakout near the all-time high (11-Sep-2026)

## 10.1 Arun’s two further corrections

**(1) The pattern must sit at or near the all-time high.** The saucer forms just under the
ATH and the breakout goes into, or close to, blue sky. **SKFINDIA was 34% below its Jun-2024
ATH, so it never qualified in the first place** — v2 was happily finding saucers part-way
down a long decline, which is a different (and much weaker) animal.

**(2) The trigger needs a SHELF.** On CHOLAHLDNG 21-Apr-2025 he said: *"the base is correct,
I don’t see any breakout from the base."* He is right. v2’s "close above the prior 60-day
high" fires **continuously** while price simply walks up the right-hand side of a saucer —
there was no consolidation under Rs1,958; the prior 20 closes spanned Rs1,559-1,873, a **20%
range**. That is a rising price, not a breakout. **A breakout needs something tight to break
out of.**

## 10.2 The v3 spec

Saucer recognition is **unchanged from v2** (curvature > 0, R² ≥ 0.70, trough centred
0.30-0.70, no-V ≥ 0.40, depth 20-70%, 15% minimal lift-off, split guard, phantom purge,
IPO-age exception, liquidity ≥ Rs2 cr). The **trigger** is replaced:

| Condition | Rule |
|---|---|
| **Shelf** | over the prior **S = 15** bars, `(max close − min close) / max close ≤ 12%`; flags recorded for S = 20 and S = 30 |
| **Breakout** | `close > shelf high` |
| **Near ATH** | `close ≥ 0.90 × ATH`, where ATH is the running max of closes **strictly before** day t; `dist_to_ath_pct` recorded, plus flags for ≥ 0.95×ATH and a new high |
| **Volume** | `≥ 3 ×` the prior 20-bar median; the actual multiple is recorded |
| **Candle** | `close > previous close` |
| **Base** | the saucer must have qualified **earlier** (q < t), within **150 bars** |
| **Fill** | next-day open |

**ATH split guard.** `market_data.db` is not retroactively split-adjusted, so a pre-split row
sits at the old price scale and would fake an unreachable ATH. If the close series contains
any day-over-day move < −35%, the series is **truncated to the bars after the last such
move** and the ATH is computed only from those. `hist_bars` records how much history backs
each ATH, and `split_cut` flags whether truncation happened. **19 symbols were skipped
entirely** for having too little history left after truncation.

## 10.3 Expected checks — 9 of 9 PASS

| Symbol | Expected | Result | Detail |
|---|---|---|---|
| **KMEW** | fire 12-Sep-2025 @ Rs1,090.55, shelf Rs972.85, −8.3% vs ATH, 29.9x | **PASS** | close Rs1,090.55, shelf Rs972.85, dATH **−8.33%**, **29.9x**, entry 15-Sep-2025 @ Rs1,127.05 |
| **CENTURYPLY** | fire 30-Mar-2017 @ Rs257.30 | **PASS** | shelf Rs249.85 (range 3.2%), dATH −2.0%, 3.55x, entry 31-Mar-2017 @ Rs258.00 |
| **JAYSREETEA** | fire 12-Aug-2009 | **PASS** | close Rs102.97, shelf Rs101.92, dATH −2.93%, 6.1x |
| **MONARCH** | fire 06-Oct-2023 | **PASS** | close Rs193.67, shelf Rs171.02, dATH −2.95%, 7.48x |
| **NAM-INDIA** | fire 06-Jun-2025 | **PASS** | close Rs790.50, shelf Rs748.95, dATH −1.91%, 4.28x |
| **SAPPHIRE** | fire 06-Oct-2022 | **PASS** | close Rs300.30, shelf Rs297.75, dATH +0.86%, 3.37x |
| **COROMANDEL** | fire 29-Sep-2009 | **PASS** | close Rs103.62, shelf Rs100.95, dATH −7.96%, 4.2x |
| **CHOLAHLDNG** | must NOT fire Apr-2025 | **PASS** | correctly absent (its only event is 24-Oct-2025) |
| **SKFINDIA** | must NOT fire May-2025 | **PASS** | correctly absent (its events are 04-Mar-2021 and 27-Jun-2023, both near ATH at the time) |

**Note on KMEW.** `probe_shelf.py` reports "no shelf breakout" for KMEW because it tests the
**shelf high** against the ATH (972.85 / 1,189.70 = 18% below). The v3 spec tests the
**close** (1,090.55 / 1,189.70 = 8.3% below), which is what makes it fire. The two are
deliberately different conditions; v3 implements the spec.

## 10.4 Frequency — v3 is the tightest of the three screens

| | v1 | v2 | **v3** |
|---|---|---|---|
| Raw events | 1,698 | 6,109 | **1,437** |
| De-duplicated | 1,513 | 4,372 | **889** |
| Distinct symbols | 793 | 1,252 | **579** |
| Mean per year | 63 | 175 | **40** |

**Funnel:** 7,874 saucer bases qualified → **1,437 triggered**, **5,842 expired without a
shelf breakout**. The shelf + ATH requirement rejects ~79% of the bases that v2 would have
traded, which is precisely the point of both corrections.

Per-year: 2023 **143**, 2025 **121**, 2024 96, 2017 64, 2022 60, 2021 59, 2026 54 (partial),
2020 38, against 2008 **1**, 2012 9, 2013 11. The 2009 pile-up that dominated v1 and v2
(160 / 203 events) collapses to **23** — the ATH condition removes most of the post-crash
bounce population, which was the single biggest "market beta wearing a saucer costume"
worry in 8.2. That is a real improvement in the construct, though it does **not** remove the
need for the null control.

Other distributions: distance to ATH median **−5.2%** (25% of events are at a **new** ATH,
49% within 5%); shelf range median **7.8%**; volume multiple median **5.0x** (445 events
≥ 5x, 192 ≥ 9x).

## 10.5 Duplicate series — the named pair is NOT a duplicate here

Scanned **all 2,686** daily symbols by md5 of the full (date, close) series, and again on the
last 250 shared bars to catch renamed tickers whose histories differ in length.

- **Only one true duplicate group exists: `CRESTO` = `SILLYMONKS`** (1,829 bars,
  05-Jan-2015 → 10-Sep-2026, identical). **SILLYMONKS is dropped**, CRESTO kept.
- **`JSWDULUX` / `AKZOINDIA` are NOT identical in our DB** — 5,362 overlapping dates and the
  closes do **not** match (JSWDULUX 5,824 bars from 27-Jan-2003; AKZOINDIA 5,361 from
  03-Jan-2005). The pair was flagged as a known duplicate, and in this data it is not one.
  Reported rather than silently applied.
- Written to `results/duplicate_series.csv`.

## 10.6 v3 sample list — 8 blind picks

Ranked by `pattern_quality` only, at most one per year, spread across caps. Forward returns
and SuperTrend outcomes computed but **never consulted**.

| # | Symbol | Breakout | Close | Shelf high | Shelf rng | dATH | Entry | Entry px | Score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **CENTURYPLY** | 30-Mar-2017 | 257.30 | 249.85 | 3.2% | −2.0% | 31-Mar-2017 | 258.00 | 0.824 |
| 2 | **TRACXN** | 04-Jul-2023 | 90.35 | 87.10 | 9.2% | −9.7% | 05-Jul-2023 | 90.70 | 0.808 |
| 3 | **RBZJEWEL** | 06-Jan-2025 | 228.29 | 217.42 | 10.7% | −1.7% | 07-Jan-2025 | 235.01 | 0.736 |
| 4 | **SAPPHIRE** | 06-Oct-2022 | 300.30 | 297.75 | 5.3% | +0.9% | 07-Oct-2022 | 300.00 | 0.733 |
| 5 | **MANYAVAR** | 23-Sep-2024 | 1,341.69 | 1,278.66 | 5.5% | −6.4% | 24-Sep-2024 | 1,341.69 | 0.694 |
| 6 | **GESHIP** | 05-Mar-2026 | 1,389.40 | 1,354.60 | 5.0% | −4.5% | 06-Mar-2026 | 1,395.60 | 0.660 |
| 7 | **ADANIGREEN** | 08-Nov-2021 | 1,226.00 | 1,206.55 | 4.9% | −9.9% | 09-Nov-2021 | 1,215.00 | 0.586 |
| 8 | **ICICIBANK** | 29-Oct-2018 | 349.40 | 327.10 | 6.4% | −3.6% | 30-Oct-2018 | 347.45 | 0.552 |

**Live candidates (triggered in the last 10 trading days, ≥ 28-Aug-2026)** — these are
screen output, **not recommendations**, and nothing has been backtested yet:
TBOTEK (31-Aug, Rs1,745.10), GUFICBIO (02-Sep, Rs438.80), COFORGE (28-Aug, Rs2,014.60,
**at a new ATH**), SMLMAH (04-Sep, Rs6,298.50, +8.5% above ATH), INNOVACAP (04-Sep),
NAZARA (28-Aug), SMCGLOBAL (07-Sep), WINDLAS (02-Sep), KPRMILL (31-Aug).

## 10.7 v3 files

| File | Purpose |
|---|---|
| `scripts/detect_rounding_base_v3.py` | the v3 detector |
| `scripts/summarise_events_v3.py` | per-year, expected checks, blind short-list, live candidates |
| `scripts/make_verify_list_v3.py` | verification list + duplicate drop |
| `scripts/dup_series_scan.py` | universe-wide duplicate-series scan |
| `scripts/probe_shelf.py`, `scripts/ath_filter_v2.py` | Arun’s probes (kept) |
| `results/rounding_base_events_v3.csv` | v3 event table, 1,437 events × 52 columns |
| `results/verify_list_v3.csv` | **889 rows**, dd-Mon-yyyy |
| `results/summary_v3.txt`, `results/duplicate_series.csv` | summary and duplicate list |

---

# 11. Backtest phase — PRE-REGISTERED BEFORE RUNNING (11-Sep-2026)

**The question:** *can the v3 pattern become a trading system whose returns beat holding
NIFTYBEES?* Everything below is fixed **before** a single sweep cell runs, exactly so the bar
cannot be moved after seeing results.

## 11.1 Ranking metric and adoption bar (pre-registered)

**Ranking metric:** **after-tax net CAGR**, with **Calmar** as the tie-break.

**Adoption bar — ALL of the following must hold:**

1. **After-tax net CAGR > NIFTYBEES buy-and-hold** over the same window, **and**
2. **Max drawdown no worse than NIFTYBEES’s** over that window, **and**
3. **≥ 20% after-tax net CAGR** (25 bps costs, 30-seed **median**; the **worst seed** is
   stated alongside) — *Arun’s addendum, 11-Sep-2026*, **and**
4. **Robust across 30 seeds**: the **worst seed still beats NIFTYBEES on CAGR**, **and**
5. **Both windows pass** (pre-2016 and 2016+), **and**
6. **Beats the date-matched null control** (11.4) after tax.

**Verdict labels:** clears every criterion → **STRATEGY**. Beats NIFTYBEES but lands **below
20% CAGR** → **SIGNAL, not STRATEGY** (explicitly, per Arun’s addendum). Fails to beat
NIFTYBEES, or fails the null → **NO EDGE**. Fails only the incremental test vs the ATH
control → **NO INCREMENTAL EDGE OVER OPEN ALPHA**.

## 11.2 Book construction

v3 entries, **next-open fill**, **16 slots at 6.25% of NAV**, **Rs10L** book, **NSE cash CNC**,
liquidity ≥ Rs2 cr 20-day median traded value at the trigger, **SILLYMONKS dropped** as a
duplicate series. Slot contention resolved by a **random-selection seed ensemble — 30 seeds
for anything reported as a decision**, reported as **median [min..max] plus the worst seed**.
Costs **25 / 40 / 60 bps per side**; **idle cash at 5.5% p.a.**; **after-tax with Indian FY
loss-netting**, 20% STCG / 12.5% LTCG above 365 days.

## 11.3 The grid (cell count disclosed up front)

| Axis | Values | n |
|---|---|---|
| Exit | ST(7,3), ST(10,3), ST(14,4), Donchian-20-low, Donchian-10-low, 15-SMA trail, 50-EMA trail | 7 |
| Hard stop | none, −8% close | 2 |
| Time stop | none, 120 bars | 2 |
| Shelf S | 15, 20 | 2 |
| Volume K | 2, 3, 5 | 3 |
| ATH proximity | ≥ 0.90×, ≥ 0.95×, > ATH | 3 |
| OBV filter | off, on | 2 |
| Market gate | none, NIFTY > 100-SMA | 2 |

**7 × 2 × 2 × 2 × 3 × 3 × 2 × 2 = 2,016 cells**, each on a 10-seed scan, with the
survivors re-run on **30 seeds**. The multiple-testing haircut applies to 2,016 — stated now
so the eventual winner is discounted honestly. **Plateau, not peak**: the neighbourhood of
any winner is reported, and a winner whose neighbours disagree is treated as noise.

## 11.4 Null controls — the decisive one

This pattern is **a subset of Open Alpha’s ATH-breakout entries**, so the question that
actually matters is not "does it make money" but "does the saucer + shelf shape add anything
to simply buying strength near the ATH".

- **(a) Date-matched ATH control** — "close within 10% of ATH on ≥ 3× volume", **no saucer,
  no shelf**, same book, same exits, same seeds. **If v3 does not beat this after tax, the
  verdict is "no incremental edge over Open Alpha" and is reported as such, plainly.**
- **(b) Random entries, date-matched** — same number of entries on the same dates, drawn from
  the liquid universe.
- **(c) Promotion-shrinkage check** per the brief.

## 11.5 Benchmarks and portfolio fit

Every table carries **NIFTYBEES buy-and-hold** (with the NIFTY 50 index as the proxy where
NIFTYBEES history is missing — **which one is used will be stated explicitly**), plus
**Open Alpha** and **True North** at equal size from `research/154_multi_system_blends/scripts/export_curves.py`,
plus the **TN + OA 50-50 blend**. Correlation (daily and monthly) and blend value per section 8
of the agent brief.

## 11.6 Report package

House-format **YoY table** (year cells with intra-year drawdown beneath, BEST CAGR / LEAST DD /
BEST OVERALL columns), **log growth-of-Rs100** vs NIFTY 50 / Midcap 150 / Smallcap 250 with a
**drawdown panel**, **cost ladder** (25/40/60), **tradeability gate** (win rate, avg win/loss,
expectancy net of costs, max losing streak, trades/yr, capacity), **outlier dependence**
(top-10 trades removed; winners capped at +50% and +100%), **two windows**, **seed band**.
Tearsheet via `research/_utilities/tearsheet.py`. `results/RESULTS.md` with the bold verdict.
Then the app study page, `research/INDEX.md`, `TODO.md`, and a dated review in `ops_center.py`.

## 11.7 Falsification plan

The idea is **killed** if: after-tax net CAGR does not beat NIFTYBEES on the 30-seed median;
**or** the worst seed loses to NIFTYBEES; **or** either window fails; **or** it does not beat
the date-matched ATH control (11.4a) — in which case it is redundant with Open Alpha and is
**not** deployed regardless of its standalone numbers.


## 11.8 Backtest phase — status: NOT STARTED, deferred to after 15:40 IST

**Nothing in section 11 has been run.** Phase A (generating the 18 entry-variant event
sets) was launched at **14:43 IST on 11-Sep-2026 and killed ~2 minutes later**, before any
variant file was completed. The reason:

- The market was **open** (NSE cash/F&O 09:15-15:30, closing session to ~15:40).
- The 4-core VPS was already carrying **another agent's research sweep**
  (`research/159_oa_honest_reoptimization/scripts/sweep_honest.py` at ~86% CPU, plus
  `cand8y.py`) **and** the live paper jobs (`fly_paper.py`) and gunicorn, at a
  **load average above 9**.
- Adding CPU-heavy research to that while short option positions are monitored **in
  memory** by the gunicorn process is the same failure class as the rogue REST poller
  that starved the live monitors on 2026-08-14.

The job was killed **by PID**, its partial output (`rounding_base_events_v3_s15_k2_a0.90.csv`)
**deleted**, and the process confirmed gone. **No live engine or service was touched and no
service was restarted.**

**To resume (after 15:40 IST, verify with `TZ=Asia/Kolkata date` on the VPS first):**
```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd
# check the box is quiet first:
uptime; ps -eo pcpu,pid,args --sort=-pcpu | head -6
setsid nohup bash research/159_rounding_base_breakout/scripts/gen_variants.sh   > /tmp/r159_variants.log 2>&1 < /dev/null &
tail -f /tmp/r159_variants.log      # ~50 min for 18 variants, sequential and niced
```
The driver **skips any variant file that already exists**, so it is safe to re-run.

## 11.9 Research-number collision — flagged, not silently fixed

`research/159_` is claimed **twice**: this study (`159_rounding_base_breakout`, assigned in
the original brief) and `159_oa_honest_reoptimization`, created by another agent while this
one was in flight. `research/160_quality_growth_near_ath` also exists, so the next free
number is **161**. This folder was **not** renamed: it is already committed under 159 across
six commits and reported to Arun under that path, and renaming mid-flight would invalidate
every path already handed over. **Arun / the coordinator should decide** whether this study
or the OA one gets renumbered.


---

# 12. Backtest phase — live run log (11-Sep-2026, after market close)

| Time (IST) | Event | Notes |
|---|---|---|
| 15:47 | Market confirmed CLOSED (`TZ=Asia/Kolkata date` on the VPS = 15:47 Fri). Phase A relaunched, `nice -n 10` | 18 entry variants, sequential; ~6 min each under box contention → ETA ~17:35 |
| 15:52 | `bt_core.py` written — book engine: 16 slots @ 6.25%, Rs10L, next-open fills both sides, seeded slot draw, 5.5% idle cash, FY loss-netted tax | exits as CLOSE signals filled at the NEXT open, stated so neither leg gets look-ahead |
| 15:55 | **Engine smoke PASSED.** 0.3 s per 21.7-year simulation | 889 v3 events, 579 symbols, all 889 land on the calendar |
| 15:55 | **NIFTYBEES buy-and-hold measured: CAGR 12.29%, MaxDD −59.71%, Calmar 0.206** (2005-01-03 → 2026-09-11, 21.7 yrs). Pre-2016 12.68% / −59.71%; 2016+ 11.86% / −36.34% | this is the bar |
| 15:57 | Null controls built: **11,862 near-ATH volume-thrust events** (no saucer, no shelf) + liquid matrix (2,491 × 5,378) | **v3’s 889 events are a 7.5% subset of this population** — exactly the "is it just Open Alpha?" question |
| 16:00 | 30-seed decisive comparison launched (V3 vs CTRL_ATH vs CTRL_DM vs CTRL_RND, all 7 exits × stop) | |
| 16:02 | **First cell, ST(7,3):** V3 10.82% / −20.9% / Calmar 0.518 · CTRL_ATH 15.98% / −39.3% / 0.411 · **CTRL_DM 8.41%** / −22.4% / 0.369 | V3 **beats** the date-matched control by **+2.4pp**, but **loses to NIFTYBEES on CAGR** with this exit |

## 12.1 Early finding — the binding constraint is utilisation, not per-trade edge

V3’s 30-seed CAGR range on ST(7,3) is **10.82% – 10.83%**: essentially **zero seed variance**.
That is diagnostic. Seed variance only appears when more candidates compete than there are
free slots; with **889 events across 21.7 years (≈41/yr) and 16 slots**, the book is almost
never contended and therefore **sits largely in cash**, earning 5.5% instead of equity returns.

The same pattern shows in the control: CTRL_ATH has **1,477 trades** against V3’s 583 and
earns **15.98%** — more *because it is more fully invested*, not because each trade is better
(V3’s Calmar 0.518 beats CTRL_ATH’s 0.411, and V3’s drawdown is half as deep).

This matters for the verdict: the pre-registered book (16 slots @ 6.25%) is the right
*honest* test of the spec as written, but it structurally caps what a 41-events-per-year
signal can return. Slot count was **not** in the pre-registered grid, so any slot-sensitivity
result will be reported **explicitly as post-hoc**, never as the headline.


## 12.2 Progress log (continued)

| Time (IST) | Event | Notes |
|---|---|---|
| 16:22 | **30-seed decisive comparison COMPLETE** (`compare_v3_vs_controls.csv`) | V3 beats the date-matched control in **10 of 14** exit configs |
| 16:26 | **Full final analysis** on ST(14,4), 16 slots, 25 bps, 30 seeds | cost ladder, two windows, YoY, outliers, blend vs OA |
| 16:30 | **Post-hoc slot sensitivity** (4 → 20 slots) | CAGR peaks at **15.96% (10 slots)**; the 20% floor is unreachable at ANY slot count |

## 12.3 Results as they stand — the adoption bar, criterion by criterion

Best configuration found: **ST(14,4) trail, no hard stop, shelf S=15, K=3×, ATH ≥ 0.90,
16 slots @ 6.25%, 25 bps, after tax, 30 seeds.**

| # | Pre-registered criterion | Result | Verdict |
|---|---|---|---|
| 1 | after-tax net CAGR > NIFTYBEES | **14.60%** vs **12.29%** | **PASS** |
| 2 | MaxDD no worse than NIFTYBEES | **−24.94%** vs **−59.71%** | **PASS** (less than half) |
| 3 | **≥ 20% after-tax CAGR** (Arun’s floor) | **14.60%** | **FAIL** |
| 4 | worst of 30 seeds still beats NIFTYBEES | **14.57%** > 12.29% | **PASS** |
| 5 | **both windows pass** | pre-2016 **8.64%** vs NIFTYBEES **12.68%** | **FAIL** |
| 6 | beats the date-matched near-ATH control | **+4.73pp** (14.60 vs 9.87) | **PASS** |

**Two criteria fail, so the verdict cannot be STRATEGY.**

### The per-trade signal is genuinely strong
Win rate **46.3%**, average win **+37.9%**, average loss **−11.3%**, **expectancy +11.45% per
trade**, 21.8 trades/yr, max losing streak 14. Cost ladder is nearly flat (14.60 / 14.25 /
13.62% at 25 / 40 / 60 bps) because turnover is low. This is a real edge at the trade level.

### But four things stop it being a system
1. **It is a post-2016 phenomenon.** 2016+ CAGR **20.98%**; pre-2016 **8.64%** while NIFTYBEES
   made 12.68%. Textbook regime dependence.
2. **Extreme outlier dependence.** Compounding the median seed’s 472 trade returns gives
   8.8e11; **removing the ten best trades collapses it to 7.0e6** — a factor of ~125,000.
   Capping winners at +50% gives 3.4e5. The growth is a handful of lottery tickets. (Open
   Alpha, by contrast, keeps ~90% of its growth rate with its ten best trades deleted.)
3. **The book cannot be filled.** 889 events over 21.7 years (~41/yr) against 16 slots leaves
   the book ~40% invested; the 30-seed CAGR band is **14.57-14.75%**, i.e. essentially zero
   seed variance, because slots are almost never contended. Post-hoc, slot count does not
   rescue it: 4 / 6 / 8 / 10 / 12 / 16 / 20 slots give 15.09 / 15.87 / 15.76 / **15.96** /
   15.36 / 14.60 / 13.48%. **A plateau at ~15-16%, never 20%.**
4. **It dilutes Open Alpha rather than adding to it.** Daily correlation **0.468**, monthly
   **0.617** (the complement bar is < ~0.4). OA alone over the shared window: **34.90% CAGR,
   −25.10% DD, Calmar 1.390**. Adding V3 makes every blend worse, monotonically:
   90/10 → 34.24% / 1.368 · 80/20 → 33.51% / 1.344 · 67/33 → 32.43% / 1.310.


## 12.4 Sweep crash and fix (11-Sep-2026)

| Time (IST) | Event | Notes |
|---|---|---|
| 17:12 | **All 18 entry variants finished** (`/tmp/r159_variants.log`) | S × K × ATH = 2 × 3 × 3 |
| 17:10 | **All three sweep shards died instantly** | `ValueError: assignment destination is read-only` at `gate[:100] = True` |
| 18:05 | **Fixed and relaunched** | `pandas.Series.to_numpy()` on a comparison returns a **read-only view**; the market-gate array was being written in place. Fix: `np.array(..., copy=True)` before the assignment. The same latent bug was present in `bt_final.py` (it would only have fired with `--gate 1`, which was never used) and is fixed there too |
| 18:06 | 3 shards relaunched `nice -n 10`, PIDs verified alive; panel cache hit (623 symbols) | ~0.9 s/cell → ~10 min per shard |

**No result already reported is affected.** The crash hit only the 2,016-cell *plateau* sweep.
Everything in §12.3 — the 30-seed final analysis, the cost ladder, the two windows, the
outlier test, the null controls and the blend — came from `bt_compare.py`, `bt_final.py` and
the slot probe, all of which completed before the crash and none of which touches that line.

### A note on reading the comparison log
`/tmp/r159_cmp.log` shows v3 CAGRs spanning **5.2% to 14.6%** across the exit family. The low
end is real but is **not** the system: SMA-15 (5.23%) and Donchian-10 (5.34%) are the fastest
trails, and they are exactly the exits that also **lose to the date-matched control**. The
headline configuration is the slow trail **ST(14,4) at 14.60%**. Quoting the fast-trail cells
as the system’s return would understate it; quoting 14.60% without saying it is the best of
2,016 cells would overstate it. Both are stated.


## 12.5 Sweep complete — 2,016 of 2,016 cells (11-Sep-2026, 18:11 IST)

| | |
|---|---|
| Cells reaching Arun’s **20% CAGR floor** | **0 of 2,016** |
| Best cell | **14.63%** — `ST(14,4) · no stop · no time stop · shelf 15 · K=3 · ATH≥0.90 · no OBV · no gate` |
| Cells beating NIFTYBEES on CAGR **and** drawdown | **22 (1%)** |
| Median / p90 / min cell | **7.34% / 9.88% / 3.72%** |

The sweep **confirms** the verdict reached before it and adds the plateau evidence:
the **exit axis is a true plateau** (ST(14,4) leads on the median of all 288 of its cells,
and every one of the top 15 cells uses it), while the **entry axes are largely inert** —
shelf length 15 vs 20 differs by 0.015pp, the NIFTY-above-100-SMA gate by 0.035pp, and
requiring a *new* all-time high actively **hurts** (6.87% vs 7.77% at ≥ 0.90×). The OBV
accumulation filter costs 1.05pp of CAGR and buys 7.3pp of drawdown — it de-levers rather
than selects.

**STATUS: COMPLETE.** Verdict **SIGNAL, not STRATEGY** and **no incremental value to the
book**. Published at `/app/backtest/rounding-base-shelf-breakout-research159`. Nothing
deployed; no live engine touched; no service restarted.

---

## ADDENDUM, 11-Sep-2026 18:55 IST — the "dilutes Open Alpha" limb is WEAKENED, not withdrawn

A parallel study, `research/159_oa_honest_reoptimization` (a different session, running the
same evening), found that **Open Alpha’s published ~34.9% CAGR — and research/142’s 40.8% —
rests on a same-bar look-ahead fill**: the signal is `close > pivot` and the fill is
`max(pivot, open)` **on that same bar**, i.e. the entry price is taken from a bar whose close
is what generated the signal. Their own sweep marks that cell `placeable: NO`
(`open_same_REFERENCE`, 43.97% CAGR in their stage A), and their **placeable** short-trail
cells come out **negative** (−2.25% to −2.54% CAGR); with much longer trails their honest
cells reach roughly **18–23%** (their stage A2 `close_same` trail-75 cell: 23.20% CAGR,
−45.4% drawdown, Calmar 0.502, 39.3% win rate — and their stage B figures are **pre-tax**).

**What this does to section 3.4 of this document.** The blend test here compared *this*
book — honest next-open fills on both legs, after tax — against the **research/154 Open Alpha
NAV curve**, which inherits that look-ahead. So the comparison was **an honest book against an
inflated one**, and the conclusion that adding this sleeve "dilutes Open Alpha at every
weight" is **not safe as stated**. Against an honest OA curve, this sleeve’s relative
standing would improve, possibly materially.

**What this does NOT change.** The verdict rests on two failures that never touch Open Alpha:

- **criterion 3** — 14.60% after-tax CAGR against Arun’s **20% floor**, with **0 of 2,016
  cells** reaching it and a 4-to-20-slot plateau at 15–16%; and
- **criterion 5** — the **pre-2016 window** (8.64% against NIFTYBEES’ 12.68%).

Both are measured against NIFTYBEES and against the study’s own sweep, not against Open
Alpha. The outlier dependence (ten trades of 472 carrying the result) and the under-filled
book are likewise independent. **The verdict stands: SIGNAL, not STRATEGY.** What is now
open is only whether it would *complement* an honestly-measured Open Alpha — and that
question is deferred to `research/161_ath_base_age_breakout`, which builds its own honest
in-engine OA proxy rather than reusing the r/154 curve.

**Correlation is unaffected** by the fill assumption in direction: 0.468 daily / 0.617 monthly
is a co-movement measurement, and this pattern remains by construction a **subset** of
all-time-high breakout entries.
