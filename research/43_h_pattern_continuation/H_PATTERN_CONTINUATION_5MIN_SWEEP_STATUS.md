# H-Pattern Continuation — Pole→Flag→Continuation on 5-min, Stocks + NIFTY50

**STATUS: DONE** — full run complete (375 stocks + NIFTY50, 6.39M trades, 290s).
Verdict + full tables in `results/RESULTS.md`. Phase-1 read: thin short-biased
gross edge, **breakout|MM** best, does not survive 6 bps costs (net −0.035R,
PF 0.95) → filters in Phase 2.

The "H pattern": an impulse leg (left post of the H) → a retracement that stalls
at a level forming a sideways consolidation (the crossbar) → continuation in the
direction of the impulse (right post). Mechanically this is a **pole + flag
continuation**. We trade it *with the trend the impulse defines* (bearish in the
user's example chart, but detected symmetrically for longs too).

---

## 1. The Ask

**What you asked (verbatim):**
> "H pattern - you can see after a down fall, slight retracement and now its
> forming an h pattern, we need to identify such moments in stocks and trade
> this h breakout in the direction of the trend which is bearish in this
> example.... it need not be a breakout, we can take the trade as it forms, or
> we can move to 2 min tf and take the trade on break of this h, can take prev
> day's level breaks into consideration, cpr etc so and so.... pls backtest
> completely"

**What we're actually testing:**
Across the liquid 5-min universe (≈380 F&O / Nifty-500 names) **and** NIFTY50 as
a separate cohort, 2018→2026: detect every intraday "H" (pole→flag→continuation)
in both directions, then compare **three entry styles** —
**(A) breakout of the crossbar, (B) fade as it forms at the crossbar level,
(C) retest of the broken crossbar** — crossed with a grid of exit/target
policies. Rank by R-multiple expectancy and profit factor to answer: *does the H
continuation have a tradable edge, and which entry style captures it best?*

**Decisions locked with the user (2026-05-29):**
- Entry: test **all three** styles and compare comprehensively.
- Universe: **both** — full stock universe + NIFTY50, reported as separate cohorts.
- Filters: **pure H baseline first** — no CPR / prev-day-level / daily-trend
  filters yet. Those + the **2-min entry refinement** are **Phase 2**.

**Known data gap:** the DB has **no 2-minute data**. The "drop to 2-min to enter
on the break" idea is *not backtestable* here; it is deferred to Phase 2 pending
a Kite 2-min download (VPS-only per project rule).

---

## 2. The Base — exact mechanics being tested

All on **5-minute bars**, regular session **09:15–15:25 IST**, **intraday only**
(pole + flag + trade all inside one day; hard square-off at 15:25). ATR = Wilder
14-period computed on the continuous 5-min series (carries across days so the
first bars of a day have a stable volatility unit).

### Pattern detection (bearish H; bullish is the mirror) — FIXED for baseline
1. **Pole (impulse / left post):** within a lookback of ≤ `POLE_MAX=10` bars, a
   swing-high → swing-low drop with `pole = high_hi − low_lo ≥ POLE_ATR=2.5 × ATR`,
   high before low, and **downward efficiency** `pole / Σ|Δclose| ≥ EFF=0.5`
   (clean, not choppy).
2. **Flag (crossbar / consolidation):** the next `FLAG_MIN=3 … FLAG_MAX=12` bars
   form a counter-trend/sideways band with:
   - `flag_high < pole_high` (a *lower* high — the retracement fails),
   - retrace `(flag_high − low_lo) / pole ≤ RETR_MAX=0.7`,
   - band holds: `flag_low ≥ low_lo` (no new low yet — hasn't broken out),
   - channel width `flag_high − flag_low ≤ WIDTH_ATR=1.5 × ATR`.
3. **Armed** at the last flag bar `f`. `crossbar_top=flag_high`,
   `breakout_level=flag_low`, `pole_height=pole_high−low_lo`.
   One position per day at a time; scanning resumes after a trade closes.

### Entry styles (the comparison axis the user asked for)
- **A — Breakout:** short when a later bar trades below `flag_low − buf`
  (stop-entry). Cancel if price closes back above `flag_high` or not triggered
  within `BREAK_WAIT=10` bars.
- **B — Fade-as-forms:** short at `flag_high` (the crossbar) on the flag bar that
  tags the top of the band — earliest entry, tightest stop, more failures.
- **C — Retest:** require a breakout below `flag_low` first, then short on the
  pullback that retags `flag_low` (now resistance) within `RETEST_BARS=10`.
  Fewest fills, expected higher win-rate.

Invalidation/stop reference for all shorts: `stop = flag_high + STOP_BUF×ATR`
(`STOP_BUF=0.25`). `R = |entry − stop|`.

### Exit / target grid (crossed with entry style)
- `1R`, `2R`, `3R` fixed targets
- `MM` measured-move: `breakout_level − pole_height`
- `TRAIL` ATR-ratchet trail (`TRAIL_ATR=1.0`)
- **All** variants also force **EOD square-off at 15:25** and a `MAXHOLD=30`-bar
  time stop.

**Costs:** round-trip `COST_BPS=6` (≈3 bps/side: brokerage + slippage on liquid
names) deducted from each trade's return.

### Success criterion
Rank variants by **expectancy in R per trade**, gated on: ≥ 200 trades (stocks
cohort), profit factor > 1.2, and positive expectancy after costs. Report
long/short split and win-rate alongside.

---

## 3. Plan — variant grid + cell count

| Axis | Values | n |
|---|---|---|
| Entry style | breakout, fade, retest | 3 |
| Target/exit | 1R, 2R, 3R, MM, TRAIL | 5 |
| Cohort | stocks (≈380), NIFTY50 | 2 |
| Direction | long + short (both reported, not a separate run) | — |

**Cells = 3 × 5 × 2 = 30 result rows.** Detection runs **once per symbol**; all 15
entry×target variants simulate on the same detected H catalog (cheap), so cost is
dominated by the one-time per-symbol bar scan.

Detection-threshold sensitivity (POLE_ATR, RETR_MAX, FLAG_MAX, WIDTH_ATR) is held
fixed for this baseline and becomes a Phase-2 sweep if the baseline shows an edge.

---

## 4. Status (live log)

**State header:** writing engine + smoke test → full run pending.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-05-29 | Folder + STATUS doc created | sections 1–8 written before launch |
| 2026-05-29 | Engine written + smoke-tested | RELIANCE/TCS/SBIN/NIFTY50, 18s |
| 2026-05-29 | Fixed look-ahead in fade entry | was entering at whole-flag max (stop never at risk); now enters at arm-bar close |
| 2026-05-29 | Split gross vs net R | flat-bps cost dominates tight intraday stops; gross is the honest edge read |
| 2026-05-29 | Full ~380-symbol run launched | background, resumable; ETA ~35–40 min |
| 2026-05-29 | Run DONE — 375 stocks + NIFTY50 | 6.39M trades in 290s; rankings + RESULTS.md written |

**FINAL (full universe, 6.39M stock trades):** breakout|MM best — gross +0.163R,
net −0.035R, PF 0.95. Breakout beats fade/retest on net (fade gross-equal but 2×
cost from its tight stop). Shorts > longs in every variant (bearish skew). No
variant net-positive after 6 bps, but breakout|MM is close → Phase-2 filters.

**Smoke-test read (3 stocks + NIFTY50 — NOT conclusive):**
- **breakout** = best entry style: gross +0.18R *and* lowest cost (0.24R). fade
  worst (gross +0.12R but 0.50R cost — tight stop). retest in between.
- Raw H has a **thin positive gross edge that does NOT survive ~6 bps costs** →
  net negative everywhere. Filters (Phase 2) needed to lift the edge.
- **Shorts > longs** (breakout|3R short net −0.01R vs long −0.11R) — H leans bearish.
- Letting winners run (3R / measured-move) beats 1R.

*(rows appended live as the run progresses)*

---

## 5. Crash Recovery — resume without Claude

- **What finished:** `wc -l results/symbol_variant_stats.csv` — one row per
  symbol×variant. `cut -d, -f1 results/symbol_variant_stats.csv | sort -u` lists
  symbols already processed.
- **Is it still running?** `ps aux | grep run_h_sweep` (or check the bg shell).
  Tail progress: `tail -f /tmp/h_sweep.log`.
- **Resume:** re-run the SAME command — the runner skips symbols already present
  in `symbol_variant_stats.csv`:
  `cd <repo> && python3 research/43_h_pattern_continuation/scripts/run_h_sweep.py --cohort stocks >> /tmp/h_sweep.log 2>&1`
- **Aggregate-only** (if all symbols done but ranking missing):
  `python3 research/43_h_pattern_continuation/scripts/run_h_sweep.py --aggregate-only`
- **Do NOT touch:** `results/symbol_variant_stats.csv` while a run is live
  (it's appended to). Safe to read any time.

---

## 6. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| `scripts/h_pattern.py` | Detector + trade simulator (pure logic) | yes |
| `scripts/run_h_sweep.py` | Per-symbol driver, incremental aggregation | yes |
| `H_PATTERN_CONTINUATION_5MIN_SWEEP_STATUS.md` | This file | yes |
| `results/symbol_variant_stats.csv` | Per-symbol×variant tallies (resume key) | yes (small) |
| `results/h_patterns_sample.csv` | Sample of detected H setups (audit) | yes (small) |
| `results/ranking_stocks.csv` | Final per-variant ranking, stocks cohort | yes |
| `results/ranking_nifty50.csv` | Final per-variant ranking, NIFTY50 cohort | yes |
| `results/RESULTS.md` | Final honest verdict + Phase-2 plan | yes |

---

## 7. Findings (during + final)

*(populated as results land)*

---

## 8. Phase 2 — context-filter gating (DONE)

**FINAL (full universe, 218s):** filters work. `none` reproduces P1 exactly
(sanity ✓). **Prev-day-low break is the decisive filter** — the only single
filter that flips the edge net-positive. Best = **short-only breakout|MM with
`pdl+trend`**: short-net **+0.060R**, PF 1.03, **+1,130 total R over 32.6k short
trades**. Edge is entirely short-side (drop longs); CPR adds nothing (redundant
with pdl). Positive after costs but **thin (PF ~1.03, fails the PF>1.2 gate)** —
treat as a confluence overlay, not standalone. Full verdict: `results/RESULTS_P2.md`.
Next lever: detection-threshold sweep on short-only `pdl+trend`.

---


Carry forward the P1 winner — **breakout entry, MM & 3R targets, both directions
(short emphasised)** — and gate trades on confluence filters to lift the gross
edge above costs. Filters use **daily** data (prev session, no look-ahead):

- **pdl** — breakout level breaks the *previous day's low* (short) / *high* (long).
- **cpr_w** — previous day's CPR is *narrow* (width/price < 0.5%) → trend day expected.
- **cpr_pos** — entry sits *below* CPR bottom (short) / *above* CPR top (long).
- **trend** — daily close > / < SMA200 agrees with the trade direction.

CPR from prev-day H/L/C: `P=(H+L+C)/3`, `BC=(H+L)/2`, `TC=2P−BC`.

**Combos tested (× MM, 3R targets):** none, pdl, cpr_w, cpr_pos, trend,
pdl+trend, cpr_pos+trend, pdl+cpr_pos, cpr_pos+cpr_w, all(pdl+cpr_pos+trend) = 10
combos × 2 targets = **20 variants**, full universe + NIFTY50.

**Gate to "tradable":** net expectancy > 0, PF > 1.2, ≥ ~2,000 filtered trades.

Scripts: `scripts/run_h_p2.py` (imports detection from `h_pattern.py`).
Output: `results/ranking_p2_stocks.csv`, `ranking_p2_nifty50.csv`,
`symbol_variant_stats_p2.csv` (resume key), `RESULTS_P2.md`.

### Deferred beyond P2
- **2-min entry refinement** — requires Kite 2-min download (VPS-only).
- **Detection-threshold sweep** (POLE_ATR, RETR_MAX, FLAG_MAX, WIDTH) if a
  filtered variant clears costs.
