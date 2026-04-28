# Short-Options Signal Sweep — STATUS

**Created:** 2026-04-28
**Owner:** Claude (driving) / Arun (deciding)
**Authoritative resume point** if Claude's context is lost.

---

## 0. Requirements / Quick start (read this first)

If you're picking this up cold, here's the one-screen brief.

**What we're doing:** intraday signal-quality backtest. For 6 strategies
(A/B/C/D on NIFTY, E/F on 10 stocks), measure how the **underlying**
behaves after each signal — to judge whether the situation was
favourable for shorting slightly-OTM options. **No options data
involved.**

**Success criterion:** for each (strategy × variant × symbol), produce
distributions of MAE_against, MFE_with, EOD net drift, and win-rates
under 5 different exit policies. Rank by `mean(net_pts) − 0.5·std(net_pts)`.

**Run order:**

1. `scripts/data_loader.py` — pulls NIFTY + 10 stocks 5-min from
   `backtest_data/market_data.db` table `market_data_unified`. Caches.
2. `scripts/indicators.py` — pure functions: RSI(14), EMA(9), VWAP,
   CPR (both conventions), OR15 levels, day-running extremes.
3. `scripts/signals.py` — one signal generator per path/strategy.
4. `scripts/run_phase1.py` — walk forward day-by-day, log signals +
   trajectories incrementally to `results/phase1_signals.csv` and
   `results/phase1_trajectories.csv`.
5. `scripts/run_phase2.py` — replay 5 exit policies, write
   `results/phase2_exits.csv`.
6. `scripts/run_phase3.py` — rank, write `results/RESULTS.md`.

**Universe (locked):**
- NIFTY index: `NIFTY50` 5-min, 2024-03-04 → 2026-03-19 (clipped to
  match daily-end for CPR).
- 10 stocks: RELIANCE TCS HDFCBANK INFY ICICIBANK SBIN BHARTIARTL ITC
  KOTAKBANK HINDUNILVR; same period.
- Daily for CPR: same period, table `market_data_unified` timeframe
  `day`.

**Data prerequisite check (one command):**
```python
python -c "import sqlite3; con=sqlite3.connect('backtest_data/market_data.db'); print(con.execute(\"SELECT symbol, timeframe, COUNT(*) FROM market_data_unified WHERE symbol IN ('NIFTY50','RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','SBIN','BHARTIARTL','ITC','KOTAKBANK','HINDUNILVR') AND timeframe IN ('5minute','day') GROUP BY symbol, timeframe ORDER BY symbol, timeframe\").fetchall())"
```

**Always-true rules:**
- CSV writes are **append-only and incremental** — every signal/row
  flushes immediately. Never batch.
- Each phase **resumable** — re-running skips already-completed
  (date × strategy × variant × symbol) cells.
- Sweep status (this file) updated at every state transition (launch,
  per-cell completion ticks, errors, finish).

**Key interpretations locked here (don't re-litigate):**
- "2nd/3rd/4th candle after OR15" = 5-min candles starting at 09:35,
  09:40, 09:45 (1st post-OR is 09:30 = excluded).
- Per-path, **one signal per day** = the first signal that fires in
  that path's window.
- 10-min and 15-min timeframes are **resampled from 5-min** by grouping
  consecutive pairs/triples starting at 09:15. Trailing partial groups
  are dropped.

---

## 1. Goal + scope

Identify intraday entry signals that put the **underlying** in a favourable
state for **selling slightly-OTM options** (naked or as credit spreads).
This first-pass backtest measures *signal quality on the underlying* — no
option-premium modelling. A signal is "favourable" if, post-signal, the
underlying shows low adverse excursion against the signal direction and
drifts with-or-sideways through the rest of the session.

### Universe

| Bucket | Symbols | Timeframes | Data status |
|---|---|---|---|
| Index | NIFTY50 only (BNF dropped — weekly options discontinued Nov 2024) | 5-min | ✅ in `market_data_unified` 2024-03-01 → 2026-03-25 (33,993 rows). Daily 740 rows 2023-03-20 → 2026-03-19 for CPR |
| Stocks | LOCKED 2026-04-28: all 10 stocks for which we have 5-min intraday — RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC, KOTAKBANK, HINDUNILVR | 5/10/15 min (10/15 resampled from 5) | ✅ 2018-2025 each in `market_data_unified` (~1.3M rows total). Backtest period clipped to 2024-03-01 → 2026-03-25 to match NIFTY window. |

### Period

2024-03-01 → 2026-03-25 (~2 years of intraday). Top-up to current date
deferred to Phase 3 if Phases 1–2 look promising.

### Out of scope (first pass)

- Option premium P&L
- BANKNIFTY (universe locked to NIFTY)
- Live deployment / order execution

---

## 2. Strategies under test

### Path A — early breakout + RSI confirm (NIFTY index)
- OR window: 09:15–09:30 (OR15)
- Trigger: 2nd / 3rd / 4th 5-min candle after OR breaks OR high/low AND
  RSI(14) ≥ 60 (long) or ≤ 40 (short)
- Single signal in window 09:30–09:45

### Path B — post-11:00 RSI drift (NIFTY index)
- Window: 11:00 – 15:15
- Trigger: candle close with RSI ≥ 60 → long; ≤ 40 → short
- No price-level requirement

### Path C — post-12:00 day-extreme break + range compression (NIFTY index)
- Window: 12:00 – 15:15
- Trigger: candle breaks day-high (long) / day-low (short) AND day's
  range so far is "compressed"
- "Compressed" sweep: range/open ≤ {0.4%, 0.6%, 0.8%, 1.0%, off}
- RSI gate: tested as separate variants (with / without)

### Path D — post-12:00 RSI drift + CPR alignment (NIFTY index)
- Window: 12:00 – 15:15
- Trigger: RSI ≥ 60 + bullish CPR direction → long;
  RSI ≤ 40 + bearish CPR direction → short
- **Two CPR conventions** swept as separate variants:
  - **D-priceCPR**: price above Top Central = bullish, below Bottom
    Central = bearish
  - **D-cprDelta**: today's CPR > yesterday's CPR = bullish, lower = bearish

### Strategy E — first-candle Open=Low / Open=High (HELD STOCKS)
- Timeframe sweep: 5 / 10 / 15 min
- Day's first candle: Open=Low → long bias (trigger when price crosses
  candle high); Open=High → short bias (trigger when price crosses
  candle low)
- Filter variants: {base, +CPR, +RSI≥60/≤40, +CPR+RSI}

### Strategy F — EMA9 cross VWAP + CPR alignment (HELD STOCKS)
- Long: EMA(9) crosses above VWAP AND price > CPR
- Short: EMA(9) crosses below VWAP AND price < CPR
- One signal per day per stock (first cross of session)
- Timeframe TBD per Strategy E result (likely 5-min)

---

## 3. Open parameters

### Sweep grid (defaults locked, parameters explored)

| Parameter | Values |
|---|---|
| Gap filter (NIFTY paths) | 0.3% / 0.5% / 0.7% / 1.0% / off |
| RSI thresholds | 40/60 (base) / 35/65 / 30/70 |
| Path C range gate | 0.4% / 0.6% / 0.8% / 1.0% / off |
| OR window | OR15 (locked) |

### Pending user input

(none — all open params resolved 2026-04-28)

---

## 4. Phased plan

### Phase 1 — Signal logger (no exits applied)

For every signal that fires, walk forward candle-by-candle to 15:15 IST
and log:

- `signal_id, date, path, variant, symbol, timeframe`
- `signal_time, signal_price, direction (long/short)`
- `MAE_against` (max adverse pts) + `time_to_MAE`
- `MFE_with` (max favourable pts) + `time_to_MFE`
- `net_pts_at_15_15`
- `mae_before_mfe` (boolean — adverse excursion happened before
  favourable; trailing exits fail when this is true)
- Per-candle close price stored in a separate trajectory CSV (one row
  per signal × candle) for replay

Output: `research/short_options_sweep/phase1_signals.csv`

### Phase 2 — Exit-policy replay

Apply 5 exit policies to the same signal set; emit per-policy P&L stats:

| Policy | Logic |
|---|---|
| **T0** Time only | Hold to 15:15 |
| **T1** Time + hard SL | 15:15 OR underlying moves −SL pts against signal. Sweep SL = {30, 50, 75} for NIFTY; scaled-by-ATR for stocks |
| **T2** Time + trail | When +TR pts in favour, trail SL to entry; hard SL still on; 15:15 backstop. Sweep TR = {10, 20, 30} |
| **T3** RSI-reversal | Exit when RSI re-crosses 50 against direction; else 15:15 |
| **T4** Level-reversal | Exit when price re-crosses signal level (OR for A; signal candle open for B/D; day extreme for C; first-candle level for E; EMA-VWAP cross for F); else 15:15 |

Output: `research/short_options_sweep/phase2_exits.csv` — one row per
(signal, exit_policy, params).

### Phase 3 — Comparative ranking

Rank by `mean(net_pts) - 0.5 * std(net_pts)` (penalise variance) per
(strategy × variant × exit_policy) and output the top 5 per strategy.

---

## 5. Status (running log — update at every state transition)

| Date | Event | Notes |
|---|---|---|
| 2026-04-28 | Spec drafted, MEMORY entry created, STATUS file created | — |
| 2026-04-28 | Stock universe locked to the 10 intraday stocks; STATUS doc moved from `docs/` to `research/29_short_options_signal_sweep/` | All open params resolved |
| 2026-04-28 | `scripts/data_loader.py` built and verified | NIFTY 5min 33,597 rows over 450 sessions; 10 stocks each ~33,045 rows; resampling 5→10/15min works |
| 2026-04-28 | `scripts/indicators.py` built and verified | RSI(14), EMA, VWAP (NaN for indices, ok), CPR (both conventions), OR15, day-running extremes, gap |
| 2026-04-28 | `scripts/signals.py` built — Path A only (B/C/D/E/F are stubs raising `NotImplementedError`) | Path A signal: post-OR break of 09:35/09:40/09:45 candle close + RSI confirm |
| 2026-04-28 | `scripts/run_phase1.py` built with `--smoke` mode | Resumable; appends incrementally to phase1_signals.csv + phase1_trajectories.csv |
| 2026-04-28 | **Smoke test passed** — Path A, NIFTY, 2024-03-04 → 2024-04-30 (34 sessions) | 8 signals fired (24% session hit-rate); 5 wins / 3 losses by EOD; net +117 pts (+15/trade mean); 560 trajectory rows logged; runtime 0.2s |
| 2026-04-28 | Path B implemented (post-11 RSI zone-entry, first-of-day) | Stub replaced; signature compatible with `run_cell` dispatcher |
| 2026-04-28 | `run_phase1.py` refactored to a generic `run_cell(callable, kwargs, ...)` dispatcher | One code path for all signal generators |
| 2026-04-28 | **Full Phase-1 sweep complete for Paths A + B on NIFTY** (18 variants, 2024-03-04 → 2026-03-19) | 2,956 new signals logged in 62s; Phase-1 CSV row count: signals 2,964, trajectories ~110k |
| 2026-04-28 | `summarize_phase1.py` built — per-variant aggregate stats (win-rate, mean/std net, MAE percentiles, adverse-first rate, %MAE≤30) | |
| 2026-04-28 | Path C/D/E/F generators implemented in `signals.py`; `run_phase1.py` extended to dispatch all paths with per-(symbol,timeframe) data registry; resampled walk-forward tracks 5-min trajectories | smoke check on 30-day window verified all 6 generators fire |
| 2026-04-28 | **Full Phase-1 sweep complete on all 6 paths** (A+B+C+D+E+F, 2024-03-04 → 2026-03-19) | A:1,802 / B:1,162 / C:1,963 / D:1,701 / E:6,501 / F:2,037 = **15,166 signals total** across 47 unique cells. CSV: signals 6.6 MB, trajectories ~17 MB |
| 2026-04-28 | `run_phase2.py` built — 5 exit policies (T0 / T1×3 SL / T2×3 TR / T3 RSI / T4 LVL) with per-instrument scaling for stocks (75th-pct |close-diff| as ATR proxy, factors 0.5/1.0/1.5 for SL and 0.25/0.5/1.0 for TR) | |
| 2026-04-28 | **Phase 2 complete** — exit-policy replay over 15,127 signals with forward trajectories | **136,143 exit rows** written; 39 signals had no forward path (signal fired on last candle of day) and were skipped. Runtime ~5 min. CSV size 17.2 MB |
| 2026-04-28 | `run_phase3.py` built and run — aggregation by (path × variant × symbol × timeframe × exit_policy × exit_params), composite score `mean - 0.5·std`, full ranking + per-path top-5 + auto-interpretation | **1,476 ranked configurations**. `phase3_ranking.csv` and `RESULTS.md` written |
| 2026-04-28 | **SWEEP COMPLETE** | See section 8 below for findings |

---

## 8. Final findings

### Top 3 configurations across all strategies (by composite score `mean − 0.5·std`)

The composite formula penalises std heavily, so it favours stock paths (low absolute pts std) over NIFTY paths (50–100 pt std). For a fair within-path ranking, see the per-path tables in `results/RESULTS.md`.

1. **Path E `cpr_10min` / ITC 10-min / T2_TR0.11** — n=42, mean +0.37 pts, win-rate 24%, composite -0.62. ITC's low absolute volatility lifts it to the top despite a tiny edge.
2. **Path E `cpr_10min` / ITC 10-min / T2_TR0.22** — same n=42 (TR rarely activated on this stock).
3. **Path E `cpr_rsi_rsi40_60_10min` / ITC 10-min / T2_TR0.11** — n=39, mean +0.32, composite -0.63.

### Per-path best raw signal quality (T0 time-only exit)

| Path | Best variant | Symbol/TF | n | mean_net | win-rate |
|---|---|---|---:|---:|---:|
| A | `gapoff_rsi30_70` | NIFTY 5m | 99 | **+16.89** | 57.6% |
| B | `rsi30_70_from1100` | NIFTY 5m | 305 | +6.08 | 53.8% |
| C | `rng0.004_norsi` | NIFTY 5m | 45 | **+16.05** | **64.4%** |
| D | `priceCPR_rsi30_70` | NIFTY 5m | 238 | +5.13 | 52.1% |
| E | `cpr_rsi_rsi40_60_10min` | HDFCBANK 10m | 54 | +0.12 | 51.9% |
| F | `priceCPR` | ITC 5m | 216 | -0.02 | 50.5% |

### Honest read

- **Path C (post-12 day-extreme break + tight range gate) is the standout for NIFTY shorting OTM options.** A 0.4% range cap gives +16 pt mean drift with **64% win-rate at simple time-only exit**. Sample is small (45 signals over 2 years) but the pattern is consistent across the rng/rsi sub-grid: tighter range → cleaner breakouts. RSI gate has effectively zero effect because day-extreme breaks already coincide with extreme RSI.
- **Path A (early breakout + extreme RSI 30/70)** scores +16.9 mean but std is huge (~120). T2_TR30 trail-to-entry lifts mean to +19.3 pts but std stays ~80. These are explosive directional moves — good for *directional* trades, mediocre for premium-selling on the directional leg.
- **Paths B and D show modest +5–7 pt edges** at extreme RSI thresholds (30/70). Adding T1_SL30 (hard stop at -30 pts) is the best Phase-2 addition — knocks std down ~25% while preserving most of the mean.
- **Strategies E and F (stocks) basically have zero edge by EOD** — mean net pts hovers around 0, win-rates 50%. Filter variants (CPR/RSI) reduce signal count without lifting quality. NOT promising for short-options at the daily-EOD horizon — the underlying random-walks. Stop here unless re-framed to a much shorter holding window (next 30–60 min).
- **Best exit policy across NIFTY paths is T1_SL30 or T2_TR10/30** — both reduce std meaningfully while preserving the directional drift. T4 (level reversal) underperforms because price chops back through the signal level constantly.
- **`pct_sl_first` runs 17–47%** on NIFTY trail/stop policies — most signals don't get whipsawed within a 30-pt window, but a meaningful minority do. For an option-seller this is the rate at which one would be forced to roll/close intraday.

### Recommendation for Phase 4 (option-premium-level backtest)

**Promote** to Phase 4 with real option-chain data:
1. **Path C `rng0.004_norsi`** — best win-rate, moderate sample. Directional follow-through favours buying the in-direction wing of an iron condor or selling the out-of-direction strangle leg.
2. **Path A `gapoff_rsi30_70`** — high mean drift; run option backtest to see whether premium decay outweighs directional whip risk.
3. **Path B `rsi30_70_from1100` with T1_SL30** — broad sample (305 signals), modest edge, well-behaved with hard SL.

**Skip** Path D (essentially same edge as Path B with smaller sample), Path E, Path F (no daily-EOD edge to monetise via options).

### Smoke-test signal table

| # | Date | Time | Dir | Entry | RSI | Gap% | MAE_against | MFE_with | Net@EOD | MAE_first |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2024-03-12 | 09:35 | long | 22425 | 60.8 | +0.01 | 126.1 | 20.7 | -91.2 | no |
| 1 | 2024-03-15 | 09:35 | short | 22031 | 30.4 | -0.37 | 17.6 | 92.9 | +34.2 | yes |
| 2 | 2024-03-27 | 09:35 | long | 22109 | 70.1 | +0.22 | 7.3 | 81.4 | +38.4 | no |
| 3 | 2024-03-28 | 09:35 | long | 22230 | 69.9 | +0.18 | 0.0 | 272.9 | +112.8 | no |
| 4 | 2024-04-02 | 09:35 | long | 22479 | 60.3 | -0.01 | 81.8 | 1.5 | -14.7 | no |
| 5 | 2024-04-08 | 09:35 | long | 22620 | 76.3 | +0.29 | 28.3 | 75.4 | +41.1 | yes |
| 6 | 2024-04-12 | 09:40 | short | 22641 | 26.3 | -0.34 | 54.5 | 128.3 | +121.2 | no |
| 7 | 2024-04-30 | 09:35 | long | 22711 | 74.0 | +0.16 | 136.4 | 68.4 | -123.7 | yes |

Sample size is tiny — these numbers are not conclusions, just a sanity
check that the pipeline produces sensible signals.

### Phase-1 full sweep — Paths A + B on NIFTY (2024-03-04 → 2026-03-19)

#### Path A (15 variants — gap × RSI grid)

| Variant | n | win% | mean_net | std | mean_MAE | p90_MAE | %MAE≤30 | %adv_first |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gapoff_rsi30_70 | 99 | 57.6 | +16.9 | 119 | 73 | 186 | 38.4 | 38.4 |
| gap0.007_rsi30_70 | 89 | 57.3 | +14.9 | 116 | 71 | 181 | 39.3 | 34.8 |
| gap0.005_rsi30_70 | 84 | 57.1 | +13.7 | 116 | 70 | 181 | 39.3 | 34.5 |
| gap0.010_rsi30_70 | 96 | 57.3 | +14.7 | 117 | 74 | 192 | 38.5 | 37.5 |
| gap0.003_rsi30_70 | 64 | 54.7 | +9.6 | 118 | 72 | 196 | 39.1 | 34.4 |
| (35/65 RSI variants) | 98–135 | 55–57 | +12–16 | 128–133 | 77–80 | 194–205 | 35–38 | 41–44 |
| (40/60 RSI variants) | 125–166 | 54–55 | +11–13 | 137–143 | 82–85 | 203–206 | 35–36 | 41–43 |

#### Path B (3 variants — post-11 RSI drift)

| Variant | n | win% | mean_net | std | mean_MAE | p90_MAE | %MAE≤30 | %adv_first |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **rsi30_70_from1100** | 313 | 52.4 | +5.9 | 85 | **48** | **126** | **54.3** | 47.0 |
| rsi35_65_from1100 | 405 | 51.4 | +3.4 | 94 | 58 | 144 | 45.4 | 52.4 |
| rsi40_60_from1100 | 444 | 51.8 | +2.9 | 98 | 59 | 148 | 43.0 | 54.5 |

#### Takeaways (Phase-1 raw, no exits applied)

1. **Path B with RSI 30/70 is the most short-options-friendly raw
   signal.** 54% of signals stay within a 30-pt MAE; mean MAE 48 pts;
   p90 only 126 pts. ~155 signals/year. *But net is barely positive
   (+5.9/trade)* — it's saying "the move usually doesn't develop, but
   it also doesn't move much against you" — exactly the regime a short
   premium writer wants.
2. **Path A has higher conviction (mean_net +14–17, win% 57–58) but
   more adverse risk** — mean MAE 70–85 pts, p90 ~190 pts. Bigger
   wins, bigger losses — short premium would frequently get torched
   without a stop.
3. **Stricter RSI (30/70) beats 35/65 beats 40/60** uniformly on every
   metric. Worth carrying 30/70 as the default for downstream paths.
4. **Gap filter does not add edge** for Path A. `gapoff` performs
   identically to `gap0.5%` and `gap1.0%`. Drop it from Phase 2/3
   priority list.
5. **High adverse-first rate (35–55%)** means trailing exits (T2) will
   fail often. Hard SL (T1) is the right exit prior; we'll confirm in
   Phase 2.

### Pause for user review

Stopping here per the agreed cadence. Next, on confirmation: implement
Paths C and D, run their sweep, then move to stocks (E, F).

---

## 6. Crash recovery — how Arun resumes without Claude

If Claude is unavailable or context is lost, here's what to do:

### A) Check what's been built

```bash
ls research/short_options_sweep/    # scripts and CSV outputs live here
cat docs/SHORT-OPTIONS-SIGNAL-SWEEP-STATUS.md    # this file = source of truth
```

### B) Check what's running

```bash
# Any background python sweeps?
ps -ef | grep python | grep short_options
# Check incremental CSV output:
tail -n 20 research/short_options_sweep/phase1_signals.csv
tail -n 20 research/short_options_sweep/phase2_exits.csv
```

### C) Resume Phase 1 (signal logger)

The Phase 1 script (TBD path: `research/short_options_sweep/run_phase1.py`)
**must skip already-completed (date × strategy × variant) cells** by
reading existing CSV first. Script template lives in section 7 below.

```bash
python research/short_options_sweep/run_phase1.py
# Outputs append to phase1_signals.csv incrementally — safe to ctrl-C and resume
```

### D) Resume Phase 2 (exit replay)

Phase 2 reads phase1_signals.csv + phase1_trajectories.csv, applies each
exit policy, writes phase2_exits.csv. Pure post-processing — safe to
restart.

```bash
python research/short_options_sweep/run_phase2.py
```

### E) Files NOT to touch during a sweep

- `research/short_options_sweep/phase1_signals.csv` (being appended)
- `research/short_options_sweep/phase1_trajectories.csv` (being appended)
- Any `*.lock` heartbeat files in that folder

---

## 7. Final aggregation

When Phase 3 completes, the deliverables are:

1. `research/short_options_sweep/RESULTS.md` — top configurations per
   strategy with summary stats and Arun's commentary slots.
2. `research/short_options_sweep/heatmaps/*.png` — visual matrix of
   `(strategy × exit_policy)` win-rates and net-pts.
3. Recommendation in this STATUS doc on which signals (if any) graduate
   to Phase 4 (option-premium-level backtest with real chain data).

Ranking criterion: `mean(net_pts) - 0.5 * std(net_pts)`, tie-break by
`% of signals with MAE_against ≤ SL_threshold` (proxy for stop-out rate
in live).
