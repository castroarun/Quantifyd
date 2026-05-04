# Master/Child SuperTrend on NIFTY & BANKNIFTY — 15/30/60-min Sweep

**STATUS: DONE** — see [results/RESULTS.md](results/RESULTS.md)

## 1. The Ask

**What the user asked (verbatim):**
> Find the best SuperTrend that I can rely on for trend-seeking, not point-wise winners — I'll be selling options and managing. Master ST (MST) defines the bias for a debit spread (biweekly/weekly). A Child ST (CST) — or some indicator/levels (stochastic, bands, RSI, pivots) — defines the hedge that converts the debit spread into a condor (contra credit spread). Want an ALWAYS-ON system. Assess NIFTY vs BNF and recommend.

**What we are actually testing:**
1. **MST stickiness:** which (timeframe, ATR period, multiplier) SuperTrend cell on NIFTY50 and BANKNIFTY produces the **stickiest, lowest-whipsaw trend signal** at the 30-min anchor (with 15-min and 60-min as comparators), measured by avg trend duration, flips/year, and %-bars-in-dominant-direction — *not* point P&L.
2. **CST / hedge trigger:** within an active MST trend, which counter-signal family (shorter SuperTrend, Stochastic, RSI, Bollinger Band tag) most cleanly flags the points where a hedge leg should be opened (i.e. where adverse excursion begins inside an MST trend).
3. **NIFTY vs BNF head-to-head:** which underlying offers cleaner MST signals and clearer CST triggers, given identical mechanics.

We are **not** building or backtesting an options system. No options pricing, no IV, no Greeks. Underlying-only signal-quality assessment.

## 2. The Base — what is being tested

### Universe
- NIFTY50 (index)
- BANKNIFTY (index)

### Period
- 30-min and 15-min: from `2024-03-01` (first 5-min bar resampled) to `2026-03-25`
- 60-min: from `2024-03-19` (first 60-min bar) to `2026-03-19`
- Daily: NOT in scope (too slow for weekly/biweekly options)
- Rationale: ~2 years of intraday data is sufficient to evaluate trend stickiness across multiple regimes (2024 election, 2024 Q4 correction, 2025 recovery, 2025-26 range).

### MST signal definition (canonical SuperTrend)
- HL2 = (high + low) / 2
- ATR(period) on True Range
- Upper Band = HL2 + multiplier × ATR
- Lower Band = HL2 − multiplier × ATR
- Direction flips when close crosses the active band
- Final UpperBand / LowerBand follow the standard non-decreasing/non-increasing rule

### Resampling rules (5-min → 30-min / 15-min)
- Sessions: 09:15 IST onwards
- 30-min buckets: 09:15, 09:45, 10:15, …, 15:15 (last partial bucket 15:15-15:30 included)
- 15-min buckets: 09:15, 09:30, …, 15:15
- OHLC aggregated; volume summed (volume not used in this study)

### MST grid axes
| Axis | Values | Count |
|---|---|---|
| Underlying | NIFTY50, BANKNIFTY | 2 |
| Timeframe | 15min, 30min, 60min | 3 |
| ATR period | 7, 10, 14, 21, 30, 50 | 6 |
| Multiplier | 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0 | 7 |

Total MST cells: **2 × 3 × 6 × 7 = 252**.

### Stickiness metrics (per cell)
1. `n_flips` — total direction flips over the period
2. `flips_per_month` — annualised flip rate
3. `avg_trend_bars` — mean bars between flips
4. `avg_trend_calendar_days` — mean calendar days between flips
5. `median_trend_calendar_days`
6. `pct_bars_long` — % of bars in long trend
7. `pct_bars_short`
8. `dominant_direction_pct` — max(long, short)
9. `mean_max_adverse_excursion_pts` — within-trend MAE (points). Tells us how far price moves *against* the active trend — informs CST hedge-leg trigger sizing.
10. `mean_max_favorable_excursion_pts`
11. `mfe_mae_ratio` — risk-reward of trend signal
12. `stuck_trend_score` — composite (defined in script): rewards long avg trend duration, penalises high flip count, rewards high dominant-direction %
13. `weekly_flip_alignment` — fraction of trends that span at least one full weekly options cycle (Thu–Thu)

### Composite ranking — `stuck_trend_score`

User preference (2026-05-04): **trend quality between flips matters more than flip count**.
Reweighted accordingly:

```
stuck_trend_score =
    0.40 × normalize(mfe_mae_ratio)                   # quality of each trend = primary
  + 0.25 × normalize(avg_trend_calendar_days)         # must still last ~1 week
  + 0.20 × normalize(dominant_direction_pct)          # clean direction, less chop
  + 0.15 × normalize(1 / flips_per_month)             # penalise whipsaw, but only lightly
```
…where each component is min-max normalised across all 252 cells before weighting. Top score wins.

### Quality gates (any cell must clear, else excluded)
- `flips_per_month <= 12` (relaxed — user is OK with more flips if trend quality is better)
- `flips_per_month >= 0.5` (else trend is so slow it never triggers a setup)
- `avg_trend_calendar_days >= 3` (must roughly outlast a weekly cycle on average)
- `n_flips >= 6` (need enough flips for stats to be meaningful)

### CST evaluation (Phase B)
Locked to top-2 MST cells per index from Phase A. For each MST trend segment, evaluate four CST families:
1. **Shorter SuperTrend** — same TF, period ∈ {3, 5, 7}, multiplier ∈ {1.0, 1.5, 2.0}
2. **Stochastic %K/%D** — (5,3,3), (14,3,3); trigger on contra-direction extreme cross
3. **RSI(14)** — overbought (>70) or oversold (<30) reversal in counter-MST direction
4. **Bollinger Bands** — (20, 2.0); tag of opposite band + reversal candle

CST scoring: for each MST trend, mark the maximum adverse excursion point. Score each CST family by:
- `lead_time_bars` — bars between CST trigger and MAE peak (positive = CST leads, negative = lags)
- `false_positive_rate` — CST triggers that did NOT precede a meaningful MAE
- `mae_capture_pct` — average % of MAE drawdown the CST would have avoided

## 3. Plan

### Phase A — MST sweep (252 cells)

| Step | Estimated time |
|---|---|
| Resample 5-min → 15-min, 30-min for both indices | <1 min |
| Compute SuperTrend for 252 cells | ~3-5 min |
| Compute stickiness metrics per cell | ~2 min |
| Apply quality gates + composite ranking | <1 min |
| Write `mst_ranking.csv` and top-10 per index | instant |

### Phase B — CST evaluation (against top-2 MSTs per index)

| Step | Estimated time |
|---|---|
| Extract MST trend segments for top-2 MSTs (4 segment-sets) | <1 min |
| Generate CST signals for each family within each MST segment | ~2 min |
| Score CST families on lead-time / false-positive / MAE-capture | <1 min |
| Write `cst_evaluation.csv` | instant |

Total wall time: **< 15 min** for the entire study.

### Phase C — write-up
`results/RESULTS.md` with:
- Top-3 MSTs per index, side-by-side
- NIFTY vs BNF head-to-head
- Best CST family per chosen MST
- Recommendation: which underlying, which MST, which CST/indicator

## 4. Status (live running log)

| Date/time IST | Event | Notes |
|---|---|---|
| 2026-05-04 | STATUS doc written | Plan locked; user confirmed 30-min anchor + open to other TFs |
| 2026-05-04 | User clarified: trend quality > flip count | Composite reweighted to 0.40·MFE/MAE + 0.25·days + 0.20·dom + 0.15·1/flips |
| 2026-05-04 | Phase A launched | 252 cells |
| 2026-05-04 | Phase A done (381s) | All 252 cells written; 60-min cells dominate global score, 30-min is the practical anchor |
| 2026-05-04 | Phase B launched | 4 MST winners × 13 CST configs = 52 evals |
| 2026-05-04 | Phase B done | Stochastic(14,3,3) OB80/OS20 wins consistently across all 4 MSTs |
| 2026-05-04 | RESULTS.md written | Recommendation: NIFTY 30-min ST(p21,m5.0) + Stoch(14,3,3) CST |
| 2026-05-04 | Phase A-prime: break-of-extreme entry sweep (252 cells, 403s) | MFE/MAE 2.13 → 3.62 (+70%) at cost of ~2hr lag and 6% flips filtered. Edge confirmed. |
| 2026-05-04 | RESULTS.md updated with breakout findings | Recommendation unchanged: stay with p21,m5.0; layer break-of-extreme entry on top |
| 2026-05-04 | Design doc written | `docs/Design/MST-INDEX-STRATEGY-DESIGN.md` — Claude-Code-readable spec for `/app/mst` (signals only, no order placement in Phase 1) |
| 2026-05-04 | CST 'exit-zone' variant tested | Rejected: lead-time turns sharply negative (+8.8 → −18 bars on NIFTY 30m), coverage halves. Original rule confirmed correct. |

## 5. Crash Recovery

If Claude disappears mid-run, the user can:

1. **Check what finished:**
   ```powershell
   Get-ChildItem research\35_nifty_bnf_master_child_supertrend\results
   # mst_ranking.csv present + complete (504 rows incl. header) = Phase A done
   # cst_evaluation.csv present = Phase B done
   ```
2. **Re-run Phase A from scratch (idempotent):**
   ```powershell
   python research\35_nifty_bnf_master_child_supertrend\scripts\run_mst_sweep.py
   ```
3. **Re-run Phase B (requires Phase A output):**
   ```powershell
   python research\35_nifty_bnf_master_child_supertrend\scripts\run_cst_evaluation.py
   ```
4. **DO NOT** rerun the data download — all data is local in `backtest_data/market_data.db`. Sweep reads from there.
5. **Files safe to inspect:** all under `research/35_nifty_bnf_master_child_supertrend/results/`.

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `MST_CST_INDEX_15_30_60MIN_SWEEP_STATUS.md` | This doc | yes |
| `scripts/supertrend.py` | SuperTrend calc + helper indicators | yes |
| `scripts/run_mst_sweep.py` | Phase A runner (252 cells) | yes |
| `scripts/run_cst_evaluation.py` | Phase B runner | yes |
| `results/mst_ranking.csv` | Per-cell stickiness metrics + composite score | yes (small) |
| `results/mst_top10_NIFTY50.csv` | Top-10 MSTs for NIFTY | yes |
| `results/mst_top10_BANKNIFTY.csv` | Top-10 MSTs for BANKNIFTY | yes |
| `results/cst_evaluation.csv` | CST family scores per MST winner | yes |
| `results/RESULTS.md` | Final findings + recommendation | yes |

## 7. Findings

Full writeup: [results/RESULTS.md](results/RESULTS.md). One-line summary:

**Use NIFTY50, MST = SuperTrend(ATR=21, mult=5.0) on 30-min, CST = Stochastic(14,3,3) OB80/OS20 on 30-min.**

- NIFTY beats BANKNIFTY at 30-min: MFE/MAE 2.13 vs 1.77, weekly-cycle survival 61% vs 53%
- 30-min is the right anchor: 60-min has higher composite score but its CSTs lag the MAE (negative lead time) — 30-min CSTs lead by 8.8 bars (~4.4 hours), enough to leg in the contra credit spread
- Stochastic(14,3,3) wins over RSI/BB/shorter-ST as CST because it's the only family that consistently leads the MAE peak rather than confirming it
- ~3.2 MST flips/month, avg trend ~8.8 calendar days, ~61% of trends fully span a weekly Thu-Thu cycle
