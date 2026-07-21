# Structure-Based Swing Systems — S/R Levels, Chart Patterns, MTF Confluence & Breakouts (Daily, F&O Universe)

STATUS: PHASE 1b RUNNING — 30-min intraday pass (daily pass: NO EDGE, results/RESULTS.md)

## 1. The Ask

**What you asked:** "do a fresh comprehensive research and optimization to find
the best swing trading systems... look not only at candle patterns but also
chart patterns, resistance/support, multi-timeframe etc. No bias or reference
to past systems. Also look at identifying breakout patterns."

**What we're actually testing:** Across the F&O universe (shortable both
ways) on DAILY bars 2005→2017 in-sample, do any of 8 pre-registered
STRUCTURE-based signal families — pivot-level breakouts, support/resistance
bounces, volatility-contraction breakouts, flag continuations, double
bottoms/tops, weekly-daily multi-timeframe confluence, 52-week-high volume
breakouts, inside-bar/NR7 compressions — produce positive NET per-trade
returns (t ≥ 2.5) that then survive validation (2018–2022H1)?

**Bias controls:** Hypotheses drawn from classical technical-analysis
structure literature, NOT from any prior study in this repo. Both LONG and
SHORT tested for every family. Only the *methodology* is inherited: cost
model, chronological splits, stage gates, multiple-testing ledger.

## 2. The Base

- **Universe:** current F&O stock list (~86 names) — shortable via futures.
  Survivorship caveat: current constituents (stated on any positive result).
- **Data:** `market_data_unified` day bars, warmup from 2000.
- **Splits (daily convention):** IS 2005-01-01→2017-12-31 · Val
  2018-01-01→2022-06-30 · **OOS 2022-07-01+ QUARANTINED** (one look, only
  with user authorization, only for a full G1+Val survivor).
- **Fills:** signal on close of bar t → enter at OPEN of t+1; exit at OPEN
  of t+1+h (fixed-horizon screening, h ∈ {3,5,10,15} sessions). Non-overlap
  per (name, cell). Causal-only: pivots confirmed w bars after they form.
- **Costs:** futures-proxy 10bps round-trip (5bp/side) NET; gross also kept.
- **Success criterion (G1):** net > 0 AND pooled t ≥ 2.5 AND ≥55% of names
  positive AND ≥60% of years positive. Survivors → Val: net > 0, t ≥ 2.0,
  no sign flip vs IS.

## 3. Plan — families & grid (104 cells, all pre-registered)

| Family | Signal | Axes | Cells |
|---|---|---|---|
| SR1 pivot-level break | close crosses last confirmed pivot-high (mirror: pivot-low) | w∈{3,5} × h4 × dir2 | 16 |
| SR2 level bounce | ≥2-touch pivot-low cluster (1% tol), low tags level (≤0.5%), close holds above (mirror at resistance) | h4 × dir2 | 8 |
| CP1 contraction break | 20d range-width in bottom p% of 252d, close breaks consolidation extreme | p∈{10,20} × h4 × dir2 | 16 |
| CP2 flag | ≥8%/5d impulse, ≤5%-deep drift 3-8 bars, break of flag extreme | h4 × dir2 | 8 |
| CP3 double bottom/top | two pivot lows within 1.5%, ≥10 bars apart, neckline ≥4%, close breaks neckline | h4 × dir2 | 8 |
| MTF1 weekly×daily | weekly close > 30w SMA (mirror <) AND daily close breaks {20,55}d extreme | dtrig2 × h4 × dir2 | 16 |
| BR1 52w-high volume break | close > 252d high (mirror low) AND vol ≥ {1.5,2}× 20d avg | v2 × h4 × dir2 | 16 |
| BR2 compression break | inside-bar → mother-bar break; NR7 → range break | sub2 × h4 × dir2 | 16 |

Multiple-testing ledger for this program: **104 cells** (program running
total updated at aggregation). Phase 1b (pre-registered, run only if ≥1
family passes G1): 60-min intraday timing layer on the passing family
(daily setup → intraday entry), grid ≤16 cells.

## 4. Status / event log

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-21 ~15:55 IST | Pre-registration written; runner authored | 104 cells |
| 2026-07-21 (launch row added by runner) | G1 IS screen launched on VPS | log: results/g1_screen.log |

## 5. Crash recovery

- Progress: `tail -5 research/87_structure_swing/results/g1_screen.log`;
  completed cells: `wc -l research/87_structure_swing/results/g1_screen.csv`
- Alive? `ps -eo args | grep '[r]un_g1_daily_screen'`
- Resume: rerun `venv/bin/python3 research/87_structure_swing/scripts/run_g1_daily_screen.py`
  — it skips cells already in the CSV.
- Do NOT touch: `backtest_data/market_data.db` (read-only here).

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `scripts/run_g1_daily_screen.py` | detectors + fixed-horizon sim + pooled stats | yes |
| `results/g1_screen.csv` | per-cell aggregates (incremental) | yes |
| `results/g1_screen.log` | run log | yes (small) |
| `results/RESULTS.md` | final verdict | yes (after) |

## 7. Findings

(populated during/after the run)


## PHASE 1b PRE-REGISTRATION (2026-07-21): 30-min intraday pass

Same 8 families, identical bar-parameterized detectors now operating on
30-MIN bars (structure at intraday scale), F&O universe, 5-min resampled.
MTF1 becomes daily-trend (30-bar daily SMA) x 30m-breakout — true MTF.
- Splits (intraday convention): IS 2015-02..2021-09 · Val 2021-10..2023-12
  · OOS 2024+ QUARANTINED.
- Horizons in bars: {13, 39, 65, 130} ~= 1/3/5/10 sessions.
- Costs 10bps RT. Controls IN THE PRIMARY PASS: per-cell excess vs
  unconditional drift AND date-matched universe mean; primary gate =
  rel_net>0 & t_rel>=2.5 & names+>=0.55 (absolute net reported alongside).
- Cells: 13 param-variants x 4 h x 2 dir = 104. Ledger r/87+88: 238.
- 60m pass contingent on a 30m family passing.
Runner: scripts/run_g2_intraday_screen.py · log results/g2_intraday.log

## PHASE 1c PRE-REGISTRATION (2026-07-21): explicit chart-pattern geometry (daily)

Three classical patterns not covered by the statistical families, coded as
explicit pivot-sequence geometry, daily bars, same universe/splits/costs,
scored date-matched (primary) + absolute (reported):
- **CP4 head-and-shoulders** (short) / inverse (long): pivot highs P1<P2>P3,
  shoulders within 3%, neckline = min of intervening pivot lows; signal on
  close breaking neckline. 8 cells.
- **CP5 triangles/wedges**: >=2 successively lower pivot highs AND >=2
  successively higher pivot lows (sym); rising/falling wedge sub-variants by
  slope signs; break beyond the last pivot bound sets direction. 3 subs x
  h4 x dir2 = 24 cells.
- **CP6 cup-with-handle** (long) / inverted (short): recovery to within 3%
  of a >=30-bar-old peak after >=15% base depth, handle <=5% for 3-10 bars,
  break of peak. 8 cells.
Gate: as phase 1 (rel_net>0, t_rel>=2.5, names+>=0.55; Val same as before).
Cells +40 → program ledger 278. Runner: scripts/run_g3_patterns2.py
