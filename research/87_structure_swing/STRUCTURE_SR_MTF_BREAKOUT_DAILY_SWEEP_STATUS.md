# Structure-Based Swing Systems — S/R Levels, Chart Patterns, MTF Confluence & Breakouts (Daily, F&O Universe)

STATUS: DONE — NO EDGE absolute; borderline relative short SIGNAL failed Val (see results/RESULTS.md)

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
