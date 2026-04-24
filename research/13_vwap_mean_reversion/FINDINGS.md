# VWAP Mean-Reversion — Findings (NEGATIVE RESULT)

**Status:** Killed. No viable configuration found across 3 iterations.

## Thesis tested

When an intraday price is >N×ATR stretched from its cumulative VWAP in the
11:30–14:00 window, a fade toward VWAP should have positive expectancy.
Designed explicitly as a counter-ORB system (same universe, opposite trade).

## Three iterations, all losing

| Version | Universe | Spec changes | Portfolio PF | Net P&L (Rs, 2yr) |
|---|---|---|---:|---:|
| v1 | 15 ORB high-beta | Baseline (1 ATR stop / 1 ATR target) | 0.17 avg | −Rs 27.5 L |
| v2 | 14 defensives | + Stochastic(14,3,3), RSI(14) reversal filters | 0.13–0.14 | −Rs 7.5 to −Rs 24 L |
| v3 | 14 defensives | Target = VWAP (R:R ~1:2.5), tighter entry 2.0–3.0σ | 0.29–0.41 | −Rs 2.9 to −Rs 7 L |

Best single cell: COALINDIA long-only at 2.5σ = +Rs 4,212 on 43 trades.
Single stock with 43 trades — **noise, not signal**.

## Why it fails

1. **Win rate structurally stuck at 20–30%.** Across 8,000+ trades in 9 variants,
   WR never crossed 33%. At 1:1 R:R this gives −0.40R expectancy.
2. **VWAP target doesn't save it.** Raising target to VWAP (R:R ~1:2.5) cut
   WR further to 23–26% as price reached stop before VWAP more often.
3. **Filters cut volume, not edge.** Stochastic + RSI reduced trade count by
   70% but PF only improved from 0.13 → 0.41. Filters were rejecting mostly
   random-noise trades, not systematic losers.
4. **Indian cash intraday structure:** high-beta ORB stocks trend
   post-10:30 (why ORB wins). Defensive stocks range but the ranges are small
   relative to 5-min noise — stops trigger on wicks before mean-reversion plays
   out. 0.15% round-trip cost eats whatever sliver of edge remains.

## What would have rescued it (not tried)

- **15-min timeframe** instead of 5-min — cleaner signal, less whipsaw. Not
  stored in our DB at that resolution; would need to resample.
- **Virgin CPR proximity as entry gate** (user's third suggestion) — only take
  fade when price is at an untouched previous-week pivot level. Didn't test
  because the base rate across 3 variants was already so poor that this
  filter alone was unlikely to flip PF from 0.4 to >1.0.
- **Different thesis entirely** — e.g., "first-90-min opening reversal" which
  is closer to an opening-range fade than a midday mean-revert.

## Decision

Move on. Next intraday system to prototype: **volume breakout** (in
`research/14_volume_breakout/`).

## Artifacts

- `scripts/run_vwap_mr_backtest.py` — v1
- `scripts/run_vwap_mr_v2.py` — v2 with Stoch + RSI filters
- `scripts/run_vwap_mr_v3.py` — v3 with VWAP target + direction variants
- `results/trades.csv`, `trades_v2.csv` — per-trade logs
- `results/summary.csv`, `summary_v2.csv`, `summary_v3.csv` — portfolio metrics
- `logs/run_v1.log`, `run_v2.log`, `run_v3.log` — full run output
