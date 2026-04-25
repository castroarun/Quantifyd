# Intraday MA Crossover — Findings (NEGATIVE — system DOA)

**Status:** Killed. No viable cell across 32 (TF × variant) combinations, no
profitable stock in any cell.

## Sweep tested

- **Universe:** 15 non-ORB liquid F&O stocks (banks, IT, auto, consumer,
  metals, infra) — 2024-03-18 to 2026-03-12
- **Signal TFs:** 5-min, 10-min, 15-min, 30-min (all resampled from 5-min DB)
- **EMA pairs (per TF):** 5m=20/50, 10m=10/30, 15m=8/21, 30m=5/13. Plus a
  faster pair variant for whipsaw stress test.
- **Filter variants:** baseline, +VWAP, +HTF (60-min EMA21 slope), +RSI50,
  +CPR (prev-day pivot), +BB(20,2) middle, all-confluence, fast-pair
- **Trade rules:** entry next bar open, 1×ATR stop, 1.5R target, reverse-cross
  exit, time stop (24/12/8/5 bars per TF), EOD 15:15
- **Sizing:** Rs 2,500 risk / trade, Rs 3L cap, 0.15% round-trip costs

**32 backtests, ~67,000 trades total.**

## Headline (sorted by Sharpe — least bad first)

| TF | Variant | Trades | WR% | PF | Net P&L | Sharpe | MaxDD% |
|---|---|---:|---:|---:|---:|---:|---:|
| 30m | confluence | 1587 | 36.9 | 0.53 | −Rs 7.2 L | −7.15 | 246% |
| 30m | htf | 1680 | 36.0 | 0.52 | −Rs 7.8 L | −7.55 | 267% |
| 30m | vwap | 2576 | 36.7 | 0.55 | −Rs 11.0 L | −8.44 | 372% |
| 15m | confluence | 1539 | 37.2 | 0.50 | −Rs 6.6 L | −9.14 | 222% |
| ...all 32 cells negative... |
| 5m | fast_pair | 5386 | 39.6 | 0.39 | **−Rs 22.7 L** | **−20.80** | 757% |

**Best PF:** 0.55 (10m vwap). Need >1.0. **Worst:** 0.36 (5m baseline).

## Per-stock check — no rescue available

For both top variants (30m confluence, 30m htf), **all 15 stocks are
negative**. Best stock (JINDALSTEL on 30m htf) PF 0.81 — still loses money.
Unlike the volume-breakout sweep where curating the universe rescued the
strategy, here the universe is uniformly bad.

## Filter signal — what we did learn

Filters DO help, just not enough:
- HTF / confluence cut trade count by ~50% and reduce loss by ~40%
- Fast pair (9/21) is consistently worst across all TFs — confirms whipsaw
  hypothesis
- 30-min TF is least-bad; 5-min is worst — slower TFs filter noise

But the *base signal* has no edge. Filters can amplify good signal or attenuate
bad signal; they can't manufacture edge that isn't there.

## Why it fails

1. **WR stuck at 36-40%** across all 32 cells. At 1:1.5 R:R, you need WR > 40%
   to clear costs. We hover right at break-even before slippage, then costs
   tip it deeply negative.
2. **MA crossovers fire constantly** in choppy markets — wins are small (1.5R
   target hits), losses are bigger than 1R (slippage on stops, gap-throughs).
3. **Indian cash intraday is structurally non-trending at 5-30 min scale.**
   What ORB exploits is the *opening-range* structure (specific time-of-day
   compression+release), not generic intraday trend.
4. **0.15% round-trip cost** on 6700+ trades / variant ate any edge that
   might have existed.

## Decision

Kill MA-crossover-as-primary-signal. The pattern is now clear across:
- Research 13 (VWAP mean-reversion): dead
- Research 15 (MA crossover any TF): dead
- Research 14 (Volume breakout): viable on curated universe

**The pattern:** intraday systems on Indian cash equities work when they
exploit a *specific structural setup* (opening range, volume confirmation,
compression breakout). Generic price-MA-derived directional systems do not.

## What might be worth trying next (not in this study)

- **Range-compression breakout** (Bollinger Band squeeze → expansion entry on
  volume confirmation). Different signal entirely from MA cross.
- **Opening-15-min reversal** (fade the first 15-min direction if it stalls
  before 11:00). Inverse of ORB, narrowly time-windowed.
- **Pair trading on F&O futures** (originally proposed) — completely different
  edge family, market-neutral.

## Artifacts

- `scripts/run_intraday_ma_crossover.py` — sweep runner (--tf flag selects TF)
- `results/tf_5min/`, `tf_10min/`, `tf_15min/`, `tf_30min/` — per-TF outputs
- `results/summary_all_tfs.csv` — 32-row consolidated summary
- `SWEEP-STATUS.md` — live status during the run (now historical)
- `logs/run_*min.log` — full stdout per run
