# 3-Bar Reversal — Findings (NEGATIVE)

**Status:** Killed. No viable cell across 17 (signal × filter × SL/target) configurations.

## Pattern tested

### LONG setup
- **Bar 1:** strong GREEN — body ≥60% of (high-low), close > open
- **Bar 2:** strong RED — body ≥60% of (high-low), close < open
- **Bar 3:** strong GREEN, body ≥60%, AND `close > max(bar1.high, bar2.high)`

### SHORT setup (mirrored)
- **Bar 1:** strong RED · **Bar 2:** strong GREEN · **Bar 3:** strong RED
  closing below `min(bar1.low, bar2.low)`

### MACD confirmation (2 variants)
- `macd_cross` — MACD line just crossed above (long) / below (short) signal line
- `macd_zero` — MACD line is positive (long) / negative (short)

### Trade rules
- Entry next bar's open after pattern bar 3 closes
- Initial SL: bar 1's low (long) / high (short). Variant: bar 2.
- Target: 2× SL distance (R:R 1:2). Variants tested: 1.5R, 2R, 3R.
- Time stop: 24 bars. EOD safety: 15:15.
- Sizing: Rs 2,500 risk per trade, Rs 3L cap, 0.15% round-trip costs.
- Universe: 15 non-ORB liquid F&O stocks (research/16 set).
- Period: 2024-03-18 → 2026-03-12 (~500 trading days).

## Phase 1 — signal × TF × MACD (6 cells)

| Variant | Trades | WR% | PF | Net P&L | Sharpe |
|---|---:|---:|---:|---:|---:|
| 5min_macd_cross | 134 | 33.6 | 0.48 | −Rs 61K | −5.63 |
| 5min_macd_zero | 594 | 37.4 | 0.59 | −Rs 189K | −4.69 |
| 10min_macd_cross | 46 | 37.0 | 0.43 | −Rs 26K | −6.89 |
| 10min_macd_zero | 236 | 36.4 | 0.53 | −Rs 109K | −4.78 |
| 15min_macd_cross | 23 | 39.1 | 0.46 | −Rs 11K | −5.06 |
| **15min_macd_zero** (best) | 181 | 34.8 | **0.59** | −Rs 72K | **−3.62** |

`macd_cross` (true crossover) is too restrictive — 23-134 trades only across 2 years.
`macd_zero` (above/below 0) is more permissive and produces better PF in every TF.
**15-min wins on Sharpe** (slower bars filter noise as expected).

## Phase 2 — confluence filters on best (15min_macd_zero) — 7 cells

| Filter added | Trades | WR% | PF |
|---|---:|---:|---:|
| **none** (baseline) | 181 | 34.8 | 0.59 |
| **bb** (close > BB middle) | 176 | 35.2 | 0.59 |
| cpr (close vs prior pivot) | 179 | 34.6 | 0.59 |
| rsi50 | 181 | 34.8 | 0.59 |
| confluence (CPR+HTF+RSI50) | 156 | 32.7 | 0.54 |
| htf (60m EMA21 slope) | 157 | 32.5 | 0.53 |
| stoch (cross in OS/OB) | 2 | 0.0 | 0.00 |

**No filter moves the needle.** PF locked at 0.59 for unrestricted filters.
HTF and confluence reduce trade count without lifting WR — they over-filter
without finding edge. Stochastic crossover in OS/OB zone is too restrictive
(2 hits in 2 years).

## Phase 3 — SL/target tweaks on best (15min_macd_zero_bb)

| Variant | Trades | WR% | PF | Net P&L |
|---|---:|---:|---:|---:|
| bar1 SL + 2R target | 176 | 35.2 | 0.59 | −Rs 70K |
| bar2 SL + 2R target | 176 | 34.1 | 0.61 | −Rs 71K |
| **bar1 SL + 1.5R target** (best) | 176 | **38.1** | **0.65** | **−Rs 59K** |
| bar1 SL + 3R target | 176 | 34.7 | 0.58 | −Rs 72K |

Lower target lifts WR (38.1%) and PF (0.65) by trading R:R for hit rate.
Still negative — costs eat the thin positive expectancy.

## Why it fails

1. **WR structurally locked at 33-38% across all 17 cells.** No variant
   broke 40%, even with tightest filters or tightest targets.
2. **At 35% WR with 1:2 R:R**, expectancy = 0.35×2 − 0.65×1 = +0.05R
   per trade — almost exactly break-even before costs.
3. **0.15% round-trip costs** consume that thin margin. Slippage in
   real markets typically wider on 5-min volatile bars.
4. **The 3-bar pattern is rare AND not strongly predictive.** When the
   pattern fires, follow-through is essentially coin-flip with mild
   directional bias — same finding as BB squeeze fires (research/16).

## Cumulative score across the research arc

| Research | Strategy | Best PF | Status |
|---|---|---:|---|
| 13 | VWAP fade (counter-trend) | 0.41 | Dead |
| 14 | Volume breakout intraday (walk-forward) | 0.79 | Dead OOS |
| 15 | MA crossover (32 cells) | 0.55 | Dead |
| 16 | BB squeeze + 30 cells of variants | 0.65 | Dead |
| **18** | **3-bar reversal + MACD + 17 cells** | **0.65** | **Dead** |
| 17 | EOD swing momentum (daily bars) | **1.44 OOS** | **PASS** |

**5 clean negative results on intraday directional signals on Indian cash
5-min/10-min/15-min bars.** The single positive edge is the EOD daily-bar
swing momentum scanner (research/17) — which works because the signal is
anchored to a *structural* multi-day momentum factor, not to an intraday
price-action pattern.

## Decision

Stop hunting for intraday directional patterns. The data has been
consistent across 5 different strategy families: WR ceiling on Indian
cash intraday bars sits at ~37-40% for any directional pattern, which
combined with realistic costs cannot produce positive expectancy.

The two known-positive intraday systems in this codebase remain:
- **ORB** (live, anchored to opening-range structure)
- **EOD swing scanner from research/17** (daily bars, anchored to
  multi-day momentum)

## Artifacts

- `scripts/run_3bar.py` — 3-phase backtest engine
- `results/phase{1,2,3}_summary.csv` — per-phase variant metrics
- `results/phase{1,2,3}_trades.csv` — per-trade logs
- `results/best_phase{1,2}.json` — auto-detected best variant pointers
- `3BAR-STATUS.md` — live status during runs (now historical)
- `logs/phase{1,2,3}.log` — full run output
