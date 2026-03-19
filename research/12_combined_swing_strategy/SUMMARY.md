# 12: Combined Swing Trading Strategy (EMA Trend + RSI Mean Reversion)

**Period:** March 17, 2026
**Backtest:** Jan 2018 - Nov 2025 (~7.8 years), 78 F&O stocks, 60-minute data
**Status:** Validated trading edge — to be deployed with F&O futures lots

## Strategy Overview

Two uncorrelated swing trading edges on 60-minute intraday data, designed for **futures lot execution** on NSE F&O stocks. This is a **trading system** (signal-driven, leveraged) not an investment strategy.

### Strategy Components

| Strategy | Entry | Exit | Timeframe |
|----------|-------|------|-----------|
| **EMA Trend** | EMA(25) crosses above EMA(60) AND ADX > 20 | EMA(25) crosses below EMA(60) | 60-min bars |
| **RSI Mean Reversion** | RSI(14) < 30 AND price > SMA(200) | RSI > 50 | 60-min bars |
| **NR7 Breakout** | Daily NR7 compression breakout | Hold N days | Daily (DROPPED — drag on risk-adjusted returns) |

### Capital Allocation (Recommended)

- 60% to Trend Following, 40% to Mean Reversion, 0% to Breakout
- Max 20 simultaneous trend positions, 10 mean reversion positions
- 5% max capital per position (i.e. per F&O lot notional)

---

## Recommended Configs

### Best for Cash (No Leverage)

**`E25_60_ADX20_R14_30_P20_10`** — New best from long+short sweep. Highest CAGR with best Calmar.

| Metric | Value |
|--------|-------|
| CAGR | **9.18%** |
| Sharpe | 1.48 |
| Sortino | 1.94 |
| Max Drawdown | 7.40% |
| Calmar | **1.24** |
| Profit Factor | 1.58 |
| Win Rate | 46.1% |
| Total Trades | 3,878 (498/yr) |
| Total PnL | Rs 99.3L on Rs 1Cr |

**Previous best:** E25_60_ADX20_R21_30_P20_10 (8.19% CAGR, Calmar 1.03). New config uses RSI(14) instead of RSI(21) which adds more mean reversion trades with strong edge.

### Best for Futures (3x Leverage)

**`E25_60_ADX20_R14_30_P20_10`** — Same config as cash best. Best CAGR AND best Calmar at 3x.

| Metric | Cash | Futures 3x |
|--------|------|-----------|
| CAGR | 9.18% | **27.5%** |
| Sharpe | 1.48 | 1.48 |
| Sortino | 1.94 | 1.94 |
| Max Drawdown | 7.40% | **22.2%** |
| Calmar | 1.24 | **1.24** |
| Profit Factor | 1.58 | 1.58 |
| Win Rate | 46.1% | 46.1% |
| Total Trades | 3,878 (498/yr) | 3,878 |
| Total PnL | Rs 99.3L | **Rs 297.9L** |
| Margin Required | 100% | ~20-25% |

**Runner-up for conservative futures:** E20_50_ADX25_R14_30_P20_10 — 19.9% CAGR at 3x but only 16.8% MaxDD (vs 22.2%). Better if you want tighter drawdown control.

---

## Top 10 Configs — Cash vs Futures 3x Side by Side

| Rank | Config | PF | Win% | Trades | Cash CAGR | Cash MaxDD | Cash Calmar | Fut 3x CAGR | Fut 3x MaxDD | Fut 3x Calmar |
|------|--------|-----|------|--------|-----------|------------|-------------|-------------|--------------|---------------|
| **NEW** | **E25_60_ADX20_R14_30_P20_10** | 1.58 | 46.1% | 3,878 | **9.18%** | 7.40% | **1.24** | **27.5%** | 22.2% | **1.24** |
| 1 | E20_50_ADX25_R14_30_P20_10 | 1.48 | 48.4% | 3,217 | 6.63% | 5.59% | 1.19 | 19.9% | **16.8%** | 1.19 |
| 2 | E25_60_ADX20_R21_30_P20_10 | 1.63 | 37.3% | 2,765 | 8.19% | 7.95% | 1.03 | 24.6% | 23.9% | 1.03 |
| 3 | E25_60_ADX30_R21_30_P20_10 | 1.75 | 40.2% | 1,163 | 4.79% | **4.90%** | 0.98 | 14.4% | **14.7%** | 0.98 |
| 4 | E25_60_ADX30_R21_30_P15_10 | 1.72 | 39.9% | 1,127 | 5.68% | 5.80% | 0.98 | 17.0% | 17.4% | 0.98 |
| 5 | E25_60_ADX20_R21_30_P15_10 | 1.59 | 37.4% | 2,258 | **8.27%** | 8.85% | 0.93 | **24.8%** | 26.6% | 0.93 |
| 6 | E20_50_ADX25_R14_30_P15_10 | 1.45 | 48.7% | 3,055 | 7.04% | 7.78% | 0.90 | 21.1% | 23.3% | 0.90 |
| 7 | E25_60_ADX30_R21_30_P8_5 | **1.68** | 40.6% | 926 | 5.58% | 6.54% | 0.85 | 16.7% | 19.6% | 0.85 |
| 8 | E25_60_ADX20_R21_30_P8_5 | 1.60 | 37.3% | 1,370 | 6.64% | 8.32% | 0.80 | 19.9% | 25.0% | 0.80 |
| 9 | E20_50_ADX30_R21_30_P20_10 | **1.85** | 42.4% | 1,009 | 4.32% | 5.34% | 0.81 | 13.0% | 16.0% | 0.81 |
| 10 | E20_50_ADX25_R14_30_P8_5 | 1.45 | 50.3% | 2,238 | 6.49% | 7.89% | 0.82 | 19.5% | 23.7% | 0.82 |

### Reading the Table

- **Sharpe, Sortino, Calmar, PF, Win Rate** are identical for cash and futures (they are ratios — leverage scales numerator and denominator equally)
- **CAGR and MaxDD scale linearly** with leverage (3x shown)
- **NEW** is the overall best — highest CAGR (27.5% at 3x) AND best Calmar (1.24). Discovered in Phase 4 long+short sweep
- **Rank 1** is the conservative futures pick — lowest MaxDD at 3x (16.8%) with Calmar 1.19
- **Rank 2** was the previous cash best (8.19%) — now superseded by E25_60_ADX20_R14_30 (9.18%)
- **Rank 9** has the cleanest edge (PF 1.85) but generates only 1,009 trades over 7.8 years

---

## Top 5 by Profit Factor (Trade Quality)

| Rank | Config | PF | Sharpe | Sortino | Cash MaxDD | Fut 3x MaxDD | Win% | Trades |
|------|--------|-----|--------|---------|------------|-------------|------|--------|
| 1 | E20_50_ADX30_R21_30_P20_10 | **1.85** | 1.52 | 2.16 | 5.34% | 16.0% | 42.4% | 1,009 |
| 2 | E20_50_ADX30_R21_30_P15_10 | **1.82** | 1.46 | 2.06 | 6.99% | 21.0% | 42.1% | 997 |
| 3 | E25_60_ADX30_R21_30_P20_10 | **1.75** | 1.47 | 1.99 | 4.90% | 14.7% | 40.2% | 1,163 |
| 4 | E25_60_ADX30_R21_30_P15_10 | **1.72** | 1.40 | 1.90 | 5.80% | 17.4% | 39.9% | 1,127 |
| 5 | E25_60_ADX30_R21_30_P8_5 | **1.68** | 1.27 | 1.70 | 6.54% | 19.6% | 40.6% | 926 |

**Note:** ADX>30 dominates the PF rankings — the stricter trend filter produces fewer but much cleaner trades.

---

## Top 5 by Sharpe Ratio (Risk-Adjusted Return)

| Rank | Config | Sharpe | Sortino | PF | Cash MaxDD | Fut 3x MaxDD | Calmar |
|------|--------|--------|---------|-----|------------|-------------|--------|
| 1 | E20_50_ADX30_R21_30_P20_10 | **1.52** | 2.16 | 1.85 | 5.34% | 16.0% | 0.81 |
| 2 | E25_60_ADX30_R21_30_P20_10 | **1.47** | 1.99 | 1.75 | 4.90% | 14.7% | 0.98 |
| 3 | E20_50_ADX30_R21_30_P15_10 | **1.46** | 2.06 | 1.82 | 6.99% | 21.0% | 0.75 |
| 4 | E25_60_ADX20_R21_30_P20_10 | **1.40** | 1.82 | 1.63 | 7.95% | 23.9% | 1.03 |
| 5 | E25_60_ADX30_R21_30_P15_10 | **1.40** | 1.90 | 1.72 | 5.80% | 17.4% | 0.98 |

---

## Per-Strategy Breakdown

### EMA Trend Following (Primary Edge)

| Config | Trades | PF | Cash PnL | Fut 3x PnL | Notes |
|--------|--------|-----|----------|-----------|-------|
| EMA(25/60) ADX>20 | 2,622 | **1.60** | 79.3L | 237.9L | Best CAGR — slower EMA catches bigger moves |
| EMA(25/60) ADX>30 | 1,020 | **1.70** | 38.1L | 114.3L | Best PF — stricter filter, fewer but cleaner trades |
| EMA(20/50) ADX>30 | 866 | **1.78** | 33.1L | 99.4L | Highest per-trade PF, fewer trades |
| EMA(20/50) ADX>25 | 1,961 | **1.48** | 45.6L | 136.8L | Good balance of volume and quality |
| EMA(9/21) ADX>30 | 1,510 | **1.27** | 13.8L | 41.4L | Fast EMA — too noisy, worst PF |

**Key:** Slower EMAs (20/50, 25/60) consistently outperform faster ones. ADX>30 gives highest PF but fewer trades.

### RSI Mean Reversion (Secondary Edge)

| Config | Trades | PF | Cash PnL | Fut 3x PnL | Notes |
|--------|--------|-----|----------|-----------|-------|
| RSI(21) < 25 | 15 | **49.05** | 2.2L | 6.5L | Very rare signals, extremely high PF |
| RSI(14) < 30 | 1,256 | **1.51** | 19.9L | 59.6L | Sweet spot — enough trades with strong edge |
| RSI(21) < 30 | 143 | **2.48** | 6.2L | 18.7L | High PF but low trade count |
| RSI(21) < 35 | 825 | **1.68** | 19.6L | 58.7L | Looser filter, still profitable |
| RSI(7) < 20 | 1,734 | **0.99** | -0.9L | -2.8L | Too sensitive — no edge |
| RSI(7) < 25 | 2,799 | **0.98** | -2.3L | -7.0L | Negative edge — avoid |

**Key:** RSI(7) is too noisy and destroys edge. RSI(14)<30 with SMA(200) filter is the sweet spot.

---

## Consistency & Robustness

### Across Parameter Space (Cash Metrics)

- **72 of 72 configs profitable** in Phase 3 (all positive total PnL over 7.8 years)
- **PF never drops below 1.21** across all 113 configs tested (excluding breakout-only)
- **Cash MaxDD range: 3.5% to 14.7%** (Futures 3x: 10.5% to 44.1%)
- **Sharpe range: 0.77 to 1.52** — all configs show positive risk-adjusted returns

### Futures 3x Safety Check

| MaxDD Bucket (Fut 3x) | Configs | Verdict |
|------------------------|---------|---------|
| < 15% | 5 configs | Very safe — ADX>30 configs |
| 15% - 20% | 12 configs | Safe — recommended operating range |
| 20% - 25% | 18 configs | Acceptable — needs discipline |
| 25% - 35% | 25 configs | Aggressive — use with smaller position sizes |
| > 35% | 12 configs | Dangerous — fast EMAs + ADX>20, avoid at 3x |

### Structural Findings

| Finding | Evidence |
|---------|----------|
| NR7 Breakout = drag | Every no-breakout config dominates on Calmar. Breakout adds CAGR but MaxDD jumps 2-3x |
| Slower EMAs win | EMA(25/60) and EMA(20/50) consistently top rankings. EMA(9/21) is worst |
| ADX>30 = quality filter | Highest PF (1.75-1.85) but fewer trades (~1,000). ADX>20 gives most trades (~2,700) |
| Stop loss irrelevant >8% | SL8 through SL50 produce identical results — EMA crossover exit handles risk |
| Mean reversion uncorrelated | MR trades fire independently of trend signals — genuine diversification |
| **Shorts = negative edge** | Trend shorts PF 0.70-0.80, MR shorts PF 0.69-0.80. Indian markets have structural long bias. Tested across 5 configs, all destructive |
| RSI(14) > RSI(21) for MR | RSI(14)<30 generates 1,256 trades (PF 1.52) vs RSI(21)<30 only 143 trades (PF 2.48). More trades with decent edge = higher total PnL |

---

## Long+Short Testing (Phase 4)

**Tested:** Adding short signals to both EMA Trend and RSI Mean Reversion. 23 configs across 5 base parameter sets x 4 modes (LONG_ONLY, TREND_LS, MR_LS, BOTH_LS) + RSI short entry threshold variants.

### Verdict: Shorts Destroy the Edge — Stay Long-Only

#### EMA Trend Shorts (Short on downward EMA crossover + ADX filter)

| Config | Long-Only CAGR | +Trend Shorts CAGR | Trend Short PF | Short PnL |
|--------|---------------|---------------------|----------------|-----------|
| E20_50_ADX25 | 6.65% | 2.82% | **0.70** | -Rs 32.9L |
| E25_60_ADX20 | 8.19% | 3.21% | **0.73** | -Rs 31.6L |
| E20_50_ADX30 | 4.32% | 2.65% | **0.72** | -Rs 15.6L |
| E25_60_ADX30 | 4.79% | 3.34% | **0.80** | -Rs 12.3L |
| E25_60_ADX20_R14 | 9.18% | 4.49% | **0.73** | -Rs 32.0L |

**Every trend short has PF < 1.0** — consistent money losers. Indian markets have a structural long bias; shorting EMA crossovers catches too many V-shaped recoveries. MaxDD also jumps significantly (e.g. 7.9% → 16.1% for E25_60_ADX20).

#### RSI Mean Reversion Shorts (Short on RSI>70 + Price<SMA200 — contradiction signal)

| Config | Long-Only CAGR | +MR Shorts CAGR | MR Short PF | Short Trades |
|--------|---------------|-----------------|-------------|-------------|
| E20_50_ADX25 | 6.65% | 5.64% | **0.80** | 887 |
| E25_60_ADX20 | 8.19% | 7.90% | **0.69** | 116 |
| E25_60_ADX20_R14 | 9.18% | 8.23% | **0.79** | 886 |

Less destructive than trend shorts (fewer trades, smaller losses) but still negative edge. The "overbought below SMA200" contradiction doesn't produce reliable short signals.

#### RSI Short Entry Threshold Sensitivity

| RSI Short Entry | CAGR | Short Trades | Short PF | Calmar |
|----------------|------|-------------|----------|--------|
| RSI > 65 (loose) | 4.44% | 1,976 | 0.77 | 0.52 |
| RSI > 70 (default) | 5.64% | 887 | 0.80 | 0.84 |
| RSI > 75 (strict) | 6.24% | 273 | 0.80 | **1.14** |

Stricter threshold = fewer shorts = less damage. But even at RSI>75 (only 273 shorts), CAGR drops from 6.65% to 6.24%. No threshold makes shorts profitable.

#### Both Shorts Combined — Worst Performance

| Config | CAGR | Calmar | PF | MaxDD |
|--------|------|--------|----|-------|
| E20_50_ADX25 BOTH_LS | 1.53% | 0.15 | 1.04 | 9.94% |
| E25_60_ADX20 BOTH_LS | 2.78% | 0.18 | 1.11 | 15.53% |
| E25_60_ADX20_R14 BOTH_LS | 3.16% | 0.25 | 1.09 | 12.74% |

### Why Shorts Don't Work Here

1. **Structural long bias** in Indian equities — GDP growth, inflation, and index construction all favor longs
2. **EMA crossover shorts** catch too many dead-cat bounces and V-recoveries
3. **RSI overbought + below SMA200** is rare AND unreliable — stocks below SMA200 that spike in RSI are often mean-reverting back UP, not continuing down
4. **Slippage asymmetry** — short entries face unfavorable fills, compounding the negative edge

---

## Sweep Phases

1. **Phase 1 (v1):** 9 configs — allocation + strategy params. Abandoned (too slow at ~60s/config recomputing signals each time)
2. **Phase 2 (v2):** 41 configs — precompute signals once, vary allocation (15 combos), position sizing (12), stops (8), max position size (6)
3. **Phase 3 (v3):** 72 configs — vary EMA periods (6 pairs x 3 ADX thresholds x 3 position counts) + RSI settings (7 variants x 3 position counts)
4. **Phase 4 (long+short):** 23 configs — test short signals for EMA trend (downward crossover) and RSI MR (overbought contradiction). **Result: shorts destroy the edge, stay long-only**

## Files

| File | Purpose |
|------|---------|
| `scripts/combined_strategy_backtest.py` | Main engine (~700 lines) — signals, portfolio sim, metrics |
| `scripts/run_combined_sweep.py` | Phase 1 sweep (9 configs, abandoned) |
| `scripts/run_combined_sweep_v2.py` | Phase 2 sweep (41 configs — allocation/sizing/risk) |
| `scripts/run_combined_sweep_v3.py` | Phase 3 sweep (72 configs — EMA/ADX/RSI params) |
| `scripts/research_trend_following.py` | Individual trend strategy research |
| `scripts/research_trend_part2.py` | Trend research continuation |
| `scripts/research_mean_reversion.py` | Individual mean reversion research |
| `scripts/research_breakout_priceaction.py` | Individual breakout research |
| `scripts/research_multiTF_intraday.py` | Multi-timeframe intraday research |
| `scripts/research_multiTF_orb_deep.py` | ORB deep dive (5-min data, PF 1.29) |
| `results/combined_sweep_v2.csv` | Phase 2 full results (41 rows) |
| `results/combined_sweep_v3.csv` | Phase 3 full results (72 rows) |
| `results/combined_sweep_results.csv` | Phase 1 results (9 rows) |
| `results/combined_strategy_results.csv` | Initial baseline result |
| `results/research_results_trend.csv` | Trend strategy variants (individual) |
| `results/research_results_meanrev.csv` | Mean reversion variants (individual) |
| `results/research_results_breakout.csv` | Breakout variants (individual) |
| `results/research_results_multiTF.csv` | Multi-timeframe research |
| `results/research_results_orb_deep.csv` | ORB deep dive on 5-min data |
| `scripts/run_longshort_sweep.py` | Phase 4 long+short sweep engine (23 configs) |
| `results/longshort_sweep.csv` | Phase 4 full results — shorts vs long-only comparison |

## Next Steps / Recommendations

1. **Deploy with futures lots, LONG-ONLY** — shorts tested and confirmed destructive (PF 0.70-0.80). The long-only edge (PF 1.48-1.85, Sharpe 1.3-1.5) is real and consistent
2. **Best config: E25_60_ADX20_R14_30_P20_10** — 9.18% cash / 27.5% at 3x, Calmar 1.24, best across all sweeps
3. **Conservative alternative: E20_50_ADX25_R14_30_P20_10** — 6.65% cash / 19.9% at 3x, but MaxDD only 16.8% at 3x (vs 22.2%)
4. **Drop NR7 breakout entirely** — confirmed drag on risk-adjusted returns
5. **Drop all short signals** — tested EMA trend shorts (PF 0.70-0.80) and RSI contradiction shorts (PF 0.69-0.80). Indian markets have structural long bias
6. **Expand 5-min data** to test ORB (Opening Range Breakout) as a third component — currently only 10 stocks have 5-min data, PF 1.29 in preliminary tests
7. **Build live trading engine** similar to KC6 system — scheduled scans on 60-min candles, auto-execute via Kite API with F&O lots
8. **Position sizing for futures** — need to map each stock's lot size and margin to calculate actual number of lots per signal
