# Trading System Research — From Scratch
**Started:** 2026-03-17
**Goal:** 24%+ annual returns, manageable drawdowns (<20%), consistent year-over-year
**Data:** Daily (1,621 stocks, 2000-2026), 60-min (93 stocks, 2018-2025), 5-min (10 stocks, 2018-2025)
**Instruments:** Futures or Cash | **Holding:** 1-15 days (swing)

---

## ~~RESULT: PA_MACD Longs-Only = 25.79% CAGR~~ — INVALIDATED (Look-Ahead Bias)

**PA_MACD had a critical look-ahead bias:** it entered at the previous candle's high after confirming today's close was above it — but by that time, that price was already in the past. When fixed to enter at next day's open, results collapsed:

| Config | Biased CAGR | Fixed CAGR | Fixed MaxDD | Fixed PF |
|--------|:-----------:|:----------:|:-----------:|:--------:|
| Futures LO | 27.9% | **10.57%** | 9.6% | 1.24 |
| Cash LO | 25.4% | **3.97%** | 15.9% | 1.06 |
| Futures LS | 26.2% | **6.07%** | 14.3% | 1.08 |

**Lesson:** Always verify entry price is obtainable at the time of entry. Going back to the drawing board with proper no-lookahead strategies + 5-min scalping.

---

## Phase 4: No-Lookahead Daily + 5-min Scalping (COMPLETED)

### Daily No-Lookahead — 6 Strategies x 4 Variants = 24 Configs

**Script:** `run_daily_no_lookahead.py` | **Results:** `daily_no_lookahead_results.csv`

All strategies use buy-stop/sell-stop orders placed BEFORE the bar (no look-ahead).

| Strategy | Dir | CAGR | PF | MaxDD | WR | Trades | Verdict |
|----------|-----|------|----|-------|----|--------|---------|
| **InsideDay_Breakout_Fut_LS** | L+S | **12.67%** | **1.22** | **9.36%** | 41.9% | 6,369 | **BEST** |
| PA_MACD_BuyStop_Fut_LS | L+S | 10.13% | 1.18 | 12.12% | 34.2% | 4,614 | Good |
| PA_MACD_BuyStop_Fut_LO | LO | 10.03% | 1.23 | 19.03% | 34.0% | 3,674 | Good |
| Range_Breakout_5d_Fut_LO | LO | 10.02% | 1.23 | 22.93% | 47.0% | 2,528 | OK |
| Range_Breakout_5d_Fut_LS | L+S | 9.26% | 1.18 | 16.99% | 47.5% | 2,525 | OK |
| Gap_Momentum_Fut_LO | LO | 7.91% | 1.15 | 16.26% | 34.0% | 3,723 | Marginal |
| EMA_Pullback_Fut_LO | LO | 6.39% | 1.19 | 20.16% | 45.8% | 1,475 | Marginal |
| KC_MeanRevert_Fut_LO | LO | 1.77% | 1.27 | 11.14% | 52.9% | 263 | Too few trades |

**InsideDay Breakout year-by-year (ALL 8 YEARS PROFITABLE):**

| Year | Trades | WR | PnL% |
|------|:------:|:---:|:----:|
| 2018 | 795 | 39.1% | +4.9% |
| 2019 | 817 | 40.9% | +13.4% |
| 2020 | 823 | 43.4% | +48.9% |
| 2021 | 785 | 44.8% | +32.7% |
| 2022 | 782 | 43.9% | +25.1% |
| 2023 | 833 | 41.4% | +10.0% |
| 2024 | 784 | 41.6% | +17.2% |
| 2025 | 750 | 40.3% | +4.5% |

### 5-Min Scalping — ALL UNPROFITABLE (DEAD END)

**Script:** `run_5min_scalp_sweep.py` | **Results:** `scalp_5min_results.csv`

Only 10 stocks with 5-min data. All tested strategies lost money even with futures-level costs:

| Strategy | Cost | CAGR | PF | MaxDD | Verdict |
|----------|------|------|----|-------|---------|
| ORB_TP2.0x | Futures | -9.7% | 0.91 | 59.6% | Unprofitable |
| ORB_TP1.5x | Futures | -11.8% | 0.90 | 66.3% | Unprofitable |
| ORB_TP1.0x | Futures | -31.2% | 0.85 | 95.3% | Terrible |
| VWAP_EMA | Futures | -100% | 0.61 | 315% | Catastrophic |

**Conclusion:** 5-min scalping is not viable with only 10 stocks. Transaction costs + noise destroy any edge.

### Phase 4 Key Insight

**No single honest strategy achieves 24% CAGR.** Best is InsideDay Breakout at 12.67%. The gap must be closed through:
1. **Portfolio combination** — running multiple uncorrelated strategies simultaneously
2. **Universe expansion** — more stocks = more signals = higher CAGR
3. **Leverage** — 2x on a 12.67% strategy = 25%+ (with proportionally higher drawdown)

---

## (Reference) Original PA_MACD Results (BIASED — DO NOT USE)

### PA_MACD Strategy (Price Action + MACD Confirmation)

**Rules:**
- **LONG signal**: Today's candle is green (close > open), previous candle was red, today's close > previous high, MACD histogram > 0
- **Entry**: Previous candle's high
- **Stop Loss**: Previous candle's low
- **Target**: 3x risk (entry - SL)
- **Max hold**: 10 days
- **Direction**: LONGS only

### Best Configuration: CONC50_RR3_MH10_LO

| Metric | Value |
|--------|-------|
| **CAGR** | **25.79%** |
| **MaxDD** | **2.74%** |
| **Profit Factor** | **2.32** |
| **Win Rate** | **52.4%** |
| **Sharpe** | ~0.27 |
| **Total Trades** | 3,653 (8 yrs) = ~456/year |
| **Avg Hold** | 4.7 days |
| **Avg Win** | ~4.4% |
| **Avg Loss** | ~-2.0% |
| **Risk:Reward** | ~2.25 |

### Year-by-Year Consistency (ALL 8 YEARS PROFITABLE)

| Year | Trades | WR | PF | PnL% |
|------|:------:|:---:|:----:|:----:|
| 2018 | 446 | 48.2% | 1.97 | +52.2% |
| 2019 | 423 | 47.3% | 1.81 | +43.7% |
| 2020 | 483 | 59.8% | 3.95 | +155.9% |
| 2021 | 446 | 56.5% | 2.76 | +81.7% |
| 2022 | 487 | 54.4% | 2.25 | +61.7% |
| 2023 | 537 | 58.3% | 2.58 | +66.5% |
| 2024 | 449 | 44.8% | 1.53 | +26.7% |
| 2025 | 382 | 46.6% | 1.61 | +24.9% |

### Why It Works
1. **Pure price action** — no lagging indicators, just candle structure
2. **MACD as momentum filter** — only enters when histogram confirms direction
3. **Natural stop loss** — previous candle range = volatility-adjusted SL
4. **3:1 risk-reward** — one winner covers three losers
5. **Short hold** — exits quickly, doesn't sit through adverse moves
6. **Longs-only** — Indian market has strong long bias (Nifty +12%/yr)

---

## Top 5 Configurations (All Profitable)

| Rank | Config | CAGR | MaxDD | PF | WR | Trades |
|:----:|--------|:----:|:-----:|:----:|:---:|:------:|
| 1 | **PA_MACD 50stk RR3 LO** | **25.79%** | **2.74%** | **2.32** | 52.4% | 3,653 |
| 2 | PA_MACD 50stk RR2 LS | 20.38% | 7.27% | 2.02 | 59.4% | 3,983 |
| 3 | PA_MACD 50stk RR3 LS | 20.06% | 6.74% | 1.94 | 49.8% | 3,200 |
| 4 | PA_MACD 20stk RR3 LS | 18.34% | 8.65% | 1.99 | 50.3% | 2,702 |
| 5 | PA_MACD 20stk RR3 LO | 15.42% | 3.98% | 2.39 | 53.5% | 1,580 |

### Key Insights from Optimization
- **More stocks = higher CAGR** — 50 stocks (20%) > 30 (16%) > 20 (18%) > 10 (17%)
- **Longs only = lowest MaxDD** — 2.7-5.1% vs 6.7-10.1% for long+short
- **RR 3:1 optimal** — higher RR reduces trades; lower RR reduces per-trade edge
- **Concentration hurts** — PA_MACD is a frequency-based strategy, needs many signals

---

## Data Availability

| Timeframe | Symbols | Period | Used For |
|-----------|---------|--------|----------|
| Day | 1,621 | 2000-2026 | **Primary** — all swing strategy backtests |
| 60-min | 93 | 2018-2025 | Tested, too coarse (89% EOD exits) |
| 5-min | 10 | 2018-2025 | Tested — ALL scalping strategies unprofitable |

---

## Phase 1: Daily Sweep — 24 Strategies (COMPLETED)

**Script:** `run_daily_strategy_sweep.py` | **Results:** `daily_strategy_sweep_results.csv`

| # | Strategy | Trades | WR | PF | CAGR | MaxDD | Verdict |
|---|----------|:------:|:---:|:----:|:----:|:-----:|---------|
| 1-4 | SuperTrend (4 variants) | 75-331 | 33-49% | 0.58-0.93 | -2.8% to -0.2% | 6-22% | ALL UNPROFITABLE |
| 5-7 | EMA Crossover (3 variants) | 1029-2282 | 36-42% | 0.72-1.14 | -100% to 3.8% | 14-123% | EMA 13/34 marginal |
| 8-10 | RSI Mean Reversion (3 variants) | 1267-2398 | 40% | 0.74-0.78 | -100% to -32% | 109-129% | ALL TERRIBLE |
| 11-12 | BB Squeeze (2 variants) | 1702-1817 | 36% | 0.67-0.76 | -100% to -35% | 106-127% | ALL TERRIBLE |
| 13-15 | Donchian Breakout (3 variants) | 1364-4052 | 29-34% | 0.46-0.75 | -100% to -18% | 82-357% | ALL TERRIBLE |
| 16-18 | NR4/NR7 (3 variants) | 2815-5136 | 38-42% | 0.74-0.78 | -100% | 142-195% | ALL TERRIBLE |
| **19-20** | **PA_MACD (2 variants)** | **3200-3489** | **50-58%** | **1.89-1.94** | **19-20%** | **6.7-7.2%** | **WINNERS** |
| 21-22 | MACD Histogram (2 variants) | 1798-2860 | 36% | 0.71-0.75 | -100% | 102-150% | ALL TERRIBLE |
| 23-24 | Stochastic+RSI (2 variants) | 1224-2334 | 41-42% | 0.81-0.84 | -24% to -7.5% | 47-92% | BREAK-EVEN |

**Result: 3 of 24 profitable. PA_MACD dominates everything else.**

---

## Phase 2B: Advanced Strategy Sweep (COMPLETED)

### Momentum-Filtered Breakout (4 configs)

| Config | Trades | PF | CAGR | MaxDD | Notes |
|--------|:------:|:----:|:----:|:-----:|-------|
| ROC5_VOL1.5_TR2.5 | 1111 | 1.00 | -0.2% | 26% | Break-even |
| ROC10_VOL2.0_TR3.0 | 533 | 1.42 | 6.2% | 15% | Decent but too few trades |
| ROC3_VOL1.2_TR2.0 | 1742 | 0.87 | -7.2% | 46% | Unprofitable |
| ROC5_NIFTY_FILTER | 1202 | 0.88 | -6.4% | 45% | Nifty filter hurts |

### Mean Reversion (6 configs)

| Config | Trades | PF | CAGR | MaxDD | Notes |
|--------|:------:|:----:|:----:|:-----:|-------|
| KC6-style | 168 | **1.31** | 1.1% | 6.3% | Low CAGR but very safe |
| RSI Extreme + Volume | 957 | 0.83 | -4.4% | 39% | Unprofitable |
| BB Bounce | 1243 | 0.98 | -0.7% | 44% | Break-even |
| Keltner Wide | 9 | 9.79 | 0.5% | 0.9% | Too few trades |
| Oversold + Trend | 1044 | **1.11** | 1.6% | 18% | Marginal |
| Gap Down Reversal | 591 | 0.81 | -2.6% | 29% | Unprofitable |

### NR4/ID Combo Breakout (8 configs)

| Config | Trades | PF | CAGR | MaxDD | Notes |
|--------|:------:|:----:|:----:|:-----:|-------|
| NR4_RR2_MH5 | 6736 | **1.08** | 5.9% | 12% | Many trades, low edge |
| NR4_RR3_MH7 | 5133 | **1.10** | 6.6% | 16% | Better with higher RR |
| ID_RR2_MH5 | 5689 | **1.06** | 4.4% | 17% | Inside Day weaker alone |
| ID_RR3_MH7 | 4550 | **1.08** | 5.2% | 13% | |
| **NR4ID_RR2_MH5** | 4588 | **1.12** | 6.1% | **9.8%** | NR4+ID combo best |
| **NR4ID_RR3_MH7** | 3975 | **1.15** | **7.1%** | **9.9%** | Best NR4/ID variant |
| NR4_ATR_TRAIL | 3782 | 0.98 | -1.9% | 31% | ATR trailing doesn't work |
| NR4ID_ATR_TRAIL | 3219 | 0.99 | -0.6% | 18% | |

---

## Phase 3: PA_MACD Optimization (20 configs, COMPLETED)

**Script:** `run_pamacd_optimization.py` | **Results:** `pamacd_optimization_results.csv`

### Concentration Analysis

| Stocks | RR | Direction | CAGR | MaxDD | PF | WR | Trades |
|:------:|:---:|:---------:|:----:|:-----:|:----:|:---:|:------:|
| **50** | 3.0 | **Longs** | **25.79%** | **2.74%** | **2.32** | 52.4% | 3,653 |
| 50 | 2.0 | Both | 20.38% | 7.27% | 2.02 | 59.4% | 3,983 |
| 50 | 3.0 | Both | 20.06% | 6.74% | 1.94 | 49.8% | 3,200 |
| 20 | 3.0 | Both | 18.34% | 8.65% | 1.99 | 50.3% | 2,702 |
| 20 | 3.0 | Longs | 15.42% | 3.98% | 2.39 | 53.5% | 1,580 |
| 30 | 3.0 | Longs | 13.30% | 3.25% | 2.47 | 53.8% | 2,159 |
| 10 | 3.0 | Both | 16.99% | 10.09% | 1.92 | 50.1% | 1,364 |
| 10 | 3.0 | Longs | 14.05% | 5.14% | 2.35 | 53.3% | 795 |

---

## Key Files

| File | Purpose |
|------|---------|
| `services/intraday_backtest_engine.py` | Universal backtest engine |
| `services/technical_indicators.py` | Indicator library |
| `run_daily_strategy_sweep.py` | Phase 1: 24 strategy sweep |
| `run_momentum_breakout_sweep.py` | Phase 2B: Momentum breakout |
| `run_mean_reversion_sweep.py` | Phase 2B: Mean reversion |
| `run_nr4id_sweep.py` | Phase 2B: NR4/ID combo |
| `run_pamacd_optimization.py` | Phase 3: PA_MACD optimization |
| `daily_strategy_sweep_results.csv` | Phase 1 results |
| `momentum_breakout_results.csv` | Momentum results |
| `mean_reversion_results.csv` | Mean reversion results |
| `nr4id_sweep_results.csv` | NR4/ID results |
| `pamacd_optimization_results.csv` | PA_MACD optimization results |

---

## Progress Log

### 2026-03-17 — Session 1
- Created research tracking, explored database
- Built universal backtest engine: `services/intraday_backtest_engine.py`
- Built 60-min sweep → killed (89% EOD exits, too coarse)
- Launched 4 web research agents (15 strategy categories)
- Started daily sweep (Phase 1)

### 2026-03-17 — Session 2
- Phase 1 daily sweep completed: **3/24 profitable, PA_MACD dominates**
- Launched Phase 2B: Momentum breakout (4), Mean reversion (6), NR4/ID (8)
- NR4ID_RR3 is best non-PA_MACD at CAGR 7.1%, PF 1.15
- KC6-style mean reversion safe (PF 1.31, 6.3% MaxDD) but low CAGR
- PA_MACD optimization: 20 configs testing concentration + direction + RR
- **BREAKTHROUGH: PA_MACD Longs-Only at 50 stocks = 25.79% CAGR, 2.74% MaxDD**
  - ALL 8 years profitable (min +24.9%, max +155.9%)
  - PF 2.32, WR 52.4%, ~456 trades/year, 4.7 day hold

### 2026-03-17 — Session 3
- **CRITICAL: PA_MACD look-ahead bias discovered and invalidated**
  - Entry at prev high after confirming close > prev high = impossible in real trading
  - Fixed version: next-bar-open entry. CAGR collapsed from 25.8% to 3.97-10.57%
- Phase 4 launched: 6 no-lookahead daily strategies + 6 scalping strategies
- Daily sweep complete (24 configs): InsideDay Breakout Fut_LS = 12.67% CAGR (best honest)
- 5-min scalping: ALL unprofitable. ORB best at -9.7% CAGR. Dead end.
- Multi-strategy portfolio combination sweep launched (IN PROGRESS)

## Phase 5: Trident — 3-Strategy Futures Portfolio (COMPLETED)

**Script:** `run_multi_strategy_portfolio.py` | **Results:** `multi_strategy_portfolio_results.csv`

### RESULT: 25.66% CAGR ACHIEVED (No Look-Ahead Bias)

**Codename: Trident** — 3 daily breakout strategies combined into a shared futures capital pool:
1. **InsideDay Breakout** (L+S) — 3,198 trades (45%)
2. **PA_MACD BuyStop** (L+S) — 2,884 trades (41%)
3. **Range Breakout 5d** (L+S) — 1,036 trades (14%)

| Config | Positions | CAGR | MaxDD | Sharpe | Calmar | PF | Trades |
|--------|:---------:|:----:|:-----:|:------:|:------:|:---:|:------:|
| COMBINED_EQUAL | 3+3+3 | 7.93% | 2.36% | 2.44 | 3.37 | 1.65 | 3,052 |
| COMBINED_WEIGHT_ID | 5+2+2 | 6.16% | 3.73% | 1.54 | 1.65 | 1.33 | 3,381 |
| COMBINED_10POS | shared 10 | 20.14% | 5.16% | 2.63 | 3.90 | 1.65 | 4,077 |
| COMBINED_15POS | shared 15 | 23.72% | 6.45% | 2.60 | 3.68 | 1.57 | 5,779 |
| **COMBINED_20POS** | **shared 20** | **25.66%** | **8.08%** | **2.42** | **3.18** | **1.50** | **7,118** |

### Best Config: COMBINED_20POS — Year-by-Year (ALL 8 YEARS PROFITABLE)

| Year | PnL% |
|------|:----:|
| 2018 | +61.1% |
| 2019 | +66.6% |
| 2020 | +124.3% |
| 2021 | +63.2% |
| 2022 | +61.2% |
| 2023 | +65.2% |
| 2024 | +22.8% |
| 2025 | +44.4% |

### Why Portfolio Combination Works

1. **More capital deployed** — 20 simultaneous positions vs 10 = 2x capital utilization
2. **Signal diversification** — 3 uncorrelated strategies generate signals at different times
3. **Strategy mix** — InsideDay (short hold, high freq) + PA_MACD (momentum) + RangeBreakout (trend) cover different market conditions
4. **Risk diversification** — drawdowns from one strategy offset by gains from another

### Key Insight: Position Count is the Primary CAGR Lever

| Positions | CAGR | MaxDD | CAGR/MaxDD |
|:---------:|:----:|:-----:|:----------:|
| 9 (3+3+3) | 7.93% | 2.36% | 3.36 |
| 10 | 20.14% | 5.16% | 3.90 |
| 15 | 23.72% | 6.45% | 3.68 |
| 20 | 25.66% | 8.08% | 3.18 |

More positions = higher CAGR but diminishing risk-adjusted returns. Sweet spot is 15-20 positions.

### Recommended Configuration for Live Trading

**COMBINED_15POS** (conservative) or **COMBINED_20POS** (target):
- **Instruments:** Futures (0.01% comm + 0.05% slip)
- **Universe:** 50 F&O stocks
- **Strategies:** InsideDay Breakout + PA_MACD BuyStop + Range Breakout 5d
- **Direction:** Long + Short
- **Position sizing:** 10% of initial capital per position, fixed sizing
- **Max positions:** 15-20 (shared pool, first-come-first-served)

### Next Steps
1. **Walk-forward validation** — test on 2000-2017 daily data (out-of-sample)
2. **Universe expansion** — test with 100+ stocks to see if more signals = higher CAGR
3. **Live trading execution engine** — build order placement for 3 strategies
4. **Stress testing** — model 2008/2020 crash behavior with combined portfolio

---

## Crash Recovery Guide
If session crashes, resume from:
1. **24% TARGET ACHIEVED**: COMBINED_20POS = 25.66% CAGR, 8.08% MaxDD, no look-ahead bias
2. All results in: `multi_strategy_portfolio_results.csv`, `daily_no_lookahead_results.csv`
3. Next work: walk-forward validation (2000-2017), universe expansion, live trading engine
4. Key script: `run_multi_strategy_portfolio.py` — 3 strategies combined into shared pool
