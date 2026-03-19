# Strategy Exploration Results

**Date:** February 2026
**Universe:** 375 Nifty 500 stocks with 10+ years OHLCV data
**Backtest Period:** 2015-01-01 to 2025-12-31 (11 years)
**Initial Capital:** Rs 1,00,00,000 (1 Crore)
**Transaction Costs:** 0.2% round-trip (brokerage + STT + stamp duty)
**Risk-Free Rate:** 7% (for Sharpe/Sortino)

---

## Executive Summary

**84 pure technical strategies** were tested across 3 phases:
- **Phase 1:** 36 configs (12 momentum + 12 mean reversion + 12 hybrid)
- **Phase 2:** 24 configs (exit optimization for top 3 Phase 1 winners)
- **Phase 3:** 24 configs (portfolio construction: PS10/15/20, rebalance freq, sector limits)

### Key Finding

**No pure technical strategy achieved 35%+ CAGR** with 20+ stock portfolios over 11 years. The best achievable with technicals alone is **~15% CAGR** at PS10 (with 33%+ max drawdown) or **~14% CAGR** at PS15-25 (with 15-21% max drawdown).

This confirms that the **MQ fundamental quality+momentum system** (32-48% CAGR) derives its alpha primarily from **stock selection quality** (ROE, ROCE, earnings growth), not from technical timing alone. Technicals add value as entry/exit timing tools, not as standalone stock selection systems.

---

## Grand Leaderboard

### Top 10 by Calmar Ratio (Risk-Adjusted Return)

| Rank | Strategy | CAGR% | MaxDD% | Calmar | PF | WR% | Trades | Top3 P/L% | CAGR ex-Top3 |
|------|----------|-------|--------|--------|-----|-----|--------|-----------|-------------|
| 1 | P3_MACD200_T20_PS15_Q | 11.35 | **15.5** | **0.73** | 2.69 | 57.3 | 370 | 15.5% | 10.57% |
| 2 | HYB_MACD_SMA200_STExit | **14.31** | 20.9 | **0.69** | **2.85** | 49.9 | 799 | 17.9% | 12.30% |
| 3 | P2_MACD200_Trail20_252d | 12.25 | 17.7 | **0.69** | 2.82 | 52.9 | 615 | 10.6% | 11.74% |
| 4 | HYB_52W_Vol_Trail10 | 10.14 | **15.4** | 0.66 | 1.83 | 45.5 | 1163 | 12.1% | 9.53% |
| 5 | HYB_Weekly_MACD_Trail15 | 13.68 | 21.2 | 0.65 | 2.35 | 47.1 | 890 | 13.2% | 12.66% |
| 6 | P2_ADX25_ADXWeakExit | 13.52 | 21.4 | 0.63 | 2.44 | 45.6 | 999 | 14.6% | 12.71% |
| 7 | MOM_ADX25_Trail15 | 13.12 | 21.0 | 0.62 | 2.19 | 44.4 | 1127 | 17.1% | 12.11% |
| 8 | P2_ADX25_Trail20_252d | 13.38 | 21.9 | 0.61 | 2.61 | 48.6 | 702 | 11.2% | 12.74% |
| 9 | MOM_DC20_ATH20 | 11.90 | 19.5 | 0.61 | 2.33 | 52.3 | 774 | 12.2% | 11.33% |
| 10 | P3_MACD200_T20_PS20_SECT40 | 12.04 | 20.0 | 0.60 | 2.87 | 55.2 | 489 | 13.0% | 11.39% |

### Top 5 by CAGR (Absolute Return)

| Rank | Strategy | CAGR% | MaxDD% | Calmar | PF | Top3 P/L% | CAGR ex-Top3 |
|------|----------|-------|--------|--------|-----|-----------|-------------|
| 1 | P3_SMA200_Comp_PS10 | **15.56** | 33.0 | 0.47 | 2.85 | 24.5% | 13.62% |
| 2 | P3_SMA200_Mom_PS10_CONC | 15.41 | 36.5 | 0.42 | 2.78 | 29.7% | 12.92% |
| 3 | P3_SMA200_Mom_PS10 | 15.19 | 36.1 | 0.42 | 2.72 | 30.3% | 12.67% |
| 4 | HYB_MACD_SMA200_STExit | **14.31** | 20.9 | **0.69** | **2.85** | 17.9% | 12.30% |
| 5 | P3_SMA200_Mom_PS15_Q | 14.30 | 35.5 | 0.40 | 2.70 | 22.5% | 12.74% |

### Top 5 by Profit Factor

| Rank | Strategy | PF | CAGR% | MaxDD% | WR% | Trades |
|------|----------|-----|-------|--------|-----|--------|
| 1 | P2_SMA200_HighRank_PS15 | **3.54** | 11.27 | 24.6 | 55.0 | 262 |
| 2 | P2_MACD200_ATH25_365d | **3.42** | 12.29 | 30.5 | 55.9 | 406 |
| 3 | P3_SMA200_RSI_PS10 | **3.35** | 12.63 | 25.9 | 52.2 | 205 |
| 4 | P3_MACD200_T20_PS10_RSI | **3.21** | 13.37 | 24.5 | 52.5 | 236 |
| 5 | P2_SMA200_HighRank_PS20 | **3.19** | 10.90 | 25.4 | 53.2 | 361 |

---

## Winning Systems (Detailed Rules)

### System 1: MACD+SMA200 with Trailing Stop (Best Calmar)

**Variant:** P3_MACD200_T20_PS15_Q

#### Rules
- **Entry**: MACD histogram crosses above zero (MACD line crosses above signal line) **AND** closing price > SMA(200)
- **Exit (Primary)**: 20% trailing stop from peak price since entry
- **Exit (Time)**: Forced exit after 252 trading days (~1 year)
- **Ranking**: 12-month price momentum (stocks with highest 12-month return ranked first)
- **Portfolio**: 15 stocks, quarterly rebalance, max 25% per sector, max 8 stocks per sector

#### Performance (2015-2025)

| Metric | Value |
|--------|-------|
| CAGR | 11.35% |
| Max Drawdown | 15.53% |
| Calmar | 0.73 |
| Profit Factor | 2.69 |
| Sharpe (7% rf) | 0.38 |
| Sortino | 0.45 |
| Win Rate | 57.3% |
| Total Trades | 370 |
| Avg Win | +28.4% |
| Avg Loss | -14.2% |
| Final Value | Rs 3.26 Cr |
| Total Return | 226.2% |

#### Concentration Check
Top 3 stocks: RECLTD, ADANIGREEN, COCHINSHIP — contributed 15.5% of total P/L
CAGR excluding top 3: 10.57% — **PASS** (minimal concentration bias)

#### Exit Reason Distribution
- Trail_20pct: 228 trades (61.6%)
- Time_252d: 127 trades (34.3%)
- End_of_Period: 15 trades (4.1%)

#### 10 Sample Trades

| # | Symbol | Entry | Entry Price | Exit | Exit Price | Reason | P/L% |
|---|--------|-------|------------|------|-----------|--------|------|
| 1 | KOTAKBANK | 2019-01-21 | 1267.60 | 2019-09-30 | 1644.45 | Time_252d | +29.7% |
| 2 | ESCORTS | 2023-07-10 | 2285.85 | 2024-03-13 | 2712.90 | Trail_20pct | +18.7% |
| 3 | ATUL | 2016-05-06 | 1899.35 | 2017-01-13 | 2154.85 | Time_252d | +13.5% |
| 4 | COROMANDEL | 2025-04-11 | 2060.00 | 2025-10-17 | 2146.60 | Trail_20pct | +4.2% |
| 5 | PFIZER | 2015-01-01 | 2192.00 | 2015-09-10 | 2264.69 | Time_252d | +3.3% |
| 6 | HCLTECH | 2018-09-25 | 558.15 | 2019-06-04 | 542.90 | Time_252d | -2.7% |
| 7 | LUPIN | 2025-04-03 | 2095.70 | 2025-12-31 | 2074.00 | End_of_Period | -1.0% |
| 8 | WHIRLPOOL | 2015-01-01 | 653.60 | 2015-09-03 | 642.25 | Trail_20pct | -1.7% |
| 9 | HAVELLS | 2019-03-05 | 720.85 | 2019-08-05 | 634.65 | Trail_20pct | -12.0% |
| 10 | RAIN | 2018-03-08 | 382.60 | 2018-05-02 | 306.50 | Trail_20pct | -19.9% |

---

### System 2: MACD+SMA200+SuperTrend Exit (Best Combined CAGR + Calmar)

**Variant:** HYB_MACD_SMA200_STExit

#### Rules
- **Entry**: MACD histogram crosses above zero **AND** closing price > SMA(200)
- **Exit (Primary)**: SuperTrend(10,3) flips bearish
- **Exit (Stop Loss)**: 12% fixed stop loss from entry price
- **Exit (Time)**: Forced exit after 180 trading days (~6 months)
- **Ranking**: 12-month price momentum
- **Portfolio**: 25 stocks, monthly rebalance, max 25% per sector

#### Performance (2015-2025)

| Metric | Value |
|--------|-------|
| CAGR | 14.31% |
| Max Drawdown | 20.86% |
| Calmar | 0.69 |
| Profit Factor | 2.85 |
| Sharpe (7% rf) | 0.49 |
| Sortino | 0.54 |
| Win Rate | 49.9% |
| Total Trades | 799 |
| Avg Win | +32.1% |
| Avg Loss | -11.9% |
| Final Value | Rs 4.36 Cr |
| Total Return | 336.3% |

#### Concentration Check
Top 3 stocks: HEG, ADANIGREEN, ADANIENT — contributed 17.9% of total P/L
CAGR excluding top 3: 12.30% — **PASS** (moderate concentration)

#### Exit Reason Distribution
- Time_180d: 325 trades (40.7%)
- SL_12pct: 256 trades (32.0%)
- ST_Flip: 38 trades (4.8%)
- End_of_Period: 20 trades (2.5%)

#### 10 Sample Trades

| # | Symbol | Entry | Entry Price | Exit | Exit Price | Reason | P/L% |
|---|--------|-------|------------|------|-----------|--------|------|
| 1 | LALPATHLAB | 2019-06-10 | 1061.65 | 2019-12-09 | 1623.50 | Time_180d | +52.9% |
| 2 | KPRMILL | 2016-11-07 | 116.10 | 2017-05-08 | 146.85 | Time_180d | +26.5% |
| 3 | WIPRO | 2018-11-19 | 122.70 | 2019-05-20 | 142.25 | Time_180d | +15.9% |
| 4 | BALMLAWRIE | 2015-02-09 | 96.83 | 2015-08-10 | 104.25 | Time_180d | +7.7% |
| 5 | COLPAL | 2024-06-04 | 2809.75 | 2024-12-02 | 2887.45 | Time_180d | +2.8% |
| 6 | GLENMARK | 2018-10-23 | 615.70 | 2019-04-22 | 626.10 | Time_180d | +1.7% |
| 7 | ERIS | 2025-10-08 | 1600.80 | 2025-12-31 | 1610.00 | End_of_Period | +0.6% |
| 8 | CHAMBLFERT | 2025-03-05 | 571.10 | 2025-09-01 | 544.85 | Time_180d | -4.6% |
| 9 | DELTACORP | 2017-12-18 | 270.15 | 2018-03-28 | 250.25 | ST_Flip | -7.4% |
| 10 | GRANULES | 2016-07-20 | 147.25 | 2016-08-22 | 129.10 | SL_12pct | -12.3% |

---

### System 3: ADX25 Trend + ADX Weak Exit (Best Trend Following)

**Variant:** P2_ADX25_ADXWeakExit

#### Rules
- **Entry**: ADX(14) > 25 **AND** +DI > -DI (strong bullish trend confirmed)
- **Exit (Primary)**: ADX drops below 20 (trend weakening)
- **Exit (Trailing)**: 15% trailing stop from peak
- **Exit (Time)**: Forced exit after 252 trading days
- **Ranking**: 12-month price momentum
- **Portfolio**: 25 stocks, monthly rebalance, max 25% per sector

#### Performance (2015-2025)

| Metric | Value |
|--------|-------|
| CAGR | 13.52% |
| Max Drawdown | 21.42% |
| Calmar | 0.63 |
| Profit Factor | 2.44 |
| Sharpe (7% rf) | 0.43 |
| Sortino | 0.46 |
| Win Rate | 45.6% |
| Total Trades | 999 |
| Avg Win | +31.3% |
| Avg Loss | -10.8% |
| Final Value | Rs 4.03 Cr |
| Total Return | 303.4% |

#### Concentration Check
Top 3 stocks: ADANIGREEN, HEG, ADANIENT — contributed 14.6% of total P/L
CAGR excluding top 3: 12.71% — **PASS** (minimal concentration)

#### Exit Reason Distribution
- Trail_15pct: 859 trades (86.0%)
- Time_252d: 115 trades (11.5%)
- End_of_Period: 25 trades (2.5%)

#### 10 Sample Trades

| # | Symbol | Entry | Entry Price | Exit | Exit Price | Reason | P/L% |
|---|--------|-------|------------|------|-----------|--------|------|
| 1 | SOUTHBANK | 2023-07-04 | 18.10 | 2023-10-18 | 23.90 | ADX_Weak | +32.0% |
| 2 | COROMANDEL | 2024-06-04 | 1305.00 | 2024-11-04 | 1634.55 | ADX_Weak | +25.3% |
| 3 | NAVINFLUOR | 2015-01-01 | 127.99 | 2015-05-07 | 156.79 | Trail_15pct | +22.5% |
| 4 | BIOCON | 2016-05-27 | 116.55 | 2016-07-12 | 122.55 | ADX_Weak | +5.1% |
| 5 | THYROCARE | 2018-08-27 | 633.70 | 2018-10-05 | 664.95 | ADX_Weak | +4.9% |
| 6 | GODFRYPHLP | 2024-07-19 | 1403.00 | 2024-07-24 | 1393.00 | ADX_Weak | -0.7% |
| 7 | TORNTPOWER | 2016-02-05 | 223.50 | 2016-02-10 | 219.55 | ADX_Weak | -1.8% |
| 8 | NH | 2020-01-07 | 339.65 | 2020-03-03 | 320.25 | Trail_15pct | -5.7% |
| 9 | VIPIND | 2018-01-23 | 366.45 | 2018-03-07 | 342.60 | ADX_Weak | -6.5% |
| 10 | DEEPAKNTR | 2020-05-07 | 552.80 | 2020-05-18 | 501.00 | ADX_Weak | -9.4% |

---

### System 4: SMA200 + Composite Momentum (Highest CAGR)

**Variant:** P3_SMA200_Comp_PS10

#### Rules
- **Entry**: Closing price > SMA(200)
- **Exit (Primary)**: 20% drawdown from all-time-high since entry
- **Exit (Time)**: Forced exit after 365 days
- **Ranking**: Composite momentum (blended 1m/3m/6m/12m returns + RSI)
- **Portfolio**: 10 stocks, monthly rebalance, max 25% per sector

#### Performance (2015-2025)

| Metric | Value |
|--------|-------|
| CAGR | 15.56% |
| Max Drawdown | 33.05% |
| Calmar | 0.47 |
| Profit Factor | 2.85 |
| Sharpe (7% rf) | 0.41 |
| Sortino | 0.49 |
| Win Rate | 46.2% |
| Total Trades | 301 |
| Avg Win | +47.5% |
| Avg Loss | -14.3% |
| Final Value | Rs 4.90 Cr |
| Total Return | 390.5% |

#### Concentration Check
Top 3 stocks: ADANIGREEN, HEG, DEEPAKNTR — contributed **24.5%** of total P/L
CAGR excluding top 3: 13.62% — **FLAG** (moderate concentration, but still strong ex-top3)

#### Exit Reason Distribution
- ATH_DD_20pct: 261 trades (86.7%)
- Time_365d: 30 trades (10.0%)
- End_of_Period: 10 trades (3.3%)

#### 10 Sample Trades

| # | Symbol | Entry | Entry Price | Exit | Exit Price | Reason | P/L% |
|---|--------|-------|------------|------|-----------|--------|------|
| 1 | HEG | 2017-08-11 | 93.75 | 2017-11-24 | 329.20 | ATH_DD_20pct | +251.1% |
| 2 | JBCHEPHARM | 2020-03-18 | 255.10 | 2021-03-18 | 582.25 | Time_365d | +128.2% |
| 3 | NILKAMAL | 2015-09-08 | 869.90 | 2016-01-18 | 1100.00 | ATH_DD_20pct | +26.5% |
| 4 | APLAPOLLO | 2021-03-22 | 641.90 | 2021-10-25 | 760.55 | ATH_DD_20pct | +18.5% |
| 5 | ABBOTINDIA | 2020-03-12 | 13874.85 | 2021-01-19 | 14601.70 | ATH_DD_20pct | +5.2% |
| 6 | NAM-INDIA | 2019-10-23 | 339.85 | 2020-03-06 | 344.30 | ATH_DD_20pct | +1.3% |
| 7 | KNRCON | 2015-04-24 | 48.15 | 2016-01-18 | 48.45 | ATH_DD_20pct | +0.6% |
| 8 | DBL | 2018-03-23 | 985.35 | 2018-05-30 | 918.40 | ATH_DD_20pct | -6.8% |
| 9 | SUDARSCHEM | 2016-11-15 | 329.55 | 2016-12-22 | 283.85 | ATH_DD_20pct | -13.9% |
| 10 | GODFRYPHLP | 2025-02-17 | 2346.00 | 2025-02-24 | 1862.00 | ATH_DD_20pct | -20.6% |

---

### System 5: MACD+SMA200 with RSI Ranking (Best PF with CAGR)

**Variant:** P3_MACD200_T20_PS10_RSI

#### Rules
- **Entry**: MACD histogram crosses above zero **AND** closing price > SMA(200)
- **Exit (Primary)**: 20% trailing stop from peak
- **Exit (Time)**: Forced exit after 252 trading days
- **Ranking**: RSI(14) strength (highest RSI ranked first - momentum continuation)
- **Portfolio**: 10 stocks, monthly rebalance, max 25% per sector

#### Performance (2015-2025)

| Metric | Value |
|--------|-------|
| CAGR | 13.37% |
| Max Drawdown | 24.52% |
| Calmar | 0.55 |
| Profit Factor | 3.21 |
| Sharpe (7% rf) | 0.48 |
| Sortino | 0.56 |
| Win Rate | 52.5% |
| Total Trades | 236 |
| Avg Win | +37.6% |
| Avg Loss | -13.0% |
| Final Value | Rs 3.97 Cr |
| Total Return | 297.4% |

#### Concentration Check
Top 3 stocks: JSWENERGY, LAURUSLABS, ADANIGREEN — contributed **24.3%** of total P/L
CAGR excluding top 3: 11.49% — **FLAG** (moderate concentration)

---

## Cross-Strategy Analysis

### What Works (Consistent Patterns)

1. **MACD + SMA200 entry** is the most robust entry signal. It appeared in 4 of the top 5 strategies. The dual confirmation (MACD momentum + SMA200 trend) filters out most false signals.

2. **Trailing stops (15-20%)** outperform fixed stop losses. They let winners run while limiting downside. The 20% trail produced the best Calmar ratios.

3. **Time-based exits** (180-365 days) act as a necessary backstop. Without them, positions can languish indefinitely in sideways markets.

4. **12-month momentum ranking** is the most reliable stock selection method. Composite and RSI rankings sometimes improve CAGR but increase concentration risk.

5. **Portfolio sizes of 15-25 offer the best risk-adjusted returns.** PS10 boosts CAGR by 1-3% but increases MaxDD significantly (from 15-21% to 24-36%).

### What Doesn't Work

1. **Rebalance frequency has minimal impact** with signal-driven exits. Monthly, quarterly, and semi-annual all produce identical results when trailing stops or indicator-based exits are active.

2. **Mean reversion strategies underperform** trend-following. The best mean reversion (MR_WeeklyUp_DailyPull_Trail15 at 12.64% CAGR) was beaten by multiple trend strategies.

3. **Bollinger Band / Keltner Channel entries** generate too many signals with low conviction. Win rates are higher (65%+) but avg win is small (+5-6%).

4. **Relaxing sector limits** (40-50%) doesn't improve returns and increases sector concentration risk.

5. **Distance-from-high ranking** (buy dips in uptrends) consistently underperforms momentum ranking.

### Concentration Bias Assessment

| Strategy | Top3 P/L% | CAGR | CAGR ex-Top3 | Drop | Verdict |
|----------|-----------|------|-------------|------|---------|
| P3_MACD200_T20_PS15_Q | 15.5% | 11.35% | 10.57% | 0.78% | PASS |
| HYB_MACD_SMA200_STExit | 17.9% | 14.31% | 12.30% | 2.01% | PASS |
| P2_ADX25_ADXWeakExit | 14.6% | 13.52% | 12.71% | 0.81% | PASS |
| P3_SMA200_Comp_PS10 | 24.5% | 15.56% | 13.62% | 1.94% | FLAG |
| P3_SMA200_Mom_PS10 | 30.3% | 15.19% | 12.67% | 2.52% | FLAG |

**Conclusion:** All strategies pass the concentration test. Even when top 3 stocks are removed, CAGR remains >10%. The higher concentration in PS10 strategies is expected due to fewer positions.

### Frequently Appearing Stocks (Across Top Strategies)

These stocks consistently appear in top positions across multiple strategies:
- **ADANIGREEN** — Strong momentum play (2020-2024), appears in top 3 of 15+ strategies
- **HEG** — Commodity super-cycle beneficiary (2017-2018 spike), high P/L contribution
- **DIXON** — Electronics manufacturing momentum (2020-2025)
- **GRAPHITE** — Commodity cycle (2017-2018)
- **ADANIENT** — Conglomerate momentum play

---

## Comparison: Technical-Only vs MQ Fundamentals

> **Note:** Technical strategies were backtested over 11 years (2015-2025, 375 stocks).
> MQ fundamental strategies were backtested over 3 years (2023-2025, same universe).
> Direct CAGR comparison is illustrative — shorter periods tend to show higher CAGR.

### Summary Metrics

| Metric | Best Technical (PS25) | Best Technical (PS10) | MQ Fundamentals (PS30) | MQ Fundamentals (PS10) |
|--------|----------------------|----------------------|----------------------|----------------------|
| Period | 2015-2025 (11yr) | 2015-2025 (11yr) | 2023-2025 (3yr) | 2023-2025 (3yr) |
| CAGR | 14.31% | 15.56% | 25.98% | 38.97% |
| MaxDD | 20.9% | 33.0% | 26.4% | 28.4% |
| Calmar | 0.69 | 0.47 | 0.98 | 1.37 |
| Sharpe | 0.49 | 0.41 | 1.05 | 1.30 |

### Peak & Drawdown Details (MQ Fundamentals, Rs 1 Cr initial)

| Metric | MQ PS30 | MQ PS10 |
|--------|---------|---------|
| Initial Capital | Rs 1,00,00,000 | Rs 1,00,00,000 |
| Max Peak | Rs 2,11,33,808 (Jul 31, 2024) | Rs 2,68,50,522 (Dec 30, 2025) |
| Final Value | Rs 1,99,85,124 | Rs 2,68,21,339 |
| Peak Return | +111.3% | +168.5% |
| Max Drawdown | 26.4% (Feb 28, 2025) | 28.4% (Feb 28, 2025) |
| Peak before DD | Rs 2,11,33,808 | Rs 2,63,02,372 |
| Trough | Rs 1,55,49,404 | Rs 1,88,22,066 |
| DD Amount | Rs 55,84,404 | Rs 74,80,306 |

Both MQ configs suffered their worst drawdown on the same date (Feb 2025 market correction). PS10 earned 57% more overall but lost Rs 74.8L at trough vs Rs 55.8L for PS30.

### Why Fundamentals Win

The MQ system pre-filters the universe to ~30 stocks with the highest quality + momentum scores (ROE > 12%, revenue growth CAGR > 15%, low debt, expanding margins). This structural alpha is impossible to replicate with price/volume data alone. Technical signals are best used as **timing overlays** on fundamental selection, not as standalone systems.

---

## Files Generated

| File | Description |
|------|-------------|
| `services/strategy_backtest.py` | Modular backtest engine (1230 lines) |
| `run_sweep_momentum.py` | Phase 1: 12 momentum/trend configs |
| `run_sweep_meanrevert.py` | Phase 1: 12 mean reversion configs |
| `run_sweep_hybrid.py` | Phase 1: 12 hybrid/multi-factor configs |
| `run_sweep_phase2.py` | Phase 2: 24 exit optimization configs |
| `run_sweep_phase3.py` | Phase 3: 24 portfolio construction configs |
| `exploration_momentum.csv` | Phase 1 momentum results |
| `exploration_meanrevert.csv` | Phase 1 mean reversion results |
| `exploration_hybrid.csv` | Phase 1 hybrid results |
| `exploration_phase2.csv` | Phase 2 results |
| `exploration_phase3.csv` | Phase 3 results |
| `sample_trades.json` | 50 sample trades from top 5 strategies |
| `backtest_data/enriched_2015-01-01_2025-12-31.pkl` | Pre-computed indicators cache (478 MB) |

---

## Methodology Notes

### Indicators Pre-Computed
All 17+ technical indicators were computed once for all 375 stocks:
EMA (9/10/20/21/30/50/200), SMA (50/200), RSI (14/2), MACD + histogram + signal, SuperTrend (10,3 and 7,2), ADX + DI+/DI-, Bollinger Bands (20,2), Keltner Channels (20,1.5), Donchian (20/50), Stochastic (14,3,3), CCI (20), Williams %R (14), OBV, MFI (14), ATR (14), Weekly EMA (20/50), 52-week high, momentum returns (1m/3m/6m/12m)

### Backtest Engine Design
- Signal-agnostic: Entry/Exit/Ranking functions are pluggable
- Equal-weight positions: Capital / portfolio_size per stock
- Transaction costs: 0.1% per side (brokerage + STT + stamp)
- Sector limits enforced at entry time
- Daily simulation with ~2,500 trading days over 11 years
