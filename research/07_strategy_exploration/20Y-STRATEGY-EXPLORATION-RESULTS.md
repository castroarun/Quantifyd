# 20-Year Strategy Exploration Results (2005-2025)

**Universe**: 375 Nifty 500 stocks | **Period**: Jan 2005 - Dec 2025 | **Capital**: Rs 1 Crore | **Portfolio Size**: 25 stocks

**Total configs tested**: 108 (48 EMA crossover + 36 price action + 24 long+short)
**Per-stock analysis**: 371 stocks across 9 strategies, 13,405 total trades

---

## 1. EMA Crossover Grid (48 Configs)

**8 EMA pairs** x **6 exit variants** (Trail 10/15/20%, ATH 15/20%, SL5+TP30)

### Best by Calmar (Risk-Adjusted)

| Rank | Config | CAGR% | P/L% | MaxDD% | Calmar | PF | WR% | Trades | Top3% |
|------|--------|-------|------|--------|--------|-----|-----|--------|-------|
| 1 | **EMA_20_50_Trail10** | 8.43 | 447 | 20.8 | **0.41** | 2.21 | 44.5 | 2,360 | 8.1 |
| 2 | EMA_20_50_Trail20 | 8.82 | 489 | 26.1 | 0.34 | 2.79 | 51.9 | 1,095 | 7.9 |
| 3 | EMA_20_50_Trail15 | 8.69 | 475 | 27.0 | 0.32 | 2.64 | 49.4 | 1,414 | 7.2 |
| 4 | EMA_20_50_ATH15 | 8.92 | 501 | 28.5 | 0.31 | 2.84 | 48.0 | 1,292 | 9.7 |
| 5 | EMA_13_34_Trail10 | 8.28 | 431 | 27.5 | 0.30 | 2.06 | 43.1 | 2,592 | 6.5 |
| 6 | EMA_13_34_SL5_TP30 | 8.56 | 461 | 28.4 | 0.30 | 1.90 | 30.1 | 3,224 | 5.5 |
| 7 | EMA_10_30_SL5_TP30 | 8.46 | 450 | 28.8 | 0.29 | 1.84 | 30.1 | 3,324 | 5.1 |
| 8 | EMA_10_50_SL5_TP30 | 8.21 | 424 | 28.6 | 0.29 | 1.81 | 29.9 | 3,246 | 5.2 |
| 9 | EMA_50_200_Trail10 | 6.61 | 284 | 22.9 | 0.29 | 2.01 | 42.1 | 1,762 | 8.6 |
| 10 | EMA_50_200_ATH15 | 7.93 | 397 | 27.5 | 0.29 | 2.62 | 45.6 | 1,112 | 10.8 |

### Key Findings - EMA Crossover

1. **EMA 20/50 is the best pair overall** — highest Calmar (0.41) with Trail10, lowest MaxDD (20.8%)
2. **Wider crossovers = lower drawdowns**: 20/50 (DD 20-31%), 50/200 (DD 22-34%) vs 5/20 (DD 38-47%)
3. **Trail 10% is the best exit for risk-adjusted returns** — forces faster exits, limits DD
4. **SL5+TP30 gives lowest concentration bias** (top3 share only 5-6%) but lower CAGR
5. **Fast crossovers (5/20, 8/21, 9/21) generate more trades but worse returns** — whipsaws hurt
6. **No EMA crossover system exceeded 9.2% CAGR** over 20 years on PS25

---

## 2. Price Action Indicators (36 Configs)

**12 indicators** x **3 exit variants** (Trail15, ATH20, Indicator-Matched Exit)

### Best by CAGR

| Rank | Config | CAGR% | P/L% | MaxDD% | Calmar | PF | WR% | Trades | Top3% |
|------|--------|-------|------|--------|--------|-----|-----|--------|-------|
| 1 | **PA_Stoch20_ATH20** | **10.45** | **706** | 33.7 | 0.31 | **3.28** | 47.8 | 1,153 | 8.2 |
| 2 | PA_WR_neg80_ATH20 | 10.45 | 706 | 33.7 | 0.31 | 3.28 | 47.8 | 1,153 | 8.2 |
| 3 | PA_Stoch20_Trail15 | 10.35 | 692 | 30.3 | 0.34 | 2.73 | 47.4 | 1,888 | 5.3 |
| 4 | PA_WR_neg80_Trail15 | 10.35 | 692 | 30.3 | 0.34 | 2.73 | 47.4 | 1,888 | 5.3 |
| 5 | PA_RSI2_10_Trail15 | 10.31 | 684 | 35.1 | 0.29 | 2.60 | 47.1 | 1,948 | 7.6 |
| 6 | PA_RSI2_10_ATH20 | 10.16 | 663 | 36.1 | 0.28 | 3.06 | 47.4 | 1,193 | 7.9 |
| 7 | PA_RSI14_30_ATH20 | 10.02 | 643 | 32.1 | 0.31 | 3.22 | 47.3 | 1,077 | 8.0 |
| 8 | PA_RSI14_30_Trail15 | 9.91 | 628 | 30.1 | 0.33 | 2.72 | 47.4 | 1,728 | 6.2 |
| 9 | PA_ADX30_Trail15 | 9.85 | 618 | 42.0 | 0.23 | 2.45 | 45.1 | 1,861 | 6.8 |
| 10 | PA_CCI_neg100_ATH20 | 9.62 | 587 | 31.2 | 0.31 | 3.39 | 50.2 | 928 | 9.3 |

### Best by Calmar

| Rank | Config | CAGR% | P/L% | MaxDD% | Calmar | PF | WR% | Trades |
|------|--------|-------|------|--------|--------|-----|-----|--------|
| 1 | **PA_SuperTrend_Trail15** | 5.38 | 201 | 15.2 | **0.35** | 2.19 | 43.5 | 749 |
| 2 | PA_SuperTrend_ATH20 | 7.12 | 324 | 20.3 | 0.35 | 2.79 | 45.7 | 634 |
| 3 | PA_Stoch20_Trail15 | 10.35 | 692 | 30.3 | 0.34 | 2.73 | 47.4 | 1,888 |
| 4 | PA_WR_neg80_Trail15 | 10.35 | 692 | 30.3 | 0.34 | 2.73 | 47.4 | 1,888 |
| 5 | PA_BB_Lower_ATH20 | 9.57 | 581 | 29.0 | 0.33 | 3.07 | 48.8 | 1,031 |

### Best by Profit Factor

| Rank | Config | PF | CAGR% | P/L% | WR% | Trades |
|------|--------|-----|-------|------|-----|--------|
| 1 | **PA_CCI_neg100_ATH20** | **3.39** | 9.62 | 587 | 50.2 | 928 |
| 2 | PA_Stoch20_ATH20 | 3.28 | 10.45 | 706 | 47.8 | 1,153 |
| 3 | PA_WR_neg80_ATH20 | 3.28 | 10.45 | 706 | 47.8 | 1,153 |
| 4 | PA_RSI14_30_ATH20 | 3.22 | 10.02 | 643 | 47.3 | 1,077 |
| 5 | PA_MFI20_ATH20 | 3.13 | 8.99 | 509 | 49.9 | 914 |

### Key Findings - Price Action

1. **Stochastic(20) and Williams%R(-80) produce identical results** — they are mathematically equivalent signals
2. **Mean reversion entries (RSI, Stoch, BB, CCI, WR) outperform trend entries** — ~10% CAGR vs ~9% for MACD/ADX
3. **ATH20 exit consistently achieves highest profit factors** (3.0-3.4) — lets winners run more
4. **Indicator-matched exits (RSI70, BB upper, Stoch80) hurt CAGR** — exit too early, reduce average win
5. **SuperTrend has lowest MaxDD (15.2%)** but also lowest CAGR (5.38%) — too few signals
6. **ADX30 entry has highest MaxDD (42-55%)** — ADX confirms trend but catches late entries before reversals

---

## 3. Long + Short Systems (24 Configs)

### All Results

| Rank | Config | CAGR% | P/L% | MaxDD% | Calmar | PF | WR% | Trades |
|------|--------|-------|------|--------|--------|-----|-----|--------|
| 1 | **LS_MACDSMA_ST_Trail15_PS30** | 6.12 | 248 | 17.1 | **0.36** | 1.88 | 43.9 | 1,570 |
| 2 | LS_MACDSMA_ST_Trail15_PS20 | 6.45 | 272 | 19.2 | 0.34 | 1.86 | 42.8 | 1,167 |
| 3 | LS_MACDSMA_ST_ATH20_PS30 | 6.61 | 283 | 21.1 | 0.31 | 2.37 | 43.5 | 1,218 |
| 4 | LS_MACDSMA_ST_ATH20_PS20 | 6.34 | 264 | 21.3 | 0.30 | 2.10 | 42.6 | 966 |
| 5 | LS_EMA2050_Trail20_PS40 | 3.44 | 103 | 13.6 | 0.25 | 1.24 | 38.6 | 3,918 |
| 6 | LS_EMA2050_Trail15_PS40 | 3.84 | 121 | 17.7 | 0.22 | 1.28 | 39.5 | 3,309 |
| 7 | LS_ADX_RSI_Trail15_PS20 | 5.88 | 232 | 27.2 | 0.22 | 1.39 | 39.0 | 2,074 |
| 8 | LS_EMA2050_Trail15_PS30 | 3.63 | 112 | 17.7 | 0.21 | 1.26 | 39.8 | 2,572 |

### Key Findings - Long+Short

1. **MACD+SMA200 long / SuperTrend short is the best L/S combination** — 6.12-6.61% CAGR with 17-21% MaxDD
2. **Short-selling drag is significant** — best L/S at 6.6% CAGR vs 8.8% for long-only EMA 20/50
3. **Shorts have low win rates (~30-35%)** — Indian market's long-term upward bias hurts shorts
4. **L/S does reduce MaxDD**: best L/S MaxDD = 13.6-17.1% vs best long-only MaxDD = 20.8%
5. **Pure EMA crossover L/S is weak** (1.7-3.8% CAGR) — trend-following doesn't translate well to shorts
6. **ADX long + RSI overbought short decent** but high drawdowns (27-37%)

---

## 4. Per-Stock Signal Analysis

### Top 20 Stocks (Best Total P/L Across All 9 Strategies)

| Rank | Symbol | Trades | WR% | Total P/L% | Avg P/L% | Active In | Best Strategy |
|------|--------|--------|-----|-----------|----------|-----------|---------------|
| 1 | **ADANIGREEN** | 47 | 55.3 | **2,450** | 52.1 | 9/9 | ADX25 (+521%) |
| 2 | SHREECEM | 65 | 63.1 | 2,164 | 33.3 | 8/9 | SMA200_Mom |
| 3 | AUROPHARMA | 39 | 71.8 | 1,975 | 50.6 | 9/9 | SMA200_Mom |
| 4 | GRANULES | 54 | 51.9 | 1,745 | 32.3 | 9/9 | RSI30 |
| 5 | AJANTPHARM | 67 | 62.7 | 1,680 | 25.1 | 9/9 | SMA200_Mom |
| 6 | HEG | 54 | 50.0 | 1,579 | 29.2 | 9/9 | ADX25 |
| 7 | RELIANCE | 61 | 68.9 | 1,566 | 25.7 | 9/9 | RSI30 |
| 8 | DEEPAKNTR | 60 | 51.7 | 1,564 | 26.1 | 9/9 | MACD |
| 9 | EICHERMOT | 73 | 65.8 | 1,546 | 21.2 | 9/9 | MACD_SMA200 |
| 10 | DIXON | 36 | 58.3 | 1,514 | 42.1 | 9/9 | SMA200_Mom |
| 11 | ASTRAL | 38 | 63.2 | 1,509 | 39.7 | 9/9 | SMA200_Mom |
| 12 | ADANIENT | 83 | 33.7 | 1,505 | 18.1 | 9/9 | SMA200_Mom |
| 13 | SRF | 55 | 58.2 | 1,279 | 23.3 | 9/9 | RSI30 |
| 14 | GMDCLTD | 60 | 36.7 | 1,268 | 21.1 | 9/9 | MACD_SMA200 |
| 15 | ITC | 69 | 75.4 | 1,256 | 18.2 | 9/9 | EMA_9/21 |
| 16 | IPCALAB | 54 | 74.1 | 1,250 | 23.2 | 9/9 | ADX25 |
| 17 | DIVISLAB | 75 | 62.7 | 1,245 | 16.6 | 9/9 | SMA200_Mom |
| 18 | CERA | 41 | 68.3 | 1,237 | 30.2 | 9/9 | ADX25 |
| 19 | KPRMILL | 49 | 61.2 | 1,140 | 23.3 | 9/9 | MACD |
| 20 | BHARTIARTL | 69 | 60.9 | 1,135 | 16.4 | 9/9 | MACD |

### Bottom 5 Stocks (Consistent Losers)

| Rank | Symbol | Trades | WR% | Total P/L% | Avg P/L% |
|------|--------|--------|-----|-----------|----------|
| 367 | PNB | 25 | 28.0 | -216 | -8.6 |
| 368 | NFL | 35 | 20.0 | -237 | -6.8 |
| 369 | ASHOKA | 20 | 15.0 | -241 | -12.1 |
| 370 | BASF | 40 | 22.5 | -244 | -6.1 |
| 371 | NATIONALUM | 45 | 26.7 | -252 | -5.6 |

### Key Findings - Per-Stock

1. **ADANIGREEN is the #1 stock across ALL strategies** — 2,450% total P/L, 52.1% avg per trade
2. **Top 3 stocks by avg P/L per trade**: ADANIGREEN (52.1%), AUROPHARMA (50.6%), DIXON (42.1%)
3. **Most consistent winners**: ITC (75.4% WR), IPCALAB (74.1% WR), LINDEINDIA (75.0% WR)
4. **Highest trade count leaders**: ADANIENT (83), DIVISLAB (75), EICHERMOT (73), BAJFINANCE (92)
5. **PSU/metal stocks dominate bottom**: NATIONALUM, NFL, PNB, SAIL, BPCL all net negative
6. **All top 20 stocks are active in 8-9 of 9 strategies** — signal-agnostic alpha generators

---

## 5. Winning System Rules

### System 1: PA_Stoch20_ATH20 (Best CAGR: 10.45%)

| Metric | Value |
|--------|-------|
| CAGR | 10.45% |
| MaxDD | 33.7% |
| Calmar | 0.31 |
| Profit Factor | 3.28 |
| Win Rate | 47.8% |
| Total Trades | 1,153 |
| Avg Win | +50.2% |
| Avg Loss | -14.0% |
| Top 3 Share | 8.2% |
| CAGR ex Top3 | 10.21% |

**Rules:**
- **Entry**: Stochastic K(14,3) < 20 (oversold) AND Close > SMA(200) (uptrend filter)
- **Exit**: ATH drawdown 20% from peak since entry OR time limit 365 days
- **Ranking**: 12-month momentum (best movers first)
- **Portfolio**: 25 stocks, monthly rebalance, max 25% sector, Rs 1 Cr capital

**Sample Trades:**

| Symbol | Entry | Price | Exit | Price | Reason | Days | P/L% |
|--------|-------|-------|------|-------|--------|------|------|
| TRIDENT | 2014-02-14 | 1.45 | 2014-12-01 | 2.65 | ATH_DD_20pct | 290 | +82.8% |
| ASTRAZEN | 2009-04-08 | 520.00 | 2010-02-02 | 826.80 | ATH_DD_20pct | 300 | +59.0% |
| NAVINFLUOR | 2016-02-01 | 308.31 | 2016-11-21 | 443.91 | ATH_DD_20pct | 294 | +44.0% |
| BRITANNIA | 2005-10-13 | 122.12 | 2006-05-19 | 150.16 | ATH_DD_20pct | 218 | +23.0% |
| INDHOTEL | 2014-07-17 | 89.05 | 2015-05-26 | 96.95 | ATH_DD_20pct | 313 | +8.9% |
| TORNTPHARM | 2006-01-12 | 52.40 | 2006-03-07 | 51.95 | ATH_DD_20pct | 54 | -0.9% |
| WIPRO | 2018-12-21 | 118.75 | 2019-09-23 | 117.35 | ATH_DD_20pct | 276 | -1.2% |
| BLUESTARCO | 2025-01-13 | 1,853.40 | 2025-04-25 | 1,739.20 | ATH_DD_20pct | 102 | -6.2% |
| BLUESTARCO | 2008-03-18 | 192.80 | 2008-07-01 | 172.25 | ATH_DD_20pct | 105 | -10.7% |
| LT | 2008-02-11 | 737.21 | 2008-03-10 | 606.27 | ATH_DD_20pct | 28 | -17.8% |

---

### System 2: EMA_20_50_Trail10 (Best Calmar: 0.41)

| Metric | Value |
|--------|-------|
| CAGR | 8.43% |
| MaxDD | 20.8% |
| Calmar | 0.41 |
| Profit Factor | 2.21 |
| Win Rate | 44.5% |
| Total Trades | 2,360 |
| Avg Win | +21.5% |
| Avg Loss | -7.8% |
| Top 3 Share | 8.1% |
| CAGR ex Top3 | 8.28% |

**Rules:**
- **Entry**: EMA(20) crosses above EMA(50) (golden crossover)
- **Exit**: Trailing stop 10% from peak since entry OR time limit 252 days
- **Ranking**: 12-month momentum
- **Portfolio**: 25 stocks, monthly rebalance

**Sample Trades:**

| Symbol | Entry | Price | Exit | Price | Reason | Days | P/L% |
|--------|-------|-------|------|-------|--------|------|------|
| RELIANCE | 2005-06-08 | 51.20 | 2006-02-15 | 83.60 | Time_252d | 252 | +63.3% |
| NTPC | 2007-03-13 | 109.12 | 2007-10-19 | 154.39 | Trail_10pct | 220 | +41.5% |
| RALLIS | 2019-07-12 | 154.25 | 2019-12-16 | 168.80 | Trail_10pct | 157 | +9.4% |
| SONATSOFTW | 2010-09-16 | 19.10 | 2010-11-16 | 20.60 | Trail_10pct | 61 | +7.9% |
| ASHOKLEY | 2021-09-22 | 61.70 | 2021-11-22 | 64.35 | Trail_10pct | 61 | +4.3% |
| DRREDDY | 2022-10-03 | 883.95 | 2023-05-12 | 893.20 | Trail_10pct | 221 | +1.0% |
| BAJAJHLDNG | 2007-07-25 | 868.27 | 2007-11-12 | 858.71 | Trail_10pct | 110 | -1.1% |
| APLAPOLLO | 2014-01-14 | 15.10 | 2014-02-12 | 13.80 | Trail_10pct | 29 | -8.6% |
| HEIDELBERG | 2013-05-17 | 46.37 | 2013-05-23 | 40.90 | Trail_10pct | 6 | -11.8% |
| DELTACORP | 2010-04-30 | 39.90 | 2010-05-07 | 34.60 | Trail_10pct | 7 | -13.3% |

---

### System 3: PA_RSI2_10_Trail15 (Highest Trade Count Mean Reversion)

| Metric | Value |
|--------|-------|
| CAGR | 10.31% |
| MaxDD | 35.1% |
| Calmar | 0.29 |
| Profit Factor | 2.60 |
| Win Rate | 47.1% |
| Total Trades | 1,948 |
| Avg Win | +32.9% |
| Avg Loss | -11.2% |
| Top 3 Share | 7.6% |
| CAGR ex Top3 | 10.08% |

**Rules:**
- **Entry**: RSI(2) < 10 (extreme oversold) AND Close > SMA(200) (uptrend)
- **Exit**: Trailing stop 15% from peak since entry OR time limit 252 days
- **Ranking**: 12-month momentum
- **Portfolio**: 25 stocks, monthly rebalance

**Sample Trades:**

| Symbol | Entry | Price | Exit | Price | Reason | Days | P/L% |
|--------|-------|-------|------|-------|--------|------|------|
| BHARTIARTL | 2005-03-29 | 101.00 | 2005-12-06 | 167.55 | Time_252d | 252 | +65.9% |
| CENTURYPLY | 2021-06-28 | 404.75 | 2021-11-29 | 586.20 | Trail_15pct | 154 | +44.8% |
| MARICO | 2007-03-12 | 25.91 | 2007-11-19 | 33.44 | Time_252d | 252 | +29.1% |
| KTKBANK | 2023-11-07 | 204.95 | 2024-02-12 | 231.50 | Trail_15pct | 97 | +13.0% |
| NESTLEIND | 2011-10-19 | 206.20 | 2012-06-27 | 225.00 | Time_252d | 252 | +9.1% |
| ZENSARTECH | 2010-05-10 | 30.64 | 2010-11-03 | 30.40 | Trail_15pct | 177 | -0.8% |
| JINDALSTEL | 2010-01-29 | 627.95 | 2010-05-19 | 618.35 | Trail_15pct | 110 | -1.5% |
| OFSS | 2008-07-29 | 906.50 | 2008-08-20 | 852.50 | Trail_15pct | 22 | -6.0% |
| CIPLA | 2008-06-25 | 207.40 | 2008-10-10 | 194.40 | Trail_15pct | 107 | -6.3% |
| SAIL | 2008-01-21 | 203.50 | 2008-02-11 | 184.85 | Trail_15pct | 21 | -9.2% |

---

### System 4: LS_MACDSMA_ST_Trail15_PS30 (Best Long+Short)

| Metric | Value |
|--------|-------|
| CAGR | 6.12% |
| MaxDD | 17.1% |
| Calmar | 0.36 |
| Profit Factor | 1.88 |
| Win Rate | 43.9% |
| Total Trades | 1,570 |
| Top 3 Share | 11.1% |
| CAGR ex Top3 | 5.88% |

**Rules:**
- **Long Entry**: MACD crosses above signal line AND Close > SMA(200)
- **Long Exit**: Trailing stop 15% from peak OR time limit 252 days
- **Short Entry**: SuperTrend(10,3) flips to bearish
- **Short Exit**: Trailing stop 15% (from trough) OR fixed SL 10% OR time limit 126 days
- **Portfolio**: 30 stocks (15 long + 15 short), monthly rebalance
- **Short Ranking**: Weakest 12-month momentum stocks first

---

## 6. Cross-Strategy Analysis

### What Works Over 20 Years

| Principle | Evidence |
|-----------|----------|
| **Mean reversion in uptrend** | RSI, Stoch, BB entries in uptrend (>SMA200) achieve 9.5-10.5% CAGR |
| **Wider EMA crossovers** | 20/50 and 13/34 outperform 5/20 and 8/21 on risk-adjusted basis |
| **Trail 10% for risk control** | Tight trailing stop gives lowest MaxDD (20.8%) but slightly lower CAGR |
| **ATH 20% for max CAGR** | ATH drawdown exit lets winners run, achieves highest profit factors (3.0+) |
| **Uptrend filter is critical** | All entries require Close > SMA(200) — without this, drawdowns explode |
| **12-month momentum ranking** | Consistently the best stock selection method across all systems |

### What Doesn't Work

| Approach | Problem |
|----------|---------|
| **Fast EMA crossovers (5/20, 8/21)** | Whipsaw losses, 40-47% MaxDD, low CAGR |
| **Indicator-matched exits** (RSI70, BB upper) | Exit too early, reduce average win size significantly |
| **Pure short-selling** | Indian market's structural uptrend makes shorts a drag |
| **ADX entry alone** | High MaxDD (42-55%), catches trends late before reversal |
| **SL5 + TP30 fixed exits** | Too mechanical, ~30% win rate despite decent CAGR due to skew |

### Concentration Bias Assessment

All strategies show **low concentration** (top 3 stocks contribute 5-12% of total P/L):

| Metric | Best Systems |
|--------|-------------|
| Lowest Top3 Share | PA_Stoch20_Trail15 (5.3%) |
| Avg Top3 Share | ~7-9% across all systems |
| CAGR ex Top3 Drop | Typically 0.2-0.5% — strategies are robust |

**Verdict**: No concentration bias concern. CAGR-ex-top3 remains within 0.5% of headline CAGR for all top systems.

---

## 7. Summary Comparison Table

| System | Type | CAGR% | P/L% | MaxDD% | Calmar | PF | WR% | Trades |
|--------|------|-------|------|--------|--------|-----|-----|--------|
| PA_Stoch20_ATH20 | Mean Revert | **10.45** | **706** | 33.7 | 0.31 | 3.28 | 47.8 | 1,153 |
| PA_RSI2_10_Trail15 | Mean Revert | 10.31 | 684 | 35.1 | 0.29 | 2.60 | 47.1 | 1,948 |
| PA_RSI14_30_ATH20 | Mean Revert | 10.02 | 643 | 32.1 | 0.31 | 3.22 | 47.3 | 1,077 |
| PA_ADX30_Trail15 | Trend | 9.85 | 618 | 42.0 | 0.23 | 2.45 | 45.1 | 1,861 |
| EMA_20_50_Trail10 | Trend | 8.43 | 447 | **20.8** | **0.41** | 2.21 | 44.5 | 2,360 |
| EMA_20_50_ATH15 | Trend | 8.92 | 501 | 28.5 | 0.31 | 2.84 | 48.0 | 1,292 |
| LS_MACDSMA_ST_PS30 | L+S | 6.12 | 248 | 17.1 | 0.36 | 1.88 | 43.9 | 1,570 |

**Recommendation**:
- **For max CAGR**: PA_Stoch20_ATH20 (10.45%, but 33.7% MaxDD)
- **For best risk-adjusted**: EMA_20_50_Trail10 (8.43%, only 20.8% MaxDD)
- **For lowest drawdown**: LS_MACDSMA_ST_PS30 (6.12%, only 17.1% MaxDD)
- **For diversification**: Combine EMA crossover (trend) + RSI2 (mean reversion) signals

---

*Generated: Feb 2026 | Engine: strategy_backtest.py | Data: market_data.db (2000-2025)*
