# Strategy Research Findings: Price-Based Indicator Combinations
> Scanned 107 strategies across 473 stocks (Nifty 500) | 3+ years daily data | Feb 2026

---

## Executive Summary

Tested **107 strategy combinations** using 20+ technical indicators across **473 Indian stocks** with 3+ years of daily OHLCV data. Strategies span mean reversion, trend following, breakout, multi-indicator, and top-down (weekly→daily) approaches.

### Key Finding
**Mean reversion strategies dominate for high win rates.** The Connors RSI(2) family consistently delivers 65-70% win rates across massive trade counts (15,000-40,000+ trades across all stocks). Keltner Channel mean reversion achieves the best risk-adjusted returns.

---

## Tier 1: HIGHEST WIN RATE (65%+ WR, 250+ trades)

| # | Strategy | Trades | WR% | PF | Avg Hold | Avg Win | Avg Loss | Symbols |
|---|----------|--------|-----|-----|----------|---------|----------|---------|
| 1 | **KC6 Lower + SMA200** | 1,007 | **69.02** | **1.79** | 5.7d | +4.76% | -5.91% | 333 |
| 2 | **CCI14 < -100 + EMA50** | 8,127 | **69.79** | 1.44 | 9.4d | +4.53% | -7.24% | 469 |
| 3 | **RSI2 < 5 + exit SMA10** | 22,044 | **69.49** | 1.30 | 5.2d | +3.05% | -5.33% | 468 |
| 4 | **BB10 Lower + EMA50** | 9,233 | **69.21** | 1.43 | 5.3d | +3.34% | -5.24% | 471 |
| 5 | **RSI+BB+MFI Triple Oversold** | 813 | **68.63** | 1.46 | 10.9d | +5.02% | -7.51% | 309 |
| 6 | **RSI5 < 20 + EMA50** | 9,606 | **67.76** | 1.46 | 7.2d | +4.14% | -5.98% | 471 |
| 7 | **Pullback to EMA50** | 13,634 | **67.51** | 1.33 | 11.4d | +4.68% | -7.29% | 469 |
| 8 | **RSI+CCI+WR Quad Oversold** | 10,208 | **67.46** | 1.33 | 3.7d | +2.73% | -4.24% | 466 |
| 9 | **RSI2 < 10 + ADX + EMA200** | 25,102 | **67.40** | 1.33 | 3.6d | +2.57% | -3.99% | 470 |
| 10 | **RSI2 < 5 + Weekly MACD bull** | 15,688 | **67.26** | 1.35 | 3.6d | +2.62% | -3.99% | 466 |
| 11 | **RSI2 < 5 + EMA Stack** | 15,145 | **67.02** | 1.34 | 3.7d | +2.59% | -3.93% | 465 |
| 12 | **RSI2 < 3 + SMA200** | 17,876 | **66.98** | 1.33 | 3.7d | +2.67% | -4.08% | 468 |
| 13 | **W:MACD+ADX D:Stoch+RSI** | 5,108 | **66.93** | 1.42 | 9.7d | +5.06% | -7.23% | 463 |
| 14 | **RSI2 < 5 + ADX Trending** | 21,660 | **66.66** | 1.29 | 3.6d | +2.61% | -4.04% | 468 |
| 15 | **RSI2 < 5 + SMA200 (Connors)** | 23,606 | **66.44** | 1.27 | 3.7d | +2.57% | -4.03% | 468 |

---

## Tier 2: BEST RISK-ADJUSTED RETURNS (Sharpe & Profit Factor)

| # | Strategy | Trades | WR% | PF | Sharpe | Total Ret% | MaxDD% |
|---|----------|--------|-----|-----|--------|------------|--------|
| 1 | **KC6 Lower + SMA200** | 1,007 | 69.02 | **1.79** | **1.48** | 1,464% | 67% |
| 2 | **WR2 < -98 + SMA200** | 659 | 58.73 | **1.67** | **1.34** | 897% | 97% |
| 3 | **BB10 Lower + EMA50** | 9,233 | 69.21 | 1.43 | **0.96** | 6,450% | 75% |
| 4 | **RSI2 < 5 + Weekly MACD** | 15,688 | 67.26 | 1.35 | **0.90** | 7,158% | 81% |
| 5 | **RSI5 < 20 + EMA50** | 9,606 | 67.76 | 1.46 | **0.89** | 8,444% | 84% |
| 6 | **ROC9 < -8 + SMA200** | 5,420 | 63.10 | **1.55** | **0.87** | 8,952% | 86% |
| 7 | **RSI2 < 10 + ADX + EMA200** | 25,102 | 67.40 | 1.33 | **0.87** | 10,906% | 81% |

---

## Tier 3: HIGHEST PROFIT FACTOR (trend-following style)

| # | Strategy | Trades | WR% | PF | Avg Win | Avg Loss |
|---|----------|--------|-----|-----|---------|----------|
| 1 | KC6 Lower + SMA200 | 1,007 | 69.02 | **1.79** | +4.76% | -5.91% |
| 2 | W:ST+ADX D:EMA+Vol | 2,903 | 41.37 | **1.71** | +12.55% | -5.18% |
| 3 | ST + EMA Aligned | 311 | 57.56 | **1.70** | +10.43% | -8.31% |
| 4 | HY: RSI2 entry, EMA exit | 14,603 | 44.72 | **1.69** | +10.42% | -5.00% |
| 5 | W:AllBull D:MACD+ST | 3,967 | 54.02 | **1.69** | +11.71% | -8.14% |
| 6 | CMF Positive + EMA50 | 9,875 | 50.87 | **1.68** | +10.05% | -6.20% |
| 7 | TSI + Supertrend + Volume | 10,782 | 49.94 | **1.66** | +12.22% | -7.34% |
| 8 | HY: BB entry, ST exit | 5,290 | 54.59 | **1.66** | +8.31% | -6.00% |
| 9 | MACD + RSI + Supertrend | 14,840 | 58.73 | **1.62** | +9.29% | -8.16% |
| 10 | EMA 9/21 Cross + Volume | 5,528 | 39.94 | **1.60** | +12.50% | -5.19% |

---

## Strategy Categories Analysis

### Category A: Mean Reversion (RSI-2/Connors Style)
**Best overall category. Highest win rates, most trades, most consistent.**

| Strategy | Trades | WR% | PF | Hold |
|----------|--------|-----|-----|------|
| RSI(2) < 5 + SMA200 | 23,606 | 66.44 | 1.27 | 3.7d |
| RSI(2) < 10 + EMA200 | 33,939 | 66.07 | 1.21 | 3.6d |
| RSI(2) < 5 + EMA Stack | 15,145 | 67.02 | 1.34 | 3.7d |
| RSI(2) < 10 + ADX + EMA200 | 25,102 | 67.40 | 1.33 | 3.6d |
| RSI(2) < 3 + SMA200 | 17,876 | 66.98 | 1.33 | 3.7d |

**Pattern**: Buy extreme short-term oversold (RSI-2 < 5-10) in long-term uptrend (above 200-day MA). Exit when price recovers to 5-day SMA. Hold ~3-4 days. Works because **extreme short-term dips in uptrending stocks tend to bounce**.

### Category B: Keltner/Bollinger Mean Reversion
**Highest individual WR and PF.**

| Strategy | Trades | WR% | PF | Hold |
|----------|--------|-----|-----|------|
| Keltner(6,1.3) + SMA200 | 1,007 | **69.02** | **1.79** | 5.7d |
| BB(10,1.5) + EMA50 | 9,233 | **69.21** | 1.43 | 5.3d |
| BB Lower + SMA200 | 5,385 | 64.18 | 1.44 | 8.9d |

**Pattern**: Price touches lower channel band while overall trend is up. Snaps back to middle band. Keltner(6, 1.3 ATR) is the tightest, producing fewer but higher-quality signals.

### Category C: Multi-Indicator Oversold Confirmation
**Triple/quad indicator oversold gives high confidence entries.**

| Strategy | Trades | WR% | PF | Hold |
|----------|--------|-----|-----|------|
| RSI+CCI+WR Quad | 10,208 | 67.46 | 1.33 | 3.7d |
| RSI+WR+Stoch Triple | 7,160 | 65.61 | 1.31 | 3.9d |
| RSI+BB+MFI Triple | 813 | 68.63 | 1.46 | 10.9d |

### Category D: Trend Following (Lower WR, Higher PF)
**Win 35-55% of the time, but winners are 2-3x the size of losers.**

| Strategy | Trades | WR% | PF | Avg Win | Avg Loss |
|----------|--------|-----|-----|---------|----------|
| EMA 9/21 + Volume | 5,528 | 39.94 | 1.60 | +12.50% | -5.19% |
| MACD + ST + ADX | 12,740 | 39.08 | 1.49 | +9.59% | -4.12% |
| ADX + MACD + EMA | 28,718 | 49.94 | 1.54 | +7.96% | -5.16% |

### Category E: Top-Down (Weekly Filter + Daily Entry)
**Weekly trend confirmation reduces noise; slightly fewer trades but better quality.**

| Strategy | Trades | WR% | PF | Hold |
|----------|--------|-----|-----|------|
| W:MACD+ADX D:Stoch+RSI | 5,108 | **66.93** | 1.42 | 9.7d |
| W:RSI D:BB Lower | 3,679 | 63.06 | 1.37 | 9.0d |
| W:Triple D:BB Lower | 3,106 | 63.62 | 1.45 | 8.9d |
| W:ST+ADX D:EMA+Vol | 2,903 | 41.37 | **1.71** | 17.3d |

### Category F: Hybrid (Mean Reversion Entry, Trend Exit)
**Interesting approach: Enter oversold, hold for trend. Lower WR, much higher PF.**

| Strategy | Trades | WR% | PF | Avg Win | Avg Loss |
|----------|--------|-----|-----|---------|----------|
| RSI2 entry → EMA exit | 14,603 | 44.72 | **1.69** | +10.42% | -5.00% |
| BB entry → ST exit | 5,290 | 54.59 | **1.66** | +8.31% | -6.00% |
| KC entry → RSI exit | 990 | 57.37 | **1.57** | +8.32% | -7.11% |

---

## Parameter Sensitivity Analysis

### RSI(2) Threshold Sensitivity
| RSI < | Trades | WR% | PF |
|-------|--------|-----|-----|
| 3 | 17,876 | 66.98 | 1.33 |
| 5 | 23,606 | 66.44 | 1.27 |
| 8 | 30,116 | 66.06 | 1.23 |
| 10 | 33,939 | 66.07 | 1.21 |
| 15 | 40,974 | 66.36 | 1.20 |

**Takeaway**: Stricter thresholds (RSI < 3) give higher WR and PF but fewer trades. RSI < 5 is the sweet spot for high volume.

### Stop Loss / Take Profit Sensitivity (RSI2 < 5 + SMA200)
| SL% | TP% | Trades | WR% | PF | Total Ret |
|-----|-----|--------|-----|-----|-----------|
| 5 | 10 | 24,536 | 65.61 | 1.21 | 7,381% |
| 6 | 12 | 24,089 | 66.09 | 1.23 | 7,809% |
| **8** | **15** | **23,606** | **66.44** | **1.27** | **8,485%** |
| 10 | 20 | 23,376 | 66.51 | 1.28 | 8,780% |
| 12 | 25 | 23,249 | 66.50 | 1.30 | 9,183% |
| 15 | 30 | 23,172 | 66.49 | 1.32 | 9,578% |

**Takeaway**: Wider stops improve PF and total return slightly. The 8/15 or 10/20 SL/TP ratios are the sweet spots.

### Hold Period Sensitivity (RSI2 < 5 + SMA200)
| Max Hold | Trades | WR% | PF |
|----------|--------|-----|-----|
| 5 days | 24,711 | 64.30 | 1.22 |
| 8 days | 23,748 | 66.30 | 1.26 |
| **10 days** | **23,606** | **66.44** | **1.27** |
| 15 days | 23,539 | 66.47 | 1.27 |
| 20+ days | 23,536 | 66.47 | 1.27 |

**Takeaway**: Most trades exit within 5-10 days. Holding beyond 10 days adds no benefit.

### Weekly Filter Impact (RSI2 < 5 + SMA200 base)
| Weekly Filter | Trades | WR% | PF | Sharpe |
|---------------|--------|-----|-----|--------|
| None | 23,606 | 66.44 | 1.27 | 0.70 |
| W:MACD Bullish | 15,688 | **67.26** | **1.35** | **0.90** |
| W:ADX Trending | 21,660 | 66.66 | 1.29 | 0.76 |
| W:Supertrend Bull | 21,923 | 66.45 | 1.27 | 0.71 |
| W:RSI Bullish | 20,445 | 66.41 | 1.28 | 0.72 |
| W:RSI+ST Double | 27,529 | 66.14 | 1.23 | 0.63 |

**Takeaway**: **Weekly MACD filter is the best weekly add-on**. Boosts WR from 66.44→67.26%, PF from 1.27→1.35, and Sharpe from 0.70→0.90. Trades drop by 33% but quality improves significantly.

---

## Top 10 Recommended Strategies

### For HIGH WIN RATE seekers:
1. **KC6 Lower + SMA200**: 69% WR, 1.79 PF, 5.7d hold (1,007 trades)
2. **CCI14 < -100 + EMA50**: 69.8% WR, 1.44 PF, 9.4d hold (8,127 trades)
3. **BB10 Lower + EMA50**: 69.2% WR, 1.43 PF, 5.3d hold (9,233 trades)

### For HIGH VOLUME seekers (most trades, still 65%+):
4. **RSI2 < 10 + ADX + EMA200**: 67.4% WR, 1.33 PF (25,102 trades)
5. **RSI2 < 5 + SMA200**: 66.4% WR, 1.27 PF (23,606 trades)
6. **RSI2 < 5 + Weekly MACD**: 67.3% WR, 1.35 PF (15,688 trades)

### For RISK-ADJUSTED seekers:
7. **ROC9 < -8 + SMA200**: 63.1% WR, 1.55 PF, 0.87 Sharpe (5,420 trades)
8. **RSI5 < 20 + EMA50**: 67.8% WR, 1.46 PF, 0.89 Sharpe (9,606 trades)

### For TREND FOLLOWERS:
9. **HY: RSI2 entry → EMA exit**: 44.7% WR, 1.69 PF, +10.4% avg win (14,603 trades)
10. **CMF Positive + EMA50**: 50.9% WR, 1.68 PF, +10.1% avg win (9,875 trades)

---

## Key Observations

1. **Mean reversion beats trend following for win rate** — RSI(2)/WR(2) in uptrend delivers 65-69% WR consistently across 468-471 symbols
2. **Trend following wins on profit factor** — Lower WR (35-55%) but 1.5-1.7 PF due to large winners
3. **Weekly MACD is the best weekly filter** — Boosts quality significantly (+1% WR, +28% Sharpe)
4. **Keltner(6, 1.3 ATR) is the best single setup** — 69% WR, 1.79 PF, highest Sharpe
5. **EMA stack (9>21>50>200) is the best trend filter** — Adds ~1% WR and +0.07 PF vs simple SMA200
6. **ADX > 25 as additional filter** — Consistently improves results (+0.5-1% WR)
7. **Hold period of 3-10 days is optimal** for mean reversion in Indian stocks
8. **Volume confirmation helps trend strategies** more than mean reversion
9. **Multiple oversold indicators together** (RSI + CCI + WR + Stoch) don't dramatically improve WR but add confidence
10. **The Calmar ratio challenge**: Per-trade sequential equity curves accumulate large drawdowns over 15K+ trades; portfolio-level Calmar would be much higher with proper capital allocation

---

## Trade Verification: Strategy #1 — KC6 Lower + SMA200

> Manual verification data for the top-ranked strategy. 10 random trades sampled across the full history (2006–2025).

### Strategy Rules
- **Entry**: Close < Keltner Channel(6, 1.3 ATR) Lower Band **AND** Close > SMA(200)
- **Exit**: Close > KC6 Middle Band **OR** Stop Loss 8% **OR** Take Profit 15% **OR** Max Hold 15 days

### Overall Stats
| Metric | Value |
|--------|-------|
| Total Trades | 1,004 |
| Wins / Losses | 693 / 311 |
| Win Rate | 69.0% |
| Avg Win | +4.77% |
| Avg Loss | -5.91% |
| Symbols Traded | 333 |

### Exit Reason Breakdown
| Exit Reason | Count | % of Total | Win Rate | Avg P/L | Avg Hold |
|-------------|-------|------------|----------|---------|----------|
| Signal (KC6 Mid) | 821 | 81.8% | 82.5% | +3.33% | 5.5d |
| Stop Loss (8%) | 151 | 15.0% | 0.0% | -10.04% | 6.1d |
| Take Profit (15%) | 15 | 1.5% | 100.0% | +21.00% | 2.1d |
| Max Hold (15d) | 17 | 1.7% | 5.9% | -4.06% | 15.0d |

### 10 Random Sample Trades (for manual chart verification)

| # | Result | Symbol | Entry Date | Entry Price | Exit Date | Exit Price | P/L% | P/L Abs | Hold | Exit Reason | KC6 Lower | KC6 Mid | SMA200 | RSI14 | ATR6 (%) |
|---|--------|--------|------------|-------------|-----------|------------|------|---------|------|-------------|-----------|---------|--------|-------|----------|
| 1 | LOSS | HAL | 2025-07-18 | 4651.70 | 2025-08-04 | 4537.90 | -2.45% | -113.80 | 11d | SIGNAL_KC6_MID | 4668.72 | 4807.25 | 4303.48 | 19.8 | 106.56 (2.3%) |
| 2 | WIN | MANKIND | 2025-01-13 | 2595.95 | 2025-01-20 | 2704.70 | +4.19% | +108.75 | 5d | SIGNAL_KC6_MID | 2608.79 | 2768.23 | 2406.43 | 23.3 | 122.65 (4.7%) |
| 3 | LOSS | MOIL | 2024-08-05 | 438.75 | 2024-08-21 | 397.95 | -9.30% | -40.80 | 11d | STOP_LOSS | 439.76 | 472.19 | 364.94 | 16.5 | 24.95 (5.7%) |
| 4 | WIN | PTC | 2024-06-04 | 180.45 | 2024-06-07 | 209.50 | +16.10% | +29.05 | 3d | TAKE_PROFIT | 181.30 | 207.93 | 180.24 | 21.5 | 20.48 (11.3%) |
| 5 | LOSS | SONATSOFTW | 2024-04-18 | 681.60 | 2024-04-26 | 681.05 | -0.08% | -0.55 | 6d | SIGNAL_KC6_MID | 681.95 | 711.62 | 648.68 | 22.5 | 22.82 (3.4%) |
| 6 | WIN | GAIL | 2024-03-13 | 162.55 | 2024-03-26 | 173.92 | +6.99% | +11.37 | 8d | SIGNAL_KC6_MID | 163.36 | 173.73 | 125.24 | 25.6 | 7.98 (4.9%) |
| 7 | WIN | LT | 2023-10-25 | 2916.10 | 2023-11-06 | 2975.45 | +2.04% | +59.35 | 8d | SIGNAL_KC6_MID | 2917.86 | 2993.16 | 2437.58 | 23.1 | 57.92 (2.0%) |
| 8 | WIN | SBIN | 2022-12-23 | 574.00 | 2022-12-26 | 597.10 | +4.02% | +23.10 | 1d | SIGNAL_KC6_MID | 574.28 | 593.44 | 522.87 | 19.8 | 14.73 (2.6%) |
| 9 | WIN | ANGELONE | 2021-08-23 | 1072.20 | 2021-08-30 | 1200.70 | +11.98% | +128.50 | 5d | SIGNAL_KC6_MID | 1072.99 | 1169.62 | 569.80 | 23.8 | 74.33 (6.9%) |
| 10 | WIN | MUTHOOTFIN | 2019-11-13 | 650.10 | 2019-11-14 | 711.50 | +9.44% | +61.40 | 1d | SIGNAL_KC6_MID | 650.62 | 673.78 | 606.60 | 25.6 | 17.81 (2.7%) |

### Verification Notes
- **7 wins, 3 losses** in this sample — consistent with the 69% win rate
- All entries show `close < KC6 Lower` and `close > SMA200` confirmed
- Signal exits (KC6 mid cross) dominate — the bounce-to-midline thesis holds
- RSI(14) at entry ranges 16–30, confirming oversold conditions at every entry
- ATR(6) as % of price ranges 2–11%, showing varied volatility contexts
- Best trade: PTC +16.10% in 3 days (hit 15% take profit)
- Worst trade: MOIL -9.30% (hit 8% stop loss during Aug 2024 selloff)
- Avg hold in sample: ~5.9 days — consistent with overall 5.7d average

### Win Rate by Year (selected)
| Year | Trades | WR% | Avg P/L |
|------|--------|-----|---------|
| 2021 | 95 | 80.0% | +2.85% |
| 2022 | 90 | 63.3% | +1.13% |
| 2023 | 83 | 71.1% | +1.70% |
| 2024 | 300 | 74.3% | +2.42% |
| 2025 | 142 | 60.6% | +0.05% |

### Top Symbols by Trade Count
| Symbol | Trades | WR% | Avg P/L | Total P/L |
|--------|--------|-----|---------|-----------|
| BAJFINANCE | 21 | 90.5% | +8.43% | +177.11% |
| RELIANCE | 20 | 90.0% | +1.98% | +39.65% |
| SBIN | 13 | 84.6% | +2.14% | +27.83% |
| TATASTEEL | 12 | 83.3% | +3.06% | +36.66% |
| BAJAJFINSV | 11 | 90.9% | +2.43% | +26.69% |

### Exit Mode Optimization: Close vs Limit Order

Compared two exit approaches for the KC6 mid signal:
- **V1 (Original)**: Exit when daily **close** > KC6 mid. Exit price = close.
- **V2 (Limit Order)**: Exit when daily **high** > KC6 mid. Exit price = KC6 mid value (simulates a limit sell order placed at the mid band each day).

| Metric | V1 (Close) | V2 (Limit) | Delta |
|--------|-----------|-----------|-------|
| Total Trades | 1,007 | 1,012 | +5 |
| **Win Rate** | 66.4% | **74.6%** | **+8.2%** |
| Avg Win | +4.78% | +3.95% | -0.83% |
| Avg Loss | -5.74% | -5.02% | +0.72% |
| **Avg P/L per trade** | +1.25% | **+1.67%** | **+0.42%** |
| **Avg Hold Days** | 5.3d | **3.4d** | **-1.9d** |
| **Profit Factor** | 1.65 | **2.31** | **+0.66** |
| **Total Cumulative P/L** | +1,257% | **+1,688%** | **+431%** |
| Max Hold exits | 14 | 0 | -14 |
| Stop Loss exits | 217 (21.5%) | 140 (13.8%) | -77 |

**Why V2 wins**: Exiting at the KC6 mid (limit order) catches the mean reversion earlier in the day. Price often touches the mid band intraday but closes back below it — V1 misses that exit, holds another day, and risks a reversal. V2 takes the profit at the target level immediately.

**Key improvements**:
- **WR jumps from 66% to 75%** — 8 percentage points
- **Hold period drops from 5.3 to 3.4 days** — capital freed up faster
- **77 fewer stop losses** — earlier exits prevent trades from deteriorating
- **Zero max-hold exits** — all trades resolve before 15 days
- **PF jumps from 1.65 to 2.31** — a massive improvement in risk/reward

**Recommended for live trading**: Use V2 — place a daily limit sell order at the KC6(6) mid band value. This is the realistic execution model.

#### V2 Sample Trades (10 random, limit order exit)

| # | Result | Symbol | Entry Date | Entry Price | Exit Date | Exit Price | P/L% | P/L Abs | Hold | Exit Reason | KC6 Lower | KC6 Mid (exit) | SMA200 | RSI14 | ATR6 (%) |
|---|--------|--------|------------|-------------|-----------|------------|------|---------|------|-------------|-----------|----------------|--------|-------|----------|
| 1 | WIN | INFY | 2024-10-31 | 1757.25 | 2024-11-06 | 1793.28 | +2.05% | +36.03 | 4d | SIGNAL_KC6_MID | 1764.70 | 1793.28 | 1675.23 | 19.3 | 42.61 (2.4%) |
| 2 | LOSS | GRANULES | 2024-09-12 | 564.90 | 2024-09-23 | 558.80 | -1.08% | -6.10 | 7d | SIGNAL_KC6_MID | 578.72 | 558.80 | 474.84 | 21.8 | 55.40 (9.8%) |
| 3 | LOSS | POONAWALLA | 2022-05-10 | 239.25 | 2022-05-12 | 220.11 | -8.00% | -19.14 | 2d | STOP_LOSS | 240.75 | 244.13 | 218.02 | 13.7 | 18.78 (7.9%) |
| 4 | WIN | PERSISTENT | 2022-04-18 | 2107.70 | 2022-04-20 | 2127.02 | +0.92% | +19.32 | 2d | SIGNAL_KC6_MID | 2114.97 | 2127.02 | 1962.93 | 23.2 | 79.35 (3.8%) |
| 5 | WIN | IRCTC | 2022-02-24 | 738.05 | 2022-02-25 | 792.83 | +7.42% | +54.78 | 1d | SIGNAL_KC6_MID | 738.72 | 792.83 | 680.62 | 24.9 | 39.42 (5.3%) |
| 6 | LOSS | MARUTI | 2020-09-21 | 6626.95 | 2020-09-28 | 6605.49 | -0.32% | -21.46 | 5d | SIGNAL_KC6_MID | 6628.47 | 6605.49 | 6214.40 | 19.4 | 239.33 (3.6%) |
| 7 | LOSS | ULTRACEMCO | 2017-09-22 | 3999.60 | 2017-09-27 | 3959.30 | -1.01% | -40.30 | 3d | SIGNAL_KC6_MID | 4006.57 | 3959.30 | 3915.09 | 28.2 | 97.54 (2.4%) |
| 8 | WIN | MARUTI | 2016-01-07 | 4267.90 | 2016-01-15 | 4294.41 | +0.62% | +26.51 | 6d | SIGNAL_KC6_MID | 4320.96 | 4294.41 | 4169.65 | 15.7 | 122.35 (2.9%) |
| 9 | WIN | ASIANPAINT | 2014-04-28 | 500.55 | 2014-05-02 | 511.83 | +2.25% | +11.28 | 3d | SIGNAL_KC6_MID | 500.99 | 511.83 | 487.55 | 24.6 | 15.82 (3.2%) |
| 10 | LOSS | MARUTI | 2010-01-06 | 1461.65 | 2010-01-15 | 1434.78 | -1.84% | -26.87 | 7d | SIGNAL_KC6_MID | 1463.21 | 1434.78 | 1249.71 | 20.1 | 45.37 (3.1%) |

**V2 sample notes**: 5 wins, 5 losses in this sample. Notice exit prices exactly match KC6 mid values (limit order fill). IRCTC +7.42% in 1 day is a classic bounce. POONAWALLA hit SL at exactly -8.00% (intraday low-based stop).

### Stop Loss Optimization (V2 Model)

Tested SL values from 3% to 8% while keeping V2 limit exit, TP=15%, Max Hold=15d constant.

#### Initial Run (3-year data, ~1K trades)

| SL% | Trades | WR% | Avg Win | Avg Loss | Avg P/L | PF | R:R | Exp(R) | Hold | SL Exits | Total P/L |
|-----|--------|-----|---------|----------|---------|-----|-----|--------|------|----------|-----------|
| 3% | 1,047 | 53.5% | +4.07% | -2.95% | +0.81% | 1.59 | **1.38** | +0.268R | 2.4d | 478 (45.7%) | +843% |
| **4%** | 1,039 | 62.7% | +4.05% | -3.74% | +1.14% | 1.81 | 1.08 | **+0.285R** | 2.7d | 358 (34.5%) | +1,184% |
| 5% | 1,025 | 68.6% | +4.00% | -4.33% | +1.38% | 2.01 | 0.92 | +0.276R | 3.0d | 271 (26.4%) | +1,415% |
| 6% | 1,017 | 71.4% | +3.99% | -4.64% | +1.52% | 2.14 | 0.86 | +0.253R | 3.1d | 211 (20.7%) | +1,544% |
| 8% | 1,012 | 74.6% | +3.95% | -5.02% | +1.67% | 2.31 | 0.79 | +0.209R | 3.4d | 140 (13.8%) | +1,688% |

#### Expanded Run (20-year data 2000-2025, ~2K trades) — Feb 2026

After backfilling historical data from 2005 (963K new rows, 1.8M total), re-ran the full SL sweep on 476 stocks across 20 years of market history including the 2008 GFC, 2020 COVID crash, and multiple bull/bear cycles.

| SL% | Trades | WR% | Avg Win | Avg Loss | Avg P/L | PF | R:R | Exp(R) | Hold | SL Exits | Total P/L |
|-----|--------|-----|---------|----------|---------|-----|-----|--------|------|----------|-----------|
| 3% | 1,999 | 49.0% | +4.56% | -2.97% | +0.72% | 1.48 | **1.54** | +0.240R | 2.2d | 1,007 (50.4%) | +1,441% |
| 4% | 1,982 | 57.4% | +4.47% | -3.84% | +0.93% | 1.57 | 1.16 | +0.232R | 2.5d | 805 (40.6%) | +1,839% |
| **5%** | 1,942 | 64.7% | +4.42% | -4.51% | +1.27% | 1.79 | 0.98 | **+0.253R** | 2.8d | 606 (31.2%) | +2,457% |
| 6% | 1,925 | 68.5% | +4.40% | -4.98% | +1.45% | 1.92 | 0.88 | +0.242R | 2.9d | 480 (24.9%) | +2,789% |
| 8% | 1,917 | 72.4% | +4.39% | -5.71% | +1.60% | 2.01 | 0.77 | +0.200R | 3.1d | 345 (18.0%) | +3,062% |

**Key: R:R** = Avg Win / |Avg Loss|. **Exp(R)** = Expected value per unit of risk (higher = more capital-efficient).

**Findings (20-year validation)**:
- **Strategy holds across 20 years** — all SL levels remain profitable with positive expectancy across GFC 2008, COVID 2020, and multiple cycles
- Trade counts nearly doubled (~1K to ~2K) confirming robust sample size
- **SL=5% emerges as the best balance** on expanded data — best expectancy (+0.253R), near 1:1 R:R (0.98), 64.7% WR, PF 1.79
- **SL=3%** still too tight — 50% of trades stopped out, WR drops below 50%
- **SL=8%** still maximizes total P/L (+3,062%) but R:R of 0.77 means each loss costs 1.3x a typical win
- **Avg win stays remarkably stable** (~+4.4%) across all SL levels — the exits are the same, only survival changes
- Win rates are slightly lower on 20-year data vs 3-year (e.g. SL=5%: 64.7% vs 68.6%) — the strategy was slightly overfitted to recent conditions but remains solidly profitable

**Recommendation**: **SL=5%** for live trading (revised from SL=4% based on expanded data). SL=5% gives the best capital efficiency (+0.253R), near-perfect 1:1 risk-reward, and 64.7% WR. For aggressive sizing, SL=6% offers higher PF (1.92) with acceptable R:R (0.88).

---

## TTM Squeeze Backtest — Validation Against Our Data

> Tested John Carter's TTM Squeeze (volatility breakout) strategy on the same Nifty 500 universe and 20-year dataset used for our KC6 mean reversion strategy. See `TTM_Squeeze_Deep_Dive_Report.docx` for the research brief.

### What Is TTM Squeeze?

The TTM Squeeze detects when Bollinger Bands contract inside Keltner Channels (low volatility "squeeze"), then trades the directional breakout when the squeeze "fires" (BB expands back outside KC). Entry on first green dot (BB exits KC) + positive rising momentum. Exit on momentum fading or zero-line cross.

### TTM Squeeze vs KC6 Mean Reversion — Key Difference

| Aspect | TTM Squeeze | KC6+SMA200 (Ours) |
|--------|-------------|-------------------|
| Type | **Breakout** (buy expansion) | **Mean Reversion** (buy dip) |
| Entry | Volatility expanding, momentum rising | Price at KC lower band, oversold |
| Exit | Momentum fading (2 bars) | Price bounces to KC mid (limit order) |
| KC usage | KC(20, 1.5) for squeeze detection | KC(6, 1.3) for entry/exit signals |
| Bollinger | BB(20, 2.0) vs KC for compression | Not used |

### Backtest Results (476 stocks, 2000-2025)

| Variant | Trades | WR% | Avg Win | Avg Loss | Avg P/L | PF | R:R | Hold | Total P/L |
|---------|--------|-----|---------|----------|---------|-----|-----|------|-----------|
| Carter Rules (5+ bars, SMA200) | 7,122 | 39.0% | +4.92% | -2.62% | +0.32% | 1.20 | 1.87 | 3.7d | +2,282% |
| No SMA200 Filter | 9,150 | 39.0% | +5.03% | -2.59% | +0.38% | 1.24 | 1.94 | 3.7d | +3,509% |
| 3+ Squeeze Bars (relaxed) | 9,338 | 38.8% | +5.01% | -2.63% | +0.33% | 1.20 | 1.90 | 3.7d | +3,076% |
| BB(1.5) Variant (Carter tip) | 10,716 | 38.2% | +4.89% | -2.62% | +0.25% | 1.15 | 1.87 | 3.5d | +2,660% |
| **KC6+SMA200 V2 (SL=5%)** | **1,942** | **64.7%** | **+4.42%** | **-4.51%** | **+1.27%** | **1.79** | **0.98** | **2.8d** | **+2,457%** |

### Key Findings

1. **TTM Squeeze "92% win rate" claim is busted** — actual WR on Indian stocks is **39%** across all variants. The report's cited backtests were cherry-picked single-stock or tiny-sample results.

2. **TTM Squeeze IS profitable** — but barely. PF of 1.20 means you earn $1.20 for every $1 risked. Our KC6 strategy has PF 1.79 — nearly 50% more efficient.

3. **R:R ratio is TTM's strength** — at 1.87, each win is almost 2x each loss. This compensates for the low 39% WR. Our KC6 strategy trades the opposite: higher WR (65%) but 1:1 R:R.

4. **Longer squeeze bars don't help** — contrary to Carter's claim that "more dots = bigger move," our data shows WR is essentially flat (38-41%) regardless of squeeze duration (5 bars vs 20+ bars).

5. **Total P/L is comparable** — TTM Squeeze (Carter Rules) generates +2,282% vs KC6 V2's +2,457%, but TTM needs 3.7x more trades (7,122 vs 1,942) to get there. KC6 is far more selective and capital-efficient.

6. **Year-by-year stability** — TTM Squeeze has negative years (2008: -1.10%, 2011: -0.14%, 2012: -0.25%, 2015: -0.20%, 2016: -0.22%). KC6 mean reversion tends to perform well in volatile/down years (more oversold dips to buy).

### Verdict: Not a Replacement, Potentially Complementary

**TTM Squeeze is NOT better than our KC6 strategy** for Indian stocks. KC6 wins on WR (65% vs 39%), PF (1.79 vs 1.20), avg P/L per trade (+1.27% vs +0.32%), and capital efficiency.

However, TTM Squeeze could complement KC6 as a **second strategy** because:
- They trade different market conditions (breakout vs mean reversion)
- They rarely overlap (KC6 enters on dips, TTM enters on breakouts)
- Combined, they could increase trade frequency and diversify signal sources

**For now: KC6+SMA200 V2 remains our primary strategy.** TTM Squeeze is a lower-priority secondary signal that needs further filtering to improve its 39% WR before live deployment.

---

## Data Coverage

### Current Database (after 2005 backfill — Feb 2026)

| Metric | Value |
|--------|-------|
| **Symbols** | 476 (Nifty 500 universe) |
| **Total daily rows** | 1,824,747 |
| **Date range** | 2000-01-03 to 2025-12-31 |
| **Database size** | 972.5 MB |

### Stock Coverage by Earliest Data Date

| Year Range | Stocks | % of Total |
|------------|--------|------------|
| 2000-2004 | 40 | 8.4% |
| 2005-2006 | 203 | 42.6% |
| 2007-2008 | 39 | 8.2% |
| 2009-2010 | 28 | 5.9% |
| 2011-2014 | 21 | 4.4% |
| 2015-2017 | 54 | 11.3% |
| 2018-2019 | 30 | 6.3% |
| 2020-2021 | 19 | 4.0% |
| 2022-2023 | 29 | 6.1% |
| 2024+ | 13 | 2.7% |
| **TOTAL** | **476** | |

**Key**: 51% of stocks (243) now have data from 2006 or earlier — a massive improvement from the previous 8.4% (40 stocks). The backfill added **963,427 new daily rows** (more than doubling the database from 768K to 1.8M rows).

### Indicators Computed
- EMA, SMA, RSI(2/5/14/21), Stochastic(5/14/21), Williams %R(2/5/14), CCI(14/20), MFI(10/14), MACD(12/26/9 + 8/17/9), ADX, Supertrend(7/2 + 10/3), Parabolic SAR, Bollinger(10/1.5 + 20/2), Keltner(6/1.3 + 20/2), Donchian(20), OBV, CMF, ROC(9/12), TSI, ATR
- **Weekly Indicators**: EMA50, EMA10/20, RSI14, MACD, ADX, Supertrend
- **Scan Time**: 3.5 minutes (vectorized backtesting)
