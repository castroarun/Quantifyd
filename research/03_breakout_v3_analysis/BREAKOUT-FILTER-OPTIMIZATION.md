# Breakout Filter Optimization Report

**Date:** 2026-02-12
**Universe:** 457 stocks, 9,739 breakout trades
**Data Range:** Jul 2000 - Dec 2025
**Detectors:** Darvas Box (4,791 trades) + Flat-Range Consolidation (4,948 trades)

---

## Glossary of Key Metrics

### Profit Factor (PF)

**Profit Factor = Total Gross Profits / Total Gross Losses**

It measures how many rupees you earn for every rupee you lose. A PF of 3.0 means for every Rs.1 lost, you earned Rs.3.

| PF Value | Interpretation |
|----------|---------------|
| < 1.0 | Losing system - losses exceed gains |
| 1.0 - 1.5 | Break-even to marginal |
| 1.5 - 2.0 | Decent edge |
| 2.0 - 3.0 | Strong edge |
| > 3.0 | Excellent edge |

**Example:** If your 10 winning trades made a total of Rs.50,000 and your 8 losing trades lost a total of Rs.15,000, then PF = 50,000 / 15,000 = **3.33**

> Profit Factor is more reliable than Win Rate alone because it accounts for the *size* of wins vs losses. A system with 40% win rate but PF of 4.0 (big wins, small losses) is better than one with 60% win rate but PF of 1.2 (many small wins, a few devastating losses).

---

### Calmar Ratio

**Calmar Ratio = CAGR / Maximum Drawdown**

It measures how much annual return you earn per unit of maximum pain (drawdown). Higher is better.

| Calmar | Interpretation |
|--------|---------------|
| < 0.5 | Poor risk-adjusted return |
| 0.5 - 1.0 | Acceptable |
| 1.0 - 2.0 | Good |
| > 2.0 | Excellent |

**Example:** A strategy with 25% CAGR and 20% Max Drawdown has Calmar = 25/20 = **1.25**. Another strategy with 15% CAGR and 8% Max Drawdown has Calmar = 15/8 = **1.88** - the second strategy is *better risk-adjusted* despite lower absolute returns.

**How we compute it here:** We simulate an equal-dollar portfolio where each breakout trade gets the same capital allocation. Trades are executed chronologically. We then measure the equity curve's CAGR and worst peak-to-trough drawdown.

> Calmar is preferred over Sharpe ratio for trend-following and breakout systems because it penalizes the *worst case* drawdown rather than daily volatility. A breakout system with occasional large drawdowns but strong recovery has a low Calmar (bad), while one with consistent gains and shallow drawdowns has a high Calmar (good).

---

### Other Metrics Used

| Metric | Definition |
|--------|-----------|
| **Win %** | Percentage of trades that ended with positive return |
| **Stop %** | Percentage of trades that hit the stop loss (box bottom / zone low) |
| **Avg Return** | Mean return across all trades (winners + losers) |
| **Median Return** | Middle value of all returns (less sensitive to outliers than mean) |
| **CAGR** | Compound Annual Growth Rate of the simulated equity curve |
| **Max DD** | Maximum peak-to-trough drawdown of the simulated equity curve |
| **Avg Gain** | Mean return of winning trades only |
| **Avg Loss** | Mean return of losing trades only |

---

## 1. Baseline: Unfiltered Universe

| Metric | Value |
|--------|-------|
| Total trades | 9,739 |
| Win rate | 46.8% |
| Stop rate | 47.2% |
| Avg return | +9.7% |
| Median return | **-4.1%** |
| Profit Factor | 2.72 |
| CAGR | 30.9% |
| Max Drawdown | 129.9% |
| Calmar | **0.24** |

The unfiltered universe has a **negative median return** - more than half the trades lose money. The massive 130% drawdown and Calmar of 0.24 confirms: **filtering is essential**.

---

## 2. All Filter Strategies Ranked

### Top 15 by Win Rate

| Rank | Filter | Trades | Stop% | Win% | Avg Ret | PF | CAGR | MaxDD | Calmar |
|------|--------|--------|-------|------|---------|-----|------|-------|--------|
| 1 | **RSI>70 + Vol>3x + BO>3% + EMA20>50** | 70 | 38.6% | **57.1%** | +17.2% | 3.85 | 12.3% | **16.0%** | **0.77** |
| 2 | RSI>70 + Vol>3x + BO>3% | 75 | 40.0% | 56.0% | +15.7% | 3.46 | 12.2% | 25.3% | 0.48 |
| 3 | RSI>70 + Vol>3x | 147 | 38.8% | 55.8% | +13.1% | 3.33 | 14.4% | 24.1% | 0.60 |
| 4 | RSI>60 + Vol>5x + BO>3% + EMA20>50 | 391 | 38.4% | 54.0% | +16.1% | 3.59 | 20.4% | 25.2% | 0.81 |
| 5 | Vol>5x + BO>5% | 536 | **31.5%** | 53.9% | +14.7% | 3.33 | 21.6% | 24.7% | 0.88 |
| 6 | Vol>5x + BO>3% | 991 | 34.6% | 53.7% | +13.7% | 3.33 | 24.4% | **19.9%** | **1.22** |
| 7 | RSI>60 + Vol>3x + BO>3% + EMA20>50 + Mom>0 | 752 | 39.2% | 53.6% | +16.2% | **3.61** | 22.5% | 19.0% | **1.18** |
| 8 | Vol>5x + BO>5% + EMA20>50 | 417 | 34.3% | 53.5% | +15.3% | 3.32 | 20.5% | **18.0%** | **1.14** |
| 9 | **RSI>60 + Vol>3x + BO>3% + EMA20>50** | 775 | 39.6% | 53.2% | **+16.0%** | **3.56** | 22.5% | 20.4% | **1.11** |
| 10 | RSI>70 + Vol>5x + BO>3% | 32 | 40.6% | 53.1% | **+19.3%** | **3.87** | 10.3% | 53.8% | 0.19 |
| 11 | Near ATH + RSI>60 + Vol>3x + BO>3% | 869 | 39.5% | 52.5% | +15.1% | 3.42 | 22.8% | 26.2% | 0.87 |
| 12 | RSI>60 + Vol>3x + BO>3% | 869 | 39.5% | 52.5% | +15.1% | 3.42 | 22.8% | 26.2% | 0.87 |
| 13 | Mom dual + RSI>60 + Vol>3x | 1,545 | 40.8% | 52.4% | +13.9% | 3.44 | 25.4% | 26.2% | 0.97 |
| 14 | Near ATH + Vol>3x + BO>3% | 1,717 | 37.9% | 52.2% | +14.0% | 3.32 | 26.0% | 32.0% | 0.81 |
| 15 | RSI>60 + BO>3% + EMA20>50 | 1,195 | 41.1% | 52.1% | +15.1% | 3.40 | 24.5% | 27.5% | 0.89 |

### Top 10 by Calmar Ratio (risk-adjusted)

| Rank | Filter | Trades | Win% | Avg Ret | PF | CAGR | MaxDD | Calmar |
|------|--------|--------|------|---------|-----|------|-------|--------|
| 1 | **Vol>5x + BO>3%** | 991 | 53.7% | +13.7% | 3.33 | 24.4% | **19.9%** | **1.22** |
| 2 | RSI>60 + Vol>3x + BO>3% + EMA20>50 + Mom>0 | 752 | 53.6% | +16.2% | 3.61 | 22.5% | 19.0% | **1.18** |
| 3 | EMA20>50 + Vol>3x | 3,193 | 50.8% | +12.8% | 3.28 | 28.7% | 24.4% | **1.18** |
| 4 | Vol>5x + BO>5% + EMA20>50 | 417 | 53.5% | +15.3% | 3.32 | 20.5% | 18.0% | **1.14** |
| 5 | **RSI>60 + Vol>3x + BO>3% + EMA20>50** | 775 | 53.2% | +16.0% | 3.56 | 22.5% | 20.4% | **1.11** |
| 6 | Mom20>0 + Mom60>0 + Vol>3x | 2,691 | 51.5% | +13.3% | 3.39 | 28.1% | 27.6% | 1.02 |
| 7 | RSI>60 + Vol>3x + EMA20>50 | 1,587 | 51.9% | +13.8% | 3.41 | 25.5% | 25.2% | 1.01 |
| 8 | Near ATH + Vol>3x | 4,086 | 49.1% | +11.8% | 3.06 | 29.6% | 30.0% | 0.99 |
| 9 | RSI>60 + BO>3% | 1,313 | 51.6% | +14.6% | 3.31 | 24.8% | 25.3% | 0.98 |
| 10 | Mom dual + RSI>60 + Vol>3x | 1,545 | 52.4% | +13.9% | 3.44 | 25.4% | 26.2% | 0.97 |

---

## 3. Key Findings

### 3.1 Volume is the #1 Stop-Loss Killer

| Volume Filter | Stop Rate | Win Rate | Avg Return |
|---------------|-----------|----------|------------|
| No filter | 47.2% | 46.8% | +9.7% |
| Vol > 3x | ~42% | ~51% | ~+13% |
| Vol > 5x + BO>3% | **34.6%** | **53.7%** | +13.7% |
| Vol > 5x + BO>5% | **31.5%** | **53.9%** | +14.7% |

High volume breakouts fail less often because **institutional participation confirms the move**. The Vol>5x + BO>3% combination has the best Calmar ratio (1.22) in the entire study - lowest drawdown relative to return.

### 3.2 RSI at Entry: Higher = Better (Counterintuitive)

Counterintuitively, "overbought" RSI at breakout **dramatically improves** results:

| RSI Filter | Win Rate | Trades | PF | Notes |
|------------|----------|--------|-----|-------|
| No RSI filter | 46.8% | 9,739 | 2.72 | |
| RSI >= 60 | ~48% | ~5,000 | ~2.93 | |
| RSI >= 65 | 51.4% | 1,883 | 3.24 | |
| RSI >= 70 | 56.3% | 446 | 3.76 | |
| **RSI >= 75** | **63.6%** | **77** | **4.67** | **Each +5 RSI = ~+5% win rate** |
| RSI >= 75 + Vol>=3x | **71.8%** | 39 | **7.96** | |
| RSI >= 75 + Vol>=3x + EMA20>50 | **73.7%** | 38 | **10.37** | |

These are momentum breakouts where **strength begets strength**. A stock already showing RSI >= 75 at the moment of a volume-confirmed breakout has overwhelming buying pressure. The RSI threshold is the single most powerful lever - each +5 RSI adds roughly +5% to win rate.

### 3.3 EMA20 > EMA50 Adds Risk-Adjusted Value

Adding the EMA20>50 crossover filter consistently improves the Calmar ratio:

| Filter | Without EMA20>50 | With EMA20>50 | Calmar Improvement |
|--------|------------------|---------------|-------------------|
| RSI>60 + Vol>3x + BO>3% | Calmar 0.87 | **Calmar 1.11** | +28% |
| Vol>5x + BO>5% | Calmar 0.88 | **Calmar 1.14** | +30% |
| RSI>70 + Vol>3x + BO>3% | Calmar 0.48 | **Calmar 0.77** | +60% |

The EMA20>50 crossover confirms the medium-term trend is intact, reducing false breakouts in choppy markets.

### 3.4 Breakout Magnitude (BO>3%) is the Minimum Useful Threshold

Breakouts of less than 3% above the consolidation range have significantly higher failure rates. BO>5% further improves results but halves the trade count.

### 3.5 New Discovery: Volume Trend > Volume Spike

Volume Trend (10d avg / 50d avg volume) is a stronger filter than a single-day volume spike:

| Filter | Win% | Trades | PF | Calmar |
|--------|------|--------|-----|--------|
| Vol>=5x (single day spike) | 50.5% | 1,807 | 2.97 | - |
| **VolTrend>=1.2x (rising 10d avg)** | - | - | - | - |
| BO>=5% + VolTrend>=1.2x | **60.1%** | 203 | **4.46** | **3.21** |
| Vol>=7x + VolTrend>=1.2x | **58.1%** | 258 | **3.96** | **2.59** |

Rising volume over 10 days leading into breakout (not just a one-day pop) indicates sustained accumulation by institutions.

### 3.6 Filters That Do NOT Work

| Filter | Win% | Calmar | Why It Fails |
|--------|------|--------|-------------|
| Above EMA200 alone | 46.9% | 0.65 | Too generic, nearly all trending stocks pass |
| Dual momentum (Mom20+60 > 0) | 48.5% | 0.45 | Weak signal, doesn't capture breakout quality |
| RSI>70 + BO>5% (no volume) | 50.0% | 0.18 | Without volume confirmation, large breakouts are unreliable |
| Vol>=15x+ | 47.9% | - | Extreme spikes are often news-driven gap-ups that fade |
| Vol>=20x+ | **36.5%** | - | **Worse than baseline** - these are panic/news events, not real breakouts |

---

## 4. Darvas Box vs Flat-Range Consolidation

### Head-to-Head with Best Filters

| Filter | Darvas Trades | Darvas Win% | Darvas PF | Darvas Calmar | Flat Trades | Flat Win% | Flat PF | Flat Calmar |
|--------|---------------|-------------|-----------|---------------|-------------|-----------|---------|-------------|
| RSI>70 + Vol>3x + BO>3% + EMA20>50 | 50 | **58.0%** | 3.81 | **0.67** | 20 | 55.0% | 3.93 | 0.13 |
| Vol>5x + BO>5% | 285 | **55.4%** | **3.48** | **1.01** | 251 | 52.2% | 3.17 | 0.38 |
| RSI>60 + Vol>3x + BO>3% + EMA20>50 | 500 | 52.2% | 3.45 | **1.08** | 275 | **54.9%** | **3.79** | 0.45 |
| RSI>60 + Vol>3x + BO>3% | 550 | 51.1% | 3.27 | **0.80** | 319 | **54.9%** | **3.72** | 0.49 |
| Vol>5x + BO>3% | 509 | **55.6%** | **3.62** | **1.51** | 482 | 51.7% | 3.03 | 0.32 |
| Near ATH + Vol>3x + BO>3% | 901 | **52.7%** | **3.51** | **0.93** | 816 | 51.6% | 3.10 | 0.51 |

### Summary

- **Darvas wins on Calmar ratio** - consistently 2-3x better risk-adjusted returns (lower drawdowns, smoother equity curve)
- **Flat wins on Profit Factor** with EMA/RSI filters - when Flat breakouts work, they produce slightly larger winners
- **Darvas is better for volume-based filters** (Vol>5x) - the new-high requirement aligns naturally with strong volume
- **Flat is better with momentum/EMA filters** - the tight-range pattern benefits more from trend confirmation

**Recommendation:** Use **Darvas as the primary detector** for its superior Calmar ratio. Use **Flat-Range as a secondary/complementary screen** when Darvas signals are scarce.

---

## 5. Winner vs Loser Profile

### 5.1 Filtered Universe (RSI>60 + Vol>3x + BO>3%)

**869 trades: 456 winners (52.5%) vs 413 losers (47.5%)**

#### Indicator Comparison

| Metric | Winners | Losers | Difference | Signal Strength |
|--------|---------|--------|------------|-----------------|
| RSI14 | 64.96 | 64.63 | +0.33 | WEAK |
| Volume Ratio | 6.61x | 6.51x | +0.10x | WEAK |
| Breakout % | 5.86% | 5.51% | +0.35% | WEAK |
| Box Height % | 12.06% | 11.74% | +0.32% | WEAK |
| **ATH Proximity** | **83.8%** | **78.3%** | **+5.56%** | **MODERATE** |
| Momentum 20d | 6.92% | 7.85% | -0.93% | WEAK |
| Momentum 60d | 15.05% | 15.09% | -0.03% | WEAK |
| Volume Trend | 1.09x | 1.06x | +0.04x | WEAK |
| Risk % (stop distance) | 15.53% | 15.05% | +0.48% | WEAK |

#### EMA Alignment

| Metric | Winners | Losers | Difference |
|--------|---------|--------|------------|
| Above EMA20 | 100.0% | 100.0% | 0% |
| Above EMA50 | 99.8% | 99.8% | 0% |
| **Above EMA200** | **91.3%** | **87.4%** | **+3.9%** |
| EMA20 > EMA50 | 90.4% | 87.9% | +2.5% |
| EMA50 > EMA200 | 100.0% | 100.0% | 0% |

#### Return Distribution

| Metric | Winners (456) | Losers (413) |
|--------|---------------|--------------|
| Mean return | **+40.7%** | -13.2% |
| Median return | +28.0% | -13.4% |
| Std deviation | 43.6% | 4.9% |
| Best trade | +299.9% | -0.1% |
| Worst trade | +0.2% | -25.0% |
| **Avg max gain before exit** | **+55.3%** | **+10.7%** |
| **Avg days to peak** | **95 days** | **18 days** |

### 5.2 Key Takeaways

1. **Winners are 3x the size of losers** (+40.7% avg gain vs -13.2% avg loss) - the system has positive expectancy despite ~50/50 win rate because winners are *much* bigger than losers.

2. **ATH proximity is the strongest discriminator** - winners are 5.6% closer to their all-time high at entry. Stocks near ATH with breakout volume tend to push through to new highs.

3. **Losers fail fast** - avg peak gain of only +10.7% before reversing, peaking at just 18 days. Winners run for 95 days on average. *If a breakout hasn't moved meaningfully in 3-4 weeks, it's likely to fail.*

4. **Above EMA200 matters** - 91.3% of winners were above their 200-day EMA vs 87.4% of losers. Long-term trend alignment provides a small but consistent edge.

5. **83% of losers hit the stop loss** (avg -14.7% return). Only 17% are still open but underwater (avg -5.6%). The stop loss is doing its job - cutting losers before they get worse.

### 5.3 Full Universe Winner vs Loser (No Filters)

| Metric | Winners (4,556) | Losers (5,183) | Difference | Discriminating? |
|--------|-----------------|----------------|------------|-----------------|
| RSI14 | 60.34 | 59.66 | +0.69 | WEAK |
| Volume Ratio | 3.88x | 3.74x | +0.14x | WEAK |
| Breakout % | 2.36% | 2.17% | +0.19% | WEAK |
| Box Height % | 11.43% | 10.92% | +0.51% | MODERATE |
| ATH Proximity | 82.5% | 81.0% | +1.4% | WEAK |
| Momentum 20d | 4.69% | 4.56% | +0.13% | WEAK |
| Momentum 60d | 10.53% | 9.92% | +0.61% | WEAK |
| Risk % | 12.20% | 11.63% | +0.57% | MODERATE |

Without filters, **no single indicator strongly discriminates** winners from losers. This confirms that the *combination* of filters is what creates the edge, not any individual metric.

---

## 6. High-Conviction Strategies (65%+ Win Rate)

An aggressive search across 7,400+ filter combinations found multiple strategies exceeding 65% win rate. The key unlocks were:

- **RSI >= 75** (not just 70) - extreme momentum at breakout is highly predictive
- **Volume Trend > 1.2x** (rising volume *into* breakout, not just a one-day spike)
- **60-day Momentum >= 10%** - stock already in a strong medium-term trend
- **Box Height 5-12%** - tight consolidation before breakout

### 6.1 Best Strategies by Trade Count at 65%+

| Strategy | Trades | Win% | Avg Ret | PF | Calmar | Notes |
|----------|--------|------|---------|-----|--------|-------|
| **BO>=5% + Mom60>=10% + VolTrend>1.2x** | **90** | **68.9%** | +26.3% | **6.08** | **2.00** | **Best balance of signals + win rate** |
| Vol>=3x + BO>=5% + AbvEMA200 + VolTrend>1.2x | 126 | 64.3% | +16.1% | 4.44 | 1.50 | Most signals at ~64% |
| BO>=5% + AbvEMA200 + VolTrend>1.2x | 142 | 63.4% | +15.8% | 4.45 | 1.73 | Simpler 3-filter version |
| BO>=5% + EMA20>50 + VolTrend>1.2x | 169 | 62.1% | +18.3% | 4.86 | 2.78 | Best Calmar at 60%+ with volume |

### 6.2 Highest Win Rate Strategies

| Strategy | Trades | Win% | Avg Ret | PF | Calmar | Notes |
|----------|--------|------|---------|-----|--------|-------|
| RSI>=65 + Vol>=7x + BO>=3% + Mom60>=10% + Box5-12% | 18 | **83.3%** | +64.4% | **22.11** | **9.17** | Highest PF + Calmar found |
| RSI>=65 + Vol>=7x + BO>=5% + Mom60>=10% | 22 | **81.8%** | +53.0% | **20.19** | 1.23 | |
| Vol>=7x + BO>=7% + EMA20>50 + Mom20>=10% | 21 | **81.0%** | +31.0% | 11.28 | **2.80** | |
| Vol>=7x + BO>=5% + Mom60>=20% + Darvas | 29 | **79.3%** | +32.2% | **13.10** | 1.32 | Darvas-only variant |
| **RSI>=75 + Vol>=3x + EMA20>50** | **38** | **73.7%** | +20.3% | **10.37** | **1.48** | **Best practical high-conviction** |
| RSI>=75 + Vol>=3x | 39 | 71.8% | +19.2% | 7.96 | 1.23 | Simpler 2-filter version |

> **Note on small sample sizes:** Strategies with <30 trades have very high win rates but are statistically less reliable. The 83.3% win rate (18 trades) could fluctuate significantly with new data. Strategies with 50+ trades are more robust.

### 6.3 What Makes These Work

The 65%+ strategies share a common pattern: **they demand multiple independent confirmations of genuine momentum**.

1. **Volume confirmation** (Vol>=3x or VolTrend>1.2x) ensures institutional participation, not just retail noise
2. **Price momentum** (RSI>=75 or Mom60>=10%) confirms the stock is already in a strong uptrend
3. **Breakout magnitude** (BO>=5%) filters out marginal breakouts that are easily reversed
4. **Tight consolidation** (Box 5-12%) means the breakout comes from a compressed base with built-up energy

Each filter individually lifts win rate by only 2-5%, but stacking 3-4 independent confirmations compounds the edge to 65-75%+.

---

## 7. Recommended Entry Filter Tiers

| Tier | Criteria | Trades | Win% | Avg Ret | PF | Calmar | Use Case |
|------|----------|--------|------|---------|-----|--------|----------|
| **ALPHA** | RSI>=75 + Vol>=3x + EMA20>50 | 38 | **73.7%** | +20.3% | **10.37** | **1.48** | Sniper - max conviction |
| **TIER 1** | BO>=5% + Mom60>=10% + VolTrend>1.2x | 90 | **68.9%** | +26.3% | **6.08** | **2.00** | **Primary screener (RECOMMENDED)** |
| **TIER 2** | RSI>=60 + Vol>=3x + BO>=3% + EMA20>50 | 775 | 53.2% | +16.0% | 3.56 | 1.11 | Standard screener |
| **TIER 3** | RSI>=60 + Vol>=3x | 1,773 | 50.9% | +13.2% | 3.27 | 0.87 | Watchlist generation |
| BASELINE | No filters | 9,739 | 46.8% | +9.7% | 2.72 | 0.24 | - |

### TIER 1 is the Recommended Default

**Breakout >= 5% + 60-day Momentum >= 10% + Volume Trend >= 1.2x**

- **90 trades** across 25 years (~3.6 per year) - selective but actionable
- **68.9% win rate** - more than 2 out of 3 trades are profitable
- **+26.3% average return** - strong absolute performance
- **Profit Factor of 6.08** - you earn Rs.6.08 for every Rs.1 lost
- **Calmar of 2.00** - excellent risk-adjusted return
- No RSI dependency - works purely on price action, volume trend, and momentum

### ALPHA Tier for Maximum Conviction

**RSI >= 75 + Volume >= 3x average + EMA20 > EMA50**

- Only ~1.5 signals per year - these are *rare* events
- **73.7% win rate** with PF of 10.37
- When this fires, it's an extremely strong momentum breakout
- Use for larger position sizing or adding to existing positions

### When to Use Each Tier

- **ALPHA:** The rarest, highest-conviction signal. Only fires ~1.5x/year. When it does, allocate larger capital. Think of it as an "all-in" indicator - 3 out of 4 times it's right.
- **TIER 1:** The primary trading system. ~3.6 signals/year, 69% win rate, best Calmar. Use this as your **default entry filter for new positions**.
- **TIER 2:** When TIER 1 signals are scarce and you need more opportunities. 53% win rate is still profitable due to the 3.56 PF (winners are 3x bigger than losers).
- **TIER 3:** For watchlist building only. 51% win rate is barely above coin flip, but the PF of 3.27 makes it net positive over time.

---

## 8. Implementation Notes

### Filter Parameters for Code Integration

```python
# ALPHA (Maximum Conviction - ~1.5 signals/year)
BREAKOUT_FILTERS_ALPHA = {
    'min_rsi14': 75,              # RSI(14) >= 75 at breakout
    'min_volume_ratio': 3.0,      # Breakout volume >= 3x 50-day average
    'require_ema20_gt_50': True,  # EMA(20) > EMA(50) at breakout
}

# TIER 1 (Recommended Default - ~3.6 signals/year)
BREAKOUT_FILTERS = {
    'min_breakout_pct': 5.0,      # Close >= 5% above consolidation high
    'min_mom_60d': 10.0,          # 60-day price momentum >= 10%
    'min_vol_trend': 1.2,         # 10d avg volume / 50d avg volume >= 1.2
}

# TIER 2 (Standard Screener - ~31 signals/year)
BREAKOUT_FILTERS_STANDARD = {
    'min_rsi14': 60,              # RSI(14) >= 60 at breakout
    'min_volume_ratio': 3.0,      # Breakout volume >= 3x 50-day average
    'min_breakout_pct': 3.0,      # Close >= 3% above consolidation high
    'require_ema20_gt_50': True,  # EMA(20) > EMA(50) at breakout
}

# TIER 3 (Broad Watchlist - ~71 signals/year)
BREAKOUT_FILTERS_BROAD = {
    'min_rsi14': 60,              # RSI(14) >= 60 at breakout
    'min_volume_ratio': 3.0,      # Breakout volume >= 3x 50-day average
}
```

### Files Referenced

- `services/consolidation_breakout.py` - DarvasBoxDetector + FlatRangeDetector
- `services/technical_indicators.py` - EMA, RSI, Stochastics, Ichimoku, Supertrend
- `services/mq_backtest_engine.py` - Backtest engine with breakout detection
- `breakout_analysis_full.csv` - Raw data (9,739 trades with indicator values)
- `analyze_filters_v3.py` - Calmar ratio analysis script
- `find_65pct_strategy.py` - Aggressive 65%+ win rate search script

---

## 9. Enhanced Indicator Analysis (Phase 2)

### 9.1 Expanded Dataset

The initial breakout CSV had 21 columns. We enriched it to **53 columns** by computing 32 new technical indicators from cached daily OHLCV data (3.2M rows across 457 stocks in `market_data.db`).

| Category | New Indicators | Purpose |
|----------|---------------|---------|
| **Oscillators** | Stochastics (K/D), Williams %R, RSI(7) | Short-term overbought/oversold confirmation |
| **Trend** | MACD histogram/signal, ADX/+DI/-DI | Trend strength and direction |
| **Volatility** | Bollinger %B/width/squeeze, ATR% | Volatility regime and squeeze detection |
| **Volume** | MFI, OBV (bullish/bearish) | Volume-price confirmation |
| **Trend Proxy** | Supertrend, PSAR (simplified), CCI | Additional trend signals |
| **EMA Stack** | Above EMA9/21/100, EMA9>21, EMA20 rising | Multi-timeframe trend alignment |
| **Weekly (Top-Down)** | Weekly EMA20/50, Weekly EMA20>50, Weekly RSI, Weekly MACD | Higher timeframe trend context |

**Coverage:** ~100% of trades had valid indicator values (only edge cases with <60 days of history were missing).

### 9.2 New Indicator Win Rate Analysis

Individual indicator pass rates and win rates on the full 9,739-trade dataset:

| Indicator Filter | Trades | Win% | Lift vs Baseline | Notes |
|-----------------|--------|------|-----------------|-------|
| Weekly EMA20>50 (wEMA20>50) | ~5,500 | 50.8% | +4.0% | **Best weekly filter** |
| Weekly MACD Positive (wMACD+) | ~5,800 | 49.6% | +2.8% | Good broad filter |
| MACD Bullish (line > signal) | ~5,200 | 48.4% | +1.6% | Marginal alone |
| RSI(7) > 70 | ~3,800 | 51.2% | +4.4% | **Strong short-term momentum** |
| RSI(7) > 80 | ~2,400 | 53.1% | +6.3% | **Very strong confirmation** |
| Williams %R > -20 | ~3,600 | 50.9% | +4.1% | Overbought = bullish for breakouts |
| BB %B > 1.0 (above upper band) | ~3,300 | 51.5% | +4.7% | **Price above Bollinger upper** |
| ADX > 25 (trending) | ~4,500 | 48.8% | +2.0% | Trend strength confirmation |
| MFI > 60 | ~4,200 | 49.1% | +2.3% | Volume-weighted momentum |
| CCI > 100 | ~3,200 | 50.4% | +3.6% | Strong momentum |
| OBV Bullish | ~5,100 | 48.2% | +1.4% | Minimal signal alone |

> **Key insight:** No single new indicator exceeds 53% win rate alone (vs 46.8% baseline), confirming that the **combination** of multiple filters is what creates the edge. However, RSI(7) and Williams %R are the strongest new individual discriminators.

---

## 10. Top-Down Weekly-to-Daily Approach

### 10.1 Concept

Instead of only using daily indicators at the moment of breakout, we first filter for stocks where the **weekly trend is bullish**, then apply daily breakout filters. This top-down approach ensures we're trading breakouts that align with the higher timeframe.

### 10.2 Weekly Filter Effectiveness

| Weekly Base | Pass Rate | Win% | vs No Weekly Filter |
|-------------|-----------|------|-------------------|
| wEMA20>50 (weekly EMA20 > EMA50) | 56% | +3-5% win | **Best discriminator** |
| wEMA50 (above weekly EMA50) | 62% | +2-3% win | Good broad filter |
| wMACD+ (weekly MACD > 0) | 60% | +2-3% win | Trend confirmation |
| wRSI>60 (weekly RSI > 60) | 45% | +3-4% win | More selective |
| wEMA20>50 + wMACD+ | 48% | +4-5% win | **Best weekly combo** |

### 10.3 Best Top-Down Strategies

| Weekly Base | Daily Overlay | Trades | Win% | PF | Calmar |
|-------------|--------------|--------|------|-----|--------|
| wEMA20>50 + wRSI>60 | BO>=5% + VolTr>=1.2 | 80 | **67.5%** | 6.60 | 1.55 |
| wEMA20>50 + wMACD+ | BO>=5% + VolTr>=1.2 | 132 | 64.4% | 5.17 | 0.85 |
| wEMA20>50 + wMACD+ | RSI>=70 + ATH>=90% | 253 | 58.9% | 3.72 | 1.12 |
| wEMA50 + wMACD+ | ATH>=90% + Vol>=3x + BO>=3% | 669 | 56.5% | 3.70 | **1.74** |
| wEMA20>50 + wRSI>60 | ATH>=90% + Vol>=3x + BO>=3% | 423 | 56.0% | 4.17 | 1.70 |

> **Key finding:** Weekly EMA20>50 is the single most valuable weekly filter. Adding it to daily breakout strategies consistently lifts win rate by 3-5% while preserving trade count. The weekly trend alignment acts as a "permission filter" - only trade breakouts when the weekly trend supports the direction.

---

## 11. The 250-Trade Frontier

### 11.1 The Hard Constraint

After testing **40 building blocks** across **6 phases** (2-filter through 6-filter combos + top-down weekly-daily layering), totaling 5,264 shortlisted strategies:

> **No strategy achieves 250+ trades AND 65%+ win rate simultaneously.**

This is an efficiency frontier - you can have high trade count OR high win rate, but not both at the extreme. The tradeoff:

| Target | Best Strategy | Trades | Win% | PF | Calmar |
|--------|--------------|--------|------|-----|--------|
| **Max trades at 65%+** | BO>=3% + Mom10>=5 + VolTr>=1.2 + ATH>=85% + wEMA50 + wEMA20>50 | **247** | **65.2%** | 5.98 | 1.78 |
| **Max win% at 250+** | BO>=3% + Mom10>=5 + VolTr>=1.2 + ATH>=85% + wEMA20>50 | **251** | **64.5%** | 5.91 | 1.78 |
| **Best PF at 200+/65%+** | BO>=3% + VolTr>=1.2 + ATH>=85% + EMA20>50 + wEMA20>50 + WillR>-20 | 220 | 65.9% | **6.32** | 1.70 |
| **Best Calmar at 200+/65%+** | Vol>=3x + VolTr>=1.2 + ATH>=90% + wEMA20>50 + MFI>60 + RSI7>80 | 208 | 65.9% | 5.47 | **1.94** |

### 11.2 Best Win% at Each Trade-Count Tier

| Min Trades | Strategy | N | Win% | PF | Calmar |
|------------|----------|---|------|-----|--------|
| **500+** | Vol>=3x + VolTr>=1.2 + ATH>=85% | 530 | 58.5% | 4.54 | 1.94 |
| **400+** | Vol>=3x + VolTr>=1.2 + ATH>=90% | 417 | 59.2% | 4.61 | 1.73 |
| **300+** | Vol>=5x + BO>=3% + ATH>=90% + WillR>-20 | 304 | 61.2% | 4.05 | 1.05 |
| **250+** | BO>=3% + Mom10>=5 + VolTr>=1.2 + ATH>=85% + wEMA20>50 | 251 | 64.5% | 5.91 | 1.78 |
| **200+** | BO>=3% + VolTr>=1.2 + ATH>=85% + EMA20>50 + wEMA20>50 + WillR>-20 | 220 | 65.9% | 6.32 | 1.70 |
| **150+** | BO>=3% + VolTr>=1.2 + ATH>=90% + wEMA20>50 + RSI7>80 | 151 | **67.5%** | 6.54 | **3.23** |
| **100+** | Vol>=3x + Mom60>=15 + Mom10>=5 + VolTr>=1.2 + ATH>=90% + BB>upper | 107 | **72.0%** | **8.01** | 1.02 |

### 11.3 Best Calmar at Each Trade-Count Tier

| Min Trades | Strategy | N | Win% | PF | Calmar |
|------------|----------|---|------|-----|--------|
| **500+** | BO>=5% + wMACD+ | 652 | 55.4% | 3.51 | **1.99** |
| **400+** | Vol>=5x + ATH>=90% + RSI7>80 | 450 | 58.2% | 3.88 | **2.40** |
| **250+** | Vol>=5x + VolTr>=1.2 + MACD+ + RSI7>80 | 263 | 60.5% | 4.37 | **3.60** |
| **200+** | Vol>=5x + VolTr>=1.2 + EMA20>50 + RSI7>80 | 210 | 64.3% | 5.05 | **4.66** |
| **150+** | Vol>=5x + Mom10>=5 + VolTr>=1.2 + EMA20>50 + RSI7>80 | 199 | 64.8% | 5.17 | **4.68** |

### 11.4 Key Discoveries

1. **VolTrend>=1.2 + ATH>=85%** is the backbone of every high-trade-count strategy. Rising 10-day volume trend combined with near-ATH positioning is the most reliable broad filter.

2. **Weekly EMA20>50 (wEMA20>50)** is the most impactful weekly top-down filter. It lifts win rate by 3-5% with minimal trade reduction.

3. **RSI(7) > 80** is the strongest short-term confirmation indicator. Stocks with extreme short-term RSI at breakout outperform significantly. This is consistent with the RSI(14) >= 75 finding from Phase 1 - strength begets strength.

4. **Williams %R > -20** (overbought zone) is highly complementary to other filters. It acts as a "momentum lock" confirming the breakout is occurring from a position of strength.

5. **Bollinger Band breakout (BB %B > 1.0)** combined with momentum filters produces the highest win rates at the 100+ trade tier (72.0%).

6. **BO>=3% (not 5%)** is needed for 250+ trade volume. Lowering the breakout magnitude threshold from 5% to 3% roughly doubles trade count while only sacrificing ~2-3% win rate.

7. **Best Calmar strategies use Vol>=5x + RSI7>80** - the combination of high volume spike plus short-term overbought RSI produces the smoothest equity curves (Calmar 4.66 at 210 trades).

---

## 12. Revised Entry Filter Tiers (V2)

Based on the expanded analysis with 53 indicators and top-down weekly approach:

| Tier | Criteria | Trades | Win% | PF | Calmar | Signals/Year |
|------|----------|--------|------|-----|--------|-------------|
| **ALPHA** | RSI>=75 + Vol>=3x + EMA20>50 | 38 | **73.7%** | **10.37** | 1.48 | ~1.5 |
| **TIER 1A** | BO>=3% + VolTr>=1.2 + ATH>=90% + wEMA20>50 + RSI7>80 | 151 | **67.5%** | **6.54** | **3.23** | ~6 |
| **TIER 1B** | BO>=3% + VolTr>=1.2 + ATH>=85% + EMA20>50 + wEMA20>50 + WillR>-20 | 220 | **65.9%** | **6.32** | 1.70 | ~9 |
| **TIER 2** | BO>=3% + Mom10>=5 + VolTr>=1.2 + ATH>=85% + wEMA20>50 | 251 | 64.5% | 5.91 | 1.78 | ~10 |
| **TIER 3** | Vol>=3x + VolTr>=1.2 + ATH>=85% | 530 | 58.5% | 4.54 | 1.94 | ~21 |
| **CALMAR OPT** | Vol>=5x + VolTr>=1.2 + EMA20>50 + RSI7>80 | 210 | 64.3% | 5.05 | **4.66** | ~8 |
| BASELINE | No filters | 9,739 | 46.8% | 2.72 | 0.24 | ~390 |

### Tier Descriptions

**ALPHA (Sniper):** Unchanged from V1. Rare, extremely high-conviction momentum breakouts. RSI>=75 at breakout means the stock is already surging. ~1.5 signals/year - use for max position sizing.

**TIER 1A (Top-Down Momentum):** New in V2. Requires weekly trend alignment (wEMA20>50) plus near-ATH positioning (>=90%) plus short-term RSI overbought (RSI7>80). The triple confirmation of weekly trend + daily ATH proximity + short-term momentum produces **67.5% win rate with a Calmar of 3.23** - the best risk-adjusted strategy with 150+ trades.

**TIER 1B (Volume + Breadth):** New in V2. The highest win rate achievable at 200+ trades. Uses weekly top-down filter plus Williams %R overbought confirmation. **65.9% win rate with 220 trades** - the practical "sweet spot" for active trading with high win rate.

**TIER 2 (Active Trader):** The closest to the 250-trade/65% target. 251 trades at 64.5% win rate. Uses weekly EMA20>50 filter with daily momentum/volume trend. ~10 signals/year provides consistent deal flow.

**TIER 3 (Broad Screener):** Simple 3-filter strategy with 530 trades (21/year). Lower win rate but excellent Calmar of 1.94 means smooth equity curve. Good for generating watchlists or lower-conviction entries.

**CALMAR OPT (Smoothest Equity Curve):** Specifically optimized for risk-adjusted returns. Calmar of **4.66** is exceptional - meaning CAGR is 4.66x the maximum drawdown. Uses Vol>=5x (strong institutional buying) + RSI7>80 (short-term momentum confirmation).

### When to Use Each Tier

- **ALPHA:** The rarest signal. When it fires, go heavy. 3 out of 4 trades are winners.
- **TIER 1A:** Primary system for traders who want fewer, higher-quality trades (~6/year). Best Calmar with meaningful trade count.
- **TIER 1B:** Primary system for traders who want more activity (~9/year) at 65%+ win rate. **Recommended default for most traders.**
- **TIER 2:** For traders who need 250+ trades over the backtest period. Slightly below 65% win rate but excellent PF.
- **TIER 3:** Watchlist generation. Every signal gets evaluated but position sizing is smaller.
- **CALMAR OPT:** For risk-averse accounts. The smoothest ride with the least drawdown pain.

---

## 13. Implementation Notes (V2)

### Updated Filter Parameters for Code Integration

```python
# ALPHA (Maximum Conviction - ~1.5 signals/year)
BREAKOUT_FILTER_ALPHA = {
    'min_rsi14': 75,              # RSI(14) >= 75 at breakout
    'min_volume_ratio': 3.0,      # Breakout volume >= 3x 50-day average
    'require_ema20_gt_50': True,  # EMA(20) > EMA(50) at breakout
    # 73.7% win rate | PF 10.37 | Calmar 1.48 | ~1.5 signals/year
}

# TIER 1A (Top-Down Momentum - ~6 signals/year)
BREAKOUT_FILTER_TIER1A = {
    'min_breakout_pct': 3.0,      # Close >= 3% above consolidation high
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 90,      # Within 10% of all-time high
    'require_weekly_ema20_gt_50': True,  # Weekly EMA(20) > EMA(50)
    'min_rsi7': 80,               # RSI(7) >= 80 (short-term overbought)
    # 67.5% win rate | PF 6.54 | Calmar 3.23 | ~6 signals/year
}

# TIER 1B (Active High-Win - ~9 signals/year)
BREAKOUT_FILTER_TIER1B = {
    'min_breakout_pct': 3.0,      # Close >= 3% above consolidation high
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 85,      # Within 15% of all-time high
    'require_ema20_gt_50': True,  # EMA(20) > EMA(50) at breakout
    'require_weekly_ema20_gt_50': True,  # Weekly EMA(20) > EMA(50)
    'min_williams_r': -20,        # Williams %R >= -20 (overbought zone)
    # 65.9% win rate | PF 6.32 | Calmar 1.70 | ~9 signals/year
}

# TIER 2 (Active Trader - ~10 signals/year)
BREAKOUT_FILTER_TIER2 = {
    'min_breakout_pct': 3.0,      # Close >= 3% above consolidation high
    'min_mom_10d': 5.0,           # 10-day momentum >= 5%
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 85,      # Within 15% of all-time high
    'require_weekly_ema20_gt_50': True,  # Weekly EMA(20) > EMA(50)
    # 64.5% win rate | PF 5.91 | Calmar 1.78 | ~10 signals/year
}

# TIER 3 (Broad Screener - ~21 signals/year)
BREAKOUT_FILTER_TIER3 = {
    'min_volume_ratio': 3.0,      # Breakout volume >= 3x 50-day average
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 85,      # Within 15% of all-time high
    # 58.5% win rate | PF 4.54 | Calmar 1.94 | ~21 signals/year
}

# CALMAR OPTIMIZED (Smoothest Equity Curve - ~8 signals/year)
BREAKOUT_FILTER_CALMAR = {
    'min_volume_ratio': 5.0,      # Breakout volume >= 5x 50-day average
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'require_ema20_gt_50': True,  # EMA(20) > EMA(50) at breakout
    'min_rsi7': 80,               # RSI(7) >= 80 (short-term overbought)
    # 64.3% win rate | PF 5.05 | Calmar 4.66 | ~8 signals/year
}
```

### New Indicators Required for V2 Filters

The V2 tier system requires the following indicators that were **not** in the V1 system:

| Indicator | Computation | Used By |
|-----------|------------|---------|
| **RSI(7)** | 7-period RSI (shorter than standard 14) | TIER 1A, CALMAR OPT |
| **Williams %R (14)** | -100 * (HH14 - Close) / (HH14 - LL14) | TIER 1B |
| **Weekly EMA20/50** | EMA(20) and EMA(50) on weekly OHLCV | TIER 1A, 1B, 2 |
| **10-day Momentum** | (Close / Close[10] - 1) * 100 | TIER 2 |
| **ATH Proximity** | Close / All-Time-High * 100 | Already in V1 |

### Files Referenced

- `services/consolidation_breakout.py` - DarvasBoxDetector + FlatRangeDetector + Filter constants
- `services/technical_indicators.py` - EMA, RSI, Stochastics, MACD, ADX, Bollinger, MFI, CCI, Williams %R, OBV, Supertrend
- `services/mq_backtest_engine.py` - Backtest engine with breakout detection
- `backtest_data/market_data.db` - SQLite database with 3.2M daily OHLCV rows
- `breakout_analysis_full.csv` - Raw data (9,739 trades, 21 columns)
- `breakout_analysis_enhanced.csv` - Enhanced data (9,739 trades, 53 columns)
- `enhance_breakout_data.py` - Script to generate enhanced CSV from price data
- `find_advanced_strategies.py` - First-round advanced strategy search (risk-reward, composites, detector-specific)
- `find_250trade_strategies.py` - Second-round exhaustive search targeting 250+ trades

---

## 14. Deep Dive: OR Combinations Break the 250-Trade Barrier (Phase 3)

### 14.1 The Key Insight: OR Logic Instead of AND

Previous searches used only AND logic (all filters must pass simultaneously), which inherently limits trade count - the more filters you add, the fewer trades survive. The breakthrough: **use OR logic to combine complementary strategies**, taking a trade whenever ANY of the component strategies fires.

This works because different strategies capture **different types of strong breakouts**:
- ALPHA catches extreme RSI momentum breakouts (~1.5/year)
- T1B catches weekly-aligned near-ATH breakouts (~9/year)
- MomVol catches volume-confirmed 60-day momentum breakouts (~5/year)
- Calmar catches high-volume RSI7 overbought breakouts (~8/year)
- BBupper_Mom catches Bollinger band breakouts with momentum (~10/year)

Since these strategies have **low overlap** (different stocks trigger different strategies), the union captures far more trades while each individual trade still meets a high-conviction filter.

### 14.2 OR Combination Results

| OR Combination | Trades | Win% | PF | Calmar | Trades/Year |
|---------------|--------|------|-----|--------|------------|
| **ALPHA OR T1B_WillR OR MomVol** | **332** | **66.9%** | **6.44** | 1.96 | ~13 |
| **ALPHA OR T1A_ATH90 OR Calmar** | **325** | **65.2%** | **5.46** | **5.19** | ~13 |
| ALPHA OR T1B_WillR OR Calmar | **393** | **65.4%** | **5.73** | 2.52 | ~16 |
| T1B_WillR OR MomVol OR Calmar | **439** | 64.0% | 5.51 | 2.79 | ~18 |
| T1B_WillR OR Calmar OR BBupper_Mom | **535** | 62.4% | 5.12 | **3.01** | ~21 |
| ALPHA OR T1B_WillR | 251 | **67.3%** | **6.64** | 1.97 | ~10 |
| ALPHA OR MomVol | 169 | **69.8%** | **8.01** | 1.71 | ~7 |

> **The 250-trade/65% target is achieved:** ALPHA OR T1B_WillR OR MomVol gives **332 trades at 66.9% win rate** with PF of 6.44. This is ~13 signals per year over 25 years.

### 14.3 Component Strategy Definitions

| Strategy | Filters | Trades | Win% |
|----------|---------|--------|------|
| **ALPHA** | RSI>=75 + Vol>=3x + EMA20>50 | 38 | 73.7% |
| **T1A_ATH90** | BO>=3% + VolTr>=1.2 + ATH>=90% + wEMA20>50 + RSI7>80 | 151 | 67.5% |
| **T1B_WillR** | BO>=3% + VolTr>=1.2 + ATH>=85% + EMA20>50 + wEMA20>50 + WillR>-20 | 220 | 65.9% |
| **MomVol** | Mom60>=15 + VolTr>=1.2 + ATH>=90% + Vol>=3x | 134 | 67.9% |
| **Calmar** | Vol>=5x + VolTr>=1.2 + EMA20>50 + RSI7>80 | 210 | 64.3% |
| **BBupper_Mom** | BB>upper + Mom60>=15 + VolTr>=1.2 + ATH>=90% | 258 | 61.6% |
| **Trend9_BO** | All 9 trend signals bullish + BO>=3% + VolTr>=1.2 | 312 | 57.4% |

### 14.4 Best Calmar OR Combination

**ALPHA OR T1A_ATH90 OR Calmar: 325 trades, 65.2% win, Calmar 5.19**

This is exceptional risk-adjusted performance. The Calmar of 5.19 means the CAGR is more than 5x the maximum drawdown. This combination works because:
- ALPHA and T1A_ATH90 capture momentum breakouts (high win rate)
- Calmar captures volume-confirmed breakouts with extreme short-term RSI (smooth equity curve)
- The three strategies have low overlap, providing diversification

---

## 15. Consolidation Duration Analysis

### 15.1 Duration Distribution

Consolidation duration was computed by measuring how many consecutive days price stayed within the box range before breakout:

| Duration Band | Trades | Win% | PF | Calmar | Notes |
|--------------|--------|------|-----|--------|-------|
| 5-15 days (short) | ~2,800 | 46.2% | 2.58 | - | Below baseline |
| 15-30 days (medium) | ~2,100 | 48.6% | 2.88 | - | At baseline |
| 30-60 days (long) | ~2,400 | 50.2% | 3.12 | - | **Slightly better** |
| 60-120 days (very long) | ~1,200 | 52.1% | 3.45 | - | **Noticeably better** |
| >= 40 days | ~4,100 | 51.5% | 3.31 | - | Good general filter |

### 15.2 Consolidation Days + Best Filters

| Strategy | Trades | Win% | PF | Calmar |
|----------|--------|------|-----|--------|
| RSI 70-80 + CCI>100 + CD>=40 | 167 | **68.9%** | 3.89 | 0.76 |
| MFI 80+ + ATH 95-100% + CD>=40 + Mom10>=5 | 288 | **64.9%** | 4.63 | 1.48 |
| ATR <2% + CD>=40 + Mom10>=5 + Weekly>=4 | 277 | 64.3% | 5.38 | 1.22 |
| ATR <2% + WR>-20 + Mom10>=3 + Mom20>=5 | 276 | 65.2% | 4.28 | 1.49 |

### 15.3 Key Finding

Longer consolidation (>= 40 days) adds 2-3% to win rate. The "coiled spring" effect is real: stocks that consolidate for longer accumulate more buying pressure, and when they break out, the move is more reliable. However, consolidation duration alone is a weak filter - it only becomes powerful when combined with momentum and volatility filters.

---

## 16. New Standalone AND Strategies Discovered

### 16.1 Low-Volatility Breakouts (ATR-based)

A surprising finding: **low ATR (< 2%) stocks breaking out with momentum have very high win rates**, even with relaxed other filters:

| Strategy | Trades | Win% | PF | Calmar |
|----------|--------|------|-----|--------|
| ATR <2% + WR>-20 + Mom10>=3 + Mom20>=5 | 276 | **65.2%** | 4.28 | 1.49 |
| ATR <2% + CD>=40 + Mom10>=5 + Weekly>=4 | 277 | 64.3% | 5.38 | 1.22 |
| ATR <2% + ATH 95-100% + WR>-20 + Mom10>=3 | 326 | 62.0% | 4.67 | 1.31 |
| ATR <2% + Mom10>=5 + Weekly>=5 | 325 | 61.8% | 4.73 | 1.69 |

Low-volatility stocks (ATR < 2%) represent stable, large-cap names. When these stocks break out with momentum confirmation, it signals genuine institutional accumulation rather than noise.

### 16.2 Multi-Oscillator Confirmation

| Strategy | Trades | Win% | PF | Calmar |
|----------|--------|------|-----|--------|
| StK 90+ + MFI 80+ + ATH 95-100% + Mom10>=5 | 309 | 64.1% | 4.60 | 1.61 |
| MFI 80+ + ATH 95-100% + CD>=40 + Mom10>=5 | 288 | 64.9% | 4.63 | 1.48 |

When multiple oscillators (Stochastics, MFI) are in overbought territory AND the stock is near its all-time high, it confirms genuine broad-based buying pressure across price and volume dimensions.

---

## 17. Revised Entry Filter Tiers (V3 - Final)

### 17.1 Recommended System: Multi-Strategy OR

The recommended approach is no longer a single filter set, but a **multi-strategy system** where a trade is taken when ANY qualifying strategy fires:

| System | Components | Trades | Win% | PF | Calmar | /Year |
|--------|-----------|--------|------|-----|--------|-------|
| **SNIPER** | ALPHA OR MomVol | 169 | **69.8%** | **8.01** | 1.71 | ~7 |
| **PRIMARY** | ALPHA OR T1B_WillR OR MomVol | **332** | **66.9%** | **6.44** | 1.96 | **~13** |
| **BALANCED** | ALPHA OR T1A_ATH90 OR Calmar | 325 | 65.2% | 5.46 | **5.19** | ~13 |
| **ACTIVE** | ALPHA OR T1B_WillR OR Calmar | **393** | **65.4%** | **5.73** | 2.52 | **~16** |
| **HIGH-VOLUME** | T1B_WillR OR Calmar OR BBupper_Mom | **535** | 62.4% | 5.12 | 3.01 | **~21** |

### 17.2 Recommended Default: PRIMARY

**Take a trade when ANY of these three fire:**

1. **ALPHA**: RSI(14) >= 75 + Volume >= 3x + EMA20 > EMA50
2. **T1B_WillR**: BO >= 3% + VolTrend >= 1.2 + ATH >= 85% + EMA20>50 + Weekly EMA20>50 + Williams %R > -20
3. **MomVol**: Mom60 >= 15% + VolTrend >= 1.2 + ATH >= 90% + Volume >= 3x

**Result: 332 trades (13/year), 66.9% win rate, PF 6.44, Calmar 1.96**

This means 2 out of 3 trades win, you earn Rs.6.44 for every Rs.1 lost, and the annual return is nearly 2x the worst drawdown.

### 17.3 For Risk-Averse Accounts: BALANCED

**ALPHA OR T1A_ATH90 OR Calmar: 325 trades, 65.2% win, Calmar 5.19**

The Calmar of 5.19 is exceptional. The equity curve is extremely smooth with shallow drawdowns. Best for accounts where capital preservation is priority.

---

## 18. Implementation Notes (V3)

### Multi-Strategy OR Implementation

```python
# V3 Multi-Strategy System
# A trade is taken when ANY component strategy fires

STRATEGY_ALPHA = {
    'min_rsi14': 75,
    'min_volume_ratio': 3.0,
    'require_ema20_gt_50': True,
}

STRATEGY_T1B = {
    'min_breakout_pct': 3.0,
    'min_vol_trend': 1.2,
    'min_ath_proximity': 85,
    'require_ema20_gt_50': True,
    'require_weekly_ema20_gt_50': True,
    'min_williams_r': -20,
}

STRATEGY_MOMVOL = {
    'min_mom_60d': 15.0,
    'min_vol_trend': 1.2,
    'min_ath_proximity': 90,
    'min_volume_ratio': 3.0,
}

STRATEGY_CALMAR = {
    'min_volume_ratio': 5.0,
    'min_vol_trend': 1.2,
    'require_ema20_gt_50': True,
    'min_rsi7': 80,
}

STRATEGY_BB_MOM = {
    'min_bb_pct_b': 1.0,       # Above Bollinger upper band
    'min_mom_60d': 15.0,
    'min_vol_trend': 1.2,
    'min_ath_proximity': 90,
}

# System presets (OR combinations)
SYSTEM_SNIPER = [STRATEGY_ALPHA, STRATEGY_MOMVOL]
# 169 trades, 69.8% win, PF 8.01, Calmar 1.71

SYSTEM_PRIMARY = [STRATEGY_ALPHA, STRATEGY_T1B, STRATEGY_MOMVOL]
# 332 trades, 66.9% win, PF 6.44, Calmar 1.96

SYSTEM_BALANCED = [STRATEGY_ALPHA, STRATEGY_T1A, STRATEGY_CALMAR]
# 325 trades, 65.2% win, PF 5.46, Calmar 5.19

SYSTEM_ACTIVE = [STRATEGY_ALPHA, STRATEGY_T1B, STRATEGY_CALMAR]
# 393 trades, 65.4% win, PF 5.73, Calmar 2.52

SYSTEM_HIGH_VOLUME = [STRATEGY_T1B, STRATEGY_CALMAR, STRATEGY_BB_MOM]
# 535 trades, 62.4% win, PF 5.12, Calmar 3.01
```

### Files Referenced

- `deep_dive_strategies.py` - Phase 3 deep-dive search with OR combinations, consolidation days, indicator bands, K-of-N
- `services/consolidation_breakout.py` - Filter constants and detectors
- `breakout_analysis_enhanced.csv` - 53-column enhanced dataset
- `backtest_data/market_data.db` - 3.2M daily OHLCV rows for consolidation days computation

---

*Generated: 2026-02-12*
*Updated: 2026-02-12 - Phase 1: Initial filter optimization, 65%+ win rate strategies*
*Updated: 2026-02-12 - Phase 2: 32 new indicators, top-down weekly approach, 250-trade frontier analysis, revised V2 tier system*
*Updated: 2026-02-12 - Phase 3: OR combinations break 250-trade barrier, consolidation days, low-ATR strategies, V3 multi-strategy system*
