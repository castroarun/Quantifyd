# Breakout V3 System - Backtest Results

## Backtest Parameters

| Parameter | Value |
|-----------|-------|
| **Data Period** | May 2003 - Oct 2025 (22.9 years) |
| **Stock Universe** | 476 NSE stocks (F&O segment) |
| **Total Trades Analysed** | 9,739 breakout trades |
| **Price Data** | Daily OHLCV from `market_data_unified` (4.3M rows) |
| **Starting Capital** | Rs. 10,00,000 (10 Lakhs) |
| **Idle Cash** | 6.5% liquid fund return |
| **Futures Carry Cost** | 7% annualised |
| **Option Pricing** | Black-Scholes, 30% IV baseline |

---

## Chosen Setup: PRIMARY System + 5x Leverage + Married Put 10% OTM

| Metric | Value |
|--------|-------|
| System | PRIMARY (ALPHA OR T1B OR MOMVOL) |
| Total qualifying trades | 332 |
| Trades taken (capital available) | 251 |
| Trades skipped (no cash) | 81 |
| Margin calls | 0 |
| Leverage | 5x futures |
| Position size | 5% of portfolio per trade |
| Hedge | Married Put 10% OTM, rolled monthly |
| **CAGR** | **39.4%** |
| **Max Drawdown** | **17.7%** |
| **Calmar Ratio** | **2.23** |
| Win Rate | 64.2% |
| Avg Win (hedged) | +188.1% |
| Avg Loss (hedged) | -48.0% |
| Wipeout trades | **0** |
| Final Equity | Rs. 20,151 Lakhs |
| Liquid Fund Interest | Rs. 1,116 Lakhs |

---

## V3 Strategy Rules

### Component Strategies

**ALPHA** - High-conviction momentum (73.7% win, PF 10.37, 38 trades)
- RSI(14) >= 75
- Volume >= 3x 50-day average
- EMA(20) > EMA(50)

**T1B** - Breakout with weekly trend confirmation (65.9% win, PF 6.32, 220 trades)
- Close >= 3% above consolidation high
- 10d avg vol / 50d avg vol >= 1.2
- Within 15% of all-time high
- EMA(20) > EMA(50)
- Weekly EMA(20) > Weekly EMA(50)
- Williams %R >= -20

**MOMVOL** - Momentum + volume surge (67.9% win, PF 7.12, 134 trades)
- 60-day momentum >= 15%
- 10d avg vol / 50d avg vol >= 1.2
- Within 10% of all-time high
- Volume >= 3x 50-day average

**T1A** - Breakout with short-term RSI (67.5% win, PF 6.54, 151 trades)
- Close >= 3% above consolidation high
- 10d avg vol / 50d avg vol >= 1.2
- Within 10% of all-time high
- Weekly EMA(20) > Weekly EMA(50)
- RSI(7) >= 80

**CALMAR** - Volume extreme + trend (64.3% win, PF 5.05, 210 trades)
- Volume >= 5x 50-day average
- 10d avg vol / 50d avg vol >= 1.2
- EMA(20) > EMA(50)
- RSI(7) >= 80

**BB_MOM** - Bollinger Band breakout + momentum (61.6% win, PF 4.80, 258 trades)
- Price above Bollinger upper band (Pct B >= 1.0)
- 60-day momentum >= 15%
- 10d avg vol / 50d avg vol >= 1.2
- Within 10% of all-time high

### System Presets (OR logic across strategies)

| System | Strategies | Trades | Win% | PF | Calmar | Signals/yr |
|--------|-----------|--------|------|-----|--------|-----------|
| **SNIPER** | ALPHA + MOMVOL | 169 | 69.8% | 8.01 | 1.71 | ~7 |
| **PRIMARY** | ALPHA + T1B + MOMVOL | 332 | 66.9% | 6.44 | 1.96 | ~13 |
| **BALANCED** | ALPHA + T1A + CALMAR | 325 | 65.2% | 5.46 | 5.19 | ~13 |
| **ACTIVE** | ALPHA + T1B + CALMAR | 393 | 65.4% | 5.73 | 2.52 | ~16 |
| **HIGH_VOLUME** | T1B + CALMAR + BB_MOM | 535 | 62.4% | 5.12 | 3.01 | ~21 |

### Exit Rules

1. **STOP** - Price hits stop loss -> exit immediately
2. **OPEN** - Held 125 trading days (~6 months) -> exit at market close

### Stop Loss Modes

| Mode | Description | Win% | Avg Return | PF | Severe Losses |
|------|-------------|------|------------|-----|---------------|
| **FIXED** | SL = consolidation low (original) | 67.5% | +21.8% | 6.38 | 2 |
| **TRAIL 20%** | SL = highest high - 20% | 61.7% | +15.2% | 4.96 | **0** |
| **TRAIL 25%** | SL = highest high - 25% | 69.0% | +20.5% | 6.09 | 25 |
| **TRAIL 30%** | SL = highest high - 30% | 73.5% | +22.8% | 6.81 | 29 |
| **RATCHET 25%** | max(consol_low, peak-25%) | 66.0% | +20.0% | 6.35 | 1 |

**Key findings:**
- TRAIL-20% eliminates ALL severe losses (>-20%) while maintaining PF 4.96
- TRAIL-30% beats FIXED on every metric (73.5% WR, PF 6.81, +7558.7% total)
- RATCHET offers no advantage over pure trailing
- 21 trades had >30% unrealized gain but ended as FIXED-SL losses; TRAIL-20% rescued 11/21

---

## 15 Random Trades for Manual Verification

Seed=42, randomly selected from 332 PRIMARY trades.

| # | Symbol | Entry | Exit | Days | Strategy | Reason | Entry Rs | Exit Rs | Peak Rs | Stop Rs | Base% | Hedged% |
|---|--------|-------|------|------|----------|--------|----------|---------|---------|---------|-------|---------|
| 1 | TVSMOTOR | 2025-08-18 | 2026-02-12 | 178 | T1B | OPEN | 3,219.7 | 3,718.8 | 3,734.8 | 2,730.3 | +15.5% | +61.8% |
| 2 | BAJFINANCE | 2014-05-22 | 2014-11-16 | 178 | T1B | OPEN | 20.5 | 30.5 | 34.0 | 17.0 | +48.8% | +228.3% |
| 3 | INDUSINDBK | 2003-07-31 | 2003-09-23 | 54 | MOMVOL | STOP | 25.3 | 22.5 | 29.9 | 22.5 | -10.9% | -54.6% |
| 4 | TECHM | 2021-10-26 | 2022-02-13 | 110 | MOMVOL | STOP | 1,507.8 | 1,313.3 | 1,777.7 | 1,313.3 | -12.9% | -59.6% |
| 5 | CAMS | 2021-06-07 | 2021-12-02 | 178 | T1B | OPEN | 2,751.1 | 2,907.9 | 4,066.1 | 2,489.7 | +5.7% | +12.8% |
| 6 | INTELLECT | 2021-03-15 | 2021-09-09 | 178 | T1B | OPEN | 522.6 | 662.1 | 892.1 | 424.9 | +26.7% | +117.8% |
| 7 | M&M | 2017-12-15 | 2018-06-11 | 178 | T1B | OPEN | 741.7 | 891.5 | 933.8 | 660.1 | +20.2% | +85.3% |
| 8 | BRITANNIA | 2013-08-13 | 2013-08-28 | 15 | MOMVOL | STOP | 373.6 | 335.2 | 380.4 | 335.2 | -10.3% | -52.2% |
| 9 | INDIGO | 2024-03-26 | 2024-09-20 | 178 | T1B | OPEN | 3,492.1 | 4,867.9 | 5,035.5 | 2,985.7 | +39.4% | +181.3% |
| 10 | ASIANPAINT | 2011-05-24 | 2011-06-17 | 24 | MOMVOL | STOP | 296.4 | 263.8 | 338.7 | 263.8 | -11.0% | -52.4% |
| 11 | SHK | 2024-08-19 | 2025-02-13 | 178 | T1B | OPEN | 225.8 | 188.5 | 336.2 | 180.2 | -16.5% | -65.7% |
| 12 | BALMLAWRIE | 2023-09-01 | 2023-10-03 | 32 | T1B | STOP | 161.6 | 138.0 | 172.3 | 138.0 | -14.6% | -52.7% |
| 13 | ICICIBANK | 2003-08-21 | 2004-02-15 | 178 | T1B | OPEN | 32.4 | 56.5 | 64.0 | 26.6 | +74.2% | +355.3% |
| 14 | CIPLA | 2003-08-05 | 2004-01-30 | 178 | MOMVOL | OPEN | 68.9 | 91.8 | 112.8 | 60.5 | +33.2% | +150.3% |
| 15 | BAJFINANCE | 2012-02-01 | 2012-07-28 | 178 | ALPHA | OPEN | 7.5 | 10.0 | 10.5 | 6.0 | +33.3% | +150.8% |

### Cost Breakdown Per Trade

| # | Symbol | Base% | x5 Lev | Carry | Put Cost | Net Hedged | RSI14 | Vol | ATH% | BO% |
|---|--------|-------|--------|-------|----------|-----------|-------|-----|------|-----|
| 1 | TVSMOTOR | +15.5% | +77.5% | -3.4% | -2.3% | +61.8% | 65 | 4x | 99% | 6% |
| 2 | BAJFINANCE | +48.8% | +244.0% | -3.4% | -2.3% | +228.3% | 55 | 3x | 97% | 8% |
| 3 | INDUSINDBK | -10.9% | -54.5% | -1.0% | -0.7% | -54.6% | 69 | 4x | 98% | 2% |
| 4 | TECHM | -12.9% | -64.5% | -2.1% | -1.4% | -59.6% | 62 | 5x | 97% | 1% |
| 5 | CAMS | +5.7% | +28.5% | -3.4% | -2.3% | +12.8% | 68 | 2x | 97% | 5% |
| 6 | INTELLECT | +26.7% | +133.5% | -3.4% | -2.3% | +117.8% | 70 | 2x | 98% | 4% |
| 7 | M&M | +20.2% | +101.0% | -3.4% | -2.3% | +85.3% | 60 | 3x | 95% | 3% |
| 8 | BRITANNIA | -10.3% | -51.5% | -0.3% | -0.4% | -52.2% | 61 | 5x | 95% | 0% |
| 9 | INDIGO | +39.4% | +197.0% | -3.4% | -2.3% | +181.3% | 61 | 2x | 98% | 6% |
| 10 | ASIANPAINT | -11.0% | -55.0% | -0.5% | -0.4% | -52.4% | 71 | 3x | 96% | 0% |
| 11 | SHK | -16.5% | -82.5% | -3.4% | -2.3% | -65.7% | 65 | 4x | 92% | 5% |
| 12 | BALMLAWRIE | -14.6% | -73.0% | -0.6% | -0.4% | -52.7% | 61 | 7x | 96% | 6% |
| 13 | ICICIBANK | +74.2% | +371.0% | -3.4% | -2.3% | +355.3% | 53 | 8x | 93% | 6% |
| 14 | CIPLA | +33.2% | +166.0% | -3.4% | -2.3% | +150.3% | 56 | 4x | 93% | 3% |
| 15 | BAJFINANCE | +33.3% | +166.5% | -3.4% | -2.3% | +150.8% | 91 | 4x | 88% | 7% |

---

## Portfolio Simulation Results

### Best and Worst Trades

**Best 10:**

| # | Date | Symbol | Base% | Hedged% | Portfolio |
|---|------|--------|-------|---------|-----------|
| 101 | 2020-08-21 | AFFLE | +131.7% | +642.8% | Rs. 4.15 Cr |
| 128 | 2021-07-05 | GRAVITA | +127.7% | +622.8% | Rs. 9.56 Cr |
| 119 | 2021-04-29 | SHAREINDIA | +125.5% | +611.8% | Rs. 7.13 Cr |
| 14 | 2003-07-31 | GRASIM | +109.7% | +532.8% | Rs. 9.05 L |
| 5 | 2003-06-27 | LT | +106.9% | +518.8% | Rs. 9.80 L |
| 18 | 2003-10-28 | BHARTIARTL | +104.8% | +508.3% | Rs. 8.85 L |
| 20 | 2003-12-09 | ADANIENT | +102.2% | +495.3% | Rs. 12.09 L |
| 12 | 2003-07-30 | M&M | +100.6% | +487.3% | Rs. 9.05 L |
| 217 | 2024-07-01 | ZENTEC | +89.5% | +431.8% | Rs. 115.5 Cr |
| 71 | 2017-07-10 | TATACONSUM | +89.3% | +430.8% | Rs. 2.04 Cr |

**Worst 10:**

| # | Date | Symbol | Base% | Hedged% | DD at time |
|---|------|--------|-------|---------|-----------|
| 121 | 2021-05-27 | VGUARD | -10.6% | -65.7% | 0.0% |
| 222 | 2024-08-19 | SHK | -16.5% | -65.7% | 0.0% |
| 36 | 2007-10-01 | NTPC | -12.2% | -65.0% | 0.2% |
| 229 | 2024-09-17 | KAYNES | -17.9% | -64.8% | 7.2% |
| 227 | 2024-09-05 | KEC | -17.9% | -62.6% | 2.4% |
| 75 | 2018-05-22 | BERGEPAINT | -9.6% | -62.2% | 0.0% |
| 73 | 2018-05-11 | ASIANPAINT | -11.3% | -61.1% | 0.0% |
| 199 | 2023-12-15 | HCLTECH | -14.3% | -60.2% | 0.0% |
| 33 | 2006-02-28 | MARUTI | -13.0% | -59.6% | 7.2% |
| 141 | 2021-10-26 | TECHM | -12.9% | -59.6% | 0.0% |

---

## Options Hedging Strategies Compared

### All Strategies at 3x Leverage, 5% Position (Head-to-Head)

| Strategy | CAGR | MaxDD | Calmar | Win% | AvgWin | AvgLoss | Severe | Wipeout |
|----------|------|-------|--------|------|--------|---------|--------|---------|
| **NAKED FUTURES** | 27.3% | 10.9% | 2.49 | 65.7% | 115.2% | -37.1% | 85 | 0 |
| Covered Call 5% OTM | 12.5% | 7.9% | 1.57 | 71.7% | 39.0% | -32.6% | 60 | 0 |
| Covered Call 10% OTM | 9.3% | 10.5% | 0.88 | 68.7% | 34.6% | -35.5% | 75 | 0 |
| Covered Call 15% OTM | 9.7% | 13.2% | 0.73 | 66.9% | 39.3% | -36.7% | 83 | 0 |
| **Married Put 5% OTM** | 23.9% | 6.7% | 3.57 | 55.7% | 110.6% | -21.6% | 26 | 0 |
| **Married Put 10% OTM** | 26.9% | 9.5% | 2.83 | 63.9% | 111.6% | -29.2% | 92 | 0 |
| Married Put 15% OTM | 27.3% | 10.7% | 2.55 | 65.1% | 114.9% | -35.1% | 86 | 0 |
| Collar (5/5) | 8.1% | 2.4% | 3.40 | 67.5% | 17.9% | -13.0% | 0 | 0 |
| Collar (10/10) | 8.6% | 7.1% | 1.21 | 67.2% | 28.6% | -26.7% | 0 | 0 |
| Collar (10/5) | 4.7% | 7.0% | 0.67 | 61.4% | 14.3% | -18.2% | 0 | 0 |
| Collar (15/10) | 9.0% | 8.7% | 1.03 | 64.5% | 33.8% | -27.9% | 86 | 0 |
| Ratio Write 2:1 (7%) | -7.6% | 85.9% | -0.09 | 35.2% | 32.1% | -48.2% | 129 | 56 |
| Ratio Write 2:1 (10%) | -10.4% | 92.6% | -0.11 | 32.8% | 25.4% | -49.1% | 149 | 56 |

### Leverage Sweep - Naked Futures

| Leverage | CAGR | MaxDD | Calmar | Win% | Severe | Wipeout | Rs.10L -> |
|----------|------|-------|--------|------|--------|---------|-----------|
| 1x | 12.2% | 2.6% | 4.63 | 63.0% | 0 | 0 | 139L |
| 2x | 20.2% | 6.3% | 3.21 | 65.1% | 32 | 0 | 674L |
| 3x | 27.3% | 10.9% | 2.49 | 65.7% | 85 | 0 | 2,508L |
| 5x | 39.5% | 19.9% | 1.98 | 66.6% | 102 | **23** | 20,640L |
| 7x | 51.0% | 27.1% | 1.88 | 66.9% | 104 | **72** | 1,25,785L |

### Leverage Sweep - Married Put 10% OTM

| Leverage | CAGR | MaxDD | Calmar | Win% | Severe | Wipeout | Rs.10L -> |
|----------|------|-------|--------|------|--------|---------|-----------|
| 1x | 11.9% | 2.0% | 6.06 | 60.2% | 0 | 0 | 133L |
| 2x | 19.9% | 5.3% | 3.75 | 63.0% | 0 | 0 | 637L |
| 3x | 26.9% | 9.5% | 2.83 | 63.9% | 92 | 0 | 2,360L |
| 5x | **39.4%** | **17.7%** | **2.23** | 64.2% | 104 | **0** | **20,151L** |
| 7x | 49.4% | 25.4% | 1.94 | 64.2% | 108 | 16 | 99,079L |

### Can Married Put Tame High Leverage? (5x)

| Strategy | CAGR | MaxDD | Calmar | Win% | Severe | Wipeout | Rs.10L -> |
|----------|------|-------|--------|------|--------|---------|-----------|
| Naked (baseline) | 39.5% | 19.9% | 1.98 | 66.6% | 102 | **23** | 20,640L |
| Married Put 5% | 34.9% | 13.2% | 2.65 | 57.2% | 118 | **0** | 9,589L |
| **Married Put 10%** | **39.4%** | **17.7%** | **2.23** | 64.2% | 104 | **0** | **20,151L** |
| Married Put 15% | 39.6% | 19.6% | 2.02 | 66.0% | 102 | 3 | 20,805L |
| Collar 15/5 | 6.4% | 18.1% | 0.36 | 59.9% | 113 | 0 | 42L |
| Collar 15/10 | 12.3% | 16.6% | 0.74 | 65.1% | 103 | 0 | 142L |
| Collar 20/10 | 16.0% | 17.3% | 0.93 | 64.5% | 104 | 0 | 303L |

### Top Risk-Adjusted Combinations (Calmar > 3, CAGR > 15%)

| Strategy | Lev | Pos% | CAGR | MaxDD | Calmar | Win% | Severe | Final |
|----------|-----|------|------|-------|--------|------|--------|-------|
| MP-10% | 2x | 3% | 17.1% | 2.0% | **8.53** | 63.0% | 0 | 369L |
| MP-5% | 2x | 3% | 15.5% | 2.2% | 7.14 | 54.2% | 0 | 270L |
| Naked | 2x | 3% | 17.3% | 3.0% | 5.84 | 65.1% | 32 | 389L |
| MP-5% | 3x | 3% | 20.2% | 3.6% | 5.54 | 55.7% | 26 | 676L |
| MP-10% | 3x | 3% | 22.4% | 4.4% | 5.05 | 63.9% | 92 | 1,020L |
| MP-5% | 5x | 3% | 28.6% | 5.8% | 4.96 | 57.2% | 118 | 3,187L |
| Naked | 3x | 3% | 22.7% | 5.0% | 4.50 | 65.7% | 85 | 1,092L |
| MP-5% | 2x | 5% | 17.6% | 3.9% | 4.49 | 54.2% | 0 | 412L |
| MP-10% | 2x | 5% | 19.9% | 5.3% | 3.75 | 63.0% | 0 | 637L |
| MP-10% | 5x | 3% | 31.8% | 8.6% | 3.70 | 64.2% | 104 | 5,580L |
| MP-5% | 3x | 5% | 23.9% | 6.7% | 3.57 | 55.7% | 26 | 1,371L |
| Naked | 2x | 5% | 20.2% | 6.3% | 3.21 | 65.1% | 32 | 674L |
| Naked | 5x | 3% | 32.0% | 10.0% | 3.19 | 66.6% | 102 | 5,854L |

### IV Sensitivity (3x leverage, 5% position)

| Strategy | IV 20% | IV 30% | IV 40% | IV 50% |
|----------|--------|--------|--------|--------|
| Naked | 27.3% / 10.9% | 27.3% / 10.9% | 27.3% / 10.9% | 27.3% / 10.9% |
| CC 10% | 5.4% / 16.1% | 9.3% / 10.5% | 14.7% / 8.2% | 20.9% / 7.3% |
| MP 10% | 29.0% / 8.4% | 26.9% / 9.5% | 23.3% / 11.6% | 18.2% / 15.7% |
| Collar 10/10 | 7.1% / 7.8% | 8.6% / 7.1% | 10.1% / 6.3% | 11.5% / 5.8% |

*Format: CAGR / MaxDD*

---

## Option Premium Estimates (% of stock price, 30% IV)

| OTM % | 30-day Call | 30-day Put | 60-day Call | 60-day Put |
|-------|-----------|-----------|-----------|-----------|
| 3% | 2.38% | 1.90% | 4.00% | 3.01% |
| 5% | 1.72% | 1.28% | 3.24% | 2.30% |
| 7% | 1.22% | 0.82% | 2.60% | 1.71% |
| 10% | 0.69% | 0.38% | 1.82% | 1.04% |
| 15% | 0.23% | 0.08% | 0.96% | 0.39% |

---

## Key Findings

### Covered Call: BAD for breakouts
- Caps the runners which ARE the entire edge
- CAGR drops from 27.3% to 9.3% at 3x leverage
- Only improves win rate marginally (65.7% -> 68.7%)

### Married Put: THE winner
- CAGR cost is only 0.1-0.5% vs naked at same leverage
- Eliminates ALL wipeout trades at 5x (23 -> 0)
- Calmar improves at every leverage level
- At 10% OTM: costs only 0.38%/month in premium

### Collar: Mediocre for breakouts
- Caps both sides - you lose the runners AND pay complexity cost
- Best Calmar at low leverage but terrible absolute returns
- Only makes sense if you need bounded risk profile

### Ratio Write: DISASTROUS
- -7.6% CAGR, 85.9% drawdown, 56 wipeout trades
- Naked call exposure + breakout runners = catastrophe

---

## Charts

Generated in `verification_output/`:
- `portfolio_growth_drawdown.png` - Equity curve + drawdown chart
- `return_distribution.png` - Base vs hedged return distributions
- `yearly_returns.png` - Year-by-year average hedged returns

---

## TradingView Pine Script - SuperTrend-Style Trade Setup

Auto-detects PRIMARY system signals (ALPHA OR T1B OR MOMVOL) on any NSE daily chart and shows the **full trade lifecycle** with SuperTrend-style trailing stop visualization.

**Trade Visualization (SuperTrend-style):**
- **Entry signals** (`plotshape` triangles below bar, color-coded by strategy):
  - Orange = ALPHA | Green = T1B | Blue = MOMVOL
- **Trailing stop line** (step-line plot that ratchets up with price):
  - Green step-line below price while trade is active
  - Red step-line at exit when stop is hit
  - Visually shows the SL trail stepping up as stock makes new highs
- **Exit signals** (`plotshape` triangles above bar):
  - Green triangle = profitable exit | Red triangle = loss exit
  - Shows exit price and P&L% in label
- **Entry price line** (blue dotted) from entry to exit for reference
- **Light green background** while trade is active
- **Info table** with SL mode, trade stats, and current indicators

**Stop Loss Modes:**
- **FIXED** (default): SL = consolidation low. Original system, 67.5% WR, PF 6.38
- **TRAIL**: SL = highest high since entry minus X%. Recommended 20% (0 severe losses)
- **RATCHET**: SL = max(consol_low, peak minus X%). Starts at consol low, ratchets up

**Trade Rules (as per V3 system):**
- Entry: First bar where ALPHA or T1B or MOMVOL filter passes (not already in trade)
- Stop loss: Depends on SL mode (FIXED/TRAIL/RATCHET)
- Exit: Price hits stop loss OR 125 trading days elapse (whichever first)

**File:** `verification_output/breakout_v3_primary.pine`

**How to use:**
1. Open any NSE stock on TradingView (daily timeframe)
2. Pine Editor > Paste script > Add to Chart
3. Scroll through history to see complete trade setups with trailing SL
4. Change SL Mode in Settings to compare FIXED vs TRAIL vs RATCHET
5. Toggle individual strategies or visual elements on/off

**Settings:**
- Consolidation Length (default 60) - lookback for breakout and stop loss
- Max Holding Period (default 125 trading days)
- Stop Loss Mode: FIXED / TRAIL / RATCHET
- Trail % (default 20%) - for TRAIL/RATCHET modes
- Show/hide ALPHA, T1B, MOMVOL signals individually
- Show/hide Stop Loss trail and Entry price lines

**Note:** ATH uses 500-bar lookback (approximation). Weekly EMA uses `request.security` for weekly timeframe data. Only one trade at a time (new signals ignored while in trade).

---

## Files Reference

| File | Purpose |
|------|---------|
| `services/consolidation_breakout.py` | V3 strategy constants and filter functions |
| `breakout_analysis_enhanced.csv` | 9,739 trades, 53 indicator columns |
| `backtest_data/market_data.db` | SQLite price database (4.3M rows, 476 symbols) |
| `run_options_hedged_backtest.py` | Options hedging backtest script |
| `run_trade_verification.py` | Trade verification + Pine Script generator |
| `run_futures_backtest.py` | Base futures leverage backtest |
| `run_liquid_fund_backtest.py` | Liquid fund idle cash simulation |
| `run_trailing_sl_backtest.py` | Trailing SL comparison (Fixed vs Trail vs Ratchet) |
| `trailing_sl_results.csv` | Per-trade trailing SL comparison data |
| `verification_output/` | Charts, Pine Script, trade logs |
