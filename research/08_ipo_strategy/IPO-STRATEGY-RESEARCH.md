# IPO Launch Strategy Research

## Overview

Research into short-term breakout trading of recently-listed (IPO) stocks on the Indian market. The strategy buys stocks within their first 10-50 trading days when price breaches the initial all-time high with volume confirmation, then exits via various rule-based strategies.

**Phase 1-2:** Nifty 500 universe (145 IPO candidates, 476 symbols)
**Phase 3 (Expanded):** All NSE equities via yfinance (1151 IPO candidates, 1621 symbols)
**Phase 4-5 (65% WR Optimization):** Entry quality filters + exit optimization (483 configs)
**Backtest period:** 2015-2025 (sequential trades, 5% position size, Rs.1 Crore initial capital)
**Total configs tested:** 2,201 (938 on Nifty 500 + 780 on expanded + 483 optimization)

---

## Strategy Rules

### Entry Signal (Base)
1. Stock age: 10-50 trading days since listing
2. **Price breakout**: Close > highest high of first N days (ATH lookback)
3. **First-time breach**: Must be the first time price exceeds the initial ATH
4. **Volume confirmation**: Today's volume >= X x average volume of past M days
5. Day 1 volume excluded from averages (abnormal IPO listing day volume)

### Entry Quality Filters (Phase 4-5, Optional)
6. **Listing Gain**: Day 1 close > Day 1 open (IPO listed positive)
7. **Breakout Strength**: Close must be >= N% above the initial ATH (e.g., 3%)
8. **Gap-Up**: Today's open > yesterday's close (positive overnight sentiment)

### Entry Parameters Tested
| Parameter | Values Tested | Best Value (Expanded) |
|-----------|--------------|----------------------|
| ATH lookback (days) | 3, 5, 7, 10 | **10** (highest PF + most trades) |
| Volume multiplier | 1.0x, 1.25x, 1.5x, 2.0x, 2.5x | **1.0x-1.25x** (minimal impact on expanded universe) |
| Volume avg period | 15, 20, 30 days | **15-20** (marginal differences) |

### Exit Strategies Tested (13 variants)

**Expanded Universe Results (1151 IPO stocks):**

| Exit Strategy | Parameters | Avg Hold | Avg Win Rate | Avg PF |
|---------------|-----------|----------|-------------|--------|
| **Fixed Target 15/7** | Target +15%, SL -7% | 29 days | 44.7% | 1.65 |
| **Fixed Target 20/10** | Target +20%, SL -10% | 48 days | 47.0% | 1.73 |
| **Fixed Target 25/10** | Target +25%, SL -10% | 59 days | 43.0% | 1.73 |
| **Fixed Target 30/15** | Target +30%, SL -15% | 97 days | 50.6% | 1.97 |
| **Fixed Target 40/15** | Target +40%, SL -15% | 116 days | 46.9% | 2.11 |
| **Fixed Target 60/20** | Target +60%, SL -20% | 210 days | 42.6% | 1.97 |
| **Time Exit 60D/SL15** | Max 60 days, SL -15% | 70 days | 49.6% | 2.14 |
| **Time Exit 90D/SL15** | Max 90 days, SL -15% | 97 days | 48.4% | 2.25 |
| **EMA Cross 8/21** | Exit on fast < slow | 47 days | 40.2% | 2.30 |
| **EMA Cross 10/30** | Exit on fast < slow | 62 days | 40.6% | 2.39 |
| **ATR Trail 10/3** | 3x ATR(10) trailing | 28 days | 43.1% | 2.03 |
| **Trailing SL 8%** | 8% trail from peak | 9 days | 44.4% | 1.43 |
| **Supertrend 14/2** | ATR(14), mult 2.0 | 299 days | 47.0% | 2.65 |

---

## Key Findings (Expanded Universe)

### 1. Survivorship Bias Was Massive

The original Nifty 500 results were severely inflated by survivorship bias. Expanding from 145 to 1151 IPO candidates revealed the true edge:

| Metric | Nifty 500 (145 IPOs) | Expanded (1151 IPOs) | Delta |
|--------|---------------------|---------------------|-------|
| Avg trades per config | 44 | 270 | +6x |
| Avg win rate | 57% | 45% | **-12pp** |
| Avg profit factor | 3.15 | 1.98 | **-1.17** |
| Statistical reliability | Poor (26-67 trades) | Good (156-420 trades) | Much better |

**Verdict:** The strategy still has a genuine edge (PF ~2.0), but the original 57-73% win rates were fantasy — driven by only including stocks that became large enough to enter the Nifty 500. Real-world performance with a proper universe is ~45-53% win rate with PF 1.6-2.8.

### 2. Volume Multiplier Has Minimal Impact (Changed!)

On the expanded universe, volume confirmation barely matters:

| Vol Multiplier | Avg Win Rate | Avg PF | Avg Trades | Avg Median Return |
|---------------|-------------|--------|------------|-------------------|
| 1.0x (no filter) | 45.3% | 1.97 | 331 | -4.5% |
| 1.25x | 45.8% | 2.01 | 307 | -3.3% |
| 1.5x | 45.3% | 1.97 | 280 | -3.8% |
| 2.0x | 44.6% | 1.93 | 236 | -4.5% |
| 2.5x | 45.6% | 2.01 | 198 | -3.3% |

**Verdict:** Only 0.5% win rate spread across all multipliers. On the original Nifty 500 data, 2.5x looked 6pp better — that was survivorship bias, not a real volume filter effect. Use 1.0x-1.25x for maximum trade count with no loss of edge.

### 3. ATH Lookback: 10 Days Now Optimal (Changed!)

| ATH Lookback | Avg Win Rate | Avg PF | Avg Trades | Avg Median Return |
|-------------|-------------|--------|------------|-------------------|
| 3 days | 45.4% | 1.98 | 221 | -3.3% |
| 5 days | 44.2% | 1.87 | 248 | -4.8% |
| 7 days | 45.8% | 1.97 | 272 | -3.3% |
| **10 days** | **45.7%** | **2.09** | **339** | **-4.0%** |

**Verdict:** 10-day lookback gives the highest PF (2.09) and most trades (339). With the broader universe, the wider ATH lookback captures more genuine breakouts. 7-day is nearly tied. 5-day is weakest.

### 4. Trailing SL 8% + ATH10 is the Best All-Round System

On the composite score (win rate, PF, median return, max loss, trade count), the top 14 configs are ALL trailing SL 8% with ATH 10-day lookback. This strategy:
- Captures the initial breakout momentum
- Exits quickly (avg 10 days) before most false breakouts reverse
- Limits max loss to -8%
- Generates 250-420 trades (high statistical significance)

### 5. Fixed Target 30/15 Has Highest Win Rate

The only configs exceeding 53% win rate on the expanded universe are FIX_T30_SL15 with 2.5x volume and ATH 10d. The wide target (+30%) and loose stop (-15%) allow time for the breakout to develop. Median return is +30% (target hit for most winners), avg hold 83-87 days.

### 6. EMA Cross Exits Have Highest Raw PF

Despite only 40-42% win rate, EMA cross exits (10/30) produce PF 2.7+ because winners are huge (+50% avg) while losers are small (-10% avg). This is a "big catch" strategy that loses most trades but captures multi-bagger IPO moves when they happen.

### 7. Supertrend Remains Impractical

Supertrend produced PF 2.65 and the highest expectancy per trade, but with 299-day average holds. Even on the expanded universe, this is "buy IPO and hold for a year" — not a breakout strategy.

---

## Best Systems (Expanded Universe)

### Tier 1: Quick Scalp (ATH 10d, ~10 day holds, 250-420 trades)

Best risk-adjusted systems — high trade count, controlled max loss, fast capital turnover.

| Rank | System | Win% | PF | Trades | Median Ret | Avg Hold | Max Loss |
|------|--------|------|-----|--------|-----------|----------|----------|
| 1 | **TRAIL_8PCT / V2.0 / ATH10 / VA20** | **46.6%** | **2.51** | 298 | -0.5% | 10d | -8.0% |
| 2 | TRAIL_8PCT / V2.5 / ATH10 / VA30 | 45.9% | 2.81 | 233 | -0.7% | 11d | -8.0% |
| 3 | TRAIL_8PCT / V2.5 / ATH10 / VA20 | 46.6% | 2.69 | 249 | -0.7% | 11d | -8.0% |
| 4 | TRAIL_8PCT / V1.0 / ATH10 / VA20 | 46.2% | 2.20 | 418 | -0.7% | 11d | -8.0% |
| 5 | TRAIL_8PCT / V1.25 / ATH10 / VA15 | 46.1% | 2.18 | 388 | -0.7% | 11d | -8.0% |

### Tier 2: Position Trades (Fixed Target, ~90 day holds, 190-310 trades)

Higher win rate, larger returns per trade, but longer capital tie-up.

| Rank | System | Win% | PF | Trades | Median Ret | Avg Hold | Max Loss |
|------|--------|------|-----|--------|-----------|----------|----------|
| 1 | **FIX_T30_SL15 / V2.5 / ATH10 / VA30** | **53.6%** | **2.22** | 233 | +30.1% | 83d | -30.0% |
| 2 | FIX_T30_SL15 / V2.5 / ATH10 / VA20 | 53.0% | 2.18 | 249 | +30.0% | 87d | -30.0% |
| 3 | FIX_T30_SL15 / V1.25 / ATH7 / VA15 | 52.6% | 2.08 | 312 | +7.0% | 99d | -30.0% |
| 4 | FIX_T40_SL15 / V1.5 / ATH3 / VA15 | 48.9% | 2.35 | 235 | -1.4% | 124d | -30.0% |
| 5 | TIME_60D_SL15 / V1.25 / ATH3 / VA15 | 52.9% | 2.39 | 255 | +1.0% | 72d | -15.0% |

### Tier 3: Big-Catch Trend Following (EMA cross, ~50-65 day holds)

Low win rate but massive winners — "catch the ZOMATO/DELHIVERY type runs."

| Rank | System | Win% | PF | Trades | Median Ret | Avg Hold |
|------|--------|------|-----|--------|-----------|----------|
| 1 | EMA_10_30 / V1.25 / ATH10 / VA15 | 41.8% | 2.79 | 388 | -3.2% | 66d |
| 2 | EMA_8_21 / V1.25 / ATH7 / VA15 | 41.7% | 2.73 | 312 | -2.5% | 48d |
| 3 | EMA_10_30 / V1.0 / ATH10 / VA20 | 41.6% | 2.74 | 418 | -3.2% | 66d |

---

## Best per Exit Strategy (Expanded Universe)

| Exit Strategy | Best Config | Win% | PF | Trades | Median Ret | Avg Hold |
|---------------|------------|------|-----|--------|-----------|----------|
| **Fixed Target** | FIX_T30_SL15 / V2.5 / ATH10 / VA20 | 53.0% | 2.18 | 249 | +30.0% | 87d |
| **Time Exit** | TIME_60D_SL15 / V1.0 / ATH10 / VA20 | 49.5% | 2.05 | 418 | -0.2% | 70d |
| **ATR Trail** | ATR_10_3 / V1.0 / ATH10 / VA20 | 44.7% | 2.05 | 418 | -2.0% | 28d |
| **Trailing SL** | TRAIL_8PCT / V2.0 / ATH10 / VA20 | 46.6% | 2.51 | 298 | -0.5% | 10d |
| **EMA Cross** | EMA_8_21 / V1.0 / ATH7 / VA20 | 42.4% | 2.69 | 335 | -2.5% | 48d |

---

## Phase 4-5: 65%+ Win Rate Optimization (483 configs)

Phase 3's best win rate was 53.6% — too low for many traders. Phase 4-5 explored two levers to push past 65%:

1. **Entry quality filters** — RSI, EMA, ADX, MFI, listing gain, breakout %, gap-up, above avg price
2. **Asymmetric exit ratios** — tight targets with wide stop losses (T5-T10 / SL15-SL20)

### What We Added to the Backtester

New parameters added to `IPOConfig`:

| Category | Parameter | Values Tested |
|----------|-----------|---------------|
| **Entry: RSI** | `use_rsi_filter`, `rsi_min`, `rsi_period` | RSI(7/10) > 50/55/60/65 |
| **Entry: EMA** | `use_ema_filter`, `ema_filter_period` | Close > EMA(5/7) |
| **Entry: ADX** | `use_adx_filter`, `adx_min`, `adx_period` | ADX(7) > 15/20/25 |
| **Entry: MFI** | `use_mfi_filter`, `mfi_min`, `mfi_period` | MFI(7) > 50/60 |
| **Entry: Breakout %** | `min_breakout_pct` | 1/2/3/5/7/10% |
| **Entry: Listing Gain** | `require_listing_gain` | Day 1 close > Day 1 open |
| **Entry: Gap-Up** | `require_gap_up` | Today open > yesterday close |
| **Entry: Above Avg** | `require_above_avg_price` | Close > avg since listing |
| **Exit: Hybrid** | `hybrid_exit`, `hybrid_target_pct`, `hybrid_trail_pct` | target_or_trail, target_with_breakeven |
| **Exit: Breakeven** | `use_breakeven_stop`, `breakeven_trigger_pct` | Move SL to entry after +3/5/7% |
| **Exit: Trail Tighten** | `use_time_tightening`, `tighten_after_days` | Tighten trail after N days |

### Key Discoveries

**1. Listing Gain + Breakout 3% is the only filter combo that matters.**

IPOs that (a) list positive (day 1 close > open) AND (b) break out 3%+ above their initial ATH have dramatically higher success rates. This combo reduces the universe from 259 to 51 trades but concentrates on the highest-quality setups.

| Filter | Trades | Best WR | Best PF | Effect |
|--------|--------|---------|---------|--------|
| None (base) | 259 | 53.6% | 2.22 | Baseline |
| RSI(7) > 50-65 | 226-233 | 53.6% | 2.22 | **Zero effect** |
| EMA(5/7) | 233 | 53.6% | 2.22 | **Zero effect** |
| ADX(7) > 15-25 | 232 | 53.9% | 2.24 | **Zero effect** |
| MFI(7) > 50-60 | 232 | 53.9% | 2.25 | **Zero effect** |
| Above Avg Price | 233 | 53.6% | 2.22 | **Zero effect** (trivially true) |
| Breakout > 3% | 131 | 55.0% | 2.30 | +1.4pp WR |
| Listing Gain | 99 | 53.5% | 2.28 | Filters weak IPOs |
| **LG + BRK3%** | **64** | **60.9%** | **2.95** | **+7.3pp WR** |
| **LG + BRK3% + T30/SL20** | **51** | **62.7%** | **2.55** | **+9.1pp WR** |

**Why are RSI/EMA/ADX/MFI useless?** Because a stock breaking its ATH for the first time is, by definition, in a strong uptrend with elevated momentum indicators. These filters are redundant with the breakout signal itself.

**2. Win rate is primarily driven by Target/SL ratio, NOT entry filters.**

The breakthrough to 65%+ came from making SL much wider than the target:

| Target | SL | Ratio (SL:Target) | WR (base) | WR (LG+BRK3) |
|--------|----|--------------------|-----------|---------------|
| 30% | 15% | 0.5:1 | 53.6% | 60.9% |
| 30% | 20% | 0.67:1 | 54.1% | 62.7% |
| 10% | 15% | 1.5:1 | 66.0% | 72.5% |
| 8% | 15% | 1.9:1 | 68.0% | 74.5% |
| 8% | 20% | 2.5:1 | 73.0% | 82.4% |
| 6% | 20% | 3.3:1 | 75.3% | 84.3% |
| 5% | 20% | 4.0:1 | 77.2% | 86.3% |

The wider SL gives trades room to breathe through initial post-breakout volatility before hitting the target.

**3. Time-capped exits produce the highest profit factors at 65%+ WR.**

Holding for exactly 20-30 days with a wide SL (no fixed target) captures natural momentum without the constraint of a tight target:

| Exit Style | Best Config | WR% | PF | Expectancy |
|-----------|------------|-----|-----|------------|
| Fixed T5/SL20 | LG+BRK3 | 86.3% | 2.33 | +4.07% |
| Fixed T10/SL20 | LG+BRK3 | 80.4% | 2.70 | +7.27% |
| **Time 20d/SL20** | **LG+BRK3** | **66.7%** | **4.25** | **+13.72%** |
| **Time 30d/SL20** | **LG+BRK3** | **66.7%** | **4.14** | **+14.43%** |
| Fixed T40/SL30 | LG+BRK3 | 66.7% | 2.74 | +19.39% |

---

## Best Systems: Phase 4-5 Results (65%+ Win Rate)

### Pareto-Optimal Systems (no other config has BOTH higher WR AND higher PF)

| System | WR% | PF | Exp/Trade | Total Ret% | Trades | Avg Hold | Category |
|--------|-----|-----|-----------|-----------|--------|----------|----------|
| LG+BRK3 + T5/SL20 | **86.3%** | 2.33 | +4.07% | 207% | 51 | 31d | Scalper |
| LG+BRK3 + T6/SL20 | **84.3%** | **2.73** | +5.50% | 281% | 51 | 37d | Scalper |
| LG+BRK3 + TIME20d/SL20 | 66.7% | **4.25** | +13.72% | 700% | 51 | 29d | Swing |
| LG+BRK3 + TIME30d/SL20 | 66.7% | 4.14 | **+14.43%** | 736% | 51 | 42d | Swing |

### All Systems with WR >= 65%, PF >= 2.0, Trades >= 30

| # | System | WR% | PF | Exp/Trade | Ret% | Trades | Hold | Style |
|---|--------|-----|-----|-----------|------|--------|------|-------|
| 1 | LG+BRK3 + T5/SL20 | 86.3 | 2.33 | +4.07 | 207 | 51 | 31d | Scalper |
| 2 | LG+BRK3 + T6/SL20 | 84.3 | 2.73 | +5.50 | 281 | 51 | 37d | Scalper |
| 3 | LG+BRK3 + T8/SL20 | 82.4 | 2.56 | +5.91 | 301 | 51 | 39d | Scalper |
| 4 | LG+BRK3 + T10/SL20 | 80.4 | 2.70 | +7.27 | 371 | 51 | 44d | Scalper |
| 5 | LG+BRK3 + T5/SL15 | 80.4 | 1.91 | +3.17 | 162 | 51 | 19d | Scalper |
| 6 | LG+BRK3 + T6/SL15 | 76.5 | 1.91 | +3.76 | 192 | 51 | 22d | Scalper |
| 7 | LG+BRK3 + T3/SL10 | 74.5 | 1.66 | +2.07 | 106 | 51 | 7d | Quick |
| 8 | LG+BRK3 + T8/SL15 | 74.5 | 1.87 | +4.15 | 212 | 51 | 24d | Scalper |
| 9 | LG+BRK3 + T10/SL15 | 72.5 | 2.06 | +5.41 | 276 | 51 | 26d | Scalper |
| 10 | LG+BRK3 + T3/SL7 | 72.5 | 1.80 | +2.22 | 113 | 51 | 5d | Quick |
| 11 | LG+BRK3 + T4/SL10 | 72.5 | 1.72 | +2.41 | 123 | 51 | 9d | Quick |
| 12 | BRK10 + T5/SL7 | 72.4 | 2.81 | +4.96 | 144 | 29 | 10d | Quick |
| 13 | LG+BRK5 + T10/SL15 | 72.4 | 2.23 | +6.14 | 178 | 29 | 26d | Scalper |
| 14 | LG+BRK3 + T5/SL10 | 70.6 | 1.66 | +2.37 | 121 | 51 | 10d | Quick |
| 15 | GAPUP+LG+BRK3 + T5/SL7 | 69.8 | 1.94 | +2.75 | 118 | 43 | 9d | Quick |
| 16 | LG+BRK3 + T3/SL5 | 68.6 | 2.01 | +2.40 | 122 | 51 | 5d | Quick |
| 17 | LG+BRK3 + T4/SL7 | 68.6 | 1.69 | +2.14 | 109 | 51 | 7d | Quick |
| 18 | LG+BRK7 + T10/SL15 | 68.4 | 1.96 | +5.57 | 106 | 19 | 32d | Scalper |
| 19 | LG+BRK3 + TIME20d/SL20 | 66.7 | 4.25 | +13.72 | 700 | 51 | 29d | Swing |
| 20 | LG+BRK3 + TIME30d/SL20 | 66.7 | 4.14 | +14.43 | 736 | 51 | 42d | Swing |
| 21 | LG+BRK3 + TIME20d/SL15 | 66.7 | 4.09 | +13.52 | 690 | 51 | 27d | Swing |
| 22 | LG+BRK3 + TIME30d/SL15 | 66.7 | 3.75 | +13.95 | 712 | 51 | 38d | Swing |
| 23 | LG+BRK3 + T40/SL30 | 66.7 | 2.74 | +19.39 | 989 | 51 | 162d | Hold |
| 24 | GAPUP+LG+BRK3 + T30/SL25 | 65.1 | 2.41 | +13.74 | 591 | 43 | 102d | Hold |
| 25 | GAPUP+LG+BRK3 + T8/SL10 | 65.1 | 1.82 | +3.52 | 152 | 43 | 14d | Scalper |
| 26 | BRK10 + T8/SL10 | 65.5 | 2.05 | +4.48 | 130 | 29 | 14d | Quick |
| 27 | LG+BRK5 + T5/SL7 | 65.5 | 1.81 | +2.69 | 78 | 29 | 6d | Quick |

### High-Volume Unfiltered Systems (259 trades, no entry filters)

| System | WR% | PF | Exp/Trade | Total Ret% | Avg Hold |
|--------|-----|-----|-----------|-----------|----------|
| T5/SL20 | 77.2 | 1.50 | +2.31 | 598 | 37d |
| T6/SL20 | 75.3 | 1.53 | +2.58 | 668 | 41d |
| T5/SL15 | 73.0 | 1.55 | +2.33 | 602 | 28d |
| T8/SL20 | 73.0 | 1.55 | +3.04 | 788 | 52d |
| T10/SL20 | 71.0 | 1.65 | +3.79 | 982 | 57d |
| T10/SL15 | 66.0 | 1.67 | +3.67 | 950 | 43d |
| T3/SL10 | 68.0 | 1.32 | +1.15 | 299 | 14d |
| T8/SL15 | 68.0 | 1.56 | +2.90 | 751 | 39d |

Note: Unfiltered base systems achieve 65-77% WR but PF is always < 2.0. Entry filters (LG+BRK3%) are needed for PF > 2.0.

---

## Recommended Systems for Live Trading (Updated Phase 4-5)

### System 1: "IPO Scalper" (Best WR + PF Balance)
```
Entry: ATH breakout (10-day lookback) + Volume >= 2.5x of 30-day avg
       + Listing Gain (day 1 close > open)
       + Breakout strength >= 3% above initial ATH
Exit:  Fixed Target +6% / Stop Loss -20%
Expected: 84.3% win rate, PF 2.73, ~51 trades per decade, avg hold 37 days
Median return: +8.3%
```
**Profile:** 8 out of 10 trades win. Tight +6% target captures quick post-breakout momentum. Wide -20% SL gives losers room — most don't hit -20%, they just slowly fade and eventually trigger. Only 51 trades over the backtest period (~5/year), so this is a selective, high-conviction strategy. Best for traders who want maximum consistency.

**Why it works:** IPOs that list positive AND break out 3%+ above their ATH are the strongest momentum stocks. Taking just +6% is modest — most of these stocks eventually go much higher — but it locks in gains before any reversal. The 20% SL rarely fires because these filtered stocks have genuine institutional buying pressure.

### System 2: "IPO Swing" (Best Profit Factor)
```
Entry: ATH breakout (10-day lookback) + Volume >= 2.5x of 30-day avg
       + Listing Gain (day 1 close > open)
       + Breakout strength >= 3% above initial ATH
Exit:  Time-based: hold exactly 20 days, Stop Loss -20%
Expected: 66.7% win rate, PF 4.25, ~51 trades per decade, avg hold 29 days
Expectancy: +13.72% per trade
```
**Profile:** 2 out of 3 trades win, and winners are 4.25x larger than losers in aggregate. No fixed target — you hold for 20 trading days and exit at whatever price. This captures the full natural momentum of the breakout move instead of capping it at a target. Total return 700% on 51 trades. Best for traders who want maximum capital growth per trade.

**Why PF = 4.25:** Winners average +20-40% returns over 20 days (IPO momentum is explosive), while losers are capped at -20% SL. The time exit acts as a natural trailing stop — after 20 days, most of the initial breakout energy has played out.

### System 3: "IPO Quick Scalp" (Original, Pre-Phase 4)
```
Entry: ATH breakout (10-day lookback) + Volume >= 2.0x of 20-day avg
Exit:  Trailing SL 8% from peak
Expected: 46.6% win rate, PF 2.51, ~298 trades per decade, avg hold 10 days
Max loss per trade: -8%
```
**Profile:** High-frequency quick trades. Lower win rate but 6x more trades than the filtered systems. Median return is slightly negative (-0.5%) but the tail of winners drives profitability. 298 trades = ~30/year. Best for active traders who want high trade frequency with systematic execution.

### System 4: "IPO Maximum Growth" (Highest Total Return at 65%+ WR)
```
Entry: ATH breakout (10-day lookback) + Volume >= 2.5x of 30-day avg
       + Listing Gain (day 1 close > open)
       + Breakout strength >= 3% above initial ATH
Exit:  Fixed Target +40% / Stop Loss -30%
Expected: 66.7% win rate, PF 2.74, ~51 trades per decade, avg hold 162 days
Expectancy: +19.39% per trade, Total return: 989%
Median return: +41.7%
```
**Profile:** Patient strategy. Hold for 5+ months on average. Targets +40% and gives losers up to -30% room. Produces the highest total return (989%) and expectancy (+19.39% per trade) of any 65%+ WR system. Best for long-term capital allocators.

---

## Strategy Decision Matrix

| Priority | Choose | System | WR% | PF | Trades/Decade |
|----------|--------|--------|-----|-----|---------------|
| Max Consistency | **System 1** | LG+BRK3+T6/SL20 | 84.3 | 2.73 | ~51 |
| Max PF | **System 2** | LG+BRK3+TIME20d/SL20 | 66.7 | 4.25 | ~51 |
| Max Trade Frequency | **System 3** | TRAIL 8% | 46.6 | 2.51 | ~298 |
| Max Total Return | **System 4** | LG+BRK3+T40/SL30 | 66.7 | 2.74 | ~51 |
| Max Win Rate | Variant | LG+BRK3+T5/SL20 | 86.3 | 2.33 | ~51 |
| Best Quick Scalp | Variant | LG+BRK3+T3/SL5 | 68.6 | 2.01 | ~51 |

---

## Survivorship Bias: The Full Picture

### What Changed When We Expanded the Universe

The original Nifty 500 research found "73% win rate, PF 5.27" for the best config. This was misleading because:

1. **Only survivors were tested.** Nifty 500 stocks are companies that grew large and successful. Their IPO breakouts naturally worked because these were destined-to-succeed companies.

2. **Failed IPOs were invisible.** Stocks that listed, broke out briefly, then collapsed to penny stock levels (or delisted) weren't in the database at all.

3. **The volume filter appeared powerful because it was filtering already-good stocks.** On Nifty 500 stocks, higher volume confirmation selected the most dramatic breakouts of already-successful companies. On the full NSE universe, volume confirmation adds only 0.5pp to win rate.

### What Survived the Expansion

The **directional edge is real**: PF consistently 1.5-2.8 across all exit strategies. The strategy genuinely identifies a profitable entry point (first ATH breach after IPO). The **exit strategy matters more than the entry filter**: ATR trail, EMA cross, and fixed target exits all produce PF 2.0+ regardless of entry parameters.

---

## Caveats & Limitations

1. **Partial Universe Coverage:** While we expanded from 476 to 1621 symbols, many NSE micro/nano-cap IPOs may still be missing from yfinance. Stocks that delisted quickly or changed ticker may not have data.

2. **No Slippage/Impact:** IPO stocks in their first 10-50 days may have wide spreads and thin order books. Our backtest uses closing prices with no slippage adjustment. This is especially problematic for micro-cap IPOs in the expanded universe.

3. **Sequential Capital:** Trades are sized as 5% of current capital and executed sequentially. In practice, multiple IPOs may signal simultaneously. With 270 trades over 10 years (~27/year), concurrent positions are likely.

4. **Period Dependency:** 2015-2025 was a strong bull market for Indian equities, especially mid/small-caps and IPOs. The 2021-2022 IPO boom may skew results.

5. **Negative Median Returns:** Most configs have negative median returns, meaning more than half of individual trades lose money. The strategy's edge comes from fat-tailed winners. This requires strict systematic execution — cherry-picking "good" setups will underperform.

6. **CAGR Numbers Unreliable:** Sequential compounding with outlier returns produces misleading CAGR figures. Per-trade metrics (win rate, PF, expectancy, median return) are the reliable comparison metrics.

---

## Files

| File | Purpose |
|------|---------|
| `services/ipo_strategy.py` | IPO Strategy Backtester (IPOConfig, IPOStrategyBacktester, preload_data) |
| `run_ipo_sweep.py` | Phase 1 sweep: 158 configs (Nifty 500, all exits x entry params) |
| `run_ipo_sweep_practical.py` | Phase 2 practical sweep: 780 configs (Nifty 500, short-term exits) |
| `run_ipo_expanded_sweep.py` | Phase 3 expanded sweep: 780 configs (1151 IPO stocks) |
| `run_ipo_65wr_sweep.py` | Phase 4 sweep: 337 configs (entry filters + hybrid exits) |
| `run_ipo_phase5_sweep.py` | Phase 5 sweep: 165 configs (wider SL ratios + filter combos) |
| `download_ipo_data.py` | yfinance data downloader for all NSE equities |
| `ipo_sweep_results.csv` | Phase 1 results (158 rows, Nifty 500) |
| `ipo_sweep_practical.csv` | Phase 2 results (780 rows, Nifty 500) |
| `ipo_expanded_sweep.csv` | Phase 3 results (780 rows, expanded universe) |
| `ipo_65wr_sweep.csv` | Phase 4 results (318 rows, entry filters + hybrid exits) |
| `ipo_phase5_sweep.csv` | Phase 5 results (165 rows, wider SL + filter combos) |

---

## Raw Data Summary

**Phase 1 Sweep (158 configs, Nifty 500):**
- Tested 7 exit strategy families with default entry params
- Key finding: Supertrend exits dominate by raw PF but hold 800+ days (not practical)

**Phase 2 Practical Sweep (780 configs, Nifty 500):**
- 13 short-term exit strategies × 5 volume multipliers × 4 ATH lookbacks × 3 vol avg periods
- 660 configs passed the practical filter (hold<200d, PF>=1.5)
- Best: FIX_T15_SL7 / V2.5 / ATH7 / VA30 at 73.1% win rate (26 trades) — **inflated by survivorship bias**

**Phase 3 Expanded Sweep (780 configs, 1151 IPO stocks):**
- Same 13 × 5 × 4 × 3 sweep on the full NSE equity universe
- 664 configs with PF >= 1.0 and hold < 200 days
- 270 avg trades per config (6x more than Nifty 500)
- Win rates: 40-54% (vs 45-73% on Nifty 500)
- PF: 1.0-2.8 (vs 1.5-5.3 on Nifty 500)
- Best composite: TRAIL_8PCT / V2.0 / ATH10 / VA20 (Win=46.6%, PF=2.51, 298 trades)
- Best win rate: FIX_T30_SL15 / V2.5 / ATH10 / VA30 (Win=53.6%, PF=2.22, 233 trades)
- Best PF: TRAIL_8PCT / V2.5 / ATH10 / VA30 (PF=2.81, Win=45.9%, 233 trades)

**Phase 4 Sweep (318 configs, 1151 IPO stocks):**
- Added 10 entry quality filters: RSI(7/10), EMA(5/7), ADX(7), MFI(7), breakout %, listing gain, gap-up, above avg price, consecutive close
- Added 3 hybrid exit modes: target+trail, target+breakeven, trail tightening
- Tested tight target configs: T8-T15 / SL3-SL7 / ATH5-ATH10
- Combined filter stacking: up to 4 filters simultaneously
- Key finding: RSI/EMA/ADX/MFI have ZERO effect. LG+BRK3% is the only useful combo.
- Best win rate: LG+BRK3%+T30/SL15 at 60.9% WR (64 trades, PF 2.95)
- Best with RSI: RSI7>55+LG+BRK3%+T30/SL15 at 61.7% WR (60 trades, PF 3.08)
- 42 configs hit 65%+ WR: 0 (ceiling was 61.7%)

**Phase 5 Sweep (165 configs, 1151 IPO stocks):**
- Breakthrough: asymmetric R:R with wide SL (SL = 2-4x target)
- Wide SL configs: T5-T10 / SL10-SL20, T30-T50 / SL20-SL30
- Time-capped exits: TIME10-30d / SL10-SL20
- All combos with LG+BRK3%, LG+BRK5-10%, GAPUP+LG
- 42 configs hit 65%+ WR
- Best WR: LG+BRK3+T5/SL20 at 86.3% WR (51 trades, PF 2.33)
- Best PF at 65%+: LG+BRK3+TIME20d/SL20 at 66.7% WR (51 trades, PF 4.25)
- Best total return at 65%+: LG+BRK3+T40/SL30 at 66.7% WR (989% total, PF 2.74)
