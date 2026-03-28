# Intraday Options Data Vendor Research

**Date:** 2026-03-28
**Purpose:** Purchase 2 years of historical 1-min NIFTY + BANKNIFTY options data for backtesting straddle/strangle strategies.

---

## Vendor Comparison

### TrueData (truedata.in)

| Item | Detail |
|------|--------|
| Plan | Velocity Standard |
| Monthly cost | Rs 1,949/month |
| Annual cost | Rs 23,388/year |
| **2-year cost** | **~Rs 46,800** |
| Data available | 1-min IEOD for all NSE F&O instruments |
| History depth | 5+ years |
| Access method | Python SDK, API |
| Pros | Well-documented Python SDK, widely used in algo trading community, reliable data quality |
| Cons | Most expensive option |

### Global Data Feeds (globaldatafeeds.in)

| Item | Detail |
|------|--------|
| Pricing | Custom quotes only, no public price list |
| Forum reports | Rs 4,200 + GST/year for BANKNIFTY weekly options IEOD |
| Bulk package | Rs 98,000 for 3-year combined NIFTY + BANKNIFTY (one-time) |
| **2-year estimate** | **~Rs 65,000-70,000** (extrapolating 3yr rate) |
| Access method | Manual negotiation required |
| Pros | Cheaper per year than TrueData, bulk historical packages available |
| Cons | Need to negotiate, less community documentation |

### Free Alternative: TradingTuitions

- Offered free 1-min F&O data from 2021+
- Updates have paused — worth checking if existing archive covers required date range

---

## Recommendation

**TrueData is the better fit** — cheaper for 2 years, Python SDK makes it easy to download and load into SQLite, most commonly used source in the Indian algo trading community.

---

## Other Platforms Evaluated (Not Suitable)

These are SaaS backtesting tools — no raw data download:

- StockMock
- Opstra
- AlgoTest
- Sensibull

---

## NSE Bhav Copy (EOD) — Already Downloaded

While researching vendors, we also downloaded NSE's free EOD bhav copy data as a baseline:

| Metric | Value |
|--------|-------|
| Total rows | 1,390,537 |
| Date range | 2024-03-01 to 2026-03-27 |
| Trading days | 509 |
| NIFTY rows | 821,868 |
| BANKNIFTY rows | 568,669 |
| Table | `nse_options_bhav` in `market_data.db` |

### BS Calibration Against Real EOD Premiums

| IV | Avg BS Straddle | Ratio (Real/BS) | Match |
|----|-----------------|-----------------|-------|
| 10% | 176.30 | 1.335 | |
| 12% | 211.22 | 1.114 | * |
| **15%** | **263.68** | **0.892** | **Best fit** |
| 18% | 316.19 | 0.744 | |
| 20% | 351.22 | 0.670 | |

- **0-3 DTE**: BS@15% matches almost perfectly (ratio 1.015)
- **4-7 DTE**: BS@15% underestimates (ratio 0.782)

### Limitation

This is EOD close-price data only — not intraday snapshots at specific entry/exit times (e.g., 11:00 entry, 12:00 exit). Real intraday backtest requires vendor data above.

---

## BB Expansion Backtest Results (BS-Simulated Premiums)

### NIFTY50 — Best Config

| Parameter | Value |
|-----------|-------|
| Filter | BB Only (no additional filters needed) |
| Exit time | 15:20 |
| Strike | ATM |
| IV | 0.20 |
| Stop Loss | None |
| Target Profit | None |
| Trades | 345 |
| Win Rate | 99.4% |
| Total PnL | Rs 9,64,896 |
| Max Drawdown | Rs 2,568 |
| Sharpe | 23.17 |
| Profit Factor | 325.29 |

### BANKNIFTY — Best Config

| Parameter | Value |
|-----------|-------|
| Filter | BB Only |
| Exit time | 15:20 |
| Strike | ATM |
| IV | 0.20 |
| Stop Loss | 1.5x |
| Trades | 351 |
| Win Rate | 95.4% |
| Total PnL | Rs 5,57,116 |
| Max Drawdown | Rs 12,474 |
| Sharpe | 14.07 |
| Profit Factor | 21.25 |

### Key Findings from Phase Optimization

1. **Filter**: BB Only is best — additional filters (CPR, Narrow Range, No Gap) reduce trade count without improving metrics
2. **Exit time is the biggest lever**: Later exits = more premium decay captured (15:20 >> 12:00)
3. **Strike**: ATM beats OTM at all distances (0.3% to 1.0% OTM)
4. **IV sensitivity**: Higher IV = more premium collected = better results (IV 0.20 best, but this inflates premiums beyond real market)
5. **Stop Loss**: No SL needed for NIFTY (99.4% WR); 1.5x SL helps BANKNIFTY
6. **Target Profit**: No TP needed — letting positions run to exit time is optimal

### Caveat

These results use Black-Scholes simulated premiums, not real market prices. The BS model at IV=0.20 overestimates premiums by ~33% vs real EOD data. Results at IV=0.15 would be more conservative and realistic for 0-3 DTE trades.

---

## Entry Filter Optimization Summary

Tested 22 filters on top of BB Cooloff baseline across 453 trading days:

### Top Filters (by safety rate improvement)

| Filter | Edge vs Baseline | Safe Rate (<0.5%) |
|--------|-----------------|-------------------|
| CPR Above | +6.5% | 81.3% |
| Narrow Prev Day Range | +5.5% | 80.3% |
| Prev Day High Broken | +4.9% | 79.7% |
| Opening Range < 0.3% | +4.3% | 79.1% |

### Filters to Avoid

| Filter | Edge vs Baseline | Notes |
|--------|-----------------|-------|
| Gap Down | -10.9% | Strongly negative |
| CPR Below | -8.5% | Bearish bias hurts |

### Best Combined Setup

**BB + Narrow Prev Day + Exit 12:00 + DTE 1 = 100% safety rate** (61 qualifying days over 2 years)

---

## Volatility Filter Comparison

| Filter | Median Activation | <0.5% Safety | Edge vs No Filter |
|--------|-------------------|---------------|-------------------|
| BB Bandwidth Cooloff | 10:55 | 74.8% | +40% |
| ATR Squeeze | 10:35 | 55.5% | +21% |
| ATR + BB Combined | 10:55 | 70.6% | +37% |
| KC Band Containment | 09:20 | 40.5% | +5% |
| STARC Band Containment | 09:20 | 38.2% | +3% |
| Fractal Band Containment | 09:20 | 39.1% | +4% |
| No Filter (baseline) | 09:15 | 34.8% | — |

**BB Bandwidth Cooloff is the clear winner** — +40% edge over no filter. KC/STARC/Fractal bands fire on the first candle (09:20) providing almost no filtering value.
