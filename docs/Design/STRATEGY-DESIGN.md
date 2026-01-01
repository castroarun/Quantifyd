# Covered Call Strategy Design for Indian Stocks

**Status:** Final Recommendation
**Date:** 2025-12-30
**Based on:** Comprehensive research of Indian F&O markets, strategy variations, and data sources

---

## Executive Summary

After analyzing Indian F&O market structure, various covered call strategies, and available data infrastructure, I recommend the **Adaptive Delta-Based Covered Call Strategy** with the following key parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Stock Universe** | Top 15 F&O stocks by liquidity | Best bid-ask spreads, reliable execution |
| **Strike Selection** | 0.25-0.35 Delta (IV-adjusted) | Balance of premium and upside |
| **Expiry** | Monthly, 25-30 DTE entry | Best liquidity, avoid weekly gamma |
| **Exit Rule** | 50% profit OR 10 DTE | Capture decay, avoid expiry risk |
| **Position Size** | Max 10% portfolio per stock | Diversification |
| **Expected Return** | 12-18% annually | Conservative estimate |

---

## Part 1: The Optimal Strategy

### 1.1 Strategy Name: "Adaptive Covered Call"

The strategy adapts to market conditions using:
1. **IV Percentile** for strike selection
2. **Trend Filter** for coverage ratio
3. **Event Calendar** for risk management

### 1.2 Stock Universe (Recommended 15)

Based on F&O liquidity analysis:

| Tier | Stocks | Lot Value (Approx) | Options Liquidity |
|------|--------|-------------------|-------------------|
| **Tier 1** | Reliance, HDFC Bank, TCS, Infosys, ICICI Bank | Rs 6-10 lakh | Excellent |
| **Tier 2** | SBI, Axis Bank, Bharti Airtel, ITC, Kotak Bank | Rs 5-8 lakh | Very Good |
| **Tier 3** | Bajaj Finance, L&T, Tata Motors, M&M, Tata Steel | Rs 4-7 lakh | Good |

**Selection Criteria:**
- Nifty 50 constituent
- Average daily OI > 50,000 contracts
- Bid-ask spread < 1% of premium
- No frequent F&O ban history

### 1.3 Strike Selection Logic

```
IF IV_Percentile < 25%:
    # Low volatility - sell closer to ATM for meaningful premium
    Target_Delta = 0.35 (approximately 3-4% OTM)

ELIF IV_Percentile 25-50%:
    # Normal volatility - balanced approach
    Target_Delta = 0.30 (approximately 4-5% OTM)

ELIF IV_Percentile 50-75%:
    # Elevated volatility - wider strikes for safety
    Target_Delta = 0.25 (approximately 5-7% OTM)

ELSE (IV_Percentile > 75%):
    # High volatility - very wide strikes or consider collar
    Target_Delta = 0.20 (approximately 7-10% OTM)
    Consider adding protective put (collar)
```

### 1.4 Entry Rules

| Condition | Action |
|-----------|--------|
| DTE = 25-30 days | Enter new covered call |
| Stock above 20-day SMA | Use standard delta |
| Stock below 20-day SMA | Sell ATM/ITM for protection |
| Major event in next 7 days | Skip or use collar |
| Stock in F&O ban | No new positions |

### 1.5 Exit Rules

| Trigger | Action |
|---------|--------|
| **50% profit reached** | Close call, re-evaluate |
| **DTE = 10** | Close regardless of P&L |
| **Stock breaches strike** | Roll up and out OR accept assignment |
| **Stock drops > 10%** | Close call, reassess stock |
| **IV spikes > 50%** | Consider buying back call |

### 1.6 Rolling Rules

```
When to Roll:
├── Stock within 2% of strike AND DTE > 7
│   └── Roll UP and OUT to next month, higher strike
│       └── Only if net credit or minimal debit
│
├── Stock 5%+ above strike
│   └── Accept assignment OR aggressive roll (likely debit)
│
├── Stock declined, call at 80% profit
│   └── Close call, wait for bounce, re-sell
│
└── DTE < 7, stock near strike
    └── Close position (avoid gamma risk)
```

---

## Part 2: Market Condition Adaptations

### 2.1 Bull Market Regime

**Indicators:** Nifty above 50 DMA, FII net buyers > 5 consecutive days

| Adjustment | Implementation |
|------------|----------------|
| Coverage Ratio | 50-75% of holdings (leave some uncovered) |
| Strike Selection | 0.20 delta (8-10% OTM) |
| Duration | Weekly options on Nifty for flexibility |

### 2.2 Bear Market Regime

**Indicators:** Nifty below 50 DMA, India VIX > 20

| Adjustment | Implementation |
|------------|----------------|
| Coverage Ratio | 100% of holdings |
| Strike Selection | 0.40-0.50 delta (ATM/ITM) |
| Additional | Add protective puts (collar) |
| Duration | 45 DTE for more premium cushion |

### 2.3 Sideways Regime

**Indicators:** Nifty in 3% range for 2+ weeks, VIX < 14

| Adjustment | Implementation |
|------------|----------------|
| Coverage Ratio | 100% |
| Strike Selection | 0.35 delta (ATM-ish) for max theta |
| Duration | Weekly options to compound |

### 2.4 Event Calendar Management

| Event | Lead Time | Action |
|-------|-----------|--------|
| Union Budget | 2 weeks before | Reduce positions, use collars |
| RBI Policy | 1 week before | Normal, slightly wider strikes |
| Quarterly Results | 1 week before | Skip stock-specific calls |
| Elections | 1 month before | Reduce overall exposure |
| Expiry Day | 1 day before | All positions closed |

---

## Part 3: Capital Requirements & Sizing

### 3.1 Minimum Capital

| Approach | Capital Needed | Positions |
|----------|----------------|-----------|
| **Single Stock** | Rs 10-15 lakh | 1 covered call |
| **Diversified (5 stocks)** | Rs 50-75 lakh | 5 covered calls |
| **Full Strategy (10 stocks)** | Rs 1-1.5 crore | 10 covered calls |

### 3.2 Position Sizing Rules

```python
MAX_POSITION_SIZE = 0.10  # 10% of portfolio per stock
MAX_SECTOR_EXPOSURE = 0.30  # 30% max in one sector
CASH_BUFFER = 0.10  # 10% cash for margin calls/opportunities

# Example: Rs 1 crore portfolio
# Max per stock: Rs 10 lakh
# Max in Banking sector: Rs 30 lakh
# Minimum cash: Rs 10 lakh
```

### 3.3 Margin Optimization

**Pledging Strategy:**
1. Hold stocks in demat account
2. Pledge stocks for margin (haircut: 10-15% for large caps)
3. Use margin for short call positions
4. Net additional capital needed: ~15-20% of stock value

---

## Part 4: Risk Management

### 4.1 Risk Metrics to Monitor

| Metric | Threshold | Action if Breached |
|--------|-----------|-------------------|
| Portfolio Delta | -0.3 to -0.5 | Reduce if too negative |
| Max Drawdown | 15% | Review all positions |
| Single Stock Loss | 20% | Close position, reassess |
| Sector Concentration | 30% | Rebalance |

### 4.2 Maximum Loss Scenarios

**Worst Case Analysis:**

| Scenario | Stock Move | Call P&L | Net P&L |
|----------|------------|----------|---------|
| Crash (stock -30%) | -30% | +3% (premium kept) | -27% |
| Sharp Rally (+20%) | +5% (capped) | -12% (ITM call) | -7% vs buy-hold |
| Flat | 0% | +3% (premium) | +3% |

### 4.3 Black Swan Protection

For extreme events:
1. **Portfolio Puts**: Buy 5% OTM Nifty puts (1-2% of portfolio)
2. **Stop Loss**: Hard stop at 25% portfolio drawdown
3. **Cash Reserve**: Always maintain 10% liquid

---

## Part 5: Expected Returns

### 5.1 Return Projections

| Market Condition | Annual Premium Yield | Stock Return | Total Return |
|-----------------|---------------------|--------------|--------------|
| Bull (+15%) | 10-12% | 10-15% (capped) | 20-27% |
| Flat (0%) | 12-15% | 0% | 12-15% |
| Bear (-15%) | 8-10% | -15% | -5% to -7% |
| **Weighted Average** | **10-12%** | **5-8%** | **15-20%** |

### 5.2 Historical Benchmarks

Based on similar strategies globally (adjusted for Indian market):
- **CBOE BXM Index** (US covered call index): ~8-10% annually
- **Indian market premium**: +2-3% due to higher IV
- **Realistic expectation**: 12-18% annually

---

## Part 6: Implementation Architecture

### 6.1 System Components

```
┌─────────────────────────────────────────────────────────┐
│                   COVERED CALL SYSTEM                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ Data Layer   │───▶│ Strategy     │───▶│ Execution │  │
│  │              │    │ Engine       │    │ Layer     │  │
│  └──────────────┘    └──────────────┘    └───────────┘  │
│         │                   │                   │        │
│         ▼                   ▼                   ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ • Kite API   │    │ • Strike     │    │ • Order   │  │
│  │ • NSE Data   │    │   Selection  │    │   Mgmt    │  │
│  │ • Greeks     │    │ • Position   │    │ • Alerts  │  │
│  │   Calculator │    │   Sizing     │    │ • Logging │  │
│  └──────────────┘    └──────────────┘    └───────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Data Pipeline

```
Daily (9:00 AM):
├── Fetch stock prices (Kite API)
├── Fetch option chain (Kite API)
├── Calculate Greeks (Black-Scholes)
├── Calculate IV Percentile
├── Check F&O ban list (NSE)
└── Generate signals

Weekly (Saturday):
├── Download NSE bhav copies
├── Update IV percentile history
├── Performance report
└── Rebalance signals
```

### 6.3 Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **Language** | Python 3.10+ | Existing kiteconnect setup |
| **Broker API** | Kite Connect | Already have API key |
| **Data Storage** | SQLite/PostgreSQL | Positions, trades, history |
| **Greeks** | py_vollib + scipy | Black-Scholes implementation |
| **Scheduling** | APScheduler | Daily signal generation |
| **UI** | Flask + HTML | Simple dashboard |
| **Alerts** | Telegram Bot | Trade notifications |

---

## Part 7: Deployment Phases

### Phase 1: Paper Trading (4 weeks)
- Implement strategy logic
- Generate paper signals
- Track hypothetical P&L
- Validate against manual calculation

### Phase 2: Small Capital (8 weeks)
- Deploy with 1-2 positions
- Rs 15-20 lakh capital
- Real execution via Kite
- Monitor and refine

### Phase 3: Scale Up (Ongoing)
- Increase to 5-10 positions
- Add more stocks to universe
- Implement collar strategy
- Full automation

---

## Part 8: Success Metrics

### 8.1 KPIs to Track

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Win Rate** | >70% | Profitable trades / Total trades |
| **Average Premium Yield** | >1% monthly | Premium / Stock value |
| **Max Drawdown** | <15% | Peak to trough |
| **Sharpe Ratio** | >1.0 | Risk-adjusted return |
| **Assignment Rate** | <20% | Assigned / Total positions |

### 8.2 Review Schedule

| Frequency | Review Items |
|-----------|--------------|
| Daily | P&L, positions, alerts |
| Weekly | Performance vs benchmark |
| Monthly | Strategy parameter review |
| Quarterly | Major adjustments, stock universe |

---

## Appendix A: Recommended Stock Universe Details

| Stock | Sector | Lot Size | Contract Value | IV (Typical) |
|-------|--------|----------|----------------|--------------|
| RELIANCE | Oil & Gas | 250 | Rs 6.5L | 18-25% |
| HDFCBANK | Banking | 550 | Rs 9L | 15-22% |
| TCS | IT | 175 | Rs 7L | 16-24% |
| INFY | IT | 300 | Rs 5.5L | 18-26% |
| ICICIBANK | Banking | 700 | Rs 8L | 18-25% |
| SBIN | Banking | 1500 | Rs 9L | 22-32% |
| AXISBANK | Banking | 600 | Rs 6.5L | 20-28% |
| BHARTIARTL | Telecom | 475 | Rs 7.5L | 18-26% |
| ITC | FMCG | 1600 | Rs 7L | 14-20% |
| KOTAKBANK | Banking | 400 | Rs 7L | 16-24% |
| BAJFINANCE | Finance | 125 | Rs 9L | 25-35% |
| LT | Infra | 150 | Rs 5.5L | 18-26% |
| TATAMOTORS | Auto | 550 | Rs 5L | 28-40% |
| MARUTI | Auto | 100 | Rs 12L | 18-26% |
| TATASTEEL | Metals | 550 | Rs 8L | 25-35% |

*Note: Lot sizes and values are approximate and change periodically.*

---

## Appendix B: Greeks Calculation Reference

```python
# Black-Scholes for European options (Index options)
# Use py_vollib for fast calculation

from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega

# For American options (Stock options)
# Use binomial tree or approximation
from py_vollib.black_scholes_merton import black_scholes_merton

# IV Calculation (from option price)
from py_vollib.black_scholes.implied_volatility import implied_volatility
```

---

## Appendix C: Transaction Cost Model

```python
def calculate_costs(premium, num_lots, lot_size):
    """
    Calculate total transaction costs for covered call trade
    """
    total_premium = premium * num_lots * lot_size

    # Brokerage (Zerodha)
    brokerage = 20  # Flat per order

    # STT (on sell side)
    stt = total_premium * 0.0005  # 0.05%

    # Exchange charges
    exchange = total_premium * 0.00053

    # SEBI charges
    sebi = total_premium * 0.000001

    # GST
    gst = (brokerage + exchange) * 0.18

    # Stamp duty (buy side only, 0 for selling options)
    stamp = 0

    total_cost = brokerage + stt + exchange + sebi + gst + stamp

    return {
        'total_cost': total_cost,
        'net_premium': total_premium - total_cost,
        'cost_percentage': (total_cost / total_premium) * 100
    }
```

---

**Document Version:** 1.0
**Author:** Claude (Agentic Analysis)
**Review Status:** Ready for implementation
