# Covered_Calls - Product Requirements Document

**Status:** Approved
**Last Updated:** 2025-12-30
**Author:** Claude (Agentic Analysis)

---

## 1. Objective

Build an **Adaptive Covered Call Strategy System** for Indian F&O stocks that:
- Automatically selects optimal strikes based on IV percentile
- Generates entry/exit signals for covered call positions
- Executes trades via Kite Connect API
- Tracks performance and provides real-time monitoring

**Target Return:** 12-18% annually (premium + stock appreciation)

---

## 2. Strategy Summary

### The Adaptive Delta-Based Covered Call

| Component | Specification |
|-----------|---------------|
| **Universe** | Top 15 NSE F&O stocks by liquidity |
| **Strike Selection** | 0.20-0.35 Delta based on IV percentile |
| **Expiry** | Monthly options, 25-30 DTE entry |
| **Exit** | 50% profit OR 10 DTE |
| **Position Size** | Max 10% portfolio per stock |

### IV Percentile Rules

| IV Percentile | Target Delta | Approx OTM % |
|---------------|--------------|--------------|
| < 25% | 0.35 | 3-4% |
| 25-50% | 0.30 | 4-5% |
| 50-75% | 0.25 | 5-7% |
| > 75% | 0.20 | 7-10% |

---

## 3. Features

### 3.1 Core Features (MVP)

| Feature | Priority | Description | Status |
|---------|----------|-------------|--------|
| Kite Connect Integration | High | Auth, data, orders | Pending |
| Greeks Calculator | High | Black-Scholes, IV, Delta | Pending |
| Strike Selector | High | IV-based delta targeting | Pending |
| Position Tracker | High | Track open covered calls | Pending |
| Signal Generator | High | Entry/exit alerts | Pending |
| Dashboard | Medium | Web UI for monitoring | Pending |

### 3.2 Phase 2 Features

| Feature | Priority | Description |
|---------|----------|-------------|
| Backtest Engine | Medium | Historical strategy testing |
| Telegram Alerts | Medium | Trade notifications |
| Auto-Execution | Medium | One-click order placement |
| Collar Strategy | Low | Protective puts addition |
| Multi-strategy | Low | Wheel, ratio writes |

### 3.3 User Stories

| As a... | I want to... | So that... |
|---------|--------------|------------|
| Trader | See today's covered call opportunities | I can quickly identify trades |
| Trader | Get alerts when positions need rolling | I don't miss expiry |
| Investor | Track my covered call P&L | I know my strategy performance |
| Analyst | Backtest different parameters | I can optimize the strategy |

---

## 4. Data Requirements

### 4.1 Data Sources (Finalized)

| Source | Data Type | Cost | Priority |
|--------|-----------|------|----------|
| **Kite Connect** | Live quotes, historical OHLCV | Rs 2000/mo | Primary |
| **NSE Bhav Copies** | EOD options data | Free | Backup |
| **yfinance** | Stock prices (adjusted) | Free | Supplementary |
| **py_vollib** | Greeks calculation | Free | Core |

### 4.2 Data Fields

**From Kite Connect:**
- Instrument tokens, lot sizes
- OHLCV for stocks and options
- Live quotes (LTP, bid, ask, OI)

**Calculated:**
- Implied Volatility (from option prices)
- Delta, Gamma, Theta, Vega
- IV Percentile (252-day lookback)

---

## 5. Technical Architecture

### 5.1 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Broker API | Kite Connect 5.x |
| Database | SQLite (positions.db) |
| Greeks | py_vollib + scipy |
| Web UI | Flask + Bootstrap |
| Scheduling | APScheduler |
| Alerts | python-telegram-bot |

### 5.2 Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    COVERED CALL SYSTEM                    │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  [Data Layer]         [Strategy]         [Execution]     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ Kite Client │───▶│ Strike      │───▶│ Order Mgr   │   │
│  │ NSE Parser  │    │ Selector    │    │ Scheduler   │   │
│  │ Greeks Calc │    │ Position    │    │ Alerts      │   │
│  │ IV Tracker  │    │ Manager     │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
│         │                  │                  │           │
│         ▼                  ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                    SQLite Database                   │ │
│  │  [positions] [trades] [iv_history] [instruments]    │ │
│  └─────────────────────────────────────────────────────┘ │
│                           │                               │
│                           ▼                               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                   Flask Dashboard                    │ │
│  │  [Positions] [Signals] [P&L] [Backtest]             │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 6. Stock Universe

### Recommended 15 Stocks

| Symbol | Sector | Lot Size | Contract Value |
|--------|--------|----------|----------------|
| RELIANCE | Oil & Gas | 250 | ~Rs 6.5L |
| HDFCBANK | Banking | 550 | ~Rs 9L |
| TCS | IT | 175 | ~Rs 7L |
| INFY | IT | 300 | ~Rs 5.5L |
| ICICIBANK | Banking | 700 | ~Rs 8L |
| SBIN | Banking | 1500 | ~Rs 9L |
| AXISBANK | Banking | 600 | ~Rs 6.5L |
| BHARTIARTL | Telecom | 475 | ~Rs 7.5L |
| ITC | FMCG | 1600 | ~Rs 7L |
| KOTAKBANK | Banking | 400 | ~Rs 7L |
| BAJFINANCE | Finance | 125 | ~Rs 9L |
| LT | Infra | 150 | ~Rs 5.5L |
| TATAMOTORS | Auto | 550 | ~Rs 5L |
| MARUTI | Auto | 100 | ~Rs 12L |
| TATASTEEL | Metals | 550 | ~Rs 8L |

---

## 7. Capital Requirements

| Configuration | Capital | Positions |
|---------------|---------|-----------|
| Minimum | Rs 15 lakh | 1-2 stocks |
| Recommended | Rs 50 lakh | 5 stocks |
| Full Strategy | Rs 1 crore | 10 stocks |

---

## 8. Risk Parameters

| Parameter | Limit |
|-----------|-------|
| Max position size | 10% of portfolio |
| Max sector exposure | 30% of portfolio |
| Max drawdown trigger | 15% |
| Single stock loss limit | 20% |
| Cash buffer | 10% minimum |

---

## 9. Success Metrics

| Metric | Target |
|--------|--------|
| Annual Return | 12-18% |
| Win Rate | >70% |
| Max Drawdown | <15% |
| Sharpe Ratio | >1.0 |
| Premium Yield | >1% monthly |

---

## 10. Timeline

| Week | Milestone |
|------|-----------|
| 1-2 | Core infrastructure (Kite, Greeks) |
| 3 | Strategy engine (signals) |
| 4 | Execution layer (paper trading) |
| 5 | Dashboard UI |
| 6 | Backtest validation |
| 7-8 | Paper trading (4 weeks) |
| 9+ | Live deployment |

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| API downtime | Medium | Fallback to manual, queue orders |
| Wrong Greeks | High | Validate against broker Greeks |
| Flash crash | High | Position limits, stop-loss |
| F&O ban | Medium | Daily ban list check |
| Physical settlement | Medium | Close before expiry |

---

## 12. Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Data source? | Kite Connect (primary) |
| Multiple assets? | Yes, 10-15 stocks |
| Real-time scanning? | Yes, daily signals |
| Broker integration? | Kite Connect for execution |

---

## 13. Approval

- [x] Strategy designed and documented
- [x] Data source selected (Kite Connect)
- [x] Stock universe defined
- [x] Risk parameters set
- [ ] Code implementation started
- [ ] Paper trading validated
- [ ] Live deployment approved

---

## 14. Related Documents

- [STRATEGY-DESIGN.md](Design/STRATEGY-DESIGN.md) - Full strategy specification
- [DEVELOPMENT-PLAN.md](DEVELOPMENT-PLAN.md) - Implementation roadmap
- [GLOSSARY.md](GLOSSARY.md) - Options terminology

---

**Document Version:** 1.0
**Approved By:** [Pending]
**Next Review:** After paper trading phase
