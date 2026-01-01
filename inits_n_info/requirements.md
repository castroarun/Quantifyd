# Project Requirements

**Status:** Pending
**Last Updated:** 2025-12-30
**Designer Agent:** Not yet invoked

---

## Purpose

This file captures structured requirements gathered by the **Designer Agent** before architecture design begins.

## How to Populate

Invoke the Designer agent:
```
@designer covered call backtest
```

The Designer will:
1. Search for relevant data sources/libraries
2. Ask clarifying questions (data, strategies, outputs)
3. Document requirements in this file

---

## Requirements (To Be Filled)

### 1. Project Overview
- **Name:** Covered_Calls
- **Type:** Python Research/Backtest Tool
- **Description:** Backtest and analyze covered call options strategies

### 2. User Requirements
| User Story | Priority | Notes |
|------------|----------|-------|
| As a researcher, I want to backtest covered calls on any ticker | High | Core functionality |
| As a trader, I want to compare strike selection methods | High | Delta vs % OTM |
| As an analyst, I want performance metrics | High | Sharpe, returns, drawdown |
| As a user, I want visualizations | Medium | Equity curve, distributions |

### 3. Technical Requirements
- **Language:** Python 3.10+
- **Libraries:** Pandas, NumPy, Matplotlib, Plotly
- **Data Storage:** CSV/Parquet files
- **Notebooks:** Jupyter for research

### 4. Data Requirements
- **Stock Data:** OHLC + adjusted close
- **Options Data:** Strikes, expirations, prices, Greeks
- **Source:** TBD (Yahoo Finance, CBOE, other)

### 5. Output Requirements
- Performance metrics table
- Equity curve chart
- Trade log
- Comparison to benchmark (buy-and-hold)

### 6. Integrations
- [ ] Data source API
- [ ] Broker API (optional, future)
- [ ] Export to PDF/HTML (optional)

---

## Approval

- [ ] Requirements reviewed by user
- [ ] Data source selected
- [ ] Ready for development
