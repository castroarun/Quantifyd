# Project Requirements: Covered Calls Backtester

**Status:** Final Requirements
**Version:** 1.0
**Last Updated:** 2025-12-30
**Author:** Requirements Analyst

---

## Overview

A web-based backtesting application for **Covered Call** options strategies on Indian stock options (NSE/BSE). The system will analyze historical performance of covered call strategies across multiple stocks (Reliance, TCS, HDFC Bank, etc.), comparing different strike selection methods and exit strategies to identify optimal approaches.

### Project Goals

1. **Backtest covered call strategies** on Indian equity options
2. **Compare multiple strategies** (Delta-based, % OTM, ATM strike selection)
3. **Evaluate exit rules** (Hold to expiry, profit targets, stop losses)
4. **Generate comprehensive metrics** and visualizations
5. **Export results** in multiple formats for analysis

### Target Users

- **Retail Options Traders** - Testing strategy ideas before deployment
- **Research Analysts** - Validating covered call performance across market cycles
- **Portfolio Managers** - Assessing risk-adjusted returns vs buy-and-hold

---

## Design System

### Visual Identity

| Element | Specification |
|---------|---------------|
| **Primary Color** | Blue (#0d6efd) - Following QuantFlow branding |
| **Accent Color** | Green (#198754) for positive metrics, Red (#dc3545) for negative |
| **Background** | Light gray (#f8f9fa) |
| **Card Background** | White |
| **Typography** | System fonts (Bootstrap 5 defaults) |

### UI Framework

- **CSS Framework:** Bootstrap 5.3.x
- **Charts:** Chart.js 4.x (already in use in existing project)
- **Icons:** Font Awesome 6.x
- **Data Tables:** DataTables with Bootstrap 5 styling (existing pattern)

### Design Patterns (from kiteconnect01)

1. **Sticky Navbar** - Consistent navigation across pages
2. **Card-based Layout** - Metric cards with hover effects
3. **Progress Sections** - Show/hide based on task status
4. **Loading Spinners** - Bootstrap spinners during async operations
5. **Responsive Tables** - Max-height with scroll for large datasets

---

## Structure (Pages/Screens)

### Page Architecture

```
/                           # Landing/Dashboard
/login                      # Zerodha OAuth login
/logout                     # Session cleanup
/backtest                   # Main backtesting interface
/backtest/results           # Results view (or inline in /backtest)
/data-management            # Download/manage historical data
```

### Detailed Page Specifications

#### 1. Landing Page (`/`)

- **Purpose:** Entry point, show login status, quick navigation
- **Components:**
  - Login status indicator
  - Quick links to backtest and data management
  - Summary of available data (stocks downloaded, date range)

#### 2. Backtest Page (`/backtest`)

This is the **primary interface** with 4 sections:

**Section A: Configuration Panel**
- Stock selector (multi-select: Reliance, TCS, HDFC Bank, etc.)
- Date range picker (start/end dates)
- Position sizing: Fixed 1 contract (display only for MVP)
- DTE selection: 30-45 days (Monthly options)

**Section B: Strategy Comparison Settings**

| Strategy Type | Options to Compare |
|---------------|-------------------|
| Strike Selection | Delta-based (0.30, 0.40), % OTM (2%, 5%), ATM |
| Exit Rules | Hold to expiry, 50% profit target, 2x stop loss |

**Section C: Results Dashboard**

| Metric | Description |
|--------|-------------|
| Total Return (%) | Cumulative return over backtest period |
| Premium Yield (%) | Total premium collected / Capital |
| Win Rate (%) | % of trades that were profitable |
| Max Drawdown (%) | Largest peak-to-trough decline |
| Assignment Rate (%) | % of options assigned at expiry |
| Sharpe Ratio | Risk-adjusted return metric |
| vs Buy-and-Hold (%) | Relative performance comparison |

**Section D: Visualization Panel**

10 charts to display:
1. Equity Curve (strategy vs underlying)
2. Returns Distribution (histogram)
3. Monthly Returns Heatmap
4. Drawdown Chart
5. Premium Collected Over Time
6. Strike Selection Impact
7. DTE Impact Analysis
8. Win/Loss Distribution
9. P&L by Market Condition (bull/bear/sideways)
10. Rolling Sharpe Ratio

#### 3. Data Management Page (`/data-management`)

- Data download interface (from Zerodha Kite)
- Stock selection for bulk download
- Download status and progress
- Available data summary table
- Cache management

---

## Components Required

### Reusable UI Components

| Component | Description | Source |
|-----------|-------------|--------|
| `NavBar` | Top navigation with login/logout | Adapt from kiteconnect01 |
| `MetricCard` | Display single metric with icon | Adapt from kiteconnect01 |
| `LoadingSpinner` | Bootstrap spinner wrapper | Existing |
| `ProgressBar` | Task progress indicator | Existing |
| `DataTable` | Sortable, searchable table | Existing DataTables integration |
| `ChartContainer` | Chart.js wrapper with title | Adapt from existing |
| `DateRangePicker` | Start/end date selector | HTML5 date inputs |
| `MultiSelect` | Stock selector dropdown | New or Bootstrap Select |
| `StrategyToggle` | Enable/disable strategy options | New |
| `ExportButton` | Download results dropdown | Adapt from kiteconnect01 |

### Backend Services

| Service | Description | Reference |
|---------|-------------|-----------|
| `KiteService` | Zerodha API authentication | `services/kite_service.py` |
| `DataManager` | Historical data download/cache | `services/centralized_data_manager.py` |
| `BacktestEngine` | Core backtesting logic | `services/backtest_service.py` |
| `GreeksCalculator` | Options Greeks (Delta, Theta, etc.) | `utils/greeks_calculator.py` |
| `CoveredCallStrategy` | Covered call specific logic | **New** |
| `MetricsCalculator` | Performance metrics | Adapt from existing |
| `ReportGenerator` | PDF/CSV export | Adapt from kiteconnect01 |

---

## Technical Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| HTML5/Jinja2 | - | Templating |
| Bootstrap | 5.3.x | CSS framework |
| Chart.js | 4.4.x | Visualizations |
| DataTables | 1.13.x | Data tables |
| jQuery | 3.x | DOM manipulation (DataTables dependency) |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Core language |
| Flask | 2.x | Web framework |
| Flask-Session | - | Session management |
| Pandas | 2.x | Data processing |
| NumPy | - | Numerical calculations |
| pandas-ta | - | Technical indicators |
| scipy | - | Statistics (Greeks calculation) |

### Data & Storage

| Technology | Purpose |
|------------|---------|
| SQLite | Local database for cached data |
| JSON | Configuration files, token storage |
| CSV | Data export format |
| Parquet | Efficient historical data storage (optional) |

### APIs

| API | Purpose |
|-----|---------|
| Zerodha Kite Connect | Historical data, authentication |
| yfinance (fallback) | Free stock data if Kite unavailable |

### Infrastructure

| Component | Specification |
|-----------|---------------|
| **Environment** | Web Application (Browser-based) |
| **Deployment** | Local development initially, Vercel/Railway for production |
| **Session** | File-based session storage |

---

## Data Requirements

### Primary Data Source

**Zerodha Kite Connect API**
- Historical OHLCV data for underlying stocks
- Options chain data (when available)
- Real-time quotes for live features

### Data Fields Required

**Stock (Underlying) Data:**

| Field | Type | Description |
|-------|------|-------------|
| timestamp | datetime | Trading date/time |
| open | float | Opening price |
| high | float | High price |
| low | float | Low price |
| close | float | Closing price |
| volume | int | Trading volume |

**Options Data (Estimated/Calculated):**

| Field | Type | Description |
|-------|------|-------------|
| expiry_date | date | Option expiration |
| strike | float | Strike price |
| option_type | str | 'CE' (Call) |
| premium | float | Option price (estimated) |
| delta | float | Greeks - Delta |
| theta | float | Greeks - Theta |
| iv | float | Implied Volatility |

### Data Estimation Note

Since historical options data is limited from Zerodha API, the system will use:
1. **Black-Scholes model** for theoretical option pricing
2. **GreeksCalculator** class (existing in kiteconnect01) for Delta, Gamma, Theta, Vega
3. **ATR-based volatility proxy** for premium estimation (existing pattern)

### Indian Market Specifics

| Parameter | Value |
|-----------|-------|
| Risk-free rate | 7% (India) |
| Lot sizes | Stock-specific (e.g., RELIANCE: 250, TCS: 150) |
| Expiry | Monthly (last Thursday) |
| Trading hours | 9:15 AM - 3:30 PM IST |

---

## Existing Code to Reuse

### From `C:\Users\Castro\Documents\Projects\quantflow\kiteconnect01`

#### Services (Direct Reuse/Adaptation)

| File | Components | Adaptation Needed |
|------|------------|-------------------|
| `services/kite_service.py` | `get_kite()`, `get_access_token()`, `save_access_token()` | Minor: Update paths |
| `services/centralized_data_manager.py` | `get_data_manager()` | Moderate: Focus on stock data |
| `services/backtest_service.py` | `estimate_atm_option_premium()`, position tracking | Moderate: Adapt for covered calls |
| `utils/greeks_calculator.py` | `GreeksCalculator` class (Black-Scholes) | Direct reuse |

#### Strategies (Reference/Adapt)

| File | Concepts to Adapt |
|------|-------------------|
| `strategies/iron_condor_backtest.py` | Strike selection logic, P&L calculation |
| `services/nifty_backtest_service.py` | Multi-stock backtesting patterns |

#### Templates (UI Patterns)

| Template | Reusable Elements |
|----------|-------------------|
| `templates/nifty_backtest.html` | Form layout, progress tracking, results tables |
| `templates/dashboard.html` | Navbar, metric cards, market indices display |
| `templates/data_management.html` | Download UI, status tables |

#### Static Assets

| Asset | Description |
|-------|-------------|
| `static/css/styles.css` | Base styling |
| `static/js/dashboard.js` | Chart patterns, AJAX calls |

### Dependencies Already Resolved

From `requirements.txt`:
```
Flask
Flask-Session
python-dotenv
kiteconnect
pandas
numpy
yfinance
APScheduler
requests
Jinja2
openpyxl
reportlab
pytz
pandas-ta
scipy (add for Greeks)
```

---

## Constraints & Notes

### Technical Constraints

1. **Options Data Limitation**
   - Historical options prices not directly available from Zerodha
   - Must estimate premiums using Black-Scholes
   - Validate estimates against known market behavior

2. **API Rate Limits**
   - Zerodha API has rate limits
   - Implement caching and batch requests
   - Use centralized data manager pattern

3. **Browser Compatibility**
   - Target modern browsers (Chrome, Firefox, Edge)
   - Bootstrap 5 provides good cross-browser support

### Business Constraints

1. **Position Sizing**
   - MVP uses fixed 1 contract
   - Future: Add position sizing options

2. **Single Stock Focus**
   - Compare strategies on same underlying
   - Future: Portfolio-level backtesting

### Design Decisions

1. **Monolithic Flask App**
   - Single Flask application (consistent with kiteconnect01)
   - Templates with Jinja2
   - No separate frontend build step

2. **Session-based Auth**
   - File-based sessions (Flask-Session)
   - Token stored in JSON file
   - Pattern proven in existing project

3. **Async Background Tasks**
   - Use APScheduler for long-running backtests
   - Pattern: task_status dict with progress polling

---

## Out of Scope (for MVP)

| Feature | Reason | Future Phase |
|---------|--------|--------------|
| Multi-leg strategies (spreads, condors) | Complexity | Phase 2 |
| Real-time execution | Requires broker integration | Phase 3 |
| Portfolio backtesting (multiple underlyings) | MVP is single-stock | Phase 2 |
| Machine learning optimization | Research scope | Phase 3 |
| Mobile app | Web-first | Phase 3 |
| Social sharing / public deployment | Personal use first | Phase 2 |
| Dividend adjustments | Edge case handling | Phase 2 |
| Stock splits handling | Edge case handling | Phase 2 |
| Intraday backtesting | Daily timeframe for MVP | Phase 2 |
| Custom strategy builder | Fixed strategies for MVP | Phase 2 |

---

## Appendix A: Covered Call Strategy Logic

### Entry Logic

```
For each expiry cycle:
1. On entry date (e.g., Monday after prior expiry):
   - Own underlying shares (simulated)
   - Identify target strike based on method:
     a. Delta-based: Find strike with target delta (e.g., 0.30)
     b. % OTM: Find strike X% above current price
     c. ATM: Strike closest to current price
   - Calculate (estimate) premium for selected strike
   - Record: Entry price, strike, premium, DTE
```

### Exit Logic

```
For each open position:
Check daily:
1. If HOLD_TO_EXPIRY strategy:
   - Exit at expiry, calculate assignment/expire
2. If PROFIT_TARGET strategy:
   - Exit if option price <= 50% of entry premium
3. If STOP_LOSS strategy:
   - Exit if option price >= 2x entry premium

At expiry:
- If stock >= strike: Assigned (sell stock at strike + keep premium)
- If stock < strike: Expires worthless (keep stock + premium)
```

### P&L Calculation

```
Covered Call P&L =
  + Premium received
  + Stock appreciation (up to strike)
  - Stock depreciation (unlimited downside)
  - Assignment gain/loss (if assigned)
```

---

## Appendix B: Chart Specifications

| Chart | Type | X-Axis | Y-Axis | Notes |
|-------|------|--------|--------|-------|
| Equity Curve | Line | Date | Portfolio Value | Compare vs buy-and-hold |
| Returns Distribution | Histogram | Return % | Frequency | Normal dist overlay |
| Monthly Returns | Heatmap | Month | Year | Color intensity = return |
| Drawdown | Area | Date | Drawdown % | Fill below zero |
| Premium Collected | Bar | Month | Premium Rs | Cumulative option |
| Strike Impact | Grouped Bar | Strike Method | Return % | Compare strategies |
| DTE Impact | Line | DTE Range | Win Rate % | - |
| Win/Loss | Pie | - | - | Win vs Loss count |
| Market Condition | Stacked Bar | Condition | P&L | Bull/Bear/Sideways |
| Rolling Sharpe | Line | Date | Sharpe Ratio | 12-month rolling |

---

## Appendix C: Export Formats

### CSV Export
- Trade log with all fields
- Daily equity values
- Summary metrics

### PDF Export
- Executive summary (1 page)
- Metrics table
- Key charts (equity curve, returns dist)
- Trade log (paginated)

### Browser View
- Interactive charts
- Sortable/filterable tables
- Print-friendly CSS

---

## Document Approval

- [x] Requirements gathered from user
- [x] Existing codebase analyzed
- [x] Technical feasibility confirmed
- [ ] Ready for development (pending review)

---

**End of Document**
