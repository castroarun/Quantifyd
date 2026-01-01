# Covered Calls - Development Plan

**Strategy:** Adaptive Delta-Based Covered Call for Indian Stocks
**Status:** Ready for Implementation
**Date:** 2025-12-30

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Kite Connect Integration
```
src/data/kite_client.py
├── Authentication (OAuth flow)
├── Session management
├── Instrument data fetching
├── Historical data API
└── Order placement
```

**Tasks:**
- [ ] Move API keys to .env file
- [ ] Implement token refresh logic
- [ ] Create instrument cache (daily refresh)
- [ ] Historical data fetcher with rate limiting

#### 1.2 Greeks Calculator
```
src/metrics/greeks.py
├── Black-Scholes model
├── IV calculation (from price)
├── Delta, Gamma, Theta, Vega
└── IV Percentile tracker
```

**Tasks:**
- [ ] Implement Black-Scholes functions
- [ ] Build IV history storage (SQLite)
- [ ] IV percentile calculation (252-day lookback)

#### 1.3 Database Schema
```sql
-- positions.db
CREATE TABLE stocks (
    id INTEGER PRIMARY KEY,
    symbol TEXT UNIQUE,
    lot_size INTEGER,
    sector TEXT,
    added_date DATE
);

CREATE TABLE options (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    expiry DATE,
    strike REAL,
    option_type TEXT,
    ltp REAL,
    iv REAL,
    delta REAL,
    oi INTEGER,
    timestamp DATETIME
);

CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    stock_qty INTEGER,
    stock_avg_price REAL,
    call_strike REAL,
    call_expiry DATE,
    call_premium REAL,
    call_qty INTEGER,
    entry_date DATE,
    status TEXT
);

CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    position_id INTEGER,
    trade_type TEXT,
    symbol TEXT,
    qty INTEGER,
    price REAL,
    timestamp DATETIME,
    pnl REAL
);

CREATE TABLE iv_history (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    date DATE,
    atm_iv REAL,
    iv_percentile REAL
);
```

---

### Phase 2: Strategy Engine (Week 2-3)

#### 2.1 Strike Selection Module
```
src/strategies/strike_selector.py
├── Delta-based selection
├── IV percentile adjustment
├── Liquidity filter
└── Strike rounding (to available strikes)
```

**Logic:**
```python
def select_strike(spot_price, iv_percentile, option_chain):
    # Determine target delta based on IV percentile
    if iv_percentile < 25:
        target_delta = 0.35
    elif iv_percentile < 50:
        target_delta = 0.30
    elif iv_percentile < 75:
        target_delta = 0.25
    else:
        target_delta = 0.20

    # Find strike closest to target delta
    # Filter for liquidity (OI > 5000)
    # Return selected strike
```

#### 2.2 Position Manager
```
src/strategies/position_manager.py
├── Entry signal generation
├── Exit signal generation
├── Roll decision logic
├── Position sizing
└── Portfolio limits check
```

#### 2.3 Signal Generator
```
src/strategies/signals.py
├── Daily scan of universe
├── Entry conditions check
├── Exit conditions check
├── Alert generation
```

---

### Phase 3: Execution Layer (Week 3-4)

#### 3.1 Order Management
```
src/execution/orders.py
├── Market order execution
├── Limit order with timeout
├── Order status tracking
├── Retry logic
└── Slippage logging
```

#### 3.2 Scheduler
```
src/execution/scheduler.py
├── Morning scan (9:15 AM)
├── Mid-day check (1:00 PM)
├── EOD positions check (3:00 PM)
├── Weekly rebalance (Saturday)
```

#### 3.3 Alerts
```
src/execution/alerts.py
├── Telegram bot integration
├── Trade notifications
├── Daily P&L summary
└── Risk alerts
```

---

### Phase 4: Dashboard & Monitoring (Week 4-5)

#### 4.1 Flask Web UI
```
src/app/
├── routes.py
├── templates/
│   ├── dashboard.html
│   ├── positions.html
│   ├── trades.html
│   └── settings.html
└── static/
    ├── css/
    └── js/
```

#### 4.2 Dashboard Features
- Current positions with Greeks
- P&L tracking (daily/weekly/monthly)
- IV percentile chart
- Trade history
- Signal log

---

### Phase 5: Backtesting (Week 5-6)

#### 5.1 Backtest Engine
```
src/backtest/
├── engine.py
├── data_loader.py
├── performance.py
└── visualization.py
```

#### 5.2 Backtest Metrics
- Total return
- Sharpe ratio
- Max drawdown
- Win rate
- Average premium yield
- Comparison to buy-and-hold

---

## File Structure (Final)

```
Covered_Calls/
├── .env                          # API keys (gitignored)
├── .env.example                  # Template
├── requirements.txt              # Dependencies
├── README.md                     # Project overview
│
├── docs/
│   ├── APP_PRD.md
│   ├── DEV-CLOCK.md
│   ├── PROJECT-STATUS.md
│   ├── GLOSSARY.md
│   ├── DEVELOPMENT-PLAN.md       # This file
│   └── Design/
│       └── STRATEGY-DESIGN.md    # Strategy specification
│
├── data/
│   ├── instruments.csv           # Daily instrument dump
│   ├── positions.db              # SQLite database
│   └── iv_history/               # Historical IV data
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── kite_client.py        # Kite Connect wrapper
│   │   ├── nse_data.py           # NSE bhav copy parser
│   │   └── database.py           # SQLite operations
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── greeks.py             # Black-Scholes Greeks
│   │   ├── iv_percentile.py      # IV percentile tracker
│   │   └── performance.py        # P&L calculations
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── strike_selector.py    # Strike selection logic
│   │   ├── position_manager.py   # Position management
│   │   ├── signals.py            # Signal generation
│   │   └── covered_call.py       # Main strategy class
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── orders.py             # Order management
│   │   ├── scheduler.py          # APScheduler jobs
│   │   └── alerts.py             # Telegram notifications
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py             # Backtest engine
│   │   ├── data_loader.py        # Historical data
│   │   └── visualization.py      # Charts
│   │
│   └── app/
│       ├── __init__.py
│       ├── routes.py             # Flask routes
│       ├── templates/            # HTML templates
│       └── static/               # CSS/JS
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_greeks_validation.ipynb
│   ├── 03_backtest_analysis.ipynb
│   └── 04_live_monitoring.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_greeks.py
│   ├── test_strike_selector.py
│   └── test_backtest.py
│
└── scripts/
    ├── run_daily_scan.py
    ├── run_backtest.py
    └── setup_database.py
```

---

## Dependencies (requirements.txt)

```
# Core
kiteconnect==5.0.1
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0

# Greeks calculation
py_vollib>=1.0.1
mibian>=0.1.3

# Database
sqlalchemy>=2.0.0

# Web Framework
Flask>=2.3.0
Flask-Session>=0.5.0

# Scheduling
APScheduler>=3.10.0

# Visualization
matplotlib>=3.7.0
plotly>=5.15.0

# Telegram Bot
python-telegram-bot>=20.0

# Environment
python-dotenv>=1.0.0

# Data
yfinance>=0.2.28
requests>=2.28.0

# Testing
pytest>=7.0.0
```

---

## Quick Start Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Initialize database
python scripts/setup_database.py

# Run daily scan (manual)
python scripts/run_daily_scan.py

# Start dashboard
flask --app src.app run

# Run backtest
python scripts/run_backtest.py --start 2023-01-01 --end 2024-12-31
```

---

## Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Data layer complete | Kite integration working |
| 2 | Greeks calculator | IV percentile tracking |
| 3 | Strategy engine | Signal generation |
| 4 | Execution layer | Paper trading ready |
| 5 | Dashboard | Web UI live |
| 6 | Backtest | 2-year backtest results |
| 7 | Paper trading | 4 weeks simulated |
| 8 | Go live | First real trade |

---

## Risk Checklist

- [ ] API keys in .env (not committed)
- [ ] Rate limiting implemented
- [ ] Error handling for API failures
- [ ] Position size limits enforced
- [ ] Stop-loss logic implemented
- [ ] Margin check before orders
- [ ] F&O ban list checked daily
- [ ] Corporate action handling
- [ ] Expiry day handling (close before)

---

**Next Action:** Start with Phase 1.1 - Kite Connect Integration
