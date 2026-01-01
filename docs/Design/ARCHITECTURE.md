# Technical Architecture: Covered Calls Backtester

**Version:** 1.0
**Last Updated:** 2025-12-30
**Author:** Software Architect
**Status:** Approved for Development

---

## 1. System Overview

### 1.1 High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                              COVERED CALLS BACKTESTER                              |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +-----------+     +------------------+     +-------------------+                 |
|  |  Browser  |<--->|   Flask Web App  |<--->|   SQLite Database |                 |
|  | (Client)  |     |   (app.py)       |     |   (backtest.db)   |                 |
|  +-----------+     +------------------+     +-------------------+                 |
|       ^                    |                         ^                            |
|       |                    v                         |                            |
|       |           +------------------+               |                            |
|       |           |   Services Layer |               |                            |
|       |           +------------------+               |                            |
|       |           |                  |               |                            |
|       |     +-----v------+   +-------v--------+      |                            |
|       |     | KiteService|   | BacktestEngine |------+                            |
|       |     +-----+------+   +-------+--------+                                   |
|       |           |                  |                                            |
|       |           v                  v                                            |
|       |     +-----------+   +------------------+                                  |
|       |     | Zerodha   |   | GreeksCalculator |                                  |
|       |     | Kite API  |   | (Black-Scholes)  |                                  |
|       |     +-----------+   +------------------+                                  |
|       |                                                                           |
|       +<------------- Jinja2 Templates + Bootstrap 5 + Chart.js ---------------+ |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### 1.2 Request Flow Diagram

```
User Action                      Server Processing                    Response
-----------                      -----------------                    --------
    |                                   |                                 |
    |  1. Configure Backtest            |                                 |
    +---------------------------------->|                                 |
    |                                   |                                 |
    |                    2. Validate Parameters                           |
    |                    3. Queue Background Task                         |
    |                                   |                                 |
    |  4. Return Task ID                |                                 |
    |<----------------------------------+                                 |
    |                                   |                                 |
    |  5. Poll /api/status              |                                 |
    +---------------------------------->|                                 |
    |                                   |                                 |
    |                    6. Check task_status dict                        |
    |                    7. Return progress %                             |
    |                                   |                                 |
    |  8. Progress Update               |                                 |
    |<----------------------------------+                                 |
    |                                   |                                 |
    |  [Background: BacktestEngine]     |                                 |
    |       - Load stock data           |                                 |
    |       - Calculate Greeks          |                                 |
    |       - Simulate trades           |                                 |
    |       - Calculate metrics         |                                 |
    |                                   |                                 |
    |  9. Task Complete                 |                                 |
    +---------------------------------->|                                 |
    |                                   |                                 |
    |                   10. Return full results                           |
    |<----------------------------------+                                 |
```

---

## 2. Component Architecture

### 2.1 Layer Diagram

```
+-----------------------------------------------------------------------------------+
|                                PRESENTATION LAYER                                  |
+-----------------------------------------------------------------------------------+
|  templates/                                                                       |
|  +----------------+  +----------------+  +-------------------+                    |
|  | base.html      |  | backtest.html  |  | data_management.  |                    |
|  | (Navbar, CSS)  |  | (Main UI)      |  | html              |                    |
|  +----------------+  +----------------+  +-------------------+                    |
|                                                                                   |
|  static/js/                                                                       |
|  +----------------+  +----------------+  +-------------------+                    |
|  | backtest.js    |  | charts.js      |  | data_manager.js   |                    |
|  | (Form logic)   |  | (Chart.js)     |  | (Download UI)     |                    |
|  +----------------+  +----------------+  +-------------------+                    |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                                APPLICATION LAYER                                   |
+-----------------------------------------------------------------------------------+
|  app.py (Flask Routes)                                                            |
|  +------------------+  +------------------+  +------------------+                  |
|  | /                |  | /backtest        |  | /data-management |                  |
|  | /login, /logout  |  | /api/backtest/*  |  | /api/data/*      |                  |
|  +------------------+  +------------------+  +------------------+                  |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                                SERVICES LAYER                                      |
+-----------------------------------------------------------------------------------+
|  services/                                                                        |
|  +---------------------+  +------------------------+  +----------------------+    |
|  | kite_service.py     |  | covered_call_service.py|  | data_manager.py      |    |
|  | - get_kite()        |  | - CoveredCallEngine    |  | - download_data()    |    |
|  | - save_access_token |  | - run_backtest()       |  | - load_data()        |    |
|  | - get_access_token  |  | - calculate_pnl()      |  | - get_date_range()   |    |
|  +---------------------+  +------------------------+  +----------------------+    |
|                                                                                   |
|  +---------------------+  +------------------------+  +----------------------+    |
|  | greeks_calculator.py|  | metrics_calculator.py  |  | report_generator.py  |    |
|  | - calculate_delta() |  | - sharpe_ratio()       |  | - generate_csv()     |    |
|  | - calculate_theta() |  | - max_drawdown()       |  | - generate_pdf()     |    |
|  | - option_price()    |  | - win_rate()           |  | - trade_log()        |    |
|  +---------------------+  +------------------------+  +----------------------+    |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                                DATA LAYER                                          |
+-----------------------------------------------------------------------------------+
|  backtest_data/                                                                   |
|  +---------------------+  +------------------------+                              |
|  | market_data.db      |  | backtest_results.db    |                              |
|  | - market_data_      |  | - backtest_runs        |                              |
|  |   unified           |  | - trade_log            |                              |
|  | - download_status   |  | - daily_equity         |                              |
|  +---------------------+  +------------------------+                              |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                              EXTERNAL SERVICES                                     |
+-----------------------------------------------------------------------------------+
|  +-----------------------+  +-----------------------+                             |
|  | Zerodha Kite Connect  |  | yfinance (Fallback)   |                             |
|  | - Historical data     |  | - Free stock data     |                             |
|  | - OAuth               |  | - No options data     |                             |
|  +-----------------------+  +-----------------------+                             |
+-----------------------------------------------------------------------------------+
```

### 2.2 Component Descriptions

| Component | Responsibility | Reuse Source |
|-----------|----------------|--------------|
| **app.py** | Flask routes, session management, task orchestration | Adapt from kiteconnect01/app.py |
| **kite_service.py** | Zerodha OAuth, token management | Direct copy from kiteconnect01 |
| **data_manager.py** | Centralized data download/storage | Adapt CentralizedDataManager |
| **covered_call_service.py** | Core backtest logic for covered calls | **New** - Primary development |
| **greeks_calculator.py** | Black-Scholes Greeks calculation | Direct copy from kiteconnect01 |
| **metrics_calculator.py** | Performance metrics (Sharpe, drawdown) | New, reference existing patterns |
| **report_generator.py** | CSV/PDF export functionality | Adapt from kiteconnect01 |

---

## 3. Data Flow

### 3.1 Backtest Execution Flow

```
+----------------+     +------------------+     +-------------------+
|   User Input   |---->|   Validation     |---->|   Queue Task      |
| - Stocks       |     | - Date range     |     | - APScheduler     |
| - Strategies   |     | - Stock exists   |     | - task_status{}   |
| - Date range   |     | - Strategy valid |     |                   |
+----------------+     +------------------+     +-------------------+
                                                        |
                                                        v
+------------------+     +------------------+     +-------------------+
|   Load Data      |<----|   Background     |     |   Poll Status     |
| - From SQLite    |     |   Worker         |     | - GET /api/status |
| - Date filter    |     |                  |     | - Returns %       |
+------------------+     +------------------+     +-------------------+
        |
        v
+------------------+     +------------------+     +-------------------+
|  For Each Stock  |---->| Generate Options |---->|  Simulate Trade   |
|  For Each Month  |     | - Find expiry    |     | - Entry/Exit      |
|                  |     | - Calc premium   |     | - Track P&L       |
|                  |     | - Calc Greeks    |     | - Apply rules     |
+------------------+     +------------------+     +-------------------+
        |
        v
+------------------+     +------------------+     +-------------------+
| Aggregate Results|---->| Calculate Metrics|---->|  Store Results    |
| - By strategy    |     | - Sharpe ratio   |     | - backtest_runs   |
| - By stock       |     | - Drawdown       |     | - trade_log       |
|                  |     | - Win rate       |     | - daily_equity    |
+------------------+     +------------------+     +-------------------+
        |
        v
+------------------+
| Return to Client |
| - Metrics JSON   |
| - Chart data     |
| - Trade log      |
+------------------+
```

### 3.2 Option Premium Estimation Flow

```
Input: Stock price, Strike, DTE, Volatility
                    |
                    v
+-------------------+     +----------------------+
| Calculate IV      |---->| Black-Scholes Model  |
| - ATR proxy       |     | - d1, d2 calculation |
| - Historical vol  |     | - N(d1), N(d2)       |
+-------------------+     +----------------------+
                                    |
                                    v
+-------------------+     +----------------------+
| Calculate Greeks  |<----|  Option Price        |
| - Delta           |     |  (Theoretical)       |
| - Theta           |     |                      |
| - Gamma, Vega     |     |                      |
+-------------------+     +----------------------+
        |
        v
+-------------------+
| Apply Adjustments |
| - Liquidity       |
| - Bid-ask spread  |
| - Min premium     |
+-------------------+
        |
        v
Output: Estimated premium, Delta, Theta
```

---

## 4. Database Schema

### 4.1 Entity Relationship Diagram

```
+------------------+         +-------------------+
|  market_data     |         |  backtest_runs    |
+------------------+         +-------------------+
| id (PK)          |         | id (PK)           |
| symbol           |    +----| created_at        |
| timeframe        |    |    | config_json       |
| date             |    |    | total_return      |
| open             |    |    | sharpe_ratio      |
| high             |    |    | max_drawdown      |
| low              |    |    | win_rate          |
| close            |    |    | total_trades      |
| volume           |    |    +-------------------+
+------------------+    |            |
                        |            | 1:N
                        |            v
+------------------+    |    +-------------------+
| download_status  |    |    |  trade_log        |
+------------------+    |    +-------------------+
| id (PK)          |    |    | id (PK)           |
| symbol           |    +----| backtest_id (FK)  |
| timeframe        |         | symbol            |
| start_date       |         | entry_date        |
| end_date         |         | exit_date         |
| total_candles    |         | entry_price       |
| last_updated     |         | exit_price        |
+------------------+         | strike            |
                             | premium_received  |
                             | pnl               |
                             | exit_reason       |
                             +-------------------+
                                     |
                                     | 1:N
                                     v
                             +-------------------+
                             |  daily_equity     |
                             +-------------------+
                             | id (PK)           |
                             | backtest_id (FK)  |
                             | date              |
                             | portfolio_value   |
                             | drawdown          |
                             +-------------------+
```

### 4.2 Table Definitions

```sql
-- Market data (reuse from kiteconnect01 pattern)
CREATE TABLE IF NOT EXISTS market_data_unified (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    date DATETIME NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, date)
);

-- Backtest run metadata
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    name TEXT,
    config_json TEXT NOT NULL,
    -- Summary metrics
    total_return REAL,
    premium_yield REAL,
    win_rate REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    assignment_rate REAL,
    vs_buy_hold REAL,
    total_trades INTEGER,
    profitable_trades INTEGER,
    losing_trades INTEGER
);

-- Individual trade records
CREATE TABLE IF NOT EXISTS trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    -- Entry details
    entry_date DATE NOT NULL,
    stock_entry_price REAL NOT NULL,
    strike_price REAL NOT NULL,
    premium_received REAL NOT NULL,
    delta_at_entry REAL,
    dte_at_entry INTEGER,
    -- Exit details
    exit_date DATE,
    stock_exit_price REAL,
    option_exit_price REAL,
    exit_reason TEXT,  -- 'EXPIRY', 'ASSIGNED', 'PROFIT_TARGET', 'STOP_LOSS', 'EARLY_EXIT'
    -- P&L
    stock_pnl REAL,
    option_pnl REAL,
    total_pnl REAL,
    return_pct REAL,
    -- Strategy metadata
    strike_method TEXT,  -- 'DELTA_30', 'DELTA_40', 'OTM_2PCT', 'OTM_5PCT', 'ATM'
    exit_strategy TEXT,  -- 'HOLD_TO_EXPIRY', 'PROFIT_TARGET_50', 'STOP_LOSS_2X'
    FOREIGN KEY (backtest_id) REFERENCES backtest_runs(id)
);

-- Daily equity curve for charts
CREATE TABLE IF NOT EXISTS daily_equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    date DATE NOT NULL,
    portfolio_value REAL NOT NULL,
    daily_return REAL,
    cumulative_return REAL,
    drawdown REAL,
    rolling_sharpe REAL,
    FOREIGN KEY (backtest_id) REFERENCES backtest_runs(id),
    UNIQUE(backtest_id, date)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_market_symbol_tf ON market_data_unified(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_market_date ON market_data_unified(date);
CREATE INDEX IF NOT EXISTS idx_trade_backtest ON trade_log(backtest_id);
CREATE INDEX IF NOT EXISTS idx_equity_backtest ON daily_equity(backtest_id);
```

---

## 5. API Endpoints

### 5.1 Authentication Routes

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Landing page, login status | No |
| GET | `/login` | Redirect to Zerodha OAuth | No |
| GET | `/zerodha/callback` | OAuth callback handler | No |
| GET | `/logout` | Clear session, redirect | Yes |

### 5.2 Backtest Routes

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| GET | `/backtest` | Main backtest page | - |
| POST | `/api/backtest/run` | Start backtest | JSON config |
| GET | `/api/backtest/status` | Get task progress | - |
| GET | `/api/backtest/results/<id>` | Get results by ID | - |
| GET | `/api/backtest/history` | List past backtests | - |
| GET | `/api/backtest/export/<id>/<format>` | Export CSV/PDF | - |

### 5.3 Data Management Routes

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| GET | `/data-management` | Data download page | - |
| POST | `/api/data/download` | Start data download | symbols[], dates |
| GET | `/api/data/status` | Download progress | - |
| GET | `/api/data/available` | List available data | - |
| DELETE | `/api/data/<symbol>` | Delete symbol data | - |

### 5.4 API Request/Response Examples

**Start Backtest Request:**
```json
POST /api/backtest/run
{
  "stocks": ["RELIANCE", "TCS", "HDFCBANK"],
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "strategies": {
    "strike_selection": ["DELTA_30", "OTM_2PCT", "ATM"],
    "exit_rules": ["HOLD_TO_EXPIRY", "PROFIT_TARGET_50"]
  },
  "position_size": 1,
  "dte_range": [30, 45]
}
```

**Backtest Results Response:**
```json
{
  "id": 42,
  "status": "completed",
  "metrics": {
    "total_return": 24.5,
    "premium_yield": 8.2,
    "win_rate": 72.3,
    "max_drawdown": -12.4,
    "sharpe_ratio": 1.45,
    "assignment_rate": 28.5,
    "vs_buy_hold": 5.2
  },
  "strategy_comparison": [
    {"strategy": "DELTA_30 + HOLD_TO_EXPIRY", "return": 22.1, "sharpe": 1.32},
    {"strategy": "OTM_2PCT + PROFIT_TARGET_50", "return": 26.8, "sharpe": 1.58}
  ],
  "chart_data": {
    "equity_curve": [...],
    "monthly_returns": [...],
    "drawdown": [...]
  },
  "trade_count": 156,
  "period": "2020-01-01 to 2024-12-31"
}
```

---

## 6. File/Folder Structure

```
Covered_Calls/
├── app.py                          # Flask application entry point
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore patterns
│
├── services/                       # Business logic layer
│   ├── __init__.py
│   ├── kite_service.py             # Zerodha API (copy from kiteconnect01)
│   ├── data_manager.py             # Centralized data handling
│   ├── covered_call_service.py     # Core backtest engine (NEW)
│   ├── greeks_calculator.py        # Black-Scholes Greeks (copy)
│   ├── metrics_calculator.py       # Performance metrics (NEW)
│   └── report_generator.py         # CSV/PDF export (adapt)
│
├── templates/                      # Jinja2 HTML templates
│   ├── base.html                   # Base layout with navbar
│   ├── index.html                  # Landing page
│   ├── backtest.html               # Main backtest interface
│   ├── data_management.html        # Data download page
│   └── components/                 # Reusable template parts
│       ├── _metric_card.html
│       ├── _progress_bar.html
│       └── _chart_container.html
│
├── static/                         # Static assets
│   ├── css/
│   │   └── styles.css              # Custom styles
│   └── js/
│       ├── backtest.js             # Backtest page logic
│       ├── charts.js               # Chart.js configurations
│       └── data_manager.js         # Data download logic
│
├── backtest_data/                  # Data storage (gitignored)
│   ├── market_data.db              # SQLite - stock OHLCV
│   └── backtest_results.db         # SQLite - backtest results
│
├── docs/                           # Documentation
│   ├── DEV-CLOCK.md                # Time tracking
│   ├── PROJECT-STATUS.md           # 9-step workflow
│   ├── GLOSSARY.md                 # Domain terminology
│   └── Design/                     # Architecture docs
│       ├── APP_PRD.md              # Requirements (approved)
│       ├── ARCHITECTURE.md         # This document
│       ├── implementation-workflow.drawio
│       └── mockups/
│           └── backtest-dashboard.html
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_greeks.py
│   ├── test_backtest_engine.py
│   └── test_metrics.py
│
└── scripts/                        # Utility scripts
    ├── migrate_data.py             # Data migration helpers
    └── seed_sample_data.py         # Development seed data
```

---

## 7. Technology Decisions

### 7.1 Technology Stack Summary

| Layer | Technology | Version | Justification |
|-------|------------|---------|---------------|
| **Backend** | Python | 3.10+ | Existing kiteconnect01 stack |
| **Web Framework** | Flask | 2.x | Matches existing project, simple |
| **Template Engine** | Jinja2 | 3.x | Flask default, proven |
| **Session** | Flask-Session | - | File-based, no Redis needed |
| **Task Queue** | APScheduler | - | Background tasks, no Celery |
| **Database** | SQLite | 3.x | Single file, no server setup |
| **Data Processing** | Pandas | 2.x | Industry standard for finance |
| **Math/Stats** | NumPy, SciPy | - | Greeks calculation |
| **CSS Framework** | Bootstrap | 5.3.x | Existing pattern |
| **Charts** | Chart.js | 4.x | Existing pattern |
| **Tables** | DataTables | 1.13.x | Existing pattern |
| **API Client** | kiteconnect | - | Official Zerodha SDK |

### 7.2 Key Trade-off Decisions

| Decision | Chosen Option | Alternative | Rationale |
|----------|---------------|-------------|-----------|
| **Database** | SQLite | PostgreSQL | Simpler local dev, sufficient for backtesting |
| **Task Queue** | APScheduler | Celery + Redis | No infrastructure overhead, proven in kiteconnect01 |
| **Frontend** | Jinja2 + Bootstrap | React SPA | Faster dev, matches existing pattern, no build step |
| **Options Data** | Estimated (Black-Scholes) | Historical API | Historical options data unavailable from Zerodha |
| **Charts** | Chart.js | Plotly, Highcharts | Already in use, lightweight, free |
| **State Mgmt** | Server session | JWT/localStorage | Simpler, matches existing pattern |

### 7.3 Security Considerations

| Concern | Mitigation |
|---------|------------|
| API Keys | Store in `.env`, never commit |
| Session Tokens | HTTP-only cookies, short TTL |
| SQL Injection | Parameterized queries via SQLAlchemy/sqlite3 |
| XSS | Jinja2 auto-escaping enabled |
| CSRF | Flask-WTF CSRF protection |

---

## 8. Implementation Tasks

### 8.1 Task Breakdown with Complexity

| ID | Task | Description | Complexity | Est. Hours | Dependencies |
|----|------|-------------|------------|------------|--------------|
| **Phase 1: Foundation** |
| T01 | Project Setup | Create folder structure, requirements.txt, .env | Low | 1 | - |
| T02 | Copy Reusable Code | kite_service.py, greeks_calculator.py | Low | 1 | T01 |
| T03 | Adapt Data Manager | Modify CentralizedDataManager for this project | Medium | 3 | T02 |
| T04 | Database Schema | Create SQLite tables for backtest results | Low | 2 | T01 |
| **Phase 2: Core Engine** |
| T05 | Option Premium Estimator | Black-Scholes pricing with ATR volatility | Medium | 4 | T02, T04 |
| T06 | Strike Selection Logic | Delta-based, % OTM, ATM methods | Medium | 4 | T05 |
| T07 | Covered Call Entry Logic | Monthly cycle detection, position entry | Medium | 4 | T06 |
| T08 | Exit Rules Engine | Hold, profit target, stop loss | Medium | 4 | T07 |
| T09 | P&L Calculator | Per-trade and portfolio P&L | Medium | 3 | T08 |
| T10 | Metrics Calculator | Sharpe, drawdown, win rate, etc. | Medium | 3 | T09 |
| **Phase 3: Web Interface** |
| T11 | Flask App Setup | Routes, session, APScheduler | Low | 2 | T04 |
| T12 | Base Template | Navbar, layout, Bootstrap setup | Low | 2 | T11 |
| T13 | Data Management Page | Download UI, status tracking | Medium | 4 | T03, T12 |
| T14 | Backtest Config Form | Stock select, date range, strategies | Medium | 4 | T12 |
| T15 | Results Dashboard | Metrics cards, strategy comparison | High | 6 | T10, T14 |
| T16 | Chart Integration | 10 visualizations with Chart.js | High | 8 | T15 |
| **Phase 4: Export & Polish** |
| T17 | CSV Export | Trade log, metrics, equity curve | Low | 2 | T15 |
| T18 | PDF Export | Summary report with charts | Medium | 4 | T16, T17 |
| T19 | Error Handling | User-friendly error messages | Medium | 3 | T11-T18 |
| T20 | Testing | Unit tests for core logic | Medium | 4 | T05-T10 |
| T21 | Documentation | README, setup guide | Low | 2 | T20 |

### 8.2 Dependency Graph

```
T01 (Project Setup)
 |
 +---> T02 (Copy Reusable) ---> T03 (Data Manager)
 |                                    |
 +---> T04 (DB Schema) <--------------+
              |
              v
        T05 (Premium Estimator)
              |
              v
        T06 (Strike Selection)
              |
              v
        T07 (Entry Logic)
              |
              v
        T08 (Exit Rules)
              |
              v
        T09 (P&L Calculator)
              |
              v
        T10 (Metrics) --------+
              |               |
              v               v
        T11 (Flask) ---> T12 (Base Template)
              |               |
              +-------+-------+
                      |
              +-------+-------+
              |               |
              v               v
        T13 (Data Mgmt)  T14 (Config Form)
              |               |
              +-------+-------+
                      |
                      v
                T15 (Results Dashboard)
                      |
                      v
                T16 (Charts) -----+
                      |           |
              +-------+-------+   |
              |               |   |
              v               v   v
        T17 (CSV)       T18 (PDF Export)
              |               |
              +-------+-------+
                      |
                      v
                T19 (Error Handling)
                      |
                      v
                T20 (Testing)
                      |
                      v
                T21 (Documentation)
```

### 8.3 Complexity Definitions

| Level | Description | Typical Hours | Risk |
|-------|-------------|---------------|------|
| **Low** | Copy/adapt existing code, simple logic | 1-2 | Minimal |
| **Medium** | New logic with known patterns | 3-4 | Moderate |
| **High** | Complex new features, UI integration | 5-8 | Higher |

### 8.4 Estimated Timeline

| Phase | Tasks | Duration | Cumulative |
|-------|-------|----------|------------|
| Phase 1: Foundation | T01-T04 | 1 day | Day 1 |
| Phase 2: Core Engine | T05-T10 | 3 days | Day 4 |
| Phase 3: Web Interface | T11-T16 | 4 days | Day 8 |
| Phase 4: Export & Polish | T17-T21 | 2 days | Day 10 |

**Total Estimated: 10 working days (2 weeks)**

---

## 9. Reusable Code Reference

### 9.1 Direct Copy (Minimal Changes)

| File | Source | Changes Needed |
|------|--------|----------------|
| `kite_service.py` | `quantflow/kiteconnect01/services/` | Update TOKEN_FILE path |
| `greeks_calculator.py` | `quantflow/kiteconnect01/utils/` | None |

### 9.2 Adapt/Modify

| File | Source | Modifications |
|------|--------|---------------|
| `data_manager.py` | `centralized_data_manager.py` | Remove unused timeframes, add stock-specific helpers |
| `report_generator.py` | Various report utils in kiteconnect01 | Adapt for covered call report format |
| `base.html` | `nifty_backtest.html` | Simplify navbar, update branding |

### 9.3 New Development

| File | Reference Pattern | Notes |
|------|-------------------|-------|
| `covered_call_service.py` | `iron_condor_backtest.py` | Similar structure, different strategy logic |
| `metrics_calculator.py` | Various metric calcs in kiteconnect01 | Consolidate into single service |
| `backtest.html` | `nifty_backtest.html` | Similar form/results pattern |

---

## 10. Appendix

### 10.1 Indian Market Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Risk-free rate | 7% (0.07) | India benchmark |
| Trading days/year | 252 | NSE trading calendar |
| Market hours | 9:15 AM - 3:30 PM IST | NSE timings |
| Monthly expiry | Last Thursday | NSE F&O expiry |
| Lot sizes | Stock-specific | RELIANCE: 250, TCS: 150, etc. |

### 10.2 Covered Call Lot Sizes (Reference)

| Stock | Lot Size | Approximate Margin |
|-------|----------|-------------------|
| RELIANCE | 250 | ~3L |
| TCS | 150 | ~6L |
| HDFCBANK | 550 | ~4L |
| INFY | 300 | ~5L |
| ICICIBANK | 700 | ~4L |
| KOTAKBANK | 400 | ~4L |

### 10.3 Glossary Cross-Reference

See `docs/GLOSSARY.md` for domain-specific terminology used in this architecture.

---

**Document Approval:**

- [x] Architecture reviewed
- [x] Technology decisions justified
- [x] Implementation tasks defined
- [x] Dependencies mapped
- [ ] Ready for development

---

**End of Architecture Document**
