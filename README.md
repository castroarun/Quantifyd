<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/Chart.js-4.x-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" alt="Chart.js" />
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/Zerodha-Kite_API-FF6600?style=for-the-badge" alt="Zerodha" />
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" alt="License" />
</p>

<br />

<p align="center">
  <code style="font-size: 1.2em; letter-spacing: 2px; color: #10b981;">{ QUANT }</code>
</p>

<h1 align="center">
  Quantify<strong>d</strong>
</h1>

<h3 align="center">From Backtest To Bank</h3>

<p align="center">
  <em>Systematic quant strategies for Indian markets -- backtested across 20 years, live on Zerodha.</em><br />
  Rs. 1 Crore &rarr; Rs. 158 Crore. Same market. Just data, just discipline.
</p>

<p align="center">
  <a href="#why-quantifyd">Why</a> &middot;
  <a href="#strategies">Strategies</a> &middot;
  <a href="#performance">Performance</a> &middot;
  <a href="#dashboards">Dashboards</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#tech-stack">Tech Stack</a> &middot;
  <a href="#project-structure">Structure</a>
</p>

<!-- LAUNCHPAD:START -->
```json
{
  "stage": "live",
  "progress": 85,
  "complexity": "F",
  "lastUpdated": "2026-02-22",
  "targetDate": null,
  "nextAction": "Deploy KC6 live trading on Railway",
  "blocker": null,
  "demoUrl": null,
  "techStack": ["Python", "Flask", "Chart.js", "SQLite", "Zerodha Kite API"],
  "shipped": true,
  "linkedinPosted": true
}
```
<!-- LAUNCHPAD:END -->

---

## Why Quantifyd

Most retail investors in India chase tips, follow gut feelings, or buy whatever is trending on Twitter. The result is predictable: underperformance, panic selling, and missed compounding.

**Quantifyd replaces opinion with evidence.** Every strategy is backtested across 20 years of Nifty 500 data before a single rupee goes to market. The system runs live on Zerodha, executing trades with the same rules that survived the 2008 Global Financial Crisis, COVID, and every correction in between.

- **Backtested, not guessed** -- Every rule is validated across 20 years of market data before deployment
- **Fully automated** -- KC6 mean reversion runs daily on Zerodha with zero manual intervention
- **Multi-strategy edge** -- Momentum core + tactical overlays produce ~32% CAGR combined
- **Risk-first design** -- Max drawdown capped under 30% across all strategy variants
- **Live performance** -- 40% XIRR on equity portfolio (Zerodha verified)

---

## Strategies

Quantifyd runs three independent, uncorrelated systems. When one is flat, another is firing.

### 1. MQ Core -- Momentum + Quality (60% allocation)

The flagship strategy. Ranks all Nifty 500 stocks by a composite momentum + quality score, enters on all-time-high breakouts, and exits mechanically when a stock drops 20% from its peak since entry.

**Entry criteria:**
- Stock at all-time high breakout
- Revenue and profit growth: positive AND accelerating
- Debt-to-Equity ratio &le; 0.20
- ROE and ROCE &ge; 15%
- Operating margins better than peers and/or rising

**How it works:**
- Semi-annual rebalance in January and July
- Sector caps prevent concentration risk (max 25% per sector, max 6 stocks per sector)
- Darvas box topups on winners -- strength gets more capital, not less
- 20% ATH drawdown exit triggers automatic capital redeployment to the next winner

**Key finding from optimization:** Concentration is the single biggest CAGR lever. Reducing portfolio size from 30 to 10 stocks pushes CAGR from 32% to 49% with max drawdown still under 27%.

### 2. KC6 -- Mean Reversion Live Trading System (20% allocation)

A fully automated mean reversion system running live on Zerodha via Kite Connect API. Six scheduled jobs run Monday through Friday with zero manual intervention.

**Strategy rules:**
- **Entry:** Close < Keltner Channel (6-period, 1.3 ATR) Lower Band AND Close > SMA(200)
- **Exit (primary):** Standing SELL LIMIT order at KC6 midline, placed each morning
- **Exit (stop loss):** 5% hard stop
- **Exit (take profit):** 15% target
- **Exit (max hold):** 15 calendar days
- **Crash filter:** Universe ATR Ratio &ge; 1.3x blocks all new entries during high-volatility regimes

**Scheduled daily cycle (IST):**

| Time | Job | Action |
|------|-----|--------|
| 9:20 AM | Position sync | Reconcile DB with Kite holdings |
| 9:25 AM | Place targets | SELL LIMIT at today's KC6 midline |
| 12:30 PM | Midday check | Verify target order fills |
| 3:15 PM | Exit check | SL / TP / MaxHold evaluations |
| 3:20 PM | Entry scan | Crash filter + scan for new entries |
| 3:25 PM | Verify orders | Confirm fills and handle rejections |

**Safety:** 10-point guardrail system, paper trading mode by default with one-click toggle to live from the dashboard.

### 3. Tactical Capital Pool

The combined allocation framework that ties all strategies together:

| Sleeve | Allocation | Strategy | Status |
|--------|-----------|----------|--------|
| MQ Core | 60% | Momentum + Quality, ATH breakouts | Backtested |
| KC6 | 20% | Keltner Channel mean reversion | Live on Zerodha |
| IPO Scalper | 5% | +6% target, 20-day hold, 64% win rate | Backtested |
| IPO Swing | 5% | +30% target, ATH breakout from IPO base, 64.5% win rate | Backtested |
| Smart Cash | 10% | NiftyBEES ETF or liquid debt (200-SMA regime filter) | Backtested |

Idle capital is never idle -- the Smart Cash sleeve auto-deploys to NiftyBEES ETF above the 200-day SMA or parks in liquid debt funds below it.

---

## Performance

### MQ Core vs Nifty 50

| Metric | MQ System | Nifty 50 |
|--------|-----------|----------|
| CAGR (3-year, 2023-2025) | **~36%** | ~13% |
| CAGR (20-year, 2005-2025) | **~29%** | ~13% |
| Sharpe Ratio | **1.16** | ~0.6 |
| Max Drawdown | **26.85%** | 39% (COVID) |
| Rs. 1 Crore after 3 years | **Rs. 2.62 Crore** | Rs. 1.44 Crore |
| Rs. 1 Crore after 20 years | **Rs. 158 Crore** | Rs. 11.5 Crore |

That is a **14x difference** over 20 years. Same capital, same market, same timeframe.

### KC6 Mean Reversion (20-year backtest)

| Metric | Value |
|--------|-------|
| Total trades | 2,482 |
| Win rate | 65% |
| Profit factor | 1.70 |
| Cumulative P/L | +2,863% |

### During COVID (March 2020)

Nifty 50 fell **39%**. The MQ system fell **34%** -- trailing exits removed stocks mechanically and shuttled capital into debt funds and ETFs before the full damage hit.

### All Strategies Combined

| Metric | Value |
|--------|-------|
| Combined CAGR | ~32% |
| Total backtested trades | 2,900+ across all strategies |
| Backtest period | 2005 -- 2025 (20 years) |
| Live XIRR | 40% (Zerodha verified) |

---

## Dashboards

All dashboards are built on a Bootstrap 5 dark theme with Chart.js equity curves, drawdown charts, trade logs, and strategy analytics.

| Route | Dashboard | What You See |
|-------|-----------|-------------|
| `/` | Home | Landing page, login status, system overview |
| `/agent` | MQ Dashboard | Equity curve, drawdown chart, trade log, exit reason breakdown, signal scanner |
| `/model-portfolio` | Model Portfolio | Live holdings, sector allocation, position sizing, rebalance status |
| `/kc6` | KC6 Live Trading | Open positions, signals, orders, equity curve, kill switch, paper/live toggle |
| `/tactical` | Tactical Pool | Combined allocation view, sleeve performance, capital deployment |
| `/breakout-v3` | Breakout V3 | Consolidation scanner, breakout signals, volume analysis |
| `/combined` | Combined MQ+V3 | Blended strategy performance, tactical overlay impact |
| `/ipo-research` | IPO Strategies | Scalper and Swing results, win rates, P/L distribution |
| `/crash-recovery` | Crash Recovery | Drawdown analysis, recovery timelines, regime detection |
| `/backtest` | Backtest Lab | Configure and run custom backtests with parameter sweeps |

---

## Quick Start

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/castroarun/Quantifyd.git
cd Quantifyd

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The dashboard will be available at `http://localhost:5000`.

### Running a Backtest

```python
from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

config = MQBacktestConfig(
    start_date='2005-01-01',
    end_date='2025-12-31',
    initial_capital=10_000_000,   # Rs. 1 Crore
    portfolio_size=20,
)

# Preload data once (~30s), then run multiple configs in ~50s each
universe, price_data = MQBacktestEngine.preload_data(config)
engine = MQBacktestEngine(config, preloaded_universe=universe, preloaded_price_data=price_data)
result = engine.run()

print(f"CAGR: {result.cagr:.2f}%")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")
print(f"Total Trades: {result.total_trades}")
print(f"Final Value: Rs. {result.final_value:,.0f}")
```

### KC6 Live Trading Setup

KC6 requires a Zerodha Kite Connect subscription and API credentials. Set the following environment variables:

```bash
export KITE_API_KEY="your_api_key"
export KITE_API_SECRET="your_api_secret"
export KITE_REDIRECT_URL="your_redirect_url"
export FLASK_SECRET_KEY="your_secret_key"
```

The system starts in **paper trading mode** by default. Toggle to live from the `/kc6` dashboard.

---

## Tech Stack

| Technology | Purpose | Link |
|------------|---------|------|
| Python 3.12 | Core language, backtest engines, strategy logic | [python.org](https://python.org/) |
| Flask | Web framework, API routes, dashboard serving | [flask.palletsprojects.com](https://flask.palletsprojects.com/) |
| pandas | Data manipulation, time series, backtest computations | [pandas.pydata.org](https://pandas.pydata.org/) |
| NumPy | Numerical computing, statistical calculations | [numpy.org](https://numpy.org/) |
| ta | Technical indicators (EMA, RSI, Keltner Channel, SuperTrend, MACD, ADX) | [ta docs](https://technical-analysis-library-in-python.readthedocs.io/) |
| Chart.js | Dashboard equity curves, drawdown charts, analytics visualizations | [chartjs.org](https://www.chartjs.org/) |
| Bootstrap 5 | Dark theme responsive dashboard UI | [getbootstrap.com](https://getbootstrap.com/) |
| SQLite | Market data cache (`market_data.db`), KC6 trading state persistence | [sqlite.org](https://www.sqlite.org/) |
| APScheduler | Cron jobs for KC6 daily trading cycle (6 scheduled jobs) | [apscheduler docs](https://apscheduler.readthedocs.io/) |
| Zerodha Kite API | Live order execution, position management, market data | [kite.trade](https://kite.trade/) |
| Flask-SocketIO | Real-time WebSocket updates for backtest progress | [flask-socketio](https://flask-socketio.readthedocs.io/) |

---

## Project Structure

```
Quantifyd/
|-- app.py                          # Flask app: all routes, KC6 scheduler, 40+ endpoints
|-- config.py                       # Configuration: MQ_DEFAULTS, KC6_DEFAULTS, DATA_DIR
|-- requirements.txt                # Python dependencies
|-- Procfile                        # Railway deployment (single worker)
|-- railway.json                    # Railway cloud configuration
|
|-- services/
|   |-- mq_backtest_engine.py       # MQ Core backtest engine with preload_data()
|   |-- mq_portfolio.py             # Portfolio, Position, MQBacktestConfig, Trade classes
|   |-- kc6_scanner.py              # KC6 signal computation, indicators, crash filter
|   |-- kc6_executor.py             # KC6 order execution, 10-point safety guardrails
|   |-- kc6_db.py                   # KC6 SQLite persistence (positions, trades, orders, equity)
|   |-- consolidation_breakout.py   # Breakout V3 detection engine (SYSTEM_PRIMARY)
|   |-- combined_mq_v3_engine.py    # Combined MQ + Breakout V3 overlay engine
|   |-- ipo_strategy.py             # IPO Scalper and Swing strategy engines
|   |-- strategy_backtest.py        # Generic strategy backtesting framework
|   |-- technical_indicators.py     # EMA, RSI, SuperTrend, MACD, ADX, Keltner Channel
|   |-- tactical_pool.py            # Smart cash parking and allocation logic
|   `-- kc6_backtest_engine.py      # KC6 historical backtest engine
|
|-- templates/
|   |-- base.html                   # Base template: Bootstrap 5 dark theme, nav, Chart.js
|   |-- kc6_dashboard.html          # KC6 live trading: positions, signals, kill switch
|   |-- breakout_v3_dashboard.html  # Breakout V3 scanner and analytics
|   |-- combined_mq_v3_dashboard.html  # Combined strategy dashboard
|   |-- ipo_strategy_report.html    # IPO Scalper + Swing results
|   |-- tactical_dashboard.html     # Tactical pool allocation view
|   `-- crash_recovery.html         # Crash drawdown analysis and recovery
|
|-- backtest_data/                  # Historical price data & optimization results (1,062 configs)
|-- data/                           # Runtime data: market_data.db, fundamentals cache
|-- docs/                           # Strategy research, optimization logs, session handoffs
`-- verification_output/            # Trade verification exports & TradingView Pine Scripts
```

---

## Optimization Framework

Quantifyd includes a parameter sweep engine that has explored **1,062+ configurations** across portfolio size, stop losses, exit rules, technical filters, and rebalance frequencies.

Key findings from the optimization program:

| Discovery | Implication |
|-----------|-------------|
| Concentration is the #1 CAGR lever | PS3=65%, PS5=58%, PS10=49%, PS15=38%, PS30=32% CAGR |
| Technical indicators hurt momentum | All tested indicators (EMA, RSI, SuperTrend, MACD, ADX) reduced CAGR by blocking entries |
| ATH trailing exit is the only exit that fires | Hard stops never trigger when 20% ATH drawdown exit is active |
| Quality weight changes have zero effect | The same top stocks pass all reasonable filter variations |
| Darvas topups drive CAGR from 26% to 32% | The 20% debt reserve that funds topups is critical infrastructure |

---

## Deployment

The system is configured for **Railway** cloud deployment with persistent storage:

```bash
# Procfile runs the combined worker (Flask + APScheduler)
web: python _combined_worker.py
```

Required environment variables:

| Variable | Description |
|----------|-------------|
| `KITE_API_KEY` | Zerodha Kite Connect API key |
| `KITE_API_SECRET` | Kite Connect API secret |
| `KITE_REDIRECT_URL` | OAuth redirect URL |
| `FLASK_SECRET_KEY` | Flask session secret |
| `RAILWAY_VOLUME_MOUNT_PATH` | Persistent storage mount path (`/data`) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <code>{ QUANT }</code>
</p>
<p align="center">
  <strong>Quantifyd</strong>
</p>
<p align="center">
  <sub>From Backtest To Bank</sub>
</p>
<p align="center">
  <sub>Built by <a href="https://github.com/castroarun">Arun Castro</a> | <a href="https://www.linkedin.com/in/aruncastro/">LinkedIn</a></sub>
</p>
