<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-Web-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Chart.js-Viz-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" alt="Chart.js" />
  <img src="https://img.shields.io/badge/License-Private-red?style=for-the-badge" alt="License" />
</p>

<h1 align="center">
  <span style="color:#888">{ </span>Quantify<strong style="color:#22c55e">d</strong><span style="color:#888"> }</span>
</h1>

<h3 align="center">From Backtest To Bank</h3>

<p align="center">
  <em>Data-driven quant strategies for Indian markets.</em><br />
  20-year backtested. Live on Zerodha. 40% XIRR.
</p>

<p align="center">
  <a href="#features">Features</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#tech-stack">Tech Stack</a> &middot;
  <a href="#strategies">Strategies</a> &middot;
  <a href="#project-structure">Project Structure</a>
</p>

<!-- LAUNCHPAD:START -->
```json
{
  "stage": "building",
  "progress": 60,
  "complexity": "F",
  "lastUpdated": "2026-02-22",
  "targetDate": null,
  "nextAction": "Deploy KC6 live trading on Railway",
  "blocker": null,
  "demoUrl": null,
  "techStack": ["Python", "Flask", "Pandas", "Chart.js", "SQLite"],
  "shipped": false,
  "linkedinPosted": false
}
```
<!-- LAUNCHPAD:END -->

---

## Why Quantifyd

Most retail investors in India chase tips, follow gut feelings, or buy whatever is trending on Twitter. The result is predictable: underperformance, panic selling, and missed compounding.

Quantifyd replaces opinion with evidence. Every strategy is backtested across 20 years of Nifty 500 data before a single rupee goes to market. The system runs live on Zerodha, executing trades with the same rules that survived 2008, COVID, and every correction in between.

- **Backtested, not guessed** -- Every rule is validated across 20 years of market data before deployment
- **Fully automated** -- KC6 mean reversion runs daily on Zerodha with zero manual intervention
- **Multi-strategy edge** -- Momentum core + tactical overlays produce ~32% CAGR combined
- **Risk-first design** -- Max drawdown capped under 30% across all strategy variants

---

## Features

| Feature | Description |
|---------|-------------|
| **MQ Core Engine** | Momentum + Quality stock selection from Nifty 500. ATH breakout entry, 20% trailing exit from peak. Semi-annual rebalance with sector caps. |
| **KC6 Live Trading** | Automated Keltner Channel mean reversion on Zerodha. 6 scheduled jobs daily, 10-point safety guardrails, paper/live toggle from dashboard. |
| **IPO Strategies** | Two IPO overlays -- Scalper (+6% target, 20-day hold) and Swing (+30% target, ATH breakout) -- both running 64%+ win rates. |
| **Smart Cash Parking** | Idle capital auto-deployed to NiftyBEES ETF or liquid debt fund based on the 200-day SMA regime filter. |
| **Dark Theme Dashboards** | Bootstrap 5 + Chart.js equity curves, drawdown charts, trade logs, and strategy analytics. One dashboard per strategy. |
| **Optimization Framework** | Sweep engine for parameter tuning: portfolio size, stop losses, exit rules, technical filters. Incremental CSV output, preloaded data, batch execution. |
| **Breakout V3 Overlay** | Consolidation breakout detection with volume confirmation. Tactical position sizing layered on top of MQ Core. |

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
    initial_capital=10_000_000,
    portfolio_size=20,
)

universe, price_data = MQBacktestEngine.preload_data(config)
engine = MQBacktestEngine(config, preloaded_universe=universe, preloaded_price_data=price_data)
result = engine.run()

print(f"CAGR: {result.cagr:.2f}%")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")
```

---

## Tech Stack

| Technology | Purpose | Link |
|------------|---------|------|
| Python 3.12 | Core language | [python.org](https://python.org/) |
| Flask | Web framework & API routes | [flask.palletsprojects.com](https://flask.palletsprojects.com/) |
| Pandas | Data manipulation & backtest logic | [pandas.pydata.org](https://pandas.pydata.org/) |
| NumPy | Numerical computing | [numpy.org](https://numpy.org/) |
| ta | Technical indicators (EMA, RSI, Keltner, SuperTrend) | [ta docs](https://technical-analysis-library-in-python.readthedocs.io/) |
| Chart.js | Dashboard equity curves & analytics charts | [chartjs.org](https://www.chartjs.org/) |
| SQLite | Market data cache & KC6 trading state | [sqlite.org](https://www.sqlite.org/) |
| Bootstrap 5 | Dark theme dashboard UI | [getbootstrap.com](https://getbootstrap.com/) |
| APScheduler | Cron jobs for KC6 daily trading cycle | [apscheduler docs](https://apscheduler.readthedocs.io/) |

---

## Strategies

### Performance Summary

| Strategy | Type | CAGR | Win Rate | Backtest Period | Status |
|----------|------|------|----------|-----------------|--------|
| **MQ Core (PS20)** | Momentum + Quality | ~29% | -- | 2005 -- 2025 | Backtested |
| **MQ Concentrated (PS10)** | Momentum + Quality | ~49% | -- | 2023 -- 2025 | Backtested |
| **KC6** | Mean Reversion | -- | 65% | 2005 -- 2025 | Live on Zerodha |
| **IPO Scalper** | Breakout | -- | 64% | 2005 -- 2025 | Backtested |
| **IPO Swing** | Breakout | -- | 64.5% | 2005 -- 2025 | Backtested |
| **Combined System** | MQ + Tactical Overlays | ~32% | -- | 2023 -- 2025 | Backtested |

### MQ Core

The flagship strategy. Ranks Nifty 500 stocks by a composite momentum + quality score, enters on all-time-high breakouts, and exits when a stock drops 20% from its peak since entry. Semi-annual rebalance in January and July. Sector caps prevent concentration risk.

Key finding from optimization: **concentration is the single biggest CAGR lever.** Reducing portfolio size from 30 to 10 stocks pushes CAGR from 32% to 49% with max drawdown still under 27%.

### KC6

A fully automated mean reversion system. Enters when price closes below the Keltner Channel (6-period, 1.3 ATR) lower band while above the 200-day SMA. Exits via standing SELL LIMIT at KC6 midline, or hard stops at 5% loss / 15% gain / 15-day max hold. A crash filter (universe ATR ratio >= 1.3x) blocks all entries during high-volatility regimes. 2,482 trades backtested. Profit factor 1.70.

### IPO Strategies

Two complementary overlays for recently listed stocks:

- **Scalper**: Targets +6% from entry with a 20-day hold window. Quick in, quick out.
- **Swing**: Targets +30% on ATH breakouts from IPO bases. Longer hold, larger payoff.

Both run 64%+ win rates across the full backtest period.

---

## Project Structure

```
Quantifyd/
├── app.py                          # Flask app, all dashboard routes, KC6 scheduled jobs
├── config.py                       # Configuration, KC6_DEFAULTS, DATA_DIR
├── requirements.txt                # Python dependencies
├── Procfile                        # Railway deployment (single worker)
├── railway.json                    # Railway cloud config
│
├── services/
│   ├── mq_backtest_engine.py       # MQ Core backtest engine with preload_data()
│   ├── mq_portfolio.py             # Portfolio, Position, MQBacktestConfig, Trade
│   ├── kc6_scanner.py              # KC6 signal computation & crash filter
│   ├── kc6_executor.py             # KC6 order execution, 10-point safety guardrails
│   ├── kc6_db.py                   # KC6 SQLite persistence layer
│   ├── consolidation_breakout.py   # Breakout V3 detection (SYSTEM_PRIMARY)
│   ├── combined_mq_v3_engine.py    # Combined MQ + V3 overlay engine
│   ├── ipo_strategy.py             # IPO Scalper & Swing strategies
│   ├── technical_indicators.py     # EMA, RSI, SuperTrend, MACD, ADX
│   └── tactical_pool.py            # Smart cash parking logic
│
├── templates/
│   ├── base.html                   # Base template (Bootstrap 5 dark theme)
│   ├── kc6_dashboard.html          # KC6 live trading dashboard
│   ├── breakout_v3_dashboard.html  # Breakout V3 analytics
│   ├── combined_mq_v3_dashboard.html
│   ├── ipo_strategy_report.html    # IPO strategy results
│   └── tactical_dashboard.html     # Tactical pool dashboard
│
├── backtest_data/                  # Historical price data & optimization results
├── data/                           # Runtime data (market_data.db, fundamentals cache)
├── docs/                           # Strategy research, optimization logs, handoff docs
└── verification_output/            # Trade verification & Pine Script exports
```

---

## Dashboards

The app serves multiple strategy-specific dashboards, all built on a Bootstrap 5 dark theme with Chart.js visualizations:

| Route | Dashboard | Contents |
|-------|-----------|----------|
| `/` | MQ Portfolio | Equity curve, drawdown chart, trade log, exit reason breakdown |
| `/kc6` | KC6 Live Trading | Positions, signals, orders, equity curve, kill switch, mode toggle |
| `/breakout-v3` | Breakout V3 | Consolidation scanner, breakout signals, volume analysis |
| `/combined` | Combined MQ+V3 | Blended strategy performance, tactical overlay impact |
| `/ipo` | IPO Strategies | Scalper and Swing results, win rates, P/L distribution |

---

## Deployment

The system is configured for Railway cloud deployment:

```bash
# Procfile runs the combined worker (Flask + APScheduler)
web: python _combined_worker.py
```

Required environment variables on Railway:

| Variable | Description |
|----------|-------------|
| `KITE_API_KEY` | Zerodha Kite Connect API key |
| `KITE_API_SECRET` | Kite Connect API secret |
| `KITE_REDIRECT_URL` | OAuth redirect URL |
| `FLASK_SECRET_KEY` | Flask session secret |
| `RAILWAY_VOLUME_MOUNT_PATH` | Persistent storage path (`/data`) |

---

## License

Private -- All rights reserved.

---

<p align="center">
  <strong>{ </strong>Quantify<strong>d }</strong>
</p>
<p align="center">
  <sub>From Backtest To Bank</sub>
</p>
<p align="center">
  <sub>Built by <a href="https://github.com/castroarun">Arun Castro</a></sub>
</p>
