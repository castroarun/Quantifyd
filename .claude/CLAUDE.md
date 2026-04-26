# Quantifyd - Project Instructions

## Project Overview

Indian stock market backtesting system: Momentum + Quality (MQ) portfolio strategy with Breakout V3 overlay. Flask app with Bootstrap 5 dark theme dashboards, Chart.js, SQLite.

**Tech stack:** Python 3.12, Flask, pandas, numpy, ta library, Chart.js (CDN)

## Jira

- Project key: **CC** (if needed)

---

## ALL NEW PAGES GO IN THE REACT APP AT `/app/*` — NOT JINJA

**Default convention as of 2026-04-26:** any new strategy page, dashboard, or
user-facing UI must be built inside the React SPA at `frontend/`, served at
`/app/<route>`. The legacy Jinja templates at root paths (`/orb`, `/nas`,
`/collar`, `/kc6`, `/strangle`, etc.) are *frozen* — do not add new ones.

**Stack & layout:**
- Source root: `frontend/src/`
- Pages: `frontend/src/pages/<Name>.tsx` + `<Name>.module.css` (CSS modules, one per page)
- Shared components: `frontend/src/components/{Avatar,Cards,Chip,DataTable,Layout,Sidebar,StatusDot,TopBar,Icons}/...`
- Routing: React Router v6 in `frontend/src/App.tsx`. Add new routes there.
- Backend mount: `app.py` serves the SPA at `/app/*` (catch-all to `index.html`); JSON APIs live at `/api/<strategy>/*`.
- Design language: **dark theme**, CSS modules, no Bootstrap, no Chart.js CDN.
  Match the existing `Nas.tsx` / `Orb.tsx` / `Nwv.tsx` patterns for cards,
  tables, status chips, and metric tiles. Re-use `MetricCard`, `StrategyCard`,
  `Chip`, `DataTable`, `StatusDot` rather than rolling new ones.
- Build: `cd frontend && npm run build` produces `frontend/dist/` which Flask
  serves under `/app/*`. Frontend-only changes do NOT require a backend
  restart (Flask serves new static files on next request; user just hard-refreshes).

**When migrating an existing Jinja page:**
1. Build the React equivalent under `frontend/src/pages/`.
2. Add the route in `App.tsx` and a sidebar entry in `Sidebar.tsx`.
3. Retire the old Jinja page by replacing its handler with
   `return redirect('/app/<route>', code=302)` (precedent: the old `/orb` →
   `/app/orb` redirect at app.py around line 6069).
4. Keep the JSON API endpoints (`/api/<strategy>/*`) unchanged so the React
   page can talk to the same backend without refactoring services.

**Reference pages to copy from:**
- `frontend/src/pages/Nas.tsx` — multi-variant strategy page (closest pattern
  for any new variant-based dashboard like Strangle, Collar v2).
- `frontend/src/pages/Orb.tsx` — single-strategy with rich tables + chart.
- `frontend/src/pages/Nwv.tsx` — recent build, uses the latest patterns.

**Standing migration debt:** `/collar`, `/kc6` were built in Jinja before
this convention. Migrate to `/app/collar`, `/app/kc6` when their roadmap
allows. (`/strangle` migrated to `/app/strangle` on 2026-04-26.)

---

## LIVE-STATUS MD CONVENTION (long-running tasks)

For any task that runs longer than ~5 minutes, spawns background processes,
or could be interrupted mid-flight (sweeps, backtests, deployments,
migrations, multi-step ops), maintain a **live-status MD file** that lets
the user resume independently if Claude crashes or context is lost.

**Naming:** never call it `PROGRESS.md` — use a name that signals the
*nature* of the running work, ending in `-STATUS.md`. Examples:

| Task type | File name |
|---|---|
| Backtest sweep | `SWEEP-STATUS.md` |
| Multi-step deployment | `DEPLOY-STATUS.md` |
| Schema migration | `MIGRATION-STATUS.md` |
| Forensic investigation | `FORENSIC-STATUS.md` |
| Long live-trading run | `RUN-STATUS.md` |

Keep the file in the same folder as the artifacts it tracks (e.g., the
research/ subfolder, or `docs/` for cross-cutting work).

**Required sections:**

1. **Goal + scope** — one paragraph: what we're doing, what success looks
   like, what universe / period / parameters.
2. **Plan** — the variants / steps / configurations in a table or list,
   including any cells already known to be skipped and why.
3. **Status** — per-task or per-cell state (RUNNING / COMPLETED / FAILED),
   with bash background process IDs, log paths, heartbeat file paths,
   and last-known progress line.
4. **Crash recovery** — full instructions for the *human* to resume
   without Claude:
   - How to check what finished (heartbeat / summary files)
   - How to check whether background processes are still alive
   - How to restart any missing/killed step (full commands)
   - How to aggregate partial results
   - Which files NOT to touch
5. **Final aggregation** — what artifacts get produced when everything
   completes, where to look, ranking criteria.

Update the file at every meaningful state transition (launch, per-step
progress, completion, failure) — not just at the very end. The file is
the authoritative source if Claude's context is unavailable.

---

## NO BACKEND RESTART DURING MARKET HOURS

NSE cash + F&O session: **09:15 – 15:30 IST, Mon–Fri**. During this window,
`sudo systemctl restart quantifyd` on the VPS is **prohibited**.

**Why**
- NAS executors (all 8 squeeze/916 variants currently in paper mode) hold
  intraday state in memory. A restart loses today's open legs, closed-today
  records, and daily P&L from the NAS page until the next entry cycle.
- ORB open positions are exchange-safe (SL-M orders survive restart), but
  in-memory state (OR levels, catchup bookkeeping) still hiccups.
- Gunicorn worker teardown on SIGTERM can leave SQLite WAL inconsistent
  mid-trade, with silent data loss on the rollback.

**What this means in practice**
- **Python / Flask / service changes** → deploy after 15:30 IST only.
- **Frontend-only changes** (`frontend/src/**/*.tsx`, `templates/*.html`,
  static assets) → safe any time. Pull on VPS without restart; Flask serves
  updated static files on the next request. Hard-refresh the browser to
  pick up a new bundle hash.
- **Config tweaks** (`config.py` constants like `ORB_DEFAULTS['risk_per_trade_pct']`)
  → technically requires restart to take effect. Queue for after-close
  unless the change is strictly needed before the next trading session.
- **Emergency exceptions** (prod bug actively losing money, stuck order,
  kill-switch needed): restart is acceptable — prefer `/api/<strategy>/kill-switch`
  first if available.

**Deployment cheatsheet** (during market):
```
# Frontend only — no restart, safe
git push && ssh vps 'cd /home/arun/quantifyd && git reset --hard origin/master'

# Backend — wait until after 15:30 IST
git push && ssh vps 'cd /home/arun/quantifyd && git reset --hard origin/master &&
                     sudo systemctl restart quantifyd'
```

---

## ACTIVE TASK: MQ Strategy Optimization

### Context

We are optimizing the MQ portfolio strategy to maximize CAGR while keeping MaxDD < 30%. Backtest period: 2023-01-01 to 2025-12-31, initial capital Rs.1 Crore, universe = Nifty 500 (375 symbols with data).

### Current Best Results

| Config | CAGR | Sharpe | MaxDD | Notes |
|--------|------|--------|-------|-------|
| PS30_HSL50_ATH20_EQ95 | 32.19% | 1.05 | 27.0% | Best PS30 baseline |
| PS30_TOPUP30_CD3_HSL50 | 32.24% | 1.05 | 27.07% | Topups + cooldown 3d |
| PS25_HSL50 | 34.48% | 1.10 | 25.81% | |
| PS20_HSL50 | 37.84% | 1.16 | 26.85% | |
| PS15_HSL50 | 38.45% | 1.16 | 28.07% | |
| PS10_SEC70_POS30_TOP30_BIM | **48.66%** | **1.30** | 26.35% | Best risk-adjusted |

### Key Findings (Don't Re-test These)

1. **HSL >= 50% all produce same result** — hard stop never fires when ATH20 exit is active
2. **Quality weights have ZERO effect** — same 30 stocks regardless of weight tweaks
3. **Fundamental filter variations have ZERO effect** — same 30 stocks pass all filters
4. **Concentration is the #1 CAGR lever** — PS3=65%, PS5=58%, PS10=49%, PS15=38%, PS30=32%
5. **ATH drawdown exit is the only exit that fires** — no hard stops triggered with HSL50
6. **Technical indicators HURT PS30 performance** — All tested (EMA, RSI, SuperTrend, MACD, ADX) reduce CAGR from 32% to 3-28% by blocking momentum entries and Darvas topups
7. **STREND_atr7_m3.0 is best risk-adjusted PS30**: 27.79% CAGR, 16.65% MaxDD, Calmar 1.67 (baseline: 27% MaxDD, Calmar 1.19)

### Pending Optimization Sweeps

Full details in: `docs/OPTIMIZATION-PICKUP.md`

| Priority | Sweep | Configs | Status | Script |
|----------|-------|---------|--------|--------|
| 1 | Technical Indicators | 27 | **17/27 done** | `run_agent3_technical_optimization.py` — No indicator beats baseline |
| 2 | Exit Rules | 16 | 0/16 done | `run_exit_optimization.py` (needs rewrite) |
| 3 | Rebalance Frequency | 24 | 0/24 done | Needs new script |
| 4 | Combined MQ+V3 | 15 | 0/15 done | `run_combined_optimization.py` |

### To Resume: Read `docs/OPTIMIZATION-PICKUP.md` First

That doc has:
- Exact halt points and what went wrong
- Fixed script patterns for each sweep
- Mandatory agent rules (below)

---

## MANDATORY RULES FOR OPTIMIZATION AGENTS

When spawning agents to run backtests, these rules are **non-negotiable**:

### 1. ALWAYS preload data

```python
from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

# Load once (takes ~30s)
universe, price_data = MQBacktestEngine.preload_data(MQBacktestConfig())

# Pass to EVERY engine instance
engine = MQBacktestEngine(config,
    preloaded_universe=universe,
    preloaded_price_data=price_data)
result = engine.run()
```

Without preloading: ~190s per config. With preloading: ~50s per config.

### 2. ALWAYS write CSV incrementally

```python
import csv, os

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_SWEEPNAME.csv')
FIELDNAMES = ['label','cagr','sharpe','sortino','max_drawdown','calmar',
              'total_trades','win_rate','final_value','total_return_pct','topups']

# Write header once at start
with open(OUTPUT_CSV, 'w', newline='') as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# After EACH config completes, append immediately:
with open(OUTPUT_CSV, 'a', newline='') as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
```

Never batch CSV writes at the end. Timeout = lost data.

### 3. Use correct base parameters

```python
# NOTE: Do NOT set debt_reserve_pct explicitly — default 0.20 is correct
# (it funds Darvas topups which drive CAGR from 26% to 32%)
base = dict(
    portfolio_size=30,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,           # NOT 0.20 or 0.30
    rebalance_ath_drawdown=0.20,   # NOT 0.10 or 0.15
)
```

### 4. Max 8 configs per bash call

- Each PS30 config takes ~50-60s with preloading
- Data loading takes ~30s
- Total per bash call: 30 + (8 x 55) = ~470s (under 600s timeout)
- For more configs, run multiple sequential bash calls

### 5. DO NOT waste agent turns

- Do NOT run `help()` on classes — read source files directly with Read tool
- Do NOT test individual imports — write the complete script, then run it
- Do NOT debug interactively — if script has errors, read the file, fix with Edit, re-run
- Write script using Write tool (not heredoc in bash)

### 6. Skip already-completed configs

```python
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['label'] for row in csv.DictReader(f)}
    print(f'Skipping {len(done)} already-completed configs')

for label, params in configs:
    if label in done:
        continue
    # ... run backtest
```

### 7. Print progress on every config

```python
print(f'[{i}/{total}] {label} ...', end='', flush=True)
# ... run ...
print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% Sharpe={row["sharpe"]:.2f}')
sys.stdout.flush()
```

### 8. Suppress logging

```python
import logging
logging.disable(logging.WARNING)
```

---

## Centralized Database Reference

### Primary: `backtest_data/market_data.db` (1.24 GB)

Table: `market_data_unified` — Columns: `id, symbol, timeframe, date, open, high, low, close, volume, created_at`
Index: `(symbol, timeframe, date)` composite

| Timeframe | Symbols | Date Range | Rows |
|-----------|---------|------------|------|
| day | 1,621 | 2000-2026 | 3.4M |
| 60minute | 93 | 2018-2025 | 1.2M |
| 5minute | 10 | 2018-2025 | 1.3M |
| 30minute | 49 | Sep-Nov 2025 | 24K |

**5-min stocks (only 10):** BHARTIARTL, HDFCBANK, HINDUNILVR, ICICIBANK, INFY, ITC, KOTAKBANK, RELIANCE, SBIN, TCS

### Download: `services/data_manager.py` > `CentralizedDataManager`

- Kite API, token at `backtest_data/access_token.json`
- 5-min chunk: 7 days/request, rate limit: 0.35s (3 req/sec)
- F&O universe: 86 stocks (`FNO_LOT_SIZES` dict, lines 42-127)

### Other DBs

| DB | Size | Purpose |
|----|------|---------|
| backtest_results.db | 708 KB | Runs, trades, equity curves |
| kc6_trading.db | 52 KB | KC6 live trading state |
| mq_agent.db | 40 KB | MQ agent runs |

---

## Key Files

| File | Purpose |
|------|---------|
| `services/mq_portfolio.py` | Portfolio, Position, MQBacktestConfig, Trade classes |
| `services/mq_backtest_engine.py` | MQBacktestEngine with `preload_data()` static method |
| `services/combined_mq_v3_engine.py` | Combined MQ + V3 overlay engine |
| `services/consolidation_breakout.py` | V3 breakout detection, SYSTEM_PRIMARY |
| `services/technical_indicators.py` | EMA, RSI, etc. |
| `app.py` | Flask app with all dashboard routes |
| `docs/OPTIMIZATION-PICKUP.md` | Full halt points + continuation guide |
| `verification_output/mq_ath_trailing.pine` | TradingView Pine Script for ATH trailing |
| `backtest_data/optimization_results.json` | Previous 1,062 optimization results |
| `optimization_focused_results.csv` | Focused batch results (3 rows) |

## MQBacktestEngine Key APIs

```python
# Static method - load data once, reuse across configs
universe, price_data = MQBacktestEngine.preload_data(config)

# Constructor with preloaded data
engine = MQBacktestEngine(config,
    preloaded_universe=universe,
    preloaded_price_data=price_data)

# Run backtest - returns BacktestResult dataclass
result = engine.run()

# BacktestResult fields:
# .cagr, .sharpe_ratio, .sortino_ratio, .max_drawdown, .calmar_ratio
# .total_trades, .win_rate, .avg_win_pct, .avg_loss_pct
# .final_value, .total_return_pct, .total_topups
# .equity_curve (dict), .trades (list), .exit_reason_counts (dict)
```

## MQBacktestConfig Key Parameters

```python
MQBacktestConfig(
    start_date='2023-01-01', end_date='2025-12-31',
    initial_capital=10_000_000,
    portfolio_size=30,              # Number of stocks to hold
    equity_allocation_pct=0.95,     # % in equity (rest in debt)
    hard_stop_loss=0.50,            # Exit if stock drops 50% from entry
    rebalance_ath_drawdown=0.20,    # Exit if stock drops 20% from peak since entry
    rebalance_months=[1, 7],        # Semi-annual rebalance
    max_sector_weight=0.25,         # Max 25% in one sector
    max_stocks_per_sector=6,
    max_position_size=0.10,         # Max 10% in one stock
    topup_pct_of_initial=0.20,      # Darvas topup = 20% of initial capital
    # Technical filters (all False by default):
    use_ema_entry, use_rsi_filter, use_supertrend, use_macd, use_adx, use_weekly_filter
)
```

---

## KC6 V2 Live Trading System

### What It Is

Fully automated **KC6 mean reversion live trading system** on Zerodha (Nifty 500). Paper trading mode by default, toggle to live from dashboard. Runs daily via APScheduler cron jobs.

### Strategy Rules

- **Entry**: Close < KC(6, 1.3 ATR) Lower AND Close > SMA(200)
- **Exit (primary)**: Standing SELL LIMIT at KC6 mid, placed each morning
- **Exit (SL)**: 5% stop loss
- **Exit (TP)**: 15% take profit
- **Exit (MaxHold)**: 15 days
- **Crash filter**: Universe ATR Ratio >= 1.3x blocks all new entries
- **Backtest**: 2,482 trades, 65% win rate, PF 1.70, +2,863% P/L over 20 years

### KC6 Files

| File | Purpose |
|------|---------|
| `services/kc6_db.py` | SQLite persistence (positions, trades, orders, daily state, equity curve) |
| `services/kc6_scanner.py` | Signal computation (indicators, crash filter, entries, exits, target prices) |
| `services/kc6_executor.py` | Order execution with 10-point safety guardrails, target order lifecycle |
| `templates/kc6_dashboard.html` | Dashboard with Chart.js equity curve, positions, signals, trades |
| `config.py` | `KC6_DEFAULTS` dict (~line 170), `DATA_DIR` Railway-aware |
| `app.py` | Routes `/kc6`, `/api/kc6/*` (~line 1795), 6 scheduled jobs (~line 1987) |

### Scheduled Jobs (Mon-Fri)

| Time | Job | What |
|------|-----|------|
| 9:20 AM | Position sync | Compare DB vs Kite holdings |
| 9:25 AM | Place targets | SELL LIMIT at today's KC6 mid |
| 12:30 PM | Midday check | Check if target orders filled |
| 3:15 PM | Exit check | Target fills + SL/TP/MaxHold |
| 3:20 PM | Entry scan | Crash filter + new entries |
| 3:25 PM | Verify orders | Check order fills/rejections |

### API Endpoints

`/kc6` (dashboard), `/api/kc6/state`, `/api/kc6/scan` (POST), `/api/kc6/scan/status/<id>`, `/api/kc6/kill-switch` (POST), `/api/kc6/trades`, `/api/kc6/orders`, `/api/kc6/equity-curve`, `/api/kc6/toggle-mode` (POST)

### Railway Cloud Deployment

Config ready: `Procfile` (single worker), `railway.json`, `DATA_DIR` reads `RAILWAY_VOLUME_MOUNT_PATH` env var. Needs: Railway project creation, volume at `/data`, env vars (KITE_API_KEY, KITE_API_SECRET, KITE_REDIRECT_URL, FLASK_SECRET_KEY, RAILWAY_VOLUME_MOUNT_PATH).

### Full Handoff Doc

For complete implementation details, code references, and next steps: `docs/KC6-SESSION-HANDOFF.md`
