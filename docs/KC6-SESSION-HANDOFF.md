# KC6 V2 Live Trading System - Session Handoff

> **Last updated**: 2026-02-13
> **Status**: Core implementation complete, ready for Railway deployment + paper trading

---

## What Was Built

A fully automated **KC6 V2 mean reversion live trading system** on Zerodha for Indian stocks (Nifty 500). Runs daily via APScheduler cron jobs. Paper trading mode first, toggle to live from dashboard.

### Strategy Rules (from `sl_sweep_v2.py` backtest)

| Rule | Detail |
|------|--------|
| **Entry** | Close < KC(6, 1.3 ATR) Lower AND Close > SMA(200) |
| **Exit (primary)** | Standing SELL LIMIT at KC6 mid, placed each morning |
| **Exit (SL)** | 5% stop loss |
| **Exit (TP)** | 15% take profit |
| **Exit (MaxHold)** | 15 days max holding period |
| **Crash filter** | Universe ATR Ratio >= 1.3x blocks all new entries |
| **Order type** | CNC (delivery) equity, LIMIT orders |
| **Universe** | Nifty 500 |

### Backtest Results (20 years, Nifty 500)
- 2,482 trades, 65% win rate, profit factor 1.70, +2,863% total P/L
- Crash filter blocks 6.8% of trades but improves total P/L by +351%

---

## Files Created/Modified

### New Files

| File | Purpose | Lines |
|------|---------|-------|
| `services/kc6_db.py` | SQLite trading state persistence (positions, trades, orders, daily state, equity curve) | ~500 |
| `services/kc6_scanner.py` | Signal computation engine (indicators, crash filter, entry/exit scanning, target prices) | ~460 |
| `services/kc6_executor.py` | Order placement with 10-point safety guardrails, target limit order lifecycle | ~790 |
| `templates/kc6_dashboard.html` | Live dashboard with Chart.js equity curve, positions, signals, trades | ~620 |

### Modified Files

| File | What Changed |
|------|-------------|
| `config.py` | Added `KC6_DEFAULTS` dict (~30 lines). Updated `DATA_DIR` to support Railway volume via `RAILWAY_VOLUME_MOUNT_PATH` env var |
| `app.py` | Added KC6 routes (`/kc6`, `/api/kc6/*`), 6 scheduled jobs, scan task runner (~200 lines after line 1791) |
| `Procfile` | Updated to `--workers 1 --threads 4` (single worker for APScheduler) |
| `railway.json` | Added healthcheck, updated start command |

---

## Architecture

### Database (`backtest_data/kc6_trading.db`)

4 tables:
- `kc6_positions` - Active/closed positions (includes target_order_id, target_order_price, kc6_mid_today)
- `kc6_trades` - Completed trade history (entry/exit prices, pnl, hold_days, exit_reason)
- `kc6_orders` - Full order audit log (paper + live, BUY + SELL, all statuses)
- `kc6_daily_state` - Daily snapshot (ATR ratio, crash filter status, positions count, daily P/L)

### Scheduled Jobs (APScheduler cron, Mon-Fri)

| Time | Function | What |
|------|----------|------|
| 9:20 AM | `_kc6_position_sync` | Sync DB positions with Kite holdings |
| 9:25 AM | `_kc6_place_targets` | Cancel stale targets, place SELL LIMIT at today's KC6 mid |
| 12:30 PM | `_kc6_midday_fill_check` | Check if any target orders filled during morning |
| 3:15 PM | `_kc6_check_exits` | Check target fills + SL/TP/MaxHold for remaining positions |
| 3:20 PM | `_kc6_full_scan` | Compute crash filter + scan for new entries + execute |
| 3:25 PM | `_kc6_verify_orders` | Verify order fills/rejections via Kite API |

### API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/kc6` | Dashboard page |
| GET | `/api/kc6/state` | Positions, signals, stats, config |
| POST | `/api/kc6/scan` | Trigger manual scan (async) |
| GET | `/api/kc6/scan/status/<id>` | Poll scan progress |
| POST | `/api/kc6/kill-switch` | Emergency exit all positions |
| GET | `/api/kc6/trades` | Trade history |
| GET | `/api/kc6/orders` | Order audit log |
| GET | `/api/kc6/equity-curve` | Cumulative P/L data for chart |
| POST | `/api/kc6/toggle-mode` | Switch paper/live |

### Dashboard Features

- Paper/Live mode badge + crash filter status badge
- 6 metric cards (ATR ratio, positions, win rate, profit factor, total trades, total P/L)
- **Equity curve chart** (Chart.js) with per-trade tooltips showing symbol, P/L%, exit reason
- Active positions table with target order status (LIMIT SET / NO ORDER)
- Entry signals, exit signals, recent trades tables
- Manual Scan, Kill Switch, Toggle Mode, Refresh buttons
- Auto-refresh every 60 seconds

### Safety Guardrails (10-point pre-order check)

1. `live_trading_enabled` is True
2. `paper_trading_mode` is False
3. Kite API authenticated
4. Market hours (9:15 AM - 3:30 PM IST, weekdays)
5. Daily order count < max_daily_orders (5)
6. Daily loss < max_daily_loss_pct (3%)
7. Active positions < max_positions (5)
8. Universe ATR ratio < threshold (crash filter not active)
9. Symbol in Nifty 500 universe
10. No duplicate position

### Paper Trading Mode

- **ON by default** (`paper_trading_mode: True, live_trading_enabled: False`)
- Full pipeline runs: data loading, indicator computation, signal scanning, order logging
- Orders logged to `kc6_orders` with status `PAPER` or `PAPER_TARGET`
- Positions created/closed normally in DB
- Trade records created in `kc6_trades` with full P/L
- Target fill simulation: checks if day's high > target price
- Equity curve chart renders from completed trades
- No Kite API calls made

---

## Railway Cloud Deployment

### Already configured:
- `Procfile` - gunicorn, single worker, 4 threads
- `railway.json` - Nixpacks builder, healthcheck, restart policy
- `config.py` - DATA_DIR reads `RAILWAY_VOLUME_MOUNT_PATH` env var
- `kc6_db.py` - Uses `DATA_DIR` from config (persists on Railway volume)
- `requirements.txt` - All dependencies listed

### Deployment Steps:

1. **Create Railway project** → Deploy from GitHub
2. **Add Volume**: Service → Settings → Volumes → mount at `/data`
3. **Set env vars**:
   ```
   KITE_API_KEY=<your_key>
   KITE_API_SECRET=<your_secret>
   KITE_REDIRECT_URL=https://<app>.railway.app/zerodha/callback
   FLASK_SECRET_KEY=<random_string>
   RAILWAY_VOLUME_MOUNT_PATH=/data
   ```
4. **Update Zerodha App** redirect URL at developers.kite.trade
5. Push to GitHub → Railway auto-deploys

### Important Notes:
- Volume at `/data` persists: KC6 trading DB, market data DB, Kite access token
- Single worker prevents APScheduler from duplicating cron jobs
- Flask sessions are ephemeral (user re-logs after redeploy - acceptable)
- Market data DB (`market_data.db`) needs to exist on the volume for paper mode scanning. Either:
  - Copy it manually via `railway shell` + upload, or
  - Use Kite API (live mode) which fetches data directly

---

## Key Code References

| What | File:Line | Notes |
|------|-----------|-------|
| KC6_DEFAULTS config | `config.py:~170` | All 16 strategy parameters |
| DATA_DIR Railway logic | `config.py:~15` | Checks RAILWAY_VOLUME_MOUNT_PATH |
| DB singleton | `services/kc6_db.py:499` | `get_kc6_db()` |
| Equity curve query | `services/kc6_db.py:465` | `get_equity_curve()` |
| Indicator functions | `services/kc6_scanner.py:1-80` | ema, sma, atr_series, keltner |
| Universe ATR ratio | `services/kc6_scanner.py` | `compute_universe_atr_ratio()` |
| Entry scanning | `services/kc6_scanner.py` | `scan_entries()` |
| Exit scanning | `services/kc6_scanner.py` | `scan_exits()` |
| Target price computation | `services/kc6_scanner.py` | `compute_target_prices()` |
| 10-point guardrails | `services/kc6_executor.py:35` | `_check_guardrails()` |
| Paper entry order | `services/kc6_executor.py:156` | `place_entry_order()` paper branch |
| Paper exit order | `services/kc6_executor.py:248` | `place_exit_order()` paper branch |
| Target order lifecycle | `services/kc6_executor.py:462-687` | cancel_stale, place, check_fills |
| Pipeline functions | `services/kc6_executor.py:694` | run_place_targets, run_exit_check, run_entry_scan |
| KC6 routes | `app.py:~1795` | All /kc6 and /api/kc6/* routes |
| KC6 scheduled jobs | `app.py:~1987` | All 6 cron job functions + registration |
| Chart.js equity curve | `templates/kc6_dashboard.html:~549` | renderEquityCurve() |

---

## What's NOT Done Yet / Next Steps

1. **Deploy to Railway** - Config is ready, user needs to create the project on railway.app and set env vars
2. **Market data on Railway** - The paper mode scanner reads from local `market_data.db`. Options:
   - Upload the DB to Railway volume via `railway shell`
   - Or switch to Kite API data loading (requires Kite auth on Railway)
3. **Paper trading verification** - Run for a few days in paper mode, verify trades log correctly and equity curve renders
4. **Live mode activation** - Only after paper trading validates, toggle from dashboard
5. **Potential enhancements**:
   - Email/Telegram notifications on trade execution
   - Drawdown tracking and max drawdown metric
   - Position sizing based on account equity (not fixed capital)
   - Multi-timeframe confirmation signals

---

## Reusable Code Sources

| What | Source | Used In |
|------|--------|---------|
| Indicator functions (ema, sma, atr, keltner) | `crash_filter_v3.py:28-41` | `kc6_scanner.py` |
| Universe ATR ratio computation | `crash_filter_v3.py:54-60, 78` | `kc6_scanner.py` |
| DB singleton pattern | `services/mq_agent_db.py:33-47` | `kc6_db.py` |
| Kite API access | `services/kite_service.py:170-178` | `kc6_executor.py` via `get_kite()` |
| Scheduler pattern | `app.py:1765-1788` | KC6 job registration |
| Dashboard template | `templates/mq_dashboard.html` | `kc6_dashboard.html` layout |
| Nifty 500 universe | `services/nifty500_universe.py` | `kc6_executor.py` guardrail #9 |

---

## Schema Migration Note

If the DB already exists from an earlier version without target order columns, the migration block in `kc6_db.py:124-132` auto-adds `kc6_mid_today`, `target_order_id`, `target_order_price` columns via ALTER TABLE.
