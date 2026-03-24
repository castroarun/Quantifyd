# Live Trading Architecture

> Quantifyd real-time execution infrastructure — connections, data flows, and the pattern for onboarding new live strategies.

---

## System Overview

```
                    Zerodha Kite Connect
                           |
          +----------------+----------------+
          |                |                |
     REST API        KiteTicker WS     Order Updates
   (historical,      (live ticks)     (on_order_update)
    orders, quotes)       |                |
          |               v                v
          |     +-------------------+      |
          +---->|  MaruthiTicker    |<-----+
                | (singleton)       |
                +-------------------+
                  |    |    |    |
           Candle  Hard  Pending  SocketIO
           Aggr.   SL    Trigger  Broadcast
             |     Check  Check   (dashboard)
             v       |      |
        +-------------------+
        | MaruthiExecutor   |
        | (singleton)       |
        +-------------------+
             |          |
        MaruthiDB    Kite Orders
        (SQLite)     (REST API)
```

---

## Connection Pool

### 1. KiteTicker WebSocket (Single Connection)

**File:** `services/maruthi_ticker.py` > `MaruthiTicker`

One persistent WebSocket connection carries ALL tick data:

| Token | Instrument | Consumer |
|-------|-----------|----------|
| Auto-resolved | MARUTI (NSE equity) | MaruthiTicker candle aggregation |
| 256265 | NIFTY 50 | NAS strategy (forwarded via `_forward_to_nas`) |
| 260105 | BANKNIFTY | BNF strategy (forwarded via `_forward_to_bnf`) |

**Callbacks registered:**
```python
kws.on_ticks          = _on_ticks           # Price data (binary)
kws.on_order_update   = _on_order_update    # Order status changes (text/JSON)
kws.on_connect        = _on_connect
kws.on_close          = _on_close
kws.on_error          = _on_error
kws.on_reconnect      = _on_reconnect
```

**Key property:** `on_order_update` is pushed by Zerodha — no polling needed. Fires instantly when any order is placed, modified, filled, cancelled, or rejected.

### 2. Flask-SocketIO (Dashboard Push)

**File:** `app.py` line ~80

```python
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
```

**Events emitted:**

| Event | Source | Payload | Frequency |
|-------|--------|---------|-----------|
| `maruthi_tick` | `MaruthiTicker._emit_tick()` | `{ltp, master_st, child_st, hard_sl, regime, master_dir, child_dir, candle}` | ~2/sec (throttled) |
| `price_tick` | `KiteTickerService` (covered calls) | `{symbol, ltp, ...}` | Per tick |
| `ticker_status` | On client connect | `{status: 'connected'}` | Once |

**Client connection:** Dashboard loads `socket.io.min.js` from CDN, connects automatically on page load.

### 3. Kite REST API

**File:** `services/kite_service.py`

Used for: historical data, order placement, order status, quotes, instruments, holdings.

**Rate limits:** 3 requests/sec for historical data, 10 req/sec for orders/quotes.

**Auth:** OAuth2 flow (`/login` → `/zerodha/callback`) + TOTP auto-login (`services/kite_auth.py`).

### 4. SQLite Databases (Per-Strategy)

Each strategy has its own SQLite DB with thread-safe access via `threading.Lock()`:

| Strategy | DB File | Singleton Accessor |
|----------|---------|-------------------|
| Maruthi | `backtest_data/maruthi_trading.db` | `get_maruthi_db()` |
| KC6 | `backtest_data/kc6_trading.db` | Via `KC6DB()` |
| BNF | `backtest_data/bnf_trading.db` | `get_bnf_db()` |
| NAS | `backtest_data/nas_trading.db` | `get_nas_db()` |

---

## Tick Data Flow (Maruthi)

```
KiteTicker WS
    │
    ▼
_on_ticks(ws, ticks)
    │
    ├─── process_tick() ──► CandleAggregator (30-min OHLCV)
    │                            │
    │                            └─► on_candle_close callback
    │                                    │
    │                                    ▼
    │                            executor.run_candle_check(df)
    │                              - compute SuperTrend
    │                              - detect signals
    │                              - place trigger orders
    │                              - trail hard SL
    │                              - update regime DB
    │
    ├─── _check_hard_sl_tick(ltp) ──► instant SL breach check
    │
    ├─── _check_pending_triggers(ltp) ──► paper mode fill simulation
    │
    ├─── _emit_tick(ltp) ──► SocketIO broadcast (throttled ~2/sec)
    │                            │
    │                            ▼
    │                      Dashboard (real-time proximity gauges)
    │
    ├─── _forward_to_nas(tick) ──► NAS ticker (NIFTY ticks)
    │
    └─── _forward_to_bnf(tick) ──► BNF executor (BANKNIFTY spot)
```

## Order Lifecycle

```
Signal detected (candle close)
    │
    ▼
execute_actions()
    ├─── Cancel opposite pending orders (reversal handling)
    ├─── Skip if same-direction already pending
    │
    ▼
place_futures_entry()
    ├─── Fetch futures candle from Kite historical API
    ├─── Compute trigger = futures candle low - 5 (SELL) or high + 5 (BUY)
    ├─── Compute limit = trigger ± 5 pts slippage
    ├─── Log order in DB (status: PENDING)
    ├─── Place SL-L order on Kite (live mode)
    └─── Create position in DB (status: PENDING)
              │
              ▼
    ┌─── Fill Detection ─────────────────────────┐
    │                                             │
    │  Paper Mode:                                │
    │    _check_pending_triggers() on every tick   │
    │    Simulates fill when LTP crosses trigger   │
    │                                             │
    │  Live Mode:                                 │
    │    on_order_update callback from KiteTicker  │
    │    Zerodha pushes COMPLETE/CANCELLED/REJECTED│
    │    Instant — no polling                      │
    │                                             │
    │  Fallback:                                  │
    │    run_verify_triggers() on candle close     │
    │    Polls kite.orders() as safety net         │
    └─────────────────────────────────────────────┘
              │
              ▼
    on_trigger_fill(position, fill_price)
        ├─── Activate position (PENDING → ACTIVE)
        ├─── Place protective option (PE for BULL, CE for BEAR)
        ├─── Recalculate hard SL from master ST + ATR
        └─── Dashboard updates automatically via SocketIO

    Cross-Day Persistence:
        Zerodha cancels unfilled SL-L at 3:30 PM EOD
        → re_place_pending_orders() at 9:16 AM re-places on Kite
        → Handles contract expiry roll if needed

    Gap Protection (9:16 AM + 9:21 AM):
        If market opens PAST the trigger price (gap through):
        1. re_place_pending_orders() detects gap → does NOT re-place on Kite
           - SELL trigger: LTP already below trigger = gap down
           - BUY trigger: LTP already above trigger = gap up
           - Position marked GAP_PENDING (not PENDING or ACTIVE)
        2. handle_gap_entry() at 9:21 AM (5 min confirmation):
           - Gap filled (price retraced)? → Cancel position
           - Signal reversed (regime changed)? → Cancel position
           - Gap holds + signal intact? → Enter at MARKET + full hedges:
             * Protective option (5% OTM insurance)
             * Short option (1 strike OTM premium income)
             * Hard SL from master ST + ATR
```

## Dashboard Real-Time Updates

### Live Bar Components

| Component | Data Source | Update Frequency |
|-----------|-----------|-----------------|
| Signal Wave (7-bar waveform) | SocketIO connection state | Continuous animation when ticks flowing |
| LTP | `maruthi_tick` SocketIO event | ~2/sec |
| Master/Child direction | `maruthi_tick` event (regime data) | ~2/sec |
| Proximity gauges (Master, Child, Hard SL) | Computed client-side from LTP + regime levels | ~2/sec |

### Proximity Gauge Logic

```
refRange = LTP × 1.5%  (reference scale)
distance = |LTP - ST_level|
fill%    = 100 - (distance / refRange × 100)  // fills UP as price approaches

Color:
  > 60% away  → normal (blue/purple/grey)
  30-60% away → amber warning
  < 30% away  → danger (orange for ST, red for SL)
```

### Signal Wave States

| State | CSS Class | Visual | Condition |
|-------|-----------|--------|-----------|
| Active | `signal-wave` | Green bars animating | Connected + ticks flowing + market hours |
| Stale | `signal-wave stale` | Amber slow pulse | No tick for 15+ seconds |
| Inactive | `signal-wave inactive` | Flat grey 2px bars | Ticker OFF or market closed |

### Polling Fallbacks

| Endpoint | Interval | Purpose |
|----------|----------|---------|
| `/api/maruthi/state` | 30s | Full state refresh (positions, stats, trades) |
| `/api/maruthi/ticker/status` | 30s | Connection status fallback |
| `/api/maruthi/mtm` | 10s | Mark-to-market P&L |

---

## Singleton Pattern

All core services use singletons to prevent state loss:

```python
# Executor — preserves _current_master_atr across calls
_executor_instance = None
def get_maruthi_executor(config=None) -> MaruthiExecutor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = MaruthiExecutor(config)
    return _executor_instance

# Ticker — one WebSocket connection
_maruthi_ticker = None
def get_maruthi_ticker(config=None) -> MaruthiTicker:
    ...

# DB — one connection manager
_maruthi_db = None
def get_maruthi_db() -> MaruthiDB:
    ...
```

**Why singletons matter:** Creating new instances resets computed state (e.g., `_current_master_atr` defaults to 100.0, causing incorrect hard SL calculations).

---

## Trailing Hard SL

```python
# Formula
buffer = hard_sl_atr_mult × ATR(7) on 30-min candles
BULL: SL = Master ST - buffer  (trails UP only)
BEAR: SL = Master ST + buffer  (trails DOWN only)

# Safety: SL must never cross to wrong side of Master ST
# If prev_hard_sl drifted past Master ST (e.g., ST reversed direction),
# the SL resets to master_st ± buffer
```

**Computed:** On every candle close in `run_candle_check()`.
**Checked:** On every tick in `_check_hard_sl_tick()`.
**On breach:** Closes ALL positions immediately, sets regime to FLAT.

---

## Scheduled Jobs (APScheduler)

| Time | Strategy | Job | Function |
|------|----------|-----|----------|
| 9:00 AM | Maruthi | TOTP auto-login + start ticker | `_maruthi_auto_login_and_start()` |
| 9:16 AM | Maruthi | Re-place pending orders | `_maruthi_re_place_pending()` |
| 9:20 AM | KC6 | Position sync | `_kc6_position_sync()` |
| 9:25 AM | KC6 | Place target orders | `_kc6_place_targets()` |
| 12:30 PM | KC6 | Midday check | `_kc6_midday_check()` |
| 3:00 PM | Maruthi | EOD protection | `_maruthi_eod_protection()` |
| 3:15 PM | KC6 | Exit check | `_kc6_exit_check()` |
| 3:15 PM | Maruthi | Roll check | `_maruthi_roll_check()` |
| 3:15 PM | BNF | Exit check | `_bnf_exit_check()` |
| 3:20 PM | KC6 | Entry scan | `_kc6_entry_scan()` |
| 3:20 PM | BNF | Daily scan | `_bnf_daily_scan()` |
| 3:25 PM | KC6 | Verify orders | `_kc6_verify_orders()` |
| 3:30 PM | Maruthi | Market close | `_maruthi_market_close()` |

---

## Onboarding a New Live Strategy

Follow this checklist to add a new strategy to the live trading system.

### 1. Service Layer (3 files)

```
services/
├── {strategy}_scanner.py     # Signal detection
├── {strategy}_executor.py    # Order execution + state management
└── {strategy}_db.py          # SQLite persistence
```

**Scanner** must implement:
- `scan()` → returns signals with entry/exit levels
- Use indicators from `services/technical_indicators.py` or compute inline

**Executor** must implement:
- Singleton pattern (`get_{strategy}_executor()`)
- `run_candle_check(df)` or `run_scan()` — main strategy logic
- `execute_actions(actions)` — order placement
- `get_state()` → dict for dashboard API
- `_check_guardrails()` — safety checks before orders
- Kill switch method to close all positions

**DB** must implement:
- Thread-safe SQLite with `threading.Lock()`
- Tables: `positions`, `trades`, `orders`, `signals`, `regime/state`, `settings`
- Singleton accessor: `get_{strategy}_db()`
- Auto-migration for schema changes

### 2. Configuration

Add `{STRATEGY}_DEFAULTS` dict to `config.py`:
```python
STRATEGY_DEFAULTS = {
    'symbol': '...',
    'exchange': 'NSE',
    'exchange_fo': 'NFO',
    'lot_size': ...,
    'capital': ...,
    'enabled': True,
    'paper_trading_mode': True,
    'live_trading_enabled': False,
    'max_daily_orders': ...,
}
```

### 3. API Endpoints (Standard Pattern)

Add to `app.py`:
```python
GET  /api/{strategy}/state
POST /api/{strategy}/scan
POST /api/{strategy}/kill-switch
GET  /api/{strategy}/trades
GET  /api/{strategy}/orders
GET  /api/{strategy}/signals
GET  /api/{strategy}/equity-curve
POST /api/{strategy}/toggle-mode
POST /api/{strategy}/toggle-enabled
```

### 4. Real-Time Data (If Needed)

**Option A: Piggyback on MaruthiTicker** (preferred for NSE instruments)
- Add instrument token to `_on_connect` subscription list
- Add forwarding method: `_forward_to_{strategy}(tick)`
- Call it from `_on_ticks`

**Option B: Own KiteTicker** (if different exchange or complex subscription)
- Create `services/{strategy}_ticker.py`
- Follow MaruthiTicker pattern (CandleAggregator, singleton, SocketIO emit)

### 5. SocketIO Broadcast (Dashboard Push)

Emit from ticker on each relevant tick:
```python
def _emit_tick(self, ltp):
    import time
    now = time.time()
    if now - self._last_emit_time < 0.5:
        return
    self._last_emit_time = now

    from app import socketio
    socketio.emit('{strategy}_tick', {
        'ltp': round(ltp, 1),
        'ts': now,
        # ... strategy-specific regime data
    })
```

Dashboard listens:
```javascript
socket.on('{strategy}_tick', function(d) { ... });
```

### 6. Order Fill Detection

**Paper mode:** Tick-level simulation in `_check_pending_triggers(ltp)`
**Live mode:** `on_order_update` callback from KiteTicker (instant, no polling)
**Fallback:** `run_verify_triggers()` on candle close

### 7. Scheduled Jobs

Register in `app.py` using APScheduler:
```python
scheduler.add_job(
    _strategy_job_function,
    CronTrigger(hour=H, minute=M, day_of_week='mon-fri'),
    id='strategy_job_name',
    replace_existing=True,
)
```

### 8. Dashboard

Create `templates/{strategy}_dashboard.html`:
- Extend `base.html`
- Include `socket.io.min.js` + `chart.js`
- Listen for `{strategy}_tick` SocketIO event
- Implement signal wave + proximity gauges (copy from Maruthi)
- Add route in `app.py`: `@app.route('/{strategy}')`

### 9. Boot Sequence

Add to `app.py` `__main__` block:
```python
if STRATEGY_DEFAULTS.get('enabled', True):
    ticker = get_{strategy}_ticker(STRATEGY_DEFAULTS)
    if not ticker.is_connected:
        ticker.start()
```

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Real-time push | SocketIO (not SSE) | Flask already uses SocketIO; SSE blocks threads in threading mode |
| Fill detection | `on_order_update` callback | Zero latency, no polling — Zerodha pushes via same WebSocket |
| Tick forwarding | Single KiteTicker, forward to strategies | Kite limits to ~3 concurrent WS connections |
| DB per strategy | Separate SQLite files | Isolation, no cross-contamination, easy backup |
| Singletons | Module-level `_instance` | Prevent state loss (ATR, SL) from re-instantiation |
| Trigger orders | SL-L (Stop Loss-Limit) | Sits dormant on exchange, fires without our intervention |
| Trigger buffer | 5 pts from candle H/L | Avoids whipsaws on exact-level touches |
| Futures candle | Fetched from Kite historical API | Trigger prices must match the instrument being traded |
| Cross-day orders | Morning re-placement at 9:16 AM | Zerodha cancels unfilled SL-L at 3:30 PM EOD |

---

*Last updated: 2026-03-23*
