# ORB + VWAP + RSI + CPR Paper Trading Module — Build Spec v2

**Version:** 2.0 (supersedes v1)
**Target app:** Existing NAS strategy dashboard (the one that already runs ATM/OTM strategy pages with Zerodha Kite auto-login and live prices)
**Goal:** Add a new strategy page that runs the ORB+VWAP+RSI+CPR strategy in **paper trading mode** during live market hours, on Nifty / Bank Nifty / Sensex simultaneously.
**Mode:** Paper trading only. No real broker orders. Virtual position book + MTM tracking.

**Companion file:** `ORB_VWAP_RSI_CPR_Filter_v2.1.pine` (Pine Script source of truth)

---

## What changed from v1

If you saw an earlier version of this PRD, the substantive changes are:

1. **CPR data fetch pattern explicitly specified.** The Pine Script v2.0 had a subtle lookahead bug where `request.security` with `[1]` indexing returned wrong CPR values on the first bar of new sessions. Python implementation MUST use the safe pattern documented in Section 2.4.
2. **Live data integration section expanded** with concrete patterns matching what the existing NAS app already does.
3. **Build sequence reordered** — Phase 1 now ends with running the engine against historical data and matching TradingView's signal count exactly, before any live integration.
4. **New Section 11** — debugging and observability requirements. The most expensive bugs in this kind of system come from data alignment issues that are invisible without good logging.
5. **New Section 14** — "How to talk to Arun during the build" — explicit instructions for what to ask vs decide alone.

---

## 1. Context for Claude Code

This module mirrors the logic of a TradingView Pine Script strategy that has been visually validated on Bank Nifty 5-min over the last 6 months. The Pine script uses Opening Range Breakout entries with three confirmation filters (VWAP direction, higher-TF RSI, CPR direction) and one regime filter (skip wide-CPR days).

The Python implementation here MUST produce **identical signals** to the Pine script for any given OHLCV+VWAP+CPR state. Treat the Pine Script (`ORB_VWAP_RSI_CPR_Filter_v2.1.pine`) as the source of truth for signal logic. If there's any doubt about logic, reference the Pine version.

**What's already built in the NAS app (do NOT rebuild):**
- Zerodha Kite Connect auto-login flow
- Live LTP / OHLC fetcher (whatever method the existing ATM/OTM strategy page uses — reuse the same)
- Existing UI framework (sidebar nav, page routing, design tokens — match the ATM/OTM page styling)
- SQLite database (extend with new tables, don't create separate DB)
- Telegram alerts pipeline (if it exists, hook into it)

**What you need to build:**
- New strategy page accessible from sidebar
- ORB signal generator with all four filters
- Virtual paper trading position book
- Live MTM tracking
- Strategy report panel (matches TradingView's strategy report metrics)

---

## 2. Strategy logic — exact specification

### 2.1 Instruments

Run three independent instances in parallel:
1. **NIFTY** — symbol `NIFTY 50` for spot price, lot size **75**
2. **BANKNIFTY** — symbol `NIFTY BANK` for spot price, lot size **35**
3. **SENSEX** — symbol `SENSEX` for spot price, lot size **20**

Each instance maintains its own state independently (own OR levels, own positions, own P&L). They share the same code but isolated runtime.

Lot sizes are configurable in case SEBI revises them again. Pull from a config table on startup, fall back to these defaults.

### 2.2 Time-based logic (IST timezone, hardcode `Asia/Kolkata`)

- **Session start:** 09:15:00 IST
- **OR window:** First 15 minutes (configurable: 15/30/60). Default = 15. So OR = high/low between 09:15:00 and 09:29:59.
- **OR finalized at:** 09:30:00 (no further OR updates after this)
- **Last entry time:** 14:00:00 (no new positions after this)
- **EOD square-off:** 15:20:00 (close all paper positions at 15:20 LTP)

### 2.3 Opening Range calculation

For each instrument, on 5-minute candles:
- Track high and low between 09:15 and (09:15 + OR_minutes)
- `OR_high = max(high)` over OR window
- `OR_low = min(low)` over OR window
- Both reset to `None` at the start of each new trading day

### 2.4 CPR calculation — CRITICAL: data fetch pattern

CPR uses the **previous completed trading day's HLC**. Calculation is done ONCE at session start (or at module start if mid-session), then values are FIXED for the whole day.

```
prev_high  = previous trading day's high (full-day candle)
prev_low   = previous trading day's low
prev_close = previous trading day's close

pivot   = (prev_high + prev_low + prev_close) / 3
bc      = (prev_high + prev_low) / 2
tc      = (pivot - bc) + pivot
width   = abs(tc - bc)
width_pct = (width / current_price) * 100
```

**MANDATORY data fetch pattern (lookahead-safe):**

```python
def fetch_previous_day_hlc(kite, instrument_token, today_date):
    """
    Fetches the most recent COMPLETED daily candle's HLC.
    Handles weekends, holidays, and special trading days correctly.
    
    DO NOT use date arithmetic like (today - 1 day) — this breaks on
    Mondays (returns Sunday, no data) and after holidays.
    """
    # Fetch last 7 calendar days of daily candles
    from_dt = today_date - timedelta(days=7)
    to_dt   = today_date - timedelta(days=1)  # explicitly exclude today
    
    candles = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_dt,
        to_date=to_dt,
        interval='day'
    )
    
    if not candles:
        raise ValueError(f"No daily candles found for {instrument_token} between {from_dt} and {to_dt}")
    
    # Most recent completed daily candle = last entry in the list
    prev_day = candles[-1]
    return prev_day['high'], prev_day['low'], prev_day['close']
```

**WHY THIS PATTERN MATTERS:**
- Naive `today - timedelta(days=1)` returns wrong date on Mondays and after holidays.
- Using today's developing daily candle as "previous day" leaks future data into your filter.
- The Pine Script equivalent had this exact bug in v2.0 — it caused CPR-aligned filters to evaluate against incorrect TC/BC values on the first bar of new sessions, leaking incorrectly-allowed trades into backtests. Don't repeat this mistake in Python.

CPR levels are FIXED for the whole trading day. Calculate once at 09:14:00 IST, store in `orb_daily_state` table, then read from cache for the rest of the day. Do NOT recalculate intraday.

### 2.5 VWAP

Standard intraday VWAP, resets at session start:
```
typical_price = (high + low + close) / 3
cumulative_pv = sum(typical_price * volume) since 09:15
cumulative_v  = sum(volume) since 09:15
vwap = cumulative_pv / cumulative_v
```

**For indices that don't have proper volume,** use the same VWAP method the existing ATM/OTM page uses. If indices don't trade with reliable volume:
- For NIFTY → use NIFTY-FUT (current month future) for VWAP calculation
- For BANKNIFTY → use BANKNIFTY-FUT
- For SENSEX → use SENSEX-FUT

The breakout signal is still evaluated on the spot index's OHLC, but the VWAP filter compares spot price against the futures-derived VWAP. This is how most professional intraday systems handle it.

### 2.6 RSI (higher timeframe)

- Timeframe: 15-minute (configurable)
- Length: 14 (configurable)
- Standard Wilder's RSI on close prices (use `pandas-ta` or `ta-lib` or roll your own — all produce same output if implemented correctly)
- `RSI_long_threshold = 60`, `RSI_short_threshold = 40` (both configurable)

**Edge case:** At 09:30:00 (when OR finalizes and signal evaluation begins), there may not be enough 15-min bars for a meaningful RSI(14). Two options:
- (Preferred) Seed from history: at startup, fetch last 30 days of 15-min candles via Kite historical API, prime the RSI calculator with that data, then stream new bars on top.
- (Fallback) Use yesterday's last 15-min RSI value as the seed for today's first calculation.

Document which approach you chose in code comments.

### 2.7 Filters (all independently toggleable in settings)

```python
# Filter 1: VWAP direction
vwap_long_ok  = (not use_vwap_filter) or (current_price > vwap)
vwap_short_ok = (not use_vwap_filter) or (current_price < vwap)

# Filter 2: RSI confirmation (uses 15-min RSI)
rsi_long_ok  = (not use_rsi_filter) or (rsi_15m > rsi_long_threshold)
rsi_short_ok = (not use_rsi_filter) or (rsi_15m < rsi_short_threshold)

# Filter 3: CPR direction
cpr_long_ok  = (not use_cpr_dir_filter) or (current_price > cpr_tc)
cpr_short_ok = (not use_cpr_dir_filter) or (current_price < cpr_bc)

# Filter 4: CPR width (skip wide-CPR days entirely)
is_wide_cpr  = cpr_width_pct > cpr_width_threshold  # default 0.5%
cpr_width_ok = (not use_cpr_width_filter) or (not is_wide_cpr)
```

### 2.8 Entry rules

```python
# Detect breakout (close-based, not intra-bar)
long_breakout  = or_finalized and (close > or_high) and (prev_close <= or_high)
short_breakout = or_finalized and (close < or_low)  and (prev_close >= or_low)

# Combined entry signal
long_signal = (
    allow_longs and long_breakout 
    and vwap_long_ok and rsi_long_ok and cpr_long_ok and cpr_width_ok
    and current_time < last_entry_time
    and trades_today < max_trades_per_day  # default 1
    and position_is_flat
)

short_signal = (similarly mirrored with shorts)
```

**Entry execution (paper):** Record virtual entry at the close price of the bar that triggered the breakout signal. Quantity = 1 lot of the instrument's lot size.

### 2.9 Exit rules

Three exit conditions, whichever hits first:

**A. Stop Loss** (configurable type):
- `OR Opposite` (default) — Long SL = OR_low, Short SL = OR_high
- `Fixed Points` — entry ± fixed_sl_points  
- `ATR Multiple` — entry ± (ATR_14 * atr_multiplier)

**B. Target** (configurable type):
- `R Multiple` (default 1.5R) — target = entry + (R * |entry - SL|)
- `Fixed Points` — entry ± fixed_target_points
- `OR Range Multiple` — target = entry + (or_range * multiplier)

**C. EOD square-off** at 15:20:00 IST — close at current LTP regardless of P&L.

For paper trades: monitor every tick (or every 1-min, whatever cadence the existing ATM/OTM page uses for live monitoring). When SL or Target price is breached by the LTP, close position at the trigger price (not at LTP — at the SL/target level itself, to mirror how a real stop order would execute).

### 2.10 Configuration defaults

```python
DEFAULTS = {
    'or_minutes': 15,
    'use_vwap_filter': True,
    'use_rsi_filter': True,
    'rsi_timeframe': '15min',
    'rsi_length': 14,
    'rsi_long_threshold': 60,
    'rsi_short_threshold': 40,
    'use_cpr_dir_filter': True,
    'use_cpr_width_filter': True,
    'cpr_width_threshold_pct': 0.5,  # %
    'sl_type': 'OR Opposite',
    'fixed_sl_points': 50.0,
    'atr_length': 14,
    'atr_sl_multiple': 1.5,
    'target_type': 'R Multiple',
    'r_multiple': 1.5,
    'fixed_target_points': 100.0,
    'or_range_multiple': 1.0,
    'allow_longs': True,
    'allow_shorts': True,
    'max_trades_per_day': 1,
    'last_entry_time': '14:00:00',
    'eod_squareoff_time': '15:20:00',
}
```

---

## 3. Database schema

Add these tables to the existing SQLite DB. Follow the same naming/style conventions as existing tables.

```sql
CREATE TABLE IF NOT EXISTS orb_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument TEXT NOT NULL,  -- 'NIFTY' | 'BANKNIFTY' | 'SENSEX'
    config_json TEXT NOT NULL, -- full config dict serialized
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orb_daily_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument TEXT NOT NULL,
    trade_date DATE NOT NULL,
    or_high REAL,
    or_low REAL,
    or_finalized INTEGER DEFAULT 0,
    cpr_pivot REAL,
    cpr_tc REAL,
    cpr_bc REAL,
    cpr_width_pct REAL,
    is_wide_cpr_day INTEGER DEFAULT 0,
    prev_day_high REAL,    -- audit: which day's HLC was used for CPR
    prev_day_low REAL,
    prev_day_close REAL,
    prev_day_date DATE,    -- audit: explicit date of "previous trading day"
    trades_taken INTEGER DEFAULT 0,
    UNIQUE(instrument, trade_date)
);

CREATE TABLE IF NOT EXISTS orb_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument TEXT NOT NULL,
    signal_time TIMESTAMP NOT NULL,
    direction TEXT NOT NULL,  -- 'LONG' | 'SHORT'
    entry_price REAL NOT NULL,
    sl_price REAL NOT NULL,
    target_price REAL NOT NULL,
    or_high REAL NOT NULL,
    or_low REAL NOT NULL,
    vwap_at_entry REAL,
    rsi_at_entry REAL,
    cpr_pivot REAL,
    cpr_tc REAL,
    cpr_bc REAL,
    cpr_width_pct REAL,
    filter_state TEXT,  -- JSON of which filters were on at signal time
    notes TEXT
);

CREATE TABLE IF NOT EXISTS orb_paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES orb_signals(id),
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL,
    quantity INTEGER NOT NULL,  -- lot size
    entry_time TIMESTAMP NOT NULL,
    entry_price REAL NOT NULL,
    sl_price REAL NOT NULL,
    target_price REAL NOT NULL,
    exit_time TIMESTAMP,
    exit_price REAL,
    exit_reason TEXT,  -- 'TARGET' | 'SL' | 'EOD' | 'MANUAL'
    pnl_points REAL,
    pnl_inr REAL,
    status TEXT DEFAULT 'OPEN'  -- 'OPEN' | 'CLOSED'
);

CREATE TABLE IF NOT EXISTS orb_mtm_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER REFERENCES orb_paper_positions(id),
    timestamp TIMESTAMP NOT NULL,
    ltp REAL NOT NULL,
    unrealized_pnl REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orb_signals_instrument_time ON orb_signals(instrument, signal_time);
CREATE INDEX IF NOT EXISTS idx_orb_positions_status ON orb_paper_positions(status, instrument);
CREATE INDEX IF NOT EXISTS idx_orb_mtm_position ON orb_mtm_history(position_id, timestamp);
```

Note the `prev_day_*` audit columns in `orb_daily_state` — these let us debug CPR mismatches between the dashboard and what we visually expect on TradingView. Always populate them.

---

## 4. Module structure

Create new directory `strategies/orb_paper/` (or wherever existing strategies live in the NAS app):

```
strategies/orb_paper/
├── __init__.py
├── config.py              # DEFAULTS dict + config CRUD
├── indicators.py          # VWAP, RSI, ATR calculators (pure functions)
├── cpr.py                 # CPR calculator + safe data fetch (Section 2.4)
├── orb_engine.py          # Core: signal generator, entry/exit logic per instrument
├── paper_trader.py        # Virtual position book + MTM updater
├── scheduler.py           # Time-based hooks (session start, OR window, EOD)
├── api.py                 # FastAPI/Flask routes for the dashboard page
├── tests/
│   ├── test_indicators.py
│   ├── test_cpr.py        # CRITICAL — test the lookahead-safe pattern
│   ├── test_signals.py    # CRITICAL — must match Pine Script outputs
│   └── test_paper_trader.py
```

Frontend (whatever the existing app uses):

If React: `src/pages/OrbPaperTradingPage.jsx` + `src/components/orb/...`
If Streamlit: `pages/orb_paper_trading.py`

---

## 5. Key implementation details

### 5.1 Live data integration

**REUSE the existing live data fetch infrastructure** from the ATM/OTM strategy page. Do not create a new Kite WebSocket connection. Whatever the existing code does (subscribe to ticks, poll LTP, etc.), do the same for these three instruments.

Concrete steps:
1. Open the ATM/OTM strategy page source code
2. Find the function/class that subscribes to live prices
3. Note its calling pattern (callback? generator? async iterator?)
4. Add NIFTY 50, NIFTY BANK, SENSEX instrument tokens to the same subscription
5. Route the price updates to your new `orb_engine` instances

If the existing app uses 1-min candle aggregation, aggregate further to 5-min for ORB logic (see 5.2).

### 5.2 5-min candle aggregation

Maintain a rolling buffer of 5-min OHLCV candles per instrument. Either:
- **Preferred:** Use Kite's historical API to fetch 5-min candles when initializing (for VWAP and RSI seed data), then update from live ticks
- **Fallback:** Aggregate from 1-min if existing infra streams 1-min

A 5-min candle for 09:15:00 covers 09:15:00–09:19:59 inclusive. The candle "closes" at 09:20:00. ORB logic should evaluate breakouts on candle CLOSE, NOT intra-bar.

### 5.3 Signal evaluation cadence

After 09:30:00 (OR finalized) and before 14:00:00 (last entry), evaluate the signal logic at the close of each 5-min candle. The check should run once at 09:30:00, 09:35:00, 09:40:00, ... 13:55:00.

For running positions (between entry and exit), monitor every tick or every 1-min — whatever cadence the existing app uses. Check SL/target on every price update.

### 5.4 EOD square-off

At 15:20:00 IST sharp, scan all open paper positions and close them at the current LTP. Mark `exit_reason = 'EOD'`. Set `status = 'CLOSED'`.

### 5.5 New trading day reset

At 09:14:00 IST (or any time before market open), for each instrument:
- Reset OR state (or_high, or_low, or_finalized)
- Calculate today's CPR using yesterday's HLC (use the safe fetch pattern from Section 2.4)
- Reset trades_taken counter to 0
- Insert new row in `orb_daily_state` with all CPR fields populated, including the audit columns

If the module starts mid-session (e.g., user opens dashboard at 11:30 AM), it should:
- Detect that today's `orb_daily_state` row already exists OR create it
- Calculate OR from already-elapsed 5-min candles using historical API
- Mark or_finalized=True if past 09:30
- Resume normal evaluation from current time forward

### 5.6 Tick rate limits and error handling

Kite WebSocket can disconnect, throw errors, send malformed data. The existing ATM/OTM page presumably handles this — match its retry and reconnection logic. If it doesn't, add basic patterns:
- Reconnect on disconnect with exponential backoff (1s, 2s, 4s, 8s, max 60s)
- Skip malformed ticks, log them, never let one bad tick crash the engine
- If no ticks received for >30 seconds during market hours, alert the user and pause new entries

---

## 6. Dashboard UI requirements

The page should match the existing NAS app's design language (look at the ATM/OTM page for reference styling).

### 6.1 Page layout

```
┌─────────────────────────────────────────────────────────────────┐
│  ORB Paper Trading                          [Settings] [Pause] │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬─────────────┐                   │
│  │   NIFTY     │  BANKNIFTY  │   SENSEX    │  ← Instrument tabs│
│  └─────────────┴─────────────┴─────────────┘                   │
│                                                                 │
│  ┌─────────── Today's State ────────────┐  ┌─── Live MTM ────┐ │
│  │ OR High: 25,432.50                   │  │   +₹2,450       │ │
│  │ OR Low:  25,378.20                   │  │   (3 positions) │ │
│  │ OR Range: 54.30 (0.21%)              │  └─────────────────┘ │
│  │ CPR: TC 25,420 / Pivot 25,395 / BC..│                      │
│  │ CPR Width: 0.32% ✓ OK                │                      │
│  │ Used prev day: 14-Apr-2026 HLC       │  ← audit row        │
│  │ VWAP: 25,401.25                      │                      │
│  │ RSI(15m): 62.5 ↑                     │                      │
│  │ Trades today: 1 / 1                  │                      │
│  └──────────────────────────────────────┘                      │
│                                                                 │
│  ┌─── Open Paper Positions ──────────────────────────────────┐ │
│  │ # │ Side │ Entry  │ SL    │ Target │ LTP   │ MTM    │ Age │ │
│  │ 1 │ LONG │ 25,410 │25,378 │ 25,458 │25,425 │ +₹1,125│ 23m │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─── Today's Signals ────────────────────────────────────────┐ │
│  │ 09:35 LONG  Entry 25,410 → Open                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─── Strategy Report (Last 30 days) ─────────────────────────┐ │
│  │ Total Trades: 23                                          │ │
│  │ Win Rate: 56.5%                                           │ │
│  │ Total P&L: +₹4,825                                        │ │
│  │ Profit Factor: 1.34                                       │ │
│  │ Avg Win: ₹845  /  Avg Loss: ₹620                          │ │
│  │ Max Drawdown: -₹2,150                                     │ │
│  │ Exit Breakdown: Target 8 / SL 11 / EOD 4                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─── Equity Curve ───────────────────────────────────────────┐ │
│  │  [line chart of cumulative P&L]                           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

The "Used prev day" audit row in Today's State is non-obvious but VERY useful — it lets Arun verify at a glance which day's HLC the system used to calculate CPR. If you see "Used prev day: 13-Apr" on a Tuesday, something is wrong.

### 6.2 Settings panel

A drawer/modal that exposes ALL the config options from `DEFAULTS`. Per-instrument (so Arun can have different filters for Nifty vs Bank Nifty). Save updates `orb_config` table.

Group settings the same way the Pine script does:
- ▶ Opening Range (or_minutes, last_entry_time, eod_squareoff_time)
- ▶ VWAP & RSI Filters
- ▶ CPR Filters
- ▶ Risk Management
- ▶ Direction Control

### 6.3 Refresh cadence

- Live MTM section: update every 5-10 seconds from latest LTP
- Position table: update on every status change + every 30 seconds
- Today's State: update on every 5-min candle close
- Strategy Report: refresh on demand (button) or every 5 min

### 6.4 Pause/Resume

Master switch per instrument. When paused: no new entries triggered, but existing positions still monitored for SL/target/EOD exit.

---

## 7. Critical correctness checks (the "match Pine Script" tests)

Before declaring this working, write these tests in `tests/test_signals.py`:

```python
def test_no_signal_when_inside_or_window():
    # At 09:20 (still in OR window), even if price breaks pre-set OR_high,
    # no signal should fire because OR is not yet finalized.
    pass

def test_long_signal_blocked_when_below_vwap():
    # Setup: price > OR_high, but price < VWAP, vwap_filter=True
    # Expected: long_signal = False
    pass

def test_long_signal_blocked_when_rsi_below_threshold():
    # Setup: all other filters pass, but RSI(15m) = 55 (below 60 threshold)
    # Expected: long_signal = False
    pass

def test_long_signal_blocked_inside_cpr():
    # Setup: BC < price < TC, cpr_dir_filter=True
    # Expected: both long_signal and short_signal = False
    pass

def test_no_trades_on_wide_cpr_day():
    # Setup: cpr_width_pct = 0.7%, threshold = 0.5%, cpr_width_filter=True
    # Expected: cpr_width_ok = False, both signals = False all day
    pass

def test_one_trade_per_day_blocks_second_entry():
    # Setup: max_trades_per_day=1, already have 1 closed trade
    # Expected: subsequent breakouts blocked
    pass

def test_no_entry_after_last_entry_time():
    # Setup: current_time = 14:05, breakout signal valid otherwise
    # Expected: long_signal = False
    pass

def test_eod_squareoff_at_1520():
    # Setup: open position at 14:50, time advances to 15:20
    # Expected: position.exit_reason == 'EOD'
    pass

def test_or_opposite_sl():
    # Long entry at 100, OR_low = 95
    # Expected: position.sl_price == 95
    pass

def test_r_multiple_target():
    # Long entry at 100, SL at 95 (risk = 5), R = 1.5
    # Expected: target = 100 + (5 * 1.5) = 107.5
    pass

# CPR-specific tests (Section 2.4 fix verification)
def test_cpr_uses_yesterdays_completed_candle_not_today():
    # Setup: today is 25-Apr (Friday), yesterday is 24-Apr (Thursday)
    # Expected: CPR calculated from 24-Apr's HLC, NOT today's developing
    pass

def test_cpr_handles_monday_correctly():
    # Setup: today is Monday 28-Apr
    # Expected: prev_day used = Friday 25-Apr (NOT Sunday 27-Apr)
    pass

def test_cpr_handles_post_holiday_correctly():
    # Setup: today is 11-Apr-2026 (assume 10-Apr was a trading holiday)
    # Expected: prev_day used = 09-Apr-2026 (NOT 10-Apr which had no data)
    pass
```

Each test must pass before live deployment.

---

## 8. Build sequence (suggested phases)

### Phase 1 — Core engine + signal validation (2 days)
- Set up module structure
- Implement `indicators.py` (VWAP, RSI, ATR)
- Implement `cpr.py` with the safe data fetch pattern (Section 2.4)
- Implement `orb_engine.py` signal logic
- Write all tests in `test_signals.py` and `test_cpr.py` and make them pass
- **Acceptance gate:** Run engine against the last 30 trading days of historical Bank Nifty 5-min data. Compare total signal count, signal directions, and signal timestamps against the TradingView strategy report. Tolerance: ±1 signal per month due to bar-close timing differences. If divergence is larger, debug before moving on.

### Phase 2 — Paper trading + persistence (1 day)
- Add `paper_trader.py` (position book, MTM updater)
- Database schema migrations
- Verify open → close lifecycle: signal → position open → SL/target/EOD → position closed → P&L recorded
- Test with 5-10 simulated signals to verify each exit reason path

### Phase 3 — Live data integration (1 day)
- Hook into existing live data infra (whatever the ATM/OTM page uses)
- 5-min candle aggregation
- Scheduler for session start, OR finalization, last entry, EOD
- Run live during a market session in OBSERVATION MODE — log signals to DB but don't open paper positions yet. Verify against TradingView in real-time.

### Phase 4 — Dashboard UI (2 days)
- Page scaffold matching ATM/OTM page styling
- Today's State panel (with audit row)
- Open Positions table
- Today's Signals list
- Strategy Report panel
- Equity curve chart
- Settings drawer

### Phase 5 — Polish + alerts (0.5 day)
- Telegram alerts on signal entry / position close (if pipeline exists)
- Pause/resume per instrument
- Manual close position button
- Export trades to CSV

### Phase 6 — Live paper trading (5+ trading days)
- Flip from observation mode to actual paper position opening
- Run for 5 consecutive trading days, watch for errors
- Compare daily P&L against what Arun would have manually computed by reading TradingView signals

---

## 9. Things NOT to do

- **Do NOT execute real broker orders.** This is paper trading only. Even if you accidentally have order placement code from another module, do not call `kite.place_order()` from this module.
- **Do NOT recalculate CPR intraday.** CPR is calculated once at 09:14:00 from yesterday's HLC, then used as a constant for the whole day.
- **Do NOT trade on the OR window itself.** The first signal can only fire AFTER 09:30:00 (or after OR_minutes elapses).
- **Do NOT use intra-bar prices for breakout detection.** Breakouts are evaluated on candle CLOSE only. A price that touches OR_high mid-bar but closes below it is NOT a breakout.
- **Do NOT use lookahead-biased data** in any calculation. Yesterday's daily candle for CPR is fine because it's complete; today's developing daily candle must NOT be used for any CPR or filter logic. See Section 2.4.
- **Do NOT skip the Pine Script reference.** When in doubt about logic, the Pine script is the source of truth.
- **Do NOT use `today - timedelta(days=1)`** for "previous trading day". This breaks on Mondays and after holidays. Use Kite historical API to fetch the most recent completed daily candle.

---

## 10. Reference: Pine Script source of truth

The Pine Script v2.1 file (`ORB_VWAP_RSI_CPR_Filter_v2.1.pine`) implements the exact same logic this module should produce. Key Pine functions and their Python equivalents:

| Pine                              | Python                                       |
|-----------------------------------|----------------------------------------------|
| `ta.vwap(hlc3)`                   | Custom rolling VWAP from session start       |
| `ta.rsi(close, 14)` on 15m TF     | `pandas-ta` or manual Wilder's RSI           |
| `ta.atr(14)`                      | `pandas-ta` ATR                              |
| `request.security(D, [h,l,c])` then `[1]` | `kite.historical_data(interval='day')[-1]` (see Section 2.4) |
| `ta.change(dayofmonth)`           | Date change detection in scheduler           |
| `barstate.islast`                 | "current bar" flag in live mode              |

When porting, treat the Pine script as the spec — if Python output differs from Pine output for any test case, the Python is wrong.

---

## 11. Debugging and observability

This is the section that will save the most time during integration and live debugging.

### 11.1 Logging requirements

Every signal evaluation should log a structured record. At minimum:

```python
logger.info("orb_signal_eval", extra={
    "instrument": "BANKNIFTY",
    "timestamp": "2026-04-15T09:35:00+05:30",
    "candle_close": 56250.50,
    "or_high": 56240.00,
    "or_low": 56180.00,
    "or_finalized": True,
    "vwap": 56210.30,
    "rsi_15m": 62.4,
    "cpr_pivot": 56100.00,
    "cpr_tc": 56150.00,
    "cpr_bc": 56050.00,
    "cpr_width_pct": 0.18,
    "is_wide_cpr": False,
    "long_breakout": True,
    "short_breakout": False,
    "vwap_long_ok": True,
    "rsi_long_ok": True,
    "cpr_long_ok": True,
    "cpr_width_ok": True,
    "long_signal": True,
    "short_signal": False,
    "trades_today": 0,
    "decision": "ENTER_LONG"
})
```

When debugging "why did/didn't this signal fire?", these logs are the single source of truth.

### 11.2 Audit trail in DB

Every CPR calculation MUST persist its inputs (`prev_day_high`, `prev_day_low`, `prev_day_close`, `prev_day_date`) to `orb_daily_state`. When the dashboard shows a CPR value that looks wrong, Arun should be able to query this table and see exactly which day's HLC was used.

### 11.3 Signal-to-Pine reconciliation tool

Build a small CLI utility (`tools/reconcile_with_pine.py`) that:
1. Takes a date range and instrument
2. Reads the Python signals from `orb_signals` table
3. Asks Arun to manually input the TradingView signal count for that period
4. Outputs side-by-side comparison and flags any mismatches

This will get used heavily during Phase 1 acceptance.

### 11.4 Health check endpoint

Add a `/api/orb/health` endpoint that returns:
```json
{
  "status": "ok",
  "instruments": {
    "NIFTY":     {"last_tick_age_sec": 2,  "or_finalized": true,  "open_positions": 0},
    "BANKNIFTY": {"last_tick_age_sec": 1,  "or_finalized": true,  "open_positions": 1},
    "SENSEX":    {"last_tick_age_sec": 8,  "or_finalized": true,  "open_positions": 0}
  },
  "kite_connected": true,
  "last_signal_eval": "2026-04-15T09:35:00+05:30"
}
```

If `last_tick_age_sec` exceeds 30 during market hours, alert.

---

## 12. Definition of done

This module is "done" when:

1. All tests in `test_signals.py` and `test_cpr.py` pass
2. Running against last 30 trading days of historical data produces signals matching the TradingView strategy report (within ±1 signal per month due to bar-close timing differences)
3. Live paper trading runs for 5 consecutive trading days without errors
4. Dashboard page shows correct real-time state for all 3 instruments
5. EOD square-off fires reliably at 15:20:00 every day
6. Paper P&L matches what Arun could verify by manually checking Kite charts at signal times
7. The CPR audit row in the dashboard shows the correct previous trading day on every Monday and post-holiday day during the 5-day live run

---

## 13. Future extensions (do NOT build now, but design with these in mind)

- **Options selling overlay:** simulate selling 1-strike-OTM credit spread on each signal, track parallel options P&L
- **Multi-instrument scaling:** add F&O stocks (REC, COALINDIA, etc.)
- **Live execution mode:** flip a switch to send real orders via Kite (after paper validation)
- **Backtest mode:** same engine, but replay historical data fast for parameter optimization
- **Telegram bot:** /pause, /resume, /status commands

Keep the engine modular so these slot in without rewriting core logic. Specifically:
- `paper_trader.py` should have a clean interface that an `options_paper_trader.py` can mirror later
- The signal generator output should be a structured dict (not a tuple), so adding new fields like option strike doesn't break callers

---

## 14. How to talk to Arun during the build

**Ask Arun before:**
- Choosing between FastAPI vs Flask vs Streamlit if the existing app uses multiple frameworks
- Adding a new Python dependency (e.g., `pandas-ta`) — confirm it's OK to install
- Any design decision that affects how the page looks vs the existing ATM/OTM page
- Skipping any of the tests in Section 7
- Deviating from the Pine Script logic in any way (even "small improvements")

**Do NOT ask before:**
- Adding new files to the `strategies/orb_paper/` directory
- Adding new tables to the existing SQLite DB (per Section 3)
- Writing tests
- Standard Python/React patterns
- Adding logging (just do it)
- Adding error handling and retries (just do it)

**At the end of each phase, report to Arun:**
- What got built
- Which tests pass
- Any deviations from this PRD (with rationale)
- What you need from him to start the next phase

---

## 15. Handoff sequence (DO THIS FIRST)

When you (Claude Code) start this task, do these in EXACTLY this order:

1. **Read this entire PRD end-to-end.** Don't skim.
2. **Read the Pine Script** (`ORB_VWAP_RSI_CPR_Filter_v2.1.pine`).
3. **Open the existing ATM/OTM strategy page code** and answer these questions for yourself:
   - How does it auto-log into Kite? (which file, which class, which method)
   - How does it fetch live prices? (WebSocket subscription? REST polling? At what cadence?)
   - What does the page UI look like? (framework, component structure, styling tokens)
   - Which database tables does it use?
   - How does it handle session lifecycle (login token expiry, reconnection)?
4. **Write a 1-page implementation plan** as a markdown doc and show it to Arun before writing any code. Include: which files you'll create, which existing files you'll touch, which dependencies you'll add, what your Phase 1 acceptance test will be.
5. **Wait for Arun to approve the plan**, then start Phase 1.

Do not skip step 4. Even if you think you understand the existing app perfectly, the plan-review step has caught misunderstandings every single time.
