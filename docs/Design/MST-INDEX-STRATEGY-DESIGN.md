# MST Index Strategy — Design & Implementation Spec

**For: Claude Code, future implementation session**
**Status:** SPEC ONLY — do not start coding until user gives explicit go-ahead.
**Source research:** `research/35_nifty_bnf_master_child_supertrend/` (RESULTS.md has the full backtest evidence)
**Target app:** Quantifyd (`http://94.136.185.54:5000/app`)

---

## 1. What this strategy is

An always-on, signal-driven options operator's bias engine on **NIFTY 30-min**.

- **MST (Master SuperTrend)** sets the directional bias — long or short. Used to enter a debit spread (bull call spread when MST=long, bear put spread when MST=short) at biweekly-to-weekly expiries.
- **CST (Child Stochastic)** signals exhaustion within the active MST trend. Used to convert the debit spread into an **iron condor** by adding a contra credit spread.
- Operator places, manages, and rolls the options manually based on these signals. **Phase 1 of this build is signal-only — no order placement.** Order automation is a later phase.

This page is **observational + alerting** — it shows the live MST/CST state, history, and pings the operator at signal events. It does NOT trade.

---

## 2. The signal rules (locked from research/35)

### 2.1 MST — SuperTrend(ATR=21, multiplier=5.0) on NIFTY50 30-min

- Compute SuperTrend on 30-min OHLC using Wilder ATR with period=21, multiplier=5.0
- Direction `+1` (long bias) or `-1` (short bias)
- A direction flip occurs at the **close of bar `i`** when close crosses the active band

### 2.2 MST entry filter — break-of-extreme (mandatory)

When MST flips at close of bar `i`:

- Record `flip_high = high[i]` and `flip_low = low[i]`
- For LONG flip: bias becomes "armed long" but does NOT activate until a subsequent bar prints `high > flip_high` (intra-bar break is enough — a stop-buy at `flip_high` would fill)
- For SHORT flip: bias becomes "armed short" until `low < flip_low`
- If MST flips again before the breakout occurs, the prior flip is **discarded** — no trade was triggered

This is the validated edge: lifts MFE/MAE from 2.13× → 3.62× at the cost of ~2 hours entry lag and ~6% of flips filtered.

### 2.3 CST — Stochastic(14, 3, 3) on the same 30-min bars

- `%K_raw[i] = 100 × (close[i] − min(low, k=14)) / (max(high, k=14) − min(low, k=14))`
- `%K = SMA(%K_raw, 3)` (smoothing)
- `%D = SMA(%K, 3)`
- All computed on close of completed 30-min bars

### 2.4 CST trigger rules

| MST state | CST trigger condition (at bar close) | Action |
|---|---|---|
| **LONG (active)** | `%K[i-1] >= %D[i-1]` AND `%K[i] < %D[i]` AND `%K[i-1] >= 80` | Operator alert: ADD bear call spread → debit becomes iron condor |
| **SHORT (active)** | `%K[i-1] <= %D[i-1]` AND `%K[i] > %D[i]` AND `%K[i-1] <= 20` | Operator alert: ADD bull put spread → debit becomes iron condor |

Notes:
- The `%K_prev >= 80` (or `<= 20`) gate is what makes this lead the MAE peak — without it, every Stoch cross fires
- If MST is "armed but not active" (waiting for breakout), CST events are IGNORED
- If MST flips, any pending CST state is reset
- A single MST trend can fire the CST multiple times — operator decides whether to roll the credit spread or treat each cross as informational

### 2.5 Bar timing & timezone

- All bars are NSE IST. Session: 09:15 – 15:30
- 30-min buckets: 09:15, 09:45, 10:15, 10:45, 11:15, 11:45, 12:15, 12:45, 13:15, 13:45, 14:15, 14:45, 15:15
- 13 bars per regular session. Last bar (15:15) closes at 15:30.
- All signal evaluation happens **at bar close**. Never act on intra-bar Stoch values — they repaint.
- For the breakout filter: a stop order at `flip_high`/`flip_low` would fill intra-bar; for the alerting layer, treat the first 30-min close where the break already happened as the activation event.

---

## 3. Implementation phases

### Phase 1 (MVP) — Signal generation + dashboard

In scope:
1. Backend service that computes MST and CST in real time on 30-min NIFTY bars
2. SQLite persistence of state and signal history
3. React page at `/app/mst` showing live state, recent flips, recent CST events, and chart
4. Telegram/email alert on every MST flip activation and every CST trigger

Out of scope:
- Order placement (manual operator workflow for now)
- Backtest re-runs from UI (research/35 already has the historical evidence)
- BANKNIFTY (NIFTY only for Phase 1)

### Phase 2 — Optional automation

(Re-spec after Phase 1 runs in paper for ≥ 1 month.)

- Auto-place debit spread on MST activation (Kite API; requires options chain selection logic)
- Auto-place contra credit spread on CST trigger
- Position management, P&L tracking, kill switch

---

## 4. Data layer

### 4.1 Live data source

**Use the existing Kite WebSocket ticker** (already wired for ORB/NAS — see `services/kite_ws_manager.py` if it exists, else how Maruthi/ORB pull live ticks).

NIFTY50 instrument token: confirm at runtime via `kite.instruments("NSE")` lookup. NIFTY 50 is an index, not a tradable symbol — historical-data API uses instrument_token of the index.

For 30-min bars: subscribe to NIFTY ticks → aggregate locally into 30-min OHLC bars at the canonical bucket boundaries (09:15, 09:45, …, 15:15). Close each bar at the bucket end.

Alternative if WS subscription to NIFTY index ticks isn't supported: poll `kite.historical_data(instrument_token, from_date, to_date, '30minute')` every 30 minutes at +5s past the bar close. Slightly higher latency (~10s) but simpler.

### 4.2 Historical seed

On startup, load the last ~200 30-min bars of NIFTY from `backtest_data/market_data.db` (table `market_data_unified`, symbol=`NIFTY50`, timeframe=`30minute` if present, else resample from `5minute`). 200 bars covers the longest ATR period (50) plus enough warmup for Stoch.

### 4.3 New SQLite DB / table

Add to `backtest_data/mst_signals.db` (new file):

```sql
CREATE TABLE mst_bars (
    bar_dt TEXT PRIMARY KEY,        -- ISO 8601 IST
    open REAL, high REAL, low REAL, close REAL,
    atr REAL,
    st_upper REAL, st_lower REAL,
    direction INTEGER,              -- +1, -1, or 0 if not seeded
    stoch_k REAL, stoch_d REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE mst_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,       -- 'flip_armed' | 'flip_activated' | 'flip_discarded' | 'cst_trigger'
    direction INTEGER,              -- +1 (long) or -1 (short)
    bar_dt TEXT NOT NULL,
    price REAL,                     -- close at flip, or breakout level for activation, or close at CST
    flip_high REAL,                 -- only for flip_armed: the level to break
    flip_low REAL,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_mst_events_bar ON mst_events(bar_dt);
CREATE INDEX idx_mst_events_type ON mst_events(event_type);
```

Persist every completed 30-min bar to `mst_bars`. Persist every state transition to `mst_events`.

---

## 5. Signal engine — `services/mst_engine.py`

Single-class engine, similar pattern to `services/kc6_scanner.py`.

```python
class MSTEngine:
    """NIFTY 30-min Master SuperTrend + Stochastic CST signal engine.

    State machine:
      IDLE                              - no MST signal yet (warmup)
      ARMED_LONG                        - MST flipped long on bar close, waiting for high break
      ARMED_SHORT                       - MST flipped short on bar close, waiting for low break
      ACTIVE_LONG                       - MST long & breakout confirmed; CST is being monitored
      ACTIVE_SHORT                      - MST short & breakout confirmed
    """
    ATR_PERIOD = 21
    MULTIPLIER = 5.0
    STOCH_K = 14
    STOCH_D = 3
    STOCH_SMOOTH = 3
    STOCH_OB = 80
    STOCH_OS = 20

    def on_new_bar(self, bar):
        # 1. Append bar to rolling buffer
        # 2. Recompute SuperTrend, ATR, Stoch on the buffer
        # 3. State transitions:
        #    a. If direction flipped on this bar's close: emit 'flip_armed', store flip_high/low,
        #       transition to ARMED_LONG or ARMED_SHORT
        #    b. If state is ARMED_LONG and bar.high > flip_high: emit 'flip_activated',
        #       transition to ACTIVE_LONG
        #    c. Mirror for ARMED_SHORT/ACTIVE_SHORT
        #    d. If state is ARMED_* and direction flips again: emit 'flip_discarded',
        #       reset to ARMED_<new direction>
        #    e. If state is ACTIVE_* and Stoch %K crosses %D from extreme: emit 'cst_trigger'
        # 4. Persist bar and any events
        # 5. Return list of events for this bar (for the alerting layer)
```

**Reuse**: the SuperTrend and Stochastic implementations from `research/35_nifty_bnf_master_child_supertrend/scripts/supertrend.py` are correct and tested — port them verbatim into `services/mst_engine.py` (or import).

### 5.1 Scheduling

Add to `app.py` near the existing KC6/ORB scheduled jobs:

```python
# 30-min cron at 09:46, 10:16, 10:46, ..., 15:16 IST (1 minute after bar close)
scheduler.add_job(mst_evaluate_latest_bar, 'cron',
                  day_of_week='mon-fri', hour='9-15',
                  minute='16,46', id='mst_30min_eval')
```

`mst_evaluate_latest_bar()` pulls the just-closed 30-min bar from Kite (or from a WS-aggregated buffer), feeds it to `MSTEngine.on_new_bar()`, persists the result, and dispatches alerts.

---

## 6. Backend API — Flask routes in `app.py`

Match the convention used by `/api/orb/*`, `/api/nas/*`, `/api/kc6/*`.

| Route | Method | Returns |
|---|---|---|
| `/api/mst/state` | GET | `{ state, mst_direction, current_close, last_flip_dt, armed_levels: {high,low}, last_cst_dt, stoch_k, stoch_d, atr }` |
| `/api/mst/bars?limit=200` | GET | Last N persisted 30-min bars with computed indicators (for chart) |
| `/api/mst/events?limit=50&type=*` | GET | Recent events from `mst_events` |
| `/api/mst/scan` | POST | Force a re-evaluation now (debug aid; no production use) |
| `/api/mst/alerts/test` | POST | Send a test alert through the configured channel |

All JSON, all behind the existing CORS/session middleware. No auth changes needed beyond what other endpoints already require.

---

## 7. Frontend — React page at `/app/mst`

Per CLAUDE.md project rules (binding from 2026-04-26): all new pages live in `frontend/src/pages/`. Match the design language of `Nas.tsx` / `Orb.tsx` / `Nwv.tsx`.

### 7.1 Files to create

```
frontend/src/pages/Mst.tsx
frontend/src/pages/Mst.module.css
```

### 7.2 Layout (top to bottom)

1. **Page header** — "MST · NIFTY 30-min" + subtitle "SuperTrend(21, 5.0) + Stoch(14,3,3)"

2. **Status strip** (4 metric cards in a row, reuse `MetricCard`):
   - Current MST state (`ACTIVE_LONG` / `ACTIVE_SHORT` / `ARMED_LONG` / `ARMED_SHORT` / `IDLE`) with a colored `StatusDot`
   - Bias direction with "Long since" / "Short since" timestamp
   - Last CST trigger ("none yet" or "12m ago at 25,123")
   - Stoch reading: `%K = 78.4 / %D = 75.1` with a small chip indicating "approaching OB" / "approaching OS" / "neutral"

3. **Chart panel** — single 30-min OHLC chart with overlays:
   - SuperTrend line (green when long, red when short)
   - Markers at MST flip-armed events (yellow dot)
   - Markers at MST flip-activated events (green/red triangle)
   - Markers at CST triggers (orange diamond)
   - Stoch panel below the price chart with %K, %D, and 80/20 reference lines

   Use the same chart library already in use elsewhere in the SPA — check what `Orb.tsx` uses. Do NOT introduce a new charting dep.

4. **Recent events table** (reuse `DataTable`):
   - Columns: Bar time · Event · Direction · Price · Notes
   - Filter chips above for: All / Flip activations / CST triggers / Discarded
   - Default = last 30 events

5. **Operator playbook panel** (static markdown, no logic):
   - "When MST activates LONG: enter bull call spread, weekly or biweekly, ~7 DTE"
   - "When MST activates SHORT: enter bear put spread, weekly or biweekly, ~7 DTE"
   - "When CST triggers: convert to iron condor by selling contra credit spread"
   - "When MST flips opposite: close existing condor, place new debit spread in new direction"

   This is just text — the operator runs the trades manually for now.

### 7.3 Sidebar entry

In `frontend/src/components/Sidebar/Sidebar.tsx`, add a new menu item under the Strategies group: "MST" → `/app/mst`. Use a trend-line icon from the existing `Icons` component.

### 7.4 Routing

In `frontend/src/App.tsx`:
```tsx
<Route path="/app/mst" element={<Mst />} />
```

### 7.5 Build & deploy

`cd frontend && npm run build` produces `frontend/dist/` which Flask serves under `/app/*`. Frontend-only changes do NOT require a backend restart — see CLAUDE.md "NO BACKEND RESTART DURING MARKET HOURS" for deploy rules.

The backend changes (new SQLite DB, new APIs, new scheduler job) DO require a restart and must be deployed after 15:30 IST.

---

## 8. Alerting

Use whatever channel is already configured for ORB/NAS/KC6 alerts. Likely Telegram. Two alert types:

### 8.1 MST flip-activated

```
🟢 MST LONG ACTIVATED
NIFTY 30m · 11:45 IST · 25,124
Flip @ 25,098 · break above 25,118 confirmed
ATR21 = 145
Action: enter bull call spread (weekly/biweekly, ~7 DTE)
```

### 8.2 CST trigger

```
🔶 CST TRIGGER (within active LONG)
NIFTY 30m · 13:15 IST · 25,189
%K crossed below %D from above 80 (K=78.2, D=80.4)
Action: SELL bear call spread above current price → condor
```

### 8.3 Optional: armed/discarded
- Armed events: low-priority log line, no push
- Discarded events: log only

---

## 9. Testing approach

Phase 1 has no order placement, so risk is low. Two test layers:

### 9.1 Replay test (offline)

Build `tests/test_mst_engine_replay.py`:
- Load NIFTY 30-min bars 2024-03-01 → 2026-03-25 from `market_data.db`
- Feed each bar to `MSTEngine.on_new_bar()` sequentially
- Compare emitted events to a frozen golden file generated from `research/35_nifty_bnf_master_child_supertrend/scripts/run_mst_sweep_breakout.py` output for cell `NIFTY50_30min_p21_m5.0`
- All events (flip_armed, flip_activated, flip_discarded, cst_trigger) must match within ±1 bar

### 9.2 Live paper run

After deploy, monitor `/app/mst` for 4 weeks of live data:
- Verify alerts fire at expected times (cross-check against TradingView with same indicator settings)
- Verify state machine never gets stuck
- Verify DB persistence is consistent across restarts

---

## 10. Open questions for the user (resolve before coding)

1. **Alert channel** — Telegram already in use? Or email? Or both?
2. **NIFTY ticker source** — does the existing Kite WS infra already subscribe to NIFTY 50 index ticks? Or do we need polling-via-historical-data?
3. **Phase 2 timing** — when do we revisit auto-order-placement? Default = "after 4 weeks of clean Phase 1 paper signal."
4. **Sidebar grouping** — is there a "Signals" or "Index Strategies" group, or does it slot into the existing Strategies list?
5. **BANKNIFTY** — research/35 also recommended a BNF cell. Do you want a `/app/mst-bnf` mirror in Phase 1, or strictly NIFTY only? (Recommendation: strictly NIFTY for Phase 1.)
6. **Conflict with KC6 on NIFTY?** KC6 doesn't trade indices, so no conflict — confirm.

---

## 11. References

- Research artifacts: `research/35_nifty_bnf_master_child_supertrend/RESULTS.md` and the CSV outputs
- SuperTrend & Stoch reference impl: `research/35_nifty_bnf_master_child_supertrend/scripts/supertrend.py`
- App architecture: `docs/Design/ARCHITECTURE.md`, `docs/Design/LIVE-TRADING-ARCHITECTURE.md`
- React page patterns: `frontend/src/pages/Nas.tsx`, `Orb.tsx`, `Nwv.tsx`
- CLAUDE.md rules to honor:
  - "ALL NEW PAGES GO IN THE REACT APP AT `/app/*` — NOT JINJA"
  - "NO BACKEND RESTART DURING MARKET HOURS" (09:15–15:30 IST)
  - "LIVE-STATUS MD CONVENTION" (apply if implementation runs >5 min)
