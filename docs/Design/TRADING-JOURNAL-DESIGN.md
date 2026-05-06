# Trading Journal — Implementation Design

**Date:** 2026-05-06
**Status:** Design — pre-implementation
**Companion doc:** [TRADING-JOURNAL-RESEARCH.md](TRADING-JOURNAL-RESEARCH.md) (feature survey + research basis)
**Project convention:** [CLAUDE.md](../../.claude/CLAUDE.md) — React SPA at `/app/<route>`, light editorial design system, hairline borders, Inter type, navy/amber accents.

---

## 1. Architectural principles

### 1.1 The journal is a *layer above* execution data

Existing strategy DBs are the **source of truth for what happened**:

| Source DB | Source table(s) | Strategies |
|---|---|---|
| `backtest_data/orb_live.db` | `orb_positions`, `orb_signals` | ORB Cash, ORB Index |
| `backtest_data/kc6_trading.db` | `kc6_trades`, `kc6_positions` | KC6 mean-reversion |
| `backtest_data/nas.db` | `nas_legs`, `nas_strangles` | NAS strangles |
| `backtest_data/maruthi.db` | `maruthi_trades` | Maruthi (disabled) |
| `backtest_data/nifty_strangle.db` | `nifty_strangle_trades` | NIFTY weekly strangle |
| `backtest_data/trident.db` | `trident_trades` | Trident |

The journal **never re-records** these. It maintains its own
`journal_trades` table that is a **projection** over them, plus
journal-only enrichment (tags, notes, scores, reviews). On every
journal page load, a sync function reconciles new trades from each
source DB into `journal_trades`.

### 1.2 New journal-only tables (in a new DB: `backtest_data/journal.db`)

Keeping the journal in its own DB avoids polluting per-strategy DBs
with cross-strategy concerns and makes the journal feature
self-contained (deploy / migrate / wipe independently).

### 1.3 The journal is a generator of insights, not a recorder of events

Any time a journal page asks "what's the win-rate by tag?" or
"equity curve filtered by ORB", that should be a query against
`journal_trades` joined with `journal_tags`. No precomputation in
MVP. Add materialised summaries (`daily_summaries` table) only when
slow.

---

## 2. Data model

### 2.1 New DB: `backtest_data/journal.db`

```sql
-- Master trade row, one per round-trip across all strategies
CREATE TABLE journal_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_db       TEXT NOT NULL,        -- 'orb_live' | 'kc6' | 'nas' | 'maruthi' | 'nifty_strangle' | 'trident' | 'manual'
    source_table    TEXT NOT NULL,        -- 'orb_positions' | 'kc6_trades' | 'nas_legs' | …
    source_id       INTEGER,              -- PK in source table; NULL for manual entries
    strategy        TEXT NOT NULL,        -- 'ORB-CASH' | 'ORB-INDEX' | 'KC6' | 'NAS-916' | 'DIAMOND-SHORT' | 'LONG-MR' | 'LONG-TC' | 'MANUAL' | …
    instrument      TEXT NOT NULL,        -- 'RELIANCE' | 'NIFTY25APR24800CE' | …
    instrument_type TEXT NOT NULL,        -- 'EQUITY' | 'INDEX' | 'OPTION' | 'FUTURE'
    direction       TEXT NOT NULL,        -- 'LONG' | 'SHORT'
    qty             INTEGER NOT NULL,
    entry_price     REAL NOT NULL,
    entry_time      TIMESTAMP NOT NULL,
    exit_price      REAL,                 -- NULL while open
    exit_time       TIMESTAMP,
    exit_reason     TEXT,                 -- 'TARGET' | 'SL' | 'EOD' | 'TRAIL' | 'MANUAL' | 'KILL_SWITCH'
    pnl_gross       REAL,                 -- before brokerage
    pnl_charges     REAL,                 -- brokerage + STT + GST + exchange + SEBI
    pnl_net         REAL,                 -- gross - charges
    r_multiple      REAL,                 -- pnl_net / initial_risk
    initial_risk    REAL,                 -- |entry_price - sl_price| * qty
    hold_minutes    INTEGER,              -- exit_time - entry_time
    mode            TEXT NOT NULL,        -- 'PAPER' | 'LIVE'
    grade           INTEGER,              -- 1-5 process score, journal-only
    UNIQUE (source_db, source_table, source_id)
);

CREATE INDEX idx_jt_strategy ON journal_trades(strategy);
CREATE INDEX idx_jt_entry_date ON journal_trades(date(entry_time));
CREATE INDEX idx_jt_instrument ON journal_trades(instrument);

-- Trade-level enrichment captured at trade time (NIFTY regime, gap, etc.)
CREATE TABLE journal_trade_context (
    trade_id         INTEGER PRIMARY KEY,
    nifty_5m_trend   TEXT,                -- 'UP' | 'DOWN' | 'FLAT'
    nifty_daily_adx  REAL,
    nifty_gap_pct    REAL,
    india_vix        REAL,
    gap_tier         TEXT,                -- 'FLAT' | 'SMALL' | 'MEDIUM' | 'LARGE'
    cpr_width_pct    REAL,
    rsi_at_entry     REAL,
    vwap_at_entry    REAL,
    mae              REAL,                -- max adverse excursion ₹/share
    mfe              REAL,                -- max favourable excursion ₹/share
    slippage_entry   REAL,                -- signal_price - actual_entry_price (signed)
    slippage_exit    REAL,
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE CASCADE
);

-- Free-form notes, markdown, multi-paragraph
CREATE TABLE journal_trade_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id    INTEGER NOT NULL,
    body_md     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE CASCADE
);

-- Image attachments (paths into a static/screenshots/ folder, never blobs)
CREATE TABLE journal_trade_screenshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id    INTEGER NOT NULL,
    file_path   TEXT NOT NULL,            -- relative to DATA_DIR/journal_screenshots/
    caption     TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE CASCADE
);

-- Tag dictionary
CREATE TABLE journal_tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    category    TEXT NOT NULL,            -- 'STRATEGY' | 'SETUP' | 'MISTAKE' | 'CONVICTION' | 'CUSTOM'
    color_hex   TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Many-to-many tag → trade
CREATE TABLE journal_trade_tags (
    trade_id    INTEGER NOT NULL,
    tag_id      INTEGER NOT NULL,
    PRIMARY KEY (trade_id, tag_id),
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id)   REFERENCES journal_tags(id)   ON DELETE CASCADE
);

-- Daily review (one row per trading day)
CREATE TABLE journal_daily_review (
    trade_date     DATE PRIMARY KEY,
    pre_market_md  TEXT,                  -- intent log (filled before open)
    post_close_md  TEXT,                  -- post-close 5-question review
    rule_violations INTEGER DEFAULT 0,
    discipline_score INTEGER,             -- 1-5
    nifty_close    REAL,
    nifty_chg_pct  REAL,
    pnl_gross      REAL,
    pnl_net        REAL,
    trades_count   INTEGER,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Materialised daily summary (rebuild nightly + on-demand)
CREATE TABLE journal_daily_summary (
    trade_date     DATE NOT NULL,
    strategy       TEXT NOT NULL,         -- '*' = all strategies aggregate
    trades_count   INTEGER,
    wins           INTEGER,
    losses         INTEGER,
    pnl_gross      REAL,
    pnl_net        REAL,
    largest_win    REAL,
    largest_loss   REAL,
    avg_r          REAL,
    PRIMARY KEY (trade_date, strategy)
);

-- Kite reconciliation
CREATE TABLE journal_kite_reconciliation (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        INTEGER,              -- NULL if Kite trade has no journal match
    kite_order_id   TEXT,
    kite_trade_id   TEXT,
    kite_symbol     TEXT,
    kite_side       TEXT,
    kite_qty        INTEGER,
    kite_price      REAL,
    kite_filled_at  TIMESTAMP,
    status          TEXT NOT NULL,        -- 'MATCHED' | 'UNMATCHED_KITE_ONLY' | 'UNMATCHED_JOURNAL_ONLY'
    reviewed        BOOLEAN DEFAULT 0,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE SET NULL
);

-- Missed-signal log
CREATE TABLE journal_missed_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy        TEXT NOT NULL,
    signal_time     TIMESTAMP NOT NULL,
    instrument      TEXT NOT NULL,
    direction       TEXT,
    expected_entry  REAL,
    expected_sl     REAL,
    expected_target REAL,
    miss_reason     TEXT,                 -- 'KITE_REJECTED' | 'CAPITAL_BLOCKED' | 'MANUAL_OVERRIDE' | 'KILL_SWITCH' | 'UNKNOWN'
    estimated_pnl   REAL,                 -- if EOD price had been hit
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.2 Reuse vs build

| Concern | Reuse | Build new |
|---|---|---|
| ORB trade source | `services/orb_db.py` (`orb_positions`) | thin reader `services/journal/sources/orb_source.py` |
| KC6 trade source | `services/kc6_db.py` (`kc6_trades`) | `services/journal/sources/kc6_source.py` |
| NAS trade source | `services/nas_*` modules | `services/journal/sources/nas_source.py` |
| Kite reconciliation | `services/kite_service.py` (KiteService) | `services/journal/kite_reconciler.py` |
| Charges calc (STT/GST/brokerage) | — | `services/journal/charges.py` (new — Indian intraday formulas) |
| MAE/MFE for intraday trades | `market_data_unified` (5-min OHLCV via existing `data_manager`) | `services/journal/excursion.py` |
| NIFTY regime tagging | `services/premarket_brief.py` already computes ADX/VIX/gap | `services/journal/regime_tagger.py` (calls existing premarket logic) |
| Frontend layout | `Layout`, `Sidebar`, `TopBar`, `MetricCard`, `Chip`, `DataTable`, `StatusDot` | `JournalCalendar`, `EquityCurveSvg`, `RDistributionChart`, `TagPicker` |

Charges calc is the only non-trivial new service: brokerage = ₹20 or
0.03% per executed order (whichever is lower) for Zerodha intraday;
STT = 0.025% on sell side intraday equity; GST = 18% on (brokerage +
exchange tx + SEBI); exchange tx ≈ 0.00322% NSE; SEBI ≈ ₹10/crore;
stamp ≈ 0.003% on buy side. Encapsulate this once in `charges.py`.

---

## 3. Auto-import path

### 3.1 Sync flow

```
┌─────────────────────────┐
│ Strategy DBs (read-only)│
│ orb / kc6 / nas / …     │
└────────────┬────────────┘
             │ poll on:
             │  • app startup
             │  • every /api/journal/* call (cheap, <50ms)
             │  • nightly cron at 16:00 IST
             ▼
┌─────────────────────────┐
│ services/journal/sync.py│
│  for each source:       │
│    sources/orb_source   │
│    .pull_new_trades()   │
│    → upsert into        │
│    journal_trades       │
│    (UNIQUE constraint   │
│     on source_db+id     │
│     prevents dupes)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ journal.db              │
│  journal_trades         │
│  + journal_trade_context│
└─────────────────────────┘
```

### 3.2 Manual entry

For trades placed outside any system (e.g. discretionary one-off
buys), the user can manually add a trade via `/api/journal/trades`
POST. `source_db='manual'`, `source_id=NULL`. The UI exposes this on
`/app/journal` via a "Log manual trade" button.

### 3.3 Kite tradebook reconciliation (Phase 2)

Daily cron at 16:30 IST:

1. Pull `kite.orders()` and `kite.trades()` for today.
2. For each Kite trade, try to match against `journal_trades` on
   (instrument, side, qty, fill_time within ±60s).
3. Insert matched rows into `journal_kite_reconciliation` with
   status='MATCHED'.
4. Any Kite trade with no journal match → status='UNMATCHED_KITE_ONLY'
   (we placed something off-system).
5. Any journal trade with no Kite match → status='UNMATCHED_JOURNAL_ONLY'
   (paper-mode trade, or system thought it filled but didn't).
6. Surface unmatched count on `/app/journal` header.

---

## 4. Pages

All pages live under `/app/journal/*`. React Router v6, components
in `frontend/src/pages/Journal*.tsx` and shared journal components
in `frontend/src/components/Journal/`.

### 4.1 Page tree

| Route | Page | Tier |
|---|---|---|
| `/app/journal` | **JournalCalendar** — month-view P&L heatmap + recent-trades panel + day drill-in | 1 |
| `/app/journal/day/:date` | **JournalDay** — single-day page: all trades that day, daily review form, NIFTY context | 1 |
| `/app/journal/trade/:id` | **JournalTrade** — single-trade detail (entry/exit chart, R, MAE/MFE, tags, screenshots, notes) | 1 |
| `/app/journal/insights` | **JournalInsights** — equity curve, drawdown chart, per-strategy attribution, win-rate by tag, R-distribution | 1 |
| `/app/journal/strategies/:name` | **JournalStrategy** — single-strategy deep-dive (e.g. `/app/journal/strategies/DIAMOND-SHORT`) | 2 |
| `/app/journal/reconciliation` | **JournalReconciliation** — Kite vs journal, unmatched fills | 2 |
| `/app/journal/missed` | **JournalMissed** — missed signals + override log | 3 |
| `/app/journal/review` | **JournalReview** — weekly system grading dashboard | 3 |
| `/app/journal/export` | **JournalExport** — CSV / FY tax pack export | 3 |

### 4.2 Page-by-page detail

#### `/app/journal` — JournalCalendar (MVP)

Two-column layout:

**Left (60%):** Month calendar heatmap. Each day cell shows:
- Date number
- Net P&L for the day (large numeral, P&L colour)
- Trade count (small)
- Cell background tinted by P&L magnitude (green / red, 5-step scale)

Click a day → navigate to `/app/journal/day/:date`.

Above calendar: month metric strip — month-net P&L, win-rate,
trades, best day, worst day.

**Right (40%):** Recent-trades stream (last 30 trades, all
strategies). Each row: time, instrument, strategy chip, side, P&L,
R-multiple, mistake-tag dot if present. Click row → trade detail.

Header: page title + "Log manual trade" button + filter chips
(strategy multi-select, date range).

#### `/app/journal/day/:date` — JournalDay (MVP)

Top: day metrics (date, NIFTY close + change, day-net P&L,
discipline score badge, trade count, Pre-Market Brief preview link).

Middle: timeline of trades for the day, sorted by entry time. Each
trade: instrument, strategy, side, entry/exit prices, P&L,
R-multiple, hold-time. Click → trade detail.

Bottom: **Daily review** form (markdown editor). 5 prompts:
1. Did I follow the system on every trade today?
2. Any rule violations?
3. What surprised me?
4. What's tomorrow's priority?
5. Discipline score (1-5).

#### `/app/journal/trade/:id` — JournalTrade (MVP)

Layout:

**Header:** instrument, strategy chip, side, mode (paper/live), date.

**Metric strip:** Entry / Exit / P&L net / R-multiple / Hold-time / Grade.

**Chart section:** 5-min candle chart of the trade window with entry
and exit markers. Entry = green up-triangle for long, red down for
short. Exit marked with reason. SL and target as horizontal dashed
lines. Hover candle → OHLC tooltip. (Phase 2: replay scrubber.)

**Excursion strip:** MAE and MFE expressed in ₹/share, in % from
entry, and in R-units. Bar visualisation: SL-line at left, target
at right, MAE marker in red, MFE marker in green, exit position
shown.

**Tags section:** chip cloud, click to add/remove. Strategy tag is
read-only (driven by source). Add Setup / Mistake / Conviction tags.

**Notes section:** markdown editor. Auto-saves on blur.

**Screenshots:** drag-drop slot. Max 5 images per trade. Phase 2:
auto-snapshot the chart on entry/exit.

**Source link:** "View in ORB dashboard →" — link back to the
strategy page where this trade originated.

#### `/app/journal/insights` — JournalInsights (MVP)

Top metric strip: total trades, total net P&L, win-rate, profit
factor, expectancy R, Sharpe, max DD, current DD.

Section 1 — **Equity curve.** Line chart of cumulative net P&L over
time. Filter pills above: All / per-strategy. Toggle: gross vs net,
₹ vs R.

Section 2 — **Drawdown.** Underwater chart (drawdown from rolling
peak) plus a top-5 drawdown windows table (start, end, depth,
duration, recovery date).

Section 3 — **Per-strategy attribution.** Table: strategy, trades,
win-rate, PF, expectancy R, total net P&L, contribution %. Sortable.

Section 4 — **Win-rate by tag.** Grouped bars per tag category
(Setup, Mistake, Conviction). Lets you see "Mistake = override:
WR 42%" type patterns.

Section 5 — **R-distribution.** Histogram of R-multiples across all
trades. Median, mean, expectancy lines.

#### Phase 2/3 pages — outline only

- **JournalStrategy** — same as Insights but scoped to one strategy;
  adds Expected EV vs Realised EV chart.
- **JournalReconciliation** — Kite vs journal table, unmatched-only
  filter, "mark as reviewed" action.
- **JournalMissed** — missed-signal log, sortable by reason and
  estimated cost.
- **JournalReview** — weekly Sunday card per system: Green/Amber/Red
  + drift metrics + "should I size up/down/pause" suggestion.

---

## 5. Backend API

All endpoints under `/api/journal/*`, JSON responses.

### 5.1 MVP endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/journal/trades` | List trades. Query params: `from`, `to`, `strategy`, `instrument`, `tag`, `mode`, `limit`, `offset` |
| GET | `/api/journal/trades/<id>` | Single trade with full context, tags, notes, screenshots, source link |
| POST | `/api/journal/trades` | Manual trade entry |
| PATCH | `/api/journal/trades/<id>` | Update grade, exit-price corrections, mode toggle |
| DELETE | `/api/journal/trades/<id>` | Delete (manual entries only) |
| GET | `/api/journal/calendar?month=YYYY-MM` | Month-grid: per-day net P&L + trade count |
| GET | `/api/journal/day/<date>` | Day page bundle: trades, daily review, NIFTY context |
| POST | `/api/journal/day/<date>/review` | Save daily review |
| GET | `/api/journal/insights?from=&to=&strategy=` | Insights bundle: equity curve, DD, per-strategy attribution, R-distribution |
| GET | `/api/journal/tags` | All tags |
| POST | `/api/journal/tags` | Create tag |
| POST | `/api/journal/trades/<id>/tags` | Attach tags to trade `{tag_ids: [...]}` |
| DELETE | `/api/journal/trades/<id>/tags/<tag_id>` | Detach tag |
| POST | `/api/journal/trades/<id>/notes` | Save notes (markdown body) |
| POST | `/api/journal/trades/<id>/screenshots` | Multipart upload, max 5MB image |
| POST | `/api/journal/sync` | Force-sync from all source DBs (also runs on every GET) |

### 5.2 Phase 2 endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/journal/reconciliation?from=&to=` | Kite-vs-journal reconciliation |
| POST | `/api/journal/reconciliation/<id>/review` | Mark reviewed |
| GET | `/api/journal/strategies/<name>` | Strategy deep-dive bundle |
| POST | `/api/journal/regime/backfill?from=&to=` | Backfill NIFTY-regime context for old trades |

### 5.3 Phase 3 endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/journal/missed?from=&to=` | Missed-signal log |
| GET | `/api/journal/review/weekly?week=YYYY-Www` | Weekly system grading |
| GET | `/api/journal/export?fy=YYYY-YY&format=csv` | FY tax-pack export |

---

## 6. Phasing

### Phase 1 — MVP (week 1)

Goal: a working journal with real ORB and KC6 trades from day one.

- [ ] Create `backtest_data/journal.db` and migrate schema (Tier-1 tables).
- [ ] Implement `services/journal/db.py` (connection helper, schema bootstrap).
- [ ] Implement `services/journal/sources/orb_source.py` (read `orb_positions` → upsert).
- [ ] Implement `services/journal/sources/kc6_source.py` (read `kc6_trades` → upsert).
- [ ] Implement `services/journal/sync.py` (orchestrator).
- [ ] Implement `services/journal/charges.py` (Indian intraday charges calc).
- [ ] Implement `services/journal/api.py` — Flask blueprint with MVP endpoints.
- [ ] Register blueprint in `app.py`.
- [ ] Build React pages: Calendar, Day, Trade, Insights.
- [ ] Add to Sidebar (Journal item between "Holdings" and "Reports").
- [ ] Seed test data: backfill all existing ORB trades from April-May 2026.
- [ ] Smoke-test on `/app/journal` with real data.

**Deliverable:** Arun opens `/app/journal`, sees a calendar with the
last 3 weeks of ORB trades, can click into a day, click into a
trade, add tags and notes.

### Phase 2 — Differentiators (weeks 2-3)

- [ ] Phase-2 schema additions: `journal_trade_context` enrichment fields, `journal_kite_reconciliation`.
- [ ] `services/journal/excursion.py` — MAE/MFE from `market_data_unified`.
- [ ] `services/journal/regime_tagger.py` — NIFTY-regime auto-tag at trade time + backfill.
- [ ] `services/journal/kite_reconciler.py` — daily 16:30 IST cron.
- [ ] R-multiple computation (already in schema, just populate).
- [ ] Slippage tracker (entry signal price already available in `orb_signals`).
- [ ] Daily review form persisted to `journal_daily_review`.
- [ ] Trade-grade scoring UI on trade page.
- [ ] Mistake-type taxonomy seed tags.
- [ ] JournalReconciliation page.

### Phase 3 — Killer features (months later, prioritised when needed)

- [ ] `journal_missed_signals` schema + ORB diff job.
- [ ] Override log instrumentation (every kill-switch click → log row).
- [ ] Pre-market intent log + Pre-Market Brief auto-attach.
- [ ] Weekly system-grading job (Sunday 18:00 IST).
- [ ] Trade replay UI (5-min candle scrubber).
- [ ] FY tax-pack export.
- [ ] Expected EV vs Realised EV per strategy.

---

## 7. Test data plan — seeding from real trades

The user already has weeks of real ORB trades from April-May 2026 in
`backtest_data/orb_live.db` (`orb_positions` table). These are the
gold-standard test fixtures.

### 7.1 Seed sequence

1. Build the journal schema (Phase 1 step 1).
2. Run `python -c "from services.journal.sync import sync_all; sync_all()"`.
3. The `orb_source.pull_new_trades()` reader projects every CLOSED
   ORB position into `journal_trades`. ~15-30 rows expected.
4. Backfill regime context for these trades: `POST
   /api/journal/regime/backfill?from=2026-04-17&to=today`.
5. Manually grade and tag the first 10 trades to dogfood the UI.

### 7.2 Synthetic data for empty-strategy demos

For the three new intraday systems (Diamond Short, Long-MR,
Long-TC), no live trades exist yet. Seed with simulated trades from
the walk-forward results in
`research/37_intraday_75wr_quest/results/` so the per-strategy
pages aren't empty when the user demos the journal. Mark these
rows with `mode='PAPER'` and a `simulation` tag so they're
visually distinct.

### 7.3 Acceptance — what "done" looks like for MVP

- Calendar shows real April-May ORB days with correct P&L.
- Click any day → see the actual trades that occurred.
- Click any trade → see entry/exit/SL/target as recorded by the
  ORB engine.
- Insights page shows ~30 trades, an equity curve, an honest
  win-rate (ORB live live-mode is at ~50% WR, that's the truth).
- Add a tag, refresh, tag persists.
- Add markdown notes, refresh, notes persist.

---

## 8. Open questions / decisions deferred

1. **Screenshots — local file storage or cloud?** Phase 1: local
   `DATA_DIR/journal_screenshots/`, served by Flask. Phase 3: maybe
   S3 if the volume gets unwieldy.
2. **Markdown rendering — server or client?** Client-side via
   `react-markdown`. Server stores raw md.
3. **TradingView replay vs custom canvas?** MVP: static SVG chart
   from `market_data_unified`. Phase 3: consider TradingView
   lightweight-charts library.
4. **Multi-user?** No. Single-trader app, no auth on journal beyond
   the existing app session.

---

## 9. Files (planned)

| Path | Purpose | Phase |
|---|---|---|
| `services/journal/__init__.py` | Package marker | 1 |
| `services/journal/db.py` | SQLite connection + schema bootstrap | 1 |
| `services/journal/sync.py` | Orchestrator: pull from all sources | 1 |
| `services/journal/sources/orb_source.py` | ORB-positions reader | 1 |
| `services/journal/sources/kc6_source.py` | KC6-trades reader | 1 |
| `services/journal/sources/nas_source.py` | NAS-legs reader | 1 |
| `services/journal/charges.py` | Indian intraday charges calc | 1 |
| `services/journal/api.py` | Flask blueprint | 1 |
| `services/journal/excursion.py` | MAE/MFE from 5-min OHLCV | 2 |
| `services/journal/regime_tagger.py` | NIFTY regime auto-tag | 2 |
| `services/journal/kite_reconciler.py` | Daily Kite vs journal diff | 2 |
| `frontend/src/pages/JournalCalendar.tsx` + `.module.css` | Calendar page | 1 |
| `frontend/src/pages/JournalDay.tsx` + `.module.css` | Day page | 1 |
| `frontend/src/pages/JournalTrade.tsx` + `.module.css` | Trade detail | 1 |
| `frontend/src/pages/JournalInsights.tsx` + `.module.css` | Insights page | 1 |
| `frontend/src/components/Journal/CalendarHeatmap.tsx` | Reusable heatmap | 1 |
| `frontend/src/components/Journal/EquityCurve.tsx` | SVG equity curve | 1 |
| `frontend/src/components/Journal/TagPicker.tsx` | Tag chip picker | 1 |
| `frontend/src/components/Journal/RDistribution.tsx` | R-multiple histogram | 1 |
| `backtest_data/journal.db` | New journal database | 1 |
| `frontend/mockups/journal/01_calendar_overview.html` | Static mockup | 0 (this PR) |
| `frontend/mockups/journal/02_trade_detail.html` | Static mockup | 0 (this PR) |
| `frontend/mockups/journal/03_insights_dashboard.html` | Static mockup | 0 (this PR) |

---

**End of design.** Next step: build the three HTML mockups
(`frontend/mockups/journal/`), then begin Phase-1 implementation
behind a feature flag.
