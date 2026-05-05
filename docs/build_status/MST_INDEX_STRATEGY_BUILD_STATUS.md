# MST Index Strategy — Build Status

**STATUS: DEPLOYED ON VPS — LIVE AT http://94.136.185.54:5000/app/mst (paper mode)**

Spec: [docs/Design/MST-INDEX-STRATEGY-DESIGN.md](../Design/MST-INDEX-STRATEGY-DESIGN.md)
Research backing: `research/35_*` and `research/36_*`
Started: 2026-05-05 18:18 IST (Tuesday post-market)
Operator: Claude Opus 4.7 (1M context) driving autonomously per user authorization

## 1. The Ask

**What the user asked (verbatim):**
> ok lets proceed!! you have all controls.... I also need a manual switch option to paper trade/live/switch off (refer other pages)

**What I'm building:**
A complete live-trading + paper-trading + observability system for the MST/CST/Pyramid NIFTY strategy spec'd in the design doc. New page at `http://94.136.185.54:5000/app/mst`. Mode switch (off/paper/live) mirroring the KC6 + NAS pattern.

## 2. Phases

| Phase | Deliverable | Status |
|---|---|---|
| 1 — Foundation | trading_calendar.py + holidays JSON + mst_db.py + mst_engine.py (signal-only) + replay test | RUNNING |
| 2 — Backend integration | Flask API + NasTicker hook + scheduler + mode toggle endpoint | pending |
| 3 — Order execution | mst_executor.py + Kite order placement + margin pre-check | pending |
| 4 — Frontend | Mst.tsx + Mst.module.css + Sidebar entry + routing | pending |
| 5 — Deploy & shadow | Push to VPS, run shadow mode 1 week | pending |

## 3. Build log

| Time IST | Event | Notes |
|---|---|---|
| 2026-05-05 18:18 | Build started | Confirmed post-market window for backend deploys |
| 2026-05-05 18:20 | Studied KC6 + NAS mode-switch pattern | Two booleans (`enabled` + `paper_trading_mode`) compose into Off/Paper/Live UI |
| 2026-05-05 18:30 | Phase 1: trading_calendar.py + nse_holidays_2026.json built + smoke-tested | Holiday-shifted weekly expiry math validated for May 12 + Apr 14 (holiday) cases |
| 2026-05-05 18:40 | Phase 1: mst_db.py with 5-table schema + smoke-tested | Bars, events, positions, orders, equity tables |
| 2026-05-05 18:55 | Phase 1: mst_engine.py with full state machine + signal logic | 6-state FSM with break-of-extreme entry, multi-CST policy, pyramid trigger D AND B |
| 2026-05-05 19:05 | Phase 1: replay test passes against 6.3-yr historical data | flip_armed=325 vs 322 expected, flip_activated=302 (exact match), pyramid_fired=145, condor_built=441, flip_discarded=22 — all within tolerance of research/36 numbers |
| 2026-05-05 19:08 | User requested ORB switch-off | Toggled enabled=False on VPS via API + persisted in config.py |
| 2026-05-05 19:20 | Phase 3: mst_executor.py with paper + live order paths | Paper simulates fills; live calls kite.place_order with NRML, LIMIT-at-mid → MARKET fallback at 30s |
| 2026-05-05 19:25 | Phase 2: MST_DEFAULTS in config.py (30 keys) | All thresholds, timings, lot size, mode flags |
| 2026-05-05 19:35 | Phase 2: mst_bootstrap.py + 30-min aggregator + NasTicker subscriber pattern | NasTicker patched with additional_subscribers list; aggregator buckets 5-min into 30-min |
| 2026-05-05 19:40 | Phase 2: 8 Flask `/api/mst/*` routes + scheduler cron | state, bars, events, positions, equity-curve, scan, kill-switch, toggle-mode; T-1 check daily 15:25 |
| 2026-05-05 20:00 | Phase 4: Mst.tsx + Mst.module.css + types + sidebar entry + route | Three-button mode switch (Off/Paper/Live) + Kill switch button; metrics, positions, events sections; full strategy rules block |
| 2026-05-05 20:05 | Frontend npm build clean (88 modules, 364 kB JS) | Static bundle written to static/app/, served by Flask under /app/* |
| 2026-05-05 20:10 | All imports + syntax checks pass | Ready to commit and deploy |
| 2026-05-05 20:15 | Pushed to GitHub | Commit 5d16249 (143 files, ~400k insertions including research artifacts) |
| 2026-05-05 20:18 | VPS git pull + restart | Service active; ORB came up with enabled=False per persisted config |
| 2026-05-05 20:19 | Hot-fix: missing `date` import in app.py for /api/mst/state | Pushed dcdf523, restarted |
| 2026-05-05 20:25 | NIFTY 30-min downloaded to VPS market_data.db | 39 API calls, 20,372 bars from 2020-01-01 |
| 2026-05-05 20:26 | Restart + verify: engine seeded with 250 bars, MST page returns HTTP 200 | Mode endpoint toggles paper/live correctly; state machine = NO_POSITION (correct — engine only enters on next live MST flip + break-of-extreme) |
| 2026-05-05 20:28 | Public URL http://94.136.185.54:5000/app/mst returns 200 | Live for user testing |

## 4. Crash Recovery

If interrupted, the user can resume by:
1. Reading this STATUS doc to see what phase is in progress
2. Running `git log --oneline -20` to see what's been committed
3. The design doc is the canonical spec — work is always traceable to a spec section
4. Each phase is committed atomically with a descriptive message; no partial commits

## 5. Files

(Updated as build progresses.)

| File | Purpose | Phase | Status |
|---|---|---|---|
| `services/trading_calendar.py` | NSE trading-day calendar with holiday awareness | 1 | pending |
| `config/nse_holidays_2026.json` | NSE 2026 holiday list | 1 | pending |
| `services/mst_db.py` | SQLite schema (5 tables) + helpers | 1 | pending |
| `services/mst_engine.py` | Signal computation + state machine | 1 | pending |
| `tests/test_mst_engine_replay.py` | Replay against 6.3-yr NIFTY 30-min | 1 | pending |
| `services/mst_executor.py` | Order placement (paper/live) | 3 | pending |
| `app.py` (additions) | `/api/mst/*` routes + scheduler | 2 | pending |
| `frontend/src/pages/Mst.tsx` | React page | 4 | pending |
| `frontend/src/pages/Mst.module.css` | Page styles | 4 | pending |
| `frontend/src/components/Sidebar/Sidebar.tsx` (edit) | Add MST nav entry | 4 | pending |
| `frontend/src/App.tsx` (edit) | Add `/app/mst` route | 4 | pending |
| `frontend/src/api/types.ts` (edit) | Add MST type definitions | 4 | pending |
| `config.py` (edit) | Add `MST_DEFAULTS` dict | 2 | pending |

## 6. Mode switch — three-way (Off / Paper / Live)

Mirrors KC6/NAS pattern. Two booleans in `MST_DEFAULTS`:
- `enabled: bool` — system active vs halted
- `paper_trading_mode: bool` — when enabled, this controls paper vs live

UI shows three buttons: **Off** / **Paper** / **Live**. Single endpoint `/api/mst/toggle-mode` accepts `{mode: 'off'|'paper'|'live'}` and sets both booleans accordingly:

| Mode | enabled | paper_trading_mode | Behavior |
|---|---|---|---|
| `off` | false | (irrelevant) | No signal evaluation, no entries, no alerts |
| `paper` | true | true | Full signal evaluation + alerts, but orders are simulated (logged, not placed) |
| `live` | true | false | Full signal evaluation + alerts + REAL orders via Kite |

Default at first deploy: **paper**. User flips to **live** after shadow validation.

## 7. Findings / decisions during build

(Updated as build progresses.)
