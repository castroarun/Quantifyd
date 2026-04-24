# NWV — Nifty Weekly View · plan and live status

**Owner:** Arun · **Author:** Claude Opus 4.7 (1M)
**Started:** 2026-04-24 (Fri evening)
**Instrument:** NIFTY weekly options (expiring the following **Tuesday**)

This is the single source of truth for the NWV system. Updated after every
commit so a fresh session or the user can pick up mid-stream.

---

## System summary

A rule-based view generator for NIFTY weekly options, anchored on:

1. **Weekly CPR** (computed from last week's H/L/C)
2. **Monday's first 30-min candle** (09:15 → 09:45) — close vs CPR + body direction
3. **Gap %** — Mon open vs last Fri close

Produces a **final view** in one of: `bearish`, `neutral_to_bearish`,
`neutral`, `neutral_to_bullish`, `bullish`, `ignore`, with conviction score.
Each view maps to a trade template (strangle / debit spread / strangle +
optional short OTM call), a strike range based on weekly pivots, and a
mandatory time stop.

Expiry is the **following Tuesday**, so the trade holds Mon → Fri NRML and
exits Friday 15:15 — leaves one trading day before gamma blast-off.

---

## Decisions locked with user (2026-04-24)

| # | Decision | Value |
|---|---|---|
| 1 | Weekly CPR wide threshold | ≥ 0.80% of spot |
| 2 | Weekly CPR narrow threshold | ≤ 0.40% of spot |
| 3 | Gap "considerable" threshold | ≥ 0.70% of prev close |
| 4 | Gap "small" threshold | 0.30–0.70% |
| 5 | Pivots | Weekly, all S2/S1/PP/R1/R2 |
| 6 | Inside-CPR candle handling | treat as `neutral` |
| 7 | Doji body | neutralize the view |
| 8 | Entry time | 09:45 Monday (no wait for 10:15 confirmation) |
| 9 | Monthly CPR override | **SOFT** (demote view by one step, do not kill) |
| 10 | Time stop | **Friday 15:15 IST** (Tuesday expiry ⇒ 1-day buffer) |
| 11 | Product | NRML (weekly hold, not MIS) |
| 12 | Uniform exit on Friday (winners + losers) | YES — uniform |
| 13 | Max concurrent positions | 1 (one view per week) |
| 14 | Lots per leg | 1 (v1) |
| 15 | Phase 0 scope | view-only dashboard + WhatsApp, NO orders |

---

## View matrix (symmetric, both sides)

### Bearish-side cases (1st candle close **below** CPR)

| CPR bucket | 1st candle body | Base view |
|---|---|---|
| narrow | any | IGNORE |
| wide | bearish | neutral_to_bearish |
| wide | bullish | neutral |
| wide | doji | neutral |
| normal | bearish | **bearish** |
| normal | bullish | neutral |
| normal | doji | neutral |

### Bullish-side cases (1st candle close **above** CPR)

Mirror of the above: wide+bullish ⇒ neutral_to_bullish, normal+bullish ⇒
**bullish**, etc.

### Inside-CPR cases (1st candle closes between BC and TC)

Any CPR bucket ⇒ `neutral` (unless narrow → IGNORE).

### Gap dampener (applied after base view)

If |gap%| ≥ 0.70% in the **same direction** as the view:

| Base view | After dampening |
|---|---|
| bearish (gap down ≥ 0.70%) | neutral_to_bearish |
| neutral_to_bearish (gap down ≥ 0.70%) | neutral |
| bullish (gap up ≥ 0.70%) | neutral_to_bullish |
| neutral_to_bullish (gap up ≥ 0.70%) | neutral |
| neutral | unchanged |

Counter-direction gaps (gap up on bearish setup) don't dampen.

---

## Enhancements from research (fold into Phase 0 dashboard)

### Tier 1 (included in Phase 0)

- **1.1 Daily CPR confluence** — Mon-Fri daily; confirms or conflicts with weekly view. Displayed on dashboard; not applied as automatic modifier in Phase 0.
- **1.2 Pivot cluster detection** — weekly + daily pivots within 0.25% of each other form "stacked levels." Highlighted on dashboard for strike-selection guidance.
- **1.3 ADX(14) daily at 09:45** — disambiguates neutral into chop (< 20) vs trend-building (> 25). Shown on dashboard; used as a conviction modifier.
- **1.4 India VIX percentile (60-day)** — premium-richness filter. Shown.
- **2.5 First-candle quality** — range % and wick position used for conviction bumping.

### Tier 2 (included in Phase 0)

- **2.1 Monthly CPR overlay (SOFT)** — if spot beyond monthly TC/BC, demote opposing-direction views by one step. Applied as a modifier.
- **2.4 Friday 15:15 mandatory exit** — rule for future trading phases; dashboard shows the time stop.

### Deferred to Phase 1+

- 2.2 Max pain
- 2.3 Carry-in bias
- All Tier 3

---

## Final view pipeline (as built)

```
Monday 09:45 IST
  │
  ├─ Base view          = matrix_lookup(weekly_cpr_bucket, first_candle_pos, first_candle_body)
  │
  ├─ Gap dampener       = apply_gap_dampener(base, gap_pct, gap_direction)
  │
  ├─ Monthly override   = soft_demote_if_against_macro(view, spot, monthly_cpr)
  │
  ├─ Conviction score:
  │     start at 3 (baseline)
  │     + first_candle_range adjustments (Tier 2.5)
  │     + wick position adjustments (Tier 2.5)
  │     + ADX modifiers (Tier 1.3)
  │     + VIX regime modifiers (Tier 1.4)
  │     clamp to [0, 5]
  │
  └─ Output:
       {
         final_view, conviction (0-5),
         instrument_choice (strangle / put_debit_spread / call_debit_spread / none),
         expected_range (low, high),
         stacked_supports, stacked_resistances,
         monday_gap_pct, first_candle, vix_pct,
         adx, monthly_override_applied, time_stop
       }
```

---

## Phase 0 scope (this week)

- Compute the view
- Display on `/app/nwv` dashboard
- WhatsApp alert at 09:46 Monday with full breakdown
- Log every view to `nwv_views` table for backtest-style review
- **No trades placed** (automate Phase 1)

---

## Deliverables and status

Symbols: 🔴 not started · 🟡 in progress · 🟢 done

### F0 — Plan doc
**Status:** 🟡 (this file)

### F1 — `services/nwv_db.py`
**Status:** 🟢 done · commit `67eb2b5`
Tables:
- `nwv_weekly_state` — one row per week: prev-week H/L/C, weekly pivots, CPR levels, CPR width bucket, week_start, generated_at
- `nwv_views` — one row per Monday: first-candle data, gap metrics, all enhancement inputs, final view, conviction, expected range, notes

### F2 — `services/nwv_engine.py`
**Status:** 🟢 done · commit `67eb2b5` · 24 sanity tests pass
Classes: `NwvEngine` with methods:
- `compute_weekly_state(week_start_date)` → populates `nwv_weekly_state`
- `compute_view()` → reads last-Friday's close, today's first 30-min, VIX, ADX etc., returns full view dict and persists to `nwv_views`
- `get_latest_view()` → retrieves for dashboard/API

### F3 — scheduled jobs in `app.py` + data-source helpers
**Status:** 🟢 done · commit `af8493c`
Also introduced `services/nwv_data.py` with Kite/yfinance adapters
(weekly + monthly HLC, first 30-min candle, VIX + percentile, daily
pivots, ADX Wilder 14).
- Sunday 22:00 IST cron: compute next week's weekly state (CPR + pivots from last week)
- Monday 09:46 IST cron: compute and persist view; fire WhatsApp notification
- Monthly-start cron: compute monthly CPR (reused across 4-5 weekly views)

### F4 — API endpoints
**Status:** 🟢 done · commit `af8493c`
  - `GET  /api/nwv/view`
  - `GET  /api/nwv/weekly-state`
  - `GET  /api/nwv/views-history?n=20`
  - `POST /api/nwv/recompute` (manual trigger)
- `GET /api/nwv/view` → latest view JSON
- `GET /api/nwv/weekly-state` → current week's pivots, CPR, stacked-level candidates
- `GET /api/nwv/views-history?n=20` → recent views log

### F5 — React page `/app/nwv`
**Status:** 🟢 done · commit `8e73636`
Route + sidebar entry wired. No new npm deps. Typecheck clean.
Simple single-page component showing the view breakdown in the exact layout
user approved:

```
NIFTY Weekly View · 2026-04-28
══════════════════════════════
Base view           bearish
Gap tier            none
Monthly CPR         spot inside (no override)
Carry-in            last Fri inside PP–R1 (no bias)  [Phase 1 only]
VIX percentile      42 (neutral regime)
ADX (daily)         22 (building trend)
First-candle range  0.28% (normal)
First-candle wick   body in bottom 22%
Stacked supports    23,740 ←  wS1 + dS2 cluster
Stacked resistances 24,020 ←  wR1 + dR1 cluster
Max pain            [deferred]
─────────────────────────────
FINAL VIEW          bearish
Conviction          4/5
Suggested trade     debit put spread, 24,000/23,700
Expected range      below 23,740
Time stop           Fri 15:15
```

### F6 — WhatsApp notification
**Status:** 🟢 done · commit `af8493c` (bundled with F3)
Fires at end of `_nwv_compute_view()`. Compact card with CPR bucket,
gap tier, 1st candle, VIX/ADX, instrument + expected range + time
stop + dashboard URL. `send_alert` priority=high.
Reuse `services/notifications.py`. Fires once at 09:46 Monday with compact
view card (≤ 500 chars) pointing to `/app/nwv` for full breakdown.

---

## Deploy plan

Same pattern as ORB bundle:
1. Commit + push each logical piece
2. User pulls on VPS outside market hours
3. `systemctl restart quantifyd.service`
4. Wait for Monday 09:46 IST fire — verify dashboard + WhatsApp

---

## Rollback

Same as ORB bundle — `git reset --hard <commit-before-nwv>`, restart.

---

## Progress log (append-only)

### 2026-04-24 · Fri · session 1
- User approved the Phase 0 dashboard layout.
- Locked soft Monthly override, Friday 15:15 time stop (Tuesday expiry).
- Plan doc landed. Starting F1 (DB layer).
- `18:30 IST` F1 + F2 landed · `67eb2b5`. Engine with 24 tests passing.
- `18:45 IST` F3 + F4 + F6 landed · `af8493c`. Sunday-22:00 weekly
  state build, Monday-09:46 view compute + WhatsApp, 4 API endpoints.
- `19:00 IST` F5 landed · `8e73636`. `/app/nwv` React dashboard +
  sidebar entry. Typecheck clean.
- **All Phase 0 deliverables complete.** Ready to push + deploy.

Deploy next — user to pull on VPS outside market hours and rebuild
the frontend. Sunday 22:00 cron will pre-populate the next week's
state; Monday 09:46 will fire the first live view.

<!-- APPEND FUTURE ENTRIES BELOW -->
