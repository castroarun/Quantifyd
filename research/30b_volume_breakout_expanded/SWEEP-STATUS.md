# Volume-Breakout Expanded Sweep (79 stocks) — STATUS

**Owner:** Claude (driving) / Arun (deciding)
**Created:** 2026-04-30
**Authoritative resume point if Claude's context is lost.**

---

## 1. Goal

Identify universe-wide "volume leaders" — stocks that consistently exhibit
the volume-confirmed first-candle breakout pattern with a tradable edge.
Expansion of the prior 10-stock run at `research/30_volume_breakout/` to
all 79 stocks with 5-min intraday data, plus a 10-min timeframe added
to the variant grid.

User intent: find a deployable subset of stocks for an intraday volume-
breakout strategy (or confirm the signal is better used as a filter on
top of an existing strategy).

## 2. The signal (one-line)

The first 5-min OR 10-min OR 15-min candle of today's session **closes
above the previous day's high** (LONG) or **below the previous day's low**
(SHORT) **with elevated volume** (today's first-bar volume > vol_mult ×
20-day average of first-bar volume).

Optional filters (variant-tested): minimum gap%, 5-min RSI confirm.

Exit policies (13 tested in parallel per signal): `T_NO` (hold to 15:25),
`T_HARD_SL` (first-bar opposite extreme), `T_ATR_SL_{0.3,0.5,1.0}`,
`T_CHANDELIER_{1.0,1.5,2.0}`, `T_R_TARGET_{1R,1.5R,2R,3R}`, `T_STEP_TRAIL`.

## 3. Universe

**Cohort A** — 10 stocks, full 5-min history since 2018-01-01:
RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC,
KOTAKBANK, HINDUNILVR

**Cohort B** — 69 stocks, 5-min history since 2024-03-18:
ADANIENT, ADANIPORTS, AMBUJACEM, APOLLOHOSP, ASIANPAINT, AXISBANK,
BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BANKBARODA, BEL, BPCL, BRITANNIA,
CHOLAFIN, CIPLA, COALINDIA, COFORGE, COLPAL, CUMMINSIND, DABUR,
DELHIVERY, DIVISLAB, DLF, DRREDDY, EICHERMOT, FEDERALBNK, GAIL,
GODREJPROP, GRASIM, HAL, HAVELLS, HCLTECH, HDFCLIFE, HEROMOTOCO,
HINDALCO, IDFCFIRSTB, INDUSINDBK, IOC, IRCTC, JINDALSTEL, JSWSTEEL,
LT, M&M, MARICO, MARUTI, MCX, MUTHOOTFIN, NESTLEIND, NTPC, ONGC,
PAYTM, PERSISTENT, PIDILITIND, PNB, POWERGRID, SBILIFE, SHREECEM,
SIEMENS, SUNPHARMA, TATACONSUM, TATAPOWER, TATASTEEL, TECHM, TITAN,
TRENT, ULTRACEMCO, VEDL, VOLTAS, WIPRO

Period: per-stock available range, capped at 2026-03-25 (data end).

## 4. Variant grid

- 79 stocks
- timeframes ∈ {5min, 10min, 15min} — 10-min ADDED in this expanded run
- vol_mult ∈ {1.5, 2.0, 3.0}
- gap_pct ∈ {0%, 0.3%, 0.5%, off}
- RSI_filter ∈ {off, on (≥60 long / ≤40 short)}
- direction ∈ {long, short}

Total cells = 79 × 3 × 3 × 4 × 2 × 2 = **11,376 cells**. Cells with
n<10 signals get dropped from the ranking (statistically meaningless).

## 5. Live status (running log — update at every state transition)

| Date/time | Event | Notes |
|---|---|---|
| 2026-04-30 11:51 IST | First subagent launched | Created folder structure, scripts |
| 2026-04-30 12:10 IST | Subagent exited prematurely after 8/79 stocks | Claimed a "watcher" but no process actually running |
| 2026-04-30 12:10 IST | Resumed via direct background bash (bash ID `bx5ulacha`) | Picked up at stock 9 via CSV skip-set |
| 2026-04-30 13:24 IST | Progress 47/79 — JSWSTEEL just done | 73.9 min elapsed, 102,118 signal rows logged |
| | (live in progress) | |

**Current background job:** `bx5ulacha`
**Estimated completion:** ~2.5 hours total wall (started 12:10 IST → est. 14:40 IST done)
**Pace:** ~110s/stock average (Cohort A long-history stocks slower at ~380s; Cohort B faster at ~70s)

To check progress yourself any time:
```bash
tail -5 research/30b_volume_breakout_expanded/results/run.log
wc -l research/30b_volume_breakout_expanded/results/volbreakout_signals.csv
```

## 6. Crash recovery — how Arun resumes without Claude

The script is **resumable**. It reads existing `volbreakout_signals.csv`,
builds a `(symbol, tf, variant, direction, date)` skip-set, and only
computes cells not yet logged.

### A) Check what finished

```bash
# Last completed stock:
tail -3 research/30b_volume_breakout_expanded/results/run.log

# Total signal rows logged so far:
wc -l research/30b_volume_breakout_expanded/results/volbreakout_signals.csv

# Is the background process still alive?
# If you started via "bash run_in_background", process won't show in ps
# (it's a Claude-managed background). Just check the log mtime:
ls -la research/30b_volume_breakout_expanded/results/run.log
# If mtime is more than 5 min old, the process likely died — resume it.
```

### B) Resume the sweep

```bash
cd /c/Users/Castro/Documents/Projects/Covered_Calls
python research/30b_volume_breakout_expanded/scripts/run_volbreakout_expanded.py
# Will skip cells already in volbreakout_signals.csv and continue from
# the next unfinished stock/variant
```

This is safe to run concurrently if you accidentally start two — the
skip-set dedupe handles it. CSV writes are append-only and flushed
per-cell.

### C) After all 79 stocks finish, run aggregation

If the script's main loop finishes signal generation but crashes on the
aggregation/markdown step, run with the aggregate-only flag:

```bash
python research/30b_volume_breakout_expanded/scripts/run_volbreakout_expanded.py --aggregate-only
```

This rebuilds `volbreakout_ranking.csv`, `volume_leaders.csv`, and
`RESULTS.md` from the existing signals CSV without re-running signal
generation.

### D) Files NOT to touch during the sweep

- `results/volbreakout_signals.csv` (being appended; may be 50-100 MB)
- `results/run.log` (live progress log)
- This `SWEEP-STATUS.md` (will be updated at state transitions)

### E) Files safe to inspect

- `scripts/run_volbreakout_expanded.py` — read-only review
- `results/RESULTS.md` (only exists after aggregation completes)
- `results/volbreakout_ranking.csv` (only exists after aggregation completes)

## 7. Outputs (when finished)

| File | Purpose | Committable? |
|---|---|---|
| `scripts/run_volbreakout_expanded.py` | Sweep runner | yes |
| `SWEEP-STATUS.md` | This file | yes |
| `results/run.log` | Per-stock progress | yes |
| `results/volbreakout_signals.csv` | Per-signal × per-policy results (~50-100 MB) | NO — gitignored |
| `results/volbreakout_ranking.csv` | Per-cell aggregate (one row per cell) | yes |
| `results/volume_leaders.csv` | Per-stock leaderboard (best variant + best exit) | yes |
| `results/RESULTS.md` | Final report with top picks + promote candidates | yes |

## 8. What "promote candidates" means

A stock makes the promote list if it passes ALL three robustness gates:

1. **Best-cell Sharpe ≥ 0.5** — has a clear edge, not noise
2. **n ≥ 15** in that best cell — enough sample size to trust
3. **Robust across ≥ 3 variants** — Sharpe ≥ 0.3 in 3 or more cells

These are the stocks that genuinely exhibit a volume-leader pattern
worth deploying with real capital. Anything below the gate is either
overfit (small n), one-trick-pony (single-variant), or noise.

## 9. Pre-run benchmark (10-stock prior run, research/30_volume_breakout/)

For comparison once the expanded run completes:

| Stock | Sharpe | n | Mean% | WR% | Notes |
|---|---|---|---|---|---|
| RELIANCE 15m long | **1.045** | 11 | +1.09 | 90.9 | Top of prior run; vm=2.0, gap≥0%, RSI on |
| BHARTIARTL 15m long | 0.610 | 11 | +0.78 | 81.8 | vm=1.5, gapoff, RSI on |
| TCS 15m short | 0.523 | 12 | +0.66 | 75.0 | vm=1.5, gap≥0.5%, no RSI |
| INFY 5m long | 0.513 | 10 | +0.54 | 80.0 | vm=3.0, gapoff, RSI on |
| SBIN 15m long | 0.485 | 14 | +0.77 | 78.6 | vm=2.0, gap≥0.5%, no RSI |

Compare these to the expanded universe results to see whether new
stocks beat RELIANCE.

## 10. Final deliverables (will be added once aggregation completes)

This section will be populated when the sweep finishes:

- Total signals fired
- Top 10 configurations across all 79 stocks
- Top 15 "volume leaders" leaderboard
- Promote candidates (passing the robustness gate)
- Honest read: deployable subset vs filter-only

---

**Last updated:** 2026-04-30 13:24 IST
**Update cadence:** at every state transition (subagent exit, full-sweep
complete, aggregation complete, errors)
