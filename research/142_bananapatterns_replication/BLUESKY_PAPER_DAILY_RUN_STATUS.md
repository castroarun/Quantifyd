# BlueSky ATH-Breakout ₹10L Paper Book — G5 Soak (adopted trail-20 spec)

STATUS: RUNNING (cron since 2026-09-02; first real run 18:40 IST 2026-09-02)

## The Ask

G5 gate for research/142's adopted taxable spec: prove the backtest tracks reality on
live quotes before any capital. Arun's go: 2026-09-02 ("go"). Intended eventual use:
50-50 monthly-rebalanced blend with the live momentum-paper book (the study's capstone).

## The Base (exact spec being soaked)

₹10,00,000 paper. Universe: NSE dailies in market_data.db (≥260 rows, active ≤14d),
20d-median TV ≥ ₹5cr, ETFs excluded, NO mcap floor. Setup: prev close within 20% of the
all-time-high close and below it; IBD-RS percentile (2×r63+r126+r189+r252, ranked over
eligibles, as of t−1) ≥ 70. Signal: today's close > prior ATH-close, only while
NIFTYBEES ≥ its 200-DMA. Entry NEXT day, buy-stop semantics: fill = open if open ≥
pivot, else pivot if high ≥ pivot, else MISS (recorded). Exits at close: ≤0.92×buy
(stop) or < SMA20 (trail). 8 slots, 18.75% of NAV, RS-desc selection, 25bps/side.

## Plumbing

| Piece | Where |
|---|---|
| Service | `services/bluesky_paper.py` (self-guards: aborts before 15:35 IST; `--dry` anytime) |
| Cron | 18:40 IST Mon–Fri → log `/tmp/bluesky_paper.log` (backup of pre-install crontab in /tmp/ct_bak_bluesky_*) |
| State | `backtest_data/bluesky_paper_state.json` (lockfile + atomic replace — statefile-race lesson applied) |
| Data | reads canonical DB; if today's bars missing, merges kite.ohlc quotes IN MEMORY (canonical DB never written) |
| UI | `/app/bluesky-paper` ← `static/app/bluesky_paper.json` (no backend restart in the whole pipeline) |
| Registry | Strategies index row `bluesky-paper`; ops-center group + dated review 2026-12-05; docs/LABS_AND_JOBS_REFERENCE.md |

## Pass criterion (pre-registered 2026-09-02, review 2026-12-05)

Over ~a quarter: per-trade return distribution consistent with the trail-20 backtest
ensemble; realized fills within ~0.5% of modeled max(open, pivot); pivot miss-rate and
gap costs documented; gate behaviour correct. Then decide G6 sizing or park.

## Crash recovery

- Check health: `tail /tmp/bluesky_paper.log`; state file mtime; `/app/bluesky-paper` page.
- Re-run manually (idempotent per day is NOT guaranteed — one run per trading day; if a
  run died mid-way the lockfile may remain: remove `backtest_data/bluesky_paper_state.lock`
  after confirming no process, then rerun `venv/bin/python services/bluesky_paper.py`).
- Preview without writing: `venv/bin/python services/bluesky_paper.py --dry`.

## Log

| Date | Event |
|---|---|
| 2026-09-02 | Built, dry-run OK (1,219 live quotes, gate OK), cron installed (67→69 lines verified), page live, pushed `62655e1`/`c7a1b3f` |

## SOAK CLOCK RESTARTED 2026-09-03 — spec revision (Arun: "Proceed")

The book was re-seeded on the revised spec: **16 slots @6.25%, NO market gate**
(SMA200 retired after the gate audit + bake-off; DD10 evaluated, not adopted —
see GATE_BAKEOFF_DAILY_SWEEP_STATUS.md §7). The 2026-09-02 seed's 5 open positions
were spec-invalid (entered by the NaN-dead gate) and were replaced by the correct
15-position portfolio (median seed 5, 1,310 backfilled trades, momentum-NAV parity
rescale). Arun's ₹2.5L deposits carried with original timestamps; dividend HWM
re-anchored at ₹11,35,026. **The Dec-05 soak review judges fills/tracking from
03-Sep-2026 forward on the new spec** — pass criterion unchanged (fills within
~0.5% of modeled, miss-rate, distribution consistency). Note: with the gate retired
the book now emits buy-stops every night there are signals (21 pending on day one).
