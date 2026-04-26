# Small/Micro-Cap Intraday ORB Breakout — Live Status

**Owner:** research/19_smallcap_intraday/
**Spec:** `data/future_plans.json` → plan id `smallcap-intraday-orb`
**Created:** 2026-04-26

---

## 1. Goal + scope

Build a separate intraday breakout system targeting small/micro-cap NSE stocks
(higher intraday volatility than the existing ORB universe), with cash-delivery
sizing (no leverage), 30-min opening range, 0.5% slippage cost, Rs 50K hard
position cap, and Rs 3L total capital. Period: 2024-03-18 → 2026-03-12 (matches
local 5-min data window, ~500 trading days).

Success = Out-of-sample PF ≥ 1.20, Sharpe ≥ 0.8, MaxDD ≤ 15%, 2-4 trades/day
average. First-pass goal in this scaffold is to (a) build a defensible
universe, (b) implement the engine per the spec, (c) prove the engine runs
end-to-end on 5 stocks. Full sweep is the next session.

---

## 2. Plan

### Step 1+2+3 — Universe selection (`scripts/build_universe.py`)

| # | Filter | Rule | Expected impact |
|---|--------|------|-----------------|
| 1 | Base candidates | All symbols in `market_data_unified` 5-min timeframe with ≥500 bars since 2024-03-18, **excluding** the 15 ORB stocks and **excluding** index symbols (BANKNIFTY, NIFTY50). | Spec calls for Smallcap 250 + Microcap 250 base; the local DB does NOT have 5-min data for true small-caps (only 81 5-min symbols, all Nifty 500). **See "Spec deviation" below.** |
| 2 | Turnover filter | Avg daily turnover (close × volume) over last 30 trading days ≥ Rs 50 Cr. | Drops illiquid names. |
| 3 | Circuit-history filter | Count of daily \|move\| ≥ 5% over last 60 trading days < 3. | Drops circuit-prone names. |

Output: `results/universe_selection.csv` (full diagnostics) +
`results/universe.csv` (final symbols, one per line).

### Step 4 — Backtest (`scripts/run_smallcap_backtest.py`)

- 5-min bars, 2024-03-18 to 2026-03-12
- 30-min OR (09:15-09:45)
- 5-min CLOSE breakout (long > OR_high, short < OR_low)
- Filters: VWAP align, RSI not extreme (skip if >75 long / <25 short), gap >2% skip
- Drop CPR width filter (small caps have inherently wide CPR)
- Last entry 13:30, EOD 15:18
- SL = OR opposite edge, target = 1.5R **or** 2R (2 variants this pass)
- Trail at 14:00: if in profit, lock entry + 0.5R (V9t_lock50 pattern)
- Cost: 0.5% round-trip
- Sizing: Rs 2,400 risk/trade, max 6 concurrent, max 50K notional/position
- Capital: Rs 3L, no leverage

Output per variant: `results/summary.csv`, `results/trades.csv`,
`results/equity_<variant>.csv`.

### Spec deviation — flagged for human review

Spec says base universe = Nifty Smallcap 250 + Microcap 250. Local
`backtest_data/market_data.db` contains 5-min bars for only **81 symbols** —
75 of which are Nifty 500 large/mid-caps, 6 are indices/recent IPOs. There is
**zero overlap** with genuine small/microcap names.

The universe builder honors the user's stated fallback: "all 5-min stocks not
in Nifty 500 + ORB list". After excluding ORB and indices, that fallback
yields only 4 names (COFORGE, DELHIVERY, PAYTM and one more — recent IPOs not
in the static N500 CSV). To produce a workable list of `~60-100 names` per
the spec, the builder also accepts a `--include-mid` flag (default ON for
this dev pass) which keeps Nifty-500-but-not-ORB 5-min symbols as proxy
candidates. **This is a mid-cap proxy, not true small-cap.** A future task
should download 5-min bars for the actual Smallcap 250 / Microcap 250 lists
via `services/data_manager.py`.

---

## 3. Status

| Step | State | Notes |
|------|-------|-------|
| Scaffold dirs created | DONE | `research/19_smallcap_intraday/{scripts,results,logs}` |
| `SMALLCAP-STATUS.md` | DONE | This file |
| `scripts/build_universe.py` | DONE | Runnable; outputs CSV diagnostics |
| `scripts/run_smallcap_backtest.py` | DONE | Runnable with `--limit N` for dev tests |
| Universe filter run (full) | NOT STARTED | Awaiting human review of Spec deviation |
| Backtest dev-run (5 stocks) | DONE | Sanity check that engine produces trades |
| Full-universe backtest sweep | NOT STARTED | Next session, after universe review |

Heartbeat / log paths (when sweeps run):
- Heartbeat: `logs/heartbeat.txt`
- Run logs: `logs/build_universe.log`, `logs/backtest_<variant>.log`

---

## 4. Crash recovery

This scaffold is **not** running anything in the background. There are no
spawned processes to babysit.

### To resume (no Claude needed)

1. Review `results/universe_selection.csv` — confirm universe makes sense
   given the Spec deviation above.
2. Decide path forward:
   - **Path A** — accept mid-cap proxy: just run
     `python scripts/run_smallcap_backtest.py` with no `--limit` flag.
   - **Path B** — get true small-caps: run a Kite 5-min backfill for
     Smallcap 250 / Microcap 250 names first (use
     `services/data_manager.py.CentralizedDataManager.download_data()`
     with `timeframe='5minute'`, then re-run `build_universe.py`).
3. To re-run universe selection from scratch (idempotent):
   ```
   cd c:/Users/Castro/Documents/Projects/Covered_Calls
   python research/19_smallcap_intraday/scripts/build_universe.py
   ```
4. To run backtest dev pass (5 stocks):
   ```
   python research/19_smallcap_intraday/scripts/run_smallcap_backtest.py --limit 5
   ```
5. To run full sweep (all universe stocks, both 1.5R and 2R variants):
   ```
   python research/19_smallcap_intraday/scripts/run_smallcap_backtest.py
   ```

### Files NOT to touch

- `backtest_data/market_data.db` — read-only, shared with all other strategies.
- `data/nifty500_list.csv` — read-only universe reference.
- Anything outside `research/19_smallcap_intraday/`.

### Check whether anything is running

```
ps aux | grep run_smallcap_backtest    # Linux/WSL
tasklist | findstr python              # Windows
```

The scripts use `print(..., flush=True)` and tail the heartbeat at
`research/19_smallcap_intraday/logs/heartbeat.txt` — last line shows the
in-progress symbol/variant.

---

## 5. Final aggregation

After the full sweep runs, the deliverables are:

| File | Contents |
|------|----------|
| `results/universe_selection.csv` | All candidate stocks + filter diagnostics |
| `results/universe.csv` | Final universe list (one symbol per line) |
| `results/summary.csv` | Per-stock + portfolio metrics, both variants |
| `results/trades.csv` | Full per-trade log, both variants |
| `results/equity_target_1.5R.csv` | Daily equity curve, 1.5R variant |
| `results/equity_target_2.0R.csv` | Daily equity curve, 2R variant |

Ranking criteria for variants (from spec, in priority order):
1. OOS PF ≥ 1.20
2. OOS Sharpe ≥ 0.8
3. OOS MaxDD ≤ 15%
4. 2-4 trades/day average

The walk-forward train/test split (train 2024-Q2..2025-Q1, test
2025-Q2..2026-Q1) is **not** implemented in this first pass — it lives in
the next iteration once the basic engine is validated end-to-end.
