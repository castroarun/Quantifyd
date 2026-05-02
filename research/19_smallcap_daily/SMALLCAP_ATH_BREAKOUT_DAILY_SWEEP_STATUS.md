# Small/Micro-Cap Daily ATH/52w Breakout — Live Status

**Owner:** `research/19_smallcap_daily/`
**Spec:** `data/future_plans.json` -> plan id `smallcap-daily-ath-breakout`
**Created:** 2026-04-26
**Pivot note:** Previously named `19_smallcap_intraday`. The 5-min intraday
scaffold (`build_universe.py`, `run_smallcap_backtest.py`) is **PARKED**
and removed from this folder — local 5-min DB lacked genuine small/micro-
cap symbols. The folder is now reorganised around DAILY bars where we
have 1,623 symbols since 2018+. This file replaces `SMALLCAP-STATUS.md`.

---

## 1. Goal + scope

Build a daily-bar EOD breakout scanner for the small/micro-cap universe
that sits *outside* the Nifty 500 already covered by research/17 (the
project's only walk-forward-validated edge). Same family of rules as
research/17 D1_fixed_25_8 (252-day breakout + volume + 200-SMA regime,
fixed 25% target, 8% backstop), applied to a quality-filtered small-cap
universe with stricter volume confirmation and higher cost assumption.

Period: **2018-01-01 to 2025-12-31** (~8 yr). Capital Rs 10 L.

Pass criteria (matches spec gates):
- OOS Profit Factor >= 1.20
- OOS Sharpe >= 0.8
- OOS MaxDD <= 30%
- Trade frequency 30-100/yr across the curated universe

---

## 2. Plan

### Step 1 — Universe selection (`scripts/build_daily_universe.py`)

| # | Filter | Rule |
|---|--------|------|
| 1 | Base candidates | All `market_data_unified` symbols, `timeframe='day'`, >=1500 daily bars since 2018-01-01. |
| 2 | Exclude Nifty 500 | Drop names in `data/nifty500_list.csv` (research/17 covers them). |
| 3 | Turnover band | Trailing-60-day avg daily turnover (close * volume) in **[Rs 5 Cr, Rs 100 Cr]**. |
| 4 | Circuit / freeze filter | <4 days in last 60 trading days where \|today_close - prev_close\|/prev_close >= 10%. |
| 5 | Volatility floor | Trailing 30-day avg of ATR(14)/close >= 1%. |
| 6 | EQ-series filter | **SKIPPED** — no series metadata in DB. Documented gap. |

Output: `results/daily_universe_selection.csv` (per-candidate diagnostics)
+ `results/daily_universe.csv` (final list, one symbol per line).

### Step 2 — 6-variant sweep (`scripts/run_smallcap_daily_backtest.py`)

Baseline (matches `future_plans.json` spec):
- Close > 252-day high (excluding today)
- Volume >= 2.5x trailing 50-day avg
- Close > 200-day SMA
- Stop = max(entry - 2*ATR(14), entry * 0.92)
- Target = entry * 1.25 (fixed 25%)
- Risk 1%/trade, max 10 concurrent, notional cap = equity/10
- Cost 0.30% round-trip, 60-day max-hold safety

Six orthogonal variants:

| # | Name | Change vs baseline |
|---|------|--------------------|
| 1 | `baseline_252_25pct_8pct` | (none — exact spec) |
| 2 | `vol_2x` | Volume threshold 2.0x |
| 3 | `vol_3x` | Volume threshold 3.0x |
| 4 | `target_30pct` | Target 30% |
| 5 | `target_20pct` | Target 20% |
| 6 | `cost_50bps` | Round-trip cost 0.50% |

Output: `results/daily_summary.csv`, `results/daily_trades_<variant>.csv`,
`results/daily_equity_<variant>.csv`.

### Step 3 — Walk-forward (`scripts/walk_forward_daily.py`)

- Train: 2018-01-01 -> 2022-12-31
- Test:  2023-01-01 -> 2025-12-31
- Top 3 from sweep ranked by Sharpe with PF >= 1.0

Output: `results/daily_walk_forward.csv`

---

## 3. Status

| Step | State | Notes |
|------|-------|-------|
| Folder rename to `19_smallcap_daily` | DONE | Old `19_smallcap_intraday` renamed |
| Old intraday scripts removed | DONE | `build_universe.py`, `run_smallcap_backtest.py` deleted |
| `SMALLCAP-DAILY-STATUS.md` (this file) | DONE | Replaces `SMALLCAP-STATUS.md` |
| `scripts/build_daily_universe.py` | DONE | 117 symbols selected |
| Universe build run | DONE | See `results/daily_universe_selection.csv` (475 candidates -> 117 final) |
| `scripts/run_smallcap_daily_backtest.py` | DONE | 6-variant sweep completed (~5 min) |
| 6-variant sweep run | DONE | All 6 variants exceed full-period gates; see `results/daily_summary.csv` |
| `scripts/walk_forward_daily.py` | DONE | IS/OOS validation completed |
| Walk-forward run | DONE | All top 3 PASS OOS gates with strong margin |
| `FINDINGS.md` | DONE | Full writeup |

Heartbeat / log paths:
- Heartbeat (universe): `logs/universe_heartbeat.txt`
- Heartbeat (backtest): `logs/backtest_heartbeat.txt`
- Run log (sweep): `logs/run_phase1.log`
- Run log (walk-forward): `logs/walk_forward.log`

---

## 4. Crash recovery

Nothing is currently running in the background. All three runs (universe
build, 6-variant sweep, walk-forward) are complete. If you need to
re-run any step from scratch:

### To re-run universe selection (idempotent)
```
cd c:/Users/Castro/Documents/Projects/Covered_Calls
python research/19_smallcap_daily/scripts/build_daily_universe.py
```
Runtime ~22s. Overwrites `results/daily_universe_selection.csv` and
`results/daily_universe.csv`.

### To re-run the 6-variant sweep
```
python research/19_smallcap_daily/scripts/run_smallcap_daily_backtest.py
```
Runtime ~5 min. Overwrites `results/daily_summary.csv` and per-variant
trade/equity CSVs.

### To re-run walk-forward
```
python research/19_smallcap_daily/scripts/walk_forward_daily.py
```
Runtime ~2 min. Overwrites `results/daily_walk_forward.csv`.

### Check whether anything is running
```
tasklist | findstr python              # Windows
ps aux | grep run_smallcap_daily       # Linux/WSL
```

Tail the heartbeat files:
```
tail -1 research/19_smallcap_daily/logs/universe_heartbeat.txt
tail -1 research/19_smallcap_daily/logs/backtest_heartbeat.txt
```

### Files NOT to touch

- `backtest_data/market_data.db` — read-only, shared with all other strategies
- `data/nifty500_list.csv` — read-only universe reference
- Anything outside `research/19_smallcap_daily/`
- `research/17_eod_breakout_scan/` — the validated reference implementation

---

## 5. Final aggregation

| Artifact | Contents |
|----------|----------|
| `results/daily_universe_selection.csv` | All 475 non-Nifty500 candidates + diagnostics |
| `results/daily_universe.csv` | Final 117 symbols (one per line) |
| `results/daily_summary.csv` | 6-variant sweep metrics |
| `results/daily_trades_<variant>.csv` | Per-trade log per variant (6 files) |
| `results/daily_equity_<variant>.csv` | Daily equity curve per variant (6 files) |
| `results/daily_walk_forward.csv` | IS vs OOS comparison for top 3 variants |
| `FINDINGS.md` | Full writeup (verdict, top contributors, caveats) |

Ranking criteria (priority order):
1. OOS PF >= 1.20
2. OOS Sharpe >= 0.8
3. OOS MaxDD <= 30%
4. CAGR (informational tie-breaker)
