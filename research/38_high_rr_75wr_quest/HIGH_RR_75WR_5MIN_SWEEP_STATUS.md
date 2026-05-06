# High-RR 75% WR Quest — Tight-SL / Wide-TP Intraday on 310-Stock 5-min Universe

STATUS: RUNNING (Stage A)

## 1. Headline

Find intraday 5-min equity systems where **TP > SL** (favorable RR), WR >= 75%
out-of-sample, PF >= 2.5, MaxDD <= 10% test, n >= 30 test trades.

## 2. The Ask

**What you asked:**
> "Find intraday 5-min systems with WR >= 75% OOS, TP > SL (favorable RR,
> opposite of research/37 winners), PF >= 2.5, MaxDD <= 10% OOS, n >= 30."

**What we're actually testing:**
The research/37 winners all use TP 0.5% / SL 1.5% (RR 1:3). They win 75%+
of the time but each win is small. The user now wants the *other* shape —
**tight stop, larger target, still 75% WR**. Empirically rare. Pattern
families to test:

- **Inside-bar / NR4 / NR7 breakouts** — tight consolidation -> tight stop;
  breakout often runs further than the consolidation height (1.5%+ TP).
- **Failed-breakout reversal** — false break of OR / prior-day high faded
  with stop above the false high (often <0.5%), target VWAP/midline (often
  natural 2:1).
- **Multi-bar trend confirmation entries** — 2-3 bar pattern fires, stop
  below structural low (often <0.6%), target measured move.
- **VWAP V-reversal** — sharp reversal off VWAP after extension; stop above
  rejection wick; target back to opposite side.
- **Liquidity-grab / stop-run reversal** — break beyond a swing extreme,
  immediate reversal; tight stop above the false high.
- **Compression breakout** — extreme low-volatility cluster (low ATR), tight
  stop below cluster low, target measured move 1.5%-2.0%+.

## 3. The Base — what's being tested

- **Universe:** 310 NSE stocks with 5-min data (`backtest_data/market_data.db`,
  `market_data_unified` table, timeframe='5minute'), 2024-03-18 to 2026-03-25.
- **Walk-forward split:**
  - Train: 2024-03-18 to 2025-09-30 (18 months)
  - Test:  2025-10-01 to 2026-03-25 (~6 months)
- **Direction:** long and short, treated independently (two parallel sweeps).
- **Exit policies tested (RR > 1):**
  - TP 1.0% / SL 0.4% (RR 2.5:1)
  - TP 1.5% / SL 0.5% (RR 3:1)
  - TP 1.5% / SL 0.6% (RR 2.5:1)
  - TP 2.0% / SL 0.7% (RR 2.86:1)
  - TP 1.0% / SL 0.5% (RR 2:1) — borderline, kept for sanity
- **Hold cap:** session close (no overnight).
- **Pass criteria (OOS / test only):**
  - WR >= 0.75
  - PF >= 2.5
  - MaxDD <= 10%
  - n >= 30 trades
  - Train WR also >= 0.72 (drift tolerance)

## 4. Plan — variant grid + cell count

**Stage A** — universe screen, asks per stock: "what's the WR of a generic
inside-bar / NR4 / VWAP-V reversal / compression-break / failed-breakout /
multi-bar-confirmation pattern at TP=1% / SL=0.4%?" Output: per-stock CSV.
Cohort = top 25-40 stocks per pattern.

**Stage B** — pattern-specific full sweeps on each cohort:
- Pattern x [drop/window/RSI/vol-axis values] x exit grid (5 cells).
- Estimated ~150-300 cells per pattern, 6 patterns = ~1500 cells total.

**Stage C** — confluence stack on the top survivors (NIFTY-regime gating).

**Stage D** — walk-forward validation on top per-pattern winners.

**Stage E** — combined portfolio backtest if any survive.

## 5. Status

### State header

- **Phase:** Stage A in flight (per-stock pattern screen)
- **Started:** 2026-05-06
- **Last completed step:** Stage A launched, 308 stocks x 7 patterns x 2 dirs.
  Rate ~0.33 stocks/s, ETA ~15 min.
- **Next:** When Stage A done -> 99_summarize.py to find diamonds per pattern,
  then 02_pattern_sweep.py per pattern in parallel.

### Event log

| Date/time | Event | Notes |
|---|---|---|
| 2026-05-06 | Folder + STATUS doc created | Mirrors research/37 layout |
| 2026-05-06 | Stage A: pattern signal lib written | scripts/pattern_lib.py |
| 2026-05-06 | Stage A: per-stock screen launched | scripts/01_pattern_screen.py |

### Live findings during the run

(To be populated.)

## 6. Crash Recovery

To resume independently of Claude:

```powershell
cd c:\Users\Castro\Documents\Projects\Covered_Calls

# Check Stage A progress
ls research\38_high_rr_75wr_quest\results

# Check for any running python processes
Get-Process python -ErrorAction SilentlyContinue

# Stage A — per-stock screen (resumable, skips already-done symbols)
python research\38_high_rr_75wr_quest\scripts\01_pattern_screen.py

# Stage B — per-pattern sweeps (run after Stage A done, or if cohort files exist)
python research\38_high_rr_75wr_quest\scripts\02_pattern_sweep.py --pattern inside_bar
python research\38_high_rr_75wr_quest\scripts\02_pattern_sweep.py --pattern failed_breakout
python research\38_high_rr_75wr_quest\scripts\02_pattern_sweep.py --pattern vwap_v_reversal
python research\38_high_rr_75wr_quest\scripts\02_pattern_sweep.py --pattern compression_break
python research\38_high_rr_75wr_quest\scripts\02_pattern_sweep.py --pattern multi_bar_confirm
python research\38_high_rr_75wr_quest\scripts\02_pattern_sweep.py --pattern stop_run_reversal

# Stage D — walk-forward on top survivors
python research\38_high_rr_75wr_quest\scripts\03_walk_forward.py
```

**Files NOT to touch while a sweep runs:**
- `results/01_*_perstock.csv` (incremental writes from screen)
- `results/02_*_ranking.csv` (incremental writes from per-pattern sweep)

**Safe to inspect any time:**
- `logs/*.log`
- All STATUS / RESULTS docs

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `HIGH_RR_75WR_5MIN_SWEEP_STATUS.md` | This file | yes |
| `scripts/pattern_lib.py` | 6 pattern signal generators | yes |
| `scripts/01_pattern_screen.py` | Stage A per-stock screen across patterns | yes |
| `scripts/02_pattern_sweep.py` | Stage B full param sweep per pattern | yes |
| `scripts/03_walk_forward.py` | Stage D walk-forward validation | yes |
| `scripts/04_combined_portfolio.py` | Stage E combined backtest (if winners) | yes |
| `results/01_*_perstock.csv` | Per-stock screen output (small) | yes |
| `results/02_*_ranking.csv` | Per-pattern full sweep ranking | yes |
| `results/03_walk_forward.csv` | Walk-forward results | yes |
| `results/*_diamonds.txt` | Cohort lists | yes |
| `logs/*.log` | Per-stage logs | yes |
| `HIGH_RR_75WR_5MIN_SWEEP_RESULTS.md` | Final findings | yes |

## 8. Findings

(To be populated as data comes in.)
