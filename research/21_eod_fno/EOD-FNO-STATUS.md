# EOD-FNO-STATUS

Live status doc for research/21 — F&O-only EOD breakout backtest.
Mirrors research/17 + research/19 engines; restricted universe.

---

## 1. Goal + Scope

Build a third validated EOD breakout system, restricted to the **F&O
universe** (~80 stocks). Two sister systems already passed walk-forward:

- research/17 (Nifty 500): OOS PF 1.44, Sharpe 0.95
- research/19 (small/micro caps): OOS PF 1.46, Sharpe 1.43

Hypothesis: if the same `vol_3x` rule wins on F&O names, it becomes the
underlier for options strategies (covered calls / cash-secured puts /
broken-wing flies). F&O eligibility itself is the liquidity filter.

**Period:** 2018-01-01 to 2025-12-31. **Capital:** Rs 10,00,000.

**Success:** any of the 3 walk-forward variants delivers OOS PF >= 1.20
AND Sharpe >= 0.8 AND MaxDD <= 30%.

---

## 2. Plan

| Step | What | Script | Output |
|------|------|--------|--------|
| 1 | Build F&O universe (>=1500 daily bars since 2018) | `run_fno_backtest.py` (auto on first run) | `results/fno_universe.csv` |
| 2 | Sweep 6 variants on full period | `run_fno_backtest.py` | `results/fno_summary.csv`, `fno_trades*.csv`, `fno_equity_*.csv` |
| 3 | Walk-forward top-3 variants (IS 2018-2022, OOS 2023-2025) | `walk_forward_fno.py` | `results/fno_walk_forward.csv` |
| 4 | Write FINDINGS.md with verdict + recommendation | manual | `FINDINGS.md` |

### Variants

Same 6-variant sweep family as research/19 (orthogonal cells around the
validated `vol_3x` winner shape). Costs default to **0.20%** round-trip
(F&O names = tighter spreads vs research/19's 0.30%).

| Variant | Vol mult | Target | Cost |
|---|---|---|---|
| baseline_252_25pct_8pct | 2.5x | 25% | 0.20% |
| vol_2x | 2.0x | 25% | 0.20% |
| vol_3x | 3.0x | 25% | 0.20% |
| target_30pct | 2.5x | 30% | 0.20% |
| target_20pct | 2.5x | 20% | 0.20% |
| cost_30bps | 2.5x | 25% | 0.30% |

Walk-forward runs `vol_3x`, `baseline_252_25pct_8pct`, `target_30pct`.

---

## 3. Status

| Step | State | Notes |
|---|---|---|
| 1. Universe build | COMPLETED | 81 candidates -> 76 kept (>=1500 bars) |
| 2. Full-period sweep | COMPLETED | 6 variants; vol_3x wins (PF 1.86, Sharpe 1.13, MaxDD 20.0%) |
| 3. Walk-forward | COMPLETED | All 3 variants PASS (vol_3x OOS: PF 1.91, Sharpe 1.04, MaxDD 9.1%) |
| 4. Findings | COMPLETED | Returned in agent message (no MD file per harness rule) |

Heartbeat file: `logs/backtest_heartbeat.txt` (updated as each variant runs).

---

## 4. Crash Recovery

If Claude / shell dies mid-run, the human can resume independently.

### What finished?

```bash
cat research/21_eod_fno/logs/backtest_heartbeat.txt
ls research/21_eod_fno/results/
```

`fno_summary.csv` is appended after EACH variant — open it to see which
variants completed. `fno_walk_forward.csv` is written only at the end of
`walk_forward_fno.py`.

### Restart everything

```bash
cd c:/Users/Castro/Documents/Projects/Covered_Calls
python research/21_eod_fno/scripts/run_fno_backtest.py
python research/21_eod_fno/scripts/walk_forward_fno.py
```

The universe CSV (`results/fno_universe.csv`) is cached after the first
run — re-running won't re-query the DB unless the file is deleted.

### Restart only the walk-forward

```bash
cd c:/Users/Castro/Documents/Projects/Covered_Calls
python research/21_eod_fno/scripts/walk_forward_fno.py
```

### DO NOT TOUCH

- `backtest_data/market_data.db` — read-only price DB
- Any file in `research/17_eod_breakout_scan/` or `research/19_smallcap_daily/`
  — these are sister systems' artifacts

---

## 5. Final Aggregation

When all steps complete:

| Artifact | What it is |
|---|---|
| `results/fno_universe.csv` | Final F&O symbol list with bar counts |
| `results/fno_summary.csv` | One row per variant — full-period metrics |
| `results/fno_walk_forward.csv` | IS vs OOS metrics for top-3 variants |
| `results/fno_trades.csv` | Aggregate trade log (all variants) |
| `results/fno_trades_<variant>.csv` | Per-variant trade log |
| `results/fno_equity_<variant>.csv` | Per-variant daily equity curve |
| `results/fno_equity.csv` | Equity for `vol_3x` (canonical winner shape) |
| `FINDINGS.md` | Verdict (PASS/FAIL on each gate) + recommendation |

### Ranking criteria (for FINDINGS.md)

1. OOS profit factor (must be >= 1.20)
2. OOS Sharpe (must be >= 0.8)
3. OOS MaxDD (must be <= 30%)
4. Tiebreak: OOS Calmar, then trade count (more = more reliable signal)

If at least one variant passes all 3 gates -> recommend paper trade.
If all 3 fail -> recommend kill (or refine if margins are close).
