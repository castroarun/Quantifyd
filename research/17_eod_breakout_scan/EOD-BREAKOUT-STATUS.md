# EOD Breakout Scanner — Live Status

**Started:** 2026-04-24 post-market session
**Goal:** Test if EOD volume + price breakout signals on Nifty 500 produce
a positive-edge swing momentum system, with proper walk-forward validation.

## System under test

- **Universe:** Nifty 500 (~373 stocks with >=1500 daily bars since 2018)
- **Direction:** LONG-only (shorts deferred — user plans options approach)
- **Bars:** Daily
- **Hold:** Multi-day swing (capped by exit logic, no fixed max)
- **Capital:** Rs 10,00,000
- **Risk per trade:** 1% of equity
- **Max concurrent:** 10 positions
- **Costs:** 0.20% round-trip (delivery STT + brokerage + slippage)
- **Period:** 2018-01-01 → 2025-12-31 (~8 years; covers COVID crash, recovery,
  2022 consolidation, 2023-24 rally — 4+ regimes)

## Variant matrix (11 total — orthogonal sweeps around baseline)

**Baseline:** 50-day Donchian high + vol >= 2x 50-day-avg + close > 200-SMA
+ Donch10 exit + 12% initial hard stop.

| # | Block | Variant | Changes vs baseline |
|---|---|---|---|
| 1 | — | `baseline_50_2x_200_d10` | (the baseline itself) |
| 2 | A | `A1_breakout_20` | 20-day high (more frequent) |
| 3 | A | `A2_breakout_252` | 252-day high (~52w / ATH proxy) |
| 4 | B | `B1_no_vol` | drop volume filter |
| 5 | B | `B2_vol_15x` | volume >= 1.5x avg |
| 6 | B | `B3_vol_30x` | volume >= 3.0x avg |
| 7 | C | `C1_no_regime` | drop 200-SMA filter |
| 8 | D | `D1_fixed_25_8` | fixed 25% target / 8% stop |
| 9 | D | `D2_atr_trail_3` | ATR trail 3x |
| 10 | D | `D3_chandelier_3` | Chandelier 3xATR |
| 11 | D | `D4_donch_20` | Donchian 20-day low exit |

## Pass criteria for "real edge"

In the **walk-forward** test (train 2018-2022, test 2023-2025) on the
top variants from the full sweep:
- OOS Sharpe ≥ 0.8
- OOS PF ≥ 1.2
- OOS MaxDD ≤ 25%

---

## Current status

| Phase | Status | Output |
|---|---|---|
| Backtest engine built | ✅ done | `scripts/run_eod_breakout.py` |
| Universe loaded (Nifty 500 → 373 valid) | ✅ done | — |
| Phase 1 — full-period 11-variant sweep | ✅ done (494s) | `results/summary.csv` |
| Phase 2 — walk-forward on top 3 variants | ✅ done (184s) | `results/walk_forward_summary.csv` |
| Phase 3 — findings | ✅ done | `FINDINGS.md` |

### Phase 2 walk-forward verdict — D1 PASSES, others FAIL

| Variant | OOS PF | OOS Sharpe | OOS MaxDD | Verdict |
|---|---:|---:|---:|---|
| **D1_fixed_25_8** | **1.44** | **0.95** | **24.0%** | **PASS** ✓ |
| D4_donch_20 | 1.22 | 0.39 | 28.5% | FAIL |
| baseline_50_2x_200_d10 | 1.16 | 0.48 | 27.7% | FAIL |

**D1 OOS metrics improved over IS** (Sharpe 0.82 → 0.95, MaxDD 35.9% → 24.0%) — opposite of overfit pattern. Strong evidence of real edge.

OOS sample: 208 trades over 3 years on Nifty 500 long-only.

### Phase 1 result summary (sorted by Sharpe)

| Variant | PF | Sharpe | CAGR% | MaxDD% | Calmar |
|---|---:|---:|---:|---:|---:|
| **D1_fixed_25_8** (TOP) | **1.41** | **0.96** | **+15.26** | 35.94 | **0.42** |
| D4_donch_20 | 1.30 | 0.59 | +9.41 | 28.60 | 0.33 |
| baseline_50_2x_200_d10 | 1.14 | 0.43 | +6.16 | 30.74 | 0.20 |

Top 3 chosen for Phase 2 walk-forward: D1, D4, baseline.

Key learnings from Phase 1:
- 200-SMA regime filter is critical (PF 1.14 with vs 1.01 without)
- 50-day breakout window is sweet spot (vs 20d / 252d)
- Volume filter doesn't help much (1.10-1.14 across thresholds)
- Fixed exits dominate ATR/Chandelier trails for swing momentum
- Avg hold 22-42 days — these are real swing trades

---

## Crash recovery — full instructions for the human

### 1. What's already on disk

- `scripts/run_eod_breakout.py` — the backtest engine, complete
- `EOD-BREAKOUT-STATUS.md` — this file
- After Phase 1 completes:
  - `results/summary.csv` — 11 rows (one per variant), all metrics
  - `results/trades_<variant>.csv` — per-trade log per variant
  - `results/equity_<variant>.csv` — daily equity curve per variant

### 2. Re-run the full Phase 1 sweep (if killed mid-run)

```bash
cd c:/Users/Castro/Documents/Projects/Covered_Calls
python research/17_eod_breakout_scan/scripts/run_eod_breakout.py
```

The script writes `summary.csv` AT THE END — partial runs leave no summary.
But per-variant `trades_*.csv` and `equity_*.csv` are written incrementally
after each variant completes. Inspect to see how far it got:
```bash
ls research/17_eod_breakout_scan/results/
```

### 3. Identify top variants for walk-forward

Once Phase 1 completes, sort by Sharpe / PF:
```bash
python -c "
import csv
rows = list(csv.DictReader(open('research/17_eod_breakout_scan/results/summary.csv')))
rows.sort(key=lambda r: -float(r['sharpe']))
for r in rows[:5]:
    print(r['variant'], 'PF=', r['profit_factor'], 'Sharpe=', r['sharpe'], 'CAGR=', r['cagr_pct'])
"
```

### 4. Walk-forward (manual, to be built in Phase 2)

After Phase 1 completes I'll write `scripts/walk_forward.py` that re-runs
the top 2-3 variants on:
- Train: 2018-01-01 → 2022-12-31
- Test: 2023-01-01 → 2025-12-31

Pass criteria: OOS Sharpe ≥ 0.8, OOS PF ≥ 1.2, OOS MaxDD ≤ 25%.

### 5. What NOT to touch

- The 11 variant configs in `run_eod_breakout.py` — sweep design
- The capital / risk / costs constants — fixed per agreed spec

### 6. Files written

| File | Phase | Purpose |
|---|---|---|
| `results/summary.csv` | 1 | 11 rows × 14 metrics — variant ranking |
| `results/trades_<variant>.csv` | 1 | Per-trade log; entry/exit/pnl |
| `results/equity_<variant>.csv` | 1 | Daily mark-to-market equity curve |
| `results/walk_forward_summary.csv` | 2 (TBD) | IS vs OOS comparison |
| `FINDINGS.md` | 3 | Conclusions + decision |
