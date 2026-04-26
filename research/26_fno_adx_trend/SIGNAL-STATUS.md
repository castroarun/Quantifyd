# Signal 5 — ADX Trend-Strength Entry — STATUS

## Goal

ADX(14) just crossed above THR (default 25) from below within last 5 bars,
AND +DI > -DI for LONG (or -DI > +DI for SHORT). Tests both directions.

Pass criteria (OOS 2023-2025): PF ≥ 1.20 AND Sharpe ≥ 0.8 AND MaxDD ≤ 30%.

## Plan (10 cells = 5 LONG + 5 SHORT)

| # | Variant | ADX thr | Regime | DI margin |
|---|---|---|---|---|
| 1 | adx_25_pure          | 25 | — | 0 |
| 2 | adx_25_above_50sma   | 25 | 50-SMA | 0 |
| 3 | adx_25_above_200sma  | 25 | 200-SMA | 0 |
| 4 | adx_30               | 30 | — | 0 |
| 5 | adx_25_confluence    | 25 | 50-SMA | > 5 |

## Status

DONE — all 10 variants (5 LONG + 5 SHORT) run, walk-forward on top 3.

**Verdict: PARTIAL PASS — LONG only. `adx_30` LONG is the standout: OOS PF=1.77,
Sharpe=1.30, MaxDD=16.47%, CAGR=15.50% on 225 trades. All SHORT variants DOA.**

Full-period (2018-2025):

| Variant             | Dir   | Trades | WR%  | PF   | CAGR%  | Sharpe | MaxDD% |
|---------------------|-------|--------|------|------|--------|--------|--------|
| adx_25_pure         | LONG  |  658   | 37.7 | 1.32 | +9.29  | 0.74   | 28.04  |
| adx_25_above_50sma  | LONG  |  658   | 36.6 | 1.31 | +8.70  | 0.71   | 26.64  |
| adx_25_above_200sma | LONG  |  601   | 36.3 | 1.27 | +6.70  | 0.59   | 27.80  |
| adx_30              | LONG  |  614   | 38.6 | 1.43 | +9.59  | 0.78   | 29.11  |
| adx_25_confluence   | LONG  |  642   | 37.7 | 1.37 | +10.07 | 0.81   | 26.43  |
| adx_*               | SHORT | 428-618| 25-29| 0.60-0.67 | -7 to -12 | <0 | 50-67 |

Walk-forward (LONG, top 3):

| Variant           | OOS PF | OOS Sharpe | OOS MaxDD% | OOS CAGR% | Verdict |
|-------------------|--------|------------|------------|-----------|---------|
| adx_30            | 1.77   | 1.30       | 16.47      | 15.50     | PASS |
| adx_25_confluence | 1.31   | 0.68       | 17.95      |  7.52     | FAIL |
| adx_25_pure       | 1.22   | 0.44       | 15.08      |  4.70     | FAIL |

The strict ADX threshold (30) materially improves OOS edge — confluence and
pure variants degraded significantly between IS and OOS, suggesting overfit
to 2018-2022 IS data. ADX>30 is rarer (614 trades vs 658 for ADX>25) but
selects clearer trend establishments. SHORT side confirms research-21 thesis:
F&O downtrends don't follow through.

## Crash recovery

```bash
cd research/26_fno_adx_trend
python scripts/run_adx_sweep.py
python scripts/walk_forward_adx.py
```

## Final aggregation

`FINDINGS.md` records verdicts.
