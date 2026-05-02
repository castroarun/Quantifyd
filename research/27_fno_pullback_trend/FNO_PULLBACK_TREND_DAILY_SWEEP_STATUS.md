# Signal 6 — Pullback in Trend — STATUS

## Goal

"Buy the dip in uptrend" / "Sell the rally in downtrend". LONG: close > 200-SMA
AND 50-SMA > 200-SMA AND close < 20-EMA AND RSI(14) < threshold. SHORT mirrors.

Pass criteria (OOS 2023-2025): PF ≥ 1.20 AND Sharpe ≥ 0.8 AND MaxDD ≤ 30%.

## Plan (10 cells = 5 LONG + 5 SHORT)

| # | Variant | RSI thr | + MACD | + Vol climax (≥1.5x) |
|---|---|---|---|---|
| 1 | pb_rsi40                | 40 | — | — |
| 2 | pb_rsi30                | 30 | — | — |
| 3 | pb_rsi40_macd           | 40 | yes | — |
| 4 | pb_rsi40_volume_climax  | 40 | — | yes |
| 5 | pb_rsi40_confluence     | 40 | yes | yes |

## Status

DONE — all 10 variants (5 LONG + 5 SHORT) run, walk-forward on top 3.

**Verdict: PARTIAL PASS — LONG only. `pb_rsi40` is the most deployable: OOS
PF=1.55, Sharpe=1.01, MaxDD=18.19%, CAGR=12.99% on 233 trades. SHORT side DOA.**

Full-period (2018-2025):

| Variant                  | Dir   | Trades | WR%  | PF   | CAGR%  | Sharpe | MaxDD% |
|--------------------------|-------|--------|------|------|--------|--------|--------|
| pb_rsi40                 | LONG  |  557   | 42.5 | 1.50 | +10.93 | 0.81   | 27.21  |
| pb_rsi30                 | LONG  |  182   | 42.9 | 1.48 |  +3.70 | 0.53   | 14.08  |
| pb_rsi40_macd            | LONG  |  102   | 38.2 | 1.62 |  +2.38 | 0.61   |  6.79  |
| pb_rsi40_volume_climax   | LONG  |  415   | 41.9 | 1.50 |  +8.52 | 0.80   | 18.91  |
| pb_rsi40_confluence      | LONG  |   16   | 50.0 | 3.28 |  +0.91 | 0.72   |  1.94  |
| pb_*                     | SHORT | 33-456 | 22-32| 0.45-0.78 | -1 to -6 | <0 | 11-47 |

Walk-forward (LONG, top 3):

| Variant             | OOS Trades | OOS PF | OOS Sharpe | OOS MaxDD% | OOS CAGR% | Verdict |
|---------------------|-----------|--------|------------|------------|-----------|---------|
| pb_rsi40_confluence |    12     | 4.31   | 1.36       |  1.93      |  2.27     | PASS but n=12 (statistically thin) |
| pb_rsi40_macd       |    56     | 1.61   | 0.70       |  5.64      |  3.26     | FAIL (Sharpe<0.8) |
| pb_rsi40            |   233     | 1.55   | 1.01       | 18.19      | 12.99     | PASS (deployable) |

`pb_rsi40_confluence` looks great in stats but only 12 trades OOS — not
deployable as a primary signal. `pb_rsi40` is the real winner: 233 trades,
12.99% CAGR, Sharpe 1.01.

## Crash recovery

```bash
cd research/27_fno_pullback_trend
python scripts/run_pullback_sweep.py
python scripts/walk_forward_pullback.py
```

## Final aggregation

`FINDINGS.md` records verdicts for both directions.
