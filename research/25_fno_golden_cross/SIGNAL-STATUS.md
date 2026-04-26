# Signal 4 — Golden / Death Cross — STATUS

## Goal

Classic + EMA + stochastic-confirmed cross signals on the F&O 76 universe.
Tests both directions (golden cross LONG, death cross SHORT). Cross must be
"recent" — within last 5 bars — to allow late entries.

Pass criteria (OOS 2023-2025): PF ≥ 1.20 AND Sharpe ≥ 0.8 AND MaxDD ≤ 30%.

## Plan (12 cells = 6 LONG + 6 SHORT)

| # | Variant | MA type | Fast/Slow | + Stoch | + ADX | + 200-SMA |
|---|---|---|---|---|---|---|
| 1 | gc_sma_50_200            | SMA | 50/200 | — | — | — |
| 2 | gc_ema_50_200            | EMA | 50/200 | — | — | — |
| 3 | gc_ema_20_50             | EMA | 20/50  | — | — | — |
| 4 | gc_sma_50_200_stoch      | SMA | 50/200 | yes (K up <30) | — | — |
| 5 | gc_sma_50_200_adx        | SMA | 50/200 | — | >25 | — |
| 6 | gc_sma_50_200_confluence | SMA | 50/200 | yes | >25 | yes |
| 7-12 | dc_* mirror for SHORT side  |  |  |  |  |  |

## Status

DONE — all 12 variants (6 LONG + 6 SHORT) run, walk-forward complete on top 3.

**Verdict: PARTIAL PASS — LONG side only. Best variants: `gc_sma_50_200_adx`
(OOS PF=1.82, Sharpe=0.81, MaxDD=6.02%, CAGR=4.41%) and `gc_ema_20_50`
(OOS PF=1.46, Sharpe=0.90, MaxDD=24.16%, CAGR=11.03%). All 6 SHORT/death-cross
variants are unprofitable (PF 0.31-0.76).**

Full-period (2018-2025):

| Variant                   | Dir   | Trades | WR%  | PF   | CAGR%  | Sharpe | MaxDD% |
|---------------------------|-------|--------|------|------|--------|--------|--------|
| gc_sma_50_200             | LONG  |  324   | 39.5 | 1.49 | +5.66  | 0.69   | 13.39  |
| gc_ema_50_200             | LONG  |  384   | 38.0 | 1.38 | +5.27  | 0.62   | 13.94  |
| gc_ema_20_50              | LONG  |  682   | 38.7 | 1.40 | +10.43 | 0.82   | 24.28  |
| gc_sma_50_200_stoch       | LONG  |  137   | 36.5 | 1.28 | +1.50  | 0.32   | 14.50  |
| gc_sma_50_200_adx         | LONG  |  219   | 41.1 | 1.59 | +4.46  | 0.65   |  8.15  |
| gc_sma_50_200_confluence  | LONG  |   45   | 31.1 | 1.06 | +0.12  | 0.06   |  8.22  |
| dc_*                      | SHORT | 45-694 | 17-31| 0.31-0.76 | -1.97 to -8.28 | <0 | 15-63 |

Walk-forward (LONG, top 3 by full-period PF):

| Variant            | OOS PF | OOS Sharpe | OOS MaxDD% | OOS CAGR% | Verdict |
|--------------------|--------|------------|------------|-----------|---------|
| gc_sma_50_200_adx  | 1.82   | 0.81       |  6.02      |  4.41     | PASS |
| gc_sma_50_200      | 1.47   | 0.64       | 10.11      |  4.21     | FAIL (Sharpe<0.8) |
| gc_ema_20_50       | 1.46   | 0.90       | 24.16      | 11.03     | PASS |

Note: `gc_sma_50_200_adx` has the cleanest stats (low DD, high PF) but only
4.41% OOS CAGR — too few trades. `gc_ema_20_50` is more deployable in
practice (251 OOS trades, 11% CAGR) but 24% DD is near the ceiling.

SHORT-side death-cross variants confirm research/22's finding: F&O symbols
do not sustain bearish trend continuation reliably.

## Crash recovery

```bash
cd research/25_fno_golden_cross
python scripts/run_cross_sweep.py
python scripts/walk_forward_cross.py
```

## Final aggregation

`FINDINGS.md` records best LONG and best SHORT variant + verdicts.
