# Signal 2 — Donchian 55-day Breakout — STATUS

## Goal

Faster-than-252 trend-continuation signal (Turtle classic). LONG side only.

Pass criteria (OOS 2023-2025): PF ≥ 1.20 AND Sharpe ≥ 0.8 AND MaxDD ≤ 30%.

## Plan

| # | Variant | vol | 200-SMA | ADX | ATR floor |
|---|---|---|---|---|---|
| 1 | baseline_55          | 2.0x | no  | no  | no |
| 2 | 55_vol_3x_200sma     | 3.0x | yes | no  | no |
| 3 | 55_vol_3x_adx        | 3.0x | no  | >20 | no |
| 4 | 55_vol_3x_atr_floor  | 3.0x | no  | no  | yes (≥1%) |
| 5 | 55_vol_3x_confluence | 3.0x | yes | >20 | yes |

## Status

DONE — all 5 variants run, walk-forward complete on top 3.

**Verdict: PASS. Best variant `55_vol_3x_atr_floor`. OOS PF=1.97, Sharpe=1.25,
MaxDD=7.74%, CAGR=13.47% on 175 trades.**

Full-period (2018-2025):

| Variant              | Trades | WR%  | PF   | CAGR%  | Sharpe | MaxDD% |
|----------------------|--------|------|------|--------|--------|--------|
| baseline_55          |  610   | 37.7 | 1.37 |  +8.67 |  0.70  | 26.64  |
| 55_vol_3x_200sma     |  441   | 41.5 | 1.67 | +11.11 |  1.02  | 22.10  |
| 55_vol_3x_adx        |  403   | 40.2 | 1.56 |  +8.71 |  0.84  | 19.02  |
| 55_vol_3x_atr_floor  |  469   | 42.0 | 1.69 | +11.81 |  1.02  | 22.11  |
| 55_vol_3x_confluence |  372   | 39.0 | 1.50 |  +7.27 |  0.75  | 19.11  |

Walk-forward (top 3): ALL THREE PASS the gates.

| Variant              | OOS PF | OOS Sharpe | OOS MaxDD% | OOS CAGR% | Verdict |
|----------------------|--------|------------|------------|-----------|---------|
| 55_vol_3x_200sma     | 1.93   | 1.21       | 7.76       | 12.88     | PASS |
| 55_vol_3x_adx        | 1.89   | 1.09       | 8.63       | 10.77     | PASS |
| 55_vol_3x_atr_floor  | 1.97   | 1.25       | 7.74       | 13.47     | PASS (best) |

This is the strongest signal in the 6-family sweep. Recommended for live
deployment with bull call debit spreads.

## Crash recovery

```bash
cd research/23_fno_donchian_55
python scripts/run_donchian_sweep.py
python scripts/walk_forward_donchian.py
```

## Final aggregation

`FINDINGS.md` records best variant and verdict.
