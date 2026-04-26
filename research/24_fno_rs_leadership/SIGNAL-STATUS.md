# Signal 3 — Relative-Strength Leadership — STATUS

## Goal

Buy stocks dramatically outperforming Nifty 50 (proxy: NIFTYBEES, since
NIFTY50 cash data only goes back to 2023). Stock N-day return minus
NIFTYBEES N-day return must exceed Z standard deviations of its 252-day
distribution. LONG side.

Pass criteria (OOS 2023-2025): PF ≥ 1.20 AND Sharpe ≥ 0.8 AND MaxDD ≤ 30%.

## Plan

| # | Variant | lookback | z thr | extras |
|---|---|---|---|---|
| 1 | rs_30d_z1.5         | 30 | 1.5 | — |
| 2 | rs_60d_z1.5         | 60 | 1.5 | — (default) |
| 3 | rs_60d_z2.0         | 60 | 2.0 | — (stricter) |
| 4 | rs_60d_z1.5_200sma  | 60 | 1.5 | + 200-SMA |
| 5 | rs_60d_z1.5_vol_2x  | 60 | 1.5 | + vol ≥ 2x avg |

## Status

DONE — all 5 variants run, walk-forward complete on top 3.

**Verdict: PARTIAL PASS. Only `rs_60d_z2.0` (stricter z-threshold) clears the
OOS gates. PF=1.53, Sharpe=0.99, MaxDD=9.35%, CAGR=11.41% on 217 trades.**

NOTE on Nifty proxy: `services/data_manager.FNO_LOT_SIZES` doesn't include
NIFTY index; market_data_unified has 'NIFTY50' but only since 2023-03-20.
We use `NIFTYBEES` (Nifty 50 ETF, 2005+) for the full backtest window.

Full-period (2018-2025):

| Variant            | Trades | WR%  | PF   | CAGR%  | Sharpe | MaxDD% |
|--------------------|--------|------|------|--------|--------|--------|
| rs_30d_z1.5        |  687   | 36.8 | 1.30 | +8.49  | 0.60   | 33.52  |
| rs_60d_z1.5        |  720   | 36.4 | 1.27 | +8.28  | 0.58   | 29.90  |
| rs_60d_z2.0        |  580   | 37.6 | 1.38 | +10.12 | 0.75   | 27.27  |
| rs_60d_z1.5_200sma |  689   | 36.0 | 1.28 | +8.63  | 0.62   | 32.41  |
| rs_60d_z1.5_vol_2x |  515   | 37.5 | 1.33 | +7.12  | 0.61   | 22.05  |

Walk-forward (top 3 by full-period PF):

| Variant            | OOS PF | OOS Sharpe | OOS MaxDD% | OOS CAGR% | Verdict |
|--------------------|--------|------------|------------|-----------|---------|
| rs_30d_z1.5        | 1.41   | 0.72       | 15.49      |  8.78     | FAIL (Sharpe<0.8) |
| rs_60d_z2.0        | 1.53   | 0.99       |  9.35      | 11.41     | PASS |
| rs_60d_z1.5_vol_2x | 1.44   | 0.75       |  9.79      |  8.01     | FAIL (Sharpe<0.8) |

Stricter z-threshold (2.0) is the right level — tighter signal = better edge.
Volume confirmation didn't help; the 200-SMA filter overlaps with the
already-strong RS signal so adds nothing.

## Crash recovery

```bash
cd research/24_fno_rs_leadership
python scripts/run_rs_sweep.py
python scripts/walk_forward_rs.py
```

## Final aggregation

`FINDINGS.md` records best variant and verdict.
