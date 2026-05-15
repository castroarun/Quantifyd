# VOLSURGE + PDR/PWR-Break, Narrow Weekly CPR — Underlying Sweep

## Setup

- Universe: 79 F&O stocks (FNO_LOT_SIZES minus TATAMOTORS, ZOMATO)
- Period: 2018-01-01 -> 2026-05-15 (per-stock, clipped to available 5-min)
- Timeframes: 5min, 10min, 15min, 30min, 60min (10/15/30/60 resampled from 5-min)
- Grid: trend['sma50', 'sma200', 'hh20'] x theta_cpr[0.25, 0.5, 0.75, 1.0] x k[1.5, 2.0, 3.0] x clean['loose', 'strict'] x clearroom[False, True] x carry['sameday', 'carry']
- Direction = daily-trend selector (up->long, down->short, flat->skip)
- 13 exit policies scored in parallel per signal
- Total signal rows: **163** (long 112 / short 51)

> Per ex#9: the confluence is a NECESSARY-but-NOT-SUFFICIENT probabilistic edge — judged on expectancy/Sharpe over a population, never on individual outcomes.

## Top 15 configs (Sharpe, n>=15, mean>0)

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Exp% | Sharpe |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|

## Per-stock leaders (best Sharpe, n>=10)

| Rank | Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Exp% | Sharpe | HiQ | MidQ |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ICICIBANK | 30min | 30min_sma50_cpr1.00_vm2.0_loose_croff_sameday | long | T_R_TARGET_1.0R | 10 | 1.265 | 90.0 | 1.265 | 1.0150 | 11 | 11 |
| 2 | SBIN | 30min | 30min_sma50_cpr1.00_vm2.0_loose_croff_sameday | long | T_NO | 12 | 0.438 | 83.3 | 0.438 | 0.2347 | 0 | 0 |
| 3 | RELIANCE | 30min | 30min_sma50_cpr1.00_vm2.0_loose_croff_sameday | long | T_R_TARGET_2.0R | 14 | 0.237 | 57.1 | 0.237 | 0.1199 | 0 | 0 |
| 4 | KOTAKBANK | 30min | 30min_sma50_cpr1.00_vm2.0_loose_croff_sameday | long | T_R_TARGET_1.0R | 10 | 0.321 | 50.0 | 0.321 | 0.1177 | 0 | 0 |
| 5 | HINDUNILVR | 30min | 30min_sma50_cpr1.00_vm2.0_loose_croff_sameday | short | T_STEP_TRAIL | 10 | 0.153 | 40.0 | 0.153 | 0.1056 | 0 | 0 |
| 6 | HDFCBANK | 30min | 30min_sma50_cpr1.00_vm2.0_loose_croff_sameday | long | T_R_TARGET_1.0R | 10 | 0.096 | 70.0 | 0.096 | 0.0981 | 0 | 0 |
| 7 | INFY | 30min | 30min_sma50_cpr1.00_vm2.0_loose_croff_sameday | long | T_ATR_SL_0.3 | 12 | -0.619 | 0.0 | -0.619 | -0.0000 | 0 | 0 |
| 8 | ITC | 30min | 30min_sma50_cpr1.00_vm2.0_loose_croff_sameday | long | T_ATR_SL_0.3 | 15 | -0.528 | 0.0 | -0.528 | -0.0000 | 0 | 0 |

## Promote candidates (Sharpe>=0.5, n>=15, MidQ>=3)

Robustness gate (research/34 style): best cell strong AND the edge survives across >=3 variants.

_No stocks passed the gate._

## Direction asymmetry (top 20 leaders)

| Symbol | Long Sharpe (n) | Short Sharpe (n) | Bias |
|---|---:|---:|---|
| ICICIBANK | 1.015 (n=10) | 0.000 (n=0) | LONG |
| SBIN | 0.235 (n=12) | 0.000 (n=0) | LONG |
| RELIANCE | 0.120 (n=14) | 0.000 (n=0) | weak |
| KOTAKBANK | 0.118 (n=10) | 0.000 (n=0) | weak |
| HINDUNILVR | 0.034 (n=13) | 0.106 (n=10) | weak |
| HDFCBANK | 0.098 (n=10) | 0.000 (n=0) | weak |
| INFY | -0.000 (n=12) | 0.000 (n=0) | weak |
| ITC | -0.000 (n=15) | 0.000 (n=0) | weak |

## Timeframe sweet-spot (cells n>=15, mean>0)

| TF | n_cells | avg_mean% | avg_WR% | avg_Sharpe |
|---|---:|---:|---:|---:|
| 5min | 0 | - | - | - |
| 10min | 0 | - | - | - |
| 15min | 0 | - | - | - |
| 30min | 0 | - | - | - |
| 60min | 0 | - | - | - |

## Exit-policy comparison (cells n>=15, mean>0)

| ExitPolicy | n_cells | avg_mean% | avg_WR% | avg_Sharpe |
|---|---:|---:|---:|---:|
| T_NO | 0 | 0.0000 | 0.0 | 0.0000 |
| T_HARD_SL | 0 | 0.0000 | 0.0 | 0.0000 |
| T_ATR_SL_0.3 | 0 | 0.0000 | 0.0 | 0.0000 |
| T_ATR_SL_0.5 | 0 | 0.0000 | 0.0 | 0.0000 |
| T_ATR_SL_1.0 | 0 | 0.0000 | 0.0 | 0.0000 |
| T_CHANDELIER_1.0 | 0 | 0.0000 | 0.0 | 0.0000 |
| T_CHANDELIER_1.5 | 0 | 0.0000 | 0.0 | 0.0000 |
| T_CHANDELIER_2.0 | 0 | 0.0000 | 0.0 | 0.0000 |
| T_R_TARGET_1.0R | 0 | 0.0000 | 0.0 | 0.0000 |
| T_R_TARGET_1.5R | 0 | 0.0000 | 0.0 | 0.0000 |
| T_R_TARGET_2.0R | 0 | 0.0000 | 0.0 | 0.0000 |
| T_R_TARGET_3.0R | 0 | 0.0000 | 0.0 | 0.0000 |
| T_STEP_TRAIL | 0 | 0.0000 | 0.0 | 0.0000 |
