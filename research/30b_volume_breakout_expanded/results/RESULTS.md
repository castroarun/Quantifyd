# Volume-Breakout EXPANDED — Backtest Results

## Setup

- Period: per-stock available range, capped 2026-03-25
- Stocks: 79 (10 Cohort A since 2018; 69 Cohort B since 2024-03-18)
- Timeframes: 5min, 10min, 15min
- Variant grid: vol_mult ∈ {1.5, 2.0, 3.0} × gap_pct ∈ {0%, 0.3%, 0.5%, off} × RSI ∈ {off, on(40/60)}
- Direction: long & short (independent)
- 13 exit policies tested per signal in parallel
- Total signal rows processed: **164,327**
- Ranked cells (n>=5): **123,851**

## Top 10 configurations across all stocks (n>=15, mean>0)

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| RELIANCE | 15min | s_vm3.0_gapoff_rsi40_60 | short | T_R_TARGET_1.0R | 15 | 1.110 | 93.3 | 1.74 | 0.8581 |
| GODREJPROP | 10min | s_vm1.5_gap0.003_norsi | short | T_R_TARGET_1.0R | 16 | 0.932 | 75.0 | 7.43 | 0.8392 |
| HAL | 10min | s_vm1.5_gapoff_rsi40_60 | short | T_NO | 17 | 1.074 | 82.3 | 3.59 | 0.8276 |
| HAL | 10min | s_vm1.5_gapoff_rsi40_60 | short | T_ATR_SL_1.0 | 17 | 1.074 | 82.3 | 3.59 | 0.8276 |
| HAL | 10min | s_vm1.5_gapoff_rsi40_60 | short | T_CHANDELIER_2.0 | 17 | 1.074 | 82.3 | 3.59 | 0.8276 |
| RELIANCE | 15min | s_vm3.0_gapoff_rsi40_60 | short | T_R_TARGET_1.5R | 15 | 1.546 | 93.3 | 2.40 | 0.7956 |
| GODREJPROP | 10min | s_vm1.5_gap0.005_norsi | short | T_R_TARGET_1.0R | 15 | 0.883 | 73.3 | 7.25 | 0.7731 |
| RELIANCE | 15min | s_vm3.0_gap0.000_norsi | short | T_R_TARGET_1.0R | 17 | 1.028 | 88.2 | 2.30 | 0.7692 |
| RELIANCE | 15min | s_vm3.0_gapoff_rsi40_60 | short | T_R_TARGET_2.0R | 15 | 1.625 | 93.3 | 2.52 | 0.7620 |
| RELIANCE | 15min | s_vm3.0_gapoff_norsi | short | T_R_TARGET_1.0R | 18 | 0.989 | 88.9 | 2.20 | 0.7591 |

## Top 25 Volume Leaders (best per-stock cell, n>=10)

| # | Stock | Cohort | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe | RobustCells | Promote? |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **GODREJPROP** | B | 15min | s_vm1.5_gap0.003_rsi40_60 | short | T_R_TARGET_1.0R | 10 | 1.527 | 100.0 | 999.00 | 2.3225 | 194 | — |
| 2 | **HAL** | B | 5min | s_vm1.5_gap0.005_rsi40_60 | short | T_NO | 10 | 1.409 | 90.0 | 17.46 | 1.3545 | 155 | — |
| 3 | **RELIANCE** | A | 15min | s_vm3.0_gap0.000_rsi40_60 | short | T_R_TARGET_1.0R | 14 | 1.166 | 92.9 | 1.84 | 0.8806 | 400 | — |
| 4 | **DELHIVERY** | B | 15min | s_vm2.0_gapoff_norsi | short | T_R_TARGET_1.0R | 10 | 1.398 | 80.0 | 2.97 | 0.8414 | 166 | — |
| 5 | **ONGC** | B | 10min | s_vm1.5_gap0.000_rsi40_60 | short | T_STEP_TRAIL | 10 | 0.556 | 90.0 | 1.97 | 0.8127 | 52 | — |
| 6 | **PERSISTENT** | B | 5min | l_vm3.0_gapoff_norsi | long | T_R_TARGET_1.0R | 11 | 1.424 | 81.8 | 2.22 | 0.7962 | 20 | — |
| 7 | **TATACONSUM** | B | 10min | l_vm1.5_gap0.005_rsi40_60 | long | T_R_TARGET_1.0R | 10 | 0.541 | 80.0 | 2.29 | 0.7083 | 5 | — |
| 8 | **NESTLEIND** | B | 10min | l_vm2.0_gap0.000_rsi40_60 | long | T_R_TARGET_1.5R | 11 | 1.185 | 72.7 | 4.87 | 0.7009 | 17 | — |
| 9 | **JINDALSTEL** | B | 5min | l_vm3.0_gapoff_norsi | long | T_CHANDELIER_1.0 | 10 | 1.460 | 80.0 | 2.73 | 0.6918 | 27 | — |
| 10 | **EICHERMOT** | B | 5min | l_vm3.0_gapoff_norsi | long | T_R_TARGET_1.0R | 10 | 0.504 | 80.0 | 1.41 | 0.6530 | 10 | — |
| 11 | **PIDILITIND** | B | 10min | s_vm2.0_gapoff_norsi | short | T_R_TARGET_1.0R | 10 | 0.525 | 70.0 | 3.59 | 0.6400 | 6 | — |
| 12 | **LT** | B | 15min | l_vm2.0_gap0.003_rsi40_60 | long | T_R_TARGET_3.0R | 11 | 0.676 | 81.8 | 0.94 | 0.5737 | 59 | — |
| 13 | **WIPRO** | B | 10min | s_vm2.0_gap0.003_rsi40_60 | short | T_R_TARGET_1.0R | 12 | 0.678 | 83.3 | 1.05 | 0.5724 | 62 | — |
| 14 | **HAVELLS** | B | 10min | l_vm1.5_gap0.000_rsi40_60 | long | T_R_TARGET_1.0R | 14 | 0.613 | 78.6 | 1.62 | 0.5670 | 4 | — |
| 15 | **COFORGE** | B | 5min | l_vm2.0_gap0.003_norsi | long | T_R_TARGET_1.0R | 13 | 0.981 | 69.2 | 4.68 | 0.5535 | 86 | — |
| 16 | **BAJAJ-AUTO** | B | 10min | l_vm2.0_gap0.003_norsi | long | T_R_TARGET_1.0R | 13 | 0.445 | 84.6 | 1.04 | 0.5260 | 31 | — |
| 17 | **VEDL** | B | 15min | l_vm2.0_gapoff_rsi40_60 | long | T_R_TARGET_1.0R | 18 | 0.689 | 72.2 | 2.10 | 0.5090 | 15 | ✅ YES |
| 18 | **BAJFINANCE** | B | 10min | s_vm1.5_gap0.000_rsi40_60 | short | T_R_TARGET_1.0R | 13 | 0.535 | 76.9 | 1.54 | 0.5046 | 52 | — |
| 19 | **ASIANPAINT** | B | 10min | l_vm1.5_gap0.005_rsi40_60 | long | T_R_TARGET_1.0R | 13 | 0.416 | 76.9 | 1.22 | 0.4808 | 6 | — |
| 20 | **IOC** | B | 5min | s_vm3.0_gap0.000_norsi | short | T_STEP_TRAIL | 11 | 0.103 | 54.5 | 999.00 | 0.4605 | 4 | — |
| 21 | **BPCL** | B | 10min | s_vm3.0_gap0.000_norsi | short | T_ATR_SL_0.3 | 13 | 1.340 | 69.2 | 3.00 | 0.4559 | 36 | — |
| 22 | **CHOLAFIN** | B | 10min | l_vm1.5_gap0.003_norsi | long | T_ATR_SL_0.5 | 11 | 1.028 | 63.6 | 3.19 | 0.4345 | 96 | — |
| 23 | **VOLTAS** | B | 10min | l_vm3.0_gap0.000_norsi | long | T_R_TARGET_1.0R | 10 | 0.727 | 70.0 | 1.50 | 0.4146 | 9 | — |
| 24 | **MARUTI** | B | 10min | l_vm1.5_gapoff_rsi40_60 | long | T_R_TARGET_1.0R | 24 | 0.454 | 75.0 | 1.21 | 0.4143 | 4 | — |
| 25 | **ULTRACEMCO** | B | 5min | s_vm1.5_gapoff_norsi | short | T_R_TARGET_1.0R | 11 | 0.278 | 72.7 | 2.08 | 0.3990 | 1 | — |

## Promote candidates (1 stocks pass robustness gate)

Gate: best-cell Sharpe >= 0.5 AND n >= 15 AND robust across >= 3 cells (Sharpe>=0.3 + mean>0).

| Stock | TF | Variant | Dir | n | mean% | WR% | Sharpe | RobustCells |
|---|---|---|---|---:|---:|---:|---:|---:|
| **VEDL** | 15min | l_vm2.0_gapoff_rsi40_60 | long | 18 | 0.689 | 72.2 | 0.5090 | 15 |

## Direction comparison (cells with n>=10)

| Direction | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| long | 45591 | -0.0606 | 39.7 | -0.0018 |
| short | 33930 | -0.0331 | 42.5 | 0.0073 |

## Timeframe comparison (cells with n>=10, mean>0)

| TF | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| 5min | 11499 | 0.2869 | 48.0 | 0.0944 |
| 10min | 11635 | 0.2817 | 49.5 | 0.1008 |
| 15min | 11719 | 0.2953 | 49.7 | 0.1049 |

## Volume threshold sweep (cells with n>=10, mean>0)

| vol_mult | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| 1.5x | 17572 | 0.2478 | 48.4 | 0.0867 |
| 2.0x | 12094 | 0.3000 | 49.2 | 0.1040 |
| 3.0x | 5187 | 0.3960 | 51.3 | 0.1361 |

## Exit policy comparison (cells with n>=10, mean>0)

| Exit | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| T_NO | 2446 | 0.3293 | 54.6 | 0.1118 |
| T_HARD_SL | 2881 | 0.3032 | 43.0 | 0.0895 |
| T_ATR_SL_0.3 | 2950 | 0.2683 | 40.6 | 0.0791 |
| T_ATR_SL_0.5 | 2669 | 0.2878 | 50.0 | 0.0940 |
| T_ATR_SL_1.0 | 2510 | 0.3079 | 54.3 | 0.1040 |
| T_CHANDELIER_1.0 | 2567 | 0.2845 | 53.8 | 0.1027 |
| T_CHANDELIER_1.5 | 2495 | 0.3140 | 54.5 | 0.1058 |
| T_CHANDELIER_2.0 | 2465 | 0.3262 | 54.7 | 0.1111 |
| T_R_TARGET_1.0R | 2841 | 0.2450 | 55.2 | 0.1288 |
| T_R_TARGET_1.5R | 2922 | 0.2774 | 50.5 | 0.1103 |
| T_R_TARGET_2.0R | 2940 | 0.2918 | 47.6 | 0.1003 |
| T_R_TARGET_3.0R | 2878 | 0.2846 | 45.4 | 0.0936 |
| T_STEP_TRAIL | 2289 | 0.2289 | 35.3 | 0.0686 |
