# Volume-Confirmed First-Candle Breakout — EXPANDED (79 stocks)

## Setup

- Universe: 79 stocks (Cohort A = 10 stocks since 2018, Cohort B = 69 stocks since 2024-03-18)
- Period end: 2026-03-25 (data cap)
- Timeframes: 5min, **10min (NEW)**, 15min — first candle of session
- Variant grid: vol_mult ∈ {1.5, 2.0, 3.0} × gap_pct ∈ {0%, 0.3%, 0.5%, off} × RSI ∈ {off, on(40/60)}
- Direction: long & short
- 13 exit policies tested per signal in parallel
- Total signals fired: **230,180**
  - Long: 138,150
  - Short: 92,030
  - Cohort A: 58,384
  - Cohort B: 171,796

## Top 10 configurations across all stocks (by Sharpe, n>=10, mean>0)

| Symbol | Coh | TF | Variant | Dir | ExitPolicy | n | mean% | WR% | Payoff | Sharpe |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| HAL | B | 5min | s_vm1.5_gap0.005_rsi40_60 | short | T_NO | 10 | 1.409 | 90.0 | 17.46 | 1.3545 |
| HAL | B | 5min | s_vm1.5_gap0.005_rsi40_60 | short | T_ATR_SL_1.0 | 10 | 1.409 | 90.0 | 17.46 | 1.3545 |
| HAL | B | 5min | s_vm1.5_gap0.005_rsi40_60 | short | T_CHANDELIER_2.0 | 10 | 1.409 | 90.0 | 17.46 | 1.3545 |
| HAL | B | 5min | s_vm1.5_gap0.005_rsi40_60 | short | T_CHANDELIER_1.5 | 10 | 1.405 | 90.0 | 17.42 | 1.3505 |
| HAL | B | 5min | s_vm1.5_gap0.005_rsi40_60 | short | T_CHANDELIER_1.0 | 10 | 1.471 | 90.0 | 18.23 | 1.2993 |
| HAL | B | 5min | s_vm1.5_gap0.003_rsi40_60 | short | T_NO | 11 | 1.314 | 90.9 | 16.13 | 1.2691 |
| HAL | B | 5min | s_vm1.5_gap0.003_rsi40_60 | short | T_ATR_SL_1.0 | 11 | 1.314 | 90.9 | 16.13 | 1.2691 |
| HAL | B | 5min | s_vm1.5_gap0.003_rsi40_60 | short | T_CHANDELIER_2.0 | 11 | 1.314 | 90.9 | 16.13 | 1.2691 |
| HAL | B | 5min | s_vm1.5_gap0.003_rsi40_60 | short | T_CHANDELIER_1.5 | 11 | 1.311 | 90.9 | 16.08 | 1.2659 |
| HAL | B | 5min | s_vm1.5_gap0.003_rsi40_60 | short | T_CHANDELIER_1.0 | 11 | 1.370 | 90.9 | 16.81 | 1.2193 |

## Top 15 Volume Leaders (per-stock best Sharpe, n>=10)

| Rank | Symbol | Coh | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe | HiQ cells | MidQ cells |
|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | HAL | B | 5min | s_vm1.5_gap0.005_rsi40_60 | short | T_NO | 10 | 1.409 | 90.0 | 17.46 | 1.3545 | 96 | 155 |
| 2 | HDFCAMC | B | 10min | l_vm1.5_gap0.005_rsi40_60 | long | T_NO | 11 | 1.461 | 90.9 | 9.38 | 0.9957 | 166 | 271 |
| 3 | RBLBANK | B | 10min | l_vm2.0_gap0.000_norsi | long | T_R_TARGET_1.0R | 13 | 1.079 | 92.3 | 0.63 | 0.9194 | 20 | 55 |
| 4 | RELIANCE | A | 15min | s_vm3.0_gap0.000_rsi40_60 | short | T_R_TARGET_1.0R | 14 | 1.166 | 92.9 | 1.84 | 0.8806 | 203 | 400 |
| 5 | PERSISTENT | B | 5min | l_vm3.0_gapoff_norsi | long | T_R_TARGET_1.0R | 11 | 1.424 | 81.8 | 2.22 | 0.7962 | 8 | 20 |
| 6 | LAURUSLABS | B | 10min | l_vm2.0_gapoff_rsi40_60 | long | T_NO | 13 | 1.422 | 84.6 | 1.86 | 0.7719 | 83 | 154 |
| 7 | COCHINSHIP | B | 15min | s_vm1.5_gap0.000_norsi | short | T_R_TARGET_1.0R | 14 | 0.695 | 78.6 | 2.55 | 0.6930 | 22 | 38 |
| 8 | EICHERMOT | B | 5min | l_vm3.0_gapoff_norsi | long | T_R_TARGET_1.0R | 10 | 0.504 | 80.0 | 1.41 | 0.6530 | 8 | 10 |
| 9 | GODFRYPHLP | B | 10min | l_vm2.0_gap0.000_norsi | long | T_CHANDELIER_1.0 | 10 | 1.884 | 80.0 | 1.89 | 0.6114 | 75 | 192 |
| 10 | AARTIIND | B | 10min | s_vm2.0_gapoff_norsi | short | T_NO | 10 | 1.942 | 80.0 | 3.28 | 0.6026 | 28 | 78 |
| 11 | 3MINDIA | B | 5min | s_vm1.5_gapoff_norsi | short | T_R_TARGET_1.0R | 13 | 0.599 | 84.6 | 0.83 | 0.5924 | 8 | 18 |
| 12 | ACC | B | 5min | s_vm2.0_gapoff_norsi | short | T_R_TARGET_1.0R | 15 | 0.850 | 86.7 | 1.73 | 0.5743 | 6 | 9 |
| 13 | LT | B | 15min | l_vm2.0_gap0.003_rsi40_60 | long | T_R_TARGET_3.0R | 11 | 0.676 | 81.8 | 0.94 | 0.5737 | 18 | 59 |
| 14 | WIPRO | B | 10min | s_vm2.0_gap0.003_rsi40_60 | short | T_R_TARGET_1.0R | 12 | 0.678 | 83.3 | 1.05 | 0.5724 | 14 | 62 |
| 15 | ADANIGREEN | B | 15min | s_vm1.5_gap0.003_rsi40_60 | short | T_NO | 11 | 2.953 | 81.8 | 1.70 | 0.5566 | 6 | 13 |

## Promote candidates (Sharpe>=0.5, n>=15, MidQ_cells>=3)

These pass the robustness gate — best cell strong, n>=15, AND signal consistent across at least 3 different variants (Sharpe>=0.3 each).

| Symbol | Coh | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe | HiQ | MidQ | LongBestSharpe | ShortBestSharpe |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACC | B | 5min | s_vm2.0_gapoff_norsi | short | T_R_TARGET_1.0R | 15 | 0.850 | 86.7 | 0.5743 | 6 | 9 | 0.104 | 0.574 |
| VEDL | B | 15min | l_vm2.0_gapoff_rsi40_60 | long | T_R_TARGET_1.0R | 18 | 0.689 | 72.2 | 0.5090 | 7 | 15 | 0.509 | 0.184 |

## Direction asymmetry (top 20 leaders)

| Symbol | Long_best_Sharpe (n) | Short_best_Sharpe (n) | Bias |
|---|---:|---:|---|
| HAL | 0.174 (n=27) | 1.354 (n=10) | SHORT |
| HDFCAMC | 0.996 (n=11) | 0.197 (n=12) | LONG |
| RBLBANK | 0.919 (n=13) | 0.174 (n=15) | LONG |
| RELIANCE | 0.408 (n=14) | 0.881 (n=14) | SHORT |
| PERSISTENT | 0.796 (n=11) | 0.147 (n=12) | LONG |
| LAURUSLABS | 0.772 (n=13) | 0.000 (n=0) | LONG |
| COCHINSHIP | 0.149 (n=24) | 0.693 (n=14) | SHORT |
| EICHERMOT | 0.653 (n=10) | 0.116 (n=12) | LONG |
| GODFRYPHLP | 0.611 (n=10) | 0.468 (n=11) | both |
| AARTIIND | 0.089 (n=12) | 0.603 (n=10) | SHORT |
| 3MINDIA | 0.385 (n=11) | 0.592 (n=13) | SHORT |
| ACC | 0.104 (n=18) | 0.574 (n=15) | SHORT |
| LT | 0.574 (n=11) | 0.138 (n=13) | LONG |
| WIPRO | 0.043 (n=10) | 0.572 (n=12) | SHORT |
| ADANIGREEN | 0.111 (n=18) | 0.557 (n=11) | SHORT |
| COFORGE | 0.553 (n=13) | 0.396 (n=11) | both |
| BAJAJ-AUTO | 0.526 (n=13) | 0.127 (n=11) | LONG |
| VEDL | 0.509 (n=18) | 0.184 (n=10) | LONG |
| BAJFINANCE | 0.160 (n=11) | 0.505 (n=13) | SHORT |
| ASTRAL | 0.452 (n=14) | 0.496 (n=11) | both |

## Timeframe sweet spot (top 20 leaders)

| Symbol | 5min | 10min | 15min | Best |
|---|---:|---:|---:|---|
| HAL | 1.354 | 1.199 | 0.707 | 5min |
| HDFCAMC | 0.524 | 0.996 | 0.913 | 10min |
| RBLBANK | 0.448 | 0.919 | 0.177 | 10min |
| RELIANCE | 0.440 | 0.657 | 0.881 | 15min |
| PERSISTENT | 0.796 | 0.119 | 0.248 | 5min |
| LAURUSLABS | 0.337 | 0.772 | 0.745 | 10min |
| COCHINSHIP | 0.301 | 0.114 | 0.693 | 15min |
| EICHERMOT | 0.653 | 0.249 | 0.116 | 5min |
| GODFRYPHLP | 0.565 | 0.611 | 0.412 | 10min |
| AARTIIND | 0.331 | 0.603 | 0.451 | 10min |
| 3MINDIA | 0.592 | 0.436 | 0.262 | 5min |
| ACC | 0.574 | 0.195 | 0.152 | 5min |
| LT | 0.393 | 0.344 | 0.574 | 15min |
| WIPRO | 0.223 | 0.572 | 0.521 | 10min |
| ADANIGREEN | 0.141 | 0.268 | 0.557 | 15min |
| COFORGE | 0.553 | 0.396 | 0.180 | 5min |
| BAJAJ-AUTO | 0.321 | 0.526 | 0.467 | 10min |
| VEDL | 0.240 | 0.231 | 0.509 | 15min |
| BAJFINANCE | 0.368 | 0.505 | 0.430 | 10min |
| ASTRAL | 0.496 | 0.459 | 0.000 | 5min |

## Cohort A (10 stocks, 8 yrs) vs Cohort B (69 stocks, 2 yrs)

| Cohort | n_stocks | avg_best_Sharpe | avg_best_mean% |
|---|---:|---:|---:|
| A (long history) | 10 | 0.3372 | 0.6565 |
| B (2-year only)  | 109 | 0.3146 | 0.6906 |

## Exit policy comparison (across cells with n>=10, mean>0)

| ExitPolicy | n_cells | avg_mean% | avg_WR% | avg_capEff | avg_Sharpe |
|---|---:|---:|---:|---:|---:|
| T_R_TARGET_1.0R | 4093 | 0.2594 | 54.9 | 0.22 | 0.1178 |
| T_NO | 3747 | 0.4071 | 53.9 | 0.20 | 0.1125 |
| T_CHANDELIER_2.0 | 3710 | 0.3925 | 53.9 | 0.19 | 0.1095 |
| T_CHANDELIER_1.5 | 3782 | 0.3834 | 53.7 | 0.19 | 0.1063 |
| T_ATR_SL_1.0 | 3812 | 0.3837 | 53.3 | 0.19 | 0.1053 |
| T_CHANDELIER_1.0 | 3734 | 0.3375 | 53.4 | 0.18 | 0.1051 |
| T_R_TARGET_1.5R | 4237 | 0.2963 | 49.9 | 0.22 | 0.1011 |
| T_R_TARGET_2.0R | 4370 | 0.3209 | 47.2 | 0.21 | 0.0935 |
| T_ATR_SL_0.5 | 4143 | 0.3541 | 48.3 | 0.18 | 0.0918 |
| T_R_TARGET_3.0R | 4393 | 0.3416 | 44.9 | 0.20 | 0.0893 |
| T_HARD_SL | 4386 | 0.3652 | 42.4 | 0.20 | 0.0845 |
| T_ATR_SL_0.3 | 4512 | 0.3460 | 39.6 | 0.20 | 0.0796 |
| T_STEP_TRAIL | 3361 | 0.2614 | 34.5 | 0.18 | 0.0655 |

## Timeframe comparison (cells with n>=10, mean>0)

| TF | n_cells | avg_mean% | avg_WR% | avg_Sharpe |
|---|---:|---:|---:|---:|
| 5min | 17216 | 0.3416 | 47.4 | 0.0918 |
| 10min | 17154 | 0.3441 | 48.7 | 0.0998 |
| 15min | 17910 | 0.3406 | 48.9 | 0.0990 |

## Comparison to prior 10-stock run (research/30)

- RELIANCE best in this expanded run: TF=15min, Variant=s_vm3.0_gap0.000_rsi40_60, Dir=short, Exit=T_R_TARGET_1.0R, n=14, mean%=1.166, Sharpe=0.8806
- Prior run RELIANCE best: 15min, l_vm2.0_gap0.000_rsi40_60, T_NO, n=11, mean=1.094%, Sharpe=1.0451
- Stocks beating the RELIANCE prior-best Sharpe (1.045): ['HAL']
