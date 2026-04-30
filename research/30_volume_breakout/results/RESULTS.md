# Volume-Confirmed First-Candle Breakout — Backtest Results

## Setup

- Period: 2024-03-01 to 2026-03-25
- Stocks (10): RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC, KOTAKBANK, HINDUNILVR
- Timeframes: 5min, 15min (first candle of session)
- Variant grid: vol_mult ∈ {1.5, 2.0, 3.0}  ×  gap_pct ∈ {0%, 0.3%, 0.5%, off}  ×  RSI ∈ {off, on(40/60)}
- Direction: long & short, evaluated independently
- 13 exit policies tested per signal in parallel
- Total signals fired: **11412**
  - Long: 5723
  - Short: 5689

## Signals fired per stock x direction

| Stock | Long | Short | Total |
|---|---:|---:|---:|
| RELIANCE | 530 | 592 | 1122 |
| TCS | 564 | 504 | 1068 |
| HDFCBANK | 597 | 620 | 1217 |
| INFY | 820 | 882 | 1702 |
| ICICIBANK | 713 | 364 | 1077 |
| SBIN | 820 | 260 | 1080 |
| BHARTIARTL | 303 | 456 | 759 |
| ITC | 378 | 452 | 830 |
| KOTAKBANK | 514 | 934 | 1448 |
| HINDUNILVR | 484 | 625 | 1109 |

## Top 10 configurations across all stocks (by Sharpe, n>=10, mean>0)

| Symbol | TF | Variant | Dir | ExitPolicy | n | mean% | std% | WR% | Payoff | CapEff | Sharpe | Expect% |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RELIANCE | 15min | l_vm2.0_gap0.000_rsi40_60 | long | T_NO | 11 | 1.094 | 0.952 | 90.9 | 6.39 | 0.69 | 1.0451 | 1.094 |
| RELIANCE | 15min | l_vm2.0_gap0.000_rsi40_60 | long | T_ATR_SL_1.0 | 11 | 1.094 | 0.952 | 90.9 | 6.39 | 0.69 | 1.0451 | 1.094 |
| RELIANCE | 15min | l_vm2.0_gap0.000_rsi40_60 | long | T_CHANDELIER_1.0 | 11 | 1.094 | 0.952 | 90.9 | 6.39 | 0.69 | 1.0451 | 1.094 |
| RELIANCE | 15min | l_vm2.0_gap0.000_rsi40_60 | long | T_CHANDELIER_1.5 | 11 | 1.094 | 0.952 | 90.9 | 6.39 | 0.69 | 1.0451 | 1.094 |
| RELIANCE | 15min | l_vm2.0_gap0.000_rsi40_60 | long | T_CHANDELIER_2.0 | 11 | 1.094 | 0.952 | 90.9 | 6.39 | 0.69 | 1.0451 | 1.094 |
| RELIANCE | 15min | l_vm2.0_gap0.000_norsi | long | T_CHANDELIER_1.0 | 12 | 0.954 | 1.023 | 83.3 | 3.12 | 0.63 | 0.7766 | 0.954 |
| RELIANCE | 15min | l_vm2.0_gap0.000_norsi | long | T_NO | 12 | 0.923 | 1.074 | 83.3 | 2.12 | 0.61 | 0.7156 | 0.923 |
| RELIANCE | 15min | l_vm2.0_gap0.000_norsi | long | T_CHANDELIER_2.0 | 12 | 0.923 | 1.074 | 83.3 | 2.12 | 0.61 | 0.7156 | 0.923 |
| RELIANCE | 15min | l_vm2.0_gap0.000_rsi40_60 | long | T_ATR_SL_0.5 | 11 | 0.910 | 1.122 | 81.8 | 2.14 | 0.62 | 0.6636 | 0.910 |
| RELIANCE | 15min | l_vm2.0_gap0.000_norsi | long | T_CHANDELIER_1.5 | 12 | 0.895 | 1.126 | 83.3 | 1.64 | 0.59 | 0.6624 | 0.895 |

## Top 5 by Sharpe (any n>=5)

| Symbol | TF | Variant | Dir | ExitPolicy | n | mean% | WR% | Sharpe |
|---|---|---|---|---|---:|---:|---:|---:|
| RELIANCE | 15min | l_vm3.0_gap0.000_rsi40_60 | long | T_NO | 5 | 1.637 | 100.0 | 1.4666 |
| RELIANCE | 15min | l_vm3.0_gap0.000_rsi40_60 | long | T_ATR_SL_0.5 | 5 | 1.637 | 100.0 | 1.4666 |
| RELIANCE | 15min | l_vm3.0_gap0.000_rsi40_60 | long | T_ATR_SL_1.0 | 5 | 1.637 | 100.0 | 1.4666 |
| RELIANCE | 15min | l_vm3.0_gap0.000_rsi40_60 | long | T_CHANDELIER_1.0 | 5 | 1.637 | 100.0 | 1.4666 |
| RELIANCE | 15min | l_vm3.0_gap0.000_rsi40_60 | long | T_CHANDELIER_1.5 | 5 | 1.637 | 100.0 | 1.4666 |

## Best per-stock signal (any direction, n>=10, mean>0)

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|
| RELIANCE | 15min | l_vm2.0_gap0.000_rsi40_60 | long | T_NO | 11 | 1.094 | 90.9 | 1.0451 | payoff=6.39, capeff=0.69 |
| TCS | 15min | s_vm1.5_gap0.005_norsi | short | T_ATR_SL_0.3 | 12 | 0.655 | 75.0 | 0.5234 | payoff=1.65, capeff=0.60 |
| HDFCBANK | 15min | s_vm1.5_gap0.003_norsi | short | T_R_TARGET_1.0R | 21 | 0.434 | 81.0 | 0.4221 | payoff=1.15, capeff=0.69 |
| INFY | 5min | l_vm3.0_gapoff_rsi40_60 | long | T_R_TARGET_1.0R | 10 | 0.535 | 80.0 | 0.5126 | payoff=1.09, capeff=0.58 |
| ICICIBANK | 15min | l_vm2.0_gap0.000_rsi40_60 | long | T_R_TARGET_1.0R | 14 | 0.358 | 64.3 | 0.3112 | payoff=1.89, capeff=0.51 |
| SBIN | 15min | l_vm2.0_gap0.005_norsi | long | T_NO | 14 | 0.770 | 78.6 | 0.4850 | payoff=2.05, capeff=0.56 |
| BHARTIARTL | 15min | l_vm1.5_gapoff_rsi40_60 | long | T_ATR_SL_0.5 | 11 | 0.780 | 81.8 | 0.6104 | payoff=1.93, capeff=0.53 |
| ITC | 15min | s_vm1.5_gapoff_rsi40_60 | short | T_R_TARGET_1.0R | 13 | 0.499 | 61.5 | 0.2374 | payoff=2.11, capeff=0.44 |
| KOTAKBANK | 5min | s_vm3.0_gap0.000_norsi | short | T_R_TARGET_1.0R | 14 | 0.660 | 71.4 | 0.4464 | payoff=1.83, capeff=0.57 |
| HINDUNILVR | 15min | l_vm2.0_gap0.000_norsi | long | T_R_TARGET_1.0R | 10 | 0.612 | 60.0 | 0.2296 | payoff=1.70, capeff=0.51 |

## Exit policy comparison (averaged across cells with n>=10, mean>0)

| ExitPolicy | n_cells | avg_mean% | avg_WR% | avg_capEff | avg_Sharpe |
|---|---:|---:|---:|---:|---:|
| T_NO | 330 | 0.3084 | 58.3 | 0.24 | 0.1580 |
| T_CHANDELIER_2.0 | 321 | 0.3017 | 57.3 | 0.23 | 0.1512 |
| T_R_TARGET_1.0R | 355 | 0.2359 | 55.3 | 0.29 | 0.1508 |
| T_CHANDELIER_1.5 | 333 | 0.2807 | 58.1 | 0.22 | 0.1458 |
| T_CHANDELIER_1.0 | 359 | 0.2748 | 56.1 | 0.23 | 0.1449 |
| T_ATR_SL_1.0 | 319 | 0.2870 | 57.0 | 0.22 | 0.1423 |
| T_R_TARGET_1.5R | 351 | 0.2399 | 51.4 | 0.26 | 0.1324 |
| T_R_TARGET_2.0R | 358 | 0.2233 | 48.1 | 0.23 | 0.1181 |
| T_R_TARGET_3.0R | 371 | 0.2301 | 46.3 | 0.22 | 0.1141 |
| T_ATR_SL_0.5 | 372 | 0.2257 | 51.0 | 0.19 | 0.1118 |
| T_HARD_SL | 375 | 0.2397 | 44.1 | 0.21 | 0.1091 |
| T_ATR_SL_0.3 | 363 | 0.2183 | 42.2 | 0.21 | 0.0956 |
| T_STEP_TRAIL | 305 | 0.1739 | 36.9 | 0.20 | 0.0855 |

## Direction comparison (cells with n>=10)

| Direction | n_cells | avg_mean% | avg_WR% | avg_Sharpe |
|---|---:|---:|---:|---:|
| long  | 3445 | 0.0956 | 45.3 | 0.0590 |
| short | 3523 | 0.1040 | 46.6 | 0.0632 |

## Volume threshold (vm) sweep — viable cells only

| vol_mult | n_cells | avg_mean% | avg_WR% | avg_Sharpe |
|---|---:|---:|---:|---:|
| 1.5 | 2379 | 0.2189 | 49.2 | 0.1098 |
| 2.0 | 1810 | 0.2707 | 52.0 | 0.1410 |
| 3.0 | 323 | 0.3436 | 56.1 | 0.1802 |
