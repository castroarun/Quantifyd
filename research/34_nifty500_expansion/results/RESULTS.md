# CPR-Compression Range Breakout (CCRB) — Backtest Results

## Setup

- Universe: 79 stocks (10 Cohort A since 2018-01-01; 69 Cohort B since 2024-03-18)
- Period end: 2026-03-25 (data cap)
- Timeframes: 5min, 10min, 15min
- Daily-bar setup filter:
  - today_cpr_width / today_open <= today_narrow ∈ {0.30%, 0.40%, 0.50%}
  - yesterday_ctx ∈ {W (wide CPR >= 0.50/0.65/0.80%), N (narrow range <= 0.50/0.70/0.90%), W_OR_N, W_AND_N}
- Intraday trigger: first fresh transition past prev_day_high (long) / prev_day_low (short), 09:20-14:00 IST
- Volume filter: off, vm1.5, vm2.0 (vs 20-day same-bar-position avg)
- Direction: long & short (independent)
- 13 exit policies tested per signal in parallel
- Total signal rows: **875,493** (long: 470,445, short: 405,048)
- Cohort A signals: 278,611; Cohort B signals: 596,882
- Ranked cells (n>=5): **536,471**
- Per-stock leaders found: **94**

## Top 10 configurations across all stocks (n>=15, mean>0)

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_STEP_TRAIL | 15 | 0.669 | 86.7 | 999.00 | 1.2834 |
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_NO | 15 | 0.666 | 86.7 | 29.94 | 1.2624 |
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_HARD_SL | 15 | 0.666 | 86.7 | 29.94 | 1.2624 |
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_ATR_SL_0.5 | 15 | 0.666 | 86.7 | 29.94 | 1.2624 |
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_ATR_SL_1.0 | 15 | 0.666 | 86.7 | 29.94 | 1.2624 |
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_CHANDELIER_1.0 | 15 | 0.666 | 86.7 | 29.94 | 1.2624 |
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_CHANDELIER_1.5 | 15 | 0.666 | 86.7 | 29.94 | 1.2624 |
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_CHANDELIER_2.0 | 15 | 0.666 | 86.7 | 29.94 | 1.2624 |
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_R_TARGET_1.5R | 15 | 0.666 | 86.7 | 29.94 | 1.2624 |
| HDFCLIFE | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_R_TARGET_2.0R | 15 | 0.666 | 86.7 | 29.94 | 1.2624 |

## Top 15 CPR-Compression Leaders (per-stock best cell, n>=10)

| # | Stock | Cohort | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe | RobustCells | Promote? |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **DLF** | B | 5min | t0.0030_ctxW_w0.0050_n0.0070_5min_vm2.0_l | long | T_NO | 12 | 0.556 | 100.0 | 999.00 | 4.6593 | 1776 | - |
| 2 | **HAL** | B | 15min | t0.0030_ctxW_OR_N_w0.0080_n0.0070_15min_off_s | short | T_R_TARGET_1.0R | 10 | 0.702 | 100.0 | 999.00 | 4.6564 | 696 | - |
| 3 | **APOLLOHOSP** | B | 10min | t0.0040_ctxW_OR_N_w0.0065_n0.0090_10min_vm1.5_l | long | T_NO | 10 | 0.491 | 100.0 | 999.00 | 3.3422 | 490 | - |
| 4 | **HDFCAMC** | B | 10min | t0.0040_ctxW_w0.0065_n0.0070_10min_vm1.5_s | short | T_STEP_TRAIL | 11 | 0.773 | 90.9 | 999.00 | 1.5299 | 641 | - |
| 5 | **ASHOKLEY** | B | 15min | t0.0030_ctxW_w0.0050_n0.0070_15min_vm2.0_s | short | T_NO | 10 | 0.586 | 80.0 | 999.00 | 1.3123 | 940 | - |
| 6 | **HDFCLIFE** | B | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_STEP_TRAIL | 15 | 0.669 | 86.7 | 999.00 | 1.2834 | 352 | YES |
| 7 | **BDL** | B | 5min | t0.0030_ctxW_w0.0050_n0.0070_5min_vm1.5_l | long | T_NO | 14 | 1.406 | 85.7 | 17.93 | 1.1878 | 945 | - |
| 8 | **HINDUNILVR** | A | 15min | t0.0040_ctxW_w0.0065_n0.0070_15min_vm1.5_s | short | T_NO | 12 | 0.774 | 83.3 | 999.00 | 1.1017 | 168 | - |
| 9 | **ADANIGREEN** | B | 15min | t0.0030_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_NO | 14 | 0.751 | 85.7 | 18.51 | 1.0459 | 206 | - |
| 10 | **ITC** | A | 10min | t0.0030_ctxW_OR_N_w0.0080_n0.0090_10min_vm2.0_s | short | T_ATR_SL_0.3 | 18 | 0.525 | 88.9 | 1.40 | 1.0269 | 296 | YES |
| 11 | **IDFCFIRSTB** | B | 5min | t0.0030_ctxW_w0.0065_n0.0070_5min_off_l | long | T_R_TARGET_1.0R | 18 | 0.709 | 88.9 | 2.42 | 1.0005 | 1680 | YES |
| 12 | **CHENNPETRO** | B | 5min | t0.0040_ctxW_w0.0050_n0.0070_5min_vm1.5_s | short | T_NO | 10 | 1.046 | 80.0 | 3.29 | 0.9813 | 439 | - |
| 13 | **ASIANPAINT** | B | 5min | t0.0030_ctxW_w0.0065_n0.0070_5min_off_s | short | T_NO | 10 | 0.649 | 80.0 | 27.68 | 0.9647 | 266 | - |
| 14 | **HCLTECH** | B | 10min | t0.0040_ctxW_OR_N_w0.0065_n0.0090_10min_vm1.5_s | short | T_NO | 12 | 0.434 | 83.3 | 82.78 | 0.9616 | 400 | - |
| 15 | **HDFCBANK** | A | 10min | t0.0050_ctxW_OR_N_w0.0065_n0.0070_10min_vm2.0_l | long | T_NO | 12 | 0.764 | 83.3 | 3.96 | 0.9488 | 572 | - |

## Promote candidates (9 stocks pass robustness gate)

Gate: best-cell Sharpe >= 0.5 AND n >= 15 AND robust across >= 3 cells (Sharpe>=0.3 + mean>0).

| Stock | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe | RobustCells |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **HDFCLIFE** | 15min | t0.0050_ctxW_OR_N_w0.0080_n0.0090_15min_off_s | short | T_STEP_TRAIL | 15 | 0.669 | 86.7 | 1.2834 | 352 |
| **ITC** | 10min | t0.0030_ctxW_OR_N_w0.0080_n0.0090_10min_vm2.0_s | short | T_ATR_SL_0.3 | 18 | 0.525 | 88.9 | 1.0269 | 296 |
| **IDFCFIRSTB** | 5min | t0.0030_ctxW_w0.0065_n0.0070_5min_off_l | long | T_R_TARGET_1.0R | 18 | 0.709 | 88.9 | 1.0005 | 1680 |
| **SUZLON** | 15min | t0.0050_ctxW_w0.0080_n0.0070_15min_off_s | short | T_NO | 16 | 0.863 | 81.2 | 0.6917 | 408 |
| **RVNL** | 10min | t0.0030_ctxW_OR_N_w0.0050_n0.0090_10min_off_s | short | T_CHANDELIER_2.0 | 15 | 1.152 | 80.0 | 0.6723 | 464 |
| **BHARTIARTL** | 5min | t0.0030_ctxW_w0.0080_n0.0070_5min_vm2.0_s | short | T_R_TARGET_1.0R | 20 | 1.450 | 80.0 | 0.6672 | 882 |
| **BAJAJFINSV** | 15min | t0.0030_ctxW_w0.0050_n0.0070_15min_vm1.5_l | long | T_ATR_SL_0.3 | 16 | 0.402 | 87.5 | 0.6396 | 165 |
| **REDINGTON** | 10min | t0.0050_ctxW_w0.0050_n0.0070_10min_off_s | short | T_NO | 15 | 0.866 | 73.3 | 0.5184 | 164 |
| **CDSL** | 5min | t0.0030_ctxW_OR_N_w0.0080_n0.0090_5min_off_l | long | T_R_TARGET_1.0R | 18 | 0.701 | 77.8 | 0.5102 | 164 |

## Direction comparison (cells with n>=10)

| Direction | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| long | 200668 | -0.0199 | 44.7 | 0.0329 |
| short | 161031 | 0.0206 | 49.0 | 0.0450 |

## Timeframe comparison (cells with n>=10, mean>0)

| TF | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| 5min | 71601 | 0.3271 | 55.5 | 0.1673 |
| 10min | 60774 | 0.2946 | 56.0 | 0.1608 |
| 15min | 52019 | 0.2664 | 55.0 | 0.1504 |

## today_narrow_threshold sweep (cells with n>=10, mean>0)

| today_narrow | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| 0.30% | 51453 | 0.3206 | 56.5 | 0.1797 |
| 0.40% | 61636 | 0.2967 | 55.7 | 0.1611 |
| 0.50% | 71305 | 0.2860 | 54.6 | 0.1458 |

## yesterday_ctx breakdown (cells with n>=10, mean>0)

| ctx | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| ctxW | 184394 | 0.2992 | 55.5 | 0.1604 |
| ctxN | 0 | 0.0000 | 0.0 | 0.0000 |
| ctxW_OR_N | 140064 | 0.2961 | 55.5 | 0.1594 |
| ctxW_AND_N | 0 | 0.0000 | 0.0 | 0.0000 |

## Volume mode breakdown (cells with n>=10, mean>0)

| vol_mode | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| off | 114190 | 0.2609 | 54.6 | 0.1345 |
| vm1.5 | 43836 | 0.3538 | 56.8 | 0.2024 |
| vm2.0 | 26368 | 0.3745 | 57.3 | 0.2026 |

## Exit policy comparison (cells with n>=10, mean>0)

| Exit | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| T_NO | 14349 | 0.3134 | 56.2 | 0.1677 |
| T_HARD_SL | 14125 | 0.3077 | 56.5 | 0.1651 |
| T_ATR_SL_0.3 | 14719 | 0.2595 | 50.7 | 0.1260 |
| T_ATR_SL_0.5 | 14316 | 0.2918 | 55.4 | 0.1502 |
| T_ATR_SL_1.0 | 14236 | 0.3096 | 56.4 | 0.1666 |
| T_CHANDELIER_1.0 | 14126 | 0.2958 | 56.2 | 0.1624 |
| T_CHANDELIER_1.5 | 14097 | 0.3045 | 56.6 | 0.1678 |
| T_CHANDELIER_2.0 | 14280 | 0.3107 | 56.3 | 0.1681 |
| T_R_TARGET_1.0R | 14473 | 0.2914 | 56.9 | 0.1665 |
| T_R_TARGET_1.5R | 14324 | 0.3080 | 56.5 | 0.1660 |
| T_R_TARGET_2.0R | 14366 | 0.3110 | 56.3 | 0.1642 |
| T_R_TARGET_3.0R | 14172 | 0.3124 | 56.5 | 0.1652 |
| T_STEP_TRAIL | 12811 | 0.2731 | 51.0 | 0.1492 |

---

## Comparison vs research/30b (volume-breakout)

- Stocks in CCRB leaderboard: **94**
- Stocks in vol-breakout leaderboard: **78**
- In both: **57**  |  Only CCRB: 37  |  Only vol-breakout: 21

### Where CCRB beats vol-breakout (43 stocks)

(These are CPR-compression specialists.)

| Symbol | Coh | CCRB TF/Dir/n/Sharpe | Vol TF/Dir/n/Sharpe | SameDir | SameTF | Δ |
|---|---|---|---|---|---|---:|
| DLF | B | 5min/long/12/4.659 | 5min/short/14/0.356 | N | Y | +4.304 |
| HAL | B | 15min/short/10/4.656 | 5min/short/10/1.354 | Y | N | +3.302 |
| APOLLOHOSP | B | 10min/long/10/3.342 | 10min/short/15/0.114 | N | Y | +3.228 |
| HDFCLIFE | B | 15min/short/15/1.283 | 10min/short/12/0.174 | Y | N | +1.110 |
| HDFCBANK | A | 10min/long/12/0.949 | 15min/short/46/0.168 | N | N | +0.781 |
| HINDUNILVR | A | 15min/short/12/1.102 | 5min/short/24/0.352 | Y | N | +0.749 |
| IDFCFIRSTB | B | 5min/long/18/1.000 | 5min/long/17/0.273 | Y | Y | +0.728 |
| ITC | A | 10min/short/18/1.027 | 15min/short/17/0.314 | Y | N | +0.713 |
| HCLTECH | B | 10min/short/12/0.962 | 10min/short/10/0.249 | Y | Y | +0.713 |
| AXISBANK | B | 15min/long/12/0.916 | 15min/long/11/0.251 | Y | Y | +0.665 |
| NTPC | B | 10min/short/10/0.814 | 15min/short/11/0.181 | Y | N | +0.634 |
| BANKBARODA | B | 10min/short/10/0.778 | 15min/short/11/0.200 | Y | N | +0.578 |
| DIVISLAB | B | 5min/long/11/0.712 | 10min/long/14/0.173 | Y | N | +0.538 |
| CIPLA | B | 15min/long/10/0.719 | 15min/short/19/0.184 | N | Y | +0.535 |
| BRITANNIA | B | 10min/long/10/0.655 | 15min/short/12/0.127 | N | N | +0.528 |
| ASIANPAINT | B | 5min/short/10/0.965 | 10min/long/13/0.481 | N | N | +0.484 |
| TITAN | B | 5min/long/10/0.729 | 10min/long/17/0.264 | Y | N | +0.465 |
| ADANIENT | B | 10min/short/10/0.723 | 15min/short/13/0.261 | Y | N | +0.463 |
| INFY | A | 10min/long/14/0.664 | 5min/long/29/0.208 | Y | N | +0.457 |
| BHARTIARTL | A | 5min/short/20/0.667 | 5min/short/24/0.216 | Y | Y | +0.451 |
| ULTRACEMCO | B | 10min/short/10/0.846 | 5min/short/11/0.399 | Y | N | +0.447 |
| BEL | B | 5min/short/10/0.732 | 15min/short/11/0.302 | Y | N | +0.429 |
| KOTAKBANK | A | 10min/short/10/0.732 | 10min/long/22/0.336 | N | Y | +0.396 |
| TECHM | B | 5min/long/10/0.565 | 5min/long/16/0.184 | Y | Y | +0.382 |
| HEROMOTOCO | B | 15min/short/20/0.458 | 10min/short/11/0.124 | Y | N | +0.334 |
| BAJAJFINSV | B | 15min/long/16/0.640 | 15min/long/10/0.333 | Y | Y | +0.307 |
| FEDERALBNK | B | 5min/long/14/0.591 | 15min/long/14/0.297 | Y | N | +0.294 |
| ADANIPORTS | B | 10min/long/14/0.610 | 10min/short/12/0.325 | N | Y | +0.285 |
| MARUTI | B | 15min/long/13/0.668 | 10min/long/24/0.414 | Y | N | +0.254 |
| ICICIBANK | A | 10min/long/14/0.643 | 15min/short/17/0.393 | N | N | +0.250 |

### Where vol-breakout beats CCRB (14 stocks)

(These are volume specialists.)

| Symbol | Coh | CCRB TF/Dir/n/Sharpe | Vol TF/Dir/n/Sharpe | SameDir | SameTF | Δ |
|---|---|---|---|---|---|---:|
| VEDL | B | 5min/long/10/0.126 | 15min/long/18/0.509 | Y | N | -0.383 |
| RELIANCE | A | 5min/long/10/0.560 | 15min/short/14/0.881 | N | N | -0.321 |
| LT | B | 10min/long/10/0.258 | 15min/long/11/0.574 | Y | N | -0.315 |
| PERSISTENT | B | 10min/short/10/0.483 | 5min/long/11/0.796 | N | N | -0.314 |
| IOC | B | 10min/short/10/0.177 | 5min/short/11/0.461 | Y | N | -0.284 |
| CHOLAFIN | B | 10min/long/19/0.259 | 10min/long/11/0.434 | Y | Y | -0.176 |
| BAJAJ-AUTO | B | 10min/short/10/0.376 | 10min/long/13/0.526 | N | Y | -0.150 |
| SBILIFE | B | 10min/long/14/0.068 | 5min/short/11/0.211 | N | N | -0.143 |
| M&M | B | 5min/long/14/0.143 | 5min/short/15/0.214 | N | Y | -0.071 |
| POWERGRID | B | 10min/short/11/0.310 | 10min/long/10/0.359 | N | Y | -0.050 |
| COFORGE | B | 5min/long/12/0.514 | 5min/long/13/0.553 | Y | Y | -0.039 |
| DRREDDY | B | 10min/long/34/0.104 | 5min/long/10/0.132 | Y | N | -0.027 |
| INDUSINDBK | B | 5min/short/46/0.166 | 10min/long/23/0.190 | N | N | -0.024 |
| SBIN | A | 15min/long/27/0.281 | 15min/long/21/0.284 | Y | Y | -0.003 |

### Stocks robust on BOTH (CCRB Sharpe >= 0.4 AND vol Sharpe >= 0.4): 8

| Symbol | Coh | CCRB Sharpe | Vol Sharpe | SameDir | SameTF |
|---|---|---:|---:|---|---|
| **HAL** | B | 4.656 | 1.354 | Y | N |
| **EICHERMOT** | B | 0.799 | 0.653 | Y | Y |
| **ASIANPAINT** | B | 0.965 | 0.481 | N | N |
| **RELIANCE** | A | 0.560 | 0.881 | N | N |
| **PERSISTENT** | B | 0.483 | 0.796 | N | N |
| **BAJFINANCE** | B | 0.691 | 0.505 | Y | N |
| **MARUTI** | B | 0.668 | 0.414 | Y | N |
| **COFORGE** | B | 0.514 | 0.553 | Y | Y |

### Stocks ONLY found by CCRB (not by vol-breakout): 37

| Symbol | Coh | TF | Dir | n | Sharpe | RobustCells |
|---|---|---|---|---:|---:|---:|
| HDFCAMC | B | 10min | short | 11 | 1.5299 | 641 |
| ASHOKLEY | B | 15min | short | 10 | 1.3123 | 940 |
| BDL | B | 5min | long | 14 | 1.1878 | 945 |
| ADANIGREEN | B | 15min | short | 14 | 1.0459 | 206 |
| CHENNPETRO | B | 5min | short | 10 | 0.9813 | 439 |
| DIXON | B | 15min | short | 14 | 0.7121 | 197 |
| GODREJCP | B | 10min | long | 14 | 0.7008 | 1587 |
| SUZLON | B | 15min | short | 16 | 0.6917 | 408 |
| INDIGO | B | 15min | long | 14 | 0.6740 | 374 |
| RVNL | B | 10min | short | 15 | 0.6723 | 464 |
| BHEL | B | 15min | long | 10 | 0.6521 | 166 |
| CANBK | B | 5min | long | 10 | 0.6345 | 743 |
| HINDZINC | B | 10min | short | 10 | 0.6302 | 157 |
| RECLTD | B | 5min | long | 11 | 0.6141 | 49 |
| BSE | B | 5min | long | 10 | 0.5935 | 703 |
| GMDCLTD | B | 5min | short | 10 | 0.5635 | 48 |
| REDINGTON | B | 10min | short | 15 | 0.5184 | 164 |
| CDSL | B | 5min | long | 18 | 0.5102 | 164 |
| AUBANK | B | 15min | long | 10 | 0.4989 | 142 |
| HINDPETRO | B | 5min | long | 12 | 0.4989 | 9 |

### Stocks ONLY in vol-breakout leaderboard (no CCRB leader): 21

AMBUJACEM, BPCL, COLPAL, DABUR, DELHIVERY, GAIL, GODREJPROP, GRASIM, HAVELLS, IRCTC, JINDALSTEL, MARICO, MCX, NESTLEIND, ONGC, PIDILITIND, SHREECEM, SIEMENS, TATACONSUM, VOLTAS, WIPRO

---

## Honest Read

- Average best-cell Sharpe across stocks in BOTH leaderboards: CCRB=0.746 vs vol-breakout=0.322
- CCRB beats vol-breakout in 43 / 57 shared stocks (75%).
- Promote-gate passes: CCRB=9; vol-breakout=1.
- Direction agreement (same best dir): 37 / 57 stocks.
- Timeframe agreement (same best TF): 19 / 57 stocks.

Interpretation: CCRB and vol-breakout target different setups. CCRB requires a
daily-bar geometric filter (CPR compression) that gates the day; vol-breakout
triggers off the first candle's volume regardless of CPR. Stocks where they
disagree on direction or timeframe are likely catching different regimes.
Stocks robust on BOTH are the highest-conviction names — the daily geometric
filter and the first-bar volume confirmation could be combined as a higher-bar
entry rule.
