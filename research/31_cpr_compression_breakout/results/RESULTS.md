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
- Total signal rows: **447,528** (long: 236,188, short: 211,340)
- Cohort A signals: 177,576; Cohort B signals: 269,952
- Ranked cells (n>=5): **346,528**
- Per-stock leaders found: **74**

## Top 10 configurations across all stocks (n>=15, mean>0)

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| GRASIM | 15min | t0.0050_ctxW_w0.0050_n0.0070_15min_off_s | short | T_R_TARGET_1.0R | 19 | 0.707 | 73.7 | 2.64 | 0.6383 |
| GRASIM | 15min | t0.0050_ctxW_OR_N_w0.0050_n0.0050_15min_off_s | short | T_R_TARGET_1.0R | 19 | 0.707 | 73.7 | 2.64 | 0.6383 |
| GRASIM | 15min | t0.0050_ctxW_OR_N_w0.0050_n0.0070_15min_off_s | short | T_R_TARGET_1.0R | 19 | 0.707 | 73.7 | 2.64 | 0.6383 |
| GRASIM | 15min | t0.0050_ctxW_w0.0050_n0.0070_15min_off_s | short | T_R_TARGET_1.5R | 19 | 0.709 | 73.7 | 2.65 | 0.6072 |
| GRASIM | 15min | t0.0050_ctxW_OR_N_w0.0050_n0.0050_15min_off_s | short | T_R_TARGET_1.5R | 19 | 0.709 | 73.7 | 2.65 | 0.6072 |
| GRASIM | 15min | t0.0050_ctxW_OR_N_w0.0050_n0.0070_15min_off_s | short | T_R_TARGET_1.5R | 19 | 0.709 | 73.7 | 2.65 | 0.6072 |
| GRASIM | 15min | t0.0050_ctxW_w0.0050_n0.0070_15min_off_s | short | T_STEP_TRAIL | 19 | 0.735 | 73.7 | 2.94 | 0.6059 |
| GRASIM | 15min | t0.0050_ctxW_OR_N_w0.0050_n0.0050_15min_off_s | short | T_STEP_TRAIL | 19 | 0.735 | 73.7 | 2.94 | 0.6059 |
| GRASIM | 15min | t0.0050_ctxW_OR_N_w0.0050_n0.0070_15min_off_s | short | T_STEP_TRAIL | 19 | 0.735 | 73.7 | 2.94 | 0.6059 |
| GRASIM | 15min | t0.0050_ctxW_w0.0050_n0.0070_15min_off_s | short | T_NO | 19 | 0.726 | 73.7 | 2.70 | 0.5934 |

## Top 15 CPR-Compression Leaders (per-stock best cell, n>=10)

| # | Stock | Cohort | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe | RobustCells | Promote? |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **NESTLEIND** | B | 5min | t0.0030_ctxW_OR_N_w0.0065_n0.0090_5min_off_s | short | T_R_TARGET_1.0R | 10 | 0.648 | 90.0 | 10.87 | 1.4209 | 35 | - |
| 2 | **GRASIM** | B | 15min | t0.0050_ctxW_w0.0065_n0.0070_15min_off_s | short | T_NO | 11 | 0.842 | 90.9 | 1.43 | 1.2157 | 487 | - |
| 3 | **APOLLOHOSP** | B | 15min | t0.0050_ctxW_w0.0065_n0.0070_15min_off_l | long | T_R_TARGET_1.0R | 11 | 0.422 | 81.8 | 5.00 | 0.8657 | 192 | - |
| 4 | **DLF** | B | 5min | t0.0040_ctxW_w0.0050_n0.0070_5min_vm1.5_s | short | T_R_TARGET_1.0R | 10 | 0.851 | 90.0 | 0.86 | 0.8634 | 332 | - |
| 5 | **ULTRACEMCO** | B | 10min | t0.0040_ctxW_OR_N_w0.0080_n0.0090_10min_off_s | short | T_ATR_SL_0.5 | 10 | 0.706 | 90.0 | 0.81 | 0.8464 | 70 | - |
| 6 | **NTPC** | B | 10min | t0.0030_ctxW_w0.0065_n0.0070_10min_off_s | short | T_R_TARGET_1.0R | 10 | 0.786 | 90.0 | 2.95 | 0.8141 | 488 | - |
| 7 | **ITC** | A | 10min | t0.0030_ctxW_OR_N_w0.0065_n0.0090_10min_vm2.0_s | short | T_ATR_SL_0.3 | 11 | 0.530 | 81.8 | 1.99 | 0.8034 | 106 | - |
| 8 | **TATACONSUM** | B | 10min | t0.0030_ctxW_OR_N_w0.0065_n0.0090_10min_off_l | long | T_ATR_SL_0.3 | 10 | 0.912 | 80.0 | 2.54 | 0.7352 | 265 | - |
| 9 | **TITAN** | B | 5min | t0.0050_ctxW_w0.0050_n0.0070_5min_vm1.5_l | long | T_R_TARGET_1.0R | 10 | 0.661 | 80.0 | 2.77 | 0.7288 | 145 | - |
| 10 | **IDFCFIRSTB** | B | 5min | t0.0030_ctxW_OR_N_w0.0065_n0.0050_5min_off_l | long | T_R_TARGET_1.0R | 10 | 0.606 | 80.0 | 2.51 | 0.7204 | 458 | - |
| 11 | **DABUR** | B | 10min | t0.0030_ctxW_OR_N_w0.0050_n0.0090_10min_off_s | short | T_R_TARGET_1.0R | 12 | 0.364 | 83.3 | 2.97 | 0.6847 | 86 | - |
| 12 | **MARUTI** | B | 15min | t0.0040_ctxW_OR_N_w0.0050_n0.0090_15min_off_l | long | T_ATR_SL_0.3 | 13 | 0.810 | 76.9 | 2.58 | 0.6682 | 175 | - |
| 13 | **BHARTIARTL** | A | 5min | t0.0030_ctxW_w0.0080_n0.0070_5min_vm2.0_s | short | T_R_TARGET_1.0R | 10 | 1.450 | 80.0 | 2.84 | 0.6672 | 773 | - |
| 14 | **PIDILITIND** | B | 10min | t0.0050_ctxW_OR_N_w0.0065_n0.0070_10min_off_s | short | T_NO | 10 | 0.362 | 80.0 | 4.89 | 0.6665 | 146 | - |
| 15 | **BANKBARODA** | B | 5min | t0.0050_ctxW_w0.0050_n0.0070_5min_vm1.5_s | short | T_R_TARGET_1.0R | 12 | 0.793 | 75.0 | 3.34 | 0.6414 | 164 | - |

## Promote candidates (1 stocks pass robustness gate)

Gate: best-cell Sharpe >= 0.5 AND n >= 15 AND robust across >= 3 cells (Sharpe>=0.3 + mean>0).

| Stock | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe | RobustCells |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **KOTAKBANK** | 15min | t0.0030_ctxN_w0.0065_n0.0090_15min_off_l | long | T_NO | 15 | 0.316 | 73.3 | 0.5308 | 105 |

## Direction comparison (cells with n>=10)

| Direction | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| long | 99229 | 0.0238 | 47.4 | 0.0268 |
| short | 87165 | 0.0109 | 49.3 | 0.0284 |

## Timeframe comparison (cells with n>=10, mean>0)

| TF | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| 5min | 36854 | 0.2386 | 55.4 | 0.1278 |
| 10min | 32380 | 0.2260 | 55.7 | 0.1245 |
| 15min | 27655 | 0.2034 | 54.4 | 0.1161 |

## today_narrow_threshold sweep (cells with n>=10, mean>0)

| today_narrow | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| 0.30% | 26132 | 0.2400 | 55.6 | 0.1335 |
| 0.40% | 32438 | 0.2175 | 55.1 | 0.1196 |
| 0.50% | 38319 | 0.2194 | 55.0 | 0.1197 |

## yesterday_ctx breakdown (cells with n>=10, mean>0)

| ctx | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| ctxW | 94786 | 0.2268 | 55.2 | 0.1237 |
| ctxN | 2103 | 0.1158 | 55.5 | 0.1093 |
| ctxW_OR_N | 72938 | 0.2235 | 55.2 | 0.1230 |
| ctxW_AND_N | 0 | 0.0000 | 0.0 | 0.0000 |

## Volume mode breakdown (cells with n>=10, mean>0)

| vol_mode | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| off | 70660 | 0.1982 | 54.8 | 0.1134 |
| vm1.5 | 17089 | 0.2950 | 56.6 | 0.1532 |
| vm2.0 | 9140 | 0.2941 | 55.7 | 0.1451 |

## Exit policy comparison (cells with n>=10, mean>0)

| Exit | n_cells | avg_mean_pct | avg_WR | avg_Sharpe |
|---|---:|---:|---:|---:|
| T_NO | 7450 | 0.2363 | 56.1 | 0.1293 |
| T_HARD_SL | 7388 | 0.2302 | 56.2 | 0.1275 |
| T_ATR_SL_0.3 | 7823 | 0.2027 | 50.5 | 0.1061 |
| T_ATR_SL_0.5 | 7566 | 0.2165 | 55.2 | 0.1166 |
| T_ATR_SL_1.0 | 7397 | 0.2290 | 56.2 | 0.1249 |
| T_CHANDELIER_1.0 | 7396 | 0.2198 | 56.0 | 0.1227 |
| T_CHANDELIER_1.5 | 7388 | 0.2303 | 56.2 | 0.1265 |
| T_CHANDELIER_2.0 | 7410 | 0.2353 | 56.1 | 0.1290 |
| T_R_TARGET_1.0R | 7673 | 0.2148 | 56.4 | 0.1267 |
| T_R_TARGET_1.5R | 7585 | 0.2247 | 56.1 | 0.1259 |
| T_R_TARGET_2.0R | 7571 | 0.2313 | 55.9 | 0.1260 |
| T_R_TARGET_3.0R | 7442 | 0.2318 | 56.1 | 0.1272 |
| T_STEP_TRAIL | 6800 | 0.2144 | 50.5 | 0.1155 |

---

## Comparison vs research/30b (volume-breakout)

- Stocks in CCRB leaderboard: **74**
- Stocks in vol-breakout leaderboard: **78**
- In both: **73**  |  Only CCRB: 1  |  Only vol-breakout: 5

### Where CCRB beats vol-breakout (43 stocks)

(These are CPR-compression specialists.)

| Symbol | Coh | CCRB TF/Dir/n/Sharpe | Vol TF/Dir/n/Sharpe | SameDir | SameTF | Δ |
|---|---|---|---|---|---|---:|
| GRASIM | B | 15min/short/11/1.216 | 15min/short/11/0.223 | Y | Y | +0.993 |
| APOLLOHOSP | B | 15min/long/11/0.866 | 10min/short/15/0.114 | N | N | +0.751 |
| NESTLEIND | B | 5min/short/10/1.421 | 10min/long/11/0.701 | N | N | +0.720 |
| NTPC | B | 10min/short/10/0.814 | 15min/short/11/0.181 | Y | N | +0.634 |
| GAIL | B | 10min/short/10/0.637 | 15min/short/29/0.086 | Y | N | +0.551 |
| DLF | B | 5min/short/10/0.863 | 5min/short/14/0.356 | Y | Y | +0.508 |
| ITC | A | 10min/short/11/0.803 | 15min/short/17/0.314 | Y | N | +0.490 |
| TITAN | B | 5min/long/10/0.729 | 10min/long/17/0.264 | Y | N | +0.465 |
| BHARTIARTL | A | 5min/short/10/0.667 | 5min/short/24/0.216 | Y | Y | +0.451 |
| IDFCFIRSTB | B | 5min/long/10/0.720 | 5min/long/17/0.273 | Y | Y | +0.448 |
| ULTRACEMCO | B | 10min/short/10/0.846 | 5min/short/11/0.399 | Y | N | +0.447 |
| HDFCBANK | A | 10min/long/11/0.610 | 15min/short/46/0.168 | N | N | +0.442 |
| BANKBARODA | B | 5min/short/12/0.641 | 15min/short/11/0.200 | Y | N | +0.442 |
| TECHM | B | 5min/long/10/0.565 | 5min/long/16/0.184 | Y | Y | +0.382 |
| DIVISLAB | B | 5min/long/11/0.529 | 10min/long/14/0.173 | Y | N | +0.356 |
| DABUR | B | 10min/short/12/0.685 | 10min/long/11/0.343 | N | Y | +0.342 |
| HEROMOTOCO | B | 15min/short/10/0.458 | 10min/short/11/0.124 | Y | N | +0.334 |
| MARUTI | B | 15min/long/13/0.668 | 10min/long/24/0.414 | Y | N | +0.254 |
| HDFCLIFE | B | 15min/short/11/0.423 | 10min/short/12/0.174 | Y | N | +0.249 |
| INFY | A | 10min/long/11/0.451 | 5min/long/29/0.208 | Y | N | +0.243 |
| PAYTM | B | 5min/long/14/0.267 | 10min/long/16/0.032 | Y | N | +0.235 |
| AMBUJACEM | B | 15min/long/14/0.442 | 5min/short/10/0.232 | N | N | +0.210 |
| KOTAKBANK | A | 15min/long/15/0.531 | 10min/long/22/0.336 | Y | N | +0.195 |
| JSWSTEEL | B | 5min/short/13/0.202 | 15min/long/13/0.014 | N | N | +0.187 |
| ICICIBANK | A | 10min/long/11/0.577 | 15min/short/17/0.393 | N | N | +0.184 |
| TCS | A | 5min/short/13/0.387 | 15min/short/38/0.220 | Y | N | +0.166 |
| HCLTECH | B | 5min/short/14/0.398 | 10min/short/10/0.249 | Y | N | +0.149 |
| COLPAL | B | 15min/long/12/0.365 | 15min/short/12/0.220 | N | Y | +0.146 |
| MUTHOOTFIN | B | 5min/long/12/0.414 | 10min/short/13/0.295 | N | N | +0.119 |
| SUNPHARMA | B | 10min/long/12/0.351 | 10min/long/14/0.234 | Y | Y | +0.117 |

### Where vol-breakout beats CCRB (30 stocks)

(These are volume specialists.)

| Symbol | Coh | CCRB TF/Dir/n/Sharpe | Vol TF/Dir/n/Sharpe | SameDir | SameTF | Δ |
|---|---|---|---|---|---|---:|
| GODREJPROP | B | 10min/short/10/0.608 | 15min/short/10/2.322 | Y | N | -1.715 |
| HAL | B | 5min/short/10/0.499 | 5min/short/10/1.354 | Y | Y | -0.856 |
| DELHIVERY | B | 5min/short/10/0.422 | 15min/short/10/0.841 | Y | N | -0.419 |
| JINDALSTEL | B | 10min/long/10/0.299 | 5min/long/10/0.692 | Y | N | -0.393 |
| VEDL | B | 5min/long/10/0.126 | 15min/long/18/0.509 | Y | N | -0.383 |
| HAVELLS | B | 15min/long/14/0.204 | 10min/long/14/0.567 | Y | N | -0.363 |
| EICHERMOT | B | 5min/long/22/0.327 | 5min/long/10/0.653 | Y | Y | -0.326 |
| BAJAJ-AUTO | B | 5min/long/11/0.205 | 10min/long/13/0.526 | Y | N | -0.322 |
| RELIANCE | A | 5min/long/10/0.560 | 15min/short/14/0.881 | N | N | -0.321 |
| LT | B | 10min/long/10/0.258 | 15min/long/11/0.574 | Y | N | -0.315 |
| PERSISTENT | B | 10min/short/10/0.483 | 5min/long/11/0.796 | N | N | -0.314 |
| COFORGE | B | 5min/long/10/0.276 | 5min/long/13/0.553 | Y | Y | -0.278 |
| SIEMENS | B | 5min/long/17/0.161 | 10min/short/12/0.370 | N | N | -0.209 |
| CHOLAFIN | B | 10min/long/10/0.232 | 10min/long/11/0.434 | Y | Y | -0.203 |
| CUMMINSIND | B | 5min/long/24/0.080 | 15min/long/11/0.268 | Y | N | -0.189 |
| SHREECEM | B | 15min/short/12/0.202 | 10min/short/12/0.380 | Y | N | -0.178 |
| BAJFINANCE | B | 5min/long/10/0.357 | 10min/short/13/0.505 | N | N | -0.148 |
| SBILIFE | B | 10min/long/14/0.068 | 5min/short/11/0.211 | N | N | -0.143 |
| M&M | B | 5min/long/14/0.143 | 5min/short/15/0.214 | N | Y | -0.071 |
| VOLTAS | B | 10min/long/12/0.352 | 10min/long/10/0.415 | Y | Y | -0.063 |
| BEL | B | 5min/short/10/0.247 | 15min/short/11/0.302 | Y | N | -0.055 |
| BRITANNIA | B | 15min/long/14/0.076 | 15min/short/12/0.127 | N | Y | -0.051 |
| POWERGRID | B | 10min/short/11/0.310 | 10min/long/10/0.359 | N | Y | -0.050 |
| IRCTC | B | 5min/short/10/0.168 | 15min/short/10/0.207 | Y | N | -0.039 |
| DRREDDY | B | 10min/long/17/0.104 | 5min/long/10/0.132 | Y | N | -0.027 |
| INDUSINDBK | B | 5min/short/23/0.166 | 10min/long/23/0.190 | N | N | -0.024 |
| MARICO | B | 5min/short/16/0.067 | 10min/short/10/0.087 | Y | N | -0.021 |
| SBIN | A | 15min/long/27/0.281 | 15min/long/21/0.284 | Y | Y | -0.003 |
| FEDERALBNK | B | 10min/short/11/0.295 | 15min/long/14/0.297 | N | N | -0.002 |
| ASIANPAINT | B | 15min/short/10/0.481 | 10min/long/13/0.481 | N | N | -0.000 |

### Stocks robust on BOTH (CCRB Sharpe >= 0.4 AND vol Sharpe >= 0.4): 10

| Symbol | Coh | CCRB Sharpe | Vol Sharpe | SameDir | SameTF |
|---|---|---:|---:|---|---|
| **GODREJPROP** | B | 0.608 | 2.322 | Y | N |
| **NESTLEIND** | B | 1.421 | 0.701 | N | N |
| **HAL** | B | 0.499 | 1.354 | Y | Y |
| **TATACONSUM** | B | 0.735 | 0.708 | Y | Y |
| **RELIANCE** | A | 0.560 | 0.881 | N | N |
| **PIDILITIND** | B | 0.667 | 0.640 | Y | Y |
| **PERSISTENT** | B | 0.483 | 0.796 | N | N |
| **DELHIVERY** | B | 0.422 | 0.841 | Y | N |
| **MARUTI** | B | 0.668 | 0.414 | Y | N |
| **ASIANPAINT** | B | 0.481 | 0.481 | N | N |

### Stocks ONLY found by CCRB (not by vol-breakout): 1

| Symbol | Coh | TF | Dir | n | Sharpe | RobustCells |
|---|---|---|---|---:|---:|---:|
| COALINDIA | B | 10min | long | 10 | 0.0206 | 0 |

### Stocks ONLY in vol-breakout leaderboard (no CCRB leader): 5

BPCL, IOC, MCX, ONGC, WIPRO

---

## Honest Read

- Average best-cell Sharpe across stocks in BOTH leaderboards: CCRB=0.430 vs vol-breakout=0.369
- CCRB beats vol-breakout in 43 / 73 shared stocks (59%).
- Promote-gate passes: CCRB=1; vol-breakout=1.
- Direction agreement (same best dir): 48 / 73 stocks.
- Timeframe agreement (same best TF): 20 / 73 stocks.

Interpretation: CCRB and vol-breakout target different setups. CCRB requires a
daily-bar geometric filter (CPR compression) that gates the day; vol-breakout
triggers off the first candle's volume regardless of CPR. Stocks where they
disagree on direction or timeframe are likely catching different regimes.
Stocks robust on BOTH are the highest-conviction names — the daily geometric
filter and the first-bar volume confirmation could be combined as a higher-bar
entry rule.
