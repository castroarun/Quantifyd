# Short-Options Signal Sweep — Re-ranked (Sharpe-style)

Re-ranking metric: `(mean_net / std_net) × win_rate_fraction`. 
Filters: n_signals ≥ 30, mean_net > 0.

Eligible configurations: **733** of 1476.


## Top 10 across ALL strategies

| path | variant | symbol | tf | exit | params | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| C | rng0.004_norsi | NIFTY50 | 5min | T0 | time_only | 45 | 16.05 | 49.78 | 64.4 | 0.2078 |
| C | rng0.004_rsi40_60 | NIFTY50 | 5min | T0 | time_only | 45 | 16.05 | 49.78 | 64.4 | 0.2078 |
| C | rng0.004_norsi | NIFTY50 | 5min | T1_SL75.0 | sl_pts=75.0 | 45 | 15.65 | 50.48 | 64.4 | 0.1998 |
| C | rng0.004_rsi40_60 | NIFTY50 | 5min | T1_SL75.0 | sl_pts=75.0 | 45 | 15.65 | 50.48 | 64.4 | 0.1998 |
| C | rng0.004_norsi | NIFTY50 | 5min | T1_SL50.0 | sl_pts=50.0 | 45 | 14.96 | 51.01 | 64.4 | 0.1889 |
| C | rng0.004_rsi40_60 | NIFTY50 | 5min | T1_SL50.0 | sl_pts=50.0 | 45 | 14.96 | 51.01 | 64.4 | 0.1889 |
| E | rsi_rsi40_60_15min | KOTAKBANK | 15min | T3_RSI | rsi_cross_50 | 32 | 3.35 | 11.84 | 53.1 | 0.1503 |
| E | rsi_rsi40_60_15min | HINDUNILVR | 15min | T0 | time_only | 56 | 4.42 | 15.49 | 51.8 | 0.1476 |
| E | cpr_rsi_rsi40_60_10min | KOTAKBANK | 10min | T3_RSI | rsi_cross_50 | 31 | 3.75 | 14.21 | 54.8 | 0.1448 |
| E | base_15min | HINDUNILVR | 15min | T0 | time_only | 64 | 5.09 | 17.12 | 48.4 | 0.1441 |

## Top 5 — Path A

| path | variant | symbol | tf | exit | params | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| A | gap0.003_rsi35_65 | NIFTY50 | 5min | T3_RSI | rsi_cross_50 | 98 | 18.95 | 89.57 | 52.0 | 0.1101 |
| A | gap0.005_rsi30_70 | NIFTY50 | 5min | T1_SL50.0 | sl_pts=50.0 | 84 | 20.30 | 96.15 | 50.0 | 0.1055 |
| A | gap0.003_rsi30_70 | NIFTY50 | 5min | T3_RSI | rsi_cross_50 | 64 | 15.37 | 80.36 | 53.1 | 0.1016 |
| A | gap0.003_rsi30_70 | NIFTY50 | 5min | T1_SL50.0 | sl_pts=50.0 | 64 | 19.97 | 95.78 | 48.4 | 0.1010 |
| A | gap0.005_rsi30_70 | NIFTY50 | 5min | T1_SL75.0 | sl_pts=75.0 | 84 | 18.93 | 101.67 | 53.6 | 0.0998 |

## Top 5 — Path B

| path | variant | symbol | tf | exit | params | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| B | rsi30_70_from1100 | NIFTY50 | 5min | T1_SL30.0 | sl_pts=30.0 | 305 | 7.36 | 65.86 | 42.6 | 0.0476 |
| B | rsi40_60_from1100 | NIFTY50 | 5min | T1_SL50.0 | sl_pts=50.0 | 444 | 6.98 | 80.48 | 45.9 | 0.0399 |
| B | rsi30_70_from1100 | NIFTY50 | 5min | T2_TR30.0 | tr_pts=30.0;hard_sl=50.0 | 305 | 6.44 | 69.60 | 42.3 | 0.0392 |
| B | rsi30_70_from1100 | NIFTY50 | 5min | T2_TR20.0 | tr_pts=20.0;hard_sl=50.0 | 305 | 6.83 | 67.08 | 38.4 | 0.0391 |
| B | rsi30_70_from1100 | NIFTY50 | 5min | T0 | time_only | 305 | 6.08 | 86.02 | 53.8 | 0.0380 |

## Top 5 — Path C

| path | variant | symbol | tf | exit | params | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| C | rng0.004_norsi | NIFTY50 | 5min | T0 | time_only | 45 | 16.05 | 49.78 | 64.4 | 0.2078 |
| C | rng0.004_rsi40_60 | NIFTY50 | 5min | T0 | time_only | 45 | 16.05 | 49.78 | 64.4 | 0.2078 |
| C | rng0.004_norsi | NIFTY50 | 5min | T1_SL75.0 | sl_pts=75.0 | 45 | 15.65 | 50.48 | 64.4 | 0.1998 |
| C | rng0.004_rsi40_60 | NIFTY50 | 5min | T1_SL75.0 | sl_pts=75.0 | 45 | 15.65 | 50.48 | 64.4 | 0.1998 |
| C | rng0.004_norsi | NIFTY50 | 5min | T1_SL50.0 | sl_pts=50.0 | 45 | 14.96 | 51.01 | 64.4 | 0.1889 |

## Top 5 — Path D

| path | variant | symbol | tf | exit | params | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| D | priceCPR_rsi30_70 | NIFTY50 | 5min | T1_SL30.0 | sl_pts=30.0 | 238 | 6.97 | 64.98 | 43.7 | 0.0469 |
| D | priceCPR_rsi30_70 | NIFTY50 | 5min | T1_SL50.0 | sl_pts=50.0 | 238 | 5.76 | 70.69 | 47.5 | 0.0387 |
| D | cprDelta_rsi30_70 | NIFTY50 | 5min | T1_SL30.0 | sl_pts=30.0 | 145 | 5.44 | 61.79 | 42.8 | 0.0377 |
| D | priceCPR_rsi30_70 | NIFTY50 | 5min | T2_TR30.0 | tr_pts=30.0;hard_sl=50.0 | 238 | 6.03 | 67.23 | 41.6 | 0.0373 |
| D | cprDelta_rsi30_70 | NIFTY50 | 5min | T3_RSI | rsi_cross_50 | 145 | 5.21 | 63.40 | 43.4 | 0.0357 |

## Top 5 — Path E

| path | variant | symbol | tf | exit | params | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| E | rsi_rsi40_60_15min | KOTAKBANK | 15min | T3_RSI | rsi_cross_50 | 32 | 3.35 | 11.84 | 53.1 | 0.1503 |
| E | rsi_rsi40_60_15min | HINDUNILVR | 15min | T0 | time_only | 56 | 4.42 | 15.49 | 51.8 | 0.1476 |
| E | cpr_rsi_rsi40_60_10min | KOTAKBANK | 10min | T3_RSI | rsi_cross_50 | 31 | 3.75 | 14.21 | 54.8 | 0.1448 |
| E | base_15min | HINDUNILVR | 15min | T0 | time_only | 64 | 5.09 | 17.12 | 48.4 | 0.1441 |
| E | base_15min | KOTAKBANK | 15min | T3_RSI | rsi_cross_50 | 39 | 3.26 | 11.80 | 51.3 | 0.1419 |

## Top 5 — Path F

| path | variant | symbol | tf | exit | params | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| F | priceCPR | SBIN | 5min | T0 | time_only | 201 | 1.15 | 7.86 | 51.7 | 0.0754 |
| F | priceCPR | SBIN | 5min | T3_RSI | rsi_cross_50 | 201 | 0.81 | 6.34 | 40.3 | 0.0517 |
| F | priceCPR | SBIN | 5min | T4_LVL | level_for_T4_cross | 201 | 0.89 | 7.15 | 34.8 | 0.0433 |
| F | priceCPR | SBIN | 5min | T1_SL1.57 | sl_pts=1.57 | 201 | 0.86 | 7.22 | 35.8 | 0.0429 |
| F | priceCPR | RELIANCE | 5min | T0 | time_only | 206 | 0.56 | 9.54 | 49.5 | 0.0289 |

## Old composite-rank top 5 — for comparison

| path | variant | symbol | tf | exit | params | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| E | cpr_10min | ITC | 10min | T2_TR0.11 | tr_pts=0.11;hard_sl=0.45 | 42 | 0.37 | 1.98 | 23.8 | 0.0444 |
| E | cpr_10min | ITC | 10min | T2_TR0.22 | tr_pts=0.22;hard_sl=0.45 | 42 | 0.37 | 1.98 | 23.8 | 0.0444 |
| E | cpr_rsi_rsi40_60_10min | ITC | 10min | T2_TR0.11 | tr_pts=0.11;hard_sl=0.45 | 39 | 0.32 | 1.90 | 25.6 | 0.0437 |
| E | cpr_rsi_rsi40_60_10min | ITC | 10min | T2_TR0.22 | tr_pts=0.22;hard_sl=0.45 | 39 | 0.30 | 1.92 | 25.6 | 0.0397 |
| E | cpr_rsi_rsi40_60_10min | ITC | 10min | T2_TR0.45 | tr_pts=0.45;hard_sl=0.45 | 39 | 0.28 | 1.93 | 25.6 | 0.0375 |
