# Short-Options Signal Sweep — RESULTS

Total exit rows: **136,143**  •  Total signals: **15,127**
Total ranked configurations: **1,476**

Composite score = `mean(net_pts) - 0.5 × std(net_pts)`. 
`pct_sl_first` = percent of signals where the policy's hard-SL was hit first (lower = fewer stop-outs).

## Top 5 configurations across ALL strategies

| path | variant | symbol | timeframe | exit_policy | exit_params | n_signals | mean_net | std_net | win_rate | pct_sl_first | composite_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E | cpr_10min | ITC | 10min | T2_TR0.11 | tr_pts=0.11;hard_sl=0.45 | 42 | 0.369 | 1.978 | 23.810 | 45.238 | -0.620 |
| E | cpr_10min | ITC | 10min | T2_TR0.22 | tr_pts=0.22;hard_sl=0.45 | 42 | 0.369 | 1.978 | 23.810 | 45.238 | -0.620 |
| E | cpr_rsi_rsi40_60_10min | ITC | 10min | T2_TR0.11 | tr_pts=0.11;hard_sl=0.45 | 39 | 0.324 | 1.902 | 25.641 | 43.590 | -0.627 |
| E | cpr_rsi_rsi40_60_10min | ITC | 10min | T2_TR0.22 | tr_pts=0.22;hard_sl=0.45 | 39 | 0.297 | 1.920 | 25.641 | 46.154 | -0.663 |
| E | cpr_rsi_rsi40_60_10min | ITC | 10min | T2_TR0.45 | tr_pts=0.45;hard_sl=0.45 | 39 | 0.282 | 1.926 | 25.641 | 48.718 | -0.681 |

## Top 5 — Path **A**

| path | variant | symbol | timeframe | exit_policy | exit_params | n_signals | mean_net | std_net | win_rate | pct_sl_first | composite_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | gap0.003_rsi30_70 | NIFTY50 | 5min | T2_TR30.0 | tr_pts=30.0;hard_sl=50.0 | 64 | 19.296 | 81.188 | 35.938 | 23.438 | -21.298 |
| A | gap0.003_rsi30_70 | NIFTY50 | 5min | T2_TR10.0 | tr_pts=10.0;hard_sl=50.0 | 64 | 12.468 | 67.848 | 23.438 | 15.625 | -21.456 |
| A | gap0.003_rsi30_70 | NIFTY50 | 5min | T2_TR20.0 | tr_pts=20.0;hard_sl=50.0 | 64 | 16.381 | 75.982 | 32.812 | 21.875 | -21.610 |
| A | gap0.005_rsi30_70 | NIFTY50 | 5min | T2_TR10.0 | tr_pts=10.0;hard_sl=50.0 | 84 | 11.110 | 71.170 | 23.810 | 17.857 | -24.475 |
| A | gap0.005_rsi30_70 | NIFTY50 | 5min | T2_TR20.0 | tr_pts=20.0;hard_sl=50.0 | 84 | 14.111 | 77.253 | 32.143 | 22.619 | -24.516 |

## Top 5 — Path **B**

| path | variant | symbol | timeframe | exit_policy | exit_params | n_signals | mean_net | std_net | win_rate | pct_sl_first | composite_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | rsi30_70_from1100 | NIFTY50 | 5min | T1_SL30.0 | sl_pts=30.0 | 305 | 7.359 | 65.861 | 42.623 | 46.885 | -25.571 |
| B | rsi30_70_from1100 | NIFTY50 | 5min | T2_TR10.0 | tr_pts=10.0;hard_sl=50.0 | 305 | 5.061 | 62.320 | 32.459 | 20.000 | -26.099 |
| B | rsi30_70_from1100 | NIFTY50 | 5min | T2_TR20.0 | tr_pts=20.0;hard_sl=50.0 | 305 | 6.833 | 67.077 | 38.361 | 24.590 | -26.705 |
| B | rsi30_70_from1100 | NIFTY50 | 5min | T2_TR30.0 | tr_pts=30.0;hard_sl=50.0 | 305 | 6.444 | 69.600 | 42.295 | 27.869 | -28.356 |
| B | rsi30_70_from1100 | NIFTY50 | 5min | T4_LVL | level_for_T4_cross | 305 | 4.390 | 66.083 | 34.426 | 60.656 | -28.652 |

## Top 5 — Path **C**

| path | variant | symbol | timeframe | exit_policy | exit_params | n_signals | mean_net | std_net | win_rate | pct_sl_first | composite_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C | rng0.004_norsi | NIFTY50 | 5min | T0 | time_only | 45 | 16.050 | 49.781 | 64.444 | 0.000 | -8.840 |
| C | rng0.004_rsi40_60 | NIFTY50 | 5min | T0 | time_only | 45 | 16.050 | 49.781 | 64.444 | 0.000 | -8.840 |
| C | rng0.004_norsi | NIFTY50 | 5min | T2_TR10.0 | tr_pts=10.0;hard_sl=50.0 | 45 | 6.831 | 32.457 | 37.778 | 4.444 | -9.397 |
| C | rng0.004_rsi40_60 | NIFTY50 | 5min | T2_TR10.0 | tr_pts=10.0;hard_sl=50.0 | 45 | 6.831 | 32.457 | 37.778 | 4.444 | -9.397 |
| C | rng0.004_norsi | NIFTY50 | 5min | T1_SL75.0 | sl_pts=75.0 | 45 | 15.650 | 50.482 | 64.444 | 2.222 | -9.591 |

## Top 5 — Path **D**

| path | variant | symbol | timeframe | exit_policy | exit_params | n_signals | mean_net | std_net | win_rate | pct_sl_first | composite_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D | cprDelta_rsi30_70 | NIFTY50 | 5min | T1_SL30.0 | sl_pts=30.0 | 145 | 5.444 | 61.794 | 42.759 | 42.069 | -25.453 |
| D | priceCPR_rsi30_70 | NIFTY50 | 5min | T1_SL30.0 | sl_pts=30.0 | 238 | 6.973 | 64.985 | 43.697 | 44.958 | -25.519 |
| D | priceCPR_rsi40_60 | NIFTY50 | 5min | T4_LVL | level_for_T4_cross | 413 | 0.297 | 53.387 | 25.666 | 70.944 | -26.397 |
| D | cprDelta_rsi30_70 | NIFTY50 | 5min | T3_RSI | rsi_cross_50 | 145 | 5.215 | 63.397 | 43.448 | 44.138 | -26.484 |
| D | cprDelta_rsi30_70 | NIFTY50 | 5min | T2_TR10.0 | tr_pts=10.0;hard_sl=50.0 | 145 | 1.000 | 55.068 | 30.345 | 18.621 | -26.534 |

## Top 5 — Path **E**

| path | variant | symbol | timeframe | exit_policy | exit_params | n_signals | mean_net | std_net | win_rate | pct_sl_first | composite_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E | cpr_10min | ITC | 10min | T2_TR0.11 | tr_pts=0.11;hard_sl=0.45 | 42 | 0.369 | 1.978 | 23.810 | 45.238 | -0.620 |
| E | cpr_10min | ITC | 10min | T2_TR0.22 | tr_pts=0.22;hard_sl=0.45 | 42 | 0.369 | 1.978 | 23.810 | 45.238 | -0.620 |
| E | cpr_rsi_rsi40_60_10min | ITC | 10min | T2_TR0.11 | tr_pts=0.11;hard_sl=0.45 | 39 | 0.324 | 1.902 | 25.641 | 43.590 | -0.627 |
| E | cpr_rsi_rsi40_60_10min | ITC | 10min | T2_TR0.22 | tr_pts=0.22;hard_sl=0.45 | 39 | 0.297 | 1.920 | 25.641 | 46.154 | -0.663 |
| E | cpr_rsi_rsi40_60_10min | ITC | 10min | T2_TR0.45 | tr_pts=0.45;hard_sl=0.45 | 39 | 0.282 | 1.926 | 25.641 | 48.718 | -0.681 |

## Top 5 — Path **F**

| path | variant | symbol | timeframe | exit_policy | exit_params | n_signals | mean_net | std_net | win_rate | pct_sl_first | composite_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F | priceCPR | ITC | 5min | T2_TR0.11 | tr_pts=0.11;hard_sl=0.45 | 216 | -0.031 | 1.375 | 14.352 | 42.593 | -0.718 |
| F | priceCPR | ITC | 5min | T2_TR0.22 | tr_pts=0.22;hard_sl=0.45 | 216 | -0.054 | 1.387 | 15.741 | 48.611 | -0.747 |
| F | priceCPR | ITC | 5min | T1_SL0.22 | sl_pts=0.22 | 216 | -0.036 | 1.433 | 18.056 | 81.019 | -0.752 |
| F | priceCPR | ITC | 5min | T3_RSI | rsi_cross_50 | 216 | -0.008 | 1.493 | 35.185 | 59.722 | -0.755 |
| F | priceCPR | ITC | 5min | T2_TR0.45 | tr_pts=0.45;hard_sl=0.45 | 216 | -0.054 | 1.443 | 17.593 | 56.481 | -0.776 |

## Best mean-net per (path × exit_policy)

| path | exit_policy | composite_score |
|---|---|---|
| A | T0 | -42.859 |
| A | T1_SL30.0 | -29.503 |
| A | T1_SL50.0 | -27.778 |
| A | T1_SL75.0 | -31.899 |
| A | T2_TR10.0 | -21.456 |
| A | T2_TR20.0 | -21.610 |
| A | T2_TR30.0 | -21.298 |
| A | T3_RSI | -24.812 |
| A | T4_LVL | -29.345 |
| B | T0 | -36.930 |
| B | T1_SL30.0 | -25.571 |
| B | T1_SL50.0 | -31.927 |
| B | T1_SL75.0 | -34.621 |
| B | T2_TR10.0 | -26.099 |
| B | T2_TR20.0 | -26.705 |
| B | T2_TR30.0 | -28.356 |
| B | T3_RSI | -29.421 |
| B | T4_LVL | -28.652 |
| C | T0 | -8.840 |
| C | T1_SL30.0 | -14.070 |
| C | T1_SL50.0 | -10.550 |
| C | T1_SL75.0 | -9.591 |
| C | T2_TR10.0 | -9.397 |
| C | T2_TR20.0 | -11.982 |
| C | T2_TR30.0 | -11.812 |
| C | T3_RSI | -14.665 |
| C | T4_LVL | -15.778 |
| D | T0 | -36.570 |
| D | T1_SL30.0 | -25.453 |
| D | T1_SL50.0 | -29.586 |
| D | T1_SL75.0 | -33.080 |
| D | T2_TR10.0 | -26.534 |
| D | T2_TR20.0 | -26.942 |
| D | T2_TR30.0 | -27.088 |
| D | T3_RSI | -26.484 |
| D | T4_LVL | -26.397 |
| E | T0 | -1.677 |
| E | T1_SL0.22 | -0.739 |
| E | T1_SL0.45 | -0.794 |
| E | T1_SL0.52 | -1.840 |
| E | T1_SL0.67 | -0.990 |
| E | T1_SL0.7 | -2.217 |
| E | T1_SL0.8 | -2.367 |
| E | T1_SL0.9 | -1.257 |
| E | T1_SL1.0 | -3.582 |
| E | T1_SL1.02 | -2.935 |
| E | T1_SL1.05 | -2.036 |
| E | T1_SL1.1 | -4.013 |
| E | T1_SL1.35 | -1.428 |
| E | T1_SL1.4 | -2.738 |
| E | T1_SL1.45 | -5.261 |
| E | T1_SL1.57 | -2.571 |
| E | T1_SL1.6 | -2.893 |
| E | T1_SL2.0 | -4.136 |
| E | T1_SL2.05 | -4.019 |
| E | T1_SL2.1 | -2.961 |
| E | T1_SL2.2 | -4.404 |
| E | T1_SL2.4 | -2.865 |
| E | T1_SL2.9 | -6.064 |
| E | T1_SL3.0 | -4.679 |
| E | T1_SL3.07 | -4.330 |
| E | T1_SL3.3 | -4.661 |
| E | T1_SL4.1 | -7.073 |
| E | T1_SL4.35 | -5.970 |
| E | T1_SL6.15 | -7.194 |
| E | T2_TR0.11 | -0.620 |
| E | T2_TR0.22 | -0.620 |
| E | T2_TR0.26 | -1.633 |
| E | T2_TR0.35 | -2.371 |
| E | T2_TR0.4 | -2.509 |
| E | T2_TR0.45 | -0.681 |
| E | T2_TR0.5 | -3.476 |
| E | T2_TR0.51 | -3.489 |
| E | T2_TR0.52 | -1.687 |
| E | T2_TR0.55 | -3.982 |
| E | T2_TR0.7 | -2.401 |
| E | T2_TR0.73 | -5.737 |
| E | T2_TR0.8 | -2.527 |
| E | T2_TR0.9 | -1.072 |
| E | T2_TR1.0 | -3.550 |
| E | T2_TR1.02 | -3.536 |
| E | T2_TR1.05 | -1.831 |
| E | T2_TR1.1 | -3.908 |
| E | T2_TR1.4 | -2.464 |
| E | T2_TR1.45 | -5.804 |
| E | T2_TR1.6 | -2.540 |
| E | T2_TR2.0 | -3.643 |
| E | T2_TR2.05 | -3.526 |
| E | T2_TR2.2 | -3.938 |
| E | T2_TR2.9 | -5.966 |
| E | T2_TR4.1 | -6.004 |
| E | T3_RSI | -0.790 |
| E | T4_LVL | -0.929 |
| F | T0 | -1.446 |
| F | T1_SL0.22 | -0.752 |
| F | T1_SL0.45 | -0.816 |
| F | T1_SL0.52 | -2.599 |
| F | T1_SL0.67 | -0.867 |
| F | T1_SL0.7 | -2.027 |
| F | T1_SL0.8 | -2.217 |
| F | T1_SL0.9 | -1.547 |
| F | T1_SL1.0 | -2.930 |
| F | T1_SL1.02 | -3.070 |
| F | T1_SL1.05 | -2.879 |
| F | T1_SL1.1 | -3.557 |
| F | T1_SL1.35 | -1.768 |
| F | T1_SL1.4 | -2.384 |
| F | T1_SL1.45 | -5.071 |
| F | T1_SL1.57 | -2.745 |
| F | T1_SL1.6 | -2.594 |
| F | T1_SL2.0 | -3.531 |
| F | T1_SL2.05 | -3.465 |
| F | T1_SL2.1 | -2.675 |
| F | T1_SL2.2 | -3.779 |
| F | T1_SL2.4 | -2.882 |
| F | T1_SL2.9 | -5.910 |
| F | T1_SL3.0 | -3.902 |
| F | T1_SL3.07 | -4.012 |
| F | T1_SL3.3 | -4.486 |
| F | T1_SL4.1 | -7.941 |
| F | T1_SL4.35 | -6.599 |
| F | T1_SL6.15 | -8.667 |
| F | T2_TR0.11 | -0.718 |
| F | T2_TR0.22 | -0.747 |
| F | T2_TR0.26 | -2.668 |
| F | T2_TR0.35 | -1.891 |
| F | T2_TR0.4 | -2.166 |
| F | T2_TR0.45 | -0.776 |
| F | T2_TR0.5 | -2.956 |
| F | T2_TR0.51 | -3.176 |
| F | T2_TR0.52 | -2.665 |
| F | T2_TR0.55 | -3.454 |
| F | T2_TR0.7 | -1.961 |
| F | T2_TR0.73 | -5.448 |
| F | T2_TR0.8 | -2.237 |
| F | T2_TR0.9 | -1.347 |
| F | T2_TR1.0 | -3.116 |
| F | T2_TR1.02 | -3.243 |
| F | T2_TR1.05 | -2.701 |
| F | T2_TR1.1 | -3.577 |
| F | T2_TR1.4 | -2.121 |
| F | T2_TR1.45 | -5.483 |
| F | T2_TR1.6 | -2.354 |
| F | T2_TR2.0 | -3.274 |
| F | T2_TR2.05 | -3.246 |
| F | T2_TR2.2 | -3.595 |
| F | T2_TR2.9 | -5.661 |
| F | T2_TR4.1 | -7.178 |
| F | T3_RSI | -0.755 |
| F | T4_LVL | -0.883 |

## Strategy quality at T0 (time-only) — best variant per path

| path | variant | symbol | timeframe | exit_policy | exit_params | n_signals | mean_net | std_net | win_rate | pct_sl_first | composite_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F | priceCPR | ITC | 5min | T0 | time_only | 216 | -0.023 | 2.846 | 50.463 | 0.000 | -1.446 |
| E | cpr_rsi_rsi40_60_10min | HDFCBANK | 10min | T0 | time_only | 54 | 0.122 | 3.598 | 51.852 | 0.000 | -1.677 |
| C | rng0.004_norsi | NIFTY50 | 5min | T0 | time_only | 45 | 16.050 | 49.781 | 64.444 | 0.000 | -8.840 |
| D | priceCPR_rsi30_70 | NIFTY50 | 5min | T0 | time_only | 238 | 5.133 | 83.405 | 52.101 | 0.000 | -36.570 |
| B | rsi30_70_from1100 | NIFTY50 | 5min | T0 | time_only | 305 | 6.081 | 86.023 | 53.770 | 0.000 | -36.930 |
| A | gapoff_rsi30_70 | NIFTY50 | 5min | T0 | time_only | 99 | 16.890 | 119.497 | 57.576 | 0.000 | -42.859 |

## Honest read

- **Best overall config:** Path E / `cpr_10min` / ITC 10min with `T2_TR0.11` (tr_pts=0.11;hard_sl=0.45) — n=42, mean=0.37, std=1.98, win-rate=23.8%, composite=-0.62.
- **Raw signal quality (T0 time-only):**
  - Path F best: `priceCPR` / ITC 5min → mean=-0.02, win-rate=50.5%, n=216
  - Path E best: `cpr_rsi_rsi40_60_10min` / HDFCBANK 10min → mean=+0.12, win-rate=51.9%, n=54
  - Path C best: `rng0.004_norsi` / NIFTY50 5min → mean=+16.05, win-rate=64.4%, n=45
  - Path D best: `priceCPR_rsi30_70` / NIFTY50 5min → mean=+5.13, win-rate=52.1%, n=238
  - Path B best: `rsi30_70_from1100` / NIFTY50 5min → mean=+6.08, win-rate=53.8%, n=305
  - Path A best: `gapoff_rsi30_70` / NIFTY50 5min → mean=+16.89, win-rate=57.6%, n=99
- **Effect of exit policies:**
  - Path A: best policy `T2_TR30.0` (tr_pts=30.0;hard_sl=50.0) lifts composite by 21.56 vs T0 (mean: 19.30 vs 16.89).
  - Path B: best policy `T1_SL30.0` (sl_pts=30.0) lifts composite by 11.36 vs T0 (mean: 7.36 vs 6.08).
  - Path C: best policy `T0` (time_only) lifts composite by 0.00 vs T0 (mean: 16.05 vs 16.05).
  - Path D: best policy `T1_SL30.0` (sl_pts=30.0) lifts composite by 11.12 vs T0 (mean: 5.44 vs 5.13).
  - Path E: best policy `T2_TR0.11` (tr_pts=0.11;hard_sl=0.45) lifts composite by 1.06 vs T0 (mean: 0.37 vs 0.12).
  - Path F: best policy `T2_TR0.11` (tr_pts=0.11;hard_sl=0.45) lifts composite by 0.73 vs T0 (mean: -0.03 vs -0.02).
- **Short-options viability proxy:**
  For shorting OTM options, we want LOW adverse excursion (signals that don't whip against direction) and HIGH probability the underlying stays with us through EOD. Configs with high `win_rate` and small absolute `mean_net` (drift) are *more* viable than configs with high mean_net and high std (large directional moves cut both ways for an option seller, but only one side benefits).
