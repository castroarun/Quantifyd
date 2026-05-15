# VOLSURGE + PDR/PWR-Break, Narrow Weekly CPR — Underlying Sweep

## Setup

- Universe: 79 F&O stocks (FNO_LOT_SIZES minus TATAMOTORS, ZOMATO)
- Period: 2018-01-01 -> 2026-05-15 (per-stock, clipped to available 5-min)
- Timeframes: 5min, 10min, 15min, 30min, 60min (10/15/30/60 resampled from 5-min)
- Grid: trend['sma50','sma200','hh20'] x theta_cpr[0.25,0.5,0.75,1.0] x k[1.5,2.0,3.0] x clean['loose','strict'] x clearroom[False,True] x carry['sameday','carry']
- Direction = daily-trend selector (up->long, down->short, flat->skip)
- 13 exit policies scored in parallel per signal
- Total signal rows: **121,278** (long 81,524 / short 39,754)

> Per ex#9: the confluence is a NECESSARY-but-NOT-SUFFICIENT probabilistic edge — judged on expectancy/Sharpe over a population, never on individual outcomes.

## Top 15 configs (Sharpe, n>=15, mean>0)

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Exp% | Sharpe |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| TCS | 60min | 60min_sma200_cpr0.75_vm1.5_loose_croff_carry | long | T_NO | 18 | 2.559 | 94.4 | 1.37 | 2.559 | 1.0852 |
| TCS | 60min | 60min_sma200_cpr1.00_vm1.5_loose_croff_carry | long | T_NO | 22 | 2.342 | 86.4 | 2.95 | 2.342 | 0.9139 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_CHANDELIER_1.0 | 15 | 1.021 | 86.7 | 1.47 | 1.021 | 0.8774 |
| TCS | 60min | 60min_sma50_cpr0.75_vm1.5_loose_croff_carry | long | T_NO | 16 | 2.457 | 87.5 | 1.64 | 2.457 | 0.8415 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_carry | long | T_R_TARGET_1.0R | 15 | 1.023 | 86.7 | 1.01 | 1.023 | 0.8215 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_CHANDELIER_1.5 | 15 | 0.907 | 86.7 | 1.17 | 0.907 | 0.8189 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_ATR_SL_1.0 | 15 | 0.985 | 86.7 | 1.35 | 0.986 | 0.7931 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_R_TARGET_1.0R | 15 | 0.879 | 86.7 | 1.19 | 0.879 | 0.7857 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_HARD_SL | 15 | 0.982 | 86.7 | 1.31 | 0.982 | 0.7839 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_R_TARGET_3.0R | 15 | 0.982 | 86.7 | 1.31 | 0.982 | 0.7839 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_NO | 15 | 0.975 | 86.7 | 1.25 | 0.975 | 0.7652 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_CHANDELIER_2.0 | 15 | 0.975 | 86.7 | 1.25 | 0.975 | 0.7652 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_R_TARGET_1.5R | 15 | 1.105 | 86.7 | 1.46 | 1.105 | 0.7633 |
| TCS | 60min | 60min_sma200_cpr0.75_vm1.5_loose_croff_carry | long | T_ATR_SL_1.0 | 18 | 2.309 | 83.3 | 1.72 | 2.309 | 0.7571 |
| ICICIBANK | 10min | 10min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_R_TARGET_2.0R | 15 | 1.347 | 86.7 | 1.75 | 1.347 | 0.7375 |

## Per-stock leaders (best Sharpe, n>=10)

| Rank | Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Exp% | Sharpe | HiQ | MidQ |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ICICIBANK | 5min | 5min_sma50_cpr1.00_vm1.5_loose_croff_sameday | long | T_R_TARGET_1.0R | 12 | 1.011 | 91.7 | 1.011 | 1.8279 | 264 | 328 |
| 2 | TCS | 60min | 60min_sma200_cpr0.75_vm1.5_loose_croff_carry | long | T_NO | 18 | 2.559 | 94.4 | 2.559 | 1.0852 | 37 | 54 |
| 3 | HDFCBANK | 30min | 30min_sma200_cpr1.00_vm2.0_strict_croff_carry | long | T_R_TARGET_1.5R | 10 | 1.043 | 80.0 | 1.044 | 0.7458 | 12 | 20 |
| 4 | HINDUNILVR | 5min | 5min_sma50_cpr1.00_vm1.5_strict_croff_sameday | long | T_CHANDELIER_1.5 | 10 | 0.947 | 80.0 | 0.947 | 0.7378 | 46 | 96 |
| 5 | VEDL | 30min | 30min_sma200_cpr0.75_vm1.5_loose_croff_carry | long | T_CHANDELIER_1.5 | 11 | 6.096 | 81.8 | 6.096 | 0.6967 | 4 | 8 |
| 6 | BHARTIARTL | 10min | 10min_sma200_cpr1.00_vm1.5_loose_croff_carry | long | T_NO | 10 | 2.716 | 80.0 | 2.716 | 0.6626 | 12 | 30 |
| 7 | SBIN | 15min | 15min_sma50_cpr0.75_vm1.5_strict_croff_sameday | long | T_NO | 10 | 1.151 | 70.0 | 1.151 | 0.6249 | 33 | 84 |
| 8 | RELIANCE | 15min | 15min_sma200_cpr0.50_vm1.5_loose_croff_sameday | long | T_R_TARGET_1.0R | 10 | 0.478 | 80.0 | 0.478 | 0.5905 | 31 | 105 |
| 9 | ITC | 30min | 30min_sma50_cpr1.00_vm1.5_loose_croff_carry | short | T_NO | 10 | 2.860 | 80.0 | 2.860 | 0.5768 | 1 | 7 |
| 10 | BEL | 60min | 60min_sma200_cpr0.75_vm1.5_loose_croff_carry | long | T_NO | 11 | 2.265 | 81.8 | 2.265 | 0.5528 | 3 | 6 |
| 11 | HEROMOTOCO | 15min | 15min_sma200_cpr1.00_vm2.0_loose_croff_carry | long | T_CHANDELIER_2.0 | 11 | 2.205 | 72.7 | 2.205 | 0.5312 | 28 | 45 |
| 12 | INFY | 10min | 10min_sma200_cpr1.00_vm1.5_strict_croff_carry | long | T_R_TARGET_1.5R | 11 | 0.992 | 72.7 | 0.992 | 0.4724 | 1 | 10 |
| 13 | TATASTEEL | 30min | 30min_sma200_cpr1.00_vm1.5_loose_croff_carry | long | T_R_TARGET_2.0R | 10 | 1.402 | 60.0 | 1.402 | 0.3656 | 0 | 4 |
| 14 | JSWSTEEL | 60min | 60min_sma200_cpr0.75_vm1.5_loose_croff_sameday | long | T_NO | 11 | 0.473 | 72.7 | 0.473 | 0.3431 | 0 | 6 |
| 15 | KOTAKBANK | 10min | 10min_sma50_cpr0.75_vm1.5_loose_croff_sameday | long | T_R_TARGET_2.0R | 10 | 0.928 | 60.0 | 0.928 | 0.2736 | 0 | 0 |
| 16 | GRASIM | 60min | 60min_sma50_cpr0.75_vm1.5_loose_croff_carry | long | T_CHANDELIER_2.0 | 11 | 0.886 | 54.5 | 0.886 | 0.1484 | 0 | 0 |
| 17 | MARUTI | 15min | 15min_sma200_cpr1.00_vm1.5_loose_croff_sameday | long | T_STEP_TRAIL | 10 | 0.346 | 50.0 | 0.346 | 0.1348 | 0 | 0 |
| 18 | TATAPOWER | 30min | 30min_sma50_cpr1.00_vm1.5_loose_croff_carry | long | T_NO | 10 | 0.304 | 40.0 | 0.304 | 0.0453 | 0 | 0 |
| 19 | MUTHOOTFIN | 60min | 60min_sma200_cpr0.75_vm1.5_loose_croff_sameday | long | T_ATR_SL_0.3 | 10 | -0.754 | 0.0 | -0.754 | -0.0000 | 0 | 0 |

## Promote candidates (Sharpe>=0.5, n>=15, MidQ>=3)

Robustness gate (research/34 style): best cell strong AND the edge survives across >=3 variants.

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe | MidQ | LongSh | ShortSh |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TCS | 60min | 60min_sma200_cpr0.75_vm1.5_loose_croff_carry | long | T_NO | 18 | 2.559 | 94.4 | 1.0852 | 54 | 1.085 | 0.828 |

## Direction asymmetry (top 20 leaders)

| Symbol | Long Sharpe (n) | Short Sharpe (n) | Bias |
|---|---:|---:|---|
| ICICIBANK | 1.828 (n=12) | 0.215 (n=10) | LONG |
| TCS | 1.085 (n=18) | 0.828 (n=10) | both |
| HDFCBANK | 0.746 (n=10) | 0.362 (n=10) | LONG |
| HINDUNILVR | 0.738 (n=10) | 0.381 (n=10) | LONG |
| VEDL | 0.697 (n=11) | 0.000 (n=0) | LONG |
| BHARTIARTL | 0.663 (n=10) | 0.000 (n=0) | LONG |
| SBIN | 0.625 (n=10) | 0.000 (n=0) | LONG |
| RELIANCE | 0.590 (n=10) | 0.419 (n=12) | both |
| ITC | 0.261 (n=10) | 0.577 (n=10) | SHORT |
| BEL | 0.553 (n=11) | 0.000 (n=0) | LONG |
| HEROMOTOCO | 0.531 (n=11) | 0.000 (n=0) | LONG |
| INFY | 0.472 (n=11) | 0.247 (n=10) | LONG |
| TATASTEEL | 0.366 (n=10) | 0.000 (n=0) | LONG |
| JSWSTEEL | 0.343 (n=11) | 0.000 (n=0) | LONG |
| KOTAKBANK | 0.274 (n=10) | -0.000 (n=12) | LONG |
| GRASIM | 0.148 (n=11) | 0.000 (n=0) | weak |
| MARUTI | 0.135 (n=10) | 0.000 (n=0) | weak |
| TATAPOWER | 0.045 (n=10) | 0.000 (n=0) | weak |
| MUTHOOTFIN | -0.000 (n=10) | 0.000 (n=0) | weak |

## Timeframe sweet-spot (cells n>=15, mean>0)

| TF | n_cells | avg_mean% | avg_WR% | avg_Sharpe |
|---|---:|---:|---:|---:|
| 5min | 144 | 0.3570 | 47.9 | 0.1217 |
| 10min | 229 | 0.4119 | 47.2 | 0.1221 |
| 15min | 381 | 0.5020 | 50.7 | 0.1579 |
| 30min | 495 | 0.4267 | 52.1 | 0.1217 |
| 60min | 439 | 0.4501 | 52.3 | 0.1222 |

## Exit-policy comparison (cells n>=15, mean>0)

| ExitPolicy | n_cells | avg_mean% | avg_WR% | avg_Sharpe |
|---|---:|---:|---:|---:|
| T_CHANDELIER_1.5 | 151 | 0.6023 | 55.8 | 0.1716 |
| T_CHANDELIER_2.0 | 153 | 0.5837 | 57.7 | 0.1704 |
| T_R_TARGET_1.0R | 129 | 0.2955 | 59.8 | 0.1605 |
| T_R_TARGET_2.0R | 133 | 0.4215 | 52.2 | 0.1392 |
| T_NO | 187 | 0.5064 | 54.7 | 0.1342 |
| T_R_TARGET_1.5R | 141 | 0.3394 | 54.2 | 0.1331 |
| T_R_TARGET_3.0R | 138 | 0.4743 | 49.5 | 0.1325 |
| T_CHANDELIER_1.0 | 136 | 0.3636 | 51.3 | 0.1218 |
| T_ATR_SL_1.0 | 156 | 0.4788 | 50.9 | 0.1187 |
| T_HARD_SL | 138 | 0.5203 | 47.5 | 0.1177 |
| T_ATR_SL_0.5 | 71 | 0.3597 | 39.9 | 0.0829 |
| T_STEP_TRAIL | 113 | 0.2919 | 33.7 | 0.0667 |
| T_ATR_SL_0.3 | 42 | 0.2117 | 25.3 | 0.0343 |

---

## Honest interpretation (analysis, not auto-generated)

**No robust universe-wide edge — same verdict as research/34 VOLBO.**
Aggregate avg Sharpe ~0.12-0.16, WR ~48-52% — coin-flip on average across
70,148 tested cells. **Only TCS (60min, sma200, cpr<=0.75, vm1.5, loose,
carry, long, T_NO)** clears the robustness gate; 1-in-79 across 70k cells is
~ multiple-comparison noise -> treat as a candidate, not a proven edge.
Strong **LONG bias** (short side ~ no edge, contradicting the ex#4 symmetry
hope). Loosest filters win (vm1.5 + loose candle + carry); stricter
volume/candle did not help. 15min marginally best TF; Chandelier 1.5/2.0
best exit. **Recommendation:** do NOT deploy as an automated strategy —
walk-forward / OOS-validate the TCS-60min cell + the
"long+sma200+vm1.5+loose+carry+Chandelier" archetype, or keep VOLSURGE as
the discretionary /app/scanner screen only.
