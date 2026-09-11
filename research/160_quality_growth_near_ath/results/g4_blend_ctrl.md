# research/160 G4 - correlation and blend value

Overlap window **2018-08 .. 2026-08** (8.00 years), monthly returns, 360 paths (OA seed x TN offset; QG_P tiled across seeds like TN). Drawdowns from the running peak of the full curve. Cash sleeve 5% p.a.

## Monthly return correlation (median across the 360 paths)

| pair | median | min | max |
|---|---:|---:|---:|
| QG_P vs TN | **0.471** | 0.422 | 0.542 |
| QG_P vs OA | **0.730** | 0.650 | 0.800 |
| TN vs OA | **0.413** | 0.282 | 0.566 |

A complement is normally wanted below ~0.40 monthly (the bar r/154 used).

## Blend value against the deployed TN+OA pair

| book | CAGR | [min..max] | MaxDD | Calmar | dCalmar vs pair | paired wins |
|---|---:|---:|---:|---:|---:|---:|
| TN+OA 50-50 (the deployed pair) | 33.68 | [27.81..38.34] | -14.01 | 2.369 | +0.000 | 0/360 |
| + QG_P at 10% | 32.73 | [27.73..37.28] | -14.90 | 2.209 | -0.172 | 1/360 |
| + CASH at 10% (the null) | 30.71 | [25.51..34.85] | -12.34 | 2.467 | +0.091 | 360/360 |
| + QG_P at 20% | 31.93 | [27.60..36.18] | -15.85 | 2.032 | -0.356 | 1/360 |
| + CASH at 20% (the null) | 27.77 | [23.22..31.40] | -10.65 | 2.586 | +0.212 | 360/360 |
| + QG_P at 25% | 31.48 | [27.52..35.61] | -16.41 | 1.941 | -0.442 | 1/360 |
| + CASH at 25% (the null) | 26.30 | [22.07..29.68] | -9.80 | 2.652 | +0.288 | 360/360 |
| + QG_P at 33% | 30.84 | [27.36..34.68] | -17.34 | 1.795 | -0.581 | 0/360 |
| + CASH at 33% (the null) | 23.97 | [20.24..26.96] | -8.50 | 2.794 | +0.427 | 360/360 |
| + QG_P at 40% | 30.29 | [27.20..33.85] | -18.25 | 1.665 | -0.707 | 0/360 |
| + CASH at 40% (the null) | 21.94 | [18.64..24.59] | -7.43 | 2.931 | +0.577 | 360/360 |
| QG_P standalone | 25.11 | [20.01..29.19] | -36.84 | 0.676 | -1.668 | 0/360 |
| TN standalone | 21.58 | [14.51..25.60] | -16.03 | 1.380 | -1.102 | 0/360 |
| OA standalone | 43.73 | [40.72..50.36] | -23.74 | 1.886 | -0.491 | 7/360 |

## Stress windows (return %, drawdown from the full-curve peak)

| window | TN+OA 50-50 (the deployed pair) | + QG_P at 10% | + CASH at 10% (the null) |
|---|---:|---:|---:|
| 2020 crash | -1.4 (-6.3) | -2.8 (-8.3) | -1.2 (-4.9) |
| 2022H1 grind | -5.3 (-11.0) | -7.1 (-11.9) | -4.6 (-9.7) |
