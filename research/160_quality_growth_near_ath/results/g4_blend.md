# research/160 G4 - correlation and blend value

Overlap window **2018-08 .. 2026-08** (8.00 years), monthly returns, 360 paths (OA seed x TN offset; QG_B tiled across seeds like TN). Drawdowns from the running peak of the full curve. Cash sleeve 5% p.a.

## Monthly return correlation (median across the 360 paths)

| pair | median | min | max |
|---|---:|---:|---:|
| QG_B vs TN | **0.374** | 0.303 | 0.456 |
| QG_B vs OA | **0.624** | 0.538 | 0.723 |
| TN vs OA | **0.413** | 0.282 | 0.566 |

A complement is normally wanted below ~0.40 monthly (the bar r/154 used).

## Blend value against the deployed TN+OA pair

| book | CAGR | [min..max] | MaxDD | Calmar | dCalmar vs pair | paired wins |
|---|---:|---:|---:|---:|---:|---:|
| TN+OA 50-50 (the deployed pair) | 33.68 | [27.81..38.34] | -14.01 | 2.369 | +0.000 | 0/360 |
| + QG_B at 10% | 32.26 | [27.45..36.96] | -14.43 | 2.232 | -0.108 | 39/360 |
| + CASH at 10% (the null) | 30.71 | [25.51..34.85] | -12.34 | 2.467 | +0.091 | 360/360 |
| + QG_B at 20% | 30.99 | [27.04..35.54] | -15.64 | 2.017 | -0.355 | 16/360 |
| + CASH at 20% (the null) | 27.77 | [23.22..31.40] | -10.65 | 2.586 | +0.212 | 360/360 |
| + QG_B at 25% | 30.42 | [26.83..34.82] | -16.45 | 1.887 | -0.501 | 5/360 |
| + CASH at 25% (the null) | 26.30 | [22.07..29.68] | -9.80 | 2.652 | +0.288 | 360/360 |
| + QG_B at 33% | 29.47 | [26.46..33.64] | -17.96 | 1.693 | -0.709 | 0/360 |
| + CASH at 33% (the null) | 23.97 | [20.24..26.96] | -8.50 | 2.794 | +0.427 | 360/360 |
| + QG_B at 40% | 28.59 | [26.12..32.60] | -19.31 | 1.543 | -0.863 | 0/360 |
| + CASH at 40% (the null) | 21.94 | [18.64..24.59] | -7.43 | 2.931 | +0.577 | 360/360 |
| QG_B standalone | 21.02 | [17.04..23.77] | -31.10 | 0.671 | -1.732 | 0/360 |
| TN standalone | 21.58 | [14.51..25.60] | -16.03 | 1.380 | -1.102 | 0/360 |
| OA standalone | 43.73 | [40.72..50.36] | -23.74 | 1.886 | -0.491 | 7/360 |

## Stress windows (return %, drawdown from the full-curve peak)

| window | TN+OA 50-50 (the deployed pair) | + QG_B at 10% | + CASH at 10% (the null) |
|---|---:|---:|---:|
| 2020 crash | -1.4 (-6.3) | -2.3 (-6.4) | -1.2 (-4.9) |
| 2022H1 grind | -5.3 (-11.0) | -7.3 (-12.6) | -4.6 (-9.7) |
