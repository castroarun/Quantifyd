# research/162 Part C — correlation and blend value vs True North + Base Age

Overlap window **2018-08 .. 2026-08** (8.00 years), monthly returns, 360 paths (Base Age seed x True North offset; Quality Summit tiled across seeds like True North). Drawdowns from the running peak of the full curve. Cash sleeve 5.0% p.a. After tax, 25 bps a side.

**Cash-yield inconsistency, stated:** True North 6.5%, Base Age 5.5%, Quality Summit 5.0% — the pair is flattered by a few tenths of a point, i.e. the bias runs AGAINST the candidate.

## Return correlation (median across the 360 paths)

| pair | monthly | daily |
|---|---:|---:|
| QS baseline vs TN | **0.374** | 0.440 |
| QS baseline vs BA | **0.717** | 0.671 |
| QS best (W1) vs TN | **0.378** | 0.403 |
| QS best (W1) vs BA | **0.651** | 0.649 |
| TN vs BA | **0.337** | 0.430 |

A complement is normally wanted below ~0.40 monthly (the r/154 bar).

## Blend value against the honest pair

| book | CAGR | [min..max] | MaxDD | Calmar | dCalmar vs pair | paths improved |
|---|---:|---:|---:|---:|---:|---:|
| **TN + Base Age 50-50 (the honest pair)** | 24.42 | [19.00..27.37] | -13.71 | 1.769 | +0.000 | 0/360 |
| + QS baseline at 10% | 24.12 | [19.48..27.11] | -13.98 | 1.742 | +0.001 | 181/360 |
| + QS baseline at 20% | 23.74 | [19.93..26.80] | -15.03 | 1.565 | -0.188 | 70/360 |
| + QS baseline at 33% | 23.15 | [20.47..26.35] | -17.06 | 1.357 | -0.389 | 27/360 |
| + QS best (W1) at 10% | 24.36 | [19.37..27.41] | -13.48 | 1.834 | +0.054 | 237/360 |
| + QS best (W1) at 20% | 24.31 | [19.70..27.38] | -14.13 | 1.717 | -0.064 | 125/360 |
| + QS best (W1) at 33% | 24.02 | [20.07..27.26] | -17.06 | 1.383 | -0.343 | 8/360 |
| + CASH at 10% (the null) | 22.49 | [17.67..25.11] | -12.21 | 1.826 | +0.052 | 360/360 |
| + CASH at 20% (the null) | 20.55 | [16.32..22.86] | -10.74 | 1.900 | +0.118 | 360/360 |
| + CASH at 33% (the null) | 18.04 | [14.55..19.94] | -8.87 | 2.040 | +0.238 | 360/360 |
| TN standalone | 21.58 | [14.51..25.60] | -16.03 | 1.380 | -0.431 | 0/360 |
| BA standalone | 25.40 | [22.28..27.43] | -19.46 | 1.317 | -0.434 | 34/360 |
| QS baseline standalone | 21.02 | [17.04..23.77] | -31.10 | 0.671 | -1.135 | 0/360 |
| QS best (W1) standalone | 21.16 | [17.36..25.56] | -37.03 | 0.574 | -1.163 | 0/360 |

## Stress windows — return % (intra-window drawdown from the full-curve peak)

| window | TN + Base Age 50-50 (the honest pair) | + QS baseline at 20% | + QS best (W1) at 20% | + CASH at 20% (the null) |
|---|---:|---:|---:|---:|
| 2020 crash | -10.0 (-10.0) | -10.2 (-11.1) | -9.1 (-10.4) | -7.8 (-7.8) |
| 2022H1 grind | -11.8 (-11.8) | -14.8 (-14.9) | -12.2 (-12.6) | -9.1 (-9.1) |

