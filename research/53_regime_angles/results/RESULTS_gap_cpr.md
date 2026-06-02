# Gap & CPR regimes — NIFTY 30-min (2020-01-02->2026-05-05, 1564 days)

Range day = small intraday |close-open| (good for short-vol). Higher range-rate = more short-vol-friendly.

## Range-day rate + avg move by GAP bucket
| gap bucket | n | avg move %% | range-day rate |
|---|---|---|---|
| gap dn >0.5% | 155 | 0.92 | 0.40 |
| dn 0.15-0.5 | 207 | 0.53 | 0.52 |
| flat ±0.15 | 446 | 0.54 | 0.51 |
| up 0.15-0.5 | 480 | 0.54 | 0.54 |
| gap up >0.5% | 276 | 0.70 | 0.45 |

## Range-day rate by CPR width (narrow->wide)
| CPR width | n | avg move %% | range-day rate |
|---|---|---|---|
| Q1-narrow | 391 | 0.51 | 0.53 |
| Q2 | 391 | 0.50 | 0.58 |
| Q3 | 391 | 0.58 | 0.48 |
| Q4-wide | 391 | 0.81 | 0.41 |

## CPR: narrow vs wide range-day rate, PER YEAR (corr cpr_width vs |move| in parens)
| year | narrow CPR range-rate | wide CPR range-rate | corr |
|---|---|---|---|
| 2020 | 0.58 | 0.32 | 0.30 |
| 2021 | 0.53 | 0.39 | 0.18 |
| 2022 | 0.58 | 0.37 | 0.09 |
| 2023 | 0.55 | 0.48 | 0.11 |
| 2024 | 0.50 | 0.50 | 0.04 |
| 2025 | 0.52 | 0.50 | 0.06 |
| 2026 | 0.48 | 0.45 | 0.10 |

- CPR theory holds if WIDE CPR range-rate > NARROW consistently. Gap: large gaps (up or dn) should show higher move / lower range-rate if gaps drive trend; if symmetric gap-fill, range-rate stays ~flat.