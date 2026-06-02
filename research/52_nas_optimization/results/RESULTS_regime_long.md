# Opening-range validation over 6 years (NIFTY 30-min, 2020-01-01->2026-05-05, 1565 days)

Tighter open should predict a smaller-move (range) day. **all-period corr(open30, |move|) = 0.52** (positive = tight open -> calmer day). Per-year stability across COVID(2020)/bull(2021)/bear(2022)/2023-26 is the real test.

## Per-year correlation (open30 vs |move|)

| year | corr |
|---|---|
| 2020 | 0.56 |
| 2021 | 0.30 |
| 2022 | 0.28 |
| 2023 | 0.24 |
| 2024 | 0.57 |
| 2025 | 0.40 |
| 2026 | 0.45 |

## Range-day rate by opening-30min quartile (pooled)
| quartile | n | avg move %% | range-day rate |
|---|---|---|---|
| Q1-tight | 392 | 0.41 | 0.63 |
| Q2 | 391 | 0.45 | 0.60 |
| Q3 | 391 | 0.58 | 0.46 |
| Q4-wide | 391 | 0.95 | 0.31 |

## Tight (Q1) vs Wide (Q4) range-day rate — PER YEAR
| year | tight-open range-rate | wide-open range-rate |
|---|---|---|
| 2020 | 0.76 | 0.26 |
| 2021 | 0.60 | 0.35 |
| 2022 | 0.65 | 0.35 |
| 2023 | 0.56 | 0.39 |
| 2024 | 0.63 | 0.37 |
| 2025 | 0.66 | 0.37 |
| 2026 | 0.76 | 0.20 |

- If tight>wide every year, the opening-range filter is robust across regimes (not a 2024-26 fluke).