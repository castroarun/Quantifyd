# Part (c) — NIFTY range/trend regime (2024-03-04→2026-03-25, 451 days)

Proxy: short premium-selling profits on small-|close-open| (range) days. Causal morning features only. **~500 days, per-year stability shown — the most robust part of this study, but it's an UNDERLYING proxy (no historical option P&L).**

## Feature → |move| correlation (positive = feature flags a TREND/move day)

| feature | all | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| gap_pct | 0.22 | 0.14 | 0.26 | 0.27 |
| open15_pct | 0.36 | 0.35 | 0.37 | 0.39 |
| prevrng_pct | 0.00 | 0.01 | 0.04 | -0.16 |
| prevret_pct | -0.03 | -0.03 | 0.05 | -0.22 |

## Avg intraday move %% + range-day rate by weekday
| weekday | n | avg move %% | range-day rate |
|---|---|---|---|
| Mon | 88 | 0.48 | 0.48 |
| Tue | 95 | 0.46 | 0.49 |
| Wed | 89 | 0.39 | 0.66 |
| Thu | 90 | 0.53 | 0.48 |
| Fri | 89 | 0.59 | 0.38 |

## Range-day rate by opening-15min-range quartile
| quartile | n | avg move %% | range-day rate |
|---|---|---|---|
| Q1-tight | 113 | 0.37 | 0.63 |
| Q2 | 113 | 0.45 | 0.53 |
| Q3 | 112 | 0.50 | 0.47 |
| Q4-wide | 113 | 0.64 | 0.36 |

## Range-day rate by |overnight gap| quartile
| quartile | n | avg move %% | range-day rate |
|---|---|---|---|
| Q1-flat | 113 | 0.52 | 0.47 |
| Q2 | 113 | 0.45 | 0.50 |
| Q3 | 112 | 0.48 | 0.54 |
| Q4-biggap | 113 | 0.52 | 0.50 |

- Read: features with a STABLE per-year correlation are usable filters; flickering-sign ones are noise.
- Expiry-day shifted historically (Thu->Tue) so weekday!=DTE across all years; use as regime hint, combine with the 0/1-DTE options finding.