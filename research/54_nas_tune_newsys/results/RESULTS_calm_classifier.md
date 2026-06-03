# research/54 Stage 6 — multi-year calm-day classifier (NIFTY 5-min, 451 days, OOS per-year)

Predict today's intraday move %% from 09:20-known features; LOWER predicted move => sell. OOS = train on all OTHER years, test on held-out year. **VALIDATION layer (years of real paths).**

## Per-year OOS: corr(predicted, actual move) — multi-feature vs opening-range alone
| year | n | corr multi | corr fc-only | calm30 avg move %% | all-day avg move %% |
|---|---|---|---|---|---|
| 2024 | 180 | 0.34 | 0.32 | 0.38 | 0.49 |
| 2025 | 215 | 0.30 | 0.33 | 0.39 | 0.48 |
| 2026 | 56 | 0.39 | 0.35 | 0.43 | 0.55 |

## Pooled feature corr with today's move (sign/strength)
| feature | corr |
|---|---|
| fc_range | 0.33 |
| gap | 0.22 |
| pd_range | 0.01 |
| pd_move | -0.02 |
| wd | 0.10 |

## Read
- If `corr multi` > `corr fc-only` in MOST years -> stacking adds robust predictive power over opening-range alone.
- `calm30 avg move` << `all-day avg move` every year => the classifier's calmest-30%% days really are calmer (tradeable filter).
- Causal features only (known by 09:20). Linear model = conservative; a tree could do better but risks overfit on this n.