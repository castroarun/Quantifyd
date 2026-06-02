# Capstone — combined-edge prototype vs baseline (BS model, real NIFTY 5-min 2024-26, IV 17%)

BASELINE: ATM straddle every day, 1.3x premium stop. COMBINED: tight-open days only (open-15min < median), +/-0.4% underlying-move stop. Both 1-DTE structure, exit 14:45.

| metric | Baseline | Combined |
|---|---|---|
| n | 453 | 225 |
| perday | -1632 | -270 |
| win | 53 | 40 |
| worst | -14196 | -6366 |
| worst5 | -13256 | -5244 |
| sharpe | -5.8 | -2.0 |

**Read the LIFT, not the level** (BS-modelled premiums). The combined config trades fewer days (tight-open filter) with a bounded move-stop. Compare per-day, win%, and especially worst/worst5 (tail).
- Filter validity: opening-range->range-day is robust over 6 yrs (regime_long).
- 28-day REAL premiums have only ~4 one-DTE days (~2 after the tight filter) -> real-level confirmation needs the recorder to accumulate; this multi-year model is the lift/tail lens.