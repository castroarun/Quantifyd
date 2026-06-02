# Regime angles — gap days & CPR width vs short-vol favourability

STATUS: RUNNING

## Ask
"Look at different angles: gap up/down opening days (by DTE), how the range / 28-day
options system performed on those; narrow CPR vs wide CPR days."

## Layers
- **(1) Years (robust):** NIFTY 30-min 2020-26 (~1,565 days). Per day: overnight gap%,
  CPR width% (from prior H/L/C), opening-range%. Target: intraday |close-open| (small =
  range day = good for short-vol) + range-day rate. By gap bucket + CPR-width quartile,
  per-year stability. Tests CPR theory: WIDE CPR -> range day; NARROW CPR -> trend day.
- **(2) 28-day options (signal):** ATM straddle (real chain) P&L by gap bucket & CPR
  bucket, split by DTE. Small n -> directional.

## Caveats
28-day options = signal; years = underlying proxy. CPR width from prior-day H/L/C.


## VERDICT (DONE)
Gap (years 1564d): big gaps esp DOWN -> bigger moves, fewer range days (BAD for short-vol). CPR width:
BACKWARDS vs textbook (wide CPR -> more move) AND range-rate skill DECAYED to ~noise by 2024-26 (corr 0.30->0.04);
move-magnitude link persists (wide=bigger move). 28d real options: CPR narrow +13k / wide -17k (consistent);
gap-down +6.3k BEST but CONTRADICTS years -> small-sample luck (gap-fills), trust the years. 1-DTE positive across
all gap types (edge holds regardless of gap). Net: gap/CPR are minor/noisy secondary filters; 1-DTE + tight-open
remain the robust edges. STATUS: DONE.