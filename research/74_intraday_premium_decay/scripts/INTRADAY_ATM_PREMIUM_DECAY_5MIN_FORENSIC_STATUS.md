# Intraday ATM Premium Decay-then-Rise — ~15:00 crush / ~15:15 pickup, 53 days per-minute NIFTY chain

STATUS: RUNNING

## The Ask

**What you asked:** "in the live paper trading on our NAS systems, and on personal observation,
I'm finding the option premiums decay quickly around 3 PM every day and rise around 3:15 PM.
Please check — find the exact time range of the sudden decay, and any concurrency with
indicators / day's price-action patterns / CPRs at that time."

**What we're actually testing:** Across all recorded days in `options_data.db` (per-snapshot
NIFTY option chain with IV + greeks, ~3s cadence), what is the *average* intraday path of the
ATM straddle premium (CE+PE) and ATM IV in the final ~2 hours? Specifically:
1. Is there a repeatable **decay** into ~15:00 and a **rise** into ~15:15–15:30? Pin the minutes.
2. Is the move a **volatility** story (ATM IV dips then rises — vega) or a **directional** one
   (underlying drifts, a held strike gains intrinsic — delta)? The ATM straddle re-picked each
   minute is direction-neutral, so a *rise* in it while time-to-expiry falls ⟹ IV rose.
3. Does it depend on **DTE** (expiry-day theta is violent) and on **day type** (range vs trend)?

## The Base — what's measured

- **Instrument:** front (nearest) weekly expiry, ATM strike = round(spot/50)*50, re-picked each
  snapshot. Straddle premium = ATM_CE.ltp + ATM_PE.ltp. ATM IV = mean(ATM_CE.iv, ATM_PE.iv).
- **Window:** 13:40 → 15:30, resampled to 1-min (mean within minute).
- **Normalisation:** per day, index premium to the 14:00 value (=100); IV as absolute % and Δ
  from 14:00; |Δspot| = |underlying − underlying@14:00|. Then average across days per minute.
- **Universe/period:** all 53 trading days, 2026-04-20 → 2026-07-07.
- **Splits:** DTE 0-1 vs 2+; (pass 2) range vs trend day, CPR width.

## Plan

Pass 1 (this run): confirm + time the decay/rise on ALL days; IV-vs-direction decompose; DTE split.
Pass 2 (if confirmed): concurrency with day-type (range/trend), CPR width, and the exact
sub-minute shape around the 15:00/15:15 inflection.

## Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-07 | Folder + STATUS written; pass-1 script built | 53 days, 3s cadence confirmed |
