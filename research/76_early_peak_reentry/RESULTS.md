# research/76 — "Early-peak then erode?" for the 09:16 ATM short straddle

**Verdict: HYPOTHESIS NOT SUPPORTED (NO EDGE for book-early/re-enter). Surfaced the real driver = move-stop churn, not the decay curve.**

## The ask
User observed the NAS trades go into profit in the first 30–60 min after the 09:16 entry, then
"volatility seeps in" and erodes it — proposing: book profit early, re-enter later.

## Method
Reconstructed the pure 09:16 ATM short straddle (held, fixed-strike, real chain premiums) intraday
P&L path across all recorded days. Source: options_data.db (underlying_spot TABLE for ATM +
option_chain ±3-min windows). P&L(T) = (credit_0916 − prem_T)·65 per lot. Read-only.
Script: `verify_early_peak.py`. n = **14 clean days** (recorder gaps + strict ATM/±3min filter cut
~54 → 14). DTE0-1: 9, DTE2+: 5.

## Findings
- First 15–30 min is on average **NEGATIVE** (09:31 −216, 09:46 −85/lot); turns green ~+60 min (10:16 +227).
- P&L **builds through the day, peaks near the CLOSE**: 15:00 +887, 15:15 +686/lot; per-day peak at
  15:00–15:15 on **57%** of days; only **7% (1/14)** peaked within the first 60 min.
- Early-peak (best of +15/30/60m) vs EOD: +329 → +686, **give-back = −357/lot (it grows, not erodes)**.
  Erosion (peak60 > EOD) on only 21% of days.
- Mild give-back in the **11:15→13:00 lull** (+579 → +409) that recovers — consistent with research/75.
- Booking early + re-entering would **forfeit** the bulk of theta, which concentrates late
  (consistent with research/74: 15:15 = peak near-expiry decay).

## Real driver of the user's live "erosion"
Not the straddle's decay curve (which favours holding to EOD) but the **move-stop churn**: during
intraday vol a leg gets run over, the stop books the loss and re-centers. Idealized held straddle
recovers by EOD; the live stops don't. Today: live ATM2 gave back from +4.1k peak on the late move;
10-lot paper ATM2 churned −31.9k across 8 move-stop legs.

## Caveats
- Small sample (14 days), single recent regime (2026 Apr–Jul). Recorder gaps limited it.
- Idealized held straddle (no stops/slippage) — the LIVE shape differs because of stops.

## Next (owed, not done)
1. Expand n (relax strike/window; use more of the 54 days) to confirm the late-peak shape.
2. Backtest the ACTUAL lever: reduce move-stop churn — wider bands / hold-through-lull / fewer
   re-centers / no-re-enter — vs the current 0.4% re-center, net of costs, per DTE.
3. If anything, test a LATE-exit tilt (hold to ~15:00) not an early exit — decay is back-loaded.
