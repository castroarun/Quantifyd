# research/77 — 09:16 straddle: per-leg SL-tightening sweep

**Verdict: G1 PROBE. Your instinct is HALF-right. The intuitive move — trailing/ratcheting the SL down
as premiums decay, or profit-locking to breakeven — is REFUTED: every one of those whipsaws you out of
legs that recover and LOSES vs the plain 30%-off-entry stop. BUT a single LATE re-anchor (at ~13:00,
reset the SL to 30% above the then-current decayed premium) shows a modest edge: +933 vs +628/lot
(+306/lot, +~₹600/day at 2 lots), a shallower worst day (−3,301 vs −5,548), and better on BOTH DTE
buckets. Small sample (n=15) → paper-test, don't flip live.**

## Method
Reconstructed 09:16 ATM straddle from options_data.db (spot=underlying_spot TABLE; premiums=option_chain
that strike, full 09:16–15:15 series). Each leg walks its own premium path; stops when premium ≥ its
(policy) SL, booking that leg; the other runs to 15:15. Net of 0.15%/leg. Per 1 lot=65. **No re-enter**
after a stop (isolates the SL-timing effect — caveat). n=**15 days** (DTE0-1: 9, DTE2+: 6).

## Results (mean net/lot vs BASELINE)
| Policy | meanNet | vs BASE | win% | worst | avg stops | net-protection (saved−premature) |
|---|---|---|---|---|---|---|
| **BASELINE (30% off entry)** | **+628** | 0 | 67% | −5,548 | 0.87 | — |
| **TIME_13:00** (re-anchor SL @1pm) | **+933** | **+306** | 67% | **−3,301** | 1.07 | **+864** (only positive one) |
| TIME_12:00 | +698 | +70 | 67% | −3,566 | 1.07 | −2,659 |
| TIME_11:15 | +595 | −33 | 67% | −3,566 | 1.07 | −4,199 |
| PLOCK_40 | +525 | −103 | 60% | −3,566 | 0.93 | −5,249 |
| PLOCK_50 / 60 | +393 | −235 | 60% | −5,548 | 0.93 | −7,228 |
| TRAIL_BE (to breakeven) | +195 | −432 | **27%** | −1,491 | 1.60 | −10,185 |
| RATCHET_0.30 | +129 | −498 | 47% | −1,919 | 1.73 | −11,177 |
| RATCHET_0.20 / 0.40 | +109 / +132 | −518 / −496 | 60/40% | | ~1.7 | ~−11,400 |
Cost sensitivity (0.10/0.15/0.20%): rankings stable; TIME_13:00 stays top (+945/+933/+921).

## Why (the mechanism)
- **Tightening early or continuously backfires** because straddle P&L is BACK-LOADED (research/76: peaks
  ~15:00). A leg often spikes intraday then decays back; a tight/ratcheted/breakeven stop books that spike
  as a loss and forfeits the late theta. TRAIL_BE's win% collapses to **27%** — you get stopped constantly.
  The "premature stop-out" cost (₹30–47k across the sample) dwarfs the give-back it saves.
- **A single LATE re-anchor works** because by ~1pm the decay has banked the gain and the day's structure
  is set: resetting the stop to 30% above the *now-cheap* premium protects the banked profit against a
  late-afternoon spike (the give-back research/76 found) WITHOUT whipsawing during the still-active
  morning/midday. It's the ONLY policy where saved > premature (net +864).

## So, to your question
Yes — the fixed 30%-off-MORNING-premium stop IS too loose late in the day. But the fix is **NOT** to trail
it down as it decays (that whipsaws) — it's to **re-anchor it ONCE, LATE (~1pm), to 30% above the
then-current premium**. Tighten late and once, not early and continuously.

## Caveats (why this is G1, not a live change)
- **n=15 days, one regime** (2026 Apr–Jul). The +306 edge is suggestive, and TIME_13:00 was the winner
  among 11 policies tested → multiple-testing / small-sample risk; could be partly luck.
- Reconstructed/idealized: no re-enter, no slippage beyond 0.15%, nearest-90s premiums.
- Recorder gaps cap the clean sample at ~15 days (same bottleneck as research/76).

## Next
1. Paper-shadow "re-anchor SL @13:00 to 30% above current" on the live 916 systems; accumulate forward days.
2. Re-run when the recorder has ≥40 clean days; check monotonicity across re-anchor times (12:30/13:00/13:30)
   and per-DTE stability before any live change.
