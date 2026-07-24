# research/77 P2 — SL tightening, EXTENDED to 743 days via synthetic straddle (NIFTY 5-min + real India VIX)

**Verdict: the 15-day real result DID NOT REPLICATE — treat P1 as noise. On 743 days of real NIFTY price
action (2023-01→2026-03), a synthetic BS straddle (real intraday VIX as IV) says: the 30%-off-morning stop
IS too loose (your core intuition holds), but the SPECIFIC "re-anchor @1 PM" is only a small, consistent
positive (+11 to +32/lot/yr) — NOT the star. On the big sample the AGGRESSIVE tighteners win by cutting the
fat left tail: trail-to-breakeven +176/lot, ratchet +159/lot vs baseline. This is the OPPOSITE of the
15-day ranking. Net: "some tightening beats the loose 30%" is robust; WHICH one is NOT settled (the two
samples disagree) and needs real-premium data. SYNTHETIC caveat is large — see below.**

## Method
Synthetic 09:16 ATM straddle: premiums = Black-Scholes on NIFTY 5-min spot, IV = **real India VIX 5-min**,
theta from calendar time-to-Thursday-expiry, r=6.5%. Same per-leg SL policies. n=**743 days**
(2023-01-01→2026-03-26; DTE0-1 298, DTE2+ 445; mean VIX 13.7). Net 0.15%/leg, per 1 lot=65. market_data.db.

## Results (mean net/lot vs BASELINE, n=743)
| Policy | meanNet | vs BASE | median | win% | worst | std |
|---|---|---|---|---|---|---|
| **TRAIL_BE** | −257 | **+176** | −559 | 29% | −7,944 | 1,943 |
| **RATCHET_0.30** | −274 | **+159** | −426 | 38% | −7,944 | 1,619 |
| TIME_12:00 | −370 | +63 | +124 | 53% | −10,009 | 2,225 |
| TIME_11:15 | −374 | +59 | +53 | 51% | −10,009 | 2,188 |
| TIME_14:00 | −388 | +45 | +215 | 54% | −10,009 | 2,334 |
| TIME_13:30 | −395 | +38 | +205 | 54% | −9,537 | 2,308 |
| **TIME_13:00** | −409 | **+24** | +130 | 53% | −10,009 | 2,290 |
| PLOCK_50 | −410 | +23 | +200 | 54% | −10,009 | 2,309 |
| **BASELINE (30% off entry)** | **−433** | 0 | +205 | 55% | −10,009 | 2,385 |
Per-year TIME_13:00 vs BASE: 2023 +30, 2024 +32, 2025 +11, 2026 +23 — small but positive every year.

## What it says (vs the 15-day P1)
1. **P1 (15 real days) does NOT hold.** There, TIME_13:00 was best (+306) and trail/ratchet were worst.
   On 743 days it FLIPS: trail/ratchet best (+176/+159), TIME_13:00 only +24. ⇒ the 15-day sample was a
   benign, low-move window; its ranking was noise. This is exactly why we don't trade 15-day results.
2. **Your core intuition is CONFIRMED on the big sample:** the fixed 30%-off-morning stop is the WORST of
   all policies (−433). *Every* tightening beats it. The 30% left tail (worst −10,009) is the problem.
3. **The mechanism at scale = TAIL-CUTTING, not give-back-locking.** BASELINE median is +205 (most days
   the straddle decays fine) but the mean is dragged negative by big-move days. Tight/trailing stops chop
   those tails (worst −7,944, std 1,943 for trail_be) — that's where the +176 comes from. You win LESS
   often (trail_be win% 29%) but lose far less on the bad days.
4. **"Re-anchor @1 PM"** = a small, robust positive (+24 mean, +11..+32/yr) — safe and cheap, but not the
   biggest lever on price action alone.

## BIG caveats (why this is an ESTIMATE, not a green light)
- **Synthetic = BS at fair value → NO vol-risk-premium.** Real short straddles earn IV>realized (why the
  live book made +12k Fri); BS doesn't, so the −433 baseline is a synthetic artifact, NOT a claim the live
  systems lose. Only the RELATIVE policy comparison is meaningful — and even that omits real skew,
  weekly-vs-30d-VIX term structure, IV-crush-at-open, and bid/ask. Real premiums move differently, which is
  likely WHY P1 (real, 15d) and P2 (synthetic, 743d) disagree.
- Thursday weekly-expiry approx; calendar-time theta clock.
- ⇒ The robust cross-sample conclusion is only the DIRECTION: **a loose 30%-off-morning stop is
  sub-optimal; some tightening helps.** WHICH tightening is unresolved.

## Recommendation
- Do NOT hard-switch on either study. **Paper-shadow two variants on the live systems** to gather REAL
  premium days: (a) trail-to-breakeven / ratchet (the tail-cutter the big sample likes), (b) re-anchor @1PM
  (the mild consistent one). Compare on real fills.
- Re-run P1 (real premiums) when the recorder has 40+ clean days; that's the arbiter, not the synthetic.
