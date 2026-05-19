# Tri-Sleeve Combined Book — Honest Verdict

**Status: combined engine built & run (laptop smoke, representative —
Sleeve-1 is bit-identical laptop vs VPS; option legs are deterministic
flat-IV BS). Canonical VPS re-run pending for sign-off.**

## The one-line answer

**The overlays do NOT transform the base. The only defensible addition
is KC6 monetised as *credit* spreads at realistic IV — a modest
Calmar gain (1.44 → ~1.50) with marginally shallower drawdown. The
short/hedge sleeve (Sleeve 3) should be OFF: covered-calls/collar are
value-neutral-to-negative on a momentum book, and systematic-short
actively wrecks drawdown. Base-alone remains the honest benchmark; the
"big CAGR" combined cells are an optimistic-IV artifact, not bankable.**

## Numbers (gross, monthly-NAV, 2014→2026, same methodology as research/41)

| System | CAGR | MaxDD | Calmar | vs base |
|---|---|---|---|---|
| **BASE (Sleeve-1 only)** | 35.28% | −24.58% | **1.44** | — |
| + S2 credit, IV0.25, risk10% | 35.51% | −24.23% | 1.47 | +0.03 ✅ honest |
| + S2 credit, IV0.30, risk15% | 36.55% | −23.81% | 1.54 | +0.10 ✅ honest |
| + S2 debit, IV0.20, risk15% | 46.11% | −27.89% | 1.65 | +0.22 ⚠ optimistic-IV |
| + S2 debit, IV0.30, risk15% | 39.86% | −27.99% | 1.42 | −0.01 (debit fades at fair IV) |
| + S2best + S3-C short (sr10) | 33–36% | **−44 to −45%** | 0.74–0.82 | **−0.62 to −0.70 ✗ harmful** |
| + S2best + S3-A covered-call | ~45% | −27.7% | ~1.63 | rides S2 CAGR; standalone −ve |

Post-tax: base canonical = research/41's **28.9%** (we do not re-derive;
caveat C2). The honest combined uplift is ≈ +0.5–1.5pp CAGR and
~0.5–0.8pp shallower DD — real but small, and IV-assumption-fragile.

## Why each sleeve landed where it did

**Sleeve 1 (base).** Replayed bit-faithfully (35.28%/−24.58% vs
research/41 35.3%/−24.6%). Phase-0 gate passed. Sound foundation.

**Sleeve 2 (KC6 options).** 361 KC6 trades, 65–67% WR (= KC6 native
edge, faithfully reproduced). *Credit* (bull-put) spreads are the
honest contributor: small positive, slightly DD-reducing, and they
get *better* at higher IV — which is conservative, because real IV
spikes when KC6 fires on a selloff (caveat C1 cuts in our favour
here). *Debit* (bull-call) shows large CAGR but only in the cheap-IV
corner (IV0.20); at fair/high IV the debit edge evaporates. Trusting
the IV0.20 debit numbers would be dishonest — flagged, not headlined.

**Sleeve 3 (short/hedge) — the decisive negative finding.** The
standalone smoke *looked* like C (systematic-short) was the winner
(+15% of book). That was **path-blind** (a raw P&L sum ignoring when
the gains/losses land). The path-aware, risk-capped combined engine
reverses it completely: C sized to 5–10% book-risk **deepens MaxDD to
−34%…−45%** and crushes Calmar to 0.74–1.28 — shorting the weakest-RS
mid-caps bleeds through momentum bull markets. Covered-calls (A) and
collar (B) are value-destructive standalone (they sell off exactly the
fat-tail winners that *are* the momentum edge) and only ~F&O-reach
131/1770 held name-months anyway (the RS book is mostly illiquid
non-F&O mid-caps — a structural mismatch). **No Sleeve-3 variant earns
its risk. Recommend Sleeve 3 = OFF.**

## Recommendation

1. **Ship-candidate = Base + KC6 *credit* spreads (IV-conservative,
   risk ≤10–15% of book), Sleeve 3 OFF.** Expected honest effect:
   Calmar ~1.44→~1.50, DD ~0.5–0.8pp shallower, CAGR +~0.5–1.5pp.
   A marginal, defensible improvement — not a step-change.
2. **Do NOT deploy any debit-spread or short sleeve on these numbers.**
   Debit's apparent edge is an IV-assumption artifact; the short
   sleeve is drawdown-destructive.
3. **The base alone is still a perfectly honest choice.** If the user
   does not want option-execution complexity for ~+0.05 Calmar, the
   research/41 base stands on its own.
4. **Bankable confirmation needs a paid historical options-chain
   source** (caveat C1). Until then the credit-spread uplift is
   *indicative*. This is the genuine next phase, not more sweeping.

## Caveats (binding — cite with every number)

C1 flat-IV BS, no historical chains — options P&L indicative not
tradeable; debit numbers IV-fragile. C2 base post-tax = research/41's
28.9% (not re-derived). C3 single-book margin cap modelled; 0
infeasible months at tested risk-% (overlays are small vs book). C4
covered-call assignment = expiry-intrinsic cap. C5 no performance
guarantee; nothing wired live; laptop smoke pending canonical VPS
re-run for sign-off.
