# Phase 25 — keep-top8 + conditional refill: REJECTED

**Verdict: the gated refill LOSES decisively. Plain keep-top8 (no
refill while risk-off) stays the best risk-off action. Re-engaging the
universe during a downturn is the momentum-crash trap — confirmed a 3rd
time (cf. Phase-18 FORTIFIED-B, Phase-22 "no-regime").**

## Test (validated Phase-22 weekly daily-marked engine, fresh VPS data → 2026-05-15)

| Config | CAGR | Post-tax @20% | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|---|
| ref-A BASE SMOOTHEST (all-cash) | 34.2 | 28.4 | −22.2 | 1.82 | 1.54 |
| **ref-B keep-top8 (holdings-only)** | 33.6 | 28.3 | **−20.2** | 1.71 | **1.66** |
| V1 cond-refill cap-15 | 33.8 | 25.4 | −33.8 | 1.48 | 1.00 |
| V2 cond-refill cap-12 | 34.7 | 27.0 | −33.9 | 1.56 | 1.02 |
| V3 cond-refill cap-10 | 34.7 | 27.9 | −33.6 | 1.60 | 1.03 |

Every refill cap blows daily MaxDD out to **~−34%** (vs keep-top8
−20.2%) and crushes Calmar to **~1.0** (vs 1.66). A tighter cap (more
forced cash) only marginally limits the bleed — the mechanism itself is
the flaw, not its size.

## Why — and why the user's circuit-breaker intuition didn't hold

The proposal assumed the strict gate (within-10%-ATH + above-own-100SMA
+ q0.5) would self-limit refills in a panic. It does in a **deep**
crash. But the damage happens in **shallow / early-stage risk-off**:
there, fresh names DO still pass the gate, the rule buys them, and they
then roll over as the downturn develops. Per-year shows it exactly:

| Year | BASE | keep-top8 | cap-15 | cap-12 | cap-10 | note |
|---|---|---|---|---|---|---|
| 2016 | 39.5 | 27.9 | **0.4** | 7.6 | 11.3 | choppy risk-off — refill bought + bled |
| 2025 | 5.3 | −6.9 | **−16.0** | −17.9 | −19.6 | 2025 mid-cap risk-off — re-bought names fell hard |

The refilled names are precisely the ones a developing bear catches
down next. keep-top8's discipline — *hold only your own strongest
survivors, add nothing until risk-on* — is what makes it a clean
de-risk rather than a return-chase.

## Decision

**No change to the system.** keep-top8 stays "no refill while
risk-off". This is the 3rd independent confirmation that re-engaging
the universe mid-downturn loses (Phase-18 FORTIFIED-B rejected;
Phase-22 "A no-regime" Cal 0.91; Phase-25 cond-refill Cal ~1.0).

Artifacts: `phase25_cond_refill.csv`, `phase25_cond_refill_peryear.csv`.
Runner `scripts/25_kt8_conditional_refill.py` (references use the
unmodified validated Phase-22 engine; only the refill path is new).
