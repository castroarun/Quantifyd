# Phase 28 — Event-Driven Per-Stock Trailing-MA Exit — RESULTS

**Verdict: NO NET EDGE. "Instead of the gate" is KILLED (deep DD + tax-ruinous);
"in addition to the gate" is at best a wash (marginal Calmar gain eaten by STCG
churn). The market gate already does the heavy lifting; per-stock trailing on top
adds tax churn for negligible net benefit.** REF reproduced Phase 27/22 BASE exactly.

| Config | CAGR | Post-tax | MaxDD | Sharpe | Calmar | stk-exits | backfills |
|---|---|---|---|---|---|---|---|
| REF allcash+gate (locked) | 34.2% | **28.4%** | −22.2% | 1.82 | 1.54 | 0 | 0 |
| perStock L100  NO gate | 34.2% | 18.1% | −28.7% | 1.48 | 1.19 | 380 | 198 |
| perStock L150  NO gate | 34.6% | 21.2% | −31.8% | 1.47 | 1.09 | 166 | 103 |
| perStock L200  NO gate | 34.7% | 24.2% | −33.0% | 1.46 | 1.05 | 80 | 56 |
| perStock L100  +gate | 33.8% | 25.8% | −20.9% | 1.75 | 1.62 | 117 | 112 |
| perStock L150  +gate | 34.0% | 26.4% | −21.8% | 1.77 | 1.56 | 39 | 55 |
| perStock L200  +gate | 34.9% | 27.6% | −21.9% | 1.81 | 1.59 | 15 | 37 |

## Why it fails

1. **A per-stock MA can't replace a market circuit-breaker.** Standalone (no gate),
   all three lengths draw down −28.7 to −33.0% — as deep as / deeper than Phase 11's
   −30.2%. In a broad crash, stocks break their MAs one at a time *after* each has
   already fallen; there's no single fast "everyone out" signal. Event-driven cadence
   did NOT fix this. The Nifty gate cuts DD to −22% precisely because it's collective.

2. **Event-driven exit + backfill is tax-ruinous.** Every replacement held <365d and
   sold at a gain is taxed 20%. L100 standalone churns 380 exits + 198 backfills →
   post-tax collapses 34.2% gross → **18.1%** net (vs locked 28.4%). The churn, not the
   gross signal, is what kills it.

3. **No good MA length — the two failure modes trade off.** Shorter MA (L100) = more
   exits = more tax bleed; longer MA (L200) = fewer exits but laggier = DEEPER DD
   (−33%). You cannot win both ends. (NO-gate: post-tax 18→24% as L 100→200, but DD
   −28.7→−33%.)

## The "in addition to" version (per-stock + gate)

Best is **L200 + gate**: Calmar 1.59 (vs locked 1.54), DD −21.9 (vs −22.2), Sharpe
1.81 (≈1.82) — marginally better on risk, but **post-tax 27.6% vs 28.4% (−0.8pp)**.
You pay ~0.8pp of net CAGR in churn-tax to shave ~0.3pp of drawdown. A wash at best,
and it's strictly dominated by Phase 27's `allcash + weekly re-entry` (Calmar **1.72**,
post-tax **29.0%**), which improves risk AND return with far less turnover.

## Recommendation

- **Do not replace the Nifty gate** with per-stock MAs — tested, decisively worse.
- **Do not add** event-driven per-stock trailing on top either — net-negative after tax.
- The locked per-stock-SMA100 + 12% trail applied **at month-ends** (low churn) stays
  the right amount of stock-level control; the weekly event-driven version over-trades.
- If any single upgrade is worth promoting, it is **Phase 27 `allcash + weekly
  re-entry`**, not this. The "optimal exit point" (G2) refinement is NOT worth pursuing
  here — exit-tuning can trim the tax churn but cannot fix the structural −30% standalone
  DD, and the +gate prize is too small to chase.

## Caveats

- v1 exit = simple close-below-own-MA, weekly. A confirmation/buffer exit would lower
  churn (helping post-tax) but not the standalone DD; the conclusion holds.
- NIFTYBEES gate close-to-close (no true OHLC), unchanged. Single window, not re-OOS'd.
- Backfill = one-for-one into freed slots at per-slot budget; monthly RS rotation intact.
