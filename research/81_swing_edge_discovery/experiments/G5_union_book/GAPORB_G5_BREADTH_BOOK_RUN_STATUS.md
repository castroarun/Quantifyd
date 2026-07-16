# G5 Breadth Book — Construction Ledger & Findings (2015-02..2023-12, IS+Val)

STATUS: DONE — CAPACITY-CONSTRAINED (see ledger)

Underlying signal: gap>=0.25% + ORB long, validated at trade level on 77 F&O
names (A4: W12 +20.9bps t=5.62, 71% syms, 86% yrs). SIX book constructions
tried, ALL reported (no silent discards):

| v | Construction | Result | Diagnosis |
|---|---|---|---|
| G4 | 10 instruments, cap6, risk 1%/0.5% | Sharpe 1.00, DD -17%, Calmar 0.77 | best implementable so far |
| G5v1 | union 3 triggers, 78 names, cap6 | CAGR -31%, DD -98% | clock-order intake filled book with C2 (fires bar 1, worst trigger) — adverse selection |
| G5v2 | ORB-only, cap6 | Sharpe 0.49, DD -37% | taken 10.4bps vs skipped 17.4bps — early-clock adverse |
| G5v3a/b | W12, cap15/8, risk 0.25/0.4% | DD -55%, 2015 -41% | leverage: equal-RISK sizing => 3-4x gross notional; gap risk on notional |
| G5v4 | + book notional cap 150% | DD -43%, 2015 -33% | narrower but persists |
| G5v5 | equal-NOTIONAL 10%/pos (the stat A4 validated) | Sharpe 0.58, DD -43%, 2015 -32% | capacity: 12 sig/day x 4d hold ~ 3x any sane book; taken-subset selection dominates |

## Two real discoveries from the failures

1. **OR-width is a quality signal**: equal-risk sizing (1/stop-width) INVERTS
   the edge — narrow-OR breaks lose, wide-OR breaks win. (Notional-weighted
   mean negative in 2015 while equal-weighted +14.5bps.) Candidate filter for
   a future pre-registered experiment.
2. **Within-day selection is the whole game at breadth**: the F2/G3
   pre-registered gap dose-response (bigger gap = bigger edge, monotone)
   is a VALIDATED ranking available for intake selection — untested as such.

## Honest position

- Trade-level edge: REAL (strongest evidence in the study).
- Index-only implementation: validated, thin (~40 tr/yr).
- 10-name G4 book: Sharpe 1.0 / DD -17% — best implementable; capacity-free.
- 77-name breadth book: NOT yet implementable — needs a validated intake
  ranking (gap-size) — one construction remains before OOS, else declare
  capacity-constrained.
