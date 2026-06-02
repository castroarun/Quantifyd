# NAS Optimization — Disciplined Search for a Better / Uncorrelated System Set

STATUS: RUNNING

## The Ask
"With all price action, all timeframes, all indicators, VIX — try all combinations
and optimize; find a better working system or set of uncorrelated systems."

## Methodology guardrail (binding — QUANT_RESEARCH_PLAYBOOK)
**No brute-force grid on 28 days of option data** — that is multiple-testing/overfitting
(the #1 sin) and would surface false winners. Instead:
- Options layer (28 days): SMALL, pre-specified, economically-motivated variations only;
  treat results as SIGNAL/hypotheses, report the gradient (monotonic > peak), heavy caveats.
- Underlying layer (years): regime research with proper OOS — robust.
- Net-of-cost throughout. Every result labelled signal vs validated.

## Three parts
- **(a) Focused short-vol scan** (28-day chain, **DTE is a SEARCH AXIS — not fixed at 0/1**):
  axes = DTE-at-entry (0/1/2/3/4), strike (ATM / ~0.35Δ / ~0.25Δ OTM), SL mult (1.3/1.5/2.0),
  exit (time-1445 / EOD / ST7-2). Small pre-specified grid, ranked, monotonicity-checked
  (prefer monotonic-in-DTE over a single peak), multiple-testing caveat. Goal: which
  (DTE × strike × SL × exit) structure works best on real premiums.
- **(b) Diversification:** correlation of the 8 systems' daily P&L (research/51 replay, ALL
  eligible DTEs for max observations) -> uncorrelated, positive subset vs the all-8 book.
- **(c) Regime research (long history):** years of NIFTY daily (2000-26) + 5-min (2018+):
  predict intraday RANGE vs TREND days (proxy for short-vol favourability) from causal
  features (gap, opening range, prior-day range/ATR, weekday, IV proxy); OOS/per-year.

## Gates
G0 each part: cheap test, kill if no signal. Advance only what clears.

## Caveats (standing)
28 days of premiums = signal not validation. Regime proxy is underlying-only (no historical
option P&L). Forward validation needs the recorder to keep accumulating.

## Status log
| Time | Event |
|---|---|
| 2026-06-02 ~18:30 IST | folder + STATUS; starting (b) diversification |


## VERDICT (DONE)
4 findings (evidence-graded): (1) edge at 1 DTE [28d signal]; (2) tight opening range -> range day,
ROBUST over 6yr/1565d all regimes (corr 0.52, +every year); (3) stop = +/-0.4% underlying-move,
converges on 28d-real AND 2yr-stress, beats 1.3x-whipsaw and no-stop (2yr worst-day -58.8k); (4)
diversify across families, drop OTM. Recommended: 1-DTE ATM straddle, tight-open days only, cross-family,
+/-0.4% move stop. See results/RESULTS_COMBINED.md. STATUS: DONE.