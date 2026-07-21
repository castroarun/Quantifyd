# research/87 — Structure/Pattern Swing Screen: RESULTS

## Verdict: **NO EDGE (absolute); one borderline relative SIGNAL — not investable**

104 pre-registered cells: 8 families (pivot-level breaks, S/R bounces,
volatility-contraction breaks, flags, double bottoms/tops, weekly-daily MTF,
52w-high volume breaks, inside-bar/NR7) x horizons {3,5,10,15} x L/S, daily
bars, ~86 F&O names, IS 2005-2017, 10bps RT cost.

## The headline finding: the raw screen was a drift illusion
Raw IS results looked spectacular — 66 'passers', t up to 10.5, +100-150bps
at h15 — **entirely explained by unconditional drift + survivorship**.
Unconditional h15 forward return on this universe = +158bps; the best long
breakout cells earned ~the same. **Long structure breakouts on daily bars
add ~nothing over always-long exposure** (52w-high vol-breaks actually
UNDERPERFORM drift). Every long family died on the excess-over-drift gate.

## The relative short-side signal (and its fate)
Excess-over-drift left 16 short cells; date-matched (same-date universe mean)
control left 6 — dominated by CP1 SHORT (contraction-breakdown predicts
relative underperformance): IS relnet 20-49bps, t 2.5-2.9, names+ 62-68%.
**Val 2018-2022H1: collapsed.** Only CP1_p20_S_h5 cleared the Val gate
(relnet +23.5, t=2.30); the rest fell to t 0.9-1.5. 1-of-6 surviving with a
modest margin after selection = consistent with luck. Additionally the signal
is RELATIVE (hedged expression doubles costs, eating the margin).

## Method lessons (recorded for all future studies)
1. Fixed-horizon screens MUST be benchmarked vs unconditional drift AND
   date-matched universe mean — raw t-stats on drifting universes are fiction.
2. Data hygiene: zero-price rows in old daily data poison baselines (loader
   now filters open/high/low/close > 0).

Ledger: 104 cells + 2 rescoring passes + 1 Val pass. OOS untouched.
