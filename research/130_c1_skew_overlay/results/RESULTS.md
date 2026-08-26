# research/130 — Credit-Spread Skew Overlay on C1 (MA/RSI/stoch-directed) — RESULTS

**VERDICT: SIGNAL (put-side tilt real: ALWAYS_PS +0.153%/tr t=2.68; call side dead)
but NOT INVESTABLE as an overlay — strictly dominated by sizing up C1.**

On 602 C1 trades (real bhav, same windows): C1+ALWAYS_PS = +0.300%/tr but tail
DOUBLES (p05 -1.96%->-3.91%, p01 -3.09%->-6.73%) and t drops 5.06->3.62.
Running C1 at ~1.15x size gives the same +0.30% with a proportional (-2.2%) tail —
better on every axis, no extra legs/margin/indicators. Indicator direction
(stoch K>D best, +0.306 in bull-state, t 4.0) improves the overlay but is
multiple-testing-inflated (4 gates tried) and 2024/2026 are already NEGATIVE
for the overlay (decay flag). Bear call spreads: dead everywhere (drift).

Honest caveats: overlay exits forced to the parent trade dates (no independent
TP); 26 trades unpriceable (missing spread legs); margin of the extra short leg
unmodeled (worsens the overlay further). Next levers: none — the drift finding
is already better expressed through C1 sizing. Skewed-STRIKE strangles (PE
closer/CE further) remain untested but face the same dominance hurdle.

Files: scripts/run_g1_overlay.py, results/g1_analysis.txt.
