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

## Phase 1c addendum (2026-07-21): explicit chart-pattern geometry — NO EDGE

Head-and-shoulders (+inverse), triangles/wedges (pivot-convergence), and
cup-with-handle (+inverted): 40 cells, daily, date-matched controls. **0
gate passers.** Notable inversion: LONG breakouts from bullish geometry
(inverse H&S, triangle/wedge upside breaks, cup-handle) show significantly
NEGATIVE relative returns (t -2..-3.9, only 30-39% of names positive) —
buying classical bullish pattern breaks selected UNDERPERFORMERS in
2005-2017. Recorded as an observation only (no sign-flip mining; shorts are
5x-buried and the relative expression costs double). Best cell: inverted
cup-handle short h5, t=2.37, n=174 — small-sample, fails gate.

## Phase 1d addendum (2026-07-21): remaining named patterns (investingoal list)

72 cells: ascending/descending triangles, triple top/bottom, island
reversal, dead-cat bounce, megaphone, diamond, channel bounce, parabolic
break. Date-matched harness. IS gate passers: 3, all one family —
**ATDT_asc_S: the FAILED ascending triangle** (flat >=2-touch resistance +
rising support; price breaks DOWN through the rising support) predicts
relative underperformance.

**Val 2018-2022H1: 2 of 3 pre-registered cells pass; flagship h15 holds its
effect size** — IS rel +79.5bps t=3.38 -> Val rel +84.6bps t=2.69, names+
0.63. First structure signal in this study to survive IS->Val without fade.
Caveats: RELATIVE-only (absolute short is flat/negative — the signal ranks
underperformers, it does not make money shorted naked); ~75/yr signals in
Val at h15; family found within a 350-cell program ledger. Label: **SIGNAL
(relative)** pending robustness (parameter neighbors, per-year, breadth) —
OOS untouched, requires explicit authorization.

Other notables: dead-cat-bounce LONG (buying 15% crashes) shows big absolute
nets (+96..+309bps) but t_rel <=1.6, names+ <0.5 — crash mean-reversion
variance, consistent with research/84's book-level failure. Everything else
dead.

## Phase 1d user-review corrections (2026-07-21 evening)

**TATASTEEL example invalid (user caught it):** detector accepted 'rising
lows' ABOVE the flat resistance (post-breakout structures, not triangles).
Corrected definition (lows strictly below resistance): IS weakens to rel
+24.3bps t=1.94 (below gate), Val +53.6 t=3.08. Window-inconsistent =>
family downgraded to FRAGILE SIGNAL, parked. Curious footnote: the excluded
'non-triangle' instances (support-failure after breakout) carried much of
the IS signal — a different pattern, would need fresh pre-registration.

**Entry mechanics (user: 'short should be AT the line'):** tested stop-order-
at-the-line vs close-confirmation on the corrected pattern set.
- Fills on true breaks improve ~50bps (user is right per-fill: collapse bars
  like SBIN 2014-07-11 fill at the line, not 14 points lower).
- BUT the same resting order fills on every intraday poke that closes back
  above (false break / shakeout): those cases are BULLISH — rel -184bps IS
  (n=1772, only 14% of names positive), -93bps Val.
- Net trader version: IS -11.3bps t=-1.1 (vs +24.3 close-confirm); Val
  +43.8 t=2.98 (vs +53.6). **Close-below confirmation IS the edge's filter**;
  the fill giveaway is the price paid to dodge shakeouts. Hybrid (line entry
  + same-bar close-back-above invalidation) noted as possible refinement,
  not pursued while the family is window-inconsistent.

## Structured-trade test of the user's spec (2026-07-21, late evening)

Entry stop-sell AT the support line, SL above flat resistance (pattern
invalidation), target = measured move (entry - triangle height), 30-session
backstop. Corrected geometry, 74 names.
- IS 2005-2017:  n=5051  net -78.9bps/trade  t=-7.05  win 51%  avgR -0.09
- Val 2018-22H1: n=2247  net -76.4bps/trade  t=-5.46  win 51%  avgR -0.06
Same ~-77bps in both eras. Decomposition: (a) line entry admits every
shakeout, and with the SL all the way above resistance those never
invalidate quickly - they chop to the time backstop; (b) ~11-day naked short
pays ~drift (~7bps/day * 11d ~= 77bps) as rent - matches the loss almost
exactly; (c) ~1:1 R:R at 51% win minus costs is negative expectancy. The
pattern's genuine content (~+50bps RELATIVE lag vs index in Val) cannot be
monetized by stock-price SL/target - only by hedging.
