# research/64 — Calm-Day Entry Screen for Neutral / Iron-Fly Selling on NIFTY

**STATUS: PHASE 1 DONE (univariate)** · **VERDICT: SIGNAL — "calm" is strongly predictable from a single
family (volatility / range COMPRESSION). Trend, momentum, MA, Ichimoku, ADX, inside-candles add ~nothing.**
Not yet a net-₹ STRATEGY (premium tradeoff unmodeled — see caveat).

## The ask
Find the sweet-spot entry conditions for selling a neutral/iron fly on NIFTY — a "calm days" pattern.
Assess *all* indicators/combinations comprehensively (CPR multi-TF, prior day/week breaks, Bollinger,
HTF, MAs, Ichimoku, other bands, trend-then-halt, inside candles D/W, inverses, RSI, Stoch, ADX, ATR…).

## Critical framing (what "backtest" means here)
No historical NIFTH option premiums exist in the DB (recorder starts 2026-04-20; Kite intraday ~60d).
So true fly P&L over years is not computable. But a fly's edge IS *staying calm*, so the question is
fully answerable from price action: **outcome = the 2% move-stop is NOT hit within the next H trading
days (calm window = the fly wins).** Model-free, no look-ahead (features use data ≤ prior close).
calm-rate = the fly's win-rate proxy; each "stopped" window ≈ a ~₹34k loser.

## Method
NIFTY daily + India VIX from Kite, 2015-01-01→2026-06-12 (2,828 entry days). ~24 causal features across
every family requested. Outcome calm_H for H∈{3,5,8}. Univariate quintile screen on calm_5 with 3-era
stability check. `research/64_calm_day_ironfly_entry/scripts/calm_study.py`, ranking in `results/`.

## Base rates
calm (no 2% stop): **H3 = 77.3% · H5 = 59.4% · H8 = 39.4%.** (Headline = H5, ~1-week hold.)

## Findings — the predictors that matter (top-quintile vs bottom-quintile calm_5, all 3/3 eras consistent)
| Feature | calm in best quintile | worst | spread | calm when |
|---|---|---|---|---|
| **India VIX** | 81.0% | 32.4% | **0.486** | LOW |
| **ATR(14)/price** | 78.8% | 32.7% | **0.461** | LOW |
| **realized vol 20d** | 77.2% | 39.7% | 0.375 | LOW |
| realized vol 10d | 74.5% | 40.4% | 0.340 | LOW |
| **Donchian-20 width** | 71.5% | 39.9% | 0.317 | NARROW |
| range last 5d | 70.8% | 42.8% | 0.280 | NARROW |
| Stochastic %K | 73.0% | 46.8% | 0.262 | HIGH |
| **Bollinger width (squeeze)** | 68.7% | 44.7% | 0.240 | NARROW |
| **prior-day CPR width** | 66.3% | 42.6% | 0.237 | NARROW |
| prior-day gap | 64.7% | 44.9% | 0.198 | SMALL |

**One clean theme: volatility / range COMPRESSION predicts calm.** Low VIX, low ATR, low realized vol,
narrow Donchian/Bollinger/CPR/5-day range, small prior gap — these cluster and dominate. Vol clusters
(calm begets calm). Best single quintile reaches **~80% calm vs 59% base.**

## What DIED — stop testing these for calm-prediction
Near-zero spread / inconsistent across eras: **ADX (0.002!), Ichimoku cloud thickness (0.022),
MA20-50 compression (0.005), 200-DMA distance, 5-day momentum, RSI-distance, weekly CPR, MA20 slope,
20/50-DMA distance.** Binary: **inside-day 60.6% vs 59.3%** and **inside-week 61.0% vs 59.2%** — both
negligible (this challenges the inside-week filter currently in the live engine — it barely beats base
and is far weaker than any vol-compression feature). Day-of-week: Friday entries slightly calmer (62%
vs ~58%); otherwise flat.

## The decisive caveat — calm-rate ≠ net P&L
This screen ranks the **win-rate** axis only. **Low VIX = calmer but thinner premium**; high VIX = richer
premium but more stops. The net-₹ sweet spot is a *tradeoff* — you want calm AND enough credit — which
this price-only study cannot resolve. That's why the live strategy uses a VIX *floor* (≥13, for premium
richness), even though calm-rate keeps rising as VIX falls. Resolving the true optimum needs the option
premium (AlgoTest or the recorder forward), layering credit-collected × calm-rate − stop-cost.

## Next phases
- **P2 — combinations & composite:** AND-stacks of the survivors (e.g. low-ATR ∧ narrow-BB ∧ small-gap),
  a single "compression score", multiple-testing control, per-year P&L-proxy lift.
- **P3 — premium-aware net edge:** map calm-rate buckets to credit (AlgoTest / forward recorder) to find
  the net-₹ optimum (the real "sweet spot"), not just the calmest.
- **P4 — wire the winning compression-gate into the live V2 entry filter** (replacing/augmenting the weak
  inside-week leg), forward-validate via the shadow log.
