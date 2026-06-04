# MTF Compression Breakout — Results (G1–G4)

**VERDICT: NO ALPHA (beta).** Across **four independent tests** — large/midcap 5-min,
smallcap 5-min, and daily full-universe (1,099 names, 2018–2026, n=7,501 SIGNAL) —
the multi-timeframe compression+volume breakout entry **does not beat simply holding
an uptrending stock.** It makes money, but *less* than the beta-baseline on every
trend-riding exit. The **volume spike consistently fails to help (often hurts)** in
all runs. The only positive is a faint, fragile **+0.04R on tight profit targets**.
The pattern looks unbeatable on the user's example runners (TDPOWERSYS/DATAPATTNS/
KMEW) purely by **survivorship** — across the population the edge isn't there.

## Falsification bar (set before results)
SIGNAL must beat its own TREND_BASELINE (enter any uptrend day, same exits) by
≥ +0.10R with PF uplift, stable across years. **FAILED** on every trailing/stop exit;
only marginal (+0.04R) on R-targets. → NO ALPHA.

## Evidence

### G4 — Daily full universe (1,099 names, 2018–2026, n=7,501 SIGNAL) — the decisive test
Alpha delta (SIGNAL − BASELINE), net@6bp R:

| Exit | SIGNAL | BASELINE | Δ |
|---|--:|--:|--:|
| SUPERTREND_D | +0.325 | +0.932 | **−0.607** |
| HARD_SL | +0.274 | +0.340 | −0.066 |
| CHANDELIER | +0.029 | +0.220 | −0.192 |
| MAXHOLD_20 | +0.076 | +0.105 | −0.029 |
| R_2R | +0.138 | +0.096 | +0.043 |
| R_3R | +0.185 | +0.148 | +0.037 |

Per-year (SUPERTREND, SIGNAL vs BASELINE): baseline wins **every year except 2022**
(SIGNAL +0.096 vs BASE −0.326) — i.e. the compression filter only adds value in the
bear/correction year, as a *defensive* selector, never as an alpha entry in bulls.

### G1–G3 (consistent)
- **G1 (12 names):** looked good (+0.34 alpha) — but n=157, small-sample luck.
- **G2 (218 large/midcap, n=1,424):** alpha evaporated → **−0.01 to −0.13** (beta).
- **G3 (160 smallcap 5-min, 2024–26, n=631):** SIGNAL **underperforms** hold
  (Supertrend +0.26 vs base +0.40); volume spike worse; amplifies corrections.

## Volume spike — refuted (the user's core hypothesis)
VOL_SPIKE (≥3×) ≤ SIGNAL (≥1.5×) in **all four runs**. The visible spike on winners is
survivorship; requiring it removes more good trades than bad. Volume adds no edge here.

## Seven sins
- **Survivorship:** the headline failure mode — examples are remembered winners; the
  population test removes the bias and the edge with it. ✓ controlled (full universe).
- **Look-ahead:** all features causal (SMA/CPR/NR as of D-1; entry next-day open / next
  15m bar). ✓
- **Cost:** net@6bp + 0/12 sensitivity throughout. ✓
- **Regime/coverage:** 5-min smallcap data only 2024+ (flagged); daily run spans
  2018–26 incl. 2018/2020/2022 stress. ✓
- **Multiple testing:** 6 exits × 5 arms scanned; the one +0.04R survivor is within noise.

## What IS supported by the data (forward paths)
1. **The baseline itself is the strategy.** "Own uptrending names, trail on daily
   Supertrend" (+0.93R, PF 1.58) crushes the breakout timing. That is a **momentum/
   trend book** — which the existing **MQ book already is (32–48% CAGR)**. The highest-EV
   move is to improve MQ, not to bolt a breakout-entry overlay on it.
2. **Compression as a DEFENSIVE regime filter, not an entry.** 2022 is the only year the
   filter beat baseline — it keeps you out of weak breakouts in bad tape. Worth testing
   as a risk-off gate on a momentum book, never as the trigger.
3. **Tight R-targets** are the only entry-edge sliver (+0.04R) — too small to pursue alone.

## Honest caveats
- Per-signal expectancy (event study, overlapping) — a population read, not a portfolio
  curve. But the comparison is apples-to-apples (same construction for SIGNAL & BASELINE),
  so the alpha conclusion holds regardless.
- Smallcap 5-min limited to 2024–26 by data. Daily run is the multi-regime authority.

**Status: CONCLUDED — NO ALPHA. Shelve as a standalone system; revisit the compression
filter only as a defensive overlay on the momentum book.**
