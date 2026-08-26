# Credit-Spread Skew Overlay on the C1 Stock Strangle — research/130

STATUS: RUNNING — G1 launched 2026-08-26 night

## 1. Headline

On top of each r/127 C1 winged strangle, ADD a directional credit spread
(extra bull put spread, or bear call spread) — always, or directed by
MA/RSI/stochastic state at entry. Does the addition pay on real options data,
and do the indicator inputs improve the direction choice?

## 2. The Ask

**What you asked (Arun, 2026-08-26, clarified):** the credit spreads were meant
IN ADDITION to the strangle book, with the RSI / stochastics / MA references as
the inputs to study for the skew decision — not standalone regime-gated spreads.

**What we're testing:** for each of the 628 liquid C1 trades (same entries,
same exit dates), price from real bhav the EXTRA unit:
- PS = bull put spread: sell 1× PE@−2.5% + buy 1× PE@−7% (same strikes as C1's
  put side, one more unit)
- CS = bear call spread: sell 1× CE@+2.5% + buy 1× CE@+7%
Variants: ALWAYS_PS, ALWAYS_CS, and gated: PS when bull-state / CS when
bear-state for state ∈ {SMA200, EMA20>50, RSI>50, STOCH K>D} (the inputs you
named). Metric: overlay net %S0 per trade (t), and C1+overlay total vs C1 —
net, tail p05/p01. Margin note: the extra short leg adds margin; overlay must
clear its share.

**G0:** stocks drifted +1.92% per 24 sessions unconditionally (r/129 probe) —
an always-on put-side tilt harvests drift + put skew; the r/129 result predicts
the regime gates add nothing (bear states drift HIGHER), so ALWAYS_PS is the
live hypothesis and the gates are the controls. Falsification: if ALWAYS_PS
overlay net ≤ 0 after costs, or worsens the C1 tail beyond its earnings,
CONCLUDED — keep the symmetric book.

## 3. Base

Same data/costs conventions as r/127 (bhav EOD, 0.5%/side, liquidity gate
inherited from the C1 rows; overlay legs priced at the same entry/exit dates
as the parent trade — no separate exit logic in G1). Indicator states computed
causally from daily closes at entry.

## 4. Plan

One runner over the 628 C1 trades × {PS, CS} legs + 4 states ≈ minutes.
G2 (only if ALWAYS_PS passes): skewed-strike strangle variants (PE −2%/CE
+3.5% etc.), sizing of the overlay, margin measurement.

## 5-8. Log / recovery / files / findings

| When | Event |
|---|---|
| 2026-08-26 night | STATUS locked, G1 runner launched |

Runner `scripts/run_g1_overlay.py` → `results/g1_overlay.csv` + analyzer
`scripts/analyze_g1.py`; resume-safe per symbol; VPS canonical.

## VERDICT (2026-08-26): SIGNAL, not investable — dominated by C1 sizing. STATUS: DONE
