# MA/EMA-Regime Directional Credit Spreads on F&O Stocks — research/129

STATUS: RUNNING — G1 probe launched 2026-08-26 night

## 1. Headline

When a stock closes above its 100/150/200 MA (or a fast/slow EMA cross is
bullish), sell BULL PUT credit spreads; below/bearish, sell BEAR CALL spreads —
with/without SLs, or hold until the reverse cross. Does the regime signal carry
enough forward information to monetize through option credit spreads?

## 2. The Ask

**What you asked (Arun, 2026-08-26):** "When the stock crosses and closes a day
above 100/150/200 MA/EMA/DMA, go skewed — short bull credit spreads — and vice
versa; with and without SLs, or until the reverse cross. Or instead, 20-50 EMA
crossover or other EMA combinations. Roll out clear studies, assess all
possibilities."

**What we're testing (staged, so the grid can't data-mine us):**
- **G1 (price-only kill test, cheap):** for each regime signal — price vs
  SMA/EMA {100,150,200}, EMA crosses {20/50, 10/30, 50/100} — measure on the
  81-stock daily history what a credit spread actually needs: the
  **conditional tail probabilities** over the 24-session hold (the 45→21
  window): P(fwd return < −2.5%) in the bull state (what kills a bull put
  spread) and P(fwd > +2.5%) in the bear state (what kills a bear call
  spread), vs unconditional. Also mean forward return per state and t of the
  state difference. **Gate: a signal must shift the relevant tail prob by ≥20%
  relative (e.g. 30%→24%) with consistency across years, or the family dies
  here — spreads add cost to whatever edge exists, never create one.**
- **G2 (only for survivors):** options implementation on real bhav — spread
  strikes/width grid, TP/SL/reverse-cross exits, monthly cycle vs cross-event
  entries, net-of-cost.

**G0 hypothesis:** trend-state filters might tilt the drift enough that selling
the opposite-side tail is systematically overpaid. Counterparty: spread buyers
hedging/momentum-chasing. **Honest prior (heavy, stated loudly):** this family
has repeatedly FAILED in this repo — r/91 "20/200 SMA picture of power" = NO
EDGE (apparent profit was survivor drift; mechanics subtracted value vs random
entry); every technical indicator tested on MQ (EMA/RSI/SuperTrend/MACD/ADX)
reduced CAGR; r/56 dual-supertrend directional options = NO NET EDGE; r/72 RSI
regimes = SIGNAL that converged on existing momentum. The G1 gate exists to
kill this cheaply if the pattern repeats.

**Falsification:** no signal clears the tail-shift gate with per-year
consistency → verdict NO EDGE, family CONCLUDED, don't build options machinery.

**Multiple-testing control:** 12 signals × 2 sides recorded; gate requires
effect size + per-year stability, not just pooled significance; survivors get
a fresh-period split in G2.

## 3. The Base (G1)

- Universe: the same 81 F&O names (drift/survivorship noted: today's list).
- Data: market_data_unified daily, 2016-01→2026-08 (options era, matching any
  later G2); indicators causal (state known at close t, forward window t+1→t+24).
- Signals (state, not cross-day): close vs SMA100/150/200, close vs
  EMA100/150/200, EMA20>EMA50, EMA10>EMA30, EMA50>EMA100.
- Metrics per signal-state: n stock-days, mean fwd 24d return, t of
  bull-vs-bear difference, P(fwd<−2.5%), P(fwd>+2.5%), per-year table of the
  tail probs, plus the unconditional row.
- Overlapping windows: t-stats use a Newey-West-style downscale (÷√24) — noted.

## 4. Plan

G1: one pass, ~minutes. If gate passes for any signal → G2 design appended
here before any options run.

## 5. Status log

| When | Event |
|---|---|
| 2026-08-26 night | STATUS sections 1–4 locked; G1 probe launched on VPS |

## 6. Crash recovery

`research/129_ma_regime_credit_spreads/scripts/run_g1_probe.py` on VPS; output
`results/g1_probe.txt` (self-contained report). Re-run any time (read-only).

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| this STATUS | design + verdicts | yes |
| scripts/run_g1_probe.py | price-only kill test | yes |
| results/g1_probe.txt | G1 report | yes |

## 8. Findings

(pending)

## VERDICT (2026-08-26): NO EDGE — CONCLUDED at G1. See results/RESULTS.md.
STATUS: DONE
