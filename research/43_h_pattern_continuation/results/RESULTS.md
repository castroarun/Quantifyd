# H-Pattern Continuation — Results (Phase 1, pure pattern, no filters)

**Verdict: the raw H continuation is a near-breakeven base with a real but thin,
short-biased gross edge that today's intraday costs erase. Best config =
breakout entry + let winners run (measured-move / 3R) + short side. That's the
configuration to carry into Phase 2, where context filters can lift the gross
edge enough to clear costs.**

Universe: 375 liquid names + NIFTY50 (separate cohort), 5-min, 2018→2026,
intraday only, square-off 15:25. 6.39M simulated stock trades. Outcomes in
R-multiples (R = entry→stop distance). Gross = pre-cost; Net = after 6 bps
round-trip. Detection thresholds fixed (see STATUS doc §2).

## Stocks cohort — top variants (sorted by net expectancy)

| Variant | Trades | WR% | Gross R | Net R | Cost R | PF | Long net | Short net |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **breakout\|MM** | 348k | 37.3 | **+0.163** | **−0.035** | 0.197 | 0.95 | −0.058 | **−0.016** |
| breakout\|3R | 348k | 36.8 | +0.158 | −0.040 | 0.197 | 0.94 | −0.063 | −0.022 |
| breakout\|2R | 348k | 40.6 | +0.143 | −0.055 | 0.197 | 0.91 | −0.083 | −0.033 |
| breakout\|1R | 348k | 53.4 | +0.099 | −0.098 | 0.197 | 0.81 | −0.126 | −0.076 |
| retest\|MM | 292k | 34.1 | +0.041 | −0.154 | 0.195 | 0.78 | −0.196 | −0.121 |
| fade\|MM | 637k | 23.9 | +0.165 | −0.246 | 0.411 | 0.77 | −0.325 | −0.178 |
| fade\|1R | 637k | 47.3 | +0.014 | −0.397 | 0.411 | 0.43 | −0.443 | −0.357 |

(Full 15-row table in `ranking_stocks.csv`; NIFTV50 in `ranking_nifty50.csv`.)

## What the data says

1. **Breakout wins on entry style — decisively.** Highest net everywhere, and
   the *lowest cost drag* (0.197R) because the breakout stop is the full flag
   band — a wider, more cost-efficient R. Fade's tight stop gives a flashy
   paper R:R but **2× the cost in R (0.411)**, so it ends up worst on net
   despite an equal *gross* edge. Retest fills less often and the continuation
   after a pullback is weaker (gross barely +0.04R).

2. **Let winners run.** Measured-move and 3R beat 2R beat 1R on net for every
   entry style. TRAIL is poor — the ATR ratchet chops out in intraday noise.

3. **The H leans bearish.** Shorts beat longs in *every* variant
   (breakout|MM: short −0.016R vs long −0.058R). Consistent with the user's
   bearish example. A **short-only breakout|MM** is the single closest-to-edge
   configuration: gross-positive, net ≈ −0.016R.

4. **No variant survives costs net-positive — but breakout|MM is close**
   (net −0.035R, PF 0.95). The pattern carries a genuine ~+0.16R gross edge;
   ~6 bps round-trip on a ~30 bps stop costs ~0.20R, which is just larger than
   the edge. A modest filter lift (raise WR from 37%→~41%, or fatten winners)
   flips it positive.

5. **NIFTY50 alone is not tradable this way.** Index 5-min flag bands are a
   tiny % of price, so 6 bps becomes ~0.5R of cost — gross stays +0.08R but net
   is deeply negative. The index needs either a futures-cost model (~1 bp) or to
   be traded only as a regime filter, not as the instrument.

## Honesty / robustness caveats

- **All three entry styles are conditioned on a clean flag having armed at bar
  `f`.** Breakout is the most live-realistic (its trigger is causal post-arm);
  fade/retest inherit mild hindsight (we only act on bounces that became clean
  flags). This *favours* fade/retest slightly — yet they still lose to breakout,
  which strengthens the breakout conclusion.
- **Single fixed detection threshold set.** POLE_ATR=2.5, RETR_MAX=0.7,
  FLAG_MAX=12, WIDTH=1.5 ATR. Not swept. A different threshold set could shift
  trade counts and edge — that's a Phase-2 sensitivity sweep, only worth running
  if a filtered variant clears costs.
- **Cost is a flat 6 bps round-trip of notional.** Fair for liquid cash-equity
  intraday; pessimistic for index futures. The gross column isolates the raw
  edge from this assumption.
- **No overnight / multi-day H.** Pattern + trade are intraday by construction;
  a swing-timeframe H (daily bars) is untested and could behave differently.
- **No slippage on stop fills beyond the flat bps** (stops assumed filled at the
  level; gaps through the stop not modelled — mildly optimistic on tail losses).

## Recommended next step (Phase 2)

Carry forward **short-biased breakout|MM** and add the confluence filters the
user flagged, as gating axes:
1. **Prev-day low break** alignment (continuation only when the H breaks with the
   prior-day level).
2. **CPR** width/position regime + pivot confluence at the crossbar.
3. **Daily-trend agreement** (price vs 50/200 DMA) — only "with-trend" Hs.
4. **2-min entry refinement** — *blocked*: no 2-min data in the DB; needs a Kite
   download (VPS-only per project rule).

Target: lift gross +0.16R → enough that net clears 0 with PF > 1.2 at ≥ a few
thousand filtered trades.
