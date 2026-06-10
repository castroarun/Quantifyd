# V2 Iron Fly — Causal-Feature Screen — RESULTS

**VERDICT: SIGNAL (one factor, two flags) — CANDIDATE overlay, forward-paper before live.**

Screened ~25 causal technical features (known at 09:20 entry) against the full-hold P&L of the
204 VIX≥13 baseline trades (2% wings + 2% move-stop, ex-COVID, 2019-02→2026-05). Discipline:
univariate quartiles → require monotonic dose-response + per-year consistency + mechanism →
walk-forward (threshold picked on train half, applied blind to test half) → incrementality test.

## The one thing that matters: volatility COMPRESSION at entry

A short iron fly is a pure short-gamma / short-vol bet — it is **indifferent to direction and
trend**, and only cares whether a big move is coming. The screen confirms exactly that: every
feature that separates losing weeks from winners is a **compression proxy**, and everything
about trend / momentum / location is noise.

### Survivors (negative in BOTH walk-forward eras, mechanism-aligned)
| Flag (skip the entry when…) | What it measures | Univariate (VIX≥13) | Walk-forward |
|---|---|---|---|
| **Narrow prior-day daily CPR (<~0.10%)** | next-day compression | Q1 −₹3.5k/trade, neg 6/7 yrs | 0.52→1.22 & 0.59→0.92 ✓ |
| **Inside-week (last complete week inside the week before)** | weekly coil | −₹5.7k/trade, neg in both eras (−62k 19-22, −40k 23-26) | n/a (binary); neg both halves ✓ |

### Stacking them (independent — only 6 of 18 inside-weeks overlap the CPR skip)
| Entry filter on VIX≥13 base | Trades | Net P&L | Calmar | MaxDD | Green |
|---|---|---|---|---|---|
| Base (no skip) | 204 | +₹8.80L | 1.03 | −₹1.17L | 7/8 |
| skip narrow daily CPR <0.10% | 147 | +₹11.00L | 1.59 | −₹0.95L | 7/8 |
| skip inside-week | 186 | +₹9.83L | 1.15 | −₹1.17L | 7/8 |
| **skip CPR<0.10% OR inside-week** | **135** | **+₹11.45L** | **2.00** | **−₹0.78L** | **7/8** |

Incrementality proof: among CPR-survivors, inside-weeks still lose −₹44.6k (Calmar −0.08, neg
2019/2024/2025). The combined skip flags **37 of 89 negative weeks (−₹6.28L of −₹12.29L of red)**
while raising total return AND cutting DD — strictly better, not a return-for-risk trade.

## What DIED (clean negative results — these hypotheses are NOT useful for this system)
- **RSI — daily, weekly, monthly:** no monotonic signal, corr ≈ 0. Momentum is irrelevant to a
  short-vol book.
- **Moving averages** (close vs 20/50/200 DMA, 50-DMA slope, weekly 20-WMA): no usable signal.
- **Ichimoku** (price vs cloud, cloud thickness; daily & weekly): no separation.
- **Monthly pivots / monthly CPR:** non-monotonic noise (one spike bucket, no dose-response).
- **Prior-week range breaks** (open above PWH / below PWL): no signal.
- **Daily** inside-candle: no signal (only the **weekly** inside-candle matters).
- **Bollinger band-width / %B:** looked monotonic univariate but **failed walk-forward** —
  redundant with CPR and the threshold does not transfer. Dropped.

## Honest caveats (the seven sins)
- **Multiple testing:** ~25 features screened. CPR was pre-registered (prior finding); inside-week
  is NEW and earned it on mechanism + both-era negativity + incrementality, but **n=18 is thin**
  (13 in 2019-22, 5 in 2023-26) — one era is only 5 trades. Treat as STRONG CANDIDATE, not locked.
- **In-sample, one instrument/engine** (NIFTY, AlgoTest). Threshold ≈ "skip bottom-quartile
  compression", not a precise constant.
- **Convergence, not new factors:** the result is that there is essentially ONE conditioning axis
  (entry-time compression), best read two independent ways. Stacking more compression proxies
  (BBW, weekly-wide-CPR) does NOT add and risks overfit.

## Case A — late-entry simulation (BUILT, RAN, FAILED CALIBRATION — inconclusive)
User idea: on a filtered-out cycle, re-check the next trading day and enter if the combo filter
then clears (CPR not narrow AND not inside-week). Built a synthetic iron-fly engine (BS priced off
the daily India-VIX path, 2% move-stop + PT40, AlgoTest-style 4-DTE entry / 1-DTE roll). **It does
NOT pass calibration:** corr(synthetic, AlgoTest) = 0.25, sign 63%, and it prices the SKIPPED
(compression) cycles as +₹895k profit when AlgoTest books −₹265k loss — i.e. it gets the SIGN
backwards on exactly the cycles being studied. Root cause: BS-with-VIX (a 30-day calm measure) is
blind to the compression→expansion tail that makes those cycles lose; daily extremes also can't
model the intraday 2% stop. **The late-vs-scheduled / late-vs-skip deltas are therefore untrustworthy
and NOT reported as findings.** The only real-chain source over 2019-26 is AlgoTest (options_data.db
recorder only has ~8 weeks since 2026-04). **To settle Case A: run the conditional-late-entry variant
on AlgoTest.** Absent that, the daily-bars probe read stands: compression clears 70% by +1TD but the
move splits ~50/50 around the clear and DTE is lost → expected benefit marginal, **skip > delay.**

## Case B — inside-week breakout (G0 PASSED, bullish only — candidate for a G1 options backtest)
Hypothesis: instead of only skipping inside weeks, trade the expansion with a defined-risk directional
spread. G0 underlying probe (493 weeks, 46 inside weeks, daily bars): after an inside week, the FIRST
daily close that breaks the inside week's high → **78% continuation to week-end vs 55% baseline**, median
MFE 1.77% vs 1.06%; **70% reach +1.0% / 57% reach +1.5%** within 5 TD (enough to pay a call debit spread).
The DOWN side does NOT continue (43%, whipsaw) → **bullish-only, asymmetric.** n=23 up-breaks → G0
hypothesis, not proof.

**Bear-side rescue attempt (RSI/CPR/early-break filters) — FAILED.** Tested RSI≤35/≤30, mid-CPR,
day-1 (early) break on the CREDIBLE 156-down-break sample (base continuation 41%): NONE beat base
coherently. RSI≤35 makes it WORSE (36%) — oversold down-breaks BOUNCE. Mechanism: NSE structural UP
drift → down-breaks fight the tide and mean-revert; oversold = bigger bounce. Structural, timeframe-
independent → no daily filter flips it. Inside-week down subset (n=14) swings on 3-7-event cells that
contradict the big sample = noise. **Bear side stays dead; Case B is bullish-only.** (True intraday
first-30m/1h-candle break untested — needs 60min data; structural prior says it won't rescue it.) **Next: G1 real-chain (AlgoTest) call-debit-spread backtest on the up-break
trigger, net of cost.** Clean symmetry: V2 SKIPS inside weeks; Case B MONETIZES the up-break on them.

## Recommendation
1. Keep the **locked live base = VIX≥13 + 2% wings + 2% move-stop** unchanged.
2. Add **inside-week** alongside daily-CPR in the executor's **shadow/log-only** overlay set, so
   the forward-paper ledger validates BOTH compression flags on real-time data before either gates
   real money.
3. No other indicator earns a place. Stop testing trend/momentum/location features for this book.
