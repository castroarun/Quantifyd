# research/137 — Stock-Options Condor Bias · G1 RESULTS

## VERDICT: **NO USABLE BIAS.** The tilt has no basis. Concluded 2026-08-31.

Fourteen trend signals — SuperTrend at five parameter settings, weekly and daily,
plus EMA crosses, price-vs-SMA, RSI, MACD and Donchian position — tested on 78
F&O stocks over 216,099 stock-days. **None cuts the tail a sold spread cares
about.** The idea's directional half is dead; its symmetric half already exists
and is already validated.

---

## 1. What was measured, and why

A condor does not need to be right about direction often. It needs the spread it
sold to expire out of the money. So the test was not "does the signal predict
returns" but:

> Given a bullish signal, is P(fall ≥ 5% over 21 days) meaningfully lower than
> unconditional — and does the bearish mirror cut P(rise ≥ 5%)?

**Bar set before running:** −3.0 percentage points or better, on both sides.

Unconditional baseline over 216,099 stock-days: mean +1.79%, **P(fall≥5%) 20.5%**,
P(fall≥10%) 8.0%, **P(rise≥5%) 31.5%**.

## 2. Every signal, ranked by the number that matters

| signal | n bullish | mean vs uncond. | t | **Δ P(fall≥5%)** | bear **Δ P(rise≥5%)** |
|---|---:|---:|---:|---:|---:|
| close > SMA200 | 129,413 | −0.26% | −2.7 | **−1.0** | +1.2 |
| EMA 50/200 | 144,036 | −0.40% | −4.1 | −0.5 | +2.3 |
| MACD hist > 0 | 109,137 | −0.06% | −0.6 | −0.3 | −0.4 |
| weekly ST(10,3) | 134,079 | −0.40% | −4.1 | −0.2 | +2.2 |
| Donchian 55 | 79,926 | −0.36% | −3.7 | −0.1 | +2.8 |
| ST(21,5) | — | — | — | −0.1 | +1.6 |
| RSI(14) > 50 | 118,490 | −0.28% | −2.9 | +0.0 | +0.7 |
| weekly ST(7,3) | 136,465 | −0.46% | −4.7 | +0.0 | +2.4 |
| EMA 20/50 | 129,528 | −0.43% | −4.4 | +0.2 | +2.1 |
| ST(10,2) | 116,904 | −0.24% | −2.4 | +0.2 | +0.3 |
| close > SMA50 | 122,155 | −0.38% | −3.9 | +0.3 | +1.4 |
| ST(7,3) | 120,992 | −0.29% | −2.4 | +0.3 | +1.2 |
| ST(14,3) | 120,183 | −0.34% | −3.5 | +0.4 | +1.2 |
| ST(10,3) | 120,468 | −0.34% | −3.5 | +0.4 | +1.2 |

**0 of 14 pass.** The best single number is −1.0 points, a third of the bar, and
that same signal's bearish mirror moves the *wrong way* by +1.2.

## 3. Three findings, all of them negative

**The tail does not move.** Bullish states leave P(fall≥5%) between −1.0 and +0.4
points of a 20.5% baseline. A condor tilted on any of these is a condor tilted on
nothing.

**Bearish states rise MORE often.** Eleven of fourteen show a *positive* Δ on
P(rise≥5%), up to +2.8. Selling call spreads on "bearish" names would be selling
into a higher-than-average chance of exactly the move that hurts. This reproduces
research/129 precisely, which found bear states carrying higher forward drift.

**Bullish states underperform.** Every signal's mean 21-day return is *below*
unconditional, ten of them at t < −2.4, the worst at −4.7. Trend-following states
in this universe have negative forward alpha at a monthly horizon — consistent
with research/136, where 2–10 day breakout entries lost to a random pick at
t = −7.2.

## 4. Where this leaves the idea

**The tilt is dead.** SuperTrend was not the problem — five parameter settings,
weekly and daily, plus nine other constructions all say the same thing. This is
the **fifth and sixth independent kill** of the trend-state family across
research/129, /136 and now /137. It should not be re-litigated without a
genuinely new kind of signal — order flow, options positioning, fundamentals —
not another moving average.

**The symmetric version is not dead — and it already exists.** A condor with no
directional tilt does not need a cue at all; it monetises the volatility risk
premium. That is exactly **research/127**, a STRATEGY candidate: universal
±2.5% strangle with 7% wings on the same F&O universe, +0.264% of spot per trade,
t = 5.06 on n = 628, correlation to NIFTY −0.09, already running as a paper book
at `/app/stock-wings`.

And **research/130** tested adding a directional skew to a symmetric book and
found the tilt *strictly dominated* by simply sizing the symmetric book up.

So the constructive answer to "call condors on the bullish, put condors on the
bearish" is: **drop the bias, keep the structure, and it is the book you already
have.** The open work on it is the margin check, not a new signal.

## 5. Guarded

| sin | control |
|---|---|
| look-ahead | forward window starts the day *after* the signal; weekly states forward-filled so a week is known only once closed |
| multiple testing | 14 signals, all reported, ranked, none selected on; the bar was written before the run |
| survivorship | today's F&O list — absolute levels are upper bounds; the comparison is against the same pool's unconditional, which absorbs most of it |
| cost neglect | not reached — the signal failed before costs could matter |
| regime | 11.5 years spanning 2018-19, the COVID crash and the 2021-24 bull run |

## 6. Not tested, deliberately

Option-side realism — real NSE EOD chains, bid-ask, liquidity by strike — was
scheduled for G2 and never needed. The idea failed on the underlying.
