# Daily Anomaly Scan — where the edge lives (378 liquid names, 2000–2026)

**Verdict: short-term reversal is the standout — perfectly monotonic deciles,
t ≈ −9.78, and naturally market-neutral (long-short), which structurally fixes the
correlated-drawdown problem that made the breakout systems uninvestible. Build
this. Turn-of-month is a strong, simple timing overlay. Overnight drift is huge
but hard to monetize cleanly. Momentum/low-vol/52wk-high add little new.**

Cross-sectional probes: rank each day on a causal signal, decile forward returns,
dollar-neutral long-short spread + monthly t-stat. Gross of costs.

| # | Anomaly | Decile shape | LS spread | t | Read |
|---|---|---|---|--:|---|
| **1** | **Short-term reversal** (5d) | clean monotonic ↓ | **−0.59%/5d** | **−9.78** | **STRONG, market-neutral, build it** |
| 2 | Momentum 12-1 (21d) | monotonic ↑ | +0.72%/21d | 1.56 | real but modest; MQ already harvests |
| 3 | Overnight vs intraday | — | +56.9%/yr vs −32.5%/yr | — | huge but hard to trade cleanly |
| 4 | Turn-of-month | — | +0.28%/day on ToM days | — | strong simple overlay |
| 5 | Low-volatility (60d) | ↑ (inverted!) | +0.93%/21d | 2.30 | high-vol *out*performs = just beta, not a clean edge |
| 6 | 52wk-high proximity | flat | −0.21%/21d | −0.81 | no edge (slight reversal hint at D0) |

## Detail on the winner — short-term reversal

Past-5d return decile → next-5d forward return:

```
decile:  D0    D1    D2    D3    D4    D5    D6    D7    D8    D9
fwd-5d: +0.69 +0.61 +0.56 +0.51 +0.48 +0.46 +0.40 +0.34 +0.28 +0.10
```

- **Monotonic across all 10 deciles** — the cleanest possible shape. The biggest
  5-day losers (D0) earn +0.69% over the next 5 days; the biggest winners (D9) earn
  +0.10%. Long D0 / short D9 = **+0.59% per 5 days, gross**, t ≈ −9.78 (i.e.
  ~10 standard errors — overwhelming).
- **Naturally market-neutral** (long losers, short winners, dollar-balanced) → no
  single-factor beta exposure, so it should NOT have the −37–87% drawdowns that
  sank research/44. This is the structural fix.
- **The honest catch is turnover/cost.** A 5-day full decile rotation trades a lot;
  the +0.59%/5d gross spread must beat ~2 legs of round-trip cost. The build must
  test holding period (5/10/21d), quantile width (decile vs quintile vs tertile),
  rank-weighting (smoother = less turnover), and a realistic cost model — that's
  exactly where the next phase earns its keep.

## Other notable reads

- **Overnight drift (#3):** +0.226%/day overnight vs −0.129%/day intraday — i.e.
  essentially *all* the equity return accrues overnight; intraday is net negative.
  A real, globally-documented effect, but monetizing it needs trading at/near
  open+close daily across many names → cost/impact likely eats it. Better used as a
  *bias* (enter longs near close) than a standalone book.
- **Turn-of-month (#4):** +0.306%/day on days ≤3 / ≥28 vs +0.027% the rest — a 10×
  difference. Robust, low-turnover (a few days/month). Good standalone overlay or a
  filter to time the reversal/long book.
- **Low-vol inverted (#5):** in this liquid F&O universe high-vol won (t 2.30) —
  that's just the high-beta/momentum bull-market tilt we already saw in research/44,
  not a risk-adjusted edge.

## Recommendation

Build **research/46: short-term mean-reversion, long-short, market-neutral** — the
one edge here that is both statistically overwhelming AND structurally suited to a
robust (low-DD, high-Sharpe) portfolio. Layer the **turn-of-month** timing as a
free overlay. This is the opposite of the three systems we shelved: diversified,
market-neutral, and mean-reverting rather than directional-breakout.
