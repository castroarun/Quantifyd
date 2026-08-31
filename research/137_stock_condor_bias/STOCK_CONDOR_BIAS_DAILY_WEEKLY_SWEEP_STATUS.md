# Stock-Options Condor Book — Does Any Trend Signal Earn a Directional Tilt?

STATUS: **CONCLUDED — no usable bias**
Opened 2026-08-31 · research/137

---

## 1. The Ask

**What you asked:** a stock-options portfolio system. Find liquid names with
liquid options, get *some* dependable cue for direction — SuperTrend was the
example, but vary the parameters and look at weekly too — then deploy call
condors on the bullish names and put condors on the bearish ones. Manage from
there: exit or roll on a signal flip, book at the condor midpoint, extend the
breakeven with a debit spread, all decided on closing prices because NSE EOD
options data is what we can depend on.

**What this first stage tests:** the cue. Not the condor, not the management —
the claim underneath all of it, which is that *some* simple trend indicator
separates stocks that will not fall hard from stocks that will.

## 2. Why this is the right first question

A condor does not need direction to be right often. It needs the **sold spread
to stay out of the money**. So the number that decides the whole idea is not the
mean forward return — it is:

> Given a bullish signal, is the probability of a large *fall* meaningfully lower
> than it is unconditionally?

If it is not, the tilt adds nothing, and every management rule downstream is
managing a coin flip.

**This family has been killed four times already.** research/129 tested MA, EMA,
RSI and stochastic regime states for credit spreads and found **no regime state
that shrinks the sold tail** — bear states even showed *higher* 24-day drift.
SuperTrend is a fifth member of that family. It gets the same test, and it has to
clear the same bar, or this dies here for the same reason the others did.

## 3. The Base

| | |
|---|---|
| Universe | 78 F&O stocks with ≥1,000 daily bars (of 81 in `FNO_LOT_SIZES`; COALINDIA, ONGC, ZOMATO short) |
| Period | 2015-01-01 → 2026-08-29 |
| Bars | daily, and weekly resampled to Friday closes |
| Horizon | 21 trading days — one monthly option cycle |
| Decision price | close only, matching an EOD workflow |

**Signals tested** — parameters varied deliberately, not one favourite:

| family | variants |
|---|---|
| SuperTrend | (7, 3) · (10, 3) · (14, 3) · (10, 2) · (21, 5), daily **and** weekly |
| EMA cross | 20/50 · 50/200 |
| Price vs SMA | close > SMA50 · close > SMA200 |
| RSI | RSI(14) > 50 |
| MACD | histogram > 0 |
| Donchian | close in the top quarter of its 55-day range |

## 4. What is measured, per signal

For bullish states, and mirrored for bearish:

1. **Mean 21-day forward return** vs the unconditional mean — is there drift at all?
2. **P(fall ≥ 5%)** vs unconditional — *the condor number.* This is the sold put
   spread being tested.
3. **P(fall ≥ 10%)** vs unconditional — the tail that turns a defined-risk loss
   into the maximum one.
4. t-stat on the difference in means.

**Falsification, set before running:** a signal earns further work only if it cuts
P(fall ≥ 5%) by at least 3 percentage points against unconditional, on a
five-figure sample, and the same direction holds for its bearish mirror. Mean
drift alone is not enough — drift is what a rising market gives everyone.

## 5. Plan

| Stage | Test | Kill condition |
|---|---|---|
| **G1** (this) | does any cue shrink the adverse tail? | no signal cuts P(fall≥5%) by 3pts |
| G2 | condor mechanics on real NSE EOD option chains — strikes, credit, spread paid | no net credit edge after the bid-ask actually quoted |
| G3 | the management layer: flip exits, midpoint booking, breakeven extension | no variant beats hold-to-expiry |
| G4 | portfolio: how many names, correlation, margin, drawdown | worse than the books already running |

## 6. Known limitations, stated up front

- **Liquidity is assumed, not yet verified.** Stock options outside the top 20
  names have wide spreads; G2 has to price that from real chains before any
  result here means money. research/119's traps apply — expired-contract
  intraday is unobtainable from Kite, and `option_chain.lot_size` is wrong.
- **Survivorship** — today's F&O list. Names that left the segment are absent.
- Signals here are computed on the underlying, which is causal and clean; the
  option-side realism arrives at G2.

## 7. Status log

| Time (IST) | Event |
|---|---|
| 2026-08-31 18:05 | Folder and this doc written. Probe not yet launched. |
