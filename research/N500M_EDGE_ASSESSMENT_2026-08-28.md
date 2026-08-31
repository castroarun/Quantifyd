# N500M — Is It A Good System? Assessment 2026-08-28

**VERDICT: NOT YET A VALIDATED EDGE.** Encouraging shape, statistically empty sample,
and the headline is gross of costs that the book does not model. Keep it on paper;
do not size it up.

## What the headline says

32 closed trades · 59.4% wins · **+₹16,276** · avg position ₹1.3L · profit factor 1.84.

## What the numbers actually support

| metric | value | bar |
|---|---:|---|
| mean per trade | ₹508.63 | — |
| std per trade | ₹2,173.28 | — |
| **t-stat (gross)** | **1.32** | needs ~2.0 |
| mean return / trade | 0.476% of notional | — |

t = 1.32 **before costs**. That is not distinguishable from luck.

## Costs are not modelled at all

`services/n500m_executor.py` books paper fills at the signal price. No brokerage,
no STT, no slippage. Every figure above is gross. These are MARKET orders on
5–15 min breakouts, so slippage is the dominant term, not charges.

| slippage / leg | cost / trade | net total | net / trade | t-stat |
|---:|---:|---:|---:|---:|
| 0 bps (charges only) | ₹93 | ₹13,308 | ₹416 | 1.08 |
| 2 bps | ₹145 | ₹11,642 | ₹364 | 0.95 |
| **5 bps** | ₹223 | ₹9,143 | ₹286 | **0.74** |
| 10 bps | ₹353 | ₹4,979 | ₹156 | 0.40 |
| 15 bps | ₹483 | **₹814** | ₹25 | 0.07 |

At 15 bps a leg the edge is gone entirely.

**This matches the house finding.** research/109 swept 58 intraday constructions and
concluded no OHLCV intraday edge clears the ~10 bps cost floor — the line was CLOSED.
N500M is that family, and its cost curve behaves exactly as that study predicts.

## The P&L is three trades

| | |
|---|---:|
| best trade | ₹5,211 — **32% of net** |
| top 3 trades | ₹13,389 — **82% of net** |
| net without the top 3 | **₹2,887** over 29 trades (₹100/trade) |

₹100 a trade does not survive any realistic cost assumption.

## Selection risk

11 configs have actually traded; most have n ≤ 3, one has n = 6. Per-config win
rates at that sample say nothing. The page advertises the bake-off's stats —
Sharpe 4.66, **100% win rate on n = 12** — which are in-sample numbers on a
per-stock search across symbol × signal × timeframe × direction × exit. Live is
59.4%. That gap is the overfit showing, and it is the expected direction.

The per-stock architecture is itself the risk: 30 bespoke rules fitted over a short
history is a very large search space. A single universal rule applied to every name
(the research/127 shape) is far more robust to this failure.

## How much more evidence would settle it

At the observed spread of outcomes, trades needed to reach t = 2.0:

| scenario | mean/trade | n needed | at 0.41 trades/day | if CCRB doubles the rate |
|---|---:|---:|---:|---:|
| gross | ₹509 | 73 | 8 months | 4 months |
| 2 bps | ₹364 | 143 | 17 months | 8 months |
| **5 bps** | ₹286 | **231** | **27 months** | **13 months** |
| 10 bps | ₹156 | 776 | 90 months | 45 months |

So frequency IS the binding constraint — the instinct is right. But it has to be
frequency that does not buy itself more selection bias.

## Raising frequency — in order

1. **CCRB switches on Monday. Free.** 15 already-selected rules that have never been
   able to fire. No new fitting, no new selection bias, roughly doubles the rule
   count. Halves time-to-significance on its own. Do nothing else until its rate
   is observed.
2. **Model costs in the paper book. Highest value change.** Until fills carry
   slippage and charges, every future trade is also gross and the sample never
   becomes decision-grade — you can run it for two years and still not know.
3. **Only then, expand the universe** — and expand it the right way. Going from 27
   names to ~100 means re-running the per-stock search over ~500 candidate rules
   and keeping the best 30. That makes the in-sample stats *better* and the
   out-of-sample worse. If it is done: select on one period, trade another, and
   judge on the whole expanded universe rather than on the winners.
4. **Consider dropping per-stock fitting entirely** in favour of one universal rule
   across all names. Fewer parameters, far more trades, and the result means
   something.

## What not to conclude

Not "the system works, add more stocks". Not "59% win rate is good" — win rate is
not edge; a 1.26 win/loss ratio at 59% is thin, and thin does not survive costs.
Equally, this is not evidence the system is broken. It is evidence there is not yet
enough evidence.
