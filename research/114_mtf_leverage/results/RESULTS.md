# RESULTS — MTF leverage on the live momentum book (research/114)

**VERDICT: 2.5x is SURVIVABLE and materially raises return — 67.3% CAGR vs 32.9% — but it is the
same edge magnified, not new alpha (Calmar FALLS 1.50 -> 1.30), and it costs a -52% drawdown.
3.0x is NOT recommended: it survives the backtest on a knife-edge, not with margin.**

Engine unchanged from research/104 (`run_lev62`), imported not copied. Rules: rsblend, top-8,
top-22 buffer, Donchian-15, weekly NIFTYBEES-100SMA gate. 2006-2026 incl. 2008 and 2020,
daily-marked, net 0.3% round-trip, leverage applied only while the gate is risk-on.

## The frontier at the REAL Zerodha MTF rate (14.6%/yr)

| Leverage | CAGR | Max DD | Sharpe | Calmar | Margin calls |
|---|---|---|---|---|---|
| 1.0x (own cash) | 32.9% | -22.0% | 1.78 | **1.50** | 0 |
| 1.3x | 40.4% | -28.6% | 1.68 | 1.41 | 0 |
| 1.6x | 47.5% | -35.0% | 1.61 | 1.36 | 0 |
| 2.0x | 56.5% | -43.1% | 1.54 | 1.31 | 0 |
| **2.5x** | **67.3%** | **-52.0%** | 1.48 | 1.30 | **0** |
| 3.0x | 77.6% | -59.7% | 1.45 | 1.30 | 0 |

NIFTYBEES buy-and-hold over the same period: 11.6% CAGR, -59.7% DD, Calmar 0.19.

## What the rate actually costs

research/104 financed at 10.5%; the runbook puts real Zerodha MTF at ~14.6% (0.04%/day).

| Leverage | CAGR @10.5% | CAGR @14.6% | Give-up |
|---|---|---|---|
| 1.3x | 40.7% | 40.4% | -0.3pp |
| 2.0x | 58.2% | 56.5% | -1.7pp |
| 2.5x | 70.3% | 67.3% | **-3.0pp** |
| 3.0x | 82.0% | 77.6% | -4.4pp |

The rate matters less than it looks because leverage is only carried while the gate is risk-on
(~75% of months) and only on the borrowed slice. It is NOT free: 3pp/yr at 2.5x.

## Why zero margin calls is REAL, not a modelling artifact

The engine calls margin when equity/gross < 25%. Solving for a one-day fall in the holdings:

| Leverage | Equity as % of gross | One-day fall that calls it |
|---|---|---|
| 2.0x | 50.0% | 33.3% |
| 2.5x | 40.0% | **20.0%** |
| 3.0x | 33.3% | **11.1%** |

An 11.1% trigger is INSIDE the historical range — NIFTYBEES fell 10.2% on 2020-03-23 and 10.1% on
2008-10-24, and a momentum basket typically moves 1.1-1.4x the index. So the 3.0x result had to be
checked rather than trusted. It holds for a specific, verifiable reason:

| Worst day | Index | Gate state |
|---|---|---|
| 2020-03-23 | -10.2% | RISK-OFF — in cash, **28 days** before |
| 2008-10-24 | -10.1% | RISK-OFF — in cash, **73 days** before |
| 2008-01-21 | -9.4% | RISK-OFF |
| 2006-05-19 | -9.1% | **RISK-ON — holding, exposed** |
| 2020-03-12 | -7.5% | RISK-OFF |
| 2008-01-22 | -7.4% | RISK-OFF |

**On 5 of the 6 worst days in 20 years the book was already in cash.** The 100-SMA gate exits weeks
before crashes bottom, so the leverage simply is not on when the tape gaps. That is the mechanism,
and it is the same one research/104 identified.

But note the exception: **2006-05-19, -9.1%, holding.** A 1.2-beta basket falls ~10.9% that day —
against an 11.1% trigger at 3.0x. That is the entire safety margin at 3x: 0.2pp. At 2.5x the
trigger is 20.0% and the same day is nowhere near it. **This is why 2.5x and 3.0x are not the same
decision, despite identical Calmar.**

## The honest read

1. **It is leverage, not edge.** Calmar falls 1.50 -> 1.30. You are buying return with risk at a
   slightly worse exchange rate, which is exactly what leverage should do.
2. **-52% is the real question**, not 67%. The book halves at some point in a 20-year cycle. Every
   backtested recovery assumes you did not stop it at the bottom.
3. **The gate is the ONLY thing making this safe.** Disable or widen it, and 2.5x becomes ruinous.
   It must never be treated as a tunable once leverage is on.
4. **Untested, and it matters:** per-stock MTF factors. 2.5x is Arun's stated conservative floor,
   not something derived — Kite does not expose the factors. If any held name carries a lower
   factor, achievable leverage is lower, never higher. Also unmodelled: intraday square-off (the
   engine only checks at daily close), pledge haircuts, and the broker's discretionary right to
   liquidate early.
5. **Sequencing:** the book is currently unhedged and below the size where the NIFTY put hedge can
   be sized (research/105). Leverage first, protection later is the wrong order.

## Recommendation

If leverage is wanted, **1.6x-2.0x is the defensible band** (47.5-56.5% CAGR, -35 to -43% DD,
Calmar 1.31-1.36) — it keeps a 33% one-day buffer to a margin call rather than 20%. **2.5x is
defensible only with the gate untouched and a written commitment to sit through -52%.** 3.0x adds
10pp of CAGR for a 0.2pp safety margin on the one exposed crash day in 20 years; that is not a
trade, it is a coin flip with the book.
