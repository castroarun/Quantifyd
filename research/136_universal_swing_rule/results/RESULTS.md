# research/136 — Universal Swing Rule · RESULTS

## VERDICT: **NO INVESTABLE EDGE.** Concluded 2026-08-31.

Breakout **REFUTED**. Reversion is a real **SIGNAL** that does not survive being
assembled into a book. The market-regime brake that rescues the momentum book
**makes this one worse**. Nothing to deploy; one narrow idea worth keeping.

Daily bars 2015-01-01 → 2026-08-28 · 1,101 eligible symbols · 1,050,447 eligible
rows · entry at next open · date-matched random control throughout.

---

## G1 — buying strength loses, on every cell

| lookback | horizon | signal | random | **excess** | t |
|---:|---:|---:|---:|---:|---:|
| 20d high | 2d | +0.088% | +0.197% | **−0.109%** | −7.20 |
| 20d high | 5d | +0.282% | +0.431% | **−0.149%** | −6.62 |
| 20d high | 10d | +0.708% | +0.813% | **−0.105%** | −3.34 |
| 55d high | 5d | +0.242% | +0.344% | **−0.102%** | −3.64 |

231,510 signals, 8 of 11 years negative. **Every raw signal return is positive** —
without the date-matched control this reads as a working breakout system. It is not.
Short-horizon breakout momentum does not exist in this universe, which also explains
N500M's vol-BO half without needing any further story.

## G1b/c — buying weakness works, and the trend filter is the carrier

Excess over date-matched random, in bps:

| arm | 2d | 5d | 10d | 15d | 20d | 30d | shape |
|---|---:|---:|---:|---:|---:|---:|---|
| A · RSI(2)<10 | +10 | +8 | +10 | +8 | +9 | +3 | irregular |
| **B · RSI(2)<10 & close>200DMA** | **+16** | **+16** | **+24** | **+24** | **+26** | **+32** | **MONOTONIC** |
| C · RSI(2)<5 | +10 | +10 | +9 | +6 | +12 | +4 | irregular |

Arm B: 174,562 signals, t 6.3–12.6. Two negative results worth keeping: the
unfiltered signal never clears costs, and **deeper oversold does not pay more** —
arm C is no better than A, so "more extreme = better" is refuted.

## G2 — the book loses to the index

10 slots, equal notional, ₹10L, both legs charged:

| hold | cost | CAGR | MaxDD | Calmar | Sharpe |
|---:|---:|---:|---:|---:|---:|
| 15d | 20 bps | 16.06% | −41.9% | 0.38 | 0.76 |
| 20d | 30 bps | 9.32% | −46.7% | 0.20 | 0.50 |
| 30d | 30 bps | 13.53% | −46.3% | 0.29 | 0.67 |
| **NIFTYBEES** | — | **12.47%** | **−36.3%** | **0.34** | **0.89** |

**Every variant has a worse Sharpe than owning the index**, with drawdowns of −42%
to −55%. Walk-forward: 2015–2020 the book made **5.76%** against the index's
**13.57%**. The apparent edge lives entirely in 2021–2026.

**Why a real per-trade edge fails as a book:** the trades cluster. Everything becomes
oversold at the same moment, so ten "diversified" slots are one leveraged bet on the
market, entered while it falls. The stock-level 200-DMA does not protect: in a crash
a stock sits above its own average right until you buy it.

## G2b — the market brake is refuted as the fix

| variant | CAGR | MaxDD | Calmar | Sharpe |
|---|---:|---:|---:|---:|
| no gate (15d/20bps) | 16.06% | −41.9% | 0.38 | 0.76 |
| **NIFTYBEES>200DMA gate** (20d/20bps) | 8.97% | −41.1% | 0.22 | 0.52 |

Halves the return and barely moves the drawdown. **2020: ungated +37.09%, gated
−3.29%** — the gate removed the crash recovery, the book's best year.

**The mechanism: reversion needs the crash; momentum needs to avoid it.** Opposite
regime dependencies. This is why the same 200-DMA brake that is "the whole risk
story" for Momentum 30 is destructive here. Do not port that gate across families
without testing the sign.

## What survives

The per-trade reversion excess is real (t 6–12 on 174k signals) and monotonic in
horizon. It is not a book, but it may be an **entry-timing overlay**: when the
momentum book wants a name, wait for RSI(2)<10 rather than buying at signal. That
reuses an existing book's risk management instead of needing its own, and it is a
narrow, cheap test. **Not yet run.**

## What is now closed

- Short-horizon (2–30 day) breakout entries — negative edge, retired.
- Deeper-oversold as an improvement — refuted.
- Market-regime gating of a reversion book — refuted.
- Intraday remains closed from research/109 + /110.

## Sins guarded

| sin | control |
|---|---|
| look-ahead | entry at next open; all windows trailing; market gate shifted one day |
| survivorship | acknowledged — symbol list is today's Nifty-500, so absolute levels are upper bounds; the control draws from the same pool, so *excess* is the trustworthy quantity |
| overfitting | 3 arms × 6 horizons, all reported; no parameter search; no cell selected on |
| cost neglect | swept 0/20/30/50 bps at G1, 20/30 bps at G2 |
| regime | per-year and 2015-20 vs 2021-26 walk-forward |
| single-factor | date-matched random control at G1 *and* a random-entry book at G2 |
| capacity | ₹5 crore trailing turnover floor; untested beyond that |
