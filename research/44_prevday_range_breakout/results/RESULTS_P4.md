# Prev-Day Range Breakout — Phase 4 Results (market-regime filter + risk controls)

**Verdict: the regime filter works as hypothesized — a FAST (20-DMA) market-uptrend
gate lifts the signal PF to 1.31 and cuts portfolio MaxDD from −55% to −37%. But it
does not break the ceiling: even regime-filtered, the standalone book is ~+5.6%
CAGR / −37% DD / Sharpe ~0.4 (2024–26). It remains a single-factor long-beta
strategy — real signal, but only deployable as a HEDGED/DIVERSIFIED sleeve, not
standalone. Sizing alone can't de-correlate it; that's the structural limit.**

Regime built from the synthetic equal-weight market index (median daily return,
all stocks; `build_regime.py`). Causal flags (prior-day level vs MA). Window for
the portfolio: 2024+ (where broad 5-min data is dense), per your OK.

## A. Regime effect on the per-trade edge (NR7 long, 2R)

| Config | regime OFF | mkt > 20-DMA | mkt > 50-DMA |
|---|--|--|--|
| β≥1.0 | PF 1.222 (n=2535) | **PF 1.270** (n=1050) | PF 1.119 (n=694) |
| β≥1.2 | PF 1.177 (n=1565) | **PF 1.311** (n=669) | PF 1.168 (n=429) |

- **The fast 20-DMA regime helps** (PF 1.18→1.31 at β≥1.2; WR 40%→42%). It keeps
  you out of breakouts fired into a weak tape.
- **The 50-DMA regime HURTS** — too slow; it stays "on" after the market has rolled
  over and filters out early-trend entries. 200-DMA gate had ~no qualifying days in
  the dense window (the equal-weight index is choppy) — unusable.

## B. Per-year, β≥1.0, 2R — OFF vs 20-DMA isn't uniform

reg50 was mixed by year (helps 2020/2023/2025, hurts 2024); the dominant 2024–25
sample (n≈1000/yr) is what drives the aggregate. Net: regime helps on average but
is not a per-year guarantee — 2026 (partial, market-weak) stays a loser (PF ~0.84)
even gated, because the gate turns off *after* the drop begins, not before.

## C. Portfolio sim (2024+, NR7 β≥1.0, 2R, 1% risk, heat cap 6%, max 20)

| Regime gate | CAGR | MaxDD | Sharpe(mo) | taken |
|---|--:|--:|--:|--:|
| OFF | +4.8% | −54.9% | 0.31 | 849 |
| **mkt > 20-DMA** | **+5.6%** | **−37.0%** | **0.39** | 275 |
| mkt > 50-DMA | −6.8% | −33.6% | −0.61 | 173 |

Heat-cap sweep (reg50 shown; reg20 behaves the same shape) — **no free lunch**:
lower heat → lower DD *and* proportionally lower return. e.g. heat 3% → −26% DD but
−5% CAGR; heat 20% → −53% DD. You're sliding along one risk/return line, not
improving the ratio.

## Honest bottom line

- **The regime filter is the right lever and it works** — best single improvement
  found: 20-DMA market-uptrend gate → signal PF **1.31**, portfolio MaxDD −55%→**−37%**.
- **It still isn't a standalone strategy.** ~+5.6% CAGR at −37% DD / Sharpe ~0.4 is
  not investable on its own. The cause is structural and unfixable by sizing: every
  position is a long high-beta momentum bet, so the book is ~one factor (market
  beta). When the tape breaks, the 20-DMA gate reacts a few days late and the whole
  book draws down together.
- **The genuine deliverable is the SIGNAL**, not the book: long high-beta NR7
  breakout, market > 20-DMA, stop = prev-day low, 2R, swing — PF ~1.3, +0.19R/trade.

## What it's actually good for (recommendation)

1. **As one sleeve in a diversified/hedged book** — pair it with the research/43
   short H-pattern (short-biased) so the net book isn't pure long beta. Or run it
   beside uncorrelated strategies (MQ swing, ORB intraday) where its long-beta
   drawdowns are diluted.
2. **As a watchlist/scanner signal** for discretionary long entries in confirmed
   uptrends — the +0.19R/trade edge is real; the portfolio failure is purely a
   correlation/risk-budget problem, not a signal problem.
3. **NOT as a standalone all-in book.**

## Remaining levers (diminishing returns)
- **Sector/factor caps** (need sector mapping — not in this DB) for true
  de-correlation; the most promising untested lever.
- **Intraday-only variant** to dodge overnight gap losses (at the cost of letting
  fewer winners run).
- **More 5-min history** (Kite, VPS) to test pre-2024 — current robust window is
  2024–26, a short and mostly-bullish sample.
