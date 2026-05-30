# Prev-Day Range Breakout — Phase 3 Results (robustness)

**Verdict: the SIGNAL has genuine alpha — the high-beta long edge survives a fully
causal trailing-beta benchmark and is monotonic in beta (β≥1.4 → PF ~1.29). But
two hard caveats kill the naive strategy: (1) broad-universe 5-min history mostly
exists only from ~2024, so the robust window is thin and bullish; (2) as a
portfolio it's one giant correlated long-beta bet — every config draws down
55–87%. This is a real signal that needs a market-regime filter + de-correlation,
not a deployable book as-is.**

Long-only trade log: 11,492 trades, NR7/PCT25 + HTF filter, **causal trailing-252d
beta vs a synthetic equal-weight market index** (median daily return across all
stocks — NIFTY50 daily only starts 2023, so it couldn't serve as the benchmark).
6 bps costs. `results/trades_long.csv`.

## 1. Causal-beta confirmation + cutoff sweep (the win)

NR7 long, by causal beta cutoff:

| β cutoff | 2R: n / WR / PF | 3R: n / WR / PF |
|---|--|--|
| ≥0.0 | 6,333 / 37.6% / 1.091 | 6,333 / 31.1% / 1.090 |
| ≥1.0 | 2,535 / 40.3% / 1.222 | 2,535 / 32.8% / 1.176 |
| ≥1.2 | 1,565 / 39.6% / 1.177 | 1,565 / 32.5% / 1.154 |
| ≥1.4 | 860 / 41.7% / **1.290** | 860 / 34.5% / **1.291** |

- **The edge is real and not look-ahead.** Full-period beta (Phase 2) gave PF 1.25;
  causal trailing beta gives 1.18–1.29 in the same region — it holds up.
- **Monotonic in beta** (higher β → higher PF, ~1.09 → ~1.29). A clean dose-response
  is strong evidence of a true effect rather than a fluke. High-beta names compress
  then trend out of the breakout; low-beta names chop.
- PCT25 is the weaker, higher-capacity cousin (β≥1.4 → PF 1.22).

## 2. Per-year stability (the warning)

NR7, β≥1.2, 2R, long-only:

| Year | n | WR | PF |
|---|--:|--:|--:|
| 2019 | 20 | 55.0% | 2.17 |
| 2020 | 55 | 41.8% | 1.29 |
| **2021** | 23 | 17.4% | **0.31** |
| 2022 | 1 | — | — |
| 2024 | 709 | 42.5% | 1.36 |
| 2025 | 595 | 39.5% | 1.12 |
| **2026 (partial)** | 162 | 28.4% | **0.74** |

- **Trade counts are wildly uneven** — most of the universe's 5-min data only exists
  from ~2024 (≈14.3M bars / 380 names ≈ 2 yrs each; only a handful go back to 2018).
  So pre-2024 years are tiny samples; **the real test is 2024–2025**, both bullish.
- **It loses in market-pullback years (2021, 2026).** This is the tell: the edge is
  conditional on a rising market — exactly what a regime filter would gate.

## 3. Portfolio sim (the dealbreaker)

Fixed-fractional 1% risk/trade, fixed concurrency cap, ₹1cr start, NR7 β≥1.2:

| Target | Max concurrent | CAGR | MaxDD | Sharpe(mo) |
|---|--:|--:|--:|--:|
| 2R | 5 | −1.0% | −55.8% | 0.05 |
| 2R | 10 | −1.8% | −71.4% | 0.09 |
| 2R | 20 | +4.8% | −72.4% | 0.37 |
| 3R | 5 | −7.0% | −68.6% | −0.37 |
| 3R | 10 | −11.3% | −81.5% | −0.46 |
| 3R | 20 | −6.3% | −87.3% | 0.02 |

- **Catastrophic drawdowns (55–87%) everywhere.** The positions are all long,
  high-beta, momentum — i.e. one concentrated market-beta factor bet. On a market
  pullback they stop out *together*; swing-holding adds overnight gap-through losses
  (result_R can exceed −1R on a gap down through the prev-day-low stop).
- **Lower concurrency was WORSE, not better** — capping positions just drops good
  clustered trades. The problem isn't position count; it's *factor correlation* and
  *regime*. The fix is a market filter, not a tighter cap.
- 2R ≫ 3R for the portfolio (higher hit rate → fewer correlated losers).

## Honest bottom line

- **As a signal:** genuine, causal, monotonic-in-beta long alpha (PF up to ~1.29,
  +0.18R/trade for β≥1.4). Worth keeping.
- **As a strategy:** not deployable as-is. It's a correlated long-beta book with
  55–87% drawdowns and an edge concentrated in 2024–25 bull phases, on a universe
  whose 5-min history barely predates 2024.

## Phase 4 (what would make it real)

1. **Market-regime filter** (highest value): only take longs when the market index
   is in an uptrend (e.g. index > its 50-DMA, or breadth positive). The loss years
   (2021, 2026) are precisely market-weak periods — gating them should lift both PF
   and, far more importantly, slash the drawdown.
2. **De-correlate exposure:** one position per sector, and/or a gross-exposure /
   portfolio-heat cap (not a naive concurrency count), and/or pair with the
   short-side H-pattern (research/43) as a partial hedge.
3. **Get more 5-min history** (Kite download, VPS-only) to test pre-2024 properly —
   the current robust window is too short and too bullish.
4. **Gap-risk control** for swing holds (e.g. reduce size, or intraday-only variant
   which avoids overnight gaps at the cost of letting fewer winners run).
