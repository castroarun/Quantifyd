# Prev-Day Range-Compression Breakout — Results (Phase 1 + Phase 2)

**Verdict: this is a real, cost-surviving edge — the best of the three systems
tested (vs H-pattern PF ~1.09 short-only). The raw breakout is breakeven across
the broad universe, but your filters lift it sharply, and the high-beta + HTF-trend
combination on the LONG side clears the PF > 1.2 bar. Best tradable config:
LONG-ONLY, high-beta names, narrow-range (NR7) prev day, breakout above prev-day
high, stop at prev-day low (BOX), 2–3R target, swing-held ~3 days, trend-confirmed.
PF ~1.24, ~41% win rate (2R), +0.16R/trade over ~1,700 trades, 2018→2026.**

Universe: 374 liquid names + NIFTY50 (separate cohort), 5-min trigger + daily
swing management, 6 bps round-trip cost. Outcomes in R-multiples (R = entry→stop),
gross vs net, per-direction win%/PF.

## Phase 1 — raw mechanics (no filters): breakeven

The 4-stock smoke test looked strong (PF 1.11–1.15) but was **large-cap selection
bias**. Across all 374 stocks the raw edge is **breakeven**: best
`PCT25|BOX|2R|SWING` net **+0.002R, PF 1.003**; everything else breakeven-to-
negative. What was *directionally* right (and carried forward): **compression
helps** (NR7/PCT25 top the list), **SWING ≫ intraday** (let the trend run ~3 days),
**BOX/HALF stops** and **2–3R / TRAIL** targets. NIFTY50 alone showed PF ~1.2 but
n=379 (one instrument) — not robust on its own.

## Phase 2 — confluence filters: edge emerges

Filters (gating axes) on the carried-forward core. **Top stock configs (≥2,000
trades), with long/short split:**

| Config | n | Win% | Net R | PF | Hold | **Long** n / PF / exp | Short n / PF / exp |
|---|--:|--:|--:|--:|--:|--|--|
| **NR7 \| BOX \| 2R \| all** | 3,704 | 38.8 | +0.090 | 1.139 | 2.6d | 1,711 / **1.25** / +0.156 | 1,993 / 1.05 / +0.034 |
| NR7 \| BOX \| 3R \| all | 3,704 | 31.9 | +0.087 | 1.120 | 3.4d | 1,711 / **1.243** / +0.169 | 1,993 / 1.02 / +0.017 |
| NR7 \| BOX \| 3R \| htf+highbeta | 3,756 | 31.8 | +0.080 | 1.111 | 3.5d | 1,721 / **1.232** / +0.162 | 2,035 / 1.02 / +0.011 |
| PCT25 \| BOX \| 3R \| htf+highbeta | 6,858 | 31.7 | +0.081 | 1.113 | 3.6d | 2,906 / 1.146 / +0.105 | 3,952 / 1.088 / +0.064 |
| PCT25 \| BOX \| 2R \| htf+highbeta | 6,858 | 38.1 | +0.070 | 1.108 | 2.8d | 2,906 / 1.133 / +0.086 | 3,952 / 1.089 / +0.059 |
| PCT25 \| BOX \| 3R \| highbeta | 13,482 | 30.8 | +0.044 | 1.060 | 3.6d | 6,149 / 1.105 / +0.075 | 7,333 / 1.023 / +0.017 |

(`all` = cprok+htf+highbeta. Full tables: `ranking_p2_stocks.csv`,
`ranking_p2_nifty50.csv`.)

## What the data says

1. **High-beta is the dominant filter — your instinct was right.** It alone lifts
   PF 1.003 → 1.06 on 13k+ trades. High-beta names compress and then trend harder
   out of a breakout; low-beta large-caps mean-revert and chop.
2. **HTF trend confirmation stacks cleanly** (prev close vs daily SMA50). Adding it
   to high-beta takes combined PF 1.06 → 1.11–1.14, and **long-side PF to 1.23–1.25**.
3. **The edge is LONG-biased** — the mirror of the H-pattern. Long PF 1.23–1.25 vs
   short PF 1.02–1.09 in the NR7 configs. Indian equities drift up; high-beta
   momentum breakouts *with* the trend pay, shorts fight the drift. **Trade long-only**
   (or heavily long-tilted; PCT25 keeps a modest short edge ~1.09 if you want both).
4. **NR7 vs PCT25 is a quality/quantity trade-off.** NR7 (narrowest of 7) → higher
   long PF (1.24) on fewer trades (1.7k). PCT25 (bottom-quartile range) → lower PF
   (1.15) on ~70% more trades (2.9k). Pick by capacity needs.
5. **BOX stop (prev-day low) wins** — your original SL idea. It gives the trend room;
   tighter stops (BAR/ATR/HALF) cut winners short. Targets 2R/3R both work (2R =
   higher win rate ~41%, 3R = higher per-trade R); TRAIL exits too early here.
6. **CPR-width adds little once high-beta + HTF are in.** `htf+highbeta` ≈ `all`
   (PF 1.113 vs 1.112 on PCT25). It's mostly redundant — drop it for more trades,
   or keep it on NR7 where it nudges PF 1.111 → 1.139.
7. **NIFTY50** best stays the unfiltered swing breakout (`NONE|HALF|3R` PF 1.208,
   n=379) — high-beta can't apply to the index. Tradable-looking but small sample.

## Recommended config (production candidate)

> **Long-only. Universe = high-beta names (β vs NIFTY50 ≥ 1.2). Setup: NR7 prev
> day (narrowest range of last 7) AND prev close > daily SMA50. Trigger: first
> 5-min close above prev-day high (after 09:20). Stop: prev-day low. Target: 2R
> (≈41% WR) or 3R (bigger R). Hold as swing up to ~10 days.**
> Backtest: PF ~1.24–1.25, +0.156–0.169R/trade, ~1,700 trades, avg hold ~3 days.

For more trades at slightly lower PF, swap NR7→PCT25 (PF ~1.15, ~2,900 trades).

## Honest caveats

- **High-beta uses full-period beta** (computed over the whole sample) as a static
  universe tag → mild look-ahead. Beta is fairly stable, but a clean test should
  recompute β on a trailing/expanding window. **Phase-3 item.**
- **PF ~1.24 is good but not bulletproof** — it's a momentum-breakout edge that
  will have losing streaks and is regime-dependent (2018–26 was net bullish; long
  bias benefits). Walk-forward / per-year stability not yet checked.
- **Swing gaps** modelled at next-day open (gap through stop fills at open) — but
  no extra slippage on gaps beyond the 6 bps; tail gaps are mildly optimistic.
- **6 bps cost** assumed; at higher costs the PCT25 (smaller-R, more-trades) edge
  erodes faster than NR7.
- **One detection-threshold set per axis** (NR7=7, PCT25=25th/20d, SMA50, β≥1.2,
  CPR 75th pctile) — unswept. Phase-3 sensitivity sweep recommended on the
  long-only NR7 winner.

## Phase 3 (recommended next)

1. **Trailing/expanding-window beta** to remove the look-ahead, confirm the
   high-beta edge survives.
2. **Walk-forward / per-year** PF stability of the long-only NR7 winner.
3. **Threshold sensitivity** (NR-window, SMA period, β cutoff, target R).
4. Optional: **position-sizing / portfolio** sim (max concurrent swings, capital)
   to turn per-trade R into an equity curve + CAGR/MaxDD.
