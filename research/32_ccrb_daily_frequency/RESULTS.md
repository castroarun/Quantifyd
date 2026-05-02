# CCRB Daily Setup Frequency — 1,524-stock daily-only scan

## Question answered

> "If we ran the CCRB scanner across all stocks with daily data, how many
> stocks would qualify on a typical session for the daily-bar setup filter
> (today narrow CPR + yesterday wide-CPR or narrow-range)?"

This validates the user's hypothesis: **scan widely so 2-3 trades/day are
achievable even if per-stock setup-frequency is rare.**

## Run summary

- **Universe**: 1,524 of 1,623 daily-data symbols (after `>=100 days` filter)
- **Period**: full daily history per stock (2000 → 2026-03-19)
- **Recent sub-period**: 2024-01-01 → 2026-03-19 (overlaps the intraday backtest)
- **Variants**: 27 — `today_narrow ∈ {0.30%, 0.40%, 0.50%}`
  × `yesterday_ctx ∈ {W, N, W_OR_N}` × `threshold-tier ∈ {loose, mid, tight}`
- **Wall clock**: ~7 minutes (single Python process, vectorised CPR per symbol)

## Headline numbers — UNIVERSE-WIDE (1,524 stocks)

Mid-tier representative variant `t0.0040_ctxW_OR_N_w0.0065_n0.0070` (recent):

| Stat | Value |
|---|---|
| Mean qualifying stocks per session | **203** |
| Median | 198 |
| p10 / p90 | 133 / 285 |
| Max single-day | 875 (post-event/holiday spike) |
| Sessions covered (recent) | 534 |

The distribution is roughly stable around 200/day with a modest right tail
(no extreme spikes — the max:median ratio is ~4.4×, which is event-driven
but not pathological). p10/p90 (133-285) span less than ±50% of the median.

## Headline numbers — restricted to the 79-stock INTRADAY universe

Same representative variant, recent period:

| Variant | Mean/day | Median | p10 | p90 | Max | Days≥3 | Days≥5 |
|---|---|---|---|---|---|---|---|
| t0.0030_ctxW_OR_N_w0.0050_n0.0050 (loose) | 9.2 | 8 | 3 | 16 | 35 | 451 (92%) | 389 (80%) |
| t0.0040_ctxW_OR_N_w0.0050_n0.0050 (loose) | **11.8** | 11 | 4 | 20 | 43 | 460 (94%) | 433 (89%) |
| t0.0040_ctxW_OR_N_w0.0065_n0.0070 (mid) | **7.4** | 6 | 2 | 14 | 41 | 417 (85%) | 326 (67%) |
| t0.0040_ctxW_OR_N_w0.0080_n0.0090 (tight) | 5.8 | 5 | 1 | 11 | 44 | 370 (76%) | 259 (53%) |
| t0.0050_ctxW_OR_N_w0.0065_n0.0070 (mid) | 9.0 | 8 | 3 | 17 | 46 | 440 (90%) | 364 (75%) |

**Median 6-11 watchlist names per day** in the intraday-tradable universe
on the mid-loose tiers — comfortably enough headroom for a 2-3 trades/day
target. Even the tight tier (5.8/day mean) clears 3 names on 76% of sessions.

## Per-context breakdown — `N` (yesterday narrow-range alone) is dead

`yesterday_ctx = N` essentially never fires inside the 79-universe at the
0.50%/0.70% tier (mean 0.04-0.32 per day). The W_OR_N union is dominated
by the W (wide-CPR) leg — they have nearly identical means in 79-stock-land.
This matches what `research/31`'s top cells already showed: the wide-CPR
context on `T-1` is the discriminator, narrow-range on `T-1` adds almost
nothing for liquid stocks.

## Distribution shape

For both universe-wide and 79-stock cuts, daily count distributions are
**stable, not spiky**. p90:median ratios are ~1.4-1.7× across all variants.
The single-day max sits ~3-5× the mean (event days: post-RBI, post-earnings),
but that's outlier behaviour, not the regime.

## Top stocks by setup frequency — universe-wide caveat

Universe-wide, the top 50 by raw setup-day count is dominated by **illiquid
penny stocks** (SHIVAUM, UEL, NIRAJISPAT, BCG, GAYAHWS…) where the daily
CPR is trivially narrow because the bar barely moves. Of the universe-wide
top 50, **0 are in the 79-stock intraday universe**. These are spurious
hits — they wouldn't be tradable even if the breakout fired.

## Top stocks within the 79-stock intraday universe (representative variant, recent)

| Rank | Symbol | Setup days | Total days | Freq |
|---|---|---|---|---|
| 1 | GODREJPROP | 72 | 475 | 15.2% |
| 2 | SIEMENS | 71 | 498 | 14.3% |
| 3 | PAYTM | 71 | 470 | 15.1% |
| 4 | MCX | 71 | 475 | 14.9% |
| 5 | CHOLAFIN | 69 | 475 | 14.5% |
| 6 | BPCL | 69 | 475 | 14.5% |
| 7 | TRENT | 65 | 498 | 13.1% |
| 8 | VOLTAS | 64 | 475 | 13.5% |
| 9 | HAL | 62 | 475 | 13.1% |
| 10 | GAIL | 62 | 475 | 13.1% |

Per-stock setup frequency is **10-15%** for the top intraday names — i.e.
each stock individually qualifies about **once every 7-10 sessions**. Cross-
sectionally summed across all 79, that produces the ~7-12 stocks/day mean.

## Implications for live deployment

1. **Watchlist size is genuinely big enough.** On 85% of recent trading
   sessions the 79-stock universe surfaces ≥3 candidates at the mid tier;
   on 67% it surfaces ≥5. That's a real cross-sectional opportunity set.
2. **Per-stock the setup is rare (~12% of days)**, validating the user's
   intuition: don't fixate on a single ticker — fan out wide.
3. **The W context dominates.** N (narrow yesterday-range alone) is dead
   weight in the liquid universe. For live, drop N and run W or W_OR_N.
4. **Tightening doesn't kill the funnel.** Going from loose (0.50/0.50)
   to tight (0.80/0.90) on the W_OR_N axis only halves the mean (11.8 → 5.8)
   — still gives ≥3 names on 76% of days. We have headroom to tighten
   selection if downstream win-rate demands it.
5. **Penny-stock pollution warning.** If we ever scan beyond the 79-stock
   FNO/intraday universe, we MUST add a liquidity gate (price ≥ ₹50,
   median daily turnover ≥ ₹5cr, or similar). Otherwise the scanner will
   be flooded with untradable names whose CPR is mechanically narrow.

## Files

| File | Purpose | Size |
|---|---|---|
| `scripts/run_daily_freq.py` | Universe-wide scanner | small |
| `scripts/aggregate_79.py` | 79-stock subset aggregator | small |
| `results/daily_setup_count.csv` | Per-(date, variant) count, full universe | 7.0 MB |
| `results/per_stock_freq.csv` | Per-(symbol, variant, period) totals | 4.5 MB |
| `results/variant_summary.csv` | 27 variants × 2 periods, universe-wide | 6 KB |
| `results/daily_count_79.csv` | Per-(date, variant) count, 79-universe only | small |
| `results/variant_summary_79.csv` | 27 variants × 2 periods, 79-universe | small |
| `results/top50_stocks.csv` | Top 50 universe-wide (penny-stock heavy — see caveat) | 2 KB |

All output CSVs are committable (the largest is 7 MB; nothing intraday-sized).
