# Final Systems — Yearly Breakup (return + max-DD) vs Benchmarks

## Universe & period (both systems)

- **Stock universe:** survivorship-free **point-in-time MID-cap
  liquidity band** — rank **101–250** by trailing-6-month median traded
  value (close×volume), **reconstructed monthly** from all ~1,623 NSE
  daily symbols. NOT today's MQ100, NOT index membership (the supplied
  MQ100 is used only for the live pick list, never the backtest).
- **Period:** **2014-01-31 → 2026-03-19 (~12.1 years)**, monthly
  rebalances; spans the 2018–19 small-cap bear, Mar-2020 COVID crash,
  2022 correction, 2025 drawdown.
- **Core (shared):** RS-120 vs NIFTYBEES, N=15, monthly, top-22 buffer,
  q0.5 positive-month quality, 0.4% RT cost, 6.5% bear-cash.
- **SMOOTHEST** = + SMA100 regime + ATH≤10% entry + per-stock-SMA100 +
  12% trail (Phase 11 winner).
- **MAX-RETURN** = + SMA100 regime + ATH≤10% entry, and in risk-off
  short 1× Nifty (beta-hedge) instead of going to cash (Phase 10).

## Yearly table

| Year | Smooth ret% | Smooth maxDD% | MaxRet ret% | MaxRet maxDD% | Nifty 50% | Midcap150% | Smallcap250% |
|---|---|---|---|---|---|---|---|
|2014|101.1|0.0|122.5|0.0|31.6|59.4|n/a|
|2015|2.7|−4.8|7.3|−3.6|−4.3|8.0|n/a|
|2016|35.1|0.0|18.1|−6.7|4.0|4.4|n/a|
|2017|64.5|0.0|72.0|0.0|29.9|53.0|n/a|
|2018|−11.8|−14.5|−9.9|−19.3|4.8|n/a|−27.2|
|2019|10.3|−9.9|13.9|−5.6|13.6|−0.3|−8.6|
|2020|85.7|−1.1|108.4|−2.1|15.4|n/a|n/a|
|2021|108.1|−0.1|129.1|−0.1|26.0|45.1|60.1|
|2022|16.1|−3.3|27.2|−6.2|5.5|1.8|−4.9|
|2023|40.0|0.0|70.3|−3.2|21.0|42.7|n/a|
|2024|45.5|−4.8|63.0|−3.7|10.4|23.1|25.6|
|2025|−0.8|−9.5|−11.8|−8.8|11.7|4.8|n/a|
|2026|−0.7|0.0|1.4|0.0|−1.5|n/a|0.3|

**Overall:** SMOOTHEST 35.6% CAGR / −15.1% MaxDD (29.6% post-tax @20%
STCG) · MAX-RETURN 42.8% CAGR / −22.7% MaxDD (34.0% post-tax) ·
**Nifty 50 13.1% CAGR** over the same window.

## Read

- Both systems beat Nifty 50 nearly every year and ≈**3×** its CAGR;
  they also beat Nifty Midcap 150 in every comparable year.
- Worst years are exactly the small-cap bears **2018 & 2025**; Smoothest
  contains them best (−11.8% / −0.8%); Max-Return runs hotter with
  deeper dips.
- 2020/2021 (+86/+108 and +108/+129) are the CAGR engines.

## Honest caveats

1. **Benchmark data gaps (NOT fabricated):** Nifty Midcap 150 has no
   2018 / 2020 / 2026 in our on-disk data; Nifty Smallcap 250 is sparse
   (only 2018/19/21/22/24/26). Source is niftyindices, which proved
   flaky; per instruction it was **not** re-queried. Those cells are
   `n/a`. Nifty 50 (NIFTYBEES, in DB) is complete and reliable.
2. **System per-year maxDD is month-end-NAV based** — "0.0%" = no
   month-end-to-month-end decline that year, not zero intra-month risk.
   The trustworthy risk figures are the **overall** MaxDD
   (−15.1% / −22.7%) computed off the full monthly curve.
3. Parent-study caveats still bind: price-path "quality" ≠ fundamentals;
   PIT universe is a liquidity proxy; LTCG not netted; no OOS on these
   refinements; nothing wired live.

Source: `scripts/12_yearly_breakup.py` → `results/phase12_yearly_breakup.csv`.
