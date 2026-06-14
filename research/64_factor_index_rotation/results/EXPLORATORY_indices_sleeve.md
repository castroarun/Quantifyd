# EXPLORATORY (NOT published to the app) — Indian equity indices as the GTAA equity sleeve

Status: **EXPLORATORY / not decision-grade.** Committed for the record; deliberately NOT
added to `/app/backtest/*` because the headline ranking is a multiple-testing artifact
(see caveat). Tests every available Indian equity index as the equity sleeve in
{index + Gold + Nasdaq}, equal & inverse-vol, monthly, net 20 bps, window 2015-01→2026-06.

Scripts: `indices_sleeve_test.py`, `master_compare.py`, `factor_etf_test.py`.
Data: `results/indices_sleeve.csv`, `master_compare.csv`, `factor_etf_replace_nifty.csv`.

## Robust findings (trustworthy)

1. **Sensex / BSE500 / Nifty500 ARE the Nifty** — corr 0.97–1.00. Swapping Nifty→Sensex
   does nothing. Same large-cap Indian equity bet.
2. **Only 3 sectors genuinely decouple** (corr to Nifty < 0.6): **Pharma 0.45, IT 0.58,
   Media 0.59.** They are the only real diversifiers — but low/negative returns, so they do
   NOT lift Calmar. Every other sector is equity beta (corr 0.7–0.92).
3. **Midcap / Smallcap** = higher return AND higher drawdown, still ~0.87 corr to Nifty
   (higher-beta Nifty, not diversifiers). Midcap150 inv-vol 1.74 vs Nifty 1.46 — a legit
   "more aggressive sleeve" at deeper DD.
4. **Data integrity:** the Kite **Commodities** index series is CORRUPT (226% daily vol) —
   skipped. (Adds to the Quality/LowVol-index corruption found earlier.)

## The trap (why this is NOT published)

The top of the table — **FMCG 2.44, Energy 2.35, Consumption/PSE/Infra/MNC ~2.0–2.2** —
is almost certainly **overfit**:
- **28-way multiple testing** on a single benign window → best-of-N ≈ luck.
- **Inverse-vol mechanically over-weights low-vol (defensive) sleeves** → FMCG’s −6.9% book
  DD is partly construction, not edge.
- These are **2015-26 benign-window Calmars** (2.0–2.4) that the 21-year through-cycle test
  already showed deflate to ~0.8.

**Conclusion:** do NOT crown a sector sleeve. The disciplined equity sleeve remains a broad
index (Nifty / Nifty500) or the Value factor; sector-picking needs a forward thesis, not a
backtest ranking. To promote any sector sleeve it must pass a robustness gate (split-half +
per-year + 21y through-cycle) — not run yet.

## Full ranking (inverse-vol Calmar, 2015-26 — exploratory levels, treat as relative only)

See `results/indices_sleeve.csv`. Top: FMCG 2.44 / Energy 2.35 / Consumption 2.21 …
Midcap150 1.74 / Next50 1.73 / Pharma 1.69 … Nifty500 1.51 / Nifty50 1.46 / Sensex 1.43 …
IT 1.21 / Media 0.93.
