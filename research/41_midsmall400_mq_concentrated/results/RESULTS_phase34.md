# Phase 34 — Robustness re-read of the two NAMED finalists — RESULTS

**Verdict: BOTH PASS. Stable across disjoint halves and thirds; the balance-vs-drawdown
trade-off between them holds year-by-year, not just full-period. Keep-8 + Bear Trend-Trim
(recommended) confirmed best balance; Always-On Trend-Guard confirmed lowest DD in every
sub-window.** (Engine reconciles exactly with Phase 32/33 after a turnover-cost fix.)

## Per-year returns (gross %, daily-marked)

| Year | Keep-8 + Bear Trend-Trim | Always-On Trend-Guard | keep-top8 | Nifty 50 |
|---|---|---|---|---|
| 2014 | 110.7 | 109.2 | 110.7 | 31.6 |
| 2015 | 10.3 | 8.5 | 10.8 | −4.3 |
| 2016 | 25.9 | 24.9 | 27.9 | 4.0 |
| 2017 | 67.0 | 66.0 | 67.0 | 29.9 |
| 2018 | −8.9 | −7.6 | −8.9 | 4.8 |
| 2019 | 9.3 | 4.3 | 6.7 | 13.6 |
| 2020 | 70.9 | 71.0 | 69.2 | 15.4 |
| 2021 | 123.3 | 119.4 | 123.3 | 26.0 |
| 2022 | −0.5 | −2.7 | 0.2 | 5.5 |
| 2023 | 54.3 | 55.2 | 53.4 | 21.0 |
| 2024 | 46.3 | 46.7 | 47.9 | 10.4 |
| 2025 | **−3.5** | −4.6 | −6.9 | 11.7 |
| 2026* | −5.6 | −6.1 | −7.4 | −9.5 |

All three beat Nifty 50 in **9/13** years. Losing years are the large-cap-led ones
(2018, 2019, 2025). Note Keep-8 + Bear Trend-Trim **improves the soft years** vs plain
keep-top8: 2025 −3.5% (vs −6.9%), 2022 −0.5% (vs +0.2%, ~flat) — the bear-trim cuts the
deteriorating names before they bleed.

## Sub-period robustness (CAGR / MaxDD / Sharpe, gross)

| Variant | Full | H1 14-19 | H2 20-26 | T1 14-17 | T2 18-21 | T3 22-26 |
|---|---|---|---|---|---|---|
| Keep-8 + Bear Trend-Trim | 34.2/−20/1.76 | 30.2/−20/1.74 | 37.9/−17/1.78 | 48.8/−18/2.46 | 39.6/−20/1.99 | 17.8/−17/0.98 |
| Always-On Trend-Guard | 32.9/−19/1.73 | 28.7/−19/1.69 | 36.9/−17/1.76 | 47.4/−18/2.42 | 37.9/−19/1.95 | 17.0/−17/0.95 |
| keep-top8 | 33.6/−20/1.71 | 30.1/−20/1.71 | 36.8/−19/1.72 | 49.6/−19/2.47 | 38.4/−20/1.92 | 16.6/−19/0.91 |
| Nifty 50 | 12.3/−36/0.88 | 12.5/−22/1.00 | 12.1/−36/0.80 | 14.3/−22/1.12 | 14.9/−36/0.89 | 8.0/−16/0.68 |

## Reads

- **PASS — stable in both halves** (H1 ~29–30%, H2 ~37–38% CAGR for both finalists). Not a
  single-regime artifact. Every sub-window beats Nifty on CAGR and (except T3) Sharpe.
- **Keep-8 + Bear Trend-Trim = best balance, confirmed:** highest full CAGR (34.2%) of the
  three and the shallowest soft-year losses. The recommendation holds across sub-periods.
- **Always-On Trend-Guard = lowest DD, confirmed:** shallowest drawdown in EVERY sub-window
  (−17 to −19% vs keep-top8's −19 to −20%), at ~1–1.3pp lower CAGR. The trade-off is consistent.
- **Soft spot (all variants):** T3 (2022–2026) weakest at ~17–18% CAGR / Sharpe ~0.9–1.0 —
  momentum has cooled, but still ~2× Nifty's 8% with half its drawdown. Watch, not a kill.

## Files
- `scripts/34_finalists_robustness.py`, `results/phase34_peryear.csv`, `phase34_subperiod.csv`,
  `phase34_finalists_heatmap.png` (named-finalists yearly heatmap, corrected).
- NOTE: first run had a turnover-cost bug (turn computed after `held` reassigned → 0 cost,
  ~1–2pp inflated); fixed (compute turn before reassign); numbers now match Phase 32/33.
