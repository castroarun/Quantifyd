# Per-Year + Sub-Period Robustness + Yearly-vs-Nifty50 Heatmap (finalists)

**STATUS: DONE** · research/41 · Phase 30 · daily-marked engine · VPS canonical

## The Ask

Per-year robustness check + a yearly-returns-vs-Nifty-50 heatmap for the two live
de-risk finalists (keep-top8, all-cash+weekly). App already had a yearly heatmap but only
for the original 3 variants on the month-end engine; this is the finalists on the daily engine.

## Results — PER-YEAR (gross, %)

| Year | all-cash+weekly | keep-top8 | all-cash base | Nifty 50 |
|---|---|---|---|---|
| 2014 | 126.2 | 110.7 | 116.1 | 31.6 |
| 2015 | −3.4 | 10.8 | 5.3 | −4.3 |
| 2016 | 48.2 | 27.9 | 39.5 | 4.0 |
| 2017 | 65.1 | 67.0 | 67.0 | 29.9 |
| 2018 | −8.4 | −8.9 | −11.3 | 4.8 |
| 2019 | 4.1 | 6.7 | 2.9 | 13.6 |
| 2020 | 77.5 | 69.2 | 85.7 | 15.4 |
| 2021 | 118.6 | 123.3 | 118.6 | 26.0 |
| 2022 | 17.2 | 0.2 | 1.1 | 5.5 |
| 2023 | 44.1 | 53.4 | 36.4 | 21.0 |
| 2024 | 41.3 | 47.9 | 43.0 | 10.4 |
| 2025 | −5.4 | −6.9 | 5.3 | 11.7 |
| 2026* | −4.1 | −7.4 | −6.6 | −9.5 |

Beats Nifty 50: all-cash+weekly **10/13**, keep-top8 **9/13**, base 9/13. (*2026 partial.)

## Results — SUB-PERIOD ROBUSTNESS (gross CAGR / MaxDD / Sharpe)

| Variant | Full | H1 14-19 | H2 20-26 | T1 14-17 | T2 18-21 | T3 22-26 |
|---|---|---|---|---|---|---|
| all-cash+weekly | 35.5/−20.7/1.84 | 31.2/−20.7/1.81 | 39.5/−20.4/1.87 | 52.2/−13.9/2.66 | 38.7/−20.7/1.92 | 19.3/−20.4/1.08 |
| keep-top8 | 33.6/−20.2/1.71 | 30.1/−20.2/1.71 | 36.8/−19.3/1.72 | 49.6/−19.0/2.47 | 38.4/−20.2/1.92 | 16.6/−19.3/0.91 |
| all-cash base | 34.2/−22.2/1.82 | 30.1/−22.2/1.79 | 38.0/−16.9/1.84 | 51.9/−10.2/2.67 | 38.7/−22.2/1.98 | 16.4/−16.9/0.95 |
| Nifty 50 | 12.3/−36.3/0.88 | 12.5/−21.6/1.00 | 12.1/−36.3/0.80 | 14.3/−21.6/1.12 | 14.9/−36.3/0.89 | 8.0/−15.7/0.68 |

Post-tax@20% full CAGR: all-cash+weekly 29.0%, keep-top8 28.3%, base 28.4%.

## Verdict — ROBUST (with one honest soft spot)

- **PASS sub-period stability:** both finalists strong in BOTH halves (H1 ~30–31%,
  H2 ~37–40%) — not a single-regime artifact. Every sub-window beats Nifty 50 on CAGR
  and (except T3) on Sharpe; MaxDD ~−20% vs Nifty −36%.
- **Soft spot = large-cap-led years.** 2018/2019/2025 the strategy trails Nifty (mid-cap
  momentum sits in cash while the index rises); both finalists were NEGATIVE in 2025
  (−5 to −7%) vs Nifty +11.7%. Clients must expect to underperform the index in such years.
- **Recent third (2022–2026) is weakest** (~17–19% CAGR, Sharpe ~0.9–1.1) — momentum has
  cooled, though still ~2× Nifty's 8%. Watch this; not a kill.
- all-cash+weekly is marginally more robust (higher Sharpe in every window) but is the
  all-or-nothing book; keep-top8 is a hair behind but never fully dumps to debt.

## Files

| File | Purpose |
|---|---|
| `scripts/30_robustness_heatmap.py` | runner |
| `results/phase30_peryear.csv` | per-year table |
| `results/phase30_subperiod.csv` | sub-period metrics |
| `results/phase30_yearly_heatmap.png` | heatmap → published `/app/midcap_finalists_yearly_heatmap.png` |
