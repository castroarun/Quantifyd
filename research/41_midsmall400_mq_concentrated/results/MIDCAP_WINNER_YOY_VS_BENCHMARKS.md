# Mid-Cap Winner — Year-by-Year vs Benchmark Indexes

Winner = `q0.5_dd__v__REG` on `mid_120d_N15` (RS-120 + 200DMA regime +
≥50% positive-month screen, top-15, monthly, top-22 buffer).
Returns shown are **gross** (pre-STCG; index returns are also pre-tax,
so this is the fair like-for-like). Post-tax CAGR 28.9% @20% STCG —
see the detailed report.

## 1. Data-honesty note (read first)

Our `market_data.db` only has a **long-history Nifty-50 proxy**
(`NIFTYBEES`, 2005→2026). It does **not** contain 2014→2026 series for
Nifty 100, Nifty Midcap (150/100) or Nifty Smallcap 250:
`NIFTY50` index = 2023+ only; `MIDCAP`/`SMALLCAP`/`MIDSMALL` ETF
tickers = 2024+ only; no Nifty-100 proxy at all. A faithful 12-year
YoY table is therefore only possible vs **Nifty 50**. The other three
are **not fabricated** — they require a proper Kite index-history
download (see §3).

## 2. Strategy vs Nifty 50 — yearly (2014–2026)

| Year | Strategy (gross) % | Nifty 50 % | Excess (pp) | Note |
|---|---|---|---|---|
| 2014* | 133.1 | 31.6 | +101.5 | inception year (full-yr midcap rally, 1.0 base) |
| 2015 | −0.8 | −4.3 | +3.5 | |
| 2016 | 22.5 | 4.0 | +18.5 | |
| 2017 | 82.8 | 29.9 | +52.9 | |
| 2018 | −1.9 | 4.8 | −6.7 | regime risk-off (small-cap bear) |
| 2019 | 2.7 | 13.6 | −10.9 | regime risk-off |
| 2020 | 62.3 | 15.4 | +46.9 | |
| 2021 | 95.2 | 26.0 | +69.2 | |
| 2022 | 12.5 | 5.5 | +7.0 | |
| 2023 | 52.4 | 21.0 | +31.4 | |
| 2024 | 38.0 | 10.4 | +27.6 | |
| 2025 | 1.4 | 11.7 | −10.3 | regime risk-off |
| 2026 | −1.3 | −1.5 | +0.2 | partial year |

**Summary:** beat Nifty 50 in **10 of 13 years**. CAGR **35.3%**
(gross) vs Nifty 50 **13.6%** over 12.1y. The only 3 down/lag years
(2018, 2019, 2025) are precisely the **regime-gated risk-off** years —
the system deliberately sat in 6.5% cash through small-cap bears,
sacrificing index-up years to protect drawdown. That asymmetry (huge
excess in up-years, controlled give-back in gated years) is the edge.

## 3. Nifty 100 / Midcap / Smallcap — pending real data

To complete a like-for-like YoY vs **Nifty 100, Nifty Midcap 150,
Nifty Smallcap 250**, the real index history must be pulled via Kite
(it serves these indices) on the **VPS** (canonical-data rule), loaded
into `market_data.db`, then this table regenerated. Until then those
columns are intentionally blank rather than estimated. The MQ100
index's own published ~20% CAGR remains the headline hurdle the
strategy is measured against (cleared: 35.3% gross / 28.9% post-tax).

Reproduce §2: `python research/41_midsmall400_mq_concentrated/` from
`phase04_chosen_gross_nav.csv` vs `NIFTYBEES` calendar-year closes.
