# `features_pit_monthly.csv.gz` — the point-in-time fundamental panel

Built by `scripts/build_pit_panel.py` from `results/screener_cache/*.json` and the daily
closes in `backtest_data/market_data.db`. One row per (monthly decision date, symbol).

```python
p = pd.read_csv('results/features_pit_monthly.csv.gz', parse_dates=['date'])
```

**CSV.gz, not parquet.** The VPS venv is the live-trading venv and carries neither `pyarrow`
nor `fastparquet`; installing into it so a research script can write a different container is
not a trade worth making. Pandas decompresses by extension, so nothing else changes.

---

## The rule that governs every value in this file

A figure appears in a row **only if it had been filed by that row's date**.

| | rule | knob |
|---|---|---|
| Fiscal year ending 31-Mar-YYYY | usable from **01-Aug-YYYY** (year-end + 4 months) | `--lag-months` (3 builds the aggressive variant) |
| Quarter ending Q | usable from **Q + 60 days** | `QUARTER_LAG_DAYS` |

Decision dates are the **1st of each month**, 2015-01-01 → 2026-09-01. The consuming engine
forward-fills a row until the next date — that is the contract, and `holdings_check.py`
evaluates the screen the same way.

---

## Columns

| column | meaning |
|---|---|
| `date` | decision date, 1st of the month |
| `symbol` | market_data.db spelling, series suffix intact (`MODISONLTD-BE`) |
| `screener_ticker` | suffix stripped — the Screener page this row came from |
| `source` | `consolidated` or `standalone` — which page answered |
| `n_fy_usable` | filed fiscal years at this date. **`>= 4` is `has_data`** |
| `fy_latest` | the most recent filed year-end used |
| `sales_g3`, `profit_g3` | 3-year CAGR %, from the 4th-last to the latest usable FY. **NaN when the base year is ≤ 0** — a swing out of a loss is not a growth rate |
| `growth_span_yrs` | the actual span the CAGR covers (≈3.0; larger if Screener has a gap year) |
| `roe_avg3`, `roe_latest`, `n_roe_yrs` | ROE = net_profit / (equity_capital + reserves), ×100; the mean of the last 3 usable years |
| `roce_latest` | Screener's own `ROCE %` row, **never computed** — the page lumps liabilities and never splits out current liabilities, so EBIT/(assets − CL) off it would be invention |
| `is_lender` | no ROCE on any usable year while ROE is computable → bank/NBFC |
| `de_latest` | borrowings / (equity_capital + reserves) |
| `opm_latest`, `opm_slope3`, `opm_range3`, `opm_min3`, `n_opm_yrs` | annual `OPM %`: latest; OLS slope over the last 3 usable years (pp per year); max−min (pp); min (pp) |
| `opm_q_slope8`, `opm_q_std8`, `n_q_usable` | the same over the last 8 usable quarters — slope is **pp per quarter**, std is sample σ |
| `neg3` | any negative Sales or Net Profit in the last 3 usable FY |
| `face_value` | today's face value, from `#top-ratios`. A denominator, never a feature |
| `shares_pit` | `equity_capital / face_value` = share count in **crores** |
| `close_pit` | last close at or before the decision date (NaN if the last bar is > 15 days stale) |
| `mcap_pit` | `shares_pit × close_pit`, ₹ crore |
| `mcap_scale_suspect` | **True when a one-day close collapse below 0.55× lies in this row's FUTURE** — i.e. this row's price is probably on a pre-split scale and its mcap is inflated |

---

## Four things this panel cannot do — read these before using it

**1. Lenders have no debt/equity, at all.** Screener's balance sheet for banks and NBFCs
carries no `Borrowings` row — their borrowings sit inside `Other Liabilities`. So `de_latest`
is NaN for every lender, and a D/E ≤ 0.2 criterion therefore **excludes the whole financial
sector by construction**. That is not a bug and it is not the ROCE exemption's doing: the
lender ROCE carve-out in the masks is cosmetic, because D/E has already removed those names.
Screener's own website reaches the same place by a different route (it computes a D/E near 7
for a bank). Say "this is a non-financials screen" in the write-up rather than implying the
filter weighed financials and found them wanting. Related: bank OPM on this page is a
layout artifact (`operating_profit` can be negative for a bank) — never read OPM for a lender.

**2. The quarterly columns are a recent window, not a history.** Screener carries about
thirteen quarters. `opm_q_slope8` / `opm_q_std8` are therefore NaN before roughly mid-2023, so
any mask built on them is a recent-window mask and its pass rate before then is zero *by
absence*, not by rejection. For the long window use the annual `opm_slope3` / `opm_range3`.

**3. `mcap_pit` is the softest column in the file.** It needs face value, a share count and an
unadjusted price series, where every other criterion needs only the filed statements.
`market_data.db` is not retroactively split-adjusted, so pre-split months carry the old, higher
price scale and their mcap is inflated by the split ratio. `mcap_scale_suspect` marks those
rows. The *level* is sound — the share count reconciles against Screener's own market cap
within 10% for ~95% of names — it is the *history* that is suspect. Run the market-cap floor
with and without the flagged rows and report the gap.

**4. These are restated figures, not as-reported vintages.** Screener shows the numbers as
they stand today. The filing lag controls *when* a year becomes visible; it cannot undo a
later restatement. Annual restatements are usually small, but this is the residual look-ahead
in the panel and it should be named in the study.

And the bias that sits outside the file entirely: **companies delisted since have no Screener
page**, so the fundamental leg sees a surviving subset of the universe the price leg trades.
`results/coverage_audit.md` puts a number on it.
