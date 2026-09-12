# research/163 — One Cash Rate For Every Book

**Verdict: CONCLUDED — the page is now like-for-like.** Every book on `/app/mpf-report`
credits idle cash at **5% a year, post-tax, accrued daily**. No trading rule changed, no live
or paper book was touched, nothing under `services/` was edited.

Run 12-Sep-2026 on the VPS. Report page: http://94.136.185.54:5000/app/mpf-report

---

## 1. What was wrong

| Book | Its study's idle-cash assumption | Engine |
|---|---|---|
| True North | **6.5%** | `research/144_truenorth_reassessment/scripts/tn_attrib_engine.py` (`CASH_ANNUAL`) |
| Open Alpha · Base Age | **5.5%** | `research/161_ath_base_age_breakout/scripts/bt_core.py` (`IDLE_YIELD`) |
| IPO Base, Open Alpha · ATH + VIX | 5.0% | `research/159 .../build_curves.py` (`CASH_Y`) |
| Quality Summit | 5.0% | `research/160 .../qg_engine.py` (`cash_yield`) |

True North holds cash **57%** of the time, so its assumption alone was worth about a point a
year to it — against books credited at 5%.

## 2. The post-tax check

Asked for explicitly, checked by reading each engine's loop:

| Engine | Accrual | Taxed? |
|---|---|---|
| r/144 `tn_attrib_engine.run` | `if cash > 0: cash *= (1+cash_y)**(1/252)` every bar | NO — `settle_tax()` taxes `st_gain`/`lt_gain`, accumulated only inside `sell()` |
| r/161 `bt_core.simulate` | `cash *= 1 + ((1+iy)**(1/252) - 1)` every bar | NO — the FY settlement taxes `fy_st`/`fy_lt`, accumulated only on position exits |
| r/159 `oa_entry_mechanics.simulate` | same construction, `cash_yield=0.05` | NO |
| r/160 `qg_engine` | same construction, `cash_yield=0.05` | NO |

**All four credit the yield daily and none taxes it.** Nobody's 5% has to be re-labelled as
pre-tax.

## 3. Reproduction before re-running — both passed

| Book | Proof |
|---|---|
| True North at 6.5% | **5,052 of 5,066 daily NAV points identical to machine precision** (max relative difference 2e-16) against `nav_INC_cash_n8_d15_tax1.csv`. The last 14 bars (from 17-Aug-2026) differ by up to 1.3% because `market_data.db` has been refreshed since r/144 ran on 3-Sep. That data refresh is worth **+0.05 pp** of CAGR on its own and is reported separately from the yield effect below. |
| Base Age at 5.5% | **BIT-EXACT** — max relative difference **0.000e+00** against `curves161.npz['WINNER']`, and 21.26% median / 19.87% worst / −34.80% / Calmar 0.618 reproduced to the decimal. |

Neither re-run wrote anything into r/144's or r/161's results folders.

## 4. Before and after — every headline row, both windows

**20.4-year window, 2006-04-03 → 2026-09-03, after tax**

| Row | CAGR before | CAGR after | Δ | MaxDD before | MaxDD after | Δ | Calmar before → after |
|---|---|---|---|---|---|---|---|
| **True North** | 19.48% | **18.56%** | **−0.92** | −23.67% | **−24.95%** | −1.28 | 0.82 → **0.74** |
| **Open Alpha · Base Age** | 20.27% | **19.93%** | **−0.34** | −32.45% | **−32.73%** | −0.28 | 0.62 → **0.61** |
| **TN + Base Age 50-50** | 20.42% | **19.80%** | **−0.62** | −25.24% | **−25.68%** | −0.44 | 0.81 → **0.77** |
| IPO Base | 15.10% | 15.10% | 0.00 | −35.86% | −35.86% | 0.00 | 0.42 → 0.42 |
| NIFTYBEES | 10.58% | 10.58% | 0.00 | −59.71% | −59.71% | 0.00 | 0.18 → 0.18 |

**2018 window, 2018-08-01 → 2026-09-04, after tax**

| Row | CAGR before | CAGR after | Δ | MaxDD before | MaxDD after | Δ | Calmar before → after |
|---|---|---|---|---|---|---|---|
| **True North** | 20.64% | **19.80%** | **−0.84** | −18.37% | **−19.14%** | −0.77 | 1.12 → **1.03** |
| **Open Alpha · Base Age** | 26.16% | **25.54%** | **−0.62** | −26.33% | **−26.61%** | −0.28 | 0.99 → **0.96** |
| **TN + Base Age 50-50** | 24.11% | **23.40%** | **−0.71** | −19.93% | **−20.46%** | −0.53 | 1.21 → **1.14** |
| Quality Summit | 20.90% | 20.90% | 0.00 | −38.90% | −38.90% | 0.00 | 0.54 → 0.54 |
| IPO Base | 12.97% | 12.97% | 0.00 | −35.86% | −35.86% | 0.00 | 0.36 → 0.36 |
| NIFTYBEES | 10.92% | 10.92% | 0.00 | −36.34% | −36.34% | 0.00 | 0.30 → 0.30 |

Open Alpha · ATH + VIX (19.23% / −34.15% / 0.56), the entry-mechanic surface, the null
control and both gate bake-offs are **byte-identical**, asserted by
`scripts/check_unchanged.py`.

**Nothing reorders.** True North still has the best Calmar of the individual books on the
20-year window and Base Age still has the highest CAGR; the blend still has the best Calmar
of anything on the page.

**On True North's −0.92 rather than −0.97.** The pure yield effect, measured with both runs
on today's data, is **−0.97 pp** of CAGR. The page shows −0.92 because the "before" figure
came from r/144's September file and the data refresh described in §3 is worth +0.05 pp.

## 5. The Base Age invested measurement (the page's last "NOT MEASURED")

r/161's engine was re-run with the daily invested fraction recorded — market value of open
positions over NAV, the convention r/158's `oa_entry_mechanics.py` uses in `inv_acc`:

| | |
|---|---|
| **Invested, 30-seed median** | **72.89%** |
| Band across 30 seeds | 72.73% … 73.05% |
| Cash share | **27.1%** |
| Daily series (drawn seed) | `results/baseage_invested_daily.csv` |

The band is very tight because the fraction is set by how often the 16 slots are full, not by
which names win them.

**It contradicts the handover.** The handover doc asserted **~67% invested / 33% cash** for
Base Age, from memory, with no file on disk carrying it. The measured figure is **72.9%** —
about 6 points higher. The page now uses the measurement and says so.

**Consistency check.** Dropping the yield from 5.5% to 5.0% should cost roughly
(cash share) × (0.5 pp). From the measured daily series, summed exactly over the window, that
is **+0.127 pp**. The paired per-seed median delta is **+0.180 pp**, with a per-seed spread of
−1.65 to +1.05 — changing the yield perturbs integer share counts and therefore which names
win slot contention, so the spread is wide and the median carries a standard error of about
0.13 pp. **The two agree well within that noise**, which is the cheap validation the
measurement needed.

## 6. Files

Curve files the report now reads (identical columns, index and rebasing to the r/159 files
they replace; only two columns differ):

- `results/full_period_after_tax_cash05.csv`
- `results/all_systems_after_tax_cash05.csv`

The generator defaults to these; `--curves-dir` (with `--full-period-csv` / `--roster-csv`)
rebuilds the page on any other set, including the pre-12-Sep one.
