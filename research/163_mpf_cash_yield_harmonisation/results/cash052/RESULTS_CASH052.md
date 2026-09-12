# Idle Cash At 5.2% Post-Tax — Every Momentum Portfolio Book

**Verdict: CONCLUDED — re-measurement adopted and published.** Not a strategy result.
No entry, exit, stop, slot count, universe or gate moved on any book; one input changed on all
five, and the page now states the instrument that input assumes.

**Date:** 2026-09-12, 19:35 → 23:5x IST · VPS · research/163 · `results/cash052/`
**Live page:** http://94.136.185.54:5000/app/mpf-report
**Status doc:** `../../MPF_CASH_YIELD_5P2_DAILY_RUN_STATUS.md`

---

## 1. What changed

Every book on `/app/mpf-report` credited idle cash at **5.0%** a year post-tax (itself set only
hours earlier, when True North came off 6.5% and Open Alpha · Base Age off 5.5%). All five now
credit **5.2%**.

**Why 5.2%.** The rule is that idle cash sits in the best post-tax cash instrument. That is an
**arbitrage fund**: equity taxation — 20% STCG on units churned inside a year, 12.5% LTCG
beyond a year, ~0.25% exit load inside a month — so ~**6.5% pre-tax** at 2025-26 cash-futures
spreads lands at ~**5.2% post-tax**. A liquid ETF is taxed at slab: LIQUIDCASE / LIQUIDADD /
LIQUIDBETF realised 5.4–5.5% pre-tax in 2025 and ~5.0% annualised in 2026, i.e. only ~3.5%
post-tax at a 30% slab. Operating rule: **bulk in the arbitrage fund, a liquid-ETF buffer** for
money needed at the next open, because arbitrage redemptions settle T+1.

**5.2% is a flat assumption, not a measured yield.** Dated review **2026-12-15**, registered at
the top of the Ops & Review Centre.

## 2. The result

| Window | Row | CAGR 5.0 → 5.2 | Δ |
|---|---|---|---|
| 20.4y | True North | 18.56% → 18.69% | +0.13 |
| 20.4y | Open Alpha · Base Age | 19.93% → 19.99% | +0.06 |
| 20.4y | IPO Base | 15.10% → 15.26% | +0.16 |
| 20.4y | TN + Base Age 50-50 | 19.80% → 19.89% | +0.09 |
| 20.4y | NIFTYBEES | 10.58% → 10.58% | 0.00 (bit-identical) |
| 2018 | True North | 19.80% → 19.93% | +0.13 |
| 2018 | Open Alpha · Base Age | 25.54% → 25.57% | +0.03 |
| 2018 | Quality Summit | 20.90% → 20.63% | −0.27 (see §4) |
| 2018 | IPO Base | 12.97% → 13.08% | +0.11 |
| 2018 | TN + Base Age 50-50 | 23.40% → 23.48% | +0.08 |
| 2018 | NIFTYBEES | 10.92% → 10.92% | 0.00 (bit-identical) |

Open Alpha · ATH + VIX: 19.23 / −34.15 / 0.56 → 20.44 / −32.92 / 0.62, 30-seed band
15.60–22.57% (median 18.81%). **Nothing reorders on either window.**

## 3. Every harness proved itself first

No curve was written before the same script reproduced that book's own published 5.0% curve:

| Book | Gate | Result |
|---|---|---|
| True North | vs `tn_..._cash05.csv` | exact, 5,066 / 5,066 rows, max rel 2.5e-16 |
| Base Age | vs `ba_navs_30seed_cash05.npz` | **bit-exact, all 30 paths, 0.0e+00** |
| IPO Base | vs r/159 `ipo_honest_curve.csv` | exact, 5,128 / 5,128 rows; published 15.00% median reproduced |
| OA · ATH + VIX | vs r/159 `curves_after_tax.csv` **and** the published summary row | exact, 2,642 / 2,642 rows; **19.23 / −34.15 / 0.56 reproduced exactly** on `compare_all.py`'s own aligned index |
| Quality Summit | vs r/160 `F_Bb7_equity.csv` on the frozen panel | exact, max rel 1.9e-16 |

Two scripts stopped at their own gates before they were right (Base Age on a CSV decimal round
trip, ATH + VIX on a union-index comparison). Both gates were fixed to test the correct thing,
not loosened to pass.

## 4. The finding worth keeping

**Single-path re-draw swamps the cash effect.** Twenty basis points change the cash balance,
which changes integer share counts, which changes whether a buy is affordable, which re-draws
every later selection in that path — worth up to ±2 points of CAGR, against the 0.02–0.16 the
cash rate is actually worth. So the drawn path can move 30× too far (ATH + VIX, +1.20) or the
wrong way (Quality Summit, −0.27) while the book is behaving exactly as the arithmetic says.

Handled by (a) **freezing the drawn seed** at the one the 5.0% page drew, and (b) testing
consistency **paired** across the whole ensemble. All five books pass:

| Book | Invested | Predicted (1−inv)×0.2 | Paired median | Verdict |
|---|---|---|---|---|
| True North | 43.0% | +0.114 | +0.130 | CONSISTENT |
| Open Alpha · Base Age | 72.9% | +0.054 | +0.060 | CONSISTENT |
| IPO Base | 31.8% | +0.136 | +0.156 | CONSISTENT |
| Open Alpha · ATH + VIX | 79.0% | +0.042 | −0.153 (SE 0.206) | CONSISTENT |
| Quality Summit | 91.2% | +0.018 | +0.013 | CONSISTENT |

Least-invested book gains most, most-invested gains least, fully invested gains nothing. That
is the whole mechanism, visible in the ordering.

## 5. Not at 5.2%, and said so on the page

The entry-surface / null-control / gate-bake-off tables are their own 70-row, 30-seed re-runs
(~4 h). `scripts/aftertax_all_052.py` is running; until it finishes the page uses research/159's
5.0% tables and **labels that section with the rate it used**. Those rows are medians of a book
~79% invested, so the cash rate shifts every row alike by ~0.04 pp and the ordering — all that
section is for — is unchanged. The generator switches over automatically on the next regen.

## 6. Owed

**Arun, operational, not a model change:** move True North's idle cash into an arbitrage fund
with a liquid-ETF buffer sized for the gate's re-entry (the gate liquidates all and re-buys 8
names, so the buffer must cover a full re-entry within T+1 of a redemption, or the redemption
must be placed the day the gate signals). Record the fund and the date in the Capital Desk /
True North dashboard note. It is item (0) of the 2026-12-15 review.
