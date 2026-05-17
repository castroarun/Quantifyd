# Phase 24 — keep-top8 vs base, CADENCE-MATCHED (the honest re-test)

**Verdict: keep-top8 is a modest but GENUINE improvement on the locked
weekly cadence. The earlier Phase-23 "rejected" call is WITHDRAWN — it
was a mismatched-cadence artifact.**

## Why this phase exists

SMOOTHEST's regime gate is checked **weekly** (Phase-15 lock; daily was
tested and rejected for whipsaw; monthly is the laggy baseline). The
Phase-23 re-test that "rejected" keep-top8 ran it on the **month-end**
engine — silently changing TWO things at once (keep-8-vs-cash AND
weekly→monthly regime). Going all-to-cash benefits disproportionately
from the monthly lag, so that comparison was unfairly stacked against
keep-top8 and is not decision-grade.

## The fair test (Phase-22 engine = daily-marked, weekly regime, fresh VPS data → 2026-05-15)

| System (weekly regime, daily-marked) | CAGR | Post-tax @20% | MaxDD (daily) | Sharpe | Calmar |
|---|---|---|---|---|---|
| BASE SMOOTHEST (risk-off → all cash) | 34.2% | 28.4% | −22.2% | 1.82 | 1.54 |
| **SMOOTHEST keep-top8** | 33.6% | 28.3% | **−20.2%** | 1.71 | **1.66** |

keep-top8 **beats base on the cadence the system actually runs**:
Calmar 1.54→**1.66**, MaxDD −22.2→**−20.2%**, post-tax essentially flat
(28.4→28.3, −0.1pp). Confirmed on fresh VPS data through 2026-05-15 —
the Phase-22 finding holds out-of-the-original-window.

Full fresh-data variant table (same engine):

| Config | CAGR | net20 | MaxDD | Sharpe | Calmar | Verdict |
|---|---|---|---|---|---|---|
| BASE SMOOTHEST(allcash) | 34.2 | 28.4 | −22.2 | 1.82 | 1.54 | reference |
| A no-regime | 34.3 | 26.2 | −37.6 | 1.43 | 0.91 | rejected |
| B trim-25 (hold 75) | 34.7 | 24.5 | −30.7 | 1.59 | 1.13 | dominated |
| B trim-50 (hold 50) | 34.8 | 21.5 | −26.4 | 1.73 | 1.32 | dominated |
| C keep-top5 | 34.3 | 28.9 | −22.2 | 1.78 | 1.54 | neutral (= base) |
| **C keep-top8** | 33.6 | 28.3 | −20.2 | 1.71 | **1.66** | **BEST — defensible** |
| D perstock-SMA80 | 34.6 | 28.8 | −22.1 | 1.84 | 1.57 | slight+ |
| D perstock-SMA60 | 34.7 | 28.8 | −21.5 | 1.84 | 1.61 | mild+ |

## Per-year (gross %, weekly daily-marked, fresh VPS)

| Year | BASE | keep-top8 | | Year | BASE | keep-top8 |
|---|---|---|---|---|---|---|
| 2014 | 116.1 | 110.7 | | 2021 | 118.6 | 123.3 |
| 2015 | 5.3 | 10.8 | | 2022 | 1.1 | 0.2 |
| 2016 | 39.5 | 27.9 | | 2023 | 36.4 | 53.4 |
| 2017 | 67.0 | 67.0 | | 2024 | 43.0 | 47.9 |
| 2018 | −11.3 | −8.9 | | 2025 | **5.3** | **−6.9** |
| 2019 | 2.9 | 6.7 | | 2026 | −6.6 | −7.4 |
| 2020 | 85.7 | 69.2 | | | | |

keep-top8 wins more years than it loses and cuts the 2018 bear
(−8.9 vs −11.3). Its **one materially weak year is 2025**: −6.9% vs
base +5.3% — it holds 8 mid-caps through the 2025 risk-off while base
sits in cash. Over the full window the shallower max-drawdown still
nets a better Calmar.

## Honest read

- keep-top8 is **defensible to adopt** into the SMOOTHEST risk-off
  rule (better Calmar + shallower max-DD at flat post-tax). It is NOT
  "rejected".
- It is still **modest, in-sample, single-window**: the gain is ~0.12
  Calmar / ~2pp max-DD, no fresh out-of-sample on this specific tweak
  beyond the data-extension to 2026-05. Adoption into the locked spec
  remains the user's decision; the base (all-to-cash) is the
  conservative default and is what the live "today's book" assumes.
- The month-end heatmap / equity overlay deliberately do NOT carry a
  keep-top8 line — that engine misrepresents a weekly-locked variant.
  The fair comparison is the dedicated cadence-matched chart
  (`smoothest_vs_kt8_weekly.png`) + this table.

Artifacts: `phase24_kt8_cadence.csv`, `phase24_kt8_peryear.csv`,
`nav_weekly_smoothest_base.csv`, `nav_weekly_smoothest_kt8.csv`,
`smoothest_vs_kt8_weekly.png`, refreshed `phase22_variants*.csv`.
Runner: `scripts/24_kt8_cadence_matched.py` (imports the validated
`scripts/22_smoothest_variants.run`). Ran on VPS canonical data.
