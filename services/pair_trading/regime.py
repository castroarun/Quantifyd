"""Pair-Trading (Config D) — Quarterly cohort regime check.

STUB: documented but not auto-run.

The cointegration relationship between two stocks decays as macro regimes
shift. The original research/39 sweep was Engle-Granger tested on
2018-2023 daily prices and walk-forward validated on 2024-2025. To stay
live, the cohort needs to be re-tested every ~3 months:

  1. For each of the 6 active pairs, re-run Engle-Granger on the last
     12 months of daily closes and refit alpha + beta.
  2. If the new beta drifts >2σ from the prior fit, drop the pair.
  3. Optionally: re-screen the entire F&O universe for fresh cointegrated
     pairs and rotate the worst decayer out / best newcomer in.

For now, this module exists as a placeholder so the engine can call
`check_cohort_drift()` defensively without crashing. The actual refit
logic belongs in research/39's universe-screen script — wire that to a
quarterly cron job once the paper-trading shadow week closes.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List

logger = logging.getLogger(__name__)


def quarter_of(d: date) -> str:
    """Return 'YYYY-Qn' label for a given date."""
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def is_refresh_due(last_refresh_date: date, today: date) -> bool:
    """Return True if today is in a different calendar quarter from
    last_refresh_date. Conservative — fires once per quarter."""
    return quarter_of(today) != quarter_of(last_refresh_date)


def check_cohort_drift(cohort_dicts: List[Dict]) -> Dict[str, str]:
    """Stub: would re-fit alpha/beta on rolling 12-month window and
    flag any pair whose hedge ratio drifted > 2σ. For now, always returns
    'ok' for every pair so the engine can call this without crashing.

    To wire up properly:
      1. For each pair, load 252 days of daily closes for both legs from
         backtest_data/market_data.db.
      2. Run statsmodels.tsa.stattools.coint(price_a, price_b) — get pvalue.
      3. Run OLS log(P_a) ~ log(P_b) — get new alpha + beta.
      4. Compare new beta to stored beta; if |drift| > 2 * historical std,
         flag as 'drift'.
      5. Compare cointegration p-value: if > 0.05, flag as 'decohered'.
    """
    return {pair['name']: 'ok' for pair in cohort_dicts}


def schedule_quarterly_refresh_note() -> str:
    """Return a human-readable description of what needs to happen at the
    next quarter boundary. Surfaced in the API for the user."""
    return (
        "Quarterly cohort refresh due: re-run Engle-Granger cointegration on "
        "the F&O universe + refit alpha/beta on rolling 12-month window. "
        "See research/39 universe-screen script for the canonical pipeline."
    )
