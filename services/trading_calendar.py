"""NSE Trading-day Calendar with Holiday Awareness
=================================================

Single-source-of-truth for NSE trading days, weekly option expiries, and T-1
calculations. Used by MSTEngine (and any other strategy needing holiday-aware
date math).

The holiday list is maintained in `config/nse_holidays_<year>.json`, updated
annually by the operator from NSE's official calendar.

Weekly NIFTY option expiries fall on Tuesdays. If Tuesday is an NSE holiday,
the exchange shifts expiry to the previous trading day (Monday, or earlier).
The exchange's actual expiry dates can be queried via `kite.instruments('NFO')`
— prefer that as source of truth in production. This module's
`next_weekly_expiry()` is a calendar-based fallback.
"""
from __future__ import annotations
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

NIFTY_WEEKLY_EXPIRY_WEEKDAY = 1  # 0=Mon, 1=Tue, ..., 6=Sun


class NSETradingCalendar:
    """NSE trading-day calendar with holiday awareness.

    Loads the holiday list for a given year from a JSON file.
    Provides is_trading_day, previous_trading_day, next_trading_day, and
    weekly-expiry helpers.

    Lazy-loads holiday lists for additional years on demand (e.g. when
    computing a date in 2027 from a 2026 base).
    """

    HOLIDAYS_PATH = Path(__file__).resolve().parents[1] / "config"

    def __init__(self, year: Optional[int] = None):
        self._holidays_by_year: dict[int, set[date]] = {}
        seed_year = year or date.today().year
        self._ensure_year_loaded(seed_year)

    def _ensure_year_loaded(self, year: int) -> None:
        if year in self._holidays_by_year:
            return
        path = self.HOLIDAYS_PATH / f"nse_holidays_{year}.json"
        if not path.exists():
            logger.warning(
                f"NSE holidays file missing for {year}: {path}. "
                f"Treating all weekdays in {year} as trading days. "
                f"Update config/nse_holidays_{year}.json from NSE's official calendar."
            )
            self._holidays_by_year[year] = set()
            return
        with open(path) as f:
            data = json.load(f)
        self._holidays_by_year[year] = {
            date.fromisoformat(d) for d in data.get("holidays", [])
        }
        logger.info(
            f"Loaded {len(self._holidays_by_year[year])} NSE holidays for {year} from {path.name}"
        )

    def is_holiday(self, d: date) -> bool:
        self._ensure_year_loaded(d.year)
        return d in self._holidays_by_year[d.year]

    def is_trading_day(self, d: date) -> bool:
        if d.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        return not self.is_holiday(d)

    def previous_trading_day(self, d: date) -> date:
        prev = d - timedelta(days=1)
        # Bound: don't loop more than 14 days backwards
        for _ in range(14):
            if self.is_trading_day(prev):
                return prev
            prev -= timedelta(days=1)
        raise RuntimeError(f"No trading day found within 14 days before {d}")

    def next_trading_day(self, d: date) -> date:
        nxt = d + timedelta(days=1)
        for _ in range(14):
            if self.is_trading_day(nxt):
                return nxt
            nxt += timedelta(days=1)
        raise RuntimeError(f"No trading day found within 14 days after {d}")

    def trading_days_between(self, start: date, end: date) -> int:
        """Number of trading days strictly between (start, end] — i.e., includes end if trading day, excludes start."""
        n = 0
        d = start + timedelta(days=1)
        while d <= end:
            if self.is_trading_day(d):
                n += 1
            d += timedelta(days=1)
        return n

    # ---- Weekly expiry helpers (NIFTY) ----

    def next_weekly_expiry(self, from_date: date, min_dte: int = 6) -> date:
        """Next NIFTY weekly Tuesday expiry that is at least `min_dte` calendar
        days after `from_date`. If the candidate Tuesday is a holiday, the
        exchange shifts expiry to the previous trading day; this function
        applies the same rule.

        For production use, prefer `kite.instruments('NFO')` filtered to
        NIFTY weekly contracts — those reflect the actual exchange-published
        expiry dates including all shifts. This calendar-based version is a
        fallback when Kite is unavailable.
        """
        # Step 1: find the next Tuesday at least min_dte days away
        d = from_date + timedelta(days=min_dte)
        # Walk forward to the next Tuesday on/after d
        days_until_tue = (NIFTY_WEEKLY_EXPIRY_WEEKDAY - d.weekday()) % 7
        candidate = d + timedelta(days=days_until_tue)

        # Step 2: if candidate is a holiday, shift to previous trading day.
        # If shift would push us below min_dte, skip to the next week.
        while True:
            if self.is_trading_day(candidate):
                if (candidate - from_date).days >= min_dte:
                    return candidate
                # Holiday-shifted date doesn't meet min_dte; skip a week
                candidate += timedelta(days=7)
                continue
            shifted = self.previous_trading_day(candidate)
            if (shifted - from_date).days >= min_dte:
                return shifted
            # Shifted date too close; try the Tuesday after
            candidate += timedelta(days=7)

    def t_minus_1(self, expiry: date) -> date:
        """T-1 trading day before the given expiry."""
        return self.previous_trading_day(expiry)


# Module-level singleton for convenience
_default_calendar: Optional[NSETradingCalendar] = None


def get_default_calendar() -> NSETradingCalendar:
    global _default_calendar
    if _default_calendar is None:
        _default_calendar = NSETradingCalendar()
    return _default_calendar
