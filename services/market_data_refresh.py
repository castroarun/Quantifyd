"""Live intraday refresh — keeps market_data.db current during market hours.

Designed to be invoked from APScheduler in app.py every 5 min between
09:15-15:30 IST Mon-Fri. Each invocation fetches the past ~15 min of 5-min
bars for the N500M universe (and any other universes that need it) and
appends to `market_data_unified`.

Defensive design:
- VPS-only (relies on services.data_manager._enforce_vps_only_writes guard)
- Idempotent — uses INSERT OR REPLACE via data_manager._store_data, which
  already handles dedup via the (symbol, timeframe, date) unique index
- Bounded — max 30 symbols per call, ~10s per symbol = under 5 min budget
- Resilient — per-symbol exception handling so one Kite hiccup doesn't
  poison the whole tick

Public API:
- refresh_5min(symbols: list[str]) → dict   { 'success': N, 'failed': N, 'skipped': N }
- refresh_n500m_universe() → dict           convenience wrapper for N500M's 27 stocks
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dtime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Market hours (IST) — refresh skipped outside this window
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 35)   # 5 min after close to capture the last bar

# Look-back window for each refresh tick
LOOKBACK_MINUTES = 15

# Cap per call to stay under APScheduler's 5 min budget
MAX_SYMBOLS_PER_TICK = 30


def _within_market_hours(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now()
    if now.weekday() >= 5:  # Sat/Sun
        return False
    t = now.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def refresh_5min(symbols: list[str], now: Optional[datetime] = None,
                 lookback_min: int = LOOKBACK_MINUTES) -> dict:
    """Pull the past `lookback_min` of 5-min bars for `symbols` from Kite.

    Returns counts: {success, failed, skipped, symbols_processed}.
    """
    now = now or datetime.now()
    if not _within_market_hours(now):
        logger.debug(f"[refresh_5min] skipped — outside market hours ({now.time()})")
        return {"success": 0, "failed": 0, "skipped": len(symbols),
                "symbols_processed": 0, "reason": "outside_market_hours"}

    if not symbols:
        return {"success": 0, "failed": 0, "skipped": 0, "symbols_processed": 0}

    if len(symbols) > MAX_SYMBOLS_PER_TICK:
        logger.warning(f"[refresh_5min] capping {len(symbols)} → {MAX_SYMBOLS_PER_TICK} symbols")
        symbols = symbols[:MAX_SYMBOLS_PER_TICK]

    # Lazy imports to avoid import cycles + so this module is cheap to load
    from services.kite_service import get_kite
    from services.data_manager import CentralizedDataManager

    kite = get_kite()
    if kite is None:
        logger.warning("[refresh_5min] no Kite session, skipping tick")
        return {"success": 0, "failed": len(symbols), "skipped": 0,
                "symbols_processed": 0, "reason": "no_kite"}

    dm = CentralizedDataManager(kite=kite)

    from_dt = now - timedelta(minutes=lookback_min)
    to_dt = now

    logger.info(f"[refresh_5min] fetching {len(symbols)} symbols  "
                f"window={from_dt.strftime('%H:%M')}-{to_dt.strftime('%H:%M')}")

    success, failed, errors = dm.download_data(
        symbols=symbols, timeframe="5minute",
        from_date=from_dt, to_date=to_dt,
    )

    if errors:
        # Log only first 3 to avoid log spam
        for e in errors[:3]:
            logger.warning(f"[refresh_5min] error: {e}")

    return {
        "success": success,
        "failed": failed,
        "skipped": 0,
        "symbols_processed": len(symbols),
        "errors_sample": errors[:3] if errors else [],
    }


def refresh_n500m_universe(now: Optional[datetime] = None) -> dict:
    """Convenience wrapper — refresh just the N500M trading universe."""
    from services.n500m_configs import stocks_to_watch
    symbols = stocks_to_watch()
    return refresh_5min(symbols, now=now)
