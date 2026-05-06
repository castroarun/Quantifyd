"""MST Bootstrap & Wiring
========================

Singleton management for the MST engine + executor + 30-min aggregator.

Wires the engine into:
  - NasTicker (subscribes to 5-min candles, aggregates to 30-min)
  - APScheduler (T-1 close cron at 15:25 every weekday)
  - Notifications (email alerts on state transitions)

Imported and called once at app startup.
"""
from __future__ import annotations
import logging
import threading
from datetime import datetime, date, time as dtime, timedelta
from collections import deque
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Module-level singletons
_engine = None
_executor = None
_aggregator = None
_lock = threading.Lock()


class MST30MinAggregator:
    """Aggregate 5-min candles from NasTicker into 30-min OHLC bars on the
    canonical NSE bucket boundaries: 09:15, 09:45, 10:15, ..., 15:15.

    Each 30-min bucket spans 30 minutes starting at the boundary minute.
    On 5-min candle close that completes a bucket, emits a 30-min bar to
    the registered callback.
    """

    BUCKET_MINUTES = 30
    BUCKET_STARTS_FROM = dtime(9, 15)
    BUCKET_LAST_START = dtime(15, 15)

    def __init__(self):
        self.current_bucket_start: Optional[datetime] = None
        self.current: Optional[dict] = None
        self.on_30min_close = None  # set by bootstrap

    @classmethod
    def _bucket_start_for(cls, dt: datetime) -> Optional[datetime]:
        t = dt.time()
        if t < cls.BUCKET_STARTS_FROM:
            return None
        # Find the most recent bucket-start at or before dt
        # Buckets: 09:15, 09:45, 10:15, ..., 15:15
        minutes_since_915 = (dt.hour - 9) * 60 + (dt.minute - 15)
        if minutes_since_915 < 0:
            return None
        bucket_offset = (minutes_since_915 // cls.BUCKET_MINUTES) * cls.BUCKET_MINUTES
        bucket_dt = dt.replace(hour=9, minute=15, second=0, microsecond=0) + \
                    timedelta(minutes=bucket_offset)
        return bucket_dt

    def on_5min_candle(self, candle: dict) -> None:
        """Called by NasTicker on each 5-min candle close.

        candle: {date: datetime, open, high, low, close, volume}
        """
        candle_dt = candle["date"]
        if not isinstance(candle_dt, datetime):
            return

        bucket_start = self._bucket_start_for(candle_dt)
        if bucket_start is None:
            return

        if self.current_bucket_start is None:
            # First candle ever — start a new bucket
            self.current_bucket_start = bucket_start
            self.current = {
                "bar_dt": bucket_start.isoformat(),
                "open": candle["open"], "high": candle["high"],
                "low": candle["low"], "close": candle["close"],
                "volume": candle.get("volume", 0),
            }
            return

        if bucket_start == self.current_bucket_start:
            # Same bucket — extend
            self.current["high"] = max(self.current["high"], candle["high"])
            self.current["low"] = min(self.current["low"], candle["low"])
            self.current["close"] = candle["close"]
            self.current["volume"] = self.current.get("volume", 0) + candle.get("volume", 0)
        else:
            # Bucket boundary crossed — emit completed bar, start new bucket
            completed = self.current
            if self.on_30min_close:
                try:
                    self.on_30min_close(completed)
                except Exception as e:
                    logger.exception(f"[MST] 30-min close callback failed: {e}")
            self.current_bucket_start = bucket_start
            self.current = {
                "bar_dt": bucket_start.isoformat(),
                "open": candle["open"], "high": candle["high"],
                "low": candle["low"], "close": candle["close"],
                "volume": candle.get("volume", 0),
            }


def get_engine():
    """Return the singleton MST engine, creating it if needed."""
    global _engine, _executor, _aggregator
    if _engine is None:
        with _lock:
            if _engine is None:
                _bootstrap()
    return _engine


def get_executor():
    if _executor is None:
        get_engine()  # triggers bootstrap
    return _executor


def get_aggregator():
    if _aggregator is None:
        get_engine()
    return _aggregator


def _bootstrap():
    """Initialize engine, executor, aggregator. Loads historical seed bars."""
    global _engine, _executor, _aggregator
    from config import MST_DEFAULTS
    from services.mst_engine import MSTEngine
    from services.mst_executor import MSTExecutor
    from services.trading_calendar import get_default_calendar
    from services.kite_service import get_kite

    cfg = MST_DEFAULTS

    # Try to get Kite for live mode; fall back to None if unavailable
    kite = None
    try:
        if cfg.get("live_trading_enabled") and not cfg.get("paper_trading_mode"):
            kite = get_kite()
    except Exception as e:
        logger.warning(f"[MST] Kite unavailable, forcing paper mode: {e}")

    paper_mode = cfg.get("paper_trading_mode", True) or kite is None
    enabled = cfg.get("enabled", True)

    _executor = MSTExecutor(kite=kite, paper_mode=paper_mode)
    _executor.LOTS = cfg["lots_per_leg"]
    _executor.LOT_SIZE = cfg["lot_size"]
    _executor.SPREAD_WIDTH = cfg["spread_width"]

    _engine = MSTEngine(
        executor=_executor,
        calendar=get_default_calendar(),
        paper_mode=paper_mode,
        enabled=enabled,
    )
    _engine.SPREAD_WIDTH = cfg["spread_width"]
    _engine.RESET_WIDTH = cfg["reset_width"]
    _engine.MIN_DTE = cfg["min_dte_at_entry"]
    _engine.MIN_CREDIT_PER_LOT = cfg["min_credit_per_lot"]
    _engine.PYRAMID_MAX_LEVEL = cfg["pyramid_max_level"]
    _engine.OTM_OFFSET = cfg.get("debit_otm_offset", 50)
    _engine.STRIKE_INTERVAL = cfg.get("strike_interval", 50)

    _aggregator = MST30MinAggregator()
    _aggregator.on_30min_close = _on_30min_close

    # Seed historical 30-min bars (last 200) from market_data.db so indicators
    # are warm at startup.
    _seed_historical_bars()

    logger.info(f"[MST] Bootstrapped: enabled={enabled} paper_mode={paper_mode} "
                f"buffer={len(_engine.bars)} bars")


def _seed_historical_bars():
    """Load the last 200 NIFTY 30-min bars from market_data.db into engine buffer."""
    import sqlite3
    from pathlib import Path
    market_db = Path(__file__).resolve().parents[1] / "backtest_data" / "market_data.db"
    if not market_db.exists():
        logger.warning(f"[MST] market_data.db missing; engine warmup skipped")
        return
    con = sqlite3.connect(str(market_db))
    cur = con.cursor()
    cur.execute("""
        SELECT date, open, high, low, close
        FROM market_data_unified
        WHERE symbol='NIFTY50' AND timeframe='30minute'
        ORDER BY date DESC LIMIT 250
    """)
    rows = list(reversed(cur.fetchall()))
    con.close()
    for row in rows:
        bar_dt, o, h, l, c = row
        _engine.add_historical_bar({
            "bar_dt": bar_dt, "open": o, "high": h, "low": l, "close": c,
        })
    logger.info(f"[MST] Seeded {len(rows)} historical 30-min bars (last={rows[-1][0] if rows else 'none'})")


def _on_30min_close(bar: dict) -> None:
    """Aggregator → engine handoff. Called when a 30-min bucket completes."""
    if _engine is None:
        return
    try:
        events = _engine.on_30min_bar(bar)
        if events:
            logger.info(f"[MST] {len(events)} event(s) at {bar['bar_dt']}: "
                        f"{[e['type'] for e in events]}")
            _dispatch_alerts(bar, events)
    except Exception as e:
        logger.exception(f"[MST] on_30min_bar handler failed: {e}")


def _dispatch_alerts(bar: dict, events: list[dict]) -> None:
    """Send email alerts for high-priority events."""
    try:
        from config import MST_DEFAULTS
        from services.notifications import NotificationService
        notifier = NotificationService(MST_DEFAULTS)

        for evt in events:
            etype = evt["type"]
            if etype == "flip_armed" or etype == "cst_subsequent" or etype == "flip_discarded":
                priority = "low"
            elif etype in ("flip_activated", "condor_built", "pyramid_fired",
                           "rolled", "mst_flip_close", "t_minus_1_close"):
                priority = "high"
            elif etype == "kill_switch":
                priority = "critical"
            else:
                priority = "normal"

            title = f"MST · {etype.upper()}"
            msg_parts = [f"{k}={v}" for k, v in evt.items() if k != "type"]
            message = f"NIFTY 30m · {bar['bar_dt']} · " + " · ".join(msg_parts)
            notifier.send_alert(
                alert_type=_alert_type_for(etype),
                title=title, message=message, data=evt, priority=priority,
            )
    except Exception as e:
        logger.warning(f"[MST] Alert dispatch failed: {e}")


def _alert_type_for(event_type: str) -> str:
    """Map MST event types to NotificationService alert types."""
    return {
        "flip_armed": "system_alert",
        "flip_activated": "trade_entry",
        "flip_discarded": "system_alert",
        "condor_built": "trade_entry",
        "cst_subsequent": "system_alert",
        "pyramid_fired": "trade_entry",
        "rolled": "trade_entry",
        "mst_flip_close": "trade_exit",
        "t_minus_1_close": "trade_exit",
        "kill_switch": "system_alert",
    }.get(event_type, "system_alert")


def register_nas_ticker_subscriber():
    """Hook MST aggregator into NasTicker's 5-min candle close pipeline.

    Called once at app startup AFTER NasTicker is instantiated.
    """
    from services.nas_ticker import get_nas_ticker
    aggregator = get_aggregator()
    ticker = get_nas_ticker()

    # Add to NasTicker's additional_subscribers list (we add this attribute)
    if not hasattr(ticker, "additional_subscribers"):
        ticker.additional_subscribers = []

    # Also wrap the aggregator's on_candle to ensure it gets the 5-min candle
    def _mst_5min_subscriber(candle: dict):
        try:
            aggregator.on_5min_candle(candle)
        except Exception as e:
            logger.exception(f"[MST] 5-min subscriber error: {e}")

    ticker.additional_subscribers.append(_mst_5min_subscriber)
    logger.info("[MST] Registered as NasTicker 5-min subscriber")


def t_minus_1_check_cron():
    """Scheduler-invoked: at 15:25 every weekday, check if today is T-1 of any
    open position. If so, close + roll over."""
    if _engine is None:
        return
    today = date.today()
    try:
        events = _engine.on_t_minus_1_check(today)
        if events:
            logger.info(f"[MST T-1 cron] {len(events)} events: {[e['type'] for e in events]}")
            _dispatch_alerts({"bar_dt": datetime.now().isoformat()}, events)
    except Exception as e:
        logger.exception(f"[MST T-1 cron] handler failed: {e}")


def set_mode(mode: str) -> dict:
    """Mode toggle — off / paper / live.

    Updates engine.enabled, engine.paper_mode, MST_DEFAULTS dict, and
    re-instantiates the executor with the right Kite handle.
    """
    from config import MST_DEFAULTS
    from services.kite_service import get_kite

    if mode not in ("off", "paper", "live"):
        return {"error": f"Invalid mode: {mode}"}

    eng = get_engine()
    if mode == "off":
        eng.enabled = False
        MST_DEFAULTS["enabled"] = False
    elif mode == "paper":
        MST_DEFAULTS["enabled"] = True
        MST_DEFAULTS["paper_trading_mode"] = True
        MST_DEFAULTS["live_trading_enabled"] = False
        eng.enabled = True
        eng.paper_mode = True
        if _executor is not None:
            _executor.paper_mode = True
            _executor.kite = None
    elif mode == "live":
        MST_DEFAULTS["enabled"] = True
        MST_DEFAULTS["paper_trading_mode"] = False
        MST_DEFAULTS["live_trading_enabled"] = True
        eng.enabled = True
        eng.paper_mode = False
        # Attach Kite handle to executor
        try:
            kite = get_kite()
            if _executor is not None:
                _executor.paper_mode = False
                _executor.kite = kite
        except Exception as e:
            logger.error(f"[MST] Cannot switch to LIVE — Kite unavailable: {e}")
            # Roll back to paper to avoid silent failure
            MST_DEFAULTS["paper_trading_mode"] = True
            MST_DEFAULTS["live_trading_enabled"] = False
            eng.paper_mode = True
            return {"error": f"Kite unavailable, stayed in paper mode: {e}",
                    "mode": "paper"}

    return {
        "mode": mode,
        "enabled": MST_DEFAULTS["enabled"],
        "paper_trading_mode": MST_DEFAULTS["paper_trading_mode"],
        "live_trading_enabled": MST_DEFAULTS["live_trading_enabled"],
    }
