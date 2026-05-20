"""NAS intraday Mark-to-Market snapshots
=========================================

The NAS DBs store only entry/exit per leg — no per-minute MTM — so the
EOD report could never draw a real intraday P&L curve (and a naive
leg-level reconstruction does NOT reconcile to the strategy's booked
net_pnl; it even sign-flips for adjusted strangles).

This module records, every few minutes during the session, each system's
**day P&L = realized + unrealized**, where:

  realized   = Σ net_pnl of today's completed trades (authoritative,
               same source the EOD report headline uses)
  unrealized = Σ over open legs of the signed mark-to-market using a
               live Kite quote (SELL: (entry-ltp)*qty; BUY: (ltp-entry)*qty)

As legs close, unrealized → 0 and realized → the booked total, so the
curve endpoint reconciles exactly to the headline P&L. Decoupled from
ticker internals on purpose (read-only kite.ltp, tiny volume).

Bulletproof: every failure is swallowed per-system; this runs inside the
market-hours scheduler and must never raise.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, date, time as dtime

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS nas_mtm_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snap_date TEXT NOT NULL,
    ts        TEXT NOT NULL,
    realized  REAL,
    unrealized REAL,
    day_pnl   REAL
);
CREATE INDEX IF NOT EXISTS idx_nas_mtm_date ON nas_mtm_snapshots(snap_date);
"""


def _conn(db):
    c = sqlite3.connect(db.db_path, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def _ensure(c):
    c.executescript(_DDL)


def _today() -> str:
    return date.today().isoformat()


def _realized_today(db) -> float:
    """Sum (entry - exit) * qty over today's closed legs — matches the
    dashboard's per-system today_pnl computation (app.py line ~3879),
    so the live curve reconciles with the card. Reading nas_trades.net_pnl
    alone misses individual SL_HIT closes inside an open strangle."""
    total = 0.0
    try:
        for p in (db.get_today_closed_positions() or []):
            ep = p.get("entry_price"); xp = p.get("exit_price")
            q = p.get("qty") or 0
            if ep is None or xp is None or not q:
                continue
            side = str(p.get("transaction_type") or "SELL").upper()
            total += (ep - xp) * q if side == "SELL" else (xp - ep) * q
    except Exception:
        pass
    return total


def _unrealized_now(db, kite) -> float:
    """Signed MTM of open legs via a live Kite quote. Best-effort."""
    try:
        active = db.get_active_positions() or []
    except Exception:
        return 0.0
    legs = [p for p in active
            if p.get("entry_price") is not None and p.get("qty")]
    if not legs:
        return 0.0
    syms = sorted({f"NFO:{p['tradingsymbol']}" for p in legs
                   if p.get("tradingsymbol")})
    ltp = {}
    try:
        # chunk to stay well under the quote-size cap
        for i in range(0, len(syms), 200):
            q = kite.ltp(syms[i:i + 200]) or {}
            for k, v in q.items():
                ltp[k.split(":", 1)[-1]] = v.get("last_price")
    except Exception as e:
        logger.warning(f"[NAS-MTM] kite.ltp failed: {e}")
        return 0.0
    u = 0.0
    for p in legs:
        lp = ltp.get(p.get("tradingsymbol"))
        if lp is None:
            continue
        ep = float(p["entry_price"]); q = float(p["qty"])
        side = str(p.get("transaction_type") or "SELL").upper()
        u += (ep - lp) * q if side == "SELL" else (lp - ep) * q
    return u


def snapshot_all() -> None:
    """Scheduler entry — append one MTM row per system. Never raises."""
    now = datetime.now()
    if not (dtime(9, 15) <= now.time() <= dtime(15, 30)):
        return
    try:
        from services.trading_calendar import get_default_calendar
        if not get_default_calendar().is_trading_day(now.date()):
            return
    except Exception:
        if now.weekday() >= 5:
            return
    try:
        from services.kite_service import get_kite
        kite = get_kite()
    except Exception as e:
        logger.warning(f"[NAS-MTM] no kite client: {e}")
        return
    try:
        from services.nas_eod_report import SYSTEMS, _resolve
    except Exception:
        return
    ts = now.isoformat(timespec="seconds")
    d = _today()
    for name, _grp, getter in SYSTEMS:
        try:
            db = _resolve(getter)
            realized = _realized_today(db)
            unreal = _unrealized_now(db, kite)
            c = _conn(db)
            try:
                _ensure(c)
                c.execute(
                    "INSERT INTO nas_mtm_snapshots "
                    "(snap_date, ts, realized, unrealized, day_pnl) "
                    "VALUES (?,?,?,?,?)",
                    (d, ts, round(realized, 2), round(unreal, 2),
                     round(realized + unreal, 2)))
                c.commit()
            finally:
                c.close()
        except Exception as e:
            logger.warning(f"[NAS-MTM] snapshot failed for {name}: {e}")


def get_today_curve(db) -> list:
    """[(ts, day_pnl), ...] ascending for today. Never raises."""
    try:
        c = _conn(db)
        try:
            _ensure(c)
            rows = c.execute(
                "SELECT ts, day_pnl FROM nas_mtm_snapshots "
                "WHERE snap_date=? ORDER BY id", (_today(),)).fetchall()
            return [(r["ts"], r["day_pnl"]) for r in rows]
        finally:
            c.close()
    except Exception:
        return []
