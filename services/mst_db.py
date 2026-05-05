"""MST Trading Database
=====================

SQLite persistence for the MST (Master SuperTrend / Child Stochastic) NIFTY
options strategy. Mirrors the KC6 pattern in services/kc6_db.py.

Tables:
  mst_bars      - 30-min OHLC bars + computed indicators
  mst_events    - signal events (flip_armed, flip_activated, cst_trigger,
                  condor_built, pyramid_fired, rolled, t_minus_1_close, etc.)
  mst_positions - one row per options leg
  mst_orders    - order audit trail
  mst_equity    - daily P&L curve
"""
from __future__ import annotations
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Resolve DB path Railway-style (DATA_DIR env var) like KC6, else local backtest_data
import os
DATA_DIR = os.getenv("RAILWAY_VOLUME_MOUNT_PATH") or str(
    Path(__file__).resolve().parents[1] / "backtest_data"
)
DB_PATH = Path(DATA_DIR) / "mst_trading.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mst_bars (
    bar_dt TEXT PRIMARY KEY,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    atr21 REAL,
    st_upper REAL, st_lower REAL, st_value REAL,
    direction INTEGER,
    stoch_k REAL, stoch_d REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mst_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    direction INTEGER,
    bar_dt TEXT NOT NULL,
    price REAL,
    flip_high REAL,
    flip_low REAL,
    pyramid_level INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_mst_events_bar ON mst_events(bar_dt);
CREATE INDEX IF NOT EXISTS idx_mst_events_type ON mst_events(event_type);

CREATE TABLE IF NOT EXISTS mst_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_label TEXT NOT NULL,
    direction INTEGER NOT NULL,
    pyramid_level INTEGER NOT NULL DEFAULT 1,
    leg_role TEXT NOT NULL,
    side TEXT NOT NULL,
    instrument_token INTEGER,
    tradingsymbol TEXT NOT NULL,
    strike INTEGER NOT NULL,
    option_type TEXT NOT NULL,
    qty INTEGER NOT NULL,
    entry_price REAL,
    entry_time TEXT,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING',
    pnl_inr REAL,
    order_id TEXT,
    paper_mode INTEGER NOT NULL DEFAULT 0,
    expiry_date TEXT NOT NULL,
    t_minus_1_date TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_mst_positions_status ON mst_positions(status);
CREATE INDEX IF NOT EXISTS idx_mst_positions_week ON mst_positions(week_label);
CREATE INDEX IF NOT EXISTS idx_mst_positions_t_minus_1 ON mst_positions(t_minus_1_date);

CREATE TABLE IF NOT EXISTS mst_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_dt TEXT,
    leg_id INTEGER,
    order_id TEXT,
    side TEXT,
    qty INTEGER,
    price REAL,
    order_type TEXT,
    status TEXT,
    error_msg TEXT,
    paper_mode INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (leg_id) REFERENCES mst_positions(id)
);

CREATE TABLE IF NOT EXISTS mst_equity (
    date TEXT PRIMARY KEY,
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    open_legs INTEGER DEFAULT 0,
    state TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db() -> None:
    """Create tables if they don't exist. Idempotent."""
    with _conn() as con:
        con.executescript(SCHEMA_SQL)
    logger.info(f"MST DB initialized at {DB_PATH}")


# ---- Bar persistence ----

def save_bar(bar: dict[str, Any]) -> None:
    """Insert or update a 30-min bar with computed indicators."""
    with _conn() as con:
        con.execute("""
            INSERT INTO mst_bars (bar_dt, open, high, low, close, atr21,
                                  st_upper, st_lower, st_value, direction,
                                  stoch_k, stoch_d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bar_dt) DO UPDATE SET
              open=excluded.open, high=excluded.high, low=excluded.low,
              close=excluded.close, atr21=excluded.atr21,
              st_upper=excluded.st_upper, st_lower=excluded.st_lower,
              st_value=excluded.st_value, direction=excluded.direction,
              stoch_k=excluded.stoch_k, stoch_d=excluded.stoch_d
        """, (
            bar["bar_dt"], bar["open"], bar["high"], bar["low"], bar["close"],
            bar.get("atr21"), bar.get("st_upper"), bar.get("st_lower"),
            bar.get("st_value"), bar.get("direction"),
            bar.get("stoch_k"), bar.get("stoch_d"),
        ))


def get_recent_bars(limit: int = 200) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM mst_bars ORDER BY bar_dt DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_last_bar() -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM mst_bars ORDER BY bar_dt DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


# ---- Events ----

def log_event(event_type: str, direction: Optional[int], bar_dt: str,
              price: Optional[float] = None, flip_high: Optional[float] = None,
              flip_low: Optional[float] = None, pyramid_level: Optional[int] = None,
              notes: str = "") -> int:
    with _conn() as con:
        cur = con.execute("""
            INSERT INTO mst_events (event_type, direction, bar_dt, price,
                                    flip_high, flip_low, pyramid_level, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (event_type, direction, bar_dt, price, flip_high, flip_low,
              pyramid_level, notes))
        return cur.lastrowid


def get_recent_events(limit: int = 50, event_type: Optional[str] = None) -> list[dict]:
    with _conn() as con:
        if event_type:
            rows = con.execute(
                "SELECT * FROM mst_events WHERE event_type=? ORDER BY id DESC LIMIT ?",
                (event_type, limit)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM mst_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ---- Positions ----

def insert_position(pos: dict[str, Any]) -> int:
    with _conn() as con:
        cur = con.execute("""
            INSERT INTO mst_positions (
                week_label, direction, pyramid_level, leg_role, side,
                instrument_token, tradingsymbol, strike, option_type, qty,
                entry_price, entry_time, status, paper_mode, expiry_date,
                t_minus_1_date, order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pos["week_label"], pos["direction"], pos.get("pyramid_level", 1),
            pos["leg_role"], pos["side"], pos.get("instrument_token"),
            pos["tradingsymbol"], pos["strike"], pos["option_type"], pos["qty"],
            pos.get("entry_price"), pos.get("entry_time"),
            pos.get("status", "PENDING"), int(pos.get("paper_mode", False)),
            pos["expiry_date"], pos["t_minus_1_date"], pos.get("order_id"),
        ))
        return cur.lastrowid


def update_position(leg_id: int, **fields) -> None:
    if not fields:
        return
    keys = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [leg_id]
    with _conn() as con:
        con.execute(f"UPDATE mst_positions SET {keys} WHERE id=?", values)


def get_open_positions() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM mst_positions WHERE status='OPEN' ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_positions_for_t_minus_1(t_minus_1_date: str) -> list[dict]:
    """Return all OPEN positions whose T-1 day is the given date."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM mst_positions WHERE status='OPEN' AND t_minus_1_date=?",
            (t_minus_1_date,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_positions_today_closed(today_iso: str) -> list[dict]:
    """Return positions closed today (entry or exit happened today)."""
    with _conn() as con:
        rows = con.execute("""
            SELECT * FROM mst_positions WHERE status='CLOSED'
            AND DATE(exit_time) = ? ORDER BY id
        """, (today_iso,)).fetchall()
    return [dict(r) for r in rows]


# ---- Orders ----

def log_order(leg_id: Optional[int], order_id: Optional[str], side: str,
              qty: int, price: Optional[float], order_type: str, status: str,
              paper_mode: bool, bar_dt: Optional[str] = None,
              error_msg: Optional[str] = None) -> int:
    with _conn() as con:
        cur = con.execute("""
            INSERT INTO mst_orders (bar_dt, leg_id, order_id, side, qty,
                                    price, order_type, status, error_msg, paper_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (bar_dt, leg_id, order_id, side, qty, price, order_type,
              status, error_msg, int(paper_mode)))
        return cur.lastrowid


# ---- Equity ----

def upsert_equity(d: str, realized: float, unrealized: float, total: float,
                  open_legs: int, state: str, notes: str = "") -> None:
    with _conn() as con:
        con.execute("""
            INSERT INTO mst_equity (date, realized_pnl, unrealized_pnl, total_pnl,
                                    open_legs, state, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
              realized_pnl=excluded.realized_pnl,
              unrealized_pnl=excluded.unrealized_pnl,
              total_pnl=excluded.total_pnl,
              open_legs=excluded.open_legs,
              state=excluded.state,
              notes=excluded.notes
        """, (d, realized, unrealized, total, open_legs, state, notes))


def get_equity_curve(days: int = 30) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM mst_equity ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]
