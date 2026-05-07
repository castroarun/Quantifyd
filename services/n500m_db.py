"""Nifty 500 Intraday Momentum — SQLite persistence.

Tables:
  - n500m_settings:      singleton key/value (mode = OFF/PAPER/LIVE, etc.)
  - n500m_daily_state:   per-(symbol, signal_type, date) setup data
  - n500m_signals:       every signal evaluation (fired or skipped + reason)
  - n500m_positions:     active/closed intraday positions
  - n500m_orders:        order audit log (PAPER- prefixed in paper mode)
  - n500m_equity:        daily NAV snapshot for the equity-curve chart
"""
from __future__ import annotations

import os
import sqlite3
import threading
import logging
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)

_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
DATA_DIR = (Path(_volume_path) / "data") if _volume_path else (Path(__file__).parent.parent / "backtest_data")
DB_PATH = str(DATA_DIR / "n500m_trading.db")


class N500mDB:
    """SQLite persistence for the Nifty 500 Intraday Momentum scanner."""

    def __init__(self):
        self.db_path = DB_PATH
        self.db_lock = threading.Lock()
        self._init_database()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_database(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS n500m_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS n500m_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,            -- 'ccrb' / 'volbo'
                        trade_date DATE NOT NULL,
                        timeframe TEXT NOT NULL,
                        direction TEXT NOT NULL,              -- 'long' / 'short'
                        prev_day_high REAL,
                        prev_day_low REAL,
                        prev_day_close REAL,
                        prev_day_cpr_width_pct REAL,
                        prev_day_range_pct REAL,
                        today_open REAL,
                        today_cpr_width_pct REAL,
                        today_pivot REAL,
                        today_tc REAL,
                        today_bc REAL,
                        gap_pct REAL,
                        setup_qualifies INTEGER DEFAULT 0,    -- daily-bar gates
                        setup_reason TEXT,                    -- why qualifies / fails
                        signal_fired INTEGER DEFAULT 0,
                        UNIQUE(symbol, signal_type, trade_date, timeframe, direction)
                    );
                    CREATE INDEX IF NOT EXISTS idx_n500m_state_date
                        ON n500m_daily_state(trade_date);

                    CREATE TABLE IF NOT EXISTS n500m_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        trade_date DATE NOT NULL,
                        signal_time TIMESTAMP NOT NULL,
                        timeframe TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        sl_price REAL,
                        target_price REAL,
                        atr_pts REAL,
                        candle_open REAL,
                        candle_high REAL,
                        candle_low REAL,
                        candle_close REAL,
                        candle_volume INTEGER,
                        vm_ratio REAL,
                        action_taken TEXT,                    -- 'ENTERED' / 'SKIPPED:liquidity' / 'SKIPPED:max_concurrent' / 'SKIPPED:duplicate'
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_n500m_sig_date
                        ON n500m_signals(trade_date);

                    CREATE TABLE IF NOT EXISTS n500m_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        trade_date DATE NOT NULL,
                        timeframe TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        qty INTEGER NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        sl_price REAL,
                        target_price REAL,
                        atr_pts REAL,
                        exit_policy TEXT NOT NULL,
                        variant_raw TEXT,
                        expected_sharpe REAL,
                        exit_price REAL,
                        exit_time TIMESTAMP,
                        exit_reason TEXT,
                        pnl_pts REAL,
                        pnl_inr REAL,
                        kite_entry_order_id TEXT,
                        kite_sl_order_id TEXT,
                        kite_exit_order_id TEXT,
                        status TEXT DEFAULT 'OPEN',           -- OPEN / CLOSED / CANCELLED
                        mode TEXT NOT NULL DEFAULT 'PAPER',   -- PAPER / LIVE
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_n500m_pos_status
                        ON n500m_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_n500m_pos_date
                        ON n500m_positions(trade_date);

                    CREATE TABLE IF NOT EXISTS n500m_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER,
                        symbol TEXT NOT NULL,
                        tradingsymbol TEXT,
                        transaction_type TEXT NOT NULL,
                        qty INTEGER NOT NULL,
                        order_type TEXT DEFAULT 'MARKET',
                        price REAL,
                        trigger_price REAL,
                        kite_order_id TEXT,
                        status TEXT DEFAULT 'PLACED',
                        exchange TEXT DEFAULT 'NSE',
                        product TEXT DEFAULT 'MIS',
                        mode TEXT NOT NULL DEFAULT 'PAPER',
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_n500m_ord_pos
                        ON n500m_orders(position_id);

                    CREATE TABLE IF NOT EXISTS n500m_equity (
                        trade_date DATE PRIMARY KEY,
                        starting_nav REAL,
                        ending_nav REAL,
                        realized_pnl REAL DEFAULT 0,
                        unrealized_pnl REAL DEFAULT 0,
                        n_trades INTEGER DEFAULT 0,
                        n_wins INTEGER DEFAULT 0,
                        mode TEXT NOT NULL DEFAULT 'PAPER',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                # Seed defaults
                conn.execute(
                    "INSERT OR IGNORE INTO n500m_settings(key, value) VALUES('mode', 'OFF')"
                )
                conn.execute(
                    "INSERT OR IGNORE INTO n500m_settings(key, value) VALUES('kill_switch', '0')"
                )
                conn.commit()
                logger.info(f"[N500M] Database initialized at {self.db_path}")
            finally:
                conn.close()

    # --- Settings ------------------------------------------------------

    def get_setting(self, key: str, default: str = "") -> str:
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT value FROM n500m_settings WHERE key=?", (key,)
                ).fetchone()
                return row["value"] if row else default
            finally:
                conn.close()

    def set_setting(self, key: str, value: str):
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO n500m_settings(key, value, updated_at) "
                    "VALUES(?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                    "updated_at=CURRENT_TIMESTAMP",
                    (key, str(value))
                )
                conn.commit()
            finally:
                conn.close()

    def get_mode(self) -> str:
        """Returns 'OFF' / 'PAPER' / 'LIVE'. Defaults to OFF (fail-closed)."""
        m = self.get_setting("mode", "OFF").upper()
        return m if m in ("OFF", "PAPER", "LIVE") else "OFF"

    def set_mode(self, mode: str):
        m = mode.upper()
        if m not in ("OFF", "PAPER", "LIVE"):
            raise ValueError(f"Invalid mode: {mode}")
        self.set_setting("mode", m)
        logger.warning(f"[N500M] Mode changed to {m}")

    def is_kill_switch(self) -> bool:
        return self.get_setting("kill_switch", "0") == "1"

    def set_kill_switch(self, on: bool):
        self.set_setting("kill_switch", "1" if on else "0")
        logger.warning(f"[N500M] Kill-switch {'ENGAGED' if on else 'released'}")

    # --- Positions -----------------------------------------------------

    def insert_position(self, **fields) -> int:
        cols = ", ".join(fields.keys())
        ph = ", ".join("?" * len(fields))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO n500m_positions ({cols}) VALUES ({ph})",
                    list(fields.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def update_position(self, position_id: int, **fields):
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"UPDATE n500m_positions SET {sets} WHERE id=?",
                    list(fields.values()) + [position_id]
                )
                conn.commit()
            finally:
                conn.close()

    def get_open_positions(self) -> list[dict]:
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM n500m_positions WHERE status='OPEN' "
                    "ORDER BY entry_time"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_positions(self, trade_date=None) -> list[dict]:
        if trade_date is None:
            trade_date = date.today().isoformat()
        elif isinstance(trade_date, date):
            trade_date = trade_date.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM n500m_positions WHERE trade_date=? "
                    "ORDER BY entry_time DESC",
                    (trade_date,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # --- Daily state / signals ----------------------------------------

    def upsert_daily_state(self, **fields) -> None:
        keys = list(fields.keys())
        cols = ", ".join(keys)
        ph = ", ".join("?" * len(keys))
        update_clause = ", ".join(
            f"{k}=excluded.{k}" for k in keys
            if k not in ("symbol", "signal_type", "trade_date", "timeframe", "direction")
        )
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"INSERT INTO n500m_daily_state ({cols}) VALUES ({ph}) "
                    f"ON CONFLICT(symbol, signal_type, trade_date, timeframe, direction) "
                    f"DO UPDATE SET {update_clause}",
                    list(fields.values())
                )
                conn.commit()
            finally:
                conn.close()

    def insert_signal(self, **fields) -> int:
        cols = ", ".join(fields.keys())
        ph = ", ".join("?" * len(fields))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO n500m_signals ({cols}) VALUES ({ph})",
                    list(fields.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_today_signals(self, trade_date=None) -> list[dict]:
        if trade_date is None:
            trade_date = date.today().isoformat()
        elif isinstance(trade_date, date):
            trade_date = trade_date.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM n500m_signals WHERE trade_date=? "
                    "ORDER BY signal_time DESC",
                    (trade_date,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # --- Orders --------------------------------------------------------

    def insert_order(self, **fields) -> int:
        cols = ", ".join(fields.keys())
        ph = ", ".join("?" * len(fields))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO n500m_orders ({cols}) VALUES ({ph})",
                    list(fields.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    # --- Equity curve --------------------------------------------------

    def upsert_equity(self, trade_date, **fields):
        if isinstance(trade_date, date):
            trade_date = trade_date.isoformat()
        fields["trade_date"] = trade_date
        keys = list(fields.keys())
        cols = ", ".join(keys)
        ph = ", ".join("?" * len(keys))
        update_clause = ", ".join(
            f"{k}=excluded.{k}" for k in keys if k != "trade_date"
        )
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"INSERT INTO n500m_equity ({cols}) VALUES ({ph}) "
                    f"ON CONFLICT(trade_date) DO UPDATE SET {update_clause}",
                    list(fields.values())
                )
                conn.commit()
            finally:
                conn.close()

    def get_equity_curve(self, mode: str = "PAPER", limit: int = 200) -> list[dict]:
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM n500m_equity WHERE mode=? "
                    "ORDER BY trade_date DESC LIMIT ?",
                    (mode, limit)
                ).fetchall()
                return [dict(r) for r in reversed(rows)]
            finally:
                conn.close()


_db_singleton: N500mDB | None = None
_db_lock = threading.Lock()


def get_db() -> N500mDB:
    global _db_singleton
    if _db_singleton is None:
        with _db_lock:
            if _db_singleton is None:
                _db_singleton = N500mDB()
    return _db_singleton
