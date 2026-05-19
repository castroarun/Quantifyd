"""Pair-Trading (Config D) — Database Layer.

SQLite persistence for the 6-pair cointegrated pair-trading engine.
Pattern follows OrbDB / Intraday75WrDB exactly (singleton, thread-safe, WAL).

DB file: backtest_data/pair_trading.db

Tables:
  - pair_daily_state          — per-day mode, capital, P&L summary
  - pair_positions            — open + closed pair-positions (BOTH legs together)
  - pair_trades               — closed pair-trades audit (denormalised reports)
  - pair_orders               — Kite order audit per leg (paper or live)
  - pair_equity_curve         — daily NAV
  - pair_signals_log          — every signal evaluation (z-score per pair per day)
  - pair_alpha_beta_history   — rolling re-fit log for quarterly cohort refresh
"""

from __future__ import annotations

import os
import sqlite3
import threading
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
DATA_DIR = (
    Path(_volume_path) / "data"
    if _volume_path
    else Path(__file__).parent.parent.parent / "backtest_data"
)
DB_PATH = str(DATA_DIR / "pair_trading.db")


class PairTradingDB:
    """SQLite layer for the 6-pair cointegrated pair-trading engine."""

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
                conn.executescript(
                    """
                    -- Per-day mode + capital + P&L summary
                    CREATE TABLE IF NOT EXISTS pair_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_date DATE NOT NULL,
                        mode TEXT NOT NULL DEFAULT 'paper',
                        capital REAL,
                        realized_pnl REAL DEFAULT 0,
                        unrealized_pnl REAL DEFAULT 0,
                        open_pairs INTEGER DEFAULT 0,
                        scan_completed INTEGER DEFAULT 0,
                        weekly_sl_count INTEGER DEFAULT 0,
                        circuit_breaker_active INTEGER DEFAULT 0,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(trade_date)
                    );
                    CREATE INDEX IF NOT EXISTS idx_pair_daily_date
                        ON pair_daily_state(trade_date);

                    -- Pair-positions: each row = one pair-trade with BOTH legs
                    -- Convention: direction = +1 (long_spread = long A, short B)
                    --             direction = -1 (short_spread = short A, long B)
                    CREATE TABLE IF NOT EXISTS pair_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair_name TEXT NOT NULL,
                        symA TEXT NOT NULL,
                        symB TEXT NOT NULL,
                        direction INTEGER NOT NULL,
                        entry_date DATE NOT NULL,
                        entry_z REAL NOT NULL,
                        target_z REAL DEFAULT 0,
                        stop_z REAL,
                        hold_cap_days INTEGER,
                        lookback INTEGER,
                        alpha REAL,
                        beta REAL,
                        -- Leg A
                        legA_tradingsymbol TEXT,
                        legA_qty INTEGER,
                        legA_entry_price REAL,
                        legA_lot_size INTEGER,
                        legA_kite_order_id TEXT,
                        -- Leg B
                        legB_tradingsymbol TEXT,
                        legB_qty INTEGER,
                        legB_entry_price REAL,
                        legB_lot_size INTEGER,
                        legB_kite_order_id TEXT,
                        -- exit
                        exit_date DATE,
                        exit_z REAL,
                        legA_exit_price REAL,
                        legB_exit_price REAL,
                        legA_kite_exit_order_id TEXT,
                        legB_kite_exit_order_id TEXT,
                        exit_reason TEXT,
                        gross_pnl_inr REAL,
                        cost_inr REAL,
                        net_pnl_inr REAL,
                        days_held INTEGER,
                        status TEXT DEFAULT 'OPEN',
                        paper_mode INTEGER DEFAULT 1,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_pair_pos_status
                        ON pair_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_pair_pos_pair
                        ON pair_positions(pair_name);
                    CREATE INDEX IF NOT EXISTS idx_pair_pos_entry
                        ON pair_positions(entry_date);

                    -- Closed-trade audit (denormalised view for the journal)
                    CREATE TABLE IF NOT EXISTS pair_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER NOT NULL,
                        pair_name TEXT NOT NULL,
                        direction INTEGER NOT NULL,
                        entry_date DATE NOT NULL,
                        exit_date DATE NOT NULL,
                        days_held INTEGER,
                        entry_z REAL,
                        exit_z REAL,
                        gross_pnl_inr REAL,
                        cost_inr REAL,
                        net_pnl_inr REAL,
                        exit_reason TEXT,
                        paper_mode INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(position_id) REFERENCES pair_positions(id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_pair_trades_pair
                        ON pair_trades(pair_name);
                    CREATE INDEX IF NOT EXISTS idx_pair_trades_exit
                        ON pair_trades(exit_date);

                    -- Order audit per leg (PAPER fills get fake order_id)
                    CREATE TABLE IF NOT EXISTS pair_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER,
                        pair_name TEXT NOT NULL,
                        leg TEXT NOT NULL,                  -- 'A' or 'B'
                        tradingsymbol TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,     -- BUY or SELL
                        qty INTEGER NOT NULL,
                        order_type TEXT DEFAULT 'MARKET',
                        price REAL,
                        kite_order_id TEXT,
                        status TEXT DEFAULT 'PLACED',
                        exchange TEXT DEFAULT 'NFO',
                        product TEXT DEFAULT 'NRML',
                        paper_mode INTEGER DEFAULT 1,
                        leg_role TEXT,                      -- 'ENTRY' or 'EXIT'
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_pair_orders_pos
                        ON pair_orders(position_id);

                    -- Daily NAV (for the equity curve chart)
                    CREATE TABLE IF NOT EXISTS pair_equity_curve (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_date DATE NOT NULL,
                        nav REAL NOT NULL,
                        realized_pnl REAL DEFAULT 0,
                        unrealized_pnl REAL DEFAULT 0,
                        open_pairs INTEGER DEFAULT 0,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(trade_date)
                    );

                    -- Signal-evaluation log (one row per pair per scan)
                    CREATE TABLE IF NOT EXISTS pair_signals_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_time TIMESTAMP NOT NULL,
                        trade_date DATE NOT NULL,
                        pair_name TEXT NOT NULL,
                        symA TEXT, symB TEXT,
                        priceA REAL, priceB REAL,
                        spread REAL,
                        spread_mu REAL, spread_sd REAL,
                        z REAL,
                        action TEXT NOT NULL,           -- ENTRY_LONG, ENTRY_SHORT, EXIT_MR, EXIT_STOP, EXIT_TIME, NO_ACTION, BLOCKED
                        block_reason TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_pair_sig_date
                        ON pair_signals_log(trade_date);
                    CREATE INDEX IF NOT EXISTS idx_pair_sig_pair
                        ON pair_signals_log(pair_name);

                    -- Rolling alpha/beta re-fit log (quarterly cohort refresh)
                    CREATE TABLE IF NOT EXISTS pair_alpha_beta_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        refit_date DATE NOT NULL,
                        pair_name TEXT NOT NULL,
                        symA TEXT NOT NULL,
                        symB TEXT NOT NULL,
                        alpha REAL,
                        beta REAL,
                        window_days INTEGER,
                        cointegration_pvalue REAL,
                        half_life REAL,
                        applied INTEGER DEFAULT 0,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_pair_ab_date
                        ON pair_alpha_beta_history(refit_date);
                    """
                )
                conn.commit()
                logger.info(f"[PairTrading] Database initialized at {self.db_path}")
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Daily state
    # ------------------------------------------------------------------

    def get_or_create_daily_state(self, trade_date) -> Dict[str, Any]:
        if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
            trade_date = trade_date.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM pair_daily_state WHERE trade_date=?",
                    (trade_date,),
                ).fetchone()
                if row:
                    return dict(row)
                conn.execute(
                    "INSERT INTO pair_daily_state (trade_date) VALUES (?)",
                    (trade_date,),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM pair_daily_state WHERE trade_date=?",
                    (trade_date,),
                ).fetchone()
                return dict(row)
            finally:
                conn.close()

    def update_daily_state(self, trade_date, **kwargs):
        if not kwargs:
            return
        if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
            trade_date = trade_date.isoformat()
        cols = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [trade_date]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"UPDATE pair_daily_state SET {cols} WHERE trade_date=?",
                    vals,
                )
                conn.commit()
            finally:
                conn.close()

    def get_recent_daily_states(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM pair_daily_state ORDER BY trade_date DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Pair-positions
    # ------------------------------------------------------------------

    def add_position(self, **kwargs) -> int:
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO pair_positions ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def update_position(self, position_id: int, **kwargs):
        if not kwargs:
            return
        cols = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [position_id]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"UPDATE pair_positions SET {cols} WHERE id=?", vals
                )
                conn.commit()
            finally:
                conn.close()

    def close_position(
        self,
        position_id: int,
        exit_date,
        exit_z: float,
        legA_exit_price: float,
        legB_exit_price: float,
        exit_reason: str,
        gross_pnl_inr: float,
        cost_inr: float,
        net_pnl_inr: float,
        days_held: int,
        legA_kite_exit_order_id: Optional[str] = None,
        legB_kite_exit_order_id: Optional[str] = None,
    ):
        if isinstance(exit_date, date) and not isinstance(exit_date, datetime):
            exit_date = exit_date.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE pair_positions
                       SET status='CLOSED',
                           exit_date=?, exit_z=?,
                           legA_exit_price=?, legB_exit_price=?,
                           legA_kite_exit_order_id=?, legB_kite_exit_order_id=?,
                           exit_reason=?,
                           gross_pnl_inr=?, cost_inr=?, net_pnl_inr=?,
                           days_held=?
                       WHERE id=?""",
                    (
                        exit_date, exit_z,
                        legA_exit_price, legB_exit_price,
                        legA_kite_exit_order_id, legB_kite_exit_order_id,
                        exit_reason,
                        gross_pnl_inr, cost_inr, net_pnl_inr,
                        days_held,
                        position_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_open_positions(self, pair_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.db_lock:
            conn = self._get_conn()
            try:
                if pair_name:
                    rows = conn.execute(
                        "SELECT * FROM pair_positions WHERE status='OPEN' AND pair_name=? ORDER BY entry_date",
                        (pair_name,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM pair_positions WHERE status='OPEN' ORDER BY entry_date"
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_position_by_id(self, position_id: int) -> Optional[Dict[str, Any]]:
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM pair_positions WHERE id=?", (position_id,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_recent_closed_positions(self, limit: int = 50,
                                    pair_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.db_lock:
            conn = self._get_conn()
            try:
                if pair_name:
                    rows = conn.execute(
                        "SELECT * FROM pair_positions WHERE status='CLOSED' AND pair_name=? "
                        "ORDER BY exit_date DESC LIMIT ?",
                        (pair_name, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM pair_positions WHERE status='CLOSED' "
                        "ORDER BY exit_date DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def add_trade(self, **kwargs) -> int:
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO pair_trades ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def list_trades(self, date_from: Optional[str] = None, date_to: Optional[str] = None,
                    pair_name: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM pair_trades WHERE 1=1"
        params: List[Any] = []
        if date_from:
            sql += " AND exit_date >= ?"
            params.append(date_from)
        if date_to:
            sql += " AND exit_date <= ?"
            params.append(date_to)
        if pair_name:
            sql += " AND pair_name = ?"
            params.append(pair_name)
        sql += " ORDER BY exit_date DESC LIMIT ?"
        params.append(limit)
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def add_order(self, **kwargs) -> int:
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO pair_orders ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def list_orders(self, position_id: Optional[int] = None,
                    limit: int = 200) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM pair_orders WHERE 1=1"
        params: List[Any] = []
        if position_id is not None:
            sql += " AND position_id = ?"
            params.append(position_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def upsert_equity_curve_point(self, trade_date, nav: float,
                                   realized_pnl: float = 0.0,
                                   unrealized_pnl: float = 0.0,
                                   open_pairs: int = 0):
        if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
            trade_date = trade_date.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO pair_equity_curve
                         (trade_date, nav, realized_pnl, unrealized_pnl, open_pairs)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(trade_date) DO UPDATE SET
                         nav=excluded.nav,
                         realized_pnl=excluded.realized_pnl,
                         unrealized_pnl=excluded.unrealized_pnl,
                         open_pairs=excluded.open_pairs""",
                    (trade_date, nav, realized_pnl, unrealized_pnl, open_pairs),
                )
                conn.commit()
            finally:
                conn.close()

    def get_equity_curve(self, date_from: Optional[str] = None,
                         date_to: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM pair_equity_curve WHERE 1=1"
        params: List[Any] = []
        if date_from:
            sql += " AND trade_date >= ?"
            params.append(date_from)
        if date_to:
            sql += " AND trade_date <= ?"
            params.append(date_to)
        sql += " ORDER BY trade_date ASC"
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Signal log
    # ------------------------------------------------------------------

    def log_signal(self, **kwargs) -> int:
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO pair_signals_log ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def list_signals(self, trade_date: Optional[str] = None,
                     pair_name: Optional[str] = None,
                     limit: int = 500) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM pair_signals_log WHERE 1=1"
        params: List[Any] = []
        if trade_date:
            sql += " AND trade_date = ?"
            params.append(trade_date)
        if pair_name:
            sql += " AND pair_name = ?"
            params.append(pair_name)
        sql += " ORDER BY scan_time DESC LIMIT ?"
        params.append(limit)
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Alpha/beta history
    # ------------------------------------------------------------------

    def add_alpha_beta_refit(self, **kwargs) -> int:
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO pair_alpha_beta_history ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def list_alpha_beta_refits(self, pair_name: Optional[str] = None,
                                limit: int = 50) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM pair_alpha_beta_history WHERE 1=1"
        params: List[Any] = []
        if pair_name:
            sql += " AND pair_name = ?"
            params.append(pair_name)
        sql += " ORDER BY refit_date DESC LIMIT ?"
        params.append(limit)
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_db_instance: Optional[PairTradingDB] = None
_db_init_lock = threading.Lock()


def get_pair_trading_db() -> PairTradingDB:
    global _db_instance
    if _db_instance is None:
        with _db_init_lock:
            if _db_instance is None:
                _db_instance = PairTradingDB()
    return _db_instance
