"""Intraday 75WR — Database Layer

SQLite persistence for the three-system intraday engine.
Pattern follows OrbDB exactly (singleton, thread-safe, WAL).

DB file: backtest_data/intraday_75wr.db

Tables:
  - i75_daily_state    — per-system daily reset markers + nifty regime cache
  - i75_positions      — open + closed positions across all 3 systems
  - i75_trades         — closed-trade audit (denormalised for reports)
  - i75_orders         — Kite order audit log (paper + live)
  - i75_signals        — every signal evaluated (incl. paper, blocked, taken)
  - i75_equity_curve   — per-day P&L summary per system
"""

import os
import sqlite3
import threading
import logging
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)

_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
DATA_DIR = Path(_volume_path) / "data" if _volume_path else Path(__file__).parent.parent.parent / "backtest_data"
DB_PATH = str(DATA_DIR / "intraday_75wr.db")


class Intraday75WrDB:
    """SQLite layer for the three-system intraday-75WR engine."""

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
                    CREATE TABLE IF NOT EXISTS i75_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        trade_date DATE NOT NULL,
                        nifty_b3_change_pct REAL,
                        nifty_below_vwap_b3 INTEGER,
                        nifty_first_bearish INTEGER,
                        nifty_gap_pct REAL,
                        nifty_b6_close REAL,
                        nifty_day_open REAL,
                        nifty_filter_pass INTEGER,
                        signals_evaluated INTEGER DEFAULT 0,
                        signals_taken INTEGER DEFAULT 0,
                        signals_blocked INTEGER DEFAULT 0,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(system_id, trade_date)
                    );

                    CREATE TABLE IF NOT EXISTS i75_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        instrument TEXT NOT NULL,
                        trade_date DATE NOT NULL,
                        direction TEXT NOT NULL,
                        qty INTEGER NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        sl_price REAL NOT NULL,
                        target_price REAL NOT NULL,
                        exit_price REAL,
                        exit_time TIMESTAMP,
                        exit_reason TEXT,
                        pnl_pts REAL,
                        pnl_inr REAL,
                        kite_entry_order_id TEXT,
                        kite_exit_order_id TEXT,
                        kite_sl_order_id TEXT,
                        status TEXT DEFAULT 'OPEN',
                        paper_mode INTEGER DEFAULT 1,
                        signal_meta TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_i75_pos_status
                        ON i75_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_i75_pos_date
                        ON i75_positions(trade_date);
                    CREATE INDEX IF NOT EXISTS idx_i75_pos_system
                        ON i75_positions(system_id);

                    CREATE TABLE IF NOT EXISTS i75_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        instrument TEXT NOT NULL,
                        signal_time TIMESTAMP NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL,
                        sl_price REAL,
                        target_price REAL,
                        rsi REAL,
                        vwap REAL,
                        gap_pct REAL,
                        nifty_filter_value TEXT,
                        nifty_filter_pass INTEGER,
                        signal_meta TEXT,
                        action_taken TEXT,
                        block_reason TEXT,
                        paper_mode INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_i75_sig_date
                        ON i75_signals(signal_time);
                    CREATE INDEX IF NOT EXISTS idx_i75_sig_system
                        ON i75_signals(system_id);

                    CREATE TABLE IF NOT EXISTS i75_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        position_id INTEGER,
                        instrument TEXT NOT NULL,
                        tradingsymbol TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        qty INTEGER NOT NULL,
                        order_type TEXT DEFAULT 'MARKET',
                        price REAL,
                        trigger_price REAL,
                        kite_order_id TEXT,
                        status TEXT DEFAULT 'PLACED',
                        exchange TEXT DEFAULT 'NSE',
                        product TEXT DEFAULT 'MIS',
                        paper_mode INTEGER DEFAULT 1,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_i75_orders_pos
                        ON i75_orders(position_id);
                    CREATE INDEX IF NOT EXISTS idx_i75_orders_system
                        ON i75_orders(system_id);

                    CREATE TABLE IF NOT EXISTS i75_equity_curve (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        trade_date DATE NOT NULL,
                        daily_pnl REAL DEFAULT 0,
                        cumulative_pnl REAL DEFAULT 0,
                        trades_count INTEGER DEFAULT 0,
                        wins INTEGER DEFAULT 0,
                        losses INTEGER DEFAULT 0,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(system_id, trade_date)
                    );
                """)
                conn.commit()
                logger.info(f"[I75] DB initialized at {self.db_path}")
            finally:
                conn.close()

    # --- Daily state -------------------------------------------------------

    def get_or_create_daily_state(self, system_id, trade_date):
        if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
            trade_date = trade_date.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM i75_daily_state WHERE system_id=? AND trade_date=?",
                    (system_id, trade_date)
                ).fetchone()
                if row:
                    return dict(row)
                conn.execute(
                    "INSERT INTO i75_daily_state (system_id, trade_date) VALUES (?, ?)",
                    (system_id, trade_date)
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM i75_daily_state WHERE system_id=? AND trade_date=?",
                    (system_id, trade_date)
                ).fetchone()
                return dict(row)
            finally:
                conn.close()

    def update_daily_state(self, system_id, trade_date, **kwargs):
        if not kwargs:
            return
        if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
            trade_date = trade_date.isoformat()
        cols = ', '.join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [system_id, trade_date]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"UPDATE i75_daily_state SET {cols} WHERE system_id=? AND trade_date=?",
                    vals
                )
                conn.commit()
            finally:
                conn.close()

    def increment_counter(self, system_id, trade_date, field):
        """Increment one of: signals_evaluated / signals_taken / signals_blocked."""
        if field not in ('signals_evaluated', 'signals_taken', 'signals_blocked'):
            return
        if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
            trade_date = trade_date.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"UPDATE i75_daily_state SET {field}={field}+1 "
                    f"WHERE system_id=? AND trade_date=?",
                    (system_id, trade_date)
                )
                conn.commit()
            finally:
                conn.close()

    # --- Positions ----------------------------------------------------------

    def add_position(self, **kwargs):
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO i75_positions ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def update_position(self, position_id, **kwargs):
        if not kwargs:
            return
        cols = ', '.join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [position_id]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE i75_positions SET {cols} WHERE id=?", vals)
                conn.commit()
            finally:
                conn.close()

    def close_position(self, position_id, exit_price, exit_time, exit_reason,
                       pnl_pts, pnl_inr, kite_exit_order_id=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE i75_positions SET status='CLOSED', exit_price=?, "
                    "exit_time=?, exit_reason=?, pnl_pts=?, pnl_inr=?, "
                    "kite_exit_order_id=? WHERE id=?",
                    (exit_price, exit_time, exit_reason, pnl_pts, pnl_inr,
                     kite_exit_order_id, position_id)
                )
                conn.commit()
            finally:
                conn.close()

    def get_open_positions(self, system_id=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM i75_positions WHERE status='OPEN'"
                params = []
                if system_id:
                    sql += " AND system_id=?"
                    params.append(system_id)
                sql += " ORDER BY entry_time DESC"
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def count_open_positions_all_systems(self):
        """Count of OPEN positions across all 3 systems (combined cap check)."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM i75_positions WHERE status='OPEN'"
                ).fetchone()
                return int(row['cnt']) if row else 0
            finally:
                conn.close()

    def get_today_positions(self, system_id=None):
        today = date.today().isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM i75_positions WHERE trade_date=?"
                params = [today]
                if system_id:
                    sql += " AND system_id=?"
                    params.append(system_id)
                sql += " ORDER BY entry_time"
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_closed(self, system_id=None):
        today = date.today().isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM i75_positions WHERE status='CLOSED' AND trade_date=?"
                params = [today]
                if system_id:
                    sql += " AND system_id=?"
                    params.append(system_id)
                sql += " ORDER BY exit_time"
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def has_position_today(self, system_id, instrument):
        """True if this stock already has a position today (any status) for this system.
        Used to enforce one-trade-per-stock-per-day-per-system."""
        today = date.today().isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT id FROM i75_positions WHERE system_id=? AND "
                    "instrument=? AND trade_date=? LIMIT 1",
                    (system_id, instrument, today)
                ).fetchone()
                return row is not None
            finally:
                conn.close()

    # --- Signals ------------------------------------------------------------

    def log_signal(self, **kwargs):
        kwargs.setdefault('created_at', datetime.now().isoformat())
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO i75_signals ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_recent_signals(self, system_id=None, limit=50, day=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM i75_signals WHERE 1=1"
                params = []
                if system_id:
                    sql += " AND system_id=?"
                    params.append(system_id)
                if day:
                    sql += " AND DATE(signal_time)=?"
                    params.append(day)
                sql += " ORDER BY signal_time DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # --- Orders -------------------------------------------------------------

    def log_order(self, **kwargs):
        kwargs.setdefault('created_at', datetime.now().isoformat())
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO i75_orders ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    # --- Trades / equity curve ---------------------------------------------

    def get_recent_trades(self, system_id=None, limit=50):
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM i75_positions WHERE status='CLOSED'"
                params = []
                if system_id:
                    sql += " AND system_id=?"
                    params.append(system_id)
                sql += " ORDER BY exit_time DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_stats(self, system_id=None):
        """Aggregate trading statistics from all closed positions."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM i75_positions WHERE status='CLOSED'"
                params = []
                if system_id:
                    sql += " AND system_id=?"
                    params.append(system_id)
                rows = conn.execute(sql, params).fetchall()
                if not rows:
                    return {
                        'total_trades': 0, 'winners': 0, 'losers': 0,
                        'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
                        'max_win': 0, 'max_loss': 0, 'profit_factor': 0,
                    }
                trades = [dict(r) for r in rows]
                pnls = [t.get('pnl_inr', 0) or 0 for t in trades]
                wins = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p <= 0]
                total_pnl = sum(pnls)
                gross_wins = sum(wins)
                gross_losses = abs(sum(losses))
                return {
                    'total_trades': len(trades),
                    'winners': len(wins),
                    'losers': len(losses),
                    'win_rate': round(len(wins) / len(trades) * 100, 1),
                    'total_pnl': round(total_pnl, 2),
                    'avg_pnl': round(total_pnl / len(trades), 2),
                    'max_win': round(max(pnls), 2),
                    'max_loss': round(min(pnls), 2),
                    'profit_factor': round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0,
                }
            finally:
                conn.close()

    def get_equity_curve(self, system_id=None, start=None, end=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = ("SELECT trade_date, "
                       "SUM(pnl_inr) AS daily_pnl, "
                       "COUNT(*) AS trades, "
                       "SUM(CASE WHEN pnl_inr > 0 THEN 1 ELSE 0 END) AS wins, "
                       "SUM(CASE WHEN pnl_inr <= 0 THEN 1 ELSE 0 END) AS losses "
                       "FROM i75_positions WHERE status='CLOSED'")
                params = []
                if system_id:
                    sql += " AND system_id=?"
                    params.append(system_id)
                if start:
                    sql += " AND trade_date >= ?"
                    params.append(start)
                if end:
                    sql += " AND trade_date <= ?"
                    params.append(end)
                sql += " GROUP BY trade_date ORDER BY trade_date"
                rows = conn.execute(sql, params).fetchall()
                curve = []
                cumulative = 0.0
                for r in rows:
                    d = dict(r)
                    daily = d.get('daily_pnl') or 0
                    cumulative += daily
                    d['daily_pnl'] = round(daily, 2)
                    d['cumulative_pnl'] = round(cumulative, 2)
                    curve.append(d)
                return curve
            finally:
                conn.close()


# --- Singleton ---------------------------------------------------------

_instance = None
_instance_lock = threading.Lock()


def get_i75_db():
    """Return process-wide singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Intraday75WrDB()
    return _instance
