"""
NAS ATM — Nifty ATM Strangle — Database Layer
================================================
SQLite persistence for intraday Nifty ATM options strangle system.
Pattern follows NasDB exactly, with nas_atm_ table prefix.

Tables:
  - nas_atm_state: single-row system state
  - nas_atm_positions: active/closed option positions
  - nas_atm_trades: completed trades with P&L
  - nas_atm_orders: order audit log
  - nas_atm_signals: signal event history
  - nas_atm_daily_state: EOD snapshots for equity curve
"""

import sqlite3
import threading
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)

import os
from pathlib import Path

_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
DATA_DIR = Path(_volume_path) / "data" if _volume_path else Path(__file__).parent.parent / "backtest_data"
DB_PATH = str(DATA_DIR / "nas_atm_trading.db")


class NasAtmDB:
    """SQLite persistence for NAS ATM intraday strangle trading system."""

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
                    -- System state (single row)
                    CREATE TABLE IF NOT EXISTS nas_atm_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        atr_value REAL,
                        atr_ma REAL,
                        is_squeezing INTEGER DEFAULT 0,
                        squeeze_count INTEGER DEFAULT 0,
                        spot_price REAL,
                        day_open REAL,
                        vix REAL,
                        daily_atr REAL,
                        daily_pnl REAL DEFAULT 0,
                        total_adjustments INTEGER DEFAULT 0,
                        last_scan_time TIMESTAMP,
                        last_signal_time TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    INSERT OR IGNORE INTO nas_atm_state (id) VALUES (1);

                    -- Active and closed positions
                    CREATE TABLE IF NOT EXISTS nas_atm_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strangle_id INTEGER,
                        leg VARCHAR(5) NOT NULL,
                        tradingsymbol VARCHAR(60) NOT NULL,
                        exchange VARCHAR(10) DEFAULT 'NFO',
                        transaction_type VARCHAR(5) NOT NULL,
                        instrument_type VARCHAR(5),
                        qty INTEGER NOT NULL,
                        strike REAL,
                        expiry_date DATE,
                        entry_price REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        exit_price REAL,
                        exit_time TIMESTAMP,
                        exit_reason VARCHAR(30),
                        sl_price REAL,
                        kite_order_id VARCHAR(50),
                        signal_type VARCHAR(30),
                        adjustment_count INTEGER DEFAULT 0,
                        status VARCHAR(20) DEFAULT 'PENDING',
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_nas_atm_pos_status
                        ON nas_atm_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_nas_atm_pos_strangle
                        ON nas_atm_positions(strangle_id, status);

                    -- Completed trades (closed position pairs)
                    CREATE TABLE IF NOT EXISTS nas_atm_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strangle_id INTEGER,
                        trade_date DATE NOT NULL,
                        entry_time TIMESTAMP,
                        exit_time TIMESTAMP,
                        spot_at_entry REAL,
                        spot_at_exit REAL,
                        call_strike REAL,
                        put_strike REAL,
                        call_entry_premium REAL,
                        put_entry_premium REAL,
                        call_exit_premium REAL,
                        put_exit_premium REAL,
                        total_premium_collected REAL,
                        total_premium_paid REAL,
                        gross_pnl REAL,
                        net_pnl REAL,
                        lots INTEGER,
                        adjustments INTEGER DEFAULT 0,
                        exit_reason VARCHAR(30),
                        expiry_date DATE,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_nas_atm_trades_date
                        ON nas_atm_trades(trade_date);

                    -- Order audit log
                    CREATE TABLE IF NOT EXISTS nas_atm_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tradingsymbol VARCHAR(60),
                        transaction_type VARCHAR(5),
                        qty INTEGER,
                        price REAL,
                        order_type VARCHAR(10) DEFAULT 'MARKET',
                        status VARCHAR(20),
                        kite_order_id VARCHAR(50),
                        position_id INTEGER,
                        signal_type VARCHAR(30),
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- Signal event history
                    CREATE TABLE IF NOT EXISTS nas_atm_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_type VARCHAR(30) NOT NULL,
                        atr_value REAL,
                        atr_ma REAL,
                        squeeze_count INTEGER,
                        spot_price REAL,
                        vix REAL,
                        call_strike REAL,
                        put_strike REAL,
                        call_premium REAL,
                        put_premium REAL,
                        action_taken TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- Daily state snapshots for equity curve
                    CREATE TABLE IF NOT EXISTS nas_atm_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_date DATE NOT NULL UNIQUE,
                        trades_taken INTEGER DEFAULT 0,
                        adjustments INTEGER DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        cumulative_pnl REAL DEFAULT 0,
                        win_count INTEGER DEFAULT 0,
                        loss_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
                logger.info(f"[NAS-ATM] Database initialized at {self.db_path}")
            finally:
                conn.close()

    # --- State ---------------------------------------------------------

    def get_state(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT * FROM nas_atm_state WHERE id=1").fetchone()
                return dict(row) if row else {}
            finally:
                conn.close()

    def update_state(self, **kwargs):
        if not kwargs:
            return
        kwargs['updated_at'] = datetime.now().isoformat()
        cols = ', '.join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values())
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE nas_atm_state SET {cols} WHERE id=1", vals)
                conn.commit()
            finally:
                conn.close()

    # --- Positions -----------------------------------------------------

    def add_position(self, **kwargs):
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO nas_atm_positions ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()))
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_active_positions(self, leg=None, strangle_id=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM nas_atm_positions WHERE status='ACTIVE'"
                params = []
                if leg:
                    sql += " AND leg=?"
                    params.append(leg)
                if strangle_id:
                    sql += " AND strangle_id=?"
                    params.append(strangle_id)
                sql += " ORDER BY created_at DESC"
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_positions_by_strangle(self, strangle_id):
        """Get all positions (any status) for a given strangle_id."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM nas_atm_positions WHERE strangle_id=? ORDER BY id",
                    (strangle_id,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_closed_positions(self):
        """Get today's closed positions for display."""
        today = datetime.now().strftime('%Y-%m-%d')
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM nas_atm_positions WHERE status='CLOSED' AND created_at >= ? ORDER BY id",
                    (today,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def close_position(self, position_id, exit_price, exit_reason):
        now = datetime.now().isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE nas_atm_positions
                    SET status='CLOSED', exit_price=?, exit_time=?,
                        exit_reason=?, updated_at=?
                    WHERE id=?
                """, (exit_price, now, exit_reason, now, position_id))
                conn.commit()
            finally:
                conn.close()

    def update_position(self, position_id, **kwargs):
        if not kwargs:
            return
        kwargs['updated_at'] = datetime.now().isoformat()
        cols = ', '.join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [position_id]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE nas_atm_positions SET {cols} WHERE id=?", vals)
                conn.commit()
            finally:
                conn.close()

    def get_next_strangle_id(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COALESCE(MAX(strangle_id), 0) + 1 AS next_id FROM nas_atm_positions"
                ).fetchone()
                return row['next_id']
            finally:
                conn.close()

    # --- Orders --------------------------------------------------------

    def log_order(self, **kwargs):
        kwargs['created_at'] = datetime.now().isoformat()
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO nas_atm_orders ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()))
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_today_order_count(self):
        today = date.today().isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM nas_atm_orders WHERE created_at >= ?",
                    (today,)).fetchone()
                return row['cnt']
            finally:
                conn.close()

    # --- Signals -------------------------------------------------------

    def log_signal(self, **kwargs):
        kwargs['created_at'] = datetime.now().isoformat()
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"INSERT INTO nas_atm_signals ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()))
                conn.commit()
            finally:
                conn.close()

    def get_recent_signals(self, limit=20):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM nas_atm_signals ORDER BY created_at DESC LIMIT ?",
                    (limit,)).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # --- Trades --------------------------------------------------------

    def add_trade(self, **kwargs):
        kwargs['created_at'] = datetime.now().isoformat()
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO nas_atm_trades ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()))
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_recent_trades(self, limit=50):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM nas_atm_trades ORDER BY created_at DESC LIMIT ?",
                    (limit,)).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # --- Stats ---------------------------------------------------------

    def get_stats(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                trades = conn.execute("SELECT * FROM nas_atm_trades").fetchall()
                if not trades:
                    return {
                        'total_trades': 0, 'winners': 0, 'losers': 0,
                        'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
                        'max_win': 0, 'max_loss': 0, 'profit_factor': 0,
                        'total_adjustments': 0,
                    }
                trades = [dict(t) for t in trades]
                wins = [t for t in trades if (t.get('net_pnl') or 0) > 0]
                losses = [t for t in trades if (t.get('net_pnl') or 0) <= 0]
                pnls = [t.get('net_pnl', 0) or 0 for t in trades]
                total_pnl = sum(pnls)
                gross_wins = sum(t.get('net_pnl', 0) or 0 for t in wins)
                gross_losses = abs(sum(t.get('net_pnl', 0) or 0 for t in losses))

                return {
                    'total_trades': len(trades),
                    'winners': len(wins),
                    'losers': len(losses),
                    'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
                    'total_pnl': round(total_pnl, 2),
                    'avg_pnl': round(total_pnl / len(trades), 2) if trades else 0,
                    'max_win': round(max(pnls), 2) if pnls else 0,
                    'max_loss': round(min(pnls), 2) if pnls else 0,
                    'profit_factor': round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0,
                    'total_adjustments': sum(t.get('adjustments', 0) or 0 for t in trades),
                }
            finally:
                conn.close()

    # --- Daily State ---------------------------------------------------

    def save_daily_state(self, **kwargs):
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        update_cols = ', '.join(f"{k}=excluded.{k}" for k in kwargs if k != 'trade_date')
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"""
                    INSERT INTO nas_atm_daily_state ({cols}) VALUES ({placeholders})
                    ON CONFLICT(trade_date) DO UPDATE SET {update_cols}
                """, list(kwargs.values()))
                conn.commit()
            finally:
                conn.close()

    def get_equity_curve(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM nas_atm_daily_state ORDER BY trade_date"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()


# --- Singleton ---------------------------------------------------------

_instance = None
_instance_lock = threading.Lock()


def get_nas_atm_db():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NasAtmDB()
    return _instance
