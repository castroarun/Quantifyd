"""
Maruthi Always-On Strategy - Database Layer
=============================================

SQLite persistence for the Maruthi dual-SuperTrend futures + options strategy.

Tables:
- maruthi_regime: Current regime state (BULL/BEAR/FLAT)
- maruthi_positions: Active and closed positions (futures, options)
- maruthi_trades: Completed trade history
- maruthi_orders: Order audit log
- maruthi_signals: Signal history (master/child flips)
- maruthi_daily_state: Daily system snapshot
"""

import sqlite3
import threading
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pathlib import Path

from config import DATA_DIR

logger = logging.getLogger(__name__)

DB_DIR = DATA_DIR
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / 'maruthi_trading.db'

db_lock = threading.Lock()


class MaruthiDB:
    """Persistence layer for Maruthi always-on strategy."""

    def __init__(self):
        self.db_path = DB_PATH
        self._init_database()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        with db_lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    -- Current regime state (single row, updated on master flips)
                    CREATE TABLE IF NOT EXISTS maruthi_regime (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        regime VARCHAR(10) DEFAULT 'FLAT',
                        master_st_value REAL,
                        master_direction INTEGER,
                        child_direction INTEGER,
                        hard_sl_price REAL,
                        regime_start_time TIMESTAMP,
                        last_signal_time TIMESTAMP,
                        last_candle_time TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- Insert default regime row if not exists
                    INSERT OR IGNORE INTO maruthi_regime (id, regime)
                    VALUES (1, 'FLAT');

                    -- Positions: futures, short options, protective options
                    CREATE TABLE IF NOT EXISTS maruthi_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_type VARCHAR(20) NOT NULL,
                        tradingsymbol VARCHAR(50) NOT NULL,
                        exchange VARCHAR(10) DEFAULT 'NFO',
                        transaction_type VARCHAR(5) NOT NULL,
                        qty INTEGER NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        trigger_price REAL,
                        sl_price REAL,
                        instrument_type VARCHAR(10),
                        strike REAL,
                        expiry_date DATE,
                        kite_order_id VARCHAR(50),
                        regime VARCHAR(10),
                        signal_type VARCHAR(30),
                        status VARCHAR(20) DEFAULT 'PENDING',
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_maruthi_pos_status
                        ON maruthi_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_maruthi_pos_type
                        ON maruthi_positions(position_type, status);

                    -- Completed trades
                    CREATE TABLE IF NOT EXISTS maruthi_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER,
                        position_type VARCHAR(20),
                        tradingsymbol VARCHAR(50),
                        transaction_type VARCHAR(5),
                        qty INTEGER,
                        entry_price REAL,
                        entry_time TIMESTAMP,
                        exit_price REAL,
                        exit_time TIMESTAMP,
                        pnl_abs REAL,
                        pnl_pct REAL,
                        exit_reason VARCHAR(30),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (position_id) REFERENCES maruthi_positions(id)
                    );

                    -- Order audit log
                    CREATE TABLE IF NOT EXISTS maruthi_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tradingsymbol VARCHAR(50) NOT NULL,
                        exchange VARCHAR(10) DEFAULT 'NFO',
                        transaction_type VARCHAR(5) NOT NULL,
                        qty INTEGER NOT NULL,
                        price REAL,
                        trigger_price REAL,
                        order_type VARCHAR(10) DEFAULT 'SL',
                        product VARCHAR(10) DEFAULT 'NRML',
                        variety VARCHAR(10) DEFAULT 'regular',
                        status VARCHAR(20) DEFAULT 'PENDING',
                        kite_order_id VARCHAR(50),
                        position_id INTEGER,
                        signal_type VARCHAR(30),
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (position_id) REFERENCES maruthi_positions(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_maruthi_ord_status
                        ON maruthi_orders(status);

                    -- Signal history
                    CREATE TABLE IF NOT EXISTS maruthi_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_type VARCHAR(30) NOT NULL,
                        regime VARCHAR(10),
                        master_direction INTEGER,
                        child_direction INTEGER,
                        master_st_value REAL,
                        child_st_value REAL,
                        candle_time TIMESTAMP,
                        candle_high REAL,
                        candle_low REAL,
                        candle_close REAL,
                        action_taken VARCHAR(50),
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- Daily state snapshot
                    CREATE TABLE IF NOT EXISTS maruthi_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_date DATE UNIQUE NOT NULL,
                        regime VARCHAR(10),
                        futures_count INTEGER DEFAULT 0,
                        short_options_count INTEGER DEFAULT 0,
                        protective_options_count INTEGER DEFAULT 0,
                        total_margin_used REAL,
                        daily_pnl REAL,
                        cumulative_pnl REAL,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Regime
    # =========================================================================

    def get_regime(self) -> dict:
        """Get current regime state."""
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT * FROM maruthi_regime WHERE id = 1").fetchone()
                return dict(row) if row else {'regime': 'FLAT'}
            finally:
                conn.close()

    def update_regime(self, **kwargs) -> None:
        """Update regime state. Pass any column as keyword arg."""
        if not kwargs:
            return
        kwargs['updated_at'] = datetime.now().isoformat()
        cols = ', '.join(f'{k} = ?' for k in kwargs)
        vals = list(kwargs.values())
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE maruthi_regime SET {cols} WHERE id = 1", vals)
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Positions
    # =========================================================================

    def add_position(self, **kwargs) -> int:
        """Add a new position. Returns position ID."""
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO maruthi_positions ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_active_positions(self, position_type: str = None) -> List[dict]:
        """Get active positions, optionally filtered by type."""
        with db_lock:
            conn = self._get_conn()
            try:
                if position_type:
                    rows = conn.execute(
                        "SELECT * FROM maruthi_positions WHERE status = 'ACTIVE' AND position_type = ? ORDER BY id",
                        (position_type,)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM maruthi_positions WHERE status = 'ACTIVE' ORDER BY id"
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_pending_positions(self) -> List[dict]:
        """Get positions with pending trigger orders."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM maruthi_positions WHERE status = 'PENDING' ORDER BY id"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def activate_position(self, position_id: int, entry_price: float) -> None:
        """Mark a pending position as active (trigger order filled)."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE maruthi_positions SET status = 'ACTIVE', entry_price = ?, updated_at = ? WHERE id = ?",
                    (entry_price, datetime.now().isoformat(), position_id)
                )
                conn.commit()
            finally:
                conn.close()

    def close_position(self, position_id: int, exit_price: float, exit_reason: str) -> None:
        """Close a position and record the trade."""
        with db_lock:
            conn = self._get_conn()
            try:
                pos = conn.execute(
                    "SELECT * FROM maruthi_positions WHERE id = ?", (position_id,)
                ).fetchone()
                if not pos:
                    return

                pos = dict(pos)
                now = datetime.now().isoformat()

                # Calculate PnL
                if pos['transaction_type'] == 'BUY':
                    pnl_abs = (exit_price - pos['entry_price']) * pos['qty']
                else:  # SELL
                    pnl_abs = (pos['entry_price'] - exit_price) * pos['qty']

                pnl_pct = (pnl_abs / (pos['entry_price'] * pos['qty'])) * 100 if pos['entry_price'] else 0

                # Update position status
                conn.execute(
                    "UPDATE maruthi_positions SET status = 'CLOSED', updated_at = ? WHERE id = ?",
                    (now, position_id)
                )

                # Insert trade record
                conn.execute("""
                    INSERT INTO maruthi_trades
                    (position_id, position_type, tradingsymbol, transaction_type, qty,
                     entry_price, entry_time, exit_price, exit_time, pnl_abs, pnl_pct, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position_id, pos['position_type'], pos['tradingsymbol'],
                    pos['transaction_type'], pos['qty'],
                    pos['entry_price'], pos['entry_time'],
                    exit_price, now, pnl_abs, pnl_pct, exit_reason
                ))
                conn.commit()
            finally:
                conn.close()

    def cancel_position(self, position_id: int) -> None:
        """Cancel a pending position (trigger never fired)."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE maruthi_positions SET status = 'CANCELLED', updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), position_id)
                )
                conn.commit()
            finally:
                conn.close()

    def update_position(self, position_id: int, **kwargs) -> None:
        """Update any fields on a position."""
        if not kwargs:
            return
        kwargs['updated_at'] = datetime.now().isoformat()
        cols = ', '.join(f'{k} = ?' for k in kwargs)
        vals = list(kwargs.values()) + [position_id]
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE maruthi_positions SET {cols} WHERE id = ?", vals)
                conn.commit()
            finally:
                conn.close()

    def get_active_futures_count(self) -> int:
        """Count active futures positions."""
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM maruthi_positions WHERE status = 'ACTIVE' AND position_type = 'FUTURES'"
                ).fetchone()
                return row['cnt'] if row else 0
            finally:
                conn.close()

    def get_last_short_option(self) -> Optional[dict]:
        """Get the most recent active short option position."""
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    """SELECT * FROM maruthi_positions
                       WHERE status = 'ACTIVE' AND position_type = 'SHORT_OPTION'
                       ORDER BY id DESC LIMIT 1"""
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    # =========================================================================
    # Orders
    # =========================================================================

    def log_order(self, **kwargs) -> int:
        """Log an order to audit trail. Returns order ID."""
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO maruthi_orders ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def update_order(self, order_id: int, **kwargs) -> None:
        """Update an order record."""
        if not kwargs:
            return
        kwargs['updated_at'] = datetime.now().isoformat()
        cols = ', '.join(f'{k} = ?' for k in kwargs)
        vals = list(kwargs.values()) + [order_id]
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE maruthi_orders SET {cols} WHERE id = ?", vals)
                conn.commit()
            finally:
                conn.close()

    def get_pending_orders(self) -> List[dict]:
        """Get orders awaiting fill verification."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM maruthi_orders WHERE status IN ('PENDING', 'PLACED') ORDER BY id"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_order_count(self) -> int:
        """Count orders placed today."""
        with db_lock:
            conn = self._get_conn()
            try:
                today = date.today().isoformat()
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM maruthi_orders WHERE DATE(created_at) = ?",
                    (today,)
                ).fetchone()
                return row['cnt'] if row else 0
            finally:
                conn.close()

    # =========================================================================
    # Signals
    # =========================================================================

    def log_signal(self, **kwargs) -> int:
        """Log a signal event."""
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO maruthi_signals ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_recent_signals(self, limit: int = 50) -> List[dict]:
        """Get recent signals."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM maruthi_signals ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # =========================================================================
    # Trades & Stats
    # =========================================================================

    def get_recent_trades(self, limit: int = 50) -> List[dict]:
        """Get recent completed trades."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM maruthi_trades ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_stats(self) -> dict:
        """Get aggregate trading statistics."""
        with db_lock:
            conn = self._get_conn()
            try:
                trades = conn.execute("SELECT * FROM maruthi_trades").fetchall()
                if not trades:
                    return {'total_trades': 0, 'total_pnl': 0, 'win_rate': 0}

                total = len(trades)
                wins = sum(1 for t in trades if t['pnl_abs'] > 0)
                total_pnl = sum(t['pnl_abs'] for t in trades)

                return {
                    'total_trades': total,
                    'wins': wins,
                    'losses': total - wins,
                    'win_rate': (wins / total * 100) if total else 0,
                    'total_pnl': total_pnl,
                }
            finally:
                conn.close()

    # =========================================================================
    # Daily State
    # =========================================================================

    def save_daily_state(self, **kwargs) -> None:
        """Upsert daily state snapshot."""
        today = date.today().isoformat()
        kwargs['trade_date'] = today
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        conflict_updates = ', '.join(f'{k} = excluded.{k}' for k in kwargs if k != 'trade_date')

        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"""INSERT INTO maruthi_daily_state ({cols}) VALUES ({placeholders})
                        ON CONFLICT(trade_date) DO UPDATE SET {conflict_updates}""",
                    list(kwargs.values())
                )
                conn.commit()
            finally:
                conn.close()

    def get_equity_curve(self) -> List[dict]:
        """Get daily PnL series for charting."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT trade_date, cumulative_pnl FROM maruthi_daily_state ORDER BY trade_date"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()


# Singleton
_instance = None
_instance_lock = threading.Lock()

def get_maruthi_db() -> MaruthiDB:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MaruthiDB()
    return _instance
