"""
KC6 Trading State Database
============================

Persistence layer for the KC6 mean reversion live trading system.
Stores positions, completed trades, order audit log, and daily state.

Tables:
- kc6_positions: Active/closed positions
- kc6_trades: Completed trade history
- kc6_orders: Order audit log (paper + live)
- kc6_daily_state: Daily system state snapshot
"""

import sqlite3
import json
import threading
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

from config import DATA_DIR
DB_DIR = DATA_DIR
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / 'kc6_trading.db'

db_lock = threading.Lock()


class KC6TradingDB:
    """Manager for KC6 trading state persistence."""

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
                    CREATE TABLE IF NOT EXISTS kc6_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(20) NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_date DATE NOT NULL,
                        qty INTEGER NOT NULL,
                        sl_price REAL,
                        tp_price REAL,
                        kc6_mid_at_entry REAL,
                        kc6_mid_today REAL,
                        sma200_at_entry REAL,
                        target_order_id VARCHAR(50),
                        target_order_price REAL,
                        status VARCHAR(20) DEFAULT 'ACTIVE',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS kc6_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER,
                        symbol VARCHAR(20) NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_date DATE NOT NULL,
                        exit_price REAL NOT NULL,
                        exit_date DATE NOT NULL,
                        qty INTEGER NOT NULL,
                        pnl_pct REAL,
                        pnl_abs REAL,
                        hold_days INTEGER,
                        exit_reason VARCHAR(30),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (position_id) REFERENCES kc6_positions(id)
                    );

                    CREATE TABLE IF NOT EXISTS kc6_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(4) NOT NULL,
                        qty INTEGER NOT NULL,
                        price REAL,
                        order_type VARCHAR(10) DEFAULT 'LIMIT',
                        product VARCHAR(5) DEFAULT 'CNC',
                        status VARCHAR(20) DEFAULT 'PENDING',
                        kite_order_id VARCHAR(50),
                        position_id INTEGER,
                        exit_reason VARCHAR(30),
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (position_id) REFERENCES kc6_positions(id)
                    );

                    CREATE TABLE IF NOT EXISTS kc6_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_date DATE NOT NULL UNIQUE,
                        universe_atr_ratio REAL,
                        crash_filter_active BOOLEAN DEFAULT 0,
                        positions_count INTEGER DEFAULT 0,
                        entry_signals INTEGER DEFAULT 0,
                        exit_signals INTEGER DEFAULT 0,
                        orders_placed INTEGER DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_positions_status
                        ON kc6_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_positions_symbol
                        ON kc6_positions(symbol);
                    CREATE INDEX IF NOT EXISTS idx_trades_date
                        ON kc6_trades(exit_date);
                    CREATE INDEX IF NOT EXISTS idx_orders_status
                        ON kc6_orders(status);
                    CREATE INDEX IF NOT EXISTS idx_daily_date
                        ON kc6_daily_state(trade_date);
                """)
                # Schema migration: add target order columns if missing
                try:
                    conn.execute("SELECT target_order_id FROM kc6_positions LIMIT 1")
                except sqlite3.OperationalError:
                    conn.executescript("""
                        ALTER TABLE kc6_positions ADD COLUMN kc6_mid_today REAL;
                        ALTER TABLE kc6_positions ADD COLUMN target_order_id VARCHAR(50);
                        ALTER TABLE kc6_positions ADD COLUMN target_order_price REAL;
                    """)
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Positions
    # =========================================================================

    def add_position(self, symbol: str, entry_price: float, entry_date: str,
                     qty: int, sl_price: float, tp_price: float,
                     kc6_mid: float = None, sma200: float = None) -> int:
        """Add a new active position. Returns position ID."""
        with db_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("""
                    INSERT INTO kc6_positions
                    (symbol, entry_price, entry_date, qty, sl_price, tp_price,
                     kc6_mid_at_entry, sma200_at_entry, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
                """, (symbol, entry_price, entry_date, qty, sl_price, tp_price,
                      kc6_mid, sma200))
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def get_active_positions(self) -> List[Dict]:
        """Get all active positions."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM kc6_positions WHERE status='ACTIVE' ORDER BY entry_date"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_position_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get active position for a symbol (if any)."""
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM kc6_positions WHERE symbol=? AND status='ACTIVE' LIMIT 1",
                    (symbol,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def close_position(self, position_id: int, exit_price: float,
                       exit_date: str, exit_reason: str) -> Dict:
        """Close a position and record the trade."""
        with db_lock:
            conn = self._get_conn()
            try:
                pos = conn.execute(
                    "SELECT * FROM kc6_positions WHERE id=?", (position_id,)
                ).fetchone()
                if not pos:
                    raise ValueError(f"Position {position_id} not found")

                pos = dict(pos)
                entry_price = pos['entry_price']
                entry_date = pos['entry_date']
                qty = pos['qty']

                pnl_pct = round((exit_price / entry_price - 1) * 100, 2)
                pnl_abs = round((exit_price - entry_price) * qty, 2)

                entry_dt = datetime.strptime(entry_date, '%Y-%m-%d') if isinstance(entry_date, str) else entry_date
                exit_dt = datetime.strptime(exit_date, '%Y-%m-%d') if isinstance(exit_date, str) else exit_date
                hold_days = (exit_dt - entry_dt).days

                # Update position status
                conn.execute("""
                    UPDATE kc6_positions SET status='CLOSED', updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (position_id,))

                # Insert trade record
                conn.execute("""
                    INSERT INTO kc6_trades
                    (position_id, symbol, entry_price, entry_date, exit_price, exit_date,
                     qty, pnl_pct, pnl_abs, hold_days, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (position_id, pos['symbol'], entry_price, entry_date,
                      exit_price, exit_date, qty, pnl_pct, pnl_abs, hold_days, exit_reason))

                conn.commit()
                return {
                    'symbol': pos['symbol'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_abs': pnl_abs,
                    'hold_days': hold_days,
                    'exit_reason': exit_reason,
                }
            finally:
                conn.close()

    def get_active_positions_count(self) -> int:
        """Get count of active positions."""
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kc6_positions WHERE status='ACTIVE'"
                ).fetchone()
                return row['cnt']
            finally:
                conn.close()

    # =========================================================================
    # Target Order Tracking
    # =========================================================================

    def update_target_order(self, position_id: int, kite_order_id: str,
                            target_price: float, kc6_mid_today: float):
        """Update position with today's target limit order details."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE kc6_positions SET
                        target_order_id=?, target_order_price=?,
                        kc6_mid_today=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (kite_order_id, target_price, kc6_mid_today, position_id))
                conn.commit()
            finally:
                conn.close()

    def clear_target_order(self, position_id: int):
        """Clear target order info (after cancel or fill)."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE kc6_positions SET
                        target_order_id=NULL, target_order_price=NULL,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (position_id,))
                conn.commit()
            finally:
                conn.close()

    def get_positions_with_target_orders(self) -> List[Dict]:
        """Get active positions that have a standing target order."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM kc6_positions WHERE status='ACTIVE' AND target_order_id IS NOT NULL"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_positions_without_target_orders(self) -> List[Dict]:
        """Get active positions that need a target order placed."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM kc6_positions WHERE status='ACTIVE' AND target_order_id IS NULL"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # =========================================================================
    # Orders
    # =========================================================================

    def log_order(self, symbol: str, side: str, qty: int, price: float,
                  order_type: str = 'LIMIT', product: str = 'CNC',
                  status: str = 'PENDING', kite_order_id: str = None,
                  position_id: int = None, exit_reason: str = None) -> int:
        """Log an order (paper or live). Returns order ID."""
        with db_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("""
                    INSERT INTO kc6_orders
                    (symbol, side, qty, price, order_type, product, status,
                     kite_order_id, position_id, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, side, qty, price, order_type, product, status,
                      kite_order_id, position_id, exit_reason))
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def update_order_status(self, order_id: int, status: str,
                            kite_order_id: str = None, error_message: str = None):
        """Update order status after fill/rejection."""
        with db_lock:
            conn = self._get_conn()
            try:
                updates = ["status=?", "updated_at=CURRENT_TIMESTAMP"]
                params = [status]
                if kite_order_id:
                    updates.append("kite_order_id=?")
                    params.append(kite_order_id)
                if error_message:
                    updates.append("error_message=?")
                    params.append(error_message)
                params.append(order_id)
                conn.execute(
                    f"UPDATE kc6_orders SET {', '.join(updates)} WHERE id=?",
                    params
                )
                conn.commit()
            finally:
                conn.close()

    def get_pending_orders(self) -> List[Dict]:
        """Get all pending orders."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM kc6_orders WHERE status='PENDING' ORDER BY created_at"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_order_count(self) -> int:
        """Get number of orders placed today."""
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kc6_orders WHERE DATE(created_at)=DATE('now')"
                ).fetchone()
                return row['cnt']
            finally:
                conn.close()

    def get_recent_orders(self, limit: int = 50) -> List[Dict]:
        """Get recent orders for audit display."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM kc6_orders ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # =========================================================================
    # Trades
    # =========================================================================

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get completed trade history."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM kc6_trades ORDER BY exit_date DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_pnl(self) -> float:
        """Get total P/L from trades closed today."""
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(pnl_abs), 0) as total FROM kc6_trades WHERE DATE(exit_date)=DATE('now')"
                ).fetchone()
                return row['total']
            finally:
                conn.close()

    def get_today_loss_pct(self, capital: float) -> float:
        """Get today's loss as % of capital."""
        pnl = self.get_today_pnl()
        if pnl >= 0 or capital <= 0:
            return 0.0
        return abs(pnl / capital) * 100

    # =========================================================================
    # Daily State
    # =========================================================================

    def save_daily_state(self, trade_date: str, universe_atr_ratio: float,
                         crash_filter_active: bool, positions_count: int = 0,
                         entry_signals: int = 0, exit_signals: int = 0,
                         orders_placed: int = 0, daily_pnl: float = 0,
                         notes: str = None):
        """Save or update daily state snapshot."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO kc6_daily_state
                    (trade_date, universe_atr_ratio, crash_filter_active, positions_count,
                     entry_signals, exit_signals, orders_placed, daily_pnl, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trade_date) DO UPDATE SET
                        universe_atr_ratio=excluded.universe_atr_ratio,
                        crash_filter_active=excluded.crash_filter_active,
                        positions_count=excluded.positions_count,
                        entry_signals=excluded.entry_signals,
                        exit_signals=excluded.exit_signals,
                        orders_placed=excluded.orders_placed,
                        daily_pnl=excluded.daily_pnl,
                        notes=excluded.notes
                """, (trade_date, universe_atr_ratio, crash_filter_active,
                      positions_count, entry_signals, exit_signals,
                      orders_placed, daily_pnl, notes))
                conn.commit()
            finally:
                conn.close()

    def get_daily_state(self, trade_date: str = None) -> Optional[Dict]:
        """Get daily state for a date (defaults to today)."""
        if not trade_date:
            trade_date = date.today().isoformat()
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM kc6_daily_state WHERE trade_date=?", (trade_date,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_recent_daily_states(self, limit: int = 30) -> List[Dict]:
        """Get recent daily states for dashboard chart."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM kc6_daily_state ORDER BY trade_date DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get overall trading statistics."""
        with db_lock:
            conn = self._get_conn()
            try:
                total = conn.execute("SELECT COUNT(*) as cnt FROM kc6_trades").fetchone()['cnt']
                if total == 0:
                    return {'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0,
                            'total_pnl': 0, 'profit_factor': 0}

                wins = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kc6_trades WHERE pnl_pct > 0"
                ).fetchone()['cnt']
                avg_pnl = conn.execute(
                    "SELECT AVG(pnl_pct) as avg FROM kc6_trades"
                ).fetchone()['avg']
                total_pnl = conn.execute(
                    "SELECT SUM(pnl_abs) as total FROM kc6_trades"
                ).fetchone()['total']
                gross_profit = conn.execute(
                    "SELECT COALESCE(SUM(pnl_abs), 0) as gp FROM kc6_trades WHERE pnl_abs > 0"
                ).fetchone()['gp']
                gross_loss = conn.execute(
                    "SELECT COALESCE(ABS(SUM(pnl_abs)), 0) as gl FROM kc6_trades WHERE pnl_abs < 0"
                ).fetchone()['gl']

                pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

                return {
                    'total_trades': total,
                    'wins': wins,
                    'losses': total - wins,
                    'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                    'avg_pnl': round(avg_pnl, 2) if avg_pnl else 0,
                    'total_pnl': round(total_pnl, 2) if total_pnl else 0,
                    'profit_factor': round(pf, 2),
                }
            finally:
                conn.close()

    def get_equity_curve(self) -> List[Dict]:
        """Get cumulative P/L over time for equity curve chart."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT exit_date, pnl_abs, pnl_pct, symbol, exit_reason
                    FROM kc6_trades
                    ORDER BY exit_date ASC, id ASC
                """).fetchall()

                curve = []
                cumulative_pnl = 0
                for r in rows:
                    r = dict(r)
                    cumulative_pnl += r['pnl_abs']
                    curve.append({
                        'date': r['exit_date'],
                        'pnl': r['pnl_abs'],
                        'pnl_pct': r['pnl_pct'],
                        'cumulative_pnl': round(cumulative_pnl, 2),
                        'symbol': r['symbol'],
                        'exit_reason': r['exit_reason'],
                    })
                return curve
            finally:
                conn.close()


# Singleton
_instance = None
_instance_lock = threading.Lock()


def get_kc6_db() -> KC6TradingDB:
    """Get singleton KC6TradingDB instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = KC6TradingDB()
    return _instance
