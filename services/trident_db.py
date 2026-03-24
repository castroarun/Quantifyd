"""
Trident Trading State Database
================================

Persistence layer for the Trident live trading system (PA_MACD + RangeBreakout).
Stores positions, completed trades, order audit log, and daily state.
Modeled after KC6 DB pattern.
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
DB_PATH = DB_DIR / 'trident_trading.db'

db_lock = threading.Lock()


class TridentTradingDB:
    """Manager for Trident trading state persistence."""

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
                    CREATE TABLE IF NOT EXISTS trident_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(20) NOT NULL,
                        direction VARCHAR(5) NOT NULL,
                        strategy VARCHAR(20) NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_date DATE NOT NULL,
                        qty INTEGER NOT NULL,
                        sl_price REAL NOT NULL,
                        tp_price REAL NOT NULL,
                        max_hold_days INTEGER DEFAULT 10,
                        status VARCHAR(20) DEFAULT 'ACTIVE',
                        kite_order_id VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS trident_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER,
                        symbol VARCHAR(20) NOT NULL,
                        direction VARCHAR(5) NOT NULL,
                        strategy VARCHAR(20) NOT NULL,
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
                        FOREIGN KEY (position_id) REFERENCES trident_positions(id)
                    );

                    CREATE TABLE IF NOT EXISTS trident_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(5) NOT NULL,
                        direction VARCHAR(5),
                        strategy VARCHAR(20),
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
                        FOREIGN KEY (position_id) REFERENCES trident_positions(id)
                    );

                    CREATE TABLE IF NOT EXISTS trident_pending_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(20) NOT NULL,
                        direction VARCHAR(5) NOT NULL,
                        strategy VARCHAR(20) NOT NULL,
                        stop_level REAL NOT NULL,
                        sl_price REAL NOT NULL,
                        tp_price REAL NOT NULL,
                        max_hold_days INTEGER DEFAULT 10,
                        signal_date DATE NOT NULL,
                        status VARCHAR(20) DEFAULT 'PENDING',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS trident_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_date DATE NOT NULL UNIQUE,
                        positions_count INTEGER DEFAULT 0,
                        pamacd_signals INTEGER DEFAULT 0,
                        rb_signals INTEGER DEFAULT 0,
                        entries_placed INTEGER DEFAULT 0,
                        exits_placed INTEGER DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_tri_pos_status ON trident_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_tri_pos_symbol ON trident_positions(symbol);
                    CREATE INDEX IF NOT EXISTS idx_tri_trades_date ON trident_trades(exit_date);
                    CREATE INDEX IF NOT EXISTS idx_tri_orders_status ON trident_orders(status);
                    CREATE INDEX IF NOT EXISTS idx_tri_pending_status ON trident_pending_signals(status);
                """)
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Positions
    # =========================================================================

    def add_position(self, symbol: str, direction: str, strategy: str,
                     entry_price: float, entry_date: str, qty: int,
                     sl_price: float, tp_price: float, max_hold_days: int = 10,
                     kite_order_id: str = None) -> int:
        with db_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("""
                    INSERT INTO trident_positions
                    (symbol, direction, strategy, entry_price, entry_date, qty,
                     sl_price, tp_price, max_hold_days, kite_order_id, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
                """, (symbol, direction, strategy, entry_price, entry_date, qty,
                      sl_price, tp_price, max_hold_days, kite_order_id))
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def get_active_positions(self) -> List[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM trident_positions WHERE status='ACTIVE' ORDER BY entry_date"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_active_positions_count(self) -> int:
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM trident_positions WHERE status='ACTIVE'"
                ).fetchone()
                return row['cnt']
            finally:
                conn.close()

    def get_position_by_symbol(self, symbol: str) -> Optional[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM trident_positions WHERE symbol=? AND status='ACTIVE' LIMIT 1",
                    (symbol,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def close_position(self, position_id: int, exit_price: float,
                       exit_date: str, exit_reason: str) -> Dict:
        with db_lock:
            conn = self._get_conn()
            try:
                pos = conn.execute(
                    "SELECT * FROM trident_positions WHERE id=?", (position_id,)
                ).fetchone()
                if not pos:
                    raise ValueError(f"Position {position_id} not found")
                pos = dict(pos)

                entry = pos['entry_price']
                qty = pos['qty']
                direction = pos['direction']

                if direction == 'LONG':
                    pnl_abs = (exit_price - entry) * qty
                    pnl_pct = (exit_price - entry) / entry * 100
                else:
                    pnl_abs = (entry - exit_price) * qty
                    pnl_pct = (entry - exit_price) / entry * 100

                entry_dt = datetime.strptime(pos['entry_date'][:10], '%Y-%m-%d')
                exit_dt = datetime.strptime(exit_date[:10], '%Y-%m-%d')
                hold_days = (exit_dt - entry_dt).days

                # Record trade
                conn.execute("""
                    INSERT INTO trident_trades
                    (position_id, symbol, direction, strategy, entry_price, entry_date,
                     exit_price, exit_date, qty, pnl_pct, pnl_abs, hold_days, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (position_id, pos['symbol'], direction, pos['strategy'],
                      entry, pos['entry_date'], exit_price, exit_date,
                      qty, round(pnl_pct, 2), round(pnl_abs, 2),
                      hold_days, exit_reason))

                # Close position
                conn.execute("""
                    UPDATE trident_positions SET status='CLOSED', updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (position_id,))
                conn.commit()

                trade = {
                    'symbol': pos['symbol'], 'direction': direction,
                    'strategy': pos['strategy'], 'entry_price': entry,
                    'exit_price': exit_price, 'pnl_pct': round(pnl_pct, 2),
                    'pnl_abs': round(pnl_abs, 2), 'hold_days': hold_days,
                    'exit_reason': exit_reason,
                }
                logger.info(f"[Trident] Closed {pos['symbol']} {direction} "
                           f"PnL={pnl_pct:+.2f}% ({exit_reason})")
                return trade
            finally:
                conn.close()

    # =========================================================================
    # Pending Signals
    # =========================================================================

    def add_pending_signal(self, symbol: str, direction: str, strategy: str,
                           stop_level: float, sl_price: float, tp_price: float,
                           max_hold_days: int, signal_date: str) -> int:
        with db_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("""
                    INSERT INTO trident_pending_signals
                    (symbol, direction, strategy, stop_level, sl_price, tp_price,
                     max_hold_days, signal_date, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """, (symbol, direction, strategy, stop_level, sl_price, tp_price,
                      max_hold_days, signal_date))
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def get_pending_signals(self) -> List[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM trident_pending_signals WHERE status='PENDING'"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def expire_pending_signals(self, before_date: str):
        """Expire pending signals from previous days."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE trident_pending_signals
                    SET status='EXPIRED', updated_at=CURRENT_TIMESTAMP
                    WHERE status='PENDING' AND signal_date < ?
                """, (before_date,))
                conn.commit()
            finally:
                conn.close()

    def fill_pending_signal(self, signal_id: int):
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE trident_pending_signals SET status='FILLED' WHERE id=?
                """, (signal_id,))
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Orders (audit log)
    # =========================================================================

    def log_order(self, symbol: str, side: str, qty: int, price: float,
                  order_type: str = 'LIMIT', status: str = 'PAPER',
                  direction: str = None, strategy: str = None,
                  position_id: int = None, kite_order_id: str = None,
                  exit_reason: str = None, error_message: str = None) -> int:
        with db_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("""
                    INSERT INTO trident_orders
                    (symbol, side, direction, strategy, qty, price, order_type,
                     status, kite_order_id, position_id, exit_reason, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, side, direction, strategy, qty, price, order_type,
                      status, kite_order_id, position_id, exit_reason, error_message))
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    # =========================================================================
    # Trades (history)
    # =========================================================================

    def get_trades(self, limit: int = 50) -> List[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM trident_trades ORDER BY exit_date DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_order_count(self) -> int:
        today = date.today().isoformat()
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM trident_orders WHERE DATE(created_at)=?",
                    (today,)
                ).fetchone()
                return row['cnt']
            finally:
                conn.close()

    def get_today_loss_pct(self, capital: float) -> float:
        today = date.today().isoformat()
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(pnl_abs), 0) as total FROM trident_trades WHERE exit_date=?",
                    (today,)
                ).fetchone()
                total = row['total']
                if total >= 0:
                    return 0.0
                return abs(total) / capital * 100
            finally:
                conn.close()

    def get_stats(self) -> Dict:
        with db_lock:
            conn = self._get_conn()
            try:
                total = conn.execute("SELECT COUNT(*) as cnt FROM trident_trades").fetchone()['cnt']
                if total == 0:
                    return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                            'avg_pnl_pct': 0, 'total_pnl': 0}
                wins = conn.execute(
                    "SELECT COUNT(*) as cnt FROM trident_trades WHERE pnl_abs > 0"
                ).fetchone()['cnt']
                gross_profit = conn.execute(
                    "SELECT COALESCE(SUM(pnl_abs), 0) as total FROM trident_trades WHERE pnl_abs > 0"
                ).fetchone()['total']
                gross_loss = abs(conn.execute(
                    "SELECT COALESCE(SUM(pnl_abs), 0) as total FROM trident_trades WHERE pnl_abs <= 0"
                ).fetchone()['total'])
                avg_pnl = conn.execute(
                    "SELECT COALESCE(AVG(pnl_pct), 0) as avg FROM trident_trades"
                ).fetchone()['avg']
                total_pnl = conn.execute(
                    "SELECT COALESCE(SUM(pnl_abs), 0) as total FROM trident_trades"
                ).fetchone()['total']

                # Per-strategy stats
                strat_stats = {}
                for row in conn.execute("""
                    SELECT strategy, COUNT(*) as cnt,
                           SUM(CASE WHEN pnl_abs > 0 THEN 1 ELSE 0 END) as wins,
                           COALESCE(SUM(pnl_abs), 0) as pnl
                    FROM trident_trades GROUP BY strategy
                """).fetchall():
                    r = dict(row)
                    strat_stats[r['strategy']] = {
                        'trades': r['cnt'], 'wins': r['wins'],
                        'win_rate': round(r['wins'] / r['cnt'] * 100, 1) if r['cnt'] > 0 else 0,
                        'pnl': round(r['pnl'], 2),
                    }

                return {
                    'total_trades': total,
                    'wins': wins,
                    'win_rate': round(wins / total * 100, 1),
                    'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
                    'avg_pnl_pct': round(avg_pnl, 2),
                    'total_pnl': round(total_pnl, 2),
                    'by_strategy': strat_stats,
                }
            finally:
                conn.close()

    def get_equity_curve(self) -> List[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT exit_date as date, SUM(pnl_abs) as daily_pnl
                    FROM trident_trades GROUP BY exit_date ORDER BY exit_date
                """).fetchall()
                curve = []
                cumulative = 0
                for r in rows:
                    cumulative += r['daily_pnl']
                    curve.append({'date': r['date'], 'daily_pnl': round(r['daily_pnl'], 2),
                                  'cumulative_pnl': round(cumulative, 2)})
                return curve
            finally:
                conn.close()

    def save_daily_state(self, trade_date: str, **kwargs):
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO trident_daily_state
                    (trade_date, positions_count, pamacd_signals, rb_signals,
                     entries_placed, exits_placed, daily_pnl, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (trade_date, kwargs.get('positions_count', 0),
                      kwargs.get('pamacd_signals', 0), kwargs.get('rb_signals', 0),
                      kwargs.get('entries_placed', 0), kwargs.get('exits_placed', 0),
                      kwargs.get('daily_pnl', 0), kwargs.get('notes', '')))
                conn.commit()
            finally:
                conn.close()


# Singleton
_instance = None


def get_trident_db() -> TridentTradingDB:
    global _instance
    if _instance is None:
        _instance = TridentTradingDB()
    return _instance
