"""
Collar Trading State Database
==============================

Persistence layer for the Collar paper-trading system.
A "collar" = 3 legs per entry signal:
    1. Long 1 lot Futures (bullish base)
    2. Long 1 lot OTM Put (downside protection)
    3. Short 1 lot OTM Call (premium cap)

Mirrors the kc6_db.py singleton pattern. Paper-only - no live Kite orders.

Tables:
- collar_positions: One row per collar (aggregate)
- collar_legs: Normalized leg rows (FUT/PUT/CALL) per position
- collar_trades: Closed collar archive
- collar_daily_state: Daily system state snapshot
"""

import sqlite3
import threading
import logging
from datetime import datetime, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from config import DATA_DIR
DB_DIR = DATA_DIR
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / 'collar_trading.db'

db_lock = threading.Lock()


class CollarTradingDB:
    """Manager for Collar paper-trading state persistence."""

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
                    CREATE TABLE IF NOT EXISTS collar_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol VARCHAR(20) NOT NULL,
                        entry_date DATE NOT NULL,
                        spot_at_entry REAL NOT NULL,

                        future_strike REAL,
                        put_strike REAL NOT NULL,
                        call_strike REAL NOT NULL,
                        expiry_date DATE NOT NULL,

                        future_entry_price REAL NOT NULL,
                        put_entry_price REAL NOT NULL,
                        call_entry_price REAL NOT NULL,

                        lot_size INTEGER NOT NULL,
                        qty INTEGER NOT NULL,

                        kc6_mid_today REAL,
                        sl_price REAL,
                        tp_price REAL,

                        status VARCHAR(20) DEFAULT 'OPEN',
                        exit_date DATE,
                        future_exit_price REAL,
                        put_exit_price REAL,
                        call_exit_price REAL,
                        gross_pnl REAL,
                        net_pnl REAL,
                        exit_reason VARCHAR(30),

                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS collar_legs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER NOT NULL,
                        leg_type VARCHAR(4) NOT NULL,     -- FUT | PUT | CALL
                        side VARCHAR(4) NOT NULL,         -- BUY | SELL
                        strike REAL,                      -- NULL for FUT
                        qty INTEGER NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        entry_date DATE NOT NULL,
                        exit_date DATE,
                        leg_pnl REAL,
                        FOREIGN KEY (position_id) REFERENCES collar_positions(id)
                    );

                    CREATE TABLE IF NOT EXISTS collar_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER,
                        symbol VARCHAR(20) NOT NULL,
                        entry_date DATE NOT NULL,
                        exit_date DATE NOT NULL,
                        spot_at_entry REAL,
                        spot_at_exit REAL,
                        put_strike REAL,
                        call_strike REAL,
                        expiry_date DATE,
                        qty INTEGER,
                        future_pnl REAL,
                        put_pnl REAL,
                        call_pnl REAL,
                        gross_pnl REAL,
                        net_pnl REAL,
                        pnl_pct REAL,
                        hold_days INTEGER,
                        exit_reason VARCHAR(30),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (position_id) REFERENCES collar_positions(id)
                    );

                    CREATE TABLE IF NOT EXISTS collar_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_date DATE NOT NULL UNIQUE,
                        universe_atr_ratio REAL,
                        crash_filter_active BOOLEAN DEFAULT 0,
                        positions_count INTEGER DEFAULT 0,
                        entry_signals INTEGER DEFAULT 0,
                        exit_signals INTEGER DEFAULT 0,
                        entries_taken INTEGER DEFAULT 0,
                        exits_taken INTEGER DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_collar_positions_status
                        ON collar_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_collar_positions_symbol
                        ON collar_positions(symbol);
                    CREATE INDEX IF NOT EXISTS idx_collar_legs_position
                        ON collar_legs(position_id);
                    CREATE INDEX IF NOT EXISTS idx_collar_trades_date
                        ON collar_trades(exit_date);
                    CREATE INDEX IF NOT EXISTS idx_collar_daily_date
                        ON collar_daily_state(trade_date);
                """)
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Positions
    # =========================================================================

    def add_collar(self, symbol: str, entry_date: str, spot_at_entry: float,
                   put_strike: float, call_strike: float, expiry_date: str,
                   future_entry_price: float, put_entry_price: float,
                   call_entry_price: float, lot_size: int,
                   sl_price: float = None, tp_price: float = None,
                   kc6_mid: float = None) -> int:
        """
        Create a new collar position + 3 legs. Returns position_id.
        """
        qty = lot_size
        with db_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("""
                    INSERT INTO collar_positions
                    (symbol, entry_date, spot_at_entry,
                     future_strike, put_strike, call_strike, expiry_date,
                     future_entry_price, put_entry_price, call_entry_price,
                     lot_size, qty,
                     kc6_mid_today, sl_price, tp_price, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """, (symbol, entry_date, spot_at_entry,
                      spot_at_entry, put_strike, call_strike, expiry_date,
                      future_entry_price, put_entry_price, call_entry_price,
                      lot_size, qty,
                      kc6_mid, sl_price, tp_price))
                position_id = cursor.lastrowid

                # Insert 3 legs
                legs = [
                    ('FUT', 'BUY', None, future_entry_price),
                    ('PUT', 'BUY', put_strike, put_entry_price),
                    ('CALL', 'SELL', call_strike, call_entry_price),
                ]
                for leg_type, side, strike, entry_price in legs:
                    conn.execute("""
                        INSERT INTO collar_legs
                        (position_id, leg_type, side, strike, qty,
                         entry_price, entry_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (position_id, leg_type, side, strike, qty,
                          entry_price, entry_date))

                conn.commit()
                return position_id
            finally:
                conn.close()

    def get_open_positions(self) -> List[Dict]:
        """Get all open collar positions (aggregate view)."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM collar_positions WHERE status='OPEN' ORDER BY entry_date"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_open_positions_with_legs(self) -> List[Dict]:
        """Get all open positions with each position's legs attached as a list."""
        positions = self.get_open_positions()
        with db_lock:
            conn = self._get_conn()
            try:
                for pos in positions:
                    legs = conn.execute(
                        "SELECT * FROM collar_legs WHERE position_id=? ORDER BY id",
                        (pos['id'],)
                    ).fetchall()
                    pos['legs'] = [dict(l) for l in legs]
                return positions
            finally:
                conn.close()

    def get_position_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get open collar for a symbol (if any)."""
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM collar_positions WHERE symbol=? AND status='OPEN' LIMIT 1",
                    (symbol,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_open_count(self) -> int:
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM collar_positions WHERE status='OPEN'"
                ).fetchone()
                return row['cnt']
            finally:
                conn.close()

    def get_legs_for_position(self, position_id: int) -> List[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM collar_legs WHERE position_id=? ORDER BY id",
                    (position_id,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def close_collar(self, position_id: int, exit_date: str,
                     future_exit_price: float, put_exit_price: float,
                     call_exit_price: float, exit_reason: str,
                     spot_at_exit: float = None) -> Dict:
        """
        Close a collar: update position + legs, insert trade archive row.
        P/L math:
          future_pnl = (future_exit - future_entry) * qty
          put_pnl    = (put_exit - put_entry) * qty       (long put)
          call_pnl   = (call_entry - call_exit) * qty     (short call)
          gross_pnl  = sum of the above
          net_pnl    = gross_pnl (fees = 0 in paper mode)
        """
        with db_lock:
            conn = self._get_conn()
            try:
                pos = conn.execute(
                    "SELECT * FROM collar_positions WHERE id=?", (position_id,)
                ).fetchone()
                if not pos:
                    raise ValueError(f"Position {position_id} not found")
                pos = dict(pos)

                qty = pos['qty']
                fut_pnl = round((future_exit_price - pos['future_entry_price']) * qty, 2)
                put_pnl = round((put_exit_price - pos['put_entry_price']) * qty, 2)
                call_pnl = round((pos['call_entry_price'] - call_exit_price) * qty, 2)
                gross_pnl = round(fut_pnl + put_pnl + call_pnl, 2)
                net_pnl = gross_pnl  # Paper mode: no fees

                # Hold days
                entry_dt = datetime.strptime(pos['entry_date'], '%Y-%m-%d') \
                    if isinstance(pos['entry_date'], str) else pos['entry_date']
                exit_dt = datetime.strptime(exit_date, '%Y-%m-%d') \
                    if isinstance(exit_date, str) else exit_date
                hold_days = (exit_dt - entry_dt).days

                # pnl_pct based on initial spot notional (1 lot cash-equivalent)
                notional = pos['spot_at_entry'] * qty
                pnl_pct = round((net_pnl / notional) * 100, 2) if notional else 0.0

                # Update position
                conn.execute("""
                    UPDATE collar_positions SET
                        status='CLOSED', exit_date=?,
                        future_exit_price=?, put_exit_price=?, call_exit_price=?,
                        gross_pnl=?, net_pnl=?, exit_reason=?,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (exit_date, future_exit_price, put_exit_price, call_exit_price,
                      gross_pnl, net_pnl, exit_reason, position_id))

                # Update legs
                leg_updates = [
                    ('FUT', future_exit_price, fut_pnl),
                    ('PUT', put_exit_price, put_pnl),
                    ('CALL', call_exit_price, call_pnl),
                ]
                for leg_type, exit_price, leg_pnl in leg_updates:
                    conn.execute("""
                        UPDATE collar_legs SET
                            exit_price=?, exit_date=?, leg_pnl=?
                        WHERE position_id=? AND leg_type=?
                    """, (exit_price, exit_date, leg_pnl, position_id, leg_type))

                # Archive trade
                conn.execute("""
                    INSERT INTO collar_trades
                    (position_id, symbol, entry_date, exit_date,
                     spot_at_entry, spot_at_exit, put_strike, call_strike, expiry_date,
                     qty, future_pnl, put_pnl, call_pnl, gross_pnl, net_pnl,
                     pnl_pct, hold_days, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (position_id, pos['symbol'], pos['entry_date'], exit_date,
                      pos['spot_at_entry'], spot_at_exit,
                      pos['put_strike'], pos['call_strike'], pos['expiry_date'],
                      qty, fut_pnl, put_pnl, call_pnl, gross_pnl, net_pnl,
                      pnl_pct, hold_days, exit_reason))

                conn.commit()
                return {
                    'symbol': pos['symbol'],
                    'future_pnl': fut_pnl,
                    'put_pnl': put_pnl,
                    'call_pnl': call_pnl,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'pnl_pct': pnl_pct,
                    'hold_days': hold_days,
                    'exit_reason': exit_reason,
                }
            finally:
                conn.close()

    # =========================================================================
    # Trades
    # =========================================================================

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM collar_trades ORDER BY exit_date DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_pnl(self) -> float:
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(net_pnl), 0) as total FROM collar_trades "
                    "WHERE DATE(exit_date)=DATE('now')"
                ).fetchone()
                return row['total']
            finally:
                conn.close()

    def get_equity_curve(self) -> List[Dict]:
        """Cumulative net P/L across all closed collars."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT exit_date, net_pnl, pnl_pct, symbol, exit_reason
                    FROM collar_trades
                    ORDER BY exit_date ASC, id ASC
                """).fetchall()
                curve = []
                cum = 0
                for r in rows:
                    r = dict(r)
                    cum += (r['net_pnl'] or 0)
                    curve.append({
                        'date': r['exit_date'],
                        'pnl': r['net_pnl'],
                        'pnl_pct': r['pnl_pct'],
                        'cumulative_pnl': round(cum, 2),
                        'symbol': r['symbol'],
                        'exit_reason': r['exit_reason'],
                    })
                return curve
            finally:
                conn.close()

    # =========================================================================
    # Daily State
    # =========================================================================

    def save_daily_state(self, trade_date: str, universe_atr_ratio: float = None,
                         crash_filter_active: bool = False,
                         positions_count: int = 0, entry_signals: int = 0,
                         exit_signals: int = 0, entries_taken: int = 0,
                         exits_taken: int = 0, daily_pnl: float = 0,
                         notes: str = None):
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO collar_daily_state
                    (trade_date, universe_atr_ratio, crash_filter_active,
                     positions_count, entry_signals, exit_signals,
                     entries_taken, exits_taken, daily_pnl, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trade_date) DO UPDATE SET
                        universe_atr_ratio=excluded.universe_atr_ratio,
                        crash_filter_active=excluded.crash_filter_active,
                        positions_count=excluded.positions_count,
                        entry_signals=excluded.entry_signals,
                        exit_signals=excluded.exit_signals,
                        entries_taken=excluded.entries_taken,
                        exits_taken=excluded.exits_taken,
                        daily_pnl=excluded.daily_pnl,
                        notes=excluded.notes
                """, (trade_date, universe_atr_ratio, crash_filter_active,
                      positions_count, entry_signals, exit_signals,
                      entries_taken, exits_taken, daily_pnl, notes))
                conn.commit()
            finally:
                conn.close()

    def get_daily_state(self, trade_date: str = None) -> Optional[Dict]:
        if not trade_date:
            trade_date = date.today().isoformat()
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM collar_daily_state WHERE trade_date=?",
                    (trade_date,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict:
        with db_lock:
            conn = self._get_conn()
            try:
                total = conn.execute(
                    "SELECT COUNT(*) as cnt FROM collar_trades"
                ).fetchone()['cnt']
                if total == 0:
                    return {'total_trades': 0, 'wins': 0, 'losses': 0,
                            'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0,
                            'profit_factor': 0}

                wins = conn.execute(
                    "SELECT COUNT(*) as cnt FROM collar_trades WHERE net_pnl > 0"
                ).fetchone()['cnt']
                avg_pnl = conn.execute(
                    "SELECT AVG(pnl_pct) as avg FROM collar_trades"
                ).fetchone()['avg']
                total_pnl = conn.execute(
                    "SELECT SUM(net_pnl) as total FROM collar_trades"
                ).fetchone()['total']
                gross_profit = conn.execute(
                    "SELECT COALESCE(SUM(net_pnl),0) as gp FROM collar_trades "
                    "WHERE net_pnl > 0"
                ).fetchone()['gp']
                gross_loss = conn.execute(
                    "SELECT COALESCE(ABS(SUM(net_pnl)),0) as gl FROM collar_trades "
                    "WHERE net_pnl < 0"
                ).fetchone()['gl']
                pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

                return {
                    'total_trades': total,
                    'wins': wins,
                    'losses': total - wins,
                    'win_rate': round(wins / total * 100, 1) if total else 0,
                    'avg_pnl': round(avg_pnl, 2) if avg_pnl else 0,
                    'total_pnl': round(total_pnl, 2) if total_pnl else 0,
                    'profit_factor': round(pf, 2) if pf != float('inf') else 'inf',
                }
            finally:
                conn.close()


# Singleton
_instance = None
_instance_lock = threading.Lock()


def get_collar_db() -> CollarTradingDB:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CollarTradingDB()
    return _instance
