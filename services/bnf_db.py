"""
BNF Squeeze & Fire — Database Layer
=====================================
SQLite persistence for BankNifty options trading system.
Two modes: Squeeze (non-directional strangles) + Fire (directional naked sell).
Pattern follows MaruthiDB exactly.
"""

import sqlite3
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Use same DATA_DIR pattern as other services
import os
from pathlib import Path

_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
DATA_DIR = Path(_volume_path) / "data" if _volume_path else Path(__file__).parent.parent / "backtest_data"
DB_PATH = str(DATA_DIR / "bnf_trading.db")


class BnfDB:
    """SQLite persistence for BNF Squeeze & Fire trading system."""

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
                    CREATE TABLE IF NOT EXISTS bnf_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        bb_state VARCHAR(15) DEFAULT 'UNKNOWN',
                        squeeze_count INTEGER DEFAULT 0,
                        bb_width REAL,
                        bb_width_ma REAL,
                        sma_value REAL,
                        atr_value REAL,
                        direction VARCHAR(10) DEFAULT 'NEUTRAL',
                        trend_strength REAL DEFAULT 0,
                        last_close REAL,
                        last_scan_time TIMESTAMP,
                        last_signal_time TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    INSERT OR IGNORE INTO bnf_state (id) VALUES (1);

                    -- Active and closed positions
                    CREATE TABLE IF NOT EXISTS bnf_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mode VARCHAR(10) NOT NULL,
                        tradingsymbol VARCHAR(50) NOT NULL,
                        exchange VARCHAR(10) DEFAULT 'NFO',
                        transaction_type VARCHAR(5) NOT NULL,
                        instrument_type VARCHAR(10),
                        qty INTEGER NOT NULL,
                        strike REAL,
                        expiry_date DATE,
                        entry_price REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        sl_price REAL,
                        sl_reason VARCHAR(30),
                        kite_order_id VARCHAR(50),
                        signal_type VARCHAR(30),
                        hold_bars_remaining INTEGER,
                        max_loss_rupees REAL,
                        status VARCHAR(20) DEFAULT 'PENDING',
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_bnf_pos_status
                        ON bnf_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_bnf_pos_mode_status
                        ON bnf_positions(mode, status);

                    -- Completed trades
                    CREATE TABLE IF NOT EXISTS bnf_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER,
                        mode VARCHAR(10),
                        tradingsymbol VARCHAR(50),
                        instrument_type VARCHAR(10),
                        transaction_type VARCHAR(5),
                        qty INTEGER,
                        strike REAL,
                        entry_price REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        exit_price REAL,
                        exit_time TIMESTAMP,
                        pnl_abs REAL,
                        pnl_pct REAL,
                        exit_reason VARCHAR(30),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (position_id) REFERENCES bnf_positions(id)
                    );

                    -- Order audit trail
                    CREATE TABLE IF NOT EXISTS bnf_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tradingsymbol VARCHAR(50) NOT NULL,
                        exchange VARCHAR(10) DEFAULT 'NFO',
                        transaction_type VARCHAR(5) NOT NULL,
                        qty INTEGER NOT NULL,
                        price REAL,
                        order_type VARCHAR(10) DEFAULT 'MARKET',
                        product VARCHAR(10) DEFAULT 'NRML',
                        status VARCHAR(20) DEFAULT 'PENDING',
                        kite_order_id VARCHAR(50),
                        position_id INTEGER,
                        signal_type VARCHAR(30),
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (position_id) REFERENCES bnf_positions(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_bnf_ord_status
                        ON bnf_orders(status);

                    -- Signal event history
                    CREATE TABLE IF NOT EXISTS bnf_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_type VARCHAR(30) NOT NULL,
                        mode VARCHAR(10),
                        bb_state VARCHAR(15),
                        squeeze_count INTEGER,
                        direction VARCHAR(10),
                        trend_strength REAL,
                        spot_price REAL,
                        atr_value REAL,
                        action_taken VARCHAR(100),
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- Daily state snapshots
                    CREATE TABLE IF NOT EXISTS bnf_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_date DATE UNIQUE NOT NULL,
                        bb_state VARCHAR(15),
                        squeeze_positions INTEGER DEFAULT 0,
                        fire_positions INTEGER DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        cumulative_pnl REAL DEFAULT 0,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    # ─── State ───────────────────────────────────────────────

    def get_state(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT * FROM bnf_state WHERE id=1").fetchone()
                return dict(row) if row else {}
            finally:
                conn.close()

    def update_state(self, **kwargs):
        kwargs['updated_at'] = datetime.now().isoformat()
        sets = ', '.join(f'{k}=?' for k in kwargs)
        vals = list(kwargs.values())
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE bnf_state SET {sets} WHERE id=1", vals)
                conn.commit()
            finally:
                conn.close()

    # ─── Positions ───────────────────────────────────────────

    def add_position(self, **kwargs):
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join('?' * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO bnf_positions ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()))
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_active_positions(self, mode=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                if mode:
                    rows = conn.execute(
                        "SELECT * FROM bnf_positions WHERE status='ACTIVE' AND mode=? ORDER BY entry_time",
                        (mode,)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM bnf_positions WHERE status='ACTIVE' ORDER BY entry_time").fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_pending_positions(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM bnf_positions WHERE status='PENDING' ORDER BY entry_time").fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def activate_position(self, position_id, entry_price=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                if entry_price:
                    conn.execute(
                        "UPDATE bnf_positions SET status='ACTIVE', entry_price=?, updated_at=? WHERE id=?",
                        (entry_price, datetime.now().isoformat(), position_id))
                else:
                    conn.execute(
                        "UPDATE bnf_positions SET status='ACTIVE', updated_at=? WHERE id=?",
                        (datetime.now().isoformat(), position_id))
                conn.commit()
            finally:
                conn.close()

    def close_position(self, position_id, exit_price, exit_reason):
        with self.db_lock:
            conn = self._get_conn()
            try:
                pos = conn.execute("SELECT * FROM bnf_positions WHERE id=?", (position_id,)).fetchone()
                if not pos:
                    return

                pos = dict(pos)
                pnl_abs = 0
                pnl_pct = 0

                if pos['transaction_type'] == 'SELL':
                    # Sold option: profit = entry - exit
                    pnl_pts = pos['entry_price'] - exit_price
                else:
                    # Bought option: profit = exit - entry
                    pnl_pts = exit_price - pos['entry_price']

                pnl_abs = pnl_pts * pos['qty']
                if pos['entry_price'] > 0:
                    pnl_pct = pnl_pts / pos['entry_price'] * 100

                # Cap loss
                if pos.get('max_loss_rupees') and pnl_abs < 0:
                    pnl_abs = max(pnl_abs, -pos['max_loss_rupees'])

                now = datetime.now().isoformat()

                conn.execute(
                    "UPDATE bnf_positions SET status='CLOSED', updated_at=? WHERE id=?",
                    (now, position_id))

                conn.execute("""
                    INSERT INTO bnf_trades
                    (position_id, mode, tradingsymbol, instrument_type, transaction_type,
                     qty, strike, entry_price, entry_time, exit_price, exit_time,
                     pnl_abs, pnl_pct, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (position_id, pos['mode'], pos['tradingsymbol'], pos['instrument_type'],
                      pos['transaction_type'], pos['qty'], pos['strike'],
                      pos['entry_price'], pos['entry_time'],
                      exit_price, now, round(pnl_abs, 2), round(pnl_pct, 2), exit_reason))

                conn.commit()
            finally:
                conn.close()

    def cancel_position(self, position_id):
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE bnf_positions SET status='CANCELLED', updated_at=? WHERE id=?",
                    (datetime.now().isoformat(), position_id))
                conn.commit()
            finally:
                conn.close()

    def update_position(self, position_id, **kwargs):
        kwargs['updated_at'] = datetime.now().isoformat()
        sets = ', '.join(f'{k}=?' for k in kwargs)
        vals = list(kwargs.values()) + [position_id]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE bnf_positions SET {sets} WHERE id=?", vals)
                conn.commit()
            finally:
                conn.close()

    # ─── Orders ──────────────────────────────────────────────

    def log_order(self, **kwargs):
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join('?' * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO bnf_orders ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()))
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def update_order(self, order_id, **kwargs):
        kwargs['updated_at'] = datetime.now().isoformat()
        sets = ', '.join(f'{k}=?' for k in kwargs)
        vals = list(kwargs.values()) + [order_id]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE bnf_orders SET {sets} WHERE id=?", vals)
                conn.commit()
            finally:
                conn.close()

    def get_pending_orders(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM bnf_orders WHERE status='PENDING' ORDER BY created_at").fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_order_count(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM bnf_orders WHERE date(created_at)=date('now')").fetchone()
                return row['cnt'] if row else 0
            finally:
                conn.close()

    # ─── Signals ─────────────────────────────────────────────

    def log_signal(self, **kwargs):
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join('?' * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO bnf_signals ({cols}) VALUES ({placeholders})",
                    list(kwargs.values()))
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_recent_signals(self, limit=50):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM bnf_signals ORDER BY created_at DESC LIMIT ?",
                    (limit,)).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ─── Trades ──────────────────────────────────────────────

    def get_recent_trades(self, limit=50):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM bnf_trades ORDER BY exit_time DESC LIMIT ?",
                    (limit,)).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_stats(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                total = conn.execute("SELECT COUNT(*) as c FROM bnf_trades").fetchone()['c']
                wins = conn.execute("SELECT COUNT(*) as c FROM bnf_trades WHERE pnl_abs > 0").fetchone()['c']
                losses = conn.execute("SELECT COUNT(*) as c FROM bnf_trades WHERE pnl_abs <= 0").fetchone()['c']
                total_pnl = conn.execute("SELECT COALESCE(SUM(pnl_abs), 0) as s FROM bnf_trades").fetchone()['s']

                # By mode
                sq_trades = conn.execute("SELECT COUNT(*) as c FROM bnf_trades WHERE mode='SQUEEZE'").fetchone()['c']
                fr_trades = conn.execute("SELECT COUNT(*) as c FROM bnf_trades WHERE mode='FIRE'").fetchone()['c']
                sq_pnl = conn.execute("SELECT COALESCE(SUM(pnl_abs), 0) as s FROM bnf_trades WHERE mode='SQUEEZE'").fetchone()['s']
                fr_pnl = conn.execute("SELECT COALESCE(SUM(pnl_abs), 0) as s FROM bnf_trades WHERE mode='FIRE'").fetchone()['s']

                return {
                    'total_trades': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                    'total_pnl': round(total_pnl, 2),
                    'squeeze_trades': sq_trades,
                    'fire_trades': fr_trades,
                    'squeeze_pnl': round(sq_pnl, 2),
                    'fire_pnl': round(fr_pnl, 2),
                }
            finally:
                conn.close()

    # ─── Daily State ─────────────────────────────────────────

    def save_daily_state(self, **kwargs):
        trade_date = kwargs.pop('trade_date')
        cols = ', '.join(['trade_date'] + list(kwargs.keys()))
        placeholders = ', '.join('?' * (len(kwargs) + 1))
        updates = ', '.join(f'{k}=excluded.{k}' for k in kwargs)
        vals = [trade_date] + list(kwargs.values())

        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"""INSERT INTO bnf_daily_state ({cols}) VALUES ({placeholders})
                        ON CONFLICT(trade_date) DO UPDATE SET {updates}""",
                    vals)
                conn.commit()
            finally:
                conn.close()

    def get_equity_curve(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT trade_date, cumulative_pnl FROM bnf_daily_state ORDER BY trade_date").fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()


# ─── Singleton ───────────────────────────────────────────────

_bnf_db = None
_bnf_db_lock = threading.Lock()


def get_bnf_db():
    global _bnf_db
    if _bnf_db is None:
        with _bnf_db_lock:
            if _bnf_db is None:
                _bnf_db = BnfDB()
    return _bnf_db
