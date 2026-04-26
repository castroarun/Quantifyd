"""
EOD Breakout Scanner — Database Layer
======================================
SQLite persistence for the daily-bar EOD breakout system. Hosts THREE
sub-systems (Nifty 500, Small-cap, F&O) in one DB, distinguished by
`system_id` column on every row. Pattern mirrors orb_db.py.

Tables:
  - eod_signals: every breakout signal generated each EOD (system_id, date, symbol, breakout_high, vol_ratio, atr, entry_planned, stop_planned, target_planned, status)
  - eod_positions: open positions (one row per symbol per fill)
  - eod_trades: closed trades log
  - eod_daily_state: per-system daily snapshot (signals_generated, fills, equity, day_pnl)
  - eod_equity_curve: per-system daily equity series (for charts)
"""

import sqlite3
import threading
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
DATA_DIR = Path(_volume_path) / "data" if _volume_path else Path(__file__).parent.parent / "backtest_data"
DB_PATH = str(DATA_DIR / "eod_breakout.db")


# Sub-system identifiers — used as system_id column value
SYSTEM_NIFTY500 = 'nifty500'
SYSTEM_SMALLCAP = 'smallcap'
SYSTEM_FNO      = 'fno'

ALL_SYSTEMS = (SYSTEM_NIFTY500, SYSTEM_SMALLCAP, SYSTEM_FNO)


class EodBreakoutDB:
    """SQLite persistence for the EOD breakout multi-system scanner."""

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
                    CREATE TABLE IF NOT EXISTS eod_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        signal_date DATE NOT NULL,
                        symbol TEXT NOT NULL,
                        direction TEXT DEFAULT 'LONG',
                        signal_close REAL NOT NULL,
                        breakout_high REAL,
                        vol_ratio REAL,
                        atr REAL,
                        sma_200 REAL,
                        rank_score REAL,
                        status TEXT DEFAULT 'PENDING',
                        position_id INTEGER,
                        spread_structure TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(system_id, signal_date, symbol)
                    );

                    CREATE INDEX IF NOT EXISTS idx_eod_signals_status
                        ON eod_signals(system_id, status);
                    CREATE INDEX IF NOT EXISTS idx_eod_signals_date
                        ON eod_signals(signal_date DESC);

                    CREATE TABLE IF NOT EXISTS eod_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_date DATE NOT NULL,
                        entry_date DATE NOT NULL,
                        entry_price REAL NOT NULL,
                        qty INTEGER NOT NULL,
                        initial_stop REAL NOT NULL,
                        target REAL NOT NULL,
                        atr_at_entry REAL,
                        risk_inr REAL,
                        notional_inr REAL,
                        status TEXT DEFAULT 'OPEN',
                        signal_id INTEGER,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_eod_positions_status
                        ON eod_positions(system_id, status);

                    CREATE TABLE IF NOT EXISTS eod_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        position_id INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        entry_date DATE NOT NULL,
                        exit_date DATE NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        qty INTEGER NOT NULL,
                        days_held INTEGER,
                        exit_reason TEXT,
                        gross_pnl REAL,
                        net_pnl REAL,
                        cost_inr REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_eod_trades_system
                        ON eod_trades(system_id, exit_date DESC);

                    CREATE TABLE IF NOT EXISTS eod_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        trade_date DATE NOT NULL,
                        signals_generated INTEGER DEFAULT 0,
                        fills_today INTEGER DEFAULT 0,
                        exits_today INTEGER DEFAULT 0,
                        open_positions INTEGER DEFAULT 0,
                        day_pnl REAL DEFAULT 0,
                        cumulative_pnl REAL DEFAULT 0,
                        equity REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(system_id, trade_date)
                    );

                    CREATE TABLE IF NOT EXISTS eod_equity_curve (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        date DATE NOT NULL,
                        equity REAL NOT NULL,
                        UNIQUE(system_id, date)
                    );

                    CREATE INDEX IF NOT EXISTS idx_eod_equity_system
                        ON eod_equity_curve(system_id, date);
                """)
                # Idempotent migration: add columns to eod_signals for direction
                # and spread structure JSON. CREATE TABLE IF NOT EXISTS won't
                # add new columns to an existing table.
                for stmt in (
                    "ALTER TABLE eod_signals ADD COLUMN direction TEXT DEFAULT 'LONG'",
                    "ALTER TABLE eod_signals ADD COLUMN spread_structure TEXT",
                ):
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError as e:
                        if 'duplicate column' not in str(e).lower():
                            raise
                conn.commit()
                logger.info(f"[EOD-BREAKOUT] Database initialized at {self.db_path}")
            finally:
                conn.close()

    # ---- Signals ----

    def add_signal(self, system_id, signal_date, symbol, **kwargs):
        """Insert (or update if exists) a signal row. Returns signal_id."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                # ON CONFLICT DO UPDATE for idempotency on re-runs
                cols = ['system_id', 'signal_date', 'symbol'] + list(kwargs.keys())
                vals = [system_id, signal_date, symbol] + list(kwargs.values())
                placeholders = ', '.join(['?'] * len(cols))
                update_clauses = ', '.join(f'{k}=excluded.{k}' for k in kwargs.keys())
                sql = (
                    f"INSERT INTO eod_signals ({', '.join(cols)}) VALUES ({placeholders}) "
                    f"ON CONFLICT(system_id, signal_date, symbol) DO UPDATE SET {update_clauses} "
                    f"RETURNING id"
                )
                row = conn.execute(sql, vals).fetchone()
                conn.commit()
                return row['id'] if row else None
            finally:
                conn.close()

    def get_pending_signals(self, system_id):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM eod_signals WHERE system_id=? AND status='PENDING' ORDER BY signal_date, rank_score DESC",
                    (system_id,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_recent_signals(self, system_id, limit=50):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM eod_signals WHERE system_id=? ORDER BY signal_date DESC, id DESC LIMIT ?",
                    (system_id, limit)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def update_signal_status(self, signal_id, status, position_id=None, notes=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                if position_id is not None:
                    conn.execute(
                        "UPDATE eod_signals SET status=?, position_id=?, notes=? WHERE id=?",
                        (status, position_id, notes, signal_id)
                    )
                else:
                    conn.execute(
                        "UPDATE eod_signals SET status=?, notes=? WHERE id=?",
                        (status, notes, signal_id)
                    )
                conn.commit()
            finally:
                conn.close()

    # ---- Positions ----

    def add_position(self, **kwargs):
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO eod_positions ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_open_positions(self, system_id=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                if system_id:
                    rows = conn.execute(
                        "SELECT * FROM eod_positions WHERE status='OPEN' AND system_id=? ORDER BY entry_date DESC",
                        (system_id,)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM eod_positions WHERE status='OPEN' ORDER BY system_id, entry_date DESC"
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def count_open_positions(self, system_id):
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM eod_positions WHERE status='OPEN' AND system_id=?",
                    (system_id,)
                ).fetchone()
                return row['n'] if row else 0
            finally:
                conn.close()

    def close_position(self, position_id, exit_price, exit_date, exit_reason,
                       gross_pnl, net_pnl, cost_inr, days_held):
        """Mark position closed and add a trade row."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                pos = conn.execute(
                    "SELECT * FROM eod_positions WHERE id=?", (position_id,)
                ).fetchone()
                if not pos:
                    return None
                conn.execute(
                    "UPDATE eod_positions SET status='CLOSED' WHERE id=?", (position_id,)
                )
                conn.execute("""
                    INSERT INTO eod_trades
                        (system_id, position_id, symbol, entry_date, exit_date,
                         entry_price, exit_price, qty, days_held, exit_reason,
                         gross_pnl, net_pnl, cost_inr)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pos['system_id'], position_id, pos['symbol'],
                    pos['entry_date'], exit_date,
                    pos['entry_price'], exit_price, pos['qty'], days_held,
                    exit_reason, gross_pnl, net_pnl, cost_inr,
                ))
                conn.commit()
                return conn.execute("SELECT last_insert_rowid() AS id").fetchone()['id']
            finally:
                conn.close()

    def get_recent_trades(self, system_id=None, limit=100):
        with self.db_lock:
            conn = self._get_conn()
            try:
                if system_id:
                    rows = conn.execute(
                        "SELECT * FROM eod_trades WHERE system_id=? ORDER BY exit_date DESC, id DESC LIMIT ?",
                        (system_id, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM eod_trades ORDER BY exit_date DESC, id DESC LIMIT ?",
                        (limit,)
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ---- Daily state + equity ----

    def upsert_daily_state(self, system_id, trade_date, **kwargs):
        with self.db_lock:
            conn = self._get_conn()
            try:
                cols = ['system_id', 'trade_date'] + list(kwargs.keys())
                vals = [system_id, trade_date] + list(kwargs.values())
                placeholders = ', '.join(['?'] * len(cols))
                update_clauses = ', '.join(f'{k}=excluded.{k}' for k in kwargs.keys())
                conn.execute(
                    f"INSERT INTO eod_daily_state ({', '.join(cols)}) VALUES ({placeholders}) "
                    f"ON CONFLICT(system_id, trade_date) DO UPDATE SET {update_clauses}",
                    vals
                )
                conn.commit()
            finally:
                conn.close()

    def get_daily_state(self, system_id, trade_date):
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM eod_daily_state WHERE system_id=? AND trade_date=?",
                    (system_id, trade_date)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def upsert_equity_point(self, system_id, date, equity):
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO eod_equity_curve (system_id, date, equity) VALUES (?, ?, ?) "
                    "ON CONFLICT(system_id, date) DO UPDATE SET equity=excluded.equity",
                    (system_id, date, equity)
                )
                conn.commit()
            finally:
                conn.close()

    def get_equity_curve(self, system_id):
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT date, equity FROM eod_equity_curve WHERE system_id=? ORDER BY date",
                    (system_id,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ---- Stats ----

    def get_stats(self, system_id):
        """Aggregate metrics for one sub-system."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute("""
                    SELECT
                        COUNT(*) AS total_trades,
                        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                        SUM(CASE WHEN net_pnl < 0 THEN 1 ELSE 0 END) AS losses,
                        COALESCE(SUM(net_pnl), 0) AS total_pnl,
                        COALESCE(SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END), 0) AS gross_wins,
                        COALESCE(SUM(CASE WHEN net_pnl < 0 THEN net_pnl ELSE 0 END), 0) AS gross_losses,
                        COALESCE(AVG(days_held), 0) AS avg_days_held
                    FROM eod_trades WHERE system_id=?
                """, (system_id,)).fetchone()
                stats = dict(row) if row else {}
                # Open positions count — same connection so no re-entrant lock deadlock
                pos_row = conn.execute(
                    "SELECT COUNT(*) AS n FROM eod_positions WHERE status='OPEN' AND system_id=?",
                    (system_id,)
                ).fetchone()
                stats['open_positions'] = pos_row['n'] if pos_row else 0
            finally:
                conn.close()
        # Compute derived metrics outside the lock
        tt = stats.get('total_trades', 0) or 0
        stats['win_rate'] = round(100 * (stats.get('wins') or 0) / tt, 2) if tt else 0
        gw = stats.get('gross_wins') or 0
        gl = stats.get('gross_losses') or 0
        stats['profit_factor'] = round(gw / abs(gl), 2) if gl else 0
        return stats


_instance = None
_instance_lock = threading.Lock()


def get_eod_breakout_db():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = EodBreakoutDB()
    return _instance
