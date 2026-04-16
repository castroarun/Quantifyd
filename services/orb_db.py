"""
ORB -- Opening Range Breakout -- Database Layer
=================================================
SQLite persistence for intraday cash equity ORB trading system.
Pattern follows NasDB / NasAtmDB exactly (singleton, thread-safe, WAL).

Tables:
  - orb_daily_state: per-stock daily OR + CPR + filter data
  - orb_positions: active/closed equity positions
  - orb_signals: signal event history (with filter outcomes)
  - orb_orders: order audit log
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
DB_PATH = str(DATA_DIR / "orb_trading.db")


class OrbDB:
    """SQLite persistence for ORB intraday cash equity trading system."""

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
                    CREATE TABLE IF NOT EXISTS orb_daily_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        instrument TEXT NOT NULL,
                        trade_date DATE NOT NULL,
                        or_high REAL,
                        or_low REAL,
                        or_range REAL,
                        or_finalized INTEGER DEFAULT 0,
                        cpr_pivot REAL,
                        cpr_tc REAL,
                        cpr_bc REAL,
                        cpr_width_pct REAL,
                        is_wide_cpr_day INTEGER DEFAULT 0,
                        prev_day_high REAL,
                        prev_day_low REAL,
                        prev_day_close REAL,
                        prev_day_date DATE,
                        gap_pct REAL,
                        today_open REAL,
                        vwap REAL,
                        rsi_15m REAL,
                        trades_taken INTEGER DEFAULT 0,
                        UNIQUE(instrument, trade_date)
                    );

                    CREATE TABLE IF NOT EXISTS orb_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        instrument TEXT NOT NULL,
                        trade_date DATE NOT NULL,
                        direction TEXT NOT NULL,
                        qty INTEGER NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        sl_price REAL NOT NULL,
                        target_price REAL NOT NULL,
                        or_high REAL,
                        or_low REAL,
                        exit_price REAL,
                        exit_time TIMESTAMP,
                        exit_reason TEXT,
                        pnl_pts REAL,
                        pnl_inr REAL,
                        kite_entry_order_id TEXT,
                        kite_exit_order_id TEXT,
                        status TEXT DEFAULT 'OPEN',
                        gap_pct REAL,
                        vwap_at_entry REAL,
                        rsi_at_entry REAL,
                        cpr_tc REAL,
                        cpr_bc REAL,
                        cpr_width_pct REAL,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_orb_pos_status
                        ON orb_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_orb_pos_date
                        ON orb_positions(trade_date);

                    CREATE TABLE IF NOT EXISTS orb_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        instrument TEXT NOT NULL,
                        signal_time TIMESTAMP NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        sl_price REAL NOT NULL,
                        target_price REAL NOT NULL,
                        or_high REAL,
                        or_low REAL,
                        gap_pct REAL,
                        vwap REAL,
                        rsi_15m REAL,
                        cpr_pivot REAL,
                        cpr_tc REAL,
                        cpr_bc REAL,
                        cpr_width_pct REAL,
                        filters_passed TEXT,
                        action_taken TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS orb_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER,
                        instrument TEXT NOT NULL,
                        tradingsymbol TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        qty INTEGER NOT NULL,
                        order_type TEXT DEFAULT 'MARKET',
                        price REAL,
                        kite_order_id TEXT,
                        status TEXT DEFAULT 'PLACED',
                        exchange TEXT DEFAULT 'NSE',
                        product TEXT DEFAULT 'MIS',
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_orb_orders_pos
                        ON orb_orders(position_id);

                    CREATE TABLE IF NOT EXISTS orb_notifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT,
                        data TEXT,
                        priority TEXT DEFAULT 'normal',
                        read INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_orb_notif_read
                        ON orb_notifications(read);
                    CREATE INDEX IF NOT EXISTS idx_orb_notif_created
                        ON orb_notifications(created_at);
                """)
                conn.commit()
                logger.info(f"[ORB] Database initialized at {self.db_path}")
            finally:
                conn.close()

    # --- Daily State ---------------------------------------------------

    def get_or_create_daily_state(self, instrument, trade_date):
        """Get daily state for instrument+date, creating if absent."""
        if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
            trade_date = trade_date.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM orb_daily_state WHERE instrument=? AND trade_date=?",
                    (instrument, trade_date)
                ).fetchone()
                if row:
                    return dict(row)
                conn.execute(
                    "INSERT INTO orb_daily_state (instrument, trade_date) VALUES (?, ?)",
                    (instrument, trade_date)
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM orb_daily_state WHERE instrument=? AND trade_date=?",
                    (instrument, trade_date)
                ).fetchone()
                return dict(row)
            finally:
                conn.close()

    def update_daily_state(self, instrument, trade_date, **kwargs):
        """Update specific fields on a daily state row."""
        if not kwargs:
            return
        if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
            trade_date = trade_date.isoformat()
        cols = ', '.join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [instrument, trade_date]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"UPDATE orb_daily_state SET {cols} WHERE instrument=? AND trade_date=?",
                    vals
                )
                conn.commit()
            finally:
                conn.close()

    def get_daily_states_for_date(self, trade_date=None):
        """Get all daily state rows for a given date (defaults to today)."""
        if trade_date is None:
            trade_date = date.today().isoformat()
        elif isinstance(trade_date, date) and not isinstance(trade_date, datetime):
            trade_date = trade_date.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM orb_daily_state WHERE trade_date=? ORDER BY instrument",
                    (trade_date,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # --- Positions -----------------------------------------------------

    def add_position(self, **kwargs):
        """Insert a new position. Returns position_id."""
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO orb_positions ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def close_position(self, position_id, exit_price, exit_time, exit_reason,
                       pnl_pts, pnl_inr, kite_exit_order_id=None):
        """Close a position with exit details and P&L."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE orb_positions
                    SET status='CLOSED', exit_price=?, exit_time=?,
                        exit_reason=?, pnl_pts=?, pnl_inr=?,
                        kite_exit_order_id=?
                    WHERE id=?
                """, (exit_price, exit_time, exit_reason,
                      pnl_pts, pnl_inr, kite_exit_order_id, position_id))
                conn.commit()
            finally:
                conn.close()

    def update_position(self, position_id, **kwargs):
        """Update arbitrary fields on a position row."""
        if not kwargs:
            return
        cols = ', '.join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [position_id]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE orb_positions SET {cols} WHERE id=?", vals)
                conn.commit()
            finally:
                conn.close()

    def get_open_positions(self, instrument=None):
        """Get all OPEN positions, optionally filtered by instrument."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM orb_positions WHERE status='OPEN'"
                params = []
                if instrument:
                    sql += " AND instrument=?"
                    params.append(instrument)
                sql += " ORDER BY entry_time DESC"
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_positions(self, instrument=None):
        """Get all positions (any status) for today."""
        today = date.today().isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM orb_positions WHERE trade_date=?"
                params = [today]
                if instrument:
                    sql += " AND instrument=?"
                    params.append(instrument)
                sql += " ORDER BY entry_time"
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_closed(self, instrument=None):
        """Get today's closed positions."""
        today = date.today().isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM orb_positions WHERE status='CLOSED' AND trade_date=?"
                params = [today]
                if instrument:
                    sql += " AND instrument=?"
                    params.append(instrument)
                sql += " ORDER BY exit_time"
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # --- Signals -------------------------------------------------------

    def log_signal(self, **kwargs):
        """Log an ORB signal event. Returns signal_id."""
        kwargs['created_at'] = datetime.now().isoformat()
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO orb_signals ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_recent_signals(self, limit=30):
        """Get the most recent signals."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM orb_signals ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # --- Orders --------------------------------------------------------

    def log_order(self, **kwargs):
        """Log a Kite order. Returns order row id."""
        kwargs['created_at'] = datetime.now().isoformat()
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO orb_orders ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def update_order(self, order_id, **kwargs):
        """Update order fields (e.g. status after fill check)."""
        if not kwargs:
            return
        cols = ', '.join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [order_id]
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE orb_orders SET {cols} WHERE id=?", vals)
                conn.commit()
            finally:
                conn.close()

    # --- Stats ---------------------------------------------------------

    def get_recent_trades(self, limit=50):
        """Get recent closed positions (most recent first)."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM orb_positions WHERE status='CLOSED' "
                    "ORDER BY exit_time DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_stats(self):
        """Compute aggregate trading statistics from all closed positions."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM orb_positions WHERE status='CLOSED'"
                ).fetchall()
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
                    'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
                    'total_pnl': round(total_pnl, 2),
                    'avg_pnl': round(total_pnl / len(trades), 2) if trades else 0,
                    'max_win': round(max(pnls), 2) if pnls else 0,
                    'max_loss': round(min(pnls), 2) if pnls else 0,
                    'profit_factor': round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0,
                }
            finally:
                conn.close()

    def get_equity_curve(self):
        """
        Build daily P&L equity curve from closed positions.
        Returns list of dicts: [{trade_date, daily_pnl, cumulative_pnl, trades, wins, losses}, ...]
        """
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT trade_date,
                           SUM(pnl_inr) as daily_pnl,
                           COUNT(*) as trades,
                           SUM(CASE WHEN pnl_inr > 0 THEN 1 ELSE 0 END) as wins,
                           SUM(CASE WHEN pnl_inr <= 0 THEN 1 ELSE 0 END) as losses
                    FROM orb_positions
                    WHERE status='CLOSED'
                    GROUP BY trade_date
                    ORDER BY trade_date
                """).fetchall()

                curve = []
                cumulative = 0.0
                for r in rows:
                    d = dict(r)
                    daily_pnl = d.get('daily_pnl', 0) or 0
                    cumulative += daily_pnl
                    d['cumulative_pnl'] = round(cumulative, 2)
                    d['daily_pnl'] = round(daily_pnl, 2)
                    curve.append(d)
                return curve
            finally:
                conn.close()

    # --- Notifications ---------------------------------------------------

    def log_notification(self, type, title, message='', data='', priority='normal'):
        """Insert a notification record. Returns notification id."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    "INSERT INTO orb_notifications (type, title, message, data, priority) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (type, title, message, data, priority)
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_notifications(self, limit=50, unread_only=False):
        """Get recent notifications, newest first."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM orb_notifications"
                if unread_only:
                    sql += " WHERE read=0"
                sql += " ORDER BY created_at DESC LIMIT ?"
                rows = conn.execute(sql, (limit,)).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def mark_read(self, notification_id):
        """Mark a single notification as read."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE orb_notifications SET read=1 WHERE id=?",
                    (notification_id,)
                )
                conn.commit()
            finally:
                conn.close()

    def mark_all_read(self):
        """Mark all notifications as read."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute("UPDATE orb_notifications SET read=1 WHERE read=0")
                conn.commit()
            finally:
                conn.close()

    def get_unread_count(self):
        """Return count of unread notifications."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM orb_notifications WHERE read=0"
                ).fetchone()
                return row['cnt'] if row else 0
            finally:
                conn.close()

    # --- State -------------------------------------------------------------

    def get_state(self):
        """
        Get latest system state for dashboard.
        Returns dict with today's daily states, open positions, and config summary.
        """
        today = date.today().isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                # Latest daily states for today
                daily_rows = conn.execute(
                    "SELECT * FROM orb_daily_state WHERE trade_date=? ORDER BY instrument",
                    (today,)
                ).fetchall()

                # Open positions
                open_rows = conn.execute(
                    "SELECT * FROM orb_positions WHERE status='OPEN' ORDER BY entry_time"
                ).fetchall()

                return {
                    'trade_date': today,
                    'daily_states': [dict(r) for r in daily_rows],
                    'open_positions': [dict(r) for r in open_rows],
                    'instruments_tracked': len(daily_rows),
                    'positions_open': len(open_rows),
                }
            finally:
                conn.close()


# --- Singleton ---------------------------------------------------------

_instance = None
_instance_lock = threading.Lock()


def get_orb_db():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = OrbDB()
    return _instance
