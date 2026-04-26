"""
Nifty ORB Strangle — SQLite Persistence (Phase 3 paper-trade)
==============================================================

Singleton DB for the multi-variant Nifty ORB strangle paper-trading system.
One DB file holds all 8 variants — every row is keyed by `variant_id`.

Tables:
    strangle_variants     — config snapshot per variant (one row per variant)
    strangle_positions    — open + closed strangle positions
    strangle_legs         — PE / CE leg rows under each position
    strangle_trades       — closed-trades archive
    strangle_daily_state  — per-variant per-day state snapshot

Mirrors `services/collar_db.py` patterns (singleton + db_lock + WAL).
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
DB_PATH = DB_DIR / 'strangle_trading.db'

db_lock = threading.Lock()


class StrangleTradingDB:
    """SQLite persistence for the Nifty ORB strangle multi-variant paper system."""

    def __init__(self):
        self.db_path = DB_PATH
        self._init_database()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_database(self):
        with db_lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    -- Variant configuration snapshot
                    CREATE TABLE IF NOT EXISTS strangle_variants (
                        variant_id  VARCHAR(40) PRIMARY KEY,
                        name        VARCHAR(80) NOT NULL,
                        config_json TEXT NOT NULL,
                        enabled     INTEGER DEFAULT 1,
                        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- Strangle positions (one row per cross-leg pair)
                    CREATE TABLE IF NOT EXISTS strangle_positions (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        variant_id      VARCHAR(40) NOT NULL,
                        symbol          VARCHAR(20) NOT NULL DEFAULT 'NIFTY',
                        entry_date      DATE NOT NULL,
                        entry_ts        TIMESTAMP NOT NULL,
                        direction       VARCHAR(10) NOT NULL,    -- LONG | SHORT (break direction)
                        spot_at_entry   REAL NOT NULL,
                        or_high         REAL NOT NULL,
                        or_low          REAL NOT NULL,
                        or_width_pct    REAL NOT NULL,
                        sl_price        REAL NOT NULL,           -- underlying spot SL
                        expiry_date     DATE NOT NULL,
                        pe_strike       REAL NOT NULL,
                        ce_strike       REAL NOT NULL,
                        pe_entry_price  REAL NOT NULL,
                        ce_entry_price  REAL NOT NULL,
                        pe_delta_at_entry REAL,
                        ce_delta_at_entry REAL,
                        lot_size        INTEGER NOT NULL,
                        qty             INTEGER NOT NULL,
                        status          VARCHAR(20) DEFAULT 'OPEN',
                        exit_date       DATE,
                        exit_ts         TIMESTAMP,
                        pe_exit_price   REAL,
                        ce_exit_price   REAL,
                        spot_at_exit    REAL,
                        gross_pnl       REAL,
                        net_pnl         REAL,
                        exit_reason     VARCHAR(30),
                        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (variant_id) REFERENCES strangle_variants(variant_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_strangle_pos_variant_status
                        ON strangle_positions(variant_id, status);
                    CREATE INDEX IF NOT EXISTS idx_strangle_pos_entry_date
                        ON strangle_positions(entry_date);

                    -- Per-leg detail (PE + CE under each position)
                    CREATE TABLE IF NOT EXISTS strangle_legs (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id     INTEGER NOT NULL,
                        variant_id      VARCHAR(40) NOT NULL,
                        leg_type        VARCHAR(2) NOT NULL,    -- PE | CE
                        side            VARCHAR(4) NOT NULL,    -- SELL (paper, both legs short)
                        tradingsymbol   VARCHAR(60),
                        strike          REAL NOT NULL,
                        expiry_date     DATE NOT NULL,
                        qty             INTEGER NOT NULL,
                        entry_price     REAL NOT NULL,
                        delta_at_entry  REAL,
                        iv_at_entry     REAL,
                        exit_price      REAL,
                        leg_pnl         REAL,
                        mtm_now         REAL,
                        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (position_id) REFERENCES strangle_positions(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_strangle_legs_pos
                        ON strangle_legs(position_id);

                    -- Closed-trades archive
                    CREATE TABLE IF NOT EXISTS strangle_trades (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id     INTEGER NOT NULL,
                        variant_id      VARCHAR(40) NOT NULL,
                        entry_date      DATE NOT NULL,
                        exit_date       DATE NOT NULL,
                        entry_ts        TIMESTAMP,
                        exit_ts         TIMESTAMP,
                        direction       VARCHAR(10),
                        spot_at_entry   REAL,
                        spot_at_exit    REAL,
                        pe_strike       REAL,
                        ce_strike       REAL,
                        pe_entry        REAL,
                        ce_entry        REAL,
                        pe_exit         REAL,
                        ce_exit         REAL,
                        gross_pnl       REAL,
                        net_pnl         REAL,
                        costs           REAL,
                        exit_reason     VARCHAR(30),
                        hold_minutes    INTEGER,
                        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (position_id) REFERENCES strangle_positions(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_strangle_trades_variant_date
                        ON strangle_trades(variant_id, exit_date);

                    -- Per-variant per-day state
                    CREATE TABLE IF NOT EXISTS strangle_daily_state (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        variant_id          VARCHAR(40) NOT NULL,
                        trade_date          DATE NOT NULL,
                        or_high             REAL,
                        or_low              REAL,
                        or_width_pct        REAL,
                        day_filter_passed   INTEGER DEFAULT 0,    -- 1 = filters allow entry
                        signal_seen         INTEGER DEFAULT 0,    -- break occurred
                        rsi_confirmed       INTEGER DEFAULT 0,
                        entry_taken         INTEGER DEFAULT 0,
                        exit_taken          INTEGER DEFAULT 0,
                        exit_reason         VARCHAR(30),
                        daily_pnl           REAL DEFAULT 0,
                        notes               TEXT,
                        last_event_ts       TIMESTAMP,
                        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(variant_id, trade_date)
                    );

                    CREATE INDEX IF NOT EXISTS idx_strangle_daily_variant_date
                        ON strangle_daily_state(variant_id, trade_date);
                """)
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Variant config snapshots
    # =========================================================================

    def upsert_variant(self, variant_id: str, name: str, config_json: str,
                       enabled: bool = True):
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO strangle_variants (variant_id, name, config_json, enabled)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(variant_id) DO UPDATE SET
                        name=excluded.name,
                        config_json=excluded.config_json,
                        enabled=excluded.enabled,
                        updated_at=CURRENT_TIMESTAMP
                """, (variant_id, name, config_json, 1 if enabled else 0))
                conn.commit()
            finally:
                conn.close()

    def list_variants(self) -> List[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM strangle_variants ORDER BY variant_id"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # =========================================================================
    # Positions
    # =========================================================================

    def open_position(self, variant_id: str, entry_date: str, entry_ts: str,
                      direction: str, spot_at_entry: float,
                      or_high: float, or_low: float, or_width_pct: float,
                      sl_price: float, expiry_date: str,
                      pe_strike: float, ce_strike: float,
                      pe_entry_price: float, ce_entry_price: float,
                      pe_delta: float, ce_delta: float,
                      lot_size: int,
                      pe_tradingsymbol: str = None, ce_tradingsymbol: str = None,
                      pe_iv: float = None, ce_iv: float = None) -> int:
        qty = lot_size
        with db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute("""
                    INSERT INTO strangle_positions
                    (variant_id, entry_date, entry_ts, direction, spot_at_entry,
                     or_high, or_low, or_width_pct, sl_price, expiry_date,
                     pe_strike, ce_strike, pe_entry_price, ce_entry_price,
                     pe_delta_at_entry, ce_delta_at_entry,
                     lot_size, qty, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """, (variant_id, entry_date, entry_ts, direction, spot_at_entry,
                      or_high, or_low, or_width_pct, sl_price, expiry_date,
                      pe_strike, ce_strike, pe_entry_price, ce_entry_price,
                      pe_delta, ce_delta, lot_size, qty))
                position_id = cur.lastrowid

                # Insert legs
                for leg_type, strike, entry_price, delta_v, iv_v, tsym in [
                    ('PE', pe_strike, pe_entry_price, pe_delta, pe_iv, pe_tradingsymbol),
                    ('CE', ce_strike, ce_entry_price, ce_delta, ce_iv, ce_tradingsymbol),
                ]:
                    conn.execute("""
                        INSERT INTO strangle_legs
                        (position_id, variant_id, leg_type, side, tradingsymbol,
                         strike, expiry_date, qty, entry_price, delta_at_entry, iv_at_entry)
                        VALUES (?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?)
                    """, (position_id, variant_id, leg_type, tsym,
                          strike, expiry_date, qty, entry_price, delta_v, iv_v))

                conn.commit()
                return position_id
            finally:
                conn.close()

    def get_open_positions(self, variant_id: str = None) -> List[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                if variant_id:
                    rows = conn.execute(
                        "SELECT * FROM strangle_positions "
                        "WHERE status='OPEN' AND variant_id=? "
                        "ORDER BY entry_ts DESC",
                        (variant_id,)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM strangle_positions WHERE status='OPEN' "
                        "ORDER BY entry_ts DESC"
                    ).fetchall()
                positions = [dict(r) for r in rows]
                # Attach legs
                for p in positions:
                    legs = conn.execute(
                        "SELECT * FROM strangle_legs WHERE position_id=? ORDER BY id",
                        (p['id'],)
                    ).fetchall()
                    p['legs'] = [dict(l) for l in legs]
                return positions
            finally:
                conn.close()

    def get_position(self, position_id: int) -> Optional[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                r = conn.execute(
                    "SELECT * FROM strangle_positions WHERE id=?", (position_id,)
                ).fetchone()
                if not r:
                    return None
                p = dict(r)
                legs = conn.execute(
                    "SELECT * FROM strangle_legs WHERE position_id=? ORDER BY id",
                    (position_id,)
                ).fetchall()
                p['legs'] = [dict(l) for l in legs]
                return p
            finally:
                conn.close()

    def update_leg_mtm(self, position_id: int, leg_type: str, mtm_now: float):
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE strangle_legs SET mtm_now=? "
                    "WHERE position_id=? AND leg_type=?",
                    (mtm_now, position_id, leg_type),
                )
                conn.commit()
            finally:
                conn.close()

    def close_position(self, position_id: int, exit_date: str, exit_ts: str,
                       pe_exit_price: float, ce_exit_price: float,
                       spot_at_exit: float, exit_reason: str,
                       costs: float = 0.0) -> Dict:
        """
        Close a strangle position. P&L per leg (we sell both legs):
            pe_pnl  = (pe_entry - pe_exit)  * qty
            ce_pnl  = (ce_entry - ce_exit)  * qty
            gross   = pe_pnl + ce_pnl
            net     = gross - costs
        """
        with db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM strangle_positions WHERE id=?", (position_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"Position {position_id} not found")
                pos = dict(row)
                qty = pos['qty']

                pe_pnl = round((pos['pe_entry_price'] - pe_exit_price) * qty, 2)
                ce_pnl = round((pos['ce_entry_price'] - ce_exit_price) * qty, 2)
                gross_pnl = round(pe_pnl + ce_pnl, 2)
                net_pnl = round(gross_pnl - (costs or 0.0), 2)

                # Hold minutes
                try:
                    entry_dt = datetime.fromisoformat(pos['entry_ts'])
                    exit_dt = datetime.fromisoformat(exit_ts)
                    hold_min = max(int((exit_dt - entry_dt).total_seconds() / 60), 0)
                except Exception:
                    hold_min = None

                conn.execute("""
                    UPDATE strangle_positions SET
                        status='CLOSED', exit_date=?, exit_ts=?,
                        pe_exit_price=?, ce_exit_price=?, spot_at_exit=?,
                        gross_pnl=?, net_pnl=?, exit_reason=?,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (exit_date, exit_ts, pe_exit_price, ce_exit_price, spot_at_exit,
                      gross_pnl, net_pnl, exit_reason, position_id))

                conn.execute("""
                    UPDATE strangle_legs SET exit_price=?, leg_pnl=?
                    WHERE position_id=? AND leg_type='PE'
                """, (pe_exit_price, pe_pnl, position_id))
                conn.execute("""
                    UPDATE strangle_legs SET exit_price=?, leg_pnl=?
                    WHERE position_id=? AND leg_type='CE'
                """, (ce_exit_price, ce_pnl, position_id))

                # Archive
                conn.execute("""
                    INSERT INTO strangle_trades
                    (position_id, variant_id, entry_date, exit_date, entry_ts, exit_ts,
                     direction, spot_at_entry, spot_at_exit, pe_strike, ce_strike,
                     pe_entry, ce_entry, pe_exit, ce_exit, gross_pnl, net_pnl, costs,
                     exit_reason, hold_minutes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (position_id, pos['variant_id'], pos['entry_date'], exit_date,
                      pos['entry_ts'], exit_ts, pos['direction'],
                      pos['spot_at_entry'], spot_at_exit,
                      pos['pe_strike'], pos['ce_strike'],
                      pos['pe_entry_price'], pos['ce_entry_price'],
                      pe_exit_price, ce_exit_price,
                      gross_pnl, net_pnl, costs, exit_reason, hold_min))

                conn.commit()
                return {
                    'position_id': position_id,
                    'variant_id': pos['variant_id'],
                    'pe_pnl': pe_pnl,
                    'ce_pnl': ce_pnl,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'hold_minutes': hold_min,
                }
            finally:
                conn.close()

    # =========================================================================
    # Trades + stats
    # =========================================================================

    def get_trades(self, variant_id: str = None, limit: int = 50) -> List[Dict]:
        with db_lock:
            conn = self._get_conn()
            try:
                if variant_id:
                    rows = conn.execute(
                        "SELECT * FROM strangle_trades WHERE variant_id=? "
                        "ORDER BY exit_ts DESC LIMIT ?",
                        (variant_id, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM strangle_trades ORDER BY exit_ts DESC LIMIT ?",
                        (limit,)
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_today_pnl(self, variant_id: str = None) -> float:
        with db_lock:
            conn = self._get_conn()
            try:
                if variant_id:
                    r = conn.execute(
                        "SELECT COALESCE(SUM(net_pnl), 0) AS t FROM strangle_trades "
                        "WHERE variant_id=? AND DATE(exit_date)=DATE('now')",
                        (variant_id,)
                    ).fetchone()
                else:
                    r = conn.execute(
                        "SELECT COALESCE(SUM(net_pnl), 0) AS t FROM strangle_trades "
                        "WHERE DATE(exit_date)=DATE('now')"
                    ).fetchone()
                return float(r['t'] or 0.0)
            finally:
                conn.close()

    def get_stats(self, variant_id: str = None) -> Dict:
        with db_lock:
            conn = self._get_conn()
            try:
                where = "WHERE variant_id=?" if variant_id else ""
                params = (variant_id,) if variant_id else ()
                total_row = conn.execute(
                    f"SELECT COUNT(*) AS c, COALESCE(SUM(net_pnl), 0) AS pnl "
                    f"FROM strangle_trades {where}", params
                ).fetchone()
                total = total_row['c']
                total_pnl = float(total_row['pnl'] or 0.0)
                if total == 0:
                    return {
                        'total_trades': 0, 'wins': 0, 'losses': 0,
                        'win_rate': 0.0, 'total_pnl': 0.0, 'avg_pnl': 0.0,
                        'profit_factor': 0.0,
                    }
                wins = conn.execute(
                    f"SELECT COUNT(*) AS c FROM strangle_trades {where}"
                    f"{' AND' if where else 'WHERE'} net_pnl>0", params
                ).fetchone()['c']
                gp = conn.execute(
                    f"SELECT COALESCE(SUM(net_pnl),0) AS v FROM strangle_trades {where}"
                    f"{' AND' if where else 'WHERE'} net_pnl>0", params
                ).fetchone()['v']
                gl = conn.execute(
                    f"SELECT COALESCE(ABS(SUM(net_pnl)),0) AS v FROM strangle_trades {where}"
                    f"{' AND' if where else 'WHERE'} net_pnl<0", params
                ).fetchone()['v']
                pf = (gp / gl) if gl > 0 else float('inf')
                return {
                    'total_trades': total,
                    'wins': wins,
                    'losses': total - wins,
                    'win_rate': round(wins / total * 100, 1),
                    'total_pnl': round(total_pnl, 2),
                    'avg_pnl': round(total_pnl / total, 2) if total else 0.0,
                    'profit_factor': round(pf, 2) if pf != float('inf') else 'inf',
                }
            finally:
                conn.close()

    def get_equity_curve(self, variant_id: str) -> List[Dict]:
        """Cumulative net P/L over time for one variant."""
        with db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT exit_date, exit_ts, net_pnl, exit_reason "
                    "FROM strangle_trades WHERE variant_id=? "
                    "ORDER BY exit_ts ASC, id ASC",
                    (variant_id,)
                ).fetchall()
                curve = []
                cum = 0.0
                for r in rows:
                    cum += float(r['net_pnl'] or 0.0)
                    curve.append({
                        'date': r['exit_date'],
                        'ts': r['exit_ts'],
                        'pnl': float(r['net_pnl'] or 0.0),
                        'cumulative_pnl': round(cum, 2),
                        'exit_reason': r['exit_reason'],
                    })
                return curve
            finally:
                conn.close()

    # =========================================================================
    # Daily state
    # =========================================================================

    def upsert_daily_state(self, variant_id: str, trade_date: str, **fields):
        """Insert or update today's snapshot for a variant. Pass keyword args for
        the columns you want to set."""
        cols = [
            'or_high', 'or_low', 'or_width_pct', 'day_filter_passed',
            'signal_seen', 'rsi_confirmed', 'entry_taken', 'exit_taken',
            'exit_reason', 'daily_pnl', 'notes', 'last_event_ts',
        ]
        vals = {c: fields.get(c) for c in cols}
        with db_lock:
            conn = self._get_conn()
            try:
                # Bool-coerce
                for k in ('day_filter_passed', 'signal_seen', 'rsi_confirmed',
                          'entry_taken', 'exit_taken'):
                    if vals[k] is not None:
                        vals[k] = 1 if vals[k] else 0
                conn.execute(f"""
                    INSERT INTO strangle_daily_state
                        (variant_id, trade_date, or_high, or_low, or_width_pct,
                         day_filter_passed, signal_seen, rsi_confirmed,
                         entry_taken, exit_taken, exit_reason, daily_pnl, notes,
                         last_event_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(variant_id, trade_date) DO UPDATE SET
                        or_high=COALESCE(excluded.or_high, or_high),
                        or_low=COALESCE(excluded.or_low, or_low),
                        or_width_pct=COALESCE(excluded.or_width_pct, or_width_pct),
                        day_filter_passed=COALESCE(excluded.day_filter_passed, day_filter_passed),
                        signal_seen=COALESCE(excluded.signal_seen, signal_seen),
                        rsi_confirmed=COALESCE(excluded.rsi_confirmed, rsi_confirmed),
                        entry_taken=COALESCE(excluded.entry_taken, entry_taken),
                        exit_taken=COALESCE(excluded.exit_taken, exit_taken),
                        exit_reason=COALESCE(excluded.exit_reason, exit_reason),
                        daily_pnl=COALESCE(excluded.daily_pnl, daily_pnl),
                        notes=COALESCE(excluded.notes, notes),
                        last_event_ts=COALESCE(excluded.last_event_ts, last_event_ts)
                """, (variant_id, trade_date,
                      vals['or_high'], vals['or_low'], vals['or_width_pct'],
                      vals['day_filter_passed'], vals['signal_seen'],
                      vals['rsi_confirmed'], vals['entry_taken'],
                      vals['exit_taken'], vals['exit_reason'],
                      vals['daily_pnl'], vals['notes'], vals['last_event_ts']))
                conn.commit()
            finally:
                conn.close()

    def get_daily_state(self, variant_id: str, trade_date: str = None) -> Optional[Dict]:
        if trade_date is None:
            trade_date = date.today().isoformat()
        with db_lock:
            conn = self._get_conn()
            try:
                r = conn.execute(
                    "SELECT * FROM strangle_daily_state "
                    "WHERE variant_id=? AND trade_date=?",
                    (variant_id, trade_date)
                ).fetchone()
                return dict(r) if r else None
            finally:
                conn.close()


# Singleton ----------------------------------------------------------------
_instance = None
_instance_lock = threading.Lock()


def get_strangle_db() -> StrangleTradingDB:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = StrangleTradingDB()
    return _instance
