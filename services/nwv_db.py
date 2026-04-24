"""
NWV — Nifty Weekly View — Database Layer
==========================================
Persistent store for the Nifty Weekly View system (Phase 0: view-only).

Tables:
  - nwv_weekly_state  one row per Monday-starting week: prev-week H/L/C,
                      weekly pivots (S2/S1/PP/R1/R2), CPR levels, CPR
                      width bucket, monthly overlay.
  - nwv_views         one row per Monday 09:46: first-candle metrics,
                      gap tier, all enhancement inputs, final view,
                      conviction, suggested trade, expected range.

Pattern mirrors services/orb_db.py — singleton, thread-safe, WAL mode.
DB file lives at backtest_data/nwv_trading.db (gitignored per F4 of the
MONDAY bundle).
"""

import os
import sqlite3
import threading
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
DATA_DIR = Path(_volume_path) / "data" if _volume_path else Path(__file__).parent.parent / "backtest_data"
DB_PATH = str(DATA_DIR / "nwv_trading.db")


class NwvDB:
    """SQLite persistence for the Nifty Weekly View system."""

    def __init__(self):
        self.db_path = DB_PATH
        self.db_lock = threading.Lock()
        self._init_database()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ─── schema ───────────────────────────────────────────────

    def _init_database(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS nwv_weekly_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        week_start DATE NOT NULL UNIQUE,       -- Monday of the trade week
                        prev_week_high REAL NOT NULL,
                        prev_week_low  REAL NOT NULL,
                        prev_week_close REAL NOT NULL,
                        prev_fri_close REAL,                   -- strictly last Friday's close for gap calc
                        pivot_pp REAL NOT NULL,
                        pivot_s1 REAL NOT NULL,
                        pivot_s2 REAL NOT NULL,
                        pivot_r1 REAL NOT NULL,
                        pivot_r2 REAL NOT NULL,
                        cpr_tc REAL NOT NULL,
                        cpr_bc REAL NOT NULL,
                        cpr_width_pct REAL NOT NULL,           -- abs(TC-BC)/spot * 100
                        cpr_bucket TEXT NOT NULL,              -- wide / normal / narrow
                        monthly_tc REAL,
                        monthly_bc REAL,
                        monthly_pivot REAL,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_nwv_week_start
                        ON nwv_weekly_state(week_start);

                    CREATE TABLE IF NOT EXISTS nwv_views (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        week_start DATE NOT NULL,              -- FK-ish to nwv_weekly_state.week_start
                        generated_at TIMESTAMP NOT NULL,

                        -- Monday open + 1st 30-min candle
                        mon_open REAL,
                        first_candle_open REAL,
                        first_candle_high REAL,
                        first_candle_low REAL,
                        first_candle_close REAL,
                        first_candle_volume INTEGER,
                        first_candle_body TEXT,                -- bullish / bearish / doji
                        first_candle_pos  TEXT,                -- above_cpr / below_cpr / inside_cpr
                        first_candle_range_pct REAL,
                        first_candle_wick_pos_pct REAL,        -- where body sits within range (0=bottom, 100=top)

                        -- Gap
                        gap_pct REAL,
                        gap_tier TEXT,                         -- none / small / considerable
                        gap_direction TEXT,                    -- up / down

                        -- Enhancements
                        vix_value REAL,
                        vix_pct_rank REAL,                     -- 60-day percentile 0..100
                        adx_daily REAL,
                        adx_bucket TEXT,                       -- chop / light / building / trend

                        -- Monthly overlay
                        monthly_override_side TEXT,            -- bullish / bearish / null
                        monthly_override_applied INTEGER DEFAULT 0,

                        -- Stacked levels (pivot clusters)
                        stacked_supports TEXT,                 -- JSON list of {price, components}
                        stacked_resistances TEXT,              -- JSON list

                        -- Final
                        base_view TEXT NOT NULL,
                        final_view TEXT NOT NULL,
                        conviction INTEGER NOT NULL,           -- 0..5
                        instrument_choice TEXT,                -- strangle / put_debit_spread / call_debit_spread / none
                        expected_range_low REAL,
                        expected_range_high REAL,
                        time_stop TEXT,                        -- "Fri 15:15"
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_nwv_views_week
                        ON nwv_views(week_start);
                    CREATE INDEX IF NOT EXISTS idx_nwv_views_created
                        ON nwv_views(created_at);
                """)
                conn.commit()
                logger.info(f"[NWV] Database initialized at {self.db_path}")
            finally:
                conn.close()

    # ─── weekly_state CRUD ────────────────────────────────────

    def upsert_weekly_state(self, row: Dict[str, Any]) -> int:
        """Insert or replace a week's state. Returns row id."""
        cols = ', '.join(row.keys())
        placeholders = ', '.join(['?'] * len(row))
        update_clause = ', '.join(f"{k}=excluded.{k}" for k in row if k != 'week_start')
        sql = (f"INSERT INTO nwv_weekly_state ({cols}) VALUES ({placeholders}) "
               f"ON CONFLICT(week_start) DO UPDATE SET {update_clause}")
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(sql, list(row.values()))
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_weekly_state(self, week_start: date) -> Optional[Dict[str, Any]]:
        if isinstance(week_start, date) and not isinstance(week_start, datetime):
            week_start = week_start.isoformat()
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM nwv_weekly_state WHERE week_start = ?",
                    (week_start,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def latest_weekly_state(self) -> Optional[Dict[str, Any]]:
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM nwv_weekly_state "
                    "ORDER BY week_start DESC LIMIT 1"
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    # ─── views CRUD ───────────────────────────────────────────

    def insert_view(self, row: Dict[str, Any]) -> int:
        cols = ', '.join(row.keys())
        placeholders = ', '.join(['?'] * len(row))
        with self.db_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    f"INSERT INTO nwv_views ({cols}) VALUES ({placeholders})",
                    list(row.values())
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def latest_view(self) -> Optional[Dict[str, Any]]:
        with self.db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM nwv_views ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def recent_views(self, n: int = 20) -> List[Dict[str, Any]]:
        with self.db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM nwv_views ORDER BY created_at DESC LIMIT ?",
                    (int(n),)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()


# ─── module-level singleton ───────────────────────────────────

_db_singleton: Optional[NwvDB] = None
_db_lock = threading.Lock()


def get_nwv_db() -> NwvDB:
    global _db_singleton
    if _db_singleton is None:
        with _db_lock:
            if _db_singleton is None:
                _db_singleton = NwvDB()
    return _db_singleton
