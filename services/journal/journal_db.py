"""
Journal SQLite persistence layer.

Single DB at backtest_data/journal.db. Schema is the projection over all
strategy DBs plus journal-only enrichment (tags, notes, grades, reviews).

See docs/Design/TRADING-JOURNAL-DESIGN.md for the full contract.
"""

import sqlite3
import threading
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / 'journal.db'

# Coarse lock — journal writes are infrequent (sync, tag, note edits)
db_lock = threading.Lock()


SCHEMA_SQL = """
-- 1. Master trade row, one per round-trip across all strategies
CREATE TABLE IF NOT EXISTS journal_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_db       TEXT NOT NULL,
    source_table    TEXT NOT NULL,
    source_id       INTEGER,
    strategy        TEXT NOT NULL,
    instrument      TEXT NOT NULL,
    instrument_type TEXT NOT NULL,
    direction       TEXT NOT NULL,
    qty             INTEGER NOT NULL,
    entry_price     REAL NOT NULL,
    entry_time      TIMESTAMP NOT NULL,
    exit_price      REAL,
    exit_time       TIMESTAMP,
    exit_reason     TEXT,
    pnl_gross       REAL,
    pnl_charges     REAL,
    pnl_net         REAL,
    r_multiple      REAL,
    initial_risk    REAL,
    hold_minutes    INTEGER,
    mode            TEXT NOT NULL DEFAULT 'LIVE',
    grade           INTEGER,
    mistake_flag    INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_db, source_table, source_id)
);

CREATE INDEX IF NOT EXISTS idx_jt_strategy ON journal_trades(strategy);
CREATE INDEX IF NOT EXISTS idx_jt_entry_date ON journal_trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_jt_instrument ON journal_trades(instrument);

-- 2. Per-trade context (regime, gap, MAE/MFE, slippage)
CREATE TABLE IF NOT EXISTS journal_trade_context (
    trade_id         INTEGER PRIMARY KEY,
    nifty_5m_trend   TEXT,
    nifty_daily_adx  REAL,
    nifty_gap_pct    REAL,
    india_vix        REAL,
    gap_tier         TEXT,
    cpr_width_pct    REAL,
    rsi_at_entry     REAL,
    vwap_at_entry    REAL,
    mae              REAL,
    mfe              REAL,
    slippage_entry   REAL,
    slippage_exit    REAL,
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE CASCADE
);

-- 3. Notes (markdown body — single row per trade for MVP)
CREATE TABLE IF NOT EXISTS journal_trade_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id    INTEGER NOT NULL UNIQUE,
    body_md     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE CASCADE
);

-- 4. Screenshots (paths only, never blobs)
CREATE TABLE IF NOT EXISTS journal_trade_screenshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id    INTEGER NOT NULL,
    file_path   TEXT NOT NULL,
    caption     TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE CASCADE
);

-- 5. Tag dictionary
CREATE TABLE IF NOT EXISTS journal_tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    category    TEXT NOT NULL,
    color_hex   TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Many-to-many tag <-> trade
CREATE TABLE IF NOT EXISTS journal_trade_tags (
    trade_id    INTEGER NOT NULL,
    tag_id      INTEGER NOT NULL,
    PRIMARY KEY (trade_id, tag_id),
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id)   REFERENCES journal_tags(id)   ON DELETE CASCADE
);

-- 7. Daily review (one row per trading day)
CREATE TABLE IF NOT EXISTS journal_daily_review (
    trade_date     DATE PRIMARY KEY,
    pre_market_md  TEXT,
    post_close_md  TEXT,
    rule_violations INTEGER DEFAULT 0,
    discipline_score INTEGER,
    nifty_close    REAL,
    nifty_chg_pct  REAL,
    pnl_gross      REAL,
    pnl_net        REAL,
    trades_count   INTEGER,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. Materialised daily summary (per-strategy + per-day)
CREATE TABLE IF NOT EXISTS journal_daily_summary (
    trade_date     DATE NOT NULL,
    strategy       TEXT NOT NULL,
    trades_count   INTEGER,
    wins           INTEGER,
    losses         INTEGER,
    pnl_gross      REAL,
    pnl_net        REAL,
    largest_win    REAL,
    largest_loss   REAL,
    avg_r          REAL,
    PRIMARY KEY (trade_date, strategy)
);

-- 9. Kite reconciliation (Phase 2 placeholder; create now to lock schema)
CREATE TABLE IF NOT EXISTS journal_kite_reconciliation (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        INTEGER,
    kite_order_id   TEXT,
    kite_trade_id   TEXT,
    kite_symbol     TEXT,
    kite_side       TEXT,
    kite_qty        INTEGER,
    kite_price      REAL,
    kite_filled_at  TIMESTAMP,
    status          TEXT NOT NULL,
    reviewed        INTEGER DEFAULT 0,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES journal_trades(id) ON DELETE SET NULL
);

-- (system_grade table reserved for Phase 3 weekly grading)
CREATE TABLE IF NOT EXISTS journal_system_grade (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    week_iso      TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    grade         TEXT NOT NULL,  -- GREEN | AMBER | RED
    trades_count  INTEGER,
    win_rate      REAL,
    profit_factor REAL,
    drift_score   REAL,
    notes         TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (week_iso, strategy)
);
"""


# Default seed tags — created once on first init.
SEED_TAGS = [
    # SETUP
    ('breakout', 'SETUP', '#1E3A8A'),
    ('mean-reversion', 'SETUP', '#0F6E56'),
    ('trend-continuation', 'SETUP', '#B45309'),
    ('gap-fill', 'SETUP', '#6B21A8'),
    # CONVICTION
    ('high-conviction', 'CONVICTION', '#0F6E56'),
    ('a-grade', 'CONVICTION', '#1E3A8A'),
    ('low-conviction', 'CONVICTION', '#888780'),
    # MISTAKE
    ('moved-stop', 'MISTAKE', '#A32D2D'),
    ('early-entry', 'MISTAKE', '#A32D2D'),
    ('oversized', 'MISTAKE', '#A32D2D'),
    ('fomo-entry', 'MISTAKE', '#A32D2D'),
    ('override-against-system', 'MISTAKE', '#A32D2D'),
    ('manual-exit-too-early', 'MISTAKE', '#A32D2D'),
    # CUSTOM
    ('flagship', 'CUSTOM', '#8C6A20'),
    ('simulation', 'CUSTOM', '#888780'),
]


class JournalDB:
    """SQLite manager for the trading journal."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path))
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA foreign_keys = ON')
        return c

    def _init_schema(self):
        with db_lock:
            with self._conn() as c:
                c.executescript(SCHEMA_SQL)
                # Seed tags if empty
                row = c.execute('SELECT COUNT(*) AS n FROM journal_tags').fetchone()
                if row['n'] == 0:
                    c.executemany(
                        'INSERT OR IGNORE INTO journal_tags(name, category, color_hex) VALUES (?,?,?)',
                        SEED_TAGS,
                    )

    # ------------------------------------------------------------------
    # Trade upsert (used by sync sources)
    # ------------------------------------------------------------------
    def upsert_trade(self, payload: Dict[str, Any]) -> int:
        """Insert if (source_db, source_table, source_id) is new, else update.
        Returns the journal trade id.
        """
        cols = [
            'source_db', 'source_table', 'source_id', 'strategy', 'instrument',
            'instrument_type', 'direction', 'qty', 'entry_price', 'entry_time',
            'exit_price', 'exit_time', 'exit_reason', 'pnl_gross', 'pnl_charges',
            'pnl_net', 'r_multiple', 'initial_risk', 'hold_minutes', 'mode',
        ]
        vals = [payload.get(k) for k in cols]
        with db_lock:
            with self._conn() as c:
                # Try to fetch existing
                row = c.execute(
                    'SELECT id FROM journal_trades WHERE source_db=? AND source_table=? AND source_id IS ?',
                    (payload.get('source_db'), payload.get('source_table'), payload.get('source_id')),
                ).fetchone()
                if row:
                    set_clause = ', '.join([f'{k}=?' for k in cols])
                    c.execute(
                        f'UPDATE journal_trades SET {set_clause}, updated_at=CURRENT_TIMESTAMP WHERE id=?',
                        vals + [row['id']],
                    )
                    return int(row['id'])
                placeholders = ','.join(['?'] * len(cols))
                cur = c.execute(
                    f'INSERT INTO journal_trades({",".join(cols)}) VALUES ({placeholders})',
                    vals,
                )
                return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # Read APIs
    # ------------------------------------------------------------------
    def list_trades(self,
                    date_from: Optional[str] = None,
                    date_to: Optional[str] = None,
                    strategy: Optional[str] = None,
                    instrument: Optional[str] = None,
                    mode: Optional[str] = None,
                    limit: int = 200,
                    offset: int = 0) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if date_from:
            clauses.append("date(entry_time) >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("date(entry_time) <= ?")
            params.append(date_to)
        if strategy and strategy != 'ALL':
            clauses.append("strategy = ?")
            params.append(strategy)
        if instrument:
            clauses.append("instrument = ?")
            params.append(instrument)
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
        sql = f'SELECT * FROM journal_trades {where} ORDER BY entry_time DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def get_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute('SELECT * FROM journal_trades WHERE id=?', (trade_id,)).fetchone()
            if not row:
                return None
            trade = dict(row)
            ctx = c.execute('SELECT * FROM journal_trade_context WHERE trade_id=?', (trade_id,)).fetchone()
            trade['context'] = dict(ctx) if ctx else None
            note = c.execute('SELECT body_md, updated_at FROM journal_trade_notes WHERE trade_id=?', (trade_id,)).fetchone()
            trade['notes'] = dict(note) if note else None
            shots = c.execute('SELECT id, file_path, caption, created_at FROM journal_trade_screenshots WHERE trade_id=? ORDER BY id', (trade_id,)).fetchall()
            trade['screenshots'] = [dict(r) for r in shots]
            tags = c.execute(
                """
                SELECT t.id, t.name, t.category, t.color_hex
                FROM journal_tags t
                JOIN journal_trade_tags jt ON jt.tag_id = t.id
                WHERE jt.trade_id = ?
                ORDER BY t.category, t.name
                """,
                (trade_id,)
            ).fetchall()
            trade['tags'] = [dict(r) for r in tags]
            return trade

    def update_trade(self, trade_id: int, fields: Dict[str, Any]) -> bool:
        allowed = {'grade', 'mistake_flag', 'mode', 'exit_price', 'exit_reason',
                   'pnl_net', 'pnl_gross', 'pnl_charges', 'r_multiple'}
        sets = []
        vals = []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f'{k}=?')
                vals.append(v)
        if not sets:
            return False
        vals.append(trade_id)
        with db_lock:
            with self._conn() as c:
                c.execute(
                    f"UPDATE journal_trades SET {', '.join(sets)}, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    vals,
                )
        return True

    # ------------------------------------------------------------------
    # Calendar / summary
    # ------------------------------------------------------------------
    def daily_summary(self, date_from: str, date_to: str, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        params: List[Any] = [date_from, date_to]
        strat_clause = ''
        if strategy and strategy != 'ALL':
            strat_clause = ' AND strategy = ?'
            params.append(strategy)
        sql = f"""
            SELECT
                date(entry_time) AS trade_date,
                COUNT(*) AS trades,
                SUM(CASE WHEN COALESCE(pnl_net, pnl_gross, 0) > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN COALESCE(pnl_net, pnl_gross, 0) < 0 THEN 1 ELSE 0 END) AS losses,
                COALESCE(SUM(pnl_gross), 0) AS pnl_gross,
                COALESCE(SUM(pnl_net), 0) AS pnl_net,
                MAX(COALESCE(pnl_net, pnl_gross, 0)) AS best,
                MIN(COALESCE(pnl_net, pnl_gross, 0)) AS worst
            FROM journal_trades
            WHERE date(entry_time) BETWEEN ? AND ?
              AND exit_time IS NOT NULL
              {strat_clause}
            GROUP BY date(entry_time)
            ORDER BY trade_date
        """
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    def save_note(self, trade_id: int, body_md: str) -> None:
        with db_lock:
            with self._conn() as c:
                c.execute(
                    """
                    INSERT INTO journal_trade_notes(trade_id, body_md)
                    VALUES (?, ?)
                    ON CONFLICT(trade_id) DO UPDATE SET
                        body_md = excluded.body_md,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (trade_id, body_md),
                )

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------
    def list_tags(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                'SELECT id, name, category, color_hex FROM journal_tags ORDER BY category, name'
            ).fetchall()]

    def create_tag(self, name: str, category: str, color_hex: Optional[str] = None) -> int:
        with db_lock:
            with self._conn() as c:
                cur = c.execute(
                    'INSERT OR IGNORE INTO journal_tags(name, category, color_hex) VALUES(?,?,?)',
                    (name, category, color_hex),
                )
                if cur.lastrowid:
                    return int(cur.lastrowid)
                row = c.execute('SELECT id FROM journal_tags WHERE name=?', (name,)).fetchone()
                return int(row['id'])

    def attach_tags(self, trade_id: int, tag_ids: List[int]) -> None:
        with db_lock:
            with self._conn() as c:
                for tid in tag_ids:
                    c.execute(
                        'INSERT OR IGNORE INTO journal_trade_tags(trade_id, tag_id) VALUES(?,?)',
                        (trade_id, tid),
                    )

    def detach_tag(self, trade_id: int, tag_id: int) -> None:
        with db_lock:
            with self._conn() as c:
                c.execute(
                    'DELETE FROM journal_trade_tags WHERE trade_id=? AND tag_id=?',
                    (trade_id, tag_id),
                )

    # ------------------------------------------------------------------
    # Daily review
    # ------------------------------------------------------------------
    def get_daily_review(self, trade_date: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                'SELECT * FROM journal_daily_review WHERE trade_date=?',
                (trade_date,),
            ).fetchone()
            return dict(row) if row else None

    def save_daily_review(self, trade_date: str, fields: Dict[str, Any]) -> None:
        allowed = {'pre_market_md', 'post_close_md', 'rule_violations', 'discipline_score'}
        clean = {k: v for k, v in fields.items() if k in allowed}
        if not clean:
            return
        cols = ['trade_date'] + list(clean.keys())
        vals = [trade_date] + list(clean.values())
        placeholders = ','.join(['?'] * len(cols))
        sets = ', '.join([f'{k}=excluded.{k}' for k in clean.keys()])
        with db_lock:
            with self._conn() as c:
                c.execute(
                    f"""
                    INSERT INTO journal_daily_review({','.join(cols)})
                    VALUES ({placeholders})
                    ON CONFLICT(trade_date) DO UPDATE SET
                        {sets},
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    vals,
                )

    # ------------------------------------------------------------------
    # Insights bundle data
    # ------------------------------------------------------------------
    def equity_curve(self, date_from: Optional[str] = None,
                     date_to: Optional[str] = None,
                     strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = ['exit_time IS NOT NULL']
        params: List[Any] = []
        if date_from:
            clauses.append("date(entry_time) >= ?"); params.append(date_from)
        if date_to:
            clauses.append("date(entry_time) <= ?"); params.append(date_to)
        if strategy and strategy != 'ALL':
            clauses.append("strategy = ?"); params.append(strategy)
        where = 'WHERE ' + ' AND '.join(clauses)
        with self._conn() as c:
            rows = c.execute(
                f"""
                SELECT date(entry_time) AS d,
                       COALESCE(SUM(pnl_net), 0) AS day_net,
                       COUNT(*) AS trades
                FROM journal_trades
                {where}
                GROUP BY date(entry_time)
                ORDER BY d
                """,
                params,
            ).fetchall()
        running = 0.0
        out = []
        for r in rows:
            running += float(r['day_net'] or 0)
            out.append({'date': r['d'], 'day_net': r['day_net'], 'cum_net': running, 'trades': r['trades']})
        return out

    def per_strategy_attribution(self, date_from: Optional[str] = None,
                                 date_to: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = ['exit_time IS NOT NULL']
        params: List[Any] = []
        if date_from:
            clauses.append("date(entry_time) >= ?"); params.append(date_from)
        if date_to:
            clauses.append("date(entry_time) <= ?"); params.append(date_to)
        where = 'WHERE ' + ' AND '.join(clauses)
        with self._conn() as c:
            rows = c.execute(
                f"""
                SELECT strategy,
                       COUNT(*) AS trades,
                       SUM(CASE WHEN COALESCE(pnl_net,pnl_gross,0) > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN COALESCE(pnl_net,pnl_gross,0) < 0 THEN 1 ELSE 0 END) AS losses,
                       COALESCE(SUM(pnl_net),0) AS pnl_net,
                       COALESCE(SUM(pnl_gross),0) AS pnl_gross,
                       AVG(r_multiple) AS avg_r,
                       SUM(CASE WHEN COALESCE(pnl_net,pnl_gross,0) > 0 THEN COALESCE(pnl_net,pnl_gross,0) ELSE 0 END) AS sum_wins,
                       SUM(CASE WHEN COALESCE(pnl_net,pnl_gross,0) < 0 THEN COALESCE(pnl_net,pnl_gross,0) ELSE 0 END) AS sum_losses
                FROM journal_trades
                {where}
                GROUP BY strategy
                ORDER BY pnl_net DESC
                """,
                params,
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            t = d['trades'] or 1
            d['win_rate'] = round((d['wins'] or 0) / t * 100, 2)
            sw = float(d['sum_wins'] or 0); sl = abs(float(d['sum_losses'] or 0))
            d['profit_factor'] = round(sw / sl, 2) if sl > 0 else None
            out.append(d)
        return out

    def r_distribution(self, date_from: Optional[str] = None,
                       date_to: Optional[str] = None,
                       strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = ['exit_time IS NOT NULL', 'r_multiple IS NOT NULL']
        params: List[Any] = []
        if date_from:
            clauses.append("date(entry_time) >= ?"); params.append(date_from)
        if date_to:
            clauses.append("date(entry_time) <= ?"); params.append(date_to)
        if strategy and strategy != 'ALL':
            clauses.append("strategy = ?"); params.append(strategy)
        where = 'WHERE ' + ' AND '.join(clauses)
        with self._conn() as c:
            rows = c.execute(
                f'SELECT r_multiple FROM journal_trades {where}',
                params,
            ).fetchall()
        return [{'r': float(r['r_multiple'])} for r in rows]

    def win_rate_by_tag(self, date_from: Optional[str] = None,
                        date_to: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = ['t.exit_time IS NOT NULL']
        params: List[Any] = []
        if date_from:
            clauses.append("date(t.entry_time) >= ?"); params.append(date_from)
        if date_to:
            clauses.append("date(t.entry_time) <= ?"); params.append(date_to)
        where = 'WHERE ' + ' AND '.join(clauses)
        with self._conn() as c:
            rows = c.execute(
                f"""
                SELECT g.name, g.category, g.color_hex,
                       COUNT(*) AS trades,
                       SUM(CASE WHEN COALESCE(t.pnl_net,t.pnl_gross,0) > 0 THEN 1 ELSE 0 END) AS wins,
                       COALESCE(SUM(t.pnl_net), 0) AS pnl_net
                FROM journal_trades t
                JOIN journal_trade_tags jt ON jt.trade_id = t.id
                JOIN journal_tags g ON g.id = jt.tag_id
                {where}
                GROUP BY g.id
                ORDER BY g.category, g.name
                """,
                params,
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            t = d['trades'] or 1
            d['win_rate'] = round((d['wins'] or 0) / t * 100, 2)
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Drawdown windows (top 5)
    # ------------------------------------------------------------------
    def drawdown_windows(self, date_from: Optional[str] = None,
                         date_to: Optional[str] = None) -> List[Dict[str, Any]]:
        eq = self.equity_curve(date_from, date_to)
        if not eq:
            return []
        peak = -1e18
        peak_date = eq[0]['date']
        windows = []
        cur_window = None
        for row in eq:
            cum = row['cum_net']
            if cum >= peak:
                if cur_window:
                    cur_window['recovery_date'] = row['date']
                    cur_window['duration_days'] = _days_between(cur_window['start_date'], row['date'])
                    windows.append(cur_window)
                    cur_window = None
                peak = cum
                peak_date = row['date']
            else:
                dd_amt = peak - cum
                if cur_window is None:
                    cur_window = {
                        'start_date': peak_date,
                        'trough_date': row['date'],
                        'depth': -dd_amt,
                        'recovery_date': None,
                        'duration_days': None,
                    }
                else:
                    if -dd_amt < cur_window['depth']:
                        cur_window['depth'] = -dd_amt
                        cur_window['trough_date'] = row['date']
        if cur_window:
            windows.append(cur_window)
        windows.sort(key=lambda w: w['depth'])
        return windows[:5]


def _days_between(d1: str, d2: str) -> int:
    try:
        a = datetime.strptime(d1, '%Y-%m-%d').date()
        b = datetime.strptime(d2, '%Y-%m-%d').date()
        return (b - a).days
    except Exception:
        return 0


_journal_db_instance: Optional[JournalDB] = None


def get_journal_db() -> JournalDB:
    """Singleton accessor."""
    global _journal_db_instance
    if _journal_db_instance is None:
        _journal_db_instance = JournalDB()
    return _journal_db_instance
