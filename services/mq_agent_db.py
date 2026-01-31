"""
MQ Agent State Database
========================

Persistence layer for the Momentum + Quality agent system.
Stores portfolio state, agent execution logs, and signal queue.

Tables:
- mq_portfolio: Current portfolio state (single active row)
- mq_agent_runs: Agent execution history
- mq_signals: Active/resolved signal queue
"""

import sqlite3
import json
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Database configuration
DB_DIR = Path(__file__).parent.parent / 'backtest_data'
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / 'mq_agent.db'

# Thread lock for database operations
db_lock = threading.Lock()


class MQAgentDB:
    """Manager for MQ agent state persistence."""

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
                    CREATE TABLE IF NOT EXISTS mq_portfolio (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        initial_capital DECIMAL(15,2) NOT NULL,
                        equity_deployed DECIMAL(15,2) DEFAULT 0,
                        debt_fund_balance DECIMAL(15,2) DEFAULT 0,
                        total_value DECIMAL(15,2) DEFAULT 0,
                        positions_count INTEGER DEFAULT 0,
                        status VARCHAR(20) DEFAULT 'active'
                    );

                    CREATE TABLE IF NOT EXISTS mq_agent_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_type VARCHAR(30) NOT NULL,
                        run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        duration_seconds REAL,
                        status VARCHAR(20) DEFAULT 'running',
                        signals_count INTEGER DEFAULT 0,
                        report_path VARCHAR(255),
                        report_type VARCHAR(30),
                        summary TEXT,
                        error_message TEXT
                    );

                    CREATE TABLE IF NOT EXISTS mq_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_run_id INTEGER,
                        signal_type VARCHAR(30) NOT NULL,
                        symbol VARCHAR(20),
                        priority VARCHAR(10) DEFAULT 'MEDIUM',
                        details TEXT,
                        status VARCHAR(20) DEFAULT 'pending',
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved_date TIMESTAMP,
                        FOREIGN KEY (agent_run_id) REFERENCES mq_agent_runs(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_signals_status
                        ON mq_signals(status);
                    CREATE INDEX IF NOT EXISTS idx_signals_type
                        ON mq_signals(signal_type);
                    CREATE INDEX IF NOT EXISTS idx_runs_type
                        ON mq_agent_runs(agent_type);
                    CREATE INDEX IF NOT EXISTS idx_runs_date
                        ON mq_agent_runs(run_date);

                    CREATE TABLE IF NOT EXISTS mq_regime (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        regime VARCHAR(20) DEFAULT 'UNKNOWN',
                        index_close REAL DEFAULT 0,
                        index_200dma REAL DEFAULT 0,
                        vix REAL,
                        above_200dma BOOLEAN DEFAULT 1,
                        vix_ok BOOLEAN DEFAULT 1
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Market Regime
    # =========================================================================

    def save_regime(self, regime: str, index_close: float, index_200dma: float,
                    vix: float = None, above_200dma: bool = True, vix_ok: bool = True):
        """Save current market regime (upsert single row)."""
        with db_lock:
            conn = self._get_conn()
            try:
                existing = conn.execute("SELECT id FROM mq_regime ORDER BY id DESC LIMIT 1").fetchone()
                if existing:
                    conn.execute("""
                        UPDATE mq_regime SET
                            regime = ?, index_close = ?, index_200dma = ?,
                            vix = ?, above_200dma = ?, vix_ok = ?,
                            updated_date = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (regime, index_close, index_200dma, vix, above_200dma, vix_ok, existing['id']))
                else:
                    conn.execute("""
                        INSERT INTO mq_regime (regime, index_close, index_200dma, vix, above_200dma, vix_ok)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (regime, index_close, index_200dma, vix, above_200dma, vix_ok))
                conn.commit()
            finally:
                conn.close()

    def get_regime(self) -> Optional[Dict]:
        """Get the latest market regime."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM mq_regime ORDER BY id DESC LIMIT 1").fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # =========================================================================
    # Portfolio State
    # =========================================================================

    def get_portfolio_state(self) -> Optional[Dict]:
        """Get the current active portfolio state."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM mq_portfolio WHERE status = 'active' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def upsert_portfolio_state(
        self,
        initial_capital: float,
        equity_deployed: float,
        debt_fund_balance: float,
        total_value: float,
        positions_count: int,
        status: str = 'active'
    ) -> int:
        """Create or update portfolio state. Returns row id."""
        with db_lock:
            conn = self._get_conn()
            try:
                existing = conn.execute(
                    "SELECT id FROM mq_portfolio WHERE status = 'active' ORDER BY id DESC LIMIT 1"
                ).fetchone()

                if existing:
                    conn.execute("""
                        UPDATE mq_portfolio SET
                            equity_deployed = ?, debt_fund_balance = ?,
                            total_value = ?, positions_count = ?,
                            status = ?, updated_date = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (equity_deployed, debt_fund_balance, total_value,
                          positions_count, status, existing['id']))
                    conn.commit()
                    return existing['id']
                else:
                    cursor = conn.execute("""
                        INSERT INTO mq_portfolio
                            (initial_capital, equity_deployed, debt_fund_balance,
                             total_value, positions_count, status)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (initial_capital, equity_deployed, debt_fund_balance,
                          total_value, positions_count, status))
                    conn.commit()
                    return cursor.lastrowid
            finally:
                conn.close()

    # =========================================================================
    # Agent Runs
    # =========================================================================

    def start_agent_run(self, agent_type: str) -> int:
        """Record the start of an agent run. Returns run_id."""
        with db_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "INSERT INTO mq_agent_runs (agent_type, status) VALUES (?, 'running')",
                    (agent_type,)
                )
                conn.commit()
                run_id = cursor.lastrowid
                logger.info(f"Agent run started: {agent_type} (run_id={run_id})")
                return run_id
            finally:
                conn.close()

    def complete_agent_run(
        self,
        run_id: int,
        signals_count: int = 0,
        report_path: str = None,
        report_type: str = None,
        summary: str = None,
        duration_seconds: float = None
    ):
        """Mark an agent run as completed."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE mq_agent_runs SET
                        status = 'completed',
                        signals_count = ?,
                        report_path = ?,
                        report_type = ?,
                        summary = ?,
                        duration_seconds = ?
                    WHERE id = ?
                """, (signals_count, report_path, report_type, summary,
                      duration_seconds, run_id))
                conn.commit()
                logger.info(f"Agent run completed: run_id={run_id}, signals={signals_count}")
            finally:
                conn.close()

    def fail_agent_run(self, run_id: int, error_message: str):
        """Mark an agent run as failed."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE mq_agent_runs SET
                        status = 'failed',
                        error_message = ?
                    WHERE id = ?
                """, (error_message, run_id))
                conn.commit()
                logger.error(f"Agent run failed: run_id={run_id}, error={error_message}")
            finally:
                conn.close()

    def get_recent_runs(self, limit: int = 20, agent_type: str = None) -> List[Dict]:
        """Get recent agent runs, optionally filtered by type."""
        conn = self._get_conn()
        try:
            if agent_type:
                rows = conn.execute(
                    "SELECT * FROM mq_agent_runs WHERE agent_type = ? ORDER BY run_date DESC LIMIT ?",
                    (agent_type, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM mq_agent_runs ORDER BY run_date DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_run(self, run_id: int) -> Optional[Dict]:
        """Get a specific agent run."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM mq_agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_run_report_path(self, run_id: int, report_path: str):
        """Update the report_path for a completed run."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE mq_agent_runs SET report_path = ? WHERE id = ?",
                    (report_path, run_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get_latest_run(self, agent_type: str) -> Optional[Dict]:
        """Get the most recent run for an agent type."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM mq_agent_runs WHERE agent_type = ? ORDER BY id DESC LIMIT 1",
                (agent_type,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # =========================================================================
    # Signals
    # =========================================================================

    def add_signal(
        self,
        run_id: int,
        signal_type: str,
        symbol: str,
        priority: str = 'MEDIUM',
        details: str = None
    ) -> int:
        """Add a single signal. Returns signal_id."""
        with db_lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("""
                    INSERT INTO mq_signals
                        (agent_run_id, signal_type, symbol, priority, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (run_id, signal_type, symbol, priority, details))
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def add_signals_batch(self, run_id: int, signals: List[Dict]):
        """Add multiple signals at once."""
        if not signals:
            return
        with db_lock:
            conn = self._get_conn()
            try:
                conn.executemany("""
                    INSERT INTO mq_signals
                        (agent_run_id, signal_type, symbol, priority, details)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    (run_id, s['signal_type'], s.get('symbol', ''),
                     s.get('priority', 'MEDIUM'),
                     s.get('details', '') if isinstance(s.get('details'), str)
                     else json.dumps(s.get('details', {})))
                    for s in signals
                ])
                conn.commit()
            finally:
                conn.close()

    def get_active_signals(self, limit: int = 50) -> List[Dict]:
        """Get pending signals ordered by priority then date."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT s.*, r.agent_type
                FROM mq_signals s
                LEFT JOIN mq_agent_runs r ON s.agent_run_id = r.id
                WHERE s.status = 'pending'
                ORDER BY
                    CASE s.priority
                        WHEN 'HIGH' THEN 1
                        WHEN 'MEDIUM' THEN 2
                        WHEN 'LOW' THEN 3
                    END,
                    s.created_date DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def dismiss_signal(self, signal_id: int):
        """Mark a signal as dismissed."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE mq_signals SET
                        status = 'dismissed',
                        resolved_date = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (signal_id,))
                conn.commit()
            finally:
                conn.close()

    def execute_signal(self, signal_id: int):
        """Mark a signal as executed."""
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE mq_signals SET
                        status = 'executed',
                        resolved_date = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (signal_id,))
                conn.commit()
            finally:
                conn.close()

    def expire_old_signals(self, days: int = 7):
        """Expire pending signals older than N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE mq_signals SET
                        status = 'expired',
                        resolved_date = CURRENT_TIMESTAMP
                    WHERE status = 'pending' AND created_date < ?
                """, (cutoff,))
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Reports (convenience queries)
    # =========================================================================

    def get_recent_reports(self, limit: int = 10) -> List[Dict]:
        """Get recent agent runs that have reports."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT id, agent_type, run_date, report_path, report_type, summary
                FROM mq_agent_runs
                WHERE report_path IS NOT NULL AND status = 'completed'
                ORDER BY run_date DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# =============================================================================
# Singleton
# =============================================================================

_instance = None
_instance_lock = threading.Lock()


def get_agent_db() -> MQAgentDB:
    """Get or create the singleton MQAgentDB instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MQAgentDB()
    return _instance
