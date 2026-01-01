"""
Backtest Database Manager
=========================

Handles storage and retrieval of backtest runs, trade logs, and equity curves.
Uses SQLite for lightweight, portable storage.

Tables:
- backtest_runs: Metadata and summary metrics for each backtest
- trade_log: Individual trade records with entry/exit details
- daily_equity: Daily equity curve for charting
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import threading
import logging

logger = logging.getLogger(__name__)


def _convert_date(value: Any) -> Optional[str]:
    """Convert date/timestamp values to string for SQLite"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d')
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d')
    return str(value)

# Database configuration
DB_DIR = Path(__file__).parent.parent / 'backtest_data'
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / 'backtest_results.db'

# Thread lock for database operations
db_lock = threading.Lock()


class BacktestDatabase:
    """
    Manager for backtest results storage and retrieval
    """

    def __init__(self):
        self.db_path = DB_PATH
        self._init_database()

    def _init_database(self):
        """Initialize the backtest results database schema"""
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Backtest run metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    name TEXT,
                    config_json TEXT NOT NULL,
                    -- Configuration summary
                    symbols TEXT,
                    start_date DATE,
                    end_date DATE,
                    strike_method TEXT,
                    exit_strategy TEXT,
                    -- Summary metrics
                    total_return REAL,
                    premium_yield REAL,
                    win_rate REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    assignment_rate REAL,
                    vs_buy_hold REAL,
                    total_trades INTEGER,
                    profitable_trades INTEGER,
                    losing_trades INTEGER,
                    -- Status
                    status TEXT DEFAULT 'running',  -- running, completed, failed
                    error_message TEXT
                )
            """)

            # Individual trade records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    lot_size INTEGER NOT NULL,
                    -- Entry details
                    entry_date DATE NOT NULL,
                    stock_entry_price REAL NOT NULL,
                    strike_price REAL NOT NULL,
                    premium_received REAL NOT NULL,
                    delta_at_entry REAL,
                    theta_at_entry REAL,
                    iv_at_entry REAL,
                    dte_at_entry INTEGER,
                    expiry_date DATE,
                    -- Exit details
                    exit_date DATE,
                    stock_exit_price REAL,
                    option_exit_price REAL,
                    exit_reason TEXT,  -- EXPIRY, ASSIGNED, PROFIT_TARGET, STOP_LOSS
                    -- P&L
                    stock_pnl REAL,
                    option_pnl REAL,
                    total_pnl REAL,
                    return_pct REAL,
                    -- Strategy metadata
                    strike_method TEXT,
                    exit_strategy TEXT,
                    FOREIGN KEY (backtest_id) REFERENCES backtest_runs(id)
                )
            """)

            # Daily equity curve for charts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_return REAL,
                    cumulative_return REAL,
                    drawdown REAL,
                    rolling_sharpe REAL,
                    positions_count INTEGER,
                    premium_collected REAL,
                    FOREIGN KEY (backtest_id) REFERENCES backtest_runs(id),
                    UNIQUE(backtest_id, date)
                )
            """)

            # Indexes for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_backtest
                ON trade_log(backtest_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_symbol
                ON trade_log(symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_equity_backtest
                ON daily_equity(backtest_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_equity_date
                ON daily_equity(backtest_id, date)
            """)

            conn.commit()
            conn.close()

        logger.info(f"Backtest database initialized at: {self.db_path}")

    # =========================================================================
    # Backtest Run Management
    # =========================================================================

    def create_backtest_run(
        self,
        name: str,
        config: Dict,
        symbols: List[str],
        start_date: str,
        end_date: str,
        strike_method: str,
        exit_strategy: str
    ) -> int:
        """
        Create a new backtest run entry

        Returns:
            backtest_id: The ID of the newly created run
        """
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO backtest_runs
                (name, config_json, symbols, start_date, end_date,
                 strike_method, exit_strategy, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'running')
            """, (
                name,
                json.dumps(config),
                ','.join(symbols),
                start_date,
                end_date,
                strike_method,
                exit_strategy
            ))

            backtest_id = cursor.lastrowid
            conn.commit()
            conn.close()

        logger.info(f"Created backtest run #{backtest_id}: {name}")
        return backtest_id

    def update_backtest_metrics(
        self,
        backtest_id: int,
        metrics: Dict
    ):
        """Update a backtest run with final metrics"""
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE backtest_runs
                SET total_return = ?,
                    premium_yield = ?,
                    win_rate = ?,
                    max_drawdown = ?,
                    sharpe_ratio = ?,
                    assignment_rate = ?,
                    vs_buy_hold = ?,
                    total_trades = ?,
                    profitable_trades = ?,
                    losing_trades = ?,
                    status = 'completed'
                WHERE id = ?
            """, (
                metrics.get('total_return'),
                metrics.get('premium_yield'),
                metrics.get('win_rate'),
                metrics.get('max_drawdown'),
                metrics.get('sharpe_ratio'),
                metrics.get('assignment_rate'),
                metrics.get('vs_buy_hold'),
                metrics.get('total_trades'),
                metrics.get('profitable_trades'),
                metrics.get('losing_trades'),
                backtest_id
            ))

            conn.commit()
            conn.close()

        logger.info(f"Updated metrics for backtest #{backtest_id}")

    def mark_backtest_failed(self, backtest_id: int, error_message: str):
        """Mark a backtest as failed with error message"""
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE backtest_runs
                SET status = 'failed', error_message = ?
                WHERE id = ?
            """, (error_message, backtest_id))

            conn.commit()
            conn.close()

    def get_backtest_run(self, backtest_id: int) -> Dict:
        """Get a single backtest run by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM backtest_runs WHERE id = ?
        """, (backtest_id,))

        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        conn.close()

        if row:
            result = dict(zip(columns, row))
            result['config'] = json.loads(result.get('config_json', '{}'))
            return result
        return None

    def get_recent_backtests(self, limit: int = 10) -> List[Dict]:
        """Get list of recent backtest runs"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT id, created_at, name, symbols, start_date, end_date,
                   strike_method, exit_strategy, total_return, win_rate,
                   max_drawdown, sharpe_ratio, total_trades, status
            FROM backtest_runs
            ORDER BY created_at DESC
            LIMIT ?
        """, conn, params=(limit,))

        conn.close()
        return df.to_dict('records')

    # =========================================================================
    # Trade Log Management
    # =========================================================================

    def add_trade(
        self,
        backtest_id: int,
        trade: Dict
    ) -> int:
        """
        Add a trade record to the log

        Args:
            backtest_id: Parent backtest run ID
            trade: Dict with trade details

        Returns:
            trade_id: The ID of the newly created trade
        """
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO trade_log
                (backtest_id, symbol, lot_size, entry_date, stock_entry_price,
                 strike_price, premium_received, delta_at_entry, theta_at_entry,
                 iv_at_entry, dte_at_entry, expiry_date, exit_date, stock_exit_price,
                 option_exit_price, exit_reason, stock_pnl, option_pnl, total_pnl,
                 return_pct, strike_method, exit_strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                backtest_id,
                trade.get('symbol'),
                trade.get('lot_size', 1),
                trade.get('entry_date'),
                trade.get('stock_entry_price'),
                trade.get('strike_price'),
                trade.get('premium_received'),
                trade.get('delta_at_entry'),
                trade.get('theta_at_entry'),
                trade.get('iv_at_entry'),
                trade.get('dte_at_entry'),
                trade.get('expiry_date'),
                trade.get('exit_date'),
                trade.get('stock_exit_price'),
                trade.get('option_exit_price'),
                trade.get('exit_reason'),
                trade.get('stock_pnl'),
                trade.get('option_pnl'),
                trade.get('total_pnl'),
                trade.get('return_pct'),
                trade.get('strike_method'),
                trade.get('exit_strategy')
            ))

            trade_id = cursor.lastrowid
            conn.commit()
            conn.close()

        return trade_id

    def add_trades_batch(self, backtest_id: int, trades: List[Dict]):
        """Add multiple trades in a single transaction"""
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for trade in trades:
                cursor.execute("""
                    INSERT INTO trade_log
                    (backtest_id, symbol, lot_size, entry_date, stock_entry_price,
                     strike_price, premium_received, delta_at_entry, theta_at_entry,
                     iv_at_entry, dte_at_entry, expiry_date, exit_date, stock_exit_price,
                     option_exit_price, exit_reason, stock_pnl, option_pnl, total_pnl,
                     return_pct, strike_method, exit_strategy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_id,
                    trade.get('symbol'),
                    trade.get('lot_size', 1),
                    _convert_date(trade.get('entry_date')),
                    trade.get('stock_entry_price'),
                    trade.get('strike_price'),
                    trade.get('premium_received'),
                    trade.get('delta_at_entry'),
                    trade.get('theta_at_entry'),
                    trade.get('iv_at_entry'),
                    trade.get('dte_at_entry'),
                    _convert_date(trade.get('expiry_date')),
                    _convert_date(trade.get('exit_date')),
                    trade.get('stock_exit_price'),
                    trade.get('option_exit_price'),
                    trade.get('exit_reason'),
                    trade.get('stock_pnl'),
                    trade.get('option_pnl'),
                    trade.get('total_pnl'),
                    trade.get('return_pct'),
                    trade.get('strike_method'),
                    trade.get('exit_strategy')
                ))

            conn.commit()
            conn.close()

        logger.info(f"Added {len(trades)} trades to backtest #{backtest_id}")

    def get_trades(self, backtest_id: int) -> pd.DataFrame:
        """Get all trades for a backtest"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT * FROM trade_log
            WHERE backtest_id = ?
            ORDER BY entry_date
        """, conn, params=(backtest_id,))

        conn.close()
        return df

    def get_trades_by_symbol(self, backtest_id: int, symbol: str) -> pd.DataFrame:
        """Get trades for a specific symbol"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT * FROM trade_log
            WHERE backtest_id = ? AND symbol = ?
            ORDER BY entry_date
        """, conn, params=(backtest_id, symbol))

        conn.close()
        return df

    # =========================================================================
    # Equity Curve Management
    # =========================================================================

    def add_equity_points(self, backtest_id: int, equity_data: List[Dict]):
        """Add daily equity curve data points"""
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for point in equity_data:
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_equity
                    (backtest_id, date, portfolio_value, daily_return,
                     cumulative_return, drawdown, rolling_sharpe,
                     positions_count, premium_collected)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_id,
                    _convert_date(point.get('date')),
                    point.get('portfolio_value'),
                    point.get('daily_return'),
                    point.get('cumulative_return'),
                    point.get('drawdown'),
                    point.get('rolling_sharpe'),
                    point.get('positions_count', 0),
                    point.get('premium_collected', 0)
                ))

            conn.commit()
            conn.close()

        logger.info(f"Added {len(equity_data)} equity points to backtest #{backtest_id}")

    def get_equity_curve(self, backtest_id: int) -> pd.DataFrame:
        """Get the equity curve for a backtest"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT date, portfolio_value, daily_return, cumulative_return,
                   drawdown, rolling_sharpe, positions_count, premium_collected
            FROM daily_equity
            WHERE backtest_id = ?
            ORDER BY date
        """, conn, params=(backtest_id,))

        conn.close()

        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        return df

    # =========================================================================
    # Strategy Comparison
    # =========================================================================

    def get_strategy_comparison(
        self,
        backtest_ids: List[int]
    ) -> pd.DataFrame:
        """Get comparison metrics for multiple backtests"""
        if not backtest_ids:
            return pd.DataFrame()

        conn = sqlite3.connect(self.db_path)

        placeholders = ','.join(['?' for _ in backtest_ids])
        df = pd.read_sql_query(f"""
            SELECT id, name, strike_method, exit_strategy,
                   total_return, premium_yield, win_rate, max_drawdown,
                   sharpe_ratio, assignment_rate, vs_buy_hold, total_trades
            FROM backtest_runs
            WHERE id IN ({placeholders})
            ORDER BY total_return DESC
        """, conn, params=backtest_ids)

        conn.close()
        return df

    # =========================================================================
    # Cleanup
    # =========================================================================

    def delete_backtest(self, backtest_id: int):
        """Delete a backtest and all associated data"""
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete in order (children first)
            cursor.execute("DELETE FROM daily_equity WHERE backtest_id = ?", (backtest_id,))
            cursor.execute("DELETE FROM trade_log WHERE backtest_id = ?", (backtest_id,))
            cursor.execute("DELETE FROM backtest_runs WHERE id = ?", (backtest_id,))

            conn.commit()
            conn.close()

        logger.info(f"Deleted backtest #{backtest_id} and all associated data")


# =============================================================================
# Singleton Instance
# =============================================================================

_db_instance = None
_db_lock = threading.Lock()


def get_backtest_db() -> BacktestDatabase:
    """Get singleton instance of BacktestDatabase"""
    global _db_instance

    with _db_lock:
        if _db_instance is None:
            _db_instance = BacktestDatabase()

    return _db_instance
