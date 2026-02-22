"""
Tactical Capital Pool Manager
================================

Manages capital allocation across two pools:
- MQ Core (60%): Momentum+Quality portfolio, always deployed
- Tactical Pool (40%): Shared between KC6 mean reversion, IPO Scalper,
  and IPO Swing — idle cash parked in liquid debt fund (~7% p.a.)

Priority rules when capital is tight:
1. Buffer (15% of pool) always stays in debt fund
2. IPO Scalper gets priority (84% WR, +5.5% expectancy)
3. IPO Swing next (64.5% WR, +16.17% expectancy per trade)
4. KC6 fills remaining slots
"""

import sqlite3
import json
import logging
import threading
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict

from config import DATA_DIR

logger = logging.getLogger(__name__)

DB_DIR = DATA_DIR
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / 'tactical_pool.db'

_lock = threading.Lock()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TacticalConfig:
    """Capital allocation configuration."""
    total_capital: float = 10_000_000       # Rs. 1 Crore

    # Pool splits (60/40)
    mq_pct: float = 0.60                    # 60% MQ Core
    tactical_pct: float = 0.40              # 40% Tactical Pool (KC6 + IPO Scalper + IPO Swing)

    # Tactical pool rules
    debt_fund_return_pct: float = 7.0       # Annual return on idle cash
    buffer_pct: float = 0.0                # No buffer — all tactical capital available for trades

    # KC6 limits
    kc6_max_positions: int = 7
    kc6_position_size: float = 400_000      # Rs. 4L per KC6 trade
    kc6_max_daily_entries: int = 3

    # IPO Scalper limits (System 1: T6/SL20, 84.3% WR)
    ipo_max_positions: int = 2
    ipo_position_size: float = 650_000      # Rs. 6.5L per IPO Scalper trade

    # IPO Swing limits (BRK7+LG, T30/SL15, 64.5% WR, 16.17% expectancy)
    ipo_swing_max_positions: int = 2
    ipo_swing_position_size: float = 500_000  # Rs. 5L per IPO Swing trade

    @property
    def mq_capital(self) -> float:
        return self.total_capital * self.mq_pct

    @property
    def tactical_capital(self) -> float:
        return self.total_capital * self.tactical_pct

    @property
    def tactical_buffer(self) -> float:
        return self.tactical_capital * self.buffer_pct


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PoolPosition:
    """An active position in the tactical pool."""
    id: int
    strategy: str               # 'kc6', 'ipo', or 'ipo_swing'
    symbol: str
    entry_date: str
    entry_price: float
    qty: int
    capital_deployed: float     # Actual Rs. drawn from pool
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = 'ACTIVE'      # ACTIVE, CLOSED
    exit_date: str = ''
    exit_price: float = 0.0
    realized_pnl: float = 0.0
    exit_reason: str = ''


@dataclass
class PoolSnapshot:
    """Point-in-time snapshot of the tactical pool."""
    timestamp: str
    debt_fund_balance: float        # Cash in liquid debt fund
    kc6_deployed: float             # Capital in active KC6 positions
    kc6_positions: int              # Number of active KC6 positions
    ipo_deployed: float             # Capital in active IPO positions
    ipo_positions: int              # Number of active IPO positions
    ipo_swing_deployed: float       # Capital in IPO Swing positions
    ipo_swing_positions: int        # Number of active IPO Swing positions
    total_pool_value: float         # debt + kc6 + ipo + darvas (at market)
    debt_interest_earned: float     # Cumulative interest from debt fund
    kc6_realized_pnl: float         # Cumulative KC6 P&L
    ipo_realized_pnl: float         # Cumulative IPO P&L
    ipo_swing_realized_pnl: float   # Cumulative IPO Swing P&L


# =============================================================================
# Database Manager
# =============================================================================

class TacticalPoolDB:
    """SQLite persistence for tactical pool state."""

    def __init__(self, config: TacticalConfig = None):
        self.config = config or TacticalConfig()
        self.db_path = DB_PATH
        self._init_database()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        with _lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS pool_config (
                        id INTEGER PRIMARY KEY,
                        total_capital REAL NOT NULL,
                        mq_pct REAL NOT NULL,
                        tactical_pct REAL NOT NULL,
                        debt_fund_return_pct REAL DEFAULT 7.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS pool_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy VARCHAR(10) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        entry_date DATE NOT NULL,
                        entry_price REAL NOT NULL,
                        qty INTEGER NOT NULL,
                        capital_deployed REAL NOT NULL,
                        current_price REAL DEFAULT 0,
                        unrealized_pnl REAL DEFAULT 0,
                        status VARCHAR(10) DEFAULT 'ACTIVE',
                        exit_date DATE,
                        exit_price REAL DEFAULT 0,
                        realized_pnl REAL DEFAULT 0,
                        exit_reason VARCHAR(30),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS pool_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_date DATE NOT NULL,
                        debt_fund_balance REAL NOT NULL,
                        kc6_deployed REAL DEFAULT 0,
                        kc6_positions INTEGER DEFAULT 0,
                        ipo_deployed REAL DEFAULT 0,
                        ipo_positions INTEGER DEFAULT 0,
                        ipo_swing_deployed REAL DEFAULT 0,
                        ipo_swing_positions INTEGER DEFAULT 0,
                        total_pool_value REAL NOT NULL,
                        debt_interest_earned REAL DEFAULT 0,
                        kc6_realized_pnl REAL DEFAULT 0,
                        ipo_realized_pnl REAL DEFAULT 0,
                        ipo_swing_realized_pnl REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS pool_activity_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        action VARCHAR(30) NOT NULL,
                        strategy VARCHAR(10),
                        symbol VARCHAR(20),
                        amount REAL,
                        details TEXT,
                        debt_balance_after REAL
                    );

                    CREATE INDEX IF NOT EXISTS idx_pool_pos_status
                        ON pool_positions(status);
                    CREATE INDEX IF NOT EXISTS idx_pool_pos_strategy
                        ON pool_positions(strategy);
                    CREATE INDEX IF NOT EXISTS idx_pool_snap_date
                        ON pool_snapshots(snapshot_date);
                """)
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # State Queries
    # =========================================================================

    def get_active_positions(self, strategy: str = None) -> List[dict]:
        """Get all active positions, optionally filtered by strategy."""
        with _lock:
            conn = self._get_conn()
            try:
                if strategy:
                    rows = conn.execute(
                        "SELECT * FROM pool_positions WHERE status='ACTIVE' AND strategy=? ORDER BY entry_date",
                        (strategy,)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM pool_positions WHERE status='ACTIVE' ORDER BY strategy, entry_date"
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_closed_positions(self, strategy: str = None, limit: int = 50) -> List[dict]:
        """Get closed positions with optional strategy filter."""
        with _lock:
            conn = self._get_conn()
            try:
                if strategy:
                    rows = conn.execute(
                        "SELECT * FROM pool_positions WHERE status='CLOSED' AND strategy=? ORDER BY exit_date DESC LIMIT ?",
                        (strategy, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM pool_positions WHERE status='CLOSED' ORDER BY exit_date DESC LIMIT ?",
                        (limit,)
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_pool_state(self) -> dict:
        """Get current pool state: deployed capital, available capital, positions."""
        active = self.get_active_positions()
        kc6_positions = [p for p in active if p['strategy'] == 'kc6']
        ipo_positions = [p for p in active if p['strategy'] == 'ipo']
        ipo_swing_positions = [p for p in active if p['strategy'] == 'ipo_swing']

        kc6_deployed = sum(p['capital_deployed'] for p in kc6_positions)
        ipo_deployed = sum(p['capital_deployed'] for p in ipo_positions)
        ipo_swing_deployed = sum(p['capital_deployed'] for p in ipo_swing_positions)
        total_deployed = kc6_deployed + ipo_deployed + ipo_swing_deployed

        tactical_total = self.config.tactical_capital
        debt_balance = tactical_total - total_deployed

        # Cumulative realized P&L
        kc6_pnl = self._sum_realized_pnl('kc6')
        ipo_pnl = self._sum_realized_pnl('ipo')
        ipo_swing_pnl = self._sum_realized_pnl('ipo_swing')

        # Adjust debt balance for realized P&L (profits/losses return to pool)
        debt_balance += kc6_pnl + ipo_pnl + ipo_swing_pnl

        buffer = self.config.tactical_buffer
        available = max(0, debt_balance - buffer)

        return {
            'tactical_total': tactical_total,
            'debt_fund_balance': debt_balance,
            'kc6_deployed': kc6_deployed,
            'kc6_positions': len(kc6_positions),
            'kc6_active': kc6_positions,
            'ipo_deployed': ipo_deployed,
            'ipo_positions': len(ipo_positions),
            'ipo_active': ipo_positions,
            'ipo_swing_deployed': ipo_swing_deployed,
            'ipo_swing_positions': len(ipo_swing_positions),
            'ipo_swing_active': ipo_swing_positions,
            'total_deployed': total_deployed,
            'available_capital': available,
            'buffer': buffer,
            'utilization_pct': (total_deployed / tactical_total * 100) if tactical_total > 0 else 0,
            'kc6_realized_pnl': kc6_pnl,
            'ipo_realized_pnl': ipo_pnl,
            'ipo_swing_realized_pnl': ipo_swing_pnl,
            'total_realized_pnl': kc6_pnl + ipo_pnl + ipo_swing_pnl,
        }

    def _sum_realized_pnl(self, strategy: str) -> float:
        with _lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(realized_pnl), 0) as total FROM pool_positions WHERE status='CLOSED' AND strategy=?",
                    (strategy,)
                ).fetchone()
                return row['total']
            finally:
                conn.close()

    # =========================================================================
    # Capital Allocation
    # =========================================================================

    def can_allocate(self, strategy: str, amount: float = None) -> dict:
        """Check if capital can be allocated for a new position.

        Returns dict with 'allowed', 'reason', 'max_amount'.
        """
        state = self.get_pool_state()
        cfg = self.config

        if amount is None:
            if strategy == 'kc6':
                amount = cfg.kc6_position_size
            elif strategy == 'ipo':
                amount = cfg.ipo_position_size
            elif strategy == 'ipo_swing':
                amount = cfg.ipo_swing_position_size
            else:
                amount = cfg.kc6_position_size  # fallback

        # Check position limits
        if strategy == 'kc6' and state['kc6_positions'] >= cfg.kc6_max_positions:
            return {'allowed': False, 'reason': f'KC6 at max positions ({cfg.kc6_max_positions})', 'max_amount': 0}

        if strategy == 'ipo' and state['ipo_positions'] >= cfg.ipo_max_positions:
            return {'allowed': False, 'reason': f'IPO Scalper at max positions ({cfg.ipo_max_positions})', 'max_amount': 0}

        if strategy == 'ipo_swing' and state['ipo_swing_positions'] >= cfg.ipo_swing_max_positions:
            return {'allowed': False, 'reason': f'IPO Swing at max positions ({cfg.ipo_swing_max_positions})', 'max_amount': 0}

        # Check available capital (after buffer)
        available = state['available_capital']
        if available < amount:
            return {'allowed': False, 'reason': f'Insufficient capital: Rs.{available:,.0f} available, Rs.{amount:,.0f} needed', 'max_amount': available}

        return {'allowed': True, 'reason': 'OK', 'max_amount': available}

    def allocate(self, strategy: str, symbol: str, entry_price: float,
                 qty: int, capital: float = None) -> Optional[int]:
        """Allocate capital from the pool for a new position.

        Returns position ID if successful, None if allocation denied.
        """
        cfg = self.config
        if capital is None:
            if strategy == 'kc6':
                capital = cfg.kc6_position_size
            elif strategy == 'ipo':
                capital = cfg.ipo_position_size
            else:
                capital = cfg.ipo_swing_position_size

        check = self.can_allocate(strategy, capital)
        if not check['allowed']:
            logger.warning(f"Allocation denied for {strategy}/{symbol}: {check['reason']}")
            return None

        entry_date = date.today().isoformat()

        with _lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("""
                    INSERT INTO pool_positions
                    (strategy, symbol, entry_date, entry_price, qty, capital_deployed, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE')
                """, (strategy, symbol, entry_date, entry_price, qty, capital))
                position_id = cursor.lastrowid

                # Log the allocation
                conn.execute("""
                    INSERT INTO pool_activity_log
                    (action, strategy, symbol, amount, details, debt_balance_after)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ('ALLOCATE', strategy, symbol, capital,
                      json.dumps({'entry_price': entry_price, 'qty': qty}),
                      self.config.tactical_capital - capital))

                conn.commit()
                logger.info(f"Allocated Rs.{capital:,.0f} for {strategy}/{symbol} (pos #{position_id})")
                return position_id
            finally:
                conn.close()

    def release(self, position_id: int, exit_price: float, exit_reason: str) -> float:
        """Release capital back to pool on position exit.

        Returns realized P&L.
        """
        with _lock:
            conn = self._get_conn()
            try:
                pos = conn.execute(
                    "SELECT * FROM pool_positions WHERE id=? AND status='ACTIVE'",
                    (position_id,)
                ).fetchone()
                if not pos:
                    logger.warning(f"Position {position_id} not found or already closed")
                    return 0.0

                pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                realized_pnl = pos['capital_deployed'] * pnl_pct
                exit_date = date.today().isoformat()

                conn.execute("""
                    UPDATE pool_positions SET
                        status='CLOSED', exit_date=?, exit_price=?,
                        realized_pnl=?, exit_reason=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (exit_date, exit_price, realized_pnl, exit_reason, position_id))

                # Log the release
                conn.execute("""
                    INSERT INTO pool_activity_log
                    (action, strategy, symbol, amount, details)
                    VALUES (?, ?, ?, ?, ?)
                """, ('RELEASE', pos['strategy'], pos['symbol'],
                      pos['capital_deployed'] + realized_pnl,
                      json.dumps({'exit_price': exit_price, 'pnl': round(realized_pnl, 2),
                                  'pnl_pct': round(pnl_pct * 100, 2), 'reason': exit_reason})))

                conn.commit()
                logger.info(f"Released {pos['strategy']}/{pos['symbol']} pos #{position_id}: "
                            f"P&L Rs.{realized_pnl:,.0f} ({pnl_pct*100:+.1f}%)")
                return realized_pnl
            finally:
                conn.close()

    # =========================================================================
    # Snapshots & History
    # =========================================================================

    def save_snapshot(self) -> dict:
        """Save a daily snapshot of pool state."""
        state = self.get_pool_state()
        snap_date = date.today().isoformat()

        with _lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO pool_snapshots
                    (snapshot_date, debt_fund_balance, kc6_deployed, kc6_positions,
                     ipo_deployed, ipo_positions, ipo_swing_deployed, ipo_swing_positions,
                     total_pool_value, kc6_realized_pnl, ipo_realized_pnl, ipo_swing_realized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (snap_date, state['debt_fund_balance'],
                      state['kc6_deployed'], state['kc6_positions'],
                      state['ipo_deployed'], state['ipo_positions'],
                      state['ipo_swing_deployed'], state['ipo_swing_positions'],
                      state['tactical_total'] + state['total_realized_pnl'],
                      state['kc6_realized_pnl'], state['ipo_realized_pnl'],
                      state['ipo_swing_realized_pnl']))
                conn.commit()
            finally:
                conn.close()
        return state

    def get_snapshots(self, limit: int = 365) -> List[dict]:
        """Get pool snapshots for charting."""
        with _lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM pool_snapshots ORDER BY snapshot_date DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_activity_log(self, limit: int = 50) -> List[dict]:
        """Get recent activity log entries."""
        with _lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM pool_activity_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_strategy_stats(self, strategy: str) -> dict:
        """Get aggregate stats for a strategy."""
        with _lock:
            conn = self._get_conn()
            try:
                closed = conn.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
                           SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losers,
                           SUM(realized_pnl) as total_pnl,
                           AVG(realized_pnl) as avg_pnl,
                           AVG(JULIANDAY(exit_date) - JULIANDAY(entry_date)) as avg_hold_days
                    FROM pool_positions
                    WHERE strategy=? AND status='CLOSED'
                """, (strategy,)).fetchone()

                total = closed['total'] or 0
                winners = closed['winners'] or 0
                win_rate = (winners / total * 100) if total > 0 else 0

                active_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM pool_positions WHERE strategy=? AND status='ACTIVE'",
                    (strategy,)
                ).fetchone()['cnt']

                return {
                    'strategy': strategy,
                    'active_positions': active_count,
                    'total_closed': total,
                    'winners': winners,
                    'losers': closed['losers'] or 0,
                    'win_rate': round(win_rate, 1),
                    'total_pnl': round(closed['total_pnl'] or 0, 2),
                    'avg_pnl': round(closed['avg_pnl'] or 0, 2),
                    'avg_hold_days': round(closed['avg_hold_days'] or 0, 1),
                }
            finally:
                conn.close()

    # =========================================================================
    # Full System State (for dashboard API)
    # =========================================================================

    def get_dashboard_state(self) -> dict:
        """Get complete state for the tactical dashboard."""
        cfg = self.config
        pool_state = self.get_pool_state()
        kc6_stats = self.get_strategy_stats('kc6')
        ipo_stats = self.get_strategy_stats('ipo')
        ipo_swing_stats = self.get_strategy_stats('ipo_swing')
        recent_activity = self.get_activity_log(limit=20)
        closed_trades = self.get_closed_positions(limit=30)
        snapshots = self.get_snapshots(limit=365)

        return {
            'config': {
                'total_capital': cfg.total_capital,
                'mq_capital': cfg.mq_capital,
                'tactical_capital': cfg.tactical_capital,
                'mq_pct': cfg.mq_pct * 100,
                'tactical_pct': cfg.tactical_pct * 100,
                'debt_fund_return': cfg.debt_fund_return_pct,
                'kc6_position_size': cfg.kc6_position_size,
                'ipo_position_size': cfg.ipo_position_size,
                'ipo_swing_position_size': cfg.ipo_swing_position_size,
                'kc6_max_positions': cfg.kc6_max_positions,
                'ipo_max_positions': cfg.ipo_max_positions,
                'ipo_swing_max_positions': cfg.ipo_swing_max_positions,
            },
            'pool': pool_state,
            'kc6_stats': kc6_stats,
            'ipo_stats': ipo_stats,
            'ipo_swing_stats': ipo_swing_stats,
            'activity_log': recent_activity,
            'closed_trades': closed_trades,
            'snapshots': snapshots,

            # Strategy reference data (from backtest research)
            'strategy_reference': {
                'kc6': {
                    'name': 'KC6 Mean Reversion',
                    'backtest_trades': 2482,
                    'backtest_years': 20,
                    'trades_per_year': 124,
                    'win_rate': 65.0,
                    'profit_factor': 1.70,
                    'avg_hold_days': 7,
                    'max_hold_days': 15,
                    'entry_rule': 'Close < KC(6, 1.3 ATR) Lower AND Close > SMA(200)',
                    'exit_rules': ['KC6 Mid target', 'SL 5%', 'TP 15%', 'MaxHold 15d'],
                    'crash_filter': 'Universe ATR Ratio >= 1.3x',
                },
                'ipo': {
                    'name': 'IPO Breakout (System 1)',
                    'backtest_trades': 51,
                    'backtest_years': 10,
                    'trades_per_year': 5.1,
                    'win_rate': 84.3,
                    'profit_factor': 2.73,
                    'avg_hold_days': 37,
                    'entry_rule': 'Listing Gain + Close > ATH(10) + BRK >= 3% + Vol >= 2.5x',
                    'exit_rules': ['Target +6%', 'SL -20%', 'Time exit 20d (hybrid)'],
                    'filters': 'Listing Gain required, Breakout >= 3% above initial ATH',
                },
                'ipo_swing': {
                    'name': 'IPO Swing (BRK7+LG)',
                    'backtest_trades': 31,
                    'backtest_years': 10,
                    'trades_per_year': 3.1,
                    'win_rate': 64.5,
                    'profit_factor': 3.42,
                    'expectancy_pct': 16.17,
                    'avg_hold_days': 51,
                    'median_return_pct': 31.8,
                    'entry_rule': 'Listing Gain + Breakout >= 7% above ATH',
                    'exit_rules': ['Target +30%', 'SL -15%'],
                    'filters': 'Listing Gain required, Breakout >= 7% (stricter than Scalper)',
                },
            },
        }
