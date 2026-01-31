"""
MQ Rebalance Agent
===================

Semi-annual portfolio rebalance (January 1 + July 1).

Steps:
1. Check ATH drawdown for all positions
2. Exit positions with >20% drawdown
3. Run fresh screening for replacements
4. Generate entry signals for top-ranked new stocks
"""

import json
import logging
import time
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

from .mq_agent_db import get_agent_db, MQAgentDB
from .mq_agent_reports import RebalanceReport

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'

# Config
from config import MQ_DEFAULTS


class RebalanceAgent:
    """Semi-annual portfolio rebalance agent."""

    def __init__(self, db: MQAgentDB = None):
        self.db = db or get_agent_db()

    def run(self, portfolio_symbols: List[str] = None) -> RebalanceReport:
        """Run the rebalance process."""
        start_time = time.time()
        run_id = self.db.start_agent_run('rebalance')

        try:
            symbols = portfolio_symbols or self._get_portfolio_symbols()

            if not symbols:
                report = RebalanceReport(run_date=datetime.now())
                self.db.complete_agent_run(run_id, 0, summary="No positions to rebalance")
                return report

            # Step 1-2: Check ATH drawdown, identify exits
            exits = []
            for symbol in symbols:
                drawdown = self._calculate_ath_drawdown(symbol)
                if drawdown is not None and drawdown >= MQ_DEFAULTS['rebalance_ath_drawdown']:
                    exits.append({
                        'symbol': symbol,
                        'drawdown_pct': round(drawdown * 100, 1),
                        'reason': f'ATH drawdown {drawdown:.1%} > {MQ_DEFAULTS["rebalance_ath_drawdown"]:.0%} threshold',
                    })

            logger.info(f"Rebalance: {len(exits)} exits from {len(symbols)} positions")

            # Step 3: Run fresh screening for replacements
            screening = None
            entries = []

            if exits:
                from .mq_screening_agent import ScreeningAgent
                screener = ScreeningAgent(self.db)
                screening = screener.run()

                # Step 4: Select replacements not already in portfolio
                kept_symbols = set(symbols) - {e['symbol'] for e in exits}
                for candidate in screening.top_ranked:
                    if len(entries) >= len(exits):
                        break
                    if candidate['symbol'] not in kept_symbols:
                        entries.append(candidate)

                # Generate signals
                for exit_info in exits:
                    self.db.add_signal(
                        run_id, 'EXIT', exit_info['symbol'], 'HIGH',
                        json.dumps(exit_info)
                    )
                for entry_info in entries:
                    self.db.add_signal(
                        run_id, 'ENTRY', entry_info['symbol'], 'HIGH',
                        json.dumps(entry_info)
                    )

            # Build report
            report = RebalanceReport(
                run_date=datetime.now(),
                exits=exits,
                entries=entries,
                portfolio_before={'symbols': symbols, 'count': len(symbols)},
                portfolio_after={
                    'symbols': list((set(symbols) - {e['symbol'] for e in exits}) | {e['symbol'] for e in entries}),
                    'count': len(symbols),
                },
                screening_used=screening,
            )

            duration = time.time() - start_time
            self.db.complete_agent_run(
                run_id,
                signals_count=len(exits) + len(entries),
                report_type='rebalance',
                summary=f"Rebalance: {len(exits)} exits, {len(entries)} entries",
                duration_seconds=round(duration, 1),
            )

            logger.info(f"Rebalance complete: {len(exits)} exits, {len(entries)} entries in {duration:.1f}s")
            return report

        except Exception as e:
            self.db.fail_agent_run(run_id, str(e))
            logger.error(f"Rebalance failed: {e}")
            raise

    def _get_portfolio_symbols(self) -> List[str]:
        """Get current portfolio symbols."""
        # Try from active entry signals or latest screening
        signals = self.db.get_active_signals(limit=50)
        return list(set(
            s['symbol'] for s in signals
            if s.get('symbol') and s.get('signal_type') in ('ENTRY', 'WATCH')
        ))[:30]

    def _calculate_ath_drawdown(self, symbol: str) -> Optional[float]:
        """Calculate drawdown from 52-week high."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cutoff = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            rows = conn.execute("""
                SELECT close FROM market_data_unified
                WHERE symbol = ? AND date >= ?
                ORDER BY date
            """, (symbol, cutoff)).fetchall()
            conn.close()

            if not rows or len(rows) < 5:
                return None

            closes = [row[0] for row in rows]
            high_52w = max(closes)
            current = closes[-1]

            if high_52w <= 0:
                return None

            return (high_52w - current) / high_52w

        except Exception:
            return None
