"""
MQ Monitoring Agent
====================

Daily post-market monitoring of the 30-stock portfolio.
Runs weekdays at 4:30 PM IST.

Checks:
- Consolidation detection for each stock
- Breakout alerts with topup signals
- ATH drawdown tracking
- Position size and sector concentration warnings
"""

import json
import logging
import time
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

from .mq_agent_db import get_agent_db, MQAgentDB
from .mq_agent_reports import MonitoringSignal, MonitoringReport
from .consolidation_breakout import detect_consolidation, detect_breakout

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'

# Config
from config import MQ_DEFAULTS


class MonitoringAgent:
    """Daily portfolio monitoring agent."""

    def __init__(self, db: MQAgentDB = None):
        self.db = db or get_agent_db()

    def run(self, portfolio_symbols: List[str] = None) -> MonitoringReport:
        """Run daily monitoring on portfolio stocks."""
        start_time = time.time()
        run_id = self.db.start_agent_run('monitoring')

        try:
            symbols = portfolio_symbols or self._get_portfolio_symbols()
            if not symbols:
                logger.warning("No portfolio symbols to monitor")
                report = MonitoringReport(run_date=datetime.now())
                self.db.complete_agent_run(run_id, 0, summary="No positions to monitor")
                return report

            signals = []
            consolidation_count = 0
            breakout_count = 0

            for symbol in symbols:
                # 1. Consolidation detection
                try:
                    zone = detect_consolidation(symbol)
                    if zone.is_consolidating:
                        consolidation_count += 1
                        signals.append(MonitoringSignal(
                            signal_type='WATCH',
                            symbol=symbol,
                            priority='LOW',
                            details={
                                'days': zone.days_in_range,
                                'range_pct': round(zone.range_pct * 100, 1),
                                'range_high': zone.range_high,
                                'range_low': zone.range_low,
                            },
                            message=f'{symbol} consolidating Day {zone.days_in_range} ({zone.range_pct*100:.1f}% range)'
                        ))

                        # 2. Breakout detection (only for consolidating stocks)
                        breakout = detect_breakout(symbol, zone)
                        if breakout and breakout.is_breakout:
                            breakout_count += 1
                            signals.append(MonitoringSignal(
                                signal_type='TOPUP',
                                symbol=symbol,
                                priority='HIGH',
                                details={
                                    'volume_ratio': round(breakout.volume_ratio, 1),
                                    'breakout_price': breakout.breakout_price,
                                    'range_high': zone.range_high,
                                },
                                message=f'{symbol} BREAKOUT! Vol {breakout.volume_ratio:.1f}x avg'
                            ))
                except Exception as e:
                    logger.debug(f"Consolidation check failed for {symbol}: {e}")

                # 3. ATH drawdown tracking
                try:
                    drawdown = self._calculate_ath_drawdown(symbol)
                    if drawdown is not None and drawdown > 0.15:
                        is_exit = drawdown >= MQ_DEFAULTS['rebalance_ath_drawdown']
                        signals.append(MonitoringSignal(
                            signal_type='EXIT' if is_exit else 'WARNING',
                            symbol=symbol,
                            priority='HIGH' if is_exit else 'MEDIUM',
                            details={'drawdown_pct': round(drawdown * 100, 1)},
                            message=f'{symbol} down {drawdown:.1%} from ATH' + (' - EXIT CANDIDATE' if is_exit else '')
                        ))
                except Exception as e:
                    logger.debug(f"ATH drawdown check failed for {symbol}: {e}")

            # 4. Position size and sector warnings
            position_warnings = self._check_position_limits(symbols)
            sector_warnings = self._check_sector_limits(symbols)

            for w in position_warnings:
                signals.append(MonitoringSignal(
                    signal_type='WARNING', symbol=w['symbol'],
                    priority='MEDIUM', details=w,
                    message=f"{w['symbol']} at {w['weight_pct']:.1f}% of portfolio (limit: 10%)"
                ))

            for w in sector_warnings:
                signals.append(MonitoringSignal(
                    signal_type='WARNING', symbol=w['sector'],
                    priority='MEDIUM', details=w,
                    message=f"{w['sector']} sector at {w['weight_pct']:.1f}% ({w['count']} stocks)"
                ))

            # Get portfolio value from DB
            portfolio_state = self.db.get_portfolio_state()
            portfolio_value = portfolio_state['total_value'] if portfolio_state else 0
            debt_balance = portfolio_state['debt_fund_balance'] if portfolio_state else 0

            report = MonitoringReport(
                run_date=datetime.now(),
                portfolio_value=portfolio_value,
                debt_fund_balance=debt_balance,
                signals=signals,
                position_warnings=position_warnings,
                sector_warnings=sector_warnings,
                consolidation_count=consolidation_count,
                breakout_count=breakout_count,
            )

            # Save signals to DB
            if signals:
                self.db.add_signals_batch(run_id, [
                    {
                        'signal_type': s.signal_type,
                        'symbol': s.symbol,
                        'priority': s.priority,
                        'details': json.dumps(s.details),
                    }
                    for s in signals
                ])

            duration = time.time() - start_time
            self.db.complete_agent_run(
                run_id,
                signals_count=len(signals),
                report_type='daily_brief',
                summary=f"Monitored {len(symbols)} stocks: {len(signals)} signals ({consolidation_count} consolidating, {breakout_count} breakouts)",
                duration_seconds=round(duration, 1),
            )

            logger.info(f"Monitoring complete: {len(signals)} signals in {duration:.1f}s")
            return report

        except Exception as e:
            self.db.fail_agent_run(run_id, str(e))
            logger.error(f"Monitoring failed: {e}")
            raise

    def _get_portfolio_symbols(self) -> List[str]:
        """Get current portfolio symbols from the latest screening or DB."""
        # Check if there's a recent screening result with top stocks
        runs = self.db.get_recent_runs(limit=5, agent_type='screening')
        if runs:
            # Use the most recent screening's top 30
            latest = runs[0]
            if latest.get('summary'):
                logger.info(f"Using latest screening from {latest['run_date']}")

        # Fall back to getting symbols from active signals or empty list
        signals = self.db.get_active_signals(limit=50)
        symbols = list(set(s['symbol'] for s in signals if s.get('symbol')))
        return symbols[:30]

    def _calculate_ath_drawdown(self, symbol: str) -> Optional[float]:
        """Calculate drawdown from 52-week high using market data."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cutoff = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            df = conn.execute("""
                SELECT close FROM market_data_unified
                WHERE symbol = ? AND date >= ?
                ORDER BY date
            """, (symbol, cutoff)).fetchall()
            conn.close()

            if not df or len(df) < 5:
                return None

            closes = [row[0] for row in df]
            high_52w = max(closes)
            current = closes[-1]

            if high_52w <= 0:
                return None

            drawdown = (high_52w - current) / high_52w
            return drawdown

        except Exception:
            return None

    def _check_position_limits(self, symbols: List[str]) -> List[Dict]:
        """Check for positions exceeding 10% weight."""
        warnings = []
        if len(symbols) <= 1:
            return warnings

        # Get latest prices for all symbols
        prices = self._get_latest_prices(symbols)
        if not prices:
            return warnings

        # Approximate equal-weight check
        equal_weight = 100.0 / len(symbols)
        # In reality we'd need actual position values, but for monitoring
        # we flag based on price movement divergence from equal weight
        # This is a simplified check - full implementation needs portfolio values
        return warnings

    def _check_sector_limits(self, symbols: List[str]) -> List[Dict]:
        """Check for sector concentration exceeding limits."""
        warnings = []
        try:
            from .nifty500_universe import get_nifty500
            universe = get_nifty500()

            sector_counts = {}
            for symbol in symbols:
                stock = universe.get(symbol)
                if stock:
                    sector = stock.sector
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1

            for sector, count in sector_counts.items():
                weight_pct = (count / len(symbols)) * 100 if symbols else 0
                if count > MQ_DEFAULTS['max_stocks_per_sector'] or weight_pct > MQ_DEFAULTS['max_sector_weight'] * 100:
                    warnings.append({
                        'sector': sector,
                        'count': count,
                        'weight_pct': round(weight_pct, 1),
                        'limit_count': MQ_DEFAULTS['max_stocks_per_sector'],
                        'limit_weight_pct': MQ_DEFAULTS['max_sector_weight'] * 100,
                    })
        except Exception as e:
            logger.debug(f"Sector limit check failed: {e}")

        return warnings

    def _get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest closing prices for symbols from market data."""
        prices = {}
        try:
            conn = sqlite3.connect(str(DB_PATH))
            placeholders = ','.join('?' * len(symbols))
            rows = conn.execute(f"""
                SELECT symbol, close FROM market_data_unified
                WHERE symbol IN ({placeholders})
                AND date = (SELECT MAX(date) FROM market_data_unified WHERE symbol = market_data_unified.symbol)
                GROUP BY symbol
            """, symbols).fetchall()
            conn.close()

            for row in rows:
                prices[row[0]] = row[1]
        except Exception as e:
            logger.debug(f"Could not fetch prices: {e}")

        return prices
