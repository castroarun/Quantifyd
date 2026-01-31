"""
MQ Backtest Agent
==================

On-demand backtesting with benchmark comparison.
Thin wrapper around MQBacktestEngine that adds:
- Benchmark curves (Nifty 50, Nifty 500)
- Agent DB persistence
- Report generation trigger
"""

import logging
import time
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Optional, Callable

try:
    import yfinance as yf
except ImportError:
    yf = None

from .mq_agent_db import get_agent_db, MQAgentDB
from .mq_agent_reports import BacktestReportData
from .mq_backtest_engine import MQBacktestEngine, MQBacktestConfig, BacktestResult
from .nifty500_universe import get_nifty500
from .fundamental_data_service import assess_quality

logger = logging.getLogger(__name__)

# Config
from config import MQ_DEFAULTS


class BacktestAgent:
    """On-demand backtest runner with benchmark comparison."""

    def __init__(self, db: MQAgentDB = None):
        self.db = db or get_agent_db()

    def run(
        self,
        config: MQBacktestConfig = None,
        progress_callback: Callable = None,
    ) -> BacktestReportData:
        """Run a full backtest with benchmarks."""
        start_time = time.time()
        run_id = self.db.start_agent_run('backtest')

        try:
            config = config or MQBacktestConfig()

            # Step 1: Pre-load quality scores (cached)
            logger.info("Loading quality scores...")
            quality_scores = self._load_quality_scores()

            # Step 2: Run backtest engine
            logger.info(f"Running backtest: {config.start_date} to {config.end_date}")
            engine = MQBacktestEngine(config)
            result = engine.run(quality_scores, progress_callback)

            # Step 3: Load benchmark curves
            logger.info("Loading benchmark data...")
            benchmarks = self._load_benchmark_curves(
                config.start_date, config.end_date, config.initial_capital
            )

            # Step 4: Build report data
            # Convert equity curve dates to strings for JSON
            equity_curve = {}
            if hasattr(result, 'daily_equity') and result.daily_equity:
                for date, value in result.daily_equity.items():
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                    equity_curve[date_str] = round(value, 2)

            # Compute benchmark CAGRs for comparison
            start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
            bt_years = (end_dt - start_dt).days / 365.25
            benchmark_cagrs = {}
            for bm_name, bm_data in benchmarks.items():
                if bm_data:
                    sorted_dates = sorted(bm_data.keys())
                    bm_start = bm_data[sorted_dates[0]]
                    bm_end = bm_data[sorted_dates[-1]]
                    if bm_start > 0 and bt_years > 0:
                        benchmark_cagrs[bm_name] = round(
                            ((bm_end / bm_start) ** (1 / bt_years) - 1) * 100, 1
                        )

            report = BacktestReportData(
                run_date=datetime.now(),
                config={
                    'start_date': str(config.start_date),
                    'end_date': str(config.end_date),
                    'initial_capital': config.initial_capital,
                    'portfolio_size': config.portfolio_size,
                },
                metrics={
                    'total_return_pct': round(result.total_return_pct, 1),
                    'cagr': round(result.cagr, 1),
                    'sharpe_ratio': round(result.sharpe_ratio, 2),
                    'sortino_ratio': round(result.sortino_ratio, 2),
                    'max_drawdown': round(result.max_drawdown, 1),
                    'calmar_ratio': round(result.calmar_ratio, 2),
                    'win_rate': round(result.win_rate, 1) if hasattr(result, 'win_rate') else 0,
                    'total_trades': result.total_trades if hasattr(result, 'total_trades') else 0,
                    'total_topups': result.total_topups if hasattr(result, 'total_topups') else 0,
                    'initial_capital': config.initial_capital,
                    'final_value': round(result.final_value, 2),
                    'benchmark_cagrs': benchmark_cagrs,
                    'exit_reason_counts': result.exit_reason_counts,
                    'exit_reason_pnl': result.exit_reason_pnl,
                    'avg_win_pct': result.avg_win_pct if hasattr(result, 'avg_win_pct') else 0,
                    'avg_loss_pct': result.avg_loss_pct if hasattr(result, 'avg_loss_pct') else 0,
                    'winning_trades': result.winning_trades if hasattr(result, 'winning_trades') else 0,
                    'losing_trades': result.losing_trades if hasattr(result, 'losing_trades') else 0,
                },
                equity_curve=equity_curve,
                benchmark_curves=benchmarks,
                trade_count=result.total_trades if hasattr(result, 'total_trades') else 0,
                topup_count=result.total_topups if hasattr(result, 'total_topups') else 0,
                sector_allocation=result.sector_allocation if hasattr(result, 'sector_allocation') else {},
            )

            # Update portfolio state in DB
            self.db.upsert_portfolio_state(
                initial_capital=config.initial_capital,
                equity_deployed=result.final_value * 0.8 if hasattr(result, 'final_value') else 0,
                debt_fund_balance=result.final_value * 0.2 if hasattr(result, 'final_value') else 0,
                total_value=result.final_value if hasattr(result, 'final_value') else 0,
                positions_count=len(result.final_positions) if hasattr(result, 'final_positions') else 0,
            )

            duration = time.time() - start_time
            self.db.complete_agent_run(
                run_id,
                signals_count=0,
                report_type='backtest',
                summary=f"CAGR {report.metrics['cagr']}%, Sharpe {report.metrics['sharpe_ratio']}, MaxDD {report.metrics['max_drawdown']}%",
                duration_seconds=round(duration, 1),
            )

            logger.info(f"Backtest complete in {duration:.1f}s: CAGR={report.metrics['cagr']}%")
            return report

        except Exception as e:
            self.db.fail_agent_run(run_id, str(e))
            logger.error(f"Backtest failed: {e}")
            raise

    def _load_quality_scores(self) -> Dict[str, float]:
        """Load cached quality scores for the universe."""
        scores = {}
        try:
            universe = get_nifty500()
            for symbol in universe.symbols[:50]:  # Sample for speed
                try:
                    result = assess_quality(
                        symbol, universe.is_financial(symbol), use_cache=True
                    )
                    if result and result.composite_score > 0:
                        scores[symbol] = result.composite_score
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not load quality scores: {e}")

        return scores

    def _load_benchmark_curves(
        self, start_date, end_date, initial_capital: float = 10_000_000
    ) -> Dict[str, Dict[str, float]]:
        """Load Nifty 50 and 500 benchmark curves, normalized to same starting capital."""
        benchmarks = {}

        if not yf:
            return benchmarks

        tickers = {
            'Nifty 50': '^NSEI',
            'Nifty 500': '^CRSLDX',
        }

        start_str = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d')
        end_str = end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d')

        for name, ticker_symbol in tickers.items():
            try:
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(start=start_str, end=end_str)

                if hist.empty:
                    continue

                # Normalize to same starting capital as strategy
                base = hist['Close'].iloc[0]
                if base <= 0:
                    continue

                normalized = {}
                for date, close in hist['Close'].items():
                    date_str = date.strftime('%Y-%m-%d')
                    normalized[date_str] = round((close / base) * initial_capital, 2)

                benchmarks[name] = normalized
            except Exception as e:
                logger.warning(f"Could not load benchmark {name}: {e}")

        return benchmarks
