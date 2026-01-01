"""
CPR Strategy Optimizer
======================

Grid search optimization agent for CPR-based covered call strategy parameters.
Runs backtests across all parameter combinations and ranks by composite score.
"""

import itertools
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .cpr_covered_call_service import CPRCoveredCallEngine, CPRBacktestConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Default Optimization Grid
# =============================================================================

DEFAULT_OPTIMIZATION_GRID = {
    'narrow_cpr_threshold': [0.3, 0.5, 0.7, 1.0],
    'otm_strike_pct': [3.0, 5.0, 7.0, 10.0],
    'dte_min': [25, 30, 35],
    'dte_max': [30, 35, 40, 45],
    'premium_double_threshold': [1.5, 2.0, 2.5, 3.0],
    'premium_erosion_target': [60.0, 70.0, 75.0, 80.0, 85.0],
    'dte_exit_threshold': [7, 10, 14],
    'enable_r1_exit': [True, False],
    'use_closer_r1': [True, False],
}

# Quick grid for faster testing
QUICK_OPTIMIZATION_GRID = {
    'narrow_cpr_threshold': [0.5, 1.0],
    'otm_strike_pct': [5.0, 7.0],
    'dte_min': [30],
    'dte_max': [35, 40],
    'premium_double_threshold': [2.0],
    'premium_erosion_target': [75.0, 80.0],
    'dte_exit_threshold': [10],
    'enable_r1_exit': [True],
    'use_closer_r1': [True],
}


@dataclass
class OptimizationResult:
    """Result from a single parameter combination backtest"""
    params: Dict[str, Any]
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_pnl: float
    composite_score: float


class CPRStrategyOptimizer:
    """
    Grid search optimizer for CPR covered call strategy.

    Runs backtests across all parameter combinations and ranks by composite score.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        stock_data: Dict[str, pd.DataFrame],
        intraday_data: Dict[str, pd.DataFrame],
        initial_capital: float = 1000000.0
    ):
        """
        Initialize optimizer.

        Args:
            symbols: List of stock symbols
            start_date: Backtest start date
            end_date: Backtest end date
            stock_data: Dictionary of symbol -> daily OHLCV DataFrame
            intraday_data: Dictionary of symbol -> 30-min OHLCV DataFrame
            initial_capital: Initial capital for each backtest
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = stock_data
        self.intraday_data = intraday_data
        self.initial_capital = initial_capital
        self.results: List[OptimizationResult] = []

    def generate_param_combinations(
        self,
        grid: Optional[Dict[str, List[Any]]] = None,
        max_combinations: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations from grid.

        Args:
            grid: Parameter grid (uses DEFAULT_OPTIMIZATION_GRID if None)
            max_combinations: Maximum number of combinations to generate

        Returns:
            List of parameter dictionaries
        """
        if grid is None:
            grid = DEFAULT_OPTIMIZATION_GRID

        # Filter valid DTE combinations (min < max)
        valid_combinations = []

        keys = list(grid.keys())
        values = list(grid.values())

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            # Validate DTE range
            if params.get('dte_min', 0) >= params.get('dte_max', 100):
                continue

            valid_combinations.append(params)

            if len(valid_combinations) >= max_combinations:
                break

        logger.info(f"Generated {len(valid_combinations)} valid parameter combinations")
        return valid_combinations

    def run_single_backtest(self, params: Dict[str, Any]) -> Optional[OptimizationResult]:
        """
        Run a single backtest with given parameters.

        Args:
            params: Parameter dictionary

        Returns:
            OptimizationResult or None if backtest failed
        """
        try:
            config = CPRBacktestConfig(
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                narrow_cpr_threshold=params.get('narrow_cpr_threshold', 0.5),
                otm_strike_pct=params.get('otm_strike_pct', 5.0),
                dte_min=params.get('dte_min', 30),
                dte_max=params.get('dte_max', 35),
                enable_premium_rollout=params.get('enable_premium_rollout', True),
                premium_double_threshold=params.get('premium_double_threshold', 2.0),
                premium_erosion_target=params.get('premium_erosion_target', 75.0),
                dte_exit_threshold=params.get('dte_exit_threshold', 10),
                enable_r1_exit=params.get('enable_r1_exit', True),
                use_closer_r1=params.get('use_closer_r1', True),
            )

            engine = CPRCoveredCallEngine(config)
            results = engine.run_backtest(self.stock_data, self.intraday_data)

            # Calculate composite score
            composite_score = self._calculate_composite_score(results)

            return OptimizationResult(
                params=params,
                total_return=results.get('total_return', 0),
                annualized_return=results.get('annualized_return', 0),
                sharpe_ratio=results.get('sharpe_ratio', 0),
                max_drawdown=results.get('max_drawdown', 0),
                win_rate=results.get('win_rate', 0),
                total_trades=results.get('total_trades', 0),
                avg_pnl=results.get('avg_pnl', 0),
                composite_score=composite_score
            )

        except Exception as e:
            logger.error(f"Error in backtest with params {params}: {e}")
            return None

    def _calculate_composite_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate composite score from backtest results.

        Weighted scoring:
        - Annualized Return: 30%
        - Win Rate: 20%
        - Sharpe Ratio (scaled): 25%
        - Max Drawdown (inverted): 15%
        - Trade Frequency: 10%

        Args:
            results: Backtest results dictionary

        Returns:
            Composite score (0-100)
        """
        annualized_return = results.get('annualized_return', 0)
        win_rate = results.get('win_rate', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = abs(results.get('max_drawdown', 0))
        total_trades = results.get('total_trades', 0)

        # Normalize components
        # Return: cap at 50% annual
        return_score = min(annualized_return / 50, 1.0) * 100 if annualized_return > 0 else 0

        # Win rate: already in %
        winrate_score = win_rate

        # Sharpe: scale by 10 (0.5 = 5 points, 2.0 = 20 points, cap at 30)
        sharpe_score = min(sharpe_ratio * 10, 30) if sharpe_ratio > 0 else 0

        # Drawdown: inverted (lower is better)
        # 100 - drawdown, cap negative impact at 50%
        drawdown_score = max(100 - max_drawdown * 2, 0)

        # Trade frequency: more trades = better signal quality (cap at 20 trades)
        trade_score = min(total_trades / 20, 1.0) * 100

        # Weighted average
        composite = (
            return_score * 0.30 +
            winrate_score * 0.20 +
            sharpe_score * 0.25 +
            drawdown_score * 0.15 +
            trade_score * 0.10
        )

        return composite

    def run_optimization(
        self,
        grid: Optional[Dict[str, List[Any]]] = None,
        max_combinations: int = 500,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Run full grid search optimization.

        Args:
            grid: Parameter grid (uses DEFAULT_OPTIMIZATION_GRID if None)
            max_combinations: Maximum combinations to test
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting CPR strategy optimization...")

        combinations = self.generate_param_combinations(grid, max_combinations)
        total = len(combinations)
        self.results = []

        if progress_callback:
            progress_callback(0, f"Testing {total} parameter combinations...")

        for i, params in enumerate(combinations):
            result = self.run_single_backtest(params)

            if result:
                self.results.append(result)

            if progress_callback and (i + 1) % 5 == 0:
                pct = ((i + 1) / total) * 100
                progress_callback(pct, f"Tested {i + 1}/{total} combinations...")

        # Sort by composite score
        self.results.sort(key=lambda x: x.composite_score, reverse=True)

        # Prepare summary
        best_results = self.results[:10] if len(self.results) >= 10 else self.results

        summary = {
            'total_tested': total,
            'successful_tests': len(self.results),
            'best_sharpe': self.results[0].sharpe_ratio if self.results else 0,
            'best_return': self.results[0].total_return if self.results else 0,
            'best_params': self.results[0].params if self.results else {},
            'top_10': [
                {
                    'params': r.params,
                    'total_return': r.total_return,
                    'annualized_return': r.annualized_return,
                    'sharpe_ratio': r.sharpe_ratio,
                    'max_drawdown': r.max_drawdown,
                    'win_rate': r.win_rate,
                    'total_trades': r.total_trades,
                    'composite_score': r.composite_score
                }
                for r in best_results
            ]
        }

        logger.info(f"Optimization complete. Best Sharpe: {summary['best_sharpe']:.2f}")

        return summary

    def get_best_params(self, metric: str = 'composite_score', top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N parameter sets by specified metric.

        Args:
            metric: Metric to sort by ('composite_score', 'sharpe_ratio', 'total_return')
            top_n: Number of results to return

        Returns:
            List of top parameter dictionaries
        """
        if not self.results:
            return []

        sorted_results = sorted(
            self.results,
            key=lambda x: getattr(x, metric, 0),
            reverse=True
        )

        return [r.params for r in sorted_results[:top_n]]

    def export_results(self) -> pd.DataFrame:
        """
        Export all results to DataFrame.

        Returns:
            DataFrame with all optimization results
        """
        if not self.results:
            return pd.DataFrame()

        data = []
        for r in self.results:
            row = r.params.copy()
            row.update({
                'total_return': r.total_return,
                'annualized_return': r.annualized_return,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown': r.max_drawdown,
                'win_rate': r.win_rate,
                'total_trades': r.total_trades,
                'avg_pnl': r.avg_pnl,
                'composite_score': r.composite_score
            })
            data.append(row)

        return pd.DataFrame(data)


def run_quick_optimization(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    stock_data: Dict[str, pd.DataFrame],
    intraday_data: Dict[str, pd.DataFrame],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Run quick optimization with reduced parameter grid.

    Args:
        symbols: List of stock symbols
        start_date: Backtest start date
        end_date: Backtest end date
        stock_data: Dictionary of symbol -> daily OHLCV DataFrame
        intraday_data: Dictionary of symbol -> 30-min OHLCV DataFrame
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with optimization results
    """
    optimizer = CPRStrategyOptimizer(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        stock_data=stock_data,
        intraday_data=intraday_data
    )

    return optimizer.run_optimization(
        grid=QUICK_OPTIMIZATION_GRID,
        max_combinations=50,
        progress_callback=progress_callback
    )