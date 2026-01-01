"""
Metrics Calculator for Covered Calls Backtester
================================================

Calculates all performance metrics required for backtest analysis.

Metrics Calculated:
- Total Return (%)
- Premium Yield (%)
- Win Rate (%)
- Max Drawdown (%)
- Sharpe Ratio
- Assignment Rate (%)
- Covered Call vs Buy-and-Hold comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.07  # 7% India benchmark


class MetricsCalculator:
    """
    Calculate performance metrics for covered call backtests
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        """
        Initialize calculator

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate

    # =========================================================================
    # Return Metrics
    # =========================================================================

    def calculate_total_return(
        self,
        initial_value: float,
        final_value: float
    ) -> float:
        """
        Calculate total percentage return

        Args:
            initial_value: Starting portfolio value
            final_value: Ending portfolio value

        Returns:
            Total return as percentage (e.g., 25.5 for 25.5%)
        """
        if initial_value <= 0:
            return 0.0
        return ((final_value - initial_value) / initial_value) * 100

    def calculate_annualized_return(
        self,
        total_return_pct: float,
        days: int
    ) -> float:
        """
        Calculate annualized return from total return

        Args:
            total_return_pct: Total return as percentage
            days: Number of days in period

        Returns:
            Annualized return as percentage
        """
        if days <= 0:
            return 0.0

        # Convert to decimal, annualize, convert back to percentage
        total_decimal = total_return_pct / 100
        years = days / 365
        annualized = (1 + total_decimal) ** (1 / years) - 1
        return annualized * 100

    def calculate_premium_yield(
        self,
        total_premium_collected: float,
        average_stock_exposure: float
    ) -> float:
        """
        Calculate premium yield (income from selling calls)

        Args:
            total_premium_collected: Sum of all premiums received
            average_stock_exposure: Average capital tied up in stock

        Returns:
            Premium yield as percentage
        """
        if average_stock_exposure <= 0:
            return 0.0
        return (total_premium_collected / average_stock_exposure) * 100

    # =========================================================================
    # Risk Metrics
    # =========================================================================

    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series
    ) -> float:
        """
        Calculate maximum drawdown from equity curve

        Args:
            equity_curve: Series of portfolio values indexed by date

        Returns:
            Max drawdown as positive percentage (e.g., 15.0 for 15% drawdown)
        """
        if len(equity_curve) == 0:
            return 0.0

        # Calculate running maximum
        running_max = equity_curve.cummax()

        # Calculate drawdown at each point
        drawdown = (running_max - equity_curve) / running_max

        # Get maximum drawdown
        max_dd = drawdown.max()

        return max_dd * 100

    def calculate_drawdown_series(
        self,
        equity_curve: pd.Series
    ) -> pd.Series:
        """
        Calculate drawdown at each point in time

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Series of drawdown percentages
        """
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max
        return drawdown * 100

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate Sharpe Ratio

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns

        Args:
            returns: Series of period returns (as decimals)
            periods_per_year: Number of periods per year (252 for daily)

        Returns:
            Annualized Sharpe Ratio
        """
        if len(returns) < 2:
            return 0.0

        # Remove NaN values
        returns = returns.dropna()

        if len(returns) < 2:
            return 0.0

        # Calculate excess returns (annualized risk-free rate converted to per-period)
        rf_per_period = self.risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period

        # Calculate Sharpe
        mean_excess = excess_returns.mean()
        std_returns = returns.std()

        if std_returns == 0:
            return 0.0

        # Annualize
        sharpe = (mean_excess / std_returns) * np.sqrt(periods_per_year)

        return sharpe

    def calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 60,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio

        Args:
            returns: Series of period returns
            window: Rolling window size (days)
            periods_per_year: Number of periods per year

        Returns:
            Series of rolling Sharpe ratios
        """
        def sharpe_calc(x):
            if len(x) < 2 or x.std() == 0:
                return 0.0
            rf_per_period = self.risk_free_rate / periods_per_year
            excess = x - rf_per_period
            return (excess.mean() / x.std()) * np.sqrt(periods_per_year)

        return returns.rolling(window=window).apply(sharpe_calc, raw=False)

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Calculate Sortino Ratio (only penalizes downside volatility)

        Args:
            returns: Series of period returns
            periods_per_year: Number of periods per year

        Returns:
            Annualized Sortino Ratio
        """
        if len(returns) < 2:
            return 0.0

        returns = returns.dropna()
        rf_per_period = self.risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period

        # Only consider negative returns for downside deviation
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return float('inf')  # No negative returns

        downside_std = negative_returns.std()

        if downside_std == 0:
            return 0.0

        mean_excess = excess_returns.mean()
        sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)

        return sortino

    # =========================================================================
    # Trade Metrics
    # =========================================================================

    def calculate_win_rate(
        self,
        winning_trades: int,
        total_trades: int
    ) -> float:
        """
        Calculate win rate (percentage of profitable trades)

        Args:
            winning_trades: Number of trades with positive P&L
            total_trades: Total number of trades

        Returns:
            Win rate as percentage
        """
        if total_trades <= 0:
            return 0.0
        return (winning_trades / total_trades) * 100

    def calculate_assignment_rate(
        self,
        assigned_trades: int,
        total_trades: int
    ) -> float:
        """
        Calculate assignment rate (percentage of calls that got assigned)

        For covered calls, assignment means stock was called away at strike.

        Args:
            assigned_trades: Number of trades that resulted in assignment
            total_trades: Total number of trades

        Returns:
            Assignment rate as percentage
        """
        if total_trades <= 0:
            return 0.0
        return (assigned_trades / total_trades) * 100

    def calculate_average_trade(
        self,
        trades_pnl: List[float]
    ) -> Dict[str, float]:
        """
        Calculate average trade statistics

        Args:
            trades_pnl: List of P&L values for each trade

        Returns:
            Dict with average_win, average_loss, profit_factor, etc.
        """
        if not trades_pnl:
            return {
                'average_trade': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'profit_factor': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0
            }

        wins = [p for p in trades_pnl if p > 0]
        losses = [p for p in trades_pnl if p < 0]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'average_trade': np.mean(trades_pnl),
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max(trades_pnl) if trades_pnl else 0.0,
            'max_loss': min(trades_pnl) if trades_pnl else 0.0,
            'total_wins': len(wins),
            'total_losses': len(losses)
        }

    # =========================================================================
    # Comparison Metrics
    # =========================================================================

    def calculate_vs_buy_hold(
        self,
        covered_call_return: float,
        buy_hold_return: float
    ) -> float:
        """
        Calculate excess return of covered call vs buy-and-hold

        Args:
            covered_call_return: Total return from covered call strategy (%)
            buy_hold_return: Total return from buy-and-hold (%)

        Returns:
            Excess return (positive = covered call outperformed)
        """
        return covered_call_return - buy_hold_return

    def calculate_buy_hold_return(
        self,
        stock_prices: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate buy-and-hold return for comparison

        Args:
            stock_prices: DataFrame with columns = symbols, rows = dates
            weights: Optional dict of {symbol: weight}, default equal weight

        Returns:
            Total buy-and-hold return as percentage
        """
        if len(stock_prices) < 2:
            return 0.0

        # Default to equal weight
        symbols = stock_prices.columns.tolist()
        if weights is None:
            weights = {s: 1.0 / len(symbols) for s in symbols}

        total_return = 0.0
        for symbol in symbols:
            if symbol not in stock_prices.columns:
                continue

            prices = stock_prices[symbol].dropna()
            if len(prices) < 2:
                continue

            stock_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            total_return += stock_return * weights.get(symbol, 0)

        return total_return * 100

    # =========================================================================
    # Comprehensive Metrics
    # =========================================================================

    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        initial_capital: float = 1000000.0
    ) -> Dict[str, float]:
        """
        Calculate all metrics from equity curve and trades

        Args:
            equity_curve: Series of daily portfolio values
            trades: DataFrame with trade log (must have 'total_pnl', 'exit_reason' columns)
            initial_capital: Starting capital

        Returns:
            Dict with all calculated metrics
        """
        metrics = {}

        # Return metrics
        if len(equity_curve) > 0:
            final_value = equity_curve.iloc[-1]
            metrics['total_return'] = self.calculate_total_return(
                initial_capital, final_value
            )

            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            metrics['annualized_return'] = self.calculate_annualized_return(
                metrics['total_return'], days
            )

            # Risk metrics
            metrics['max_drawdown'] = self.calculate_max_drawdown(equity_curve)

            # Daily returns for Sharpe
            daily_returns = equity_curve.pct_change().dropna()
            metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(daily_returns)
            metrics['sortino_ratio'] = self.calculate_sortino_ratio(daily_returns)
        else:
            metrics['total_return'] = 0.0
            metrics['annualized_return'] = 0.0
            metrics['max_drawdown'] = 0.0
            metrics['sharpe_ratio'] = 0.0
            metrics['sortino_ratio'] = 0.0

        # Trade metrics
        if len(trades) > 0:
            total_trades = len(trades)
            profitable = len(trades[trades['total_pnl'] > 0])
            assigned = len(trades[trades['exit_reason'] == 'ASSIGNED'])

            metrics['total_trades'] = total_trades
            metrics['profitable_trades'] = profitable
            metrics['losing_trades'] = total_trades - profitable
            metrics['win_rate'] = self.calculate_win_rate(profitable, total_trades)
            metrics['assignment_rate'] = self.calculate_assignment_rate(
                assigned, total_trades
            )

            # Trade statistics
            trade_stats = self.calculate_average_trade(
                trades['total_pnl'].tolist()
            )
            metrics.update(trade_stats)

            # Premium yield
            if 'premium_received' in trades.columns:
                total_premium = trades['premium_received'].sum()
                avg_stock_value = trades['stock_entry_price'].mean() * \
                                  trades.get('lot_size', pd.Series([1] * len(trades))).mean()
                metrics['premium_yield'] = self.calculate_premium_yield(
                    total_premium, avg_stock_value * len(trades)
                )
            else:
                metrics['premium_yield'] = 0.0
        else:
            metrics['total_trades'] = 0
            metrics['profitable_trades'] = 0
            metrics['losing_trades'] = 0
            metrics['win_rate'] = 0.0
            metrics['assignment_rate'] = 0.0
            metrics['premium_yield'] = 0.0

        return metrics

    # =========================================================================
    # Strategy Comparison
    # =========================================================================

    def compare_strategies(
        self,
        results: List[Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple strategy results

        Args:
            results: List of dicts with strategy name and metrics

        Returns:
            DataFrame with comparison of all strategies
        """
        comparison_columns = [
            'name', 'total_return', 'annualized_return', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'assignment_rate', 'total_trades'
        ]

        data = []
        for result in results:
            row = {col: result.get(col, 0) for col in comparison_columns}
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by total return descending
        df = df.sort_values('total_return', ascending=False)

        return df


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Calculate daily returns from equity curve"""
    return equity_curve.pct_change().dropna()


def calculate_cumulative_returns(daily_returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns from daily returns"""
    return (1 + daily_returns).cumprod() - 1


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """Calculate rolling volatility"""
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol
