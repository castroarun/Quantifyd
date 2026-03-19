"""
Strategy Optimizer - Automated Backtesting of All Indicator Combinations
Finds optimal covered call strategies by testing multiple parameter combinations.
"""

import itertools
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from services.covered_call_service import (
    CoveredCallEngine, BacktestConfig, StrikeMethod, ExitStrategy
)
from config import LOT_SIZES, DEFAULT_CAPITAL
from data_simulator import SimulatedDataManager, generate_all_stock_data


@dataclass
class StrategyResult:
    """Result from a single strategy backtest"""
    strategy_name: str
    config: Dict[str, Any]
    total_return_pct: float
    annualized_return_pct: float
    win_rate: float
    total_trades: int
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_pnl: float
    avg_holding_days: float
    score: float = 0.0  # Composite score for ranking


class StrategyOptimizer:
    """
    Automated strategy optimizer that tests all indicator combinations
    and finds the best performing covered call strategies.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        initial_capital: float = DEFAULT_CAPITAL,
        max_parallel: int = 4
    ):
        self.symbols = symbols or list(LOT_SIZES.keys())[:10]  # Top 10 by default
        self.start_date = start_date or datetime(2023, 1, 1)
        self.end_date = end_date or datetime(2024, 12, 31)
        self.initial_capital = initial_capital
        self.max_parallel = max_parallel
        self.results: List[StrategyResult] = []
        self.data_manager = SimulatedDataManager(self.start_date, self.end_date)
        self.stock_data = {}

    def load_data(self) -> bool:
        """Load historical data for all symbols"""
        print(f"Loading data for {len(self.symbols)} symbols...")

        for symbol in self.symbols:
            try:
                data = self.data_manager.get_stock_data(
                    symbol=symbol,
                    from_date=self.start_date - timedelta(days=250),  # Extra for indicators
                    to_date=self.end_date
                )
                if data is not None and len(data) > 100:
                    self.stock_data[symbol] = data
                    print(f"  {symbol}: {len(data)} bars loaded")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")

        print(f"Loaded data for {len(self.stock_data)} symbols")
        return len(self.stock_data) > 0

    def generate_strategy_configs(self) -> List[Dict[str, Any]]:
        """Generate all strategy combinations to test"""
        configs = []

        # Base parameters (always tested)
        strike_methods = [
            "DELTA_30", "DELTA_40", "OTM_2PCT", "OTM_5PCT",
            "ATR_BASED", "ADAPTIVE_DELTA", "BOLLINGER_UPPER", "PIVOT_R1"
        ]

        exit_strategies = [
            "HOLD_TO_EXPIRY", "PROFIT_TARGET", "STOP_LOSS",
            "PROFIT_TARGET_AND_STOP_LOSS"
        ]

        # Entry filter combinations
        entry_filters = [
            # No filter (baseline)
            {"name": "No Filter", "filters": {}},

            # Single filters
            {"name": "Bullish EMA", "filters": {"trend_filter_mode": "BULLISH"}},
            {"name": "Bullish Aligned EMA", "filters": {"trend_filter_mode": "BULLISH_ALIGNED"}},
            {"name": "Golden Cross", "filters": {"trend_filter_mode": "GOLDEN_CROSS"}},
            {"name": "RSI 40-70", "filters": {"use_rsi_filter": True, "rsi_min": 40, "rsi_max": 70}},
            {"name": "RSI 30-60", "filters": {"use_rsi_filter": True, "rsi_min": 30, "rsi_max": 60}},
            {"name": "Supertrend", "filters": {"use_supertrend_filter": True}},
            {"name": "MACD Bullish", "filters": {"use_macd_filter": True, "macd_mode": "BULLISH"}},
            {"name": "ADX Strong Trend", "filters": {"use_adx_filter": True, "adx_threshold": 25}},
            {"name": "ADX Weak Trend", "filters": {"use_adx_filter": True, "adx_threshold": 20}},
            {"name": "Bollinger Not OB", "filters": {"use_bollinger_filter": True}},
            {"name": "VWAP Above", "filters": {"use_vwap_filter": True, "vwap_mode": "ABOVE"}},
            {"name": "Stochastic OB", "filters": {"use_stochastic_filter": True}},
            {"name": "Williams %R", "filters": {"use_williams_filter": True}},

            # Dual filter combinations
            {"name": "EMA+RSI", "filters": {"trend_filter_mode": "BULLISH", "use_rsi_filter": True}},
            {"name": "EMA+Supertrend", "filters": {"trend_filter_mode": "BULLISH", "use_supertrend_filter": True}},
            {"name": "EMA+MACD", "filters": {"trend_filter_mode": "BULLISH", "use_macd_filter": True}},
            {"name": "RSI+Supertrend", "filters": {"use_rsi_filter": True, "use_supertrend_filter": True}},
            {"name": "RSI+MACD", "filters": {"use_rsi_filter": True, "use_macd_filter": True}},
            {"name": "ADX+MACD", "filters": {"use_adx_filter": True, "use_macd_filter": True}},
            {"name": "Bollinger+RSI", "filters": {"use_bollinger_filter": True, "use_rsi_filter": True}},
            {"name": "VWAP+RSI", "filters": {"use_vwap_filter": True, "use_rsi_filter": True}},
            {"name": "Supertrend+ADX", "filters": {"use_supertrend_filter": True, "use_adx_filter": True}},

            # Triple filter combinations (best of breed)
            {"name": "EMA+RSI+Supertrend", "filters": {
                "trend_filter_mode": "BULLISH", "use_rsi_filter": True, "use_supertrend_filter": True}},
            {"name": "EMA+RSI+MACD", "filters": {
                "trend_filter_mode": "BULLISH", "use_rsi_filter": True, "use_macd_filter": True}},
            {"name": "ADX+RSI+MACD", "filters": {
                "use_adx_filter": True, "use_rsi_filter": True, "use_macd_filter": True}},
            {"name": "Supertrend+RSI+Bollinger", "filters": {
                "use_supertrend_filter": True, "use_rsi_filter": True, "use_bollinger_filter": True}},
        ]

        # Exit variations
        exit_params = [
            {"profit_target_pct": 50, "stop_loss_multiple": 2.0, "allow_sl_adjustment": False},
            {"profit_target_pct": 40, "stop_loss_multiple": 2.0, "allow_sl_adjustment": False},
            {"profit_target_pct": 60, "stop_loss_multiple": 2.0, "allow_sl_adjustment": False},
            {"profit_target_pct": 50, "stop_loss_multiple": 1.5, "allow_sl_adjustment": False},
            {"profit_target_pct": 50, "stop_loss_multiple": 2.0, "allow_sl_adjustment": True},  # Roll-up
            {"profit_target_pct": 50, "stop_loss_multiple": 2.0, "use_dte_exit": True, "dte_exit_threshold": 7},
            {"profit_target_pct": 50, "stop_loss_multiple": 2.0, "use_trailing_stop": True},
        ]

        # Generate combinations
        config_id = 0
        for strike_method in strike_methods:
            for exit_strategy in exit_strategies:
                for entry_filter in entry_filters:
                    for exit_param in exit_params:
                        # Skip invalid combinations
                        if exit_strategy == "HOLD_TO_EXPIRY" and exit_param.get("use_trailing_stop"):
                            continue
                        if exit_strategy == "HOLD_TO_EXPIRY" and exit_param.get("use_dte_exit"):
                            continue

                        config = {
                            "id": config_id,
                            "name": f"{strike_method}_{entry_filter['name']}_{exit_strategy}",
                            "strike_method": strike_method,
                            "exit_strategy": exit_strategy,
                            **entry_filter["filters"],
                            **exit_param
                        }
                        configs.append(config)
                        config_id += 1

        print(f"Generated {len(configs)} strategy configurations to test")
        return configs

    def run_single_backtest(self, config: Dict[str, Any]) -> Optional[StrategyResult]:
        """Run a single backtest configuration"""
        try:
            # Build BacktestConfig
            backtest_config = BacktestConfig(
                symbols=list(self.stock_data.keys()),
                start_date=self.start_date,
                end_date=self.end_date,
                strike_method=StrikeMethod(config.get("strike_method", "DELTA_30")),
                exit_strategy=ExitStrategy(config.get("exit_strategy", "PROFIT_TARGET_AND_STOP_LOSS")),
                initial_capital=self.initial_capital,
                profit_target_pct=config.get("profit_target_pct", 50.0),
                stop_loss_multiple=config.get("stop_loss_multiple", 2.0),
                allow_sl_adjustment=config.get("allow_sl_adjustment", False),
                # Trend filter
                trend_filter_mode=config.get("trend_filter_mode", "NONE"),
                # RSI filter
                use_rsi_filter=config.get("use_rsi_filter", False),
                rsi_period=config.get("rsi_period", 14),
                rsi_min=config.get("rsi_min", 40.0),
                rsi_max=config.get("rsi_max", 70.0),
                # Supertrend
                use_supertrend_filter=config.get("use_supertrend_filter", False),
                supertrend_period=config.get("supertrend_period", 10),
                supertrend_multiplier=config.get("supertrend_multiplier", 3.0),
                # MACD
                use_macd_filter=config.get("use_macd_filter", False),
                macd_fast=config.get("macd_fast", 12),
                macd_slow=config.get("macd_slow", 26),
                macd_signal=config.get("macd_signal", 9),
                macd_mode=config.get("macd_mode", "BULLISH"),
                # ADX
                use_adx_filter=config.get("use_adx_filter", False),
                adx_period=config.get("adx_period", 14),
                adx_threshold=config.get("adx_threshold", 25.0),
                adx_require_bullish=config.get("adx_require_bullish", True),
                # Bollinger
                use_bollinger_filter=config.get("use_bollinger_filter", False),
                bollinger_period=config.get("bollinger_period", 20),
                bollinger_std=config.get("bollinger_std", 2.0),
                # VWAP
                use_vwap_filter=config.get("use_vwap_filter", False),
                vwap_mode=config.get("vwap_mode", "ABOVE"),
                # Stochastic
                use_stochastic_filter=config.get("use_stochastic_filter", False),
                stochastic_k_period=config.get("stochastic_k_period", 14),
                stochastic_d_period=config.get("stochastic_d_period", 3),
                stochastic_overbought=config.get("stochastic_overbought", 70.0),
                # Williams %R
                use_williams_filter=config.get("use_williams_filter", False),
                williams_period=config.get("williams_period", 14),
                williams_overbought=config.get("williams_overbought", -20.0),
                # Advanced exits
                use_dte_exit=config.get("use_dte_exit", False),
                dte_exit_threshold=config.get("dte_exit_threshold", 7),
                use_trailing_stop=config.get("use_trailing_stop", False),
                trailing_stop_activation=config.get("trailing_stop_activation", 25.0),
                trailing_stop_distance=config.get("trailing_stop_distance", 15.0),
                # ATR-based strike
                atr_multiplier=config.get("atr_multiplier", 1.5),
            )

            # Run backtest
            engine = CoveredCallEngine(backtest_config)
            results = engine.run_backtest(self.stock_data)

            # Extract metrics from results
            metrics = results.get("metrics", {})
            total_trades = metrics.get("total_trades", 0)

            if not results or total_trades < 3:
                return None  # Skip strategies with too few trades

            # Get return (key is 'total_return', not 'total_return_pct')
            total_return = metrics.get("total_return", 0)

            # Calculate days in backtest period
            days = (self.end_date - self.start_date).days
            years = days / 365.25

            # Annualized return
            if total_return > -100:
                annualized = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
            else:
                annualized = -100

            # Extract other metrics (using correct key names)
            win_rate = metrics.get("win_rate", 0)
            max_drawdown = metrics.get("max_drawdown", 0)  # Not max_drawdown_pct
            sharpe = metrics.get("sharpe_ratio", 0)
            profit_factor = metrics.get("profit_factor", 1.0)
            avg_pnl = metrics.get("avg_win", 0) - abs(metrics.get("avg_loss", 0))  # Compute from avg_win/avg_loss
            avg_hold = metrics.get("avg_holding_days", 0)

            # Calculate composite score
            # Weighted factors: Return (30%), Win Rate (20%), Sharpe (25%), Drawdown (15%), Trade Frequency (10%)
            score = (
                annualized * 0.30 +
                win_rate * 0.20 +
                sharpe * 10 * 0.25 +  # Scale sharpe
                (100 - abs(max_drawdown)) * 0.15 +
                min(total_trades / 10, 10) * 0.10  # Cap trade frequency bonus
            )

            return StrategyResult(
                strategy_name=config.get("name", "Unknown"),
                config=config,
                total_return_pct=total_return,
                annualized_return_pct=annualized,
                win_rate=win_rate,
                total_trades=total_trades,
                max_drawdown_pct=max_drawdown,
                sharpe_ratio=sharpe,
                profit_factor=profit_factor,
                avg_trade_pnl=avg_pnl,
                avg_holding_days=avg_hold,
                score=score
            )

        except Exception as e:
            print(f"  Error in config {config.get('id')}: {e}")
            return None

    def run_optimization(self, sample_size: int = None) -> List[StrategyResult]:
        """Run the full optimization process"""
        print("=" * 60)
        print("COVERED CALL STRATEGY OPTIMIZER")
        print("=" * 60)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: Rs {self.initial_capital:,.0f}")
        print("=" * 60)

        # Load data
        if not self.load_data():
            print("ERROR: No data loaded. Exiting.")
            return []

        # Generate configs
        configs = self.generate_strategy_configs()

        # Sample if requested
        if sample_size and sample_size < len(configs):
            import random
            configs = random.sample(configs, sample_size)
            print(f"Sampled {sample_size} configurations for testing")

        # Run backtests
        print(f"\nRunning {len(configs)} backtests...")
        completed = 0

        for config in configs:
            result = self.run_single_backtest(config)
            if result:
                self.results.append(result)

            completed += 1
            if completed % 50 == 0:
                print(f"  Progress: {completed}/{len(configs)} ({100*completed/len(configs):.1f}%)")

        # Sort by score
        self.results.sort(key=lambda x: x.score, reverse=True)

        print(f"\nCompleted! {len(self.results)} valid strategies found.")
        return self.results

    def get_top_strategies(self, n: int = 20) -> List[StrategyResult]:
        """Get top N strategies by score"""
        return self.results[:n]

    def print_results(self, top_n: int = 20):
        """Print formatted results"""
        print("\n" + "=" * 80)
        print(f"TOP {top_n} COVERED CALL STRATEGIES")
        print("=" * 80)

        top = self.get_top_strategies(top_n)

        for i, result in enumerate(top, 1):
            print(f"\n{'-' * 80}")
            print(f"#{i} - {result.strategy_name}")
            print(f"{'-' * 80}")
            print(f"  Total Return:      {result.total_return_pct:>8.2f}%")
            print(f"  Annualized Return: {result.annualized_return_pct:>8.2f}%")
            print(f"  Win Rate:          {result.win_rate:>8.1f}%")
            print(f"  Total Trades:      {result.total_trades:>8d}")
            print(f"  Max Drawdown:      {result.max_drawdown_pct:>8.2f}%")
            print(f"  Sharpe Ratio:      {result.sharpe_ratio:>8.2f}")
            print(f"  Profit Factor:     {result.profit_factor:>8.2f}")
            print(f"  Avg Trade P&L:     Rs {result.avg_trade_pnl:>8,.0f}")
            print(f"  Avg Holding Days:  {result.avg_holding_days:>8.1f}")
            print(f"  COMPOSITE SCORE:   {result.score:>8.2f}")

            # Print key config details
            config = result.config
            print(f"\n  Configuration:")
            print(f"    Strike Method: {config.get('strike_method')}")
            print(f"    Exit Strategy: {config.get('exit_strategy')}")

            filters = []
            if config.get('trend_filter_mode', 'NONE') != 'NONE':
                filters.append(f"EMA:{config['trend_filter_mode']}")
            if config.get('use_rsi_filter'):
                filters.append(f"RSI:{config.get('rsi_min', 40)}-{config.get('rsi_max', 70)}")
            if config.get('use_supertrend_filter'):
                filters.append("Supertrend")
            if config.get('use_macd_filter'):
                filters.append(f"MACD:{config.get('macd_mode', 'BULLISH')}")
            if config.get('use_adx_filter'):
                filters.append(f"ADX>{config.get('adx_threshold', 25)}")
            if config.get('use_bollinger_filter'):
                filters.append("Bollinger")
            if config.get('use_vwap_filter'):
                filters.append(f"VWAP:{config.get('vwap_mode', 'ABOVE')}")
            if config.get('use_stochastic_filter'):
                filters.append("Stochastic")
            if config.get('use_williams_filter'):
                filters.append("Williams%R")

            if filters:
                print(f"    Entry Filters: {', '.join(filters)}")
            else:
                print(f"    Entry Filters: None")

            print(f"    Profit Target: {config.get('profit_target_pct', 50)}%")
            print(f"    Stop Loss: {config.get('stop_loss_multiple', 2.0)}x premium")
            if config.get('allow_sl_adjustment'):
                print(f"    SL Adjustment: Yes (Roll-up)")
            if config.get('use_dte_exit'):
                print(f"    DTE Exit: {config.get('dte_exit_threshold', 7)} days")
            if config.get('use_trailing_stop'):
                print(f"    Trailing Stop: {config.get('trailing_stop_activation', 25)}% / {config.get('trailing_stop_distance', 15)}%")

    def save_results(self, filepath: str = None):
        """Save results to JSON file"""
        if not filepath:
            filepath = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output = {
            "metadata": {
                "symbols": self.symbols,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "initial_capital": self.initial_capital,
                "total_strategies_tested": len(self.results),
                "generated_at": datetime.now().isoformat()
            },
            "top_strategies": [
                {
                    "rank": i + 1,
                    "name": r.strategy_name,
                    "total_return_pct": r.total_return_pct,
                    "annualized_return_pct": r.annualized_return_pct,
                    "win_rate": r.win_rate,
                    "total_trades": r.total_trades,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "sharpe_ratio": r.sharpe_ratio,
                    "profit_factor": r.profit_factor,
                    "avg_trade_pnl": r.avg_trade_pnl,
                    "avg_holding_days": r.avg_holding_days,
                    "score": r.score,
                    "config": r.config
                }
                for i, r in enumerate(self.results[:50])  # Top 50
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main entry point for optimization"""
    # Configure optimizer
    optimizer = StrategyOptimizer(
        symbols=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                 "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "AXISBANK"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=1000000
    )

    # Run optimization (sample 500 for faster testing, remove for full run)
    results = optimizer.run_optimization(sample_size=500)

    # Print and save results
    if results:
        optimizer.print_results(top_n=20)
        optimizer.save_results()

    return optimizer


if __name__ == "__main__":
    main()
