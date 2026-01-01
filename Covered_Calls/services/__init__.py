"""
Services module for Covered Calls Backtester

Components:
- kite_service: Zerodha Kite Connect authentication and API access
- greeks_calculator: Black-Scholes options pricing and Greeks
- data_manager: Market data download and storage
- backtest_db: Backtest results storage and retrieval
"""

from .kite_service import (
    get_kite,
    get_kite_with_refresh,
    get_access_token,
    save_access_token,
    invalidate_token,
    refresh_token,
    get_login_url,
    is_authenticated,
)
from .greeks_calculator import GreeksCalculator
from .data_manager import (
    CentralizedDataManager,
    get_data_manager,
    get_lot_size,
    get_stock_universe,
    FNO_LOT_SIZES,
    NIFTY_50,
    TOP_10_LIQUID,
)
from .backtest_db import (
    BacktestDatabase,
    get_backtest_db,
)
from .metrics_calculator import (
    MetricsCalculator,
    calculate_daily_returns,
    calculate_cumulative_returns,
)
from .covered_call_service import (
    CoveredCallEngine,
    BacktestConfig,
    StrikeMethod,
    ExitStrategy,
    run_backtest,
)
from .iv_percentile import (
    IVPercentileService,
    IVMetrics,
    get_iv_service,
    get_adaptive_delta,
    get_iv_regime_color,
    get_regime_description,
)
from .holdings_service import (
    get_holdings,
    get_fundamentals,
    get_historical_prices,
    get_portfolio_summary,
    get_trading_data,
    format_currency,
    STOCK_SECTORS,
    STOCK_DESCRIPTIONS,
    STOCK_LOGOS,
    STOCK_NAMES,
    INDUSTRY_PE,
)
from .claude_chat_service import (
    chat_sync as portfolio_chat,
    clear_history as clear_chat_history,
    get_suggested_questions,
)

__all__ = [
    # Kite Service
    "get_kite",
    "get_kite_with_refresh",
    "get_access_token",
    "save_access_token",
    "invalidate_token",
    "refresh_token",
    "get_login_url",
    "is_authenticated",
    # Greeks Calculator
    "GreeksCalculator",
    # Data Manager
    "CentralizedDataManager",
    "get_data_manager",
    "get_lot_size",
    "get_stock_universe",
    "FNO_LOT_SIZES",
    "NIFTY_50",
    "TOP_10_LIQUID",
    # Backtest Database
    "BacktestDatabase",
    "get_backtest_db",
    # Metrics Calculator
    "MetricsCalculator",
    "calculate_daily_returns",
    "calculate_cumulative_returns",
    # Covered Call Service
    "CoveredCallEngine",
    "BacktestConfig",
    "StrikeMethod",
    "ExitStrategy",
    "run_backtest",
    # IV Percentile Service
    "IVPercentileService",
    "IVMetrics",
    "get_iv_service",
    "get_adaptive_delta",
    "get_iv_regime_color",
    "get_regime_description",
    # Holdings Service
    "get_holdings",
    "get_fundamentals",
    "get_historical_prices",
    "get_portfolio_summary",
    "get_trading_data",
    "format_currency",
    "STOCK_SECTORS",
    "STOCK_DESCRIPTIONS",
    "STOCK_LOGOS",
    "STOCK_NAMES",
    "INDUSTRY_PE",
    # Claude Chat Service
    "portfolio_chat",
    "clear_chat_history",
    "get_suggested_questions",
]
