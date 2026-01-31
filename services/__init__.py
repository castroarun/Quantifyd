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
from .cpr_covered_call_service import (
    CPRCoveredCallEngine,
    CPRBacktestConfig,
    CPRExitReason,
    run_cpr_backtest,
)
from .intraday_data_bridge import (
    IntradayDataBridge,
    get_intraday_bridge,
)
from .cpr_strategy_optimizer import (
    CPRStrategyOptimizer,
    run_quick_optimization,
)
from .nifty500_universe import (
    Nifty500Universe,
    Stock as Nifty500Stock,
    load_nifty500,
    get_nifty500,
    reload_nifty500,
    cache_instruments,
    get_instrument_token,
    get_instrument_tokens_bulk,
    check_data_coverage,
)
from .fundamental_data_service import (
    FundamentalCriteria,
    QualityScore,
    assess_quality,
    assess_quality_batch,
    rank_quality_stocks,
    clear_cache as clear_fundamentals_cache,
)
from .momentum_filter import (
    MomentumResult,
    MomentumScreenResult,
    calculate_momentum,
    screen_momentum,
    screen_momentum_fast,
)
from .consolidation_breakout import (
    ConsolidationZone,
    BreakoutSignal,
    ConsolidationScreenResult,
    detect_consolidation,
    detect_breakout,
    screen_consolidation_breakout,
)
from .mq_portfolio import (
    MQBacktestConfig,
    Position as MQPosition,
    Trade as MQTrade,
    TopupRecord,
    Portfolio as MQPortfolio,
    TransactionCostModel,
    ExitReason as MQExitReason,
)
from .mq_backtest_engine import (
    MQBacktestEngine,
    BacktestResult as MQBacktestResult,
    run_mq_backtest,
)
from .mq_agent_db import (
    MQAgentDB,
    get_agent_db,
)
from .mq_agent_reports import (
    MarketRegime,
    MonitoringSignal,
    ScreeningReport,
    MonitoringReport,
    RebalanceReport,
    BacktestReportData,
)
from .mq_screening_agent import ScreeningAgent
from .mq_monitoring_agent import MonitoringAgent
from .mq_rebalance_agent import RebalanceAgent
from .mq_backtest_agent import BacktestAgent
from .mq_reporting_agent import ReportingAgent

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
    # CPR Covered Call Service
    "CPRCoveredCallEngine",
    "CPRBacktestConfig",
    "CPRExitReason",
    "run_cpr_backtest",
    # Intraday Data Bridge
    "IntradayDataBridge",
    "get_intraday_bridge",
    # CPR Strategy Optimizer
    "CPRStrategyOptimizer",
    "run_quick_optimization",
    # Nifty 500 Universe
    "Nifty500Universe",
    "Nifty500Stock",
    "load_nifty500",
    "get_nifty500",
    "reload_nifty500",
    "cache_instruments",
    "get_instrument_token",
    "get_instrument_tokens_bulk",
    "check_data_coverage",
    # Fundamental Data Service
    "FundamentalCriteria",
    "QualityScore",
    "assess_quality",
    "assess_quality_batch",
    "rank_quality_stocks",
    "clear_fundamentals_cache",
    # Momentum Filter
    "MomentumResult",
    "MomentumScreenResult",
    "calculate_momentum",
    "screen_momentum",
    "screen_momentum_fast",
    # Consolidation & Breakout
    "ConsolidationZone",
    "BreakoutSignal",
    "ConsolidationScreenResult",
    "detect_consolidation",
    "detect_breakout",
    "screen_consolidation_breakout",
    # MQ Portfolio Management
    "MQBacktestConfig",
    "MQPosition",
    "MQTrade",
    "TopupRecord",
    "MQPortfolio",
    "TransactionCostModel",
    "MQExitReason",
    # MQ Backtest Engine
    "MQBacktestEngine",
    "MQBacktestResult",
    "run_mq_backtest",
    # MQ Agent System
    "MQAgentDB",
    "get_agent_db",
    "MarketRegime",
    "MonitoringSignal",
    "ScreeningReport",
    "MonitoringReport",
    "RebalanceReport",
    "BacktestReportData",
    "ScreeningAgent",
    "MonitoringAgent",
    "RebalanceAgent",
    "BacktestAgent",
    "ReportingAgent",
]
