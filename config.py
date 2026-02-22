"""
Configuration settings for Covered Calls Backtester
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Data directory: use Railway volume if available, else local backtest_data/
_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
DATA_DIR = Path(_volume_path) / "data" if _volume_path else BASE_DIR / "backtest_data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Database paths
MARKET_DATA_DB = DATA_DIR / "market_data.db"
BACKTEST_RESULTS_DB = DATA_DIR / "backtest_results.db"

# Zerodha API Configuration
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_REDIRECT_URL = os.getenv("KITE_REDIRECT_URL", "http://127.0.0.1:5000/zerodha/callback")
TOKEN_FILE = DATA_DIR / "access_token.json"

# Flask Configuration
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

class Config:
    SECRET_KEY = FLASK_SECRET_KEY
    SESSION_TYPE = "filesystem"
    SESSION_FILE_DIR = str(DATA_DIR / "flask_session")
    SESSION_PERMANENT = False
    TEMPLATES_AUTO_RELOAD = True

# Indian Market Parameters
RISK_FREE_RATE = 0.07  # 7% India benchmark
TRADING_DAYS_PER_YEAR = 252
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# Backtest Configuration
DEFAULT_DTE_MIN = 30
DEFAULT_DTE_MAX = 45
DEFAULT_POSITION_SIZE = 1  # 1 contract = 1 lot

# Strike Selection Methods
STRIKE_METHODS = [
    "DELTA_30",
    "DELTA_40",
    "OTM_2PCT",
    "OTM_5PCT",
    "ATM",
    "ADAPTIVE_DELTA",  # IV Percentile-based dynamic delta selection
    "ATR_BASED",  # ATR-based OTM distance calculation
    "PIVOT_R1",  # Strike at R1 resistance (pivot point)
    "PIVOT_R2",  # Strike at R2 resistance (pivot point)
    "BOLLINGER_UPPER"  # Strike at Upper Bollinger Band (natural resistance)
]

# ATR-based strike selection defaults
DEFAULT_ATR_MULTIPLIER = 1.5  # Strike = Current Price + (ATR x Multiplier)

# Exit Rules
EXIT_RULES = [
    "HOLD_TO_EXPIRY",
    "PROFIT_TARGET",
    "STOP_LOSS",
    "PROFIT_TARGET_AND_STOP_LOSS"
]

# Default Exit Parameters
DEFAULT_PROFIT_TARGET_PCT = 50  # Close when 50% of max profit captured
DEFAULT_STOP_LOSS_MULTIPLE = 2.0  # Close when loss exceeds 2x premium
DEFAULT_CAPITAL = 1000000  # ₹10 lakh default capital

# Stock Lot Sizes (NSE F&O)
LOT_SIZES = {
    "RELIANCE": 250,
    "TCS": 150,
    "HDFCBANK": 550,
    "INFY": 300,
    "ICICIBANK": 700,
    "KOTAKBANK": 400,
    "SBIN": 750,
    "BHARTIARTL": 950,
    "ITC": 1600,
    "AXISBANK": 600,
}

# Supported Timeframes
TIMEFRAMES = ["day", "minute", "5minute", "15minute", "hour"]
DEFAULT_TIMEFRAME = "day"

# CPR (Central Pivot Range) Strategy Defaults
CPR_DEFAULTS = {
    'narrow_cpr_threshold': 0.5,      # Skip CPR narrower than 0.5% of price
    'otm_strike_pct': 5.0,            # Sell calls at 5% OTM
    'dte_min': 30,                    # Minimum days to expiry
    'dte_max': 35,                    # Maximum days to expiry
    'enable_premium_rollout': True,   # Enable premium doubling rollout
    'premium_double_threshold': 2.0,  # Roll when premium doubles (2x)
    'premium_erosion_target': 75.0,   # Exit when 75% premium captured
    'dte_exit_threshold': 10,         # Exit when DTE reaches 10
    'enable_r1_exit': True,           # Exit on R1 breach
    'use_closer_r1': True,            # Use closer of current/previous R1
}

# Momentum + Quality Strategy Defaults
MQ_DEFAULTS = {
    # Portfolio
    'portfolio_size': 30,
    'equity_allocation_pct': 0.80,
    'debt_reserve_pct': 0.20,
    'debt_fund_annual_return': 0.065,
    'max_position_size': 0.10,
    'max_sector_weight': 0.25,
    'max_stocks_per_sector': 6,

    # Momentum Filter
    'ath_proximity_threshold': 0.10,  # Within 10% of 52-week high

    # Fundamental Quality (Non-Financial)
    'min_revenue_growth_3y_cagr': 0.15,
    'require_revenue_positive_each_year': True,
    'max_debt_to_equity': 0.20,
    'min_opm_3y': 0.15,
    'require_opm_no_decline': True,

    # Fundamental Quality (Financial)
    'min_roa': 0.01,
    'min_roe': 0.12,

    # Consolidation & Breakout
    'consolidation_days': 20,
    'consolidation_range_pct': 0.05,
    'breakout_volume_multiplier': 1.5,
    'topup_pct_of_initial': 0.20,
    'topup_cooldown_days': 5,

    # Exit Rules
    'rebalance_ath_drawdown': 0.20,
    'quarterly_decline_threshold': 0.10,
    'hard_stop_loss': 0.30,

    # Backtest
    'start_date': '2023-01-01',
    'end_date': '2025-12-31',
    'initial_capital': 10_000_000,
    'brokerage_pct': 0.0003,
    'stt_pct': 0.001,
    'gst_pct': 0.18,
    'stamp_duty_pct': 0.00015,
    'slippage_pct': 0.001,
    'rebalance_months': [1, 7],

    # Composite Ranking Weights
    'weight_revenue': 0.30,
    'weight_debt': 0.25,
    'weight_opm': 0.25,
    'weight_opm_growth': 0.20,
}

# KC6 Mean Reversion Strategy Defaults
KC6_DEFAULTS = {
    # Keltner Channel
    'kc_ema_period': 6,
    'kc_atr_period': 6,
    'kc_multiplier': 1.3,

    # Trend filter
    'sma_period': 200,

    # Exit rules
    'sl_pct': 5.0,
    'tp_pct': 15.0,
    'max_hold_days': 15,

    # Crash filter (Universe ATR Ratio)
    'atr_ratio_threshold': 1.3,
    'atr_lookback': 14,
    'atr_avg_window': 50,

    # Position sizing
    'max_positions': 5,
    'position_size_pct': 0.10,

    # Safety
    'paper_trading_mode': True,
    'live_trading_enabled': False,
    'max_daily_orders': 5,
    'max_daily_loss_pct': 3.0,
}

# Nifty 500 Universe
NIFTY500_CSV = DATA_DIR.parent / 'data' / 'nifty500_list.csv'
