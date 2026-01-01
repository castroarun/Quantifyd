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
DATA_DIR = BASE_DIR / "backtest_data"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

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
