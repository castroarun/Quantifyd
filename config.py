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
    'enabled': True,
    'paper_trading_mode': True,
    'live_trading_enabled': False,
    'max_daily_orders': 5,
    'max_daily_loss_pct': 3.0,
}

# Maruthi Always-On Strategy Defaults
MARUTHI_DEFAULTS = {
    # Instrument
    'symbol': 'MARUTI',
    'exchange': 'NSE',
    'exchange_fo': 'NFO',
    'lot_size': 50,   # MARUTI F&O lot size (verified from Kite instruments API)
    'strike_interval': 100,  # Strike gap for MARUTI options

    # SuperTrend Settings
    'master_atr_period': 7,
    'master_multiplier': 5.0,  # Master ST(7,5)
    'child_atr_period': 7,
    'child_multiplier': 2.0,   # Child ST(7,2)

    # Timeframe
    'candle_interval': '30minute',
    'candle_minutes': 30,

    # Position Limits
    'max_futures_lots': 5,
    'lots_per_signal': 1,
    'capital': 1_500_000,  # 15L max

    # Trailing Hard Stop Loss — follows master ST, only moves in favorable direction
    # Buffer = hard_sl_atr_mult × ATR(master_period). 1.0x ATR ≈ 75-125 pts on 30-min MARUTI
    'hard_sl_atr_mult': 1.0,  # ATR multiplier for SL distance from master ST
    'hard_sl_buffer': 0,      # Fixed fallback (0 = use ATR-based, >0 = override with fixed points)

    # Protective Options — strike = spot ± (protective_otm_strikes × strike_interval)
    # 5 strikes × 100 = 500 pts OTM. E.g., spot 12500 BEAR → buy 13000CE
    'protective_otm_strikes': 5,  # Number of strikes OTM for protective options

    # Contract Management
    'min_expiry_days_new': 4,    # Don't open options with ≤4 days to expiry
    'roll_expiry_days': 4,       # Roll existing options at 4 days to expiry
    'futures_roll_day_last': True,  # Roll futures on last day of expiry (first half)

    # Option Strike Selection
    'option_otm_strikes': 1,  # 1 strike OTM for short options

    # Safety
    'enabled': False,              # DISABLED — algo has bugs, manual trading only
    'paper_trading_mode': True,    # Force paper mode as safety net
    'live_trading_enabled': False,
    'auto_start_ticker': False,    # Don't auto-start WebSocket ticker
    'max_daily_orders': 20,
    'max_daily_loss_pct': 5.0,

    # TOTP
    'totp_secret': '',  # Set via env var KITE_TOTP_SECRET
    'kite_user_id': '',  # Set via env var KITE_USER_ID
    'kite_password': '',  # Set via env var KITE_PASSWORD
}

# BNF Squeeze & Fire Strategy Defaults
BNF_DEFAULTS = {
    # Instrument
    'symbol': 'BANKNIFTY',
    'exchange': 'NSE',
    'exchange_fo': 'NFO',
    'lot_size': 15,
    'strike_interval': 100,

    # BB Squeeze-Fire Indicators
    'bb_period': 10,
    'atr_period': 14,
    'sma_period': 20,
    'squeeze_min_bars': 3,
    'trend_atr_min': 0.5,

    # Squeeze Mode (Non-Directional Strangles)
    'squeeze_strike_atr': 1.5,     # Strike distance in ATR (each side)
    'squeeze_hold_bars': 10,       # Max hold in daily bars
    'squeeze_max_loss_rupees': 30000,
    'squeeze_lots': 5,
    'max_squeeze_positions': 2,

    # Fire Mode (Directional Naked Sell)
    'fire_strike_atr': 0.5,       # OTM distance in ATR
    'fire_hold_bars': 7,          # Max hold in daily bars
    'fire_sl_mult': 3.0,          # Exit if option val >= 3x premium
    'fire_max_loss_rupees': 20000,
    'fire_lots': 5,
    'max_fire_positions': 1,

    # Capital
    'capital': 10_00_000,          # 10L

    # Safety
    'enabled': True,
    'paper_trading_mode': True,
    'live_trading_enabled': False,
    'max_daily_orders': 10,
}

# NAS — Nifty ATR Strangle (Intraday Options Selling) Defaults
NAS_DEFAULTS = {
    # Instrument
    'symbol': 'NIFTY',
    'exchange': 'NSE',
    'exchange_fo': 'NFO',
    'lot_size': 75,             # Nifty F&O lot size
    'strike_interval': 50,      # Nifty options strike gap

    # ATR Squeeze Detection (5-min candles)
    'atr_period': 14,           # 14-bar ATR on 5-min = ~70 min lookback
    'atr_ma_period': 50,        # 50-bar SMA of ATR = ~250 min (full day avg)
    'min_squeeze_bars': 1,      # ATR must be below MA for 1 completed bar
    'candle_interval': '5minute',
    'candle_minutes': 5,

    # Strike Selection — premium-based (find strike with target premium)
    'target_entry_premium': 20.0,  # Entry: find strikes with ~Rs 20 premium each leg
    'min_leg_premium': 5.0,        # No trades at or below Rs 5 premium
    'max_leg_premium': 24.0,       # No trades above Rs 24 premium
    'min_otm_distance': 100,       # Min 100 pts OTM from spot
    # ATR fallback (used only when live quotes fail)
    'strike_distance_atr': 1.5,    # OTM distance = 1.5x daily ATR each side
    'daily_atr_period': 14,        # ATR period on daily bars for strike calc

    # Position Sizing
    'lots_per_leg': 10,         # 10 lots per leg = 750 qty
    'max_strangles': 1,         # Only 1 strangle at a time

    # Adjustment Rules
    'premium_double_trigger': 2.0,   # Cross-leg imbalance trigger (leg1 >= 2x leg2)
    'adj_min_premium': 4.0,          # Target premium floor — below this, close both
    'adj_max_premium': 24.0,         # Target premium ceiling — above this, flip direction
    'max_adjustments_per_leg': 999,  # No limit
    'max_adjustments_total': 999,    # No limit
    'adjustment_wait_bars': 1,       # Wait 1 bar (5 min) before re-entering

    # Exit Rules
    'profit_target_pct': 70.0,  # Close all when 70% of premium captured
    'eod_squareoff_time': '15:15',  # Mandatory EOD squareoff
    'time_exit': '14:45',       # Close if still open at 2:45 PM
    'entry_start_time': '09:30',    # No entries before 9:30
    'entry_end_time': '14:30',      # No entries after 2:30 PM

    # Filters
    'skip_expiry_day': False,   # Trade on expiry days too — intraday with SL protection
    'max_vix': None,            # VIX filter disabled — intraday with SL protection
    'min_combined_premium': 0,  # Disabled — BS underestimates OTM premiums heavily
    'max_spot_move_pct': 0.5,   # Skip if Nifty already moved > 0.5% from open

    # Capital & Risk
    'capital': 300_000,         # 3L margin for short strangle
    'max_daily_loss': 15_000,   # Daily loss circuit breaker
    'max_daily_orders': 20,     # Order limit per day

    # Safety
    'enabled': True,
    'paper_trading_mode': True,
    'live_trading_enabled': False,
}

# NAS ATM — Nifty ATR Strangle (ATM, SL-based, cascading re-entry)
NAS_ATM_DEFAULTS = {
    # Instrument (same as NAS OTM)
    'symbol': 'NIFTY',
    'exchange': 'NSE',
    'exchange_fo': 'NFO',
    'lot_size': 75,
    'strike_interval': 50,

    # ATR Squeeze Detection (shared with NAS OTM — same ticker)
    'atr_period': 14,
    'atr_ma_period': 50,
    'min_squeeze_bars': 1,
    'candle_interval': '5minute',
    'candle_minutes': 5,

    # Strike Selection — ATM
    'strike_mode': 'ATM',           # Always enter at-the-money

    # Position Sizing
    'lots_per_leg': 5,              # 5 lots per leg = 375 qty
    'max_strangles': 1,             # Only 1 active strangle at a time
    'max_reentries': 5,             # Max 5 SL re-entry cycles per day

    # Stop Loss — per-leg percentage
    'leg_sl_pct': 0.30,             # 30% of entry premium (sell@100 → SL@130)

    # On SL hit: trail surviving leg to cost, enter new ATM
    'trail_to_cost_on_sl': True,
    're_enter_on_sl': True,

    # Exit Rules
    'eod_squareoff_time': '15:15',  # Mandatory EOD squareoff
    'time_exit': '15:15',           # Close all by 3:15 PM
    'entry_start_time': '09:30',
    'entry_end_time': '14:50',      # No new entries after 2:50 PM

    # Filters
    'skip_expiry_day': False,

    # Capital & Risk
    'capital': 500_000,
    'max_daily_loss': 25_000,
    'max_daily_orders': 40,

    # Safety
    'enabled': True,
    'paper_trading_mode': True,
    'live_trading_enabled': False,
}

NAS_ATM2_DEFAULTS = {
    **NAS_ATM_DEFAULTS,
    # On SL hit: close BOTH legs, then immediately re-enter new ATM straddle
    'trail_to_cost_on_sl': False,
    're_enter_on_sl': True,
    'exit_both_on_sl': True,
}

NAS_ATM4_DEFAULTS = {
    **NAS_ATM_DEFAULTS,
    'max_rolls': 1,              # Only 1 roll allowed per strangle
    'trail_to_cost_on_sl': False,
    're_enter_on_sl': False,
}

# --- 916 Variants: Same strategies, mandatory 9:16 AM entry (no squeeze wait) ---

NAS_916_OTM_DEFAULTS = {
    **NAS_DEFAULTS,
    'entry_start_time': '09:16',
    'skip_squeeze': True,           # Enter at 9:16 regardless of ATR squeeze
}

NAS_916_ATM_DEFAULTS = {
    **NAS_ATM_DEFAULTS,
    'entry_start_time': '09:16',
}

NAS_916_ATM2_DEFAULTS = {
    **NAS_ATM2_DEFAULTS,
    'entry_start_time': '09:16',
}

NAS_916_ATM4_DEFAULTS = {
    **NAS_ATM4_DEFAULTS,
    'entry_start_time': '09:16',
}

# --- ORB: Opening Range Breakout (Cash Equity Intraday) ---

ORB_DEFAULTS = {
    # Universe — 15 high-beta F&O stocks
    'universe': [
        'ADANIENT', 'TATASTEEL', 'BEL', 'VEDL', 'BPCL', 'M&M', 'BAJFINANCE',
        'TRENT', 'HAL', 'IRCTC', 'GRASIM', 'GODREJPROP', 'RELIANCE', 'AXISBANK', 'APOLLOHOSP',
    ],

    # System Control
    'enabled': True,
    'live_trading_enabled': True,      # Direct live (MIS), no paper mode

    # Capital & Position Sizing
    # 15 stocks, avg 2.9 trades/day, P75=5. Size for 5 concurrent trades.
    # Per-trade = capital / max_concurrent = 100K/5 = Rs 20,000
    # Min margin = 1.2x per-trade = Rs 24,000
    'capital': 100_000,                # Total fund — change this to scale up
    'max_concurrent_trades': 5,        # P75 of daily trades for 15-stock universe
    'margin_buffer_multiplier': 1.2,   # Need 1.2x per-trade alloc as available margin

    # Opening Range
    'or_minutes': 15,                  # 09:15 - 09:30

    # Entry Rules
    'last_entry_time': '14:00',        # No entries after 2 PM
    'eod_exit_time': '15:18',          # Hard EOD backup. V9t_lock50 (Calmar 676): at 14:30 lock 50% profit + ride to 15:18 hard squareoff. See docs/ORB-VARIANTS-FINDINGS.md
    'max_trades_per_day': 1,           # Per stock per day

    # Exit Rules
    'sl_type': 'or_opposite',          # SL at OR opposite boundary
    'target_type': 'r_multiple',
    'r_multiple': 1.5,                 # 1.5x risk:reward

    # Filters
    'use_vwap_filter': True,
    'use_rsi_filter': True,
    'rsi_long_threshold': 60,
    'rsi_short_threshold': 40,
    'use_cpr_dir_filter': True,
    'use_cpr_width_filter': True,
    'cpr_width_threshold_pct': 0.65,   # 250-day sweep 2026-04-21: 0.65 beats 0.5 on Calmar 263 vs 230 (+161 trades, DD flat). See docs/ORB-VARIANTS-FINDINGS.md
    'use_gap_filter': True,
    'gap_long_block_pct': 1.0,         # Block longs only on large gap-ups (>1%)

    # Direction control
    'allow_longs': True,
    'allow_shorts': True,

    # Risk Management
    'daily_loss_limit': 3_000,         # Rs 3K daily loss cap (3% of capital)

    # Notifications
    'email_enabled': True,
    'email_from': 'arun.castromin@gmail.com',
    'email_to': 'arun.castromin@gmail.com',
    'email_app_password': os.getenv('GMAIL_APP_PASSWORD', ''),
    'smtp_host': 'smtp.gmail.com',
    'smtp_port': 587,
    'whatsapp_enabled': True,
    'twilio_account_sid': os.getenv('TWILIO_ACCOUNT_SID', ''),
    'twilio_auth_token': os.getenv('TWILIO_AUTH_TOKEN', ''),
    'twilio_whatsapp_from': os.getenv('TWILIO_FROM_WHATSAPP', 'whatsapp:+14155238886'),
    'twilio_whatsapp_to': os.getenv('TWILIO_TO_WHATSAPP', ''),
    'notify_on_entry': True,
    'notify_on_exit': True,
    'notify_on_sl': True,
    'notify_on_target': True,
    'notify_on_blocked': False,        # Too many, keep quiet
    'notify_on_risk': True,
    'notify_on_system': True,
    'notify_eod_report': True,
    'notify_midmorning_status': True,   # 10:30 mid-morning status (email + WhatsApp)

    # Catchup (recovery-only) config
    # If LTP has run more than this many R past the first breakout candle close,
    # skip — the move has extended too far and R:R is now unfavourable.
    # R = the initial OR-based risk (breakout_close - or_low for long).
    'catchup_max_slippage_r': 0.5,      # default: half the intended risk
}

# Nifty 500 Universe
NIFTY500_CSV = DATA_DIR.parent / 'data' / 'nifty500_list.csv'
