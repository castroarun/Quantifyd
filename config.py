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

# Collar Strategy Defaults (KC6 signal on F&O stocks, 3-leg options collar)
# Long Fut + Long OTM Put (protection) + Short OTM Call (premium cap)
COLLAR_DEFAULTS = {
    # Signal source — reuses KC6 config for indicator params
    # (KC6 entry: close < KC6_lower AND close > SMA200)

    # Strike selection (% OTM from entry spot)
    'put_otm_pct': 5.0,     # Put strike at spot * (1 - put_otm_pct/100), rounded DOWN to interval
    'call_otm_pct': 5.0,    # Call strike at spot * (1 + call_otm_pct/100), rounded UP to interval

    # Exit rules (mirror KC6 on underlying spot)
    'sl_pct': 5.0,          # underlying falls 5% from entry -> force exit all 3 legs
    'tp_pct': 15.0,         # underlying rises 15% from entry -> force exit all 3 legs
    'max_hold_days': 15,
    'min_expiry_days': 7,   # Roll to next month if <7 days to current-month expiry
    'expiry_exit_days': 3,  # Force exit if <3 days to expiry (avoid gamma risk)

    # Universe: Nifty 500 ∩ F&O universe (intersection computed at scan time)

    # Position sizing (1 lot per collar)
    'max_positions': 5,

    # Black-Scholes paper-pricing assumptions
    'iv_assumed': 0.25,     # Flat 25% IV for both entry and exit pricing
    'risk_free_rate': 0.065,

    # Crash filter (same KC6 threshold on universe ATR ratio)
    'atr_ratio_threshold': 1.3,

    # Safety (paper only)
    'enabled': True,
    'paper_trading_mode': True,
    'live_trading_enabled': False,
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
    'premium_double_trigger': 2.0,       # Same-leg trigger: losing leg >= 2.0x its OWN entry premium
    'premium_half_trigger': 0.5,         # Same-leg trigger: winning leg <= 0.5x its own entry premium
    'premium_cross_leg_trigger': 2.5,    # Cross-leg trigger: losing leg >= 2.5x the OTHER leg's CURRENT premium
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

    # Watch universe — top 10 candidates from research/20 universe-optimization
    # walk-forward (train 2024-2025 / test 2025-2026, all passed PF>=1 both
    # periods + Sharpe>=0.5). DISPLAY-ONLY for now — scanner skips these,
    # frontend renders them greyed out. Promote to 'universe' once the live
    # 15 stabilize and we're ready to expand. Order = test-Sharpe descending.
    'watch_universe': [
        'WIPRO', 'TCS', 'PNB', 'DLF', 'GAIL', 'HINDUNILVR', 'PAYTM',
        'FEDERALBNK', 'COLPAL', 'NTPC',
    ],

    # System Control
    'enabled': True,
    'live_trading_enabled': True,      # Direct live (MIS), no paper mode

    # Capital & Position Sizing
    # 15 stocks, avg 2.9 trades/day, P75=5. Size for 5 concurrent trades.
    # Per-trade = capital / max_concurrent = 100K/5 = Rs 20,000
    # Min margin = 1.2x per-trade = Rs 24,000
    'capital': 300_000,                # Total DEPOSIT allocated to ORB (3L, scaled +50% from 2L on 2026-04-23)
    'mis_leverage': 5,                 # Zerodha MIS leverage on Nifty 500 cash ~5x
    'max_concurrent_trades': 5,        # P75 of daily trades for 15-stock universe
    'margin_buffer_multiplier': 1.2,   # Need 1.2x per-trade alloc as available margin

    # Risk-based sizing (overrides the old "qty = floor(alloc / entry)" rule).
    # qty = floor((capital × risk_per_trade_pct) / |entry - SL|), capped by
    # max_notional_per_trade so a very tight OR doesn't size silly on a
    # high-priced stock. SL stays OR-opposite — only the sizing changes.
    'use_risk_based_sizing': True,
    'risk_per_trade_pct': 0.010,       # 1.0% of capital = Rs 3,000 per trade at 3L
    'max_notional_per_trade': 300_000, # Rs 3L ceiling per position (matches capital × lev / max_concurrent)

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

    # Staleness guards — reject entries on stale signals that can otherwise
    # fire post-restart (service comes back mid-session and walks all
    # post-OR candles, taking the latest valid breakout even if it is
    # hours old). See 2026-04-24 VEDL/TRENT incident and
    # docs/MONDAY-WORK-PLAN.md.
    'signal_age_max_mins': 15,         # breakout candle cannot be older than this
    'signal_drift_max_pct': 0.005,     # 0.5% max drift between breakout close and current close
    'entry_end_time': '14:00',         # no fresh entries after this IST
    'post_restart_cooldown_mins': 5,   # log-only pass during the first N min after engine start

    # Book-level drawdown cut — two-tier aggregate stop on unrealized P&L.
    # Computed in monitor_positions() every 30s. One-shot per day each.
    'book_drawdown_soft_inr': -7500,   # soft: halve qty on every losing position
    'book_drawdown_hard_inr': -15000,  # hard: flatten the entire book
    'enforce_book_drawdown': True,

    # Tail hedge — buy NIFTY OTM option when book is heavily one-sided.
    # Triggered once/day during eval window; held till `hedge_exit_time`.
    'hedge_enabled': True,
    'hedge_paper_mode': True,          # log-only for v1; flip to False after shadow week
    'hedge_skew_threshold': 0.70,      # |short-long|/total notional ≥ this
    'hedge_min_positions': 10,         # active ORB positions ≥ this
    'hedge_otm_pct': 0.015,            # 1.5% OTM from spot
    'hedge_lots': 1,
    'hedge_eval_start': '10:00',
    'hedge_eval_end': '14:00',
    'hedge_exit_time': '15:15',        # square off 5 min before ORB EOD squareoff

    # Risk Management
    # Expressed as % of `capital`. Computed at runtime so scaling the ORB
    # capital auto-scales the loss cap. daily_loss_limit (Rs) overrides
    # the pct if set — kept for back-compat, leave None for pct-based.
    'daily_loss_limit_pct': 0.03,      # 3% of capital (Rs 3K at 1L capital)
    'daily_loss_limit': None,          # Rs override (None -> use pct above)
    # Panic threshold: if realized + unrealized loss exceeds this multiple
    # of daily_loss_limit, force-close ALL open positions (not just block
    # new entries). Two-tier: 1.0x blocks new entries, 1.5x closes open.
    'daily_loss_panic_multiplier': 1.5,
    # Enforcement toggle — when False the gate still computes numbers for
    # the UI but never blocks entries or force-closes. Useful while scaling
    # up position size via leverage and the cap hasn't been re-tuned yet.
    'enforce_daily_loss_cap': False,

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

    # Exchange-side SL-M (hard stop at Kite). When True, every ORB entry
    # is immediately followed by an SL-M order on the exchange, surviving
    # service restarts / crashes. Monitor still polls for target + tightens
    # the SL-M via modify_order on 14:30 trail.
    'use_exchange_sl_m': True,
}

# Nifty 500 Universe
NIFTY500_CSV = DATA_DIR.parent / 'data' / 'nifty500_list.csv'


# =============================================================================
# Nifty ORB Strangle (Phase 3 paper-trade build, 8 variants)
# =============================================================================
# Cross-leg credit-spread strangle on NIFTY index options driven by an ORB
# breakout signal on the 5-min NIFTY index. PAPER ONLY. Each variant runs
# independently — same structural strategy, different OR window / RSI filter.
#
# Strategy in plain English:
#   1. Compute the OR (open-range high/low) over the variant's window.
#   2. After the OR closes, watch 5-min closes for a break (close > OR_high
#      = LONG, close < OR_low = SHORT). RSI on 5-min must confirm direction
#      (variant-configurable thresholds; V6 takes any break).
#   3. Skip the day if OR_width > 0.67% of spot (Q4 noise).
#   4. On entry, sell two short legs (PE + CE) with delta targets that skew
#      toward the break direction. Buy nothing — we're collecting credit on
#      both sides; the loss leg is wider than the win leg.
#   5. Exit on (a) underlying breaching its OR-anchored SL, or
#      (b) 15:25 EOD square-off, whichever comes first.

STRANGLE_DEFAULTS = {
    # Index + lot
    'symbol': 'NIFTY',
    'exchange_index': 'NSE',
    'exchange_options': 'NFO',
    'lot_size': 65,                    # Nifty options lot size (65 from 2026)
    'strike_interval': 50,             # Nifty options strike gap

    # Polling / scheduling
    'mtm_poll_seconds': 60,            # Master tick interval
    'eod_squareoff_time': '15:25',     # Force-close all open variants
    'no_entry_after': '14:00',         # No fresh entries after this IST
    'session_open': '09:15',
    'session_close': '15:30',

    # SL math
    'sl_or_width_multiplier': 1.0,     # SL = OR_high (short break) / OR_low (long break) at multiplier=1.0

    # IV bounds for back-implied vol when chain has no IV field
    'iv_floor': 0.12,
    'iv_cap':   0.25,
    'iv_default': 0.18,                # used when chain is unavailable
    'risk_free_rate': 0.065,

    # Costs (per leg per side)
    'slippage_per_leg_per_side': 1.0,  # Rs 1 per option per fill
    'brokerage_round_trip': 80.0,      # Rs 80 round-trip (4 fills total)
    'stt_pct_on_credit': 0.0005,       # 0.05% of total notional credit

    # Strike scanning bounds (% of spot)
    'strike_scan_pct': 0.10,           # ±10% around spot

    # Safety
    'enabled': True,
    'paper_trading_mode': True,
    'live_trading_enabled': False,
}

# Per-variant configs. ID is the stable key used by the dashboard, DB and routes.
STRANGLE_VARIANTS = [
    {
        'id': 'or60-std',
        'name': 'OR60 Standard',
        'or_min': 60,
        'rsi_lo_long': 60.0, 'rsi_hi_short': 40.0,
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': False, 'calm_threshold_pct': 0.40,
        'apply_cpr_against_filter': False,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 84,
        'backtest_wins_per_year': 150,
        'backtest_trades_per_year': 180,
    },
    {
        'id': 'or45-std',
        'name': 'OR45 Standard',
        'or_min': 45,
        'rsi_lo_long': 60.0, 'rsi_hi_short': 40.0,
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': False, 'calm_threshold_pct': 0.40,
        'apply_cpr_against_filter': False,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 82,
        'backtest_wins_per_year': 165,
        'backtest_trades_per_year': 200,
    },
    {
        'id': 'or30-std',
        'name': 'OR30 Standard',
        'or_min': 30,
        'rsi_lo_long': 60.0, 'rsi_hi_short': 40.0,
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': False, 'calm_threshold_pct': 0.40,
        'apply_cpr_against_filter': False,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 79,
        'backtest_wins_per_year': 175,
        'backtest_trades_per_year': 220,
    },
    {
        'id': 'or15-std',
        'name': 'OR15 Standard',
        'or_min': 15,
        'rsi_lo_long': 60.0, 'rsi_hi_short': 40.0,
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': False, 'calm_threshold_pct': 0.40,
        'apply_cpr_against_filter': False,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 75,
        'backtest_wins_per_year': 180,
        'backtest_trades_per_year': 240,
    },
    {
        'id': 'or5-std',
        'name': 'OR5 Standard',
        'or_min': 5,
        'rsi_lo_long': 60.0, 'rsi_hi_short': 40.0,
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': False, 'calm_threshold_pct': 0.40,
        'apply_cpr_against_filter': False,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 70,
        'backtest_wins_per_year': 175,
        'backtest_trades_per_year': 250,
    },
    {
        'id': 'or60-norsi',
        'name': 'OR60 No-RSI',
        'or_min': 60,
        'rsi_lo_long': None, 'rsi_hi_short': None,    # take any break
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': False, 'calm_threshold_pct': 0.40,
        'apply_cpr_against_filter': False,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 78,
        'backtest_wins_per_year': 175,
        'backtest_trades_per_year': 225,
    },
    {
        'id': 'or60-tight',
        'name': 'OR60 Tight RSI',
        'or_min': 60,
        'rsi_lo_long': 65.0, 'rsi_hi_short': 35.0,
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': False, 'calm_threshold_pct': 0.40,
        'apply_cpr_against_filter': False,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 86,
        'backtest_wins_per_year': 110,
        'backtest_trades_per_year': 130,
    },
    {
        'id': 'or60-calm',
        'name': 'OR60 Calm-Only',
        'or_min': 60,
        'rsi_lo_long': 60.0, 'rsi_hi_short': 40.0,
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': True, 'calm_threshold_pct': 0.40,  # OR60 width < 0.40% of spot
        'apply_cpr_against_filter': False,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 88,
        'backtest_wins_per_year': 90,
        'backtest_trades_per_year': 105,
    },
    {
        # V9: OR60 + CPR-against filter. Hypothesis (validated in
        # nifty_orb_cpr_filter_results.csv): tame breakouts that haven't cleared
        # the CPR zone are friendlier to short-strangle theta capture than fully
        # extended trend breaks. Backtest: 95.5% WR / 24 trades/yr.
        'id': 'or60-cpr-against',
        'name': 'OR60 CPR-Against',
        'or_min': 60,
        'rsi_lo_long': 60.0, 'rsi_hi_short': 40.0,
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': False, 'calm_threshold_pct': 0.40,
        'apply_cpr_against_filter': True,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 96,
        'backtest_wins_per_year': 23,
        'backtest_trades_per_year': 24,
    },
    {
        # V10: OR30 + CPR-against. Earlier entry, more theta runway. Backtest:
        # 93.1% WR / 15 wins/yr.
        'id': 'or30-cpr-against',
        'name': 'OR30 CPR-Against',
        'or_min': 30,
        'rsi_lo_long': 60.0, 'rsi_hi_short': 40.0,
        'apply_q4_filter': True, 'q4_threshold_pct': 0.67,
        'apply_calm_filter': False, 'calm_threshold_pct': 0.40,
        'apply_cpr_against_filter': True,
        'pe_delta_target_long': -0.22, 'ce_delta_target_long': 0.10,
        'pe_delta_target_short': -0.10, 'ce_delta_target_short': 0.22,
        'lot_size': 65,
        'enabled': True,
        'backtest_wr_pct': 93,
        'backtest_wins_per_year': 15,
        'backtest_trades_per_year': 16,
    },
]

# Convenience lookup: id -> variant dict
STRANGLE_VARIANTS_BY_ID = {v['id']: v for v in STRANGLE_VARIANTS}
