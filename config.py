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
    'enabled': False,
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
    'enabled': False,
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
    'enabled': False,
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
    'lot_size': 65,             # Nifty F&O lot size (DEAD FIELD — executors import LOT_SIZE=65 from services/nas_scanner.py; kept here for documentation/dashboard display only)
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
    'lots_per_leg': 2,          # 2026-05-23: dropped 4→2 ahead of live flip (130 qty/leg, ~Rs 4L margin per OTM strangle). Inherited by NAS_916_OTM_DEFAULTS via spread.
    'max_strangles': 1,         # Only 1 strangle at a time

    # Day-of-week filter — skip OTM trading on Wed/Thu (expiry-week trend days
    # are unkind to range/premium-decay strategies). Mon=0, Tue=1, Wed=2, Thu=3, Fri=4.
    'skip_weekdays': (),  # 2026-06-03: no day fully skipped; non-live days run as PAPER (see live_weekdays)
    # DTE entry gate (research/51, 2026-06-02): only open new positions at <= this
    # many days to weekly expiry. Replay showed the edge is at 1 DTE; 4+ DTE bleeds.
    # 1 => trade only 0 & 1 DTE. Inherited by 916_OTM via spread.
    # 2026-06-03: gate OFF operationally (live = skip_weekdays-driven); kept as a
    # backtest-study question, not a live gate.
    'max_dte_at_entry': None,
    # Live/paper by weekday: REAL Kite orders only Mon/Tue/Fri; every other day
    # runs as PAPER (signals + DB + P&L, no real orders). Mon=0 Tue=1 Wed=2 Thu=3 Fri=4.
    'live_weekdays': (0, 1, 4),

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
    # 2026-05-29: raised 20→40 after the morning cross-leg roll-thrash burned
    # the 20-order budget by 11:38 and gagged Sq-OTM (couldn't rebalance). Safe
    # now that the cross-leg roll carries a 90s cooldown (nas_ticker
    # _check_premium_tick) rate-limiting adjustments — the cap is the runaway
    # backstop, the cooldown is the rate limit. Inherited by 916-OTM via spread.
    'max_daily_orders': 40,     # Order limit per day (OTM family)
    'adj_cooldown_sec': 90,     # Min secs between cross-leg rolls (anti-thrash)

    # Safety
    # 2026-05-25: flipped back to PAPER after today's go-live. User on
    # road-trip 26-28 May; system runs in paper to keep DB + EOD report
    # populated without sending live orders to Kite. Restored to LIVE
    # when user explicitly flips via /api/nas/master-mode on return.
    # Inherited by NAS_916_OTM_DEFAULTS via spread.
    'enabled': True,
    'paper_trading_mode': True,
    'live_trading_enabled': False,
}

# NAS ATM — Nifty ATR Strangle (ATM, SL-based, cascading re-entry)
NAS_ATM_DEFAULTS = {
    # DTE entry gate (research/51, 2026-06-02): only open new positions at <= this
    # many days to weekly expiry (1 => trade only 0 & 1 DTE — where the edge is).
    # Inherited by ATM2/ATM4 + all 916_ATM* via spread.
    # 2026-06-03: gate OFF operationally (see NAS_DEFAULTS note).
    'max_dte_at_entry': None,
    # Live/paper by weekday: REAL Kite orders only Mon/Tue/Fri; else PAPER.
    'live_weekdays': (0, 1, 4),
    # Instrument (same as NAS OTM)
    'symbol': 'NIFTY',
    'exchange': 'NSE',
    'exchange_fo': 'NFO',
    'lot_size': 65,             # DEAD FIELD — see note on NAS_DEFAULTS.lot_size. Inherited by NAS_ATM2/ATM4 + all 916_ATM* variants via spread.
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
    'lots_per_leg': 1,              # GO-LIVE 2026-05-01: dropped 5→1 (75 qty)
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
    # 2026-05-25: flipped back to PAPER after today's go-live. Inherited
    # by NAS_ATM2/ATM4 + all 916_ATM* via spread (covers 6 of 8 variants).
    # User restored LIVE explicitly via dashboard when ready.
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
    # Skip Wed/Thu — cascading 5-re-entry geometry is brittle on trending
    # expiry-week days. Basic ATM (no cascade) stays enabled all weekdays.
    'skip_weekdays': (),  # 2026-06-03: no day fully skipped; non-live days run as PAPER (see live_weekdays)
}

NAS_ATM4_DEFAULTS = {
    **NAS_ATM_DEFAULTS,
    'max_rolls': 1,              # Only 1 roll allowed per strangle
    'trail_to_cost_on_sl': False,
    're_enter_on_sl': False,
    # Same Wed/Thu skip as ATM 2.0 — roll-to-match also struggles on
    # strong directional days. Inherited by NAS_916_ATM4_DEFAULTS via spread.
    'skip_weekdays': (),  # 2026-06-03: no day fully skipped; non-live days run as PAPER (see live_weekdays)
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

# 2026-06-05: NAS enabled flags LOCKED (post-spread). Squeeze OFF — over-margin: all 8
# variants exceed ~Rs52L; squeeze entries rejected (insufficient funds) and re-spammed
# every 5-min candle. 9:16 family ON — it holds the day's live positions. Re-enable the
# squeeze when the book is funded / re-sized.
NAS_DEFAULTS['enabled'] = True
NAS_ATM_DEFAULTS['enabled'] = True
NAS_ATM2_DEFAULTS['enabled'] = True
NAS_ATM4_DEFAULTS['enabled'] = True
NAS_916_OTM_DEFAULTS['enabled'] = True
NAS_916_ATM_DEFAULTS['enabled'] = True
NAS_916_ATM2_DEFAULTS['enabled'] = True
NAS_916_ATM4_DEFAULTS['enabled'] = True
NAS_DEFAULTS['force_paper'] = True
# 2026-06-07 (Sun) LIVE flip for Mon 06-08 (user-directed): 3 squeeze ATMs + 3 916 ATMs
# go LIVE (real money) under master 'live'. NAS base + 916_OTM kept PAPER (force_paper). 916_OTM
# kept ENABLED but PAPER via force_paper (honored in _apply_nas_master_mode). 1-lot ATM
# each (~Rs9L initial margin, fits Rs38L collateral-incl avail). live_weekdays=(Mon,Tue,Fri).
NAS_916_OTM_DEFAULTS['force_paper'] = True
# 2026-06-08 churn-breaker (research/60): re-entry cooldown on every live ATM variant.
for _nas_cd in (NAS_ATM_DEFAULTS, NAS_ATM2_DEFAULTS, NAS_ATM4_DEFAULTS,
                NAS_916_ATM_DEFAULTS, NAS_916_ATM2_DEFAULTS, NAS_916_ATM4_DEFAULTS):
    _nas_cd['reentry_cooldown_min'] = 15

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

    # System Control — three-way mode (Off / Paper / Live)
    # Off:   enabled=False
    # Paper: enabled=True, paper_trading_mode=True, live_trading_enabled=False
    # Live:  enabled=True, paper_trading_mode=False, live_trading_enabled=True
    'enabled': False,                   # 2026-05-05: re-enabled in PAPER mode (call sites wrapped)
    'paper_trading_mode': True,        # Paper: signals + DB log + reports, no Kite orders
    'live_trading_enabled': False,     # MIS live trading (only when paper_trading_mode=False)

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
    'signal_age_max_mins': 30,         # breakout candle cannot be older than this. 2026-04-27 sweep (research/28): 30 beats 15 on PF (1.62 vs 1.60), Sharpe (4.13 vs 3.94), MaxDD (32.07% vs 34.57%), Calmar (5.33 vs 4.75) on 15-stock 2024-2026 sample.
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
    'notify_eod_report': False,
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
    'enabled': False,
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
        'enabled': False,
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
        'enabled': False,
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
        'enabled': False,
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
        'enabled': False,
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
        'enabled': False,
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
        'enabled': False,
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
        'enabled': False,
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
        'enabled': False,
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
        'enabled': False,
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
        'enabled': False,
        'backtest_wr_pct': 93,
        'backtest_wins_per_year': 15,
        'backtest_trades_per_year': 16,
    },
]

# Convenience lookup: id -> variant dict
STRANGLE_VARIANTS_BY_ID = {v['id']: v for v in STRANGLE_VARIANTS}

# =============================================================================
# MST Index Strategy Defaults (NIFTY 30-min Master/Child SuperTrend + Pyramid)
# =============================================================================
# Spec: docs/Design/MST-INDEX-STRATEGY-DESIGN.md
# Research: research/35_*, research/36_*
MST_DEFAULTS = {
    # System control (paper-first; flip to live after shadow validation)
    # DISABLED 2026-05-15 after incident: live-tick pipeline froze May 7 14:45,
    # spurious credit_too_low rolls, REJECTED real-leg closes, mode reverted to
    # paper on restart. Do NOT re-enable until root causes fixed + reconciled.
    "enabled": False,                   # Master switch (off|paper|live in UI maps to enabled+paper_trading_mode)
    "paper_trading_mode": True,         # When enabled, true=paper, false=live
    "live_trading_enabled": False,      # Hard guard — also required for orders to actually go to Kite

    # Underlying
    "underlying": "NIFTY50",
    "underlying_token": 256265,         # NIFTY 50 index instrument_token (Kite)
    "options_root": "NIFTY",            # Used to filter NFO option contracts
    "lot_size": 65,                     # Contracts per lot (NIFTY = 65 as of 2026-05-05; verify at startup via kite.instruments)
    "lots_per_leg": 1,                  # Phase 1: 1 lot per leg per pyramid level

    # Bar configuration
    "timeframe_min": 30,                # 30-min bars; aggregated from NasTicker 5-min ticks

    # MST signal (SuperTrend) — research/35
    "atr_period": 21,
    "multiplier": 5.0,

    # CST signal (Stochastic) — research/35
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "stoch_ob": 80,                     # Long-bias CST fires when K crosses below D from K_prev>=80
    "stoch_os": 20,                     # Short-bias mirror

    # Pyramid trigger — research/36 (D_cumulative AND B) OR safety_wing_breach
    "pyramid_max_level": 2,             # 1=condor only; 2=condor + pyramid; cap at 2
    "pyramid_d_lookback": 6,            # cumulative D: bars after CST to count over
    "pyramid_d_threshold": 3,           # cumulative D: require net (above-below) >= 3
    "pyramid_ob_exit_threshold": 70,    # %K must drop below this before re-entering OB (long)
    "pyramid_os_exit_threshold": 30,    # mirror for short
    "pyramid_safety_wing_pct": 0.5,     # safety: fires at K3 + 50% × (K4-K3) breach

    # Spread structure
    "spread_width": 200,                # Standard 200/200/200 condor on NIFTY
    "reset_width": 100,                 # Reading D — narrow spot-centered when credit too low
    "min_credit_per_lot": 1000,         # rupees/lot threshold; below → roll-and-reset
    "debit_otm_offset": 50,             # NEW 2026-05-05: 1st OTM anchor (50pt on NIFTY).
                                        # Long bias: anchor=ATM+50; Short bias: anchor=ATM-50.
                                        # Whole condor shifts together; widths preserved.
                                        # Set to 0 to restore original ATM-anchored behavior.
    "strike_interval": 50,              # NIFTY weekly options strike spacing

    # DTE rule
    "min_dte_at_entry": 6,              # Universal — applies to every new entry

    # T-1 close timing
    "t_minus_1_close_hour": 15,
    "t_minus_1_close_minute": 25,       # IST — 5 min before market close

    # Order placement
    "limit_timeout_s": 30,              # Fall back to MARKET if LIMIT not filled
    "abort_on_leg_rejection": True,     # If any leg fails, close all already-placed legs

    # Notifications (uses services/notifications.py)
    "email_enabled": False,
    "whatsapp_enabled": False,
}


# =============================================================================
# Intraday 75% WR Quest — THREE Configs (A / B / C)
# =============================================================================
# Source: research/37_intraday_75wr_quest/FINAL_LIVE_SETUP.md
#
# CONFIG LAYOUT (locked 2026-05-06):
#   Config A — "3-System Original":  A1 Diamond Short + A2 Long-MR + A3 Long-TC,
#              ALL with TP 0.5% / SL 1.5% / EOD 15:25
#   Config B — "3-System Cost-Resilient": same A1/A2/A3 signals,
#              wider TP 2.0% / SL 1.5% / EOD 15:25
#   Config C — "Multi-Bar SHORT Bounce": research/38 walk-forward winner.
#              4 consecutive bearish 5-min bars + RSI<=55 + below VWAP +
#              NIFTY < own VWAP, TP 1.5% / SL 1.0% / EOD 15:25
#
# All THREE default to PAPER MODE (Off / Paper / Live three-state, mirroring
# ORB_DEFAULTS). Same shape of dict so engine_base can read them.
#
#   off:   enabled=False
#   paper: enabled=True, paper_trading_mode=True,  live_trading_enabled=False
#   live:  enabled=True, paper_trading_mode=False, live_trading_enabled=True
#
# Position sizing is FIXED-Rs risk (risk_per_trade_rs), capped by
# max_notional_per_trade. Concurrency is enforced ACROSS ALL THREE CONFIGS
# combined (max 5 total open positions), see engine_base.py.

# Cohort file paths (stocks lists computed by research/37 + research/38).
_RESEARCH_37 = BASE_DIR / 'research' / '37_intraday_75wr_quest' / 'results'
INTRADAY_75WR_COHORT_SHORT = str(_RESEARCH_37 / '07_short_diamonds.txt')
INTRADAY_75WR_COHORT_LONG_MR = str(_RESEARCH_37 / '11c_long_reversal_diamonds.txt')
INTRADAY_75WR_COHORT_LONG_TC = str(_RESEARCH_37 / '11b_trend_pullback_diamonds.txt')

# --- System 1: Diamond Short (25 stocks, 09:45 IST scan) ---------------------

DIAMOND_SHORT_DEFAULTS = {
    'system_id': 'diamond_short',
    'system_name': 'Diamond Short',
    'direction': 'SHORT',

    # Universe — 25 short-bias diamond stocks (research/37 stage 7)
    'universe': [
        'ZEEL', 'EDELWEISS', 'ASHOKA', 'CDSL', 'BANDHANBNK',
        'KNRCON', 'RAIN', 'GMDCLTD', 'HEG', 'NATCOPHARM',
        'SJVN', 'PRAJIND', 'TIINDIA', 'SUZLON', 'AMBER',
        'RCF', 'NETWORK18', 'NAM-INDIA', 'BAYERCROP', 'TCIEXP',
        'AARTIIND', 'NATIONALUM', 'IDEA', 'LTTS', 'NBCC',
    ],

    # Mode (paper default)
    'enabled': False,
    'paper_trading_mode': True,
    'live_trading_enabled': False,

    # Capital / sizing
    'capital': 200_000,                 # Rs 2L deposit allocated
    'mis_leverage': 5,                  # Zerodha MIS leverage
    'risk_per_trade_rs': 3000,          # Fixed Rs 3K cap per trade (NOT pct)
    'max_concurrent': 5,                # Per-system cap (combined cap also 5)
    'max_notional_per_trade': 200_000,
    'daily_loss_limit_pct': 0.03,       # 3% of capital
    'enforce_daily_loss_cap': True,

    # Entry timing — single scan at 09:45 IST (bar 6, after first 30 min)
    'entry_time': '09:45',
    'entry_window_seconds': 60,         # Allow 60s tolerance around scan time

    # Signal: short when stock < VWAP AND RSI(14) < rsi_threshold AND NIFTY weak
    'rsi_threshold': 40,                # Volume variant (research): WR 79%
    'nifty_filter': 'b3_change_neg',    # NIFTY first-30-min change < 0
    'require_below_vwap': True,

    # Exit
    'tp_pct': 0.5,                      # 0.5% TP
    'sl_pct': 1.5,                      # 1.5% SL
    'max_hold_bars': 60,                # full session hold
    'eod_squareoff_time': '15:25',
}


# --- System 2: Long Mean-Reversion (15 stocks, continuous 11:15-13:15) -------

LONG_MR_DEFAULTS = {
    'system_id': 'long_mr',
    'system_name': 'Long Mean-Reversion',
    'direction': 'LONG',

    # Universe — 15 long-reversal diamond stocks (research/37 stage 11c)
    'universe': [
        'BALRAMCHIN', 'AUROPHARMA', 'DCBBANK', 'CYIENT', 'APLLTD',
        'CERA', 'CGCL', 'BOSCHLTD', 'EIHOTEL', 'ASTRAL',
        'CESC', 'CAPLIPOINT', 'AAVAS', 'ALKEM', 'APLAPOLLO',
    ],

    # Mode (paper default)
    'enabled': False,
    'paper_trading_mode': True,
    'live_trading_enabled': False,

    # Capital / sizing
    'capital': 200_000,
    'mis_leverage': 5,
    'risk_per_trade_rs': 3000,
    'max_concurrent': 5,
    'max_notional_per_trade': 200_000,
    'daily_loss_limit_pct': 0.03,
    'enforce_daily_loss_cap': True,

    # Entry timing — continuous scan in 11:15-13:15 window (bar 24-48)
    'entry_window_start': '11:15',
    'entry_window_end': '13:15',
    'scan_cadence_minutes': 5,

    # Signal: stock down >= 2% AND RSI bounce 28 -> 35 AND NIFTY not crashing
    'drop_pct': -2.0,
    'rsi_oversold': 28,
    'rsi_lift': 35,
    'rsi_lookback_bars': 6,
    'require_3bar_break': True,
    'require_bullish_bar': True,
    'nifty_filter': 'b3_not_crashing',  # NIFTY first-30-min > -0.5%

    # Exit
    'tp_pct': 0.5,
    'sl_pct': 1.5,
    'max_hold_bars': 60,
    'eod_squareoff_time': '15:25',
}


# --- System 3: Long Trend-Continuation (30 stocks, continuous 09:15-10:30) ---

LONG_TC_DEFAULTS = {
    'system_id': 'long_tc',
    'system_name': 'Long Trend-Continuation',
    'direction': 'LONG',

    # Universe — 30 trend-pullback diamond stocks (research/37 stage 11b)
    'universe': [
        'MARICO', 'FINEORG', 'CCL', 'ASAHIINDIA', 'GALAXYSURF',
        'ASTRAZEN', 'JKPAPER', 'M&MFIN', 'JUBLFOOD', 'WHIRLPOOL',
        'GODREJAGRO', 'INDIANB', 'ZYDUSWELL', 'CHOLAFIN', 'WELCORP',
        'CUB', 'AMBER', 'INDIACEM', 'PNBHOUSING', 'HINDZINC',
        'JKCEMENT', 'SOBHA', 'MGL', 'GRINDWELL', 'BAJFINANCE',
        'DIXON', 'APLAPOLLO', 'DBL', 'BANDHANBNK', 'GODFRYPHLP',
    ],

    # Mode (paper default)
    'enabled': False,
    'paper_trading_mode': True,
    'live_trading_enabled': False,

    # Capital / sizing
    'capital': 200_000,
    'mis_leverage': 5,
    'risk_per_trade_rs': 3000,
    'max_concurrent': 5,
    'max_notional_per_trade': 200_000,
    'daily_loss_limit_pct': 0.03,
    'enforce_daily_loss_cap': True,

    # Entry timing — continuous scan first 75 min (09:15-10:30, bar 0-15)
    'entry_window_start': '09:15',
    'entry_window_end': '10:30',
    'scan_cadence_minutes': 5,
    'bar_min': 7,                       # earliest entry bar (after first hour)
    'bar_max': 15,                      # latest entry bar (10:30)

    # Signal: gap-up >= 0.5% + first-hour strength + pullback to VWAP within
    # 0.3% + RSI >= 45 + bullish current bar + NIFTY also gap-up bullish
    'gap_min_pct': 0.5,
    'first_hour_strength_pct': 0.5,
    'pullback_mode': 'vwap_within_0p3', # research/37 11b rank-4 winner
    'rsi_floor': 45,
    'nifty_filter': 'nifty_strong_both', # NIFTY gap-up AND bullish at b6

    # Exit
    'tp_pct': 0.5,
    'sl_pct': 1.5,
    'max_hold_bars': 60,
    'eod_squareoff_time': '15:25',
}


# Combined cap across all three configs (engine_base enforces this across A+B+C).
INTRADAY_75WR_COMBINED_MAX_CONCURRENT = 5


# ============================================================================
# CONFIG A — 3-System Original (TP 0.5% / SL 1.5%)
# ============================================================================
# Container for A1 Diamond Short + A2 Long-MR + A3 Long-TC. Each sub-signal is
# a separate scan with its own entry window and cohort, but they share Config
# A's exit profile, capital pool, paper/live flags and concurrency cap.

INTRADAY_CONFIG_A_DEFAULTS = {
    'config_id': 'A',
    'config_name': '3-System Original (TP 0.5/SL 1.5)',

    # 3-state mode (PAPER MODE LOCK by default)
    'enabled': False,
    'paper_trading_mode': True,
    'live_trading_enabled': False,

    # Capital / sizing
    'capital': 300_000,                 # Rs 3L deposit
    'mis_leverage': 5,
    'risk_per_trade_rs': 3000,
    'max_concurrent': 5,                # COMBINED across A+B+C (engine_base enforces)
    'max_notional_per_trade': 300_000,
    'daily_loss_limit_rs': 9000,        # 3 SLs at Rs 3K = Rs 9K (research/37 spec)
    'enforce_daily_loss_cap': True,
    'cost_per_side_pct': 0.05,          # 0.05% / side (used for paper-mode P&L)

    # Cohort files (resolved at engine init time)
    'cohort_short_path': INTRADAY_75WR_COHORT_SHORT,
    'cohort_long_mr_path': INTRADAY_75WR_COHORT_LONG_MR,
    'cohort_long_tc_path': INTRADAY_75WR_COHORT_LONG_TC,

    # Exit profile (Config A signature: tight TP, asymmetric SL)
    'tp_pct': 0.5,
    'sl_pct': 1.5,
    'eod_squareoff_time': '15:25',
    'max_hold_bars': 60,                # full session

    # Sub-signal A1 — Diamond Short (single 09:45 IST scan)
    'a1_enabled': True,
    'a1_entry_time': '09:45',
    'a1_entry_window_seconds': 60,
    'a1_rsi_threshold': 40,
    'a1_require_below_vwap': True,
    'a1_nifty_filter': 'b3_change_neg',     # NIFTY first-30-min change < 0

    # Sub-signal A2 — Long Mean-Reversion (continuous 11:15-13:15)
    'a2_enabled': True,
    'a2_entry_window_start': '11:15',
    'a2_entry_window_end': '13:15',
    'a2_drop_pct': -2.0,
    'a2_rsi_oversold': 28,
    'a2_rsi_lift': 35,
    'a2_rsi_lookback_bars': 6,
    'a2_require_3bar_break': True,
    'a2_require_bullish_bar': True,
    'a2_nifty_filter': 'b3_not_crashing',   # NIFTY first-30-min > -0.5%

    # Sub-signal A3 — Long Trend-Continuation (continuous 09:15-10:30)
    'a3_enabled': True,
    'a3_entry_window_start': '09:15',
    'a3_entry_window_end': '10:30',
    'a3_bar_min': 7,
    'a3_bar_max': 15,
    'a3_gap_min_pct': 0.5,
    'a3_first_hour_strength_pct': 0.5,
    'a3_pullback_mode': 'vwap_within_0p3',
    'a3_rsi_floor': 45,
    'a3_nifty_filter': 'nifty_strong_both',
}


# ============================================================================
# CONFIG B — 3-System Cost-Resilient (TP 2.0% / SL 1.5%)
# ============================================================================
# Same A1/A2/A3 signals as Config A; only the TP is wider. WR drops (53.5%
# OOS) but per-trade gain is higher, surviving 0.10%/side cost.

INTRADAY_CONFIG_B_DEFAULTS = {
    **INTRADAY_CONFIG_A_DEFAULTS,
    'config_id': 'B',
    'config_name': '3-System Cost-Resilient (TP 2.0/SL 1.5)',

    # 3-state mode (PAPER MODE LOCK)
    'enabled': False,
    'paper_trading_mode': True,
    'live_trading_enabled': False,

    # Capital / sizing — independent capital pool (separate Rs 3L)
    'capital': 300_000,
    'mis_leverage': 5,
    'risk_per_trade_rs': 3000,
    'max_concurrent': 5,
    'max_notional_per_trade': 300_000,
    'daily_loss_limit_rs': 9000,
    'enforce_daily_loss_cap': True,
    'cost_per_side_pct': 0.10,          # B is the cost-resilient profile

    # Exit profile — wider TP is the only delta from A
    'tp_pct': 2.0,
    'sl_pct': 1.5,
    'eod_squareoff_time': '15:25',
    'max_hold_bars': 60,
}


# ============================================================================
# CONFIG C — Multi-Bar SHORT Bounce (TP 1.5% / SL 1.0%)
# ============================================================================
# research/38 walk-forward winner. Continuous scan looking for 4 consecutive
# bearish 5-min bars + RSI<=55 + below own VWAP + NIFTY < its own VWAP.

INTRADAY_CONFIG_C_DEFAULTS = {
    'config_id': 'C',
    'config_name': 'Multi-Bar SHORT Bounce (TP 1.5/SL 1.0)',

    # 3-state mode (PAPER MODE LOCK)
    'enabled': False,
    'paper_trading_mode': True,
    'live_trading_enabled': False,

    # Capital / sizing
    'capital': 300_000,
    'mis_leverage': 5,
    'risk_per_trade_rs': 3000,
    'max_concurrent': 5,                # COMBINED cap with A+B
    'max_notional_per_trade': 300_000,
    'daily_loss_limit_rs': 9000,
    'enforce_daily_loss_cap': True,
    'cost_per_side_pct': 0.05,

    # Cohort — same 25 short-diamonds as A1/B1
    'cohort_short_path': INTRADAY_75WR_COHORT_SHORT,

    # Exit profile (favorable RR 1.5:1)
    'tp_pct': 1.5,
    'sl_pct': 1.0,
    'eod_squareoff_time': '15:25',
    'max_hold_bars': 60,

    # Continuous scan window (per-bar throughout session)
    'entry_window_start': '09:30',      # 4 bars need to elapse before signal
    'entry_window_end': '15:00',        # last entry 30 min before EOD
    'scan_cadence_minutes': 5,

    # Signal params
    'n_bars_consecutive': 4,            # 4 consecutive bearish 5-min bars
    'require_lower_highs': True,        # each bar's high <= prior bar's high
    'rsi_max': 55,                      # RSI(14) <= 55
    'require_below_vwap': True,
    'nifty_below_own_vwap_required': True,
    'min_bar_idx': 4,
    'last_bars_skip': 5,                # don't enter in the last 5 bars
}


# Convenience map for engine_base / api lookup
INTRADAY_75WR_CONFIGS = {
    'A': INTRADAY_CONFIG_A_DEFAULTS,
    'B': INTRADAY_CONFIG_B_DEFAULTS,
    'C': INTRADAY_CONFIG_C_DEFAULTS,
}


# =============================================================================
# Config D — 6-Pair Cointegrated Pair Trading (carry-forward F&O futures)
# =============================================================================
# Source: research/39_carry_forward_75wr_quest/CARRY_FORWARD_75WR_DAILY_SWEEP_RESULTS.md
# Walk-forward backtest: WR 78.7%, PF 3.57, MaxDD 0.06%, n=108 trades.
# Daily EOD process at 16:00 IST (post F&O close):
#   1. Fetch latest daily-close prices for both legs of each pair
#   2. Compute spread = log(P_a) - alpha - beta * log(P_b)
#   3. Compute z-score on rolling lookback (20 or 40 days per pair)
#   4. ENTRY (long spread)  if z <= -entry_z: BUY pair-A futures + SELL pair-B futures
#   5. ENTRY (short spread) if z >= +entry_z: SELL pair-A futures + BUY pair-B futures
#   6. EXIT on first of: |z| crosses 0 (mean revert) | |z| >= stop_z | hold-cap days
#
# Mode tri-state (mirrors ORB / KC6 / NAS pattern):
#   off:   enabled=False
#   paper: enabled=True, paper_trading_mode=True,  live_trading_enabled=False
#   live:  enabled=True, paper_trading_mode=False, live_trading_enabled=True
#
# Position sizing: Rs.6,000 total risk per pair-trade (Rs.3,000 per leg).
# F&O futures margin ~12-15% notional => ~Rs.40-50K margin per pair.
# Max concurrent = 5 pairs (out of 6) => peak D-margin ~Rs.2L on Rs.10L base.
#
# Concurrency cap is INTERNAL to Config D — does not interact with
# intraday A/B/C concurrency cap (different account segment, different
# margin pool: F&O carry-forward NRML vs MIS cash equity).

PAIR_TRADING_DEFAULTS = {
    # Mode (paper default)
    'enabled': False,
    'paper_trading_mode': True,         # PAPER MODE LOCK — no real Kite orders
    'live_trading_enabled': False,      # Hard guard — also required for live orders

    # Capital + sizing
    'capital': 1_000_000,               # Rs.10L base capital allocated to D
    'risk_per_pair_rs': 6000,           # Rs.6K per pair-trade (Rs.3K per leg)
    'max_concurrent': 5,                # Max 5 of 6 pairs open at any time

    # Cost model (F&O futures, much cheaper than CNC)
    'cost_per_side_pct': 0.03,          # 0.03% per side per leg => 0.06% RT/leg => 0.12% RT/pair
    'cost_stress_per_side_pct': 0.10,   # Stress: 0.20%/leg => 0.40%/pair (used for paper-mode reporting)

    # Cohort refresh policy
    'cohort_refresh_quarterly': True,
    'rolling_alpha_beta_window_days': 252,  # 12-month rolling re-fit window

    # Daily price-history depth needed for z-score (longest lookback + buffer)
    'history_lookback_days': 90,            # 60d (max lookback) + 30d buffer

    # The 6 winning cointegrated pairs (alpha/beta from
    # research/39 results/05_pair_walk_forward_relaxed.csv)
    # Schema per pair:
    #   name        — pair label (used as DB key)
    #   symA, symB  — F&O underlying symbols (futures contract resolved at runtime)
    #   alpha, beta — TRAIN-FIT regression coefficients
    #                 spread = log(P_a) - alpha - beta * log(P_b)
    #   half_life   — empirical mean-reversion half-life in days (from universe screen)
    #   entry_z     — |z| threshold to open a position
    #   stop_z      — |z| threshold to stop out (999.0 = no stop)
    #   hold_days   — hard time-exit (calendar days held)
    #   lookback    — rolling window for z-score mean/std
    #   te_wr       — out-of-sample test win rate (informational)
    #   te_pf       — out-of-sample test profit factor (informational)
    'pairs': [
        {
            'name': 'HAVELLS-MARICO',
            'symA': 'HAVELLS', 'symB': 'MARICO',
            'alpha': -2.6023, 'beta': 1.5551,
            'half_life': 22.44,
            'entry_z': 2.0, 'stop_z': 4.0,
            'hold_days': 20, 'lookback': 20,
            'te_wr': 93.75, 'te_pf': 8.50,
        },
        {
            'name': 'BAJFINANCE-KOTAKBANK',
            'symA': 'BAJFINANCE', 'symB': 'KOTAKBANK',
            'alpha': -11.4604, 'beta': 2.3819,
            'half_life': 25.67,
            'entry_z': 2.0, 'stop_z': 4.0,
            'hold_days': 20, 'lookback': 20,
            'te_wr': 83.33, 'te_pf': 7.05,
        },
        {
            'name': 'DABUR-HINDUNILVR',
            'symA': 'DABUR', 'symB': 'HINDUNILVR',
            'alpha': 0.1893, 'beta': 0.7838,
            'half_life': 23.89,
            'entry_z': 2.0, 'stop_z': 5.0,
            'hold_days': 20, 'lookback': 20,
            'te_wr': 78.95, 'te_pf': 4.71,
        },
        {
            'name': 'COFORGE-HCLTECH',
            'symA': 'COFORGE', 'symB': 'HCLTECH',
            'alpha': -4.67, 'beta': 1.6289,
            'half_life': 24.92,
            'entry_z': 2.0, 'stop_z': 4.0,
            'hold_days': 15, 'lookback': 20,
            'te_wr': 76.19, 'te_pf': 3.39,
        },
        {
            'name': 'DABUR-TCS',
            'symA': 'DABUR', 'symB': 'TCS',
            'alpha': 2.1594, 'beta': 0.5126,
            'half_life': 26.71,
            'entry_z': 2.0, 'stop_z': 999.0,
            'hold_days': 10, 'lookback': 20,
            'te_wr': 71.43, 'te_pf': 3.79,
        },
        {
            'name': 'APOLLOHOSP-COFORGE',
            'symA': 'APOLLOHOSP', 'symB': 'COFORGE',
            'alpha': 1.8089, 'beta': 0.9682,
            'half_life': 22.15,
            'entry_z': 2.0, 'stop_z': 999.0,
            'hold_days': 10, 'lookback': 40,
            'te_wr': 75.00, 'te_pf': 1.93,
        },
    ],

    # F&O exchange + product for futures legs
    'exchange': 'NFO',
    'product': 'NRML',                  # Carry-forward (NOT MIS)
    'order_type': 'MARKET',             # Both legs MARKET to ensure simultaneous fill

    # Weekly loss circuit-breaker (D-specific): 3 SL exits in 5 sessions => halt week
    'weekly_sl_circuit_breaker_count': 3,
    'weekly_sl_circuit_breaker_window_days': 5,
}
