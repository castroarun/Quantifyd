"""SCC App Configuration."""
import sys
from pathlib import Path

# Add parent directory to path for importing services
PARENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT_DIR))

SCC_PORT = 5001
SCC_DEBUG = True
PARENT_API_URL = 'http://127.0.0.1:5000'

# Strategy metadata for the SCC dashboard
STRATEGY_META = {
    'maruthi': {
        'name': 'Maruthi SuperTrend',
        'type': 'F&O',
        'system_type': 'Positional',
        'timeframe': '30min',
        'instruments': 'MARUTI Futures + Options',
        'description': 'Dual SuperTrend (7,5 master + 7,2 child) always-on strategy on MARUTI.',
        'api_base': '/api/maruthi',
        'parent_proxy': True,  # Needs parent app for ticker
    },
    'kc6': {
        'name': 'KC6 Mean Reversion',
        'type': 'Equity',
        'system_type': 'Swing',
        'timeframe': 'Daily',
        'instruments': 'Nifty 500 Equities',
        'description': 'Keltner Channel(6, 1.3 ATR) mean reversion with SMA(200) filter.',
        'api_base': '/api/kc6',
        'parent_proxy': False,  # Can read DB directly
    },
    'nas': {
        'name': 'NAS Nifty Strangle',
        'type': 'F&O',
        'system_type': 'Intraday',
        'timeframe': '5min',
        'instruments': 'NIFTY Options',
        'description': 'SuperTrend-based directional strangle on Nifty with cascading entries.',
        'api_base': '/api/nas',
        'parent_proxy': True,
    },
    'nas_atm': {
        'name': 'NAS ATM Strangle',
        'type': 'F&O',
        'system_type': 'Intraday',
        'timeframe': '5min',
        'instruments': 'NIFTY ATM Options',
        'description': 'ATM strangle with cross-leg adjustment and premium bounds.',
        'api_base': '/api/nas-atm',
        'parent_proxy': True,
    },
    'bnf': {
        'name': 'BNF Squeeze & Fire',
        'type': 'F&O',
        'system_type': 'Swing',
        'timeframe': 'Daily',
        'instruments': 'BANKNIFTY Options',
        'description': 'Bollinger squeeze detection with directional breakout on BankNifty.',
        'api_base': '/api/bnf',
        'parent_proxy': False,
    },
    'trident': {
        'name': 'Trident Multi-Strategy',
        'type': 'F&O',
        'system_type': 'Swing',
        'timeframe': 'Daily',
        'instruments': 'Nifty 500 F&O',
        'description': 'Multi-strategy scanner combining momentum, mean reversion, and squeeze.',
        'api_base': '/api/trident',
        'parent_proxy': False,
    },
}

# Blueprint data (from the SCC JSX design)
BLUEPRINTS = [
    {
        'id': 'maruthi', 'name': 'Maruthi SuperTrend', 'version': 'v1.0',
        'system_type': 'Positional', 'holding_period': 'Days–Weeks', 'timeframe': '30min',
        'instruments': 'MARUTI Futures + Options', 'capital': '15L',
        'description': 'Dual SuperTrend always-on strategy. Master (7,5) sets regime, Child (7,2) triggers entries. Futures + option hedges.',
        'entry': ['Master ST flips direction → new regime', 'Child ST flips in regime direction → signal candle', 'Signal candle high/low breached → entry trigger', 'Max 5 futures lots, 1 short option per lot'],
        'exit': ['Master ST flips opposite → close all positions', 'Hard SL: Master ST ± 1×ATR (trailing, tick-level)', 'Child flips against → keep positions, short counter option'],
        'stop_loss': ['Trailing hard SL on every tick via WebSocket', 'Backup SL-M on Zerodha as safety net', 'ATR-based buffer (~75-125 pts on MARUTI)'],
        'position_sizing': '1 lot (50 shares) per signal. Max 5 lots.',
        'indicators': ['SuperTrend(7,5)', 'SuperTrend(7,2)', 'ATR(7)', 'Wilder RMA'],
        'tags': ['always-on', 'futures', 'options', 'supertrend'],
    },
    {
        'id': 'kc6', 'name': 'KC6 Mean Reversion', 'version': 'v2.0',
        'system_type': 'Swing', 'holding_period': '1–15 days', 'timeframe': 'Daily',
        'instruments': 'Nifty 500 Equities', 'capital': '10L',
        'description': 'Buy when close < KC(6,1.3) lower AND close > SMA(200). Exit at KC mid or SL/TP/MaxHold.',
        'entry': ['Close < KC(6, 1.3 ATR) Lower Band', 'Close > SMA(200)', 'Universe ATR Ratio < 1.3x (crash filter)'],
        'exit': ['Standing SELL LIMIT at KC6 mid (placed each morning)', '5% stop loss', '15% take profit', '15 days max hold'],
        'stop_loss': ['5% from entry price', 'Hard exit at 15 days'],
        'position_sizing': 'Equal weight. Max 10 concurrent.',
        'indicators': ['KC(6,1.3)', 'SMA(200)', 'ATR(14)'],
        'tags': ['mean-reversion', 'equity', 'swing'],
        'backtest': {'total_trades': 2482, 'win_rate': 65.0, 'profit_factor': 1.70, 'sharpe': 1.42, 'max_dd': -12.3},
    },
    {
        'id': 'nas', 'name': 'NAS Nifty Strangle', 'version': 'v1.0',
        'system_type': 'Intraday', 'holding_period': 'Intraday', 'timeframe': '5min',
        'instruments': 'NIFTY Options', 'capital': '5L',
        'description': 'SuperTrend directional strangle on Nifty with cascading entries (1 per candle, max 5).',
        'entry': ['SuperTrend direction flip on 5-min', 'Cascading: 1 strangle per candle close', 'Max 5 strangles active'],
        'exit': ['SuperTrend flips opposite', 'EOD squareoff at 15:15', 'Premium bounds: 4-24'],
        'stop_loss': ['Cross-leg adjustment (alternating OUT/IN)', 'EOD hard close'],
        'position_sizing': '1 lot per signal. Max 5.',
        'indicators': ['SuperTrend(10,1.5)', 'ATR(10)'],
        'tags': ['intraday', 'options', 'strangle', 'nifty'],
    },
    {
        'id': 'bnf', 'name': 'BNF Squeeze & Fire', 'version': 'v1.0',
        'system_type': 'Swing', 'holding_period': '1–5 days', 'timeframe': 'Daily',
        'instruments': 'BANKNIFTY Options', 'capital': '3L',
        'description': 'Detects Bollinger Band squeeze on BankNifty daily. Enters directional on squeeze fire.',
        'entry': ['BB Width < BB Width MA (squeeze active)', 'Squeeze fires (BB expands past KC)', 'Direction from histogram'],
        'exit': ['Histogram reverses', 'Target: 2× ATR', 'Time: 5 days max'],
        'stop_loss': ['Below/above squeeze range', 'Max 30% premium loss'],
        'position_sizing': 'Risk 1%. Max 1 position.',
        'indicators': ['BB(20,2.0)', 'KC(20,1.5)', 'TTM Histogram'],
        'tags': ['squeeze', 'volatility-breakout', 'banknifty'],
    },
    {
        'id': 'covered_call', 'name': 'Covered Call Writer', 'version': 'v1.0',
        'system_type': 'Monthly', 'holding_period': 'Expiry to expiry', 'timeframe': 'Daily',
        'instruments': 'F&O Stocks + CE', 'capital': '25L',
        'description': '7-layer conviction scanner identifies safe stocks for covered call writing. Targets 1.5–2.5% monthly yield.',
        'entry': ['7-layer score ≥65 (GREEN)', 'Sell ~10-delta CE, 25–30 DTE', 'IV Percentile > 66%', 'No earnings within 7 days'],
        'exit': ['Hold to expiry (default)', 'Roll up+out if stock within 3% of strike with >10 DTE', 'Close if stock gaps past strike'],
        'stop_loss': ['Underlying -10% → close all', 'Below 200 DMA weekly → exit', 'Earnings within 7 days → close'],
        'position_sizing': '1 lot per stock. Max 3 concurrent.',
        'indicators': ['ADX(14)', 'RSI(14)', 'BB(20,2)', 'HV(20)', 'SMA(200)'],
        'tags': ['income', 'covered-call', 'monthly', 'options'],
        'backtest': {'total_trades': 94, 'win_rate': 78.7, 'profit_factor': 2.64, 'sharpe': 2.8, 'max_dd': -5.2},
    },
    {
        'id': 'wide_strangle', 'name': 'Wide Strangle', 'version': 'v1.0',
        'system_type': 'Monthly', 'holding_period': '15–25 days', 'timeframe': 'Daily',
        'instruments': 'Nifty 50 F&O Stocks', 'capital': '10L',
        'description': '7-factor range-squeeze scanner deploys wide strangles (10% OTM) on range-bound large caps.',
        'entry': ['Range Detector squeeze signal', '7-factor score top 3–5 (sector-diversified)', 'IV Rank > 50%', 'India VIX 15–25', 'No earnings within 15 days'],
        'exit': ['50% of max premium (profit target)', 'Either leg loss > 2× premium (stop)', '7 days before expiry (time exit)', 'India VIX > 30', 'ADX > 30 (trend break)'],
        'stop_loss': ['2× premium per leg', 'VIX spike exit', 'Earnings blackout 7 days before'],
        'position_sizing': 'Equal notional. Max 4–5 concurrent. 50% capital reserve.',
        'indicators': ['BB(10,2)', 'ATR(14)', 'ADX(14)', 'HV(20)', 'Beta'],
        'tags': ['strangle', 'range-bound', 'monthly', 'options'],
    },
]
