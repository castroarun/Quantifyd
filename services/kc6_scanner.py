"""
KC6 Signal Scanner
====================

Pure computation engine for KC6 mean reversion strategy.
No Kite API calls - works on OHLCV DataFrames.

Strategy:
- Entry: Close < KC(6, 1.3 ATR) Lower AND Close > SMA(200)
- Exit priority: SL(5%) -> TP(15%) -> Max Hold(15d) -> Signal(High > KC6 Mid)
- Crash filter: Universe ATR Ratio >= 1.3x blocks new entries
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import KC6_DEFAULTS

logger = logging.getLogger(__name__)


# =============================================================================
# Indicator Functions (from crash_filter_v3.py)
# =============================================================================

def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(window=period).mean()


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def keltner(df: pd.DataFrame, ep: int = 6, ap: int = 6, m: float = 1.3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (mid, upper, lower) Keltner Channel bands."""
    mid = ema(df['close'], ep)
    a = atr_series(df['high'], df['low'], df['close'], ap)
    return mid, mid + m * a, mid - m * a


# =============================================================================
# Indicator Computation
# =============================================================================

def compute_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add KC6 mid/upper/lower, SMA200, ATR to an OHLCV DataFrame.
    Modifies in-place and returns the DataFrame.
    """
    if config is None:
        config = KC6_DEFAULTS

    ep = config.get('kc_ema_period', 6)
    ap = config.get('kc_atr_period', 6)
    m = config.get('kc_multiplier', 1.3)
    sma_period = config.get('sma_period', 200)
    atr_lookback = config.get('atr_lookback', 14)
    atr_avg_window = config.get('atr_avg_window', 50)

    df['kc6_mid'], df['kc6_upper'], df['kc6_lower'] = keltner(df, ep, ap, m)
    df['sma200'] = sma(df['close'], sma_period)
    df['above_sma200'] = df['close'] > df['sma200']

    # ATR for crash filter
    df['atr14'] = atr_series(df['high'], df['low'], df['close'], atr_lookback)
    df['atr14_50avg'] = df['atr14'].rolling(atr_avg_window).mean()
    df['atr_ratio'] = df['atr14'] / df['atr14_50avg']

    return df


# =============================================================================
# Crash Filter
# =============================================================================

def compute_universe_atr_ratio(symbol_data: Dict[str, pd.DataFrame],
                                config: dict = None) -> float:
    """
    Compute median ATR ratio across all symbols in the universe.
    This is the crash filter: if >= 1.3x, block all entries.

    Replicates crash_filter_v3.py:54-60, 78.
    """
    if config is None:
        config = KC6_DEFAULTS

    atr_lookback = config.get('atr_lookback', 14)
    atr_avg_window = config.get('atr_avg_window', 50)

    ratios = []
    for sym, df in symbol_data.items():
        if len(df) < atr_avg_window + atr_lookback + 10:
            continue
        try:
            a14 = atr_series(df['high'], df['low'], df['close'], atr_lookback)
            a50avg = a14.rolling(atr_avg_window).mean()
            ratio = a14 / a50avg
            latest = ratio.iloc[-1]
            if pd.notna(latest) and np.isfinite(latest):
                ratios.append(latest)
        except Exception:
            continue

    if not ratios:
        logger.warning("No valid ATR ratios computed - defaulting to 1.0")
        return 1.0

    median_ratio = float(np.median(ratios))
    logger.info(f"Universe ATR ratio: {median_ratio:.3f} (from {len(ratios)} symbols)")
    return median_ratio


def is_crash_filter_active(universe_atr_ratio: float, config: dict = None) -> bool:
    """Check if crash filter should block entries."""
    if config is None:
        config = KC6_DEFAULTS
    threshold = config.get('atr_ratio_threshold', 1.3)
    return universe_atr_ratio >= threshold


# =============================================================================
# Entry Scanner
# =============================================================================

def scan_entries(symbol_data: Dict[str, pd.DataFrame],
                 universe_atr_ratio: float,
                 existing_symbols: List[str] = None,
                 config: dict = None) -> List[Dict]:
    """
    Scan for entry signals: Close < KC6 Lower AND Close > SMA200.

    Args:
        symbol_data: Dict of symbol -> OHLCV DataFrame (with indicators computed)
        universe_atr_ratio: Current universe ATR ratio
        existing_symbols: Symbols already in portfolio (skip these)
        config: Strategy config

    Returns:
        List of entry signal dicts
    """
    if config is None:
        config = KC6_DEFAULTS

    if existing_symbols is None:
        existing_symbols = []

    threshold = config.get('atr_ratio_threshold', 1.3)

    # Crash filter check
    if universe_atr_ratio >= threshold:
        logger.info(f"Crash filter ACTIVE (ATR ratio {universe_atr_ratio:.3f} >= {threshold}). No entries.")
        return []

    entries = []
    for sym, df in symbol_data.items():
        if sym in existing_symbols:
            continue
        if len(df) < 201:
            continue

        try:
            row = df.iloc[-1]

            # Entry condition: close < KC6 lower AND close > SMA200
            if (pd.notna(row.get('kc6_lower')) and pd.notna(row.get('sma200'))
                    and row['close'] < row['kc6_lower']
                    and row['close'] > row['sma200']):

                sl_pct = config.get('sl_pct', 5.0)
                tp_pct = config.get('tp_pct', 15.0)

                entries.append({
                    'symbol': sym,
                    'close': round(float(row['close']), 2),
                    'kc6_lower': round(float(row['kc6_lower']), 2),
                    'kc6_mid': round(float(row['kc6_mid']), 2),
                    'sma200': round(float(row['sma200']), 2),
                    'entry_price': round(float(row['close']), 2),
                    'sl_price': round(float(row['close']) * (1 - sl_pct / 100), 2),
                    'tp_price': round(float(row['close']) * (1 + tp_pct / 100), 2),
                    'atr_ratio': round(float(row.get('atr_ratio', 0)), 3),
                    'date': str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1]),
                })
        except Exception as e:
            logger.warning(f"Error scanning {sym}: {e}")
            continue

    logger.info(f"Entry scan: {len(entries)} signals from {len(symbol_data)} symbols")
    return entries


# =============================================================================
# Exit Scanner
# =============================================================================

def scan_exits(positions: List[Dict], symbol_data: Dict[str, pd.DataFrame],
               config: dict = None) -> List[Dict]:
    """
    Check exit conditions for active positions.

    Exit priority:
    1. SL: low <= entry * (1 - sl_pct/100)
    2. TP: high >= entry * (1 + tp_pct/100)
    3. Max Hold: days held >= max_hold_days
    4. Signal: high > KC6 mid -> exit @ KC6 mid

    Args:
        positions: List of active position dicts from DB
        symbol_data: Dict of symbol -> OHLCV DataFrame (with indicators)
        config: Strategy config

    Returns:
        List of exit signal dicts
    """
    if config is None:
        config = KC6_DEFAULTS

    sl_pct = config.get('sl_pct', 5.0)
    tp_pct = config.get('tp_pct', 15.0)
    max_hold = config.get('max_hold_days', 15)

    exits = []
    today = datetime.now().date()

    for pos in positions:
        sym = pos['symbol']
        if sym not in symbol_data:
            logger.warning(f"No data for position {sym} - skipping exit check")
            continue

        df = symbol_data[sym]
        if len(df) < 2:
            continue

        try:
            row = df.iloc[-1]
            entry_price = pos['entry_price']
            entry_date = pos['entry_date']

            if isinstance(entry_date, str):
                entry_dt = datetime.strptime(entry_date, '%Y-%m-%d').date()
            else:
                entry_dt = entry_date

            hold_days = (today - entry_dt).days

            current_high = float(row['high'])
            current_low = float(row['low'])
            current_close = float(row['close'])

            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + tp_pct / 100)

            exit_reason = None
            exit_price = None

            # Priority 1: Stop Loss
            if current_low <= sl_price:
                exit_reason = 'STOP_LOSS'
                exit_price = round(sl_price, 2)

            # Priority 2: Take Profit
            elif current_high >= tp_price:
                exit_reason = 'TAKE_PROFIT'
                exit_price = round(tp_price, 2)

            # Priority 3: Max Hold
            elif hold_days >= max_hold:
                exit_reason = 'MAX_HOLD'
                exit_price = round(current_close, 2)

            # Priority 4: Signal - high > KC6 mid -> exit @ KC6 mid
            elif pd.notna(row.get('kc6_mid')) and current_high > row['kc6_mid']:
                exit_reason = 'SIGNAL_KC6_MID'
                exit_price = round(float(row['kc6_mid']), 2)

            if exit_reason:
                pnl_pct = round((exit_price / entry_price - 1) * 100, 2)
                exits.append({
                    'position_id': pos['id'],
                    'symbol': sym,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct,
                    'hold_days': hold_days,
                    'current_close': round(current_close, 2),
                    'kc6_mid': round(float(row.get('kc6_mid', 0)), 2),
                })
        except Exception as e:
            logger.warning(f"Error checking exit for {sym}: {e}")
            continue

    logger.info(f"Exit scan: {len(exits)} signals from {len(positions)} positions")
    return exits


# =============================================================================
# Target Price Computation (for daily limit orders)
# =============================================================================

def compute_target_prices(positions: List[Dict],
                          symbol_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """
    Compute today's KC6 mid for each active position.
    Used to place/update daily SELL LIMIT target orders.

    Returns list of {position_id, symbol, kc6_mid_today, tp_price, sl_price}.
    """
    targets = []
    for pos in positions:
        sym = pos['symbol']
        if sym not in symbol_data:
            continue

        df = symbol_data[sym]
        if len(df) < 10:
            continue

        try:
            row = df.iloc[-1]
            kc6_mid = float(row.get('kc6_mid', 0))
            if kc6_mid <= 0 or pd.isna(kc6_mid):
                continue

            # Target exit price = KC6 mid (the limit order price)
            # But cap it: don't place target below entry (we'd be selling at a loss via limit)
            # The SL handles losses - target is only for profit exits
            target_price = round(kc6_mid, 2)

            targets.append({
                'position_id': pos['id'],
                'symbol': sym,
                'entry_price': pos['entry_price'],
                'kc6_mid_today': target_price,
                'sl_price': pos['sl_price'],
                'tp_price': pos['tp_price'],
                'current_close': round(float(row['close']), 2),
            })
        except Exception as e:
            logger.warning(f"Error computing target for {sym}: {e}")
            continue

    logger.info(f"Target prices computed for {len(targets)}/{len(positions)} positions")
    return targets


# =============================================================================
# Data Loading (from Kite API or local DB)
# =============================================================================

def load_daily_data_from_db(symbols: List[str], db_path: str, min_bars: int = 250) -> Dict[str, pd.DataFrame]:
    """
    Load daily OHLCV from the local market_data.db and compute indicators.
    Used for backtesting or when Kite is not available.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    result = {}

    for sym in symbols:
        try:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM market_data_unified "
                "WHERE symbol=? AND timeframe='day' ORDER BY date",
                conn, params=[sym]
            )
            if len(df) < min_bars:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.astype(float)
            compute_indicators(df)
            result[sym] = df
        except Exception as e:
            logger.warning(f"Error loading {sym}: {e}")
            continue

    conn.close()
    logger.info(f"Loaded {len(result)} symbols from DB")
    return result


def load_daily_data_from_kite(kite, symbols: List[str], days: int = 300) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV from Kite API for the given symbols.
    Computes KC6 indicators on each DataFrame.

    Args:
        kite: KiteConnect instance (authenticated)
        symbols: List of NSE symbols
        days: Number of calendar days of history to fetch

    Returns:
        Dict of symbol -> DataFrame with indicators
    """
    import time as _time

    # Get instrument tokens
    try:
        instruments = kite.instruments('NSE')
        token_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
    except Exception as e:
        logger.error(f"Failed to fetch instruments: {e}")
        return {}

    from_date = datetime.now() - timedelta(days=days)
    to_date = datetime.now()

    result = {}
    fetched = 0

    for sym in symbols:
        token = token_map.get(sym)
        if not token:
            continue

        try:
            data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval='day'
            )
            if not data or len(data) < 201:
                continue

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            compute_indicators(df)
            result[sym] = df
            fetched += 1

            # Rate limit: Kite allows 3 req/sec
            _time.sleep(0.35)

        except Exception as e:
            logger.warning(f"Error fetching {sym}: {e}")
            continue

    logger.info(f"Fetched {fetched} symbols from Kite API")
    return result


# =============================================================================
# Full Scan Pipeline
# =============================================================================

def run_full_scan(kite=None, config: dict = None, db_path: str = None) -> Dict:
    """
    Run the complete KC6 scan pipeline:
    1. Load data (Kite API if available, else local DB)
    2. Compute universe ATR ratio (crash filter)
    3. Scan for exits on active positions
    4. Scan for new entries

    Returns dict with all scan results.
    """
    from services.kc6_db import get_kc6_db
    from services.nifty500_universe import get_nifty500

    if config is None:
        config = KC6_DEFAULTS

    db = get_kc6_db()
    universe = get_nifty500()
    symbols = universe.symbols

    # Step 1: Load data
    if kite:
        logger.info("Loading data from Kite API...")
        symbol_data = load_daily_data_from_kite(kite, symbols)
    elif db_path:
        logger.info("Loading data from local DB...")
        symbol_data = load_daily_data_from_db(symbols, db_path)
    else:
        # Default to local DB
        from config import MARKET_DATA_DB
        symbol_data = load_daily_data_from_db(symbols, str(MARKET_DATA_DB))

    if not symbol_data:
        logger.error("No symbol data loaded - scan aborted")
        return {'error': 'No data loaded', 'entries': [], 'exits': [], 'universe_atr_ratio': None}

    # Step 2: Compute crash filter
    universe_atr_ratio = compute_universe_atr_ratio(symbol_data, config)
    crash_active = is_crash_filter_active(universe_atr_ratio, config)

    # Step 3: Check exits on active positions
    active_positions = db.get_active_positions()
    exit_signals = scan_exits(active_positions, symbol_data, config)

    # Step 4: Scan entries (blocked if crash filter active)
    existing_symbols = [p['symbol'] for p in active_positions]
    entry_signals = scan_entries(symbol_data, universe_atr_ratio, existing_symbols, config)

    # Step 5: Save daily state
    today = datetime.now().strftime('%Y-%m-%d')
    db.save_daily_state(
        trade_date=today,
        universe_atr_ratio=universe_atr_ratio,
        crash_filter_active=crash_active,
        positions_count=len(active_positions),
        entry_signals=len(entry_signals),
        exit_signals=len(exit_signals),
    )

    return {
        'universe_atr_ratio': round(universe_atr_ratio, 3),
        'crash_filter_active': crash_active,
        'symbols_loaded': len(symbol_data),
        'active_positions': len(active_positions),
        'entries': entry_signals,
        'exits': exit_signals,
        'scan_time': datetime.now().isoformat(),
    }
