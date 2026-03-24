"""
Trident Scanner
==================

Signal detection for PA_MACD + RangeBreakout strategies.
Pure computation — no Kite API calls. Reads from market_data.db.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_daily_data_from_db(symbols: List[str], db_path: str,
                            lookback_days: int = 300) -> Dict[str, pd.DataFrame]:
    """Load daily OHLCV from market_data.db for given symbols."""
    import sqlite3
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    conn = sqlite3.connect(db_path)
    data = {}
    for sym in symbols:
        df = pd.read_sql_query(
            """SELECT date, open, high, low, close, volume
               FROM market_data_unified
               WHERE symbol=? AND timeframe='day' AND date >= ?
               ORDER BY date""",
            conn, params=(sym, start_date)
        )
        if len(df) >= 30:
            df['date'] = pd.to_datetime(df['date'])
            data[sym] = df
    conn.close()
    logger.info(f"Loaded {len(data)}/{len(symbols)} symbols from DB")
    return data


def load_daily_data_from_kite(kite, symbols: List[str],
                               lookback_days: int = 300) -> Dict[str, pd.DataFrame]:
    """Load daily data from Kite API with rate limiting."""
    import time
    from datetime import date

    end = date.today()
    start = end - timedelta(days=lookback_days)
    data = {}
    for sym in symbols:
        try:
            records = kite.historical_data(
                instrument_token=_get_instrument_token(kite, sym),
                from_date=start, to_date=end, interval='day'
            )
            if records and len(records) >= 30:
                df = pd.DataFrame(records)
                df.rename(columns={'date': 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                data[sym] = df
            time.sleep(0.35)  # Rate limit: 3 req/sec
        except Exception as e:
            logger.warning(f"Failed to load {sym} from Kite: {e}")
    logger.info(f"Loaded {len(data)}/{len(symbols)} symbols from Kite")
    return data


def _get_instrument_token(kite, symbol: str) -> int:
    """Get instrument token for NSE equity."""
    instruments = kite.instruments('NSE')
    for inst in instruments:
        if inst['tradingsymbol'] == symbol:
            return inst['instrument_token']
    raise ValueError(f"Instrument token not found for {symbol}")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR(14) and MACD to DataFrame."""
    # ATR(14)
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(span=14, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    return df


def scan_pamacd_signals(symbol: str, df: pd.DataFrame) -> List[Dict]:
    """
    PA_MACD BuyStop: scan latest bar.
    LONG: today green, prev red, close > prev high, MACD hist > 0
    SHORT: today red, prev green, close < prev low, MACD hist < 0
    Returns pending stop signals for next day.
    """
    if len(df) < 30:
        return []
    i = len(df) - 1

    o_today = df['open'].iloc[i]
    c_today = df['close'].iloc[i]
    o_prev = df['open'].iloc[i - 1]
    c_prev = df['close'].iloc[i - 1]
    h_prev = df['high'].iloc[i - 1]
    l_prev = df['low'].iloc[i - 1]
    macd_hist = df['macd_hist'].iloc[i]

    if pd.isna(macd_hist):
        return []

    signals = []
    today_green = c_today > o_today
    today_red = c_today < o_today
    prev_green = c_prev > o_prev
    prev_red = c_prev < o_prev

    if today_green and prev_red and c_today > h_prev and macd_hist > 0:
        entry_stop = h_prev
        sl = l_prev
        risk = entry_stop - sl
        if risk > 0:
            tp = entry_stop + 3 * risk
            signals.append({
                'symbol': symbol, 'direction': 'LONG', 'strategy': 'PA_MACD',
                'stop_level': round(entry_stop, 2), 'sl_price': round(sl, 2),
                'tp_price': round(tp, 2), 'max_hold_days': 10,
                'risk': round(risk, 2), 'rr_ratio': 3.0,
            })

    if today_red and prev_green and c_today < l_prev and macd_hist < 0:
        entry_stop = l_prev
        sl = h_prev
        risk = sl - entry_stop
        if risk > 0:
            tp = entry_stop - 3 * risk
            signals.append({
                'symbol': symbol, 'direction': 'SHORT', 'strategy': 'PA_MACD',
                'stop_level': round(entry_stop, 2), 'sl_price': round(sl, 2),
                'tp_price': round(tp, 2), 'max_hold_days': 10,
                'risk': round(risk, 2), 'rr_ratio': 3.0,
            })

    return signals


def scan_range_breakout_signals(symbol: str, df: pd.DataFrame) -> List[Dict]:
    """
    RangeBreakout 5d: check if today's bar broke 5d high/low (close-based).
    Returns immediate signals (already triggered).
    """
    if len(df) < 6:
        return []
    i = len(df) - 1

    highs_5d = df['high'].iloc[i - 5:i].max()
    lows_5d = df['low'].iloc[i - 5:i].min()
    c_today = df['close'].iloc[i]

    # First-breakout-only: yesterday's high must be below 5d high
    h_prev = df['high'].iloc[i - 1]
    l_prev = df['low'].iloc[i - 1]

    signals = []

    # LONG: close above 5d high + first breakout
    if c_today > highs_5d and h_prev < highs_5d:
        entry = highs_5d
        sl = lows_5d
        risk = entry - sl
        if risk > 0:
            tp = entry + 3 * risk
            signals.append({
                'symbol': symbol, 'direction': 'LONG', 'strategy': 'RangeBreak5d',
                'entry_price': round(entry, 2), 'sl_price': round(sl, 2),
                'tp_price': round(tp, 2), 'max_hold_days': 15,
                'risk': round(risk, 2), 'rr_ratio': 3.0,
                'range_high': round(highs_5d, 2), 'range_low': round(lows_5d, 2),
            })

    # SHORT: close below 5d low + first breakout
    if c_today < lows_5d and l_prev > lows_5d:
        entry = lows_5d
        sl = highs_5d
        risk = sl - entry
        if risk > 0:
            tp = entry - 3 * risk
            signals.append({
                'symbol': symbol, 'direction': 'SHORT', 'strategy': 'RangeBreak5d',
                'entry_price': round(entry, 2), 'sl_price': round(sl, 2),
                'tp_price': round(tp, 2), 'max_hold_days': 15,
                'risk': round(risk, 2), 'rr_ratio': 3.0,
                'range_high': round(highs_5d, 2), 'range_low': round(lows_5d, 2),
            })

    return signals


def check_exits(positions: List[Dict], symbol_data: Dict[str, pd.DataFrame],
                config: dict) -> List[Dict]:
    """Check SL/TP/MaxHold for active positions."""
    exits = []
    today = datetime.now().strftime('%Y-%m-%d')

    for pos in positions:
        sym = pos['symbol']
        if sym not in symbol_data:
            continue
        df = symbol_data[sym]
        if len(df) == 0:
            continue

        latest = df.iloc[-1]
        h = latest['high']
        l = latest['low']
        c = latest['close']

        direction = pos['direction']
        sl = pos['sl_price']
        tp = pos['tp_price']
        entry = pos['entry_price']
        entry_date = pos['entry_date'][:10]

        # Calculate hold days
        try:
            hold_days = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
        except Exception:
            hold_days = 0

        exit_signal = None

        if direction == 'LONG':
            if l <= sl:
                exit_signal = {'reason': 'STOP_LOSS', 'price': sl}
            elif h >= tp:
                exit_signal = {'reason': 'TAKE_PROFIT', 'price': tp}
            elif hold_days >= pos.get('max_hold_days', 10):
                exit_signal = {'reason': 'MAX_HOLD', 'price': c}
        else:  # SHORT
            if h >= sl:
                exit_signal = {'reason': 'STOP_LOSS', 'price': sl}
            elif l <= tp:
                exit_signal = {'reason': 'TAKE_PROFIT', 'price': tp}
            elif hold_days >= pos.get('max_hold_days', 10):
                exit_signal = {'reason': 'MAX_HOLD', 'price': c}

        if exit_signal:
            exits.append({
                'position_id': pos['id'],
                'symbol': sym,
                'direction': direction,
                'exit_price': exit_signal['price'],
                'exit_reason': exit_signal['reason'],
                'hold_days': hold_days,
            })

    return exits


def check_pending_triggers(pending_signals: List[Dict],
                           symbol_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Check if any pending stop signals have been triggered by today's price."""
    triggered = []
    for sig in pending_signals:
        sym = sig['symbol']
        if sym not in symbol_data:
            continue
        df = symbol_data[sym]
        if len(df) == 0:
            continue

        latest = df.iloc[-1]
        h = latest['high']
        l = latest['low']

        if sig['direction'] == 'LONG' and h >= sig['stop_level']:
            triggered.append({**sig, 'entry_price': sig['stop_level']})
        elif sig['direction'] == 'SHORT' and l <= sig['stop_level']:
            triggered.append({**sig, 'entry_price': sig['stop_level']})

    return triggered


def run_full_scan(symbol_data: Dict[str, pd.DataFrame],
                  config: dict) -> Dict:
    """
    Complete scan pipeline:
    1. Compute indicators for all symbols
    2. Scan for PA_MACD pending signals
    3. Scan for RangeBreakout immediate signals
    Returns scan result dict.
    """
    pamacd_signals = []
    rb_signals = []

    for sym, df in symbol_data.items():
        df = compute_indicators(df)
        symbol_data[sym] = df

        pamacd_signals.extend(scan_pamacd_signals(sym, df))
        rb_signals.extend(scan_range_breakout_signals(sym, df))

    result = {
        'scan_time': datetime.now().isoformat(),
        'symbols_scanned': len(symbol_data),
        'pamacd_signals': pamacd_signals,
        'rb_signals': rb_signals,
        'total_signals': len(pamacd_signals) + len(rb_signals),
    }

    logger.info(f"[Trident] Scan complete: {len(pamacd_signals)} PA_MACD, "
                f"{len(rb_signals)} RangeBreak signals")
    return result
