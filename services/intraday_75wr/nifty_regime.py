"""Live NIFTY regime computation for the intraday-75WR engine.

Ports research/37 build_nifty_regime() to live: pulls today's NIFTY 50 5-min
candles via the data_manager (or Kite directly if not yet ingested) and
computes the regime tags used by every config's NIFTY filter:

    {
        'gap_pct':         pct gap-up / gap-down at open
        'day_open':        first 5-min open
        'prev_close':      last close from previous trading day
        'b3_close':        bar_idx=3 close (09:30 IST)
        'b3_vwap':         VWAP at end of bar 3
        'b3_rsi':          RSI(14) at bar 3
        'b3_change_pct':   (b3_close - day_open)/day_open * 100
        'below_vwap_b3':   bool, b3_close < b3_vwap
        'below_vwap_b6':   bool, b6_close < b6_vwap
        'b6_close':        bar_idx=6 close (09:45 IST)
        'first_bearish':   bar_idx=0 close < open
        'first_bullish':   bar_idx=0 close > open
        'current_close':   most-recent 5-min close (used by Config C)
        'current_vwap':    VWAP at most-recent 5-min bar
    }

Regime is *frozen at 09:45* for Config A1/B1. For A2/A3/C, called per-bar.
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, time
from typing import Optional

import numpy as np
import pandas as pd

from services.intraday_75wr.signal_lib import _indicators

logger = logging.getLogger(__name__)


def _fetch_5min_from_db(symbol: str, lookback_days: int = 5) -> pd.DataFrame:
    """Pull symbol's 5-min OHLCV from market_data.db.
    Falls back to empty DataFrame on any error."""
    try:
        import sqlite3
        from config import MARKET_DATA_DB
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        conn = sqlite3.connect(str(MARKET_DATA_DB))
        try:
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume '
                'FROM market_data_unified '
                'WHERE symbol=? AND timeframe=? AND date BETWEEN ? AND ? '
                'ORDER BY date',
                conn,
                params=(symbol, '5minute', start.isoformat(), end.isoformat()),
            )
        finally:
            conn.close()
        if df.empty:
            return df
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df
    except Exception as e:
        logger.warning(f'[nifty_regime] DB fetch failed: {e}')
        return pd.DataFrame()


def _fetch_5min_from_kite(symbol: str, lookback_days: int = 5) -> pd.DataFrame:
    """Fallback: direct Kite API fetch. Used if market_data.db is stale."""
    try:
        from services.kite_service import get_kite
        kite = get_kite()
        # Resolve token
        instruments = kite.instruments('NSE') or []
        # NIFTY 50 has tradingsymbol 'NIFTY 50' and segment INDICES
        token = None
        for ins in instruments:
            if ins.get('tradingsymbol') == symbol:
                token = ins['instrument_token']
                break
        if token is None:
            ins_idx = kite.instruments('INDICES') or []
            for ins in ins_idx:
                if ins.get('tradingsymbol') in (symbol, symbol.replace(' ', '')):
                    token = ins['instrument_token']
                    break
        if token is None:
            return pd.DataFrame()
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        candles = kite.historical_data(
            instrument_token=token, from_date=start, to_date=end,
            interval='5minute', oi=False,
        )
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.warning(f'[nifty_regime] Kite fetch failed for {symbol}: {e}')
        return pd.DataFrame()


def fetch_nifty_5min(today: Optional[date] = None) -> pd.DataFrame:
    """Pull NIFTY 5-min for today + a few days back (for prev_close)."""
    # Try DB first (faster)
    df = _fetch_5min_from_db('NIFTY50', lookback_days=5)
    if df.empty:
        df = _fetch_5min_from_db('NIFTY 50', lookback_days=5)
    if df.empty:
        df = _fetch_5min_from_kite('NIFTY 50', lookback_days=5)
    return df


def compute_regime(today: Optional[date] = None,
                   df_nifty: Optional[pd.DataFrame] = None) -> dict:
    """Compute the NIFTY regime tags used by all 4 signals.

    df_nifty is optional — if not provided, fetched from DB / Kite.
    Caller can pass a pre-built DataFrame in tests.
    """
    if today is None:
        today = date.today()
    if df_nifty is None:
        df_nifty = fetch_nifty_5min(today)

    if df_nifty is None or df_nifty.empty:
        logger.warning('[nifty_regime] empty NIFTY data — regime will be empty')
        return {}

    df = _indicators.enrich(df_nifty.copy())

    # Today's session
    today_df = df[df['session'] == today].copy()
    if today_df.empty:
        # No NIFTY data for today (early in the session, or weekend) — return empty
        logger.warning(
            f'[nifty_regime] no NIFTY 5-min rows for {today} — empty regime'
        )
        return {}

    # Pre-day prev_close — last close of previous session
    prev_sessions = sorted(set(df['session'].unique()) - {today})
    prev_close = None
    if prev_sessions:
        prev_session = prev_sessions[-1]
        prev_close = float(df[df['session'] == prev_session]['close'].iloc[-1])

    day_open = float(today_df['open'].iloc[0])

    # bar 0
    bar0 = today_df.iloc[0]
    first_close = float(bar0['close'])
    first_open = float(bar0['open'])
    first_bearish = first_close < first_open
    first_bullish = first_close > first_open

    # bar 3
    bar3 = today_df[today_df['bar_idx'] == 3]
    bar3_close = float(bar3.iloc[0]['close']) if not bar3.empty else None
    bar3_vwap = float(bar3.iloc[0]['vwap']) if not bar3.empty else None
    bar3_rsi = float(bar3.iloc[0]['rsi']) if not bar3.empty else None

    # bar 6
    bar6 = today_df[today_df['bar_idx'] == 6]
    bar6_close = float(bar6.iloc[0]['close']) if not bar6.empty else None
    bar6_vwap = float(bar6.iloc[0]['vwap']) if not bar6.empty else None

    # current bar
    last = today_df.iloc[-1]
    current_close = float(last['close'])
    current_vwap = float(last['vwap'])

    out = {
        'today': str(today),
        'gap_pct': (
            (day_open - prev_close) / prev_close * 100.0
            if prev_close else None
        ),
        'day_open': day_open,
        'prev_close': prev_close,
        'b3_close': bar3_close,
        'b3_vwap': bar3_vwap,
        'b3_rsi': bar3_rsi,
        'b3_change_pct': (
            (bar3_close - day_open) / day_open * 100.0
            if bar3_close else None
        ),
        'below_vwap_b3': (
            bool(bar3_close < bar3_vwap)
            if bar3_close is not None and bar3_vwap is not None else None
        ),
        'b6_close': bar6_close,
        'b6_vwap': bar6_vwap,
        'below_vwap_b6': (
            bool(bar6_close < bar6_vwap)
            if bar6_close is not None and bar6_vwap is not None else None
        ),
        'first_bearish': bool(first_bearish),
        'first_bullish': bool(first_bullish),
        'current_close': current_close,
        'current_vwap': current_vwap,
    }
    return out
