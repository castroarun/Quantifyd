"""
Nifty ORB Strangle — Scanner / Pure-Compute Layer
==================================================

Signal-detection helpers for the multi-variant Nifty ORB strangle paper system.
This module is intentionally side-effect free: it loads 5-min NIFTY bars
(from the local DB or the Kite API) and returns plain dicts describing the
state of each variant on a given day.

Reuses the OR + RSI patterns established in:
    backtest_nifty_orb_pct_sl.py
    backtest_nifty_orb_phase1e.py
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import MARKET_DATA_DB, STRANGLE_DEFAULTS

logger = logging.getLogger(__name__)


SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)


# =============================================================================
# Indicators
# =============================================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI on a close series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# =============================================================================
# Data loading
# =============================================================================

def _or_end_time(or_min: int) -> dtime:
    total = 9 * 60 + 15 + or_min
    return dtime(total // 60, total % 60)


def load_nifty_5min_local(start_date: date = None,
                          end_date: date = None) -> pd.DataFrame:
    """Load NIFTY50 5-min OHLC from market_data.db (local). Empty df if missing."""
    conn = sqlite3.connect(str(MARKET_DATA_DB))
    try:
        params = ['NIFTY50']
        q = ("SELECT date, open, high, low, close FROM market_data_unified "
             "WHERE symbol=? AND timeframe='5minute'")
        if start_date:
            q += " AND date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            q += " AND date <= ?"
            params.append((end_date + timedelta(days=1)).isoformat())
        q += " ORDER BY date"
        df = pd.read_sql(q, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    df = df.between_time(SESSION_OPEN, SESSION_CLOSE)
    df['rsi5m'] = rsi(df['close'], 14)
    return df


def load_nifty_5min_kite(lookback_days: int = 3) -> pd.DataFrame:
    """Load NIFTY 5-min from Kite (intraday). Falls back to empty df on error."""
    try:
        from services.kite_service import get_kite
        kite = get_kite()
        instrument_token = 256265   # NIFTY 50 index
        to_date = datetime.now()
        from_date = to_date - timedelta(days=lookback_days)
        rows = []
        chunk_start = from_date
        while chunk_start < to_date:
            chunk_end = min(chunk_start + timedelta(days=7), to_date)
            try:
                candles = kite.historical_data(
                    instrument_token, chunk_start, chunk_end,
                    interval='5minute', oi=False)
                for c in candles:
                    rows.append({'date': c['date'], 'open': c['open'],
                                 'high': c['high'], 'low': c['low'],
                                 'close': c['close']})
            except Exception as e:
                logger.warning(f"[Strangle] Kite 5-min chunk failed: {e}")
            chunk_start = chunk_end
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.between_time(SESSION_OPEN, SESSION_CLOSE)
        df['rsi5m'] = rsi(df['close'], 14)
        return df
    except Exception as e:
        logger.warning(f"[Strangle] Kite 5-min unavailable: {e}")
        return pd.DataFrame()


# =============================================================================
# OR detection
# =============================================================================

def detect_or_window(intraday_5min_df: pd.DataFrame, or_min: int,
                     trade_date: date = None) -> Optional[Tuple[float, float, datetime]]:
    """
    Compute the opening range for a given window (in minutes).
    Returns (or_high, or_low, or_end_ts) or None if not enough bars.
    `intraday_5min_df` must be 5-min bars indexed by datetime, RSI optional.
    `trade_date` defaults to today (uses index.date()).
    """
    if intraday_5min_df is None or intraday_5min_df.empty:
        return None
    if trade_date is None:
        trade_date = datetime.now().date()

    sess = intraday_5min_df[intraday_5min_df.index.normalize() ==
                            pd.Timestamp(trade_date)]
    if sess.empty:
        return None
    or_end = _or_end_time(or_min)
    or_bars = sess.between_time(SESSION_OPEN, or_end)
    if len(or_bars) < 1:
        return None
    or_high = float(or_bars['high'].max())
    or_low = float(or_bars['low'].min())
    or_end_ts = or_bars.index[-1].to_pydatetime()
    return or_high, or_low, or_end_ts


# =============================================================================
# CPR (Central Pivot Range)
# =============================================================================

def compute_cpr(prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
    """Standard floor-trader CPR from previous trading day's H/L/C.

    P  = (H + L + C) / 3
    BC = (H + L) / 2
    TC = 2*P - BC
    CPR_low  = min(BC, TC)
    CPR_high = max(BC, TC)
    """
    p = (prev_high + prev_low + prev_close) / 3.0
    bc = (prev_high + prev_low) / 2.0
    tc = 2 * p - bc
    return {
        'P': p,
        'BC': bc,
        'TC': tc,
        'CPR_low': min(bc, tc),
        'CPR_high': max(bc, tc),
        'CPR_width_pct': abs(tc - bc) / p * 100.0 if p else 0.0,
    }


def check_cpr_against_filter(direction: str, entry_close: float,
                             cpr: Dict[str, float]) -> bool:
    """Return True if signal PASSES the CPR-against filter (i.e. is allowed).

    A break passes when it has NOT yet cleared the CPR zone in its direction:
      LONG break  : entry close <= CPR_high   (still inside or below CPR)
      SHORT break : entry close >= CPR_low    (still inside or above CPR)

    The hypothesis (validated in `backtest_nifty_orb_cpr.py`) is that
    "tame" breakouts that haven't escaped CPR are friendlier to a short-strangle
    P&L profile than fully-extended trend breaks.
    """
    if cpr is None:
        return True   # filter cannot run -> don't block (failsafe)
    if direction == 'LONG':
        return entry_close <= float(cpr['CPR_high'])
    if direction == 'SHORT':
        return entry_close >= float(cpr['CPR_low'])
    return True


def load_prev_day_hlc_local(symbol: str = 'NIFTY50',
                            ref_date: date = None) -> Optional[Tuple[float, float, float, date]]:
    """Look up the most recent NIFTY daily H/L/C strictly BEFORE ref_date
    from market_data.db. Returns (H, L, C, that_date) or None.
    """
    if ref_date is None:
        ref_date = datetime.now().date()
    conn = sqlite3.connect(str(MARKET_DATA_DB))
    try:
        row = conn.execute(
            "SELECT date, high, low, close FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' AND date < ? "
            "ORDER BY date DESC LIMIT 1",
            (symbol, ref_date.isoformat())
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    d_str, h, l, c = row
    try:
        d = datetime.fromisoformat(d_str.replace(' ', 'T')).date() \
            if 'T' in d_str or ' ' in d_str else datetime.strptime(d_str[:10], '%Y-%m-%d').date()
    except Exception:
        d = datetime.strptime(d_str[:10], '%Y-%m-%d').date()
    return float(h), float(l), float(c), d


def load_prev_day_hlc_kite(ref_date: date = None,
                           lookback_days: int = 7) -> Optional[Tuple[float, float, float, date]]:
    """Kite-based fallback: fetch NIFTY 50 daily candles for last `lookback_days`
    and return the most recent one strictly BEFORE ref_date.
    """
    if ref_date is None:
        ref_date = datetime.now().date()
    try:
        from services.kite_service import get_kite
        kite = get_kite()
        instrument_token = 256265   # NIFTY 50 index
        to_date = datetime.combine(ref_date, dtime(15, 30))
        from_date = to_date - timedelta(days=lookback_days)
        candles = kite.historical_data(instrument_token, from_date, to_date,
                                       interval='day', oi=False)
        for c in reversed(candles or []):
            cd = c['date'].date() if hasattr(c['date'], 'date') else c['date']
            if cd < ref_date:
                return float(c['high']), float(c['low']), float(c['close']), cd
    except Exception as e:
        logger.warning(f"[Strangle] Kite daily lookup failed: {e}")
    return None


# =============================================================================
# Entry signal detection
# =============================================================================

@dataclass
class EntrySignal:
    direction: str        # 'LONG' or 'SHORT'
    entry_ts: datetime
    entry_price: float
    or_high: float
    or_low: float
    or_width_pct: float
    rsi_at_entry: Optional[float]


def check_entry_signal(intraday_5min_df: pd.DataFrame,
                       variant_config: Dict,
                       trade_date: date = None,
                       no_entry_after: dtime = None,
                       cpr: Dict[str, float] = None) -> Optional[EntrySignal]:
    """
    Scan today's 5-min bars (after OR window closes) and return the first
    valid break that satisfies the variant's RSI filter, with K=0 strict
    confirmation. Returns None if no signal yet.

    K=0 strict: when a 5-min candle closes outside the OR boundary AND RSI
    confirms direction (per variant), entry happens immediately. If RSI
    doesn't confirm on the break candle, the variant skips this day's signal
    entirely (no waiting). For the no-RSI variant (rsi_lo_long is None),
    simply enter on the first 5-min close outside OR.

    Universal day filters (Q4 + per-variant calm filter) applied here too —
    if either fails, returns None.
    """
    if trade_date is None:
        trade_date = datetime.now().date()
    if no_entry_after is None:
        try:
            ne = STRANGLE_DEFAULTS.get('no_entry_after', '14:00')
            h, m = map(int, ne.split(':'))
            no_entry_after = dtime(h, m)
        except Exception:
            no_entry_after = dtime(14, 0)

    or_min = int(variant_config['or_min'])
    or_res = detect_or_window(intraday_5min_df, or_min, trade_date)
    if or_res is None:
        return None
    or_high, or_low, or_end_ts = or_res
    if or_low <= 0:
        return None
    or_width_pct = (or_high - or_low) / or_low * 100.0

    # Universal Q4 skip
    if variant_config.get('apply_q4_filter', True) and \
            or_width_pct > float(variant_config.get('q4_threshold_pct', 0.67)):
        return None
    # Calm-only filter (V8): need OR width below threshold (note: variant's own OR window)
    if variant_config.get('apply_calm_filter', False) and \
            or_width_pct >= float(variant_config.get('calm_threshold_pct', 0.40)):
        return None

    # Get post-OR bars for the day
    sess = intraday_5min_df[intraday_5min_df.index.normalize() ==
                            pd.Timestamp(trade_date)]
    if sess.empty:
        return None
    or_end_time = _or_end_time(or_min)
    post = sess.between_time(or_end_time, SESSION_CLOSE).iloc[1:]
    if post.empty:
        return None

    rsi_lo_long = variant_config.get('rsi_lo_long')      # None for no-RSI variant
    rsi_hi_short = variant_config.get('rsi_hi_short')

    for ts, row in post.iterrows():
        if ts.time() >= no_entry_after:
            return None
        close = float(row['close'])
        rsi_v = float(row['rsi5m']) if 'rsi5m' in row and pd.notna(row['rsi5m']) else None

        direction = None
        if close > or_high:
            direction = 'LONG'
        elif close < or_low:
            direction = 'SHORT'
        else:
            continue

        # K=0 strict: confirm or skip
        if rsi_lo_long is None and rsi_hi_short is None:
            # No RSI filter (V6)
            confirmed = True
        elif direction == 'LONG':
            confirmed = (rsi_v is not None and rsi_v > float(rsi_lo_long))
        else:
            confirmed = (rsi_v is not None and rsi_v < float(rsi_hi_short))

        if not confirmed:
            # K=0 strict: skip the day's signal
            return None

        # CPR-against filter (V9/V10): require break NOT to have cleared CPR
        if variant_config.get('apply_cpr_against_filter', False):
            if not check_cpr_against_filter(direction, close, cpr):
                # Break cleared CPR -> too directional, skip entry
                return None

        return EntrySignal(
            direction=direction,
            entry_ts=ts.to_pydatetime(),
            entry_price=close,
            or_high=or_high,
            or_low=or_low,
            or_width_pct=or_width_pct,
            rsi_at_entry=rsi_v,
        )
    return None


# =============================================================================
# SL / EOD checks
# =============================================================================

def compute_sl_price(direction: str, or_high: float, or_low: float,
                     multiplier: float = 1.0) -> float:
    """SL price for the underlying (paper):
       LONG break: SL = OR_low (multiplier=1.0); below this, the break is invalidated
       SHORT break: SL = OR_high
    Multiplier scales the OR-width retracement (e.g. 1.0 = full opposite OR).
    """
    if direction == 'LONG':
        # SL at OR_high - mult * (OR_high - OR_low). mult=1.0 -> SL=OR_low
        return or_high - multiplier * (or_high - or_low)
    else:
        return or_low + multiplier * (or_high - or_low)


def is_sl_breached(direction: str, sl_price: float, ltp: float) -> bool:
    """LONG: LTP <= SL. SHORT: LTP >= SL."""
    if direction == 'LONG':
        return ltp <= sl_price
    return ltp >= sl_price
