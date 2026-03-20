"""
BNF Squeeze & Fire — Scanner
==============================
Computes BB squeeze/fire indicators on BankNifty daily bars.
Detects signals for both modes:
  - SQUEEZE: BB contracting → sell strangles (non-directional)
  - FIRE: BB expanding after squeeze → sell naked options (directional)

Uses same indicator logic as backtest (run_fire_naked_sell.py + nondirectional_simulator.py).
"""

import logging
import sqlite3
import calendar
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Constants
LOT_SIZE = 15           # BankNifty lot size
STRIKE_STEP = 100       # BankNifty strike interval
RISK_FREE_RATE = 0.065  # India risk-free rate
TRADING_DAYS_PER_YEAR = 252


# ─── Black-Scholes ──────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_iv(atr, close_price):
    if atr <= 0 or close_price <= 0:
        return 0.15
    daily_std = (atr / close_price) / 1.2
    return max(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 1.20, 0.08)


def round_strike(price):
    return round(price / STRIKE_STEP) * STRIKE_STEP


# ─── Expiry Logic ───────────────────────────────────────────

def get_last_thursday(year, month):
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:
        d -= timedelta(days=1)
    return d


def get_monthly_expiry(entry_date, min_dte=15):
    if isinstance(entry_date, str):
        entry_date = date.fromisoformat(entry_date[:10])
    expiry = get_last_thursday(entry_date.year, entry_date.month)
    if expiry <= entry_date:
        m = (entry_date.month % 12) + 1
        y = entry_date.year + (1 if entry_date.month == 12 else 0)
        expiry = get_last_thursday(y, m)
    dte = (expiry - entry_date).days
    if dte < min_dte:
        m = (expiry.month % 12) + 1
        y = expiry.year + (1 if expiry.month == 12 else 0)
        expiry = get_last_thursday(y, m)
        dte = (expiry - entry_date).days
    return dte, expiry


# ─── Scanner ────────────────────────────────────────────────

class BnfScanner:
    """
    Scans BankNifty daily bars for squeeze and fire signals.

    Config keys (from BNF_DEFAULTS):
      bb_period, atr_period, sma_period, squeeze_min_bars, trend_atr_min,
      squeeze_strike_atr, fire_strike_atr, fire_hold_bars, fire_sl_mult,
      fire_max_loss_rupees, squeeze_hold_bars, squeeze_max_loss_rupees
    """

    def __init__(self, config: dict):
        self.cfg = config

    def load_daily_bars(self, lookback_days=120):
        """Load recent BankNifty daily bars from market_data.db."""
        from config import MARKET_DATA_DB

        conn = sqlite3.connect(str(MARKET_DATA_DB))
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        df = pd.read_sql_query(
            """SELECT date, open, high, low, close, volume
               FROM market_data_unified
               WHERE symbol='BANKNIFTY' AND timeframe='day'
               AND date >= ? ORDER BY date""",
            conn, params=(cutoff,))
        conn.close()

        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} BankNifty daily bars")
        return df

    def compute_indicators(self, df):
        """Compute BB squeeze/fire indicators on daily OHLC DataFrame."""
        cfg = self.cfg
        bb_period = cfg.get('bb_period', 10)
        atr_period = cfg.get('atr_period', 14)
        sma_period = cfg.get('sma_period', 20)
        squeeze_min = cfg.get('squeeze_min_bars', 3)
        trend_atr_min = cfg.get('trend_atr_min', 0.5)

        high, low, close = df['high'], df['low'], df['close']

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_period).mean()

        # Bollinger Bands
        sma_bb = close.rolling(bb_period).mean()
        std_bb = close.rolling(bb_period).std()
        df['bb_upper'] = sma_bb + 2.0 * std_bb
        df['bb_lower'] = sma_bb - 2.0 * std_bb

        # BB Width & Width MA
        bb_width = (2 * 2.0 * std_bb) / sma_bb * 100
        bb_width_ma = bb_width.rolling(bb_period).mean()
        df['bb_width'] = bb_width
        df['bb_width_ma'] = bb_width_ma

        # Squeeze / Expand
        df['is_squeezing'] = (bb_width < bb_width_ma).astype(int)
        df['is_expanding'] = (bb_width > bb_width_ma).astype(int)

        # Consecutive squeeze count
        squeeze_groups = (df['is_squeezing'] != df['is_squeezing'].shift(1)).cumsum()
        df['squeeze_count'] = df.groupby(squeeze_groups)['is_squeezing'].cumsum()

        # Squeeze Fire: expanding NOW + was squeezing for N bars before
        sq_sum = df['is_squeezing'].rolling(squeeze_min).sum()
        df['squeeze_fire'] = ((df['is_expanding'] == 1) &
                              (sq_sum.shift(1) >= squeeze_min)).astype(int)

        # SMA for direction
        df['sma'] = close.rolling(sma_period).mean()

        # Trend strength
        df['trend_strength'] = (close - df['sma']).abs() / df['atr']

        # Trending filter
        if trend_atr_min > 0:
            df['is_trending'] = (df['trend_strength'] >= trend_atr_min).astype(int)
        else:
            df['is_trending'] = 1

        # Direction
        df['direction'] = np.where(close > df['sma'], 'BULLISH', 'BEARISH')

        # Fire signals (directional)
        df['fire_long'] = ((df['squeeze_fire'] == 1) &
                           (df['is_trending'] == 1) &
                           (close > df['sma'])).astype(int)
        df['fire_short'] = ((df['squeeze_fire'] == 1) &
                            (df['is_trending'] == 1) &
                            (close < df['sma'])).astype(int)

        # Squeeze signal (non-directional) — BB is squeezing
        df['squeeze_signal'] = df['is_squeezing']

        # IV estimate
        df['est_iv'] = df.apply(
            lambda r: estimate_iv(r['atr'], r['close'])
            if r['atr'] > 0 and not np.isnan(r['atr']) else 0.15, axis=1)

        return df

    def scan(self, df=None):
        """
        Run full scan. Returns dict with current state and any signals.

        Returns:
            {
                'bb_state': 'SQUEEZING' | 'EXPANDING',
                'squeeze_count': int,
                'bb_width': float,
                'bb_width_ma': float,
                'direction': 'BULLISH' | 'BEARISH',
                'trend_strength': float,
                'is_trending': bool,
                'spot': float,
                'atr': float,
                'sma': float,
                'iv': float,
                'signals': [
                    {'type': 'FIRE_LONG' | 'FIRE_SHORT' | 'SQUEEZE_ENTRY', ...}
                ],
                'squeeze_strikes': {'call': int, 'put': int} | None,
                'fire_strike': int | None,
            }
        """
        if df is None:
            df = self.load_daily_bars()

        df = self.compute_indicators(df)

        if len(df) < 30:
            return {'error': 'Not enough data', 'signals': []}

        latest = df.iloc[-1]
        cfg = self.cfg

        result = {
            'bb_state': 'SQUEEZING' if latest['is_squeezing'] else 'EXPANDING',
            'squeeze_count': int(latest['squeeze_count']) if latest['is_squeezing'] else 0,
            'bb_width': round(float(latest['bb_width']), 2),
            'bb_width_ma': round(float(latest['bb_width_ma']), 2),
            'direction': str(latest['direction']),
            'trend_strength': round(float(latest['trend_strength']), 2),
            'is_trending': bool(latest['is_trending']),
            'spot': round(float(latest['close']), 2),
            'atr': round(float(latest['atr']), 2),
            'sma': round(float(latest['sma']), 2),
            'iv': round(float(latest['est_iv']), 4),
            'date': str(latest['date'])[:10],
            'signals': [],
            'squeeze_strikes': None,
            'fire_strike': None,
        }

        spot = latest['close']
        atr = latest['atr']
        iv = latest['est_iv']

        # Squeeze mode: compute strangle strikes
        sq_strike_atr = cfg.get('squeeze_strike_atr', 1.5)
        call_strike = round_strike(spot + sq_strike_atr * atr)
        put_strike = round_strike(spot - sq_strike_atr * atr)
        result['squeeze_strikes'] = {'call': call_strike, 'put': put_strike}

        # Fire mode: compute directional strike
        fire_strike_atr = cfg.get('fire_strike_atr', 0.5)
        if latest['direction'] == 'BULLISH':
            fire_strike = round_strike(spot - fire_strike_atr * atr)
        else:
            fire_strike = round_strike(spot + fire_strike_atr * atr)
        result['fire_strike'] = fire_strike

        # Get expiry info
        try:
            today = date.today()
            dte, expiry = get_monthly_expiry(today, min_dte=15)
            T = max(dte, 1) / 365.0
            result['expiry'] = str(expiry)
            result['dte'] = dte
        except Exception:
            dte = 20
            T = dte / 365.0
            result['expiry'] = 'unknown'
            result['dte'] = dte

        # Check for fire signals
        if latest['fire_long']:
            premium = bs_put(spot, fire_strike, T, RISK_FREE_RATE, iv)
            result['signals'].append({
                'type': 'FIRE_LONG',
                'action': 'SELL PUT',
                'strike': fire_strike,
                'premium': round(premium, 2),
                'hold_bars': cfg.get('fire_hold_bars', 7),
                'sl_mult': cfg.get('fire_sl_mult', 3.0),
                'max_loss': cfg.get('fire_max_loss_rupees', 20000),
            })

        if latest['fire_short']:
            premium = bs_call(spot, fire_strike, T, RISK_FREE_RATE, iv)
            result['signals'].append({
                'type': 'FIRE_SHORT',
                'action': 'SELL CALL',
                'strike': fire_strike,
                'premium': round(premium, 2),
                'hold_bars': cfg.get('fire_hold_bars', 7),
                'sl_mult': cfg.get('fire_sl_mult', 3.0),
                'max_loss': cfg.get('fire_max_loss_rupees', 20000),
            })

        # Squeeze entry signal — only if squeezing AND no active squeeze positions
        if latest['is_squeezing'] and latest['squeeze_count'] >= cfg.get('squeeze_min_bars', 3):
            call_prem = bs_call(spot, call_strike, T, RISK_FREE_RATE, iv)
            put_prem = bs_put(spot, put_strike, T, RISK_FREE_RATE, iv)
            result['signals'].append({
                'type': 'SQUEEZE_ENTRY',
                'action': 'SELL STRANGLE',
                'call_strike': call_strike,
                'put_strike': put_strike,
                'call_premium': round(call_prem, 2),
                'put_premium': round(put_prem, 2),
                'total_premium': round(call_prem + put_prem, 2),
                'hold_bars': cfg.get('squeeze_hold_bars', 10),
                'max_loss': cfg.get('squeeze_max_loss_rupees', 30000),
            })

        return result

    def check_exits(self, positions, current_spot, current_iv=None):
        """
        Check if any active positions need to exit.

        Returns list of (position_id, exit_reason, est_exit_price).
        """
        cfg = self.cfg
        exits = []

        today = date.today()

        for pos in positions:
            exit_reason = None
            est_exit_price = None

            # Check hold bars
            bars_remaining = pos.get('hold_bars_remaining', 0)
            if bars_remaining is not None and bars_remaining <= 0:
                exit_reason = 'hold_complete'

            # Check expiry
            if pos.get('expiry_date'):
                try:
                    exp = date.fromisoformat(str(pos['expiry_date'])[:10])
                    if today >= exp:
                        exit_reason = 'expiry'
                except (ValueError, TypeError):
                    pass

            # Check SL for fire mode (option value > sl_mult * premium)
            if pos['mode'] == 'FIRE' and pos.get('sl_price') and not exit_reason:
                # Estimate current option value
                try:
                    exp_str = str(pos.get('expiry_date', ''))[:10]
                    exp_date = date.fromisoformat(exp_str) if exp_str else today + timedelta(days=15)
                    rem_dte = max((exp_date - today).days, 1)
                    T = rem_dte / 365.0
                    iv = current_iv or estimate_iv(cfg.get('_last_atr', 500), current_spot)

                    strike = pos['strike']
                    if pos['instrument_type'] == 'PE':
                        current_val = bs_put(current_spot, strike, T, RISK_FREE_RATE, iv)
                    else:
                        current_val = bs_call(current_spot, strike, T, RISK_FREE_RATE, iv)

                    if current_val >= pos['sl_price']:
                        exit_reason = 'stop_loss'
                        est_exit_price = round(current_val, 2)
                except Exception as e:
                    logger.warning(f"SL check error for pos {pos['id']}: {e}")

            if exit_reason:
                exits.append((pos['id'], exit_reason, est_exit_price))

        return exits
