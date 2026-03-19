"""
Comprehensive Strategy Research Scanner
========================================

Scans 100+ indicator combinations across 395 stocks to find strategies with:
- 250+ trades
- 65%+ win rate
- 1+ Calmar ratio
- Good profit factor (>1.5)

Uses top-down approach (weekly criteria → daily entries) and
multi-indicator combinations based on deep research.

Independent of consolidation/Darvas/box strategies.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import json
import time
import warnings
import multiprocessing as mp
from functools import partial
import traceback
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================================
# DATABASE
# ============================================================================

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'
RESULTS_PATH = Path(__file__).parent / 'backtest_data' / 'strategy_research_results.json'
CSV_RESULTS_PATH = Path(__file__).parent / 'strategy_research_results.csv'

def load_all_daily_data() -> Dict[str, pd.DataFrame]:
    """Load all daily OHLCV data from DB into memory."""
    conn = sqlite3.connect(DB_PATH)

    # Get all symbols
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day' ORDER BY symbol",
        conn
    )['symbol'].tolist()

    print(f"Loading daily data for {len(symbols)} stocks...")

    all_data = {}
    for sym in symbols:
        df = pd.read_sql_query(
            """SELECT date, open, high, low, close, volume
               FROM market_data_unified
               WHERE symbol=? AND timeframe='day'
               ORDER BY date""",
            conn, params=[sym]
        )
        if len(df) < 200:  # Need at least 200 days for indicators
            continue
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        all_data[sym] = df

    conn.close()
    print(f"Loaded {len(all_data)} stocks with 200+ days of data")
    return all_data


# ============================================================================
# INDICATOR CALCULATIONS (Vectorized for speed)
# ============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def dema(series: pd.Series, period: int) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2

def tema(series: pd.Series, period: int) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3

def hull_ma(series: pd.Series, period: int) -> pd.Series:
    half_period = max(int(period / 2), 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    wma_half = wma(series, half_period)
    wma_full = wma(series, period)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_period)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    return (((df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)) * 100).fillna(50)

def stochastic_d(stoch_k: pd.Series, period: int = 3) -> pd.Series:
    return stoch_k.rolling(window=period).mean()

def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hh = df['high'].rolling(window=period).max()
    ll = df['low'].rolling(window=period).min()
    return (-100 * (hh - df['close']) / (hh - ll).replace(0, np.nan)).fillna(-50)

def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    s = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((tp - s) / (0.015 * mad)).fillna(0)

def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos = mf.where(tp > tp.shift(1), 0)
    neg = mf.where(tp <= tp.shift(1), 0)
    pos_sum = pos.rolling(window=period).sum()
    neg_sum = neg.rolling(window=period).sum()
    return (100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))).fillna(50)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h_l = df['high'] - df['low']
    h_c = (df['high'] - df['close'].shift(1)).abs()
    l_c = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI)"""
    h = df['high']
    l = df['low']
    c = df['close']
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    plus_dm = h.diff().where((h.diff() > -l.diff()) & (h.diff() > 0), 0)
    minus_dm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    return adx_val.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0):
    """Returns (middle, upper, lower, pctb, width)"""
    mid = series.rolling(window=period).mean()
    s = series.rolling(window=period).std()
    upper = mid + std * s
    lower = mid - std * s
    pctb = (series - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / mid * 100
    return mid, upper, lower, pctb.fillna(0.5), width.fillna(0)

def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, mult: float = 2.0):
    """Returns (middle, upper, lower)"""
    mid = ema(df['close'], ema_period)
    a = atr(df, atr_period)
    return mid, mid + mult * a, mid - mult * a

def donchian(df: pd.DataFrame, period: int = 20):
    """Returns (upper, lower, middle)"""
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    return upper, lower, (upper + lower) / 2

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (vectorized)"""
    sign = np.sign(df['close'].diff())
    return (sign * df['volume']).cumsum()

def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change"""
    return ((series / series.shift(period)) - 1) * 100

def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow"""
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    mfv = mfm * df['volume']
    return (mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()).fillna(0)

def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
    """True Strength Index"""
    diff = series.diff()
    double_smooth = ema(ema(diff, long_period), short_period)
    double_smooth_abs = ema(ema(diff.abs(), long_period), short_period)
    return (100 * double_smooth / double_smooth_abs.replace(0, np.nan)).fillna(0)

def supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Returns direction: 1=uptrend, -1=downtrend"""
    a = atr(df, atr_period)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a

    direction = pd.Series(1, index=df.index, dtype=int)
    st = pd.Series(lower.iloc[0], index=df.index, dtype=float)

    for i in range(1, len(df)):
        if df['close'].iloc[i] > st.iloc[i-1]:
            st.iloc[i] = lower.iloc[i]
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < st.iloc[i-1]:
            st.iloc[i] = upper.iloc[i]
            direction.iloc[i] = -1
        else:
            st.iloc[i] = st.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1 and lower.iloc[i] > st.iloc[i]:
                st.iloc[i] = lower.iloc[i]
            elif direction.iloc[i] == -1 and upper.iloc[i] < st.iloc[i]:
                st.iloc[i] = upper.iloc[i]

    return direction

def psar_direction(df: pd.DataFrame, af_start=0.02, af_inc=0.02, af_max=0.2) -> pd.Series:
    """Parabolic SAR direction: 1=uptrend, -1=downtrend"""
    h = df['high'].values
    l = df['low'].values
    n = len(df)
    sar = np.zeros(n)
    ep = np.zeros(n)
    af = np.zeros(n)
    trend = np.zeros(n)

    sar[0] = l[0]; ep[0] = h[0]; af[0] = af_start; trend[0] = 1

    for i in range(1, n):
        if trend[i-1] == 1:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[i-2] if i > 1 else l[i-1])
            if l[i] < sar[i]:
                trend[i] = -1; sar[i] = ep[i-1]; ep[i] = l[i]; af[i] = af_start
            else:
                trend[i] = 1
                if h[i] > ep[i-1]:
                    ep[i] = h[i]; af[i] = min(af[i-1] + af_inc, af_max)
                else:
                    ep[i] = ep[i-1]; af[i] = af[i-1]
        else:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], h[i-1], h[i-2] if i > 1 else h[i-1])
            if h[i] > sar[i]:
                trend[i] = 1; sar[i] = ep[i-1]; ep[i] = h[i]; af[i] = af_start
            else:
                trend[i] = -1
                if l[i] < ep[i-1]:
                    ep[i] = l[i]; af[i] = min(af[i-1] + af_inc, af_max)
                else:
                    ep[i] = ep[i-1]; af[i] = af[i-1]

    return pd.Series(trend, index=df.index)


def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily OHLCV to weekly."""
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return weekly


# ============================================================================
# PRE-COMPUTE ALL INDICATORS
# ============================================================================

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators on a daily dataframe. Returns enriched df."""
    d = df.copy()
    c = d['close']

    # --- Moving Averages ---
    for p in [5, 8, 9, 10, 12, 13, 20, 21, 26, 50, 100, 200]:
        d[f'ema_{p}'] = ema(c, p)
        d[f'sma_{p}'] = sma(c, p)

    d['dema_20'] = dema(c, 20)
    d['tema_20'] = tema(c, 20)
    d['hull_20'] = hull_ma(c, 20)
    d['wma_20'] = wma(c, 20)

    # --- Momentum ---
    for p in [2, 5, 7, 9, 14, 21]:
        d[f'rsi_{p}'] = rsi(c, p)

    d['stoch_k_14'] = stochastic_k(d, 14)
    d['stoch_d_14'] = stochastic_d(d['stoch_k_14'], 3)
    d['stoch_k_5'] = stochastic_k(d, 5)
    d['stoch_d_5'] = stochastic_d(d['stoch_k_5'], 3)
    d['stoch_k_21'] = stochastic_k(d, 21)
    d['stoch_d_21'] = stochastic_d(d['stoch_k_21'], 9)

    d['williams_r_14'] = williams_r(d, 14)
    d['williams_r_2'] = williams_r(d, 2)
    d['williams_r_5'] = williams_r(d, 5)

    d['cci_20'] = cci(d, 20)
    d['cci_14'] = cci(d, 14)

    d['mfi_14'] = mfi(d, 14)
    d['mfi_10'] = mfi(d, 10)

    d['roc_12'] = roc(c, 12)
    d['roc_9'] = roc(c, 9)
    d['roc_20'] = roc(c, 20)

    d['tsi'] = tsi(c, 25, 13)

    # --- Trend ---
    d['adx_14'], d['plus_di_14'], d['minus_di_14'] = adx(d, 14)

    d['macd_line'], d['macd_signal'], d['macd_hist'] = macd(c, 12, 26, 9)
    d['macd_fast_line'], d['macd_fast_signal'], d['macd_fast_hist'] = macd(c, 8, 17, 9)

    d['supertrend_10_3'] = supertrend(d, 10, 3.0)
    d['supertrend_7_2'] = supertrend(d, 7, 2.0)
    d['supertrend_14_3'] = supertrend(d, 14, 3.0)

    d['psar_dir'] = psar_direction(d)

    # --- Volatility ---
    d['atr_14'] = atr(d, 14)
    d['atr_pct'] = d['atr_14'] / c * 100  # ATR as % of price

    d['bb_mid'], d['bb_upper'], d['bb_lower'], d['bb_pctb'], d['bb_width'] = bollinger_bands(c, 20, 2.0)
    d['bb_mid_10'], d['bb_upper_10'], d['bb_lower_10'], d['bb_pctb_10'], d['bb_width_10'] = bollinger_bands(c, 10, 1.5)

    d['kc_mid'], d['kc_upper'], d['kc_lower'] = keltner_channels(d, 20, 10, 2.0)
    d['kc_mid_6'], d['kc_upper_6'], d['kc_lower_6'] = keltner_channels(d, 6, 6, 1.3)

    d['dc_upper_20'], d['dc_lower_20'], d['dc_mid_20'] = donchian(d, 20)
    d['dc_upper_55'], d['dc_lower_55'], d['dc_mid_55'] = donchian(d, 55)

    # --- Volume ---
    d['obv'] = obv(d)
    d['obv_ema_20'] = ema(d['obv'], 20)
    d['cmf_20'] = cmf(d, 20)
    d['vol_sma_20'] = sma(d['volume'], 20)
    d['vol_ratio'] = d['volume'] / d['vol_sma_20'].replace(0, np.nan)

    # --- Derived signals ---
    # BB squeeze (BB inside Keltner)
    d['bb_squeeze'] = (d['bb_upper'] < d['kc_upper']) & (d['bb_lower'] > d['kc_lower'])

    # Price position
    d['above_ema_200'] = c > d['ema_200']
    d['above_sma_200'] = c > d['sma_200']
    d['above_ema_50'] = c > d['ema_50']
    d['above_ema_20'] = c > d['ema_20']
    d['above_ema_9'] = c > d['ema_9']

    # EMA alignment (bullish stack)
    d['ema_stack_bullish'] = (d['ema_9'] > d['ema_21']) & (d['ema_21'] > d['ema_50']) & (d['ema_50'] > d['ema_200'])
    d['ema_20_50_bull'] = d['ema_20'] > d['ema_50']
    d['ema_9_21_bull'] = d['ema_9'] > d['ema_21']

    # Golden/Death cross
    d['golden_cross'] = (d['sma_50'] > d['sma_200']) & (d['sma_50'].shift(1) <= d['sma_200'].shift(1))
    d['death_cross'] = (d['sma_50'] < d['sma_200']) & (d['sma_50'].shift(1) >= d['sma_200'].shift(1))

    # MACD crossovers
    d['macd_cross_up'] = (d['macd_line'] > d['macd_signal']) & (d['macd_line'].shift(1) <= d['macd_signal'].shift(1))
    d['macd_cross_down'] = (d['macd_line'] < d['macd_signal']) & (d['macd_line'].shift(1) >= d['macd_signal'].shift(1))
    d['macd_positive'] = d['macd_line'] > 0

    # EMA crossovers
    d['ema_9_21_cross_up'] = (d['ema_9'] > d['ema_21']) & (d['ema_9'].shift(1) <= d['ema_21'].shift(1))
    d['ema_9_21_cross_down'] = (d['ema_9'] < d['ema_21']) & (d['ema_9'].shift(1) >= d['ema_21'].shift(1))
    d['ema_12_26_cross_up'] = (d['ema_12'] > d['ema_26']) & (d['ema_12'].shift(1) <= d['ema_26'].shift(1))
    d['ema_20_50_cross_up'] = (d['ema_20'] > d['ema_50']) & (d['ema_20'].shift(1) <= d['ema_50'].shift(1))

    # Stochastic crossovers
    d['stoch_cross_up_14'] = (d['stoch_k_14'] > d['stoch_d_14']) & (d['stoch_k_14'].shift(1) <= d['stoch_d_14'].shift(1))
    d['stoch_cross_down_14'] = (d['stoch_k_14'] < d['stoch_d_14']) & (d['stoch_k_14'].shift(1) >= d['stoch_d_14'].shift(1))

    # Supertrend flips
    d['st_flip_up_10_3'] = (d['supertrend_10_3'] == 1) & (d['supertrend_10_3'].shift(1) == -1)
    d['st_flip_down_10_3'] = (d['supertrend_10_3'] == -1) & (d['supertrend_10_3'].shift(1) == 1)

    # PSAR flips
    d['psar_flip_up'] = (d['psar_dir'] == 1) & (d['psar_dir'].shift(1) == -1)
    d['psar_flip_down'] = (d['psar_dir'] == -1) & (d['psar_dir'].shift(1) == 1)

    # Donchian breakouts
    d['dc_breakout_up_20'] = d['close'] > d['dc_upper_20'].shift(1)
    d['dc_breakout_down_20'] = d['close'] < d['dc_lower_20'].shift(1)

    # Candle patterns (simple)
    d['bullish_engulf'] = (d['close'] > d['open']) & (d['close'].shift(1) < d['open'].shift(1)) & \
                          (d['close'] > d['open'].shift(1)) & (d['open'] < d['close'].shift(1))
    d['hammer'] = ((d['high'] - d['low']) > 3 * abs(d['close'] - d['open'])) & \
                  ((d['close'] - d['low']) / (d['high'] - d['low']).replace(0, np.nan) > 0.6) & \
                  ((d['open'] - d['low']) / (d['high'] - d['low']).replace(0, np.nan) > 0.6)

    # ATH proximity
    d['ath'] = d['high'].cummax()
    d['ath_pct'] = (d['ath'] - d['close']) / d['ath'] * 100  # distance from ATH in %

    return d


def compute_weekly_indicators(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly indicators and map back to daily index."""
    weekly = daily_to_weekly(daily_df)
    if len(weekly) < 52:
        return pd.DataFrame(index=daily_df.index)

    w = weekly.copy()
    c = w['close']

    w['w_ema_10'] = ema(c, 10)
    w['w_ema_20'] = ema(c, 20)
    w['w_ema_50'] = ema(c, 50)
    w['w_sma_50'] = sma(c, 50)
    w['w_rsi_14'] = rsi(c, 14)
    w['w_macd_line'], w['w_macd_signal'], w['w_macd_hist'] = macd(c, 12, 26, 9)
    w['w_adx'], w['w_plus_di'], w['w_minus_di'] = adx(w, 14)
    w['w_supertrend'] = supertrend(w, 10, 3.0)

    w['w_above_ema_50'] = c > w['w_ema_50']
    w['w_ema_10_above_20'] = w['w_ema_10'] > w['w_ema_20']
    w['w_rsi_bullish'] = w['w_rsi_14'] > 50
    w['w_rsi_not_ob'] = w['w_rsi_14'] < 70
    w['w_macd_bullish'] = w['w_macd_line'] > w['w_macd_signal']
    w['w_adx_trending'] = w['w_adx'] > 20
    w['w_supertrend_bull'] = w['w_supertrend'] == 1

    # Forward fill weekly values to daily dates
    weekly_cols = [col for col in w.columns if col.startswith('w_')]
    w_subset = w[weekly_cols]

    # Reindex to daily
    w_daily = w_subset.reindex(daily_df.index, method='ffill')

    return w_daily


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

@dataclass
class StrategyResult:
    name: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    total_return_pct: float
    max_drawdown_pct: float
    calmar_ratio: float
    profit_factor: float
    avg_hold_days: float
    sharpe_ratio: float
    symbols_traded: int
    trades_per_year: float


def backtest_strategy(all_enriched: Dict[str, pd.DataFrame],
                      entry_fn, exit_fn,
                      stop_loss_pct: float = 10.0,
                      take_profit_pct: float = 20.0,
                      max_hold_days: int = 30,
                      weekly_filter_fn=None,
                      weekly_data: Dict[str, pd.DataFrame] = None) -> StrategyResult:
    """
    Vectorized-ish backtester for a strategy across all stocks.

    entry_fn(row) -> bool : row from enriched df, returns True to enter
    exit_fn(row) -> bool : row from enriched df, returns True to exit (before SL/TP)
    weekly_filter_fn(row) -> bool : weekly row filter (must pass to allow entry)

    Returns StrategyResult with all metrics.
    """
    all_trades = []

    for sym, df in all_enriched.items():
        try:
            w_df = weekly_data.get(sym) if weekly_data else None
            trades = _backtest_single_stock(
                df, entry_fn, exit_fn, stop_loss_pct, take_profit_pct,
                max_hold_days, weekly_filter_fn, w_df
            )
            for t in trades:
                t['symbol'] = sym
            all_trades.extend(trades)
        except Exception:
            continue

    return _compute_metrics(all_trades)


def _backtest_single_stock(df: pd.DataFrame, entry_fn, exit_fn,
                           stop_loss_pct, take_profit_pct, max_hold_days,
                           weekly_filter_fn, w_df) -> List[Dict]:
    """Backtest a strategy on a single stock."""
    trades = []
    in_trade = False
    entry_price = 0
    entry_date = None
    entry_idx = 0

    # Start from row 200 to ensure all indicators are computed
    start_idx = 200

    for i in range(start_idx, len(df)):
        row = df.iloc[i]

        if not in_trade:
            # Check weekly filter if present
            if weekly_filter_fn is not None and w_df is not None and len(w_df) > 0:
                try:
                    date = df.index[i]
                    if date in w_df.index:
                        w_row = w_df.loc[date]
                    else:
                        # Find closest prior weekly date
                        prior = w_df.index[w_df.index <= date]
                        if len(prior) == 0:
                            continue
                        w_row = w_df.loc[prior[-1]]

                    if not weekly_filter_fn(w_row):
                        continue
                except Exception:
                    continue

            # Check entry
            try:
                if entry_fn(row):
                    in_trade = True
                    entry_price = row['close']
                    entry_date = df.index[i]
                    entry_idx = i
            except Exception:
                continue
        else:
            # Check exit conditions
            current_price = row['close']
            pnl_pct = (current_price / entry_price - 1) * 100
            hold_days = i - entry_idx

            exit_reason = None

            # Stop loss
            if pnl_pct <= -stop_loss_pct:
                exit_reason = 'stop_loss'
            # Take profit
            elif pnl_pct >= take_profit_pct:
                exit_reason = 'take_profit'
            # Max hold
            elif hold_days >= max_hold_days:
                exit_reason = 'max_hold'
            # Strategy exit
            else:
                try:
                    if exit_fn(row):
                        exit_reason = 'signal'
                except Exception:
                    pass

            if exit_reason:
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'hold_days': hold_days,
                    'exit_reason': exit_reason
                })
                in_trade = False

    return trades


def _compute_metrics(trades: List[Dict]) -> StrategyResult:
    """Compute strategy metrics from list of trades."""
    if len(trades) == 0:
        return StrategyResult(
            name='', total_trades=0, wins=0, losses=0, win_rate=0,
            avg_win_pct=0, avg_loss_pct=0, total_return_pct=0,
            max_drawdown_pct=0, calmar_ratio=0, profit_factor=0,
            avg_hold_days=0, sharpe_ratio=0, symbols_traded=0, trades_per_year=0
        )

    df = pd.DataFrame(trades)
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]

    total = len(df)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / total * 100 if total > 0 else 0

    avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0

    gross_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    total_return = df['pnl_pct'].sum()

    # Equity curve for drawdown and Sharpe
    cumulative = (1 + df['pnl_pct'] / 100).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak * 100
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

    # Calmar ratio (annualized return / max drawdown)
    if len(df) > 0 and 'entry_date' in df.columns:
        try:
            date_range = (pd.to_datetime(df['exit_date'].max()) - pd.to_datetime(df['entry_date'].min())).days
            years = max(date_range / 365.25, 0.5)
            annual_return = total_return / years
            calmar = annual_return / max_dd if max_dd > 0 else 0
            trades_per_year = total / years
        except Exception:
            calmar = 0
            trades_per_year = 0
            years = 1
    else:
        calmar = 0
        trades_per_year = 0
        years = 1

    # Sharpe (simplified)
    if len(df) > 1:
        sharpe = df['pnl_pct'].mean() / df['pnl_pct'].std() * np.sqrt(trades_per_year) if df['pnl_pct'].std() > 0 else 0
    else:
        sharpe = 0

    avg_hold = df['hold_days'].mean()
    symbols_traded = df['symbol'].nunique() if 'symbol' in df.columns else 0

    return StrategyResult(
        name='', total_trades=total, wins=n_wins, losses=n_losses,
        win_rate=round(win_rate, 2), avg_win_pct=round(avg_win, 2),
        avg_loss_pct=round(avg_loss, 2), total_return_pct=round(total_return, 2),
        max_drawdown_pct=round(max_dd, 2), calmar_ratio=round(calmar, 2),
        profit_factor=round(profit_factor, 2), avg_hold_days=round(avg_hold, 1),
        sharpe_ratio=round(sharpe, 2), symbols_traded=symbols_traded,
        trades_per_year=round(trades_per_year, 1)
    )


# ============================================================================
# STRATEGY DEFINITIONS - 120+ combinations
# ============================================================================

def define_all_strategies():
    """Define 120+ strategy combinations based on research."""
    strategies = []

    # =========================================================================
    # CATEGORY 1: MEAN REVERSION (RSI-based)
    # =========================================================================

    # 1. Connors RSI(2) - Buy when RSI(2) < 5, price > SMA(200)
    strategies.append({
        'name': 'Connors_RSI2_lt5_above_SMA200',
        'entry': lambda r: r['rsi_2'] < 5 and r['above_sma_200'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 15, 'hold': 10,
        'weekly_filter': None
    })

    # 2. RSI(2) < 10 + above EMA 200
    strategies.append({
        'name': 'RSI2_lt10_above_EMA200',
        'entry': lambda r: r['rsi_2'] < 10 and r['above_ema_200'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 15, 'hold': 10,
        'weekly_filter': None
    })

    # 3. RSI(2) < 5 + EMA stack bullish
    strategies.append({
        'name': 'RSI2_lt5_EMA_stack_bullish',
        'entry': lambda r: r['rsi_2'] < 5 and r['ema_stack_bullish'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 15, 'hold': 10,
        'weekly_filter': None
    })

    # 4. RSI(2) < 10 + weekly RSI bullish
    strategies.append({
        'name': 'RSI2_lt10_weekly_RSI_bullish',
        'entry': lambda r: r['rsi_2'] < 10 and r['above_sma_200'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': lambda w: w.get('w_rsi_bullish', False)
    })

    # 5. RSI(5) < 20 + above EMA 50
    strategies.append({
        'name': 'RSI5_lt20_above_EMA50',
        'entry': lambda r: r['rsi_5'] < 20 and r['above_ema_50'],
        'exit': lambda r: r['rsi_5'] > 70,
        'sl': 8, 'tp': 15, 'hold': 15,
        'weekly_filter': None
    })

    # 6. RSI(14) oversold bounce + above SMA 200
    strategies.append({
        'name': 'RSI14_oversold_bounce_SMA200',
        'entry': lambda r: r['rsi_14'] < 30 and r['above_sma_200'],
        'exit': lambda r: r['rsi_14'] > 65,
        'sl': 10, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # 7. RSI(2) < 5 + MACD positive
    strategies.append({
        'name': 'RSI2_lt5_MACD_positive',
        'entry': lambda r: r['rsi_2'] < 5 and r['macd_positive'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': None
    })

    # 8. RSI(2) < 10 + ADX trending + above EMA200
    strategies.append({
        'name': 'RSI2_lt10_ADX_trending_EMA200',
        'entry': lambda r: r['rsi_2'] < 10 and r['adx_14'] > 25 and r['above_ema_200'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 15, 'hold': 10,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 2: WILLIAMS %R MEAN REVERSION
    # =========================================================================

    # 9. Williams %R(2) < -95 + above SMA200
    strategies.append({
        'name': 'WR2_lt95_above_SMA200',
        'entry': lambda r: r['williams_r_2'] < -95 and r['above_sma_200'],
        'exit': lambda r: r['williams_r_2'] > -20,
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': None
    })

    # 10. Williams %R(5) < -90 + above EMA50
    strategies.append({
        'name': 'WR5_lt90_above_EMA50',
        'entry': lambda r: r['williams_r_5'] < -90 and r['above_ema_50'],
        'exit': lambda r: r['williams_r_5'] > -20,
        'sl': 8, 'tp': 15, 'hold': 15,
        'weekly_filter': None
    })

    # 11. WR(2) < -95 + EMA stack
    strategies.append({
        'name': 'WR2_lt95_EMA_stack',
        'entry': lambda r: r['williams_r_2'] < -95 and r['ema_stack_bullish'],
        'exit': lambda r: r['williams_r_2'] > -30,
        'sl': 7, 'tp': 12, 'hold': 10,
        'weekly_filter': None
    })

    # 12. WR(14) oversold + weekly bullish
    strategies.append({
        'name': 'WR14_oversold_weekly_bull',
        'entry': lambda r: r['williams_r_14'] < -80 and r['above_ema_200'],
        'exit': lambda r: r['williams_r_14'] > -20,
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': lambda w: w.get('w_rsi_bullish', False)
    })

    # =========================================================================
    # CATEGORY 3: KELTNER CHANNEL MEAN REVERSION
    # =========================================================================

    # 13. Keltner(6,1.3) lower band touch + above SMA200
    strategies.append({
        'name': 'Keltner6_lower_SMA200',
        'entry': lambda r: r['close'] < r['kc_lower_6'] and r['above_sma_200'],
        'exit': lambda r: r['close'] > r['kc_mid_6'],
        'sl': 8, 'tp': 15, 'hold': 15,
        'weekly_filter': None
    })

    # 14. Keltner(20,2) lower band + RSI < 40
    strategies.append({
        'name': 'Keltner20_lower_RSI_lt40',
        'entry': lambda r: r['close'] < r['kc_lower'] and r['rsi_14'] < 40,
        'exit': lambda r: r['close'] > r['kc_mid'],
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # 15. Keltner(6,1.3) lower + weekly uptrend
    strategies.append({
        'name': 'Keltner6_lower_weekly_up',
        'entry': lambda r: r['close'] < r['kc_lower_6'] and r['above_ema_200'],
        'exit': lambda r: r['close'] > r['kc_mid_6'],
        'sl': 8, 'tp': 12, 'hold': 15,
        'weekly_filter': lambda w: w.get('w_above_ema_50', False)
    })

    # =========================================================================
    # CATEGORY 4: BOLLINGER BAND STRATEGIES
    # =========================================================================

    # 16. BB lower band touch + RSI < 30
    strategies.append({
        'name': 'BB_lower_RSI_lt30',
        'entry': lambda r: r['bb_pctb'] < 0.05 and r['rsi_14'] < 30,
        'exit': lambda r: r['bb_pctb'] > 0.5,
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # 17. BB lower + above SMA200
    strategies.append({
        'name': 'BB_lower_above_SMA200',
        'entry': lambda r: r['close'] < r['bb_lower'] and r['above_sma_200'],
        'exit': lambda r: r['close'] > r['bb_mid'],
        'sl': 8, 'tp': 15, 'hold': 15,
        'weekly_filter': None
    })

    # 18. BB squeeze + Donchian breakout
    strategies.append({
        'name': 'BB_squeeze_Donchian_breakout',
        'entry': lambda r: r['bb_squeeze'] and r['dc_breakout_up_20'],
        'exit': lambda r: r['close'] < r['ema_20'],
        'sl': 8, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # 19. BB squeeze release + volume spike
    strategies.append({
        'name': 'BB_squeeze_release_volume',
        'entry': lambda r: not r['bb_squeeze'] and r.get('bb_squeeze', False) == False and r['vol_ratio'] > 1.5 and r['close'] > r['bb_upper'],
        'exit': lambda r: r['close'] < r['ema_20'],
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # 20. BB(10,1.5) lower band mean reversion
    strategies.append({
        'name': 'BB10_lower_mean_reversion',
        'entry': lambda r: r['close'] < r['bb_lower_10'] and r['above_ema_50'],
        'exit': lambda r: r['close'] > r['bb_mid_10'],
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 5: EMA CROSSOVER STRATEGIES
    # =========================================================================

    # 21. EMA 9/21 cross up + above EMA 200
    strategies.append({
        'name': 'EMA_9_21_crossup_above200',
        'entry': lambda r: r['ema_9_21_cross_up'] and r['above_ema_200'],
        'exit': lambda r: r['ema_9_21_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 22. EMA 9/21 cross + RSI 45-70
    strategies.append({
        'name': 'EMA_9_21_cross_RSI_filter',
        'entry': lambda r: r['ema_9_21_cross_up'] and 45 < r['rsi_14'] < 70,
        'exit': lambda r: r['ema_9_21_cross_down'] or r['rsi_14'] > 75,
        'sl': 8, 'tp': 15, 'hold': 25,
        'weekly_filter': None
    })

    # 23. EMA 9/21 cross + volume confirmation
    strategies.append({
        'name': 'EMA_9_21_cross_volume',
        'entry': lambda r: r['ema_9_21_cross_up'] and r['vol_ratio'] > 1.3,
        'exit': lambda r: r['ema_9_21_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 24. EMA 12/26 cross + MACD confirmation
    strategies.append({
        'name': 'EMA_12_26_cross_MACD',
        'entry': lambda r: r['ema_12_26_cross_up'] and r['macd_hist'] > 0,
        'exit': lambda r: r['macd_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 25. EMA 20/50 cross + ADX > 25
    strategies.append({
        'name': 'EMA_20_50_cross_ADX',
        'entry': lambda r: r['ema_20_50_cross_up'] and r['adx_14'] > 25,
        'exit': lambda r: r['close'] < r['ema_20'],
        'sl': 12, 'tp': 25, 'hold': 30,
        'weekly_filter': None
    })

    # 26. EMA 9/21 cross + weekly EMA bullish
    strategies.append({
        'name': 'EMA_9_21_cross_weekly_up',
        'entry': lambda r: r['ema_9_21_cross_up'] and r['above_ema_200'],
        'exit': lambda r: r['ema_9_21_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': lambda w: w.get('w_ema_10_above_20', False)
    })

    # =========================================================================
    # CATEGORY 6: MACD STRATEGIES
    # =========================================================================

    # 27. MACD cross up + RSI < 60
    strategies.append({
        'name': 'MACD_cross_RSI_lt60',
        'entry': lambda r: r['macd_cross_up'] and r['rsi_14'] < 60,
        'exit': lambda r: r['macd_cross_down'] or r['rsi_14'] > 75,
        'sl': 10, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # 28. MACD cross up + above EMA 200 + ADX > 20
    strategies.append({
        'name': 'MACD_cross_EMA200_ADX',
        'entry': lambda r: r['macd_cross_up'] and r['above_ema_200'] and r['adx_14'] > 20,
        'exit': lambda r: r['macd_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 29. MACD cross + OBV confirmation
    strategies.append({
        'name': 'MACD_cross_OBV_bullish',
        'entry': lambda r: r['macd_cross_up'] and r['obv'] > r['obv_ema_20'],
        'exit': lambda r: r['macd_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 30. MACD + RSI + EMA triple confirmation
    strategies.append({
        'name': 'Triple_MACD_RSI_EMA',
        'entry': lambda r: r['macd_cross_up'] and r['rsi_14'] > 50 and r['rsi_14'] < 70 and r['ema_9_21_bull'],
        'exit': lambda r: r['macd_cross_down'] or r['rsi_14'] > 80,
        'sl': 10, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # 31. MACD + Supertrend + ADX
    strategies.append({
        'name': 'MACD_Supertrend_ADX',
        'entry': lambda r: r['macd_cross_up'] and r['supertrend_10_3'] == 1 and r['adx_14'] > 25,
        'exit': lambda r: r['supertrend_10_3'] == -1 or r['macd_cross_down'],
        'sl': 12, 'tp': 25, 'hold': 30,
        'weekly_filter': None
    })

    # 32. MACD + PSAR + EMA200
    strategies.append({
        'name': 'MACD_PSAR_EMA200',
        'entry': lambda r: r['macd_cross_up'] and r['psar_dir'] == 1 and r['above_ema_200'],
        'exit': lambda r: r['psar_flip_down'] or r['macd_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 33. Fast MACD(8,17,9) cross + RSI filter
    strategies.append({
        'name': 'Fast_MACD_cross_RSI',
        'entry': lambda r: r['macd_fast_hist'] > 0 and r.get('macd_fast_hist', 0) > 0 and r['rsi_14'] < 65 and r['rsi_14'] > 40,
        'exit': lambda r: r['macd_fast_hist'] < 0 and r['rsi_14'] > 70,
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 7: SUPERTREND STRATEGIES
    # =========================================================================

    # 34. Supertrend flip up + RSI 40-65
    strategies.append({
        'name': 'Supertrend_flip_RSI_filter',
        'entry': lambda r: r['st_flip_up_10_3'] and 40 < r['rsi_14'] < 65,
        'exit': lambda r: r['st_flip_down_10_3'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 35. Supertrend + EMA alignment
    strategies.append({
        'name': 'Supertrend_EMA_aligned',
        'entry': lambda r: r['st_flip_up_10_3'] and r['ema_9_21_bull'] and r['above_ema_50'],
        'exit': lambda r: r['st_flip_down_10_3'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 36. Supertrend + ADX trending
    strategies.append({
        'name': 'Supertrend_ADX_trending',
        'entry': lambda r: r['st_flip_up_10_3'] and r['adx_14'] > 25 and r['plus_di_14'] > r['minus_di_14'],
        'exit': lambda r: r['st_flip_down_10_3'],
        'sl': 12, 'tp': 25, 'hold': 30,
        'weekly_filter': None
    })

    # 37. Supertrend(7,2) fast + volume
    strategies.append({
        'name': 'Supertrend_7_2_volume',
        'entry': lambda r: r['supertrend_7_2'] == 1 and r.get('supertrend_7_2_prev', 0) != 1 and r['vol_ratio'] > 1.5,
        'exit': lambda r: r['supertrend_7_2'] == -1,
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # 38. Supertrend + Weekly supertrend
    strategies.append({
        'name': 'Supertrend_weekly_aligned',
        'entry': lambda r: r['st_flip_up_10_3'] and r['above_ema_200'],
        'exit': lambda r: r['st_flip_down_10_3'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': lambda w: w.get('w_supertrend_bull', False)
    })

    # =========================================================================
    # CATEGORY 8: STOCHASTIC STRATEGIES
    # =========================================================================

    # 39. Stoch(14) cross up from oversold + above EMA 200
    strategies.append({
        'name': 'Stoch14_oversold_cross_EMA200',
        'entry': lambda r: r['stoch_cross_up_14'] and r['stoch_k_14'] < 30 and r['above_ema_200'],
        'exit': lambda r: r['stoch_k_14'] > 80,
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # 40. Stoch(5) cross up + EMA 9/21 bullish
    strategies.append({
        'name': 'Stoch5_cross_EMA_bull',
        'entry': lambda r: r['stoch_k_5'] > r['stoch_d_5'] and r.get('stoch_k_5_prev', 100) <= r.get('stoch_d_5_prev', 0) and r['ema_9_21_bull'],
        'exit': lambda r: r['stoch_k_5'] > 80 and r['stoch_k_5'] < r['stoch_d_5'],
        'sl': 8, 'tp': 15, 'hold': 15,
        'weekly_filter': None
    })

    # 41. Stoch(14) oversold + RSI < 35
    strategies.append({
        'name': 'Stoch14_RSI_double_oversold',
        'entry': lambda r: r['stoch_k_14'] < 20 and r['rsi_14'] < 35 and r['above_sma_200'],
        'exit': lambda r: r['stoch_k_14'] > 70 or r['rsi_14'] > 65,
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # 42. Stoch(21,9) slow cross + weekly RSI
    strategies.append({
        'name': 'SlowStoch_weekly_RSI',
        'entry': lambda r: r['stoch_k_21'] < 30 and r['above_ema_200'],
        'exit': lambda r: r['stoch_k_21'] > 75,
        'sl': 10, 'tp': 18, 'hold': 25,
        'weekly_filter': lambda w: w.get('w_rsi_bullish', False)
    })

    # =========================================================================
    # CATEGORY 9: CCI STRATEGIES
    # =========================================================================

    # 43. CCI(20) oversold bounce + above EMA 200
    strategies.append({
        'name': 'CCI20_oversold_EMA200',
        'entry': lambda r: r['cci_20'] < -100 and r['above_ema_200'],
        'exit': lambda r: r['cci_20'] > 100,
        'sl': 10, 'tp': 15, 'hold': 25,
        'weekly_filter': None
    })

    # 44. CCI(14) zero line cross + MACD positive
    strategies.append({
        'name': 'CCI14_zero_MACD_positive',
        'entry': lambda r: r['cci_14'] > 0 and r['macd_positive'] and r['above_ema_50'],
        'exit': lambda r: r['cci_14'] < -100,
        'sl': 10, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 10: MFI STRATEGIES
    # =========================================================================

    # 45. MFI(14) oversold + above EMA200
    strategies.append({
        'name': 'MFI14_oversold_EMA200',
        'entry': lambda r: r['mfi_14'] < 20 and r['above_ema_200'],
        'exit': lambda r: r['mfi_14'] > 80,
        'sl': 10, 'tp': 15, 'hold': 25,
        'weekly_filter': None
    })

    # 46. MFI(10) + RSI double confirmation
    strategies.append({
        'name': 'MFI10_RSI_double_confirm',
        'entry': lambda r: r['mfi_10'] < 25 and r['rsi_14'] < 35 and r['above_sma_200'],
        'exit': lambda r: r['mfi_10'] > 75 or r['rsi_14'] > 65,
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 11: ADX + DI STRATEGIES
    # =========================================================================

    # 47. ADX > 25 + +DI cross above -DI + above EMA50
    strategies.append({
        'name': 'ADX_DI_cross_EMA50',
        'entry': lambda r: r['adx_14'] > 25 and r['plus_di_14'] > r['minus_di_14'] and r['above_ema_50'],
        'exit': lambda r: r['plus_di_14'] < r['minus_di_14'] or r['adx_14'] < 20,
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 48. ADX rising + MACD positive + EMA aligned
    strategies.append({
        'name': 'ADX_rising_MACD_EMA',
        'entry': lambda r: r['adx_14'] > 20 and r['macd_positive'] and r['ema_9_21_bull'],
        'exit': lambda r: r['adx_14'] < 15 or r['macd_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 12: PARABOLIC SAR STRATEGIES
    # =========================================================================

    # 49. PSAR flip up + ADX > 25
    strategies.append({
        'name': 'PSAR_flip_ADX_trending',
        'entry': lambda r: r['psar_flip_up'] and r['adx_14'] > 25,
        'exit': lambda r: r['psar_flip_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 50. PSAR flip + EMA 200 + volume
    strategies.append({
        'name': 'PSAR_flip_EMA200_volume',
        'entry': lambda r: r['psar_flip_up'] and r['above_ema_200'] and r['vol_ratio'] > 1.2,
        'exit': lambda r: r['psar_flip_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 13: DONCHIAN CHANNEL BREAKOUT
    # =========================================================================

    # 51. Donchian 20 breakout + volume + above EMA50
    strategies.append({
        'name': 'Donchian20_breakout_vol_EMA50',
        'entry': lambda r: r['dc_breakout_up_20'] and r['vol_ratio'] > 1.5 and r['above_ema_50'],
        'exit': lambda r: r['close'] < r['dc_mid_20'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 52. Donchian 55 breakout + ADX
    strategies.append({
        'name': 'Donchian55_breakout_ADX',
        'entry': lambda r: r['close'] > r['dc_upper_55'].item() if hasattr(r['dc_upper_55'], 'item') else r['close'] > r['dc_upper_55'] and r['adx_14'] > 20,
        'exit': lambda r: r['close'] < r['dc_mid_55'],
        'sl': 12, 'tp': 25, 'hold': 30,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 14: TSI STRATEGIES
    # =========================================================================

    # 53. TSI cross above 0 + above EMA 200
    strategies.append({
        'name': 'TSI_cross_zero_EMA200',
        'entry': lambda r: r['tsi'] > 0 and r['above_ema_200'],
        'exit': lambda r: r['tsi'] < -10,
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 15: ROC STRATEGIES
    # =========================================================================

    # 54. ROC(12) positive + RSI not overbought
    strategies.append({
        'name': 'ROC12_positive_RSI_filter',
        'entry': lambda r: r['roc_12'] > 5 and r['rsi_14'] < 70 and r['above_ema_50'],
        'exit': lambda r: r['roc_12'] < -5 or r['rsi_14'] > 80,
        'sl': 10, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # 55. ROC(9) negative reversal (mean reversion)
    strategies.append({
        'name': 'ROC9_reversal_mean_revert',
        'entry': lambda r: r['roc_9'] < -8 and r['above_sma_200'],
        'exit': lambda r: r['roc_9'] > 5,
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 16: VOLUME STRATEGIES
    # =========================================================================

    # 56. OBV bullish + EMA cross
    strategies.append({
        'name': 'OBV_bullish_EMA_cross',
        'entry': lambda r: r['obv'] > r['obv_ema_20'] and r['ema_9_21_cross_up'],
        'exit': lambda r: r['obv'] < r['obv_ema_20'] and r['ema_9_21_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 57. CMF(20) positive + above EMA 50
    strategies.append({
        'name': 'CMF_positive_EMA50',
        'entry': lambda r: r['cmf_20'] > 0.1 and r['above_ema_50'] and r['rsi_14'] < 70,
        'exit': lambda r: r['cmf_20'] < -0.1,
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 58. Volume spike + RSI oversold + above EMA200
    strategies.append({
        'name': 'Volume_spike_RSI_oversold',
        'entry': lambda r: r['vol_ratio'] > 2.0 and r['rsi_14'] < 35 and r['above_ema_200'],
        'exit': lambda r: r['rsi_14'] > 60,
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 17: ATH PROXIMITY STRATEGIES
    # =========================================================================

    # 59. Near ATH (< 5%) + strong trend
    strategies.append({
        'name': 'Near_ATH_strong_trend',
        'entry': lambda r: r['ath_pct'] < 5 and r['adx_14'] > 25 and r['ema_stack_bullish'],
        'exit': lambda r: r['close'] < r['ema_21'] or r['ath_pct'] > 15,
        'sl': 8, 'tp': 15, 'hold': 25,
        'weekly_filter': None
    })

    # 60. ATH breakout + volume
    strategies.append({
        'name': 'ATH_breakout_volume',
        'entry': lambda r: r['ath_pct'] < 1 and r['vol_ratio'] > 1.5,
        'exit': lambda r: r['close'] < r['ema_20'],
        'sl': 8, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 18: MULTI-INDICATOR COMBOS (RESEARCH-BACKED)
    # =========================================================================

    # 61. Triple: RSI + MACD + OBV
    strategies.append({
        'name': 'Triple_RSI_MACD_OBV',
        'entry': lambda r: r['rsi_14'] > 50 and r['macd_positive'] and r['obv'] > r['obv_ema_20'] and r['above_ema_200'],
        'exit': lambda r: r['rsi_14'] > 80 or (r['macd_cross_down'] and r['obv'] < r['obv_ema_20']),
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 62. MACD + RSI + Supertrend
    strategies.append({
        'name': 'MACD_RSI_Supertrend',
        'entry': lambda r: r['macd_positive'] and r['rsi_14'] > 50 and r['rsi_14'] < 70 and r['supertrend_10_3'] == 1,
        'exit': lambda r: r['supertrend_10_3'] == -1 or r['rsi_14'] > 80,
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # 63. EMA + RSI + ADX + Volume
    strategies.append({
        'name': 'Quad_EMA_RSI_ADX_Vol',
        'entry': lambda r: r['ema_9_21_cross_up'] and r['rsi_14'] > 45 and r['rsi_14'] < 65 and r['adx_14'] > 20 and r['vol_ratio'] > 1.2,
        'exit': lambda r: r['ema_9_21_cross_down'] or r['rsi_14'] > 80,
        'sl': 10, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # 64. Supertrend + RSI + EMA (research-backed combination)
    strategies.append({
        'name': 'Research_ST_RSI_EMA',
        'entry': lambda r: r['supertrend_10_3'] == 1 and r['rsi_14'] > 50 and r['above_ema_50'],
        'exit': lambda r: r['supertrend_10_3'] == -1 or r['rsi_14'] > 80,
        'sl': 12, 'tp': 25, 'hold': 30,
        'weekly_filter': None
    })

    # 65. Keltner + RSI + MACD
    strategies.append({
        'name': 'Keltner_RSI_MACD',
        'entry': lambda r: r['close'] < r['kc_lower'] and r['rsi_14'] < 35 and r['macd_hist'] > r.get('macd_hist_prev', -999),
        'exit': lambda r: r['close'] > r['kc_mid'] or r['rsi_14'] > 70,
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 19: TOP-DOWN (WEEKLY FILTER + DAILY ENTRY)
    # =========================================================================

    # 66. Weekly EMA50 bull + Daily RSI(2) < 10
    strategies.append({
        'name': 'W_EMA50_D_RSI2_lt10',
        'entry': lambda r: r['rsi_2'] < 10 and r['above_ema_200'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': lambda w: w.get('w_above_ema_50', False)
    })

    # 67. Weekly MACD bull + Daily EMA cross
    strategies.append({
        'name': 'W_MACD_D_EMA_cross',
        'entry': lambda r: r['ema_9_21_cross_up'] and r['above_ema_50'],
        'exit': lambda r: r['ema_9_21_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': lambda w: w.get('w_macd_bullish', False)
    })

    # 68. Weekly ADX trending + Daily MACD cross
    strategies.append({
        'name': 'W_ADX_D_MACD_cross',
        'entry': lambda r: r['macd_cross_up'] and r['above_ema_50'],
        'exit': lambda r: r['macd_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': lambda w: w.get('w_adx_trending', False)
    })

    # 69. Weekly Supertrend bull + Daily Stoch oversold
    strategies.append({
        'name': 'W_Supertrend_D_Stoch_oversold',
        'entry': lambda r: r['stoch_k_14'] < 25 and r['above_ema_50'],
        'exit': lambda r: r['stoch_k_14'] > 80,
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': lambda w: w.get('w_supertrend_bull', False)
    })

    # 70. Weekly RSI bull + Daily BB lower touch
    strategies.append({
        'name': 'W_RSI_D_BB_lower',
        'entry': lambda r: r['close'] < r['bb_lower'] and r['above_sma_200'],
        'exit': lambda r: r['close'] > r['bb_mid'],
        'sl': 8, 'tp': 12, 'hold': 15,
        'weekly_filter': lambda w: w.get('w_rsi_bullish', False) and w.get('w_rsi_not_ob', False)
    })

    # 71. Weekly EMA aligned + Daily RSI pullback
    strategies.append({
        'name': 'W_EMA_aligned_D_RSI_pullback',
        'entry': lambda r: r['rsi_14'] < 40 and r['above_ema_200'] and r['ema_20_50_bull'],
        'exit': lambda r: r['rsi_14'] > 65,
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': lambda w: w.get('w_ema_10_above_20', False) and w.get('w_above_ema_50', False)
    })

    # 72. Weekly Supertrend + ADX + Daily EMA cross + volume
    strategies.append({
        'name': 'W_ST_ADX_D_EMA_vol',
        'entry': lambda r: r['ema_9_21_cross_up'] and r['vol_ratio'] > 1.3 and r['above_ema_200'],
        'exit': lambda r: r['ema_9_21_cross_down'],
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': lambda w: w.get('w_supertrend_bull', False) and w.get('w_adx_trending', False)
    })

    # =========================================================================
    # CATEGORY 20: PULLBACK TO MOVING AVERAGE STRATEGIES
    # =========================================================================

    # 73. Pullback to EMA 21 in uptrend
    strategies.append({
        'name': 'Pullback_EMA21_uptrend',
        'entry': lambda r: abs(r['close'] - r['ema_21']) / r['ema_21'] * 100 < 1.5 and r['ema_stack_bullish'] and r['rsi_14'] < 55,
        'exit': lambda r: r['close'] < r['ema_50'] or r['rsi_14'] > 75,
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # 74. Pullback to EMA 50 in strong trend
    strategies.append({
        'name': 'Pullback_EMA50_strong_trend',
        'entry': lambda r: abs(r['close'] - r['ema_50']) / r['ema_50'] * 100 < 2 and r['above_ema_200'] and r['adx_14'] > 25,
        'exit': lambda r: r['close'] > r['ema_20'] and r['rsi_14'] > 65,
        'sl': 10, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # 75. Pullback to EMA 9 with volume dry up
    strategies.append({
        'name': 'Pullback_EMA9_low_volume',
        'entry': lambda r: abs(r['close'] - r['ema_9']) / r['ema_9'] * 100 < 1 and r['ema_stack_bullish'] and r['vol_ratio'] < 0.7,
        'exit': lambda r: r['vol_ratio'] > 1.5 and r['close'] > r['ema_9'],
        'sl': 5, 'tp': 10, 'hold': 10,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 21: CANDLE PATTERN + INDICATOR COMBOS
    # =========================================================================

    # 76. Bullish engulfing + above EMA 200 + RSI < 50
    strategies.append({
        'name': 'Bullish_engulf_EMA200_RSI',
        'entry': lambda r: r['bullish_engulf'] and r['above_ema_200'] and r['rsi_14'] < 50,
        'exit': lambda r: r['rsi_14'] > 70 or r['close'] < r['ema_21'],
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # 77. Hammer + volume spike + above EMA 50
    strategies.append({
        'name': 'Hammer_volume_EMA50',
        'entry': lambda r: r['hammer'] and r['vol_ratio'] > 1.5 and r['above_ema_50'],
        'exit': lambda r: r['rsi_14'] > 70,
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 22: MOMENTUM + TREND COMBOS
    # =========================================================================

    # 78. ROC(12) > 0 + ADX > 25 + EMA aligned
    strategies.append({
        'name': 'ROC_ADX_EMA_momentum',
        'entry': lambda r: r['roc_12'] > 3 and r['adx_14'] > 25 and r['ema_9_21_bull'] and r['above_ema_50'],
        'exit': lambda r: r['roc_12'] < -3 or r['adx_14'] < 20,
        'sl': 10, 'tp': 20, 'hold': 25,
        'weekly_filter': None
    })

    # 79. TSI > 0 + Supertrend bull + volume
    strategies.append({
        'name': 'TSI_Supertrend_volume',
        'entry': lambda r: r['tsi'] > 5 and r['supertrend_10_3'] == 1 and r['vol_ratio'] > 1.2,
        'exit': lambda r: r['tsi'] < -5 or r['supertrend_10_3'] == -1,
        'sl': 10, 'tp': 20, 'hold': 30,
        'weekly_filter': None
    })

    # =========================================================================
    # CATEGORY 23: PARAMETER VARIATIONS OF TOP STRATEGIES
    # =========================================================================

    # Connors-style with different RSI thresholds
    for rsi_thresh in [3, 5, 8, 15]:
        for exit_ma in [5, 10]:
            strategies.append({
                'name': f'MeanRevert_RSI2_lt{rsi_thresh}_exit_SMA{exit_ma}',
                'entry': lambda r, t=rsi_thresh: r['rsi_2'] < t and r['above_sma_200'],
                'exit': lambda r, m=exit_ma: r['close'] > r[f'sma_{m}'],
                'sl': 8, 'tp': 15, 'hold': 12,
                'weekly_filter': None
            })

    # Williams %R variations
    for wr_thresh in [-90, -95, -98]:
        strategies.append({
            'name': f'WR2_lt{abs(wr_thresh)}_SMA200',
            'entry': lambda r, t=wr_thresh: r['williams_r_2'] < t and r['above_sma_200'],
            'exit': lambda r: r['williams_r_2'] > -20,
            'sl': 8, 'tp': 12, 'hold': 10,
            'weekly_filter': None
        })

    # EMA crossover with different periods
    for fast, slow in [(5, 13), (8, 21), (10, 30), (12, 26)]:
        strategies.append({
            'name': f'EMA_{fast}_{slow}_cross_filtered',
            'entry': lambda r, f=fast, s=slow: r[f'ema_{f}'] > r[f'ema_{s}'] and r.get(f'ema_{f}_prev', 0) <= r.get(f'ema_{s}_prev', 999) and r['above_ema_200'] and r['rsi_14'] < 70,
            'exit': lambda r, f=fast, s=slow: r[f'ema_{f}'] < r[f'ema_{s}'],
            'sl': 10, 'tp': 20, 'hold': 30,
            'weekly_filter': None
        })

    # RSI mean reversion + different weekly filters
    strategies.append({
        'name': 'RSI2_lt5_W_MACD_bull',
        'entry': lambda r: r['rsi_2'] < 5 and r['above_sma_200'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': lambda w: w.get('w_macd_bullish', False)
    })

    strategies.append({
        'name': 'RSI2_lt5_W_Supertrend_bull',
        'entry': lambda r: r['rsi_2'] < 5 and r['above_sma_200'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': lambda w: w.get('w_supertrend_bull', False)
    })

    strategies.append({
        'name': 'RSI2_lt5_W_ADX_trending',
        'entry': lambda r: r['rsi_2'] < 5 and r['above_sma_200'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': lambda w: w.get('w_adx_trending', False)
    })

    # Different stop-loss/take-profit combos for best entries
    for sl, tp in [(5, 10), (6, 12), (8, 15), (10, 20), (12, 25)]:
        strategies.append({
            'name': f'RSI2_lt5_SMA200_SL{sl}_TP{tp}',
            'entry': lambda r: r['rsi_2'] < 5 and r['above_sma_200'],
            'exit': lambda r: r['close'] > r['sma_5'],
            'sl': sl, 'tp': tp, 'hold': 10,
            'weekly_filter': None
        })

    # =========================================================================
    # CATEGORY 24: MULTI-TIMEFRAME COMBOS
    # =========================================================================

    # 100. Weekly EMA + RSI bull + Daily MACD + Supertrend
    strategies.append({
        'name': 'W_full_bull_D_MACD_ST',
        'entry': lambda r: r['macd_cross_up'] and r['supertrend_10_3'] == 1,
        'exit': lambda r: r['supertrend_10_3'] == -1,
        'sl': 12, 'tp': 25, 'hold': 30,
        'weekly_filter': lambda w: w.get('w_above_ema_50', False) and w.get('w_rsi_bullish', False) and w.get('w_macd_bullish', False)
    })

    # 101. Weekly all green + Daily pullback entry
    strategies.append({
        'name': 'W_all_green_D_pullback',
        'entry': lambda r: r['rsi_14'] < 45 and r['close'] > r['ema_50'] and r['ema_20_50_bull'],
        'exit': lambda r: r['rsi_14'] > 70,
        'sl': 10, 'tp': 18, 'hold': 25,
        'weekly_filter': lambda w: w.get('w_above_ema_50', False) and w.get('w_supertrend_bull', False) and w.get('w_rsi_not_ob', False)
    })

    # 102. Weekly EMA10>20 + Daily Keltner lower touch
    strategies.append({
        'name': 'W_EMA_bull_D_Keltner_lower',
        'entry': lambda r: r['close'] < r['kc_lower_6'] and r['above_ema_200'],
        'exit': lambda r: r['close'] > r['kc_mid_6'],
        'sl': 8, 'tp': 12, 'hold': 15,
        'weekly_filter': lambda w: w.get('w_ema_10_above_20', False) and w.get('w_rsi_bullish', False)
    })

    # =========================================================================
    # CATEGORY 25: HYBRID MEAN-REVERSION + TREND
    # =========================================================================

    # 103. RSI oversold in strong uptrend
    strategies.append({
        'name': 'RSI_oversold_strong_uptrend',
        'entry': lambda r: r['rsi_14'] < 35 and r['ema_stack_bullish'] and r['adx_14'] > 20,
        'exit': lambda r: r['rsi_14'] > 60,
        'sl': 8, 'tp': 15, 'hold': 15,
        'weekly_filter': None
    })

    # 104. Stochastic oversold + EMA stack + weekly bull
    strategies.append({
        'name': 'Stoch_oversold_EMA_stack_W',
        'entry': lambda r: r['stoch_k_14'] < 20 and r['ema_9_21_bull'] and r['above_ema_200'],
        'exit': lambda r: r['stoch_k_14'] > 75,
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': lambda w: w.get('w_rsi_bullish', False)
    })

    # 105. BB lower + EMA stack + volume spike
    strategies.append({
        'name': 'BB_lower_EMA_stack_vol',
        'entry': lambda r: r['close'] < r['bb_lower'] and r['ema_9_21_bull'] and r['vol_ratio'] > 1.5,
        'exit': lambda r: r['close'] > r['bb_mid'],
        'sl': 8, 'tp': 12, 'hold': 15,
        'weekly_filter': None
    })

    # 106-115: More variations with different hold periods
    for hold in [5, 8, 12, 15, 20]:
        strategies.append({
            'name': f'RSI2_lt5_SMA200_hold{hold}',
            'entry': lambda r: r['rsi_2'] < 5 and r['above_sma_200'],
            'exit': lambda r: r['close'] > r['sma_5'],
            'sl': 8, 'tp': 15, 'hold': hold,
            'weekly_filter': None
        })

    # 116-120: Combo entry with weekly double filter
    strategies.append({
        'name': 'W_double_D_RSI2_lt10',
        'entry': lambda r: r['rsi_2'] < 10 and r['above_ema_200'],
        'exit': lambda r: r['close'] > r['sma_5'],
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': lambda w: w.get('w_rsi_bullish', False) and w.get('w_supertrend_bull', False)
    })

    strategies.append({
        'name': 'W_triple_D_BB_lower',
        'entry': lambda r: r['close'] < r['bb_lower'] and r['above_sma_200'],
        'exit': lambda r: r['close'] > r['bb_mid'],
        'sl': 8, 'tp': 12, 'hold': 15,
        'weekly_filter': lambda w: w.get('w_rsi_bullish', False) and w.get('w_above_ema_50', False) and w.get('w_macd_bullish', False)
    })

    strategies.append({
        'name': 'W_MACD_ADX_D_Stoch_oversold',
        'entry': lambda r: r['stoch_k_14'] < 20 and r['rsi_14'] < 40 and r['above_ema_50'],
        'exit': lambda r: r['stoch_k_14'] > 70 or r['rsi_14'] > 60,
        'sl': 8, 'tp': 15, 'hold': 20,
        'weekly_filter': lambda w: w.get('w_macd_bullish', False) and w.get('w_adx_trending', False)
    })

    strategies.append({
        'name': 'W_all_bull_D_WR2_oversold',
        'entry': lambda r: r['williams_r_2'] < -95 and r['above_sma_200'],
        'exit': lambda r: r['williams_r_2'] > -30,
        'sl': 8, 'tp': 12, 'hold': 10,
        'weekly_filter': lambda w: w.get('w_rsi_bullish', False) and w.get('w_supertrend_bull', False) and w.get('w_macd_bullish', False)
    })

    strategies.append({
        'name': 'W_EMA_RSI_D_CCI_oversold',
        'entry': lambda r: r['cci_20'] < -100 and r['above_ema_200'],
        'exit': lambda r: r['cci_20'] > 50,
        'sl': 10, 'tp': 15, 'hold': 20,
        'weekly_filter': lambda w: w.get('w_ema_10_above_20', False) and w.get('w_rsi_not_ob', False)
    })

    print(f"Total strategies defined: {len(strategies)}")
    return strategies


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_single_strategy(strat: Dict, all_enriched: Dict[str, pd.DataFrame],
                        weekly_data: Dict[str, pd.DataFrame]) -> Dict:
    """Run a single strategy and return result dict."""
    name = strat['name']
    try:
        result = backtest_strategy(
            all_enriched=all_enriched,
            entry_fn=strat['entry'],
            exit_fn=strat['exit'],
            stop_loss_pct=strat['sl'],
            take_profit_pct=strat['tp'],
            max_hold_days=strat['hold'],
            weekly_filter_fn=strat.get('weekly_filter'),
            weekly_data=weekly_data
        )
        result.name = name
        return asdict(result)
    except Exception as e:
        print(f"  ERROR in {name}: {e}")
        return {'name': name, 'error': str(e)}


def main():
    start_time = time.time()
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY RESEARCH SCANNER")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Load data
    print("STEP 1: Loading all daily data...")
    all_data = load_all_daily_data()
    print(f"  Loaded {len(all_data)} stocks")
    print()

    # Step 2: Compute indicators
    print("STEP 2: Computing indicators for all stocks...")
    all_enriched = {}
    weekly_data = {}
    total = len(all_data)

    for idx, (sym, df) in enumerate(all_data.items(), 1):
        if idx % 50 == 0 or idx == 1:
            print(f"  Processing {idx}/{total}: {sym}...")
        try:
            enriched = compute_all_indicators(df)
            all_enriched[sym] = enriched

            # Weekly data
            w = compute_weekly_indicators(df)
            if len(w) > 0:
                weekly_data[sym] = w
        except Exception as e:
            print(f"  WARNING: Skipping {sym}: {e}")
            continue

    print(f"  Enriched {len(all_enriched)} stocks with indicators")
    print(f"  Weekly data for {len(weekly_data)} stocks")
    elapsed = time.time() - start_time
    print(f"  Time so far: {elapsed:.0f}s")
    print()

    # Step 3: Define strategies
    print("STEP 3: Defining strategies...")
    strategies = define_all_strategies()
    print()

    # Step 4: Run all strategies
    print("STEP 4: Running all strategies...")
    results = []

    for idx, strat in enumerate(strategies, 1):
        name = strat['name']
        print(f"  [{idx}/{len(strategies)}] {name}...", end='', flush=True)

        try:
            result = run_single_strategy(strat, all_enriched, weekly_data)
            results.append(result)

            if 'error' not in result:
                trades = result.get('total_trades', 0)
                wr = result.get('win_rate', 0)
                pf = result.get('profit_factor', 0)
                calmar = result.get('calmar_ratio', 0)
                print(f" trades={trades}, WR={wr}%, PF={pf}, Calmar={calmar}")
            else:
                print(f" ERROR: {result['error']}")
        except Exception as e:
            print(f" EXCEPTION: {e}")
            results.append({'name': name, 'error': str(e)})

        # Periodic save
        if idx % 20 == 0:
            _save_interim_results(results)

    # Step 5: Filter and rank
    print()
    print("=" * 80)
    print("STEP 5: FILTERING RESULTS")
    print("=" * 80)

    # Convert to DataFrame
    df_results = pd.DataFrame([r for r in results if 'error' not in r])

    if len(df_results) == 0:
        print("No valid results!")
        return

    # Save all results
    df_results.to_csv(CSV_RESULTS_PATH, index=False)
    print(f"All results saved to: {CSV_RESULTS_PATH}")

    # Filter for criteria
    qualifying = df_results[
        (df_results['total_trades'] >= 250) &
        (df_results['win_rate'] >= 65) &
        (df_results['calmar_ratio'] >= 1.0) &
        (df_results['profit_factor'] >= 1.5)
    ].sort_values('calmar_ratio', ascending=False)

    print(f"\nStrategies meeting ALL criteria (250+ trades, 65%+ WR, 1+ Calmar, 1.5+ PF):")
    print(f"  {len(qualifying)} strategies found!")

    if len(qualifying) > 0:
        print()
        print(qualifying[['name', 'total_trades', 'win_rate', 'calmar_ratio',
                          'profit_factor', 'sharpe_ratio', 'total_return_pct',
                          'max_drawdown_pct', 'avg_hold_days', 'symbols_traded']].to_string())

    # Also show "near miss" strategies
    near_miss = df_results[
        (df_results['total_trades'] >= 200) &
        (df_results['win_rate'] >= 60) &
        (df_results['profit_factor'] >= 1.3)
    ].sort_values('win_rate', ascending=False)

    print(f"\n\nNear-miss strategies (200+ trades, 60%+ WR, 1.3+ PF):")
    print(f"  {len(near_miss)} strategies found!")

    if len(near_miss) > 0:
        print()
        cols = ['name', 'total_trades', 'win_rate', 'calmar_ratio',
                'profit_factor', 'sharpe_ratio', 'total_return_pct',
                'max_drawdown_pct', 'avg_hold_days', 'symbols_traded']
        print(near_miss[cols].head(30).to_string())

    # Top strategies by different criteria
    print("\n\n" + "=" * 80)
    print("TOP STRATEGIES BY CATEGORY")
    print("=" * 80)

    for metric, label in [
        ('win_rate', 'Win Rate'),
        ('calmar_ratio', 'Calmar Ratio'),
        ('profit_factor', 'Profit Factor'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('total_return_pct', 'Total Return %')
    ]:
        valid = df_results[df_results['total_trades'] >= 100].sort_values(metric, ascending=False)
        print(f"\nTop 10 by {label} (100+ trades):")
        if len(valid) > 0:
            print(valid[['name', 'total_trades', 'win_rate', metric, 'profit_factor', 'avg_hold_days']].head(10).to_string())

    # Save JSON results
    json_results = {
        'run_date': datetime.now().isoformat(),
        'total_stocks': len(all_enriched),
        'total_strategies': len(strategies),
        'qualifying_strategies': qualifying.to_dict('records') if len(qualifying) > 0 else [],
        'near_miss_strategies': near_miss.to_dict('records') if len(near_miss) > 0 else [],
        'all_results': df_results.to_dict('records'),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"CSV saved to: {CSV_RESULTS_PATH}")


def _save_interim_results(results):
    """Save interim results to JSON."""
    valid = [r for r in results if 'error' not in r]
    if valid:
        with open(RESULTS_PATH, 'w') as f:
            json.dump({'interim': True, 'results': valid}, f, indent=2, default=str)


if __name__ == '__main__':
    main()
