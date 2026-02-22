"""
Strategy Exploration Backtest Engine
=====================================

Generic backtest engine for exploring diverse stock trading strategies
on the Indian market (Nifty 500 universe).

Features:
- Pluggable entry/exit/ranking functions
- Full indicator enrichment (17+ indicators)
- Concentration bias detection
- Profit factor, Calmar, Sharpe, Sortino metrics
- Sector-limited portfolio construction
- Transaction cost modelling

Usage:
    universe, price_data = preload_exploration_data('2015-01-01', '2025-12-31')
    enriched = enrich_with_indicators(price_data)
    config = StrategyConfig(name='test', start_date='2015-01-01', end_date='2025-12-31')
    explorer = StrategyExplorer(universe, enriched, config)
    result = explorer.run(entry_fn, rank_fn, exit_fn)
"""

import sqlite3
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple, Optional, Any
from pathlib import Path

from services.nifty500_universe import load_nifty500, Nifty500Universe
from services.technical_indicators import (
    add_ema_signals, add_rsi_signals, add_supertrend_signals,
    add_bollinger_signals, add_adx_signals, add_macd_signals,
    add_stoch_signals, add_donchian_signals, calc_ema, calc_atr,
    add_obv_signals, add_mfi_signals, add_cci_signals,
    add_williams_r_signals, calc_keltner_channels, calc_rsi,
)

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'
TRADING_DAYS_PER_YEAR = 252
WARMUP_DAYS = 400  # days before start_date for indicator computation
TRANSACTION_COST_PCT = 0.002  # 0.2% round-trip (brokerage + STT + stamp)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StrategyConfig:
    """Configuration for a strategy exploration run."""
    name: str = 'unnamed'
    start_date: str = '2015-01-01'
    end_date: str = '2025-12-31'
    initial_capital: float = 10_000_000  # Rs 1 Crore
    portfolio_size: int = 25
    max_sector_pct: float = 0.25  # max 25% in one sector
    max_stocks_per_sector: int = 8
    rebalance_freq: str = 'monthly'  # monthly, quarterly, semi_annual
    equity_allocation_pct: float = 0.95
    allow_short: bool = False  # if True, engine runs short_entry_fn / short_exit_fn too


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class ExplorationResult:
    """Complete result from a strategy exploration run."""
    strategy_name: str
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    profit_factor: float
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    final_value: float
    total_return_pct: float
    top3_pnl_pct: float
    top3_symbols: list
    cagr_ex_top3: float
    trade_list: list
    equity_curve: dict
    exit_reason_counts: dict


# =============================================================================
# Data Loading
# =============================================================================

def preload_exploration_data(
    start_date: str,
    end_date: str,
    db_path: Path = None,
) -> Tuple[Nifty500Universe, Dict[str, pd.DataFrame]]:
    """
    Load universe and OHLCV data from SQLite.
    Loads WARMUP_DAYS before start_date for indicator computation.
    """
    db_path = db_path or DB_PATH
    universe = load_nifty500()
    symbols = universe.symbols

    # Calculate warmup start
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    warmup_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')

    conn = sqlite3.connect(db_path)
    placeholders = ','.join('?' * len(symbols))
    query = f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol IN ({placeholders})
          AND timeframe = 'day'
          AND date >= ? AND date <= ?
        ORDER BY symbol, date
    """
    params = symbols + [warmup_start, end_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    price_data = {}
    for symbol, group in df.groupby('symbol'):
        sdf = group.set_index('date').sort_index()
        sdf = sdf[['open', 'high', 'low', 'close', 'volume']]
        if len(sdf) >= 100:  # need at least 100 days
            price_data[symbol] = sdf

    print(f"Loaded {len(price_data)} stocks with data ({len(df):,} candles)")
    return universe, price_data


# =============================================================================
# Indicator Enrichment
# =============================================================================

def enrich_with_indicators(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Compute ALL indicators for each symbol's DataFrame.
    Returns Dict[symbol -> enriched DataFrame].
    """
    enriched = {}
    done = 0
    total = len(price_data)

    for symbol, df in price_data.items():
        try:
            edf = _enrich_single(df)
            enriched[symbol] = edf
        except Exception as e:
            logger.debug(f"Failed to enrich {symbol}: {e}")
            # Still include with basic data
            enriched[symbol] = df.copy()

        done += 1
        if done % 25 == 0 or done == total:
            print(f"  Enriched {done}/{total} stocks...", flush=True)

    print(f"Enrichment complete: {len(enriched)} stocks", flush=True)
    return enriched


CACHE_DIR = Path(__file__).parent.parent / 'backtest_data'


def save_enriched_cache(enriched: Dict[str, pd.DataFrame], start_date: str, end_date: str):
    """Save enriched data to pickle for fast reuse."""
    cache_path = CACHE_DIR / f'enriched_{start_date}_{end_date}.pkl'
    with open(cache_path, 'wb') as f:
        pickle.dump(enriched, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Saved enriched cache: {cache_path.name} ({size_mb:.1f} MB)", flush=True)
    return cache_path


def load_enriched_cache(start_date: str, end_date: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Load enriched data from pickle cache if it exists."""
    cache_path = CACHE_DIR / f'enriched_{start_date}_{end_date}.pkl'
    if cache_path.exists():
        print(f"Loading enriched cache: {cache_path.name}...", flush=True)
        with open(cache_path, 'rb') as f:
            enriched = pickle.load(f)
        print(f"Loaded {len(enriched)} stocks from cache", flush=True)
        return enriched
    return None


def _enrich_single(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators to a single stock's DataFrame."""
    edf = df.copy()

    # SMA 50, 200
    edf['sma_50'] = edf['close'].rolling(50).mean()
    edf['sma_200'] = edf['close'].rolling(200).mean()

    # EMA 9, 21, 50, 200
    edf['ema_9'] = calc_ema(edf['close'], 9)
    edf['ema_21'] = calc_ema(edf['close'], 21)
    edf['ema_50'] = calc_ema(edf['close'], 50)
    edf['ema_200'] = calc_ema(edf['close'], 200)

    # EMA crossover columns for various pairs
    for fast, slow in [(10, 30), (20, 50), (50, 200)]:
        f_col = f'ema_{fast}'
        s_col = f'ema_{slow}'
        if f_col not in edf.columns:
            edf[f_col] = calc_ema(edf['close'], fast)
        if s_col not in edf.columns:
            edf[s_col] = calc_ema(edf['close'], slow)
        bull = edf[f_col] > edf[s_col]
        edf[f'ema_cross_up_{fast}_{slow}'] = bull & ~bull.shift(1).fillna(False)
        edf[f'ema_cross_down_{fast}_{slow}'] = ~bull & bull.shift(1).fillna(True)
        edf[f'ema_bullish_{fast}_{slow}'] = bull

    # RSI 14 and RSI 2
    edf['rsi_14'] = calc_rsi(edf['close'], 14)
    edf['rsi_2'] = calc_rsi(edf['close'], 2)

    # MACD (12/26/9)
    try:
        tmp = add_macd_signals(edf)
        for col in ['macd', 'macd_signal', 'macd_histogram', 'macd_bullish',
                     'macd_cross_up', 'macd_cross_down', 'macd_positive']:
            edf[col] = tmp[col]
    except Exception:
        pass

    # SuperTrend (10, 3.0)
    try:
        tmp = add_supertrend_signals(edf, 10, 3.0)
        edf['st_10_3'] = tmp['supertrend']
        edf['st_dir_10_3'] = tmp['supertrend_direction']
        edf['st_flip_up_10_3'] = tmp['supertrend_flip_up']
        edf['st_flip_down_10_3'] = tmp['supertrend_flip_down']
        edf['st_bullish_10_3'] = tmp['supertrend_bullish']
    except Exception:
        pass

    # SuperTrend (7, 2.0)
    try:
        tmp = add_supertrend_signals(edf, 7, 2.0)
        edf['st_dir_7_2'] = tmp['supertrend_direction']
        edf['st_flip_up_7_2'] = tmp['supertrend_flip_up']
        edf['st_bullish_7_2'] = tmp['supertrend_bullish']
    except Exception:
        pass

    # ADX (14)
    try:
        tmp = add_adx_signals(edf, 14)
        edf['adx'] = tmp['adx']
        edf['plus_di'] = tmp['plus_di']
        edf['minus_di'] = tmp['minus_di']
        edf['adx_trending'] = tmp['adx_trending']
        edf['adx_bullish'] = tmp['adx_bullish']
    except Exception:
        pass

    # Bollinger Bands (20, 2.0)
    try:
        tmp = add_bollinger_signals(edf, 20, 2.0)
        edf['bb_middle'] = tmp['bb_middle']
        edf['bb_upper'] = tmp['bb_upper']
        edf['bb_lower'] = tmp['bb_lower']
        edf['bb_squeeze'] = tmp['bb_squeeze']
        edf['bb_below_lower'] = tmp['bb_below_lower']
        edf['bb_above_upper'] = tmp['bb_above_upper']
        edf['bb_width'] = tmp['bb_width']
    except Exception:
        pass

    # Keltner Channels (20, 2.0 ATR)
    try:
        kc_mid, kc_upper, kc_lower = calc_keltner_channels(edf, 20, 10, 2.0)
        edf['kc_middle'] = kc_mid
        edf['kc_upper'] = kc_upper
        edf['kc_lower'] = kc_lower
        edf['kc_below_lower'] = edf['close'] < kc_lower
    except Exception:
        pass

    # Donchian (20 and 50)
    try:
        tmp20 = add_donchian_signals(edf, 20)
        edf['dc_upper_20'] = tmp20['dc_upper']
        edf['dc_lower_20'] = tmp20['dc_lower']
        edf['dc_breakout_up_20'] = tmp20['dc_breakout_up']
    except Exception:
        pass
    try:
        tmp50 = add_donchian_signals(edf, 50)
        edf['dc_upper_50'] = tmp50['dc_upper']
        edf['dc_lower_50'] = tmp50['dc_lower']
        edf['dc_breakout_up_50'] = tmp50['dc_breakout_up']
    except Exception:
        pass

    # Stochastic (14, 3)
    try:
        tmp = add_stoch_signals(edf, 14, 3)
        edf['stoch_k'] = tmp['stoch_k']
        edf['stoch_d'] = tmp['stoch_d']
        edf['stoch_oversold'] = tmp['stoch_oversold']
        edf['stoch_overbought'] = tmp['stoch_overbought']
    except Exception:
        pass

    # CCI (20)
    try:
        tmp = add_cci_signals(edf, 20)
        edf['cci'] = tmp['cci']
        edf['cci_oversold'] = tmp['cci_oversold']
        edf['cci_overbought'] = tmp['cci_overbought']
    except Exception:
        pass

    # Williams %R (14)
    try:
        tmp = add_williams_r_signals(edf, 14)
        edf['williams_r'] = tmp['williams_r']
        edf['wr_oversold'] = tmp['wr_oversold']
        edf['wr_overbought'] = tmp['wr_overbought']
    except Exception:
        pass

    # OBV
    try:
        tmp = add_obv_signals(edf)
        edf['obv'] = tmp['obv']
        edf['obv_ema'] = tmp['obv_ema']
        edf['obv_bullish'] = tmp['obv_bullish']
    except Exception:
        pass

    # MFI (14)
    try:
        tmp = add_mfi_signals(edf, 14)
        edf['mfi'] = tmp['mfi']
        edf['mfi_oversold'] = tmp['mfi_oversold']
        edf['mfi_overbought'] = tmp['mfi_overbought']
    except Exception:
        pass

    # ATR (14)
    try:
        edf['atr_14'] = calc_atr(edf, 14)
    except Exception:
        pass

    # 52-week high (rolling 252-day max)
    edf['high_52w'] = edf['high'].rolling(252, min_periods=50).max()
    edf['dist_from_52w'] = edf['close'] / edf['high_52w']

    # Volume average (20-day)
    edf['vol_avg_20'] = edf['volume'].rolling(20).mean()
    edf['vol_ratio'] = edf['volume'] / edf['vol_avg_20'].replace(0, np.nan)

    # Weekly EMA 20/50 (resample to weekly, compute, forward-fill to daily)
    try:
        weekly = edf['close'].resample('W').last().dropna()
        w_ema_20 = calc_ema(weekly, 20)
        w_ema_50 = calc_ema(weekly, 50)
        edf['weekly_ema_20'] = w_ema_20.reindex(edf.index, method='ffill')
        edf['weekly_ema_50'] = w_ema_50.reindex(edf.index, method='ffill')
        edf['weekly_trend_up'] = edf['weekly_ema_20'] > edf['weekly_ema_50']
    except Exception:
        edf['weekly_trend_up'] = False

    # Momentum returns (for ranking)
    edf['ret_1m'] = edf['close'].pct_change(21)
    edf['ret_3m'] = edf['close'].pct_change(63)
    edf['ret_6m'] = edf['close'].pct_change(126)
    edf['ret_12m'] = edf['close'].pct_change(252)

    return edf


# =============================================================================
# Utility: get calc_rsi from technical_indicators
# =============================================================================

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# =============================================================================
# Entry Signal Functions
# =============================================================================

def entry_ema_crossover(df, idx, fast=20, slow=50):
    """True if fast EMA just crossed above slow EMA."""
    col = f'ema_cross_up_{fast}_{slow}'
    if col in df.columns:
        return bool(df[col].iloc[idx])
    # Fallback: compute on-the-fly
    f_col = f'ema_{fast}' if f'ema_{fast}' in df.columns else None
    s_col = f'ema_{slow}' if f'ema_{slow}' in df.columns else None
    if f_col and s_col and idx > 0:
        return df[f_col].iloc[idx] > df[s_col].iloc[idx] and df[f_col].iloc[idx-1] <= df[s_col].iloc[idx-1]
    return False


def entry_price_above_sma(df, idx, period=200):
    """True if close > SMA(period)."""
    col = f'sma_{period}'
    if col not in df.columns:
        return False
    val = df[col].iloc[idx]
    if pd.isna(val):
        return False
    return df['close'].iloc[idx] > val


def entry_rsi_oversold(df, idx, col='rsi_14', threshold=30):
    """True if RSI < threshold AND close > SMA200 (uptrend filter)."""
    if col not in df.columns or 'sma_200' not in df.columns:
        return False
    rsi_val = df[col].iloc[idx]
    sma_val = df['sma_200'].iloc[idx]
    if pd.isna(rsi_val) or pd.isna(sma_val):
        return False
    return rsi_val < threshold and df['close'].iloc[idx] > sma_val


def entry_bollinger_lower(df, idx):
    """True if close < lower Bollinger AND close > SMA200."""
    if 'bb_below_lower' not in df.columns or 'sma_200' not in df.columns:
        return False
    sma_val = df['sma_200'].iloc[idx]
    if pd.isna(sma_val):
        return False
    return bool(df['bb_below_lower'].iloc[idx]) and df['close'].iloc[idx] > sma_val


def entry_keltner_lower(df, idx):
    """True if close < lower Keltner AND close > SMA200."""
    if 'kc_below_lower' not in df.columns or 'sma_200' not in df.columns:
        return False
    sma_val = df['sma_200'].iloc[idx]
    if pd.isna(sma_val):
        return False
    return bool(df['kc_below_lower'].iloc[idx]) and df['close'].iloc[idx] > sma_val


def entry_supertrend_bullish(df, idx, period=10, mult=3):
    """True if SuperTrend just flipped to bullish."""
    col = f'st_flip_up_{period}_{mult}'
    if col not in df.columns:
        return False
    return bool(df[col].iloc[idx])


def entry_donchian_breakout(df, idx, period=50):
    """True if close > N-day high (breakout)."""
    col = f'dc_breakout_up_{period}'
    if col not in df.columns:
        return False
    return bool(df[col].iloc[idx])


def entry_macd_crossover(df, idx):
    """True if MACD crossed above signal line."""
    if 'macd_cross_up' not in df.columns:
        return False
    return bool(df['macd_cross_up'].iloc[idx])


def entry_adx_trending(df, idx, threshold=25):
    """True if ADX > threshold AND DI+ > DI-."""
    if 'adx' not in df.columns or 'plus_di' not in df.columns:
        return False
    adx_val = df['adx'].iloc[idx]
    if pd.isna(adx_val):
        return False
    return adx_val > threshold and df['plus_di'].iloc[idx] > df['minus_di'].iloc[idx]


def entry_volume_breakout(df, idx, mult=2.0):
    """True if volume > mult x 20d avg AND close > SMA50."""
    if 'vol_ratio' not in df.columns or 'sma_50' not in df.columns:
        return False
    vr = df['vol_ratio'].iloc[idx]
    sma_val = df['sma_50'].iloc[idx]
    if pd.isna(vr) or pd.isna(sma_val):
        return False
    return vr > mult and df['close'].iloc[idx] > sma_val


def entry_weekly_trend_daily_pullback(df, idx, rsi_threshold=40):
    """True if weekly EMA20 > weekly EMA50 AND daily RSI < threshold."""
    if 'weekly_trend_up' not in df.columns or 'rsi_14' not in df.columns:
        return False
    wtu = df['weekly_trend_up'].iloc[idx]
    rsi_val = df['rsi_14'].iloc[idx]
    if pd.isna(rsi_val):
        return False
    return bool(wtu) and rsi_val < rsi_threshold


def entry_52w_high_proximity(df, idx, threshold=0.90):
    """True if close is within threshold of 52-week high."""
    if 'dist_from_52w' not in df.columns:
        return False
    val = df['dist_from_52w'].iloc[idx]
    if pd.isna(val):
        return False
    return val >= threshold


def entry_cci_oversold(df, idx, threshold=-100):
    """True if CCI < threshold AND price > EMA50."""
    if 'cci' not in df.columns or 'ema_50' not in df.columns:
        return False
    cci_val = df['cci'].iloc[idx]
    ema_val = df['ema_50'].iloc[idx]
    if pd.isna(cci_val) or pd.isna(ema_val):
        return False
    return cci_val < threshold and df['close'].iloc[idx] > ema_val


def entry_stoch_oversold(df, idx, threshold=20):
    """True if Stoch < threshold AND close > SMA200."""
    if 'stoch_k' not in df.columns or 'sma_200' not in df.columns:
        return False
    sk = df['stoch_k'].iloc[idx]
    sma_val = df['sma_200'].iloc[idx]
    if pd.isna(sk) or pd.isna(sma_val):
        return False
    return sk < threshold and df['close'].iloc[idx] > sma_val


def entry_williams_r_oversold(df, idx, threshold=-80):
    """True if Williams %R < threshold AND close > SMA200."""
    if 'williams_r' not in df.columns or 'sma_200' not in df.columns:
        return False
    wr_val = df['williams_r'].iloc[idx]
    sma_val = df['sma_200'].iloc[idx]
    if pd.isna(wr_val) or pd.isna(sma_val):
        return False
    return wr_val < threshold and df['close'].iloc[idx] > sma_val


def entry_mfi_oversold(df, idx, threshold=20):
    """True if MFI < threshold AND close > SMA200."""
    if 'mfi' not in df.columns or 'sma_200' not in df.columns:
        return False
    mfi_val = df['mfi'].iloc[idx]
    sma_val = df['sma_200'].iloc[idx]
    if pd.isna(mfi_val) or pd.isna(sma_val):
        return False
    return mfi_val < threshold and df['close'].iloc[idx] > sma_val


def entry_bollinger_squeeze_breakout(df, idx):
    """True if in Bollinger squeeze and just broke above upper band."""
    if 'bb_squeeze' not in df.columns or 'bb_above_upper' not in df.columns:
        return False
    if idx < 1:
        return False
    # Was in squeeze yesterday, broke out today
    was_squeeze = bool(df['bb_squeeze'].iloc[idx-1])
    breakout = bool(df['bb_above_upper'].iloc[idx])
    return was_squeeze and breakout


# =============================================================================
# Ranking Functions
# =============================================================================

def rank_momentum_12m(df, idx):
    """12-month price return (higher = better)."""
    if 'ret_12m' not in df.columns:
        return 0.0
    val = df['ret_12m'].iloc[idx]
    return float(val) if not pd.isna(val) else 0.0


def rank_momentum_6m(df, idx):
    """6-month price return."""
    if 'ret_6m' not in df.columns:
        return 0.0
    val = df['ret_6m'].iloc[idx]
    return float(val) if not pd.isna(val) else 0.0


def rank_rsi_strength(df, idx):
    """RSI value (higher = stronger momentum)."""
    if 'rsi_14' not in df.columns:
        return 50.0
    val = df['rsi_14'].iloc[idx]
    return float(val) if not pd.isna(val) else 50.0


def rank_distance_from_high(df, idx):
    """Close / 52w high (closer to 1 = near ATH = better)."""
    if 'dist_from_52w' not in df.columns:
        return 0.0
    val = df['dist_from_52w'].iloc[idx]
    return float(val) if not pd.isna(val) else 0.0


def rank_composite_momentum(df, idx):
    """Weighted: 40% 12m + 30% 6m + 30% 3m returns."""
    r12 = df['ret_12m'].iloc[idx] if 'ret_12m' in df.columns else 0
    r6 = df['ret_6m'].iloc[idx] if 'ret_6m' in df.columns else 0
    r3 = df['ret_3m'].iloc[idx] if 'ret_3m' in df.columns else 0
    r12 = float(r12) if not pd.isna(r12) else 0
    r6 = float(r6) if not pd.isna(r6) else 0
    r3 = float(r3) if not pd.isna(r3) else 0
    return 0.4 * r12 + 0.3 * r6 + 0.3 * r3


# =============================================================================
# Exit Signal Functions
# =============================================================================

def exit_fixed_stop_loss(pos, df, idx, pct=10):
    """Exit if close < entry_price * (1 - pct/100)."""
    close = df['close'].iloc[idx]
    stop = pos['entry_price'] * (1 - pct / 100)
    if close < stop:
        return True, f'SL_{pct}pct', close
    return False, '', 0


def exit_trailing_stop(pos, df, idx, pct=15):
    """Exit if close < peak_since_entry * (1 - pct/100)."""
    close = df['close'].iloc[idx]
    stop = pos['peak_price'] * (1 - pct / 100)
    if close < stop:
        return True, f'Trail_{pct}pct', close
    return False, '', 0


def exit_ath_drawdown(pos, df, idx, pct=20):
    """Exit if close < ATH_since_entry * (1 - pct/100)."""
    close = df['close'].iloc[idx]
    stop = pos['peak_price'] * (1 - pct / 100)
    if close < stop:
        return True, f'ATH_DD_{pct}pct', close
    return False, '', 0


def exit_time_based(pos, df, idx, max_days=90):
    """Exit if holding > max_days."""
    current_date = df.index[idx]
    hold_days = (current_date - pos['entry_date']).days
    if hold_days >= max_days:
        return True, f'Time_{max_days}d', df['close'].iloc[idx]
    return False, '', 0


def exit_take_profit(pos, df, idx, pct=30):
    """Exit if close > entry_price * (1 + pct/100)."""
    close = df['close'].iloc[idx]
    target = pos['entry_price'] * (1 + pct / 100)
    if close >= target:
        return True, f'TP_{pct}pct', close
    return False, '', 0


def exit_indicator_reversal(pos, df, idx, indicator='supertrend_10_3'):
    """Exit if indicator flips bearish."""
    if indicator == 'supertrend_10_3':
        col = 'st_flip_down_10_3'
        if col in df.columns and bool(df[col].iloc[idx]):
            return True, 'ST_Flip', df['close'].iloc[idx]
    elif indicator == 'macd':
        if 'macd_cross_down' in df.columns and bool(df['macd_cross_down'].iloc[idx]):
            return True, 'MACD_Flip', df['close'].iloc[idx]
    elif indicator == 'ema':
        col = 'ema_cross_down_20_50'
        if col in df.columns and bool(df[col].iloc[idx]):
            return True, 'EMA_Flip', df['close'].iloc[idx]
    return False, '', 0


def exit_rsi_overbought(pos, df, idx, threshold=70, col='rsi_14'):
    """Exit if RSI > threshold."""
    if col not in df.columns:
        return False, '', 0
    val = df[col].iloc[idx]
    if pd.isna(val):
        return False, '', 0
    if val > threshold:
        return True, f'RSI_{threshold}', df['close'].iloc[idx]
    return False, '', 0


def exit_rsi2_overbought(pos, df, idx, threshold=90):
    """Exit if RSI(2) > threshold."""
    return exit_rsi_overbought(pos, df, idx, threshold=threshold, col='rsi_2')


def exit_bollinger_mid(pos, df, idx):
    """Exit if close > Bollinger middle band."""
    if 'bb_middle' not in df.columns:
        return False, '', 0
    val = df['bb_middle'].iloc[idx]
    if pd.isna(val):
        return False, '', 0
    if df['close'].iloc[idx] > val:
        return True, 'BB_Mid', df['close'].iloc[idx]
    return False, '', 0


def exit_bollinger_upper(pos, df, idx):
    """Exit if close > Bollinger upper band."""
    if 'bb_above_upper' not in df.columns:
        return False, '', 0
    if bool(df['bb_above_upper'].iloc[idx]):
        return True, 'BB_Upper', df['close'].iloc[idx]
    return False, '', 0


def exit_keltner_mid(pos, df, idx):
    """Exit if close > Keltner middle channel."""
    if 'kc_middle' not in df.columns:
        return False, '', 0
    val = df['kc_middle'].iloc[idx]
    if pd.isna(val):
        return False, '', 0
    if df['close'].iloc[idx] > val:
        return True, 'KC_Mid', df['close'].iloc[idx]
    return False, '', 0


def exit_cci_overbought(pos, df, idx, threshold=100):
    """Exit if CCI > threshold."""
    if 'cci' not in df.columns:
        return False, '', 0
    val = df['cci'].iloc[idx]
    if pd.isna(val):
        return False, '', 0
    if val > threshold:
        return True, f'CCI_{threshold}', df['close'].iloc[idx]
    return False, '', 0


def exit_stoch_overbought(pos, df, idx, threshold=80):
    """Exit if Stochastic > threshold."""
    if 'stoch_k' not in df.columns:
        return False, '', 0
    val = df['stoch_k'].iloc[idx]
    if pd.isna(val):
        return False, '', 0
    if val > threshold:
        return True, f'Stoch_{threshold}', df['close'].iloc[idx]
    return False, '', 0


def exit_williams_r_overbought(pos, df, idx, threshold=-20):
    """Exit if Williams %R > threshold."""
    if 'williams_r' not in df.columns:
        return False, '', 0
    val = df['williams_r'].iloc[idx]
    if pd.isna(val):
        return False, '', 0
    if val > threshold:
        return True, 'WR_OB', df['close'].iloc[idx]
    return False, '', 0


def exit_mfi_overbought(pos, df, idx, threshold=80):
    """Exit if MFI > threshold."""
    if 'mfi' not in df.columns:
        return False, '', 0
    val = df['mfi'].iloc[idx]
    if pd.isna(val):
        return False, '', 0
    if val > threshold:
        return True, f'MFI_{threshold}', df['close'].iloc[idx]
    return False, '', 0


def exit_donchian_low(pos, df, idx, period=20):
    """Exit if close < Donchian lower channel."""
    col = f'dc_lower_{period}'
    if col not in df.columns:
        return False, '', 0
    val = df[col].iloc[idx]
    if pd.isna(val):
        return False, '', 0
    if df['close'].iloc[idx] < val:
        return True, f'DC_Low_{period}', df['close'].iloc[idx]
    return False, '', 0


def exit_adx_weakening(pos, df, idx, threshold=20):
    """Exit if ADX drops below threshold."""
    if 'adx' not in df.columns:
        return False, '', 0
    val = df['adx'].iloc[idx]
    if pd.isna(val):
        return False, '', 0
    if val < threshold:
        return True, f'ADX_Weak_{threshold}', df['close'].iloc[idx]
    return False, '', 0


def make_combined_exit(exit_rules):
    """
    Create a combined exit function from a list of (exit_fn, kwargs) tuples.
    First triggered exit wins.
    """
    def combined_exit(pos, df, idx):
        for exit_fn, kwargs in exit_rules:
            triggered, reason, price = exit_fn(pos, df, idx, **kwargs)
            if triggered:
                return triggered, reason, price
        return False, '', 0
    return combined_exit


# =============================================================================
# Strategy Explorer
# =============================================================================

class StrategyExplorer:
    """Main simulation engine for strategy exploration."""

    def __init__(self, universe: Nifty500Universe, enriched_data: Dict[str, pd.DataFrame],
                 config: StrategyConfig):
        self.universe = universe
        self.data = enriched_data
        self.config = config
        self.positions = {}  # symbol -> position dict
        self.trades = []  # completed trades
        self.daily_equity = {}  # date_str -> portfolio value
        self.cash = config.initial_capital * config.equity_allocation_pct
        self.debt_reserve = config.initial_capital * (1 - config.equity_allocation_pct)

    def run(self, entry_fn, rank_fn, exit_fn, allowed_symbols_fn=None) -> ExplorationResult:
        """
        Main simulation loop.
        entry_fn(df, idx) -> bool
        rank_fn(df, idx) -> float score
        exit_fn(pos, df, idx) -> (bool, reason, price)
        allowed_symbols_fn(date) -> set of symbols (optional, restricts entry universe)
        """
        config = self.config
        start_dt = pd.Timestamp(config.start_date)
        end_dt = pd.Timestamp(config.end_date)

        # Build common trading day index from all symbols
        all_dates = set()
        for sym, df in self.data.items():
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            all_dates.update(df.index[mask].tolist())
        trading_days = sorted(all_dates)

        if not trading_days:
            return self._empty_result()

        # Pre-build index maps: for each symbol, map date -> row index
        sym_date_idx = {}
        for sym, df in self.data.items():
            date_to_idx = {d: i for i, d in enumerate(df.index)}
            sym_date_idx[sym] = date_to_idx

        # Determine rebalance days
        rebalance_days = self._get_rebalance_days(trading_days)

        # Sector lookup
        sector_map = {}
        for s in self.universe.stocks:
            sector_map[s.symbol] = s.sector

        pos_size = (config.initial_capital * config.equity_allocation_pct) / config.portfolio_size

        for day_num, date in enumerate(trading_days):
            # 1. Update peak prices for all positions
            for sym in list(self.positions.keys()):
                if sym not in self.data:
                    continue
                idx_map = sym_date_idx.get(sym, {})
                idx = idx_map.get(date)
                if idx is None:
                    continue
                df = self.data[sym]
                close = df['close'].iloc[idx]
                if close > self.positions[sym]['peak_price']:
                    self.positions[sym]['peak_price'] = close

            # 2. Check exits for all positions
            for sym in list(self.positions.keys()):
                if sym not in self.data:
                    continue
                idx_map = sym_date_idx.get(sym, {})
                idx = idx_map.get(date)
                if idx is None:
                    continue
                df = self.data[sym]
                pos = self.positions[sym]

                triggered, reason, price = exit_fn(pos, df, idx)
                if triggered:
                    self._close_position(sym, price, date, reason)

            # 3. Check if we should enter new positions
            is_rebalance = date in rebalance_days
            need_fills = len(self.positions) < config.portfolio_size

            if is_rebalance or need_fills:
                slots = config.portfolio_size - len(self.positions)
                if slots > 0:
                    # Find candidates
                    candidates = []
                    held_symbols = set(self.positions.keys())

                    # Get allowed symbols for this date (if restricted)
                    if allowed_symbols_fn is not None:
                        allowed = allowed_symbols_fn(date)
                        scan_universe = [s for s in self.data if s in allowed]
                    else:
                        scan_universe = list(self.data.keys())

                    for sym in scan_universe:
                        if sym in held_symbols:
                            continue
                        idx_map = sym_date_idx.get(sym, {})
                        idx = idx_map.get(date)
                        if idx is None or idx < 250:  # need warmup
                            continue
                        df = self.data[sym]

                        try:
                            if entry_fn(df, idx):
                                score = rank_fn(df, idx)
                                candidates.append((sym, score))
                        except Exception:
                            continue

                    # Sort by score descending
                    candidates.sort(key=lambda x: x[1], reverse=True)

                    # Enter top N respecting sector limits
                    entered = 0
                    for sym, score in candidates:
                        if entered >= slots:
                            break

                        # Sector limit check
                        sector = sector_map.get(sym, 'Unknown')
                        sector_count = sum(
                            1 for s, p in self.positions.items()
                            if sector_map.get(s, 'Unknown') == sector
                        )
                        if sector_count >= config.max_stocks_per_sector:
                            continue

                        sector_value = sum(
                            p['qty'] * self.data[s]['close'].iloc[
                                sym_date_idx.get(s, {}).get(date, -1)
                            ]
                            for s, p in self.positions.items()
                            if sector_map.get(s, 'Unknown') == sector
                            and sym_date_idx.get(s, {}).get(date) is not None
                        )
                        total_value = self._portfolio_value(date, sym_date_idx)
                        if total_value > 0 and sector_value / total_value > config.max_sector_pct:
                            continue

                        # Enter position
                        idx_map = sym_date_idx.get(sym, {})
                        idx = idx_map.get(date)
                        if idx is None:
                            continue
                        entry_price = self.data[sym]['close'].iloc[idx]
                        if entry_price <= 0 or pd.isna(entry_price):
                            continue

                        qty = int(pos_size / entry_price)
                        if qty < 1:
                            continue

                        cost = qty * entry_price * (1 + TRANSACTION_COST_PCT / 2)
                        if cost > self.cash:
                            continue

                        self.cash -= cost
                        self.positions[sym] = {
                            'symbol': sym,
                            'entry_price': entry_price,
                            'entry_date': date,
                            'entry_idx': idx,
                            'qty': qty,
                            'peak_price': entry_price,
                        }
                        entered += 1

            # 4. Record daily equity
            equity = self._portfolio_value(date, sym_date_idx) + self.debt_reserve
            self.daily_equity[date.strftime('%Y-%m-%d')] = equity

        # Close any remaining positions at end
        last_date = trading_days[-1]
        for sym in list(self.positions.keys()):
            idx_map = sym_date_idx.get(sym, {})
            idx = idx_map.get(last_date)
            if idx is not None:
                price = self.data[sym]['close'].iloc[idx]
                self._close_position(sym, price, last_date, 'End_of_Period')
            else:
                # Find last available date
                df = self.data[sym]
                if len(df) > 0:
                    price = df['close'].iloc[-1]
                    self._close_position(sym, price, last_date, 'End_of_Period')

        return self._compute_result()

    def run_long_short(self, long_entry_fn, long_exit_fn, short_entry_fn, short_exit_fn,
                       rank_fn) -> ExplorationResult:
        """
        Long+Short simulation. Long and short books managed independently.
        short_entry_fn(df, idx) -> bool (signal to go short)
        short_exit_fn(pos, df, idx) -> (bool, reason, price)
        """
        config = self.config
        start_dt = pd.Timestamp(config.start_date)
        end_dt = pd.Timestamp(config.end_date)

        all_dates = set()
        for sym, df in self.data.items():
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            all_dates.update(df.index[mask].tolist())
        trading_days = sorted(all_dates)
        if not trading_days:
            return self._empty_result()

        sym_date_idx = {}
        for sym, df in self.data.items():
            sym_date_idx[sym] = {d: i for i, d in enumerate(df.index)}

        rebalance_days = self._get_rebalance_days(trading_days)
        sector_map = {s.symbol: s.sector for s in self.universe.stocks}

        # Split portfolio: half long, half short
        long_slots = config.portfolio_size // 2
        short_slots = config.portfolio_size - long_slots
        pos_size = (config.initial_capital * config.equity_allocation_pct) / config.portfolio_size

        for day_num, date in enumerate(trading_days):
            # 1. Update peaks/troughs for all positions
            for sym in list(self.positions.keys()):
                if sym not in self.data:
                    continue
                idx = sym_date_idx.get(sym, {}).get(date)
                if idx is None:
                    continue
                close = self.data[sym]['close'].iloc[idx]
                pos = self.positions[sym]
                if pos.get('direction', 'long') == 'long':
                    if close > pos['peak_price']:
                        pos['peak_price'] = close
                else:  # short
                    if close < pos.get('trough_price', pos['entry_price']):
                        pos['trough_price'] = close

            # 2. Check exits
            for sym in list(self.positions.keys()):
                if sym not in self.data:
                    continue
                idx = sym_date_idx.get(sym, {}).get(date)
                if idx is None:
                    continue
                df = self.data[sym]
                pos = self.positions[sym]
                direction = pos.get('direction', 'long')
                efn = long_exit_fn if direction == 'long' else short_exit_fn
                triggered, reason, price = efn(pos, df, idx)
                if triggered:
                    self._close_position(sym, price, date, reason)

            # 3. Enter new positions
            is_rebalance = date in rebalance_days
            long_count = sum(1 for p in self.positions.values() if p.get('direction', 'long') == 'long')
            short_count = sum(1 for p in self.positions.values() if p.get('direction', 'long') == 'short')
            need_long = long_count < long_slots
            need_short = short_count < short_slots

            if is_rebalance or need_long or need_short:
                held = set(self.positions.keys())

                for direction, need, max_slots, current_count, entry_fn in [
                    ('long', need_long, long_slots, long_count, long_entry_fn),
                    ('short', need_short, short_slots, short_count, short_entry_fn),
                ]:
                    if not need:
                        continue
                    slots = max_slots - current_count
                    if slots <= 0:
                        continue

                    candidates = []
                    for sym in self.data:
                        if sym in held:
                            continue
                        idx = sym_date_idx.get(sym, {}).get(date)
                        if idx is None or idx < 250:
                            continue
                        df = self.data[sym]
                        try:
                            if entry_fn(df, idx):
                                score = rank_fn(df, idx)
                                candidates.append((sym, score))
                        except Exception:
                            continue

                    # For shorts, reverse ranking (weakest stocks first)
                    candidates.sort(key=lambda x: x[1], reverse=(direction == 'long'))

                    entered = 0
                    for sym, score in candidates:
                        if entered >= slots:
                            break
                        sector = sector_map.get(sym, 'Unknown')
                        sector_count = sum(
                            1 for s, p in self.positions.items()
                            if sector_map.get(s, 'Unknown') == sector
                        )
                        if sector_count >= config.max_stocks_per_sector:
                            continue

                        idx = sym_date_idx.get(sym, {}).get(date)
                        if idx is None:
                            continue
                        entry_price = self.data[sym]['close'].iloc[idx]
                        if entry_price <= 0 or pd.isna(entry_price):
                            continue

                        qty = int(pos_size / entry_price)
                        if qty < 1:
                            continue

                        cost = qty * entry_price * (1 + TRANSACTION_COST_PCT / 2)
                        if cost > self.cash:
                            continue

                        self.cash -= cost
                        self.positions[sym] = {
                            'symbol': sym,
                            'entry_price': entry_price,
                            'entry_date': date,
                            'entry_idx': idx,
                            'qty': qty,
                            'peak_price': entry_price,
                            'trough_price': entry_price,
                            'direction': direction,
                        }
                        held.add(sym)
                        entered += 1

            # 4. Record daily equity
            equity = self._portfolio_value_ls(date, sym_date_idx) + self.debt_reserve
            self.daily_equity[date.strftime('%Y-%m-%d')] = equity

        # Close remaining
        last_date = trading_days[-1]
        for sym in list(self.positions.keys()):
            idx = sym_date_idx.get(sym, {}).get(last_date)
            price = self.data[sym]['close'].iloc[idx] if idx is not None else self.data[sym]['close'].iloc[-1]
            self._close_position(sym, price, last_date, 'End_of_Period')

        return self._compute_result()

    def _portfolio_value_ls(self, date, sym_date_idx):
        """Portfolio value for long+short book."""
        pos_value = 0
        for sym, pos in self.positions.items():
            idx = sym_date_idx.get(sym, {}).get(date)
            if idx is not None:
                close = self.data[sym]['close'].iloc[idx]
            else:
                close = pos['entry_price']
            if pos.get('direction', 'long') == 'long':
                pos_value += pos['qty'] * close
            else:  # short: profit when price falls
                pos_value += pos['qty'] * (2 * pos['entry_price'] - close)
        return self.cash + pos_value

    def _close_position(self, sym, exit_price, exit_date, reason):
        """Close a position and record the trade."""
        pos = self.positions.pop(sym)
        direction = pos.get('direction', 'long')
        if direction == 'long':
            pnl_pct = (exit_price / pos['entry_price'] - 1) * 100
            pnl_abs = pos['qty'] * (exit_price - pos['entry_price'])
        else:  # short
            pnl_pct = (pos['entry_price'] / exit_price - 1) * 100
            pnl_abs = pos['qty'] * (pos['entry_price'] - exit_price)
        hold_days = (exit_date - pos['entry_date']).days

        # Transaction cost on exit
        proceeds = pos['qty'] * exit_price * (1 - TRANSACTION_COST_PCT / 2)
        if direction == 'long':
            self.cash += proceeds
        else:
            # For shorts: return the collateral + profit (or - loss)
            self.cash += pos['qty'] * pos['entry_price'] + pnl_abs - pos['qty'] * exit_price * TRANSACTION_COST_PCT / 2

        self.trades.append({
            'symbol': sym,
            'entry_price': pos['entry_price'],
            'entry_date': pos['entry_date'],
            'exit_price': exit_price,
            'exit_date': exit_date,
            'exit_reason': reason,
            'pnl_pct': pnl_pct,
            'pnl_abs': pnl_abs,
            'hold_days': hold_days,
            'qty': pos['qty'],
        })

    def _portfolio_value(self, date, sym_date_idx):
        """Total portfolio value = cash + positions market value."""
        pos_value = 0
        for sym, pos in self.positions.items():
            idx_map = sym_date_idx.get(sym, {})
            idx = idx_map.get(date)
            if idx is not None:
                pos_value += pos['qty'] * self.data[sym]['close'].iloc[idx]
            else:
                pos_value += pos['qty'] * pos['entry_price']
        return self.cash + pos_value

    def _get_rebalance_days(self, trading_days):
        """Return set of rebalance dates based on config.rebalance_freq."""
        rebalance_set = set()
        freq = self.config.rebalance_freq

        if freq == 'monthly':
            months = list(range(1, 13))
        elif freq == 'quarterly':
            months = [1, 4, 7, 10]
        elif freq == 'semi_annual':
            months = [1, 7]
        else:
            months = list(range(1, 13))

        current_month = None
        for d in trading_days:
            if d.month in months and d.month != current_month:
                rebalance_set.add(d)
                current_month = d.month
            if d.month not in months:
                current_month = None

        return rebalance_set

    def _compute_result(self) -> ExplorationResult:
        """Compute all metrics from trades and equity curve."""
        config = self.config

        # Build equity series
        if self.daily_equity:
            dates = sorted(self.daily_equity.keys())
            values = [self.daily_equity[d] for d in dates]
            equity_series = pd.Series(values, index=pd.to_datetime(dates))
        else:
            return self._empty_result()

        initial = config.initial_capital
        final = values[-1] if values else initial

        # CAGR
        start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
        years = (end_dt - start_dt).days / 365.25
        cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 and initial > 0 else 0

        # Daily returns
        daily_returns = equity_series.pct_change().dropna()

        # Sharpe ratio (7% risk-free)
        if len(daily_returns) > 10 and daily_returns.std() > 0:
            excess = daily_returns.mean() - (0.07 / TRADING_DAYS_PER_YEAR)
            sharpe = (excess / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sharpe = 0.0

        # Sortino ratio
        if len(daily_returns) > 10:
            downside = daily_returns[daily_returns < 0]
            ds_std = downside.std() if len(downside) > 0 else 0
            if ds_std > 0:
                excess = daily_returns.mean() - (0.07 / TRADING_DAYS_PER_YEAR)
                sortino = (excess / ds_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                sortino = 0.0
        else:
            sortino = 0.0

        # Max drawdown
        if len(equity_series) > 1:
            rolling_max = equity_series.cummax()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_dd = abs(drawdowns.min()) * 100
        else:
            max_dd = 0.0

        # Calmar ratio
        calmar = (cagr / max_dd) if max_dd > 0 else 0.0

        # Trade stats
        trades = self.trades
        winners = [t for t in trades if t['pnl_abs'] > 0]
        losers = [t for t in trades if t['pnl_abs'] <= 0]
        win_rate = len(winners) / len(trades) * 100 if trades else 0

        avg_win = np.mean([t['pnl_pct'] for t in winners]) if winners else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losers]) if losers else 0

        # Profit Factor
        total_win_pnl = sum(t['pnl_abs'] for t in winners) if winners else 0
        total_loss_pnl = abs(sum(t['pnl_abs'] for t in losers)) if losers else 0
        profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else (
            999.0 if total_win_pnl > 0 else 0.0
        )

        # Exit reason counts
        exit_counts = {}
        for t in trades:
            r = t['exit_reason']
            exit_counts[r] = exit_counts.get(r, 0) + 1

        # Total return
        total_return = (final / initial - 1) * 100

        # Concentration check
        conc = self._concentration_check(trades, equity_series, initial, years)

        return ExplorationResult(
            strategy_name=config.name,
            cagr=round(cagr, 2),
            sharpe=round(sharpe, 2),
            sortino=round(sortino, 2),
            max_drawdown=round(max_dd, 2),
            calmar=round(calmar, 2),
            profit_factor=round(profit_factor, 2),
            total_trades=len(trades),
            win_rate=round(win_rate, 2),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            final_value=round(final, 2),
            total_return_pct=round(total_return, 2),
            top3_pnl_pct=round(conc['top3_pnl_pct'], 2),
            top3_symbols=conc['top3_symbols'],
            cagr_ex_top3=round(conc['cagr_ex_top3'], 2),
            trade_list=trades,
            equity_curve=self.daily_equity,
            exit_reason_counts=exit_counts,
        )

    def _concentration_check(self, trades, equity_series, initial, years):
        """Check if top 3 stocks dominate P/L."""
        if not trades:
            return {'top3_symbols': [], 'top3_pnl_pct': 0, 'cagr_ex_top3': 0}

        # Group by symbol
        sym_pnl = {}
        for t in trades:
            sym_pnl[t['symbol']] = sym_pnl.get(t['symbol'], 0) + t['pnl_abs']

        total_pnl = sum(sym_pnl.values())
        if total_pnl == 0:
            return {'top3_symbols': [], 'top3_pnl_pct': 0, 'cagr_ex_top3': 0}

        # Top 3 by absolute contribution
        sorted_syms = sorted(sym_pnl.items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = sorted_syms[:3]
        top3_syms = [s[0] for s in top3]
        top3_pnl = sum(s[1] for s in top3)
        top3_pct = (top3_pnl / total_pnl * 100) if total_pnl != 0 else 0

        # CAGR excluding top 3: subtract their P/L from final value
        ex_pnl = total_pnl - top3_pnl
        ex_final = initial + ex_pnl
        if years > 0 and initial > 0 and ex_final > 0:
            cagr_ex = ((ex_final / initial) ** (1 / years) - 1) * 100
        else:
            cagr_ex = 0

        return {
            'top3_symbols': top3_syms,
            'top3_pnl_pct': top3_pct,
            'cagr_ex_top3': cagr_ex,
        }

    def _empty_result(self) -> ExplorationResult:
        return ExplorationResult(
            strategy_name=self.config.name,
            cagr=0, sharpe=0, sortino=0, max_drawdown=0, calmar=0,
            profit_factor=0, total_trades=0, win_rate=0,
            avg_win_pct=0, avg_loss_pct=0,
            final_value=self.config.initial_capital,
            total_return_pct=0, top3_pnl_pct=0, top3_symbols=[],
            cagr_ex_top3=0, trade_list=[], equity_curve={},
            exit_reason_counts={},
        )
