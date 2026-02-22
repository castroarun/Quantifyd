"""
Technical Indicators for MQ Strategy Enhancement
=================================================

Provides various technical analysis indicators for entry/exit signals:
- EMA crossovers
- RSI (Relative Strength Index)
- Stochastics
- Ichimoku Cloud
- Supertrend

All functions accept pandas DataFrames with OHLCV columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate EMA for a series."""
    return series.ewm(span=period, adjust=False).mean()


def add_ema_signals(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.DataFrame:
    """
    Add EMA columns and crossover signals to dataframe.

    Returns df with columns:
    - ema_fast, ema_slow
    - ema_bullish: fast > slow
    - ema_cross_up: fast crossed above slow
    - ema_cross_down: fast crossed below slow
    """
    df = df.copy()
    df['ema_fast'] = calc_ema(df['close'], fast)
    df['ema_slow'] = calc_ema(df['close'], slow)
    df['ema_bullish'] = df['ema_fast'] > df['ema_slow']
    df['ema_cross_up'] = df['ema_bullish'] & ~df['ema_bullish'].shift(1).fillna(False)
    df['ema_cross_down'] = ~df['ema_bullish'] & df['ema_bullish'].shift(1).fillna(True)
    return df


def price_above_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Check if close is above EMA of given period."""
    ema = calc_ema(df['close'], period)
    return df['close'] > ema


# =============================================================================
# RSI (Relative Strength Index)
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


def add_rsi_signals(df: pd.DataFrame, period: int = 14,
                    overbought: float = 70, oversold: float = 30) -> pd.DataFrame:
    """
    Add RSI and signals to dataframe.

    Returns df with columns:
    - rsi
    - rsi_overbought: RSI > overbought threshold
    - rsi_oversold: RSI < oversold threshold
    - rsi_bullish: RSI > 50 (momentum confirmation)
    """
    df = df.copy()
    df['rsi'] = calc_rsi(df['close'], period)
    df['rsi_overbought'] = df['rsi'] > overbought
    df['rsi_oversold'] = df['rsi'] < oversold
    df['rsi_bullish'] = df['rsi'] > 50
    return df


# =============================================================================
# Stochastics
# =============================================================================

def calc_stochastics(df: pd.DataFrame, k_period: int = 14,
                     d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic %K and %D.

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA of %K
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()

    stoch_k = ((df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)) * 100
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def add_stoch_signals(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                      overbought: float = 80, oversold: float = 20) -> pd.DataFrame:
    """
    Add Stochastic oscillator and signals.

    Returns df with columns:
    - stoch_k, stoch_d
    - stoch_overbought: %K > overbought
    - stoch_oversold: %K < oversold
    - stoch_cross_up: %K crossed above %D from oversold
    - stoch_cross_down: %K crossed below %D from overbought
    """
    df = df.copy()
    df['stoch_k'], df['stoch_d'] = calc_stochastics(df, k_period, d_period)

    df['stoch_overbought'] = df['stoch_k'] > overbought
    df['stoch_oversold'] = df['stoch_k'] < oversold

    k_above_d = df['stoch_k'] > df['stoch_d']
    df['stoch_cross_up'] = k_above_d & ~k_above_d.shift(1).fillna(False) & (df['stoch_k'] < 50)
    df['stoch_cross_down'] = ~k_above_d & k_above_d.shift(1).fillna(True) & (df['stoch_k'] > 50)

    return df


# =============================================================================
# Ichimoku Cloud
# =============================================================================

def calc_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
                  senkou_b: int = 52) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud components.

    Returns dict with:
    - tenkan_sen: Conversion line (9-period high+low / 2)
    - kijun_sen: Base line (26-period high+low / 2)
    - senkou_a: Leading span A ((tenkan + kijun) / 2, shifted forward 26)
    - senkou_b: Leading span B (52-period high+low / 2, shifted forward 26)
    - chikou: Lagging span (close shifted back 26)
    """
    high_tenkan = df['high'].rolling(window=tenkan).max()
    low_tenkan = df['low'].rolling(window=tenkan).min()
    tenkan_sen = (high_tenkan + low_tenkan) / 2

    high_kijun = df['high'].rolling(window=kijun).max()
    low_kijun = df['low'].rolling(window=kijun).min()
    kijun_sen = (high_kijun + low_kijun) / 2

    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

    high_senkou = df['high'].rolling(window=senkou_b).max()
    low_senkou = df['low'].rolling(window=senkou_b).min()
    senkou_b_line = ((high_senkou + low_senkou) / 2).shift(kijun)

    chikou = df['close'].shift(-kijun)

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b_line,
        'chikou': chikou,
    }


def add_ichimoku_signals(df: pd.DataFrame, tenkan: int = 9,
                         kijun: int = 26, senkou_b: int = 52) -> pd.DataFrame:
    """
    Add Ichimoku cloud and signals.

    Returns df with columns:
    - tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou
    - cloud_top, cloud_bottom
    - price_above_cloud: Close > both senkou spans
    - price_below_cloud: Close < both senkou spans
    - tenkan_above_kijun: Tenkan > Kijun (bullish)
    - ichimoku_bullish: Price above cloud AND tenkan > kijun
    """
    df = df.copy()
    ichi = calc_ichimoku(df, tenkan, kijun, senkou_b)

    for key, val in ichi.items():
        df[key] = val

    df['cloud_top'] = df[['senkou_a', 'senkou_b']].max(axis=1)
    df['cloud_bottom'] = df[['senkou_a', 'senkou_b']].min(axis=1)

    df['price_above_cloud'] = df['close'] > df['cloud_top']
    df['price_below_cloud'] = df['close'] < df['cloud_bottom']
    df['tenkan_above_kijun'] = df['tenkan_sen'] > df['kijun_sen']
    df['ichimoku_bullish'] = df['price_above_cloud'] & df['tenkan_above_kijun']

    return df


# =============================================================================
# Supertrend
# =============================================================================

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def calc_supertrend(df: pd.DataFrame, atr_period: int = 10,
                    multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Supertrend indicator.

    Returns:
    - supertrend: The supertrend line value
    - direction: 1 for uptrend, -1 for downtrend
    """
    df = df.copy()
    atr = calc_atr(df, atr_period)

    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        if df['close'].iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < supertrend.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]

            if direction.iloc[i] == 1 and lower_band.iloc[i] > supertrend.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
            elif direction.iloc[i] == -1 and upper_band.iloc[i] < supertrend.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]

    return supertrend, direction


def add_supertrend_signals(df: pd.DataFrame, atr_period: int = 10,
                           multiplier: float = 3.0) -> pd.DataFrame:
    """
    Add Supertrend and signals.

    Returns df with columns:
    - supertrend: The supertrend line
    - supertrend_direction: 1 (uptrend) or -1 (downtrend)
    - supertrend_bullish: direction == 1
    - supertrend_flip_up: Just flipped to uptrend
    - supertrend_flip_down: Just flipped to downtrend
    """
    df = df.copy()
    df['supertrend'], df['supertrend_direction'] = calc_supertrend(df, atr_period, multiplier)

    df['supertrend_bullish'] = df['supertrend_direction'] == 1
    df['supertrend_flip_up'] = (df['supertrend_direction'] == 1) & (df['supertrend_direction'].shift(1) == -1)
    df['supertrend_flip_down'] = (df['supertrend_direction'] == -1) & (df['supertrend_direction'].shift(1) == 1)

    return df


# =============================================================================
# MACD (Moving Average Convergence Divergence)
# =============================================================================

def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26,
              signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD, Signal line, and Histogram.
    """
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_macd_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                     signal: int = 9) -> pd.DataFrame:
    """
    Add MACD and signals.

    Returns df with columns:
    - macd, macd_signal, macd_histogram
    - macd_bullish: MACD > Signal
    - macd_cross_up: MACD crossed above Signal
    - macd_cross_down: MACD crossed below Signal
    - macd_positive: MACD > 0 (above zero line)
    """
    df = df.copy()
    df['macd'], df['macd_signal'], df['macd_histogram'] = calc_macd(
        df['close'], fast, slow, signal
    )

    df['macd_bullish'] = df['macd'] > df['macd_signal']
    df['macd_cross_up'] = df['macd_bullish'] & ~df['macd_bullish'].shift(1).fillna(False)
    df['macd_cross_down'] = ~df['macd_bullish'] & df['macd_bullish'].shift(1).fillna(True)
    df['macd_positive'] = df['macd'] > 0

    return df


# =============================================================================
# Bollinger Bands
# =============================================================================

def calc_bollinger_bands(df: pd.DataFrame, period: int = 20,
                         std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Returns: (middle_band, upper_band, lower_band)
    """
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return middle, upper, lower


def add_bollinger_signals(df: pd.DataFrame, period: int = 20,
                          std_dev: float = 2.0) -> pd.DataFrame:
    """
    Add Bollinger Bands and signals.

    Returns df with columns:
    - bb_middle, bb_upper, bb_lower
    - bb_width: Band width as % of middle
    - bb_pct_b: %B indicator (where price is within bands)
    - bb_squeeze: Width < 20-day avg width (volatility contraction)
    - bb_above_upper: Price > upper band (overbought)
    - bb_below_lower: Price < lower band (oversold)
    """
    df = df.copy()
    df['bb_middle'], df['bb_upper'], df['bb_lower'] = calc_bollinger_bands(
        df, period, std_dev
    )

    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
    df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    avg_width = df['bb_width'].rolling(window=20).mean()
    df['bb_squeeze'] = df['bb_width'] < avg_width

    df['bb_above_upper'] = df['close'] > df['bb_upper']
    df['bb_below_lower'] = df['close'] < df['bb_lower']

    return df


# =============================================================================
# ADX (Average Directional Index) - Trend Strength
# =============================================================================

def calc_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX, +DI, and -DI.

    Returns: (adx, plus_di, minus_di)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Smoothed values
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def add_adx_signals(df: pd.DataFrame, period: int = 14,
                    trend_threshold: float = 25) -> pd.DataFrame:
    """
    Add ADX and signals.

    Returns df with columns:
    - adx, plus_di, minus_di
    - adx_trending: ADX > threshold (strong trend)
    - adx_bullish: +DI > -DI (uptrend)
    - adx_bearish: -DI > +DI (downtrend)
    - adx_strong_uptrend: ADX > threshold AND +DI > -DI
    """
    df = df.copy()
    df['adx'], df['plus_di'], df['minus_di'] = calc_adx(df, period)

    df['adx_trending'] = df['adx'] > trend_threshold
    df['adx_bullish'] = df['plus_di'] > df['minus_di']
    df['adx_bearish'] = df['minus_di'] > df['plus_di']
    df['adx_strong_uptrend'] = df['adx_trending'] & df['adx_bullish']

    return df


# =============================================================================
# OBV (On-Balance Volume)
# =============================================================================

def calc_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['volume'].iloc[0]

    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return obv


def add_obv_signals(df: pd.DataFrame, ema_period: int = 20) -> pd.DataFrame:
    """
    Add OBV and signals.

    Returns df with columns:
    - obv
    - obv_ema: EMA of OBV
    - obv_bullish: OBV > OBV_EMA (volume confirming)
    - obv_divergence: Price up but OBV down (bearish divergence)
    """
    df = df.copy()
    df['obv'] = calc_obv(df)
    df['obv_ema'] = calc_ema(df['obv'], ema_period)
    df['obv_bullish'] = df['obv'] > df['obv_ema']

    # Simple divergence check (5-day)
    price_up = df['close'] > df['close'].shift(5)
    obv_down = df['obv'] < df['obv'].shift(5)
    df['obv_divergence'] = price_up & obv_down

    return df


# =============================================================================
# VWAP (Volume Weighted Average Price)
# =============================================================================

def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate VWAP.
    Note: For daily data, this resets each day. For proper intraday VWAP,
    you'd need to group by date.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap


def add_vwap_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VWAP and signals.

    Returns df with columns:
    - vwap
    - price_above_vwap: Close > VWAP (bullish)
    """
    df = df.copy()
    df['vwap'] = calc_vwap(df)
    df['price_above_vwap'] = df['close'] > df['vwap']
    return df


# =============================================================================
# Parabolic SAR
# =============================================================================

def calc_parabolic_sar(df: pd.DataFrame, af_start: float = 0.02,
                       af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """
    Calculate Parabolic SAR.
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)

    sar = np.zeros(n)
    ep = np.zeros(n)  # Extreme point
    af = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend

    # Initialize
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = af_start
    trend[0] = 1

    for i in range(1, n):
        if trend[i-1] == 1:  # Uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])

            if low[i] < sar[i]:  # Reversal
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # Downtrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])

            if high[i] > sar[i]:  # Reversal
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]

    return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)


def add_psar_signals(df: pd.DataFrame, af_start: float = 0.02,
                     af_increment: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    """
    Add Parabolic SAR and signals.

    Returns df with columns:
    - psar, psar_trend
    - psar_bullish: Price above SAR (uptrend)
    - psar_flip_up: Just flipped to uptrend
    - psar_flip_down: Just flipped to downtrend
    """
    df = df.copy()
    df['psar'], df['psar_trend'] = calc_parabolic_sar(df, af_start, af_increment, af_max)

    df['psar_bullish'] = df['psar_trend'] == 1
    df['psar_flip_up'] = (df['psar_trend'] == 1) & (df['psar_trend'].shift(1) == -1)
    df['psar_flip_down'] = (df['psar_trend'] == -1) & (df['psar_trend'].shift(1) == 1)

    return df


# =============================================================================
# CCI (Commodity Channel Index)
# =============================================================================

def calc_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci = (typical_price - sma) / (0.015 * mad)
    return cci.fillna(0)


def add_cci_signals(df: pd.DataFrame, period: int = 20,
                    overbought: float = 100, oversold: float = -100) -> pd.DataFrame:
    """
    Add CCI and signals.

    Returns df with columns:
    - cci
    - cci_overbought: CCI > overbought
    - cci_oversold: CCI < oversold
    - cci_bullish: CCI > 0
    """
    df = df.copy()
    df['cci'] = calc_cci(df, period)
    df['cci_overbought'] = df['cci'] > overbought
    df['cci_oversold'] = df['cci'] < oversold
    df['cci_bullish'] = df['cci'] > 0
    return df


# =============================================================================
# Williams %R
# =============================================================================

def calc_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    wr = -100 * (highest_high - df['close']) / (highest_high - lowest_low).replace(0, np.nan)
    return wr.fillna(-50)


def add_williams_r_signals(df: pd.DataFrame, period: int = 14,
                           overbought: float = -20, oversold: float = -80) -> pd.DataFrame:
    """
    Add Williams %R and signals.

    Returns df with columns:
    - williams_r
    - wr_overbought: %R > overbought (near 0)
    - wr_oversold: %R < oversold (near -100)
    """
    df = df.copy()
    df['williams_r'] = calc_williams_r(df, period)
    df['wr_overbought'] = df['williams_r'] > overbought
    df['wr_oversold'] = df['williams_r'] < oversold
    return df


# =============================================================================
# MFI (Money Flow Index) - Volume-weighted RSI
# =============================================================================

def calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = money_flow.iloc[i]
        else:
            negative_flow.iloc[i] = money_flow.iloc[i]

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))
    return mfi.fillna(50)


def add_mfi_signals(df: pd.DataFrame, period: int = 14,
                    overbought: float = 80, oversold: float = 20) -> pd.DataFrame:
    """
    Add MFI and signals.

    Returns df with columns:
    - mfi
    - mfi_overbought: MFI > overbought
    - mfi_oversold: MFI < oversold
    - mfi_bullish: MFI > 50
    """
    df = df.copy()
    df['mfi'] = calc_mfi(df, period)
    df['mfi_overbought'] = df['mfi'] > overbought
    df['mfi_oversold'] = df['mfi'] < oversold
    df['mfi_bullish'] = df['mfi'] > 50
    return df


# =============================================================================
# Keltner Channels
# =============================================================================

def calc_keltner_channels(df: pd.DataFrame, ema_period: int = 20,
                          atr_period: int = 10, multiplier: float = 2.0
                          ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Keltner Channels."""
    middle = calc_ema(df['close'], ema_period)
    atr = calc_atr(df, atr_period)
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)
    return middle, upper, lower


def add_keltner_signals(df: pd.DataFrame, ema_period: int = 20,
                        atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
    """
    Add Keltner Channels and signals.

    Returns df with columns:
    - kc_middle, kc_upper, kc_lower
    - kc_above_upper: Price > upper channel
    - kc_below_lower: Price < lower channel
    - kc_squeeze: BB inside KC (volatility contraction)
    """
    df = df.copy()
    df['kc_middle'], df['kc_upper'], df['kc_lower'] = calc_keltner_channels(
        df, ema_period, atr_period, multiplier
    )
    df['kc_above_upper'] = df['close'] > df['kc_upper']
    df['kc_below_lower'] = df['close'] < df['kc_lower']
    return df


# =============================================================================
# Donchian Channels (for breakout detection)
# =============================================================================

def calc_donchian_channels(df: pd.DataFrame, period: int = 20
                           ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Donchian Channels (highest high / lowest low)."""
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return middle, upper, lower


def add_donchian_signals(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Add Donchian Channels and signals.

    Returns df with columns:
    - dc_middle, dc_upper, dc_lower
    - dc_breakout_up: Close > previous upper (new high breakout)
    - dc_breakout_down: Close < previous lower (new low breakout)
    """
    df = df.copy()
    df['dc_middle'], df['dc_upper'], df['dc_lower'] = calc_donchian_channels(df, period)

    df['dc_breakout_up'] = df['close'] > df['dc_upper'].shift(1)
    df['dc_breakout_down'] = df['close'] < df['dc_lower'].shift(1)

    return df


# =============================================================================
# Combined Signal Generator
# =============================================================================

@dataclass
class TechnicalConfig:
    """Configuration for technical indicator signals."""
    # EMA
    use_ema: bool = False
    ema_fast: int = 9
    ema_slow: int = 21
    ema_entry_type: str = 'crossover'  # 'crossover', 'price_above', 'both'
    ema_exit_type: str = 'crossover'   # 'crossover', 'price_below', 'both'

    # RSI
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    rsi_entry_on_oversold: bool = False  # Entry when RSI recovers from oversold
    rsi_exit_on_overbought: bool = False  # Exit when RSI hits overbought

    # Stochastics
    use_stoch: bool = False
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_overbought: float = 80
    stoch_oversold: float = 20

    # Ichimoku
    use_ichimoku: bool = False
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_entry_above_cloud: bool = True
    ichimoku_exit_below_cloud: bool = True

    # Supertrend
    use_supertrend: bool = False
    supertrend_atr: int = 10
    supertrend_mult: float = 3.0
    supertrend_entry_on_flip: bool = True
    supertrend_exit_on_flip: bool = True

    # Top-down
    use_weekly_filter: bool = False
    weekly_ema_period: int = 20


def generate_signals(df: pd.DataFrame, config: TechnicalConfig) -> Dict[str, bool]:
    """
    Generate entry/exit signals based on technical config.

    Returns dict with:
    - entry_allowed: All entry conditions met
    - exit_signal: Any exit condition triggered
    - signals: Dict of individual indicator states
    """
    signals = {}
    entry_conditions = []
    exit_conditions = []

    # EMA
    if config.use_ema:
        df = add_ema_signals(df, config.ema_fast, config.ema_slow)
        latest = df.iloc[-1]

        if config.ema_entry_type in ('crossover', 'both'):
            entry_conditions.append(latest['ema_bullish'])
        if config.ema_entry_type in ('price_above', 'both'):
            entry_conditions.append(latest['close'] > latest['ema_slow'])

        if config.ema_exit_type in ('crossover', 'both'):
            exit_conditions.append(latest['ema_cross_down'])
        if config.ema_exit_type in ('price_below', 'both'):
            exit_conditions.append(latest['close'] < latest['ema_slow'])

        signals['ema_bullish'] = latest['ema_bullish']

    # RSI
    if config.use_rsi:
        df = add_rsi_signals(df, config.rsi_period, config.rsi_overbought, config.rsi_oversold)
        latest = df.iloc[-1]

        if config.rsi_entry_on_oversold:
            # Entry when RSI was oversold and now recovering
            was_oversold = df['rsi_oversold'].shift(1).iloc[-1]
            entry_conditions.append(was_oversold and not latest['rsi_oversold'])
        else:
            # Entry requires RSI > 50 (momentum confirmation)
            entry_conditions.append(latest['rsi_bullish'])

        if config.rsi_exit_on_overbought:
            exit_conditions.append(latest['rsi_overbought'])

        signals['rsi'] = latest['rsi']
        signals['rsi_bullish'] = latest['rsi_bullish']

    # Stochastics
    if config.use_stoch:
        df = add_stoch_signals(df, config.stoch_k, config.stoch_d,
                               config.stoch_overbought, config.stoch_oversold)
        latest = df.iloc[-1]

        entry_conditions.append(not latest['stoch_overbought'])  # Don't enter overbought
        exit_conditions.append(latest['stoch_cross_down'])

        signals['stoch_k'] = latest['stoch_k']

    # Ichimoku
    if config.use_ichimoku:
        df = add_ichimoku_signals(df, config.ichimoku_tenkan, config.ichimoku_kijun)
        latest = df.iloc[-1]

        if config.ichimoku_entry_above_cloud:
            entry_conditions.append(latest['price_above_cloud'])

        if config.ichimoku_exit_below_cloud:
            exit_conditions.append(latest['price_below_cloud'])

        signals['ichimoku_bullish'] = latest['ichimoku_bullish']

    # Supertrend
    if config.use_supertrend:
        df = add_supertrend_signals(df, config.supertrend_atr, config.supertrend_mult)
        latest = df.iloc[-1]

        if config.supertrend_entry_on_flip:
            entry_conditions.append(latest['supertrend_bullish'])

        if config.supertrend_exit_on_flip:
            exit_conditions.append(latest['supertrend_flip_down'])

        signals['supertrend_bullish'] = latest['supertrend_bullish']

    return {
        'entry_allowed': all(entry_conditions) if entry_conditions else True,
        'exit_signal': any(exit_conditions) if exit_conditions else False,
        'signals': signals,
    }


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices + np.random.rand(100) * 2,
        'low': prices - np.random.rand(100) * 2,
        'close': prices + np.random.randn(100) * 0.5,
        'volume': np.random.randint(1000000, 5000000, 100),
    })
    df = df.set_index('date')

    # Test each indicator
    print("Testing EMA...")
    df_ema = add_ema_signals(df, 9, 21)
    print(f"  EMA bullish: {df_ema['ema_bullish'].iloc[-1]}")

    print("Testing RSI...")
    df_rsi = add_rsi_signals(df, 14)
    print(f"  RSI: {df_rsi['rsi'].iloc[-1]:.1f}")

    print("Testing Stochastics...")
    df_stoch = add_stoch_signals(df, 14, 3)
    print(f"  Stoch %K: {df_stoch['stoch_k'].iloc[-1]:.1f}")

    print("Testing Ichimoku...")
    df_ichi = add_ichimoku_signals(df, 9, 26)
    print(f"  Ichimoku bullish: {df_ichi['ichimoku_bullish'].iloc[-1]}")

    print("Testing Supertrend...")
    df_st = add_supertrend_signals(df, 10, 3.0)
    print(f"  Supertrend direction: {df_st['supertrend_direction'].iloc[-1]}")

    print("\nAll indicators working!")
