"""
Range Detection Engine for Non-Directional Options Strategies
=============================================================

Detects range-bound (consolidation) regimes in NIFTY/BANKNIFTY using multiple
indicator combinations. Measures how well each signal predicts that price will
stay within a defined zone over a forward-looking period.

Key Metrics:
- Zone Hold Rate: % of times price stayed within predicted range
- Avg Zone Duration: How many periods the range held
- Zone Break Magnitude: When it fails, how far price moved beyond the zone
- Signal Frequency: How often the signal fires (usability)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'


# =============================================================================
# Data Loading & Resampling
# =============================================================================

def load_index_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data from database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe=? ORDER BY date",
        conn, params=(symbol, timeframe)
    )
    conn.close()

    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe."""
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled


def get_all_timeframes(symbol: str) -> Dict[str, pd.DataFrame]:
    """Load and create all timeframes from available data."""
    data = {}

    # Daily data (36 months)
    daily = load_index_data(symbol, 'day')
    if not daily.empty:
        data['daily'] = daily

    # Hourly data (24 months)
    hourly = load_index_data(symbol, '60minute')
    if not hourly.empty:
        data['60min'] = hourly
        # Resample to higher intraday timeframes
        data['120min'] = resample_ohlcv(hourly, '2h')
        data['240min'] = resample_ohlcv(hourly, '4h')

    # 5-min data (60 days)
    fivemin = load_index_data(symbol, '5minute')
    if not fivemin.empty:
        data['5min'] = fivemin
        data['15min'] = resample_ohlcv(fivemin, '15min')
        data['30min'] = resample_ohlcv(fivemin, '30min')

    return data


# =============================================================================
# Technical Indicators for Range Detection
# =============================================================================

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands + Bandwidth."""
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma * 100  # as percentage
    return upper, sma, lower, bandwidth


def calc_keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5):
    """Keltner Channels."""
    ema = df['close'].ewm(span=period, adjust=False).mean()
    atr = calc_atr(df, period)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return upper, ema, lower


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    high, low, close = df['high'], df['low'], df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calc_atr(df, period)

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return adx


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_aroon(df: pd.DataFrame, period: int = 25) -> pd.Series:
    """Aroon Oscillator - measures trend strength. Near 0 = range-bound."""
    aroon_up = df['high'].rolling(period + 1).apply(
        lambda x: x.argmax() / period * 100, raw=True)
    aroon_down = df['low'].rolling(period + 1).apply(
        lambda x: x.argmin() / period * 100, raw=True)
    return aroon_up - aroon_down


def calc_linear_regression_r2(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """R-squared of linear regression - low R² = no clear trend."""
    def r2_window(window):
        if len(window) < period:
            return np.nan
        x = np.arange(len(window))
        y = window.values
        if np.std(y) == 0:
            return 0.0
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation ** 2

    return df['close'].rolling(period).apply(r2_window, raw=False)


def calc_historical_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Annualized historical volatility from log returns."""
    log_returns = np.log(df['close'] / df['close'].shift(1))
    return log_returns.rolling(period).std() * np.sqrt(252) * 100


def calc_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Choppiness Index - high values = range-bound, low = trending."""
    atr_sum = calc_atr(df, 1).rolling(period).sum()  # sum of 1-period ATR
    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()
    price_range = high_max - low_min
    price_range = price_range.replace(0, np.nan)
    ci = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    return ci


def calc_cpr(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Central Pivot Range - narrow CPR = price likely to stay in range.
    Returns: pivot, bc (bottom central), tc (top central), cpr_width_pct
    """
    pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    bc = (df['high'].shift(1) + df['low'].shift(1)) / 2
    tc = 2 * pivot - bc
    # Ensure tc > bc
    tc_final = pd.concat([tc, bc], axis=1).max(axis=1)
    bc_final = pd.concat([tc, bc], axis=1).min(axis=1)
    cpr_width = (tc_final - bc_final) / pivot * 100
    return pivot, bc_final, tc_final, cpr_width


def calc_narrow_range(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """NR-N pattern: True when today's range is narrowest in N days."""
    daily_range = df['high'] - df['low']
    min_range = daily_range.rolling(period).min()
    return (daily_range == min_range).astype(int)


def calc_inside_bar(df: pd.DataFrame) -> pd.Series:
    """Inside bar: today's high < yesterday's high AND today's low > yesterday's low."""
    return ((df['high'] < df['high'].shift(1)) &
            (df['low'] > df['low'].shift(1))).astype(int)


def calc_ttm_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                     kc_period: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """
    TTM Squeeze: Bollinger Bands inside Keltner Channels = low volatility squeeze.
    Returns 1 when squeeze is ON (range-bound expected).
    """
    bb_upper, _, bb_lower, _ = calc_bollinger_bands(df, bb_period, bb_std)
    kc_upper, _, kc_lower = calc_keltner_channels(df, kc_period, kc_mult)

    squeeze = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
    return squeeze


def calc_vwap_bands(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    VWAP-like bands using volume-weighted price.
    Returns: vwap, upper_band, lower_band
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, 1)  # avoid div by zero

    cum_tp_vol = (typical_price * vol).rolling(period).sum()
    cum_vol = vol.rolling(period).sum()
    vwap = cum_tp_vol / cum_vol

    # Standard deviation bands
    sq_diff = ((typical_price - vwap) ** 2 * vol).rolling(period).sum()
    vwap_std = np.sqrt(sq_diff / cum_vol)

    return vwap, vwap + 2 * vwap_std, vwap - 2 * vwap_std


def calc_macd_near_zero(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram near zero line = no momentum = range-bound."""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    # Normalize by price
    return (histogram / df['close'] * 100).abs()


# =============================================================================
# Range Detection Signals
# =============================================================================

@dataclass
class RangeSignal:
    """A signal that indicates range-bound conditions."""
    name: str
    category: str  # volatility, trend, price_action, mean_reversion, combined
    description: str
    # These will be populated during backtesting
    signal_series: Optional[pd.Series] = None


def generate_signals(df: pd.DataFrame) -> List[RangeSignal]:
    """Generate all range-detection signals for a given OHLCV dataframe."""
    signals = []

    # --- VOLATILITY SIGNALS ---

    # 1. Bollinger Bandwidth Squeeze
    for period in [10, 20]:
        _, _, _, bw = calc_bollinger_bands(df, period)
        bw_ma = bw.rolling(period).mean()
        # Signal: bandwidth below its own moving average = contraction
        sig = (bw < bw_ma).astype(int)
        signals.append(RangeSignal(
            name=f'BB_SQUEEZE_{period}',
            category='volatility',
            description=f'Bollinger BW < {period}-MA (contraction)',
            signal_series=sig
        ))

    # 2. ATR Contraction
    for period in [7, 14, 20]:
        atr = calc_atr(df, period)
        atr_ma = atr.rolling(period * 2).mean()
        sig = (atr < atr_ma).astype(int)
        signals.append(RangeSignal(
            name=f'ATR_CONTRACT_{period}',
            category='volatility',
            description=f'ATR({period}) < {period*2}-MA ATR',
            signal_series=sig
        ))

    # 3. TTM Squeeze
    sig = calc_ttm_squeeze(df)
    signals.append(RangeSignal(
        name='TTM_SQUEEZE',
        category='volatility',
        description='Bollinger inside Keltner (TTM Squeeze ON)',
        signal_series=sig
    ))

    # 4. Historical Volatility Low
    for period in [10, 20, 30]:
        hv = calc_historical_volatility(df, period)
        hv_ma = hv.rolling(period * 2).mean()
        sig = (hv < hv_ma).astype(int)
        signals.append(RangeSignal(
            name=f'HV_LOW_{period}',
            category='volatility',
            description=f'HV({period}) below its {period*2}-MA',
            signal_series=sig
        ))

    # 5. Choppiness Index High
    for period in [7, 14]:
        ci = calc_choppiness_index(df, period)
        # CI > 61.8 = choppy/range-bound (Fibonacci level)
        sig = (ci > 61.8).astype(int)
        signals.append(RangeSignal(
            name=f'CHOP_HIGH_{period}',
            category='volatility',
            description=f'Choppiness({period}) > 61.8',
            signal_series=sig
        ))

    # --- TREND STRENGTH SIGNALS ---

    # 6. ADX Low (no trend)
    for period in [7, 14, 20]:
        adx = calc_adx(df, period)
        for threshold in [20, 25]:
            sig = (adx < threshold).astype(int)
            signals.append(RangeSignal(
                name=f'ADX_LOW_{period}_{threshold}',
                category='trend',
                description=f'ADX({period}) < {threshold}',
                signal_series=sig
            ))

    # 7. Aroon Oscillator near zero
    for period in [14, 25]:
        aroon = calc_aroon(df, period)
        sig = (aroon.abs() < 50).astype(int)
        signals.append(RangeSignal(
            name=f'AROON_FLAT_{period}',
            category='trend',
            description=f'|Aroon({period})| < 50',
            signal_series=sig
        ))

    # 8. Linear Regression R² low
    for period in [10, 20]:
        r2 = calc_linear_regression_r2(df, period)
        sig = (r2 < 0.3).astype(int)
        signals.append(RangeSignal(
            name=f'R2_LOW_{period}',
            category='trend',
            description=f'LinReg R²({period}) < 0.3',
            signal_series=sig
        ))

    # --- PRICE ACTION SIGNALS ---

    # 9. Narrow Range (NR4, NR7)
    for period in [4, 7]:
        sig = calc_narrow_range(df, period)
        signals.append(RangeSignal(
            name=f'NR{period}',
            category='price_action',
            description=f'Narrowest range in {period} bars',
            signal_series=sig
        ))

    # 10. Inside Bar
    sig = calc_inside_bar(df)
    signals.append(RangeSignal(
        name='INSIDE_BAR',
        category='price_action',
        description='Inside bar pattern',
        signal_series=sig
    ))

    # 11. CPR Narrow
    _, _, _, cpr_width = calc_cpr(df)
    cpr_ma = cpr_width.rolling(20).mean()
    sig = (cpr_width < cpr_ma).astype(int)
    signals.append(RangeSignal(
        name='CPR_NARROW',
        category='price_action',
        description='CPR width below 20-MA',
        signal_series=sig
    ))

    # --- MEAN REVERSION SIGNALS ---

    # 12. RSI in mid-zone
    for period in [7, 14]:
        rsi = calc_rsi(df, period)
        sig = ((rsi > 40) & (rsi < 60)).astype(int)
        signals.append(RangeSignal(
            name=f'RSI_MID_{period}',
            category='mean_reversion',
            description=f'RSI({period}) between 40-60',
            signal_series=sig
        ))

    # 13. MACD near zero
    macd_dist = calc_macd_near_zero(df)
    macd_threshold = macd_dist.rolling(50).quantile(0.3)
    sig = (macd_dist < macd_threshold).astype(int)
    signals.append(RangeSignal(
        name='MACD_FLAT',
        category='mean_reversion',
        description='MACD histogram near zero (bottom 30th pctl)',
        signal_series=sig
    ))

    # 14. Price near moving average (hugging)
    for ma_period in [20, 50]:
        ma = df['close'].rolling(ma_period).mean()
        distance = ((df['close'] - ma) / ma * 100).abs()
        dist_threshold = distance.rolling(ma_period * 2).quantile(0.3)
        sig = (distance < dist_threshold).astype(int)
        signals.append(RangeSignal(
            name=f'PRICE_HUG_MA{ma_period}',
            category='mean_reversion',
            description=f'Price within bottom 30th pctl distance from MA{ma_period}',
            signal_series=sig
        ))

    # --- COMBINED SIGNALS ---

    # 15. Squeeze + Low ADX
    ttm = calc_ttm_squeeze(df)
    adx14 = calc_adx(df, 14)
    sig = ((ttm == 1) & (adx14 < 25)).astype(int)
    signals.append(RangeSignal(
        name='SQUEEZE_ADX',
        category='combined',
        description='TTM Squeeze ON + ADX(14) < 25',
        signal_series=sig
    ))

    # 16. BB Squeeze + Choppiness High
    _, _, _, bw20 = calc_bollinger_bands(df, 20)
    bw20_ma = bw20.rolling(20).mean()
    ci14 = calc_choppiness_index(df, 14)
    sig = ((bw20 < bw20_ma) & (ci14 > 61.8)).astype(int)
    signals.append(RangeSignal(
        name='BB_CHOP',
        category='combined',
        description='BB Squeeze + Choppiness > 61.8',
        signal_series=sig
    ))

    # 17. Multi-signal consensus (3 of 5 core signals agree)
    core_signals = pd.DataFrame({
        'ttm': calc_ttm_squeeze(df),
        'adx_low': (calc_adx(df, 14) < 25).astype(int),
        'chop_high': (calc_choppiness_index(df, 14) > 61.8).astype(int),
        'rsi_mid': ((calc_rsi(df, 14) > 40) & (calc_rsi(df, 14) < 60)).astype(int),
        'bw_low': (bw20 < bw20_ma).astype(int),
    })
    consensus_count = core_signals.sum(axis=1)
    for min_agree in [3, 4]:
        sig = (consensus_count >= min_agree).astype(int)
        signals.append(RangeSignal(
            name=f'CONSENSUS_{min_agree}of5',
            category='combined',
            description=f'{min_agree}/5 core range signals agree',
            signal_series=sig
        ))

    return signals


# =============================================================================
# Backtesting Engine
# =============================================================================

@dataclass
class SignalResult:
    """Result of backtesting a single signal on a single timeframe."""
    symbol: str
    timeframe: str
    signal_name: str
    category: str

    # Forward-looking holding periods tested
    hold_periods: List[int] = field(default_factory=list)

    # Per holding period metrics
    zone_hold_rates: Dict[int, float] = field(default_factory=dict)
    avg_zone_durations: Dict[int, float] = field(default_factory=dict)
    avg_break_magnitudes: Dict[int, float] = field(default_factory=dict)
    max_break_magnitudes: Dict[int, float] = field(default_factory=dict)

    # Overall
    signal_frequency: float = 0.0  # % of time signal is active
    total_signals: int = 0
    total_bars: int = 0


def backtest_signal(
    df: pd.DataFrame,
    signal: RangeSignal,
    symbol: str,
    timeframe: str,
    hold_periods: List[int] = None,
    zone_width_atr_mult: float = 1.5,
) -> SignalResult:
    """
    Backtest a range-detection signal.

    When signal fires, we define a zone:
      zone_center = current close
      zone_width = zone_width_atr_mult * ATR(14)
      zone_upper = zone_center + zone_width
      zone_lower = zone_center - zone_width

    Then check if price stays within that zone for each holding period.
    """
    if hold_periods is None:
        hold_periods = [1, 3, 5, 7, 10, 15, 20]

    result = SignalResult(
        symbol=symbol,
        timeframe=timeframe,
        signal_name=signal.name,
        category=signal.category,
        hold_periods=hold_periods,
    )

    sig = signal.signal_series
    if sig is None or sig.empty:
        return result

    # Calculate ATR for zone width
    atr = calc_atr(df, 14)

    # Align everything
    aligned = pd.DataFrame({
        'close': df['close'],
        'high': df['high'],
        'low': df['low'],
        'signal': sig,
        'atr': atr,
    }).dropna()

    if len(aligned) < 50:
        return result

    result.total_bars = len(aligned)
    signal_mask = aligned['signal'] == 1
    result.total_signals = signal_mask.sum()
    result.signal_frequency = result.total_signals / result.total_bars * 100

    if result.total_signals == 0:
        return result

    # Get signal entry points
    signal_indices = aligned.index[signal_mask]

    for hp in hold_periods:
        holds_in_zone = 0
        total_tested = 0
        break_magnitudes = []
        zone_durations = []

        for entry_idx in signal_indices:
            entry_pos = aligned.index.get_loc(entry_idx)

            # Need enough forward data
            if entry_pos + hp >= len(aligned):
                continue

            entry_close = aligned.loc[entry_idx, 'close']
            entry_atr = aligned.loc[entry_idx, 'atr']

            if pd.isna(entry_atr) or entry_atr == 0:
                continue

            zone_width = zone_width_atr_mult * entry_atr
            zone_upper = entry_close + zone_width
            zone_lower = entry_close - zone_width

            # Check forward prices
            forward_slice = aligned.iloc[entry_pos + 1: entry_pos + hp + 1]
            forward_highs = forward_slice['high']
            forward_lows = forward_slice['low']

            # Did price stay in zone for the entire holding period?
            max_high = forward_highs.max()
            min_low = forward_lows.min()

            stayed_in_zone = (max_high <= zone_upper) and (min_low >= zone_lower)
            total_tested += 1

            if stayed_in_zone:
                holds_in_zone += 1
                zone_durations.append(hp)
            else:
                # How far did it break?
                upside_break = max(0, (max_high - zone_upper) / entry_close * 100)
                downside_break = max(0, (zone_lower - min_low) / entry_close * 100)
                break_mag = max(upside_break, downside_break)
                break_magnitudes.append(break_mag)

                # How many bars before it broke?
                for j in range(len(forward_slice)):
                    bar = forward_slice.iloc[j]
                    if bar['high'] > zone_upper or bar['low'] < zone_lower:
                        zone_durations.append(j)
                        break
                else:
                    zone_durations.append(hp)

        if total_tested > 0:
            result.zone_hold_rates[hp] = holds_in_zone / total_tested * 100
            result.avg_zone_durations[hp] = np.mean(zone_durations) if zone_durations else 0
            result.avg_break_magnitudes[hp] = np.mean(break_magnitudes) if break_magnitudes else 0
            result.max_break_magnitudes[hp] = max(break_magnitudes) if break_magnitudes else 0

    return result


def run_full_backtest(
    symbol: str,
    hold_periods: List[int] = None,
    zone_width_atr_mult: float = 1.5,
) -> List[SignalResult]:
    """Run all signals across all timeframes for a symbol."""

    if hold_periods is None:
        hold_periods = [1, 3, 5, 7, 10, 15, 20]

    all_timeframes = get_all_timeframes(symbol)
    results = []

    for tf_name, df in all_timeframes.items():
        if len(df) < 60:
            print(f"  Skipping {tf_name}: only {len(df)} bars")
            continue

        print(f"  Processing {tf_name} ({len(df)} bars)...")
        signals = generate_signals(df)

        for signal in signals:
            result = backtest_signal(
                df, signal, symbol, tf_name,
                hold_periods=hold_periods,
                zone_width_atr_mult=zone_width_atr_mult,
            )
            results.append(result)

    return results


def results_to_dataframe(results: List[SignalResult]) -> pd.DataFrame:
    """Convert results to a flat DataFrame for analysis."""
    rows = []
    for r in results:
        for hp in r.hold_periods:
            if hp not in r.zone_hold_rates:
                continue
            rows.append({
                'symbol': r.symbol,
                'timeframe': r.timeframe,
                'signal': r.signal_name,
                'category': r.category,
                'hold_period': hp,
                'zone_hold_rate': r.zone_hold_rates.get(hp, 0),
                'avg_zone_duration': r.avg_zone_durations.get(hp, 0),
                'avg_break_magnitude': r.avg_break_magnitudes.get(hp, 0),
                'max_break_magnitude': r.max_break_magnitudes.get(hp, 0),
                'signal_frequency': r.signal_frequency,
                'total_signals': r.total_signals,
                'total_bars': r.total_bars,
            })
    return pd.DataFrame(rows)
