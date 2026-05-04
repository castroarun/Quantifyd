"""SuperTrend + helper indicators. Vectorised numpy implementation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    return np.maximum.reduce([tr1, tr2, tr3])


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    tr = true_range(high, low, close)
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (direction, supertrend_line, atr).

    direction: +1 (long/bullish) or -1 (short/bearish). NaN where ATR not yet seeded.
    supertrend_line: the active band (lower when long, upper when short).
    """
    n = len(close)
    atr = atr_wilder(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_final = np.full(n, np.nan)
    lower_final = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int8)
    st_line = np.full(n, np.nan)

    seed = period  # first index where ATR is valid
    if seed >= n:
        return direction.astype(float), st_line, atr

    upper_final[seed] = upper_basic[seed]
    lower_final[seed] = lower_basic[seed]
    direction[seed] = 1 if close[seed] > upper_basic[seed] else -1
    st_line[seed] = lower_final[seed] if direction[seed] == 1 else upper_final[seed]

    for i in range(seed + 1, n):
        # Final upper band (non-decreasing in downtrend)
        if upper_basic[i] < upper_final[i - 1] or close[i - 1] > upper_final[i - 1]:
            upper_final[i] = upper_basic[i]
        else:
            upper_final[i] = upper_final[i - 1]
        # Final lower band (non-increasing in uptrend)
        if lower_basic[i] > lower_final[i - 1] or close[i - 1] < lower_final[i - 1]:
            lower_final[i] = lower_basic[i]
        else:
            lower_final[i] = lower_final[i - 1]

        # Direction flip rules
        prev_dir = direction[i - 1]
        if prev_dir == 1:
            direction[i] = -1 if close[i] < lower_final[i] else 1
        else:
            direction[i] = 1 if close[i] > upper_final[i] else -1

        st_line[i] = lower_final[i] if direction[i] == 1 else upper_final[i]

    direction_f = direction.astype(float)
    direction_f[:seed] = np.nan
    return direction_f, st_line, atr


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[period] = gain[1 : period + 1].mean()
    avg_loss[period] = loss[1 : period + 1].mean()
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    out = 100 - 100 / (1 + rs)
    return out


def stochastic(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    k_period: int = 14, d_period: int = 3, smooth: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    k_raw = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        hh = high[i - k_period + 1 : i + 1].max()
        ll = low[i - k_period + 1 : i + 1].min()
        if hh - ll > 0:
            k_raw[i] = 100 * (close[i] - ll) / (hh - ll)
    # smooth %K
    k = pd.Series(k_raw).rolling(smooth, min_periods=smooth).mean().values
    d = pd.Series(k).rolling(d_period, min_periods=d_period).mean().values
    return k, d


def bollinger(close: np.ndarray, period: int = 20, mult: float = 2.0):
    s = pd.Series(close)
    mid = s.rolling(period).mean().values
    sd = s.rolling(period).std(ddof=0).values
    upper = mid + mult * sd
    lower = mid - mult * sd
    return upper, mid, lower


def resample_5min_to(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Resample 5-min OHLCV to 15min/30min/60min. df indexed by datetime."""
    rule = {"15min": "15min", "30min": "30min", "60min": "60min"}[target]
    out = df.resample(rule, label="left", closed="left", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    # Keep only NSE session bars: 09:15 to 15:30 IST
    mask = (out.index.time >= pd.Timestamp("09:15").time()) & (
        out.index.time <= pd.Timestamp("15:30").time()
    )
    return out[mask]
