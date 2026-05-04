"""EMA + Stochastics indicators for the 20/50-EMA + Stoch(14,5,3) strategy.

Stochastics(14, 5, 3) convention used here:
- Raw %K        = 100 * (close - LL14) / (HH14 - LL14)
- Smoothed %K   = 5-period SMA of raw %K       (the line plotted as %K)
- %D            = 3-period SMA of smoothed %K  (the slower line)

Both lines bounded [0, 100]. Pure functions, no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def stochastics(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    k_smooth: int = 5,
    d_smooth: int = 3,
) -> pd.DataFrame:
    """Return DataFrame with columns ['k', 'd'] aligned to input index."""
    ll = low.rolling(window=k_period, min_periods=k_period).min()
    hh = high.rolling(window=k_period, min_periods=k_period).max()
    rng = (hh - ll).replace(0.0, np.nan)
    raw_k = 100.0 * (close - ll) / rng
    k = raw_k.rolling(window=k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(window=d_smooth, min_periods=d_smooth).mean()
    return pd.DataFrame({"k": k, "d": d}, index=close.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder ATR."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder ADX. Returns the ADX series (no +DI/-DI)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.fillna(0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.fillna(0)

    alpha = 1.0 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_w.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_w.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 7,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Standard Supertrend.

    Returns DataFrame with columns ['st', 'dir'] where:
      - st = the supertrend line (price level)
      - dir = +1 for uptrend (st = lower band), -1 for downtrend (st = upper band)
    """
    a = atr(high, low, close, atr_period)
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a

    n = len(close)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.full(n, 1, dtype=int)

    c = close.values
    u = upper.values
    l = lower.values

    bootstrapped = False
    for i in range(n):
        if np.isnan(u[i]) or np.isnan(l[i]):
            # ATR not ready yet
            final_upper[i] = np.nan
            final_lower[i] = np.nan
            st[i] = np.nan
            direction[i] = 1
            continue

        if not bootstrapped:
            # First valid bar: seed from raw bands, choose initial direction
            # by comparing close to the midpoint of the bands.
            final_upper[i] = u[i]
            final_lower[i] = l[i]
            mid = (u[i] + l[i]) / 2.0
            direction[i] = 1 if c[i] >= mid else -1
            st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
            bootstrapped = True
            continue

        # Adjust bands to be sticky
        final_upper[i] = u[i] if (u[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
        final_lower[i] = l[i] if (l[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]) else final_lower[i - 1]

        # Determine direction
        if st[i - 1] == final_upper[i - 1]:
            direction[i] = -1 if c[i] <= final_upper[i] else 1
        else:
            direction[i] = 1 if c[i] >= final_lower[i] else -1

        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.DataFrame({"st": st, "dir": direction}, index=close.index)


if __name__ == "__main__":
    # Sanity smoke
    from data30 import load_30min, INTRADAY_STOCKS

    sym = INTRADAY_STOCKS[0]
    df = load_30min(sym)
    if df.empty:
        print(f"{sym}: no data")
    else:
        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        st = stochastics(df["high"], df["low"], df["close"])
        df["k"] = st["k"]
        df["d"] = st["d"]
        df["rsi14"] = rsi(df["close"], 14)
        df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
        print(df.tail(5)[["close", "ema20", "ema50", "k", "d", "rsi14", "atr14"]])
