"""research/176 — single-stock always-on directional trend engine.

Fresh code. Deliberately reuses NOTHING from services/maruthi_*.py (that algo was
disabled 2026-03-25 with 9 live bugs).

Conventions locked in the STATUS doc:
  * signal decided at the CLOSE of bar t, using only data through bar t
  * honest fill  = next bar's OPEN            (fill='next_open')
  * optimistic   = same bar's CLOSE           (fill='signal_close')  [reference arm]
  * P&L is open-to-open for next_open, close-to-close for signal_close
  * cost is a ROUND TRIP in bps; each one-way leg costs half of it
  * flat cash earns 5.2% p.a. post-tax (house standard)
"""
import numpy as np
import pandas as pd
import sqlite3

DB = '/home/arun/quantifyd/backtest_data/market_data.db'
IDLE_YIELD = 0.052           # post-tax arbitrage-fund standard
TRADING_DAYS = 252


# ----------------------------------------------------------------------------- data
def connect():
    return sqlite3.connect('file:' + DB + '?mode=ro', uri=True)


def load_bars(con, symbol, timeframe='day', start=None, end=None):
    q = ("select date, open, high, low, close, volume from market_data_unified "
         "where symbol=? and timeframe=? order by date")
    df = pd.read_sql_query(q, con, params=(symbol, timeframe))
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')
    df = df.drop_duplicates('date').set_index('date').sort_index()
    # phantom-row purge: O==H==L==C and volume 0  (Kite holiday placeholders)
    ph = (df['open'] == df['high']) & (df['high'] == df['low']) & \
         (df['low'] == df['close']) & (df['volume'] == 0)
    df = df[~ph]
    df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    return df


def resample_intraday(df5, rule):
    """rule e.g. '15min','30min','60min' — from 5-minute bars."""
    o = df5.resample(rule, label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    return o.dropna(subset=['open', 'close'])


def split_defect_flags(df):
    """Overnight jumps big enough to be an unadjusted corporate action.

    market_data.db is NOT retroactively split-adjusted (memory 2026-09-01), so a
    pre-split row keeps the old price scale and shows up as a one-bar step.
    Returns (n_flags, worst_ratio, list_of_dates).
    """
    c = df['close'].values
    o = df['open'].values
    if len(c) < 3:
        return 0, 1.0, []
    r = o[1:] / c[:-1]
    bad = (r < 0.65) | (r > 1.55)
    idx = np.where(bad)[0] + 1
    worst = float(max(r.max(), 1.0 / r.min())) if len(r) else 1.0
    return int(bad.sum()), worst, [str(df.index[i].date()) for i in idx]


# ------------------------------------------------------------------------ indicators
def atr_wilder(high, low, close, period):
    prev_c = np.empty_like(close)
    prev_c[0] = close[0]
    prev_c[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr = np.empty_like(tr)
    atr[:] = np.nan
    if len(tr) < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    a = 1.0 / period
    v = atr[period - 1]
    for i in range(period, len(tr)):
        v = v + a * (tr[i] - v)
        atr[i] = v
    return atr


def supertrend_dir(high, low, close, period, mult):
    """Band-locked SuperTrend (TradingView convention). Returns +1/-1 direction
    array; NaN warm-up region encoded as 0."""
    n = len(close)
    d = np.zeros(n, dtype=np.int8)
    atr = atr_wilder(high, low, close, period)
    hl2 = (high + low) / 2.0
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    start = period
    if n <= start:
        return d
    fub = ub[start]
    flb = lb[start]
    dirn = 1
    d[start] = 1
    for i in range(start + 1, n):
        cub, clb = ub[i], lb[i]
        # band lock: upper only moves down, lower only moves up, while unbroken
        if cub < fub or close[i - 1] > fub:
            fub = cub
        if clb > flb or close[i - 1] < flb:
            flb = clb
        if dirn == 1:
            if close[i] < flb:
                dirn = -1
        else:
            if close[i] > fub:
                dirn = 1
        d[i] = dirn
    return d


def ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def ema_cross_dir(close, fast, slow):
    n = len(close)
    d = np.zeros(n, dtype=np.int8)
    if n <= slow:
        return d
    ef = ema(close, fast)
    es = ema(close, slow)
    d[slow:] = np.where(ef[slow:] > es[slow:], 1, -1).astype(np.int8)
    return d


def mst_dir(master, child):
    """Arun's MST machine reduced to a posture series.

    Master ST sets the regime; the child ST times entry inside it. Posture is
    LONG only while master is bull AND child has confirmed bull at least once
    since the master flipped; symmetric on the short side. Between a master flip
    and the first child confirmation the book is FLAT (that is the 'wait for the
    signal candle' behaviour of the live design, without the lot-stacking, which
    is tested separately in stage 4).
    """
    n = len(master)
    out = np.zeros(n, dtype=np.int8)
    armed = 0
    prev_m = 0
    for i in range(n):
        m = master[i]
        if m != prev_m:
            armed = 0
            prev_m = m
        if m == 0:
            out[i] = 0
            continue
        if armed == 0 and child[i] == m:
            armed = 1
        out[i] = m if armed else 0
    return out


# --------------------------------------------------------------------------- P&L
def run_book(df, direction, fill, cost_bps_rt, idle=IDLE_YIELD, lag_bars=0):
    """direction: int8 array of the DESIRED posture decided at each bar close.

    fill='next_open'     -> posture applies from the next bar's open
    fill='signal_close'  -> posture applies from the signal bar's close (optimistic)
    lag_bars: extra bars of execution delay on top of the fill convention.

    Returns dict of metrics + the per-bar net return array.
    """
    o = df['open'].values.astype(float)
    c = df['close'].values.astype(float)
    n = len(c)
    shift = 1 + lag_bars
    if fill == 'next_open':
        px = o
    else:
        px = c
        shift = 0 + lag_bars
    # posture held over interval i -> i+1  (px[i] .. px[i+1])
    held = np.zeros(n, dtype=np.int8)
    if shift < n:
        held[shift:] = direction[:n - shift] if shift else direction
    ret = np.zeros(n)
    ret[:-1] = px[1:] / px[:-1] - 1.0
    ret[-1] = 0.0
    # bar-length in years, for the idle-cash leg
    days = (df.index[-1] - df.index[0]).days
    years = max(days / 365.25, 1e-9)
    bar_yr = years / max(n - 1, 1)

    gross = held * ret
    idle_leg = (held == 0) * ((1 + idle) ** bar_yr - 1)
    chg = np.abs(np.diff(np.concatenate([[0], held]))).astype(float)
    one_way = cost_bps_rt / 2.0 / 10000.0
    cost = chg * one_way
    net = gross + idle_leg - cost

    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    maxdd = float(dd.min())
    total = float(eq[-1])
    cagr = total ** (1.0 / years) - 1.0 if total > 0 else -1.0

    # trades = posture changes into a non-zero posture
    switches = int((chg > 0).sum())
    # per-trade stats on non-zero spells
    trades = _spell_stats(held, ret, one_way)

    return dict(
        cagr=cagr, maxdd=maxdd,
        calmar=(cagr / abs(maxdd)) if maxdd < 0 else np.nan,
        total_mult=total, years=years,
        switches_per_yr=switches / years,
        n_trades=trades['n'], win_rate=trades['wr'],
        avg_win=trades['aw'], avg_loss=trades['al'], expectancy=trades['exp'],
        time_in_mkt=float((held != 0).mean()),
        long_share=float((held > 0).mean()), short_share=float((held < 0).mean()),
        ret=net, held=held, bar_ret=ret, bar_yr=bar_yr,
    )


def _spell_stats(held, ret, one_way):
    """Per-trade stats. A trade is one contiguous spell of a non-zero posture."""
    n = len(held)
    if n == 0:
        return dict(n=0, wr=np.nan, aw=np.nan, al=np.nan, exp=np.nan)
    contrib = held * ret
    ch = np.flatnonzero(np.diff(held.astype(np.int16))) + 1
    starts = np.concatenate([[0], ch])
    sums = np.add.reduceat(contrib, starts)
    vals = held[starts]
    a = sums[vals != 0] - 2.0 * one_way
    if a.size == 0:
        return dict(n=0, wr=np.nan, aw=np.nan, al=np.nan, exp=np.nan)
    w = a[a > 0]
    l = a[a <= 0]
    return dict(n=int(a.size), wr=float(w.size / a.size),
                aw=float(w.mean()) if w.size else 0.0,
                al=float(l.mean()) if l.size else 0.0,
                exp=float(a.mean()))


def buy_and_hold(df, fill='next_open', cost_bps_rt=20.0):
    px = df['open'].values.astype(float) if fill == 'next_open' else df['close'].values.astype(float)
    days = (df.index[-1] - df.index[0]).days
    years = max(days / 365.25, 1e-9)
    total = px[-1] / px[0] * (1 - cost_bps_rt / 10000.0)
    ret = np.zeros(len(px))
    ret[:-1] = px[1:] / px[:-1] - 1.0
    eq = np.cumprod(1 + ret)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    maxdd = float(dd.min())
    cagr = total ** (1.0 / years) - 1.0 if total > 0 else -1.0
    return dict(cagr=cagr, maxdd=maxdd,
                calmar=(cagr / abs(maxdd)) if maxdd < 0 else np.nan, years=years)


# ------------------------------------------------------------------------ the grid
ST_GRID = [(p, m) for p in (7, 10, 14, 21) for m in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)]
EMA_GRID = [(5, 20), (9, 21), (10, 30), (20, 50), (21, 55), (50, 100), (50, 200)]
MST_GRID = [(7, m, 7, c) for m in (4.0, 5.0, 6.0) for c in (1.5, 2.0, 2.5)]


def all_directions(df):
    """Compute every signal cell's posture array once per symbol/timeframe."""
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    st_cache = {}
    for (p, m) in ST_GRID:
        st_cache[(p, m)] = supertrend_dir(h, l, c, p, m)
    for m in (4.0, 5.0, 6.0):
        if (7, m) not in st_cache:
            st_cache[(7, m)] = supertrend_dir(h, l, c, 7, m)
    out = {}
    for (p, m) in ST_GRID:
        out[f'ST_{p}_{m}'] = st_cache[(p, m)]
    for (f, s) in EMA_GRID:
        out[f'EMA_{f}_{s}'] = ema_cross_dir(c, f, s)
    for (mp, mm, cp, cm) in MST_GRID:
        out[f'MST_{mp}_{mm}_{cp}_{cm}'] = mst_dir(st_cache[(mp, mm)], st_cache[(cp, cm)])
    return out


def apply_policy(direction, policy):
    if policy == 'long_flat':
        return np.where(direction > 0, 1, 0).astype(np.int8)
    if policy == 'long_short':
        return direction.astype(np.int8)
    if policy == 'short_flat':
        return np.where(direction < 0, -1, 0).astype(np.int8)
    raise ValueError(policy)
