"""
Sideways / Low-Volatility Zone Analysis for NIFTY50 and BANKNIFTY
Uses 5-minute data from market_data.db to identify regime zones
and compute comprehensive stats for neutral options strategy design.
"""

import logging
logging.disable(logging.WARNING)

import sqlite3
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
SYMBOLS = ['NIFTY50', 'BANKNIFTY']

# Expiry weekday: Thursday=3 for NIFTY, Wednesday=2 for BANKNIFTY
EXPIRY_WEEKDAY = {'NIFTY50': 3, 'BANKNIFTY': 2}


def load_data(symbol):
    """Load 5-min data for a symbol from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",
        conn, params=(symbol,)
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['trading_date'] = df['date'].dt.date
    return df


# ---------------------------------------------------------------------------
# Indicator computations
# ---------------------------------------------------------------------------

def compute_atr_squeeze(df):
    """ATR(14) via Wilder's RMA < SMA(ATR, 50). Handles overnight gap."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    trading_date = df['trading_date'].values

    n = len(df)
    tr = np.empty(n, dtype=np.float64)

    # True Range: first bar of each day uses H-L only (no prev close gap)
    for i in range(n):
        if i == 0 or trading_date[i] != trading_date[i - 1]:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))

    # Wilder's RMA (exponential with alpha = 1/period)
    period = 14
    atr = np.full(n, np.nan)
    # Seed with SMA of first `period` values
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # SMA of ATR over 50 bars
    atr_series = pd.Series(atr)
    atr_sma50 = atr_series.rolling(50, min_periods=50).mean().values

    squeeze = np.where(np.isnan(atr) | np.isnan(atr_sma50), False, atr < atr_sma50)
    return squeeze.astype(bool)


def compute_bb_contraction(df):
    """BB(20,2) bandwidth < SMA(bandwidth, 50)."""
    close = df['close']
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bandwidth = ((upper - lower) / sma20) * 100  # as percentage

    bw_sma50 = bandwidth.rolling(50, min_periods=50).mean()
    contraction = (bandwidth < bw_sma50) & bandwidth.notna() & bw_sma50.notna()
    return contraction.values.astype(bool)


def compute_adx(df, period=14):
    """ADX using Wilder's smoothing. Returns boolean array where ADX < 20."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    trading_date = df['trading_date'].values
    n = len(df)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        if trading_date[i] != trading_date[i - 1]:
            # First candle of day: no directional movement from prev day
            tr[i] = high[i] - low[i]
            plus_dm[i] = 0.0
            minus_dm[i] = 0.0
        else:
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i] = up if (up > down and up > 0) else 0.0
            minus_dm[i] = down if (down > up and down > 0) else 0.0
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))

    # Wilder smoothing
    def wilder_smooth(arr, p):
        out = np.full(n, np.nan)
        if n >= p + 1:
            out[p] = np.sum(arr[1:p + 1])
            for i in range(p + 1, n):
                out[i] = out[i - 1] - out[i - 1] / p + arr[i]
        return out

    atr_s = wilder_smooth(tr, period)
    plus_dm_s = wilder_smooth(plus_dm, period)
    minus_dm_s = wilder_smooth(minus_dm, period)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_s / atr_s
        minus_di = 100 * minus_dm_s / atr_s
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

    dx = np.where(np.isnan(dx) | np.isinf(dx), 0.0, dx)

    # ADX = Wilder smooth of DX
    adx = np.full(n, np.nan)
    start = period + period  # need 2*period bars
    if n > start:
        adx[start] = np.nanmean(dx[period + 1:start + 1])
        for i in range(start + 1, n):
            if not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    weak_trend = np.where(np.isnan(adx), False, adx < 20)
    return weak_trend.astype(bool)


def compute_fdi(df, window=30):
    """Fractal Dimension Index on rolling window.
    On 5-min data, FDI values typically range 1.1-1.45, so we use adaptive
    threshold: FDI > median(FDI) signals sideways/choppy regime.
    Classic 1.5 threshold never fires on intraday data.
    """
    close = df['close'].values
    n = len(df)
    fdi_values = np.full(n, np.nan)

    for i in range(window, n):
        segment = close[i - window + 1:i + 1]
        max_p = np.max(segment)
        min_p = np.min(segment)
        price_range = max_p - min_p
        if price_range <= 0:
            fdi_values[i] = 2.0  # flat = max choppiness
            continue

        # Normalized path length
        length = 0.0
        for j in range(1, window):
            length += abs((segment[j] - segment[j - 1]) / price_range)

        # FDI = 1 + ln(length) / ln(2*(window-1))
        if length > 0:
            fdi_values[i] = 1.0 + np.log(length) / np.log(2 * (window - 1))
        else:
            fdi_values[i] = 1.0

    # Adaptive threshold: use rolling median as dynamic threshold
    # Above median = choppier than usual = sideways signal
    fdi_series = pd.Series(fdi_values)
    fdi_median = fdi_series.rolling(200, min_periods=50).median()
    fdi_signal = np.where(np.isnan(fdi_values) | np.isnan(fdi_median),
                          False, fdi_values > fdi_median)

    return fdi_signal.astype(bool)


def compute_combined_score(df):
    """Compute all 4 indicators and combined score."""
    print("    ATR squeeze...", end='', flush=True)
    atr_sq = compute_atr_squeeze(df)
    print(" done. BB contraction...", end='', flush=True)
    bb_ct = compute_bb_contraction(df)
    print(" done. ADX...", end='', flush=True)
    adx_low = compute_adx(df)
    print(" done. FDI...", end='', flush=True)
    fdi_sw = compute_fdi(df)
    print(" done.", flush=True)

    score = atr_sq.astype(int) + bb_ct.astype(int) + adx_low.astype(int) + fdi_sw.astype(int)

    df['atr_squeeze'] = atr_sq
    df['bb_contraction'] = bb_ct
    df['adx_low'] = adx_low
    df['fdi_sideways'] = fdi_sw
    df['score'] = score

    df['regime'] = 'Mixed'
    df.loc[score >= 3, 'regime'] = 'Sideways'
    df.loc[score <= 1, 'regime'] = 'Trending'

    return df


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def compute_daily_stats(df, symbol):
    """Aggregate 5-min bars into daily stats."""
    daily = df.groupby('trading_date').agg(
        day_open=('open', 'first'),
        day_high=('high', 'max'),
        day_low=('low', 'min'),
        day_close=('close', 'last'),
        total_candles=('close', 'count'),
        sideways_candles=('regime', lambda x: (x == 'Sideways').sum()),
        trending_candles=('regime', lambda x: (x == 'Trending').sum()),
    ).reset_index()

    daily['intraday_range_pts'] = daily['day_high'] - daily['day_low']
    daily['intraday_range_pct'] = (daily['intraday_range_pts'] / daily['day_open']) * 100

    # Close to close move
    daily['prev_close'] = daily['day_close'].shift(1)
    daily['c2c_move_pts'] = (daily['day_close'] - daily['prev_close']).abs()
    daily['c2c_move_pct'] = (daily['c2c_move_pts'] / daily['prev_close']).abs() * 100

    # Max move from open
    daily['max_from_open_pts'] = np.maximum(
        (daily['day_high'] - daily['day_open']).abs(),
        (daily['day_low'] - daily['day_open']).abs()
    )
    daily['max_from_open_pct'] = (daily['max_from_open_pts'] / daily['day_open']) * 100

    # Sideways %
    daily['sideways_pct'] = daily['sideways_candles'] / daily['total_candles'] * 100
    daily['day_regime'] = 'Mixed'
    daily.loc[daily['sideways_pct'] > 60, 'day_regime'] = 'Sideways Day'
    daily.loc[daily['sideways_pct'] < 30, 'day_regime'] = 'Trending Day'

    # DTE
    expiry_wd = EXPIRY_WEEKDAY[symbol]
    dte_list = []
    for d in daily['trading_date']:
        dt = pd.Timestamp(d)
        wd = dt.weekday()
        days_until = (expiry_wd - wd) % 7
        if days_until == 0:
            dte = 0
        else:
            # Count business days until expiry (simple weekday count)
            dte = 0
            check = dt
            while True:
                check += timedelta(days=1)
                if check.weekday() < 5:
                    dte += 1
                if check.weekday() == expiry_wd:
                    break
                # Safety: don't loop forever
                if dte > 10:
                    break
        dte_list.append(dte)
    daily['dte'] = dte_list

    return daily


# ---------------------------------------------------------------------------
# Sideways zone duration and breakout analysis
# ---------------------------------------------------------------------------

def compute_zone_analysis(df):
    """Find contiguous sideways zones and compute durations + post-breakout moves."""
    is_sideways = (df['regime'] == 'Sideways').values
    trading_dates = df['trading_date'].values
    timestamps = df['date'].values
    close = df['close'].values
    n = len(df)

    zones = []  # list of (start_idx, end_idx, duration_minutes)
    in_zone = False
    start = 0

    for i in range(n):
        if is_sideways[i] and not in_zone:
            in_zone = True
            start = i
        elif not is_sideways[i] and in_zone:
            # Zone ended at i-1, but only count intraday zones
            end = i - 1
            if trading_dates[start] == trading_dates[end]:
                dur = (pd.Timestamp(timestamps[end]) - pd.Timestamp(timestamps[start])).total_seconds() / 60 + 5
                zones.append((start, end, dur))
            in_zone = False
        elif is_sideways[i] and in_zone:
            # Check if new day started - break the zone
            if trading_dates[i] != trading_dates[i - 1]:
                end = i - 1
                if trading_dates[start] == trading_dates[end]:
                    dur = (pd.Timestamp(timestamps[end]) - pd.Timestamp(timestamps[start])).total_seconds() / 60 + 5
                    zones.append((start, end, dur))
                in_zone = True
                start = i

    # Close last zone if open
    if in_zone:
        end = n - 1
        if trading_dates[start] == trading_dates[end]:
            dur = (pd.Timestamp(timestamps[end]) - pd.Timestamp(timestamps[start])).total_seconds() / 60 + 5
            zones.append((start, end, dur))

    # Post-breakout analysis: for zones > 30 min, measure move after zone ends
    breakout_30 = []
    breakout_60 = []
    breakout_120 = []

    for start_idx, end_idx, dur in zones:
        if dur < 30:
            continue
        exit_idx = end_idx + 1
        if exit_idx >= n:
            continue
        exit_date = trading_dates[exit_idx]
        exit_close = close[end_idx]

        for offset, result_list in [(6, breakout_30), (12, breakout_60), (24, breakout_120)]:
            target = exit_idx + offset
            if target < n and trading_dates[target] == exit_date:
                move_pct = abs(close[target] - exit_close) / exit_close * 100
                result_list.append(move_pct)

    return zones, breakout_30, breakout_60, breakout_120


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_header(title):
    print()
    print('=' * 80)
    print(f'  {title}')
    print('=' * 80)


def print_subheader(title):
    print()
    print(f'--- {title} ---')
    print()


def fmt(val, decimals=2):
    if pd.isna(val):
        return 'N/A'
    return f'{val:.{decimals}f}'


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    results = {}

    for symbol in SYMBOLS:
        print(f'\nLoading {symbol} 5-min data...', flush=True)
        df = load_data(symbol)
        print(f'  Loaded {len(df)} candles, {df["trading_date"].nunique()} trading days '
              f'({df["date"].min().date()} to {df["date"].max().date()})')

        print(f'  Computing indicators for {symbol}...')
        df = compute_combined_score(df)

        print(f'  Computing daily stats...')
        daily = compute_daily_stats(df, symbol)

        print(f'  Computing zone analysis...')
        zones, br30, br60, br120 = compute_zone_analysis(df)

        results[symbol] = {
            'df': df,
            'daily': daily,
            'zones': zones,
            'breakout_30': br30,
            'breakout_60': br60,
            'breakout_120': br120,
        }

    # ===================================================================
    # SECTION A: Overall Regime Distribution
    # ===================================================================
    print_header('SECTION A: Overall Regime Distribution')

    for symbol in SYMBOLS:
        df = results[symbol]['df']
        daily = results[symbol]['daily']
        print_subheader(f'{symbol} - 5-min Candle Regime')

        total = len(df)
        for regime in ['Sideways', 'Mixed', 'Trending']:
            cnt = (df['regime'] == regime).sum()
            print(f'  {regime:12s}: {cnt:6d} candles ({cnt/total*100:5.1f}%)')

        print()
        print(f'  {"Indicator":25s} {"Active":>8s} {"Pct":>7s}')
        print(f'  {"-"*25} {"-"*8} {"-"*7}')
        for col, label in [('atr_squeeze', 'ATR Squeeze'),
                           ('bb_contraction', 'BB Contraction'),
                           ('adx_low', 'ADX < 20'),
                           ('fdi_sideways', 'FDI > 1.5')]:
            cnt = df[col].sum()
            print(f'  {label:25s} {cnt:8d} {cnt/total*100:6.1f}%')

        print()
        print(f'  Day-Level Regime:')
        total_days = len(daily)
        for regime in ['Sideways Day', 'Mixed', 'Trending Day']:
            cnt = (daily['day_regime'] == regime).sum()
            print(f'    {regime:15s}: {cnt:4d} days ({cnt/total_days*100:5.1f}%)')

    # ===================================================================
    # SECTION B: Sideways vs Trending Day Comparison
    # ===================================================================
    print_header('SECTION B: Sideways vs Trending Day Comparison')

    for symbol in SYMBOLS:
        daily = results[symbol]['daily']
        print_subheader(symbol)

        sw = daily[daily['day_regime'] == 'Sideways Day']
        tr = daily[daily['day_regime'] == 'Trending Day']

        metrics = [
            ('Intraday Range (pts)', 'intraday_range_pts'),
            ('Intraday Range (%)', 'intraday_range_pct'),
            ('Abs C2C Move (pts)', 'c2c_move_pts'),
            ('Abs C2C Move (%)', 'c2c_move_pct'),
            ('Max Move from Open (pts)', 'max_from_open_pts'),
            ('Max Move from Open (%)', 'max_from_open_pct'),
        ]

        header = f'  {"Metric":28s} | {"Sideways Avg":>13s} {"Med":>9s} | {"Trending Avg":>13s} {"Med":>9s} | {"Ratio":>6s}'
        print(header)
        print(f'  {"-"*28}-+-{"-"*13}-{"-"*9}-+-{"-"*13}-{"-"*9}-+-{"-"*6}')

        for label, col in metrics:
            sw_avg = sw[col].mean()
            sw_med = sw[col].median()
            tr_avg = tr[col].mean()
            tr_med = tr[col].median()
            ratio = tr_avg / sw_avg if sw_avg > 0 else float('nan')
            print(f'  {label:28s} | {fmt(sw_avg):>13s} {fmt(sw_med):>9s} | {fmt(tr_avg):>13s} {fmt(tr_med):>9s} | {fmt(ratio, 1):>6s}x')

        # 90th percentile
        print()
        print(f'  90th Percentile C2C Move:')
        sw_p90 = sw['c2c_move_pct'].quantile(0.9) if len(sw) > 0 else float('nan')
        tr_p90 = tr['c2c_move_pct'].quantile(0.9) if len(tr) > 0 else float('nan')
        print(f'    Sideways Days: {fmt(sw_p90)}%')
        print(f'    Trending Days: {fmt(tr_p90)}%')

    # ===================================================================
    # SECTION C: DTE Analysis
    # ===================================================================
    print_header('SECTION C: DTE Analysis (Days to Expiry)')

    for symbol in SYMBOLS:
        daily = results[symbol]['daily']
        print_subheader(f'{symbol} (Expiry: {"Thu" if EXPIRY_WEEKDAY[symbol]==3 else "Wed"})')

        dte_range = range(0, 5)
        header = f'  {"DTE":>4s} | {"Days":>5s} | {"SW Days":>7s} {"TR Days":>7s} {"SW%":>5s} | {"Avg Range%":>10s} {"Avg C2C%":>9s} | {"Safety":>7s}'
        print(header)
        print(f'  {"-"*4}-+-{"-"*5}-+-{"-"*7}-{"-"*7}-{"-"*5}-+-{"-"*10}-{"-"*9}-+-{"-"*7}')

        for dte in dte_range:
            subset = daily[daily['dte'] == dte]
            if len(subset) == 0:
                continue
            sw_d = (subset['day_regime'] == 'Sideways Day').sum()
            tr_d = (subset['day_regime'] == 'Trending Day').sum()
            sw_pct = sw_d / len(subset) * 100

            sw_subset = subset[subset['day_regime'] == 'Sideways Day']
            avg_range = sw_subset['intraday_range_pct'].mean() if len(sw_subset) > 0 else float('nan')
            avg_c2c = sw_subset['c2c_move_pct'].mean() if len(sw_subset) > 0 else float('nan')

            # Safety score: % of all days at this DTE where C2C < 0.5%
            safety = (subset['c2c_move_pct'] < 0.5).sum() / len(subset) * 100 if len(subset) > 0 else 0

            print(f'  {dte:4d} | {len(subset):5d} | {sw_d:7d} {tr_d:7d} {sw_pct:4.0f}% | {fmt(avg_range):>10s} {fmt(avg_c2c):>9s} | {safety:5.1f}%')

    # ===================================================================
    # SECTION D: Time-of-Day Analysis
    # ===================================================================
    print_header('SECTION D: Time-of-Day Regime Analysis')

    for symbol in SYMBOLS:
        df = results[symbol]['df']
        print_subheader(symbol)

        df['hour'] = df['date'].dt.hour
        df['bar_range_pct'] = (df['high'] - df['low']) / df['open'] * 100

        header = f'  {"Hour":>5s} | {"Candles":>8s} | {"SW%":>6s} {"TR%":>6s} {"MX%":>6s} | {"Avg Bar Range%":>15s}'
        print(header)
        print(f'  {"-"*5}-+-{"-"*8}-+-{"-"*6}-{"-"*6}-{"-"*6}-+-{"-"*15}')

        for hour in range(9, 16):
            h_df = df[df['hour'] == hour]
            if len(h_df) == 0:
                continue
            total = len(h_df)
            sw_pct = (h_df['regime'] == 'Sideways').sum() / total * 100
            tr_pct = (h_df['regime'] == 'Trending').sum() / total * 100
            mx_pct = (h_df['regime'] == 'Mixed').sum() / total * 100
            avg_range = h_df['bar_range_pct'].mean()
            print(f'  {hour:5d} | {total:8d} | {sw_pct:5.1f}% {tr_pct:5.1f}% {mx_pct:5.1f}% | {fmt(avg_range, 4):>15s}')

    # ===================================================================
    # SECTION E: Sideways Zone Duration Distribution
    # ===================================================================
    print_header('SECTION E: Sideways Zone Duration Distribution')

    for symbol in SYMBOLS:
        zones = results[symbol]['zones']
        print_subheader(symbol)

        durs = [z[2] for z in zones]
        if not durs:
            print('  No sideways zones found.')
            continue

        print(f'  Total sideways zones: {len(durs)}')
        print(f'  Average duration: {np.mean(durs):.1f} min')
        print(f'  Median duration:  {np.median(durs):.1f} min')
        print(f'  Max duration:     {np.max(durs):.0f} min')
        print()

        buckets = [
            ('<15 min', 0, 15),
            ('15-30 min', 15, 30),
            ('30-60 min', 30, 60),
            ('1-2 hr', 60, 120),
            ('2-3 hr', 120, 180),
            ('>3 hr', 180, 9999),
        ]
        print(f'  {"Duration":12s} | {"Count":>6s} | {"Pct":>6s}')
        print(f'  {"-"*12}-+-{"-"*6}-+-{"-"*6}')
        for label, lo, hi in buckets:
            cnt = sum(1 for d in durs if lo <= d < hi)
            print(f'  {label:12s} | {cnt:6d} | {cnt/len(durs)*100:5.1f}%')

    # ===================================================================
    # SECTION F: Post-Sideways Breakout Analysis
    # ===================================================================
    print_header('SECTION F: What Happens After Sideways Zones End (>30 min zones)')

    for symbol in SYMBOLS:
        br30 = results[symbol]['breakout_30']
        br60 = results[symbol]['breakout_60']
        br120 = results[symbol]['breakout_120']
        print_subheader(symbol)

        for label, data in [('Next 30 min', br30), ('Next 60 min', br60), ('Next 120 min', br120)]:
            if not data:
                print(f'  {label}: No data')
                continue
            arr = np.array(data)
            print(f'  {label} ({len(arr)} samples):')
            print(f'    Avg move: {arr.mean():.3f}%   Median: {np.median(arr):.3f}%   '
                  f'90th pct: {np.percentile(arr, 90):.3f}%   Max: {arr.max():.3f}%')
            # Proportion of moves < 0.3%, < 0.5%, < 1.0%
            pct_03 = (arr < 0.3).sum() / len(arr) * 100
            pct_05 = (arr < 0.5).sum() / len(arr) * 100
            pct_10 = (arr < 1.0).sum() / len(arr) * 100
            print(f'    Move < 0.3%: {pct_03:.1f}%   < 0.5%: {pct_05:.1f}%   < 1.0%: {pct_10:.1f}%')
            print()

    # ===================================================================
    # SECTION G: Optimal Entry Windows (Hour x DTE x Regime)
    # ===================================================================
    print_header('SECTION G: Optimal Entry Windows (Hour x DTE)')

    for symbol in SYMBOLS:
        df = results[symbol]['df']
        daily = results[symbol]['daily']
        print_subheader(symbol)

        # Merge DTE into 5-min df
        dte_map = dict(zip(daily['trading_date'], daily['dte']))
        df['dte'] = df['trading_date'].map(dte_map)
        df['hour'] = df['date'].dt.hour

        print(f'  Sideways % by Hour x DTE:')
        print()
        header = f'  {"Hour":>5s} |' + ''.join(f' {"DTE="+str(d):>8s}' for d in range(5)) + ' |  Total'
        print(header)
        print(f'  {"-"*5}-+' + '-' * (9 * 5) + '-+-------')

        for hour in range(9, 16):
            row = f'  {hour:5d} |'
            for dte in range(5):
                subset = df[(df['hour'] == hour) & (df['dte'] == dte)]
                if len(subset) == 0:
                    row += f' {"--":>8s}'
                else:
                    sw_pct = (subset['regime'] == 'Sideways').sum() / len(subset) * 100
                    row += f' {sw_pct:7.1f}%'
            # Total for this hour
            h_all = df[df['hour'] == hour]
            if len(h_all) > 0:
                sw_all = (h_all['regime'] == 'Sideways').sum() / len(h_all) * 100
                row += f' | {sw_all:5.1f}%'
            print(row)

        # Find top-5 sweet spots
        print()
        print(f'  Top 10 Sweet Spots (highest sideways %):')
        combos = []
        for hour in range(9, 16):
            for dte in range(5):
                subset = df[(df['hour'] == hour) & (df['dte'] == dte)]
                if len(subset) >= 20:  # minimum sample size
                    sw_pct = (subset['regime'] == 'Sideways').sum() / len(subset) * 100
                    avg_range = subset['bar_range_pct'].mean()
                    combos.append((hour, dte, sw_pct, avg_range, len(subset)))

        combos.sort(key=lambda x: x[2], reverse=True)
        print(f'  {"Hour":>5s} {"DTE":>4s} {"SW%":>7s} {"AvgRange%":>10s} {"Samples":>8s}')
        for hour, dte, sw_pct, avg_range, n_samples in combos[:10]:
            print(f'  {hour:5d} {dte:4d} {sw_pct:6.1f}% {avg_range:9.4f}% {n_samples:8d}')

    print()
    print('=' * 80)
    print('  Analysis complete.')
    print('=' * 80)


if __name__ == '__main__':
    main()
