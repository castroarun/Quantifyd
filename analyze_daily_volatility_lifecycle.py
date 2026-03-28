"""
Daily Volatility Lifecycle Analysis
====================================
Studies the repeating pattern of morning volatility spikes followed by afternoon calm
in NIFTY50 and BANKNIFTY 5-minute data, to help time neutral options strategy entries.

Indicators:
1. Bollinger Bandwidth cool-off timing
2. Keltner Channel band containment
3. ATR squeeze activation

Output: Timing statistics, safety metrics for straddle sellers, DTE cross-tabs,
best entry windows, and concrete strategy recommendations.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import warnings
warnings.filterwarnings('ignore')

DB_PATH = 'backtest_data/market_data.db'
SYMBOLS = ['NIFTY50', 'BANKNIFTY']

# Expiry day changes:
# NIFTY: Thursday(3) until Nov 20, 2024; Tuesday(1) from Nov 26, 2024
# BANKNIFTY: Wednesday(2) until Nov 13, 2024; weekly discontinued after that (monthly only, last Thu)
from datetime import date as date_type
NIFTY_EXPIRY_CUTOVER = date_type(2024, 11, 25)  # Mon — first week with Tue expiry
BNF_WEEKLY_END = date_type(2024, 11, 13)  # Last BNF weekly expiry

# Time slots for distribution/entry analysis
TIME_SLOTS = [time(10, 0), time(10, 30), time(11, 0), time(11, 30), time(12, 0)]
TIME_SLOT_LABELS = ['10:00', '10:30', '11:00', '11:30', '12:00']


def load_data(symbol):
    """Load 5-minute data for a symbol from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",
        conn, params=(symbol,)
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df['trading_date'] = df.index.date
    return df


def compute_indicators(df):
    """Compute BB, KC, ATR indicators on continuous 5-min series."""
    close = df['close']
    high = df['high']
    low = df['low']

    # --- Bollinger Bands (20, 2) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df['bb_bandwidth'] = (bb_upper - bb_lower) / bb_mid * 100

    # --- ATR(14) with gap handling ---
    prev_close = close.shift(1)
    # Detect first bar of day
    df['is_first_bar'] = df['trading_date'] != pd.Series(df['trading_date']).shift(1).values
    # True Range: use high-low only for first bar (no gap inflation)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Override first bar TR
    tr[df['is_first_bar']] = (high - low)[df['is_first_bar']]
    df['tr'] = tr
    df['atr14'] = tr.ewm(span=14, adjust=False).mean()

    # --- Keltner Channel (20, 1.5 ATR) ---
    kc_mid = close.ewm(span=20, adjust=False).mean()
    df['kc_upper'] = kc_mid + 1.5 * df['atr14']
    df['kc_lower'] = kc_mid - 1.5 * df['atr14']
    df['kc_mid'] = kc_mid

    # --- ATR Squeeze: ATR(14) < SMA(ATR, 50) ---
    df['atr_sma50'] = df['atr14'].rolling(50).mean()
    df['atr_squeeze'] = df['atr14'] < df['atr_sma50']

    # --- Candle inside KC bands ---
    df['inside_kc'] = (df['high'] < df['kc_upper']) & (df['low'] > df['kc_lower'])

    # --- STARC Bands: SMA(5) +/- 2 * ATR(15) ---
    starc_mid = close.rolling(5).mean()
    atr15 = tr.ewm(span=15, adjust=False).mean()
    df['starc_upper'] = starc_mid + 2 * atr15
    df['starc_lower'] = starc_mid - 2 * atr15
    df['inside_starc'] = (df['high'] < df['starc_upper']) & (df['low'] > df['starc_lower'])

    # --- Fractal Bands (Williams Fractals, n=2) ---
    # Fractal high: bar where high is highest of 5 bars (2 on each side)
    # Fractal low: bar where low is lowest of 5 bars (2 on each side)
    n_frac = 2
    frac_upper = pd.Series(np.nan, index=df.index)
    frac_lower = pd.Series(np.nan, index=df.index)
    h = high.values
    l = low.values
    for i in range(n_frac, len(df) - n_frac):
        if h[i] == max(h[i-n_frac:i+n_frac+1]):
            frac_upper.iloc[i] = h[i]
        if l[i] == min(l[i-n_frac:i+n_frac+1]):
            frac_lower.iloc[i] = l[i]
    # Forward-fill fractals to create bands
    df['fractal_upper'] = frac_upper.ffill()
    df['fractal_lower'] = frac_lower.ffill()
    df['inside_fractal'] = (df['high'] < df['fractal_upper']) & (df['low'] > df['fractal_lower'])

    return df


def build_expiry_dates(symbol, all_trading_dates):
    """Build a sorted list of expiry dates for a symbol.
    Uses actual trading dates to handle holidays (if expiry falls on holiday,
    it moves to the previous trading day)."""
    import calendar
    trading_set = set(all_trading_dates)
    min_date = min(all_trading_dates)
    max_date = max(all_trading_dates)
    expiries = []

    if symbol == 'NIFTY50':
        # Pre-Nov 2024: Thursday weekly. Post-Nov 2024: Tuesday weekly.
        d = min_date
        while d <= max_date + timedelta(days=10):
            if d < NIFTY_EXPIRY_CUTOVER:
                target_wd = 3  # Thursday
            else:
                target_wd = 1  # Tuesday
            if d.weekday() == target_wd:
                # Find actual expiry: this day or previous trading day if holiday
                exp = d
                while exp not in trading_set and exp >= min_date:
                    exp -= timedelta(days=1)
                if exp >= min_date:
                    expiries.append(exp)
            d += timedelta(days=1)
    elif symbol == 'BANKNIFTY':
        # Pre-Nov 2024: Wednesday weekly. Post-Nov 2024: monthly last Thursday.
        d = min_date
        while d <= max_date + timedelta(days=10):
            if d <= BNF_WEEKLY_END:
                if d.weekday() == 2:  # Wednesday
                    exp = d
                    while exp not in trading_set and exp >= min_date:
                        exp -= timedelta(days=1)
                    if exp >= min_date:
                        expiries.append(exp)
            d += timedelta(days=1)
        # Post-Nov 2024: last Thursday of each month
        y, m = BNF_WEEKLY_END.year, BNF_WEEKLY_END.month
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
        while date_type(y, m, 1) <= max_date + timedelta(days=31):
            last_day = calendar.monthrange(y, m)[1]
            exp = date_type(y, m, last_day)
            while exp.weekday() != 3:  # Thursday
                exp -= timedelta(days=1)
            # Adjust for holidays
            while exp not in trading_set and exp >= min_date:
                exp -= timedelta(days=1)
            if exp >= min_date:
                expiries.append(exp)
            if m == 12:
                y, m = y + 1, 1
            else:
                m += 1

    return sorted(set(expiries))


def compute_dte(trading_date, trading_dates_sorted, expiry_dates_sorted):
    """Compute DTE as actual market trading days to next expiry.
    0 = expiry day itself."""
    import bisect
    idx = bisect.bisect_left(expiry_dates_sorted, trading_date)
    if idx >= len(expiry_dates_sorted):
        return None
    next_expiry = expiry_dates_sorted[idx]

    # Count trading days between trading_date and next_expiry (inclusive of both = 0-based)
    start_idx = bisect.bisect_left(trading_dates_sorted, trading_date)
    end_idx = bisect.bisect_left(trading_dates_sorted, next_expiry)
    dte = end_idx - start_idx  # 0 if same day
    return dte


def analyze_daily(df, symbol):
    """Analyze each trading day and return a DataFrame of daily metrics."""
    days = df.groupby('trading_date')
    all_trading_dates = sorted(days.groups.keys())
    expiry_dates = build_expiry_dates(symbol, all_trading_dates)
    trading_dates_sorted = all_trading_dates
    expiry_dates_sorted = expiry_dates
    records = []

    for day_date, day_df in days:
        n = len(day_df)
        if n < 30:  # skip partial days
            continue

        # Skip days where BB bandwidth is all NaN (first day of data)
        bw = day_df['bb_bandwidth']
        if bw.isna().all():
            continue

        times = day_df.index.time
        closes = day_df['close'].values
        highs = day_df['high'].values
        lows = day_df['low'].values
        day_open = closes[0]
        day_close = closes[-1]
        day_high = highs.max()
        day_low = lows.min()

        # --- Indicator 1: BB Bandwidth Peak & Cool-off ---
        bw_vals = bw.values
        bw_valid = ~np.isnan(bw_vals)
        if not bw_valid.any():
            continue

        peak_idx = np.nanargmax(bw_vals)
        peak_time = times[peak_idx]
        peak_bw = bw_vals[peak_idx]
        cooloff_threshold = peak_bw * 0.50

        cooloff_idx = None
        cooloff_time = None
        for i in range(peak_idx + 1, n):
            if not np.isnan(bw_vals[i]) and bw_vals[i] <= cooloff_threshold:
                cooloff_idx = i
                cooloff_time = times[i]
                break

        # --- Indicator 2: KC Band Containment ---
        inside = day_df['inside_kc'].values
        containment_idx = None
        containment_time = None
        for i in range(n):
            if inside[i]:
                containment_idx = i
                containment_time = times[i]
                break

        # % of candles after containment that stay inside
        containment_pct = None
        if containment_idx is not None and containment_idx < n - 1:
            remaining = inside[containment_idx + 1:]
            if len(remaining) > 0:
                containment_pct = np.mean(remaining) * 100

        # --- Indicator 3: ATR Squeeze activation ---
        squeeze = day_df['atr_squeeze'].values
        squeeze_idx = None
        squeeze_time = None
        for i in range(n):
            if squeeze[i]:
                squeeze_idx = i
                squeeze_time = times[i]
                break

        # --- Indicator 4: STARC Band Containment ---
        inside_starc = day_df['inside_starc'].values
        starc_idx = None
        starc_time = None
        for i in range(n):
            if inside_starc[i]:
                starc_idx = i
                starc_time = times[i]
                break

        # --- Indicator 5: Fractal Band Containment ---
        inside_fractal = day_df['inside_fractal'].values
        fractal_idx = None
        fractal_time = None
        for i in range(n):
            if not np.isnan(day_df['fractal_upper'].values[i]) and inside_fractal[i]:
                fractal_idx = i
                fractal_time = times[i]
                break

        # --- DTE ---
        dte = compute_dte(day_date, trading_dates_sorted, expiry_dates_sorted)
        if dte is None:
            dte = 0

        # --- Helper: compute max adverse from a given bar index to EOD ---
        def _max_adverse_from(idx):
            if idx is None or idx >= n - 1:
                return None, None
            ref = closes[idx]
            up = highs[idx:].max() - ref
            dn = ref - lows[idx:].min()
            pts = max(up, dn)
            return pts, pts / ref * 100

        # --- Max adverse for each filter ---
        atr_adv_pts, atr_adv_pct = _max_adverse_from(squeeze_idx)
        kc_adv_pts, kc_adv_pct = _max_adverse_from(containment_idx)
        starc_adv_pts, starc_adv_pct = _max_adverse_from(starc_idx)
        fractal_adv_pts, fractal_adv_pct = _max_adverse_from(fractal_idx)

        # --- Combined: ATR Squeeze + BB Cooloff (whichever fires LATER) ---
        combined_idx = None
        combined_time = None
        if squeeze_idx is not None and cooloff_idx is not None:
            combined_idx = max(squeeze_idx, cooloff_idx)
            combined_time = times[combined_idx]
        elif cooloff_idx is not None:
            combined_idx = cooloff_idx
            combined_time = cooloff_time
        elif squeeze_idx is not None:
            combined_idx = squeeze_idx
            combined_time = squeeze_time
        combined_adv_pts, combined_adv_pct = _max_adverse_from(combined_idx)

        # --- Pre/Post cooloff metrics ---
        # Use cooloff_idx as the divider; if no cooloff, skip post-cooloff metrics
        pre_range_pts = None
        pre_range_pct = None
        post_range_pts = None
        post_range_pct = None
        max_adverse_pts = None
        max_adverse_pct = None
        cooloff_to_close_pts = None
        cooloff_to_close_pct = None

        if cooloff_idx is not None:
            # Pre-cooloff
            pre_high = highs[:cooloff_idx + 1].max()
            pre_low = lows[:cooloff_idx + 1].min()
            pre_range_pts = pre_high - pre_low
            pre_range_pct = pre_range_pts / day_open * 100

            # Post-cooloff
            if cooloff_idx < n - 1:
                post_high = highs[cooloff_idx:].max()
                post_low = lows[cooloff_idx:].min()
                post_range_pts = post_high - post_low
                post_range_pct = post_range_pts / closes[cooloff_idx] * 100

                # Max adverse move from cooloff close
                cooloff_close = closes[cooloff_idx]
                post_closes = closes[cooloff_idx:]
                post_highs = highs[cooloff_idx:]
                post_lows = lows[cooloff_idx:]
                max_up = post_highs.max() - cooloff_close
                max_down = cooloff_close - post_lows.min()
                max_adverse_pts = max(max_up, max_down)
                max_adverse_pct = max_adverse_pts / cooloff_close * 100

                cooloff_to_close_pts = day_close - cooloff_close
                cooloff_to_close_pct = cooloff_to_close_pts / cooloff_close * 100

        # --- Entry window analysis: max adverse from each time slot to EOD ---
        entry_adverse = {}
        for slot in TIME_SLOTS:
            slot_label = slot.strftime('%H:%M')
            # Find the bar at or just after this time
            mask = times >= slot
            if not mask.any():
                entry_adverse[slot_label + '_pts'] = None
                entry_adverse[slot_label + '_pct'] = None
                continue
            entry_idx = np.argmax(mask)
            entry_close = closes[entry_idx]
            if entry_idx < n - 1:
                rem_highs = highs[entry_idx:]
                rem_lows = lows[entry_idx:]
                up = rem_highs.max() - entry_close
                down = entry_close - rem_lows.min()
                adv = max(up, down)
                entry_adverse[slot_label + '_pts'] = adv
                entry_adverse[slot_label + '_pct'] = adv / entry_close * 100
            else:
                entry_adverse[slot_label + '_pts'] = 0
                entry_adverse[slot_label + '_pct'] = 0

        # --- Consecutive inside-band bars after containment ---
        consec_inside = 0
        if containment_idx is not None:
            for i in range(containment_idx, n):
                if inside[i]:
                    consec_inside += 1
                else:
                    break

        # --- No-filter max adverse: from open (9:15) to EOD ---
        # This is what you'd face if you entered at market open without waiting for BB cooloff
        nofilter_max_up = day_high - day_open
        nofilter_max_down = day_open - day_low
        nofilter_adverse_pts = max(nofilter_max_up, nofilter_max_down)
        nofilter_adverse_pct = nofilter_adverse_pts / day_open * 100

        record = {
            'date': day_date,
            'weekday': day_date.weekday(),
            'weekday_name': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][day_date.weekday()] if day_date.weekday() < 5 else 'Wknd',
            'dte': dte,
            'n_bars': n,
            'day_range_pts': day_high - day_low,
            'day_range_pct': (day_high - day_low) / day_open * 100,
            'nofilter_adverse_pts': nofilter_adverse_pts,
            'nofilter_adverse_pct': nofilter_adverse_pct,
            'peak_time': peak_time,
            'peak_bw': peak_bw,
            'cooloff_time': cooloff_time,
            'cooloff_idx': cooloff_idx,
            'containment_time': containment_time,
            'containment_idx': containment_idx,
            'containment_pct': containment_pct,
            'squeeze_time': squeeze_time,
            'squeeze_idx': squeeze_idx,
            'starc_time': starc_time,
            'starc_idx': starc_idx,
            'fractal_time': fractal_time,
            'fractal_idx': fractal_idx,
            'atr_adv_pct': atr_adv_pct,
            'kc_adv_pct': kc_adv_pct,
            'starc_adv_pct': starc_adv_pct,
            'fractal_adv_pct': fractal_adv_pct,
            'combined_time': combined_time,
            'combined_idx': combined_idx,
            'combined_adv_pct': combined_adv_pct,
            'consec_inside': consec_inside,
            'pre_range_pts': pre_range_pts,
            'pre_range_pct': pre_range_pct,
            'post_range_pts': post_range_pts,
            'post_range_pct': post_range_pct,
            'max_adverse_pts': max_adverse_pts,
            'max_adverse_pct': max_adverse_pct,
            'cooloff_to_close_pts': cooloff_to_close_pts,
            'cooloff_to_close_pct': cooloff_to_close_pct,
        }
        record.update(entry_adverse)
        records.append(record)

    return pd.DataFrame(records)


def time_to_minutes(t):
    """Convert time object to minutes since midnight."""
    if t is None:
        return None
    return t.hour * 60 + t.minute


def minutes_to_time_str(m):
    """Convert minutes since midnight to HH:MM string."""
    if m is None or np.isnan(m):
        return 'N/A'
    h = int(m) // 60
    mn = int(m) % 60
    return f'{h:02d}:{mn:02d}'


def print_header(title, char='-'):
    width = 80
    print()
    print(char * width)
    print(f'  {title}')
    print(char * width)


def print_subheader(title):
    print(f'\n  --- {title} ---')


def run_analysis():
    results = {}

    for symbol in SYMBOLS:
        print(f'\nLoading {symbol} 5-minute data...')
        df = load_data(symbol)
        print(f'  Loaded {len(df)} bars, {df["trading_date"].nunique()} trading days')
        print(f'  Date range: {df.index.min()} to {df.index.max()}')

        print(f'Computing indicators for {symbol}...')
        df = compute_indicators(df)

        print(f'Analyzing daily patterns for {symbol}...')
        daily = analyze_daily(df, symbol)
        print(f'  Analyzed {len(daily)} complete trading days')
        results[symbol] = daily

    # =========================================================================
    # SECTION 1: Daily Volatility Lifecycle Timing
    # =========================================================================
    print_header('SECTION 1: DAILY VOLATILITY LIFECYCLE TIMING', '=')

    for symbol in SYMBOLS:
        daily = results[symbol]
        print_subheader(symbol)

        # Convert times to minutes for stats
        peak_mins = daily['peak_time'].apply(time_to_minutes).dropna()
        cooloff_mins = daily['cooloff_time'].apply(time_to_minutes).dropna()
        contain_mins = daily['containment_time'].apply(time_to_minutes).dropna()
        squeeze_mins = daily['squeeze_time'].apply(time_to_minutes).dropna()

        starc_mins = daily['starc_time'].apply(time_to_minutes).dropna()
        fractal_mins = daily['fractal_time'].apply(time_to_minutes).dropna()
        combined_mins = daily['combined_time'].apply(time_to_minutes).dropna()

        n_days = len(daily)
        n_cooloff = cooloff_mins.count()
        n_contain = contain_mins.count()
        n_squeeze = squeeze_mins.count()
        n_starc = starc_mins.count()
        n_fractal = fractal_mins.count()
        n_combined = combined_mins.count()

        print(f'  Total trading days analyzed: {n_days}')
        print(f'  Days with BB cooloff found:  {n_cooloff} ({n_cooloff/n_days*100:.1f}%)')
        print(f'  Days with ATR squeeze:       {n_squeeze} ({n_squeeze/n_days*100:.1f}%)')
        print(f'  Days with ATR+BB combined:   {n_combined} ({n_combined/n_days*100:.1f}%)')
        print(f'  Days with KC containment:    {n_contain} ({n_contain/n_days*100:.1f}%)')
        print(f'  Days with STARC containment: {n_starc} ({n_starc/n_days*100:.1f}%)')
        print(f'  Days with Fractal contain:   {n_fractal} ({n_fractal/n_days*100:.1f}%)')

        print(f'\n  {"Indicator":<25} {"Mean":>8} {"Median":>8} {"P25":>8} {"P75":>8}')
        print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')

        for label, mins in [('BB Peak', peak_mins), ('BB Cool-off', cooloff_mins),
                            ('ATR Squeeze', squeeze_mins), ('ATR+BB Combined', combined_mins),
                            ('KC Containment', contain_mins),
                            ('STARC Containment', starc_mins), ('Fractal Containment', fractal_mins)]:
            if len(mins) > 0:
                print(f'  {label:<25} {minutes_to_time_str(mins.mean()):>8} '
                      f'{minutes_to_time_str(mins.median()):>8} '
                      f'{minutes_to_time_str(mins.quantile(0.25)):>8} '
                      f'{minutes_to_time_str(mins.quantile(0.75)):>8}')
            else:
                print(f'  {label:<25} {"N/A":>8} {"N/A":>8} {"N/A":>8} {"N/A":>8}')

        # Distribution: % of days cool off by each time
        print(f'\n  Cool-off time distribution:')
        print(f'  {"By Time":<12} {"% Days":>8} {"Cumulative":>12}')
        print(f'  {"-"*12} {"-"*8} {"-"*12}')
        cum = 0
        thresholds = [(10, 0, '10:00'), (10, 30, '10:30'), (11, 0, '11:00'),
                      (11, 30, '11:30'), (12, 0, '12:00'), (15, 30, 'Later')]
        for h, m, label in thresholds:
            threshold_min = h * 60 + m
            pct = (cooloff_mins <= threshold_min).sum() / n_days * 100
            delta = pct - cum
            print(f'  {label:<12} {delta:>7.1f}% {pct:>11.1f}%')
            cum = pct

    # =========================================================================
    # SECTION 2: Pre vs Post Cool-off Comparison
    # =========================================================================
    print_header('SECTION 2: PRE vs POST COOL-OFF COMPARISON', '=')

    for symbol in SYMBOLS:
        daily = results[symbol]
        has_cooloff = daily['cooloff_time'].notna()
        d = daily[has_cooloff]
        print_subheader(f'{symbol} ({len(d)} days with cool-off)')

        pre_range = d['pre_range_pct'].dropna()
        post_range = d['post_range_pct'].dropna()
        adverse = d['max_adverse_pct'].dropna()

        print(f'  {"Metric":<30} {"Pre-Cooloff":>14} {"Post-Cooloff":>14} {"Ratio":>8}')
        print(f'  {"-"*30} {"-"*14} {"-"*14} {"-"*8}')

        def fmt_compare(label, pre_s, post_s):
            pm, pp = pre_s.mean(), post_s.mean()
            ratio = pp / pm if pm > 0 else 0
            print(f'  {label:<30} {pm:>13.2f}% {pp:>13.2f}% {ratio:>7.2f}x')

        fmt_compare('Range (mean %)', pre_range, post_range)
        fmt_compare('Range (median %)',
                    pd.Series([pre_range.median()]), pd.Series([post_range.median()]))

        print(f'\n  Pre-cooloff range:  mean={d["pre_range_pts"].mean():.1f} pts, '
              f'median={d["pre_range_pts"].median():.1f} pts')
        print(f'  Post-cooloff range: mean={d["post_range_pts"].mean():.1f} pts, '
              f'median={d["post_range_pts"].median():.1f} pts')
        print(f'  Max adverse after cooloff: mean={adverse.mean():.2f}%, '
              f'median={adverse.median():.2f}%')

        # Volatility per bar
        n_pre = d['cooloff_idx'].mean()
        n_post = d['n_bars'].mean() - d['cooloff_idx'].mean()
        vol_per_bar_pre = pre_range.mean() / max(n_pre, 1)
        vol_per_bar_post = post_range.mean() / max(n_post, 1)
        print(f'\n  Avg bars pre-cooloff: {n_pre:.0f}, post-cooloff: {n_post:.0f}')
        print(f'  Volatility per bar:  pre={vol_per_bar_pre:.4f}%, post={vol_per_bar_post:.4f}%')
        print(f'  Morning is {vol_per_bar_pre/vol_per_bar_post:.1f}x more volatile per bar')

    # =========================================================================
    # SECTION 3: Post Cool-off Safety for Straddle Sellers
    # =========================================================================
    print_header('SECTION 3: POST COOL-OFF SAFETY FOR STRADDLE SELLERS', '=')

    for symbol in SYMBOLS:
        daily = results[symbol]
        adverse = daily['max_adverse_pct'].dropna()
        adverse_pts = daily['max_adverse_pts'].dropna()

        print_subheader(f'{symbol} - Max Adverse Move After Cool-off')

        print(f'  {"Statistic":<20} {"Points":>10} {"Percent":>10}')
        print(f'  {"-"*20} {"-"*10} {"-"*10}')
        for label, func in [('Mean', 'mean'), ('Median', 'median'),
                            ('P75', lambda s: s.quantile(0.75)),
                            ('P90', lambda s: s.quantile(0.90)),
                            ('P95', lambda s: s.quantile(0.95)),
                            ('Max', 'max')]:
            if callable(func):
                v_pts = func(adverse_pts)
                v_pct = func(adverse)
            else:
                v_pts = getattr(adverse_pts, func)()
                v_pct = getattr(adverse, func)()
            print(f'  {label:<20} {v_pts:>9.1f} {v_pct:>9.2f}%')

        # Safety rates
        thresholds = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
        print(f'\n  Safety Rate (% of days post-cooloff move stays within threshold):')
        print(f'  {"Threshold":<12} {"% Days Safe":>12} {"Days":>8}')
        print(f'  {"-"*12} {"-"*12} {"-"*8}')
        for t in thresholds:
            safe = (adverse <= t).sum()
            pct = safe / len(adverse) * 100
            print(f'  < {t:.2f}%      {pct:>11.1f}% {safe:>7d}/{len(adverse)}')

        # Cooloff to close direction
        c2c = daily['cooloff_to_close_pct'].dropna()
        print(f'\n  Cool-off close to EOD close: mean={c2c.mean():.3f}%, '
              f'median={c2c.median():.3f}%, std={c2c.std():.3f}%')
        bullish = (c2c > 0).sum()
        bearish = (c2c < 0).sum()
        print(f'  Direction: {bullish} bullish ({bullish/len(c2c)*100:.1f}%), '
              f'{bearish} bearish ({bearish/len(c2c)*100:.1f}%)')

    # =========================================================================
    # SECTION 4: DTE Cross-tabulation
    # =========================================================================
    print_header('SECTION 4: DTE CROSS-TABULATION', '=')

    # Define all filters: (label, adverse_col, time_col)
    FILTERS = [
        ('No Filter', 'nofilter_adverse_pct', None),
        ('BB Cooloff', 'max_adverse_pct', 'cooloff_time'),
        ('ATR Squeeze', 'atr_adv_pct', 'squeeze_time'),
        ('ATR+BB Combined', 'combined_adv_pct', 'combined_time'),
        ('KC Bands', 'kc_adv_pct', 'containment_time'),
        ('STARC Bands', 'starc_adv_pct', 'starc_time'),
        ('Fractal Bands', 'fractal_adv_pct', 'fractal_time'),
    ]

    for symbol in SYMBOLS:
        daily = results[symbol]
        if symbol == 'NIFTY50':
            expiry_note = 'Thu before Nov 2024, Tue after'
        else:
            expiry_note = 'Wed before Nov 2024, monthly-only after'
        print_subheader(f'{symbol} (Expiry: {expiry_note})')

        # --- Table 4A: Filter Activation Timing ---
        print(f'  Filter Activation Time (median):')
        print(f'  {"Filter":<18} {"Median":>8} {"Mean":>8} {"Days Active":>12}')
        print(f'  {"-"*18} {"-"*8} {"-"*8} {"-"*12}')
        for flabel, adv_col, t_col in FILTERS:
            if t_col is None:
                print(f'  {"No Filter":<18} {"09:15":>8} {"09:15":>8} {len(daily):>5}/{len(daily)}')
                continue
            t_mins = daily[t_col].apply(time_to_minutes).dropna()
            n_active = len(t_mins)
            med = minutes_to_time_str(t_mins.median()) if n_active > 0 else 'N/A'
            mean = minutes_to_time_str(t_mins.mean()) if n_active > 0 else 'N/A'
            print(f'  {flabel:<18} {med:>8} {mean:>8} {n_active:>5}/{len(daily)}')

        # --- Table 4B: All Filters x DTE - <0.5% Safety Rate ---
        print(f'\n  <0.5% Safety Rate by Filter x DTE (% of days max adverse < 0.5%):')
        header = f'  {"DTE":<5} {"Days":>5}'
        for flabel, _, _ in FILTERS:
            header += f' {flabel:>14}'
        print(header)
        print(f'  {"-"*5} {"-"*5}' + f' {"-"*14}' * len(FILTERS))

        max_dte = int(daily['dte'].max()) + 1 if len(daily) > 0 else 5
        max_dte = min(max_dte, 25)
        for dte in range(max_dte):
            subset = daily[daily['dte'] == dte]
            n = len(subset)
            if n < 3:
                continue
            row = f'  {dte:<5} {n:>5}'
            for flabel, adv_col, _ in FILTERS:
                adv = subset[adv_col].dropna()
                if len(adv) > 0:
                    safe = (adv <= 0.5).sum() / len(adv) * 100
                    row += f' {safe:>13.1f}%'
                else:
                    row += f' {"N/A":>14}'
            print(row)

        # --- Table 4C: All Filters x DTE - Mean Max Adverse ---
        print(f'\n  Mean Max Adverse (%) by Filter x DTE:')
        print(header)
        print(f'  {"-"*5} {"-"*5}' + f' {"-"*14}' * len(FILTERS))

        for dte in range(max_dte):
            subset = daily[daily['dte'] == dte]
            n = len(subset)
            if n < 3:
                continue
            row = f'  {dte:<5} {n:>5}'
            for flabel, adv_col, _ in FILTERS:
                adv = subset[adv_col].dropna()
                if len(adv) > 0:
                    row += f' {adv.mean():>13.2f}%'
                else:
                    row += f' {"N/A":>14}'
            print(row)

        # --- Table 4D: Filter Edge over No Filter (all DTEs combined) ---
        print(f'\n  FILTER COMPARISON (all DTEs combined, {len(daily)} days):')
        print(f'  {"Filter":<18} {"Avg MaxAdv":>10} {"Med MaxAdv":>10} '
              f'{"<0.3%":>7} {"<0.5%":>7} {"<0.75%":>7} {"<1.0%":>7} {"Edge vs Raw":>11}')
        print(f'  {"-"*18} {"-"*10} {"-"*10} '
              f'{"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*11}')

        raw_safe05 = 0
        for flabel, adv_col, _ in FILTERS:
            adv = daily[adv_col].dropna()
            if len(adv) == 0:
                continue
            s03 = (adv <= 0.3).sum() / len(adv) * 100
            s05 = (adv <= 0.5).sum() / len(adv) * 100
            s075 = (adv <= 0.75).sum() / len(adv) * 100
            s10 = (adv <= 1.0).sum() / len(adv) * 100
            if flabel == 'No Filter':
                raw_safe05 = s05
                edge_str = '  baseline'
            else:
                edge_str = f'{s05 - raw_safe05:>+10.1f}%'
            print(f'  {flabel:<18} {adv.mean():>9.2f}% {adv.median():>9.2f}% '
                  f'{s03:>6.1f}% {s05:>6.1f}% {s075:>6.1f}% {s10:>6.1f}% {edge_str}')

    # =========================================================================
    # SECTION 5: Best Entry Windows
    # =========================================================================
    print_header('SECTION 5: BEST ENTRY WINDOWS', '=')

    for symbol in SYMBOLS:
        daily = results[symbol]
        print_subheader(f'{symbol} - Max Adverse Move from Entry Time to EOD')

        # Entry time x DTE cross-tab
        # Use only DTEs with enough data (>= 10 days)
        dte_counts = daily['dte'].value_counts()
        dte_list = sorted([d for d, c in dte_counts.items() if c >= 10])
        if not dte_list:
            dte_list = sorted(daily['dte'].unique())[:5]

        print(f'\n  Max Adverse Move (mean %) by Entry Time x DTE:')
        header = f'  {"Entry":>8}'
        for dte in dte_list:
            header += f' {"DTE"+str(dte):>8}'
        header += f' {"All":>8}'
        print(header)
        print(f'  {"-"*8}' + f' {"-"*8}' * (len(dte_list) + 1))

        best_val = 999
        best_combo = ''
        for slot_label in TIME_SLOT_LABELS:
            col = f'{slot_label}_pct'
            if col not in daily.columns:
                continue
            row_str = f'  {slot_label:>8}'
            for dte in dte_list:
                subset = daily[daily['dte'] == dte]
                vals = subset[col].dropna()
                if len(vals) > 0:
                    v = vals.mean()
                    row_str += f' {v:>7.2f}%'
                    if v < best_val:
                        best_val = v
                        best_combo = f'{slot_label} / DTE {dte}'
                else:
                    row_str += f' {"N/A":>8}'
            all_vals = daily[col].dropna()
            row_str += f' {all_vals.mean():>7.2f}%'
            print(row_str)

        print(f'\n  Best combo (lowest avg max adverse): {best_combo} = {best_val:.2f}%')

        # Safety rate cross-tab (% days < 0.5%)
        print(f'\n  Safety Rate (<0.5% max adverse) by Entry Time x DTE:')
        print(header)
        print(f'  {"-"*8}' + f' {"-"*8}' * (len(dte_list) + 1))

        best_safe = 0
        best_safe_combo = ''
        for slot_label in TIME_SLOT_LABELS:
            col = f'{slot_label}_pct'
            if col not in daily.columns:
                continue
            row_str = f'  {slot_label:>8}'
            for dte in dte_list:
                subset = daily[daily['dte'] == dte]
                vals = subset[col].dropna()
                if len(vals) > 0:
                    safe = (vals <= 0.5).sum() / len(vals) * 100
                    row_str += f' {safe:>7.1f}%'
                    if safe > best_safe:
                        best_safe = safe
                        best_safe_combo = f'{slot_label} / DTE {dte}'
                else:
                    row_str += f' {"N/A":>8}'
            all_vals = daily[col].dropna()
            all_safe = (all_vals <= 0.5).sum() / len(all_vals) * 100
            row_str += f' {all_safe:>7.1f}%'
            print(row_str)

        print(f'\n  Safest combo (<0.5%): {best_safe_combo} = {best_safe:.1f}% of days')

        # Also show <1.0% safety
        print(f'\n  Safety Rate (<1.0% max adverse) by Entry Time x DTE:')
        print(header)
        print(f'  {"-"*8}' + f' {"-"*8}' * (len(dte_list) + 1))

        for slot_label in TIME_SLOT_LABELS:
            col = f'{slot_label}_pct'
            if col not in daily.columns:
                continue
            row_str = f'  {slot_label:>8}'
            for dte in dte_list:
                subset = daily[daily['dte'] == dte]
                vals = subset[col].dropna()
                if len(vals) > 0:
                    safe = (vals <= 1.0).sum() / len(vals) * 100
                    row_str += f' {safe:>7.1f}%'
                else:
                    row_str += f' {"N/A":>8}'
            all_vals = daily[col].dropna()
            all_safe = (all_vals <= 1.0).sum() / len(all_vals) * 100
            row_str += f' {all_safe:>7.1f}%'
            print(row_str)

    # =========================================================================
    # SECTION 6: Consecutive Sideways Behavior
    # =========================================================================
    print_header('SECTION 6: CONSECUTIVE SIDEWAYS BEHAVIOR', '=')

    for symbol in SYMBOLS:
        daily = results[symbol]
        has_contain = daily['containment_time'].notna()
        d = daily[has_contain]
        consec = d['consec_inside']
        contain_pct = d['containment_pct'].dropna()

        print_subheader(symbol)
        print(f'  Days with KC containment: {len(d)}')
        print(f'\n  Consecutive inside-band bars after first containment:')
        print(f'    Mean:   {consec.mean():.1f} bars ({consec.mean()*5:.0f} min)')
        print(f'    Median: {consec.median():.0f} bars ({consec.median()*5:.0f} min)')
        print(f'    P75:    {consec.quantile(0.75):.0f} bars')
        print(f'    P90:    {consec.quantile(0.90):.0f} bars')
        print(f'    Max:    {consec.max():.0f} bars')

        print(f'\n  % of remaining candles that stay inside bands after containment:')
        print(f'    Mean:   {contain_pct.mean():.1f}%')
        print(f'    Median: {contain_pct.median():.1f}%')
        print(f'    P25:    {contain_pct.quantile(0.25):.1f}%')

        # Breakout probability after N consecutive inside bars
        print(f'\n  Breakout probability after N consecutive inside bars:')
        print(f'  {"N bars inside":<15} {"% days breakout occurs after":>30}')
        print(f'  {"-"*15} {"-"*30}')
        for n_thresh in [1, 3, 5, 10, 15, 20]:
            days_reaching = (consec >= n_thresh).sum()
            days_breaking = (consec == n_thresh).sum()  # broke out exactly at N
            if days_reaching > 0:
                # Actually: of days that reached N consecutive, what % broke out by N+2?
                # Simpler: what % of days have EXACTLY n_thresh consecutive (broke out at N)?
                # Better interpretation: of days reaching N, what % did NOT reach N+5?
                days_not_5more = ((consec >= n_thresh) & (consec < n_thresh + 5)).sum()
                pct = days_not_5more / days_reaching * 100
                print(f'  {n_thresh:>5} bars     {pct:>10.1f}% break within next 5 bars  '
                      f'({days_reaching} days reached {n_thresh})')
            else:
                print(f'  {n_thresh:>5} bars     {"N/A":>10}')

    # =========================================================================
    # SECTION 7: Weekly/Monthly Patterns
    # =========================================================================
    print_header('SECTION 7: WEEKLY / MONTHLY PATTERNS', '=')

    for symbol in SYMBOLS:
        daily = results[symbol]
        print_subheader(f'{symbol} - Day of Week Analysis')

        print(f'  {"Day":<6} {"Days":>6} {"DayRange%":>10} {"Cooloff":>8} '
              f'{"MaxAdv%":>8} {"<0.5%":>7} {"<1.0%":>7}')
        print(f'  {"-"*6} {"-"*6} {"-"*10} {"-"*8} {"-"*8} {"-"*7} {"-"*7}')

        cooloff_mins = daily['cooloff_time'].apply(time_to_minutes)
        for wd, name in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri']):
            subset = daily[daily['weekday'] == wd]
            n = len(subset)
            if n == 0:
                continue
            dr = subset['day_range_pct'].mean()
            co = minutes_to_time_str(cooloff_mins[subset.index].mean())
            adv = subset['max_adverse_pct'].dropna()
            adv_m = adv.mean() if len(adv) > 0 else 0
            safe05 = (adv <= 0.5).sum() / max(len(adv), 1) * 100
            safe10 = (adv <= 1.0).sum() / max(len(adv), 1) * 100
            print(f'  {name:<6} {n:>6} {dr:>9.2f}% {co:>8} '
                  f'{adv_m:>7.2f}% {safe05:>6.1f}% {safe10:>6.1f}%')

        # Monthly patterns
        daily_copy = daily.copy()
        daily_copy['month'] = pd.to_datetime(daily_copy['date']).dt.month
        daily_copy['year_month'] = pd.to_datetime(daily_copy['date']).dt.to_period('M')

        print_subheader(f'{symbol} - Monthly Volatility (Day Range %)')
        monthly = daily_copy.groupby('month')['day_range_pct'].agg(['mean', 'median', 'count'])
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f'  {"Month":<6} {"Days":>6} {"Mean %":>8} {"Median %":>10}')
        print(f'  {"-"*6} {"-"*6} {"-"*8} {"-"*10}')
        for m, row in monthly.iterrows():
            print(f'  {month_names[m]:<6} {row["count"]:>6.0f} {row["mean"]:>7.2f}% '
                  f'{row["median"]:>9.2f}%')

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header('SUMMARY: OPTIMAL STRATEGY PARAMETERS', '=')

    for symbol in SYMBOLS:
        daily = results[symbol]
        cooloff_mins = daily['cooloff_time'].apply(time_to_minutes).dropna()
        adverse = daily['max_adverse_pct'].dropna()

        print_subheader(symbol)

        # Find best entry time overall
        best_entry = None
        best_entry_adv = 999
        for slot_label in TIME_SLOT_LABELS:
            col = f'{slot_label}_pct'
            if col in daily.columns:
                v = daily[col].dropna().mean()
                if v < best_entry_adv:
                    best_entry_adv = v
                    best_entry = slot_label

        # Find best DTE (only from DTEs with enough data)
        best_dte = None
        best_dte_adv = 999
        dte_counts = daily['dte'].value_counts()
        valid_dtes = sorted([d for d, c in dte_counts.items() if c >= 10])
        for dte in valid_dtes:
            subset = daily[daily['dte'] == dte]
            adv = subset['max_adverse_pct'].dropna()
            if len(adv) > 0 and adv.mean() < best_dte_adv:
                best_dte_adv = adv.mean()
                best_dte = dte

        # Median cooloff
        med_cooloff = minutes_to_time_str(cooloff_mins.median())

        # P90 adverse
        p90_adv = adverse.quantile(0.90)

        print(f'  1. VOLATILITY LIFECYCLE:')
        print(f'     - BB Bandwidth peaks around {minutes_to_time_str(daily["peak_time"].apply(time_to_minutes).median())}')
        print(f'     - Cools off (50% of peak) by {med_cooloff} (median)')
        print(f'     - {(cooloff_mins <= 11*60).sum()/len(cooloff_mins)*100:.0f}% of days cool off by 11:00')

        print(f'\n  2. RECOMMENDED ENTRY WINDOW:')
        print(f'     - Best entry time: {best_entry} (avg max adverse: {best_entry_adv:.2f}%)')
        print(f'     - Best DTE: {best_dte} (avg max adverse: {best_dte_adv:.2f}%)')
        print(f'     - Wait for BB bandwidth to drop below 50% of morning peak')

        print(f'\n  3. RISK SIZING:')
        print(f'     - After cooloff, median max adverse move: {adverse.median():.2f}%')
        print(f'     - P90 max adverse: {p90_adv:.2f}%')
        print(f'     - Set straddle strikes {p90_adv:.1f}% away for 90% safety')

        contain_pct = daily['containment_pct'].dropna()
        print(f'\n  4. BAND CONTAINMENT:')
        print(f'     - After first KC containment, {contain_pct.mean():.0f}% of remaining bars stay inside (mean)')
        print(f'     - Containment is a strong confirmation signal for low-vol regime')

        # Day of week recommendation
        wd_safest = None
        wd_safest_rate = 0
        for wd in range(5):
            subset = daily[daily['weekday'] == wd]
            adv = subset['max_adverse_pct'].dropna()
            if len(adv) > 0:
                safe = (adv <= 1.0).sum() / len(adv) * 100
                if safe > wd_safest_rate:
                    wd_safest_rate = safe
                    wd_safest = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][wd]

        print(f'\n  5. BEST DAY OF WEEK: {wd_safest} ({wd_safest_rate:.0f}% of days < 1% adverse)')

    print('\n' + '=' * 80)
    print('  Analysis complete.')
    print('=' * 80)


if __name__ == '__main__':
    run_analysis()
