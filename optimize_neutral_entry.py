#!/usr/bin/env python3
"""
Neutral Entry Optimization for NIFTY50 & BANKNIFTY
====================================================
Finds the best entry filters + exit times for straddle/strangle sellers.

Base signal: BB Bandwidth Cooloff (bandwidth drops below 50% of day's peak).
Tests 22 additional filters on top of BB cooloff.
Tests 9 exit times from 12:00 to 15:25.
Analyzes by DTE (days to expiry).

Data: 5-minute candles from market_data.db (~456 trading days, Mar 2024 - Mar 2026)
"""

import os, sys, time, warnings, csv
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'optimization_neutral_entry.csv')

SYMBOLS = ['NIFTY50', 'BANKNIFTY']
EXIT_TIMES = ['12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:15', '15:25']
BB_PERIOD = 20
BB_STD = 2.0
BB_COOLOFF_PCT = 0.50  # bandwidth drops below 50% of day's peak

# ─── Expiry date calculation ───


def build_expiry_dates(trading_dates, symbol):
    """Build list of expiry dates for a symbol based on actual trading days."""
    from datetime import date as dt_date
    trading_set = set(trading_dates)

    # Generate all potential expiry dates in range
    start = min(trading_dates) - timedelta(days=7)
    end = max(trading_dates) + timedelta(days=7)

    expiries = []
    current = start
    while current <= end:
        is_expiry = False

        if symbol == 'NIFTY50':
            # Before Nov 25, 2024: Thursday weekly expiry
            # From Nov 25, 2024: Tuesday weekly expiry
            cutoff = dt_date(2024, 11, 25)
            if current < cutoff:
                is_expiry = (current.weekday() == 3)  # Thursday
            else:
                is_expiry = (current.weekday() == 1)  # Tuesday
        elif symbol == 'BANKNIFTY':
            # Before Nov 13, 2024: Wednesday weekly expiry
            # After: monthly only (last Thursday of month)
            cutoff = dt_date(2024, 11, 13)
            if current < cutoff:
                is_expiry = (current.weekday() == 2)  # Wednesday
            else:
                # Last Thursday of month
                if current.weekday() == 3:  # Thursday
                    # Check if next Thursday is in a different month
                    next_thu = current + timedelta(days=7)
                    is_expiry = (next_thu.month != current.month)

        if is_expiry:
            # If expiry day is not a trading day, use previous trading day
            exp = current
            while exp not in trading_set and exp >= start:
                exp -= timedelta(days=1)
            if exp in trading_set:
                expiries.append(exp)

        current += timedelta(days=1)

    return sorted(set(expiries))


def calc_dte(trade_date, expiry_dates, trading_dates):
    """Calculate DTE = number of trading days to next expiry."""
    trading_set = set(trading_dates)
    for exp in expiry_dates:
        if exp >= trade_date:
            # Count trading days between trade_date and exp (inclusive of exp, exclusive of trade_date)
            dte = sum(1 for d in trading_dates if trade_date < d <= exp)
            return dte
    return -1  # No expiry found


# ─── Data loading ───


def load_data():
    """Load 5-min and daily data for both symbols."""
    conn = sqlite3.connect(DB_PATH)

    data = {}
    daily_data = {}

    for sym in SYMBOLS:
        print(f"  Loading 5-min data for {sym}...", end='', flush=True)
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='5minute' ORDER BY date",
            conn, params=(sym,)
        )
        df['datetime'] = pd.to_datetime(df['date'])
        df['trade_date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.strftime('%H:%M')
        df = df.sort_values('datetime').reset_index(drop=True)
        data[sym] = df
        print(f" {len(df)} candles, {df['trade_date'].nunique()} days", flush=True)

        print(f"  Loading daily data for {sym}...", end='', flush=True)
        ddf = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date",
            conn, params=(sym,)
        )
        ddf['trade_date'] = pd.to_datetime(ddf['date']).dt.date
        ddf = ddf.sort_values('trade_date').reset_index(drop=True)
        daily_data[sym] = ddf
        print(f" {len(ddf)} days", flush=True)

    conn.close()
    return data, daily_data


# ─── Indicator computation ───


def compute_bb_cooloff(day_df):
    """
    Compute BB bandwidth cooloff for a single day's 5-min data.
    Uses rolling BB(20,2) across the day's candles.
    Returns cooloff bar index (within day_df) or None.

    We need prior data for BB warmup, so this is called with extended data.
    """
    closes = day_df['close'].values
    n = len(closes)
    if n < BB_PERIOD:
        return None

    # Compute BB bandwidth for each bar
    bandwidths = np.full(n, np.nan)
    for i in range(BB_PERIOD - 1, n):
        window = closes[i - BB_PERIOD + 1:i + 1]
        mid = np.mean(window)
        std = np.std(window, ddof=1)
        if mid > 0:
            upper = mid + BB_STD * std
            lower = mid - BB_STD * std
            bandwidths[i] = (upper - lower) / mid * 100

    return bandwidths


def compute_day_analysis(sym, df_5min, daily_df, trading_dates, expiry_dates):
    """
    For each trading day, compute BB cooloff time and all entry filters.
    Returns a list of dicts, one per day where BB cooloff occurs.
    """
    results = []

    # Get unique trading days from 5-min data
    all_days = sorted(df_5min['trade_date'].unique())

    # Build daily lookup
    daily_lookup = {}
    for _, row in daily_df.iterrows():
        daily_lookup[row['trade_date']] = row

    # Build weekly high/low lookup (previous 5 trading days)
    daily_dates = sorted(daily_lookup.keys())

    # We need BB warmup, so compute BB across all data continuously
    # but track day boundaries for cooloff detection
    closes_all = df_5min['close'].values
    n_total = len(closes_all)

    # Compute BB bandwidth for entire series
    print("    Computing BB bandwidth...", end='', flush=True)
    bandwidths = np.full(n_total, np.nan)
    for i in range(BB_PERIOD - 1, n_total):
        window = closes_all[i - BB_PERIOD + 1:i + 1]
        mid = np.mean(window)
        std = np.std(window, ddof=1)
        if mid > 0:
            upper = mid + BB_STD * std
            lower = mid - BB_STD * std
            bandwidths[i] = (upper - lower) / mid * 100
    print(" done", flush=True)

    df_5min = df_5min.copy()
    df_5min['bandwidth'] = bandwidths

    # Compute VWAP (volume may be 0 for index)
    has_volume = (df_5min['volume'] > 0).any()

    processed = 0
    skipped_no_cooloff = 0

    for day_idx, day in enumerate(all_days):
        day_mask = df_5min['trade_date'] == day
        day_df = df_5min[day_mask].copy()

        if len(day_df) < 10:
            continue

        times = day_df['time'].values

        # Only consider candles from 9:15 onwards (market hours)
        market_mask = times >= '09:15'
        if not market_mask.any():
            continue

        day_bw = day_df['bandwidth'].values

        # Find BB peak bandwidth for the day (running peak)
        # Cooloff = first bar after peak where bw drops below 50% of peak
        peak_bw = np.nan
        cooloff_idx = None  # index within day_df

        for i in range(len(day_df)):
            bw = day_bw[i]
            if np.isnan(bw):
                continue
            if np.isnan(peak_bw) or bw > peak_bw:
                peak_bw = bw
            elif bw < peak_bw * BB_COOLOFF_PCT and peak_bw > 0:
                # Only trigger cooloff after at least seeing a peak
                # Must be during market hours and after 9:30 (need some data)
                if times[i] >= '09:30':
                    cooloff_idx = i
                    break

        if cooloff_idx is None:
            skipped_no_cooloff += 1
            continue

        cooloff_time = times[cooloff_idx]
        cooloff_price = day_df['close'].values[cooloff_idx]
        cooloff_high = day_df['high'].values[cooloff_idx]
        cooloff_low = day_df['low'].values[cooloff_idx]

        # ─── Get previous day data ───
        day_pos_in_daily = None
        for di, dd in enumerate(daily_dates):
            if dd >= day:
                if di > 0:
                    day_pos_in_daily = di
                break

        if day_pos_in_daily is None or day_pos_in_daily < 1:
            continue

        prev_day_date = daily_dates[day_pos_in_daily - 1]
        if prev_day_date not in daily_lookup:
            continue
        prev_day = daily_lookup[prev_day_date]
        prev_h = prev_day['high']
        prev_l = prev_day['low']
        prev_c = prev_day['close']
        prev_o = prev_day['open']

        # ─── FILTER A: CPR ───
        pivot = (prev_h + prev_l + prev_c) / 3
        tc = 2 * pivot - (prev_h + prev_l) / 2
        bc = (prev_h + prev_l) / 2
        # Ensure tc > bc
        if tc < bc:
            tc, bc = bc, tc
        cpr_width = abs(tc - bc) / pivot * 100 if pivot > 0 else 0

        if cooloff_price > tc:
            cpr_position = 'Above'
        elif cooloff_price < bc:
            cpr_position = 'Below'
        else:
            cpr_position = 'Inside'

        # ─── FILTER B: Previous day context ───
        prev_range_pct = (prev_h - prev_l) / prev_c * 100 if prev_c > 0 else 0
        prev_trend = 'Bullish' if prev_c > prev_o else 'Bearish'

        # Inside day check (at cooloff time)
        today_open = day_df['open'].values[0]
        today_high_at_cooloff = day_df['high'].values[:cooloff_idx + 1].max()
        today_low_at_cooloff = day_df['low'].values[:cooloff_idx + 1].min()
        inside_day = (today_open >= prev_l and today_open <= prev_h and
                      today_high_at_cooloff < prev_h and today_low_at_cooloff > prev_l)

        # ─── FILTER C: Prev day/week high-low break ───
        broken_prev_high = today_high_at_cooloff > prev_h
        broken_prev_low = today_low_at_cooloff < prev_l
        no_prev_day_break = not broken_prev_high and not broken_prev_low

        # Previous week high/low (last 5 trading days)
        week_start = max(0, day_pos_in_daily - 5)
        prev_week_dates = daily_dates[week_start:day_pos_in_daily]
        if len(prev_week_dates) >= 2:
            prev_week_high = max(daily_lookup[d]['high'] for d in prev_week_dates)
            prev_week_low = min(daily_lookup[d]['low'] for d in prev_week_dates)
            broken_week_high = today_high_at_cooloff > prev_week_high
            broken_week_low = today_low_at_cooloff < prev_week_low
            no_prev_week_break = not broken_week_high and not broken_week_low
        else:
            no_prev_week_break = None

        # ─── FILTER D: Gap analysis ───
        gap_pct = (today_open - prev_c) / prev_c * 100 if prev_c > 0 else 0
        # Gap filled = price retraced to prev close by cooloff
        if gap_pct > 0:
            gap_filled = today_low_at_cooloff <= prev_c
        elif gap_pct < 0:
            gap_filled = today_high_at_cooloff >= prev_c
        else:
            gap_filled = True

        # ─── FILTER E: Opening Range (first 30 min, 9:15-9:45) ───
        or_mask = (times >= '09:15') & (times < '09:45')
        or_candles = day_df[or_mask]
        if len(or_candles) >= 4:
            or_high = or_candles['high'].max()
            or_low = or_candles['low'].min()
            or_range_pct = (or_high - or_low) / ((or_high + or_low) / 2) * 100
            or_breakout = today_high_at_cooloff > or_high or today_low_at_cooloff < or_low
        else:
            or_high = or_low = or_range_pct = None
            or_breakout = None

        # ─── FILTER F: VWAP position (skip if no volume) ───
        vwap_above = None
        if has_volume:
            cum_vol = day_df['volume'].values[:cooloff_idx + 1].cumsum()
            tp = (day_df['high'].values[:cooloff_idx + 1] +
                  day_df['low'].values[:cooloff_idx + 1] +
                  day_df['close'].values[:cooloff_idx + 1]) / 3
            cum_tp_vol = (tp * day_df['volume'].values[:cooloff_idx + 1]).cumsum()
            if cum_vol[-1] > 0:
                vwap = cum_tp_vol[-1] / cum_vol[-1]
                vwap_above = cooloff_price > vwap

        # ─── FILTER G: Narrow cooloff candles ───
        # Last 3 candles before cooloff: are they narrow range?
        if cooloff_idx >= 3:
            recent_ranges = (day_df['high'].values[cooloff_idx - 3:cooloff_idx] -
                           day_df['low'].values[cooloff_idx - 3:cooloff_idx])
            all_ranges = day_df['high'].values[:cooloff_idx] - day_df['low'].values[:cooloff_idx]
            avg_range = np.mean(all_ranges[all_ranges > 0]) if (all_ranges > 0).any() else 1
            narrow_cooloff_candles = np.mean(recent_ranges) < avg_range * 0.7
        else:
            narrow_cooloff_candles = None

        # ─── EXIT TIME analysis ───
        # Max adverse move = max absolute % move from cooloff price during held period
        exit_data = {}
        for exit_time in EXIT_TIMES:
            if cooloff_time >= exit_time:
                continue  # Can't exit before entry

            # Get candles from cooloff to exit time
            held_mask = (times > cooloff_time) & (times <= exit_time)
            held_candles = day_df[held_mask]

            if len(held_candles) == 0:
                continue

            # Max adverse = max absolute % deviation from entry
            held_highs = held_candles['high'].values
            held_lows = held_candles['low'].values
            max_up = (held_highs.max() - cooloff_price) / cooloff_price * 100
            max_down = (cooloff_price - held_lows.min()) / cooloff_price * 100
            max_adverse = max(abs(max_up), abs(max_down))

            # Final P&L (for a neutral position, adverse is what matters)
            exit_close = held_candles['close'].values[-1]
            final_move_pct = abs(exit_close - cooloff_price) / cooloff_price * 100

            exit_data[exit_time] = {
                'max_adverse': max_adverse,
                'final_move_pct': final_move_pct,
                'max_up': max_up,
                'max_down': max_down,
            }

        # Also compute entry-to-EOD max adverse
        eod_mask = times > cooloff_time
        eod_candles = day_df[eod_mask]
        if len(eod_candles) > 0:
            eod_highs = eod_candles['high'].values
            eod_lows = eod_candles['low'].values
            max_up_eod = (eod_highs.max() - cooloff_price) / cooloff_price * 100
            max_down_eod = (cooloff_price - eod_lows.min()) / cooloff_price * 100
            max_adverse_eod = max(abs(max_up_eod), abs(max_down_eod))
        else:
            max_adverse_eod = 0

        # ─── DTE ───
        dte = calc_dte(day, expiry_dates, list(trading_dates))

        results.append({
            'date': day,
            'symbol': sym,
            'cooloff_time': cooloff_time,
            'cooloff_price': cooloff_price,
            'max_adverse_eod': max_adverse_eod,
            # Filters
            'cpr_width': cpr_width,
            'cpr_position': cpr_position,
            'prev_range_pct': prev_range_pct,
            'prev_trend': prev_trend,
            'inside_day': inside_day,
            'no_prev_day_break': no_prev_day_break,
            'broken_prev_high': broken_prev_high,
            'broken_prev_low': broken_prev_low,
            'no_prev_week_break': no_prev_week_break,
            'gap_pct': gap_pct,
            'gap_filled': gap_filled,
            'or_breakout': or_breakout,
            'or_range_pct': or_range_pct,
            'narrow_cooloff_candles': narrow_cooloff_candles,
            'dte': dte,
            # Exit data
            'exit_data': exit_data,
        })

        processed += 1

    print(f"    {sym}: {processed} days with BB cooloff, {skipped_no_cooloff} days without cooloff", flush=True)
    return results


# ─── Analysis functions ───


def safety_stats(max_advs):
    """Compute safety statistics for a list of max adverse moves."""
    if len(max_advs) == 0:
        return {}
    arr = np.array(max_advs)
    return {
        'days': len(arr),
        'avg': np.mean(arr),
        'med': np.median(arr),
        'p75': np.percentile(arr, 75),
        'p90': np.percentile(arr, 90),
        'lt_0.3': np.mean(arr < 0.3) * 100,
        'lt_0.5': np.mean(arr < 0.5) * 100,
        'lt_0.75': np.mean(arr < 0.75) * 100,
        'lt_1.0': np.mean(arr < 1.0) * 100,
    }


def print_table(title, headers, rows, col_widths=None):
    """Print a formatted ASCII table."""
    if col_widths is None:
        col_widths = []
        for j, h in enumerate(headers):
            w = len(str(h))
            for r in rows:
                w = max(w, len(str(r[j])) if j < len(r) else 0)
            col_widths.append(w + 2)

    print(f"\n{'=' * 80}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'=' * 80}", flush=True)

    # Header
    hdr = ''.join(str(h).ljust(col_widths[j]) for j, h in enumerate(headers))
    print(hdr, flush=True)
    print('-' * sum(col_widths), flush=True)

    for row in rows:
        line = ''.join(str(row[j] if j < len(row) else '').ljust(col_widths[j]) for j in range(len(headers)))
        print(line, flush=True)


def run_filter_analysis(all_results, sym_label):
    """Run all filter tests and produce output tables for one symbol or combined."""

    # Compute adaptive thresholds
    cpr_widths = [r['cpr_width'] for r in all_results if r['cpr_width'] is not None]
    median_cpr = np.median(cpr_widths) if cpr_widths else 0.5

    prev_ranges = [r['prev_range_pct'] for r in all_results]
    median_prev_range = np.median(prev_ranges) if prev_ranges else 1.0

    # ─── Define filters ───
    filters = {
        '01. BB Only (baseline)': lambda r: True,
        '02. BB + Narrow CPR': lambda r: r['cpr_width'] < median_cpr,
        '03. BB + Wide CPR': lambda r: r['cpr_width'] >= median_cpr,
        '04. BB + CPR Inside': lambda r: r['cpr_position'] == 'Inside',
        '05. BB + CPR Above': lambda r: r['cpr_position'] == 'Above',
        '06. BB + CPR Below': lambda r: r['cpr_position'] == 'Below',
        '07. BB + No Prev Day Break': lambda r: r['no_prev_day_break'],
        '08. BB + Prev Day High Broken': lambda r: r['broken_prev_high'],
        '09. BB + Prev Day Low Broken': lambda r: r['broken_prev_low'],
        '10. BB + No Prev Week Break': lambda r: r.get('no_prev_week_break') == True,
        '11. BB + Inside Day': lambda r: r['inside_day'],
        '12. BB + Narrow Prev Day': lambda r: r['prev_range_pct'] < median_prev_range,
        '13. BB + Wide Prev Day': lambda r: r['prev_range_pct'] >= median_prev_range,
        '14. BB + No Gap (|gap|<0.1%)': lambda r: abs(r['gap_pct']) < 0.1,
        '15. BB + Gap Up (>0.3%)': lambda r: r['gap_pct'] > 0.3,
        '16. BB + Gap Down (<-0.3%)': lambda r: r['gap_pct'] < -0.3,
        '17. BB + Gap Filled': lambda r: r['gap_filled'],
        '18. BB + No ORB': lambda r: r.get('or_breakout') == False,
        '19. BB + ORB happened': lambda r: r.get('or_breakout') == True,
        '20. BB + Narrow Cooloff Candles': lambda r: r.get('narrow_cooloff_candles') == True,
        '21. BB + Prev Day Bullish': lambda r: r['prev_trend'] == 'Bullish',
        '22. BB + Prev Day Bearish': lambda r: r['prev_trend'] == 'Bearish',
    }

    # Get baseline stats for edge calculation
    baseline_advs = [r['max_adverse_eod'] for r in all_results]
    baseline_stats = safety_stats(baseline_advs)
    baseline_lt05 = baseline_stats.get('lt_0.5', 0)

    # ─── TABLE 1: Entry Filter Comparison ───
    table1_rows = []
    for fname, ffunc in filters.items():
        matching = [r for r in all_results if ffunc(r)]
        advs = [r['max_adverse_eod'] for r in matching]
        stats = safety_stats(advs)
        if stats:
            edge = stats['lt_0.5'] - baseline_lt05
            table1_rows.append((
                fname,
                stats['days'],
                f"{stats['avg']:.3f}",
                f"{stats['med']:.3f}",
                f"{stats['lt_0.3']:.1f}",
                f"{stats['lt_0.5']:.1f}",
                f"{stats['lt_0.75']:.1f}",
                f"{stats['lt_1.0']:.1f}",
                f"{edge:+.1f}" if fname != '01. BB Only (baseline)' else '-',
            ))

    # Sort by <0.5% safety rate (descending)
    table1_rows.sort(key=lambda x: -float(x[5]))

    print_table(
        f"TABLE 1: Entry Filter Comparison - {sym_label} (ranked by <0.5% safety)",
        ['Filter', 'Days', 'AvgAdv%', 'MedAdv%', '<0.3%', '<0.5%', '<0.75%', '<1.0%', 'Edge'],
        table1_rows,
        [35, 6, 9, 9, 8, 8, 8, 8, 8]
    )

    # ─── TABLE 2: Exit Time Analysis (BB cooloff baseline) ───
    table2_rows = []
    for exit_time in EXIT_TIMES:
        advs = []
        for r in all_results:
            if exit_time in r['exit_data']:
                advs.append(r['exit_data'][exit_time]['max_adverse'])
        stats = safety_stats(advs)
        if stats:
            table2_rows.append((
                exit_time,
                stats['days'],
                f"{stats['avg']:.3f}",
                f"{stats['med']:.3f}",
                f"{stats['lt_0.3']:.1f}",
                f"{stats['lt_0.5']:.1f}",
                f"{stats['lt_0.75']:.1f}",
                f"{stats['lt_1.0']:.1f}",
            ))
    # Add EOD row
    eod_stats = safety_stats([r['max_adverse_eod'] for r in all_results])
    if eod_stats:
        table2_rows.append((
            'EOD(full)',
            eod_stats['days'],
            f"{eod_stats['avg']:.3f}",
            f"{eod_stats['med']:.3f}",
            f"{eod_stats['lt_0.3']:.1f}",
            f"{eod_stats['lt_0.5']:.1f}",
            f"{eod_stats['lt_0.75']:.1f}",
            f"{eod_stats['lt_1.0']:.1f}",
        ))

    print_table(
        f"TABLE 2: Exit Time Analysis (BB Cooloff Baseline) - {sym_label}",
        ['Exit', 'Days', 'AvgAdv%', 'MedAdv%', '<0.3%', '<0.5%', '<0.75%', '<1.0%'],
        table2_rows,
        [12, 6, 9, 9, 8, 8, 8, 8]
    )

    # ─── TABLE 3: Best Filters x DTE ───
    # Get top 5 filters by <0.5% safety rate (excluding baseline)
    filter_rankings = []
    for fname, ffunc in filters.items():
        if 'baseline' in fname:
            continue
        matching = [r for r in all_results if ffunc(r)]
        advs = [r['max_adverse_eod'] for r in matching]
        stats = safety_stats(advs)
        if stats and stats['days'] >= 10:
            filter_rankings.append((fname, ffunc, stats['lt_0.5'], stats['days']))

    filter_rankings.sort(key=lambda x: -x[2])
    top5_filters = filter_rankings[:5]

    # Get DTE range
    dtes = sorted(set(r['dte'] for r in all_results if r['dte'] >= 0))
    if len(dtes) > 8:
        # Group into buckets
        dte_buckets = [(0, 'DTE=0'), (1, 'DTE=1'), (2, 'DTE=2'), (3, 'DTE=3'), (4, 'DTE=4+')]
    else:
        dte_buckets = [(d, f'DTE={d}') for d in dtes[:6]]

    table3_rows = []
    for fname, ffunc, _, _ in top5_filters:
        row = [fname[:30]]
        for dte_val, dte_label in dte_buckets:
            if dte_label.endswith('+'):
                matching = [r for r in all_results if ffunc(r) and r['dte'] >= dte_val]
            else:
                matching = [r for r in all_results if ffunc(r) and r['dte'] == dte_val]
            advs = [r['max_adverse_eod'] for r in matching]
            stats = safety_stats(advs)
            if stats and stats['days'] >= 3:
                row.append(f"{stats['lt_0.5']:.0f}%({stats['days']})")
            else:
                row.append('-')
        table3_rows.append(tuple(row))

    dte_headers = ['Filter'] + [b[1] for b in dte_buckets]
    dte_widths = [32] + [12] * len(dte_buckets)

    print_table(
        f"TABLE 3: Top 5 Filters x DTE (<0.5% safety rate) - {sym_label}",
        dte_headers,
        table3_rows,
        dte_widths
    )

    # ─── TABLE 4: Top 3 Filters x Exit Time ───
    top3_filters = filter_rankings[:3]

    table4_rows = []
    for fname, ffunc, _, _ in top3_filters:
        row = [fname[:30]]
        for exit_time in EXIT_TIMES:
            matching = [r for r in all_results if ffunc(r) and exit_time in r['exit_data']]
            advs = [r['exit_data'][exit_time]['max_adverse'] for r in matching]
            stats = safety_stats(advs)
            if stats and stats['days'] >= 5:
                row.append(f"{stats['lt_0.5']:.0f}%({stats['days']})")
            else:
                row.append('-')
        row.append(f"{safety_stats([r['max_adverse_eod'] for r in all_results if ffunc(r)])['lt_0.5']:.0f}%")
        table4_rows.append(tuple(row))

    print_table(
        f"TABLE 4: Top 3 Filters x Exit Time (<0.5% safety rate) - {sym_label}",
        ['Filter'] + EXIT_TIMES + ['EOD'],
        table4_rows,
        [32] + [10] * (len(EXIT_TIMES) + 1)
    )

    # ─── TABLE 5: Combined Best ───
    # Test all combos of top3 filter + best exit + best DTE
    print(f"\n{'=' * 80}", flush=True)
    print(f"  TABLE 5: Best Combined Configurations - {sym_label}", flush=True)
    print(f"{'=' * 80}", flush=True)

    best_configs = []
    for fname, ffunc, _, _ in top3_filters:
        for exit_time in EXIT_TIMES:
            for dte_val, dte_label in dte_buckets:
                if dte_label.endswith('+'):
                    matching = [r for r in all_results if ffunc(r) and r['dte'] >= dte_val
                               and exit_time in r['exit_data']]
                else:
                    matching = [r for r in all_results if ffunc(r) and r['dte'] == dte_val
                               and exit_time in r['exit_data']]

                if len(matching) < 5:
                    continue

                advs = [r['exit_data'][exit_time]['max_adverse'] for r in matching]
                stats = safety_stats(advs)
                if stats:
                    best_configs.append((
                        f"{fname[:25]} | {exit_time} | {dte_label}",
                        stats['days'],
                        stats['lt_0.5'],
                        stats['avg'],
                        stats['med'],
                        stats['lt_1.0'],
                    ))

    best_configs.sort(key=lambda x: -x[2])

    print(f"{'Config':<55} {'Days':>5} {'<0.5%':>7} {'Avg':>7} {'Med':>7} {'<1.0%':>7}", flush=True)
    print('-' * 90, flush=True)
    for cfg in best_configs[:20]:
        print(f"{cfg[0]:<55} {cfg[1]:>5} {cfg[2]:>6.1f}% {cfg[3]:>6.3f} {cfg[4]:>6.3f} {cfg[5]:>6.1f}%", flush=True)

    # ─── TABLE 6: Filter Combinations (stacking) ───
    print(f"\n{'=' * 80}", flush=True)
    print(f"  TABLE 6: Stacked Filter Combinations - {sym_label}", flush=True)
    print(f"{'=' * 80}", flush=True)

    if len(top3_filters) >= 2:
        combo_tests = []

        # Pairs
        for i in range(min(3, len(top3_filters))):
            for j in range(i + 1, min(3, len(top3_filters))):
                fi_name, fi_func, _, _ = top3_filters[i]
                fj_name, fj_func, _, _ = top3_filters[j]
                combined = lambda r, a=fi_func, b=fj_func: a(r) and b(r)
                matching = [r for r in all_results if combined(r)]
                advs = [r['max_adverse_eod'] for r in matching]
                stats = safety_stats(advs)
                if stats and stats['days'] >= 5:
                    combo_tests.append((
                        f"{fi_name[:20]} + {fj_name[:20]}",
                        stats['days'],
                        stats['lt_0.5'],
                        stats['avg'],
                        stats['lt_1.0'],
                    ))

        # Triple
        if len(top3_filters) >= 3:
            f1, f2, f3 = top3_filters[0][1], top3_filters[1][1], top3_filters[2][1]
            combined3 = lambda r: f1(r) and f2(r) and f3(r)
            matching = [r for r in all_results if combined3(r)]
            advs = [r['max_adverse_eod'] for r in matching]
            stats = safety_stats(advs)
            if stats and stats['days'] >= 3:
                combo_tests.append((
                    f"Top3 combined",
                    stats['days'],
                    stats['lt_0.5'],
                    stats['avg'],
                    stats['lt_1.0'],
                ))

        # Add baseline for comparison
        bl = safety_stats([r['max_adverse_eod'] for r in all_results])
        combo_tests.insert(0, ('BB Only (baseline)', bl['days'], bl['lt_0.5'], bl['avg'], bl['lt_1.0']))

        print(f"{'Combo':<50} {'Days':>5} {'<0.5%':>7} {'Avg':>7} {'<1.0%':>7}", flush=True)
        print('-' * 80, flush=True)
        for ct in combo_tests:
            print(f"{ct[0]:<50} {ct[1]:>5} {ct[2]:>6.1f}% {ct[3]:>6.3f} {ct[4]:>6.1f}%", flush=True)

    return filter_rankings


def save_csv_summary(all_results_by_sym, all_combined):
    """Save a CSV summary of all days with filters and outcomes."""
    rows = []
    for r in all_combined:
        row = {
            'date': r['date'],
            'symbol': r['symbol'],
            'cooloff_time': r['cooloff_time'],
            'cooloff_price': f"{r['cooloff_price']:.2f}",
            'max_adverse_eod': f"{r['max_adverse_eod']:.4f}",
            'dte': r['dte'],
            'cpr_width': f"{r['cpr_width']:.4f}",
            'cpr_position': r['cpr_position'],
            'prev_range_pct': f"{r['prev_range_pct']:.4f}",
            'prev_trend': r['prev_trend'],
            'inside_day': r['inside_day'],
            'no_prev_day_break': r['no_prev_day_break'],
            'gap_pct': f"{r['gap_pct']:.4f}",
            'gap_filled': r['gap_filled'],
            'or_breakout': r['or_breakout'],
            'narrow_cooloff_candles': r['narrow_cooloff_candles'],
        }
        # Add exit time max adverse
        for et in EXIT_TIMES:
            if et in r['exit_data']:
                row[f'max_adv_{et.replace(":", "")}'] = f"{r['exit_data'][et]['max_adverse']:.4f}"
            else:
                row[f'max_adv_{et.replace(":", "")}'] = ''
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV summary saved: {OUTPUT_CSV} ({len(rows)} rows)", flush=True)


# ─── Main ───


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  NEUTRAL ENTRY OPTIMIZATION: NIFTY50 & BANKNIFTY", flush=True)
    print("  BB Bandwidth Cooloff + Entry Filters + Exit Times + DTE", flush=True)
    print("=" * 80, flush=True)

    # Step 1: Load data
    print("\n[STEP 1] Loading data...", flush=True)
    data_5min, daily_data = load_data()

    # Step 2: Build expiry dates and compute analysis
    print("\n[STEP 2] Computing BB cooloff and filters for each day...", flush=True)

    all_results_by_sym = {}
    all_combined = []

    for sym in SYMBOLS:
        print(f"\n  Processing {sym}...", flush=True)
        df_5min = data_5min[sym]
        daily_df = daily_data[sym]

        # Trading dates from 5-min data
        trading_dates_dt = sorted(df_5min['trade_date'].unique())
        trading_dates = [d for d in trading_dates_dt]

        # Build expiry dates
        expiry_dates = build_expiry_dates(trading_dates, sym)
        print(f"    Built {len(expiry_dates)} expiry dates", flush=True)

        # Compute analysis
        results = compute_day_analysis(sym, df_5min, daily_df, trading_dates, expiry_dates)
        all_results_by_sym[sym] = results
        all_combined.extend(results)

    # Step 3: Print cooloff time distribution
    print(f"\n{'=' * 80}", flush=True)
    print("  BB COOLOFF TIME DISTRIBUTION", flush=True)
    print(f"{'=' * 80}", flush=True)
    for sym in SYMBOLS:
        results = all_results_by_sym[sym]
        times_list = [r['cooloff_time'] for r in results]
        if times_list:
            # Bucket by hour
            hour_counts = defaultdict(int)
            for t in times_list:
                h = t[:2]
                hour_counts[h] += 1
            print(f"\n  {sym} ({len(results)} days with cooloff):", flush=True)
            for h in sorted(hour_counts.keys()):
                bar = '#' * (hour_counts[h] // 2)
                print(f"    {h}:xx  {hour_counts[h]:>4} days  {bar}", flush=True)

            median_time = sorted(times_list)[len(times_list) // 2]
            print(f"    Median cooloff: {median_time}", flush=True)

    # Step 4: Run analysis per symbol and combined
    print("\n[STEP 3] Running filter analysis...", flush=True)

    for sym in SYMBOLS:
        results = all_results_by_sym[sym]
        if results:
            run_filter_analysis(results, sym)

    if all_combined:
        print("\n" + "=" * 80, flush=True)
        print("  >>> COMBINED ANALYSIS (NIFTY50 + BANKNIFTY) <<<", flush=True)
        run_filter_analysis(all_combined, "COMBINED")

    # Step 5: Save CSV
    save_csv_summary(all_results_by_sym, all_combined)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
