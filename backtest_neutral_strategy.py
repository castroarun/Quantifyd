"""
Neutral Strategy Backtest: Sell Straddles/Strangles on NIFTY50 & BANKNIFTY
Uses Black-Scholes simulated premiums with BB Cooloff entry logic.
Vectorized BS pricing for performance.
"""

import sqlite3
import math
import csv
import os
import sys
import time
from datetime import datetime, timedelta, time as dt_time
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_neutral_trades.csv')

SYMBOLS = ['NIFTY50', 'BANKNIFTY']
LOT_SIZES = {'NIFTY50': 75, 'BANKNIFTY': 30}
STRIKE_INTERVALS = {'NIFTY50': 50, 'BANKNIFTY': 100}
RISK_FREE_RATE = 0.065
INITIAL_CAPITAL = 1_000_000  # 10 Lakhs
NUM_LOTS = 1

BB_PERIOD = 20
BB_STD = 2.0
BB_COOLOFF_THRESHOLD = 0.50
CPR_PERIOD = 20


# ---------------------------------------------------------------------------
# Vectorized Black-Scholes
# ---------------------------------------------------------------------------
def bs_price_vec(S, K, T, r, sigma, option_type='CE'):
    """Vectorized Black-Scholes. S, K, T can be numpy arrays."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    # Handle T <= 0 (intrinsic value only)
    T_safe = np.maximum(T, 1e-10)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_safe) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == 'CE':
        price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
        # Where T <= 0, use intrinsic
        expired = T <= 1e-10
        if np.any(expired):
            price = np.where(expired, np.maximum(0.0, S - K), price)
    else:
        price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * norm.cdf(-d1)
        expired = T <= 1e-10
        if np.any(expired):
            price = np.where(expired, np.maximum(0.0, K - S), price)

    return price


def bs_price_scalar(S, K, T, r, sigma, option_type='CE'):
    """Scalar Black-Scholes for single entry pricing."""
    if T <= 1e-10:
        return max(0.0, S - K) if option_type == 'CE' else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'CE':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_atm_strike(spot, interval):
    return round(spot / interval) * interval


def get_otm_strikes(spot, interval, otm_pct):
    if otm_pct <= 0:
        atm = get_atm_strike(spot, interval)
        return atm, atm
    ce_raw = spot * (1 + otm_pct / 100.0)
    pe_raw = spot * (1 - otm_pct / 100.0)
    ce_strike = math.ceil(ce_raw / interval) * interval
    pe_strike = math.floor(pe_raw / interval) * interval
    return ce_strike, pe_strike


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    print("Loading data from database...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    data_5m = {}
    data_day = {}

    for sym in SYMBOLS:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close FROM market_data_unified "
            "WHERE symbol=? AND timeframe='5minute' ORDER BY date",
            conn, params=(sym,)
        )
        df['date'] = pd.to_datetime(df['date'])
        df['trade_date'] = df['date'].dt.date
        df['time'] = df['date'].dt.time
        # Pre-compute time as seconds since midnight for fast comparison
        df['time_sec'] = df['date'].dt.hour * 3600 + df['date'].dt.minute * 60 + df['date'].dt.second
        data_5m[sym] = df
        print(f"  {sym} 5min: {len(df)} candles, {df['trade_date'].nunique()} days", flush=True)

        df_d = pd.read_sql_query(
            "SELECT date, open, high, low, close FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date",
            conn, params=(sym,)
        )
        df_d['date'] = pd.to_datetime(df_d['date']).dt.date
        data_day[sym] = df_d
        print(f"  {sym} day: {len(df_d)} rows", flush=True)

    conn.close()
    return data_5m, data_day


# ---------------------------------------------------------------------------
# Expiry calendar
# ---------------------------------------------------------------------------
def build_expiry_calendar(trading_days):
    from datetime import date
    td_sorted = sorted(trading_days)
    expiries = {'NIFTY50': [], 'BANKNIFTY': []}
    nifty_switch = date(2024, 11, 25)
    bn_switch = date(2024, 11, 13)

    for d in td_sorted:
        if d < nifty_switch:
            if d.weekday() == 3:
                expiries['NIFTY50'].append(d)
        else:
            if d.weekday() == 1:
                expiries['NIFTY50'].append(d)

    for d in td_sorted:
        if d < bn_switch:
            if d.weekday() == 2:
                expiries['BANKNIFTY'].append(d)
        else:
            if d.weekday() == 3:
                next_week = d + timedelta(days=7)
                if next_week.month != d.month:
                    expiries['BANKNIFTY'].append(d)

    return expiries


def get_dte(trade_date, expiry_list, trading_days_set):
    for exp in expiry_list:
        if exp >= trade_date:
            dte = sum(1 for d in trading_days_set if trade_date < d <= exp)
            return max(dte, 1)
    return 5


# ---------------------------------------------------------------------------
# BB Cooloff (vectorized per day)
# ---------------------------------------------------------------------------
def compute_bb_cooloff(closes):
    """Returns cooloff bar index or None. Fully numpy."""
    n = len(closes)
    if n < BB_PERIOD + 2:
        return None

    # Rolling mean and std
    cum = np.cumsum(closes)
    cum_sq = np.cumsum(closes ** 2)

    # For indices BB_PERIOD-1 to n-1
    start = BB_PERIOD
    end_idx = n
    sums = cum[start - 1:end_idx] - np.concatenate([[0], cum[:end_idx - start]])
    sum_sqs = cum_sq[start - 1:end_idx] - np.concatenate([[0], cum_sq[:end_idx - start]])
    means = sums / BB_PERIOD
    variances = sum_sqs / BB_PERIOD - means ** 2
    variances = np.maximum(variances, 0)
    stds = np.sqrt(variances)

    # Bandwidth
    safe_means = np.where(means == 0, 1, means)
    bw = 2 * BB_STD * stds / safe_means * 100

    if len(bw) == 0:
        return None

    peak_bw = np.max(bw)
    if peak_bw == 0:
        return None

    peak_pos = np.argmax(bw)
    threshold = peak_bw * BB_COOLOFF_THRESHOLD

    # Find first position after peak where bw < threshold
    after_peak = bw[peak_pos + 1:]
    if len(after_peak) == 0:
        return None

    below = np.where(after_peak < threshold)[0]
    if len(below) == 0:
        return None

    # Convert back to original candle index
    cooloff_pos_in_bw = peak_pos + 1 + below[0]
    cooloff_candle_idx = (BB_PERIOD - 1) + cooloff_pos_in_bw

    return int(cooloff_candle_idx)


def compute_cpr(prev_high, prev_low, prev_close):
    pivot = (prev_high + prev_low + prev_close) / 3.0
    tc = 2 * pivot - (prev_high + prev_low) / 2.0
    bc = (prev_high + prev_low) / 2.0
    return pivot, tc, bc


# ---------------------------------------------------------------------------
# Precompute per-day data for fast backtesting
# ---------------------------------------------------------------------------
def precompute_day_info(data_5m, data_day, expiry_calendar, trading_days_set, symbol):
    """Precompute all per-day info needed for backtesting.
    Returns list of dicts with day-level data including cooloff bar, filters, etc.
    """
    df = data_5m[symbol]
    df_day = data_day[symbol]
    expiry_list = expiry_calendar[symbol]
    interval = STRIKE_INTERVALS[symbol]

    grouped = df.groupby('trade_date')
    trade_dates = sorted(grouped.groups.keys())

    # Build daily data lookup
    day_lookup = {}
    for _, row in df_day.iterrows():
        day_lookup[row['date']] = row

    # Compute rolling 20-day median range %
    day_dates_sorted = sorted(day_lookup.keys())
    range_pcts = []
    day_range_pct = {}
    for d in day_dates_sorted:
        r = day_lookup[d]
        rng = (r['high'] - r['low']) / r['close'] * 100 if r['close'] > 0 else 0
        range_pcts.append(rng)
        day_range_pct[d] = rng

    day_median_range = {}
    for i, d in enumerate(day_dates_sorted):
        if i < CPR_PERIOD:
            day_median_range[d] = np.median(range_pcts[:i + 1]) if i > 0 else range_pcts[0]
        else:
            day_median_range[d] = np.median(range_pcts[i - CPR_PERIOD:i])

    days_info = []

    for td_idx, td in enumerate(trade_dates):
        day_df = grouped.get_group(td).reset_index(drop=True)
        closes = day_df['close'].values.astype(np.float64)
        n = len(day_df)
        if n < BB_PERIOD + 5:
            continue

        # BB Cooloff
        cooloff_idx = compute_bb_cooloff(closes)
        if cooloff_idx is None:
            continue

        entry_time_sec = day_df.iloc[cooloff_idx]['time_sec']
        # Skip if before 9:30
        if entry_time_sec < 9 * 3600 + 30 * 60:
            continue

        spot_at_entry = closes[cooloff_idx]
        entry_dt = day_df.iloc[cooloff_idx]['date']

        # Gap check
        gap_pct = None
        if td_idx > 0:
            prev_td = trade_dates[td_idx - 1]
            if prev_td in day_lookup:
                prev_close = day_lookup[prev_td]['close']
                today_open = day_df.iloc[0]['open']
                if prev_close > 0:
                    gap_pct = (today_open - prev_close) / prev_close * 100

        # Narrow prev day check
        narrow_prev = None
        if td_idx > 0:
            prev_td = trade_dates[td_idx - 1]
            if prev_td in day_range_pct and prev_td in day_median_range:
                narrow_prev = day_range_pct[prev_td] < day_median_range[prev_td]

        # CPR check (spot > TC)
        cpr_above = None
        if td_idx > 0:
            prev_td = trade_dates[td_idx - 1]
            if prev_td in day_lookup:
                pr = day_lookup[prev_td]
                _, tc, _ = compute_cpr(pr['high'], pr['low'], pr['close'])
                cpr_above = spot_at_entry > tc

        # DTE
        dte = get_dte(td, expiry_list, trading_days_set)

        # Store bar-level data for exit scanning (all bars from cooloff+1 to end)
        exit_start = cooloff_idx + 1
        if exit_start >= n:
            continue

        bar_closes = closes[exit_start:]
        bar_time_secs = day_df['time_sec'].values[exit_start:]
        bar_datetimes = day_df['date'].values[exit_start:]
        bar_times = day_df['time'].values[exit_start:]

        # Compute T for each exit bar
        # hours_elapsed from entry for each bar
        entry_ts = entry_dt.timestamp()
        bar_ts = pd.to_datetime(bar_datetimes).to_series().apply(lambda x: x.timestamp()).values
        hours_elapsed = (bar_ts - entry_ts) / 3600.0
        T_bars = np.maximum((dte - hours_elapsed / 6.25) / 252.0, 1e-10)

        days_info.append({
            'trade_date': td,
            'cooloff_idx': cooloff_idx,
            'spot_at_entry': spot_at_entry,
            'entry_time': day_df.iloc[cooloff_idx]['time'],
            'entry_time_sec': entry_time_sec,
            'entry_dt': entry_dt,
            'gap_pct': gap_pct,
            'narrow_prev': narrow_prev,
            'cpr_above': cpr_above,
            'dte': dte,
            'bar_closes': bar_closes,
            'bar_time_secs': bar_time_secs,
            'bar_times': bar_times,
            'T_bars': T_bars,
        })

    return days_info


# ---------------------------------------------------------------------------
# Main backtest logic (vectorized exit scanning)
# ---------------------------------------------------------------------------
def run_backtest_fast(days_info, symbol, iv, otm_pct=0.0,
                      use_cpr=False, use_narrow=False, use_no_gap=False,
                      exit_time_str='13:00', sl_multiple=None, tp_fraction=None,
                      label=''):
    """Run backtest using precomputed day info. Vectorized BS for exit bars."""

    interval = STRIKE_INTERVALS[symbol]
    lot_size = LOT_SIZES[symbol]
    exit_h, exit_m = map(int, exit_time_str.split(':'))
    exit_time_sec = exit_h * 3600 + exit_m * 60

    trades = []

    for info in days_info:
        # Apply filters
        if use_no_gap and info['gap_pct'] is not None and info['gap_pct'] < -0.1:
            continue
        if use_narrow and (info['narrow_prev'] is None or not info['narrow_prev']):
            continue
        if info['entry_time_sec'] >= exit_time_sec:
            continue
        if use_cpr and (info['cpr_above'] is None or not info['cpr_above']):
            continue

        spot = info['spot_at_entry']
        dte = info['dte']
        T_entry = dte / 252.0

        ce_strike, pe_strike = get_otm_strikes(spot, interval, otm_pct)

        ce_prem_entry = bs_price_scalar(spot, ce_strike, T_entry, RISK_FREE_RATE, iv, 'CE')
        pe_prem_entry = bs_price_scalar(spot, pe_strike, T_entry, RISK_FREE_RATE, iv, 'PE')
        entry_premium = ce_prem_entry + pe_prem_entry

        if entry_premium < 1.0:
            continue

        bar_closes = info['bar_closes']
        bar_time_secs = info['bar_time_secs']
        bar_times = info['bar_times']
        T_bars = info['T_bars']
        n_bars = len(bar_closes)

        if n_bars == 0:
            continue

        # Vectorized BS pricing for all exit bars at once
        ce_prems = bs_price_vec(bar_closes, ce_strike, T_bars, RISK_FREE_RATE, iv, 'CE')
        pe_prems = bs_price_vec(bar_closes, pe_strike, T_bars, RISK_FREE_RATE, iv, 'PE')
        total_prems = ce_prems + pe_prems

        # Find exit bar
        exit_idx = None
        exit_reason = 'eod'

        if sl_multiple is not None or tp_fraction is not None:
            # Need to check SL/TP before time exit, bar by bar order matters
            for i in range(n_bars):
                # SL check
                if sl_multiple is not None and total_prems[i] > sl_multiple * entry_premium:
                    exit_idx = i
                    exit_reason = f'SL_{sl_multiple}x'
                    break
                # TP check
                if tp_fraction is not None and total_prems[i] < tp_fraction * entry_premium:
                    exit_idx = i
                    exit_reason = f'TP_{tp_fraction}'
                    break
                # Time exit check
                if bar_time_secs[i] >= exit_time_sec:
                    exit_idx = i
                    exit_reason = 'time'
                    break
        else:
            # No SL/TP, just find first bar at or after exit time
            time_hits = np.where(bar_time_secs >= exit_time_sec)[0]
            if len(time_hits) > 0:
                exit_idx = time_hits[0]
                exit_reason = 'time'

        if exit_idx is None:
            exit_idx = n_bars - 1
            exit_reason = 'eod'

        exit_premium = float(total_prems[exit_idx])
        exit_time_val = bar_times[exit_idx]

        pnl_per_lot = (entry_premium - exit_premium) * lot_size
        total_pnl = pnl_per_lot * NUM_LOTS

        hold_minutes = (bar_time_secs[exit_idx] - info['entry_time_sec']) / 60.0

        trades.append({
            'date': str(info['trade_date']),
            'symbol': symbol,
            'entry_time': str(info['entry_time']),
            'exit_time': str(exit_time_val),
            'exit_reason': exit_reason,
            'spot_at_entry': round(spot, 2),
            'ce_strike': ce_strike,
            'pe_strike': pe_strike,
            'entry_premium': round(entry_premium, 2),
            'exit_premium': round(exit_premium, 2),
            'pnl': round(total_pnl, 2),
            'lots': NUM_LOTS,
            'dte': dte,
            'filter_used': label,
            'hold_minutes': round(hold_minutes, 1),
        })

    metrics = compute_metrics(trades, label)
    return metrics, trades


def compute_metrics(trades, label):
    if not trades:
        return {
            'label': label, 'trades': 0, 'win_rate': 0, 'total_pnl': 0,
            'avg_pnl': 0, 'profit_factor': 0, 'max_dd': 0, 'sharpe': 0,
            'avg_entry_prem': 0, 'avg_exit_prem': 0, 'avg_hold_min': 0
        }

    n = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl = total_pnl / n

    gross_wins = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else (999.0 if gross_wins > 0 else 0)

    cum_pnl = np.cumsum([t['pnl'] for t in trades])
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0

    pnls = np.array([t['pnl'] for t in trades])
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = float((np.mean(pnls) / np.std(pnls)) * math.sqrt(252))
    else:
        sharpe = 0

    avg_entry = float(np.mean([t['entry_premium'] for t in trades]))
    avg_exit = float(np.mean([t['exit_premium'] for t in trades]))
    avg_hold = float(np.mean([t['hold_minutes'] for t in trades]))

    return {
        'label': label,
        'trades': n,
        'win_rate': round(wins / n * 100, 1),
        'total_pnl': round(total_pnl, 0),
        'avg_pnl': round(avg_pnl, 0),
        'profit_factor': round(pf, 2),
        'max_dd': round(max_dd, 0),
        'sharpe': round(sharpe, 2),
        'avg_entry_prem': round(avg_entry, 1),
        'avg_exit_prem': round(avg_exit, 1),
        'avg_hold_min': round(avg_hold, 0),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_table(title, rows, columns=None):
    if not rows:
        print(f"\n{title}: No results.\n")
        return

    if columns is None:
        columns = list(rows[0].keys())

    widths = {}
    for col in columns:
        widths[col] = max(len(str(col)), max(len(str(r.get(col, ''))) for r in rows))

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = ' | '.join(str(col).rjust(widths[col]) for col in columns)
    print(header)
    print('-' * len(header))
    for r in rows:
        line = ' | '.join(str(r.get(col, '')).rjust(widths[col]) for col in columns)
        print(line)
    print()


def pick_best(results, metric='total_pnl'):
    if not results:
        return None
    return max(results, key=lambda r: r[metric])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    data_5m, data_day = load_data()

    all_trading_days = set()
    for sym in SYMBOLS:
        all_trading_days.update(data_5m[sym]['trade_date'].unique())
    trading_days_set = all_trading_days
    expiry_calendar = build_expiry_calendar(sorted(all_trading_days))

    print(f"\nExpiry dates: NIFTY50={len(expiry_calendar['NIFTY50'])}, "
          f"BANKNIFTY={len(expiry_calendar['BANKNIFTY'])}", flush=True)

    all_trades = []
    final_best = {}

    cols = ['label', 'trades', 'win_rate', 'total_pnl', 'avg_pnl',
            'profit_factor', 'max_dd', 'sharpe', 'avg_entry_prem', 'avg_exit_prem', 'avg_hold_min']

    for symbol in SYMBOLS:
        print(f"\n{'#' * 80}")
        print(f"  BACKTESTING: {symbol}")
        print(f"{'#' * 80}", flush=True)

        # Precompute day info once per symbol
        print(f"  Precomputing day info...", end='', flush=True)
        t_pre = time.time()
        days_info = precompute_day_info(data_5m, data_day, expiry_calendar,
                                         trading_days_set, symbol)
        print(f" {time.time()-t_pre:.1f}s ({len(days_info)} signal days)", flush=True)

        # ==================================================================
        # PHASE 1: Filter combinations
        # ==================================================================
        print("\n>>> PHASE 1: Filter Combinations (exit=13:00, ATM, IV=0.15)", flush=True)
        phase1_configs = [
            ('BB_Only',        False, False, False),
            ('BB+CPR',         True,  False, False),
            ('BB+Narrow',      False, True,  False),
            ('BB+CPR+Narrow',  True,  True,  False),
            ('BB+NoGap',       False, False, True),
            ('BB+CPR+Narrow+NoGap', True, True, True),
        ]

        phase1_results = []
        for lbl, cpr, narrow, nogap in phase1_configs:
            full_label = f"{symbol}_{lbl}"
            print(f"  {full_label}...", end='', flush=True)
            t1 = time.time()
            m, tr = run_backtest_fast(days_info, symbol, iv=0.15, otm_pct=0.0,
                                       use_cpr=cpr, use_narrow=narrow, use_no_gap=nogap,
                                       exit_time_str='13:00', label=full_label)
            print(f" {time.time()-t1:.1f}s | {m['trades']} trades PnL={m['total_pnl']}", flush=True)
            phase1_results.append(m)
            all_trades.extend(tr)

        print_table(f"{symbol} Phase 1: Filter Combinations", phase1_results, cols)
        best_p1 = pick_best(phase1_results)
        best_p1_label = best_p1['label'].replace(f"{symbol}_", "")
        best_cpr = 'CPR' in best_p1_label
        best_narrow = 'Narrow' in best_p1_label
        best_nogap = 'NoGap' in best_p1_label
        print(f"  -> Best: {best_p1_label} (PnL={best_p1['total_pnl']})", flush=True)

        # ==================================================================
        # PHASE 2: Exit Time
        # ==================================================================
        print("\n>>> PHASE 2: Exit Time", flush=True)
        phase2_results = []
        for et in ['12:00', '12:30', '13:00', '14:00', '15:20']:
            full_label = f"{symbol}_{best_p1_label}_Exit{et.replace(':','')}"
            print(f"  {full_label}...", end='', flush=True)
            t1 = time.time()
            m, tr = run_backtest_fast(days_info, symbol, iv=0.15, otm_pct=0.0,
                                       use_cpr=best_cpr, use_narrow=best_narrow, use_no_gap=best_nogap,
                                       exit_time_str=et, label=full_label)
            print(f" {time.time()-t1:.1f}s | {m['trades']} trades PnL={m['total_pnl']}", flush=True)
            phase2_results.append(m)
            all_trades.extend(tr)

        print_table(f"{symbol} Phase 2: Exit Time", phase2_results, cols)
        best_p2 = pick_best(phase2_results)
        best_exit = '13:00'
        for et in ['12:00', '12:30', '13:00', '14:00', '15:20']:
            if f"Exit{et.replace(':','')}" in best_p2['label']:
                best_exit = et
                break
        print(f"  -> Best exit: {best_exit} (PnL={best_p2['total_pnl']})", flush=True)

        # ==================================================================
        # PHASE 3: Strike Distance
        # ==================================================================
        print("\n>>> PHASE 3: Strike Distance (OTM %)", flush=True)
        phase3_results = []
        for otm in [0.0, 0.3, 0.5, 0.8, 1.0]:
            otm_label = f"OTM{otm}" if otm > 0 else "ATM"
            full_label = f"{symbol}_{best_exit.replace(':','')}_{otm_label}"
            print(f"  {full_label}...", end='', flush=True)
            t1 = time.time()
            m, tr = run_backtest_fast(days_info, symbol, iv=0.15, otm_pct=otm,
                                       use_cpr=best_cpr, use_narrow=best_narrow, use_no_gap=best_nogap,
                                       exit_time_str=best_exit, label=full_label)
            print(f" {time.time()-t1:.1f}s | {m['trades']} trades PnL={m['total_pnl']}", flush=True)
            phase3_results.append(m)
            all_trades.extend(tr)

        print_table(f"{symbol} Phase 3: Strike Distance", phase3_results, cols)
        best_p3 = pick_best(phase3_results)
        best_otm = 0.0
        for otm in [1.0, 0.8, 0.5, 0.3]:
            if f"OTM{otm}" in best_p3['label']:
                best_otm = otm
                break
        print(f"  -> Best OTM: {best_otm}% (PnL={best_p3['total_pnl']})", flush=True)

        # ==================================================================
        # PHASE 4: IV Sensitivity
        # ==================================================================
        print("\n>>> PHASE 4: IV Sensitivity", flush=True)
        phase4_results = []
        for iv_val in [0.12, 0.13, 0.15, 0.18, 0.20]:
            full_label = f"{symbol}_IV{iv_val}"
            print(f"  {full_label}...", end='', flush=True)
            t1 = time.time()
            m, tr = run_backtest_fast(days_info, symbol, iv=iv_val, otm_pct=best_otm,
                                       use_cpr=best_cpr, use_narrow=best_narrow, use_no_gap=best_nogap,
                                       exit_time_str=best_exit, label=full_label)
            print(f" {time.time()-t1:.1f}s | {m['trades']} trades PnL={m['total_pnl']}", flush=True)
            phase4_results.append(m)
            all_trades.extend(tr)

        print_table(f"{symbol} Phase 4: IV Sensitivity", phase4_results, cols)
        best_p4 = pick_best(phase4_results)
        best_iv = 0.15
        for iv_val in [0.12, 0.13, 0.15, 0.18, 0.20]:
            if f"IV{iv_val}" in best_p4['label']:
                best_iv = iv_val
                break
        print(f"  -> Best IV: {best_iv} (PnL={best_p4['total_pnl']})", flush=True)

        # ==================================================================
        # PHASE 5: Stop Loss
        # ==================================================================
        print("\n>>> PHASE 5: Stop Loss", flush=True)
        phase5_results = []
        for sl_label, sl_val in [('NoSL', None), ('SL_2x', 2.0), ('SL_1.5x', 1.5)]:
            full_label = f"{symbol}_{sl_label}"
            print(f"  {full_label}...", end='', flush=True)
            t1 = time.time()
            m, tr = run_backtest_fast(days_info, symbol, iv=best_iv, otm_pct=best_otm,
                                       use_cpr=best_cpr, use_narrow=best_narrow, use_no_gap=best_nogap,
                                       exit_time_str=best_exit, sl_multiple=sl_val, label=full_label)
            print(f" {time.time()-t1:.1f}s | {m['trades']} trades PnL={m['total_pnl']}", flush=True)
            phase5_results.append(m)
            all_trades.extend(tr)

        print_table(f"{symbol} Phase 5: Stop Loss", phase5_results, cols)
        best_p5 = pick_best(phase5_results)
        best_sl = None
        if 'SL_2x' in best_p5['label']:
            best_sl = 2.0
        elif 'SL_1.5x' in best_p5['label']:
            best_sl = 1.5
        print(f"  -> Best SL: {best_sl} (PnL={best_p5['total_pnl']})", flush=True)

        # ==================================================================
        # PHASE 6: Target Profit
        # ==================================================================
        print("\n>>> PHASE 6: Target Profit", flush=True)
        phase6_results = []
        for tp_label, tp_val in [('NoTP', None), ('TP_50pct', 0.50), ('TP_30pct', 0.30)]:
            full_label = f"{symbol}_{tp_label}"
            print(f"  {full_label}...", end='', flush=True)
            t1 = time.time()
            m, tr = run_backtest_fast(days_info, symbol, iv=best_iv, otm_pct=best_otm,
                                       use_cpr=best_cpr, use_narrow=best_narrow, use_no_gap=best_nogap,
                                       exit_time_str=best_exit, sl_multiple=best_sl, tp_fraction=tp_val,
                                       label=full_label)
            print(f" {time.time()-t1:.1f}s | {m['trades']} trades PnL={m['total_pnl']}", flush=True)
            phase6_results.append(m)
            all_trades.extend(tr)

        print_table(f"{symbol} Phase 6: Target Profit", phase6_results, cols)
        best_p6 = pick_best(phase6_results)
        best_tp = None
        if 'TP_50pct' in best_p6['label']:
            best_tp = 0.50
        elif 'TP_30pct' in best_p6['label']:
            best_tp = 0.30
        print(f"  -> Best TP: {best_tp} (PnL={best_p6['total_pnl']})", flush=True)

        final_best[symbol] = {
            'filter': best_p1_label,
            'exit_time': best_exit,
            'otm_pct': best_otm,
            'iv': best_iv,
            'sl': best_sl,
            'tp': best_tp,
            'metrics': best_p6,
        }

    # ==================================================================
    # Save trades to CSV
    # ==================================================================
    csv_cols = ['date', 'symbol', 'entry_time', 'exit_time', 'exit_reason',
                'spot_at_entry', 'ce_strike', 'pe_strike', 'entry_premium',
                'exit_premium', 'pnl', 'lots', 'dte', 'filter_used']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction='ignore')
        writer.writeheader()
        for t in all_trades:
            writer.writerow(t)
    print(f"\nTrades saved: {OUTPUT_CSV} ({len(all_trades)} trades)", flush=True)

    # ==================================================================
    # FINAL OPTIMAL CONFIG SUMMARY
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("  FINAL OPTIMAL CONFIGURATION SUMMARY")
    print(f"{'=' * 80}")
    for sym in SYMBOLS:
        fb = final_best[sym]
        m = fb['metrics']
        print(f"\n  {sym}:")
        print(f"    Filter:     {fb['filter']}")
        print(f"    Exit Time:  {fb['exit_time']}")
        print(f"    OTM %:      {fb['otm_pct']}% ({'Straddle' if fb['otm_pct']==0 else 'Strangle'})")
        print(f"    IV:         {fb['iv']}")
        print(f"    Stop Loss:  {fb['sl'] if fb['sl'] else 'None'}")
        print(f"    Target:     {fb['tp'] if fb['tp'] else 'None'}")
        print(f"    ---")
        print(f"    Trades:     {m['trades']}")
        print(f"    Win Rate:   {m['win_rate']}%")
        print(f"    Total PnL:  Rs {m['total_pnl']:,.0f}")
        print(f"    Avg PnL:    Rs {m['avg_pnl']:,.0f}")
        print(f"    PF:         {m['profit_factor']}")
        print(f"    Max DD:     Rs {m['max_dd']:,.0f}")
        print(f"    Sharpe:     {m['sharpe']}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
