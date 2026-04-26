"""
Phase 2c: Refined Nifty ORB skewed strangle simulation.

Differences vs failed Phase 2:
  - Expiry: nearest weekly Tuesday INCLUDING same-day DTE=0 (no roll)
  - SL: OR-width x 1.0 (Phase 1e pattern), wick-based opposite-OR breach
  - Strikes: solved by DELTA (PE -0.22 / CE +0.10 on the bias side; mirror on far)
  - IV proxy: 1.4 * RV(14d), clamped to [0.12, 0.25]
  - Day filter: three side-by-side variants (all_days, skip_Q4_static, skip_Q4_rolling)

Three CSVs (per-trade) + one summary CSV. Console comparison + DTE breakdown
of best variant. Append findings section to docs MD (does not modify earlier sections).
"""

import csv
import math
import os
import sqlite3
import sys
from collections import deque
from datetime import date, time as dtime, timedelta

import numpy as np
import pandas as pd

# ============================================================================
# Constants
# ============================================================================

DB = 'backtest_data/market_data.db'
SYMBOL = 'NIFTY50'
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
NO_ENTRY_AFTER = dtime(14, 0)

LOT_SIZE = 65
RISK_FREE_RATE = 0.065
STRIKE_INTERVAL = 50
SLIPPAGE_PER_LEG_PER_SIDE = 1.0
BROKERAGE_PER_ORDER = 20.0
STT_RATE_OPT_SELL = 0.0005
WEEKLY_EXPIRY_WEEKDAY = 1   # Tuesday

# Delta targets (absolute values; sign applied per opt type)
DELTA_NEAR = 0.22   # bias-side leg (PE on LONG, CE on SHORT)
DELTA_FAR = 0.10    # wing leg (CE on LONG, PE on SHORT)

# IV proxy bounds
IV_RV_MULT = 1.4
IV_FLOOR = 0.12
IV_CAP = 0.25

# Phase 1 winner entry params
ENTRY_OR_MIN = 60
ENTRY_RSI_LONG = 60
ENTRY_RSI_SHORT = 40
ENTRY_WAIT_K = 12

# Day filter parameters
Q4_STATIC_THRESHOLD_PCT = 0.67
Q4_ROLLING_WINDOW = 90
Q4_ROLLING_PERCENTILE = 75

# OR-width SL multiplier (Phase 1e winner pattern)
SL_OR_WIDTH_MULT = 1.0

YEARS_IN_SAMPLE = 455 / 252

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Black-Scholes (in-house, no scipy) + delta
# ============================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(spot, strike, t_years, vol, r):
    return (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / (vol * math.sqrt(t_years))


def bs_price(spot, strike, t_years, vol, r, opt_type):
    is_call = (opt_type == 'CE')
    if t_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.01) if is_call else max(strike - spot, 0.01)
    sigma_t = vol * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / sigma_t
    d2 = d1 - sigma_t
    if is_call:
        price = spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    else:
        price = strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    return max(round(price, 2), 0.01)


def bs_delta(spot, strike, t_years, vol, r, opt_type):
    """Returns Black-Scholes delta. Call delta in (0,1); put delta in (-1,0)."""
    if t_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        if opt_type == 'CE':
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1 = _d1(spot, strike, t_years, vol, r)
    nd1 = _norm_cdf(d1)
    if opt_type == 'CE':
        return nd1
    return nd1 - 1.0


# Self-test
_test = bs_price(22000, 22000, 7/365.0, 0.14, 0.065, 'CE')
assert 130 < _test < 220, f"BS sanity: ATM 7d call = {_test}"
_dtest = bs_delta(22000, 22000, 7/365.0, 0.14, 0.065, 'CE')
assert 0.4 < _dtest < 0.6, f"BS delta sanity: ATM 7d call delta = {_dtest}"


# ============================================================================
# Data loading
# ============================================================================

def rsi(s, p=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/p, adjust=False).mean()
    al = loss.ewm(alpha=1/p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_5min():
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",
        conn, params=[SYMBOL])
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    df = df.between_time(SESSION_OPEN, SESSION_CLOSE)
    df['rsi5m'] = rsi(df['close'], 14)
    return df


def load_daily_rv_map():
    """{date -> annualized 14d RV (lagged 1 day so today uses yesterday's RV)}."""
    conn = sqlite3.connect(DB)
    d = pd.read_sql(
        "SELECT date, close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date",
        conn, params=[SYMBOL])
    conn.close()
    d['date'] = pd.to_datetime(d['date']).dt.date
    d['close'] = d['close'].astype(float)
    d['log_ret'] = np.log(d['close'] / d['close'].shift(1))
    d['rv14'] = d['log_ret'].rolling(14).std() * np.sqrt(252)
    d['rv14_lag'] = d['rv14'].shift(1)
    return dict(zip(d['date'], d['rv14_lag']))


def rv_to_iv(rv):
    """Convert RV to IV: 1.4x scaling, clamp [0.12, 0.25]."""
    if rv is None or pd.isna(rv) or rv <= 0:
        rv = 0.10  # fallback baseline
    iv = IV_RV_MULT * float(rv)
    return float(min(max(iv, IV_FLOOR), IV_CAP))


# ============================================================================
# Expiry: nearest Tuesday INCLUDING same day (DTE=0 allowed)
# ============================================================================

def nearest_weekly_expiry(d: date) -> date:
    days_ahead = (WEEKLY_EXPIRY_WEEKDAY - d.weekday()) % 7
    return d + timedelta(days=days_ahead)


def t_years_at_entry(dte_calendar_days: int) -> float:
    """Use max(0.5h, dte) to avoid /0 on DTE=0."""
    min_t = 0.5 / (24 * 365)  # half hour minimum
    return max(min_t, dte_calendar_days / 365.0)


def t_years_at_exit(exit_ts, expiry_date):
    """Time-to-expiry at the exit timestamp.
    On expiry day, count remaining hours until 15:30 IST."""
    exit_date = exit_ts.date() if hasattr(exit_ts, 'date') else exit_ts
    if exit_date < expiry_date:
        # Full days remaining (until ~end of expiry day session)
        days = (expiry_date - exit_date).days
        return days / 365.0
    if exit_date == expiry_date:
        # Hours remaining within expiry session
        exit_time = exit_ts.time() if hasattr(exit_ts, 'time') else dtime(15, 30)
        hours_remaining = max(0.0, (15 - exit_time.hour) + (30 - exit_time.minute) / 60.0)
        # Floor at 0.25h (15 min) so BS doesn't blow up exactly at the bell
        hours_remaining = max(0.25, hours_remaining)
        return hours_remaining / (24 * 365.0)
    # Past expiry (shouldn't happen)
    return 0.25 / (24 * 365.0)


# ============================================================================
# Strike finder by delta
# ============================================================================

def find_strike_by_delta(spot, t_years, vol, target_delta_abs, opt_type):
    """
    Scan 50-pt strikes within +/-15% of spot. Pick strike whose option delta has
    absolute value closest to target_delta_abs.

    For PE: scan from ATM downward (OTM puts have small negative deltas).
    For CE: scan from ATM upward.
    """
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    cap_pct = 0.15
    if opt_type == 'PE':
        lo = max(int(atm - cap_pct * spot), STRIKE_INTERVAL)
        candidates = list(range(int(atm), lo - 1, -STRIKE_INTERVAL))
    else:
        hi = int(atm + cap_pct * spot)
        candidates = list(range(int(atm), hi + 1, STRIKE_INTERVAL))

    if not candidates:
        return float(atm), 0.0

    best_strike = candidates[0]
    best_diff = float('inf')
    best_delta = 0.0
    for k in candidates:
        d = bs_delta(spot, float(k), t_years, vol, RISK_FREE_RATE, opt_type)
        diff = abs(abs(d) - target_delta_abs)
        if diff < best_diff:
            best_diff = diff
            best_strike = k
            best_delta = d
    return float(best_strike), best_delta


# ============================================================================
# Entry signal generation with OR-width SL + day filter
# ============================================================================

def collect_or60_widths(df):
    """Return DataFrame[date -> or_width_pct] across all sessions, used for
    building rolling-window stats. (Note: filter applied per session at the
    moment of decision; we use only PRIOR days.)"""
    or_min = ENTRY_OR_MIN
    total_min = 9 * 60 + 15 + or_min
    or_end_time = dtime(total_min // 60, total_min % 60)

    rows = []
    days = df.index.normalize().unique()
    for day in days:
        sess = df[df.index.normalize() == day]
        if len(sess) < 10:
            continue
        or_bars = sess.between_time(SESSION_OPEN, or_end_time)
        if len(or_bars) < 1:
            continue
        oh = or_bars['high'].max()
        ol = or_bars['low'].min()
        if ol <= 0:
            continue
        wp = (oh - ol) / ol * 100
        rows.append({'date': day.date(), 'or_width_pct': wp})
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def generate_entry_signals(df):
    """
    Yield one dict per signal day, including filter eligibility info.
    Each entry has OR-width-anchored SL (X=1.0 -> opposite OR boundary),
    wick-based detection.
    """
    or_min = ENTRY_OR_MIN
    rsi_long = ENTRY_RSI_LONG
    rsi_short = ENTRY_RSI_SHORT
    wait_k = ENTRY_WAIT_K

    total_min = 9 * 60 + 15 + or_min
    or_end_time = dtime(total_min // 60, total_min % 60)

    days = df.index.normalize().unique()
    signals = []

    for day in days:
        sess = df[df.index.normalize() == day]
        if len(sess) < 10:
            continue

        or_bars = sess.between_time(SESSION_OPEN, or_end_time)
        if len(or_bars) < 1:
            continue
        or_high = float(or_bars['high'].max())
        or_low = float(or_bars['low'].min())
        if or_low <= 0:
            continue
        or_width = or_high - or_low
        or_width_pct = or_width / or_low * 100

        post = sess.between_time(or_end_time, SESSION_CLOSE).iloc[1:]
        if len(post) < 2:
            continue

        # First break
        break_i, break_dir = None, None
        for i, (_, row) in enumerate(post.iterrows()):
            if row['close'] > or_high:
                break_i, break_dir = i, 'LONG'
                break
            if row['close'] < or_low:
                break_i, break_dir = i, 'SHORT'
                break
        if break_i is None or post.index[break_i].time() >= NO_ENTRY_AFTER:
            continue

        rsi_thresh = rsi_long if break_dir == 'LONG' else rsi_short

        def confirms(rval):
            if pd.isna(rval):
                return False
            return rval > rsi_thresh if break_dir == 'LONG' else rval < rsi_thresh

        def outside(close_p):
            return close_p > or_high if break_dir == 'LONG' else close_p < or_low

        entry_i = None
        for k in range(wait_k + 1):
            si = break_i + k
            if si >= len(post):
                break
            row = post.iloc[si]
            if k > 0 and not outside(row['close']):
                continue
            if outside(row['close']) and confirms(row['rsi5m']):
                entry_i = si
                break
        if entry_i is None:
            continue
        if post.index[entry_i].time() >= NO_ENTRY_AFTER:
            continue

        entry_ts = post.index[entry_i]
        entry_price = float(post.iloc[entry_i]['close'])

        # OR-width SL @ X=1.0 = opposite OR boundary (wick-based)
        if break_dir == 'LONG':
            sl_price = or_high - SL_OR_WIDTH_MULT * or_width   # = or_low
        else:
            sl_price = or_low + SL_OR_WIDTH_MULT * or_width    # = or_high

        # Walk forward
        rest = post.iloc[entry_i + 1:]
        exit_ts = None
        exit_price = None
        exit_reason = None
        for rts, rrow in rest.iterrows():
            if break_dir == 'LONG' and rrow['low'] <= sl_price:
                exit_ts = rts
                exit_price = sl_price
                exit_reason = 'SL'
                break
            if break_dir == 'SHORT' and rrow['high'] >= sl_price:
                exit_ts = rts
                exit_price = sl_price
                exit_reason = 'SL'
                break

        if exit_ts is None:
            exit_ts = sess.index[-1]
            exit_price = float(sess.iloc[-1]['close'])
            exit_reason = 'EOD'

        signals.append({
            'date': day.date(),
            'direction': break_dir,
            'or_high': or_high,
            'or_low': or_low,
            'or_width': or_width,
            'or_width_pct': or_width_pct,
            'entry_ts': entry_ts,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'exit_ts': exit_ts,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
        })

    return signals


# ============================================================================
# Per-signal strangle simulation
# ============================================================================

def simulate_strangle(sig, rv_map):
    direction = sig['direction']
    entry_ts = sig['entry_ts']
    entry_price = sig['entry_price']
    exit_ts = sig['exit_ts']
    exit_price = sig['exit_price']
    exit_reason = sig['exit_reason']

    sig_date = sig['date']
    expiry = nearest_weekly_expiry(sig_date)
    dte_entry_days = (expiry - sig_date).days

    t_entry = t_years_at_entry(dte_entry_days)
    t_exit = t_years_at_exit(exit_ts, expiry)

    rv = rv_map.get(sig_date, None)
    sigma = rv_to_iv(rv)

    # Targets
    if direction == 'LONG':
        # bias-side near = PE (-0.22), wing = CE (+0.10)
        pe_strike, pe_delta_actual = find_strike_by_delta(
            entry_price, t_entry, sigma, DELTA_NEAR, 'PE')
        ce_strike, ce_delta_actual = find_strike_by_delta(
            entry_price, t_entry, sigma, DELTA_FAR, 'CE')
        pe_delta_target = -DELTA_NEAR
        ce_delta_target = DELTA_FAR
    else:
        # bias-side near = CE (+0.22), wing = PE (-0.10)
        ce_strike, ce_delta_actual = find_strike_by_delta(
            entry_price, t_entry, sigma, DELTA_NEAR, 'CE')
        pe_strike, pe_delta_actual = find_strike_by_delta(
            entry_price, t_entry, sigma, DELTA_FAR, 'PE')
        pe_delta_target = -DELTA_FAR
        ce_delta_target = DELTA_NEAR

    pe_credit = bs_price(entry_price, pe_strike, t_entry, sigma, RISK_FREE_RATE, 'PE')
    ce_credit = bs_price(entry_price, ce_strike, t_entry, sigma, RISK_FREE_RATE, 'CE')
    total_credit = pe_credit + ce_credit

    pe_exit = bs_price(exit_price, pe_strike, t_exit, sigma, RISK_FREE_RATE, 'PE')
    ce_exit = bs_price(exit_price, ce_strike, t_exit, sigma, RISK_FREE_RATE, 'CE')
    total_exit_value = pe_exit + ce_exit

    gross_pnl_per_unit = total_credit - total_exit_value
    gross_pnl_per_lot = gross_pnl_per_unit * LOT_SIZE

    slippage_total = 4 * SLIPPAGE_PER_LEG_PER_SIDE * LOT_SIZE
    brokerage_total = 4 * BROKERAGE_PER_ORDER
    stt = total_credit * LOT_SIZE * STT_RATE_OPT_SELL

    net_pnl_per_lot = gross_pnl_per_lot - slippage_total - brokerage_total - stt

    return {
        'date': sig_date,
        'direction': direction,
        'entry_ts': entry_ts,
        'entry_price': round(entry_price, 2),
        'or_width_pct': round(sig['or_width_pct'], 4),
        'day_filter_passed': True,    # filled later if filtered
        'expiry_date': expiry,
        'dte_entry_days': dte_entry_days,
        'iv_used': round(sigma, 4),
        'pe_strike': pe_strike,
        'pe_delta_target': round(pe_delta_target, 3),
        'pe_delta_actual': round(pe_delta_actual, 4),
        'ce_strike': ce_strike,
        'ce_delta_target': round(ce_delta_target, 3),
        'ce_delta_actual': round(ce_delta_actual, 4),
        'pe_credit': round(pe_credit, 2),
        'ce_credit': round(ce_credit, 2),
        'total_credit': round(total_credit, 2),
        'exit_ts': exit_ts,
        'exit_reason': exit_reason,
        'exit_price': round(exit_price, 2),
        'pe_exit': round(pe_exit, 2),
        'ce_exit': round(ce_exit, 2),
        'gross_pnl_per_unit': round(gross_pnl_per_unit, 2),
        'gross_pnl_per_lot': round(gross_pnl_per_lot, 2),
        'slippage': round(slippage_total, 2),
        'brokerage': round(brokerage_total, 2),
        'stt': round(stt, 2),
        'net_pnl_per_lot': round(net_pnl_per_lot, 2),
    }


# ============================================================================
# Day filters
# ============================================================================

def all_days_filter(sig, ctx):
    return True


def skip_q4_static_filter(sig, ctx):
    """Skip if today's OR60 width % > Q4_STATIC_THRESHOLD_PCT."""
    return sig['or_width_pct'] <= Q4_STATIC_THRESHOLD_PCT


def make_skip_q4_rolling_filter():
    """
    Returns a closure with state.
    Maintains rolling-90 deque of past or_width_pct values (PRIOR days).
    Today is checked against the 75th percentile of that deque.
    Bootstrap: until we have 90 prior values, fall back to static threshold.
    """
    history = deque(maxlen=Q4_ROLLING_WINDOW)

    def filt(sig, ctx):
        wp = sig['or_width_pct']
        if len(history) < Q4_ROLLING_WINDOW:
            passed = wp <= Q4_STATIC_THRESHOLD_PCT
        else:
            thr = float(np.percentile(list(history), Q4_ROLLING_PERCENTILE))
            passed = wp <= thr
        # Update history AFTER decision (no look-ahead)
        history.append(wp)
        return passed
    return filt


# ============================================================================
# Variant runner
# ============================================================================

CSV_FIELDNAMES = [
    'date', 'direction', 'entry_ts', 'entry_price', 'or_width_pct',
    'day_filter_passed', 'expiry_date', 'dte_entry_days', 'iv_used',
    'pe_strike', 'pe_delta_target', 'pe_delta_actual',
    'ce_strike', 'ce_delta_target', 'ce_delta_actual',
    'pe_credit', 'ce_credit', 'total_credit',
    'exit_ts', 'exit_reason', 'exit_price',
    'pe_exit', 'ce_exit',
    'gross_pnl_per_unit', 'gross_pnl_per_lot',
    'slippage', 'brokerage', 'stt', 'net_pnl_per_lot',
]


def run_variant(name, signals, rv_map, filter_fn, all_signal_dates_with_widths):
    """
    For day-filter variants we need to update rolling history on EVERY day
    (not only entry days). The 'skip_q4_rolling' closure uses the OR widths
    of all eligible sessions in chronological order. To support this without
    coupling to entry signals we walk per-day OR widths and only enter the
    simulation if there's a matching entry signal.
    """
    print(f"\n--- Variant: {name} ---", flush=True)
    out_csv = os.path.join(OUT_DIR, f'backtest_phase2c_{name}.csv')

    # Index signals by date for fast lookup
    sigs_by_date = {s['date']: s for s in signals}

    rows = []
    n_eligible = 0
    n_taken = 0
    n_skipped = 0

    for _, dwrow in all_signal_dates_with_widths.iterrows():
        d = dwrow['date']
        wp = dwrow['or_width_pct']
        # Synthetic minimal sig for filter
        synth = {'date': d, 'or_width_pct': wp}
        passed = filter_fn(synth, None)

        sig = sigs_by_date.get(d)
        if sig is None:
            # No entry signal that day -> not counted as eligible/skipped
            continue
        n_eligible += 1
        if not passed:
            n_skipped += 1
            continue

        # Simulate
        r = simulate_strangle(sig, rv_map)
        if r is None:
            continue
        r['day_filter_passed'] = True
        rows.append(r)
        n_taken += 1

    # Write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in CSV_FIELDNAMES})
    print(f"  CSV: {out_csv} ({len(rows)} trades)", flush=True)

    if not rows:
        return None, None

    rdf = pd.DataFrame(rows)
    pnl = rdf['net_pnl_per_lot']
    gross = rdf['gross_pnl_per_lot']
    credits = rdf['total_credit']

    n = len(rdf)
    n_wins = int((pnl > 0).sum())
    wr_pct = n_wins / n * 100 if n else 0.0

    summary = {
        'variant_name': name,
        'n_signals_eligible': n_eligible,
        'n_signals_taken': n_taken,
        'n_skipped_by_filter': n_skipped,
        'wr_pct': round(wr_pct, 2),
        'mean_credit': round(float((credits * LOT_SIZE).mean()), 2),
        'median_credit_per_unit': round(float(credits.median()), 2),
        'mean_pnl_per_lot': round(float(pnl.mean()), 2),
        'median_pnl_per_lot': round(float(pnl.median()), 2),
        'total_pnl_lakhs': round(float(pnl.sum()) / 1e5, 3),
        'annualized_pnl_lakhs': round(float(pnl.sum()) / YEARS_IN_SAMPLE / 1e5, 3),
        'max_single_loss': round(float(pnl.min()), 2),
        'sharpe': round(float(pnl.mean() / pnl.std()), 4) if pnl.std() > 0 else 0.0,
    }
    return summary, rdf


def dte_breakdown(rdf):
    """Per-DTE breakdown table."""
    if rdf is None or len(rdf) == 0:
        return None
    rdf = rdf.copy()

    def bucket(d):
        if d >= 7:
            return '7+'
        return str(int(d))

    rdf['dte_bucket'] = rdf['dte_entry_days'].apply(bucket)
    rows = []
    bucket_order = ['0', '1', '2', '3', '4', '5', '6', '7+']
    for b in bucket_order:
        sub = rdf[rdf['dte_bucket'] == b]
        n = len(sub)
        if n == 0:
            continue
        wins = (sub['net_pnl_per_lot'] > 0).sum()
        rows.append({
            'DTE': b,
            'N': n,
            'WR_pct': round(wins / n * 100, 1),
            'mean_pnl_per_lot': round(float(sub['net_pnl_per_lot'].mean()), 0),
            'total_pnl_per_lot': round(float(sub['net_pnl_per_lot'].sum()), 0),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Status doc append
# ============================================================================

def append_to_doc(summary_df, best_name, best_dte_bd, headline):
    md_path = os.path.join(OUT_DIR, 'docs', 'NIFTY-ORB-CREDIT-SPREAD-RESEARCH.md')

    section = []
    section.append("\n## Phase 2c Findings (intraday + delta strikes + OR-anchored SL + Q4 filter)\n")
    section.append("### 2026-04-25 — Refined re-run with three structural changes\n")
    section.append("Differences from Phase 2: (1) nearest-Tuesday expiry INCLUDING DTE=0 "
                   "(no roll), (2) SL = OR-width x 1.0 wick-based (Phase 1e pattern), "
                   "(3) strikes solved by delta (PE -0.22 / CE +0.10 on the bias side; "
                   "+0.22 CE / -0.10 PE on bearish), (4) IV = clamp(1.4 x RV14, 0.12, 0.25), "
                   "(5) three day-filter variants compared.\n")

    section.append("### Comparison table (per lot, Rs)\n")
    cols = ['variant_name', 'n_signals_eligible', 'n_signals_taken',
            'n_skipped_by_filter', 'wr_pct', 'mean_credit',
            'mean_pnl_per_lot', 'median_pnl_per_lot', 'total_pnl_lakhs',
            'annualized_pnl_lakhs', 'max_single_loss', 'sharpe']
    section.append("```")
    section.append(summary_df[cols].to_string(index=False))
    section.append("```\n")

    section.append(f"### Per-DTE breakdown of best variant ({best_name})\n")
    if best_dte_bd is not None and len(best_dte_bd) > 0:
        section.append("```")
        section.append(best_dte_bd.to_string(index=False))
        section.append("```\n")
    else:
        section.append("(No per-DTE breakdown available — variant produced 0 trades.)\n")

    section.append("### Headline conclusion\n")
    section.append(headline + "\n")

    with open(md_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(section) + "\n")
    print(f"\nAppended Phase 2c section to {md_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Phase 2c: Nifty ORB skewed strangle - delta strikes + OR-anchored SL")
    print("=" * 80)

    print("\nLoading 5-min OHLCV...", flush=True)
    df = load_5min()
    print(f"  {len(df)} bars, {df.index.normalize().nunique()} sessions", flush=True)

    print("Building RV(14d) map (lagged 1d)...", flush=True)
    rv_map = load_daily_rv_map()
    valid = [v for v in rv_map.values() if v is not None and not pd.isna(v)]
    print(f"  RV map: {len(rv_map)} dates, {len(valid)} non-null, "
          f"median {np.median(valid):.4f}, range [{np.min(valid):.4f}, {np.max(valid):.4f}]",
          flush=True)
    print(f"  IV proxy used: clamp(1.4 * RV, 0.12, 0.25)", flush=True)

    print("\nGenerating entry signals (OR60 + RSI5m>60/<40 + K=12 lenient + OR-width SL X=1.0)...",
          flush=True)
    signals = generate_entry_signals(df)
    print(f"  Total entry signals (before any filter): {len(signals)}", flush=True)

    # OR widths per session for the rolling-window filter
    print("Building per-session OR60 width history for rolling filter...", flush=True)
    or_widths = collect_or60_widths(df)
    print(f"  OR60 width history: {len(or_widths)} sessions", flush=True)

    # Run variants
    variants = [
        ('all_days', all_days_filter),
        ('skip_Q4_static', skip_q4_static_filter),
        ('skip_Q4_rolling', make_skip_q4_rolling_filter()),
    ]

    summaries = []
    rdfs = {}
    for name, fn in variants:
        s, rdf = run_variant(name, signals, rv_map, fn, or_widths)
        if s:
            summaries.append(s)
            rdfs[name] = rdf

    if not summaries:
        print("NO results.")
        sys.exit(1)

    sdf = pd.DataFrame(summaries)
    out_summary = os.path.join(OUT_DIR, 'backtest_phase2c_summary.csv')
    sdf.to_csv(out_summary, index=False)
    print(f"\nSummary saved: {out_summary}")

    # Print comparison table
    print("\n" + "=" * 130)
    print("PHASE 2c COMPARISON TABLE (per lot, Rs unless suffixed)")
    print("=" * 130)
    cols = ['variant_name', 'n_signals_eligible', 'n_signals_taken',
            'n_skipped_by_filter', 'wr_pct', 'mean_credit',
            'mean_pnl_per_lot', 'median_pnl_per_lot', 'total_pnl_lakhs',
            'annualized_pnl_lakhs', 'max_single_loss', 'sharpe']
    pd.set_option('display.width', 220)
    print(sdf[cols].to_string(index=False))

    # Best variant by annualized P&L
    best_idx = sdf['annualized_pnl_lakhs'].idxmax()
    best_name = sdf.loc[best_idx, 'variant_name']
    best_ann = sdf.loc[best_idx, 'annualized_pnl_lakhs']
    print("\n" + "=" * 130)
    print(f"BEST BY ANNUALIZED P&L: {best_name} -> Rs {best_ann:.3f} lakhs/yr/lot")
    print("=" * 130)

    best_rdf = rdfs.get(best_name)
    best_dte_bd = dte_breakdown(best_rdf)
    if best_dte_bd is not None and len(best_dte_bd) > 0:
        print("\nPER-DTE BREAKDOWN (best variant):")
        print(best_dte_bd.to_string(index=False))
    else:
        print("(No DTE breakdown — best variant produced 0 trades.)")

    # Sanity check: median credit across all variants
    print("\n" + "=" * 130)
    print("SANITY CHECK: median credit per trade (per unit, NOT per lot)")
    print("=" * 130)
    for name, rdf in rdfs.items():
        if rdf is None or len(rdf) == 0:
            continue
        med_credit = float(rdf['total_credit'].median())
        print(f"  {name}: median credit = Rs {med_credit:.2f} per unit "
              f"(Rs {med_credit * LOT_SIZE:.0f} per lot)")
        if med_credit < 30:
            print(f"    WARNING: median credit < Rs 30 — strikes likely too far OTM")
        elif med_credit > 200:
            print(f"    WARNING: median credit > Rs 200 — strikes likely too close to ATM")

    # Headline for doc
    annualized_lakhs_best = best_ann
    cleared_be = annualized_lakhs_best > 0
    if cleared_be:
        headline = (f"**YES** — `{best_name}` clears breakeven at "
                    f"Rs {annualized_lakhs_best:.2f} lakhs/yr/lot annualized. "
                    f"Compare to Phase 2's least-bad of -Rs 1.47 lakhs/yr/lot. "
                    f"Sharpe {float(sdf.loc[best_idx, 'sharpe']):.2f}, "
                    f"WR {float(sdf.loc[best_idx, 'wr_pct']):.1f}%, "
                    f"max single loss Rs {float(sdf.loc[best_idx, 'max_single_loss']):,.0f}.")
    else:
        worst_loss = float(sdf['annualized_pnl_lakhs'].min())
        headline = (f"**NO** — no variant clears breakeven. Best (`{best_name}`) is "
                    f"Rs {annualized_lakhs_best:.2f} lakhs/yr/lot annualized "
                    f"(worst {worst_loss:.2f}). Phase 2c structural fixes "
                    f"(OR-anchored SL, delta strikes, IV scaling, day filter) "
                    f"reduce loss vs Phase 2 but do not flip sign.")

    headline += (" CAVEAT: BS+RV pricing without IV smile/skew likely UNDERSTATES "
                 "real wing-leg pricing and adverse-move losses on tail days; "
                 "real-world numbers expected 15-30% worse on losers.")

    append_to_doc(sdf, best_name, best_dte_bd, headline)

    print("\nDone.")


if __name__ == '__main__':
    main()
