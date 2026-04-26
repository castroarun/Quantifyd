"""
Phase 2: Nifty ORB Skewed Strangle - Real Rupee P&L Simulation

Reuses the Phase 1 winner entry config:
    OR60m + RSI5m>60 long / <40 short + K=12 lenient abort

For each entry signal, simulate a skewed short strangle:
    LONG (bullish break)  -> short PE near-ATM (~Rs 40 prem) + short CE far-OTM (~Rs 20 prem)
    SHORT (bearish break) -> short CE near-ATM (~Rs 40 prem) + short PE far-OTM (~Rs 20 prem)

Hold until:
    - Underlying breaches SL price (% retracement from entry), OR
    - EOD square-off

Mark-to-market both legs at exit using BS, compute net P&L per lot.

Sweep over SL retracement %: 0.30, 0.40, 0.50 (the meaningful Phase-1 zone).

================================================================================
Decisions / assumptions (please read):
================================================================================

1) Weekly expiry day: TUESDAY.
   SEBI / NSE migrated NIFTY weekly options expiry from Thursday to Tuesday in
   late 2024 (effective ~early-Sep-2024 per NSE circulars; subsequent
   regulatory tweaks settled it on Tuesday for current sample as of 2026-04).
   For simplicity and consistency we use **Tuesday** as the weekly expiry day
   for the ENTIRE sample. This is a known approximation: pre-Sep-2024 trades
   would have been on Thursday weekly; using Tuesday throughout slightly
   shortens DTE on average (Tue is earlier in the week than Thu) and therefore
   slightly UNDERSTATES theta collected.

2) IV proxy: 14-day realized vol (RV) from daily Nifty close-to-close log
   returns, annualized by sqrt(252). Lagged by 1 day (use yesterday's RV
   for today's pricing -> no look-ahead).
   This UNDERESTIMATES real Nifty option IV because the index vol smile
   typically prices ATM IV at 1.2-1.5x RV and OTM IV higher still.
   Real-world losses on SL-hit days will be 20-40% worse than these sims.

3) Strike interval: 50 points (Nifty options ladder).

4) Lot size: 65 (post Nov-2024 SEBI revision).

5) Risk-free rate: 6.5%.

6) Costs per round-trip per lot:
   - Slippage: Rs 1 per leg per side x 4 fills x 65 = Rs 260
   - Brokerage: Rs 20 per executed order x 4 orders = Rs 80 (FLAT, not per-lot)
   - STT on options sell side: 0.05% on entry premium (sell-to-open)
   - Other taxes ignored (exch txn, GST, SEBI - small noise)

7) Same-day expiry handling: if signal day == expiry day (DTE=0), we use
   NEXT week's expiry to avoid 0DTE pin-risk for this v1.

8) Entry signal logic copied verbatim from backtest_nifty_orb_pct_sl.py
   with default Phase-1 winner params (OR60, RSI5m>60/<40, K=12, lenient).
"""

import csv
import math
import os
import sqlite3
import sys
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
SLIPPAGE_PER_LEG_PER_SIDE = 1.0    # Rs per leg per side
BROKERAGE_PER_ORDER = 20.0          # Rs flat per order
STT_RATE_OPT_SELL = 0.0005          # 0.05% on premium received (sell to open)
WEEKLY_EXPIRY_WEEKDAY = 1           # 0=Mon, 1=Tue (current Nifty weekly)

# Strangle target premiums
NEAR_ATM_TARGET_PREM = 40.0   # Skewed-toward-direction leg
FAR_OTM_TARGET_PREM = 20.0    # Far-OTM leg

DEFAULT_IV = 0.14             # Fallback IV if RV unavailable

# Phase-1 winner entry params
ENTRY_OR_MIN = 60
ENTRY_RSI_LONG = 60
ENTRY_RSI_SHORT = 40
ENTRY_WAIT_K = 12

# SL retracement % sweep (matches Phase 1 meaningful zone)
SL_PCT_SWEEP = [0.30, 0.40, 0.50]

YEARS_IN_SAMPLE = 455 / 252   # ~1.81 yrs

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Black-Scholes (in-house, no scipy)
# ============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t_years: float,
             vol: float, r: float, opt_type: str) -> float:
    """
    European Black-Scholes price.
    opt_type: 'CE' (call) or 'PE' (put)
    Returns >= 0.01 floor.
    """
    is_call = (opt_type == 'CE')
    if t_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        if is_call:
            return max(spot - strike, 0.01)
        return max(strike - spot, 0.01)

    sigma_t = vol * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / sigma_t
    d2 = d1 - sigma_t

    if is_call:
        price = spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    else:
        price = strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

    return max(round(price, 2), 0.01)


# Self-test on import: ATM call spot=22000, strike=22000, T=7d, vol=0.14
# Approximate ATM closed-form: 0.4 * S * sigma * sqrt(T) ~= 0.4 * 22000 * 0.14 * sqrt(7/365) ~= Rs 170
# Sanity range Rs 130-220 (loose; covers normal +/-30% bracket).
_test = bs_price(22000, 22000, 7/365.0, 0.14, 0.065, 'CE')
assert 130 < _test < 220, f"BS sanity check failed: ATM 7d call = {_test}"


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
    """
    Build {date -> annualized 14d RV} from daily Nifty close-to-close log returns.
    Lagged by 1 day -> use yesterday's RV as today's IV input (no look-ahead).
    """
    conn = sqlite3.connect(DB)
    d = pd.read_sql(
        "SELECT date, close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date",
        conn, params=[SYMBOL])
    conn.close()
    d['date'] = pd.to_datetime(d['date']).dt.date
    d['close'] = d['close'].astype(float)
    d['log_ret'] = np.log(d['close'] / d['close'].shift(1))
    # 14d rolling stdev annualized
    d['rv14'] = d['log_ret'].rolling(14).std() * np.sqrt(252)
    # Lag by 1 (use yesterday's RV for today)
    d['rv14_lag'] = d['rv14'].shift(1)
    rv_map = dict(zip(d['date'], d['rv14_lag']))
    return rv_map


# ============================================================================
# Expiry mechanics
# ============================================================================

def next_weekly_expiry(d: date) -> date:
    """
    Return the nearest weekly expiry date (Tuesday) on or after d.
    If d itself is Tuesday, return d (DTE=0); caller decides whether to roll.
    """
    days_ahead = (WEEKLY_EXPIRY_WEEKDAY - d.weekday()) % 7
    return d + timedelta(days=days_ahead)


def pick_expiry_for_signal(signal_date: date) -> date:
    """
    Pick weekly expiry s.t. DTE >= 1.
    If signal_date IS expiry day, roll to next week.
    """
    e = next_weekly_expiry(signal_date)
    if e == signal_date:
        e = e + timedelta(days=7)
    return e


# ============================================================================
# Strike finder
# ============================================================================

def find_strike_by_target_prem(spot: float, t_years: float, vol: float,
                               target_prem: float, opt_type: str) -> float:
    """
    Find the 50-point strike whose BS price is closest to target_prem.
    For PE: scan strikes from spot DOWNWARD (OTM puts).
    For CE: scan strikes from spot UPWARD (OTM calls).
    Cap range to +/- 10% of spot.
    """
    # Round spot to nearest strike
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    cap_pct = 0.10
    if opt_type == 'PE':
        lo = max(atm - cap_pct * spot, 1.0)
        candidates = list(range(int(atm), int(lo) - 1, -STRIKE_INTERVAL))
    else:  # CE
        hi = atm + cap_pct * spot
        candidates = list(range(int(atm), int(hi) + 1, STRIKE_INTERVAL))

    if not candidates:
        return float(atm)

    best_strike = candidates[0]
    best_diff = float('inf')
    for k in candidates:
        p = bs_price(spot, float(k), t_years, vol, RISK_FREE_RATE, opt_type)
        diff = abs(p - target_prem)
        if diff < best_diff:
            best_diff = diff
            best_strike = k
    return float(best_strike)


# ============================================================================
# Entry signal generation (copied from backtest_nifty_orb_pct_sl.py, distilled)
# ============================================================================

def generate_entry_signals(df, sl_pct: float):
    """
    Yield one dict per entry signal with:
       date, direction, entry_ts, entry_price, sl_price,
       exit_ts, exit_price, exit_reason ('SL' or 'EOD')
    Entry config is the Phase-1 winner; sl_pct sweeps.
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
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()

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

        if break_dir == 'LONG':
            sl_price = entry_price * (1 - sl_pct / 100.0)
        else:
            sl_price = entry_price * (1 + sl_pct / 100.0)

        # Walk forward bar-by-bar -> SL hit (wick) or EOD
        rest = post.iloc[entry_i + 1:]
        exit_ts = None
        exit_price = None
        exit_reason = None
        for rts, rrow in rest.iterrows():
            if break_dir == 'LONG' and rrow['low'] <= sl_price:
                exit_ts = rts
                exit_price = sl_price  # SL fill at trigger
                exit_reason = 'SL'
                break
            if break_dir == 'SHORT' and rrow['high'] >= sl_price:
                exit_ts = rts
                exit_price = sl_price
                exit_reason = 'SL'
                break

        if exit_ts is None:
            # EOD square-off at last bar's close
            last = sess.iloc[-1]
            exit_ts = sess.index[-1]
            exit_price = float(last['close'])
            exit_reason = 'EOD'

        signals.append({
            'date': day.date(),
            'direction': break_dir,
            'entry_ts': entry_ts,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'exit_ts': exit_ts,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
        })

    return signals


# ============================================================================
# Phase 2 sim per signal
# ============================================================================

def simulate_strangle(sig: dict, rv_map: dict) -> dict | None:
    """
    Given an entry signal, simulate skewed short strangle P&L.
    Returns row dict or None if data missing.
    """
    direction = sig['direction']
    entry_ts = sig['entry_ts']
    entry_price = sig['entry_price']
    exit_ts = sig['exit_ts']
    exit_price = sig['exit_price']
    exit_reason = sig['exit_reason']

    sig_date = sig['date']
    expiry = pick_expiry_for_signal(sig_date)

    dte_entry_days = (expiry - sig_date).days
    t_entry = dte_entry_days / 365.0

    exit_date = exit_ts.date() if hasattr(exit_ts, 'date') else exit_ts
    dte_exit_days = max((expiry - exit_date).days, 0)
    t_exit = dte_exit_days / 365.0

    # IV from RV map; lookup yesterday's value (already lagged)
    sigma = rv_map.get(sig_date, None)
    if sigma is None or pd.isna(sigma) or sigma <= 0:
        sigma = DEFAULT_IV

    # Choose legs based on direction
    if direction == 'LONG':
        # Bullish bias -> short PE near-ATM (~40), short CE far-OTM (~20)
        pe_strike = find_strike_by_target_prem(
            entry_price, t_entry, sigma, NEAR_ATM_TARGET_PREM, 'PE')
        ce_strike = find_strike_by_target_prem(
            entry_price, t_entry, sigma, FAR_OTM_TARGET_PREM, 'CE')
    else:  # SHORT
        # Bearish bias -> short CE near-ATM (~40), short PE far-OTM (~20)
        ce_strike = find_strike_by_target_prem(
            entry_price, t_entry, sigma, NEAR_ATM_TARGET_PREM, 'CE')
        pe_strike = find_strike_by_target_prem(
            entry_price, t_entry, sigma, FAR_OTM_TARGET_PREM, 'PE')

    pe_credit = bs_price(entry_price, pe_strike, t_entry, sigma, RISK_FREE_RATE, 'PE')
    ce_credit = bs_price(entry_price, ce_strike, t_entry, sigma, RISK_FREE_RATE, 'CE')
    total_credit = pe_credit + ce_credit

    pe_exit = bs_price(exit_price, pe_strike, t_exit, sigma, RISK_FREE_RATE, 'PE')
    ce_exit = bs_price(exit_price, ce_strike, t_exit, sigma, RISK_FREE_RATE, 'CE')
    total_exit_value = pe_exit + ce_exit

    # We sold to open -> P&L per share (per index unit) = credit - exit_value
    gross_pnl_per_unit = total_credit - total_exit_value
    gross_pnl_per_lot = gross_pnl_per_unit * LOT_SIZE

    # Costs
    slippage_total = 4 * SLIPPAGE_PER_LEG_PER_SIDE * LOT_SIZE  # 4 fills
    brokerage_total = 4 * BROKERAGE_PER_ORDER  # 4 orders flat
    stt = total_credit * LOT_SIZE * STT_RATE_OPT_SELL  # entry sell side only

    net_pnl_per_lot = gross_pnl_per_lot - slippage_total - brokerage_total - stt

    return {
        'date': sig_date,
        'direction': direction,
        'entry_ts': entry_ts,
        'entry_price': round(entry_price, 2),
        'expiry': expiry,
        'dte_entry_days': dte_entry_days,
        'iv_used': round(sigma, 4),
        'pe_strike': pe_strike,
        'ce_strike': ce_strike,
        'pe_credit': round(pe_credit, 2),
        'ce_credit': round(ce_credit, 2),
        'total_credit': round(total_credit, 2),
        'exit_ts': exit_ts,
        'exit_price': round(exit_price, 2),
        'exit_reason': exit_reason,
        'pe_exit': round(pe_exit, 2),
        'ce_exit': round(ce_exit, 2),
        'total_exit_value': round(total_exit_value, 2),
        'gross_pnl_per_unit': round(gross_pnl_per_unit, 2),
        'gross_pnl_per_lot': round(gross_pnl_per_lot, 2),
        'slippage_total': round(slippage_total, 2),
        'brokerage_total': round(brokerage_total, 2),
        'stt': round(stt, 2),
        'net_pnl_per_lot': round(net_pnl_per_lot, 2),
    }


# ============================================================================
# Per-variant runner + summary
# ============================================================================

def run_variant(df, rv_map, sl_pct: float):
    print(f"\n--- Running Phase 2 sim for SL={sl_pct}% ---", flush=True)
    sigs = generate_entry_signals(df, sl_pct)
    print(f"  Entry signals generated: {len(sigs)}", flush=True)

    rows = []
    for s in sigs:
        r = simulate_strangle(s, rv_map)
        if r is not None:
            rows.append(r)

    out_csv = os.path.join(OUT_DIR, f'backtest_phase2_OR60_SL{sl_pct:.2f}.csv')
    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"  Per-trade detail saved: {out_csv}", flush=True)

    # Summary stats
    if not rows:
        return None

    rdf = pd.DataFrame(rows)
    pnl = rdf['net_pnl_per_lot']
    gross = rdf['gross_pnl_per_lot']
    credits = rdf['total_credit']

    wins_mask = gross > 0   # gross > 0 = win per spec
    n = len(rdf)
    n_wins = int(wins_mask.sum())
    n_losses = n - n_wins
    wr_pct = n_wins / n * 100 if n else 0.0

    summary = {
        'variant': f'OR60_SL{sl_pct:.2f}',
        'sl_pct': sl_pct,
        'N': n,
        'wins': n_wins,
        'losses': n_losses,
        'WR_pct': round(wr_pct, 1),
        'mean_pnl_rs': round(float(pnl.mean()), 0),
        'median_pnl_rs': round(float(pnl.median()), 0),
        'std_pnl_rs': round(float(pnl.std()), 0),
        'total_pnl_rs': round(float(pnl.sum()), 0),
        'annualized_pnl_rs': round(float(pnl.sum()) / YEARS_IN_SAMPLE, 0),
        'max_loss_rs': round(float(pnl.min()), 0),
        'max_win_rs': round(float(pnl.max()), 0),
        'p05_pnl_rs': round(float(pnl.quantile(0.05)), 0),
        'p95_pnl_rs': round(float(pnl.quantile(0.95)), 0),
        'avg_credit_rs': round(float((credits * LOT_SIZE).mean()), 0),
        'avg_credit_kept_on_wins_rs': round(float(gross[wins_mask].mean()), 0) if n_wins else 0,
        'avg_loss_on_losses_rs': round(float(gross[~wins_mask].mean()), 0) if n_losses else 0,
        'sharpe_simple': round(float(pnl.mean() / pnl.std()), 3) if pnl.std() > 0 else 0.0,
    }
    return summary


def main():
    print("=" * 80)
    print("Phase 2: Nifty ORB Skewed Strangle - Real Rupee P&L")
    print("=" * 80)

    print("\nLoading 5-min OHLCV...", flush=True)
    df = load_5min()
    print(f"  {len(df)} bars, {df.index.normalize().nunique()} sessions", flush=True)

    print("Building RV(14d) -> IV proxy map from daily data...", flush=True)
    rv_map = load_daily_rv_map()
    valid_rv = [v for v in rv_map.values() if v is not None and not pd.isna(v)]
    print(f"  RV map: {len(rv_map)} dates, {len(valid_rv)} non-null. "
          f"Median RV={np.median(valid_rv):.3f}, "
          f"min={np.min(valid_rv):.3f}, max={np.max(valid_rv):.3f}", flush=True)

    summaries = []
    for sl_pct in SL_PCT_SWEEP:
        s = run_variant(df, rv_map, sl_pct)
        if s:
            summaries.append(s)

    # Save + print summary
    if summaries:
        sdf = pd.DataFrame(summaries)
        out_summary = os.path.join(OUT_DIR, 'backtest_phase2_summary.csv')
        sdf.to_csv(out_summary, index=False)
        print(f"\nSummary saved: {out_summary}")

        print("\n" + "=" * 110)
        print("PHASE 2 SUMMARY (per lot, Rs)")
        print("=" * 110)
        cols = ['variant', 'N', 'WR_pct', 'mean_pnl_rs', 'median_pnl_rs',
                'total_pnl_rs', 'annualized_pnl_rs', 'max_loss_rs', 'sharpe_simple']
        pd.set_option('display.width', 200)
        print(sdf[cols].to_string(index=False))

        # Best variant
        best = sdf.loc[sdf['annualized_pnl_rs'].idxmax()]
        print("\n" + "=" * 110)
        print(f"BEST BY ANNUALIZED P&L: {best['variant']} "
              f"-> Rs {best['annualized_pnl_rs']:,.0f}/yr/lot "
              f"(WR {best['WR_pct']}%, mean Rs {best['mean_pnl_rs']:,.0f}, "
              f"max loss Rs {best['max_loss_rs']:,.0f})")
        print("=" * 110)
    else:
        print("\nNO SUMMARIES GENERATED.")
        sys.exit(1)


if __name__ == '__main__':
    main()
