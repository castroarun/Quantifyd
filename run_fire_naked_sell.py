"""
BNF Fire Mode — Naked Option Selling with Directional Bias
============================================================
Long signal → Sell OTM PUT (bullish — profits from theta + upward move)
Short signal → Sell OTM CALL (bearish — profits from theta + downward move)

Advantage over debit spreads: theta works FOR us, profit even if price hovers.
Risk: naked selling = unlimited risk, so we use strict SL.

Sweep: strike distance, hold bars, SL levels, squeeze min, trend confirm
"""

import sys
import os
import csv
import time
import logging
import sqlite3
import calendar
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.stats import norm

logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')

LOT_SIZE = 15
RISK_FREE_RATE = 0.065
TRADING_DAYS_PER_YEAR = 252
STRIKE_STEP = 100
CAPITAL = 10_00_000


# === Black-Scholes ===

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def estimate_iv(atr, close):
    daily_std = (atr / close) / 1.2
    return max(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 1.20, 0.08)

def round_strike(price):
    return round(price / STRIKE_STEP) * STRIKE_STEP


# === Expiry ===

def get_last_thursday(year, month):
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:
        d -= timedelta(days=1)
    return d

def get_monthly_expiry(entry_date, min_dte=15):
    expiry = get_last_thursday(entry_date.year, entry_date.month)
    if expiry <= entry_date:
        m, y = (entry_date.month % 12) + 1, entry_date.year + (1 if entry_date.month == 12 else 0)
        expiry = get_last_thursday(y, m)
    dte = (expiry - entry_date).days
    if dte < min_dte:
        m, y = (expiry.month % 12) + 1, expiry.year + (1 if expiry.month == 12 else 0)
        expiry = get_last_thursday(y, m)
        dte = (expiry - entry_date).days
    return dte, expiry


# === Simulator ===

def compute_indicators(df, bb_period=10, atr_period=14, sma_period=20,
                       squeeze_min_bars=3, trend_atr_min=0.5):
    high, low, close = df['high'], df['low'], df['close']

    # ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_period).mean()

    # BB squeeze/expand
    sma_bb = close.rolling(bb_period).mean()
    std_bb = close.rolling(bb_period).std()
    bb_width = (2 * 2.0 * std_bb) / sma_bb * 100
    bb_width_ma = bb_width.rolling(bb_period).mean()
    df['bb_squeeze'] = (bb_width < bb_width_ma).astype(int)
    df['bb_expanding'] = (bb_width > bb_width_ma).astype(int)

    # Squeeze fire
    sq_sum = df['bb_squeeze'].rolling(squeeze_min_bars).sum()
    df['squeeze_fire'] = ((df['bb_expanding'] == 1) & (sq_sum.shift(1) >= squeeze_min_bars)).astype(int)

    # SMA + trend
    df['sma'] = close.rolling(sma_period).mean()
    price_dist = (close - df['sma']).abs() / df['atr']

    if trend_atr_min > 0:
        df['is_trending'] = (price_dist >= trend_atr_min).astype(int)
    else:
        df['is_trending'] = 1

    # Signals
    df['long_signal'] = ((df['squeeze_fire'] == 1) & (df['is_trending'] == 1) & (close > df['sma'])).astype(int)
    df['short_signal'] = ((df['squeeze_fire'] == 1) & (df['is_trending'] == 1) & (close < df['sma'])).astype(int)

    # IV
    df['est_iv'] = df.apply(
        lambda r: estimate_iv(r['atr'], r['close']) if r['atr'] > 0 and not np.isnan(r['atr']) else 0.15, axis=1)

    return df


def run_backtest(df, strike_atr=1.0, hold_bars=7, sl_multiplier=2.0,
                 squeeze_min_bars=3, trend_atr_min=0.5,
                 lots=5, max_loss_rupees=30000, min_gap=5):
    """
    Run naked option selling backtest.
    Long signal → sell PUT at spot - strike_atr * ATR
    Short signal → sell CALL at spot + strike_atr * ATR
    SL: exit if option value > sl_multiplier * collected premium
    Max loss cap per trade.
    """
    df = compute_indicators(df.copy(), squeeze_min_bars=squeeze_min_bars, trend_atr_min=trend_atr_min)

    trades = []
    equity = CAPITAL
    last_trade_end = -min_gap
    trade_id = 0

    for i in range(50, len(df)):
        row = df.iloc[i]

        if i <= last_trade_end:
            continue

        direction = None
        if row['long_signal'] == 1 and (i - last_trade_end) >= min_gap:
            direction = 'long'  # sell PUT
        elif row['short_signal'] == 1 and (i - last_trade_end) >= min_gap:
            direction = 'short'  # sell CALL

        if direction is None:
            continue

        spot = row['close']
        atr = row['atr']
        iv = row['est_iv']
        entry_date_str = str(row['date'])[:10]

        try:
            entry_date_obj = pd.Timestamp(row['date']).date()
        except:
            entry_date_obj = date.fromisoformat(entry_date_str)

        dte, expiry = get_monthly_expiry(entry_date_obj, min_dte=15)
        T_entry = max(dte, 1) / 365.0

        # Determine strike and price
        if direction == 'long':
            # Sell OTM PUT (strike below spot)
            strike = round_strike(spot - strike_atr * atr)
            premium = bs_put(spot, strike, T_entry, RISK_FREE_RATE, iv)
            opt_type = 'put'
        else:
            # Sell OTM CALL (strike above spot)
            strike = round_strike(spot + strike_atr * atr)
            premium = bs_call(spot, strike, T_entry, RISK_FREE_RATE, iv)
            opt_type = 'call'

        if premium <= 0:
            continue

        # SL threshold: exit if option value exceeds this
        sl_threshold = premium * sl_multiplier

        # Simulate forward
        hold_end = min(i + hold_bars, len(df) - 1)

        # Check expiry within hold
        for j in range(i + 1, hold_end + 1):
            fwd_date = str(df.iloc[j]['date'])[:10]
            if fwd_date >= str(expiry):
                hold_end = j
                break

        exit_idx = hold_end
        exit_reason = 'hold_complete'

        for j in range(i + 1, hold_end + 1):
            fwd = df.iloc[j]
            rem_dte = max(dte - (j - i), 0)
            T_rem = max(rem_dte, 0.5) / 365.0

            if opt_type == 'put':
                current_val = bs_put(fwd['close'], strike, T_rem, RISK_FREE_RATE, iv)
            else:
                current_val = bs_call(fwd['close'], strike, T_rem, RISK_FREE_RATE, iv)

            if current_val >= sl_threshold:
                exit_idx = j
                exit_reason = 'stop_loss'
                break

        # Exit valuation
        exit_row = df.iloc[exit_idx]
        rem_dte_exit = max(dte - (exit_idx - i), 0)
        T_exit = max(rem_dte_exit, 0.5) / 365.0

        if opt_type == 'put':
            exit_val = bs_put(exit_row['close'], strike, T_exit, RISK_FREE_RATE, iv)
        else:
            exit_val = bs_call(exit_row['close'], strike, T_exit, RISK_FREE_RATE, iv)

        # P&L: we sold at premium, buy back at exit_val
        pnl_points = premium - exit_val
        cost = 40 * lots * 2 + 2 * 2 * lots * LOT_SIZE  # brokerage + slippage
        pnl_rupees = pnl_points * lots * LOT_SIZE - cost
        pnl_rupees = max(pnl_rupees, -max_loss_rupees)

        pnl_pct = pnl_points / premium * 100 if premium > 0 else 0

        trades.append({
            'id': trade_id,
            'direction': direction,
            'opt_type': opt_type,
            'entry_date': entry_date_str,
            'exit_date': str(exit_row['date'])[:10],
            'entry_spot': round(spot, 2),
            'exit_spot': round(exit_row['close'], 2),
            'strike': strike,
            'premium': round(premium, 2),
            'exit_value': round(exit_val, 2),
            'pnl_points': round(pnl_points, 2),
            'pnl_rupees': round(pnl_rupees, 2),
            'pnl_pct': round(pnl_pct, 2),
            'lots': lots,
            'days_held': exit_idx - i,
            'dte': dte,
            'exit_reason': exit_reason,
        })

        equity += pnl_rupees
        last_trade_end = exit_idx
        trade_id += 1

    # Compute summary
    if not trades:
        return {'total_trades': 0, 'cagr': 0}, []

    winners = [t for t in trades if t['pnl_rupees'] > 0]
    losers = [t for t in trades if t['pnl_rupees'] <= 0]

    total = len(trades)
    wr = len(winners) / total * 100
    gp = sum(t['pnl_rupees'] for t in winners) if winners else 0
    gl = abs(sum(t['pnl_rupees'] for t in losers)) if losers else 1
    pf = gp / gl if gl > 0 else float('inf')

    final = CAPITAL + sum(t['pnl_rupees'] for t in trades)
    first = pd.Timestamp(trades[0]['entry_date'])
    last = pd.Timestamp(trades[-1]['exit_date'])
    years = max((last - first).days / 365.25, 0.1)
    cagr = ((final / CAPITAL) ** (1 / years) - 1) * 100

    # Max DD
    eq = CAPITAL
    peak = eq
    max_dd = 0
    for t in trades:
        eq += t['pnl_rupees']
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    calmar = cagr / max_dd if max_dd > 0 else 0

    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']
    long_avg = np.mean([t['pnl_pct'] for t in long_trades]) if long_trades else 0
    short_avg = np.mean([t['pnl_pct'] for t in short_trades]) if short_trades else 0

    exit_reasons = {}
    for t in trades:
        exit_reasons[t['exit_reason']] = exit_reasons.get(t['exit_reason'], 0) + 1

    summary = {
        'total_trades': total,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'total_pnl': round(sum(t['pnl_rupees'] for t in trades), 0),
        'avg_pnl_pct': round(np.mean([t['pnl_pct'] for t in trades]), 2),
        'long_avg_pnl_pct': round(long_avg, 2),
        'short_avg_pnl_pct': round(short_avg, 2),
        'cagr': round(cagr, 2),
        'max_drawdown': round(max_dd, 2),
        'calmar': round(calmar, 2),
        'final_equity': round(final, 0),
        'total_return_pct': round((final / CAPITAL - 1) * 100, 2),
        'avg_days_held': round(np.mean([t['days_held'] for t in trades]), 1),
        'avg_premium': round(np.mean([t['premium'] for t in trades]), 1),
        'exit_reasons': exit_reasons,
    }
    return summary, trades


def main():
    # Load data
    print('Loading BankNifty daily data...')
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume FROM market_data_unified
           WHERE symbol='BANKNIFTY' AND timeframe='day'
           AND date >= '2023-01-01' AND date <= '2025-12-31' ORDER BY date""",
        conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    print(f'Loaded {len(df)} bars')

    # === SWEEP ===
    configs = []
    for strike_atr in [0.5, 1.0, 1.5, 2.0]:
        for hold in [5, 7, 10, 14]:
            for sl_mult in [1.5, 2.0, 2.5, 3.0]:
                for sq_min in [3, 5]:
                    for tf_min in [0.0, 0.5, 1.0]:
                        for ml in [20000, 30000, 50000]:
                            label = f'STK{strike_atr}_H{hold}_SL{sl_mult}x_SQ{sq_min}_TF{tf_min}_ML{ml//1000}k'
                            configs.append((label, strike_atr, hold, sl_mult, sq_min, tf_min, ml))

    total = len(configs)
    print(f'Configs: {total}')

    print(f'\n{"Label":<50} {"Tr":>3} {"L/S":>5} {"WR%":>5} {"PF":>6} '
          f'{"CAGR%":>7} {"DD%":>5} {"Calm":>5} {"AvgPnl%":>8} {"TotalPnL":>10} {"AvgPrem":>7} {"Exits":>20}')
    print('-' * 145)

    results = []
    best_cagr = -999
    best_label = ''

    for label, strike_atr, hold, sl_mult, sq_min, tf_min, ml in configs:
        summary, trades = run_backtest(
            df, strike_atr=strike_atr, hold_bars=hold, sl_multiplier=sl_mult,
            squeeze_min_bars=sq_min, trend_atr_min=tf_min, max_loss_rupees=ml)

        if summary['total_trades'] == 0:
            continue

        results.append({'label': label, **summary})

        if summary['cagr'] > best_cagr and summary['total_trades'] >= 3:
            best_cagr = summary['cagr']
            best_label = label

        # Print notable ones
        if summary['total_trades'] >= 5 and summary['cagr'] > 3:
            exits = summary.get('exit_reasons', {})
            exit_str = ' '.join(f'{k}={v}' for k, v in exits.items())
            print(
                f'{label:<50} {summary["total_trades"]:>3} '
                f'{summary["long_trades"]:>2}/{summary["short_trades"]:<2} '
                f'{summary["win_rate"]:>5.1f} {summary["profit_factor"]:>6.2f} '
                f'{summary["cagr"]:>7.2f} {summary["max_drawdown"]:>5.2f} '
                f'{summary["calmar"]:>5.2f} {summary["avg_pnl_pct"]:>8.2f} '
                f'{summary["total_pnl"]:>10,.0f} {summary["avg_premium"]:>7.1f} '
                f'{exit_str:>20}'
            )

    # === TOP RESULTS ===
    if not results:
        print('\nNo configs with trades!')
        return

    rdf = pd.DataFrame(results)
    has = rdf[rdf['total_trades'] >= 3]

    print(f'\n{"="*100}')
    print(f'TOP 20 BY CAGR (min 3 trades, {len(has)} qualifying):')
    print(f'{"="*100}')
    top = has.nlargest(20, 'cagr')
    for _, r in top.iterrows():
        print(f'  {r["label"]:<50} T={r["total_trades"]:>2}({r["long_trades"]:.0f}L/{r["short_trades"]:.0f}S) '
              f'WR={r["win_rate"]}% PF={r["profit_factor"]:.2f} '
              f'CAGR={r["cagr"]}% DD={r["max_drawdown"]}% Calm={r["calmar"]:.2f} '
              f'PnL=Rs{r["total_pnl"]:+,.0f}')

    print(f'\n{"="*100}')
    print(f'TOP 20 BY CALMAR:')
    print(f'{"="*100}')
    cal = has[has['max_drawdown'] > 0].nlargest(20, 'calmar')
    for _, r in cal.iterrows():
        print(f'  {r["label"]:<50} T={r["total_trades"]:>2}({r["long_trades"]:.0f}L/{r["short_trades"]:.0f}S) '
              f'WR={r["win_rate"]}% PF={r["profit_factor"]:.2f} '
              f'CAGR={r["cagr"]}% DD={r["max_drawdown"]}% Calm={r["calmar"]:.2f}')

    # Param analysis
    print(f'\n{"="*100}')
    print('PARAM IMPACT ON CAGR:')
    for col in ['strike_atr', 'hold', 'sl_mult', 'sq_min', 'tf_min', 'ml']:
        # Extract param from label
        pass

    # Direction analysis
    print(f'\nDIRECTION: Long avg PnL%={has["long_avg_pnl_pct"].mean():.2f}  Short avg PnL%={has["short_avg_pnl_pct"].mean():.2f}')

    # === Print best config trades ===
    print(f'\n{"="*100}')
    print(f'BEST CONFIG TRADES: {best_label}')
    print(f'{"="*100}')
    summary, trades = run_backtest(
        df, **_parse_label(best_label))
    for t in trades:
        dir_sym = 'SELL PUT' if t['direction'] == 'long' else 'SELL CALL'
        print(f'  #{t["id"]:>2} {dir_sym:<12} {t["entry_date"]} -> {t["exit_date"]} '
              f'Spot={t["entry_spot"]:>8.0f} Strike={t["strike"]:>6.0f} '
              f'Prem={t["premium"]:>6.1f} Exit={t["exit_value"]:>6.1f} '
              f'PnL={t["pnl_pct"]:>+7.1f}% Rs={t["pnl_rupees"]:>+9,.0f} {t["exit_reason"]}')

    # === Compare with debit spread results ===
    print(f'\n{"="*100}')
    print('COMPARISON: Naked Sell vs Debit Spread (Fire mode)')
    print(f'{"="*100}')
    print(f'  Debit Spread best: CAGR=3.80% | DD=1.97% | Calmar=1.93 | 30 trades | PnL=Rs+99,117')
    print(f'  Naked Sell best:   CAGR={summary["cagr"]:.2f}% | DD={summary["max_drawdown"]:.2f}% | '
          f'Calmar={summary["calmar"]:.2f} | {summary["total_trades"]} trades | PnL=Rs{summary["total_pnl"]:+,.0f}')


def _parse_label(label):
    """Parse config label back to params."""
    parts = label.split('_')
    params = {}
    for p in parts:
        if p.startswith('STK'):
            params['strike_atr'] = float(p[3:])
        elif p.startswith('H'):
            params['hold_bars'] = int(p[1:])
        elif p.endswith('x'):
            params['sl_multiplier'] = float(p[2:-1])
        elif p.startswith('SQ'):
            params['squeeze_min_bars'] = int(p[2:])
        elif p.startswith('TF'):
            params['trend_atr_min'] = float(p[2:])
        elif p.endswith('k'):
            params['max_loss_rupees'] = int(p[2:-1]) * 1000
    return params


if __name__ == '__main__':
    main()
