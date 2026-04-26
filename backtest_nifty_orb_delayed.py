"""
Phase 1b: Delayed-entry RSI confirmation sweep.

For each ORB break, instead of REQUIRING RSI confirmation on the break candle,
we wait up to K candles for RSI to come into range. If price re-enters the OR
during the wait window, we ABORT (skip the signal — it's a failed break).

Compare across:
  - OR TF: 15, 30 min
  - RSI threshold (long/short): (55,45), (60,40), (65,35)
  - Wait window K (5-min candles): 0, 1, 2, 3, 6, 12 (=0/5/10/15/30/60 min wait)

K=0 is the strict-confirmation case (same as my original test).
"""

import sqlite3
from datetime import time as dtime

import numpy as np
import pandas as pd

DB = 'backtest_data/market_data.db'
SYMBOL = 'NIFTY50'
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
NO_ENTRY_AFTER = dtime(14, 0)


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_df():
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


def run_variant(df, or_min, rsi_lo, rsi_hi, wait_k):
    """
    Returns list of trade dicts and counters for break_count / aborted.
    rsi_lo: lower bound for LONG (rsi > rsi_lo)
    rsi_hi: upper bound for SHORT (rsi < rsi_hi)
    """
    trades = []
    raw_break_count = 0
    aborted_count = 0
    confirmed_immediate = 0
    confirmed_delayed = 0
    never_confirmed = 0

    or_end_time = dtime(9, 15 + or_min)
    days = df.index.normalize().unique()

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

        # Find first break candle
        break_idx = None
        break_dir = None
        for i, (ts, row) in enumerate(post.iterrows()):
            if row['close'] > or_high:
                break_idx, break_dir = i, 'LONG'
                break
            if row['close'] < or_low:
                break_idx, break_dir = i, 'SHORT'
                break
        if break_idx is None:
            continue
        if post.index[break_idx].time() >= NO_ENTRY_AFTER:
            continue

        raw_break_count += 1

        # Wait for RSI confirmation within K candles
        entry_idx = None
        for k in range(wait_k + 1):
            scan_i = break_idx + k
            if scan_i >= len(post):
                break
            row = post.iloc[scan_i]
            # Abort if price re-entered OR (failed break)
            if k > 0:
                if break_dir == 'LONG' and row['close'] <= or_high:
                    entry_idx = 'aborted'
                    break
                if break_dir == 'SHORT' and row['close'] >= or_low:
                    entry_idx = 'aborted'
                    break
            # Check if RSI confirms
            r = row['rsi5m']
            if pd.isna(r):
                continue
            if break_dir == 'LONG' and r > rsi_lo:
                entry_idx = scan_i
                if k == 0:
                    confirmed_immediate += 1
                else:
                    confirmed_delayed += 1
                break
            if break_dir == 'SHORT' and r < rsi_hi:
                entry_idx = scan_i
                if k == 0:
                    confirmed_immediate += 1
                else:
                    confirmed_delayed += 1
                break
        else:
            # ran through wait_k without confirmation
            never_confirmed += 1
            continue

        if entry_idx == 'aborted':
            aborted_count += 1
            continue
        if entry_idx is None:
            never_confirmed += 1
            continue

        entry_row = post.iloc[entry_idx]
        entry_ts = post.index[entry_idx]
        break_ts = post.index[break_idx]
        wait_min = (entry_ts - break_ts).total_seconds() / 60

        # Skip if entry crossed past 14:00
        if entry_ts.time() >= NO_ENTRY_AFTER:
            continue

        # Scan rest for SL (opposite OR breach, close-based)
        rest = post.iloc[entry_idx + 1:]
        sl_hit = False
        sl_time = None
        for rts, rrow in rest.iterrows():
            if break_dir == 'LONG' and rrow['close'] < or_low:
                sl_hit = True
                sl_time = rts
                break
            if break_dir == 'SHORT' and rrow['close'] > or_high:
                sl_hit = True
                sl_time = rts
                break

        trades.append({
            'date': day.date(),
            'direction': break_dir,
            'break_ts': break_ts,
            'entry_ts': entry_ts,
            'wait_min': wait_min,
            'entry_price': float(entry_row['close']),
            'sl_hit': sl_hit,
            'sl_time': sl_time,
            'mins_to_sl': (sl_time - entry_ts).total_seconds() / 60 if sl_hit else None,
        })

    return {
        'trades': trades,
        'raw_breaks': raw_break_count,
        'confirmed_immediate': confirmed_immediate,
        'confirmed_delayed': confirmed_delayed,
        'aborted': aborted_count,
        'never_confirmed': never_confirmed,
    }


def summarize(out, label):
    trades = out['trades']
    n = len(trades)
    if n == 0:
        return {'variant': label, 'N_taken': 0, 'WR_pct': 0.0,
                'raw_breaks': out['raw_breaks'], 'aborted': out['aborted'],
                'never_conf': out['never_confirmed'],
                'imm_conf': out['confirmed_immediate'],
                'delayed_conf': out['confirmed_delayed'],
                'med_wait_min': None, 'med_min_to_sl': None}

    df = pd.DataFrame(trades)
    wins = df[~df['sl_hit']]
    losses = df[df['sl_hit']]
    return {
        'variant': label,
        'N_taken': n,
        'WR_pct': round(len(wins) / n * 100, 1),
        'raw_breaks': out['raw_breaks'],
        'aborted': out['aborted'],
        'never_conf': out['never_confirmed'],
        'imm_conf': out['confirmed_immediate'],
        'delayed_conf': out['confirmed_delayed'],
        'med_wait_min': round(float(df['wait_min'].median()), 1),
        'med_min_to_sl': round(float(losses['mins_to_sl'].median()), 0) if len(losses) else None,
    }


def main():
    print("Loading NIFTY50 5-min data...")
    df = load_df()
    print(f"  {len(df)} bars, {df.index.normalize().nunique()} sessions")

    rows = []
    for or_min in [15, 30]:
        for (lo, hi) in [(55, 45), (60, 40), (65, 35)]:
            for k in [0, 1, 2, 3, 6, 12]:
                label = f"OR{or_min}m  RSI>{lo}/<{hi}  K={k} ({k*5}min wait)"
                out = run_variant(df, or_min, lo, hi, k)
                rows.append(summarize(out, label))

    res = pd.DataFrame(rows)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 60)

    # Print full table
    print("\n" + "=" * 150)
    print("DELAYED-ENTRY RSI CONFIRMATION SWEEP")
    print("Win = opposite OR not breached before EOD")
    print("=" * 150)
    print(res.to_string(index=False))

    # Top 10 by WR with N >= 50 (statistical floor)
    res_meaningful = res[res['N_taken'] >= 50].copy()
    print("\n" + "-" * 100)
    print("TOP 10 by Win Rate (N >= 50)")
    print("-" * 100)
    top = res_meaningful.sort_values('WR_pct', ascending=False).head(10)
    print(top.to_string(index=False))

    res.to_csv('nifty_orb_delayed_sweep.csv', index=False)
    print(f"\nSaved: nifty_orb_delayed_sweep.csv")


if __name__ == '__main__':
    main()
