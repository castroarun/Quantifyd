"""
Phase 1: Nifty ORB Obedience Stats
===================================
For each trading day on NIFTY50 (5-min, 2024-03-01 -> 2026-03-25),
identify first ORB break after the OR window on each of several OR TFs,
record features at entry, and check whether price breaches the opposite OR
boundary before EOD.

Binary outcome per signal:
    WIN  = no opposite-OR breach by EOD (theta collected, credit spread profitable)
    LOSS = opposite-OR breach (SL triggered on credit spread)

This is the "obedience rate" of Nifty to its opening range.
"""

import sqlite3
from datetime import time as dtime

import numpy as np
import pandas as pd

DB = 'backtest_data/market_data.db'
SYMBOL = 'NIFTY50'
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)

# OR windows (minutes from 9:15)
OR_WINDOWS = [5, 10, 15, 30]

# No entries after this time
NO_ENTRY_AFTER = dtime(14, 0)


# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - sig


# -----------------------------------------------------------------------------
# Data load
# -----------------------------------------------------------------------------

def load_nifty_5min():
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",
        conn, params=[SYMBOL])
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    # Filter to regular session (drop any pre/post-market if present)
    df = df.between_time(SESSION_OPEN, SESSION_CLOSE)
    return df


def add_features(df):
    df = df.copy()
    df['rsi5m'] = rsi(df['close'], 14)
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    df['macd_h'] = macd_hist(df['close'])

    # 15-min RSI: resample close to 15-min, compute RSI, forward-fill back to 5-min index
    close15 = df['close'].resample('15min').last().dropna()
    rsi15 = rsi(close15, 14)
    df['rsi15m'] = rsi15.reindex(df.index, method='ffill')

    return df


# -----------------------------------------------------------------------------
# Signal builder
# -----------------------------------------------------------------------------

def build_signals(df):
    """
    For each (session, OR window, direction), record the first break after OR end.
    Returns DataFrame with one row per signal.
    """
    signals = []
    prev_close = None
    days = df.index.normalize().unique()

    for day in days:
        sess = df[df.index.normalize() == day]
        if len(sess) < 10:
            continue

        # Find session's first bar at/after 9:15
        sess_open_price = sess.iloc[0]['open']
        day_gap_pct = 0.0
        if prev_close is not None and prev_close > 0:
            day_gap_pct = (sess_open_price / prev_close - 1) * 100

        for win_min in OR_WINDOWS:
            or_end = dtime(9, 15 + win_min)
            or_bars = sess.between_time(SESSION_OPEN, or_end)
            if len(or_bars) < 1:
                continue
            or_high = or_bars['high'].max()
            or_low = or_bars['low'].min()
            or_width = or_high - or_low

            # Bars after OR window
            post = sess.between_time(or_end, SESSION_CLOSE).iloc[1:]  # skip the or_end bar itself
            if len(post) < 2:
                continue

            # Find first bar with close breaching OR
            break_row = None
            break_dir = None
            for ts, row in post.iterrows():
                if row['close'] > or_high:
                    break_row = (ts, row)
                    break_dir = 'LONG'
                    break
                if row['close'] < or_low:
                    break_row = (ts, row)
                    break_dir = 'SHORT'
                    break

            if break_row is None:
                continue
            if break_row[0].time() >= NO_ENTRY_AFTER:
                continue

            ts, row = break_row
            # Scan remaining bars for SL (opposite OR breach, close-based)
            rest = post.loc[post.index > ts]
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

            eod_close = sess.iloc[-1]['close']
            signals.append({
                'date': day.date(),
                'or_min': win_min,
                'direction': break_dir,
                'entry_time': ts,
                'entry_price': float(row['close']),
                'or_high': float(or_high),
                'or_low': float(or_low),
                'or_width_pct': float(or_width / row['close'] * 100),
                'gap_pct': day_gap_pct,
                'rsi5m': float(row['rsi5m']) if not np.isnan(row['rsi5m']) else np.nan,
                'rsi15m': float(row['rsi15m']) if not np.isnan(row['rsi15m']) else np.nan,
                'above_sma50': bool(row['close'] > row['sma50']) if not np.isnan(row['sma50']) else None,
                'above_sma200': bool(row['close'] > row['sma200']) if not np.isnan(row['sma200']) else None,
                'macd_bullish': bool(row['macd_h'] > 0) if not np.isnan(row['macd_h']) else None,
                'sl_hit': sl_hit,
                'sl_time': sl_time,
                'eod_price': float(eod_close),
                'pnl_pct_underlying': (float(eod_close) / float(row['close']) - 1) * 100
                                      if break_dir == 'LONG'
                                      else (float(row['close']) / float(eod_close) - 1) * 100,
            })

        prev_close = sess.iloc[-1]['close']

    return pd.DataFrame(signals)


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------

def winrate_stats(df):
    n = len(df)
    if n == 0:
        return {'n': 0, 'wr': 0.0, 'avg_win_move': 0.0, 'avg_loss_move': 0.0}
    wins = df[~df['sl_hit']]
    losses = df[df['sl_hit']]
    return {
        'n': n,
        'wr': round(len(wins) / n * 100, 1),
        'avg_win_move': round(wins['pnl_pct_underlying'].mean(), 2) if len(wins) else 0,
        'avg_loss_move': round(losses['pnl_pct_underlying'].mean(), 2) if len(losses) else 0,
    }


def print_table(title, rows):
    print(f"\n{'-' * 90}\n{title}\n{'-' * 90}")
    print(f"{'Slice':40s} {'N':>6s} {'WinRate':>9s} {'AvgWinMv':>10s} {'AvgLossMv':>10s}")
    for label, s in rows:
        print(f"{label:40s} {s['n']:>6d} {s['wr']:>8.1f}% {s['avg_win_move']:>+9.2f}% {s['avg_loss_move']:>+9.2f}%")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Loading NIFTY50 5-min data...")
    df = load_nifty_5min()
    print(f"  {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(f"  Sessions: {df.index.normalize().nunique()}")

    print("Computing features (RSI, SMA, MACD, 15m RSI)...")
    df = add_features(df)

    print("Building signals for all OR windows...")
    sigs = build_signals(df)
    print(f"  Total raw signals: {len(sigs)}")
    sigs.to_csv('nifty_orb_signals.csv', index=False)

    # === 1. Baseline by OR TF ===
    rows = []
    for w in OR_WINDOWS:
        s = winrate_stats(sigs[sigs['or_min'] == w])
        rows.append((f"OR{w}m — all signals", s))
    print_table("1. Baseline by OR TF (no filter)", rows)

    # === 2. By direction per OR TF ===
    rows = []
    for w in OR_WINDOWS:
        for d in ['LONG', 'SHORT']:
            s = winrate_stats(sigs[(sigs['or_min'] == w) & (sigs['direction'] == d)])
            rows.append((f"OR{w}m / {d}", s))
    print_table("2. By OR TF x direction", rows)

    # === 3. RSI filter (5-min RSI) on OR15 (user's current production TF) ===
    base = sigs[sigs['or_min'] == 15]
    rows = []
    rows.append(("OR15 / no filter", winrate_stats(base)))
    for lo, hi in [(50, 50), (55, 45), (60, 40), (65, 35), (70, 30)]:
        m = ((base['direction'] == 'LONG') & (base['rsi5m'] > lo)) | \
            ((base['direction'] == 'SHORT') & (base['rsi5m'] < hi))
        rows.append((f"OR15 / RSI5m>{lo} L or <{hi} S", winrate_stats(base[m])))
    print_table("3. OR15 x 5-min RSI filter", rows)

    # === 4. RSI filter (15-min RSI, production style) on OR15 ===
    rows = []
    for lo, hi in [(50, 50), (55, 45), (60, 40), (65, 35), (70, 30)]:
        m = ((base['direction'] == 'LONG') & (base['rsi15m'] > lo)) | \
            ((base['direction'] == 'SHORT') & (base['rsi15m'] < hi))
        rows.append((f"OR15 / RSI15m>{lo} L or <{hi} S", winrate_stats(base[m])))
    print_table("4. OR15 x 15-min RSI filter", rows)

    # === 5. Same 5-min RSI filter across ALL OR TFs ===
    rows = []
    for w in OR_WINDOWS:
        b = sigs[sigs['or_min'] == w]
        m = ((b['direction'] == 'LONG') & (b['rsi5m'] > 60)) | \
            ((b['direction'] == 'SHORT') & (b['rsi5m'] < 40))
        rows.append((f"OR{w}m / RSI5m>60 L or <40 S", winrate_stats(b[m])))
    print_table("5. 5-min RSI>60/<40 filter across OR TFs", rows)

    # === 6. Gap filter on OR15 / RSI5m>60/<40 (best variant so far) ===
    b = sigs[sigs['or_min'] == 15]
    rsi_m = ((b['direction'] == 'LONG') & (b['rsi5m'] > 60)) | \
            ((b['direction'] == 'SHORT') & (b['rsi5m'] < 40))
    filt = b[rsi_m]
    rows = [("OR15 / RSI5m>60/<40 / gap ANY", winrate_stats(filt))]
    for cap in [0.5, 1.0, 1.5]:
        rows.append((f"OR15 / RSI5m / |gap|<={cap}%", winrate_stats(filt[filt['gap_pct'].abs() <= cap])))
        rows.append((f"OR15 / RSI5m / |gap|>{cap}%",  winrate_stats(filt[filt['gap_pct'].abs() > cap])))
    print_table("6. OR15 + RSI5m filter + gap slicing", rows)

    # === 7. MA alignment ===
    rows = []
    for w in [15]:
        b = sigs[sigs['or_min'] == w].copy()
        rsi_m = ((b['direction'] == 'LONG') & (b['rsi5m'] > 60)) | \
                ((b['direction'] == 'SHORT') & (b['rsi5m'] < 40))
        b = b[rsi_m]
        rows.append((f"OR{w} / RSI5m (base)", winrate_stats(b)))
        # SMA50 alignment
        ma50_m = ((b['direction'] == 'LONG') & (b['above_sma50'] == True)) | \
                 ((b['direction'] == 'SHORT') & (b['above_sma50'] == False))
        rows.append((f"OR{w} / RSI5m / +SMA50 align", winrate_stats(b[ma50_m])))
        # SMA200 alignment
        ma200_m = ((b['direction'] == 'LONG') & (b['above_sma200'] == True)) | \
                  ((b['direction'] == 'SHORT') & (b['above_sma200'] == False))
        rows.append((f"OR{w} / RSI5m / +SMA200 align", winrate_stats(b[ma200_m])))
        # MACD alignment
        macd_m = ((b['direction'] == 'LONG') & (b['macd_bullish'] == True)) | \
                 ((b['direction'] == 'SHORT') & (b['macd_bullish'] == False))
        rows.append((f"OR{w} / RSI5m / +MACD align", winrate_stats(b[macd_m])))
        # All three aligned
        all_m = ma50_m & ma200_m & macd_m
        rows.append((f"OR{w} / RSI5m / MA50+MA200+MACD all", winrate_stats(b[all_m])))
    print_table("7. Add MA / MACD alignment to OR15+RSI5m", rows)

    # === 8. Signal rate per session ===
    sess_count = df.index.normalize().nunique()
    rows = []
    for w in OR_WINDOWS:
        n = len(sigs[sigs['or_min'] == w])
        rows.append((f"OR{w}m signals/session", {
            'n': n, 'wr': round(n / sess_count * 100, 1),
            'avg_win_move': 0, 'avg_loss_move': 0,
        }))
    print(f"\n{'-'*90}\n8. Signal frequency (over {sess_count} sessions)\n{'-'*90}")
    for lab, s in rows:
        print(f"  {lab:40s} {s['n']:>6d}   ({s['wr']:4.1f}% of sessions)")

    # === 9. Time-to-SL for the losers ===
    b = sigs[(sigs['or_min'] == 15) & (sigs['sl_hit'])].copy()
    if len(b):
        # minutes between entry and sl
        b['mins_to_sl'] = (pd.to_datetime(b['sl_time']) - pd.to_datetime(b['entry_time'])).dt.total_seconds() / 60
        print(f"\n{'-'*90}\n9. OR15 losers: time-to-SL distribution (n={len(b)})\n{'-'*90}")
        print(f"  mean   {b['mins_to_sl'].mean():.0f} min")
        print(f"  median {b['mins_to_sl'].median():.0f} min")
        print(f"  p10 {b['mins_to_sl'].quantile(0.1):.0f} | p25 {b['mins_to_sl'].quantile(0.25):.0f} | "
              f"p75 {b['mins_to_sl'].quantile(0.75):.0f} | p90 {b['mins_to_sl'].quantile(0.9):.0f}")

    print("\nSaved raw signals: nifty_orb_signals.csv")


if __name__ == '__main__':
    main()
