"""
KC6 Intraday Backtest - Phase 1 POC
====================================
5-minute timeframe, longs + shorts, EOD squareoff, realistic MIS costs.

Universe (10 blue chips with 8+ years of 5-min data):
  BHARTIARTL, HDFCBANK, HINDUNILVR, ICICIBANK, INFY,
  ITC, KOTAKBANK, RELIANCE, SBIN, TCS

Period: 2018-01-01 -> 2026-03-13 (~8 years)

Signal:
  Long:  close < KC_lower  AND  close > SMA200
  Short: close > KC_upper  AND  close < SMA200

Exits (both sides):
  Primary: KC_mid touch
  SL: 1% from entry
  TP: 2% from entry
  EOD squareoff: close of last bar of the session

Costs: MIS round-trip ~0.20% (brokerage + STT + slippage)
"""

import sqlite3
import sys
from datetime import time as dtime

import numpy as np
import pandas as pd

DB = 'backtest_data/market_data.db'
UNIVERSE = ['BHARTIARTL', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY',
            'ITC', 'KOTAKBANK', 'RELIANCE', 'SBIN', 'TCS']

# Strategy params
KC_EMA = 6
KC_ATR = 6
KC_MULT = 1.3
SMA_PERIOD = 200
SL_PCT = 0.01   # 1%
TP_PCT = 0.02   # 2%
COST_RT = 0.0020  # 0.20% round-trip MIS cost


def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def sma(s, p):
    return s.rolling(p).mean()


def atr(h, l, c, p):
    hl = h - l
    hc = (h - c.shift(1)).abs()
    lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


def compute(df):
    df['kc_mid'] = ema(df['close'], KC_EMA)
    a = atr(df['high'], df['low'], df['close'], KC_ATR)
    df['kc_upper'] = df['kc_mid'] + KC_MULT * a
    df['kc_lower'] = df['kc_mid'] - KC_MULT * a
    df['sma200'] = sma(df['close'], SMA_PERIOD)
    return df


def load_symbol(conn, sym):
    q = (
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date"
    )
    df = pd.read_sql(q, conn, params=[sym])
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return compute(df)


def backtest_symbol(df, sym):
    """Walk the bars, open/close positions. Returns list of trade dicts."""
    trades = []
    position = None

    # For EOD squareoff: find last bar of each session day
    days = df.index.date
    last_bar_mask = np.zeros(len(df), dtype=bool)
    # last bar of a day = the row where next row's date differs (or end of data)
    for i in range(len(df) - 1):
        if days[i] != days[i + 1]:
            last_bar_mask[i] = True
    last_bar_mask[-1] = True

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    kc_lowers = df['kc_lower'].values
    kc_uppers = df['kc_upper'].values
    kc_mids = df['kc_mid'].values
    sma200s = df['sma200'].values
    timestamps = df.index

    for i in range(len(df)):
        # Skip until indicators warm up
        if np.isnan(sma200s[i]) or np.isnan(kc_lowers[i]):
            continue

        # Exit logic
        if position is not None:
            entry = position['entry']
            side = position['side']
            exit_price = None
            reason = None

            if side == 'LONG':
                sl = entry * (1 - SL_PCT)
                tp = entry * (1 + TP_PCT)
                mid = kc_mids[i]
                if lows[i] <= sl:
                    exit_price, reason = sl, 'SL'
                elif highs[i] >= tp:
                    exit_price, reason = tp, 'TP'
                elif not np.isnan(mid) and highs[i] >= mid and mid > entry:
                    exit_price, reason = mid, 'MID'
                elif last_bar_mask[i]:
                    exit_price, reason = closes[i], 'EOD'
            else:  # SHORT
                sl = entry * (1 + SL_PCT)
                tp = entry * (1 - TP_PCT)
                mid = kc_mids[i]
                if highs[i] >= sl:
                    exit_price, reason = sl, 'SL'
                elif lows[i] <= tp:
                    exit_price, reason = tp, 'TP'
                elif not np.isnan(mid) and lows[i] <= mid and mid < entry:
                    exit_price, reason = mid, 'MID'
                elif last_bar_mask[i]:
                    exit_price, reason = closes[i], 'EOD'

            if exit_price is not None:
                if side == 'LONG':
                    gross = exit_price / entry - 1
                else:
                    gross = entry / exit_price - 1
                net = gross - COST_RT
                trades.append({
                    'symbol': sym,
                    'side': side,
                    'entry_time': position['entry_time'],
                    'exit_time': timestamps[i],
                    'entry': round(entry, 2),
                    'exit': round(exit_price, 2),
                    'gross_pct': gross,
                    'net_pct': net,
                    'reason': reason,
                })
                position = None

        # Entry logic: only if flat and not last bar of the day
        if position is None and not last_bar_mask[i]:
            c = closes[i]
            lo = kc_lowers[i]
            up = kc_uppers[i]
            s200 = sma200s[i]
            if c < lo and c > s200:
                position = {'side': 'LONG', 'entry': c, 'entry_time': timestamps[i]}
            elif c > up and c < s200:
                position = {'side': 'SHORT', 'entry': c, 'entry_time': timestamps[i]}

    return trades


def stats(df, label):
    n = len(df)
    if n == 0:
        print(f"{label:24s} N=    0")
        return
    wins = df[df['net_pct'] > 0]
    losses = df[df['net_pct'] <= 0]
    wr = len(wins) / n
    avg_w = wins['net_pct'].mean() if len(wins) else 0
    avg_l = losses['net_pct'].mean() if len(losses) else 0
    gross = df['gross_pct'].sum()
    net = df['net_pct'].sum()
    gsum_w = wins['net_pct'].sum()
    gsum_l = losses['net_pct'].sum()
    pf = -gsum_w / gsum_l if gsum_l < 0 else float('inf')
    print(
        f"{label:24s} N={n:6d}  WR={wr*100:5.1f}%  "
        f"AvgW={avg_w*100:+.2f}%  AvgL={avg_l*100:+.2f}%  "
        f"Gross={gross*100:+7.1f}%  Net={net*100:+7.1f}%  PF={pf:4.2f}"
    )


def main():
    conn = sqlite3.connect(DB)
    all_trades = []

    for sym in UNIVERSE:
        print(f"[{sym:12s}] loading...", end=' ', flush=True)
        df = load_symbol(conn, sym)
        if len(df) < 300:
            print(f"skip ({len(df)} bars)")
            continue
        date_min = df.index.min().strftime('%Y-%m-%d')
        date_max = df.index.max().strftime('%Y-%m-%d')
        print(f"{len(df):6d} bars  {date_min}->{date_max}  backtest...", end=' ', flush=True)
        trades = backtest_symbol(df, sym)
        print(f"{len(trades):5d} trades")
        all_trades.extend(trades)

    conn.close()

    if not all_trades:
        print("\nNO TRADES. Check data/signal.")
        return

    df = pd.DataFrame(all_trades)
    print(f"\n{'='*100}")
    print(f"PHASE 1 RESULTS  -  {len(df)} total trades")
    print('='*100)
    print(f"{'segment':24s} {'N':>8s}  {'WR':>7s}  {'AvgW':>9s}  {'AvgL':>9s}  {'Gross':>10s}  {'Net':>10s}  {'PF':>5s}")
    print('-'*100)

    stats(df, 'ALL')
    stats(df[df['side'] == 'LONG'], 'LONG')
    stats(df[df['side'] == 'SHORT'], 'SHORT')

    print("\nBy exit reason:")
    for reason in ['MID', 'SL', 'TP', 'EOD']:
        stats(df[df['reason'] == reason], f'  {reason}')

    print("\nBy symbol (ALL):")
    for sym in sorted(df['symbol'].unique()):
        stats(df[df['symbol'] == sym], f'  {sym}')

    print("\nBy symbol x side:")
    for sym in sorted(df['symbol'].unique()):
        for side in ['LONG', 'SHORT']:
            sub = df[(df['symbol'] == sym) & (df['side'] == side)]
            stats(sub, f'  {sym} {side}')

    # Yearly breakdown
    df['year'] = pd.to_datetime(df['exit_time']).dt.year
    print("\nBy year:")
    for y in sorted(df['year'].unique()):
        stats(df[df['year'] == y], f'  {y}')

    # Save
    out = 'backtest_kc6_intraday_5min_results.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
