"""
Verify Strategy #1: Keltner(6, 1.3 ATR) Lower Band + SMA200
Compare two exit modes:
  V1 (Original): Exit when close > KC6 mid, exit price = close
  V2 (Limit Order): Exit when high > KC6 mid, exit price = KC6 mid
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(window=p).mean()

def atr(df, p=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def keltner(df, ep=6, ap=6, m=1.3):
    mid = ema(df['close'], ep)
    a = atr(df, ap)
    return mid, mid + m * a, mid - m * a

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    ag = g.ewm(span=p, adjust=False).mean()
    al = l.ewm(span=p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def run_backtest(symbols, conn, use_limit_exit=False):
    """Run backtest with either close-based or high-based (limit order) exit."""
    all_trades = []
    mode = "LIMIT (high > KC6 mid, exit @ KC6 mid)" if use_limit_exit else "CLOSE (close > KC6 mid, exit @ close)"

    for sym in symbols:
        df = pd.read_sql_query(
            "SELECT date,open,high,low,close,volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date",
            conn, params=[sym])
        if len(df) < 200:
            continue
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.astype(float)

        c = df['close']
        df['sma_200'] = sma(c, 200)
        df['kc6_m'], df['kc6_u'], df['kc6_l'] = keltner(df, 6, 6, 1.3)
        df['rsi_14'] = rsi(c, 14)
        df['atr_6'] = atr(df, 6)
        df['above200'] = c > df['sma_200']

        sl_pct = 8; tp_pct = 15; max_hold = 15
        n = len(df)
        i = 200

        while i < n:
            row = df.iloc[i]
            if row['close'] < row['kc6_l'] and row['above200']:
                entry_price = row['close']
                entry_date = df.index[i]
                entry_i = i

                entry_info = {
                    'symbol': sym,
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'entry_price': round(entry_price, 2),
                    'entry_kc6_lower': round(row['kc6_l'], 2),
                    'entry_kc6_mid': round(row['kc6_m'], 2),
                    'entry_sma200': round(row['sma_200'], 2),
                    'entry_rsi14': round(row['rsi_14'], 2),
                    'entry_atr6': round(row['atr_6'], 2),
                    'entry_atr_pct': round(row['atr_6'] / row['close'] * 100, 2),
                }

                i += 1
                while i < n:
                    cur = df.iloc[i]
                    hold = i - entry_i

                    # For SL/TP checks, use intraday low/high for realism
                    low_pnl = (cur['low'] / entry_price - 1) * 100
                    high_pnl = (cur['high'] / entry_price - 1) * 100

                    exit_reason = None
                    exit_price = None

                    # Check SL first (worst case intraday)
                    if low_pnl <= -sl_pct:
                        exit_reason = 'STOP_LOSS'
                        exit_price = round(entry_price * (1 - sl_pct / 100), 2)
                    # Check TP (best case intraday)
                    elif high_pnl >= tp_pct:
                        exit_reason = 'TAKE_PROFIT'
                        exit_price = round(entry_price * (1 + tp_pct / 100), 2)
                    # Max hold
                    elif hold >= max_hold:
                        exit_reason = 'MAX_HOLD'
                        exit_price = round(cur['close'], 2)
                    # Signal exit — the key difference
                    elif use_limit_exit:
                        # V2: high touches KC6 mid → exit at KC6 mid price (limit order)
                        if cur['high'] > cur['kc6_m']:
                            exit_reason = 'SIGNAL_KC6_MID'
                            exit_price = round(cur['kc6_m'], 2)
                    else:
                        # V1: close must be above KC6 mid → exit at close
                        if cur['close'] > cur['kc6_m']:
                            exit_reason = 'SIGNAL_KC6_MID'
                            exit_price = round(cur['close'], 2)

                    if exit_reason:
                        pnl = (exit_price / entry_price - 1) * 100
                        trade = {
                            **entry_info,
                            'exit_date': df.index[i].strftime('%Y-%m-%d'),
                            'exit_price': exit_price,
                            'exit_kc6_mid': round(cur['kc6_m'], 2),
                            'hold_days': hold,
                            'exit_reason': exit_reason,
                            'pnl_pct': round(pnl, 2),
                            'pnl_abs': round(exit_price - entry_price, 2),
                            'win': pnl > 0,
                        }
                        all_trades.append(trade)
                        i += 1
                        break
                    i += 1
            else:
                i += 1

    return pd.DataFrame(all_trades)


def print_stats(df_trades, label):
    """Print summary stats for a set of trades."""
    df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
    df_trades = df_trades.sort_values('entry_date', ascending=False)

    total = len(df_trades)
    wins = df_trades['win'].sum()
    losses = total - wins
    wr = wins / total * 100
    avg_win = df_trades[df_trades['win']]['pnl_pct'].mean()
    avg_loss = df_trades[~df_trades['win']]['pnl_pct'].mean()
    avg_hold = df_trades['hold_days'].mean()
    total_pnl = df_trades['pnl_pct'].sum()
    avg_pnl = df_trades['pnl_pct'].mean()

    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")
    print(f"  Total: {total} | Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
    print(f"  Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:+.2f}% | Avg P/L: {avg_pnl:+.2f}%")
    print(f"  Avg Hold: {avg_hold:.1f}d | Symbols: {df_trades['symbol'].nunique()}")
    print(f"  Total Cumulative P/L: {total_pnl:+.1f}%")

    # Profit factor
    gross_profit = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    print(f"  Profit Factor: {pf:.2f}")

    # Exit reason breakdown
    print(f"\n  Exit Breakdown:")
    for reason, grp in df_trades.groupby('exit_reason'):
        cnt = len(grp)
        wr_r = grp['win'].sum() / cnt * 100
        avg_pnl_r = grp['pnl_pct'].mean()
        avg_hold_r = grp['hold_days'].mean()
        print(f"    {reason:>15s}: {cnt:>4d} ({cnt/total*100:>5.1f}%) | WR: {wr_r:>5.1f}% | Avg P/L: {avg_pnl_r:>+6.2f}% | Hold: {avg_hold_r:.1f}d")

    # Win rate by year (last 5 years)
    df_trades['year'] = df_trades['entry_date'].dt.year
    print(f"\n  Win Rate by Year (recent):")
    for yr, grp in df_trades.groupby('year'):
        if yr >= 2021:
            cnt = len(grp)
            wr_y = grp['win'].sum() / cnt * 100
            avg_pnl_y = grp['pnl_pct'].mean()
            print(f"    {yr}: {cnt:>4d} trades | WR: {wr_y:>5.1f}% | Avg P/L: {avg_pnl_y:>+6.2f}%")

    return {
        'total': total, 'wins': wins, 'wr': wr, 'avg_win': avg_win,
        'avg_loss': avg_loss, 'avg_pnl': avg_pnl, 'avg_hold': avg_hold,
        'pf': pf, 'total_pnl': total_pnl
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()

    print("=" * 100)
    print("KC6 + SMA200 EXIT MODE COMPARISON")
    print("Entry: Close < KC6(6, 1.3 ATR) Lower AND Close > SMA200")
    print("SL: 8% | TP: 15% | Max Hold: 15 days")
    print("=" * 100)

    # Run V1: Original (close > KC6 mid)
    print("\nRunning V1 (close-based exit)...")
    df_v1 = run_backtest(symbols, conn, use_limit_exit=False)
    s1 = print_stats(df_v1.copy(), "V1: EXIT when CLOSE > KC6 Mid (exit price = close)")

    # Run V2: Limit order (high > KC6 mid, exit at KC6 mid)
    print("\nRunning V2 (limit order exit)...")
    df_v2 = run_backtest(symbols, conn, use_limit_exit=True)
    s2 = print_stats(df_v2.copy(), "V2: EXIT when HIGH > KC6 Mid (exit price = KC6 mid)")

    conn.close()

    # Side-by-side comparison
    print(f"\n\n{'='*100}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*100}")
    print(f"  {'Metric':<25s} {'V1 (Close)':<20s} {'V2 (Limit)':<20s} {'Delta':<15s}")
    print(f"  {'-'*80}")
    print(f"  {'Total Trades':<25s} {s1['total']:<20d} {s2['total']:<20d} {s2['total']-s1['total']:+d}")
    print(f"  {'Win Rate':<25s} {s1['wr']:<20.1f} {s2['wr']:<20.1f} {s2['wr']-s1['wr']:+.1f}%")
    print(f"  {'Avg Win':<25s} {s1['avg_win']:<+20.2f} {s2['avg_win']:<+20.2f} {s2['avg_win']-s1['avg_win']:+.2f}%")
    print(f"  {'Avg Loss':<25s} {s1['avg_loss']:<+20.2f} {s2['avg_loss']:<+20.2f} {s2['avg_loss']-s1['avg_loss']:+.2f}%")
    print(f"  {'Avg P/L per trade':<25s} {s1['avg_pnl']:<+20.2f} {s2['avg_pnl']:<+20.2f} {s2['avg_pnl']-s1['avg_pnl']:+.2f}%")
    print(f"  {'Avg Hold Days':<25s} {s1['avg_hold']:<20.1f} {s2['avg_hold']:<20.1f} {s2['avg_hold']-s1['avg_hold']:+.1f}d")
    print(f"  {'Profit Factor':<25s} {s1['pf']:<20.2f} {s2['pf']:<20.2f} {s2['pf']-s1['pf']:+.2f}")
    print(f"  {'Total Cumulative P/L':<25s} {s1['total_pnl']:<+20.1f} {s2['total_pnl']:<+20.1f} {s2['total_pnl']-s1['total_pnl']:+.1f}%")


if __name__ == '__main__':
    main()
