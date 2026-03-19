"""
KC6 V2 vs Cost-Price Exit on Nifty 100 (Last 2 Years)
======================================================
Standard V2: Exit when high > KC6 mid, at KC6 mid price.
Cost-Price:  If KC6 mid < entry price at time of signal, DON'T exit.
             Wait until price returns to cost (entry) price, then exit.

Universe: Nifty 100 large caps
Period:   Last 2 years (~Feb 2024 to Feb 2026)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'

# Nifty 100 constituents (as of early 2026)
NIFTY_100 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
    'HINDUNILVR', 'BHARTIARTL', 'SBIN', 'ITC', 'BAJFINANCE',
    'KOTAKBANK', 'LT', 'AXISBANK', 'MARUTI', 'TITAN',
    'SUNPHARMA', 'HCLTECH', 'ASIANPAINT', 'NTPC', 'TATAMOTORS',
    'WIPRO', 'ULTRACEMCO', 'POWERGRID', 'NESTLEIND', 'TECHM',
    'JSWSTEEL', 'TATASTEEL', 'ADANIPORTS', 'BAJAJFINSV', 'ONGC',
    'HINDALCO', 'COALINDIA', 'INDUSINDBK', 'DIVISLAB', 'GRASIM',
    'CIPLA', 'EICHERMOT', 'DRREDDY', 'APOLLOHOSP', 'SBILIFE',
    'TATACONSUM', 'BRITANNIA', 'HEROMOTOCO', 'BAJAJ-AUTO', 'BPCL',
    'ADANIENT', 'HDFCLIFE', 'GODREJCP', 'DABUR', 'PIDILITIND',
    'M&M', 'SHREECEM', 'BERGEPAINT', 'HAVELLS', 'SIEMENS',
    'AMBUJACEM', 'ICICIGI', 'INDIGO', 'IOC', 'DLF',
    'ABB', 'VEDL', 'TRENT', 'BEL', 'TATAPOWER',
    'HINDPETRO', 'BANKBARODA', 'PNB', 'CHOLAFIN', 'GAIL',
    'ADANIGREEN', 'ADANIPOWER', 'TORNTPHARM', 'MARICO', 'MCDOWELL-N',
    'LTIM', 'SAIL', 'JINDALSTEL', 'HAL', 'BHEL',
    'RECLTD', 'PFC', 'CANBK', 'IRCTC', 'ZOMATO',
    'JIOFIN', 'ATGL', 'DMART', 'POLYCAB', 'MAXHEALTH',
    'LICI', 'PAYTM', 'UNIONBANK', 'IDBI', 'PERSISTENT',
    'IDFCFIRSTB', 'LTTS', 'BSE', 'PHOENIXLTD', 'YESBANK',
]


def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma(s, p):
    return s.rolling(window=p).mean()

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


def run_backtest(symbols, conn, sl_pct=5, tp_pct=15, max_hold=15,
                 start_date='2024-02-01', mode='v2_standard'):
    """
    Run KC6 backtest with two exit modes:

    mode='v2_standard': Normal V2 - exit at KC6 mid when high > KC6 mid
    mode='cost_price':  If KC6 mid < entry price, skip the signal exit.
                        Wait for close >= entry price, then exit at entry price.
    """
    all_trades = []

    for sym in symbols:
        df = pd.read_sql_query(
            "SELECT date,open,high,low,close,volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date",
            conn, params=[sym])
        if len(df) < 250:
            continue

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.astype(float)

        c = df['close']
        df['sma_200'] = sma(c, 200)
        df['kc6_m'], df['kc6_u'], df['kc6_l'] = keltner(df, 6, 6, 1.3)
        df['above200'] = c > df['sma_200']

        # Find the start index for our 2-year window
        # We need SMA200 warmup, so compute on all data but only trade from start_date
        start_dt = pd.Timestamp(start_date)
        trade_start_idx = None
        for idx in range(200, len(df)):
            if df.index[idx] >= start_dt:
                trade_start_idx = idx
                break

        if trade_start_idx is None:
            continue

        n = len(df)
        i = trade_start_idx

        while i < n:
            row = df.iloc[i]
            if row['close'] < row['kc6_l'] and row['above200']:
                entry_price = row['close']
                entry_i = i
                entry_date = df.index[i]

                i += 1
                while i < n:
                    cur = df.iloc[i]
                    hold = i - entry_i

                    low_pnl = (cur['low'] / entry_price - 1) * 100
                    high_pnl = (cur['high'] / entry_price - 1) * 100

                    exit_reason = None
                    exit_price = None

                    # SL (intraday low)
                    if low_pnl <= -sl_pct:
                        exit_reason = 'STOP_LOSS'
                        exit_price = round(entry_price * (1 - sl_pct / 100), 2)

                    # TP (intraday high)
                    elif high_pnl >= tp_pct:
                        exit_reason = 'TAKE_PROFIT'
                        exit_price = round(entry_price * (1 + tp_pct / 100), 2)

                    # Max hold
                    elif hold >= max_hold:
                        if mode == 'cost_price':
                            # Even at max hold, exit at close
                            exit_reason = 'MAX_HOLD'
                            exit_price = round(cur['close'], 2)
                        else:
                            exit_reason = 'MAX_HOLD'
                            exit_price = round(cur['close'], 2)

                    # V2 signal exit: high > KC6 mid
                    elif cur['high'] > cur['kc6_m']:
                        if mode == 'v2_standard':
                            # Standard: always exit at KC6 mid
                            exit_reason = 'SIGNAL_KC6_MID'
                            exit_price = round(cur['kc6_m'], 2)
                        elif mode == 'cost_price':
                            # Cost-price: only exit if KC6 mid >= entry price
                            if cur['kc6_m'] >= entry_price:
                                exit_reason = 'SIGNAL_KC6_MID'
                                exit_price = round(cur['kc6_m'], 2)
                            else:
                                # KC6 mid is below cost - check if close >= entry
                                if cur['close'] >= entry_price:
                                    exit_reason = 'COST_PRICE_EXIT'
                                    exit_price = round(entry_price, 2)
                                # else: don't exit, keep holding

                    if exit_reason:
                        pnl = (exit_price / entry_price - 1) * 100
                        all_trades.append({
                            'symbol': sym,
                            'entry_date': entry_date.strftime('%Y-%m-%d'),
                            'exit_date': df.index[i].strftime('%Y-%m-%d'),
                            'entry_price': round(entry_price, 2),
                            'exit_price': exit_price,
                            'pnl_pct': round(pnl, 2),
                            'hold_days': hold,
                            'exit_reason': exit_reason,
                            'win': pnl > 0,
                        })
                        i += 1
                        break
                    i += 1
            else:
                i += 1

    return pd.DataFrame(all_trades)


def print_stats(df_t, label, sl_pct=5):
    """Print detailed stats for a variant."""
    if len(df_t) == 0:
        print(f"\n  {label}: NO TRADES")
        return None

    total = len(df_t)
    wins = df_t['win'].sum()
    losses = total - wins
    wr = wins / total * 100
    avg_win = df_t[df_t['win']]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = df_t[~df_t['win']]['pnl_pct'].mean() if losses > 0 else 0
    avg_pnl = df_t['pnl_pct'].mean()
    avg_hold = df_t['hold_days'].mean()
    total_pnl = df_t['pnl_pct'].sum()
    max_dd = df_t['pnl_pct'].min()

    gross_profit = df_t[df_t['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(df_t[df_t['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    exp_r = (wr/100 * avg_win / sl_pct) + ((1 - wr/100) * avg_loss / sl_pct) if sl_pct > 0 else 0

    print(f"\n  {'='*90}")
    print(f"  {label}")
    print(f"  {'='*90}")
    print(f"  Total: {total} | Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
    print(f"  Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:+.2f}% | Avg P/L: {avg_pnl:+.2f}%")
    print(f"  Profit Factor: {pf:.2f} | R:R: {rr:.2f} | Expectancy: {exp_r:+.3f}R")
    print(f"  Avg Hold: {avg_hold:.1f}d | Total P/L: {total_pnl:+.1f}% | Worst Trade: {max_dd:+.2f}%")
    print(f"  Symbols traded: {df_t['symbol'].nunique()}")

    # Exit reason breakdown
    print(f"\n  Exit Breakdown:")
    for reason, grp in df_t.groupby('exit_reason'):
        cnt = len(grp)
        wr_r = grp['win'].sum() / cnt * 100 if cnt > 0 else 0
        avg_pnl_r = grp['pnl_pct'].mean()
        print(f"    {reason:>18s}: {cnt:>4d} ({cnt/total*100:>5.1f}%) | "
              f"WR: {wr_r:>5.1f}% | Avg P/L: {avg_pnl_r:>+6.2f}%")

    return {
        'label': label, 'total': total, 'wins': wins,
        'wr': round(wr, 1), 'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2), 'avg_pnl': round(avg_pnl, 2),
        'pf': round(pf, 2), 'rr': round(rr, 2), 'exp_r': round(exp_r, 3),
        'avg_hold': round(avg_hold, 1), 'total_pnl': round(total_pnl, 1),
    }


def main():
    conn = sqlite3.connect(DB_PATH)

    # Check which Nifty 100 stocks have data
    all_db_symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()

    available = [s for s in NIFTY_100 if s in all_db_symbols]
    missing = [s for s in NIFTY_100 if s not in all_db_symbols]

    start_date = '2024-02-01'

    print("=" * 100)
    print("KC6 V2 vs COST-PRICE EXIT -- NIFTY 100 (Last 2 Years)")
    print("=" * 100)
    print(f"Universe: Nifty 100 ({len(available)} with data, {len(missing)} missing)")
    print(f"Period: {start_date} to present (~2 years)")
    print(f"Entry: Close < KC6(6,1.3) Lower AND Close > SMA200")
    print(f"SL: 5% | TP: 15% | Max Hold: 15d")
    print()
    print(f"V2 Standard: Exit at KC6 mid when high > KC6 mid")
    print(f"Cost-Price:  If KC6 mid < entry price, wait for close >= entry price")
    if missing:
        print(f"\nMissing from DB: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
    print()

    # Run V2 Standard
    print("Running V2 Standard...")
    df_v2 = run_backtest(available, conn, sl_pct=5, tp_pct=15, max_hold=15,
                         start_date=start_date, mode='v2_standard')
    s_v2 = print_stats(df_v2, "V2 STANDARD (exit @ KC6 mid)")

    # Run Cost-Price variant
    print("\n\nRunning Cost-Price variant...")
    df_cp = run_backtest(available, conn, sl_pct=5, tp_pct=15, max_hold=15,
                         start_date=start_date, mode='cost_price')
    s_cp = print_stats(df_cp, "COST-PRICE (wait for cost if KC6 mid < entry)")

    conn.close()

    # Comparison table
    if s_v2 and s_cp:
        print(f"\n\n{'='*100}")
        print("COMPARISON TABLE: V2 Standard vs Cost-Price Exit")
        print(f"{'='*100}")
        header = (f"  {'Variant':<40s} {'Trades':>7s} {'WR%':>6s} {'AvgWin':>8s} {'AvgLoss':>9s} "
                  f"{'AvgP/L':>8s} {'PF':>6s} {'R:R':>6s} {'Exp(R)':>8s} {'Hold':>6s} {'TotalP/L':>10s}")
        print(header)
        print(f"  {'-'*95}")

        for r in [s_v2, s_cp]:
            print(f"  {r['label']:<40s} {r['total']:>7d} {r['wr']:>5.1f}% "
                  f"{r['avg_win']:>+7.2f}% {r['avg_loss']:>+8.2f}% "
                  f"{r['avg_pnl']:>+7.2f}% {r['pf']:>5.2f} {r['rr']:>5.2f} "
                  f"{r['exp_r']:>+7.3f}R {r['avg_hold']:>5.1f}d {r['total_pnl']:>+9.1f}%")

        # Delta analysis
        print(f"\n  Delta (Cost-Price vs V2 Standard):")
        print(f"    Win Rate:     {s_cp['wr'] - s_v2['wr']:+.1f}pp")
        print(f"    Avg P/L:      {s_cp['avg_pnl'] - s_v2['avg_pnl']:+.2f}%")
        print(f"    Profit Factor: {s_cp['pf'] - s_v2['pf']:+.2f}")
        print(f"    Expectancy:   {s_cp['exp_r'] - s_v2['exp_r']:+.3f}R")
        print(f"    Avg Hold:     {s_cp['avg_hold'] - s_v2['avg_hold']:+.1f}d")
        print(f"    Total P/L:    {s_cp['total_pnl'] - s_v2['total_pnl']:+.1f}%")

    # Show some sample trades unique to cost-price variant
    if len(df_cp) > 0:
        cp_exits = df_cp[df_cp['exit_reason'] == 'COST_PRICE_EXIT']
        if len(cp_exits) > 0:
            print(f"\n\n{'='*100}")
            print(f"SAMPLE COST-PRICE EXITS ({len(cp_exits)} total)")
            print(f"{'='*100}")
            print(f"  {'Symbol':<12s} {'Entry Date':>12s} {'Exit Date':>12s} {'Entry':>8s} "
                  f"{'Exit':>8s} {'P/L%':>7s} {'Hold':>5s}")
            print(f"  {'-'*70}")
            for _, t in cp_exits.head(15).iterrows():
                print(f"  {t['symbol']:<12s} {t['entry_date']:>12s} {t['exit_date']:>12s} "
                      f"{t['entry_price']:>8.2f} {t['exit_price']:>8.2f} "
                      f"{t['pnl_pct']:>+6.2f}% {t['hold_days']:>4d}d")


if __name__ == '__main__':
    main()
