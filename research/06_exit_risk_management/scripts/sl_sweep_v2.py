"""
SL Sweep for V2 (Limit Order) Model
Test SL values: 3%, 4%, 5%, 6%, 8% with V2 exit (high > KC6 mid, exit @ KC6 mid)
Keep TP=15%, Max Hold=15 days constant.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
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


def run_backtest(symbols, conn, sl_pct=8, tp_pct=15, max_hold=15):
    """Run V2 backtest with configurable SL."""
    all_trades = []

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
        df['above200'] = c > df['sma_200']

        n = len(df)
        i = 200

        while i < n:
            row = df.iloc[i]
            if row['close'] < row['kc6_l'] and row['above200']:
                entry_price = row['close']
                entry_i = i

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
                        exit_reason = 'MAX_HOLD'
                        exit_price = round(cur['close'], 2)
                    # V2 limit exit: high > KC6 mid
                    elif cur['high'] > cur['kc6_m']:
                        exit_reason = 'SIGNAL_KC6_MID'
                        exit_price = round(cur['kc6_m'], 2)

                    if exit_reason:
                        pnl = (exit_price / entry_price - 1) * 100
                        all_trades.append({
                            'symbol': sym,
                            'entry_price': entry_price,
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


def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()

    sl_values = [3, 4, 5, 6, 8]
    tp_pct = 15
    max_hold = 15

    print("=" * 120)
    print("SL SWEEP: V2 Limit Order Model (KC6 Lower + SMA200)")
    print("Entry: Close < KC6(6, 1.3) Lower AND Close > SMA200")
    print("Exit: High > KC6 Mid -> exit @ KC6 mid (limit order)")
    print(f"TP: {tp_pct}% | Max Hold: {max_hold}d | SL: testing {sl_values}")
    print("=" * 120)

    results = []

    for sl in sl_values:
        print(f"\nRunning SL={sl}%...")
        df_t = run_backtest(symbols, conn, sl_pct=sl, tp_pct=tp_pct, max_hold=max_hold)

        total = len(df_t)
        wins = df_t['win'].sum()
        wr = wins / total * 100 if total > 0 else 0
        avg_win = df_t[df_t['win']]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = df_t[~df_t['win']]['pnl_pct'].mean() if (total - wins) > 0 else 0
        avg_pnl = df_t['pnl_pct'].mean()
        avg_hold = df_t['hold_days'].mean()
        total_pnl = df_t['pnl_pct'].sum()

        gross_profit = df_t[df_t['pnl_pct'] > 0]['pnl_pct'].sum()
        gross_loss = abs(df_t[df_t['pnl_pct'] < 0]['pnl_pct'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Exit reason counts
        sl_exits = len(df_t[df_t['exit_reason'] == 'STOP_LOSS'])
        tp_exits = len(df_t[df_t['exit_reason'] == 'TAKE_PROFIT'])
        sig_exits = len(df_t[df_t['exit_reason'] == 'SIGNAL_KC6_MID'])
        mh_exits = len(df_t[df_t['exit_reason'] == 'MAX_HOLD'])

        # Risk/reward ratio (avg win / abs avg loss)
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Expected value per R
        # If we define 1R = SL%, then avg win in R = avg_win / sl, avg loss in R = avg_loss / sl
        avg_win_r = avg_win / sl if sl > 0 else 0
        avg_loss_r = avg_loss / sl if sl > 0 else 0
        expectancy_r = (wr/100 * avg_win_r) + ((1 - wr/100) * avg_loss_r)

        r = {
            'sl_pct': sl,
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': round(wr, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_pnl': round(avg_pnl, 2),
            'avg_hold': round(avg_hold, 1),
            'profit_factor': round(pf, 2),
            'total_pnl': round(total_pnl, 1),
            'rr_ratio': round(rr, 2),
            'expectancy_r': round(expectancy_r, 3),
            'sl_exits': sl_exits,
            'tp_exits': tp_exits,
            'sig_exits': sig_exits,
            'mh_exits': mh_exits,
            'sl_exit_pct': round(sl_exits / total * 100, 1),
        }
        results.append(r)

        print(f"  SL={sl}%: {total} trades | WR={wr:.1f}% | PF={pf:.2f} | "
              f"Avg P/L={avg_pnl:+.2f}% | Hold={avg_hold:.1f}d | "
              f"R:R={rr:.2f} | Expectancy={expectancy_r:.3f}R | "
              f"SL exits={sl_exits} ({sl_exits/total*100:.1f}%)")

    conn.close()

    # Summary table
    print(f"\n\n{'='*120}")
    print("SL SWEEP COMPARISON TABLE")
    print(f"{'='*120}")
    header = (f"  {'SL%':>4s} | {'Trades':>6s} | {'WR%':>5s} | {'Avg Win':>8s} | "
              f"{'Avg Loss':>9s} | {'Avg P/L':>8s} | {'PF':>5s} | {'R:R':>5s} | "
              f"{'Exp(R)':>7s} | {'Hold':>5s} | {'SL Exits':>9s} | {'Total P/L':>10s}")
    print(header)
    print(f"  {'-'*115}")
    for r in results:
        print(f"  {r['sl_pct']:>4d}% | {r['total_trades']:>6d} | {r['win_rate']:>5.1f} | "
              f"{r['avg_win']:>+8.2f}% | {r['avg_loss']:>+8.2f}% | {r['avg_pnl']:>+8.2f}% | "
              f"{r['profit_factor']:>5.2f} | {r['rr_ratio']:>5.2f} | "
              f"{r['expectancy_r']:>+7.3f} | {r['avg_hold']:>5.1f}d | "
              f"{r['sl_exits']:>4d} ({r['sl_exit_pct']:>4.1f}%) | {r['total_pnl']:>+10.1f}%")

    # Best by each metric
    print(f"\n{'='*120}")
    print("BEST BY METRIC:")
    print(f"{'='*120}")

    best_wr = max(results, key=lambda x: x['win_rate'])
    best_pf = max(results, key=lambda x: x['profit_factor'])
    best_avg = max(results, key=lambda x: x['avg_pnl'])
    best_total = max(results, key=lambda x: x['total_pnl'])
    best_exp = max(results, key=lambda x: x['expectancy_r'])
    best_rr = max(results, key=lambda x: x['rr_ratio'])

    print(f"  Best Win Rate:     SL={best_wr['sl_pct']}% -> {best_wr['win_rate']:.1f}%")
    print(f"  Best Profit Factor: SL={best_pf['sl_pct']}% -> {best_pf['profit_factor']:.2f}")
    print(f"  Best Avg P/L:      SL={best_avg['sl_pct']}% -> {best_avg['avg_pnl']:+.2f}%")
    print(f"  Best Total P/L:    SL={best_total['sl_pct']}% -> {best_total['total_pnl']:+.1f}%")
    print(f"  Best Expectancy:   SL={best_exp['sl_pct']}% -> {best_exp['expectancy_r']:+.3f}R")
    print(f"  Best R:R ratio:    SL={best_rr['sl_pct']}% -> {best_rr['rr_ratio']:.2f}")

    # The key insight: what's the tradeoff?
    print(f"\n{'='*120}")
    print("KEY TRADEOFF ANALYSIS:")
    print(f"{'='*120}")
    for r in results:
        sl_cost = r['sl_exits'] * r['sl_pct']  # total % lost to SL exits
        sig_gain = r['sig_exits'] * r['avg_win'] if r['avg_win'] > 0 else 0
        print(f"  SL={r['sl_pct']}%: Trades chopped by SL = {r['sl_exits']} "
              f"(cost ~{sl_cost:.0f}% total) | "
              f"Signal wins preserved = {r['sig_exits']} | "
              f"Net: {r['total_pnl']:+.1f}%")


if __name__ == '__main__':
    main()
