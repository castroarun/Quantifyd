"""
TTM Squeeze Backtest on Nifty 500 Universe
===========================================
Tests John Carter's TTM Squeeze strategy on our 20-year Indian stock data.

Strategy Rules (from Carter):
  Entry: 5+ consecutive squeeze-ON bars, then first squeeze-OFF bar,
         momentum histogram positive and rising (long only).
         Price must be above SMA(200) for trend filter.
  Exit:  Two consecutive momentum color-change bars (momentum decreasing),
         OR momentum crosses zero line,
         OR max hold 10 bars (Carter's 8-10 bar rule),
         OR stop loss at consolidation range low.

We also test a simplified version and compare with our KC6 mean reversion.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'


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


def bollinger_bands(df, length=20, mult=2.0):
    mid = sma(df['close'], length)
    std = df['close'].rolling(window=length).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower


def keltner_channels(df, length=20, atr_mult=1.5):
    mid = ema(df['close'], length)
    a = atr(df, length)
    upper = mid + atr_mult * a
    lower = mid - atr_mult * a
    return mid, upper, lower


def momentum_simple(df, length=12):
    """Carter's momentum: close minus average of (HL midpoint, SMA).
    Positive = bullish, negative = bearish. Rising = strengthening."""
    highest = df['high'].rolling(window=length).max()
    lowest = df['low'].rolling(window=length).min()
    midpoint = (highest + lowest) / 2
    sma_mid = sma(df['close'], length)
    return df['close'] - (midpoint + sma_mid) / 2


def run_ttm_squeeze_backtest(symbols, conn, min_squeeze_bars=5, max_hold=10,
                              sl_pct=5, use_sma200_filter=True,
                              bb_length=20, bb_mult=2.0,
                              kc_length=20, kc_mult=1.5, mom_length=12):
    """Run TTM Squeeze backtest with Carter's rules."""
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

        # Compute indicators
        bb_mid, bb_upper, bb_lower = bollinger_bands(df, bb_length, bb_mult)
        kc_mid, kc_upper, kc_lower = keltner_channels(df, kc_length, kc_mult)
        df['sma_200'] = sma(df['close'], 200)

        # Squeeze detection: BB inside KC
        df['squeeze_on'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)

        # Momentum (simplified for speed)
        df['momentum'] = momentum_simple(df, mom_length)
        df['mom_prev'] = df['momentum'].shift(1)

        # Momentum histogram colors
        # Cyan: above 0 and increasing, Dark blue: above 0 and decreasing
        # Red: below 0 and decreasing, Yellow: below 0 and increasing
        df['mom_rising'] = df['momentum'] > df['mom_prev']

        # Count consecutive squeeze bars
        squeeze_count = []
        count = 0
        for sq in df['squeeze_on']:
            if sq:
                count += 1
            else:
                count = 0
            squeeze_count.append(count)
        df['squeeze_count'] = squeeze_count

        # Previous bar was squeeze ON, current bar is squeeze OFF = "squeeze fires"
        df['squeeze_fires'] = (~df['squeeze_on']) & (df['squeeze_on'].shift(1) == True)
        # Get the squeeze count from the bar before the fire
        df['fire_squeeze_len'] = df['squeeze_count'].shift(1)

        n = len(df)
        i = 200

        while i < n:
            row = df.iloc[i]

            # Entry conditions
            if (row['squeeze_fires'] and
                row['fire_squeeze_len'] >= min_squeeze_bars and
                row['momentum'] > 0 and
                row['mom_rising']):

                # Optional SMA200 filter
                if use_sma200_filter and row['close'] <= row['sma_200']:
                    i += 1
                    continue

                entry_price = row['close']
                entry_i = i
                entry_date = df.index[i]

                # Consolidation range for stop loss (low of squeeze period)
                lookback = int(row['fire_squeeze_len']) + 2
                consol_low = df['low'].iloc[max(0, i - lookback):i + 1].min()
                sl_price_consol = consol_low
                sl_price_pct = entry_price * (1 - sl_pct / 100)
                sl_price = max(sl_price_consol, sl_price_pct)  # Use tighter of the two

                i += 1
                while i < n:
                    cur = df.iloc[i]
                    hold = i - entry_i

                    exit_reason = None
                    exit_price = None

                    # SL check (intraday low)
                    if cur['low'] <= sl_price:
                        exit_reason = 'STOP_LOSS'
                        exit_price = round(sl_price, 2)

                    # Momentum crosses zero against position
                    elif cur['momentum'] < 0:
                        exit_reason = 'MOM_ZERO_CROSS'
                        exit_price = round(cur['close'], 2)

                    # Two consecutive decreasing momentum bars (above zero but fading)
                    elif hold >= 2:
                        prev1 = df.iloc[i - 1]
                        if (not cur['mom_rising'] and not prev1['mom_rising']
                            and cur['momentum'] > 0):
                            exit_reason = 'MOM_FADING'
                            exit_price = round(cur['close'], 2)

                    # Max hold
                    elif hold >= max_hold:
                        exit_reason = 'MAX_HOLD'
                        exit_price = round(cur['close'], 2)

                    if exit_reason:
                        pnl = (exit_price / entry_price - 1) * 100
                        all_trades.append({
                            'symbol': sym,
                            'entry_date': entry_date.strftime('%Y-%m-%d'),
                            'entry_price': round(entry_price, 2),
                            'exit_date': df.index[i].strftime('%Y-%m-%d'),
                            'exit_price': exit_price,
                            'pnl_pct': round(pnl, 2),
                            'hold_days': hold,
                            'exit_reason': exit_reason,
                            'win': pnl > 0,
                            'squeeze_bars': int(row['fire_squeeze_len']),
                        })
                        i += 1
                        break
                    i += 1
            else:
                i += 1

    return pd.DataFrame(all_trades)


def print_stats(df_t, label):
    """Print summary statistics."""
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

    gross_profit = df_t[df_t['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(df_t[df_t['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"{'=' * 100}")
    print(f"  Total: {total} | Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
    print(f"  Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:+.2f}% | Avg P/L: {avg_pnl:+.2f}%")
    print(f"  Avg Hold: {avg_hold:.1f}d | Profit Factor: {pf:.2f} | R:R: {rr:.2f}")
    print(f"  Total Cumulative P/L: {total_pnl:+.1f}%")
    print(f"  Symbols: {df_t['symbol'].nunique()}")

    # Exit reason breakdown
    print(f"\n  Exit Breakdown:")
    for reason, grp in df_t.groupby('exit_reason'):
        cnt = len(grp)
        wr_r = grp['win'].sum() / cnt * 100
        avg_pnl_r = grp['pnl_pct'].mean()
        print(f"    {reason:>15s}: {cnt:>4d} ({cnt / total * 100:>5.1f}%) | "
              f"WR: {wr_r:>5.1f}% | Avg P/L: {avg_pnl_r:>+6.2f}%")

    # Squeeze duration analysis
    if 'squeeze_bars' in df_t.columns:
        print(f"\n  Win Rate by Squeeze Duration:")
        for sq_len, grp in df_t.groupby(pd.cut(df_t['squeeze_bars'], bins=[0, 5, 8, 12, 20, 100])):
            if len(grp) > 5:
                cnt = len(grp)
                wr_s = grp['win'].sum() / cnt * 100
                avg_pnl_s = grp['pnl_pct'].mean()
                print(f"    {str(sq_len):>15s} bars: {cnt:>4d} trades | "
                      f"WR: {wr_s:>5.1f}% | Avg P/L: {avg_pnl_s:>+6.2f}%")

    # Win rate by year (recent)
    df_t['entry_date'] = pd.to_datetime(df_t['entry_date'])
    df_t['year'] = df_t['entry_date'].dt.year
    print(f"\n  Win Rate by Year:")
    for yr, grp in df_t.groupby('year'):
        if yr >= 2008:
            cnt = len(grp)
            if cnt >= 5:
                wr_y = grp['win'].sum() / cnt * 100
                avg_pnl_y = grp['pnl_pct'].mean()
                print(f"    {yr}: {cnt:>4d} trades | WR: {wr_y:>5.1f}% | Avg P/L: {avg_pnl_y:>+6.2f}%")

    return {
        'total': total, 'wins': wins, 'wr': wr,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'avg_pnl': avg_pnl, 'avg_hold': avg_hold,
        'pf': pf, 'rr': rr, 'total_pnl': total_pnl,
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()

    print("=" * 100)
    print("TTM SQUEEZE BACKTEST -- NIFTY 500 UNIVERSE (20-YEAR DATA)")
    print("=" * 100)
    print(f"Symbols: {len(symbols)} | Data: 2000-2025")
    print()

    # Test 1: Carter's exact rules (5+ squeeze bars, SMA200 filter)
    print("\n[TEST 1] Carter's Rules: 5+ squeeze bars, SMA200 filter, SL=5%")
    print("  BB(20, 2.0) vs KC(20, 1.5) | Momentum(12) | Max hold 10d")
    df1 = run_ttm_squeeze_backtest(
        symbols, conn, min_squeeze_bars=5, max_hold=10, sl_pct=5,
        use_sma200_filter=True)
    s1 = print_stats(df1, "TTM Squeeze - Carter's Rules (SMA200 filter)")

    # Test 2: Without SMA200 filter (pure squeeze)
    print("\n\n[TEST 2] Pure Squeeze: 5+ bars, NO SMA200 filter, SL=5%")
    df2 = run_ttm_squeeze_backtest(
        symbols, conn, min_squeeze_bars=5, max_hold=10, sl_pct=5,
        use_sma200_filter=False)
    s2 = print_stats(df2, "TTM Squeeze - No SMA200 Filter")

    # Test 3: Relaxed squeeze (3+ bars)
    print("\n\n[TEST 3] Relaxed: 3+ squeeze bars, SMA200 filter, SL=5%")
    df3 = run_ttm_squeeze_backtest(
        symbols, conn, min_squeeze_bars=3, max_hold=10, sl_pct=5,
        use_sma200_filter=True)
    s3 = print_stats(df3, "TTM Squeeze - 3+ Bars (SMA200 filter)")

    # Test 4: Tighter BB (1.5 std dev as Carter suggests for more signals)
    print("\n\n[TEST 4] Tight BB(20, 1.5) vs KC(20, 1.5), 5+ bars, SMA200, SL=5%")
    df4 = run_ttm_squeeze_backtest(
        symbols, conn, min_squeeze_bars=5, max_hold=10, sl_pct=5,
        use_sma200_filter=True, bb_mult=1.5)
    s4 = print_stats(df4, "TTM Squeeze - BB(1.5) Variant")

    conn.close()

    # Comparison table
    print(f"\n\n{'=' * 100}")
    print("COMPARISON TABLE")
    print(f"{'=' * 100}")
    print(f"  {'Variant':<35s} {'Trades':>7s} {'WR%':>6s} {'AvgWin':>8s} {'AvgLoss':>8s} "
          f"{'AvgP/L':>8s} {'PF':>6s} {'R:R':>6s} {'Hold':>6s} {'TotalP/L':>10s}")
    print(f"  {'-' * 95}")

    for label, stats in [
        ("Carter Rules (SMA200)", s1),
        ("No SMA200 Filter", s2),
        ("3+ Squeeze Bars", s3),
        ("BB(1.5) Variant", s4),
    ]:
        if stats:
            print(f"  {label:<35s} {stats['total']:>7d} {stats['wr']:>5.1f}% "
                  f"{stats['avg_win']:>+7.2f}% {stats['avg_loss']:>+7.2f}% "
                  f"{stats['avg_pnl']:>+7.2f}% {stats['pf']:>5.2f} {stats['rr']:>5.2f} "
                  f"{stats['avg_hold']:>5.1f}d {stats['total_pnl']:>+9.1f}%")

    # Reference: our KC6 strategy
    print(f"\n  {'--- For Reference ---':<35s}")
    print(f"  {'KC6+SMA200 V2 (SL=5%)':<35s} {'1942':>7s} {'64.7%':>6s} "
          f"{'+4.42%':>8s} {'-4.51%':>8s} {'+1.27%':>8s} {'1.79':>6s} {'0.98':>6s} "
          f"{'2.8d':>6s} {'+2457%':>10s}")


if __name__ == '__main__':
    main()
