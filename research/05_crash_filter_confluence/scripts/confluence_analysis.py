"""
Confluence Analysis: KC6 Mean Reversion + TTM Squeeze
=====================================================
Find trades where BOTH strategies trigger within a short window.
If both a mean reversion dip AND a squeeze breakout align,
does the combined signal produce better results?
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


def compute_indicators(df):
    """Compute all indicators needed for both strategies."""
    c = df['close']

    # KC6 indicators (our strategy)
    kc6_mid = ema(c, 6)
    kc6_atr = atr(df, 6)
    df['kc6_m'] = kc6_mid
    df['kc6_u'] = kc6_mid + 1.3 * kc6_atr
    df['kc6_l'] = kc6_mid - 1.3 * kc6_atr
    df['sma_200'] = sma(c, 200)
    df['above200'] = c > df['sma_200']

    # TTM Squeeze indicators
    bb_mid = sma(c, 20)
    bb_std = c.rolling(window=20).std()
    df['bb_upper'] = bb_mid + 2.0 * bb_std
    df['bb_lower'] = bb_mid - 2.0 * bb_std

    kc20_mid = ema(c, 20)
    kc20_atr = atr(df, 20)
    df['kc20_upper'] = kc20_mid + 1.5 * kc20_atr
    df['kc20_lower'] = kc20_mid - 1.5 * kc20_atr

    # Squeeze ON: BB inside KC
    df['squeeze_on'] = (df['bb_lower'] > df['kc20_lower']) & (df['bb_upper'] < df['kc20_upper'])

    # Momentum
    highest = df['high'].rolling(window=12).max()
    lowest = df['low'].rolling(window=12).min()
    midpoint = (highest + lowest) / 2
    sma_mid = sma(c, 12)
    df['momentum'] = c - (midpoint + sma_mid) / 2
    df['mom_prev'] = df['momentum'].shift(1)
    df['mom_rising'] = df['momentum'] > df['mom_prev']

    # Squeeze count
    squeeze_count = []
    count = 0
    for sq in df['squeeze_on']:
        if sq:
            count += 1
        else:
            count = 0
        squeeze_count.append(count)
    df['squeeze_count'] = squeeze_count

    # Squeeze fires
    df['squeeze_fires'] = (~df['squeeze_on']) & (df['squeeze_on'].shift(1) == True)
    df['fire_squeeze_len'] = df['squeeze_count'].shift(1)

    return df


def run_combined_backtest(symbols, conn, sl_pct=5, tp_pct=15, max_hold=15,
                          window_days=3):
    """
    Run both strategies and tag each KC6 trade with whether TTM Squeeze
    also fired within +/- window_days of the KC6 entry.
    """
    all_kc6_trades = []

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

        df = compute_indicators(df)

        # Collect TTM squeeze fire dates for this symbol
        squeeze_fire_indices = set()
        for i in range(200, len(df)):
            row = df.iloc[i]
            if (row['squeeze_fires'] and
                row['fire_squeeze_len'] >= 5 and
                row['momentum'] > 0 and
                row['mom_rising'] and
                row['above200']):
                squeeze_fire_indices.add(i)

        # Run KC6 V2 backtest (our strategy)
        n = len(df)
        i = 200

        while i < n:
            row = df.iloc[i]
            if row['close'] < row['kc6_l'] and row['above200']:
                entry_price = row['close']
                entry_i = i
                entry_date = df.index[i]

                # Check if TTM squeeze fired within window
                ttm_match = False
                for offset in range(-window_days, window_days + 1):
                    if (entry_i + offset) in squeeze_fire_indices:
                        ttm_match = True
                        break

                # Also check squeeze state at entry
                squeeze_on_at_entry = row['squeeze_on']
                mom_positive = row['momentum'] > 0

                i += 1
                while i < n:
                    cur = df.iloc[i]
                    hold = i - entry_i

                    low_pnl = (cur['low'] / entry_price - 1) * 100
                    high_pnl = (cur['high'] / entry_price - 1) * 100

                    exit_reason = None
                    exit_price = None

                    # SL
                    if low_pnl <= -sl_pct:
                        exit_reason = 'STOP_LOSS'
                        exit_price = round(entry_price * (1 - sl_pct / 100), 2)
                    # TP
                    elif high_pnl >= tp_pct:
                        exit_reason = 'TAKE_PROFIT'
                        exit_price = round(entry_price * (1 + tp_pct / 100), 2)
                    # Max hold
                    elif hold >= max_hold:
                        exit_reason = 'MAX_HOLD'
                        exit_price = round(cur['close'], 2)
                    # V2 limit exit
                    elif cur['high'] > cur['kc6_m']:
                        exit_reason = 'SIGNAL_KC6_MID'
                        exit_price = round(cur['kc6_m'], 2)

                    if exit_reason:
                        pnl = (exit_price / entry_price - 1) * 100
                        all_kc6_trades.append({
                            'symbol': sym,
                            'entry_date': entry_date.strftime('%Y-%m-%d'),
                            'entry_price': round(entry_price, 2),
                            'exit_price': exit_price,
                            'pnl_pct': round(pnl, 2),
                            'hold_days': hold,
                            'exit_reason': exit_reason,
                            'win': pnl > 0,
                            'ttm_confluence': ttm_match,
                            'squeeze_on_at_entry': squeeze_on_at_entry,
                            'mom_positive_at_entry': mom_positive,
                        })
                        i += 1
                        break
                    i += 1
            else:
                i += 1

    return pd.DataFrame(all_kc6_trades)


def print_group_stats(df_t, label):
    """Print stats for a group of trades."""
    total = len(df_t)
    if total == 0:
        print(f"  {label}: NO TRADES")
        return None
    wins = df_t['win'].sum()
    wr = wins / total * 100
    avg_win = df_t[df_t['win']]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = df_t[~df_t['win']]['pnl_pct'].mean() if (total - wins) > 0 else 0
    avg_pnl = df_t['pnl_pct'].mean()
    avg_hold = df_t['hold_days'].mean()
    total_pnl = df_t['pnl_pct'].sum()

    gross_profit = df_t[df_t['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(df_t[df_t['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    exp_r = (wr / 100 * avg_win / sl_pct) + ((1 - wr / 100) * avg_loss / sl_pct) if sl_pct > 0 else 0

    return {
        'label': label,
        'total': total, 'wins': wins, 'wr': round(wr, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'avg_pnl': round(avg_pnl, 2), 'avg_hold': round(avg_hold, 1),
        'pf': round(pf, 2), 'rr': round(rr, 2),
        'exp_r': round(exp_r, 3),
        'total_pnl': round(total_pnl, 1),
    }


sl_pct = 5  # global for exp_r calc


def main():
    global sl_pct
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()

    print("=" * 120)
    print("CONFLUENCE ANALYSIS: KC6 Mean Reversion + TTM Squeeze")
    print("=" * 120)
    print(f"Symbols: {len(symbols)} | Data: 2000-2025")
    print(f"KC6 V2: SL={sl_pct}%, TP=15%, Max Hold=15d, Limit exit @ KC6 mid")
    print(f"TTM Squeeze: BB(20,2) vs KC(20,1.5), 5+ bars, SMA200 filter")
    print(f"Confluence window: +/- 3 days from KC6 entry")
    print()

    print("Running combined backtest...")
    df = run_combined_backtest(symbols, conn, sl_pct=sl_pct, window_days=3)
    conn.close()

    print(f"Total KC6 trades: {len(df)}")
    print(f"  With TTM confluence: {df['ttm_confluence'].sum()}")
    print(f"  Without TTM confluence: {(~df['ttm_confluence']).sum()}")
    print(f"  Squeeze ON at entry: {df['squeeze_on_at_entry'].sum()}")
    print(f"  Momentum positive at entry: {df['mom_positive_at_entry'].sum()}")

    # Split into groups
    df_all = df
    df_conf = df[df['ttm_confluence'] == True]
    df_no_conf = df[df['ttm_confluence'] == False]
    df_squeeze_on = df[df['squeeze_on_at_entry'] == True]
    df_squeeze_off = df[df['squeeze_on_at_entry'] == False]
    df_mom_pos = df[df['mom_positive_at_entry'] == True]
    df_mom_neg = df[df['mom_positive_at_entry'] == False]

    # Combo: squeeze ON + momentum positive
    df_squeeze_mom = df[(df['squeeze_on_at_entry'] == True) & (df['mom_positive_at_entry'] == True)]
    # Combo: squeeze ON + momentum negative (divergence)
    df_squeeze_mom_neg = df[(df['squeeze_on_at_entry'] == True) & (df['mom_positive_at_entry'] == False)]

    results = []
    for label, subset in [
        ("ALL KC6 trades", df_all),
        ("WITH TTM confluence (+/-3d)", df_conf),
        ("WITHOUT TTM confluence", df_no_conf),
        ("Squeeze ON at KC6 entry", df_squeeze_on),
        ("Squeeze OFF at KC6 entry", df_squeeze_off),
        ("Momentum > 0 at KC6 entry", df_mom_pos),
        ("Momentum < 0 at KC6 entry", df_mom_neg),
        ("Squeeze ON + Mom > 0", df_squeeze_mom),
        ("Squeeze ON + Mom < 0", df_squeeze_mom_neg),
    ]:
        r = print_group_stats(subset, label)
        if r:
            results.append(r)

    # Print comparison table
    print(f"\n\n{'=' * 120}")
    print("CONFLUENCE COMPARISON TABLE")
    print(f"{'=' * 120}")
    header = (f"  {'Group':<35s} {'Trades':>7s} {'WR%':>6s} {'AvgWin':>8s} {'AvgLoss':>9s} "
              f"{'AvgP/L':>8s} {'PF':>6s} {'R:R':>6s} {'Exp(R)':>8s} {'Hold':>6s} {'TotalP/L':>10s}")
    print(header)
    print(f"  {'-' * 115}")

    for r in results:
        print(f"  {r['label']:<35s} {r['total']:>7d} {r['wr']:>5.1f}% "
              f"{r['avg_win']:>+7.2f}% {r['avg_loss']:>+8.2f}% "
              f"{r['avg_pnl']:>+7.2f}% {r['pf']:>5.2f} {r['rr']:>5.2f} "
              f"{r['exp_r']:>+7.3f}R {r['avg_hold']:>5.1f}d {r['total_pnl']:>+9.1f}%")

    # Year-by-year for confluence vs non-confluence
    print(f"\n\n{'=' * 120}")
    print("YEAR-BY-YEAR: CONFLUENCE vs NON-CONFLUENCE")
    print(f"{'=' * 120}")
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['year'] = df['entry_date'].dt.year

    print(f"  {'Year':<6s} | {'--- WITH Confluence ---':^40s} | {'--- WITHOUT Confluence ---':^40s}")
    print(f"  {'':6s} | {'Trades':>7s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} | "
          f"{'Trades':>7s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s}")
    print(f"  {'-' * 90}")

    for yr in sorted(df['year'].unique()):
        if yr >= 2008:
            yr_conf = df[(df['year'] == yr) & (df['ttm_confluence'] == True)]
            yr_no = df[(df['year'] == yr) & (df['ttm_confluence'] == False)]

            if len(yr_conf) >= 3 and len(yr_no) >= 3:
                wr_c = yr_conf['win'].sum() / len(yr_conf) * 100
                pnl_c = yr_conf['pnl_pct'].mean()
                gp_c = yr_conf[yr_conf['pnl_pct'] > 0]['pnl_pct'].sum()
                gl_c = abs(yr_conf[yr_conf['pnl_pct'] < 0]['pnl_pct'].sum())
                pf_c = gp_c / gl_c if gl_c > 0 else 999

                wr_n = yr_no['win'].sum() / len(yr_no) * 100
                pnl_n = yr_no['pnl_pct'].mean()
                gp_n = yr_no[yr_no['pnl_pct'] > 0]['pnl_pct'].sum()
                gl_n = abs(yr_no[yr_no['pnl_pct'] < 0]['pnl_pct'].sum())
                pf_n = gp_n / gl_n if gl_n > 0 else 999

                print(f"  {yr:<6d} | {len(yr_conf):>7d} {wr_c:>5.1f}% {pnl_c:>+7.2f}% {pf_c:>5.2f} | "
                      f"{len(yr_no):>7d} {wr_n:>5.1f}% {pnl_n:>+7.2f}% {pf_n:>5.2f}")
            elif len(yr_no) >= 3:
                wr_n = yr_no['win'].sum() / len(yr_no) * 100
                pnl_n = yr_no['pnl_pct'].mean()
                print(f"  {yr:<6d} | {'<3 trades':>30s}   | "
                      f"{len(yr_no):>7d} {wr_n:>5.1f}% {pnl_n:>+7.2f}%     -")


if __name__ == '__main__':
    main()
