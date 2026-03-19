"""
Confluence V2: KC6 Entry Quality Filters Using TTM Squeeze Context
===================================================================
Instead of requiring both to fire at the same time (impossible),
check if TTM Squeeze context BEFORE the KC6 entry improves results.

Hypothesis: A KC6 dip that happens AFTER a recent squeeze release
(stock was coiling, then dipped) may bounce harder than a random dip.

Also test: Does the squeeze state (ON/OFF) at KC6 entry matter?
Does momentum direction at KC6 entry predict outcomes?
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
    """Compute all indicators for both strategies."""
    c = df['close']

    # KC6 (our strategy)
    kc6_mid = ema(c, 6)
    kc6_atr = atr(df, 6)
    df['kc6_m'] = kc6_mid
    df['kc6_l'] = kc6_mid - 1.3 * kc6_atr
    df['sma_200'] = sma(c, 200)
    df['above200'] = c > df['sma_200']

    # TTM Squeeze components
    bb_mid = sma(c, 20)
    bb_std = c.rolling(window=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    kc20_mid = ema(c, 20)
    kc20_atr = atr(df, 20)
    kc20_upper = kc20_mid + 1.5 * kc20_atr
    kc20_lower = kc20_mid - 1.5 * kc20_atr

    # Squeeze ON/OFF
    df['squeeze_on'] = (bb_lower > kc20_lower) & (bb_upper < kc20_upper)

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

    # Days since last squeeze ended
    # Track when squeeze was last ON
    last_squeeze_bar = []
    last_on = -999
    for idx, sq in enumerate(df['squeeze_on']):
        if sq:
            last_on = idx
        last_squeeze_bar.append(last_on)
    df['last_squeeze_bar'] = last_squeeze_bar
    df['bars_since_squeeze'] = range(len(df))
    df['bars_since_squeeze'] = df.index.map(
        lambda x: 0  # placeholder, compute below
    )

    # Momentum
    highest = df['high'].rolling(window=12).max()
    lowest = df['low'].rolling(window=12).min()
    midpoint = (highest + lowest) / 2
    sma_mid = sma(c, 12)
    df['momentum'] = c - (midpoint + sma_mid) / 2

    # BB width (volatility measure)
    df['bb_width'] = (bb_upper - bb_lower) / bb_mid * 100

    # BB %B (position within bands)
    df['bb_pctb'] = (c - bb_lower) / (bb_upper - bb_lower)

    return df


def run_analysis(symbols, conn, sl_pct=5, tp_pct=15, max_hold=15):
    """Run KC6 V2 backtest with squeeze context tags."""
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

        df = compute_indicators(df)

        # Precompute: for each bar, how many bars ago was the last squeeze ON?
        last_on_idx = -999
        bars_since = []
        for idx in range(len(df)):
            if df.iloc[idx]['squeeze_on']:
                last_on_idx = idx
            bars_since.append(idx - last_on_idx if last_on_idx >= 0 else 999)

        # Run KC6 V2 backtest
        n = len(df)
        i = 200

        while i < n:
            row = df.iloc[i]
            if row['close'] < row['kc6_l'] and row['above200']:
                entry_price = row['close']
                entry_i = i
                entry_date = df.index[i]

                # Context at entry
                squeeze_on = row['squeeze_on']
                days_since_squeeze = bars_since[i]
                bb_width = row['bb_width']
                bb_pctb = row['bb_pctb']
                momentum = row['momentum']

                # Categorize squeeze recency
                if squeeze_on:
                    squeeze_recency = 'DURING_SQUEEZE'
                elif days_since_squeeze <= 5:
                    squeeze_recency = 'JUST_FIRED_0-5d'
                elif days_since_squeeze <= 15:
                    squeeze_recency = 'RECENT_6-15d'
                elif days_since_squeeze <= 30:
                    squeeze_recency = 'MODERATE_16-30d'
                else:
                    squeeze_recency = 'STALE_30d+'

                # BB width category
                if bb_width < 5:
                    vol_regime = 'LOW_VOL'
                elif bb_width < 10:
                    vol_regime = 'NORMAL_VOL'
                else:
                    vol_regime = 'HIGH_VOL'

                i += 1
                while i < n:
                    cur = df.iloc[i]
                    hold = i - entry_i
                    low_pnl = (cur['low'] / entry_price - 1) * 100
                    high_pnl = (cur['high'] / entry_price - 1) * 100

                    exit_reason = None
                    exit_price = None

                    if low_pnl <= -sl_pct:
                        exit_reason = 'STOP_LOSS'
                        exit_price = round(entry_price * (1 - sl_pct / 100), 2)
                    elif high_pnl >= tp_pct:
                        exit_reason = 'TAKE_PROFIT'
                        exit_price = round(entry_price * (1 + tp_pct / 100), 2)
                    elif hold >= max_hold:
                        exit_reason = 'MAX_HOLD'
                        exit_price = round(cur['close'], 2)
                    elif cur['high'] > cur['kc6_m']:
                        exit_reason = 'SIGNAL_KC6_MID'
                        exit_price = round(cur['kc6_m'], 2)

                    if exit_reason:
                        pnl = (exit_price / entry_price - 1) * 100
                        all_trades.append({
                            'symbol': sym,
                            'entry_date': entry_date.strftime('%Y-%m-%d'),
                            'pnl_pct': round(pnl, 2),
                            'hold_days': hold,
                            'exit_reason': exit_reason,
                            'win': pnl > 0,
                            'squeeze_recency': squeeze_recency,
                            'vol_regime': vol_regime,
                            'bb_width': round(bb_width, 2),
                            'bb_pctb': round(bb_pctb, 3),
                            'days_since_squeeze': days_since_squeeze,
                        })
                        i += 1
                        break
                    i += 1
            else:
                i += 1

    return pd.DataFrame(all_trades)


def stats_row(df_t, label, sl=5):
    """Compute stats for a group."""
    total = len(df_t)
    if total < 5:
        return None
    wins = df_t['win'].sum()
    wr = wins / total * 100
    avg_win = df_t[df_t['win']]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = df_t[~df_t['win']]['pnl_pct'].mean() if (total - wins) > 0 else 0
    avg_pnl = df_t['pnl_pct'].mean()
    avg_hold = df_t['hold_days'].mean()
    total_pnl = df_t['pnl_pct'].sum()
    gp = df_t[df_t['pnl_pct'] > 0]['pnl_pct'].sum()
    gl = abs(df_t[df_t['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    exp_r = (wr / 100 * avg_win / sl) + ((1 - wr / 100) * avg_loss / sl)

    return {
        'label': label, 'total': total, 'wr': round(wr, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'avg_pnl': round(avg_pnl, 2), 'pf': round(pf, 2),
        'rr': round(rr, 2), 'exp_r': round(exp_r, 3),
        'avg_hold': round(avg_hold, 1), 'total_pnl': round(total_pnl, 1),
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()

    print("=" * 130)
    print("CONFLUENCE V2: KC6 Entry Quality by TTM Squeeze Context")
    print("=" * 130)
    print(f"Base: KC6 V2 (SL=5%, TP=15%, max hold=15d, limit exit @ KC6 mid)")
    print(f"Question: Does TTM Squeeze context at the time of KC6 entry predict trade quality?")
    print()

    df = run_analysis(symbols, conn)
    conn.close()

    print(f"Total trades: {len(df)}")
    print()

    # 1. Squeeze recency analysis
    print("=" * 130)
    print("1. KC6 TRADE QUALITY BY SQUEEZE RECENCY")
    print("   (How recently did the stock complete a TTM Squeeze before the KC6 dip?)")
    print("=" * 130)

    header = (f"  {'Squeeze Context':<25s} {'Trades':>7s} {'WR%':>6s} {'AvgWin':>8s} {'AvgLoss':>9s} "
              f"{'AvgP/L':>8s} {'PF':>6s} {'R:R':>6s} {'Exp(R)':>8s} {'Hold':>6s} {'TotalP/L':>10s}")
    print(header)
    print(f"  {'-' * 125}")

    for recency in ['DURING_SQUEEZE', 'JUST_FIRED_0-5d', 'RECENT_6-15d',
                     'MODERATE_16-30d', 'STALE_30d+']:
        subset = df[df['squeeze_recency'] == recency]
        r = stats_row(subset, recency)
        if r:
            print(f"  {r['label']:<25s} {r['total']:>7d} {r['wr']:>5.1f}% "
                  f"{r['avg_win']:>+7.2f}% {r['avg_loss']:>+8.2f}% "
                  f"{r['avg_pnl']:>+7.2f}% {r['pf']:>5.2f} {r['rr']:>5.2f} "
                  f"{r['exp_r']:>+7.3f}R {r['avg_hold']:>5.1f}d {r['total_pnl']:>+9.1f}%")
        else:
            cnt = len(subset)
            print(f"  {recency:<25s} {cnt:>7d} (too few trades)")

    # All trades baseline
    r_all = stats_row(df, 'ALL TRADES')
    print(f"  {'-' * 125}")
    print(f"  {r_all['label']:<25s} {r_all['total']:>7d} {r_all['wr']:>5.1f}% "
          f"{r_all['avg_win']:>+7.2f}% {r_all['avg_loss']:>+8.2f}% "
          f"{r_all['avg_pnl']:>+7.2f}% {r_all['pf']:>5.2f} {r_all['rr']:>5.2f} "
          f"{r_all['exp_r']:>+7.3f}R {r_all['avg_hold']:>5.1f}d {r_all['total_pnl']:>+9.1f}%")

    # 2. Volatility regime analysis
    print(f"\n\n{'=' * 130}")
    print("2. KC6 TRADE QUALITY BY VOLATILITY REGIME (BB Width at entry)")
    print("   LOW = BB width < 5% | NORMAL = 5-10% | HIGH = > 10%")
    print("=" * 130)
    print(header)
    print(f"  {'-' * 125}")

    for vol in ['LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL']:
        subset = df[df['vol_regime'] == vol]
        r = stats_row(subset, vol)
        if r:
            print(f"  {r['label']:<25s} {r['total']:>7d} {r['wr']:>5.1f}% "
                  f"{r['avg_win']:>+7.2f}% {r['avg_loss']:>+8.2f}% "
                  f"{r['avg_pnl']:>+7.2f}% {r['pf']:>5.2f} {r['rr']:>5.2f} "
                  f"{r['exp_r']:>+7.3f}R {r['avg_hold']:>5.1f}d {r['total_pnl']:>+9.1f}%")

    # 3. BB %B at entry (where within bands is the price?)
    print(f"\n\n{'=' * 130}")
    print("3. KC6 TRADE QUALITY BY BB %B AT ENTRY")
    print("   %B < 0 = below lower band | 0-0.2 = near lower | 0.2-0.5 = lower half | > 0.5 = upper half")
    print("=" * 130)
    print(header)
    print(f"  {'-' * 125}")

    for label, cond in [
        ('Below BB lower (<0)', df['bb_pctb'] < 0),
        ('Near BB lower (0-0.2)', (df['bb_pctb'] >= 0) & (df['bb_pctb'] < 0.2)),
        ('Lower half (0.2-0.5)', (df['bb_pctb'] >= 0.2) & (df['bb_pctb'] < 0.5)),
        ('Upper half (>0.5)', df['bb_pctb'] >= 0.5),
    ]:
        subset = df[cond]
        r = stats_row(subset, label)
        if r:
            print(f"  {r['label']:<25s} {r['total']:>7d} {r['wr']:>5.1f}% "
                  f"{r['avg_win']:>+7.2f}% {r['avg_loss']:>+8.2f}% "
                  f"{r['avg_pnl']:>+7.2f}% {r['pf']:>5.2f} {r['rr']:>5.2f} "
                  f"{r['exp_r']:>+7.3f}R {r['avg_hold']:>5.1f}d {r['total_pnl']:>+9.1f}%")
        else:
            print(f"  {label:<25s} {len(subset):>7d} (too few trades)")


if __name__ == '__main__':
    main()
