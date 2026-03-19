"""
Crash Filter V2: Market-Level Circuit Breakers
===============================================
Round 1 showed that individual stock indicators can't distinguish
crash dips from normal dips. This round tests MARKET-LEVEL filters:

1. Market Breadth: % of stocks above SMA200 (if breadth collapsing, stop)
2. Rolling SL Counter: If N+ stop losses hit in last M calendar days, pause
3. Signal Surge: If signals/day exceeds rolling average by X std devs
4. Combined: Breadth + SL counter as a dual circuit breaker
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import timedelta
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


def compute_market_breadth(all_symbol_data):
    """
    Compute daily market breadth: % of stocks above their SMA200.
    Also compute breadth momentum (is breadth falling fast?).
    """
    # Collect daily above200 for all symbols into a single DataFrame
    daily_above = {}
    for sym, df in all_symbol_data.items():
        if 'above200' in df.columns:
            daily_above[sym] = df['above200'].astype(int)

    breadth_df = pd.DataFrame(daily_above)
    # % of stocks above SMA200 each day
    breadth_pct = breadth_df.mean(axis=1) * 100  # 0-100%
    # 10-day change in breadth (how fast is it dropping?)
    breadth_change_10 = breadth_pct - breadth_pct.shift(10)
    # 5-day change
    breadth_change_5 = breadth_pct - breadth_pct.shift(5)

    return breadth_pct, breadth_change_5, breadth_change_10


def run_backtest_with_market_context(symbols, conn, sl_pct=5, tp_pct=15, max_hold=15):
    """
    Run V2 backtest with market-level context for each trade.
    Process chronologically so we can track rolling SL counts.
    """
    # Phase 1: Load all data and compute indicators
    print("  Phase 1: Loading data and computing indicators...")
    all_symbol_data = {}
    entry_signals = []  # (date, symbol, idx_in_df)

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

        all_symbol_data[sym] = df

        # Collect all entry signal dates
        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['close'] < row['kc6_l'] and row['above200']:
                entry_signals.append((df.index[i], sym, i))

    entry_signals.sort(key=lambda x: x[0])

    # Phase 2: Compute market breadth
    print("  Phase 2: Computing market breadth...")
    breadth_pct, breadth_chg5, breadth_chg10 = compute_market_breadth(all_symbol_data)

    # Count signals per day
    signals_per_day = defaultdict(int)
    for dt, sym, idx in entry_signals:
        signals_per_day[dt] += 1

    # Phase 3: Run chronological backtest
    print("  Phase 3: Running chronological backtest...")

    all_trades = []
    sl_history = []  # list of (date, symbol) for every SL hit
    in_trade = set()  # symbols currently in a trade

    for entry_date, sym, entry_i in entry_signals:
        if sym in in_trade:
            continue

        df = all_symbol_data[sym]
        row = df.iloc[entry_i]
        entry_price = row['close']

        # Market context at entry
        breadth_at_entry = breadth_pct.get(entry_date, np.nan)
        breadth_chg5_at_entry = breadth_chg5.get(entry_date, np.nan)
        breadth_chg10_at_entry = breadth_chg10.get(entry_date, np.nan)
        sigs_today = signals_per_day[entry_date]

        # Rolling SL count: how many SL hits in last N calendar days?
        sl_last_5d = sum(1 for d, s in sl_history
                         if d >= entry_date - timedelta(days=7))
        sl_last_10d = sum(1 for d, s in sl_history
                          if d >= entry_date - timedelta(days=14))
        sl_last_20d = sum(1 for d, s in sl_history
                          if d >= entry_date - timedelta(days=30))

        # Execute trade
        in_trade.add(sym)
        n = len(df)
        j = entry_i + 1
        exit_found = False

        while j < n:
            cur = df.iloc[j]
            hold = j - entry_i
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

                if exit_reason == 'STOP_LOSS':
                    sl_history.append((df.index[j], sym))

                all_trades.append({
                    'symbol': sym,
                    'entry_date': entry_date,
                    'exit_date': df.index[j],
                    'pnl_pct': round(pnl, 2),
                    'hold_days': hold,
                    'exit_reason': exit_reason,
                    'win': pnl > 0,
                    'signals_today': sigs_today,
                    'breadth': round(breadth_at_entry, 1) if pd.notna(breadth_at_entry) else np.nan,
                    'breadth_chg5': round(breadth_chg5_at_entry, 1) if pd.notna(breadth_chg5_at_entry) else np.nan,
                    'breadth_chg10': round(breadth_chg10_at_entry, 1) if pd.notna(breadth_chg10_at_entry) else np.nan,
                    'sl_last_5d': sl_last_5d,
                    'sl_last_10d': sl_last_10d,
                    'sl_last_20d': sl_last_20d,
                })
                in_trade.discard(sym)
                exit_found = True
                break
            j += 1

        if not exit_found:
            in_trade.discard(sym)

    return pd.DataFrame(all_trades)


def filter_stats(df, mask, label, baseline_pnl):
    """Compute stats for filtered subset."""
    allowed = df[~mask]
    blocked = df[mask]
    if len(allowed) == 0:
        return None

    al_wr = allowed['win'].sum() / len(allowed) * 100
    al_avg = allowed['pnl_pct'].mean()
    al_total = allowed['pnl_pct'].sum()
    gp = allowed[allowed['pnl_pct'] > 0]['pnl_pct'].sum()
    gl = abs(allowed[allowed['pnl_pct'] < 0]['pnl_pct'].sum())
    al_pf = gp / gl if gl > 0 else float('inf')
    al_exp = (al_wr/100 * allowed[allowed['win']]['pnl_pct'].mean() / 5) + \
             ((1-al_wr/100) * (allowed[~allowed['win']]['pnl_pct'].mean() if (~allowed['win']).sum() > 0 else 0) / 5)

    bl_sl = len(blocked[blocked['exit_reason'] == 'STOP_LOSS'])
    total_sl = len(df[df['exit_reason'] == 'STOP_LOSS'])
    bl_wins = blocked['win'].sum()
    total_wins = df['win'].sum()

    return {
        'label': label,
        'blocked': len(blocked),
        'blocked_pct': round(len(blocked)/len(df)*100, 1),
        'sl_caught': bl_sl,
        'sl_catch_pct': round(bl_sl/total_sl*100, 1) if total_sl > 0 else 0,
        'wins_lost': bl_wins,
        'wins_lost_pct': round(bl_wins/total_wins*100, 1) if total_wins > 0 else 0,
        'allowed': len(allowed),
        'wr': round(al_wr, 1),
        'avg_pnl': round(al_avg, 2),
        'pf': round(al_pf, 2),
        'exp_r': round(al_exp, 3),
        'total_pnl': round(al_total, 1),
        'pnl_delta': round(al_total - baseline_pnl, 1),
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()

    print("=" * 130)
    print("CRASH FILTER V2: Market-Level Circuit Breakers")
    print("=" * 130)

    df = run_backtest_with_market_context(symbols, conn, sl_pct=5, tp_pct=15, max_hold=15)
    conn.close()

    total = len(df)
    baseline_wr = df['win'].sum() / total * 100
    baseline_pnl = df['pnl_pct'].sum()
    baseline_avg = df['pnl_pct'].mean()
    gp = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
    gl = abs(df[df['pnl_pct'] < 0]['pnl_pct'].sum())
    baseline_pf = gp / gl if gl > 0 else float('inf')

    print(f"\nBaseline: {total} trades | WR: {baseline_wr:.1f}% | "
          f"Avg P/L: {baseline_avg:+.2f}% | PF: {baseline_pf:.2f} | "
          f"Total P/L: {baseline_pnl:+.1f}%")

    # =====================================================================
    # PART 1: Breadth analysis
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 1: TRADE QUALITY BY MARKET BREADTH (% stocks above SMA200)")
    print(f"{'='*130}")

    df_valid = df.dropna(subset=['breadth'])
    print(f"  Trades with breadth data: {len(df_valid)}/{len(df)}")
    print(f"\n  {'Breadth %':>12s} {'Total':>7s} {'Wins':>6s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} {'SL hits':>8s}")
    print(f"  {'-'*60}")

    for lo, hi, label in [(0, 20, '0-20%'), (20, 35, '20-35%'), (35, 50, '35-50%'),
                           (50, 65, '50-65%'), (65, 80, '65-80%'), (80, 100, '80-100%')]:
        mask = (df_valid['breadth'] >= lo) & (df_valid['breadth'] < hi)
        sub = df_valid[mask]
        if len(sub) > 5:
            wr = sub['win'].sum() / len(sub) * 100
            avg = sub['pnl_pct'].mean()
            sl = len(sub[sub['exit_reason'] == 'STOP_LOSS'])
            gpx = sub[sub['pnl_pct'] > 0]['pnl_pct'].sum()
            glx = abs(sub[sub['pnl_pct'] < 0]['pnl_pct'].sum())
            pf = gpx / glx if glx > 0 else float('inf')
            print(f"  {label:>12s} {len(sub):>7d} {sub['win'].sum():>6d} {wr:>5.1f}% {avg:>+7.2f}% {pf:>5.2f} {sl:>7d}")

    # Breadth change (5-day momentum)
    print(f"\n  {'Breadth 5d chg':>15s} {'Total':>7s} {'Wins':>6s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} {'SL hits':>8s}")
    print(f"  {'-'*60}")

    for lo, hi, label in [(-99, -15, '<-15pp'), (-15, -10, '-15 to -10'),
                           (-10, -5, '-10 to -5'), (-5, 0, '-5 to 0'),
                           (0, 5, '0 to +5'), (5, 99, '>+5pp')]:
        mask = (df_valid['breadth_chg5'] >= lo) & (df_valid['breadth_chg5'] < hi)
        sub = df_valid[mask]
        if len(sub) > 5:
            wr = sub['win'].sum() / len(sub) * 100
            avg = sub['pnl_pct'].mean()
            sl = len(sub[sub['exit_reason'] == 'STOP_LOSS'])
            gpx = sub[sub['pnl_pct'] > 0]['pnl_pct'].sum()
            glx = abs(sub[sub['pnl_pct'] < 0]['pnl_pct'].sum())
            pf = gpx / glx if glx > 0 else float('inf')
            print(f"  {label:>15s} {len(sub):>7d} {sub['win'].sum():>6d} {wr:>5.1f}% {avg:>+7.2f}% {pf:>5.2f} {sl:>7d}")

    # =====================================================================
    # PART 2: Rolling SL counter analysis
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 2: TRADE QUALITY BY ROLLING STOP-LOSS COUNT")
    print(f"{'='*130}")

    print(f"\n  {'SL in last ~5d':>15s} {'Total':>7s} {'Wins':>6s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} {'SL hits':>8s}")
    print(f"  {'-'*60}")

    for lo, hi, label in [(0, 0, '0 SLs'), (1, 1, '1 SL'), (2, 2, '2 SLs'),
                           (3, 5, '3-5 SLs'), (6, 10, '6-10 SLs'), (11, 999, '11+ SLs')]:
        mask = (df['sl_last_5d'] >= lo) & (df['sl_last_5d'] <= hi)
        sub = df[mask]
        if len(sub) > 5:
            wr = sub['win'].sum() / len(sub) * 100
            avg = sub['pnl_pct'].mean()
            sl = len(sub[sub['exit_reason'] == 'STOP_LOSS'])
            gpx = sub[sub['pnl_pct'] > 0]['pnl_pct'].sum()
            glx = abs(sub[sub['pnl_pct'] < 0]['pnl_pct'].sum())
            pf = gpx / glx if glx > 0 else float('inf')
            print(f"  {label:>15s} {len(sub):>7d} {sub['win'].sum():>6d} {wr:>5.1f}% {avg:>+7.2f}% {pf:>5.2f} {sl:>7d}")

    print(f"\n  {'SL in last ~10d':>15s} {'Total':>7s} {'Wins':>6s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} {'SL hits':>8s}")
    print(f"  {'-'*60}")

    for lo, hi, label in [(0, 0, '0 SLs'), (1, 2, '1-2 SLs'), (3, 5, '3-5 SLs'),
                           (6, 10, '6-10 SLs'), (11, 20, '11-20 SLs'), (21, 999, '21+ SLs')]:
        mask = (df['sl_last_10d'] >= lo) & (df['sl_last_10d'] <= hi)
        sub = df[mask]
        if len(sub) > 5:
            wr = sub['win'].sum() / len(sub) * 100
            avg = sub['pnl_pct'].mean()
            sl = len(sub[sub['exit_reason'] == 'STOP_LOSS'])
            gpx = sub[sub['pnl_pct'] > 0]['pnl_pct'].sum()
            glx = abs(sub[sub['pnl_pct'] < 0]['pnl_pct'].sum())
            pf = gpx / glx if glx > 0 else float('inf')
            print(f"  {label:>15s} {len(sub):>7d} {sub['win'].sum():>6d} {wr:>5.1f}% {avg:>+7.2f}% {pf:>5.2f} {sl:>7d}")

    # =====================================================================
    # PART 3: Test all market-level filters
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 3: MARKET-LEVEL FILTER COMPARISON")
    print(f"{'='*130}")
    print(f"  Baseline: {total} trades | WR: {baseline_wr:.1f}% | PF: {baseline_pf:.2f} | Total P/L: {baseline_pnl:+.1f}%\n")

    filters = []

    # Breadth filters
    for thresh in [30, 35, 40, 45]:
        mask = df['breadth'] < thresh
        mask = mask.fillna(False)
        f = filter_stats(df, mask, f"Breadth < {thresh}%", baseline_pnl)
        if f: filters.append(f)

    # Breadth change filters
    for thresh in [-5, -8, -10, -15]:
        mask = df['breadth_chg5'] < thresh
        mask = mask.fillna(False)
        f = filter_stats(df, mask, f"Breadth 5d chg < {thresh}pp", baseline_pnl)
        if f: filters.append(f)

    for thresh in [-10, -15, -20]:
        mask = df['breadth_chg10'] < thresh
        mask = mask.fillna(False)
        f = filter_stats(df, mask, f"Breadth 10d chg < {thresh}pp", baseline_pnl)
        if f: filters.append(f)

    # Rolling SL counter filters
    for thresh in [2, 3, 5, 8]:
        mask = df['sl_last_5d'] >= thresh
        f = filter_stats(df, mask, f"SL(5d) >= {thresh}", baseline_pnl)
        if f: filters.append(f)

    for thresh in [3, 5, 8, 10]:
        mask = df['sl_last_10d'] >= thresh
        f = filter_stats(df, mask, f"SL(10d) >= {thresh}", baseline_pnl)
        if f: filters.append(f)

    # Combined: breadth + SL counter
    for b_thresh, sl_thresh in [(40, 3), (45, 3), (40, 5), (45, 5)]:
        mask = (df['breadth'].fillna(100) < b_thresh) | (df['sl_last_5d'] >= sl_thresh)
        f = filter_stats(df, mask, f"Breadth<{b_thresh} OR SL5d>={sl_thresh}", baseline_pnl)
        if f: filters.append(f)

    # Combined: breadth change + SL counter
    for bc_thresh, sl_thresh in [(-8, 3), (-10, 3), (-8, 5), (-10, 5)]:
        mask = (df['breadth_chg5'].fillna(0) < bc_thresh) | (df['sl_last_5d'] >= sl_thresh)
        f = filter_stats(df, mask, f"BChg5<{bc_thresh} OR SL5d>={sl_thresh}", baseline_pnl)
        if f: filters.append(f)

    # Combined: breadth + breadth change (double confirmation)
    for b_thresh, bc_thresh in [(50, -8), (50, -10), (45, -5), (45, -8)]:
        mask = (df['breadth'].fillna(100) < b_thresh) & (df['breadth_chg5'].fillna(0) < bc_thresh)
        f = filter_stats(df, mask, f"Breadth<{b_thresh} AND BChg5<{bc_thresh}", baseline_pnl)
        if f: filters.append(f)

    # Print
    print(f"  {'Filter':<35s} {'Blk':>5s} {'Blk%':>5s} {'SLcatch':>8s} {'WinsLost':>9s} "
          f"{'Alwd':>5s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} {'Exp(R)':>7s} {'TotalP/L':>10s} {'Delta':>8s}")
    print(f"  {'-'*115}")

    # Sort by best total P/L
    filters.sort(key=lambda x: x['total_pnl'], reverse=True)

    for f in filters:
        print(f"  {f['label']:<35s} {f['blocked']:>5d} {f['blocked_pct']:>4.1f}% "
              f"{f['sl_caught']:>3d}({f['sl_catch_pct']:>4.1f}%) "
              f"{f['wins_lost']:>3d}({f['wins_lost_pct']:>4.1f}%) "
              f"{f['allowed']:>5d} {f['wr']:>5.1f}% {f['avg_pnl']:>+7.2f}% "
              f"{f['pf']:>5.2f} {f['exp_r']:>+6.3f} {f['total_pnl']:>+9.1f}% {f['pnl_delta']:>+7.1f}%")

    # =====================================================================
    # PART 4: Best filters applied to crash periods
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 4: CRASH PERIOD IMPACT OF TOP FILTERS")
    print(f"{'='*130}")

    crash_periods = [
        ('2008 GFC', '2008-01-01', '2008-12-31'),
        ('2011 Euro', '2011-07-01', '2012-01-31'),
        ('2015-16', '2015-08-01', '2016-03-31'),
        ('2018 NBFC', '2018-08-01', '2018-12-31'),
        ('2020 COVID', '2020-02-01', '2020-04-30'),
        ('2022 War', '2022-01-01', '2022-07-31'),
    ]

    # Top filters to test
    top_filters = [
        ("No Filter", pd.Series(False, index=df.index)),
        ("SL(5d) >= 3", df['sl_last_5d'] >= 3),
        ("SL(10d) >= 5", df['sl_last_10d'] >= 5),
        ("Breadth < 40%", df['breadth'].fillna(100) < 40),
        ("BChg5 < -10pp", df['breadth_chg5'].fillna(0) < -10),
        ("Breadth<50 AND BChg5<-8", (df['breadth'].fillna(100) < 50) & (df['breadth_chg5'].fillna(0) < -8)),
        ("Breadth<40 OR SL5d>=3", (df['breadth'].fillna(100) < 40) | (df['sl_last_5d'] >= 3)),
        ("BChg5<-10 OR SL5d>=3", (df['breadth_chg5'].fillna(0) < -10) | (df['sl_last_5d'] >= 3)),
    ]

    for period_name, start, end in crash_periods:
        crash_mask = (df['entry_date'] >= start) & (df['entry_date'] <= end)
        crash_trades = df[crash_mask]
        if len(crash_trades) == 0:
            continue

        ct_pnl = crash_trades['pnl_pct'].sum()
        ct_wr = crash_trades['win'].sum() / len(crash_trades) * 100
        ct_sl = len(crash_trades[crash_trades['exit_reason'] == 'STOP_LOSS'])

        print(f"\n  {period_name}: {len(crash_trades)} trades | WR: {ct_wr:.0f}% | "
              f"P/L: {ct_pnl:+.1f}% | SL hits: {ct_sl}")
        print(f"  {'Filter':<35s} {'Kept':>5s} {'Blocked':>8s} {'WR%':>6s} {'TotalP/L':>10s} {'SL blocked':>11s}")
        print(f"  {'-'*80}")

        for fname, fmask in top_filters:
            fmask_crash = fmask[crash_mask]
            allowed_c = crash_trades[~fmask_crash]
            blocked_c = crash_trades[fmask_crash]

            al_count = len(allowed_c)
            bl_count = len(blocked_c)
            al_wr = allowed_c['win'].sum() / al_count * 100 if al_count > 0 else 0
            al_pnl = allowed_c['pnl_pct'].sum() if al_count > 0 else 0
            bl_sl = len(blocked_c[blocked_c['exit_reason'] == 'STOP_LOSS']) if bl_count > 0 else 0

            marker = ' <<<' if al_pnl > ct_pnl + 20 else ''
            print(f"  {fname:<35s} {al_count:>5d} {bl_count:>8d} "
                  f"{al_wr:>5.1f}% {al_pnl:>+9.1f}% {bl_sl:>10d}{marker}")

    # =====================================================================
    # PART 5: Full year-by-year with top 3 filters
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 5: YEAR-BY-YEAR -- TOP FILTERS")
    print(f"{'='*130}")

    df['year'] = df['entry_date'].dt.year

    best_3 = [
        ("SL(5d)>=3", df['sl_last_5d'] >= 3),
        ("Breadth<40 OR SL5d>=3", (df['breadth'].fillna(100) < 40) | (df['sl_last_5d'] >= 3)),
        ("BChg5<-10 OR SL5d>=3", (df['breadth_chg5'].fillna(0) < -10) | (df['sl_last_5d'] >= 3)),
    ]

    print(f"  {'Year':<6s} | {'BASELINE':^20s} | ", end='')
    for fname, _ in best_3:
        print(f"{fname:^20s} | ", end='')
    print()

    print(f"  {'':6s} | {'Trades':>6s} {'WR':>5s} {'P/L':>8s} | ", end='')
    for _ in best_3:
        print(f"{'Trades':>6s} {'WR':>5s} {'P/L':>8s} | ", end='')
    print()
    print(f"  {'-'*100}")

    for yr in sorted(df['year'].unique()):
        base_yr = df[df['year'] == yr]
        if len(base_yr) < 3:
            continue

        b_wr = base_yr['win'].sum() / len(base_yr) * 100
        b_tot = base_yr['pnl_pct'].sum()

        print(f"  {yr:<6d} | {len(base_yr):>6d} {b_wr:>4.0f}% {b_tot:>+7.1f}% | ", end='')

        for fname, fmask in best_3:
            filt_yr = base_yr[~fmask[base_yr.index]]
            if len(filt_yr) > 0:
                f_wr = filt_yr['win'].sum() / len(filt_yr) * 100
                f_tot = filt_yr['pnl_pct'].sum()
                print(f"{len(filt_yr):>6d} {f_wr:>4.0f}% {f_tot:>+7.1f}% | ", end='')
            else:
                print(f"{'--':>6s} {'--':>5s} {'--':>8s} | ", end='')
        print()

    # Grand totals
    print(f"  {'-'*100}")
    print(f"  {'TOTAL':<6s} | {len(df):>6d} {baseline_wr:>4.0f}% {baseline_pnl:>+7.1f}% | ", end='')
    for fname, fmask in best_3:
        filt = df[~fmask]
        f_wr = filt['win'].sum() / len(filt) * 100
        f_tot = filt['pnl_pct'].sum()
        print(f"{len(filt):>6d} {f_wr:>4.0f}% {f_tot:>+7.1f}% | ", end='')
    print()


if __name__ == '__main__':
    main()
