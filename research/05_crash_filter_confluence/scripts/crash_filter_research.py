"""
Crash Filter Research: Protecting KC6 V2 From Market Crashes
=============================================================
Investigate patterns before/during crash periods and test filters
that could block entries during market-wide selloffs.

Key crash periods in Indian markets:
  - 2008 GFC (Jan-Nov 2008)
  - 2011 Euro crisis (Aug-Dec 2011)
  - 2015-16 China/commodity crash (Aug 2015 - Feb 2016)
  - 2018 IL&FS/NBFC crisis (Sep-Oct 2018)
  - 2020 COVID (Feb-Apr 2020)
  - 2022 Russia/Ukraine + rate hikes (Jan-Jun 2022)

Filters to test:
  1. Signal clustering: If N+ stocks trigger KC6 entry on same day, skip all
  2. ATR expansion: If ATR(14) is X% above its 50-day average, skip
  3. Consecutive loss circuit breaker: After N consecutive SL hits, pause trading
  4. Market regime: Track a broad index (or % of stocks above SMA200)
  5. KC6 band width: When bands widen beyond threshold, skip
  6. Drawdown from recent high: If stock dropped >X% in Y days, skip
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
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


def run_full_backtest_with_context(symbols, conn, sl_pct=5, tp_pct=15, max_hold=15):
    """
    Run V2 backtest but capture rich context for each trade:
    - How many other stocks also triggered that day
    - ATR ratio (current vs 50-day avg)
    - Distance from recent high
    - KC band width
    - Consecutive losses before this trade
    """
    # First pass: collect all entry dates across all symbols
    entry_dates_by_day = defaultdict(list)  # date -> list of symbols

    all_symbol_data = {}

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
        df['atr_14'] = atr(df, 14)
        df['atr_50avg'] = df['atr_14'].rolling(50).mean()
        df['atr_ratio'] = df['atr_14'] / df['atr_50avg']
        df['high_20'] = df['high'].rolling(20).max()
        df['drawdown_20'] = (c / df['high_20'] - 1) * 100
        df['kc_width'] = (df['kc6_u'] - df['kc6_l']) / df['kc6_m'] * 100

        all_symbol_data[sym] = df

        # Scan for entry signals
        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['close'] < row['kc6_l'] and row['above200']:
                entry_dates_by_day[df.index[i].strftime('%Y-%m-%d')].append(sym)

    # Count signals per day
    signals_per_day = {d: len(syms) for d, syms in entry_dates_by_day.items()}

    # Second pass: run actual backtest with context
    all_trades = []
    consecutive_losses = 0  # global tracker across all trades chronologically

    # We need to process trades in chronological order for consecutive loss tracking
    # First collect all potential entries
    potential_entries = []
    for sym, df in all_symbol_data.items():
        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['close'] < row['kc6_l'] and row['above200']:
                potential_entries.append((df.index[i], sym, i))

    potential_entries.sort(key=lambda x: x[0])

    # Track which symbols are "in trade" to avoid overlapping
    in_trade = {}  # sym -> entry_idx

    for entry_date, sym, entry_i in potential_entries:
        if sym in in_trade:
            continue

        df = all_symbol_data[sym]
        row = df.iloc[entry_i]
        entry_price = row['close']
        date_str = entry_date.strftime('%Y-%m-%d')

        # Context
        signals_today = signals_per_day.get(date_str, 1)
        atr_ratio = row['atr_ratio'] if pd.notna(row['atr_ratio']) else 1.0
        drawdown_20 = row['drawdown_20'] if pd.notna(row['drawdown_20']) else 0
        kc_width = row['kc_width'] if pd.notna(row['kc_width']) else 0

        # How far below KC6 lower is the entry? (deeper = more extended)
        kc_depth = (row['close'] / row['kc6_l'] - 1) * 100

        n = len(df)
        j = entry_i + 1
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

                if pnl <= 0 and exit_reason == 'STOP_LOSS':
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                all_trades.append({
                    'symbol': sym,
                    'entry_date': entry_date,
                    'pnl_pct': round(pnl, 2),
                    'hold_days': hold,
                    'exit_reason': exit_reason,
                    'win': pnl > 0,
                    'signals_today': signals_today,
                    'atr_ratio': round(atr_ratio, 2),
                    'drawdown_20': round(drawdown_20, 2),
                    'kc_width': round(kc_width, 2),
                    'kc_depth': round(kc_depth, 2),
                    'consec_losses_before': consecutive_losses - 1 if exit_reason == 'STOP_LOSS' else consecutive_losses,
                })
                if sym in in_trade:
                    del in_trade[sym]
                j += 1
                break
            j += 1

    return pd.DataFrame(all_trades)


def test_filter(df, filter_name, condition):
    """Test a filter: condition=True means BLOCK the trade."""
    blocked = df[condition]
    allowed = df[~condition]

    total_all = len(df)
    total_blocked = len(blocked)
    total_allowed = len(allowed)

    # Stats for blocked trades (these would have been avoided)
    bl_wins = blocked['win'].sum() if len(blocked) > 0 else 0
    bl_losses = total_blocked - bl_wins
    bl_pnl = blocked['pnl_pct'].sum() if len(blocked) > 0 else 0
    bl_sl = len(blocked[blocked['exit_reason'] == 'STOP_LOSS']) if len(blocked) > 0 else 0

    # Stats for allowed trades (these would still be taken)
    al_wr = allowed['win'].sum() / len(allowed) * 100 if len(allowed) > 0 else 0
    al_avg = allowed['pnl_pct'].mean() if len(allowed) > 0 else 0
    al_total = allowed['pnl_pct'].sum() if len(allowed) > 0 else 0
    al_pf = 0
    if len(allowed) > 0:
        gp = allowed[allowed['pnl_pct'] > 0]['pnl_pct'].sum()
        gl = abs(allowed[allowed['pnl_pct'] < 0]['pnl_pct'].sum())
        al_pf = gp / gl if gl > 0 else float('inf')

    # What % of SL trades did we catch?
    total_sl = len(df[df['exit_reason'] == 'STOP_LOSS'])
    sl_catch_rate = bl_sl / total_sl * 100 if total_sl > 0 else 0

    # What % of wins did we lose?
    total_wins = df['win'].sum()
    wins_lost = bl_wins / total_wins * 100 if total_wins > 0 else 0

    return {
        'filter': filter_name,
        'blocked': total_blocked,
        'blocked_pct': round(total_blocked / total_all * 100, 1),
        'blocked_pnl': round(bl_pnl, 1),
        'sl_caught': bl_sl,
        'sl_catch_rate': round(sl_catch_rate, 1),
        'wins_lost': bl_wins,
        'wins_lost_pct': round(wins_lost, 1),
        'allowed': total_allowed,
        'allowed_wr': round(al_wr, 1),
        'allowed_avg_pnl': round(al_avg, 2),
        'allowed_total_pnl': round(al_total, 1),
        'allowed_pf': round(al_pf, 2),
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()

    print("=" * 120)
    print("CRASH FILTER RESEARCH: Protecting KC6 V2 from Market Crashes")
    print("=" * 120)
    print(f"Running full backtest with trade context on {len(symbols)} stocks...")
    print()

    df = run_full_backtest_with_context(symbols, conn, sl_pct=5, tp_pct=15, max_hold=15)
    conn.close()

    total = len(df)
    baseline_wr = df['win'].sum() / total * 100
    baseline_pnl = df['pnl_pct'].sum()
    baseline_avg = df['pnl_pct'].mean()
    gp = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
    gl = abs(df[df['pnl_pct'] < 0]['pnl_pct'].sum())
    baseline_pf = gp / gl if gl > 0 else float('inf')

    print(f"Baseline: {total} trades | WR: {baseline_wr:.1f}% | "
          f"Avg P/L: {baseline_avg:+.2f}% | PF: {baseline_pf:.2f} | "
          f"Total P/L: {baseline_pnl:+.1f}%")

    # =====================================================================
    # PART 1: Study the crash patterns
    # =====================================================================
    print(f"\n\n{'='*120}")
    print("PART 1: CRASH PATTERN ANALYSIS")
    print(f"{'='*120}")

    # Look at distribution of signals_today for winning vs losing trades
    print("\n  A. Signals Per Day Distribution (wins vs losses):")
    print(f"  {'Signals/Day':>12s} {'Total':>7s} {'Wins':>6s} {'Losses':>7s} {'WR%':>6s} {'AvgP/L':>8s} {'SL hits':>8s}")
    print(f"  {'-'*60}")

    for bucket, label in [(1, '1'), (2, '2'), (3, '3'), ((4,5), '4-5'),
                           ((6,10), '6-10'), ((11,99), '11+')]:
        if isinstance(bucket, tuple):
            mask = (df['signals_today'] >= bucket[0]) & (df['signals_today'] <= bucket[1])
        else:
            mask = df['signals_today'] == bucket
        sub = df[mask]
        if len(sub) > 0:
            wr = sub['win'].sum() / len(sub) * 100
            avg = sub['pnl_pct'].mean()
            sl = len(sub[sub['exit_reason'] == 'STOP_LOSS'])
            print(f"  {label:>12s} {len(sub):>7d} {sub['win'].sum():>6d} "
                  f"{len(sub)-sub['win'].sum():>7d} {wr:>5.1f}% {avg:>+7.2f}% {sl:>7d}")

    # ATR ratio distribution
    print("\n  B. ATR Ratio at Entry (current ATR / 50-day avg ATR):")
    print(f"  {'ATR Ratio':>12s} {'Total':>7s} {'Wins':>6s} {'Losses':>7s} {'WR%':>6s} {'AvgP/L':>8s} {'SL hits':>8s}")
    print(f"  {'-'*60}")

    for lo, hi, label in [(0, 1.0, '<1.0'), (1.0, 1.3, '1.0-1.3'),
                           (1.3, 1.5, '1.3-1.5'), (1.5, 2.0, '1.5-2.0'),
                           (2.0, 3.0, '2.0-3.0'), (3.0, 99, '3.0+')]:
        mask = (df['atr_ratio'] >= lo) & (df['atr_ratio'] < hi)
        sub = df[mask]
        if len(sub) > 0:
            wr = sub['win'].sum() / len(sub) * 100
            avg = sub['pnl_pct'].mean()
            sl = len(sub[sub['exit_reason'] == 'STOP_LOSS'])
            print(f"  {label:>12s} {len(sub):>7d} {sub['win'].sum():>6d} "
                  f"{len(sub)-sub['win'].sum():>7d} {wr:>5.1f}% {avg:>+7.2f}% {sl:>7d}")

    # 20-day drawdown distribution
    print("\n  C. 20-Day Drawdown at Entry (how far from recent high):")
    print(f"  {'Drawdown':>12s} {'Total':>7s} {'Wins':>6s} {'Losses':>7s} {'WR%':>6s} {'AvgP/L':>8s} {'SL hits':>8s}")
    print(f"  {'-'*60}")

    for lo, hi, label in [(0, -3, '0 to -3%'), (-3, -6, '-3 to -6%'),
                           (-6, -10, '-6 to -10%'), (-10, -15, '-10 to -15%'),
                           (-15, -25, '-15 to -25%'), (-25, -99, '-25%+')]:
        mask = (df['drawdown_20'] <= lo) & (df['drawdown_20'] > hi)
        sub = df[mask]
        if len(sub) > 0:
            wr = sub['win'].sum() / len(sub) * 100
            avg = sub['pnl_pct'].mean()
            sl = len(sub[sub['exit_reason'] == 'STOP_LOSS'])
            print(f"  {label:>12s} {len(sub):>7d} {sub['win'].sum():>6d} "
                  f"{len(sub)-sub['win'].sum():>7d} {wr:>5.1f}% {avg:>+7.2f}% {sl:>7d}")

    # KC band width distribution
    print("\n  D. KC Band Width at Entry (wider = more volatile):")
    print(f"  {'KC Width':>12s} {'Total':>7s} {'Wins':>6s} {'Losses':>7s} {'WR%':>6s} {'AvgP/L':>8s} {'SL hits':>8s}")
    print(f"  {'-'*60}")

    for lo, hi, label in [(0, 3, '<3%'), (3, 5, '3-5%'), (5, 8, '5-8%'),
                           (8, 12, '8-12%'), (12, 20, '12-20%'), (20, 99, '20%+')]:
        mask = (df['kc_width'] >= lo) & (df['kc_width'] < hi)
        sub = df[mask]
        if len(sub) > 0:
            wr = sub['win'].sum() / len(sub) * 100
            avg = sub['pnl_pct'].mean()
            sl = len(sub[sub['exit_reason'] == 'STOP_LOSS'])
            print(f"  {label:>12s} {len(sub):>7d} {sub['win'].sum():>6d} "
                  f"{len(sub)-sub['win'].sum():>7d} {wr:>5.1f}% {avg:>+7.2f}% {sl:>7d}")

    # =====================================================================
    # PART 2: Test individual filters
    # =====================================================================
    print(f"\n\n{'='*120}")
    print("PART 2: INDIVIDUAL FILTER TESTS")
    print(f"{'='*120}")
    print(f"  Baseline: {total} trades | WR: {baseline_wr:.1f}% | PF: {baseline_pf:.2f} | Total P/L: {baseline_pnl:+.1f}%")
    print()

    filters = []

    # Filter 1: Signal clustering
    for threshold in [3, 5, 8, 10]:
        f = test_filter(df, f"Signals/day >= {threshold}",
                       df['signals_today'] >= threshold)
        filters.append(f)

    # Filter 2: ATR ratio
    for threshold in [1.5, 2.0, 2.5, 3.0]:
        f = test_filter(df, f"ATR ratio >= {threshold}",
                       df['atr_ratio'] >= threshold)
        filters.append(f)

    # Filter 3: 20-day drawdown
    for threshold in [-8, -10, -12, -15]:
        f = test_filter(df, f"Drawdown20 <= {threshold}%",
                       df['drawdown_20'] <= threshold)
        filters.append(f)

    # Filter 4: KC band width
    for threshold in [8, 10, 12, 15]:
        f = test_filter(df, f"KC width >= {threshold}%",
                       df['kc_width'] >= threshold)
        filters.append(f)

    # Filter 5: Combo - signals + ATR
    for sig_t, atr_t in [(3, 1.5), (5, 2.0), (3, 2.0)]:
        f = test_filter(df, f"Sig>={sig_t} AND ATR>={atr_t}",
                       (df['signals_today'] >= sig_t) & (df['atr_ratio'] >= atr_t))
        filters.append(f)

    # Filter 6: Combo - signals + drawdown
    for sig_t, dd_t in [(3, -10), (5, -10), (3, -8)]:
        f = test_filter(df, f"Sig>={sig_t} AND DD<={dd_t}%",
                       (df['signals_today'] >= sig_t) & (df['drawdown_20'] <= dd_t))
        filters.append(f)

    # Filter 7: Triple combo
    for sig_t, atr_t, dd_t in [(3, 1.5, -8), (3, 1.5, -10), (5, 1.5, -10)]:
        f = test_filter(df, f"Sig>={sig_t}+ATR>={atr_t}+DD<={dd_t}",
                       (df['signals_today'] >= sig_t) & (df['atr_ratio'] >= atr_t) & (df['drawdown_20'] <= dd_t))
        filters.append(f)

    # Print results
    print(f"  {'Filter':<35s} {'Blocked':>8s} {'BlkPnL':>8s} {'SL Caught':>10s} "
          f"{'Wins Lost':>10s} {'Allowed':>8s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} {'TotalP/L':>10s}")
    print(f"  {'-'*115}")

    for f in filters:
        print(f"  {f['filter']:<35s} {f['blocked']:>5d}({f['blocked_pct']:>4.1f}%) "
              f"{f['blocked_pnl']:>+7.1f}% {f['sl_caught']:>4d}({f['sl_catch_rate']:>4.1f}%) "
              f"{f['wins_lost']:>4d}({f['wins_lost_pct']:>4.1f}%) "
              f"{f['allowed']:>8d} {f['allowed_wr']:>5.1f}% {f['allowed_avg_pnl']:>+7.2f}% "
              f"{f['allowed_pf']:>5.2f} {f['allowed_total_pnl']:>+9.1f}%")

    # =====================================================================
    # PART 3: Best filters - show crash period impact
    # =====================================================================
    print(f"\n\n{'='*120}")
    print("PART 3: CRASH PERIOD IMPACT OF BEST FILTERS")
    print(f"{'='*120}")

    # Define crash periods
    crash_periods = [
        ('2008 GFC', '2008-01-01', '2008-12-31'),
        ('2011 Euro', '2011-07-01', '2012-01-31'),
        ('2015-16 China', '2015-08-01', '2016-03-31'),
        ('2018 NBFC', '2018-08-01', '2018-12-31'),
        ('2020 COVID', '2020-02-01', '2020-04-30'),
        ('2022 War/Rates', '2022-01-01', '2022-07-31'),
    ]

    # Best single filters to test on crash periods
    best_filters = [
        ("No Filter (baseline)", pd.Series([False] * len(df), index=df.index)),
        ("Signals/day >= 5", df['signals_today'] >= 5),
        ("ATR ratio >= 2.0", df['atr_ratio'] >= 2.0),
        ("Drawdown20 <= -10%", df['drawdown_20'] <= -10),
        ("Sig>=3 AND ATR>=1.5", (df['signals_today'] >= 3) & (df['atr_ratio'] >= 1.5)),
        ("Sig>=3 AND DD<=-10%", (df['signals_today'] >= 3) & (df['drawdown_20'] <= -10)),
        ("Sig>=3+ATR>=1.5+DD<=-10", (df['signals_today'] >= 3) & (df['atr_ratio'] >= 1.5) & (df['drawdown_20'] <= -10)),
    ]

    for period_name, start, end in crash_periods:
        crash_mask = (df['entry_date'] >= start) & (df['entry_date'] <= end)
        crash_trades = df[crash_mask]
        if len(crash_trades) == 0:
            continue

        print(f"\n  {period_name} ({start} to {end}): {len(crash_trades)} trades")
        print(f"  {'Filter':<35s} {'Trades':>7s} {'Blocked':>8s} {'WR%':>6s} {'TotalP/L':>10s} {'SL blocked':>11s}")
        print(f"  {'-'*85}")

        for fname, fmask in best_filters:
            blocked_in_crash = crash_trades[fmask[crash_mask].values if hasattr(fmask, 'values') else fmask.loc[crash_mask.values].values]
            allowed_in_crash = crash_trades[~fmask[crash_mask].values if hasattr(fmask, 'values') else ~fmask.loc[crash_mask.values].values]

            bl_count = len(blocked_in_crash)
            al_count = len(allowed_in_crash)
            al_wr = allowed_in_crash['win'].sum() / al_count * 100 if al_count > 0 else 0
            al_pnl = allowed_in_crash['pnl_pct'].sum() if al_count > 0 else 0
            bl_sl = len(blocked_in_crash[blocked_in_crash['exit_reason'] == 'STOP_LOSS']) if bl_count > 0 else 0

            print(f"  {fname:<35s} {al_count:>7d} {bl_count:>8d} "
                  f"{al_wr:>5.1f}% {al_pnl:>+9.1f}% {bl_sl:>10d}")

    # =====================================================================
    # PART 4: Year-by-year with best filter
    # =====================================================================
    print(f"\n\n{'='*120}")
    print("PART 4: YEAR-BY-YEAR -- BASELINE vs BEST FILTER")
    print(f"{'='*120}")

    df['year'] = df['entry_date'].dt.year

    # Use the best combo filter
    best_mask = (df['signals_today'] >= 3) & (df['atr_ratio'] >= 1.5) & (df['drawdown_20'] <= -10)
    df_filtered = df[~best_mask]

    print(f"  {'Year':<6s} | {'--- BASELINE ---':^35s} | {'--- WITH FILTER ---':^35s} | {'Saved':>8s}")
    print(f"  {'':6s} | {'Trades':>7s} {'WR%':>6s} {'AvgP/L':>8s} {'TotalP/L':>10s} | "
          f"{'Trades':>7s} {'WR%':>6s} {'AvgP/L':>8s} {'TotalP/L':>10s} | {'P/L':>8s}")
    print(f"  {'-'*95}")

    for yr in sorted(df['year'].unique()):
        base_yr = df[df['year'] == yr]
        filt_yr = df_filtered[df_filtered['entry_date'].dt.year == yr]

        if len(base_yr) < 3:
            continue

        b_wr = base_yr['win'].sum() / len(base_yr) * 100
        b_avg = base_yr['pnl_pct'].mean()
        b_tot = base_yr['pnl_pct'].sum()

        if len(filt_yr) > 0:
            f_wr = filt_yr['win'].sum() / len(filt_yr) * 100
            f_avg = filt_yr['pnl_pct'].mean()
            f_tot = filt_yr['pnl_pct'].sum()
        else:
            f_wr = 0
            f_avg = 0
            f_tot = 0

        saved = f_tot - b_tot
        print(f"  {yr:<6d} | {len(base_yr):>7d} {b_wr:>5.1f}% {b_avg:>+7.2f}% {b_tot:>+9.1f}% | "
              f"{len(filt_yr):>7d} {f_wr:>5.1f}% {f_avg:>+7.2f}% {f_tot:>+9.1f}% | {saved:>+7.1f}%")

    # Grand totals
    f_total_pnl = df_filtered['pnl_pct'].sum()
    print(f"  {'-'*95}")
    print(f"  {'TOTAL':<6s} | {len(df):>7d} {baseline_wr:>5.1f}% {baseline_avg:>+7.2f}% {baseline_pnl:>+9.1f}% | "
          f"{len(df_filtered):>7d} {df_filtered['win'].sum()/len(df_filtered)*100:>5.1f}% "
          f"{df_filtered['pnl_pct'].mean():>+7.2f}% {f_total_pnl:>+9.1f}% | "
          f"{f_total_pnl - baseline_pnl:>+7.1f}%")


if __name__ == '__main__':
    main()
