"""
ORB Deep Dive - Focused optimization of the best-performing strategy
=====================================================================
ORB15 with daily EMA20 trend filter achieved PF=1.136, WR=49.4%, 12,357 trades.
Now testing:
1. Finer ORB windows (10, 15, 20, 25 min)
2. Different daily trend filters (EMA10, EMA20, EMA50, EMA20+50 combo)
3. Volume filter on breakout bar
4. Time-of-day restrictions for entry
5. ORB range width filters (min/max range)
6. Per-stock performance breakdown
7. Long-only vs short-only analysis
8. Monthly performance consistency
"""

import sqlite3
import pandas as pd
import numpy as np
import csv
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research_results_orb_deep.csv')

FIELDNAMES = [
    'strategy', 'params', 'symbol', 'direction', 'total_trades', 'wins', 'losses',
    'win_rate', 'profit_factor', 'avg_win_pct', 'avg_loss_pct',
    'total_pnl_pct', 'max_win_pct', 'max_loss_pct',
    'avg_hold_bars', 'best_entry_hour', 'pct_months_profitable'
]

SLIPPAGE_PCT = 0.0005
BROKERAGE_RS = 40
ASSUMED_CAPITAL_PER_TRADE = 500_000
BROKERAGE_PCT = (BROKERAGE_RS / ASSUMED_CAPITAL_PER_TRADE) * 100

START_DATE = '2024-03-18'
END_DATE = '2026-03-12'


def load_data():
    conn = sqlite3.connect(DB_PATH)

    print("Loading 5-min data...", flush=True)
    q5 = """SELECT symbol, date, open, high, low, close, volume
            FROM market_data_unified
            WHERE timeframe='5minute'
            AND date >= ? AND date <= ?
            ORDER BY symbol, date"""
    df5 = pd.read_sql(q5, conn, params=[START_DATE, END_DATE + ' 23:59:59'])
    df5['date'] = pd.to_datetime(df5['date'])
    df5['trade_date'] = df5['date'].dt.date
    df5['time_str'] = df5['date'].dt.strftime('%H:%M')
    df5['hour'] = df5['date'].dt.hour
    df5['minute'] = df5['date'].dt.minute
    df5['bar_num'] = df5.groupby(['symbol', 'trade_date']).cumcount()
    print(f"  Loaded {len(df5):,} rows for {df5['symbol'].nunique()} symbols", flush=True)

    print("Loading daily data...", flush=True)
    symbols = df5['symbol'].unique().tolist()
    placeholders = ','.join(['?'] * len(symbols))
    qd = f"""SELECT symbol, date, open, high, low, close, volume
             FROM market_data_unified
             WHERE timeframe='day' AND symbol IN ({placeholders})
             AND date >= '2023-11-01' AND date <= ?
             ORDER BY symbol, date"""
    dfd = pd.read_sql(qd, conn, params=symbols + [END_DATE + ' 23:59:59'])
    dfd['date'] = pd.to_datetime(dfd['date'])
    print(f"  Loaded {len(dfd):,} daily rows", flush=True)

    conn.close()
    return df5, dfd, symbols


def compute_daily_features(dfd):
    """Compute multiple EMAs for trend filtering."""
    daily_map = {}
    for sym, gdf in dfd.groupby('symbol'):
        gdf = gdf.sort_values('date').copy()
        gdf['ema10'] = gdf['close'].ewm(span=10, adjust=False).mean()
        gdf['ema20'] = gdf['close'].ewm(span=20, adjust=False).mean()
        gdf['ema50'] = gdf['close'].ewm(span=50, adjust=False).mean()
        gdf['trade_date'] = gdf['date'].dt.date
        gdf['trend_ema10'] = gdf['close'] > gdf['ema10']
        gdf['trend_ema20'] = gdf['close'] > gdf['ema20']
        gdf['trend_ema50'] = gdf['close'] > gdf['ema50']
        gdf['trend_ema20_50'] = (gdf['close'] > gdf['ema20']) & (gdf['ema20'] > gdf['ema50'])
        gdf['daily_atr'] = (gdf['high'] - gdf['low']).rolling(14).mean()
        gdf['daily_atr_pct'] = gdf['daily_atr'] / gdf['close'] * 100
        daily_map[sym] = gdf.set_index('trade_date')
    return daily_map


def apply_costs_vec(entry_prices, exit_prices, is_long):
    """Vectorized. is_long: boolean array."""
    entry_prices = entry_prices.values if hasattr(entry_prices, 'values') else np.asarray(entry_prices)
    exit_prices = exit_prices.values if hasattr(exit_prices, 'values') else np.asarray(exit_prices)
    is_long = is_long.values if hasattr(is_long, 'values') else np.asarray(is_long)

    pnl = np.zeros(len(entry_prices))

    # Longs
    lm = is_long
    if lm.any():
        ee = entry_prices[lm] * (1 + SLIPPAGE_PCT)
        ex = exit_prices[lm] * (1 - SLIPPAGE_PCT)
        pnl[lm] = (ex - ee) / ee * 100 - BROKERAGE_PCT

    # Shorts
    sm = ~is_long
    if sm.any():
        ee = entry_prices[sm] * (1 - SLIPPAGE_PCT)
        ex = exit_prices[sm] * (1 + SLIPPAGE_PCT)
        pnl[sm] = (ee - ex) / ee * 100 - BROKERAGE_PCT

    return pnl


def run_orb(df5, daily_map, orb_bars, trend_filter, max_entry_time, target_mult,
            vol_filter_mult=None, min_range_pct=None, max_range_pct=None,
            direction_filter=None, atr_range_filter=None):
    """
    Run ORB strategy with many filter options.

    Args:
        orb_bars: Number of 5-min bars for opening range (3=15min, 4=20min, 6=30min)
        trend_filter: 'ema10', 'ema20', 'ema50', 'ema20_50', None
        max_entry_time: Latest entry time e.g. '12:00', '13:00', '14:00'
        target_mult: Target as multiple of range width (1.0, 1.5, 2.0)
        vol_filter_mult: If set, breakout bar volume must be > X * avg volume of ORB bars
        min_range_pct: Min ORB range as % of open price
        max_range_pct: Max ORB range as % of open price
        direction_filter: 'long_only', 'short_only', None
        atr_range_filter: If set, only trade when ORB range < X * daily ATR
    """
    # Compute ORB per symbol per day
    orb_mask = df5['bar_num'] < orb_bars
    orb_data = df5[orb_mask].groupby(['symbol', 'trade_date']).agg(
        orb_high=('high', 'max'),
        orb_low=('low', 'min'),
        orb_open=('open', 'first'),
        orb_vol_avg=('volume', 'mean')
    ).reset_index()
    orb_data['orb_range'] = orb_data['orb_high'] - orb_data['orb_low']
    orb_data['orb_range_pct'] = orb_data['orb_range'] / orb_data['orb_open'] * 100
    orb_data = orb_data[orb_data['orb_range'] > 0]

    # Range filters
    if min_range_pct is not None:
        orb_data = orb_data[orb_data['orb_range_pct'] >= min_range_pct]
    if max_range_pct is not None:
        orb_data = orb_data[orb_data['orb_range_pct'] <= max_range_pct]

    # Add daily trend info
    if trend_filter is not None:
        trend_col = f'trend_{trend_filter}'
        trend_rows = []
        for sym in orb_data['symbol'].unique():
            if sym in daily_map:
                dm = daily_map[sym]
                for td in orb_data[orb_data['symbol'] == sym]['trade_date'].unique():
                    if td in dm.index:
                        row = dm.loc[td]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[-1]
                        val = row.get(trend_col, None)
                        atr_val = row.get('daily_atr', None)
                        if val is not None:
                            trend_rows.append({'symbol': sym, 'trade_date': td,
                                             'trend_up': bool(val), 'daily_atr': atr_val})
        if not trend_rows:
            return pd.DataFrame()
        trend_df = pd.DataFrame(trend_rows)
        orb_data = orb_data.merge(trend_df, on=['symbol', 'trade_date'], how='inner')
    else:
        orb_data['trend_up'] = True  # no filter
        orb_data['daily_atr'] = np.nan

    # ATR range filter
    if atr_range_filter is not None and 'daily_atr' in orb_data.columns:
        orb_data = orb_data.dropna(subset=['daily_atr'])
        orb_data = orb_data[orb_data['orb_range'] < atr_range_filter * orb_data['daily_atr']]

    if len(orb_data) == 0:
        return pd.DataFrame()

    # Merge with 5-min data
    post_orb = df5.merge(
        orb_data[['symbol', 'trade_date', 'orb_high', 'orb_low', 'orb_range',
                  'orb_range_pct', 'orb_vol_avg', 'trend_up']],
        on=['symbol', 'trade_date'], how='inner')

    post_orb = post_orb[(post_orb['bar_num'] >= orb_bars) & (post_orb['time_str'] <= max_entry_time)]

    # Volume filter on breakout bar
    if vol_filter_mult is not None:
        post_orb = post_orb[post_orb['volume'] > vol_filter_mult * post_orb['orb_vol_avg']]

    # Identify breakouts
    if direction_filter == 'long_only':
        post_orb['long_break'] = post_orb['close'] > post_orb['orb_high']
        post_orb['short_break'] = False
    elif direction_filter == 'short_only':
        post_orb['long_break'] = False
        post_orb['short_break'] = post_orb['close'] < post_orb['orb_low']
    else:
        post_orb['long_break'] = post_orb['close'] > post_orb['orb_high']
        post_orb['short_break'] = post_orb['close'] < post_orb['orb_low']

    # Apply trend filter direction
    if trend_filter is not None:
        post_orb['long_break'] = post_orb['long_break'] & post_orb['trend_up']
        post_orb['short_break'] = post_orb['short_break'] & ~post_orb['trend_up']

    post_orb['any_break'] = post_orb['long_break'] | post_orb['short_break']

    # First breakout per symbol per day
    first_breaks = post_orb[post_orb['any_break']].sort_values(
        ['symbol', 'trade_date', 'bar_num']).groupby(['symbol', 'trade_date']).first().reset_index()

    if len(first_breaks) == 0:
        return pd.DataFrame()

    first_breaks['entry_price'] = first_breaks['close']
    first_breaks['entry_bar_num'] = first_breaks['bar_num']
    first_breaks['entry_hour'] = first_breaks['hour']
    first_breaks['is_long'] = first_breaks['long_break']

    # Compute SL/Target
    first_breaks['sl'] = np.where(first_breaks['is_long'],
        first_breaks['orb_low'], first_breaks['orb_high'])
    first_breaks['target'] = np.where(first_breaks['is_long'],
        first_breaks['entry_price'] + target_mult * first_breaks['orb_range'],
        first_breaks['entry_price'] - target_mult * first_breaks['orb_range'])

    # Scan for exits
    entry_cols = ['symbol', 'trade_date', 'entry_price', 'entry_bar_num', 'entry_hour',
                  'is_long', 'sl', 'target', 'orb_range_pct']
    exit_scan = df5.merge(first_breaks[entry_cols], on=['symbol', 'trade_date'], how='inner')
    exit_scan = exit_scan[exit_scan['bar_num'] > exit_scan['entry_bar_num']]

    if len(exit_scan) == 0:
        return pd.DataFrame()

    exit_scan['hit_sl'] = np.where(exit_scan['is_long'],
        exit_scan['low'] <= exit_scan['sl'], exit_scan['high'] >= exit_scan['sl'])
    exit_scan['hit_tgt'] = np.where(exit_scan['is_long'],
        exit_scan['high'] >= exit_scan['target'], exit_scan['low'] <= exit_scan['target'])
    exit_scan['hit_eod'] = exit_scan['time_str'] >= '15:20'
    exit_scan['hit_any'] = exit_scan['hit_sl'] | exit_scan['hit_tgt'] | exit_scan['hit_eod']

    exits = exit_scan[exit_scan['hit_any']].sort_values(
        ['symbol', 'trade_date', 'bar_num']).groupby(['symbol', 'trade_date']).first().reset_index()

    if len(exits) == 0:
        return pd.DataFrame()

    exits['exit_price'] = np.where(exits['hit_sl'], exits['sl'],
        np.where(exits['hit_tgt'], exits['target'], exits['close']))
    exits['hold_bars'] = exits['bar_num'] - exits['entry_bar_num']

    exits['pnl_pct'] = apply_costs_vec(exits['entry_price'], exits['exit_price'], exits['is_long'])
    exits['direction'] = np.where(exits['is_long'], 'long', 'short')
    exits['month'] = pd.to_datetime(exits['trade_date'].astype(str)).dt.to_period('M')

    return exits[['symbol', 'trade_date', 'pnl_pct', 'entry_hour', 'hold_bars',
                  'direction', 'is_long', 'month', 'orb_range_pct', 'entry_price',
                  'exit_price', 'sl', 'target']]


def summarize(trades_df, strategy_name, params_str, symbol='ALL', direction='both'):
    if trades_df is None or len(trades_df) == 0:
        return None

    pnls = trades_df['pnl_pct'].values
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    gross_wins = wins.sum() if len(wins) > 0 else 0
    gross_losses = abs(losses.sum()) if len(losses) > 0 else 0.001

    # Monthly consistency
    if 'month' in trades_df.columns:
        monthly_pnl = trades_df.groupby('month')['pnl_pct'].sum()
        pct_months_profit = (monthly_pnl > 0).sum() / len(monthly_pnl) * 100 if len(monthly_pnl) > 0 else 0
    else:
        pct_months_profit = 0

    hour_pnl = trades_df.groupby('entry_hour')['pnl_pct'].sum()
    best_hour = int(hour_pnl.idxmax()) if len(hour_pnl) > 0 else 0

    return {
        'strategy': strategy_name,
        'params': params_str,
        'symbol': symbol,
        'direction': direction,
        'total_trades': n,
        'wins': int(len(wins)),
        'losses': int(len(losses)),
        'win_rate': round(len(wins) / n * 100, 2),
        'profit_factor': round(gross_wins / gross_losses, 3),
        'avg_win_pct': round(wins.mean(), 4) if len(wins) > 0 else 0,
        'avg_loss_pct': round(losses.mean(), 4) if len(losses) > 0 else 0,
        'total_pnl_pct': round(pnls.sum(), 4),
        'max_win_pct': round(pnls.max(), 4),
        'max_loss_pct': round(pnls.min(), 4),
        'avg_hold_bars': round(trades_df['hold_bars'].mean(), 1),
        'best_entry_hour': best_hour,
        'pct_months_profitable': round(pct_months_profit, 1),
    }


def write_result(row, first_write_flag):
    mode = 'w' if first_write_flag[0] else 'a'
    with open(OUTPUT_CSV, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if first_write_flag[0]:
            writer.writeheader()
            first_write_flag[0] = False
        writer.writerow(row)


def main():
    t0 = time.time()
    first_write = [True]

    df5, dfd, symbols = load_data()
    daily_map = compute_daily_features(dfd)
    del dfd

    configs = []

    # Phase 1: ORB window + trend filter combos
    for orb_min in [10, 15, 20, 25, 30]:
        orb_bars = orb_min // 5
        for trend in [None, 'ema10', 'ema20', 'ema50', 'ema20_50']:
            for tgt in [1.0, 1.5, 2.0]:
                tf_label = trend if trend else 'NoTrend'
                label = f"ORB{orb_min}_{tf_label}_T{tgt}"
                configs.append((label, {
                    'orb_bars': orb_bars, 'trend_filter': trend,
                    'max_entry_time': '14:00', 'target_mult': tgt,
                }))

    # Phase 2: Best combos with entry time filters
    for max_entry in ['11:00', '12:00', '13:00']:
        for tgt in [1.0, 1.5]:
            label = f"ORB15_ema20_T{tgt}_before{max_entry.replace(':','')}"
            configs.append((label, {
                'orb_bars': 3, 'trend_filter': 'ema20',
                'max_entry_time': max_entry, 'target_mult': tgt,
            }))

    # Phase 3: Volume filter on breakout bar
    for vol_mult in [1.5, 2.0, 3.0]:
        label = f"ORB15_ema20_T1.5_VOL{vol_mult}"
        configs.append((label, {
            'orb_bars': 3, 'trend_filter': 'ema20',
            'max_entry_time': '14:00', 'target_mult': 1.5,
            'vol_filter_mult': vol_mult,
        }))

    # Phase 4: ORB range size filters
    for min_rng in [0.3, 0.5]:
        for max_rng in [1.5, 2.0]:
            label = f"ORB15_ema20_T1.5_RNG{min_rng}-{max_rng}"
            configs.append((label, {
                'orb_bars': 3, 'trend_filter': 'ema20',
                'max_entry_time': '14:00', 'target_mult': 1.5,
                'min_range_pct': min_rng, 'max_range_pct': max_rng,
            }))

    # Phase 5: ATR range filter
    for atr_f in [0.5, 0.75, 1.0]:
        label = f"ORB15_ema20_T1.5_ATR{atr_f}"
        configs.append((label, {
            'orb_bars': 3, 'trend_filter': 'ema20',
            'max_entry_time': '14:00', 'target_mult': 1.5,
            'atr_range_filter': atr_f,
        }))

    # Phase 6: Direction-only
    for d in ['long_only', 'short_only']:
        label = f"ORB15_ema20_T1.5_{d}"
        configs.append((label, {
            'orb_bars': 3, 'trend_filter': 'ema20',
            'max_entry_time': '14:00', 'target_mult': 1.5,
            'direction_filter': d,
        }))

    total = len(configs)
    print(f"\n{'='*80}")
    print(f"Running {total} ORB configs across {len(symbols)} symbols")
    print(f"{'='*80}\n", flush=True)

    results_summary = []

    for ci, (label, params) in enumerate(configs):
        t1 = time.time()

        trades = run_orb(df5, daily_map, **params)

        if trades is not None and len(trades) > 0:
            # Aggregate
            agg = summarize(trades, 'ORB', label, 'ALL', 'both')
            if agg:
                write_result(agg, first_write)
                results_summary.append(agg)

            # Per-direction
            for d in ['long', 'short']:
                dt = trades[trades['direction'] == d]
                if len(dt) >= 10:
                    dr = summarize(dt, 'ORB', label, 'ALL', d)
                    if dr:
                        write_result(dr, first_write)

            # Per-symbol for top configs
            if agg and agg['profit_factor'] > 1.05:
                for sym in trades['symbol'].unique():
                    st = trades[trades['symbol'] == sym]
                    if len(st) >= 10:
                        sr = summarize(st, 'ORB', label, sym, 'both')
                        if sr:
                            write_result(sr, first_write)

        elapsed = time.time() - t1
        if results_summary and results_summary[-1]['params'] == label:
            r = results_summary[-1]
            marker = ' ***' if r['profit_factor'] > 1.2 else (' **' if r['profit_factor'] > 1.1 else '')
            print(f"[{ci+1}/{total}] {label:45s} | Tr={r['total_trades']:5d} WR={r['win_rate']:5.1f}% "
                  f"PF={r['profit_factor']:5.3f} PnL={r['total_pnl_pct']:8.1f}% "
                  f"MoPr={r['pct_months_profitable']:4.0f}% | {elapsed:.1f}s{marker}", flush=True)
        else:
            print(f"[{ci+1}/{total}] {label:45s} | NO TRADES | {elapsed:.1f}s", flush=True)

    total_time = time.time() - t0

    print(f"\n{'='*80}")
    print(f"COMPLETED in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Results: {OUTPUT_CSV}")
    print(f"{'='*80}\n")

    if not results_summary:
        print("No results!")
        return

    df_res = pd.DataFrame(results_summary)
    df_res = df_res.sort_values('profit_factor', ascending=False)

    # All configs with PF > 1.0
    profitable = df_res[df_res['profit_factor'] > 1.0]
    print(f"\nPROFITABLE CONFIGS (PF > 1.0): {len(profitable)} of {len(df_res)}")
    print("-" * 130)
    print(f"{'Params':48s} {'Trades':>6s} {'WR%':>6s} {'PF':>6s} {'AvgW%':>7s} {'AvgL%':>8s} "
          f"{'TotPnL%':>9s} {'MoPr%':>6s} {'HoldBars':>8s}")
    print("-" * 130)
    for _, r in profitable.iterrows():
        marker = ' ***' if r['profit_factor'] > 1.2 else (' **' if r['profit_factor'] > 1.1 else '')
        print(f"{r['params']:48s} {r['total_trades']:6d} {r['win_rate']:5.1f}% {r['profit_factor']:6.3f} "
              f"{r['avg_win_pct']:6.4f}% {r['avg_loss_pct']:7.4f}% {r['total_pnl_pct']:8.1f}% "
              f"{r['pct_months_profitable']:5.1f}% {r['avg_hold_bars']:7.1f}{marker}")

    # Per-stock breakdown for best config
    if len(profitable) > 0:
        best_label = profitable.iloc[0]['params']
        print(f"\n\nPER-STOCK BREAKDOWN FOR BEST: {best_label}")
        all_rows = pd.read_csv(OUTPUT_CSV)
        stock_rows = all_rows[(all_rows['params'] == best_label) & (all_rows['symbol'] != 'ALL')
                              & (all_rows['direction'] == 'both')]
        stock_rows = stock_rows.sort_values('total_pnl_pct', ascending=False)
        print(f"{'Symbol':15s} {'Trades':>7s} {'WR%':>7s} {'PF':>7s} {'TotPnL%':>9s} {'MoPr%':>6s}")
        print("-" * 60)
        for _, r in stock_rows.iterrows():
            marker = ' ***' if r['profit_factor'] > 1.2 else ''
            print(f"{r['symbol']:15s} {r['total_trades']:7.0f} {r['win_rate']:6.1f}% "
                  f"{r['profit_factor']:7.3f} {r['total_pnl_pct']:8.2f}% {r['pct_months_profitable']:5.1f}%{marker}")

        # Long vs Short for best config
        print(f"\n\nLONG vs SHORT FOR BEST: {best_label}")
        for d in ['long', 'short']:
            dr = all_rows[(all_rows['params'] == best_label) & (all_rows['symbol'] == 'ALL')
                          & (all_rows['direction'] == d)]
            if len(dr) > 0:
                r = dr.iloc[0]
                print(f"  {d:6s}: Trades={r['total_trades']:5.0f} WR={r['win_rate']:5.1f}% "
                      f"PF={r['profit_factor']:5.3f} PnL={r['total_pnl_pct']:8.2f}%")

    # Top 5 with best monthly consistency
    consistent = df_res[df_res['total_trades'] >= 100].sort_values('pct_months_profitable', ascending=False)
    if len(consistent) > 0:
        print(f"\n\nMOST CONSISTENT (by % months profitable, 100+ trades):")
        print("-" * 130)
        for _, r in consistent.head(10).iterrows():
            print(f"{r['params']:48s} Tr={r['total_trades']:5d} WR={r['win_rate']:5.1f}% "
                  f"PF={r['profit_factor']:5.3f} PnL={r['total_pnl_pct']:8.1f}% MoPr={r['pct_months_profitable']:5.1f}%")


if __name__ == '__main__':
    main()
