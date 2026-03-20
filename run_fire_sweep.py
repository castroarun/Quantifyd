"""
BNF Fire Mode — Directional Debit Spread Optimization
======================================================
Phase 1: Coarse sweep (~360 configs, ~15 min)
Phase 2: Refine around best (if needed)
"""

import sys
import os
import csv
import time
import logging
import sqlite3
import pandas as pd
import numpy as np

logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.directional_simulator import DirectionalSimulator, DirectionalConfig

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fire_sweep_summary.csv')
TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fire_sweep_trades.csv')

FIELDNAMES = [
    'label',
    'spread_width_atr', 'long_offset_atr', 'hold_bars',
    'squeeze_min', 'trend_atr_min',
    'pt_pct', 'sl_pct', 'use_di', 'use_rsi', 'lots',
    'total_trades', 'long_trades', 'short_trades',
    'win_rate', 'profit_factor', 'total_pnl',
    'avg_pnl_pct', 'long_avg_pnl_pct', 'short_avg_pnl_pct',
    'cagr', 'sharpe', 'max_drawdown', 'calmar',
    'final_equity', 'total_return_pct',
    'avg_days_held', 'avg_dte', 'avg_debit_pts', 'avg_squeeze_bars',
    'exit_hold', 'exit_pt', 'exit_sl', 'exit_expiry',
]


def load_data():
    print('Loading BankNifty daily data...')
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume FROM market_data_unified
           WHERE symbol='BANKNIFTY' AND timeframe='day'
           AND date >= '2023-01-01' AND date <= '2025-12-31' ORDER BY date""",
        conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    print(f'Loaded {len(df)} bars')
    return df


def run_single(label, config, df):
    sim = DirectionalSimulator(config, df)
    summary = sim.run()
    trades = sim.get_trade_log()

    exit_reasons = summary.get('exit_reasons', {})
    row = {
        'label': label,
        'spread_width_atr': config.spread_width_atr,
        'long_offset_atr': config.long_strike_offset_atr,
        'hold_bars': config.hold_bars,
        'squeeze_min': config.squeeze_min_bars,
        'trend_atr_min': config.trend_atr_min if config.use_trend_confirm else 0,
        'pt_pct': config.profit_target_pct,
        'sl_pct': config.stop_loss_pct,
        'use_di': config.use_di_filter,
        'use_rsi': config.use_rsi_filter,
        'lots': config.lots_per_trade,
        'total_trades': summary['total_trades'],
        'long_trades': summary.get('long_trades', 0),
        'short_trades': summary.get('short_trades', 0),
        'win_rate': summary['win_rate'],
        'profit_factor': summary['profit_factor'],
        'total_pnl': summary['total_pnl'],
        'avg_pnl_pct': summary.get('avg_pnl_pct', 0),
        'long_avg_pnl_pct': summary.get('long_avg_pnl_pct', 0),
        'short_avg_pnl_pct': summary.get('short_avg_pnl_pct', 0),
        'cagr': summary['cagr'],
        'sharpe': summary['sharpe'],
        'max_drawdown': summary['max_drawdown'],
        'calmar': summary.get('calmar', 0),
        'final_equity': summary.get('final_equity', 0),
        'total_return_pct': summary.get('total_return_pct', 0),
        'avg_days_held': summary.get('avg_days_held', 0),
        'avg_dte': summary.get('avg_dte', 0),
        'avg_debit_pts': summary.get('avg_debit_pts', 0),
        'avg_squeeze_bars': summary.get('avg_squeeze_bars', 0),
        'exit_hold': exit_reasons.get('hold_complete', 0),
        'exit_pt': exit_reasons.get('profit_target', 0),
        'exit_sl': exit_reasons.get('stop_loss', 0),
        'exit_expiry': exit_reasons.get('expiry', 0),
    }
    return row, trades


def main():
    df = load_data()

    configs = []

    # Coarse sweep: most impactful params
    for sw in [0.5, 1.0, 1.5, 2.0, 2.5]:
        for lo in [0.0, 0.5]:
            for hb in [3, 5, 7, 10, 14]:
                for sq in [2, 3, 5]:
                    for tt in [0.0, 0.5, 1.0, 1.5]:
                        for pt, sl in [(0.50, 0.40), (1.00, 0.60), (1.50, 0.70)]:
                            # SMA only bias (most configs)
                            tf_str = f'TF{tt}' if tt > 0 else 'noTF'
                            label = (
                                f'SW{sw}_LO{lo}_H{hb}_SQ{sq}_{tf_str}_'
                                f'PT{int(pt*100)}_SL{int(sl*100)}_SMA_L5'
                            )
                            config = DirectionalConfig(
                                spread_width_atr=sw,
                                long_strike_offset_atr=lo,
                                hold_bars=hb,
                                squeeze_min_bars=sq,
                                use_trend_confirm=tt > 0,
                                trend_atr_min=tt if tt > 0 else 1.0,
                                profit_target_pct=pt,
                                stop_loss_pct=sl,
                                lots_per_trade=5,
                            )
                            configs.append((label, config))

    # Add DI and DI+RSI variants for promising combos
    for sw in [1.0, 1.5, 2.0]:
        for hb in [5, 7, 10]:
            for sq in [3, 5]:
                for tt in [0.0, 1.0]:
                    for pt, sl in [(0.50, 0.40), (1.00, 0.60)]:
                        tf_str = f'TF{tt}' if tt > 0 else 'noTF'
                        for di, rsi, name in [(True, False, 'DI'), (True, True, 'DI_RSI')]:
                            label = (
                                f'SW{sw}_LO0.0_H{hb}_SQ{sq}_{tf_str}_'
                                f'PT{int(pt*100)}_SL{int(sl*100)}_{name}_L5'
                            )
                            config = DirectionalConfig(
                                spread_width_atr=sw,
                                long_strike_offset_atr=0.0,
                                hold_bars=hb,
                                squeeze_min_bars=sq,
                                use_trend_confirm=tt > 0,
                                trend_atr_min=tt if tt > 0 else 1.0,
                                profit_target_pct=pt,
                                stop_loss_pct=sl,
                                use_di_filter=di,
                                use_rsi_filter=rsi,
                                lots_per_trade=5,
                            )
                            configs.append((label, config))

    # Also test long-only (no shorts) — quick check if shorts hurt
    for sw in [1.0, 1.5, 2.0]:
        for hb in [5, 7, 10]:
            for sq in [3, 5]:
                for pt, sl in [(0.50, 0.40), (1.00, 0.60)]:
                    label = f'LONGONLY_SW{sw}_H{hb}_SQ{sq}_PT{int(pt*100)}_SL{int(sl*100)}'
                    config = DirectionalConfig(
                        spread_width_atr=sw,
                        hold_bars=hb,
                        squeeze_min_bars=sq,
                        use_trend_confirm=False,
                        profit_target_pct=pt,
                        stop_loss_pct=sl,
                        lots_per_trade=5,
                        # We can't disable shorts in config, but we'll track long-only in analysis
                    )
                    configs.append((label, config))

    # Deduplicate
    seen = set()
    unique = []
    for label, config in configs:
        if label not in seen:
            seen.add(label)
            unique.append((label, config))
    configs = unique

    total = len(configs)
    print(f'Total configs: {total} (est {total * 2.5 / 60:.0f} min)')

    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        print(f'Skipping {len(done)} done')

    if not done:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    all_trades = []
    completed = len(done)
    batch_start = time.time()

    for label, config in configs:
        if label in done:
            continue

        t0 = time.time()
        try:
            row, trades = run_single(label, config, df)
            elapsed = time.time() - t0
            completed += 1

            with open(OUTPUT_CSV, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            if row['total_trades'] >= 3:
                for t in trades:
                    t['config_label'] = label
                all_trades.extend(trades)

            # Print every 100th or if notable
            if completed % 100 == 0 or (row['total_trades'] >= 5 and row['cagr'] > 5):
                print(
                    f'[{completed}/{total}] {label} '
                    f'{elapsed:.1f}s | '
                    f'T={row["total_trades"]}({row["long_trades"]}L/{row["short_trades"]}S) '
                    f'WR={row["win_rate"]}% PF={row["profit_factor"]} '
                    f'CAGR={row["cagr"]}% DD={row["max_drawdown"]}%'
                )
                sys.stdout.flush()

        except Exception as e:
            print(f'[ERROR] {label}: {e}')
            sys.stdout.flush()

    total_time = time.time() - batch_start
    print(f'\nDone: {completed} configs in {total_time:.0f}s ({total_time/60:.1f}min)')

    # Write trades
    if all_trades:
        trade_fields = list(all_trades[0].keys())
        with open(TRADES_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=trade_fields)
            w.writeheader()
            w.writerows(all_trades)

    # Analysis
    results = pd.read_csv(OUTPUT_CSV)
    has_trades = results[results['total_trades'] >= 3]

    print(f'\n{"="*100}')
    print(f'{len(results)} total configs | {len(has_trades)} with >= 3 trades')
    print(f'{"="*100}')

    if has_trades.empty:
        print('No configs with >= 3 trades!')
        return

    print('\nTOP 20 BY CAGR:')
    top = has_trades.nlargest(20, 'cagr')
    cols = ['label', 'total_trades', 'long_trades', 'short_trades',
            'win_rate', 'profit_factor', 'cagr', 'max_drawdown', 'calmar',
            'avg_pnl_pct', 'total_pnl']
    print(top[cols].to_string(index=False))

    print('\nTOP 20 BY CALMAR (DD > 0):')
    cal_df = has_trades[has_trades['max_drawdown'] > 0]
    if not cal_df.empty:
        top_cal = cal_df.nlargest(20, 'calmar')
        print(top_cal[cols].to_string(index=False))

    # Direction analysis
    print('\n--- Direction Analysis ---')
    print(f'Avg Long PnL%: {has_trades["long_avg_pnl_pct"].mean():.2f}')
    print(f'Avg Short PnL%: {has_trades["short_avg_pnl_pct"].mean():.2f}')
    print(f'Configs where long > short: {(has_trades["long_avg_pnl_pct"] > has_trades["short_avg_pnl_pct"]).sum()}/{len(has_trades)}')

    # Param importance
    print('\n--- Param Impact on CAGR (mean) ---')
    for param in ['spread_width_atr', 'hold_bars', 'squeeze_min', 'trend_atr_min', 'pt_pct']:
        grouped = has_trades.groupby(param)['cagr'].mean()
        print(f'\n{param}:')
        print(grouped.sort_values(ascending=False).to_string())


if __name__ == '__main__':
    main()
