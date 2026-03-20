"""
BNF Squeeze & Fire — Parameter Sweep
=====================================
Optimizes both modes simultaneously.
Reports separate + combined statistics for each config.
"""

import sys
import os
import csv
import time
import logging
import sqlite3
import pandas as pd

logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.squeeze_fire_simulator import SqueezeFireSimulator, SqueezeFireConfig

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'squeeze_fire_summary.csv')
TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'squeeze_fire_trades.csv')

START_DATE = '2023-01-01'
END_DATE = '2025-12-31'

FIELDNAMES = [
    'label',
    # Config params
    'sq_strike_atr', 'sq_hold', 'sq_tf_max', 'sq_max_loss',
    'fr_width_atr', 'fr_hold', 'fr_sq_min', 'fr_tf_min',
    'fr_pt_pct', 'fr_sl_pct', 'fr_di', 'fr_rsi',
    'sq_lots', 'fr_lots',
    # Combined stats
    'total_trades', 'win_rate', 'profit_factor', 'total_pnl',
    'cagr', 'sharpe', 'max_drawdown', 'calmar',
    'final_equity', 'total_return_pct',
    # Squeeze stats
    'sq_trades', 'sq_wr', 'sq_pf', 'sq_avg_pnl_pct', 'sq_total_pnl',
    # Fire stats
    'fr_trades', 'fr_wr', 'fr_pf', 'fr_avg_pnl_pct', 'fr_total_pnl',
    'fr_long', 'fr_short',
    # Extras
    'avg_days_held', 'avg_dte',
]


def load_data():
    print('Loading BankNifty daily data...')
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume FROM market_data_unified
           WHERE symbol='BANKNIFTY' AND timeframe='day'
           AND date >= ? AND date <= ? ORDER BY date""",
        conn, params=(START_DATE, END_DATE)
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    print(f'Loaded {len(df)} bars: {df["date"].iloc[0].date()} to {df["date"].iloc[-1].date()}')
    return df


def build_configs():
    configs = []

    # Squeeze params to sweep
    sq_strikes = [1.0, 1.5, 2.0]
    sq_holds = [7, 10, 14]
    sq_tf_maxs = [1.5, 2.0, 2.5]
    sq_max_losses = [20000, 30000, 50000]

    # Fire params to sweep
    fr_widths = [1.0, 1.5, 2.0]
    fr_holds = [5, 7, 10]
    fr_sq_mins = [2, 3, 5]
    fr_tf_mins = [0.5, 1.0, 1.5]
    fr_pt_sls = [(0.50, 0.40), (0.80, 0.50), (1.00, 0.60)]
    fr_filters = [
        (False, False, 'none'),
        (True, False, 'DI'),
        (True, True, 'DI_RSI'),
    ]
    lot_combos = [(5, 5), (3, 5), (5, 3)]

    # Full sweep would be massive, so use a 2-phase approach:
    # Phase 1: Fix fire params at baseline, sweep squeeze params
    baseline_fire = dict(
        fire_spread_width_atr=1.5, fire_hold_bars=7,
        fire_squeeze_min_bars=3, fire_trend_atr_min=1.0,
        fire_profit_target_pct=0.50, fire_stop_loss_pct=0.40,
        fire_use_di_filter=False, fire_use_rsi_filter=False,
    )

    for ss in sq_strikes:
        for sh in sq_holds:
            for stf in sq_tf_maxs:
                for sml in sq_max_losses:
                    label = f'SQ_s{ss}_h{sh}_tf{stf}_ml{sml//1000}k__FR_base'
                    config = SqueezeFireConfig(
                        squeeze_strike_atr=ss, squeeze_hold_bars=sh,
                        squeeze_trend_atr_max=stf, squeeze_max_loss_rupees=sml,
                        squeeze_lots=5, fire_lots=5,
                        **baseline_fire,
                    )
                    configs.append((label, config))

    # Phase 2: Fix squeeze at best-known, sweep fire params
    baseline_squeeze = dict(
        squeeze_strike_atr=1.5, squeeze_hold_bars=10,
        squeeze_trend_atr_max=2.0, squeeze_max_loss_rupees=30000,
    )

    for fw in fr_widths:
        for fh in fr_holds:
            for fsm in fr_sq_mins:
                for ftm in fr_tf_mins:
                    for pt, sl in fr_pt_sls:
                        for di, rsi, filt_name in fr_filters:
                            for sl_lots, fl_lots in lot_combos:
                                label = (
                                    f'SQ_base__FR_w{fw}_h{fh}_sq{fsm}_tf{ftm}_'
                                    f'pt{int(pt*100)}_sl{int(sl*100)}_{filt_name}_'
                                    f'L{sl_lots}_{fl_lots}'
                                )
                                config = SqueezeFireConfig(
                                    fire_spread_width_atr=fw, fire_hold_bars=fh,
                                    fire_squeeze_min_bars=fsm, fire_trend_atr_min=ftm,
                                    fire_profit_target_pct=pt, fire_stop_loss_pct=sl,
                                    fire_use_di_filter=di, fire_use_rsi_filter=rsi,
                                    squeeze_lots=sl_lots, fire_lots=fl_lots,
                                    **baseline_squeeze,
                                )
                                configs.append((label, config))

    # Phase 3: Also test squeeze-only and fire-only for comparison
    for ss in sq_strikes:
        for sh in sq_holds:
            for stf in sq_tf_maxs:
                label = f'SQONLY_s{ss}_h{sh}_tf{stf}'
                config = SqueezeFireConfig(
                    squeeze_enabled=True, fire_enabled=False,
                    squeeze_strike_atr=ss, squeeze_hold_bars=sh,
                    squeeze_trend_atr_max=stf, squeeze_max_loss_rupees=30000,
                    squeeze_lots=5,
                )
                configs.append((label, config))

    return configs


def run_single(label, config, df):
    sim = SqueezeFireSimulator(config, df)
    summary = sim.run()
    trades = sim.get_trade_log()

    sq = summary.get('squeeze_stats', {})
    fr = summary.get('fire_stats', {})

    row = {
        'label': label,
        'sq_strike_atr': config.squeeze_strike_atr,
        'sq_hold': config.squeeze_hold_bars,
        'sq_tf_max': config.squeeze_trend_atr_max,
        'sq_max_loss': config.squeeze_max_loss_rupees,
        'fr_width_atr': config.fire_spread_width_atr,
        'fr_hold': config.fire_hold_bars,
        'fr_sq_min': config.fire_squeeze_min_bars,
        'fr_tf_min': config.fire_trend_atr_min,
        'fr_pt_pct': config.fire_profit_target_pct,
        'fr_sl_pct': config.fire_stop_loss_pct,
        'fr_di': config.fire_use_di_filter,
        'fr_rsi': config.fire_use_rsi_filter,
        'sq_lots': config.squeeze_lots,
        'fr_lots': config.fire_lots,
        'total_trades': summary['total_trades'],
        'win_rate': summary['win_rate'],
        'profit_factor': summary['profit_factor'],
        'total_pnl': summary['total_pnl'],
        'cagr': summary['cagr'],
        'sharpe': summary['sharpe'],
        'max_drawdown': summary['max_drawdown'],
        'calmar': summary['calmar'],
        'final_equity': summary['final_equity'],
        'total_return_pct': summary['total_return_pct'],
        'sq_trades': sq.get('count', 0),
        'sq_wr': sq.get('win_rate', 0),
        'sq_pf': sq.get('pf', 0),
        'sq_avg_pnl_pct': sq.get('avg_pnl_pct', 0),
        'sq_total_pnl': sq.get('total_pnl', 0),
        'fr_trades': fr.get('count', 0),
        'fr_wr': fr.get('win_rate', 0),
        'fr_pf': fr.get('pf', 0),
        'fr_avg_pnl_pct': fr.get('avg_pnl_pct', 0),
        'fr_total_pnl': fr.get('total_pnl', 0),
        'fr_long': summary.get('fire_long', 0),
        'fr_short': summary.get('fire_short', 0),
        'avg_days_held': summary.get('avg_days_held', 0),
        'avg_dte': summary.get('avg_dte', 0),
    }
    return row, trades


def main():
    df = load_data()
    all_configs = build_configs()
    total = len(all_configs)
    print(f'\nTotal configs: {total}')

    # Resume support
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        print(f'Skipping {len(done)} already-completed')

    if not done:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    all_trades = []
    completed = len(done)

    for label, config in all_configs:
        if label in done:
            continue

        t0 = time.time()
        try:
            row, trades = run_single(label, config, df)
            elapsed = time.time() - t0
            completed += 1

            with open(OUTPUT_CSV, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            for t in trades:
                t['config_label'] = label
            all_trades.extend(trades)

            print(
                f'[{completed}/{total}] {label} '
                f'{elapsed:.1f}s | '
                f'T={row["total_trades"]}(S{row["sq_trades"]}+F{row["fr_trades"]}) '
                f'WR={row["win_rate"]}% PF={row["profit_factor"]} '
                f'CAGR={row["cagr"]}% DD={row["max_drawdown"]}%'
            )
            sys.stdout.flush()

        except Exception as e:
            print(f'[ERROR] {label}: {e}')
            sys.stdout.flush()

    # Write trades
    if all_trades:
        trade_fields = list(all_trades[0].keys())
        with open(TRADES_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=trade_fields)
            w.writeheader()
            w.writerows(all_trades)
        print(f'\nTrades: {TRADES_CSV} ({len(all_trades)} trades)')

    # Show top results
    if os.path.exists(OUTPUT_CSV):
        results = pd.read_csv(OUTPUT_CSV)
        results = results[results['total_trades'] >= 10]

        print('\n' + '='*80)
        print('TOP 10 BY CAGR (min 10 trades)')
        print('='*80)
        top = results.nlargest(10, 'cagr')
        print(top[['label', 'total_trades', 'sq_trades', 'fr_trades',
                    'win_rate', 'profit_factor', 'cagr', 'max_drawdown', 'calmar']].to_string(index=False))

        print('\n' + '='*80)
        print('TOP 10 BY CALMAR (risk-adjusted, min 10 trades)')
        print('='*80)
        top_cal = results[results['max_drawdown'] > 0].nlargest(10, 'calmar')
        print(top_cal[['label', 'total_trades', 'sq_trades', 'fr_trades',
                        'win_rate', 'profit_factor', 'cagr', 'max_drawdown', 'calmar']].to_string(index=False))

        # Best squeeze-only
        sq_only = results[results['label'].str.startswith('SQONLY')]
        if not sq_only.empty:
            print('\n' + '='*80)
            print('TOP 5 SQUEEZE-ONLY')
            print('='*80)
            top_sq = sq_only.nlargest(5, 'cagr')
            print(top_sq[['label', 'total_trades', 'win_rate', 'profit_factor',
                          'cagr', 'max_drawdown', 'calmar']].to_string(index=False))


if __name__ == '__main__':
    main()
