"""
BankNifty Directional Debit Spread — V1 Backtest Sweep
=======================================================

Inverse of non-directional system:
- BB Squeeze Fire (expansion after squeeze) → breakout signal
- Close vs SMA(20) → long/short bias
- Debit spreads: Bull Call Spread (long) / Bear Put Spread (short)

Sweep parameters:
- Spread width: 1.0, 1.5, 2.0 ATR
- Long strike offset: ATM (0), 0.5 ATR OTM
- Hold bars: 5, 7, 10
- Squeeze min bars: 2, 3, 5
- Trend confirm threshold: off, 0.5, 1.0, 1.5 ATR
- Profit target / SL: (50/40), (80/50), (100/60)
- Bias filters: SMA only, SMA+DI, SMA+RSI, SMA+DI+RSI
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
from services.directional_simulator import DirectionalSimulator, DirectionalConfig

# =============================================================================
# Load BankNifty daily data
# =============================================================================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'directional_v1_summary.csv')
TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'directional_v1_trades.csv')

START_DATE = '2023-01-01'
END_DATE = '2025-12-31'

FIELDNAMES = [
    'label', 'spread_width_atr', 'long_offset_atr', 'hold_bars',
    'squeeze_min', 'trend_atr_min', 'pt_pct', 'sl_pct',
    'use_di', 'use_rsi', 'lots',
    'total_trades', 'long_trades', 'short_trades',
    'win_rate', 'profit_factor', 'total_pnl',
    'avg_pnl_pct', 'long_avg_pnl_pct', 'short_avg_pnl_pct',
    'cagr', 'sharpe', 'max_drawdown', 'calmar',
    'final_equity', 'total_return_pct',
    'avg_days_held', 'avg_dte', 'avg_debit_pts', 'avg_squeeze_bars',
    'exit_hold', 'exit_pt', 'exit_sl', 'exit_expiry',
]


def load_banknifty_data():
    print(f'Loading BankNifty daily data from {DB_PATH}...')
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol = 'NIFTY BANK'
        AND timeframe = 'day'
        AND date >= ? AND date <= ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(START_DATE, END_DATE))
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    print(f'Loaded {len(df)} bars: {df["date"].iloc[0].date()} to {df["date"].iloc[-1].date()}')
    return df


def build_configs():
    """Build all parameter combinations to test."""
    configs = []

    spread_widths = [1.0, 1.5, 2.0]
    long_offsets = [0.0, 0.5]
    hold_bars_list = [5, 7, 10]
    squeeze_mins = [2, 3, 5]
    trend_thresholds = [0.0, 0.5, 1.0, 1.5]  # 0 = off
    pt_sl_combos = [(0.50, 0.40), (0.80, 0.50), (1.00, 0.60)]
    bias_filters = [
        (False, False, 'SMA'),
        (True, False, 'SMA_DI'),
        (False, True, 'SMA_RSI'),
        (True, True, 'SMA_DI_RSI'),
    ]
    lots_list = [3, 5]

    for sw in spread_widths:
        for lo in long_offsets:
            for hb in hold_bars_list:
                for sq in squeeze_mins:
                    for tt in trend_thresholds:
                        for pt, sl in pt_sl_combos:
                            for di, rsi, bias_name in bias_filters:
                                for lots in lots_list:
                                    tf_str = f'TF{tt}' if tt > 0 else 'noTF'
                                    label = (
                                        f'SW{sw}_LO{lo}_H{hb}_SQ{sq}_{tf_str}_'
                                        f'PT{int(pt*100)}_SL{int(sl*100)}_{bias_name}_L{lots}'
                                    )
                                    config = DirectionalConfig(
                                        spread_width_atr=sw,
                                        long_strike_offset_atr=lo,
                                        hold_bars=hb,
                                        squeeze_min_bars=sq,
                                        use_trend_confirm=tt > 0,
                                        trend_atr_min=tt,
                                        profit_target_pct=pt,
                                        stop_loss_pct=sl,
                                        use_di_filter=di,
                                        use_rsi_filter=rsi,
                                        lots_per_trade=lots,
                                    )
                                    configs.append((label, config))

    return configs


def run_single(label, config, df):
    """Run a single backtest config."""
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
    df = load_banknifty_data()
    all_configs = build_configs()
    total = len(all_configs)
    print(f'\nTotal configs to test: {total}')

    # Check already done
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        print(f'Skipping {len(done)} already-completed configs')

    # Write headers if fresh
    if not done:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Track all trades for best configs
    all_trades = []
    completed = len(done)

    for i, (label, config) in enumerate(all_configs):
        if label in done:
            continue

        t0 = time.time()
        try:
            row, trades = run_single(label, config, df)
            elapsed = time.time() - t0
            completed += 1

            # Write summary row
            with open(OUTPUT_CSV, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            # Collect trades
            for t in trades:
                t['config_label'] = label
            all_trades.extend(trades)

            print(
                f'[{completed}/{total}] {label} '
                f'{elapsed:.1f}s | '
                f'Trades={row["total_trades"]} '
                f'WR={row["win_rate"]}% '
                f'PF={row["profit_factor"]} '
                f'CAGR={row["cagr"]}% '
                f'DD={row["max_drawdown"]}%'
            )
            sys.stdout.flush()

        except Exception as e:
            print(f'[{completed}/{total}] {label} ERROR: {e}')
            sys.stdout.flush()

    # Write all trades
    if all_trades:
        trade_fields = list(all_trades[0].keys())
        with open(TRADES_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=trade_fields)
            w.writeheader()
            w.writerows(all_trades)
        print(f'\nTrade log: {TRADES_CSV} ({len(all_trades)} trades)')

    print(f'\nSummary: {OUTPUT_CSV} ({completed} configs)')

    # Show top 10 by CAGR
    if os.path.exists(OUTPUT_CSV):
        results = pd.read_csv(OUTPUT_CSV)
        results = results[results['total_trades'] >= 5]  # Min 5 trades
        top = results.nlargest(10, 'cagr')
        print('\n=== TOP 10 BY CAGR (min 5 trades) ===')
        print(top[['label', 'total_trades', 'win_rate', 'profit_factor',
                    'cagr', 'max_drawdown', 'calmar', 'total_return_pct']].to_string(index=False))

        # Top 10 by Calmar
        top_calmar = results[results['max_drawdown'] > 0].nlargest(10, 'calmar')
        print('\n=== TOP 10 BY CALMAR (risk-adjusted) ===')
        print(top_calmar[['label', 'total_trades', 'win_rate', 'profit_factor',
                          'cagr', 'max_drawdown', 'calmar']].to_string(index=False))


if __name__ == '__main__':
    main()
