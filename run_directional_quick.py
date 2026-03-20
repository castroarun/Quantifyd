"""
Quick sanity test — run ~12 representative configs to validate the simulator works
and get a baseline read on signal quality before full sweep.
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

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')

START_DATE = '2023-01-01'
END_DATE = '2025-12-31'


def load_data():
    print(f'Loading BankNifty daily data...')
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


def main():
    df = load_data()

    configs = [
        # Baseline: ATM, 1.5 ATR width, 7 day hold, SMA only, no trend confirm
        ('baseline_SW1.5_H7_SQ3', DirectionalConfig(
            spread_width_atr=1.5, hold_bars=7, squeeze_min_bars=3,
            use_trend_confirm=False, lots_per_trade=5,
        )),
        # Wider spread
        ('wide_SW2.0_H7_SQ3', DirectionalConfig(
            spread_width_atr=2.0, hold_bars=7, squeeze_min_bars=3,
            use_trend_confirm=False, lots_per_trade=5,
        )),
        # Narrow spread
        ('narrow_SW1.0_H7_SQ3', DirectionalConfig(
            spread_width_atr=1.0, hold_bars=7, squeeze_min_bars=3,
            use_trend_confirm=False, lots_per_trade=5,
        )),
        # OTM long leg
        ('otm_SW1.5_LO0.5_H7_SQ3', DirectionalConfig(
            spread_width_atr=1.5, long_strike_offset_atr=0.5, hold_bars=7,
            squeeze_min_bars=3, use_trend_confirm=False, lots_per_trade=5,
        )),
        # With trend confirmation
        ('trend_SW1.5_H7_SQ3_TF1.0', DirectionalConfig(
            spread_width_atr=1.5, hold_bars=7, squeeze_min_bars=3,
            use_trend_confirm=True, trend_atr_min=1.0, lots_per_trade=5,
        )),
        # Stricter trend
        ('trend_SW1.5_H7_SQ3_TF1.5', DirectionalConfig(
            spread_width_atr=1.5, hold_bars=7, squeeze_min_bars=3,
            use_trend_confirm=True, trend_atr_min=1.5, lots_per_trade=5,
        )),
        # Longer squeeze
        ('longsq_SW1.5_H7_SQ5', DirectionalConfig(
            spread_width_atr=1.5, hold_bars=7, squeeze_min_bars=5,
            use_trend_confirm=False, lots_per_trade=5,
        )),
        # Longer hold
        ('longhold_SW1.5_H10_SQ3', DirectionalConfig(
            spread_width_atr=1.5, hold_bars=10, squeeze_min_bars=3,
            use_trend_confirm=False, lots_per_trade=5,
        )),
        # Shorter hold
        ('shorthold_SW1.5_H5_SQ3', DirectionalConfig(
            spread_width_atr=1.5, hold_bars=5, squeeze_min_bars=3,
            use_trend_confirm=False, lots_per_trade=5,
        )),
        # Aggressive PT/SL
        ('aggr_SW1.5_H7_SQ3_PT80_SL50', DirectionalConfig(
            spread_width_atr=1.5, hold_bars=7, squeeze_min_bars=3,
            use_trend_confirm=False, profit_target_pct=0.80, stop_loss_pct=0.50,
            lots_per_trade=5,
        )),
        # With DI filter
        ('di_SW1.5_H7_SQ3', DirectionalConfig(
            spread_width_atr=1.5, hold_bars=7, squeeze_min_bars=3,
            use_trend_confirm=False, use_di_filter=True, lots_per_trade=5,
        )),
        # Full combo: trend + DI + RSI
        ('combo_SW1.5_H7_SQ3_TF1.0', DirectionalConfig(
            spread_width_atr=1.5, hold_bars=7, squeeze_min_bars=3,
            use_trend_confirm=True, trend_atr_min=1.0,
            use_di_filter=True, use_rsi_filter=True, lots_per_trade=5,
        )),
    ]

    print(f'\nRunning {len(configs)} quick test configs...\n')
    print(f'{"Label":<40} {"Trades":>6} {"L/S":>7} {"WR%":>5} {"PF":>6} '
          f'{"CAGR%":>7} {"DD%":>6} {"Calmar":>7} {"AvgPnl%":>8} {"TotalPnL":>10}')
    print('-' * 115)

    for label, config in configs:
        t0 = time.time()
        sim = DirectionalSimulator(config, df)
        summary = sim.run()
        elapsed = time.time() - t0

        lt = summary.get('long_trades', 0)
        st = summary.get('short_trades', 0)

        print(
            f'{label:<40} {summary["total_trades"]:>6} '
            f'{lt:>3}/{st:<3} '
            f'{summary["win_rate"]:>5.1f} '
            f'{summary["profit_factor"]:>6.2f} '
            f'{summary["cagr"]:>7.2f} '
            f'{summary["max_drawdown"]:>6.2f} '
            f'{summary.get("calmar", 0):>7.2f} '
            f'{summary.get("avg_pnl_pct", 0):>8.2f} '
            f'{summary.get("total_pnl", 0):>10.0f}'
        )

        # Print trade details for first config
        if label == 'baseline_SW1.5_H7_SQ3':
            trades = sim.get_trade_log()
            if trades:
                print(f'\n  --- Sample trades (baseline) ---')
                for t in trades[:10]:
                    print(
                        f'  #{t["trade_id"]} {t["direction"]:>5} '
                        f'{t["entry_date"]} -> {t["exit_date"]} '
                        f'Spot={t["entry_spot"]:>8.0f} '
                        f'Strikes={t["long_strike"]:.0f}/{t["short_strike"]:.0f} '
                        f'Debit={t["net_debit"]:>6.1f} '
                        f'PnL={t["pnl_pct"]:>7.1f}% '
                        f'Rs={t["pnl_rupees"]:>8.0f} '
                        f'{t["exit_reason"]}'
                    )
                print()

    sys.stdout.flush()


if __name__ == '__main__':
    main()
