"""
BNF Squeeze & Fire — Quick sanity test
Validates both modes work, shows separate + combined stats
"""

import sys
import os
import time
import logging
import sqlite3
import pandas as pd

logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.squeeze_fire_simulator import SqueezeFireSimulator, SqueezeFireConfig

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'


def load_data():
    print('Loading BankNifty daily data...')
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume
           FROM market_data_unified
           WHERE symbol = 'BANKNIFTY' AND timeframe = 'day'
           AND date >= ? AND date <= ? ORDER BY date""",
        conn, params=(START_DATE, END_DATE)
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    print(f'Loaded {len(df)} bars: {df["date"].iloc[0].date()} to {df["date"].iloc[-1].date()}')
    return df


def print_mode_stats(label, stats):
    if stats['count'] == 0:
        print(f'  {label}: No trades')
        return
    print(f'  {label}: {stats["count"]} trades | '
          f'WR={stats["win_rate"]}% | PF={stats["pf"]} | '
          f'Avg PnL={stats["avg_pnl_pct"]}% | Total={stats["total_pnl"]:,.0f} Rs')


def run_config(label, config, df):
    t0 = time.time()
    sim = SqueezeFireSimulator(config, df)
    summary = sim.run()
    elapsed = time.time() - t0
    trades = sim.get_trade_log()

    print(f'\n{"="*70}')
    print(f' {label}  ({elapsed:.1f}s)')
    print(f'{"="*70}')

    print(f'\n  COMBINED: {summary["total_trades"]} trades | '
          f'WR={summary["win_rate"]}% | PF={summary["profit_factor"]} | '
          f'CAGR={summary["cagr"]}% | MaxDD={summary["max_drawdown"]}% | '
          f'Calmar={summary["calmar"]} | Sharpe={summary["sharpe"]}')
    print(f'  Capital: 10L -> {summary["final_equity"]:,.0f} Rs ({summary["total_return_pct"]}%)')

    print(f'\n  --- By Mode ---')
    print_mode_stats('SQUEEZE (strangles)', summary['squeeze_stats'])
    print_mode_stats('FIRE (spreads)', summary['fire_stats'])
    if summary['fire_long']:
        print_mode_stats('  FIRE Long', summary['fire_long_stats'])
    if summary['fire_short']:
        print_mode_stats('  FIRE Short', summary['fire_short_stats'])

    print(f'\n  Exit reasons: {summary["exit_reasons"]}')
    print(f'  Avg hold: {summary["avg_days_held"]} days | Avg DTE: {summary["avg_dte"]}')

    # Print trade log
    if trades:
        print(f'\n  --- All Trades ---')
        print(f'  {"#":>3} {"Mode":<8} {"Dir":<7} {"Entry":<12} {"Exit":<12} '
              f'{"Spot":>8} {"Strikes":>15} {"Prem/Dbt":>8} {"PnL%":>7} {"PnL Rs":>9} {"Exit":>14}')
        for t in trades:
            strikes = f'{t["strike_1"]:.0f}/{t["strike_2"]:.0f}'
            print(f'  {t["trade_id"]:>3} {t["mode"]:<8} {t["direction"]:<7} '
                  f'{t["entry_date"]:<12} {t["exit_date"]:<12} '
                  f'{t["entry_spot"]:>8.0f} {strikes:>15} '
                  f'{t["premium_or_debit"]:>8.1f} {t["pnl_pct"]:>7.1f} '
                  f'{t["pnl_rupees"]:>9.0f} {t["exit_reason"]:>14}')

    return summary, trades


def main():
    df = load_data()

    configs = [
        # 1. Both modes on — baseline
        ('COMBINED baseline', SqueezeFireConfig(
            squeeze_enabled=True, fire_enabled=True,
            squeeze_strike_atr=1.5, squeeze_hold_bars=10,
            squeeze_trend_filter=True, squeeze_trend_atr_max=2.0,
            fire_spread_width_atr=1.5, fire_hold_bars=7,
            fire_squeeze_min_bars=3, fire_trend_atr_min=1.0,
            fire_profit_target_pct=0.50, fire_stop_loss_pct=0.40,
            squeeze_lots=5, fire_lots=5,
        )),

        # 2. Squeeze only (for comparison)
        ('SQUEEZE ONLY', SqueezeFireConfig(
            squeeze_enabled=True, fire_enabled=False,
            squeeze_strike_atr=1.5, squeeze_hold_bars=10,
            squeeze_trend_filter=True, squeeze_trend_atr_max=2.0,
            squeeze_lots=5,
        )),

        # 3. Fire only (for comparison)
        ('FIRE ONLY', SqueezeFireConfig(
            squeeze_enabled=False, fire_enabled=True,
            fire_spread_width_atr=1.5, fire_hold_bars=7,
            fire_squeeze_min_bars=3, fire_trend_atr_min=1.0,
            fire_profit_target_pct=0.50, fire_stop_loss_pct=0.40,
            fire_lots=5,
        )),

        # 4. Combined with wider fire spread
        ('COMBINED wide_fire', SqueezeFireConfig(
            squeeze_enabled=True, fire_enabled=True,
            squeeze_strike_atr=1.5, squeeze_hold_bars=10,
            squeeze_trend_filter=True, squeeze_trend_atr_max=2.0,
            fire_spread_width_atr=2.0, fire_hold_bars=10,
            fire_squeeze_min_bars=3, fire_trend_atr_min=1.0,
            fire_profit_target_pct=0.80, fire_stop_loss_pct=0.50,
            squeeze_lots=5, fire_lots=5,
        )),

        # 5. Combined with no trend confirm on fire
        ('COMBINED no_fire_trend', SqueezeFireConfig(
            squeeze_enabled=True, fire_enabled=True,
            squeeze_strike_atr=1.5, squeeze_hold_bars=10,
            squeeze_trend_filter=True, squeeze_trend_atr_max=2.0,
            fire_spread_width_atr=1.5, fire_hold_bars=7,
            fire_squeeze_min_bars=3, fire_trend_atr_min=0.0,
            fire_profit_target_pct=0.50, fire_stop_loss_pct=0.40,
            squeeze_lots=5, fire_lots=5,
        )),

        # 6. Combined with DI + RSI filters on fire
        ('COMBINED fire_DI_RSI', SqueezeFireConfig(
            squeeze_enabled=True, fire_enabled=True,
            squeeze_strike_atr=1.5, squeeze_hold_bars=10,
            squeeze_trend_filter=True, squeeze_trend_atr_max=2.0,
            fire_spread_width_atr=1.5, fire_hold_bars=7,
            fire_squeeze_min_bars=3, fire_trend_atr_min=1.0,
            fire_use_di_filter=True, fire_use_rsi_filter=True,
            fire_profit_target_pct=0.50, fire_stop_loss_pct=0.40,
            squeeze_lots=5, fire_lots=5,
        )),

        # 7. Longer squeeze min bars for fire
        ('COMBINED fire_SQ5', SqueezeFireConfig(
            squeeze_enabled=True, fire_enabled=True,
            squeeze_strike_atr=1.5, squeeze_hold_bars=10,
            squeeze_trend_filter=True, squeeze_trend_atr_max=2.0,
            fire_spread_width_atr=1.5, fire_hold_bars=7,
            fire_squeeze_min_bars=5, fire_trend_atr_min=1.0,
            fire_profit_target_pct=0.50, fire_stop_loss_pct=0.40,
            squeeze_lots=5, fire_lots=5,
        )),
    ]

    print(f'\nRunning {len(configs)} configs...')
    for label, config in configs:
        run_config(label, config, df)

    sys.stdout.flush()


if __name__ == '__main__':
    main()
