"""
MQ Concentration Sweep - Full Period (2010-2026)
================================================

Tests PS5, PS10, PS15, PS20, PS25, PS30 over the full 16-year period
to get true long-term performance for each concentration level.

Output: mq_concentration_sweep_results.csv
"""

import csv
import logging
import os
import sys
import time

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

import numpy as np
import pandas as pd

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'mq_concentration_sweep_results.csv')

FIELDNAMES = [
    'label', 'portfolio_size', 'period',
    'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
    'profit_factor', 'total_trades', 'win_rate',
    'avg_win_pct', 'avg_loss_pct',
    'final_value', 'total_return_pct',
    'exit_reason_counts', 'topups',
]

INITIAL_CAPITAL = 10_000_000


def compute_metrics(result, initial_capital):
    """Extract comprehensive metrics from MQ BacktestResult."""
    eq = pd.Series(result.daily_equity, dtype=float)
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    final = float(eq.iloc[-1])
    years = (eq.index[-1] - eq.index[0]).days / 365.25

    cagr = ((final / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    total_ret = (final / initial_capital - 1) * 100

    daily_ret = eq.pct_change().dropna()
    rf_daily = 0.07 / 252
    excess = daily_ret - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    downside = daily_ret[daily_ret < 0]
    sortino = float(excess.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0

    peak = eq.expanding().max()
    dd = ((eq - peak) / peak * 100)
    max_dd = abs(float(dd.min()))
    calmar = cagr / max_dd if max_dd > 0 else 0

    trades = result.trade_log
    winners = [t for t in trades if t.return_pct > 0]
    losers = [t for t in trades if t.return_pct <= 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.return_pct * 100 for t in winners]) if winners else 0
    avg_loss = np.mean([t.return_pct * 100 for t in losers]) if losers else 0

    total_wins = sum(t.net_pnl for t in winners)
    total_losses = abs(sum(t.net_pnl for t in losers))
    pf = total_wins / total_losses if total_losses > 0 else float('inf')

    return {
        'cagr': round(cagr, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_drawdown': round(max_dd, 1),
        'calmar': round(calmar, 2),
        'profit_factor': round(pf, 2),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'final_value': round(final, 0),
        'total_return_pct': round(total_ret, 1),
        'exit_reason_counts': str(result.exit_reason_counts),
        'topups': result.total_topups,
    }


if __name__ == '__main__':
    # Skip already done
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        print(f'Skipping {len(done)} already-completed configs')
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Define configs
    configs = []

    # Full period: 2010-2026
    for ps in [5, 10, 15, 20, 25, 30]:
        label = f'PS{ps}_2010_2026'
        configs.append({
            'label': label,
            'portfolio_size': ps,
            'start_date': '2010-01-01',
            'end_date': '2026-02-16',
            'period': '2010-2026',
        })

    # Also test 2015-2025 for comparison with technical strategies
    for ps in [5, 10, 15, 20, 25, 30]:
        label = f'PS{ps}_2015_2025'
        configs.append({
            'label': label,
            'portfolio_size': ps,
            'start_date': '2015-01-01',
            'end_date': '2025-12-31',
            'period': '2015-2025',
        })

    todo = [c for c in configs if c['label'] not in done]
    total = len(todo)
    if total == 0:
        print('All configs already done!')
        sys.exit(0)

    print(f'\nRunning {total} configs ({len(done)} already done)...\n')

    # Group by period to preload data once per period
    periods = {}
    for c in todo:
        key = (c['start_date'], c['end_date'])
        if key not in periods:
            periods[key] = []
        periods[key].append(c)

    i = 0
    for (start, end), period_configs in periods.items():
        print(f'Preloading data for {start} to {end}...')
        t0 = time.time()
        base_config = MQBacktestConfig(start_date=start, end_date=end,
                                        initial_capital=INITIAL_CAPITAL)
        universe, price_data = MQBacktestEngine.preload_data(base_config)
        print(f'Data loaded in {time.time()-t0:.0f}s ({len(price_data)} stocks)\n')

        for cfg in period_configs:
            i += 1
            label = cfg['label']
            ps = cfg['portfolio_size']

            print(f'[{i}/{total}] {label} ...', end='', flush=True)
            t1 = time.time()

            mq_config = MQBacktestConfig(
                start_date=cfg['start_date'],
                end_date=cfg['end_date'],
                initial_capital=INITIAL_CAPITAL,
                portfolio_size=ps,
                equity_allocation_pct=0.95,
                debt_reserve_pct=0.05,
                hard_stop_loss=0.50,
                rebalance_ath_drawdown=0.20,
                daily_ath_drawdown_exit=True,
                immediate_replacement=True,
                idle_cash_to_nifty_etf=True,
                idle_cash_to_debt=True,
            )

            engine = MQBacktestEngine(mq_config,
                                       preloaded_universe=universe,
                                       preloaded_price_data=price_data)
            result = engine.run()
            elapsed = time.time() - t1

            metrics = compute_metrics(result, INITIAL_CAPITAL)
            row = {
                'label': label,
                'portfolio_size': ps,
                'period': cfg['period'],
                **metrics,
            }

            with open(OUTPUT_CSV, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            print(f' {elapsed:.0f}s | CAGR={metrics["cagr"]:.2f}% '
                  f'MaxDD={metrics["max_drawdown"]:.1f}% '
                  f'Calmar={metrics["calmar"]:.2f} '
                  f'PF={metrics["profit_factor"]:.2f} '
                  f'Trades={metrics["total_trades"]}')
            sys.stdout.flush()

    print(f'\nDone! Results in {OUTPUT_CSV}')
