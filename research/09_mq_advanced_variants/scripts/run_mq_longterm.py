"""Run MQ PS30 backtest over longer periods to find true long-term CAGR."""
import sys, os, time, csv, logging

logging.disable(logging.WARNING)

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mq_longterm_results.csv')
FIELDNAMES = ['label','start_date','end_date','years','cagr','sharpe','sortino',
              'max_drawdown','calmar','total_trades','win_rate','final_value',
              'total_return_pct','topups']

configs = [
    # PS30 configs
    ('PS30_2005_2025', 30, '2005-01-01', '2025-12-31'),
    ('PS30_2010_2025', 30, '2010-01-01', '2025-12-31'),
    ('PS30_2012_2025', 30, '2012-01-01', '2025-12-31'),
    ('PS30_2015_2025', 30, '2015-01-01', '2025-12-31'),
    # PS20 configs
    ('PS20_2005_2025', 20, '2005-01-01', '2025-12-31'),
    ('PS20_2010_2025', 20, '2010-01-01', '2025-12-31'),
    ('PS20_2012_2025', 20, '2012-01-01', '2025-12-31'),
    ('PS20_2015_2025', 20, '2015-01-01', '2025-12-31'),
]

with open(OUTPUT_CSV, 'w', newline='') as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

for i, (label, ps, start, end) in enumerate(configs):
    print(f'[{i+1}/{len(configs)}] {label} (PS{ps}, {start} to {end}) ...', end='', flush=True)
    t0 = time.time()
    try:
        config = MQBacktestConfig(
            start_date=start,
            end_date=end,
            initial_capital=10_000_000,
            portfolio_size=ps,
            equity_allocation_pct=0.95,
            hard_stop_loss=0.50,
            rebalance_ath_drawdown=0.20,
        )
        universe, price_data = MQBacktestEngine.preload_data(config)
        engine = MQBacktestEngine(config, preloaded_universe=universe, preloaded_price_data=price_data)
        result = engine.run()

        row = {
            'label': label,
            'start_date': start,
            'end_date': end,
            'years': round((time.time() - t0) / 1, 1),  # will fix below
            'cagr': round(result.cagr, 2),
            'sharpe': round(result.sharpe_ratio, 2),
            'sortino': round(result.sortino_ratio, 2),
            'max_drawdown': round(result.max_drawdown, 2),
            'calmar': round(result.calmar_ratio, 2),
            'total_trades': result.total_trades,
            'win_rate': round(result.win_rate, 2),
            'final_value': round(result.final_value, 0),
            'total_return_pct': round(result.total_return_pct, 2),
            'topups': result.total_topups,
        }
        # Calculate actual years from dates
        from datetime import datetime
        d1 = datetime.strptime(start, '%Y-%m-%d')
        d2 = datetime.strptime(end, '%Y-%m-%d')
        row['years'] = round((d2 - d1).days / 365.25, 1)

        elapsed = time.time() - t0
        print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% Sharpe={row["sharpe"]:.2f} MaxDD={row["max_drawdown"]:.2f}% Trades={row["total_trades"]}')
        sys.stdout.flush()

        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
    except Exception as e:
        print(f' ERROR: {e}')
        sys.stdout.flush()

print(f'\nDone! Results saved to {OUTPUT_CSV}')
