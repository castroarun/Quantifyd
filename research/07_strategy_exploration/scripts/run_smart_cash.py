"""
Smart Cash Deployment: Test idle cash strategies for MQ portfolios.

Scenarios (for each PS30 and PS10):
  1. DAILY_DEBT     - Daily ATH exits + idle cash swept to debt fund (6.5% p.a.)
  2. DAILY_NIFTY    - Daily ATH exits + idle cash to NIFTYBEES (when Nifty < 200 SMA) else debt
  3. DAILY_NIFTY_REPLACE - Daily ATH exits + immediate replacement + leftover idle → NIFTYBEES/debt
  4. REBAL_DEBT     - Rebalance-only exits + idle cash to debt fund
  5. REBAL_NIFTY    - Rebalance-only exits + idle cash to NIFTYBEES/debt

Run one at a time. Results appended to CSV incrementally.
"""
import csv, os, sys, time, logging
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smart_cash_results.csv')
FIELDNAMES = [
    'scenario', 'portfolio_size', 'cagr', 'sharpe', 'sortino', 'max_drawdown',
    'calmar', 'total_trades', 'win_rate', 'avg_win_pct', 'avg_loss_pct',
    'final_value', 'total_return_pct', 'total_topups',
]

START_DATE = '2005-01-01'
END_DATE = '2025-11-07'

# Skip already done
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['scenario'] for row in csv.DictReader(f)}
    print(f'Skipping {len(done)} already-completed scenarios')
else:
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# Define scenarios
scenarios = []

for ps, ps_label in [(30, 'PS30'), (10, 'PS10')]:
    base_params = dict(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=10_000_000,
        portfolio_size=ps,
        equity_allocation_pct=0.95,
        hard_stop_loss=0.50,
        rebalance_ath_drawdown=0.20,
    )

    # Extra params for PS10 (concentrated)
    if ps == 10:
        base_params.update(dict(
            max_sector_weight=0.70,
            max_position_size=0.30,
            topup_pct_of_initial=0.30,
            rebalance_months=[1, 3, 5, 7, 9, 11],
        ))

    # Scenario 1: Daily exit + idle cash to debt fund
    scenarios.append({
        'name': f'DAILY_DEBT_{ps_label}',
        'config': MQBacktestConfig(
            **base_params,
            daily_ath_drawdown_exit=True,
            immediate_replacement=False,
            idle_cash_to_nifty_etf=False,
            idle_cash_to_debt=True,
        ),
    })

    # Scenario 2: Daily exit + smart NIFTYBEES/debt
    scenarios.append({
        'name': f'DAILY_NIFTY_{ps_label}',
        'config': MQBacktestConfig(
            **base_params,
            daily_ath_drawdown_exit=True,
            immediate_replacement=False,
            idle_cash_to_nifty_etf=True,
            idle_cash_to_debt=False,
        ),
    })

    # Scenario 3: Daily exit + immediate replacement + leftover to NIFTYBEES/debt
    scenarios.append({
        'name': f'DAILY_NIFTY_REPLACE_{ps_label}',
        'config': MQBacktestConfig(
            **base_params,
            daily_ath_drawdown_exit=True,
            immediate_replacement=True,
            idle_cash_to_nifty_etf=True,
            idle_cash_to_debt=False,
        ),
    })

    # Scenario 4: Rebalance-only exits + idle cash to debt
    scenarios.append({
        'name': f'REBAL_DEBT_{ps_label}',
        'config': MQBacktestConfig(
            **base_params,
            daily_ath_drawdown_exit=False,
            immediate_replacement=False,
            idle_cash_to_nifty_etf=False,
            idle_cash_to_debt=True,
        ),
    })

    # Scenario 5: Rebalance-only exits + smart NIFTYBEES/debt
    scenarios.append({
        'name': f'REBAL_NIFTY_{ps_label}',
        'config': MQBacktestConfig(
            **base_params,
            daily_ath_drawdown_exit=False,
            immediate_replacement=False,
            idle_cash_to_nifty_etf=True,
            idle_cash_to_debt=False,
        ),
    })

# Preload data once
print("Preloading data...")
t0 = time.time()
universe, price_data = MQBacktestEngine.preload_data(MQBacktestConfig(
    start_date=START_DATE, end_date=END_DATE
))
print(f"Data loaded in {time.time()-t0:.0f}s ({len(universe.stocks)} stocks)")
sys.stdout.flush()

# Run scenarios
todo = [s for s in scenarios if s['name'] not in done]
total = len(todo)
print(f"\nRunning {total} scenarios ({len(done)} already done)...\n")

for i, scenario in enumerate(todo):
    t0 = time.time()
    name = scenario['name']
    print(f'[{i+1}/{total}] {name} ...', end='', flush=True)

    engine = MQBacktestEngine(
        scenario['config'],
        preloaded_universe=universe,
        preloaded_price_data=price_data,
    )
    result = engine.run()

    row = {
        'scenario': name,
        'portfolio_size': scenario['config'].portfolio_size,
        'cagr': round(result.cagr, 2),
        'sharpe': round(result.sharpe_ratio, 2),
        'sortino': round(result.sortino_ratio, 2),
        'max_drawdown': round(result.max_drawdown, 1),
        'calmar': round(result.calmar_ratio, 2),
        'total_trades': result.total_trades,
        'win_rate': round(result.win_rate, 1),
        'avg_win_pct': round(result.avg_win_pct, 1),
        'avg_loss_pct': round(result.avg_loss_pct, 1),
        'final_value': round(result.final_value, 0),
        'total_return_pct': round(result.total_return_pct, 1),
        'total_topups': result.total_topups,
    }

    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    elapsed = time.time() - t0
    print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% MaxDD={row["max_drawdown"]:.1f}% Trades={row["total_trades"]} Final=Rs {row["final_value"]:,.0f}')
    sys.stdout.flush()

print(f"\nDone! Results in {OUTPUT_CSV}")
