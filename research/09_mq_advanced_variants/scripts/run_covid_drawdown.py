"""Extract MQ equity curve during COVID period for visualization."""
import sys, os, logging, json
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

# Run PS20 backtest for 2018-2022 (wider window around COVID)
config = MQBacktestConfig(
    start_date='2018-01-01',
    end_date='2022-01-01',
    initial_capital=10_000_000,
    portfolio_size=20,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,
    rebalance_ath_drawdown=0.20,
    idle_cash_to_nifty_etf=True,
    idle_cash_to_debt=True,
    nifty_etf_above_sma=True,
    nifty_sma_confirm_days=2,
)

print("Loading data...", flush=True)
universe, price_data = MQBacktestEngine.preload_data(config)
print(f"Data loaded. Running backtest...", flush=True)

engine = MQBacktestEngine(config, preloaded_universe=universe, preloaded_price_data=price_data)
result = engine.run()

print(f"CAGR: {result.cagr:.2f}%, MaxDD: {result.max_drawdown:.2f}%, Trades: {result.total_trades}", flush=True)

# Extract equity curve for COVID zoom (Jan 2020 - Dec 2020)
eq = result.daily_equity  # dict of date_str -> value
dates_sorted = sorted(eq.keys())

# Filter to COVID period
covid_dates = [d for d in dates_sorted if '2020-01' <= d <= '2020-12-31']
# Also get pre-COVID peak (Dec 2019 - Jan 2020)
pre_covid = [d for d in dates_sorted if '2019-12' <= d <= '2020-01-01']

all_dates = pre_covid + covid_dates
# Sample weekly (every 5 trading days)
sampled = []
for i in range(0, len(all_dates), 5):
    d = all_dates[i]
    sampled.append({'date': d, 'value': eq[d]})
# Always include last date
if all_dates[-1] != sampled[-1]['date']:
    d = all_dates[-1]
    sampled.append({'date': d, 'value': eq[d]})

# Find pre-COVID peak
peak = max(eq[d] for d in all_dates if d <= '2020-02-20')
print(f"\nPre-COVID peak: Rs.{peak:,.0f}")

# Calculate drawdown from peak for each sampled point
print("\n--- MQ Equity Curve (COVID period, weekly) ---")
print("date,value,drawdown_pct")
for p in sampled:
    dd = (p['value'] - peak) / peak * 100
    print(f"{p['date']},{p['value']:.0f},{dd:.2f}")

# Find worst drawdown in COVID period
worst_dd = 0
worst_date = ''
for d in all_dates:
    dd = (eq[d] - peak) / peak * 100
    if dd < worst_dd:
        worst_dd = dd
        worst_date = d
print(f"\nMQ Worst COVID drawdown: {worst_dd:.2f}% on {worst_date}")
print(f"MQ Peak: Rs.{peak:,.0f}, Trough: Rs.{eq[worst_date]:,.0f}")

# Full period equity curve (monthly) for broader context
print("\n--- Full Period Monthly Equity Curve ---")
print("date,value")
monthly_dates = []
current_month = ''
for d in dates_sorted:
    m = d[:7]
    if m != current_month:
        current_month = m
        monthly_dates.append(d)
for d in monthly_dates:
    print(f"{d},{eq[d]:.0f}")
