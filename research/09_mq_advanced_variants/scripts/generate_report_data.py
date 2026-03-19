"""
Generate all data for the MQ Investment Report HTML.
Runs:
  1. NIFTYBEES benchmark (2015-2025)
  2. Basic MQ PS30 backtest (2015-2025) - baseline with cash sitting idle
  3. MQ + Capital Recycling PS30 (2015-2025) - full system with daily exits + replacement + NIFTYBEES/debt
Output: report_data.json
"""
import json, os, sys, time, sqlite3
import pandas as pd
import numpy as np
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_data.json')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')

START_DATE = '2010-01-01'
END_DATE = '2026-02-16'

# ═══════════════════════════════════════════════════════════
# 1. NIFTYBEES BENCHMARK
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("LOADING NIFTYBEES BENCHMARK DATA")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)
nifty = pd.read_sql_query(
    "SELECT date, close FROM market_data_unified WHERE symbol='NIFTYBEES' ORDER BY date",
    conn
)
conn.close()

nifty['date'] = pd.to_datetime(nifty['date'])
nifty = nifty.set_index('date')
nifty = nifty[(nifty.index >= START_DATE) & (nifty.index <= END_DATE)]

nifty_start = nifty['close'].iloc[0]
nifty_end = nifty['close'].iloc[-1]
years = (nifty.index[-1] - nifty.index[0]).days / 365.25
nifty_cagr = ((nifty_end / nifty_start) ** (1 / years) - 1) * 100
nifty_total_ret = ((nifty_end / nifty_start) - 1) * 100

# Normalize to Rs 1 Cr
nifty_eq = (nifty['close'] / nifty_start) * 10_000_000

# Yearly returns
nifty_yearly = nifty['close'].resample('A').last()
nifty_yr_ret = nifty_yearly.pct_change().dropna() * 100

# Max drawdown
nifty_peak = nifty['close'].expanding().max()
nifty_dd = ((nifty['close'] - nifty_peak) / nifty_peak * 100).min()

# Monthly equity for charts
nifty_monthly = nifty_eq.resample('M').last()

print(f"Period: {nifty.index[0].date()} to {nifty.index[-1].date()} ({years:.1f} yrs)")
print(f"CAGR: {nifty_cagr:.1f}%  |  Total: {nifty_total_ret:.0f}%  |  MaxDD: {nifty_dd:.1f}%")
print(f"Rs 1 Cr -> Rs {nifty_eq.iloc[-1]:,.0f}")

# Best/worst years
best_yr = nifty_yr_ret.idxmax()
worst_yr = nifty_yr_ret.idxmin()
print(f"Best year: {best_yr.year} ({nifty_yr_ret.max():.1f}%)")
print(f"Worst year: {worst_yr.year} ({nifty_yr_ret.min():.1f}%)")

# ═══════════════════════════════════════════════════════════
# 2. BASIC MQ PS30 BACKTEST (2015-2025) - Baseline
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RUNNING BASIC MQ PS30 BACKTEST (2015-2025)")
print("=" * 60)

config_basic = MQBacktestConfig(
    start_date=START_DATE,
    end_date=END_DATE,
    initial_capital=10_000_000,
    portfolio_size=30,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,
    rebalance_ath_drawdown=0.20,
)

t0 = time.time()
print("Preloading data...")
universe, price_data = MQBacktestEngine.preload_data(config_basic)
print(f"Data loaded in {time.time()-t0:.0f}s ({len(universe.stocks)} stocks)")
sys.stdout.flush()

t0 = time.time()
engine_basic = MQBacktestEngine(config_basic, preloaded_universe=universe, preloaded_price_data=price_data)
r_basic = engine_basic.run()
print(f"Basic MQ done in {time.time()-t0:.0f}s")
print(f"CAGR: {r_basic.cagr:.2f}%  |  MaxDD: {r_basic.max_drawdown:.1f}%  |  Sharpe: {r_basic.sharpe_ratio:.2f}")
print(f"Trades: {r_basic.total_trades}  |  WR: {r_basic.win_rate:.1f}%  |  Final: Rs {r_basic.final_value:,.0f}")
sys.stdout.flush()

# Save partial result in case full system times out
partial = {'basic_done': True, 'basic_cagr': r_basic.cagr}
with open(OUTPUT + '.partial', 'w') as f:
    json.dump(partial, f)

# ═══════════════════════════════════════════════════════════
# 3. MQ + CAPITAL RECYCLING PS30 (2015-2025) - Full System
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RUNNING MQ + CAPITAL RECYCLING PS30 (2015-2025)")
print("=" * 60)

config_full = MQBacktestConfig(
    start_date=START_DATE,
    end_date=END_DATE,
    initial_capital=10_000_000,
    portfolio_size=30,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,
    rebalance_ath_drawdown=0.20,
    daily_ath_drawdown_exit=True,
    immediate_replacement=True,
    idle_cash_to_nifty_etf=True,
    idle_cash_to_debt=True,
)

t0 = time.time()
engine_full = MQBacktestEngine(config_full, preloaded_universe=universe, preloaded_price_data=price_data)
r_full = engine_full.run()
print(f"Full System done in {time.time()-t0:.0f}s")
print(f"CAGR: {r_full.cagr:.2f}%  |  MaxDD: {r_full.max_drawdown:.1f}%  |  Sharpe: {r_full.sharpe_ratio:.2f}")
print(f"Trades: {r_full.total_trades}  |  WR: {r_full.win_rate:.1f}%  |  Final: Rs {r_full.final_value:,.0f}")
sys.stdout.flush()

# ═══════════════════════════════════════════════════════════
# 4. EXTRACT CURRENT HOLDINGS & EXIT REASONS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXTRACTING HOLDINGS & EXIT REASONS")
print("=" * 60)

def extract_holdings_and_exits(result):
    holdings = []
    exit_counts = {}
    for t in result.trade_log:
        reason = str(t.exit_reason)
        exit_counts[reason] = exit_counts.get(reason, 0) + 1
        if 'end_of_period' in reason.lower() or 'END_OF' in reason:
            holdings.append({
                'symbol': t.symbol,
                'sector': t.sector,
                'entry_price': round(t.entry_price, 1),
                'return_pct': round(t.return_pct, 1),
                'entry_date': str(t.entry_date.date()) if hasattr(t.entry_date, 'date') else str(t.entry_date),
            })
    return holdings, exit_counts

holdings_basic, exits_basic = extract_holdings_and_exits(r_basic)
holdings_full, exits_full = extract_holdings_and_exits(r_full)

print(f"Basic MQ: {len(holdings_basic)} holdings, exits: {exits_basic}")
print(f"Full System: {len(holdings_full)} holdings, exits: {exits_full}")

# ═══════════════════════════════════════════════════════════
# 5. LIVE PORTFOLIO SNAPSHOT (Jan 2025 - Latest)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RUNNING 1-YEAR LIVE PORTFOLIO SNAPSHOT (2025-01-01 to latest)")
print("=" * 60)

SNAPSHOT_START = '2025-01-01'

config_snapshot = MQBacktestConfig(
    start_date=SNAPSHOT_START,
    end_date=END_DATE,
    initial_capital=10_000_000,
    portfolio_size=30,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,
    rebalance_ath_drawdown=0.20,
)

t0 = time.time()
engine_snap = MQBacktestEngine(config_snapshot, preloaded_universe=universe, preloaded_price_data=price_data)
r_snap = engine_snap.run()
print(f"Snapshot done in {time.time()-t0:.0f}s")
print(f"CAGR: {r_snap.cagr:.2f}%  |  Final: Rs {r_snap.final_value:,.0f}  |  Trades: {r_snap.total_trades}")

# Extract live holdings from final_positions (still open at end date)
live_holdings = []
for p in r_snap.final_positions:
    live_holdings.append({
        'symbol': p['symbol'],
        'sector': p['sector'],
        'entry_date': p['entry_date'],
        'entry_price': round(p['entry_price'], 1),
        'current_price': round(p['current_price'], 1),
        'return_pct': round(p['return_pct'], 1),
        'pnl': round(p['pnl'], 0),
        'value': round(p['value'], 0),
        'drawdown_from_ath': round(p['drawdown_from_ath'], 1),
    })

# Extract closed trades (already exited)
closed_trades = []
for t in r_snap.trade_log:
    reason = str(t.exit_reason)
    closed_trades.append({
        'symbol': t.symbol,
        'sector': t.sector,
        'entry_date': str(t.entry_date.date()) if hasattr(t.entry_date, 'date') else str(t.entry_date),
        'entry_price': round(t.entry_price, 1),
        'exit_price': round(t.exit_price, 1),
        'return_pct': round(t.return_pct, 1),
        'net_pnl': round(t.net_pnl, 0),
        'holding_days': t.holding_days,
        'exit_reason': reason.replace('ExitReason.', ''),
    })

# Sort
live_holdings.sort(key=lambda x: x['return_pct'], reverse=True)
closed_trades.sort(key=lambda x: x['return_pct'], reverse=True)

# Portfolio-level stats
total_invested = 10_000_000
portfolio_return_pct = round((r_snap.final_value - total_invested) / total_invested * 100, 1)
total_unrealized_pnl = sum(h['pnl'] for h in live_holdings)
winners = [h for h in live_holdings if h['return_pct'] > 0]
losers = [h for h in live_holdings if h['return_pct'] < 0]

print(f"Live holdings: {len(live_holdings)} stocks | Closed: {len(closed_trades)} trades")
print(f"Portfolio return: {portfolio_return_pct}% | Unrealized P/L: Rs {total_unrealized_pnl:,.0f}")
print(f"Winners: {len(winners)} | Losers: {len(losers)}")
if live_holdings:
    print(f"Best: {live_holdings[0]['symbol']} ({live_holdings[0]['return_pct']:+.1f}%)")
    print(f"Worst: {live_holdings[-1]['symbol']} ({live_holdings[-1]['return_pct']:+.1f}%)")
sys.stdout.flush()

# ═══════════════════════════════════════════════════════════
# 6. COMPILE REPORT DATA
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPILING REPORT DATA")
print("=" * 60)

# Process equity curves (daily_equity is Dict[str, float])
eq_basic = pd.Series(r_basic.daily_equity)
eq_basic.index = pd.to_datetime(eq_basic.index)
eq_basic_monthly = eq_basic.resample('M').last()
eq_basic_yearly = eq_basic.resample('A').last()
eq_basic_yr_ret = eq_basic_yearly.pct_change().dropna() * 100

eq_full = pd.Series(r_full.daily_equity)
eq_full.index = pd.to_datetime(eq_full.index)
eq_full_monthly = eq_full.resample('M').last()
eq_full_yearly = eq_full.resample('A').last()
eq_full_yr_ret = eq_full_yearly.pct_change().dropna() * 100

# Compute drawdown series for chart
def get_drawdown_series(equity_monthly):
    peak = equity_monthly.expanding().max()
    return ((equity_monthly - peak) / peak * 100)

nifty_dd_series = get_drawdown_series(nifty_monthly)
basic_dd_series = get_drawdown_series(eq_basic_monthly)
full_dd_series = get_drawdown_series(eq_full_monthly)

report = {
    'generated_at': time.strftime('%Y-%m-%d %H:%M'),
    'period': {'start': START_DATE, 'end': END_DATE, 'years': round(years, 1)},
    'benchmark': {
        'name': 'Nifty 50 (via NIFTYBEES ETF)',
        'cagr': round(nifty_cagr, 1),
        'total_return_pct': round(nifty_total_ret, 0),
        'max_drawdown': round(float(nifty_dd), 1),
        'final_value': round(float(nifty_eq.iloc[-1]), 0),
        'best_year': {'year': int(best_yr.year), 'return': round(float(nifty_yr_ret.max()), 1)},
        'worst_year': {'year': int(worst_yr.year), 'return': round(float(nifty_yr_ret.min()), 1)},
        'yearly_returns': {str(k.year): round(float(v), 1) for k, v in nifty_yr_ret.items()},
        'equity_monthly': {str(k.date()): round(float(v), 0) for k, v in nifty_monthly.items()},
        'drawdown_monthly': {str(k.date()): round(float(v), 1) for k, v in nifty_dd_series.items()},
    },
    'mq_basic': {
        'name': 'Basic MQ 30-Stock',
        'cagr': round(r_basic.cagr, 2),
        'sharpe': round(r_basic.sharpe_ratio, 2),
        'sortino': round(r_basic.sortino_ratio, 2),
        'max_drawdown': round(r_basic.max_drawdown, 1),
        'calmar': round(r_basic.calmar_ratio, 2),
        'total_trades': r_basic.total_trades,
        'win_rate': round(r_basic.win_rate, 1),
        'avg_win_pct': round(r_basic.avg_win_pct, 1),
        'avg_loss_pct': round(r_basic.avg_loss_pct, 1),
        'final_value': round(r_basic.final_value, 0),
        'total_return_pct': round(r_basic.total_return_pct, 1),
        'yearly_returns': {str(k.year): round(float(v), 1) for k, v in eq_basic_yr_ret.items()},
        'equity_monthly': {str(k.date()): round(float(v), 0) for k, v in eq_basic_monthly.items()},
        'drawdown_monthly': {str(k.date()): round(float(v), 1) for k, v in basic_dd_series.items()},
        'exit_reasons': exits_basic,
        'holdings': sorted(holdings_basic, key=lambda x: x['return_pct'], reverse=True),
    },
    'mq_full': {
        'name': 'MQ + Capital Recycling',
        'cagr': round(r_full.cagr, 2),
        'sharpe': round(r_full.sharpe_ratio, 2),
        'sortino': round(r_full.sortino_ratio, 2),
        'max_drawdown': round(r_full.max_drawdown, 1),
        'calmar': round(r_full.calmar_ratio, 2),
        'total_trades': r_full.total_trades,
        'win_rate': round(r_full.win_rate, 1),
        'avg_win_pct': round(r_full.avg_win_pct, 1),
        'avg_loss_pct': round(r_full.avg_loss_pct, 1),
        'final_value': round(r_full.final_value, 0),
        'total_return_pct': round(r_full.total_return_pct, 1),
        'yearly_returns': {str(k.year): round(float(v), 1) for k, v in eq_full_yr_ret.items()},
        'equity_monthly': {str(k.date()): round(float(v), 0) for k, v in eq_full_monthly.items()},
        'drawdown_monthly': {str(k.date()): round(float(v), 1) for k, v in full_dd_series.items()},
        'exit_reasons': exits_full,
        'holdings': sorted(holdings_full, key=lambda x: x['return_pct'], reverse=True),
    },
    'live_portfolio': {
        'period_start': SNAPSHOT_START,
        'period_end': END_DATE,
        'initial_capital': total_invested,
        'final_value': round(r_snap.final_value, 0),
        'portfolio_return_pct': portfolio_return_pct,
        'unrealized_pnl': round(total_unrealized_pnl, 0),
        'num_holdings': len(live_holdings),
        'num_winners': len(winners),
        'num_losers': len(losers),
        'num_closed': len(closed_trades),
        'live_holdings': live_holdings,
        'closed_trades': closed_trades,
    },
}

with open(OUTPUT, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nReport data saved to {OUTPUT}")
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")
print(f"{'System':>30} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Rs 1 Cr ->':>18}")
print(f"{'-'*75}")
print(f"{'Nifty 50':>30} {nifty_cagr:>7.1f}% {nifty_dd:>7.1f}% {'--':>8} Rs {nifty_eq.iloc[-1]:>14,.0f}")
print(f"{'Basic MQ':>30} {r_basic.cagr:>7.2f}% {r_basic.max_drawdown:>7.1f}% {r_basic.sharpe_ratio:>7.2f} Rs {r_basic.final_value:>14,.0f}")
print(f"{'MQ + Capital Recycling':>30} {r_full.cagr:>7.2f}% {r_full.max_drawdown:>7.1f}% {r_full.sharpe_ratio:>7.2f} Rs {r_full.final_value:>14,.0f}")

# Clean up partial file
if os.path.exists(OUTPUT + '.partial'):
    os.remove(OUTPUT + '.partial')

print("\nDone!")
