"""
Combined MQ + Tactical Pool Backtest — 15 Years (2010-2025)
============================================================
Runs MQ Core (60%) and KC6 (40%) simultaneously over the full data range.
Also tracks NIFTYBEES as benchmark for fair comparison.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine, _load_niftybees_data
from services.mq_portfolio import MQBacktestConfig
from services.kc6_backtest_engine import KC6BacktestEngine, KC6BacktestConfig

print("=" * 70)
print("COMBINED MQ + TACTICAL POOL BACKTEST -- 15 YEARS")
print("=" * 70)

# =============================================================================
# Configuration
# =============================================================================
TOTAL_CAPITAL = 10_000_000  # Rs. 1 Crore
MQ_PCT = 0.60
TACTICAL_PCT = 0.40
MQ_CAPITAL = TOTAL_CAPITAL * MQ_PCT        # 6,000,000
KC6_CAPITAL = TOTAL_CAPITAL * TACTICAL_PCT  # 4,000,000

START_DATE = '2010-01-01'
END_DATE = '2025-12-31'

# KC6 needs 1-year SMA200 warmup
KC6_DATA_START = '2009-01-01'

print(f"\nTotal Capital:  Rs.{TOTAL_CAPITAL:,.0f} (Rs.{TOTAL_CAPITAL/1e5:.0f}L)")
print(f"MQ Core (60%):  Rs.{MQ_CAPITAL:,.0f} (Rs.{MQ_CAPITAL/1e5:.0f}L)")
print(f"Tactical (40%): Rs.{KC6_CAPITAL:,.0f} (Rs.{KC6_CAPITAL/1e5:.0f}L)")
print(f"Period: {START_DATE} to {END_DATE} (15 years)")

# =============================================================================
# Phase 1: Load shared data
# =============================================================================
print("\n[Phase 1] Loading shared market data...")

loader_config = MQBacktestConfig(
    start_date=KC6_DATA_START,
    end_date=END_DATE,
)
universe, price_data = MQBacktestEngine.preload_data(loader_config)
print(f"  Loaded {len(price_data)} symbols")

# =============================================================================
# Phase 2: Run MQ Core backtest (60% allocation)
# =============================================================================
print("\n[Phase 2] Running MQ Core backtest (60% = Rs.60L, 15 years)...")
print("  This may take several minutes...")

mq_config = MQBacktestConfig(
    start_date=START_DATE,
    end_date=END_DATE,
    initial_capital=int(MQ_CAPITAL),
    portfolio_size=30,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,
    rebalance_ath_drawdown=0.20,
    daily_ath_drawdown_exit=True,
    immediate_replacement=True,
)

mq_engine = MQBacktestEngine(
    mq_config,
    preloaded_universe=universe,
    preloaded_price_data=price_data,
)
mq_result = mq_engine.run()

print(f"  MQ CAGR:        {mq_result.cagr}%")
print(f"  MQ MaxDD:       {mq_result.max_drawdown}%")
print(f"  MQ Sharpe:      {mq_result.sharpe_ratio}")
print(f"  MQ Final Value: Rs.{mq_result.final_value:,.0f} (Rs.{mq_result.final_value/1e5:.1f}L)")
print(f"  MQ Trades:      {mq_result.total_trades}")
sys.stdout.flush()

# =============================================================================
# Phase 3: Run KC6 backtest (40% tactical allocation)
# =============================================================================
print("\n[Phase 3] Running KC6 backtest (40% = Rs.40L, 15 years)...")
print("  This may take several minutes...")

kc6_config = KC6BacktestConfig(
    start_date=KC6_DATA_START,
    end_date=END_DATE,
    initial_capital=int(KC6_CAPITAL),
    max_positions=7,
    position_size_pct=0.10,
    sl_pct=5.0,
    tp_pct=15.0,
    max_hold_days=15,
    debt_rate=7.0,
)

kc6_engine = KC6BacktestEngine(
    kc6_config,
    price_data,
    list(price_data.keys()),
)
kc6_result = kc6_engine.run()

print(f"  KC6 CAGR:        {kc6_result.cagr:.2f}%")
print(f"  KC6 MaxDD:       {kc6_result.max_drawdown:.1f}%")
print(f"  KC6 Sharpe:      {kc6_result.sharpe}")
print(f"  KC6 Final Value: Rs.{kc6_result.final_value:,.0f} (Rs.{kc6_result.final_value/1e5:.1f}L)")
print(f"  KC6 Trades:      {kc6_result.total_trades}")
print(f"  KC6 Win Rate:    {kc6_result.win_rate:.1f}%")
sys.stdout.flush()

# =============================================================================
# Phase 4: Load NIFTYBEES benchmark
# =============================================================================
print("\n[Phase 4] Loading NIFTYBEES benchmark...")

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'
nifty_df = _load_niftybees_data(START_DATE, END_DATE, DB_PATH)
if nifty_df is not None and len(nifty_df) > 0:
    # Scale to Rs.100L starting capital
    nifty_start_price = nifty_df['close'].iloc[0]
    nifty_df['equity'] = nifty_df['close'] / nifty_start_price * TOTAL_CAPITAL
    nifty_eq = nifty_df['equity']
    nifty_eq.index = pd.to_datetime(nifty_df.index if isinstance(nifty_df.index, pd.DatetimeIndex) else nifty_df['date'] if 'date' in nifty_df.columns else nifty_df.index)
    print(f"  NIFTYBEES: {len(nifty_df)} days, {nifty_df.index[0]} to {nifty_df.index[-1]}")

    # Nifty metrics
    nifty_final = nifty_eq.iloc[-1]
    nifty_years = (nifty_eq.index[-1] - nifty_eq.index[0]).days / 365.25
    nifty_cagr = ((nifty_final / TOTAL_CAPITAL) ** (1/nifty_years) - 1) * 100
    nifty_peak = nifty_eq.expanding().max()
    nifty_dd = (nifty_eq - nifty_peak) / nifty_peak * 100
    nifty_maxdd = abs(float(nifty_dd.min()))
    nifty_calmar = nifty_cagr / nifty_maxdd if nifty_maxdd > 0 else 0

    # Nifty Sharpe
    nifty_ret = nifty_eq.pct_change().dropna()
    rf_daily = 0.07 / 252
    nifty_excess = nifty_ret - rf_daily
    nifty_sharpe = float(nifty_excess.mean() / nifty_excess.std() * np.sqrt(252)) if nifty_excess.std() > 0 else 0

    print(f"  Nifty CAGR:  {nifty_cagr:.2f}%")
    print(f"  Nifty MaxDD: {nifty_maxdd:.2f}%")
    print(f"  Nifty Sharpe: {nifty_sharpe:.2f}")
    print(f"  Nifty Calmar: {nifty_calmar:.2f}")
else:
    print("  WARNING: No NIFTYBEES data found!")
    nifty_cagr = 0
    nifty_maxdd = 0
    nifty_calmar = 0
    nifty_sharpe = 0

sys.stdout.flush()

# =============================================================================
# Phase 5: Combine equity curves
# =============================================================================
print("\n[Phase 5] Combining equity curves...")

mq_eq = pd.Series(mq_result.daily_equity, dtype=float)
mq_eq.index = pd.to_datetime(mq_eq.index)
mq_eq = mq_eq.sort_index()

kc6_eq = pd.Series(kc6_result.daily_equity, dtype=float)
kc6_eq.index = pd.to_datetime(kc6_eq.index)
kc6_eq = kc6_eq.sort_index()

print(f"  MQ equity:  {len(mq_eq)} days ({mq_eq.index[0].date()} to {mq_eq.index[-1].date()})")
print(f"  KC6 equity: {len(kc6_eq)} days ({kc6_eq.index[0].date()} to {kc6_eq.index[-1].date()})")

combined_df = pd.DataFrame({
    'mq': mq_eq,
    'kc6': kc6_eq,
}).sort_index()

combined_df = combined_df.ffill().dropna()
combined_df['combined'] = combined_df['mq'] + combined_df['kc6']
combined_eq = combined_df['combined']

# Add Nifty to combined_df
if nifty_df is not None and len(nifty_df) > 0:
    nifty_series = pd.Series(nifty_eq.values, index=pd.to_datetime(nifty_eq.index))
    combined_df['nifty'] = nifty_series
    combined_df['nifty'] = combined_df['nifty'].ffill()

print(f"  Combined:   {len(combined_eq)} days ({combined_eq.index[0].date()} to {combined_eq.index[-1].date()})")
print(f"  Start:      Rs.{combined_eq.iloc[0]:,.0f} (Rs.{combined_eq.iloc[0]/1e5:.1f}L)")
print(f"  End:        Rs.{combined_eq.iloc[-1]:,.0f} (Rs.{combined_eq.iloc[-1]/1e5:.1f}L)")

# =============================================================================
# Phase 6: Calculate combined metrics
# =============================================================================
print("\n[Phase 6] Calculating combined metrics...")

initial = combined_eq.iloc[0]
final = combined_eq.iloc[-1]
years = (combined_eq.index[-1] - combined_eq.index[0]).days / 365.25
cagr = ((final / initial) ** (1 / years) - 1) * 100

daily_ret = combined_eq.pct_change().dropna()
rf_daily = 0.07 / 252
excess = daily_ret - rf_daily
sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

downside = daily_ret[daily_ret < 0]
sortino = float(excess.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0

peak = combined_eq.expanding().max()
dd = (combined_eq - peak) / peak * 100
max_dd = abs(float(dd.min()))

max_dd_date = dd.idxmin()
peak_before_dd = combined_eq[:max_dd_date].idxmax()
trough_value = combined_eq[max_dd_date]
peak_value = combined_eq[peak_before_dd]

calmar = cagr / max_dd if max_dd > 0 else 0

# MQ-only and KC6-only drawdowns
mq_common = combined_df['mq']
kc6_common = combined_df['kc6']
mq_peak = mq_common.expanding().max()
mq_dd = (mq_common - mq_peak) / mq_peak * 100
mq_maxdd_common = abs(float(mq_dd.min()))

kc6_peak = kc6_common.expanding().max()
kc6_dd = (kc6_common - kc6_peak) / kc6_peak * 100
kc6_maxdd_common = abs(float(kc6_dd.min()))

# =============================================================================
# Report
# =============================================================================
print("\n" + "=" * 70)
print("COMBINED SYSTEM RESULTS -- 15 YEARS")
print("=" * 70)

print(f"\n{'Metric':<25} {'MQ Core (60%)':<18} {'KC6 (40%)':<18} {'Combined':<18}")
print("-" * 79)
print(f"{'Allocation':<25} {'Rs.60L':<18} {'Rs.40L':<18} {'Rs.100L':<18}")
print(f"{'Final Value':<25} {'Rs.'+f'{mq_result.final_value/1e5:.1f}L':<18} {'Rs.'+f'{kc6_result.final_value/1e5:.1f}L':<18} {'Rs.'+f'{final/1e5:.1f}L':<18}")
print(f"{'CAGR':<25} {mq_result.cagr:>6.2f}%{'':<10} {kc6_result.cagr:>6.2f}%{'':<10} {cagr:>6.2f}%")
print(f"{'Max Drawdown':<25} {mq_maxdd_common:>6.2f}%{'':<10} {kc6_maxdd_common:>6.1f}%{'':<11} {max_dd:>6.2f}%")
print(f"{'Sharpe Ratio':<25} {mq_result.sharpe_ratio:>6.2f}{'':<11} {kc6_result.sharpe:>6.2f}{'':<11} {sharpe:>6.2f}")
print(f"{'Sortino Ratio':<25} {mq_result.sortino_ratio:>6.2f}{'':<11} {kc6_result.sortino:>6.2f}{'':<11} {sortino:>6.2f}")
print(f"{'Calmar Ratio':<25} {float(mq_result.cagr)/mq_maxdd_common if mq_maxdd_common else 0:>6.2f}{'':<11} {kc6_result.calmar:>6.2f}{'':<11} {calmar:>6.2f}")
print(f"{'Total Trades':<25} {mq_result.total_trades:>6d}{'':<11} {kc6_result.total_trades:>6d}{'':<11} {mq_result.total_trades + kc6_result.total_trades:>6d}")

print(f"\nDrawdown Detail:")
print(f"  Peak before DD:     {peak_before_dd.strftime('%Y-%m-%d')} (Rs.{peak_value/1e5:.1f}L)")
print(f"  Max DD date:        {max_dd_date.strftime('%Y-%m-%d')} (Rs.{trough_value/1e5:.1f}L)")
print(f"  Peak -> Trough:     Rs.{(peak_value - trough_value)/1e5:.1f}L lost")

print(f"\n{'='*70}")
print(f"COMPARISON vs NIFTY 50 (SAME PERIOD: {START_DATE} to {END_DATE})")
print(f"{'='*70}")
print(f"\n{'Metric':<25} {'Our System':<18} {'Nifty 50':<18} {'Advantage':<18}")
print(f"{'-'*70}")
print(f"{'CAGR':<25} {cagr:>6.2f}%{'':<10} {nifty_cagr:>6.2f}%{'':<10} {'+' if cagr > nifty_cagr else ''}{cagr - nifty_cagr:.1f}%")
print(f"{'Max Drawdown':<25} {max_dd:>6.2f}%{'':<10} {nifty_maxdd:>6.2f}%{'':<10} {nifty_maxdd - max_dd:.1f}% better")
print(f"{'Calmar Ratio':<25} {calmar:>6.2f}{'':<11} {nifty_calmar:>6.2f}{'':<11} {calmar/nifty_calmar:.1f}x better" if nifty_calmar > 0 else "")
print(f"{'Sharpe Ratio':<25} {sharpe:>6.2f}{'':<11} {nifty_sharpe:>6.2f}")

print(f"\n{'='*70}")

# Save
output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_equity_curve_15yr.csv')
combined_df.to_csv(output_csv, index=True)
print(f"\nEquity curves saved to: {output_csv}")
print("Done!")
