"""
Combined MQ + Tactical Pool Backtest
=====================================

Runs MQ Core (60%) and KC6 (40%) simultaneously, combines equity curves
to get exact combined MaxDD, CAGR, Sharpe, and Calmar ratios.

MQ Core: Momentum+Quality, 30 stocks, semi-annual rebalance, ATH20 exit
KC6: Mean reversion, KC(6, 1.3ATR), SMA200 filter, max 7 positions
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig
from services.kc6_backtest_engine import KC6BacktestEngine, KC6BacktestConfig

print("=" * 70)
print("COMBINED MQ + TACTICAL POOL BACKTEST")
print("=" * 70)

# =============================================================================
# Configuration
# =============================================================================
TOTAL_CAPITAL = 10_000_000  # Rs. 1 Crore
MQ_PCT = 0.60               # 60% -> Rs. 60L
TACTICAL_PCT = 0.40          # 40% -> Rs. 40L

MQ_CAPITAL = TOTAL_CAPITAL * MQ_PCT        # 6,000,000
KC6_CAPITAL = TOTAL_CAPITAL * TACTICAL_PCT  # 4,000,000

# Backtest period (same as MQ optimization baseline)
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'

# KC6 needs SMA200 warmup, so load data from earlier
# KC6 start_date = 2022-01-01 → engine warmup skips to 2023-01-01
KC6_DATA_START = '2022-01-01'

print(f"\nTotal Capital:  Rs.{TOTAL_CAPITAL:,.0f} (Rs.{TOTAL_CAPITAL/1e5:.0f}L)")
print(f"MQ Core (60%):  Rs.{MQ_CAPITAL:,.0f} (Rs.{MQ_CAPITAL/1e5:.0f}L)")
print(f"Tactical (40%): Rs.{KC6_CAPITAL:,.0f} (Rs.{KC6_CAPITAL/1e5:.0f}L)")
print(f"Period: {START_DATE} to {END_DATE}")

# =============================================================================
# Phase 1: Load shared data (with extra lookback for KC6 SMA200)
# =============================================================================
print("\n[Phase 1] Loading shared market data...")

# Use KC6_DATA_START so preload goes 400 days before 2022-01-01 (~Nov 2020)
# This gives enough history for both MQ momentum lookback AND KC6 SMA200
loader_config = MQBacktestConfig(
    start_date=KC6_DATA_START,
    end_date=END_DATE,
)
universe, price_data = MQBacktestEngine.preload_data(loader_config)
print(f"  Loaded {len(price_data)} symbols")

# =============================================================================
# Phase 2: Run MQ Core backtest (60% allocation)
# =============================================================================
print("\n[Phase 2] Running MQ Core backtest (60% = Rs.60L)...")

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
print(f"  MQ Calmar:      {mq_result.calmar_ratio}")
print(f"  MQ Final Value: Rs.{mq_result.final_value:,.0f} (Rs.{mq_result.final_value/1e5:.1f}L)")
print(f"  MQ Trades:      {mq_result.total_trades}")

# =============================================================================
# Phase 3: Run KC6 backtest (40% tactical allocation)
# =============================================================================
print("\n[Phase 3] Running KC6 backtest (40% = Rs.40L)...")

kc6_config = KC6BacktestConfig(
    start_date=KC6_DATA_START,  # Engine auto-warmup skips to 2023-01-01
    end_date=END_DATE,
    initial_capital=int(KC6_CAPITAL),
    max_positions=7,
    position_size_pct=0.10,   # 10% of 40L = Rs.4L per trade
    sl_pct=5.0,
    tp_pct=15.0,
    max_hold_days=15,
    debt_rate=7.0,            # Match tactical pool debt fund rate
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
print(f"  KC6 Calmar:      {kc6_result.calmar}")
print(f"  KC6 Final Value: Rs.{kc6_result.final_value:,.0f} (Rs.{kc6_result.final_value/1e5:.1f}L)")
print(f"  KC6 Trades:      {kc6_result.total_trades}")
print(f"  KC6 Win Rate:    {kc6_result.win_rate:.1f}%")
print(f"  KC6 Profit Factor: {kc6_result.profit_factor:.2f}")
print(f"  KC6 Max Concurrent: {kc6_result.max_concurrent}")
print(f"  KC6 Crash Filter Days: {kc6_result.crash_filter_days}")

# =============================================================================
# Phase 4: Combine equity curves
# =============================================================================
print("\n[Phase 4] Combining equity curves...")

# Convert to pandas Series
mq_eq = pd.Series(mq_result.daily_equity, dtype=float)
mq_eq.index = pd.to_datetime(mq_eq.index)
mq_eq = mq_eq.sort_index()

kc6_eq = pd.Series(kc6_result.daily_equity, dtype=float)
kc6_eq.index = pd.to_datetime(kc6_eq.index)
kc6_eq = kc6_eq.sort_index()

print(f"  MQ equity curve:  {len(mq_eq)} days ({mq_eq.index[0].date()} to {mq_eq.index[-1].date()})")
print(f"  KC6 equity curve: {len(kc6_eq)} days ({kc6_eq.index[0].date()} to {kc6_eq.index[-1].date()})")

# Use union of dates with forward-fill for alignment
combined_df = pd.DataFrame({
    'mq': mq_eq,
    'kc6': kc6_eq,
}).sort_index()

# Forward-fill gaps, then drop rows before both series have started
combined_df = combined_df.ffill()
combined_df = combined_df.dropna()

# Combined equity = MQ value + KC6 value
combined_df['combined'] = combined_df['mq'] + combined_df['kc6']
combined_eq = combined_df['combined']

print(f"  Combined curve:   {len(combined_eq)} days ({combined_eq.index[0].date()} to {combined_eq.index[-1].date()})")
print(f"  Start value:      Rs.{combined_eq.iloc[0]:,.0f} (Rs.{combined_eq.iloc[0]/1e5:.1f}L)")
print(f"  End value:        Rs.{combined_eq.iloc[-1]:,.0f} (Rs.{combined_eq.iloc[-1]/1e5:.1f}L)")

# =============================================================================
# Phase 5: Calculate combined metrics
# =============================================================================
print("\n[Phase 5] Calculating combined metrics...")

initial = combined_eq.iloc[0]
final = combined_eq.iloc[-1]

# Years
years = (combined_eq.index[-1] - combined_eq.index[0]).days / 365.25

# CAGR
cagr = ((final / initial) ** (1 / years) - 1) * 100

# Daily returns
daily_ret = combined_eq.pct_change().dropna()

# Sharpe (7% risk-free rate)
rf_daily = 0.07 / 252
excess = daily_ret - rf_daily
sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

# Sortino
downside = daily_ret[daily_ret < 0]
sortino = float(excess.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0

# Max Drawdown
peak = combined_eq.expanding().max()
dd = (combined_eq - peak) / peak * 100
max_dd = abs(float(dd.min()))

# Drawdown detail
max_dd_date = dd.idxmin()
peak_before_dd = combined_eq[:max_dd_date].idxmax()
trough_value = combined_eq[max_dd_date]
peak_value = combined_eq[peak_before_dd]

# Calmar
calmar = cagr / max_dd if max_dd > 0 else 0

# Also compute MQ-only and KC6-only drawdowns on the common period for comparison
mq_common = combined_df['mq']
kc6_common = combined_df['kc6']

mq_peak = mq_common.expanding().max()
mq_dd = (mq_common - mq_peak) / mq_peak * 100
mq_maxdd_common = abs(float(mq_dd.min()))

kc6_peak = kc6_common.expanding().max()
kc6_dd = (kc6_common - kc6_peak) / kc6_peak * 100
kc6_maxdd_common = abs(float(kc6_dd.min()))

# Find the combined DD at MQ's worst point
mq_worst_date = mq_dd.idxmin()
combined_dd_at_mq_worst = dd.loc[mq_worst_date] if mq_worst_date in dd.index else 0

# =============================================================================
# Report
# =============================================================================
print("\n" + "=" * 70)
print("COMBINED SYSTEM RESULTS")
print("=" * 70)

print(f"\n{'Metric':<25} {'MQ Core (60%)':<18} {'KC6 (40%)':<18} {'Combined':<18}")
print("-" * 79)
print(f"{'Allocation':<25} {'Rs.60L':<18} {'Rs.40L':<18} {'Rs.100L':<18}")
print(f"{'Final Value':<25} {'Rs.'+f'{mq_result.final_value/1e5:.1f}L':<18} {'Rs.'+f'{kc6_result.final_value/1e5:.1f}L':<18} {'Rs.'+f'{final/1e5:.1f}L':<18}")
print(f"{'CAGR':<25} {mq_result.cagr:>6.2f}%{'':<10} {kc6_result.cagr:>6.2f}%{'':<10} {cagr:>6.2f}%")
print(f"{'Max Drawdown':<25} {mq_maxdd_common:>6.2f}%{'':<10} {kc6_maxdd_common:>6.1f}%{'':<11} {max_dd:>6.2f}%")
print(f"{'Sharpe Ratio':<25} {mq_result.sharpe_ratio:>6.2f}{'':<11} {kc6_result.sharpe:>6.2f}{'':<11} {sharpe:>6.2f}")
print(f"{'Sortino Ratio':<25} {mq_result.sortino_ratio:>6.2f}{'':<11} {kc6_result.sortino:>6.2f}{'':<11} {sortino:>6.2f}")
print(f"{'Calmar Ratio':<25} {mq_result.calmar_ratio:>6.2f}{'':<11} {kc6_result.calmar:>6.2f}{'':<11} {calmar:>6.2f}")
print(f"{'Total Trades':<25} {mq_result.total_trades:>6d}{'':<11} {kc6_result.total_trades:>6d}{'':<11} {mq_result.total_trades + kc6_result.total_trades:>6d}")

print(f"\n{'Drawdown Detail':}")
print(f"  Peak before DD:     {peak_before_dd.strftime('%Y-%m-%d')} (Rs.{peak_value/1e5:.1f}L)")
print(f"  Max DD date:        {max_dd_date.strftime('%Y-%m-%d')} (Rs.{trough_value/1e5:.1f}L)")
print(f"  Peak -> Trough:     Rs.{(peak_value - trough_value)/1e5:.1f}L lost")
print(f"  Recovery needed:    {max_dd/(100-max_dd)*100:.1f}% to recover")

print(f"\n  When MQ hit its worst DD ({mq_worst_date.strftime('%Y-%m-%d')}):")
print(f"    MQ drawdown:      {mq_dd.loc[mq_worst_date]:.2f}%")
kc6_dd_val = kc6_dd.loc[mq_worst_date] if mq_worst_date in kc6_dd.index else 0
print(f"    KC6 drawdown:     {kc6_dd_val:.2f}%")
print(f"    Combined DD:      {combined_dd_at_mq_worst:.2f}%")
print(f"    KC6 cushioned by: {abs(float(mq_dd.loc[mq_worst_date])) - abs(float(combined_dd_at_mq_worst)):.2f}%")

# Nifty 50 comparison
nifty_cagr = 11.2
nifty_maxdd = 36.3
nifty_calmar = nifty_cagr / nifty_maxdd

print(f"\n{'='*70}")
print(f"COMPARISON vs NIFTY 50")
print(f"{'='*70}")
print(f"\n{'Metric':<25} {'Our System':<18} {'Nifty 50 (16yr)':<18} {'Advantage':<18}")
print(f"{'-'*70}")
print(f"{'CAGR':<25} {cagr:>6.2f}%{'':<10} {nifty_cagr:>6.2f}%{'':<10} {'+' if cagr > nifty_cagr else ''}{cagr - nifty_cagr:.1f}%")
print(f"{'Max Drawdown':<25} {max_dd:>6.2f}%{'':<10} {nifty_maxdd:>6.2f}%{'':<10} {nifty_maxdd - max_dd:.1f}% better")
print(f"{'Calmar Ratio':<25} {calmar:>6.2f}{'':<11} {nifty_calmar:>6.2f}{'':<11} {calmar/nifty_calmar:.1f}x better")

# IPO strategies note
print(f"\n{'='*70}")
print("NOTE: This backtest covers MQ + KC6 only.")
print("IPO Scalper + IPO Swing are NOT included (no backtest engine).")
print("Their ~Rs.4.2L expected annual return would further IMPROVE returns")
print("without significantly increasing drawdowns (uncorrelated strategies).")
print(f"{'='*70}")

# Save combined equity curve to CSV
output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_equity_curve.csv')
combined_df.to_csv(output_csv, index=True)
print(f"\nEquity curves saved to: {output_csv}")

print("\nDone!")
