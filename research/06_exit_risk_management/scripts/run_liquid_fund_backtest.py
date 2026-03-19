"""
Liquid Fund Enhanced Backtest for V3 Breakout Systems
=====================================================

Simulates parking idle cash in liquid/overnight funds at 6.5% p.a.
When breakout signals fire, capital is withdrawn for trades.
When trades exit, capital + P&L returns to liquid fund.

This answers: "How much extra return do we earn by not letting cash sit idle?"
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from services.consolidation_breakout import (
    SYSTEM_SNIPER, SYSTEM_PRIMARY, SYSTEM_BALANCED,
    SYSTEM_ACTIVE, SYSTEM_HIGH_VOLUME,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
LIQUID_FUND_RATE = 0.065   # 6.5% p.a. (typical liquid/overnight fund)
DAILY_RATE = (1 + LIQUID_FUND_RATE) ** (1/365) - 1
CAPITAL_PER_TRADE = 100_000  # Rs.1L per trade
STARTING_CAPITAL = 1_000_000  # Rs.10L total capital

SYSTEMS = {
    'SNIPER': SYSTEM_SNIPER,
    'PRIMARY': SYSTEM_PRIMARY,
    'BALANCED': SYSTEM_BALANCED,
    'ACTIVE': SYSTEM_ACTIVE,
    'HIGH_VOLUME': SYSTEM_HIGH_VOLUME,
}


def apply_system_mask(df, system):
    """Vectorized: return boolean mask for trades matching ANY strategy in system."""
    mask = pd.Series(False, index=df.index)
    for strategy in system:
        strat_mask = pd.Series(True, index=df.index)
        for key, value in strategy.items():
            if key == 'min_rsi14': strat_mask &= df['rsi14'].fillna(0) >= value
            elif key == 'min_rsi7': strat_mask &= df['rsi7'].fillna(0) >= value
            elif key == 'min_volume_ratio': strat_mask &= df['volume_ratio'].fillna(0) >= value
            elif key == 'min_vol_trend': strat_mask &= df['vol_trend'].fillna(0) >= value
            elif key == 'min_breakout_pct': strat_mask &= df['breakout_pct'].fillna(0) >= value
            elif key == 'min_ath_proximity': strat_mask &= df['ath_proximity'].fillna(0) >= value
            elif key == 'min_williams_r': strat_mask &= df['williams_r'].fillna(-100) >= value
            elif key == 'min_mom_60d': strat_mask &= df['mom_60d'].fillna(0) >= value
            elif key == 'min_mom_10d': strat_mask &= df['mom_10d'].fillna(0) >= value
            elif key == 'min_bb_pct_b': strat_mask &= df['bb_pct_b'].fillna(0) >= value
            elif key == 'require_ema20_gt_50' and value: strat_mask &= df['ema20_above_50'].fillna(0) == 1
            elif key == 'require_weekly_ema20_gt_50' and value: strat_mask &= df['w_ema20_gt_50'].fillna(0) == 1
        mask |= strat_mask
    return mask


def estimate_holding_calendar_days(row):
    """Estimate how long capital is locked in a trade (calendar days)."""
    if row['exit_reason'] == 'OPEN':
        return int(125 * 1.43)  # Full 125 trading day window ~179 cal days
    else:  # STOP
        # Peak happens at days_to_peak, then price falls to stop
        # Estimate exit at ~1.5x days_to_peak + 10 buffer, capped at 125 trading days
        trading_days = min(int(row['days_to_peak'] * 1.5) + 10, 125)
        return int(trading_days * 1.43)


def run_liquid_fund_simulation(trades_df, system_name, starting_capital=None):
    """Run the full simulation for a system's qualifying trades."""
    if starting_capital is None:
        starting_capital = STARTING_CAPITAL
    trades = trades_df.sort_values('date').copy()
    n_total = len(trades)

    if n_total == 0:
        return None

    # Estimate exit dates
    trades['holding_days'] = trades.apply(estimate_holding_calendar_days, axis=1)
    trades['exit_date'] = trades['date'] + pd.to_timedelta(trades['holding_days'], unit='D')

    sim_start = trades['date'].min()
    sim_end = max(trades['exit_date'].max(), trades['date'].max())
    years = (sim_end - sim_start).days / 365.25

    # ====================================================================
    # ENHANCED SIMULATION (liquid fund + trades)
    # ====================================================================
    liquid_fund = float(starting_capital)
    active = []  # list of (exit_date, capital_deployed, trade_return_pct)
    trades_taken = 0
    trades_skipped = 0
    total_liquid_interest = 0.0
    total_trade_pnl = 0.0
    max_concurrent = 0
    last_event_date = sim_start

    # Yearly tracking
    yearly_interest = {}
    yearly_trade_pnl = {}

    for _, row in trades.iterrows():
        entry_date = row['date']
        exit_date = row['exit_date']
        trade_return = row['trade_return']
        year = entry_date.year

        # 1. Accrue liquid fund interest from last event to now
        if entry_date > last_event_date:
            days_gap = (entry_date - last_event_date).days
            interest = liquid_fund * ((1 + DAILY_RATE) ** days_gap - 1)
            liquid_fund += interest
            total_liquid_interest += interest
            yearly_interest[year] = yearly_interest.get(year, 0) + interest

        # 2. Process exits that occurred before this entry
        still_active = []
        for (ex_date, cap_deployed, ret_pct) in active:
            if ex_date <= entry_date:
                returned = cap_deployed * (1 + ret_pct / 100)
                liquid_fund += returned
                pnl = returned - cap_deployed
                total_trade_pnl += pnl
                ex_year = ex_date.year
                yearly_trade_pnl[ex_year] = yearly_trade_pnl.get(ex_year, 0) + pnl
            else:
                still_active.append((ex_date, cap_deployed, ret_pct))
        active = still_active

        # 3. Try to take this trade
        if liquid_fund >= CAPITAL_PER_TRADE:
            liquid_fund -= CAPITAL_PER_TRADE
            active.append((exit_date, CAPITAL_PER_TRADE, trade_return))
            trades_taken += 1
            max_concurrent = max(max_concurrent, len(active))
        else:
            trades_skipped += 1

        last_event_date = entry_date

    # Final: accrue interest to sim_end and close all active trades
    if sim_end > last_event_date:
        days_gap = (sim_end - last_event_date).days
        interest = liquid_fund * ((1 + DAILY_RATE) ** days_gap - 1)
        liquid_fund += interest
        total_liquid_interest += interest

    for (ex_date, cap_deployed, ret_pct) in active:
        returned = cap_deployed * (1 + ret_pct / 100)
        liquid_fund += returned
        pnl = returned - cap_deployed
        total_trade_pnl += pnl

    final_equity_enhanced = liquid_fund

    # ====================================================================
    # BASE SIMULATION (no liquid fund, same trade allocation)
    # ====================================================================
    base_trade_pnl = 0.0
    for _, row in trades.iterrows():
        base_trade_pnl += CAPITAL_PER_TRADE * (row['trade_return'] / 100)
    base_equity = starting_capital + base_trade_pnl

    # ====================================================================
    # PURE LIQUID FUND (no trades at all)
    # ====================================================================
    total_days = (sim_end - sim_start).days
    pure_liquid = starting_capital * ((1 + DAILY_RATE) ** total_days)

    # ====================================================================
    # COMPUTE METRICS
    # ====================================================================
    base_cagr = ((base_equity / starting_capital) ** (1/years) - 1) * 100
    enhanced_cagr = ((final_equity_enhanced / starting_capital) ** (1/years) - 1) * 100
    pure_liquid_cagr = ((pure_liquid / starting_capital) ** (1/years) - 1) * 100

    total_gains = final_equity_enhanced - STARTING_CAPITAL
    liquid_pct = (total_liquid_interest / total_gains * 100) if total_gains > 0 else 0

    # Capital utilization
    avg_holding = trades['holding_days'].mean()
    total_deployed_days = trades_taken * avg_holding
    total_available_days = total_days * (starting_capital / CAPITAL_PER_TRADE)  # slots x total days
    utilization = (total_deployed_days / total_available_days * 100) if total_available_days > 0 else 0

    return {
        'system': system_name,
        'trades_total': n_total,
        'trades_taken': trades_taken,
        'trades_skipped': trades_skipped,
        'max_concurrent': max_concurrent,
        'years': round(years, 1),
        'base_equity': base_equity,
        'enhanced_equity': final_equity_enhanced,
        'pure_liquid': pure_liquid,
        'base_cagr': round(base_cagr, 2),
        'enhanced_cagr': round(enhanced_cagr, 2),
        'pure_liquid_cagr': round(pure_liquid_cagr, 2),
        'cagr_boost': round(enhanced_cagr - base_cagr, 2),
        'trade_pnl': round(total_trade_pnl, 0),
        'liquid_interest': round(total_liquid_interest, 0),
        'liquid_pct_of_gains': round(liquid_pct, 1),
        'capital_utilization': round(utilization, 1),
        'avg_holding_days': round(avg_holding, 0),
        'yearly_interest': yearly_interest,
        'yearly_trade_pnl': yearly_trade_pnl,
    }


def main():
    # Load data
    df = pd.read_csv('breakout_analysis_enhanced.csv')
    df['date'] = pd.to_datetime(df['date'])

    print('=' * 95)
    print('LIQUID FUND ENHANCED BACKTEST - V3 BREAKOUT SYSTEMS')
    print('=' * 95)
    print(f'Dataset: {len(df)} trades | {df["symbol"].nunique()} stocks | {df["date"].min().date()} to {df["date"].max().date()}')
    print(f'Capital: Rs.{STARTING_CAPITAL/100000:.0f}L | Per trade: Rs.{CAPITAL_PER_TRADE/100000:.0f}L | Liquid fund: {LIQUID_FUND_RATE*100:.1f}% p.a.')
    print(f'Assumption: OPEN trades held 125 trading days (~179 cal days)')
    print(f'            STOP trades held ~1.5x days_to_peak + buffer')
    print('=' * 95)

    results = {}
    for sys_name, system in SYSTEMS.items():
        mask = apply_system_mask(df, system)
        filtered = df[mask].copy()
        r = run_liquid_fund_simulation(filtered, sys_name)
        if r:
            results[sys_name] = r

    # ====================================================================
    # SECTION 1: System-by-System Detail
    # ====================================================================
    for name, r in results.items():
        print(f'\n{"=" * 50}')
        print(f'  {name}')
        print(f'{"=" * 50}')
        print(f'  Trades: {r["trades_taken"]} taken, {r["trades_skipped"]} skipped')
        print(f'  Max concurrent: {r["max_concurrent"]} trades open at once')
        print(f'  Avg holding: {r["avg_holding_days"]:.0f} calendar days')
        print(f'  Capital utilization: {r["capital_utilization"]:.1f}%')
        print(f'  Period: {r["years"]} years')
        print()
        print(f'  {"Metric":<25} {"Base":>15} {"+ Liquid Fund":>15} {"Boost":>12}')
        print(f'  {"-"*67}')
        print(f'  {"Final Equity":<25} Rs.{r["base_equity"]:>12,.0f} Rs.{r["enhanced_equity"]:>12,.0f}  +Rs.{r["enhanced_equity"]-r["base_equity"]:>8,.0f}')
        print(f'  {"CAGR":<25} {r["base_cagr"]:>14.2f}% {r["enhanced_cagr"]:>14.2f}%  +{r["cagr_boost"]:.2f}%')
        print(f'  {"Trade P&L":<25} Rs.{r["trade_pnl"]:>12,.0f}')
        print(f'  {"Liquid Fund Interest":<25} {"":>15} Rs.{r["liquid_interest"]:>12,.0f}  ({r["liquid_pct_of_gains"]:.1f}% of gains)')
        print(f'  {"Pure Liquid (no trades)":<25} Rs.{r["pure_liquid"]:>12,.0f} CAGR: {r["pure_liquid_cagr"]:.2f}%')

    # ====================================================================
    # SECTION 2: Comparison Table
    # ====================================================================
    print(f'\n\n{"=" * 95}')
    print('COMPARISON SUMMARY')
    print(f'{"=" * 95}')
    print(f'{"System":<14} {"Trades":>7} {"Skip":>5} {"MaxC":>5} {"Base CAGR":>10} {"Enh CAGR":>10} {"Boost":>8} {"Liquid Int":>14} {"% Gains":>8} {"Util%":>6}')
    print(f'{"-" * 95}')
    for name, r in results.items():
        print(f'{name:<14} {r["trades_taken"]:>7} {r["trades_skipped"]:>5} {r["max_concurrent"]:>5} {r["base_cagr"]:>9.2f}% {r["enhanced_cagr"]:>9.2f}% {"+"+str(r["cagr_boost"])+"%" :>8} Rs.{r["liquid_interest"]:>10,.0f} {r["liquid_pct_of_gains"]:>7.1f}% {r["capital_utilization"]:>5.1f}%')

    # ====================================================================
    # SECTION 3: What if we increase capital?
    # ====================================================================
    print(f'\n\n{"=" * 95}')
    print('SENSITIVITY: WHAT IF WE CHANGE STARTING CAPITAL?')
    print(f'{"=" * 95}')
    print('(Using PRIMARY system)')

    primary_trades = df[apply_system_mask(df, SYSTEM_PRIMARY)].copy()

    for cap_multiplier, label in [(5, 'Rs.5L'), (10, 'Rs.10L'), (20, 'Rs.20L'), (50, 'Rs.50L')]:
        cap = cap_multiplier * 100_000
        r = run_liquid_fund_simulation(primary_trades, 'PRIMARY', starting_capital=cap)
        if r:
            print(f'  {label:>8} starting: CAGR {r["enhanced_cagr"]:.2f}% | Liquid interest Rs.{r["liquid_interest"]:,.0f} ({r["liquid_pct_of_gains"]:.1f}% of gains) | {r["trades_skipped"]} skipped | Util: {r["capital_utilization"]:.1f}%')

    # ====================================================================
    # SECTION 4: Year-by-year for PRIMARY
    # ====================================================================
    print(f'\n\n{"=" * 95}')
    print('YEAR-BY-YEAR: PRIMARY SYSTEM (Enhanced)')
    print(f'{"=" * 95}')
    r = results.get('PRIMARY', {})
    if r:
        all_years = sorted(set(list(r.get('yearly_interest', {}).keys()) + list(r.get('yearly_trade_pnl', {}).keys())))
        print(f'{"Year":<8} {"Trade P&L":>14} {"Liquid Interest":>16} {"Total":>14} {"Liquid %":>10}')
        print(f'{"-" * 65}')
        for year in all_years:
            tp = r['yearly_trade_pnl'].get(year, 0)
            li = r['yearly_interest'].get(year, 0)
            total = tp + li
            pct = (li / total * 100) if total > 0 else 0
            print(f'{year:<8} Rs.{tp:>11,.0f} Rs.{li:>13,.0f} Rs.{total:>11,.0f} {pct:>8.1f}%')

    # ====================================================================
    # SECTION 5: Key Insight
    # ====================================================================
    print(f'\n\n{"=" * 95}')
    print('KEY INSIGHTS')
    print(f'{"=" * 95}')

    if 'PRIMARY' in results:
        r = results['PRIMARY']
        print(f'  1. PRIMARY system: {r["trades_taken"]} trades over {r["years"]} years = ~{r["trades_taken"]/r["years"]:.0f} trades/year')
        print(f'  2. Capital utilization only {r["capital_utilization"]:.1f}% - most capital sits idle')
        print(f'  3. Liquid fund adds Rs.{r["liquid_interest"]:,.0f} ({r["liquid_pct_of_gains"]:.1f}% of total gains)')
        print(f'  4. CAGR boost: {r["base_cagr"]:.2f}% -> {r["enhanced_cagr"]:.2f}% (+{r["cagr_boost"]:.2f}%)')
        print(f'  5. Max {r["max_concurrent"]} trades open simultaneously (of 10 possible slots)')
        print(f'  6. {r["trades_skipped"]} trades skipped due to insufficient capital')
        print(f'  7. In years with NO breakout signals, capital still earns {LIQUID_FUND_RATE*100:.1f}% in liquid fund')
        print(f'  8. Pure liquid fund over same period: Rs.{r["pure_liquid"]:,.0f} ({r["pure_liquid_cagr"]:.2f}% CAGR)')

    print(f'\n  RECOMMENDATION:')
    print(f'  - Park ALL idle capital in liquid/overnight funds (Parag Parikh Liquid, HDFC Overnight, etc.)')
    print(f'  - These offer instant/T+1 redemption - no delay when signals fire')
    print(f'  - The {LIQUID_FUND_RATE*100:.1f}% idle return meaningfully boosts overall portfolio CAGR')
    print(f'  - Combined strategy: Breakout alpha + risk-free idle returns = best of both worlds')


if __name__ == '__main__':
    main()
