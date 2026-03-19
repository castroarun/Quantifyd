"""
Trade Verification & Visualization
====================================

Picks random PRIMARY system trades, looks up actual prices from market_data.db,
simulates the full 5x married put portfolio, and generates:
1. Detailed trade log with entry/exit prices, signals, P&L
2. Portfolio growth and drawdown charts
3. Pine Script for TradingView verification
"""
import pandas as pd
import numpy as np
import sqlite3
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from math import log, sqrt, exp
from scipy.stats import norm

from services.consolidation_breakout import SYSTEM_PRIMARY

# =========================================================================
# Constants
# =========================================================================
STARTING_CAPITAL = 1_000_000
LEVERAGE = 5
POS_PCT = 0.05
LIQUID_FUND_RATE = 0.065
DAILY_RATE = (1 + LIQUID_FUND_RATE) ** (1/365) - 1
FUTURES_CARRY_ANNUAL = 0.07
ROLLOVER_COST = 0.002
RISK_FREE_RATE = 0.065
OTM_PUT = 10  # 10% OTM married put
IV = 0.30
DB_PATH = Path('backtest_data/market_data.db')
CSV_PATH = Path('breakout_analysis_enhanced.csv')
OUTPUT_DIR = Path('verification_output')


def apply_mask(df, system):
    mask = pd.Series(False, index=df.index)
    for strategy in system:
        sm = pd.Series(True, index=df.index)
        for k, v in strategy.items():
            if k == 'min_rsi14': sm &= df['rsi14'].fillna(0) >= v
            elif k == 'min_rsi7': sm &= df['rsi7'].fillna(0) >= v
            elif k == 'min_volume_ratio': sm &= df['volume_ratio'].fillna(0) >= v
            elif k == 'min_vol_trend': sm &= df['vol_trend'].fillna(0) >= v
            elif k == 'min_breakout_pct': sm &= df['breakout_pct'].fillna(0) >= v
            elif k == 'min_ath_proximity': sm &= df['ath_proximity'].fillna(0) >= v
            elif k == 'min_williams_r': sm &= df['williams_r'].fillna(-100) >= v
            elif k == 'min_mom_60d': sm &= df['mom_60d'].fillna(0) >= v
            elif k == 'min_mom_10d': sm &= df['mom_10d'].fillna(0) >= v
            elif k == 'min_bb_pct_b': sm &= df['bb_pct_b'].fillna(0) >= v
            elif k == 'require_ema20_gt_50' and v: sm &= df['ema20_above_50'].fillna(0) == 1
            elif k == 'require_weekly_ema20_gt_50' and v: sm &= df['w_ema20_gt_50'].fillna(0) == 1
        mask |= sm
    return mask


def hold_days(row):
    if row['exit_reason'] == 'OPEN':
        return int(125 * 1.43)
    else:
        return int(min(int(row['days_to_peak'] * 1.5) + 10, 125) * 1.43)


def identify_strategy(row):
    """Identify which PRIMARY sub-strategy triggered."""
    # ALPHA: RSI>=75 + Vol>=3x + EMA20>50
    alpha = (row.get('rsi14', 0) or 0) >= 75 and \
            (row.get('volume_ratio', 0) or 0) >= 3.0 and \
            (row.get('ema20_above_50', 0) or 0) == 1
    # T1B: BO>=3% + VolTr>=1.2 + ATH>=85% + EMA20>50 + wEMA20>50 + WillR>=-20
    t1b = (row.get('breakout_pct', 0) or 0) >= 3.0 and \
          (row.get('vol_trend', 0) or 0) >= 1.2 and \
          (row.get('ath_proximity', 0) or 0) >= 85 and \
          (row.get('ema20_above_50', 0) or 0) == 1 and \
          (row.get('w_ema20_gt_50', 0) or 0) == 1 and \
          (row.get('williams_r', -100) if pd.notna(row.get('williams_r')) else -100) >= -20
    # MOMVOL: Mom60>=15 + VolTr>=1.2 + ATH>=90% + Vol>=3x
    momvol = (row.get('mom_60d', 0) or 0) >= 15 and \
             (row.get('vol_trend', 0) or 0) >= 1.2 and \
             (row.get('ath_proximity', 0) or 0) >= 90 and \
             (row.get('volume_ratio', 0) or 0) >= 3.0
    matched = []
    if alpha: matched.append('ALPHA')
    if t1b: matched.append('T1B')
    if momvol: matched.append('MOMVOL')
    return ' + '.join(matched) if matched else 'UNKNOWN'


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0) / S * 100
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price / S * 100


def get_price_data(symbol, date_str, conn):
    """Get OHLCV data around the breakout date."""
    query = """
    SELECT date, open, high, low, close, volume
    FROM market_data_unified
    WHERE symbol = ? AND timeframe = 'day'
    AND date >= date(?, '-30 days') AND date <= date(?, '+250 days')
    ORDER BY date
    """
    return pd.read_sql(query, conn, params=(symbol, date_str, date_str))


def compute_hedged_return(base_return, hold_d):
    """Compute married put hedged return at 5x leverage."""
    carry_cost = FUTURES_CARRY_ANNUAL * (hold_d / 365.0) * 100
    n_rollovers = max(0, (hold_d / 30.0) - 1)
    rollover_cost = ROLLOVER_COST * n_rollovers * 100

    option_periods = max(1, hold_d / 30)
    put_premium = bs_put_price(100, 100 * (1 - OTM_PUT / 100), 30 / 365, RISK_FREE_RATE, IV)
    total_premium = put_premium * option_periods

    floored_return = max(base_return, -OTM_PUT)
    lev_return = (floored_return * LEVERAGE) - carry_cost - rollover_cost - (total_premium * LEVERAGE)
    lev_return = max(lev_return, -100.0)
    return lev_return


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print('=' * 110)
    print('TRADE VERIFICATION - PRIMARY SYSTEM, 5x LEVERAGE, MARRIED PUT 10% OTM')
    print('=' * 110)

    # Load trades
    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'])
    primary = df[apply_mask(df, SYSTEM_PRIMARY)].copy()
    primary['holding_days'] = primary.apply(hold_days, axis=1)
    primary['exit_date'] = primary['date'] + pd.to_timedelta(primary['holding_days'], unit='D')
    primary = primary.sort_values('date').reset_index(drop=True)

    print(f'\nTotal PRIMARY trades: {len(primary)}')
    print(f'Date range: {primary["date"].min().date()} to {primary["date"].max().date()}')

    # Connect to price database
    conn = sqlite3.connect(DB_PATH)

    # =====================================================================
    # SECTION 1: PICK 15 RANDOM TRADES FOR VERIFICATION
    # =====================================================================
    print('\n\n' + '=' * 110)
    print('SECTION 1: 15 RANDOM TRADES FOR MANUAL VERIFICATION')
    print('=' * 110)

    random.seed(42)
    sample_idx = random.sample(range(len(primary)), min(15, len(primary)))
    sample_trades = primary.iloc[sample_idx].copy()

    detailed_trades = []

    for i, (_, row) in enumerate(sample_trades.iterrows()):
        symbol = row['symbol']
        entry_date = row['date']
        exit_date = row['exit_date']
        base_return = row['trade_return']
        hold_d = row['holding_days']

        # Get actual prices
        prices = get_price_data(symbol, str(entry_date.date()), conn)

        entry_price = None
        exit_price = None
        peak_price = None
        stop_price = None

        if len(prices) > 0:
            prices['date'] = pd.to_datetime(prices['date'])
            # Entry: close on breakout date (or nearest)
            entry_mask = prices['date'] >= entry_date
            if entry_mask.any():
                entry_row = prices[entry_mask].iloc[0]
                entry_price = entry_row['close']

                # Calculate actual exit price from return %
                exit_price = entry_price * (1 + base_return / 100)

                # Peak price from max_gain
                peak_price = entry_price * (1 + row['max_gain'] / 100)

                # Stop price (risk_pct below entry)
                stop_price = entry_price * (1 - row['risk_pct'] / 100)

        # Identify which strategy triggered
        strategy = identify_strategy(row)

        # Compute hedged return
        hedged_ret = compute_hedged_return(base_return, hold_d)

        # Option premium details
        option_periods = max(1, hold_d / 30)
        put_prem_pct = bs_put_price(100, 100 * (1 - OTM_PUT / 100), 30 / 365, RISK_FREE_RATE, IV)
        total_put_cost = put_prem_pct * option_periods

        # Carry cost
        carry = FUTURES_CARRY_ANNUAL * (hold_d / 365.0) * 100

        detailed_trades.append({
            'trade_num': i + 1,
            'symbol': symbol,
            'entry_date': str(entry_date.date()),
            'exit_date': str(exit_date.date()),
            'hold_days': hold_d,
            'entry_price': entry_price,
            'exit_price': round(exit_price, 2) if exit_price else None,
            'peak_price': round(peak_price, 2) if peak_price else None,
            'stop_price': round(stop_price, 2) if stop_price else None,
            'strategy': strategy,
            'exit_reason': row['exit_reason'],
            'base_return': round(base_return, 2),
            'max_gain': round(row['max_gain'], 2),
            'risk_pct': round(row['risk_pct'], 2),
            'hedged_return': round(hedged_ret, 2),
            'carry_cost': round(carry, 2),
            'put_premium': round(total_put_cost, 2),
            'rsi14': round(row['rsi14'], 1) if pd.notna(row['rsi14']) else None,
            'rsi7': round(row['rsi7'], 1) if pd.notna(row.get('rsi7')) else None,
            'volume_ratio': round(row['volume_ratio'], 1),
            'ath_proximity': round(row['ath_proximity'], 1),
            'breakout_pct': round(row['breakout_pct'], 1),
            'detector': row['detector'],
        })

    # Print detailed trade table
    print(f'\n  {"#":>3} {"Symbol":<12} {"Entry":>12} {"Exit":>12} {"Days":>5} {"Strategy":<16} {"Reason":>6} {"Entry Rs":>10} {"Exit Rs":>10} {"Peak Rs":>10} {"Stop Rs":>10} {"Base%":>7} {"Hedged%":>8}')
    print(f'  {"-"*130}')
    for t in detailed_trades:
        ep = f'{t["entry_price"]:,.1f}' if t["entry_price"] else 'N/A'
        xp = f'{t["exit_price"]:,.1f}' if t["exit_price"] else 'N/A'
        pp = f'{t["peak_price"]:,.1f}' if t["peak_price"] else 'N/A'
        sp = f'{t["stop_price"]:,.1f}' if t["stop_price"] else 'N/A'
        print(f'  {t["trade_num"]:>3} {t["symbol"]:<12} {t["entry_date"]:>12} {t["exit_date"]:>12} {t["hold_days"]:>5} {t["strategy"]:<16} {t["exit_reason"]:>6} {ep:>10} {xp:>10} {pp:>10} {sp:>10} {t["base_return"]:>+6.1f}% {t["hedged_return"]:>+7.1f}%')

    # Print cost breakdown for each trade
    print(f'\n  COST BREAKDOWN PER TRADE:')
    print(f'  {"#":>3} {"Symbol":<12} {"Base%":>7} {"x5 Lev":>8} {"Carry":>7} {"PutCost":>8} {"Net Hedged":>11} {"RSI14":>6} {"Vol":>5} {"ATH%":>6} {"BO%":>5}')
    print(f'  {"-"*95}')
    for t in detailed_trades:
        raw_5x = t['base_return'] * 5
        print(f'  {t["trade_num"]:>3} {t["symbol"]:<12} {t["base_return"]:>+6.1f}% {raw_5x:>+7.1f}% -{t["carry_cost"]:>5.1f}% -{t["put_premium"]:>6.1f}% {t["hedged_return"]:>+10.1f}% {t["rsi14"] or 0:>5.0f} {t["volume_ratio"]:>4.0f}x {t["ath_proximity"]:>5.0f}% {t["breakout_pct"]:>4.0f}%')

    # =====================================================================
    # SECTION 2: FULL PORTFOLIO SIMULATION WITH DETAILED LOG
    # =====================================================================
    print('\n\n' + '=' * 110)
    print('SECTION 2: FULL PORTFOLIO SIMULATION (5x, 5%, Married Put 10% OTM)')
    print('=' * 110)

    trades = primary.copy()
    sim_start = trades['date'].min()
    sim_end = max(trades['exit_date'].max(), trades['date'].max())
    years = (sim_end - sim_start).days / 365.25

    cash = float(STARTING_CAPITAL)
    active = []
    taken = 0
    skipped = 0
    total_int = 0.0
    last_dt = sim_start
    margin_calls = 0

    # Track equity curve
    equity_dates = [sim_start]
    equity_values = [STARTING_CAPITAL]
    peak_equity = STARTING_CAPITAL
    max_dd = 0.0
    dd_dates = [sim_start]
    dd_values = [0.0]

    # Full trade log
    trade_log = []

    for idx, (_, row) in enumerate(trades.iterrows()):
        entry = row['date']
        base_return = row['trade_return']
        exit_d = row['exit_date']
        hold_d = row['holding_days']

        hedged_return = compute_hedged_return(base_return, hold_d)

        # Accrue liquid fund interest
        if entry > last_dt:
            days = (entry - last_dt).days
            interest = cash * ((1 + DAILY_RATE) ** days - 1)
            cash += interest
            total_int += interest

        # Process exits
        still = []
        for (ed, margin, lev_ret) in active:
            if ed <= entry:
                returned = margin * (1 + lev_ret / 100)
                returned = max(returned, 0)
                if lev_ret <= -95:
                    margin_calls += 1
                cash += returned
            else:
                still.append((ed, margin, lev_ret))
        active = still

        # Position sizing
        portfolio_val = cash + sum(m for (_, m, _) in active)
        trade_margin = portfolio_val * POS_PCT

        if cash >= trade_margin and trade_margin > 0:
            cash -= trade_margin
            active.append((exit_d, trade_margin, hedged_return))
            taken += 1
            status = 'TAKEN'
        else:
            skipped += 1
            status = 'SKIPPED'

        # Track equity
        current_equity = cash + sum(m for (_, m, _) in active)
        peak_equity = max(peak_equity, current_equity)
        dd = (peak_equity - current_equity) / peak_equity * 100
        max_dd = max(max_dd, dd)

        equity_dates.append(entry)
        equity_values.append(current_equity)
        dd_dates.append(entry)
        dd_values.append(-dd)

        trade_log.append({
            'num': taken if status == 'TAKEN' else 0,
            'date': str(entry.date()),
            'symbol': row['symbol'],
            'status': status,
            'base_return': round(base_return, 2),
            'hedged_return': round(hedged_return, 2),
            'margin': round(trade_margin, 0) if status == 'TAKEN' else 0,
            'portfolio': round(portfolio_val, 0),
            'cash_after': round(cash, 0),
            'equity': round(current_equity, 0),
            'drawdown': round(dd, 2),
            'active_positions': len(active),
        })

        last_dt = entry

    # Final close
    if sim_end > last_dt:
        days = (sim_end - last_dt).days
        interest = cash * ((1 + DAILY_RATE) ** days - 1)
        cash += interest
        total_int += interest

    for (ed, margin, lev_ret) in active:
        returned = margin * (1 + lev_ret / 100)
        returned = max(returned, 0)
        cash += returned

    final_equity = cash
    cagr = ((final_equity / STARTING_CAPITAL) ** (1 / years) - 1) * 100

    print(f'\n  Trades taken: {taken} | Skipped: {skipped} | Margin calls: {margin_calls}')
    print(f'  Final equity: Rs.{final_equity:,.0f} ({final_equity/100000:.0f}L)')
    print(f'  CAGR: {cagr:.1f}% | Max drawdown: {max_dd:.1f}% | Calmar: {cagr/max_dd:.2f}')
    print(f'  Liquid fund interest: Rs.{total_int:,.0f}')
    print(f'  Rs.10L -> Rs.{final_equity/100000:.0f}L over {years:.1f} years')

    # Print sample of trade log (first 20 taken trades)
    print(f'\n  TRADE LOG (first 20 taken trades):')
    print(f'  {"#":>4} {"Date":>12} {"Symbol":<12} {"Base%":>7} {"Hedged%":>8} {"Margin":>12} {"Portfolio":>14} {"Cash":>14} {"Equity":>14} {"DD%":>7} {"Active":>7}')
    print(f'  {"-"*120}')
    shown = 0
    for t in trade_log:
        if t['status'] == 'TAKEN' and shown < 20:
            print(f'  {t["num"]:>4} {t["date"]:>12} {t["symbol"]:<12} {t["base_return"]:>+6.1f}% {t["hedged_return"]:>+7.1f}% Rs.{t["margin"]:>9,.0f} Rs.{t["portfolio"]:>11,.0f} Rs.{t["cash_after"]:>11,.0f} Rs.{t["equity"]:>11,.0f} {t["drawdown"]:>6.1f}% {t["active_positions"]:>6}')
            shown += 1

    # Print worst 10 trades
    taken_trades = [t for t in trade_log if t['status'] == 'TAKEN']
    worst = sorted(taken_trades, key=lambda x: x['hedged_return'])[:10]
    print(f'\n  WORST 10 TRADES:')
    print(f'  {"#":>4} {"Date":>12} {"Symbol":<12} {"Base%":>7} {"Hedged%":>8} {"DD at time":>11}')
    print(f'  {"-"*60}')
    for t in worst:
        print(f'  {t["num"]:>4} {t["date"]:>12} {t["symbol"]:<12} {t["base_return"]:>+6.1f}% {t["hedged_return"]:>+7.1f}% {t["drawdown"]:>10.1f}%')

    # Print best 10 trades
    best = sorted(taken_trades, key=lambda x: x['hedged_return'], reverse=True)[:10]
    print(f'\n  BEST 10 TRADES:')
    print(f'  {"#":>4} {"Date":>12} {"Symbol":<12} {"Base%":>7} {"Hedged%":>8} {"Portfolio at time":>18}')
    print(f'  {"-"*70}')
    for t in best:
        print(f'  {t["num"]:>4} {t["date"]:>12} {t["symbol"]:<12} {t["base_return"]:>+6.1f}% {t["hedged_return"]:>+7.1f}% Rs.{t["portfolio"]:>14,.0f}')

    # =====================================================================
    # SECTION 3: CHARTS
    # =====================================================================
    print('\n\n' + '=' * 110)
    print('SECTION 3: GENERATING CHARTS')
    print('=' * 110)

    # Chart 1: Portfolio Growth
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('PRIMARY System | 5x Leverage | Married Put 10% OTM | Rs.10L Starting Capital',
                 fontsize=14, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1a1a2e')

    ax1 = axes[0]
    ax1.set_facecolor('#16213e')
    eq_dates_plot = pd.to_datetime(equity_dates)
    eq_values_L = [v / 100000 for v in equity_values]

    ax1.fill_between(eq_dates_plot, eq_values_L, alpha=0.3, color='#3fb950')
    ax1.plot(eq_dates_plot, eq_values_L, color='#3fb950', linewidth=1.5, label='Portfolio (Lakhs)')
    ax1.set_ylabel('Portfolio Value (Lakhs)', color='white', fontsize=11)
    ax1.set_title(f'Portfolio Growth: Rs.10L -> Rs.{final_equity/100000:.0f}L | CAGR: {cagr:.1f}%',
                  color='#3fb950', fontsize=12)
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='white')
    ax1.spines['bottom'].set_color('#333')
    ax1.spines['left'].set_color('#333')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add milestone markers
    milestones = [10, 50, 100, 500, 1000, 5000, 10000]
    for m in milestones:
        if max(eq_values_L) > m:
            idx = next((i for i, v in enumerate(eq_values_L) if v >= m), None)
            if idx:
                ax1.axhline(y=m, color='#555', linestyle='--', alpha=0.3)
                ax1.annotate(f'Rs.{m}L', xy=(eq_dates_plot[idx], m),
                           fontsize=8, color='#888', va='bottom')

    ax1.legend(loc='upper left', facecolor='#16213e', edgecolor='#333',
              labelcolor='white')

    # Chart 2: Drawdown
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    dd_dates_plot = pd.to_datetime(dd_dates)

    ax2.fill_between(dd_dates_plot, dd_values, alpha=0.4, color='#f85149')
    ax2.plot(dd_dates_plot, dd_values, color='#f85149', linewidth=1, label='Drawdown %')
    ax2.set_ylabel('Drawdown %', color='white', fontsize=11)
    ax2.set_title(f'Max Drawdown: {max_dd:.1f}%', color='#f85149', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.spines['bottom'].set_color('#333')
    ax2.spines['left'].set_color('#333')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axhline(y=-max_dd, color='#f85149', linestyle='--', alpha=0.5)
    ax2.annotate(f'Max DD: -{max_dd:.1f}%', xy=(dd_dates_plot[-1], -max_dd),
                fontsize=9, color='#f85149', ha='right')
    ax2.legend(loc='lower left', facecolor='#16213e', edgecolor='#333',
              labelcolor='white')

    plt.tight_layout()
    chart_path = OUTPUT_DIR / 'portfolio_growth_drawdown.png'
    plt.savefig(chart_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f'  Saved: {chart_path}')

    # Chart 3: Trade return distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Trade Return Distribution - 5x Married Put', fontsize=14,
                 fontweight='bold', color='white')
    fig.patch.set_facecolor('#1a1a2e')

    hedged_returns = [t['hedged_return'] for t in taken_trades]
    base_returns = [t['base_return'] for t in taken_trades]

    ax3 = axes[0]
    ax3.set_facecolor('#16213e')
    ax3.hist(base_returns, bins=40, color='#58a6ff', alpha=0.7, edgecolor='#333', label='Base (stock) return')
    ax3.axvline(x=0, color='white', linestyle='-', alpha=0.5)
    ax3.axvline(x=np.mean(base_returns), color='#d29922', linestyle='--',
               label=f'Mean: {np.mean(base_returns):.1f}%')
    ax3.set_xlabel('Return %', color='white')
    ax3.set_ylabel('Count', color='white')
    ax3.set_title('Base Stock Returns', color='#58a6ff')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#16213e', edgecolor='#333', labelcolor='white')
    ax3.spines['bottom'].set_color('#333')
    ax3.spines['left'].set_color('#333')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax4 = axes[1]
    ax4.set_facecolor('#16213e')
    ax4.hist(hedged_returns, bins=40, color='#3fb950', alpha=0.7, edgecolor='#333', label='Hedged (5x + MP)')
    ax4.axvline(x=0, color='white', linestyle='-', alpha=0.5)
    ax4.axvline(x=np.mean(hedged_returns), color='#d29922', linestyle='--',
               label=f'Mean: {np.mean(hedged_returns):.1f}%')
    ax4.set_xlabel('Return %', color='white')
    ax4.set_ylabel('Count', color='white')
    ax4.set_title('5x Leveraged + Married Put Returns', color='#3fb950')
    ax4.tick_params(colors='white')
    ax4.legend(facecolor='#16213e', edgecolor='#333', labelcolor='white')
    ax4.spines['bottom'].set_color('#333')
    ax4.spines['left'].set_color('#333')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()
    chart_path2 = OUTPUT_DIR / 'return_distribution.png'
    plt.savefig(chart_path2, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f'  Saved: {chart_path2}')

    # Chart 4: Year-by-year returns
    yearly = {}
    for t in taken_trades:
        year = t['date'][:4]
        if year not in yearly:
            yearly[year] = []
        yearly[year].append(t['hedged_return'])

    years_list = sorted(yearly.keys())
    year_returns = [np.mean(yearly[y]) for y in years_list]
    year_counts = [len(yearly[y]) for y in years_list]
    year_win_pcts = [sum(1 for r in yearly[y] if r > 0) / len(yearly[y]) * 100 for y in years_list]

    fig, ax5 = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax5.set_facecolor('#16213e')

    colors = ['#3fb950' if r > 0 else '#f85149' for r in year_returns]
    bars = ax5.bar(years_list, year_returns, color=colors, alpha=0.8, edgecolor='#333')

    for i, (bar, cnt, wp) in enumerate(zip(bars, year_counts, year_win_pcts)):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{cnt}t\n{wp:.0f}%w', ha='center', va='bottom', fontsize=7, color='#aaa')

    ax5.set_title('Year-by-Year Average Hedged Return (5x + MP)', color='white', fontsize=13)
    ax5.set_xlabel('Year', color='white')
    ax5.set_ylabel('Avg Return %', color='white')
    ax5.tick_params(colors='white', rotation=45)
    ax5.axhline(y=0, color='white', linestyle='-', alpha=0.3)
    ax5.grid(True, alpha=0.15, color='white', axis='y')
    ax5.spines['bottom'].set_color('#333')
    ax5.spines['left'].set_color('#333')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    plt.tight_layout()
    chart_path3 = OUTPUT_DIR / 'yearly_returns.png'
    plt.savefig(chart_path3, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f'  Saved: {chart_path3}')

    conn.close()

    # =====================================================================
    # SECTION 4: PINE SCRIPT FOR TRADINGVIEW
    # =====================================================================
    print('\n\n' + '=' * 110)
    print('SECTION 4: PINE SCRIPT FOR TRADINGVIEW')
    print('=' * 110)
    print()

    # Generate Pine Script with the sample trades embedded
    pine_trades = []
    for t in detailed_trades:
        if t['entry_price'] is not None:
            pine_trades.append(t)

    pine_script = generate_pine_script(pine_trades)
    pine_path = OUTPUT_DIR / 'breakout_v3_verification.pine'
    with open(pine_path, 'w') as f:
        f.write(pine_script)
    print(f'  Pine Script saved to: {pine_path}')
    print(f'  Generic indicator - works on any stock')
    print()
    print('  HOW TO USE:')
    print('  1. Open TradingView, search for any stock from the list below (NSE:SYMBOL)')
    print('  2. Pine Editor > New Script > Paste the script > Add to Chart')
    print('  3. Click Settings (gear icon) on the indicator')
    print('  4. Enter the Entry/Exit dates and price levels from the table below')
    print('  5. The indicator draws entry, exit, stop loss, and peak levels automatically')
    print('  6. Switch to DAILY timeframe and navigate to the entry date')
    print()

    # Also print a simplified version for quick manual checking
    print('  QUICK VERIFICATION CHECKLIST:')
    print(f'  {"Symbol":<12} {"Entry Date":>12} {"Entry Rs":>10} {"Exit Rs":>10} {"Stop Rs":>10} {"Peak Rs":>10} {"Return":>8} {"Check"}')
    print(f'  {"-"*90}')
    for t in pine_trades:
        check = 'WIN' if t['base_return'] > 0 else 'LOSS'
        print(f'  {t["symbol"]:<12} {t["entry_date"]:>12} {t["entry_price"]:>10,.1f} {t["exit_price"]:>10,.1f} {t["stop_price"]:>10,.1f} {t["peak_price"]:>10,.1f} {t["base_return"]:>+7.1f}% [{check}]')

    print(f'\n  All output saved to: {OUTPUT_DIR}/')


def generate_pine_script(trades):
    """Generate a simple Pine Script v5 - arrows only, no boxes.

    3 arrow marks + 1 stop loss line:
    - Blue arrow down = consolidation start (lookback bars before entry)
    - Green arrow up = breakout entry
    - Red dashed line = stop loss (entry to exit)
    - Green/Red arrow down = exit (win/loss)
    """
    lines = []
    lines.append('// @version=5')
    lines.append('// Breakout V3 Verifier - Simple Arrows')
    lines.append('indicator("BV3", overlay=true, max_labels_count=10, max_lines_count=10)')
    lines.append('')
    lines.append('// ===== INPUTS =====')
    lines.append('i_strategy   = input.string("T1B", "Strategy", options=["ALPHA","T1B","MOMVOL","CALMAR","BB_MOM"])')
    lines.append('i_entry_year = input.int(2024, "Entry Year", minval=2000, maxval=2030)')
    lines.append('i_entry_mon  = input.int(3, "Entry Month", minval=1, maxval=12)')
    lines.append('i_entry_day  = input.int(26, "Entry Day", minval=1, maxval=31)')
    lines.append('i_exit_year  = input.int(2024, "Exit Year", minval=2000, maxval=2030)')
    lines.append('i_exit_mon   = input.int(9, "Exit Month", minval=1, maxval=12)')
    lines.append('i_exit_day   = input.int(20, "Exit Day", minval=1, maxval=31)')
    lines.append('i_entry_price = input.float(3492.1, "Entry Price")')
    lines.append('i_stop_price  = input.float(2985.7, "Stop Loss")')
    lines.append('i_exit_price  = input.float(4867.9, "Exit Price")')
    lines.append('i_lookback    = input.int(60, "Consolidation Start (trading days before entry)", minval=5, maxval=200)')
    lines.append('')
    lines.append('// ===== COMPUTED =====')
    lines.append('entryTime  = timestamp(i_entry_year, i_entry_mon, i_entry_day)')
    lines.append('exitTime   = timestamp(i_exit_year, i_exit_mon, i_exit_day)')
    lines.append('// Convert trading days to calendar days (x 7/5) for time-based detection')
    lines.append('consolTime = entryTime - i_lookback * 86400000 * 7 / 5')
    lines.append('baseRet    = i_entry_price > 0 ? (i_exit_price - i_entry_price) / i_entry_price * 100 : 0')
    lines.append('hedgedRet  = baseRet * 5 - 3.4 - 2.3')
    lines.append('isWin      = i_exit_price >= i_entry_price')
    lines.append('')
    lines.append('// ===== BAR DETECTION =====')
    lines.append('isConsolBar = time >= consolTime and time[1] < consolTime')
    lines.append('isEntryBar  = time >= entryTime and time[1] < entryTime')
    lines.append('isExitBar   = time >= exitTime and time[1] < exitTime')
    lines.append('')
    lines.append('// 1. CONSOLIDATION START (blue arrow down above candle)')
    lines.append('if isConsolBar')
    lines.append('    label.new(bar_index, high, "CONSOL\\nSTART", style=label.style_label_down, color=color.blue, textcolor=color.white, size=size.small)')
    lines.append('')
    lines.append('// 2. ENTRY (green arrow up below candle)')
    lines.append('if isEntryBar')
    lines.append('    label.new(bar_index, low, i_strategy + " ENTRY\\nRs." + str.tostring(close, "#.0"), style=label.style_label_up, color=color.green, textcolor=color.white, size=size.normal)')
    lines.append('')
    lines.append('// 3. STOP LOSS LINE (red dashed, entry to exit)')
    lines.append('var line slLine = na')
    lines.append('if isEntryBar and i_stop_price > 0')
    lines.append('    slLine := line.new(bar_index, i_stop_price, bar_index + 1, i_stop_price, color=color.red, style=line.style_dashed, width=1, extend=extend.right)')
    lines.append('if isExitBar and not na(slLine)')
    lines.append('    line.set_extend(slLine, extend.none)')
    lines.append('    line.set_x2(slLine, bar_index)')
    lines.append('')
    lines.append('// 4. EXIT (arrow down above candle, green=win red=loss)')
    lines.append('if isExitBar')
    lines.append('    exitCol = isWin ? color.green : color.red')
    lines.append('    exitTxt = (isWin ? "WIN " : "LOSS ") + str.tostring(baseRet, "+#.1") + "%\\nRs." + str.tostring(close, "#.0")')
    lines.append('    label.new(bar_index, high, exitTxt, style=label.style_label_down, color=exitCol, textcolor=color.white, size=size.normal)')
    lines.append('')
    lines.append('// ===== INFO TABLE =====')
    lines.append('var table t = table.new(position.top_right, 2, 5, bgcolor=color.new(color.black, 70), border_width=1)')
    lines.append('if barstate.islast and i_entry_price > 0')
    lines.append('    table.cell(t, 0, 0, "Strategy",    text_color=color.white,  text_size=size.small)')
    lines.append('    table.cell(t, 1, 0, i_strategy,    text_color=color.yellow, text_size=size.small)')
    lines.append('    table.cell(t, 0, 1, "Entry",       text_color=color.white,  text_size=size.small)')
    lines.append('    table.cell(t, 1, 1, "Rs." + str.tostring(i_entry_price, "#.0"), text_color=color.green, text_size=size.small)')
    lines.append('    table.cell(t, 0, 2, "Stop",        text_color=color.white,  text_size=size.small)')
    lines.append('    table.cell(t, 1, 2, "Rs." + str.tostring(i_stop_price, "#.0"), text_color=color.red, text_size=size.small)')
    lines.append('    table.cell(t, 0, 3, "Base",        text_color=color.white,  text_size=size.small)')
    lines.append('    table.cell(t, 1, 3, str.tostring(baseRet, "+#.1") + "%", text_color=(isWin ? color.green : color.red), text_size=size.small)')
    lines.append('    table.cell(t, 0, 4, "5x Hedged",   text_color=color.white,  text_size=size.small)')
    lines.append('    table.cell(t, 1, 4, str.tostring(hedgedRet, "+#.1") + "%", text_color=(hedgedRet > 0 ? color.green : color.red), text_size=size.small)')

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
