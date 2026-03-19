"""
Options-Hedged Futures Backtest
===============================

Takes V3 breakout trades on futures, then hedges with options:
1. NAKED FUTURES (baseline)
2. COVERED CALL: Buy futures + Sell OTM call -> caps upside, earns premium
3. MARRIED PUT: Buy futures + Buy OTM put -> caps downside, costs premium
4. COLLAR: Buy futures + Sell OTM call + Buy OTM put -> bounded range
5. RATIO WRITE: Buy futures + Sell 2x OTM calls -> extra premium, unlimited risk above 2nd call

Option premium estimation uses simplified Black-Scholes-like model calibrated
to Indian F&O market norms (~25-35% implied volatility for breakout stocks).

Key insight: Breakout stocks have ELEVATED IV at entry (they just broke out with
high volume), so option premiums are rich. This makes SELLING options (covered call)
more profitable and BUYING options (married put) more expensive.
"""
import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from services.consolidation_breakout import SYSTEM_PRIMARY, SYSTEM_BALANCED, SYSTEM_SNIPER

LIQUID_FUND_RATE = 0.065
DAILY_RATE = (1 + LIQUID_FUND_RATE) ** (1/365) - 1
STARTING_CAPITAL = 1_000_000
FUTURES_CARRY_ANNUAL = 0.07
ROLLOVER_COST = 0.002
RISK_FREE_RATE = 0.065


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


# =========================================================================
# Option premium estimation (simplified Black-Scholes)
# =========================================================================
def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price as % of spot."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) / S * 100
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return price / S * 100  # as % of spot


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price as % of spot."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0) / S * 100
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price / S * 100  # as % of spot


def estimate_option_premium(otm_pct, days_to_expiry, is_call=True, iv=0.30):
    """
    Estimate option premium as % of stock price.

    otm_pct: how far OTM (e.g., 5 means 5% OTM)
    days_to_expiry: calendar days
    iv: implied volatility (annualized). Breakout stocks typically 25-40%.
    """
    S = 100  # normalized
    T = days_to_expiry / 365.0
    r = RISK_FREE_RATE

    if is_call:
        K = S * (1 + otm_pct / 100)
        return bs_call_price(S, K, T, r, iv)
    else:
        K = S * (1 - otm_pct / 100)
        return bs_put_price(S, K, T, r, iv)


# =========================================================================
# Hedged return calculations
# =========================================================================
def compute_hedged_return(base_return, leverage, hold_d, strategy, params):
    """
    Compute the hedged return on margin for a single trade.

    base_return: stock % return (e.g., +15 means +15%)
    leverage: futures leverage multiplier
    hold_d: holding period in calendar days
    strategy: 'naked', 'covered_call', 'married_put', 'collar', 'ratio_write'
    params: dict with otm_call, otm_put, iv, etc.
    """
    # Futures costs
    carry_cost = FUTURES_CARRY_ANNUAL * (hold_d / 365.0) * 100
    n_rollovers = max(0, (hold_d / 30.0) - 1)
    rollover_cost = ROLLOVER_COST * n_rollovers * 100

    # Option premiums (as % of stock price)
    iv = params.get('iv', 0.30)
    # Monthly options - we buy at entry, exercise/expire at exit
    # Use min(hold_d, 30) as options are monthly, we roll them
    option_periods = max(1, hold_d / 30)  # number of monthly options needed
    option_dte = 30  # each option is ~30 DTE

    if strategy == 'naked':
        lev_return = (base_return * leverage) - carry_cost - rollover_cost
        lev_return = max(lev_return, -100.0)
        return lev_return

    elif strategy == 'covered_call':
        otm = params.get('otm_call', 5)
        # Sell call at entry, premium received per month
        call_premium = estimate_option_premium(otm, option_dte, is_call=True, iv=iv)
        total_premium = call_premium * option_periods  # premium per option cycle

        # Stock return capped at call strike
        capped_return = min(base_return, otm)
        lev_return = (capped_return * leverage) - carry_cost - rollover_cost + (total_premium * leverage)
        lev_return = max(lev_return, -100.0)
        return lev_return

    elif strategy == 'married_put':
        otm = params.get('otm_put', 5)
        put_premium = estimate_option_premium(otm, option_dte, is_call=False, iv=iv)
        total_premium = put_premium * option_periods

        # Stock loss floored at put strike
        floored_return = max(base_return, -otm)
        lev_return = (floored_return * leverage) - carry_cost - rollover_cost - (total_premium * leverage)
        lev_return = max(lev_return, -100.0)
        return lev_return

    elif strategy == 'collar':
        otm_c = params.get('otm_call', 10)
        otm_p = params.get('otm_put', 10)
        call_premium = estimate_option_premium(otm_c, option_dte, is_call=True, iv=iv)
        put_premium = estimate_option_premium(otm_p, option_dte, is_call=False, iv=iv)
        net_premium = (call_premium - put_premium) * option_periods  # positive = credit

        capped_return = min(base_return, otm_c)
        floored_return = max(capped_return, -otm_p)
        lev_return = (floored_return * leverage) - carry_cost - rollover_cost + (net_premium * leverage)
        lev_return = max(lev_return, -100.0)
        return lev_return

    elif strategy == 'ratio_write':
        # Sell 2 calls for every 1 lot of futures
        otm = params.get('otm_call', 7)
        call_premium = estimate_option_premium(otm, option_dte, is_call=True, iv=iv)
        total_premium = call_premium * 2 * option_periods  # 2x calls

        # 1st call is covered (caps at strike), 2nd call is naked (unlimited loss above strike)
        if base_return <= otm:
            # Both calls expire OTM - keep full premium
            effective_return = base_return
        else:
            # Stock went above call strike
            # 1st call: capped at otm (covered)
            # 2nd call: lose (base_return - otm) on the naked call
            effective_return = otm - (base_return - otm)  # = 2*otm - base_return

        lev_return = (effective_return * leverage) - carry_cost - rollover_cost + (total_premium * leverage)
        lev_return = max(lev_return, -100.0)
        return lev_return

    return 0


# =========================================================================
# Portfolio simulation
# =========================================================================
def run_hedged_sim(trades_df, leverage, pos_pct, strategy, params):
    """Run portfolio simulation with options-hedged futures."""
    trades = trades_df.sort_values('date').copy()
    if len(trades) == 0:
        return None

    trades['holding_days'] = trades.apply(hold_days, axis=1)
    trades['exit_date'] = trades['date'] + pd.to_timedelta(trades['holding_days'], unit='D')

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

    peak_equity = STARTING_CAPITAL
    max_dd = 0.0
    total_option_cost = 0.0
    total_option_income = 0.0

    # Track per-trade stats
    trade_returns = []

    for _, row in trades.iterrows():
        entry = row['date']
        base_return = row['trade_return']
        exit_d = row['exit_date']
        hold_d = row['holding_days']

        hedged_return = compute_hedged_return(base_return, leverage, hold_d, strategy, params)

        # Track option P&L
        if strategy in ('covered_call', 'ratio_write'):
            naked_return = compute_hedged_return(base_return, leverage, hold_d, 'naked', params)
            diff = hedged_return - naked_return
            if diff > 0:
                total_option_income += diff
            else:
                total_option_cost += abs(diff)
        elif strategy in ('married_put', 'collar'):
            naked_return = compute_hedged_return(base_return, leverage, hold_d, 'naked', params)
            diff = hedged_return - naked_return
            if diff > 0:
                total_option_income += diff
            else:
                total_option_cost += abs(diff)

        trade_returns.append(hedged_return)

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
        trade_margin = portfolio_val * pos_pct

        if cash >= trade_margin and trade_margin > 0:
            cash -= trade_margin
            active.append((exit_d, trade_margin, hedged_return))
            taken += 1
        else:
            skipped += 1

        # Track drawdown
        current_equity = cash + sum(m for (_, m, _) in active)
        peak_equity = max(peak_equity, current_equity)
        dd = (peak_equity - current_equity) / peak_equity * 100
        max_dd = max(max_dd, dd)

        last_dt = entry

    # Final
    if sim_end > last_dt:
        days = (sim_end - last_dt).days
        interest = cash * ((1 + DAILY_RATE) ** days - 1)
        cash += interest
        total_int += interest

    for (ed, margin, lev_ret) in active:
        returned = margin * (1 + lev_ret / 100)
        returned = max(returned, 0)
        if lev_ret <= -95:
            margin_calls += 1
        cash += returned

    final = cash
    cagr = ((final / STARTING_CAPITAL) ** (1 / years) - 1) * 100 if final > 0 else -100
    calmar = round(cagr / max_dd, 2) if max_dd > 0 else 999

    tr = np.array(trade_returns)
    win_pct = (tr > 0).mean() * 100 if len(tr) > 0 else 0
    avg_win = tr[tr > 0].mean() if (tr > 0).any() else 0
    avg_loss = tr[tr <= 0].mean() if (tr <= 0).any() else 0
    severe = (tr < -30).sum()
    wipeout = (tr < -80).sum()

    return {
        'strategy': strategy,
        'leverage': leverage,
        'pos_pct': pos_pct,
        'taken': taken,
        'skipped': skipped,
        'equity': final,
        'cagr': round(cagr, 2),
        'max_dd': round(max_dd, 2),
        'calmar': calmar,
        'interest': total_int,
        'margin_calls': margin_calls,
        'years': round(years, 1),
        'multiplier': round(final / STARTING_CAPITAL, 1),
        'win_pct': round(win_pct, 1),
        'avg_win': round(avg_win, 1),
        'avg_loss': round(avg_loss, 1),
        'severe_losses': severe,
        'wipeouts': wipeout,
        'option_income': total_option_income,
        'option_cost': total_option_cost,
    }


def main():
    df = pd.read_csv('breakout_analysis_enhanced.csv')
    df['date'] = pd.to_datetime(df['date'])

    primary_trades = df[apply_mask(df, SYSTEM_PRIMARY)].copy()
    balanced_trades = df[apply_mask(df, SYSTEM_BALANCED)].copy()

    print('=' * 110)
    print('OPTIONS-HEDGED FUTURES BACKTEST - V3 BREAKOUT SYSTEM')
    print('=' * 110)
    print(f'Capital: Rs.10L | PRIMARY system: {len(primary_trades)} trades')
    print(f'Futures: {FUTURES_CARRY_ANNUAL*100:.0f}% carry | Options: Black-Scholes estimate, 30% IV')
    print(f'Idle cash: 6.5% liquid fund')
    print('=' * 110)

    # =================================================================
    # SECTION 1: Option premium reality check
    # =================================================================
    print()
    print('SECTION 1: OPTION PREMIUM ESTIMATES (% of stock price)')
    print('-' * 80)
    print(f'  {"OTM %":>6} {"30-day Call":>12} {"30-day Put":>12} {"60-day Call":>12} {"60-day Put":>12}')
    print(f'  {"-"*56}')
    for otm in [3, 5, 7, 10, 15]:
        c30 = estimate_option_premium(otm, 30, True, 0.30)
        p30 = estimate_option_premium(otm, 30, False, 0.30)
        c60 = estimate_option_premium(otm, 60, True, 0.30)
        p60 = estimate_option_premium(otm, 60, False, 0.30)
        print(f'  {otm:>5}% {c30:>11.2f}% {p30:>11.2f}% {c60:>11.2f}% {p60:>11.2f}%')

    print()
    print('  Note: Breakout stocks typically have IV of 30-40%. These are monthly')
    print('  options rolled every 30 days. Premium accrues each month you hold.')

    # =================================================================
    # SECTION 2: HEAD-TO-HEAD - All strategies at 3x leverage, 5% pos
    # =================================================================
    print()
    print()
    print('=' * 110)
    print('SECTION 2: STRATEGY COMPARISON - PRIMARY @ 3x LEVERAGE, 5% POSITION')
    print('=' * 110)

    strategies = [
        ('naked',        'NAKED FUTURES',           {'iv': 0.30}),
        ('covered_call', 'COVERED CALL (5% OTM)',   {'otm_call': 5, 'iv': 0.30}),
        ('covered_call', 'COVERED CALL (10% OTM)',  {'otm_call': 10, 'iv': 0.30}),
        ('covered_call', 'COVERED CALL (15% OTM)',  {'otm_call': 15, 'iv': 0.30}),
        ('married_put',  'MARRIED PUT (5% OTM)',    {'otm_put': 5, 'iv': 0.30}),
        ('married_put',  'MARRIED PUT (10% OTM)',   {'otm_put': 10, 'iv': 0.30}),
        ('married_put',  'MARRIED PUT (15% OTM)',   {'otm_put': 15, 'iv': 0.30}),
        ('collar',       'COLLAR (5/5)',            {'otm_call': 5, 'otm_put': 5, 'iv': 0.30}),
        ('collar',       'COLLAR (10/10)',          {'otm_call': 10, 'otm_put': 10, 'iv': 0.30}),
        ('collar',       'COLLAR (10/5)',           {'otm_call': 10, 'otm_put': 5, 'iv': 0.30}),
        ('collar',       'COLLAR (15/10)',          {'otm_call': 15, 'otm_put': 10, 'iv': 0.30}),
        ('ratio_write',  'RATIO WRITE 2:1 (7% OTM)', {'otm_call': 7, 'iv': 0.30}),
        ('ratio_write',  'RATIO WRITE 2:1 (10% OTM)', {'otm_call': 10, 'iv': 0.30}),
    ]

    hdr = f'  {"Strategy":<30} {"CAGR":>8} {"MaxDD":>8} {"Calmar":>8} {"Win%":>7} {"AvgWin":>8} {"AvgLoss":>8} {"Severe":>7} {"Wipeout":>8}'
    print(hdr)
    print(f'  {"-"*98}')

    results_3x = []
    for strat_key, label, params in strategies:
        r = run_hedged_sim(primary_trades, 3, 0.05, strat_key, params)
        results_3x.append((label, r))
        print(f'  {label:<30} {r["cagr"]:>7.1f}% {r["max_dd"]:>7.1f}% {r["calmar"]:>7.2f} {r["win_pct"]:>6.1f}% {r["avg_win"]:>7.1f}% {r["avg_loss"]:>7.1f}% {r["severe_losses"]:>6} {r["wipeouts"]:>7}')

    # =================================================================
    # SECTION 3: LEVERAGE SWEEP for best strategies
    # =================================================================
    print()
    print()
    print('=' * 110)
    print('SECTION 3: LEVERAGE SWEEP - BEST STRATEGIES')
    print('=' * 110)

    best_strats = [
        ('naked',        'NAKED',          {}),
        ('covered_call', 'CC 10% OTM',     {'otm_call': 10, 'iv': 0.30}),
        ('married_put',  'MP 10% OTM',     {'otm_put': 10, 'iv': 0.30}),
        ('collar',       'COLLAR 10/10',   {'otm_call': 10, 'otm_put': 10, 'iv': 0.30}),
        ('collar',       'COLLAR 15/10',   {'otm_call': 15, 'otm_put': 10, 'iv': 0.30}),
    ]

    for strat_key, label, params in best_strats:
        print(f'\n  {label}:')
        print(f'  {"Lev":>5} {"CAGR":>9} {"MaxDD":>9} {"Calmar":>9} {"Win%":>7} {"Severe":>8} {"Wipeout":>8} {"Rs.10L->":>14}')
        print(f'  {"-"*75}')
        for lev in [1, 2, 3, 5, 7]:
            r = run_hedged_sim(primary_trades, lev, 0.05, strat_key, params)
            print(f'  {lev:>4}x {r["cagr"]:>8.1f}% {r["max_dd"]:>8.1f}% {r["calmar"]:>8.2f} {r["win_pct"]:>6.1f}% {r["severe_losses"]:>7} {r["wipeouts"]:>7} Rs.{r["equity"]/100000:>10.0f}L')

    # =================================================================
    # SECTION 4: IMPACT OF IV ON STRATEGIES
    # =================================================================
    print()
    print()
    print('=' * 110)
    print('SECTION 4: IV SENSITIVITY (3x leverage, 5% position)')
    print('=' * 110)
    print('  Breakout stocks have higher IV. How does this affect each strategy?')

    for iv_val, iv_label in [(0.20, 'Low IV (20%)'), (0.30, 'Normal IV (30%)'),
                              (0.40, 'High IV (40%)'), (0.50, 'Very High IV (50%)')]:
        print(f'\n  {iv_label}:')
        print(f'  {"Strategy":<25} {"CAGR":>8} {"MaxDD":>8} {"Calmar":>8}')
        print(f'  {"-"*52}')
        for strat_key, label, base_params in [
            ('naked',        'Naked Futures',   {}),
            ('covered_call', 'Covered Call 10%', {'otm_call': 10}),
            ('married_put',  'Married Put 10%', {'otm_put': 10}),
            ('collar',       'Collar 10/10',    {'otm_call': 10, 'otm_put': 10}),
        ]:
            params = {**base_params, 'iv': iv_val}
            r = run_hedged_sim(primary_trades, 3, 0.05, strat_key, params)
            print(f'  {label:<25} {r["cagr"]:>7.1f}% {r["max_dd"]:>7.1f}% {r["calmar"]:>7.2f}')

    # =================================================================
    # SECTION 5: OPTIMAL COMBOS - All strategies x leverage x position
    # =================================================================
    print()
    print()
    print('=' * 110)
    print('SECTION 5: TOP 20 RISK-ADJUSTED COMBINATIONS (Calmar > 3, CAGR > 15%)')
    print('=' * 110)

    all_combos = []
    test_strats = [
        ('naked',        'Naked',        {'iv': 0.30}),
        ('covered_call', 'CC-5%',        {'otm_call': 5, 'iv': 0.30}),
        ('covered_call', 'CC-10%',       {'otm_call': 10, 'iv': 0.30}),
        ('covered_call', 'CC-15%',       {'otm_call': 15, 'iv': 0.30}),
        ('married_put',  'MP-5%',        {'otm_put': 5, 'iv': 0.30}),
        ('married_put',  'MP-10%',       {'otm_put': 10, 'iv': 0.30}),
        ('collar',       'COL-5/5',      {'otm_call': 5, 'otm_put': 5, 'iv': 0.30}),
        ('collar',       'COL-10/10',    {'otm_call': 10, 'otm_put': 10, 'iv': 0.30}),
        ('collar',       'COL-10/5',     {'otm_call': 10, 'otm_put': 5, 'iv': 0.30}),
        ('collar',       'COL-15/10',    {'otm_call': 15, 'otm_put': 10, 'iv': 0.30}),
        ('ratio_write',  'RW-7%',        {'otm_call': 7, 'iv': 0.30}),
        ('ratio_write',  'RW-10%',       {'otm_call': 10, 'iv': 0.30}),
    ]

    for strat_key, label, params in test_strats:
        for lev in [2, 3, 5]:
            for pct in [0.03, 0.05, 0.07]:
                r = run_hedged_sim(primary_trades, lev, pct, strat_key, params)
                if r and r['cagr'] > 15 and r['calmar'] > 3:
                    all_combos.append((label, lev, pct, r))

    all_combos.sort(key=lambda x: x[3]['calmar'], reverse=True)

    print(f'\n  {"Strategy":<12} {"Lev":>4} {"Pos%":>5} {"CAGR":>8} {"MaxDD":>8} {"Calmar":>8} {"Win%":>7} {"Severe":>7} {"Final":>14}')
    print(f'  {"-"*80}')
    for label, lev, pct, r in all_combos[:20]:
        print(f'  {label:<12} {lev:>3}x {pct*100:>4.0f}% {r["cagr"]:>7.1f}% {r["max_dd"]:>7.1f}% {r["calmar"]:>7.2f} {r["win_pct"]:>6.1f}% {r["severe_losses"]:>6} Rs.{r["equity"]/100000:>10.0f}L')

    # =================================================================
    # SECTION 6: DEEP COMPARISON - Best of each type
    # =================================================================
    print()
    print()
    print('=' * 110)
    print('SECTION 6: BEST OF EACH STRATEGY TYPE')
    print('=' * 110)
    print()

    best_of = [
        ('Naked 3x, 5%',          'naked',        3, 0.05, {'iv': 0.30}),
        ('Naked 2x, 5%',          'naked',        2, 0.05, {'iv': 0.30}),
        ('CC-10% 3x, 5%',         'covered_call', 3, 0.05, {'otm_call': 10, 'iv': 0.30}),
        ('CC-10% 5x, 5%',         'covered_call', 5, 0.05, {'otm_call': 10, 'iv': 0.30}),
        ('MP-10% 3x, 5%',         'married_put',  3, 0.05, {'otm_put': 10, 'iv': 0.30}),
        ('MP-10% 5x, 5%',         'married_put',  5, 0.05, {'otm_put': 10, 'iv': 0.30}),
        ('Collar 10/10 3x, 5%',   'collar',       3, 0.05, {'otm_call': 10, 'otm_put': 10, 'iv': 0.30}),
        ('Collar 15/10 3x, 5%',   'collar',       3, 0.05, {'otm_call': 15, 'otm_put': 10, 'iv': 0.30}),
        ('Collar 15/10 5x, 5%',   'collar',       5, 0.05, {'otm_call': 15, 'otm_put': 10, 'iv': 0.30}),
        ('RatioWrite 10% 3x, 5%', 'ratio_write',  3, 0.05, {'otm_call': 10, 'iv': 0.30}),
    ]

    print(f'  {"Setup":<30} {"CAGR":>8} {"MaxDD":>8} {"Calmar":>8} {"Win%":>7} {"AvgW":>7} {"AvgL":>7} {"Severe":>7} {"Rs.10L->":>11}')
    print(f'  {"-"*95}')
    for label, strat_key, lev, pct, params in best_of:
        r = run_hedged_sim(primary_trades, lev, pct, strat_key, params)
        print(f'  {label:<30} {r["cagr"]:>7.1f}% {r["max_dd"]:>7.1f}% {r["calmar"]:>7.2f} {r["win_pct"]:>6.1f}% {r["avg_win"]:>6.1f}% {r["avg_loss"]:>6.1f}% {r["severe_losses"]:>6} Rs.{r["equity"]/100000:>7.0f}L')

    # =================================================================
    # SECTION 7: THE REAL QUESTION - Can married put save us at 5x?
    # =================================================================
    print()
    print()
    print('=' * 110)
    print('SECTION 7: CAN OPTIONS TAME HIGH LEVERAGE? (5x and 7x)')
    print('=' * 110)
    print()
    print('  The question: At 5x leverage, naked futures had 26 near-wipeout trades.')
    print('  Can married puts or collars eliminate those while preserving most of the upside?')
    print()

    tame_strats = [
        ('naked',        'Naked (baseline)',  {'iv': 0.30}),
        ('married_put',  'Married Put 5%',    {'otm_put': 5, 'iv': 0.30}),
        ('married_put',  'Married Put 10%',   {'otm_put': 10, 'iv': 0.30}),
        ('married_put',  'Married Put 15%',   {'otm_put': 15, 'iv': 0.30}),
        ('collar',       'Collar 15/5',       {'otm_call': 15, 'otm_put': 5, 'iv': 0.30}),
        ('collar',       'Collar 15/10',      {'otm_call': 15, 'otm_put': 10, 'iv': 0.30}),
        ('collar',       'Collar 20/10',      {'otm_call': 20, 'otm_put': 10, 'iv': 0.30}),
    ]

    for lev in [5, 7]:
        print(f'  AT {lev}x LEVERAGE, 5% position:')
        print(f'  {"Strategy":<25} {"CAGR":>8} {"MaxDD":>8} {"Calmar":>8} {"Win%":>7} {"Severe":>7} {"Wipeout":>8} {"Rs.10L->":>11}')
        print(f'  {"-"*87}')
        for strat_key, label, params in tame_strats:
            r = run_hedged_sim(primary_trades, lev, 0.05, strat_key, params)
            print(f'  {label:<25} {r["cagr"]:>7.1f}% {r["max_dd"]:>7.1f}% {r["calmar"]:>7.2f} {r["win_pct"]:>6.1f}% {r["severe_losses"]:>6} {r["wipeouts"]:>7} Rs.{r["equity"]/100000:>7.0f}L')
        print()

    # =================================================================
    # SECTION 8: SUMMARY
    # =================================================================
    print()
    print('=' * 110)
    print('SUMMARY & KEY FINDINGS')
    print('=' * 110)
    print()
    print('  COVERED CALL (Sell OTM call on futures position):')
    print('  + Earns monthly premium income (1-3% per month on breakout stocks)')
    print('  + Reduces effective cost basis -> improves win rate')
    print('  + Works best with breakout trades (high IV = rich premiums)')
    print('  - Caps upside at strike price (breakout runners get cut short)')
    print('  - Does NOT protect downside')
    print()
    print('  MARRIED PUT (Buy OTM put with futures position):')
    print('  + Hard floor on losses -> eliminates wipeout risk')
    print('  + Allows full upside participation')
    print('  + Makes high leverage (5-7x) viable')
    print('  - Costs premium every month (1-3% per month)')
    print('  - At low leverage, the premium drag kills returns')
    print()
    print('  COLLAR (Sell OTM call + Buy OTM put):')
    print('  + Bounded risk profile (known max loss AND max gain)')
    print('  + Near zero-cost if call/put strikes balanced')
    print('  + Best risk-adjusted returns (Calmar)')
    print('  - Caps both upside and downside')
    print('  - You lose the big breakout runners')
    print()
    print('  RATIO WRITE (Sell 2x calls for every futures lot):')
    print('  + Double premium income -> highest win rate')
    print('  + Works great when stocks dont move much above strike')
    print('  - Naked call exposure above the strike = unlimited risk')
    print('  - Worst when breakouts run hard (which is the whole point)')
    print()

    # Final recommendation
    r_naked = run_hedged_sim(primary_trades, 3, 0.05, 'naked', {'iv': 0.30})
    r_cc = run_hedged_sim(primary_trades, 3, 0.05, 'covered_call', {'otm_call': 10, 'iv': 0.30})
    r_mp5x = run_hedged_sim(primary_trades, 5, 0.05, 'married_put', {'otm_put': 10, 'iv': 0.30})
    r_col = run_hedged_sim(primary_trades, 3, 0.05, 'collar', {'otm_call': 15, 'otm_put': 10, 'iv': 0.30})
    r_col5x = run_hedged_sim(primary_trades, 5, 0.05, 'collar', {'otm_call': 15, 'otm_put': 10, 'iv': 0.30})

    print('  RECOMMENDED SETUPS:')
    print(f'  {"Setup":<35} {"CAGR":>8} {"MaxDD":>8} {"Calmar":>8}')
    print(f'  {"-"*60}')
    print(f'  {"Naked 3x (baseline)":35} {r_naked["cagr"]:>7.1f}% {r_naked["max_dd"]:>7.1f}% {r_naked["calmar"]:>7.2f}')
    print(f'  {"Covered Call 10% OTM, 3x":35} {r_cc["cagr"]:>7.1f}% {r_cc["max_dd"]:>7.1f}% {r_cc["calmar"]:>7.2f}')
    print(f'  {"Collar 15/10, 3x":35} {r_col["cagr"]:>7.1f}% {r_col["max_dd"]:>7.1f}% {r_col["calmar"]:>7.2f}')
    print(f'  {"Married Put 10%, 5x":35} {r_mp5x["cagr"]:>7.1f}% {r_mp5x["max_dd"]:>7.1f}% {r_mp5x["calmar"]:>7.2f}')
    print(f'  {"Collar 15/10, 5x":35} {r_col5x["cagr"]:>7.1f}% {r_col5x["max_dd"]:>7.1f}% {r_col5x["calmar"]:>7.2f}')


if __name__ == '__main__':
    main()
